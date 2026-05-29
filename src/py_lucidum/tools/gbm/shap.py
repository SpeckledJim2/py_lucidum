from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import duckdb

from py_lucidum.core import Dataset, is_numeric_kind, json_number, parse_positive_float, quote_ident, sql_literal

from .store import GbmModelStore


FLAME_PERCENTILES = (0, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 100)
BOX_PERCENTILES = (0, 5, 25, 50, 75, 95, 100)
MAX_FACTOR_LEVELS = 500
MAX_LINE_SERIES = 80
MAX_HEATMAP_CELLS = 40000
STACKED_OTHER_LABEL = "Other"


def shap_config(dataset: Dataset, store: GbmModelStore, model_id: str) -> dict[str, Any]:
    manifest = store.manifest(model_id)
    features = model_features(dataset, store, model_id)
    shap_path = store.artifact_path(model_id, "shap_long")
    warnings: list[str] = []
    if not shap_path.exists():
        return {
            "model_id": model_id,
            "label": manifest.get("label") or model_id,
            "has_shap": False,
            "row_count": 0,
            "features": [],
            "default_feature_1": "",
            "warnings": ["This GBM was trained without saved SHAP rows."],
        }
    with dataset.lock:
        shap_columns = parquet_columns(dataset.con, shap_path)
        row_count = int(dataset.con.execute(f"SELECT COUNT(*) FROM read_parquet({sql_literal(str(shap_path))})").fetchone()[0] or 0)
    available = []
    for feature in features:
        if feature["name"] in shap_columns:
            available.append(feature)
        else:
            warnings.append(f"SHAP values for {feature['name']} are missing from the saved artifact.")
    return {
        "model_id": model_id,
        "label": manifest.get("label") or model_id,
        "has_shap": bool(row_count and available),
        "row_count": row_count,
        "features": available,
        "default_feature_1": available[0]["name"] if available else "",
        "warnings": warnings,
    }


def shap_plot(dataset: Dataset, store: GbmModelStore, model_id: str, request: dict[str, Any]) -> dict[str, Any]:
    config = shap_config(dataset, store, model_id)
    if not config["has_shap"]:
        raise ValueError("This GBM was trained without saved SHAP rows")
    features = {feature["name"]: feature for feature in config["features"]}
    feature_1 = normalise_feature_name(request.get("feature_1"))
    feature_2 = normalise_feature_name(request.get("feature_2"), allow_none=True)
    if not feature_1 or feature_1 not in features:
        raise ValueError("Choose a valid SHAP Feature 1")
    if feature_2 and feature_2 not in features:
        raise ValueError("Choose a valid SHAP Feature 2")

    factor_1 = bool(request.get("factor_1"))
    factor_2 = bool(request.get("factor_2"))
    banding_1 = normalise_banding(request.get("banding_1"), features[feature_1].get("band_suggestion"))
    banding_2 = normalise_banding(request.get("banding_2"), features[feature_2].get("band_suggestion") if feature_2 else None)
    tail_fraction = normalise_tail_fraction(request.get("tail_percent"))

    with dataset.lock:
        if not feature_2:
            if feature_is_continuous(features[feature_1], factor_1):
                payload = flame_plot(dataset, store, model_id, feature_1, banding_1, tail_fraction, features[feature_1])
            else:
                payload = box_plot(dataset, store, model_id, feature_1, banding_1, tail_fraction, factor_1, features[feature_1])
        elif feature_is_continuous(features[feature_1], factor_1) and feature_is_continuous(features[feature_2], factor_2):
            payload = surface_plot(dataset, store, model_id, feature_1, feature_2, banding_1, banding_2, tail_fraction, features)
        elif not feature_is_continuous(features[feature_1], factor_1) and not feature_is_continuous(features[feature_2], factor_2):
            payload = heatmap_plot(dataset, store, model_id, feature_1, feature_2, banding_1, banding_2, tail_fraction, factor_1, factor_2, features)
        else:
            payload = lines_plot(dataset, store, model_id, feature_1, feature_2, banding_1, banding_2, tail_fraction, factor_1, factor_2, features)
    payload.update(
        {
            "model_id": model_id,
            "feature_1": feature_payload(features[feature_1], banding_1, factor_1),
            "feature_2": feature_payload(features[feature_2], banding_2, factor_2) if feature_2 else None,
            "tail_percent": tail_fraction * 100,
        }
    )
    return payload


def stacked_shap_plot(dataset: Dataset, store: GbmModelStore, model_id: str, request: dict[str, Any]) -> dict[str, Any]:
    config = shap_config(dataset, store, model_id)
    if not config["has_shap"]:
        raise ValueError("This GBM was trained without saved SHAP rows")
    features = {feature["name"]: feature for feature in config["features"]}
    model_feature = normalise_feature_name(request.get("model_feature"))
    if not model_feature or model_feature not in features:
        raise ValueError("Choose a valid Stacked SHAP model feature")

    all_feature_names = [feature["name"] for feature in config["features"]]
    banding = normalise_banding(request.get("banding"), features[model_feature].get("band_suggestion"))
    tail_fraction = normalise_tail_fraction(request.get("tail_percent"))
    x_sort = normalise_stacked_x_sort(request.get("x_sort"))
    feature_limit = normalise_num_features(request.get("num_features"))

    with dataset.lock:
        cte = stacked_joined_cte(dataset, store, model_id, model_feature, all_feature_names)
        if feature_is_numeric(features[model_feature]):
            payload = numeric_stacked_shap_plot(
                dataset,
                cte,
                model_feature,
                banding,
                tail_fraction,
                x_sort,
                all_feature_names,
                feature_limit,
            )
        else:
            payload = categorical_stacked_shap_plot(
                dataset,
                cte,
                model_feature,
                x_sort,
                all_feature_names,
                feature_limit,
            )
    payload.update(
        {
            "plot_type": "stacked_shap",
            "model_id": model_id,
            "title": f"SHAP Values by {model_feature}",
            "model_feature": feature_payload(features[model_feature], banding, False),
            "x_sort": x_sort,
            "tail_percent": tail_fraction * 100,
            "num_features": "all" if feature_limit is None else feature_limit,
            "banding": banding,
        }
    )
    return payload


def numeric_stacked_shap_plot(
    dataset: Dataset,
    cte: str,
    model_feature: str,
    banding: float,
    tail_fraction: float,
    x_sort: str,
    feature_names: list[str],
    feature_limit: int | None,
) -> dict[str, Any]:
    bounds = numeric_bounds(dataset.con, cte, "raw_model", tail_fraction)
    banded = band_expr("TRY_CAST(raw_model AS DOUBLE)", banding, bounds)
    select_sql = stacked_feature_select_sql(feature_names)
    sql = f"""
{cte}
SELECT
    {banded} AS x_sort,
    COUNT(*) AS row_count,
    {select_sql},
    {stacked_total_select_sql(feature_names)}
FROM joined
WHERE TRY_CAST(raw_model AS DOUBLE) IS NOT NULL
GROUP BY x_sort
"""
    rows = clean_numeric_rows(query_dicts(dataset.con, sql), ["x_sort", "total_shap", *feature_names])
    for row in rows:
        row["x"] = json_number(row.get("x_sort"))
    sorted_rows = sort_stacked_rows(rows, x_sort, numeric=True)
    return stacked_payload(
        dataset,
        cte,
        sorted_rows,
        feature_names,
        feature_limit,
        warnings=numeric_missing_warnings(dataset.con, cte, [(model_feature, "raw_model")]),
    )


def categorical_stacked_shap_plot(
    dataset: Dataset,
    cte: str,
    model_feature: str,
    x_sort: str,
    feature_names: list[str],
    feature_limit: int | None,
) -> dict[str, Any]:
    select_sql = stacked_feature_select_sql(feature_names)
    sql = f"""
{cte}
SELECT
    COALESCE(CAST(raw_model AS VARCHAR), '(missing)') AS x,
    COALESCE(CAST(raw_model AS VARCHAR), '(missing)') AS x_sort,
    COUNT(*) AS row_count,
    {select_sql},
    {stacked_total_select_sql(feature_names)}
FROM joined
GROUP BY x, x_sort
"""
    rows = clean_numeric_rows(query_dicts(dataset.con, sql), ["total_shap", *feature_names])
    sorted_rows = sort_stacked_rows(rows, x_sort, numeric=False)
    return stacked_payload(dataset, cte, sorted_rows, feature_names, feature_limit, warnings=[])


def stacked_payload(
    dataset: Dataset,
    cte: str,
    rows: list[dict[str, Any]],
    feature_names: list[str],
    feature_limit: int | None,
    *,
    warnings: list[str],
) -> dict[str, Any]:
    display_features = stacked_display_features(rows, feature_names, feature_limit)
    hidden_features = [name for name in feature_names if name not in display_features]
    if hidden_features:
        display_features = [*display_features, STACKED_OTHER_LABEL]
    payload_rows = []
    for row in rows:
        contributions = {name: json_number(row.get(name)) or 0.0 for name in display_features if name != STACKED_OTHER_LABEL}
        if hidden_features:
            contributions[STACKED_OTHER_LABEL] = sum(json_number(row.get(name)) or 0.0 for name in hidden_features)
        payload_rows.append(
            {
                "x": row.get("x"),
                "x_sort": json_number(row.get("x_sort")) if json_number(row.get("x_sort")) is not None else row.get("x_sort"),
                "row_count": int(row.get("row_count") or 0),
                "total_shap": json_number(row.get("total_shap")) or 0.0,
                "contributions": contributions,
            }
        )
    return {
        "row_count": sum(row["row_count"] for row in payload_rows),
        "total_shap_rows": joined_count(dataset.con, cte),
        "display_features": display_features,
        "all_feature_count": len(feature_names),
        "rows": payload_rows,
        "y_domain": stacked_y_domain(payload_rows, display_features),
        "warnings": warnings,
    }


def stacked_display_features(rows: list[dict[str, Any]], feature_names: list[str], feature_limit: int | None) -> list[str]:
    if feature_limit is None or feature_limit >= len(feature_names):
        return list(feature_names)
    total_weight = sum(int(row.get("row_count") or 0) for row in rows)
    if total_weight <= 0:
        total_weight = len(rows) or 1
    scores = []
    for feature_name in feature_names:
        score = sum(int(row.get("row_count") or 0) * abs(json_number(row.get(feature_name)) or 0.0) for row in rows) / total_weight
        scores.append((feature_name, score))
    scores.sort(key=lambda item: (-item[1], item[0].lower()))
    return [name for name, _score in scores[: max(0, feature_limit)]]


def stacked_y_domain(rows: list[dict[str, Any]], display_features: list[str]) -> list[float | int] | None:
    values: list[float] = []
    for row in rows:
        positives = 0.0
        negatives = 0.0
        contributions = row.get("contributions") if isinstance(row.get("contributions"), dict) else {}
        for feature_name in display_features:
            value = json_number(contributions.get(feature_name))
            if value is None:
                continue
            if value >= 0:
                positives += value
            else:
                negatives += value
        values.extend([positives, negatives])
        total = json_number(row.get("total_shap"))
        if total is not None:
            values.append(total)
    return numeric_domain_from_values(values)


def sort_stacked_rows(rows: list[dict[str, Any]], x_sort: str, *, numeric: bool) -> list[dict[str, Any]]:
    def natural_key(row: dict[str, Any]) -> tuple[int, float | str, str]:
        if numeric:
            number = json_number(row.get("x_sort"))
            return (0 if number is not None else 1, number if number is not None else 0.0, str(row.get("x") or ""))
        label = str(row.get("x") or "")
        return (1 if label == "(missing)" else 0, str(row.get("x_sort") or label).lower(), label.lower())

    if x_sort == "descending":
        return sorted(rows, key=lambda row: (-(json_number(row.get("total_shap")) or 0.0), natural_key(row)))
    return sorted(rows, key=natural_key)


def stacked_feature_select_sql(feature_names: list[str]) -> str:
    return ",\n    ".join(f"AVG({quote_ident(name)}) AS {quote_ident(name)}" for name in feature_names)


def stacked_total_select_sql(feature_names: list[str]) -> str:
    expression = " + ".join(quote_ident(name) for name in feature_names) or "0.0"
    return f"AVG({expression}) AS total_shap"


def stacked_joined_cte(dataset: Dataset, store: GbmModelStore, model_id: str, model_feature: str, feature_names: list[str]) -> str:
    shap_path = store.artifact_path(model_id, "shap_long")
    shap_columns = [
        "__lucidum_row_id",
        *[
            f"COALESCE(TRY_CAST({quote_ident(feature_name)} AS DOUBLE), 0.0) AS {quote_ident(feature_name)}"
            for feature_name in feature_names
        ],
    ]
    shap_sql = ",\n      ".join(shap_columns)
    joined_shap_sql = ",\n    ".join(f"shap.{quote_ident(feature_name)}" for feature_name in feature_names)
    return f"""
WITH joined AS (
  SELECT
    base.raw_model,
    {joined_shap_sql}
  FROM (
    SELECT
      ROW_NUMBER() OVER () AS __lucidum_row_id,
      {quote_ident(model_feature)} AS raw_model
    FROM {dataset.relation_sql()}
  ) base
  INNER JOIN (
    SELECT
      {shap_sql}
    FROM read_parquet({sql_literal(str(shap_path))})
  ) shap USING (__lucidum_row_id)
)
"""


def normalise_stacked_x_sort(value: Any) -> str:
    text = str(value or "alpha").strip().lower()
    return "descending" if text in {"descending", "desc"} else "alpha"


def normalise_num_features(value: Any) -> int | None:
    text = str(value if value is not None else "all").strip().lower()
    if text in {"", "all", "none"}:
        return None
    try:
        number = int(float(text))
    except ValueError:
        return None
    return number if number > 0 else None


def model_features(dataset: Dataset, store: GbmModelStore, model_id: str) -> list[dict[str, Any]]:
    raw_features = store.read_json(store.artifact_path(model_id, "feature_config"), [])
    if not isinstance(raw_features, list):
        raw_features = []
    shap_importance = shap_summary_importance(dataset, store, model_id)
    try:
        dataset_columns = dataset.column_map()
    except duckdb.Error:
        dataset_columns = {}
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in raw_features:
        if not isinstance(item, dict):
            continue
        name = str(item.get("name") or "").strip()
        if not name or name in seen or item.get("include") is False:
            continue
        column = dataset_columns.get(name)
        kind = str(item.get("kind") or (column.kind if column else "categorical"))
        gain = finite_float(item.get("gain")) or 0.0
        mean_abs_shap = finite_float(item.get("mean_abs_shap"))
        if mean_abs_shap is None:
            mean_abs_shap = shap_importance.get(name)
        row = {
            "name": name,
            "kind": kind,
            "duckdb_type": column.duckdb_type if column else str(item.get("duckdb_type") or ""),
            "gain": round(gain, 3),
            "band_suggestion": column.band_suggestion if column else None,
        }
        if mean_abs_shap is not None:
            row["mean_abs_shap"] = mean_abs_shap
        rows.append(row)
        seen.add(name)
    use_shap = any(finite_float(row.get("mean_abs_shap")) is not None for row in rows)
    return sorted(
        rows,
        key=lambda row: (
            -feature_importance_value(row, use_shap=use_shap),
            str(row["name"]).lower(),
        ),
    )


def shap_summary_importance(dataset: Dataset, store: GbmModelStore, model_id: str) -> dict[str, float]:
    path = store.artifact_path(model_id, "shap_summary")
    if not path.exists():
        return {}
    try:
        with dataset.lock:
            columns = parquet_columns(dataset.con, path)
            if not {"feature", "mean_abs_shap"}.issubset(columns):
                return {}
            records = dataset.con.execute(
                f"""
SELECT feature, mean_abs_shap
FROM read_parquet({sql_literal(str(path))})
WHERE feature IS NOT NULL
"""
            ).fetchall()
    except duckdb.Error:
        return {}
    values: dict[str, float] = {}
    for feature, value in records:
        name = str(feature or "").strip()
        number = json_number(value)
        if name and number is not None:
            values[name] = float(number)
    return values


def feature_importance_value(feature: dict[str, Any], *, use_shap: bool) -> float:
    key = "mean_abs_shap" if use_shap else "gain"
    value = finite_float(feature.get(key))
    return value if value is not None else 0.0


def flame_plot(
    dataset: Dataset,
    store: GbmModelStore,
    model_id: str,
    feature: str,
    banding: float,
    tail_fraction: float,
    feature_info: dict[str, Any],
) -> dict[str, Any]:
    cte = joined_cte(dataset, store, model_id, feature)
    bounds = numeric_bounds(dataset.con, cte, "raw_1", tail_fraction)
    banded = band_expr("TRY_CAST(raw_1 AS DOUBLE)", banding, bounds)
    percentile_sql = ",\n    ".join(percentile_selects("shap_1", FLAME_PERCENTILES))
    sql = f"""
{cte}
SELECT
    {banded} AS x,
    COUNT(*) AS row_count,
    {percentile_sql}
FROM joined
WHERE TRY_CAST(raw_1 AS DOUBLE) IS NOT NULL
  AND shap_1 IS NOT NULL
GROUP BY x
ORDER BY x
"""
    percentile_keys = [percentile_key(percentile) for percentile in FLAME_PERCENTILES]
    rows = clean_numeric_rows(query_dicts(dataset.con, sql), ["x", *percentile_keys])
    return {
        "plot_type": "flame",
        "title": f"SHAP flame plot: {feature}",
        "x_feature": feature,
        "y_label": "SHAP",
        "row_count": sum(int(row.get("row_count") or 0) for row in rows),
        "total_shap_rows": joined_count(dataset.con, cte),
        "banding": banding,
        "percentiles": list(FLAME_PERCENTILES),
        "x_domain": numeric_domain(rows, ["x"]),
        "y_domain": numeric_domain(rows, percentile_keys),
        "rows": rows,
        "warnings": numeric_missing_warnings(dataset.con, cte, [("Feature 1", "raw_1")]),
    }


def box_plot(
    dataset: Dataset,
    store: GbmModelStore,
    model_id: str,
    feature: str,
    banding: float,
    tail_fraction: float,
    force_factor: bool,
    feature_info: dict[str, Any],
) -> dict[str, Any]:
    cte = joined_cte(dataset, store, model_id, feature)
    level = factor_level_sql(feature_info, "raw_1", banding, tail_fraction, force_factor, dataset.con, cte)
    percentile_sql = ",\n    ".join(percentile_selects("shap_1", BOX_PERCENTILES))
    order_sql = "ORDER BY sort_value, level" if feature_is_numeric(feature_info) and force_factor else "ORDER BY p50 DESC NULLS LAST, level"
    sql = f"""
{cte}
SELECT
    {level["label"]} AS level,
    {level["sort"]} AS sort_value,
    COUNT(*) AS row_count,
    AVG(shap_1) AS mean,
    {percentile_sql}
FROM joined
WHERE shap_1 IS NOT NULL
GROUP BY level, sort_value
{order_sql}
LIMIT {MAX_FACTOR_LEVELS + 1}
"""
    rows = query_dicts(dataset.con, sql)
    warnings: list[str] = []
    if len(rows) > MAX_FACTOR_LEVELS:
        rows = rows[:MAX_FACTOR_LEVELS]
        order_label = "numeric band order" if feature_is_numeric(feature_info) and force_factor else "median SHAP"
        warnings.append(f"Showing the first {MAX_FACTOR_LEVELS} levels sorted by {order_label}.")
    return {
        "plot_type": "box",
        "title": f"SHAP box plot: {feature}",
        "x_feature": feature,
        "y_label": "SHAP",
        "row_count": sum(int(row.get("row_count") or 0) for row in rows),
        "total_shap_rows": joined_count(dataset.con, cte),
        "rows": clean_numeric_rows(rows, ["mean", *[percentile_key(percentile) for percentile in BOX_PERCENTILES]]),
        "warnings": warnings,
    }


def surface_plot(
    dataset: Dataset,
    store: GbmModelStore,
    model_id: str,
    feature_1: str,
    feature_2: str,
    banding_1: float,
    banding_2: float,
    tail_fraction: float,
    features: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    cte = joined_cte(dataset, store, model_id, feature_1, feature_2)
    bounds_1 = numeric_bounds(dataset.con, cte, "raw_1", tail_fraction)
    bounds_2 = numeric_bounds(dataset.con, cte, "raw_2", tail_fraction)
    banded_1 = band_expr("TRY_CAST(raw_1 AS DOUBLE)", banding_1, bounds_1)
    banded_2 = band_expr("TRY_CAST(raw_2 AS DOUBLE)", banding_2, bounds_2)
    sql = f"""
{cte}
SELECT
    {banded_2} AS x,
    {banded_1} AS y,
    COUNT(*) AS row_count,
    AVG(shap_1 + shap_2) AS z
FROM joined
WHERE TRY_CAST(raw_1 AS DOUBLE) IS NOT NULL
  AND TRY_CAST(raw_2 AS DOUBLE) IS NOT NULL
  AND shap_1 IS NOT NULL
  AND shap_2 IS NOT NULL
GROUP BY x, y
ORDER BY y, x
"""
    sparse_rows = clean_numeric_rows(query_dicts(dataset.con, sql), ["x", "y", "z"])
    rows, x_values, y_values = dense_surface_rows(sparse_rows)
    return {
        "plot_type": "surface",
        "title": f"SHAP surface plot: {feature_1} x {feature_2}",
        "x_feature": feature_2,
        "y_feature": feature_1,
        "z_label": "SHAP",
        "row_count": sum(int(row.get("row_count") or 0) for row in sparse_rows),
        "total_shap_rows": joined_count(dataset.con, cte),
        "x_domain": numeric_domain_from_values(x_values),
        "y_domain": numeric_domain_from_values(y_values),
        "grid": {"x_values": x_values, "y_values": y_values, "data_shape": [len(y_values), len(x_values)]},
        "rows": rows,
        "warnings": numeric_missing_warnings(dataset.con, cte, [("Feature 1", "raw_1"), ("Feature 2", "raw_2")]),
    }


def lines_plot(
    dataset: Dataset,
    store: GbmModelStore,
    model_id: str,
    feature_1: str,
    feature_2: str,
    banding_1: float,
    banding_2: float,
    tail_fraction: float,
    factor_1: bool,
    factor_2: bool,
    features: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    cte = joined_cte(dataset, store, model_id, feature_1, feature_2)
    first_is_continuous = feature_is_continuous(features[feature_1], factor_1)
    x_alias, series_alias = ("raw_1", "raw_2") if first_is_continuous else ("raw_2", "raw_1")
    x_feature, series_feature = (feature_1, feature_2) if first_is_continuous else (feature_2, feature_1)
    x_banding = banding_1 if first_is_continuous else banding_2
    series_banding = banding_2 if first_is_continuous else banding_1
    series_force_factor = factor_2 if first_is_continuous else factor_1
    bounds = numeric_bounds(dataset.con, cte, x_alias, tail_fraction)
    x_expr = band_expr(f"TRY_CAST({x_alias} AS DOUBLE)", x_banding, bounds)
    series_level = factor_level_sql(features[series_feature], series_alias, series_banding, tail_fraction, series_force_factor, dataset.con, cte)
    sql = f"""
{cte}
SELECT
    {x_expr} AS x,
    {series_level["label"]} AS series,
    {series_level["sort"]} AS series_sort,
    COUNT(*) AS row_count,
    AVG(shap_1 + shap_2) AS y
FROM joined
WHERE TRY_CAST({x_alias} AS DOUBLE) IS NOT NULL
  AND shap_1 IS NOT NULL
  AND shap_2 IS NOT NULL
GROUP BY x, series, series_sort
ORDER BY series_sort, series, x
"""
    rows = clean_numeric_rows(query_dicts(dataset.con, sql), ["x", "y"])
    series_names = []
    seen: set[str] = set()
    for row in rows:
        name = str(row.get("series"))
        if name not in seen:
            series_names.append(name)
            seen.add(name)
    warnings = numeric_missing_warnings(dataset.con, cte, [(x_feature, x_alias)])
    if len(series_names) > MAX_LINE_SERIES:
        keep = set(series_names[:MAX_LINE_SERIES])
        rows = [row for row in rows if str(row.get("series")) in keep]
        warnings.append(f"Showing the first {MAX_LINE_SERIES} line series.")
    return {
        "plot_type": "lines",
        "title": f"SHAP lines plot: {x_feature} x {series_feature}",
        "x_feature": x_feature,
        "series_feature": series_feature,
        "y_label": "SHAP",
        "row_count": sum(int(row.get("row_count") or 0) for row in rows),
        "total_shap_rows": joined_count(dataset.con, cte),
        "x_domain": numeric_domain(rows, ["x"]),
        "y_domain": numeric_domain(rows, ["y"]),
        "rows": rows,
        "warnings": warnings,
    }


def heatmap_plot(
    dataset: Dataset,
    store: GbmModelStore,
    model_id: str,
    feature_1: str,
    feature_2: str,
    banding_1: float,
    banding_2: float,
    tail_fraction: float,
    factor_1: bool,
    factor_2: bool,
    features: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    cte = joined_cte(dataset, store, model_id, feature_1, feature_2)
    y_level = factor_level_sql(features[feature_1], "raw_1", banding_1, tail_fraction, factor_1, dataset.con, cte)
    x_level = factor_level_sql(features[feature_2], "raw_2", banding_2, tail_fraction, factor_2, dataset.con, cte)
    sql = f"""
{cte}
SELECT
    {x_level["label"]} AS x,
    {x_level["sort"]} AS x_sort,
    {y_level["label"]} AS y,
    {y_level["sort"]} AS y_sort,
    COUNT(*) AS row_count,
    AVG(shap_1 + shap_2) AS z
FROM joined
WHERE shap_1 IS NOT NULL
  AND shap_2 IS NOT NULL
GROUP BY x, x_sort, y, y_sort
ORDER BY y_sort, y, x_sort, x
LIMIT {MAX_HEATMAP_CELLS + 1}
"""
    rows = clean_numeric_rows(query_dicts(dataset.con, sql), ["z"])
    warnings: list[str] = []
    if len(rows) > MAX_HEATMAP_CELLS:
        rows = rows[:MAX_HEATMAP_CELLS]
        warnings.append(f"Showing the first {MAX_HEATMAP_CELLS} heatmap cells.")
    return {
        "plot_type": "heatmap",
        "title": f"SHAP heatmap: {feature_1} x {feature_2}",
        "x_feature": feature_2,
        "y_feature": feature_1,
        "z_label": "SHAP",
        "row_count": sum(int(row.get("row_count") or 0) for row in rows),
        "total_shap_rows": joined_count(dataset.con, cte),
        "rows": rows,
        "warnings": warnings,
    }


def joined_cte(dataset: Dataset, store: GbmModelStore, model_id: str, feature_1: str, feature_2: str | None = None) -> str:
    shap_path = store.artifact_path(model_id, "shap_long")
    base_columns = [
        "ROW_NUMBER() OVER () AS __lucidum_row_id",
        f"{quote_ident(feature_1)} AS raw_1",
    ]
    shap_columns = [
        "__lucidum_row_id",
        f"TRY_CAST({quote_ident(feature_1)} AS DOUBLE) AS shap_1",
    ]
    if feature_2:
        base_columns.append(f"{quote_ident(feature_2)} AS raw_2")
        shap_columns.append(f"TRY_CAST({quote_ident(feature_2)} AS DOUBLE) AS shap_2")
    else:
        base_columns.append("NULL AS raw_2")
        shap_columns.append("NULL::DOUBLE AS shap_2")
    base_sql = ",\n    ".join(base_columns)
    shap_sql = ",\n    ".join(shap_columns)
    return f"""
WITH joined AS (
  SELECT
    base.raw_1,
    base.raw_2,
    shap.shap_1,
    shap.shap_2
  FROM (
    SELECT
      {base_sql}
    FROM {dataset.relation_sql()}
  ) base
  INNER JOIN (
    SELECT
      {shap_sql}
    FROM read_parquet({sql_literal(str(shap_path))})
  ) shap USING (__lucidum_row_id)
)
"""


def numeric_bounds(con: duckdb.DuckDBPyConnection, cte: str, raw_alias: str, tail_fraction: float) -> tuple[float | None, float | None]:
    raw = f"TRY_CAST({raw_alias} AS DOUBLE)"
    q = max(0.0, min(0.49, tail_fraction))
    sql = f"""
{cte}
SELECT
    quantile_cont({raw}, {q}) AS lower_bound,
    quantile_cont({raw}, {1 - q}) AS upper_bound
FROM joined
WHERE {raw} IS NOT NULL
"""
    row = con.execute(sql).fetchone()
    if not row:
        return None, None
    return finite_float(row[0]), finite_float(row[1])


def band_expr(raw_expr: str, banding: float, bounds: tuple[float | None, float | None]) -> str:
    lower, upper = bounds
    if lower is None or upper is None:
        return "NULL"
    band = max(float(banding), 1e-12)
    return (
        f"CASE WHEN {raw_expr} IS NULL THEN NULL ELSE "
        f"GREATEST({sql_float(lower)}, LEAST({sql_float(upper)}, FLOOR({raw_expr} / {sql_float(band)}) * {sql_float(band)})) END"
    )


def factor_level_sql(
    feature: dict[str, Any],
    raw_alias: str,
    banding: float,
    tail_fraction: float,
    force_factor: bool,
    con: duckdb.DuckDBPyConnection,
    cte: str,
) -> dict[str, str]:
    if feature_is_numeric(feature) and force_factor:
        raw = f"TRY_CAST({raw_alias} AS DOUBLE)"
        grouped = band_expr(raw, banding, numeric_bounds(con, cte, raw_alias, tail_fraction))
        return {
            "label": f"CASE WHEN {raw} IS NULL THEN '(missing)' ELSE CAST({grouped} AS VARCHAR) END",
            "sort": f"COALESCE({grouped}, 1e308)",
        }
    return {
        "label": f"COALESCE(CAST({raw_alias} AS VARCHAR), '(missing)')",
        "sort": f"COALESCE(CAST({raw_alias} AS VARCHAR), '(missing)')",
    }


def percentile_selects(value_expr: str, percentiles: tuple[int, ...]) -> list[str]:
    selects: list[str] = []
    for percentile in percentiles:
        if percentile == 0:
            selects.append(f"MIN({value_expr}) AS {percentile_key(percentile)}")
        elif percentile == 100:
            selects.append(f"MAX({value_expr}) AS {percentile_key(percentile)}")
        else:
            selects.append(f"quantile_cont({value_expr}, {percentile / 100}) AS {percentile_key(percentile)}")
    return selects


def percentile_key(percentile: int) -> str:
    return f"p{percentile}"


def numeric_missing_warnings(con: duckdb.DuckDBPyConnection, cte: str, aliases: list[tuple[str, str]]) -> list[str]:
    warnings: list[str] = []
    for label, alias in aliases:
        raw = f"TRY_CAST({alias} AS DOUBLE)"
        row = con.execute(f"{cte}\nSELECT COUNT(*) FROM joined WHERE {raw} IS NULL").fetchone()
        count = int(row[0] or 0)
        if count:
            warnings.append(f"{label}: omitted {count:,} rows with missing numeric values.")
    return warnings


def joined_count(con: duckdb.DuckDBPyConnection, cte: str) -> int:
    return int(con.execute(f"{cte}\nSELECT COUNT(*) FROM joined").fetchone()[0] or 0)


def parquet_columns(con: duckdb.DuckDBPyConnection, path: Path) -> set[str]:
    rows = con.execute(f"DESCRIBE SELECT * FROM read_parquet({sql_literal(str(path))})").fetchall()
    return {str(row[0]) for row in rows}


def query_dicts(con: duckdb.DuckDBPyConnection, sql: str) -> list[dict[str, Any]]:
    rows = con.execute(sql).fetchall()
    names = [description[0] for description in con.description]
    return [dict(zip(names, row)) for row in rows]


def clean_numeric_rows(rows: list[dict[str, Any]], keys: list[str]) -> list[dict[str, Any]]:
    for row in rows:
        for key in keys:
            if key in row:
                row[key] = json_number(row[key])
    return rows


def numeric_domain(rows: list[dict[str, Any]], keys: list[str]) -> list[float | int] | None:
    values: list[float | int] = []
    for row in rows:
        for key in keys:
            number = json_number(row.get(key))
            if number is not None:
                values.append(number)
    return numeric_domain_from_values(values)


def numeric_domain_from_values(values: list[Any]) -> list[float | int] | None:
    cleaned = [json_number(value) for value in values]
    numbers = [value for value in cleaned if value is not None]
    if not numbers:
        return None
    return [min(numbers), max(numbers)]


def dense_surface_rows(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[float | int], list[float | int]]:
    x_values = sorted({value for row in rows if (value := json_number(row.get("x"))) is not None})
    y_values = sorted({value for row in rows if (value := json_number(row.get("y"))) is not None})
    row_by_cell = {
        (json_number(row.get("x")), json_number(row.get("y"))): row
        for row in rows
        if json_number(row.get("x")) is not None and json_number(row.get("y")) is not None
    }
    dense_rows: list[dict[str, Any]] = []
    for y_value in y_values:
        for x_value in x_values:
            source = row_by_cell.get((x_value, y_value))
            z_value = json_number(source.get("z")) if source else None
            has_data = source is not None and z_value is not None
            dense_rows.append(
                {
                    "x": x_value,
                    "y": y_value,
                    "z": z_value,
                    "row_count": int(source.get("row_count") or 0) if source else 0,
                    "has_data": has_data,
                }
            )
    return dense_rows, x_values, y_values


def feature_payload(feature: dict[str, Any], banding: float, factor: bool) -> dict[str, Any]:
    payload = {
        "name": feature["name"],
        "kind": feature["kind"],
        "gain": feature.get("gain", 0.0),
        "banding": banding,
        "factor": bool(factor),
    }
    mean_abs_shap = finite_float(feature.get("mean_abs_shap"))
    if mean_abs_shap is not None:
        payload["mean_abs_shap"] = mean_abs_shap
    return payload


def normalise_feature_name(value: Any, *, allow_none: bool = False) -> str:
    text = str(value or "").strip()
    if allow_none and text.lower() in {"", "none", "__none__"}:
        return ""
    return text


def normalise_banding(value: Any, default: Any = None) -> float:
    parsed = parse_positive_float(value)
    if parsed is not None:
        return float(parsed)
    fallback = parse_positive_float(default)
    return float(fallback if fallback is not None else 1.0)


def normalise_tail_fraction(value: Any) -> float:
    text = str(value if value is not None else "1").strip()
    if text.endswith("%"):
        text = text[:-1].strip()
    try:
        number = float(text)
    except ValueError:
        return 0.01
    if not math.isfinite(number) or number <= 0:
        return 0.0
    return min(number / 100, 0.49)


def feature_is_numeric(feature: dict[str, Any]) -> bool:
    return is_numeric_kind(str(feature.get("kind") or ""))


def feature_is_continuous(feature: dict[str, Any], force_factor: bool) -> bool:
    return feature_is_numeric(feature) and not force_factor


def finite_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def sql_float(value: float) -> str:
    return repr(float(value))


__all__ = ["shap_config", "shap_plot", "stacked_shap_plot"]
