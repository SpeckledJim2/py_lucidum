from __future__ import annotations

import importlib.util
import json
import math
import os
import pickle
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

from py_lucidum.core import (
    Dataset,
    denominator_valid_condition,
    is_numeric_kind,
    json_number,
    quote_ident,
    sql_literal,
    weighted_value_sql,
)

from .store import GlmModelStore
from .tabulation import (
    _base_row,
    _base_value,
    _column_tokens,
    _feature_spec_map,
    _feature_transform_bounds,
    _json_value,
    _term_groups,
)
from .training import MissingGlmDependency, formula_context, glm_dependencies, offset_values_for_frame
from .validation import TARGET_COLUMN


DEFAULT_GLM_OVERLAY_SAMPLE_SEED = 2026
MAX_GLM_OVERLAY_SAMPLE_ROWS = 100_000
MAX_GLM_OVERLAY_PREDICTION_CELLS = 2_000_000
GLM_OVERLAY_CHUNK_CELLS = 100_000


def empty_glm_partial_dependence_warning(message: str) -> dict[str, Any]:
    return {
        "mode": "glm",
        "model_id": "",
        "feature": "",
        "method": "none",
        "percentiles": [50],
        "rows": [],
        "warnings": [message],
        "scale": {"method": "none", "target": None, "source_mean": None},
        "sample": {},
        "transform": {"mode": "none"},
    }


def build_glm_partial_dependence_overlay(
    dataset: Dataset,
    request: dict[str, Any],
    *,
    feature_spec: Any,
    x_col: str,
    x_sql: dict[str, str],
    x_group_kind: str,
    denominator: dict[str, str | None],
) -> dict[str, Any]:
    if should_isolate_glm_overlay():
        return build_glm_partial_dependence_overlay_in_subprocess(
            dataset,
            request,
            feature_spec=feature_spec,
            x_col=x_col,
            x_sql=x_sql,
            x_group_kind=x_group_kind,
            denominator=denominator,
        )
    return _build_glm_partial_dependence_overlay_impl(
        dataset,
        request,
        feature_spec=feature_spec,
        x_col=x_col,
        x_sql=x_sql,
        x_group_kind=x_group_kind,
        denominator=denominator,
    )


def should_isolate_glm_overlay() -> bool:
    return (
        ("lightgbm" in sys.modules or importlib.util.find_spec("lightgbm") is not None)
        and not os.environ.get("PY_LUCIDUM_GLM_OVERLAY_WORKER")
    )


def build_glm_partial_dependence_overlay_in_subprocess(
    dataset: Dataset,
    request: dict[str, Any],
    *,
    feature_spec: Any,
    x_col: str,
    x_sql: dict[str, str],
    x_group_kind: str,
    denominator: dict[str, str | None],
) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="lucidum-glm-overlay-") as tmp_dir:
        tmp_path = Path(tmp_dir)
        request_path = tmp_path / "request.json"
        response_path = tmp_path / "response.json"
        request_path.write_text(
            json.dumps(
                {
                    "dataset_path": str(dataset.path),
                    "request": request,
                    "feature_spec": feature_spec,
                    "x_col": x_col,
                    "x_sql": x_sql,
                    "x_group_kind": x_group_kind,
                    "denominator": denominator,
                },
                default=str,
            ),
            encoding="utf-8",
        )
        completed = subprocess.run(
            [sys.executable, "-m", "py_lucidum.tools.glm.overlay_worker", str(request_path), str(response_path)],
            check=False,
            capture_output=True,
            text=True,
            env={
                **os.environ,
                "PY_LUCIDUM_SKIP_LIGHTGBM_PRELOAD": "1",
                "PY_LUCIDUM_GLM_OVERLAY_WORKER": "1",
            },
        )
        if completed.returncode != 0:
            detail = (completed.stderr or completed.stdout or "").strip()
            if len(detail) > 800:
                detail = f"{detail[:800]}..."
            suffix = f" {detail}" if detail else ""
            return empty_glm_partial_dependence_warning(f"GLM overlay worker exited unexpectedly with code {completed.returncode}.{suffix}")
        if not response_path.exists():
            return empty_glm_partial_dependence_warning("GLM overlay worker exited without writing a response.")
        response = json.loads(response_path.read_text(encoding="utf-8"))
    if not response.get("ok"):
        return empty_glm_partial_dependence_warning(str(response.get("error") or "GLM overlay worker failed."))
    result = response.get("result")
    if not isinstance(result, dict):
        return empty_glm_partial_dependence_warning("GLM overlay worker returned an invalid response.")
    return result


def _build_glm_partial_dependence_overlay_impl(
    dataset: Dataset,
    request: dict[str, Any],
    *,
    feature_spec: Any,
    x_col: str,
    x_sql: dict[str, str],
    x_group_kind: str,
    denominator: dict[str, str | None],
) -> dict[str, Any]:
    store = GlmModelStore(dataset.path)
    model_id = store.active_model_id()
    if not model_id:
        return empty_glm_partial_dependence_warning("No active GLM is available for GLM overlay.")
    estimator_path = store.artifact_path(model_id, "estimator")
    if not estimator_path.exists():
        return empty_glm_partial_dependence_warning("Rebuild the active GLM before using GLM overlay; estimator.pkl is missing.")
    try:
        _glum, _glr, _glrcv, np, pd = glm_dependencies()
    except MissingGlmDependency as exc:
        return empty_glm_partial_dependence_warning(str(exc))

    with estimator_path.open("rb") as handle:
        estimator = pickle.load(handle)
    manifest = store.manifest(model_id)
    source_columns = store.source_columns(manifest)
    if x_col not in source_columns:
        return empty_glm_partial_dependence_warning(f"The active GLM source does not include {x_col}.")

    context = formula_context(np)
    offset_terms = [str(term) for term in (manifest.get("offset_terms") or manifest.get("formula", {}).get("offset_terms") or [])]
    groups = _term_groups(estimator, offset_terms, source_columns)
    all_features = sorted({feature for features in groups for feature in features})
    interaction = any(x_col in features and len(features) > 1 for features in groups)
    relation = glm_overlay_relation_sql(store, model_id, manifest, source_columns)
    try:
        filter_sql = dataset.normalise_filter_for_relation(request.get("filter"), relation)
    except ValueError as exc:
        return empty_glm_partial_dependence_warning(f"GLM overlay could not use the current filter: {exc}")

    overlay_denominator = normalise_glm_overlay_denominator(denominator, source_columns)
    target_mean = glm_prediction_mean(dataset, relation, overlay_denominator, filter_sql)
    initial_rows = glm_x_group_rows(dataset, relation, x_sql, overlay_denominator, filter_sql)
    group_mapping = glm_low_weight_group_mapping(initial_rows, x_group_kind, str(request.get("lowGroup") or "0"))
    x_rows = [row for row in initial_rows if usable_x_value(row)]
    x_value_count = len(x_rows)
    if not x_rows:
        return {
            "mode": "glm",
            "model_id": model_id,
            "feature": x_col,
            "method": "sampled_marginal" if interaction else "base_profile",
            "percentiles": [50],
            "rows": [],
            "warnings": ["No GLM overlay x-axis groups matched the current chart selection."],
            "scale": {"method": "none", "target": None, "source_mean": None},
            "sample": {},
            "transform": {"mode": str(request.get("transform") or "none")},
        }

    required_columns = glm_required_columns(source_columns, manifest, all_features, offset_terms, x_col, overlay_denominator)
    seed = glm_overlay_seed(manifest)
    sample_limit = overlay_sample_limit(x_value_count=x_value_count)
    sample_frame, population_row_count = glm_sample_frame(
        dataset,
        relation,
        required_columns,
        overlay_denominator,
        filter_sql,
        seed=seed,
        sample_limit=sample_limit,
    )
    if sample_frame.empty:
        return empty_glm_partial_dependence_warning("No GLM overlay rows matched the current chart selection.")

    kinds = {column.name: column.kind for column in dataset.valid_schema_columns()}
    spec_rows = _feature_spec_map(feature_spec)
    transform_bounds = _feature_transform_bounds(estimator, source_columns)
    method = "sampled_marginal" if interaction else "base_profile"
    if interaction:
        source_rows = sampled_marginal_rows(
            estimator,
            manifest,
            sample_frame,
            x_rows,
            x_col=x_col,
            denominator=overlay_denominator,
            offset_terms=offset_terms,
            context=context,
            np=np,
            pd=pd,
        )
    else:
        source_rows = base_profile_rows(
            estimator,
            manifest,
            sample_frame,
            x_rows,
            x_col=x_col,
            denominator=overlay_denominator,
            offset_terms=offset_terms,
            context=context,
            all_features=all_features,
            kinds=kinds,
            spec_rows=spec_rows,
            transform_bounds=transform_bounds,
            np=np,
            pd=pd,
        )
    rows = aggregate_source_rows(source_rows, group_mapping)
    scale = scale_glm_overlay_rows(rows, target_mean, manifest=manifest)
    if x_group_kind == "numeric":
        clean_numeric_labels(rows, request.get("bandWidth"))
    warnings: list[str] = []
    if scale.get("warning"):
        warnings.append(str(scale["warning"]))
    prediction_cell_count = len(sample_frame) * x_value_count if interaction else x_value_count
    if interaction and int(population_row_count) > len(sample_frame):
        warnings.append(
            "GLM overlay used a deterministic sample of "
            f"{len(sample_frame):,} from {int(population_row_count):,} eligible rows."
        )
    return {
        "mode": "glm",
        "model_id": model_id,
        "feature": x_col,
        "method": method,
        "percentiles": [50],
        "rows": rows,
        "warnings": warnings,
        "scale": {key: value for key, value in scale.items() if key != "warning"},
        "sample": {
            "population_row_count": int(population_row_count),
            "sample_row_count": int(len(sample_frame)) if interaction else int(min(len(sample_frame), population_row_count)),
            "x_value_count": int(x_value_count),
            "prediction_cell_count": int(prediction_cell_count),
            "max_sample_rows": MAX_GLM_OVERLAY_SAMPLE_ROWS,
            "max_prediction_cells": MAX_GLM_OVERLAY_PREDICTION_CELLS,
            "seed": seed,
        },
        "transform": {"mode": str(request.get("transform") or "none")},
    }


def glm_prediction_mean(
    dataset: Dataset,
    relation: str,
    denominator: dict[str, str | None],
    filter_sql: str,
) -> float | int | None:
    checks = ["TRY_CAST(glm_prediction AS DOUBLE) IS NOT NULL"]
    denominator_condition = denominator_valid_condition([], denominator)
    if denominator_condition != "TRUE":
        checks.append(denominator_condition)
    valid_condition = " AND ".join(checks)
    weight_expr = weighted_value_sql(denominator, valid_condition)
    where_sql = f"\nWHERE ({filter_sql})" if filter_sql else ""
    sql = f"""
SELECT
  SUM(CASE WHEN {valid_condition} THEN TRY_CAST(glm_prediction AS DOUBLE) ELSE NULL END) AS numerator,
  COALESCE(SUM({weight_expr}), 0) AS denominator
FROM {relation}
{where_sql}
"""
    row = dataset.con.execute(sql).fetchone()
    numerator = json_number(row[0] if row else None)
    denominator_value = json_number(row[1] if row else None)
    if numerator is None or denominator_value in (None, 0):
        return None
    return json_number(float(numerator) / float(denominator_value))


def glm_overlay_relation_sql(store: GlmModelStore, model_id: str, manifest: dict[str, Any], source_columns: list[str]) -> str:
    prediction_path = store.artifact_path(model_id, "predictions")
    columns_sql = ",\n    ".join(quote_ident(name) for name in source_columns)
    suffix = f",\n    {columns_sql}" if columns_sql else ""
    return f"""(
SELECT
  base.*,
  prediction.glm_prediction
FROM (
  SELECT
    ROW_NUMBER() OVER () AS __lucidum_row_id{suffix}
  FROM {store.dataset_relation_sql()}
) base
INNER JOIN read_parquet({sql_literal(str(prediction_path))}) prediction USING (__lucidum_row_id)
)"""


def normalise_glm_overlay_denominator(denominator: dict[str, str | None], source_columns: list[str]) -> dict[str, str | None]:
    column = str(denominator.get("column") or "").strip()
    if column and column in source_columns:
        return denominator
    return {"column": None, "label": "Average row value", "bar_label": "Row count"}


def glm_x_group_rows(
    dataset: Dataset,
    relation: str,
    x_sql: dict[str, str],
    denominator: dict[str, str | None],
    filter_sql: str,
) -> list[dict[str, Any]]:
    valid_condition = denominator_valid_condition([], denominator)
    weight_expr = weighted_value_sql(denominator, valid_condition)
    where_sql = f"\n  WHERE ({filter_sql})" if filter_sql else ""
    quantile_cte = ""
    keyed_from = "base"
    rownum_expr = "__rownum"
    source_columns = "*"
    x_value_expr = "x_key"
    x_value_select = "MIN(__x_value) AS x_value"
    if x_sql.get("quantile_count"):
        quantile_cte = f""",
quantiles AS (
  SELECT
    __rownum,
    NTILE({x_sql['quantile_count']}) OVER (ORDER BY __x_raw, __rownum) AS __x_quantile
  FROM (
    SELECT
      __rownum,
      {x_sql['raw']} AS __x_raw
    FROM base
    WHERE {x_sql['raw']} IS NOT NULL
  ) non_missing
)"""
        keyed_from = "base LEFT JOIN quantiles USING (__rownum)"
        rownum_expr = "base.__rownum"
        source_columns = "base.*"
        x_value_expr = x_sql["raw"]
        x_value_select = "AVG(__x_value) AS x_value"
    sql = f"""
WITH base AS (
  SELECT ROW_NUMBER() OVER () AS __rownum, * FROM {relation}{where_sql}
){quantile_cte},
keyed AS (
  SELECT
    {rownum_expr} AS __rownum,
    {x_sql['key']} AS x_key,
    {x_sql['label']} AS x_label,
    {x_sql['sort']} AS x_sort,
    {x_value_expr} AS __x_value,
    {weight_expr} AS __weight_value,
    {source_columns}
  FROM {keyed_from}
),
valid AS (
  SELECT *
  FROM keyed
  WHERE __weight_value IS NOT NULL
    AND TRY_CAST(glm_prediction AS DOUBLE) IS NOT NULL
)
SELECT
  x_label,
  MIN(x_sort) AS x_sort,
  MIN(__rownum) AS original_order,
  COALESCE(SUM(__weight_value), 0) AS volume,
  {x_value_select}
FROM valid
GROUP BY x_label
"""
    cursor = dataset.con.execute(sql)
    raw_rows = [dict(zip([d[0] for d in cursor.description], row)) for row in cursor.fetchall()]
    return [
        {
            "x": str(row.get("x_label")),
            "x_sort": row.get("x_sort"),
            "original_order": int(row.get("original_order") or 0),
            "volume": json_number(row.get("volume")) or 0,
            "x_value": _json_value(row.get("x_value")),
            "is_tail": False,
        }
        for row in raw_rows
    ]


def glm_required_columns(
    source_columns: list[str],
    manifest: dict[str, Any],
    all_features: list[str],
    offset_terms: list[str],
    x_col: str,
    denominator: dict[str, str | None],
) -> list[str]:
    requested = set(all_features)
    requested.add(x_col)
    model_denominator = str(manifest.get("denominator_column") or "").strip()
    if model_denominator:
        requested.add(model_denominator)
    selected_denominator = str(denominator.get("column") or "").strip()
    if selected_denominator:
        requested.add(selected_denominator)
    for expression in offset_terms:
        requested.update(_column_tokens(expression, source_columns))
    return [column for column in source_columns if column in requested]


def glm_sample_frame(
    dataset: Dataset,
    relation: str,
    columns: list[str],
    denominator: dict[str, str | None],
    filter_sql: str,
    *,
    seed: int,
    sample_limit: int,
) -> tuple[Any, int]:
    valid_condition = denominator_valid_condition([], denominator)
    where_parts = ["TRY_CAST(glm_prediction AS DOUBLE) IS NOT NULL"]
    if valid_condition != "TRUE":
        where_parts.append(f"({valid_condition})")
    if filter_sql:
        where_parts.append(f"({filter_sql})")
    where_sql = f"WHERE {' AND '.join(where_parts)}" if where_parts else ""
    count = int(dataset.con.execute(f"SELECT COUNT(*) FROM {relation} {where_sql}").fetchone()[0] or 0)
    select_columns = ["__lucidum_row_id", *[quote_ident(name) for name in columns]]
    select_sql = ",\n  ".join(select_columns)
    sql = f"""
SELECT
  {select_sql}
FROM {relation}
{where_sql}
ORDER BY hash(__lucidum_row_id + {int(seed)}), __lucidum_row_id
LIMIT {max(1, int(sample_limit))}
"""
    return dataset.con.execute(sql).fetchdf(), count


def overlay_sample_limit(*, x_value_count: int) -> int:
    if x_value_count <= 0:
        return 1
    by_cells = max(1, MAX_GLM_OVERLAY_PREDICTION_CELLS // max(1, int(x_value_count)))
    return min(MAX_GLM_OVERLAY_SAMPLE_ROWS, by_cells)


def sampled_marginal_rows(
    estimator: Any,
    manifest: dict[str, Any],
    sample_frame: Any,
    x_rows: list[dict[str, Any]],
    *,
    x_col: str,
    denominator: dict[str, str | None],
    offset_terms: list[str],
    context: dict[str, Any],
    np: Any,
    pd: Any,
) -> list[dict[str, Any]]:
    if sample_frame.empty:
        return []
    chunk_size = max(1, min(len(x_rows), GLM_OVERLAY_CHUNK_CELLS // max(1, len(sample_frame))))
    by_x: dict[str, dict[str, float]] = {}
    for start in range(0, len(x_rows), chunk_size):
        chunk = x_rows[start : start + chunk_size]
        frames = []
        for row in chunk:
            frame = sample_frame.copy()
            frame["__overlay_x"] = str(row["x"])
            frame[x_col] = row["x_value"]
            frames.append(frame)
        block = pd.concat(frames, ignore_index=True)
        numerators = predict_glm_numerators(estimator, manifest, block, offset_terms, context, np, pd)
        weights = overlay_weights(block, denominator, np, pd)
        for x_value, numerator, weight in zip(block["__overlay_x"], numerators, weights):
            if not math.isfinite(float(weight or 0)):
                continue
            value = json_number(numerator)
            if value is None:
                continue
            bucket = by_x.setdefault(str(x_value), {"num": 0.0, "den": 0.0})
            bucket["num"] += float(value)
            bucket["den"] += float(weight)
    rows: list[dict[str, Any]] = []
    for row in x_rows:
        bucket = by_x.get(str(row["x"]), {"num": 0.0, "den": 0.0})
        rows.append({**row, "p50": json_number(bucket["num"] / bucket["den"]) if bucket["den"] else None})
    return rows


def base_profile_rows(
    estimator: Any,
    manifest: dict[str, Any],
    sample_frame: Any,
    x_rows: list[dict[str, Any]],
    *,
    x_col: str,
    denominator: dict[str, str | None],
    offset_terms: list[str],
    context: dict[str, Any],
    all_features: list[str],
    kinds: dict[str, str],
    spec_rows: dict[str, dict[str, Any]],
    transform_bounds: dict[str, dict[str, float]],
    np: Any,
    pd: Any,
) -> list[dict[str, Any]]:
    base = _base_row(sample_frame)
    for feature in sorted(set([*all_features, x_col])):
        if feature not in sample_frame.columns:
            continue
        value, _inferred, _clipped = _base_value(
            sample_frame,
            sample_frame,
            feature,
            kinds.get(feature, "categorical"),
            spec_rows.get(feature, {}),
            transform_bounds.get(feature, {}),
            pd,
        )
        base[feature] = value
    frame = pd.DataFrame([dict(base, **{x_col: row["x_value"]}) for row in x_rows])
    numerators = predict_glm_numerators(estimator, manifest, frame, offset_terms, context, np, pd)
    weights = overlay_weights(frame, denominator, np, pd)
    rows: list[dict[str, Any]] = []
    for row, numerator, weight in zip(x_rows, numerators, weights):
        value = json_number(numerator)
        denominator_value = json_number(weight)
        rows.append({**row, "p50": json_number(float(value) / float(denominator_value)) if value is not None and denominator_value not in (None, 0) else None})
    return rows


def predict_glm_numerators(
    estimator: Any,
    manifest: dict[str, Any],
    frame: Any,
    offset_terms: list[str],
    context: dict[str, Any],
    np: Any,
    pd: Any,
) -> Any:
    work = frame.copy()
    work[TARGET_COLUMN] = 0.0
    valid = pd.Series(True, index=work.index)
    predict_kwargs: dict[str, Any] = {"context": context}
    offset_values = offset_values_for_frame(work, offset_terms, context, np, pd)
    if offset_values is not None:
        offset_numeric = pd.to_numeric(offset_values, errors="coerce")
        valid = valid & offset_numeric.notna() & np.isfinite(offset_numeric.astype(float))
    model_denominator = str(manifest.get("denominator_column") or "").strip()
    denominator_values = None
    if model_denominator and model_denominator in work.columns:
        denominator_values = pd.to_numeric(work[model_denominator], errors="coerce")
        valid = valid & denominator_values.notna() & np.isfinite(denominator_values.astype(float))
    output = pd.Series(np.nan, index=work.index, dtype=float)
    if not bool(valid.any()):
        return output
    if offset_values is not None:
        predict_kwargs["offset"] = offset_numeric.loc[valid].astype(float).to_numpy()
    predictions = estimator.predict(work.loc[valid].copy(), **predict_kwargs)
    values = pd.to_numeric(pd.Series(predictions, index=work.index[valid]), errors="coerce")
    if denominator_values is not None:
        values = values * denominator_values.loc[valid].astype(float).to_numpy()
    output.loc[valid] = values
    return output


def overlay_weights(frame: Any, denominator: dict[str, str | None], np: Any, pd: Any) -> Any:
    column = str(denominator.get("column") or "").strip()
    if column and column in frame.columns:
        values = pd.to_numeric(frame[column], errors="coerce")
        return values.where(values.notna() & np.isfinite(values.astype(float)), np.nan)
    return pd.Series(1.0, index=frame.index, dtype=float)


def aggregate_source_rows(source_rows: list[dict[str, Any]], group_mapping: list[dict[str, Any]]) -> list[dict[str, Any]]:
    mapping = {str(row.get("source_x")): row for row in group_mapping}
    buckets: dict[str, dict[str, Any]] = {}
    for source in source_rows:
        mapped = mapping.get(str(source.get("x")))
        if mapped is None:
            continue
        label = str(mapped.get("final_x") or "")
        bucket = buckets.setdefault(
            label,
            {
                "x": label,
                "x_sort": mapped.get("final_x_sort"),
                "original_order": int(mapped.get("final_original_order") or 0),
                "volume": 0.0,
                "is_tail": bool(mapped.get("final_is_tail")),
                "__num": 0.0,
                "__den": 0.0,
            },
        )
        volume = float(source.get("volume") or 0)
        bucket["volume"] += volume
        value = json_number(source.get("p50"))
        if value is not None and volume:
            bucket["__num"] += float(value) * volume
            bucket["__den"] += volume
    rows: list[dict[str, Any]] = []
    for bucket in buckets.values():
        rows.append(
            {
                "x": bucket["x"],
                "x_sort": bucket["x_sort"],
                "original_order": bucket["original_order"],
                "volume": json_number(bucket["volume"]) or 0,
                "is_tail": bool(bucket["is_tail"]),
                "p50": json_number(bucket["__num"] / bucket["__den"]) if bucket["__den"] else None,
            }
        )
    return sorted(rows, key=lambda row: int(row.get("original_order") or 0))


def scale_glm_overlay_rows(rows: list[dict[str, Any]], target_mean: Any, *, manifest: dict[str, Any]) -> dict[str, Any]:
    target = json_number(target_mean)
    source_mean = weighted_overlay_average(rows)
    if target is None or source_mean is None:
        return {
            "method": "none",
            "target": target,
            "source_mean": source_mean,
            "warning": "GLM overlay could not be scaled to fitted values.",
        }
    if glm_uses_positive_scale(manifest):
        if source_mean == 0:
            return {
                "method": "none",
                "target": target,
                "source_mean": source_mean,
                "warning": "GLM overlay could not be scaled because the base-profile mean is zero.",
            }
        factor = float(target) / float(source_mean)
        for row in rows:
            value = json_number(row.get("p50"))
            row["p50"] = json_number(float(value) * factor) if value is not None else None
        return {"method": "multiply", "target": json_number(target), "source_mean": json_number(source_mean), "factor": json_number(factor)}
    shift = float(target) - float(source_mean)
    for row in rows:
        value = json_number(row.get("p50"))
        row["p50"] = json_number(float(value) + shift) if value is not None else None
    return {"method": "add", "target": json_number(target), "source_mean": json_number(source_mean), "shift": json_number(shift)}


def weighted_overlay_average(rows: list[dict[str, Any]]) -> float | int | None:
    numerator = 0.0
    denominator = 0.0
    for row in rows:
        value = json_number(row.get("p50"))
        weight = json_number(row.get("volume"))
        if value is None or weight is None:
            continue
        numerator += float(value) * float(weight)
        denominator += float(weight)
    return json_number(numerator / denominator) if denominator else None


def glm_uses_positive_scale(manifest: dict[str, Any]) -> bool:
    family = str(manifest.get("family") or "").strip().lower()
    return family in {"binomial", "gamma", "inverse.gaussian", "negative.binomial", "poisson", "tweedie"}


def usable_x_value(row: dict[str, Any]) -> bool:
    if row.get("x_value") is None:
        return False
    label = str(row.get("x") or "")
    return label not in {"(missing)", "Missing"}


def glm_overlay_seed(manifest: dict[str, Any]) -> int:
    raw = manifest.get("seed") or manifest.get("random_seed")
    try:
        seed = int(raw)
    except (TypeError, ValueError, OverflowError):
        return DEFAULT_GLM_OVERLAY_SAMPLE_SEED
    return seed if seed >= 0 else DEFAULT_GLM_OVERLAY_SAMPLE_SEED


def glm_low_weight_group_mapping(rows: list[dict[str, Any]], x_kind: str, threshold: str) -> list[dict[str, Any]]:
    total_volume = sum(float(row.get("volume") or 0) for row in rows)
    threshold_value = parse_group_threshold(threshold, total_volume)
    normalised = list(rows)
    missing_rows: list[dict[str, Any]] = []
    if x_kind == "quantile":
        missing_rows = [row for row in normalised if row["x"] == "Missing"]
        normalised = [row for row in normalised if row["x"] != "Missing"]
    if threshold_value <= 0 or len(normalised) < 3:
        return [glm_group_mapping_row(row, row) for row in [*normalised, *missing_rows]]
    if x_kind in {"integer", "numeric", "date", "datetime", "quantile"}:
        ordered = sorted(normalised, key=lambda r: (r.get("x_sort") is None, r.get("x_sort")))
        low: list[dict[str, Any]] = []
        high: list[dict[str, Any]] = []
        cumulative = 0.0
        for row in ordered:
            volume = float(row.get("volume") or 0)
            if cumulative + volume <= threshold_value:
                low.append(row)
                cumulative += volume
            else:
                break
        cumulative = 0.0
        for row in reversed(ordered[len(low) :]):
            volume = float(row.get("volume") or 0)
            if cumulative + volume <= threshold_value:
                high.append(row)
                cumulative += volume
            else:
                break
        high = list(reversed(high))
        middle = ordered[len(low) : len(ordered) - len(high) if high else len(ordered)]
        mapping: list[dict[str, Any]] = []
        mapping.extend(glm_tail_mapping_rows(low, "Low tail") if len(low) > 1 else [glm_group_mapping_row(row, row) for row in low])
        mapping.extend(glm_group_mapping_row(row, row) for row in middle)
        mapping.extend(glm_tail_mapping_rows(high, "High tail") if len(high) > 1 else [glm_group_mapping_row(row, row) for row in high])
        mapping.extend(glm_group_mapping_row(row, row) for row in missing_rows)
        return mapping
    rare = [row for row in normalised if float(row.get("volume") or 0) <= threshold_value]
    common = [row for row in normalised if float(row.get("volume") or 0) > threshold_value]
    mapping = [glm_group_mapping_row(row, row) for row in common]
    if len(rare) > 1:
        mapping.extend(glm_tail_mapping_rows(rare, "Other"))
    else:
        mapping.extend(glm_group_mapping_row(row, row) for row in rare)
    return mapping


def parse_group_threshold(value: str, total_volume: float) -> float:
    raw = value.strip().lower()
    if raw in {"", "0", "none", "-"}:
        return 0
    if raw.endswith("%"):
        parsed = json_number(raw[:-1])
        return total_volume * float(parsed) / 100 if parsed else 0
    return float(json_number(raw) or 0)


def glm_group_mapping_row(source: dict[str, Any], final: dict[str, Any], *, label: str | None = None, is_tail: bool | None = None) -> dict[str, Any]:
    return {
        "source_x": source.get("x"),
        "final_x": label if label is not None else final.get("x"),
        "final_x_sort": final.get("x_sort"),
        "final_original_order": final.get("original_order"),
        "final_is_tail": bool(final.get("is_tail")) if is_tail is None else is_tail,
    }


def glm_tail_mapping_rows(rows: list[dict[str, Any]], label: str) -> list[dict[str, Any]]:
    if not rows:
        return []
    final = {
        "x": label,
        "x_sort": rows[0].get("x_sort"),
        "original_order": min(int(row.get("original_order") or 0) for row in rows),
        "is_tail": True,
    }
    return [glm_group_mapping_row(row, final, label=label, is_tail=True) for row in rows]


def clean_numeric_labels(rows: list[dict[str, Any]], band_width: Any) -> None:
    from py_lucidum.tools.line_bar.query import clean_partial_numeric_labels

    clean_partial_numeric_labels(rows, band_width)


__all__ = ["build_glm_partial_dependence_overlay", "empty_glm_partial_dependence_warning"]
