from __future__ import annotations

import math
from typing import Any

from py_lucidum.core import ColumnInfo, Dataset, is_numeric_kind, json_number, parse_positive_float, quote_ident


def chart(dataset: Dataset, request: dict[str, Any]) -> dict[str, Any]:
    with dataset.lock:
        columns = dataset.column_map()
        x_col = str(request.get("x") or "")
        if x_col not in columns:
            raise ValueError("Choose a valid x-axis feature")

        filter_sql = dataset.normalise_filter(request.get("filter"))
        responses = normalise_responses(request.get("responses"), columns)
        x_info = columns[x_col]
        x_sql = build_x_sql(
            x_col=x_col,
            kind=x_info.kind,
            band_width=request.get("bandWidth"),
            date_bucket=request.get("dateBucket"),
        )
        sigma_multiplier = float(request.get("sigma") or 0)
        include_sigma = sigma_multiplier > 0 and len(responses) >= 2
        row_count = dataset.row_count()
        filtered_row_count = dataset.filtered_row_count(filter_sql)
        sql = build_chart_sql(dataset.relation_sql(), x_sql, responses, include_sigma, filter_sql)
        raw_rows = [
            dict(zip([d[0] for d in dataset.con.description], row))
            for row in dataset.con.execute(sql).fetchall()
        ]

        grouped_rows = apply_low_weight_grouping(
            rows=raw_rows,
            responses=responses,
            x_kind=x_info.kind,
            threshold=str(request.get("lowGroup") or "0"),
        )
        sorted_rows = sort_rows(grouped_rows, x_info.kind, str(request.get("sort") or "alpha"))
        max_groups = int(request.get("maxGroups") or 10000)
        if len(sorted_rows) > max_groups:
            sorted_rows = sorted_rows[:max_groups]

        transform = str(request.get("transform") or "none")
        warnings: list[str] = []
        display_rows = apply_transform(sorted_rows, responses, transform, sigma_multiplier, warnings)

        return {
            "x": x_col,
            "x_kind": x_info.kind,
            "row_count": row_count,
            "filtered_row_count": filtered_row_count,
            "filter": filter_sql,
            "responses": [
                {"label": r["label"], "numerator": r["numerator"], "denominator": r["denominator"]}
                for r in responses
            ],
            "rows": display_rows,
            "warnings": warnings,
        }


def normalise_responses(raw: Any, columns: dict[str, ColumnInfo]) -> list[dict[str, str | None]]:
    responses: list[dict[str, str | None]] = []
    if not isinstance(raw, list):
        return responses
    for item in raw:
        if not isinstance(item, dict):
            continue
        numerator = item.get("numerator")
        if not numerator or numerator not in columns:
            continue
        denominator = item.get("denominator") or None
        if denominator == "__none__":
            denominator = None
        if denominator not in (None, "__count__") and denominator not in columns:
            denominator = None
        label = item.get("label") or str(numerator)
        responses.append({"label": str(label), "numerator": str(numerator), "denominator": denominator})
    return responses[:2]


def build_x_sql(x_col: str, kind: str, band_width: Any, date_bucket: Any) -> dict[str, str]:
    col = quote_ident(x_col)
    if is_numeric_kind(kind):
        raw = f"TRY_CAST({col} AS DOUBLE)"
        width = parse_positive_float(band_width)
        if width:
            key = f"FLOOR({raw} / {width}) * {width}"
        else:
            key = raw
        if kind == "integer":
            label = f"CASE WHEN {key} IS NULL THEN '(missing)' ELSE CAST(TRY_CAST({key} AS BIGINT) AS VARCHAR) END"
        else:
            label = f"CASE WHEN {key} IS NULL THEN '(missing)' ELSE CAST({key} AS VARCHAR) END"
        return {
            "key": key,
            "label": label,
            "sort": key,
        }
    if kind in ("date", "datetime"):
        bucket = str(date_bucket or "none").lower()
        if bucket in {"hour", "day", "week", "month", "year"}:
            key = f"DATE_TRUNC('{bucket}', {col})"
        else:
            key = col
        return {
            "key": key,
            "label": f"CASE WHEN {key} IS NULL THEN '(missing)' ELSE CAST({key} AS VARCHAR) END",
            "sort": key,
        }
    key = f"COALESCE(CAST({col} AS VARCHAR), '(missing)')"
    return {"key": key, "label": key, "sort": key}


def response_parts(response: dict[str, str | None], index: int) -> tuple[str, str, str]:
    numerator = f"TRY_CAST({quote_ident(str(response['numerator']))} AS DOUBLE)"
    denominator = response["denominator"]
    num_alias = f"resp{index}_num"
    den_alias = f"resp{index}_den"
    value_alias = f"resp{index}"
    if denominator == "__count__":
        num_expr = f"SUM({numerator})"
        den_expr = "COUNT(*)"
    elif denominator:
        den_col = f"TRY_CAST({quote_ident(str(denominator))} AS DOUBLE)"
        num_expr = f"SUM({numerator})"
        den_expr = f"SUM({den_col})"
    else:
        num_expr = f"SUM({numerator})"
        den_expr = f"COUNT({numerator})"
    value_expr = f"{num_alias} / NULLIF({den_alias}, 0)"
    return (
        f"{num_expr} AS {num_alias}",
        f"{den_expr} AS {den_alias}",
        f"{value_expr} AS {value_alias}",
    )


def build_chart_sql(
    relation: str,
    x_sql: dict[str, str],
    responses: list[dict[str, str | None]],
    include_sigma: bool,
    filter_sql: str = "",
) -> str:
    metric_selects: list[str] = []
    value_selects: list[str] = []
    for index, response in enumerate(responses):
        num_expr, den_expr, value_expr = response_parts(response, index)
        metric_selects.extend([num_expr, den_expr])
        value_selects.append(value_expr)

    metric_sql = ",\n      ".join(metric_selects)
    value_sql = ",\n    ".join(value_selects)
    if metric_sql:
        metric_sql = ",\n      " + metric_sql
    if value_sql:
        value_sql = ",\n    " + value_sql

    sigma_sql = ""
    sigma_join = ""
    sigma_output = ",\n    sigma.sigma_se,\n    sigma.valid_folds"
    if include_sigma:
        fold_metrics: list[str] = []
        fold_values: list[str] = []
        for index, response in enumerate(responses[:2]):
            num_expr, den_expr, value_expr = response_parts(response, index)
            fold_metrics.extend([num_expr, den_expr])
            fold_values.append(value_expr)
        sigma_sql = f""",
folds AS (
  SELECT
    x_key,
    x_label,
    __fold,
    {', '.join(fold_metrics)}
  FROM keyed
  GROUP BY x_key, x_label, __fold
),
fold_values AS (
  SELECT
    x_key,
    x_label,
    __fold,
    {', '.join(fold_values)}
  FROM folds
),
sigma AS (
  SELECT
    x_key,
    x_label,
    STDDEV_SAMP(resp0 - resp1) / SQRT(COUNT(*)) AS sigma_se,
    COUNT(*) AS valid_folds
  FROM fold_values
  WHERE resp0 IS NOT NULL AND resp1 IS NOT NULL
  GROUP BY x_key, x_label
)"""
        sigma_join = "LEFT JOIN sigma ON agg_values.x_key IS NOT DISTINCT FROM sigma.x_key AND agg_values.x_label = sigma.x_label"
    else:
        sigma_output = ",\n    NULL AS sigma_se,\n    NULL AS valid_folds"
        sigma_join = ""

    where_sql = f"\n  WHERE ({filter_sql})" if filter_sql else ""
    return f"""
WITH base AS (
  SELECT ROW_NUMBER() OVER () AS __rownum, * FROM {relation}{where_sql}
),
keyed AS (
  SELECT
    __rownum,
    {x_sql['key']} AS x_key,
    {x_sql['label']} AS x_label,
    {x_sql['sort']} AS x_sort,
    CAST(hash(__rownum) % 20 AS INTEGER) AS __fold,
    *
  FROM base
),
agg AS (
  SELECT
    x_key,
    x_label,
    MIN(x_sort) AS x_sort,
    MIN(__rownum) AS original_order,
    COUNT(*) AS volume
    {metric_sql}
  FROM keyed
  GROUP BY x_key, x_label
),
agg_values AS (
  SELECT
    *{value_sql}
  FROM agg
)
{sigma_sql}
SELECT
    agg_values.*{sigma_output}
FROM agg_values
{sigma_join}
"""


def parse_group_threshold(value: str, total_volume: float) -> float:
    raw = value.strip().lower()
    if raw in {"", "0", "none", "-"}:
        return 0
    if raw.endswith("%"):
        parsed = parse_positive_float(raw[:-1])
        return total_volume * parsed / 100 if parsed else 0
    return parse_positive_float(raw) or 0


def apply_low_weight_grouping(
    rows: list[dict[str, Any]],
    responses: list[dict[str, str | None]],
    x_kind: str,
    threshold: str,
) -> list[dict[str, Any]]:
    total_volume = sum(float(row.get("volume") or 0) for row in rows)
    threshold_value = parse_group_threshold(threshold, total_volume)
    if threshold_value <= 0 or len(rows) < 3:
        return [normalise_row(row, responses) for row in rows]

    normalised = [normalise_row(row, responses) for row in rows]
    if x_kind in {"integer", "numeric", "date", "datetime"}:
        ordered = sorted(normalised, key=lambda r: (r["x_sort"] is None, r["x_sort"]))
        low: list[dict[str, Any]] = []
        high: list[dict[str, Any]] = []
        cumulative = 0.0
        for row in ordered:
            volume = float(row["volume"] or 0)
            if cumulative + volume <= threshold_value:
                low.append(row)
                cumulative += volume
            else:
                break
        cumulative = 0.0
        for row in reversed(ordered[len(low):]):
            volume = float(row["volume"] or 0)
            if cumulative + volume <= threshold_value:
                high.append(row)
                cumulative += volume
            else:
                break
        high = list(reversed(high))
        middle = ordered[len(low): len(ordered) - len(high) if high else len(ordered)]
        result: list[dict[str, Any]] = []
        if len(low) > 1:
            result.append(combine_rows(low, "Low tail", responses, is_tail=True))
        else:
            result.extend(low)
        result.extend(middle)
        if len(high) > 1:
            result.append(combine_rows(high, "High tail", responses, is_tail=True))
        else:
            result.extend(high)
        return result

    rare = [row for row in normalised if float(row["volume"] or 0) <= threshold_value]
    common = [row for row in normalised if float(row["volume"] or 0) > threshold_value]
    if len(rare) > 1:
        common.append(combine_rows(rare, "Other", responses, is_tail=True))
    else:
        common.extend(rare)
    return common


def normalise_row(row: dict[str, Any], responses: list[dict[str, str | None]]) -> dict[str, Any]:
    result = {
        "x": str(row.get("x_label")),
        "x_sort": row.get("x_sort"),
        "original_order": int(row.get("original_order") or 0),
        "volume": int(row.get("volume") or 0),
        "is_tail": False,
        "sigma_se": json_number(row.get("sigma_se")),
        "valid_folds": json_number(row.get("valid_folds")),
    }
    for index, _ in enumerate(responses):
        result[f"resp{index}_num"] = json_number(row.get(f"resp{index}_num"))
        result[f"resp{index}_den"] = json_number(row.get(f"resp{index}_den"))
        result[f"resp{index}"] = json_number(row.get(f"resp{index}"))
    return result


def combine_rows(rows: list[dict[str, Any]], label: str, responses: list[dict[str, str | None]], is_tail: bool) -> dict[str, Any]:
    combined = {
        "x": label,
        "x_sort": rows[0].get("x_sort"),
        "original_order": min(int(row.get("original_order") or 0) for row in rows),
        "volume": sum(int(row.get("volume") or 0) for row in rows),
        "is_tail": is_tail,
        "sigma_se": None,
        "valid_folds": None,
    }
    for index, _ in enumerate(responses):
        num = sum(float(row.get(f"resp{index}_num") or 0) for row in rows)
        den = sum(float(row.get(f"resp{index}_den") or 0) for row in rows)
        combined[f"resp{index}_num"] = json_number(num)
        combined[f"resp{index}_den"] = json_number(den)
        combined[f"resp{index}"] = json_number(num / den) if den else None
    return combined


def sort_rows(rows: list[dict[str, Any]], x_kind: str, sort: str) -> list[dict[str, Any]]:
    if x_kind not in {"categorical"}:
        return sorted(rows, key=lambda r: (r["x_sort"] is None, r["x_sort"]))
    if sort == "volume":
        return sorted(rows, key=lambda r: (not r.get("is_tail"), -(r["volume"] or 0), str(r["x"]).lower()))
    if sort in {"actual", "response"}:
        return sorted(rows, key=lambda r: (r.get("resp0") is None, -(r.get("resp0") or 0), str(r["x"]).lower()))
    if sort == "expected":
        return sorted(rows, key=lambda r: (r.get("resp1") is None, -(r.get("resp1") or 0), str(r["x"]).lower()))
    return sorted(rows, key=lambda r: str(r["x"]).lower())


def apply_transform(
    rows: list[dict[str, Any]],
    responses: list[dict[str, str | None]],
    transform: str,
    sigma_multiplier: float,
    warnings: list[str],
) -> list[dict[str, Any]]:
    averages: dict[int, float | None] = {}
    for index, _ in enumerate(responses):
        num = sum(float(row.get(f"resp{index}_num") or 0) for row in rows)
        den = sum(float(row.get(f"resp{index}_den") or 0) for row in rows)
        averages[index] = num / den if den else None

    display: list[dict[str, Any]] = []
    invalid_count = 0
    for row in rows:
        out = {
            "x": row["x"],
            "volume": row["volume"],
            "is_tail": bool(row.get("is_tail")),
            "valid_folds": row.get("valid_folds"),
        }
        for index, _ in enumerate(responses):
            out[f"resp{index}"] = transform_value(row.get(f"resp{index}"), transform, averages[index])
            if row.get(f"resp{index}") is not None and out[f"resp{index}"] is None:
                invalid_count += 1
        if sigma_multiplier > 0 and len(responses) >= 2 and row.get("sigma_se") is not None and row.get("resp1") is not None:
            se = float(row["sigma_se"])
            expected = float(row["resp1"])
            out["resp1_low"] = transform_value(expected - sigma_multiplier * se, transform, averages[1])
            out["resp1_high"] = transform_value(expected + sigma_multiplier * se, transform, averages[1])
        display.append(out)

    if invalid_count:
        warnings.append(f"{invalid_count} response values could not be shown because they are outside the {transform} transform domain.")
    return display


def transform_value(value: Any, transform: str, average: float | None) -> float | int | None:
    number = json_number(value)
    if number is None:
        return None
    x = float(number)
    try:
        if transform == "log":
            return json_number(math.log(x)) if x > 0 else None
        if transform == "exp":
            return json_number(math.exp(x))
        if transform == "logit":
            return json_number(math.log(x / (1 - x))) if 0 < x < 1 else None
        if transform == "zero":
            return json_number(x - average) if average is not None else None
        if transform == "one":
            return json_number(x / average) if average not in (None, 0) else None
    except (OverflowError, ValueError):
        return None
    return json_number(x)


__all__ = [
    "apply_low_weight_grouping",
    "apply_transform",
    "build_chart_sql",
    "build_x_sql",
    "chart",
    "combine_rows",
    "normalise_responses",
    "normalise_row",
    "parse_group_threshold",
    "response_parts",
    "sort_rows",
    "transform_value",
]
