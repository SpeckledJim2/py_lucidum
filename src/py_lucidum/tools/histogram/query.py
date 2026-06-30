from __future__ import annotations

import math
from typing import Any

from py_lucidum.core import ColumnInfo, Dataset, is_numeric_kind, json_number, normalise_denominator, quote_ident


AUTO_MIN_BINS = 10
AUTO_MAX_BINS = 200
MAX_EXPLICIT_BINS = 10_000
SAMPLE_LIMIT = 100_000
SAMPLE_SEED = 2026
PERCENTILES = (
    ("0.1st percentile", 0.001),
    ("0.5th percentile", 0.005),
    ("1st percentile", 0.01),
    ("5th percentile", 0.05),
    ("10th percentile", 0.10),
    ("25th percentile", 0.25),
    ("Median", 0.50),
    ("75th percentile", 0.75),
    ("90th percentile", 0.90),
    ("95th percentile", 0.95),
    ("99th percentile", 0.99),
    ("99.5th percentile", 0.995),
    ("99.9th percentile", 0.999),
)
PERCENTILE_VALUES_SQL = "[" + ", ".join(str(percentile) for _, percentile in PERCENTILES) + "]"


def histogram(dataset: Dataset, request: dict[str, Any]) -> dict[str, Any]:
    with dataset.lock:
        source_id = dataset.normalise_source(request.get("source"))
        relation = dataset.relation_sql_for_source(source_id)
        columns = dataset.column_map_for_source(source_id)
        actual = normalise_actual(request, columns)
        denominator = normalise_denominator(request.get("denominator", request.get("weight")), columns)
        distribution = normalise_distribution(request.get("distribution"))
        y_axis = normalise_y_axis(request.get("yAxis"))
        log_scale = normalise_log_scale(request.get("logScale"))
        sample_mode = normalise_sample_mode(request.get("sampleMode"))
        filter_sql = dataset.normalise_filter_for_relation(request.get("filter"), relation)

        row_count = dataset.row_count_for_source(source_id)
        counts = validity_counts(dataset, relation, actual, denominator, filter_sql, log_scale)
        valid_count = int(counts.get("valid_count") or 0)
        filtered_row_count = int(counts.get("filtered_count") or 0)
        sample_values = should_sample_values(sample_mode, valid_count)
        stats, extent, stats_sampled_count = stats_and_extent_rows(
            dataset,
            relation,
            actual,
            denominator,
            filter_sql,
            log_scale,
            counts,
            sample_values,
        )
        bins_count = stats_sampled_count if sample_values else valid_count
        bins_requested = normalise_bins(request.get("bins"), bins_count)
        warnings = histogram_warnings(denominator, counts, log_scale, sample_values, stats_sampled_count)

        if valid_count <= 0:
            warnings.append("No valid histogram values were found after filtering.")
            rows: list[dict[str, Any]] = []
            sampled_valid_count = 0
            bins_used = 0
            binning = continuous_binning_metadata(actual)
        else:
            if extent is None:
                rows = []
                sampled_valid_count = 0
                bins_used = 0
                binning = continuous_binning_metadata(actual)
            else:
                integer_plan = integer_bin_plan(actual, denominator, log_scale, extent, bins_requested)
                if integer_plan:
                    bins_used = integer_plan["bins"]
                    binning = integer_binning_metadata(actual, integer_plan)
                    rows, sampled_valid_count = integer_histogram_rows(
                        dataset,
                        relation,
                        actual,
                        denominator,
                        filter_sql,
                        log_scale,
                        distribution,
                        y_axis,
                        integer_plan,
                        sample_values,
                    )
                else:
                    bins_used = bin_count_for_extent(bins_requested, extent)
                    binning = continuous_binning_metadata(actual, extent)
                    rows, sampled_valid_count = histogram_rows(
                        dataset,
                        relation,
                        actual,
                        denominator,
                        filter_sql,
                        log_scale,
                        distribution,
                        y_axis,
                        bins_used,
                        extent,
                        sample_values,
                    )

        return {
            "source": source_id,
            "actual": actual["numerator"],
            "filter": filter_sql,
            "row_count": row_count,
            "filtered_row_count": filtered_row_count,
            "valid_count": valid_count,
            "sampled_valid_count": sampled_valid_count,
            "bins": bins_used,
            "bins_requested": request.get("bins", "auto") or "auto",
            "distribution": distribution,
            "y_axis": y_axis,
            "log_scale": log_scale,
            "sample_mode": sample_mode,
            "stats_exact": not sample_values,
            "stats_sampled_count": stats_sampled_count,
            "binning": binning,
            "response": {
                "label": actual["label"],
                "numerator": actual["numerator"],
                "value_column": "actual" if denominator.get("column") is None else "actual_per_weight",
            },
            "denominator": {
                "column": denominator["column"],
                "label": denominator["label"],
                "bar_label": denominator["bar_label"],
                "value": json_number(counts.get("weight_sum")),
                "missing_response_rows": json_number(counts.get("missing_actual_count")),
                "missing_weight_rows": json_number(counts.get("missing_weight_count")),
                "zero_weight_rows": json_number(counts.get("zero_weight_count")),
                "negative_weight_rows": json_number(counts.get("negative_weight_count")),
            },
            "stats": stats,
            "rows": rows,
            "warnings": warnings,
        }


def normalise_actual(request: dict[str, Any], columns: dict[str, ColumnInfo]) -> dict[str, str]:
    numerator = str(request.get("actual") or request.get("numerator") or "").strip()
    column = columns.get(numerator)
    if not numerator or column is None or not is_numeric_kind(column.kind):
        raise ValueError("Choose a valid numeric Actual column")
    return {"label": str(request.get("label") or numerator), "numerator": numerator, "kind": column.kind}


def normalise_distribution(value: Any) -> str:
    raw = str(value or "incremental").strip().lower()
    return "cumulative" if raw in {"cumulative", "cum"} else "incremental"


def normalise_y_axis(value: Any) -> str:
    raw = str(value or "sum").strip().lower()
    return "probability" if raw in {"probability", "prob", "percent", "percentage"} else "sum"


def normalise_log_scale(value: Any) -> str:
    raw = str(value or "none").strip().lower()
    aliases = {"": "none", "-": "none", "off": "none", "x axis": "x", "y axis": "y", "both axes": "both"}
    raw = aliases.get(raw, raw)
    return raw if raw in {"none", "x", "y", "both"} else "none"


def normalise_sample_mode(value: Any) -> str:
    raw = str(value or "100k").strip().lower().replace(",", "")
    if raw in {"all", "use all", "full"}:
        return "all"
    return "100k"


def normalise_bins(value: Any, valid_count: int) -> int:
    raw = str(value if value is not None else "auto").strip().lower().replace(",", "")
    if raw in {"", "auto"}:
        return max(AUTO_MIN_BINS, min(AUTO_MAX_BINS, int(round(math.sqrt(max(valid_count, 1))))))
    try:
        parsed = int(float(raw))
    except ValueError:
        return max(AUTO_MIN_BINS, min(AUTO_MAX_BINS, int(round(math.sqrt(max(valid_count, 1))))))
    return max(1, min(MAX_EXPLICIT_BINS, parsed))


def should_sample_values(sample_mode: str, valid_count: int) -> bool:
    return sample_mode == "100k" and valid_count > SAMPLE_LIMIT


def bin_count_for_extent(requested: int, extent: dict[str, Any]) -> int:
    minimum = json_number(extent.get("bin_min"))
    maximum = json_number(extent.get("bin_max"))
    if minimum is None or maximum is None:
        return 0
    if minimum == maximum:
        return 1
    return max(1, requested)


def integer_bin_plan(
    actual: dict[str, str],
    denominator: dict[str, str | None],
    log_scale: str,
    extent: dict[str, Any],
    requested: int,
) -> dict[str, int] | None:
    if denominator.get("column") or log_scale in {"x", "both"}:
        return None
    value_min = json_number(extent.get("value_min"))
    value_max = json_number(extent.get("value_max"))
    if value_min is None or value_max is None:
        return None
    if not float(value_min).is_integer() or not float(value_max).is_integer():
        return None
    non_integral_count = int(extent.get("non_integral_count") or 0)
    if actual.get("kind") != "integer" and non_integral_count:
        return None
    min_int = int(round(float(value_min)))
    max_int = int(round(float(value_max)))
    if min_int > max_int:
        return None
    if max(abs(min_int), abs(max_int)) >= 9_007_199_254_740_992:
        return None
    level_count = max_int - min_int + 1
    requested_count = max(1, int(requested))
    step = max(1, math.ceil(level_count / requested_count))
    bin_count = max(1, math.ceil(level_count / step))
    return {"min": min_int, "max": max_int, "step": step, "bins": bin_count}


def continuous_binning_metadata(actual: dict[str, str], extent: dict[str, Any] | None = None) -> dict[str, Any]:
    return {
        "mode": "continuous",
        "kind": actual.get("kind") or "numeric",
        "min": json_number((extent or {}).get("value_min")),
        "max": json_number((extent or {}).get("value_max")),
        "step": None,
    }


def integer_binning_metadata(actual: dict[str, str], plan: dict[str, int]) -> dict[str, Any]:
    return {
        "mode": "integer",
        "kind": actual.get("kind") or "integer",
        "min": int(plan["min"]),
        "max": int(plan["max"]),
        "step": int(plan["step"]),
    }


def actual_sql(actual: dict[str, str]) -> str:
    return f"TRY_CAST({quote_ident(actual['numerator'])} AS DOUBLE)"


def weight_sql(denominator: dict[str, str | None]) -> str:
    column = denominator.get("column")
    if column:
        return f"TRY_CAST({quote_ident(str(column))} AS DOUBLE)"
    return "1.0"


def value_sql(actual: dict[str, str], denominator: dict[str, str | None]) -> str:
    actual_expr = actual_sql(actual)
    column = denominator.get("column")
    if column:
        return f"{actual_expr} / NULLIF({weight_sql(denominator)}, 0)"
    return actual_expr


def valid_condition(actual: dict[str, str], denominator: dict[str, str | None], log_scale: str = "none") -> str:
    value_expr = value_sql(actual, denominator)
    checks = [f"{actual_sql(actual)} IS NOT NULL"]
    column = denominator.get("column")
    if column:
        weight_expr = weight_sql(denominator)
        checks.extend([f"{weight_expr} IS NOT NULL", f"{weight_expr} > 0"])
    if log_scale in {"x", "both"}:
        checks.append(f"{value_expr} > 0")
    return " AND ".join(checks)


def value_ctes_sql(
    relation: str,
    actual: dict[str, str],
    denominator: dict[str, str | None],
    filter_sql: str,
    log_scale: str,
    sample_values: bool,
) -> str:
    value_expr = value_sql(actual, denominator)
    bin_value = f"LOG10({value_expr})" if log_scale in {"x", "both"} else value_expr
    weight_expr = weight_sql(denominator)
    valid = valid_condition(actual, denominator, log_scale)
    where_sql = f"\n  WHERE ({filter_sql})" if filter_sql else ""
    sampled_cte = (
        f""",
sampled AS (
  SELECT * FROM valid_values
  USING SAMPLE reservoir({SAMPLE_LIMIT} ROWS) REPEATABLE ({SAMPLE_SEED})
)"""
        if sample_values
        else ""
    )
    return f"""filtered AS (
  SELECT * FROM {relation}{where_sql}
),
valid_values AS (
  SELECT
    {value_expr} AS value,
    {bin_value} AS bin_value,
    {weight_expr} AS weight_value
  FROM filtered
  WHERE {valid}
){sampled_cte}"""


def value_source_name(sample_values: bool) -> str:
    return "sampled" if sample_values else "valid_values"


def validity_counts(
    dataset: Dataset,
    relation: str,
    actual: dict[str, str],
    denominator: dict[str, str | None],
    filter_sql: str,
    log_scale: str,
) -> dict[str, Any]:
    actual_expr = actual_sql(actual)
    weight_expr = weight_sql(denominator)
    value_expr = value_sql(actual, denominator)
    valid = valid_condition(actual, denominator, log_scale)
    where_sql = f"\n  WHERE ({filter_sql})" if filter_sql else ""
    if denominator.get("column"):
        weight_counts = f"""
    SUM(CASE WHEN {weight_expr} IS NULL THEN 1 ELSE 0 END) AS missing_weight_count,
    SUM(CASE WHEN {weight_expr} = 0 THEN 1 ELSE 0 END) AS zero_weight_count,
    SUM(CASE WHEN {weight_expr} < 0 THEN 1 ELSE 0 END) AS negative_weight_count,"""
    else:
        weight_counts = """
    0 AS missing_weight_count,
    0 AS zero_weight_count,
    0 AS negative_weight_count,"""
    nonpositive_sql = (
        f"SUM(CASE WHEN {actual_expr} IS NOT NULL AND {value_expr} <= 0 THEN 1 ELSE 0 END)"
        if log_scale in {"x", "both"}
        else "0"
    )
    sql = f"""
WITH filtered AS (
  SELECT * FROM {relation}{where_sql}
)
SELECT
    COUNT(*) AS filtered_count,
    SUM(CASE WHEN {actual_expr} IS NULL THEN 1 ELSE 0 END) AS missing_actual_count,
    {weight_counts}
    SUM(CASE WHEN {valid} THEN 1 ELSE 0 END) AS valid_count,
    SUM(CASE WHEN {valid} AND {value_expr} = 0 THEN 1 ELSE 0 END) AS zero_count,
    SUM(CASE WHEN {valid} THEN {weight_expr} ELSE NULL END) AS weight_sum,
    {nonpositive_sql} AS nonpositive_count
FROM filtered
"""
    cursor = dataset.con.execute(sql)
    fetched = cursor.fetchone()
    return dict(zip([d[0] for d in cursor.description], fetched or []))


def stats_and_extent_rows(
    dataset: Dataset,
    relation: str,
    actual: dict[str, str],
    denominator: dict[str, str | None],
    filter_sql: str,
    log_scale: str,
    counts: dict[str, Any],
    sample_values: bool,
) -> tuple[list[dict[str, Any]], dict[str, Any] | None, int]:
    value_source = value_source_name(sample_values)
    sql = f"""
WITH {value_ctes_sql(relation, actual, denominator, filter_sql, log_scale, sample_values)}
SELECT
    COUNT(*) AS numeric_count,
    AVG(value) AS mean,
    SUM(value * weight_value) / NULLIF(SUM(weight_value), 0) AS weighted_mean,
    STDDEV_SAMP(value) AS stddev,
    MIN(value) AS minimum,
    MAX(value) AS maximum,
    quantile_cont(value, {PERCENTILE_VALUES_SQL}) AS percentiles,
    MIN(bin_value) AS bin_min,
    MAX(bin_value) AS bin_max,
    SUM(CASE WHEN value = FLOOR(value) THEN 0 ELSE 1 END) AS non_integral_count
FROM {value_source}
"""
    cursor = dataset.con.execute(sql)
    row = dict(zip([d[0] for d in cursor.description], cursor.fetchone() or []))
    stats_sampled_count = int(row.get("numeric_count") or 0)
    extent = {
        "value_min": row.get("minimum"),
        "value_max": row.get("maximum"),
        "bin_min": row.get("bin_min"),
        "bin_max": row.get("bin_max"),
        "non_integral_count": row.get("non_integral_count"),
    }
    if json_number(extent.get("bin_min")) is None or json_number(extent.get("bin_max")) is None:
        extent = None
    return stats_rows_from_summary(row, counts), extent, stats_sampled_count


def stats_rows_from_summary(row: dict[str, Any], counts: dict[str, Any]) -> list[dict[str, Any]]:
    percentiles = row.get("percentiles")
    if not isinstance(percentiles, (list, tuple)):
        percentiles = []
    valid_count = int(counts.get("valid_count") or 0)
    stats = [
        {"statistic": "Numeric count", "value": valid_count},
        {"statistic": "NA count", "value": max(0, int(counts.get("filtered_count") or 0) - valid_count)},
        {"statistic": "Zero count", "value": json_number(counts.get("zero_count")) or 0},
        {"statistic": "Mean", "value": json_number(row.get("mean"))},
        {"statistic": "Weighted mean", "value": json_number(row.get("weighted_mean"))},
        {"statistic": "Std deviation", "value": json_number(row.get("stddev"))},
        {"statistic": "Minimum", "value": json_number(row.get("minimum"))},
    ]
    stats.extend(
        {"statistic": label, "value": json_number(percentiles[index] if index < len(percentiles) else None)}
        for index, (label, _) in enumerate(PERCENTILES)
    )
    stats.append({"statistic": "Maximum", "value": json_number(row.get("maximum"))})
    stats.append({"statistic": "Weight sum", "value": json_number(counts.get("weight_sum"))})
    return stats


def integer_histogram_rows(
    dataset: Dataset,
    relation: str,
    actual: dict[str, str],
    denominator: dict[str, str | None],
    filter_sql: str,
    log_scale: str,
    distribution: str,
    y_axis: str,
    plan: dict[str, int],
    sample_values: bool,
) -> tuple[list[dict[str, Any]], int]:
    value_source = value_source_name(sample_values)
    min_int = int(plan["min"])
    max_int = int(plan["max"])
    step = max(1, int(plan["step"]))
    bins = max(1, int(plan["bins"]))
    sql = f"""
WITH {value_ctes_sql(relation, actual, denominator, filter_sql, log_scale, sample_values)},
params AS (
  SELECT
    {min_int}::BIGINT AS min_level,
    {max_int}::BIGINT AS max_level,
    {step}::BIGINT AS step,
    {bins}::INTEGER AS bin_count
),
indexed AS (
  SELECT
    LEAST(
      params.bin_count - 1,
      GREATEST(0, FLOOR((source_values.value - params.min_level) / params.step)::INTEGER)
    ) AS bin_index,
    source_values.weight_value
  FROM {value_source} source_values, params
),
agg AS (
  SELECT
    bin_index,
    COUNT(*) AS row_count,
    COALESCE(SUM(weight_value), 0) AS volume
  FROM indexed
  GROUP BY bin_index
),
bin_numbers AS (
  SELECT bin_index FROM range(0, {bins}) AS generated_bins(bin_index)
),
bin_bounds AS (
  SELECT
    bins.bin_index,
    params.min_level + bins.bin_index * params.step AS label_lower,
    LEAST(params.max_level, params.min_level + (bins.bin_index + 1) * params.step - 1) AS label_upper
  FROM bin_numbers bins
  CROSS JOIN params
),
bin_rows AS (
  SELECT
    bounds.bin_index,
    bounds.label_lower - 0.5 AS bin_lower,
    bounds.label_upper + 0.5 AS bin_upper,
    bounds.label_lower AS bin_label_lower,
    bounds.label_upper AS bin_label_upper,
    COALESCE(agg.row_count, 0) AS row_count,
    COALESCE(agg.volume, 0) AS volume
  FROM bin_bounds bounds
  LEFT JOIN agg ON bounds.bin_index = agg.bin_index
),
with_totals AS (
  SELECT
    *,
    SUM(volume) OVER () AS total_volume,
    SUM(row_count) OVER () AS sampled_valid_count
  FROM bin_rows
),
cumulative AS (
  SELECT
    *,
    SUM(volume) OVER (ORDER BY bin_index ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) AS cumulative_volume
  FROM with_totals
)
SELECT
    bin_index,
    bin_lower,
    bin_upper,
    (bin_lower + bin_upper) / 2 AS bin_mid,
    bin_label_lower,
    bin_label_upper,
    row_count,
    volume,
    volume / NULLIF(total_volume, 0) AS probability,
    cumulative_volume,
    cumulative_volume / NULLIF(total_volume, 0) AS cumulative_probability,
    sampled_valid_count
FROM cumulative
ORDER BY bin_index
"""
    rows = normalise_histogram_rows(dataset, sql, distribution, y_axis)
    sampled = int(rows[0].pop("sampled_valid_count", 0) or 0) if rows else 0
    for row in rows[1:]:
        row.pop("sampled_valid_count", None)
    return rows, sampled


def histogram_rows(
    dataset: Dataset,
    relation: str,
    actual: dict[str, str],
    denominator: dict[str, str | None],
    filter_sql: str,
    log_scale: str,
    distribution: str,
    y_axis: str,
    bins: int,
    extent: dict[str, Any],
    sample_values: bool,
) -> tuple[list[dict[str, Any]], int]:
    value_source = value_source_name(sample_values)
    bin_min = float(extent["bin_min"])
    bin_max = float(extent["bin_max"])
    value_min = float(extent["value_min"])
    value_max = float(extent["value_max"])
    if bins <= 1 or bin_min == bin_max:
        return single_bin_row(dataset, relation, actual, denominator, filter_sql, log_scale, distribution, y_axis, extent, sample_values)
    bin_width = (bin_max - bin_min) / bins
    lower_expr = "POWER(10, params.bin_min + bins.bin_index * params.bin_width)" if log_scale in {"x", "both"} else "params.bin_min + bins.bin_index * params.bin_width"
    upper_expr = "POWER(10, params.bin_min + (bins.bin_index + 1) * params.bin_width)" if log_scale in {"x", "both"} else "params.bin_min + (bins.bin_index + 1) * params.bin_width"
    mid_expr = "SQRT(bin_lower * bin_upper)" if log_scale in {"x", "both"} else "(bin_lower + bin_upper) / 2"
    sql = f"""
WITH {value_ctes_sql(relation, actual, denominator, filter_sql, log_scale, sample_values)},
params AS (
  SELECT
    {bin_min}::DOUBLE AS bin_min,
    {bin_max}::DOUBLE AS bin_max,
    {bin_width}::DOUBLE AS bin_width,
    {bins}::INTEGER AS bin_count
),
indexed AS (
  SELECT
    CASE
      WHEN source_values.bin_value = params.bin_max THEN params.bin_count - 1
      ELSE LEAST(params.bin_count - 1, GREATEST(0, FLOOR((source_values.bin_value - params.bin_min) / params.bin_width)::INTEGER))
    END AS bin_index,
    source_values.value,
    source_values.weight_value
  FROM {value_source} source_values, params
),
agg AS (
  SELECT
    bin_index,
    COUNT(*) AS row_count,
    COALESCE(SUM(weight_value), 0) AS volume
  FROM indexed
  GROUP BY bin_index
),
bin_numbers AS (
  SELECT bin_index FROM range(0, {bins}) AS generated_bins(bin_index)
),
bin_rows AS (
  SELECT
    bins.bin_index,
    {lower_expr} AS bin_lower,
    CASE WHEN bins.bin_index = params.bin_count - 1 THEN {value_max}::DOUBLE ELSE {upper_expr} END AS bin_upper,
    COALESCE(agg.row_count, 0) AS row_count,
    COALESCE(agg.volume, 0) AS volume
  FROM bin_numbers bins
  CROSS JOIN params
  LEFT JOIN agg ON bins.bin_index = agg.bin_index
),
with_totals AS (
  SELECT
    *,
    SUM(volume) OVER () AS total_volume,
    SUM(row_count) OVER () AS sampled_valid_count
  FROM bin_rows
),
cumulative AS (
  SELECT
    *,
    SUM(volume) OVER (ORDER BY bin_index ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) AS cumulative_volume
  FROM with_totals
)
SELECT
    bin_index,
    CASE WHEN bin_index = 0 THEN {value_min}::DOUBLE ELSE bin_lower END AS bin_lower,
    bin_upper,
    {mid_expr} AS bin_mid,
    row_count,
    volume,
    volume / NULLIF(total_volume, 0) AS probability,
    cumulative_volume,
    cumulative_volume / NULLIF(total_volume, 0) AS cumulative_probability,
    sampled_valid_count
FROM cumulative
ORDER BY bin_index
"""
    rows = normalise_histogram_rows(dataset, sql, distribution, y_axis)
    sampled = int(rows[0].pop("sampled_valid_count", 0) or 0) if rows else 0
    for row in rows[1:]:
        row.pop("sampled_valid_count", None)
    return rows, sampled


def single_bin_row(
    dataset: Dataset,
    relation: str,
    actual: dict[str, str],
    denominator: dict[str, str | None],
    filter_sql: str,
    log_scale: str,
    distribution: str,
    y_axis: str,
    extent: dict[str, Any],
    sample_values: bool,
) -> tuple[list[dict[str, Any]], int]:
    value_source = value_source_name(sample_values)
    value = float(extent["value_min"])
    pad = max(abs(value) * 0.005, 0.5)
    if log_scale in {"x", "both"}:
        lower = value / 1.005 if value > 0 else value
        upper = value * 1.005 if value > 0 else value
    else:
        lower = value - pad
        upper = value + pad
    sql = f"""
WITH {value_ctes_sql(relation, actual, denominator, filter_sql, log_scale, sample_values)},
agg AS (
  SELECT COUNT(*) AS row_count, COALESCE(SUM(weight_value), 0) AS volume FROM {value_source}
)
SELECT
    0 AS bin_index,
    {lower}::DOUBLE AS bin_lower,
    {upper}::DOUBLE AS bin_upper,
    {value}::DOUBLE AS bin_mid,
    row_count,
    volume,
    1.0 AS probability,
    volume AS cumulative_volume,
    1.0 AS cumulative_probability,
    row_count AS sampled_valid_count
FROM agg
"""
    rows = normalise_histogram_rows(dataset, sql, distribution, y_axis)
    sampled = int(rows[0].pop("sampled_valid_count", 0) or 0) if rows else 0
    return rows, sampled


def normalise_histogram_rows(dataset: Dataset, sql: str, distribution: str, y_axis: str) -> list[dict[str, Any]]:
    cursor = dataset.con.execute(sql)
    raw_rows = [dict(zip([d[0] for d in cursor.description], row)) for row in cursor.fetchall()]
    rows: list[dict[str, Any]] = []
    for raw in raw_rows:
        height_key = "cumulative_probability" if distribution == "cumulative" and y_axis == "probability" else (
            "cumulative_volume" if distribution == "cumulative" else ("probability" if y_axis == "probability" else "volume")
        )
        lower = json_number(raw.get("bin_lower"))
        upper = json_number(raw.get("bin_upper"))
        rows.append({
            "bin_index": int(raw.get("bin_index") or 0),
            "bin_lower": lower,
            "bin_upper": upper,
            "bin_mid": json_number(raw.get("bin_mid")),
            "bin_label": histogram_bin_label(raw, lower, upper),
            "row_count": json_number(raw.get("row_count")) or 0,
            "volume": json_number(raw.get("volume")) or 0,
            "probability": json_number(raw.get("probability")) or 0,
            "cumulative_volume": json_number(raw.get("cumulative_volume")) or 0,
            "cumulative_probability": json_number(raw.get("cumulative_probability")) or 0,
            "height": json_number(raw.get(height_key)) or 0,
            "sampled_valid_count": json_number(raw.get("sampled_valid_count")) or 0,
        })
    return rows


def histogram_bin_label(raw: dict[str, Any], lower: Any, upper: Any) -> str:
    label_lower = json_number(raw.get("bin_label_lower"))
    label_upper = json_number(raw.get("bin_label_upper"))
    if label_lower is None or label_upper is None:
        return bin_label(lower, upper)
    return integer_bin_label(label_lower, label_upper)


def integer_bin_label(lower: Any, upper: Any) -> str:
    if lower == upper:
        return format_bound(lower)
    return f"{format_bound(lower)} to {format_bound(upper)}"


def bin_label(lower: Any, upper: Any) -> str:
    return f"{format_bound(lower)} to {format_bound(upper)}"


def format_bound(value: Any) -> str:
    number = json_number(value)
    if number is None:
        return ""
    if isinstance(number, int):
        return f"{number:,}"
    abs_value = abs(float(number))
    if abs_value != 0 and (abs_value < 0.001 or abs_value >= 1_000_000):
        return f"{float(number):.4g}"
    text = f"{float(number):,.6f}".rstrip("0").rstrip(".")
    return "0" if text == "-0" else text


def histogram_warnings(
    denominator: dict[str, str | None],
    counts: dict[str, Any],
    log_scale: str,
    sample_values: bool,
    stats_sampled_count: int,
) -> list[str]:
    warnings: list[str] = []
    missing_actual = int(counts.get("missing_actual_count") or 0)
    if missing_actual:
        warnings.append(f"{missing_actual:,} rows excluded because Actual was missing.")
    if denominator.get("column"):
        missing_weight = int(counts.get("missing_weight_count") or 0)
        zero_weight = int(counts.get("zero_weight_count") or 0)
        negative_weight = int(counts.get("negative_weight_count") or 0)
        label = denominator["label"]
        if missing_weight:
            warnings.append(f"{missing_weight:,} rows excluded because {label} was missing.")
        if zero_weight:
            warnings.append(f"{zero_weight:,} rows excluded because {label} was zero.")
        if negative_weight:
            warnings.append(f"{negative_weight:,} rows excluded because {label} was negative.")
    nonpositive = int(counts.get("nonpositive_count") or 0)
    if log_scale in {"x", "both"} and nonpositive:
        warnings.append(f"{nonpositive:,} nonpositive values excluded for log x-scale.")
    if sample_values:
        warnings.append(
            f"Histogram bars and distribution statistics use a deterministic {stats_sampled_count:,}-row sample; "
            "row counts and exclusions are exact. Use all for exact distribution statistics."
        )
    return warnings


__all__ = ["histogram"]
