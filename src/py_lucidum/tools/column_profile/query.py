from __future__ import annotations

import datetime as dt
import math
from decimal import Decimal
from typing import Any

from py_lucidum.core import ColumnInfo, Dataset, is_numeric_kind, json_number, quote_ident


PROFILE_TOP_VALUE_LIMIT = 5
PROFILE_HISTOGRAM_BINS = 10
PROFILE_DETAIL_TOP_VALUE_LIMIT = 2000
PROFILE_DETAIL_HISTOGRAM_BINS = 20
PROFILE_DETAIL_EXACT_LEVEL_BIN_LIMIT = 100


def profile(dataset: Dataset, request: dict[str, Any]) -> dict[str, Any]:
    with dataset.lock:
        columns = dataset.column_map()
        filter_sql = dataset.normalise_filter(request.get("filter"))
        row_count = dataset.row_count()
        filtered_row_count = dataset.filtered_row_count(filter_sql)
        column_profiles = [
            profile_column(dataset, column, filter_sql, filtered_row_count)
            for column in columns.values()
        ]
        return {
            "row_count": row_count,
            "filtered_row_count": filtered_row_count,
            "filter": filter_sql,
            "columns": column_profiles,
        }


def profile_detail(dataset: Dataset, request: dict[str, Any]) -> dict[str, Any]:
    with dataset.lock:
        columns = dataset.column_map()
        column_name = str(request.get("column") or "")
        if column_name not in columns:
            raise ValueError("Choose a valid profile column")
        column = columns[column_name]
        filter_sql = dataset.normalise_filter(request.get("filter"))
        row_count = dataset.row_count()
        filtered_row_count = dataset.filtered_row_count(filter_sql)
        stats = column_stats(dataset, column, filter_sql)
        missing_count = int(stats.get("missing_count") or 0)
        distinct_count = int(stats.get("distinct_count") or 0)
        detail: dict[str, Any] = {
            "name": column.name,
            "duckdb_type": column.duckdb_type,
            "kind": column.kind,
            "row_count": row_count,
            "filtered_row_count": filtered_row_count,
            "missing_count": missing_count,
            "non_missing_count": max(0, filtered_row_count - missing_count),
            "distinct_count": distinct_count,
            "filter": filter_sql,
        }

        if is_numeric_kind(column.kind):
            bin_count = numeric_detail_bin_count(distinct_count)
            histogram = (
                numeric_level_distribution(dataset, column, filter_sql)
                if bin_count == distinct_count
                else numeric_distribution(
                    dataset,
                    column,
                    filter_sql,
                    stats,
                    bin_count=bin_count,
                )
            )
            detail["histogram"] = histogram
            detail["stats"] = numeric_detail_stats(dataset, column, filter_sql)
            detail["zero_count"] = numeric_zero_count(dataset, column, filter_sql)
        elif column.kind in {"date", "datetime"}:
            histogram = temporal_distribution(
                dataset,
                column,
                filter_sql,
                bin_count=PROFILE_DETAIL_HISTOGRAM_BINS,
            )
            detail["histogram"] = histogram
            detail["stats"] = temporal_detail_stats(dataset, column, filter_sql)
        else:
            detail["value_counts"] = top_column_values(
                dataset,
                column,
                filter_sql,
                limit=PROFILE_DETAIL_TOP_VALUE_LIMIT,
            )
            detail["blank_count"] = categorical_blank_count(dataset, column, filter_sql)

        return detail


def numeric_detail_bin_count(distinct_count: int) -> int:
    if 0 < distinct_count <= PROFILE_DETAIL_EXACT_LEVEL_BIN_LIMIT:
        return distinct_count
    return PROFILE_DETAIL_HISTOGRAM_BINS


def numeric_level_distribution(
    dataset: Dataset,
    column: ColumnInfo,
    filter_sql: str,
) -> list[dict[str, Any]]:
    column_sql = quote_ident(column.name)
    sql = f"""
WITH {base_cte(dataset, filter_sql)}
SELECT {column_sql} AS value, COUNT(*) AS count
FROM base
WHERE {column_sql} IS NOT NULL
GROUP BY {column_sql}
ORDER BY {column_sql}
"""
    return [
        {
            "bin": index,
            "lower": json_value(value),
            "upper": json_value(value),
            "count": int(count or 0),
        }
        for index, (value, count) in enumerate(dataset.con.execute(sql).fetchall())
    ]


def profile_column(
    dataset: Dataset,
    column: ColumnInfo,
    filter_sql: str,
    filtered_row_count: int,
) -> dict[str, Any]:
    stats = column_stats(dataset, column, filter_sql)
    missing_count = int(stats.get("missing_count") or 0)
    distinct_count = int(stats.get("distinct_count") or 0)
    missing_rate = (missing_count / filtered_row_count) if filtered_row_count else 0
    min_value = json_value(stats.get("min_value")) if column.kind in {"integer", "numeric", "date", "datetime"} else None
    max_value = json_value(stats.get("max_value")) if column.kind in {"integer", "numeric", "date", "datetime"} else None

    return {
        "name": column.name,
        "duckdb_type": column.duckdb_type,
        "kind": column.kind,
        "missing_count": missing_count,
        "missing_rate": missing_rate,
        "distinct_count": distinct_count,
        "min": min_value,
        "max": max_value,
    }


def base_cte(dataset: Dataset, filter_sql: str, include_row_number: bool = False) -> str:
    where_sql = f"\n  WHERE ({filter_sql})" if filter_sql else ""
    source = dataset.relation_sql()
    if include_row_number:
        return f"base AS (\n  SELECT ROW_NUMBER() OVER () AS __rownum, * FROM {source}{where_sql}\n)"
    return f"base AS (\n  SELECT * FROM {source}{where_sql}\n)"


def column_stats(dataset: Dataset, column: ColumnInfo, filter_sql: str) -> dict[str, Any]:
    column_sql = quote_ident(column.name)
    min_max_sql = (
        f",\n    MIN({column_sql}) AS min_value,\n    MAX({column_sql}) AS max_value"
        if column.kind in {"integer", "numeric", "date", "datetime"}
        else ",\n    NULL AS min_value,\n    NULL AS max_value"
    )
    sql = f"""
WITH {base_cte(dataset, filter_sql)}
SELECT
    COALESCE(SUM(CASE WHEN {column_sql} IS NULL THEN 1 ELSE 0 END), 0) AS missing_count,
    COUNT(DISTINCT {column_sql}) AS distinct_count
    {min_max_sql}
FROM base
"""
    cursor = dataset.con.execute(sql)
    row = cursor.fetchone()
    return dict(zip([description[0] for description in cursor.description], row or []))


def top_column_values(
    dataset: Dataset,
    column: ColumnInfo,
    filter_sql: str,
    *,
    limit: int = PROFILE_TOP_VALUE_LIMIT,
) -> list[dict[str, Any]]:
    column_sql = quote_ident(column.name)
    safe_limit = max(0, int(limit))
    sql = f"""
WITH {base_cte(dataset, filter_sql, include_row_number=True)},
ranked AS (
  SELECT
    {column_sql} AS value,
    COUNT(*) AS count,
    MIN(__rownum) AS first_row
  FROM base
  WHERE {column_sql} IS NOT NULL
  GROUP BY {column_sql}
)
SELECT value, count
FROM ranked
ORDER BY count DESC, first_row
LIMIT {safe_limit}
"""
    return [
        {"value": json_value(value), "count": int(count or 0)}
        for value, count in dataset.con.execute(sql).fetchall()
    ]


def numeric_zero_count(
    dataset: Dataset,
    column: ColumnInfo,
    filter_sql: str,
) -> int:
    column_sql = quote_ident(column.name)
    sql = f"""
WITH {base_cte(dataset, filter_sql)}
SELECT COUNT(*) AS count
FROM base
WHERE {column_sql} = 0
"""
    row = dataset.con.execute(sql).fetchone()
    return int(row[0] if row else 0)


def categorical_blank_count(
    dataset: Dataset,
    column: ColumnInfo,
    filter_sql: str,
) -> int:
    column_sql = quote_ident(column.name)
    sql = f"""
WITH {base_cte(dataset, filter_sql)}
SELECT COUNT(*) AS count
FROM base
WHERE CAST({column_sql} AS VARCHAR) = ''
"""
    row = dataset.con.execute(sql).fetchone()
    return int(row[0] if row else 0)


def numeric_distribution(
    dataset: Dataset,
    column: ColumnInfo,
    filter_sql: str,
    stats: dict[str, Any],
    *,
    bin_count: int = PROFILE_HISTOGRAM_BINS,
) -> list[dict[str, Any]]:
    minimum = finite_float(stats.get("min_value"))
    maximum = finite_float(stats.get("max_value"))
    safe_bin_count = max(1, int(bin_count))
    if minimum is None or maximum is None:
        return histogram_bins(0, 0, {}, safe_bin_count)

    column_sql = quote_ident(column.name)
    if minimum == maximum:
        sql = f"""
WITH {base_cte(dataset, filter_sql)}
SELECT COUNT(*) AS count
FROM base
WHERE {column_sql} IS NOT NULL
"""
        count = int(dataset.con.execute(sql).fetchone()[0] or 0)
        return histogram_bins(minimum, maximum, {0: count}, safe_bin_count)

    span = maximum - minimum
    bin_sql = f"""
WITH {base_cte(dataset, filter_sql)},
bins AS (
  SELECT
    LEAST(
      {safe_bin_count - 1},
      GREATEST(
        0,
        CAST(FLOOR(((CAST({column_sql} AS DOUBLE) - ?) / ?) * {safe_bin_count}) AS INTEGER)
      )
    ) AS bin
  FROM base
  WHERE {column_sql} IS NOT NULL
)
SELECT bin, COUNT(*) AS count
FROM bins
GROUP BY bin
ORDER BY bin
"""
    counts = {
        int(bin_index): int(count or 0)
        for bin_index, count in dataset.con.execute(bin_sql, [minimum, span]).fetchall()
    }
    return histogram_bins(minimum, maximum, counts, safe_bin_count)


def histogram_bins(
    minimum: float,
    maximum: float,
    counts: dict[int, int],
    bin_count: int = PROFILE_HISTOGRAM_BINS,
) -> list[dict[str, Any]]:
    safe_bin_count = max(1, int(bin_count))
    if not counts and minimum == maximum == 0:
        return []
    span = maximum - minimum
    bins: list[dict[str, Any]] = []
    for index in range(safe_bin_count):
        lower = minimum if span == 0 else minimum + (span * index / safe_bin_count)
        upper = maximum if span == 0 else minimum + (span * (index + 1) / safe_bin_count)
        bins.append({
            "bin": index,
            "lower": json_number(lower),
            "upper": json_number(upper),
            "count": counts.get(index, 0),
        })
    return bins


def numeric_detail_stats(dataset: Dataset, column: ColumnInfo, filter_sql: str) -> dict[str, Any]:
    column_sql = quote_ident(column.name)
    value_sql = f"CAST({column_sql} AS DOUBLE)"
    sql = f"""
WITH {base_cte(dataset, filter_sql)}
SELECT
    MIN({column_sql}) AS min,
    QUANTILE_CONT({value_sql}, 0.01) AS p1,
    QUANTILE_CONT({value_sql}, 0.05) AS p5,
    QUANTILE_CONT({value_sql}, 0.25) AS p25,
    QUANTILE_CONT({value_sql}, 0.5) AS median,
    AVG({value_sql}) AS mean,
    QUANTILE_CONT({value_sql}, 0.75) AS p75,
    QUANTILE_CONT({value_sql}, 0.95) AS p95,
    QUANTILE_CONT({value_sql}, 0.99) AS p99,
    MAX({column_sql}) AS max,
    STDDEV_SAMP({value_sql}) AS sd
FROM base
WHERE {column_sql} IS NOT NULL
"""
    row = one_row_dict(dataset, sql)
    return {key: json_value(row.get(key)) for key in ("min", "p1", "p5", "p25", "median", "mean", "p75", "p95", "p99", "max", "sd")}


def temporal_detail_stats(dataset: Dataset, column: ColumnInfo, filter_sql: str) -> dict[str, Any]:
    column_sql = quote_ident(column.name)
    sql = f"""
WITH {base_cte(dataset, filter_sql)}
SELECT
    MIN({column_sql}) AS min,
    QUANTILE_CONT({column_sql}, 0.25) AS p25,
    QUANTILE_CONT({column_sql}, 0.5) AS median,
    QUANTILE_CONT({column_sql}, 0.75) AS p75,
    MAX({column_sql}) AS max
FROM base
WHERE {column_sql} IS NOT NULL
"""
    row = one_row_dict(dataset, sql)
    return {key: json_value(row.get(key)) for key in ("min", "p25", "median", "p75", "max")}


def temporal_distribution(
    dataset: Dataset,
    column: ColumnInfo,
    filter_sql: str,
    *,
    bin_count: int = PROFILE_DETAIL_HISTOGRAM_BINS,
) -> list[dict[str, Any]]:
    column_sql = quote_ident(column.name)
    safe_bin_count = max(1, int(bin_count))
    epoch_sql = f"EPOCH({column_sql})"
    stats_sql = f"""
WITH {base_cte(dataset, filter_sql)}
SELECT
    MIN({epoch_sql}) AS min_epoch,
    MAX({epoch_sql}) AS max_epoch
FROM base
WHERE {column_sql} IS NOT NULL
"""
    stats = one_row_dict(dataset, stats_sql)
    minimum = finite_float(stats.get("min_epoch"))
    maximum = finite_float(stats.get("max_epoch"))
    if minimum is None or maximum is None:
        return []

    if minimum == maximum:
        count_sql = f"""
WITH {base_cte(dataset, filter_sql)}
SELECT COUNT(*) AS count
FROM base
WHERE {column_sql} IS NOT NULL
"""
        count = int(dataset.con.execute(count_sql).fetchone()[0] or 0)
        return temporal_histogram_bins(minimum, maximum, {0: count}, safe_bin_count, column)

    span = maximum - minimum
    bin_sql = f"""
WITH {base_cte(dataset, filter_sql)},
bins AS (
  SELECT
    LEAST(
      {safe_bin_count - 1},
      GREATEST(
        0,
        CAST(FLOOR((({epoch_sql} - ?) / ?) * {safe_bin_count}) AS INTEGER)
      )
    ) AS bin
  FROM base
  WHERE {column_sql} IS NOT NULL
)
SELECT bin, COUNT(*) AS count
FROM bins
GROUP BY bin
ORDER BY bin
"""
    counts = {
        int(bin_index): int(count or 0)
        for bin_index, count in dataset.con.execute(bin_sql, [minimum, span]).fetchall()
    }
    return temporal_histogram_bins(minimum, maximum, counts, safe_bin_count, column)


def temporal_histogram_bins(
    minimum: float,
    maximum: float,
    counts: dict[int, int],
    bin_count: int,
    column: ColumnInfo,
) -> list[dict[str, Any]]:
    span = maximum - minimum
    bins: list[dict[str, Any]] = []
    for index in range(max(1, int(bin_count))):
        lower_epoch = minimum if span == 0 else minimum + (span * index / bin_count)
        upper_epoch = maximum if span == 0 else minimum + (span * (index + 1) / bin_count)
        bins.append({
            "bin": index,
            "lower": temporal_epoch_value(lower_epoch, column),
            "upper": temporal_epoch_value(upper_epoch, column),
            "count": counts.get(index, 0),
        })
    return bins


def temporal_epoch_value(value: float, column: ColumnInfo) -> str | None:
    if not math.isfinite(value):
        return None
    duckdb_type = column.duckdb_type.upper()
    if "TIME" in duckdb_type and "TIMESTAMP" not in duckdb_type:
        total_microseconds = int(round((value % 86_400) * 1_000_000)) % (86_400 * 1_000_000)
        hours, remainder = divmod(total_microseconds, 3_600 * 1_000_000)
        minutes, remainder = divmod(remainder, 60 * 1_000_000)
        seconds, microseconds = divmod(remainder, 1_000_000)
        return dt.time(int(hours), int(minutes), int(seconds), int(microseconds)).isoformat()
    return dt.datetime.fromtimestamp(value, tz=dt.timezone.utc).replace(tzinfo=None).isoformat()


def one_row_dict(dataset: Dataset, sql: str, parameters: list[Any] | None = None) -> dict[str, Any]:
    cursor = dataset.con.execute(sql, parameters or [])
    row = cursor.fetchone()
    return dict(zip([description[0] for description in cursor.description], row or []))


def finite_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def json_value(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, (dt.date, dt.datetime, dt.time)):
        return value.isoformat()
    if isinstance(value, (int, float, Decimal)):
        return json_number(value)
    return str(value)
