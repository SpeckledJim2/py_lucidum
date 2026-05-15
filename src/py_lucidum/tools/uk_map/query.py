from __future__ import annotations

from typing import Any

from py_lucidum.core import ColumnInfo, Dataset, is_numeric_kind, json_number, quote_ident
from py_lucidum.tools.line_bar.query import (
    denominator_warnings,
    denominator_valid_condition,
    normalise_denominator,
    response_parts,
    response_summary,
    summarize_denominator,
    weighted_value_sql,
)


LEVELS = {
    "area": {
        "column_key": "postcode_area",
        "request_key": "areaColumn",
        "default_column": "PostcodeArea",
        "join_property": "PostcodeArea",
        "label": "areas",
    },
    "sector": {
        "column_key": "postcode_sector",
        "request_key": "sectorColumn",
        "default_column": "PostcodeSector",
        "join_property": "PostcodeSector",
        "label": "sectors",
    },
    "unit": {
        "column_key": "postcode_unit",
        "request_key": "unitColumn",
        "default_column": "PostcodeUnit",
        "join_property": "PostcodeUnit",
        "label": "units",
    },
}

COORDINATE_COLUMNS = {
    "latitude": {
        "request_key": "latitudeColumn",
        "default_column": "lat",
        "label": "latitude",
    },
    "longitude": {
        "request_key": "longitudeColumn",
        "default_column": "long",
        "label": "longitude",
    },
}


def summary(dataset: Dataset, request: dict[str, Any], defaults: dict[str, str] | None = None) -> dict[str, Any]:
    with dataset.lock:
        columns = dataset.column_map()
        level = normalise_level(request.get("level"))
        response = normalise_response(request, columns)
        denominator = normalise_denominator(request.get("denominator", request.get("weight")), columns)
        app_defaults = defaults or {}
        join_column = normalise_join_column(level, request, app_defaults, columns)
        filter_sql = dataset.normalise_filter(request.get("filter"))

        row_count = dataset.row_count()
        filtered_row_count = dataset.filtered_row_count(filter_sql)
        denominator_summary = summarize_denominator(dataset, [response], denominator, filter_sql)
        response_summaries = response_summary(dataset, [response], denominator, filter_sql)
        point_summary: dict[str, Any] | None = None
        if level == "unit":
            latitude_column = normalise_coordinate_column("latitude", request, app_defaults, columns)
            longitude_column = normalise_coordinate_column("longitude", request, app_defaults, columns)
            rows, point_summary = unit_rows(
                dataset,
                join_column,
                latitude_column,
                longitude_column,
                response,
                denominator,
                filter_sql,
            )
        else:
            rows = map_rows(dataset, join_column, response, denominator, filter_sql)
        warnings = denominator_warnings(denominator, denominator_summary)
        if not rows:
            if level == "unit" and point_summary and point_summary["summary_count"]:
                warnings.append(f"No plot-ready {join_column} points were found after filtering.")
            else:
                warnings.append(f"No non-empty {join_column} values were found after filtering.")

        level_info = LEVELS[level]
        payload = {
            "level": level,
            "level_label": level_info["label"],
            "join_column": join_column,
            "join_property": level_info["join_property"],
            "row_count": row_count,
            "filtered_row_count": filtered_row_count,
            "filter": filter_sql,
            "denominator": {
                "column": denominator["column"],
                "label": denominator["label"],
                "bar_label": denominator["bar_label"],
                "value": json_number(denominator_summary.get("value")),
                "missing_response_rows": json_number(denominator_summary.get("missing_response_rows")),
                "missing_weight_rows": json_number(denominator_summary.get("missing_weight_rows")),
                "zero_weight_rows": json_number(denominator_summary.get("zero_weight_rows")),
                "negative_weight_rows": json_number(denominator_summary.get("negative_weight_rows")),
            },
            "response": {
                "label": response["label"],
                "numerator": response["numerator"],
                "value": response_summaries[0]["value"] if response_summaries else None,
                "numerator_total": response_summaries[0]["numerator"] if response_summaries else None,
                "denominator": response_summaries[0]["denominator"] if response_summaries else None,
            },
            "rows": rows,
            "warnings": warnings,
        }
        if point_summary:
            payload["point_summary"] = point_summary
        return payload


def normalise_level(raw: Any) -> str:
    level = str(raw or "area").strip().lower()
    if level in {"area", "areas"}:
        return "area"
    if level in {"sector", "sectors"}:
        return "sector"
    if level in {"unit", "units"}:
        return "unit"
    raise ValueError("Choose a valid UK map level")


def normalise_response(request: dict[str, Any], columns: dict[str, ColumnInfo]) -> dict[str, str]:
    numerator = str(request.get("numerator") or request.get("actual") or "")
    if not numerator or numerator not in columns or not is_numeric_kind(columns[numerator].kind):
        raise ValueError("Choose a valid numeric Actual column")
    return {"label": str(request.get("label") or numerator), "numerator": numerator}


def normalise_join_column(
    level: str,
    request: dict[str, Any],
    defaults: dict[str, str],
    columns: dict[str, ColumnInfo],
) -> str:
    level_info = LEVELS[level]
    request_key = str(level_info["request_key"])
    defaults_key = str(level_info["column_key"])
    default_column = str(level_info["default_column"])
    raw = (
        request.get(request_key)
        or request.get(defaults_key)
        or defaults.get(defaults_key)
        or default_column
    )
    column = str(raw or "").strip()
    if column not in columns:
        raise ValueError(f"Choose a valid {level.replace('_', ' ')} postcode column")
    return column


def normalise_coordinate_column(
    name: str,
    request: dict[str, Any],
    defaults: dict[str, str],
    columns: dict[str, ColumnInfo],
) -> str:
    info = COORDINATE_COLUMNS[name]
    request_key = str(info["request_key"])
    default_column = str(info["default_column"])
    raw = request.get(request_key) or request.get(name) or defaults.get(name) or default_column
    column = str(raw or "").strip()
    if column not in columns or not is_numeric_kind(columns[column].kind):
        raise ValueError(f"Choose a valid numeric {info['label']} column")
    return column


def map_rows(
    dataset: Dataset,
    join_column: str,
    response: dict[str, str],
    denominator: dict[str, str | None],
    filter_sql: str = "",
) -> list[dict[str, Any]]:
    sql = build_summary_sql(dataset.relation_sql(), join_column, response, denominator, filter_sql)
    cursor = dataset.con.execute(sql)
    rows = [dict(zip([d[0] for d in cursor.description], row)) for row in cursor.fetchall()]
    return [
        {
            "key": row["key"],
            "row_count": json_number(row.get("row_count")),
            "numerator": json_number(row.get("resp0_num")),
            "denominator": json_number(row.get("resp0_den")),
            "volume": json_number(row.get("resp0_den")),
            "value": json_number(row.get("resp0")),
        }
        for row in rows
    ]


def build_summary_sql(
    relation: str,
    join_column: str,
    response: dict[str, str],
    denominator: dict[str, str | None],
    filter_sql: str = "",
) -> str:
    valid_condition = denominator_valid_condition([response], denominator)
    weight_expr = weighted_value_sql(denominator, valid_condition)
    num_expr, den_expr, value_expr = response_parts(response, 0)
    join_expr = f"NULLIF(TRIM(CAST({quote_ident(join_column)} AS VARCHAR)), '')"
    where_sql = f"\n  WHERE ({filter_sql})" if filter_sql else ""
    return f"""
WITH base AS (
  SELECT * FROM {relation}{where_sql}
),
keyed AS (
  SELECT
    {join_expr} AS __map_key,
    {weight_expr} AS __weight_value,
    *
  FROM base
),
summary AS (
  SELECT
    __map_key AS key,
    COUNT(*) AS row_count,
    {num_expr},
    {den_expr}
  FROM keyed
  WHERE __map_key IS NOT NULL
  GROUP BY __map_key
)
SELECT
    *,
    {value_expr}
FROM summary
ORDER BY key
"""


def unit_rows(
    dataset: Dataset,
    join_column: str,
    latitude_column: str,
    longitude_column: str,
    response: dict[str, str],
    denominator: dict[str, str | None],
    filter_sql: str = "",
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    sql = build_unit_summary_sql(
        dataset.relation_sql(),
        join_column,
        latitude_column,
        longitude_column,
        response,
        denominator,
        filter_sql,
    )
    cursor = dataset.con.execute(sql)
    raw_rows = [dict(zip([d[0] for d in cursor.description], row)) for row in cursor.fetchall()]
    rows: list[dict[str, Any]] = []
    missing_value_count = 0
    missing_coordinate_count = 0
    for row in raw_rows:
        value = json_number(row.get("resp0"))
        latitude = json_number(row.get("latitude"))
        longitude = json_number(row.get("longitude"))
        if value is None:
            missing_value_count += 1
            continue
        if latitude is None or longitude is None:
            missing_coordinate_count += 1
            continue
        rows.append(
            {
                "key": row["key"],
                "row_count": json_number(row.get("row_count")),
                "numerator": json_number(row.get("resp0_num")),
                "denominator": json_number(row.get("resp0_den")),
                "volume": json_number(row.get("resp0_den")),
                "value": value,
                "latitude": latitude,
                "longitude": longitude,
            }
        )
    return rows, {
        "summary_count": len(raw_rows),
        "plotted_count": len(rows),
        "missing_value_count": missing_value_count,
        "missing_coordinate_count": missing_coordinate_count,
    }


def build_unit_summary_sql(
    relation: str,
    join_column: str,
    latitude_column: str,
    longitude_column: str,
    response: dict[str, str],
    denominator: dict[str, str | None],
    filter_sql: str = "",
) -> str:
    valid_condition = denominator_valid_condition([response], denominator)
    weight_expr = weighted_value_sql(denominator, valid_condition)
    num_expr, den_expr, value_expr = response_parts(response, 0)
    join_expr = f"NULLIF(TRIM(CAST({quote_ident(join_column)} AS VARCHAR)), '')"
    latitude_expr = f"TRY_CAST({quote_ident(latitude_column)} AS DOUBLE)"
    longitude_expr = f"TRY_CAST({quote_ident(longitude_column)} AS DOUBLE)"
    valid_latitude = f"CASE WHEN {latitude_expr} BETWEEN -90 AND 90 THEN {latitude_expr} ELSE NULL END"
    valid_longitude = f"CASE WHEN {longitude_expr} BETWEEN -180 AND 180 THEN {longitude_expr} ELSE NULL END"
    where_sql = f"\n  WHERE ({filter_sql})" if filter_sql else ""
    return f"""
WITH base AS (
  SELECT * FROM {relation}{where_sql}
),
keyed AS (
  SELECT
    {join_expr} AS __map_key,
    {valid_latitude} AS __latitude,
    {valid_longitude} AS __longitude,
    {weight_expr} AS __weight_value,
    *
  FROM base
),
summary AS (
  SELECT
    __map_key AS key,
    COUNT(*) AS row_count,
    AVG(__latitude) AS latitude,
    AVG(__longitude) AS longitude,
    {num_expr},
    {den_expr}
  FROM keyed
  WHERE __map_key IS NOT NULL
  GROUP BY __map_key
)
SELECT
    *,
    {value_expr}
FROM summary
ORDER BY key
"""


__all__ = [
    "build_unit_summary_sql",
    "build_summary_sql",
    "map_rows",
    "unit_rows",
    "normalise_coordinate_column",
    "normalise_join_column",
    "normalise_level",
    "normalise_response",
    "summary",
]
