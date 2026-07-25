from __future__ import annotations

from typing import Any

import duckdb

from py_lucidum.core import (
    ColumnInfo,
    Dataset,
    denominator_warnings,
    denominator_valid_condition,
    has_denominator_column,
    is_numeric_kind,
    json_number,
    metric_relation_context,
    normalise_denominator,
    normalise_denominator_source,
    quote_ident,
    relation_row_count,
    response_parts,
    response_summary_for_relation,
    summarize_denominator_for_relation,
    weighted_value_sql,
)
from py_lucidum.tools.uk_map.smoothing import (
    MAX_SMOOTHING_LEVEL,
    SECTOR_SMOOTHING_LOAD_WARNING,
    build_smoothed_sector_sql,
    normalise_smoothing_level,
)


LEVELS = {
    "area": {
        "column_key": "postcode_area",
        "request_key": "areaColumn",
        "default_column": "PostcodeArea",
        "aliases": ("PostcodeArea", "POSTCODE_AREA"),
        "join_property": "PostcodeArea",
        "label": "areas",
    },
    "sector": {
        "column_key": "postcode_sector",
        "request_key": "sectorColumn",
        "default_column": "PostcodeSector",
        "aliases": ("PostcodeSector", "POSTCODE_SECTOR"),
        "join_property": "PostcodeSector",
        "label": "sectors",
    },
    "unit": {
        "column_key": "postcode_unit",
        "request_key": "unitColumn",
        "default_column": "PostcodeUnit",
        "aliases": ("PostcodeUnit", "POSTCODE_UNIT"),
        "join_property": "PostcodeUnit",
        "label": "units",
    },
}

COORDINATE_COLUMNS = {
    "latitude": {
        "request_key": "latitudeColumn",
        "default_column": "lat",
        "aliases": ("lat", "latitude", "LATITUDE"),
        "label": "latitude",
    },
    "longitude": {
        "request_key": "longitudeColumn",
        "default_column": "long",
        "aliases": ("long", "longitude", "LONGITUDE", "LONGiTUDE"),
        "label": "longitude",
    },
}

UNIT_POINT_FIELDS = (
    "key",
    "row_count",
    "numerator",
    "denominator",
    "volume",
    "value",
    "latitude",
    "longitude",
)


def summary(dataset: Dataset, request: dict[str, Any], defaults: dict[str, str] | None = None) -> dict[str, Any]:
    with dataset.lock:
        source_id = dataset.normalise_source(request.get("source"))
        raw_response = str(request.get("numerator") or request.get("actual") or "").strip()
        raw_denominator = request.get("denominator", request.get("weight"))
        denominator_source = normalise_denominator_source(
            dataset,
            request.get("denominatorSource"),
            raw_denominator,
        )
        fields = [(raw_response, source_id)]
        if has_denominator_column(raw_denominator):
            fields.append((str(raw_denominator), denominator_source))
        context = metric_relation_context(dataset, source_id=source_id, fields=fields)
        relation = context["relation"]
        columns = context["columns"]
        level = normalise_level(request.get("level"))
        smoothing_level = normalise_smoothing_level(request.get("smoothingLevel"))
        compact_unit_points = level == "unit" and bool(request.get("compactUnitPoints"))
        response = normalise_response(request, columns)
        denominator = normalise_denominator(request.get("denominator", request.get("weight")), columns)
        app_defaults = defaults or {}
        join_column = normalise_join_column(level, request, app_defaults, columns)
        filter_sql = dataset.normalise_filter_for_relation(request.get("filter"), relation)

        point_summary: dict[str, Any] | None = None
        if level == "unit":
            latitude_column = normalise_coordinate_column("latitude", request, app_defaults, columns)
            longitude_column = normalise_coordinate_column("longitude", request, app_defaults, columns)
            row_count = context["row_count"]
            filtered_row_count = relation_row_count(dataset, relation, filter_sql)
            denominator_summary = summarize_denominator_for_relation(dataset, relation, [response], denominator, filter_sql)
            response_summaries = response_summary_for_relation(dataset, relation, [response], denominator, filter_sql)
            rows_or_points, point_summary = unit_rows(
                dataset,
                join_column,
                latitude_column,
                longitude_column,
                response,
                denominator,
                filter_sql,
                source_id=source_id,
                relation=relation,
                compact=compact_unit_points,
            )
            smoothing = smoothing_metadata(0, smoothing_level, point_summary["plotted_count"])
            smoothing_warning = None
            rows = [] if compact_unit_points else rows_or_points
        else:
            row_count = context["row_count"]
            filtered_row_count = relation_row_count(dataset, relation, filter_sql)
            denominator_summary = summarize_denominator_for_relation(dataset, relation, [response], denominator, filter_sql)
            response_summaries = response_summary_for_relation(dataset, relation, [response], denominator, filter_sql)
            if level == "sector":
                rows, smoothing, smoothing_warning = sector_rows(
                    dataset,
                    join_column,
                    response,
                    denominator,
                    filter_sql,
                    source_id=source_id,
                    relation=relation,
                    smoothing_level=smoothing_level,
                )
                smoothing["requested_level"] = smoothing_level
            else:
                rows = map_rows(
                    dataset,
                    join_column,
                    response,
                    denominator,
                    filter_sql,
                    source_id=source_id,
                    relation=relation,
                )
                smoothing = smoothing_metadata(0, smoothing_level, len(rows))
                smoothing_warning = None
        warnings = denominator_warnings(denominator, denominator_summary, [response])
        if smoothing_warning:
            warnings.append(smoothing_warning)
        plotted_count = int(point_summary["plotted_count"]) if point_summary else len(rows)
        if plotted_count == 0:
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
            "source": source_id,
            "row_count": row_count,
            "filtered_row_count": filtered_row_count,
            "filter": filter_sql,
            "denominator": {
                "column": denominator["column"],
                "source": denominator_source,
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
            "smoothing": smoothing,
            "warnings": warnings,
        }
        if compact_unit_points:
            payload["unit_points"] = rows_or_points
        else:
            payload["rows"] = rows
        if point_summary:
            payload["point_summary"] = point_summary
        return payload


def smoothing_metadata(level: int, requested_level: int, matched_rows: int) -> dict[str, Any]:
    return {
        "level": level,
        "requested_level": requested_level,
        "max_level": MAX_SMOOTHING_LEVEL,
        "applied": False,
        "method": "none",
        "matched_rows": matched_rows,
        "smoothed_rows": 0,
        "fallback_rows": 0,
        "contributing_rows": 0,
    }


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
    configured = request.get(request_key) or request.get(defaults_key) or defaults.get(defaults_key)
    raw = (
        request.get(request_key)
        or request.get(defaults_key)
        or defaults.get(defaults_key)
        or default_column
    )
    column = str(raw or "").strip()
    if configured:
        if column not in columns:
            raise ValueError(f"Choose a valid {level.replace('_', ' ')} postcode column")
        return column
    resolved = resolve_alias_column(column, tuple(level_info["aliases"]), columns)
    if not resolved:
        raise ValueError(f"Choose a valid {level.replace('_', ' ')} postcode column")
    return resolved


def normalise_coordinate_column(
    name: str,
    request: dict[str, Any],
    defaults: dict[str, str],
    columns: dict[str, ColumnInfo],
) -> str:
    info = COORDINATE_COLUMNS[name]
    request_key = str(info["request_key"])
    default_column = str(info["default_column"])
    configured = request.get(request_key) or request.get(name) or defaults.get(name)
    raw = configured or default_column
    column = str(raw or "").strip()
    if configured:
        if column not in columns or not is_numeric_kind(columns[column].kind):
            raise ValueError(f"Choose a valid numeric {info['label']} column")
        return column
    column = resolve_alias_column(column, tuple(info["aliases"]), columns) or ""
    if not column or not is_numeric_kind(columns[column].kind):
        raise ValueError(f"Choose a valid numeric {info['label']} column")
    return column


def resolve_alias_column(
    requested: str,
    aliases: tuple[str, ...],
    columns: dict[str, ColumnInfo],
) -> str | None:
    if requested in columns:
        return requested
    for alias in aliases:
        if alias in columns:
            return alias
    return None


def map_rows(
    dataset: Dataset,
    join_column: str,
    response: dict[str, str],
    denominator: dict[str, str | None],
    filter_sql: str = "",
    source_id: Any = None,
    relation: str | None = None,
) -> list[dict[str, Any]]:
    sql = build_summary_sql(
        relation or dataset.relation_sql_for_source(source_id),
        join_column,
        response,
        denominator,
        filter_sql,
    )
    cursor = dataset.con.execute(sql)
    column_names = [d[0] for d in cursor.description]
    rows = [dict(zip(column_names, row)) for row in cursor.fetchall()]
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


def sector_rows(
    dataset: Dataset,
    join_column: str,
    response: dict[str, str],
    denominator: dict[str, str | None],
    filter_sql: str = "",
    source_id: Any = None,
    relation: str | None = None,
    *,
    smoothing_level: int = 0,
) -> tuple[list[dict[str, Any]], dict[str, Any], str | None]:
    if smoothing_level <= 0:
        rows = map_rows(
            dataset,
            join_column,
            response,
            denominator,
            filter_sql,
            source_id=source_id,
            relation=relation,
        )
        return rows, sector_smoothing_metadata(0, len(rows)), None

    raw_sql = build_summary_sql(
        relation or dataset.relation_sql_for_source(source_id),
        join_column,
        response,
        denominator,
        filter_sql,
        order_by=False,
    )
    try:
        cursor = dataset.con.execute(build_smoothed_sector_sql(raw_sql, smoothing_level))
    except (duckdb.Error, OSError, ValueError) as exc:
        rows = map_rows(
            dataset,
            join_column,
            response,
            denominator,
            filter_sql,
            source_id=source_id,
            relation=relation,
        )
        metadata = sector_smoothing_metadata(smoothing_level, len(rows))
        metadata["method"] = "shared_edge_weighted_numerator"
        metadata["warning"] = SECTOR_SMOOTHING_LOAD_WARNING
        metadata["fallback_rows"] = len(rows)
        return rows, metadata, f"{SECTOR_SMOOTHING_LOAD_WARNING} {exc}"

    return smoothed_sector_rows_from_cursor(cursor, smoothing_level)


def sector_smoothing_metadata(level: int, matched_rows: int) -> dict[str, Any]:
    return {
        "level": level,
        "max_level": MAX_SMOOTHING_LEVEL,
        "applied": False,
        "method": "none" if level <= 0 else "shared_edge_weighted_numerator",
        "matched_rows": matched_rows,
        "target_rows": matched_rows,
        "smoothed_rows": 0,
        "fallback_rows": 0,
        "contributing_rows": 0,
    }


def smoothed_sector_rows_from_cursor(
    cursor: Any,
    smoothing_level: int,
) -> tuple[list[dict[str, Any]], dict[str, Any], str | None]:
    column_names = [d[0] for d in cursor.description]
    raw_rows = [dict(zip(column_names, row)) for row in cursor.fetchall()]
    if not raw_rows:
        metadata = sector_smoothing_metadata(smoothing_level, 0)
        metadata["applied"] = True
        return [], metadata, None

    first = raw_rows[0]
    rows = [
        {
            "key": row["key"],
            "row_count": json_number(row.get("row_count")),
            "numerator": json_number(row.get("numerator")),
            "denominator": json_number(row.get("denominator")),
            "volume": json_number(row.get("volume")),
            "value": json_number(row.get("value")),
            "raw_numerator": json_number(row.get("raw_numerator")),
            "raw_denominator": json_number(row.get("raw_denominator")),
            "raw_volume": json_number(row.get("raw_volume")),
            "raw_value": json_number(row.get("raw_value")),
            "raw_row_count": json_number(row.get("raw_row_count")),
            "smoothing_contributing_sectors": int(row.get("smoothing_contributing_sectors") or 0),
        }
        for row in raw_rows
    ]
    metadata = {
        "level": smoothing_level,
        "max_level": MAX_SMOOTHING_LEVEL,
        "applied": True,
        "method": "shared_edge_weighted_numerator",
        "matched_rows": int(first.get("__matched_rows") or 0),
        "target_rows": int(first.get("__target_rows") or 0),
        "smoothed_rows": int(first.get("__smoothed_rows") or 0),
        "fallback_rows": int(first.get("__fallback_rows") or 0),
        "contributing_rows": int(first.get("__contributing_rows") or 0),
    }
    return rows, metadata, None


def build_summary_sql(
    relation: str,
    join_column: str,
    response: dict[str, str],
    denominator: dict[str, str | None],
    filter_sql: str = "",
    *,
    order_by: bool = True,
) -> str:
    valid_condition = denominator_valid_condition([response], denominator)
    weight_expr = weighted_value_sql(denominator, valid_condition)
    num_expr, den_expr, value_expr = response_parts(response, 0)
    join_expr = f"NULLIF(TRIM(CAST({quote_ident(join_column)} AS VARCHAR)), '')"
    where_sql = f"\n  WHERE ({filter_sql})" if filter_sql else ""
    order_sql = "\nORDER BY key" if order_by else ""
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
{order_sql}
"""


def unit_rows(
    dataset: Dataset,
    join_column: str,
    latitude_column: str,
    longitude_column: str,
    response: dict[str, str],
    denominator: dict[str, str | None],
    filter_sql: str = "",
    source_id: Any = None,
    relation: str | None = None,
    *,
    compact: bool = False,
) -> tuple[list[dict[str, Any]] | dict[str, list[Any]], dict[str, int]]:
    sql = build_unit_summary_sql(
        relation or dataset.relation_sql_for_source(source_id),
        join_column,
        latitude_column,
        longitude_column,
        response,
        denominator,
        filter_sql,
    )
    cursor = dataset.con.execute(sql)
    column_names = [d[0] for d in cursor.description]
    column_indexes = {name: index for index, name in enumerate(column_names)}
    raw_rows = cursor.fetchall()
    if compact:
        return compact_unit_points(raw_rows, column_indexes)

    raw_dicts = [dict(zip(column_names, row)) for row in raw_rows]
    rows: list[dict[str, Any]] = []
    missing_value_count = 0
    missing_coordinate_count = 0
    for row in raw_dicts:
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
        "summary_count": len(raw_dicts),
        "plotted_count": len(rows),
        "missing_value_count": missing_value_count,
        "missing_coordinate_count": missing_coordinate_count,
    }


def compact_unit_points(
    raw_rows: list[tuple[Any, ...]],
    column_indexes: dict[str, int],
) -> tuple[dict[str, list[Any]], dict[str, int]]:
    points: dict[str, list[Any]] = {field: [] for field in UNIT_POINT_FIELDS}
    missing_value_count = 0
    missing_coordinate_count = 0
    key_index = column_indexes["key"]
    row_count_index = column_indexes["row_count"]
    latitude_index = column_indexes["latitude"]
    longitude_index = column_indexes["longitude"]
    numerator_index = column_indexes["resp0_num"]
    denominator_index = column_indexes["resp0_den"]
    value_index = column_indexes["resp0"]
    for row in raw_rows:
        value = json_number(row[value_index])
        latitude = json_number(row[latitude_index])
        longitude = json_number(row[longitude_index])
        if value is None:
            missing_value_count += 1
            continue
        if latitude is None or longitude is None:
            missing_coordinate_count += 1
            continue
        denominator = json_number(row[denominator_index])
        points["key"].append(row[key_index])
        points["row_count"].append(json_number(row[row_count_index]))
        points["numerator"].append(json_number(row[numerator_index]))
        points["denominator"].append(denominator)
        points["volume"].append(denominator)
        points["value"].append(value)
        points["latitude"].append(latitude)
        points["longitude"].append(longitude)
    return points, {
        "summary_count": len(raw_rows),
        "plotted_count": len(points["key"]),
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
    "compact_unit_points",
    "map_rows",
    "sector_rows",
    "unit_rows",
    "normalise_coordinate_column",
    "normalise_join_column",
    "normalise_level",
    "normalise_response",
    "summary",
]
