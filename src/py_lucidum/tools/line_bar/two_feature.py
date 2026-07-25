from __future__ import annotations

import math
from typing import Any

from py_lucidum.core import (
    ColumnInfo,
    Dataset,
    ModelPredictionSource,
    denominator_exclusion_warnings,
    denominator_warnings,
    has_denominator_column,
    is_numeric_kind,
    json_number,
    normalise_denominator,
    normalise_denominator_source,
    parse_positive_float,
    quote_ident,
    sql_literal,
)


DEFAULT_MAX_GROUPS = 10_000
DEFAULT_HEATMAP_MAX_GROUPS = 100_000
DEFAULT_TABLE_PAGE_SIZE = 10_000
MAX_LINE_SERIES = 80
MAX_DENSE_GRID_CELLS = 40_000
DATE_BUCKETS = {"hour", "day", "week", "month", "year"}


def has_two_groupings(request: dict[str, Any]) -> bool:
    raw = request.get("groupings")
    return isinstance(raw, list) and len(raw) == 2


def chart_max_groups(result: dict[str, Any]) -> int:
    default = DEFAULT_HEATMAP_MAX_GROUPS if result["plot_type"] == "heatmap" else DEFAULT_MAX_GROUPS
    return positive_int(result["request"].get("maxGroups"), default)


def chart(dataset: Dataset, request: dict[str, Any]) -> dict[str, Any]:
    from . import query as line_bar_query

    with dataset.lock:
        result = build_result(dataset, request)
        fetched = fetch_chart_rows(dataset, result)
        rows = fetched["rows"]
        warnings = [*result["warnings"]]
        missing_warnings = continuous_missing_warnings(rows, result["groupings"])
        group_count = fetched["group_count"]
        max_groups = chart_max_groups(result)
        groups_truncated = group_count > max_groups
        if groups_truncated:
            group_label = "grouped heatmap cells" if result["plot_type"] == "heatmap" else "two-feature groups"
            warnings.append(line_bar_query.overlarge_chart_message(max_groups, group_label))

        dense_grid_cells = 0
        dense_grid_too_large = False
        series_truncated = False
        if rows and result["plot_type"] == "surface":
            first_values = {row["group0_sort"] for row in rows if not row["group0_missing"]}
            second_values = {row["group1_sort"] for row in rows if not row["group1_missing"]}
            dense_grid_cells = len(first_values) * len(second_values)
            dense_grid_too_large = dense_grid_cells > MAX_DENSE_GRID_CELLS
            if dense_grid_too_large:
                rows = []
                warnings.append(
                    f"The two-feature surface needs {dense_grid_cells:,} grid cells; "
                    f"the chart limit is {MAX_DENSE_GRID_CELLS:,}. Use Table view to inspect all groups."
                )
        elif rows and result["plot_type"] == "lines":
            series_index = 1 if result["groupings"][0]["continuous"] else 0
            series_key = f"group{series_index}"
            series_sort_key = f"group{series_index}_sort"
            ordered_series = sorted(
                {
                    (row.get(series_sort_key) is None, sortable_value(row.get(series_sort_key)), str(row.get(series_key) or ""))
                    for row in rows
                }
            )
            if len(ordered_series) > MAX_LINE_SERIES:
                keep = {item[2] for item in ordered_series[:MAX_LINE_SERIES]}
                rows = [row for row in rows if str(row.get(series_key) or "") in keep]
                series_truncated = True
                warnings.append(
                    f"Showing the first {MAX_LINE_SERIES} line series. "
                    "Use Table view to inspect every grouped value."
                )

        warnings.extend(missing_warnings)
        return {
            **base_payload(result),
            "group_count": group_count,
            "max_groups": max_groups,
            "groups_truncated": groups_truncated,
            "dense_grid_cells": dense_grid_cells,
            "dense_grid_too_large": dense_grid_too_large,
            "series_truncated": series_truncated,
            "rows": rows,
            "warnings": warnings,
        }


def table(dataset: Dataset, request: dict[str, Any]) -> dict[str, Any]:
    with dataset.lock:
        result = build_result(dataset, request)
        page_size = positive_int(request.get("tablePageSize"), DEFAULT_TABLE_PAGE_SIZE)
        page = positive_int(request.get("tablePage"), 1)
        fetched = fetch_table_rows(dataset, result, page=page, page_size=page_size)
        return {
            **base_payload(result),
            "rows": fetched["rows"],
            "summary": fetched["summary"],
            "warnings": result["warnings"],
            "table": {
                "search": fetched["search"],
                "page": fetched["page"],
                "page_size": page_size,
                "page_count": fetched["page_count"],
                "match_count": fetched["match_count"],
                "group_count": fetched["group_count"],
            },
        }


def base_payload(result: dict[str, Any]) -> dict[str, Any]:
    return {
        "source": result["source_id"],
        "groupings": result["groupings"],
        "plot_type": result["plot_type"],
        "tail_percent": result["tail_percent"],
        "row_count": result["row_count"],
        "filtered_row_count": result["filtered_row_count"],
        "filter": result["filter_sql"],
        "responses": [
            {
                "label": response["label"],
                "numerator": response["numerator"],
                **({"source": response["source"]} if response.get("source") else {}),
            }
            for response in result["responses"]
        ],
        **({"field_sources": result["field_sources"]} if result.get("field_sources") else {}),
        "denominator": {
            "column": result["denominator"]["column"],
            "source": result["denominator_source"],
            "label": result["denominator"]["label"],
            "bar_label": result["denominator"]["bar_label"],
            "value": json_number(result["denominator_summary"].get("value")),
            "missing_response_rows": json_number(result["denominator_summary"].get("missing_response_rows")),
            "missing_weight_rows": json_number(result["denominator_summary"].get("missing_weight_rows")),
            "zero_weight_rows": json_number(result["denominator_summary"].get("zero_weight_rows")),
            "negative_weight_rows": json_number(result["denominator_summary"].get("negative_weight_rows")),
        },
        "response_summaries": result["response_summaries"],
        "exclusion_warnings": result["exclusion_warnings"],
        "transform": {"mode": "none", "values": []},
    }


def build_result(dataset: Dataset, request: dict[str, Any]) -> dict[str, Any]:
    from . import query as line_bar_query

    raw_groupings = request.get("groupings")
    if not isinstance(raw_groupings, list) or len(raw_groupings) != 2:
        raise ValueError("Choose exactly two grouping features")

    context = two_feature_context(dataset, request, raw_groupings)
    relation = context["relation"]
    columns = context["columns"]
    legacy_tail_percent = normalise_tail_percent(request.get("tailPercent"))
    groupings = [
        normalise_grouping(
            raw,
            columns,
            source_id=context["field_sources"]["groupings"][index],
            index=index,
            legacy_tail_percent=legacy_tail_percent,
        )
        for index, raw in enumerate(raw_groupings)
    ]
    if grouping_identity(groupings[0]) == grouping_identity(groupings[1]):
        raise ValueError("Choose two different grouping features")

    filter_sql = dataset.normalise_filter_for_relation(request.get("filter"), relation)
    responses = line_bar_query.normalise_responses(request.get("responses"), columns)
    denominator = normalise_denominator(request.get("denominator", request.get("weight")), columns)
    denominator_source = normalise_denominator_source(
        dataset,
        request.get("denominatorSource"),
        request.get("denominator", request.get("weight")),
    )
    grouped_sql = build_grouped_sql(
        relation=relation,
        groupings=groupings,
        responses=responses,
        denominator=denominator,
        filter_sql=filter_sql,
    )
    denominator_summary = line_bar_query.relation_denominator_summary(
        dataset,
        relation,
        responses,
        denominator,
        filter_sql,
    )
    response_summaries = line_bar_query.relation_response_summary(
        dataset,
        relation,
        responses,
        denominator,
        filter_sql,
    )
    return {
        "source_id": context["source_id"],
        "field_sources": context["field_sources"],
        "groupings": groupings,
        "plot_type": plot_type(groupings),
        "tail_percent": groupings[0]["tail_percent"],
        "row_count": context["row_count"],
        "filtered_row_count": line_bar_query.relation_row_count(dataset, relation, filter_sql),
        "filter_sql": filter_sql,
        "responses": responses,
        "denominator": denominator,
        "denominator_source": denominator_source,
        "denominator_summary": denominator_summary,
        "response_summaries": response_summaries,
        "grouped_sql": grouped_sql,
        "exclusion_warnings": denominator_exclusion_warnings(denominator, denominator_summary, responses),
        "warnings": denominator_warnings(denominator, denominator_summary, responses),
        "request": request,
    }


def two_feature_context(
    dataset: Dataset,
    request: dict[str, Any],
    raw_groupings: list[Any],
) -> dict[str, Any]:
    from . import query as line_bar_query

    source_id = dataset.normalise_source(request.get("source"))
    dataset_columns = dataset.column_map()
    columns = dict(dataset_columns)
    prediction_sources: dict[str, ModelPredictionSource] = {}
    grouping_sources: list[str] = []
    for raw in raw_groupings:
        item = raw if isinstance(raw, dict) else {}
        feature = str(item.get("feature") or "").strip()
        source = line_bar_query.field_source_id(dataset, item.get("source"), source_id)
        grouping_sources.append(source)
        line_bar_query.add_field_column(
            dataset,
            columns,
            dataset_columns,
            prediction_sources,
            feature,
            source,
        )

    response_sources: list[str] = []
    raw_responses = request.get("responses")
    if isinstance(raw_responses, list):
        for raw in raw_responses:
            if not isinstance(raw, dict):
                continue
            source = line_bar_query.field_source_id(dataset, raw.get("source"), source_id)
            response_sources.append(source)
            line_bar_query.add_field_column(
                dataset,
                columns,
                dataset_columns,
                prediction_sources,
                str(raw.get("numerator") or ""),
                source,
            )

    raw_denominator = request.get("denominator", request.get("weight"))
    denominator_source = normalise_denominator_source(
        dataset,
        request.get("denominatorSource"),
        raw_denominator,
    )
    if has_denominator_column(raw_denominator):
        line_bar_query.add_field_column(
            dataset,
            columns,
            dataset_columns,
            prediction_sources,
            str(raw_denominator),
            denominator_source,
        )

    relation = (
        line_bar_query.mixed_relation_sql(
            dataset,
            list(prediction_sources.values()),
            preserve_dataset_rows=denominator_source != "dataset",
        )
        if prediction_sources
        else dataset.relation_sql_for_source(source_id)
    )
    return {
        "source_id": source_id,
        "relation": relation,
        "columns": columns,
        "row_count": line_bar_query.relation_row_count(dataset, relation),
        "field_sources": {
            "groupings": grouping_sources,
            "responses": response_sources,
            "denominator": denominator_source,
        },
    }


def normalise_grouping(
    raw: Any,
    columns: dict[str, ColumnInfo],
    *,
    source_id: str,
    index: int,
    legacy_tail_percent: float,
) -> dict[str, Any]:
    item = raw if isinstance(raw, dict) else {}
    feature = str(item.get("feature") or "").strip()
    column = columns.get(feature)
    if column is None:
        raise ValueError(f"Choose a valid Feature {index + 1}")
    quantile_mode = "quantile" if use_quantiles(item.get("quantileMode")) and is_numeric_kind(column.kind) else "off"
    date_bucket = normalise_date_bucket(item.get("dateBucket")) if column.kind in {"date", "datetime"} else "none"
    factor_override_supported = is_numeric_kind(column.kind) or column.kind in {"date", "datetime"}
    as_factor = boolean_flag(item.get("asFactor")) and factor_override_supported
    band_width = parse_positive_float(item.get("bandWidth")) or 0.0
    continuous = (
        (
            is_numeric_kind(column.kind)
            and quantile_mode == "off"
        )
        or column.kind in {"date", "datetime"}
    ) and not as_factor
    group_kind = "quantile" if quantile_mode == "quantile" else column.kind
    tail_percent = (
        normalise_tail_percent(item.get("tailPercent"))
        if "tailPercent" in item
        else legacy_tail_percent
    )
    return {
        "feature": feature,
        "source": source_id,
        "kind": column.kind,
        "group_kind": group_kind,
        "band_width": json_number(band_width) or 0,
        "quantile_mode": quantile_mode,
        "date_bucket": date_bucket,
        "as_factor": as_factor,
        "continuous": continuous,
        "tail_percent": tail_percent,
    }


def grouping_identity(grouping: dict[str, Any]) -> tuple[str, str]:
    return str(grouping["source"]), str(grouping["feature"])


def plot_type(groupings: list[dict[str, Any]]) -> str:
    continuous_count = sum(bool(grouping["continuous"]) for grouping in groupings)
    if continuous_count == 2:
        return "surface"
    if continuous_count == 1:
        return "lines"
    return "heatmap"


def build_grouped_sql(
    *,
    relation: str,
    groupings: list[dict[str, Any]],
    responses: list[dict[str, str]],
    denominator: dict[str, str | None],
    filter_sql: str,
) -> str:
    from . import query as line_bar_query

    required: list[str] = []
    for column in [
        *(grouping["feature"] for grouping in groupings),
        *(response["numerator"] for response in responses),
        denominator.get("column"),
    ]:
        if column and column not in required:
            required.append(str(column))
    source_columns = "".join(f",\n    {quote_ident(column)}" for column in required)
    where_sql = f"\n  WHERE ({filter_sql})" if filter_sql else ""
    response_columns = "".join(
        f",\n    TRY_CAST({quote_ident(response['numerator'])} AS DOUBLE) AS __resp{index}_value"
        for index, response in enumerate(responses)
    )
    denominator_sql = (
        f"TRY_CAST({quote_ident(str(denominator['column']))} AS DOUBLE)"
        if denominator.get("column")
        else "1"
    )

    base_group_columns = []
    for index, grouping in enumerate(groupings):
        column = quote_ident(grouping["feature"])
        if is_numeric_kind(grouping["kind"]):
            base_group_columns.append(f"TRY_CAST({column} AS DOUBLE) AS __group{index}_raw")
        else:
            base_group_columns.append(f"{column} AS __group{index}_raw")
    base_group_sql = "".join(f",\n    {column}" for column in base_group_columns)

    tail_bound_indices = {
        index
        for index, grouping in enumerate(groupings)
        if grouping_uses_tail_bounds(grouping)
    }
    bounds_columns = []
    for index, grouping in enumerate(groupings):
        if index in tail_bound_indices:
            tail_fraction = float(grouping["tail_percent"]) / 100
            bounds_columns.extend(
                [
                    f"quantile_cont(__group{index}_raw, {tail_fraction}) AS __group{index}_lower",
                    f"quantile_cont(__group{index}_raw, {1 - tail_fraction}) AS __group{index}_upper",
                ]
            )
        else:
            bounds_columns.extend(
                [
                    f"CAST(NULL AS DOUBLE) AS __group{index}_lower",
                    f"CAST(NULL AS DOUBLE) AS __group{index}_upper",
                ]
            )
    bounds_sql = ",\n    ".join(bounds_columns)
    bounds_cte = (
        f""",
bounds AS (
  SELECT
    {bounds_sql}
  FROM base
)"""
        if tail_bound_indices
        else ""
    )
    bounds_join = "\n  CROSS JOIN bounds" if tail_bound_indices else ""

    quantile_columns = []
    for index, grouping in enumerate(groupings):
        if grouping["quantile_mode"] == "quantile":
            count = normalise_quantile_count(grouping["band_width"])
            quantile_columns.append(
                f"""CASE WHEN __group{index}_raw IS NULL THEN NULL ELSE
      NTILE({count}) OVER (
        PARTITION BY (__group{index}_raw IS NULL)
        ORDER BY __group{index}_raw, __rownum
      )
    END AS __group{index}_quantile"""
            )
        else:
            quantile_columns.append(f"CAST(NULL AS BIGINT) AS __group{index}_quantile")
    quantile_sql = ",\n    ".join(quantile_columns)

    prepared_group_columns = []
    for index, grouping in enumerate(groupings):
        prepared_group_columns.extend(
            group_sql_columns(
                grouping,
                index,
                use_tail_bounds=index in tail_bound_indices,
            )
        )
    prepared_group_sql = ",\n    ".join(prepared_group_columns)

    valid_condition = line_bar_query.aliased_valid_condition(
        len(responses),
        bool(denominator.get("column")),
    )
    metric_sql = line_bar_query.grouped_metric_sql(responses)
    value_sql = line_bar_query.grouped_value_sql(responses)
    categorical_tail_indices = {
        index
        for index, grouping in enumerate(groupings)
        if grouping_uses_categorical_tail(grouping)
    }
    categorical_tail_ctes = "".join(
        categorical_tail_mapping_sql(index, float(groupings[index]["tail_percent"]))
        for index in sorted(categorical_tail_indices)
    )
    mapping_joins = "".join(
        f"""
  JOIN group{index}_map
    ON weighted.group{index}_key IS NOT DISTINCT FROM group{index}_map.source_group_key
   AND weighted.group{index}_label IS NOT DISTINCT FROM group{index}_map.source_group_label"""
        for index in sorted(categorical_tail_indices)
    )
    mapped_group_columns = []
    for index in range(2):
        if index in categorical_tail_indices:
            mapped_group_columns.extend(
                [
                    f"group{index}_map.final_group_key AS final_group{index}_key",
                    f"group{index}_map.final_group_label AS final_group{index}_label",
                    f"group{index}_map.final_group_sort AS final_group{index}_sort",
                ]
            )
        else:
            mapped_group_columns.extend(
                [
                    f"weighted.group{index}_key AS final_group{index}_key",
                    f"weighted.group{index}_label AS final_group{index}_label",
                    f"weighted.group{index}_sort AS final_group{index}_sort",
                ]
            )
    mapped_group_sql = ",\n    ".join(mapped_group_columns)
    return f"""
WITH base AS (
  SELECT
    ROW_NUMBER() OVER () AS __rownum{source_columns}{base_group_sql}{response_columns},
    {denominator_sql} AS __denominator_value
  FROM {relation}{where_sql}
),
quantiled AS (
  SELECT
    base.*,
    {quantile_sql}
  FROM base
){bounds_cte},
prepared AS (
  SELECT
    quantiled.*,
    {prepared_group_sql}
  FROM quantiled{bounds_join}
),
weighted AS (
  SELECT
    *,
    CASE WHEN {valid_condition} THEN __denominator_value ELSE NULL END AS __weight_value
  FROM prepared
){categorical_tail_ctes},
mapped AS (
  SELECT
    weighted.*,
    {mapped_group_sql}
  FROM weighted{mapping_joins}
),
grouped AS (
  SELECT
    final_group0_key AS group0_key,
    final_group0_label AS group0_label,
    MIN(final_group0_sort) AS group0_sort,
    MIN(group0_raw_start) AS group0_start,
    MAX(group0_raw_end) AS group0_end,
    final_group1_key AS group1_key,
    final_group1_label AS group1_label,
    MIN(final_group1_sort) AS group1_sort,
    MIN(group1_raw_start) AS group1_start,
    MAX(group1_raw_end) AS group1_end,
    COALESCE(SUM(__weight_value), 0) AS volume,
    COUNT(*) AS row_count
    {metric_sql}
  FROM mapped
  GROUP BY final_group0_key, final_group0_label, final_group1_key, final_group1_label
),
grouped_values AS (
  SELECT
    *{value_sql}
  FROM grouped
),
sort_ready AS (
  SELECT * FROM grouped_values
)"""


def grouping_uses_tail_bounds(grouping: dict[str, Any]) -> bool:
    return (
        float(grouping["tail_percent"]) > 0
        and is_numeric_kind(grouping["kind"])
        and grouping["quantile_mode"] == "off"
        and parse_positive_float(grouping["band_width"]) is not None
    )


def grouping_uses_categorical_tail(grouping: dict[str, Any]) -> bool:
    return (
        float(grouping["tail_percent"]) > 0
        and not is_numeric_kind(grouping["kind"])
        and grouping["kind"] not in {"date", "datetime"}
    )


def categorical_tail_mapping_sql(index: int, tail_percent: float) -> str:
    threshold_fraction = tail_percent / 100
    return f""",
group{index}_marginal AS (
  SELECT
    group{index}_key AS source_group_key,
    group{index}_label AS source_group_label,
    MIN(group{index}_sort) AS source_group_sort,
    COALESCE(SUM(__weight_value), 0) AS volume
  FROM weighted
  GROUP BY group{index}_key, group{index}_label
),
group{index}_tail_summary AS (
  SELECT
    COALESCE(SUM(volume), 0) AS total_volume,
    COUNT(*) AS group_count
  FROM group{index}_marginal
),
group{index}_tail_marked AS (
  SELECT
    *,
    volume <= (
      SELECT total_volume * {threshold_fraction}
      FROM group{index}_tail_summary
    ) AS is_rare
  FROM group{index}_marginal
),
group{index}_tail_stats AS (
  SELECT
    SUM(CASE WHEN is_rare THEN 1 ELSE 0 END) AS rare_count,
    MIN(CASE WHEN is_rare THEN source_group_sort ELSE NULL END) AS rare_sort
  FROM group{index}_tail_marked
),
group{index}_map AS (
  SELECT
    source_group_key,
    source_group_label,
    CASE
      WHEN (SELECT group_count FROM group{index}_tail_summary) >= 3
        AND is_rare
        AND (SELECT rare_count FROM group{index}_tail_stats) > 1
      THEN 'tail:other'
      ELSE source_group_key
    END AS final_group_key,
    CASE
      WHEN (SELECT group_count FROM group{index}_tail_summary) >= 3
        AND is_rare
        AND (SELECT rare_count FROM group{index}_tail_stats) > 1
      THEN 'Other'
      ELSE source_group_label
    END AS final_group_label,
    CASE
      WHEN (SELECT group_count FROM group{index}_tail_summary) >= 3
        AND is_rare
        AND (SELECT rare_count FROM group{index}_tail_stats) > 1
      THEN (SELECT rare_sort FROM group{index}_tail_stats)
      ELSE source_group_sort
    END AS final_group_sort
  FROM group{index}_tail_marked
)"""


def group_sql_columns(
    grouping: dict[str, Any],
    index: int,
    *,
    use_tail_bounds: bool = False,
) -> list[str]:
    raw = f"__group{index}_raw"
    quantile = f"__group{index}_quantile"
    if grouping["quantile_mode"] == "quantile":
        return [
            f"{quantile} AS group{index}_key",
            f"CASE WHEN {quantile} IS NULL THEN 'Missing' ELSE 'Q' || CAST({quantile} AS VARCHAR) END AS group{index}_label",
            f"COALESCE({quantile}, 1000001) AS group{index}_sort",
            f"{raw} AS group{index}_raw_start",
            f"{raw} AS group{index}_raw_end",
        ]
    if is_numeric_kind(grouping["kind"]):
        lower = f"__group{index}_lower"
        upper = f"__group{index}_upper"
        width = parse_positive_float(grouping["band_width"])
        if width:
            grouped_value = (
                f"GREATEST({lower}, LEAST({upper}, {raw}))"
                if use_tail_bounds
                else raw
            )
            grouped = (
                f"CASE WHEN {raw} IS NULL THEN NULL "
                f"ELSE FLOOR({grouped_value} / {float(width)}) * {float(width)} END"
            )
        else:
            grouped = raw
        label_cast = (
            f"CAST(TRY_CAST({grouped} AS BIGINT) AS VARCHAR)"
            if grouping["kind"] == "integer" and not width
            else f"CAST({grouped} AS VARCHAR)"
        )
        return [
            f"{grouped} AS group{index}_key",
            f"CASE WHEN {raw} IS NULL THEN '(missing)' ELSE {label_cast} END AS group{index}_label",
            f"{grouped} AS group{index}_sort",
            f"{raw} AS group{index}_raw_start",
            f"{raw} AS group{index}_raw_end",
        ]
    if grouping["kind"] in {"date", "datetime"}:
        bucket = grouping["date_bucket"]
        grouped = f"DATE_TRUNC('{bucket}', {raw})" if bucket in DATE_BUCKETS else raw
        if bucket in {"day", "week", "month", "year"}:
            grouped = f"CAST({grouped} AS DATE)"
        return [
            f"{grouped} AS group{index}_key",
            f"CASE WHEN {grouped} IS NULL THEN '(missing)' ELSE CAST({grouped} AS VARCHAR) END AS group{index}_label",
            f"{grouped} AS group{index}_sort",
            f"CAST(NULL AS DOUBLE) AS group{index}_raw_start",
            f"CAST(NULL AS DOUBLE) AS group{index}_raw_end",
        ]
    grouped = f"COALESCE(CAST({raw} AS VARCHAR), '(missing)')"
    return [
        f"{grouped} AS group{index}_key",
        f"{grouped} AS group{index}_label",
        f"{grouped} AS group{index}_sort",
        f"CAST(NULL AS DOUBLE) AS group{index}_raw_start",
        f"CAST(NULL AS DOUBLE) AS group{index}_raw_end",
    ]


def fetch_chart_rows(dataset: Dataset, result: dict[str, Any]) -> dict[str, Any]:
    max_groups = chart_max_groups(result)
    sql = f"""
{result['grouped_sql']},
group_total AS (
  SELECT COUNT(*) AS group_count FROM sort_ready
),
page_rows AS (
  SELECT *, TRUE AS __has_row
  FROM sort_ready
  WHERE (SELECT group_count FROM group_total) <= {max_groups}
)
SELECT
  group_total.group_count AS __group_count,
  page_rows.*
FROM group_total
LEFT JOIN page_rows ON TRUE
ORDER BY
  page_rows.group0_sort IS NULL,
  page_rows.group0_sort,
  LOWER(CAST(page_rows.group0_label AS VARCHAR)),
  page_rows.group1_sort IS NULL,
  page_rows.group1_sort,
  LOWER(CAST(page_rows.group1_label AS VARCHAR))
"""
    rows = fetch_dict_rows(dataset, sql)
    group_count = int(rows[0].get("__group_count") or 0) if rows else 0
    display_rows = [
        normalise_row(row, result["responses"], result["groupings"])
        for row in rows
        if row.get("__has_row")
    ]
    return {"group_count": group_count, "rows": display_rows}


def fetch_table_rows(
    dataset: Dataset,
    result: dict[str, Any],
    *,
    page: int,
    page_size: int,
) -> dict[str, Any]:
    from . import query as line_bar_query

    search = str(result["request"].get("tableSearch") or "").strip()
    search_sql = table_search_condition(search, result["groupings"])
    summary_selects = line_bar_query.table_summary_selects(result["responses"])
    sql = f"""
{result['grouped_sql']},
group_total AS (
  SELECT COUNT(*) AS group_count FROM sort_ready
),
matched AS (
  SELECT *
  FROM sort_ready
  WHERE {search_sql}
),
match_summary AS (
  SELECT
    COUNT(*) AS match_count,
    COALESCE(SUM(volume), 0) AS summary_volume,
    COALESCE(SUM(row_count), 0) AS summary_row_count{summary_selects}
  FROM matched
),
page_info_base AS (
  SELECT
    group_total.group_count,
    match_summary.match_count,
    GREATEST(1, CAST(CEIL(CAST(match_summary.match_count AS DOUBLE) / {page_size}) AS BIGINT)) AS page_count
  FROM group_total
  CROSS JOIN match_summary
),
page_info AS (
  SELECT
    group_count,
    match_count,
    page_count,
    LEAST({page}, page_count) AS page
  FROM page_info_base
),
numbered AS (
  SELECT
    matched.*,
    ROW_NUMBER() OVER (
      ORDER BY
        group0_sort IS NULL,
        group0_sort,
        LOWER(CAST(group0_label AS VARCHAR)),
        group1_sort IS NULL,
        group1_sort,
        LOWER(CAST(group1_label AS VARCHAR))
    ) AS __row_index
  FROM matched
),
page_rows AS (
  SELECT numbered.*, TRUE AS __has_row
  FROM numbered
  CROSS JOIN page_info
  WHERE numbered.__row_index > (page_info.page - 1) * {page_size}
    AND numbered.__row_index <= page_info.page * {page_size}
)
SELECT
  page_info.group_count AS __group_count,
  page_info.match_count AS __match_count,
  page_info.page_count AS __page_count,
  page_info.page AS __page,
  match_summary.*,
  page_rows.*
FROM page_info
CROSS JOIN match_summary
LEFT JOIN page_rows ON TRUE
ORDER BY page_rows.__row_index NULLS LAST
"""
    rows = fetch_dict_rows(dataset, sql)
    first = rows[0] if rows else {}
    display_rows = [
        normalise_row(row, result["responses"], result["groupings"])
        for row in rows
        if row.get("__has_row")
    ]
    return {
        "search": search,
        "page": int(first.get("__page") or 1),
        "page_count": int(first.get("__page_count") or 1),
        "match_count": int(first.get("__match_count") or 0),
        "group_count": int(first.get("__group_count") or 0),
        "rows": display_rows,
        "summary": line_bar_query.table_summary_from_sql_row(first, result["responses"]),
    }


def fetch_dict_rows(dataset: Dataset, sql: str) -> list[dict[str, Any]]:
    cursor = dataset.con.execute(sql)
    columns = [description[0] for description in cursor.description]
    return [dict(zip(columns, row)) for row in cursor.fetchall()]


def normalise_row(
    row: dict[str, Any],
    responses: list[dict[str, str]],
    groupings: list[dict[str, Any]],
) -> dict[str, Any]:
    normalised: dict[str, Any] = {
        "volume": json_number(row.get("volume")) or 0,
        "row_count": json_number(row.get("row_count")) or 0,
    }
    for index, grouping in enumerate(groupings):
        label = str(row.get(f"group{index}_label"))
        sort_value = row.get(f"group{index}_sort")
        if is_numeric_kind(grouping["kind"]):
            sort_value = json_number(sort_value)
            if (
                is_numeric_kind(grouping["kind"])
                and grouping["quantile_mode"] == "off"
                and grouping["band_width"]
            ):
                label = clean_numeric_label(sort_value, grouping["band_width"], label)
        normalised[f"group{index}"] = label
        normalised[f"group{index}_sort"] = sort_value
        normalised[f"group{index}_missing"] = row.get(f"group{index}_key") is None
        start = json_number(row.get(f"group{index}_start"))
        end = json_number(row.get(f"group{index}_end"))
        if start is not None:
            normalised[f"group{index}_start"] = start
        if end is not None:
            normalised[f"group{index}_end"] = end
    for index, _ in enumerate(responses):
        normalised[f"resp{index}_num"] = json_number(row.get(f"resp{index}_num"))
        normalised[f"resp{index}_den"] = json_number(row.get(f"resp{index}_den"))
        normalised[f"resp{index}"] = json_number(row.get(f"resp{index}"))
    return normalised


def clean_numeric_label(value: Any, band_width: Any, fallback: str) -> str:
    from . import query as line_bar_query

    decimal_places = line_bar_query.decimal_places_for_band_width(band_width)
    if decimal_places is None:
        return fallback
    label = line_bar_query.format_numeric_band_label(value, decimal_places)
    return label if label is not None else fallback


def table_search_condition(search: str, groupings: list[dict[str, Any]]) -> str:
    if not search:
        return "TRUE"
    needle = sql_literal(search.lower())
    candidates: list[str] = []
    for index, grouping in enumerate(groupings):
        label = f"group{index}_label"
        sort_value = f"group{index}_sort"
        candidates.extend(
            [
                f"contains(LOWER(CAST({label} AS VARCHAR)), {needle})",
                f"contains(LOWER(CAST({sort_value} AS VARCHAR)), {needle})",
            ]
        )
        if is_numeric_kind(grouping["kind"]):
            candidates.extend(
                [
                    f"contains(LOWER(format('{{:,.12f}}', TRY_CAST({sort_value} AS DOUBLE))), {needle})",
                    f"contains(LOWER(format('{{:,}}', TRY_CAST({sort_value} AS BIGINT))), {needle})",
                ]
            )
    return "(" + " OR ".join(candidates) + ")"


def continuous_missing_warnings(
    rows: list[dict[str, Any]],
    groupings: list[dict[str, Any]],
) -> list[str]:
    warnings: list[str] = []
    for index, grouping in enumerate(groupings):
        if not grouping["continuous"]:
            continue
        count = sum(int(row.get("row_count") or 0) for row in rows if row.get(f"group{index}_missing"))
        if count:
            warnings.append(
                f"{grouping['feature']}: omitted {count:,} rows with missing values from the two-feature chart; "
                "they remain available in Table view."
            )
    return warnings


def normalise_tail_percent(value: Any) -> float:
    if value is None or str(value).strip() == "":
        return 0.0
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        numeric = math.nan
    if numeric == 0:
        return 0.0
    parsed = parse_positive_float(value)
    if parsed is None:
        return 0.0
    return min(49.0, max(0.0, float(parsed)))


def normalise_quantile_count(value: Any) -> int:
    parsed = parse_positive_float(value)
    if parsed is None:
        return 1
    return min(1000, max(1, int(math.floor(parsed + 0.5))))


def use_quantiles(value: Any) -> bool:
    if value is True:
        return True
    if value is False or value is None:
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "on", "quantile", "quantiles"}


def boolean_flag(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def normalise_date_bucket(value: Any) -> str:
    bucket = str(value or "none").strip().lower()
    return bucket if bucket in DATE_BUCKETS else "none"


def positive_int(value: Any, default: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        parsed = default
    return max(1, parsed)


def sortable_value(value: Any) -> tuple[int, Any]:
    if value is None:
        return (2, "")
    number = json_number(value)
    if number is not None:
        return (0, number)
    return (1, str(value).lower())


__all__ = ["chart", "has_two_groupings", "table"]
