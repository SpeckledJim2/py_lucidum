from __future__ import annotations

import math
from decimal import Decimal, InvalidOperation
from typing import Any

from py_lucidum.core import (
    ColumnInfo,
    complete_source_relation_context,
    Dataset,
    ModelPredictionSource,
    build_denominator_summary_sql,
    build_response_summary_sql,
    denominator_exclusion_warnings,
    denominator_valid_condition,
    denominator_warnings,
    has_denominator_column,
    normalise_denominator,
    normalise_denominator_source,
    is_numeric_kind,
    json_number,
    parse_positive_float,
    quote_ident,
    response_parts,
    response_summary,
    sql_literal,
    summarize_denominator,
    suggested_band_width,
    weighted_value_sql,
)
from py_lucidum.tools.gbm.shap import FLAME_PERCENTILES as SHAP_RIBBON_PERCENTILES
from py_lucidum.tools.gbm.shap import percentile_key, percentile_selects
from py_lucidum.tools.gbm.validation import CROSS_ENTROPY_OBJECTIVES, LOG_LINK_OBJECTIVES


BINARY_LINK_OBJECTIVES = {"binary", *CROSS_ENTROPY_OBJECTIVES}
DEFAULT_MAX_GROUPS = 10000
DEFAULT_TABLE_PAGE_SIZE = 10000
DATE_BUCKETS = {"hour", "day", "week", "month", "year"}
DATE_BUCKET_INTERVALS = {
    "hour": "INTERVAL '1 hour'",
    "day": "INTERVAL '1 day'",
    "week": "INTERVAL '1 week'",
    "month": "INTERVAL '1 month'",
    "year": "INTERVAL '1 year'",
}
EMPTY_PERIOD_VALUES = {"show", "skip"}
MISSING_VALUES = {"show", "hide"}


def overlarge_chart_message(max_groups: int, group_label: str = "x-axis groups") -> str:
    return (
        f"More than {max_groups:,} {group_label}; too many to plot. "
        "Use Table view to inspect all groups, or choose grouping, banding, or filtering."
    )


def request_with_single_grouping(request: dict[str, Any]) -> dict[str, Any]:
    raw_groupings = request.get("groupings")
    if not isinstance(raw_groupings, list) or len(raw_groupings) != 1:
        return request
    grouping = raw_groupings[0]
    if not isinstance(grouping, dict):
        return request
    return {
        **request,
        "x": grouping.get("feature"),
        "xSource": grouping.get("source"),
        "bandWidth": grouping.get("bandWidth", 0),
        "quantileMode": grouping.get("quantileMode", "off"),
        "dateBucket": grouping.get("dateBucket", "none"),
        "missings": grouping.get("missings", request.get("missings", "show")),
    }


def chart(dataset: Dataset, request: dict[str, Any], feature_spec: Any | None = None) -> dict[str, Any]:
    from .two_feature import chart as two_feature_chart
    from .two_feature import has_two_groupings

    if has_two_groupings(request):
        return two_feature_chart(dataset, request)
    request = request_with_single_grouping(request)
    with dataset.lock:
        result = build_grouped_result(dataset, request, feature_spec=feature_spec, include_partial_dependence=True)
        chart_result = fetch_chart_rows(dataset, result)
        chart_rows = chart_result["rows"]
        group_count = chart_result["group_count"]
        max_groups = normalise_positive_int(request.get("maxGroups"), DEFAULT_MAX_GROUPS)
        chart_too_large = group_count > max_groups
        transform = str(request.get("transform") or "none")
        warnings = list(result["warnings"])
        if chart_too_large:
            warnings.append(overlarge_chart_message(max_groups))
        transform_metadata = transform_metadata_for_result(dataset, result, transform, warnings)
        glm_overlay_context = build_glm_overlay_chart_context(
            result,
            chart_rows,
            transform_metadata=transform_metadata,
        )
        display_rows, transform_metadata = apply_transform(
            chart_rows,
            result["responses"],
            transform,
            result["sigma_multiplier"],
            warnings,
            x_kind=result["x_group_kind"],
            base=request.get("base"),
            band_width=request.get("bandWidth"),
            transform_metadata=transform_metadata,
        )
        partial_dependence = None if chart_too_large else result["partial_dependence"]
        if partial_dependence:
            transform_partial_dependence_overlay(
                partial_dependence,
                transform,
                warnings,
                x_kind=result["x_group_kind"],
                base=request.get("base"),
                band_width=request.get("bandWidth"),
            )
            order_partial_dependence_rows(partial_dependence, chart_rows)
            warnings.extend(partial_dependence_warnings(partial_dependence))

        payload = {
            "x": result["x_col"],
            "x_kind": result["x_kind"],
            "x_group_kind": result["x_group_kind"],
            "date_bucket": result["date_bucket"],
            "empty_periods": result["empty_periods"],
            "missings": result["missings"],
            "source": result["source_id"],
            "row_count": result["row_count"],
            "filtered_row_count": result["filtered_row_count"],
            "filter": result["filter_sql"],
            "group_count": group_count,
            "max_groups": max_groups,
            "groups_truncated": group_count > max_groups,
            "responses": [
                {"label": r["label"], "numerator": r["numerator"], **({"source": r["source"]} if r.get("source") else {})}
                for r in result["responses"]
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
            "rows": display_rows,
            "exclusion_warnings": result["exclusion_warnings"],
            "warnings": warnings,
            "transform": transform_metadata,
            "glm_overlay_context": glm_overlay_context,
        }
        if partial_dependence:
            payload["partial_dependence"] = partial_dependence
        return payload


def build_glm_overlay_chart_context(
    result: dict[str, Any],
    chart_rows: list[dict[str, Any]],
    *,
    transform_metadata: dict[str, Any],
) -> dict[str, Any]:
    field_sources = result.get("field_sources") if isinstance(result.get("field_sources"), dict) else {}
    response_sources = field_sources.get("responses") if isinstance(field_sources.get("responses"), list) else []
    summaries = result.get("response_summaries") if isinstance(result.get("response_summaries"), list) else []
    responses: list[dict[str, Any]] = []
    model_id = ""
    for index, response in enumerate(result["responses"]):
        source = (
            str(response_sources[index])
            if index < len(response_sources) and response_sources[index]
            else str(response.get("source") or result.get("source_id") or "dataset")
        )
        summary = summaries[index] if index < len(summaries) and isinstance(summaries[index], dict) else {}
        item = {
            "label": response["label"],
            "numerator": response["numerator"],
            "source": source,
            "value": json_number(summary.get("value")),
            "numerator_total": json_number(summary.get("numerator")),
            "denominator_total": json_number(summary.get("denominator")),
        }
        responses.append(item)
        if response["numerator"] == "glm_prediction" and source.startswith("glm:") and source.endswith(":predictions"):
            model_id = source[len("glm:") : -len(":predictions")]

    points: list[dict[str, Any]] = []
    for display_order, row in enumerate(chart_rows):
        raw_points = row.get("glm_overlay_points")
        if not isinstance(raw_points, list):
            continue
        for raw_point in raw_points:
            if not isinstance(raw_point, dict):
                continue
            points.append(
                {
                    **raw_point,
                    "final_x": row.get("x"),
                    "final_x_sort": row.get("x_sort"),
                    "final_original_order": int(row.get("original_order") or raw_point.get("final_original_order") or 0),
                    "final_is_tail": bool(row.get("is_tail")),
                    "final_display_order": display_order,
                }
            )
    x_group_kind = str(result.get("x_group_kind") or "")
    return {
        "eligible": bool(
            model_id
            and points
            and x_group_kind in {"categorical", "integer", "numeric", "quantile"}
        ),
        "model_id": model_id,
        "feature": result.get("x_col"),
        "x_kind": result.get("x_kind"),
        "x_group_kind": x_group_kind,
        "points": points,
        "responses": responses,
        "denominator": {
            **dict(result.get("denominator") or {}),
            "source": result.get("denominator_source"),
        },
        "missing_response_rows": json_number((result.get("denominator_summary") or {}).get("missing_response_rows")),
        "transform": transform_metadata,
        "request": glm_overlay_request_context(result.get("request") or {}),
    }


def glm_overlay_request_context(request: dict[str, Any]) -> dict[str, Any]:
    return {
        str(key): value
        for key, value in request.items()
        if str(key) != "partialDependence"
    }


def glm_overlay(
    dataset: Dataset,
    payload: dict[str, Any],
    feature_spec: Any | None = None,
) -> dict[str, Any]:
    from .two_feature import has_two_groupings

    raw_request = payload.get("request")
    chart_context_payload = payload.get("chart_context")
    if not isinstance(raw_request, dict) or not isinstance(chart_context_payload, dict):
        raise ValueError("The current Line and Bar chart context is required.")
    if has_two_groupings(raw_request):
        raise ValueError("GLM overlay is available for one-feature charts.")
    request = request_with_single_grouping(raw_request)
    if partial_dependence_mode(request.get("partialDependence")) != "glm":
        raise ValueError("Choose GLM partial dependence.")
    if glm_overlay_request_context(request) != chart_context_payload.get("request"):
        raise ValueError("The Line and Bar chart changed; refresh it before adding the GLM overlay.")

    schema = dataset.valid_schema_columns()
    columns = {column.name: column for column in schema}
    x_col = str(request.get("x") or "")
    if x_col not in columns:
        raise ValueError("Choose a valid x-axis feature")
    x_kind = columns[x_col].kind
    if str(chart_context_payload.get("feature") or "") != x_col:
        raise ValueError("The Line and Bar chart changed; refresh it before adding the GLM overlay.")
    if str(chart_context_payload.get("x_kind") or "") != x_kind:
        raise ValueError("The Line and Bar chart schema changed; refresh it before adding the GLM overlay.")

    quantile_count = (
        normalise_quantile_count(request.get("bandWidth"))
        if use_quantiles(request.get("quantileMode")) and is_numeric_kind(x_kind)
        else None
    )
    date_bucket = normalise_date_bucket(request.get("dateBucket")) if x_kind in {"date", "datetime"} else "none"
    x_sql = build_x_sql(
        x_col=x_col,
        kind=x_kind,
        band_width=request.get("bandWidth"),
        date_bucket=date_bucket,
        quantile_count=quantile_count,
    )
    x_group_kind = "quantile" if quantile_count else x_kind
    if str(chart_context_payload.get("x_group_kind") or "") != x_group_kind:
        raise ValueError("The Line and Bar grouping changed; refresh it before adding the GLM overlay.")

    denominator = dict(chart_context_payload.get("denominator") or {})
    filter_sql = dataset.normalise_filter(request.get("filter"))
    effective_filter_sql = line_bar_analysis_filter(
        filter_sql,
        [x_col] if normalise_missings(request.get("missings")) == "hide" else [],
    )
    overlay_request = {**request, "filter": effective_filter_sql}

    from py_lucidum.tools.glm.overlay import (
        build_glm_partial_dependence_overlay,
        glm_partial_dependence_model_id,
    )
    from py_lucidum.tools.glm.store import GlmModelStore

    store = GlmModelStore(dataset.path, dataset=dataset)
    model_id = glm_partial_dependence_model_id(overlay_request) or store.active_model_id()
    model_context: dict[str, Any] = {}
    if model_id:
        model_context = {
            "model_id": model_id,
            "source_id": store.source_id(model_id),
            "estimator_path": str(store.artifact_path(model_id, "estimator")),
            "manifest": store.manifest(model_id),
        }
    partial_dependence = build_glm_partial_dependence_overlay(
        dataset,
        overlay_request,
        feature_spec=feature_spec or {},
        x_col=x_col,
        x_sql=x_sql,
        x_group_kind=x_group_kind,
        denominator=denominator,
        chart_context=chart_context_payload,
        model_context=model_context,
        source_columns=[column.name for column in schema],
        kinds={column.name: column.kind for column in schema},
    )

    warnings: list[str] = []
    transform_partial_dependence_overlay(
        partial_dependence,
        str(request.get("transform") or "none"),
        warnings,
        x_kind=x_group_kind,
        base=request.get("base"),
        band_width=request.get("bandWidth"),
    )
    final_rows = glm_overlay_context_final_rows(chart_context_payload)
    order_partial_dependence_rows(partial_dependence, final_rows)
    warnings.extend(partial_dependence_warnings(partial_dependence))
    return {
        "partial_dependence": partial_dependence,
        "warnings": warnings,
    }


def glm_overlay_context_final_rows(chart_context: dict[str, Any]) -> list[dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    for point in chart_context.get("points") or []:
        if not isinstance(point, dict):
            continue
        label = str(point.get("final_x") or "")
        rows.setdefault(
            label,
            {
                "x": label,
                "x_sort": point.get("final_x_sort"),
                "original_order": int(point.get("final_original_order") or 0),
                "display_order": int(point.get("final_display_order") or 0),
                "volume": 0.0,
                "is_tail": bool(point.get("final_is_tail")),
            },
        )
        rows[label]["volume"] += float(json_number(point.get("volume")) or 0)
    return sorted(rows.values(), key=lambda row: int(row.get("display_order") or 0))


def table(dataset: Dataset, request: dict[str, Any], feature_spec: Any | None = None) -> dict[str, Any]:
    from .two_feature import has_two_groupings
    from .two_feature import table as two_feature_table

    if has_two_groupings(request):
        return two_feature_table(dataset, request)
    request = request_with_single_grouping(request)
    with dataset.lock:
        result = build_grouped_result(dataset, request, feature_spec=feature_spec, include_partial_dependence=False)
        page_size = normalise_positive_int(request.get("tablePageSize"), DEFAULT_TABLE_PAGE_SIZE)
        page = normalise_positive_int(request.get("tablePage"), 1)
        table_result = fetch_table_rows(dataset, result, page=page, page_size=page_size)
        page_rows = table_result["rows"]
        transform = str(request.get("transform") or "none")
        warnings = list(result["warnings"])
        transform_metadata = transform_metadata_for_result(dataset, result, transform, warnings)
        display_rows, transform_metadata = apply_transform(
            page_rows,
            result["responses"],
            transform,
            result["sigma_multiplier"],
            warnings,
            x_kind=result["x_group_kind"],
            base=request.get("base"),
            band_width=request.get("bandWidth"),
            transform_metadata=transform_metadata,
        )
        return {
            "x": result["x_col"],
            "x_kind": result["x_kind"],
            "x_group_kind": result["x_group_kind"],
            "date_bucket": result["date_bucket"],
            "empty_periods": result["empty_periods"],
            "missings": result["missings"],
            "source": result["source_id"],
            "row_count": result["row_count"],
            "filtered_row_count": result["filtered_row_count"],
            "filter": result["filter_sql"],
            "responses": [
                {"label": r["label"], "numerator": r["numerator"], **({"source": r["source"]} if r.get("source") else {})}
                for r in result["responses"]
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
            "rows": display_rows,
            "summary": transform_table_summary(table_result["summary"], result["responses"], transform, transform_metadata),
            "exclusion_warnings": result["exclusion_warnings"],
            "warnings": warnings,
            "transform": transform_metadata,
            "table": {
                "search": table_result["search"],
                "page": table_result["page"],
                "page_size": page_size,
                "page_count": table_result["page_count"],
                "match_count": table_result["match_count"],
                "group_count": table_result["group_count"],
            },
        }


def banding_suggestion(dataset: Dataset, request: dict[str, Any]) -> dict[str, Any]:
    with dataset.lock:
        feature = str(request.get("feature") or "").strip()
        if not feature:
            raise ValueError("Choose a numeric feature")
        if uses_field_sources(request):
            context_request = {**request, "x": feature}
            context = chart_context(dataset, context_request)
            x_source = str((context.get("field_sources") or {}).get("x") or context["source_id"])
            filter_sql = dataset.normalise_filter_for_relation(request.get("filter"), context["relation"])
            suggestion = relation_band_suggestion_for_column(
                dataset,
                context["relation"],
                context["columns"],
                feature,
                filter_sql,
            )
            return {
                "feature": feature,
                "source": x_source,
                "band_suggestion": suggestion,
            }
        source = dataset.normalise_source(request.get("xSource") or request.get("source"))
        filter_sql = dataset.normalise_filter(request.get("filter"), source_id=source)
        suggestion = dataset.band_suggestion_for_column(source, feature, filter_sql)
        return {
            "feature": feature,
            "source": source,
            "band_suggestion": suggestion,
        }


def relation_band_suggestion_for_column(
    dataset: Dataset,
    relation: str,
    columns: dict[str, ColumnInfo],
    feature: str,
    filter_sql: str = "",
    sample_limit: int = 100_000,
) -> float | int | None:
    column = columns.get(feature)
    if column is None:
        raise ValueError("Choose a valid feature for the selected data source")
    if not is_numeric_kind(column.kind):
        raise ValueError("Choose a numeric feature for banding")
    limit = max(1, min(int(sample_limit), 100_000))
    where_sql = f"WHERE ({filter_sql})" if filter_sql else ""
    raw = quote_ident(column.name)
    sql = f"""
WITH sample AS (
  SELECT TRY_CAST({raw} AS DOUBLE) AS value
  FROM {relation}
  {where_sql}
  LIMIT {limit}
)
SELECT
  STDDEV_SAMP(value) AS std,
  MIN(value) AS min_value,
  MAX(value) AS max_value
FROM sample
"""
    row = dataset.con.execute(sql).fetchone()
    if not row:
        return None
    stddev, min_value, max_value = row
    if column.kind == "integer" and min_value is not None and max_value is not None:
        if max_value - min_value < 120:
            return 1
    return suggested_band_width(stddev)


def build_grouped_result(
    dataset: Dataset,
    request: dict[str, Any],
    *,
    feature_spec: Any | None = None,
    include_partial_dependence: bool,
) -> dict[str, Any]:
    context = chart_context(dataset, request)
    relation = context["relation"]
    columns = context["columns"]
    x_col = str(request.get("x") or "")
    if x_col not in columns:
        raise ValueError("Choose a valid x-axis feature")

    filter_sql = dataset.normalise_filter_for_relation(request.get("filter"), relation)
    responses = normalise_responses(request.get("responses"), columns)
    denominator = normalise_denominator(request.get("denominator", request.get("weight")), columns)
    denominator_source = normalise_denominator_source(
        dataset,
        request.get("denominatorSource"),
        request.get("denominator", request.get("weight")),
    )
    x_info = columns[x_col]
    missings = normalise_missings(request.get("missings"))
    effective_filter_sql = line_bar_analysis_filter(
        filter_sql,
        [x_col] if missings == "hide" else [],
    )
    date_bucket = normalise_date_bucket(request.get("dateBucket")) if x_info.kind in {"date", "datetime"} else "none"
    empty_periods = normalise_empty_periods(request.get("emptyPeriods"))
    quantile_count = (
        normalise_quantile_count(request.get("bandWidth"))
        if use_quantiles(request.get("quantileMode")) and is_numeric_kind(x_info.kind)
        else None
    )
    x_sql = build_x_sql(
        x_col=x_col,
        kind=x_info.kind,
        band_width=request.get("bandWidth"),
        date_bucket=date_bucket,
        quantile_count=quantile_count,
    )
    x_group_kind = "quantile" if quantile_count else x_info.kind
    sigma_multiplier = float(request.get("sigma") or 0)
    include_sigma = sigma_multiplier > 0 and len(responses) >= 2
    denominator_summary = relation_denominator_summary(
        dataset,
        relation,
        responses,
        denominator,
        effective_filter_sql,
    )
    partial_dependence = None
    if include_partial_dependence or str(request.get("sort") or "alpha") == "shap":
        partial_dependence = build_partial_dependence_overlay(
            dataset,
            {**request, "filter": effective_filter_sql},
            feature_spec=feature_spec or {},
            columns=columns,
            x_col=x_col,
            x_sql=x_sql,
            x_group_kind=x_group_kind,
            responses=responses,
            denominator=denominator,
        )
    grouped_sql = build_grouped_pipeline_sql(
        relation=relation,
        x_col=x_col,
        x_sql=x_sql,
        responses=responses,
        denominator=denominator,
        include_sigma=include_sigma,
        filter_sql=effective_filter_sql,
        x_group_kind=x_group_kind,
        date_bucket=date_bucket,
        empty_periods=empty_periods,
        low_group=str(request.get("lowGroup") or "0"),
        sort=str(request.get("sort") or "alpha"),
        shap_medians=partial_dependence_medians(partial_dependence),
    )
    response_summaries = relation_response_summary(
        dataset,
        relation,
        responses,
        denominator,
        effective_filter_sql,
    )
    return {
        "source_id": context["source_id"],
        "field_sources": context.get("field_sources"),
        "x_col": x_col,
        "x_kind": x_info.kind,
        "x_group_kind": x_group_kind,
        "date_bucket": date_bucket,
        "empty_periods": empty_periods,
        "missings": missings,
        "row_count": context["row_count"],
        "filtered_row_count": relation_row_count(dataset, relation, effective_filter_sql),
        "filter_sql": filter_sql,
        "effective_filter_sql": effective_filter_sql,
        "responses": responses,
        "denominator": denominator,
        "denominator_source": denominator_source,
        "denominator_summary": denominator_summary,
        "response_summaries": response_summaries,
        "grouped_sql": grouped_sql,
        "sigma_multiplier": sigma_multiplier,
        "partial_dependence": partial_dependence,
        "exclusion_warnings": denominator_exclusion_warnings(denominator, denominator_summary, responses),
        "warnings": denominator_warnings(denominator, denominator_summary, responses),
        "request": request,
    }


def fetch_chart_rows(dataset: Dataset, result: dict[str, Any]) -> dict[str, Any]:
    max_groups = normalise_positive_int(result["request"].get("maxGroups"), DEFAULT_MAX_GROUPS)
    order_terms = sort_order_terms(result["x_group_kind"], str(result["request"].get("sort") or "alpha"))
    sql = f"""
{result['grouped_sql']},
group_total AS (
  SELECT COUNT(*) AS group_count FROM sort_ready
),
numbered AS (
  SELECT
    sort_ready.*,
    ROW_NUMBER() OVER (ORDER BY {order_terms}) AS __row_index
  FROM sort_ready
),
page_rows AS (
  SELECT
    *,
    TRUE AS __has_row
  FROM numbered
  WHERE (SELECT group_count FROM group_total) <= {max_groups}
    AND __row_index <= {max_groups}
)
SELECT
  group_total.group_count AS __group_count,
  page_rows.*
FROM group_total
LEFT JOIN page_rows ON TRUE
ORDER BY page_rows.__row_index NULLS LAST
"""
    rows = fetch_dict_rows(dataset, sql)
    group_count = int(rows[0].get("__group_count") or 0) if rows else 0
    display_rows = [
        normalise_grouped_sql_row(row, result["responses"])
        for row in rows
        if row.get("__has_row")
    ]
    if result["x_group_kind"] == "numeric":
        clean_numeric_band_labels(display_rows, result["x_group_kind"], result["request"].get("bandWidth"))
    return {"group_count": group_count, "rows": display_rows}


def fetch_table_rows(dataset: Dataset, result: dict[str, Any], *, page: int, page_size: int) -> dict[str, Any]:
    search = str(result["request"].get("tableSearch") or "").strip()
    order_terms = sort_order_terms(result["x_group_kind"], str(result["request"].get("sort") or "alpha"))
    search_sql = table_search_condition(search, result["x_kind"])
    summary_selects = table_summary_selects(result["responses"])
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
    ROW_NUMBER() OVER (ORDER BY {order_terms}) AS __row_index
  FROM matched
),
page_rows AS (
  SELECT
    numbered.*,
    TRUE AS __has_row
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
        normalise_grouped_sql_row(row, result["responses"])
        for row in rows
        if row.get("__has_row")
    ]
    if result["x_group_kind"] == "numeric":
        clean_numeric_band_labels(display_rows, result["x_group_kind"], result["request"].get("bandWidth"))
    return {
        "search": search,
        "page": int(first.get("__page") or 1),
        "page_count": int(first.get("__page_count") or 1),
        "match_count": int(first.get("__match_count") or 0),
        "group_count": int(first.get("__group_count") or 0),
        "rows": display_rows,
        "summary": table_summary_from_sql_row(first, result["responses"]),
    }


def fetch_dict_rows(dataset: Dataset, sql: str) -> list[dict[str, Any]]:
    cursor = dataset.con.execute(sql)
    columns = [description[0] for description in cursor.description]
    return [dict(zip(columns, row)) for row in cursor.fetchall()]


def normalise_grouped_sql_row(row: dict[str, Any], responses: list[dict[str, str]]) -> dict[str, Any]:
    normalised = {
        "x": str(row.get("x")),
        "x_sort": row.get("x_sort"),
        "original_order": int(row.get("original_order") or 0),
        "volume": json_number(row.get("volume")) or 0,
        "row_count": json_number(row.get("row_count")) or 0,
        "is_tail": bool(row.get("is_tail")),
        "sigma_se": json_number(row.get("sigma_se")),
        "valid_folds": json_number(row.get("valid_folds")),
        "sigma_folds": row.get("sigma_folds"),
    }
    x_start = json_number(row.get("x_start"))
    x_end = json_number(row.get("x_end"))
    if x_start is not None:
        normalised["x_start"] = x_start
    if x_end is not None:
        normalised["x_end"] = x_end
    overlay_points = row.get("glm_overlay_points")
    if isinstance(overlay_points, list):
        normalised["glm_overlay_points"] = [
            {
                "source_x": point.get("source_x"),
                "x_value": point.get("x_value"),
                "volume": json_number(point.get("volume")) or 0,
            }
            for point in overlay_points
            if isinstance(point, dict)
        ]
    for index, _ in enumerate(responses):
        normalised[f"resp{index}_num"] = json_number(row.get(f"resp{index}_num"))
        normalised[f"resp{index}_den"] = json_number(row.get(f"resp{index}_den"))
        normalised[f"resp{index}"] = json_number(row.get(f"resp{index}"))
    return normalised


def table_summary_selects(responses: list[dict[str, str]]) -> str:
    if not responses:
        return ""
    selects: list[str] = []
    for index, _ in enumerate(responses):
        selects.append(f"COALESCE(SUM(resp{index}_num), 0) AS summary_resp{index}_num")
        selects.append(f"COALESCE(SUM(resp{index}_den), 0) AS summary_resp{index}_den")
    return ",\n    " + ",\n    ".join(selects)


def table_summary_from_sql_row(row: dict[str, Any], responses: list[dict[str, str]]) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "volume": json_number(row.get("summary_volume")) or 0,
        "row_count": json_number(row.get("summary_row_count")) or 0,
        "responses": [],
    }
    for index, _ in enumerate(responses):
        numerator = json_number(row.get(f"summary_resp{index}_num")) or 0
        denominator = json_number(row.get(f"summary_resp{index}_den")) or 0
        summary["responses"].append(numerator / denominator if denominator else None)
    return summary


def transform_table_summary(
    summary: dict[str, Any],
    responses: list[dict[str, str]],
    transform: str,
    transform_metadata: dict[str, Any],
) -> dict[str, Any]:
    references = transform_metadata.get("values") if isinstance(transform_metadata, dict) else []
    transformed = {
        "volume": summary.get("volume", 0),
        "row_count": summary.get("row_count", 0),
        "responses": [],
    }
    for index, _ in enumerate(responses):
        summary_responses = summary.get("responses")
        value = summary_responses[index] if isinstance(summary_responses, list) and index < len(summary_responses) else None
        reference = json_number(references[index]) if isinstance(references, list) and index < len(references) else None
        transformed["responses"].append(transform_value(value, transform, reference))
    return transformed


def build_grouped_pipeline_sql(
    *,
    relation: str,
    x_col: str,
    x_sql: dict[str, str],
    responses: list[dict[str, str]],
    denominator: dict[str, str | None],
    include_sigma: bool,
    filter_sql: str,
    x_group_kind: str,
    date_bucket: str,
    empty_periods: str,
    low_group: str,
    sort: str,
    shap_medians: dict[str, float] | None,
) -> str:
    required_columns = required_grouped_columns(x_col, responses, denominator)
    source_selects = ",\n    ".join(quote_ident(column) for column in required_columns)
    source_select_sql = ",\n    " + source_selects if source_selects else ""
    where_sql = f"\n  WHERE ({filter_sql})" if filter_sql else ""
    response_selects = ",\n    ".join(
        f"TRY_CAST({quote_ident(response['numerator'])} AS DOUBLE) AS __resp{index}_value"
        for index, response in enumerate(responses)
    )
    denominator_select = denominator_value_select_sql(denominator)
    quantile_cte = ""
    keyed_from = "base"
    rownum_expr = "__rownum"
    x_bound_select = ""
    x_bound_agg = ",\n    NULL AS x_start,\n    NULL AS x_end"
    x_model_value_select = f",\n    {x_sql.get('model_value', x_sql['key'])} AS __x_model_value"
    x_value_agg = "MIN(__x_model_value) AS x_value"
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
        x_bound_select = f",\n    {x_sql['raw']} AS __x_raw_value"
        x_bound_agg = ",\n    MIN(__x_raw_value) AS x_start,\n    MAX(__x_raw_value) AS x_end"
        x_value_agg = "AVG(__x_raw_value) AS x_value"

    response_sql = f",\n    {response_selects}" if response_selects else ""
    valid_condition = aliased_valid_condition(len(responses), bool(denominator.get("column")))
    metric_sql = grouped_metric_sql(responses)
    value_sql = grouped_value_sql(responses)
    fold_select = f",\n    CAST(hash({rownum_expr}) % 20 AS INTEGER) AS __fold" if include_sigma else ""
    sigma_ctes = sigma_pipeline_ctes(include_sigma)
    final_rows_sql = final_rows_pipeline_sql(
        responses=responses,
        include_sigma=include_sigma,
        x_group_kind=x_group_kind,
        low_group=low_group,
    )
    display_rows_sql = date_display_rows_ctes(
        responses,
        date_bucket=date_bucket,
        empty_periods=empty_periods,
    )
    shap_sql = shap_medians_cte(sort, shap_medians or {})
    shap_join = "LEFT JOIN shap_medians ON display_rows.x = shap_medians.x" if sort == "shap" else ""
    shap_select = "shap_medians.median AS __shap_median" if sort == "shap" else "NULL AS __shap_median"
    return f"""
WITH base AS (
  SELECT
    ROW_NUMBER() OVER () AS __rownum{source_select_sql}
  FROM {relation}{where_sql}
){quantile_cte},
prepared AS (
  SELECT
    {rownum_expr} AS __rownum,
    {x_sql['key']} AS x_key,
    {x_sql['label']} AS x_label,
    {x_sql['sort']} AS x_sort{x_model_value_select}{x_bound_select}{fold_select}{response_sql},
    {denominator_select} AS __denominator_value
  FROM {keyed_from}
),
weighted AS (
  SELECT
    *,
    CASE WHEN {valid_condition} THEN __denominator_value ELSE NULL END AS __weight_value
  FROM prepared
),
initial_agg AS (
  SELECT
    x_key,
    x_label,
    MIN(x_sort) AS x_sort,
    {x_value_agg},
    MIN(__rownum) AS original_order,
    COALESCE(SUM(__weight_value), 0) AS volume,
    COUNT(*) AS row_count{x_bound_agg}
    {metric_sql}
  FROM weighted
  GROUP BY x_key, x_label
),
initial_values AS (
  SELECT
    *{value_sql}
  FROM initial_agg
)
{sigma_ctes}
{final_rows_sql}
{display_rows_sql}
{shap_sql},
sort_ready AS (
  SELECT
    display_rows.*,
    {shap_select}
  FROM display_rows
  {shap_join}
)"""


def required_grouped_columns(
    x_col: str,
    responses: list[dict[str, str]],
    denominator: dict[str, str | None],
) -> list[str]:
    required: list[str] = []
    for column in [x_col, *(response["numerator"] for response in responses), denominator.get("column")]:
        if column and column not in required:
            required.append(str(column))
    return required


def denominator_value_select_sql(denominator: dict[str, str | None]) -> str:
    column = denominator.get("column")
    if column:
        return f"TRY_CAST({quote_ident(str(column))} AS DOUBLE)"
    return "1"


def aliased_valid_condition(response_count: int, has_denominator: bool) -> str:
    checks = [f"__resp{index}_value IS NOT NULL" for index in range(response_count)]
    if has_denominator:
        checks.append("__denominator_value IS NOT NULL")
    return " AND ".join(checks) if checks else "TRUE"


def grouped_metric_sql(responses: list[dict[str, str]]) -> str:
    if not responses:
        return ""
    selects: list[str] = []
    for index, _ in enumerate(responses):
        selects.append(f"SUM(CASE WHEN __weight_value IS NOT NULL THEN __resp{index}_value ELSE NULL END) AS resp{index}_num")
        selects.append(f"COALESCE(SUM(__weight_value), 0) AS resp{index}_den")
    return ",\n    " + ",\n    ".join(selects)


def grouped_value_sql(responses: list[dict[str, str]]) -> str:
    if not responses:
        return ""
    values = [
        f"resp{index}_num / NULLIF(resp{index}_den, 0) AS resp{index}"
        for index, _ in enumerate(responses)
    ]
    return ",\n    " + ",\n    ".join(values)


def group_threshold_expression(value: str) -> tuple[bool, str]:
    raw = value.strip().lower()
    if raw in {"", "0", "none", "-"}:
        return False, "0"
    if raw.endswith("%"):
        parsed = parse_positive_float(raw[:-1])
        if not parsed:
            return False, "0"
        return True, f"total_volume * {float(parsed)} / 100"
    parsed = parse_positive_float(raw)
    if not parsed:
        return False, "0"
    return True, str(float(parsed))


def sigma_pipeline_ctes(include_sigma: bool) -> str:
    if not include_sigma:
        return ""
    return """,
initial_folds AS (
  SELECT
    x_key,
    x_label,
    __fold,
    SUM(CASE WHEN __weight_value IS NOT NULL THEN __resp0_value ELSE NULL END) AS resp0_num,
    COALESCE(SUM(__weight_value), 0) AS resp0_den,
    SUM(CASE WHEN __weight_value IS NOT NULL THEN __resp1_value ELSE NULL END) AS resp1_num,
    COALESCE(SUM(__weight_value), 0) AS resp1_den
  FROM weighted
  GROUP BY x_key, x_label, __fold
)"""


def final_sigma_ctes(include_sigma: bool, *, grouped: bool) -> str:
    if not include_sigma:
        return ""
    return ("""
,
final_folds AS (
  SELECT
    group_map.final_group_id,
    initial_folds.__fold,
    SUM(initial_folds.resp0_num) AS resp0_num,
    SUM(initial_folds.resp0_den) AS resp0_den,
    SUM(initial_folds.resp1_num) AS resp1_num,
    SUM(initial_folds.resp1_den) AS resp1_den
  FROM initial_folds
  INNER JOIN group_map
    ON initial_folds.x_key IS NOT DISTINCT FROM group_map.source_x_key
   AND initial_folds.x_label = group_map.source_x_label
  GROUP BY group_map.final_group_id, initial_folds.__fold
)""" if grouped else f"""
,
final_folds AS (
  SELECT
    {group_identity_id('x_key', 'x_label')} AS final_group_id,
    __fold,
    resp0_num,
    resp0_den,
    resp1_num,
    resp1_den
  FROM initial_folds
)""") + """
,
fold_values AS (
  SELECT
    *,
    resp0_num / NULLIF(resp0_den, 0) AS resp0,
    resp1_num / NULLIF(resp1_den, 0) AS resp1
  FROM final_folds
),
sigma AS (
  SELECT
    final_group_id,
    STDDEV_SAMP(resp0 - resp1) / SQRT(COUNT(*)) AS sigma_se,
    COUNT(*) AS valid_folds,
    LIST(struct_pack(
      fold := __fold,
      resp0_num := resp0_num,
      resp0_den := resp0_den,
      resp1_num := resp1_num,
      resp1_den := resp1_den
    )) AS sigma_folds
  FROM fold_values
  WHERE resp0 IS NOT NULL AND resp1 IS NOT NULL
  GROUP BY final_group_id
)"""


def final_rows_pipeline_sql(
    *,
    responses: list[dict[str, str]],
    include_sigma: bool,
    x_group_kind: str,
    low_group: str,
) -> str:
    enabled, threshold_expr = group_threshold_expression(low_group)
    if not enabled:
        return no_group_final_rows_sql(responses, include_sigma)
    if x_group_kind in {"integer", "numeric", "date", "datetime", "quantile"}:
        return ordered_group_final_rows_sql(responses, include_sigma, threshold_expr)
    return categorical_group_final_rows_sql(responses, include_sigma, threshold_expr)


def date_display_rows_ctes(
    responses: list[dict[str, str]],
    *,
    date_bucket: str,
    empty_periods: str,
) -> str:
    interval = DATE_BUCKET_INTERVALS.get(date_bucket) if empty_periods == "show" else None
    if interval is None:
        return """,
display_rows AS (
  SELECT * FROM final_rows
)"""

    response_selects = "".join(
        f""",
    CAST(0 AS DOUBLE) AS resp{index}_num,
    CAST(0 AS DOUBLE) AS resp{index}_den,
    CAST(NULL AS DOUBLE) AS resp{index}"""
        for index, _ in enumerate(responses)
    )
    period_label = "CAST(date_periods.period AS VARCHAR)"
    return f""",
date_bounds AS (
  SELECT
    MIN(x_sort) AS min_period,
    MAX(x_sort) AS max_period
  FROM initial_values
  WHERE x_sort IS NOT NULL
),
date_periods AS (
  SELECT generated.period
  FROM date_bounds
  CROSS JOIN LATERAL generate_series(
    date_bounds.min_period,
    date_bounds.max_period,
    {interval}
  ) AS generated(period)
),
empty_date_periods AS (
  SELECT
    {group_identity_id('date_periods.period', period_label)} AS final_group_id,
    {period_label} AS x,
    date_periods.period AS x_sort,
    ROW_NUMBER() OVER (ORDER BY date_periods.period) AS original_order,
    CAST(0 AS DOUBLE) AS volume,
    CAST(0 AS BIGINT) AS row_count,
    CAST(NULL AS DOUBLE) AS x_start,
    CAST(NULL AS DOUBLE) AS x_end,
    FALSE AS is_tail,
    NULL AS glm_overlay_points{response_selects},
    CAST(NULL AS DOUBLE) AS sigma_se,
    CAST(NULL AS BIGINT) AS valid_folds,
    NULL AS sigma_folds
  FROM date_periods
  WHERE NOT EXISTS (
    SELECT 1
    FROM initial_values
    WHERE initial_values.x_sort IS NOT DISTINCT FROM date_periods.period
  )
),
display_rows AS (
  SELECT * FROM final_rows
  UNION ALL
  SELECT * FROM empty_date_periods
)"""


def no_group_final_rows_sql(responses: list[dict[str, str]], include_sigma: bool) -> str:
    response_selects = final_response_selects("initial_values", responses)
    sigma_select = sigma_output_select("sigma") if include_sigma else null_sigma_output_select()
    sigma_join = f"LEFT JOIN sigma ON {group_identity_id('initial_values.x_key', 'initial_values.x_label')} = sigma.final_group_id" if include_sigma else ""
    sigma_ctes = cte_fragment(final_sigma_ctes(include_sigma, grouped=False))
    return f""",
{sigma_ctes + ',' if sigma_ctes else ''}
final_rows AS (
  SELECT
    {group_identity_id('initial_values.x_key', 'initial_values.x_label')} AS final_group_id,
    initial_values.x_label AS x,
    initial_values.x_sort,
    initial_values.original_order,
    initial_values.volume,
    initial_values.row_count,
    initial_values.x_start,
    initial_values.x_end,
    FALSE AS is_tail,
    [struct_pack(
      source_x := initial_values.x_label,
      x_value := initial_values.x_value,
      volume := initial_values.volume
    )] AS glm_overlay_points{response_selects}
    {sigma_select}
  FROM initial_values
  {sigma_join}
)"""


def ordered_group_final_rows_sql(
    responses: list[dict[str, str]],
    include_sigma: bool,
    threshold_expr: str,
) -> str:
    missing_union = f"""
  UNION ALL
  SELECT
    {group_identity_id('x_key', 'x_label')} AS source_group_id,
    x_key AS source_x_key,
    x_label AS source_x_label,
    {group_identity_id('x_key', 'x_label')} AS final_group_id,
    x_label AS final_label,
    x_sort AS final_x_sort,
    original_order AS final_original_order,
    FALSE AS final_is_tail
  FROM initial_values
  WHERE x_key IS NULL"""
    return f""",
group_totals AS (
  SELECT
    COALESCE(SUM(volume), 0) AS total_volume,
    COUNT(*) FILTER (WHERE x_key IS NOT NULL) AS candidate_group_count
  FROM initial_values
),
group_threshold AS (
  SELECT
    {threshold_expr} AS threshold_value,
    total_volume,
    candidate_group_count
  FROM group_totals
),
ordered_candidates AS (
  SELECT
    initial_values.*,
    SUM(volume) OVER (ORDER BY x_sort ASC NULLS LAST ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) AS low_cume
  FROM initial_values
  WHERE x_key IS NOT NULL
),
low_marked AS (
  SELECT
    *,
    low_cume <= (SELECT threshold_value FROM group_threshold) AS is_low
  FROM ordered_candidates
),
remaining_candidates AS (
  SELECT
    *,
    SUM(volume) OVER (ORDER BY x_sort DESC NULLS FIRST ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) AS high_cume
  FROM low_marked
  WHERE NOT is_low
),
high_marked AS (
  SELECT
    low_marked.*,
    COALESCE(remaining_candidates.high_cume <= (SELECT threshold_value FROM group_threshold), FALSE) AS is_high
  FROM low_marked
  LEFT JOIN remaining_candidates
    ON low_marked.x_key IS NOT DISTINCT FROM remaining_candidates.x_key
   AND low_marked.x_label = remaining_candidates.x_label
),
tail_counts AS (
  SELECT
    SUM(CASE WHEN is_low THEN 1 ELSE 0 END) AS low_count,
    SUM(CASE WHEN is_high THEN 1 ELSE 0 END) AS high_count
  FROM high_marked
),
low_tail_stats AS (
  SELECT
    MIN(x_sort) AS x_sort,
    MIN(original_order) AS original_order
  FROM high_marked
  WHERE is_low
),
high_tail_stats AS (
  SELECT
    MIN(x_sort) AS x_sort,
    MIN(original_order) AS original_order
  FROM high_marked
  WHERE is_high
),
group_map AS (
  SELECT
    {group_identity_id('x_key', 'x_label')} AS source_group_id,
    x_key AS source_x_key,
    x_label AS source_x_label,
    CASE
      WHEN (SELECT candidate_group_count FROM group_threshold) >= 3 AND is_low AND (SELECT low_count FROM tail_counts) > 1 THEN 'tail:low'
      WHEN (SELECT candidate_group_count FROM group_threshold) >= 3 AND is_high AND (SELECT high_count FROM tail_counts) > 1 THEN 'tail:high'
      ELSE {group_identity_id('x_key', 'x_label')}
    END AS final_group_id,
    CASE
      WHEN (SELECT candidate_group_count FROM group_threshold) >= 3 AND is_low AND (SELECT low_count FROM tail_counts) > 1 THEN 'Low tail'
      WHEN (SELECT candidate_group_count FROM group_threshold) >= 3 AND is_high AND (SELECT high_count FROM tail_counts) > 1 THEN 'High tail'
      ELSE x_label
    END AS final_label,
    CASE
      WHEN (SELECT candidate_group_count FROM group_threshold) >= 3 AND is_low AND (SELECT low_count FROM tail_counts) > 1 THEN (SELECT x_sort FROM low_tail_stats)
      WHEN (SELECT candidate_group_count FROM group_threshold) >= 3 AND is_high AND (SELECT high_count FROM tail_counts) > 1 THEN (SELECT x_sort FROM high_tail_stats)
      ELSE x_sort
    END AS final_x_sort,
    CASE
      WHEN (SELECT candidate_group_count FROM group_threshold) >= 3 AND is_low AND (SELECT low_count FROM tail_counts) > 1 THEN (SELECT original_order FROM low_tail_stats)
      WHEN (SELECT candidate_group_count FROM group_threshold) >= 3 AND is_high AND (SELECT high_count FROM tail_counts) > 1 THEN (SELECT original_order FROM high_tail_stats)
      ELSE original_order
    END AS final_original_order,
    (SELECT candidate_group_count FROM group_threshold) >= 3
      AND ((is_low AND (SELECT low_count FROM tail_counts) > 1) OR (is_high AND (SELECT high_count FROM tail_counts) > 1)) AS final_is_tail
  FROM high_marked
  {missing_union}
)
{grouped_final_rows_sql(responses, include_sigma)}"""


def categorical_group_final_rows_sql(
    responses: list[dict[str, str]],
    include_sigma: bool,
    threshold_expr: str,
) -> str:
    return f""",
group_totals AS (
  SELECT
    COALESCE(SUM(volume), 0) AS total_volume,
    COUNT(*) AS group_count
  FROM initial_values
),
group_threshold AS (
  SELECT
    {threshold_expr} AS threshold_value,
    total_volume,
    group_count
  FROM group_totals
),
rare_marked AS (
  SELECT
    *,
    volume <= (SELECT threshold_value FROM group_threshold) AS is_rare
  FROM initial_values
),
rare_counts AS (
  SELECT
    SUM(CASE WHEN is_rare THEN 1 ELSE 0 END) AS rare_count
  FROM rare_marked
),
rare_stats AS (
  SELECT
    MIN(x_sort) AS x_sort,
    MIN(original_order) AS original_order
  FROM rare_marked
  WHERE is_rare
),
group_map AS (
  SELECT
    {group_identity_id('x_key', 'x_label')} AS source_group_id,
    x_key AS source_x_key,
    x_label AS source_x_label,
    CASE
      WHEN (SELECT group_count FROM group_threshold) >= 3 AND is_rare AND (SELECT rare_count FROM rare_counts) > 1 THEN 'tail:other'
      ELSE {group_identity_id('x_key', 'x_label')}
    END AS final_group_id,
    CASE
      WHEN (SELECT group_count FROM group_threshold) >= 3 AND is_rare AND (SELECT rare_count FROM rare_counts) > 1 THEN 'Other'
      ELSE x_label
    END AS final_label,
    CASE
      WHEN (SELECT group_count FROM group_threshold) >= 3 AND is_rare AND (SELECT rare_count FROM rare_counts) > 1 THEN (SELECT x_sort FROM rare_stats)
      ELSE x_sort
    END AS final_x_sort,
    CASE
      WHEN (SELECT group_count FROM group_threshold) >= 3 AND is_rare AND (SELECT rare_count FROM rare_counts) > 1 THEN (SELECT original_order FROM rare_stats)
      ELSE original_order
    END AS final_original_order,
    (SELECT group_count FROM group_threshold) >= 3 AND is_rare AND (SELECT rare_count FROM rare_counts) > 1 AS final_is_tail
  FROM rare_marked
)
{grouped_final_rows_sql(responses, include_sigma)}"""


def grouped_final_rows_sql(responses: list[dict[str, str]], include_sigma: bool) -> str:
    response_selects = final_response_aggregate_selects(responses)
    response_values = final_response_value_selects(responses)
    sigma_select = sigma_output_select("sigma") if include_sigma else null_sigma_output_select()
    sigma_join = "LEFT JOIN sigma ON grouped_values.final_group_id = sigma.final_group_id" if include_sigma else ""
    sigma_ctes = cte_fragment(final_sigma_ctes(include_sigma, grouped=True))
    return f""",
{sigma_ctes + ',' if sigma_ctes else ''}
grouped_final AS (
  SELECT
    group_map.final_group_id,
    MIN(group_map.final_label) AS x,
    MIN(group_map.final_x_sort) AS x_sort,
    MIN(group_map.final_original_order) AS original_order,
    COALESCE(SUM(initial_values.volume), 0) AS volume,
    COALESCE(SUM(initial_values.row_count), 0) AS row_count,
    CASE WHEN BOOL_OR(group_map.final_is_tail) THEN NULL ELSE MIN(initial_values.x_start) END AS x_start,
    CASE WHEN BOOL_OR(group_map.final_is_tail) THEN NULL ELSE MAX(initial_values.x_end) END AS x_end,
    BOOL_OR(group_map.final_is_tail) AS is_tail,
    LIST(
      struct_pack(
        source_x := initial_values.x_label,
        x_value := initial_values.x_value,
        volume := initial_values.volume
      )
      ORDER BY initial_values.original_order
    ) AS glm_overlay_points{response_selects}
  FROM initial_values
  INNER JOIN group_map
    ON initial_values.x_key IS NOT DISTINCT FROM group_map.source_x_key
   AND initial_values.x_label = group_map.source_x_label
  GROUP BY group_map.final_group_id
),
grouped_values AS (
  SELECT
    *{response_values}
  FROM grouped_final
),
final_rows AS (
  SELECT
    grouped_values.*{sigma_select}
  FROM grouped_values
  {sigma_join}
)"""


def final_response_selects(prefix: str, responses: list[dict[str, str]]) -> str:
    if not responses:
        return ""
    selects: list[str] = []
    for index, _ in enumerate(responses):
        selects.append(f"{prefix}.resp{index}_num")
        selects.append(f"{prefix}.resp{index}_den")
        selects.append(f"{prefix}.resp{index}")
    return ",\n    " + ",\n    ".join(selects)


def cte_fragment(sql: str) -> str:
    stripped = sql.strip()
    if stripped.startswith(","):
        stripped = stripped[1:].strip()
    return stripped


def final_response_aggregate_selects(responses: list[dict[str, str]]) -> str:
    if not responses:
        return ""
    selects: list[str] = []
    for index, _ in enumerate(responses):
        selects.append(f"SUM(initial_values.resp{index}_num) AS resp{index}_num")
        selects.append(f"SUM(initial_values.resp{index}_den) AS resp{index}_den")
    return ",\n    " + ",\n    ".join(selects)


def final_response_value_selects(responses: list[dict[str, str]]) -> str:
    if not responses:
        return ""
    selects = [
        f"resp{index}_num / NULLIF(resp{index}_den, 0) AS resp{index}"
        for index, _ in enumerate(responses)
    ]
    return ",\n    " + ",\n    ".join(selects)


def sigma_output_select(prefix: str) -> str:
    return f""",
    {prefix}.sigma_se,
    {prefix}.valid_folds,
    {prefix}.sigma_folds"""


def null_sigma_output_select() -> str:
    return """,
    NULL AS sigma_se,
    NULL AS valid_folds,
    NULL AS sigma_folds"""


def group_identity_id(key_sql: str, label_sql: str) -> str:
    return f"'id:' || COALESCE(CAST({key_sql} AS VARCHAR), '<NULL>') || ':' || COALESCE(CAST({label_sql} AS VARCHAR), '<NULL>')"


def shap_medians_cte(sort: str, medians: dict[str, float]) -> str:
    if sort != "shap":
        return ""
    if not medians:
        return """,
shap_medians(x, median) AS (
  SELECT * FROM (VALUES (CAST(NULL AS VARCHAR), CAST(NULL AS DOUBLE))) WHERE FALSE
)"""
    values = ",\n    ".join(
        f"({sql_literal(label)}, {float(value)})"
        for label, value in medians.items()
    )
    return f""",
shap_medians(x, median) AS (
  VALUES
    {values}
)"""


def sort_order_terms(x_kind: str, sort: str) -> str:
    if x_kind != "categorical":
        return "x_sort IS NULL, x_sort"
    if sort == "volume":
        return "NOT is_tail, volume DESC, LOWER(CAST(x AS VARCHAR))"
    if sort in {"actual", "response"}:
        return "resp0 IS NULL, resp0 DESC, LOWER(CAST(x AS VARCHAR))"
    if sort == "expected":
        return "resp1 IS NULL, resp1 DESC, LOWER(CAST(x AS VARCHAR))"
    if sort == "shap":
        return "is_tail, __shap_median IS NULL, __shap_median DESC, LOWER(CAST(x AS VARCHAR))"
    return "LOWER(CAST(x AS VARCHAR))"


def table_search_condition(search: str, x_kind: str) -> str:
    if not search:
        return "TRUE"
    needle = sql_literal(search.lower())
    candidates = [
        f"contains(LOWER(CAST(x AS VARCHAR)), {needle})",
        f"contains(LOWER(CAST(x_sort AS VARCHAR)), {needle})",
    ]
    if x_kind in {"integer", "numeric"}:
        candidates.extend(
            [
                f"contains(LOWER(format('{{:,.12f}}', TRY_CAST(x AS DOUBLE))), {needle})",
                f"contains(LOWER(format('{{:,.12f}}', TRY_CAST(x_sort AS DOUBLE))), {needle})",
                f"contains(LOWER(format('{{:,}}', TRY_CAST(x AS BIGINT))), {needle})",
                f"contains(LOWER(format('{{:,}}', TRY_CAST(x_sort AS BIGINT))), {needle})",
            ]
        )
    return "(" + " OR ".join(candidates) + ")"


def transform_metadata_for_result(
    dataset: Dataset,
    result: dict[str, Any],
    transform: str,
    warnings: list[str],
) -> dict[str, Any]:
    averages = {
        index: json_number(summary.get("value"))
        for index, summary in enumerate(result["response_summaries"])
    }
    base_text = str(result["request"].get("base") or "").strip()
    metadata: dict[str, Any] = {
        "mode": transform,
        "base": base_text,
        "reference": "overall_average",
        "base_x": None,
        "values": [json_number(averages.get(index)) for index, _ in enumerate(result["responses"])],
    }
    if transform not in {"zero", "one"} or not base_text:
        return metadata

    base_row = sql_base_reference_row(dataset, result)
    if base_row is None:
        warnings.append(f"Base value {base_text} could not be matched on the x-axis; using overall response averages for the {transform} transform.")
        return metadata

    values: dict[int, float | None] = {}
    failed_indexes: list[int] = []
    for index, _ in enumerate(result["responses"]):
        reference = json_number(base_row.get(f"resp{index}"))
        if reference is None or (transform == "one" and reference == 0):
            values[index] = averages.get(index)
            failed_indexes.append(index)
        else:
            values[index] = float(reference)
    if failed_indexes:
        labels = ", ".join(result["responses"][index]["label"] for index in failed_indexes)
        warnings.append(f"Base value {base_text} has no usable {labels} response reference; using overall response averages for those {transform} transforms.")

    metadata.update(
        {
            "reference": "base",
            "base_x": base_row.get("x"),
            "values": [json_number(values.get(index)) for index, _ in enumerate(result["responses"])],
            "fallback_responses": failed_indexes,
        }
    )
    return metadata


def sql_base_reference_row(dataset: Dataset, result: dict[str, Any]) -> dict[str, Any] | None:
    base_text = str(result["request"].get("base") or "").strip()
    if not base_text or result["x_group_kind"] == "quantile":
        return None
    if result["x_group_kind"] in {"integer", "numeric"}:
        return sql_numeric_base_reference_row(dataset, result, base_text)
    target = sql_literal(base_text.lower())
    sql = f"""
{result['grouped_sql']}
SELECT *
FROM sort_ready
WHERE LOWER(TRIM(CAST(x AS VARCHAR))) = {target}
LIMIT 1
"""
    rows = fetch_dict_rows(dataset, sql)
    return normalise_grouped_sql_row(rows[0], result["responses"]) if rows else None


def sql_numeric_base_reference_row(dataset: Dataset, result: dict[str, Any], base_text: str) -> dict[str, Any] | None:
    base_number = json_number(base_text)
    if base_number is None:
        return None
    width = parse_positive_float(result["request"].get("bandWidth"))
    contains_sort = "2"
    if width:
        contains_sort = f"CASE WHEN x_sort <= {float(base_number)} AND {float(base_number)} < x_sort + {float(width)} THEN 0 ELSE 2 END"
    sql = f"""
{result['grouped_sql']},
candidates AS (
  SELECT
    *,
    MIN(CASE WHEN NOT is_tail THEN 0 ELSE 1 END) OVER () AS tail_priority
  FROM sort_ready
  WHERE TRY_CAST(x_sort AS DOUBLE) IS NOT NULL
),
usable_candidates AS (
  SELECT *
  FROM candidates
  WHERE CASE WHEN tail_priority = 0 THEN NOT is_tail ELSE TRUE END
)
SELECT *
FROM usable_candidates
ORDER BY
  {contains_sort},
  CASE WHEN TRY_CAST(x_sort AS DOUBLE) = {float(base_number)} THEN 1 ELSE 2 END,
  ABS(TRY_CAST(x_sort AS DOUBLE) - {float(base_number)}),
  TRY_CAST(x_sort AS DOUBLE)
LIMIT 1
"""
    rows = fetch_dict_rows(dataset, sql)
    return normalise_grouped_sql_row(rows[0], result["responses"]) if rows else None


def normalise_positive_int(value: Any, default: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        parsed = default
    return max(1, parsed)


def table_row_matches(row: dict[str, Any], search: str, x_kind: str = "") -> bool:
    needle = search.strip().lower()
    if not needle:
        return True
    raw_candidates = [row.get("x"), row.get("x_sort")]
    candidates = list(raw_candidates)
    if x_kind in {"integer", "numeric"}:
        candidates.extend(format_table_numeric_search_label(value) for value in raw_candidates)
    return any(needle in str(value).lower() for value in candidates if value is not None)


def format_table_numeric_search_label(value: Any) -> str | None:
    number = json_number(value)
    if number is None:
        return None
    if float(number).is_integer():
        return f"{int(number):,}"
    label = f"{float(number):,.12f}".rstrip("0").rstrip(".")
    return label if label != "-0" else "0"


def build_table_summary(
    rows: list[dict[str, Any]],
    responses: list[dict[str, str]],
    transform: str,
    transform_metadata: dict[str, Any],
) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "volume": json_number(sum(float(row.get("volume") or 0) for row in rows)) or 0,
        "row_count": json_number(sum(float(row.get("row_count") or 0) for row in rows)) or 0,
        "responses": [],
    }
    references = transform_metadata.get("values") if isinstance(transform_metadata, dict) else []
    for index, _ in enumerate(responses):
        numerator = sum(float(row.get(f"resp{index}_num") or 0) for row in rows)
        denominator = sum(float(row.get(f"resp{index}_den") or 0) for row in rows)
        average = numerator / denominator if denominator else None
        reference = json_number(references[index]) if isinstance(references, list) and index < len(references) else None
        summary["responses"].append(transform_value(average, transform, reference))
    return summary


def chart_context(dataset: Dataset, request: dict[str, Any]) -> dict[str, Any]:
    source_id = dataset.normalise_source(request.get("source"))
    if not uses_field_sources(request):
        relation = dataset.relation_sql_for_source(source_id)
        return {
            "source_id": source_id,
            "relation": relation,
            "columns": dataset.column_map_for_source(source_id),
            "row_count": dataset.row_count_for_source(source_id),
            "field_sources": None,
        }
    return mixed_chart_context(dataset, request, source_id)


def uses_field_sources(request: dict[str, Any]) -> bool:
    if str(request.get("xSource") or "").strip():
        return True
    denominator_source = str(request.get("denominatorSource") or "").strip()
    if denominator_source and denominator_source != "dataset":
        return True
    raw_responses = request.get("responses")
    if not isinstance(raw_responses, list):
        return False
    return any(isinstance(item, dict) and str(item.get("source") or "").strip() for item in raw_responses)


def mixed_chart_context(dataset: Dataset, request: dict[str, Any], source_id: str) -> dict[str, Any]:
    dataset_columns = dataset.column_map()
    columns = dict(dataset_columns)
    prediction_sources: dict[str, ModelPredictionSource] = {}
    field_sources: dict[str, Any] = {"x": "dataset", "responses": [], "denominator": "dataset"}
    requested_fields: list[tuple[str, str]] = []
    x_col = str(request.get("x") or "")
    x_source = field_source_id(dataset, request.get("xSource"), source_id)
    field_sources["x"] = x_source
    requested_fields.append((x_col, x_source))
    raw_responses = request.get("responses")
    if isinstance(raw_responses, list):
        for item in raw_responses:
            if not isinstance(item, dict):
                continue
            response_source = field_source_id(dataset, item.get("source"), source_id)
            field_sources["responses"].append(response_source)
            requested_fields.append((str(item.get("numerator") or ""), response_source))
    raw_denominator = request.get("denominator", request.get("weight"))
    denominator_source = normalise_denominator_source(
        dataset,
        request.get("denominatorSource"),
        raw_denominator,
    )
    field_sources["denominator"] = denominator_source
    if has_denominator_column(raw_denominator):
        requested_fields.append((str(raw_denominator), denominator_source))

    complete_context = complete_source_relation_context(
        dataset,
        source_id=source_id,
        fields=requested_fields,
    )
    if complete_context is not None:
        return {
            **complete_context,
            "field_sources": field_sources,
        }

    for column_name, field_source in requested_fields:
        add_field_column(
            dataset,
            columns,
            dataset_columns,
            prediction_sources,
            column_name,
            field_source,
        )
    relation = mixed_relation_sql(
        dataset,
        list(prediction_sources.values()),
        preserve_dataset_rows=denominator_source != "dataset",
    )
    return {
        "source_id": source_id,
        "relation": relation,
        "columns": columns,
        "row_count": relation_row_count(dataset, relation),
        "field_sources": field_sources,
    }


def field_source_id(dataset: Dataset, raw_source: Any, fallback_source: str) -> str:
    raw = str(raw_source or "").strip()
    if raw:
        return dataset.normalise_source(raw)
    return fallback_source or "dataset"


def add_field_column(
    dataset: Dataset,
    columns: dict[str, ColumnInfo],
    dataset_columns: dict[str, ColumnInfo],
    prediction_sources: dict[str, ModelPredictionSource],
    column_name: str,
    source_id: str,
) -> None:
    if not column_name:
        return
    prediction_source = dataset.model_prediction_source(source_id)
    if prediction_source is not None:
        source_columns = dataset.column_map_for_source(source_id)
        source_column = source_columns.get(column_name)
        model_prediction_columns = {
            "gbm_prediction",
            "gbm_prediction_rate",
            "gbm_tabulated_prediction",
            "glm_prediction",
            "glm_prediction_rate",
            "glm_tabulated_prediction",
        }
        if column_name in model_prediction_columns and source_column is not None and is_numeric_kind(source_column.kind):
            if not prediction_source.relation_sql:
                raise ValueError("Choose a valid model prediction source")
            prediction_sources[f"{prediction_source.source_id}:{column_name}"] = ModelPredictionSource(
                source_id=prediction_source.source_id,
                column=column_name,
                relation_sql=prediction_source.relation_sql,
                active=prediction_source.active,
                binding=prediction_source.bindings.get(column_name),
                bindings=prediction_source.bindings,
            )
            columns[column_name] = ColumnInfo(name=column_name, duckdb_type="DOUBLE", kind="numeric")
            return
        if column_name == prediction_source.column:
            if not prediction_source.column or not prediction_source.relation_sql:
                raise ValueError("Choose a valid model prediction source")
            prediction_sources[prediction_source.source_id] = prediction_source
            columns[column_name] = ColumnInfo(name=column_name, duckdb_type="DOUBLE", kind="numeric")
            return
    if column_name in dataset_columns:
        columns[column_name] = dataset_columns[column_name]


def mixed_relation_sql(
    dataset: Dataset,
    prediction_sources: list[ModelPredictionSource],
    *,
    preserve_dataset_rows: bool = False,
) -> str:
    positional_sql = positional_mixed_relation_sql(dataset, prediction_sources)
    if positional_sql:
        return positional_sql
    return keyed_mixed_relation_sql(
        dataset,
        prediction_sources,
        preserve_dataset_rows=preserve_dataset_rows,
    )


def positional_mixed_relation_sql(dataset: Dataset, prediction_sources: list[ModelPredictionSource]) -> str:
    if not prediction_sources:
        return ""
    bindings = []
    for source in prediction_sources:
        binding = source.binding
        if binding is None or source.column not in binding.columns:
            return ""
        if not dataset.model_source_binding_eligible(binding):
            return ""
        bindings.append(binding)
    base_where_sql = bindings[0].base_where_sql
    if any(binding.base_where_sql != base_where_sql for binding in bindings):
        return ""

    prediction_columns = {source.column for source in prediction_sources}
    dataset_columns = [
        column.name for column in dataset.valid_schema_columns() if column.name not in prediction_columns
    ]
    source_column_sql = ",\n    ".join(quote_ident(name) for name in dataset_columns)
    source_column_suffix = f",\n    {source_column_sql}" if source_column_sql else ""
    where_sql = f"\n  WHERE {base_where_sql}" if base_where_sql else ""
    joins: list[str] = []
    selects = [f"base.{quote_ident(name)}" for name in dataset_columns]
    for index, source in enumerate(prediction_sources):
        alias = f"prediction_{index}"
        joins.append(f"POSITIONAL JOIN {source.binding.relation_sql} {alias}")
        selects.append(f"{alias}.{quote_ident(source.column)} AS {quote_ident(source.column)}")
    select_sql = ",\n  ".join(selects) if selects else "*"
    join_sql = "\n".join(joins)
    return f"""(
SELECT
  {select_sql}
FROM (
  SELECT
    ROW_NUMBER() OVER () AS __lucidum_row_id{source_column_suffix}
  FROM {dataset.relation_sql()}
  {where_sql}
) base
{join_sql}
)"""


def keyed_mixed_relation_sql(
    dataset: Dataset,
    prediction_sources: list[ModelPredictionSource],
    *,
    preserve_dataset_rows: bool = False,
) -> str:
    prediction_columns = {source.column for source in prediction_sources}
    dataset_columns = [
        column.name for column in dataset.valid_schema_columns() if column.name not in prediction_columns
    ]
    source_column_sql = ",\n    ".join(quote_ident(name) for name in dataset_columns)
    source_column_suffix = f",\n    {source_column_sql}" if source_column_sql else ""
    joins: list[str] = []
    selects = [f"base.{quote_ident(name)}" for name in dataset_columns]
    scope_sql = ""
    if prediction_sources and not preserve_dataset_rows:
        scope_parts = [
            f"SELECT __lucidum_row_id FROM {source.relation_sql}"
            for source in prediction_sources
        ]
        scope_sql = ",\nprediction_scope AS (\n  " + "\n  UNION\n  ".join(scope_parts) + "\n)"
        joins.append("INNER JOIN prediction_scope scope USING (__lucidum_row_id)")
    for index, source in enumerate(prediction_sources):
        alias = f"prediction_{index}"
        joins.append(f"LEFT JOIN {source.relation_sql} {alias} USING (__lucidum_row_id)")
        selects.append(f"{alias}.{quote_ident(source.column)} AS {quote_ident(source.column)}")
    select_sql = ",\n  ".join(selects) if selects else "*"
    join_sql = "\n".join(joins)
    return f"""(
WITH dataset_rows AS (
  SELECT
    ROW_NUMBER() OVER () AS __lucidum_row_id{source_column_suffix}
  FROM {dataset.relation_sql()}
){scope_sql}
SELECT
  {select_sql}
FROM dataset_rows base
{join_sql}
)"""


def relation_row_count(dataset: Dataset, relation: str, filter_sql: str = "") -> int:
    where_sql = f" WHERE ({filter_sql})" if filter_sql else ""
    value = dataset.con.execute(f"SELECT COUNT(*) FROM {relation}{where_sql}").fetchone()[0]
    return int(value)


def relation_denominator_summary(
    dataset: Dataset,
    relation: str,
    responses: list[dict[str, str]],
    denominator: dict[str, str | None],
    filter_sql: str = "",
) -> dict[str, Any]:
    sql = build_denominator_summary_sql(relation, responses, denominator, filter_sql)
    cursor = dataset.con.execute(sql)
    fetched = cursor.fetchone()
    return dict(zip([d[0] for d in cursor.description], fetched or []))


def relation_response_summary(
    dataset: Dataset,
    relation: str,
    responses: list[dict[str, str]],
    denominator: dict[str, str | None],
    filter_sql: str = "",
) -> list[dict[str, Any]]:
    if not responses:
        return []
    sql = build_response_summary_sql(relation, responses, denominator, filter_sql)
    cursor = dataset.con.execute(sql)
    fetched = cursor.fetchone()
    row = dict(zip([d[0] for d in cursor.description], fetched or []))
    return [
        {
            "label": response["label"],
            "value": json_number(row.get(f"resp{index}")),
            "numerator": json_number(row.get(f"resp{index}_num")),
            "denominator": json_number(row.get(f"resp{index}_den")),
        }
        for index, response in enumerate(responses)
    ]


def normalise_responses(raw: Any, columns: dict[str, ColumnInfo]) -> list[dict[str, str]]:
    responses: list[dict[str, str]] = []
    if not isinstance(raw, list):
        return responses
    for item in raw:
        if not isinstance(item, dict):
            continue
        numerator = item.get("numerator")
        if not numerator or numerator not in columns:
            continue
        if not is_numeric_kind(columns[str(numerator)].kind):
            continue
        label = item.get("label") or str(numerator)
        source = str(item.get("source") or "").strip()
        responses.append({"label": str(label), "numerator": str(numerator), **({"source": source} if source else {})})
    return responses[:3]


def use_quantiles(value: Any) -> bool:
    if value is True:
        return True
    if value is False or value is None:
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "on", "quantile", "quantiles"}


def normalise_date_bucket(value: Any) -> str:
    bucket = str(value or "none").strip().lower()
    return bucket if bucket in DATE_BUCKETS else "none"


def normalise_empty_periods(value: Any) -> str:
    mode = str(value or "show").strip().lower()
    return mode if mode in EMPTY_PERIOD_VALUES else "show"


def normalise_missings(value: Any) -> str:
    mode = str(value or "show").strip().lower()
    return mode if mode in MISSING_VALUES else "show"


def line_bar_analysis_filter(filter_sql: str, hidden_features: list[str]) -> str:
    conditions = [f"({filter_sql})"] if filter_sql else []
    conditions.extend(f"{quote_ident(feature)} IS NOT NULL" for feature in dict.fromkeys(hidden_features))
    return " AND ".join(conditions)


def normalise_quantile_count(value: Any) -> int:
    parsed = parse_positive_float(value)
    if parsed is None:
        return 1
    count = int(math.floor(parsed + 0.5))
    return min(1000, max(1, count))


def decimal_places_for_band_width(value: Any) -> int | None:
    parsed = parse_positive_float(value)
    if parsed is None:
        return None
    try:
        decimal = Decimal(str(value)).normalize()
    except InvalidOperation:
        return None
    if decimal == decimal.to_integral_value():
        return 0
    return max(0, -decimal.as_tuple().exponent)


def format_numeric_band_label(value: Any, decimal_places: int) -> str | None:
    number = json_number(value)
    if number is None:
        return None
    try:
        quant = Decimal("1").scaleb(-decimal_places)
        rounded = Decimal(str(number)).quantize(quant)
    except InvalidOperation:
        return None
    if rounded == 0:
        rounded = abs(rounded)
    label = format(rounded, "f")
    if "." in label:
        label = label.rstrip("0").rstrip(".")
    return label


def clean_numeric_band_labels(rows: list[dict[str, Any]], x_kind: str, band_width: Any) -> None:
    if x_kind != "numeric":
        return
    decimal_places = decimal_places_for_band_width(band_width)
    if decimal_places is None:
        return
    for row in rows:
        if row.get("is_tail"):
            continue
        label = format_numeric_band_label(row.get("x_sort"), decimal_places)
        if label is not None:
            row["x"] = label


def build_x_sql(x_col: str, kind: str, band_width: Any, date_bucket: Any, quantile_count: int | None = None) -> dict[str, str]:
    col = quote_ident(x_col)
    if is_numeric_kind(kind):
        raw = f"TRY_CAST({col} AS DOUBLE)"
        if quantile_count:
            return {
                "key": "__x_quantile",
                "label": "CASE WHEN __x_quantile IS NULL THEN 'Missing' ELSE 'Q' || CAST(__x_quantile AS VARCHAR) END",
                "sort": "COALESCE(__x_quantile, 1000001)",
                "raw": raw,
                "quantile_count": str(quantile_count),
            }
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
    return {"key": key, "label": key, "sort": key, "model_value": col}


def build_chart_sql(
    relation: str,
    x_sql: dict[str, str],
    responses: list[dict[str, str]],
    denominator: dict[str, str | None],
    include_sigma: bool,
    filter_sql: str = "",
) -> str:
    valid_condition = denominator_valid_condition(responses, denominator)
    weight_expr = weighted_value_sql(denominator, valid_condition)
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
    sigma_output = ",\n    sigma.sigma_se,\n    sigma.valid_folds,\n    sigma.sigma_folds"
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
    *,
    {', '.join(fold_values)}
  FROM folds
),
sigma AS (
  SELECT
    x_key,
    x_label,
    STDDEV_SAMP(resp0 - resp1) / SQRT(COUNT(*)) AS sigma_se,
    COUNT(*) AS valid_folds,
    LIST(struct_pack(
      fold := __fold,
      resp0_num := resp0_num,
      resp0_den := resp0_den,
      resp1_num := resp1_num,
      resp1_den := resp1_den
    )) AS sigma_folds
    FROM fold_values
  WHERE resp0 IS NOT NULL AND resp1 IS NOT NULL
  GROUP BY x_key, x_label
)"""
        sigma_join = "LEFT JOIN sigma ON agg_values.x_key IS NOT DISTINCT FROM sigma.x_key AND agg_values.x_label = sigma.x_label"
    else:
        sigma_output = ",\n    NULL AS sigma_se,\n    NULL AS valid_folds,\n    NULL AS sigma_folds"
        sigma_join = ""

    where_sql = f"\n  WHERE ({filter_sql})" if filter_sql else ""
    quantile_cte = ""
    keyed_from = "base"
    rownum_expr = "__rownum"
    source_columns = "*"
    x_bound_keyed_sql = ""
    x_bound_agg_sql = ",\n    NULL AS x_start,\n    NULL AS x_end"
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
        x_bound_keyed_sql = f",\n    {x_sql['raw']} AS __x_raw_value"
        x_bound_agg_sql = ",\n    MIN(__x_raw_value) AS x_start,\n    MAX(__x_raw_value) AS x_end"
    return f"""
WITH base AS (
  SELECT ROW_NUMBER() OVER () AS __rownum, * FROM {relation}{where_sql}
){quantile_cte},
keyed AS (
  SELECT
    {rownum_expr} AS __rownum,
    {x_sql['key']} AS x_key,
    {x_sql['label']} AS x_label,
    {x_sql['sort']} AS x_sort,
    CAST(hash({rownum_expr}) % 20 AS INTEGER) AS __fold{x_bound_keyed_sql},
    {weight_expr} AS __weight_value,
    {source_columns}
  FROM {keyed_from}
),
agg AS (
  SELECT
    x_key,
    x_label,
    MIN(x_sort) AS x_sort,
    MIN(__rownum) AS original_order,
    COALESCE(SUM(__weight_value), 0) AS volume,
    COUNT(*) AS row_count{x_bound_agg_sql}
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
    responses: list[dict[str, str]],
    x_kind: str,
    threshold: str,
) -> list[dict[str, Any]]:
    total_volume = sum(float(row.get("volume") or 0) for row in rows)
    threshold_value = parse_group_threshold(threshold, total_volume)
    normalised = [normalise_row(row, responses) for row in rows]
    missing_rows: list[dict[str, Any]] = []
    if x_kind in {"integer", "numeric", "date", "datetime", "quantile"}:
        missing_rows = [
            row
            for row in normalised
            if row.get("x_sort") is None or (x_kind == "quantile" and row.get("x") == "Missing")
        ]
        normalised = [row for row in normalised if row not in missing_rows]
    if threshold_value <= 0 or len(normalised) < 3:
        return normalised + missing_rows

    if x_kind in {"integer", "numeric", "date", "datetime", "quantile"}:
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
        return result + missing_rows

    rare = [row for row in normalised if float(row["volume"] or 0) <= threshold_value]
    common = [row for row in normalised if float(row["volume"] or 0) > threshold_value]
    if len(rare) > 1:
        common.append(combine_rows(rare, "Other", responses, is_tail=True))
    else:
        common.extend(rare)
    return common


def normalise_row(row: dict[str, Any], responses: list[dict[str, str]]) -> dict[str, Any]:
    result = {
        "x": str(row.get("x_label")),
        "x_sort": row.get("x_sort"),
        "original_order": int(row.get("original_order") or 0),
        "volume": json_number(row.get("volume")) or 0,
        "row_count": json_number(row.get("row_count")) or 0,
        "is_tail": False,
        "sigma_se": json_number(row.get("sigma_se")),
        "valid_folds": json_number(row.get("valid_folds")),
        "sigma_folds": row.get("sigma_folds"),
    }
    x_start = json_number(row.get("x_start"))
    x_end = json_number(row.get("x_end"))
    if x_start is not None:
        result["x_start"] = x_start
    if x_end is not None:
        result["x_end"] = x_end
    for index, _ in enumerate(responses):
        result[f"resp{index}_num"] = json_number(row.get(f"resp{index}_num"))
        result[f"resp{index}_den"] = json_number(row.get(f"resp{index}_den"))
        result[f"resp{index}"] = json_number(row.get(f"resp{index}"))
    return result


def combine_rows(rows: list[dict[str, Any]], label: str, responses: list[dict[str, str]], is_tail: bool) -> dict[str, Any]:
    sigma_se, valid_folds = combine_sigma(rows) if len(responses) >= 2 else (None, None)
    combined = {
        "x": label,
        "x_sort": rows[0].get("x_sort"),
        "original_order": min(int(row.get("original_order") or 0) for row in rows),
        "volume": json_number(sum(float(row.get("volume") or 0) for row in rows)) or 0,
        "row_count": json_number(sum(float(row.get("row_count") or 0) for row in rows)) or 0,
        "is_tail": is_tail,
        "sigma_se": sigma_se,
        "valid_folds": valid_folds,
        "sigma_folds": None,
    }
    for index, _ in enumerate(responses):
        num = sum(float(row.get(f"resp{index}_num") or 0) for row in rows)
        den = sum(float(row.get(f"resp{index}_den") or 0) for row in rows)
        combined[f"resp{index}_num"] = json_number(num)
        combined[f"resp{index}_den"] = json_number(den)
        combined[f"resp{index}"] = json_number(num / den) if den else None
    return combined


def combine_sigma(rows: list[dict[str, Any]]) -> tuple[float | int | None, float | int | None]:
    fold_totals: dict[int, dict[str, float]] = {}
    for row in rows:
        components = row.get("sigma_folds") or []
        if not isinstance(components, list):
            continue
        for component in components:
            if not isinstance(component, dict):
                continue
            fold = component.get("fold")
            if fold is None:
                continue
            bucket = fold_totals.setdefault(
                int(fold),
                {"resp0_num": 0.0, "resp0_den": 0.0, "resp1_num": 0.0, "resp1_den": 0.0},
            )
            for key in bucket:
                value = json_number(component.get(key))
                if value is not None:
                    bucket[key] += float(value)

    diffs: list[float] = []
    for totals in fold_totals.values():
        if totals["resp0_den"] and totals["resp1_den"]:
            diffs.append(totals["resp0_num"] / totals["resp0_den"] - totals["resp1_num"] / totals["resp1_den"])
    valid_folds = len(diffs)
    if valid_folds < 2:
        return None, json_number(valid_folds) if valid_folds else None
    mean = sum(diffs) / valid_folds
    variance = sum((value - mean) ** 2 for value in diffs) / (valid_folds - 1)
    return json_number(math.sqrt(variance) / math.sqrt(valid_folds)), json_number(valid_folds)


def sort_rows(
    rows: list[dict[str, Any]],
    x_kind: str,
    sort: str,
    shap_medians: dict[str, float] | None = None,
) -> list[dict[str, Any]]:
    if x_kind not in {"categorical"}:
        return sorted(rows, key=lambda r: (r["x_sort"] is None, r["x_sort"]))
    if sort == "volume":
        return sorted(rows, key=lambda r: (not r.get("is_tail"), -(r["volume"] or 0), str(r["x"]).lower()))
    if sort in {"actual", "response"}:
        return sorted(rows, key=lambda r: (r.get("resp0") is None, -(r.get("resp0") or 0), str(r["x"]).lower()))
    if sort == "expected":
        return sorted(rows, key=lambda r: (r.get("resp1") is None, -(r.get("resp1") or 0), str(r["x"]).lower()))
    if sort == "shap":
        medians = shap_medians or {}
        return sorted(rows, key=lambda r: (r.get("is_tail"), medians.get(str(r["x"])) is None, -(medians.get(str(r["x"])) or 0), str(r["x"]).lower()))
    return sorted(rows, key=lambda r: str(r["x"]).lower())


def build_partial_dependence_overlay(
    dataset: Dataset,
    request: dict[str, Any],
    *,
    feature_spec: Any,
    columns: dict[str, ColumnInfo],
    x_col: str,
    x_sql: dict[str, str],
    x_group_kind: str,
    responses: list[dict[str, str]],
    denominator: dict[str, str | None],
) -> dict[str, Any] | None:
    mode = partial_dependence_mode(request.get("partialDependence"))
    if mode == "none":
        return None
    overlays: dict[str, dict[str, Any]] = {}
    if mode in {"shap", "both"}:
        try:
            overlays["shap"] = build_shap_partial_dependence_overlay(
                dataset,
                request,
                columns=columns,
                x_col=x_col,
                x_sql=x_sql,
                x_group_kind=x_group_kind,
                responses=responses,
                denominator=denominator,
            )
        except Exception as exc:
            overlays["shap"] = empty_shap_partial_dependence_warning(f"SHAP overlay failed: {exc}")
    if mode in {"glm", "both"}:
        try:
            from py_lucidum.tools.glm.overlay import build_glm_partial_dependence_overlay

            overlays["glm"] = build_glm_partial_dependence_overlay(
                dataset,
                request,
                feature_spec=feature_spec,
                x_col=x_col,
                x_sql=x_sql,
                x_group_kind=x_group_kind,
                denominator=denominator,
            )
        except Exception as exc:
            from py_lucidum.tools.glm.overlay import empty_glm_partial_dependence_warning

            overlays["glm"] = empty_glm_partial_dependence_warning(f"GLM overlay failed: {exc}")
    if mode == "shap":
        return overlays.get("shap")
    if mode == "glm":
        return overlays.get("glm")
    align_both_overlay_means(overlays)
    warnings: list[str] = []
    for overlay in overlays.values():
        warnings.extend(str(warning) for warning in (overlay.get("warnings") or []) if warning)
    return {
        "mode": "both",
        "overlays": overlays,
        "rows": overlays.get("shap", {}).get("rows", []),
        "warnings": warnings,
        "transform": {"mode": str(request.get("transform") or "none")},
    }


def align_both_overlay_means(overlays: dict[str, dict[str, Any]]) -> None:
    shap = overlays.get("shap")
    glm = overlays.get("glm")
    if not isinstance(shap, dict) or not isinstance(glm, dict):
        return
    target = json_number((shap.get("scale") or {}).get("target"))
    if target is None:
        return
    rows = glm.get("rows")
    if not isinstance(rows, list) or not rows:
        return
    align_overlay_to_target(glm, target, label="GLM overlay")


def align_overlay_to_target(overlay: dict[str, Any], target: float | int, *, label: str) -> None:
    rows = [row for row in overlay.get("rows") or [] if isinstance(row, dict)]
    source_mean = weighted_average((row.get("p50"), row.get("volume")) for row in rows)
    if source_mean is None:
        overlay.setdefault("warnings", []).append(f"{label} could not be aligned to the comparison mean.")
        return
    scale = overlay.setdefault("scale", {})
    native_target = scale.get("target")
    if native_target is not None and "native_target" not in scale:
        scale["native_target"] = native_target
    scale["target"] = json_number(target)
    method = str(scale.get("method") or "").strip().lower()
    if method == "multiply":
        if source_mean == 0:
            overlay.setdefault("warnings", []).append(f"{label} could not be aligned because its mean is zero.")
            return
        factor = float(target) / float(source_mean)
        for row in rows:
            for key in partial_dependence_value_keys(overlay):
                value = json_number(row.get(key))
                row[key] = json_number(float(value) * factor) if value is not None else None
        scale["comparison"] = {"method": "multiply", "source_mean": json_number(source_mean), "factor": json_number(factor)}
        return
    shift = float(target) - float(source_mean)
    for row in rows:
        for key in partial_dependence_value_keys(overlay):
            value = json_number(row.get(key))
            row[key] = json_number(float(value) + shift) if value is not None else None
    scale["comparison"] = {"method": "add", "source_mean": json_number(source_mean), "shift": json_number(shift)}


def build_shap_partial_dependence_overlay(
    dataset: Dataset,
    request: dict[str, Any],
    *,
    columns: dict[str, ColumnInfo],
    x_col: str,
    x_sql: dict[str, str],
    x_group_kind: str,
    responses: list[dict[str, str]],
    denominator: dict[str, str | None],
) -> dict[str, Any]:
    requested_model_id = shap_partial_dependence_model_id(request)
    source = gbm_shap_overlay_source(dataset, requested_model_id)
    if source is None:
        if requested_model_id:
            return empty_shap_partial_dependence_warning(
                f"GBM SHAP values for {requested_model_id} are no longer available."
            )
        return empty_shap_partial_dependence_warning("No active GBM SHAP values are available.")
    model_id = str(source.get("model_id") or "")
    shap_source_id = str(source.get("id") or "")
    if not model_id or not shap_source_id or not source.get("has_prediction"):
        return empty_shap_partial_dependence_warning("The active GBM needs both SHAP values and predictions for SHAP ribbons.")

    shap_column = shap_value_column_for_feature(source, x_col)
    if not shap_column:
        return empty_shap_partial_dependence_warning(f"No active GBM SHAP values are available for {x_col}.")
    shap_source_columns = dataset.column_map_for_source(shap_source_id)
    if x_col not in shap_source_columns:
        return empty_shap_partial_dependence_warning(f"The active GBM SHAP source does not include {x_col}.")
    if "gbm_prediction" not in shap_source_columns:
        return empty_shap_partial_dependence_warning("The active GBM SHAP source does not include fitted predictions.")

    overlay_responses = [response for response in responses if response.get("numerator") in shap_source_columns]
    overlay_denominator = normalise_overlay_denominator(denominator, shap_source_columns)
    shap_relation = dataset.relation_sql_for_source(shap_source_id)
    filter_sql = dataset.normalise_filter_for_relation(request.get("filter"), shap_relation)
    objective = str(source.get("objective") or "")
    shap_expr = shap_response_sql(objective, quote_ident(shap_column))
    initial_sql = build_partial_dependence_sql(
        shap_relation,
        x_sql,
        shap_expr,
        overlay_responses,
        overlay_denominator,
        filter_sql,
    )
    initial_raw_rows = [
        dict(zip([d[0] for d in dataset.con.description], row))
        for row in dataset.con.execute(initial_sql).fetchall()
    ]
    initial_rows = normalise_partial_dependence_rows(initial_raw_rows)
    group_mapping = partial_low_weight_group_mapping(initial_rows, x_group_kind, str(request.get("lowGroup") or "0"))
    if partial_group_mapping_is_identity(group_mapping):
        rows = initial_rows
    elif group_mapping:
        final_sql = build_partial_dependence_sql(
            shap_relation,
            x_sql,
            shap_expr,
            overlay_responses,
            overlay_denominator,
            filter_sql,
            group_mapping=group_mapping,
        )
        raw_rows = [
            dict(zip([d[0] for d in dataset.con.description], row))
            for row in dataset.con.execute(final_sql).fetchall()
        ]
        rows = normalise_partial_dependence_rows(raw_rows)
    else:
        rows = []
    if x_group_kind == "numeric":
        clean_partial_numeric_labels(rows, request.get("bandWidth"))
    scale = scale_partial_dependence_rows(rows, objective=objective)
    warnings: list[str] = []
    if not rows:
        warnings.append("No SHAP ribbon rows matched the current chart selection.")
    if scale.get("warning"):
        warnings.append(str(scale["warning"]))
    return {
        "mode": "shap",
        "model_id": model_id,
        "feature": x_col,
        "shap_column": shap_column,
        "percentiles": list(SHAP_RIBBON_PERCENTILES),
        "rows": rows,
        "warnings": warnings,
        "scale": {key: value for key, value in scale.items() if key != "warning"},
        "transform": {"mode": str(request.get("transform") or "none")},
    }


def shap_partial_dependence_model_id(request: dict[str, Any]) -> str:
    partial_dependence = request.get("partialDependence")
    if not isinstance(partial_dependence, dict):
        return ""
    return str(
        partial_dependence.get("gbm_model_id")
        or partial_dependence.get("model_id")
        or ""
    ).strip()


def gbm_shap_overlay_source(dataset: Dataset, model_id: str = "") -> dict[str, Any] | None:
    for provider in getattr(dataset, "_source_providers", []):
        if model_id:
            requested_source = getattr(provider, "shap_overlay_source", None)
            if not callable(requested_source):
                continue
            source = requested_source(model_id, dataset)
            if isinstance(source, dict) and str(source.get("model_id") or "") == model_id:
                return source
            continue
        active_source = getattr(provider, "active_shap_overlay_source", None)
        if not callable(active_source):
            continue
        source = active_source(dataset)
        if isinstance(source, dict) and source.get("active"):
            return source
    return None


def partial_dependence_mode(raw: Any) -> str:
    if not isinstance(raw, dict):
        return "none"
    mode = str(raw.get("mode") or "none").strip().lower()
    return mode if mode in {"none", "shap", "glm", "both"} else "none"


def partial_dependence_mode_is_shap(raw: Any) -> bool:
    return partial_dependence_mode(raw) == "shap"


def empty_shap_partial_dependence_warning(message: str) -> dict[str, Any]:
    return {
        "mode": "shap",
        "model_id": "",
        "feature": "",
        "percentiles": list(SHAP_RIBBON_PERCENTILES),
        "rows": [],
        "warnings": [message],
        "scale": {"method": "none", "target": None, "source_mean": None},
        "transform": {"mode": "none"},
    }


def shap_value_column_for_feature(source: dict[str, Any], feature: str) -> str:
    for column in source.get("columns") or []:
        if not isinstance(column, dict):
            continue
        if column.get("source_role") != "gbm_shap_value":
            continue
        if str(column.get("artifact_column") or column.get("label") or "") == feature:
            return str(column.get("name") or "")
    return ""


def partial_group_mapping_is_identity(group_mapping: list[dict[str, Any]]) -> bool:
    if not group_mapping:
        return False
    for row in group_mapping:
        source_x = "" if row.get("source_x") is None else str(row.get("source_x"))
        final_x = "" if row.get("final_x") is None else str(row.get("final_x"))
        if source_x != final_x:
            return False
        if row.get("final_is_tail"):
            return False
    return True


def normalise_overlay_denominator(
    denominator: dict[str, str | None],
    columns: dict[str, ColumnInfo],
) -> dict[str, str | None]:
    column = denominator.get("column")
    if not column:
        return denominator
    if column in columns:
        return denominator
    return {"column": None, "label": "Average row value", "bar_label": "Row count"}


def shap_response_sql(objective: str, shap_column_sql: str) -> str:
    selected_objective = str(objective or "").strip().lower()
    raw = f"TRY_CAST({shap_column_sql} AS DOUBLE)"
    if selected_objective in LOG_LINK_OBJECTIVES:
        return f"EXP({raw})"
    if selected_objective in BINARY_LINK_OBJECTIVES:
        return f"1.0 / (1.0 + EXP(-({raw})))"
    return raw


def build_partial_dependence_sql(
    relation: str,
    x_sql: dict[str, str],
    shap_expr: str,
    responses: list[dict[str, str]],
    denominator: dict[str, str | None],
    filter_sql: str = "",
    *,
    group_mapping: list[dict[str, Any]] | None = None,
) -> str:
    checks = [f"{shap_expr} IS NOT NULL", "TRY_CAST(gbm_prediction AS DOUBLE) IS NOT NULL"]
    selected_response_checks = denominator_valid_condition(responses, denominator)
    if selected_response_checks != "TRUE":
        checks.append(selected_response_checks)
    valid_condition = " AND ".join(checks)
    weight_expr = weighted_value_sql(denominator, valid_condition)
    percentile_sql = ",\n    ".join(percentile_selects("__shap_response", SHAP_RIBBON_PERCENTILES))
    where_sql = f"\n  WHERE ({filter_sql})" if filter_sql else ""
    quantile_cte = ""
    keyed_from = "base"
    rownum_expr = "__rownum"
    source_columns = "*"
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
    map_cte = partial_group_map_cte(group_mapping)
    if group_mapping:
        valid_sql = f"""
valid AS (
  SELECT
    map.final_label AS x_label,
    map.final_x_sort AS x_sort,
    map.final_original_order AS original_order,
    map.final_is_tail AS is_tail,
    keyed.__shap_response,
    keyed.__gbm_prediction,
    keyed.__weight_value
  FROM keyed
  INNER JOIN group_map map ON keyed.x_label = map.source_label
  WHERE keyed.__weight_value IS NOT NULL
    AND keyed.__shap_response IS NOT NULL
    AND keyed.__gbm_prediction IS NOT NULL
)"""
    else:
        valid_sql = """
valid AS (
  SELECT
    x_label,
    x_sort,
    __rownum AS original_order,
    FALSE AS is_tail,
    __shap_response,
    __gbm_prediction,
    __weight_value
  FROM keyed
  WHERE __weight_value IS NOT NULL
    AND __shap_response IS NOT NULL
    AND __gbm_prediction IS NOT NULL
)"""
    return f"""
WITH base AS (
  SELECT ROW_NUMBER() OVER () AS __rownum, * FROM {relation}{where_sql}
){quantile_cte},
keyed AS (
  SELECT
    {rownum_expr} AS __rownum,
    {x_sql['key']} AS x_key,
    {x_sql['label']} AS x_label,
    {x_sql['sort']} AS x_sort,
    {shap_expr} AS __shap_response,
    TRY_CAST(gbm_prediction AS DOUBLE) AS __gbm_prediction,
    {weight_expr} AS __weight_value,
    {source_columns}
  FROM {keyed_from}
){map_cte},
{valid_sql}
SELECT
    x_label,
    MIN(x_sort) AS x_sort,
    MIN(original_order) AS original_order,
    BOOL_OR(is_tail) AS is_tail,
    COALESCE(SUM(__weight_value), 0) AS volume,
    SUM(__gbm_prediction) AS fitted_num,
    COALESCE(SUM(__weight_value), 0) AS fitted_den,
    {percentile_sql}
FROM valid
GROUP BY x_label
"""


def partial_group_map_cte(group_mapping: list[dict[str, Any]] | None) -> str:
    if not group_mapping:
        return ""
    values = ",\n    ".join(
        "("
        f"{partial_group_map_literal(row.get('source_x'))}, "
        f"{partial_group_map_literal(row.get('final_x'))}, "
        f"{partial_group_map_literal(row.get('final_x_sort'))}, "
        f"{int(row.get('final_original_order') or 0)}, "
        f"{'TRUE' if row.get('final_is_tail') else 'FALSE'}"
        ")"
        for row in group_mapping
    )
    return f""",
group_map(source_label, final_label, final_x_sort, final_original_order, final_is_tail) AS (
  VALUES
    {values}
)"""


def partial_group_map_literal(value: Any) -> str:
    return sql_literal("" if value is None else str(value))


def normalise_partial_dependence_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    normalised: list[dict[str, Any]] = []
    for row in rows:
        item = {
            "x": str(row.get("x_label")),
            "x_sort": row.get("x_sort"),
            "original_order": int(row.get("original_order") or 0),
            "volume": json_number(row.get("volume")) or 0,
            "fitted_num": json_number(row.get("fitted_num")) or 0,
            "fitted_den": json_number(row.get("fitted_den")) or 0,
            "is_tail": bool(row.get("is_tail")),
        }
        for percentile in SHAP_RIBBON_PERCENTILES:
            item[percentile_key(percentile)] = json_number(row.get(percentile_key(percentile)))
        normalised.append(item)
    return normalised


def partial_low_weight_group_mapping(rows: list[dict[str, Any]], x_kind: str, threshold: str) -> list[dict[str, Any]]:
    total_volume = sum(float(row.get("volume") or 0) for row in rows)
    threshold_value = parse_group_threshold(threshold, total_volume)
    normalised = list(rows)
    missing_rows: list[dict[str, Any]] = []
    if x_kind in {"integer", "numeric", "date", "datetime", "quantile"}:
        missing_rows = [
            row
            for row in normalised
            if row.get("x_sort") is None or (x_kind == "quantile" and row.get("x") == "Missing")
        ]
        normalised = [row for row in normalised if row not in missing_rows]
    if threshold_value <= 0 or len(normalised) < 3:
        return [partial_group_mapping_row(row, row) for row in [*normalised, *missing_rows]]

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
        for row in reversed(ordered[len(low):]):
            volume = float(row.get("volume") or 0)
            if cumulative + volume <= threshold_value:
                high.append(row)
                cumulative += volume
            else:
                break
        high = list(reversed(high))
        middle = ordered[len(low): len(ordered) - len(high) if high else len(ordered)]
        mapping: list[dict[str, Any]] = []
        mapping.extend(partial_tail_mapping_rows(low, "Low tail") if len(low) > 1 else [partial_group_mapping_row(row, row) for row in low])
        mapping.extend(partial_group_mapping_row(row, row) for row in middle)
        mapping.extend(partial_tail_mapping_rows(high, "High tail") if len(high) > 1 else [partial_group_mapping_row(row, row) for row in high])
        mapping.extend(partial_group_mapping_row(row, row) for row in missing_rows)
        return mapping

    rare = [row for row in normalised if float(row.get("volume") or 0) <= threshold_value]
    common = [row for row in normalised if float(row.get("volume") or 0) > threshold_value]
    mapping = [partial_group_mapping_row(row, row) for row in common]
    if len(rare) > 1:
        mapping.extend(partial_tail_mapping_rows(rare, "Other"))
    else:
        mapping.extend(partial_group_mapping_row(row, row) for row in rare)
    return mapping


def partial_group_mapping_row(source: dict[str, Any], final: dict[str, Any], *, label: str | None = None, is_tail: bool | None = None) -> dict[str, Any]:
    return {
        "source_x": source.get("x"),
        "final_x": label if label is not None else final.get("x"),
        "final_x_sort": final.get("x_sort"),
        "final_original_order": final.get("original_order"),
        "final_is_tail": bool(final.get("is_tail")) if is_tail is None else is_tail,
    }


def partial_tail_mapping_rows(rows: list[dict[str, Any]], label: str) -> list[dict[str, Any]]:
    if not rows:
        return []
    final = {
        "x": label,
        "x_sort": rows[0].get("x_sort"),
        "original_order": min(int(row.get("original_order") or 0) for row in rows),
        "is_tail": True,
    }
    return [partial_group_mapping_row(row, final, label=label, is_tail=True) for row in rows]


def weighted_average(values: Any) -> float | int | None:
    numerator = 0.0
    denominator = 0.0
    for raw_value, raw_weight in values:
        value = json_number(raw_value)
        weight = json_number(raw_weight)
        if value is None or weight is None:
            continue
        numerator += float(value) * float(weight)
        denominator += float(weight)
    return json_number(numerator / denominator) if denominator else None


def clean_partial_numeric_labels(rows: list[dict[str, Any]], band_width: Any) -> None:
    decimal_places = decimal_places_for_band_width(band_width)
    if decimal_places is None:
        return
    for row in rows:
        if row.get("is_tail"):
            continue
        label = format_numeric_band_label(row.get("x_sort"), decimal_places)
        if label is not None:
            row["x"] = label


def scale_partial_dependence_rows(rows: list[dict[str, Any]], *, objective: str) -> dict[str, Any]:
    target = weighted_average((row.get("fitted_num") / row.get("fitted_den") if row.get("fitted_den") else None, row.get("fitted_den")) for row in rows)
    source_mean = weighted_average((row.get("p50"), row.get("volume")) for row in rows)
    selected_objective = str(objective or "").strip().lower()
    positive_scale = selected_objective in LOG_LINK_OBJECTIVES or selected_objective in BINARY_LINK_OBJECTIVES
    if target is None or source_mean is None:
        return {"method": "none", "target": target, "source_mean": source_mean, "warning": "SHAP ribbons could not be scaled to fitted values."}
    if positive_scale:
        if source_mean == 0:
            return {"method": "none", "target": target, "source_mean": source_mean, "warning": "SHAP ribbons could not be scaled because the median SHAP mean is zero."}
        multiplier = float(target) / float(source_mean)
        for row in rows:
            for percentile in SHAP_RIBBON_PERCENTILES:
                key = percentile_key(percentile)
                value = json_number(row.get(key))
                row[key] = json_number(float(value) * multiplier) if value is not None else None
        return {"method": "multiply", "target": json_number(target), "source_mean": json_number(source_mean), "factor": json_number(multiplier)}
    shift = float(target) - float(source_mean)
    for row in rows:
        for percentile in SHAP_RIBBON_PERCENTILES:
            key = percentile_key(percentile)
            value = json_number(row.get(key))
            row[key] = json_number(float(value) + shift) if value is not None else None
    return {"method": "add", "target": json_number(target), "source_mean": json_number(source_mean), "shift": json_number(shift)}


def partial_dependence_medians(partial_dependence: dict[str, Any] | None) -> dict[str, float]:
    medians: dict[str, float] = {}
    if not partial_dependence:
        return medians
    overlay = shap_overlay(partial_dependence)
    for row in overlay.get("rows") or []:
        if not isinstance(row, dict):
            continue
        median = json_number(row.get("p50"))
        if median is not None:
            medians[str(row.get("x"))] = float(median)
    return medians


def transform_partial_dependence_overlay(
    partial_dependence: dict[str, Any],
    transform: str,
    warnings: list[str],
    *,
    x_kind: str,
    base: Any,
    band_width: Any,
) -> None:
    overlays = partial_dependence.get("overlays")
    if isinstance(overlays, dict):
        for key, overlay in overlays.items():
            if isinstance(overlay, dict):
                transform_partial_dependence_overlay(
                    overlay,
                    transform,
                    warnings,
                    x_kind=x_kind,
                    base=base,
                    band_width=band_width,
                )
                if key == "glm":
                    overlay["mode"] = "glm"
        partial_dependence["warnings"] = partial_dependence_warnings(partial_dependence)
        partial_dependence["transform"] = {"mode": transform}
        return
    rows = partial_dependence.get("rows")
    if not isinstance(rows, list):
        return
    overlay_label = "GLM overlay" if str(partial_dependence.get("mode") or "") == "glm" else "SHAP ribbon"
    metadata = partial_dependence_transform_metadata(rows, transform, warnings, x_kind=x_kind, base=base, band_width=band_width, overlay_label=overlay_label)
    invalid_count = 0
    reference = json_number(metadata.get("value"))
    keys = partial_dependence_value_keys(partial_dependence)
    for row in rows:
        for key in keys:
            before = row.get(key)
            row[key] = transform_value(before, transform, reference)
            if before is not None and row[key] is None:
                invalid_count += 1
    if invalid_count:
        partial_dependence.setdefault("warnings", []).append(f"{invalid_count} {overlay_label} values could not be shown because they are outside the {transform} transform domain.")
    partial_dependence["transform"] = metadata


def partial_dependence_transform_metadata(
    rows: list[dict[str, Any]],
    transform: str,
    warnings: list[str],
    *,
    x_kind: str,
    base: Any,
    band_width: Any,
    overlay_label: str = "SHAP ribbon",
) -> dict[str, Any]:
    base_text = str(base or "").strip()
    average = weighted_average((row.get("p50"), row.get("volume")) for row in rows)
    metadata: dict[str, Any] = {
        "mode": transform,
        "base": base_text,
        "reference": "overall_average",
        "base_x": None,
        "value": json_number(average),
    }
    if transform not in {"zero", "one"}:
        return metadata
    if not base_text:
        return metadata
    base_row = base_reference_row(rows, x_kind=x_kind, base=base_text, band_width=band_width)
    if base_row is None:
        warnings.append(f"Base value {base_text} could not be matched on the x-axis; using overall response averages for {overlay_label} {transform} transform.")
        return metadata
    reference = json_number(base_row.get("p50"))
    if reference is None or (transform == "one" and reference == 0):
        warnings.append(f"Base value {base_text} has no usable {overlay_label} reference; using overall response averages for the {transform} transform.")
        return metadata
    metadata.update({"reference": "base", "base_x": base_row.get("x"), "value": reference})
    return metadata


def order_partial_dependence_rows(partial_dependence: dict[str, Any], sorted_rows: list[dict[str, Any]]) -> None:
    overlays = partial_dependence.get("overlays")
    if isinstance(overlays, dict):
        for overlay in overlays.values():
            if isinstance(overlay, dict):
                order_partial_dependence_rows(overlay, sorted_rows)
        partial_dependence["rows"] = overlays.get("shap", {}).get("rows", []) if isinstance(overlays.get("shap"), dict) else []
        partial_dependence["warnings"] = partial_dependence_warnings(partial_dependence)
        return
    rows = [row for row in partial_dependence.get("rows") or [] if isinstance(row, dict)]
    by_x = {str(row.get("x")): row for row in rows}
    ordered: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in sorted_rows:
        label = str(row.get("x"))
        match = by_x.get(label)
        if match is not None:
            ordered.append(match)
            seen.add(label)
    partial_dependence["rows"] = ordered
    if rows and not ordered:
        label = "GLM overlay" if str(partial_dependence.get("mode") or "") == "glm" else "SHAP ribbon"
        partial_dependence.setdefault("warnings", []).append(f"{label} groups did not match the rendered chart groups.")


def shap_overlay(partial_dependence: dict[str, Any]) -> dict[str, Any]:
    overlays = partial_dependence.get("overlays")
    if isinstance(overlays, dict) and isinstance(overlays.get("shap"), dict):
        return overlays["shap"]
    return partial_dependence if str(partial_dependence.get("mode") or "") == "shap" else {}


def partial_dependence_warnings(partial_dependence: dict[str, Any]) -> list[str]:
    warnings: list[str] = []
    overlays = partial_dependence.get("overlays")
    if isinstance(overlays, dict):
        for overlay in overlays.values():
            if isinstance(overlay, dict):
                warnings.extend(str(warning) for warning in (overlay.get("warnings") or []) if warning)
        return warnings
    return [str(warning) for warning in (partial_dependence.get("warnings") or []) if warning]


def partial_dependence_value_keys(partial_dependence: dict[str, Any]) -> list[str]:
    raw_percentiles = partial_dependence.get("percentiles")
    if isinstance(raw_percentiles, list) and raw_percentiles:
        keys = [percentile_key(int(number)) for percentile in raw_percentiles if (number := json_number(percentile)) is not None]
        return keys or ["p50"]
    return [percentile_key(percentile) for percentile in SHAP_RIBBON_PERCENTILES]


def apply_transform(
    rows: list[dict[str, Any]],
    responses: list[dict[str, str]],
    transform: str,
    sigma_multiplier: float,
    warnings: list[str],
    *,
    reference_rows: list[dict[str, Any]] | None = None,
    x_kind: str = "",
    base: Any = None,
    band_width: Any = None,
    transform_metadata: dict[str, Any] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if transform_metadata is None:
        reference_source = reference_rows if reference_rows is not None else rows
        averages: dict[int, float | None] = {}
        for index, _ in enumerate(responses):
            num = sum(float(row.get(f"resp{index}_num") or 0) for row in reference_source)
            den = sum(float(row.get(f"resp{index}_den") or 0) for row in reference_source)
            averages[index] = num / den if den else None
        references = transform_references(reference_source, responses, transform, averages, warnings, x_kind=x_kind, base=base, band_width=band_width)
    else:
        raw_values = transform_metadata.get("values") if isinstance(transform_metadata, dict) else []
        references = {
            "values": {
                index: json_number(raw_values[index]) if isinstance(raw_values, list) and index < len(raw_values) else None
                for index, _ in enumerate(responses)
            },
            "metadata": transform_metadata,
        }

    display: list[dict[str, Any]] = []
    invalid_count = 0
    for row in rows:
        out = {
            "x": row["x"],
            "volume": row["volume"],
            "row_count": row.get("row_count", 0),
            "is_tail": bool(row.get("is_tail")),
            "valid_folds": row.get("valid_folds"),
        }
        if row.get("x_sort") is not None:
            out["x_sort"] = row.get("x_sort")
        if row.get("x_start") is not None:
            out["x_start"] = row.get("x_start")
        if row.get("x_end") is not None:
            out["x_end"] = row.get("x_end")
        for index, _ in enumerate(responses):
            out[f"resp{index}_num"] = row.get(f"resp{index}_num")
            out[f"resp{index}_den"] = row.get(f"resp{index}_den")
            out[f"resp{index}"] = transform_value(row.get(f"resp{index}"), transform, references["values"][index])
            if row.get(f"resp{index}") is not None and out[f"resp{index}"] is None:
                invalid_count += 1
        if sigma_multiplier > 0 and len(responses) >= 2 and row.get("sigma_se") is not None and row.get("resp1") is not None:
            se = float(row["sigma_se"])
            expected = float(row["resp1"])
            out["resp1_low"] = transform_value(expected - sigma_multiplier * se, transform, references["values"][1])
            out["resp1_high"] = transform_value(expected + sigma_multiplier * se, transform, references["values"][1])
        display.append(out)

    if invalid_count:
        warnings.append(f"{invalid_count} response values could not be shown because they are outside the {transform} transform domain.")
    return display, references["metadata"]


def transform_references(
    rows: list[dict[str, Any]],
    responses: list[dict[str, str]],
    transform: str,
    averages: dict[int, float | None],
    warnings: list[str],
    *,
    x_kind: str,
    base: Any,
    band_width: Any,
) -> dict[str, Any]:
    base_text = str(base or "").strip()
    metadata: dict[str, Any] = {
        "mode": transform,
        "base": base_text,
        "reference": "overall_average",
        "base_x": None,
        "values": [json_number(averages.get(index)) for index, _ in enumerate(responses)],
    }
    if transform not in {"zero", "one"} or not base_text:
        return {"values": averages, "metadata": metadata}

    base_row = base_reference_row(rows, x_kind=x_kind, base=base_text, band_width=band_width)
    if base_row is None:
        warnings.append(f"Base value {base_text} could not be matched on the x-axis; using overall response averages for the {transform} transform.")
        return {"values": averages, "metadata": metadata}

    values: dict[int, float | None] = {}
    failed_indexes: list[int] = []
    for index, _ in enumerate(responses):
        reference = json_number(base_row.get(f"resp{index}"))
        if reference is None or (transform == "one" and reference == 0):
            values[index] = averages.get(index)
            failed_indexes.append(index)
        else:
            values[index] = float(reference)
    if failed_indexes:
        labels = ", ".join(responses[index]["label"] for index in failed_indexes)
        warnings.append(f"Base value {base_text} has no usable {labels} response reference; using overall response averages for those {transform} transforms.")

    metadata.update(
        {
            "reference": "base",
            "base_x": base_row.get("x"),
            "values": [json_number(values.get(index)) for index, _ in enumerate(responses)],
            "fallback_responses": failed_indexes,
        }
    )
    return {"values": values, "metadata": metadata}


def base_reference_row(rows: list[dict[str, Any]], *, x_kind: str, base: str, band_width: Any) -> dict[str, Any] | None:
    if not rows:
        return None
    if x_kind in {"integer", "numeric"}:
        return numeric_base_reference_row(rows, base=base, band_width=band_width)
    if x_kind == "quantile":
        return None
    target = base.strip().lower()
    return next((row for row in rows if str(row.get("x") or "").strip().lower() == target), None)


def numeric_base_reference_row(rows: list[dict[str, Any]], *, base: str, band_width: Any) -> dict[str, Any] | None:
    base_number = json_number(base)
    if base_number is None:
        return None
    candidates = [row for row in rows if not row.get("is_tail") and json_number(row.get("x_sort")) is not None]
    if not candidates:
        candidates = [row for row in rows if json_number(row.get("x_sort")) is not None]
    if not candidates:
        return None
    width = parse_positive_float(band_width)
    if width:
        for row in candidates:
            start = json_number(row.get("x_sort"))
            if start is None:
                continue
            if float(start) <= float(base_number) < float(start) + width:
                return row
    exact = next((row for row in candidates if json_number(row.get("x_sort")) == base_number), None)
    if exact:
        return exact
    return min(candidates, key=lambda row: abs(float(json_number(row.get("x_sort")) or 0) - float(base_number)))


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
    "build_denominator_summary_sql",
    "build_chart_sql",
    "build_response_summary_sql",
    "build_x_sql",
    "chart",
    "table",
    "combine_rows",
    "combine_sigma",
    "denominator_warnings",
    "normalise_denominator",
    "normalise_quantile_count",
    "normalise_responses",
    "normalise_row",
    "parse_group_threshold",
    "summarize_denominator",
    "response_parts",
    "response_summary",
    "sort_rows",
    "transform_value",
    "use_quantiles",
]
