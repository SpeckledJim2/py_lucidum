from __future__ import annotations

import math
from decimal import Decimal, InvalidOperation
from typing import Any

from py_lucidum.core import (
    ColumnInfo,
    Dataset,
    ModelPredictionSource,
    build_denominator_summary_sql,
    build_response_summary_sql,
    denominator_valid_condition,
    denominator_warnings,
    normalise_denominator,
    is_numeric_kind,
    json_number,
    parse_positive_float,
    quote_ident,
    response_parts,
    response_summary,
    sql_literal,
    summarize_denominator,
    weighted_value_sql,
)
from py_lucidum.tools.gbm.shap import FLAME_PERCENTILES as SHAP_RIBBON_PERCENTILES
from py_lucidum.tools.gbm.shap import percentile_key, percentile_selects
from py_lucidum.tools.gbm.validation import CROSS_ENTROPY_OBJECTIVES, LOG_LINK_OBJECTIVES


BINARY_LINK_OBJECTIVES = {"binary", *CROSS_ENTROPY_OBJECTIVES}


def chart(dataset: Dataset, request: dict[str, Any]) -> dict[str, Any]:
    with dataset.lock:
        context = chart_context(dataset, request)
        source_id = context["source_id"]
        relation = context["relation"]
        columns = context["columns"]
        x_col = str(request.get("x") or "")
        if x_col not in columns:
            raise ValueError("Choose a valid x-axis feature")

        filter_sql = dataset.normalise_filter_for_relation(request.get("filter"), relation)
        responses = normalise_responses(request.get("responses"), columns)
        denominator = normalise_denominator(request.get("denominator", request.get("weight")), columns)
        x_info = columns[x_col]
        quantile_count = (
            normalise_quantile_count(request.get("bandWidth"))
            if use_quantiles(request.get("quantileMode")) and is_numeric_kind(x_info.kind)
            else None
        )
        x_sql = build_x_sql(
            x_col=x_col,
            kind=x_info.kind,
            band_width=request.get("bandWidth"),
            date_bucket=request.get("dateBucket"),
            quantile_count=quantile_count,
        )
        x_group_kind = "quantile" if quantile_count else x_info.kind
        sigma_multiplier = float(request.get("sigma") or 0)
        include_sigma = sigma_multiplier > 0 and len(responses) >= 2
        row_count = context["row_count"]
        filtered_row_count = relation_row_count(dataset, relation, filter_sql)
        denominator_summary = relation_denominator_summary(dataset, relation, responses, denominator, filter_sql)
        response_summaries = relation_response_summary(dataset, relation, responses, denominator, filter_sql)
        sql = build_chart_sql(relation, x_sql, responses, denominator, include_sigma, filter_sql)
        raw_rows = [
            dict(zip([d[0] for d in dataset.con.description], row))
            for row in dataset.con.execute(sql).fetchall()
        ]

        grouped_rows = apply_low_weight_grouping(
            rows=raw_rows,
            responses=responses,
            x_kind=x_group_kind,
            threshold=str(request.get("lowGroup") or "0"),
        )
        partial_dependence = build_partial_dependence_overlay(
            dataset,
            request,
            columns=columns,
            x_col=x_col,
            x_sql=x_sql,
            x_group_kind=x_group_kind,
            responses=responses,
            denominator=denominator,
        )
        partial_medians = partial_dependence_medians(partial_dependence)
        sorted_rows = sort_rows(grouped_rows, x_group_kind, str(request.get("sort") or "alpha"), partial_medians)
        clean_numeric_band_labels(sorted_rows, x_group_kind, request.get("bandWidth"))
        max_groups = int(request.get("maxGroups") or 10000)
        if len(sorted_rows) > max_groups:
            sorted_rows = sorted_rows[:max_groups]

        transform = str(request.get("transform") or "none")
        warnings: list[str] = []
        warnings.extend(denominator_warnings(denominator, denominator_summary))
        display_rows, transform_metadata = apply_transform(
            sorted_rows,
            responses,
            transform,
            sigma_multiplier,
            warnings,
            x_kind=x_group_kind,
            base=request.get("base"),
            band_width=request.get("bandWidth"),
        )
        if partial_dependence:
            transform_partial_dependence_overlay(
                partial_dependence,
                transform,
                warnings,
                x_kind=x_group_kind,
                base=request.get("base"),
                band_width=request.get("bandWidth"),
            )
            order_partial_dependence_rows(partial_dependence, sorted_rows)
            warnings.extend(partial_dependence.get("warnings") or [])

        payload = {
            "x": x_col,
            "x_kind": x_info.kind,
            "source": source_id,
            "row_count": row_count,
            "filtered_row_count": filtered_row_count,
            "filter": filter_sql,
            "responses": [
                {"label": r["label"], "numerator": r["numerator"], **({"source": r["source"]} if r.get("source") else {})}
                for r in responses
            ],
            **({"field_sources": context["field_sources"]} if context.get("field_sources") else {}),
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
            "response_summaries": response_summaries,
            "rows": display_rows,
            "warnings": warnings,
            "transform": transform_metadata,
        }
        if partial_dependence:
            payload["partial_dependence"] = partial_dependence
        return payload


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
    raw_responses = request.get("responses")
    if not isinstance(raw_responses, list):
        return False
    return any(isinstance(item, dict) and str(item.get("source") or "").strip() for item in raw_responses)


def mixed_chart_context(dataset: Dataset, request: dict[str, Any], source_id: str) -> dict[str, Any]:
    dataset_columns = dataset.column_map()
    columns = dict(dataset_columns)
    prediction_sources: dict[str, ModelPredictionSource] = {}
    field_sources: dict[str, Any] = {"x": "dataset", "responses": []}
    x_col = str(request.get("x") or "")
    x_source = field_source_id(dataset, request.get("xSource"), source_id)
    field_sources["x"] = x_source
    add_field_column(dataset, columns, dataset_columns, prediction_sources, x_col, x_source)
    raw_responses = request.get("responses")
    if isinstance(raw_responses, list):
        for item in raw_responses:
            if not isinstance(item, dict):
                continue
            response_source = field_source_id(dataset, item.get("source"), source_id)
            field_sources["responses"].append(response_source)
            add_field_column(dataset, columns, dataset_columns, prediction_sources, str(item.get("numerator") or ""), response_source)
    relation = mixed_relation_sql(dataset, list(prediction_sources.values()))
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
    if prediction_source is not None and column_name == prediction_source.column:
        if not prediction_source.column or not prediction_source.relation_sql:
            raise ValueError("Choose a valid model prediction source")
        prediction_sources[prediction_source.source_id] = prediction_source
        columns[column_name] = ColumnInfo(name=column_name, duckdb_type="DOUBLE", kind="numeric")
        return
    if column_name in dataset_columns:
        columns[column_name] = dataset_columns[column_name]


def mixed_relation_sql(dataset: Dataset, prediction_sources: list[ModelPredictionSource]) -> str:
    dataset_columns = [column.name for column in dataset.valid_schema_columns()]
    source_column_sql = ",\n    ".join(quote_ident(name) for name in dataset_columns)
    source_column_suffix = f",\n    {source_column_sql}" if source_column_sql else ""
    joins: list[str] = []
    selects = [f"base.{quote_ident(name)}" for name in dataset_columns]
    scope_sql = ""
    if prediction_sources:
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
    return responses[:2]


def use_quantiles(value: Any) -> bool:
    if value is True:
        return True
    if value is False or value is None:
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "on", "quantile", "quantiles"}


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
    return {"key": key, "label": key, "sort": key}


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
    CAST(hash({rownum_expr}) % 20 AS INTEGER) AS __fold,
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
    COALESCE(SUM(__weight_value), 0) AS volume
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
    if x_kind == "quantile":
        missing_rows = [row for row in normalised if row["x"] == "Missing"]
        normalised = [row for row in normalised if row["x"] != "Missing"]
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
        "is_tail": False,
        "sigma_se": json_number(row.get("sigma_se")),
        "valid_folds": json_number(row.get("valid_folds")),
        "sigma_folds": row.get("sigma_folds"),
    }
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
    columns: dict[str, ColumnInfo],
    x_col: str,
    x_sql: dict[str, str],
    x_group_kind: str,
    responses: list[dict[str, str]],
    denominator: dict[str, str | None],
) -> dict[str, Any] | None:
    if not partial_dependence_mode_is_shap(request.get("partialDependence")):
        return None
    source = active_gbm_shap_source(dataset)
    if source is None:
        return empty_partial_dependence_warning("No active GBM SHAP values are available.")
    model_id = str(source.get("model_id") or "")
    shap_source_id = str(source.get("id") or "")
    prediction_source = active_gbm_prediction_source(dataset, model_id)
    if not model_id or not shap_source_id or prediction_source is None:
        return empty_partial_dependence_warning("The active GBM needs both SHAP values and predictions for SHAP ribbons.")

    shap_column = shap_value_column_for_feature(source, x_col)
    if not shap_column:
        return empty_partial_dependence_warning(f"No active GBM SHAP values are available for {x_col}.")
    shap_source_columns = dataset.column_map_for_source(shap_source_id)
    if x_col not in shap_source_columns:
        return empty_partial_dependence_warning(f"The active GBM SHAP source does not include {x_col}.")
    if "gbm_prediction" not in shap_source_columns:
        return empty_partial_dependence_warning("The active GBM SHAP source does not include fitted predictions.")

    overlay_responses = [response for response in responses if response.get("numerator") in shap_source_columns]
    overlay_denominator = normalise_overlay_denominator(denominator, shap_source_columns)
    shap_relation = dataset.relation_sql_for_source(shap_source_id)
    filter_sql = dataset.normalise_filter_for_relation(request.get("filter"), shap_relation)
    shap_expr = shap_response_sql(str(source.get("objective") or prediction_source.get("objective") or ""), quote_ident(shap_column))
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
    if group_mapping:
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
    scale = scale_partial_dependence_rows(rows, objective=str(source.get("objective") or prediction_source.get("objective") or ""))
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


def partial_dependence_mode_is_shap(raw: Any) -> bool:
    if not isinstance(raw, dict):
        return False
    return str(raw.get("mode") or "none").strip().lower() == "shap"


def empty_partial_dependence_warning(message: str) -> dict[str, Any]:
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


def active_gbm_shap_source(dataset: Dataset) -> dict[str, Any] | None:
    sources = dataset.data_sources()
    return next((source for source in sources if source.get("kind") == "gbm_shap_long" and source.get("active")), None)


def active_gbm_prediction_source(dataset: Dataset, model_id: str) -> dict[str, Any] | None:
    sources = dataset.data_sources()
    return next(
        (
            source
            for source in sources
            if source.get("kind") == "gbm_predictions"
            and source.get("active")
            and str(source.get("model_id") or "") == model_id
        ),
        None,
    )


def shap_value_column_for_feature(source: dict[str, Any], feature: str) -> str:
    for column in source.get("columns") or []:
        if not isinstance(column, dict):
            continue
        if column.get("source_role") != "gbm_shap_value":
            continue
        if str(column.get("artifact_column") or column.get("label") or "") == feature:
            return str(column.get("name") or "")
    return ""


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
    SUM(__gbm_prediction * __weight_value) AS fitted_num,
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
        f"{sql_literal(str(row.get('source_x') or ''))}, "
        f"{sql_literal(str(row.get('final_x') or ''))}, "
        f"{sql_literal(str(row.get('final_x_sort') or ''))}, "
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
    if x_kind == "quantile":
        missing_rows = [row for row in normalised if row["x"] == "Missing"]
        normalised = [row for row in normalised if row["x"] != "Missing"]
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
    for row in partial_dependence.get("rows") or []:
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
    rows = partial_dependence.get("rows")
    if not isinstance(rows, list):
        return
    metadata = partial_dependence_transform_metadata(rows, transform, warnings, x_kind=x_kind, base=base, band_width=band_width)
    invalid_count = 0
    reference = json_number(metadata.get("value"))
    for row in rows:
        for percentile in SHAP_RIBBON_PERCENTILES:
            key = percentile_key(percentile)
            before = row.get(key)
            row[key] = transform_value(before, transform, reference)
            if before is not None and row[key] is None:
                invalid_count += 1
    if invalid_count:
        partial_dependence.setdefault("warnings", []).append(f"{invalid_count} SHAP ribbon values could not be shown because they are outside the {transform} transform domain.")
    partial_dependence["transform"] = metadata


def partial_dependence_transform_metadata(
    rows: list[dict[str, Any]],
    transform: str,
    warnings: list[str],
    *,
    x_kind: str,
    base: Any,
    band_width: Any,
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
        warnings.append(f"Base value {base_text} could not be matched on the x-axis; using overall response averages for SHAP ribbon {transform} transform.")
        return metadata
    reference = json_number(base_row.get("p50"))
    if reference is None or (transform == "one" and reference == 0):
        warnings.append(f"Base value {base_text} has no usable SHAP ribbon reference; using overall response averages for the {transform} transform.")
        return metadata
    metadata.update({"reference": "base", "base_x": base_row.get("x"), "value": reference})
    return metadata


def order_partial_dependence_rows(partial_dependence: dict[str, Any], sorted_rows: list[dict[str, Any]]) -> None:
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
        partial_dependence.setdefault("warnings", []).append("SHAP ribbon groups did not match the rendered chart groups.")


def apply_transform(
    rows: list[dict[str, Any]],
    responses: list[dict[str, str]],
    transform: str,
    sigma_multiplier: float,
    warnings: list[str],
    *,
    x_kind: str = "",
    base: Any = None,
    band_width: Any = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    averages: dict[int, float | None] = {}
    for index, _ in enumerate(responses):
        num = sum(float(row.get(f"resp{index}_num") or 0) for row in rows)
        den = sum(float(row.get(f"resp{index}_den") or 0) for row in rows)
        averages[index] = num / den if den else None
    references = transform_references(rows, responses, transform, averages, warnings, x_kind=x_kind, base=base, band_width=band_width)

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
