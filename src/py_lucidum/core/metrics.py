from __future__ import annotations

from typing import Any

from .dataset import Dataset
from .schema import ColumnInfo, is_numeric_kind, json_number
from .sql import quote_ident


def normalise_denominator(raw: Any, columns: dict[str, ColumnInfo]) -> dict[str, str | None]:
    value = str(raw or "__none__")
    if value in {"", "__none__"}:
        return {"column": None, "label": "Average row value", "bar_label": "Row count"}
    if value not in columns or not is_numeric_kind(columns[value].kind):
        raise ValueError("Choose a valid numeric Weight column")
    return {"column": value, "label": value, "bar_label": value}


def response_value_sql(response: dict[str, str]) -> str:
    return f"TRY_CAST({quote_ident(str(response['numerator']))} AS DOUBLE)"


def denominator_value_sql(denominator: dict[str, str | None]) -> str:
    column = denominator.get("column")
    if column:
        return f"TRY_CAST({quote_ident(str(column))} AS DOUBLE)"
    return "1"


def missing_response_condition(responses: list[dict[str, str]]) -> str:
    if not responses:
        return "FALSE"
    checks = [f"{response_value_sql(response)} IS NULL" for response in responses]
    return " OR ".join(checks)


def denominator_valid_condition(responses: list[dict[str, str]], denominator: dict[str, str | None]) -> str:
    checks = [f"{response_value_sql(response)} IS NOT NULL" for response in responses]
    column = denominator.get("column")
    if column:
        checks.append(f"{denominator_value_sql(denominator)} IS NOT NULL")
    return " AND ".join(checks) if checks else "TRUE"


def weighted_value_sql(denominator: dict[str, str | None], valid_condition: str) -> str:
    return f"CASE WHEN {valid_condition} THEN {denominator_value_sql(denominator)} ELSE NULL END"


def response_parts(response: dict[str, str], index: int) -> tuple[str, str, str]:
    numerator = f"TRY_CAST({quote_ident(str(response['numerator']))} AS DOUBLE)"
    num_alias = f"resp{index}_num"
    den_alias = f"resp{index}_den"
    value_alias = f"resp{index}"
    num_expr = f"SUM(CASE WHEN __weight_value IS NOT NULL THEN {numerator} ELSE NULL END)"
    den_expr = "COALESCE(SUM(__weight_value), 0)"
    value_expr = f"{num_alias} / NULLIF({den_alias}, 0)"
    return (
        f"{num_expr} AS {num_alias}",
        f"{den_expr} AS {den_alias}",
        f"{value_expr} AS {value_alias}",
    )


def response_summary(
    dataset: Dataset,
    responses: list[dict[str, str]],
    denominator: dict[str, str | None],
    filter_sql: str = "",
    source_id: Any = None,
) -> list[dict[str, Any]]:
    if not responses:
        return []
    sql = build_response_summary_sql(dataset.relation_sql_for_source(source_id), responses, denominator, filter_sql)
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


def build_response_summary_sql(
    relation: str,
    responses: list[dict[str, str]],
    denominator: dict[str, str | None],
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

    metric_sql = ",\n    ".join(metric_selects)
    value_sql = ",\n    ".join(value_selects)
    where_sql = f"\n  WHERE ({filter_sql})" if filter_sql else ""
    return f"""
WITH base AS (
  SELECT * FROM {relation}{where_sql}
),
weighted AS (
  SELECT
    {weight_expr} AS __weight_value,
    *
  FROM base
),
summary AS (
  SELECT
    {metric_sql}
  FROM weighted
)
SELECT
    *,
    {value_sql}
FROM summary
"""


def summarize_denominator(
    dataset: Dataset,
    responses: list[dict[str, str]],
    denominator: dict[str, str | None],
    filter_sql: str = "",
    source_id: Any = None,
) -> dict[str, Any]:
    sql = build_denominator_summary_sql(dataset.relation_sql_for_source(source_id), responses, denominator, filter_sql)
    cursor = dataset.con.execute(sql)
    fetched = cursor.fetchone()
    return dict(zip([d[0] for d in cursor.description], fetched or []))


def build_denominator_summary_sql(
    relation: str,
    responses: list[dict[str, str]],
    denominator: dict[str, str | None],
    filter_sql: str = "",
) -> str:
    valid_condition = denominator_valid_condition(responses, denominator)
    weight_expr = weighted_value_sql(denominator, valid_condition)
    response_missing = missing_response_condition(responses)
    where_sql = f"\n  WHERE ({filter_sql})" if filter_sql else ""
    column = denominator.get("column")
    if column:
        weight_sql = denominator_value_sql(denominator)
        extra_selects = f""",
    SUM(CASE WHEN {weight_sql} IS NULL THEN 1 ELSE 0 END) AS missing_weight_rows,
    SUM(CASE WHEN {weight_sql} = 0 THEN 1 ELSE 0 END) AS zero_weight_rows,
    SUM(CASE WHEN {weight_sql} < 0 THEN 1 ELSE 0 END) AS negative_weight_rows"""
    else:
        extra_selects = """,
    0 AS missing_weight_rows,
    0 AS zero_weight_rows,
    0 AS negative_weight_rows"""
    return f"""
WITH base AS (
  SELECT * FROM {relation}{where_sql}
),
weighted AS (
  SELECT
    {weight_expr} AS __weight_value,
    *
  FROM base
)
SELECT
    COALESCE(SUM(__weight_value), 0) AS value,
    SUM(CASE WHEN {response_missing} THEN 1 ELSE 0 END) AS missing_response_rows
    {extra_selects}
FROM weighted
"""


def denominator_warnings(denominator: dict[str, str | None], summary: dict[str, Any]) -> list[str]:
    warnings: list[str] = []
    missing_response_rows = int(summary.get("missing_response_rows") or 0)
    if missing_response_rows:
        warnings.append(
            f"{format_rows(missing_response_rows)} excluded from Weight because one or more selected response values were missing."
        )
    missing_weight_rows = int(summary.get("missing_weight_rows") or 0)
    if missing_weight_rows:
        warnings.append(f"{format_rows(missing_weight_rows)} excluded from Weight because {denominator['label']} was missing.")
    zero_weight_rows = int(summary.get("zero_weight_rows") or 0)
    if zero_weight_rows:
        warnings.append(f"{format_rows(zero_weight_rows)} {row_verb(zero_weight_rows)} zero {denominator['label']}.")
    negative_weight_rows = int(summary.get("negative_weight_rows") or 0)
    if negative_weight_rows:
        warnings.append(f"{format_rows(negative_weight_rows)} {row_verb(negative_weight_rows)} negative {denominator['label']}.")
    return warnings


def format_rows(count: int) -> str:
    label = "row" if count == 1 else "rows"
    return f"{count:,} {label}"


def row_verb(count: int) -> str:
    return "has" if count == 1 else "have"


__all__ = [
    "build_denominator_summary_sql",
    "build_response_summary_sql",
    "denominator_valid_condition",
    "denominator_value_sql",
    "denominator_warnings",
    "format_rows",
    "missing_response_condition",
    "normalise_denominator",
    "response_parts",
    "response_summary",
    "response_value_sql",
    "row_verb",
    "summarize_denominator",
    "weighted_value_sql",
]
