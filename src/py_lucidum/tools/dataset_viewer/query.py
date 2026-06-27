from __future__ import annotations

import datetime as dt
from decimal import Decimal
from typing import Any

from py_lucidum.core import Dataset, json_number, quote_ident


DEFAULT_TABLE_LIMIT = 100
MAX_TABLE_LIMIT = 100


def table(dataset: Dataset, request: dict[str, Any]) -> dict[str, Any]:
    with dataset.lock:
        readable_columns = dataset.valid_schema_columns()
        invalid_columns = dataset.invalid_columns()
        relation = readable_relation_sql(dataset, readable_columns)
        filter_sql = dataset.normalise_filter_for_relation(request.get("filter"), relation)
        max_rows = normalise_limit(request.get("limit"))
        rows, has_more = data_rows(dataset, relation, readable_columns, filter_sql, max_rows)
        warnings = dataset.invalid_column_warnings(invalid_columns)
        return {
            "displayed_row_count": len(rows),
            "max_rows": max_rows,
            "has_more": has_more,
            "filter": filter_sql,
            "columns": [
                {
                    "name": column.name,
                    "field": f"c{index}",
                    "kind": column.kind,
                    "duckdb_type": column.duckdb_type,
                }
                for index, column in enumerate(readable_columns)
            ],
            "rows": rows,
            "warnings": warnings,
        }


def normalise_limit(value: Any) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        parsed = DEFAULT_TABLE_LIMIT
    return max(1, min(parsed, MAX_TABLE_LIMIT))


def readable_relation_sql(dataset: Dataset, columns: list[Any]) -> str:
    projection = readable_projection(columns)
    return f"(SELECT {projection} FROM {dataset.relation_sql()})"


def readable_projection(columns: list[Any]) -> str:
    if not columns:
        return "1 AS __lucidum_empty_dataset"
    return ", ".join(quote_ident(column.name) for column in columns)


def preview_projection(columns: list[Any]) -> str:
    if not columns:
        return "1 AS __lucidum_empty_dataset"
    return ", ".join(
        f"{quote_ident(column.name)} AS {quote_ident(f'c{index}')}"
        for index, column in enumerate(columns)
    )


def data_rows(
    dataset: Dataset,
    relation: str,
    columns: list[Any],
    filter_sql: str,
    limit: int,
) -> tuple[list[dict[str, Any]], bool]:
    where_sql = f" WHERE ({filter_sql})" if filter_sql else ""
    fetch_limit = int(limit) + 1
    sql = f"""
SELECT
  {preview_projection(columns)}
FROM {relation}
{where_sql}
LIMIT {fetch_limit}
"""
    cursor = dataset.con.execute(sql)
    names = [description[0] for description in cursor.description]
    fetched = cursor.fetchall()
    has_more = len(fetched) > limit
    preview_rows = fetched[:limit]
    rows = []
    for index, row in enumerate(preview_rows, start=1):
        payload = {"__row_id": index}
        if columns:
            payload.update({name: json_value(value) for name, value in zip(names, row)})
        rows.append(payload)
    return rows, has_more


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
