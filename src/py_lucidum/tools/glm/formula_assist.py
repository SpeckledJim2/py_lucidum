from __future__ import annotations

import datetime as dt
from decimal import Decimal
from typing import Any

from py_lucidum.core import Dataset, is_numeric_kind, json_number, quote_ident


FORMULA_LEVEL_DEFAULT_LIMIT = 500
FORMULA_LEVEL_MAX_LIMIT = 2_000


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


def formula_level_limit(raw: Any) -> int:
    try:
        limit = int(raw)
    except (TypeError, ValueError):
        limit = FORMULA_LEVEL_DEFAULT_LIMIT
    return max(1, min(FORMULA_LEVEL_MAX_LIMIT, limit))


def formula_levels(dataset: Dataset, payload: dict[str, Any]) -> dict[str, Any]:
    with dataset.lock:
        column_name = str(payload.get("column") or "").strip()
        columns = dataset.column_map()
        column = columns.get(column_name)
        if column is None:
            raise ValueError("Choose a valid formula column")
        if is_numeric_kind(column.kind):
            raise ValueError("Choose a categorical formula column")

        column_sql = quote_ident(column.name)
        relation_sql = dataset.relation_sql()
        search = str(payload.get("search") or "").strip()
        limit = formula_level_limit(payload.get("limit", FORMULA_LEVEL_DEFAULT_LIMIT))
        total_row = dataset.con.execute(
            f"SELECT COUNT(DISTINCT {column_sql}) FROM {relation_sql} WHERE {column_sql} IS NOT NULL"
        ).fetchone()
        distinct_count = int(total_row[0] or 0) if total_row else 0

        where_sql = f"WHERE {column_sql} IS NOT NULL"
        parameters: list[Any] = []
        if search:
            where_sql += f" AND CAST({column_sql} AS VARCHAR) ILIKE ?"
            parameters.append(f"%{search}%")

        sql = f"""
WITH grouped AS (
  SELECT
    {column_sql} AS value,
    CAST({column_sql} AS VARCHAR) AS label,
    COUNT(*) AS count
  FROM {relation_sql}
  {where_sql}
  GROUP BY value, label
)
SELECT value, label, count
FROM grouped
ORDER BY LOWER(label), label
LIMIT {limit + 1}
"""
        rows = dataset.con.execute(sql, parameters).fetchall()
        visible_rows = rows[:limit]
        return {
            "column": column.name,
            "kind": column.kind,
            "distinct_count": distinct_count,
            "values": [
                {
                    "value": json_value(value),
                    "label": str(label if label is not None and label != "" else "(blank)"),
                    "count": int(count or 0),
                }
                for value, label, count in visible_rows
            ],
            "truncated": len(rows) > limit,
        }


__all__ = ["formula_level_limit", "formula_levels"]
