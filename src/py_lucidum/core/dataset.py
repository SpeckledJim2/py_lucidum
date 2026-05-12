from __future__ import annotations

import threading
from pathlib import Path
from typing import Any

import duckdb

from .schema import ColumnInfo, infer_kind, is_numeric_kind, suggested_band_width
from .sql import quote_ident, sql_literal


class Dataset:
    def __init__(self, path: str | Path):
        self.path = Path(path).expanduser().resolve()
        if not self.path.exists():
            raise FileNotFoundError(f"Dataset does not exist: {self.path}")
        self.con = duckdb.connect(database=":memory:")
        self._schema: list[ColumnInfo] | None = None
        self._row_count: int | None = None
        self._band_suggestions: dict[str, float | int | None] | None = None
        self._lock = threading.RLock()

    @property
    def lock(self) -> threading.RLock:
        return self._lock

    def relation_sql(self) -> str:
        path = sql_literal(str(self.path))
        suffix = self.path.suffix.lower()
        if suffix == ".parquet":
            return f"read_parquet({path})"
        if suffix == ".csv":
            return f"read_csv_auto({path}, header=true, ignore_errors=true)"
        raise ValueError("Only .csv and .parquet files are supported in this prototype")

    def reload(self) -> None:
        with self._lock:
            self._schema = None
            self._row_count = None
            self._band_suggestions = None

    def schema(self) -> dict[str, Any]:
        with self._lock:
            if self._schema is None:
                rows = self.con.execute(f"DESCRIBE SELECT * FROM {self.relation_sql()}").fetchall()
                base_schema = [
                    ColumnInfo(name=str(row[0]), duckdb_type=str(row[1]), kind=infer_kind(str(row[1])))
                    for row in rows
                ]
                suggestions = self.band_suggestions(base_schema)
                self._schema = [
                    ColumnInfo(
                        name=col.name,
                        duckdb_type=col.duckdb_type,
                        kind=col.kind,
                        band_suggestion=suggestions.get(col.name),
                    )
                    for col in base_schema
                ]
            return {
                "path": str(self.path),
                "row_count": self.row_count(),
                "columns": [
                    {
                        "name": c.name,
                        "duckdb_type": c.duckdb_type,
                        "kind": c.kind,
                        "band_suggestion": c.band_suggestion,
                    }
                    for c in self._schema
                ],
            }

    def row_count(self) -> int:
        if self._row_count is None:
            self._row_count = int(self.con.execute(f"SELECT COUNT(*) FROM {self.relation_sql()}").fetchone()[0])
        return self._row_count

    def band_suggestions(self, schema: list[ColumnInfo]) -> dict[str, float | int | None]:
        if self._band_suggestions is not None:
            return self._band_suggestions
        numeric = [col for col in schema if is_numeric_kind(col.kind)]
        if not numeric:
            self._band_suggestions = {}
            return self._band_suggestions
        select_parts: list[str] = []
        aliases: dict[str, tuple[str, str]] = {}
        for index, col in enumerate(numeric):
            raw = quote_ident(col.name)
            std_alias = f"c{index}_std"
            select_parts.append(f"STDDEV_SAMP(TRY_CAST({raw} AS DOUBLE)) AS {quote_ident(std_alias)}")
            aliases[std_alias] = (col.name, "std")
        select_sql = ",\n    ".join(select_parts)
        sql = f"""
WITH sample AS (
  SELECT * FROM {self.relation_sql()} LIMIT 10000
)
SELECT
    {select_sql}
FROM sample
"""
        row = self.con.execute(sql).fetchone()
        metrics: dict[str, dict[str, Any]] = {col.name: {} for col in numeric}
        for description, value in zip(self.con.description, row):
            metric = aliases.get(description[0])
            if metric:
                name, key = metric
                metrics[name][key] = value
        integer_columns = [col for col in numeric if col.kind == "integer"]
        if integer_columns:
            range_parts: list[str] = []
            range_aliases: dict[str, tuple[str, str]] = {}
            for index, col in enumerate(integer_columns):
                raw = quote_ident(col.name)
                min_alias = f"i{index}_min"
                max_alias = f"i{index}_max"
                range_parts.append(f"MIN(TRY_CAST({raw} AS BIGINT)) AS {quote_ident(min_alias)}")
                range_parts.append(f"MAX(TRY_CAST({raw} AS BIGINT)) AS {quote_ident(max_alias)}")
                range_aliases[min_alias] = (col.name, "min")
                range_aliases[max_alias] = (col.name, "max")
            range_sql = f"SELECT {', '.join(range_parts)} FROM {self.relation_sql()}"
            range_row = self.con.execute(range_sql).fetchone()
            for description, value in zip(self.con.description, range_row):
                metric = range_aliases.get(description[0])
                if metric:
                    name, key = metric
                    metrics[name][key] = value
        suggestions: dict[str, float | int | None] = {}
        kinds = {col.name: col.kind for col in numeric}
        for name, values in metrics.items():
            if kinds[name] == "integer" and values.get("min") is not None and values.get("max") is not None:
                if values["max"] - values["min"] < 120:
                    suggestions[name] = 1
                    continue
            suggestions[name] = suggested_band_width(values.get("std"))
        self._band_suggestions = suggestions
        return self._band_suggestions

    def column_map(self) -> dict[str, ColumnInfo]:
        return {c.name: c for c in self._schema_columns()}

    def _schema_columns(self) -> list[ColumnInfo]:
        self.schema()
        assert self._schema is not None
        return self._schema

    def normalise_filter(self, raw: Any) -> str:
        expression = str(raw or "").strip()
        if not expression:
            return ""
        forbidden = (";", "--", "/*", "*/")
        if any(token in expression for token in forbidden):
            raise ValueError("Filter must be a single DuckDB expression without statement separators or comments")
        try:
            self.con.execute(f"SELECT 1 FROM {self.relation_sql()} WHERE ({expression}) LIMIT 0")
        except duckdb.Error as exc:
            message = str(exc).splitlines()[0]
            raise ValueError(f"Invalid filter: {message}") from exc
        return expression

    def filtered_row_count(self, filter_sql: str) -> int:
        if not filter_sql:
            return self.row_count()
        value = self.con.execute(f"SELECT COUNT(*) FROM {self.relation_sql()} WHERE ({filter_sql})").fetchone()[0]
        return int(value)
