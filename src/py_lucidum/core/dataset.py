from __future__ import annotations

import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import duckdb

from .schema import ColumnInfo, duckdb_error_message, infer_kind, is_numeric_kind, suggested_band_width
from .sql import quote_ident, sql_literal


@dataclass(frozen=True)
class ModelPredictionSource:
    source_id: str
    column: str
    relation_sql: str
    active: bool = False


class Dataset:
    def __init__(self, path: str | Path):
        self.path = Path(path).expanduser().resolve()
        if not self.path.exists():
            raise FileNotFoundError(f"Dataset does not exist: {self.path}")
        self.con = duckdb.connect(database=":memory:")
        self.con.execute("PRAGMA disable_progress_bar")
        self.source_kind = "file"
        self._parquet_files: tuple[Path, ...] = ()
        self._configure_source()
        self._schema: list[ColumnInfo] | None = None
        self._invalid_column_errors: dict[str, str] | None = None
        self._row_count: int | None = None
        self._band_suggestions: dict[str, float | int | None] | None = None
        self._source_band_suggestions: dict[str, dict[str, float | int | None]] = {}
        self._lock = threading.RLock()
        self._source_providers: list[Any] = []

    @property
    def lock(self) -> threading.RLock:
        return self._lock

    @property
    def parquet_files(self) -> tuple[Path, ...]:
        return self._parquet_files

    def _configure_source(self) -> None:
        self._parquet_files = ()
        if self.path.is_dir():
            self.source_kind = "parquet_folder"
            self._parquet_files = self._direct_child_parquet_files()
            self._validate_parquet_folder_schema(self._parquet_files)
            return
        suffix = self.path.suffix.lower()
        if suffix == ".parquet":
            self.source_kind = "parquet_file"
        elif suffix == ".csv":
            self.source_kind = "csv_file"
        else:
            self.source_kind = "file"

    def _direct_child_parquet_files(self) -> tuple[Path, ...]:
        files = tuple(
            sorted(
                (
                    path
                    for path in self.path.iterdir()
                    if path.is_file() and path.suffix.lower() == ".parquet"
                ),
                key=lambda path: path.name,
            )
        )
        if not files:
            raise ValueError(f"Parquet folder contains no direct .parquet files: {self.path}")
        return files

    def _validate_parquet_folder_schema(self, files: tuple[Path, ...]) -> None:
        reference_file = files[0]
        reference_schema = self._parquet_file_schema(reference_file)
        reference_columns = dict(reference_schema)
        for path in files[1:]:
            schema = self._parquet_file_schema(path)
            columns = dict(schema)
            if columns != reference_columns:
                raise ValueError(
                    self._parquet_schema_mismatch_message(
                        reference_file,
                        path,
                        reference_columns,
                        columns,
                    )
                )

    def _parquet_file_schema(self, path: Path) -> list[tuple[str, str]]:
        try:
            rows = self.con.execute(f"DESCRIBE SELECT * FROM read_parquet({sql_literal(str(path))})").fetchall()
        except duckdb.Error as exc:
            raise ValueError(f"Could not read Parquet file {path.name}: {duckdb_error_message(exc)}") from exc
        return [(str(row[0]), str(row[1])) for row in rows]

    def _parquet_schema_mismatch_message(
        self,
        reference_file: Path,
        path: Path,
        reference_columns: dict[str, str],
        columns: dict[str, str],
    ) -> str:
        missing = [name for name in reference_columns if name not in columns]
        extra = [name for name in columns if name not in reference_columns]
        type_mismatches = [
            f"{name} expected {reference_columns[name]} got {columns[name]}"
            for name in reference_columns
            if name in columns and reference_columns[name] != columns[name]
        ]
        details: list[str] = []
        if missing:
            details.append(f"missing columns: {', '.join(missing[:5])}")
        if extra:
            details.append(f"extra columns: {', '.join(extra[:5])}")
        if type_mismatches:
            details.append(f"type mismatches: {', '.join(type_mismatches[:5])}")
        suffix = "; ".join(details) if details else "schema differs"
        return (
            "Parquet folder columns must have identical names and DuckDB types; "
            f"{path.name} differs from {reference_file.name} ({suffix})."
        )

    def _parquet_file_list_sql(self) -> str:
        return "[" + ", ".join(sql_literal(str(path)) for path in self._parquet_files) + "]"

    def relation_sql(self) -> str:
        if self.source_kind == "parquet_folder":
            return f"read_parquet({self._parquet_file_list_sql()})"
        path = sql_literal(str(self.path))
        suffix = self.path.suffix.lower()
        if suffix == ".parquet":
            return f"read_parquet({path})"
        if suffix == ".csv":
            return f"read_csv_auto({path}, header=true, ignore_errors=true)"
        raise ValueError("Only .csv and .parquet files are supported in this prototype")

    def normalise_source(self, raw: Any = None) -> str:
        source_id = str(raw or "dataset").strip()
        if source_id in {"", "dataset"}:
            return "dataset"
        for provider in self._source_providers:
            has_source = getattr(provider, "has_source", None)
            if callable(has_source) and has_source(source_id):
                return source_id
        raise ValueError("Choose a valid data source")

    def relation_sql_for_source(self, source_id: Any = None) -> str:
        normalised = self.normalise_source(source_id)
        if normalised == "dataset":
            return self.relation_sql()
        for provider in self._source_providers:
            has_source = getattr(provider, "has_source", None)
            relation_sql = getattr(provider, "relation_sql", None)
            if callable(has_source) and callable(relation_sql) and has_source(normalised):
                return str(relation_sql(normalised))
        raise ValueError("Choose a valid data source")

    def register_data_source_provider(self, provider: Any) -> None:
        if provider not in self._source_providers:
            self._source_providers.append(provider)

    def model_prediction_source(self, source_id: Any) -> ModelPredictionSource | None:
        source = str(source_id or "").strip()
        if not source or source == "dataset":
            return None
        for provider in self._source_providers:
            prediction_source = getattr(provider, "prediction_source", None)
            if not callable(prediction_source):
                continue
            raw = prediction_source(source)
            if raw is None:
                continue
            if isinstance(raw, ModelPredictionSource):
                return raw
            if isinstance(raw, dict):
                return ModelPredictionSource(
                    source_id=str(raw.get("source_id") or source),
                    column=str(raw.get("column") or ""),
                    relation_sql=str(raw.get("relation_sql") or ""),
                    active=bool(raw.get("active")),
                )
        return None

    def data_sources(self) -> list[dict[str, Any]]:
        with self._lock:
            schema = self.schema()
            invalid_columns = schema.get("invalid_columns", [])
            sources = [
                {
                    "id": "dataset",
                    "label": self.path.name,
                    "kind": "dataset",
                    "source_kind": schema["source_kind"],
                    "row_count": schema["row_count"],
                    "columns": schema["columns"],
                    "invalid_columns": invalid_columns,
                }
            ]
            for provider in self._source_providers:
                data_sources = getattr(provider, "data_sources", None)
                if callable(data_sources):
                    sources.extend(data_sources(self))
            return sources

    def reload(self) -> None:
        with self._lock:
            self._configure_source()
            self._schema = None
            self._invalid_column_errors = None
            self._row_count = None
            self._band_suggestions = None
            self._source_band_suggestions = {}

    def schema(self) -> dict[str, Any]:
        with self._lock:
            columns = self.valid_schema_columns()
            invalid_columns = self.invalid_columns()
            return {
                "path": str(self.path),
                "source_kind": self.source_kind,
                "file_size": self.file_size(),
                "file_count": self.file_count(),
                "row_count": self.row_count(),
                "columns": [self.column_payload(c) for c in columns],
                "invalid_columns": invalid_columns,
                "warnings": self.invalid_column_warnings(invalid_columns),
            }

    def file_size(self) -> int:
        if self.source_kind == "parquet_folder":
            return sum(path.stat().st_size for path in self._parquet_files)
        return self.path.stat().st_size

    def file_count(self) -> int:
        if self.source_kind == "parquet_folder":
            return len(self._parquet_files)
        return 1

    def _ensure_schema(self) -> list[ColumnInfo]:
        if self._schema is None:
            rows = self.con.execute(f"DESCRIBE SELECT * FROM {self.relation_sql()}").fetchall()
            self._schema = [
                ColumnInfo(name=str(row[0]), duckdb_type=str(row[1]), kind=infer_kind(str(row[1])))
                for row in rows
            ]
        return self._schema

    def column_payload(self, column: ColumnInfo) -> dict[str, Any]:
        return {
            "name": column.name,
            "duckdb_type": column.duckdb_type,
            "kind": column.kind,
            "band_suggestion": column.band_suggestion,
        }

    def invalid_columns(self) -> list[dict[str, str]]:
        errors = self.invalid_column_errors()
        return [
            {"name": column.name, "error": errors[column.name]}
            for column in self._ensure_schema()
            if column.name in errors
        ]

    def invalid_column_errors(self) -> dict[str, str]:
        if self._invalid_column_errors is None:
            self._invalid_column_errors = self.detect_invalid_columns(self._ensure_schema())
        return dict(self._invalid_column_errors)

    def record_invalid_column(self, name: str, error: Any) -> str:
        message = error if isinstance(error, str) else duckdb_error_message(error)
        if self._invalid_column_errors is None:
            self._invalid_column_errors = self.detect_invalid_columns(self._ensure_schema())
        self._invalid_column_errors[str(name)] = str(message)
        return str(message)

    def detect_invalid_columns(self, columns: list[ColumnInfo]) -> dict[str, str]:
        errors: dict[str, str] = {}
        for column in columns:
            try:
                self.probe_column_readable(column)
            except duckdb.Error as exc:
                errors[column.name] = duckdb_error_message(exc)
        return errors

    def probe_column_readable(self, column: ColumnInfo) -> None:
        column_sql = quote_ident(column.name)
        if column.kind == "categorical":
            sql = f"SELECT COUNT(DISTINCT {column_sql}) FROM {self.relation_sql()}"
        elif column.kind in {"integer", "numeric"}:
            sql = f"SELECT COUNT(TRY_CAST({column_sql} AS DOUBLE)) FROM {self.relation_sql()}"
        elif column.kind in {"date", "datetime"}:
            sql = f"SELECT COUNT({column_sql}), MIN({column_sql}), MAX({column_sql}) FROM {self.relation_sql()}"
        else:
            sql = f"SELECT COUNT({column_sql}) FROM {self.relation_sql()}"
        self.con.execute(sql).fetchone()

    def valid_schema_columns(self) -> list[ColumnInfo]:
        errors = self.invalid_column_errors()
        return [column for column in self._ensure_schema() if column.name not in errors]

    def invalid_column_warnings(self, invalid_columns: list[dict[str, str]] | None = None) -> list[str]:
        columns = invalid_columns if invalid_columns is not None else self.invalid_columns()
        if not columns:
            return []
        names = [column["name"] for column in columns]
        visible_names = ", ".join(names[:3])
        if len(names) > 3:
            visible_names = f"{visible_names}, and {len(names) - 3} more"
        plural = "s" if len(names) != 1 else ""
        return [f"Skipped {len(names)} unreadable column{plural}: {visible_names}."]

    def row_count(self) -> int:
        if self._row_count is None:
            self._row_count = int(self.con.execute(f"SELECT COUNT(*) FROM {self.relation_sql()}").fetchone()[0])
        return self._row_count

    def row_count_for_source(self, source_id: Any = None) -> int:
        source = self.normalise_source(source_id)
        if source == "dataset":
            return self.row_count()
        return int(self.con.execute(f"SELECT COUNT(*) FROM {self.relation_sql_for_source(source)}").fetchone()[0])

    def schema_for_source(self, source_id: Any = None) -> dict[str, Any]:
        source = self.normalise_source(source_id)
        if source == "dataset":
            return self.schema()
        columns = self.schema_columns_for_source(source)
        return {
            "path": source,
            "file_size": None,
            "row_count": self.row_count_for_source(source),
            "columns": [
                {
                    "name": c.name,
                    "duckdb_type": c.duckdb_type,
                    "kind": c.kind,
                    "band_suggestion": c.band_suggestion,
                }
                for c in columns
            ],
        }

    def schema_columns_for_source(self, source_id: Any = None) -> list[ColumnInfo]:
        source = self.normalise_source(source_id)
        if source == "dataset":
            return self.valid_schema_columns()
        relation = self.relation_sql_for_source(source)
        rows = self.con.execute(f"DESCRIBE SELECT * FROM {relation}").fetchall()
        return [
            ColumnInfo(name=str(row[0]), duckdb_type=str(row[1]), kind=infer_kind(str(row[1])))
            for row in rows
        ]

    def band_suggestion_for_column(
        self,
        source_id: Any,
        feature: Any,
        filter_sql: str = "",
        sample_limit: int = 100_000,
    ) -> float | int | None:
        source = self.normalise_source(source_id)
        feature_name = str(feature or "").strip()
        if not feature_name:
            raise ValueError("Choose a numeric feature")
        columns = {column.name: column for column in self.schema_columns_for_source(source)}
        column = columns.get(feature_name)
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
  FROM {self.relation_sql_for_source(source)}
  {where_sql}
  LIMIT {limit}
)
SELECT
  STDDEV_SAMP(value) AS std,
  MIN(value) AS min_value,
  MAX(value) AS max_value
FROM sample
"""
        row = self.con.execute(sql).fetchone()
        if not row:
            return None
        stddev, min_value, max_value = row
        if column.kind == "integer" and min_value is not None and max_value is not None:
            if max_value - min_value < 120:
                return 1
        return suggested_band_width(stddev)

    def date_bucket_suggestion_for_column(
        self,
        source_id: Any,
        feature: Any,
        filter_sql: str = "",
    ) -> dict[str, Any]:
        source = self.normalise_source(source_id)
        feature_name = str(feature or "").strip()
        if not feature_name:
            raise ValueError("Choose a date feature")
        columns = {column.name: column for column in self.schema_columns_for_source(source)}
        column = columns.get(feature_name)
        if column is None:
            raise ValueError("Choose a valid feature for the selected data source")
        if column.kind not in {"date", "datetime"}:
            raise ValueError("Choose a date or datetime feature for date bucketing")
        where_sql = f"WHERE ({filter_sql})" if filter_sql else ""
        raw = quote_ident(column.name)
        sql = f"""
WITH bounds AS (
  SELECT
    MIN({raw}) AS min_value,
    MAX({raw}) AS max_value
  FROM {self.relation_sql_for_source(source)}
  {where_sql}
)
SELECT
  min_value,
  max_value,
  CASE
    WHEN min_value IS NULL OR max_value IS NULL THEN 'none'
    WHEN max_value < min_value + INTERVAL 1 MONTH THEN 'hour'
    WHEN max_value < min_value + INTERVAL 3 MONTH THEN 'day'
    WHEN max_value < min_value + INTERVAL 12 MONTH THEN 'week'
    WHEN max_value < min_value + INTERVAL 3 YEAR THEN 'month'
    ELSE 'year'
  END AS date_bucket
FROM bounds
"""
        row = self.con.execute(sql).fetchone()
        if not row:
            return {"date_bucket": "none", "min_value": None, "max_value": None}
        min_value, max_value, date_bucket = row
        return {
            "date_bucket": str(date_bucket or "none"),
            "min_value": min_value,
            "max_value": max_value,
        }

    def band_suggestions(self, schema: list[ColumnInfo]) -> dict[str, float | int | None]:
        if self._band_suggestions is not None:
            return self._band_suggestions
        self._band_suggestions = self.band_suggestions_for_relation(schema, self.relation_sql())
        return self._band_suggestions

    def band_suggestions_for_source(
        self,
        source: str,
        schema: list[ColumnInfo],
        relation_sql: str,
    ) -> dict[str, float | int | None]:
        if source == "dataset":
            return self.band_suggestions(schema)
        if source not in self._source_band_suggestions:
            self._source_band_suggestions[source] = self.band_suggestions_for_relation(schema, relation_sql)
        return self._source_band_suggestions[source]

    def band_suggestions_for_relation(
        self,
        schema: list[ColumnInfo],
        relation_sql: str,
    ) -> dict[str, float | int | None]:
        numeric = [col for col in schema if is_numeric_kind(col.kind)]
        if not numeric:
            return {}
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
  SELECT * FROM {relation_sql} LIMIT 10000
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
            range_sql = f"SELECT {', '.join(range_parts)} FROM {relation_sql}"
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
        return suggestions

    def column_map(self) -> dict[str, ColumnInfo]:
        return {c.name: c for c in self.valid_schema_columns()}

    def all_column_map(self) -> dict[str, ColumnInfo]:
        return {c.name: c for c in self._schema_columns()}

    def column_map_for_source(self, source_id: Any = None) -> dict[str, ColumnInfo]:
        source = self.normalise_source(source_id)
        return {
            column.name: column
            for column in self.schema_columns_for_source(source)
        }

    def _schema_columns(self) -> list[ColumnInfo]:
        return self._ensure_schema()

    def normalise_filter(self, raw: Any, source_id: Any = None) -> str:
        return self.normalise_filter_for_relation(raw, self.relation_sql_for_source(source_id))

    def normalise_filter_for_relation(self, raw: Any, relation_sql: str) -> str:
        expression = str(raw or "").strip()
        if not expression:
            return ""
        forbidden = (";", "--", "/*", "*/")
        if any(token in expression for token in forbidden):
            raise ValueError("Filter must be a single DuckDB expression without statement separators or comments")
        try:
            self.con.execute(f"SELECT 1 FROM {relation_sql} WHERE ({expression}) LIMIT 0")
        except duckdb.Error as exc:
            message = str(exc).splitlines()[0]
            raise ValueError(f"Invalid filter: {message}") from exc
        return expression

    def filtered_row_count(self, filter_sql: str, source_id: Any = None) -> int:
        if not filter_sql:
            return self.row_count_for_source(source_id)
        value = self.con.execute(f"SELECT COUNT(*) FROM {self.relation_sql_for_source(source_id)} WHERE ({filter_sql})").fetchone()[0]
        return int(value)
