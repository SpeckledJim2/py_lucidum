from __future__ import annotations

from pathlib import Path
from typing import Any

import duckdb

from py_lucidum.core import Dataset, quote_ident, sql_literal


SAMPLE_COLUMN = "SAMPLE"
SAMPLE_LEVELS = ("training", "test", "validation")
SAMPLE_SEED = 2026
GENERATED_SAMPLE_FILENAME = "generated_sample.parquet"


def dataset_sample_column(dataset: Dataset) -> str | None:
    return SAMPLE_COLUMN if SAMPLE_COLUMN in dataset.column_map() else None


def generated_sample_relation_sql(path: Path) -> str:
    return f"read_parquet({sql_literal(str(path))})"


def generated_sample_is_current(dataset: Dataset, path: Path) -> bool:
    if not path.exists():
        return False
    try:
        columns = {
            str(row[0])
            for row in dataset.con.execute(f"DESCRIBE SELECT * FROM {generated_sample_relation_sql(path)}").fetchall()
        }
        if {"__lucidum_row_id", SAMPLE_COLUMN} - columns:
            return False
        row_count = int(dataset.con.execute(f"SELECT COUNT(*) FROM {generated_sample_relation_sql(path)}").fetchone()[0])
    except duckdb.Error:
        return False
    return row_count == dataset.row_count()


def sample_metadata(dataset: Dataset, generated_sample_path: Path) -> dict[str, Any]:
    with dataset.lock:
        has_dataset_sample = dataset_sample_column(dataset) is not None
        has_generated_sample = generated_sample_is_current(dataset, generated_sample_path)
        if has_dataset_sample:
            counts = sample_counts_for_relation(dataset, dataset.relation_sql(), quote_ident(SAMPLE_COLUMN))
            return {
                "column": SAMPLE_COLUMN,
                "source": "dataset",
                "row_count": counts["row_count"],
                "levels": counts["levels"],
                "has_dataset_sample": True,
                "has_generated_sample": has_generated_sample,
                "warning": "",
            }
        if has_generated_sample:
            counts = sample_counts_for_relation(dataset, generated_sample_relation_sql(generated_sample_path), quote_ident(SAMPLE_COLUMN))
            return {
                "column": SAMPLE_COLUMN,
                "source": "generated",
                "row_count": counts["row_count"],
                "levels": counts["levels"],
                "has_dataset_sample": False,
                "has_generated_sample": True,
                "warning": "A generated SAMPLE split is being reused for GBM training. For durable modelling, add a proper SAMPLE column to the original Parquet file.",
            }
        warning = "No SAMPLE column was found. GBM training will use all valid rows without early stopping unless you create a generated SAMPLE split."
        if generated_sample_path.exists():
            warning = "The generated SAMPLE split no longer matches the dataset. Recreate it before using generated sampling."
        return {
            "column": None,
            "source": "none",
            "row_count": dataset.row_count(),
            "levels": [
                {"name": level, "row_count": 0, "percent": 0.0}
                for level in SAMPLE_LEVELS
            ],
            "has_dataset_sample": False,
            "has_generated_sample": False,
            "warning": warning,
        }


def sample_counts_for_relation(dataset: Dataset, relation_sql: str, column_sql: str) -> dict[str, Any]:
    rows = dataset.con.execute(
        f"""
SELECT LOWER(TRIM(CAST({column_sql} AS VARCHAR))) AS sample, COUNT(*) AS row_count
FROM {relation_sql}
GROUP BY sample
"""
    ).fetchall()
    counts = {str(sample or ""): int(row_count) for sample, row_count in rows}
    row_count = sum(counts.values())
    return {
        "row_count": row_count,
        "levels": [
            {
                "name": level,
                "row_count": counts.get(level, 0),
                "percent": round(100.0 * counts.get(level, 0) / row_count, 1) if row_count else 0.0,
            }
            for level in SAMPLE_LEVELS
        ],
    }


def create_generated_sample(dataset: Dataset, path: Path) -> dict[str, Any]:
    with dataset.lock:
        if dataset_sample_column(dataset):
            return sample_metadata(dataset, path)
        if generated_sample_is_current(dataset, path):
            return sample_metadata(dataset, path)
        path.parent.mkdir(parents=True, exist_ok=True)
        temp = path.with_suffix(path.suffix + ".tmp")
        dataset.con.execute(
            f"""
COPY (
  WITH base AS (
    SELECT ROW_NUMBER() OVER () AS __lucidum_row_id
    FROM {dataset.relation_sql()}
  ), ranked AS (
    SELECT
      __lucidum_row_id,
      ROW_NUMBER() OVER (ORDER BY hash(__lucidum_row_id + {SAMPLE_SEED}), __lucidum_row_id) AS sample_rank,
      COUNT(*) OVER () AS row_count
    FROM base
  )
  SELECT
    __lucidum_row_id,
    CASE
      WHEN sample_rank <= CAST(FLOOR(row_count * 0.6) AS BIGINT) THEN 'training'
      WHEN sample_rank <= CAST(FLOOR(row_count * 0.8) AS BIGINT) THEN 'test'
      ELSE 'validation'
    END AS {quote_ident(SAMPLE_COLUMN)}
  FROM ranked
  ORDER BY __lucidum_row_id
) TO {sql_literal(str(temp))} (FORMAT PARQUET)
"""
        )
        temp.replace(path)
        return sample_metadata(dataset, path)


def generated_training_sample_counts(dataset: Dataset, path: Path, offset_column: str | None) -> dict[str, int]:
    where_sql = f"WHERE TRY_CAST({quote_ident(offset_column)} AS DOUBLE) > 0" if offset_column else ""
    offset_projection = f",\n    {quote_ident(offset_column)}" if offset_column else ""
    rows = dataset.con.execute(
        f"""
WITH base AS (
  SELECT
    ROW_NUMBER() OVER () AS __lucidum_row_id{offset_projection}
  FROM {dataset.relation_sql()}
)
SELECT LOWER(TRIM(CAST(sample.{quote_ident(SAMPLE_COLUMN)} AS VARCHAR))) AS sample, COUNT(*) AS row_count
FROM base
INNER JOIN {generated_sample_relation_sql(path)} sample USING (__lucidum_row_id)
{where_sql}
GROUP BY sample
"""
    ).fetchall()
    return {str(sample or ""): int(row_count) for sample, row_count in rows}


def dataset_training_sample_counts(dataset: Dataset, offset_column: str | None) -> dict[str, int]:
    denominator_filter = f"WHERE TRY_CAST({quote_ident(offset_column)} AS DOUBLE) > 0" if offset_column else ""
    rows = dataset.con.execute(
        f"""
SELECT LOWER(TRIM(CAST({quote_ident(SAMPLE_COLUMN)} AS VARCHAR))) AS sample, COUNT(*) AS row_count
FROM {dataset.relation_sql()}
{denominator_filter}
GROUP BY sample
"""
    ).fetchall()
    return {str(sample or ""): int(row_count) for sample, row_count in rows}


__all__ = [
    "GENERATED_SAMPLE_FILENAME",
    "SAMPLE_COLUMN",
    "SAMPLE_LEVELS",
    "SAMPLE_SEED",
    "create_generated_sample",
    "dataset_sample_column",
    "dataset_training_sample_counts",
    "generated_sample_is_current",
    "generated_sample_relation_sql",
    "generated_training_sample_counts",
    "sample_metadata",
]
