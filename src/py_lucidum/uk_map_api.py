"""Public helpers for UK postcode-sector smoothing."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from py_lucidum.core import Dataset, is_numeric_kind, quote_ident, sql_literal
from py_lucidum.tools.uk_map.query import build_summary_sql
from py_lucidum.tools.uk_map.smoothing import write_sector_smoothing_parquet


POSTCODE_SECTOR_PATTERN = r"^(GIR|[A-Z]{1,2}[0-9][A-Z0-9]?) [0-9]$"


def smooth_postcode_sectors(
    dataset_path: str | Path,
    output_path: str | Path,
    *,
    postcode_sector: str,
    numerator: str,
    denominator: str | None = None,
    filter: str = "",
) -> Path:
    """Write raw and N1-N5 postcode-sector values to a Parquet file.

    Omitting ``denominator`` uses one per valid numerator row, matching
    Lucidum's Average row value calculation.
    """

    source = Path(dataset_path).expanduser().resolve()
    output = Path(output_path).expanduser().resolve()
    if not source.is_file() or source.suffix.lower() != ".parquet":
        raise ValueError(f"Source dataset must be one Parquet file: {source}")
    if output == source:
        raise ValueError("Sector smoothing output must not overwrite the source Parquet file.")

    dataset = Dataset(source)
    try:
        columns = dataset.column_map()
        sector_column = _required_column(columns, postcode_sector, "postcode-sector")
        numerator_column = _required_numeric_column(columns, numerator, "numerator")
        denominator_column = None
        if denominator is not None and str(denominator).strip():
            denominator_column = _required_numeric_column(columns, denominator, "denominator")

        filter_sql = dataset.normalise_filter(filter)
        _validate_postcode_sectors(dataset, sector_column, filter_sql)
        denominator_spec = {
            "column": denominator_column,
            "label": denominator_column or "Average row value",
            "bar_label": denominator_column or "Row count",
        }
        raw_summary_sql = build_summary_sql(
            dataset.relation_sql(),
            sector_column,
            {"label": numerator_column, "numerator": numerator_column},
            denominator_spec,
            filter_sql,
            order_by=False,
        )
        written_path, _row_count = write_sector_smoothing_parquet(
            dataset.con,
            raw_summary_sql,
            output,
        )
        return written_path
    finally:
        dataset.con.close()


def _required_column(columns: dict[str, Any], raw: Any, label: str) -> str:
    name = str(raw or "").strip()
    if not name or name not in columns:
        raise ValueError(f"Choose a valid {label} column.")
    return name


def _required_numeric_column(columns: dict[str, Any], raw: Any, label: str) -> str:
    name = _required_column(columns, raw, label)
    if not is_numeric_kind(columns[name].kind):
        raise ValueError(f"Choose a numeric {label} column.")
    return name


def _validate_postcode_sectors(dataset: Dataset, column: str, filter_sql: str) -> None:
    sector_sql = f"TRIM(CAST({quote_ident(column)} AS VARCHAR))"
    filter_condition = f"({filter_sql}) AND " if filter_sql else ""
    invalid_rows = dataset.con.execute(f"""
SELECT DISTINCT {sector_sql} AS postcode_sector
FROM {dataset.relation_sql()}
WHERE {filter_condition}{sector_sql} <> ''
  AND {sector_sql} IS NOT NULL
  AND NOT regexp_full_match({sector_sql}, {sql_literal(POSTCODE_SECTOR_PATTERN)})
ORDER BY postcode_sector
LIMIT 5
""").fetchall()
    if not invalid_rows:
        return
    examples = ", ".join(repr(str(row[0])) for row in invalid_rows)
    raise ValueError(
        "Postcode-sector values must be uppercase with one space before the final digit, "
        f"for example 'AB10 1'. Invalid value examples: {examples}"
    )


__all__ = ["smooth_postcode_sectors"]
