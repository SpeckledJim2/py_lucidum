from __future__ import annotations

import json
import math
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable
from uuid import uuid4

from py_lucidum.core import json_number, sql_literal


MAX_SMOOTHING_LEVEL = 5
STATIC_DIR = Path(__file__).with_name("static")
DEFAULT_SECTOR_ADJACENCY_PATH = STATIC_DIR / "geodata" / "sector_adjacency.json"
DEFAULT_SECTOR_SMOOTHING_POOLS_PATH = STATIC_DIR / "geodata" / "sector_smoothing_pools.parquet"
SECTOR_SMOOTHING_LOAD_WARNING = "Sector smoothing adjacency could not be loaded; raw sector values are shown."


@dataclass(frozen=True)
class SectorAdjacency:
    keys: tuple[str, ...]
    neighbours: tuple[tuple[int, ...], ...]
    key_to_index: dict[str, int]
    method: str


def normalise_smoothing_level(raw: Any) -> int:
    if raw is None:
        return 0
    if isinstance(raw, str) and raw.strip().lower() in {"", "none", "off"}:
        return 0
    try:
        number = float(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Choose a smoothing level from None to {MAX_SMOOTHING_LEVEL}") from exc
    if not math.isfinite(number) or int(number) != number:
        raise ValueError(f"Choose a smoothing level from None to {MAX_SMOOTHING_LEVEL}")
    level = int(number)
    if level < 0 or level > MAX_SMOOTHING_LEVEL:
        raise ValueError(f"Choose a smoothing level from None to {MAX_SMOOTHING_LEVEL}")
    return level


@lru_cache(maxsize=4)
def load_sector_adjacency(path: str = str(DEFAULT_SECTOR_ADJACENCY_PATH)) -> SectorAdjacency:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    keys = tuple(str(key) for key in payload.get("keys") or ())
    raw_neighbours = payload.get("neighbours") or ()
    if not keys or len(raw_neighbours) != len(keys):
        raise ValueError("Sector adjacency sidecar does not match the sector key list.")

    neighbours: list[tuple[int, ...]] = []
    for index, raw_indexes in enumerate(raw_neighbours):
        indexes = tuple(int(value) for value in raw_indexes)
        if any(value < 0 or value >= len(keys) for value in indexes):
            raise ValueError("Sector adjacency sidecar includes an out-of-range neighbour index.")
        if index in indexes:
            raise ValueError("Sector adjacency sidecar includes a self-neighbour.")
        if indexes != tuple(sorted(indexes)) or len(indexes) != len(set(indexes)):
            raise ValueError("Sector adjacency sidecar neighbours must be sorted and unique.")
        neighbours.append(indexes)

    method = str(payload.get("neighbour_type") or "shared_edge")
    return SectorAdjacency(
        keys=keys,
        neighbours=tuple(neighbours),
        key_to_index={key: index for index, key in enumerate(keys)},
        method=method,
    )


@lru_cache(maxsize=32)
def sector_smoothing_pools(path: str, depth: int) -> tuple[tuple[int, ...], ...]:
    adjacency = load_sector_adjacency(path)
    if depth <= 0:
        return tuple((index,) for index in range(len(adjacency.keys)))

    pools: list[tuple[int, ...]] = []
    for start_index in range(len(adjacency.keys)):
        seen = {start_index}
        frontier = [start_index]
        for _ in range(depth):
            next_frontier: list[int] = []
            for index in frontier:
                for neighbour_index in adjacency.neighbours[index]:
                    if neighbour_index not in seen:
                        seen.add(neighbour_index)
                        next_frontier.append(neighbour_index)
            frontier = next_frontier
            if not frontier:
                break
        pools.append(tuple(sorted(seen)))
    return tuple(pools)


def smooth_sector_rows(
    rows: list[dict[str, Any]],
    smoothing_level: int,
    *,
    adjacency_path: str = str(DEFAULT_SECTOR_ADJACENCY_PATH),
) -> tuple[list[dict[str, Any]], dict[str, Any], str | None]:
    metadata: dict[str, Any] = {
        "level": smoothing_level,
        "max_level": MAX_SMOOTHING_LEVEL,
        "applied": False,
        "method": "none" if smoothing_level <= 0 else "shared_edge_weighted_numerator",
        "matched_rows": len(rows),
        "target_rows": len(rows),
        "smoothed_rows": 0,
        "fallback_rows": 0,
        "contributing_rows": 0,
    }
    if smoothing_level <= 0:
        return rows, metadata, None

    try:
        adjacency = load_sector_adjacency(adjacency_path)
        pools = sector_smoothing_pools(adjacency_path, smoothing_level)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        metadata["warning"] = "Sector smoothing adjacency could not be loaded; raw sector values are shown."
        metadata["fallback_rows"] = len(rows)
        return rows, metadata, f"{metadata['warning']} {exc}"

    rows_by_key = {str(row.get("key")): row for row in rows if row.get("key") is not None}
    smoothed_rows: list[dict[str, Any]] = []
    smoothed_count = 0
    fallback_count = 0
    contributing_total = 0

    metadata["target_rows"] = len(adjacency.keys)
    for row_index, key in enumerate(adjacency.keys):
        source_row = rows_by_key.get(key)
        next_row = dict(source_row) if source_row else empty_sector_row(key)
        add_raw_fields(next_row, source_row or {})
        pooled_numerator = 0.0
        pooled_denominator = 0.0
        contributing_count = 0
        for neighbour_index in pools[row_index]:
            neighbour_row = rows_by_key.get(adjacency.keys[neighbour_index])
            if not neighbour_row or not row_has_plottable_value(neighbour_row):
                continue
            pooled_numerator += float(neighbour_row["numerator"])
            pooled_denominator += float(neighbour_row["denominator"])
            contributing_count += 1

        if pooled_denominator <= 0 or contributing_count == 0:
            next_row["smoothing_contributing_sectors"] = 0
            fallback_count += 1
            smoothed_rows.append(next_row)
            continue

        next_row["numerator"] = json_number(pooled_numerator)
        next_row["denominator"] = json_number(pooled_denominator)
        next_row["volume"] = json_number(pooled_denominator)
        next_row["value"] = json_number(pooled_numerator / pooled_denominator)
        next_row["smoothing_contributing_sectors"] = contributing_count
        smoothed_count += 1
        contributing_total += contributing_count
        smoothed_rows.append(next_row)

    for row in rows:
        key = str(row.get("key"))
        if key in adjacency.key_to_index:
            continue
        next_row = dict(row)
        add_raw_fields(next_row, row)
        next_row["smoothing_contributing_sectors"] = 0
        fallback_count += 1
        smoothed_rows.append(next_row)

    metadata.update(
        {
            "applied": True,
            "smoothed_rows": smoothed_count,
            "fallback_rows": fallback_count,
            "contributing_rows": contributing_total,
        }
    )
    return smoothed_rows, metadata, None


def build_sector_smoothing_relation_sql(
    raw_summary_sql: str,
    smoothing_levels: Iterable[int],
    *,
    pool_path: str = str(DEFAULT_SECTOR_SMOOTHING_POOLS_PATH),
) -> str:
    levels = tuple(dict.fromkeys(normalise_smoothing_level(level) for level in smoothing_levels))
    if not levels or any(level <= 0 for level in levels):
        raise ValueError("Smoothing SQL requires at least one positive smoothing level.")

    raw_sql = raw_summary_sql.strip().rstrip(";")
    pool_literal = sql_literal(str(pool_path))
    requested_level_rows = ",\n    ".join(f"({level})" for level in levels)
    requested_level_list = ", ".join(str(level) for level in levels)
    return f"""
WITH raw_summary AS (
{raw_sql}
),
requested_levels(smoothing_level) AS (
  VALUES
    {requested_level_rows}
),
pool_rows AS (
  SELECT
    CAST(level AS INTEGER) AS smoothing_level,
    CAST(target_key AS VARCHAR) AS target_key,
    CAST(pool_key AS VARCHAR) AS pool_key,
    ROW_NUMBER() OVER (PARTITION BY CAST(level AS INTEGER)) AS __pair_order
  FROM read_parquet({pool_literal})
  WHERE level IN ({requested_level_list})
),
targets AS (
  SELECT
    smoothing_level,
    target_key AS key,
    MIN(__pair_order) AS __target_order
  FROM pool_rows
  GROUP BY smoothing_level, target_key
),
pooled AS (
  SELECT
    pool_rows.smoothing_level,
    pool_rows.target_key AS key,
    SUM(raw_summary.resp0_num) AS numerator,
    SUM(raw_summary.resp0_den) AS denominator,
    COUNT(*) AS contributing_sectors
  FROM pool_rows
  INNER JOIN raw_summary ON raw_summary.key = pool_rows.pool_key
  WHERE raw_summary.resp0 IS NOT NULL
    AND raw_summary.resp0_num IS NOT NULL
    AND raw_summary.resp0_den IS NOT NULL
    AND raw_summary.resp0_den > 0
    AND isfinite(raw_summary.resp0)
    AND isfinite(raw_summary.resp0_num)
    AND isfinite(raw_summary.resp0_den)
  GROUP BY pool_rows.smoothing_level, pool_rows.target_key
),
target_rows AS (
  SELECT
    targets.smoothing_level,
    targets.key,
    COALESCE(raw_summary.row_count, 0) AS row_count,
    CASE
      WHEN COALESCE(pooled.contributing_sectors, 0) > 0 AND COALESCE(pooled.denominator, 0) > 0
      THEN pooled.numerator
      ELSE raw_summary.resp0_num
    END AS numerator,
    CASE
      WHEN COALESCE(pooled.contributing_sectors, 0) > 0 AND COALESCE(pooled.denominator, 0) > 0
      THEN pooled.denominator
      ELSE raw_summary.resp0_den
    END AS denominator,
    CASE
      WHEN COALESCE(pooled.contributing_sectors, 0) > 0 AND COALESCE(pooled.denominator, 0) > 0
      THEN pooled.denominator
      ELSE raw_summary.resp0_den
    END AS volume,
    CASE
      WHEN COALESCE(pooled.contributing_sectors, 0) > 0 AND COALESCE(pooled.denominator, 0) > 0
      THEN pooled.numerator / NULLIF(pooled.denominator, 0)
      ELSE raw_summary.resp0
    END AS value,
    raw_summary.resp0_num AS raw_numerator,
    raw_summary.resp0_den AS raw_denominator,
    raw_summary.resp0_den AS raw_volume,
    raw_summary.resp0 AS raw_value,
    raw_summary.row_count AS raw_row_count,
    CASE
      WHEN COALESCE(pooled.contributing_sectors, 0) > 0 AND COALESCE(pooled.denominator, 0) > 0
      THEN pooled.contributing_sectors
      ELSE 0
    END AS smoothing_contributing_sectors,
    COALESCE(pooled.contributing_sectors, 0) > 0 AND COALESCE(pooled.denominator, 0) > 0 AS __smoothed,
    targets.__target_order,
    NULL::VARCHAR AS __unknown_order
  FROM targets
  LEFT JOIN raw_summary ON raw_summary.key = targets.key
  LEFT JOIN pooled
    ON pooled.smoothing_level = targets.smoothing_level
   AND pooled.key = targets.key
),
unknown_rows AS (
  SELECT
    requested_levels.smoothing_level,
    raw_summary.key,
    raw_summary.row_count,
    raw_summary.resp0_num AS numerator,
    raw_summary.resp0_den AS denominator,
    raw_summary.resp0_den AS volume,
    raw_summary.resp0 AS value,
    raw_summary.resp0_num AS raw_numerator,
    raw_summary.resp0_den AS raw_denominator,
    raw_summary.resp0_den AS raw_volume,
    raw_summary.resp0 AS raw_value,
    raw_summary.row_count AS raw_row_count,
    0 AS smoothing_contributing_sectors,
    FALSE AS __smoothed,
    NULL::BIGINT AS __target_order,
    raw_summary.key AS __unknown_order
  FROM requested_levels
  CROSS JOIN raw_summary
  LEFT JOIN targets
    ON targets.smoothing_level = requested_levels.smoothing_level
   AND targets.key = raw_summary.key
  WHERE targets.key IS NULL
),
final_rows AS (
  SELECT * FROM target_rows
  UNION ALL
  SELECT * FROM unknown_rows
),
metadata AS (
  SELECT
    requested_levels.smoothing_level,
    (SELECT COUNT(*) FROM raw_summary) AS __matched_rows,
    (
      SELECT COUNT(*)
      FROM targets
      WHERE targets.smoothing_level = requested_levels.smoothing_level
    ) AS __target_rows,
    (
      SELECT COUNT(*)
      FROM final_rows
      WHERE final_rows.smoothing_level = requested_levels.smoothing_level
        AND final_rows.__smoothed
    ) AS __smoothed_rows,
    (
      SELECT COUNT(*)
      FROM final_rows
      WHERE final_rows.smoothing_level = requested_levels.smoothing_level
        AND NOT final_rows.__smoothed
    ) AS __fallback_rows,
    COALESCE((
      SELECT SUM(smoothing_contributing_sectors)
      FROM final_rows
      WHERE final_rows.smoothing_level = requested_levels.smoothing_level
        AND final_rows.__smoothed
    ), 0) AS __contributing_rows
  FROM requested_levels
)
SELECT
  final_rows.smoothing_level,
  final_rows.key,
  final_rows.row_count,
  final_rows.numerator,
  final_rows.denominator,
  final_rows.volume,
  final_rows.value,
  final_rows.raw_numerator,
  final_rows.raw_denominator,
  final_rows.raw_volume,
  final_rows.raw_value,
  final_rows.raw_row_count,
  final_rows.smoothing_contributing_sectors,
  metadata.__matched_rows,
  metadata.__target_rows,
  metadata.__smoothed_rows,
  metadata.__fallback_rows,
  metadata.__contributing_rows,
  final_rows.__target_order,
  final_rows.__unknown_order
FROM final_rows
INNER JOIN metadata USING (smoothing_level)
"""


def build_smoothed_sector_sql(
    raw_summary_sql: str,
    smoothing_level: int,
    *,
    pool_path: str = str(DEFAULT_SECTOR_SMOOTHING_POOLS_PATH),
) -> str:
    level = normalise_smoothing_level(smoothing_level)
    if level <= 0:
        raise ValueError("Smoothing SQL requires a positive smoothing level.")

    smoothing_relation = build_sector_smoothing_relation_sql(
        raw_summary_sql,
        (level,),
        pool_path=pool_path,
    )
    return f"""
WITH smoothed_sector_rows AS (
{smoothing_relation}
)
SELECT
  key,
  row_count,
  numerator,
  denominator,
  volume,
  value,
  raw_numerator,
  raw_denominator,
  raw_volume,
  raw_value,
  raw_row_count,
  smoothing_contributing_sectors,
  __matched_rows,
  __target_rows,
  __smoothed_rows,
  __fallback_rows,
  __contributing_rows
FROM smoothed_sector_rows
WHERE smoothing_level = {level}
ORDER BY
  CASE WHEN __target_order IS NULL THEN 1 ELSE 0 END,
  __target_order,
  __unknown_order
"""


def build_sector_smoothing_output_sql(
    raw_summary_sql: str,
    *,
    pool_path: str = str(DEFAULT_SECTOR_SMOOTHING_POOLS_PATH),
) -> str:
    smoothing_relation = build_sector_smoothing_relation_sql(
        raw_summary_sql,
        range(1, MAX_SMOOTHING_LEVEL + 1),
        pool_path=pool_path,
    )
    smooth_columns = ",\n  ".join(
        f"MAX(CASE WHEN smoothing_level = {level} THEN value END) AS smooth_n{level}"
        for level in range(1, MAX_SMOOTHING_LEVEL + 1)
    )
    return f"""
WITH smoothed_sector_rows AS (
{smoothing_relation}
)
SELECT
  key AS postcode_sector,
  MAX(raw_numerator) AS numerator_sum,
  MAX(raw_denominator) AS denominator_sum,
  MAX(raw_value) AS unsmoothed,
  {smooth_columns}
FROM smoothed_sector_rows
GROUP BY key
ORDER BY key
"""


def write_sector_smoothing_parquet(
    connection: Any,
    raw_summary_sql: str,
    output_path: str | Path,
    *,
    pool_path: str = str(DEFAULT_SECTOR_SMOOTHING_POOLS_PATH),
) -> tuple[Path, int]:
    output = Path(output_path).expanduser().resolve()
    if output.suffix.lower() != ".parquet":
        raise ValueError("Sector smoothing output must use a .parquet filename.")
    if output.exists() and not output.is_file():
        raise ValueError(f"Sector smoothing output is not a file: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.stem}.tmp-{uuid4().hex}.parquet")
    output_sql = build_sector_smoothing_output_sql(raw_summary_sql, pool_path=pool_path)
    try:
        copied = connection.execute(
            f"COPY ({output_sql}) TO {sql_literal(str(temporary))} "
            "(FORMAT PARQUET, COMPRESSION ZSTD)"
        ).fetchone()
        row_count = int((copied or (0,))[0] or 0)
        temporary.replace(output)
    finally:
        temporary.unlink(missing_ok=True)
    return output, row_count


def empty_sector_row(key: str) -> dict[str, Any]:
    return {
        "key": key,
        "row_count": 0,
        "numerator": None,
        "denominator": None,
        "volume": None,
        "value": None,
    }


def add_raw_fields(target: dict[str, Any], source: dict[str, Any]) -> None:
    target["raw_numerator"] = source.get("numerator")
    target["raw_denominator"] = source.get("denominator")
    target["raw_volume"] = source.get("volume")
    target["raw_value"] = source.get("value")
    target["raw_row_count"] = source.get("row_count")


def row_has_plottable_value(row: dict[str, Any]) -> bool:
    return (
        json_number(row.get("value")) is not None
        and json_number(row.get("numerator")) is not None
        and (json_number(row.get("denominator")) or 0) > 0
    )


__all__ = [
    "DEFAULT_SECTOR_ADJACENCY_PATH",
    "DEFAULT_SECTOR_SMOOTHING_POOLS_PATH",
    "MAX_SMOOTHING_LEVEL",
    "SECTOR_SMOOTHING_LOAD_WARNING",
    "SectorAdjacency",
    "build_sector_smoothing_output_sql",
    "build_sector_smoothing_relation_sql",
    "build_smoothed_sector_sql",
    "load_sector_adjacency",
    "normalise_smoothing_level",
    "sector_smoothing_pools",
    "smooth_sector_rows",
    "write_sector_smoothing_parquet",
]
