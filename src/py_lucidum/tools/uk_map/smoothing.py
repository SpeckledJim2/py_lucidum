from __future__ import annotations

import json
import math
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

from py_lucidum.core import json_number


MAX_SMOOTHING_LEVEL = 5
STATIC_DIR = Path(__file__).with_name("static")
DEFAULT_SECTOR_ADJACENCY_PATH = STATIC_DIR / "geodata" / "sector_adjacency.json"


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
    "MAX_SMOOTHING_LEVEL",
    "SectorAdjacency",
    "load_sector_adjacency",
    "normalise_smoothing_level",
    "sector_smoothing_pools",
    "smooth_sector_rows",
]
