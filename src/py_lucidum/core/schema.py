from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ColumnInfo:
    name: str
    duckdb_type: str
    kind: str
    band_suggestion: float | int | None = None


def infer_kind(duckdb_type: str) -> str:
    t = duckdb_type.upper()
    if "INT" in t:
        return "integer"
    if any(part in t for part in ("DOUBLE", "FLOAT", "REAL", "DECIMAL")):
        return "numeric"
    if "TIMESTAMP" in t:
        return "datetime"
    if "DATE" in t or "TIME" in t:
        return "date"
    if "BOOL" in t:
        return "categorical"
    return "categorical"


def is_numeric_kind(kind: str) -> bool:
    return kind in {"integer", "numeric"}


def json_number(value: Any) -> float | int | None:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    if number.is_integer() and abs(number) < 9_007_199_254_740_992:
        return int(number)
    return number


def parse_positive_float(value: Any) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if parsed <= 0 or not math.isfinite(parsed):
        return None
    return parsed


def nice_band_steps() -> list[float]:
    steps: list[float] = []
    for exponent in range(-8, 13):
        multiplier = 10 ** exponent
        steps.extend([1 * multiplier, 2 * multiplier, 5 * multiplier])
    return sorted(steps)


def suggested_band_width(stddev: Any) -> float | int | None:
    value = parse_positive_float(stddev)
    if not value:
        return None
    steps = nice_band_steps()
    lower_index = 0
    for index, step in enumerate(steps):
        if step <= value:
            lower_index = index
        else:
            break
    suggestion_index = max(0, lower_index - 1)
    return json_number(steps[suggestion_index])
