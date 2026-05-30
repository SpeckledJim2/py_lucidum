from __future__ import annotations

import random
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation, ROUND_FLOOR
from typing import Any

from py_lucidum.core import Dataset

from .validation import ValidationResult, coerce_parameter, normalise_parameters, validate_request


DEFAULT_GRID_SAMPLES = 25
DEFAULT_GRID_SEED = 2026


class ParameterGridError(ValueError):
    pass


@dataclass(frozen=True)
class GridDimension:
    name: str
    row: dict[str, Any]
    kind: str
    count: int
    values: tuple[Any, ...] = ()
    start: Decimal | None = None
    step: Decimal | None = None

    def value_at(self, index: int) -> Any:
        if index < 0 or index >= self.count:
            raise IndexError(index)
        if self.kind == "set":
            return self.values[index]
        if self.start is None or self.step is None:
            raise ParameterGridError(f"Invalid grid dimension for {self.name}")
        return decimal_parameter_value(self.start + self.step * index)


@dataclass(frozen=True)
class ParameterGrid:
    rows: list[dict[str, Any]]
    dimensions: list[GridDimension]

    @property
    def enabled(self) -> bool:
        return bool(self.dimensions)

    @property
    def total_combinations(self) -> int:
        total = 1
        for dimension in self.dimensions:
            total *= dimension.count
        return total

    def resolved_rows(self, combination_index: int) -> list[dict[str, Any]]:
        if not self.enabled:
            return [dict(row) for row in self.rows]
        total = self.total_combinations
        if combination_index < 0 or combination_index >= total:
            raise IndexError(combination_index)
        value_indexes: dict[str, int] = {}
        remaining = combination_index
        for dimension in reversed(self.dimensions):
            value_indexes[dimension.name] = remaining % dimension.count
            remaining //= dimension.count
        resolved: list[dict[str, Any]] = []
        for row in self.rows:
            name = str(row.get("name") or "").strip()
            dimension = next((item for item in self.dimensions if item.name == name), None)
            if not dimension:
                resolved.append(dict(row))
                continue
            updated = dict(row)
            updated["value"] = dimension.value_at(value_indexes[name])
            resolved.append(updated)
        return resolved


@dataclass(frozen=True)
class GridCombination:
    ordinal: int
    combination_index: int
    parameters: list[dict[str, Any]]
    resolved_parameters: dict[str, Any]
    warnings: list[str]


@dataclass(frozen=True)
class GridRun:
    grid: ParameterGrid
    requested_samples: int
    sample_indexes: list[int]
    combinations: list[GridCombination]
    skipped: list[dict[str, Any]]
    warnings: list[str]
    messages: list[str]
    errors: list[str]

    @property
    def ok(self) -> bool:
        return not self.errors

    @property
    def enabled(self) -> bool:
        return self.grid.enabled

    def summary(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "total_combinations": self.grid.total_combinations if self.enabled else 1,
            "requested_samples": self.requested_samples,
            "sampled_count": len(self.sample_indexes),
            "trainable_count": len(self.combinations),
            "skipped_count": len(self.skipped),
            "messages": self.messages,
            "warnings": self.warnings,
            "errors": self.errors,
        }


def parse_parameter_grid(raw_parameters: Any) -> ParameterGrid:
    rows = parameter_rows(raw_parameters)
    dimensions: list[GridDimension] = []
    for row in rows:
        value = row.get("value")
        text = str(value).strip() if isinstance(value, str) else ""
        if not text:
            continue
        has_brace = "{" in text or "}" in text
        if not has_brace:
            continue
        name = str(row.get("name") or "").strip()
        if not name:
            raise ParameterGridError("Grid parameter rows must have a parameter name")
        dimensions.append(parse_grid_dimension(name, row, text))
    return ParameterGrid(rows=rows, dimensions=dimensions)


def parameter_rows(raw_parameters: Any) -> list[dict[str, Any]]:
    if isinstance(raw_parameters, list):
        rows = []
        for item in raw_parameters:
            if isinstance(item, dict) and str(item.get("name") or "").strip():
                rows.append(dict(item))
        return rows
    if isinstance(raw_parameters, dict):
        return [{"name": str(name), "value": value} for name, value in raw_parameters.items() if str(name).strip()]
    return []


def parse_grid_dimension(name: str, row: dict[str, Any], text: str) -> GridDimension:
    if not (text.startswith("{") and text.endswith("}")) or text.count("{") != 1 or text.count("}") != 1:
        raise ParameterGridError(f"{name} uses invalid grid syntax; use {{a, b, c}} or {{start, end; step}}")
    body = text[1:-1].strip()
    if not body:
        raise ParameterGridError(f"{name} grid cannot be empty")
    if "{" in body or "}" in body:
        raise ParameterGridError(f"{name} grid cannot contain nested braces")
    if ";" in body:
        return parse_range_dimension(name, row, body)
    parts = [part.strip() for part in body.split(",")]
    if any(part == "" for part in parts):
        raise ParameterGridError(f"{name} grid set contains an empty value")
    values = tuple(coerce_parameter(part) for part in parts)
    if not values:
        raise ParameterGridError(f"{name} grid cannot be empty")
    return GridDimension(name=name, row=dict(row), kind="set", count=len(values), values=values)


def parse_range_dimension(name: str, row: dict[str, Any], body: str) -> GridDimension:
    if body.count(";") != 1:
        raise ParameterGridError(f"{name} range grid must use one semicolon")
    bounds_text, step_text = [part.strip() for part in body.split(";")]
    bounds = [part.strip() for part in bounds_text.split(",")]
    if len(bounds) != 2 or not bounds[0] or not bounds[1] or not step_text:
        raise ParameterGridError(f"{name} range grid must use {{start, end; step}}")
    try:
        start = Decimal(bounds[0])
        end = Decimal(bounds[1])
        step = Decimal(step_text)
    except InvalidOperation as exc:
        raise ParameterGridError(f"{name} range grid values must be numeric") from exc
    if step == 0:
        raise ParameterGridError(f"{name} range grid step cannot be zero")
    if start < end and step < 0:
        raise ParameterGridError(f"{name} range grid step must move toward the end value")
    if start > end and step > 0:
        raise ParameterGridError(f"{name} range grid step must move toward the end value")
    if start == end:
        count = 1
    else:
        span = (end - start) / step
        if span < 0:
            raise ParameterGridError(f"{name} range grid step must move toward the end value")
        count = int(span.to_integral_value(rounding=ROUND_FLOOR)) + 1
    return GridDimension(name=name, row=dict(row), kind="range", count=count, start=start, step=step)


def decimal_parameter_value(value: Decimal) -> Any:
    text = format(value.normalize(), "f")
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    return coerce_parameter(text)


def normalise_grid_samples(raw: Any) -> int:
    try:
        value = int(raw)
    except (TypeError, ValueError):
        return DEFAULT_GRID_SAMPLES
    return max(1, value)


def scalar_grid_seed(raw_parameters: Any) -> int:
    for row in parameter_rows(raw_parameters):
        if str(row.get("name") or "").strip() != "seed":
            continue
        value = row.get("value")
        if isinstance(value, str) and ("{" in value or "}" in value):
            return DEFAULT_GRID_SEED
        try:
            seed = int(value)
        except (TypeError, ValueError):
            return DEFAULT_GRID_SEED
        return seed if seed >= 0 else DEFAULT_GRID_SEED
    return DEFAULT_GRID_SEED


def sampled_combination_indexes(total_combinations: int, requested_samples: int, seed: int) -> list[int]:
    total = max(0, int(total_combinations))
    requested = normalise_grid_samples(requested_samples)
    if total <= 0:
        return []
    if requested >= total:
        return list(range(total))
    rng = random.Random(seed)
    indexes: set[int] = set()
    while len(indexes) < requested:
        indexes.add(rng.randrange(total))
    return sorted(indexes)


def prepare_grid_run(dataset: Dataset, payload: dict[str, Any], generated_sample_path: Any = None) -> GridRun:
    requested_samples = normalise_grid_samples(payload.get("grid_samples", DEFAULT_GRID_SAMPLES))
    try:
        grid = parse_parameter_grid(payload.get("parameters"))
    except ParameterGridError as exc:
        empty_grid = ParameterGrid(rows=parameter_rows(payload.get("parameters")), dimensions=[])
        return GridRun(
            grid=empty_grid,
            requested_samples=requested_samples,
            sample_indexes=[],
            combinations=[],
            skipped=[],
            warnings=[],
            messages=[],
            errors=[str(exc)],
        )
    if not grid.enabled:
        return GridRun(
            grid=grid,
            requested_samples=requested_samples,
            sample_indexes=[],
            combinations=[],
            skipped=[],
            warnings=[],
            messages=[],
            errors=[],
        )

    total = grid.total_combinations
    indexes = sampled_combination_indexes(total, requested_samples, scalar_grid_seed(payload.get("parameters")))
    combinations: list[GridCombination] = []
    skipped: list[dict[str, Any]] = []
    warning_values: list[str] = []
    for ordinal, combination_index in enumerate(indexes, start=1):
        parameters = grid.resolved_rows(combination_index)
        combination_payload = {**payload, "parameters": parameters}
        validation = validate_request(dataset, combination_payload, generated_sample_path=generated_sample_path)
        if validation.ok:
            combinations.append(
                GridCombination(
                    ordinal=ordinal,
                    combination_index=combination_index,
                    parameters=parameters,
                    resolved_parameters=normalise_parameters(parameters),
                    warnings=validation.warnings,
                )
            )
            warning_values.extend(validation.warnings)
        else:
            skipped.append(
                {
                    "ordinal": ordinal,
                    "combination_index": combination_index,
                    "parameters": normalise_parameters(parameters),
                    "errors": validation.errors,
                }
            )
    messages = grid_messages(total, requested_samples, len(indexes), len(skipped), len(combinations))
    errors: list[str] = []
    if not combinations:
        first_error = "; ".join(skipped[0]["errors"]) if skipped else "no combinations were sampled"
        errors.append(f"No valid grid combinations remain. First error: {first_error}")
    return GridRun(
        grid=grid,
        requested_samples=requested_samples,
        sample_indexes=indexes,
        combinations=combinations,
        skipped=skipped,
        warnings=dedupe(warning_values),
        messages=messages,
        errors=errors,
    )


def grid_messages(total_combinations: int, requested_samples: int, sampled_count: int, skipped_count: int, trainable_count: int) -> list[str]:
    messages: list[str] = []
    if requested_samples >= total_combinations:
        messages.append(f"Grid has {total_combinations:,} combinations; running all {total_combinations:,}.")
    if skipped_count:
        messages.append(f"Grid sampled {sampled_count:,} combinations; skipped {skipped_count:,} invalid; training {trainable_count:,}.")
    return messages


def validate_grid_or_request(dataset: Dataset, payload: dict[str, Any], generated_sample_path: Any = None) -> dict[str, Any]:
    grid_run = prepare_grid_run(dataset, payload, generated_sample_path=generated_sample_path)
    if grid_run.enabled or grid_run.errors:
        return {
            "ok": grid_run.ok,
            "errors": grid_run.errors,
            "warnings": grid_run.warnings,
            "grid": grid_run.summary(),
        }
    validation: ValidationResult = validate_request(dataset, payload, generated_sample_path=generated_sample_path)
    return {**validation.as_payload(), "grid": grid_run.summary()}


def dedupe(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        text = str(value or "").strip()
        if text and text not in seen:
            result.append(text)
            seen.add(text)
    return result


__all__ = [
    "DEFAULT_GRID_SAMPLES",
    "GridCombination",
    "GridRun",
    "ParameterGrid",
    "ParameterGridError",
    "parse_parameter_grid",
    "prepare_grid_run",
    "sampled_combination_indexes",
    "validate_grid_or_request",
]
