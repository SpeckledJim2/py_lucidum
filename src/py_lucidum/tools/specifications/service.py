from __future__ import annotations

import csv
import os
from collections.abc import Iterable
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any
from uuid import uuid4

from py_lucidum.core import (
    Dataset,
    is_numeric_kind,
    load_features,
    load_kpis,
    load_saved_filters,
    resolve_features_path,
    resolve_filters_path,
    resolve_kpis_path,
)
from py_lucidum.core.features import FEATURE_SPEC_METADATA_COLUMNS, FEATURE_SPEC_REQUIRED_COLUMNS
from py_lucidum.core.kpis import KPI_SPEC_COLUMNS, normalise_kpi_denominator


FILTER_SPEC_COLUMNS = ["theme", "name", "expression"]
FEATURE_SPEC_DEFAULT_COLUMNS = [
    "Feature",
    "Grouping",
    "Base",
    "min",
    "max",
    "banding",
    "scenario1",
]
SPEC_KINDS = ("feature", "kpi", "filter")


def normalise_kind(raw: object) -> str:
    kind = str(raw or "").strip().lower()
    aliases = {
        "features": "feature",
        "feature_spec": "feature",
        "feature-spec": "feature",
        "kpis": "kpi",
        "kpi_spec": "kpi",
        "kpi-spec": "kpi",
        "filters": "filter",
        "filter_spec": "filter",
        "filter-spec": "filter",
    }
    kind = aliases.get(kind, kind)
    if kind not in SPEC_KINDS:
        accepted = ", ".join(SPEC_KINDS)
        raise ValueError(f"Unknown specification kind {kind!r}. Use one of: {accepted}")
    return kind


def default_columns(kind: str) -> list[str]:
    if kind == "feature":
        return list(FEATURE_SPEC_DEFAULT_COLUMNS)
    if kind == "kpi":
        return list(KPI_SPEC_COLUMNS)
    if kind == "filter":
        return list(FILTER_SPEC_COLUMNS)
    raise ValueError(f"Unknown specification kind {kind!r}")


def spec_label(kind: str) -> str:
    return {
        "feature": "Feature spec",
        "kpi": "KPI spec",
        "filter": "Filter spec",
    }[kind]


def spec_state_enabled(state: Any, kind: str) -> bool:
    if kind == "feature":
        return bool(getattr(state, "use_features", True))
    if kind == "kpi":
        return bool(getattr(state, "use_kpis", True))
    if kind == "filter":
        return bool(getattr(state, "use_saved_filters", True))
    raise ValueError(f"Unknown specification kind {kind!r}")


def spec_path(state: Any, kind: str) -> Path:
    if kind == "feature":
        resolved = getattr(state, "resolved_features_path", None)
        if resolved:
            return Path(resolved)
        return resolve_features_path(getattr(state, "features_path", None), use_features=True)  # type: ignore[return-value]
    if kind == "kpi":
        resolved = getattr(state, "resolved_kpis_path", None)
        if resolved:
            return Path(resolved)
        return resolve_kpis_path(getattr(state, "kpis_path", None), use_kpis=True)  # type: ignore[return-value]
    if kind == "filter":
        resolved = getattr(state, "resolved_filters_path", None)
        if resolved:
            return Path(resolved)
        return resolve_filters_path(getattr(state, "filters_path", None), use_saved_filters=True)  # type: ignore[return-value]
    raise ValueError(f"Unknown specification kind {kind!r}")


def read_spec_file(state: Any, kind: str) -> dict[str, Any]:
    kind = normalise_kind(kind)
    path = spec_path(state, kind)
    columns = default_columns(kind)
    rows: list[dict[str, str]] = []
    exists = path.exists()
    if exists:
        with path.open(newline="", encoding="utf-8-sig") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames:
                columns = list(reader.fieldnames)
            rows = [
                {column: str(row.get(column) or "") for column in columns}
                for row in reader
            ]
    return {
        "kind": kind,
        "label": spec_label(kind),
        "path": str(path),
        "file_name": path.name,
        "exists": exists,
        "enabled": spec_state_enabled(state, kind),
        "columns": columns,
        "rows": rows,
        "row_count": len(rows),
    }


def submitted_spec(payload: dict[str, Any], kind: str) -> tuple[list[str], list[dict[str, str]]]:
    raw_columns = payload.get("columns")
    if not isinstance(raw_columns, list):
        raise ValueError("Specification payload must include a columns array")
    columns: list[str] = []
    for column in raw_columns:
        name = str(column or "").strip()
        if not name or name.startswith("_"):
            continue
        if name in columns:
            raise ValueError(f"Specification has a duplicate column: {name}")
        columns.append(name)
    if not columns:
        raise ValueError("Specification must include at least one column")

    raw_rows = payload.get("rows")
    if raw_rows is None:
        raw_rows = []
    if not isinstance(raw_rows, list):
        raise ValueError("Specification payload rows must be an array")
    rows: list[dict[str, str]] = []
    for raw_row in raw_rows:
        if not isinstance(raw_row, dict):
            raise ValueError("Specification rows must be objects")
        rows.append({column: str(raw_row.get(column) or "") for column in columns})
    return columns, rows


def nonblank_rows(rows: Iterable[dict[str, str]], columns: list[str]) -> list[tuple[int, dict[str, str]]]:
    return [
        (index, row)
        for index, row in enumerate(rows, start=2)
        if any(str(row.get(column) or "").strip() for column in columns)
    ]


def validate_spec(dataset: Dataset, kind: str, columns: list[str], rows: list[dict[str, str]]) -> dict[str, Any]:
    kind = normalise_kind(kind)
    errors: list[str] = []
    warnings: list[str] = []
    if kind == "feature":
        validate_feature_spec(dataset, columns, rows, errors, warnings)
    elif kind == "kpi":
        validate_kpi_spec(dataset, columns, rows, errors, warnings)
    elif kind == "filter":
        validate_filter_spec(dataset, columns, rows, errors, warnings)
    if not errors:
        try:
            parse_submitted_spec(kind, columns, rows)
        except ValueError as exc:
            errors.append(str(exc))
    return {
        "kind": kind,
        "valid": not errors,
        "errors": errors,
        "warnings": warnings,
        "message": validation_message(kind, errors, warnings),
    }


def validation_message(kind: str, errors: list[str], warnings: list[str]) -> str:
    label = spec_label(kind)
    if errors:
        return f"{label} has {len(errors)} error{'s' if len(errors) != 1 else ''}"
    if warnings:
        return f"{label} is valid with {len(warnings)} warning{'s' if len(warnings) != 1 else ''}"
    return f"{label} is valid"


def validate_filter_spec(dataset: Dataset, columns: list[str], rows: list[dict[str, str]], errors: list[str], warnings: list[str]) -> None:
    if columns != FILTER_SPEC_COLUMNS:
        errors.append("filter_spec.csv must have exactly these columns: theme,name,expression")
        return
    for row_number, row in nonblank_rows(rows, columns):
        theme = str(row.get("theme") or "").strip()
        name = str(row.get("name") or "").strip()
        expression = str(row.get("expression") or "").strip()
        missing = [label for label, value in (("theme", theme), ("name", name), ("expression", expression)) if not value]
        if missing:
            errors.append(f"filter_spec.csv row {row_number} is missing: {', '.join(missing)}")
            continue
        try:
            dataset.normalise_filter(expression)
        except ValueError as exc:
            errors.append(f"filter_spec.csv row {row_number}: {exc}")


def validate_kpi_spec(dataset: Dataset, columns: list[str], rows: list[dict[str, str]], errors: list[str], warnings: list[str]) -> None:
    if columns != KPI_SPEC_COLUMNS:
        errors.append("kpi_spec.csv must have exactly these columns: group,name,actual,denominator,decimals,format")
        return
    column_map = dataset.column_map()
    for row_number, row in nonblank_rows(rows, columns):
        required = {
            "group": str(row.get("group") or "").strip(),
            "name": str(row.get("name") or "").strip(),
            "actual": str(row.get("actual") or "").strip(),
            "decimals": str(row.get("decimals") or "").strip(),
            "format": str(row.get("format") or "").strip(),
        }
        missing = [name for name, value in required.items() if not value]
        if missing:
            errors.append(f"kpi_spec.csv row {row_number} is missing: {', '.join(missing)}")
            continue
        actual = required["actual"]
        actual_column = column_map.get(actual)
        if actual_column is None:
            errors.append(f"kpi_spec.csv row {row_number} actual column does not exist: {actual}")
        elif not is_numeric_kind(actual_column.kind):
            errors.append(f"kpi_spec.csv row {row_number} actual column must be numeric: {actual}")
        denominator = normalise_kpi_denominator(row.get("denominator"))
        if denominator != "__none__":
            denominator_column = column_map.get(denominator)
            if denominator_column is None:
                errors.append(f"kpi_spec.csv row {row_number} denominator column does not exist: {denominator}")
            elif not is_numeric_kind(denominator_column.kind):
                errors.append(f"kpi_spec.csv row {row_number} denominator column must be numeric: {denominator}")


def validate_feature_spec(dataset: Dataset, columns: list[str], rows: list[dict[str, str]], errors: list[str], warnings: list[str]) -> None:
    if columns[:2] != FEATURE_SPEC_REQUIRED_COLUMNS:
        errors.append("feature_spec.csv must start with these columns: Feature,Grouping")
        return
    metadata_stop = feature_metadata_stop(columns)
    for column in columns[2:metadata_stop]:
        if column not in FEATURE_SPEC_METADATA_COLUMNS:
            errors.append(f"feature_spec.csv has an unsupported metadata column: {column}")
            return
    for row_number, row in nonblank_rows(rows, columns):
        feature = str(row.get("Feature") or "").strip()
        if not feature:
            errors.append(f"feature_spec.csv row {row_number} is missing: Feature")
            continue
        for scenario in columns[metadata_stop:]:
            value = str(row.get(scenario) or "").strip()
            if value and "feature" not in value.lower():
                warnings.append(f"feature_spec.csv row {row_number} scenario {scenario!r} value will not select the feature: {value!r}")


def feature_metadata_stop(columns: list[str]) -> int:
    index = 2
    seen: set[str] = set()
    while index < len(columns):
        column = columns[index]
        if column not in FEATURE_SPEC_METADATA_COLUMNS or column in seen:
            break
        seen.add(column)
        index += 1
    return index


def parse_submitted_spec(kind: str, columns: list[str], rows: list[dict[str, str]]) -> None:
    with TemporaryDirectory() as tmp_dir:
        path = Path(tmp_dir) / {
            "feature": "feature_spec.csv",
            "kpi": "kpi_spec.csv",
            "filter": "filter_spec.csv",
        }[kind]
        write_csv(path, columns, rows)
        if kind == "feature":
            load_features(path)
        elif kind == "kpi":
            load_kpis(path)
        elif kind == "filter":
            load_saved_filters(path)


def save_spec_file(state: Any, dataset: Dataset, kind: str, columns: list[str], rows: list[dict[str, str]]) -> dict[str, Any]:
    kind = normalise_kind(kind)
    result = validate_spec(dataset, kind, columns, rows)
    if not result["valid"]:
        return result
    path = spec_path(state, kind)
    write_csv_atomic(path, columns, rows)
    refresh_loaded_spec_state(state, kind)
    result["saved"] = True
    result["path"] = str(path)
    result["spec"] = read_spec_file(state, kind)
    result["message"] = f"{spec_label(kind)} saved"
    return result


def write_csv_atomic(path: Path, columns: list[str], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        write_csv(temp_path, columns, rows)
        os.replace(temp_path, path)
    finally:
        try:
            temp_path.unlink()
        except FileNotFoundError:
            pass


def write_csv(path: Path, columns: list[str], rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, lineterminator="\n", extrasaction="ignore")
        writer.writeheader()
        writer.writerows({column: str(row.get(column) or "") for column in columns} for row in rows)


def refresh_loaded_spec_state(state: Any, kind: str) -> None:
    if kind == "feature":
        state.resolved_features_path = resolve_features_path(state.features_path, use_features=state.use_features)
        state.feature_spec = load_features(state.features_path, use_features=state.use_features)
    elif kind == "kpi":
        state.resolved_kpis_path = resolve_kpis_path(state.kpis_path, use_kpis=state.use_kpis)
        state.kpis = load_kpis(state.kpis_path, use_kpis=state.use_kpis)
    elif kind == "filter":
        state.resolved_filters_path = resolve_filters_path(state.filters_path, use_saved_filters=state.use_saved_filters)
        state.saved_filters = load_saved_filters(state.filters_path, use_saved_filters=state.use_saved_filters)
