from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass
import itertools
import math
import pickle
import re
import time
from pathlib import Path
from typing import Any

import duckdb

from py_lucidum.core import Dataset, is_numeric_kind, quote_ident, sql_literal, suggested_band_width

from .store import GlmModelStore, json_safe_number
from .training import formula_context, glm_dependencies, offset_values_for_frame, write_dataframe_parquet
from .validation import TARGET_COLUMN


ProgressCallback = Callable[[dict[str, Any]], None]
MAX_TABULATION_CELLS = 100_000
MODEL_CROSSTAB = "__model__"


@dataclass(frozen=True)
class _TabulationModelRef:
    kind: str
    model_id: str
    ref: str
    field: str
    legacy_field: bool = False

    @property
    def label_kind(self) -> str:
        return self.kind.upper()


def _safe_id(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_") or "table"


def _as_number(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _json_value(value: Any) -> Any:
    if value is None:
        return None
    try:
        if hasattr(value, "item"):
            value = value.item()
    except Exception:
        pass
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return value


def _ordered_tabulation_values(values: Any) -> list[Any]:
    cleaned = [_json_value(value) for value in values]
    non_missing = [value for value in cleaned if value is not None]
    numeric_values = [_as_number(value) for value in non_missing]
    if non_missing and all(value is not None for value in numeric_values):
        return sorted(cleaned, key=lambda value: (value is None, _as_number(value) if value is not None else 0.0, str(value)))
    return sorted(cleaned, key=lambda value: (value is None, str(value)))


def _feature_spec_map(feature_spec: Any) -> dict[str, dict[str, Any]]:
    if not isinstance(feature_spec, dict):
        return {}
    rows = feature_spec.get("rows", [])
    if not isinstance(rows, list):
        return {}
    result: dict[str, dict[str, Any]] = {}
    for row in rows:
        if isinstance(row, dict) and row.get("feature"):
            result[str(row["feature"])] = row
    return result


def _schema_columns_and_kinds(dataset: Dataset) -> tuple[list[str], dict[str, str]]:
    columns = dataset.valid_schema_columns()
    return [column.name for column in columns], {column.name: column.kind for column in columns}


def _column_tokens(expression: str, columns: list[str]) -> list[str]:
    found: list[str] = []
    for column in sorted(columns, key=len, reverse=True):
        pattern = rf"(?<![A-Za-z0-9_]){re.escape(column)}(?![A-Za-z0-9_])"
        if re.search(pattern, expression):
            found.append(column)
    return sorted(set(found))


def _term_groups(estimator: Any, offset_terms: list[str], source_columns: list[str]) -> dict[tuple[str, ...], dict[str, Any]]:
    groups: dict[tuple[str, ...], dict[str, Any]] = {}
    spec = getattr(estimator, "X_model_spec_", None)
    source_set = set(source_columns)
    if spec is not None:
        term_variables = getattr(spec, "term_variables", {}) or {}
        term_indices = getattr(spec, "term_indices", {}) or {}
        for term in getattr(spec, "terms", []) or []:
            indices = list(term_indices.get(term, []) or [])
            if not indices:
                continue
            variables = tuple(sorted(str(name) for name in (term_variables.get(term, set()) or set()) if str(name) in source_set))
            if not variables:
                continue
            entry = groups.setdefault(variables, {"variables": list(variables), "term_indices": [], "offset_terms": []})
            entry["term_indices"].extend(indices)
    for expression in offset_terms:
        variables = tuple(_column_tokens(expression, source_columns))
        entry = groups.setdefault(variables, {"variables": list(variables), "term_indices": [], "offset_terms": []})
        entry["offset_terms"].append(expression)
    return groups


def _fit_frame_for_levels(frame: Any, manifest: dict[str, Any], pd: Any) -> Any:
    sample_column = str(manifest.get("sample_column") or "")
    training_scope = str(manifest.get("training_scope") or "all")
    if training_scope == "training" and sample_column and sample_column in frame.columns:
        return frame.loc[frame[sample_column].astype(str).str.strip().str.lower() == "training"].copy()
    return frame


def _required_tabulation_columns(
    source_columns: list[str],
    manifest: dict[str, Any],
    all_features: list[str],
    offset_terms: list[str],
) -> list[str]:
    requested = set(all_features)
    denominator_column = str(manifest.get("denominator_column") or "").strip()
    if denominator_column:
        requested.add(denominator_column)
    sample_column = str(manifest.get("sample_column") or "").strip()
    if str(manifest.get("training_scope") or "all") == "training" and sample_column:
        requested.add(sample_column)
    for expression in offset_terms:
        requested.update(_column_tokens(expression, source_columns))
    return [column for column in source_columns if column in requested]


def _tabulation_frame_from_dataset(dataset: Dataset, columns: list[str]) -> Any:
    projection = ["ROW_NUMBER() OVER () AS __lucidum_row_id", *[quote_ident(name) for name in columns]]
    with dataset.lock:
        return dataset.con.execute(f"SELECT {', '.join(projection)} FROM {dataset.relation_sql()}").fetchdf()


def _feature_transform_bounds(estimator: Any, source_columns: list[str]) -> dict[str, dict[str, float]]:
    spec = getattr(estimator, "X_model_spec_", None)
    transform_state = getattr(spec, "transform_state", {}) if spec is not None else {}
    if not isinstance(transform_state, dict):
        return {}
    bounds: dict[str, dict[str, float]] = {}
    for expression, state in transform_state.items():
        if not isinstance(state, dict):
            continue
        lower = _as_number(state.get("lower_bound"))
        upper = _as_number(state.get("upper_bound"))
        if lower is None and upper is None:
            continue
        for feature in _column_tokens(str(expression), source_columns):
            entry = bounds.setdefault(feature, {})
            if lower is not None:
                entry["lower_bound"] = max(float(lower), float(entry.get("lower_bound", lower)))
            if upper is not None:
                entry["upper_bound"] = min(float(upper), float(entry.get("upper_bound", upper)))
    return bounds


def _mode_value(series: Any) -> Any:
    values = series.dropna()
    if not len(values):
        return ""
    counts = values.map(_json_value).value_counts(dropna=True)
    if not len(counts):
        return ""
    return _json_value(counts.index[0])


def _clip_numeric_bound(
    feature: str,
    field: str,
    value: float,
    bounds: dict[str, float],
    source: str,
) -> tuple[float, dict[str, Any] | None]:
    clipped = float(value)
    bound_name = ""
    lower = _as_number(bounds.get("lower_bound"))
    upper = _as_number(bounds.get("upper_bound"))
    if lower is not None and clipped < lower:
        clipped = float(lower)
        bound_name = "lower_bound"
    if upper is not None and clipped > upper:
        clipped = float(upper)
        bound_name = "upper_bound"
    if bound_name:
        return clipped, {
            "feature": feature,
            "field": field,
            "source": source,
            "value": _round_grid_value(value),
            "clipped": _round_grid_value(clipped),
            "bound": bound_name,
        }
    return clipped, None


def _base_value(
    frame: Any,
    fit_frame: Any,
    feature: str,
    kind: str,
    spec_row: dict[str, Any],
    bounds: dict[str, float],
    pd: Any,
) -> tuple[Any, dict[str, Any] | None, dict[str, Any] | None]:
    raw = str(spec_row.get("base") or "").strip()
    if is_numeric_kind(kind):
        values = pd.to_numeric(frame[feature], errors="coerce").dropna() if feature in frame.columns else []
        raw_number = _as_number(raw)
        if raw and raw_number is not None:
            value, clipped = _clip_numeric_bound(feature, "base", float(raw_number), bounds, "feature_spec")
            return value, None, clipped
        if len(values):
            inferred = float(values.median())
        else:
            inferred = 0.0
        value, clipped = _clip_numeric_bound(feature, "base", inferred, bounds, "inferred")
        return value, {"feature": feature, "field": "base", "value": _json_value(value)}, clipped
    series = fit_frame[feature].dropna() if feature in fit_frame.columns else []
    seen_values = {_json_value(value) for value in series.unique()} if len(series) else set()
    if raw and raw in seen_values:
        return raw, None, None
    inferred = _mode_value(series)
    return inferred, {"feature": feature, "field": "base", "value": _json_value(inferred)}, None


def _base_row(frame: Any) -> dict[str, Any]:
    row: dict[str, Any] = {}
    first = frame.iloc[0].to_dict() if len(frame) else {}
    for column in frame.columns:
        row[str(column)] = _json_value(first.get(column))
    row[TARGET_COLUMN] = 0.0
    return row


def _estimated_numeric_spec(
    frame: Any,
    feature: str,
    kind: str,
    spec_row: dict[str, Any],
    bounds: dict[str, float],
    pd: Any,
) -> tuple[float, float, float, list[str], list[dict[str, Any]]]:
    warnings: list[str] = []
    clipped_bounds: list[dict[str, Any]] = []
    values = pd.to_numeric(frame[feature], errors="coerce").dropna()
    data_min = float(values.min()) if len(values) else 0.0
    data_max = float(values.max()) if len(values) else data_min + 1.0
    stddev = float(values.std()) if len(values) > 1 else abs(data_max - data_min)
    raw_min = _as_number(spec_row.get("min"))
    raw_max = _as_number(spec_row.get("max"))
    raw_band = _as_number(spec_row.get("banding"))
    if raw_band and raw_band > 0:
        band = float(raw_band)
    elif kind == "integer" and data_max - data_min < 120:
        band = 1.0
    else:
        band = _as_number(suggested_band_width(stddev)) or 1.0
    minimum = raw_min if raw_min is not None else data_min
    maximum = raw_max if raw_max is not None else data_max
    minimum, clipped = _clip_numeric_bound(feature, "min", float(minimum), bounds, "feature_spec" if raw_min is not None else "inferred")
    if clipped:
        clipped_bounds.append(clipped)
    maximum, clipped = _clip_numeric_bound(feature, "max", float(maximum), bounds, "feature_spec" if raw_max is not None else "inferred")
    if clipped:
        clipped_bounds.append(clipped)
    if maximum < minimum:
        minimum, maximum = maximum, minimum
    if raw_min is None:
        warnings.append("min")
    if raw_max is None:
        warnings.append("max")
    if raw_band is None or raw_band <= 0:
        warnings.append("banding")
    return float(minimum), float(maximum), float(band), warnings, clipped_bounds


def _round_grid_value(value: float) -> float | int:
    rounded = round(float(value), 10)
    return int(rounded) if float(rounded).is_integer() else rounded


def _numeric_levels(minimum: float, maximum: float, band: float, base_value: Any) -> list[Any]:
    if band <= 0:
        band = 1.0
    count = int(math.floor((maximum - minimum) / band)) + 1
    count = max(1, min(count, MAX_TABULATION_CELLS))
    levels = [_round_grid_value(minimum + index * band) for index in range(count)]
    if levels and float(levels[-1]) < maximum - 1e-9:
        levels.append(_round_grid_value(maximum))
    base_number = _as_number(base_value)
    if base_number is not None:
        base = _round_grid_value(base_number)
        if base not in levels:
            levels.append(base)
            levels = sorted(levels, key=lambda value: float(value))
    return levels


def _categorical_levels(frame: Any, fit_frame: Any, feature: str, base_value: Any, pd: Any) -> tuple[list[Any], set[Any]]:
    all_values = sorted([_json_value(value) for value in frame[feature].dropna().unique()], key=lambda value: str(value))
    seen = {_json_value(value) for value in fit_frame[feature].dropna().unique()}
    if base_value not in all_values and str(base_value or ""):
        all_values.append(base_value)
    return all_values, seen


def _feature_levels(
    frame: Any,
    fit_frame: Any,
    feature: str,
    kind: str,
    spec_row: dict[str, Any],
    base_value: Any,
    bounds: dict[str, float],
    np: Any,
    pd: Any,
) -> tuple[list[dict[str, Any]], dict[str, Any], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    estimated: list[dict[str, Any]] = []
    clipped_bounds: list[dict[str, Any]] = []
    unseen: list[dict[str, Any]] = []
    meta: dict[str, Any] = {"feature": feature, "kind": kind, "base": _json_value(base_value)}
    if is_numeric_kind(kind):
        minimum, maximum, band, estimated_fields, clipped_bounds = _estimated_numeric_spec(frame, feature, kind, spec_row, bounds, pd)
        if estimated_fields:
            estimated.append({"feature": feature, "fields": estimated_fields, "min": minimum, "max": maximum, "banding": band})
        meta.update({"min": minimum, "max": maximum, "banding": band})
        return [{"value": value, "status": "ok"} for value in _numeric_levels(minimum, maximum, band, base_value)], meta, estimated, clipped_bounds, unseen
    levels, seen = _categorical_levels(frame, fit_frame, feature, base_value, pd)
    rows: list[dict[str, Any]] = []
    for value in levels:
        status = "ok" if value in seen else "unseen"
        if status == "unseen":
            unseen.append({"feature": feature, "level": _json_value(value)})
        rows.append({"value": value, "status": status})
    return rows, meta, estimated, clipped_bounds, unseen


def _cartesian_table(levels_by_feature: dict[str, list[dict[str, Any]]], pd: Any) -> Any:
    features = list(levels_by_feature)
    rows: list[dict[str, Any]] = []
    for combination in itertools.product(*(levels_by_feature[feature] for feature in features)):
        row: dict[str, Any] = {}
        statuses: list[str] = []
        for feature, cell in zip(features, combination):
            row[feature] = cell["value"]
            statuses.append(str(cell.get("status") or "ok"))
        row["__status"] = "unseen" if any(status != "ok" for status in statuses) else "ok"
        rows.append(row)
    return pd.DataFrame(rows)


def _model_matrix(estimator: Any, frame: Any, context: dict[str, Any]) -> Any:
    matrix = estimator.X_model_spec_.get_model_matrix(frame, context=context)
    if hasattr(matrix, "toarray"):
        return matrix.toarray()
    if hasattr(matrix, "to_numpy"):
        return matrix.to_numpy()
    return matrix


def _group_contribution(
    estimator: Any,
    frame: Any,
    term_indices: list[int],
    offset_terms: list[str],
    context: dict[str, Any],
    np: Any,
    pd: Any,
) -> Any:
    contribution = np.zeros(len(frame), dtype=float)
    if term_indices:
        matrix = _model_matrix(estimator, frame, context)
        coefficients = np.asarray(getattr(estimator, "coef_", []), dtype=float)
        contribution = contribution + matrix[:, term_indices].dot(coefficients[term_indices])
    if offset_terms:
        offsets = offset_values_for_frame(frame, offset_terms, context, np, pd)
        if offsets is not None:
            contribution = contribution + offsets.to_numpy(dtype=float)
    return contribution


def _prediction_frame(base: dict[str, Any], variables: list[str], grid: Any, pd: Any) -> Any:
    rows = [dict(base) for _ in range(len(grid))]
    frame = pd.DataFrame(rows)
    for variable in variables:
        frame[variable] = grid[variable].to_numpy()
    frame[TARGET_COLUMN] = 0.0
    return frame


def _scored_feature_series(series: Any, meta: dict[str, Any], np: Any, pd: Any) -> Any:
    if not is_numeric_kind(str(meta.get("kind") or "")):
        return series.map(lambda value: None if pd.isna(value) else _json_value(value))
    values = pd.to_numeric(series, errors="coerce")
    minimum = _as_number(meta.get("min"))
    maximum = _as_number(meta.get("max"))
    band = _as_number(meta.get("banding"))
    if minimum is None or maximum is None or not band or band <= 0:
        return pd.Series(np.nan, index=series.index, dtype=float)
    clipped = values.clip(lower=minimum, upper=maximum)
    keys = np.floor((clipped - minimum) / band + 1e-9) * band + minimum
    keys = pd.Series(keys, index=series.index, dtype=float).mask(clipped >= maximum, maximum)
    keys = keys.round(10)
    return keys.where(values.notna())


def _normalise_lookup_keys(table: Any, features: list[str], feature_meta: dict[str, dict[str, Any]], np: Any, pd: Any) -> Any:
    lookup = table.loc[table["status"].astype(str) == "ok", [*features, "tabulated_linear"]].copy()
    for feature in features:
        lookup[feature] = _scored_feature_series(lookup[feature], feature_meta[feature], np, pd)
    lookup["tabulated_linear"] = pd.to_numeric(lookup["tabulated_linear"], errors="coerce")
    return lookup.dropna(subset=features)


def _component_from_table(frame: Any, table: Any, features: list[str], feature_meta: dict[str, dict[str, Any]], np: Any, pd: Any) -> Any:
    if table is None:
        return pd.Series(np.nan, index=frame.index, dtype=float)
    lookup = _normalise_lookup_keys(table, features, feature_meta, np, pd)
    if not features or lookup.empty:
        return pd.Series(np.nan, index=frame.index, dtype=float)
    if len(features) == 1:
        feature = features[0]
        key = _scored_feature_series(frame[feature], feature_meta[feature], np, pd)
        lookup_series = lookup.drop_duplicates(subset=[feature]).set_index(feature)["tabulated_linear"]
        return pd.to_numeric(key.map(lookup_series), errors="coerce")
    keys = pd.DataFrame(index=frame.index)
    for feature in features:
        keys[feature] = _scored_feature_series(frame[feature], feature_meta[feature], np, pd)
    merged = keys.merge(lookup.drop_duplicates(subset=features), on=features, how="left", sort=False)
    return pd.Series(pd.to_numeric(merged["tabulated_linear"], errors="coerce").to_numpy(), index=frame.index, dtype=float)


def _table_file_path(store: GlmModelStore, model_id: str, table_id: str) -> Path:
    return store.tabulations_dir(model_id) / f"{_safe_id(table_id)}.parquet"


def _parse_model_ref(value: Any, *, default_kind: str = "glm", legacy_field: bool = False) -> _TabulationModelRef | None:
    text = str(value or "").strip()
    if not text:
        return None
    kind = default_kind
    model_id = text
    if text.startswith("glm:") or text.startswith("gbm:"):
        parts = text.split(":")
        kind = parts[0]
        model_id = parts[1] if len(parts) > 1 else ""
    if kind not in {"glm", "gbm"} or not model_id:
        return None
    ref = f"{kind}:{model_id}"
    field = model_id if legacy_field and kind == "glm" else ref
    return _TabulationModelRef(kind=kind, model_id=model_id, ref=ref, field=field, legacy_field=legacy_field and kind == "glm")


def _dedupe_refs(refs: list[_TabulationModelRef]) -> list[_TabulationModelRef]:
    deduped: list[_TabulationModelRef] = []
    seen: set[str] = set()
    for ref in refs:
        key = ref.ref
        if key in seen:
            continue
        deduped.append(ref)
        seen.add(key)
    return deduped


def _requested_model_refs(payload: dict[str, Any], *, for_build: bool = False, store: GlmModelStore | None = None) -> list[_TabulationModelRef]:
    raw_refs = payload.get("model_refs")
    refs: list[_TabulationModelRef] = []
    if isinstance(raw_refs, list) and raw_refs:
        refs.extend(ref for value in raw_refs if (ref := _parse_model_ref(value)) is not None)
    raw_model_ids = [str(model_id).strip() for model_id in payload.get("model_ids", []) if str(model_id).strip()]
    raw_gbm_model_ids = [str(model_id).strip() for model_id in payload.get("gbm_model_ids", []) if str(model_id).strip()]
    if raw_model_ids:
        has_typed_model_ids = any(value.startswith("glm:") or value.startswith("gbm:") for value in raw_model_ids)
        legacy_plain = not refs and not has_typed_model_ids and not raw_gbm_model_ids
        refs.extend(
            ref
            for value in raw_model_ids
            if (ref := _parse_model_ref(value, default_kind="glm", legacy_field=legacy_plain)) is not None
        )
    refs.extend(
        ref
        for value in raw_gbm_model_ids
        if (ref := _parse_model_ref(value, default_kind="gbm")) is not None
    )
    refs = _dedupe_refs(refs)
    if refs or not for_build:
        return refs
    active = store.active_model_id() if store is not None else None
    return [_TabulationModelRef(kind="glm", model_id=active, ref=f"glm:{active}", field=active, legacy_field=True)] if active else []


def _model_ref_for_status(status: dict[str, Any]) -> str:
    return str(status.get("model_ref") or f"{status.get('model_kind') or 'glm'}:{status.get('model_id') or ''}")


def _model_field_label(model_ref: _TabulationModelRef, status: dict[str, Any] | None = None) -> str:
    label = str((status or {}).get("label") or model_ref.model_id)
    if model_ref.legacy_field:
        return label
    return f"{model_ref.label_kind} · {label}"


def _build_model_tabulations(
    dataset: Dataset,
    store: GlmModelStore,
    model_id: str,
    feature_spec: Any,
    progress_callback: ProgressCallback,
) -> dict[str, Any]:
    glum, _glr, _glrcv, np, pd = glm_dependencies()
    del glum, _glr, _glrcv
    estimator_path = store.artifact_path(model_id, "estimator")
    if not estimator_path.exists():
        return {
            "model_id": model_id,
            "status": "not_tabulatable",
            "warnings": ["Rebuild this GLM before tabulating; estimator.pkl is missing."],
            "tables": [],
        }
    with estimator_path.open("rb") as handle:
        estimator = pickle.load(handle)
    manifest = store.manifest(model_id)
    source_columns, kinds = _schema_columns_and_kinds(dataset)
    spec_rows = _feature_spec_map(feature_spec)
    context = formula_context(np)
    offset_terms = [str(term) for term in (manifest.get("offset_terms") or manifest.get("formula", {}).get("offset_terms") or [])]
    groups = _term_groups(estimator, offset_terms, source_columns)
    non_base_groups = {features: info for features, info in groups.items() if features}
    all_features = sorted({feature for features in non_base_groups for feature in features})
    required_columns = _required_tabulation_columns(source_columns, manifest, all_features, offset_terms)
    frame = _tabulation_frame_from_dataset(dataset, required_columns)
    fit_frame = _fit_frame_for_levels(frame, manifest, pd)
    transform_bounds = _feature_transform_bounds(estimator, source_columns)
    base = _base_row(frame)
    inferred_bases: list[dict[str, Any]] = []
    clipped_bounds: list[dict[str, Any]] = []
    for feature in all_features:
        value, inferred, clipped = _base_value(
            frame,
            fit_frame,
            feature,
            kinds.get(feature, "categorical"),
            spec_rows.get(feature, {}),
            transform_bounds.get(feature, {}),
            pd,
        )
        base[feature] = value
        if inferred:
            inferred_bases.append(inferred)
        if clipped:
            clipped_bounds.append(clipped)

    warnings: list[str] = []
    estimated_specs: list[dict[str, Any]] = []
    unseen_levels: list[dict[str, Any]] = []
    tables: list[dict[str, Any]] = []
    feature_meta: dict[str, dict[str, Any]] = {}
    feature_levels: dict[str, list[dict[str, Any]]] = {}
    for feature in all_features:
        levels, meta, estimated, clipped, unseen = _feature_levels(
            frame,
            fit_frame,
            feature,
            kinds.get(feature, "categorical"),
            spec_rows.get(feature, {}),
            base.get(feature),
            transform_bounds.get(feature, {}),
            np,
            pd,
        )
        feature_levels[feature] = levels
        feature_meta[feature] = meta
        estimated_specs.extend(estimated)
        clipped_bounds.extend(clipped)
        unseen_levels.extend(unseen)
    for entry in estimated_specs:
        fields = ", ".join(entry["fields"])
        warnings.append(f"Estimated {fields} for numeric GLM tabulation feature {entry['feature']} from scored rows.")
    for entry in inferred_bases:
        warnings.append(f"Estimated base for GLM tabulation feature {entry['feature']} from scored rows: {entry['value']}.")
    for entry in clipped_bounds:
        source = "feature spec" if entry.get("source") == "feature_spec" else "inferred"
        warnings.append(
            f"Clipped {source} {entry['field']} for GLM tabulation feature {entry['feature']} "
            f"from {entry['value']} to {entry['clipped']} to stay within fitted {entry['bound']}."
        )
    if unseen_levels:
        by_feature: dict[str, int] = defaultdict(int)
        for entry in unseen_levels:
            by_feature[str(entry["feature"])] += 1
        warnings.extend(
            f"{count} dataset level{'s' if count != 1 else ''} for {feature} were not seen in training and tabulate as NA."
            for feature, count in sorted(by_feature.items())
        )

    store.tabulations_dir(model_id).mkdir(parents=True, exist_ok=True)
    cumulative_adjustment = 0.0
    table_index = 1
    skipped_tables: list[dict[str, Any]] = []
    table_frames: dict[str, Any] = {}
    for features, info in sorted(non_base_groups.items(), key=lambda item: (len(item[0]), item[0])):
        table_id = "|".join(features)
        table_label = " × ".join(features)
        cell_count = 1
        for feature in features:
            cell_count *= max(1, len(feature_levels.get(feature, [])))
        progress_callback({"phase": "tabulating", "message": f"Tabulating {table_label}", "model_id": model_id, "table_id": table_id, "cells": cell_count})
        if cell_count > MAX_TABULATION_CELLS:
            warning = f"Skipped {table_label}: {cell_count:,} cells exceeds the 100,000-cell guard."
            warnings.append(warning)
            skipped = {"table_id": table_id, "label": table_label, "features": list(features), "cell_count": cell_count, "skipped": True, "warning": warning}
            skipped_tables.append(skipped)
            tables.append(skipped)
            table_index += 1
            continue
        grid = _cartesian_table({feature: feature_levels[feature] for feature in features}, pd)
        table = grid.copy()
        table["tabulated_linear"] = np.nan
        ok_mask = table["__status"].astype(str) == "ok"
        ok_grid = table.loc[ok_mask].reset_index(drop=True)
        if len(ok_grid):
            pred_frame = _prediction_frame(base, list(features), ok_grid, pd)
            contribution = _group_contribution(estimator, pred_frame, list(info["term_indices"]), list(info["offset_terms"]), context, np, pd)
            base_grid = pd.DataFrame([{feature: base.get(feature) for feature in features}])
            base_frame = _prediction_frame(base, list(features), base_grid, pd)
            base_contribution = float(_group_contribution(estimator, base_frame, list(info["term_indices"]), list(info["offset_terms"]), context, np, pd)[0])
            cumulative_adjustment += base_contribution
            table.loc[ok_mask, "tabulated_linear"] = contribution - base_contribution
            table["base_adjustment"] = base_contribution
        else:
            table["base_adjustment"] = None
        table["table_id"] = table_id
        table["status"] = table["__status"]
        table = table.drop(columns=["__status"])
        write_dataframe_parquet(table, _table_file_path(store, model_id, table_id))
        table_frames[table_id] = table
        tables.append(
            {
                "table_id": table_id,
                "label": table_label,
                "index": table_index,
                "features": list(features),
                "cell_count": int(cell_count),
                "skipped": False,
                "path": f"tabulations/{_safe_id(table_id)}.parquet",
                "min": json_safe_number(table["tabulated_linear"].min(skipna=True)),
                "max": json_safe_number(table["tabulated_linear"].max(skipna=True)),
            }
        )
        table_index += 1

    if () in groups:
        base_grid = pd.DataFrame([base])
        cumulative_adjustment += float(
            _group_contribution(
                estimator,
                base_grid,
                list(groups[()]["term_indices"]),
                list(groups[()]["offset_terms"]),
                context,
                np,
                pd,
            )[0]
        )
    base_value = float(getattr(estimator, "intercept_", 0.0) or 0.0) + cumulative_adjustment
    base_table = pd.DataFrame([{"table_id": "base", "status": "ok", "tabulated_linear": base_value, "base_adjustment": cumulative_adjustment}])
    write_dataframe_parquet(base_table, _table_file_path(store, model_id, "base"))
    tables.insert(0, {"table_id": "base", "label": "base", "index": 0, "features": [], "cell_count": 1, "skipped": False, "path": "tabulations/base.parquet", "min": base_value, "max": base_value})

    progress_callback({"phase": "scoring", "message": f"Scoring tabulated GLM {model_id}", "model_id": model_id})
    tabulated = frame[["__lucidum_row_id"]].copy()
    eta = pd.Series(base_value, index=frame.index, dtype=float)
    missing = pd.Series(False, index=frame.index, dtype=bool)
    for table_info in tables:
        if table_info["table_id"] == "base" or table_info.get("skipped"):
            continue
        features = list(table_info.get("features") or [])
        component = _component_from_table(frame, table_frames.get(str(table_info["table_id"])), features, feature_meta, np, pd)
        missing = missing | component.isna()
        eta = eta + component.fillna(0.0)
        safe_name = _safe_id(str(table_info["table_id"]))
        tabulated[f"tabulated_linear__{safe_name}"] = component

    finite_eta = (~missing) & np.isfinite(eta.astype(float))
    prediction = pd.Series(np.nan, index=frame.index, dtype=float)
    if finite_eta.any():
        inverse = estimator.link_instance.inverse(eta.loc[finite_eta].to_numpy(dtype=float))
        prediction.loc[finite_eta] = pd.to_numeric(inverse, errors="coerce")
    denominator_column = str(manifest.get("denominator_column") or "")
    if denominator_column and denominator_column in frame.columns:
        denominator = pd.to_numeric(frame[denominator_column], errors="coerce")
        valid_denominator = denominator.notna() & np.isfinite(denominator.astype(float)) & (denominator.astype(float) > 0)
        prediction = prediction * denominator
        missing = missing | ~valid_denominator
    tabulated["glm_tabulated_prediction"] = prediction
    tabulated["glm_tabulated_linear_prediction"] = eta.where(~missing, np.nan)
    tabulated["glm_tabulation_missing"] = missing

    score_frame = frame.copy()
    score_frame[TARGET_COLUMN] = 0.0
    score_mask = pd.Series(True, index=frame.index)
    offset_values = offset_values_for_frame(score_frame, offset_terms, context, np, pd)
    if offset_values is not None:
        score_mask = score_mask & offset_values.notna() & np.isfinite(offset_values.astype(float))
    try:
        exact_eta = pd.Series(np.nan, index=frame.index, dtype=float)
        exact_eta.loc[score_mask] = estimator.linear_predictor(
            score_frame.loc[score_mask],
            context=context,
            offset=offset_values.loc[score_mask].astype(float).to_numpy() if offset_values is not None else None,
        )
        error = exact_eta - tabulated["glm_tabulated_linear_prediction"]
        linear_sd_error = json_safe_number(float(error.dropna().std())) if len(error.dropna()) > 1 else 0.0
    except Exception:
        linear_sd_error = None

    write_dataframe_parquet(tabulated, store.artifact_path(model_id, "tabulated_predictions"))
    diagnostics = {
        "linear_sd_error": linear_sd_error,
        "scored_rows": int(finite_eta.sum()),
        "tabulated_row_count": int(len(tabulated)),
        "missing_tabulated_prediction_rows": int(missing.sum()),
        "estimated_spec_fields": estimated_specs,
        "estimated_base_fields": inferred_bases,
        "clipped_spec_fields": clipped_bounds,
        "unseen_levels": unseen_levels,
        "skipped_oversized_tables": skipped_tables,
    }
    manifest_payload = {
        "model_id": model_id,
        "status": "tabulated",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "max_cells": MAX_TABULATION_CELLS,
        "tables": tables,
        "feature_meta": feature_meta,
        "warnings": warnings,
        "diagnostics": diagnostics,
    }
    store.write_json(store.artifact_path(model_id, "tabulation_manifest"), manifest_payload)
    return manifest_payload


def build_tabulations(
    dataset: Dataset,
    store: GlmModelStore,
    payload: dict[str, Any],
    feature_spec: Any,
    *,
    progress_callback: ProgressCallback | None = None,
    gbm_store: Any = None,
) -> dict[str, Any]:
    progress = progress_callback or (lambda _progress: None)
    model_refs = _requested_model_refs(payload, for_build=True, store=store)
    if not model_refs:
        raise ValueError("Choose at least one model to tabulate")
    results: list[dict[str, Any]] = []
    for index, model_ref in enumerate(model_refs, start=1):
        progress({"phase": "starting", "message": f"Tabulating {model_ref.label_kind} {index} of {len(model_refs)}", "model_id": model_ref.model_id, "model_ref": model_ref.ref, "percent": int((index - 1) / len(model_refs) * 100)})
        if model_ref.kind == "gbm":
            if gbm_store is None:
                raise ValueError("GBM tabulation is unavailable because the GBM tool is not loaded")
            from py_lucidum.tools.gbm.tabulation import build_gbm_tabulations

            gbm_store.validate_model_id(model_ref.model_id)
            results.append(build_gbm_tabulations(dataset, gbm_store, model_ref.model_id, feature_spec, progress_callback=progress))
        else:
            store.validate_model_id(model_ref.model_id)
            result = _build_model_tabulations(dataset, store, model_ref.model_id, feature_spec, progress)
            result.setdefault("model_kind", "glm")
            result.setdefault("model_ref", f"glm:{model_ref.model_id}")
            results.append(result)
    dataset.reload()
    return {
        "models": results,
        "model_ids": [ref.model_id for ref in model_refs if ref.kind == "glm"],
        "gbm_model_ids": [ref.model_id for ref in model_refs if ref.kind == "gbm"],
        "model_refs": [ref.ref for ref in model_refs],
    }


def _tabulation_manifest(store: GlmModelStore, model_id: str) -> dict[str, Any] | None:
    payload = store.read_json(store.artifact_path(model_id, "tabulation_manifest"), None)
    return payload if isinstance(payload, dict) else None


def _tabulation_model_status(store: GlmModelStore, model: dict[str, Any]) -> dict[str, Any]:
    model_id = str(model.get("model_id") or "")
    manifest = _tabulation_manifest(store, model_id)
    tabulatable = store.artifact_path(model_id, "estimator").exists()
    tables = list(manifest.get("tables", [])) if manifest else []
    model_warnings = list(manifest.get("warnings", [])) if manifest else []
    if not tabulatable:
        model_warnings.append("Rebuild this GLM before tabulating; estimator.pkl is missing.")
    return {
        "model_id": model_id,
        "model_ref": f"glm:{model_id}",
        "model_kind": "glm",
        "label": model.get("label") or model_id,
        "active": bool(model.get("active")),
        "tabulatable": tabulatable,
        "tabulated": bool(manifest),
        "tables": tables,
        "warnings": model_warnings,
        "diagnostics": manifest.get("diagnostics", {}) if manifest else {},
    }


def tabulation_config(store: GlmModelStore, payload: dict[str, Any], *, gbm_store: Any = None) -> dict[str, Any]:
    requested_refs = _requested_model_refs(payload)
    glm_models = store.list_models()
    glm_statuses = [_tabulation_model_status(store, model) for model in glm_models]
    gbm_statuses: list[dict[str, Any]] = []
    if gbm_store is not None:
        from py_lucidum.tools.gbm.tabulation import tabulation_model_status as gbm_tabulation_model_status

        gbm_statuses = [gbm_tabulation_model_status(gbm_store, model) for model in gbm_store.list_models()]
    all_statuses = [*glm_statuses, *gbm_statuses]
    by_ref = {_model_ref_for_status(status): status for status in all_statuses}
    by_legacy_glm_id = {str(status.get("model_id") or ""): status for status in glm_statuses}
    if requested_refs:
        selected_statuses = [
            status
            for ref in requested_refs
            if (status := by_ref.get(ref.ref) or (by_legacy_glm_id.get(ref.model_id) if ref.legacy_field else None))
        ]
    else:
        selected_statuses = all_statuses
    union: dict[str, dict[str, Any]] = {}
    warnings: list[str] = []
    for status in selected_statuses:
        for table in status["tables"]:
            table_id = str(table.get("table_id") or "")
            if table_id and table_id not in union:
                union[table_id] = dict(table)
        warnings.extend(str(warning) for warning in status["warnings"])
    ordered_tables = sorted(union.values(), key=lambda table: (int(table.get("index", 9999)), str(table.get("table_id") or "")))
    return {"models": selected_statuses, "all_models": all_statuses, "tables": ordered_tables, "warnings": warnings}


def _read_table(store: GlmModelStore, model_id: str, table_id: str) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
    manifest = _tabulation_manifest(store, model_id)
    if not manifest:
        return None, []
    table_info = next((table for table in manifest.get("tables", []) if str(table.get("table_id") or "") == table_id), None)
    if not table_info or table_info.get("skipped"):
        return table_info, []
    return table_info, store.read_parquet_records(_table_file_path(store, model_id, table_id))


def _read_table_for_ref(
    store: GlmModelStore,
    gbm_store: Any,
    model_ref: _TabulationModelRef,
    table_id: str,
) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
    if model_ref.kind == "gbm":
        if gbm_store is None:
            return None, []
        from py_lucidum.tools.gbm.tabulation import read_table as read_gbm_table

        return read_gbm_table(gbm_store, model_ref.model_id, table_id)
    return _read_table(store, model_ref.model_id, table_id)


def _status_for_ref(store: GlmModelStore, gbm_store: Any, model_ref: _TabulationModelRef) -> dict[str, Any] | None:
    if model_ref.kind == "gbm":
        if gbm_store is None:
            return None
        from py_lucidum.tools.gbm.tabulation import tabulation_model_status as gbm_tabulation_model_status

        models = {str(model.get("model_id") or ""): model for model in gbm_store.list_models()}
        model = models.get(model_ref.model_id)
        return gbm_tabulation_model_status(gbm_store, model) if model else None
    models = {str(model.get("model_id") or ""): model for model in store.list_models()}
    model = models.get(model_ref.model_id)
    return _tabulation_model_status(store, model) if model else None


def _display_number(value: Any, scale: str) -> float | None:
    number = _as_number(value)
    if number is None:
        return None
    if scale == "exp":
        try:
            return math.exp(number)
        except OverflowError:
            return None
    return number


def _tabulation_value_column(title: Any, field: str) -> dict[str, Any]:
    return {
        "title": str(title),
        "field": field,
        "hozAlign": "right",
        "tabulation_value": True,
        "status_field": f"__status__{field}",
    }


def _tabulation_table_payload(
    *,
    table_id: str,
    scale: str,
    crosstab: str,
    columns: list[dict[str, Any]],
    rows: list[dict[str, Any]],
    notices: list[str],
) -> dict[str, Any]:
    value_fields = [str(column.get("field") or "") for column in columns if column.get("tabulation_value")]
    values = [row.get(field) for row in rows for field in value_fields if isinstance(row.get(field), (int, float))]
    return {
        "table_id": table_id,
        "scale": scale,
        "crosstab": crosstab,
        "columns": columns,
        "rows": rows,
        "notices": notices,
        "min": min(values) if values else None,
        "max": max(values) if values else None,
    }


def tabulation_table(store: GlmModelStore, payload: dict[str, Any], *, gbm_store: Any = None) -> dict[str, Any]:
    model_refs = _requested_model_refs(payload)
    table_id = str(payload.get("table_id") or "base").strip() or "base"
    crosstab = str(payload.get("crosstab") or "").strip()
    scale = "exp" if str(payload.get("scale") or "").lower() == "exp" else "linear"
    model_rows: list[tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]] = []
    feature_columns: list[str] = []
    notices: list[str] = []
    model_entries: list[dict[str, Any]] = []
    for model_ref in model_refs:
        status = _status_for_ref(store, gbm_store, model_ref)
        entry = {
            "field": model_ref.field,
            "label": _model_field_label(model_ref, status),
            "model_id": model_ref.model_id,
            "model_ref": model_ref.ref,
            "model_kind": model_ref.kind,
        }
        model_entries.append(entry)
        table_info, rows = _read_table_for_ref(store, gbm_store, model_ref, table_id)
        if not table_info:
            notices.append(f"{entry['label']} has no {table_id} tabulation.")
            continue
        features = list(table_info.get("features") or [])
        if not feature_columns:
            feature_columns = features
        if table_info.get("skipped"):
            notices.append(str(table_info.get("warning") or f"{entry['label']} skipped {table_id}."))
            continue
        model_rows.append((entry, table_info, rows))
    if crosstab and crosstab not in {MODEL_CROSSTAB, *feature_columns}:
        notices.append(f"Ignoring unknown crosstab {crosstab}.")
        crosstab = ""
    if crosstab and crosstab != MODEL_CROSSTAB and crosstab in feature_columns:
        return _tabulation_table_feature_crosstab(
            table_id=table_id,
            scale=scale,
            crosstab=crosstab,
            model_entries=model_entries,
            model_rows=model_rows,
            feature_columns=feature_columns,
            notices=notices,
        )
    return _tabulation_table_long(
        table_id=table_id,
        scale=scale,
        crosstab=crosstab,
        model_entries=model_entries,
        model_rows=model_rows,
        feature_columns=feature_columns,
        notices=notices,
    )


def _tabulation_table_long(
    *,
    table_id: str,
    scale: str,
    crosstab: str,
    model_entries: list[dict[str, Any]],
    model_rows: list[tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]],
    feature_columns: list[str],
    notices: list[str],
) -> dict[str, Any]:
    row_map: dict[tuple[Any, ...], dict[str, Any]] = {}
    for model, table_info, rows in model_rows:
        field = str(model["field"])
        features = list(table_info.get("features") or [])
        for row in rows:
            key = tuple(_json_value(row.get(feature)) for feature in features) if features else ("base",)
            target = row_map.setdefault(key, {feature: _json_value(row.get(feature)) for feature in features})
            if not features:
                target["table"] = "base"
            target[field] = _display_number(row.get("tabulated_linear"), scale)
            target[f"__status__{field}"] = str(row.get("status") or "ok")
    columns = [{"title": feature, "field": feature} for feature in feature_columns]
    if not feature_columns:
        columns = [{"title": "table", "field": "table"}]
    columns.extend(_tabulation_value_column(model["label"], str(model["field"])) for model in model_entries)
    return _tabulation_table_payload(table_id=table_id, scale=scale, crosstab=crosstab, columns=columns, rows=list(row_map.values()), notices=notices)


def _tabulation_table_feature_crosstab(
    *,
    table_id: str,
    scale: str,
    crosstab: str,
    model_entries: list[dict[str, Any]],
    model_rows: list[tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]],
    feature_columns: list[str],
    notices: list[str],
) -> dict[str, Any]:
    remaining_features = [feature for feature in feature_columns if feature != crosstab]
    crosstab_values = _ordered_tabulation_values({_json_value(row.get(crosstab)) for _, _, rows in model_rows for row in rows})
    pivot_fields = {value: f"__pivot__{index}" for index, value in enumerate(crosstab_values)}
    include_model_column = len(model_entries) > 1
    row_map: dict[tuple[Any, ...], dict[str, Any]] = {}
    model_labels = {str(model["field"]): str(model["label"]) for model in model_entries}
    for model, table_info, rows in model_rows:
        field_name = str(model["field"])
        features = list(table_info.get("features") or [])
        for row in rows:
            base_values = tuple(_json_value(row.get(feature)) for feature in remaining_features)
            key = (*base_values, field_name) if include_model_column else base_values or ("base",)
            target = row_map.setdefault(key, {feature: _json_value(row.get(feature)) for feature in remaining_features})
            if include_model_column:
                target["model"] = model_labels.get(field_name, field_name)
            pivot_value = _json_value(row.get(crosstab))
            field = pivot_fields.get(pivot_value)
            if not field:
                continue
            target[field] = _display_number(row.get("tabulated_linear"), scale)
            target[f"__status__{field}"] = str(row.get("status") or "ok")
    columns = [{"title": feature, "field": feature} for feature in remaining_features]
    if include_model_column:
        columns.append({"title": "model", "field": "model"})
    columns.extend(_tabulation_value_column(value, pivot_fields[value]) for value in crosstab_values)
    return _tabulation_table_payload(table_id=table_id, scale=scale, crosstab=crosstab, columns=columns, rows=list(row_map.values()), notices=notices)


def tabulation_plot(store: GlmModelStore, payload: dict[str, Any], *, gbm_store: Any = None) -> dict[str, Any]:
    model_refs = _requested_model_refs(payload)
    table_id = str(payload.get("table_id") or "base").strip() or "base"
    crosstab = str(payload.get("crosstab") or "").strip()
    scale = "exp" if str(payload.get("scale") or "").lower() == "exp" else "linear"
    series: list[dict[str, Any]] = []
    x_axis: list[Any] = []
    notices: list[str] = []
    first_features: list[str] = []
    for model_ref in model_refs:
        status = _status_for_ref(store, gbm_store, model_ref)
        label = _model_field_label(model_ref, status)
        table_info, rows = _read_table_for_ref(store, gbm_store, model_ref, table_id)
        if not table_info or table_info.get("skipped"):
            notices.append(f"{label} has no plottable {table_id} tabulation.")
            continue
        features = list(table_info.get("features") or [])
        first_features = first_features or features
        if len(features) == 0:
            series.append({"name": label, "type": "bar", "data": [_display_number(rows[0].get("tabulated_linear"), scale)] if rows else []})
            x_axis = ["base"]
        elif len(features) == 1:
            feature = features[0]
            ordered_values = _ordered_tabulation_values(row.get(feature) for row in rows)
            by_value = {_json_value(row.get(feature)): row for row in rows}
            ordered = [by_value[value] for value in ordered_values if value in by_value]
            x_axis = [_json_value(row.get(feature)) for row in ordered]
            series.append({"name": label, "type": "line", "showSymbol": True, "data": [_display_number(row.get("tabulated_linear"), scale) for row in ordered]})
        elif len(features) == 2:
            if crosstab not in features:
                notices.append(f"Choose a feature crosstab to plot {table_id}.")
                continue
            cross = crosstab if crosstab in features else features[1]
            x_feature = next(feature for feature in features if feature != cross)
            x_values = _ordered_tabulation_values({_json_value(row.get(x_feature)) for row in rows})
            x_axis = x_values
            cross_values = _ordered_tabulation_values({_json_value(row.get(cross)) for row in rows})
            lookup = {(_json_value(row.get(x_feature)), _json_value(row.get(cross))): row for row in rows}
            for cross_value in cross_values:
                series.append(
                    {
                        "name": f"{label} · {cross}={cross_value}",
                        "type": "line",
                        "showSymbol": True,
                        "data": [_display_number(lookup.get((x_value, cross_value), {}).get("tabulated_linear"), scale) for x_value in x_values],
                    }
                )
        else:
            notices.append(f"{table_id} has {len(features)} features; plot view is available only for 1D or 2D tables.")
    return {
        "table_id": table_id,
        "scale": scale,
        "features": first_features,
        "x_axis": x_axis,
        "series": series,
        "notices": notices,
        "plottable": bool(series),
    }


__all__ = [
    "MAX_TABULATION_CELLS",
    "build_tabulations",
    "tabulation_config",
    "tabulation_plot",
    "tabulation_table",
]
