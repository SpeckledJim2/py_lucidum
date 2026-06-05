from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable
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
from .training import data_frame_from_dataset, formula_context, glm_dependencies, offset_values_for_frame, write_dataframe_parquet
from .validation import TARGET_COLUMN


ProgressCallback = Callable[[dict[str, Any]], None]
MAX_TABULATION_CELLS = 100_000
MODEL_CROSSTAB = "__model__"


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


def _schema_kinds(dataset: Dataset) -> dict[str, str]:
    return {column.name: column.kind for column in dataset.valid_schema_columns()}


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


def _base_value(frame: Any, feature: str, kind: str, spec_row: dict[str, Any], pd: Any) -> Any:
    raw = str(spec_row.get("base") or "").strip()
    if raw:
        number = _as_number(raw)
        if is_numeric_kind(kind) and number is not None:
            return number
        return raw
    series = frame[feature].dropna() if feature in frame.columns else []
    if is_numeric_kind(kind):
        values = pd.to_numeric(frame[feature], errors="coerce").dropna() if feature in frame.columns else []
        if len(values):
            return float(values.median())
        return 0.0
    if len(series):
        return _json_value(series.iloc[0])
    return ""


def _base_row(frame: Any, features: list[str], kinds: dict[str, str], spec_rows: dict[str, dict[str, Any]], pd: Any) -> dict[str, Any]:
    row: dict[str, Any] = {}
    first = frame.iloc[0].to_dict() if len(frame) else {}
    for column in frame.columns:
        row[str(column)] = _json_value(first.get(column))
    for feature in features:
        if feature in frame.columns:
            row[feature] = _base_value(frame, feature, kinds.get(feature, "categorical"), spec_rows.get(feature, {}), pd)
    row[TARGET_COLUMN] = 0.0
    return row


def _estimated_numeric_spec(frame: Any, feature: str, kind: str, spec_row: dict[str, Any], np: Any, pd: Any) -> tuple[float, float, float, list[str]]:
    warnings: list[str] = []
    values = pd.to_numeric(frame[feature], errors="coerce").dropna()
    data_min = float(values.min()) if len(values) else 0.0
    data_max = float(values.max()) if len(values) else data_min + 1.0
    stddev = float(values.std()) if len(values) > 1 else abs(data_max - data_min)
    raw_min = _as_number(spec_row.get("min"))
    raw_max = _as_number(spec_row.get("max"))
    raw_band = _as_number(spec_row.get("banding"))
    band = raw_band if raw_band and raw_band > 0 else _as_number(suggested_band_width(stddev)) or 1.0
    minimum = raw_min if raw_min is not None else math.floor(data_min / band) * band
    maximum = raw_max if raw_max is not None else math.ceil(data_max / band) * band
    if maximum < minimum:
        minimum, maximum = maximum, minimum
    if maximum == minimum:
        maximum = minimum + band
    if raw_min is None:
        warnings.append("min")
    if raw_max is None:
        warnings.append("max")
    if raw_band is None or raw_band <= 0:
        warnings.append("banding")
    if is_numeric_kind(kind) and kind == "integer" and band >= 1:
        band = int(round(band))
    return float(minimum), float(maximum), float(band), warnings


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
    np: Any,
    pd: Any,
) -> tuple[list[dict[str, Any]], dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    estimated: list[dict[str, Any]] = []
    unseen: list[dict[str, Any]] = []
    meta: dict[str, Any] = {"feature": feature, "kind": kind, "base": _json_value(base_value)}
    if is_numeric_kind(kind):
        minimum, maximum, band, estimated_fields = _estimated_numeric_spec(frame, feature, kind, spec_row, np, pd)
        if estimated_fields:
            estimated.append({"feature": feature, "fields": estimated_fields, "min": minimum, "max": maximum, "banding": band})
        meta.update({"min": minimum, "max": maximum, "banding": band})
        return [{"value": value, "status": "ok"} for value in _numeric_levels(minimum, maximum, band, base_value)], meta, estimated, unseen
    levels, seen = _categorical_levels(frame, fit_frame, feature, base_value, pd)
    rows: list[dict[str, Any]] = []
    for value in levels:
        status = "ok" if value in seen else "unseen"
        if status == "unseen":
            unseen.append({"feature": feature, "level": _json_value(value)})
        rows.append({"value": value, "status": status})
    return rows, meta, estimated, unseen


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


def _scored_level(value: Any, meta: dict[str, Any]) -> Any:
    if not is_numeric_kind(str(meta.get("kind") or "")):
        return _json_value(value)
    number = _as_number(value)
    minimum = _as_number(meta.get("min"))
    maximum = _as_number(meta.get("max"))
    band = _as_number(meta.get("banding"))
    if number is None or minimum is None or maximum is None or not band or band <= 0:
        return None
    clipped = min(max(number, minimum), maximum)
    index = math.floor((clipped - minimum) / band + 1e-9)
    return _round_grid_value(minimum + index * band)


def _table_file_path(store: GlmModelStore, model_id: str, table_id: str) -> Path:
    return store.tabulations_dir(model_id) / f"{_safe_id(table_id)}.parquet"


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
    frame, source_columns = data_frame_from_dataset(dataset)
    kinds = _schema_kinds(dataset)
    spec_rows = _feature_spec_map(feature_spec)
    fit_frame = _fit_frame_for_levels(frame, manifest, pd)
    context = formula_context(np)
    offset_terms = [str(term) for term in (manifest.get("offset_terms") or manifest.get("formula", {}).get("offset_terms") or [])]
    groups = _term_groups(estimator, offset_terms, source_columns)
    non_base_groups = {features: info for features, info in groups.items() if features}
    all_features = sorted({feature for features in non_base_groups for feature in features})
    base = _base_row(frame, source_columns, kinds, spec_rows, pd)
    for feature in all_features:
        base[feature] = _base_value(frame, feature, kinds.get(feature, "categorical"), spec_rows.get(feature, {}), pd)

    warnings: list[str] = []
    estimated_specs: list[dict[str, Any]] = []
    unseen_levels: list[dict[str, Any]] = []
    tables: list[dict[str, Any]] = []
    feature_meta: dict[str, dict[str, Any]] = {}
    feature_levels: dict[str, list[dict[str, Any]]] = {}
    for feature in all_features:
        levels, meta, estimated, unseen = _feature_levels(
            frame,
            fit_frame,
            feature,
            kinds.get(feature, "categorical"),
            spec_rows.get(feature, {}),
            base.get(feature),
            np,
            pd,
        )
        feature_levels[feature] = levels
        feature_meta[feature] = meta
        estimated_specs.extend(estimated)
        unseen_levels.extend(unseen)
    for entry in estimated_specs:
        fields = ", ".join(entry["fields"])
        warnings.append(f"Estimated {fields} for numeric GLM tabulation feature {entry['feature']} from scored rows.")
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
        table_path = _table_file_path(store, model_id, str(table_info["table_id"]))
        table_rows = store.read_parquet_records(table_path)
        lookup: dict[tuple[Any, ...], Any] = {}
        for row in table_rows:
            if str(row.get("status") or "ok") != "ok":
                continue
            key = tuple(_json_value(row.get(feature)) for feature in features)
            lookup[key] = row.get("tabulated_linear")
        values: list[float | None] = []
        for _, source_row in frame.iterrows():
            key = tuple(_scored_level(source_row[feature], feature_meta[feature]) for feature in features)
            value = lookup.get(key)
            number = _as_number(value)
            values.append(number)
        component = pd.Series(values, index=frame.index, dtype=float)
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
) -> dict[str, Any]:
    progress = progress_callback or (lambda _progress: None)
    model_ids = [str(model_id).strip() for model_id in payload.get("model_ids", []) if str(model_id).strip()]
    if not model_ids:
        active = store.active_model_id()
        model_ids = [active] if active else []
    if not model_ids:
        raise ValueError("Choose at least one GLM model to tabulate")
    results: list[dict[str, Any]] = []
    for index, model_id in enumerate(model_ids, start=1):
        store.validate_model_id(model_id)
        progress({"phase": "starting", "message": f"Tabulating GLM {index} of {len(model_ids)}", "model_id": model_id, "percent": int((index - 1) / len(model_ids) * 100)})
        results.append(_build_model_tabulations(dataset, store, model_id, feature_spec, progress))
    dataset.reload()
    return {"models": results, "model_ids": model_ids}


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
        "label": model.get("label") or model_id,
        "active": bool(model.get("active")),
        "tabulatable": tabulatable,
        "tabulated": bool(manifest),
        "tables": tables,
        "warnings": model_warnings,
        "diagnostics": manifest.get("diagnostics", {}) if manifest else {},
    }


def tabulation_config(store: GlmModelStore, payload: dict[str, Any]) -> dict[str, Any]:
    requested = [str(model_id).strip() for model_id in payload.get("model_ids", []) if str(model_id).strip()]
    all_models = store.list_models()
    by_id = {str(model.get("model_id")): model for model in all_models}
    selected_models = [by_id[model_id] for model_id in requested if model_id in by_id] if requested else all_models
    all_statuses = [_tabulation_model_status(store, model) for model in all_models]
    selected_statuses = [_tabulation_model_status(store, model) for model in selected_models]
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


def tabulation_table(store: GlmModelStore, payload: dict[str, Any]) -> dict[str, Any]:
    model_ids = [str(model_id).strip() for model_id in payload.get("model_ids", []) if str(model_id).strip()]
    table_id = str(payload.get("table_id") or "base").strip() or "base"
    crosstab = str(payload.get("crosstab") or "").strip()
    scale = "exp" if str(payload.get("scale") or "").lower() == "exp" else "linear"
    model_rows: list[tuple[str, dict[str, Any], list[dict[str, Any]]]] = []
    feature_columns: list[str] = []
    notices: list[str] = []
    for model_id in model_ids:
        table_info, rows = _read_table(store, model_id, table_id)
        if not table_info:
            notices.append(f"{model_id} has no {table_id} tabulation.")
            continue
        features = list(table_info.get("features") or [])
        if not feature_columns:
            feature_columns = features
        if table_info.get("skipped"):
            notices.append(str(table_info.get("warning") or f"{model_id} skipped {table_id}."))
            continue
        model_rows.append((model_id, table_info, rows))
    if crosstab and crosstab not in {MODEL_CROSSTAB, *feature_columns}:
        notices.append(f"Ignoring unknown crosstab {crosstab}.")
        crosstab = ""
    if crosstab and crosstab != MODEL_CROSSTAB and crosstab in feature_columns:
        return _tabulation_table_feature_crosstab(
            table_id=table_id,
            scale=scale,
            crosstab=crosstab,
            model_ids=model_ids,
            model_rows=model_rows,
            feature_columns=feature_columns,
            notices=notices,
        )
    return _tabulation_table_long(
        table_id=table_id,
        scale=scale,
        crosstab=crosstab,
        model_ids=model_ids,
        model_rows=model_rows,
        feature_columns=feature_columns,
        notices=notices,
    )


def _tabulation_table_long(
    *,
    table_id: str,
    scale: str,
    crosstab: str,
    model_ids: list[str],
    model_rows: list[tuple[str, dict[str, Any], list[dict[str, Any]]]],
    feature_columns: list[str],
    notices: list[str],
) -> dict[str, Any]:
    row_map: dict[tuple[Any, ...], dict[str, Any]] = {}
    for model_id, table_info, rows in model_rows:
        features = list(table_info.get("features") or [])
        for row in rows:
            key = tuple(_json_value(row.get(feature)) for feature in features) if features else ("base",)
            target = row_map.setdefault(key, {feature: _json_value(row.get(feature)) for feature in features})
            if not features:
                target["table"] = "base"
            target[model_id] = _display_number(row.get("tabulated_linear"), scale)
            target[f"__status__{model_id}"] = str(row.get("status") or "ok")
    columns = [{"title": feature, "field": feature} for feature in feature_columns]
    if not feature_columns:
        columns = [{"title": "table", "field": "table"}]
    columns.extend(_tabulation_value_column(model_id, model_id) for model_id in model_ids)
    return _tabulation_table_payload(table_id=table_id, scale=scale, crosstab=crosstab, columns=columns, rows=list(row_map.values()), notices=notices)


def _tabulation_table_feature_crosstab(
    *,
    table_id: str,
    scale: str,
    crosstab: str,
    model_ids: list[str],
    model_rows: list[tuple[str, dict[str, Any], list[dict[str, Any]]]],
    feature_columns: list[str],
    notices: list[str],
) -> dict[str, Any]:
    remaining_features = [feature for feature in feature_columns if feature != crosstab]
    crosstab_values = _ordered_tabulation_values({_json_value(row.get(crosstab)) for _, _, rows in model_rows for row in rows})
    pivot_fields = {value: f"__pivot__{index}" for index, value in enumerate(crosstab_values)}
    include_model_column = len(model_ids) > 1
    row_map: dict[tuple[Any, ...], dict[str, Any]] = {}
    for model_id, table_info, rows in model_rows:
        features = list(table_info.get("features") or [])
        for row in rows:
            base_values = tuple(_json_value(row.get(feature)) for feature in remaining_features)
            key = (*base_values, model_id) if include_model_column else base_values or ("base",)
            target = row_map.setdefault(key, {feature: _json_value(row.get(feature)) for feature in remaining_features})
            if include_model_column:
                target["model"] = model_id
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


def tabulation_plot(store: GlmModelStore, payload: dict[str, Any]) -> dict[str, Any]:
    model_ids = [str(model_id).strip() for model_id in payload.get("model_ids", []) if str(model_id).strip()]
    table_id = str(payload.get("table_id") or "base").strip() or "base"
    crosstab = str(payload.get("crosstab") or "").strip()
    scale = "exp" if str(payload.get("scale") or "").lower() == "exp" else "linear"
    series: list[dict[str, Any]] = []
    x_axis: list[Any] = []
    notices: list[str] = []
    first_features: list[str] = []
    for model_id in model_ids:
        table_info, rows = _read_table(store, model_id, table_id)
        if not table_info or table_info.get("skipped"):
            notices.append(f"{model_id} has no plottable {table_id} tabulation.")
            continue
        features = list(table_info.get("features") or [])
        first_features = first_features or features
        if len(features) == 0:
            series.append({"name": model_id, "type": "bar", "data": [_display_number(rows[0].get("tabulated_linear"), scale)] if rows else []})
            x_axis = ["base"]
        elif len(features) == 1:
            feature = features[0]
            ordered_values = _ordered_tabulation_values(row.get(feature) for row in rows)
            by_value = {_json_value(row.get(feature)): row for row in rows}
            ordered = [by_value[value] for value in ordered_values if value in by_value]
            x_axis = [_json_value(row.get(feature)) for row in ordered]
            series.append({"name": model_id, "type": "line", "showSymbol": True, "data": [_display_number(row.get("tabulated_linear"), scale) for row in ordered]})
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
                        "name": f"{model_id} · {cross}={cross_value}",
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
