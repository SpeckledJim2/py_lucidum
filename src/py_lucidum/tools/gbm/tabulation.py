from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable
import math
import time
from pathlib import Path
from typing import Any

import duckdb

from py_lucidum.core import Dataset, is_numeric_kind, quote_ident, sql_literal
from py_lucidum.tools.glm.tabulation import (
    MAX_TABULATION_CELLS,
    _as_number,
    _base_row,
    _base_value,
    _cartesian_table,
    _component_from_table,
    _feature_levels,
    _feature_spec_map,
    _json_value,
    _safe_id,
    _schema_columns_and_kinds,
    _scored_feature_series,
)

from .store import GbmModelStore, json_safe_number
from .validation import init_score_transform, uses_log_offset


ProgressCallback = Callable[[dict[str, Any]], None]
MAX_GBM_TABULATION_LEAVES = 3
MAX_GBM_TABULATION_FEATURES = 2
GBM_MISSING_KEY = "__lucidum_missing__"


def _write_dataframe_parquet(frame: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(database=":memory:")
    try:
        con.register("artifact_df", frame)
        con.execute(f"COPY artifact_df TO {sql_literal(str(path))} (FORMAT PARQUET)")
    finally:
        con.close()


def gbm_tabulation_dependencies() -> tuple[Any, Any]:
    missing: list[str] = []
    try:
        import numpy as np  # type: ignore[import-not-found]
    except ImportError:
        np = None
        missing.append("numpy")
    try:
        import pandas as pd  # type: ignore[import-not-found]
    except ImportError:
        pd = None
        missing.append("pandas")
    if missing:
        raise ValueError(f"Install GBM dependencies to tabulate LightGBM models. Missing: {', '.join(missing)}")
    return np, pd


def _table_file_path(store: GbmModelStore, model_id: str, table_id: str) -> Path:
    return store.tabulations_dir(model_id) / f"{_safe_id(table_id)}.parquet"


def _tabulation_manifest(store: GbmModelStore, model_id: str) -> dict[str, Any] | None:
    payload = store.read_json(store.artifact_path(model_id, "tabulation_manifest"), None)
    return payload if isinstance(payload, dict) else None


def _read_tree_table(store: GbmModelStore, model_id: str, best_iteration: Any) -> list[dict[str, Any]]:
    path = store.artifact_path(model_id, "tree_table")
    if not path.exists():
        return []
    best = _positive_int(best_iteration)
    where_sql = "WHERE tree_index IS NOT NULL"
    params: list[Any] = []
    if best is not None:
        where_sql += " AND tree_index < ?"
        params.append(best)
    con = duckdb.connect(database=":memory:")
    try:
        try:
            rows = con.execute(
                f"""
SELECT
  tree_index,
  node_depth,
  node_index,
  left_child,
  right_child,
  parent_index,
  split_feature,
  split_gain,
  threshold,
  threshold_label,
  decision_type,
  missing_direction,
  missing_type,
  value,
  weight,
  count
FROM read_parquet({sql_literal(str(path))})
{where_sql}
ORDER BY tree_index, node_depth, node_index
""",
                params,
            ).fetchall()
        except duckdb.Error:
            return []
        names = [str(col[0]) for col in con.description]
    finally:
        con.close()
    return [dict(zip(names, row)) for row in rows]


def _positive_int(value: Any) -> int | None:
    number = _as_number(value)
    if number is None or number <= 0:
        return None
    return int(number)


def _clean_text(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    if not text or text.lower() == "nan" or text.upper() == "NA":
        return ""
    return text


def _finite_float(value: Any, default: float = 0.0) -> float:
    number = _as_number(value)
    return float(number) if number is not None else float(default)


def _group_tree_rows(rows: list[dict[str, Any]]) -> dict[int, list[dict[str, Any]]]:
    trees: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        tree = _positive_int(row.get("tree_index"))
        if tree is None:
            tree_number = int(float(row.get("tree_index") or 0))
        else:
            tree_number = tree
        trees[tree_number].append(row)
    return dict(sorted(trees.items(), key=lambda item: item[0]))


def _tree_root(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not rows:
        return None
    root = next((row for row in rows if not _clean_text(row.get("parent_index"))), None)
    if root:
        return root
    return min(rows, key=lambda row: (_finite_float(row.get("node_depth")), _clean_text(row.get("node_index"))))


def _tree_leaf_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [row for row in rows if not _clean_text(row.get("split_feature"))]


def _tree_features(rows: list[dict[str, Any]]) -> tuple[str, ...]:
    return tuple(sorted({_clean_text(row.get("split_feature")) for row in rows if _clean_text(row.get("split_feature"))}, key=lambda value: (value.lower(), value)))


def _tree_groups(rows: list[dict[str, Any]]) -> tuple[dict[tuple[str, ...], list[dict[str, Any]]], list[str]]:
    grouped: dict[tuple[str, ...], list[dict[str, Any]]] = defaultdict(list)
    warnings: list[str] = []
    for tree_index, tree_rows in _group_tree_rows(rows).items():
        leaves = _tree_leaf_rows(tree_rows)
        leaf_count = len(leaves) if leaves else 1
        if leaf_count > MAX_GBM_TABULATION_LEAVES:
            warnings.append(
                f"Tree {tree_index} has {leaf_count} leaves; GBM tabulation supports only 2 or 3 leaf trees."
            )
        features = _tree_features(tree_rows)
        if len(features) > MAX_GBM_TABULATION_FEATURES:
            warnings.append(
                f"Tree {tree_index} uses {len(features)} features; GBM tabulation supports only 1D and 2D trees."
            )
        if leaf_count > MAX_GBM_TABULATION_LEAVES or len(features) > MAX_GBM_TABULATION_FEATURES:
            continue
        grouped[features].extend(tree_rows)
    return dict(grouped), warnings


def _nodes_by_id(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {_clean_text(row.get("node_index")): row for row in rows if _clean_text(row.get("node_index"))}


def _categorical_split_values(row: dict[str, Any]) -> set[str]:
    label = _clean_text(row.get("threshold_label"))
    if label:
        return {part.strip() for part in label.split(" / ") if part.strip()}
    threshold = _clean_text(row.get("threshold"))
    if not threshold:
        return set()
    return {part.strip() for part in threshold.replace("||", " / ").split(" / ") if part.strip()}


def _left_mask(frame: Any, row: dict[str, Any], pd: Any) -> Any:
    feature = _clean_text(row.get("split_feature"))
    decision_type = _clean_text(row.get("decision_type")) or "<="
    values = frame[feature] if feature in frame.columns else pd.Series([None] * len(frame), index=frame.index)
    missing = values.isna()
    if decision_type == "==":
        categories = _categorical_split_values(row)
        condition = values.map(lambda value: str(_json_value(value)) in categories if not pd.isna(value) else False)
    else:
        numeric = pd.to_numeric(values, errors="coerce")
        threshold = _as_number(row.get("threshold"))
        if threshold is None:
            condition = pd.Series(False, index=frame.index)
        elif decision_type in {"<=", ""}:
            condition = numeric <= threshold
        elif decision_type == "<":
            condition = numeric < threshold
        elif decision_type == ">=":
            condition = numeric >= threshold
        elif decision_type == ">":
            condition = numeric > threshold
        else:
            condition = numeric <= threshold
    default_left = _clean_text(row.get("missing_direction")).lower() == "left"
    return condition.where(~missing, default_left).fillna(False).astype(bool)


def _evaluate_tree_rows(frame: Any, rows: list[dict[str, Any]], np: Any, pd: Any) -> Any:
    if not len(frame):
        return np.zeros(0, dtype=float)
    root = _tree_root(rows)
    if root is None:
        return np.zeros(len(frame), dtype=float)
    nodes = _nodes_by_id(rows)
    result = pd.Series(np.nan, index=frame.index, dtype=float)

    def visit(row: dict[str, Any], mask: Any) -> None:
        if not bool(mask.any()):
            return
        feature = _clean_text(row.get("split_feature"))
        if not feature:
            result.loc[mask] = _finite_float(row.get("value"))
            return
        left_child = nodes.get(_clean_text(row.get("left_child")))
        right_child = nodes.get(_clean_text(row.get("right_child")))
        left = _left_mask(frame.loc[mask], row, pd).reindex(frame.index, fill_value=False) & mask
        if left_child is not None:
            visit(left_child, left)
        if right_child is not None:
            visit(right_child, mask & ~left)

    visit(root, pd.Series(True, index=frame.index))
    return result.fillna(0.0).to_numpy(dtype=float)


def _evaluate_tree_group(frame: Any, rows: list[dict[str, Any]], np: Any, pd: Any) -> Any:
    total = np.zeros(len(frame), dtype=float)
    for tree_rows in _group_tree_rows(rows).values():
        total = total + _evaluate_tree_rows(frame, tree_rows, np, pd)
    return total


def _tabulation_frame_from_predictions(
    dataset: Dataset,
    store: GbmModelStore,
    model_id: str,
    columns: list[str],
    *,
    include_init_score: bool = False,
) -> Any:
    projection = ["ROW_NUMBER() OVER () AS __lucidum_row_id", *[quote_ident(name) for name in columns]]
    dataset_sql = f"SELECT {', '.join(projection)} FROM {dataset.relation_sql()}"
    prediction_path = store.artifact_path(model_id, "predictions")
    init_path = store.artifact_path(model_id, "init_score")
    init_join_sql = (
        f"\nLEFT JOIN read_parquet({sql_literal(str(init_path))}) init_score USING (__lucidum_row_id)"
        if include_init_score and init_path.exists()
        else ""
    )
    init_select_sql = ", init_score.init_score AS __lucidum_init_score" if include_init_score and init_path.exists() else ""
    with dataset.lock:
        return dataset.con.execute(
            f"""
SELECT
  prediction.__lucidum_row_id,
  prediction.gbm_prediction{init_select_sql}{',' if columns else ''}
  {', '.join(f'dataset_rows.{quote_ident(name)}' for name in columns)}
FROM read_parquet({sql_literal(str(prediction_path))}) prediction
LEFT JOIN ({dataset_sql}) dataset_rows USING (__lucidum_row_id)
{init_join_sql}
ORDER BY prediction.__lucidum_row_id
"""
        ).fetchdf()


def _feature_levels_with_missing(
    frame: Any,
    fit_frame: Any,
    feature: str,
    kind: str,
    spec_row: dict[str, Any],
    base_value: Any,
    np: Any,
    pd: Any,
) -> tuple[list[dict[str, Any]], dict[str, Any], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    levels, meta, estimated, clipped, unseen = _feature_levels(
        frame,
        fit_frame,
        feature,
        kind,
        spec_row,
        base_value,
        {},
        np,
        pd,
    )
    if feature in frame.columns and bool(frame[feature].isna().any()):
        levels.append({"value": None, "status": "ok"})
    return levels, meta, estimated, clipped, unseen


def _gbm_key_series(series: Any, meta: dict[str, Any], np: Any, pd: Any) -> Any:
    keys = _scored_feature_series(series, meta, np, pd)
    return keys.astype("object").where(keys.notna(), GBM_MISSING_KEY)


def _gbm_component_from_table(frame: Any, table: Any, features: list[str], feature_meta: dict[str, dict[str, Any]], np: Any, pd: Any) -> Any:
    if table is None:
        return pd.Series(np.nan, index=frame.index, dtype=float)
    lookup = table.loc[table["status"].astype(str) == "ok", [*features, "tabulated_linear"]].copy()
    if not features or lookup.empty:
        return pd.Series(np.nan, index=frame.index, dtype=float)
    for feature in features:
        lookup[feature] = _gbm_key_series(lookup[feature], feature_meta[feature], np, pd)
    lookup["tabulated_linear"] = pd.to_numeric(lookup["tabulated_linear"], errors="coerce")
    if len(features) == 1:
        feature = features[0]
        key = _gbm_key_series(frame[feature], feature_meta[feature], np, pd)
        lookup_series = lookup.drop_duplicates(subset=[feature]).set_index(feature)["tabulated_linear"]
        return pd.to_numeric(key.map(lookup_series), errors="coerce")
    keys = pd.DataFrame(index=frame.index)
    for feature in features:
        keys[feature] = _gbm_key_series(frame[feature], feature_meta[feature], np, pd)
    merged = keys.merge(lookup.drop_duplicates(subset=features), on=features, how="left", sort=False)
    return pd.Series(pd.to_numeric(merged["tabulated_linear"], errors="coerce").to_numpy(), index=frame.index, dtype=float)


def _inverse_transform(np: Any, values: Any, transform: str) -> Any:
    if transform == "log":
        return np.exp(values)
    if transform == "logit":
        return 1.0 / (1.0 + np.exp(-values))
    return values


def _prediction_to_linear(np: Any, pd: Any, values: Any, transform: str) -> Any:
    prediction = pd.to_numeric(values, errors="coerce")
    if transform == "log":
        valid = prediction.notna() & np.isfinite(prediction.astype(float)) & (prediction.astype(float) > 0)
        result = pd.Series(np.nan, index=prediction.index, dtype=float)
        result.loc[valid] = np.log(prediction.loc[valid].astype(float))
        return result
    if transform == "logit":
        valid = prediction.notna() & np.isfinite(prediction.astype(float)) & (prediction.astype(float) > 0) & (prediction.astype(float) < 1)
        result = pd.Series(np.nan, index=prediction.index, dtype=float)
        result.loc[valid] = np.log(prediction.loc[valid].astype(float) / (1.0 - prediction.loc[valid].astype(float)))
        return result
    return prediction


def _linear_offset(frame: Any, manifest: dict[str, Any], parameters: dict[str, Any], np: Any, pd: Any) -> Any:
    objective = str(parameters.get("objective") or "").strip().lower()
    transform = init_score_transform(objective)
    init_score = manifest.get("init_score") if isinstance(manifest.get("init_score"), dict) else {}
    if str(init_score.get("kind") or "none").lower() != "none" and "__lucidum_init_score" in frame.columns:
        return pd.to_numeric(frame["__lucidum_init_score"], errors="coerce"), transform
    if uses_log_offset({"objective": objective}):
        offset_col = str(manifest.get("offset_column") or "").strip()
        if offset_col and offset_col in frame.columns:
            offset = pd.to_numeric(frame[offset_col], errors="coerce")
            valid = offset.notna() & np.isfinite(offset.astype(float)) & (offset.astype(float) > 0)
            result = pd.Series(np.nan, index=frame.index, dtype=float)
            result.loc[valid] = np.log(offset.loc[valid].astype(float))
            return result, transform
    return pd.Series(0.0, index=frame.index, dtype=float), transform


def build_gbm_tabulations(
    dataset: Dataset,
    store: GbmModelStore,
    model_id: str,
    feature_spec: Any,
    *,
    progress_callback: ProgressCallback | None = None,
) -> dict[str, Any]:
    progress = progress_callback or (lambda _progress: None)
    np, pd = gbm_tabulation_dependencies()
    store.validate_model_id(model_id)
    manifest = store.manifest(model_id)
    tree_rows = _read_tree_table(store, model_id, manifest.get("best_iteration"))
    if not tree_rows:
        return {
            "model_id": model_id,
            "model_kind": "gbm",
            "model_ref": f"gbm:{model_id}",
            "status": "not_tabulatable",
            "warnings": ["Rebuild this GBM before tabulating; tree_table.parquet is missing or empty."],
            "tables": [],
        }
    groups, blocking_warnings = _tree_groups(tree_rows)
    if blocking_warnings:
        payload = {
            "model_id": model_id,
            "model_kind": "gbm",
            "model_ref": f"gbm:{model_id}",
            "status": "not_tabulatable",
            "warnings": blocking_warnings,
            "tables": [],
            "diagnostics": {"blocking_warnings": blocking_warnings},
        }
        store.write_json(store.artifact_path(model_id, "tabulation_manifest"), payload)
        return payload

    non_base_groups = {features: rows for features, rows in groups.items() if features}
    all_features = sorted({feature for features in non_base_groups for feature in features}, key=lambda value: (value.lower(), value))
    source_columns, kinds = _schema_columns_and_kinds(dataset)
    offset_col = str(manifest.get("offset_column") or "").strip()
    init_score = manifest.get("init_score") if isinstance(manifest.get("init_score"), dict) else {}
    include_init_score = str(init_score.get("kind") or "none").lower() != "none"
    required_columns = [column for column in source_columns if column in {*all_features, offset_col}]
    frame = _tabulation_frame_from_predictions(dataset, store, model_id, required_columns, include_init_score=include_init_score)
    fit_frame = frame
    spec_rows = _feature_spec_map(feature_spec)
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
            {},
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
    feature_meta: dict[str, dict[str, Any]] = {}
    feature_levels: dict[str, list[dict[str, Any]]] = {}
    for feature in all_features:
        levels, meta, estimated, clipped, unseen = _feature_levels_with_missing(
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
        clipped_bounds.extend(clipped)
        unseen_levels.extend(unseen)
    for entry in estimated_specs:
        fields = ", ".join(entry["fields"])
        warnings.append(f"Estimated {fields} for numeric GBM tabulation feature {entry['feature']} from scored rows.")
    for entry in inferred_bases:
        warnings.append(f"Estimated base for GBM tabulation feature {entry['feature']} from scored rows: {entry['value']}.")
    if unseen_levels:
        by_feature: dict[str, int] = defaultdict(int)
        for entry in unseen_levels:
            by_feature[str(entry["feature"])] += 1
        warnings.extend(
            f"{count} dataset level{'s' if count != 1 else ''} for {feature} were not seen in training and tabulate using tree fallback paths."
            for feature, count in sorted(by_feature.items())
        )

    store.tabulations_dir(model_id).mkdir(parents=True, exist_ok=True)
    tables: list[dict[str, Any]] = []
    skipped_tables: list[dict[str, Any]] = []
    table_frames: dict[str, Any] = {}
    base_value = 0.0
    table_index = 1
    if () in groups:
        base_frame = pd.DataFrame([base])
        base_value += float(_evaluate_tree_group(base_frame, groups[()], np, pd)[0])
    for features, rows in sorted(non_base_groups.items(), key=lambda item: (len(item[0]), item[0])):
        table_id = "|".join(features)
        table_label = " × ".join(features)
        cell_count = 1
        for feature in features:
            cell_count *= max(1, len(feature_levels.get(feature, [])))
        progress({"phase": "tabulating", "message": f"Tabulating GBM {table_label}", "model_id": model_id, "table_id": table_id, "cells": cell_count})
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
            pred_frame = pd.DataFrame([base for _ in range(len(ok_grid))])
            for feature in features:
                pred_frame[feature] = ok_grid[feature].to_numpy()
            contribution = _evaluate_tree_group(pred_frame, rows, np, pd)
            base_grid = pd.DataFrame([{feature: base.get(feature) for feature in features}])
            base_frame = pd.DataFrame([base])
            for feature in features:
                base_frame[feature] = base_grid[feature].to_numpy()
            base_contribution = float(_evaluate_tree_group(base_frame, rows, np, pd)[0])
            base_value += base_contribution
            table.loc[ok_mask, "tabulated_linear"] = contribution - base_contribution
            table["base_adjustment"] = base_contribution
        else:
            table["base_adjustment"] = None
        table["table_id"] = table_id
        table["status"] = table["__status"]
        table = table.drop(columns=["__status"])
        _write_dataframe_parquet(table, _table_file_path(store, model_id, table_id))
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

    base_table = pd.DataFrame([{"table_id": "base", "status": "ok", "tabulated_linear": base_value, "base_adjustment": base_value}])
    _write_dataframe_parquet(base_table, _table_file_path(store, model_id, "base"))
    tables.insert(0, {"table_id": "base", "label": "base", "index": 0, "features": [], "cell_count": 1, "skipped": False, "path": "tabulations/base.parquet", "min": base_value, "max": base_value})

    progress({"phase": "scoring", "message": f"Scoring tabulated GBM {model_id}", "model_id": model_id})
    tabulated = frame[["__lucidum_row_id"]].copy()
    tree_eta = pd.Series(base_value, index=frame.index, dtype=float)
    missing = pd.Series(False, index=frame.index, dtype=bool)
    for table_info in tables:
        if table_info["table_id"] == "base" or table_info.get("skipped"):
            continue
        features = list(table_info.get("features") or [])
        component = _gbm_component_from_table(frame, table_frames.get(str(table_info["table_id"])), features, feature_meta, np, pd)
        missing = missing | component.isna()
        tree_eta = tree_eta + component.fillna(0.0)
        tabulated[f"tabulated_linear__{_safe_id(str(table_info['table_id']))}"] = component

    parameters = store.model_parameters(model_id)
    linear_offset, transform = _linear_offset(frame, manifest, parameters, np, pd)
    final_linear = tree_eta + linear_offset.fillna(0.0)
    missing = missing | linear_offset.isna()
    finite_linear = (~missing) & np.isfinite(final_linear.astype(float))
    prediction = pd.Series(np.nan, index=frame.index, dtype=float)
    if bool(finite_linear.any()):
        prediction.loc[finite_linear] = pd.to_numeric(
            _inverse_transform(np, final_linear.loc[finite_linear].to_numpy(dtype=float), transform),
            errors="coerce",
        )
    tabulated["gbm_tabulated_prediction"] = prediction
    tabulated["gbm_tabulated_linear_prediction"] = final_linear.where(~missing, np.nan)
    tabulated["gbm_tabulation_missing"] = missing

    exact_linear = _prediction_to_linear(np, pd, frame["gbm_prediction"], transform)
    error = exact_linear - tabulated["gbm_tabulated_linear_prediction"]
    finite_error = error.dropna()
    mean_linear_error = json_safe_number(float(finite_error.mean())) if len(finite_error) else None
    linear_sd_error = json_safe_number(float(finite_error.std())) if len(finite_error) > 1 else 0.0
    _write_dataframe_parquet(tabulated, store.artifact_path(model_id, "tabulated_predictions"))
    diagnostics = {
        "mean_linear_error": mean_linear_error,
        "linear_sd_error": linear_sd_error,
        "scored_rows": int(finite_linear.sum()),
        "tabulated_row_count": int(len(tabulated)),
        "missing_tabulated_prediction_rows": int(missing.sum()),
        "estimated_spec_fields": estimated_specs,
        "estimated_base_fields": inferred_bases,
        "clipped_spec_fields": clipped_bounds,
        "unseen_levels": unseen_levels,
        "skipped_oversized_tables": skipped_tables,
        "max_tree_leaves": MAX_GBM_TABULATION_LEAVES,
        "max_tree_features": MAX_GBM_TABULATION_FEATURES,
    }
    manifest_payload = {
        "model_id": model_id,
        "model_kind": "gbm",
        "model_ref": f"gbm:{model_id}",
        "status": "tabulated",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "max_cells": MAX_TABULATION_CELLS,
        "max_tree_leaves": MAX_GBM_TABULATION_LEAVES,
        "max_tree_features": MAX_GBM_TABULATION_FEATURES,
        "tables": tables,
        "feature_meta": feature_meta,
        "warnings": warnings,
        "diagnostics": diagnostics,
    }
    store.write_json(store.artifact_path(model_id, "tabulation_manifest"), manifest_payload)
    return manifest_payload


def tabulation_model_status(store: GbmModelStore, model: dict[str, Any]) -> dict[str, Any]:
    model_id = str(model.get("model_id") or "")
    manifest = _tabulation_manifest(store, model_id)
    tree_table_path = store.artifact_path(model_id, "tree_table")
    tabulatable = tree_table_path.exists()
    tables = list(manifest.get("tables", [])) if manifest and manifest.get("status") == "tabulated" else []
    model_warnings = list(manifest.get("warnings", [])) if manifest else []
    if not tabulatable:
        model_warnings.append("Rebuild this GBM before tabulating; tree_table.parquet is missing.")
    blocking_warnings: list[str] = []
    if tabulatable and not tables:
        _, blocking_warnings = _tree_groups(_read_tree_table(store, model_id, model.get("best_iteration")))
        if blocking_warnings:
            tabulatable = False
            model_warnings.extend(blocking_warnings)
    if manifest and manifest.get("status") == "not_tabulatable":
        tabulatable = False
        blocking_warnings = list(manifest.get("diagnostics", {}).get("blocking_warnings", [])) or blocking_warnings
    diagnostics = dict(manifest.get("diagnostics", {}) if manifest else {})
    if blocking_warnings:
        diagnostics["blocking_warnings"] = blocking_warnings
    return {
        "model_id": model_id,
        "model_ref": f"gbm:{model_id}",
        "model_kind": "gbm",
        "label": model.get("label") or model_id,
        "active": bool(model.get("active")),
        "tabulatable": tabulatable,
        "tabulated": bool(tables),
        "tables": tables,
        "warnings": model_warnings,
        "diagnostics": diagnostics,
    }


def read_table(store: GbmModelStore, model_id: str, table_id: str) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
    manifest = _tabulation_manifest(store, model_id)
    if not manifest or manifest.get("status") != "tabulated":
        return None, []
    table_info = next((table for table in manifest.get("tables", []) if str(table.get("table_id") or "") == table_id), None)
    if not table_info or table_info.get("skipped"):
        return table_info, []
    return table_info, store.read_parquet_records(_table_file_path(store, model_id, table_id))


__all__ = [
    "MAX_GBM_TABULATION_FEATURES",
    "MAX_GBM_TABULATION_LEAVES",
    "build_gbm_tabulations",
    "read_table",
    "tabulation_model_status",
]
