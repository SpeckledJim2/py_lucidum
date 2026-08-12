"""Small public helpers for using Lucidum's GLM tabulation artifacts."""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

from py_lucidum.core import Dataset, load_features
from py_lucidum.tools.glm.store import GlmModelStore


def build_glm_tabulations(
    dataset_path: str | Path,
    *,
    model_id: str,
    feature_spec_path: str | Path,
) -> dict[str, Any]:
    """Build GLM rating tables and score the source rows from those tables."""

    from py_lucidum.tools.glm.tabulation import build_tabulations

    dataset, store = _dataset_and_store(dataset_path)
    try:
        result = build_tabulations(
            dataset,
            store,
            {"model_refs": [f"glm:{model_id}"]},
            load_features(feature_spec_path),
        )
        built = next(
            (item for item in result.get("models", []) if item.get("model_id") == model_id),
            None,
        )
        if not isinstance(built, dict) or built.get("status") != "tabulated":
            warnings = list((built or {}).get("warnings") or [])
            raise ValueError(warnings[0] if warnings else f"Could not tabulate GLM model {model_id}")
        return _tabulation_result(store, model_id, built)
    finally:
        dataset.con.close()


def score_glm_tabulations(
    dataset_path: str | Path,
    *,
    model_id: str,
) -> dict[str, Any]:
    """Re-score source rows from existing GLM rating tables."""

    from py_lucidum.tools.glm.tabulation import score_tabulations

    dataset, store = _dataset_and_store(dataset_path)
    try:
        manifest = score_tabulations(dataset, store, model_id)
        return _tabulation_result(store, model_id, manifest)
    finally:
        dataset.con.close()


def export_glm_tabulations(
    dataset_path: str | Path,
    *,
    model_id: str,
    scale: str = "auto",
) -> dict[str, Any]:
    """Export GLM rating tables to XLSX and return its index worksheet values."""

    from py_lucidum.tools.glm.tabulation import export_tabulations

    dataset, store = _dataset_and_store(dataset_path)
    try:
        resolved_scale = _resolved_export_scale(store, model_id, scale)
        result = export_tabulations(
            store,
            {"model_refs": [f"glm:{model_id}"], "scale": resolved_scale},
        )
        return {**result, "path": Path(result["path"]).resolve()}
    finally:
        dataset.con.close()


def _dataset_and_store(dataset_path: str | Path) -> tuple[Dataset, GlmModelStore]:
    dataset = Dataset(dataset_path)
    return dataset, GlmModelStore(dataset.path, dataset=dataset)


def _tabulation_result(
    store: GlmModelStore,
    model_id: str,
    manifest: dict[str, Any],
) -> dict[str, Any]:
    return {
        "model_id": model_id,
        "model_folder": store.model_dir(model_id).resolve(),
        "scoring_path": store.artifact_path(model_id, "tabulated_predictions").resolve(),
        "manifest": manifest,
        "diagnostics": dict(manifest.get("diagnostics") or {}),
        "warnings": list(manifest.get("warnings") or []),
    }


def _resolved_export_scale(store: GlmModelStore, model_id: str, scale: str) -> str:
    requested = str(scale or "auto").strip().lower()
    if requested in {"exp", "linear"}:
        return requested
    if requested != "auto":
        raise ValueError("scale must be 'auto', 'exp', or 'linear'")

    estimator_path = store.artifact_path(model_id, "estimator")
    if not estimator_path.is_file():
        raise ValueError(f"GLM estimator.pkl is unavailable for model {model_id}")
    with estimator_path.open("rb") as handle:
        estimator = pickle.load(handle)
    link = getattr(estimator, "link_instance", None)
    return "exp" if type(link).__name__ == "LogLink" else "linear"


__all__ = [
    "build_glm_tabulations",
    "export_glm_tabulations",
    "score_glm_tabulations",
]
