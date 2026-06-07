from __future__ import annotations

from pathlib import Path
from typing import Any

from py_lucidum.core import Dataset, json_number


GLM_IMPORTANCE_METRIC = "weighted_mean_abs_centered_linear_predictor_contribution"
GLM_IMPORTANCE_LABEL = "GLM eta MAD"


def feature_importance_payload(dataset: Dataset, *, gbm_store: Any = None, glm_store: Any = None) -> dict[str, Any]:
    with dataset.lock:
        dataset_features = [
            {
                "name": column.name,
                "kind": column.kind,
                "duckdb_type": column.duckdb_type,
            }
            for column in dataset.valid_schema_columns()
        ]
    gbm = active_gbm_importance(gbm_store)
    glm = active_glm_importance(glm_store)
    messages = [message for message in [gbm.get("message"), glm.get("message")] if message]
    return {
        "models": {
            "gbm": gbm,
            "glm": glm,
        },
        "dataset_features": dataset_features,
        "messages": messages,
        "has_importance": bool(gbm.get("rows") or glm.get("rows")),
    }


def active_gbm_importance(store: Any) -> dict[str, Any]:
    if store is None:
        return empty_model_payload("GBM tool is not enabled.")
    model_id = safe_active_model_id(store)
    if not model_id:
        return empty_model_payload("No active GBM is available.")
    try:
        manifest = store.manifest(model_id)
    except Exception:
        return empty_model_payload("No active GBM is available.", model_id=model_id)

    feature_rows = read_json_artifact(store, model_id, "feature_config", [])
    if not isinstance(feature_rows, list) or not feature_rows:
        feature_rows = manifest.get("feature_importance", [])
    if not isinstance(feature_rows, list):
        feature_rows = []

    shap_importance = read_gbm_shap_importance(store, model_id)
    has_saved_shap = bool(shap_importance) or any(json_number(row.get("mean_abs_shap")) is not None for row in feature_rows if isinstance(row, dict))
    metric = "mean_abs_shap" if has_saved_shap else "gain"
    metric_label = "SHAP" if has_saved_shap else "Gain"

    rows: list[dict[str, Any]] = []
    for row in feature_rows:
        if not isinstance(row, dict):
            continue
        feature = str(row.get("name") or row.get("feature") or "").strip()
        if not feature:
            continue
        if metric == "mean_abs_shap":
            importance = shap_importance.get(feature)
            if importance is None:
                importance = json_number(row.get("mean_abs_shap"))
            if importance is None:
                importance = 0.0
        else:
            importance = json_number(row.get("gain"))
            if importance is None:
                importance = 0.0
        rows.append(
            {
                "feature": feature,
                "importance": float(importance),
                "kind": str(row.get("kind") or ""),
            }
        )

    rows = ranked_rows(rows)
    if not rows:
        return model_payload(
            model_id=model_id,
            label=str(manifest.get("label") or model_id),
            metric=metric,
            metric_label=metric_label,
            rows=[],
            message="Active GBM has no saved feature importances.",
        )
    return model_payload(
        model_id=model_id,
        label=str(manifest.get("label") or model_id),
        metric=metric,
        metric_label=metric_label,
        rows=rows,
    )


def active_glm_importance(store: Any) -> dict[str, Any]:
    if store is None:
        return empty_model_payload("GLM tool is not enabled.")
    model_id = safe_active_model_id(store)
    if not model_id:
        return empty_model_payload("No active GLM is available.")
    try:
        manifest = store.manifest(model_id)
    except Exception:
        return empty_model_payload("No active GLM is available.", model_id=model_id)

    rows = read_glm_importance_rows(store, model_id, manifest)
    metric_payload = manifest.get("feature_importance_metric") if isinstance(manifest.get("feature_importance_metric"), dict) else {}
    metric = str(metric_payload.get("name") or GLM_IMPORTANCE_METRIC)
    metric_label = str(metric_payload.get("label") or GLM_IMPORTANCE_LABEL)
    if not rows:
        return model_payload(
            model_id=model_id,
            label=str(manifest.get("label") or model_id),
            metric=metric,
            metric_label=metric_label,
            rows=[],
            message="Rebuild the active GLM to calculate feature importances.",
        )
    return model_payload(
        model_id=model_id,
        label=str(manifest.get("label") or model_id),
        metric=metric,
        metric_label=metric_label,
        rows=rows,
    )


def read_glm_importance_rows(store: Any, model_id: str, manifest: dict[str, Any]) -> list[dict[str, Any]]:
    artifact_rows: list[dict[str, Any]] = []
    try:
        path = store.artifact_path(model_id, "feature_importance")
        if isinstance(path, Path) and path.exists():
            artifact_rows = store.read_parquet_records(path)
    except Exception:
        artifact_rows = []
    raw_rows = artifact_rows or manifest.get("feature_importance", [])
    if not isinstance(raw_rows, list):
        return []
    rows: list[dict[str, Any]] = []
    for row in raw_rows:
        if not isinstance(row, dict):
            continue
        feature = str(row.get("feature") or "").strip()
        importance = json_number(row.get("importance"))
        if not feature or importance is None:
            continue
        rows.append(
            {
                "feature": feature,
                "importance": float(importance),
                "term_count": int(row.get("term_count") or 0),
            }
        )
    return ranked_rows(rows)


def read_gbm_shap_importance(store: Any, model_id: str) -> dict[str, float]:
    try:
        path = store.artifact_path(model_id, "shap_summary")
        if not isinstance(path, Path) or not path.exists():
            return {}
        records = store.read_parquet_records(path)
    except Exception:
        return {}
    values: dict[str, float] = {}
    for row in records:
        feature = str(row.get("feature") or "").strip()
        value = json_number(row.get("mean_abs_shap"))
        if feature and value is not None:
            values[feature] = float(value)
    return values


def read_json_artifact(store: Any, model_id: str, artifact: str, default: Any) -> Any:
    try:
        return store.read_json(store.artifact_path(model_id, artifact), default)
    except Exception:
        return default


def safe_active_model_id(store: Any) -> str:
    try:
        return str(store.active_model_id() or "")
    except Exception:
        return ""


def ranked_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    ranked = sorted(rows, key=lambda row: (-float(row.get("importance") or 0.0), str(row.get("feature") or "").lower()))
    return [{**row, "rank": index + 1} for index, row in enumerate(ranked)]


def empty_model_payload(message: str, *, model_id: str = "") -> dict[str, Any]:
    return model_payload(model_id=model_id, label="", metric="", metric_label="", rows=[], message=message)


def model_payload(
    *,
    model_id: str,
    label: str,
    metric: str,
    metric_label: str,
    rows: list[dict[str, Any]],
    message: str = "",
) -> dict[str, Any]:
    return {
        "model_id": model_id,
        "label": label,
        "metric": metric,
        "metric_label": metric_label,
        "rows": rows,
        "message": message,
        "available": bool(rows),
    }


__all__ = ["feature_importance_payload"]
