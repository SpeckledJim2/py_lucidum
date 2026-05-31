from __future__ import annotations

import json
import math
import re
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from uuid import uuid4

import duckdb

from py_lucidum.core import Dataset, quote_ident, sql_literal

from .sample import GENERATED_SAMPLE_FILENAME
from .validation import DEFAULT_TRAINING_MODE


ARTIFACT_FILES = {
    "predictions": "predictions.parquet",
    "shap_long": "shap_values.parquet",
    "shap_summary": "shap_summary.parquet",
    "evaluation": "evaluation.parquet",
    "tree_table": "tree_table.parquet",
    "manifest": "manifest.json",
    "feature_config": "feature_config.json",
    "parameters": "parameters.json",
    "training_log": "training_log.json",
    "model": "model.txt",
}

SOURCE_KINDS = {
    "predictions": {
        "file": ARTIFACT_FILES["predictions"],
        "label": "Predictions",
        "kind": "gbm_predictions",
    },
    "shap_long": {
        "file": ARTIFACT_FILES["shap_long"],
        "label": "SHAP values",
        "kind": "gbm_shap_long",
    },
    "shap_summary": {
        "file": ARTIFACT_FILES["shap_summary"],
        "label": "SHAP summary",
        "kind": "gbm_shap_summary",
    },
}

MODEL_ID_RE = re.compile(r"[A-Za-z0-9_.-]+")
SOURCE_RE = re.compile(r"^gbm:([A-Za-z0-9_.-]+):(predictions|shap_long|shap_summary)$")


class GbmModelNameError(ValueError):
    pass


def json_safe_number(value: Any) -> float | None:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def metric_value_at_iteration(values: Any, best_iteration: Any) -> float | None:
    try:
        index = int(best_iteration) - 1
    except (TypeError, ValueError):
        return None
    if index < 0 or not isinstance(values, list) or index >= len(values):
        return None
    return json_safe_number(values[index])


def evaluation_metric_for_dataset(
    evaluation: Any,
    metric_name: str,
    best_iteration: Any,
    dataset_aliases: tuple[str, ...],
) -> float | None:
    if not isinstance(evaluation, dict):
        return None
    aliases = {alias.lower() for alias in dataset_aliases}
    metric = str(metric_name or "")
    for dataset_name, metrics in evaluation.items():
        if str(dataset_name or "").lower() not in aliases or not isinstance(metrics, dict):
            continue
        value = metric_value_at_iteration(metrics.get(metric), best_iteration)
        if value is not None:
            return value
    return None


def best_metrics_from_evaluation(evaluation: Any, metric_name: str, best_iteration: Any) -> dict[str, float | None]:
    return {
        "training": evaluation_metric_for_dataset(evaluation, metric_name, best_iteration, ("training", "train")),
        "test": evaluation_metric_for_dataset(evaluation, metric_name, best_iteration, ("test",)),
    }


def normalise_best_metrics(value: Any) -> dict[str, float | None]:
    metrics = value if isinstance(value, dict) else {}
    return {
        "training": json_safe_number(metrics.get("training")),
        "test": json_safe_number(metrics.get("test")),
    }


def dedupe_columns(columns: list[str]) -> list[str]:
    projected: list[str] = []
    seen: set[str] = set()
    for column in columns:
        name = str(column or "").strip()
        if not name or name in seen or name == "__lucidum_row_id":
            continue
        projected.append(name)
        seen.add(name)
    return projected


def prediction_source_select_sql(source_columns: list[str]) -> str:
    return ",\n  ".join(f"base.{quote_ident(name)}" for name in source_columns)


def row_number_source_projection_sql(source_columns: list[str]) -> str:
    columns_sql = ",\n    ".join(quote_ident(name) for name in source_columns)
    suffix = f",\n    {columns_sql}" if columns_sql else ""
    return f"ROW_NUMBER() OVER () AS __lucidum_row_id{suffix}"


@dataclass(frozen=True)
class GbmSourceRef:
    model_id: str
    source_kind: str


class GbmModelStore:
    def __init__(self, dataset_path: str | Path):
        self.dataset_path = Path(dataset_path).expanduser().resolve()
        self.root = self.dataset_path.parent / ".lucidum" / "models" / "gbm"

    @property
    def active_path(self) -> Path:
        return self.root / "active_model.json"

    @property
    def generated_sample_path(self) -> Path:
        return self.root / GENERATED_SAMPLE_FILENAME

    def ensure_root(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)

    def create_model_id(self, label: str | None = None) -> str:
        cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "-", str(label or "gbm").strip().lower()).strip("-")
        prefix = cleaned or "gbm"
        timestamp = time.strftime("%H%M%S")
        return f"{prefix}-{timestamp}-{uuid4().hex[:8]}"

    def validate_model_id(self, model_id: str, *, for_new_name: bool = False) -> str:
        text = str(model_id or "").strip()
        if not text or text in {".", ".."} or not MODEL_ID_RE.fullmatch(text):
            message = "Choose a valid GBM model name" if for_new_name else "Choose a valid GBM model id"
            raise GbmModelNameError(f"{message}: letters, numbers, dots, underscores, and hyphens only")
        return text

    def model_dir(self, model_id: str) -> Path:
        return self.root / self.validate_model_id(model_id)

    def create_model_dir(self, model_id: str) -> Path:
        self.ensure_root()
        path = self.model_dir(model_id)
        path.mkdir(parents=True, exist_ok=False)
        return path

    def artifact_path(self, model_id: str, artifact: str) -> Path:
        filename = ARTIFACT_FILES[artifact]
        return self.model_dir(model_id) / filename

    def write_json(self, path: Path, payload: Any) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temp = path.with_suffix(path.suffix + ".tmp")
        temp.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        temp.replace(path)

    def read_json(self, path: Path, default: Any = None) -> Any:
        if not path.exists():
            return default
        return json.loads(path.read_text(encoding="utf-8"))

    def manifest(self, model_id: str) -> dict[str, Any]:
        path = self.artifact_path(model_id, "manifest")
        manifest = self.read_json(path)
        if not isinstance(manifest, dict):
            raise ValueError("Choose a valid GBM model")
        manifest.setdefault("training_mode", DEFAULT_TRAINING_MODE)
        return manifest

    def list_models(self) -> list[dict[str, Any]]:
        if not self.root.exists():
            return []
        active = self.active_model_id()
        models: list[dict[str, Any]] = []
        for path in self.root.iterdir():
            manifest_path = path / ARTIFACT_FILES["manifest"]
            if not path.is_dir() or not manifest_path.exists():
                continue
            manifest = self.read_json(manifest_path, {})
            if isinstance(manifest, dict):
                models.append(self.model_list_item(path, manifest, active))
        return sorted(models, key=lambda item: str(item.get("created_at", "")), reverse=True)

    def model_list_item(self, path: Path, manifest: dict[str, Any], active_model_id: str | None) -> dict[str, Any]:
        item = dict(manifest)
        item.setdefault("training_mode", DEFAULT_TRAINING_MODE)
        model_id = str(item.get("model_id") or path.name)
        item["model_id"] = model_id
        item["parameters"] = self.model_parameters(model_id)
        item["best_metrics"] = self.model_best_metrics(model_id, item)
        item["active"] = model_id == active_model_id
        return item

    def model_parameters(self, model_id: str) -> dict[str, Any]:
        parameters = self.read_json(self.artifact_path(model_id, "parameters"), {})
        return dict(parameters) if isinstance(parameters, dict) else {}

    def model_best_metrics(self, model_id: str, manifest: dict[str, Any]) -> dict[str, float | None]:
        metrics = normalise_best_metrics(manifest.get("best_metrics"))
        if metrics["training"] is not None and metrics["test"] is not None:
            return metrics
        training_log = self.read_json(self.artifact_path(model_id, "training_log"), {})
        evaluation = training_log.get("evaluation") if isinstance(training_log, dict) else None
        derived = best_metrics_from_evaluation(evaluation, str(manifest.get("metric") or ""), manifest.get("best_iteration"))
        return {
            "training": metrics["training"] if metrics["training"] is not None else derived["training"],
            "test": metrics["test"] if metrics["test"] is not None else derived["test"],
        }

    def active_model_id(self) -> str | None:
        payload = self.read_json(self.active_path, {})
        if not isinstance(payload, dict):
            return None
        model_id = payload.get("model_id")
        return str(model_id) if model_id else None

    def clear_active_model(self) -> None:
        if self.active_path.exists():
            self.active_path.unlink()

    def activate_model(self, model_id: str) -> dict[str, Any]:
        manifest = self.manifest(model_id)
        self.ensure_root()
        self.write_json(self.active_path, {"model_id": model_id, "activated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})
        manifest = dict(manifest)
        manifest["active"] = True
        return manifest

    def rename_model(self, model_id: str, new_model_id: str) -> dict[str, Any]:
        new_id = self.validate_model_id(new_model_id, for_new_name=True)
        old_id = self.validate_model_id(model_id)
        manifest = self.manifest(old_id)
        if new_id == old_id:
            return self.activate_model(old_id) if self.active_model_id() == old_id else dict(manifest)
        source = self.model_dir(old_id)
        target = self.model_dir(new_id)
        if target.exists():
            raise GbmModelNameError(f"GBM model already exists: {new_id}")
        source.rename(target)
        manifest = self._renamed_manifest(manifest, old_id, new_id)
        self.write_json(self.artifact_path(new_id, "manifest"), manifest)
        self._rewrite_shap_summary_model_id(new_id)
        if self.active_model_id() == old_id:
            self.activate_model(new_id)
            manifest["active"] = True
        return manifest

    def delete_model(self, model_id: str) -> dict[str, Any]:
        deleted_id = self.validate_model_id(model_id)
        manifest = self.manifest(deleted_id)
        was_active = self.active_model_id() == deleted_id
        shutil.rmtree(self.model_dir(deleted_id))
        if was_active:
            remaining = self.list_models()
            next_id = str(remaining[0].get("model_id") or "") if remaining else ""
            if next_id:
                self.activate_model(next_id)
            else:
                self.clear_active_model()
        return manifest

    def _renamed_manifest(self, manifest: dict[str, Any], old_id: str, new_id: str) -> dict[str, Any]:
        renamed = dict(manifest)
        renamed["model_id"] = new_id
        renamed["label"] = new_id
        sources = renamed.get("sources")
        if isinstance(sources, dict):
            renamed["sources"] = {
                str(kind): self._renamed_source_id(value, old_id, new_id)
                for kind, value in sources.items()
            }
        return renamed

    def _renamed_source_id(self, value: Any, old_id: str, new_id: str) -> Any:
        if not isinstance(value, str):
            return value
        match = SOURCE_RE.match(value)
        if not match or match.group(1) != old_id:
            return value
        return self.source_id(new_id, match.group(2))

    def _rewrite_shap_summary_model_id(self, model_id: str) -> None:
        path = self.artifact_path(model_id, "shap_summary")
        if not path.exists():
            return
        con = duckdb.connect(database=":memory:")
        try:
            columns = con.execute(f"DESCRIBE SELECT * FROM read_parquet({sql_literal(str(path))})").fetchall()
            if "gbm_model_id" not in {str(row[0]) for row in columns}:
                return
            temp = path.with_suffix(path.suffix + ".tmp")
            con.execute(
                f"""
COPY (
  SELECT * REPLACE ({sql_literal(model_id)} AS gbm_model_id)
  FROM read_parquet({sql_literal(str(path))})
) TO {sql_literal(str(temp))} (FORMAT PARQUET)
"""
            )
            temp.replace(path)
        finally:
            con.close()

    def source_ref(self, source_id: str) -> GbmSourceRef | None:
        match = SOURCE_RE.match(source_id)
        if not match:
            return None
        model_id, source_kind = match.groups()
        try:
            path = self.source_path(model_id, source_kind)
        except ValueError:
            return None
        if not path.exists():
            return None
        return GbmSourceRef(model_id=model_id, source_kind=source_kind)

    def source_path(self, model_id: str, source_kind: str) -> Path:
        return self.model_dir(model_id) / str(SOURCE_KINDS[source_kind]["file"])

    def source_id(self, model_id: str, source_kind: str) -> str:
        return f"gbm:{model_id}:{source_kind}"

    def relation_sql(self, source_id: str) -> str:
        ref = self.source_ref(source_id)
        if not ref:
            raise ValueError("Choose a valid data source")
        source_path = self.source_path(ref.model_id, ref.source_kind)
        if ref.source_kind != "predictions":
            return f"read_parquet({sql_literal(str(source_path))})"
        manifest = self.manifest(ref.model_id)
        offset_col = str(manifest.get("offset_column") or "").strip()
        where_sql = f"\nWHERE TRY_CAST({quote_ident(offset_col)} AS DOUBLE) > 0" if offset_col else ""
        source_columns = self.source_columns(manifest)
        select_sql = prediction_source_select_sql(source_columns)
        base_projection_sql = row_number_source_projection_sql(source_columns)
        return f"""(
SELECT
  {select_sql}{',' if select_sql else ''}
  prediction.gbm_prediction
FROM (
  SELECT
    *
  FROM (
    SELECT
      {base_projection_sql}
    FROM {self.dataset_relation_sql()}
  ) dataset_rows
  {where_sql}
) base
INNER JOIN read_parquet({sql_literal(str(source_path))}) prediction USING (__lucidum_row_id)
)"""

    def source_columns(self, manifest: dict[str, Any]) -> list[str]:
        raw_columns = manifest.get("source_columns")
        if isinstance(raw_columns, list):
            columns = [str(name).strip() for name in raw_columns if str(name or "").strip()]
            if columns:
                return dedupe_columns(columns)
        dataset = Dataset(self.dataset_path)
        return list(dataset.column_map())

    def dataset_relation_sql(self) -> str:
        path = sql_literal(str(self.dataset_path))
        suffix = self.dataset_path.suffix.lower()
        if suffix == ".parquet":
            return f"read_parquet({path})"
        if suffix == ".csv":
            return f"read_csv_auto({path}, header=true, ignore_errors=true)"
        raise ValueError("Only .csv and .parquet files are supported in this prototype")

    def model_detail(self, model_id: str) -> dict[str, Any]:
        manifest = self.manifest(model_id)
        detail = {
            "manifest": manifest,
            "features": self.read_json(self.artifact_path(model_id, "feature_config"), []),
            "parameters": self.read_json(self.artifact_path(model_id, "parameters"), {}),
            "training_log": self.read_json(self.artifact_path(model_id, "training_log"), {}),
            "active": self.active_model_id() == model_id,
        }
        return detail

    def source_manifest_entries(self) -> list[tuple[dict[str, Any], str, dict[str, str]]]:
        entries: list[tuple[dict[str, Any], str, dict[str, str]]] = []
        for model in self.list_models():
            model_id = str(model.get("model_id") or "")
            if not model_id:
                continue
            for source_kind, info in SOURCE_KINDS.items():
                if self.source_path(model_id, source_kind).exists():
                    entries.append((model, self.source_id(model_id, source_kind), info))
        return entries


class GbmSourceProvider:
    def __init__(self, store: GbmModelStore):
        self.store = store

    def has_source(self, source_id: str) -> bool:
        return self.store.source_ref(source_id) is not None

    def relation_sql(self, source_id: str) -> str:
        return self.store.relation_sql(source_id)

    def data_sources(self, dataset: Dataset) -> list[dict[str, Any]]:
        sources: list[dict[str, Any]] = []
        for model, source_id, info in self.store.source_manifest_entries():
            schema = dataset.schema_for_source(source_id)
            label = f"{model.get('label') or model.get('model_id')} - {info['label']}"
            sources.append(
                {
                    "id": source_id,
                    "label": label,
                    "kind": info["kind"],
                    "model_id": model.get("model_id"),
                    "active": bool(model.get("active")),
                    "response_column": model.get("response_column"),
                    "offset_column": model.get("offset_column"),
                    "created_at": model.get("created_at"),
                    "objective": model.get("objective"),
                    "metric": model.get("metric"),
                    "training_mode": model.get("training_mode", DEFAULT_TRAINING_MODE),
                    "best_iteration": model.get("best_iteration"),
                    "best_metrics": model.get("best_metrics"),
                    "row_count": schema["row_count"],
                    "columns": schema["columns"],
                }
            )
        return sources


__all__ = [
    "ARTIFACT_FILES",
    "GbmModelNameError",
    "GbmModelStore",
    "GbmSourceProvider",
    "GbmSourceRef",
    "SOURCE_KINDS",
]
