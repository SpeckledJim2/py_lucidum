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

from py_lucidum.core import Dataset, ModelPredictionSource, quote_ident, sql_literal

from .validation import denominator_valid_sql, dataset_relation_sql


ARTIFACT_FILES = {
    "manifest": "manifest.json",
    "estimator": "estimator.pkl",
    "formula": "formula.txt",
    "coefficients": "coefficients.parquet",
    "feature_importance": "feature_importance.parquet",
    "predictions": "predictions.parquet",
    "diagnostics": "diagnostics.json",
    "tabulation_manifest": "tabulation_manifest.json",
    "tabulated_predictions": "tabulated_predictions.parquet",
}
SOURCE_KINDS = {
    "predictions": {
        "file": ARTIFACT_FILES["predictions"],
        "label": "Predictions",
        "kind": "glm_predictions",
    }
}
MODEL_ID_RE = re.compile(r"[A-Za-z0-9_.-]+")
SOURCE_RE = re.compile(r"^glm:([A-Za-z0-9_.-]+):predictions$")


class GlmModelNameError(ValueError):
    pass


@dataclass(frozen=True)
class GlmSourceRef:
    model_id: str
    source_kind: str


def json_safe_number(value: Any) -> float | None:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


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


def row_number_source_projection_sql(source_columns: list[str]) -> str:
    columns_sql = ",\n    ".join(quote_ident(name) for name in source_columns)
    suffix = f",\n    {columns_sql}" if columns_sql else ""
    return f"ROW_NUMBER() OVER () AS __lucidum_row_id{suffix}"


def prediction_source_select_sql(source_columns: list[str], *, include_tabulated: bool = False) -> str:
    parts = [f"base.{quote_ident(name)}" for name in source_columns]
    parts.append("prediction.glm_prediction")
    if include_tabulated:
        parts.append("tabulated.glm_tabulated_prediction")
    return ",\n  ".join(parts)


class GlmModelStore:
    def __init__(self, dataset_path: str | Path):
        self.dataset_path = Path(dataset_path).expanduser().resolve()
        self.root = self.dataset_path.parent / ".lucidum" / "models" / "glm"

    @property
    def active_path(self) -> Path:
        return self.root / "active_model.json"

    def ensure_root(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)

    def create_model_id(self, label: str | None = None) -> str:
        cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "-", str(label or "glm").strip().lower()).strip("-")
        prefix = cleaned or "glm"
        timestamp = time.strftime("%H%M%S")
        return f"{prefix}-{timestamp}-{uuid4().hex[:8]}"

    def validate_model_id(self, model_id: str, *, for_new_name: bool = False) -> str:
        text = str(model_id or "").strip()
        if not text or text in {".", ".."} or not MODEL_ID_RE.fullmatch(text):
            noun = "name" if for_new_name else "id"
            raise GlmModelNameError(f"Choose a valid GLM model {noun}: letters, numbers, dots, underscores, and hyphens only")
        return text

    def model_dir(self, model_id: str) -> Path:
        return self.root / self.validate_model_id(model_id)

    def create_model_dir(self, model_id: str) -> Path:
        self.ensure_root()
        path = self.model_dir(model_id)
        path.mkdir(parents=True, exist_ok=False)
        return path

    def artifact_path(self, model_id: str, artifact: str) -> Path:
        return self.model_dir(model_id) / ARTIFACT_FILES[artifact]

    def tabulations_dir(self, model_id: str) -> Path:
        return self.model_dir(model_id) / "tabulations"

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
            raise ValueError("Choose a valid GLM model")
        return manifest

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
        active = dict(manifest)
        active["active"] = True
        return active

    def rename_model(self, model_id: str, new_model_id: str) -> dict[str, Any]:
        old_id = self.validate_model_id(model_id)
        new_id = self.validate_model_id(new_model_id, for_new_name=True)
        manifest = self.manifest(old_id)
        if new_id == old_id:
            return self.activate_model(old_id) if self.active_model_id() == old_id else dict(manifest)
        source = self.model_dir(old_id)
        target = self.model_dir(new_id)
        if target.exists():
            raise GlmModelNameError(f"GLM model already exists: {new_id}")
        source.rename(target)
        renamed = self._renamed_manifest(manifest, old_id, new_id)
        self.write_json(self.artifact_path(new_id, "manifest"), renamed)
        if self.active_model_id() == old_id:
            self.activate_model(new_id)
            renamed["active"] = True
        return renamed

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
        return self.source_id(new_id, "predictions")

    def source_ref(self, source_id: str) -> GlmSourceRef | None:
        match = SOURCE_RE.match(source_id)
        if not match:
            return None
        model_id = match.group(1)
        try:
            path = self.source_path(model_id, "predictions")
        except ValueError:
            return None
        if not path.exists():
            return None
        return GlmSourceRef(model_id=model_id, source_kind="predictions")

    def source_path(self, model_id: str, source_kind: str) -> Path:
        if source_kind != "predictions":
            raise ValueError("Choose a valid GLM data source")
        return self.model_dir(model_id) / SOURCE_KINDS[source_kind]["file"]

    def source_id(self, model_id: str, source_kind: str = "predictions") -> str:
        if source_kind != "predictions":
            raise ValueError("Choose a valid GLM data source")
        return f"glm:{model_id}:predictions"

    def model_sources(self, model_id: str, raw_sources: Any = None) -> dict[str, str]:
        sources = {
            str(kind): str(value)
            for kind, value in (raw_sources.items() if isinstance(raw_sources, dict) else [])
            if str(kind) in SOURCE_KINDS and isinstance(value, str) and value
        }
        if self.source_path(model_id, "predictions").exists():
            sources["predictions"] = self.source_id(model_id)
        return sources

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
        model_id = str(item.get("model_id") or path.name)
        item["model_id"] = model_id
        item["sources"] = self.model_sources(model_id, item.get("sources"))
        item["diagnostics"] = self.model_diagnostics(model_id, item)
        item["metrics"] = item["diagnostics"]
        item["active"] = model_id == active_model_id
        return item

    def model_diagnostics(self, model_id: str, manifest: dict[str, Any] | None = None) -> dict[str, Any]:
        diagnostics = self.read_json(self.artifact_path(model_id, "diagnostics"), {})
        if isinstance(diagnostics, dict):
            return diagnostics
        manifest = manifest or {}
        raw = manifest.get("diagnostics")
        return dict(raw) if isinstance(raw, dict) else {}

    def source_columns(self, manifest: dict[str, Any]) -> list[str]:
        raw_columns = manifest.get("source_columns")
        if isinstance(raw_columns, list):
            columns = [str(name).strip() for name in raw_columns if str(name or "").strip()]
            if columns:
                return dedupe_columns(columns)
        dataset = Dataset(self.dataset_path)
        return list(dataset.column_map())

    def dataset_relation_sql(self) -> str:
        return dataset_relation_sql(self.dataset_path)

    def relation_sql(self, source_id: str) -> str:
        ref = self.source_ref(source_id)
        if not ref:
            raise ValueError("Choose a valid data source")
        manifest = self.manifest(ref.model_id)
        prediction_path = self.source_path(ref.model_id, "predictions")
        tabulated_prediction_path = self.artifact_path(ref.model_id, "tabulated_predictions")
        include_tabulated = tabulated_prediction_path.exists()
        denominator_col = str(manifest.get("denominator_column") or "").strip()
        where_sql = f"\n  WHERE {denominator_valid_sql(denominator_col)}" if denominator_col else ""
        source_columns = self.source_columns(manifest)
        select_sql = prediction_source_select_sql(source_columns, include_tabulated=include_tabulated)
        base_projection_sql = row_number_source_projection_sql(source_columns)
        tabulated_join_sql = (
            f"\nLEFT JOIN read_parquet({sql_literal(str(tabulated_prediction_path))}) tabulated USING (__lucidum_row_id)"
            if include_tabulated
            else ""
        )
        return f"""(
SELECT
  {select_sql}
FROM (
  SELECT
    *
  FROM (
    SELECT
      {base_projection_sql}
    FROM {self.dataset_relation_sql()}
  ) dataset_rows{where_sql}
) base
INNER JOIN read_parquet({sql_literal(str(prediction_path))}) prediction USING (__lucidum_row_id)
{tabulated_join_sql}
)"""

    def read_parquet_records(self, path: Path, *, limit: int | None = None) -> list[dict[str, Any]]:
        if not path.exists():
            return []
        limit_sql = f" LIMIT {int(limit)}" if limit is not None else ""
        con = duckdb.connect(database=":memory:")
        try:
            rows = con.execute(f"SELECT * FROM read_parquet({sql_literal(str(path))}){limit_sql}").fetchall()
            names = [str(col[0]) for col in con.description]
        finally:
            con.close()
        return [dict(zip(names, row)) for row in rows]

    def model_detail(self, model_id: str) -> dict[str, Any]:
        manifest = self.manifest(model_id)
        return {
            "manifest": manifest,
            "coefficients": self.read_parquet_records(self.artifact_path(model_id, "coefficients")),
            "feature_importance": self.read_parquet_records(self.artifact_path(model_id, "feature_importance")),
            "diagnostics": self.model_diagnostics(model_id, manifest),
            "formula": self.artifact_path(model_id, "formula").read_text(encoding="utf-8") if self.artifact_path(model_id, "formula").exists() else "",
            "active": self.active_model_id() == model_id,
        }

    def source_manifest_entries(self) -> list[tuple[dict[str, Any], str, dict[str, str]]]:
        entries: list[tuple[dict[str, Any], str, dict[str, str]]] = []
        for model in self.list_models():
            model_id = str(model.get("model_id") or "")
            if model_id and self.source_path(model_id, "predictions").exists():
                entries.append((model, self.source_id(model_id), SOURCE_KINDS["predictions"]))
        return entries


class GlmSourceProvider:
    def __init__(self, store: GlmModelStore):
        self.store = store

    def has_source(self, source_id: str) -> bool:
        return self.store.source_ref(source_id) is not None

    def relation_sql(self, source_id: str) -> str:
        return self.store.relation_sql(source_id)

    def prediction_source(self, source_id: str) -> ModelPredictionSource | None:
        ref = self.store.source_ref(source_id)
        if ref is None:
            return None
        source_path = self.store.source_path(ref.model_id, "predictions")
        tabulated_path = self.store.artifact_path(ref.model_id, "tabulated_predictions")
        relation_sql = f"read_parquet({sql_literal(str(source_path))})"
        if tabulated_path.exists():
            relation_sql = f"""(
SELECT
  prediction.__lucidum_row_id,
  prediction.glm_prediction,
  tabulated.glm_tabulated_prediction
FROM read_parquet({sql_literal(str(source_path))}) prediction
LEFT JOIN read_parquet({sql_literal(str(tabulated_path))}) tabulated USING (__lucidum_row_id)
)"""
        return ModelPredictionSource(
            source_id=source_id,
            column="glm_prediction",
            relation_sql=relation_sql,
            active=self.store.active_model_id() == ref.model_id,
        )

    def data_sources(self, dataset: Dataset) -> list[dict[str, Any]]:
        sources: list[dict[str, Any]] = []
        for model, source_id, info in self.store.source_manifest_entries():
            schema = dataset.schema_for_source(source_id)
            diagnostics = model.get("diagnostics") if isinstance(model.get("diagnostics"), dict) else {}
            label = f"{model.get('label') or model.get('model_id')} - {info['label']}"
            sources.append(
                {
                    "id": source_id,
                    "label": label,
                    "kind": info["kind"],
                    "model_id": model.get("model_id"),
                    "active": bool(model.get("active")),
                    "response_column": model.get("response_column"),
                    "denominator_column": model.get("denominator_column"),
                    "offset_column": model.get("denominator_column"),
                    "created_at": model.get("created_at"),
                    "family": model.get("family"),
                    "link": model.get("link"),
                    "training_scope": model.get("training_scope"),
                    "metrics": diagnostics,
                    "row_count": schema["row_count"],
                    "columns": schema["columns"],
                }
            )
        return sources


__all__ = [
    "ARTIFACT_FILES",
    "GlmModelNameError",
    "GlmModelStore",
    "GlmSourceProvider",
    "GlmSourceRef",
    "SOURCE_KINDS",
    "json_safe_number",
]
