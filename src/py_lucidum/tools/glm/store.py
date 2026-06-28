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

from py_lucidum.core import Dataset, ModelPredictionSource, ModelSourceBinding, dataset_workspace_metadata, quote_ident, sql_literal

from .validation import denominator_valid_sql, dataset_relation_sql


ARTIFACT_FILES = {
    "manifest": "manifest.json",
    "estimator": "estimator.pkl",
    "formula": "formula.txt",
    "coefficients": "coefficients.parquet",
    "feature_importance": "feature_importance.parquet",
    "predictions": "predictions.parquet",
    "diagnostics": "diagnostics.json",
    "tabulation_manifest": "tabulations/tabulation_manifest.json",
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


def parquet_columns(path: Path) -> set[str]:
    if not path.exists():
        return set()
    con = duckdb.connect(database=":memory:")
    try:
        rows = con.execute(f"DESCRIBE SELECT * FROM read_parquet({sql_literal(str(path))})").fetchall()
    finally:
        con.close()
    return {str(row[0]) for row in rows}


def path_cache_key(path: Path) -> tuple[str, int, int]:
    stat = path.stat()
    return (str(path), int(stat.st_size), int(stat.st_mtime_ns))


def source_columns_with_denominator(source_columns: list[str], denominator_col: str) -> list[str]:
    if not denominator_col or denominator_col in source_columns:
        return source_columns
    return dedupe_columns([*source_columns, denominator_col])


def row_number_source_projection_sql(source_columns: list[str]) -> str:
    columns_sql = ",\n    ".join(quote_ident(name) for name in source_columns)
    suffix = f",\n    {columns_sql}" if columns_sql else ""
    return f"ROW_NUMBER() OVER () AS __lucidum_row_id{suffix}"


def glm_prediction_rate_sql(denominator_col: str, *, artifact_has_rate: bool = False, base_alias: str = "base") -> str:
    if artifact_has_rate:
        return "prediction.glm_prediction_rate"
    denominator_sql = quote_ident(denominator_col)
    return (
        f"CASE WHEN TRY_CAST({base_alias}.{denominator_sql} AS DOUBLE) > 0 "
        f"THEN TRY_CAST(prediction.glm_prediction AS DOUBLE) / TRY_CAST({base_alias}.{denominator_sql} AS DOUBLE) "
        "ELSE NULL END AS glm_prediction_rate"
    )


def prediction_source_select_sql(
    source_columns: list[str],
    *,
    denominator_col: str = "",
    prediction_has_rate: bool = False,
    include_tabulated: bool = False,
) -> str:
    parts = [f"base.{quote_ident(name)}" for name in source_columns]
    parts.append("prediction.glm_prediction")
    if denominator_col:
        parts.append(glm_prediction_rate_sql(denominator_col, artifact_has_rate=prediction_has_rate))
    if include_tabulated:
        parts.append("tabulated.glm_tabulated_prediction")
    return ",\n  ".join(parts)


def artifact_identity_sql(path: Path) -> str:
    return f"(SELECT __lucidum_row_id FROM read_parquet({sql_literal(str(path))}))"


def artifact_column_relation_sql(path: Path, columns: list[str]) -> str:
    select_sql = ",\n  ".join(quote_ident(column) for column in columns)
    return f"""(
SELECT
  {select_sql}
FROM read_parquet({sql_literal(str(path))})
)"""


class GlmModelStore:
    def __init__(self, dataset_path: str | Path, dataset: Dataset | None = None):
        self.dataset_path = Path(dataset_path).expanduser().resolve()
        self._dataset = dataset
        self._workspace_stat_key: tuple[int, int] | None = None
        self._workspace_metadata: dict[str, Any] | None = None
        self._root: Path | None = None

    @property
    def root(self) -> Path:
        return self.dataset_workspace_root() / "models" / "glm"

    def dataset_workspace_root(self) -> Path:
        self._ensure_workspace_cache()
        if self._root is None:
            raise ValueError("Could not resolve dataset workspace")
        return self._root

    def dataset_metadata(self) -> dict[str, Any]:
        self._ensure_workspace_cache()
        return dict(self._workspace_metadata or {})

    def _ensure_workspace_cache(self) -> None:
        stat = self.dataset_path.stat()
        stat_key = (int(stat.st_size), int(stat.st_mtime_ns))
        if self._workspace_stat_key == stat_key and self._workspace_metadata is not None and self._root is not None:
            return
        metadata = dataset_workspace_metadata(self.dataset_path, self._dataset)
        self._workspace_metadata = metadata
        self._root = self.dataset_path.parent / ".lucidum" / "datasets" / str(metadata["slug"]) / str(metadata["signature"])
        self._workspace_stat_key = stat_key

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
        return self.model_list_item(self.model_dir(model_id), manifest, model_id)

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
            return self.model_list_item(target, renamed, new_id)
        return self.model_list_item(target, renamed, self.active_model_id())

    def delete_model(self, model_id: str) -> dict[str, Any]:
        deleted_id = self.validate_model_id(model_id)
        manifest = self.manifest(deleted_id)
        deleted = self.model_list_item(self.model_dir(deleted_id), manifest, self.active_model_id())
        was_active = self.active_model_id() == deleted_id
        shutil.rmtree(self.model_dir(deleted_id))
        if was_active:
            remaining = self.list_models()
            next_id = str(remaining[0].get("model_id") or "") if remaining else ""
            if next_id:
                self.activate_model(next_id)
            else:
                self.clear_active_model()
        return deleted

    def _renamed_manifest(self, manifest: dict[str, Any], old_id: str, new_id: str) -> dict[str, Any]:
        renamed = dict(manifest)
        renamed["model_id"] = new_id
        renamed["label"] = new_id
        return renamed

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

    def model_sources(self, model_id: str) -> dict[str, str]:
        sources: dict[str, str] = {}
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
        diagnostics = self.model_diagnostics(model_id, item)
        denominator_col = str(item.get("denominator_column") or "").strip()
        item["model_id"] = model_id
        item["denominator_column"] = denominator_col
        item["offset_column"] = denominator_col
        item["sources"] = self.model_sources(model_id)
        item["diagnostics"] = diagnostics
        item["metrics"] = diagnostics
        if diagnostics.get("training_rows") is not None:
            item["training_rows"] = diagnostics.get("training_rows")
        if diagnostics.get("scored_rows") is not None:
            item["scored_rows"] = diagnostics.get("scored_rows")
        if diagnostics.get("fitted_na_rows") is not None:
            item["fitted_na_rows"] = diagnostics.get("fitted_na_rows")
        if diagnostics.get("coefficient_count") is not None:
            item["coefficient_count"] = diagnostics.get("coefficient_count")
        item["active"] = model_id == active_model_id
        return item

    def model_diagnostics(self, model_id: str, manifest: dict[str, Any] | None = None) -> dict[str, Any]:
        diagnostics = self.read_json(self.artifact_path(model_id, "diagnostics"), {})
        if isinstance(diagnostics, dict):
            return diagnostics
        return {}

    def source_columns(self, manifest: dict[str, Any] | None = None) -> list[str]:
        del manifest
        dataset = self._dataset or Dataset(self.dataset_path)
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
        source_columns = source_columns_with_denominator(self.source_columns(manifest), denominator_col)
        prediction_has_rate = "glm_prediction_rate" in parquet_columns(prediction_path)
        select_sql = prediction_source_select_sql(
            source_columns,
            denominator_col=denominator_col,
            prediction_has_rate=prediction_has_rate,
            include_tabulated=include_tabulated,
        )
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
        manifest = self.store.manifest(ref.model_id)
        denominator_col = str(manifest.get("denominator_column") or "").strip()
        prediction_has_rate = "glm_prediction_rate" in parquet_columns(source_path)
        tabulated_path = self.store.artifact_path(ref.model_id, "tabulated_predictions")
        bindings: dict[str, ModelSourceBinding] = {}
        if not manifest.get("offset_terms"):
            base_where_sql = denominator_valid_sql(denominator_col) if denominator_col else ""
            base_columns = (denominator_col,) if denominator_col else ()
            prediction_columns = ["glm_prediction"]
            if prediction_has_rate:
                prediction_columns.append("glm_prediction_rate")
            prediction_binding = ModelSourceBinding(
                relation_sql=artifact_column_relation_sql(source_path, prediction_columns),
                columns=tuple(prediction_columns),
                identity_sqls=(artifact_identity_sql(source_path),),
                base_where_sql=base_where_sql,
                base_columns=base_columns,
                cache_key=("glm", ref.model_id, "predictions", path_cache_key(source_path)),
            )
            bindings["glm_prediction"] = prediction_binding
            if prediction_has_rate:
                bindings["glm_prediction_rate"] = prediction_binding
            if tabulated_path.exists():
                bindings["glm_tabulated_prediction"] = ModelSourceBinding(
                    relation_sql=artifact_column_relation_sql(tabulated_path, ["glm_tabulated_prediction"]),
                    columns=("glm_tabulated_prediction",),
                    identity_sqls=(artifact_identity_sql(tabulated_path),),
                    base_where_sql=base_where_sql,
                    base_columns=base_columns,
                    cache_key=("glm", ref.model_id, "tabulated_predictions", path_cache_key(tabulated_path)),
                )
        rate_select_sql = f",\n  {glm_prediction_rate_sql(denominator_col, artifact_has_rate=prediction_has_rate)}" if denominator_col else ""
        base_join_sql = ""
        if denominator_col and not prediction_has_rate:
            denominator_sql = quote_ident(denominator_col)
            base_join_sql = f"""
LEFT JOIN (
  SELECT
    ROW_NUMBER() OVER () AS __lucidum_row_id,
    {denominator_sql}
  FROM {self.store.dataset_relation_sql()}
) base USING (__lucidum_row_id)"""
        relation_sql = f"""(
SELECT
  prediction.__lucidum_row_id,
  prediction.glm_prediction{rate_select_sql}
FROM read_parquet({sql_literal(str(source_path))}) prediction
{base_join_sql}
)"""
        if tabulated_path.exists():
            relation_sql = f"""(
SELECT
  prediction.__lucidum_row_id,
  prediction.glm_prediction{rate_select_sql},
  tabulated.glm_tabulated_prediction
FROM read_parquet({sql_literal(str(source_path))}) prediction
{base_join_sql}
LEFT JOIN read_parquet({sql_literal(str(tabulated_path))}) tabulated USING (__lucidum_row_id)
)"""
        return ModelPredictionSource(
            source_id=source_id,
            column="glm_prediction",
            relation_sql=relation_sql,
            active=self.store.active_model_id() == ref.model_id,
            binding=bindings.get("glm_prediction"),
            bindings=bindings,
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
