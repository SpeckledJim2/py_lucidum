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

from .sample import GENERATED_SAMPLE_FILENAME
from .validation import DEFAULT_TRAINING_MODE, display_monotonicity, normalise_monotonicity


ARTIFACT_FILES = {
    "predictions": "predictions.parquet",
    "init_score": "init_score.parquet",
    "shap_long": "shap_values.parquet",
    "shap_summary": "shap_summary.parquet",
    "evaluation": "evaluation.parquet",
    "tree_table": "tree_table.parquet",
    "manifest": "manifest.json",
    "features": "features.json",
    "feature_config": "feature_config.parquet",
    "parameters": "parameters.json",
    "model": "model.txt",
    "tabulation_manifest": "tabulations/tabulation_manifest.json",
    "tabulated_predictions": "tabulated_predictions.parquet",
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


def feature_importance_value(row: dict[str, Any]) -> float:
    shap_value = json_safe_number(row.get("mean_abs_shap"))
    if shap_value is not None:
        return shap_value
    return json_safe_number(row.get("gain")) or 0.0


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


def source_columns_with_offset(source_columns: list[str], offset_col: str) -> list[str]:
    if not offset_col or offset_col in source_columns:
        return source_columns
    return dedupe_columns([*source_columns, offset_col])


def unique_output_column_name(base_name: str, used_names: set[str]) -> str:
    base = str(base_name or "column").strip() or "column"
    candidate = base
    suffix = 2
    while candidate in used_names:
        candidate = f"{base}_{suffix}"
        suffix += 1
    used_names.add(candidate)
    return candidate


def gbm_prediction_rate_sql(offset_col: str, *, artifact_has_rate: bool = False, base_alias: str = "base") -> str:
    if artifact_has_rate:
        return "prediction.gbm_prediction_rate"
    offset_sql = quote_ident(offset_col)
    return (
        f"CASE WHEN TRY_CAST({base_alias}.{offset_sql} AS DOUBLE) > 0 "
        f"THEN TRY_CAST(prediction.gbm_prediction AS DOUBLE) / TRY_CAST({base_alias}.{offset_sql} AS DOUBLE) "
        "ELSE NULL END AS gbm_prediction_rate"
    )


def prediction_source_select_sql(
    source_columns: list[str],
    *,
    offset_col: str = "",
    prediction_has_rate: bool = False,
    include_tabulated: bool = False,
) -> str:
    parts = [f"base.{quote_ident(name)}" for name in source_columns]
    parts.append("prediction.gbm_prediction")
    if offset_col:
        parts.append(gbm_prediction_rate_sql(offset_col, artifact_has_rate=prediction_has_rate))
    if include_tabulated:
        parts.append("tabulated.gbm_tabulated_prediction")
    return ",\n  ".join(parts)


def shap_source_select_sql(
    source_columns: list[str],
    shap_columns: list[dict[str, str]],
    *,
    include_prediction: bool = False,
    offset_col: str = "",
    prediction_has_rate: bool = False,
) -> str:
    parts = [f"base.{quote_ident(name)}" for name in source_columns]
    if include_prediction:
        parts.append("prediction.gbm_prediction")
        if offset_col:
            parts.append(gbm_prediction_rate_sql(offset_col, artifact_has_rate=prediction_has_rate))
    parts.extend(
        f"shap.{quote_ident(column['artifact_column'])} AS {quote_ident(column['name'])}"
        for column in shap_columns
    )
    return ",\n  ".join(parts)


def row_number_source_projection_sql(source_columns: list[str]) -> str:
    columns_sql = ",\n    ".join(quote_ident(name) for name in source_columns)
    suffix = f",\n    {columns_sql}" if columns_sql else ""
    return f"ROW_NUMBER() OVER () AS __lucidum_row_id{suffix}"


def artifact_identity_sql(path: Path) -> str:
    return f"(SELECT __lucidum_row_id FROM read_parquet({sql_literal(str(path))}))"


def artifact_column_relation_sql(path: Path, columns: list[str]) -> str:
    select_sql = ",\n  ".join(quote_ident(column) for column in columns)
    return f"""(
SELECT
  {select_sql}
FROM read_parquet({sql_literal(str(path))})
)"""


@dataclass(frozen=True)
class GbmSourceRef:
    model_id: str
    source_kind: str


class GbmModelStore:
    def __init__(self, dataset_path: str | Path, dataset: Dataset | None = None):
        self.dataset_path = Path(dataset_path).expanduser().resolve()
        self._dataset = dataset
        self._workspace_stat_key: tuple[int, int] | None = None
        self._workspace_metadata: dict[str, Any] | None = None
        self._root: Path | None = None

    @property
    def root(self) -> Path:
        return self.dataset_workspace_root() / "models" / "gbm"

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

    def _clean_manifest(self, manifest: dict[str, Any]) -> dict[str, Any]:
        cleaned = dict(manifest)
        cleaned.pop("source_columns", None)
        init_score = cleaned.get("init_score")
        if isinstance(init_score, dict):
            cleaned_init_score = dict(init_score)
            cleaned_init_score.pop("artifact_path", None)
            cleaned["init_score"] = cleaned_init_score
        return cleaned

    def manifest(self, model_id: str) -> dict[str, Any]:
        path = self.artifact_path(model_id, "manifest")
        manifest = self.read_json(path)
        if not isinstance(manifest, dict):
            raise ValueError("Choose a valid GBM model")
        manifest = self._clean_manifest(manifest)
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
        item = self._clean_manifest(manifest)
        item.setdefault("training_mode", DEFAULT_TRAINING_MODE)
        model_id = str(item.get("model_id") or path.name)
        parameters = self.model_parameters(model_id)
        item["model_id"] = model_id
        item["sources"] = self.model_sources(model_id)
        item["parameters"] = parameters
        item["objective"] = str(parameters.get("objective") or "")
        item["metric"] = str(parameters.get("metric") or "")
        item["best_metrics"] = self.model_best_metrics(model_id, item["metric"], item.get("best_iteration"))
        item["active"] = model_id == active_model_id
        return item

    def model_sources(self, model_id: str) -> dict[str, str]:
        sources: dict[str, str] = {}
        for source_kind in SOURCE_KINDS:
            if self.source_path(model_id, source_kind).exists():
                sources[source_kind] = self.source_id(model_id, source_kind)
        return sources

    def model_parameters(self, model_id: str) -> dict[str, Any]:
        parameters = self.read_json(self.artifact_path(model_id, "parameters"), {})
        return dict(parameters) if isinstance(parameters, dict) else {}

    def model_feature_names(self, model_id: str) -> list[str]:
        raw_features = self.read_json(self.artifact_path(model_id, "features"), [])
        if not isinstance(raw_features, list):
            return []
        names: list[str] = []
        seen: set[str] = set()
        for item in raw_features:
            name = str(item or "").strip()
            if not name or name in seen or name == "__lucidum_row_id":
                continue
            names.append(name)
            seen.add(name)
        return names

    def model_shap_importance(self, model_id: str) -> dict[str, float]:
        values: dict[str, float] = {}
        for row in self.read_parquet_records(self.artifact_path(model_id, "shap_summary")):
            feature = str(row.get("feature") or "").strip()
            value = json_safe_number(row.get("mean_abs_shap"))
            if feature and value is not None:
                values[feature] = value
        return values

    def model_feature_config(self, model_id: str, *, sort_by_importance: bool = False) -> list[dict[str, Any]]:
        feature_names = self.model_feature_names(model_id)
        if not feature_names:
            return []
        config_by_name: dict[str, dict[str, Any]] = {}
        for item in self.read_parquet_records(self.artifact_path(model_id, "feature_config")):
            name = str(item.get("name") or item.get("feature") or "").strip()
            if name:
                config_by_name[name] = item
        parameters = self.model_parameters(model_id)
        constraints = parameters.get("monotone_constraints")
        monotone_constraints = constraints if isinstance(constraints, list) else []
        shap_importance = self.model_shap_importance(model_id)
        dataset = self._dataset or Dataset(self.dataset_path)
        columns = dataset.column_map()
        rows: list[dict[str, Any]] = []
        for index, name in enumerate(feature_names):
            saved = dict(config_by_name.get(name, {}))
            column = columns.get(name)
            row = dict(saved)
            row["name"] = name
            row["include"] = True
            row["kind"] = str(saved.get("kind") or (column.kind if column else ""))
            monotonicity_value = self.feature_monotonicity_value(saved, monotone_constraints, index)
            row["monotonicity"] = display_monotonicity(monotonicity_value)
            row["monotonicity_value"] = monotonicity_value
            row["gain"] = round(json_safe_number(saved.get("gain")) or 0.0, 3)
            mean_abs_shap = json_safe_number(saved.get("mean_abs_shap"))
            if mean_abs_shap is None:
                mean_abs_shap = shap_importance.get(name)
            if mean_abs_shap is None:
                row.pop("mean_abs_shap", None)
            else:
                row["mean_abs_shap"] = mean_abs_shap
            rows.append(row)
        if sort_by_importance:
            return sorted(rows, key=lambda row: (-feature_importance_value(row), str(row["name"]).lower()))
        return rows

    @staticmethod
    def feature_monotonicity_value(saved: dict[str, Any], constraints: list[Any], index: int) -> int:
        if index < len(constraints):
            try:
                value = int(float(constraints[index]))
            except (TypeError, ValueError):
                value = 0
            if value > 0:
                return 1
            if value < 0:
                return -1
            return 0
        for key in ("monotonicity_value", "monotonicity"):
            if key not in saved:
                continue
            value = saved.get(key)
            if value is None or str(value).strip() == "":
                continue
            try:
                return normalise_monotonicity(value)
            except ValueError:
                continue
        return 0

    def model_best_metrics(self, model_id: str, metric_name: str, best_iteration: Any) -> dict[str, float | None]:
        return best_metrics_from_evaluation(
            self.read_evaluation(model_id),
            metric_name,
            best_iteration,
        )

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
        new_id = self.validate_model_id(new_model_id, for_new_name=True)
        old_id = self.validate_model_id(model_id)
        manifest = self.manifest(old_id)
        if new_id == old_id:
            return self.model_list_item(self.model_dir(old_id), manifest, self.active_model_id())
        source = self.model_dir(old_id)
        target = self.model_dir(new_id)
        if target.exists():
            raise GbmModelNameError(f"GBM model already exists: {new_id}")
        source.rename(target)
        manifest = self._renamed_manifest(manifest, old_id, new_id)
        self.write_json(self.artifact_path(new_id, "manifest"), manifest)
        if self.active_model_id() == old_id:
            self.activate_model(new_id)
            return self.model_list_item(target, manifest, new_id)
        return self.model_list_item(target, manifest, self.active_model_id())

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

    def parquet_columns(self, path: Path) -> list[str]:
        con = duckdb.connect(database=":memory:")
        try:
            rows = con.execute(f"DESCRIBE SELECT * FROM read_parquet({sql_literal(str(path))})").fetchall()
            return [str(row[0]) for row in rows]
        finally:
            con.close()

    def shap_value_columns(self, model_id: str, source_columns: list[str] | None = None) -> list[dict[str, str]]:
        used_names = set(source_columns or self.source_projection_columns())
        used_names.add("__lucidum_row_id")
        columns: list[dict[str, str]] = []
        for artifact_column in self.parquet_columns(self.artifact_path(model_id, "shap_long")):
            if artifact_column == "__lucidum_row_id":
                continue
            alias = unique_output_column_name(f"SHAP__{artifact_column}", used_names)
            columns.append({"artifact_column": artifact_column, "name": alias, "label": artifact_column})
        return columns

    def relation_sql(self, source_id: str) -> str:
        ref = self.source_ref(source_id)
        if not ref:
            raise ValueError("Choose a valid data source")
        source_path = self.source_path(ref.model_id, ref.source_kind)
        if ref.source_kind == "shap_long":
            return self.shap_relation_sql(ref.model_id, source_path)
        if ref.source_kind != "predictions":
            return f"read_parquet({sql_literal(str(source_path))})"
        manifest = self.manifest(ref.model_id)
        offset_col = str(manifest.get("offset_column") or "").strip()
        where_sql = f"\nWHERE TRY_CAST({quote_ident(offset_col)} AS DOUBLE) > 0" if offset_col else ""
        source_columns = source_columns_with_offset(self.source_projection_columns(), offset_col)
        tabulated_prediction_path = self.artifact_path(ref.model_id, "tabulated_predictions")
        include_tabulated = tabulated_prediction_path.exists()
        prediction_has_rate = "gbm_prediction_rate" in parquet_columns(source_path)
        select_sql = prediction_source_select_sql(
            source_columns,
            offset_col=offset_col,
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
  ) dataset_rows
  {where_sql}
) base
INNER JOIN read_parquet({sql_literal(str(source_path))}) prediction USING (__lucidum_row_id)
{tabulated_join_sql}
)"""

    def shap_relation_sql(self, model_id: str, source_path: Path) -> str:
        positional_sql = self.positional_shap_relation_sql(model_id, source_path)
        if positional_sql:
            return positional_sql
        manifest = self.manifest(model_id)
        offset_col = str(manifest.get("offset_column") or "").strip()
        where_sql = f"\nWHERE TRY_CAST({quote_ident(offset_col)} AS DOUBLE) > 0" if offset_col else ""
        source_columns = source_columns_with_offset(self.source_projection_columns(), offset_col)
        shap_columns = self.shap_value_columns(model_id, source_columns)
        prediction_path = self.source_path(model_id, "predictions")
        include_prediction = prediction_path.exists()
        prediction_has_rate = "gbm_prediction_rate" in parquet_columns(prediction_path)
        select_sql = shap_source_select_sql(
            source_columns,
            shap_columns,
            include_prediction=include_prediction,
            offset_col=offset_col,
            prediction_has_rate=prediction_has_rate,
        )
        base_projection_sql = row_number_source_projection_sql(source_columns)
        prediction_join_sql = (
            f"\nLEFT JOIN read_parquet({sql_literal(str(prediction_path))}) prediction USING (__lucidum_row_id)"
            if include_prediction
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
  ) dataset_rows
  {where_sql}
) base
INNER JOIN read_parquet({sql_literal(str(source_path))}) shap USING (__lucidum_row_id)
{prediction_join_sql}
)"""

    def positional_shap_relation_sql(self, model_id: str, source_path: Path) -> str:
        dataset = self._dataset
        if dataset is None:
            return ""
        manifest = self.manifest(model_id)
        offset_col = str(manifest.get("offset_column") or "").strip()
        base_where_sql = f"TRY_CAST({quote_ident(offset_col)} AS DOUBLE) > 0" if offset_col else ""
        base_columns = (offset_col,) if offset_col else ()
        source_columns = source_columns_with_offset(self.source_projection_columns(), offset_col)
        shap_columns = self.shap_value_columns(model_id, source_columns)
        if not shap_columns:
            return ""
        prediction_path = self.source_path(model_id, "predictions")
        include_prediction = prediction_path.exists()
        prediction_has_rate = "gbm_prediction_rate" in parquet_columns(prediction_path)
        identity_sqls = [artifact_identity_sql(source_path)]
        cache_key: list[Any] = ["gbm", model_id, "shap_long", path_cache_key(source_path)]
        if include_prediction:
            identity_sqls.append(artifact_identity_sql(prediction_path))
            cache_key.extend(["predictions", path_cache_key(prediction_path)])
        binding = ModelSourceBinding(
            relation_sql="gbm_shap_positional",
            columns=("gbm_shap_positional",),
            identity_sqls=tuple(identity_sqls),
            base_where_sql=base_where_sql,
            base_columns=base_columns,
            cache_key=tuple(cache_key),
        )
        if not dataset.model_source_binding_eligible(binding):
            return ""

        select_sql = shap_source_select_sql(
            source_columns,
            shap_columns,
            include_prediction=include_prediction,
            offset_col=offset_col,
            prediction_has_rate=prediction_has_rate,
        )
        source_column_sql = ",\n    ".join(quote_ident(name) for name in source_columns)
        shap_column_sql = ",\n    ".join(quote_ident(column["artifact_column"]) for column in shap_columns)
        prediction_columns = ["gbm_prediction"]
        if prediction_has_rate:
            prediction_columns.append("gbm_prediction_rate")
        prediction_column_sql = ",\n    ".join(quote_ident(column) for column in prediction_columns)
        where_sql = f"\n  WHERE {base_where_sql}" if base_where_sql else ""
        prediction_join_sql = (
            f"""
POSITIONAL JOIN (
  SELECT
    {prediction_column_sql}
  FROM read_parquet({sql_literal(str(prediction_path))})
) prediction"""
            if include_prediction
            else ""
        )
        return f"""(
SELECT
  {select_sql}
FROM (
  SELECT
    {source_column_sql}
  FROM {dataset.relation_sql()}
  {where_sql}
) base
POSITIONAL JOIN (
  SELECT
    {shap_column_sql}
  FROM read_parquet({sql_literal(str(source_path))})
) shap
{prediction_join_sql}
)"""

    def source_projection_columns(self) -> list[str]:
        dataset = self._dataset or Dataset(self.dataset_path)
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
        parameters = self.model_parameters(model_id)
        detail = {
            "manifest": manifest,
            "features": self.model_feature_config(model_id, sort_by_importance=True),
            "parameters": parameters,
            "objective": str(parameters.get("objective") or ""),
            "metric": str(parameters.get("metric") or ""),
            "evaluation": self.read_evaluation(model_id),
            "active": self.active_model_id() == model_id,
        }
        return detail

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

    def read_evaluation(self, model_id: str) -> dict[str, dict[str, list[float | None]]]:
        path = self.artifact_path(model_id, "evaluation")
        if not path.exists():
            return {}
        con = duckdb.connect(database=":memory:")
        try:
            rows = con.execute(
                f"""
SELECT dataset, metric, iteration, value
FROM read_parquet({sql_literal(str(path))})
WHERE dataset IS NOT NULL AND metric IS NOT NULL
ORDER BY dataset, metric, iteration
"""
            ).fetchall()
        finally:
            con.close()
        evaluation: dict[str, dict[str, list[float | None]]] = {}
        for dataset_name, metric_name, iteration, value in rows:
            dataset_key = str(dataset_name)
            metric_key = str(metric_name)
            try:
                index = int(iteration) - 1
            except (TypeError, ValueError):
                continue
            if index < 0:
                continue
            values = evaluation.setdefault(dataset_key, {}).setdefault(metric_key, [])
            while len(values) < index:
                values.append(None)
            number = json_safe_number(value)
            if len(values) == index:
                values.append(number)
            else:
                values[index] = number
        return evaluation

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

    def prediction_source(self, source_id: str) -> ModelPredictionSource | None:
        ref = self.store.source_ref(source_id)
        if ref is None or ref.source_kind != "predictions":
            return None
        source_path = self.store.source_path(ref.model_id, "predictions")
        manifest = self.store.manifest(ref.model_id)
        offset_col = str(manifest.get("offset_column") or "").strip()
        prediction_has_rate = "gbm_prediction_rate" in parquet_columns(source_path)
        tabulated_path = self.store.artifact_path(ref.model_id, "tabulated_predictions")
        base_where_sql = f"TRY_CAST({quote_ident(offset_col)} AS DOUBLE) > 0" if offset_col else ""
        base_columns = (offset_col,) if offset_col else ()
        prediction_columns = ["gbm_prediction"]
        if prediction_has_rate:
            prediction_columns.append("gbm_prediction_rate")
        bindings: dict[str, ModelSourceBinding] = {}
        prediction_binding = ModelSourceBinding(
            relation_sql=artifact_column_relation_sql(source_path, prediction_columns),
            columns=tuple(prediction_columns),
            identity_sqls=(artifact_identity_sql(source_path),),
            base_where_sql=base_where_sql,
            base_columns=base_columns,
            cache_key=("gbm", ref.model_id, "predictions", path_cache_key(source_path)),
        )
        bindings["gbm_prediction"] = prediction_binding
        if prediction_has_rate:
            bindings["gbm_prediction_rate"] = prediction_binding
        if tabulated_path.exists():
            bindings["gbm_tabulated_prediction"] = ModelSourceBinding(
                relation_sql=artifact_column_relation_sql(tabulated_path, ["gbm_tabulated_prediction"]),
                columns=("gbm_tabulated_prediction",),
                identity_sqls=(artifact_identity_sql(tabulated_path),),
                base_where_sql=base_where_sql,
                base_columns=base_columns,
                cache_key=("gbm", ref.model_id, "tabulated_predictions", path_cache_key(tabulated_path)),
            )
        rate_select_sql = f",\n  {gbm_prediction_rate_sql(offset_col, artifact_has_rate=prediction_has_rate)}" if offset_col else ""
        base_join_sql = ""
        if offset_col and not prediction_has_rate:
            offset_sql = quote_ident(offset_col)
            base_join_sql = f"""
LEFT JOIN (
  SELECT
    ROW_NUMBER() OVER () AS __lucidum_row_id,
    {offset_sql}
  FROM {self.store.dataset_relation_sql()}
) base USING (__lucidum_row_id)"""
        relation_sql = f"""(
SELECT
  prediction.__lucidum_row_id,
  prediction.gbm_prediction{rate_select_sql}
FROM read_parquet({sql_literal(str(source_path))}) prediction
{base_join_sql}
)"""
        if tabulated_path.exists():
            relation_sql = f"""(
SELECT
  prediction.__lucidum_row_id,
  prediction.gbm_prediction{rate_select_sql},
  tabulated.gbm_tabulated_prediction
FROM read_parquet({sql_literal(str(source_path))}) prediction
{base_join_sql}
LEFT JOIN read_parquet({sql_literal(str(tabulated_path))}) tabulated USING (__lucidum_row_id)
)"""
        return ModelPredictionSource(
            source_id=source_id,
            column="gbm_prediction",
            relation_sql=relation_sql,
            active=self.store.active_model_id() == ref.model_id,
            binding=bindings.get("gbm_prediction"),
            bindings=bindings,
        )

    def active_shap_overlay_source(self, dataset: Dataset | None = None) -> dict[str, Any] | None:
        model_id = self.store.active_model_id()
        if not model_id:
            return None
        shap_path = self.store.source_path(model_id, "shap_long")
        if not shap_path.exists():
            return None
        prediction_path = self.store.source_path(model_id, "predictions")
        manifest = self.store.manifest(model_id)
        parameters = self.store.model_parameters(model_id)
        source_columns = list(dataset.column_map()) if dataset is not None else None
        return {
            "id": self.store.source_id(model_id, "shap_long"),
            "kind": SOURCE_KINDS["shap_long"]["kind"],
            "model_id": model_id,
            "active": True,
            "objective": str(parameters.get("objective") or manifest.get("objective") or ""),
            "prediction_source_id": self.store.source_id(model_id, "predictions"),
            "prediction_path": prediction_path,
            "has_prediction": prediction_path.exists(),
            "shap_path": shap_path,
            "columns": [
                {
                    **column,
                    "source_role": "gbm_shap_value",
                }
                for column in self.store.shap_value_columns(model_id, source_columns)
            ],
        }

    def data_sources(self, dataset: Dataset) -> list[dict[str, Any]]:
        sources: list[dict[str, Any]] = []
        for model, source_id, info in self.store.source_manifest_entries():
            schema = dataset.schema_for_source(source_id)
            columns = schema["columns"]
            if info["kind"] == "gbm_shap_long":
                shap_labels = {
                    column["name"]: column["label"]
                    for column in self.store.shap_value_columns(str(model.get("model_id") or ""))
                }
                columns = [
                    {
                        **column,
                        **(
                            {
                                "label": shap_labels[column["name"]],
                                "artifact_column": shap_labels[column["name"]],
                                "source_role": "gbm_shap_value",
                            }
                            if column["name"] in shap_labels
                            else {}
                        ),
                    }
                    for column in columns
                ]
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
                    "columns": columns,
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
