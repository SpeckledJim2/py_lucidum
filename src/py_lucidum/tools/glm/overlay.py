from __future__ import annotations

import atexit
import importlib.util
import json
import math
import os
import pickle
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path
from typing import Any
from uuid import uuid4

from py_lucidum.core import (
    Dataset,
    denominator_valid_condition,
    is_numeric_kind,
    json_number,
    quote_ident,
    sql_literal,
    weighted_value_sql,
)

from .store import GlmModelStore
from .tabulation import (
    _as_number,
    _clip_numeric_bound,
    _column_tokens,
    _feature_spec_map,
    _feature_transform_bounds,
    _json_value,
    _term_groups,
)
from .training import (
    MissingGlmDependency,
    add_internal_intercept_column,
    formula_context,
    glm_dependencies,
    internal_intercept_column_from_manifest,
    offset_values_for_frame,
)
from .validation import TARGET_COLUMN


DEFAULT_GLM_OVERLAY_SAMPLE_SEED = 2026
MAX_GLM_OVERLAY_SAMPLE_ROWS = 100_000
MAX_GLM_OVERLAY_PREDICTION_CELLS = 2_000_000
GLM_OVERLAY_CHUNK_CELLS = 100_000
GLM_OVERLAY_NUMERIC_INTERACTION_BINS = 50
MAX_GLM_OVERLAY_CATEGORICAL_INTERACTION_LEVELS = 200
MAX_GLM_OVERLAY_COLLAPSED_CONTEXT_ROWS = 2_000
MAX_GLM_OVERLAY_COLLAPSED_PREDICTION_CELLS = 250_000
_GLM_OVERLAY_ESTIMATOR_CACHE: dict[str, tuple[int, int, Any]] = {}


def empty_glm_partial_dependence_warning(message: str) -> dict[str, Any]:
    return {
        "mode": "glm",
        "model_id": "",
        "feature": "",
        "method": "none",
        "percentiles": [50],
        "rows": [],
        "warnings": [message],
        "scale": {"method": "none", "target": None, "source_mean": None},
        "sample": {},
        "transform": {"mode": "none"},
    }


def load_glm_overlay_estimator(estimator_path: Path) -> Any:
    stat = estimator_path.stat()
    cache_key = str(estimator_path)
    cached = _GLM_OVERLAY_ESTIMATOR_CACHE.get(cache_key)
    if cached and cached[0] == stat.st_mtime_ns and cached[1] == stat.st_size:
        return cached[2]
    with estimator_path.open("rb") as handle:
        estimator = pickle.load(handle)
    _GLM_OVERLAY_ESTIMATOR_CACHE[cache_key] = (stat.st_mtime_ns, stat.st_size, estimator)
    return estimator


def build_glm_partial_dependence_overlay(
    dataset: Dataset,
    request: dict[str, Any],
    *,
    feature_spec: Any,
    x_col: str,
    x_sql: dict[str, str],
    x_group_kind: str,
    denominator: dict[str, str | None],
) -> dict[str, Any]:
    started = time.perf_counter()
    if should_isolate_glm_overlay():
        result = build_glm_partial_dependence_overlay_in_subprocess(
            dataset,
            request,
            feature_spec=feature_spec,
            x_col=x_col,
            x_sql=x_sql,
            x_group_kind=x_group_kind,
            denominator=denominator,
        )
    else:
        result = _build_glm_partial_dependence_overlay_impl(
            dataset,
            request,
            feature_spec=feature_spec,
            x_col=x_col,
            x_sql=x_sql,
            x_group_kind=x_group_kind,
            denominator=denominator,
        )
    result.setdefault("timings", {})["elapsed_ms"] = round((time.perf_counter() - started) * 1000, 1)
    return result


def should_isolate_glm_overlay() -> bool:
    return (
        ("lightgbm" in sys.modules or importlib.util.find_spec("lightgbm") is not None)
        and not os.environ.get("PY_LUCIDUM_GLM_OVERLAY_WORKER")
    )


def build_glm_partial_dependence_overlay_in_subprocess(
    dataset: Dataset,
    request: dict[str, Any],
    *,
    feature_spec: Any,
    x_col: str,
    x_sql: dict[str, str],
    x_group_kind: str,
    denominator: dict[str, str | None],
) -> dict[str, Any]:
    if not os.environ.get("PY_LUCIDUM_GLM_OVERLAY_ONE_SHOT"):
        result, started = persistent_glm_overlay_worker().request(
            dataset,
            request,
            feature_spec=feature_spec,
            x_col=x_col,
            x_sql=x_sql,
            x_group_kind=x_group_kind,
            denominator=denominator,
        )
        result.setdefault("timings", {})["worker_mode"] = "persistent"
        result.setdefault("timings", {})["worker_started"] = bool(started)
        return result
    return build_glm_partial_dependence_overlay_one_shot(
        dataset,
        request,
        feature_spec=feature_spec,
        x_col=x_col,
        x_sql=x_sql,
        x_group_kind=x_group_kind,
        denominator=denominator,
    )


def build_glm_partial_dependence_overlay_one_shot(
    dataset: Dataset,
    request: dict[str, Any],
    *,
    feature_spec: Any,
    x_col: str,
    x_sql: dict[str, str],
    x_group_kind: str,
    denominator: dict[str, str | None],
) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="lucidum-glm-overlay-") as tmp_dir:
        tmp_path = Path(tmp_dir)
        request_path = tmp_path / "request.json"
        response_path = tmp_path / "response.json"
        request_path.write_text(
            json.dumps(
                {
                    "dataset_path": str(dataset.path),
                    "request": request,
                    "feature_spec": feature_spec,
                    "x_col": x_col,
                    "x_sql": x_sql,
                    "x_group_kind": x_group_kind,
                    "denominator": denominator,
                },
                default=str,
            ),
            encoding="utf-8",
        )
        completed = subprocess.run(
            [sys.executable, "-m", "py_lucidum.tools.glm.overlay_worker", str(request_path), str(response_path)],
            check=False,
            capture_output=True,
            text=True,
            env={
                **os.environ,
                "PY_LUCIDUM_SKIP_LIGHTGBM_PRELOAD": "1",
                "PY_LUCIDUM_GLM_OVERLAY_WORKER": "1",
            },
        )
        if completed.returncode != 0:
            detail = (completed.stderr or completed.stdout or "").strip()
            if len(detail) > 800:
                detail = f"{detail[:800]}..."
            suffix = f" {detail}" if detail else ""
            return empty_glm_partial_dependence_warning(f"GLM overlay worker exited unexpectedly with code {completed.returncode}.{suffix}")
        if not response_path.exists():
            return empty_glm_partial_dependence_warning("GLM overlay worker exited without writing a response.")
        response = json.loads(response_path.read_text(encoding="utf-8"))
    if not response.get("ok"):
        return empty_glm_partial_dependence_warning(str(response.get("error") or "GLM overlay worker failed."))
    result = response.get("result")
    if not isinstance(result, dict):
        return empty_glm_partial_dependence_warning("GLM overlay worker returned an invalid response.")
    return result


class PersistentGlmOverlayWorker:
    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._process: subprocess.Popen[str] | None = None
        self._tmp_dir: Path | None = None

    def request(
        self,
        dataset: Dataset,
        request: dict[str, Any],
        *,
        feature_spec: Any,
        x_col: str,
        x_sql: dict[str, str],
        x_group_kind: str,
        denominator: dict[str, str | None],
    ) -> tuple[dict[str, Any], bool]:
        with self._lock:
            try:
                process, started = self._ensure_process()
                request_id = uuid4().hex
                tmp_dir = self._tmp_dir
                if tmp_dir is None:
                    raise RuntimeError("GLM overlay worker temporary directory is not available.")
                request_path = tmp_dir / f"{request_id}.request.json"
                response_path = tmp_dir / f"{request_id}.response.json"
                request_path.write_text(
                    json.dumps(
                        {
                            "dataset_path": str(dataset.path),
                            "request": request,
                            "feature_spec": feature_spec,
                            "x_col": x_col,
                            "x_sql": x_sql,
                            "x_group_kind": x_group_kind,
                            "denominator": denominator,
                        },
                        default=str,
                    ),
                    encoding="utf-8",
                )
                if process.stdin is None or process.stdout is None:
                    raise RuntimeError("GLM overlay worker pipes are not available.")
                process.stdin.write(json.dumps({"request_id": request_id, "request_path": str(request_path), "response_path": str(response_path)}) + "\n")
                process.stdin.flush()
                self._wait_for_ack(process, request_id)
                if not response_path.exists():
                    raise RuntimeError("GLM overlay worker did not write a response.")
                response = json.loads(response_path.read_text(encoding="utf-8"))
                if not response.get("ok"):
                    return empty_glm_partial_dependence_warning(str(response.get("error") or "GLM overlay worker failed.")), started
                result = response.get("result")
                if not isinstance(result, dict):
                    return empty_glm_partial_dependence_warning("GLM overlay worker returned an invalid response."), started
                return result, started
            except Exception as exc:
                self.stop()
                fallback = build_glm_partial_dependence_overlay_one_shot(
                    dataset,
                    request,
                    feature_spec=feature_spec,
                    x_col=x_col,
                    x_sql=x_sql,
                    x_group_kind=x_group_kind,
                    denominator=denominator,
                )
                fallback.setdefault("warnings", []).append(f"Persistent GLM overlay worker restarted after failure: {exc}")
                fallback.setdefault("timings", {})["worker_mode"] = "one_shot_fallback"
                return fallback, True
            finally:
                for path in (locals().get("request_path"), locals().get("response_path")):
                    if isinstance(path, Path):
                        try:
                            path.unlink(missing_ok=True)
                        except OSError:
                            pass

    def _ensure_process(self) -> tuple[subprocess.Popen[str], bool]:
        process = self._process
        if process is not None and process.poll() is None:
            return process, False
        self.stop()
        tmp_dir = Path(tempfile.mkdtemp(prefix="lucidum-glm-overlay-hot-"))
        process = subprocess.Popen(
            [sys.executable, "-m", "py_lucidum.tools.glm.overlay_worker", "--server"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            bufsize=1,
            env={
                **os.environ,
                "PY_LUCIDUM_SKIP_LIGHTGBM_PRELOAD": "1",
                "PY_LUCIDUM_GLM_OVERLAY_WORKER": "1",
            },
        )
        self._tmp_dir = tmp_dir
        self._process = process
        return process, True

    def _wait_for_ack(self, process: subprocess.Popen[str], request_id: str) -> None:
        if process.stdout is None:
            raise RuntimeError("GLM overlay worker stdout is not available.")
        while True:
            line = process.stdout.readline()
            if not line:
                raise RuntimeError(f"GLM overlay worker exited unexpectedly with code {process.poll()}.")
            try:
                message = json.loads(line)
            except json.JSONDecodeError:
                continue
            if str(message.get("request_id") or "") != request_id:
                continue
            if not message.get("ok", True):
                raise RuntimeError(str(message.get("error") or "GLM overlay worker failed."))
            return

    def stop(self) -> None:
        process = self._process
        self._process = None
        if process is not None and process.poll() is None:
            try:
                if process.stdin is not None:
                    process.stdin.write(json.dumps({"command": "shutdown"}) + "\n")
                    process.stdin.flush()
                process.wait(timeout=2)
            except Exception:
                process.terminate()
                try:
                    process.wait(timeout=2)
                except Exception:
                    process.kill()
        if process is not None:
            for pipe in (process.stdin, process.stdout):
                try:
                    if pipe is not None:
                        pipe.close()
                except OSError:
                    pass
        tmp_dir = self._tmp_dir
        self._tmp_dir = None
        if tmp_dir is not None:
            shutil.rmtree(tmp_dir, ignore_errors=True)


_PERSISTENT_GLM_OVERLAY_WORKER = PersistentGlmOverlayWorker()


def persistent_glm_overlay_worker() -> PersistentGlmOverlayWorker:
    return _PERSISTENT_GLM_OVERLAY_WORKER


def stop_persistent_glm_overlay_worker() -> None:
    _PERSISTENT_GLM_OVERLAY_WORKER.stop()


atexit.register(stop_persistent_glm_overlay_worker)


def _build_glm_partial_dependence_overlay_impl(
    dataset: Dataset,
    request: dict[str, Any],
    *,
    feature_spec: Any,
    x_col: str,
    x_sql: dict[str, str],
    x_group_kind: str,
    denominator: dict[str, str | None],
) -> dict[str, Any]:
    store = GlmModelStore(dataset.path)
    model_id = store.active_model_id()
    if not model_id:
        return empty_glm_partial_dependence_warning("No active GLM is available for GLM overlay.")
    estimator_path = store.artifact_path(model_id, "estimator")
    if not estimator_path.exists():
        return empty_glm_partial_dependence_warning("Rebuild the active GLM before using GLM overlay; estimator.pkl is missing.")
    try:
        _glum, _glr, _glrcv, np, pd = glm_dependencies()
    except MissingGlmDependency as exc:
        return empty_glm_partial_dependence_warning(str(exc))

    with estimator_path.open("rb") as handle:
        estimator = pickle.load(handle)
    manifest = store.manifest(model_id)
    source_columns = store.source_columns(manifest)
    if x_col not in source_columns:
        return empty_glm_partial_dependence_warning(f"The active GLM source does not include {x_col}.")

    context = formula_context(np)
    offset_terms = [str(term) for term in (manifest.get("offset_terms") or manifest.get("formula", {}).get("offset_terms") or [])]
    groups = _term_groups(estimator, offset_terms, source_columns)
    all_features = sorted({feature for features in groups for feature in features})
    x_interaction_groups = [tuple(features) for features in groups if x_col in features and len(features) > 1]
    interaction_partners = sorted({feature for features in x_interaction_groups for feature in features if feature != x_col})
    interaction = bool(x_interaction_groups)
    relation = glm_overlay_relation_sql(store, model_id, manifest, source_columns)
    try:
        filter_sql = dataset.normalise_filter_for_relation(request.get("filter"), relation)
    except ValueError as exc:
        return empty_glm_partial_dependence_warning(f"GLM overlay could not use the current filter: {exc}")

    overlay_denominator = normalise_glm_overlay_denominator(denominator, source_columns)
    target_mean = glm_prediction_mean(dataset, relation, overlay_denominator, filter_sql)
    initial_rows = glm_x_group_rows(dataset, relation, x_sql, overlay_denominator, filter_sql)
    group_mapping = glm_low_weight_group_mapping(initial_rows, x_group_kind, str(request.get("lowGroup") or "0"))
    x_rows = [row for row in initial_rows if usable_x_value(row)]
    x_value_count = len(x_rows)
    if not x_rows:
        return {
            "mode": "glm",
            "model_id": model_id,
            "feature": x_col,
            "method": "sampled_marginal" if interaction else "base_profile",
            "percentiles": [50],
            "rows": [],
            "warnings": ["No GLM overlay x-axis groups matched the current chart selection."],
            "scale": {"method": "none", "target": None, "source_mean": None},
            "sample": {},
            "transform": {"mode": str(request.get("transform") or "none")},
        }

    kinds = {column.name: column.kind for column in dataset.valid_schema_columns()}
    spec_rows = _feature_spec_map(feature_spec)
    transform_bounds = _feature_transform_bounds(estimator, source_columns)
    required_columns = glm_required_columns(source_columns, manifest, all_features, offset_terms, x_col, overlay_denominator)
    base = glm_base_row_from_relation(
        dataset,
        relation,
        required_columns,
        manifest,
        denominator=overlay_denominator,
        filter_sql=filter_sql,
        kinds=kinds,
        spec_rows=spec_rows,
        transform_bounds=transform_bounds,
    )
    method = "base_profile"
    sample: dict[str, Any] = {
        "population_row_count": None,
        "sample_row_count": 0,
        "x_value_count": int(x_value_count),
        "prediction_cell_count": int(x_value_count),
        "max_sample_rows": MAX_GLM_OVERLAY_SAMPLE_ROWS,
        "max_prediction_cells": MAX_GLM_OVERLAY_PREDICTION_CELLS,
    }
    fallback_reason = ""
    if not interaction:
        source_rows = base_profile_rows(
            estimator,
            manifest,
            base,
            x_rows,
            x_col=x_col,
            denominator=overlay_denominator,
            offset_terms=offset_terms,
            context=context,
            np=np,
            pd=pd,
        )
    else:
        collapsed = simple_glm_interaction_partners(x_interaction_groups, x_col)
        if collapsed:
            context_rows, fallback_reason = collapsed_interaction_context_rows(
                dataset,
                relation,
                collapsed,
                manifest,
                denominator=overlay_denominator,
                filter_sql=filter_sql,
                kinds=kinds,
                spec_rows=spec_rows,
            )
        else:
            context_rows = []
            fallback_reason = "GLM overlay uses sampled PDP because the selected feature has complex interaction terms."
        collapsed_prediction_cells = len(context_rows) * x_value_count
        if context_rows and collapsed_prediction_cells <= MAX_GLM_OVERLAY_COLLAPSED_PREDICTION_CELLS:
            method = "collapsed_marginal"
            sample.update(
                {
                    "context_row_count": int(len(context_rows)),
                    "interaction_partners": interaction_partners,
                    "prediction_cell_count": int(collapsed_prediction_cells),
                    "max_collapsed_context_rows": MAX_GLM_OVERLAY_COLLAPSED_CONTEXT_ROWS,
                    "max_collapsed_prediction_cells": MAX_GLM_OVERLAY_COLLAPSED_PREDICTION_CELLS,
                    "numeric_interaction_bins": GLM_OVERLAY_NUMERIC_INTERACTION_BINS,
                    "max_categorical_interaction_levels": MAX_GLM_OVERLAY_CATEGORICAL_INTERACTION_LEVELS,
                }
            )
            source_rows = collapsed_marginal_rows(
                estimator,
                manifest,
                base,
                context_rows,
                x_rows,
                x_col=x_col,
                denominator=overlay_denominator,
                offset_terms=offset_terms,
                context=context,
                np=np,
                pd=pd,
            )
        else:
            if context_rows and collapsed_prediction_cells > MAX_GLM_OVERLAY_COLLAPSED_PREDICTION_CELLS:
                fallback_reason = (
                    "GLM overlay uses sampled PDP because the collapsed interaction grid would require "
                    f"{collapsed_prediction_cells:,} prediction cells."
                )
            method = "sampled_marginal"
            seed = glm_overlay_seed(manifest)
            sample_limit = overlay_sample_limit(x_value_count=x_value_count)
            sample_frame, population_row_count = glm_sample_frame(
                dataset,
                relation,
                required_columns,
                overlay_denominator,
                filter_sql,
                seed=seed,
                sample_limit=sample_limit,
            )
            if sample_frame.empty:
                return empty_glm_partial_dependence_warning("No GLM overlay rows matched the current chart selection.")
            source_rows = sampled_marginal_rows(
                estimator,
                manifest,
                sample_frame,
                x_rows,
                x_col=x_col,
                denominator=overlay_denominator,
                offset_terms=offset_terms,
                context=context,
                np=np,
                pd=pd,
            )
            prediction_cell_count = len(sample_frame) * x_value_count
            sample.update(
                {
                    "population_row_count": int(population_row_count),
                    "sample_row_count": int(len(sample_frame)),
                    "interaction_partners": interaction_partners,
                    "prediction_cell_count": int(prediction_cell_count),
                    "max_collapsed_context_rows": MAX_GLM_OVERLAY_COLLAPSED_CONTEXT_ROWS,
                    "max_collapsed_prediction_cells": MAX_GLM_OVERLAY_COLLAPSED_PREDICTION_CELLS,
                    "numeric_interaction_bins": GLM_OVERLAY_NUMERIC_INTERACTION_BINS,
                    "max_categorical_interaction_levels": MAX_GLM_OVERLAY_CATEGORICAL_INTERACTION_LEVELS,
                    "seed": seed,
                    **({"fallback_reason": fallback_reason} if fallback_reason else {}),
                }
            )
    rows = aggregate_source_rows(source_rows, group_mapping)
    scale = scale_glm_overlay_rows(rows, target_mean, manifest=manifest)
    if x_group_kind == "numeric":
        clean_numeric_labels(rows, request.get("bandWidth"))
    warnings: list[str] = []
    if scale.get("warning"):
        warnings.append(str(scale["warning"]))
    if fallback_reason and method == "sampled_marginal":
        warnings.append(fallback_reason)
    if method == "sampled_marginal" and int(sample.get("population_row_count") or 0) > int(sample.get("sample_row_count") or 0):
        warnings.append(
            "GLM overlay used a deterministic sample of "
            f"{int(sample.get('sample_row_count') or 0):,} from {int(sample.get('population_row_count') or 0):,} eligible rows."
        )
    return {
        "mode": "glm",
        "model_id": model_id,
        "feature": x_col,
        "method": method,
        "percentiles": [50],
        "rows": rows,
        "warnings": warnings,
        "scale": {key: value for key, value in scale.items() if key != "warning"},
        "sample": sample,
        "transform": {"mode": str(request.get("transform") or "none")},
    }


def glm_prediction_mean(
    dataset: Dataset,
    relation: str,
    denominator: dict[str, str | None],
    filter_sql: str,
) -> float | int | None:
    checks = ["TRY_CAST(glm_prediction AS DOUBLE) IS NOT NULL"]
    denominator_condition = denominator_valid_condition([], denominator)
    if denominator_condition != "TRUE":
        checks.append(denominator_condition)
    valid_condition = " AND ".join(checks)
    weight_expr = weighted_value_sql(denominator, valid_condition)
    where_sql = f"\nWHERE ({filter_sql})" if filter_sql else ""
    sql = f"""
SELECT
  SUM(CASE WHEN {valid_condition} THEN TRY_CAST(glm_prediction AS DOUBLE) ELSE NULL END) AS numerator,
  COALESCE(SUM({weight_expr}), 0) AS denominator
FROM {relation}
{where_sql}
"""
    row = dataset.con.execute(sql).fetchone()
    numerator = json_number(row[0] if row else None)
    denominator_value = json_number(row[1] if row else None)
    if numerator is None or denominator_value in (None, 0):
        return None
    return json_number(float(numerator) / float(denominator_value))


def glm_overlay_relation_sql(store: GlmModelStore, model_id: str, manifest: dict[str, Any], source_columns: list[str]) -> str:
    prediction_path = store.artifact_path(model_id, "predictions")
    columns_sql = ",\n    ".join(quote_ident(name) for name in source_columns)
    suffix = f",\n    {columns_sql}" if columns_sql else ""
    return f"""(
SELECT
  base.*,
  prediction.glm_prediction
FROM (
  SELECT
    ROW_NUMBER() OVER () AS __lucidum_row_id{suffix}
  FROM {store.dataset_relation_sql()}
) base
INNER JOIN read_parquet({sql_literal(str(prediction_path))}) prediction USING (__lucidum_row_id)
)"""


def normalise_glm_overlay_denominator(denominator: dict[str, str | None], source_columns: list[str]) -> dict[str, str | None]:
    column = str(denominator.get("column") or "").strip()
    if column and column in source_columns:
        return denominator
    return {"column": None, "label": "Average row value", "bar_label": "Row count"}


def glm_valid_where_sql(
    denominator: dict[str, str | None],
    filter_sql: str,
    *,
    extra_conditions: list[str] | None = None,
) -> str:
    where_parts = ["TRY_CAST(glm_prediction AS DOUBLE) IS NOT NULL"]
    denominator_condition = denominator_valid_condition([], denominator)
    if denominator_condition != "TRUE":
        where_parts.append(f"({denominator_condition})")
    if filter_sql:
        where_parts.append(f"({filter_sql})")
    where_parts.extend(condition for condition in (extra_conditions or []) if condition)
    return f"WHERE {' AND '.join(where_parts)}"


def glm_base_row_from_relation(
    dataset: Dataset,
    relation: str,
    columns: list[str],
    manifest: dict[str, Any],
    *,
    denominator: dict[str, str | None],
    filter_sql: str,
    kinds: dict[str, str],
    spec_rows: dict[str, dict[str, Any]],
    transform_bounds: dict[str, dict[str, float]],
) -> dict[str, Any]:
    base = {TARGET_COLUMN: 0.0}
    for feature in columns:
        base[feature] = glm_base_value_from_relation(
            dataset,
            relation,
            feature,
            kind=kinds.get(feature, "categorical"),
            spec_row=spec_rows.get(feature, {}),
            bounds=transform_bounds.get(feature, {}),
            denominator=denominator,
            filter_sql=filter_sql,
        )
    model_denominator = str(manifest.get("denominator_column") or "").strip()
    if model_denominator and model_denominator not in base and model_denominator in kinds:
        base[model_denominator] = glm_base_value_from_relation(
            dataset,
            relation,
            model_denominator,
            kind=kinds.get(model_denominator, "numeric"),
            spec_row=spec_rows.get(model_denominator, {}),
            bounds=transform_bounds.get(model_denominator, {}),
            denominator=denominator,
            filter_sql=filter_sql,
        )
    return base


def glm_base_value_from_relation(
    dataset: Dataset,
    relation: str,
    feature: str,
    *,
    kind: str,
    spec_row: dict[str, Any],
    bounds: dict[str, float],
    denominator: dict[str, str | None],
    filter_sql: str,
) -> Any:
    raw = str(spec_row.get("base") or "").strip()
    col = quote_ident(feature)
    if is_numeric_kind(kind):
        raw_number = _as_number(raw)
        if raw and raw_number is not None:
            value, _clipped = _clip_numeric_bound(feature, "base", float(raw_number), bounds, "feature_spec")
            return _json_value(value)
        where_sql = glm_valid_where_sql(denominator, filter_sql, extra_conditions=[f"TRY_CAST({col} AS DOUBLE) IS NOT NULL"])
        sql = f"SELECT MEDIAN(TRY_CAST({col} AS DOUBLE)) AS value FROM {relation} {where_sql}"
        row = dataset.con.execute(sql).fetchone()
        inferred = json_number(row[0] if row else None)
        value, _clipped = _clip_numeric_bound(feature, "base", float(inferred if inferred is not None else 0.0), bounds, "inferred")
        return _json_value(value)
    if raw:
        return raw
    where_sql = glm_valid_where_sql(denominator, filter_sql, extra_conditions=[f"{col} IS NOT NULL"])
    sql = f"""
SELECT {col} AS value, COUNT(*) AS row_count
FROM {relation}
{where_sql}
GROUP BY value
ORDER BY row_count DESC, CAST(value AS VARCHAR)
LIMIT 1
"""
    row = dataset.con.execute(sql).fetchone()
    return _json_value(row[0]) if row else ""


def simple_glm_interaction_partners(x_interaction_groups: list[tuple[str, ...]], x_col: str) -> list[str]:
    partners: set[str] = set()
    for group in x_interaction_groups:
        if len(group) != 2:
            return []
        partners.update(feature for feature in group if feature != x_col)
    return sorted(partners) if len(partners) == 1 else []


def collapsed_interaction_context_rows(
    dataset: Dataset,
    relation: str,
    partners: list[str],
    manifest: dict[str, Any],
    *,
    denominator: dict[str, str | None],
    filter_sql: str,
    kinds: dict[str, str],
    spec_rows: dict[str, dict[str, Any]],
) -> tuple[list[dict[str, Any]], str]:
    if len(partners) != 1:
        return [], "GLM overlay uses sampled PDP because the selected feature has multiple interaction partners."
    partner = partners[0]
    if partner not in kinds:
        return [], f"GLM overlay uses sampled PDP because interaction partner {partner} is not available."
    numeric_partner = is_numeric_kind(kinds.get(partner, ""))
    if numeric_partner:
        rows = collapsed_numeric_interaction_context_rows(
            dataset,
            relation,
            partner,
            manifest,
            denominator=denominator,
            filter_sql=filter_sql,
            spec_row=spec_rows.get(partner, {}),
        )
    else:
        rows = collapsed_categorical_interaction_context_rows(
            dataset,
            relation,
            partner,
            manifest,
            denominator=denominator,
            filter_sql=filter_sql,
        )
    if not numeric_partner and len(rows) > MAX_GLM_OVERLAY_CATEGORICAL_INTERACTION_LEVELS:
        return [], (
            "GLM overlay uses sampled PDP because categorical interaction partner "
            f"{partner} has more than {MAX_GLM_OVERLAY_CATEGORICAL_INTERACTION_LEVELS:,} observed levels."
        )
    if len(rows) > MAX_GLM_OVERLAY_COLLAPSED_CONTEXT_ROWS:
        return [], (
            "GLM overlay uses sampled PDP because the collapsed interaction context would require "
            f"{len(rows):,} rows."
        )
    if not rows:
        return [], "GLM overlay uses sampled PDP because no collapsed interaction context rows matched the current chart selection."
    return rows, ""


def collapsed_numeric_interaction_context_rows(
    dataset: Dataset,
    relation: str,
    partner: str,
    manifest: dict[str, Any],
    *,
    denominator: dict[str, str | None],
    filter_sql: str,
    spec_row: dict[str, Any],
) -> list[dict[str, Any]]:
    raw_min = _as_number(spec_row.get("min"))
    raw_max = _as_number(spec_row.get("max"))
    raw_band = _as_number(spec_row.get("banding"))
    if raw_min is not None and raw_max is not None and raw_band is not None and raw_band > 0:
        count = int(math.floor(abs(float(raw_max) - float(raw_min)) / float(raw_band))) + 1
        if 0 < count <= GLM_OVERLAY_NUMERIC_INTERACTION_BINS:
            return collapsed_numeric_band_context_rows(
                dataset,
                relation,
                partner,
                manifest,
                denominator=denominator,
                filter_sql=filter_sql,
                minimum=float(min(raw_min, raw_max)),
                maximum=float(max(raw_min, raw_max)),
                band=float(raw_band),
            )
    return collapsed_numeric_quantile_context_rows(
        dataset,
        relation,
        partner,
        manifest,
        denominator=denominator,
        filter_sql=filter_sql,
    )


def collapsed_context_sql_fragments(
    manifest: dict[str, Any],
    denominator: dict[str, str | None],
) -> dict[str, str]:
    model_denominator = str(manifest.get("denominator_column") or "").strip()
    selected_denominator = str(denominator.get("column") or "").strip()
    model_expr = f"TRY_CAST({quote_ident(model_denominator)} AS DOUBLE)" if model_denominator else "CAST(NULL AS DOUBLE)"
    selected_expr = f"TRY_CAST({quote_ident(selected_denominator)} AS DOUBLE)" if selected_denominator else "CAST(NULL AS DOUBLE)"
    return {
        "model_denominator": model_denominator,
        "selected_denominator": selected_denominator,
        "model_expr": model_expr,
        "selected_expr": selected_expr,
        "denominator_weight_expr": weighted_value_sql(denominator, "TRUE"),
    }


def collapsed_context_extra_conditions(partner_expr: str, fragments: dict[str, str]) -> list[str]:
    conditions = [f"{partner_expr} IS NOT NULL"]
    if fragments["model_denominator"]:
        conditions.append(f"{fragments['model_expr']} IS NOT NULL")
    return conditions


def collapsed_numeric_quantile_context_rows(
    dataset: Dataset,
    relation: str,
    partner: str,
    manifest: dict[str, Any],
    *,
    denominator: dict[str, str | None],
    filter_sql: str,
) -> list[dict[str, Any]]:
    fragments = collapsed_context_sql_fragments(manifest, denominator)
    partner_expr = f"TRY_CAST({quote_ident(partner)} AS DOUBLE)"
    where_sql = glm_valid_where_sql(denominator, filter_sql, extra_conditions=collapsed_context_extra_conditions(partner_expr, fragments))
    sql = f"""
WITH eligible AS (
  SELECT
    ROW_NUMBER() OVER () AS __rownum,
    {partner_expr} AS __partner_value,
    {fragments['model_expr']} AS __model_denominator,
    {fragments['selected_expr']} AS __selected_denominator,
    {fragments['denominator_weight_expr']} AS __denominator_weight
  FROM {relation}
  {where_sql}
),
quantiles AS (
  SELECT
    *,
    NTILE({GLM_OVERLAY_NUMERIC_INTERACTION_BINS}) OVER (ORDER BY __partner_value, __rownum) AS __partner_group
  FROM eligible
)
SELECT
  AVG(__partner_value) AS partner_value,
  COUNT(*) AS row_count,
  COALESCE(SUM(__denominator_weight), 0) AS denominator_weight,
  AVG(__model_denominator) AS model_denominator,
  AVG(__selected_denominator) AS selected_denominator
FROM quantiles
GROUP BY __partner_group
ORDER BY MIN(__partner_value)
LIMIT {MAX_GLM_OVERLAY_COLLAPSED_CONTEXT_ROWS + 1}
"""
    return normalise_collapsed_context_rows(dataset.con.execute(sql).fetchall(), dataset.con.description, partner)


def collapsed_numeric_band_context_rows(
    dataset: Dataset,
    relation: str,
    partner: str,
    manifest: dict[str, Any],
    *,
    denominator: dict[str, str | None],
    filter_sql: str,
    minimum: float,
    maximum: float,
    band: float,
) -> list[dict[str, Any]]:
    fragments = collapsed_context_sql_fragments(manifest, denominator)
    partner_expr = f"TRY_CAST({quote_ident(partner)} AS DOUBLE)"
    clipped = f"LEAST(GREATEST({partner_expr}, {minimum}), {maximum})"
    group_expr = f"FLOOR(({clipped}) / {band}) * {band}"
    where_sql = glm_valid_where_sql(denominator, filter_sql, extra_conditions=collapsed_context_extra_conditions(partner_expr, fragments))
    sql = f"""
SELECT
  AVG({partner_expr}) AS partner_value,
  COUNT(*) AS row_count,
  COALESCE(SUM({fragments['denominator_weight_expr']}), 0) AS denominator_weight,
  AVG({fragments['model_expr']}) AS model_denominator,
  AVG({fragments['selected_expr']}) AS selected_denominator
FROM {relation}
{where_sql}
GROUP BY {group_expr}
ORDER BY MIN({group_expr})
LIMIT {MAX_GLM_OVERLAY_COLLAPSED_CONTEXT_ROWS + 1}
"""
    return normalise_collapsed_context_rows(dataset.con.execute(sql).fetchall(), dataset.con.description, partner)


def collapsed_categorical_interaction_context_rows(
    dataset: Dataset,
    relation: str,
    partner: str,
    manifest: dict[str, Any],
    *,
    denominator: dict[str, str | None],
    filter_sql: str,
) -> list[dict[str, Any]]:
    fragments = collapsed_context_sql_fragments(manifest, denominator)
    partner_expr = quote_ident(partner)
    where_sql = glm_valid_where_sql(denominator, filter_sql, extra_conditions=collapsed_context_extra_conditions(partner_expr, fragments))
    limit = min(MAX_GLM_OVERLAY_CATEGORICAL_INTERACTION_LEVELS + 1, MAX_GLM_OVERLAY_COLLAPSED_CONTEXT_ROWS + 1)
    sql = f"""
SELECT
  {partner_expr} AS partner_value,
  COUNT(*) AS row_count,
  COALESCE(SUM({fragments['denominator_weight_expr']}), 0) AS denominator_weight,
  AVG({fragments['model_expr']}) AS model_denominator,
  AVG({fragments['selected_expr']}) AS selected_denominator
FROM {relation}
{where_sql}
GROUP BY partner_value
ORDER BY row_count DESC, CAST(partner_value AS VARCHAR)
LIMIT {limit}
"""
    rows = normalise_collapsed_context_rows(dataset.con.execute(sql).fetchall(), dataset.con.description, partner)
    return rows


def normalise_collapsed_context_rows(raw_rows: list[tuple[Any, ...]], description: Any, partner: str) -> list[dict[str, Any]]:
    columns = [item[0] for item in description]
    rows: list[dict[str, Any]] = []
    for raw in raw_rows:
        row = dict(zip(columns, raw))
        denominator_weight = json_number(row.get("denominator_weight"))
        row_count = int(row.get("row_count") or 0)
        if row_count <= 0:
            continue
        rows.append(
            {
                "values": {partner: _json_value(row.get("partner_value"))},
                "row_count": row_count,
                "denominator_weight": json_number(denominator_weight if denominator_weight is not None else row_count) or 0,
                "model_denominator": json_number(row.get("model_denominator")),
                "selected_denominator": json_number(row.get("selected_denominator")),
            }
        )
    return rows


def glm_x_group_rows(
    dataset: Dataset,
    relation: str,
    x_sql: dict[str, str],
    denominator: dict[str, str | None],
    filter_sql: str,
) -> list[dict[str, Any]]:
    valid_condition = denominator_valid_condition([], denominator)
    weight_expr = weighted_value_sql(denominator, valid_condition)
    where_sql = f"\n  WHERE ({filter_sql})" if filter_sql else ""
    quantile_cte = ""
    keyed_from = "base"
    rownum_expr = "__rownum"
    source_columns = "*"
    x_value_expr = "x_key"
    x_value_select = "MIN(__x_value) AS x_value"
    if x_sql.get("quantile_count"):
        quantile_cte = f""",
quantiles AS (
  SELECT
    __rownum,
    NTILE({x_sql['quantile_count']}) OVER (ORDER BY __x_raw, __rownum) AS __x_quantile
  FROM (
    SELECT
      __rownum,
      {x_sql['raw']} AS __x_raw
    FROM base
    WHERE {x_sql['raw']} IS NOT NULL
  ) non_missing
)"""
        keyed_from = "base LEFT JOIN quantiles USING (__rownum)"
        rownum_expr = "base.__rownum"
        source_columns = "base.*"
        x_value_expr = x_sql["raw"]
        x_value_select = "AVG(__x_value) AS x_value"
    sql = f"""
WITH base AS (
  SELECT ROW_NUMBER() OVER () AS __rownum, * FROM {relation}{where_sql}
){quantile_cte},
keyed AS (
  SELECT
    {rownum_expr} AS __rownum,
    {x_sql['key']} AS x_key,
    {x_sql['label']} AS x_label,
    {x_sql['sort']} AS x_sort,
    {x_value_expr} AS __x_value,
    {weight_expr} AS __weight_value,
    {source_columns}
  FROM {keyed_from}
),
valid AS (
  SELECT *
  FROM keyed
  WHERE __weight_value IS NOT NULL
    AND TRY_CAST(glm_prediction AS DOUBLE) IS NOT NULL
)
SELECT
  x_label,
  MIN(x_sort) AS x_sort,
  MIN(__rownum) AS original_order,
  COALESCE(SUM(__weight_value), 0) AS volume,
  {x_value_select}
FROM valid
GROUP BY x_label
"""
    cursor = dataset.con.execute(sql)
    raw_rows = [dict(zip([d[0] for d in cursor.description], row)) for row in cursor.fetchall()]
    return [
        {
            "x": str(row.get("x_label")),
            "x_sort": row.get("x_sort"),
            "original_order": int(row.get("original_order") or 0),
            "volume": json_number(row.get("volume")) or 0,
            "x_value": _json_value(row.get("x_value")),
            "is_tail": False,
        }
        for row in raw_rows
    ]


def glm_required_columns(
    source_columns: list[str],
    manifest: dict[str, Any],
    all_features: list[str],
    offset_terms: list[str],
    x_col: str,
    denominator: dict[str, str | None],
) -> list[str]:
    requested = set(all_features)
    requested.add(x_col)
    model_denominator = str(manifest.get("denominator_column") or "").strip()
    if model_denominator:
        requested.add(model_denominator)
    selected_denominator = str(denominator.get("column") or "").strip()
    if selected_denominator:
        requested.add(selected_denominator)
    for expression in offset_terms:
        requested.update(_column_tokens(expression, source_columns))
    return [column for column in source_columns if column in requested]


def glm_sample_frame(
    dataset: Dataset,
    relation: str,
    columns: list[str],
    denominator: dict[str, str | None],
    filter_sql: str,
    *,
    seed: int,
    sample_limit: int,
) -> tuple[Any, int]:
    valid_condition = denominator_valid_condition([], denominator)
    where_parts = ["TRY_CAST(glm_prediction AS DOUBLE) IS NOT NULL"]
    if valid_condition != "TRUE":
        where_parts.append(f"({valid_condition})")
    if filter_sql:
        where_parts.append(f"({filter_sql})")
    where_sql = f"WHERE {' AND '.join(where_parts)}" if where_parts else ""
    count = int(dataset.con.execute(f"SELECT COUNT(*) FROM {relation} {where_sql}").fetchone()[0] or 0)
    select_columns = ["__lucidum_row_id", *[quote_ident(name) for name in columns]]
    select_sql = ",\n  ".join(select_columns)
    sql = f"""
SELECT
  {select_sql}
FROM {relation}
{where_sql}
ORDER BY hash(__lucidum_row_id + {int(seed)}), __lucidum_row_id
LIMIT {max(1, int(sample_limit))}
"""
    return dataset.con.execute(sql).fetchdf(), count


def overlay_sample_limit(*, x_value_count: int) -> int:
    if x_value_count <= 0:
        return 1
    by_cells = max(1, MAX_GLM_OVERLAY_PREDICTION_CELLS // max(1, int(x_value_count)))
    return min(MAX_GLM_OVERLAY_SAMPLE_ROWS, by_cells)


def sampled_marginal_rows(
    estimator: Any,
    manifest: dict[str, Any],
    sample_frame: Any,
    x_rows: list[dict[str, Any]],
    *,
    x_col: str,
    denominator: dict[str, str | None],
    offset_terms: list[str],
    context: dict[str, Any],
    np: Any,
    pd: Any,
) -> list[dict[str, Any]]:
    if sample_frame.empty:
        return []
    chunk_size = max(1, min(len(x_rows), GLM_OVERLAY_CHUNK_CELLS // max(1, len(sample_frame))))
    by_x: dict[str, dict[str, float]] = {}
    for start in range(0, len(x_rows), chunk_size):
        chunk = x_rows[start : start + chunk_size]
        frames = []
        for row in chunk:
            frame = sample_frame.copy()
            frame["__overlay_x"] = str(row["x"])
            frame[x_col] = row["x_value"]
            frames.append(frame)
        block = pd.concat(frames, ignore_index=True)
        numerators = predict_glm_numerators(estimator, manifest, block, offset_terms, context, np, pd)
        weights = overlay_weights(block, denominator, np, pd)
        for x_value, numerator, weight in zip(block["__overlay_x"], numerators, weights):
            if not math.isfinite(float(weight or 0)):
                continue
            value = json_number(numerator)
            if value is None:
                continue
            bucket = by_x.setdefault(str(x_value), {"num": 0.0, "den": 0.0})
            bucket["num"] += float(value)
            bucket["den"] += float(weight)
    rows: list[dict[str, Any]] = []
    for row in x_rows:
        bucket = by_x.get(str(row["x"]), {"num": 0.0, "den": 0.0})
        rows.append({**row, "p50": json_number(bucket["num"] / bucket["den"]) if bucket["den"] else None})
    return rows


def collapsed_marginal_rows(
    estimator: Any,
    manifest: dict[str, Any],
    base: dict[str, Any],
    context_rows: list[dict[str, Any]],
    x_rows: list[dict[str, Any]],
    *,
    x_col: str,
    denominator: dict[str, str | None],
    offset_terms: list[str],
    context: dict[str, Any],
    np: Any,
    pd: Any,
) -> list[dict[str, Any]]:
    if not context_rows:
        return []
    model_denominator = str(manifest.get("denominator_column") or "").strip()
    selected_denominator = str(denominator.get("column") or "").strip()
    chunk_size = max(1, min(len(x_rows), GLM_OVERLAY_CHUNK_CELLS // max(1, len(context_rows))))
    by_x: dict[str, dict[str, float]] = {}
    for start in range(0, len(x_rows), chunk_size):
        chunk = x_rows[start : start + chunk_size]
        rows: list[dict[str, Any]] = []
        for x_row in chunk:
            for context_row in context_rows:
                frame_row = dict(base)
                frame_row.update(context_row.get("values") or {})
                frame_row[x_col] = x_row["x_value"]
                if model_denominator and context_row.get("model_denominator") is not None:
                    frame_row[model_denominator] = context_row.get("model_denominator")
                if selected_denominator and context_row.get("selected_denominator") is not None:
                    frame_row[selected_denominator] = context_row.get("selected_denominator")
                frame_row["__overlay_x"] = str(x_row["x"])
                frame_row["__overlay_numerator_weight"] = context_row.get("row_count") or 0
                frame_row["__overlay_denominator_weight"] = context_row.get("denominator_weight") or 0
                rows.append(frame_row)
        block = pd.DataFrame(rows)
        numerators = predict_glm_numerators(estimator, manifest, block, offset_terms, context, np, pd)
        for x_value, numerator, numerator_weight, denominator_weight in zip(
            block["__overlay_x"],
            numerators,
            block["__overlay_numerator_weight"],
            block["__overlay_denominator_weight"],
        ):
            value = json_number(numerator)
            num_weight = json_number(numerator_weight)
            den_weight = json_number(denominator_weight)
            if value is None or num_weight is None or den_weight is None:
                continue
            if not math.isfinite(float(num_weight)) or not math.isfinite(float(den_weight)):
                continue
            bucket = by_x.setdefault(str(x_value), {"num": 0.0, "den": 0.0})
            bucket["num"] += float(value) * float(num_weight)
            bucket["den"] += float(den_weight)
    rows = []
    for x_row in x_rows:
        bucket = by_x.get(str(x_row["x"]), {"num": 0.0, "den": 0.0})
        rows.append({**x_row, "p50": json_number(bucket["num"] / bucket["den"]) if bucket["den"] else None})
    return rows


def base_profile_rows(
    estimator: Any,
    manifest: dict[str, Any],
    base: dict[str, Any],
    x_rows: list[dict[str, Any]],
    *,
    x_col: str,
    denominator: dict[str, str | None],
    offset_terms: list[str],
    context: dict[str, Any],
    np: Any,
    pd: Any,
) -> list[dict[str, Any]]:
    frame = pd.DataFrame([dict(base, **{x_col: row["x_value"]}) for row in x_rows])
    numerators = predict_glm_numerators(estimator, manifest, frame, offset_terms, context, np, pd)
    weights = overlay_weights(frame, denominator, np, pd)
    rows: list[dict[str, Any]] = []
    for row, numerator, weight in zip(x_rows, numerators, weights):
        value = json_number(numerator)
        denominator_value = json_number(weight)
        rows.append({**row, "p50": json_number(float(value) / float(denominator_value)) if value is not None and denominator_value not in (None, 0) else None})
    return rows


def predict_glm_numerators(
    estimator: Any,
    manifest: dict[str, Any],
    frame: Any,
    offset_terms: list[str],
    context: dict[str, Any],
    np: Any,
    pd: Any,
) -> Any:
    work = frame.copy()
    add_internal_intercept_column(work, internal_intercept_column_from_manifest(manifest))
    work[TARGET_COLUMN] = 0.0
    valid = pd.Series(True, index=work.index)
    predict_kwargs: dict[str, Any] = {"context": context}
    offset_values = offset_values_for_frame(work, offset_terms, context, np, pd)
    if offset_values is not None:
        offset_numeric = pd.to_numeric(offset_values, errors="coerce")
        valid = valid & offset_numeric.notna() & np.isfinite(offset_numeric.astype(float))
    model_denominator = str(manifest.get("denominator_column") or "").strip()
    denominator_values = None
    if model_denominator and model_denominator in work.columns:
        denominator_values = pd.to_numeric(work[model_denominator], errors="coerce")
        valid = valid & denominator_values.notna() & np.isfinite(denominator_values.astype(float))
    output = pd.Series(np.nan, index=work.index, dtype=float)
    if not bool(valid.any()):
        return output
    if offset_values is not None:
        predict_kwargs["offset"] = offset_numeric.loc[valid].astype(float).to_numpy()
    predictions = estimator.predict(work.loc[valid].copy(), **predict_kwargs)
    values = pd.to_numeric(pd.Series(predictions, index=work.index[valid]), errors="coerce")
    if denominator_values is not None:
        values = values * denominator_values.loc[valid].astype(float).to_numpy()
    output.loc[valid] = values
    return output


def overlay_weights(frame: Any, denominator: dict[str, str | None], np: Any, pd: Any) -> Any:
    column = str(denominator.get("column") or "").strip()
    if column and column in frame.columns:
        values = pd.to_numeric(frame[column], errors="coerce")
        return values.where(values.notna() & np.isfinite(values.astype(float)), np.nan)
    return pd.Series(1.0, index=frame.index, dtype=float)


def aggregate_source_rows(source_rows: list[dict[str, Any]], group_mapping: list[dict[str, Any]]) -> list[dict[str, Any]]:
    mapping = {str(row.get("source_x")): row for row in group_mapping}
    buckets: dict[str, dict[str, Any]] = {}
    for source in source_rows:
        mapped = mapping.get(str(source.get("x")))
        if mapped is None:
            continue
        label = str(mapped.get("final_x") or "")
        bucket = buckets.setdefault(
            label,
            {
                "x": label,
                "x_sort": mapped.get("final_x_sort"),
                "original_order": int(mapped.get("final_original_order") or 0),
                "volume": 0.0,
                "is_tail": bool(mapped.get("final_is_tail")),
                "__num": 0.0,
                "__den": 0.0,
            },
        )
        volume = float(source.get("volume") or 0)
        bucket["volume"] += volume
        value = json_number(source.get("p50"))
        if value is not None and volume:
            bucket["__num"] += float(value) * volume
            bucket["__den"] += volume
    rows: list[dict[str, Any]] = []
    for bucket in buckets.values():
        rows.append(
            {
                "x": bucket["x"],
                "x_sort": bucket["x_sort"],
                "original_order": bucket["original_order"],
                "volume": json_number(bucket["volume"]) or 0,
                "is_tail": bool(bucket["is_tail"]),
                "p50": json_number(bucket["__num"] / bucket["__den"]) if bucket["__den"] else None,
            }
        )
    return sorted(rows, key=lambda row: int(row.get("original_order") or 0))


def scale_glm_overlay_rows(rows: list[dict[str, Any]], target_mean: Any, *, manifest: dict[str, Any]) -> dict[str, Any]:
    target = json_number(target_mean)
    source_mean = weighted_overlay_average(rows)
    if target is None or source_mean is None:
        return {
            "method": "none",
            "target": target,
            "source_mean": source_mean,
            "warning": "GLM overlay could not be scaled to fitted values.",
        }
    if glm_uses_positive_scale(manifest):
        if source_mean == 0:
            return {
                "method": "none",
                "target": target,
                "source_mean": source_mean,
                "warning": "GLM overlay could not be scaled because the base-profile mean is zero.",
            }
        factor = float(target) / float(source_mean)
        for row in rows:
            value = json_number(row.get("p50"))
            row["p50"] = json_number(float(value) * factor) if value is not None else None
        return {"method": "multiply", "target": json_number(target), "source_mean": json_number(source_mean), "factor": json_number(factor)}
    shift = float(target) - float(source_mean)
    for row in rows:
        value = json_number(row.get("p50"))
        row["p50"] = json_number(float(value) + shift) if value is not None else None
    return {"method": "add", "target": json_number(target), "source_mean": json_number(source_mean), "shift": json_number(shift)}


def weighted_overlay_average(rows: list[dict[str, Any]]) -> float | int | None:
    numerator = 0.0
    denominator = 0.0
    for row in rows:
        value = json_number(row.get("p50"))
        weight = json_number(row.get("volume"))
        if value is None or weight is None:
            continue
        numerator += float(value) * float(weight)
        denominator += float(weight)
    return json_number(numerator / denominator) if denominator else None


def glm_uses_positive_scale(manifest: dict[str, Any]) -> bool:
    family = str(manifest.get("family") or "").strip().lower()
    return family in {"binomial", "gamma", "inverse.gaussian", "negative.binomial", "poisson", "tweedie"}


def usable_x_value(row: dict[str, Any]) -> bool:
    if row.get("x_value") is None:
        return False
    label = str(row.get("x") or "")
    return label not in {"(missing)", "Missing"}


def glm_overlay_seed(manifest: dict[str, Any]) -> int:
    raw = manifest.get("seed") or manifest.get("random_seed")
    try:
        seed = int(raw)
    except (TypeError, ValueError, OverflowError):
        return DEFAULT_GLM_OVERLAY_SAMPLE_SEED
    return seed if seed >= 0 else DEFAULT_GLM_OVERLAY_SAMPLE_SEED


def glm_low_weight_group_mapping(rows: list[dict[str, Any]], x_kind: str, threshold: str) -> list[dict[str, Any]]:
    total_volume = sum(float(row.get("volume") or 0) for row in rows)
    threshold_value = parse_group_threshold(threshold, total_volume)
    normalised = list(rows)
    missing_rows: list[dict[str, Any]] = []
    if x_kind == "quantile":
        missing_rows = [row for row in normalised if row["x"] == "Missing"]
        normalised = [row for row in normalised if row["x"] != "Missing"]
    if threshold_value <= 0 or len(normalised) < 3:
        return [glm_group_mapping_row(row, row) for row in [*normalised, *missing_rows]]
    if x_kind in {"integer", "numeric", "date", "datetime", "quantile"}:
        ordered = sorted(normalised, key=lambda r: (r.get("x_sort") is None, r.get("x_sort")))
        low: list[dict[str, Any]] = []
        high: list[dict[str, Any]] = []
        cumulative = 0.0
        for row in ordered:
            volume = float(row.get("volume") or 0)
            if cumulative + volume <= threshold_value:
                low.append(row)
                cumulative += volume
            else:
                break
        cumulative = 0.0
        for row in reversed(ordered[len(low) :]):
            volume = float(row.get("volume") or 0)
            if cumulative + volume <= threshold_value:
                high.append(row)
                cumulative += volume
            else:
                break
        high = list(reversed(high))
        middle = ordered[len(low) : len(ordered) - len(high) if high else len(ordered)]
        mapping: list[dict[str, Any]] = []
        mapping.extend(glm_tail_mapping_rows(low, "Low tail") if len(low) > 1 else [glm_group_mapping_row(row, row) for row in low])
        mapping.extend(glm_group_mapping_row(row, row) for row in middle)
        mapping.extend(glm_tail_mapping_rows(high, "High tail") if len(high) > 1 else [glm_group_mapping_row(row, row) for row in high])
        mapping.extend(glm_group_mapping_row(row, row) for row in missing_rows)
        return mapping
    rare = [row for row in normalised if float(row.get("volume") or 0) <= threshold_value]
    common = [row for row in normalised if float(row.get("volume") or 0) > threshold_value]
    mapping = [glm_group_mapping_row(row, row) for row in common]
    if len(rare) > 1:
        mapping.extend(glm_tail_mapping_rows(rare, "Other"))
    else:
        mapping.extend(glm_group_mapping_row(row, row) for row in rare)
    return mapping


def parse_group_threshold(value: str, total_volume: float) -> float:
    raw = value.strip().lower()
    if raw in {"", "0", "none", "-"}:
        return 0
    if raw.endswith("%"):
        parsed = json_number(raw[:-1])
        return total_volume * float(parsed) / 100 if parsed else 0
    return float(json_number(raw) or 0)


def glm_group_mapping_row(source: dict[str, Any], final: dict[str, Any], *, label: str | None = None, is_tail: bool | None = None) -> dict[str, Any]:
    return {
        "source_x": source.get("x"),
        "final_x": label if label is not None else final.get("x"),
        "final_x_sort": final.get("x_sort"),
        "final_original_order": final.get("original_order"),
        "final_is_tail": bool(final.get("is_tail")) if is_tail is None else is_tail,
    }


def glm_tail_mapping_rows(rows: list[dict[str, Any]], label: str) -> list[dict[str, Any]]:
    if not rows:
        return []
    final = {
        "x": label,
        "x_sort": rows[0].get("x_sort"),
        "original_order": min(int(row.get("original_order") or 0) for row in rows),
        "is_tail": True,
    }
    return [glm_group_mapping_row(row, final, label=label, is_tail=True) for row in rows]


def clean_numeric_labels(rows: list[dict[str, Any]], band_width: Any) -> None:
    from py_lucidum.tools.line_bar.query import clean_partial_numeric_labels

    clean_partial_numeric_labels(rows, band_width)


__all__ = ["build_glm_partial_dependence_overlay", "empty_glm_partial_dependence_warning"]
