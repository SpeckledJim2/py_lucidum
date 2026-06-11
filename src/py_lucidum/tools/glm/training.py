from __future__ import annotations

import atexit
from collections.abc import Iterator
from contextlib import contextmanager
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
import warnings
from pathlib import Path
from typing import Any, Callable
from uuid import uuid4

import duckdb

from py_lucidum.core import Dataset, quote_ident, sql_literal

from .store import GlmModelStore, json_safe_number
from .terms import column_tokens, model_matrix, term_groups
from .validation import TARGET_COLUMN, physical_sample_column, validate_request


ProgressCallback = Callable[[dict[str, Any]], None]
_GLUM_FIRST_IMPORT_SAW_LIGHTGBM: bool | None = None
FEATURE_IMPORTANCE_METRIC = "weighted_mean_abs_centered_linear_predictor_contribution"
FEATURE_IMPORTANCE_METRIC_LABEL = "GLM eta MAD"


def _elapsed_ms(started: float) -> float:
    return round((time.perf_counter() - started) * 1000, 1)


def _safe_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _result_timings(result: dict[str, Any]) -> dict[str, Any]:
    timings = result.get("timings")
    if not isinstance(timings, dict):
        timings = {}
        result["timings"] = timings
    return timings


def _merge_result_timings(result: dict[str, Any], timings: dict[str, Any]) -> dict[str, Any]:
    result_timings = _result_timings(result)
    result_timings.update(timings)
    return result_timings


def _persist_manifest_timings(store: GlmModelStore, result: dict[str, Any], timings: dict[str, Any]) -> None:
    model_id = str(result.get("model_id") or "").strip()
    if not model_id:
        return
    manifest = store.manifest(model_id)
    manifest_timings = manifest.get("timings")
    if not isinstance(manifest_timings, dict):
        manifest_timings = {}
        manifest["timings"] = manifest_timings
    manifest_timings.update(timings)
    store.write_json(store.artifact_path(model_id, "manifest"), manifest)


@contextmanager
def _suppress_tabmat_mixed_dtype_warning() -> Iterator[None]:
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=r"^Matrices do not all have the same dtype\.",
            category=UserWarning,
            module=r"^tabmat\.split_matrix$",
        )
        yield


class MissingGlmDependency(RuntimeError):
    def __init__(self, missing: str, hint: str | None = None):
        message = f"Install GLM dependencies with `pip install 'py-lucidum[glm]'` to train GLM models. Missing: {missing}"
        if hint:
            message = f"{message}. {hint}"
        super().__init__(message)
        self.missing = missing
        self.hint = hint


def preload_lightgbm_before_glum() -> bool:
    """Load LightGBM first when it is installed to avoid a native load-order crash.

    On macOS in the supported modelling environment, importing ``glum`` before
    LightGBM can make a later multi-threaded LightGBM fit segfault inside native
    dataset construction. GLM must still work in GLM-only installs, so failures
    here are intentionally non-fatal and GBM training will report its own
    dependency/runtime errors when requested.
    """
    if os.environ.get("PY_LUCIDUM_SKIP_LIGHTGBM_PRELOAD"):
        return False
    if "lightgbm" in sys.modules:
        return True
    try:
        import lightgbm  # type: ignore[import-not-found]  # noqa: F401
    except (ImportError, OSError):
        return False
    return True


def glm_dependencies() -> tuple[Any, Any, Any, Any, Any]:
    global _GLUM_FIRST_IMPORT_SAW_LIGHTGBM
    missing: list[str] = []
    preloaded_lightgbm = preload_lightgbm_before_glum()
    lightgbm_loaded = "lightgbm" in sys.modules
    try:
        import glum  # type: ignore[import-not-found]
        from glum import GeneralizedLinearRegressor  # type: ignore[import-not-found]
        from glum import GeneralizedLinearRegressorCV  # type: ignore[import-not-found]
    except ImportError:
        glum = None
        GeneralizedLinearRegressor = None
        GeneralizedLinearRegressorCV = None
        missing.append("glum")
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
        raise MissingGlmDependency(", ".join(missing))
    if _GLUM_FIRST_IMPORT_SAW_LIGHTGBM is None:
        _GLUM_FIRST_IMPORT_SAW_LIGHTGBM = lightgbm_loaded or preloaded_lightgbm
    return glum, GeneralizedLinearRegressor, GeneralizedLinearRegressorCV, np, pd


def should_isolate_glm_fit() -> bool:
    return "lightgbm" in sys.modules and _GLUM_FIRST_IMPORT_SAW_LIGHTGBM is not False


def train_model_in_subprocess(
    dataset: Dataset,
    store: GlmModelStore,
    payload: dict[str, Any],
    *,
    progress_callback: ProgressCallback | None = None,
    activate: bool = True,
    parent_dependency_ms: float = 0.0,
) -> dict[str, Any]:
    if not os.environ.get("PY_LUCIDUM_GLM_FIT_ONE_SHOT"):
        result, _started = persistent_glm_fit_worker().request(
            dataset,
            store,
            payload,
            progress_callback=progress_callback,
            activate=activate,
            parent_dependency_ms=parent_dependency_ms,
        )
        return result
    return train_model_in_subprocess_one_shot(
        dataset,
        store,
        payload,
        progress_callback=progress_callback,
        activate=activate,
        parent_dependency_ms=parent_dependency_ms,
        worker_mode="one_shot",
    )


def train_model_in_subprocess_one_shot(
    dataset: Dataset,
    store: GlmModelStore,
    payload: dict[str, Any],
    *,
    progress_callback: ProgressCallback | None = None,
    activate: bool = True,
    parent_dependency_ms: float = 0.0,
    worker_mode: str = "one_shot",
) -> dict[str, Any]:
    progress = progress_callback or (lambda _progress: None)
    progress({"phase": "fitting", "message": "Running isolated GLM fit worker", "percent": 30, "worker_mode": worker_mode})
    worker_started = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix="lucidum-glm-worker-") as tmp_dir:
        tmp_path = Path(tmp_dir)
        request_path = tmp_path / "request.json"
        response_path = tmp_path / "response.json"
        request_path.write_text(
            json.dumps(
                {
                    "dataset_path": str(dataset.path),
                    "payload": payload,
                    "activate": activate,
                }
            ),
            encoding="utf-8",
        )
        completed = subprocess.run(
            [sys.executable, "-m", "py_lucidum.tools.glm.worker", str(request_path), str(response_path)],
            check=False,
            capture_output=True,
            text=True,
            env={**os.environ, "PY_LUCIDUM_SKIP_LIGHTGBM_PRELOAD": "1"},
        )
        if completed.returncode != 0:
            detail = (completed.stderr or completed.stdout or "").strip()
            if len(detail) > 800:
                detail = f"{detail[:800]}..."
            suffix = f" {detail}" if detail else ""
            raise RuntimeError(f"GLM worker exited unexpectedly with code {completed.returncode}.{suffix}")
        if not response_path.exists():
            raise RuntimeError("GLM worker exited without writing a response")
        response = json.loads(response_path.read_text(encoding="utf-8"))
    worker_total_ms = _elapsed_ms(worker_started)
    if not response.get("ok"):
        error = str(response.get("error") or "GLM worker failed")
        raise RuntimeError(error)
    result = response.get("result")
    if not isinstance(result, dict):
        raise RuntimeError("GLM worker returned an invalid response")
    timings = _result_timings(result)
    worker_dependency_ms = _safe_float(timings.get("dependency_ms"))
    worker_timings = {
        "worker_mode": worker_mode,
        "worker_started": True,
        "worker_total_ms": worker_total_ms,
        "parent_dependency_ms": round(parent_dependency_ms, 1),
        "worker_dependency_ms": round(worker_dependency_ms, 1),
        "dependency_ms": round(parent_dependency_ms + worker_dependency_ms, 1),
    }
    _merge_result_timings(result, worker_timings)
    _persist_manifest_timings(store, result, worker_timings)
    progress({"phase": "writing", "message": "GLM worker saved artifacts", "percent": 90, "timings": result.get("timings")})
    return result


class _GlmFitWorkerResponseError(RuntimeError):
    pass


class PersistentGlmFitWorker:
    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._process: subprocess.Popen[str] | None = None
        self._tmp_dir: Path | None = None

    def request(
        self,
        dataset: Dataset,
        store: GlmModelStore,
        payload: dict[str, Any],
        *,
        progress_callback: ProgressCallback | None = None,
        activate: bool = True,
        parent_dependency_ms: float = 0.0,
    ) -> tuple[dict[str, Any], bool]:
        progress = progress_callback or (lambda _progress: None)
        progress({"phase": "fitting", "message": "Running isolated GLM fit worker", "percent": 30, "worker_mode": "persistent"})
        request_path: Path | None = None
        response_path: Path | None = None
        with self._lock:
            try:
                process, started = self._ensure_process()
                request_id = uuid4().hex
                tmp_dir = self._tmp_dir
                if tmp_dir is None:
                    raise RuntimeError("GLM fit worker temporary directory is not available.")
                request_path = tmp_dir / f"{request_id}.request.json"
                response_path = tmp_dir / f"{request_id}.response.json"
                request_path.write_text(
                    json.dumps(
                        {
                            "dataset_path": str(dataset.path),
                            "payload": payload,
                            "activate": activate,
                        },
                        default=str,
                    ),
                    encoding="utf-8",
                )
                if process.stdin is None or process.stdout is None:
                    raise RuntimeError("GLM fit worker pipes are not available.")
                worker_started = time.perf_counter()
                process.stdin.write(json.dumps({"request_id": request_id, "request_path": str(request_path), "response_path": str(response_path)}) + "\n")
                process.stdin.flush()
                self._wait_for_ack(process, request_id)
                worker_total_ms = _elapsed_ms(worker_started)
                if not response_path.exists():
                    raise RuntimeError("GLM fit worker did not write a response.")
                response = json.loads(response_path.read_text(encoding="utf-8"))
                if not response.get("ok"):
                    raise _GlmFitWorkerResponseError(str(response.get("error") or "GLM worker failed"))
                result = response.get("result")
                if not isinstance(result, dict):
                    raise RuntimeError("GLM fit worker returned an invalid response.")
                self._annotate_result(
                    store,
                    result,
                    worker_mode="persistent",
                    worker_started=started,
                    worker_total_ms=worker_total_ms,
                    parent_dependency_ms=parent_dependency_ms,
                )
                progress({"phase": "writing", "message": "GLM worker saved artifacts", "percent": 90, "timings": result.get("timings")})
                return result, started
            except _GlmFitWorkerResponseError:
                raise
            except Exception:
                self.stop()
                fallback = train_model_in_subprocess_one_shot(
                    dataset,
                    store,
                    payload,
                    progress_callback=progress_callback,
                    activate=activate,
                    parent_dependency_ms=parent_dependency_ms,
                    worker_mode="one_shot_fallback",
                )
                return fallback, True
            finally:
                for path in (request_path, response_path):
                    if isinstance(path, Path):
                        try:
                            path.unlink(missing_ok=True)
                        except OSError:
                            pass

    def _annotate_result(
        self,
        store: GlmModelStore,
        result: dict[str, Any],
        *,
        worker_mode: str,
        worker_started: bool,
        worker_total_ms: float,
        parent_dependency_ms: float,
    ) -> None:
        timings = _result_timings(result)
        worker_dependency_ms = _safe_float(timings.get("dependency_ms"))
        worker_timings = {
            "worker_mode": worker_mode,
            "worker_started": bool(worker_started),
            "worker_total_ms": round(worker_total_ms, 1),
            "parent_dependency_ms": round(parent_dependency_ms, 1),
            "worker_dependency_ms": round(worker_dependency_ms, 1),
            "dependency_ms": round(parent_dependency_ms + worker_dependency_ms, 1),
        }
        _merge_result_timings(result, worker_timings)
        _persist_manifest_timings(store, result, worker_timings)

    def _ensure_process(self) -> tuple[subprocess.Popen[str], bool]:
        process = self._process
        if process is not None and process.poll() is None:
            return process, False
        self.stop()
        tmp_dir = Path(tempfile.mkdtemp(prefix="lucidum-glm-fit-hot-"))
        process = subprocess.Popen(
            [sys.executable, "-m", "py_lucidum.tools.glm.worker", "--server"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            bufsize=1,
            env={
                **os.environ,
                "PY_LUCIDUM_SKIP_LIGHTGBM_PRELOAD": "1",
                "PY_LUCIDUM_GLM_FIT_WORKER": "1",
            },
        )
        self._tmp_dir = tmp_dir
        self._process = process
        return process, True

    def _wait_for_ack(self, process: subprocess.Popen[str], request_id: str) -> None:
        if process.stdout is None:
            raise RuntimeError("GLM fit worker stdout is not available.")
        while True:
            line = process.stdout.readline()
            if not line:
                raise RuntimeError(f"GLM fit worker exited unexpectedly with code {process.poll()}.")
            try:
                message = json.loads(line)
            except json.JSONDecodeError:
                continue
            if str(message.get("request_id") or "") != request_id:
                continue
            if not message.get("ok", True):
                raise RuntimeError(str(message.get("error") or "GLM fit worker failed."))
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


_PERSISTENT_GLM_FIT_WORKER = PersistentGlmFitWorker()


def persistent_glm_fit_worker() -> PersistentGlmFitWorker:
    return _PERSISTENT_GLM_FIT_WORKER


def stop_persistent_glm_fit_worker() -> None:
    _PERSISTENT_GLM_FIT_WORKER.stop()


atexit.register(stop_persistent_glm_fit_worker)


def write_dataframe_parquet(frame: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(database=":memory:")
    try:
        con.register("artifact_df", frame)
        con.execute(f"COPY artifact_df TO {sql_literal(str(path))} (FORMAT PARQUET)")
    finally:
        con.close()


def formula_context(np: Any) -> dict[str, Any]:
    from formulaic import transforms  # type: ignore[import-not-found]

    def ifelse(condition: Any, yes: Any, no: Any) -> Any:
        return np.asarray(np.where(condition, yes, no), dtype=float)

    def pmin(*values: Any) -> Any:
        arrays = [np.asarray(value) for value in values]
        if not arrays:
            raise ValueError("pmin requires at least one argument")
        return np.asarray(np.minimum.reduce(np.broadcast_arrays(*arrays)), dtype=float)

    def pmax(*values: Any) -> Any:
        arrays = [np.asarray(value) for value in values]
        if not arrays:
            raise ValueError("pmax requires at least one argument")
        return np.asarray(np.maximum.reduce(np.broadcast_arrays(*arrays)), dtype=float)

    return {
        "np": np,
        "numpy": np,
        "ifelse": ifelse,
        "pmin": pmin,
        "pmax": pmax,
        "log": np.log,
        "log1p": np.log1p,
        "exp": np.exp,
        "sqrt": np.sqrt,
        "abs": np.abs,
        "min": np.minimum,
        "max": np.maximum,
        "C": transforms.C,
        "poly": transforms.poly,
        "bs": transforms.basis_spline,
        "cs": transforms.cyclic_cubic_spline,
        "ns": transforms.natural_cubic_spline,
    }


def glum_family(glum: Any, family: str, family_parameter: float | None) -> Any:
    if family == "tweedie":
        return glum._distribution.TweedieDistribution(power=family_parameter if family_parameter is not None else 1.5)
    if family == "negative.binomial":
        return glum._distribution.NegativeBinomialDistribution(theta=family_parameter if family_parameter is not None else 1.0)
    return family


def glm_formula_drop_first(regularization_mode: str) -> bool:
    return str(regularization_mode or "none") == "none"


def _is_singular_matrix_error(exc: Exception) -> bool:
    message = str(exc).lower()
    return ("singular" in message and "matrix" in message) or "rank deficient" in message or "rank-deficient" in message


def _raise_actionable_singular_matrix_error(exc: Exception) -> None:
    if not _is_singular_matrix_error(exc):
        raise exc
    raise ValueError(
        "GLM formula produced a rank-deficient design matrix. For unpenalized fits, simplify redundant spline, transform, "
        "or interaction terms, try centered spline constraints or explicit no-intercept syntax such as `0 +` when appropriate, "
        "or use ridge/auto regularization. Original glum error: "
        f"{exc}"
    ) from exc


def data_frame_from_dataset(dataset: Dataset) -> tuple[Any, list[str]]:
    with dataset.lock:
        columns = [column.name for column in dataset.valid_schema_columns()]
        projection = ",\n  ".join(["ROW_NUMBER() OVER () AS __lucidum_row_id", *[quote_ident(name) for name in columns]])
        frame = dataset.con.execute(f"SELECT {projection} FROM {dataset.relation_sql()}").fetchdf()
    return frame, columns


def finite_mask(np: Any, values: Any) -> Any:
    return values.notna() & np.isfinite(values.astype(float))


def offset_values_for_frame(frame: Any, offset_terms: list[str], context: dict[str, Any], np: Any, pd: Any) -> Any | None:
    terms = [str(term or "").strip() for term in offset_terms if str(term or "").strip()]
    if not terms:
        return None
    local_context = dict(context)
    for column in frame.columns:
        local_context[str(column)] = frame[column]
    values = pd.Series(0.0, index=frame.index, dtype=float)
    for term in terms:
        try:
            evaluated = eval(term, {"__builtins__": {}}, local_context)
        except Exception as exc:
            raise ValueError(f"Could not evaluate GLM offset expression `{term}`: {exc}") from exc
        if not hasattr(evaluated, "__len__") or isinstance(evaluated, (str, bytes)):
            evaluated = np.full(len(frame), evaluated)
        series = pd.Series(evaluated, index=frame.index)
        values = values + pd.to_numeric(series, errors="coerce")
    return values


def check_target_range(np: Any, family: str, y: Any) -> None:
    values = np.asarray(y, dtype=float)
    if family in {"poisson", "negative.binomial", "tweedie"} and np.any(values < 0):
        raise ValueError(f"{family} GLMs require non-negative fitted response values")
    if family in {"gamma", "inverse.gaussian"} and np.any(values <= 0):
        raise ValueError(f"{family} GLMs require positive fitted response values")
    if family == "binomial" and (np.any(values < 0) or np.any(values > 1)):
        raise ValueError("Binomial GLMs require fitted response values between 0 and 1")


def jsonable(value: Any, np: Any, pd: Any) -> Any:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    if isinstance(value, dict):
        return {str(key): jsonable(val, np, pd) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item, np, pd) for item in value]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return value


def coefficient_rows(
    model: Any,
    fit_frame: Any,
    y_fit: Any,
    fit_weight: Any,
    context: dict[str, Any],
    np: Any,
    pd: Any,
    source_columns: list[str] | None = None,
    *,
    include_inference: bool = True,
) -> list[dict[str, Any]]:
    coefficient_features = coefficient_feature_rows(model, source_columns or [])
    coefficient_feature_by_name = {
        str(name): list(features)
        for name, features in zip([str(name) for name in getattr(model, "feature_names_", [])], coefficient_features)
    }

    def features_for_coefficient(term: str, coefficient_index: int) -> list[str]:
        if term.lower() == "intercept" or term == "(Intercept)":
            return []
        features = coefficient_feature_by_name.get(term)
        if features is None and 0 <= coefficient_index < len(coefficient_features):
            features = coefficient_features[coefficient_index]
        if not features and source_columns:
            features = column_tokens(term, source_columns)
        return list(features or [])

    table = None
    if include_inference:
        try:
            table = model.coef_table(fit_frame, y_fit, sample_weight=fit_weight, context=context)
        except Exception:
            table = None
    if table is not None:
        rows: list[dict[str, Any]] = []
        coefficient_index = 0
        for term, row in table.iterrows():
            raw_name = str(term)
            name = "(Intercept)" if raw_name.lower() == "intercept" else raw_name
            features = features_for_coefficient(raw_name, coefficient_index)
            if raw_name.lower() != "intercept" and raw_name != "(Intercept)":
                coefficient_index += 1
            rows.append(
                {
                    "term": name,
                    "features": features,
                    "estimate": jsonable(row.get("coef"), np, pd),
                    "std_error": jsonable(row.get("se"), np, pd),
                    "statistic": jsonable(row.get("z_value", row.get("t_value")), np, pd),
                    "p_value": jsonable(row.get("p_value"), np, pd),
                    "ci_lower": jsonable(row.get("ci_lower"), np, pd),
                    "ci_upper": jsonable(row.get("ci_upper"), np, pd),
                }
            )
        return rows

    has_intercept = bool(getattr(model, "fit_intercept", True))
    feature_names = [str(name) for name in getattr(model, "feature_names_", [])]
    if has_intercept:
        terms = ["(Intercept)", *feature_names]
        estimates = [getattr(model, "intercept_", None), *list(getattr(model, "coef_", []))]
    else:
        terms = feature_names
        estimates = list(getattr(model, "coef_", []))
    return [
        {
            "term": term,
            "features": (
                []
                if has_intercept and index == 0
                else features_for_coefficient(term, index - 1 if has_intercept else index)
            ),
            "estimate": jsonable(estimate, np, pd),
            "std_error": None,
            "statistic": None,
            "p_value": None,
            "ci_lower": None,
            "ci_upper": None,
        }
        for index, (term, estimate) in enumerate(zip(terms, estimates))
    ]


def coefficient_feature_rows(model: Any, source_columns: list[str]) -> list[list[str]]:
    feature_names = [str(name) for name in getattr(model, "feature_names_", [])]
    features_by_index: list[list[str]] = [[] for _ in feature_names]
    source_set = set(source_columns)
    spec = getattr(model, "X_model_spec_", None)
    if spec is not None:
        term_variables = getattr(spec, "term_variables", {}) or {}
        term_indices = getattr(spec, "term_indices", {}) or {}
        for term in getattr(spec, "terms", []) or []:
            variables = sorted(str(name) for name in (term_variables.get(term, set()) or set()) if str(name) in source_set)
            if not variables:
                continue
            for raw_index in term_indices.get(term, []) or []:
                try:
                    index = int(raw_index)
                except (TypeError, ValueError):
                    continue
                if 0 <= index < len(features_by_index):
                    features_by_index[index] = list(variables)
    if source_columns:
        for index, features in enumerate(features_by_index):
            if not features:
                features_by_index[index] = column_tokens(feature_names[index], source_columns)
    return features_by_index


def glm_feature_importance_rows(
    model: Any,
    fit_frame: Any,
    source_columns: list[str],
    fit_weight: Any,
    context: dict[str, Any],
    np: Any,
    pd: Any,
) -> list[dict[str, Any]]:
    groups = term_groups(model, [], source_columns)
    if not groups:
        return []
    matrix = np.asarray(model_matrix(model, fit_frame, context), dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] == 0:
        return []
    coefficients = np.asarray(getattr(model, "coef_", []), dtype=float)
    if coefficients.size == 0:
        return []

    weights = (
        np.asarray(fit_weight, dtype=float)
        if fit_weight is not None
        else np.ones(matrix.shape[0], dtype=float)
    )
    if weights.shape[0] != matrix.shape[0]:
        weights = np.ones(matrix.shape[0], dtype=float)
    base_mask = np.isfinite(weights) & (weights > 0)
    if not bool(base_mask.any()):
        return []

    contributions: dict[str, Any] = {}
    term_counts: dict[str, int] = {}
    for variables, info in groups.items():
        features = [str(feature) for feature in variables if str(feature).strip()]
        if not features:
            continue
        raw_indices = [int(index) for index in info.get("term_indices", [])]
        indices = [
            index
            for index in raw_indices
            if 0 <= index < matrix.shape[1] and index < coefficients.size
        ]
        if not indices:
            continue
        contribution = matrix[:, indices].dot(coefficients[indices])
        finite_mask = base_mask & np.isfinite(contribution)
        if not bool(finite_mask.any()):
            continue
        center = float(np.average(contribution[finite_mask], weights=weights[finite_mask]))
        centered = np.where(np.isfinite(contribution), contribution - center, 0.0)
        share = centered / float(len(features))
        for feature in features:
            if feature not in contributions:
                contributions[feature] = np.zeros(matrix.shape[0], dtype=float)
                term_counts[feature] = 0
            contributions[feature] = contributions[feature] + share
            term_counts[feature] += 1

    rows: list[dict[str, Any]] = []
    for feature, contribution in contributions.items():
        finite_mask = base_mask & np.isfinite(contribution)
        if not bool(finite_mask.any()):
            continue
        importance = float(np.average(np.abs(contribution[finite_mask]), weights=weights[finite_mask]))
        if not math.isfinite(importance):
            continue
        rows.append(
            {
                "feature": feature,
                "importance": jsonable(importance, np, pd),
                "term_count": int(term_counts.get(feature, 0)),
                "metric": FEATURE_IMPORTANCE_METRIC,
            }
        )
    return sorted(rows, key=lambda row: (-float(row["importance"] or 0.0), str(row["feature"]).lower()))


def safe_metric(callable_metric: Any, *args: Any, **kwargs: Any) -> float | None:
    try:
        return json_safe_number(callable_metric(*args, **kwargs))
    except Exception:
        return None


def diagnostics_payload(
    model: Any,
    fit_frame: Any,
    y_fit: Any,
    fit_weight: Any,
    context: dict[str, Any],
    np: Any,
    coefficient_count: int,
    *,
    offset_values: Any | None = None,
) -> dict[str, Any]:
    predict_kwargs = {"context": context}
    if offset_values is not None:
        predict_kwargs["offset"] = offset_values
    mu = model.predict(fit_frame, **predict_kwargs)
    family = model.family_instance
    parameters = max(1, coefficient_count)
    deviance = safe_metric(family.deviance, y_fit, mu, sample_weight=fit_weight)
    dispersion = safe_metric(family.dispersion, y_fit, mu, sample_weight=fit_weight, ddof=parameters)
    mean_target = float(np.average(np.asarray(y_fit, dtype=float), weights=np.asarray(fit_weight, dtype=float))) if fit_weight is not None else float(np.mean(y_fit))
    null_mu = np.full_like(np.asarray(y_fit, dtype=float), mean_target, dtype=float)
    null_deviance = safe_metric(family.deviance, y_fit, null_mu, sample_weight=fit_weight)
    metric_kwargs = dict(predict_kwargs)
    return {
        "deviance": deviance,
        "null_deviance": null_deviance,
        "aic": safe_metric(getattr(model, "aic", None), fit_frame, y_fit, fit_weight, **metric_kwargs),
        "aicc": safe_metric(getattr(model, "aicc", None), fit_frame, y_fit, fit_weight, **metric_kwargs),
        "bic": safe_metric(getattr(model, "bic", None), fit_frame, y_fit, fit_weight, **metric_kwargs),
        "dispersion": dispersion,
    }


def build_predictions_frame(
    frame: Any,
    model: Any,
    denominator_column: str,
    context: dict[str, Any],
    np: Any,
    pd: Any,
    *,
    offset_values: Any | None = None,
) -> tuple[Any, int, int]:
    score_frame = frame.copy()
    score_frame[TARGET_COLUMN] = np.nan
    if denominator_column:
        denominator = pd.to_numeric(score_frame[denominator_column], errors="coerce")
        score_mask = denominator.notna() & np.isfinite(denominator.astype(float)) & (denominator.astype(float) > 0)
    else:
        denominator = None
        score_mask = pd.Series(True, index=score_frame.index)
    if offset_values is not None:
        score_mask = score_mask & finite_mask(np, offset_values)

    output = score_frame.loc[score_mask, ["__lucidum_row_id"]].copy()
    predict_kwargs = {"context": context}
    if offset_values is not None:
        predict_kwargs["offset"] = offset_values.loc[score_mask].astype(float).to_numpy()
    predictions = model.predict(score_frame.loc[score_mask].copy(), **predict_kwargs)
    prediction_values = pd.to_numeric(predictions, errors="coerce")
    rate_values = prediction_values.copy() if denominator is not None else None
    if denominator is not None:
        prediction_values = prediction_values * denominator.loc[score_mask].to_numpy(dtype=float)
    finite = np.isfinite(np.asarray(prediction_values, dtype=float))
    output["glm_prediction"] = prediction_values
    output.loc[~finite, "glm_prediction"] = np.nan
    if rate_values is not None:
        rate_finite = np.isfinite(np.asarray(rate_values, dtype=float))
        output["glm_prediction_rate"] = rate_values
        output.loc[~rate_finite, "glm_prediction_rate"] = np.nan
    fitted_na_rows = int((~finite).sum())
    scored_rows = int(finite.sum())
    return output, scored_rows, fitted_na_rows


def regularization_summary(model: Any, regularization: dict[str, Any], np: Any) -> dict[str, Any]:
    summary = dict(regularization)
    mode = str(summary.get("mode") or "none")
    if mode == "auto":
        summary["selected_alpha"] = getattr(model, "alpha_", None)
        summary["selected_l1_ratio"] = getattr(model, "l1_ratio_", None)
    coefficients = np.asarray(getattr(model, "coef_", []), dtype=float)
    summary["nonzero_coefficients"] = int(np.count_nonzero(np.abs(coefficients) > 1e-10))
    summary["coefficient_count"] = int(coefficients.size)
    return summary


def train_model(
    dataset: Dataset,
    store: GlmModelStore,
    payload: dict[str, Any],
    *,
    progress_callback: ProgressCallback | None = None,
    activate: bool = True,
) -> dict[str, Any]:
    dependency_started = time.perf_counter()
    glm_dependencies()
    parent_dependency_ms = _elapsed_ms(dependency_started)
    if should_isolate_glm_fit():
        return train_model_in_subprocess(
            dataset,
            store,
            payload,
            progress_callback=progress_callback,
            activate=activate,
            parent_dependency_ms=parent_dependency_ms,
        )
    return _train_model_impl(
        dataset,
        store,
        payload,
        progress_callback=progress_callback,
        activate=activate,
        overall_started=dependency_started,
        base_timings={
            "worker_mode": "in_process",
            "worker_started": False,
            "worker_total_ms": 0.0,
            "parent_dependency_ms": parent_dependency_ms,
        },
    )


def _train_model_impl(
    dataset: Dataset,
    store: GlmModelStore,
    payload: dict[str, Any],
    *,
    progress_callback: ProgressCallback | None = None,
    activate: bool = True,
    overall_started: float | None = None,
    base_timings: dict[str, Any] | None = None,
) -> dict[str, Any]:
    started = overall_started if overall_started is not None else time.perf_counter()
    timings = dict(base_timings or {})
    dependency_started = time.perf_counter()
    glum, GeneralizedLinearRegressor, GeneralizedLinearRegressorCV, np, pd = glm_dependencies()
    local_dependency_ms = _elapsed_ms(dependency_started)
    parent_dependency_ms = _safe_float(timings.get("parent_dependency_ms"))
    timings["dependency_ms"] = round(parent_dependency_ms + local_dependency_ms, 1)
    if parent_dependency_ms:
        timings["training_dependency_ms"] = local_dependency_ms

    prep_started = time.perf_counter()
    validation = validate_request(dataset, payload)
    if not validation["ok"]:
        raise ValueError("; ".join(validation["errors"]))
    timings["validation_ms"] = _elapsed_ms(prep_started)

    progress = progress_callback or (lambda _progress: None)
    progress({"phase": "loading", "message": "Loading GLM training data", "percent": 5})
    data_load_started = time.perf_counter()
    frame, source_columns = data_frame_from_dataset(dataset)
    timings["data_load_ms"] = _elapsed_ms(data_load_started)
    prep_started = time.perf_counter()

    response_column = str(validation["response_column"])
    denominator_column = str(validation["denominator_column"] or "")
    training_scope = str(validation["training_scope"])
    family = str(validation["family"])
    family_param = validation.get("family_parameter")
    regularization = dict(validation.get("regularization") or {"mode": "none", "alpha": 0.0, "l1_ratio": 0.0, "scale_predictors": False})
    regularization_mode = str(regularization.get("mode") or "none")
    is_penalized = regularization_mode != "none"
    drop_first = glm_formula_drop_first(regularization_mode)
    formula = validation["formula"]
    fit_intercept = bool(formula.get("fit_intercept", True))
    context = formula_context(np)
    offset_terms = [str(term) for term in formula.get("offset_terms", [])]
    offset_values = offset_values_for_frame(frame, offset_terms, context, np, pd)

    response = pd.to_numeric(frame[response_column], errors="coerce")
    if denominator_column:
        denominator = pd.to_numeric(frame[denominator_column], errors="coerce")
        eligible_mask = denominator.notna() & np.isfinite(denominator.astype(float)) & (denominator.astype(float) > 0)
        target = response / denominator
        fit_weight = denominator
    else:
        eligible_mask = pd.Series(True, index=frame.index)
        target = response
        fit_weight = None

    fit_mask = eligible_mask & finite_mask(np, target)
    sample_column = physical_sample_column(dataset)
    if training_scope == "training":
        if not sample_column:
            raise ValueError("Training rows require a physical SAMPLE column")
        fit_mask = fit_mask & (frame[sample_column].astype(str).str.strip().str.lower() == "training")

    if fit_weight is not None:
        fit_mask = fit_mask & finite_mask(np, fit_weight) & (fit_weight.astype(float) > 0)
    if offset_values is not None:
        fit_mask = fit_mask & finite_mask(np, offset_values)

    if int(fit_mask.sum()) < 2:
        raise ValueError("GLM fitting needs at least two valid rows")

    y_fit = target.loc[fit_mask].astype(float)
    check_target_range(np, family, y_fit)
    fit_frame = frame.loc[fit_mask].copy()
    fit_frame[TARGET_COLUMN] = y_fit.to_numpy(dtype=float)
    fit_weight_values = fit_weight.loc[fit_mask].astype(float).to_numpy() if fit_weight is not None else None
    fit_offset_values = offset_values.loc[fit_mask].astype(float).to_numpy() if offset_values is not None else None

    progress({"phase": "fitting", "message": "Fitting GLM", "percent": 35, "training_rows": int(fit_mask.sum())})
    estimator_kwargs = {
        "family": glum_family(glum, family, float(family_param) if family_param is not None else None),
        "link": "auto",
        "fit_intercept": fit_intercept,
        "formula": str(formula["fitted"]),
        "drop_first": drop_first,
        "robust": True,
        "scale_predictors": bool(regularization.get("scale_predictors")),
    }
    if regularization_mode == "auto":
        estimator = GeneralizedLinearRegressorCV(
            l1_ratio=regularization.get("l1_ratio") or [0.0, 0.5, 1.0],
            n_alphas=int(regularization.get("n_alphas") or 50),
            cv=min(5, int(fit_mask.sum())),
            **estimator_kwargs,
        )
    elif regularization_mode == "manual":
        estimator = GeneralizedLinearRegressor(
            alpha=float(regularization.get("alpha")),
            l1_ratio=float(regularization.get("l1_ratio")),
            **estimator_kwargs,
        )
    else:
        estimator = GeneralizedLinearRegressor(
            alpha=0,
            **estimator_kwargs,
        )
    timings["prep_ms"] = _elapsed_ms(prep_started)
    fit_started = time.perf_counter()
    try:
        with _suppress_tabmat_mixed_dtype_warning():
            estimator.fit(
                fit_frame,
                sample_weight=fit_weight_values,
                store_covariance_matrix=not is_penalized,
                context=context,
                offset=fit_offset_values,
            )
    except Exception as exc:
        _raise_actionable_singular_matrix_error(exc)
    timings["fit_ms"] = _elapsed_ms(fit_started)
    regularization = regularization_summary(estimator, regularization, np)

    progress({"phase": "scoring", "message": "Scoring GLM predictions", "percent": 70})
    score_started = time.perf_counter()
    with _suppress_tabmat_mixed_dtype_warning():
        predictions, scored_rows, fitted_na_rows = build_predictions_frame(frame, estimator, denominator_column, context, np, pd, offset_values=offset_values)
        coefficients = coefficient_rows(
            estimator,
            fit_frame,
            y_fit.to_numpy(dtype=float),
            fit_weight_values,
            context,
            np,
            pd,
            source_columns,
            include_inference=not is_penalized,
        )
        feature_importance = glm_feature_importance_rows(
            estimator,
            fit_frame,
            source_columns,
            fit_weight_values,
            context,
            np,
            pd,
        )
        diagnostics = diagnostics_payload(
            estimator,
            fit_frame,
            y_fit.to_numpy(dtype=float),
            fit_weight_values,
            context,
            np,
            len(coefficients),
            offset_values=fit_offset_values,
        )
    timings["score_ms"] = _elapsed_ms(score_started)
    diagnostics.update(
        {
            "training_rows": int(fit_mask.sum()),
            "eligible_rows": int(eligible_mask.sum()),
            "scored_rows": scored_rows,
            "fitted_na_rows": fitted_na_rows,
            "coefficient_count": len(coefficients),
            "nonzero_coefficients": regularization.get("nonzero_coefficients"),
        }
    )

    label = str(payload.get("label") or f"GLM {response_column}").strip() or f"GLM {response_column}"
    model_id = store.create_model_id(label)
    model_dir = store.create_model_dir(model_id)
    manifest = {
        "model_id": model_id,
        "label": label,
        "tool": "glm",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "family": family,
        "link": "auto",
        "family_parameter": jsonable(family_param, np, pd),
        "regularization": jsonable(regularization, np, pd),
        "response_column": response_column,
        "denominator_column": denominator_column,
        "offset_column": denominator_column,
        "offset_terms": jsonable(offset_terms, np, pd),
        "training_scope": training_scope,
        "sample_column": sample_column,
        "formula": {
            "raw": formula["raw"],
            "stripped": formula["stripped"],
            "rhs": formula["rhs"],
            "fitted": formula["fitted"],
            "offset_terms": offset_terms,
            "drop_first": drop_first,
            "fit_intercept": fit_intercept,
        },
        "diagnostics": jsonable(diagnostics, np, pd),
        "feature_importance": jsonable(feature_importance, np, pd),
        "feature_importance_metric": {
            "name": FEATURE_IMPORTANCE_METRIC,
            "label": FEATURE_IMPORTANCE_METRIC_LABEL,
            "scale": "linear_predictor",
            "description": "Weighted mean absolute centered feature contribution on the GLM linear predictor scale.",
            "interaction_allocation": "split_evenly",
        },
        "source_columns": source_columns,
        "sources": {"predictions": store.source_id(model_id)},
        "dataset": store.dataset_metadata(),
        "row_count": int(len(frame)),
        "training_rows": int(fit_mask.sum()),
        "scored_rows": scored_rows,
        "fitted_na_rows": fitted_na_rows,
        "coefficient_count": len(coefficients),
        "timings": timings,
    }

    progress({"phase": "writing", "message": "Saving GLM artifacts", "percent": 90})
    artifact_write_started = time.perf_counter()
    coefficient_frame = pd.DataFrame(coefficients)
    feature_importance_frame = pd.DataFrame(
        feature_importance,
        columns=["feature", "importance", "term_count", "metric"],
    )
    write_dataframe_parquet(coefficient_frame, store.artifact_path(model_id, "coefficients"))
    write_dataframe_parquet(feature_importance_frame, store.artifact_path(model_id, "feature_importance"))
    write_dataframe_parquet(predictions, store.artifact_path(model_id, "predictions"))
    with store.artifact_path(model_id, "estimator").open("wb") as handle:
        pickle.dump(estimator, handle, protocol=pickle.HIGHEST_PROTOCOL)
    store.artifact_path(model_id, "formula").write_text(str(formula["raw"]), encoding="utf-8")
    store.write_json(store.artifact_path(model_id, "diagnostics"), manifest["diagnostics"])
    timings["artifact_write_ms"] = _elapsed_ms(artifact_write_started)
    timings["elapsed_ms"] = _elapsed_ms(started)
    progress({"phase": "writing", "message": "Saved GLM artifacts", "percent": 95, "timings": timings})
    store.write_json(store.artifact_path(model_id, "manifest"), manifest)

    result = store.activate_model(model_id) if activate else manifest
    result["coefficients"] = coefficients
    result["feature_importance"] = feature_importance
    result["diagnostics"] = manifest["diagnostics"]
    result["model_dir"] = str(model_dir)
    return result
__all__ = [
    "FEATURE_IMPORTANCE_METRIC",
    "FEATURE_IMPORTANCE_METRIC_LABEL",
    "MissingGlmDependency",
    "glm_dependencies",
    "glm_feature_importance_rows",
    "train_model",
    "write_dataframe_parquet",
]
