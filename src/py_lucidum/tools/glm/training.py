from __future__ import annotations

import atexit
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
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
INTERNAL_INTERCEPT_COLUMN_BASE = "__lucidum_glm_intercept_only"
GLM_ILL_CONDITIONED_WARNING = (
    "GLM fit produced an ill-conditioned matrix; coefficient estimates or inference may be unstable. "
    'Use centered spline constraints such as `ns(feature, df=4, constraints="center")`, explicit no-intercept syntax such as `0 +`, '
    "or ridge/auto regularization."
)
GLM_INFERENCE_WARNING = (
    "GLM coefficient inference was not fully available because one or more standard errors were non-finite. "
    "Simplify rank-deficient terms, use centered/no-intercept spline syntax, or use ridge/auto regularization."
)
GLM_PENALIZED_RANK_DEFICIENT_WARNING = (
    "GLM design matrix is rank-deficient; regularization allowed the model to be saved, "
    "but unpenalized coefficient inference would not be identifiable."
)
GLM_DESIGN_CONDITION_LIMIT = 1e12


@dataclass(frozen=True)
class PredictionFrameResult:
    frame: Any
    response_values: Any
    score_mask: Any
    scored_rows: int
    fitted_na_rows: int


def _elapsed_ms(started: float) -> float:
    return round((time.perf_counter() - started) * 1000, 1)


def _safe_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def internal_intercept_column_name(columns: list[str]) -> str:
    existing = {str(column) for column in columns}
    column = INTERNAL_INTERCEPT_COLUMN_BASE
    suffix = 2
    while column in existing:
        column = f"{INTERNAL_INTERCEPT_COLUMN_BASE}_{suffix}"
        suffix += 1
    return column


def internal_intercept_column_from_manifest(manifest: dict[str, Any]) -> str:
    formula = manifest.get("formula")
    if not isinstance(formula, dict):
        return ""
    column = str(formula.get("internal_intercept_column") or "").strip()
    return column


def add_internal_intercept_column(frame: Any, column: str) -> Any:
    if column:
        frame[column] = 1.0
    return frame


def estimator_intercept_value(estimator: Any, manifest: dict[str, Any]) -> float:
    value = _safe_float(getattr(estimator, "intercept_", 0.0))
    internal_column = internal_intercept_column_from_manifest(manifest)
    if internal_column:
        feature_names = [str(name) for name in getattr(estimator, "feature_names_", [])]
        coefficients = list(getattr(estimator, "coef_", []))
        for index, name in enumerate(feature_names):
            if name == internal_column and index < len(coefficients):
                value += _safe_float(coefficients[index])
                break
    return value


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


@contextmanager
def _capture_glm_warnings() -> Iterator[list[warnings.WarningMessage]]:
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        warnings.filterwarnings(
            "ignore",
            message=r"^Matrices do not all have the same dtype\.",
            category=UserWarning,
            module=r"^tabmat\.split_matrix$",
        )
        yield captured


def _dedupe_warnings(values: list[Any]) -> list[str]:
    warnings_out: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = str(value or "").strip()
        if not text or text in seen:
            continue
        seen.add(text)
        warnings_out.append(text)
    return warnings_out


def _glm_numerical_warning_messages(captured: list[warnings.WarningMessage]) -> list[str]:
    messages: list[str] = []
    for warning in captured:
        text = str(warning.message or "")
        lower = text.lower()
        category = getattr(warning.category, "__name__", "").lower()
        if (
            "ill-conditioned" in lower
            or "singular" in lower
            or "rank deficient" in lower
            or "rank-deficient" in lower
            or "linalg" in category
        ):
            messages.append(GLM_ILL_CONDITIONED_WARNING)
        elif "invalid value encountered in sqrt" in lower or "non-finite" in lower:
            messages.append(GLM_INFERENCE_WARNING)
    return _dedupe_warnings(messages)


def _coefficient_inference_warnings(coefficients: list[dict[str, Any]], *, include_inference: bool) -> list[str]:
    if not include_inference or not coefficients:
        return []
    for row in coefficients:
        if row.get("std_error") is None:
            return [GLM_INFERENCE_WARNING]
    return []


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


def glm_training_dependencies() -> tuple[Any, Any, Any, Any, Any, Any]:
    glum, GeneralizedLinearRegressor, GeneralizedLinearRegressorCV, np, pd = glm_dependencies()
    try:
        import polars as pl  # type: ignore[import-not-found]
    except ImportError as exc:
        raise MissingGlmDependency("polars") from exc
    return glum, GeneralizedLinearRegressor, GeneralizedLinearRegressorCV, np, pd, pl


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
        progress_path = tmp_path / "progress.json"
        request_path.write_text(
            json.dumps(
                {
                    "dataset_path": str(dataset.path),
                    "payload": payload,
                    "activate": activate,
                    "progress_path": str(progress_path),
                }
            ),
            encoding="utf-8",
        )
        completed_values: list[subprocess.CompletedProcess[str]] = []
        worker_errors: list[BaseException] = []

        def run_worker_process() -> None:
            try:
                completed_values.append(
                    subprocess.run(
                        [sys.executable, "-m", "py_lucidum.tools.glm.worker", str(request_path), str(response_path)],
                        check=False,
                        capture_output=True,
                        text=True,
                        env={**os.environ, "PY_LUCIDUM_SKIP_LIGHTGBM_PRELOAD": "1"},
                    )
                )
            except BaseException as exc:
                worker_errors.append(exc)

        worker_thread = threading.Thread(target=run_worker_process, daemon=True)
        worker_thread.start()
        last_progress_text = ""
        while worker_thread.is_alive():
            worker_thread.join(timeout=0.05)
            last_progress_text = forward_worker_progress(progress_path, progress, last_progress_text)
        last_progress_text = forward_worker_progress(progress_path, progress, last_progress_text)
        if worker_errors:
            raise worker_errors[0]
        if not completed_values:
            raise RuntimeError("GLM worker exited without returning a process result")
        completed = completed_values[0]
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
    progress({"phase": "writing", "message": "GLM worker saved artifacts", "percent": 90, "timings": result.get("timings")})
    return result


def forward_worker_progress(path: Path, callback: ProgressCallback, previous_text: str = "") -> str:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return previous_text
    if not text or text == previous_text:
        return previous_text
    try:
        progress = json.loads(text)
    except json.JSONDecodeError:
        return previous_text
    if not isinstance(progress, dict):
        return text
    callback(progress)
    return text


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
                self._wait_for_ack(process, request_id, progress)
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

    def _wait_for_ack(self, process: subprocess.Popen[str], request_id: str, progress_callback: ProgressCallback) -> None:
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
            if str(message.get("type") or "") == "progress":
                progress = message.get("progress")
                if isinstance(progress, dict):
                    progress_callback(progress)
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


def _rank_deficient_training_error(*, rank: int, columns: int, condition_number: float | None = None) -> ValueError:
    condition_text = (
        f" Condition number: {condition_number:.3g}."
        if condition_number is not None and math.isfinite(condition_number)
        else ""
    )
    return ValueError(
        "Training did not save a model: the GLM design matrix is rank-deficient "
        f"(rank {rank} of {columns}).{condition_text} "
        'For unpenalized fits, use centered spline constraints such as `ns(feature, df=4, constraints="center")`, '
        "explicit no-intercept syntax such as `0 +` when appropriate, or ridge/auto regularization."
    )


def _ill_conditioned_training_error(*, condition_number: float) -> ValueError:
    return ValueError(
        "Training did not save a model: the GLM design matrix is too ill-conditioned for an unpenalized fit "
        f"(condition number {condition_number:.3g}). "
        'Use centered spline constraints such as `ns(feature, df=4, constraints="center")`, scale or simplify redundant terms, '
        "or use ridge/auto regularization."
    )


def _raise_actionable_singular_matrix_error(exc: Exception) -> None:
    if not _is_singular_matrix_error(exc):
        raise exc
    raise ValueError(
        "Training did not save a model: the GLM design matrix is rank-deficient. For unpenalized fits, simplify redundant "
        "spline, transform, or interaction terms, use centered spline constraints or explicit no-intercept syntax such as "
        "`0 +` when appropriate, or use ridge/auto regularization. Original glum error: "
        f"{exc}"
    ) from exc


def formula_source_columns(formula: str, source_columns: list[str]) -> list[str]:
    from formulaic import Formula  # type: ignore[import-not-found]
    from formulaic.parser import DefaultFormulaParser  # type: ignore[import-not-found]

    formula_text = str(formula or "1")
    parsed = Formula(
        formula_text,
        _parser=DefaultFormulaParser(include_intercept=False),
        _context={
            "__formulaic_variables_available__": source_columns,
            "__formulaic_variables_used_lhs__": [],
        },
    )
    required = {str(variable) for variable in parsed.required_variables}
    # Formulaic can omit data arguments to stateful transforms such as bs, cs, and poly.
    required.update(column_tokens(formula_text, source_columns))
    return [column for column in source_columns if column in required]


def required_training_columns(dataset: Dataset, validation: dict[str, Any]) -> list[str]:
    source_columns = [column.name for column in dataset.valid_schema_columns()]
    requested = {
        str(validation.get("response_column") or "").strip(),
        str(validation.get("denominator_column") or "").strip(),
    }
    formula = validation.get("formula")
    if isinstance(formula, dict):
        requested.update(formula_source_columns(str(formula.get("fitted") or "1"), source_columns))
        for expression in formula.get("offset_terms") or []:
            requested.update(column_tokens(str(expression), source_columns))
    if str(validation.get("training_scope") or "all") == "training":
        sample_column = physical_sample_column(dataset)
        if sample_column:
            requested.add(sample_column)
    requested.discard("")
    return [column for column in source_columns if column in requested]


def data_frame_from_dataset(dataset: Dataset, columns: list[str]) -> Any:
    with dataset.lock:
        projection = ",\n  ".join(["ROW_NUMBER() OVER () AS __lucidum_row_id", *[quote_ident(name) for name in columns]])
        frame = dataset.con.execute(f"SELECT {projection} FROM {dataset.relation_sql()}").pl()
    return frame


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


def offset_values_for_polars_frame(
    frame: Any,
    offset_terms: list[str],
    context: dict[str, Any],
    np: Any,
    pl: Any,
) -> Any | None:
    terms = [str(term or "").strip() for term in offset_terms if str(term or "").strip()]
    if not terms:
        return None
    local_context = dict(context)
    for column in frame.columns:
        local_context[str(column)] = frame.get_column(column).to_numpy()
    values = np.zeros(frame.height, dtype=float)
    for term in terms:
        try:
            evaluated = eval(term, {"__builtins__": {}}, local_context)
        except Exception as exc:
            raise ValueError(f"Could not evaluate GLM offset expression `{term}`: {exc}") from exc
        if not hasattr(evaluated, "__len__") or isinstance(evaluated, (str, bytes)):
            evaluated = np.full(frame.height, evaluated)
        if len(evaluated) != frame.height:
            raise ValueError(f"GLM offset expression `{term}` returned {len(evaluated)} values for {frame.height} rows")
        numeric = pl.Series("offset", evaluated).cast(pl.Float64, strict=False).fill_null(float("nan")).to_numpy()
        values = values + numeric
    return values


def polars_numeric_array(frame: Any, column: str, pl: Any) -> Any:
    return (
        frame.get_column(column)
        .cast(pl.Float64, strict=False)
        .fill_null(float("nan"))
        .to_numpy()
    )


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


def _matrix_to_numpy(matrix: Any, np: Any) -> Any:
    if hasattr(matrix, "toarray"):
        return np.asarray(matrix.toarray(), dtype=float)
    if hasattr(matrix, "to_numpy"):
        return np.asarray(matrix.to_numpy(), dtype=float)
    return np.asarray(matrix, dtype=float)


def design_matrix_diagnostics(
    formula: str,
    fit_frame: Any,
    context: dict[str, Any],
    np: Any,
    *,
    ensure_full_rank: bool,
) -> dict[str, Any]:
    from formulaic import model_matrix as formulaic_model_matrix  # type: ignore[import-not-found]

    matrices = formulaic_model_matrix(formula, fit_frame, context=context, ensure_full_rank=ensure_full_rank)
    matrix = getattr(matrices, "rhs", matrices)
    values = _matrix_to_numpy(matrix, np)
    if values.ndim == 1:
        values = values.reshape((-1, 1))
    if values.ndim != 2:
        raise ValueError("Training did not save a model: the GLM formula did not produce a two-dimensional design matrix.")
    if values.shape[1] == 0:
        raise ValueError("Training did not save a model: the GLM formula produced no predictor columns.")
    if not bool(np.isfinite(values).all()):
        raise ValueError("Training did not save a model: the GLM formula produced non-finite predictor values.")
    rank = int(np.linalg.matrix_rank(values))
    try:
        condition_number = float(np.linalg.cond(values))
    except Exception:
        condition_number = math.inf
    condition_number_finite = math.isfinite(condition_number)
    return {
        "rank": rank,
        "columns": int(values.shape[1]),
        "rows": int(values.shape[0]),
        "condition_number": condition_number if condition_number_finite else None,
        "condition_number_finite": condition_number_finite,
    }


def check_unpenalized_design_matrix(diagnostics: dict[str, Any]) -> None:
    rank = int(diagnostics.get("rank") or 0)
    columns = int(diagnostics.get("columns") or 0)
    condition_value = diagnostics.get("condition_number")
    condition_number = float(condition_value) if condition_value is not None else None
    if columns and rank < columns:
        raise _rank_deficient_training_error(rank=rank, columns=columns, condition_number=condition_number)
    if condition_number is None and diagnostics.get("condition_number_finite") is False:
        raise _ill_conditioned_training_error(condition_number=math.inf)
    if condition_number is not None and math.isfinite(condition_number) and condition_number > GLM_DESIGN_CONDITION_LIMIT:
        raise _ill_conditioned_training_error(condition_number=condition_number)


def design_matrix_warnings(diagnostics: dict[str, Any], *, penalized: bool) -> list[str]:
    rank = int(diagnostics.get("rank") or 0)
    columns = int(diagnostics.get("columns") or 0)
    condition_value = diagnostics.get("condition_number")
    condition_number = float(condition_value) if condition_value is not None else None
    messages: list[str] = []
    if penalized and columns and rank < columns:
        messages.append(GLM_PENALIZED_RANK_DEFICIENT_WARNING)
    if (condition_number is None and diagnostics.get("condition_number_finite") is False) or (
        condition_number is not None and math.isfinite(condition_number) and condition_number > GLM_DESIGN_CONDITION_LIMIT
    ):
        messages.append(GLM_ILL_CONDITIONED_WARNING)
    return messages


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
    internal_intercept_column: str = "",
) -> list[dict[str, Any]]:
    coefficient_features = coefficient_feature_rows(model, source_columns or [])
    coefficient_feature_by_name = {
        str(name): list(features)
        for name, features in zip([str(name) for name in getattr(model, "feature_names_", [])], coefficient_features)
    }

    def features_for_coefficient(term: str, coefficient_index: int) -> list[str]:
        if term.lower() == "intercept" or term == "(Intercept)":
            return []
        if internal_intercept_column and term == internal_intercept_column:
            return []
        features = coefficient_feature_by_name.get(term)
        if features is None and 0 <= coefficient_index < len(coefficient_features):
            features = coefficient_features[coefficient_index]
        if not features and source_columns:
            features = column_tokens(term, source_columns)
        return list(features or [])

    def is_intercept_term(term: str) -> bool:
        return term.lower() == "intercept" or term == "(Intercept)" or bool(internal_intercept_column and term == internal_intercept_column)

    table = None
    if include_inference:
        try:
            if getattr(model, "covariance_matrix_", None) is not None:
                table = model.coef_table()
            else:
                table = model.coef_table(fit_frame, y_fit, sample_weight=fit_weight, context=context)
        except Exception:
            table = None
    if table is not None:
        rows: list[dict[str, Any]] = []
        coefficient_index = 0
        for term, row in table.iterrows():
            raw_name = str(term)
            intercept_term = is_intercept_term(raw_name)
            name = "(Intercept)" if intercept_term else raw_name
            features = [] if intercept_term else features_for_coefficient(raw_name, coefficient_index)
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
            "term": "(Intercept)" if is_intercept_term(term) else term,
            "features": (
                []
                if (has_intercept and index == 0) or is_intercept_term(term)
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
    *,
    design_matrix: Any | None = None,
) -> list[dict[str, Any]]:
    groups = term_groups(model, [], source_columns)
    if not groups:
        return []
    matrix = np.asarray(
        design_matrix if design_matrix is not None else model_matrix(model, fit_frame, context),
        dtype=float,
    )
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


def _shared_post_fit_design_matrix(
    model: Any,
    fit_frame: Any,
    fit_mask: Any,
    score_mask: Any,
    context: dict[str, Any],
    np: Any,
) -> Any | None:
    if not np.array_equal(fit_mask, score_mask):
        return None
    return np.asarray(model_matrix(model, fit_frame, context), dtype=float)


def safe_metric(callable_metric: Any, *args: Any, **kwargs: Any) -> float | None:
    try:
        return json_safe_number(callable_metric(*args, **kwargs))
    except Exception:
        return None


def diagnostics_payload(
    model: Any,
    y_fit: Any,
    mu: Any,
    fit_weight: Any,
    np: Any,
    coefficient_count: int,
) -> dict[str, Any]:
    family = model.family_instance
    parameters = max(1, coefficient_count)
    deviance = safe_metric(family.deviance, y_fit, mu, sample_weight=fit_weight)
    dispersion = safe_metric(family.dispersion, y_fit, mu, sample_weight=fit_weight, ddof=parameters)
    mean_target = float(np.average(np.asarray(y_fit, dtype=float), weights=np.asarray(fit_weight, dtype=float))) if fit_weight is not None else float(np.mean(y_fit))
    null_mu = np.full_like(np.asarray(y_fit, dtype=float), mean_target, dtype=float)
    null_deviance = safe_metric(family.deviance, y_fit, null_mu, sample_weight=fit_weight)
    coefficients = np.asarray(getattr(model, "coef_", []), dtype=float)
    nonzero = int(np.count_nonzero(np.abs(coefficients) > np.finfo(coefficients.dtype).eps))
    effective_parameters = nonzero + int(bool(getattr(model, "fit_intercept", True)))
    observation_count = int(np.asarray(y_fit).shape[0])
    log_likelihood = safe_metric(getattr(family, "log_likelihood", None), y_fit, mu, sample_weight=fit_weight)
    if log_likelihood is None:
        aic = None
        aicc = None
        bic = None
    else:
        aic = float(-2 * log_likelihood + 2 * effective_parameters)
        aicc = (
            float(aic + 2 * effective_parameters * (effective_parameters + 1) / (observation_count - effective_parameters - 1))
            if observation_count > effective_parameters + 1
            else None
        )
        bic = float(-2 * log_likelihood + np.log(observation_count) * effective_parameters)
    return {
        "deviance": deviance,
        "null_deviance": null_deviance,
        "aic": aic,
        "aicc": aicc,
        "bic": bic,
        "dispersion": dispersion,
    }


def build_predictions_frame(
    frame: Any,
    model: Any,
    denominator_column: str,
    context: dict[str, Any],
    np: Any,
    pl: Any,
    *,
    offset_values: Any | None = None,
    design_matrix: Any | None = None,
) -> PredictionFrameResult:
    if denominator_column:
        denominator = polars_numeric_array(frame, denominator_column, pl)
        score_mask = np.isfinite(denominator) & (denominator > 0)
    else:
        denominator = None
        score_mask = np.ones(frame.height, dtype=bool)
    if offset_values is not None:
        score_mask = score_mask & np.isfinite(offset_values)

    score_frame = frame.filter(pl.Series("__lucidum_score_mask", score_mask))
    predict_frame = score_frame
    if design_matrix is not None:
        if int(design_matrix.shape[0]) != score_frame.height:
            raise ValueError("Precomputed GLM design matrix does not match the scoring rows")
        predict_frame = design_matrix
    predict_kwargs = {"context": context}
    if offset_values is not None:
        predict_kwargs["offset"] = np.asarray(offset_values[score_mask], dtype=float)
    predictions = model.predict(predict_frame, **predict_kwargs)
    response_values = np.asarray(predictions, dtype=float)
    prediction_values = response_values
    rate_values = response_values if denominator is not None else None
    if denominator is not None:
        prediction_values = prediction_values * denominator[score_mask]
    finite = np.isfinite(prediction_values)
    output_columns: dict[str, Any] = {
        "__lucidum_row_id": score_frame.get_column("__lucidum_row_id"),
        "glm_prediction": pl.Series(
            "glm_prediction",
            np.where(finite, prediction_values, np.nan),
        ).fill_nan(None),
    }
    if rate_values is not None:
        rate_finite = np.isfinite(rate_values)
        output_columns["glm_prediction_rate"] = pl.Series(
            "glm_prediction_rate",
            np.where(rate_finite, rate_values, np.nan),
        ).fill_nan(None)
    fitted_na_rows = int((~finite).sum())
    scored_rows = int(finite.sum())
    return PredictionFrameResult(
        frame=pl.DataFrame(output_columns),
        response_values=response_values,
        score_mask=score_mask,
        scored_rows=scored_rows,
        fitted_na_rows=fitted_na_rows,
    )


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
    glm_training_dependencies()
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
    glum, GeneralizedLinearRegressor, GeneralizedLinearRegressorCV, np, pd, pl = glm_training_dependencies()
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
    source_columns = required_training_columns(dataset, validation)
    frame = data_frame_from_dataset(dataset, source_columns)
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
    validation_warnings = _dedupe_warnings(list(validation.get("warnings") or []))
    fit_intercept = bool(formula.get("fit_intercept", True))
    intercept_only = bool(formula.get("intercept_only", False))
    context = formula_context(np)
    offset_terms = [str(term) for term in formula.get("offset_terms", [])]

    internal_intercept_column = ""
    estimator_fitted_formula = str(formula["fitted"])
    estimator_fit_intercept = fit_intercept
    if intercept_only:
        internal_intercept_column = internal_intercept_column_name(list(frame.columns))
        frame = frame.with_columns(pl.lit(1.0).alias(internal_intercept_column))
        estimator_fitted_formula = f"{TARGET_COLUMN} ~ 0 + `{internal_intercept_column}`"
        estimator_fit_intercept = False

    offset_values = offset_values_for_polars_frame(frame, offset_terms, context, np, pl)

    response = polars_numeric_array(frame, response_column, pl)
    if denominator_column:
        denominator = polars_numeric_array(frame, denominator_column, pl)
        eligible_mask = np.isfinite(denominator) & (denominator > 0)
        with np.errstate(divide="ignore", invalid="ignore"):
            target = response / denominator
        fit_weight = denominator
    else:
        eligible_mask = np.ones(frame.height, dtype=bool)
        target = response
        fit_weight = None

    fit_mask = eligible_mask & np.isfinite(target)
    sample_column = physical_sample_column(dataset)
    if training_scope == "training":
        if not sample_column:
            raise ValueError("Training rows require a physical SAMPLE column")
        training_rows = (
            frame.get_column(sample_column)
            .cast(pl.String)
            .str.strip_chars()
            .str.to_lowercase()
            .eq("training")
            .fill_null(False)
            .to_numpy()
        )
        fit_mask = fit_mask & training_rows

    if fit_weight is not None:
        fit_mask = fit_mask & np.isfinite(fit_weight) & (fit_weight > 0)
    if offset_values is not None:
        fit_mask = fit_mask & np.isfinite(offset_values)

    if int(fit_mask.sum()) < 2:
        raise ValueError("GLM fitting needs at least two valid rows")

    y_fit = np.asarray(target[fit_mask], dtype=float)
    check_target_range(np, family, y_fit)
    fit_frame = frame.filter(pl.Series("__lucidum_fit_mask", fit_mask)).with_columns(pl.Series(TARGET_COLUMN, y_fit))
    fit_weight_values = np.asarray(fit_weight[fit_mask], dtype=float) if fit_weight is not None else None
    fit_offset_values = np.asarray(offset_values[fit_mask], dtype=float) if offset_values is not None else None
    design_diagnostics = design_matrix_diagnostics(
        estimator_fitted_formula,
        fit_frame,
        context,
        np,
        ensure_full_rank=drop_first,
    )
    if not is_penalized:
        check_unpenalized_design_matrix(design_diagnostics)
    design_warnings = design_matrix_warnings(design_diagnostics, penalized=is_penalized)

    progress({"phase": "fitting", "message": "Fitting GLM", "percent": 35, "training_rows": int(fit_mask.sum())})
    estimator_kwargs = {
        "family": glum_family(glum, family, float(family_param) if family_param is not None else None),
        "link": "auto",
        "fit_intercept": estimator_fit_intercept,
        "formula": estimator_fitted_formula,
        "drop_first": drop_first,
        "robust": True,
        "scale_predictors": bool(regularization.get("scale_predictors")),
    }
    if internal_intercept_column:
        estimator_kwargs["P1"] = np.zeros(1)
        estimator_kwargs["P2"] = np.zeros((1, 1))
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
    numerical_warnings: list[str] = []
    try:
        with _capture_glm_warnings() as captured_warnings:
            estimator.fit(
                fit_frame,
                sample_weight=fit_weight_values,
                store_covariance_matrix=not is_penalized,
                context=context,
                offset=fit_offset_values,
            )
        numerical_warnings.extend(_glm_numerical_warning_messages(captured_warnings))
    except Exception as exc:
        _raise_actionable_singular_matrix_error(exc)
    timings["fit_ms"] = _elapsed_ms(fit_started)
    regularization = regularization_summary(estimator, regularization, np)

    score_mask = np.asarray(eligible_mask, dtype=bool)
    if offset_values is not None:
        score_mask = score_mask & np.isfinite(offset_values)
    progress(
        {
            "phase": "scoring",
            "message": "Scoring GLM predictions",
            "percent": 70,
            "scoring_rows": int(score_mask.sum()),
        }
    )
    score_started = time.perf_counter()
    with _capture_glm_warnings() as captured_warnings:
        # Matching row sets can share one fitted matrix across both post-fit consumers.
        shared_design_matrix = _shared_post_fit_design_matrix(
            estimator,
            fit_frame,
            fit_mask,
            score_mask,
            context,
            np,
        )
        prediction_result = build_predictions_frame(
            frame,
            estimator,
            denominator_column,
            context,
            np,
            pl,
            offset_values=offset_values,
            design_matrix=shared_design_matrix,
        )
        predictions = prediction_result.frame
        scored_rows = prediction_result.scored_rows
        fitted_na_rows = prediction_result.fitted_na_rows
        fit_rows_within_score = fit_mask[prediction_result.score_mask]
        if int(fit_rows_within_score.sum()) != len(y_fit):
            raise RuntimeError("GLM scoring rows no longer align with the fitted rows")
        fit_predictions = (
            prediction_result.response_values
            if bool(fit_rows_within_score.all())
            else prediction_result.response_values[fit_rows_within_score]
        )
        coefficients = coefficient_rows(
            estimator,
            fit_frame,
            y_fit,
            fit_weight_values,
            context,
            np,
            pd,
            source_columns,
            include_inference=not is_penalized,
            internal_intercept_column=internal_intercept_column,
        )
        feature_importance = glm_feature_importance_rows(
            estimator,
            fit_frame,
            source_columns,
            fit_weight_values,
            context,
            np,
            pd,
            design_matrix=shared_design_matrix,
        )
        diagnostics = diagnostics_payload(
            estimator,
            y_fit,
            fit_predictions,
            fit_weight_values,
            np,
            len(coefficients),
        )
    numerical_warnings.extend(_glm_numerical_warning_messages(captured_warnings))
    model_warnings = _dedupe_warnings(
        [
            *validation_warnings,
            *design_warnings,
            *numerical_warnings,
            *_coefficient_inference_warnings(coefficients, include_inference=not is_penalized),
        ]
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
            "design_matrix": design_diagnostics,
            "warnings": model_warnings,
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
        "offset_terms": jsonable(offset_terms, np, pd),
        "training_scope": training_scope,
        "formula": {
            "drop_first": drop_first,
            "fit_intercept": fit_intercept,
            "estimator_fit_intercept": estimator_fit_intercept,
            "intercept_only": intercept_only,
            "internal_intercept_column": internal_intercept_column,
        },
        "timings": {},
    }

    progress(
        {
            "phase": "writing",
            "message": "Saving GLM artifacts",
            "percent": 90,
            "scoring_rows": int(score_mask.sum()),
        }
    )
    artifact_write_started = time.perf_counter()
    coefficient_frame = pl.DataFrame(
        coefficients,
        schema={
            "term": pl.String,
            "features": pl.List(pl.String),
            "estimate": pl.Float64,
            "std_error": pl.Float64,
            "statistic": pl.Float64,
            "p_value": pl.Float64,
            "ci_lower": pl.Float64,
            "ci_upper": pl.Float64,
        },
        strict=False,
    )
    feature_importance_frame = pl.DataFrame(
        feature_importance,
        schema={
            "feature": pl.String,
            "importance": pl.Float64,
            "term_count": pl.Int64,
            "metric": pl.String,
        },
        strict=False,
    )
    write_dataframe_parquet(coefficient_frame, store.artifact_path(model_id, "coefficients"))
    write_dataframe_parquet(feature_importance_frame, store.artifact_path(model_id, "feature_importance"))
    write_dataframe_parquet(predictions, store.artifact_path(model_id, "predictions"))
    with store.artifact_path(model_id, "estimator").open("wb") as handle:
        pickle.dump(estimator, handle, protocol=pickle.HIGHEST_PROTOCOL)
    store.artifact_path(model_id, "formula").write_text(str(formula["raw"]), encoding="utf-8")
    store.write_json(store.artifact_path(model_id, "diagnostics"), jsonable(diagnostics, np, pd))
    timings["artifact_write_ms"] = _elapsed_ms(artifact_write_started)
    timings["elapsed_ms"] = _elapsed_ms(started)
    manifest["timings"] = {
        "fit_ms": timings["fit_ms"],
        "elapsed_ms": timings["elapsed_ms"],
    }
    progress(
        {
            "phase": "writing",
            "message": "Saved GLM artifacts",
            "percent": 95,
            "scoring_rows": int(score_mask.sum()),
            "timings": timings,
        }
    )
    store.write_json(store.artifact_path(model_id, "manifest"), manifest)

    result = store.activate_model(model_id) if activate else store.model_list_item(model_dir, manifest, store.active_model_id())
    result["timings"] = jsonable(timings, np, pd)
    result["coefficients"] = coefficients
    result["feature_importance"] = feature_importance
    result["diagnostics"] = jsonable(diagnostics, np, pd)
    result["warnings"] = jsonable(model_warnings, np, pd)
    result["model_dir"] = str(model_dir)
    return result
__all__ = [
    "FEATURE_IMPORTANCE_METRIC",
    "FEATURE_IMPORTANCE_METRIC_LABEL",
    "MissingGlmDependency",
    "add_internal_intercept_column",
    "estimator_intercept_value",
    "glm_dependencies",
    "glm_training_dependencies",
    "glm_feature_importance_rows",
    "internal_intercept_column_from_manifest",
    "train_model",
    "write_dataframe_parquet",
]
