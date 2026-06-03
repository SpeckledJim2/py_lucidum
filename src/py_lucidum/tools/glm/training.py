from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
import json
import math
import subprocess
import sys
import tempfile
import time
import warnings
from pathlib import Path
from typing import Any, Callable

import duckdb

from py_lucidum.core import Dataset, quote_ident, sql_literal

from .store import GlmModelStore, json_safe_number
from .validation import TARGET_COLUMN, physical_sample_column, validate_request


ProgressCallback = Callable[[dict[str, Any]], None]
_GLUM_FIRST_IMPORT_SAW_LIGHTGBM: bool | None = None


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


def glm_dependencies() -> tuple[Any, Any, Any, Any]:
    global _GLUM_FIRST_IMPORT_SAW_LIGHTGBM
    missing: list[str] = []
    lightgbm_loaded = "lightgbm" in sys.modules
    try:
        import glum  # type: ignore[import-not-found]
        from glum import GeneralizedLinearRegressor  # type: ignore[import-not-found]
    except ImportError:
        glum = None
        GeneralizedLinearRegressor = None
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
        _GLUM_FIRST_IMPORT_SAW_LIGHTGBM = lightgbm_loaded
    return glum, GeneralizedLinearRegressor, np, pd


def should_isolate_glm_fit() -> bool:
    return "lightgbm" in sys.modules and _GLUM_FIRST_IMPORT_SAW_LIGHTGBM is not False


def train_model_in_subprocess(
    dataset: Dataset,
    payload: dict[str, Any],
    *,
    progress_callback: ProgressCallback | None = None,
    activate: bool = True,
) -> dict[str, Any]:
    progress = progress_callback or (lambda _progress: None)
    progress({"phase": "fitting", "message": "Starting isolated GLM worker", "percent": 30})
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
    if not response.get("ok"):
        error = str(response.get("error") or "GLM worker failed")
        raise RuntimeError(error)
    result = response.get("result")
    if not isinstance(result, dict):
        raise RuntimeError("GLM worker returned an invalid response")
    progress({"phase": "writing", "message": "GLM worker saved artifacts", "percent": 90})
    return result


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
        return np.where(condition, yes, no)

    def pmin(*values: Any) -> Any:
        arrays = [np.asarray(value) for value in values]
        if not arrays:
            raise ValueError("pmin requires at least one argument")
        return np.minimum.reduce(np.broadcast_arrays(*arrays))

    def pmax(*values: Any) -> Any:
        arrays = [np.asarray(value) for value in values]
        if not arrays:
            raise ValueError("pmax requires at least one argument")
        return np.maximum.reduce(np.broadcast_arrays(*arrays))

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


def data_frame_from_dataset(dataset: Dataset) -> tuple[Any, list[str]]:
    with dataset.lock:
        columns = [column.name for column in dataset.valid_schema_columns()]
        projection = ",\n  ".join(["ROW_NUMBER() OVER () AS __lucidum_row_id", *[quote_ident(name) for name in columns]])
        frame = dataset.con.execute(f"SELECT {projection} FROM {dataset.relation_sql()}").fetchdf()
    return frame, columns


def finite_mask(np: Any, values: Any) -> Any:
    return values.notna() & np.isfinite(values.astype(float))


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


def coefficient_rows(model: Any, fit_frame: Any, y_fit: Any, fit_weight: Any, context: dict[str, Any], np: Any, pd: Any) -> list[dict[str, Any]]:
    try:
        table = model.coef_table(fit_frame, y_fit, sample_weight=fit_weight, context=context)
    except Exception:
        table = None
    if table is not None:
        rows: list[dict[str, Any]] = []
        for term, row in table.iterrows():
            name = "(Intercept)" if str(term).lower() == "intercept" else str(term)
            rows.append(
                {
                    "term": name,
                    "estimate": jsonable(row.get("coef"), np, pd),
                    "std_error": jsonable(row.get("se"), np, pd),
                    "statistic": jsonable(row.get("z_value", row.get("t_value")), np, pd),
                    "p_value": jsonable(row.get("p_value"), np, pd),
                    "ci_lower": jsonable(row.get("ci_lower"), np, pd),
                    "ci_upper": jsonable(row.get("ci_upper"), np, pd),
                }
            )
        return rows

    terms = ["(Intercept)", *[str(name) for name in getattr(model, "feature_names_", [])]]
    estimates = [getattr(model, "intercept_", None), *list(getattr(model, "coef_", []))]
    return [
        {
            "term": term,
            "estimate": jsonable(estimate, np, pd),
            "std_error": None,
            "statistic": None,
            "p_value": None,
            "ci_lower": None,
            "ci_upper": None,
        }
        for term, estimate in zip(terms, estimates)
    ]


def safe_metric(callable_metric: Any, *args: Any, **kwargs: Any) -> float | None:
    try:
        return json_safe_number(callable_metric(*args, **kwargs))
    except Exception:
        return None


def diagnostics_payload(model: Any, fit_frame: Any, y_fit: Any, fit_weight: Any, context: dict[str, Any], np: Any, coefficient_count: int) -> dict[str, Any]:
    mu = model.predict(fit_frame, context=context)
    family = model.family_instance
    parameters = max(1, coefficient_count)
    deviance = safe_metric(family.deviance, y_fit, mu, sample_weight=fit_weight)
    dispersion = safe_metric(family.dispersion, y_fit, mu, sample_weight=fit_weight, ddof=parameters)
    mean_target = float(np.average(np.asarray(y_fit, dtype=float), weights=np.asarray(fit_weight, dtype=float))) if fit_weight is not None else float(np.mean(y_fit))
    null_mu = np.full_like(np.asarray(y_fit, dtype=float), mean_target, dtype=float)
    null_deviance = safe_metric(family.deviance, y_fit, null_mu, sample_weight=fit_weight)
    return {
        "deviance": deviance,
        "null_deviance": null_deviance,
        "aic": safe_metric(model.aic, fit_frame, y_fit, fit_weight, context=context),
        "aicc": safe_metric(model.aicc, fit_frame, y_fit, fit_weight, context=context),
        "bic": safe_metric(model.bic, fit_frame, y_fit, fit_weight, context=context),
        "dispersion": dispersion,
    }


def build_predictions_frame(frame: Any, model: Any, denominator_column: str, context: dict[str, Any], np: Any, pd: Any) -> tuple[Any, int, int]:
    score_frame = frame.copy()
    score_frame[TARGET_COLUMN] = np.nan
    if denominator_column:
        denominator = pd.to_numeric(score_frame[denominator_column], errors="coerce")
        score_mask = denominator.notna() & np.isfinite(denominator.astype(float)) & (denominator.astype(float) > 0)
    else:
        denominator = None
        score_mask = pd.Series(True, index=score_frame.index)

    output = score_frame.loc[score_mask, ["__lucidum_row_id"]].copy()
    predictions = model.predict(score_frame.loc[score_mask].copy(), context=context)
    prediction_values = pd.to_numeric(predictions, errors="coerce")
    if denominator is not None:
        prediction_values = prediction_values * denominator.loc[score_mask].to_numpy(dtype=float)
    finite = np.isfinite(np.asarray(prediction_values, dtype=float))
    output["glm_prediction"] = prediction_values
    output.loc[~finite, "glm_prediction"] = np.nan
    fitted_na_rows = int((~finite).sum())
    scored_rows = int(finite.sum())
    return output, scored_rows, fitted_na_rows


def train_model(
    dataset: Dataset,
    store: GlmModelStore,
    payload: dict[str, Any],
    *,
    progress_callback: ProgressCallback | None = None,
    activate: bool = True,
) -> dict[str, Any]:
    glm_dependencies()
    if should_isolate_glm_fit():
        return train_model_in_subprocess(dataset, payload, progress_callback=progress_callback, activate=activate)
    return _train_model_impl(dataset, store, payload, progress_callback=progress_callback, activate=activate)


def _train_model_impl(
    dataset: Dataset,
    store: GlmModelStore,
    payload: dict[str, Any],
    *,
    progress_callback: ProgressCallback | None = None,
    activate: bool = True,
) -> dict[str, Any]:
    glum, GeneralizedLinearRegressor, np, pd = glm_dependencies()
    validation = validate_request(dataset, payload)
    if not validation["ok"]:
        raise ValueError("; ".join(validation["errors"]))

    progress = progress_callback or (lambda _progress: None)
    progress({"phase": "loading", "message": "Loading GLM training data", "percent": 5})
    started = time.perf_counter()
    frame, source_columns = data_frame_from_dataset(dataset)

    response_column = str(validation["response_column"])
    denominator_column = str(validation["denominator_column"] or "")
    training_scope = str(validation["training_scope"])
    family = str(validation["family"])
    family_param = validation.get("family_parameter")
    formula = validation["formula"]
    context = formula_context(np)

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

    if int(fit_mask.sum()) < 2:
        raise ValueError("GLM fitting needs at least two valid rows")

    y_fit = target.loc[fit_mask].astype(float)
    check_target_range(np, family, y_fit)
    fit_frame = frame.loc[fit_mask].copy()
    fit_frame[TARGET_COLUMN] = y_fit.to_numpy(dtype=float)
    fit_weight_values = fit_weight.loc[fit_mask].astype(float).to_numpy() if fit_weight is not None else None

    progress({"phase": "fitting", "message": "Fitting GLM", "percent": 35, "training_rows": int(fit_mask.sum())})
    estimator = GeneralizedLinearRegressor(
        family=glum_family(glum, family, float(family_param) if family_param is not None else None),
        link="auto",
        alpha=0,
        fit_intercept=True,
        formula=str(formula["fitted"]),
        drop_first=False,
        robust=True,
    )
    with _suppress_tabmat_mixed_dtype_warning():
        estimator.fit(
            fit_frame,
            sample_weight=fit_weight_values,
            store_covariance_matrix=True,
            context=context,
        )

    progress({"phase": "scoring", "message": "Scoring GLM predictions", "percent": 70})
    with _suppress_tabmat_mixed_dtype_warning():
        predictions, scored_rows, fitted_na_rows = build_predictions_frame(frame, estimator, denominator_column, context, np, pd)
        coefficients = coefficient_rows(estimator, fit_frame, y_fit.to_numpy(dtype=float), fit_weight_values, context, np, pd)
        diagnostics = diagnostics_payload(estimator, fit_frame, y_fit.to_numpy(dtype=float), fit_weight_values, context, np, len(coefficients))
    diagnostics.update(
        {
            "training_rows": int(fit_mask.sum()),
            "eligible_rows": int(eligible_mask.sum()),
            "scored_rows": scored_rows,
            "fitted_na_rows": fitted_na_rows,
            "coefficient_count": len(coefficients),
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
        "response_column": response_column,
        "denominator_column": denominator_column,
        "offset_column": denominator_column,
        "training_scope": training_scope,
        "sample_column": sample_column,
        "formula": {
            "raw": formula["raw"],
            "stripped": formula["stripped"],
            "rhs": formula["rhs"],
            "fitted": formula["fitted"],
        },
        "diagnostics": jsonable(diagnostics, np, pd),
        "source_columns": source_columns,
        "sources": {"predictions": store.source_id(model_id)},
        "row_count": int(len(frame)),
        "training_rows": int(fit_mask.sum()),
        "scored_rows": scored_rows,
        "fitted_na_rows": fitted_na_rows,
        "coefficient_count": len(coefficients),
        "timings": {"elapsed_ms": round((time.perf_counter() - started) * 1000, 1)},
    }

    progress({"phase": "writing", "message": "Saving GLM artifacts", "percent": 90})
    coefficient_frame = pd.DataFrame(coefficients)
    write_dataframe_parquet(coefficient_frame, store.artifact_path(model_id, "coefficients"))
    write_dataframe_parquet(predictions, store.artifact_path(model_id, "predictions"))
    store.artifact_path(model_id, "formula").write_text(str(formula["raw"]), encoding="utf-8")
    store.write_json(store.artifact_path(model_id, "diagnostics"), manifest["diagnostics"])
    store.write_json(store.artifact_path(model_id, "manifest"), manifest)

    result = store.activate_model(model_id) if activate else manifest
    result["coefficients"] = coefficients
    result["diagnostics"] = manifest["diagnostics"]
    result["model_dir"] = str(model_dir)
    return result
__all__ = ["MissingGlmDependency", "glm_dependencies", "train_model", "write_dataframe_parquet"]
