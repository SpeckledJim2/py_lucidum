"""Save compact standalone results for externally fitted GLM and GBM models.

Most users should not read or modify this file.  The readable examples are
``01_external_glm_artifacts_demo.py`` and ``01_external_gbm_artifacts_demo.py``;
they contain the ordinary modelling workflow and make one call into this
module after fitting and prediction.

This module deliberately does not import :mod:`py_lucidum`.  It writes the
compact model, prediction, diagnostic, evaluation, tree, and SHAP artifacts
used by the reporting examples. Optional application installation is kept in
``lucidum_install.py``.
"""

from __future__ import annotations

import json
import math
import pickle
import re
import shutil
import time
import warnings as python_warnings
from pathlib import Path
from typing import Any
from uuid import uuid4

import duckdb
import numpy as np
import pandas as pd


FEATURE_IMPORTANCE_METRIC = "weighted_mean_abs_centered_linear_predictor_contribution"
GLM_COEFFICIENT_COLUMNS = [
    "term",
    "features",
    "estimate",
    "std_error",
    "statistic",
    "p_value",
    "ci_lower",
    "ci_upper",
]
MODEL_ID_RE = re.compile(r"[A-Za-z0-9_.-]+")


def save_glm_model_results(
    *,
    config: dict[str, Any],
    data: pd.DataFrame,
    formula_text: str,
    formula_context: dict[str, Any],
    model: Any,
    predictions: pd.Series,
    started: float,
) -> dict[str, Any]:
    """Save the compact fitted GLM results in their authoritative folder."""

    dataset = config["dataset"]
    model_config = config["model"]
    output = config["output"]
    regularization = model_config["regularization"]
    model_id = validate_model_id(model_config["id"])

    response = pd.to_numeric(data[str(dataset["response_numerator"])], errors="coerce")
    denominator_name = str(dataset.get("denominator") or "").strip()
    if denominator_name:
        denominator = pd.to_numeric(data[denominator_name], errors="coerce")
        scoring_mask = denominator.notna() & np.isfinite(denominator) & denominator.gt(0)
        target = response / denominator
        weights = denominator
    else:
        denominator = None
        scoring_mask = pd.Series(True, index=data.index)
        target = response
        weights = None

    sample = data[str(dataset["sample_column"])].astype("string").str.strip().str.lower()
    training_value = str(dataset["training_value"]).strip().lower()
    training_mask = (
        scoring_mask
        & target.notna()
        & np.isfinite(target)
        & sample.eq(training_value).fillna(False)
    )
    fit_frame = data.loc[training_mask]
    training_target = target.loc[training_mask].to_numpy(dtype=float)
    training_weights = (
        weights.loc[training_mask].to_numpy(dtype=float) if weights is not None else None
    )
    fit_predictions = np.asarray(
        model.predict(fit_frame, context=formula_context),
        dtype=float,
    )

    prediction_series = aligned_predictions(data, predictions)
    rate_predictions = prediction_series.loc[scoring_mask].to_numpy(dtype=float)
    saved_predictions = (
        rate_predictions * denominator.loc[scoring_mask].to_numpy(dtype=float)
        if denominator is not None
        else rate_predictions
    )
    scored_row_ids = np.flatnonzero(scoring_mask.to_numpy()).astype("int64") + 1
    source_columns = [str(name) for name in data.columns]

    alpha = float(regularization["alpha"])
    coefficients, inference_warning = glm_coefficient_rows(
        model,
        source_columns,
        include_inference=alpha == 0,
    )
    importance = glm_feature_importance(
        model,
        fit_frame,
        source_columns,
        training_weights,
        formula_context,
    )
    diagnostics = glm_diagnostics(
        model,
        training_target,
        fit_predictions,
        training_weights,
        coefficients,
    )
    diagnostics.update(
        {
            "training_rows": int(len(fit_frame)),
            "eligible_rows": int(len(saved_predictions)),
            "scored_rows": int(np.isfinite(saved_predictions).sum()),
            "fitted_na_rows": int((~np.isfinite(saved_predictions)).sum()),
            "coefficient_count": len(coefficients),
            "n_terms": len(coefficients),
            "n_features": len({feature for row in coefficients for feature in row["features"]}),
            "n_interactions": len(
                {tuple(sorted(row["features"])) for row in coefficients if len(row["features"]) > 1}
            ),
            "warnings": [inference_warning] if inference_warning else [],
        }
    )

    l1_ratio = float(regularization["l1_ratio"])
    drop_first = alpha == 0
    nonzero = int(np.count_nonzero(np.abs(np.asarray(model.coef_, dtype=float)) > 1e-10))
    family_name = str(model_config["family"])
    family_parameter = (
        float(model_config.get("family_parameter", 1.5))
        if family_name.strip().casefold() == "tweedie"
        else model_config.get("family_parameter")
    )
    manifest = {
        "model_id": model_id,
        "label": str(model_config["label"]),
        "tool": "glm",
        "created_at": utc_now(),
        "family": family_name,
        "link": str(model_config["link"]),
        "family_parameter": family_parameter,
        "regularization": {
            "mode": "none" if alpha == 0 else "manual",
            "alpha": alpha,
            "l1_ratio": l1_ratio,
            "scale_predictors": bool(regularization["scale_predictors"]),
            "selected_alpha": alpha,
            "selected_l1_ratio": l1_ratio,
            "nonzero_coefficients": nonzero,
            "coefficient_count": int(len(model.coef_)),
        },
        "response_column": str(dataset["response_numerator"]),
        "denominator_column": denominator_name,
        "offset_terms": [],
        "training_scope": "training",
        "formula": {
            "drop_first": drop_first,
            "fit_intercept": bool(model_config["fit_intercept"]),
            "estimator_fit_intercept": bool(model_config["fit_intercept"]),
            "intercept_only": False,
            "internal_intercept_column": "",
        },
        "timings": {"elapsed_ms": round((time.perf_counter() - started) * 1000, 1)},
    }

    prediction_frame = pd.DataFrame(
        {"__lucidum_row_id": scored_row_ids, "glm_prediction": saved_predictions}
    )
    if denominator is not None:
        prediction_frame["glm_prediction_rate"] = rate_predictions

    model_results_root = resolve_path(config["_config_dir"], output["model_results_root"])
    model_dir, staging = new_model_staging(model_results_root, "glm", model_id)
    try:
        write_parquet(
            pd.DataFrame(coefficients, columns=GLM_COEFFICIENT_COLUMNS),
            staging / "coefficients.parquet",
        )
        write_parquet(pd.DataFrame(importance), staging / "feature_importance.parquet")
        write_parquet(prediction_frame, staging / "predictions.parquet")
        with (staging / "estimator.pkl").open("wb") as handle:
            pickle.dump(model, handle, protocol=pickle.HIGHEST_PROTOCOL)
        (staging / "formula.txt").write_text(formula_text, encoding="utf-8")
        write_json(staging / "diagnostics.json", diagnostics)
        write_json(staging / "manifest.json", manifest)
        replace_directory(staging, model_dir, replace_existing=bool(output["replace_existing"]))
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise

    return {"model_id": model_id, "model_folder": model_dir}


def save_gbm_model_results(
    *,
    config: dict[str, Any],
    data: pd.DataFrame,
    feature_data: pd.DataFrame,
    model: Any,
    evaluation: dict[str, dict[str, list[float | None]]],
    predictions: pd.Series,
    started: float,
    warnings: list[str] | None = None,
) -> dict[str, Any]:
    """Save the compact fitted GBM results in their authoritative folder."""

    dataset = config["dataset"]
    features_config = config["features"]
    model_config = config["model"]
    training = config["training"]
    output = config["output"]
    parameters = jsonable(dict(training["parameters"]))
    model_id = validate_model_id(model_config["id"])

    feature_names = [str(name) for name in model.feature_name()]
    categorical_labels = {
        name: [str(value) for value in feature_data[name].cat.categories]
        for name in feature_names
        if isinstance(feature_data[name].dtype, pd.CategoricalDtype)
    }
    feature_rows = [
        {
            "name": name,
            "kind": (
                "categorical"
                if name in categorical_labels
                else "integer"
                if pd.api.types.is_integer_dtype(feature_data[name])
                else "numeric"
            ),
        }
        for name in feature_names
    ]

    response = pd.to_numeric(data[str(dataset["response_numerator"])], errors="coerce")
    denominator_name = str(dataset.get("denominator") or "").strip()
    if denominator_name:
        denominator = pd.to_numeric(data[denominator_name], errors="coerce")
        eligible = (
            response.notna()
            & np.isfinite(response)
            & denominator.notna()
            & np.isfinite(denominator)
            & denominator.gt(0)
        )
    else:
        denominator = None
        eligible = response.notna() & np.isfinite(response)

    sample = data[str(dataset["sample_column"])].astype("string").str.strip().str.lower()
    training_mask = sample_rows(sample, dataset["training_value"], eligible)
    test_mask = sample_rows(sample, dataset["early_stopping_value"], eligible)
    validation_mask = sample_rows(sample, dataset["validation_value"], eligible)
    best_iteration = int(model.best_iteration or model.current_iteration())

    prediction_series = aligned_predictions(data, predictions)
    saved_predictions = prediction_series.loc[eligible].to_numpy(dtype=float)
    row_ids = np.flatnonzero(eligible.to_numpy()).astype("int64") + 1

    prediction_frame = pd.DataFrame(
        {"__lucidum_row_id": row_ids, "gbm_prediction": saved_predictions}
    )
    if denominator is not None:
        prediction_frame["gbm_prediction_rate"] = (
            saved_predictions / denominator.loc[eligible].to_numpy(dtype=float)
        )

    evaluation_frame = gbm_evaluation_frame(evaluation)
    tree_frame = gbm_tree_frame(model, categorical_labels)
    shap_values, shap_summary = gbm_shap_frames(
        data=data,
        feature_frame=feature_data,
        feature_names=feature_names,
        booster=model,
        best_iteration=best_iteration,
        eligible=eligible.to_numpy(),
        requested=training["shap_rows"],
        seed=int(parameters.get("seed", 2026)),
    )
    gains = {
        name: float(value or 0.0)
        for name, value in zip(
            feature_names,
            model.feature_importance(importance_type="gain", iteration=best_iteration),
        )
    }
    shap_importance = {
        str(row["feature"]): float(row["mean_abs_shap"])
        for row in shap_summary.to_dict("records")
    }
    monotone_constraints = parameters.get("monotone_constraints")
    monotone_values = list(monotone_constraints) if isinstance(monotone_constraints, list) else []
    saved_features = []
    for index, row in enumerate(feature_rows):
        monotonicity = normalise_monotonicity(
            monotone_values[index] if index < len(monotone_values) else 0
        )
        saved_features.append(
            {
                "name": row["name"],
                "kind": row["kind"],
                "include": True,
                "monotonicity": "Increasing" if monotonicity > 0 else "Decreasing" if monotonicity < 0 else "",
                "monotonicity_value": monotonicity,
                "gain": round(gains.get(row["name"], 0.0), 3),
                "mean_abs_shap": shap_importance.get(row["name"]),
            }
        )
    saved_features.sort(key=lambda row: (-float(row["gain"]), str(row["name"]).lower()))

    stored_parameters = dict(parameters)
    stored_parameters["num_iterations"] = int(training["num_boost_round"])
    stored_parameters["early_stopping_rounds"] = int(training["early_stopping_rounds"])
    manifest = {
        "model_id": model_id,
        "label": str(model_config["label"]),
        "created_at": utc_now(),
        "training_mode": "normal",
        "response_column": str(dataset["response_numerator"]),
        "offset_column": denominator_name,
        "best_iteration": best_iteration,
        "training_rows": int(training_mask.sum()),
        "test_rows": int(test_mask.sum()),
        "validation_rows": int(validation_mask.sum()),
        "scored_rows": int(eligible.sum()),
        "sample_column": str(dataset["sample_column"]),
        "sample_source": "dataset",
        "shap_rows": int(len(shap_values)),
        "timings": {"training_seconds": round(time.perf_counter() - started, 3)},
        "warnings": [str(warning) for warning in warnings or [] if str(warning).strip()],
        "feature_scenario": {
            "name": str(features_config["scenario_column"]),
            "features": feature_names,
        },
        "feature_interaction_group_models": {
            "enabled": False,
            "error_metric": "max_absolute_error",
            "groups": [],
        },
        "init_score": {"value": "none", "kind": "none", "transform": None},
    }

    model_results_root = resolve_path(config["_config_dir"], output["model_results_root"])
    model_dir, staging = new_model_staging(model_results_root, "gbm", model_id)
    try:
        write_parquet(prediction_frame, staging / "predictions.parquet")
        write_parquet(evaluation_frame, staging / "evaluation.parquet")
        write_parquet(tree_frame, staging / "tree_table.parquet")
        if not shap_values.empty:
            write_parquet(shap_values, staging / "shap_values.parquet")
            write_parquet(shap_summary, staging / "shap_summary.parquet")
        write_parquet(pd.DataFrame(saved_features), staging / "feature_config.parquet")
        model.save_model(str(staging / "model.txt"), num_iteration=best_iteration)
        write_json(staging / "features.json", feature_names)
        write_json(staging / "parameters.json", stored_parameters)
        write_json(staging / "manifest.json", manifest)
        replace_directory(staging, model_dir, replace_existing=bool(output["replace_existing"]))
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise

    return {"model_id": model_id, "model_folder": model_dir}


# ---------------------------------------------------------------------------
# GLM display artifacts
# ---------------------------------------------------------------------------


def glm_coefficient_rows(
    model: Any,
    source_columns: list[str],
    *,
    include_inference: bool,
) -> tuple[list[dict[str, Any]], str | None]:
    feature_rows = coefficient_feature_rows(model, source_columns)
    names = [str(name) for name in getattr(model, "feature_names_", [])]
    features_by_name = {
        name: feature_rows[index] if index < len(feature_rows) else []
        for index, name in enumerate(names)
    }

    if include_inference:
        try:
            if getattr(model, "covariance_matrix_", None) is None:
                raise ValueError("the fitted model has no stored covariance matrix")
            table = model.coef_table()
            missing = {"coef", "se", "p_value"} - set(table.columns)
            if missing:
                raise ValueError(
                    "glum's coefficient table is missing " + ", ".join(sorted(missing))
                )

            rows = []
            coefficient_index = 0
            for term, values in table.iterrows():
                raw_term = str(term)
                is_intercept = raw_term.lower() == "intercept" or raw_term == "(Intercept)"
                features = []
                if not is_intercept:
                    features = features_by_name.get(
                        raw_term,
                        feature_rows[coefficient_index]
                        if coefficient_index < len(feature_rows)
                        else [],
                    )
                    coefficient_index += 1
                statistic = (
                    values.get("z_value")
                    if "z_value" in table.columns
                    else values.get("t_value")
                )
                rows.append(
                    {
                        "term": "(Intercept)" if is_intercept else raw_term,
                        "features": list(features),
                        "estimate": json_number(values.get("coef")),
                        "std_error": json_number(values.get("se")),
                        "statistic": json_number(statistic),
                        "p_value": json_number(values.get("p_value")),
                        "ci_lower": json_number(values.get("ci_lower")),
                        "ci_upper": json_number(values.get("ci_upper")),
                    }
                )

            inference_fields = (
                "std_error",
                "statistic",
                "p_value",
                "ci_lower",
                "ci_upper",
            )
            if rows and all(
                row[field] is not None
                for row in rows
                for field in inference_fields
            ):
                return rows, None
            raise ValueError("glum returned one or more non-finite inference values")
        except Exception as exc:
            inference_warning = (
                "External GLM coefficient inference could not be exported; standard errors, "
                f"statistics, p-values, and confidence intervals are blank. {exc}"
            )
            python_warnings.warn(inference_warning, RuntimeWarning, stacklevel=2)
    else:
        inference_warning = None

    coefficients = np.asarray(model.coef_, dtype=float)
    rows: list[dict[str, Any]] = []
    if bool(getattr(model, "fit_intercept", False)):
        rows.append(
            {
                "term": "(Intercept)",
                "features": [],
                "estimate": json_number(model.intercept_),
                "std_error": None,
                "statistic": None,
                "p_value": None,
                "ci_lower": None,
                "ci_upper": None,
            }
        )
    for index, name in enumerate(names):
        rows.append(
            {
                "term": name,
                "features": feature_rows[index] if index < len(feature_rows) else [],
                "estimate": json_number(coefficients[index] if index < len(coefficients) else None),
                "std_error": None,
                "statistic": None,
                "p_value": None,
                "ci_lower": None,
                "ci_upper": None,
            }
        )
    return rows, inference_warning


def coefficient_feature_rows(model: Any, source_columns: list[str]) -> list[list[str]]:
    names = [str(name) for name in getattr(model, "feature_names_", [])]
    term_names = list(getattr(getattr(model, "formula", None), "term_names", []))
    terms = [str(name) for name in term_names if str(name) not in {"1", "0", "-1"}]
    rows: list[list[str]] = []
    for index, name in enumerate(names):
        expression = terms[index] if index < len(terms) else name
        rows.append(column_tokens(expression, source_columns))
    return rows


def glm_feature_importance(
    model: Any,
    frame: pd.DataFrame,
    source_columns: list[str],
    weights: np.ndarray | None,
    formula_context: dict[str, Any],
) -> list[dict[str, Any]]:
    import polars as pl

    matrix = model.X_model_spec_.get_model_matrix(
        pl.from_pandas(frame),
        context=formula_context,
    )
    if hasattr(matrix, "toarray"):
        matrix = matrix.toarray()
    elif hasattr(matrix, "to_numpy"):
        matrix = matrix.to_numpy()
    matrix = np.asarray(matrix, dtype=float)
    coefficients = np.asarray(model.coef_, dtype=float)
    if bool(getattr(model, "fit_intercept", False)) and matrix.shape[1] == len(coefficients) + 1:
        matrix = matrix[:, 1:]
    feature_rows = coefficient_feature_rows(model, source_columns)
    grouped: dict[str, list[int]] = {}
    for index, features in enumerate(feature_rows):
        for feature in features:
            grouped.setdefault(feature, []).append(index)
    rows = []
    for feature, indexes in grouped.items():
        valid = [index for index in indexes if index < matrix.shape[1] and index < len(coefficients)]
        if not valid:
            continue
        contribution = np.asarray(matrix[:, valid].dot(coefficients[valid]), dtype=float).reshape(-1)
        finite = np.isfinite(contribution)
        if not finite.any():
            continue
        values = contribution[finite]
        active_weights = np.asarray(weights, dtype=float)[finite] if weights is not None else None
        centre = float(np.average(values, weights=active_weights)) if active_weights is not None else float(np.mean(values))
        importance = (
            float(np.average(np.abs(values - centre), weights=active_weights))
            if active_weights is not None
            else float(np.mean(np.abs(values - centre)))
        )
        rows.append({"feature": feature, "importance": importance, "metric": FEATURE_IMPORTANCE_METRIC})
    return sorted(rows, key=lambda row: (-float(row["importance"]), str(row["feature"]).lower()))


def glm_diagnostics(
    model: Any,
    target: np.ndarray,
    prediction: np.ndarray,
    weights: np.ndarray | None,
    coefficients: list[dict[str, Any]],
) -> dict[str, Any]:
    family = model.family_instance
    deviance = safe_metric(family.deviance, target, prediction, sample_weight=weights)
    log_likelihood = safe_metric(family.log_likelihood, target, prediction, sample_weight=weights)
    aic = None if log_likelihood is None else 2 * len(coefficients) - 2 * log_likelihood
    return {
        "deviance": deviance,
        "log_likelihood": log_likelihood,
        "aic": json_number(aic),
        "dispersion": safe_metric(
            family.dispersion,
            target,
            prediction,
            sample_weight=weights,
            ddof=max(1, len(coefficients)),
        ),
    }


# ---------------------------------------------------------------------------
# GBM display artifacts
# ---------------------------------------------------------------------------


def gbm_evaluation_frame(
    evaluation: dict[str, dict[str, list[float | None]]],
) -> pd.DataFrame:
    rows = []
    for dataset_name, metrics in evaluation.items():
        for metric_name, values in metrics.items():
            for iteration, value in enumerate(values, start=1):
                if json_number(value) is None:
                    continue
                rows.append(
                    {
                        "dataset": dataset_name,
                        "metric": metric_name,
                        "iteration": iteration,
                        "value": json_number(value),
                    }
                )
    return pd.DataFrame(rows, columns=["dataset", "metric", "iteration", "value"])


def gbm_tree_frame(booster: Any, categorical_labels: dict[str, list[str]]) -> pd.DataFrame:
    frame = booster.trees_to_dataframe().copy()
    if "threshold_label" not in frame.columns:
        frame["threshold_label"] = None
    if frame.empty:
        return frame
    labels = []
    for _, row in frame.iterrows():
        feature = str(row.get("split_feature") or "")
        categorical_split = feature in categorical_labels and str(row.get("decision_type") or "") == "=="
        labels.append(
            decode_categorical_threshold(row.get("threshold"), categorical_labels[feature])
            if categorical_split and not pd.isna(row.get("threshold"))
            else None
        )
    frame["threshold_label"] = labels
    return frame


def gbm_shap_frames(
    *,
    data: pd.DataFrame,
    feature_frame: pd.DataFrame,
    feature_names: list[str],
    booster: Any,
    best_iteration: int,
    eligible: np.ndarray,
    requested: Any,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    positions = np.flatnonzero(eligible)
    count = shap_row_count(requested, len(positions))
    if count == 0:
        return pd.DataFrame(), pd.DataFrame(
            columns=["feature", "mean_abs_shap", "mean_shap", "row_count"]
        )
    if count < len(positions):
        positions = np.sort(np.random.default_rng(seed).choice(positions, size=count, replace=False))
    contributions = np.asarray(
        booster.predict(
            feature_frame.loc[positions, feature_names],
            pred_contrib=True,
            num_iteration=best_iteration,
        )
    )
    if contributions.ndim == 3:
        contributions = contributions[:, :, 0]
    feature_contributions = contributions[:, : len(feature_names)]
    values = pd.DataFrame(feature_contributions, columns=feature_names)
    values.insert(0, "__lucidum_row_id", positions.astype("int64") + 1)
    summary = pd.DataFrame(
        [
            {
                "feature": name,
                "mean_abs_shap": float(np.mean(np.abs(feature_contributions[:, index]))),
                "mean_shap": float(np.mean(feature_contributions[:, index])),
                "row_count": int(len(values)),
            }
            for index, name in enumerate(feature_names)
        ]
    ).sort_values("mean_abs_shap", ascending=False, ignore_index=True)
    return values, summary


def decode_categorical_threshold(value: Any, categories: list[str]) -> str | None:
    labels = []
    for part in str(value).split("||"):
        try:
            index = int(float(part))
        except ValueError:
            continue
        labels.append(categories[index] if 0 <= index < len(categories) else str(index))
    return " / ".join(labels) if labels else None


# ---------------------------------------------------------------------------
# Result-folder staging
# ---------------------------------------------------------------------------


def new_model_staging(root: Path, model_type: str, model_id: str) -> tuple[Path, Path]:
    parent = root / model_type
    return parent / model_id, create_staging_dir(parent, model_id)


def create_staging_dir(parent: Path, model_id: str) -> Path:
    parent.mkdir(parents=True, exist_ok=True)
    path = parent / f".{model_id}.tmp-{uuid4().hex}"
    path.mkdir()
    return path


def replace_directory(staging: Path, target: Path, *, replace_existing: bool) -> None:
    if staging.parent.resolve() != target.parent.resolve():
        raise ValueError("Refusing to replace a model outside its validated parent")
    backup: Path | None = None
    if target.exists():
        if not replace_existing:
            raise FileExistsError(f"Model already exists: {target}")
        backup = target.parent / f".{target.name}.backup-{uuid4().hex}"
        target.rename(backup)
    try:
        staging.rename(target)
    except Exception:
        if backup is not None and backup.exists() and not target.exists():
            backup.rename(target)
        raise
    if backup is not None:
        shutil.rmtree(backup)


# ---------------------------------------------------------------------------
# Small serialization helpers
# ---------------------------------------------------------------------------


def aligned_predictions(data: pd.DataFrame, predictions: pd.Series) -> pd.Series:
    """Keep exported predictions tied to their original source-row positions."""

    if not isinstance(predictions, pd.Series):
        raise ValueError("Predictions must be a pandas Series aligned to the source data")
    if len(predictions) != len(data) or not predictions.index.equals(data.index):
        raise ValueError("Predictions must use the same index as the source data")
    return pd.to_numeric(predictions, errors="coerce")


def sample_rows(sample: pd.Series, value: Any, eligible: pd.Series) -> pd.Series:
    """Recreate a configured sample mask for saved-model metadata."""

    return eligible & sample.eq(str(value).strip().lower()).fillna(False)


def validate_model_id(value: Any) -> str:
    model_id = str(value or "").strip()
    if model_id in {"", ".", ".."} or not MODEL_ID_RE.fullmatch(model_id):
        raise ValueError("model.id must contain only letters, numbers, dots, underscores, and hyphens")
    return model_id


def resolve_path(config_dir: Any, value: Any) -> Path:
    path = Path(str(value)).expanduser()
    return (Path(config_dir) / path).resolve() if not path.is_absolute() else path.resolve()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + f".tmp-{uuid4().hex}")
    temp.write_text(json.dumps(jsonable(payload), indent=2, sort_keys=True), encoding="utf-8")
    temp.replace(path)


def write_parquet(frame: pd.DataFrame, path: Path) -> None:
    con = duckdb.connect(database=":memory:")
    try:
        con.register("artifact_frame", frame)
        con.execute(f"COPY artifact_frame TO {sql_literal(str(path))} (FORMAT PARQUET)")
    finally:
        con.close()


def jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    if isinstance(value, np.generic):
        return jsonable(value.item())
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return value


def json_number(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def safe_metric(function: Any, *args: Any, **kwargs: Any) -> float | None:
    try:
        return json_number(function(*args, **kwargs))
    except Exception:
        return None


def column_tokens(expression: str, columns: list[str]) -> list[str]:
    found = []
    for column in sorted(columns, key=len, reverse=True):
        if re.search(rf"(?<![A-Za-z0-9_]){re.escape(column)}(?![A-Za-z0-9_])", expression):
            found.append(column)
    return sorted(set(found))


def normalise_monotonicity(value: Any) -> int:
    try:
        number = int(float(value))
    except (TypeError, ValueError):
        return 0
    return 1 if number > 0 else -1 if number < 0 else 0


def shap_row_count(value: Any, available: int) -> int:
    if isinstance(value, str) and value.strip().lower() == "all":
        return available
    return min(available, max(0, int(value)))


def sql_literal(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
