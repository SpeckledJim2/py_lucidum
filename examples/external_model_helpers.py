"""Small, non-Lucidum helpers shared by the external modelling examples.

The GLM and GBM scripts are the examples users should read and adapt.  This
module only keeps command-line parsing, YAML checks, and routine data loading
out of their main modelling flow.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Any

import duckdb
import numpy as np
import pandas as pd
import yaml


MODEL_ID_RE = re.compile(r"[A-Za-z0-9_.-]+")

# Keep this import-independent: the external workflow must not need py_lucidum
# in order to reproduce the effective parameters used by an in-app build.
GBM_PARAMETER_DEFAULTS = {
    "objective": "poisson",
    "metric": "poisson",
    "tweedie_variance_power": 1.5,
    "data_sample_strategy": "bagging",
    "monotone_constraints_method": "advanced",
    "learning_rate": 0.3,
    "num_leaves": 5,
    "max_depth": -1,
    "min_data_in_leaf": 50,
    "feature_fraction": 1.0,
    "bagging_fraction": 1.0,
    "bagging_freq": 0,
    "lambda_l1": 0.0,
    "lambda_l2": 0.0,
    "min_gain_to_split": 0.0,
    "max_bin": 255,
    "num_threads": 0,
    "verbosity": -1,
    "seed": 42,
}
MONOTONE_CONSTRAINT_PARAMETER_NAMES = {
    "monotone_constraints",
    "monotone_constraint",
    "monotonic_cst",
    "mc",
}
MONOTONE_CONSTRAINT_METHODS = {"basic", "intermediate", "advanced"}
MONOTONE_OBJECTIVES = {
    "regression",
    "regression_l1",
    "huber",
    "fair",
    "poisson",
    "gamma",
    "tweedie",
    "binary",
}

CONFIG_KEYS = {
    "glm": {
        "config": {"dataset", "model", "output"},
        "dataset": {
            "path",
            "response_numerator",
            "denominator",
            "sample_column",
            "training_value",
        },
        "model": {
            "id",
            "label",
            "formula_path",
            "family",
            "link",
            "fit_intercept",
            "regularization",
        },
        "model.regularization": {"alpha", "l1_ratio", "scale_predictors"},
        "output": {"model_results_root", "install_in_lucidum", "replace_existing"},
    },
    "gbm": {
        "config": {"dataset", "features", "model", "training", "output"},
        "dataset": {
            "path",
            "response_numerator",
            "denominator",
            "sample_column",
            "training_value",
            "early_stopping_value",
            "validation_value",
        },
        "features": {"spec_path", "scenario_column"},
        "model": {"id", "label"},
        "training": {
            "num_boost_round",
            "early_stopping_rounds",
            "shap_rows",
            "parameters",
        },
        "output": {"model_results_root", "install_in_lucidum", "replace_existing"},
    },
}

OPTIONAL_CONFIG_KEYS = {
    "gbm": {
        "features": {"use_monotonicity"},
    },
    "glm": {
        "dataset": {"test_value", "validation_value"},
        "model": {"family_parameter", "training_scope"},
    },
}


def config_path_from_command_line(script_file: str | None, default_name: str) -> Path:
    """Return the optional YAML argument, or the config beside the script.

    When a section is run interactively rather than as a command, use the
    default file and ignore the notebook or IDE process's unrelated arguments.
    """

    if script_file is None:
        return Path(__file__).resolve().with_name(default_name)

    script_path = Path(script_file).resolve()
    if not _running_as_script(script_path):
        return script_path.with_name(default_name)

    parser = argparse.ArgumentParser()
    parser.add_argument("config", nargs="?", default=script_path.with_name(default_name))
    return Path(parser.parse_args().config).expanduser().resolve()


def load_config(path: Path, model_type: str) -> dict[str, Any]:
    """Read and perform the small set of checks needed for a clear example."""

    if not path.is_file():
        raise ValueError(f"Config file does not exist: {path}")
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("Config must be a YAML mapping")

    schema = CONFIG_KEYS[model_type]
    _expect_keys(config, schema["config"], "config")
    for section in schema:
        if section == "config":
            continue
        value = config
        for name in section.split("."):
            value = value[name]
        _expect_keys(
            value,
            schema[section],
            section,
            optional=OPTIONAL_CONFIG_KEYS.get(model_type, {}).get(section, set()),
        )

    model_id = str(config["model"]["id"] or "").strip()
    if model_id in {"", ".", ".."} or not MODEL_ID_RE.fullmatch(model_id):
        raise ValueError(
            "model.id must contain only letters, numbers, dots, underscores, and hyphens"
        )

    if model_type == "gbm":
        raw_use_monotonicity = config["features"].get("use_monotonicity", True)
        if not isinstance(raw_use_monotonicity, bool):
            raise ValueError("features.use_monotonicity must be true or false")
        config["features"]["use_monotonicity"] = raw_use_monotonicity
        sample_values = {
            str(config["dataset"][name]).strip().lower()
            for name in ("training_value", "early_stopping_value", "validation_value")
        }
        if len(sample_values) != 3:
            raise ValueError("Training, test, and validation sample values must be distinct")
    else:
        training_scope = str(config["model"].get("training_scope") or "training").strip().lower()
        if training_scope not in {"all", "training", "training_test"}:
            raise ValueError("model.training_scope must be all, training, or training_test")
        config["model"]["training_scope"] = training_scope
        sample_values = {
            "training_value": str(config["dataset"]["training_value"]).strip(),
            "test_value": str(config["dataset"].get("test_value") or "test").strip(),
            "validation_value": str(
                config["dataset"].get("validation_value") or "validation"
            ).strip(),
        }
        if not sample_values["training_value"]:
            raise ValueError("dataset.training_value must not be blank")
        if len({value.lower() for value in sample_values.values()}) != 3:
            raise ValueError("GLM Training, Test, and Validation sample values must be distinct")
        config["dataset"].update(sample_values)

    if model_type == "glm" and str(config["model"]["family"]).strip().casefold() == "tweedie":
        raw_power = config["model"].get("family_parameter", 1.5)
        try:
            power = float(raw_power)
        except (TypeError, ValueError) as exc:
            raise ValueError("model.family_parameter must be a numeric Tweedie variance power") from exc
        if not 1.0 <= power <= 2.0:
            raise ValueError("model.family_parameter must be between 1 and 2 for a Tweedie GLM")

    config["_config_dir"] = path.parent
    return config


def resolve_path(config: dict[str, Any], value: Any) -> Path:
    """Resolve a YAML path relative to the YAML file."""

    path = Path(str(value)).expanduser()
    return (Path(config["_config_dir"]) / path).resolve() if not path.is_absolute() else path.resolve()


def read_table(path: Path) -> pd.DataFrame:
    """Read one CSV or Parquet file and give it a simple zero-based index."""

    if not path.is_file() or path.suffix.lower() not in {".csv", ".parquet"}:
        raise ValueError(f"Choose one CSV or Parquet dataset: {path}")
    data = pd.read_parquet(path) if path.suffix.lower() == ".parquet" else pd.read_csv(path)
    return data.reset_index(drop=True)


def dataset_column_kinds(path: Path) -> dict[str, str]:
    """Return Lucidum-compatible feature kinds from DuckDB's source schema."""

    if not path.is_file() or path.suffix.lower() not in {".csv", ".parquet"}:
        raise ValueError(f"Choose one CSV or Parquet dataset: {path}")
    literal = "'" + str(path).replace("'", "''") + "'"
    relation = (
        f"read_parquet({literal})"
        if path.suffix.lower() == ".parquet"
        else f"read_csv_auto({literal}, header=true, ignore_errors=true)"
    )
    con = duckdb.connect(database=":memory:")
    try:
        describe = con.execute(f"DESCRIBE SELECT * FROM {relation}").fetchall()
    finally:
        con.close()
    return {str(row[0]): _infer_source_kind(str(row[1])) for row in describe}


def effective_gbm_parameters(training: dict[str, Any]) -> dict[str, Any]:
    """Mirror Lucidum's complete effective LightGBM parameter dictionary."""

    parameters = dict(GBM_PARAMETER_DEFAULTS)
    raw = training.get("parameters")
    if isinstance(raw, dict):
        for key, value in raw.items():
            name = str(key).strip()
            if name and name not in {"init_score", "num_iterations", "early_stopping_rounds"}:
                if name in MONOTONE_CONSTRAINT_PARAMETER_NAMES:
                    raise ValueError(
                        "Set monotonicity in the Feature Specification instead of "
                        f"training.parameters.{name}"
                    )
                parameters[name] = _coerce_gbm_parameter(value)
    parameters["objective"] = str(parameters.get("objective") or "poisson").strip().lower()
    parameters["metric"] = str(parameters.get("metric") or "poisson").strip().lower()
    monotone_method = str(
        parameters.get("monotone_constraints_method") or "advanced"
    ).strip().lower()
    if monotone_method not in MONOTONE_CONSTRAINT_METHODS:
        raise ValueError(
            "training.parameters.monotone_constraints_method must be "
            "basic, intermediate, or advanced"
        )
    parameters["monotone_constraints_method"] = monotone_method
    return parameters


def lightgbm_fit_parameters(parameters: dict[str, Any]) -> dict[str, Any]:
    """Return the active LightGBM parameters without changing provenance."""

    fit_parameters = dict(parameters)
    constraints = fit_parameters.get("monotone_constraints")
    has_constraints = isinstance(constraints, (list, tuple)) and any(
        int(value or 0) != 0 for value in constraints
    )
    if not has_constraints:
        fit_parameters.pop("monotone_constraints", None)
        fit_parameters.pop("monotone_constraints_method", None)
    return fit_parameters


def gbm_parameter_warnings(parameters: dict[str, Any]) -> list[str]:
    """Return the stable parameter warnings emitted by Lucidum for this workflow."""

    warnings: list[str] = []
    strategy = str(parameters.get("data_sample_strategy") or "bagging").strip().lower()
    if strategy == "bagging":
        bagging_freq = int(parameters.get("bagging_freq", 0) or 0)
        bagging_fraction = float(parameters.get("bagging_fraction", 1.0) or 1.0)
        if bagging_freq <= 0 or bagging_fraction >= 1.0:
            warnings.append(
                "data_sample_strategy=bagging is only effective when "
                "bagging_freq > 0 and bagging_fraction < 1"
            )
    return warnings


def _infer_source_kind(duckdb_type: str) -> str:
    value = duckdb_type.upper()
    if "INT" in value:
        return "integer"
    if any(part in value for part in ("DOUBLE", "FLOAT", "REAL", "DECIMAL")):
        return "numeric"
    if "TIMESTAMP" in value:
        return "datetime"
    if "DATE" in value or "TIME" in value:
        return "date"
    return "categorical"


def _coerce_gbm_parameter(value: Any) -> Any:
    if isinstance(value, bool) or value is None or isinstance(value, (int, float)):
        return value
    text = str(value).strip()
    if not text:
        return ""
    if text.lower() in {"true", "false"}:
        return text.lower() == "true"
    try:
        return int(text) if "." not in text and "e" not in text.lower() else float(text)
    except ValueError:
        return text


def strip_formula_comments(formula: str) -> str:
    """Remove ``#`` comments while retaining hashes inside quoted strings."""

    lines = []
    for line in str(formula or "").splitlines():
        quote = None
        escaped = False
        kept = []
        for char in line:
            if escaped:
                kept.append(char)
                escaped = False
                continue
            if char == "\\" and quote in {"'", '"'}:
                kept.append(char)
                escaped = True
                continue
            if char in {"'", '"', "`"}:
                if quote == char:
                    quote = None
                elif quote is None:
                    quote = char
                kept.append(char)
                continue
            if char == "#" and quote is None:
                break
            kept.append(char)
        lines.append("".join(kept).rstrip())
    return "\n".join(lines).strip()


def formulaic_context() -> dict[str, Any]:
    """Functions supported by Lucidum-style Formulaic expressions."""

    from formulaic import transforms

    def ifelse(condition: Any, yes: Any, no: Any) -> np.ndarray:
        return np.asarray(np.where(condition, yes, no), dtype=float)

    def pmin(*values: Any) -> np.ndarray:
        return np.asarray(
            np.minimum.reduce(np.broadcast_arrays(*(np.asarray(value) for value in values))),
            dtype=float,
        )

    def pmax(*values: Any) -> np.ndarray:
        return np.asarray(
            np.maximum.reduce(np.broadcast_arrays(*(np.asarray(value) for value in values))),
            dtype=float,
        )

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


def prepare_feature_data(
    data: pd.DataFrame,
    spec_path: Path,
    scenario: str,
    *,
    eligible_rows: Any | None = None,
    column_kinds: dict[str, str] | None = None,
) -> tuple[pd.DataFrame, list[str], list[str]]:
    """Select a scenario and encode categories found on denominator-eligible rows."""

    feature_spec = pd.read_csv(spec_path, dtype="string")
    require_columns(feature_spec, ["Feature", scenario])
    selected = feature_spec[scenario].fillna("").str.strip().str.lower().eq("feature")

    feature_names = []
    for value in feature_spec.loc[selected, "Feature"]:
        name = "" if pd.isna(value) else str(value).strip()
        if name and name not in feature_names:
            feature_names.append(name)

    feature_names.sort(key=lambda name: (name.casefold(), name))

    if not feature_names:
        raise ValueError(f"Feature scenario selects no usable features: {scenario}")
    require_columns(data, feature_names)

    feature_data = data[feature_names].copy()
    category_rows = (
        pd.Series(True, index=feature_data.index)
        if eligible_rows is None
        else pd.Series(eligible_rows, index=feature_data.index).fillna(False).astype(bool)
    )
    categorical_features = []
    for name in feature_names:
        kind = (column_kinds or {}).get(name)
        if kind in {"date", "datetime"}:
            raise ValueError(f"{name} cannot be used as a LightGBM feature")
        if kind in {"integer", "numeric"} or (
            kind is None and pd.api.types.is_numeric_dtype(feature_data[name])
        ):
            feature_data[name] = pd.to_numeric(
                feature_data[name], errors="coerce"
            ).astype("float64")
        else:
            values = feature_data[name].astype("string")
            if pd.api.types.is_bool_dtype(feature_data[name]):
                values = values.str.lower()
            categories = sorted(str(value) for value in values.loc[category_rows].dropna().unique())
            feature_data[name] = pd.Categorical(
                values.where(values.isin(categories)),
                categories=categories,
            )
            categorical_features.append(name)

    return feature_data, feature_names, categorical_features


def feature_monotonicity_constraints(
    spec_path: Path,
    feature_names: list[str],
    column_kinds: dict[str, str],
    *,
    enabled: bool,
    objective: str,
) -> list[int]:
    """Return Feature Specification monotonicities in fitted feature order."""

    if not enabled:
        return [0] * len(feature_names)
    feature_spec = pd.read_csv(spec_path, dtype="string")
    if "Monotonicity" not in feature_spec.columns:
        return [0] * len(feature_names)

    values_by_feature: dict[str, Any] = {}
    for row in feature_spec[["Feature", "Monotonicity"]].to_dict("records"):
        name = "" if pd.isna(row.get("Feature")) else str(row.get("Feature")).strip()
        if name:
            values_by_feature[name] = row.get("Monotonicity")

    constraints: list[int] = []
    for name in feature_names:
        raw = values_by_feature.get(name)
        text = "" if pd.isna(raw) else str(raw or "").strip().lower()
        if text == "":
            value = 0
        elif text in {"increasing", "1"}:
            value = 1
        elif text in {"decreasing", "-1"}:
            value = -1
        else:
            raise ValueError(
                f"Feature Specification Monotonicity for {name} must be "
                "Increasing, 1, Decreasing, -1, or blank"
            )
        if value and column_kinds.get(name) not in {"integer", "numeric"}:
            raise ValueError(f"{name} must be numeric to use monotonicity")
        constraints.append(value)

    selected_objective = str(objective or "").strip().lower()
    if any(constraints) and selected_objective not in MONOTONE_OBJECTIVES:
        raise ValueError(f"Monotonicity is not supported for objective {selected_objective}")
    return constraints


def require_columns(data: pd.DataFrame, names: list[str]) -> None:
    """Raise one readable error for missing configured columns."""

    missing = [name for name in names if name and name not in data.columns]
    if missing:
        raise ValueError(f"Missing columns: {', '.join(missing)}")


def evaluate_validation_metric(
    *,
    actual: Any,
    prediction: Any,
    parameters: dict[str, Any],
    evaluation: dict[str, dict[str, list[float | None]]],
    best_iteration: int,
) -> str | None:
    """Calculate one Validation metric from predictions already made.

    LightGBM evaluates its own metric definition, but it does not score the
    feature data again.  This keeps the result exact for whichever LightGBM
    metric is configured without adding Validation to fitting or early
    stopping.
    """

    metric = str(parameters.get("metric") or "").strip().lower()
    metric_model = None
    try:
        import lightgbm as lgb

        actual_values = np.asarray(actual, dtype="float64")
        prediction_values = np.asarray(prediction, dtype="float64")
        if actual_values.shape != prediction_values.shape or actual_values.size == 0:
            raise ValueError("Validation actuals and predictions must have the same non-zero length")
        if not bool(np.isfinite(actual_values).all() and np.isfinite(prediction_values).all()):
            raise ValueError("Validation actuals and predictions must be finite")

        raw_prediction = _prediction_to_raw_score(
            prediction_values,
            str(parameters.get("objective") or "regression").strip().lower(),
            parameters,
        )
        metric_data = lgb.Dataset(
            np.arange(actual_values.size, dtype="float64").reshape(-1, 1),
            label=actual_values,
            init_score=raw_prediction,
            free_raw_data=False,
            params={"min_data_in_leaf": 1, "min_data_in_bin": 1, "verbosity": -1},
        )
        metric_model = lgb.Booster(
            params=_metric_parameters(parameters),
            train_set=metric_data,
        )
        results = metric_model.eval_train()
        selected = next(
            (
                result
                for result in results
                if str(result[1] if len(result) > 1 else "").strip().lower()
                == metric
            ),
            None,
        )
        if selected is None:
            raise ValueError(f"LightGBM returned no {metric} result")

        value = float(selected[2])
        if not np.isfinite(value):
            raise ValueError(f"LightGBM returned a non-finite {metric} result")

        values: list[float | None] = [None] * max(0, int(best_iteration) - 1)
        values.append(value)
        evaluation.setdefault("validation", {})[str(selected[1])] = values
        return None
    except Exception as exc:
        return f"Validation {metric} metric could not be calculated: {exc}"
    finally:
        free_dataset = getattr(metric_model, "free_dataset", None)
        if callable(free_dataset):
            free_dataset()


def _prediction_to_raw_score(
    prediction: np.ndarray,
    objective: str,
    parameters: dict[str, Any],
) -> np.ndarray:
    """Undo LightGBM's objective transform for metric-only evaluation."""

    if objective in {"poisson", "gamma", "tweedie"}:
        if not bool((prediction > 0).all()):
            raise ValueError(f"{objective} predictions must be positive")
        return np.log(prediction)
    if objective in {"binary", "cross_entropy"}:
        if not bool(((prediction > 0) & (prediction < 1)).all()):
            raise ValueError(f"{objective} predictions must be between zero and one")
        sigmoid = float(parameters.get("sigmoid", 1.0) or 1.0)
        return np.log(prediction / (1.0 - prediction)) / sigmoid
    if objective == "cross_entropy_lambda":
        if not bool((prediction > 0).all()):
            raise ValueError("cross_entropy_lambda predictions must be positive")
        raw = np.empty_like(prediction)
        large = prediction > 20
        raw[large] = prediction[large] + np.log1p(-np.exp(-prediction[large]))
        raw[~large] = np.log(np.expm1(prediction[~large]))
        return raw
    return prediction


def _metric_parameters(parameters: dict[str, Any]) -> dict[str, Any]:
    """Keep only settings which change LightGBM's metric calculation."""

    names = {
        "objective",
        "metric",
        "sigmoid",
        "alpha",
        "fair_c",
        "tweedie_variance_power",
    }
    return {
        **{name: value for name, value in parameters.items() if name in names},
        "boost_from_average": False,
        "min_data_in_leaf": 1,
        "min_data_in_bin": 1,
        "verbosity": -1,
    }


def _running_as_script(script_path: Path) -> bool:
    import sys

    try:
        return Path(sys.argv[0]).resolve() == script_path
    except (IndexError, OSError):
        return False


def _expect_keys(
    mapping: Any,
    expected: set[str],
    label: str,
    *,
    optional: set[str] | None = None,
) -> None:
    if not isinstance(mapping, dict):
        raise ValueError(f"{label} must be a YAML mapping")
    allowed = expected | set(optional or set())
    missing = sorted(expected - set(mapping))
    unknown = sorted(set(mapping) - allowed)
    if unknown:
        raise ValueError(f"{label} has unknown keys: {', '.join(unknown)}")
    if missing:
        raise ValueError(f"{label} is missing keys: {', '.join(missing)}")
