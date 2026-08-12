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

import numpy as np
import pandas as pd
import yaml


MODEL_ID_RE = re.compile(r"[A-Za-z0-9_.-]+")

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
        "output": {"portable_root", "install", "replace_existing"},
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
            "holdout_value",
        },
        "features": {"spec_path", "scenario_column"},
        "model": {"id", "label"},
        "training": {
            "num_boost_round",
            "early_stopping_rounds",
            "shap_rows",
            "parameters",
        },
        "output": {"portable_root", "install", "replace_existing"},
    },
}


def config_path_from_command_line(script_file: str, default_name: str) -> Path:
    """Return the optional YAML argument, or the config beside the script.

    When a section is run interactively rather than as a command, use the
    default file and ignore the notebook or IDE process's unrelated arguments.
    """

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
        _expect_keys(value, schema[section], section)

    model_id = str(config["model"]["id"] or "").strip()
    if model_id in {"", ".", ".."} or not MODEL_ID_RE.fullmatch(model_id):
        raise ValueError(
            "model.id must contain only letters, numbers, dots, underscores, and hyphens"
        )

    if model_type == "gbm":
        sample_values = {
            str(config["dataset"][name]).strip().lower()
            for name in ("training_value", "early_stopping_value", "holdout_value")
        }
        if len(sample_values) != 3:
            raise ValueError("Training, test, and holdout sample values must be distinct")

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
) -> tuple[pd.DataFrame, list[str], list[str]]:
    """Select a feature scenario and prepare LightGBM categorical columns."""

    feature_spec = pd.read_csv(spec_path, dtype="string")
    require_columns(feature_spec, ["Feature", scenario])
    selected = feature_spec[scenario].fillna("").str.strip().str.lower().eq("feature")

    feature_names = []
    for value in feature_spec.loc[selected, "Feature"]:
        name = "" if pd.isna(value) else str(value).strip()
        if name and name not in feature_names:
            feature_names.append(name)

    if not feature_names:
        raise ValueError(f"Feature scenario selects no usable features: {scenario}")
    require_columns(data, feature_names)

    feature_data = data[feature_names].copy()
    categorical_features = []
    for name in feature_names:
        if pd.api.types.is_numeric_dtype(feature_data[name]):
            feature_data[name] = pd.to_numeric(feature_data[name], errors="coerce")
        else:
            categories = sorted(str(value) for value in feature_data[name].dropna().unique())
            feature_data[name] = pd.Categorical(
                feature_data[name].astype("string"),
                categories=categories,
            )
            categorical_features.append(name)

    return feature_data, feature_names, categorical_features


def require_columns(data: pd.DataFrame, names: list[str]) -> None:
    """Raise one readable error for missing configured columns."""

    missing = [name for name in names if name and name not in data.columns]
    if missing:
        raise ValueError(f"Missing columns: {', '.join(missing)}")


def _running_as_script(script_path: Path) -> bool:
    import sys

    try:
        return Path(sys.argv[0]).resolve() == script_path
    except (IndexError, OSError):
        return False


def _expect_keys(mapping: Any, expected: set[str], label: str) -> None:
    if not isinstance(mapping, dict):
        raise ValueError(f"{label} must be a YAML mapping")
    missing = sorted(expected - set(mapping))
    unknown = sorted(set(mapping) - expected)
    if unknown:
        raise ValueError(f"{label} has unknown keys: {', '.join(unknown)}")
    if missing:
        raise ValueError(f"{label} is missing keys: {', '.join(missing)}")
