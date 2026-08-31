"""Optionally copy and activate standalone model results inside Lucidum.

The 01/02/03 workflows do not depend on this module. It exists only for the
explicit installation step that synchronizes one saved model folder into the
dataset-version sidecar used by the application.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import shutil
import time
from datetime import datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

import duckdb


MODEL_ID_RE = re.compile(r"[A-Za-z0-9_.-]+")
WORKSPACE_VERSION = 1
REQUIRED_GLM_ARTIFACTS = {
    "manifest.json",
    "formula.txt",
    "estimator.pkl",
    "coefficients.parquet",
    "feature_importance.parquet",
    "predictions.parquet",
    "diagnostics.json",
}
REQUIRED_GLM_MANIFEST_FIELDS = {
    "model_id",
    "label",
    "created_at",
    "response_column",
    "denominator_column",
    "family",
    "training_scope",
}
REQUIRED_GLM_DIAGNOSTIC_COUNTS = {
    "n_terms",
    "n_features",
    "n_interactions",
    "training_rows",
}
REQUIRED_GLM_DIAGNOSTIC_METRICS = {
    "deviance",
    "aic",
    "bic",
    "gini_tr",
    "gini_te",
    "gini_vl",
}
REQUIRED_GBM_ARTIFACTS = {
    "manifest.json",
    "parameters.json",
    "features.json",
    "feature_config.parquet",
    "model.txt",
    "predictions.parquet",
    "evaluation.parquet",
    "tree_table.parquet",
}
REQUIRED_GBM_SHAP_ARTIFACTS = {"shap_values.parquet", "shap_summary.parquet"}
REQUIRED_GBM_MANIFEST_FIELDS = {
    "model_id",
    "label",
    "created_at",
    "training_mode",
    "response_column",
    "offset_column",
    "best_iteration",
    "training_rows",
    "test_rows",
    "validation_rows",
    "scored_rows",
    "sample_column",
    "sample_source",
    "shap_rows",
    "gini_tr",
    "gini_te",
    "gini_vl",
    "timings",
    "warnings",
    "feature_scenario",
    "feature_interaction_group_models",
    "init_score",
}
REQUIRED_GBM_PARAMETER_FIELDS = {
    "objective",
    "metric",
    "tweedie_variance_power",
    "data_sample_strategy",
    "monotone_constraints_method",
    "num_iterations",
    "learning_rate",
    "num_leaves",
    "max_depth",
    "min_data_in_leaf",
    "early_stopping_rounds",
    "feature_fraction",
    "bagging_fraction",
    "bagging_freq",
    "lambda_l1",
    "lambda_l2",
    "min_gain_to_split",
    "max_bin",
    "num_threads",
    "verbosity",
    "seed",
}


def install_model_in_lucidum(
    *,
    dataset_path: str | Path,
    model_folder: str | Path,
    model_type: str,
    model_id: str | None = None,
    replace_existing: bool = False,
) -> Path:
    """Copy one exact saved-model folder into Lucidum and activate it."""

    dataset = Path(dataset_path).expanduser().resolve()
    source = Path(model_folder).expanduser().resolve()
    kind = str(model_type or "").strip().lower()
    if kind not in {"glm", "gbm"}:
        raise ValueError("model_type must be 'glm' or 'gbm'")
    chosen_id = validate_model_id(model_id or source.name)
    if source.name != chosen_id:
        raise ValueError(f"model_folder must be the folder for model {chosen_id!r}")
    if not source.is_dir():
        raise ValueError(f"Model results folder does not exist: {source}")
    if not (source / "manifest.json").is_file():
        raise ValueError(f"Model results folder has no manifest.json: {source}")
    validator = validate_glm_model_folder if kind == "glm" else validate_gbm_model_folder
    validator(source, chosen_id)

    metadata = workspace_metadata(dataset)
    parent = (
        dataset.parent
        / ".lucidum"
        / "datasets"
        / str(metadata["slug"])
        / str(metadata["signature"])
        / "models"
        / kind
    )
    target = parent / chosen_id
    parent.mkdir(parents=True, exist_ok=True)
    staging = parent / f".{chosen_id}.tmp-{uuid4().hex}"
    shutil.copytree(source, staging)
    try:
        validator(staging, chosen_id)
        replace_model_and_activate(
            staging,
            target,
            active_path=parent / "active_model.json",
            active_payload={"model_id": chosen_id, "activated_at": utc_now()},
            replace_existing=replace_existing,
        )
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return target


def validate_glm_model_folder(path: Path, model_id: str) -> None:
    """Reject a GLM folder that cannot populate the Model navigator."""

    errors: list[str] = []
    missing_artifacts = sorted(
        name for name in REQUIRED_GLM_ARTIFACTS if not (path / name).is_file()
    )
    if missing_artifacts:
        errors.append("missing artifacts: " + ", ".join(missing_artifacts))

    manifest = read_json_mapping(path / "manifest.json", "manifest", errors)
    diagnostics = read_json_mapping(path / "diagnostics.json", "diagnostics", errors)

    if manifest is not None:
        missing_fields = sorted(REQUIRED_GLM_MANIFEST_FIELDS - manifest.keys())
        if missing_fields:
            errors.append("missing manifest fields: " + ", ".join(missing_fields))
        if str(manifest.get("model_id") or "").strip() != model_id:
            errors.append(f"manifest model_id must be {model_id!r}")
        for field in (
            "model_id",
            "label",
            "created_at",
            "response_column",
            "family",
            "training_scope",
        ):
            if field in manifest and not str(manifest.get(field) or "").strip():
                errors.append(f"manifest field {field} must not be blank")
        training_scope = str(manifest.get("training_scope") or "").strip()
        if "training_scope" in manifest and training_scope not in {
            "all",
            "training",
            "training_test",
        }:
            errors.append("manifest field training_scope must be all, training, or training_test")
        timings = manifest.get("timings")
        if not isinstance(timings, dict):
            errors.append("manifest field timings must be an object")
        else:
            for field in ("fit_ms", "elapsed_ms"):
                if field not in timings:
                    errors.append(f"missing manifest field: timings.{field}")
                elif not finite_non_negative(timings[field]):
                    errors.append(f"manifest field timings.{field} must be finite and non-negative")
            if (
                finite_non_negative(timings.get("fit_ms"))
                and finite_non_negative(timings.get("elapsed_ms"))
                and float(timings["elapsed_ms"]) < float(timings["fit_ms"])
            ):
                errors.append("manifest field timings.elapsed_ms must be at least timings.fit_ms")

    if diagnostics is not None:
        missing_counts = sorted(REQUIRED_GLM_DIAGNOSTIC_COUNTS - diagnostics.keys())
        missing_metrics = sorted(REQUIRED_GLM_DIAGNOSTIC_METRICS - diagnostics.keys())
        if missing_counts or missing_metrics:
            errors.append(
                "missing diagnostics fields: "
                + ", ".join([*missing_counts, *missing_metrics])
            )
        for field in sorted(REQUIRED_GLM_DIAGNOSTIC_COUNTS & diagnostics.keys()):
            if not integer_non_negative(diagnostics[field]):
                errors.append(f"diagnostics field {field} must be a non-negative integer")
        for field in sorted(REQUIRED_GLM_DIAGNOSTIC_METRICS & diagnostics.keys()):
            value = diagnostics[field]
            if value is not None and not finite_number(value):
                errors.append(f"diagnostics field {field} must be null or finite")

    if errors:
        raise ValueError("GLM model folder is incomplete: " + "; ".join(errors))


def validate_gbm_model_folder(path: Path, model_id: str) -> None:
    """Reject a GBM folder that cannot safely populate the Model navigator."""

    errors: list[str] = []
    missing_artifacts = sorted(
        name for name in REQUIRED_GBM_ARTIFACTS if not (path / name).is_file()
    )
    if missing_artifacts:
        errors.append("missing artifacts: " + ", ".join(missing_artifacts))

    manifest = read_json_mapping(path / "manifest.json", "manifest", errors)
    parameters = read_json_mapping(path / "parameters.json", "parameters", errors)
    features = read_json_list(path / "features.json", "features", errors)

    if manifest is not None:
        missing_fields = sorted(REQUIRED_GBM_MANIFEST_FIELDS - manifest.keys())
        if missing_fields:
            errors.append("missing manifest fields: " + ", ".join(missing_fields))
        if str(manifest.get("model_id") or "").strip() != model_id:
            errors.append(f"manifest model_id must be {model_id!r}")
        for field in (
            "model_id",
            "label",
            "created_at",
            "training_mode",
            "response_column",
            "sample_source",
        ):
            if field in manifest and not isinstance(manifest[field], str):
                errors.append(f"manifest field {field} must be a string")
            elif field in manifest and not manifest[field].strip():
                errors.append(f"manifest field {field} must not be blank")
        if isinstance(manifest.get("created_at"), str) and not valid_timestamp(
            manifest["created_at"]
        ):
            errors.append("manifest field created_at must be an ISO-8601 timestamp")
        if manifest.get("training_mode") not in {"normal", "ebm"}:
            errors.append("manifest field training_mode must be normal or ebm")
        for field in ("offset_column", "sample_column"):
            value = manifest.get(field)
            if value is not None and (not isinstance(value, str) or not value.strip()):
                errors.append(f"manifest field {field} must be null or a non-blank string")
        for field in (
            "best_iteration",
            "training_rows",
            "test_rows",
            "validation_rows",
            "scored_rows",
            "shap_rows",
        ):
            if field in manifest and not integer_non_negative(manifest[field]):
                errors.append(f"manifest field {field} must be a non-negative integer")
        if integer_non_negative(manifest.get("best_iteration")) and int(
            manifest["best_iteration"]
        ) < 1:
            errors.append("manifest field best_iteration must be at least 1")
        if integer_non_negative(manifest.get("training_rows")) and int(
            manifest["training_rows"]
        ) < 1:
            errors.append("manifest field training_rows must be at least 1")
        for field in ("gini_tr", "gini_te", "gini_vl"):
            value = manifest.get(field)
            if field in manifest and value is not None and not finite_number(value):
                errors.append(f"manifest field {field} must be null or finite")

        timings = manifest.get("timings")
        if not isinstance(timings, dict):
            errors.append("manifest field timings must be an object")
        elif "training_seconds" not in timings:
            errors.append("missing manifest field: timings.training_seconds")
        elif not finite_non_negative(timings["training_seconds"]):
            errors.append(
                "manifest field timings.training_seconds must be finite and non-negative"
            )

        warnings = manifest.get("warnings")
        if not isinstance(warnings, list) or not all(
            isinstance(warning, str) for warning in warnings
        ):
            errors.append("manifest field warnings must be an array of strings")

        scenario = manifest.get("feature_scenario")
        if not isinstance(scenario, dict):
            errors.append("manifest field feature_scenario must be an object")
        else:
            if not isinstance(scenario.get("name"), str) or not scenario["name"].strip():
                errors.append("manifest field feature_scenario.name must not be blank")
            scenario_features = scenario.get("features")
            if not valid_feature_list(scenario_features):
                errors.append(
                    "manifest field feature_scenario.features must be a non-empty "
                    "array of unique strings"
                )
            elif features is not None and scenario_features != features:
                errors.append("manifest feature_scenario.features must match features.json")
        group_models = manifest.get("feature_interaction_group_models")
        if not isinstance(group_models, dict):
            errors.append("manifest field feature_interaction_group_models must be an object")
        elif not {"enabled", "error_metric", "groups"}.issubset(group_models):
            errors.append(
                "manifest field feature_interaction_group_models is missing required fields"
            )
        init_score = manifest.get("init_score")
        if not isinstance(init_score, dict):
            errors.append("manifest field init_score must be an object")
        elif not {"value", "kind", "transform"}.issubset(init_score):
            errors.append("manifest field init_score is missing required fields")

        shap_rows = manifest.get("shap_rows")
        if integer_non_negative(shap_rows) and int(shap_rows) > 0:
            missing_shap = sorted(
                name for name in REQUIRED_GBM_SHAP_ARTIFACTS if not (path / name).is_file()
            )
            if missing_shap:
                errors.append("missing SHAP artifacts: " + ", ".join(missing_shap))

    if parameters is not None:
        missing_parameters = sorted(REQUIRED_GBM_PARAMETER_FIELDS - parameters.keys())
        if missing_parameters:
            errors.append("missing parameters: " + ", ".join(missing_parameters))
        for field in (
            "objective",
            "metric",
            "data_sample_strategy",
            "monotone_constraints_method",
        ):
            if field in parameters and (
                not isinstance(parameters[field], str) or not parameters[field].strip()
            ):
                errors.append(f"parameter {field} must be a non-blank string")
        integer_parameters = {
            "num_iterations",
            "num_leaves",
            "max_depth",
            "min_data_in_leaf",
            "early_stopping_rounds",
            "bagging_freq",
            "max_bin",
            "num_threads",
            "verbosity",
            "seed",
        }
        numeric_parameters = REQUIRED_GBM_PARAMETER_FIELDS - integer_parameters - {
            "objective",
            "metric",
            "data_sample_strategy",
            "monotone_constraints_method",
        }
        for field in sorted(integer_parameters & parameters.keys()):
            if not finite_number(parameters[field]) or not float(parameters[field]).is_integer():
                errors.append(f"parameter {field} must be an integer")
        for field in sorted(numeric_parameters & parameters.keys()):
            if not finite_number(parameters[field]):
                errors.append(f"parameter {field} must be finite")
        if (
            "num_iterations" in parameters
            and finite_number(parameters["num_iterations"])
            and int(parameters["num_iterations"]) < 1
        ):
            errors.append("parameter num_iterations must be at least 1")
        if (
            "num_leaves" in parameters
            and finite_number(parameters["num_leaves"])
            and int(parameters["num_leaves"]) < 2
        ):
            errors.append("parameter num_leaves must be at least 2")
        if (
            "learning_rate" in parameters
            and finite_number(parameters["learning_rate"])
            and float(parameters["learning_rate"]) <= 0
        ):
            errors.append("parameter learning_rate must be greater than 0")
        for field in ("min_data_in_leaf", "early_stopping_rounds", "bagging_freq", "num_threads"):
            if field in parameters and finite_number(parameters[field]) and int(parameters[field]) < 0:
                errors.append(f"parameter {field} must be non-negative")
        monotone_method = parameters.get("monotone_constraints_method")
        if monotone_method not in {"basic", "intermediate", "advanced"}:
            errors.append(
                "parameter monotone_constraints_method must be basic, intermediate, or advanced"
            )

    if features is not None and not valid_feature_list(features):
        errors.append("features.json must contain a non-empty array of unique strings")
    if parameters is not None and features is not None and "monotone_constraints" in parameters:
        constraints = parameters["monotone_constraints"]
        if (
            not isinstance(constraints, list)
            or len(constraints) != len(features)
            or any(value not in {-1, 0, 1} for value in constraints)
        ):
            errors.append(
                "parameter monotone_constraints must contain one -1, 0, or 1 per fitted feature"
            )

    if errors:
        raise ValueError("GBM model folder is incomplete: " + "; ".join(errors))


def read_json_mapping(path: Path, label: str, errors: list[str]) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        errors.append(f"{label}.json is unreadable: {exc}")
        return None
    if not isinstance(payload, dict):
        errors.append(f"{label}.json must contain an object")
        return None
    return payload


def read_json_list(path: Path, label: str, errors: list[str]) -> list[Any] | None:
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        errors.append(f"{label}.json is unreadable: {exc}")
        return None
    if not isinstance(payload, list):
        errors.append(f"{label}.json must contain an array")
        return None
    return payload


def valid_feature_list(value: Any) -> bool:
    return (
        isinstance(value, list)
        and bool(value)
        and all(isinstance(item, str) and bool(item.strip()) for item in value)
        and len(set(value)) == len(value)
    )


def valid_timestamp(value: str) -> bool:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return False
    return parsed.tzinfo is not None


def finite_number(value: Any) -> bool:
    if isinstance(value, bool):
        return False
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def finite_non_negative(value: Any) -> bool:
    return finite_number(value) and float(value) >= 0


def integer_non_negative(value: Any) -> bool:
    return finite_non_negative(value) and float(value).is_integer()


def workspace_metadata(path: Path) -> dict[str, Any]:
    """Reproduce Lucidum workspace-signature version 1 for one source file."""

    relation = dataset_relation(path)
    con = duckdb.connect(database=":memory:")
    try:
        describe = con.execute(f"DESCRIBE SELECT * FROM {relation}").fetchall()
        if any(str(row[0]) == "__lucidum_row_id" for row in describe):
            raise ValueError(
                "The source dataset already contains the reserved __lucidum_row_id column"
            )
        row_count = int(con.execute(f"SELECT COUNT(*) FROM {relation}").fetchone()[0])
    finally:
        con.close()
    stat = path.stat()
    schema = [{"name": str(row[0]), "duckdb_type": str(row[1])} for row in describe]
    schema_fingerprint = sha256_json(schema)
    signature = sha256_json(
        {
            "version": WORKSPACE_VERSION,
            "file_size": int(stat.st_size),
            "mtime_ns": int(stat.st_mtime_ns),
            "row_count": row_count,
            "schema_fingerprint": schema_fingerprint,
        }
    )[:20]
    return {
        "version": WORKSPACE_VERSION,
        "path": str(path),
        "name": path.name,
        "slug": dataset_slug(path),
        "signature": signature,
        "file_size": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
        "row_count": row_count,
        "schema_fingerprint": schema_fingerprint,
    }


def replace_directory(staging: Path, target: Path, *, replace_existing: bool) -> None:
    """Atomically replace only the validated model folder, with rollback."""

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


def replace_model_and_activate(
    staging: Path,
    target: Path,
    *,
    active_path: Path,
    active_payload: dict[str, Any],
    replace_existing: bool,
) -> None:
    """Atomically publish a validated folder and its active-model pointer."""

    parent = target.parent.resolve()
    if staging.parent.resolve() != parent or active_path.parent.resolve() != parent:
        raise ValueError("Refusing to replace a model outside its validated parent")
    active_staging = target.parent / f".active-model.tmp-{uuid4().hex}.json"
    model_backup: Path | None = None
    active_backup: Path | None = None
    model_published = False
    active_published = False
    write_json(active_staging, active_payload)
    try:
        if target.exists():
            if not replace_existing:
                raise FileExistsError(f"Model already exists: {target}")
            model_backup = target.parent / f".{target.name}.backup-{uuid4().hex}"
            target.rename(model_backup)
        staging.rename(target)
        model_published = True
        if active_path.exists():
            active_backup = target.parent / f".active-model.backup-{uuid4().hex}.json"
            active_path.rename(active_backup)
        active_staging.rename(active_path)
        active_published = True
    except Exception:
        if active_published and active_path.exists():
            active_path.unlink()
        if active_backup is not None and active_backup.exists() and not active_path.exists():
            active_backup.rename(active_path)
        if model_published and target.exists():
            shutil.rmtree(target)
        if model_backup is not None and model_backup.exists() and not target.exists():
            model_backup.rename(target)
        raise
    finally:
        active_staging.unlink(missing_ok=True)
    if model_backup is not None:
        shutil.rmtree(model_backup, ignore_errors=True)
    if active_backup is not None:
        active_backup.unlink(missing_ok=True)


def validate_model_id(value: Any) -> str:
    model_id = str(value or "").strip()
    if model_id in {"", ".", ".."} or not MODEL_ID_RE.fullmatch(model_id):
        raise ValueError(
            "model.id must contain only letters, numbers, dots, underscores, and hyphens"
        )
    return model_id


def dataset_relation(path: Path) -> str:
    literal = sql_literal(str(path))
    if path.suffix.lower() == ".parquet":
        return f"read_parquet({literal})"
    if path.suffix.lower() == ".csv":
        return f"read_csv_auto({literal}, header=true, ignore_errors=true)"
    raise ValueError("The examples support one CSV or Parquet file")


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp-{uuid4().hex}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(path)


def dataset_slug(path: Path) -> str:
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "-", path.name).strip(".-")
    return slug or "dataset"


def sha256_json(payload: Any) -> str:
    text = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sql_literal(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
