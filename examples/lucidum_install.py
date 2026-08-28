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
    if kind == "glm":
        validate_glm_model_folder(source, chosen_id)

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
        if kind == "glm":
            validate_glm_model_folder(staging, chosen_id)
        replace_directory(staging, target, replace_existing=replace_existing)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    write_json(
        parent / "active_model.json",
        {"model_id": chosen_id, "activated_at": utc_now()},
    )
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
