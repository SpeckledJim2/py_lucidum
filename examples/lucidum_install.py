"""Optionally copy and activate standalone model results inside Lucidum.

The 01/02/03 workflows do not depend on this module. It exists only for the
explicit installation step that synchronizes one saved model folder into the
dataset-version sidecar used by the application.
"""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import time
from pathlib import Path
from typing import Any
from uuid import uuid4

import duckdb


MODEL_ID_RE = re.compile(r"[A-Za-z0-9_.-]+")
WORKSPACE_VERSION = 1


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
        replace_directory(staging, target, replace_existing=replace_existing)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    write_json(
        parent / "active_model.json",
        {"model_id": chosen_id, "activated_at": utc_now()},
    )
    return target


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
