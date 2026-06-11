from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any


DATASET_WORKSPACE_VERSION = 1


def dataset_slug(path: str | Path) -> str:
    name = Path(path).expanduser().resolve().name
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "-", name).strip(".-")
    return slug or "dataset"


def dataset_workspace_metadata(path: str | Path, dataset: Any = None) -> dict[str, Any]:
    resolved = Path(path).expanduser().resolve()
    stat = resolved.stat()
    row_count, schema_columns = _dataset_shape(resolved, dataset)
    schema_payload = [
        {"name": str(column["name"]), "duckdb_type": str(column["duckdb_type"])}
        for column in schema_columns
    ]
    schema_fingerprint = _fingerprint(schema_payload)
    signature_payload = {
        "version": DATASET_WORKSPACE_VERSION,
        "file_size": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
        "row_count": int(row_count),
        "schema_fingerprint": schema_fingerprint,
    }
    signature = _fingerprint(signature_payload)[:20]
    return {
        "version": DATASET_WORKSPACE_VERSION,
        "path": str(resolved),
        "name": resolved.name,
        "slug": dataset_slug(resolved),
        "signature": signature,
        "file_size": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
        "row_count": int(row_count),
        "schema_fingerprint": schema_fingerprint,
    }


def dataset_workspace_root(path: str | Path, dataset: Any = None) -> Path:
    resolved = Path(path).expanduser().resolve()
    metadata = dataset_workspace_metadata(resolved, dataset)
    return resolved.parent / ".lucidum" / "datasets" / metadata["slug"] / metadata["signature"]


def _dataset_shape(path: Path, dataset: Any = None) -> tuple[int, list[dict[str, str]]]:
    active_dataset = dataset
    if active_dataset is None:
        from .dataset import Dataset

        active_dataset = Dataset(path)
    row_count = int(active_dataset.row_count())
    columns = [
        {"name": str(column.name), "duckdb_type": str(column.duckdb_type)}
        for column in active_dataset._schema_columns()
    ]
    return row_count, columns


def _fingerprint(payload: Any) -> str:
    text = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


__all__ = [
    "DATASET_WORKSPACE_VERSION",
    "dataset_slug",
    "dataset_workspace_metadata",
    "dataset_workspace_root",
]
