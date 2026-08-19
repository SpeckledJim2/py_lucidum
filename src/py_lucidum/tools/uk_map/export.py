from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

from py_lucidum.core import Dataset, dataset_workspace_root

from .query import sector_smoothing_export_request
from .smoothing import MAX_SMOOTHING_LEVEL, write_sector_smoothing_parquet


SECTOR_SMOOTHING_OUTPUT_COLUMNS = (
    "postcode_sector",
    "numerator_sum",
    "denominator_sum",
    "unsmoothed",
    *(f"smooth_n{level}" for level in range(1, MAX_SMOOTHING_LEVEL + 1)),
    *(f"numerator_n{level}" for level in range(1, MAX_SMOOTHING_LEVEL + 1)),
    *(f"denominator_n{level}" for level in range(1, MAX_SMOOTHING_LEVEL + 1)),
)


def save_sector_smoothing_sidecar(
    dataset: Dataset,
    request: dict[str, Any],
    defaults: dict[str, str] | None = None,
) -> dict[str, Any]:
    spec = sector_smoothing_export_request(dataset, request, defaults)
    output_path = sector_smoothing_sidecar_path(dataset, spec)
    replaced = output_path.is_file()
    written_path, row_count = write_sector_smoothing_parquet(
        dataset.con,
        spec["raw_summary_sql"],
        output_path,
    )
    return {
        "path": str(written_path),
        "row_count": row_count,
        "columns": list(SECTOR_SMOOTHING_OUTPUT_COLUMNS),
        "replaced": replaced,
    }


def sector_smoothing_sidecar_path(dataset: Dataset, spec: dict[str, Any]) -> Path:
    identity = {
        "source": spec["source"],
        "numerator": spec["numerator"],
        "denominator_source": spec["denominator_source"],
        "denominator": spec["denominator"],
        "postcode_sector": spec["postcode_sector"],
        "filter": spec["filter"],
    }
    digest = hashlib.sha256(
        json.dumps(identity, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()[:12]
    numerator = _filename_part(spec["numerator"])
    denominator = _filename_part(spec["denominator"] or "rows")
    filename = f"{numerator}-per-{denominator}-{digest}.parquet"
    return (
        dataset_workspace_root(dataset.path, dataset)
        / "uk_map"
        / "sector_smoothing"
        / filename
    )


def _filename_part(raw: Any) -> str:
    text = re.sub(r"[^A-Za-z0-9_.-]+", "-", str(raw or "").strip()).strip(".-")
    return (text or "value")[:32]


__all__ = [
    "SECTOR_SMOOTHING_OUTPUT_COLUMNS",
    "save_sector_smoothing_sidecar",
    "sector_smoothing_sidecar_path",
]
