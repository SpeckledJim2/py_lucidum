from __future__ import annotations

import csv
from pathlib import Path


def resolve_filters_path(filters_path: str | Path | None) -> Path:
    return Path(filters_path).expanduser().resolve() if filters_path else (Path.cwd() / "filter_spec.csv").resolve()


def load_saved_filters(filters_path: str | Path | None) -> list[dict[str, str]]:
    path = resolve_filters_path(filters_path)
    if not path.exists():
        if filters_path:
            raise FileNotFoundError(f"Filter specification file does not exist: {path}")
        return []
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != ["name", "expression"]:
            raise ValueError("filter_spec.csv must have exactly these columns: name,expression")
        filters: list[dict[str, str]] = []
        for row in reader:
            name = str(row.get("name") or "").strip()
            expression = str(row.get("expression") or "").strip()
            if name and expression:
                filters.append({"name": name, "expression": expression})
        return filters
