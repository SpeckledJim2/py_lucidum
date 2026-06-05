from __future__ import annotations

import csv
from pathlib import Path
from typing import Any


FEATURE_SPEC_REQUIRED_COLUMNS = ["Feature", "Grouping"]
FEATURE_SPEC_BASE_COLUMN = "Base"
FEATURE_SPEC_METADATA_COLUMNS = {
    "Base": "base",
    "min": "min",
    "max": "max",
    "banding": "banding",
}


def resolve_features_path(features_path: str | Path | None, use_features: bool = True) -> Path | None:
    if not use_features:
        return None
    if features_path:
        return Path(features_path).expanduser().resolve()
    root_spec = (Path.cwd() / "feature_spec.csv").resolve()
    if root_spec.exists():
        return root_spec
    return (Path.cwd() / "specs" / "feature_spec.csv").resolve()


def load_features(features_path: str | Path | None, use_features: bool = True) -> dict[str, Any]:
    path = resolve_features_path(features_path, use_features=use_features)
    if path is None:
        return empty_feature_spec()
    if not path.exists():
        if features_path:
            raise FileNotFoundError(f"Feature specification file does not exist: {path}")
        return empty_feature_spec()
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        fieldnames = list(reader.fieldnames or [])
        if fieldnames[:2] != FEATURE_SPEC_REQUIRED_COLUMNS:
            raise ValueError("feature_spec.csv must start with these columns: Feature,Grouping")
        metadata_columns: list[tuple[str, str]] = []
        scenario_start = 2
        seen_metadata: set[str] = set()
        while scenario_start < len(fieldnames):
            raw_name = fieldnames[scenario_start]
            key = FEATURE_SPEC_METADATA_COLUMNS.get(raw_name)
            if key is None or raw_name in seen_metadata:
                break
            metadata_columns.append((raw_name, key))
            seen_metadata.add(raw_name)
            scenario_start += 1
        scenario_names = fieldnames[scenario_start:]
        rows: list[dict[str, Any]] = []
        scenario_features: dict[str, list[str]] = {name: [] for name in scenario_names}
        for row in reader:
            feature = str(row.get("Feature") or "").strip()
            if not feature:
                continue
            grouping = str(row.get("Grouping") or "").strip()
            scenarios: dict[str, bool] = {}
            for scenario_name in scenario_names:
                selected = "feature" in str(row.get(scenario_name) or "").strip().lower()
                scenarios[scenario_name] = selected
                if selected:
                    scenario_features[scenario_name].append(feature)
            row_payload = {"feature": feature, "grouping": grouping, "scenarios": scenarios}
            for column_name, payload_key in metadata_columns:
                row_payload[payload_key] = str(row.get(column_name) or "").strip()
            rows.append(row_payload)
        return {
            "rows": rows,
            "scenarios": [
                {"name": scenario_name, "features": scenario_features[scenario_name]}
                for scenario_name in scenario_names
            ],
        }


def empty_feature_spec() -> dict[str, list[Any]]:
    return {"rows": [], "scenarios": []}
