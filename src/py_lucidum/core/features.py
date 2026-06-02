from __future__ import annotations

import csv
from pathlib import Path
from typing import Any


FEATURE_SPEC_REQUIRED_COLUMNS = ["Feature", "Grouping"]
FEATURE_SPEC_BASE_COLUMN = "Base"


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
        has_base_column = len(fieldnames) > 2 and fieldnames[2] == FEATURE_SPEC_BASE_COLUMN
        scenario_names = fieldnames[3:] if has_base_column else fieldnames[2:]
        rows: list[dict[str, Any]] = []
        scenario_features: dict[str, list[str]] = {name: [] for name in scenario_names}
        for row in reader:
            feature = str(row.get("Feature") or "").strip()
            if not feature:
                continue
            grouping = str(row.get("Grouping") or "").strip()
            base = str(row.get(FEATURE_SPEC_BASE_COLUMN) or "").strip() if has_base_column else ""
            scenarios: dict[str, bool] = {}
            for scenario_name in scenario_names:
                selected = "feature" in str(row.get(scenario_name) or "").strip().lower()
                scenarios[scenario_name] = selected
                if selected:
                    scenario_features[scenario_name].append(feature)
            row_payload = {"feature": feature, "grouping": grouping, "scenarios": scenarios}
            if has_base_column:
                row_payload["base"] = base
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
