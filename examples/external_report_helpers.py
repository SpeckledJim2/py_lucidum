"""Routine YAML and feature-spec preparation for the 02 report examples.

The two 02 scripts are the files intended for reading and adapting.  Keeping
path handling and blank-value fallbacks here lets those scripts remain a short
load -> chart -> write sequence.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

import yaml


CHART_COLUMNS = {
    "banding": "chart_banding",
    "quantiles": "chart_quantiles",
    "low_weights": "chart_low_weights",
    "missings": "chart_missings",
    "labels": "chart_labels",
    "sort": "chart_sort",
    "transform": "chart_transform",
    "sigma": "chart_sigma",
    "date_bucket": "chart_date_bucket",
    "empty_periods": "chart_empty_periods",
}


def config_path_from_command_line(script_file: str, default_name: str) -> Path:
    """Return the optional YAML path, or the config beside the script."""

    script_path = Path(script_file).resolve()
    if not _running_as_script(script_path):
        return script_path.with_name(default_name)
    parser = argparse.ArgumentParser()
    parser.add_argument("config", nargs="?", default=script_path.with_name(default_name))
    return Path(parser.parse_args().config).expanduser().resolve()


def load_report_settings(path: Path, model_type: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Read one report YAML and return ready-to-use settings and features."""

    report_config = _read_yaml(path)
    build_path = _resolve(path.parent, report_config["build_config"])
    build_config = _read_yaml(build_path)

    dataset_path = _resolve(build_path.parent, build_config["dataset"]["path"])
    feature_spec_path = _resolve(path.parent, report_config["features"]["spec_path"])
    output_directory = _resolve(path.parent, report_config["output"]["directory"])
    scenario = str(report_config["features"]["scenario_column"])
    feature_rows = _read_feature_spec(feature_spec_path, scenario)
    if not feature_rows:
        raise ValueError(f"Feature scenario selects no report features: {scenario}")

    defaults = dict(report_config.get("chart_defaults") or {})
    report_features = []
    for row in feature_rows:
        name = str(row["Feature"]).strip()
        controls = {}
        for setting, column in CHART_COLUMNS.items():
            value = row.get(column)
            if setting == "banding" and _blank(value):
                value = row.get("banding")
            controls[setting] = defaults.get(setting) if _blank(value) else value
        controls["base"] = row.get("Base") or ""
        report_features.append({"name": name, "controls": controls})

    dataset = build_config["dataset"]
    model_id = str(build_config["model"]["id"])
    settings = {
        "model_type": model_type,
        "model_id": model_id,
        "model_label": str(build_config["model"]["label"]),
        "model_folder": _model_folder(dataset_path, model_type, model_id),
        "dataset_path": dataset_path,
        "actual": str(dataset["response_numerator"]),
        "denominator": dataset.get("denominator"),
        "sample_column": str(dataset["sample_column"]),
        "expected": str(report_config["chart"]["expected"]),
        "expected_label": str(report_config["chart"].get("expected_label") or "Expected"),
        "expected_source": str(report_config["chart"].get("expected_source") or model_type),
        "feature_spec_path": feature_spec_path,
        "scenario": scenario,
        "reports": list(report_config["reports"]),
        "output_directory": output_directory,
        "chart_height": int(report_config["output"].get("chart_height", 600)),
        "config_path": path,
        "build_config_path": build_path,
    }
    return settings, report_features


def report_header(settings: dict[str, Any], report: dict[str, Any], script_file: str) -> dict[str, Any]:
    """Return the small provenance block shown above each HTML report."""

    samples = report["sample_values"]
    sample_text = "ALL" if isinstance(samples, str) else ", ".join(str(value) for value in samples)
    return {
        "source parquet": settings["dataset_path"],
        "model": settings["model_folder"],
        "response": settings["actual"],
        "weight": settings["denominator"] or "None",
        "expected": settings["expected"],
        "SAMPLE_ROWS": sample_text,
        "feature scenario": settings["scenario"],
        "report config": settings["config_path"].name,
        "build config": settings["build_config_path"].name,
        "script run": Path(script_file).name,
    }


def _read_yaml(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ValueError(f"Config file does not exist: {path}")
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Config must be a YAML mapping: {path}")
    return value


def _read_feature_spec(path: Path, scenario: str) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if scenario not in (reader.fieldnames or []):
            raise ValueError(f"Feature Specification has no scenario column: {scenario}")
        return [
            row
            for row in reader
            if str(row.get("Feature") or "").strip()
            and "feature" in str(row.get(scenario) or "").strip().lower()
        ]


def _model_folder(dataset_path: Path, model_type: str, model_id: str) -> Path:
    """Return the exact installed model folder used by the report."""

    from py_lucidum.core import Dataset

    dataset = Dataset(dataset_path)
    try:
        if model_type == "glm":
            from py_lucidum.tools.glm.store import GlmModelStore

            store = GlmModelStore(dataset_path, dataset=dataset)
        else:
            from py_lucidum.tools.gbm.store import GbmModelStore

            store = GbmModelStore(dataset_path, dataset=dataset)
        return store.model_dir(model_id).resolve()
    finally:
        dataset.con.close()


def _resolve(parent: Path, value: Any) -> Path:
    path = Path(str(value)).expanduser()
    return (parent / path).resolve() if not path.is_absolute() else path.resolve()


def _blank(value: Any) -> bool:
    return value is None or str(value).strip() == ""


def _running_as_script(script_path: Path) -> bool:
    import sys

    try:
        return Path(sys.argv[0]).resolve() == script_path
    except (IndexError, OSError):
        return False
