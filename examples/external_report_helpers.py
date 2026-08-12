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

    reports = [dict(report) for report in report_config["reports"]]
    for report in reports:
        report["show_feature_importance"] = _report_boolean(report, "show_feature_importance")
        report["sort_by_feature_importance"] = _report_boolean(report, "sort_by_feature_importance")

    dataset = build_config["dataset"]
    model_id = str(build_config["model"]["id"])
    needs_importance = any(
        report["show_feature_importance"] or report["sort_by_feature_importance"]
        for report in reports
    )
    model_folder, importance = _model_details(
        dataset_path,
        model_type,
        model_id,
        needs_importance=needs_importance,
    )
    if needs_importance:
        _add_feature_importance(report_features, importance, model_type, model_id)

    settings = {
        "model_type": model_type,
        "model_id": model_id,
        "model_label": str(build_config["model"]["label"]),
        "model_folder": model_folder,
        "importance_measure": _importance_measure(importance) if needs_importance else "",
        "dataset_path": dataset_path,
        "actual": str(dataset["response_numerator"]),
        "denominator": dataset.get("denominator"),
        "sample_column": str(dataset["sample_column"]),
        "expected": str(report_config["chart"]["expected"]),
        "expected_label": str(report_config["chart"].get("expected_label") or "Expected"),
        "expected_source": str(report_config["chart"].get("expected_source") or model_type),
        "feature_spec_path": feature_spec_path,
        "scenario": scenario,
        "reports": reports,
        "output_directory": output_directory,
        "chart_height": int(report_config["output"].get("chart_height", 600)),
        "config_path": path,
        "build_config_path": build_path,
    }
    return settings, report_features


def features_for_report(features: list[dict[str, Any]], report: dict[str, Any]) -> list[dict[str, Any]]:
    """Return chart-ready feature rows in the order requested by one report."""

    show_importance = report["show_feature_importance"]
    sort_by_importance = report["sort_by_feature_importance"]
    prepared = []
    for feature in features:
        row = dict(feature)
        name = row["name"]
        if show_importance and row.get("in_model"):
            row["title"] = (
                f"{name} (Rank {row['importance_rank']}, "
                f"Importance {row['importance_percent']:.1f}%)"
            )
        elif show_importance:
            row["title"] = f"{name} (Not in model)"
        else:
            row["title"] = name
        prepared.append(row)

    if sort_by_importance:
        prepared.sort(
            key=lambda row: (
                0 if row.get("in_model") else 1,
                int(row.get("importance_rank") or 0) if row.get("in_model") else 0,
                str(row["name"]).casefold(),
            )
        )
    return prepared


def report_header(settings: dict[str, Any], report: dict[str, Any], script_file: str) -> dict[str, Any]:
    """Return the small provenance block shown above each HTML report."""

    samples = report["sample_values"]
    sample_text = "ALL" if isinstance(samples, str) else ", ".join(str(value) for value in samples)
    header = {
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
    if report["show_feature_importance"] or report["sort_by_feature_importance"]:
        header["importance measure"] = settings["importance_measure"]
    return header


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


def _model_details(
    dataset_path: Path,
    model_type: str,
    model_id: str,
    *,
    needs_importance: bool,
) -> tuple[Path, dict[str, Any] | None]:
    """Return the named model folder and, when requested, its importance."""

    from py_lucidum.core import Dataset

    dataset = Dataset(dataset_path)
    try:
        if model_type == "glm":
            from py_lucidum.tools.glm.store import GlmModelStore
            from py_lucidum.tools.line_bar.importance import glm_model_importance

            store = GlmModelStore(dataset_path, dataset=dataset)
            importance = glm_model_importance(store, model_id) if needs_importance else None
        else:
            from py_lucidum.tools.gbm.store import GbmModelStore
            from py_lucidum.tools.line_bar.importance import gbm_model_importance

            store = GbmModelStore(dataset_path, dataset=dataset)
            importance = gbm_model_importance(store, model_id) if needs_importance else None
        return store.model_dir(model_id).resolve(), importance
    except Exception as exc:
        if needs_importance:
            raise ValueError(
                f"Could not read feature importance for {model_type.upper()} model '{model_id}'. "
                "Rebuild that model before creating this report."
            ) from exc
        raise
    finally:
        dataset.con.close()


def _add_feature_importance(
    features: list[dict[str, Any]],
    importance: dict[str, Any] | None,
    model_type: str,
    model_id: str,
) -> None:
    rows = list((importance or {}).get("rows") or [])
    if not rows:
        message = str((importance or {}).get("message") or "").strip()
        raise ValueError(
            message
            or (
                f"{model_type.upper()} model '{model_id}' has no saved feature importances. "
                "Rebuild the model to calculate them."
            )
        )

    total = sum(max(0.0, float(row.get("importance") or 0.0)) for row in rows)
    by_name = {str(row["feature"]): row for row in rows}
    for feature in features:
        saved = by_name.get(feature["name"])
        feature["in_model"] = saved is not None
        if saved is None:
            continue
        value = max(0.0, float(saved.get("importance") or 0.0))
        feature["importance_rank"] = int(saved["rank"])
        feature["importance_percent"] = value / total * 100 if total > 0 else 0.0


def _report_boolean(report: dict[str, Any], name: str) -> bool:
    value = report.get(name, False)
    if not isinstance(value, bool):
        raise ValueError(f"reports.{name} must be true or false")
    return value


def _importance_measure(importance: dict[str, Any] | None) -> str:
    metric = str((importance or {}).get("metric") or "")
    labels = {
        "weighted_mean_abs_centered_linear_predictor_contribution": (
            "Weighted mean absolute centred linear-predictor contribution"
        ),
        "mean_abs_shap": "Mean absolute SHAP",
        "gain": "LightGBM gain",
    }
    return labels.get(metric) or str((importance or {}).get("metric_label") or metric)


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
