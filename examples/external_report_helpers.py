"""Routine YAML and feature-spec preparation for the 02 report examples.

The two 02 scripts are the files intended for reading and adapting.  Keeping
path handling and blank-value fallbacks here lets those scripts remain a short
load -> chart -> write sequence.
"""

from __future__ import annotations

import argparse
import csv
import math
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


def load_gbm_summary_settings(
    path: Path,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Load one summary YAML and the saved results needed by the 03 script."""

    from py_lucidum.core import Dataset
    from py_lucidum.core.kpis import load_kpis
    from py_lucidum.tools.gbm.store import GbmModelStore
    from py_lucidum.tools.line_bar.importance import gbm_model_importance

    report_config = _read_yaml(path)
    build_path = _resolve(path.parent, report_config["build_config"])
    build_config = _read_yaml(build_path)
    dataset_config = build_config["dataset"]
    dataset_path = _resolve(build_path.parent, dataset_config["path"])
    kpi_path = _resolve(path.parent, report_config["kpi_spec"])
    output_directory = _resolve(path.parent, report_config["output"]["directory"])
    model_id = str(build_config["model"]["id"])

    dataset = Dataset(dataset_path)
    try:
        store = GbmModelStore(dataset_path, dataset=dataset)
        manifest = store.manifest(model_id)
        parameters = store.model_parameters(model_id)
        prediction_path = store.artifact_path(model_id, "predictions")
        if not prediction_path.is_file():
            raise ValueError(f"GBM predictions are unavailable for model {model_id}")
        kpi = _summary_kpi(
            load_kpis(kpi_path),
            str(dataset_config["response_numerator"]),
            dataset_config.get("denominator"),
            kpi_path,
        )
        performance = _gbm_performance(
            dataset,
            prediction_path,
            dataset_config,
            store.model_best_metrics(
                model_id,
                str(parameters.get("metric") or ""),
                manifest.get("best_iteration"),
            ),
            kpi,
            best_iteration=manifest.get("best_iteration"),
            metric=str(parameters.get("metric") or ""),
        )
        importance_payload = gbm_model_importance(store, model_id)
        importance = _summary_importance(importance_payload, model_id)
        model_folder = store.model_dir(model_id).resolve()
    finally:
        dataset.con.close()

    report = dict(report_config["report"])
    settings = {
        "model_id": model_id,
        "model_label": str(build_config["model"]["label"]),
        "model_folder": model_folder,
        "dataset_path": dataset_path,
        "actual": str(dataset_config["response_numerator"]),
        "denominator": dataset_config.get("denominator"),
        "sample_column": str(dataset_config["sample_column"]),
        "sample_values": [
            dataset_config["training_value"],
            dataset_config["early_stopping_value"],
            dataset_config["holdout_value"],
        ],
        "report_name": str(report["name"]),
        "report_title": str(report["title"]),
        "output_directory": output_directory,
        "chart_height": int(report_config["output"].get("chart_height", 600)),
        "importance_measure": importance["measure"],
        "config_path": path,
        "build_config_path": build_path,
        "kpi_spec_path": kpi_path,
    }
    return settings, performance, importance, parameters


def load_glm_summary_settings(path: Path) -> dict[str, Any]:
    """Load the paths and labels used by the standalone GLM summary."""

    report_config = _read_yaml(path)
    build_path = _resolve(path.parent, report_config["build_config"])
    build_config = _read_yaml(build_path)
    dataset_config = build_config["dataset"]
    report = dict(report_config["report"])
    return {
        "model_id": str(build_config["model"]["id"]),
        "dataset_path": _resolve(build_path.parent, dataset_config["path"]),
        "feature_spec_path": _resolve(path.parent, report_config["feature_spec"]),
        "kpi_spec_path": _resolve(path.parent, report_config["kpi_spec"]),
        "report_name": str(report["name"]),
        "report_title": str(report["title"]),
        "output_directory": _resolve(path.parent, report_config["output"]["directory"]),
        "config_path": path,
        "build_config_path": build_path,
    }


def glm_summary_header(settings: dict[str, Any], script_file: str) -> dict[str, Any]:
    """Return the small provenance block for the standalone GLM summary."""

    return {
        "feature spec": settings["feature_spec_path"].name,
        "KPI spec": settings["kpi_spec_path"].name,
        "report config": settings["config_path"].name,
        "build config": settings["build_config_path"].name,
        "script run": Path(script_file).name,
    }


def gbm_summary_header(settings: dict[str, Any], script_file: str) -> dict[str, Any]:
    """Return the provenance block for the standalone GBM summary."""

    return {
        "source parquet": settings["dataset_path"],
        "model": settings["model_folder"],
        "response": settings["actual"],
        "weight": settings["denominator"] or "None",
        "expected": "gbm_prediction",
        "SAMPLE_ROWS": settings["sample_values"],
        "model label": settings["model_label"],
        "KPI spec": settings["kpi_spec_path"].name,
        "report config": settings["config_path"].name,
        "build config": settings["build_config_path"].name,
        "script run": Path(script_file).name,
    }


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


def _gbm_performance(
    dataset: Any,
    prediction_path: Path,
    dataset_config: dict[str, Any],
    best_metrics: dict[str, float | None],
    kpi: dict[str, Any],
    *,
    best_iteration: Any,
    metric: str,
) -> dict[str, Any]:
    from py_lucidum.core import quote_ident, sql_literal

    actual = quote_ident(str(dataset_config["response_numerator"]))
    sample = quote_ident(str(dataset_config["sample_column"]))
    denominator_name = str(dataset_config.get("denominator") or "").strip()
    denominator = quote_ident(denominator_name) if denominator_name else ""
    valid_weight = (
        f"AND isfinite(TRY_CAST({denominator} AS DOUBLE)) "
        f"AND TRY_CAST({denominator} AS DOUBLE) > 0"
        if denominator
        else ""
    )
    weight_projection = f", TRY_CAST({denominator} AS DOUBLE) AS report_weight" if denominator else ""
    weight_summary = "SUM(report_weight) AS weight_sum," if denominator else "NULL AS weight_sum,"
    actual_summary = (
        "SUM(actual_value) / SUM(report_weight) AS actual_value," if denominator else "AVG(actual_value) AS actual_value,"
    )
    prediction_summary = (
        "SUM(prediction_value) / SUM(report_weight) AS prediction_value" if denominator else "AVG(prediction_value) AS prediction_value"
    )
    query = f"""
WITH source_rows AS (
  SELECT ROW_NUMBER() OVER ()::BIGINT AS __lucidum_row_id, *
  FROM {dataset.relation_sql()}
), eligible AS (
  SELECT
    LOWER(TRIM(CAST({sample} AS VARCHAR))) AS sample_value,
    TRY_CAST({actual} AS DOUBLE) AS actual_value,
    TRY_CAST(prediction.gbm_prediction AS DOUBLE) AS prediction_value
    {weight_projection}
  FROM source_rows
  INNER JOIN read_parquet({sql_literal(str(prediction_path))}) prediction
    USING (__lucidum_row_id)
  WHERE isfinite(TRY_CAST({actual} AS DOUBLE))
    AND isfinite(TRY_CAST(prediction.gbm_prediction AS DOUBLE))
    {valid_weight}
)
SELECT
  sample_value,
  COUNT(*)::BIGINT AS row_count,
  {weight_summary}
  {actual_summary}
  {prediction_summary}
FROM eligible
GROUP BY sample_value
"""
    results = {
        str(row[0]): {
            "row_count": int(row[1]),
            "weight": _finite_number(row[2]),
            "actual": _finite_number(row[3]),
            "prediction": _finite_number(row[4]),
        }
        for row in dataset.con.execute(query).fetchall()
    }
    roles = [
        ("Training", str(dataset_config["training_value"]), best_metrics.get("training")),
        ("Test", str(dataset_config["early_stopping_value"]), best_metrics.get("test")),
        ("Validation", str(dataset_config["holdout_value"]), best_metrics.get("validation")),
    ]
    rows = []
    for label, sample_value, metric_value in roles:
        values = results.get(sample_value.strip().lower())
        if not values:
            raise ValueError(f"The {label} SAMPLE value has no eligible scored rows: {sample_value}")
        rows.append(
            {
                "sample": label,
                "rows": f"{values['row_count']:,}",
                "weight": _format_weight(values["weight"]) if denominator else None,
                "actual": _format_kpi(values["actual"], kpi),
                "prediction": _format_kpi(values["prediction"], kpi),
                "metric": _format_metric(metric_value, metric),
                "raw": values,
            }
        )

    columns = [
        {"key": "sample", "label": "Sample"},
        {"key": "rows", "label": "Number of rows"},
    ]
    if denominator:
        columns.append({"key": "weight", "label": f"Sum of {denominator_name}"})
    columns.extend(
        [
            {"key": "actual", "label": "Actual response"},
            {"key": "prediction", "label": "Model prediction"},
            {"key": "metric", "label": f"{metric or 'Model'} metric"},
        ]
    )
    return {
        "columns": columns,
        "rows": rows,
        "metric": metric,
        "best_iteration": int(best_iteration or 0),
        "kpi": dict(kpi),
    }


def _summary_kpi(
    kpis: list[dict[str, Any]],
    actual: str,
    denominator: Any,
    path: Path,
) -> dict[str, Any]:
    denominator_name = str(denominator or "").strip() or "__none__"
    for kpi in kpis:
        if kpi["actual"] == actual and kpi["denominator"] == denominator_name:
            return dict(kpi)
    weight = denominator_name if denominator_name != "__none__" else "Average row value"
    raise ValueError(
        f"KPI specification {path} has no row for Actual {actual!r} and Weight {weight!r}"
    )


def _summary_importance(importance: dict[str, Any], model_id: str) -> dict[str, Any]:
    rows = list(importance.get("rows") or [])
    if not rows:
        raise ValueError(
            str(importance.get("message") or "")
            or f"GBM model '{model_id}' has no saved feature importances. Rebuild the model."
        )
    metric = str(importance.get("metric") or "")
    uses_shap = metric == "mean_abs_shap"
    total = sum(max(0.0, float(row.get("importance") or 0.0)) for row in rows)
    return {
        "measure": _importance_measure(importance),
        "metric": metric,
        "columns": [
            {"key": "rank", "label": "Rank"},
            {"key": "feature", "label": "Feature"},
            {"key": "importance", "label": "SHAP" if uses_shap else "Gain"},
            {"key": "share", "label": "Share"},
        ],
        "rows": [
            {
                "rank": int(row["rank"]),
                "feature": str(row["feature"]),
                "importance": (
                    f"{float(row.get('importance') or 0.0) * 100:.1f}%"
                    if uses_shap
                    else _format_compact(row.get("importance"))
                ),
                "share": f"{max(0.0, float(row.get('importance') or 0.0)) / total * 100:.1f}%" if total > 0 else "0.0%",
                "raw_importance": float(row.get("importance") or 0.0),
            }
            for row in rows
        ],
    }


def _format_kpi(value: Any, kpi: dict[str, Any]) -> str:
    number = _finite_number(value)
    if number is None:
        return "—"
    display = number * 100 if kpi["format"] == "percent" else number
    decimals = int(kpi["decimals"])
    sign = "-" if display < 0 else ""
    formatted = f"{abs(display):,.{decimals}f}"
    if kpi["format"] == "currency":
        return f"{sign}£{formatted}"
    if kpi["format"] == "percent":
        return f"{sign}{formatted}%"
    return f"{sign}{formatted}"


def _format_weight(value: Any) -> str:
    number = _finite_number(value)
    if number is None:
        return "—"
    return f"{number:,.0f}" if abs(number) >= 10 or number.is_integer() else _format_compact(number)


def _format_metric(value: Any, metric: str) -> str:
    number = _finite_number(value)
    if number is None:
        return "—"
    if str(metric or "").strip().lower() == "mape":
        return f"{number * 100:.1f}%"
    return _format_compact(number)


def _format_compact(value: Any) -> str:
    number = _finite_number(value)
    if number is None:
        return "—"
    absolute = abs(number)
    decimals = 1 if absolute >= 1000 else 2 if absolute >= 10 else 3 if absolute >= 1 else 4 if absolute >= 0.01 else 6
    return f"{number:,.{decimals}f}".rstrip("0").rstrip(".")


def _finite_number(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


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
