"""Routine YAML and feature-spec preparation for the 02 report examples.

The two 02 scripts are the files intended for reading and adapting.  Keeping
path handling and blank-value fallbacks here lets those scripts remain a short
load -> chart -> write sequence.
"""

from __future__ import annotations

import argparse
import csv
import math
import re
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
MODEL_ID_RE = re.compile(r"[A-Za-z0-9_.-]+")
DOUBLE_LIFT_CHART_KEYS = {"banding", "quantiles", "low_weights", "missings", "labels", "sigma"}


def config_path_from_command_line(script_file: str | None, default_name: str) -> Path:
    """Return the optional YAML path, or the matching example config."""

    if script_file is None:
        return Path(__file__).resolve().with_name(default_name)

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
    kpi_spec_value = report_config.get("kpi_spec")
    kpi_spec_path = (
        _resolve(path.parent, kpi_spec_value)
        if kpi_spec_value is not None and str(kpi_spec_value).strip()
        else None
    )
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
    model_results_root = _resolve(
        build_path.parent,
        build_config["output"]["model_results_root"],
    )
    model_folder = model_results_root / model_type / model_id
    needs_importance = any(
        report["show_feature_importance"] or report["sort_by_feature_importance"]
        for report in reports
    )
    model_folder, importance = _model_details(
        dataset_path,
        model_type,
        model_id,
        model_folder=model_folder,
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
        "kpi_spec_path": kpi_spec_path,
        "scenario": scenario,
        "reports": reports,
        "output_directory": output_directory,
        "chart_height": int(report_config["output"].get("chart_height", 600)),
        "config_path": path,
        "build_config_path": build_path,
    }
    return settings, report_features


def load_double_lift_settings(path: Path) -> dict[str, Any]:
    """Resolve two exact external builds for a standalone Double Lift report."""

    config = _read_yaml(path)
    _expect_exact_keys(
        config,
        {"baseline", "challenger", "reports", "output"},
        {"kpi_spec", "chart"},
        "Double Lift config",
    )
    models = {
        role: _double_lift_model(path, config[role], role)
        for role in ("baseline", "challenger")
    }
    baseline = models["baseline"]
    challenger = models["challenger"]
    if (baseline["model_type"], baseline["model_id"]) == (
        challenger["model_type"],
        challenger["model_id"],
    ):
        raise ValueError("Baseline and Challenger must be different saved models")
    for key, label in (
        ("dataset_path", "dataset"),
        ("actual", "response Numerator"),
        ("denominator", "Denominator"),
        ("sample_column", "SAMPLE column"),
    ):
        if baseline[key] != challenger[key]:
            raise ValueError(
                f"Baseline and Challenger build configs use different {label}: "
                f"{baseline[key]!s} versus {challenger[key]!s}"
            )

    raw_chart = config.get("chart") or {}
    if not isinstance(raw_chart, dict):
        raise ValueError("Double Lift chart must be a YAML mapping")
    _expect_exact_keys(raw_chart, set(), DOUBLE_LIFT_CHART_KEYS, "Double Lift chart")
    chart = {
        "banding": raw_chart.get("banding", "auto"),
        "quantiles": raw_chart.get("quantiles", 0),
        "low_weights": raw_chart.get("low_weights", "0"),
        "missings": raw_chart.get("missings", "hide"),
        "labels": raw_chart.get("labels", "none"),
        "sigma": raw_chart.get("sigma", 0),
    }
    if str(chart["banding"] or "").strip().lower() != "auto":
        try:
            if float(chart["banding"]) < 0:
                raise ValueError
        except (TypeError, ValueError) as exc:
            raise ValueError("Double Lift chart banding must be 'auto' or a non-negative number") from exc

    raw_reports = config["reports"]
    if not isinstance(raw_reports, list) or not raw_reports:
        raise ValueError("Double Lift reports must contain at least one report")
    reports = []
    seen_names: set[str] = set()
    for index, raw_report in enumerate(raw_reports, start=1):
        if not isinstance(raw_report, dict):
            raise ValueError(f"Double Lift report {index} must be a YAML mapping")
        _expect_exact_keys(raw_report, {"name", "title", "sample_values"}, set(), f"Double Lift report {index}")
        name = str(raw_report["name"] or "").strip()
        title = str(raw_report["title"] or "").strip()
        if not name or not title:
            raise ValueError(f"Double Lift report {index} needs nonblank name and title")
        if name in seen_names:
            raise ValueError(f"Double Lift report names must be unique: {name}")
        seen_names.add(name)
        reports.append({"name": name, "title": title, "sample_values": raw_report["sample_values"]})

    output = config["output"]
    if not isinstance(output, dict):
        raise ValueError("Double Lift output must be a YAML mapping")
    _expect_exact_keys(output, {"directory"}, {"chart_height"}, "Double Lift output")
    chart_height = int(output.get("chart_height", 600))
    if chart_height < 200:
        raise ValueError("Double Lift chart_height must be at least 200 pixels")
    kpi_value = config.get("kpi_spec")
    kpi_path = (
        _resolve(path.parent, kpi_value)
        if kpi_value is not None and str(kpi_value).strip()
        else None
    )
    return {
        "baseline": baseline,
        "challenger": challenger,
        "dataset_path": baseline["dataset_path"],
        "actual": baseline["actual"],
        "denominator": baseline["denominator"],
        "sample_column": baseline["sample_column"],
        "chart": chart,
        "reports": reports,
        "kpi_spec_path": kpi_path,
        "output_directory": _resolve(path.parent, output["directory"]),
        "chart_height": chart_height,
        "config_path": path.resolve(),
    }


def double_lift_report_header(
    settings: dict[str, Any],
    report: dict[str, Any],
    chart: dict[str, Any],
    script_file: str,
) -> dict[str, Any]:
    """Return visible population and two-model provenance for Double Lift."""

    metadata = chart["metadata"]
    baseline = metadata["baseline"]
    challenger = metadata["challenger"]
    return {
        "source parquet": settings["dataset_path"],
        "baseline build config": settings["baseline"]["build_config_path"],
        "baseline model": baseline["model_folder"],
        "challenger build config": settings["challenger"]["build_config_path"],
        "challenger model": challenger["model_folder"],
        "baseline": f"{baseline['model_type'].upper()} · {baseline['label']} ({baseline['model_id']})",
        "challenger": f"{challenger['model_type'].upper()} · {challenger['label']} ({challenger['model_id']})",
        "ratio": "Challenger / Baseline",
        "response": settings["actual"],
        "weight": settings["denominator"] or "None",
        "sample column": settings["sample_column"],
        "SAMPLE_ROWS": metadata["sample_values"],
        "source rows selected": f"{int(metadata['selected_rows']):,}",
        "rows available to chart": f"{int(metadata['chart_rows']):,}",
        **(
            {"KPI spec": settings["kpi_spec_path"].name}
            if settings["kpi_spec_path"] is not None
            else {}
        ),
        "comparison config": settings["config_path"].name,
        "script run": Path(script_file).name,
    }


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
    model_results_root = _resolve(
        build_path.parent,
        build_config["output"]["model_results_root"],
    )
    model_folder = (model_results_root / "gbm" / model_id).resolve()

    dataset = Dataset(dataset_path)
    try:
        store = GbmModelStore(
            dataset_path,
            dataset=dataset,
            model_root=model_folder.parent,
        )
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
            split_ginis={
                "training": manifest.get("gini_tr"),
                "test": manifest.get("gini_te"),
                "validation": manifest.get("gini_vl"),
            },
        )
        importance_payload = gbm_model_importance(store, model_id)
        importance = _summary_importance(importance_payload, model_id)
        if store.model_dir(model_id).resolve() != model_folder or not model_folder.is_dir():
            raise ValueError(f"GBM model results folder does not exist: {model_folder}")
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
            dataset_config["validation_value"],
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
    sample_values = {
        "training_value": str(dataset_config["training_value"]).strip(),
        "test_value": str(dataset_config.get("test_value") or "test").strip(),
        "validation_value": str(
            dataset_config.get("validation_value") or "validation"
        ).strip(),
    }
    report = dict(report_config["report"])
    model_id = str(build_config["model"]["id"])
    model_results_root = _resolve(
        build_path.parent,
        build_config["output"]["model_results_root"],
    )
    return {
        "model_id": model_id,
        "model_folder": (model_results_root / "glm" / model_id).resolve(),
        "install_in_lucidum": bool(build_config["output"]["install_in_lucidum"]),
        "replace_existing": bool(build_config["output"]["replace_existing"]),
        "dataset_path": _resolve(build_path.parent, dataset_config["path"]),
        "sample_column": str(dataset_config["sample_column"]),
        **sample_values,
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
        "model": settings["model_folder"],
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
            details = [
                f"Rank {row['importance_rank']}",
                f"Importance {row['importance_percent']:.1f}%",
            ]
            monotonicity = str(row.get("monotonicity") or "").strip()
            if report.get("chart_content") == "shap_only" and monotonicity:
                details.append(monotonicity)
            row["title"] = f"{name} ({', '.join(details)})"
        elif show_importance:
            row["title"] = f"{name} (Not in model)"
        else:
            monotonicity = str(row.get("monotonicity") or "").strip()
            row["title"] = (
                f"{name} ({monotonicity})"
                if report.get("chart_content") == "shap_only" and row.get("in_model") and monotonicity
                else name
            )
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
    if settings["kpi_spec_path"] is not None:
        header["KPI spec"] = settings["kpi_spec_path"].name
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
    split_ginis: dict[str, Any] | None = None,
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
        ("Training", "training", str(dataset_config["training_value"]), best_metrics.get("training")),
        ("Test", "test", str(dataset_config["early_stopping_value"]), best_metrics.get("test")),
        ("Validation", "validation", str(dataset_config["validation_value"]), best_metrics.get("validation")),
    ]
    rows = []
    for label, role, sample_value, metric_value in roles:
        values = results.get(sample_value.strip().lower())
        if not values:
            raise ValueError(f"The {label} SAMPLE value has no eligible scored rows: {sample_value}")
        gini = _finite_number((split_ginis or {}).get(role))
        rows.append(
            {
                "sample": label,
                "rows": f"{values['row_count']:,}",
                "weight": _format_weight(values["weight"]) if denominator else None,
                "actual": _format_kpi(values["actual"], kpi),
                "prediction": _format_kpi(values["prediction"], kpi),
                "metric": _format_metric(metric_value, metric),
                "gini": "—" if gini is None else f"{gini:.4f}",
                "raw": {**values, "gini": gini},
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
            {"key": "gini", "label": "Normalized Gini"},
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
            {"key": "monotonicity", "label": "Monotonicity"},
            {"key": "importance", "label": "SHAP" if uses_shap else "Gain"},
            {"key": "share", "label": "Share"},
        ],
        "rows": [
            {
                "rank": int(row["rank"]),
                "feature": str(row["feature"]),
                "monotonicity": str(row.get("monotonicity") or "None"),
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


def _double_lift_model(comparison_path: Path, raw: Any, role: str) -> dict[str, Any]:
    label = role.title()
    if not isinstance(raw, dict):
        raise ValueError(f"{label} must be a YAML mapping")
    _expect_exact_keys(raw, {"model_type", "build_config"}, set(), label)
    model_type = str(raw["model_type"] or "").strip().lower()
    if model_type not in {"glm", "gbm"}:
        raise ValueError(f"{label} model_type must be 'glm' or 'gbm'")
    build_path = _resolve(comparison_path.parent, raw["build_config"])
    build = _read_yaml(build_path)
    dataset = build.get("dataset")
    model = build.get("model")
    output = build.get("output")
    for section_name, section in (("dataset", dataset), ("model", model), ("output", output)):
        if not isinstance(section, dict):
            raise ValueError(f"{label} build config needs a {section_name} mapping: {build_path}")
    required_dataset = {"path", "response_numerator", "sample_column"}
    missing_dataset = sorted(required_dataset - set(dataset))
    if missing_dataset:
        raise ValueError(
            f"{label} build config dataset is missing: {', '.join(missing_dataset)}"
        )
    model_id = str(model.get("id") or "").strip()
    if model_id in {"", ".", ".."} or not MODEL_ID_RE.fullmatch(model_id):
        raise ValueError(f"{label} build config has an invalid model.id: {model_id!r}")
    model_label = str(model.get("label") or "").strip()
    if not model_label:
        raise ValueError(f"{label} build config needs model.label")
    model_results_root = output.get("model_results_root")
    if model_results_root is None or not str(model_results_root).strip():
        raise ValueError(f"{label} build config needs output.model_results_root")
    actual = str(dataset["response_numerator"] or "").strip()
    sample_column = str(dataset["sample_column"] or "").strip()
    if not actual or not sample_column:
        raise ValueError(f"{label} build config needs nonblank response_numerator and sample_column")
    denominator = str(dataset.get("denominator") or "").strip() or None
    root = _resolve(build_path.parent, model_results_root)
    return {
        "model_type": model_type,
        "model_id": model_id,
        "model_label": model_label,
        "model_folder": (root / model_type / model_id).resolve(),
        "build_config_path": build_path,
        "dataset_path": _resolve(build_path.parent, dataset["path"]),
        "actual": actual,
        "denominator": denominator,
        "sample_column": sample_column,
    }


def _expect_exact_keys(
    value: dict[str, Any],
    required: set[str],
    optional: set[str],
    label: str,
) -> None:
    missing = sorted(required - set(value))
    unknown = sorted(set(value) - required - optional)
    if missing:
        raise ValueError(f"{label} is missing keys: {', '.join(missing)}")
    if unknown:
        raise ValueError(f"{label} has unknown keys: {', '.join(unknown)}")


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
    model_folder: Path,
    needs_importance: bool,
) -> tuple[Path, dict[str, Any] | None]:
    """Return the named model folder and, when requested, its importance."""

    from py_lucidum.core import Dataset

    dataset = Dataset(dataset_path)
    try:
        if model_type == "glm":
            from py_lucidum.tools.glm.store import GlmModelStore
            from py_lucidum.tools.line_bar.importance import glm_model_importance

            store = GlmModelStore(
                dataset_path,
                dataset=dataset,
                model_root=model_folder.parent,
            )
            importance = glm_model_importance(store, model_id) if needs_importance else None
            manifest = store.manifest(model_id)
            formula = manifest.get("formula")
            if (
                needs_importance
                and isinstance(importance, dict)
                and isinstance(formula, dict)
                and bool(formula.get("intercept_only"))
            ):
                importance = {**importance, "intercept_only": True, "message": ""}
        else:
            from py_lucidum.tools.gbm.store import GbmModelStore
            from py_lucidum.tools.line_bar.importance import gbm_model_importance

            store = GbmModelStore(
                dataset_path,
                dataset=dataset,
                model_root=model_folder.parent,
            )
            importance = gbm_model_importance(store, model_id) if needs_importance else None
        resolved_folder = store.model_dir(model_id).resolve()
        if resolved_folder != model_folder.resolve() or not resolved_folder.is_dir():
            raise ValueError(f"{model_type.upper()} model results folder does not exist: {model_folder}")
        return resolved_folder, importance
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
        if bool((importance or {}).get("intercept_only")):
            # An intercept-only GLM legitimately contains no model features.
            # Keep every requested chart feature, but label it as not in model.
            for feature in features:
                feature["in_model"] = False
            return
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
        feature["monotonicity"] = str(saved.get("monotonicity") or "")


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
