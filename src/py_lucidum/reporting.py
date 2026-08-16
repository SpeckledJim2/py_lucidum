from __future__ import annotations

import html
import json
import math
import pickle
import re
from collections.abc import Iterable, Mapping, Sequence
from datetime import datetime
from pathlib import Path, PureWindowsPath
from typing import Any
from uuid import uuid4

from py_lucidum._version import __version__
from py_lucidum.core import Dataset, is_numeric_kind, load_features, quote_ident, sql_literal


REPORT_CONTENT_VALUES = {"actual_expected", "shap_only"}
PARTIAL_DEPENDENCE_VALUES = {"none", "glm", "shap"}
LABEL_VALUES = {"none", "bar", "line", "all"}
SORT_VALUES = {"alpha", "volume", "actual", "expected", "shap"}
TRANSFORM_VALUES = {"none", "log", "exp", "logit", "zero", "one"}
MISSINGS_VALUES = {"show", "hide"}
DATE_BUCKET_VALUES = {"none", "hour", "day", "week", "month", "year"}
EMPTY_PERIOD_VALUES = {"show", "skip"}
SIGMA_VALUES = {0, 1, 2, 5}
DEFAULT_REPORT_CHART_HEIGHT = 600


def line_bar_chart(
    dataset_path: str | Path,
    *,
    x: str,
    actual: str,
    expected: str,
    expected_source: str = "dataset",
    expected_label: str = "Expected",
    denominator: str | None = None,
    sample_column: str = "SAMPLE",
    sample_values: str | Sequence[str] = "all",
    model_id: str | None = None,
    model_folder: str | Path | None = None,
    controls: Mapping[str, Any] | None = None,
    content: str = "actual_expected",
    transform: str | None = None,
    partial_dependence: str = "none",
    feature_spec: str | Path | Mapping[str, Any] | None = None,
    title: str | None = None,
) -> dict[str, Any]:
    """Build one static Line/Bar chart specification without starting Lucidum.

    The returned dictionary is deliberately consumed by
    :func:`write_echarts_report`; callers do not need to understand Lucidum's
    internal request or ECharts payload formats.
    """

    path = Path(dataset_path).expanduser().resolve()
    settings = _normalise_chart_controls(controls, transform=transform)
    content = _choice(content, REPORT_CONTENT_VALUES, "chart content")
    partial_dependence = _choice(
        partial_dependence,
        PARTIAL_DEPENDENCE_VALUES,
        "partial dependence",
    )
    expected_source = _choice(expected_source, {"dataset", "glm", "gbm"}, "Expected source")
    if content == "shap_only" and partial_dependence != "shap":
        raise ValueError("SHAP-only charts require partial_dependence='shap'")
    if partial_dependence in {"glm", "shap"} and not str(model_id or "").strip():
        raise ValueError(f"{partial_dependence.upper()} partial dependence requires a model ID")
    if expected_source in {"glm", "gbm"} and not str(model_id or "").strip():
        raise ValueError(f"{expected_source.upper()} Expected values require a model ID")

    dataset = Dataset(path)
    try:
        source_id = _register_report_sources(
            dataset,
            expected_source=expected_source,
            model_id=str(model_id or "").strip(),
            partial_dependence=partial_dependence,
            model_folder=model_folder,
        )
        columns = dataset.column_map()
        _require_column(columns, x, "x-axis")
        _require_numeric_column(columns, actual, "Actual")
        _require_column(columns, sample_column, "SAMPLE")
        if denominator:
            _require_numeric_column(columns, denominator, "Denominator")

        source_columns = dataset.column_map_for_source(source_id)
        _require_numeric_column(source_columns, expected, "Expected")
        filter_sql, resolved_samples, selected_rows = _sample_filter(
            dataset,
            sample_column,
            sample_values,
        )
        if selected_rows <= 0:
            raise ValueError("The selected SAMPLE values contain no rows")

        feature_spec_payload = _feature_spec_payload(feature_spec)
        base = str(settings.get("base") or _feature_base(feature_spec_payload, x) or "").strip()
        request = {
            "source": "dataset",
            "x": x,
            "sort": settings["sort"],
            "lowGroup": settings["low_weights"],
            "bandWidth": settings["quantiles"] or settings["banding"],
            "quantileMode": "quantile" if settings["quantiles"] else "off",
            "dateBucket": settings["date_bucket"],
            "emptyPeriods": settings["empty_periods"],
            "missings": settings["missings"],
            "transform": settings["transform"],
            "partialDependence": {
                "mode": partial_dependence,
                **(
                    {
                        "model_id": str(model_id),
                        **(
                            {"model_folder": str(Path(model_folder).expanduser().resolve())}
                            if model_folder is not None and partial_dependence == "glm"
                            else {}
                        ),
                    }
                    if partial_dependence in {"glm", "shap"}
                    else {}
                ),
            },
            "base": base,
            "sigma": settings["sigma"],
            "filter": filter_sql,
            "denominator": denominator or "__none__",
            "denominatorSource": "dataset",
            "responses": [
                {"label": "Actual", "numerator": actual},
                {
                    "label": expected_label,
                    "numerator": expected,
                    **({"source": source_id} if source_id != "dataset" else {}),
                },
            ],
            "maxGroups": 10_000,
        }

        from py_lucidum.tools.line_bar.query import chart

        payload = chart(dataset, request, feature_spec=feature_spec_payload)
        if payload.get("groups_truncated") or not payload.get("rows"):
            raise ValueError(f"{x} did not produce a plottable Line/Bar result")
        _require_requested_overlay(
            payload,
            partial_dependence=partial_dependence,
            model_id=str(model_id or ""),
            feature=x,
        )
        if content == "shap_only" and settings["transform"] in {"zero", "one"}:
            overlay = _partial_dependence_overlay(payload, "shap")
            transform_payload = overlay.get("transform") if isinstance(overlay, dict) else None
            if not base:
                raise ValueError(f"{x} needs a Feature Specification Base for SHAP rebasing")
            if not isinstance(transform_payload, dict) or transform_payload.get("reference") != "base":
                raise ValueError(f"{x} Base {base!r} could not be resolved for SHAP rebasing")

        return {
            "kind": "line_bar",
            "title": str(title or x),
            "data": payload,
            "presentation": {
                "content": content,
                "labels": settings["labels"],
                "transform": settings["transform"],
                "sigma": settings["sigma"],
                "theme": "light",
            },
            "metadata": {
                "feature": x,
                "sample_column": sample_column,
                "sample_values": resolved_samples,
                "selected_rows": selected_rows,
                "actual": actual,
                "expected": expected,
                "expected_label": expected_label,
                "expected_source": expected_source,
                "denominator": denominator,
                "model_id": str(model_id or ""),
                "model_folder": (
                    str(Path(model_folder).expanduser().resolve())
                    if model_folder is not None
                    else ""
                ),
                "base": base,
                "controls": settings,
                "partial_dependence": partial_dependence,
            },
        }
    finally:
        dataset.con.close()


def gbm_evaluation_chart(
    dataset_path: str | Path,
    *,
    model_id: str,
    model_folder: str | Path | None = None,
    title: str = "Model evaluation",
) -> dict[str, Any]:
    """Return the saved Evaluation Log for one exact GBM as a chart spec."""

    path = Path(dataset_path).expanduser().resolve()
    model_id = str(model_id or "").strip()
    if not model_id:
        raise ValueError("Choose a GBM model ID")

    from py_lucidum.tools.gbm.store import GbmModelStore

    dataset = Dataset(path)
    try:
        store = GbmModelStore(
            path,
            dataset=dataset,
            model_root=_explicit_model_root(model_folder, model_id, "GBM"),
        )
        detail = store.model_detail(model_id)
        evaluation = detail.get("evaluation")
        has_values = any(
            isinstance(values, list) and values
            for metrics in (evaluation or {}).values()
            if isinstance(metrics, Mapping)
            for values in metrics.values()
        )
        if not has_values:
            raise ValueError(
                f"GBM model '{model_id}' has no saved evaluation history. "
                "Rebuild the model before creating this report."
            )
        return {
            "kind": "gbm_evaluation",
            "title": str(title),
            "data": {
                "manifest": detail["manifest"],
                "parameters": detail["parameters"],
                "metric": detail["metric"],
                "evaluation": evaluation,
            },
            "metadata": {
                "model_id": model_id,
                "model_folder": str(store.model_dir(model_id).resolve()),
            },
        }
    finally:
        dataset.con.close()


def write_echarts_report(
    charts: Iterable[Mapping[str, Any]],
    output_path: str | Path,
    *,
    title: str,
    metadata: Mapping[str, Any] | None = None,
    chart_height: int = DEFAULT_REPORT_CHART_HEIGHT,
) -> Path:
    """Write charts to one portable HTML document and return its path."""

    chart_list = [dict(chart) for chart in charts]
    if not chart_list:
        raise ValueError("Choose at least one chart for the report")
    if any(chart.get("kind") != "line_bar" for chart in chart_list):
        raise ValueError("This report writer currently supports Line/Bar charts only")
    chart_height = int(chart_height)
    if chart_height < 200:
        raise ValueError("chart_height must be at least 200 pixels")

    output = Path(output_path).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    package_root = Path(__file__).resolve().parent
    echarts_source = (package_root / "static" / "vendor" / "echarts" / "echarts.min.js").read_text(encoding="utf-8")
    renderer_source = (package_root / "static" / "app" / "line-bar-chart.js").read_text(encoding="utf-8")
    run_time = datetime.now().astimezone()
    timezone_name = run_time.tzname() or run_time.strftime("%z")
    generated_at = f"{run_time.day} {run_time.strftime('%b %Y, %H:%M')} {timezone_name}"
    report_metadata = {**dict(metadata or {}), "time run": generated_at, "Lucidum version": __version__}
    payload = {
        "title": str(title),
        "metadata": report_metadata,
        "charts": chart_list,
    }
    document = _report_document(
        title=str(title),
        report_metadata=report_metadata,
        charts=chart_list,
        payload=payload,
        echarts_source=echarts_source,
        renderer_source=renderer_source,
        chart_height=chart_height,
    )
    temporary = output.with_name(f".{output.name}.tmp-{uuid4().hex}")
    try:
        temporary.write_text(document, encoding="utf-8")
        temporary.replace(output)
    finally:
        if temporary.exists():
            temporary.unlink()
    return output


def write_gbm_summary_report(
    output_path: str | Path,
    *,
    title: str,
    performance: Mapping[str, Any],
    feature_importance: Mapping[str, Any],
    parameters: Mapping[str, Any],
    evaluation_chart: Mapping[str, Any],
    metadata: Mapping[str, Any] | None = None,
    chart_height: int = DEFAULT_REPORT_CHART_HEIGHT,
) -> Path:
    """Write a portable GBM model-summary HTML document."""

    if evaluation_chart.get("kind") != "gbm_evaluation":
        raise ValueError("evaluation_chart must come from gbm_evaluation_chart()")
    performance_rows = list(performance.get("rows") or [])
    importance_rows = list(feature_importance.get("rows") or [])
    if not performance_rows:
        raise ValueError("The GBM summary needs performance rows")
    if not importance_rows:
        raise ValueError("The GBM summary needs feature importance rows")
    chart_height = int(chart_height)
    if chart_height < 200:
        raise ValueError("chart_height must be at least 200 pixels")

    output = Path(output_path).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    package_root = Path(__file__).resolve().parent
    echarts_source = (package_root / "static" / "vendor" / "echarts" / "echarts.min.js").read_text(
        encoding="utf-8"
    )
    renderer_source = (
        package_root / "static" / "app" / "gbm-evaluation-chart-options.js"
    ).read_text(encoding="utf-8")
    report_metadata = _report_metadata(metadata)
    payload = {
        "title": str(title),
        "metadata": report_metadata,
        "performance": dict(performance),
        "feature_importance": dict(feature_importance),
        "parameters": dict(parameters),
        "evaluation_chart": dict(evaluation_chart),
    }
    document = _gbm_summary_document(
        title=str(title),
        report_metadata=report_metadata,
        payload=payload,
        echarts_source=echarts_source,
        renderer_source=renderer_source,
        chart_height=chart_height,
    )
    temporary = output.with_name(f".{output.name}.tmp-{uuid4().hex}")
    try:
        temporary.write_text(document, encoding="utf-8")
        temporary.replace(output)
    finally:
        if temporary.exists():
            temporary.unlink()
    return output


def write_glm_summary_report(
    output_path: str | Path,
    *,
    title: str,
    dataset_path: str | Path,
    model_id: str,
    kpi_spec_path: str | Path,
    tabulation_export: Mapping[str, Any],
    sample_column: str = "SAMPLE",
    training_value: str = "training",
    test_value: str = "test",
    validation_value: str = "validation",
    model_folder: str | Path | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> Path:
    """Write a portable GLM summary using fitted and tabulated model artifacts."""

    from py_lucidum.core.kpis import load_kpis
    from py_lucidum.tools.glm.store import GlmModelStore

    path = Path(dataset_path).expanduser().resolve()
    dataset = Dataset(path)
    try:
        store = GlmModelStore(
            path,
            dataset=dataset,
            model_root=_explicit_model_root(model_folder, model_id, "GLM"),
        )
        manifest = store.manifest(model_id)
        model_folder = store.model_dir(model_id).resolve()
        scoring_path = store.artifact_path(model_id, "tabulated_predictions").resolve()
        if not scoring_path.is_file():
            raise ValueError("Build and score GLM tabulations before writing the summary report.")
        workbook_path = Path(tabulation_export.get("path") or "").expanduser().resolve()
        index_summary = tabulation_export.get("index")
        if not workbook_path.is_file() or not isinstance(index_summary, Mapping):
            raise ValueError("tabulation_export must come from export_glm_tabulations().")

        estimator_path = store.artifact_path(model_id, "estimator")
        if not estimator_path.is_file():
            raise ValueError(f"GLM estimator.pkl is unavailable for model {model_id}")
        with estimator_path.open("rb") as handle:
            estimator = pickle.load(handle)

        response = str(manifest.get("response_column") or "").strip()
        denominator = str(manifest.get("denominator_column") or "").strip()
        resolved_sample_column = str(sample_column).strip()
        sample_values = {
            "training": str(training_value).strip(),
            "test": str(test_value).strip(),
            "validation": str(validation_value).strip(),
        }
        if not resolved_sample_column:
            raise ValueError("sample_column must not be blank")
        if any(not value for value in sample_values.values()):
            raise ValueError("Training, Test, and Validation sample values must not be blank")
        if len({value.lower() for value in sample_values.values()}) != 3:
            raise ValueError("Training, Test, and Validation sample values must be distinct")
        _require_column(dataset.column_map(), resolved_sample_column, "SAMPLE")
        kpi = _glm_summary_kpi(load_kpis(kpi_spec_path), response, denominator, Path(kpi_spec_path))
        performance = _glm_performance(
            dataset,
            store.artifact_path(model_id, "predictions"),
            response=response,
            denominator=denominator,
            sample_column=resolved_sample_column,
            sample_values=sample_values,
            estimator=estimator,
            kpi=kpi,
        )
        coefficients = _glm_coefficients(store.read_parquet_records(store.artifact_path(model_id, "coefficients")))
        actual_link = _glm_link_name(estimator)
        report_metadata = {
            "source parquet": path,
            "model": model_folder,
            "tabulated scores": scoring_path,
            "response": response,
            "weight": denominator or "None",
            "expected": "glm_prediction",
            "SAMPLE_ROWS": list(sample_values.values()),
            "model label": manifest.get("label") or model_id,
            "family / link": _glm_family_link_label(manifest, estimator, actual_link),
            **dict(metadata or {}),
        }
    finally:
        dataset.con.close()

    report_metadata = _report_metadata(report_metadata)
    payload = {
        "title": str(title),
        "metadata": report_metadata,
        "performance": performance,
        "coefficients": coefficients,
        "tabulations": {
            "path": str(workbook_path),
            "href": _file_uri(workbook_path),
            "scale": str(tabulation_export.get("scale") or ""),
            "columns": list(index_summary.get("columns") or []),
            "rows": list(index_summary.get("rows") or []),
        },
    }
    output = Path(output_path).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    document = _glm_summary_document(
        title=str(title),
        report_metadata=report_metadata,
        payload=payload,
    )
    temporary = output.with_name(f".{output.name}.tmp-{uuid4().hex}")
    try:
        temporary.write_text(document, encoding="utf-8")
        temporary.replace(output)
    finally:
        if temporary.exists():
            temporary.unlink()
    return output


def report_filename(dataset_path: str | Path, model_type: str, report_name: str) -> str:
    """Return a stable, readable HTML filename for an external report."""

    dataset = _safe_name(Path(dataset_path).stem)
    model = _safe_name(model_type)
    report = _safe_name(report_name)
    return f"{dataset}_external_{model}_{report}.html"


def _register_report_sources(
    dataset: Dataset,
    *,
    expected_source: str,
    model_id: str,
    partial_dependence: str,
    model_folder: str | Path | None,
) -> str:
    source_id = "dataset"
    needs_glm = expected_source == "glm" or partial_dependence == "glm"
    needs_gbm = expected_source == "gbm" or partial_dependence == "shap"
    if needs_glm:
        from py_lucidum.tools.glm.store import GlmModelStore, GlmSourceProvider

        store = GlmModelStore(
            dataset.path,
            dataset=dataset,
            model_root=_explicit_model_root(model_folder, model_id, "GLM"),
        )
        store.manifest(model_id)
        if not store.artifact_path(model_id, "predictions").exists():
            raise ValueError(f"GLM predictions are unavailable for model {model_id}")
        if partial_dependence == "glm" and not store.artifact_path(model_id, "estimator").exists():
            raise ValueError(f"GLM estimator.pkl is unavailable for model {model_id}")
        dataset.register_data_source_provider(GlmSourceProvider(store))
        if expected_source == "glm":
            source_id = store.source_id(model_id)
    if needs_gbm:
        from py_lucidum.tools.gbm.store import GbmModelStore, GbmSourceProvider

        store = GbmModelStore(
            dataset.path,
            dataset=dataset,
            model_root=_explicit_model_root(model_folder, model_id, "GBM"),
        )
        store.manifest(model_id)
        if not store.artifact_path(model_id, "predictions").exists():
            raise ValueError(f"GBM predictions are unavailable for model {model_id}")
        if partial_dependence == "shap" and not store.artifact_path(model_id, "shap_long").exists():
            raise ValueError(f"GBM SHAP values are unavailable for model {model_id}")
        dataset.register_data_source_provider(GbmSourceProvider(store))
        if expected_source == "gbm":
            source_id = store.source_id(model_id, "predictions")
    return source_id


def _explicit_model_root(
    model_folder: str | Path | None,
    model_id: str,
    model_type: str,
) -> Path | None:
    if model_folder is None:
        return None
    folder = Path(model_folder).expanduser().resolve()
    if folder.name != str(model_id or "").strip():
        raise ValueError(f"model_folder must be the folder for {model_type} model {model_id!r}")
    if not folder.is_dir():
        raise ValueError(f"{model_type} model folder does not exist: {folder}")
    return folder.parent


def _sample_filter(
    dataset: Dataset,
    sample_column: str,
    sample_values: str | Sequence[str],
) -> tuple[str, list[str], int]:
    column = quote_ident(sample_column)
    relation = dataset.relation_sql()
    raw_rows = dataset.con.execute(
        f"""
SELECT LOWER(TRIM(CAST({column} AS VARCHAR))) AS sample_value, COUNT(*) AS row_count
FROM {relation}
WHERE {column} IS NOT NULL
GROUP BY 1
ORDER BY 1
"""
    ).fetchall()
    available = {str(value): int(count) for value, count in raw_rows if str(value or "").strip()}
    if isinstance(sample_values, str) and sample_values.strip().lower() == "all":
        return "", list(available), dataset.row_count()
    if isinstance(sample_values, str):
        requested = [sample_values.strip().lower()]
    else:
        requested = [str(value).strip().lower() for value in sample_values]
    requested = list(dict.fromkeys(value for value in requested if value))
    if not requested:
        raise ValueError("Choose 'all' or at least one SAMPLE value")
    missing = [value for value in requested if value not in available]
    if missing:
        raise ValueError(f"SAMPLE values are unavailable: {', '.join(missing)}")
    values_sql = ", ".join(sql_literal(value) for value in requested)
    filter_sql = f"LOWER(TRIM(CAST({column} AS VARCHAR))) IN ({values_sql})"
    return filter_sql, requested, sum(available[value] for value in requested)


def _normalise_chart_controls(
    controls: Mapping[str, Any] | None,
    *,
    transform: str | None,
) -> dict[str, Any]:
    raw = dict(controls or {})
    known = {
        "banding",
        "quantiles",
        "low_weights",
        "missings",
        "labels",
        "sort",
        "transform",
        "sigma",
        "date_bucket",
        "empty_periods",
        "base",
    }
    unknown = sorted(set(raw) - known)
    if unknown:
        raise ValueError(f"Unknown chart controls: {', '.join(unknown)}")
    banding = _non_negative_number(raw.get("banding", 0), "banding")
    quantiles = _non_negative_integer(raw.get("quantiles", 0), "quantiles")
    sigma = _non_negative_integer(raw.get("sigma", 0), "sigma")
    if sigma not in SIGMA_VALUES:
        raise ValueError("sigma must be one of 0, 1, 2, or 5")
    low_weights = str(raw.get("low_weights", "0") or "0").strip()
    if low_weights not in {"0", "10", "100", "0.1%", "1%"}:
        raise ValueError("low_weights must be 0, 10, 100, 0.1%, or 1%")
    return {
        "banding": banding,
        "quantiles": quantiles,
        "low_weights": low_weights,
        "missings": _choice(raw.get("missings", "show"), MISSINGS_VALUES, "missings"),
        "labels": _choice(raw.get("labels", "none"), LABEL_VALUES, "labels"),
        "sort": _choice(raw.get("sort", "alpha"), SORT_VALUES, "sort"),
        "transform": _choice(transform if transform is not None else raw.get("transform", "none"), TRANSFORM_VALUES, "transform"),
        "sigma": sigma,
        "date_bucket": _choice(raw.get("date_bucket", "none"), DATE_BUCKET_VALUES, "date bucket"),
        "empty_periods": _choice(raw.get("empty_periods", "show"), EMPTY_PERIOD_VALUES, "empty periods"),
        "base": str(raw.get("base") or "").strip(),
    }


def _choice(value: Any, choices: set[str], label: str) -> str:
    text = str(value or "").strip().lower()
    if text not in choices:
        raise ValueError(f"Choose a valid {label}: {', '.join(sorted(choices))}")
    return text


def _non_negative_number(value: Any, label: str) -> int | float:
    if value is None or value == "":
        return 0
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be a non-negative number") from exc
    if not math.isfinite(number) or number < 0:
        raise ValueError(f"{label} must be a non-negative number")
    return int(number) if number.is_integer() else number


def _non_negative_integer(value: Any, label: str) -> int:
    number = _non_negative_number(value, label)
    if int(number) != number:
        raise ValueError(f"{label} must be a non-negative integer")
    return int(number)


def _require_column(columns: Mapping[str, Any], name: str, label: str) -> None:
    if name not in columns:
        raise ValueError(f"{label} column is missing: {name}")


def _require_numeric_column(columns: Mapping[str, Any], name: str, label: str) -> None:
    _require_column(columns, name, label)
    if not is_numeric_kind(columns[name].kind):
        raise ValueError(f"{label} column must be numeric: {name}")


def _feature_spec_payload(feature_spec: str | Path | Mapping[str, Any] | None) -> dict[str, Any]:
    if feature_spec is None:
        return {"rows": [], "scenarios": []}
    if isinstance(feature_spec, Mapping):
        return dict(feature_spec)
    return load_features(feature_spec)


def _feature_base(feature_spec: Mapping[str, Any], feature: str) -> str:
    for row in feature_spec.get("rows", []):
        if isinstance(row, Mapping) and str(row.get("feature") or "") == feature:
            return str(row.get("base") or "").strip()
    return ""


def _partial_dependence_overlay(payload: Mapping[str, Any], kind: str) -> Mapping[str, Any]:
    partial = payload.get("partial_dependence")
    if not isinstance(partial, Mapping):
        return {}
    overlays = partial.get("overlays")
    if isinstance(overlays, Mapping) and isinstance(overlays.get(kind), Mapping):
        return overlays[kind]
    return partial if partial.get("mode") == kind else {}


def _require_requested_overlay(
    payload: Mapping[str, Any],
    *,
    partial_dependence: str,
    model_id: str,
    feature: str,
) -> None:
    if partial_dependence == "none":
        return
    overlay = _partial_dependence_overlay(payload, partial_dependence)
    warnings = [str(value) for value in overlay.get("warnings", []) if value] if isinstance(overlay, Mapping) else []
    if not overlay or str(overlay.get("model_id") or "") != model_id or not overlay.get("rows"):
        detail = f": {warnings[0]}" if warnings else ""
        raise ValueError(f"{feature} could not produce the requested {partial_dependence.upper()} overlay{detail}")


def _safe_name(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9]+", "_", str(value or "").strip().lower()).strip("_")
    return cleaned or "report"


def _json_for_script(payload: Any) -> str:
    return (
        json.dumps(payload, ensure_ascii=False, separators=(",", ":"), default=str)
        .replace("&", "\\u0026")
        .replace("<", "\\u003c")
        .replace(">", "\\u003e")
    )


def _report_document(
    *,
    title: str,
    report_metadata: Mapping[str, Any],
    charts: Sequence[Mapping[str, Any]],
    payload: Mapping[str, Any],
    echarts_source: str,
    renderer_source: str,
    chart_height: int,
) -> str:
    full_width_metadata = {"source parquet", "model"}
    footer_metadata = {"importance measure"}
    provenance_html = "".join(
        f"<div><dt>{html.escape(_metadata_label(key))}</dt><dd>{html.escape(_display_value(value))}</dd></div>"
        for key, value in report_metadata.items()
        if str(key).strip().lower() in full_width_metadata
        and value is not None and value != "" and value != []
    )
    metadata_html = "".join(
        f"<div><dt>{html.escape(_metadata_label(key))}</dt><dd>{html.escape(_display_value(value))}</dd></div>"
        for key, value in report_metadata.items()
        if str(key).strip().lower() not in full_width_metadata | footer_metadata
        and value is not None and value != "" and value != []
    )
    footer_html = "".join(
        f"<div><dt>{html.escape(_metadata_label(key))}</dt><dd>{html.escape(_display_value(value))}</dd></div>"
        for key, value in report_metadata.items()
        if str(key).strip().lower() in footer_metadata
        and value is not None and value != "" and value != []
    )
    footer_section = f'<dl class="report-metadata-footer">{footer_html}</dl>' if footer_html else ""
    cards = []
    for index, chart in enumerate(charts):
        chart_metadata = chart.get("metadata") if isinstance(chart.get("metadata"), Mapping) else {}
        details = [
            _sample_label(chart_metadata.get("sample_values")),
            _base_label(chart_metadata),
        ]
        details = [detail for detail in details if detail]
        cards.append(
            "".join(
                [
                    f'<section class="chart-card" data-chart-index="{index}">',
                    f"<h2>{html.escape(str(chart.get('title') or chart_metadata.get('feature') or f'Chart {index + 1}'))}</h2>",
                    f'<p class="chart-detail">{html.escape(" · ".join(details))}</p>' if details else "",
                    f'<div id="chart-{index}" class="report-chart" role="img" aria-label="{html.escape(str(chart.get("title") or "Line and bar chart"))}"></div>',
                    '<p class="chart-warning" hidden></p>',
                    "</section>",
                ]
            )
        )
    safe_echarts = echarts_source.replace("</script", "<\\/script")
    safe_renderer = renderer_source.replace("</script", "<\\/script")
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>{html.escape(title)}</title>
  <style>
    :root {{ color-scheme: light; --page: #f5f7fa; --panel: #ffffff; --text: #243447; --muted: #64748b; --line: #d9e0e8; }}
    * {{ box-sizing: border-box; }}
    body {{ margin: 0; background: var(--page); color: var(--text); font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }}
    main {{ width: min(1500px, 100%); margin: 0 auto; padding: 24px; }}
    .report-header, .chart-card {{ background: var(--panel); border: 1px solid var(--line); border-radius: 10px; box-shadow: 0 1px 3px rgba(15, 23, 42, 0.06); }}
    .report-header {{ padding: 20px 24px; margin-bottom: 20px; }}
    h1 {{ margin: 0 0 16px; font-size: 24px; }}
    h2 {{ margin: 0; padding: 16px 20px 0; font-size: 18px; }}
    dl {{ margin: 0; }}
    dl div {{ min-width: 0; }}
    .report-provenance {{ display: grid; gap: 10px; margin-bottom: 14px; }}
    .report-provenance div {{ width: 100%; }}
    .report-metadata-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 10px 24px; padding-top: 14px; border-top: 1px solid var(--line); }}
    .report-metadata-footer {{ margin-top: 14px; padding-top: 14px; border-top: 1px solid var(--line); }}
    dt {{ color: var(--muted); font-size: 11px; font-weight: 700; letter-spacing: .04em; text-transform: uppercase; }}
    dd {{ margin: 2px 0 0; overflow-wrap: anywhere; font-size: 13px; }}
    .chart-card {{ margin: 0 0 20px; overflow: hidden; }}
    .chart-detail {{ min-height: 18px; margin: 4px 20px 0; color: var(--muted); font-size: 12px; }}
    .report-chart {{ width: 100%; height: {chart_height}px; }}
    .chart-warning {{ margin: -6px 20px 16px; color: #9a6700; font-size: 12px; }}
    @media (max-width: 700px) {{ main {{ padding: 10px; }} }}
    @media print {{ body {{ background: white; }} main {{ width: 100%; padding: 0; }} .report-header, .chart-card {{ break-inside: avoid; border-color: #bbb; box-shadow: none; }} }}
  </style>
  <script>{safe_echarts}</script>
</head>
<body>
  <main>
    <header class="report-header">
      <h1>{html.escape(title)}</h1>
      <dl class="report-provenance">{provenance_html}</dl>
      <dl class="report-metadata-grid">{metadata_html}</dl>
      {footer_section}
    </header>
    {''.join(cards)}
  </main>
  <script id="lucidum-report-data" type="application/json">{_json_for_script(payload)}</script>
  <script type="module">
{safe_renderer}
    const report = JSON.parse(document.getElementById("lucidum-report-data").textContent);
    const instances = [];
    report.charts.forEach((chartSpec, index) => {{
      const target = document.getElementById(`chart-${{index}}`);
      const instance = echarts.init(target);
      const rendered = lineBarChartOption(chartSpec.data, {{
        ...chartSpec.presentation,
        chartWidth: target.clientWidth,
        chartHeight: target.clientHeight,
      }});
      instance.setOption(rendered.option, true);
      bindLineBarChartInteractions(instance, chartSpec.data, chartSpec.presentation);
      const warning = target.closest(".chart-card").querySelector(".chart-warning");
      const messages = [...(chartSpec.data.warnings || []), ...(rendered.messages || [])].filter(Boolean);
      if (messages.length) {{ warning.textContent = messages.join(" "); warning.hidden = false; }}
      instances.push(instance);
    }});
    const resize = () => instances.forEach((instance) => instance.resize());
    if (typeof ResizeObserver === "function") new ResizeObserver(resize).observe(document.querySelector("main"));
    window.addEventListener("resize", resize);
  </script>
</body>
</html>
"""


def _gbm_summary_document(
    *,
    title: str,
    report_metadata: Mapping[str, Any],
    payload: Mapping[str, Any],
    echarts_source: str,
    renderer_source: str,
    chart_height: int,
) -> str:
    full_width_metadata = {"source parquet", "model"}
    provenance_html = _metadata_items(report_metadata, include=full_width_metadata)
    metadata_html = _metadata_items(report_metadata, exclude=full_width_metadata)
    performance = payload["performance"]
    importance = payload["feature_importance"]
    parameters = payload["parameters"]
    performance_table = _summary_table_html(
        list(performance.get("columns") or []),
        list(performance.get("rows") or []),
        table_class="performance-table",
    )
    importance_table = _summary_table_html(
        list(importance.get("columns") or []),
        list(importance.get("rows") or []),
        table_class="importance-table",
    )
    parameter_table = _summary_table_html(
        [{"key": "parameter", "label": "Parameter"}, {"key": "value", "label": "Value"}],
        [
            {"parameter": str(name), "value": _parameter_display(value)}
            for name, value in parameters.items()
        ],
        table_class="parameter-table",
    )
    importance_measure = str(importance.get("measure") or "")
    best_iteration = performance.get("best_iteration")
    metric = str(performance.get("metric") or "")
    metric_detail = ""
    if metric:
        metric_detail = f"Metric: {metric}"
        if best_iteration:
            metric_detail += f" at best iteration {int(best_iteration):,}"
    safe_echarts = echarts_source.replace("</script", "<\\/script")
    safe_renderer = renderer_source.replace("</script", "<\\/script")
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>{html.escape(title)}</title>
  <style>
    :root {{ color-scheme: light; --page: #f5f7fa; --panel: #ffffff; --text: #243447; --muted: #64748b; --line: #d9e0e8; --accent: #2563eb; }}
    * {{ box-sizing: border-box; }}
    body {{ margin: 0; background: var(--page); color: var(--text); font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }}
    main {{ width: min(1500px, 100%); margin: 0 auto; padding: 24px; }}
    .report-header, .summary-card {{ background: var(--panel); border: 1px solid var(--line); border-radius: 10px; box-shadow: 0 1px 3px rgba(15, 23, 42, 0.06); }}
    .report-header {{ padding: 20px 24px; margin-bottom: 20px; }}
    h1 {{ margin: 0 0 16px; font-size: 24px; }}
    h2 {{ margin: 0; padding: 18px 20px 0; font-size: 18px; }}
    dl {{ margin: 0; }}
    dl div {{ min-width: 0; }}
    .report-provenance {{ display: grid; gap: 10px; margin-bottom: 14px; }}
    .report-metadata-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 10px 24px; padding-top: 14px; border-top: 1px solid var(--line); }}
    dt {{ color: var(--muted); font-size: 11px; font-weight: 700; letter-spacing: .04em; text-transform: uppercase; }}
    dd {{ margin: 2px 0 0; overflow-wrap: anywhere; font-size: 13px; }}
    .summary-card {{ margin: 0 0 20px; overflow: hidden; }}
    .section-detail {{ margin: 5px 20px 0; color: var(--muted); font-size: 12px; }}
    .table-wrap {{ padding: 16px 20px 20px; overflow-x: auto; }}
    .summary-table {{ width: 100%; border-collapse: collapse; font-size: 13px; font-variant-numeric: tabular-nums; }}
    .summary-table th {{ padding: 4px 12px; border-bottom: 2px solid var(--line); color: var(--muted); font-size: 11px; letter-spacing: .04em; text-align: right; text-transform: uppercase; white-space: nowrap; }}
    .summary-table td {{ padding: 4px 12px; border-bottom: 1px solid var(--line); text-align: right; white-space: nowrap; }}
    .summary-table tbody tr:last-child td {{ border-bottom: 0; }}
    .summary-table th:first-child, .summary-table td:first-child {{ text-align: left; }}
    .importance-table th:nth-child(2), .importance-table td:nth-child(2), .parameter-table th, .parameter-table td {{ text-align: left; }}
    .importance-table td:nth-child(2) {{ font-weight: 600; }}
    .parameter-table td:first-child {{ width: 34%; color: var(--muted); font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; }}
    .parameter-table td:last-child {{ white-space: normal; overflow-wrap: anywhere; font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; }}
    .report-chart {{ width: 100%; height: {chart_height}px; }}
    .chart-warning {{ margin: 0 20px 16px; color: #9a6700; font-size: 12px; }}
    @media (max-width: 700px) {{ main {{ padding: 10px; }} .table-wrap {{ padding-inline: 10px; }} }}
    @media print {{ body {{ background: white; }} main {{ width: 100%; padding: 0; }} .report-header, .summary-card {{ break-inside: avoid; border-color: #bbb; box-shadow: none; }} }}
  </style>
  <script>{safe_echarts}</script>
</head>
<body>
  <main>
    <header class="report-header">
      <h1>{html.escape(title)}</h1>
      <dl class="report-provenance">{provenance_html}</dl>
      <dl class="report-metadata-grid">{metadata_html}</dl>
    </header>
    <section class="summary-card" data-summary-section="performance">
      <h2>Model performance</h2>
      <p class="section-detail">{html.escape(metric_detail)}</p>
      <div class="table-wrap">{performance_table}</div>
    </section>
    <section class="summary-card" data-summary-section="feature-importance">
      <h2>Feature importance</h2>
      <p class="section-detail">Importance measure: {html.escape(importance_measure)}</p>
      <div class="table-wrap">{importance_table}</div>
    </section>
    <section class="summary-card" data-summary-section="parameters">
      <h2>Model parameters</h2>
      <div class="table-wrap">{parameter_table}</div>
    </section>
    <section class="summary-card" data-summary-section="evaluation">
      <h2>{html.escape(str(payload['evaluation_chart'].get('title') or 'Model evaluation'))}</h2>
      <div id="gbm-summary-evaluation-chart" class="report-chart" role="img" aria-label="GBM model evaluation chart"></div>
      <p class="chart-warning" hidden></p>
    </section>
  </main>
  <script id="lucidum-report-data" type="application/json">{_json_for_script(payload)}</script>
  <script type="module">
{safe_renderer}
    const report = JSON.parse(document.getElementById("lucidum-report-data").textContent);
    const target = document.getElementById("gbm-summary-evaluation-chart");
    const option = gbmEvaluationChartOption(report.evaluation_chart.data, {{ viewMode: "all" }});
    if (option) {{
      const chart = echarts.init(target);
      chart.setOption(option, true);
      const resize = () => chart.resize();
      if (typeof ResizeObserver === "function") new ResizeObserver(resize).observe(document.querySelector("main"));
      window.addEventListener("resize", resize);
    }} else {{
      const warning = target.closest(".summary-card").querySelector(".chart-warning");
      warning.textContent = "No evaluation history is available.";
      warning.hidden = false;
    }}
  </script>
</body>
</html>
"""


def _glm_performance(
    dataset: Dataset,
    prediction_path: Path,
    *,
    response: str,
    denominator: str,
    sample_column: str = "SAMPLE",
    sample_values: Mapping[str, str] | None = None,
    estimator: Any,
    kpi: Mapping[str, Any],
) -> dict[str, Any]:
    if not prediction_path.is_file():
        raise ValueError("Fitted GLM predictions are unavailable.")
    actual = quote_ident(response)
    sample = quote_ident(sample_column)
    selected_samples = {
        "training": "training",
        "test": "test",
        "validation": "validation",
        **dict(sample_values or {}),
    }
    weight = quote_ident(denominator) if denominator else ""
    weight_projection = f", TRY_CAST(source.{weight} AS DOUBLE) AS report_weight" if denominator else ", 1.0 AS report_weight"
    valid_weight = f"AND isfinite(TRY_CAST(source.{weight} AS DOUBLE)) AND TRY_CAST(source.{weight} AS DOUBLE) > 0" if denominator else ""
    actual_value = f"TRY_CAST(source.{actual} AS DOUBLE) / TRY_CAST(source.{weight} AS DOUBLE)" if denominator else f"TRY_CAST(source.{actual} AS DOUBLE)"
    prediction_value = f"TRY_CAST(prediction.glm_prediction AS DOUBLE) / TRY_CAST(source.{weight} AS DOUBLE)" if denominator else "TRY_CAST(prediction.glm_prediction AS DOUBLE)"
    query = f"""
WITH source AS (
  SELECT ROW_NUMBER() OVER ()::BIGINT AS __lucidum_row_id, *
  FROM {dataset.relation_sql()}
)
SELECT
  LOWER(TRIM(CAST(source.{sample} AS VARCHAR))) AS sample_value,
  {actual_value} AS actual_value,
  {prediction_value} AS prediction_value
  {weight_projection}
FROM source
INNER JOIN read_parquet({sql_literal(str(prediction_path))}) prediction
  USING (__lucidum_row_id)
WHERE isfinite(TRY_CAST(source.{actual} AS DOUBLE))
  AND isfinite(TRY_CAST(prediction.glm_prediction AS DOUBLE))
  {valid_weight}
"""
    grouped: dict[str, list[tuple[float, float, float]]] = {}
    for sample_value, actual_number, prediction_number, weight_number in dataset.con.execute(query).fetchall():
        if not all(_finite_number(value) is not None for value in (actual_number, prediction_number, weight_number)):
            continue
        grouped.setdefault(str(sample_value), []).append(
            (float(actual_number), float(prediction_number), float(weight_number))
        )

    import numpy as np

    is_binomial = "binomial" in type(estimator.family_instance).__name__.casefold()
    rows = []
    for label, role in (("Training", "training"), ("Test", "test"), ("Validation", "validation")):
        sample_value = str(selected_samples[role]).strip().lower()
        values = grouped.get(sample_value)
        if not values:
            raise ValueError(
                f"The {label} {sample_column} value has no eligible fitted predictions: "
                f"{selected_samples[role]}"
            )
        array = np.asarray(values, dtype=float)
        y = array[:, 0]
        prediction = array[:, 1]
        weights = array[:, 2]
        actual_summary = float(np.average(y, weights=weights))
        prediction_summary = float(np.average(prediction, weights=weights))
        deviance = _safe_glm_metric(estimator.family_instance.deviance, y, prediction, weights)
        null_prediction = np.full(len(y), actual_summary, dtype=float)
        null_deviance = _safe_glm_metric(estimator.family_instance.deviance, y, null_prediction, weights)
        deviance_explained = (
            None
            if deviance is None or null_deviance is None or null_deviance == 0
            else 1.0 - deviance / null_deviance
        )
        raw = {
            "row_count": len(values),
            "weight": float(weights.sum()),
            "actual": actual_summary,
            "prediction": prediction_summary,
            "deviance": deviance,
            "deviance_explained": deviance_explained,
        }
        if is_binomial:
            auc = _weighted_auc(y, prediction, weights)
            clipped = np.clip(prediction, 1e-15, 1.0 - 1e-15)
            log_loss = float(np.average(-(y * np.log(clipped) + (1.0 - y) * np.log1p(-clipped)), weights=weights))
            raw.update({"auc": auc, "gini": None if auc is None else 2.0 * auc - 1.0, "log_loss": log_loss})
        else:
            error = prediction - y
            raw.update(
                {
                    "rmse": float(math.sqrt(np.average(error * error, weights=weights))),
                    "mae": float(np.average(np.abs(error), weights=weights)),
                }
            )
        row = {
            "sample": label,
            "rows": f"{len(values):,}",
            "weight": _format_weight(raw["weight"]) if denominator else None,
            "actual": _format_kpi(raw["actual"], kpi),
            "prediction": _format_kpi(raw["prediction"], kpi),
            "deviance": _format_compact(raw["deviance"]),
            "deviance_explained": _format_percent(raw["deviance_explained"]),
            "raw": raw,
        }
        if is_binomial:
            row.update(
                {
                    "auc": _format_percent(raw["auc"]),
                    "gini": _format_percent(raw["gini"]),
                    "log_loss": _format_compact(raw["log_loss"]),
                }
            )
        else:
            row.update({"rmse": _format_kpi(raw["rmse"], kpi), "mae": _format_kpi(raw["mae"], kpi)})
        rows.append(row)

    columns = [
        {"key": "sample", "label": "Sample"},
        {"key": "rows", "label": "Number of rows"},
    ]
    if denominator:
        columns.append({"key": "weight", "label": f"Sum of {denominator}"})
    columns.extend(
        [
            {"key": "actual", "label": "Actual response"},
            {"key": "prediction", "label": "Model prediction"},
            {"key": "deviance", "label": "Deviance"},
            {"key": "deviance_explained", "label": "Deviance explained"},
        ]
    )
    if is_binomial:
        columns.extend(
            [
                {"key": "auc", "label": "AUC"},
                {"key": "gini", "label": "Gini"},
                {"key": "log_loss", "label": "Log loss"},
            ]
        )
    else:
        columns.extend([{"key": "rmse", "label": "RMSE"}, {"key": "mae", "label": "MAE"}])
    return {
        "columns": columns,
        "rows": rows,
        "family": type(estimator.family_instance).__name__,
        "prediction_source": "glm_prediction",
        "kpi": dict(kpi),
    }


def _safe_glm_metric(metric: Any, y: Any, prediction: Any, weights: Any) -> float | None:
    try:
        return _finite_number(metric(y, prediction, sample_weight=weights))
    except Exception:
        return None


def _weighted_auc(y: Any, prediction: Any, weights: Any) -> float | None:
    import numpy as np

    if len(y) == 0 or np.any((y < 0) | (y > 1)):
        return None
    positive = np.asarray(weights, dtype=float) * np.asarray(y, dtype=float)
    negative = np.asarray(weights, dtype=float) * (1.0 - np.asarray(y, dtype=float))
    total_positive = float(positive.sum())
    total_negative = float(negative.sum())
    if total_positive <= 0 or total_negative <= 0:
        return None
    order = np.argsort(np.asarray(prediction, dtype=float), kind="mergesort")
    scores = np.asarray(prediction, dtype=float)[order]
    positive = positive[order]
    negative = negative[order]
    concordance = 0.0
    prior_negative = 0.0
    start = 0
    while start < len(scores):
        end = start + 1
        while end < len(scores) and scores[end] == scores[start]:
            end += 1
        group_positive = float(positive[start:end].sum())
        group_negative = float(negative[start:end].sum())
        concordance += group_positive * (prior_negative + 0.5 * group_negative)
        prior_negative += group_negative
        start = end
    return concordance / (total_positive * total_negative)


def _glm_summary_kpi(
    kpis: Sequence[Mapping[str, Any]],
    response: str,
    denominator: str,
    path: Path,
) -> dict[str, Any]:
    expected_denominator = denominator or "__none__"
    for kpi in kpis:
        if str(kpi.get("actual")) == response and str(kpi.get("denominator")) == expected_denominator:
            return dict(kpi)
    weight = denominator or "Average row value"
    raise ValueError(f"KPI specification {path.resolve()} has no row for Actual {response!r} and Weight {weight!r}")


def _glm_coefficients(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    rows = []
    for index, record in enumerate(records, start=1):
        estimate = _finite_number(record.get("estimate"))
        std_error = _finite_number(record.get("std_error", record.get("std.error")))
        p_value = _finite_number(record.get("p_value", record.get("p.value")))
        significance = ""
        if p_value is not None:
            significance = "significance-low" if p_value < 0.01 else "significance-medium" if p_value <= 0.05 else "significance-high"
        rows.append(
            {
                "number": index,
                "term": str(record.get("term") or ""),
                "estimate": _format_glm_number(estimate),
                "std_error": _format_glm_number(std_error),
                "p_value": _format_p_value(p_value),
                "significance": significance,
                "raw": {"estimate": estimate, "std_error": std_error, "p_value": p_value},
            }
        )
    return {
        "columns": [
            {"key": "number", "label": "#"},
            {"key": "term", "label": "term"},
            {"key": "estimate", "label": "estimate"},
            {"key": "std_error", "label": "std.error"},
            {"key": "p_value", "label": "p.value"},
        ],
        "rows": rows,
    }


def _format_glm_number(value: Any) -> str:
    number = _finite_number(value)
    return "--" if number is None else f"{number:,.4f}".rstrip("0").rstrip(".")


def _format_p_value(value: Any) -> str:
    number = _finite_number(value)
    return "--" if number is None else f"{number * 100:.2f}".rstrip("0").rstrip(".") + "%"


def _format_kpi(value: Any, kpi: Mapping[str, Any]) -> str:
    number = _finite_number(value)
    if number is None:
        return "—"
    display = number * 100 if kpi.get("format") == "percent" else number
    decimals = int(kpi.get("decimals") or 0)
    sign = "-" if display < 0 else ""
    formatted = f"{abs(display):,.{decimals}f}"
    if kpi.get("format") == "currency":
        return f"{sign}£{formatted}"
    if kpi.get("format") == "percent":
        return f"{sign}{formatted}%"
    return f"{sign}{formatted}"


def _format_weight(value: Any) -> str:
    number = _finite_number(value)
    if number is None:
        return "—"
    return f"{number:,.0f}" if abs(number) >= 10 or number.is_integer() else _format_compact(number)


def _format_compact(value: Any) -> str:
    number = _finite_number(value)
    if number is None:
        return "—"
    absolute = abs(number)
    decimals = 1 if absolute >= 1000 else 2 if absolute >= 10 else 3 if absolute >= 1 else 4 if absolute >= 0.01 else 6
    return f"{number:,.{decimals}f}".rstrip("0").rstrip(".")


def _format_percent(value: Any) -> str:
    number = _finite_number(value)
    return "—" if number is None else f"{number * 100:.1f}%"


def _finite_number(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _glm_link_name(estimator: Any) -> str:
    names = {
        "IdentityLink": "identity",
        "LogLink": "log",
        "LogitLink": "logit",
        "CloglogLink": "cloglog",
        "TweedieLink": "tweedie",
    }
    class_name = type(estimator.link_instance).__name__
    return names.get(class_name, class_name.removesuffix("Link").casefold())


def _glm_family_link_label(
    manifest: Mapping[str, Any],
    estimator: Any,
    link_name: str | None = None,
) -> str:
    family = str(manifest.get("family") or type(estimator.family_instance).__name__)
    link = link_name or _glm_link_name(estimator)
    fitted_family = estimator.family_instance
    is_tweedie = (
        family.strip().casefold() == "tweedie"
        or type(fitted_family).__name__ == "TweedieDistribution"
    )
    if not is_tweedie:
        return f"{family} / {link}"
    power = _finite_number(manifest.get("family_parameter"))
    if power is None:
        power = _finite_number(getattr(fitted_family, "power", None))
    power_label = _format_compact(power)
    return f"{family} (variance power {power_label}) / {link}"


def _file_uri(path: Any) -> str:
    text = str(path)
    if re.match(r"^[A-Za-z]:[\\/]", text) or text.startswith("\\\\"):
        return PureWindowsPath(text).as_uri()
    return Path(text).expanduser().resolve().as_uri()


def _glm_summary_document(
    *,
    title: str,
    report_metadata: Mapping[str, Any],
    payload: Mapping[str, Any],
) -> str:
    full_width_metadata = {"source parquet", "model", "tabulated scores"}
    provenance_html = _metadata_items(report_metadata, include=full_width_metadata)
    metadata_html = _metadata_items(report_metadata, exclude=full_width_metadata)
    performance = payload["performance"]
    coefficients = payload["coefficients"]
    tabulations = payload["tabulations"]
    performance_table = _summary_table_html(
        list(performance.get("columns") or []),
        list(performance.get("rows") or []),
        table_class="performance-table",
    )
    coefficient_table = _glm_coefficient_table_html(
        list(coefficients.get("columns") or []),
        list(coefficients.get("rows") or []),
    )
    tabulation_table = _glm_tabulation_table_html(
        list(tabulations.get("columns") or []),
        list(tabulations.get("rows") or []),
    )
    workbook_path = str(tabulations.get("path") or "")
    workbook_href = str(tabulations.get("href") or "")
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>{html.escape(title)}</title>
  <style>
    :root {{ color-scheme: light; --page: #f5f7fa; --panel: #ffffff; --text: #243447; --muted: #64748b; --line: #d9e0e8; --accent: #2563eb; }}
    * {{ box-sizing: border-box; }}
    body {{ margin: 0; background: var(--page); color: var(--text); font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }}
    main {{ width: min(1500px, 100%); margin: 0 auto; padding: 24px; }}
    .report-header, .summary-card {{ background: var(--panel); border: 1px solid var(--line); border-radius: 10px; box-shadow: 0 1px 3px rgba(15, 23, 42, 0.06); }}
    .report-header {{ padding: 20px 24px; margin-bottom: 20px; }}
    h1 {{ margin: 0 0 16px; font-size: 24px; }}
    h2 {{ margin: 0; padding: 18px 20px 0; font-size: 18px; }}
    dl {{ margin: 0; }}
    dl div {{ min-width: 0; }}
    .report-provenance {{ display: grid; gap: 10px; margin-bottom: 14px; }}
    .report-metadata-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 10px 24px; padding-top: 14px; border-top: 1px solid var(--line); }}
    dt {{ color: var(--muted); font-size: 11px; font-weight: 700; letter-spacing: .04em; text-transform: uppercase; }}
    dd {{ margin: 2px 0 0; overflow-wrap: anywhere; font-size: 13px; }}
    .summary-card {{ margin: 0 0 20px; overflow: hidden; }}
    .section-detail {{ margin: 5px 20px 0; color: var(--muted); font-size: 12px; overflow-wrap: anywhere; }}
    .section-detail a {{ color: var(--accent); }}
    .table-wrap {{ padding: 16px 20px 20px; overflow-x: auto; }}
    .summary-table {{ width: 100%; border-collapse: collapse; font-size: 13px; font-variant-numeric: tabular-nums; }}
    .summary-table th {{ padding: 4px 12px; border-bottom: 2px solid var(--line); color: var(--muted); font-size: 11px; letter-spacing: .04em; text-align: right; text-transform: uppercase; white-space: nowrap; }}
    .summary-table td {{ padding: 4px 12px; border-bottom: 1px solid var(--line); text-align: right; white-space: nowrap; }}
    .summary-table tbody tr:last-child td {{ border-bottom: 0; }}
    .summary-table th:first-child, .summary-table td:first-child {{ text-align: left; }}
    .coefficient-table th:nth-child(2), .coefficient-table td:nth-child(2), .tabulation-table th:nth-child(2), .tabulation-table td:nth-child(2) {{ text-align: left; }}
    .coefficient-table td:nth-child(2) {{ font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; }}
    .coefficient-table tr.significance-low {{ background: #ecfdf3; }}
    .coefficient-table tr.significance-medium {{ background: #fffbeb; }}
    .coefficient-table tr.significance-high {{ background: #fff1f2; }}
    @media (max-width: 700px) {{ main {{ padding: 10px; }} .table-wrap {{ padding-inline: 10px; }} }}
    @media print {{ body {{ background: white; }} main {{ width: 100%; padding: 0; }} .report-header, .summary-card {{ break-inside: avoid; border-color: #bbb; box-shadow: none; }} }}
  </style>
</head>
<body>
  <main>
    <header class="report-header">
      <h1>{html.escape(title)}</h1>
      <dl class="report-provenance">{provenance_html}</dl>
      <dl class="report-metadata-grid">{metadata_html}</dl>
    </header>
    <section class="summary-card" data-summary-section="performance">
      <h2>Model performance</h2>
      <p class="section-detail">Performance uses fitted <code>glm_prediction</code> values.</p>
      <div class="table-wrap">{performance_table}</div>
    </section>
    <section class="summary-card" data-summary-section="coefficients">
      <h2>Coefficients and p-values</h2>
      <div class="table-wrap">{coefficient_table}</div>
    </section>
    <section class="summary-card" data-summary-section="tabulations">
      <h2>Tabulation summary</h2>
      <p class="section-detail">Workbook: <a href="{html.escape(workbook_href, quote=True)}">{html.escape(workbook_path)}</a></p>
      <div class="table-wrap">{tabulation_table}</div>
    </section>
  </main>
  <script id="lucidum-report-data" type="application/json">{_json_for_script(payload)}</script>
</body>
</html>
"""


def _glm_coefficient_table_html(
    columns: Sequence[Mapping[str, Any]],
    rows: Sequence[Mapping[str, Any]],
) -> str:
    headings = "".join(f'<th scope="col">{html.escape(str(column["label"]))}</th>' for column in columns)
    body = "".join(
        f'<tr class="{html.escape(str(row.get("significance") or ""))}">'
        + "".join(
            f'<td>{html.escape(_display_value(row.get(str(column["key"]), "—")))}</td>'
            for column in columns
        )
        + "</tr>"
        for row in rows
    )
    return f'<table class="summary-table coefficient-table"><thead><tr>{headings}</tr></thead><tbody>{body}</tbody></table>'


def _glm_tabulation_table_html(
    columns: Sequence[Mapping[str, Any]],
    rows: Sequence[Mapping[str, Any]],
) -> str:
    headings = "".join(f'<th scope="col">{html.escape(str(column["label"]))}</th>' for column in columns)

    def display_cell(row: Mapping[str, Any], column: Mapping[str, Any]) -> str:
        key = str(column["key"])
        value = row.get(key, "—")
        number = _finite_number(value)
        if key in {"min", "max", "span"} and number is not None:
            return f"{number:.4f}"
        return _display_value(value)

    body = "".join(
        "<tr>"
        + "".join(f"<td>{html.escape(display_cell(row, column))}</td>" for column in columns)
        + "</tr>"
        for row in rows
    )
    return f'<table class="summary-table tabulation-table"><thead><tr>{headings}</tr></thead><tbody>{body}</tbody></table>'


def _report_metadata(metadata: Mapping[str, Any] | None) -> dict[str, Any]:
    run_time = datetime.now().astimezone()
    timezone_name = run_time.tzname() or run_time.strftime("%z")
    generated_at = f"{run_time.day} {run_time.strftime('%b %Y, %H:%M')} {timezone_name}"
    return {**dict(metadata or {}), "time run": generated_at, "Lucidum version": __version__}


def _metadata_items(
    metadata: Mapping[str, Any],
    *,
    include: set[str] | None = None,
    exclude: set[str] | None = None,
) -> str:
    include = {value.casefold() for value in include or set()}
    exclude = {value.casefold() for value in exclude or set()}
    items = []
    for key, value in metadata.items():
        normalised = str(key).strip().casefold()
        if include and normalised not in include:
            continue
        if normalised in exclude or value is None or value == "" or value == []:
            continue
        items.append(
            f"<div><dt>{html.escape(_metadata_label(key))}</dt>"
            f"<dd>{html.escape(_display_value(value))}</dd></div>"
        )
    return "".join(items)


def _summary_table_html(
    columns: Sequence[Mapping[str, Any]],
    rows: Sequence[Mapping[str, Any]],
    *,
    table_class: str,
) -> str:
    headings = "".join(f"<th scope=\"col\">{html.escape(str(column['label']))}</th>" for column in columns)
    body = "".join(
        "<tr>"
        + "".join(
            f"<td>{html.escape(_display_value(row.get(str(column['key']), '—')))}</td>"
            for column in columns
        )
        + "</tr>"
        for row in rows
    )
    return f'<table class="summary-table {html.escape(table_class)}"><thead><tr>{headings}</tr></thead><tbody>{body}</tbody></table>'


def _parameter_display(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (Mapping, list, tuple)):
        return json.dumps(value, ensure_ascii=False, separators=(", ", ": "))
    return str(value)


def _display_value(value: Any) -> str:
    if isinstance(value, (list, tuple, set)):
        return ", ".join(str(item) for item in value)
    if isinstance(value, bool):
        return "Yes" if value else "No"
    return str(value)


def _metadata_label(key: Any) -> str:
    text = str(key)
    return "SAMPLE_ROWS" if text.upper() == "SAMPLE_ROWS" else text.replace("_", " ").title()


def _sample_label(values: Any) -> str:
    if not isinstance(values, list) or not values:
        return ""
    return f"SAMPLE: {', '.join(str(value) for value in values)}"


def _base_label(metadata: Mapping[str, Any]) -> str:
    base = str(metadata.get("base") or "").strip()
    feature = str(metadata.get("feature") or "").strip()
    return f"Base: {feature} = {base}" if base and feature else ""


__all__ = [
    "gbm_evaluation_chart",
    "line_bar_chart",
    "report_filename",
    "write_echarts_report",
    "write_gbm_summary_report",
    "write_glm_summary_report",
]
