from __future__ import annotations

import html
import json
import math
import re
from collections.abc import Iterable, Mapping, Sequence
from datetime import datetime
from pathlib import Path
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
                    {"model_id": str(model_id)}
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
                "base": base,
                "controls": settings,
                "partial_dependence": partial_dependence,
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
) -> str:
    source_id = "dataset"
    needs_glm = expected_source == "glm" or partial_dependence == "glm"
    needs_gbm = expected_source == "gbm" or partial_dependence == "shap"
    if needs_glm:
        from py_lucidum.tools.glm.store import GlmModelStore, GlmSourceProvider

        store = GlmModelStore(dataset.path, dataset=dataset)
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

        store = GbmModelStore(dataset.path, dataset=dataset)
        store.manifest(model_id)
        if not store.artifact_path(model_id, "predictions").exists():
            raise ValueError(f"GBM predictions are unavailable for model {model_id}")
        if partial_dependence == "shap" and not store.artifact_path(model_id, "shap_long").exists():
            raise ValueError(f"GBM SHAP values are unavailable for model {model_id}")
        dataset.register_data_source_provider(GbmSourceProvider(store))
        if expected_source == "gbm":
            source_id = store.source_id(model_id, "predictions")
    return source_id


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


__all__ = ["line_bar_chart", "report_filename", "write_echarts_report"]
