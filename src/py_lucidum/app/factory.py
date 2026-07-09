from __future__ import annotations

import html
import os
import re
import threading
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import duckdb
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import FileResponse, HTMLResponse

from py_lucidum._version import __version__
from py_lucidum.core import (
    Dataset,
    denominator_warnings,
    duckdb_error_message,
    is_numeric_kind,
    json_number,
    load_features,
    load_kpis,
    load_saved_filters,
    normalise_denominator,
    response_summary,
    resolve_features_path,
    resolve_filters_path,
    resolve_kpis_path,
    summarize_denominator,
)
from py_lucidum.tools.registry import normalise_tools, register_tools, tool_payload

from .assets import NoStoreStaticFiles, no_store_file_response, no_store_html_response
from .context import AppContext
from .servers import ServerStopError, list_lucidum_servers, stop_lucidum_server
from .telemetry import TelemetryMiddleware, TelemetryStore


PACKAGE_ROOT = Path(__file__).parents[1]
PROJECT_ROOT = Path(__file__).parents[3]
STATIC_DIR = PACKAGE_ROOT / "static"
FAVICON_PATHS = (PROJECT_ROOT / "favicon.ico", STATIC_DIR / "favicon.ico")
PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
DEFAULT_KEYS = {
    "x",
    "actual",
    "expected",
    "expected2",
    "denominator",
    "postcode_area",
    "postcode_sector",
    "postcode_unit",
    "latitude",
    "longitude",
    "source",
    "line_bar_favourite",
}
TOOL_BUTTON_RE = re.compile(r'<button\b[^>]*\bdata-tool="([^"]+)"[^>]*>')
TOOL_BUTTON_BLOCK_RE = re.compile(r'\s*<button\b[^>]*\bdata-tool="([^"]+)"[^>]*>.*?</button>', re.DOTALL)
TOOL_SELECTOR_SECTION_RE = re.compile(r'<section\b[^>]*\bid="toolSelectorSection"[^>]*>')
MODEL_SIDEBAR_PANEL_RE = re.compile(r'<section\b[^>]*\bid="(gbmSidebarPanel|glmSidebarPanel)"[^>]*>')
HEADER_BUTTON_RE = re.compile(r'<(?:a|button)\b[^>]*\bid="(monitorLink|stopAppBtn)"[^>]*>')
CLASS_ATTR_RE = re.compile(r'\bclass="([^"]*)"')
ARIA_HIDDEN_ATTR_RE = re.compile(r'\s+aria-hidden="[^"]*"')
MODEL_SIDEBAR_PANEL_TOOLS = {
    "gbmSidebarPanel": "gbm",
    "glmSidebarPanel": "glm",
}


def favicon_media_type(path: Path) -> str:
    with path.open("rb") as handle:
        if handle.read(len(PNG_SIGNATURE)) == PNG_SIGNATURE:
            return "image/png"
    return "image/x-icon"


def render_initial_tool_visibility(html_text: str, enabled_tools: Sequence[str]) -> str:
    ordered_enabled = list(dict.fromkeys(str(tool) for tool in enabled_tools))
    enabled = set(ordered_enabled)
    show_tool_selector = len(ordered_enabled) > 1

    def set_hidden_state(tag: str, hidden: bool, *, hide_interaction: bool = False) -> str:
        class_match = CLASS_ATTR_RE.search(tag)
        if not class_match:
            return tag
        classes = [class_name for class_name in class_match.group(1).split() if class_name != "hidden"]
        if hidden:
            classes.append("hidden")
        class_value = " ".join(classes)
        updated = f"{tag[:class_match.start(1)]}{class_value}{tag[class_match.end(1):]}"
        if not hide_interaction:
            return updated
        updated = ARIA_HIDDEN_ATTR_RE.sub("", updated)
        updated = re.sub(r"\s+inert\b", "", updated)
        if hidden:
            updated = updated[:-1] + ' aria-hidden="true" inert>'
        return updated

    def replace_tool_button(match: re.Match[str]) -> str:
        tag = match.group(0)
        tool_id = match.group(1)
        return set_hidden_state(tag, tool_id not in enabled or not show_tool_selector)

    def replace_tool_selector_section(match: re.Match[str]) -> str:
        return set_hidden_state(match.group(0), not show_tool_selector)

    def replace_model_sidebar_panel(match: re.Match[str]) -> str:
        tag = match.group(0)
        panel_id = match.group(1)
        tool_id = MODEL_SIDEBAR_PANEL_TOOLS.get(panel_id, "")
        return set_hidden_state(tag, tool_id not in enabled, hide_interaction=True)

    html_text = TOOL_BUTTON_RE.sub(replace_tool_button, html_text)
    block_matches = list(TOOL_BUTTON_BLOCK_RE.finditer(html_text))
    if block_matches:
        blocks_by_tool = {match.group(1): match.group(0) for match in block_matches}
        ordered_tools = [
            *[tool_id for tool_id in ordered_enabled if tool_id in blocks_by_tool],
            *[match.group(1) for match in block_matches if match.group(1) not in enabled],
        ]
        replacement = "".join(blocks_by_tool[tool_id] for tool_id in ordered_tools)
        html_text = f"{html_text[:block_matches[0].start()]}{replacement}{html_text[block_matches[-1].end():]}"
    html_text = TOOL_SELECTOR_SECTION_RE.sub(replace_tool_selector_section, html_text)
    return MODEL_SIDEBAR_PANEL_RE.sub(replace_model_sidebar_panel, html_text)


def render_initial_header_button_visibility(html_text: str, header_buttons: bool) -> str:
    def set_hidden_state(tag: str, hidden: bool) -> str:
        class_match = CLASS_ATTR_RE.search(tag)
        if not class_match:
            return tag
        classes = [class_name for class_name in class_match.group(1).split() if class_name != "hidden"]
        if hidden:
            classes.append("hidden")
        class_value = " ".join(classes)
        updated = f"{tag[:class_match.start(1)]}{class_value}{tag[class_match.end(1):]}"
        updated = ARIA_HIDDEN_ATTR_RE.sub("", updated)
        updated = re.sub(r"\s+inert\b", "", updated)
        if hidden:
            updated = updated[:-1] + ' aria-hidden="true" inert>'
        return updated

    def replace_header_button(match: re.Match[str]) -> str:
        return set_hidden_state(match.group(0), not header_buttons)

    return HEADER_BUTTON_RE.sub(replace_header_button, html_text)


def index_html(dataset_name: str, enabled_tools: Sequence[str], header_buttons: bool = False) -> str:
    title = f"lucidum · {html.escape(dataset_name)}" if dataset_name else "lucidum"
    html_text = (STATIC_DIR / "index.html").read_text(encoding="utf-8")
    html_text = html_text.replace("<title>lucidum</title>", f"<title>{title}</title>", 1)
    html_text = render_initial_tool_visibility(html_text, enabled_tools)
    return render_initial_header_button_visibility(html_text, header_buttons)


def monitor_html(dataset_name: str) -> str:
    title = f"lucidum monitor · {html.escape(dataset_name)}" if dataset_name else "lucidum monitor"
    html_text = (STATIC_DIR / "monitor.html").read_text(encoding="utf-8")
    return html_text.replace("<title>lucidum monitor</title>", f"<title>{title}</title>", 1)


def feature_bases_payload(feature_spec: Any) -> dict[str, str]:
    if not isinstance(feature_spec, dict):
        return {}
    rows = feature_spec.get("rows", [])
    if not isinstance(rows, list):
        return {}
    return {
        str(row.get("feature")): str(row.get("base") or "").strip()
        for row in rows
        if isinstance(row, dict) and row.get("feature") and str(row.get("base") or "").strip()
    }


def create_app(
    dataset_path: str | Path,
    token: str | None = None,
    defaults: dict[str, str | None] | None = None,
    filters_path: str | Path | None = None,
    use_saved_filters: bool = True,
    tools: str | Sequence[str] | None = None,
    kpis: str | Path | None = None,
    kpis_path: str | Path | None = None,
    use_kpis: bool = True,
    no_kpis: bool = False,
    features: str | Path | None = None,
    features_path: str | Path | None = None,
    use_features: bool = True,
    no_features: bool = False,
    line_bar_favourites_path: str | Path | None = None,
    header_buttons: bool = False,
    title_prefix: str | None = None,
) -> FastAPI:
    enabled_tools = normalise_tools(tools)
    resolved_dataset_path = Path(dataset_path).expanduser().resolve()
    if resolved_dataset_path.is_dir() and any(tool_id in enabled_tools for tool_id in ("glm", "gbm")):
        raise ValueError(
            "Parquet folder inputs are not supported with GLM or GBM. "
            "Use a single Parquet file or deselect GLM/GBM."
        )
    allow_missing_spec_paths = "specs" in enabled_tools
    if kpis and kpis_path and Path(kpis).expanduser() != Path(kpis_path).expanduser():
        raise ValueError("Specify either kpis or kpis_path, not both")
    if features and features_path and Path(features).expanduser() != Path(features_path).expanduser():
        raise ValueError("Specify either features or features_path, not both")
    selected_kpis_path = kpis_path or kpis
    kpis_enabled = use_kpis and not no_kpis
    selected_features_path = features_path or features
    features_enabled = use_features and not no_features
    dataset = Dataset(resolved_dataset_path)
    app = FastAPI(title="py_lucidum")
    app.state.dataset = dataset
    app.state.telemetry = TelemetryStore()
    app.state.token = token
    app.state.lucidum_server_metadata = {
        "pid": os.getpid(),
        "dataset_path": str(dataset.path),
        "dataset_name": dataset.path.name,
    }
    app.state.filters_path = filters_path
    app.state.use_saved_filters = use_saved_filters
    app.state.allow_missing_spec_paths = allow_missing_spec_paths
    app.state.resolved_filters_path = resolve_filters_path(filters_path, use_saved_filters=use_saved_filters)
    app.state.saved_filters = load_saved_filters(filters_path, use_saved_filters=use_saved_filters, missing_ok=allow_missing_spec_paths)
    app.state.kpis_path = selected_kpis_path
    app.state.use_kpis = kpis_enabled
    app.state.resolved_kpis_path = resolve_kpis_path(selected_kpis_path, use_kpis=kpis_enabled)
    app.state.kpis = load_kpis(selected_kpis_path, use_kpis=kpis_enabled, missing_ok=allow_missing_spec_paths)
    app.state.features_path = selected_features_path
    app.state.use_features = features_enabled
    app.state.resolved_features_path = resolve_features_path(selected_features_path, use_features=features_enabled)
    app.state.feature_spec = load_features(selected_features_path, use_features=features_enabled, missing_ok=allow_missing_spec_paths)
    app.state.line_bar_favourites_path = (
        Path(line_bar_favourites_path).expanduser().resolve() if line_bar_favourites_path else None
    )
    app.state.enabled_tools = enabled_tools
    app.state.header_buttons = bool(header_buttons)
    app.state.title_prefix = str(title_prefix or "").strip()
    app.state.defaults = {
        key: value
        for key, value in (defaults or {}).items()
        if key in DEFAULT_KEYS and value
    }

    def check_token(request: Request) -> None:
        expected = app.state.token
        if not expected:
            return
        supplied = request.headers.get("x-lucidum-token") or request.query_params.get("token")
        if supplied != expected:
            raise HTTPException(status_code=401, detail="Invalid or missing app token")

    def schema_payload() -> dict[str, Any]:
        payload = dict(app.state.dataset.schema())
        payload["defaults"] = app.state.defaults
        payload["filters"] = app.state.saved_filters
        payload["kpis"] = app.state.kpis
        payload["feature_bases"] = feature_bases_payload(app.state.feature_spec)
        payload["tools"] = tool_payload(app.state.enabled_tools)
        payload["data_sources"] = app.state.dataset.data_sources()
        payload["app_version"] = __version__
        payload["header_buttons"] = app.state.header_buttons
        payload["title_prefix"] = app.state.title_prefix
        return payload

    @app.get("/")
    def index() -> HTMLResponse:
        return no_store_html_response(index_html(app.state.dataset.path.name, app.state.enabled_tools, app.state.header_buttons))

    app.mount("/static", NoStoreStaticFiles(directory=STATIC_DIR), name="static")

    @app.get("/monitor")
    def monitor(request: Request) -> HTMLResponse:
        check_token(request)
        return no_store_html_response(monitor_html(app.state.dataset.path.name))

    @app.api_route("/favicon.ico", methods=["GET", "HEAD"])
    def favicon() -> FileResponse:
        for path in FAVICON_PATHS:
            if path.exists():
                return no_store_file_response(path, media_type=favicon_media_type(path))
        raise HTTPException(status_code=404, detail="Favicon not found")

    @app.get("/api/schema")
    def schema(request: Request) -> dict[str, Any]:
        check_token(request)
        try:
            app.state.dataset.refresh_if_source_changed()
            return schema_payload()
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except duckdb.Error as exc:
            raise HTTPException(status_code=400, detail=duckdb_error_message(exc)) from exc

    @app.post("/api/filter/row-count")
    async def filter_row_count(request: Request) -> dict[str, Any]:
        check_token(request)
        payload = await request.json()
        dataset = app.state.dataset
        try:
            started = time.perf_counter_ns()
            with dataset.lock:
                filter_sql = dataset.normalise_filter(payload.get("filter"))
                row_count = dataset.row_count()
                filtered_row_count = dataset.filtered_row_count(filter_sql)
            elapsed_ns = time.perf_counter_ns() - started
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except duckdb.Error as exc:
            raise HTTPException(status_code=400, detail=duckdb_error_message(exc)) from exc
        return {
            "row_count": row_count,
            "filtered_row_count": filtered_row_count,
            "filter": filter_sql,
            "timings": {
                "duckdb_ns": elapsed_ns,
                "duckdb_ms": round(elapsed_ns / 1_000_000),
            },
        }

    @app.post("/api/metrics/summary")
    async def metric_summary(request: Request) -> dict[str, Any]:
        check_token(request)
        payload = await request.json()
        dataset = app.state.dataset
        try:
            started = time.perf_counter_ns()
            with dataset.lock:
                source_id = dataset.normalise_source(payload.get("source"))
                columns = dataset.column_map_for_source(source_id)
                actual = str(payload.get("actual") or "").strip()
                if actual not in columns or not is_numeric_kind(columns[actual].kind):
                    raise ValueError("Choose a valid numeric Actual column")
                denominator = normalise_denominator(payload.get("denominator", payload.get("weight")), columns)
                filter_sql = dataset.normalise_filter(payload.get("filter"), source_id=source_id)
                responses = [{"label": actual, "numerator": actual}]
                row_count = dataset.row_count_for_source(source_id)
                filtered_row_count = dataset.filtered_row_count(filter_sql, source_id=source_id)
                denominator_summary = summarize_denominator(dataset, responses, denominator, filter_sql, source_id=source_id)
                response_summaries = response_summary(dataset, responses, denominator, filter_sql, source_id=source_id)
            elapsed_ns = time.perf_counter_ns() - started
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except duckdb.Error as exc:
            raise HTTPException(status_code=400, detail=duckdb_error_message(exc)) from exc
        return {
            "source": source_id,
            "actual": {"column": actual, "label": actual},
            "denominator": {
                "column": denominator["column"],
                "label": denominator["label"],
                "bar_label": denominator["bar_label"],
                "value": json_number(denominator_summary.get("value")),
                "missing_response_rows": json_number(denominator_summary.get("missing_response_rows")),
                "missing_weight_rows": json_number(denominator_summary.get("missing_weight_rows")),
                "zero_weight_rows": json_number(denominator_summary.get("zero_weight_rows")),
                "negative_weight_rows": json_number(denominator_summary.get("negative_weight_rows")),
            },
            "response_summaries": response_summaries,
            "row_count": row_count,
            "filtered_row_count": filtered_row_count,
            "filter": filter_sql,
            "warnings": denominator_warnings(denominator, denominator_summary, responses),
            "timings": {
                "duckdb_ns": elapsed_ns,
                "duckdb_ms": round(elapsed_ns / 1_000_000),
            },
        }

    @app.post("/api/banding/suggestion")
    def banding_suggestion(request: Request, payload: dict[str, Any]) -> dict[str, Any]:
        check_token(request)
        started = time.perf_counter()
        dataset = app.state.dataset
        try:
            from py_lucidum.tools.line_bar.query import banding_suggestion as line_bar_banding_suggestion

            result = line_bar_banding_suggestion(dataset, payload)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except duckdb.Error as exc:
            raise HTTPException(status_code=400, detail=duckdb_error_message(exc)) from exc
        elapsed_ms = round((time.perf_counter() - started) * 1000, 1)
        result["timings"] = {"duckdb_ms": elapsed_ms}
        return result

    @app.post("/api/date-bucket/suggestion")
    def date_bucket_suggestion(request: Request, payload: dict[str, Any]) -> dict[str, Any]:
        check_token(request)
        started = time.perf_counter()
        dataset = app.state.dataset
        try:
            with dataset.lock:
                source = dataset.normalise_source(payload.get("xSource") or payload.get("source"))
                feature = str(payload.get("feature") or "").strip()
                filter_sql = dataset.normalise_filter(payload.get("filter"), source_id=source)
                suggestion = dataset.date_bucket_suggestion_for_column(source, feature, filter_sql)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except duckdb.Error as exc:
            raise HTTPException(status_code=400, detail=duckdb_error_message(exc)) from exc
        elapsed_ms = round((time.perf_counter() - started) * 1000, 1)
        return {
            "feature": feature,
            "source": source,
            "date_bucket": suggestion["date_bucket"],
            "min_value": suggestion["min_value"],
            "max_value": suggestion["max_value"],
            "timings": {"duckdb_ms": elapsed_ms},
        }

    @app.get("/api/health")
    def health(request: Request) -> dict[str, str]:
        check_token(request)
        return {"status": "ok"}

    @app.get("/api/telemetry")
    def telemetry(request: Request, response: Response) -> dict[str, Any]:
        check_token(request)
        response.headers["Cache-Control"] = "no-store"
        return app.state.telemetry.snapshot()

    @app.get("/api/lucidum-servers")
    def lucidum_servers(request: Request, response: Response) -> dict[str, Any]:
        check_token(request)
        response.headers["Cache-Control"] = "no-store"
        servers = list_lucidum_servers(app.state)
        return {"count": len(servers), "servers": servers}

    @app.post("/api/lucidum-servers/stop")
    def stop_lucidum_server_route(request: Request, payload: dict[str, Any]) -> dict[str, Any]:
        check_token(request)
        try:
            return stop_lucidum_server(app.state, int(payload.get("pid")), float(payload.get("create_time")))
        except (TypeError, ValueError) as exc:
            raise HTTPException(status_code=400, detail="Choose a valid lucidum server process") from exc
        except ServerStopError as exc:
            raise HTTPException(status_code=exc.status_code, detail=str(exc)) from exc

    @app.post("/api/reload")
    def reload_dataset(request: Request) -> dict[str, Any]:
        check_token(request)
        app.state.dataset.reload()
        app.state.resolved_filters_path = resolve_filters_path(
            app.state.filters_path,
            use_saved_filters=app.state.use_saved_filters,
        )
        app.state.saved_filters = load_saved_filters(
            app.state.filters_path,
            use_saved_filters=app.state.use_saved_filters,
            missing_ok=app.state.allow_missing_spec_paths,
        )
        app.state.resolved_kpis_path = resolve_kpis_path(
            app.state.kpis_path,
            use_kpis=app.state.use_kpis,
        )
        app.state.kpis = load_kpis(
            app.state.kpis_path,
            use_kpis=app.state.use_kpis,
            missing_ok=app.state.allow_missing_spec_paths,
        )
        app.state.resolved_features_path = resolve_features_path(
            app.state.features_path,
            use_features=app.state.use_features,
        )
        app.state.feature_spec = load_features(
            app.state.features_path,
            use_features=app.state.use_features,
            missing_ok=app.state.allow_missing_spec_paths,
        )
        return schema_payload()

    @app.post("/api/shutdown")
    def shutdown(request: Request) -> dict[str, str]:
        check_token(request)
        shutdown_callback = getattr(app.state, "shutdown_callback", None)
        if not callable(shutdown_callback):
            raise HTTPException(status_code=503, detail="Shutdown is only available when launched with the lucidum command")
        threading.Timer(0.2, shutdown_callback).start()
        return {"message": "py_lucidum is stopping"}

    context = AppContext(dataset=dataset, check_token=check_token)
    register_tools(app, context, enabled_tools)
    if "line_bar" not in enabled_tools and any(tool_id in enabled_tools for tool_id in ("dataset_viewer", "histogram", "uk_map")):
        from py_lucidum.tools.line_bar.routes import register_favourite_routes

        register_favourite_routes(app, context)
    if "line_bar" in enabled_tools:
        from py_lucidum.tools.line_bar.model_ratio import register_model_ratio_source_provider

        register_model_ratio_source_provider(app, dataset)

    app.add_middleware(TelemetryMiddleware, store=app.state.telemetry)
    return app
