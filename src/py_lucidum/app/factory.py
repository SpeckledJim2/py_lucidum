from __future__ import annotations

import html
import os
import threading
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import duckdb
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import FileResponse, HTMLResponse

from py_lucidum.core import (
    Dataset,
    duckdb_error_message,
    load_features,
    load_kpis,
    load_saved_filters,
    resolve_features_path,
    resolve_filters_path,
    resolve_kpis_path,
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
    "denominator",
    "postcode_area",
    "postcode_sector",
    "postcode_unit",
    "latitude",
    "longitude",
    "source",
}


def favicon_media_type(path: Path) -> str:
    with path.open("rb") as handle:
        if handle.read(len(PNG_SIGNATURE)) == PNG_SIGNATURE:
            return "image/png"
    return "image/x-icon"


def index_html(dataset_name: str) -> str:
    title = f"lucidum · {html.escape(dataset_name)}" if dataset_name else "lucidum"
    html_text = (STATIC_DIR / "index.html").read_text(encoding="utf-8")
    return html_text.replace("<title>lucidum</title>", f"<title>{title}</title>", 1)


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
) -> FastAPI:
    enabled_tools = normalise_tools(tools)
    allow_missing_spec_paths = "specs" in enabled_tools
    if kpis and kpis_path and Path(kpis).expanduser() != Path(kpis_path).expanduser():
        raise ValueError("Specify either kpis or kpis_path, not both")
    if features and features_path and Path(features).expanduser() != Path(features_path).expanduser():
        raise ValueError("Specify either features or features_path, not both")
    selected_kpis_path = kpis_path or kpis
    kpis_enabled = use_kpis and not no_kpis
    selected_features_path = features_path or features
    features_enabled = use_features and not no_features
    dataset = Dataset(dataset_path)
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
    app.state.enabled_tools = enabled_tools
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
        return payload

    @app.get("/")
    def index() -> HTMLResponse:
        return no_store_html_response(index_html(app.state.dataset.path.name))

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
        return schema_payload()

    @app.post("/api/banding/suggestion")
    def banding_suggestion(request: Request, payload: dict[str, Any]) -> dict[str, Any]:
        check_token(request)
        started = time.perf_counter()
        dataset = app.state.dataset
        try:
            with dataset.lock:
                source = dataset.normalise_source(payload.get("xSource") or payload.get("source"))
                feature = str(payload.get("feature") or "").strip()
                filter_sql = dataset.normalise_filter(payload.get("filter"), source_id=source)
                suggestion = dataset.band_suggestion_for_column(source, feature, filter_sql)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except duckdb.Error as exc:
            raise HTTPException(status_code=400, detail=duckdb_error_message(exc)) from exc
        elapsed_ms = round((time.perf_counter() - started) * 1000, 1)
        return {
            "feature": feature,
            "source": source,
            "band_suggestion": suggestion,
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

    app.add_middleware(TelemetryMiddleware, store=app.state.telemetry)
    return app
