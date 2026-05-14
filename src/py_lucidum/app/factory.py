from __future__ import annotations

import html
import threading
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse

from py_lucidum.core import Dataset, load_saved_filters, resolve_filters_path

from .assets import NoStoreStaticFiles, no_store_file_response, no_store_html_response
from .context import AppContext


PACKAGE_ROOT = Path(__file__).parents[1]
PROJECT_ROOT = Path(__file__).parents[3]
STATIC_DIR = PACKAGE_ROOT / "static"
FAVICON_PATHS = (PROJECT_ROOT / "favicon.ico", STATIC_DIR / "favicon.ico")
PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
TOOL_ALIASES = {
    "line-bar": "line_bar",
    "line_bar": "line_bar",
    "linebar": "line_bar",
    "uk-map": "uk_map",
    "uk_map": "uk_map",
    "ukmap": "uk_map",
    "map": "uk_map",
}
TOOL_METADATA = {
    "line_bar": {"id": "line_bar", "label": "Line and bar chart"},
    "uk_map": {"id": "uk_map", "label": "UK mapping"},
}
DEFAULT_TOOLS = ["line_bar", "uk_map"]


def favicon_media_type(path: Path) -> str:
    with path.open("rb") as handle:
        if handle.read(len(PNG_SIGNATURE)) == PNG_SIGNATURE:
            return "image/png"
    return "image/x-icon"


def normalise_tools(tools: str | Sequence[str] | None) -> list[str]:
    if tools is None:
        requested = DEFAULT_TOOLS
    elif isinstance(tools, str):
        requested = [part.strip() for part in tools.split(",") if part.strip()]
    else:
        requested = [str(part).strip() for part in tools if str(part).strip()]
    if not requested:
        requested = DEFAULT_TOOLS

    enabled: list[str] = []
    for name in requested:
        canonical = TOOL_ALIASES.get(name.lower())
        if not canonical:
            supported = ", ".join(sorted(TOOL_ALIASES))
            raise ValueError(f"Unknown tool '{name}'. Supported tools: {supported}")
        if canonical not in enabled:
            enabled.append(canonical)
    return enabled


def tool_payload(enabled_tools: Sequence[str]) -> list[dict[str, str]]:
    return [TOOL_METADATA[tool] for tool in enabled_tools]


def index_html(dataset_name: str) -> str:
    title = f"lucidum · {html.escape(dataset_name)}" if dataset_name else "lucidum"
    html_text = (STATIC_DIR / "index.html").read_text(encoding="utf-8")
    return html_text.replace("<title>lucidum</title>", f"<title>{title}</title>", 1)


def create_app(
    dataset_path: str | Path,
    token: str | None = None,
    defaults: dict[str, str | None] | None = None,
    filters_path: str | Path | None = None,
    use_saved_filters: bool = True,
    tools: str | Sequence[str] | None = None,
) -> FastAPI:
    enabled_tools = normalise_tools(tools)
    dataset = Dataset(dataset_path)
    app = FastAPI(title="py_lucidum")
    app.state.dataset = dataset
    app.state.token = token
    app.state.filters_path = filters_path
    app.state.use_saved_filters = use_saved_filters
    app.state.resolved_filters_path = resolve_filters_path(filters_path, use_saved_filters=use_saved_filters)
    app.state.saved_filters = load_saved_filters(filters_path, use_saved_filters=use_saved_filters)
    app.state.enabled_tools = enabled_tools
    app.state.defaults = {
        key: value
        for key, value in (defaults or {}).items()
        if key in {"x", "actual", "expected", "denominator", "postcode_area", "postcode_sector"} and value
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
        payload["tools"] = tool_payload(app.state.enabled_tools)
        return payload

    @app.get("/")
    def index() -> HTMLResponse:
        return no_store_html_response(index_html(app.state.dataset.path.name))

    app.mount("/static", NoStoreStaticFiles(directory=STATIC_DIR), name="static")

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
    if "line_bar" in enabled_tools:
        from py_lucidum.tools.line_bar import register as register_line_bar

        register_line_bar(app, context)
    if "uk_map" in enabled_tools:
        from py_lucidum.tools.uk_map import register as register_uk_map

        register_uk_map(app, context)

    return app
