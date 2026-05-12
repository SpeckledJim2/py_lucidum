from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse

from .query import Dataset


STATIC_DIR = Path(__file__).parent / "static"
PROJECT_ROOT = Path(__file__).parents[2]
FAVICON_PATHS = (PROJECT_ROOT / "favicon.ico", STATIC_DIR / "favicon.ico")
PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


def favicon_media_type(path: Path) -> str:
    with path.open("rb") as handle:
        if handle.read(len(PNG_SIGNATURE)) == PNG_SIGNATURE:
            return "image/png"
    return "image/x-icon"


def create_app(
    dataset_path: str | Path,
    token: str | None = None,
    defaults: dict[str, str | None] | None = None,
) -> FastAPI:
    dataset = Dataset(dataset_path)
    app = FastAPI(title="py_lucidum")
    app.state.dataset = dataset
    app.state.token = token
    app.state.defaults = {
        key: value
        for key, value in (defaults or {}).items()
        if key in {"x", "actual", "expected"} and value
    }

    def check_token(request: Request) -> None:
        expected = app.state.token
        if not expected:
            return
        supplied = request.headers.get("x-lucidum-token") or request.query_params.get("token")
        if supplied != expected:
            raise HTTPException(status_code=401, detail="Invalid or missing app token")

    @app.get("/")
    def index() -> FileResponse:
        return FileResponse(STATIC_DIR / "index.html")

    @app.api_route("/favicon.ico", methods=["GET", "HEAD"])
    def favicon() -> FileResponse:
        for path in FAVICON_PATHS:
            if path.exists():
                return FileResponse(path, media_type=favicon_media_type(path))
        raise HTTPException(status_code=404, detail="Favicon not found")

    @app.get("/api/schema")
    def schema(request: Request) -> dict[str, Any]:
        check_token(request)
        payload = dict(app.state.dataset.schema())
        payload["defaults"] = app.state.defaults
        return payload

    @app.post("/api/chart")
    async def chart(request: Request) -> dict[str, Any]:
        check_token(request)
        payload = await request.json()
        try:
            return app.state.dataset.chart(payload)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/api/reload")
    def reload_dataset(request: Request) -> dict[str, Any]:
        check_token(request)
        app.state.dataset.reload()
        payload = dict(app.state.dataset.schema())
        payload["defaults"] = app.state.defaults
        return payload

    return app
