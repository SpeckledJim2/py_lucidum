from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.staticfiles import StaticFiles

from py_lucidum.app.context import AppContext

from .query import summary


STATIC_DIR = Path(__file__).with_name("static")


def register(app: FastAPI, context: AppContext) -> None:
    if not any(getattr(route, "path", None) == "/tools/uk-map/static" for route in app.routes):
        app.mount("/tools/uk-map/static", StaticFiles(directory=STATIC_DIR), name="uk_map_static")

    async def summary_endpoint(request: Request) -> dict:
        context.check_token(request)
        payload = await request.json()
        try:
            return summary(context.dataset, payload, defaults=getattr(app.state, "defaults", {}))
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    app.add_api_route("/api/uk-map/summary", summary_endpoint, methods=["POST"])
