from __future__ import annotations

import time
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request

from py_lucidum.app.context import AppContext
from py_lucidum.app.assets import NoStoreStaticFiles

from .query import summary


STATIC_DIR = Path(__file__).with_name("static")


def register(app: FastAPI, context: AppContext) -> None:
    if not any(getattr(route, "path", None) == "/tools/uk-map/static" for route in app.routes):
        app.mount("/tools/uk-map/static", NoStoreStaticFiles(directory=STATIC_DIR), name="uk_map_static")

    async def summary_endpoint(request: Request) -> dict:
        context.check_token(request)
        payload = await request.json()
        try:
            started = time.perf_counter_ns()
            result = summary(context.dataset, payload, defaults=getattr(app.state, "defaults", {}))
            elapsed_ns = time.perf_counter_ns() - started
            result["timings"] = {
                "duckdb_ns": elapsed_ns,
                "duckdb_ms": round(elapsed_ns / 1_000_000),
            }
            return result
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    app.add_api_route("/api/uk-map/summary", summary_endpoint, methods=["POST"])
