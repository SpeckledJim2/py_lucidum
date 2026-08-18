from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import duckdb
from fastapi import FastAPI, HTTPException, Request

from py_lucidum.app.context import AppContext
from py_lucidum.app.assets import NoStoreStaticFiles
from py_lucidum.core import duckdb_error_message

from .export import save_sector_smoothing_sidecar
from .query import summary


STATIC_DIR = Path(__file__).with_name("static")


class _TimedDuckDbConnection:
    def __init__(self, connection: Any):
        self._connection = connection
        self.elapsed_ns = 0

    def execute(self, *args: Any, **kwargs: Any) -> "_TimedDuckDbConnection":
        started = time.perf_counter_ns()
        try:
            self._connection.execute(*args, **kwargs)
            return self
        finally:
            self.elapsed_ns += time.perf_counter_ns() - started

    def fetchone(self) -> Any:
        started = time.perf_counter_ns()
        try:
            return self._connection.fetchone()
        finally:
            self.elapsed_ns += time.perf_counter_ns() - started

    def fetchall(self) -> Any:
        started = time.perf_counter_ns()
        try:
            return self._connection.fetchall()
        finally:
            self.elapsed_ns += time.perf_counter_ns() - started

    @property
    def description(self) -> Any:
        return self._connection.description

    def __getattr__(self, name: str) -> Any:
        return getattr(self._connection, name)


def register(app: FastAPI, context: AppContext) -> None:
    if not any(getattr(route, "path", None) == "/tools/uk-map/static" for route in app.routes):
        app.mount("/tools/uk-map/static", NoStoreStaticFiles(directory=STATIC_DIR), name="uk_map_static")

    async def summary_endpoint(request: Request) -> dict:
        context.check_token(request)
        payload = await request.json()
        try:
            started = time.perf_counter_ns()
            with context.dataset.lock:
                original_connection = context.dataset.con
                timed_connection = _TimedDuckDbConnection(original_connection)
                context.dataset.con = timed_connection
                try:
                    result = summary(context.dataset, payload, defaults=getattr(app.state, "defaults", {}))
                finally:
                    context.dataset.con = original_connection
            server_ns = time.perf_counter_ns() - started
            duckdb_ns = timed_connection.elapsed_ns
            result["timings"] = {
                "server_ns": server_ns,
                "server_ms": round(server_ns / 1_000_000),
                "duckdb_ns": duckdb_ns,
                "duckdb_ms": round(duckdb_ns / 1_000_000),
            }
            return result
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    async def sector_smoothing_endpoint(request: Request) -> dict:
        context.check_token(request)
        payload = await request.json()
        try:
            with context.dataset.lock:
                return save_sector_smoothing_sidecar(
                    context.dataset,
                    payload,
                    defaults=getattr(app.state, "defaults", {}),
                )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except (duckdb.Error, OSError) as exc:
            raise HTTPException(
                status_code=500,
                detail=f"Could not save sector smoothing Parquet: {duckdb_error_message(exc)}",
            ) from exc

    app.add_api_route("/api/uk-map/summary", summary_endpoint, methods=["POST"])
    app.add_api_route("/api/uk-map/sector-smoothing", sector_smoothing_endpoint, methods=["POST"])
