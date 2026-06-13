from __future__ import annotations

import time

import duckdb
from fastapi import FastAPI, HTTPException, Request

from py_lucidum.app.context import AppContext
from py_lucidum.core import duckdb_error_message

from .query import table


def register(app: FastAPI, context: AppContext) -> None:
    async def table_endpoint(request: Request) -> dict:
        context.check_token(request)
        payload = await request.json()
        try:
            started = time.perf_counter_ns()
            result = table(context.dataset, payload)
            elapsed_ns = time.perf_counter_ns() - started
            result["timings"] = {
                "duckdb_ns": elapsed_ns,
                "duckdb_ms": round(elapsed_ns / 1_000_000),
            }
            return result
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except duckdb.Error as exc:
            raise HTTPException(status_code=400, detail=duckdb_error_message(exc)) from exc

    app.add_api_route("/api/dataset-viewer/table", table_endpoint, methods=["POST"])
