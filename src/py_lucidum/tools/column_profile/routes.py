from __future__ import annotations

import time

from fastapi import FastAPI, HTTPException, Request

from py_lucidum.app.context import AppContext

from .query import profile, profile_detail


def register(app: FastAPI, context: AppContext) -> None:
    async def summary_endpoint(request: Request) -> dict:
        context.check_token(request)
        payload = await request.json()
        try:
            started = time.perf_counter_ns()
            result = profile(context.dataset, payload)
            elapsed_ns = time.perf_counter_ns() - started
            result["timings"] = {
                "duckdb_ns": elapsed_ns,
                "duckdb_ms": round(elapsed_ns / 1_000_000),
            }
            return result
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    async def detail_endpoint(request: Request) -> dict:
        context.check_token(request)
        payload = await request.json()
        try:
            started = time.perf_counter_ns()
            result = profile_detail(context.dataset, payload)
            elapsed_ns = time.perf_counter_ns() - started
            result["timings"] = {
                "duckdb_ns": elapsed_ns,
                "duckdb_ms": round(elapsed_ns / 1_000_000),
            }
            return result
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    app.add_api_route("/api/column-profile/summary", summary_endpoint, methods=["POST"])
    app.add_api_route("/api/column-profile/detail", detail_endpoint, methods=["POST"])
