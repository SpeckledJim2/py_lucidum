from __future__ import annotations

import time

from fastapi import FastAPI, HTTPException, Request

from py_lucidum.app.context import AppContext

from .importance import feature_importance_payload
from .query import chart, table


def register(app: FastAPI, context: AppContext) -> None:
    async def chart_endpoint(request: Request) -> dict:
        context.check_token(request)
        payload = await request.json()
        try:
            started = time.perf_counter_ns()
            result = chart(context.dataset, payload, feature_spec=getattr(app.state, "feature_spec", {}))
            elapsed_ns = time.perf_counter_ns() - started
            result["timings"] = {
                "duckdb_ns": elapsed_ns,
                "duckdb_ms": round(elapsed_ns / 1_000_000),
            }
            return result
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    app.add_api_route("/api/chart", chart_endpoint, methods=["POST"])
    app.add_api_route("/api/line-bar/chart", chart_endpoint, methods=["POST"])

    async def table_endpoint(request: Request) -> dict:
        context.check_token(request)
        payload = await request.json()
        try:
            started = time.perf_counter_ns()
            result = table(context.dataset, payload, feature_spec=getattr(app.state, "feature_spec", {}))
            elapsed_ns = time.perf_counter_ns() - started
            result["timings"] = {
                "duckdb_ns": elapsed_ns,
                "duckdb_ms": round(elapsed_ns / 1_000_000),
            }
            return result
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    app.add_api_route("/api/line-bar/table", table_endpoint, methods=["POST"])

    @app.get("/api/line-bar/feature-importance")
    async def feature_importance_endpoint(request: Request) -> dict:
        context.check_token(request)
        return feature_importance_payload(
            context.dataset,
            gbm_store=getattr(app.state, "gbm_store", None),
            glm_store=getattr(app.state, "glm_store", None),
        )
