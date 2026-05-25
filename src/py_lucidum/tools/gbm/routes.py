from __future__ import annotations

from typing import Any

from fastapi import FastAPI, Request

from py_lucidum.app.context import AppContext


def register(app: FastAPI, context: AppContext) -> None:
    async def summary_endpoint(request: Request) -> dict[str, Any]:
        context.check_token(request)
        return {
            "tool": "gbm",
            "status": "not_implemented",
            "message": "GBM modelling is not implemented in this refactor slice.",
        }

    app.add_api_route("/api/gbm/summary", summary_endpoint, methods=["GET"])
