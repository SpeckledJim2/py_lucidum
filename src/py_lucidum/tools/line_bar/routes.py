from __future__ import annotations

from fastapi import FastAPI, HTTPException, Request

from py_lucidum.app.context import AppContext

from .query import chart


def register(app: FastAPI, context: AppContext) -> None:
    async def chart_endpoint(request: Request) -> dict:
        context.check_token(request)
        payload = await request.json()
        try:
            return chart(context.dataset, payload)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    app.add_api_route("/api/chart", chart_endpoint, methods=["POST"])
    app.add_api_route("/api/line-bar/chart", chart_endpoint, methods=["POST"])
