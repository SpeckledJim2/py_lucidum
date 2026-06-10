from __future__ import annotations

from fastapi import FastAPI, HTTPException, Request

from py_lucidum.app.context import AppContext

from .service import normalise_kind, read_spec_file, save_spec_file, submitted_spec, validate_spec


def register(app: FastAPI, context: AppContext) -> None:
    @app.get("/api/specs/{kind}")
    async def get_spec(request: Request, kind: str) -> dict:
        context.check_token(request)
        try:
            return read_spec_file(app.state, normalise_kind(kind))
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/api/specs/{kind}/validate")
    async def validate_spec_endpoint(request: Request, kind: str) -> dict:
        context.check_token(request)
        canonical_kind = ""
        try:
            canonical_kind = normalise_kind(kind)
            columns, rows = submitted_spec(await request.json(), canonical_kind)
            return validate_spec(context.dataset, canonical_kind, columns, rows)
        except ValueError as exc:
            return {
                "kind": canonical_kind or str(kind or ""),
                "valid": False,
                "errors": [str(exc)],
                "warnings": [],
                "row_issues": [],
                "message": str(exc),
            }

    @app.post("/api/specs/{kind}/save")
    async def save_spec_endpoint(request: Request, kind: str) -> dict:
        context.check_token(request)
        try:
            canonical_kind = normalise_kind(kind)
            columns, rows = submitted_spec(await request.json(), canonical_kind)
            result = save_spec_file(app.state, context.dataset, canonical_kind, columns, rows)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        if not result["valid"]:
            message = "; ".join(result["errors"][:3]) or result["message"]
            raise HTTPException(status_code=400, detail=message)
        return result
