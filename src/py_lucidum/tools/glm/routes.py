from __future__ import annotations

from typing import Any

from fastapi import FastAPI, HTTPException, Request

from py_lucidum.app.context import AppContext

from .formula_assist import formula_levels
from .jobs import GlmJobManager
from .store import GlmModelNameError, GlmModelStore, GlmSourceProvider
from .tabulation import export_tabulations, rebase_tabulation, reset_tabulation_rebase, tabulation_config, tabulation_plot, tabulation_table
from .training import MissingGlmDependency, glm_training_dependencies
from .validation import DENOMINATOR_COLUMN, RESPONSE_COLUMN, family_options_payload, regularization_options_payload, sample_metadata, validate_request


def register(app: FastAPI, context: AppContext) -> None:
    store = GlmModelStore(context.dataset.path, dataset=context.dataset)
    context.dataset.register_data_source_provider(GlmSourceProvider(store))
    jobs = GlmJobManager()
    app.state.glm_store = store
    app.state.glm_jobs = jobs

    def config_payload() -> dict[str, Any]:
        return {
            "tool": "glm",
            "status": "ready",
            "response": RESPONSE_COLUMN,
            "denominator": DENOMINATOR_COLUMN,
            "link": "auto",
            "sample": sample_metadata(context.dataset),
            "families": family_options_payload(),
            "regularization": regularization_options_payload(),
            "models": store.list_models(),
            "active_model_id": store.active_model_id(),
        }

    @app.get("/api/glm/summary")
    async def summary_endpoint(request: Request) -> dict[str, Any]:
        context.check_token(request)
        return config_payload()

    @app.get("/api/glm/config")
    async def config_endpoint(request: Request) -> dict[str, Any]:
        context.check_token(request)
        return config_payload()

    @app.get("/api/glm/models")
    async def models_endpoint(request: Request) -> dict[str, Any]:
        context.check_token(request)
        return {
            "models": store.list_models(),
            "active_model_id": store.active_model_id(),
        }

    @app.post("/api/glm/validate")
    async def validate_endpoint(request: Request) -> dict[str, Any]:
        context.check_token(request)
        payload = dict(await request.json())
        return validate_request(context.dataset, payload)

    @app.post("/api/glm/formula/levels")
    async def formula_levels_endpoint(request: Request) -> dict[str, Any]:
        context.check_token(request)
        payload = dict(await request.json())
        try:
            return formula_levels(context.dataset, payload)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/api/glm/build")
    async def build_endpoint(request: Request) -> dict[str, Any]:
        context.check_token(request)
        payload = dict(await request.json())
        try:
            glm_training_dependencies()
            validation = validate_request(context.dataset, payload)
            if not validation["ok"]:
                raise ValueError("; ".join(validation["errors"]))
            job = jobs.start(context.dataset, store, payload)
            return job.as_payload()
        except MissingGlmDependency as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.get("/api/glm/jobs/{job_id}")
    async def job_endpoint(request: Request, job_id: str) -> dict[str, Any]:
        context.check_token(request)
        job = jobs.get(job_id)
        if not job:
            raise HTTPException(status_code=404, detail="Choose a valid GLM job")
        return job.as_payload()

    @app.post("/api/glm/tabulations/build")
    async def tabulation_build_endpoint(request: Request) -> dict[str, Any]:
        context.check_token(request)
        payload = dict(await request.json())
        try:
            job = jobs.start_tabulations(context.dataset, store, payload, getattr(app.state, "feature_spec", {}), getattr(app.state, "gbm_store", None))
            return job.as_payload()
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.get("/api/glm/tabulations/jobs/{job_id}")
    async def tabulation_job_endpoint(request: Request, job_id: str) -> dict[str, Any]:
        context.check_token(request)
        job = jobs.get(job_id)
        if not job:
            raise HTTPException(status_code=404, detail="Choose a valid GLM tabulation job")
        return job.as_payload()

    @app.post("/api/glm/tabulations/config")
    async def tabulation_config_endpoint(request: Request) -> dict[str, Any]:
        context.check_token(request)
        payload = dict(await request.json())
        return tabulation_config(store, payload, gbm_store=getattr(app.state, "gbm_store", None))

    @app.post("/api/glm/tabulations/table")
    async def tabulation_table_endpoint(request: Request) -> dict[str, Any]:
        context.check_token(request)
        payload = dict(await request.json())
        return tabulation_table(store, payload, gbm_store=getattr(app.state, "gbm_store", None))

    @app.post("/api/glm/tabulations/plot")
    async def tabulation_plot_endpoint(request: Request) -> dict[str, Any]:
        context.check_token(request)
        payload = dict(await request.json())
        return tabulation_plot(store, payload, gbm_store=getattr(app.state, "gbm_store", None))

    @app.post("/api/glm/tabulations/export")
    async def tabulation_export_endpoint(request: Request) -> dict[str, Any]:
        context.check_token(request)
        payload = dict(await request.json())
        try:
            return export_tabulations(store, payload, gbm_store=getattr(app.state, "gbm_store", None))
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/api/glm/tabulations/rebase")
    async def tabulation_rebase_endpoint(request: Request) -> dict[str, Any]:
        context.check_token(request)
        payload = dict(await request.json())
        try:
            return rebase_tabulation(context.dataset, store, payload)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/api/glm/tabulations/rebase/reset")
    async def tabulation_rebase_reset_endpoint(request: Request) -> dict[str, Any]:
        context.check_token(request)
        payload = dict(await request.json())
        try:
            return reset_tabulation_rebase(context.dataset, store, payload)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.get("/api/glm/models/{model_id}")
    async def model_endpoint(request: Request, model_id: str) -> dict[str, Any]:
        context.check_token(request)
        try:
            return store.model_detail(model_id)
        except ValueError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.post("/api/glm/models/{model_id}/activate")
    async def activate_endpoint(request: Request, model_id: str) -> dict[str, Any]:
        context.check_token(request)
        try:
            manifest = store.activate_model(model_id)
            return {"model": manifest, "config": config_payload()}
        except ValueError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.post("/api/glm/models/{model_id}/rename")
    async def rename_endpoint(request: Request, model_id: str) -> dict[str, Any]:
        context.check_token(request)
        payload = await request.json()
        try:
            manifest = store.rename_model(model_id, str(payload.get("new_model_id") or ""))
            return {"model": manifest, "config": config_payload()}
        except GlmModelNameError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.delete("/api/glm/models/{model_id}")
    async def delete_endpoint(request: Request, model_id: str) -> dict[str, Any]:
        context.check_token(request)
        try:
            manifest = store.delete_model(model_id)
            return {"deleted_model_id": model_id, "model": manifest, "config": config_payload()}
        except ValueError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
