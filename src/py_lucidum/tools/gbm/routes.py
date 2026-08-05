from __future__ import annotations

from typing import Any

from fastapi import FastAPI, HTTPException, Request

from py_lucidum.app.context import AppContext
from py_lucidum.app.telemetry import request_operation_id

from .config import GbmConfigBuilder
from .grid import validate_grid_or_request
from .jobs import GbmJobManager
from .sample import create_generated_sample
from .shap import shap_config, shap_plot, stacked_shap_plot
from .sources import GbmSourceProvider
from .store import GbmModelNameError, GbmModelStore
from .training import MissingGbmDependency, gbm_training_dependencies
from .trees import ebm_gain_summary, tree_detail, tree_summary


def register(app: FastAPI, context: AppContext) -> None:
    store = GbmModelStore(context.dataset.path, dataset=context.dataset)
    context.dataset.register_data_source_provider(GbmSourceProvider(store))
    telemetry = getattr(app.state, "telemetry", None)
    jobs = GbmJobManager(telemetry=telemetry)
    app.state.gbm_store = store
    app.state.gbm_jobs = jobs
    config = GbmConfigBuilder(context.dataset, store, lambda: getattr(app.state, "feature_spec", None))

    def fail_operation(operation_id: str | None, exc: Exception | None = None) -> None:
        if telemetry is None:
            return
        try:
            telemetry.finish_operation(
                operation_id,
                status="failed",
                error_type=type(exc).__name__ if exc is not None else "ValidationError",
            )
        except Exception:
            pass

    def operation_phase(
        operation_id: str | None,
        name: str,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        if telemetry is None:
            return
        try:
            telemetry.update_operation_phase(
                operation_id,
                name=name,
                metadata=metadata,
            )
        except Exception:
            pass

    @app.get("/api/gbm/summary")
    async def summary_endpoint(request: Request) -> dict[str, Any]:
        context.check_token(request)
        return config.payload()

    @app.get("/api/gbm/config")
    async def config_endpoint(request: Request) -> dict[str, Any]:
        context.check_token(request)
        return config.payload()

    @app.get("/api/gbm/models")
    async def models_endpoint(request: Request) -> dict[str, Any]:
        context.check_token(request)
        active_model_id = store.active_model_id()
        return {
            "models": store.list_models(active_model_id=active_model_id),
            "active_model_id": active_model_id,
        }

    @app.post("/api/gbm/validate")
    async def validate_endpoint(request: Request) -> dict[str, Any]:
        context.check_token(request)
        operation_id = request_operation_id(request)
        payload = dict(await request.json())
        payload["feature_groupings"] = config.feature_groupings()
        operation_phase(operation_id, "validating_request")
        try:
            validation = validate_grid_or_request(context.dataset, payload, generated_sample_path=store.generated_sample_path)
        except Exception as exc:
            fail_operation(operation_id, exc)
            raise
        if not validation["ok"]:
            fail_operation(operation_id)
        else:
            operation_phase(operation_id, "awaiting_job_request")
        return validation

    @app.post("/api/gbm/sample")
    async def sample_endpoint(request: Request) -> dict[str, Any]:
        context.check_token(request)
        sample = create_generated_sample(context.dataset, store.generated_sample_path)
        return {"sample": sample, "config": config.payload()}

    @app.post("/api/gbm/train")
    async def train_endpoint(request: Request) -> dict[str, Any]:
        context.check_token(request)
        operation_id = request_operation_id(request)
        payload = dict(await request.json())
        payload["feature_groupings"] = config.feature_groupings()
        try:
            gbm_training_dependencies(
                dependency_progress=lambda stage: operation_phase(operation_id, stage),
            )
            if payload.get("create_sample"):
                operation_phase(operation_id, "creating_sample")
                create_generated_sample(context.dataset, store.generated_sample_path)
            operation_phase(operation_id, "validating_request")
            validation = validate_grid_or_request(context.dataset, payload, generated_sample_path=store.generated_sample_path)
            if not validation["ok"]:
                raise ValueError("; ".join(validation["errors"]))
            operation_phase(operation_id, "starting")
            job = jobs.start(context.dataset, store, payload, operation_id=operation_id)
            return job.as_payload()
        except MissingGbmDependency as exc:
            fail_operation(operation_id, exc)
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except ValueError as exc:
            fail_operation(operation_id, exc)
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:
            fail_operation(operation_id, exc)
            raise

    @app.get("/api/gbm/jobs/{job_id}")
    async def job_endpoint(request: Request, job_id: str) -> dict[str, Any]:
        context.check_token(request)
        job = jobs.get(job_id)
        if not job:
            raise HTTPException(status_code=404, detail="Choose a valid GBM job")
        return job.as_payload()

    @app.get("/api/gbm/models/{model_id}/trees")
    async def model_trees_endpoint(request: Request, model_id: str) -> dict[str, Any]:
        context.check_token(request)
        try:
            return tree_summary(store, model_id)
        except ValueError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.get("/api/gbm/models/{model_id}/ebm-gain-summary")
    async def model_ebm_gain_summary_endpoint(request: Request, model_id: str) -> dict[str, Any]:
        context.check_token(request)
        try:
            return ebm_gain_summary(store, model_id)
        except ValueError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.get("/api/gbm/models/{model_id}/trees/{tree_index}")
    async def model_tree_endpoint(request: Request, model_id: str, tree_index: int) -> dict[str, Any]:
        context.check_token(request)
        try:
            return tree_detail(store, model_id, tree_index)
        except ValueError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.get("/api/gbm/models/{model_id}/shap/config")
    async def model_shap_config_endpoint(request: Request, model_id: str) -> dict[str, Any]:
        context.check_token(request)
        try:
            return shap_config(context.dataset, store, model_id, feature_bases=config.feature_bases())
        except ValueError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.post("/api/gbm/models/{model_id}/shap/plot")
    async def model_shap_plot_endpoint(request: Request, model_id: str) -> dict[str, Any]:
        context.check_token(request)
        payload = dict(await request.json())
        payload["feature_bases"] = config.feature_bases()
        try:
            return shap_plot(context.dataset, store, model_id, payload)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/api/gbm/models/{model_id}/shap/stacked")
    async def model_stacked_shap_endpoint(request: Request, model_id: str) -> dict[str, Any]:
        context.check_token(request)
        payload = dict(await request.json())
        try:
            return stacked_shap_plot(context.dataset, store, model_id, payload)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.get("/api/gbm/models/{model_id}")
    async def model_endpoint(request: Request, model_id: str) -> dict[str, Any]:
        context.check_token(request)
        try:
            return store.model_detail(model_id)
        except ValueError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.post("/api/gbm/models/{model_id}/activate")
    async def activate_endpoint(request: Request, model_id: str) -> dict[str, Any]:
        context.check_token(request)
        try:
            with store.model_state_lock:
                manifest = store.activate_model(model_id)
                return {"model": manifest, "config": config.payload()}
        except ValueError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.post("/api/gbm/models/{model_id}/rename")
    async def rename_endpoint(request: Request, model_id: str) -> dict[str, Any]:
        context.check_token(request)
        payload = await request.json()
        try:
            with store.model_state_lock:
                manifest = store.rename_model(model_id, str(payload.get("new_model_id") or ""))
                return {"model": manifest, "config": config.payload()}
        except GbmModelNameError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.delete("/api/gbm/models/{model_id}")
    async def delete_endpoint(request: Request, model_id: str) -> dict[str, Any]:
        context.check_token(request)
        try:
            with store.model_state_lock:
                manifest = store.delete_model(model_id)
                return {"deleted_model_id": model_id, "model": manifest, "config": config.payload()}
        except ValueError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
