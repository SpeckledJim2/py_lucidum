from __future__ import annotations

from typing import Any

from fastapi import FastAPI, HTTPException, Request

from py_lucidum.app.context import AppContext

from .jobs import GbmJobManager
from .sample import SAMPLE_COLUMN, create_generated_sample, sample_metadata
from .sources import GbmSourceProvider
from .store import GbmModelNameError, GbmModelStore
from .training import MissingGbmDependency, gbm_dependencies
from .trees import tree_detail, tree_summary
from .validation import (
    OFFSET_COLUMN,
    RESPONSE_COLUMN,
    GBM_METRICS,
    GBM_OBJECTIVES,
    default_parameters,
    detect_sample_column,
    feature_rows,
    validate_request,
)


def register(app: FastAPI, context: AppContext) -> None:
    store = GbmModelStore(context.dataset.path)
    context.dataset.register_data_source_provider(GbmSourceProvider(store))
    jobs = GbmJobManager()
    app.state.gbm_store = store
    app.state.gbm_jobs = jobs

    def active_gains() -> dict[str, float]:
        model_id = store.active_model_id()
        if not model_id:
            return {}
        try:
            manifest = store.manifest(model_id)
        except ValueError:
            return {}
        return {
            str(item.get("name")): float(item.get("gain") or 0)
            for item in manifest.get("feature_importance", [])
            if isinstance(item, dict) and item.get("name")
        }

    def active_feature_config() -> list[dict[str, Any]] | None:
        model_id = store.active_model_id()
        if not model_id:
            return None
        try:
            features = store.read_json(store.artifact_path(model_id, "feature_config"), None)
            if isinstance(features, list) and features:
                return [item for item in features if isinstance(item, dict)]
            manifest = store.manifest(model_id)
        except ValueError:
            return None
        importance = manifest.get("feature_importance", [])
        if isinstance(importance, list) and importance:
            return [item for item in importance if isinstance(item, dict)]
        return None

    def parameter_rows() -> list[dict[str, Any]]:
        values: dict[str, Any] = {}
        model_id = store.active_model_id()
        if model_id:
            try:
                stored = store.read_json(store.artifact_path(model_id, "parameters"), {})
            except ValueError:
                stored = {}
            if isinstance(stored, dict):
                values = stored

        rows: list[dict[str, Any]] = []
        seen: set[str] = set()
        for row in default_parameters():
            item = dict(row)
            name = str(item["name"])
            if name in values:
                item["value"] = values[name]
            rows.append(item)
            seen.add(name)

        for name, value in values.items():
            text_name = str(name)
            if text_name not in seen:
                rows.append({"name": text_name, "value": value, "important": False})
        return rows

    def config_payload() -> dict[str, Any]:
        model_features = active_feature_config()
        with context.dataset.lock:
            sample = sample_metadata(context.dataset, store.generated_sample_path)
            sample_reserved = {SAMPLE_COLUMN} if sample.get("source") == "dataset" else set()
            features = feature_rows(
                context.dataset,
                active_gains(),
                model_features=model_features,
                reserved_names=sample_reserved,
            )
            sample_column = detect_sample_column(context.dataset)
        return {
            "tool": "gbm",
            "status": "ready",
            "response": RESPONSE_COLUMN,
            "offset": OFFSET_COLUMN,
            "sample_column": sample_column,
            "sample": sample,
            "parameters": parameter_rows(),
            "parameter_options": {
                "objective": list(GBM_OBJECTIVES),
                "metric": list(GBM_METRICS),
            },
            "features": features,
            "models": store.list_models(),
            "active_model_id": store.active_model_id(),
            "shap_options": [
                {"value": "0", "label": "0"},
                {"value": "10k", "label": "10k"},
                {"value": "100k", "label": "100k"},
                {"value": "all", "label": "All"},
            ],
        }

    @app.get("/api/gbm/summary")
    async def summary_endpoint(request: Request) -> dict[str, Any]:
        context.check_token(request)
        return config_payload()

    @app.get("/api/gbm/config")
    async def config_endpoint(request: Request) -> dict[str, Any]:
        context.check_token(request)
        return config_payload()

    @app.get("/api/gbm/models")
    async def models_endpoint(request: Request) -> dict[str, Any]:
        context.check_token(request)
        return {
            "models": store.list_models(),
            "active_model_id": store.active_model_id(),
        }

    @app.post("/api/gbm/validate")
    async def validate_endpoint(request: Request) -> dict[str, Any]:
        context.check_token(request)
        payload = await request.json()
        return validate_request(context.dataset, payload, generated_sample_path=store.generated_sample_path).as_payload()

    @app.post("/api/gbm/sample")
    async def sample_endpoint(request: Request) -> dict[str, Any]:
        context.check_token(request)
        sample = create_generated_sample(context.dataset, store.generated_sample_path)
        return {"sample": sample, "config": config_payload()}

    @app.post("/api/gbm/train")
    async def train_endpoint(request: Request) -> dict[str, Any]:
        context.check_token(request)
        payload = await request.json()
        try:
            gbm_dependencies()
            if payload.get("create_sample"):
                create_generated_sample(context.dataset, store.generated_sample_path)
            validation = validate_request(context.dataset, payload, generated_sample_path=store.generated_sample_path)
            if not validation.ok:
                raise ValueError("; ".join(validation.errors))
            job = jobs.start(context.dataset, store, payload)
            return job.as_payload()
        except MissingGbmDependency as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

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

    @app.get("/api/gbm/models/{model_id}/trees/{tree_index}")
    async def model_tree_endpoint(request: Request, model_id: str, tree_index: int) -> dict[str, Any]:
        context.check_token(request)
        try:
            return tree_detail(store, model_id, tree_index)
        except ValueError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

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
            manifest = store.activate_model(model_id)
            return {"model": manifest, "config": config_payload()}
        except ValueError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.post("/api/gbm/models/{model_id}/rename")
    async def rename_endpoint(request: Request, model_id: str) -> dict[str, Any]:
        context.check_token(request)
        payload = await request.json()
        try:
            manifest = store.rename_model(model_id, str(payload.get("new_model_id") or ""))
            return {"model": manifest, "config": config_payload()}
        except GbmModelNameError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.delete("/api/gbm/models/{model_id}")
    async def delete_endpoint(request: Request, model_id: str) -> dict[str, Any]:
        context.check_token(request)
        try:
            manifest = store.delete_model(model_id)
            return {"deleted_model_id": model_id, "model": manifest, "config": config_payload()}
        except ValueError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
