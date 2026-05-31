from __future__ import annotations

from typing import Any

import duckdb
from fastapi import FastAPI, HTTPException, Request

from py_lucidum.app.context import AppContext
from py_lucidum.core import json_number, sql_literal

from .grid import validate_grid_or_request
from .jobs import GbmJobManager
from .sample import SAMPLE_COLUMN, create_generated_sample, sample_metadata
from .shap import shap_config, shap_plot, stacked_shap_plot
from .sources import GbmSourceProvider
from .store import GbmModelNameError, GbmModelStore
from .training import MissingGbmDependency, gbm_dependencies
from .trees import ebm_gain_summary, tree_detail, tree_summary
from .validation import (
    OFFSET_COLUMN,
    RESPONSE_COLUMN,
    DATA_SAMPLE_STRATEGIES,
    GBM_METRICS,
    GBM_OBJECTIVES,
    DEFAULT_TRAINING_MODE,
    default_parameters,
    detect_sample_column,
    ebm_available,
    feature_rows,
    normalise_training_mode,
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

    def active_shap_importance(model_id: str) -> dict[str, float]:
        path = store.artifact_path(model_id, "shap_summary")
        if not path.exists():
            return {}
        try:
            with context.dataset.lock:
                columns = {
                    str(row[0])
                    for row in context.dataset.con.execute(
                        f"DESCRIBE SELECT * FROM read_parquet({sql_literal(str(path))})"
                    ).fetchall()
                }
                if not {"feature", "mean_abs_shap"}.issubset(columns):
                    return {}
                records = context.dataset.con.execute(
                    f"""
SELECT feature, mean_abs_shap
FROM read_parquet({sql_literal(str(path))})
WHERE feature IS NOT NULL
"""
                ).fetchall()
        except duckdb.Error:
            return {}
        values: dict[str, float] = {}
        for feature, value in records:
            name = str(feature or "").strip()
            number = json_number(value)
            if name and number is not None:
                values[name] = float(number)
        return values

    def feature_config_with_shap(features: list[dict[str, Any]], model_id: str) -> list[dict[str, Any]]:
        shap_importance = active_shap_importance(model_id)
        if not shap_importance:
            return features
        enriched: list[dict[str, Any]] = []
        for item in features:
            row = dict(item)
            name = str(row.get("name") or "").strip()
            if name in shap_importance and json_number(row.get("mean_abs_shap")) is None:
                row["mean_abs_shap"] = shap_importance[name]
            enriched.append(row)
        return enriched

    def active_feature_config() -> list[dict[str, Any]] | None:
        model_id = store.active_model_id()
        if not model_id:
            return None
        try:
            features = store.read_json(store.artifact_path(model_id, "feature_config"), None)
            if isinstance(features, list) and features:
                return feature_config_with_shap([item for item in features if isinstance(item, dict)], model_id)
            manifest = store.manifest(model_id)
        except ValueError:
            return None
        importance = manifest.get("feature_importance", [])
        if isinstance(importance, list) and importance:
            return feature_config_with_shap([item for item in importance if isinstance(item, dict)], model_id)
        return None

    def active_training_mode() -> str:
        model_id = store.active_model_id()
        if not model_id:
            return DEFAULT_TRAINING_MODE
        try:
            manifest = store.manifest(model_id)
        except ValueError:
            return DEFAULT_TRAINING_MODE
        return normalise_training_mode(manifest.get("training_mode"))

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
            if text_name == "training_mode":
                continue
            if text_name not in seen:
                rows.append({"name": text_name, "value": value, "important": False})
        return rows

    def feature_spec_payload() -> dict[str, Any]:
        spec = getattr(app.state, "feature_spec", None)
        return spec if isinstance(spec, dict) else {"rows": [], "scenarios": []}

    def feature_groupings() -> dict[str, str]:
        rows = feature_spec_payload().get("rows", [])
        if not isinstance(rows, list):
            return {}
        return {
            str(row.get("feature")): str(row.get("grouping") or "")
            for row in rows
            if isinstance(row, dict) and row.get("feature")
        }

    def feature_interaction_groupings(current_groupings: dict[str, str] | None = None) -> list[str]:
        values = (current_groupings if current_groupings is not None else feature_groupings()).values()
        return sorted({str(grouping).strip() for grouping in values if str(grouping).strip()}, key=str.lower)

    def feature_scenarios() -> list[dict[str, Any]]:
        scenarios = feature_spec_payload().get("scenarios", [])
        if not isinstance(scenarios, list):
            return []
        return [
            {
                "name": str(scenario.get("name") or ""),
                "features": [
                    str(feature)
                    for feature in scenario.get("features", [])
                    if str(feature).strip()
                ],
            }
            for scenario in scenarios
            if isinstance(scenario, dict) and scenario.get("name")
        ]

    def active_feature_scenario(current_scenarios: list[dict[str, Any]]) -> dict[str, Any] | None:
        model_id = store.active_model_id()
        if not model_id:
            return None
        try:
            manifest = store.manifest(model_id)
        except ValueError:
            return None
        stored = manifest.get("feature_scenario")
        if not isinstance(stored, dict):
            return None
        name = str(stored.get("name") or "").strip()
        if not name:
            return None
        stored_features = scenario_feature_list(stored.get("features"))
        current = {scenario["name"]: scenario for scenario in current_scenarios}.get(name)
        payload: dict[str, Any] = {"name": name, "features": stored_features}
        if not current:
            payload["status"] = "missing"
            return payload
        current_features = scenario_feature_list(current.get("features"))
        if scenario_feature_set(stored_features) == scenario_feature_set(current_features):
            payload["status"] = "current"
        else:
            payload["status"] = "stale"
            payload["current_features"] = current_features
        return payload

    def scenario_feature_list(raw_features: Any) -> list[str]:
        if not isinstance(raw_features, list):
            return []
        features: list[str] = []
        seen: set[str] = set()
        for item in raw_features:
            feature = str(item or "").strip()
            if feature and feature not in seen:
                features.append(feature)
                seen.add(feature)
        return features

    def scenario_feature_set(features: list[str]) -> set[str]:
        return {feature for feature in features if feature}

    def active_feature_interaction_constraints(current_groupings: dict[str, str], valid_groupings: list[str]) -> dict[str, Any] | None:
        model_id = store.active_model_id()
        if not model_id:
            return None
        try:
            manifest = store.manifest(model_id)
        except ValueError:
            return None
        stored = manifest.get("feature_interaction_constraints")
        if not isinstance(stored, dict):
            return None
        groups = normalise_interaction_constraint_groups(stored.get("groups"))
        if not groups:
            return None
        current_grouping_set = set(valid_groupings)
        payload_groups: list[dict[str, Any]] = []
        for group in groups:
            grouping = group["grouping"]
            features = group["features"]
            payload: dict[str, Any] = {"grouping": grouping, "features": features}
            if grouping not in current_grouping_set:
                payload["status"] = "missing"
            elif any(current_groupings.get(feature, "") != grouping for feature in features):
                payload["status"] = "stale"
            else:
                payload["status"] = "current"
            payload_groups.append(payload)
        return {
            "groupings": [group["grouping"] for group in payload_groups],
            "groups": payload_groups,
        }

    def normalise_interaction_constraint_groups(raw_groups: Any) -> list[dict[str, Any]]:
        if not isinstance(raw_groups, list):
            return []
        groups: list[dict[str, Any]] = []
        seen_groupings: set[str] = set()
        for raw_group in raw_groups:
            if not isinstance(raw_group, dict):
                continue
            grouping = str(raw_group.get("grouping") or "").strip()
            if not grouping or grouping in seen_groupings:
                continue
            features = scenario_feature_list(raw_group.get("features"))
            if not features:
                continue
            groups.append({"grouping": grouping, "features": features})
            seen_groupings.add(grouping)
        return groups

    def config_payload() -> dict[str, Any]:
        model_features = active_feature_config()
        scenarios = feature_scenarios()
        current_feature_groupings = feature_groupings()
        interaction_groupings = feature_interaction_groupings(current_feature_groupings)
        with context.dataset.lock:
            sample = sample_metadata(context.dataset, store.generated_sample_path)
            sample_reserved = {SAMPLE_COLUMN} if sample.get("source") == "dataset" else set()
            features = feature_rows(
                context.dataset,
                active_gains(),
                model_features=model_features,
                reserved_names=sample_reserved,
                feature_groupings=current_feature_groupings,
            )
            sample_column = detect_sample_column(context.dataset)
            can_use_ebm = ebm_available(context.dataset, generated_sample_path=store.generated_sample_path)
        return {
            "tool": "gbm",
            "status": "ready",
            "response": RESPONSE_COLUMN,
            "offset": OFFSET_COLUMN,
            "sample_column": sample_column,
            "sample": sample,
            "training_mode": active_training_mode(),
            "ebm_available": can_use_ebm,
            "parameters": parameter_rows(),
            "parameter_options": {
                "objective": sorted(GBM_OBJECTIVES),
                "metric": sorted(GBM_METRICS),
                "data_sample_strategy": list(DATA_SAMPLE_STRATEGIES),
            },
            "features": features,
            "feature_scenarios": scenarios,
            "active_feature_scenario": active_feature_scenario(scenarios),
            "feature_interaction_groupings": interaction_groupings,
            "active_feature_interaction_constraints": active_feature_interaction_constraints(current_feature_groupings, interaction_groupings),
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
        payload = dict(await request.json())
        payload["feature_groupings"] = feature_groupings()
        return validate_grid_or_request(context.dataset, payload, generated_sample_path=store.generated_sample_path)

    @app.post("/api/gbm/sample")
    async def sample_endpoint(request: Request) -> dict[str, Any]:
        context.check_token(request)
        sample = create_generated_sample(context.dataset, store.generated_sample_path)
        return {"sample": sample, "config": config_payload()}

    @app.post("/api/gbm/train")
    async def train_endpoint(request: Request) -> dict[str, Any]:
        context.check_token(request)
        payload = dict(await request.json())
        payload["feature_groupings"] = feature_groupings()
        try:
            gbm_dependencies()
            if payload.get("create_sample"):
                create_generated_sample(context.dataset, store.generated_sample_path)
            validation = validate_grid_or_request(context.dataset, payload, generated_sample_path=store.generated_sample_path)
            if not validation["ok"]:
                raise ValueError("; ".join(validation["errors"]))
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
            return shap_config(context.dataset, store, model_id)
        except ValueError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.post("/api/gbm/models/{model_id}/shap/plot")
    async def model_shap_plot_endpoint(request: Request, model_id: str) -> dict[str, Any]:
        context.check_token(request)
        payload = dict(await request.json())
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
