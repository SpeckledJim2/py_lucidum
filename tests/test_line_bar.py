from __future__ import annotations

import asyncio
import importlib.util
import json
import math
import os
import subprocess
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any
from unittest.mock import patch

import duckdb

from py_lucidum.app import create_app
from py_lucidum.core import Dataset, ModelPredictionSource, load_kpis, load_saved_filters, sql_literal
from py_lucidum.query import Dataset as LegacyDataset
from py_lucidum.query import build_x_sql
from py_lucidum.tools.gbm.sources import GbmSourceProvider
from py_lucidum.tools.gbm.store import GbmModelStore
from py_lucidum.tools.glm.store import GlmModelStore, GlmSourceProvider
from py_lucidum.tools.glm.training import MissingGlmDependency, glm_dependencies, train_model
from py_lucidum.tools.line_bar.query import apply_transform, chart, normalise_quantile_count


class PredictionSidecarProvider:
    def __init__(self, sources: dict[str, ModelPredictionSource]):
        self.sources = sources

    def has_source(self, source_id: str) -> bool:
        return source_id in self.sources

    def relation_sql(self, source_id: str) -> str:
        source = self.sources[source_id]
        return source.relation_sql

    def prediction_source(self, source_id: str) -> ModelPredictionSource | None:
        return self.sources.get(source_id)


def asgi_post_json(app: Any, path: str, payload: dict[str, Any]) -> tuple[int, dict[str, str], bytes]:
    messages: list[dict[str, Any]] = []
    body = json.dumps(payload).encode("utf-8")

    async def receive() -> dict[str, Any]:
        return {"type": "http.request", "body": body, "more_body": False}

    async def send(message: dict[str, Any]) -> None:
        messages.append(message)

    scope = {
        "type": "http",
        "asgi": {"version": "3.0", "spec_version": "2.3"},
        "http_version": "1.1",
        "method": "POST",
        "scheme": "http",
        "path": path,
        "raw_path": path.encode("ascii"),
        "query_string": b"",
        "headers": [(b"content-type", b"application/json")],
        "client": ("127.0.0.1", 12345),
        "server": ("testserver", 80),
    }
    asyncio.run(app(scope, receive, send))

    start = next(message for message in messages if message["type"] == "http.response.start")
    response_body = b"".join(message.get("body", b"") for message in messages if message["type"] == "http.response.body")
    headers = {key.decode("latin-1").lower(): value.decode("latin-1") for key, value in start["headers"]}
    return start["status"], headers, response_body


def asgi_get(app: Any, path: str) -> tuple[int, dict[str, str], bytes]:
    messages: list[dict[str, Any]] = []

    async def receive() -> dict[str, Any]:
        return {"type": "http.request", "body": b"", "more_body": False}

    async def send(message: dict[str, Any]) -> None:
        messages.append(message)

    scope = {
        "type": "http",
        "asgi": {"version": "3.0", "spec_version": "2.3"},
        "http_version": "1.1",
        "method": "GET",
        "scheme": "http",
        "path": path,
        "raw_path": path.encode("ascii"),
        "query_string": b"",
        "headers": [],
        "client": ("127.0.0.1", 12345),
        "server": ("testserver", 80),
    }
    asyncio.run(app(scope, receive, send))

    start = next(message for message in messages if message["type"] == "http.response.start")
    response_body = b"".join(message.get("body", b"") for message in messages if message["type"] == "http.response.body")
    headers = {key.decode("latin-1").lower(): value.decode("latin-1") for key, value in start["headers"]}
    return start["status"], headers, response_body


class LineBarToolTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.data_path = self.root / "sample.csv"
        self.data_path.write_text(
            "YoungestDriverAge,UseofVan,QuoteDate,Gross.Weight,Actual,Expected,Weight\n"
            "30,Social,2024-01-01,2500,100,90,10\n"
            "45,Social,2024-01-02,3500,200,210,20\n"
            "50,Business,2024-02-01,4000,300,290,30\n"
            "60,Business,2024-02-20,4500,400,410,40\n",
            encoding="utf-8",
        )
        self.filters_path = self.root / "filter_spec.csv"
        self.filters_path.write_text(
            "theme,name,expression\nDriver age,Older drivers,YoungestDriverAge > 40\n",
            encoding="utf-8",
        )
        self.kpis_path = self.root / "kpi_spec.csv"
        self.kpis_path.write_text(
            "group,name,actual,denominator,decimals,format\n"
            "Pricing,Actual average,Actual,N,2,number\n",
            encoding="utf-8",
        )
        self.features_path = self.root / "feature_spec.csv"
        self.features_path.write_text(
            "Feature,Grouping,scenario1\n"
            "YoungestDriverAge,DRIVER,feature\n",
            encoding="utf-8",
        )

    def request(self, filter_expression: str = "") -> dict:
        return {
            "x": "UseofVan",
            "bandWidth": "0",
            "dateBucket": "none",
            "lowGroup": "0",
            "sort": "alpha",
            "sigma": 0,
            "transform": "none",
            "filter": filter_expression,
            "denominator": "__none__",
            "maxGroups": 10000,
            "responses": [
                {"label": "Actual", "numerator": "Actual"},
                {"label": "Expected", "numerator": "Expected"},
            ],
        }

    def write_active_gbm_for_shap_ribbons(
        self,
        *,
        objective: str = "poisson",
        predictions: list[tuple[int, float]] | None = None,
        shap_values: list[tuple[int, float, float]] | None = None,
    ) -> GbmModelStore:
        store = GbmModelStore(self.data_path)
        model_dir = store.create_model_dir(f"{objective}-ribbons")
        model_id = model_dir.name
        store.write_json(
            model_dir / "manifest.json",
            {
                "model_id": model_id,
                "label": f"{objective} ribbons",
                "created_at": "2026-06-04T00:00:00Z",
                "objective": objective,
                "metric": objective,
                "response_column": "Actual",
                "offset_column": "Weight",
                "sources": {
                    "predictions": f"gbm:{model_id}:predictions",
                    "shap_long": f"gbm:{model_id}:shap_long",
                    "shap_summary": f"gbm:{model_id}:shap_summary",
                },
            },
        )
        con = duckdb.connect(database=":memory:")
        try:
            prediction_rows = predictions or [(1, 100.0), (2, 200.0), (3, 300.0), (4, 400.0)]
            prediction_values = ", ".join(f"({row_id}, {value})" for row_id, value in prediction_rows)
            con.execute(
                f"""
COPY (
  SELECT *
  FROM (VALUES {prediction_values}) AS rows(__lucidum_row_id, gbm_prediction)
) TO {sql_literal(str(model_dir / "predictions.parquet"))} (FORMAT PARQUET)
"""
            )
            if shap_values is None:
                shap_values = (
                    [(1, 0.0, 0.0), (2, 0.0, 0.0), (3, 1.0, 1.0), (4, 1.0, 1.0)]
                    if objective == "poisson"
                    else [(1, 10.0, 10.0), (2, 10.0, 10.0), (3, 20.0, 20.0), (4, 20.0, 20.0)]
                )
            shap_rows = ", ".join(f"({row_id}, {use_of_van}, {youngest_driver_age})" for row_id, use_of_van, youngest_driver_age in shap_values)
            con.execute(
                f"""
COPY (
  SELECT *
  FROM (VALUES {shap_rows}) AS rows(__lucidum_row_id, UseofVan, YoungestDriverAge)
) TO {sql_literal(str(model_dir / "shap_values.parquet"))} (FORMAT PARQUET)
"""
            )
            con.execute(
                f"""
COPY (
  SELECT 'UseofVan' AS feature, 1.0 AS mean_abs_shap, 0.0 AS mean_shap, 4 AS row_count
  UNION ALL SELECT 'YoungestDriverAge', 1.0, 0.0, 4
) TO {sql_literal(str(model_dir / "shap_summary.parquet"))} (FORMAT PARQUET)
"""
            )
        finally:
            con.close()
        store.activate_model(model_id)
        return store

    def require_glm_dependencies(self) -> None:
        try:
            glm_dependencies()
        except MissingGlmDependency as exc:
            self.skipTest(str(exc))

    def write_active_glm_for_overlay(
        self,
        dataset: Dataset,
        *,
        formula: str = "YoungestDriverAge",
        denominator_column: str = "",
    ) -> tuple[GlmModelStore, str]:
        self.require_glm_dependencies()
        store = GlmModelStore(self.data_path)
        result = train_model(
            dataset,
            store,
            {
                "label": "overlay glm",
                "formula": formula,
                "response_column": "Actual",
                "denominator_column": denominator_column,
                "family": "normal",
                "training_scope": "all",
            },
        )
        model_id = str(result["model_id"])
        dataset.register_data_source_provider(GlmSourceProvider(store))
        return store, model_id

    def write_active_gbm_importance_model(self, *, with_shap: bool = True) -> GbmModelStore:
        store = GbmModelStore(self.data_path)
        model_dir = store.create_model_dir("importance-gbm")
        store.write_json(
            model_dir / "manifest.json",
            {
                "model_id": "importance-gbm",
                "label": "Importance GBM",
                "created_at": "2026-06-07T00:00:00Z",
                "objective": "poisson",
                "metric": "poisson",
                "response_column": "Actual",
                "offset_column": "Weight",
                "feature_importance": [
                    {"name": "UseofVan", "kind": "categorical", "gain": 3.0},
                    {"name": "YoungestDriverAge", "kind": "integer", "gain": 2.0},
                ],
            },
        )
        store.write_json(
            model_dir / "feature_config.json",
            [
                {"name": "UseofVan", "kind": "categorical", "include": True, "gain": 3.0},
                {"name": "YoungestDriverAge", "kind": "integer", "include": True, "gain": 2.0},
            ],
        )
        if with_shap:
            con = duckdb.connect(database=":memory:")
            try:
                con.execute(
                    f"""
COPY (
  SELECT 'YoungestDriverAge' AS feature, 2.5 AS mean_abs_shap, 0.0 AS mean_shap, 4 AS row_count
  UNION ALL SELECT 'UseofVan', 0.5, 0.0, 4
) TO {sql_literal(str(model_dir / "shap_summary.parquet"))} (FORMAT PARQUET)
"""
                )
            finally:
                con.close()
        store.activate_model("importance-gbm")
        return store

    def write_active_glm_importance_model(self, *, with_importance: bool = True) -> GlmModelStore:
        store = GlmModelStore(self.data_path)
        model_dir = store.create_model_dir("importance-glm")
        manifest: dict[str, Any] = {
            "model_id": "importance-glm",
            "label": "Importance GLM",
            "tool": "glm",
            "created_at": "2026-06-07T00:00:00Z",
            "family": "normal",
            "link": "auto",
            "response_column": "Actual",
            "denominator_column": "",
            "source_columns": ["YoungestDriverAge", "UseofVan", "QuoteDate", "Gross.Weight", "Actual", "Expected", "Weight"],
            "sources": {"predictions": "glm:importance-glm:predictions"},
        }
        if with_importance:
            rows = [
                {
                    "feature": "UseofVan",
                    "importance": 0.75,
                    "term_count": 2,
                    "metric": "weighted_mean_abs_centered_linear_predictor_contribution",
                },
                {
                    "feature": "YoungestDriverAge",
                    "importance": 0.25,
                    "term_count": 1,
                    "metric": "weighted_mean_abs_centered_linear_predictor_contribution",
                },
            ]
            manifest["feature_importance"] = rows
            manifest["feature_importance_metric"] = {
                "name": "weighted_mean_abs_centered_linear_predictor_contribution",
                "label": "GLM eta MAD",
                "interaction_allocation": "split_evenly",
            }
            con = duckdb.connect(database=":memory:")
            try:
                con.execute(
                    f"""
COPY (
  SELECT 'UseofVan' AS feature, 0.75 AS importance, 2 AS term_count, 'weighted_mean_abs_centered_linear_predictor_contribution' AS metric
  UNION ALL SELECT 'YoungestDriverAge', 0.25, 1, 'weighted_mean_abs_centered_linear_predictor_contribution'
) TO {sql_literal(str(model_dir / "feature_importance.parquet"))} (FORMAT PARQUET)
"""
                )
            finally:
                con.close()
        store.write_json(model_dir / "manifest.json", manifest)
        store.activate_model("importance-glm")
        return store

    def dataset_with_gbm_ribbons(
        self,
        *,
        objective: str = "poisson",
        predictions: list[tuple[int, float]] | None = None,
        shap_values: list[tuple[int, float, float]] | None = None,
    ) -> Dataset:
        store = self.write_active_gbm_for_shap_ribbons(objective=objective, predictions=predictions, shap_values=shap_values)
        dataset = Dataset(self.data_path)
        dataset.register_data_source_provider(GbmSourceProvider(store))
        return dataset

    def test_app_registers_line_bar_routes_and_saved_filters(self) -> None:
        app = create_app(
            self.data_path,
            token="dev-token",
            defaults={"denominator": "Weight"},
            filters_path=self.filters_path,
            kpis_path=self.kpis_path,
            features_path=self.features_path,
            tools=["line_bar"],
        )
        paths = {route.path for route in app.routes}

        self.assertIn("/api/chart", paths)
        self.assertIn("/api/line-bar/chart", paths)
        self.assertIn("/api/column-profile/summary", paths)
        self.assertIn("/api/schema", paths)
        self.assertIn("/api/shutdown", paths)
        self.assertIn("/static", paths)
        self.assertEqual(app.state.enabled_tools, ["column_profile", "line_bar"])
        self.assertEqual(app.state.defaults["denominator"], "Weight")
        self.assertEqual(
            app.state.saved_filters,
            [{"theme": "Driver age", "name": "Older drivers", "expression": "YoungestDriverAge > 40"}],
        )
        self.assertEqual(
            app.state.kpis,
            [{"group": "Pricing", "name": "Actual average", "actual": "Actual", "denominator": "__none__", "decimals": 2, "format": "number"}],
        )
        self.assertEqual(
            app.state.feature_spec["scenarios"],
            [{"name": "scenario1", "features": ["YoungestDriverAge"]}],
        )

    def test_chart_endpoint_includes_duckdb_timing(self) -> None:
        app = create_app(self.data_path, token="", tools=["line_bar"], use_saved_filters=False, use_kpis=False)

        status, _, body = asgi_post_json(app, "/api/chart", self.request())
        payload = json.loads(body)

        self.assertEqual(status, 200)
        self.assertEqual(payload["x"], "UseofVan")
        self.assertEqual(payload["source"], "dataset")
        self.assertIn("rows", payload)
        self.assertIn("response_summaries", payload)
        self.assertIsInstance(payload["timings"]["duckdb_ns"], int)
        self.assertGreaterEqual(payload["timings"]["duckdb_ns"], 0)
        self.assertIsInstance(payload["timings"]["duckdb_ms"], int)
        self.assertGreaterEqual(payload["timings"]["duckdb_ms"], 0)

    def test_schema_exposes_feature_bases_from_feature_spec(self) -> None:
        self.features_path.write_text(
            "Feature,Grouping,Base,scenario1\n"
            "YoungestDriverAge,DRIVER,40,feature\n"
            "UseofVan,VEHICLE,Business,\n",
            encoding="utf-8",
        )
        app = create_app(
            self.data_path,
            token="",
            tools=["line_bar"],
            use_saved_filters=False,
            use_kpis=False,
            features_path=self.features_path,
        )

        status, _, body = asgi_get(app, "/api/schema")
        payload = json.loads(body)

        self.assertEqual(status, 200)
        self.assertEqual(payload["feature_bases"], {"YoungestDriverAge": "40", "UseofVan": "Business"})

    def test_line_bar_zero_transform_uses_categorical_base_for_actual_and_expected(self) -> None:
        app = create_app(self.data_path, token="", tools=["line_bar"], use_saved_filters=False, use_kpis=False)
        request = self.request()
        request.update({"transform": "zero", "base": "Business"})

        status, _, body = asgi_post_json(app, "/api/chart", request)
        payload = json.loads(body)
        by_x = {row["x"]: row for row in payload["rows"]}

        self.assertEqual(status, 200)
        self.assertEqual(payload["transform"]["reference"], "base")
        self.assertEqual(payload["transform"]["base_x"], "Business")
        self.assertEqual(payload["transform"]["values"], [350, 350])
        self.assertAlmostEqual(by_x["Business"]["resp0"], 0)
        self.assertAlmostEqual(by_x["Business"]["resp1"], 0)
        self.assertAlmostEqual(by_x["Social"]["resp0"], -200)
        self.assertAlmostEqual(by_x["Social"]["resp1"], -200)

    def test_line_bar_one_transform_uses_numeric_band_containing_base(self) -> None:
        app = create_app(self.data_path, token="", tools=["line_bar"], use_saved_filters=False, use_kpis=False)
        request = self.request()
        request.update({"x": "YoungestDriverAge", "bandWidth": "10", "transform": "one", "base": "40"})

        status, _, body = asgi_post_json(app, "/api/chart", request)
        payload = json.loads(body)
        by_x = {row["x"]: row for row in payload["rows"]}

        self.assertEqual(status, 200)
        self.assertEqual(payload["transform"]["reference"], "base")
        self.assertEqual(payload["transform"]["base_x"], "40")
        self.assertEqual(payload["transform"]["values"], [200, 210])
        self.assertAlmostEqual(by_x["40"]["resp0"], 1)
        self.assertAlmostEqual(by_x["40"]["resp1"], 1)
        self.assertAlmostEqual(by_x["30"]["resp0"], 0.5)
        self.assertAlmostEqual(by_x["30"]["resp1"], 90 / 210)
        self.assertAlmostEqual(by_x["50"]["resp0"], 300 / 200)
        self.assertAlmostEqual(by_x["50"]["resp1"], 290 / 210)
        self.assertAlmostEqual(by_x["60"]["resp0"], 400 / 200)
        self.assertAlmostEqual(by_x["60"]["resp1"], 410 / 210)

    def test_line_bar_base_transform_uses_expected_reference_for_sigma_bounds(self) -> None:
        rows = [
            {
                "x": "Base",
                "x_sort": "Base",
                "volume": 1,
                "resp0": 100,
                "resp0_num": 100,
                "resp0_den": 1,
                "resp1": 50,
                "resp1_num": 50,
                "resp1_den": 1,
                "sigma_se": 5,
                "valid_folds": 3,
            },
            {
                "x": "Other",
                "x_sort": "Other",
                "volume": 1,
                "resp0": 150,
                "resp0_num": 150,
                "resp0_den": 1,
                "resp1": 40,
                "resp1_num": 40,
                "resp1_den": 1,
                "sigma_se": 2,
                "valid_folds": 3,
            },
        ]
        warnings: list[str] = []

        display, metadata = apply_transform(
            rows,
            [{"label": "Actual", "numerator": "Actual"}, {"label": "Expected", "numerator": "Expected"}],
            "zero",
            2,
            warnings,
            x_kind="categorical",
            base="Base",
            band_width="0",
        )
        by_x = {row["x"]: row for row in display}

        self.assertEqual(warnings, [])
        self.assertEqual(metadata["values"], [100, 50])
        self.assertAlmostEqual(by_x["Base"]["resp0"], 0)
        self.assertAlmostEqual(by_x["Base"]["resp1"], 0)
        self.assertAlmostEqual(by_x["Base"]["resp1_low"], -10)
        self.assertAlmostEqual(by_x["Base"]["resp1_high"], 10)
        self.assertAlmostEqual(by_x["Other"]["resp0"], 50)
        self.assertAlmostEqual(by_x["Other"]["resp1"], -10)
        self.assertAlmostEqual(by_x["Other"]["resp1_low"], -14)
        self.assertAlmostEqual(by_x["Other"]["resp1_high"], -6)

    def test_line_bar_declared_base_falls_back_to_average_when_unusable(self) -> None:
        rows = [
            {"x": "Base", "x_sort": "Base", "volume": 1, "resp0": 0, "resp0_num": 0, "resp0_den": 1},
            {"x": "Other", "x_sort": "Other", "volume": 1, "resp0": 4, "resp0_num": 4, "resp0_den": 1},
        ]
        warnings: list[str] = []

        display, metadata = apply_transform(
            rows,
            [{"label": "Actual", "numerator": "Actual"}],
            "one",
            0,
            warnings,
            x_kind="categorical",
            base="Base",
            band_width="0",
        )

        self.assertEqual(metadata["values"], [2])
        self.assertEqual(display[0]["resp0"], 0)
        self.assertEqual(display[1]["resp0"], 2)
        self.assertIn("no usable Actual response reference", warnings[0])

    def test_dataset_exposes_default_data_source_contract(self) -> None:
        dataset = Dataset(self.data_path)
        sources = dataset.data_sources()

        self.assertEqual(len(sources), 1)
        self.assertEqual(sources[0]["id"], "dataset")
        self.assertEqual(sources[0]["kind"], "dataset")
        self.assertEqual(sources[0]["row_count"], 4)
        self.assertEqual([column["name"] for column in sources[0]["columns"]], [
            "YoungestDriverAge",
            "UseofVan",
            "QuoteDate",
            "Gross.Weight",
            "Actual",
            "Expected",
            "Weight",
        ])

        app = create_app(self.data_path, token="", tools=["line_bar"], use_saved_filters=False, use_kpis=False)
        status, _, body = asgi_get(app, "/api/schema")
        payload = json.loads(body)

        self.assertEqual(status, 200)
        self.assertEqual(payload["data_sources"][0]["id"], "dataset")
        self.assertTrue(all(column["band_suggestion"] is None for column in payload["columns"]))
        self.assertTrue(all(column["band_suggestion"] is None for column in payload["data_sources"][0]["columns"]))

    def test_schema_does_not_calculate_eager_band_suggestions(self) -> None:
        app = create_app(self.data_path, token="", tools=["line_bar"], use_saved_filters=False, use_kpis=False)

        with patch.object(Dataset, "band_suggestions_for_relation", side_effect=AssertionError("eager banding")):
            status, _, body = asgi_get(app, "/api/schema")
        payload = json.loads(body)

        self.assertEqual(status, 200)
        self.assertEqual(payload["columns"][0]["band_suggestion"], None)

    def test_lazy_banding_suggestion_endpoint_respects_filters(self) -> None:
        app = create_app(self.data_path, token="", tools=["line_bar"], use_saved_filters=False, use_kpis=False)

        status, _, body = asgi_post_json(app, "/api/banding/suggestion", {"feature": "Actual"})
        filtered_status, _, filtered_body = asgi_post_json(
            app,
            "/api/banding/suggestion",
            {"feature": "Actual", "filter": "UseofVan = 'Business'"},
        )

        payload = json.loads(body)
        filtered_payload = json.loads(filtered_body)
        self.assertEqual(status, 200)
        self.assertEqual(filtered_status, 200)
        self.assertEqual(payload["source"], "dataset")
        self.assertEqual(payload["feature"], "Actual")
        self.assertGreater(payload["band_suggestion"], 1)
        self.assertEqual(filtered_payload["band_suggestion"], 1)
        self.assertIn("duckdb_ms", payload["timings"])

    def test_lazy_banding_suggestion_errors_are_actionable(self) -> None:
        app = create_app(self.data_path, token="", tools=["line_bar"], use_saved_filters=False, use_kpis=False)

        source_status, _, source_body = asgi_post_json(
            app,
            "/api/banding/suggestion",
            {"source": "missing", "feature": "Actual"},
        )
        feature_status, _, feature_body = asgi_post_json(app, "/api/banding/suggestion", {"feature": "Missing"})
        kind_status, _, kind_body = asgi_post_json(app, "/api/banding/suggestion", {"feature": "UseofVan"})
        filter_status, _, filter_body = asgi_post_json(
            app,
            "/api/banding/suggestion",
            {"feature": "Actual", "filter": "Missing > 0"},
        )

        self.assertEqual(source_status, 400)
        self.assertIn("valid data source", json.loads(source_body)["detail"])
        self.assertEqual(feature_status, 400)
        self.assertIn("valid feature", json.loads(feature_body)["detail"])
        self.assertEqual(kind_status, 400)
        self.assertIn("numeric feature", json.loads(kind_body)["detail"])
        self.assertEqual(filter_status, 400)
        self.assertIn("Invalid filter", json.loads(filter_body)["detail"])

    def test_feature_importance_endpoint_returns_active_gbm_and_glm_feature_rows(self) -> None:
        self.write_active_gbm_importance_model()
        self.write_active_glm_importance_model()
        app = create_app(self.data_path, token="", tools=["gbm", "line_bar"], use_saved_filters=False, use_kpis=False)

        status, _, body = asgi_get(app, "/api/line-bar/feature-importance")
        payload = json.loads(body)

        self.assertEqual(status, 200)
        self.assertTrue(payload["has_importance"])
        self.assertEqual(payload["models"]["gbm"]["model_id"], "importance-gbm")
        self.assertEqual(payload["models"]["gbm"]["metric"], "mean_abs_shap")
        self.assertEqual([row["feature"] for row in payload["models"]["gbm"]["rows"]], ["YoungestDriverAge", "UseofVan"])
        self.assertEqual(payload["models"]["glm"]["model_id"], "importance-glm")
        self.assertEqual(payload["models"]["glm"]["metric_label"], "GLM eta MAD")
        self.assertEqual([row["feature"] for row in payload["models"]["glm"]["rows"]], ["UseofVan", "YoungestDriverAge"])
        self.assertIn("Gross.Weight", [row["name"] for row in payload["dataset_features"]])

    def test_feature_importance_endpoint_reports_missing_active_models(self) -> None:
        app = create_app(self.data_path, token="", tools=["gbm", "line_bar"], use_saved_filters=False, use_kpis=False)

        status, _, body = asgi_get(app, "/api/line-bar/feature-importance")
        payload = json.loads(body)

        self.assertEqual(status, 200)
        self.assertFalse(payload["has_importance"])
        self.assertIn("No active GBM is available.", payload["messages"])
        self.assertIn("No active GLM is available.", payload["messages"])

    def test_feature_importance_endpoint_reports_old_glm_without_importance_sidecar(self) -> None:
        self.write_active_glm_importance_model(with_importance=False)
        app = create_app(self.data_path, token="", tools=["glm", "line_bar"], use_saved_filters=False, use_kpis=False)

        status, _, body = asgi_get(app, "/api/line-bar/feature-importance")
        payload = json.loads(body)

        self.assertEqual(status, 200)
        self.assertFalse(payload["has_importance"])
        self.assertEqual(payload["models"]["glm"]["model_id"], "importance-glm")
        self.assertIn("Rebuild the active GLM", payload["models"]["glm"]["message"])

    def test_lazy_banding_suggestion_uses_bounded_sample(self) -> None:
        path = self.root / "large.parquet"
        con = duckdb.connect(database=":memory:")
        try:
            con.execute(
                f"""
COPY (
  SELECT CASE WHEN i < 100000 THEN 1 ELSE 1000000 END AS Value
  FROM range(100001) AS rows(i)
) TO '{path}' (FORMAT PARQUET)
"""
            )
        finally:
            con.close()
        app = create_app(path, token="", tools=["line_bar"], use_saved_filters=False, use_kpis=False)

        status, _, body = asgi_post_json(app, "/api/banding/suggestion", {"feature": "Value"})
        payload = json.loads(body)

        self.assertEqual(status, 200)
        self.assertEqual(payload["band_suggestion"], 1)

    def test_dataset_schema_excludes_and_reports_invalid_columns(self) -> None:
        original_probe = Dataset.probe_column_readable

        def fake_probe(dataset: Dataset, column: Any) -> None:
            if column.name == "UseofVan":
                raise duckdb.InvalidInputException(
                    'Invalid Input Error: Invalid string encoding found in Parquet file "/tmp/bad.parquet": '
                    'value "bad" is not valid UTF8!'
                )
            original_probe(dataset, column)

        with patch.object(Dataset, "probe_column_readable", fake_probe):
            dataset = Dataset(self.data_path)
            schema = dataset.schema()
            sources = dataset.data_sources()
            app = create_app(self.data_path, token="", tools=["line_bar"], use_saved_filters=False, use_kpis=False)
            status, _, body = asgi_get(app, "/api/schema")

        payload = json.loads(body)

        self.assertEqual(status, 200)
        self.assertNotIn("UseofVan", [column["name"] for column in schema["columns"]])
        self.assertIn("UseofVan", dataset.all_column_map())
        self.assertNotIn("UseofVan", dataset.column_map())
        self.assertEqual(schema["invalid_columns"], [
            {"name": "UseofVan", "error": "Invalid string encoding found in Parquet data."},
        ])
        self.assertEqual(schema["warnings"], ["Skipped 1 unreadable column: UseofVan."])
        self.assertNotIn("UseofVan", [column["name"] for column in sources[0]["columns"]])
        self.assertEqual(payload["invalid_columns"], schema["invalid_columns"])
        self.assertNotIn("UseofVan", [column["name"] for column in payload["data_sources"][0]["columns"]])

    def test_chart_accepts_dataset_source_and_rejects_unknown_sources(self) -> None:
        dataset = Dataset(self.data_path)
        request = self.request()
        request["source"] = "dataset"

        result = chart(dataset, request)

        self.assertEqual(result["source"], "dataset")

        request["source"] = "model-output"
        with self.assertRaisesRegex(ValueError, "valid data source"):
            chart(dataset, request)

    def test_chart_mixes_glm_and_gbm_prediction_sidecars_with_union_rows(self) -> None:
        glm_path = self.root / "glm_predictions.parquet"
        gbm_path = self.root / "gbm_predictions.parquet"
        con = duckdb.connect(database=":memory:")
        try:
            con.execute(
                f"""
COPY (
  SELECT 1 AS __lucidum_row_id, 100.0 AS glm_prediction
  UNION ALL
  SELECT 3, 300.0
) TO {sql_literal(str(glm_path))} (FORMAT PARQUET)
"""
            )
            con.execute(
                f"""
COPY (
  SELECT 2 AS __lucidum_row_id, 20.0 AS gbm_prediction
  UNION ALL
  SELECT 3, 30.0
) TO {sql_literal(str(gbm_path))} (FORMAT PARQUET)
"""
            )
        finally:
            con.close()
        dataset = Dataset(self.data_path)
        dataset.register_data_source_provider(
            PredictionSidecarProvider(
                {
                    "glm:m1:predictions": ModelPredictionSource(
                        source_id="glm:m1:predictions",
                        column="glm_prediction",
                        relation_sql=f"read_parquet({sql_literal(str(glm_path))})",
                        active=True,
                    ),
                    "gbm:m1:predictions": ModelPredictionSource(
                        source_id="gbm:m1:predictions",
                        column="gbm_prediction",
                        relation_sql=f"read_parquet({sql_literal(str(gbm_path))})",
                        active=True,
                    ),
                }
            )
        )

        result = chart(
            dataset,
            {
                "source": "dataset",
                "x": "glm_prediction",
                "xSource": "glm:m1:predictions",
                "responses": [
                    {"label": "Actual", "numerator": "Actual"},
                    {"label": "GBM", "numerator": "gbm_prediction", "source": "gbm:m1:predictions"},
                ],
                "denominator": "__none__",
                "filter": "",
                "bandWidth": 1,
                "dateBucket": "none",
                "lowGroup": "0",
                "sort": "alpha",
                "sigma": 0,
                "transform": "none",
            },
        )

        by_x = {row["x"]: row for row in result["rows"]}
        self.assertEqual(result["source"], "dataset")
        self.assertEqual(result["row_count"], 3)
        self.assertEqual(result["filtered_row_count"], 3)
        self.assertEqual(result["field_sources"]["x"], "glm:m1:predictions")
        self.assertIn("(missing)", by_x)
        self.assertIn("100", by_x)
        self.assertIn("300", by_x)
        self.assertAlmostEqual(by_x["(missing)"]["resp0"], 200)
        self.assertAlmostEqual(by_x["(missing)"]["resp1"], 20)
        self.assertIsNone(by_x["100"]["resp0"])
        self.assertIsNone(by_x["100"]["resp1"])
        self.assertAlmostEqual(by_x["300"]["resp0"], 300)
        self.assertAlmostEqual(by_x["300"]["resp1"], 30)

    def test_chart_adds_active_gbm_shap_ribbons_scaled_to_fitted_values(self) -> None:
        dataset = self.dataset_with_gbm_ribbons()
        request = self.request()
        request["partialDependence"] = {"mode": "shap"}

        result = chart(dataset, request)

        partial = result["partial_dependence"]
        self.assertEqual(partial["mode"], "shap")
        self.assertEqual(partial["feature"], "UseofVan")
        self.assertEqual(partial["percentiles"], [0, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 100])
        self.assertEqual(partial["scale"]["method"], "multiply")
        self.assertAlmostEqual(partial["scale"]["target"], 250.0)
        by_x = {row["x"]: row for row in partial["rows"]}
        self.assertAlmostEqual(by_x["Social"]["p50"], 250.0 / ((1.0 + math.exp(1.0)) / 2.0))
        self.assertAlmostEqual(by_x["Business"]["p50"], math.exp(1.0) * by_x["Social"]["p50"])

    def test_chart_shap_ribbons_respect_filter_weight_and_numeric_banding(self) -> None:
        dataset = self.dataset_with_gbm_ribbons()
        request = self.request("UseofVan = 'Business'")
        request.update(
            {
                "x": "YoungestDriverAge",
                "bandWidth": "10",
                "denominator": "Weight",
                "partialDependence": {"mode": "shap"},
            }
        )

        result = chart(dataset, request)

        partial = result["partial_dependence"]
        target = (300.0 * 30.0 + 400.0 * 40.0) / 70.0
        self.assertEqual([row["x"] for row in partial["rows"]], ["50", "60"])
        self.assertAlmostEqual(partial["scale"]["target"], target)
        weighted_p50 = sum(row["p50"] * row["volume"] for row in partial["rows"]) / sum(row["volume"] for row in partial["rows"])
        self.assertAlmostEqual(weighted_p50, target)

    def test_chart_shap_ribbons_use_quantile_banding(self) -> None:
        dataset = self.dataset_with_gbm_ribbons()
        request = self.request()
        request.update(
            {
                "x": "YoungestDriverAge",
                "bandWidth": "2",
                "quantileMode": "quantile",
                "partialDependence": {"mode": "shap"},
            }
        )

        result = chart(dataset, request)

        self.assertEqual([row["x"] for row in result["partial_dependence"]["rows"]], ["Q1", "Q2"])

    def test_chart_shap_ribbons_recompute_percentiles_for_low_weight_other_group(self) -> None:
        self.data_path.write_text(
            "YoungestDriverAge,UseofVan,QuoteDate,Gross.Weight,Actual,Expected,Weight\n"
            "10,A,2024-01-01,1000,1,1,1\n"
            "20,B,2024-01-02,1000,1,1,1\n"
            "30,C,2024-01-03,1000,1,1,1\n"
            "40,D,2024-01-04,1000,1,1,1\n",
            encoding="utf-8",
        )
        dataset = self.dataset_with_gbm_ribbons(
            objective="regression",
            predictions=[(1, 55.0), (2, 55.0), (3, 55.0), (4, 55.0)],
            shap_values=[(1, 0.0, 0.0), (2, 10.0, 10.0), (3, 100.0, 100.0), (4, 200.0, 200.0)],
        )
        request = self.request()
        request.update({"lowGroup": "1", "partialDependence": {"mode": "shap"}})

        result = chart(dataset, request)

        row = result["partial_dependence"]["rows"][0]
        self.assertEqual(row["x"], "Other")
        self.assertAlmostEqual(row["p0"], 0.0)
        self.assertAlmostEqual(row["p50"], 55.0)
        self.assertAlmostEqual(row["p100"], 200.0)

    def test_chart_shap_sort_orders_categoricals_by_scaled_median(self) -> None:
        dataset = self.dataset_with_gbm_ribbons()
        request = self.request()
        request.update({"partialDependence": {"mode": "shap"}, "sort": "shap"})

        result = chart(dataset, request)

        self.assertEqual([row["x"] for row in result["rows"]], ["Business", "Social"])
        self.assertEqual([row["x"] for row in result["partial_dependence"]["rows"]], ["Business", "Social"])

    def test_chart_shap_ribbons_apply_line_bar_base_transform(self) -> None:
        dataset = self.dataset_with_gbm_ribbons()
        request = self.request()
        request.update({"partialDependence": {"mode": "shap"}, "transform": "zero", "base": "Social"})

        result = chart(dataset, request)

        partial = result["partial_dependence"]
        by_x = {row["x"]: row for row in partial["rows"]}
        self.assertEqual(partial["transform"]["reference"], "base")
        self.assertEqual(partial["transform"]["base_x"], "Social")
        self.assertAlmostEqual(by_x["Social"]["p50"], 0.0)
        self.assertGreater(by_x["Business"]["p50"], 0)

    def test_chart_shap_ribbons_use_additive_scaling_for_identity_objective(self) -> None:
        dataset = self.dataset_with_gbm_ribbons(objective="regression")
        request = self.request()
        request["partialDependence"] = {"mode": "shap"}

        result = chart(dataset, request)

        partial = result["partial_dependence"]
        self.assertEqual(partial["scale"]["method"], "add")
        self.assertAlmostEqual(partial["scale"]["target"], 250.0)
        by_x = {row["x"]: row for row in partial["rows"]}
        self.assertAlmostEqual(by_x["Social"]["p50"], 245.0)
        self.assertAlmostEqual(by_x["Business"]["p50"], 255.0)

    def test_chart_shap_ribbons_missing_active_model_returns_warning(self) -> None:
        dataset = Dataset(self.data_path)
        request = self.request()
        request["partialDependence"] = {"mode": "shap"}

        result = chart(dataset, request)

        self.assertEqual(result["partial_dependence"]["rows"], [])
        self.assertIn("No active GBM SHAP values", result["warnings"][0])

    def test_chart_glm_overlay_uses_base_profile_without_interactions(self) -> None:
        dataset = Dataset(self.data_path)
        _store, model_id = self.write_active_glm_for_overlay(dataset, formula="C(UseofVan) + YoungestDriverAge", denominator_column="Weight")
        request = self.request()
        request.update({"denominator": "Weight", "partialDependence": {"mode": "glm"}, "transform": "zero", "base": "Social"})
        feature_spec = {
            "rows": [
                {"feature": "UseofVan", "base": "Social"},
                {"feature": "YoungestDriverAge", "base": "45"},
                {"feature": "Weight", "base": "20"},
            ]
        }

        result = chart(dataset, request, feature_spec=feature_spec)

        partial = result["partial_dependence"]
        self.assertEqual(partial["mode"], "glm")
        self.assertEqual(partial["model_id"], model_id)
        self.assertEqual(partial["feature"], "UseofVan")
        self.assertEqual(partial["method"], "base_profile")
        self.assertEqual(partial["percentiles"], [50])
        self.assertEqual(partial["transform"]["reference"], "base")
        self.assertEqual(partial["transform"]["base_x"], "Social")
        by_x = {row["x"]: row for row in partial["rows"]}
        self.assertEqual(set(by_x), {"Business", "Social"})
        self.assertAlmostEqual(by_x["Social"]["p50"], 0.0)
        self.assertIsNotNone(by_x["Business"]["p50"])

    def test_chart_glm_overlay_scales_base_profile_to_fitted_mean(self) -> None:
        self.data_path.write_text(
            "YoungestDriverAge,UseofVan,QuoteDate,Gross.Weight,Actual,Expected,Weight\n"
            "25,Social,2024-01-01,2000,100,95,5\n"
            "30,Social,2024-01-02,2400,130,125,9\n"
            "35,Business,2024-01-03,2800,180,175,8\n"
            "40,Social,2024-01-04,3200,190,185,15\n"
            "45,Business,2024-01-05,3600,260,255,14\n"
            "50,Social,2024-01-06,4000,250,245,19\n"
            "55,Business,2024-01-07,4400,330,325,22\n"
            "60,Business,2024-01-08,4800,390,385,25\n",
            encoding="utf-8",
        )
        dataset = Dataset(self.data_path)
        self.write_active_glm_for_overlay(dataset, formula="C(UseofVan) + YoungestDriverAge")
        request = self.request()
        request["partialDependence"] = {"mode": "glm"}

        result = chart(dataset, request, feature_spec={"rows": [{"feature": "YoungestDriverAge", "base": "25"}]})

        partial = result["partial_dependence"]
        self.assertEqual(partial["method"], "base_profile")
        self.assertEqual(partial["scale"]["method"], "add")
        self.assertNotAlmostEqual(partial["scale"]["source_mean"], partial["scale"]["target"])
        scaled_mean = sum(row["p50"] * row["volume"] for row in partial["rows"]) / sum(row["volume"] for row in partial["rows"])
        self.assertAlmostEqual(scaled_mean, partial["scale"]["target"])

    def test_chart_glm_overlay_uses_sampled_marginal_with_interactions(self) -> None:
        self.data_path.write_text(
            "YoungestDriverAge,UseofVan,QuoteDate,Gross.Weight,Actual,Expected,Weight\n"
            "25,Social,2024-01-01,2000,100,95,5\n"
            "30,Social,2024-01-02,2400,130,125,9\n"
            "35,Business,2024-01-03,2800,140,135,8\n"
            "40,Social,2024-01-04,3200,180,175,15\n"
            "45,Business,2024-01-05,3600,210,205,14\n"
            "50,Social,2024-01-06,4000,260,255,19\n"
            "55,Business,2024-01-07,4400,270,265,22\n"
            "60,Business,2024-01-08,4800,330,325,25\n",
            encoding="utf-8",
        )
        dataset = Dataset(self.data_path)
        _store, model_id = self.write_active_glm_for_overlay(
            dataset,
            formula="YoungestDriverAge + Weight + YoungestDriverAge:Weight",
        )
        request = self.request()
        request.update({"x": "YoungestDriverAge", "bandWidth": "10", "denominator": "Weight", "partialDependence": {"mode": "glm"}})

        result = chart(dataset, request)

        partial = result["partial_dependence"]
        self.assertEqual(partial["mode"], "glm")
        self.assertEqual(partial["model_id"], model_id)
        self.assertEqual(partial["feature"], "YoungestDriverAge")
        self.assertEqual(partial["method"], "sampled_marginal")
        self.assertEqual(partial["sample"]["population_row_count"], 8)
        self.assertEqual(partial["sample"]["sample_row_count"], 8)
        self.assertEqual(partial["sample"]["x_value_count"], 5)
        self.assertEqual(partial["sample"]["prediction_cell_count"], 40)
        self.assertEqual(partial["sample"]["max_sample_rows"], 100000)
        self.assertEqual(partial["sample"]["max_prediction_cells"], 2000000)
        self.assertEqual(partial["sample"]["seed"], 2026)
        self.assertTrue(all(row["p50"] is not None for row in partial["rows"]))

    def test_chart_both_mode_returns_shap_and_glm_overlays(self) -> None:
        dataset = self.dataset_with_gbm_ribbons(objective="regression")
        _store, _model_id = self.write_active_glm_for_overlay(dataset)
        request = self.request()
        request["partialDependence"] = {"mode": "both"}

        result = chart(dataset, request)

        partial = result["partial_dependence"]
        self.assertEqual(partial["mode"], "both")
        self.assertEqual(set(partial["overlays"]), {"shap", "glm"})
        self.assertEqual(partial["overlays"]["shap"]["mode"], "shap")
        self.assertEqual(partial["overlays"]["glm"]["mode"], "glm")
        self.assertGreater(len(partial["overlays"]["shap"]["rows"]), 0)
        self.assertGreater(len(partial["overlays"]["glm"]["rows"]), 0)
        self.assertEqual([row["x"] for row in partial["overlays"]["shap"]["rows"]], [row["x"] for row in result["rows"]])
        self.assertEqual([row["x"] for row in partial["overlays"]["glm"]["rows"]], [row["x"] for row in result["rows"]])

    def test_chart_both_mode_aligns_glm_overlay_to_shap_mean(self) -> None:
        dataset = self.dataset_with_gbm_ribbons(objective="regression")
        request = self.request()
        request["partialDependence"] = {"mode": "both"}
        fake_glm = {
            "mode": "glm",
            "model_id": "fake-glm",
            "feature": "UseofVan",
            "method": "base_profile",
            "percentiles": [50],
            "rows": [
                {"x": "Business", "x_sort": "Business", "original_order": 1, "volume": 1, "is_tail": False, "p50": 10.0},
                {"x": "Social", "x_sort": "Social", "original_order": 2, "volume": 1, "is_tail": False, "p50": 20.0},
            ],
            "warnings": [],
            "scale": {"method": "add", "target": 15.0, "source_mean": 15.0},
            "sample": {},
            "transform": {"mode": "none"},
        }

        with patch("py_lucidum.tools.glm.overlay.build_glm_partial_dependence_overlay", return_value=fake_glm):
            result = chart(dataset, request)

        partial = result["partial_dependence"]
        shap = partial["overlays"]["shap"]
        glm = partial["overlays"]["glm"]
        shap_mean = sum(row["p50"] * row["volume"] for row in shap["rows"]) / sum(row["volume"] for row in shap["rows"])
        glm_mean = sum(row["p50"] * row["volume"] for row in glm["rows"]) / sum(row["volume"] for row in glm["rows"])
        self.assertAlmostEqual(shap_mean, shap["scale"]["target"])
        self.assertAlmostEqual(glm_mean, shap["scale"]["target"])
        self.assertEqual(glm["scale"]["target"], shap["scale"]["target"])
        self.assertEqual(glm["scale"]["native_target"], 15.0)

    def test_chart_both_mode_keeps_glm_when_shap_unavailable(self) -> None:
        dataset = Dataset(self.data_path)
        self.write_active_glm_for_overlay(dataset)
        request = self.request()
        request["partialDependence"] = {"mode": "both"}

        result = chart(dataset, request)

        partial = result["partial_dependence"]
        self.assertEqual(partial["mode"], "both")
        self.assertEqual(partial["overlays"]["shap"]["rows"], [])
        self.assertGreater(len(partial["overlays"]["glm"]["rows"]), 0)
        self.assertTrue(any("No active GBM SHAP values" in warning for warning in result["warnings"]))

    def test_chart_both_mode_keeps_shap_when_glm_unavailable(self) -> None:
        dataset = self.dataset_with_gbm_ribbons(objective="regression")
        request = self.request()
        request["partialDependence"] = {"mode": "both"}

        result = chart(dataset, request)

        partial = result["partial_dependence"]
        self.assertEqual(partial["mode"], "both")
        self.assertGreater(len(partial["overlays"]["shap"]["rows"]), 0)
        self.assertEqual(partial["overlays"]["glm"]["rows"], [])
        self.assertTrue(any("No active GLM" in warning for warning in result["warnings"]))

    def test_chart_glm_overlay_dispatches_to_worker_when_lightgbm_loaded(self) -> None:
        dataset = Dataset(self.data_path)
        request = self.request()
        request["partialDependence"] = {"mode": "glm"}
        worker_result = {
            "mode": "glm",
            "model_id": "worker-model",
            "feature": "UseofVan",
            "method": "base_profile",
            "percentiles": [50],
            "rows": [],
            "warnings": [],
            "scale": {"method": "none", "target": None, "source_mean": None},
            "sample": {},
            "transform": {"mode": "none"},
        }

        with patch.dict(sys.modules, {"lightgbm": object()}):
            with patch("py_lucidum.tools.glm.overlay.build_glm_partial_dependence_overlay_in_subprocess", return_value=worker_result) as worker:
                result = chart(dataset, request)

        worker.assert_called_once()
        self.assertEqual(result["partial_dependence"]["model_id"], "worker-model")

    def test_chart_glm_overlay_dispatches_to_worker_when_lightgbm_importable_before_loaded(self) -> None:
        dataset = Dataset(self.data_path)
        request = self.request()
        request["partialDependence"] = {"mode": "glm"}
        worker_result = {
            "mode": "glm",
            "model_id": "worker-model",
            "feature": "UseofVan",
            "method": "base_profile",
            "percentiles": [50],
            "rows": [],
            "warnings": [],
            "scale": {"method": "none", "target": None, "source_mean": None},
            "sample": {},
            "transform": {"mode": "none"},
        }
        saved_lightgbm = sys.modules.pop("lightgbm", None)
        try:
            with patch("py_lucidum.tools.glm.overlay.importlib.util.find_spec", return_value=object()):
                with patch("py_lucidum.tools.glm.overlay.build_glm_partial_dependence_overlay_in_subprocess", return_value=worker_result) as worker:
                    result = chart(dataset, request)
        finally:
            if saved_lightgbm is not None:
                sys.modules["lightgbm"] = saved_lightgbm

        worker.assert_called_once()
        self.assertEqual(result["partial_dependence"]["model_id"], "worker-model")

    def test_chart_glm_overlay_worker_returns_rows_when_lightgbm_loaded(self) -> None:
        dataset = Dataset(self.data_path)
        self.write_active_glm_for_overlay(dataset)
        request = self.request()
        request["partialDependence"] = {"mode": "glm"}

        with patch.dict(sys.modules, {"lightgbm": object()}):
            result = chart(dataset, request)

        partial = result["partial_dependence"]
        self.assertEqual(partial["mode"], "glm")
        self.assertGreater(len(partial["rows"]), 0)

    def test_chart_glm_overlay_fresh_process_survives_with_lightgbm_importable(self) -> None:
        if importlib.util.find_spec("lightgbm") is None:
            self.skipTest("lightgbm is not installed")
        dataset = Dataset(self.data_path)
        self.write_active_glm_for_overlay(dataset)
        request = self.request()
        request["partialDependence"] = {"mode": "glm"}
        repo_root = Path(__file__).resolve().parents[1]
        env = os.environ.copy()
        src_path = str(repo_root / "src")
        env["PYTHONPATH"] = src_path if not env.get("PYTHONPATH") else f"{src_path}{os.pathsep}{env['PYTHONPATH']}"
        env["PYTHONFAULTHANDLER"] = "1"
        script = f"""
from pathlib import Path
import json
import sys

from py_lucidum.app import create_app
from py_lucidum.tools.line_bar.query import chart

if "lightgbm" in sys.modules:
    raise SystemExit("lightgbm was loaded before the chart request")

app = create_app(
    Path({str(self.data_path)!r}),
    token="",
    tools=["line_bar", "glm", "gbm"],
    use_saved_filters=False,
    use_kpis=False,
    use_features=False,
)
request = json.loads({json.dumps(request)!r})
result = chart(app.state.dataset, request, feature_spec={{}})
partial = result.get("partial_dependence") or {{}}
if partial.get("mode") != "glm":
    raise SystemExit(f"unexpected partial dependence payload: {{partial!r}}")
rows = partial.get("rows")
if not isinstance(rows, list) or not rows:
    raise SystemExit(f"missing GLM overlay rows: {{partial!r}}")
print(len(rows))
"""
        completed = subprocess.run(
            [sys.executable, "-c", script],
            cwd=repo_root,
            env=env,
            check=False,
            capture_output=True,
            text=True,
        )

        self.assertEqual(completed.returncode, 0, f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}")

    def test_lazy_banding_suggestion_uses_x_source_for_prediction_axes(self) -> None:
        glm_path = self.root / "glm_banding_predictions.parquet"
        gbm_path = self.root / "gbm_banding_predictions.parquet"
        con = duckdb.connect(database=":memory:")
        try:
            con.execute(
                f"""
COPY (
  SELECT 1 AS __lucidum_row_id, 101.0 AS glm_prediction
  UNION ALL
  SELECT 2, 202.0
  UNION ALL
  SELECT 3, 303.0
) TO {sql_literal(str(glm_path))} (FORMAT PARQUET)
"""
            )
            con.execute(
                f"""
COPY (
  SELECT 1 AS __lucidum_row_id, 10.0 AS gbm_prediction
  UNION ALL
  SELECT 2, 20.0
  UNION ALL
  SELECT 3, 30.0
) TO {sql_literal(str(gbm_path))} (FORMAT PARQUET)
"""
            )
        finally:
            con.close()
        app = create_app(self.data_path, token="", tools=["line_bar"], use_saved_filters=False, use_kpis=False)
        app.state.dataset.register_data_source_provider(
            PredictionSidecarProvider(
                {
                    "glm:m1:predictions": ModelPredictionSource(
                        source_id="glm:m1:predictions",
                        column="glm_prediction",
                        relation_sql=f"read_parquet({sql_literal(str(glm_path))})",
                        active=True,
                    ),
                    "gbm:m1:predictions": ModelPredictionSource(
                        source_id="gbm:m1:predictions",
                        column="gbm_prediction",
                        relation_sql=f"read_parquet({sql_literal(str(gbm_path))})",
                        active=True,
                    ),
                }
            )
        )

        status, _, body = asgi_post_json(
            app,
            "/api/banding/suggestion",
            {
                "source": "gbm:m1:predictions",
                "xSource": "glm:m1:predictions",
                "feature": "glm_prediction",
            },
        )
        payload = json.loads(body)

        self.assertEqual(status, 200)
        self.assertEqual(payload["source"], "glm:m1:predictions")
        self.assertEqual(payload["feature"], "glm_prediction")
        self.assertGreater(payload["band_suggestion"], 0)

    def test_default_saved_filters_fall_back_to_specs_directory(self) -> None:
        self.filters_path.unlink()
        specs_dir = self.root / "specs"
        specs_dir.mkdir()
        (specs_dir / "filter_spec.csv").write_text(
            "theme,name,expression\nDriver age,Spec older drivers,YoungestDriverAge > 40\n",
            encoding="utf-8",
        )
        previous_cwd = Path.cwd()
        try:
            os.chdir(self.root)
            app = create_app(self.data_path, token="dev-token", tools=["line_bar"])
        finally:
            os.chdir(previous_cwd)

        self.assertEqual(
            app.state.saved_filters,
            [{"theme": "Driver age", "name": "Spec older drivers", "expression": "YoungestDriverAge > 40"}],
        )

    def test_default_kpis_fall_back_to_specs_directory(self) -> None:
        self.kpis_path.unlink()
        specs_dir = self.root / "specs"
        specs_dir.mkdir()
        (specs_dir / "kpi_spec.csv").write_text(
            "group,name,actual,denominator,decimals,format\n"
            "Pricing,Weighted actual,Actual,Weight,1,currency\n",
            encoding="utf-8",
        )
        previous_cwd = Path.cwd()
        try:
            os.chdir(self.root)
            app = create_app(self.data_path, token="dev-token", tools=["line_bar"], use_saved_filters=False)
        finally:
            os.chdir(previous_cwd)

        self.assertEqual(
            app.state.kpis,
            [{"group": "Pricing", "name": "Weighted actual", "actual": "Actual", "denominator": "Weight", "decimals": 1, "format": "currency"}],
        )

    def test_kpi_spec_parses_denominator_aliases(self) -> None:
        self.kpis_path.write_text(
            "group,name,actual,denominator,decimals,format\n"
            "Pricing,Average actual,Actual,Average row value,2,currency\n"
            "Pricing,Expected average,Expected,,1,percent\n"
            "Pricing,Actual per row,Actual,__none__,0,number\n",
            encoding="utf-8",
        )

        self.assertEqual(
            load_kpis(self.kpis_path),
            [
                {"group": "Pricing", "name": "Average actual", "actual": "Actual", "denominator": "__none__", "decimals": 2, "format": "currency"},
                {"group": "Pricing", "name": "Expected average", "actual": "Expected", "denominator": "__none__", "decimals": 1, "format": "percent"},
                {"group": "Pricing", "name": "Actual per row", "actual": "Actual", "denominator": "__none__", "decimals": 0, "format": "number"},
            ],
        )

    def test_kpi_spec_rejects_invalid_header(self) -> None:
        self.kpis_path.write_text("name,actual,denominator,decimals,format\nActual average,Actual,N,2,number\n", encoding="utf-8")

        with self.assertRaisesRegex(ValueError, "group,name,actual,denominator,decimals,format"):
            load_kpis(self.kpis_path)

    def test_old_two_column_saved_filter_csv_is_rejected(self) -> None:
        self.filters_path.write_text("name,expression\nOld older drivers,YoungestDriverAge > 40\n", encoding="utf-8")

        with self.assertRaisesRegex(ValueError, "theme,name,expression"):
            load_saved_filters(self.filters_path)

    def test_app_loads_with_saved_filters_disabled(self) -> None:
        specs_dir = self.root / "specs"
        specs_dir.mkdir()
        (specs_dir / "filter_spec.csv").write_text(
            "theme,name,expression\nDriver age,Spec older drivers,YoungestDriverAge > 40\n",
            encoding="utf-8",
        )
        previous_cwd = Path.cwd()
        try:
            os.chdir(self.root)
            app = create_app(self.data_path, token="dev-token", tools=["line_bar"], use_saved_filters=False)
        finally:
            os.chdir(previous_cwd)

        self.assertEqual(app.state.saved_filters, [])
        self.assertIsNone(app.state.resolved_filters_path)
        self.assertFalse(app.state.use_saved_filters)

    def test_app_loads_with_kpis_disabled(self) -> None:
        previous_cwd = Path.cwd()
        try:
            os.chdir(self.root)
            app = create_app(self.data_path, token="dev-token", tools=["line_bar"], use_kpis=False)
        finally:
            os.chdir(previous_cwd)

        self.assertEqual(app.state.kpis, [])
        self.assertIsNone(app.state.resolved_kpis_path)
        self.assertFalse(app.state.use_kpis)

    def test_app_loads_with_feature_specs_disabled(self) -> None:
        previous_cwd = Path.cwd()
        try:
            os.chdir(self.root)
            app = create_app(self.data_path, token="dev-token", tools=["line_bar"], use_features=False)
        finally:
            os.chdir(previous_cwd)

        self.assertEqual(app.state.feature_spec, {"rows": [], "scenarios": []})
        self.assertIsNone(app.state.resolved_features_path)
        self.assertFalse(app.state.use_features)

    def test_reload_refreshes_feature_specs(self) -> None:
        app = create_app(
            self.data_path,
            token="",
            tools=["line_bar"],
            use_saved_filters=False,
            use_kpis=False,
            features_path=self.features_path,
        )
        self.features_path.write_text(
            "Feature,Grouping,scenario1\n"
            "UseofVan,VEHICLE,feature\n",
            encoding="utf-8",
        )

        status, _, _ = asgi_post_json(app, "/api/reload", {})

        self.assertEqual(status, 200)
        self.assertEqual(app.state.feature_spec["scenarios"], [{"name": "scenario1", "features": ["UseofVan"]}])

    def test_dataset_schema_includes_file_size(self) -> None:
        dataset = Dataset(self.data_path)
        schema = dataset.schema()

        self.assertEqual(schema["file_size"], self.data_path.stat().st_size)

    def test_regular_csv_file_path_loads_through_dataset_and_app(self) -> None:
        dataset = Dataset(self.data_path)
        schema = dataset.schema()
        app = create_app(self.data_path, tools=["line_bar"], use_saved_filters=False)
        app_schema = app.state.dataset.schema()

        self.assertIn("YoungestDriverAge", {column["name"] for column in schema["columns"]})
        self.assertIn("Actual", {column["name"] for column in app_schema["columns"]})

    def test_regular_parquet_file_path_loads_through_dataset(self) -> None:
        parquet_path = self.root / "ordinary.parquet"
        con = duckdb.connect(database=":memory:")
        con.execute(
            f"""
COPY (
  SELECT
    1::INTEGER AS id,
    123.45::DOUBLE AS premium,
    'AB'::VARCHAR AS postcode_area
) TO '{parquet_path.as_posix()}' (FORMAT PARQUET)
"""
        )

        schema = Dataset(parquet_path).schema()

        self.assertEqual(
            [column["name"] for column in schema["columns"]],
            ["id", "premium", "postcode_area"],
        )

    def test_chart_filters_and_aggregates_response_lines(self) -> None:
        dataset = Dataset(self.data_path)
        result = chart(dataset, self.request("YoungestDriverAge > 40"))

        self.assertEqual(result["row_count"], 4)
        self.assertEqual(result["filtered_row_count"], 3)
        self.assertEqual([row["x"] for row in result["rows"]], ["Business", "Social"])
        business, social = result["rows"]
        self.assertEqual(business["volume"], 2)
        self.assertEqual(business["resp0"], 350)
        self.assertEqual(business["resp0_num"], 700)
        self.assertEqual(business["resp0_den"], 2)
        self.assertEqual(business["resp1"], 350)
        self.assertEqual(business["resp1_num"], 700)
        self.assertEqual(business["resp1_den"], 2)
        self.assertEqual(social["resp0_num"], 200)
        self.assertEqual(social["resp0_den"], 1)
        self.assertEqual(social["resp1_num"], 210)
        self.assertEqual(social["resp1_den"], 1)
        self.assertEqual(result["response_summaries"][0]["label"], "Actual")
        self.assertEqual(result["response_summaries"][0]["value"], 300)
        self.assertEqual(result["response_summaries"][0]["numerator"], 900)
        self.assertEqual(result["response_summaries"][0]["denominator"], 3)
        self.assertEqual(result["response_summaries"][1]["label"], "Expected")
        self.assertAlmostEqual(result["response_summaries"][1]["value"], 910 / 3)
        self.assertEqual(result["denominator"]["label"], "Average row value")
        self.assertEqual(result["denominator"]["bar_label"], "Row count")
        self.assertEqual(result["denominator"]["value"], 3)

    def test_chart_uses_common_weight_column_for_lines_bars_and_summary(self) -> None:
        dataset = Dataset(self.data_path)
        request = self.request("YoungestDriverAge > 40")
        request["denominator"] = "Weight"

        result = chart(dataset, request)

        self.assertEqual(result["denominator"]["label"], "Weight")
        self.assertEqual(result["denominator"]["bar_label"], "Weight")
        self.assertEqual(result["denominator"]["value"], 90)
        self.assertEqual([row["x"] for row in result["rows"]], ["Business", "Social"])
        self.assertEqual(result["rows"][0]["volume"], 70)
        self.assertEqual(result["rows"][1]["volume"], 20)
        self.assertEqual(result["rows"][0]["resp0"], 10)
        self.assertEqual(result["rows"][0]["resp0_num"], 700)
        self.assertEqual(result["rows"][0]["resp0_den"], 70)
        self.assertEqual(result["rows"][0]["resp1"], 10)
        self.assertEqual(result["rows"][0]["resp1_num"], 700)
        self.assertEqual(result["rows"][0]["resp1_den"], 70)
        self.assertEqual(result["response_summaries"][0]["denominator"], 90)
        self.assertEqual(result["response_summaries"][0]["value"], 10)

    def test_average_row_value_reports_rows_with_missing_responses(self) -> None:
        self.data_path.write_text(
            "UseofVan,Actual,Expected,Weight\n"
            "Social,100,90,10\n"
            "Social,,110,20\n"
            "Business,300,290,30\n",
            encoding="utf-8",
        )
        dataset = Dataset(self.data_path)
        request = self.request()
        request["x"] = "UseofVan"

        result = chart(dataset, request)

        self.assertEqual(result["denominator"]["value"], 2)
        self.assertEqual(result["denominator"]["missing_response_rows"], 1)
        self.assertIn(
            "1 row excluded from Weight because one or more selected response values were missing.",
            result["warnings"],
        )
        social = next(row for row in result["rows"] if row["x"] == "Social")
        self.assertEqual(social["volume"], 1)
        self.assertEqual(social["resp0"], 100)
        self.assertEqual(social["resp1"], 90)

    def test_weight_column_reports_missing_zero_and_negative_values(self) -> None:
        self.data_path.write_text(
            "UseofVan,Actual,Expected,Weight\n"
            "Social,100,90,10\n"
            "Social,200,210,0\n"
            "Business,300,290,-5\n"
            "Business,400,410,\n",
            encoding="utf-8",
        )
        dataset = Dataset(self.data_path)
        request = self.request()
        request["x"] = "UseofVan"
        request["denominator"] = "Weight"

        result = chart(dataset, request)

        self.assertEqual(result["denominator"]["value"], 5)
        self.assertEqual(result["denominator"]["missing_weight_rows"], 1)
        self.assertEqual(result["denominator"]["zero_weight_rows"], 1)
        self.assertEqual(result["denominator"]["negative_weight_rows"], 1)
        self.assertIn("1 row excluded from Weight because Weight was missing.", result["warnings"])
        self.assertIn("1 row has zero Weight.", result["warnings"])
        self.assertIn("1 row has negative Weight.", result["warnings"])

    def test_chart_accepts_string_date_and_quoted_column_filters(self) -> None:
        dataset = Dataset(self.data_path)

        string_result = chart(dataset, self.request("UseofVan = 'Social'"))
        self.assertEqual(string_result["filtered_row_count"], 2)
        self.assertEqual([row["x"] for row in string_result["rows"]], ["Social"])

        quoted_result = chart(dataset, self.request('"Gross.Weight" >= 4000'))
        self.assertEqual(quoted_result["filtered_row_count"], 2)
        self.assertEqual([row["x"] for row in quoted_result["rows"]], ["Business"])

        date_result = chart(dataset, self.request("QuoteDate >= DATE '2024-02-01'"))
        self.assertEqual(date_result["filtered_row_count"], 2)
        self.assertEqual([row["x"] for row in date_result["rows"]], ["Business"])

    def test_numeric_banding_without_quantiles_still_uses_fixed_width(self) -> None:
        dataset = Dataset(self.data_path)
        request = self.request()
        request.update({"x": "YoungestDriverAge", "bandWidth": "10", "quantileMode": "off"})

        result = chart(dataset, request)

        self.assertEqual([row["x"] for row in result["rows"]], ["30", "40", "50", "60"])
        self.assertEqual([row["resp0"] for row in result["rows"]], [100, 200, 300, 400])

    def test_numeric_decimal_banding_cleans_floating_point_labels(self) -> None:
        self.data_path.write_text(
            "Score,Actual,Expected\n"
            "49.9,100,90\n"
            "50.0,200,190\n"
            "50.2,300,290\n"
            "50.4,400,390\n",
            encoding="utf-8",
        )
        dataset = Dataset(self.data_path)
        request = self.request()
        request.update({"x": "Score", "bandWidth": "0.2", "quantileMode": "off"})

        result = chart(dataset, request)
        labels = [row["x"] for row in result["rows"]]

        self.assertEqual(labels, ["49.8", "50", "50.2"])
        self.assertFalse(any("000000" in label for label in labels))

    def test_numeric_whole_number_banding_omits_decimal_suffix(self) -> None:
        self.data_path.write_text(
            "Score,Actual,Expected\n"
            "50.1,100,90\n"
            "51.2,200,190\n",
            encoding="utf-8",
        )
        dataset = Dataset(self.data_path)
        request = self.request()
        request.update({"x": "Score", "bandWidth": "1", "quantileMode": "off"})

        result = chart(dataset, request)

        self.assertEqual([row["x"] for row in result["rows"]], ["50", "51"])

    def test_numeric_quantile_banding_groups_non_missing_values(self) -> None:
        dataset = Dataset(self.data_path)
        request = self.request()
        request.update({"x": "YoungestDriverAge", "bandWidth": "4", "quantileMode": "quantile"})

        result = chart(dataset, request)

        self.assertEqual([row["x"] for row in result["rows"]], ["Q1", "Q2", "Q3", "Q4"])
        self.assertEqual([row["volume"] for row in result["rows"]], [1, 1, 1, 1])
        self.assertEqual([row["resp0"] for row in result["rows"]], [100, 200, 300, 400])

    def test_numeric_quantile_banding_keeps_missing_values_separate(self) -> None:
        self.data_path.write_text(
            "Score,Actual,Expected\n"
            "1,10,9\n"
            ",20,19\n"
            "2,30,29\n"
            "3,40,39\n"
            "4,50,49\n",
            encoding="utf-8",
        )
        dataset = Dataset(self.data_path)
        request = self.request()
        request.update({"x": "Score", "bandWidth": "2", "quantileMode": "quantile"})

        result = chart(dataset, request)

        self.assertEqual([row["x"] for row in result["rows"]], ["Q1", "Q2", "Missing"])
        self.assertEqual([row["volume"] for row in result["rows"]], [2, 2, 1])
        missing = result["rows"][2]
        self.assertEqual(missing["resp0"], 20)
        self.assertFalse(missing["is_tail"])

        low_group_request = self.request()
        low_group_request.update({"x": "Score", "bandWidth": "4", "quantileMode": "quantile", "lowGroup": "2"})
        low_group_result = chart(dataset, low_group_request)
        self.assertEqual([row["x"] for row in low_group_result["rows"]], ["Low tail", "High tail", "Missing"])
        low_tail, high_tail, missing = low_group_result["rows"]
        self.assertEqual(low_tail["resp0_num"], 40)
        self.assertEqual(low_tail["resp0_den"], 2)
        self.assertEqual(low_tail["resp1_num"], 38)
        self.assertEqual(low_tail["resp1_den"], 2)
        self.assertEqual(high_tail["resp0_num"], 90)
        self.assertEqual(high_tail["resp0_den"], 2)
        self.assertEqual(high_tail["resp1_num"], 88)
        self.assertEqual(high_tail["resp1_den"], 2)
        self.assertEqual(missing["resp0_num"], 20)
        self.assertEqual(missing["resp0_den"], 1)
        self.assertFalse(low_group_result["rows"][2]["is_tail"])

    def test_quantile_count_rounds_and_clamps_to_supported_range(self) -> None:
        self.assertEqual(normalise_quantile_count("0"), 1)
        self.assertEqual(normalise_quantile_count("0.1"), 1)
        self.assertEqual(normalise_quantile_count("2.6"), 3)
        self.assertEqual(normalise_quantile_count("10000"), 1000)

    def test_grouped_numeric_tails_keep_sigma_bars(self) -> None:
        dataset = Dataset(self.data_path)
        request = self.request()
        request.update({"x": "YoungestDriverAge", "lowGroup": "2", "sigma": 2})

        result = chart(dataset, request)

        tails = {row["x"]: row for row in result["rows"] if row["is_tail"]}
        self.assertEqual(set(tails), {"Low tail", "High tail"})
        for row in tails.values():
            self.assertEqual(row["valid_folds"], 2)
            self.assertIsNotNone(row.get("resp1_low"))
            self.assertIsNotNone(row.get("resp1_high"))

    def test_invalid_filter_is_rejected(self) -> None:
        dataset = Dataset(self.data_path)

        with self.assertRaisesRegex(ValueError, "single DuckDB expression"):
            chart(dataset, self.request("YoungestDriverAge > 40; DROP TABLE x"))

    def test_legacy_query_module_still_reexports_line_bar_helpers(self) -> None:
        x_sql = build_x_sql("YoungestDriverAge", "integer", "10", "none")

        self.assertIn("FLOOR", x_sql["key"])
        self.assertIn("YoungestDriverAge", x_sql["key"])

    def test_legacy_dataset_chart_method_still_works(self) -> None:
        dataset = LegacyDataset(self.data_path)
        result = dataset.chart(self.request("YoungestDriverAge > 40"))

        self.assertEqual(result["filtered_row_count"], 3)
        self.assertEqual([row["x"] for row in result["rows"]], ["Business", "Social"])


if __name__ == "__main__":
    unittest.main()
