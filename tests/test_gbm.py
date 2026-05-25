from __future__ import annotations

import asyncio
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any
from unittest.mock import patch

import duckdb

from py_lucidum.app import create_app
from py_lucidum.core import Dataset, sql_literal
from py_lucidum.demo import demo_dataset_path
from py_lucidum.tools.gbm.store import GbmModelStore, GbmSourceProvider
from py_lucidum.tools.gbm.training import MissingGbmDependency, should_use_offset_init_score, train_model
from py_lucidum.tools.gbm.validation import GBM_METRICS, GBM_OBJECTIVES, default_parameters, feature_rows, normalise_parameters, validate_request
from py_lucidum.tools.line_bar.query import chart
from py_lucidum.tools.uk_map.query import summary as map_summary


def asgi_get(app: Any, path: str) -> tuple[int, bytes]:
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
    status = next(message for message in messages if message["type"] == "http.response.start")["status"]
    body = b"".join(message.get("body", b"") for message in messages if message["type"] == "http.response.body")
    return status, body


def asgi_post_json(app: Any, path: str, payload: dict[str, Any]) -> tuple[int, bytes]:
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
    status = next(message for message in messages if message["type"] == "http.response.start")["status"]
    response_body = b"".join(message.get("body", b"") for message in messages if message["type"] == "http.response.body")
    return status, response_body


class GbmToolTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.data_path = self.root / "sample.csv"
        self.data_path.write_text(
            "actualNumerator,denominator,Age,Segment,PostcodeArea,PostcodeSector,PostcodeUnit,lat,long,sample\n"
            "10,100,30,A,AB,AB10 1,AB10 1AA,57.1,-2.1,training\n"
            "20,200,40,B,AB,AB10 1,AB10 1AB,57.2,-2.2,test\n"
            "30,300,50,C,CD,CD20 2,CD20 2AA,56.1,-1.1,training\n",
            encoding="utf-8",
        )

    def request_features(self) -> list[dict[str, Any]]:
        return [
            {"name": "Age", "include": True, "monotonicity": "Increasing"},
            {"name": "Segment", "include": True, "monotonicity": ""},
        ]

    def test_gbm_config_routes_work_without_lightgbm_imports(self) -> None:
        app = create_app(self.data_path, token="", tools=["gbm"], use_saved_filters=False, use_kpis=False)
        paths = {route.path for route in app.routes}

        self.assertIn("/api/gbm/config", paths)
        self.assertIn("/api/gbm/validate", paths)
        self.assertIn("/api/gbm/train", paths)
        self.assertIn("/api/gbm/models", paths)

        status, body = asgi_get(app, "/api/gbm/config")
        payload = json.loads(body)

        self.assertEqual(status, 200)
        self.assertEqual(payload["status"], "ready")
        self.assertEqual(payload["response"], "actualNumerator")
        self.assertEqual(payload["offset"], "denominator")
        self.assertEqual(payload["sample_column"], "sample")
        self.assertEqual(next(row["value"] for row in payload["parameters"] if row["name"] == "objective"), "poisson")
        self.assertEqual(next(row["value"] for row in payload["parameters"] if row["name"] == "metric"), "poisson")
        self.assertEqual(payload["parameter_options"]["objective"], list(GBM_OBJECTIVES))
        self.assertEqual(payload["parameter_options"]["metric"], list(GBM_METRICS))
        self.assertIn("Gain", Path("docs/specs/gbm-tool_plan.md").read_text(encoding="utf-8"))
        age = next(row for row in payload["features"] if row["name"] == "Age")
        self.assertEqual(age["gain"], 0.0)

    def test_train_endpoint_reports_missing_optional_dependencies(self) -> None:
        app = create_app(self.data_path, token="", tools=["gbm"], use_saved_filters=False, use_kpis=False)
        request = {
            "features": self.request_features(),
            "parameters": [{"name": "objective", "value": "poisson"}, {"name": "metric", "value": "poisson"}],
            "sample_column": "sample",
        }

        with patch("py_lucidum.tools.gbm.routes.gbm_dependencies", side_effect=MissingGbmDependency("lightgbm")):
            status, body = asgi_post_json(app, "/api/gbm/train", request)

        self.assertEqual(status, 400)
        self.assertIn("py-lucidum[gbm]", json.loads(body)["detail"])

    def test_validation_catches_missing_fixed_columns_and_invalid_monotonicity(self) -> None:
        missing_path = self.root / "missing.csv"
        missing_path.write_text("Age,Segment\n1,A\n", encoding="utf-8")
        missing_dataset = Dataset(missing_path)

        result = validate_request(missing_dataset, {"features": [{"name": "Age", "include": True}]})

        self.assertFalse(result.ok)
        self.assertIn("response column", "; ".join(result.errors))
        self.assertIn("denominator column", "; ".join(result.errors))

        dataset = Dataset(self.data_path)
        result = validate_request(
            dataset,
            {
                "features": [{"name": "Segment", "include": True, "monotonicity": "Increasing"}],
                "parameters": [{"name": "objective", "value": "poisson"}],
                "sample_column": "sample",
            },
        )

        self.assertFalse(result.ok)
        self.assertIn("must be numeric", "; ".join(result.errors))

    def test_validation_rejects_invalid_objective_and_metric(self) -> None:
        dataset = Dataset(self.data_path)

        result = validate_request(
            dataset,
            {
                "features": self.request_features(),
                "parameters": [
                    {"name": "objective", "value": "not_a_real_objective"},
                    {"name": "metric", "value": "not_a_real_metric"},
                ],
                "sample_column": "sample",
            },
        )

        self.assertFalse(result.ok)
        errors = "; ".join(result.errors)
        self.assertIn("Choose a valid LightGBM objective: not_a_real_objective", errors)
        self.assertIn("Choose a valid LightGBM metric: not_a_real_metric", errors)

    def test_cross_entropy_objectives_require_probability_response(self) -> None:
        dataset = Dataset(self.data_path)

        result = validate_request(
            dataset,
            {
                "features": self.request_features(),
                "parameters": [
                    {"name": "objective", "value": "cross_entropy"},
                    {"name": "metric", "value": "cross_entropy"},
                ],
                "sample_column": "sample",
            },
        )

        self.assertFalse(result.ok)
        self.assertIn("cross_entropy objective requires response values between 0 and 1", "; ".join(result.errors))

    def test_gain_defaults_and_sorts_descending(self) -> None:
        dataset = Dataset(self.data_path)

        rows = feature_rows(dataset, {"Segment": 1.5, "Age": 9.25})

        self.assertEqual(rows[0]["name"], "Age")
        self.assertEqual(rows[0]["gain"], 9.25)
        self.assertEqual(rows[1]["name"], "Segment")
        self.assertEqual(rows[1]["gain"], 1.5)
        self.assertEqual(next(row for row in rows if row["name"] == "sample")["gain"], 0.0)

    def test_active_model_feature_rows_mirror_saved_feature_config(self) -> None:
        dataset = Dataset(self.data_path)

        rows = feature_rows(
            dataset,
            {"Age": 9.25, "Segment": 1.5},
            model_features=[
                {"name": "Age", "include": True, "monotonicity": 1, "gain": 9.25},
                {"name": "Segment", "include": True, "monotonicity": "", "gain": 1.5},
            ],
        )
        by_name = {row["name"]: row for row in rows}

        self.assertTrue(by_name["Age"]["include"])
        self.assertEqual(by_name["Age"]["monotonicity"], "Increasing")
        self.assertTrue(by_name["Segment"]["include"])
        self.assertEqual(by_name["Segment"]["monotonicity"], "")
        self.assertFalse(by_name["sample"]["include"])
        self.assertEqual(by_name["sample"]["gain"], 0.0)

    def test_config_uses_active_model_parameters(self) -> None:
        store = self.write_model_artifacts()
        store.write_json(
            store.artifact_path("m1", "parameters"),
            {
                "objective": "gamma",
                "metric": "gamma",
                "num_iterations": 88,
                "learning_rate": 0.125,
                "custom_penalty": 2.5,
            },
        )
        app = create_app(self.data_path, token="", tools=["gbm"], use_saved_filters=False, use_kpis=False)

        status, body = asgi_get(app, "/api/gbm/config")
        payload = json.loads(body)
        parameters = {row["name"]: row["value"] for row in payload["parameters"]}
        features = {row["name"]: row for row in payload["features"]}

        self.assertEqual(status, 200)
        self.assertEqual(parameters["objective"], "gamma")
        self.assertEqual(parameters["metric"], "gamma")
        self.assertEqual(parameters["num_iterations"], 88)
        self.assertEqual(parameters["learning_rate"], 0.125)
        self.assertEqual(parameters["custom_penalty"], 2.5)
        self.assertTrue(features["Age"]["include"])
        self.assertEqual(features["Age"]["monotonicity"], "Increasing")
        self.assertTrue(features["Segment"]["include"])
        self.assertFalse(features["sample"]["include"])

    def test_activate_model_response_uses_activated_model_parameters(self) -> None:
        store = GbmModelStore(self.data_path)
        for model_id, label, learning_rate in (
            ("m1", "Model 1", 0.1),
            ("m2", "Model 2", 0.2),
        ):
            model_dir = store.create_model_dir(model_id)
            store.write_json(
                model_dir / "manifest.json",
                {
                    "model_id": model_id,
                    "label": label,
                    "created_at": f"2026-05-25T00:00:0{model_id[-1]}Z",
                    "objective": "poisson",
                    "metric": "poisson",
                    "best_iteration": 7,
                    "training_rows": 2,
                    "test_rows": 1,
                    "feature_importance": [],
                    "sources": {},
                },
            )
            store.write_json(
                model_dir / "feature_config.json",
                [
                    {"name": "Age", "kind": "integer", "include": True, "monotonicity": "Increasing", "gain": 3.0}
                    if model_id == "m1"
                    else {"name": "Segment", "kind": "categorical", "include": True, "monotonicity": "", "gain": 4.0}
                ],
            )
            store.write_json(
                model_dir / "parameters.json",
                {
                    "objective": "poisson",
                    "metric": "poisson",
                    "learning_rate": learning_rate,
                    "num_iterations": 100 + int(model_id[-1]),
                },
            )
        store.activate_model("m1")
        app = create_app(self.data_path, token="", tools=["gbm"], use_saved_filters=False, use_kpis=False)

        status, body = asgi_post_json(app, "/api/gbm/models/m2/activate", {})
        payload = json.loads(body)
        parameters = {row["name"]: row["value"] for row in payload["config"]["parameters"]}
        features = {row["name"]: row for row in payload["config"]["features"]}

        self.assertEqual(status, 200)
        self.assertEqual(payload["config"]["active_model_id"], "m2")
        self.assertEqual(parameters["learning_rate"], 0.2)
        self.assertEqual(parameters["num_iterations"], 102)
        self.assertFalse(features["Age"]["include"])
        self.assertTrue(features["Segment"]["include"])
        self.assertEqual(features["Segment"]["gain"], 4.0)

    def test_validation_accepts_sidebar_response_and_no_denominator(self) -> None:
        dataset = Dataset(self.data_path)

        result = validate_request(
            dataset,
            {
                "response": "Age",
                "offset": "__none__",
                "features": [{"name": "Segment", "include": True, "monotonicity": ""}],
                "parameters": [{"name": "objective", "value": "poisson"}],
                "sample_column": "sample",
            },
        )

        self.assertTrue(result.ok, result.errors)
        self.assertIn("offset values will be treated as 1", "; ".join(result.warnings))

    def test_poisson_init_score_is_only_used_with_real_offset_column(self) -> None:
        poisson_params = normalise_parameters([{"name": "objective", "value": "poisson"}])
        regression_params = normalise_parameters([{"name": "objective", "value": "regression"}])

        self.assertFalse(should_use_offset_init_score(poisson_params, None))
        self.assertTrue(should_use_offset_init_score(poisson_params, "denominator"))
        self.assertFalse(should_use_offset_init_score(regression_params, "denominator"))

    def test_poisson_demo_without_denominator_trains(self) -> None:
        try:
            import lightgbm  # noqa: F401
        except ImportError:
            self.skipTest("LightGBM is not installed")

        source_path = demo_dataset_path()
        repro_path = self.root / "poisson_repro.parquet"
        con = duckdb.connect(database=":memory:")
        try:
            con.execute(
                f"""
COPY (
  SELECT PREMIUM, DRIVER_AGE, LATITUDE
  FROM read_parquet({sql_literal(str(source_path))})
) TO {sql_literal(str(repro_path))} (FORMAT PARQUET)
"""
            )
        finally:
            con.close()

        dataset = Dataset(repro_path)
        parameters = default_parameters() + [
            {"name": "num_iterations", "value": 10},
            {"name": "early_stopping_rounds", "value": 0},
        ]
        store = GbmModelStore(repro_path)
        result = train_model(
            dataset,
            store,
            {
                "label": "Poisson no denominator",
                "response": "PREMIUM",
                "offset": "__none__",
                "features": [
                    {"name": "DRIVER_AGE", "include": True, "monotonicity": ""},
                    {"name": "LATITUDE", "include": True, "monotonicity": ""},
                ],
                "parameters": parameters,
                "sample_column": "",
                "shap_rows": "zero",
            },
        )

        self.assertEqual(result["objective"], "poisson")
        self.assertIsNone(result["offset_column"])
        self.assertEqual(result["training_rows"], 50000)
        con = duckdb.connect(database=":memory:")
        try:
            artifact_columns = con.execute(
                f"DESCRIBE SELECT * FROM read_parquet({sql_literal(str(store.artifact_path(result['model_id'], 'predictions')))})"
            ).fetchall()
        finally:
            con.close()
        self.assertEqual([row[0] for row in artifact_columns], ["__lucidum_row_id", "gbm_prediction"])

    def write_model_artifacts(self) -> GbmModelStore:
        store = GbmModelStore(self.data_path)
        model_dir = store.create_model_dir("m1")
        store.write_json(
            model_dir / "manifest.json",
            {
                "model_id": "m1",
                "label": "Model 1",
                "created_at": "2026-05-25T00:00:00Z",
                "objective": "poisson",
                "metric": "poisson",
                "best_iteration": 7,
                "training_rows": 2,
                "test_rows": 1,
                "feature_importance": [{"name": "Age", "gain": 12.345}, {"name": "Segment", "gain": 2.0}],
                "sources": {"predictions": "gbm:m1:predictions"},
            },
        )
        store.write_json(
            model_dir / "feature_config.json",
            [
                {"name": "Age", "kind": "integer", "include": True, "monotonicity": "Increasing", "gain": 12.345},
                {"name": "Segment", "kind": "categorical", "include": True, "monotonicity": "", "gain": 2.0},
            ],
        )
        con = duckdb.connect(database=":memory:")
        try:
            con.execute(
                f"""
COPY (
  SELECT 1 AS __lucidum_row_id, 11.5 AS gbm_prediction
  UNION ALL
  SELECT 2, 21.5
) TO '{model_dir / "predictions.parquet"}' (FORMAT PARQUET)
"""
            )
            con.execute(
                f"""
COPY (
  SELECT 1 AS __lucidum_row_id, 'Age' AS feature, 0.2 AS shap_value, 11.5 AS gbm_prediction
) TO '{model_dir / "shap_values.parquet"}' (FORMAT PARQUET)
"""
            )
            con.execute(
                f"""
COPY (
  SELECT 'Age' AS feature, 0.2 AS mean_abs_shap, 0.2 AS mean_shap, 1 AS row_count
) TO '{model_dir / "shap_summary.parquet"}' (FORMAT PARQUET)
"""
            )
        finally:
            con.close()
        store.activate_model("m1")
        return store

    def test_model_sources_are_exposed_and_chartable(self) -> None:
        store = self.write_model_artifacts()
        app = create_app(self.data_path, token="", tools=["gbm", "line_bar"], use_saved_filters=False, use_kpis=False)

        status, body = asgi_get(app, "/api/schema")
        schema = json.loads(body)
        con = duckdb.connect(database=":memory:")
        try:
            artifact_columns = con.execute(
                f"DESCRIBE SELECT * FROM read_parquet({sql_literal(str(store.artifact_path('m1', 'predictions')))})"
            ).fetchall()
        finally:
            con.close()

        self.assertEqual(status, 200)
        self.assertEqual([row[0] for row in artifact_columns], ["__lucidum_row_id", "gbm_prediction"])
        source_ids = [source["id"] for source in schema["data_sources"]]
        self.assertIn("gbm:m1:predictions", source_ids)
        self.assertIn("gbm:m1:shap_long", source_ids)
        prediction_source = next(source for source in schema["data_sources"] if source["id"] == "gbm:m1:predictions")
        prediction_columns = [column["name"] for column in prediction_source["columns"]]
        self.assertNotIn("__lucidum_row_id", prediction_columns)
        self.assertIn("Segment", prediction_columns)
        self.assertIn("gbm_prediction", prediction_columns)

        dataset = Dataset(self.data_path)
        dataset.register_data_source_provider(GbmSourceProvider(GbmModelStore(self.data_path)))
        result = chart(
            dataset,
            {
                "source": "gbm:m1:predictions",
                "x": "Segment",
                "responses": [{"label": "GBM", "numerator": "gbm_prediction"}],
                "denominator": "__none__",
                "filter": "",
                "bandWidth": 0,
                "dateBucket": "none",
                "lowGroup": "0",
                "sort": "alpha",
                "sigma": 0,
                "transform": "none",
            },
        )

        self.assertEqual(result["source"], "gbm:m1:predictions")
        self.assertEqual([row["x"] for row in result["rows"]], ["A", "B"])

    def test_uk_map_can_use_prediction_source(self) -> None:
        self.write_model_artifacts()
        dataset = Dataset(self.data_path)
        dataset.register_data_source_provider(GbmSourceProvider(GbmModelStore(self.data_path)))

        result = map_summary(
            dataset,
            {
                "source": "gbm:m1:predictions",
                "level": "area",
                "numerator": "gbm_prediction",
                "denominator": "__none__",
                "filter": "",
            },
        )

        self.assertEqual(result["source"], "gbm:m1:predictions")
        self.assertEqual(result["rows"][0]["key"], "AB")


if __name__ == "__main__":
    unittest.main()
