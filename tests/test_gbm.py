from __future__ import annotations

import asyncio
import builtins
import json
import time
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any
from unittest.mock import patch

import duckdb

from py_lucidum.app import create_app
from py_lucidum.core import Dataset, sql_literal
from py_lucidum.demo import demo_dataset_path
from py_lucidum.tools.gbm.jobs import GbmJob, GbmJobManager
from py_lucidum.tools.gbm.sample import create_generated_sample, sample_metadata
from py_lucidum.tools.gbm.store import GbmModelStore, GbmSourceProvider
from py_lucidum.tools.gbm.trees import tree_detail, tree_summary
from py_lucidum.tools.gbm.training import MissingGbmDependency, gbm_dependencies, lightgbm_progress_payload, shap_dataframes, shap_row_limit, should_use_offset_init_score, train_model, tree_dataframe, training_projection_columns, training_select_sql, write_dataframe_parquet
from py_lucidum.tools.gbm.validation import GBM_METRICS, GBM_OBJECTIVES, categorical_distinct_counts, default_parameters, feature_rows, normalise_parameters, validate_request
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


def asgi_delete(app: Any, path: str) -> tuple[int, bytes]:
    messages: list[dict[str, Any]] = []

    async def receive() -> dict[str, Any]:
        return {"type": "http.request", "body": b"", "more_body": False}

    async def send(message: dict[str, Any]) -> None:
        messages.append(message)

    scope = {
        "type": "http",
        "asgi": {"version": "3.0", "spec_version": "2.3"},
        "http_version": "1.1",
        "method": "DELETE",
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
    response_body = b"".join(message.get("body", b"") for message in messages if message["type"] == "http.response.body")
    return status, response_body


class GbmToolTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.data_path = self.root / "sample.csv"
        self.data_path.write_text(
            "actualNumerator,denominator,Age,Segment,PostcodeArea,PostcodeSector,PostcodeUnit,lat,long,SAMPLE\n"
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
        self.assertIn("/api/gbm/sample", paths)
        self.assertIn("/api/gbm/models", paths)
        self.assertIn("/api/gbm/models/{model_id}", paths)
        self.assertIn("/api/gbm/models/{model_id}/activate", paths)
        self.assertIn("/api/gbm/models/{model_id}/rename", paths)
        model_route_methods: set[str] = set()
        for route in app.routes:
            if route.path == "/api/gbm/models/{model_id}":
                model_route_methods.update(getattr(route, "methods", set()))
        self.assertIn("DELETE", model_route_methods)

        status, body = asgi_get(app, "/api/gbm/config")
        payload = json.loads(body)

        self.assertEqual(status, 200)
        self.assertEqual(payload["status"], "ready")
        self.assertEqual(payload["response"], "actualNumerator")
        self.assertEqual(payload["offset"], "denominator")
        self.assertEqual(payload["sample_column"], "SAMPLE")
        self.assertEqual(payload["sample"]["source"], "dataset")
        self.assertEqual(
            {level["name"]: level["row_count"] for level in payload["sample"]["levels"]},
            {"training": 2, "test": 1, "validation": 0},
        )
        self.assertEqual(next(row["value"] for row in payload["parameters"] if row["name"] == "objective"), "poisson")
        self.assertEqual(next(row["value"] for row in payload["parameters"] if row["name"] == "metric"), "poisson")
        self.assertEqual(payload["parameter_options"]["objective"], list(GBM_OBJECTIVES))
        self.assertEqual(payload["parameter_options"]["metric"], list(GBM_METRICS))
        self.assertEqual(
            payload["shap_options"],
            [
                {"value": "0", "label": "0"},
                {"value": "10k", "label": "10k"},
                {"value": "100k", "label": "100k"},
                {"value": "all", "label": "All"},
            ],
        )
        self.assertIn("Gain", Path("docs/specs/gbm-tool_plan.md").read_text(encoding="utf-8"))
        age = next(row for row in payload["features"] if row["name"] == "Age")
        self.assertEqual(age["gain"], 0.0)
        sample = next(row for row in payload["features"] if row["name"] == "SAMPLE")
        self.assertFalse(sample["include"])

    def test_generated_sample_sidecar_is_reused_for_missing_sample_column(self) -> None:
        data_path = self.root / "no_sample.csv"
        data_path.write_text(
            "actualNumerator,denominator,Age\n"
            + "".join(f"{index},{index + 10},{20 + index}\n" for index in range(1, 11)),
            encoding="utf-8",
        )
        dataset = Dataset(data_path)
        store = GbmModelStore(data_path)

        missing = sample_metadata(dataset, store.generated_sample_path)
        self.assertEqual(missing["source"], "none")

        generated = create_generated_sample(dataset, store.generated_sample_path)
        self.assertEqual(generated["source"], "generated")
        self.assertEqual(
            {level["name"]: level["row_count"] for level in generated["levels"]},
            {"training": 6, "test": 2, "validation": 2},
        )
        con = duckdb.connect(database=":memory:")
        try:
            first_rows = con.execute(
                f"SELECT * FROM read_parquet({sql_literal(str(store.generated_sample_path))}) ORDER BY __lucidum_row_id"
            ).fetchall()
        finally:
            con.close()

        second = create_generated_sample(dataset, store.generated_sample_path)
        con = duckdb.connect(database=":memory:")
        try:
            second_rows = con.execute(
                f"SELECT * FROM read_parquet({sql_literal(str(store.generated_sample_path))}) ORDER BY __lucidum_row_id"
            ).fetchall()
        finally:
            con.close()

        self.assertEqual(second["source"], "generated")
        self.assertEqual(first_rows, second_rows)

        result = validate_request(
            dataset,
            {
                "response": "actualNumerator",
                "offset": "denominator",
                "features": [{"name": "Age", "include": True, "monotonicity": ""}],
                "sample_column": "SAMPLE",
            },
            generated_sample_path=store.generated_sample_path,
        )
        self.assertTrue(result.ok, result.errors)

    def test_sample_route_creates_generated_sample_and_refreshes_config(self) -> None:
        data_path = self.root / "route_no_sample.csv"
        data_path.write_text(
            "actualNumerator,denominator,Age\n"
            "10,100,30\n"
            "20,200,40\n"
            "30,300,50\n"
            "40,400,60\n"
            "50,500,70\n",
            encoding="utf-8",
        )
        app = create_app(data_path, token="", tools=["gbm"], use_saved_filters=False, use_kpis=False)

        status, body = asgi_get(app, "/api/gbm/config")
        self.assertEqual(status, 200)
        self.assertEqual(json.loads(body)["sample"]["source"], "none")

        status, body = asgi_post_json(app, "/api/gbm/sample", {})
        payload = json.loads(body)

        self.assertEqual(status, 200)
        self.assertEqual(payload["sample"]["source"], "generated")
        self.assertEqual(payload["config"]["sample"]["source"], "generated")

    def test_train_endpoint_reports_missing_optional_dependencies(self) -> None:
        app = create_app(self.data_path, token="", tools=["gbm"], use_saved_filters=False, use_kpis=False)
        request = {
            "features": self.request_features(),
            "parameters": [{"name": "objective", "value": "poisson"}, {"name": "metric", "value": "poisson"}],
            "sample_column": "SAMPLE",
        }

        with patch("py_lucidum.tools.gbm.routes.gbm_dependencies", side_effect=MissingGbmDependency("lightgbm")):
            status, body = asgi_post_json(app, "/api/gbm/train", request)

        self.assertEqual(status, 400)
        self.assertIn("py-lucidum[gbm]", json.loads(body)["detail"])

    def test_gbm_dependencies_reports_missing_lightgbm_runtime(self) -> None:
        real_import = builtins.__import__

        def import_with_missing_lightgbm_runtime(name: str, *args: Any, **kwargs: Any) -> Any:
            if name == "lightgbm":
                raise OSError("Library not loaded: @rpath/libomp.dylib")
            return real_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=import_with_missing_lightgbm_runtime):
            with self.assertRaises(MissingGbmDependency) as raised:
                gbm_dependencies()

        message = str(raised.exception)
        self.assertIn("lightgbm runtime", message)
        self.assertIn("brew install libomp", message)

    def test_gbm_job_payload_includes_progress(self) -> None:
        job = GbmJob(id="j1", status="running", progress={"phase": "training", "iteration": 3})

        payload = job.as_payload()

        self.assertEqual(payload["progress"], {"phase": "training", "iteration": 3})

    def test_gbm_job_manager_records_and_preserves_progress_on_failure(self) -> None:
        dataset = Dataset(self.data_path)
        store = GbmModelStore(self.data_path)
        manager = GbmJobManager()

        def fake_train_model(dataset_arg: Dataset, store_arg: GbmModelStore, payload: dict[str, Any], progress_callback: Any = None) -> dict[str, Any]:
            self.assertIs(dataset_arg, dataset)
            self.assertIs(store_arg, store)
            self.assertEqual(payload, {"label": "broken"})
            progress_callback(
                {
                    "phase": "training",
                    "message": "training, tree 1/2, test poisson 1.2",
                    "iteration": 1,
                    "evaluation": {"test": {"poisson": [1.2]}},
                }
            )
            raise ValueError("boom")

        with patch("py_lucidum.tools.gbm.jobs.train_model", side_effect=fake_train_model):
            job = manager.start(dataset, store, {"label": "broken"})
            for _ in range(100):
                snapshot = manager.get(job.id)
                if snapshot and snapshot.status == "failed":
                    break
                time.sleep(0.01)

        snapshot = manager.get(job.id)
        self.assertIsNotNone(snapshot)
        self.assertEqual(snapshot.status, "failed")
        self.assertEqual(snapshot.error, "boom")
        self.assertEqual(snapshot.progress["phase"], "failed")
        self.assertEqual(snapshot.progress["message"], "boom")
        self.assertEqual(snapshot.progress["iteration"], 1)
        self.assertEqual(snapshot.progress["evaluation"], {"test": {"poisson": [1.2]}})

    def test_gbm_job_route_returns_progress(self) -> None:
        app = create_app(self.data_path, token="", tools=["gbm"], use_saved_filters=False, use_kpis=False)
        job = GbmJob(
            id="progress-job",
            status="running",
            progress={
                "phase": "training",
                "message": "training, tree 2/10, test poisson 1.1",
                "iteration": 2,
                "total_iterations": 10,
                "percent": 18,
                "metric": "poisson",
                "latest": [{"dataset": "test", "metric": "poisson", "value": 1.1}],
                "evaluation": {"training": {"poisson": [1.3, 1.2]}, "test": {"poisson": [1.2, 1.1]}},
            },
        )
        app.state.gbm_jobs._jobs[job.id] = job

        status, body = asgi_get(app, "/api/gbm/jobs/progress-job")
        payload = json.loads(body)

        self.assertEqual(status, 200)
        self.assertEqual(payload["progress"]["phase"], "training")
        self.assertEqual(payload["progress"]["iteration"], 2)
        self.assertEqual(payload["progress"]["latest"][0]["value"], 1.1)

    def test_lightgbm_progress_payload_is_json_safe(self) -> None:
        try:
            import numpy as np
        except ImportError as exc:  # pragma: no cover - optional dependency guard.
            self.skipTest(str(exc))

        class Env:
            iteration = 2
            begin_iteration = 0
            evaluation_result_list = [
                ("training", "poisson", np.float64(1.3), False),
                ("test", "poisson", np.float64(1.2), False),
            ]

        payload = lightgbm_progress_payload(
            Env(),
            metric_name="poisson",
            total_iterations=10,
            evaluation_result={"training": {"poisson": [np.float64(1.4), np.float64(1.3)]}, "test": {"poisson": [np.float64(1.2)]}},
        )

        self.assertEqual(payload["phase"], "training")
        self.assertEqual(payload["iteration"], 3)
        self.assertEqual(payload["total_iterations"], 10)
        self.assertEqual(payload["percent"], 27.0)
        self.assertEqual(payload["latest"], [{"dataset": "test", "metric": "poisson", "value": 1.2}, {"dataset": "training", "metric": "poisson", "value": 1.3}])
        self.assertEqual(payload["evaluation"], {"training": {"poisson": [1.4, 1.3]}, "test": {"poisson": [1.2]}})
        self.assertEqual(payload["message"], "training, tree 3/10, test poisson 1.2")

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
                "sample_column": "SAMPLE",
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
                "sample_column": "SAMPLE",
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
                "sample_column": "SAMPLE",
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
        self.assertEqual(next(row for row in rows if row["name"] == "SAMPLE")["gain"], 0.0)

    def test_feature_rows_include_invalid_columns_without_counting_them(self) -> None:
        original_probe = Dataset.probe_column_readable

        def fake_probe(dataset: Dataset, column: Any) -> None:
            if column.name == "Segment":
                raise duckdb.InvalidInputException(
                    'Invalid Input Error: Invalid string encoding found in Parquet file "/tmp/bad.parquet": '
                    'value "bad" is not valid UTF8!'
                )
            original_probe(dataset, column)

        with patch.object(Dataset, "probe_column_readable", fake_probe):
            dataset = Dataset(self.data_path)
            counts = categorical_distinct_counts(dataset)
            rows = feature_rows(dataset, {"Segment": 1.5, "Age": 9.25})
            result = validate_request(
                dataset,
                {
                    "response": "actualNumerator",
                    "offset": "denominator",
                    "features": [{"name": "Segment", "include": True, "monotonicity": ""}],
                    "parameters": default_parameters(),
                    "sample_column": "SAMPLE",
                },
            )

        by_name = {row["name"]: row for row in rows}
        self.assertNotIn("Segment", counts)
        self.assertEqual(by_name["Segment"]["kind"], "invalid")
        self.assertTrue(by_name["Segment"]["invalid"])
        self.assertFalse(by_name["Segment"]["usable"])
        self.assertFalse(by_name["Segment"]["include"])
        self.assertEqual(by_name["Segment"]["disabled_reason"], "Invalid string encoding found in Parquet data.")
        self.assertIn("valid GBM feature: Segment", "; ".join(result.errors))

    def test_gbm_config_returns_invalid_feature_rows(self) -> None:
        original_probe = Dataset.probe_column_readable

        def fake_probe(dataset: Dataset, column: Any) -> None:
            if column.name == "Segment":
                raise duckdb.InvalidInputException(
                    'Invalid Input Error: Invalid string encoding found in Parquet file "/tmp/bad.parquet": '
                    'value "bad" is not valid UTF8!'
                )
            original_probe(dataset, column)

        with patch.object(Dataset, "probe_column_readable", fake_probe):
            app = create_app(self.data_path, token="", tools=["gbm"], use_saved_filters=False, use_kpis=False)
            status, body = asgi_get(app, "/api/gbm/config")

        payload = json.loads(body)
        segment = next(row for row in payload["features"] if row["name"] == "Segment")

        self.assertEqual(status, 200)
        self.assertEqual(segment["kind"], "invalid")
        self.assertTrue(segment["invalid"])
        self.assertFalse(segment["usable"])
        self.assertFalse(segment["include"])
        self.assertEqual(segment["disabled_reason"], "Invalid string encoding found in Parquet data.")

    def test_training_projection_omits_unselected_invalid_columns(self) -> None:
        dataset = Dataset(self.data_path)
        dataset.record_invalid_column("Segment", "Invalid string encoding found in Parquet data.")
        columns = dataset.column_map()
        projection = training_projection_columns(
            response_col="actualNumerator",
            offset_col="denominator",
            sample_column="SAMPLE",
            feature_names=["Age"],
            columns=columns,
        )
        sql = training_select_sql(dataset.relation_sql(), projection, "\nWHERE TRY_CAST(denominator AS DOUBLE) > 0")

        self.assertEqual(projection, ["actualNumerator", "denominator", "SAMPLE", "Age"])
        self.assertNotIn("*", sql)
        self.assertNotIn("Segment", sql)
        self.assertIn('"Age"', sql)

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
        self.assertFalse(by_name["SAMPLE"]["include"])
        self.assertEqual(by_name["SAMPLE"]["gain"], 0.0)

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
        self.assertFalse(features["SAMPLE"]["include"])

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
                    "response_column": "actualNumerator",
                    "offset_column": "denominator",
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

    def test_rename_active_model_updates_folder_manifest_sources_and_schema(self) -> None:
        store = self.write_model_artifacts()
        con = duckdb.connect(database=":memory:")
        try:
            con.execute(
                f"""
COPY (
  SELECT 'm1' AS gbm_model_id, 'Age' AS feature, 0.2 AS mean_abs_shap, 0.2 AS mean_shap, 1 AS row_count
) TO {sql_literal(str(store.artifact_path("m1", "shap_summary")))} (FORMAT PARQUET)
"""
            )
        finally:
            con.close()
        app = create_app(self.data_path, token="", tools=["gbm", "line_bar"], use_saved_filters=False, use_kpis=False)

        status, body = asgi_post_json(app, "/api/gbm/models/m1/rename", {"new_model_id": "renamed-model"})
        payload = json.loads(body)

        self.assertEqual(status, 200)
        self.assertFalse(store.model_dir("m1").exists())
        self.assertTrue(store.model_dir("renamed-model").exists())
        self.assertEqual(payload["config"]["active_model_id"], "renamed-model")
        self.assertEqual(payload["model"]["model_id"], "renamed-model")
        self.assertEqual(payload["model"]["label"], "renamed-model")
        self.assertEqual(payload["model"]["sources"]["predictions"], "gbm:renamed-model:predictions")
        manifest = store.manifest("renamed-model")
        self.assertEqual(manifest["model_id"], "renamed-model")
        self.assertEqual(manifest["label"], "renamed-model")
        self.assertEqual(manifest["sources"]["predictions"], "gbm:renamed-model:predictions")
        self.assertEqual(store.active_model_id(), "renamed-model")

        shap_summary_path = store.artifact_path("renamed-model", "shap_summary")
        con = duckdb.connect(database=":memory:")
        try:
            shap_ids = con.execute(
                f"SELECT DISTINCT gbm_model_id FROM read_parquet({sql_literal(str(shap_summary_path))})"
            ).fetchall()
        finally:
            con.close()
        self.assertEqual(shap_ids, [("renamed-model",)])

        schema_status, schema_body = asgi_get(app, "/api/schema")
        source_ids = [source["id"] for source in json.loads(schema_body)["data_sources"]]
        self.assertEqual(schema_status, 200)
        self.assertIn("gbm:renamed-model:predictions", source_ids)
        self.assertIn("gbm:renamed-model:shap_long", source_ids)
        self.assertNotIn("gbm:m1:predictions", source_ids)
        prediction_source = next(source for source in json.loads(schema_body)["data_sources"] if source["id"] == "gbm:renamed-model:predictions")
        self.assertEqual(prediction_source["label"], "renamed-model - Predictions")

    def test_rename_model_rejects_invalid_duplicate_and_missing_models(self) -> None:
        store = self.write_model_artifacts()
        store.create_model_dir("taken")
        app = create_app(self.data_path, token="", tools=["gbm"], use_saved_filters=False, use_kpis=False)

        invalid_status, invalid_body = asgi_post_json(app, "/api/gbm/models/m1/rename", {"new_model_id": "bad/name"})
        duplicate_status, duplicate_body = asgi_post_json(app, "/api/gbm/models/m1/rename", {"new_model_id": "taken"})
        missing_status, missing_body = asgi_post_json(app, "/api/gbm/models/missing/rename", {"new_model_id": "renamed"})

        self.assertEqual(invalid_status, 400)
        self.assertIn("valid GBM model name", json.loads(invalid_body)["detail"])
        self.assertEqual(duplicate_status, 400)
        self.assertIn("already exists", json.loads(duplicate_body)["detail"])
        self.assertEqual(missing_status, 404)
        self.assertIn("valid GBM model", json.loads(missing_body)["detail"])

    def test_delete_active_model_promotes_newest_remaining_model(self) -> None:
        store = GbmModelStore(self.data_path)
        for model_id, created_at in (
            ("older", "2026-05-25T00:00:00Z"),
            ("newer", "2026-05-25T00:00:02Z"),
        ):
            model_dir = store.create_model_dir(model_id)
            store.write_json(
                model_dir / "manifest.json",
                {
                    "model_id": model_id,
                    "label": model_id,
                    "created_at": created_at,
                    "objective": "poisson",
                    "metric": "poisson",
                    "response_column": "actualNumerator",
                    "offset_column": "denominator",
                    "best_iteration": 3,
                    "training_rows": 2,
                    "sources": {},
                },
            )
        store.activate_model("older")
        app = create_app(self.data_path, token="", tools=["gbm"], use_saved_filters=False, use_kpis=False)

        status, body = asgi_delete(app, "/api/gbm/models/older")
        payload = json.loads(body)

        self.assertEqual(status, 200)
        self.assertFalse(store.model_dir("older").exists())
        self.assertTrue(store.model_dir("newer").exists())
        self.assertEqual(payload["config"]["active_model_id"], "newer")
        self.assertEqual(store.active_model_id(), "newer")

    def test_delete_final_model_clears_active_model_and_schema_sources(self) -> None:
        store = self.write_model_artifacts()
        app = create_app(self.data_path, token="", tools=["gbm", "line_bar"], use_saved_filters=False, use_kpis=False)

        status, body = asgi_delete(app, "/api/gbm/models/m1")
        payload = json.loads(body)

        self.assertEqual(status, 200)
        self.assertFalse(store.model_dir("m1").exists())
        self.assertIsNone(payload["config"]["active_model_id"])
        self.assertEqual(payload["config"]["models"], [])
        self.assertIsNone(store.active_model_id())

        schema_status, schema_body = asgi_get(app, "/api/schema")
        source_ids = [source["id"] for source in json.loads(schema_body)["data_sources"]]
        self.assertEqual(schema_status, 200)
        self.assertNotIn("gbm:m1:predictions", source_ids)

    def test_validation_accepts_sidebar_response_and_no_denominator(self) -> None:
        dataset = Dataset(self.data_path)

        result = validate_request(
            dataset,
            {
                "response": "Age",
                "offset": "__none__",
                "features": [{"name": "Segment", "include": True, "monotonicity": ""}],
                "parameters": [{"name": "objective", "value": "poisson"}],
                "sample_column": "SAMPLE",
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
        progress: list[dict[str, Any]] = []
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
                "shap_rows": "0",
            },
            progress_callback=progress.append,
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
        self.assertFalse((store.model_dir(result["model_id"]) / "tree_dump.json").exists())
        self.assertTrue(any(item.get("phase") == "training" for item in progress))
        self.assertTrue(any(item.get("phase") == "scoring" for item in progress))
        self.assertTrue(any(item.get("phase") == "artifacts" for item in progress))
        training_progress = next(item for item in progress if item.get("phase") == "training")
        self.assertIn("iteration", training_progress)
        self.assertIn("evaluation", training_progress)

    def test_training_with_default_features_does_not_read_invalid_unselected_column(self) -> None:
        try:
            import lightgbm  # noqa: F401
        except ImportError:
            self.skipTest("LightGBM is not installed")

        data_path = self.root / "invalid_unused.csv"
        data_path.write_text(
            "actualNumerator,denominator,Age,BadText,SAMPLE\n"
            "10,100,30,bad,training\n"
            "20,200,40,bad,test\n"
            "30,300,50,bad,training\n",
            encoding="utf-8",
        )
        dataset = Dataset(data_path)
        dataset.record_invalid_column("BadText", "Invalid string encoding found in Parquet data.")
        features = feature_rows(dataset)

        class GuardedConnection:
            def __init__(self, inner: Any):
                self.inner = inner
                self.sql: list[str] = []

            def execute(self, sql: str, *args: Any, **kwargs: Any) -> Any:
                text = str(sql)
                self.sql.append(text)
                if "BadText" in text or "ROW_NUMBER() OVER () AS __lucidum_row_id,\n  *" in text:
                    raise AssertionError(text)
                return self.inner.execute(sql, *args, **kwargs)

            def __getattr__(self, name: str) -> Any:
                return getattr(self.inner, name)

        guarded = GuardedConnection(dataset.con)
        dataset.con = guarded  # type: ignore[assignment]
        parameters = default_parameters() + [
            {"name": "num_iterations", "value": 3},
            {"name": "early_stopping_rounds", "value": 0},
            {"name": "min_data_in_leaf", "value": 1},
        ]
        store = GbmModelStore(data_path)

        result = train_model(
            dataset,
            store,
            {
                "label": "Invalid unused",
                "response": "actualNumerator",
                "offset": "denominator",
                "features": features,
                "parameters": parameters,
                "sample_column": "SAMPLE",
                "shap_rows": "0",
            },
        )

        self.assertIn("Age", result["source_columns"])
        self.assertNotIn("BadText", result["source_columns"])
        self.assertTrue(any("__lucidum_row_id" in sql for sql in guarded.sql))

    def test_shap_row_limit_supports_compact_choices(self) -> None:
        self.assertEqual(shap_row_limit("0", 123456), 0)
        self.assertEqual(shap_row_limit("10k", 123456), 10000)
        self.assertEqual(shap_row_limit("100k", 123456), 100000)
        self.assertEqual(shap_row_limit("all", 123456), 123456)

    def test_shap_values_are_written_as_wide_numeric_feature_columns(self) -> None:
        try:
            import numpy as np
            import pandas as pd
        except ImportError as exc:  # pragma: no cover - optional dependency guard.
            self.skipTest(str(exc))

        test_case = self

        class Booster:
            def predict(self, frame: Any, *, pred_contrib: bool, num_iteration: int) -> Any:
                test_case.assertTrue(pred_contrib)
                test_case.assertEqual(num_iteration, 3)
                test_case.assertEqual(list(frame.columns), ["Age", "Segment"])
                return np.array([[0.2, -0.1, 0.0], [0.5, 0.3, 0.0]])

        shap_frame, summary = shap_dataframes(
            np=np,
            pd=pd,
            booster=Booster(),
            feature_frame=pd.DataFrame({"Age": [30, 40], "Segment": ["A", "GU"]}),
            score_frame=pd.DataFrame({"__lucidum_row_id": [1, 2], "Age": [30, 40], "Segment": ["A", "GU"]}),
            feature_names=["Age", "Segment"],
            model_id="m1",
            shap_mode="10k",
            best_iteration=3,
        )

        self.assertEqual(list(shap_frame.columns), ["__lucidum_row_id", "Age", "Segment"])
        self.assertEqual(shap_frame["__lucidum_row_id"].tolist(), [1, 2])
        self.assertEqual(shap_frame["Age"].tolist(), [0.2, 0.5])
        self.assertNotIn("feature_value", shap_frame.columns)
        self.assertEqual(set(summary["feature"]), {"Age", "Segment"})

        shap_path = self.root / "shap_values.parquet"
        write_dataframe_parquet(shap_frame, shap_path)
        con = duckdb.connect(database=":memory:")
        try:
            artifact_columns = con.execute(f"DESCRIBE SELECT * FROM read_parquet({sql_literal(str(shap_path))})").fetchall()
        finally:
            con.close()
        self.assertEqual([row[0] for row in artifact_columns], ["__lucidum_row_id", "Age", "Segment"])
        self.assertTrue(str(artifact_columns[0][1]).startswith("BIGINT"))
        self.assertTrue(str(artifact_columns[1][1]).startswith("DOUBLE"))

    def test_tree_dataframe_adds_decoded_categorical_threshold_labels(self) -> None:
        try:
            import pandas as pd
        except ImportError as exc:  # pragma: no cover - optional dependency guard.
            self.skipTest(str(exc))

        class Booster:
            def trees_to_dataframe(self) -> Any:
                return pd.DataFrame(
                    [
                        {
                            "tree_index": 0,
                            "node_depth": 1,
                            "node_index": "0-S0",
                            "split_feature": "Segment",
                            "threshold": "0||2",
                            "decision_type": "==",
                            "value": 1.2,
                        },
                        {
                            "tree_index": 0,
                            "node_depth": 2,
                            "node_index": "0-S1",
                            "split_feature": "Age",
                            "threshold": "35",
                            "decision_type": "<=",
                            "value": 1.6,
                        },
                    ]
                )

        frame = tree_dataframe(pd, Booster(), categorical_labels={"Segment": ["A", "B", "C"]})

        labels = frame["threshold_label"].tolist()
        self.assertEqual(labels[0], "A / C")
        self.assertTrue(pd.isna(labels[1]))

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
                "response_column": "actualNumerator",
                "offset_column": "denominator",
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
  SELECT 1 AS __lucidum_row_id, 0.2 AS Age, -0.1 AS Segment
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
            con.execute(
                f"""
COPY (
  SELECT 0 AS tree_index, 1 AS node_depth, '0-S0' AS node_index, '0-L0' AS left_child, '0-S1' AS right_child,
         NULL AS parent_index, 'Segment' AS split_feature, 6.5 AS split_gain, '0||2' AS threshold,
         'A / C' AS threshold_label, '==' AS decision_type, 'left' AS missing_direction, 'None' AS missing_type,
         1.2 AS value, 3.0 AS weight, 3 AS count
  UNION ALL
  SELECT 0, 2, '0-L0', NULL, NULL, '0-S0', NULL, NULL, NULL, NULL, NULL, NULL, NULL, 0.8, 2.0, 2
  UNION ALL
  SELECT 0, 2, '0-S1', '0-L1', '0-L2', '0-S0', 'Age', 2.5, '35', NULL, '<=', 'right', 'None', 1.6, 1.0, 1
  UNION ALL
  SELECT 0, 3, '0-L1', NULL, NULL, '0-S1', NULL, NULL, NULL, NULL, NULL, NULL, NULL, 1.3, 1.0, 1
  UNION ALL
  SELECT 0, 3, '0-L2', NULL, NULL, '0-S1', NULL, NULL, NULL, NULL, NULL, NULL, NULL, 1.9, 1.0, 1
) TO {sql_literal(str(model_dir / "tree_table.parquet"))} (FORMAT PARQUET)
"""
            )
        finally:
            con.close()
        store.activate_model("m1")
        return store

    def test_tree_summary_and_detail_read_saved_artifacts(self) -> None:
        store = self.write_model_artifacts()

        summary = tree_summary(store, "m1")
        detail = tree_detail(store, "m1", 0)

        self.assertEqual(summary["trees"], [{"tree": 0, "dim": 2, "features": "Segment x Age", "gain": 9}])
        self.assertEqual(detail["tree"], 0)
        self.assertEqual(detail["root"]["type"], "split")
        self.assertEqual(detail["root"]["feature"], "Segment")
        self.assertEqual(detail["root"]["threshold"], "A / C")
        self.assertEqual(detail["root"]["children"][0]["edge_label"], "== A / C")
        self.assertTrue(detail["root"]["children"][0]["default_branch"])
        self.assertEqual(detail["root"]["children"][1]["type"], "split")
        self.assertEqual(detail["root"]["children"][1]["feature"], "Age")
        self.assertIn("Tree 0", detail["root"]["label"])
        self.assertIn(1.9, detail["values"])

    def test_tree_detail_falls_back_to_raw_categorical_codes_without_threshold_label(self) -> None:
        store = GbmModelStore(self.data_path)
        model_dir = store.create_model_dir("old-table")
        store.write_json(
            model_dir / "manifest.json",
            {
                "model_id": "old-table",
                "label": "Old table",
                "created_at": "2026-05-25T00:00:00Z",
                "objective": "poisson",
                "metric": "poisson",
                "response_column": "actualNumerator",
                "offset_column": "denominator",
                "sources": {},
            },
        )
        con = duckdb.connect(database=":memory:")
        try:
            con.execute(
                f"""
COPY (
  SELECT 0 AS tree_index, 1 AS node_depth, '0-S0' AS node_index, '0-L0' AS left_child, '0-L1' AS right_child,
         NULL AS parent_index, 'Segment' AS split_feature, 6.5 AS split_gain, '0||2' AS threshold,
         '==' AS decision_type, 'left' AS missing_direction, 'None' AS missing_type, 1.2 AS value, 3.0 AS weight, 3 AS count
  UNION ALL
  SELECT 0, 2, '0-L0', NULL, NULL, '0-S0', NULL, NULL, NULL, NULL, NULL, NULL, 0.8, 2.0, 2
  UNION ALL
  SELECT 0, 2, '0-L1', NULL, NULL, '0-S0', NULL, NULL, NULL, NULL, NULL, NULL, 1.9, 1.0, 1
) TO {sql_literal(str(model_dir / "tree_table.parquet"))} (FORMAT PARQUET)
"""
            )
        finally:
            con.close()

        detail = tree_detail(store, "old-table", 0)

        self.assertEqual(detail["root"]["threshold"], "0 / 2")
        self.assertEqual(detail["root"]["children"][0]["edge_label"], "== 0 / 2")

    def test_tree_routes_work_without_lightgbm_imports(self) -> None:
        self.write_model_artifacts()
        app = create_app(self.data_path, token="", tools=["gbm"], use_saved_filters=False, use_kpis=False)

        with patch("py_lucidum.tools.gbm.routes.gbm_dependencies", side_effect=AssertionError("should not import")):
            summary_status, summary_body = asgi_get(app, "/api/gbm/models/m1/trees")
            detail_status, detail_body = asgi_get(app, "/api/gbm/models/m1/trees/0")

        self.assertEqual(summary_status, 200)
        self.assertEqual(detail_status, 200)
        self.assertEqual(json.loads(summary_body)["trees"][0]["features"], "Segment x Age")
        self.assertEqual(json.loads(detail_body)["root"]["feature"], "Segment")

    def test_tree_endpoints_return_empty_payloads_for_missing_artifacts(self) -> None:
        store = GbmModelStore(self.data_path)
        model_dir = store.create_model_dir("empty")
        store.write_json(
            model_dir / "manifest.json",
            {
                "model_id": "empty",
                "label": "Empty",
                "created_at": "2026-05-25T00:00:00Z",
                "objective": "poisson",
                "metric": "poisson",
                "response_column": "actualNumerator",
                "offset_column": "denominator",
                "sources": {},
            },
        )

        self.assertEqual(tree_summary(store, "empty"), {"model_id": "empty", "trees": []})
        self.assertEqual(tree_detail(store, "empty", 0), {"model_id": "empty", "tree": 0, "root": None, "values": []})

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
        self.assertEqual(prediction_source["response_column"], "actualNumerator")
        self.assertEqual(prediction_source["offset_column"], "denominator")
        self.assertEqual(prediction_source["metric"], "poisson")
        self.assertEqual(prediction_source["best_iteration"], 7)
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

    def test_prediction_source_relation_uses_safe_explicit_projection(self) -> None:
        store = self.write_model_artifacts()
        original_probe = Dataset.probe_column_readable

        def fake_probe(dataset: Dataset, column: Any) -> None:
            if column.name == "Segment":
                raise duckdb.InvalidInputException(
                    'Invalid Input Error: Invalid string encoding found in Parquet file "/tmp/bad.parquet": '
                    'value "bad" is not valid UTF8!'
                )
            original_probe(dataset, column)

        with patch.object(Dataset, "probe_column_readable", fake_probe):
            relation_sql = store.relation_sql("gbm:m1:predictions")

        self.assertNotIn("base.*", relation_sql)
        self.assertNotIn('"Segment"', relation_sql)
        self.assertIn('base."Age"', relation_sql)
        self.assertIn("prediction.gbm_prediction", relation_sql)

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
