from __future__ import annotations

import asyncio
import json
import time
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any
from unittest.mock import patch

from py_lucidum.app import create_app
from py_lucidum.core import Dataset
from py_lucidum.tools.glm.store import GlmModelStore, GlmSourceProvider
from py_lucidum.tools.glm.training import MissingGlmDependency, glm_dependencies, train_model
from py_lucidum.tools.glm.validation import strip_formula_comments, validate_request


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
    body = b"".join(message.get("body", b"") for message in messages if message["type"] == "http.response.body")
    return status, body


class GlmToolTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.data_path = self.root / "sample.csv"
        self.data_path.write_text(
            "actualNumerator,denominator,Age,Segment,SAMPLE\n"
            "10,100,30,A,training\n"
            "20,200,40,B,test\n"
            "30,300,50,A,training\n"
            "45,300,55,B,training\n"
            "60,400,60,A,test\n"
            "75,500,65,B,training\n",
            encoding="utf-8",
        )

    def require_glm_dependencies(self) -> None:
        try:
            glm_dependencies()
        except MissingGlmDependency as exc:
            self.skipTest(str(exc))

    def test_glm_config_routes_work_without_optional_dependency_imports(self) -> None:
        app = create_app(self.data_path, token="", tools=["glm"], use_saved_filters=False, use_kpis=False)
        paths = {route.path for route in app.routes}

        self.assertIn("/api/glm/config", paths)
        self.assertIn("/api/glm/models", paths)
        self.assertIn("/api/glm/validate", paths)
        self.assertIn("/api/glm/build", paths)
        self.assertIn("/api/glm/jobs/{job_id}", paths)
        self.assertIn("/api/glm/models/{model_id}", paths)
        self.assertIn("/api/glm/models/{model_id}/activate", paths)
        self.assertIn("/api/glm/models/{model_id}/rename", paths)
        model_route_methods: set[str] = set()
        for route in app.routes:
            if route.path == "/api/glm/models/{model_id}":
                model_route_methods.update(getattr(route, "methods", set()))
        self.assertIn("DELETE", model_route_methods)

        with patch("py_lucidum.tools.glm.routes.glm_dependencies", side_effect=AssertionError("should not import glum")):
            status, body = asgi_get(app, "/api/glm/config")
        payload = json.loads(body)

        self.assertEqual(status, 200)
        self.assertEqual(payload["status"], "ready")
        self.assertEqual(payload["sample"]["available"], True)
        self.assertIn("tweedie", [row["value"] for row in payload["families"]])

    def test_glm_build_reports_actionable_missing_dependency(self) -> None:
        app = create_app(self.data_path, token="", tools=["glm"], use_saved_filters=False, use_kpis=False)

        with patch("py_lucidum.tools.glm.routes.glm_dependencies", side_effect=MissingGlmDependency("glum")):
            status, body = asgi_post_json(
                app,
                "/api/glm/build",
                {"formula": "Age", "response_column": "actualNumerator", "family": "normal", "training_scope": "all"},
            )
        payload = json.loads(body)

        self.assertEqual(status, 400)
        self.assertIn("pip install 'py-lucidum[glm]'", payload["detail"])
        self.assertIn("glum", payload["detail"])

    def test_formula_validation_strips_comments_accepts_rhs_and_full_forms_and_rejects_unsafe_text(self) -> None:
        dataset = Dataset(self.data_path)

        self.assertEqual(strip_formula_comments('Age + "literal # kept" # removed'), 'Age + "literal # kept"')
        rhs = validate_request(
            dataset,
            {
                "formula": "# comment\npmin(Age, 60) + ifelse(Segment == 'A', 1, 0)",
                "response_column": "actualNumerator",
                "family": "normal",
                "training_scope": "all",
            },
        )
        full = validate_request(
            dataset,
            {
                "formula": "actualNumerator ~ ns(Age, df=3) + C(Segment)",
                "family": "normal",
                "training_scope": "all",
            },
        )
        unsafe = validate_request(
            dataset,
            {
                "formula": "Age + eval('2 + 2')",
                "response_column": "actualNumerator",
                "family": "normal",
                "training_scope": "all",
            },
        )

        self.assertTrue(rhs["ok"], rhs)
        self.assertEqual(rhs["formula"]["stripped"], "pmin(Age, 60) + ifelse(Segment == 'A', 1, 0)")
        self.assertEqual(rhs["response_column"], "actualNumerator")
        self.assertEqual(rhs["formula"]["fitted"], "__lucidum_glm_target ~ pmin(Age, 60) + ifelse(Segment == 'A', 1, 0)")
        self.assertTrue(full["ok"], full)
        self.assertEqual(full["response_column"], "actualNumerator")
        self.assertEqual(full["formula"]["rhs"], "ns(Age, df=3) + C(Segment)")
        self.assertFalse(unsafe["ok"])
        self.assertIn("unsafe", unsafe["errors"][0])

    def test_training_scope_requires_physical_sample_column(self) -> None:
        no_sample_path = self.root / "no_sample.csv"
        no_sample_path.write_text("actualNumerator,Age\n1,10\n2,20\n", encoding="utf-8")
        dataset = Dataset(no_sample_path)

        result = validate_request(
            dataset,
            {"formula": "Age", "response_column": "actualNumerator", "family": "normal", "training_scope": "training"},
        )

        self.assertFalse(result["ok"])
        self.assertIn("physical SAMPLE column", "; ".join(result["errors"]))

    def test_glm_training_writes_weighted_predictions_and_publishes_source(self) -> None:
        self.require_glm_dependencies()
        dataset = Dataset(self.data_path)
        store = GlmModelStore(self.data_path)

        result = train_model(
            dataset,
            store,
            {
                "formula": "# insurance GLM\npmin(Age, 60) + ifelse(Segment == 'A', 1, 0)",
                "response_column": "actualNumerator",
                "denominator_column": "denominator",
                "family": "normal",
                "training_scope": "training",
            },
        )
        model_id = result["model_id"]
        manifest = store.manifest(model_id)

        self.assertEqual(store.active_model_id(), model_id)
        self.assertEqual(manifest["response_column"], "actualNumerator")
        self.assertEqual(manifest["denominator_column"], "denominator")
        self.assertEqual(manifest["training_scope"], "training")
        self.assertEqual(manifest["diagnostics"]["training_rows"], 4)
        self.assertIn("deviance", manifest["diagnostics"])
        self.assertTrue(store.artifact_path(model_id, "formula").exists())
        self.assertTrue(store.artifact_path(model_id, "coefficients").exists())
        self.assertTrue(store.artifact_path(model_id, "predictions").exists())
        self.assertTrue(store.artifact_path(model_id, "diagnostics").exists())

        detail = store.model_detail(model_id)
        self.assertTrue(detail["coefficients"])
        self.assertEqual(detail["coefficients"][0]["term"], "(Intercept)")

        dataset.register_data_source_provider(GlmSourceProvider(store))
        source_id = store.source_id(model_id)
        model_prediction_source = dataset.model_prediction_source(source_id)
        self.assertIsNotNone(model_prediction_source)
        self.assertEqual(model_prediction_source.column, "glm_prediction")
        self.assertIn("predictions.parquet", model_prediction_source.relation_sql)
        sources = dataset.data_sources()
        glm_source = next(source for source in sources if source["id"] == source_id)
        self.assertEqual(glm_source["kind"], "glm_predictions")
        self.assertTrue(glm_source["active"])
        self.assertIn("glm_prediction", [column["name"] for column in glm_source["columns"]])
        with dataset.lock:
            row = dataset.con.execute(f"SELECT COUNT(*), COUNT(glm_prediction) FROM {dataset.relation_sql_for_source(source_id)}").fetchone()
        self.assertEqual(row, (6, 6))

    def test_glm_model_store_can_activate_rename_and_delete_models(self) -> None:
        self.require_glm_dependencies()
        dataset = Dataset(self.data_path)
        store = GlmModelStore(self.data_path)
        first = train_model(dataset, store, {"formula": "Age", "response_column": "actualNumerator", "family": "normal", "training_scope": "all"}, activate=False)
        second = train_model(dataset, store, {"formula": "Age + C(Segment)", "response_column": "actualNumerator", "family": "normal", "training_scope": "all"})

        store.activate_model(first["model_id"])
        self.assertEqual(store.active_model_id(), first["model_id"])
        renamed = store.rename_model(first["model_id"], "renamed-glm")
        self.assertEqual(renamed["model_id"], "renamed-glm")
        self.assertEqual(store.active_model_id(), "renamed-glm")
        self.assertEqual(renamed["sources"]["predictions"], "glm:renamed-glm:predictions")
        store.delete_model("renamed-glm")
        self.assertEqual(store.active_model_id(), second["model_id"])
        store.delete_model(second["model_id"])
        self.assertIsNone(store.active_model_id())

    def test_glm_api_build_job_and_model_mutations(self) -> None:
        self.require_glm_dependencies()
        app = create_app(self.data_path, token="", tools=["glm"], use_saved_filters=False, use_kpis=False)

        status, body = asgi_post_json(
            app,
            "/api/glm/build",
            {"formula": "Age", "response_column": "actualNumerator", "family": "normal", "training_scope": "all"},
        )
        job = json.loads(body)
        self.assertEqual(status, 200)
        deadline = time.monotonic() + 60
        while time.monotonic() < deadline:
            status, body = asgi_get(app, f"/api/glm/jobs/{job['job_id']}")
            payload = json.loads(body)
            if payload["status"] not in {"queued", "running"}:
                break
            time.sleep(0.05)
        self.assertEqual(payload["status"], "succeeded", payload)
        model_id = payload["result"]["model_id"]

        status, body = asgi_get(app, f"/api/glm/models/{model_id}")
        detail = json.loads(body)
        self.assertEqual(status, 200)
        self.assertTrue(detail["coefficients"])

        status, body = asgi_post_json(app, f"/api/glm/models/{model_id}/rename", {"new_model_id": "api-glm"})
        renamed = json.loads(body)
        self.assertEqual(status, 200)
        self.assertEqual(renamed["model"]["model_id"], "api-glm")

        status, body = asgi_post_json(app, "/api/glm/models/api-glm/activate", {})
        activated = json.loads(body)
        self.assertEqual(status, 200)
        self.assertEqual(activated["model"]["model_id"], "api-glm")

        status, body = asgi_delete(app, "/api/glm/models/api-glm")
        deleted = json.loads(body)
        self.assertEqual(status, 200)
        self.assertEqual(deleted["deleted_model_id"], "api-glm")


if __name__ == "__main__":
    unittest.main()
