from __future__ import annotations

import asyncio
import duckdb
import importlib.util
import json
import os
import subprocess
import sys
import time
import unittest
import warnings
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any
from unittest.mock import patch

from py_lucidum.app import create_app
from py_lucidum.core import Dataset, sql_literal
from py_lucidum.tools.glm import tabulation as glm_tabulation
from py_lucidum.tools.glm.store import GlmModelStore, GlmSourceProvider
from py_lucidum.tools.glm.tabulation import build_tabulations, tabulation_config, tabulation_plot, tabulation_table
from py_lucidum.tools.glm.training import (
    MissingGlmDependency,
    _suppress_tabmat_mixed_dtype_warning,
    glm_dependencies,
    train_model,
)
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

    def require_glm_and_gbm_dependencies(self) -> None:
        missing = [
            name
            for name in ("glum", "lightgbm", "numpy", "pandas")
            if importlib.util.find_spec(name) is None
        ]
        if missing:
            self.skipTest(f"missing optional modelling dependencies: {', '.join(missing)}")

    def spline_data_path(self) -> Path:
        path = self.root / "spline.csv"
        rows = ["y,x,Segment\n"]
        for x_value in range(1, 51):
            segment = "A" if x_value <= 30 else "B"
            y_value = 100 + x_value + (5 if segment == "B" else 0)
            rows.append(f"{y_value},{x_value},{segment}\n")
        path.write_text("".join(rows), encoding="utf-8")
        return path

    def categorical_unseen_data_path(self) -> Path:
        path = self.root / "categorical_unseen.csv"
        path.write_text(
            "y,Segment,SAMPLE\n"
            "10,A,training\n"
            "20,B,training\n",
            encoding="utf-8",
        )
        return path

    def test_glm_dependency_load_order_keeps_threaded_lightgbm_stable(self) -> None:
        self.require_glm_and_gbm_dependencies()
        script = r"""
import threading

from py_lucidum.tools.glm.training import glm_dependencies

glm_dependencies()

import lightgbm as lgb
import numpy as np
import pandas as pd

rng = np.random.default_rng(123)
frame = pd.DataFrame({
    "a": rng.normal(size=1000),
    "b": rng.integers(0, 20, size=1000),
    "c": rng.normal(size=1000),
})
target = np.exp(1 + 0.1 * frame["a"].to_numpy() - 0.02 * frame["b"].to_numpy())
result = {}


def run() -> None:
    dataset = lgb.Dataset(frame, label=target, free_raw_data=False)
    booster = lgb.train(
        {"objective": "poisson", "metric": "mape", "verbosity": -1, "num_threads": 2},
        dataset,
        num_boost_round=10,
        valid_sets=[dataset],
        valid_names=["training"],
        callbacks=[lgb.log_evaluation(period=0)],
    )
    result["iteration"] = booster.current_iteration()


thread = threading.Thread(target=run)
thread.start()
thread.join()
if result.get("iteration") != 10:
    raise SystemExit(f"unexpected LightGBM iteration: {result!r}")
"""
        env = os.environ.copy()
        env["PYTHONFAULTHANDLER"] = "1"
        env["PYTHONUNBUFFERED"] = "1"
        completed = subprocess.run(
            [sys.executable, "-c", script],
            check=False,
            capture_output=True,
            text=True,
            env=env,
            timeout=30,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr or completed.stdout)

    def test_glm_suppresses_only_tabmat_mixed_dtype_warning(self) -> None:
        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter("always")
            with _suppress_tabmat_mixed_dtype_warning():
                warnings.warn_explicit(
                    "Matrices do not all have the same dtype. Dtypes are [dtype('float64'), dtype('int64')].",
                    UserWarning,
                    "split_matrix.py",
                    206,
                    module="tabmat.split_matrix",
                )
                warnings.warn_explicit(
                    "Different tabmat warning",
                    UserWarning,
                    "split_matrix.py",
                    206,
                    module="tabmat.split_matrix",
                )
                warnings.warn_explicit(
                    "Matrices do not all have the same dtype. Dtypes are [dtype('float64'), dtype('int64')].",
                    UserWarning,
                    "other.py",
                    1,
                    module="other.module",
                )

        messages = [str(warning.message) for warning in captured]
        self.assertEqual(len(messages), 2)
        self.assertIn("Different tabmat warning", messages)
        self.assertTrue(
            any(message.startswith("Matrices do not all have the same dtype.") for message in messages)
        )

    def test_glm_config_routes_work_without_optional_dependency_imports(self) -> None:
        app = create_app(self.data_path, token="", tools=["glm"], use_saved_filters=False, use_kpis=False)
        paths = {route.path for route in app.routes}

        self.assertIn("/api/glm/config", paths)
        self.assertIn("/api/glm/models", paths)
        self.assertIn("/api/glm/validate", paths)
        self.assertIn("/api/glm/build", paths)
        self.assertIn("/api/glm/jobs/{job_id}", paths)
        self.assertIn("/api/glm/tabulations/build", paths)
        self.assertIn("/api/glm/tabulations/jobs/{job_id}", paths)
        self.assertIn("/api/glm/tabulations/config", paths)
        self.assertIn("/api/glm/tabulations/table", paths)
        self.assertIn("/api/glm/tabulations/plot", paths)
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
        self.assertIn("regularization", payload)
        self.assertEqual(payload["regularization"]["auto_l1_ratio"], [0.0, 0.5, 1.0])

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
        offset = validate_request(
            dataset,
            {
                "formula": "actualNumerator ~ Age + offset(log(denominator))",
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
        self.assertTrue(offset["ok"], offset)
        self.assertEqual(offset["formula"]["rhs"], "Age + offset(log(denominator))")
        self.assertEqual(offset["formula"]["fitted"], "__lucidum_glm_target ~ Age")
        self.assertEqual(offset["formula"]["offset_terms"], ["log(denominator)"])

    def test_regularization_validation_defaults_and_rejects_invalid_manual_values(self) -> None:
        dataset = Dataset(self.data_path)
        base_payload = {
            "formula": "Age",
            "response_column": "actualNumerator",
            "family": "normal",
            "training_scope": "all",
        }

        default = validate_request(dataset, base_payload)
        auto = validate_request(dataset, {**base_payload, "regularization": {"mode": "auto"}})
        manual = validate_request(dataset, {**base_payload, "regularization": {"mode": "manual", "alpha": "0.25", "l1_ratio": "1"}})
        bad_alpha = validate_request(dataset, {**base_payload, "regularization": {"mode": "manual", "alpha": "0", "l1_ratio": "0.5"}})
        bad_mix = validate_request(dataset, {**base_payload, "regularization": {"mode": "manual", "alpha": "0.1", "l1_ratio": "1.5"}})

        self.assertTrue(default["ok"], default)
        self.assertEqual(default["regularization"]["mode"], "none")
        self.assertEqual(default["regularization"]["alpha"], 0.0)
        self.assertTrue(auto["ok"], auto)
        self.assertEqual(auto["regularization"]["mode"], "auto")
        self.assertEqual(auto["regularization"]["l1_ratio"], [0.0, 0.5, 1.0])
        self.assertTrue(manual["ok"], manual)
        self.assertEqual(manual["regularization"]["alpha"], 0.25)
        self.assertEqual(manual["regularization"]["l1_ratio"], 1.0)
        self.assertFalse(bad_alpha["ok"])
        self.assertIn("positive", "; ".join(bad_alpha["errors"]))
        self.assertFalse(bad_mix["ok"])
        self.assertIn("mix", "; ".join(bad_mix["errors"]))

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
        self.assertEqual(manifest["regularization"]["mode"], "none")
        self.assertEqual(manifest["diagnostics"]["training_rows"], 4)
        self.assertIn("deviance", manifest["diagnostics"])
        self.assertTrue(store.artifact_path(model_id, "formula").exists())
        self.assertTrue(store.artifact_path(model_id, "estimator").exists())
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

    def test_glm_training_with_offset_stores_terms_and_predicts(self) -> None:
        self.require_glm_dependencies()
        dataset = Dataset(self.data_path)
        store = GlmModelStore(self.data_path)

        result = train_model(
            dataset,
            store,
            {
                "formula": "actualNumerator ~ Age + offset(log(denominator))",
                "family": "normal",
                "training_scope": "all",
            },
        )
        model_id = result["model_id"]
        manifest = store.manifest(model_id)

        self.assertEqual(manifest["formula"]["fitted"], "__lucidum_glm_target ~ Age")
        self.assertEqual(manifest["offset_terms"], ["log(denominator)"])
        with dataset.lock:
            rows = dataset.con.execute(
                f"SELECT COUNT(*), COUNT(glm_prediction) FROM read_parquet({sql_literal(str(store.artifact_path(model_id, 'predictions')))})"
            ).fetchone()
        self.assertEqual(rows, (6, 6))

    def test_glm_tabulations_persist_predictions_and_publish_source_column(self) -> None:
        self.require_glm_dependencies()
        dataset = Dataset(self.data_path)
        store = GlmModelStore(self.data_path)
        result = train_model(
            dataset,
            store,
            {"formula": "Age + C(Segment)", "response_column": "actualNumerator", "family": "normal", "training_scope": "all"},
        )
        model_id = result["model_id"]
        feature_spec = {
            "rows": [
                {"feature": "Age", "grouping": "Driver", "base": "40", "min": "30", "max": "70", "banding": "5"},
                {"feature": "Segment", "grouping": "Driver", "base": "A"},
            ]
        }

        payload = build_tabulations(dataset, store, {"model_ids": [model_id]}, feature_spec)
        tab_manifest = store.read_json(store.artifact_path(model_id, "tabulation_manifest"))

        self.assertEqual(payload["model_ids"], [model_id])
        self.assertTrue(store.artifact_path(model_id, "tabulated_predictions").exists())
        self.assertIn("linear_sd_error", tab_manifest["diagnostics"])
        self.assertIn("base", [table["table_id"] for table in tab_manifest["tables"]])
        self.assertIn("Age", [table["table_id"] for table in tab_manifest["tables"]])

        dataset.register_data_source_provider(GlmSourceProvider(store))
        source_id = store.source_id(model_id)
        columns = [column["name"] for column in dataset.schema_for_source(source_id)["columns"]]
        self.assertIn("glm_prediction", columns)
        self.assertIn("glm_tabulated_prediction", columns)
        with dataset.lock:
            rows = dataset.con.execute(f"SELECT COUNT(glm_tabulated_prediction) FROM {dataset.relation_sql_for_source(source_id)}").fetchone()
        self.assertGreater(rows[0], 0)

    def test_glm_tabulation_without_feature_spec_keeps_bs_grid_inside_bounds(self) -> None:
        self.require_glm_dependencies()
        data_path = self.spline_data_path()
        dataset = Dataset(data_path)
        store = GlmModelStore(data_path)
        result = train_model(
            dataset,
            store,
            {"formula": "1 + bs(x, 3)", "response_column": "y", "family": "normal", "training_scope": "all"},
        )
        model_id = result["model_id"]

        build_tabulations(dataset, store, {"model_ids": [model_id]}, {"rows": []})
        tab_manifest = store.read_json(store.artifact_path(model_id, "tabulation_manifest"))

        self.assertIn("x", [table["table_id"] for table in tab_manifest["tables"]])
        meta = tab_manifest["feature_meta"]["x"]
        self.assertEqual(meta["min"], 1.0)
        self.assertEqual(meta["max"], 50.0)
        self.assertEqual(meta["banding"], 1.0)
        table_rows = store.read_parquet_records(store.tabulations_dir(model_id) / "x.parquet")
        levels = [row["x"] for row in table_rows if row["status"] == "ok"]
        self.assertGreaterEqual(min(levels), 1)
        self.assertLessEqual(max(levels), 50)
        with dataset.lock:
            row = dataset.con.execute(
                f"SELECT COUNT(glm_tabulated_prediction) FROM read_parquet({sql_literal(str(store.artifact_path(model_id, 'tabulated_predictions')))})"
            ).fetchone()
        self.assertEqual(row[0], 50)

    def test_glm_tabulation_estimates_blank_numeric_feature_spec_fields(self) -> None:
        self.require_glm_dependencies()
        data_path = self.spline_data_path()
        dataset = Dataset(data_path)
        store = GlmModelStore(data_path)
        result = train_model(
            dataset,
            store,
            {"formula": "1 + bs(x, 3)", "response_column": "y", "family": "normal", "training_scope": "all"},
        )
        model_id = result["model_id"]
        feature_spec = {"rows": [{"feature": "x", "grouping": "Test", "base": "", "min": "", "max": "", "banding": ""}]}

        build_tabulations(dataset, store, {"model_ids": [model_id]}, feature_spec)
        tab_manifest = store.read_json(store.artifact_path(model_id, "tabulation_manifest"))

        meta = tab_manifest["feature_meta"]["x"]
        self.assertEqual(meta["base"], 25.5)
        self.assertEqual(meta["min"], 1.0)
        self.assertEqual(meta["max"], 50.0)
        self.assertEqual(meta["banding"], 1.0)
        warnings_text = "\n".join(tab_manifest["warnings"])
        self.assertIn("Estimated min, max, banding", warnings_text)
        self.assertIn("Estimated base", warnings_text)

    def test_glm_tabulation_clips_out_of_bound_numeric_feature_spec_fields(self) -> None:
        self.require_glm_dependencies()
        data_path = self.spline_data_path()
        dataset = Dataset(data_path)
        store = GlmModelStore(data_path)
        result = train_model(
            dataset,
            store,
            {"formula": "1 + bs(x, 3)", "response_column": "y", "family": "normal", "training_scope": "all"},
        )
        model_id = result["model_id"]
        feature_spec = {"rows": [{"feature": "x", "grouping": "Test", "base": "0", "min": "0", "max": "100", "banding": "1"}]}

        build_tabulations(dataset, store, {"model_ids": [model_id]}, feature_spec)
        tab_manifest = store.read_json(store.artifact_path(model_id, "tabulation_manifest"))

        meta = tab_manifest["feature_meta"]["x"]
        self.assertEqual(meta["base"], 1.0)
        self.assertEqual(meta["min"], 1.0)
        self.assertEqual(meta["max"], 50.0)
        warnings_text = "\n".join(tab_manifest["warnings"])
        self.assertIn("Clipped feature spec min", warnings_text)
        self.assertIn("Clipped feature spec max", warnings_text)
        self.assertIn("Clipped feature spec base", warnings_text)
        table_rows = store.read_parquet_records(store.tabulations_dir(model_id) / "x.parquet")
        levels = [row["x"] for row in table_rows if row["status"] == "ok"]
        self.assertEqual(min(levels), 1)
        self.assertEqual(max(levels), 50)

    def test_glm_tabulation_defaults_blank_categorical_base_to_modal_level(self) -> None:
        self.require_glm_dependencies()
        data_path = self.spline_data_path()
        dataset = Dataset(data_path)
        store = GlmModelStore(data_path)
        result = train_model(
            dataset,
            store,
            {"formula": "C(Segment)", "response_column": "y", "family": "normal", "training_scope": "all"},
        )
        model_id = result["model_id"]
        feature_spec = {"rows": [{"feature": "Segment", "grouping": "Test", "base": ""}]}

        build_tabulations(dataset, store, {"model_ids": [model_id]}, feature_spec)
        tab_manifest = store.read_json(store.artifact_path(model_id, "tabulation_manifest"))

        self.assertEqual(tab_manifest["feature_meta"]["Segment"]["base"], "A")
        self.assertIn("Estimated base", "\n".join(tab_manifest["warnings"]))

    def test_glm_tabulation_scoring_is_vectorized_without_broad_loader(self) -> None:
        self.require_glm_dependencies()
        _glum, _glr, _glrcv, _np, pd = glm_dependencies()
        dataset = Dataset(self.data_path)
        store = GlmModelStore(self.data_path)
        result = train_model(
            dataset,
            store,
            {"formula": "Age + C(Segment)", "response_column": "actualNumerator", "family": "normal", "training_scope": "all"},
        )
        model_id = result["model_id"]
        feature_spec = {
            "rows": [
                {"feature": "Age", "grouping": "Driver", "base": "40", "min": "30", "max": "70", "banding": "5"},
                {"feature": "Segment", "grouping": "Driver", "base": "A"},
            ]
        }

        with (
            patch.object(pd.DataFrame, "iterrows", side_effect=AssertionError("tabulation scoring must not iterate rows")),
            patch("py_lucidum.tools.glm.tabulation.data_frame_from_dataset", side_effect=AssertionError("tabulation must not use the broad loader"), create=True),
        ):
            build_tabulations(dataset, store, {"model_ids": [model_id]}, feature_spec)

        with dataset.lock:
            rows = dataset.con.execute(
                f"SELECT COUNT(*), COUNT(glm_tabulated_prediction) FROM read_parquet({sql_literal(str(store.artifact_path(model_id, 'tabulated_predictions')))})"
            ).fetchone()
        self.assertEqual(rows, (6, 6))

    def test_glm_tabulation_loads_only_required_scoring_columns(self) -> None:
        self.require_glm_dependencies()
        path = self.root / "wide.csv"
        path.write_text(
            "y,Age,unused_text,unused_number\n"
            "10,30,a,1\n"
            "20,40,b,2\n"
            "30,50,c,3\n",
            encoding="utf-8",
        )
        dataset = Dataset(path)
        store = GlmModelStore(path)
        result = train_model(
            dataset,
            store,
            {"formula": "Age", "response_column": "y", "family": "normal", "training_scope": "all"},
        )
        captured_columns: list[list[str]] = []
        original_loader = glm_tabulation._tabulation_frame_from_dataset

        def capture_frame(load_dataset: Dataset, columns: list[str]) -> Any:
            captured_columns.append(list(columns))
            return original_loader(load_dataset, columns)

        with patch("py_lucidum.tools.glm.tabulation._tabulation_frame_from_dataset", side_effect=capture_frame):
            build_tabulations(dataset, store, {"model_ids": [result["model_id"]]}, {"rows": []})

        self.assertEqual(captured_columns, [["Age"]])

    def test_glm_tabulation_vectorized_categorical_unseen_rows_stay_missing(self) -> None:
        self.require_glm_dependencies()
        data_path = self.categorical_unseen_data_path()
        dataset = Dataset(data_path)
        store = GlmModelStore(data_path)
        result = train_model(
            dataset,
            store,
            {
                "formula": "C(Segment)",
                "response_column": "y",
                "family": "normal",
                "training_scope": "training",
                "regularization": {"mode": "manual", "alpha": 0.1, "l1_ratio": 0.0},
            },
        )
        model_id = result["model_id"]
        data_path.write_text(
            "y,Segment,SAMPLE\n"
            "10,A,training\n"
            "20,B,training\n"
            "30,C,test\n",
            encoding="utf-8",
        )
        dataset = Dataset(data_path)

        build_tabulations(dataset, store, {"model_ids": [model_id]}, {"rows": [{"feature": "Segment", "grouping": "Test", "base": "A"}]})
        tab_manifest = store.read_json(store.artifact_path(model_id, "tabulation_manifest"))

        self.assertEqual(tab_manifest["diagnostics"]["missing_tabulated_prediction_rows"], 1)
        self.assertIn("1 dataset level for Segment", "\n".join(tab_manifest["warnings"]))
        table_rows = store.read_parquet_records(store.tabulations_dir(model_id) / "Segment.parquet")
        self.assertEqual(next(row["status"] for row in table_rows if row["Segment"] == "C"), "unseen")
        with dataset.lock:
            rows = dataset.con.execute(
                f"SELECT COUNT(*), COUNT(glm_tabulated_prediction), SUM(CASE WHEN glm_tabulation_missing THEN 1 ELSE 0 END) FROM read_parquet({sql_literal(str(store.artifact_path(model_id, 'tabulated_predictions')))})"
            ).fetchone()
        self.assertEqual(rows, (3, 2, 1))

    def test_glm_tabulation_vectorized_interaction_lookup_writes_components(self) -> None:
        self.require_glm_dependencies()
        dataset = Dataset(self.data_path)
        store = GlmModelStore(self.data_path)
        result = train_model(
            dataset,
            store,
            {
                "formula": "Age * C(Segment)",
                "response_column": "actualNumerator",
                "family": "normal",
                "training_scope": "all",
                "regularization": {"mode": "manual", "alpha": 0.1, "l1_ratio": 0.0},
            },
        )
        model_id = result["model_id"]
        feature_spec = {
            "rows": [
                {"feature": "Age", "grouping": "Driver", "base": "40", "min": "30", "max": "70", "banding": "5"},
                {"feature": "Segment", "grouping": "Driver", "base": "A"},
            ]
        }

        build_tabulations(dataset, store, {"model_ids": [model_id]}, feature_spec)
        tab_manifest = store.read_json(store.artifact_path(model_id, "tabulation_manifest"))

        self.assertIn("Age|Segment", [table["table_id"] for table in tab_manifest["tables"]])
        with dataset.lock:
            rows = dataset.con.execute(
                f"SELECT COUNT(*), COUNT(glm_tabulated_prediction), COUNT(tabulated_linear__Age_Segment) FROM read_parquet({sql_literal(str(store.artifact_path(model_id, 'tabulated_predictions')))})"
            ).fetchone()
        self.assertEqual(rows, (6, 6, 6))

    def test_glm_tabulation_config_returns_all_model_statuses(self) -> None:
        self.require_glm_dependencies()
        dataset = Dataset(self.data_path)
        store = GlmModelStore(self.data_path)
        first = train_model(
            dataset,
            store,
            {"formula": "Age", "response_column": "actualNumerator", "family": "normal", "training_scope": "all"},
            activate=False,
        )
        second = train_model(
            dataset,
            store,
            {"formula": "Age + C(Segment)", "response_column": "actualNumerator", "family": "normal", "training_scope": "all"},
        )
        feature_spec = {"rows": [{"feature": "Age", "grouping": "Driver", "base": "40", "min": "30", "max": "70", "banding": "5"}]}
        build_tabulations(dataset, store, {"model_ids": [first["model_id"]]}, feature_spec)

        payload = tabulation_config(store, {"model_ids": [second["model_id"]]})

        self.assertEqual([model["model_id"] for model in payload["models"]], [second["model_id"]])
        all_statuses = {model["model_id"]: model for model in payload["all_models"]}
        self.assertEqual(set(all_statuses), {first["model_id"], second["model_id"]})
        self.assertTrue(all_statuses[first["model_id"]]["tabulated"])
        self.assertGreater(len(all_statuses[first["model_id"]]["tables"]), 0)
        self.assertFalse(all_statuses[second["model_id"]]["tabulated"])
        self.assertTrue(all_statuses[second["model_id"]]["tabulatable"])

    def test_glm_tabulation_plot_sorts_numeric_axes_numerically(self) -> None:
        store = GlmModelStore(self.data_path)
        model_id = "numeric-sort"
        store.model_dir(model_id).mkdir(parents=True)
        store.write_json(
            store.artifact_path(model_id, "tabulation_manifest"),
            {
                "tables": [
                    {
                        "table_id": "POSTCODE_CATEGORY",
                        "label": "POSTCODE_CATEGORY",
                        "index": 1,
                        "features": ["POSTCODE_CATEGORY"],
                        "cell_count": 4,
                        "skipped": False,
                        "path": "tabulations/POSTCODE_CATEGORY.parquet",
                    },
                    {
                        "table_id": "Age__Segment",
                        "label": "Age:Segment",
                        "index": 2,
                        "features": ["Age", "Segment"],
                        "cell_count": 6,
                        "skipped": False,
                        "path": "tabulations/Age__Segment.parquet",
                    },
                ],
                "warnings": [],
                "diagnostics": {},
            },
        )
        store.tabulations_dir(model_id).mkdir(parents=True)
        con = duckdb.connect(database=":memory:")
        try:
            con.execute(
                """
                CREATE TABLE postcode AS
                SELECT * FROM (VALUES
                    (10, 1.0, 'ok'),
                    (1, 0.1, 'ok'),
                    (3, 0.3, 'ok'),
                    (2, 0.2, 'ok')
                ) AS rows(POSTCODE_CATEGORY, tabulated_linear, status)
                """
            )
            con.execute(f"COPY postcode TO {sql_literal(str(store.tabulations_dir(model_id) / 'POSTCODE_CATEGORY.parquet'))} (FORMAT PARQUET)")
            con.execute(
                """
                CREATE TABLE age_segment AS
                SELECT * FROM (VALUES
                    (10, 'A', 1.0, 'ok'),
                    (1, 'A', 0.1, 'ok'),
                    (2, 'A', 0.2, 'ok'),
                    (10, 'B', 1.1, 'ok'),
                    (1, 'B', 0.15, 'ok'),
                    (2, 'B', 0.25, 'ok')
                ) AS rows(Age, Segment, tabulated_linear, status)
                """
            )
            con.execute(f"COPY age_segment TO {sql_literal(str(store.tabulations_dir(model_id) / 'Age__Segment.parquet'))} (FORMAT PARQUET)")
        finally:
            con.close()

        one_dimensional = tabulation_plot(store, {"model_ids": [model_id], "table_id": "POSTCODE_CATEGORY"})
        self.assertEqual(one_dimensional["x_axis"], [1, 2, 3, 10])
        self.assertEqual(one_dimensional["series"][0]["data"], [0.1, 0.2, 0.3, 1.0])

        two_dimensional = tabulation_plot(store, {"model_ids": [model_id], "table_id": "Age__Segment", "crosstab": "Segment"})
        self.assertEqual(two_dimensional["x_axis"], [1, 2, 10])
        no_crosstab = tabulation_plot(store, {"model_ids": [model_id], "table_id": "Age__Segment", "crosstab": ""})
        self.assertFalse(no_crosstab["plottable"])
        model_crosstab = tabulation_plot(store, {"model_ids": [model_id], "table_id": "Age__Segment", "crosstab": "__model__"})
        self.assertFalse(model_crosstab["plottable"])

    def test_glm_tabulation_table_crosstab_pivots_models_and_features(self) -> None:
        store = GlmModelStore(self.data_path)
        model_ids = ["tab-model-1", "tab-model-2"]
        for model_id in model_ids:
            store.model_dir(model_id).mkdir(parents=True)
            store.write_json(
                store.artifact_path(model_id, "tabulation_manifest"),
                {
                    "tables": [
                        {
                            "table_id": "Age__Segment",
                            "label": "Age:Segment",
                            "index": 1,
                            "features": ["Age", "Segment"],
                            "cell_count": 4,
                            "skipped": False,
                            "path": "tabulations/Age__Segment.parquet",
                        }
                    ],
                    "warnings": [],
                    "diagnostics": {},
                },
            )
            store.tabulations_dir(model_id).mkdir(parents=True)
        con = duckdb.connect(database=":memory:")
        try:
            for model_id, offset in [("tab-model-1", 0.0), ("tab-model-2", 1.0)]:
                con.execute(
                    f"""
                    CREATE OR REPLACE TABLE table_rows AS
                    SELECT * FROM (VALUES
                        (1, 'A', {0.1 + offset}, 'ok'),
                        (1, 'B', {0.2 + offset}, 'missing'),
                        (2, 'A', {0.3 + offset}, 'ok'),
                        (2, 'B', {0.4 + offset}, 'ok')
                    ) AS rows(Age, Segment, tabulated_linear, status)
                    """
                )
                con.execute(f"COPY table_rows TO {sql_literal(str(store.tabulations_dir(model_id) / 'Age__Segment.parquet'))} (FORMAT PARQUET)")
        finally:
            con.close()

        long_payload = tabulation_table(store, {"model_ids": model_ids, "table_id": "Age__Segment"})
        self.assertEqual([column["field"] for column in long_payload["columns"]], ["Age", "Segment", "tab-model-1", "tab-model-2"])
        self.assertTrue(all(column.get("tabulation_value") for column in long_payload["columns"][2:]))
        self.assertEqual(long_payload["min"], 0.1)
        self.assertEqual(long_payload["max"], 1.4)

        model_payload = tabulation_table(store, {"model_ids": model_ids, "table_id": "Age__Segment", "crosstab": "__model__"})
        self.assertEqual([column["field"] for column in model_payload["columns"]], ["Age", "Segment", "tab-model-1", "tab-model-2"])
        self.assertEqual(model_payload["crosstab"], "__model__")

        feature_payload = tabulation_table(store, {"model_ids": model_ids, "table_id": "Age__Segment", "crosstab": "Segment"})
        self.assertEqual([column["field"] for column in feature_payload["columns"][:2]], ["Age", "model"])
        self.assertEqual([column["title"] for column in feature_payload["columns"][2:]], ["A", "B"])
        self.assertTrue(all(column.get("tabulation_value") for column in feature_payload["columns"][2:]))
        self.assertTrue(all(column.get("status_field") for column in feature_payload["columns"][2:]))
        row = next(item for item in feature_payload["rows"] if item["Age"] == 1 and item["model"] == "tab-model-1")
        self.assertEqual(row["__pivot__0"], 0.1)
        self.assertEqual(row["__pivot__1"], 0.2)
        self.assertEqual(row["__status____pivot__1"], "missing")
        self.assertEqual(feature_payload["min"], 0.1)
        self.assertEqual(feature_payload["max"], 1.4)

    def test_glm_auto_regularization_stores_selected_penalty(self) -> None:
        self.require_glm_dependencies()
        dataset = Dataset(self.data_path)
        store = GlmModelStore(self.data_path)

        result = train_model(
            dataset,
            store,
            {
                "formula": "Age + C(Segment)",
                "response_column": "actualNumerator",
                "family": "normal",
                "training_scope": "all",
                "regularization": {"mode": "auto"},
            },
        )
        manifest = store.manifest(result["model_id"])
        regularization = manifest["regularization"]

        self.assertEqual(regularization["mode"], "auto")
        self.assertIsNotNone(regularization["selected_alpha"])
        self.assertIn(regularization["selected_l1_ratio"], [0.0, 0.5, 1.0])
        self.assertTrue(regularization["scale_predictors"])
        self.assertGreaterEqual(regularization["nonzero_coefficients"], 0)

    def test_glm_manual_lasso_omits_penalized_inference_columns(self) -> None:
        self.require_glm_dependencies()
        dataset = Dataset(self.data_path)
        store = GlmModelStore(self.data_path)

        result = train_model(
            dataset,
            store,
            {
                "formula": "Age + ifelse(Segment == 'Z', 1, 0)",
                "response_column": "actualNumerator",
                "family": "normal",
                "training_scope": "all",
                "regularization": {"mode": "manual", "alpha": 1.0, "l1_ratio": 1.0},
            },
        )
        manifest = store.manifest(result["model_id"])
        coefficients = result["coefficients"]

        self.assertEqual(manifest["regularization"]["mode"], "manual")
        self.assertEqual(manifest["regularization"]["selected_l1_ratio"], 1.0)
        self.assertTrue(manifest["regularization"]["scale_predictors"])
        self.assertTrue(any(row["estimate"] == 0 for row in coefficients if row["term"] != "(Intercept)"))
        self.assertTrue(all(row["std_error"] is None for row in coefficients))
        self.assertTrue(all(row["p_value"] is None for row in coefficients))

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

        status, body = asgi_post_json(app, "/api/glm/tabulations/config", {"model_ids": [model_id]})
        tab_config = json.loads(body)
        self.assertEqual(status, 200)
        self.assertTrue(tab_config["models"][0]["tabulatable"])
        self.assertFalse(tab_config["models"][0]["tabulated"])

        status, body = asgi_post_json(app, "/api/glm/tabulations/build", {"model_ids": [model_id]})
        tab_job = json.loads(body)
        self.assertEqual(status, 200)
        deadline = time.monotonic() + 60
        while time.monotonic() < deadline:
            status, body = asgi_get(app, f"/api/glm/tabulations/jobs/{tab_job['job_id']}")
            tab_payload = json.loads(body)
            if tab_payload["status"] not in {"queued", "running"}:
                break
            time.sleep(0.05)
        self.assertEqual(tab_payload["status"], "succeeded", tab_payload)

        status, body = asgi_post_json(app, "/api/glm/tabulations/config", {"model_ids": [model_id]})
        tab_config = json.loads(body)
        self.assertEqual(status, 200)
        table_ids = [table["table_id"] for table in tab_config["tables"]]
        self.assertIn("base", table_ids)
        self.assertIn("Age", table_ids)

        status, body = asgi_post_json(app, "/api/glm/tabulations/table", {"model_ids": [model_id], "table_id": "Age"})
        table_payload = json.loads(body)
        self.assertEqual(status, 200)
        self.assertTrue(table_payload["rows"])

        status, body = asgi_post_json(app, "/api/glm/tabulations/plot", {"model_ids": [model_id], "table_id": "Age"})
        plot_payload = json.loads(body)
        self.assertEqual(status, 200)
        self.assertTrue(plot_payload["series"])

        status, body = asgi_get(app, "/api/schema")
        schema = json.loads(body)
        source = next(item for item in schema["data_sources"] if item.get("model_id") == model_id)
        self.assertIn("glm_tabulated_prediction", [column["name"] for column in source["columns"]])

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
