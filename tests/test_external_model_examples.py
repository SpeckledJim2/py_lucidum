from __future__ import annotations

import ast
import asyncio
import importlib.util
import json
import shutil
import subprocess
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import duckdb

from py_lucidum.app import create_app
from py_lucidum.core import Dataset, sql_literal
from py_lucidum.core.features import load_features
from py_lucidum.tools.gbm.store import GbmModelStore
from py_lucidum.tools.gbm.tabulation import build_gbm_tabulations
from py_lucidum.tools.glm.store import GlmModelStore
from py_lucidum.tools.glm.tabulation import build_tabulations
from py_lucidum.tools.glm.overlay import stop_persistent_glm_overlay_worker


ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = ROOT / "examples"
GLM_SCRIPT = EXAMPLES / "external_glm_artifacts_demo.py"
GBM_SCRIPT = EXAMPLES / "external_gbm_artifacts_demo.py"
EXAMPLE_HELPERS = EXAMPLES / "external_model_helpers.py"
EXPORT_ADAPTER = EXAMPLES / "lucidum_export.py"
GLM_MODEL_ID = "EXTERNAL_BUILD-config-glm"
GBM_MODEL_ID = "EXTERNAL_BUILD-config-gbm"
REQUIRED_GLM_FILES = {
    "manifest.json",
    "formula.txt",
    "estimator.pkl",
    "coefficients.parquet",
    "feature_importance.parquet",
    "predictions.parquet",
    "diagnostics.json",
}
REQUIRED_GBM_FILES = {
    "manifest.json",
    "parameters.json",
    "features.json",
    "feature_config.parquet",
    "model.txt",
    "predictions.parquet",
    "evaluation.parquet",
    "tree_table.parquet",
    "shap_values.parquet",
    "shap_summary.parquet",
}
HAS_EXAMPLE_DEPENDENCIES = all(
    importlib.util.find_spec(name) is not None for name in ("glum", "lightgbm", "numpy", "pandas", "yaml")
)


def asgi_request(app: Any, method: str, path: str, payload: dict[str, Any] | None = None) -> tuple[int, dict[str, Any]]:
    messages: list[dict[str, Any]] = []
    body = json.dumps(payload).encode("utf-8") if payload is not None else b""

    async def receive() -> dict[str, Any]:
        return {"type": "http.request", "body": body, "more_body": False}

    async def send(message: dict[str, Any]) -> None:
        messages.append(message)

    scope = {
        "type": "http",
        "asgi": {"version": "3.0", "spec_version": "2.3"},
        "http_version": "1.1",
        "method": method,
        "scheme": "http",
        "path": path,
        "raw_path": path.encode("ascii"),
        "query_string": b"",
        "headers": [(b"content-type", b"application/json")] if payload is not None else [],
        "client": ("127.0.0.1", 12345),
        "server": ("testserver", 80),
    }
    asyncio.run(app(scope, receive, send))
    status = next(message for message in messages if message["type"] == "http.response.start")["status"]
    response_body = b"".join(
        message.get("body", b"") for message in messages if message["type"] == "http.response.body"
    )
    return status, json.loads(response_body or b"{}")


def write_reduced_dataset(source: Path, target: Path) -> int:
    con = duckdb.connect(database=":memory:")
    try:
        con.execute(
            f"""
COPY (
  SELECT * EXCLUDE (__source_position, __sample_position)
  FROM (
    SELECT
      *,
      ROW_NUMBER() OVER (PARTITION BY SAMPLE ORDER BY __source_position) AS __sample_position
    FROM (
      SELECT *, ROW_NUMBER() OVER () AS __source_position
      FROM read_parquet({sql_literal(str(source))})
    ) source_rows
  ) sampled_rows
  WHERE __sample_position <= 350
  ORDER BY __source_position
) TO {sql_literal(str(target))} (FORMAT PARQUET)
"""
        )
        return int(con.execute(f"SELECT COUNT(*) FROM read_parquet({sql_literal(str(target))})").fetchone()[0])
    finally:
        con.close()


def write_example_configs(root: Path, dataset_path: Path) -> tuple[Path, Path, Path]:
    import yaml

    formula_path = root / "formula.txt"
    feature_spec_path = root / "feature_spec.csv"
    shutil.copyfile(EXAMPLES / "external_glm_formula.txt", formula_path)
    shutil.copyfile(ROOT / "specs" / "feature_spec.csv", feature_spec_path)
    common_dataset = {
        "path": dataset_path.name,
        "response_numerator": "PREMIUM",
        "denominator": "LATITUDE",
        "sample_column": "SAMPLE",
    }
    glm_config = {
        "dataset": {**common_dataset, "training_value": "training"},
        "model": {
            "id": GLM_MODEL_ID,
            "label": "External integration GLM",
            "formula_path": formula_path.name,
            "family": "normal",
            "link": "auto",
            "fit_intercept": True,
            "regularization": {"alpha": 0.0, "l1_ratio": 0.0, "scale_predictors": False},
        },
        "output": {"portable_root": "portable", "install": True, "replace_existing": True},
    }
    gbm_config = {
        "dataset": {
            **common_dataset,
            "training_value": "training",
            "early_stopping_value": "test",
            "holdout_value": "validation",
        },
        "features": {"spec_path": feature_spec_path.name, "scenario_column": "scenario1"},
        "model": {"id": GBM_MODEL_ID, "label": "External integration GBM"},
        "training": {
            "num_boost_round": 16,
            "early_stopping_rounds": 4,
            "shap_rows": 120,
            "parameters": {
                "objective": "poisson",
                "metric": "poisson",
                "learning_rate": 0.1,
                "num_leaves": 3,
                "min_data_in_leaf": 12,
                "feature_fraction": 1.0,
                "bagging_fraction": 1.0,
                "bagging_freq": 0,
                "seed": 2026,
                "feature_fraction_seed": 2026,
                "bagging_seed": 2026,
                "verbosity": -1,
            },
        },
        "output": {"portable_root": "portable", "install": True, "replace_existing": True},
    }
    glm_path = root / "config_glm.yaml"
    gbm_path = root / "config_gbm.yaml"
    glm_path.write_text(yaml.safe_dump(glm_config, sort_keys=False), encoding="utf-8")
    gbm_path.write_text(yaml.safe_dump(gbm_config, sort_keys=False), encoding="utf-8")
    return glm_path, gbm_path, feature_spec_path


def run_builder(script: Path, config: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(script), str(config)],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
        timeout=120,
    )


class ExternalModelExampleTests(unittest.TestCase):
    def test_external_builders_do_not_import_py_lucidum(self) -> None:
        for script in (GLM_SCRIPT, GBM_SCRIPT, EXAMPLE_HELPERS, EXPORT_ADAPTER):
            tree = ast.parse(script.read_text(encoding="utf-8"), filename=str(script))
            imported = {
                alias.name
                for node in ast.walk(tree)
                if isinstance(node, ast.Import)
                for alias in node.names
            }
            imported.update(
                str(node.module or "") for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)
            )
            self.assertFalse(
                any(name == "py_lucidum" or name.startswith("py_lucidum.") for name in imported),
                f"{script.name} must remain independent of py_lucidum",
            )

    def test_training_scripts_keep_one_clear_lucidum_handoff(self) -> None:
        expected_calls = {
            GLM_SCRIPT: "save_glm_for_lucidum",
            GBM_SCRIPT: "save_gbm_for_lucidum",
        }
        for script, expected_call in expected_calls.items():
            source = script.read_text(encoding="utf-8")
            tree = ast.parse(source, filename=str(script))
            calls = [
                node.func.id
                for node in ast.walk(tree)
                if isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id.startswith("save_")
                and node.func.id.endswith("_for_lucidum")
            ]
            self.assertEqual(calls, [expected_call])
            self.assertNotIn("__lucidum_", source)
            for step in range(1, 6):
                self.assertIn(f"# %% {step}.", source)

    @unittest.skipUnless(HAS_EXAMPLE_DEPENDENCIES, "external-model example dependencies are not installed")
    def test_external_configs_fail_closed_for_unknown_keys_and_unsafe_ids(self) -> None:
        import yaml

        with TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            unknown_config = root / "unknown.yaml"
            unknown_config.write_text("unexpected: true\n", encoding="utf-8")
            unknown = subprocess.run(
                [sys.executable, str(GLM_SCRIPT), str(unknown_config)],
                cwd=ROOT,
                check=False,
                capture_output=True,
                text=True,
                timeout=30,
            )
            self.assertNotEqual(unknown.returncode, 0)
            self.assertIn("config has unknown keys: unexpected", unknown.stderr)

            _, gbm_config, _ = write_example_configs(root, root / "does-not-need-to-exist.parquet")
            payload = yaml.safe_load(gbm_config.read_text(encoding="utf-8"))
            payload["model"]["id"] = "../unsafe"
            gbm_config.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
            unsafe = subprocess.run(
                [sys.executable, str(GBM_SCRIPT), str(gbm_config)],
                cwd=ROOT,
                check=False,
                capture_output=True,
                text=True,
                timeout=30,
            )
            self.assertNotEqual(unsafe.returncode, 0)
            self.assertIn("model.id must contain only", unsafe.stderr)

    @unittest.skipUnless(HAS_EXAMPLE_DEPENDENCIES, "external-model example dependencies are not installed")
    def test_external_artifacts_install_and_work_through_lucidum(self) -> None:
        self.addCleanup(stop_persistent_glm_overlay_worker)
        with TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            dataset_path = root / "motor_fixture.parquet"
            row_count = write_reduced_dataset(ROOT / "datasets" / "motor_premiums.parquet", dataset_path)
            self.assertEqual(row_count, 1050)
            glm_config, gbm_config, feature_spec_path = write_example_configs(root, dataset_path)

            glm_run = run_builder(GLM_SCRIPT, glm_config)
            gbm_run = run_builder(GBM_SCRIPT, gbm_config)
            self.assertIn(GLM_MODEL_ID, glm_run.stdout)
            self.assertIn(GBM_MODEL_ID, gbm_run.stdout)

            dataset = Dataset(dataset_path)
            glm_store = GlmModelStore(dataset_path, dataset=dataset)
            gbm_store = GbmModelStore(dataset_path, dataset=dataset)
            glm_dir = glm_store.model_dir(GLM_MODEL_ID)
            gbm_dir = gbm_store.model_dir(GBM_MODEL_ID)

            # If the builders reproduced workspace-signature v1 incorrectly,
            # the current stores would point at different, empty directories.
            self.assertTrue(REQUIRED_GLM_FILES.issubset({path.name for path in glm_dir.iterdir()}))
            self.assertTrue(REQUIRED_GBM_FILES.issubset({path.name for path in gbm_dir.iterdir()}))
            self.assertEqual(glm_store.active_model_id(), GLM_MODEL_ID)
            self.assertEqual(gbm_store.active_model_id(), GBM_MODEL_ID)
            self.assertNotIn("feature_config.json", {path.name for path in gbm_dir.iterdir()})
            self.assertNotIn("training_log.json", {path.name for path in gbm_dir.iterdir()})

            portable_index = json.loads((root / "portable" / "lucidum_artifacts.json").read_text(encoding="utf-8"))
            self.assertEqual(portable_index["version"], 1)
            portable_models = {
                (row["model_type"], row["model_id"]): row for row in portable_index["models"]
            }
            self.assertEqual(portable_models[("glm", GLM_MODEL_ID)]["relative_path"], f"glm/{GLM_MODEL_ID}")
            self.assertEqual(portable_models[("gbm", GBM_MODEL_ID)]["relative_path"], f"gbm/{GBM_MODEL_ID}")
            self.assertNotIn("path", portable_models[("glm", GLM_MODEL_ID)]["dataset"])
            self.assertNotIn("path", portable_models[("gbm", GBM_MODEL_ID)]["dataset"])

            # Rebuilding the configured ID must leave neighbouring models alone.
            keep_dirs = [glm_store.root / "KEEP-ME", gbm_store.root / "KEEP-ME"]
            for keep_dir in keep_dirs:
                keep_dir.mkdir()
                (keep_dir / "sentinel.txt").write_text("keep", encoding="utf-8")
            run_builder(GLM_SCRIPT, glm_config)
            run_builder(GBM_SCRIPT, gbm_config)
            for keep_dir in keep_dirs:
                self.assertEqual((keep_dir / "sentinel.txt").read_text(encoding="utf-8"), "keep")

            con = duckdb.connect(database=":memory:")
            try:
                for artifact, expected_count in (
                    (glm_dir / "predictions.parquet", row_count),
                    (gbm_dir / "predictions.parquet", row_count),
                    (gbm_dir / "shap_values.parquet", 120),
                ):
                    count, unique_ids, min_id, max_id = con.execute(
                        f"""
SELECT COUNT(*), COUNT(DISTINCT __lucidum_row_id), MIN(__lucidum_row_id), MAX(__lucidum_row_id)
FROM read_parquet({sql_literal(str(artifact))})
"""
                    ).fetchone()
                    self.assertEqual(count, expected_count)
                    self.assertEqual(unique_ids, expected_count)
                    self.assertGreaterEqual(min_id, 1)
                    self.assertLessEqual(max_id, row_count)
                glm_columns = {
                    row[0]
                    for row in con.execute(
                        f"DESCRIBE SELECT * FROM read_parquet({sql_literal(str(glm_dir / 'predictions.parquet'))})"
                    ).fetchall()
                }
                gbm_columns = {
                    row[0]
                    for row in con.execute(
                        f"DESCRIBE SELECT * FROM read_parquet({sql_literal(str(gbm_dir / 'feature_config.parquet'))})"
                    ).fetchall()
                }
            finally:
                con.close()
            self.assertEqual(
                glm_columns,
                {"__lucidum_row_id", "glm_prediction", "glm_prediction_rate"},
            )
            self.assertTrue({"name", "kind", "include", "gain", "mean_abs_shap"}.issubset(gbm_columns))

            con = duckdb.connect(database=":memory:")
            try:
                for artifact, prediction, rate in (
                    (glm_dir / "predictions.parquet", "glm_prediction", "glm_prediction_rate"),
                    (gbm_dir / "predictions.parquet", "gbm_prediction", "gbm_prediction_rate"),
                ):
                    max_error = con.execute(
                        f"""
SELECT MAX(ABS(pred.{rate} - pred.{prediction} / source.LATITUDE))
FROM read_parquet({sql_literal(str(artifact))}) pred
JOIN (
  SELECT ROW_NUMBER() OVER () AS __source_row_id, *
  FROM read_parquet({sql_literal(str(dataset_path))})
) source
  ON pred.__lucidum_row_id = source.__source_row_id
"""
                    ).fetchone()[0]
                    self.assertAlmostEqual(float(max_error or 0.0), 0.0, places=12)
            finally:
                con.close()

            for manifest_path in (glm_dir / "manifest.json", gbm_dir / "manifest.json"):
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                self.assertNotIn("artifact_path", json.dumps(manifest))
                self.assertNotIn("workspace", manifest)

            app = create_app(
                dataset_path,
                defaults={"x": "DRIVER_AGE", "actual": "PREMIUM", "denominator": "LATITUDE"},
                token="",
                tools=["line_bar", "glm", "gbm"],
                features_path=feature_spec_path,
                use_saved_filters=False,
                use_kpis=False,
            )
            status, schema = asgi_request(app, "GET", "/api/schema")
            self.assertEqual(status, 200)
            schema_sources = {source["id"]: source for source in schema["data_sources"]}
            source_ids = set(schema_sources)
            glm_source = f"glm:{GLM_MODEL_ID}:predictions"
            gbm_source = f"gbm:{GBM_MODEL_ID}:predictions"
            self.assertTrue({glm_source, gbm_source}.issubset(source_ids))
            self.assertEqual(schema_sources[glm_source]["row_count"], row_count)
            self.assertEqual(schema_sources[gbm_source]["row_count"], row_count)
            self.assertEqual(schema_sources[f"gbm:{GBM_MODEL_ID}:shap_long"]["row_count"], 120)

            for family, model_id in (("glm", GLM_MODEL_ID), ("gbm", GBM_MODEL_ID)):
                status, listing = asgi_request(app, "GET", f"/api/{family}/models")
                self.assertEqual(status, 200)
                self.assertEqual(listing["active_model_id"], model_id)
                self.assertIn(model_id, {row["model_id"] for row in listing["models"]})
                status, detail = asgi_request(app, "GET", f"/api/{family}/models/{model_id}")
                self.assertEqual(status, 200)
                self.assertEqual(detail["manifest"]["model_id"], model_id)

            base_chart = {
                "x": "DRIVER_AGE",
                "bandWidth": "10",
                "dateBucket": "none",
                "lowGroup": "0",
                "sort": "alpha",
                "sigma": 0,
                "transform": "none",
                "filter": "",
                "denominator": "LATITUDE",
                "maxGroups": 10000,
            }
            chart_payloads: dict[str, dict[str, Any]] = {}
            for label, source, column in (
                ("External GLM", glm_source, "glm_prediction"),
                ("External GBM", gbm_source, "gbm_prediction"),
            ):
                request = {
                    **base_chart,
                    "responses": [
                        {"label": "Actual", "numerator": "PREMIUM", "source": "dataset"},
                        {"label": label, "numerator": column, "source": source},
                    ],
                    "partialDependence": {"mode": "none"},
                }
                status, chart_payload = asgi_request(app, "POST", "/api/chart", request)
                self.assertEqual(status, 200)
                self.assertTrue(chart_payload["rows"])
                chart_payloads[label] = {"request": request, "response": chart_payload}

            glm_chart = chart_payloads["External GLM"]
            status, overlay = asgi_request(
                app,
                "POST",
                "/api/line-bar/glm-overlay",
                {
                    "request": {**glm_chart["request"], "partialDependence": {"mode": "glm"}},
                    "chart_context": glm_chart["response"]["glm_overlay_context"],
                },
            )
            self.assertEqual(status, 200)
            self.assertEqual(overlay["partial_dependence"]["model_id"], GLM_MODEL_ID)
            self.assertTrue(overlay["partial_dependence"]["rows"])

            feature_spec = load_features(feature_spec_path)
            glm_tabulation = build_tabulations(
                dataset,
                glm_store,
                {"model_ids": [GLM_MODEL_ID]},
                feature_spec,
            )
            self.assertTrue(glm_tabulation["models"][0]["tables"])

            status, gbm_detail = asgi_request(app, "GET", f"/api/gbm/models/{GBM_MODEL_ID}")
            self.assertEqual(status, 200)
            self.assertTrue(gbm_detail["evaluation"]["training"])
            status, tree_summary = asgi_request(app, "GET", f"/api/gbm/models/{GBM_MODEL_ID}/trees")
            self.assertEqual(status, 200)
            self.assertTrue(tree_summary["trees"])
            status, tree = asgi_request(app, "GET", f"/api/gbm/models/{GBM_MODEL_ID}/trees/0")
            self.assertEqual(status, 200)
            self.assertEqual(tree["tree"], 0)
            self.assertTrue(tree["root"])
            status, shap_config = asgi_request(app, "GET", f"/api/gbm/models/{GBM_MODEL_ID}/shap/config")
            self.assertEqual(status, 200)
            self.assertTrue(shap_config["has_shap"])
            shap_feature = shap_config["default_feature_1"]
            status, shap = asgi_request(
                app,
                "POST",
                f"/api/gbm/models/{GBM_MODEL_ID}/shap/plot",
                {"feature_1": shap_feature, "tail_percent": 0, "rescale": "none"},
            )
            self.assertEqual(status, 200)
            self.assertTrue(shap["rows"])
            status, stacked = asgi_request(
                app,
                "POST",
                f"/api/gbm/models/{GBM_MODEL_ID}/shap/stacked",
                {"model_feature": shap_feature, "tail_percent": 0, "num_features": "all", "x_sort": "alpha"},
            )
            self.assertEqual(status, 200)
            self.assertTrue(stacked["rows"])

            gbm_tabulation = build_gbm_tabulations(dataset, gbm_store, GBM_MODEL_ID, feature_spec)
            self.assertEqual(gbm_tabulation["status"], "tabulated")
            self.assertTrue(gbm_tabulation["tables"])


if __name__ == "__main__":
    unittest.main()
