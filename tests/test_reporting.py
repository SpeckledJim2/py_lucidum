from __future__ import annotations

import json
import pickle
import re
from types import SimpleNamespace
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import duckdb

from py_lucidum import (
    export_glm_tabulations,
    gbm_evaluation_chart,
    line_bar_chart,
    report_filename,
    write_echarts_report,
    write_gbm_summary_report,
    write_glm_summary_report,
)
from py_lucidum.core import Dataset, sql_literal
from py_lucidum.tools.gbm.store import GbmModelStore
from py_lucidum.tools.glm.store import GlmModelStore
from py_lucidum import reporting as reporting_module


class NormalDistribution:
    def deviance(self, y: Any, prediction: Any, sample_weight: Any = None) -> float:
        weights = list(sample_weight) if sample_weight is not None else [1.0] * len(y)
        return sum(float(weight) * (float(actual) - float(fitted)) ** 2 for actual, fitted, weight in zip(y, prediction, weights, strict=True))


class BinomialDistribution(NormalDistribution):
    pass


class IdentityLink:
    pass


class LogLink:
    pass


class ReportingTests(unittest.TestCase):
    def test_dataset_chart_can_be_written_as_self_contained_html(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            dataset_path = root / "pricing data.csv"
            output_path = root / "reports" / "pricing.html"
            dataset_path.write_text(
                "AGE,ACTUAL,BENCHMARK,SAMPLE\n"
                "20,100,105,training\n"
                "20,120,110,validation\n"
                "30,140,135,validation\n",
                encoding="utf-8",
            )

            chart = line_bar_chart(
                dataset_path,
                x="AGE",
                actual="ACTUAL",
                expected="BENCHMARK",
                sample_values=["validation"],
                controls={"banding": 5, "labels": "line", "low_weights": "0"},
                title="Driver age",
            )
            result = write_echarts_report(
                [chart],
                output_path,
                title="Validation report",
                metadata={
                    "source parquet": dataset_path,
                    "model": root / ".lucidum" / "models" / "example-model",
                    "response": "ACTUAL",
                    "weight": "None",
                    "expected": "BENCHMARK",
                    "SAMPLE_ROWS": "validation",
                    "script run": "example.py",
                    "importance measure": "Mean absolute SHAP",
                },
            )

            self.assertEqual(result, output_path.resolve())
            document = output_path.read_text(encoding="utf-8")
            self.assertIn("Validation report", document)
            self.assertIn("echarts.init(target)", document)
            self.assertIn("function lineBarChartOption", document)
            self.assertIn("height: 600px", document)
            self.assertIn('class="report-provenance"', document)
            self.assertIn('class="report-metadata-grid"', document)
            self.assertIn('class="report-metadata-footer"', document)
            report_header = re.search(r'<header class="report-header">(.*?)</header>', document, re.DOTALL).group(1)
            self.assertLess(report_header.index("Source Parquet"), report_header.index("Model"))
            self.assertLess(report_header.index("Response"), report_header.index("Weight"))
            self.assertLess(report_header.index("Weight"), report_header.index("Expected"))
            self.assertLess(report_header.index("Expected"), report_header.index("SAMPLE_ROWS"))
            self.assertLess(report_header.index("SAMPLE_ROWS"), report_header.index("Importance Measure"))
            self.assertIn("Mean absolute SHAP", report_header)
            self.assertIn(str(root / ".lucidum" / "models" / "example-model"), document)
            self.assertNotIn("<script src=", document)
            self.assertNotIn("lineBarControlStrip", document)

            match = re.search(
                r'<script id="lucidum-report-data" type="application/json">(.*?)</script>',
                document,
                flags=re.DOTALL,
            )
            self.assertIsNotNone(match)
            payload = json.loads(match.group(1))
            self.assertRegex(
                payload["metadata"]["time run"],
                r"^\d{1,2} [A-Z][a-z]{2} \d{4}, \d{2}:\d{2} .+$",
            )
            self.assertEqual(payload["charts"][0]["metadata"]["sample_values"], ["validation"])
            self.assertEqual(payload["charts"][0]["metadata"]["selected_rows"], 2)
            self.assertEqual(payload["charts"][0]["presentation"]["content"], "actual_expected")
            self.assertEqual(len(payload["charts"][0]["data"]["rows"]), 2)

    def test_report_filename_is_stable_and_readable(self) -> None:
        self.assertEqual(
            report_filename("/tmp/Motor Premiums.parquet", "GBM", "All rows - rebased SHAP"),
            "motor_premiums_external_gbm_all_rows_rebased_shap.html",
        )

    def test_report_chart_height_can_be_changed(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            output_path = Path(tmp_dir) / "height.html"
            chart = {"kind": "line_bar", "title": "Test", "data": {"rows": []}}

            write_echarts_report([chart], output_path, title="Height", chart_height=800)

            document = output_path.read_text(encoding="utf-8")
            self.assertIn("height: 800px", document)
            self.assertNotIn('class="report-metadata-footer"', document)

    def test_shap_only_requires_a_named_shap_model(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            dataset_path = Path(tmp_dir) / "data.csv"
            dataset_path.write_text("X,A,E,SAMPLE\n1,1,1,training\n", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "SHAP-only charts require"):
                line_bar_chart(
                    dataset_path,
                    x="X",
                    actual="A",
                    expected="E",
                    content="shap_only",
                )

    def test_named_gbm_evaluation_chart_does_not_follow_active_model(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            dataset_path = root / "data.csv"
            dataset_path.write_text("X,ACTUAL,SAMPLE\n1,10,training\n2,20,test\n", encoding="utf-8")
            dataset = Dataset(dataset_path)
            store = GbmModelStore(dataset_path, dataset=dataset)
            model_id = "named-model"
            model_dir = store.model_dir(model_id)
            model_dir.mkdir(parents=True)
            store.write_json(store.artifact_path(model_id, "manifest"), {
                "model_id": model_id,
                "label": "Named model",
                "best_iteration": 2,
            })
            store.write_json(store.artifact_path(model_id, "parameters"), {"metric": "l2"})
            store.write_json(store.artifact_path(model_id, "features"), ["X"])
            store.write_json(store.active_path, {"model_id": "different-model"})
            con = duckdb.connect(database=":memory:")
            try:
                con.execute(
                    f"""
COPY (
  SELECT 'training' AS dataset, 'l2' AS metric, 1 AS iteration, 12.0 AS value
  UNION ALL SELECT 'training', 'l2', 2, 8.0
  UNION ALL SELECT 'test', 'l2', 1, 14.0
  UNION ALL SELECT 'test', 'l2', 2, 9.0
) TO {sql_literal(str(store.artifact_path(model_id, 'evaluation')))} (FORMAT PARQUET)
"""
                )
            finally:
                con.close()
                dataset.con.close()

            chart = gbm_evaluation_chart(dataset_path, model_id=model_id)

            self.assertEqual(chart["kind"], "gbm_evaluation")
            self.assertEqual(chart["metadata"]["model_id"], model_id)
            self.assertEqual(chart["data"]["evaluation"]["training"]["l2"], [12.0, 8.0])
            self.assertEqual(chart["data"]["evaluation"]["test"]["l2"], [14.0, 9.0])

            store.artifact_path(model_id, "evaluation").unlink()
            with self.assertRaisesRegex(ValueError, "Rebuild the model"):
                gbm_evaluation_chart(dataset_path, model_id=model_id)

    def test_gbm_summary_is_written_as_self_contained_html(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            output_path = Path(tmp_dir) / "summary.html"
            evaluation_chart = {
                "kind": "gbm_evaluation",
                "title": "Model evaluation chart",
                "data": {
                    "manifest": {"best_iteration": 2},
                    "parameters": {"early_stopping_rounds": 2},
                    "metric": "l2",
                    "evaluation": {
                        "training": {"l2": [12.0, 8.0]},
                        "test": {"l2": [14.0, 9.0]},
                    },
                },
            }
            result = write_gbm_summary_report(
                output_path,
                title="GBM model summary",
                metadata={"source parquet": "/tmp/data.parquet", "model": "/tmp/model"},
                performance={
                    "columns": [
                        {"key": "sample", "label": "Sample"},
                        {"key": "actual", "label": "Actual response"},
                    ],
                    "rows": [
                        {"sample": "Training", "actual": "£100"},
                        {"sample": "Test", "actual": "£101"},
                        {"sample": "Validation", "actual": "£99"},
                    ],
                    "metric": "l2",
                    "best_iteration": 2,
                },
                feature_importance={
                    "measure": "Mean absolute SHAP",
                    "columns": [
                        {"key": "rank", "label": "Rank"},
                        {"key": "feature", "label": "Feature"},
                        {"key": "share", "label": "Share"},
                    ],
                    "rows": [{"rank": 1, "feature": "AGE", "share": "100.0%"}],
                },
                parameters={"learning_rate": 0.1, "num_leaves": 3},
                evaluation_chart=evaluation_chart,
            )

            self.assertEqual(result, output_path.resolve())
            document = output_path.read_text(encoding="utf-8")
            self.assertIn('data-summary-section="performance"', document)
            self.assertIn('data-summary-section="feature-importance"', document)
            self.assertIn('data-summary-section="parameters"', document)
            self.assertIn('data-summary-section="evaluation"', document)
            self.assertIn("Mean absolute SHAP", document)
            self.assertIn("£100", document)
            self.assertIn("function gbmEvaluationChartOption", document)
            self.assertNotIn("Zoom tail", document)
            self.assertNotIn("<script src=", document)

    def test_glm_summary_uses_fitted_predictions_and_shared_tabulation_index(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            dataset_path = root / "data.csv"
            dataset_path.write_text(
                "Y,W,SAMPLE\n"
                "10,1,training\n20,2,training\n"
                "30,3,test\n40,4,test\n"
                "50,5,validation\n60,6,validation\n",
                encoding="utf-8",
            )
            kpi_path = root / "kpi.csv"
            kpi_path.write_text(
                "group,name,actual,denominator,decimals,format\nFINANCIAL,Rate,Y,W,2,currency\n",
                encoding="utf-8",
            )
            dataset = Dataset(dataset_path)
            store = GlmModelStore(dataset_path, dataset=dataset)
            model_id = "summary-model"
            model_dir = store.model_dir(model_id)
            model_dir.mkdir(parents=True)
            store.write_json(
                store.artifact_path(model_id, "manifest"),
                {
                    "model_id": model_id,
                    "label": "Summary model",
                    "family": "normal",
                    "link": "auto",
                    "response_column": "Y",
                    "denominator_column": "W",
                },
            )
            with store.artifact_path(model_id, "estimator").open("wb") as handle:
                pickle.dump(
                    SimpleNamespace(
                        family_instance=NormalDistribution(),
                        link_instance=IdentityLink(),
                    ),
                    handle,
                )
            con = duckdb.connect(database=":memory:")
            try:
                con.execute(
                    f"""
COPY (SELECT * FROM (VALUES
  (1, 9.0), (2, 22.0), (3, 33.0), (4, 36.0), (5, 55.0), (6, 54.0)
) prediction(__lucidum_row_id, glm_prediction))
TO {sql_literal(str(store.artifact_path(model_id, 'predictions')))} (FORMAT PARQUET)
"""
                )
                con.execute(
                    f"""
COPY (SELECT * FROM (VALUES
  (1, 999.0, 0.0, false), (2, 999.0, 0.0, false), (3, 999.0, 0.0, false),
  (4, 999.0, 0.0, false), (5, 999.0, 0.0, false), (6, 999.0, 0.0, false)
) tabulated(__lucidum_row_id, glm_tabulated_prediction, glm_tabulated_linear_prediction, glm_tabulation_missing))
TO {sql_literal(str(store.artifact_path(model_id, 'tabulated_predictions')))} (FORMAT PARQUET)
"""
                )
                con.execute(
                    f"""
COPY (SELECT * FROM (VALUES
  ('Intercept', 1.234567, 0.1, 0.005), ('X', -0.25, NULL, NULL)
) coefficients(term, estimate, std_error, p_value))
TO {sql_literal(str(store.artifact_path(model_id, 'coefficients')))} (FORMAT PARQUET)
"""
                )
            finally:
                con.close()
                dataset.con.close()
            workbook_path = model_dir / "tabulations" / "summary-model_tabulations_linear.xlsx"
            workbook_path.parent.mkdir(parents=True)
            workbook_path.write_bytes(b"xlsx placeholder")
            tabulation_export = {
                "path": workbook_path,
                "scale": "linear",
                "index": {
                    "columns": [
                        {"key": "number", "label": "#"},
                        {"key": "table_name", "label": "Table name"},
                        {"key": "dim", "label": "Dim"},
                        {"key": "cells", "label": "Cells"},
                        {"key": "min", "label": "Min"},
                        {"key": "max", "label": "Max"},
                        {"key": "span", "label": "Span"},
                    ],
                    "rows": [{"number": 1, "table_name": "base", "dim": 0, "cells": 1, "min": 1.0, "max": 1.0, "span": 0.0}],
                },
            }
            output_path = root / "summary.html"

            write_glm_summary_report(
                output_path,
                title="GLM summary",
                dataset_path=dataset_path,
                model_id=model_id,
                kpi_spec_path=kpi_path,
                tabulation_export=tabulation_export,
            )

            document = output_path.read_text(encoding="utf-8")
            payload_match = re.search(
                r'<script id="lucidum-report-data" type="application/json">(.*?)</script>',
                document,
                re.DOTALL,
            )
            self.assertIsNotNone(payload_match)
            payload = json.loads(payload_match.group(1))
            self.assertEqual(payload["performance"]["prediction_source"], "glm_prediction")
            self.assertEqual(payload["performance"]["rows"][0]["prediction"], "£10.33")
            self.assertNotIn("999", json.dumps(payload["performance"]))
            self.assertEqual(payload["tabulations"]["rows"], tabulation_export["index"]["rows"])
            self.assertEqual(payload["coefficients"]["rows"][0]["estimate"], "1.2346")
            self.assertEqual(payload["coefficients"]["rows"][0]["p_value"], "0.5%")
            self.assertEqual(payload["coefficients"]["rows"][1]["std_error"], "--")
            self.assertIn('class="significance-low"', document)
            self.assertIn(workbook_path.resolve().as_uri(), document)
            self.assertIn("<td>1.0000</td>", document)
            self.assertIn("<td>0.0000</td>", document)
            for section in ("performance", "coefficients", "tabulations"):
                self.assertIn(f'data-summary-section="{section}"', document)

    def test_glm_binomial_performance_handles_weighted_ties_and_single_class_auc(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            dataset_path = root / "binary.csv"
            dataset_path.write_text(
                "Y,SAMPLE\n0,training\n1,training\n0,test\n1,test\n1,validation\n1,validation\n",
                encoding="utf-8",
            )
            prediction_path = root / "predictions.parquet"
            con = duckdb.connect(database=":memory:")
            try:
                con.execute(
                    f"""
COPY (SELECT * FROM (VALUES (1, .5), (2, .5), (3, .2), (4, .8), (5, .7), (6, .9))
  prediction(__lucidum_row_id, glm_prediction))
TO {sql_literal(str(prediction_path))} (FORMAT PARQUET)
"""
                )
            finally:
                con.close()
            dataset = Dataset(dataset_path)
            try:
                performance = reporting_module._glm_performance(
                    dataset,
                    prediction_path,
                    response="Y",
                    denominator="",
                    estimator=SimpleNamespace(family_instance=BinomialDistribution()),
                    kpi={"format": "percent", "decimals": 1},
                )
            finally:
                dataset.con.close()

            self.assertEqual(performance["rows"][0]["auc"], "50.0%")
            self.assertEqual(performance["rows"][0]["gini"], "0.0%")
            self.assertEqual(performance["rows"][1]["auc"], "100.0%")
            self.assertEqual(performance["rows"][2]["auc"], "—")
            self.assertTrue(all(row["log_loss"] != "—" for row in performance["rows"]))

    def test_glm_workbook_links_support_windows_paths(self) -> None:
        self.assertEqual(
            reporting_module._file_uri(r"C:\reports\model tabulations.xlsx"),
            "file:///C:/reports/model%20tabulations.xlsx",
        )

    def test_glm_tabulation_export_auto_scale_uses_the_fitted_link(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            dataset_path = root / "data.csv"
            dataset_path.write_text("X,SAMPLE\n1,training\n", encoding="utf-8")
            dataset = Dataset(dataset_path)
            store = GlmModelStore(dataset_path, dataset=dataset)
            con = duckdb.connect(database=":memory:")
            try:
                for model_id, link in (("log-model", LogLink()), ("identity-model", IdentityLink())):
                    model_dir = store.model_dir(model_id)
                    store.tabulations_dir(model_id).mkdir(parents=True)
                    store.write_json(
                        store.artifact_path(model_id, "manifest"),
                        {"model_id": model_id, "label": model_id},
                    )
                    with store.artifact_path(model_id, "estimator").open("wb") as handle:
                        pickle.dump(SimpleNamespace(link_instance=link), handle)
                    store.write_json(
                        store.artifact_path(model_id, "tabulation_manifest"),
                        {
                            "model_id": model_id,
                            "status": "tabulated",
                            "tables": [
                                {
                                    "index": 1,
                                    "table_id": "base",
                                    "label": "base",
                                    "features": [],
                                    "cell_count": 1,
                                    "min": 1.0,
                                    "max": 1.0,
                                    "skipped": False,
                                }
                            ],
                        },
                    )
                    con.execute(
                        f"""
COPY (SELECT 'base' AS table_id, 'ok' AS status, 1.0 AS tabulated_linear)
TO {sql_literal(str(store.tabulations_dir(model_id) / 'base.parquet'))} (FORMAT PARQUET)
"""
                    )
            finally:
                con.close()
                dataset.con.close()

            log_export = export_glm_tabulations(dataset_path, model_id="log-model", scale="auto")
            identity_export = export_glm_tabulations(dataset_path, model_id="identity-model", scale="auto")

            self.assertEqual(log_export["scale"], "exp")
            self.assertTrue(log_export["path"].name.endswith("_tabulations_exp.xlsx"))
            self.assertEqual(identity_export["scale"], "linear")
            self.assertTrue(identity_export["path"].name.endswith("_tabulations_linear.xlsx"))


if __name__ == "__main__":
    unittest.main()
