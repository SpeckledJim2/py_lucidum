from __future__ import annotations

import json
import re
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from py_lucidum import line_bar_chart, report_filename, write_echarts_report


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


if __name__ == "__main__":
    unittest.main()
