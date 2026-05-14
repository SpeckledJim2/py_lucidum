from __future__ import annotations

import os
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from py_lucidum.app import create_app
from py_lucidum.core import Dataset
from py_lucidum.query import Dataset as LegacyDataset
from py_lucidum.query import build_x_sql
from py_lucidum.tools.line_bar.query import chart


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
        self.filters_path.write_text("name,expression\nOlder drivers,YoungestDriverAge > 40\n", encoding="utf-8")

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

    def test_app_registers_line_bar_routes_and_saved_filters(self) -> None:
        app = create_app(
            self.data_path,
            token="dev-token",
            defaults={"denominator": "Weight"},
            filters_path=self.filters_path,
            tools=["line_bar"],
        )
        paths = {route.path for route in app.routes}

        self.assertIn("/api/chart", paths)
        self.assertIn("/api/line-bar/chart", paths)
        self.assertIn("/api/schema", paths)
        self.assertIn("/api/shutdown", paths)
        self.assertIn("/static", paths)
        self.assertEqual(app.state.enabled_tools, ["line_bar"])
        self.assertEqual(app.state.defaults["denominator"], "Weight")
        self.assertEqual(app.state.saved_filters, [{"name": "Older drivers", "expression": "YoungestDriverAge > 40"}])

    def test_default_saved_filters_fall_back_to_specs_directory(self) -> None:
        self.filters_path.unlink()
        specs_dir = self.root / "specs"
        specs_dir.mkdir()
        (specs_dir / "filter_spec.csv").write_text(
            "name,expression\nSpec older drivers,YoungestDriverAge > 40\n",
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
            [{"name": "Spec older drivers", "expression": "YoungestDriverAge > 40"}],
        )

    def test_app_loads_with_saved_filters_disabled(self) -> None:
        specs_dir = self.root / "specs"
        specs_dir.mkdir()
        (specs_dir / "filter_spec.csv").write_text(
            "name,expression\nSpec older drivers,YoungestDriverAge > 40\n",
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

    def test_chart_filters_and_aggregates_response_lines(self) -> None:
        dataset = Dataset(self.data_path)
        result = chart(dataset, self.request("YoungestDriverAge > 40"))

        self.assertEqual(result["row_count"], 4)
        self.assertEqual(result["filtered_row_count"], 3)
        self.assertEqual([row["x"] for row in result["rows"]], ["Business", "Social"])
        self.assertEqual(result["rows"][0]["volume"], 2)
        self.assertEqual(result["rows"][0]["resp0"], 350)
        self.assertEqual(result["rows"][0]["resp1"], 350)
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
        self.assertEqual(result["rows"][0]["resp1"], 10)
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
