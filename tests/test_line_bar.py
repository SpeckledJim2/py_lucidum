from __future__ import annotations

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
            "YoungestDriverAge,UseofVan,QuoteDate,Gross.Weight,Actual,Expected\n"
            "30,Social,2024-01-01,2500,100,90\n"
            "45,Social,2024-01-02,3500,200,210\n"
            "50,Business,2024-02-01,4000,300,290\n"
            "60,Business,2024-02-20,4500,400,410\n",
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
            "maxGroups": 10000,
            "responses": [
                {"label": "Actual", "numerator": "Actual", "denominator": None},
                {"label": "Expected", "numerator": "Expected", "denominator": None},
            ],
        }

    def test_app_registers_line_bar_routes_and_saved_filters(self) -> None:
        app = create_app(self.data_path, token="dev-token", filters_path=self.filters_path, tools=["line_bar"])
        paths = {route.path for route in app.routes}

        self.assertIn("/api/chart", paths)
        self.assertIn("/api/line-bar/chart", paths)
        self.assertIn("/api/schema", paths)
        self.assertEqual(app.state.enabled_tools, ["line_bar"])
        self.assertEqual(app.state.saved_filters, [{"name": "Older drivers", "expression": "YoungestDriverAge > 40"}])

    def test_chart_filters_and_aggregates_response_lines(self) -> None:
        dataset = Dataset(self.data_path)
        result = chart(dataset, self.request("YoungestDriverAge > 40"))

        self.assertEqual(result["row_count"], 4)
        self.assertEqual(result["filtered_row_count"], 3)
        self.assertEqual([row["x"] for row in result["rows"]], ["Business", "Social"])
        self.assertEqual(result["rows"][0]["volume"], 2)
        self.assertEqual(result["rows"][0]["resp0"], 350)
        self.assertEqual(result["rows"][0]["resp1"], 350)

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
