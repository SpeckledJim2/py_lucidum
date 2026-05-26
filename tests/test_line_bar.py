from __future__ import annotations

import asyncio
import json
import os
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any
from unittest.mock import patch

import duckdb

from py_lucidum.app import create_app
from py_lucidum.core import Dataset, load_kpis, load_saved_filters
from py_lucidum.query import Dataset as LegacyDataset
from py_lucidum.query import build_x_sql
from py_lucidum.tools.line_bar.query import chart, normalise_quantile_count


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
            kpis_path=self.kpis_path,
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
