from __future__ import annotations

import asyncio
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any
from unittest.mock import patch

import duckdb

from py_lucidum.app import create_app, normalise_tools
from py_lucidum.core import Dataset
from py_lucidum.tools.column_profile import query as column_profile_query
from py_lucidum.tools.column_profile.query import profile, profile_detail


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


class ColumnProfileToolTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.data_path = self.root / "sample.csv"
        self.data_path.write_text(
            "Age,Segment,Score,QuoteDate\n"
            "1,A,10,2024-01-01\n"
            "2,A,,2024-01-02\n"
            ",B,30,\n"
            "4,C,40,2024-01-04\n",
            encoding="utf-8",
        )

    def column(self, payload: dict[str, Any], name: str) -> dict[str, Any]:
        return next(column for column in payload["columns"] if column["name"] == name)

    def test_default_tools_include_line_bar_first(self) -> None:
        self.assertEqual(normalise_tools(None), ["line_bar", "dataset_viewer", "column_profile", "histogram", "uk_map", "specs"])
        self.assertEqual(normalise_tools("all"), ["line_bar", "dataset_viewer", "column_profile", "histogram", "uk_map", "glm", "gbm", "specs"])
        self.assertEqual(normalise_tools("profile,line-bar"), ["line_bar", "column_profile"])
        self.assertEqual(normalise_tools("glm,line-bar"), ["line_bar", "glm"])
        self.assertEqual(normalise_tools("gbm,line-bar,map"), ["line_bar", "uk_map", "gbm"])
        self.assertEqual(normalise_tools("dataset-viewer,line-bar"), ["line_bar", "dataset_viewer"])
        with self.assertRaisesRegex(ValueError, "Tool 'glm' requires 'line-bar'"):
            normalise_tools("glm")
        with self.assertRaisesRegex(ValueError, "Tool 'gbm' requires 'line-bar'"):
            normalise_tools("gbm")
        with self.assertRaisesRegex(ValueError, "Unknown tool 'models'"):
            normalise_tools("models")

        app = create_app(self.data_path, token="dev-token")
        paths = {route.path for route in app.routes}

        self.assertEqual(app.state.enabled_tools, ["line_bar", "dataset_viewer", "column_profile", "histogram", "uk_map", "specs"])
        self.assertIn("/api/dataset-viewer/table", paths)
        self.assertIn("/api/column-profile/summary", paths)
        self.assertIn("/api/column-profile/detail", paths)
        self.assertIn("/api/chart", paths)
        self.assertIn("/api/histogram/chart", paths)
        self.assertIn("/api/uk-map/summary", paths)
        self.assertNotIn("/api/glm/summary", paths)
        self.assertNotIn("/api/gbm/summary", paths)

    def test_model_tools_can_be_enabled_without_model_dependencies(self) -> None:
        app = create_app(self.data_path, token="", tools=["line_bar", "glm", "gbm"], use_saved_filters=False, use_kpis=False)
        paths = {route.path for route in app.routes}

        self.assertEqual(app.state.enabled_tools, ["line_bar", "glm", "gbm"])
        self.assertNotIn("/api/dataset-viewer/table", paths)
        self.assertNotIn("/api/column-profile/summary", paths)
        self.assertIn("/api/chart", paths)
        self.assertIn("/api/glm/summary", paths)
        self.assertIn("/api/glm/config", paths)
        self.assertIn("/api/glm/models", paths)
        self.assertIn("/api/glm/validate", paths)
        self.assertIn("/api/glm/build", paths)
        self.assertIn("/api/gbm/summary", paths)

        status, _, body = asgi_get(app, "/api/glm/summary")
        payload = json.loads(body)

        self.assertEqual(status, 200)
        self.assertEqual(payload["tool"], "glm")
        self.assertEqual(payload["status"], "ready")
        self.assertEqual(payload["response"], "response_column")
        self.assertEqual(payload["denominator"], "denominator_column")

    def test_gbm_tool_selection_requires_line_bar_but_not_glm(self) -> None:
        app = create_app(self.data_path, token="", tools=["gbm", "line_bar"], use_saved_filters=False, use_kpis=False)
        paths = {route.path for route in app.routes}

        self.assertEqual(app.state.enabled_tools, ["line_bar", "gbm"])
        self.assertIn("/api/chart", paths)
        self.assertNotIn("/api/glm/summary", paths)
        self.assertIn("/api/gbm/summary", paths)

    def test_column_profile_can_be_enabled_without_dataset_viewer(self) -> None:
        app = create_app(self.data_path, token="", tools=["column-profile"], use_saved_filters=False, use_kpis=False)
        paths = {route.path for route in app.routes}

        self.assertEqual(app.state.enabled_tools, ["column_profile"])
        self.assertNotIn("/api/dataset-viewer/table", paths)
        self.assertIn("/api/column-profile/summary", paths)
        self.assertIn("/api/column-profile/detail", paths)
        self.assertNotIn("/api/chart", paths)
        self.assertNotIn("/api/uk-map/summary", paths)

    def test_profile_endpoint_includes_duckdb_timing(self) -> None:
        app = create_app(self.data_path, token="", tools=["column-profile"], use_saved_filters=False, use_kpis=False)

        status, _, body = asgi_post_json(app, "/api/column-profile/summary", {"filter": ""})
        payload = json.loads(body)

        self.assertEqual(status, 200)
        self.assertEqual(payload["row_count"], 4)
        self.assertEqual(payload["filtered_row_count"], 4)
        self.assertEqual([column["name"] for column in payload["columns"]], ["Age", "Segment", "Score", "QuoteDate"])
        self.assertIsInstance(payload["timings"]["duckdb_ns"], int)
        self.assertGreaterEqual(payload["timings"]["duckdb_ns"], 0)
        self.assertIsInstance(payload["timings"]["duckdb_ms"], int)
        self.assertGreaterEqual(payload["timings"]["duckdb_ms"], 0)

    def test_profile_endpoint_skips_unreadable_columns(self) -> None:
        app = create_app(self.data_path, token="", tools=["column-profile"], use_saved_filters=False, use_kpis=False)
        original_probe = Dataset.probe_column_readable
        original_summary_column_stats = column_profile_query.summary_column_stats

        def fake_probe(dataset: Dataset, column: Any) -> None:
            if column.name == "Segment":
                raise duckdb.InvalidInputException(
                    'Invalid Input Error: Invalid string encoding found in Parquet file "/tmp/bad.parquet": '
                    'value "bad" is not valid UTF8!'
                )
            original_probe(dataset, column)

        def fake_summary_column_stats(dataset: Dataset, columns: list[Any], filter_sql: str, calculation: dict[str, Any]) -> dict[str, dict[str, Any]]:
            self.assertNotIn("Segment", [column.name for column in columns])
            return original_summary_column_stats(dataset, columns, filter_sql, calculation)

        with (
            patch.object(Dataset, "probe_column_readable", fake_probe),
            patch("py_lucidum.tools.column_profile.query.summary_column_stats", side_effect=fake_summary_column_stats),
        ):
            status, _, body = asgi_post_json(app, "/api/column-profile/summary", {"filter": ""})
        payload = json.loads(body)

        self.assertEqual(status, 200)
        self.assertEqual([column["name"] for column in payload["columns"]], ["Age", "Score", "QuoteDate"])
        self.assertEqual(payload["skipped_columns"], [
            {"name": "Segment", "error": "Invalid string encoding found in Parquet data."},
        ])
        self.assertEqual(payload["warnings"], ["Skipped 1 unreadable column: Segment."])
        self.assertNotIn("/tmp/bad.parquet", json.dumps(payload))

    def test_profile_endpoint_reports_shared_invalid_columns(self) -> None:
        original_probe = Dataset.probe_column_readable

        def fake_probe(dataset: Dataset, column: Any) -> None:
            if column.name == "Segment":
                raise duckdb.InvalidInputException(
                    'Invalid Input Error: Invalid string encoding found in Parquet file "/tmp/bad.parquet": '
                    'value "bad" is not valid UTF8!'
                )
            original_probe(dataset, column)

        with patch.object(Dataset, "probe_column_readable", fake_probe):
            app = create_app(self.data_path, token="", tools=["column-profile"], use_saved_filters=False, use_kpis=False)
            status, _, body = asgi_post_json(app, "/api/column-profile/summary", {"filter": ""})
        payload = json.loads(body)

        self.assertEqual(status, 200)
        self.assertEqual([column["name"] for column in payload["columns"]], ["Age", "Score", "QuoteDate"])
        self.assertEqual(payload["skipped_columns"], [
            {"name": "Segment", "error": "Invalid string encoding found in Parquet data."},
        ])

    def test_profile_detail_endpoint_includes_duckdb_timing(self) -> None:
        app = create_app(self.data_path, token="", tools=["column-profile"], use_saved_filters=False, use_kpis=False)

        status, _, body = asgi_post_json(app, "/api/column-profile/detail", {"column": "Age", "filter": ""})
        payload = json.loads(body)

        self.assertEqual(status, 200)
        self.assertEqual(payload["name"], "Age")
        self.assertIsInstance(payload["timings"]["duckdb_ns"], int)
        self.assertGreaterEqual(payload["timings"]["duckdb_ns"], 0)
        self.assertIsInstance(payload["timings"]["duckdb_ms"], int)
        self.assertGreaterEqual(payload["timings"]["duckdb_ms"], 0)

    def test_profile_detail_endpoint_reports_unreadable_column(self) -> None:
        app = create_app(self.data_path, token="", tools=["column-profile"], use_saved_filters=False, use_kpis=False)
        original_column_stats = column_profile_query.column_stats

        def fake_column_stats(dataset: Dataset, column: Any, filter_sql: str) -> dict[str, Any]:
            if column.name == "Segment":
                raise duckdb.InvalidInputException(
                    'Invalid Input Error: Invalid string encoding found in Parquet file "/tmp/bad.parquet": '
                    'value "bad" is not valid UTF8!'
                )
            return original_column_stats(dataset, column, filter_sql)

        with patch("py_lucidum.tools.column_profile.query.column_stats", side_effect=fake_column_stats):
            status, _, body = asgi_post_json(app, "/api/column-profile/detail", {"column": "Segment", "filter": ""})

        self.assertEqual(status, 400)
        self.assertIn(b"Could not profile Segment", body)
        self.assertIn(b"Invalid string encoding found in Parquet data", body)
        self.assertNotIn(b"/tmp/bad.parquet", body)

    def test_profile_summarises_filtered_columns(self) -> None:
        payload = profile(Dataset(self.data_path), {"filter": "Segment = 'A'"})

        age = self.column(payload, "Age")
        segment = self.column(payload, "Segment")
        score = self.column(payload, "Score")
        quote_date = self.column(payload, "QuoteDate")

        self.assertEqual(payload["row_count"], 4)
        self.assertEqual(payload["filtered_row_count"], 2)
        self.assertEqual(payload["calculation"], {
            "mode": "full",
            "requested_mode": "auto",
            "profiled_row_count": 2,
            "full_row_count": 2,
            "exact": True,
            "full_available": False,
        })
        self.assertEqual(age["missing_count"], 0)
        self.assertEqual(age["distinct_count"], 2)
        self.assertEqual(age["min"], 1)
        self.assertEqual(age["max"], 2)
        self.assertNotIn("samples", age)
        self.assertNotIn("distribution", age)
        self.assertEqual(segment["distinct_count"], 1)
        self.assertNotIn("top_values", segment)
        self.assertEqual(score["missing_count"], 1)
        self.assertEqual(score["missing_rate"], 0.5)
        self.assertEqual(score["distinct_count"], 1)
        self.assertEqual(quote_date["min"], "2024-01-01")
        self.assertEqual(quote_date["max"], "2024-01-02")

    def test_profile_preview_uses_bounded_rows_when_auto_threshold_exceeded(self) -> None:
        data_path = self.root / "preview.csv"
        data_path.write_text(
            "Age,Segment\n"
            "1,A\n"
            "2,A\n"
            "3,B\n"
            "4,B\n"
            "5,C\n",
            encoding="utf-8",
        )

        with (
            patch("py_lucidum.tools.column_profile.query.PROFILE_FULL_CELL_LIMIT", 1),
            patch("py_lucidum.tools.column_profile.query.PROFILE_PREVIEW_ROW_LIMIT", 2),
        ):
            payload = profile(Dataset(data_path), {"filter": ""})

        age = self.column(payload, "Age")
        segment = self.column(payload, "Segment")

        self.assertEqual(payload["row_count"], 5)
        self.assertEqual(payload["filtered_row_count"], 5)
        self.assertEqual(payload["calculation"], {
            "mode": "preview",
            "requested_mode": "auto",
            "profiled_row_count": 2,
            "full_row_count": 5,
            "exact": False,
            "full_available": True,
        })
        self.assertEqual(age["distinct_count"], 2)
        self.assertEqual(age["min"], 1)
        self.assertEqual(age["max"], 2)
        self.assertEqual(segment["distinct_count"], 1)

    def test_profile_full_mode_forces_exact_stats_above_preview_threshold(self) -> None:
        data_path = self.root / "preview_full.csv"
        data_path.write_text(
            "Age,Segment\n"
            "1,A\n"
            "2,A\n"
            "3,B\n"
            "4,B\n"
            "5,C\n",
            encoding="utf-8",
        )

        with (
            patch("py_lucidum.tools.column_profile.query.PROFILE_FULL_CELL_LIMIT", 1),
            patch("py_lucidum.tools.column_profile.query.PROFILE_PREVIEW_ROW_LIMIT", 2),
        ):
            payload = profile(Dataset(data_path), {"filter": "", "mode": "full"})

        age = self.column(payload, "Age")
        segment = self.column(payload, "Segment")

        self.assertEqual(payload["calculation"], {
            "mode": "full",
            "requested_mode": "full",
            "profiled_row_count": 5,
            "full_row_count": 5,
            "exact": True,
            "full_available": False,
        })
        self.assertEqual(age["distinct_count"], 5)
        self.assertEqual(age["min"], 1)
        self.assertEqual(age["max"], 5)
        self.assertEqual(segment["distinct_count"], 3)

    def test_profile_endpoint_rejects_invalid_summary_mode(self) -> None:
        app = create_app(self.data_path, token="", tools=["column-profile"], use_saved_filters=False, use_kpis=False)

        status, _, body = asgi_post_json(app, "/api/column-profile/summary", {"filter": "", "mode": "random"})

        self.assertEqual(status, 400)
        self.assertIn(b"valid profile calculation mode", body)

    def test_profile_detail_returns_numeric_histogram_and_stats_under_filter(self) -> None:
        payload = profile_detail(Dataset(self.data_path), {"column": "Age", "filter": "Segment = 'A'"})

        self.assertEqual(payload["row_count"], 4)
        self.assertEqual(payload["filtered_row_count"], 2)
        self.assertEqual(payload["missing_count"], 0)
        self.assertEqual(payload["non_missing_count"], 2)
        self.assertEqual(payload["distinct_count"], 2)
        self.assertEqual(len(payload["histogram"]), 2)
        self.assertEqual(sum(bin["count"] for bin in payload["histogram"]), 2)
        self.assertEqual([(bin["lower"], bin["upper"]) for bin in payload["histogram"]], [(1, 1), (2, 2)])
        self.assertEqual(payload["stats"]["min"], 1)
        self.assertEqual(payload["stats"]["median"], 1.5)
        self.assertEqual(payload["stats"]["mean"], 1.5)
        self.assertEqual(payload["stats"]["max"], 2)
        self.assertEqual(payload["zero_count"], 0)

    def test_profile_detail_uses_twenty_bins_for_high_cardinality_numeric_columns(self) -> None:
        data_path = self.root / "high_cardinality.csv"
        rows = ["Value"]
        rows.extend(str(value) for value in range(101))
        data_path.write_text("\n".join(rows) + "\n", encoding="utf-8")

        payload = profile_detail(Dataset(data_path), {"column": "Value", "filter": ""})

        self.assertEqual(payload["distinct_count"], 101)
        self.assertEqual(len(payload["histogram"]), 20)
        self.assertEqual(sum(bin["count"] for bin in payload["histogram"]), 101)

    def test_profile_detail_returns_categorical_value_counts(self) -> None:
        payload = profile_detail(Dataset(self.data_path), {"column": "Segment", "filter": ""})

        self.assertEqual(payload["kind"], "categorical")
        self.assertEqual(payload["value_counts"], [
            {"value": "A", "count": 2},
            {"value": "B", "count": 1},
            {"value": "C", "count": 1},
        ])
        self.assertEqual(payload["blank_count"], 0)

    def test_profile_detail_returns_date_histogram_and_percentiles(self) -> None:
        payload = profile_detail(Dataset(self.data_path), {"column": "QuoteDate", "filter": ""})

        self.assertEqual(payload["kind"], "date")
        self.assertEqual(payload["missing_count"], 1)
        self.assertEqual(len(payload["histogram"]), 20)
        self.assertEqual(sum(bin["count"] for bin in payload["histogram"]), 3)
        self.assertEqual(payload["stats"]["min"], "2024-01-01")
        self.assertEqual(payload["stats"]["median"], "2024-01-02T00:00:00")
        self.assertEqual(payload["stats"]["max"], "2024-01-04")

    def test_profile_detail_returns_categorical_blank_count(self) -> None:
        data_path = self.root / "blanks.parquet"
        con = duckdb.connect(database=":memory:")
        con.execute("CREATE TABLE blanks AS SELECT * FROM (VALUES ('A'), (''), (NULL), ('B'), ('')) AS rows(Segment)")
        con.execute(f"COPY blanks TO '{data_path}' (FORMAT PARQUET)")

        payload = profile_detail(Dataset(data_path), {"column": "Segment", "filter": ""})

        self.assertEqual(payload["kind"], "categorical")
        self.assertEqual(payload["missing_count"], 1)
        self.assertEqual(payload["blank_count"], 2)

    def test_profile_detail_returns_numeric_zero_count(self) -> None:
        data_path = self.root / "zeros.csv"
        data_path.write_text(
            "Value,Subset\n"
            "0,keep\n"
            "1,keep\n"
            "0,drop\n"
            ",keep\n",
            encoding="utf-8",
        )

        payload = profile_detail(Dataset(data_path), {"column": "Value", "filter": "Subset = 'keep'"})

        self.assertEqual(payload["kind"], "integer")
        self.assertEqual(payload["filtered_row_count"], 3)
        self.assertEqual(payload["missing_count"], 1)
        self.assertEqual(payload["zero_count"], 1)

    def test_profile_endpoint_rejects_invalid_filter(self) -> None:
        app = create_app(self.data_path, token="", tools=["column-profile"], use_saved_filters=False, use_kpis=False)

        status, _, body = asgi_post_json(app, "/api/column-profile/summary", {"filter": "Segment = "})

        self.assertEqual(status, 400)
        self.assertIn(b"Invalid filter", body)

    def test_profile_detail_endpoint_rejects_invalid_column_and_filter(self) -> None:
        app = create_app(self.data_path, token="", tools=["column-profile"], use_saved_filters=False, use_kpis=False)

        status, _, body = asgi_post_json(app, "/api/column-profile/detail", {"column": "Missing", "filter": ""})
        self.assertEqual(status, 400)
        self.assertIn(b"valid profile column", body)

        status, _, body = asgi_post_json(app, "/api/column-profile/detail", {"column": "Age", "filter": "Segment = "})
        self.assertEqual(status, 400)
        self.assertIn(b"Invalid filter", body)


if __name__ == "__main__":
    unittest.main()
