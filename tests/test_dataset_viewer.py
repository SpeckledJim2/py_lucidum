from __future__ import annotations

import asyncio
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any
from unittest.mock import patch

from py_lucidum.app import create_app, normalise_tools
from py_lucidum.core import Dataset


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


class DatasetViewerToolTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.data_path = self.root / "sample.csv"
        self.data_path.write_text(
            "Age,Segment,Score,QuoteDate\n"
            "1,A,10.5,2024-01-01\n"
            "2,B,,2024-01-02\n"
            "3,C,30.25,\n"
            "4,D,40,2024-01-04\n",
            encoding="utf-8",
        )

    def post_table(self, app: Any, payload: dict[str, Any]) -> dict[str, Any]:
        status, _, body = asgi_post_json(app, "/api/dataset-viewer/table", payload)
        self.assertEqual(status, 200, body.decode("utf-8"))
        return json.loads(body)

    def post_filter_row_count(self, app: Any, payload: dict[str, Any]) -> dict[str, Any]:
        status, _, body = asgi_post_json(app, "/api/filter/row-count", payload)
        self.assertEqual(status, 200, body.decode("utf-8"))
        return json.loads(body)

    def post_metric_summary(self, app: Any, payload: dict[str, Any]) -> dict[str, Any]:
        status, _, body = asgi_post_json(app, "/api/metrics/summary", payload)
        self.assertEqual(status, 200, body.decode("utf-8"))
        return json.loads(body)

    def test_line_bar_is_first_default_tool(self) -> None:
        self.assertEqual(normalise_tools(None), ["line_bar", "dataset_viewer", "column_profile", "histogram", "uk_map", "specs"])
        self.assertEqual(normalise_tools("line-bar"), ["line_bar"])
        self.assertEqual(normalise_tools("gbm,line-bar"), ["gbm", "line_bar"])
        self.assertEqual(normalise_tools("dataset-viewer,line-bar"), ["dataset_viewer", "line_bar"])
        app = create_app(self.data_path, token="", use_saved_filters=False, use_kpis=False)
        paths = {route.path for route in app.routes}

        self.assertEqual(app.state.enabled_tools, ["line_bar", "dataset_viewer", "column_profile", "histogram", "uk_map", "specs"])
        self.assertIn("/api/dataset-viewer/table", paths)
        self.assertIn("/api/filter/row-count", paths)
        self.assertIn("/api/metrics/summary", paths)

    def test_dataset_viewer_route_is_not_registered_when_tool_is_disabled(self) -> None:
        app = create_app(self.data_path, token="", tools=["line-bar"], use_saved_filters=False, use_kpis=False)
        paths = {route.path for route in app.routes}

        self.assertEqual(app.state.enabled_tools, ["line_bar"])
        self.assertNotIn("/api/dataset-viewer/table", paths)
        self.assertIn("/api/filter/row-count", paths)
        self.assertIn("/api/metrics/summary", paths)
        status, _, _ = asgi_post_json(app, "/api/dataset-viewer/table", {"filter": "", "limit": 1000})
        self.assertEqual(status, 404)

    def test_dataset_viewer_only_app_registers_favourites_api(self) -> None:
        app = create_app(self.data_path, token="", tools=["dataset-viewer"], use_saved_filters=False, use_kpis=False)
        paths = {route.path for route in app.routes}

        self.assertEqual(app.state.enabled_tools, ["dataset_viewer"])
        self.assertIn("/api/dataset-viewer/table", paths)
        self.assertIn("/api/line-bar/favourites", paths)
        self.assertIn("/api/line-bar/favourites/{favourite_id}", paths)
        self.assertIn("/api/line-bar/favourites/order", paths)

    def test_metric_summary_route_returns_filtered_values(self) -> None:
        app = create_app(self.data_path, token="", use_saved_filters=False, use_kpis=False)

        data = self.post_metric_summary(app, {"actual": "Score", "denominator": "__none__", "filter": "Age >= 3"})

        self.assertEqual(data["source"], "dataset")
        self.assertEqual(data["row_count"], 4)
        self.assertEqual(data["filtered_row_count"], 2)
        self.assertAlmostEqual(data["response_summaries"][0]["value"], 35.125)
        self.assertAlmostEqual(data["response_summaries"][0]["numerator"], 70.25)
        self.assertEqual(data["response_summaries"][0]["denominator"], 2)
        self.assertEqual(data["denominator"]["column"], None)
        self.assertEqual(data["denominator"]["value"], 2)

    def test_metric_summary_route_uses_selected_weight(self) -> None:
        app = create_app(self.data_path, token="", use_saved_filters=False, use_kpis=False)

        data = self.post_metric_summary(app, {"actual": "Score", "denominator": "Age", "filter": "Age >= 3"})

        self.assertEqual(data["denominator"]["column"], "Age")
        self.assertEqual(data["denominator"]["value"], 7)
        self.assertAlmostEqual(data["response_summaries"][0]["value"], 70.25 / 7)
        self.assertEqual(data["warnings"], [])

    def test_metric_summary_route_validates_inputs(self) -> None:
        app = create_app(self.data_path, token="", use_saved_filters=False, use_kpis=False)

        status, _, body = asgi_post_json(app, "/api/metrics/summary", {"actual": "Segment", "denominator": "__none__"})
        self.assertEqual(status, 400)
        self.assertEqual(json.loads(body)["detail"], "Choose a valid numeric Actual column")

        status, _, body = asgi_post_json(app, "/api/metrics/summary", {"actual": "Score", "denominator": "Segment"})
        self.assertEqual(status, 400)
        self.assertEqual(json.loads(body)["detail"], "Choose a valid numeric Weight column")

        status, _, _ = asgi_post_json(app, "/api/metrics/summary", {"actual": "Score", "denominator": "__none__", "filter": "Missing > 1"})
        self.assertEqual(status, 400)

    def test_metric_summary_route_is_registered_without_chart_tools(self) -> None:
        app = create_app(self.data_path, token="", tools=["specs"], use_saved_filters=False, use_kpis=False, use_features=False)
        paths = {route.path for route in app.routes}

        self.assertIn("/api/metrics/summary", paths)
        self.assertNotIn("/api/chart", paths)
        data = self.post_metric_summary(app, {"actual": "Score", "denominator": "__none__", "filter": ""})
        self.assertAlmostEqual(data["response_summaries"][0]["value"], (10.5 + 30.25 + 40) / 3)

    def test_table_respects_filter_and_preserves_preview_order(self) -> None:
        app = create_app(self.data_path, token="", use_saved_filters=False, use_kpis=False)
        payload = self.post_table(app, {"filter": "Age >= 2", "limit": 1000})

        self.assertNotIn("row_count", payload)
        self.assertNotIn("filtered_row_count", payload)
        self.assertNotIn("truncated", payload)
        self.assertEqual(payload["displayed_row_count"], 3)
        self.assertEqual(payload["max_rows"], 100)
        self.assertFalse(payload["has_more"])
        self.assertEqual([column["name"] for column in payload["columns"]], ["Age", "Segment", "Score", "QuoteDate"])
        self.assertEqual([column["field"] for column in payload["columns"]], ["c0", "c1", "c2", "c3"])
        self.assertEqual([row["__row_id"] for row in payload["rows"]], [1, 2, 3])
        self.assertEqual(payload["rows"][0]["c0"], 2)
        self.assertEqual(payload["rows"][0]["c1"], "B")
        self.assertIsNone(payload["rows"][0]["c2"])
        self.assertEqual(payload["rows"][0]["c3"], "2024-01-02")
        self.assertEqual(payload["rows"][1]["c2"], 30.25)
        self.assertIsNone(payload["rows"][1]["c3"])
        self.assertEqual(payload["rows"][2]["c0"], 4)
        self.assertIsInstance(payload["timings"]["duckdb_ns"], int)
        self.assertGreaterEqual(payload["timings"]["duckdb_ns"], 0)
        self.assertIsInstance(payload["timings"]["duckdb_ms"], int)
        self.assertGreaterEqual(payload["timings"]["duckdb_ms"], 0)

    def test_table_caps_displayed_rows_at_one_hundred(self) -> None:
        large_path = self.root / "large.csv"
        large_path.write_text(
            "Index,Value\n" + "".join(f"{index},v{index}\n" for index in range(1, 1006)),
            encoding="utf-8",
        )
        app = create_app(large_path, token="", use_saved_filters=False, use_kpis=False)
        with patch.object(Dataset, "row_count", side_effect=AssertionError("dataset viewer should not count all rows")):
            payload = self.post_table(app, {"filter": "", "limit": 5000})

        self.assertNotIn("row_count", payload)
        self.assertNotIn("filtered_row_count", payload)
        self.assertEqual(payload["displayed_row_count"], 100)
        self.assertEqual(payload["max_rows"], 100)
        self.assertTrue(payload["has_more"])
        self.assertEqual(payload["rows"][0]["__row_id"], 1)
        self.assertEqual(payload["rows"][-1]["__row_id"], 100)
        self.assertFalse(any("first 100" in warning for warning in payload["warnings"]))

    def test_invalid_filter_returns_bad_request(self) -> None:
        app = create_app(self.data_path, token="", use_saved_filters=False, use_kpis=False)
        status, _, body = asgi_post_json(app, "/api/dataset-viewer/table", {"filter": "MissingColumn > 1", "limit": 1000})

        self.assertEqual(status, 400)
        self.assertIn("Invalid filter", body.decode("utf-8"))

    def test_filter_row_count_returns_exact_counts_and_timings(self) -> None:
        app = create_app(self.data_path, token="", use_saved_filters=False, use_kpis=False)

        unfiltered = self.post_filter_row_count(app, {"filter": ""})
        self.assertEqual(unfiltered["row_count"], 4)
        self.assertEqual(unfiltered["filtered_row_count"], 4)
        self.assertEqual(unfiltered["filter"], "")
        self.assertIsInstance(unfiltered["timings"]["duckdb_ns"], int)
        self.assertGreaterEqual(unfiltered["timings"]["duckdb_ns"], 0)
        self.assertIsInstance(unfiltered["timings"]["duckdb_ms"], int)
        self.assertGreaterEqual(unfiltered["timings"]["duckdb_ms"], 0)

        filtered = self.post_filter_row_count(app, {"filter": "Age >= 3"})
        self.assertEqual(filtered["row_count"], 4)
        self.assertEqual(filtered["filtered_row_count"], 2)
        self.assertEqual(filtered["filter"], "Age >= 3")

    def test_filter_row_count_invalid_filter_returns_bad_request(self) -> None:
        app = create_app(self.data_path, token="", use_saved_filters=False, use_kpis=False)
        status, _, body = asgi_post_json(app, "/api/filter/row-count", {"filter": "MissingColumn > 1"})

        self.assertEqual(status, 400)
        self.assertIn("Invalid filter", body.decode("utf-8"))


if __name__ == "__main__":
    unittest.main()
