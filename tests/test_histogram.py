from __future__ import annotations

import asyncio
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

from py_lucidum.app import create_app, normalise_tools
from py_lucidum.core import Dataset
from py_lucidum.tools.histogram.query import histogram


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


class HistogramToolTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.data_path = Path(self.tmp.name) / "sample.csv"
        self.data_path.write_text(
            "Actual,Weight,Segment\n"
            "1,1,A\n"
            "2,1,A\n"
            "3,2,B\n"
            "4,2,B\n"
            "5,0,B\n"
            ",3,C\n"
            "6,-1,C\n"
            "8,4,C\n",
            encoding="utf-8",
        )

    def request(self, **overrides: Any) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "source": "dataset",
            "actual": "Actual",
            "denominator": "__none__",
            "bins": 2,
            "distribution": "incremental",
            "yAxis": "sum",
            "logScale": "none",
            "sampleMode": "all",
            "filter": "",
        }
        payload.update(overrides)
        return payload

    def test_app_registers_histogram_default_after_line_bar(self) -> None:
        self.assertEqual(normalise_tools(None), ["dataset_viewer", "column_profile", "line_bar", "histogram", "uk_map"])
        app = create_app(self.data_path, token="")
        paths = {route.path for route in app.routes}

        self.assertEqual(app.state.enabled_tools, ["dataset_viewer", "column_profile", "line_bar", "histogram", "uk_map"])
        self.assertIn("/api/histogram/chart", paths)

        status, _, body = asgi_post_json(app, "/api/histogram/chart", self.request())
        payload = json.loads(body)

        self.assertEqual(status, 200)
        self.assertIn("rows", payload)
        self.assertIsInstance(payload["timings"]["duckdb_ns"], int)
        self.assertGreaterEqual(payload["timings"]["duckdb_ns"], 0)

    def test_unweighted_actual_histogram_uses_row_count_volume(self) -> None:
        result = histogram(Dataset(self.data_path), self.request(denominator="Average row value"))

        self.assertEqual(result["valid_count"], 7)
        self.assertEqual(result["sampled_valid_count"], 7)
        self.assertEqual([row["row_count"] for row in result["rows"]], [4, 3])
        self.assertEqual([row["volume"] for row in result["rows"]], [4, 3])
        self.assertEqual(result["stats"][0], {"statistic": "Numeric count", "value": 7})
        self.assertEqual(result["stats"][1], {"statistic": "NA count", "value": 1})

    def test_weighted_ratio_histogram_uses_weight_sum_volume(self) -> None:
        result = histogram(Dataset(self.data_path), self.request(denominator="Weight"))

        self.assertEqual(result["valid_count"], 5)
        self.assertEqual([row["row_count"] for row in result["rows"]], [1, 4])
        self.assertEqual([row["volume"] for row in result["rows"]], [1, 9])
        self.assertAlmostEqual(result["rows"][0]["probability"], 0.1)
        self.assertAlmostEqual(result["rows"][1]["probability"], 0.9)
        weighted_mean = next(row["value"] for row in result["stats"] if row["statistic"] == "Weighted mean")
        self.assertAlmostEqual(weighted_mean, 1.8)
        self.assertTrue(any("Weight was zero" in warning for warning in result["warnings"]))
        self.assertTrue(any("Weight was negative" in warning for warning in result["warnings"]))

    def test_probability_and_cumulative_modes(self) -> None:
        result = histogram(
            Dataset(self.data_path),
            self.request(denominator="Weight", yAxis="probability", distribution="cumulative"),
        )

        self.assertAlmostEqual(result["rows"][0]["height"], 0.1)
        self.assertAlmostEqual(result["rows"][1]["height"], 1.0)
        self.assertEqual(result["rows"][1]["cumulative_volume"], 10)

    def test_filter_selection_affects_rows_and_stats(self) -> None:
        result = histogram(Dataset(self.data_path), self.request(filter="Segment = 'A'"))

        self.assertEqual(result["filtered_row_count"], 2)
        self.assertEqual(result["valid_count"], 2)
        self.assertEqual(sum(row["row_count"] for row in result["rows"]), 2)
        mean = next(row["value"] for row in result["stats"] if row["statistic"] == "Mean")
        self.assertEqual(mean, 1.5)

    def test_auto_bins_clamp_and_zero_valid_rows_warn(self) -> None:
        result = histogram(Dataset(self.data_path), self.request(bins="auto"))

        self.assertEqual(result["bins"], 10)

        empty = histogram(Dataset(self.data_path), self.request(filter="Actual IS NULL"))
        self.assertEqual(empty["valid_count"], 0)
        self.assertEqual(empty["rows"], [])
        self.assertTrue(any("No valid histogram values" in warning for warning in empty["warnings"]))

    def test_log_x_excludes_nonpositive_values(self) -> None:
        log_path = Path(self.tmp.name) / "log_sample.csv"
        log_path.write_text(
            "Actual,Weight\n"
            "-1,1\n"
            "0,1\n"
            "10,1\n"
            "100,1\n",
            encoding="utf-8",
        )

        result = histogram(Dataset(log_path), self.request(actual="Actual", logScale="x", bins=2))

        self.assertEqual(result["valid_count"], 2)
        self.assertEqual(sum(row["row_count"] for row in result["rows"]), 2)
        self.assertTrue(any("nonpositive values excluded" in warning for warning in result["warnings"]))


if __name__ == "__main__":
    unittest.main()
