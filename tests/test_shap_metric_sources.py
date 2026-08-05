from __future__ import annotations

import asyncio
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import duckdb

from py_lucidum.app import create_app
from py_lucidum.core import sql_literal
from py_lucidum.tools.gbm.store import GbmModelStore


def asgi_post_json(app: Any, path: str, payload: dict[str, Any]) -> tuple[int, dict[str, Any]]:
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
    response_body = b"".join(
        message.get("body", b"")
        for message in messages
        if message["type"] == "http.response.body"
    )
    return start["status"], json.loads(response_body)


class ShapMetricSourceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.data_path = Path(self.tmp.name) / "shap_metrics.csv"
        self.data_path.write_text(
            "Age,Segment,PostcodeArea,PostcodeSector,PostcodeUnit,lat,long,Actual,Weight\n"
            "30,A,AB,AB10 1,AB10 1AA,57.1,-2.1,100,10\n"
            "40,B,AB,AB10 1,AB10 1AB,57.2,-2.2,200,20\n"
            "50,A,AL,AL1 1,AL1 1AA,51.7,-0.4,300,30\n"
            "60,B,AL,AL1 2,AL1 2AA,51.8,-0.3,400,40\n",
            encoding="utf-8",
        )
        self.model_id = "shap-metrics"
        self.source_id = f"gbm:{self.model_id}:shap_long"
        self.write_gbm_sidecar()
        self.app = create_app(
            self.data_path,
            token="",
            tools=["line_bar", "histogram", "uk_map", "gbm"],
            use_saved_filters=False,
            use_kpis=False,
            use_features=False,
        )

    def write_gbm_sidecar(self) -> None:
        store = GbmModelStore(self.data_path)
        model_dir = store.create_model_dir(self.model_id)
        store.write_json(
            model_dir / "manifest.json",
            {
                "model_id": self.model_id,
                "label": "SHAP metrics",
                "created_at": "2026-08-05T00:00:00Z",
                "training_mode": "normal",
                "response_column": "Actual",
                "offset_column": "Weight",
                "best_iteration": 3,
                "training_rows": 3,
                "test_rows": 0,
                "scored_rows": 4,
                "shap_rows": 3,
            },
        )
        store.write_json(
            model_dir / "parameters.json",
            {"objective": "gamma", "metric": "gamma", "num_iterations": 3},
        )
        con = duckdb.connect(database=":memory:")
        try:
            con.execute(
                f"""
COPY (
  SELECT 1 AS __lucidum_row_id, 10.0 AS gbm_prediction
  UNION ALL SELECT 2, 20.0
  UNION ALL SELECT 3, 30.0
  UNION ALL SELECT 4, 40.0
) TO {sql_literal(str(model_dir / "predictions.parquet"))} (FORMAT PARQUET)
"""
            )
            con.execute(
                f"""
COPY (
  SELECT 1 AS __lucidum_row_id, 0.1 AS Age
  UNION ALL SELECT 2, -0.2
  UNION ALL SELECT 3, 0.3
) TO {sql_literal(str(model_dir / "shap_values.parquet"))} (FORMAT PARQUET)
"""
            )
            con.execute(
                f"""
COPY (
  SELECT 'Age' AS feature, 0.2 AS mean_abs_shap, 0.06666666666666667 AS mean_shap, 3 AS row_count
) TO {sql_literal(str(model_dir / "shap_summary.parquet"))} (FORMAT PARQUET)
"""
            )
        finally:
            con.close()
        store.activate_model(self.model_id)

    def line_bar_request(self, **overrides: Any) -> dict[str, Any]:
        request = {
            "source": self.source_id,
            "x": "Segment",
            "xSource": "dataset",
            "responses": [{"label": "SHAP Age", "numerator": "SHAP__Age"}],
            "denominator": "Weight",
            "denominatorSource": "dataset",
            "filter": "",
            "bandWidth": 0,
            "dateBucket": "none",
            "lowGroup": "0",
            "sort": "alpha",
            "sigma": 0,
            "transform": "none",
            "partialDependence": {"mode": "none"},
        }
        request.update(overrides)
        return request

    def test_line_bar_one_and_two_feature_requests_return_shap_values(self) -> None:
        status, payload = asgi_post_json(self.app, "/api/chart", self.line_bar_request())

        self.assertEqual(status, 200)
        self.assertEqual(payload["row_count"], 3)
        self.assertEqual(payload["field_sources"]["x"], "dataset")
        self.assertEqual(payload["field_sources"]["responses"], [self.source_id])
        self.assertEqual(payload["field_sources"]["denominator"], "dataset")
        self.assertEqual(payload["responses"][0]["numerator"], "SHAP__Age")
        self.assertEqual({row["x"] for row in payload["rows"]}, {"A", "B"})
        self.assertTrue(all(row["resp0"] is not None for row in payload["rows"]))

        two_feature_request = self.line_bar_request(
            groupings=[
                {
                    "feature": "Age",
                    "source": "dataset",
                    "bandWidth": 10,
                    "quantileMode": "off",
                    "dateBucket": "none",
                    "asFactor": True,
                },
                {
                    "feature": "Segment",
                    "source": "dataset",
                    "bandWidth": 0,
                    "quantileMode": "off",
                    "dateBucket": "none",
                    "asFactor": False,
                },
            ],
            tailPercent=0,
        )
        two_status, two_payload = asgi_post_json(self.app, "/api/chart", two_feature_request)

        self.assertEqual(two_status, 200)
        self.assertEqual(two_payload["row_count"], 3)
        self.assertEqual(two_payload["field_sources"]["groupings"], ["dataset", "dataset"])
        self.assertEqual(two_payload["field_sources"]["responses"], [self.source_id])
        self.assertEqual(len(two_payload["rows"]), 3)
        self.assertTrue(all(row["resp0"] is not None for row in two_payload["rows"]))

    def test_map_histogram_and_metric_summary_accept_shap_numerator(self) -> None:
        metric_status, metric_payload = asgi_post_json(
            self.app,
            "/api/metrics/summary",
            {
                "source": self.source_id,
                "actual": "SHAP__Age",
                "denominator": "__none__",
                "denominatorSource": "dataset",
                "filter": "",
            },
        )
        self.assertEqual(metric_status, 200)
        self.assertEqual(metric_payload["row_count"], 3)
        self.assertAlmostEqual(metric_payload["response_summaries"][0]["value"], 0.06666666666666667)

        map_status, map_payload = asgi_post_json(
            self.app,
            "/api/uk-map/summary",
            {
                "source": self.source_id,
                "level": "area",
                "numerator": "SHAP__Age",
                "denominator": "__none__",
                "denominatorSource": "dataset",
                "filter": "",
                "areaColumn": "PostcodeArea",
            },
        )
        self.assertEqual(map_status, 200)
        self.assertEqual(map_payload["row_count"], 3)
        self.assertEqual({row["key"] for row in map_payload["rows"]}, {"AB", "AL"})
        self.assertTrue(all(row["value"] is not None for row in map_payload["rows"]))

        histogram_status, histogram_payload = asgi_post_json(
            self.app,
            "/api/histogram/chart",
            {
                "source": self.source_id,
                "actual": "SHAP__Age",
                "denominator": "__none__",
                "denominatorSource": "dataset",
                "bins": 2,
                "distribution": "incremental",
                "yAxis": "sum",
                "logScale": "none",
                "sampleMode": "all",
                "filter": "",
            },
        )
        self.assertEqual(histogram_status, 200)
        self.assertEqual(histogram_payload["row_count"], 3)
        self.assertEqual(histogram_payload["valid_count"], 3)
        self.assertEqual(sum(row["row_count"] for row in histogram_payload["rows"]), 3)


if __name__ == "__main__":
    unittest.main()
