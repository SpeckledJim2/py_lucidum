from __future__ import annotations

import asyncio
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import duckdb

from py_lucidum.app import create_app
from py_lucidum.core import Dataset, ModelPredictionSource, sql_literal
from py_lucidum.tools.gbm.validation import validate_request as validate_gbm_request
from py_lucidum.tools.glm.validation import validate_request as validate_glm_request
from py_lucidum.tools.histogram.query import histogram
from py_lucidum.tools.line_bar.query import chart
from py_lucidum.tools.uk_map.query import summary as map_summary


MODEL_SOURCE = "gbm:model-denominator:predictions"


class PredictionProvider:
    def __init__(self, source: ModelPredictionSource):
        self.source = source

    def has_source(self, source_id: str) -> bool:
        return source_id == self.source.source_id

    def relation_sql(self, source_id: str) -> str:
        if not self.has_source(source_id):
            raise ValueError("Choose a valid data source")
        return self.source.relation_sql

    def prediction_source(self, source_id: str) -> ModelPredictionSource | None:
        return self.source if self.has_source(source_id) else None

    def data_sources(self) -> list[dict[str, Any]]:
        model_kind = "glm" if self.source.column.startswith("glm_") else "gbm"
        return [
            {
                "id": self.source.source_id,
                "kind": f"{model_kind}_predictions",
                "active": True,
                "columns": [{"name": self.source.column, "kind": "numeric"}],
            }
        ]


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
    return int(start["status"]), json.loads(response_body)


class ModelDenominatorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.data_path = self.root / "sample.csv"
        self.data_path.write_text(
            "Age,Segment,PostcodeArea,PostcodeSector,PostcodeUnit,lat,long,Actual\n"
            "20,A,AB,AB10 1,AB10 1AA,57.1,-2.1,100\n"
            "30,A,AB,AB10 1,AB10 1AB,57.2,-2.2,200\n"
            "40,B,AL,AL1 1,AL1 1AA,51.7,-0.4,300\n"
            "50,B,AL,AL1 1,AL1 1AB,51.8,-0.3,400\n",
            encoding="utf-8",
        )
        prediction_path = self.root / "predictions.parquet"
        con = duckdb.connect(database=":memory:")
        try:
            con.execute(
                f"""
COPY (
  SELECT 1 AS __lucidum_row_id, 10.0 AS gbm_prediction
  UNION ALL SELECT 2, 20.0
  UNION ALL SELECT 3, 30.0
  UNION ALL SELECT 4, 40.0
) TO {sql_literal(str(prediction_path))} (FORMAT PARQUET)
"""
            )
        finally:
            con.close()
        source = ModelPredictionSource(
            source_id=MODEL_SOURCE,
            column="gbm_prediction",
            relation_sql=f"read_parquet({sql_literal(str(prediction_path))})",
            active=True,
        )
        self.dataset = Dataset(self.data_path)
        self.dataset.register_data_source_provider(PredictionProvider(source))

    def denominator_fields(self) -> dict[str, str]:
        return {
            "denominator": "gbm_prediction",
            "denominatorSource": MODEL_SOURCE,
        }

    def dataset_with_predictions(
        self,
        values: list[float | None],
        *,
        source_id: str = MODEL_SOURCE,
    ) -> Dataset:
        prediction_path = self.root / f"{source_id.replace(':', '_')}.parquet"
        selects = []
        for index, value in enumerate(values, start=1):
            value_sql = "CAST(NULL AS DOUBLE)" if value is None else str(float(value))
            selects.append(
                f"SELECT {index} AS __lucidum_row_id, {value_sql} AS gbm_prediction"
            )
        con = duckdb.connect(database=":memory:")
        try:
            con.execute(
                f"""
COPY (
  {" UNION ALL ".join(selects)}
) TO {sql_literal(str(prediction_path))} (FORMAT PARQUET)
"""
            )
        finally:
            con.close()
        dataset = Dataset(self.data_path)
        dataset.register_data_source_provider(
            PredictionProvider(
                ModelPredictionSource(
                    source_id=source_id,
                    column="gbm_prediction",
                    relation_sql=f"read_parquet({sql_literal(str(prediction_path))})",
                    active=True,
                )
            )
        )
        return dataset

    def test_one_and_two_feature_line_bar_use_prediction_denominator(self) -> None:
        request = {
            "source": "dataset",
            "x": "Segment",
            "responses": [{"label": "Actual", "numerator": "Actual"}],
            **self.denominator_fields(),
            "filter": "",
            "bandWidth": 0,
            "dateBucket": "none",
            "lowGroup": "0",
            "sort": "alpha",
            "sigma": 0,
            "transform": "none",
        }
        result = chart(self.dataset, request)
        by_group = {row["x"]: row for row in result["rows"]}

        self.assertEqual(result["denominator"]["source"], MODEL_SOURCE)
        self.assertEqual(result["field_sources"]["denominator"], MODEL_SOURCE)
        self.assertEqual(result["denominator"]["value"], 100)
        self.assertEqual(by_group["A"]["volume"], 30)
        self.assertEqual(by_group["B"]["volume"], 70)
        self.assertEqual(by_group["A"]["resp0"], 10)
        self.assertEqual(by_group["B"]["resp0"], 10)

        two_feature_result = chart(
            self.dataset,
            {
                **request,
                "groupings": [
                    {
                        "feature": "Age",
                        "source": "dataset",
                        "bandWidth": 10,
                        "quantileMode": "off",
                        "dateBucket": "none",
                        "asFactor": False,
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
            },
        )
        self.assertEqual(two_feature_result["denominator"]["source"], MODEL_SOURCE)
        self.assertEqual(two_feature_result["field_sources"]["denominator"], MODEL_SOURCE)
        self.assertEqual(sum(row["volume"] for row in two_feature_result["rows"]), 100)
        self.assertTrue(all(row["resp0"] == 10 for row in two_feature_result["rows"]))

    def test_histogram_map_and_metric_summary_use_prediction_denominator(self) -> None:
        histogram_result = histogram(
            self.dataset,
            {
                "source": "dataset",
                "actual": "Actual",
                **self.denominator_fields(),
                "bins": 2,
                "distribution": "incremental",
                "yAxis": "sum",
                "logScale": "none",
                "sampleMode": "all",
                "filter": "",
            },
        )
        self.assertEqual(histogram_result["denominator"]["source"], MODEL_SOURCE)
        self.assertEqual(histogram_result["denominator"]["value"], 100)
        weighted_mean = next(
            row["value"]
            for row in histogram_result["stats"]
            if row["statistic"] == "Weighted mean"
        )
        self.assertEqual(weighted_mean, 10)
        self.assertEqual(sum(row["volume"] for row in histogram_result["rows"]), 100)

        map_result = map_summary(
            self.dataset,
            {
                "source": "dataset",
                "level": "area",
                "numerator": "Actual",
                **self.denominator_fields(),
                "filter": "",
            },
        )
        map_rows = {row["key"]: row for row in map_result["rows"]}
        self.assertEqual(map_result["denominator"]["source"], MODEL_SOURCE)
        self.assertEqual(map_rows["AB"]["volume"], 30)
        self.assertEqual(map_rows["AL"]["volume"], 70)
        self.assertEqual(map_rows["AB"]["value"], 10)
        self.assertEqual(map_rows["AL"]["value"], 10)

        app = create_app(self.data_path, token="", tools=["line_bar"], use_saved_filters=False, use_kpis=False)
        app.state.dataset.register_data_source_provider(
            PredictionProvider(self.dataset.model_prediction_source(MODEL_SOURCE))
        )
        status, payload = asgi_post_json(
            app,
            "/api/metrics/summary",
            {
                "source": "dataset",
                "actual": "Actual",
                **self.denominator_fields(),
                "filter": "",
            },
        )
        self.assertEqual(status, 200)
        self.assertEqual(payload["denominator"]["source"], MODEL_SOURCE)
        self.assertEqual(payload["denominator"]["value"], 100)
        self.assertEqual(payload["response_summaries"][0]["value"], 10)

    def test_mixed_model_response_and_denominator_sources_join_once(self) -> None:
        glm_source_id = "glm:model-numerator:predictions"
        prediction_path = self.root / "glm_predictions.parquet"
        con = duckdb.connect(database=":memory:")
        try:
            con.execute(
                f"""
COPY (
  SELECT 1 AS __lucidum_row_id, 5.0 AS glm_prediction
  UNION ALL SELECT 2, 10.0
  UNION ALL SELECT 3, 15.0
  UNION ALL SELECT 4, 20.0
) TO {sql_literal(str(prediction_path))} (FORMAT PARQUET)
"""
            )
        finally:
            con.close()
        self.dataset.register_data_source_provider(
            PredictionProvider(
                ModelPredictionSource(
                    source_id=glm_source_id,
                    column="glm_prediction",
                    relation_sql=f"read_parquet({sql_literal(str(prediction_path))})",
                    active=True,
                )
            )
        )

        result = chart(
            self.dataset,
            {
                "source": "dataset",
                "x": "Segment",
                "responses": [
                    {
                        "label": "GLM",
                        "numerator": "glm_prediction",
                        "source": glm_source_id,
                    },
                    {"label": "Actual", "numerator": "Actual"},
                ],
                **self.denominator_fields(),
            },
        )

        self.assertEqual(result["row_count"], 4)
        self.assertEqual(sum(row["row_count"] for row in result["rows"]), 4)
        self.assertEqual(sum(row["volume"] for row in result["rows"]), 100)
        self.assertEqual(result["response_summaries"][0]["value"], 0.5)
        self.assertEqual(result["response_summaries"][1]["value"], 10)
        self.assertEqual(
            result["field_sources"],
            {
                "x": "dataset",
                "responses": [glm_source_id, "dataset"],
                "denominator": MODEL_SOURCE,
            },
        )

    def test_only_primary_model_outputs_are_valid_denominators(self) -> None:
        with self.assertRaisesRegex(ValueError, "primary model prediction"):
            chart(
                self.dataset,
                {
                    "source": "dataset",
                    "x": "Segment",
                    "responses": [{"label": "Actual", "numerator": "Actual"}],
                    "denominator": "gbm_prediction_rate",
                    "denominatorSource": MODEL_SOURCE,
                },
            )

    def test_missing_zero_and_negative_predictions_keep_existing_warning_rules(self) -> None:
        dataset = self.dataset_with_predictions([10, None, 0, -40])
        result = chart(
            dataset,
            {
                "source": "dataset",
                "x": "Segment",
                "responses": [{"label": "Actual", "numerator": "Actual"}],
                **self.denominator_fields(),
                "filter": "",
                "bandWidth": 0,
                "dateBucket": "none",
                "lowGroup": "0",
                "sort": "alpha",
                "sigma": 0,
                "transform": "none",
            },
        )

        self.assertEqual(result["row_count"], 4)
        self.assertEqual(result["denominator"]["missing_weight_rows"], 1)
        self.assertEqual(result["denominator"]["zero_weight_rows"], 1)
        self.assertEqual(result["denominator"]["negative_weight_rows"], 1)
        self.assertEqual(result["denominator"]["value"], -30)
        self.assertTrue(any("missing" in warning.lower() for warning in result["warnings"]))
        self.assertTrue(any("zero" in warning.lower() for warning in result["warnings"]))
        self.assertTrue(any("negative" in warning.lower() for warning in result["warnings"]))

    def test_model_training_validation_rejects_prediction_denominator_sources(self) -> None:
        glm_result = validate_glm_request(
            self.dataset,
            {
                "response_column": "Actual",
                "denominator_column": "gbm_prediction",
                "denominator_source": MODEL_SOURCE,
                "formula": "Actual ~ Age",
            },
        )
        gbm_result = validate_gbm_request(
            self.dataset,
            {
                "response": "Actual",
                "offset": "gbm_prediction",
                "offset_source": MODEL_SOURCE,
                "features": [{"name": "Age", "include": True}],
                "parameters": [],
            },
        )

        self.assertTrue(any("prediction chaining" in error for error in glm_result["errors"]))
        self.assertTrue(any("prediction chaining" in error for error in gbm_result.errors))

    def test_dataset_prediction_named_column_remains_a_valid_denominator(self) -> None:
        data_path = self.root / "physical_prediction.csv"
        data_path.write_text(
            "Actual,Age,Segment,gbm_prediction\n"
            "100,20,A,10\n"
            "200,30,B,20\n",
            encoding="utf-8",
        )
        dataset = Dataset(data_path)
        result = chart(
            dataset,
            {
                "source": "dataset",
                "x": "Segment",
                "responses": [{"label": "Actual", "numerator": "Actual"}],
                "denominator": "gbm_prediction",
                "denominatorSource": "dataset",
            },
        )
        glm_result = validate_glm_request(
            dataset,
            {
                "response_column": "Actual",
                "denominator_column": "gbm_prediction",
                "denominator_source": "dataset",
                "formula": "Actual ~ Age",
            },
        )
        gbm_result = validate_gbm_request(
            dataset,
            {
                "response": "Actual",
                "offset": "gbm_prediction",
                "offset_source": "dataset",
                "features": [{"name": "Age", "include": True}],
                "parameters": [],
            },
        )

        self.assertEqual(result["denominator"]["source"], "dataset")
        self.assertEqual(result["denominator"]["value"], 30)
        self.assertFalse(any("prediction chaining" in error for error in glm_result["errors"]))
        self.assertFalse(any("prediction chaining" in error for error in gbm_result.errors))
