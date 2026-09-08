from __future__ import annotations

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import duckdb

from py_lucidum import smooth_postcode_sectors
from py_lucidum.app import create_app
from py_lucidum.core import ColumnInfo, Dataset, infer_kind, is_numeric_kind, is_response_column, normalise_denominator, sql_literal
from py_lucidum.core.kpis import KPI_SPEC_COLUMNS
from py_lucidum.tools.glm.validation import validate_request as validate_glm
from py_lucidum.tools.gbm.validation import selected_response_column, response_objective_errors
from py_lucidum.tools.histogram.query import histogram
from py_lucidum.tools.line_bar.favourites import LineBarFavouriteStore
from py_lucidum.tools.line_bar.query import chart, table, normalise_responses
from py_lucidum.tools.specifications.service import validate_spec
from py_lucidum.tools.uk_map.query import summary as map_summary
from tests.test_line_bar import asgi_post_json


class BooleanResponseTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.path = self.root / "boolean_responses.parquet"
        con = duckdb.connect()
        try:
            con.execute(f"""
COPY (
  SELECT *, CAST(BooleanResponse AS INTEGER) AS NumericResponse,
    CAST(BooleanResponse AS VARCHAR) AS TextResponse
  FROM (VALUES
    (1, TRUE, 2, 'A', 'AB', 'AB10 1', 'AB10 1AA', 57.1, -2.1),
    (2, FALSE, 2, 'A', 'AB', 'AB10 1', 'AB10 1AB', 57.2, -2.2),
    (3, TRUE, 4, 'B', 'AL', 'AL1 1', 'AL1 1AA', 51.7, -0.4),
    (4, FALSE, 4, 'B', 'AL', 'AL1 1', 'AL1 1AB', 51.8, -0.5),
    (5, NULL, 2, 'A', 'AB', 'AB10 1', 'AB10 1AC', 57.3, -2.3),
    (6, TRUE, 2, 'A', 'AB', 'AB10 1', 'AB10 1AD', 57.4, -2.4)
  ) AS t(id, BooleanResponse, Exposure, Segment, PostcodeArea, PostcodeSector, PostcodeUnit, lat, long)
) TO {sql_literal(str(self.path))} (FORMAT PARQUET)
""")
        finally:
            con.close()
        self.dataset = Dataset(self.path)
        self.addCleanup(self.dataset.con.close)

    def canonical(self, value: Any) -> Any:
        if isinstance(value, dict):
            return {key: self.canonical(item) for key, item in value.items() if key != "timings"}
        if isinstance(value, list):
            items = [self.canonical(item) for item in value]
            if items and all(isinstance(item, dict) and "key" in item for item in items):
                items.sort(key=lambda item: str(item["key"]))
            return items
        if isinstance(value, str):
            return value.replace("NumericResponse", "BooleanResponse")
        return value

    def assert_response_parity(self, callback: Any) -> None:
        def compare(left: Any, right: Any, path: str = "result") -> None:
            if isinstance(left, dict) and isinstance(right, dict):
                self.assertEqual(left.keys(), right.keys(), path)
                for key in left:
                    compare(left[key], right[key], f"{path}.{key}")
            elif isinstance(left, list) and isinstance(right, list):
                self.assertEqual(len(left), len(right), path)
                for index, (a, b) in enumerate(zip(left, right)):
                    compare(a, b, f"{path}[{index}]")
            elif isinstance(left, float) and isinstance(right, float):
                self.assertAlmostEqual(left, right, places=12, msg=path)
            else:
                self.assertEqual(left, right, path)
        compare(self.canonical(callback("BooleanResponse")), self.canonical(callback("NumericResponse")))

    def line_request(self, response: str, **overrides: Any) -> dict[str, Any]:
        return {
            "x": "Segment", "bandWidth": 0, "tailPercent": "0", "sort": "alpha",
            "denominator": "__none__", "responses": [{"label": response, "numerator": response}],
            **overrides,
        }

    def test_response_eligibility_keeps_boolean_features_categorical(self) -> None:
        for dtype, expected in [("BOOLEAN", True), ("BOOL", True), ("INTEGER", True), ("DOUBLE", True),
                                ("VARCHAR", False), ("BOOLEAN[]", False), ("BOOLEAN[2]", False),
                                ("STRUCT(flag BOOLEAN)", False), ("DATE", False)]:
            with self.subTest(dtype=dtype):
                column = ColumnInfo("response", dtype, infer_kind(dtype))
                self.assertEqual(is_response_column(column), expected)
        column = self.dataset.column_map()["BooleanResponse"]
        self.assertEqual(column.kind, "categorical")
        self.assertFalse(is_numeric_kind(column.kind))
        with self.assertRaisesRegex(ValueError, "numeric Weight"):
            normalise_denominator("BooleanResponse", self.dataset.column_map())
        responses = normalise_responses([
            {"numerator": "BooleanResponse"}, {"numerator": "BooleanResponse"},
            {"numerator": "NumericResponse"},
        ], self.dataset.column_map())
        self.assertEqual([item["numerator"] for item in responses], ["BooleanResponse", "NumericResponse"])
        self.assertEqual(normalise_responses([None, {"numerator": "BooleanResponse"}], self.dataset.column_map()), [])
        for response in ["TextResponse", "Segment"]:
            self.assertEqual(normalise_responses([{"numerator": response}], self.dataset.column_map()), [])

    def test_line_bar_and_histogram_match_integer_responses(self) -> None:
        for denominator in ["__none__", "Exposure"]:
            for filter_sql in ["", "BooleanResponse IS TRUE", "BooleanResponse IS FALSE", "id < 0", "BooleanResponse IS NULL"]:
                with self.subTest(denominator=denominator, filter=filter_sql):
                    options = {"denominator": denominator, "filter": filter_sql}
                    self.assert_response_parity(lambda name: chart(self.dataset, self.line_request(name, **options)))
                    self.assert_response_parity(lambda name: table(self.dataset, self.line_request(name, **options)))
                    self.assert_response_parity(lambda name: chart(self.dataset, self.line_request(name, **options,
                        groupings=[{"feature": "Segment"}, {"feature": "PostcodeArea"}])))
                    self.assert_response_parity(lambda name: histogram(self.dataset, {"actual": name, "sampleMode": "all", "bins": 10, **options}))
        for options in [{"binWidth": 1}, {"binWidth": 0.25}, {"logScale": "x"}, {"distribution": "cumulative"}]:
            self.assert_response_parity(lambda name: histogram(self.dataset, {"actual": name, "sampleMode": "all", **options}))

    def test_map_and_smoothing_export_match_integer_responses(self) -> None:
        for level, smoothing in [("area", 0), ("sector", 0), ("sector", 1), ("unit", 0)]:
            for denominator in ["__none__", "Exposure"]:
                with self.subTest(level=level, smoothing=smoothing, denominator=denominator):
                    self.assert_response_parity(lambda name: map_summary(self.dataset, {
                        "numerator": name, "level": level, "smoothingLevel": smoothing, "denominator": denominator,
                    }))
        for filter_sql in ["BooleanResponse IS TRUE", "BooleanResponse IS FALSE", "id < 0", "BooleanResponse IS NULL"]:
            self.assert_response_parity(lambda name: map_summary(self.dataset, {"numerator": name, "filter": filter_sql}))
        paths = []
        for response in ["BooleanResponse", "NumericResponse"]:
            paths.append(smooth_postcode_sectors(self.path, self.root / f"{response}.parquet",
                postcode_sector="PostcodeSector", numerator=response, denominator="Exposure", filter="id > 1"))
        rows = [self.dataset.con.execute(f"SELECT * FROM read_parquet({sql_literal(str(path))}) ORDER BY 1").fetchall() for path in paths]
        self.assertEqual(rows[0], rows[1])

    def test_summary_kpis_and_favourites_accept_boolean_numerators(self) -> None:
        app = create_app(self.path, token="", use_kpis=False, use_saved_filters=False)
        self.addCleanup(app.state.dataset.con.close)
        def summary(name: str) -> Any:
            status, _, body = asgi_post_json(app, "/api/metrics/summary", {"actual": name, "denominator": "__none__"})
            self.assertEqual(status, 200, body)
            return json.loads(body)
        self.assert_response_parity(summary)
        result = summary("BooleanResponse")
        self.assertEqual(result["response_summaries"][0]["value"], 0.6)
        self.assertEqual(result["response_summaries"][0]["numerator"], 3)
        self.assertEqual(result["response_summaries"][0]["denominator"], 5)
        row = dict(group="Flags", name="True share", actual="BooleanResponse", denominator="N", decimals="1", format="percent")
        self.assertTrue(validate_spec(self.dataset, "kpis", KPI_SPEC_COLUMNS, [row])["valid"])
        self.assertFalse(validate_spec(self.dataset, "kpis", KPI_SPEC_COLUMNS, [{**row, "denominator": "BooleanResponse"}])["valid"])
        store = LineBarFavouriteStore(self.path, self.dataset)
        for scope in ["metrics", "line_bar_view", "histogram_view", "map_view"]:
            view = {"scope": scope, "source": "dataset", "x": "Segment", "actual": {"value": "BooleanResponse", "sourceId": "dataset"}, "denominator": "__none__"}
            saved = store.create_favourite(scope, view)
            self.assertTrue(saved["validation"]["valid"], saved)
        restored = LineBarFavouriteStore(self.path, self.dataset).list_favourites()
        self.assertEqual(len(restored), 4)
        self.assertTrue(all(item["validation"]["valid"] for item in restored))

    def test_model_validation_preserves_objective_and_missing_rules(self) -> None:
        for family in ["normal", "binomial"]:
            result = validate_glm(self.dataset, {"response_column": "BooleanResponse", "formula": "Segment", "family": family})
            self.assertTrue(result["ok"], result)
            self.assertEqual(result["family"], family)
        self.assertFalse(validate_glm(self.dataset, {"response_column": "TextResponse", "formula": "Segment"})["ok"])
        self.assertEqual(selected_response_column({"response": "BooleanResponse"}, self.dataset.column_map()), "BooleanResponse")
        for objective in ["regression", "binary", "gamma"]:
            self.assertEqual(response_objective_errors(self.dataset, objective, "BooleanResponse"),
                             response_objective_errors(self.dataset, objective, "NumericResponse"))
        self.assertIn("missing", " ".join(response_objective_errors(self.dataset, "binary", "BooleanResponse")))
        with self.assertRaisesRegex(ValueError, "numeric or Boolean"):
            selected_response_column({"response": "TextResponse"}, self.dataset.column_map())
