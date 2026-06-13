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
from py_lucidum.tools.uk_map import query as uk_map_query
from py_lucidum.tools.uk_map.query import summary
from py_lucidum.tools.uk_map.smoothing import (
    DEFAULT_SECTOR_ADJACENCY_PATH,
    load_sector_adjacency,
    normalise_smoothing_level,
    smooth_sector_rows,
)


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


class UkMapToolTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.data_path = self.root / "sample.csv"
        self.data_path.write_text(
            "PostcodeArea,PostcodeSector,PostcodeUnit,CustomArea,CustomUnit,lat,long,CustomLat,CustomLong,Actual,Weight,Flag\n"
            "AB,AB10 1,AB10 1AA,A1,U1,57.1,-2.1,58.1,-3.1,100,10,1\n"
            "AB,AB10 1,AB10 1AA,A1,U1,57.3,-2.3,58.3,-3.3,200,20,1\n"
            "AL,AL1 1,AL1 1AA,A2,U2,51.7,-0.4,52.7,-1.4,300,30,1\n"
            "AL,AL1 2,AL1 2AA,A2,U3,51.8,-0.3,52.8,-1.3,400,,0\n"
            ",,ZZ1 1ZZ,A3,U4,999,-2,53,-1,500,50,1\n",
            encoding="utf-8",
        )
        self.upper_data_path = self.root / "uppercase_sample.csv"
        self.upper_data_path.write_text(
            "POSTCODE_AREA,POSTCODE_SECTOR,POSTCODE_UNIT,LATITUDE,LONGITUDE,Actual,Weight\n"
            "AB,AB10 1,AB10 1AA,57.1,-2.1,100,10\n"
            "AB,AB10 1,AB10 1AA,57.3,-2.3,200,20\n"
            "AL,AL1 1,AL1 1AA,51.7,-0.4,300,30\n",
            encoding="utf-8",
        )
        self.mixed_coordinate_path = self.root / "mixed_coordinate_sample.csv"
        self.mixed_coordinate_path.write_text(
            "PostcodeUnit,latitude,LONGiTUDE,Actual\n"
            "AB10 1AA,57.1,-2.1,100\n"
            "AL1 1AA,51.7,-0.4,300\n",
            encoding="utf-8",
        )

    def request(self, **overrides: object) -> dict[str, object]:
        request: dict[str, object] = {
            "level": "area",
            "numerator": "Actual",
            "denominator": "__none__",
            "filter": "",
        }
        request.update(overrides)
        return request

    def test_default_tools_include_uk_map(self) -> None:
        self.assertEqual(normalise_tools(None), ["dataset_viewer", "column_profile", "line_bar", "histogram", "uk_map", "specs"])

        app = create_app(self.data_path, token="dev-token")
        paths = {route.path for route in app.routes}

        self.assertEqual(app.state.enabled_tools, ["dataset_viewer", "column_profile", "line_bar", "histogram", "uk_map", "specs"])
        self.assertIn("/api/dataset-viewer/table", paths)
        self.assertIn("/api/column-profile/summary", paths)
        self.assertIn("/api/histogram/chart", paths)
        self.assertIn("/api/uk-map/summary", paths)
        self.assertIn("/tools/uk-map/static", paths)

    def test_create_app_persists_unit_point_defaults(self) -> None:
        app = create_app(
            self.data_path,
            defaults={
                "postcode_unit": "CustomUnit",
                "latitude": "CustomLat",
                "longitude": "CustomLong",
            },
        )

        self.assertEqual(app.state.defaults["postcode_unit"], "CustomUnit")
        self.assertEqual(app.state.defaults["latitude"], "CustomLat")
        self.assertEqual(app.state.defaults["longitude"], "CustomLong")

    def test_summary_endpoint_includes_duckdb_timing(self) -> None:
        app = create_app(self.data_path, token="", tools=["uk_map"], use_saved_filters=False, use_kpis=False)

        status, _, body = asgi_post_json(app, "/api/uk-map/summary", self.request())
        payload = json.loads(body)

        self.assertEqual(status, 200)
        self.assertEqual(payload["level"], "area")
        self.assertIn("rows", payload)
        self.assertIn("response", payload)
        self.assertIsInstance(payload["timings"]["server_ns"], int)
        self.assertGreaterEqual(payload["timings"]["server_ns"], 0)
        self.assertIsInstance(payload["timings"]["server_ms"], int)
        self.assertGreaterEqual(payload["timings"]["server_ms"], 0)
        self.assertIsInstance(payload["timings"]["duckdb_ns"], int)
        self.assertGreaterEqual(payload["timings"]["duckdb_ns"], 0)
        self.assertIsInstance(payload["timings"]["duckdb_ms"], int)
        self.assertGreaterEqual(payload["timings"]["duckdb_ms"], 0)
        self.assertLessEqual(payload["timings"]["duckdb_ns"], payload["timings"]["server_ns"])
        self.assertNotIn("app_ns", payload["timings"])
        self.assertNotIn("app_ms", payload["timings"])

    def test_area_summary_uses_average_row_value(self) -> None:
        dataset = Dataset(self.data_path)
        result = summary(dataset, self.request())

        self.assertEqual(result["level"], "area")
        self.assertEqual(result["join_column"], "PostcodeArea")
        self.assertEqual(result["join_property"], "PostcodeArea")
        self.assertEqual(result["row_count"], 5)
        self.assertEqual(result["filtered_row_count"], 5)
        self.assertEqual(result["response"]["value"], 300)
        self.assertEqual(result["denominator"]["value"], 5)
        self.assertEqual([(row["key"], row["value"], row["denominator"]) for row in result["rows"]], [("AB", 150, 2), ("AL", 350, 2)])

    def test_sector_summary_applies_filter_and_weight(self) -> None:
        dataset = Dataset(self.data_path)
        result = summary(
            dataset,
            self.request(level="sector", denominator="Weight", filter="PostcodeArea = 'AL'"),
        )

        self.assertEqual(result["join_column"], "PostcodeSector")
        self.assertEqual(result["filtered_row_count"], 2)
        self.assertEqual(result["denominator"]["value"], 30)
        self.assertEqual(
            [(row["key"], row["value"], row["denominator"]) for row in result["rows"]],
            [("AL1 1", 10, 30), ("AL1 2", None, 0)],
        )
        self.assertIn("1 row excluded from Weight because Weight was missing.", result["warnings"])

    def test_sector_summary_accepts_smoothing_level(self) -> None:
        dataset = Dataset(self.data_path)
        result = summary(dataset, self.request(level="sector", smoothingLevel=1))

        self.assertEqual(result["smoothing"]["level"], 1)
        self.assertEqual(result["smoothing"]["requested_level"], 1)
        self.assertTrue(result["smoothing"]["applied"])
        self.assertEqual(result["smoothing"]["method"], "shared_edge_weighted_numerator")
        self.assertIn("raw_value", result["rows"][0])

    def test_smoothing_level_is_validated(self) -> None:
        self.assertEqual(normalise_smoothing_level("none"), 0)
        self.assertEqual(normalise_smoothing_level(5), 5)

        dataset = Dataset(self.data_path)
        with self.assertRaisesRegex(ValueError, "Choose a smoothing level from None to 5"):
            summary(dataset, self.request(level="sector", smoothingLevel=6))

        with self.assertRaisesRegex(ValueError, "Choose a smoothing level from None to 5"):
            summary(dataset, self.request(level="sector", smoothingLevel=1.5))

    def test_area_and_unit_smoothing_requests_are_no_ops(self) -> None:
        dataset = Dataset(self.data_path)

        area_result = summary(dataset, self.request(level="area", smoothingLevel=2))
        unit_result = summary(dataset, self.request(level="unit", smoothingLevel=2))

        self.assertEqual(area_result["smoothing"]["requested_level"], 2)
        self.assertEqual(area_result["smoothing"]["level"], 0)
        self.assertFalse(area_result["smoothing"]["applied"])
        self.assertEqual(unit_result["smoothing"]["requested_level"], 2)
        self.assertEqual(unit_result["smoothing"]["level"], 0)
        self.assertFalse(unit_result["smoothing"]["applied"])

    def test_sector_smoothing_pools_weighted_numerators(self) -> None:
        adjacency_path = self.root / "adjacency.json"
        adjacency_path.write_text(
            json.dumps({
                "keys": ["A", "B", "C", "D", "E", "F"],
                "neighbour_type": "shared_edge",
                "neighbours": [[1], [0, 2, 4], [1], [], [1, 5], [4]],
            }),
            encoding="utf-8",
        )
        load_sector_adjacency.cache_clear()
        self.addCleanup(load_sector_adjacency.cache_clear)
        rows = [
            {"key": "A", "row_count": 1, "numerator": 100, "denominator": 10, "volume": 10, "value": 10},
            {"key": "B", "row_count": 1, "numerator": 600, "denominator": 30, "volume": 30, "value": 20},
            {"key": "C", "row_count": 1, "numerator": None, "denominator": 0, "volume": 0, "value": None},
            {"key": "D", "row_count": 1, "numerator": 500, "denominator": 100, "volume": 100, "value": 5},
            {"key": "X", "row_count": 1, "numerator": 9, "denominator": 1, "volume": 1, "value": 9},
        ]

        smoothed, metadata, warning = smooth_sector_rows(rows, 1, adjacency_path=str(adjacency_path))
        smoothed_by_key = {row["key"]: row for row in smoothed}

        self.assertIsNone(warning)
        self.assertTrue(metadata["applied"])
        self.assertEqual(metadata["matched_rows"], 5)
        self.assertEqual(metadata["target_rows"], 6)
        self.assertEqual(metadata["smoothed_rows"], 5)
        self.assertEqual(metadata["fallback_rows"], 2)
        self.assertEqual(smoothed_by_key["A"]["value"], 17.5)
        self.assertEqual(smoothed_by_key["A"]["numerator"], 700)
        self.assertEqual(smoothed_by_key["A"]["denominator"], 40)
        self.assertEqual(smoothed_by_key["A"]["raw_value"], 10)
        self.assertEqual(smoothed_by_key["A"]["smoothing_contributing_sectors"], 2)
        self.assertEqual(smoothed_by_key["B"]["value"], 17.5)
        self.assertEqual(smoothed_by_key["C"]["value"], 20)
        self.assertIsNone(smoothed_by_key["C"]["raw_value"])
        self.assertEqual(smoothed_by_key["C"]["smoothing_contributing_sectors"], 1)
        self.assertEqual(smoothed_by_key["D"]["value"], 5)
        self.assertEqual(smoothed_by_key["D"]["smoothing_contributing_sectors"], 1)
        self.assertEqual(smoothed_by_key["E"]["value"], 20)
        self.assertEqual(smoothed_by_key["E"]["row_count"], 0)
        self.assertIsNone(smoothed_by_key["E"]["raw_value"])
        self.assertEqual(smoothed_by_key["E"]["smoothing_contributing_sectors"], 1)
        self.assertIsNone(smoothed_by_key["F"]["value"])
        self.assertEqual(smoothed_by_key["F"]["smoothing_contributing_sectors"], 0)
        self.assertEqual(smoothed_by_key["X"]["value"], 9)
        self.assertEqual(smoothed_by_key["X"]["raw_value"], 9)
        self.assertEqual(smoothed_by_key["X"]["smoothing_contributing_sectors"], 0)

    def test_sector_smoothing_level_zero_does_not_synthesise_missing_sectors(self) -> None:
        adjacency_path = self.root / "adjacency_zero.json"
        adjacency_path.write_text(
            json.dumps({
                "keys": ["A", "B"],
                "neighbour_type": "shared_edge",
                "neighbours": [[1], [0]],
            }),
            encoding="utf-8",
        )
        rows = [{"key": "A", "row_count": 1, "numerator": 100, "denominator": 10, "volume": 10, "value": 10}]

        smoothed, metadata, warning = smooth_sector_rows(rows, 0, adjacency_path=str(adjacency_path))

        self.assertIsNone(warning)
        self.assertFalse(metadata["applied"])
        self.assertEqual(metadata["target_rows"], 1)
        self.assertEqual(smoothed, rows)

    def test_sector_summary_smoothing_returns_synthetic_missing_sector_rows(self) -> None:
        data_path = self.root / "sector_smoothing_endpoint.csv"
        data_path.write_text(
            "PostcodeSector,Actual\n"
            "A,100\n",
            encoding="utf-8",
        )
        adjacency_path = self.root / "endpoint_adjacency.json"
        adjacency_path.write_text(
            json.dumps({
                "keys": ["A", "B", "C"],
                "neighbour_type": "shared_edge",
                "neighbours": [[1], [0], []],
            }),
            encoding="utf-8",
        )
        dataset = Dataset(data_path)
        original_smooth_sector_rows = uk_map_query.smooth_sector_rows

        with patch.object(
            uk_map_query,
            "smooth_sector_rows",
            side_effect=lambda rows, level: original_smooth_sector_rows(rows, level, adjacency_path=str(adjacency_path)),
        ):
            result = summary(dataset, self.request(level="sector", smoothingLevel=1))

        rows_by_key = {row["key"]: row for row in result["rows"]}
        self.assertEqual(result["smoothing"]["matched_rows"], 1)
        self.assertEqual(result["smoothing"]["target_rows"], 3)
        self.assertEqual(rows_by_key["A"]["value"], 100)
        self.assertEqual(rows_by_key["A"]["raw_value"], 100)
        self.assertEqual(rows_by_key["B"]["value"], 100)
        self.assertEqual(rows_by_key["B"]["row_count"], 0)
        self.assertIsNone(rows_by_key["B"]["raw_value"])
        self.assertEqual(rows_by_key["B"]["smoothing_contributing_sectors"], 1)
        self.assertIsNone(rows_by_key["C"]["value"])
        self.assertEqual(rows_by_key["C"]["row_count"], 0)
        self.assertIsNone(rows_by_key["C"]["raw_value"])
        self.assertEqual(rows_by_key["C"]["smoothing_contributing_sectors"], 0)

    def test_sector_adjacency_sidecar_matches_geojson_keys(self) -> None:
        geojson_path = DEFAULT_SECTOR_ADJACENCY_PATH.with_name("sectors_MappaR.geojson")
        geojson = json.loads(geojson_path.read_text(encoding="utf-8"))
        geojson_keys = [feature["properties"]["PostcodeSector"] for feature in geojson["features"]]
        adjacency = load_sector_adjacency()

        self.assertEqual(list(adjacency.keys), geojson_keys)
        self.assertEqual(len(adjacency.neighbours), len(geojson_keys))
        for index, neighbours in enumerate(adjacency.neighbours):
            self.assertEqual(list(neighbours), sorted(neighbours))
            self.assertEqual(len(neighbours), len(set(neighbours)))
            self.assertNotIn(index, neighbours)
            self.assertTrue(all(0 <= neighbour < len(geojson_keys) for neighbour in neighbours))

    def test_custom_postcode_column_default(self) -> None:
        dataset = Dataset(self.data_path)
        result = summary(
            dataset,
            self.request(),
            defaults={"postcode_area": "CustomArea", "postcode_sector": "PostcodeSector"},
        )

        self.assertEqual(result["join_column"], "CustomArea")
        self.assertEqual([row["key"] for row in result["rows"]], ["A1", "A2", "A3"])

    def test_uppercase_postcode_aliases_are_used_as_defaults(self) -> None:
        dataset = Dataset(self.upper_data_path)

        area_result = summary(dataset, self.request())
        sector_result = summary(dataset, self.request(level="sector"))
        unit_result = summary(dataset, self.request(level="unit"))

        self.assertEqual(area_result["join_column"], "POSTCODE_AREA")
        self.assertEqual([row["key"] for row in area_result["rows"]], ["AB", "AL"])
        self.assertEqual(sector_result["join_column"], "POSTCODE_SECTOR")
        self.assertEqual([row["key"] for row in sector_result["rows"]], ["AB10 1", "AL1 1"])
        self.assertEqual(unit_result["join_column"], "POSTCODE_UNIT")
        self.assertEqual(
            [(row["key"], row["latitude"], row["longitude"]) for row in unit_result["rows"]],
            [("AB10 1AA", 57.2, -2.2), ("AL1 1AA", 51.7, -0.4)],
        )

    def test_coordinate_aliases_accept_lower_latitude_and_mixed_case_longitude(self) -> None:
        dataset = Dataset(self.mixed_coordinate_path)
        result = summary(dataset, self.request(level="unit"))

        self.assertEqual(result["join_column"], "PostcodeUnit")
        self.assertEqual(
            [(row["key"], row["latitude"], row["longitude"]) for row in result["rows"]],
            [("AB10 1AA", 57.1, -2.1), ("AL1 1AA", 51.7, -0.4)],
        )

    def test_invalid_postcode_column_is_reported(self) -> None:
        dataset = Dataset(self.data_path)

        with self.assertRaisesRegex(ValueError, "Choose a valid area postcode column"):
            summary(dataset, self.request(areaColumn="MissingArea"))

    def test_unit_summary_uses_average_row_value_and_coordinates(self) -> None:
        dataset = Dataset(self.data_path)
        result = summary(dataset, self.request(level="unit"))

        self.assertEqual(result["level"], "unit")
        self.assertEqual(result["join_column"], "PostcodeUnit")
        self.assertEqual(result["join_property"], "PostcodeUnit")
        self.assertEqual(result["point_summary"], {
            "summary_count": 4,
            "plotted_count": 3,
            "missing_value_count": 0,
            "missing_coordinate_count": 1,
        })
        self.assertEqual(
            [(row["key"], row["value"], row["denominator"], row["latitude"], row["longitude"]) for row in result["rows"]],
            [
                ("AB10 1AA", 150, 2, 57.2, -2.2),
                ("AL1 1AA", 300, 1, 51.7, -0.4),
                ("AL1 2AA", 400, 1, 51.8, -0.3),
            ],
        )

    def test_compact_unit_summary_returns_aligned_point_arrays(self) -> None:
        app = create_app(self.data_path, token="", tools=["uk_map"], use_saved_filters=False, use_kpis=False)

        status, _, body = asgi_post_json(app, "/api/uk-map/summary", self.request(level="unit", compactUnitPoints=True))
        payload = json.loads(body)
        points = payload["unit_points"]

        self.assertEqual(status, 200)
        self.assertEqual(payload["level"], "unit")
        self.assertNotIn("rows", payload)
        self.assertEqual(payload["point_summary"], {
            "summary_count": 4,
            "plotted_count": 3,
            "missing_value_count": 0,
            "missing_coordinate_count": 1,
        })
        self.assertEqual(points["key"], ["AB10 1AA", "AL1 1AA", "AL1 2AA"])
        self.assertEqual(points["row_count"], [2, 1, 1])
        self.assertEqual(points["numerator"], [300, 300, 400])
        self.assertEqual(points["denominator"], [2, 1, 1])
        self.assertEqual(points["volume"], [2, 1, 1])
        self.assertEqual(points["value"], [150, 300, 400])
        self.assertEqual(points["latitude"], [57.2, 51.7, 51.8])
        self.assertEqual(points["longitude"], [-2.2, -0.4, -0.3])
        lengths = {len(values) for values in points.values()}
        self.assertEqual(lengths, {3})
        self.assertIsInstance(payload["timings"]["duckdb_ns"], int)
        self.assertGreaterEqual(payload["timings"]["duckdb_ns"], 0)

    def test_unit_summary_applies_filter_and_weight(self) -> None:
        dataset = Dataset(self.data_path)
        result = summary(
            dataset,
            self.request(level="unit", denominator="Weight", filter="PostcodeArea = 'AL'"),
        )

        self.assertEqual(result["filtered_row_count"], 2)
        self.assertEqual(result["point_summary"], {
            "summary_count": 2,
            "plotted_count": 1,
            "missing_value_count": 1,
            "missing_coordinate_count": 0,
        })
        self.assertEqual(
            [(row["key"], row["value"], row["denominator"], row["latitude"], row["longitude"]) for row in result["rows"]],
            [("AL1 1AA", 10, 30, 51.7, -0.4)],
        )
        self.assertIn("1 row excluded from Weight because Weight was missing.", result["warnings"])

    def test_custom_unit_point_column_defaults(self) -> None:
        dataset = Dataset(self.data_path)
        result = summary(
            dataset,
            self.request(level="unit"),
            defaults={
                "postcode_unit": "CustomUnit",
                "latitude": "CustomLat",
                "longitude": "CustomLong",
            },
        )

        self.assertEqual(result["join_column"], "CustomUnit")
        self.assertEqual([row["key"] for row in result["rows"]], ["U1", "U2", "U3", "U4"])
        self.assertEqual(result["rows"][0]["latitude"], 58.2)
        self.assertEqual(result["rows"][0]["longitude"], -3.2)

    def test_invalid_unit_point_columns_are_reported(self) -> None:
        dataset = Dataset(self.data_path)

        with self.assertRaisesRegex(ValueError, "Choose a valid unit postcode column"):
            summary(dataset, self.request(level="unit", unitColumn="MissingUnit"))

        with self.assertRaisesRegex(ValueError, "Choose a valid numeric latitude column"):
            summary(dataset, self.request(level="unit", latitudeColumn="PostcodeArea"))

        with self.assertRaisesRegex(ValueError, "Choose a valid numeric longitude column"):
            summary(dataset, self.request(level="unit", longitudeColumn="PostcodeArea"))


if __name__ == "__main__":
    unittest.main()
