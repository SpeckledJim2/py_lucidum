from __future__ import annotations

import asyncio
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any
from unittest.mock import patch

import duckdb

from py_lucidum import smooth_postcode_sectors
from py_lucidum.app import create_app, normalise_tools
from py_lucidum.core import Dataset, sql_literal
from py_lucidum.tools.uk_map.export import SECTOR_SMOOTHING_OUTPUT_COLUMNS
from py_lucidum.tools.uk_map import query as uk_map_query
from py_lucidum.tools.uk_map.query import summary
from py_lucidum.tools.uk_map.smoothing import (
    DEFAULT_SECTOR_ADJACENCY_PATH,
    DEFAULT_SECTOR_SMOOTHING_POOLS_PATH,
    MAX_SMOOTHING_LEVEL,
    build_smoothed_sector_sql,
    load_sector_adjacency,
    normalise_smoothing_level,
    sector_smoothing_pools,
    smooth_sector_rows,
    write_sector_smoothing_parquet,
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


def write_smoothing_pool_parquet(adjacency_path: Path, output_path: Path) -> None:
    adjacency = load_sector_adjacency(str(adjacency_path))
    rows: list[tuple[int, str, str]] = []
    for level in range(1, MAX_SMOOTHING_LEVEL + 1):
        pools = sector_smoothing_pools(str(adjacency_path), level)
        for target_index, pool in enumerate(pools):
            target_key = adjacency.keys[target_index]
            for pool_index in pool:
                rows.append((level, target_key, adjacency.keys[pool_index]))

    con = duckdb.connect(database=":memory:")
    try:
        con.execute("CREATE TABLE pools(level INTEGER, target_key VARCHAR, pool_key VARCHAR)")
        con.executemany("INSERT INTO pools VALUES (?, ?, ?)", rows)
        con.execute(f"COPY pools TO {sql_literal(str(output_path))} (FORMAT PARQUET, COMPRESSION ZSTD)")
    finally:
        con.close()


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

    def parquet_source(self) -> Path:
        path = self.root / "sample.parquet"
        if path.exists():
            return path
        con = duckdb.connect(database=":memory:")
        try:
            con.execute(
                f"COPY (SELECT * FROM read_csv_auto({sql_literal(str(self.data_path))}, header=true)) "
                f"TO {sql_literal(str(path))} (FORMAT PARQUET)"
            )
        finally:
            con.close()
        return path

    def test_default_tools_include_uk_map(self) -> None:
        self.assertEqual(normalise_tools(None), ["line_bar", "dataset_viewer", "column_profile", "histogram", "uk_map", "specs"])

        app = create_app(self.data_path, token="dev-token")
        paths = {route.path for route in app.routes}

        self.assertEqual(app.state.enabled_tools, ["line_bar", "dataset_viewer", "column_profile", "histogram", "uk_map", "specs"])
        self.assertIn("/api/dataset-viewer/table", paths)
        self.assertIn("/api/column-profile/summary", paths)
        self.assertIn("/api/histogram/chart", paths)
        self.assertIn("/api/uk-map/summary", paths)
        self.assertIn("/api/uk-map/sector-smoothing", paths)
        self.assertIn("/tools/uk-map/static", paths)

    def test_public_sector_smoothing_writes_all_levels_and_matches_map(self) -> None:
        source_path = self.parquet_source()
        output_path = self.root / "outputs" / "smoothed.parquet"
        result_path = smooth_postcode_sectors(
            source_path,
            output_path,
            postcode_sector="PostcodeSector",
            numerator="Actual",
            denominator="Weight",
            filter="Flag = 1",
        )

        self.assertEqual(result_path, output_path.resolve())
        self.assertTrue(result_path.is_file())
        con = duckdb.connect(database=":memory:")
        try:
            cursor = con.execute(f"SELECT * FROM read_parquet({sql_literal(str(result_path))})")
            columns = [item[0] for item in cursor.description]
            output_rows = {
                row[0]: dict(zip(columns, row))
                for row in cursor.fetchall()
            }
        finally:
            con.close()

        self.assertEqual(columns, list(SECTOR_SMOOTHING_OUTPUT_COLUMNS))
        self.assertGreaterEqual(len(output_rows), len(load_sector_adjacency().keys))
        self.assertEqual(output_rows["AB10 1"]["numerator_sum"], 300)
        self.assertEqual(output_rows["AB10 1"]["denominator_sum"], 30)
        self.assertEqual(output_rows["AB10 1"]["unsmoothed"], 10)

        dataset = Dataset(source_path)
        self.addCleanup(dataset.con.close)
        for level in range(1, MAX_SMOOTHING_LEVEL + 1):
            with self.subTest(level=level):
                map_result = summary(
                    dataset,
                    self.request(
                        level="sector",
                        denominator="Weight",
                        filter="Flag = 1",
                        smoothingLevel=level,
                    ),
                )
                map_rows = {row["key"]: row for row in map_result["rows"]}
                for key in ("AB10 1", "AL1 1", "AL1 2"):
                    self.assertEqual(output_rows[key][f"smooth_n{level}"], map_rows[key]["value"])
                    self.assertEqual(
                        output_rows[key][f"numerator_n{level}"],
                        map_rows[key]["numerator"],
                    )
                    self.assertEqual(
                        output_rows[key][f"denominator_n{level}"],
                        map_rows[key]["denominator"],
                    )

    def test_public_sector_smoothing_uses_row_count_without_denominator(self) -> None:
        output_path = self.root / "average.parquet"
        smooth_postcode_sectors(
            self.parquet_source(),
            output_path,
            postcode_sector="PostcodeSector",
            numerator="Actual",
            filter="PostcodeArea = 'AB'",
        )

        con = duckdb.connect(database=":memory:")
        try:
            cursor = con.execute(
                f"SELECT * "
                f"FROM read_parquet({sql_literal(str(output_path))}) "
                "WHERE postcode_sector = 'AB10 1'"
            )
            row = dict(zip((item[0] for item in cursor.description), cursor.fetchone()))
        finally:
            con.close()
        self.assertEqual(
            (row["numerator_sum"], row["denominator_sum"], row["unsmoothed"]),
            (300, 2, 150),
        )

        dataset = Dataset(self.parquet_source())
        self.addCleanup(dataset.con.close)
        for level in range(1, MAX_SMOOTHING_LEVEL + 1):
            with self.subTest(level=level):
                map_result = summary(
                    dataset,
                    self.request(
                        level="sector",
                        filter="PostcodeArea = 'AB'",
                        smoothingLevel=level,
                    ),
                )
                map_row = {item["key"]: item for item in map_result["rows"]}["AB10 1"]
                self.assertEqual(row[f"numerator_n{level}"], map_row["numerator"])
                self.assertEqual(row[f"denominator_n{level}"], map_row["denominator"])
                self.assertEqual(row[f"smooth_n{level}"], map_row["value"])

    def test_public_sector_smoothing_ignores_blanks_and_keeps_unknown_valid_sector(self) -> None:
        source_path = self.root / "unknown_sector.parquet"
        con = duckdb.connect(database=":memory:")
        try:
            con.execute(
                "CREATE TABLE sectors(PostcodeSector VARCHAR, Actual DOUBLE, Weight DOUBLE)"
            )
            con.execute(
                "INSERT INTO sectors VALUES "
                "('ZZ99 9', 30, 3), ('ZZ99 9', NULL, 4), ('', 50, 5), (NULL, 70, 7)"
            )
            con.execute(f"COPY sectors TO {sql_literal(str(source_path))} (FORMAT PARQUET)")
        finally:
            con.close()

        output_path = self.root / "unknown_sector_output.parquet"
        smooth_postcode_sectors(
            source_path,
            output_path,
            postcode_sector="PostcodeSector",
            numerator="Actual",
            denominator="Weight",
        )
        con = duckdb.connect(database=":memory:")
        try:
            unknown = con.execute(
                f"SELECT * FROM read_parquet({sql_literal(str(output_path))}) "
                "WHERE postcode_sector = 'ZZ99 9'"
            ).fetchone()
            blank_count = con.execute(
                f"SELECT COUNT(*) FROM read_parquet({sql_literal(str(output_path))}) "
                "WHERE postcode_sector IS NULL OR postcode_sector = ''"
            ).fetchone()[0]
            no_contributor = con.execute(
                f"SELECT unsmoothed, smooth_n1, numerator_n1, denominator_n1, "
                "smooth_n5, numerator_n5, denominator_n5 "
                f"FROM read_parquet({sql_literal(str(output_path))}) "
                "WHERE postcode_sector = 'AB10 1'"
            ).fetchone()
        finally:
            con.close()

        self.assertEqual(
            unknown,
            (
                "ZZ99 9",
                30,
                3,
                10,
                10,
                10,
                10,
                10,
                10,
                30,
                30,
                30,
                30,
                30,
                3,
                3,
                3,
                3,
                3,
            ),
        )
        self.assertEqual(blank_count, 0)
        self.assertEqual(no_contributor, (None, None, None, None, None, None, None))

    def test_public_sector_smoothing_validates_format_and_preserves_output(self) -> None:
        source_path = self.root / "invalid_sector.parquet"
        con = duckdb.connect(database=":memory:")
        try:
            con.execute(
                "CREATE TABLE invalid_sector(PostcodeSector VARCHAR, Actual DOUBLE, Keep INTEGER)"
            )
            con.execute("INSERT INTO invalid_sector VALUES ('ab10 1', 10, 1), ('AB10 1', 20, 0)")
            con.execute(
                f"COPY invalid_sector TO {sql_literal(str(source_path))} (FORMAT PARQUET)"
            )
        finally:
            con.close()
        output_path = self.root / "existing.parquet"
        output_path.write_bytes(b"original")

        with self.assertRaisesRegex(ValueError, "uppercase with one space"):
            smooth_postcode_sectors(
                source_path,
                output_path,
                postcode_sector="PostcodeSector",
                numerator="Actual",
            )
        self.assertEqual(output_path.read_bytes(), b"original")

        smooth_postcode_sectors(
            source_path,
            output_path,
            postcode_sector="PostcodeSector",
            numerator="Actual",
            filter="Keep = 0",
        )
        self.assertNotEqual(output_path.read_bytes(), b"original")

    def test_public_sector_smoothing_rejects_invalid_inputs(self) -> None:
        source_path = self.parquet_source()
        with self.assertRaisesRegex(ValueError, "one Parquet file"):
            smooth_postcode_sectors(
                self.root / "missing.parquet",
                self.root / "missing-output.parquet",
                postcode_sector="PostcodeSector",
                numerator="Actual",
            )
        with self.assertRaisesRegex(ValueError, "must not overwrite"):
            smooth_postcode_sectors(
                source_path,
                source_path,
                postcode_sector="PostcodeSector",
                numerator="Actual",
            )
        with self.assertRaisesRegex(ValueError, "numeric numerator"):
            smooth_postcode_sectors(
                source_path,
                self.root / "bad-numerator.parquet",
                postcode_sector="PostcodeSector",
                numerator="PostcodeArea",
            )
        with self.assertRaisesRegex(ValueError, "numeric denominator"):
            smooth_postcode_sectors(
                source_path,
                self.root / "bad-denominator.parquet",
                postcode_sector="PostcodeSector",
                numerator="Actual",
                denominator="PostcodeArea",
            )
        with self.assertRaisesRegex(ValueError, "valid postcode-sector"):
            smooth_postcode_sectors(
                source_path,
                self.root / "bad-sector-column.parquet",
                postcode_sector="MissingSector",
                numerator="Actual",
            )
        with self.assertRaisesRegex(ValueError, "Invalid filter"):
            smooth_postcode_sectors(
                source_path,
                self.root / "bad-filter.parquet",
                postcode_sector="PostcodeSector",
                numerator="Actual",
                filter="MissingColumn = 1",
            )
        with self.assertRaisesRegex(ValueError, r"\.parquet filename"):
            smooth_postcode_sectors(
                source_path,
                self.root / "not-parquet.csv",
                postcode_sector="PostcodeSector",
                numerator="Actual",
            )

    def test_sector_smoothing_writer_preserves_existing_output_on_calculation_failure(self) -> None:
        output_path = self.root / "calculation-failure.parquet"
        output_path.write_bytes(b"original")
        con = duckdb.connect(database=":memory:")
        try:
            with self.assertRaises(duckdb.Error):
                write_sector_smoothing_parquet(
                    con,
                    "SELECT * FROM missing_smoothing_source",
                    output_path,
                )
        finally:
            con.close()
        self.assertEqual(output_path.read_bytes(), b"original")
        self.assertEqual(list(self.root.glob(".calculation-failure.tmp-*.parquet")), [])

    def test_interactive_smoothing_sql_reads_only_selected_level(self) -> None:
        sql = build_smoothed_sector_sql(
            "SELECT 'AB10 1' AS key, 1 AS row_count, 10.0 AS resp0_num, "
            "2.0 AS resp0_den, 5.0 AS resp0",
            3,
        )

        self.assertIn("WHERE level IN (3)", sql)
        self.assertNotIn("WHERE level IN (1, 2, 3, 4, 5)", sql)

    def test_sector_smoothing_endpoint_saves_and_replaces_deterministic_sidecar(self) -> None:
        app = create_app(self.data_path, token="", tools=["uk_map"], use_saved_filters=False, use_kpis=False)
        request = self.request(level="sector", filter="Flag = 1")

        status, _, body = asgi_post_json(app, "/api/uk-map/sector-smoothing", request)
        first = json.loads(body)
        first_path = Path(first["path"])

        self.assertEqual(status, 200)
        self.assertFalse(first["replaced"])
        self.assertEqual(first["columns"], list(SECTOR_SMOOTHING_OUTPUT_COLUMNS))
        self.assertTrue(first_path.is_file())
        self.assertIn(".lucidum/datasets/sample.csv/", first_path.as_posix())
        self.assertIn("/uk_map/sector_smoothing/", first_path.as_posix())

        status, _, body = asgi_post_json(app, "/api/uk-map/sector-smoothing", request)
        second = json.loads(body)
        self.assertEqual(status, 200)
        self.assertTrue(second["replaced"])
        self.assertEqual(second["path"], first["path"])
        self.assertEqual(second["row_count"], first["row_count"])

        changed_request = {**request, "filter": "Flag = 0"}
        status, _, body = asgi_post_json(app, "/api/uk-map/sector-smoothing", changed_request)
        changed = json.loads(body)
        self.assertEqual(status, 200)
        self.assertFalse(changed["replaced"])
        self.assertNotEqual(changed["path"], first["path"])

        status, _, body = asgi_post_json(
            app,
            "/api/uk-map/sector-smoothing",
            self.request(level="area"),
        )
        self.assertEqual(status, 400)
        self.assertIn("only for the Sector map level", json.loads(body)["detail"])

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

    def test_sector_smoothing_endpoint_includes_duckdb_timing(self) -> None:
        app = create_app(self.data_path, token="", tools=["uk_map"], use_saved_filters=False, use_kpis=False)

        status, _, body = asgi_post_json(app, "/api/uk-map/summary", self.request(level="sector", smoothingLevel=1))
        payload = json.loads(body)

        self.assertEqual(status, 200)
        self.assertTrue(payload["smoothing"]["applied"])
        self.assertGreater(payload["timings"]["duckdb_ns"], 0)

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
        self.assertIn("1 row excluded due to missing Weight", result["warnings"])

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
        pool_path = self.root / "endpoint_smoothing_pools.parquet"
        write_smoothing_pool_parquet(adjacency_path, pool_path)
        original_build_smoothed_sector_sql = uk_map_query.build_smoothed_sector_sql

        with patch.object(
            uk_map_query,
            "build_smoothed_sector_sql",
            side_effect=lambda raw_sql, level: original_build_smoothed_sector_sql(raw_sql, level, pool_path=str(pool_path)),
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

    def test_sector_duckdb_smoothing_matches_python_reference_for_all_levels(self) -> None:
        dataset = Dataset(self.data_path)
        raw_rows = uk_map_query.map_rows(
            dataset,
            "PostcodeSector",
            {"label": "Actual", "numerator": "Actual"},
            {"column": None, "label": "Average row value", "bar_label": "Row count"},
        )

        for level in range(1, MAX_SMOOTHING_LEVEL + 1):
            with self.subTest(level=level):
                expected_rows, expected_metadata, expected_warning = smooth_sector_rows(raw_rows, level)
                result = summary(dataset, self.request(level="sector", smoothingLevel=level))
                actual_metadata = dict(result["smoothing"])
                actual_metadata.pop("requested_level", None)

                self.assertIsNone(expected_warning)
                self.assertEqual(actual_metadata, expected_metadata)
                self.assertEqual(
                    {row["key"]: row for row in result["rows"]},
                    {row["key"]: row for row in expected_rows},
                )

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

    def test_sector_smoothing_parquet_matches_adjacency_pools(self) -> None:
        adjacency = load_sector_adjacency()
        con = duckdb.connect(database=":memory:")
        try:
            pool_sql = f"read_parquet({sql_literal(str(DEFAULT_SECTOR_SMOOTHING_POOLS_PATH))})"
            columns = con.execute(f"DESCRIBE SELECT * FROM {pool_sql}").fetchall()
            rows = con.execute(f"""
                SELECT level, COUNT(*), COUNT(DISTINCT target_key), COUNT(DISTINCT pool_key)
                FROM {pool_sql}
                GROUP BY level
                ORDER BY level
            """).fetchall()
            key_rows = con.execute(f"""
                SELECT target_key FROM {pool_sql}
                UNION
                SELECT pool_key FROM {pool_sql}
            """).fetchall()
        finally:
            con.close()

        self.assertEqual([column[0] for column in columns], ["level", "target_key", "pool_key"])
        self.assertEqual([row[0] for row in rows], list(range(1, MAX_SMOOTHING_LEVEL + 1)))
        self.assertEqual({row[0] for row in key_rows}, set(adjacency.keys))
        for level, pair_count, target_count, pool_key_count in rows:
            self.assertEqual(pair_count, sum(len(pool) for pool in sector_smoothing_pools(str(DEFAULT_SECTOR_ADJACENCY_PATH), level)))
            self.assertEqual(target_count, len(adjacency.keys))
            self.assertEqual(pool_key_count, len(adjacency.keys))

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

    def test_unit_viewport_bounds_filter_rows_before_aggregation(self) -> None:
        dataset = Dataset(self.data_path)
        bounds = {"south": 51.6, "west": -0.5, "north": 51.9, "east": -0.1}

        result = summary(
            dataset,
            self.request(level="unit", filter="Flag = 1", unitViewportBounds=bounds),
        )

        self.assertEqual(result["filtered_row_count"], 1)
        self.assertEqual(result["unit_viewport"], {"applied": True, "bounds": bounds})
        self.assertEqual(
            [(row["key"], row["value"], row["latitude"], row["longitude"]) for row in result["rows"]],
            [("AL1 1AA", 300, 51.7, -0.4)],
        )

    def test_unit_viewport_bounds_are_validated(self) -> None:
        dataset = Dataset(self.data_path)
        invalid_bounds = [
            [],
            "51,-1,52,1",
            {"south": 51, "west": -1, "north": 51, "east": 1},
            {"south": -91, "west": -1, "north": 51, "east": 1},
            {"south": 50, "west": 2, "north": 51, "east": 1},
            {"south": 50, "west": -1, "north": float("nan"), "east": 1},
            {"south": 50, "west": -1, "north": float("inf"), "east": 1},
        ]
        for bounds in invalid_bounds:
            with self.subTest(bounds=bounds):
                with self.assertRaisesRegex(ValueError, "Choose valid unit viewport bounds"):
                    summary(dataset, self.request(level="unit", unitViewportBounds=bounds))

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
        self.assertEqual(payload["unit_geometry"], {"included": True, "point_count": 3})
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

    def test_compact_unit_summary_can_reuse_stable_geometry(self) -> None:
        app = create_app(self.data_path, token="", tools=["uk_map"], use_saved_filters=False, use_kpis=False)
        request = self.request(
            level="unit",
            denominator="Weight",
            filter="PostcodeArea = 'AL'",
            compactUnitPoints=True,
            compactUnitMetrics="minimal",
        )

        status, _, body = asgi_post_json(app, "/api/uk-map/summary", request)
        initial = json.loads(body)
        points = initial["unit_points"]

        self.assertEqual(status, 200)
        self.assertEqual(initial["unit_geometry"], {"included": True, "point_count": 2})
        self.assertEqual(points["key"], ["AL1 1AA", "AL1 2AA"])
        self.assertEqual(points["row_count"], [1, 1])
        self.assertEqual(points["latitude"], [51.7, 51.8])
        self.assertEqual(points["longitude"], [-0.4, -0.3])
        self.assertEqual(points["numerator"], [300, None])
        self.assertEqual(points["denominator"], [30, 0])
        self.assertEqual(points["value"], [10, None])
        self.assertEqual(initial["point_summary"], {
            "summary_count": 2,
            "plotted_count": 1,
            "missing_value_count": 1,
            "missing_coordinate_count": 0,
        })

        request["reuseUnitGeometry"] = True
        status, _, body = asgi_post_json(app, "/api/uk-map/summary", request)
        reused = json.loads(body)

        self.assertEqual(status, 200)
        self.assertEqual(reused["unit_geometry"], {"included": False, "point_count": 2})
        self.assertEqual(
            reused["unit_points"],
            {"numerator": [300, None], "denominator": [30, 0], "value": [10, None]},
        )

    def test_unit_viewport_bounds_filter_rows_before_grouping(self) -> None:
        dataset = Dataset(self.data_path)
        bounds = {"south": 57.15, "west": -2.4, "north": 57.4, "east": -2.0}

        result = summary(
            dataset,
            self.request(
                level="unit",
                compactUnitPoints=True,
                compactUnitMetrics="minimal",
                unitViewportBounds=bounds,
            ),
        )

        self.assertEqual(result["row_count"], 5)
        self.assertEqual(result["filtered_row_count"], 1)
        self.assertEqual(result["filter"], "")
        self.assertEqual(result["unit_viewport"], {"applied": True, "bounds": bounds})
        self.assertEqual(result["unit_geometry"], {"included": True, "point_count": 1})
        self.assertEqual(result["unit_points"]["key"], ["AB10 1AA"])
        self.assertEqual(result["unit_points"]["latitude"], [57.3])
        self.assertEqual(result["unit_points"]["longitude"], [-2.3])
        self.assertEqual(result["unit_points"]["denominator"], [1])
        self.assertNotIn("numerator", result["unit_points"])
        self.assertEqual(result["unit_points"]["value"], [200])

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
        self.assertIn("1 row excluded due to missing Weight", result["warnings"])

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
