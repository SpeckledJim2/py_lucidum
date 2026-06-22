from __future__ import annotations

import asyncio
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import duckdb

from py_lucidum.app import create_app
from py_lucidum.core import Dataset, sql_literal
from py_lucidum.demo import demo_dataset_path


def asgi_get_json(app: Any, path: str) -> tuple[int, dict[str, Any]]:
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
    body = b"".join(message.get("body", b"") for message in messages if message["type"] == "http.response.body")
    return start["status"], json.loads(body)


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
    body = b"".join(message.get("body", b"") for message in messages if message["type"] == "http.response.body")
    return start["status"], json.loads(body)


def copy_query_to_parquet(con: duckdb.DuckDBPyConnection, query: str, path: Path) -> None:
    con.execute(f"COPY ({query}) TO {sql_literal(str(path))} (FORMAT PARQUET)")


def write_demo_monthly_folder(root: Path) -> Path:
    source_path = demo_dataset_path()
    monthly = root / "monthly"
    monthly.mkdir()
    con = duckdb.connect(database=":memory:")
    try:
        source_sql = f"read_parquet({sql_literal(str(source_path))})"
        months = [
            str(row[0])
            for row in con.execute(
                f"""
SELECT DISTINCT strftime(QUOTE_DATE, '%Y-%m') AS month
FROM {source_sql}
ORDER BY month
"""
            ).fetchall()
        ]
        for month in months:
            copy_query_to_parquet(
                con,
                f"""
SELECT *
FROM {source_sql}
WHERE strftime(QUOTE_DATE, '%Y-%m') = {sql_literal(month)}
""",
                monthly / f"{month}.parquet",
            )
    finally:
        con.close()
    return monthly


def demo_schema_columns() -> list[str]:
    con = duckdb.connect(database=":memory:")
    try:
        rows = con.execute(
            f"DESCRIBE SELECT * FROM read_parquet({sql_literal(str(demo_dataset_path()))})"
        ).fetchall()
    finally:
        con.close()
    return [str(row[0]) for row in rows]


class ParquetFolderTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def test_monthly_parquet_folder_schema_and_default_tools(self) -> None:
        folder = write_demo_monthly_folder(self.root)
        app = create_app(folder, token="", use_saved_filters=False, use_kpis=False, use_features=False)
        filter_sql = "QUOTE_DATE >= DATE '2017-01-01'"

        status, schema = asgi_get_json(app, "/api/schema")
        self.assertEqual(status, 200)
        self.assertEqual(schema["source_kind"], "parquet_folder")
        self.assertEqual(Path(schema["path"]).name, "monthly")
        self.assertEqual(schema["file_count"], 7)
        self.assertEqual(schema["file_size"], sum(path.stat().st_size for path in folder.glob("*.parquet")))
        self.assertEqual(schema["row_count"], 50_000)
        self.assertEqual([column["name"] for column in schema["columns"]], demo_schema_columns())
        self.assertEqual(schema["data_sources"][0]["label"], "monthly")
        self.assertEqual(schema["data_sources"][0]["source_kind"], "parquet_folder")

        status, row_count = asgi_post_json(app, "/api/filter/row-count", {"filter": filter_sql})
        self.assertEqual(status, 200)
        self.assertEqual(row_count["row_count"], 50_000)
        self.assertGreater(row_count["filtered_row_count"], 0)
        self.assertLess(row_count["filtered_row_count"], 50_000)

        status, viewer = asgi_post_json(app, "/api/dataset-viewer/table", {"filter": filter_sql, "limit": 25})
        self.assertEqual(status, 200)
        self.assertEqual(viewer["displayed_row_count"], 25)
        self.assertIn("QUOTE_DATE", [column["name"] for column in viewer["columns"]])

        status, profile = asgi_post_json(app, "/api/column-profile/summary", {"filter": filter_sql})
        self.assertEqual(status, 200)
        self.assertEqual(profile["row_count"], 50_000)
        self.assertEqual(profile["filtered_row_count"], row_count["filtered_row_count"])
        self.assertEqual(len(profile["columns"]), len(schema["columns"]))

        chart_payload = {
            "x": "DRIVER_AGE",
            "responses": [{"label": "PREMIUM", "numerator": "PREMIUM"}],
            "denominator": "ANNUAL_MILEAGE",
            "filter": filter_sql,
        }
        status, chart = asgi_post_json(app, "/api/line-bar/chart", chart_payload)
        self.assertEqual(status, 200)
        self.assertEqual(chart["row_count"], 50_000)
        self.assertGreater(len(chart["rows"]), 0)

        status, histogram = asgi_post_json(
            app,
            "/api/histogram/chart",
            {"actual": "PREMIUM", "denominator": "__none__", "filter": filter_sql},
        )
        self.assertEqual(status, 200)
        self.assertEqual(histogram["row_count"], 50_000)
        self.assertGreater(histogram["valid_count"], 0)

        status, uk_map = asgi_post_json(
            app,
            "/api/uk-map/summary",
            {"level": "area", "actual": "PREMIUM", "denominator": "__none__", "filter": filter_sql},
        )
        self.assertEqual(status, 200)
        self.assertEqual(uk_map["row_count"], 50_000)
        self.assertGreater(len(uk_map["rows"]), 0)

    def test_folder_input_requires_direct_child_parquet_files(self) -> None:
        folder = self.root / "empty"
        nested = folder / "nested"
        nested.mkdir(parents=True)
        con = duckdb.connect(database=":memory:")
        try:
            copy_query_to_parquet(con, "SELECT 1 AS x", nested / "ignored.parquet")
        finally:
            con.close()
        (folder / "notes.txt").write_text("not parquet", encoding="utf-8")

        with self.assertRaisesRegex(ValueError, "contains no direct .parquet files"):
            Dataset(folder)

    def test_folder_input_ignores_nested_parquet_files(self) -> None:
        folder = self.root / "direct-only"
        nested = folder / "nested"
        nested.mkdir(parents=True)
        con = duckdb.connect(database=":memory:")
        try:
            copy_query_to_parquet(con, "SELECT 1 AS x", folder / "a.parquet")
            copy_query_to_parquet(con, "SELECT 'bad' AS x, 2 AS y", nested / "ignored.parquet")
        finally:
            con.close()

        dataset = Dataset(folder)

        self.assertEqual(dataset.row_count(), 1)
        self.assertEqual([column.name for column in dataset.valid_schema_columns()], ["x"])

    def test_folder_input_rejects_missing_or_extra_columns(self) -> None:
        folder = self.root / "bad-columns"
        folder.mkdir()
        con = duckdb.connect(database=":memory:")
        try:
            copy_query_to_parquet(con, "SELECT 1 AS x, 2 AS y", folder / "a.parquet")
            copy_query_to_parquet(con, "SELECT 1 AS x, 3 AS z", folder / "b.parquet")
        finally:
            con.close()

        with self.assertRaisesRegex(ValueError, "missing columns: y"):
            Dataset(folder)

    def test_folder_input_rejects_type_mismatches(self) -> None:
        folder = self.root / "bad-types"
        folder.mkdir()
        con = duckdb.connect(database=":memory:")
        try:
            copy_query_to_parquet(con, "SELECT 1::INTEGER AS x", folder / "a.parquet")
            copy_query_to_parquet(con, "SELECT '1'::VARCHAR AS x", folder / "b.parquet")
        finally:
            con.close()

        with self.assertRaisesRegex(ValueError, "type mismatches: x expected INTEGER got VARCHAR"):
            Dataset(folder)

    def test_folder_input_is_rejected_with_modelling_tools(self) -> None:
        folder = self.root / "valid-folder"
        folder.mkdir()
        con = duckdb.connect(database=":memory:")
        try:
            copy_query_to_parquet(con, "SELECT 1 AS x, 2 AS y", folder / "a.parquet")
        finally:
            con.close()

        for tools in (["glm", "line_bar"], ["gbm", "line_bar"], "all"):
            with self.subTest(tools=tools):
                with self.assertRaisesRegex(ValueError, "not supported with GLM or GBM"):
                    create_app(folder, token="", tools=tools, use_saved_filters=False, use_kpis=False)


if __name__ == "__main__":
    unittest.main()
