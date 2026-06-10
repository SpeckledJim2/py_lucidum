from __future__ import annotations

import asyncio
import json
import os
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

from py_lucidum.app import create_app
from py_lucidum.tools.registry import normalise_tools


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
    body = b"".join(message.get("body", b"") for message in messages if message["type"] == "http.response.body")
    headers = {key.decode("latin-1").lower(): value.decode("latin-1") for key, value in start["headers"]}
    return start["status"], headers, body


def asgi_post_json(app: Any, path: str, payload: dict[str, Any]) -> tuple[int, dict[str, str], bytes]:
    messages: list[dict[str, Any]] = []
    body_bytes = json.dumps(payload).encode("utf-8")

    async def receive() -> dict[str, Any]:
        return {"type": "http.request", "body": body_bytes, "more_body": False}

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
    headers = {key.decode("latin-1").lower(): value.decode("latin-1") for key, value in start["headers"]}
    return start["status"], headers, body


class SpecificationsToolTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.data_path = self.root / "sample.csv"
        self.data_path.write_text(
            "Age,Premium,Weight,Segment\n"
            "20,100,1,A\n"
            "40,200,2,B\n",
            encoding="utf-8",
        )

    def test_specs_tool_can_be_enabled_and_registers_routes(self) -> None:
        app = create_app(self.data_path, token="", tools=["specs"], use_saved_filters=False, use_kpis=False, use_features=False)
        paths = {route.path for route in app.routes}

        self.assertEqual(normalise_tools("specs"), ["column_profile", "specs"])
        self.assertEqual(app.state.enabled_tools, ["column_profile", "specs"])
        self.assertIn("/api/specs/{kind}", paths)
        self.assertIn("/api/specs/{kind}/validate", paths)
        self.assertIn("/api/specs/{kind}/save", paths)

    def test_get_missing_default_feature_spec_returns_editable_empty_table(self) -> None:
        previous_cwd = Path.cwd()
        try:
            os.chdir(self.root)
            app = create_app(self.data_path, token="", tools=["specs"])
            status, _, body = asgi_get(app, "/api/specs/feature")
        finally:
            os.chdir(previous_cwd)

        payload = json.loads(body)
        self.assertEqual(status, 200)
        self.assertFalse(payload["exists"])
        self.assertEqual(payload["columns"], ["Feature", "Grouping", "Base", "min", "max", "banding", "scenario1"])
        self.assertEqual(Path(payload["path"]), (self.root / "specs" / "feature_spec.csv").resolve())

    def test_save_kpi_spec_creates_file_and_refreshes_loaded_metadata(self) -> None:
        previous_cwd = Path.cwd()
        try:
            os.chdir(self.root)
            app = create_app(self.data_path, token="", tools=["specs"], use_saved_filters=False, use_features=False)
            status, _, body = asgi_post_json(
                app,
                "/api/specs/kpi/save",
                {
                    "columns": ["group", "name", "actual", "denominator", "decimals", "format"],
                    "rows": [
                        {
                            "group": "Pricing",
                            "name": "Premium",
                            "actual": "Premium",
                            "denominator": "Weight",
                            "decimals": "2",
                            "format": "currency",
                        }
                    ],
                },
            )
        finally:
            os.chdir(previous_cwd)

        payload = json.loads(body)
        self.assertEqual(status, 200)
        self.assertTrue(payload["valid"])
        self.assertEqual(app.state.kpis[0]["name"], "Premium")
        self.assertEqual((self.root / "specs" / "kpi_spec.csv").read_text(encoding="utf-8").splitlines()[0], "group,name,actual,denominator,decimals,format")

    def test_validate_filter_spec_checks_duckdb_expression(self) -> None:
        app = create_app(self.data_path, token="", tools=["specs"], use_kpis=False, use_features=False)

        status, _, body = asgi_post_json(
            app,
            "/api/specs/filter/validate",
            {
                "columns": ["theme", "name", "expression"],
                "rows": [{"theme": "Sample", "name": "Broken", "expression": "MissingColumn = 1"}],
            },
        )

        payload = json.loads(body)
        self.assertEqual(status, 200)
        self.assertFalse(payload["valid"])
        self.assertIn("Invalid filter", payload["errors"][0])
        self.assertEqual(len(payload["row_issues"]), 1)
        self.assertEqual(payload["row_issues"][0]["row_number"], 2)
        self.assertEqual(payload["row_issues"][0]["severity"], "error")
        self.assertEqual(payload["row_issues"][0]["message"], payload["errors"][0])

    def test_validate_filter_spec_reports_row_issues_for_missing_fields(self) -> None:
        app = create_app(self.data_path, token="", tools=["specs"], use_kpis=False, use_features=False)

        status, _, body = asgi_post_json(
            app,
            "/api/specs/filter/validate",
            {
                "columns": ["theme", "name", "expression"],
                "rows": [{"theme": "Sample", "name": "", "expression": "Age = 20"}],
            },
        )

        payload = json.loads(body)
        self.assertEqual(status, 200)
        self.assertFalse(payload["valid"])
        self.assertEqual(payload["errors"], ["filter_spec.csv row 2 is missing: name"])
        self.assertEqual(payload["row_issues"], [{"row_number": 2, "severity": "error", "message": payload["errors"][0]}])

    def test_validate_kpi_spec_reports_row_issues(self) -> None:
        app = create_app(self.data_path, token="", tools=["specs"], use_saved_filters=False, use_features=False)

        status, _, body = asgi_post_json(
            app,
            "/api/specs/kpi/validate",
            {
                "columns": ["group", "name", "actual", "denominator", "decimals", "format"],
                "rows": [
                    {
                        "group": "Pricing",
                        "name": "Missing actual",
                        "actual": "MissingActual",
                        "denominator": "",
                        "decimals": "2",
                        "format": "currency",
                    }
                ],
            },
        )

        payload = json.loads(body)
        self.assertEqual(status, 200)
        self.assertFalse(payload["valid"])
        self.assertEqual(payload["errors"], ["kpi_spec.csv row 2 actual column does not exist: MissingActual"])
        self.assertEqual(payload["row_issues"], [{"row_number": 2, "severity": "error", "message": payload["errors"][0]}])

    def test_save_feature_spec_updates_feature_bases_in_schema(self) -> None:
        previous_cwd = Path.cwd()
        try:
            os.chdir(self.root)
            app = create_app(self.data_path, token="", tools=["specs"], use_saved_filters=False, use_kpis=False)
            status, _, _ = asgi_post_json(
                app,
                "/api/specs/feature/save",
                {
                    "columns": ["Feature", "Grouping", "Base", "min", "max", "banding", "scenario1"],
                    "rows": [{"Feature": "Age", "Grouping": "Driver", "Base": "40", "min": "20", "max": "80", "banding": "5", "scenario1": "feature"}],
                },
            )
            schema_status, _, schema_body = asgi_get(app, "/api/schema")
        finally:
            os.chdir(previous_cwd)
        schema = json.loads(schema_body)

        self.assertEqual(status, 200)
        self.assertEqual(schema_status, 200)
        self.assertEqual(schema["feature_bases"], {"Age": "40"})

    def test_validate_feature_spec_allows_rows_not_in_dataset(self) -> None:
        app = create_app(self.data_path, token="", tools=["specs"], use_saved_filters=False, use_kpis=False)

        status, _, body = asgi_post_json(
            app,
            "/api/specs/feature/validate",
            {
                "columns": ["Feature", "Grouping", "Base", "min", "max", "banding", "scenario1"],
                "rows": [{"Feature": "FutureFeature", "Grouping": "Reference", "Base": "", "min": "", "max": "", "banding": "", "scenario1": "feature"}],
            },
        )

        payload = json.loads(body)
        self.assertEqual(status, 200)
        self.assertTrue(payload["valid"])
        self.assertEqual(payload["errors"], [])
        self.assertEqual(payload["warnings"], [])
        self.assertEqual(payload["row_issues"], [])

    def test_save_feature_spec_allows_rows_not_in_dataset(self) -> None:
        previous_cwd = Path.cwd()
        try:
            os.chdir(self.root)
            app = create_app(self.data_path, token="", tools=["specs"], use_saved_filters=False, use_kpis=False)
            status, _, body = asgi_post_json(
                app,
                "/api/specs/feature/save",
                {
                    "columns": ["Feature", "Grouping", "Base", "min", "max", "banding", "scenario1"],
                    "rows": [{"Feature": "FutureFeature", "Grouping": "Reference", "Base": "", "min": "", "max": "", "banding": "", "scenario1": "feature"}],
                },
            )
        finally:
            os.chdir(previous_cwd)

        payload = json.loads(body)
        self.assertEqual(status, 200)
        self.assertTrue(payload["valid"])
        self.assertEqual(payload["errors"], [])
        self.assertEqual(payload["row_issues"], [])
        self.assertEqual(app.state.feature_spec["rows"][0]["feature"], "FutureFeature")
        self.assertIn("FutureFeature", (self.root / "specs" / "feature_spec.csv").read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
