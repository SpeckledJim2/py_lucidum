from __future__ import annotations

import asyncio
import json
import os
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any
from urllib.parse import urlsplit

from py_lucidum.app import create_app


def asgi_request(
    app: Any,
    method: str,
    target: str,
    payload: dict[str, Any] | None = None,
    headers: dict[str, str] | None = None,
    client: tuple[str, int] = ("127.0.0.1", 12345),
) -> tuple[int, dict[str, str], bytes]:
    messages: list[dict[str, Any]] = []
    body = json.dumps(payload).encode("utf-8") if payload is not None else b""
    split = urlsplit(target)
    path = split.path or "/"
    raw_headers = [
        (key.lower().encode("latin-1"), value.encode("latin-1"))
        for key, value in (headers or {}).items()
    ]
    if payload is not None and not any(key == b"content-type" for key, _ in raw_headers):
        raw_headers.append((b"content-type", b"application/json"))

    received = False

    async def receive() -> dict[str, Any]:
        nonlocal received
        if received:
            return {"type": "http.request", "body": b"", "more_body": False}
        received = True
        return {"type": "http.request", "body": body, "more_body": False}

    async def send(message: dict[str, Any]) -> None:
        messages.append(message)

    scope = {
        "type": "http",
        "asgi": {"version": "3.0", "spec_version": "2.3"},
        "http_version": "1.1",
        "method": method.upper(),
        "scheme": "http",
        "path": path,
        "raw_path": path.encode("ascii"),
        "query_string": split.query.encode("ascii"),
        "headers": raw_headers,
        "client": client,
        "server": ("testserver", 80),
    }
    asyncio.run(app(scope, receive, send))

    start = next(message for message in messages if message["type"] == "http.response.start")
    response_body = b"".join(message.get("body", b"") for message in messages if message["type"] == "http.response.body")
    response_headers = {key.decode("latin-1").lower(): value.decode("latin-1") for key, value in start["headers"]}
    return start["status"], response_headers, response_body


def telemetry_snapshot(app: Any, token: str = "") -> dict[str, Any]:
    target = f"/api/telemetry?token={token}" if token else "/api/telemetry"
    status, _, body = asgi_request(app, "GET", target)
    assert status == 200
    return json.loads(body)


class TelemetryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.data_path = Path(self.tmp.name) / "sample.csv"
        self.data_path.write_text(
            "UseofVan,Actual,Weight\n"
            "Social,100,10\n"
            "Business,200,20\n",
            encoding="utf-8",
        )

    def test_records_successful_app_action_by_client(self) -> None:
        app = create_app(self.data_path, token="")

        status, _, _ = asgi_request(app, "GET", "/api/schema", headers={"user-agent": "Browser A"})
        snapshot = telemetry_snapshot(app)

        self.assertEqual(status, 200)
        self.assertEqual(snapshot["totals"]["total_requests"], 1)
        self.assertEqual(snapshot["totals"]["app_actions"], 1)
        self.assertEqual(snapshot["totals"]["errors"], 0)
        self.assertEqual(snapshot["totals"]["status_counts"]["200"], 1)
        self.assertEqual(len(snapshot["clients"]), 1)
        client = snapshot["clients"][0]
        self.assertEqual(client["client_ip"], "127.0.0.1")
        self.assertEqual(client["user_agent"], "Browser A")
        self.assertEqual(client["request_count"], 1)
        self.assertEqual(client["app_action_count"], 1)
        self.assertEqual(client["last_app_action"], "Load schema")
        self.assertEqual(client["last_status"], 200)
        self.assertGreaterEqual(client["last_duration_ms"], 0)

    def test_records_errors_and_recent_activity(self) -> None:
        app = create_app(self.data_path, token="")

        status, _, _ = asgi_request(app, "POST", "/api/chart", payload={"x": "Missing"})
        snapshot = telemetry_snapshot(app)

        self.assertEqual(status, 400)
        self.assertEqual(snapshot["totals"]["total_requests"], 1)
        self.assertEqual(snapshot["totals"]["app_actions"], 1)
        self.assertEqual(snapshot["totals"]["errors"], 1)
        self.assertEqual(snapshot["totals"]["status_counts"]["400"], 1)
        self.assertEqual(snapshot["clients"][0]["error_count"], 1)
        self.assertEqual(snapshot["clients"][0]["last_app_action"], "Line/bar chart")
        self.assertEqual(snapshot["totals"]["recent_error_count"], 1)
        self.assertEqual(snapshot["totals"]["recent_error_rate"], 100.0)
        self.assertEqual(snapshot["totals"]["slowest_recent_action"]["action"], "Line/bar chart")
        self.assertEqual(snapshot["totals"]["slowest_recent_action"]["status"], 400)
        event = snapshot["recent_activity"][0]
        self.assertEqual(event["path"], "/api/chart")
        self.assertEqual(event["action"], "Line/bar chart")
        self.assertEqual(event["status"], 400)
        self.assertTrue(event["error"])

    def test_groups_clients_by_ip_and_user_agent(self) -> None:
        app = create_app(self.data_path, token="")

        asgi_request(app, "GET", "/api/schema", headers={"user-agent": "Browser A"}, client=("10.0.0.5", 1111))
        asgi_request(app, "GET", "/api/schema", headers={"user-agent": "Browser B"}, client=("10.0.0.5", 2222))
        snapshot = telemetry_snapshot(app)

        self.assertEqual(snapshot["totals"]["total_requests"], 2)
        self.assertEqual(snapshot["totals"]["app_actions"], 2)
        self.assertEqual(snapshot["totals"]["clients"], 2)
        self.assertEqual(
            sorted((client["client_ip"], client["user_agent"]) for client in snapshot["clients"]),
            [("10.0.0.5", "Browser A"), ("10.0.0.5", "Browser B")],
        )

    def test_telemetry_and_monitor_use_existing_token_auth(self) -> None:
        app = create_app(self.data_path, token="dev-token")

        telemetry_status, telemetry_headers, telemetry_body = asgi_request(app, "GET", "/api/telemetry")
        monitor_status, _, monitor_body = asgi_request(app, "GET", "/monitor")
        valid_status, valid_headers, valid_body = asgi_request(app, "GET", "/monitor?token=dev-token")
        valid_api_status, valid_api_headers, _ = asgi_request(app, "GET", "/api/telemetry?token=dev-token")

        self.assertEqual(telemetry_status, 401)
        self.assertIn(b"Invalid or missing app token", telemetry_body)
        self.assertEqual(monitor_status, 401)
        self.assertIn(b"Invalid or missing app token", monitor_body)
        self.assertEqual(valid_status, 200)
        self.assertEqual(valid_headers.get("cache-control"), "no-store")
        self.assertIn(b"lucidum monitor", valid_body)
        self.assertEqual(valid_api_status, 200)
        self.assertEqual(valid_api_headers.get("cache-control"), "no-store")
        self.assertEqual(telemetry_headers.get("cache-control"), None)

    def test_telemetry_polling_is_excluded_from_totals(self) -> None:
        app = create_app(self.data_path, token="")

        first = telemetry_snapshot(app)
        second = telemetry_snapshot(app)

        self.assertEqual(first["totals"]["total_requests"], 0)
        self.assertEqual(second["totals"]["total_requests"], 0)
        self.assertEqual(second["recent_activity"], [])

    def test_static_assets_are_diagnostics_only(self) -> None:
        app = create_app(self.data_path, token="")

        monitor_js_status, _, _ = asgi_request(app, "GET", "/static/monitor.js")
        favicon_status, _, _ = asgi_request(app, "GET", "/favicon.ico")
        snapshot = telemetry_snapshot(app)
        static_assets = snapshot["diagnostics"]["static_assets"]

        self.assertEqual(monitor_js_status, 200)
        self.assertEqual(favicon_status, 200)
        self.assertEqual(snapshot["totals"]["total_requests"], 0)
        self.assertEqual(snapshot["totals"]["app_actions"], 0)
        self.assertEqual(snapshot["totals"]["errors"], 0)
        self.assertEqual(snapshot["totals"]["active_clients"], 0)
        self.assertEqual(snapshot["totals"]["clients"], 0)
        self.assertEqual(snapshot["totals"]["status_counts"], {})
        self.assertEqual(snapshot["recent_activity"], [])
        self.assertEqual(snapshot["clients"], [])
        self.assertEqual(static_assets["count"], 2)
        self.assertEqual(static_assets["error_count"], 0)
        self.assertEqual(static_assets["status"], 200)
        self.assertEqual(static_assets["path"], "/favicon.ico")
        self.assertGreaterEqual(static_assets["duration_ms"], 0)
        self.assertIsNotNone(static_assets["last_seen"])

    def test_snapshot_includes_process_memory(self) -> None:
        app = create_app(self.data_path, token="")

        snapshot = telemetry_snapshot(app)
        process = snapshot["process"]
        system_memory = process["system_memory"]

        self.assertEqual(process["pid"], os.getpid())
        self.assertGreater(process["rss_bytes"], 0)
        self.assertGreater(process["rss_mb"], 0)
        self.assertGreaterEqual(process["peak_rss_bytes"], process["rss_bytes"])
        self.assertGreaterEqual(process["peak_rss_mb"], process["rss_mb"])
        self.assertGreaterEqual(process["vms_bytes"], process["rss_bytes"])
        self.assertGreaterEqual(process["vms_mb"], process["rss_mb"])
        self.assertGreaterEqual(process["memory_percent"], 0)
        self.assertGreaterEqual(process["cpu_percent"], 0)
        self.assertGreaterEqual(process["system_cpu_percent"], 0)
        self.assertGreater(process["thread_count"], 0)
        self.assertGreater(system_memory["total_bytes"], 0)
        self.assertGreater(system_memory["total_mb"], 0)
        self.assertGreaterEqual(system_memory["available_bytes"], 0)
        self.assertGreaterEqual(system_memory["available_mb"], 0)
        self.assertGreaterEqual(system_memory["used_percent"], 0)
        if "uss_bytes" in process:
            self.assertGreaterEqual(process["uss_bytes"], 0)
            self.assertGreaterEqual(process["uss_mb"], 0)

    def test_token_query_values_and_request_bodies_are_not_stored(self) -> None:
        app = create_app(self.data_path, token="secret-token")

        asgi_request(app, "GET", "/api/schema?token=secret-token")
        asgi_request(
            app,
            "POST",
            "/api/chart?token=secret-token",
            payload={"x": "Missing", "filter": "UseofVan = 'Social'"},
        )
        _, _, body = asgi_request(app, "GET", "/api/telemetry?token=secret-token")
        payload_text = body.decode("utf-8")
        snapshot = json.loads(body)

        self.assertNotIn("secret-token", payload_text)
        self.assertNotIn("UseofVan = 'Social'", payload_text)
        self.assertEqual([event["path"] for event in snapshot["recent_activity"]], ["/api/chart", "/api/schema"])

    def test_health_checks_update_heartbeat_only(self) -> None:
        app = create_app(self.data_path, token="")

        status, _, _ = asgi_request(
            app,
            "GET",
            "/api/health",
            headers={"user-agent": "Health Check"},
            client=("10.0.0.9", 3333),
        )
        snapshot = telemetry_snapshot(app)

        self.assertEqual(status, 200)
        self.assertEqual(snapshot["totals"]["total_requests"], 0)
        self.assertEqual(snapshot["totals"]["app_actions"], 0)
        self.assertEqual(snapshot["totals"]["errors"], 0)
        self.assertEqual(snapshot["totals"]["active_clients"], 0)
        self.assertEqual(snapshot["totals"]["clients"], 0)
        self.assertEqual(snapshot["totals"]["in_flight"], 0)
        self.assertEqual(snapshot["totals"]["status_counts"], {})
        self.assertEqual(snapshot["totals"]["recent_error_count"], 0)
        self.assertEqual(snapshot["totals"]["recent_error_rate"], 0.0)
        self.assertIsNone(snapshot["totals"]["slowest_recent_action"])
        self.assertEqual(snapshot["clients"], [])
        self.assertEqual(snapshot["recent_activity"], [])
        self.assertEqual(snapshot["heartbeat"]["count"], 1)
        self.assertEqual(snapshot["heartbeat"]["status"], 200)
        self.assertGreaterEqual(snapshot["heartbeat"]["duration_ms"], 0)
        self.assertEqual(snapshot["heartbeat"]["error_count"], 0)
        self.assertIsNotNone(snapshot["heartbeat"]["last_seen"])

    def test_health_checks_do_not_overwrite_activity_metrics(self) -> None:
        app = create_app(self.data_path, token="")

        asgi_request(app, "GET", "/api/schema")
        asgi_request(app, "GET", "/api/health")
        snapshot = telemetry_snapshot(app)

        self.assertEqual(snapshot["totals"]["total_requests"], 1)
        self.assertEqual(snapshot["totals"]["app_actions"], 1)
        self.assertEqual(snapshot["totals"]["status_counts"]["200"], 1)
        self.assertEqual(snapshot["clients"][0]["request_count"], 1)
        self.assertEqual(snapshot["clients"][0]["last_path"], "/api/schema")
        self.assertEqual(snapshot["clients"][0]["last_app_action"], "Load schema")
        self.assertEqual([event["path"] for event in snapshot["recent_activity"]], ["/api/schema"])
        self.assertEqual(snapshot["totals"]["slowest_recent_action"]["action"], "Load schema")
        self.assertEqual(snapshot["totals"]["recent_error_rate"], 0.0)
        self.assertEqual(snapshot["heartbeat"]["count"], 1)
        self.assertEqual(snapshot["heartbeat"]["status"], 200)


if __name__ == "__main__":
    unittest.main()
