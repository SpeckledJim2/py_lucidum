from __future__ import annotations

import socket
import threading
from types import SimpleNamespace
import unittest

import uvicorn

from py_lucidum.cli import LucidumServer, _display_url_for_app, _run_server, ensure_port_available


class FakeServer:
    def __init__(self) -> None:
        self.event = threading.Event()
        self.thread_name: str | None = None

    def run(self) -> None:
        self.thread_name = threading.current_thread().name
        self.event.set()


class CliRuntimeTests(unittest.TestCase):
    def test_run_server_calls_uvicorn_directly_without_running_event_loop(self) -> None:
        server = FakeServer()

        _run_server(server)

        self.assertTrue(server.event.is_set())
        self.assertEqual(server.thread_name, threading.current_thread().name)

    def test_lucidum_server_started_message_uses_stop_instruction(self) -> None:
        async def app(scope: object, receive: object, send: object) -> None:
            return None

        config = uvicorn.Config(app, host="127.0.0.1", port=8000)
        server = LucidumServer(config, "http://127.0.0.1:8000/", "Use the app Stop app button to quit")

        with self.assertLogs("uvicorn.error", level="INFO") as logs:
            server._log_started_message([])

        self.assertEqual(
            logs.output,
            ["INFO:uvicorn.error:Uvicorn running on http://127.0.0.1:8000/ (Use the app Stop app button to quit)"],
        )

    def test_display_url_for_app_includes_token_and_defaults(self) -> None:
        app = SimpleNamespace(
            state=SimpleNamespace(
                token="dev-token",
                defaults={"x": "Driver Age", "actual": "AvgPrice1_5", "denominator": "Exposure", "unused": "ignored"},
            )
        )

        url = _display_url_for_app(app, "127.0.0.1", 8000)

        self.assertEqual(
            url,
            "http://127.0.0.1:8000/?token=dev-token&x=Driver+Age&actual=AvgPrice1_5&denominator=Exposure",
        )

    def test_ensure_port_available_reports_busy_port(self) -> None:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.bind(("127.0.0.1", 0))
            port = int(sock.getsockname()[1])

            with self.assertRaisesRegex(RuntimeError, f"Port {port} is already in use"):
                ensure_port_available("127.0.0.1", port)


class AsyncCliRuntimeTests(unittest.IsolatedAsyncioTestCase):
    async def test_run_server_uses_background_thread_with_running_event_loop(self) -> None:
        server = FakeServer()

        _run_server(server)

        self.assertTrue(server.event.wait(timeout=2))
        self.assertEqual(server.thread_name, "py-lucidum-uvicorn")


if __name__ == "__main__":
    unittest.main()
