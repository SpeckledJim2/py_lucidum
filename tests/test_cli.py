from __future__ import annotations

import io
import socket
import threading
from contextlib import redirect_stderr, redirect_stdout
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import uvicorn

from py_lucidum.cli import LucidumServer, _display_url_for_app, _run_server, ensure_port_available, main, run_app, serve


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
                defaults={
                    "x": "Driver Age",
                    "actual": "AvgPrice1_5",
                    "denominator": "Exposure",
                    "postcode_area": "Area",
                    "unused": "ignored",
                },
            )
        )

        url = _display_url_for_app(app, "127.0.0.1", 8000)

        self.assertEqual(
            url,
            "http://127.0.0.1:8000/?token=dev-token&x=Driver+Age&actual=AvgPrice1_5&denominator=Exposure&postcode_area=Area",
        )

    def test_ensure_port_available_reports_busy_port(self) -> None:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.bind(("127.0.0.1", 0))
            port = int(sock.getsockname()[1])

            with self.assertRaisesRegex(RuntimeError, f"Port {port} is already in use"):
                ensure_port_available("127.0.0.1", port)

    def test_run_app_checks_busy_port_before_printing(self) -> None:
        app = SimpleNamespace(state=SimpleNamespace(token="", defaults={}))
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.bind(("127.0.0.1", 0))
            port = int(sock.getsockname()[1])
            stdout = io.StringIO()

            with redirect_stdout(stdout), self.assertRaisesRegex(RuntimeError, f"Port {port} is already in use"):
                run_app(app, port=port)

        self.assertEqual(stdout.getvalue(), "")

    def test_serve_checks_busy_port_before_printing_or_building_app(self) -> None:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.bind(("127.0.0.1", 0))
            port = int(sock.getsockname()[1])
            stdout = io.StringIO()

            with (
                patch("py_lucidum.cli.create_app") as create_app_mock,
                redirect_stdout(stdout),
                self.assertRaisesRegex(RuntimeError, f"Port {port} is already in use"),
            ):
                serve("missing.parquet", port=port)

        create_app_mock.assert_not_called()
        self.assertEqual(stdout.getvalue(), "")

    def test_main_reports_runtime_error_without_traceback(self) -> None:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.bind(("127.0.0.1", 0))
            port = int(sock.getsockname()[1])
            stderr = io.StringIO()

            with (
                patch("sys.argv", ["lucidum", "missing.parquet", "--port", str(port)]),
                patch("py_lucidum.cli.create_app") as create_app_mock,
                redirect_stderr(stderr),
                self.assertRaises(SystemExit) as exit_context,
            ):
                main()

        self.assertEqual(exit_context.exception.code, 1)
        self.assertIn(f"lucidum: error: Port {port} is already in use", stderr.getvalue())
        self.assertNotIn("Traceback", stderr.getvalue())
        create_app_mock.assert_not_called()


class AsyncCliRuntimeTests(unittest.IsolatedAsyncioTestCase):
    async def test_run_server_uses_background_thread_with_running_event_loop(self) -> None:
        server = FakeServer()

        _run_server(server)

        self.assertTrue(server.event.wait(timeout=2))
        self.assertEqual(server.thread_name, "py-lucidum-uvicorn")


if __name__ == "__main__":
    unittest.main()
