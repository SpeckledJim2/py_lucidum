from __future__ import annotations

import io
import socket
import threading
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import Mock, patch

import uvicorn

from py_lucidum.cli import (
    LucidumServer,
    _display_url_for_app,
    _lan_url_hint_for_app,
    _run_server,
    _start_app_server,
    _usable_lan_ipv4,
    ensure_port_available,
    main,
    run_app,
    serve,
)


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

        opener = Mock()
        config = uvicorn.Config(app, host="127.0.0.1", port=8000)
        server = LucidumServer(
            config,
            "http://127.0.0.1:8000/",
            "Use the app Stop app button to quit",
            browser_opener=opener,
        )

        with self.assertLogs("uvicorn.error", level="INFO") as logs:
            server._log_started_message([])

        self.assertEqual(
            logs.output,
            ["INFO:uvicorn.error:Uvicorn running on http://127.0.0.1:8000/ (Use the app Stop app button to quit)"],
        )
        opener.assert_not_called()

    def test_lucidum_server_opens_browser_after_started_message(self) -> None:
        async def app(scope: object, receive: object, send: object) -> None:
            return None

        opener = Mock()
        config = uvicorn.Config(app, host="127.0.0.1", port=8000)
        server = LucidumServer(
            config,
            "http://127.0.0.1:8000/",
            "Press CTRL+C to quit",
            open_browser=True,
            browser_opener=opener,
        )

        with self.assertLogs("uvicorn.error", level="INFO"):
            server._log_started_message([])

        opener.assert_called_once_with("http://127.0.0.1:8000/")

    def test_display_url_for_app_includes_token_and_defaults(self) -> None:
        app = SimpleNamespace(
            state=SimpleNamespace(
                token="dev-token",
                defaults={
                    "x": "Driver Age",
                    "actual": "AvgPrice1_5",
                    "denominator": "Exposure",
                    "postcode_area": "Area",
                    "postcode_unit": "Unit",
                    "latitude": "lat_col",
                    "longitude": "long_col",
                    "unused": "ignored",
                },
            )
        )

        url = _display_url_for_app(app, "127.0.0.1", 8000)

        self.assertEqual(
            url,
            "http://127.0.0.1:8000/?token=dev-token&x=Driver+Age&actual=AvgPrice1_5&denominator=Exposure&postcode_area=Area&postcode_unit=Unit&latitude=lat_col&longitude=long_col",
        )

    def test_display_url_for_wildcard_bind_uses_localhost_and_lan_hint(self) -> None:
        app = SimpleNamespace(
            state=SimpleNamespace(
                token="dev-token",
                defaults={"x": "Driver Age"},
            )
        )

        self.assertEqual(
            _display_url_for_app(app, "0.0.0.0", 8000),
            "http://127.0.0.1:8000/?token=dev-token&x=Driver+Age",
        )
        with patch("py_lucidum.cli._detect_primary_lan_ipv4", return_value="192.168.1.50"):
            self.assertEqual(
                _lan_url_hint_for_app(app, "0.0.0.0", 8000),
                "http://192.168.1.50:8000/?token=dev-token&x=Driver+Age",
            )
        self.assertIsNone(_lan_url_hint_for_app(app, "127.0.0.1", 8000))

    def test_lan_url_hint_falls_back_to_placeholder_when_ip_detection_fails(self) -> None:
        app = SimpleNamespace(state=SimpleNamespace(token="dev-token", defaults={}))

        with patch("py_lucidum.cli._detect_primary_lan_ipv4", return_value=None):
            self.assertEqual(
                _lan_url_hint_for_app(app, "0.0.0.0", 8000),
                "http://<this-computer-ip>:8000/?token=dev-token",
            )

    def test_primary_lan_ipv4_detection_rejects_loopback_and_wildcard_addresses(self) -> None:
        self.assertEqual(_usable_lan_ipv4("192.168.1.50"), "192.168.1.50")
        self.assertIsNone(_usable_lan_ipv4("127.0.0.1"))
        self.assertIsNone(_usable_lan_ipv4("0.0.0.0"))
        self.assertIsNone(_usable_lan_ipv4("not-an-ip"))

    def test_ensure_port_available_reports_busy_port(self) -> None:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.bind(("127.0.0.1", 0))
            sock.listen(1)
            port = int(sock.getsockname()[1])

            with self.assertRaisesRegex(RuntimeError, f"Port {port} is already in use"):
                ensure_port_available("127.0.0.1", port)

    def test_ensure_port_available_ignores_recent_closed_connection(self) -> None:
        host = "127.0.0.1"
        errors: list[BaseException] = []

        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
            listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            listener.bind((host, 0))
            listener.listen(1)
            port = int(listener.getsockname()[1])

            def connect_once() -> None:
                try:
                    with socket.create_connection((host, port), timeout=2) as client:
                        client.sendall(b"x")
                        client.recv(1)
                except BaseException as exc:  # pragma: no cover - surfaced by assertion below.
                    errors.append(exc)

            thread = threading.Thread(target=connect_once, name="py-lucidum-port-test")
            thread.start()
            conn, _ = listener.accept()
            with conn:
                conn.recv(1)
            thread.join(timeout=2)

        self.assertFalse(thread.is_alive())
        self.assertEqual(errors, [])
        ensure_port_available(host, port)

    def test_run_app_checks_busy_port_before_printing(self) -> None:
        app = SimpleNamespace(state=SimpleNamespace(token="", defaults={}))
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.bind(("127.0.0.1", 0))
            sock.listen(1)
            port = int(sock.getsockname()[1])
            stdout = io.StringIO()

            with redirect_stdout(stdout), self.assertRaisesRegex(RuntimeError, f"Port {port} is already in use"):
                run_app(app, port=port)

        self.assertEqual(stdout.getvalue(), "")

    def test_run_app_prints_lan_hint_for_wildcard_bind(self) -> None:
        app = SimpleNamespace(state=SimpleNamespace(token="dev-token", defaults={}))
        stdout = io.StringIO()

        with (
            patch("py_lucidum.cli._detect_primary_lan_ipv4", return_value="192.168.1.50"),
            patch("py_lucidum.cli._start_app_server") as start_server_mock,
            redirect_stdout(stdout),
        ):
            url = run_app(app, host="0.0.0.0", port=8055)

        self.assertEqual(url, "http://127.0.0.1:8055/?token=dev-token")
        self.assertIn("Open locally http://127.0.0.1:8055/?token=dev-token", stdout.getvalue())
        self.assertIn(
            "Open from another device on your LAN: http://192.168.1.50:8055/?token=dev-token",
            stdout.getvalue(),
        )
        start_server_mock.assert_called_once_with(app, "0.0.0.0", 8055, url, False, False)

    def test_start_app_server_defers_open_until_server_started(self) -> None:
        app = SimpleNamespace(state=SimpleNamespace())
        url = "http://127.0.0.1:0/"

        with (
            patch("py_lucidum.cli.webbrowser.open") as opener_mock,
            patch("py_lucidum.cli._run_server") as run_server_mock,
        ):
            _start_app_server(app, "127.0.0.1", 0, url, open_browser=True, run_in_background=False)

        opener_mock.assert_not_called()
        run_server_mock.assert_called_once()
        server = run_server_mock.call_args.args[0]
        self.assertIsInstance(server, LucidumServer)
        self.assertEqual(server.display_url, url)
        self.assertTrue(server.open_browser)
        self.assertIs(server.browser_opener, opener_mock)
        self.assertEqual(run_server_mock.call_args.kwargs["run_in_background"], False)
        self.assertEqual(
            app.state.lucidum_server_metadata,
            {"host": "127.0.0.1", "port": 0, "display_url": "http://127.0.0.1:0/"},
        )
        self.assertTrue(callable(app.state.shutdown_callback))

    def test_serve_checks_busy_port_before_printing_or_building_app(self) -> None:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.bind(("127.0.0.1", 0))
            sock.listen(1)
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

    def test_serve_passes_unit_point_defaults(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            data_path = Path(tmp_dir) / "sample.csv"
            data_path.write_text("x,y\n1,2\n", encoding="utf-8")
            app = SimpleNamespace(state=SimpleNamespace(token="", defaults={}))
            stdout = io.StringIO()

            with (
                patch("py_lucidum.cli.create_app", return_value=app) as create_app_mock,
                patch("py_lucidum.cli._start_app_server") as start_server_mock,
                redirect_stdout(stdout),
            ):
                serve(
                    data_path,
                    token="",
                    postcode_unit="Unit",
                    latitude="lat_col",
                    longitude="long_col",
                )

        defaults = create_app_mock.call_args.kwargs["defaults"]
        self.assertEqual(defaults["postcode_unit"], "Unit")
        self.assertEqual(defaults["latitude"], "lat_col")
        self.assertEqual(defaults["longitude"], "long_col")
        start_server_mock.assert_called_once()

    def test_serve_passes_kpi_options_and_reports_status(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            data_path = root / "sample.csv"
            kpis_path = root / "kpi_spec.csv"
            data_path.write_text("x,y\n1,2\n", encoding="utf-8")
            kpis_path.write_text("group,name,actual,denominator,decimals,format\nY,Y,y,N,0,number\n", encoding="utf-8")
            app = SimpleNamespace(
                state=SimpleNamespace(
                    token="",
                    defaults={},
                    use_saved_filters=False,
                    resolved_filters_path=None,
                    use_kpis=True,
                    resolved_kpis_path=kpis_path,
                )
            )
            stdout = io.StringIO()

            with (
                patch("py_lucidum.cli.create_app", return_value=app) as create_app_mock,
                patch("py_lucidum.cli._start_app_server") as start_server_mock,
                redirect_stdout(stdout),
            ):
                serve(data_path, token="", kpis=kpis_path, no_filters=True)

        self.assertEqual(create_app_mock.call_args.kwargs["kpis_path"], kpis_path)
        self.assertTrue(create_app_mock.call_args.kwargs["use_kpis"])
        self.assertIn(f"KPIs: {kpis_path}", stdout.getvalue())
        start_server_mock.assert_called_once()

    def test_serve_prints_lan_hint_for_wildcard_bind(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            data_path = Path(tmp_dir) / "sample.csv"
            data_path.write_text("x,y\n1,2\n", encoding="utf-8")
            app = SimpleNamespace(
                state=SimpleNamespace(
                    token="dev-token",
                    defaults={"x": "Driver Age"},
                    use_saved_filters=False,
                    resolved_filters_path=None,
                    use_kpis=False,
                    resolved_kpis_path=None,
                )
            )
            stdout = io.StringIO()

            with (
                patch("py_lucidum.cli.create_app", return_value=app),
                patch("py_lucidum.cli._detect_primary_lan_ipv4", return_value="192.168.1.50"),
                patch("py_lucidum.cli._start_app_server") as start_server_mock,
                redirect_stdout(stdout),
            ):
                url = serve(data_path, host="0.0.0.0", port=8053, token="dev-token", x="Driver Age")

        self.assertEqual(url, "http://127.0.0.1:8053/?token=dev-token&x=Driver+Age")
        output = stdout.getvalue()
        self.assertIn("Open locally http://127.0.0.1:8053/?token=dev-token&x=Driver+Age", output)
        self.assertIn(
            "Open from another device on your LAN: http://192.168.1.50:8053/?token=dev-token&x=Driver+Age",
            output,
        )
        self.assertNotIn("Open http://0.0.0.0:8053", output)
        start_server_mock.assert_called_once_with(app, "0.0.0.0", 8053, url, False, False)

    def test_serve_prints_simple_open_for_concrete_bind(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            data_path = Path(tmp_dir) / "sample.csv"
            data_path.write_text("x,y\n1,2\n", encoding="utf-8")
            app = SimpleNamespace(
                state=SimpleNamespace(
                    token="",
                    defaults={},
                    use_saved_filters=False,
                    resolved_filters_path=None,
                    use_kpis=False,
                    resolved_kpis_path=None,
                )
            )
            stdout = io.StringIO()

            with (
                patch("py_lucidum.cli.create_app", return_value=app),
                patch("py_lucidum.cli._start_app_server") as start_server_mock,
                redirect_stdout(stdout),
            ):
                url = serve(data_path, host="127.0.0.1", port=8054, token="")

        self.assertEqual(url, "http://127.0.0.1:8054/")
        self.assertIn("Open http://127.0.0.1:8054/", stdout.getvalue())
        self.assertNotIn("Open locally", stdout.getvalue())
        self.assertNotIn("Open from another device on your LAN", stdout.getvalue())
        start_server_mock.assert_called_once_with(app, "127.0.0.1", 8054, url, False, False)

    def test_python_usage_serve_loads_user_csv_path(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            data_path = Path(tmp_dir) / "dummy.csv"
            data_path.write_text("x,Actual\n1,10\n2,20\n", encoding="utf-8")
            stdout = io.StringIO()

            with (
                patch("py_lucidum.cli._start_app_server") as start_server_mock,
                redirect_stdout(stdout),
            ):
                url = serve(data_path, token="", port=8052, open_browser=False)
            app = start_server_mock.call_args.args[0]
            schema = app.state.dataset.schema()

            self.assertEqual(url, "http://127.0.0.1:8052/")
            self.assertIn(f"py_lucidum serving {data_path.resolve()}", stdout.getvalue())
            start_server_mock.assert_called_once()
            self.assertEqual([column["name"] for column in schema["columns"]], ["x", "Actual"])
            self.assertEqual(schema["row_count"], 2)

    def test_main_reports_runtime_error_without_traceback(self) -> None:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.bind(("127.0.0.1", 0))
            sock.listen(1)
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

    def test_main_reports_missing_dataset_without_traceback(self) -> None:
        stdout = io.StringIO()
        stderr = io.StringIO()

        with (
            patch("sys.argv", ["lucidum", "missing.parquet"]),
            redirect_stdout(stdout),
            redirect_stderr(stderr),
            self.assertRaises(SystemExit) as exit_context,
        ):
            main()

        self.assertEqual(exit_context.exception.code, 1)
        self.assertEqual(stdout.getvalue(), "")
        self.assertIn("lucidum: error: Dataset does not exist:", stderr.getvalue())
        self.assertNotIn("Traceback", stderr.getvalue())

    def test_main_requires_path_or_demo(self) -> None:
        stdout = io.StringIO()
        stderr = io.StringIO()

        with (
            patch("sys.argv", ["lucidum"]),
            redirect_stdout(stdout),
            redirect_stderr(stderr),
            self.assertRaises(SystemExit) as exit_context,
        ):
            main()

        self.assertEqual(exit_context.exception.code, 2)
        self.assertEqual(stdout.getvalue(), "")
        self.assertIn("path or --demo", stderr.getvalue())

    def test_main_rejects_demo_with_path(self) -> None:
        stdout = io.StringIO()
        stderr = io.StringIO()

        with (
            patch("sys.argv", ["lucidum", "data.parquet", "--demo"]),
            redirect_stdout(stdout),
            redirect_stderr(stderr),
            self.assertRaises(SystemExit) as exit_context,
        ):
            main()

        self.assertEqual(exit_context.exception.code, 2)
        self.assertEqual(stdout.getvalue(), "")
        self.assertIn("choose either a dataset path or --demo", stderr.getvalue())

    def test_main_uses_demo_dataset_path(self) -> None:
        demo_path = Path("/tmp/py-lucidum-demo.parquet")

        with (
            patch("sys.argv", ["lucidum", "--demo", "--port", "8050"]),
            patch("py_lucidum.cli.demo_dataset_path", return_value=demo_path) as demo_path_mock,
            patch("py_lucidum.cli.serve", return_value="http://127.0.0.1:8050/") as serve_mock,
        ):
            result = main()

        self.assertEqual(result, 0)
        demo_path_mock.assert_called_once_with()
        self.assertEqual(serve_mock.call_args.kwargs["path"], demo_path)
        self.assertEqual(serve_mock.call_args.kwargs["port"], 8050)

    def test_readme_quick_start_and_common_option_launches_are_wired(self) -> None:
        demo_path = Path("/tmp/py-lucidum-demo.parquet")
        cases = [
            (
                "quick_start_demo",
                ["lucidum", "--demo", "--port", "8000"],
                {"path": demo_path, "port": 8000},
                True,
            ),
            (
                "quick_start_source_demo_path",
                ["lucidum", "datasets/motor_premiums.parquet", "--port", "8000"],
                {"path": "datasets/motor_premiums.parquet", "port": 8000},
                False,
            ),
            (
                "open_browser",
                ["lucidum", "--demo", "--open", "--port", "8000"],
                {"path": demo_path, "open_browser": True, "port": 8000},
                True,
            ),
            (
                "bind_host",
                ["lucidum", "--demo", "--host", "0.0.0.0", "--port", "8000"],
                {"path": demo_path, "host": "0.0.0.0", "port": 8000},
                True,
            ),
            (
                "no_token",
                ["lucidum", "--demo", "--no-token"],
                {"path": demo_path, "token": ""},
                True,
            ),
            (
                "initial_selections",
                [
                    "lucidum",
                    "--demo",
                    "--x",
                    "DRIVER_AGE",
                    "--actual",
                    "PREMIUM",
                    "--denominator",
                    "ANNUAL_MILEAGE",
                ],
                {
                    "path": demo_path,
                    "x": "DRIVER_AGE",
                    "actual": "PREMIUM",
                    "denominator": "ANNUAL_MILEAGE",
                },
                True,
            ),
            (
                "filters",
                ["lucidum", "--demo", "--filters", "specs/filter_spec.csv"],
                {"path": demo_path, "filters": "specs/filter_spec.csv"},
                True,
            ),
            (
                "no_filters",
                ["lucidum", "--demo", "--no-filters"],
                {"path": demo_path, "no_filters": True},
                True,
            ),
            (
                "kpis",
                ["lucidum", "--demo", "--kpis", "specs/kpi_spec.csv"],
                {"path": demo_path, "kpis": "specs/kpi_spec.csv"},
                True,
            ),
            (
                "no_kpis",
                ["lucidum", "--demo", "--no-kpis"],
                {"path": demo_path, "no_kpis": True},
                True,
            ),
            (
                "line_bar_tool",
                ["lucidum", "--demo", "--tools", "line-bar"],
                {"path": demo_path, "tools": "line-bar"},
                True,
            ),
        ]

        for name, argv, expected_kwargs, expects_demo_path in cases:
            with self.subTest(name=name):
                with (
                    patch("sys.argv", argv),
                    patch("py_lucidum.cli.demo_dataset_path", return_value=demo_path) as demo_path_mock,
                    patch("py_lucidum.cli.serve", return_value="http://127.0.0.1:8000/") as serve_mock,
                ):
                    result = main()

                self.assertEqual(result, 0)
                if expects_demo_path:
                    demo_path_mock.assert_called_once_with()
                else:
                    demo_path_mock.assert_not_called()
                for key, value in expected_kwargs.items():
                    self.assertEqual(serve_mock.call_args.kwargs[key], value)

    def test_main_passes_regular_file_path_without_demo_rewrite(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            data_path = Path(tmp_dir) / "ordinary.csv"
            data_path.write_text("x,y\n1,2\n", encoding="utf-8")

            with (
                patch("sys.argv", ["lucidum", str(data_path), "--port", "8051"]),
                patch("py_lucidum.cli.demo_dataset_path") as demo_path_mock,
                patch("py_lucidum.cli.serve", return_value="http://127.0.0.1:8051/") as serve_mock,
            ):
                result = main()

        self.assertEqual(result, 0)
        demo_path_mock.assert_not_called()
        self.assertEqual(serve_mock.call_args.kwargs["path"], str(data_path))
        self.assertEqual(serve_mock.call_args.kwargs["port"], 8051)

    def test_main_reports_unknown_tool_without_traceback(self) -> None:
        stdout = io.StringIO()
        stderr = io.StringIO()

        with (
            patch("sys.argv", ["lucidum", "missing.parquet", "--tools", "not-a-tool"]),
            redirect_stdout(stdout),
            redirect_stderr(stderr),
            self.assertRaises(SystemExit) as exit_context,
        ):
            main()

        self.assertEqual(exit_context.exception.code, 1)
        self.assertEqual(stdout.getvalue(), "")
        self.assertIn("lucidum: error: Unknown tool 'not-a-tool'", stderr.getvalue())
        self.assertNotIn("Traceback", stderr.getvalue())

    def test_main_reports_missing_filter_spec_without_traceback_or_startup_output(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            data_path = root / "sample.csv"
            missing_filters_path = root / "missing_filter_spec.csv"
            data_path.write_text("x,y\n1,2\n", encoding="utf-8")
            stdout = io.StringIO()
            stderr = io.StringIO()

            with (
                patch("sys.argv", ["lucidum", str(data_path), "--filters", str(missing_filters_path)]),
                redirect_stdout(stdout),
                redirect_stderr(stderr),
                self.assertRaises(SystemExit) as exit_context,
            ):
                main()

        self.assertEqual(exit_context.exception.code, 1)
        self.assertEqual(stdout.getvalue(), "")
        self.assertIn("lucidum: error: Filter specification file does not exist:", stderr.getvalue())
        self.assertNotIn("Traceback", stderr.getvalue())

    def test_main_reports_missing_kpi_spec_without_traceback_or_startup_output(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            data_path = root / "sample.csv"
            missing_kpis_path = root / "missing_kpi_spec.csv"
            data_path.write_text("x,y\n1,2\n", encoding="utf-8")
            stdout = io.StringIO()
            stderr = io.StringIO()

            with (
                patch("sys.argv", ["lucidum", str(data_path), "--kpis", str(missing_kpis_path)]),
                redirect_stdout(stdout),
                redirect_stderr(stderr),
                self.assertRaises(SystemExit) as exit_context,
            ):
                main()

        self.assertEqual(exit_context.exception.code, 1)
        self.assertEqual(stdout.getvalue(), "")
        self.assertIn("lucidum: error: KPI specification file does not exist:", stderr.getvalue())
        self.assertNotIn("Traceback", stderr.getvalue())


class AsyncCliRuntimeTests(unittest.IsolatedAsyncioTestCase):
    async def test_run_server_uses_background_thread_with_running_event_loop(self) -> None:
        server = FakeServer()

        _run_server(server)

        self.assertTrue(server.event.wait(timeout=2))
        self.assertEqual(server.thread_name, "py-lucidum-uvicorn")


if __name__ == "__main__":
    unittest.main()
