from __future__ import annotations

import argparse
import asyncio
import logging
import secrets
import socket
import threading
import webbrowser
from collections.abc import Sequence
from pathlib import Path
from urllib.parse import urlencode

import uvicorn

from .app import create_app


class LucidumServer(uvicorn.Server):
    def __init__(self, config: uvicorn.Config, display_url: str, stop_instruction: str) -> None:
        super().__init__(config)
        self.display_url = display_url
        self.stop_instruction = stop_instruction

    def _log_started_message(self, listeners: Sequence[socket.SocketType]) -> None:
        message = f"Uvicorn running on {self.display_url} ({self.stop_instruction})"
        logging.getLogger("uvicorn.error").info(
            message,
            extra={"color_message": message},
        )


def find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def ensure_port_available(host: str, port: int) -> None:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind((host, port))
        except OSError as exc:
            raise RuntimeError(
                f"Port {port} is already in use on {host}. Stop the existing py_lucidum app or choose another port."
            ) from exc


def _has_running_event_loop() -> bool:
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return False
    return True


def _run_server(server: uvicorn.Server, run_in_background: bool | None = None) -> None:
    if run_in_background is None:
        run_in_background = _has_running_event_loop()

    if run_in_background:
        thread = threading.Thread(target=server.run, name="py-lucidum-uvicorn", daemon=True)
        thread.start()
        return

    try:
        server.run()
    except KeyboardInterrupt:
        pass


def _display_url_for_app(app: object, host: str, port: int) -> str:
    url = f"http://{host}:{port}/"
    state = getattr(app, "state", None)
    token = getattr(state, "token", None)
    defaults = getattr(state, "defaults", {})
    params = {}
    if token:
        params["token"] = token
    if isinstance(defaults, dict):
        params.update({
            key: value
            for key, value in defaults.items()
            if key in {"x", "actual", "expected", "denominator", "postcode_area", "postcode_sector"} and value
        })
    if params:
        return f"{url}?{urlencode(params)}"
    return url


def _stop_instruction(run_in_background: bool) -> str:
    if run_in_background:
        return "Use the app Stop app button to quit"
    return "Press CTRL+C to quit"


def _print_stop_status(run_in_background: bool) -> None:
    if run_in_background:
        print("lucidum is running in the background. Use the app Stop app button to stop it.", flush=True)
    else:
        print("lucidum is still running until you press Ctrl+C in this terminal.", flush=True)


def _start_app_server(
    app: object,
    host: str,
    port: int,
    url: str,
    open_browser: bool,
    run_in_background: bool,
) -> None:
    ensure_port_available(host, port)
    if open_browser:
        webbrowser.open(url)
    config = uvicorn.Config(app, host=host, port=port, log_level="info", access_log=False)
    server = LucidumServer(config, url, _stop_instruction(run_in_background))
    state = getattr(app, "state", None)
    if state is not None:
        state.shutdown_callback = lambda: setattr(server, "should_exit", True)
    _run_server(server, run_in_background=run_in_background)


def run_app(
    app: object,
    host: str = "127.0.0.1",
    port: int | None = 8000,
    open_browser: bool = False,
    url: str | None = None,
) -> str:
    selected_port = port or find_free_port()
    ensure_port_available(host, selected_port)
    display_url = url or _display_url_for_app(app, host, selected_port)
    run_in_background = _has_running_event_loop()
    print(f"Open {display_url}", flush=True)
    _print_stop_status(run_in_background)
    _start_app_server(app, host, selected_port, display_url, open_browser, run_in_background)
    return display_url


def serve(
    path: str | Path,
    host: str = "127.0.0.1",
    port: int | None = None,
    token: str | None = None,
    open_browser: bool = False,
    x: str | None = None,
    actual: str | None = None,
    expected: str | None = None,
    denominator: str | None = None,
    postcode_area: str | None = None,
    postcode_sector: str | None = None,
    filters: str | Path | None = None,
    no_filters: bool = False,
    tools: str | Sequence[str] | None = None,
) -> str:
    selected_port = port or find_free_port()
    ensure_port_available(host, selected_port)
    selected_token = token if token is not None else secrets.token_urlsafe(18)
    defaults = {
        "x": x,
        "actual": actual,
        "expected": expected,
        "denominator": denominator,
        "postcode_area": postcode_area,
        "postcode_sector": postcode_sector,
    }
    app = create_app(
        path,
        token=selected_token,
        defaults=defaults,
        filters_path=filters,
        use_saved_filters=not no_filters,
        tools=tools,
    )
    url = _display_url_for_app(app, host, selected_port)
    run_in_background = _has_running_event_loop()
    print(f"py_lucidum serving {Path(path).resolve()}", flush=True)
    print(f"Open {url}", flush=True)
    print(f"Saved filters: {saved_filters_status(app)}", flush=True)
    _print_stop_status(run_in_background)
    _start_app_server(app, host, selected_port, url, open_browser, run_in_background)
    return url


def serve_line_bar(
    path: str | Path,
    host: str = "127.0.0.1",
    port: int | None = None,
    token: str | None = None,
    open_browser: bool = False,
    x: str | None = None,
    actual: str | None = None,
    expected: str | None = None,
    denominator: str | None = None,
    postcode_area: str | None = None,
    postcode_sector: str | None = None,
    filters: str | Path | None = None,
    no_filters: bool = False,
) -> str:
    return serve(
        path=path,
        host=host,
        port=port,
        token=token,
        open_browser=open_browser,
        x=x,
        actual=actual,
        expected=expected,
        denominator=denominator,
        postcode_area=postcode_area,
        postcode_sector=postcode_sector,
        filters=filters,
        no_filters=no_filters,
        tools=["line_bar"],
    )


def saved_filters_status(app: object) -> str:
    state = getattr(app, "state")
    if not getattr(state, "use_saved_filters", True):
        return "disabled"
    path = getattr(state, "resolved_filters_path", None)
    if not path or not Path(path).exists():
        return "none"
    resolved = Path(path)
    try:
        return str(resolved.relative_to(Path.cwd()))
    except ValueError:
        return str(resolved)


def main() -> int:
    parser = argparse.ArgumentParser(description="Launch py_lucidum for a local CSV or Parquet file.")
    parser.add_argument("path", help="Path to a CSV or Parquet file")
    parser.add_argument("--host", default="127.0.0.1", help="Bind host, e.g. 127.0.0.1 or 0.0.0.0")
    parser.add_argument("--port", type=int, default=None, help="Bind port. Defaults to a free local port.")
    parser.add_argument("--no-token", action="store_true", help="Disable the token in the URL and API requests")
    parser.add_argument("--open", action="store_true", help="Open the app with Python's configured browser/viewer")
    parser.add_argument("--x", default=None, help="Initial x-axis feature. Defaults to the first dataset column.")
    parser.add_argument("--actual", default=None, help="Initial Actual / line 1 numeric feature. Defaults to the first numeric column.")
    parser.add_argument("--expected", default=None, help="Initial Expected / line 2 numeric feature. Defaults to None.")
    parser.add_argument("--denominator", default=None, help="Initial Weight column. Defaults to Average row value.")
    parser.add_argument("--postcode-area", default=None, help="Postcode area column for UK mapping. Defaults to PostcodeArea.")
    parser.add_argument("--postcode-sector", default=None, help="Postcode sector column for UK mapping. Defaults to PostcodeSector.")
    filter_group = parser.add_mutually_exclusive_group()
    filter_group.add_argument(
        "--filters",
        default=None,
        help="Path to filter_spec.csv. Defaults to ./filter_spec.csv, then ./specs/filter_spec.csv when present.",
    )
    filter_group.add_argument(
        "--no-filters",
        action="store_true",
        help="Disable saved filters and skip default filter_spec.csv discovery.",
    )
    parser.add_argument("--tools", default=None, help="Comma-separated tools to enable. Supports line-bar and uk-map.")
    args = parser.parse_args()
    try:
        serve(
            path=args.path,
            host=args.host,
            port=args.port,
            token="" if args.no_token else secrets.token_urlsafe(18),
            open_browser=args.open,
            x=args.x,
            actual=args.actual,
            expected=args.expected,
            denominator=args.denominator,
            postcode_area=args.postcode_area,
            postcode_sector=args.postcode_sector,
            filters=args.filters,
            no_filters=args.no_filters,
            tools=args.tools,
        )
    except (RuntimeError, ValueError, OSError) as error:
        parser.exit(1, f"lucidum: error: {error}\n")
    return 0
