from __future__ import annotations

import argparse
import secrets
import socket
import webbrowser
from collections.abc import Sequence
from pathlib import Path
from urllib.parse import urlencode

import uvicorn

from .app import create_app


def find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def serve(
    path: str | Path,
    host: str = "127.0.0.1",
    port: int | None = None,
    token: str | None = None,
    open_browser: bool = False,
    x: str | None = None,
    actual: str | None = None,
    expected: str | None = None,
    filters: str | Path | None = None,
    tools: str | Sequence[str] | None = None,
) -> str:
    selected_port = port or find_free_port()
    selected_token = token if token is not None else secrets.token_urlsafe(18)
    defaults = {"x": x, "actual": actual, "expected": expected}
    app = create_app(path, token=selected_token, defaults=defaults, filters_path=filters, tools=tools)
    url = f"http://{host}:{selected_port}/"
    params = {key: value for key, value in defaults.items() if value}
    if selected_token:
        params = {"token": selected_token, **params}
    if params:
        url = f"{url}?{urlencode(params)}"
    print(f"py_lucidum serving {Path(path).resolve()}", flush=True)
    print(f"Open {url}", flush=True)
    if open_browser:
        webbrowser.open(url)
    uvicorn.run(app, host=host, port=selected_port, log_level="info")
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
    filters: str | Path | None = None,
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
        filters=filters,
        tools=["line_bar"],
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Launch py_lucidum for a local CSV or Parquet file.")
    parser.add_argument("path", help="Path to a CSV or Parquet file")
    parser.add_argument("--host", default="127.0.0.1", help="Bind host, e.g. 127.0.0.1 or 0.0.0.0")
    parser.add_argument("--port", type=int, default=None, help="Bind port. Defaults to a free local port.")
    parser.add_argument("--no-token", action="store_true", help="Disable the token in the URL and API requests")
    parser.add_argument("--open", action="store_true", help="Open the app in the default browser")
    parser.add_argument("--x", default=None, help="Initial x-axis feature. Defaults to the first dataset column.")
    parser.add_argument("--actual", default=None, help="Initial Actual / line 1 numeric feature. Defaults to the first numeric column.")
    parser.add_argument("--expected", default=None, help="Initial Expected / line 2 numeric feature. Defaults to None.")
    parser.add_argument("--filters", default=None, help="Path to filter_spec.csv. Defaults to ./filter_spec.csv when present.")
    parser.add_argument("--tools", default=None, help="Comma-separated tools to enable. Currently supports line-bar.")
    args = parser.parse_args()
    serve(
        path=args.path,
        host=args.host,
        port=args.port,
        token="" if args.no_token else secrets.token_urlsafe(18),
        open_browser=args.open,
        x=args.x,
        actual=args.actual,
        expected=args.expected,
        filters=args.filters,
        tools=args.tools,
    )
