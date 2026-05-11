from __future__ import annotations

import argparse
import secrets
import socket
import webbrowser
from pathlib import Path

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
) -> str:
    selected_port = port or find_free_port()
    selected_token = token if token is not None else secrets.token_urlsafe(18)
    app = create_app(path, token=selected_token)
    url = f"http://{host}:{selected_port}/"
    if selected_token:
        url = f"{url}?token={selected_token}"
    print(f"py_lucidum serving {Path(path).resolve()}", flush=True)
    print(f"Open {url}", flush=True)
    if open_browser:
        webbrowser.open(url)
    uvicorn.run(app, host=host, port=selected_port, log_level="info")
    return url


def main() -> None:
    parser = argparse.ArgumentParser(description="Launch py_lucidum for a local CSV or Parquet file.")
    parser.add_argument("path", help="Path to a CSV or Parquet file")
    parser.add_argument("--host", default="127.0.0.1", help="Bind host, e.g. 127.0.0.1 or 0.0.0.0")
    parser.add_argument("--port", type=int, default=None, help="Bind port. Defaults to a free local port.")
    parser.add_argument("--no-token", action="store_true", help="Disable the token in the URL and API requests")
    parser.add_argument("--open", action="store_true", help="Open the app in the default browser")
    args = parser.parse_args()
    serve(
        path=args.path,
        host=args.host,
        port=args.port,
        token=None if args.no_token else secrets.token_urlsafe(18),
        open_browser=args.open,
    )
