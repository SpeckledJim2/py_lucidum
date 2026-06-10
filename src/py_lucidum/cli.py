from __future__ import annotations

import argparse
import asyncio
import ipaddress
import secrets
import socket
import threading
import webbrowser
from collections.abc import Callable, Sequence
from pathlib import Path
from urllib.parse import urlencode

import uvicorn

from .app import create_app
from .app.servers import safe_display_url
from .demo import demo_dataset_path


DEFAULT_URL_KEYS = {
    "x",
    "actual",
    "expected",
    "denominator",
    "postcode_area",
    "postcode_sector",
    "postcode_unit",
    "latitude",
    "longitude",
    "source",
}


class LucidumServer(uvicorn.Server):
    def __init__(
        self,
        config: uvicorn.Config,
        display_url: str,
        stop_instruction: str,
        open_browser: bool = False,
        browser_opener: Callable[[str], object] | None = None,
    ) -> None:
        super().__init__(config)
        self.display_url = display_url
        self.stop_instruction = stop_instruction
        self.open_browser = open_browser
        self.browser_opener = browser_opener or webbrowser.open

    def _log_started_message(self, listeners: Sequence[socket.SocketType]) -> None:
        if self.open_browser:
            self.browser_opener(self.display_url)


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


def _query_string_for_app(app: object) -> str:
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
            if key in DEFAULT_URL_KEYS and value
        })
    return urlencode(params)


def _url_for_app(app: object, host: str, port: int) -> str:
    url = f"http://{host}:{port}/"
    query_string = _query_string_for_app(app)
    if query_string:
        return f"{url}?{query_string}"
    return url


def _is_wildcard_host(host: str) -> bool:
    return host in {"0.0.0.0", "::", "[::]"}


def _usable_lan_ipv4(address: str) -> str | None:
    try:
        ip = ipaddress.ip_address(address)
    except ValueError:
        return None
    if ip.version != 4 or ip.is_loopback or ip.is_unspecified:
        return None
    return str(ip)


def _detect_primary_lan_ipv4() -> str | None:
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
            sock.connect(("8.8.8.8", 80))
            address = sock.getsockname()[0]
    except OSError:
        return None
    return _usable_lan_ipv4(address)


def _display_url_for_app(app: object, host: str, port: int) -> str:
    display_host = "127.0.0.1" if _is_wildcard_host(host) else host
    return _url_for_app(app, display_host, port)


def _lan_url_hint_for_app(app: object, host: str, port: int) -> str | None:
    if not _is_wildcard_host(host):
        return None
    return _url_for_app(app, _detect_primary_lan_ipv4() or "<this-computer-ip>", port)


def _print_open_urls(app: object, host: str, port: int, display_url: str) -> None:
    lan_url = _lan_url_hint_for_app(app, host, port)
    if not lan_url:
        print(f"Open {display_url}", flush=True)
        return
    print(f"Open locally {display_url}", flush=True)
    print(f"Open from another device on your LAN: {lan_url}", flush=True)


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
    config = uvicorn.Config(app, host=host, port=port, log_level="warning", access_log=False)
    server = LucidumServer(config, url, _stop_instruction(run_in_background), open_browser=open_browser)
    state = getattr(app, "state", None)
    if state is not None:
        state.shutdown_callback = lambda: setattr(server, "should_exit", True)
        metadata = getattr(state, "lucidum_server_metadata", None)
        if not isinstance(metadata, dict):
            metadata = {}
            state.lucidum_server_metadata = metadata
        metadata.update({
            "host": host,
            "port": port,
            "display_url": safe_display_url(url),
        })
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
    if url:
        print(f"Open {display_url}", flush=True)
    else:
        _print_open_urls(app, host, selected_port, display_url)
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
    postcode_unit: str | None = None,
    latitude: str | None = None,
    longitude: str | None = None,
    filters: str | Path | None = None,
    no_filters: bool = False,
    kpis: str | Path | None = None,
    kpis_path: str | Path | None = None,
    no_kpis: bool = False,
    use_kpis: bool = True,
    features: str | Path | None = None,
    features_path: str | Path | None = None,
    no_features: bool = False,
    use_features: bool = True,
    tools: str | Sequence[str] | None = None,
) -> str:
    selected_port = port or find_free_port()
    ensure_port_available(host, selected_port)
    if kpis and kpis_path and Path(kpis).expanduser() != Path(kpis_path).expanduser():
        raise ValueError("Specify either kpis or kpis_path, not both")
    if features and features_path and Path(features).expanduser() != Path(features_path).expanduser():
        raise ValueError("Specify either features or features_path, not both")
    selected_kpis_path = kpis_path or kpis
    kpis_enabled = use_kpis and not no_kpis
    selected_features_path = features_path or features
    features_enabled = use_features and not no_features
    selected_token = token if token is not None else secrets.token_urlsafe(18)
    defaults = {
        "x": x,
        "actual": actual,
        "expected": expected,
        "denominator": denominator,
        "postcode_area": postcode_area,
        "postcode_sector": postcode_sector,
        "postcode_unit": postcode_unit,
        "latitude": latitude,
        "longitude": longitude,
    }
    app = create_app(
        path,
        token=selected_token,
        defaults=defaults,
        filters_path=filters,
        use_saved_filters=not no_filters,
        tools=tools,
        kpis_path=selected_kpis_path,
        use_kpis=kpis_enabled,
        features_path=selected_features_path,
        use_features=features_enabled,
    )
    url = _display_url_for_app(app, host, selected_port)
    run_in_background = _has_running_event_loop()
    print(f"lucidum serving {Path(path).resolve()}", flush=True)
    _print_open_urls(app, host, selected_port, url)
    print(f"Saved filters: {saved_filters_status(app)}", flush=True)
    print(f"KPIs: {kpis_status(app)}", flush=True)
    print(f"Feature specs: {features_status(app)}", flush=True)
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
    postcode_unit: str | None = None,
    latitude: str | None = None,
    longitude: str | None = None,
    filters: str | Path | None = None,
    no_filters: bool = False,
    kpis: str | Path | None = None,
    kpis_path: str | Path | None = None,
    no_kpis: bool = False,
    use_kpis: bool = True,
    features: str | Path | None = None,
    features_path: str | Path | None = None,
    no_features: bool = False,
    use_features: bool = True,
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
        postcode_unit=postcode_unit,
        latitude=latitude,
        longitude=longitude,
        filters=filters,
        no_filters=no_filters,
        kpis=kpis,
        kpis_path=kpis_path,
        no_kpis=no_kpis,
        use_kpis=use_kpis,
        features=features,
        features_path=features_path,
        no_features=no_features,
        use_features=use_features,
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


def kpis_status(app: object) -> str:
    state = getattr(app, "state")
    if not getattr(state, "use_kpis", True):
        return "disabled"
    path = getattr(state, "resolved_kpis_path", None)
    if not path or not Path(path).exists():
        return "none"
    resolved = Path(path)
    try:
        return str(resolved.relative_to(Path.cwd()))
    except ValueError:
        return str(resolved)


def features_status(app: object) -> str:
    state = getattr(app, "state")
    if not getattr(state, "use_features", True):
        return "disabled"
    path = getattr(state, "resolved_features_path", None)
    if not path or not Path(path).exists():
        return "none"
    resolved = Path(path)
    try:
        return str(resolved.relative_to(Path.cwd()))
    except ValueError:
        return str(resolved)


def main() -> int:
    parser = argparse.ArgumentParser(description="Launch py_lucidum for a local CSV, Parquet, or bundled demo file.")
    parser.add_argument("path", nargs="?", help="Path to a CSV or Parquet file")
    parser.add_argument("--demo", action="store_true", help="Launch the bundled motor_premiums.parquet demo dataset")
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
    parser.add_argument("--postcode-unit", default=None, help="Postcode unit column for UK mapping points. Defaults to PostcodeUnit.")
    parser.add_argument("--latitude", default=None, help="Latitude column for UK mapping points. Defaults to lat.")
    parser.add_argument("--longitude", default=None, help="Longitude column for UK mapping points. Defaults to long.")
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
    kpi_group = parser.add_mutually_exclusive_group()
    kpi_group.add_argument(
        "--kpis",
        default=None,
        help="Path to kpi_spec.csv. Defaults to ./kpi_spec.csv, then ./specs/kpi_spec.csv when present.",
    )
    kpi_group.add_argument(
        "--no-kpis",
        action="store_true",
        help="Disable KPI specs and skip default kpi_spec.csv discovery.",
    )
    feature_group = parser.add_mutually_exclusive_group()
    feature_group.add_argument(
        "--features",
        default=None,
        help="Path to feature_spec.csv. Defaults to ./feature_spec.csv, then ./specs/feature_spec.csv when present.",
    )
    feature_group.add_argument(
        "--no-features",
        action="store_true",
        help="Disable feature specs and skip default feature_spec.csv discovery.",
    )
    parser.add_argument(
        "--tools",
        default=None,
        help="Comma-separated tools to enable alongside mandatory Column Profile. Supports column-profile, line-bar, uk-map, glm, gbm, specs, and models. Requesting gbm also enables glm.",
    )
    args = parser.parse_args()
    if args.demo and args.path:
        parser.error("choose either a dataset path or --demo, not both")
    if not args.demo and not args.path:
        parser.error("the following arguments are required: path or --demo")
    path = demo_dataset_path() if args.demo else args.path
    try:
        serve(
            path=path,
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
            postcode_unit=args.postcode_unit,
            latitude=args.latitude,
            longitude=args.longitude,
            filters=args.filters,
            no_filters=args.no_filters,
            kpis=args.kpis,
            no_kpis=args.no_kpis,
            features=args.features,
            no_features=args.no_features,
            tools=args.tools,
        )
    except (RuntimeError, ValueError, OSError) as error:
        parser.exit(1, f"lucidum: error: {error}\n")
    return 0
