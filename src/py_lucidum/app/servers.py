from __future__ import annotations

import os
import threading
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

import psutil


LUCIDUM_COMMAND_NAMES = ("lucidum", "py-lucidum")
LUCIDUM_MODULE_NAMES = ("py_lucidum", "py-lucidum")
CREATE_TIME_TOLERANCE_SECONDS = 0.01


class ServerStopError(Exception):
    def __init__(self, message: str, status_code: int = 400) -> None:
        super().__init__(message)
        self.status_code = status_code


def safe_display_url(url: str | None) -> str:
    if not url:
        return ""
    split = urlsplit(str(url))
    query = urlencode([
        (key, value)
        for key, value in parse_qsl(split.query, keep_blank_values=True)
        if key.lower() != "token"
    ])
    return urlunsplit((split.scheme, split.netloc, split.path, query, split.fragment))


def process_value(process: Any, key: str, default: Any = None) -> Any:
    info = getattr(process, "info", None)
    if isinstance(info, dict) and key in info:
        return info[key]
    value = getattr(process, key, default)
    if callable(value):
        try:
            return value()
        except (psutil.Error, OSError):
            return default
    return value


def current_process() -> psutil.Process:
    return psutil.Process(os.getpid())


def process_username(process: Any) -> str:
    return str(process_value(process, "username", "") or "")


def process_pid(process: Any) -> int:
    return int(process_value(process, "pid", 0) or 0)


def process_create_time(process: Any) -> float:
    return float(process_value(process, "create_time", 0.0) or 0.0)


def process_name(process: Any) -> str:
    return str(process_value(process, "name", "") or "")


def process_cmdline(process: Any) -> list[str]:
    raw = process_value(process, "cmdline", []) or []
    return [str(part) for part in raw]


def redacted_command_part(part: str) -> str:
    if "token=" in part.lower():
        return safe_display_url(part)
    if len(part) > 90:
        return f"{part[:87]}..."
    return part


def command_label(process: Any) -> str:
    parts = [redacted_command_part(part) for part in process_cmdline(process)]
    if parts:
        first = Path(parts[0]).name or parts[0]
        label = " ".join([first, *parts[1:5]])
    else:
        label = process_name(process)
    return label[:180] if label else "Current process"


def looks_like_lucidum_process(process: Any) -> bool:
    name = Path(process_name(process).lower()).stem
    if name in LUCIDUM_COMMAND_NAMES:
        return True

    parts = process_cmdline(process)
    for index, part in enumerate(parts):
        lower = part.lower()
        base = Path(lower).name
        stem = Path(base).stem
        previous = parts[index - 1].lower() if index else ""

        if stem in LUCIDUM_COMMAND_NAMES:
            return True
        if previous == "-m" and (
            lower in LUCIDUM_MODULE_NAMES
            or lower.startswith("py_lucidum.")
            or lower.startswith("py-lucidum.")
        ):
            return True
        if "/py_lucidum/" in lower and base in {"__main__.py", "cli.py"}:
            return True
    return False


def connection_address(connection: Any) -> dict[str, Any] | None:
    if str(getattr(connection, "status", "")).upper() != "LISTEN":
        return None
    laddr = getattr(connection, "laddr", None)
    host = getattr(laddr, "ip", None)
    port = getattr(laddr, "port", None)
    if host is None and isinstance(laddr, tuple) and len(laddr) >= 2:
        host, port = laddr[0], laddr[1]
    try:
        port_number = int(port)
    except (TypeError, ValueError):
        return None
    return {"host": str(host or ""), "port": port_number}


def process_listeners(process: Any) -> list[dict[str, Any]]:
    try:
        connections = process.net_connections(kind="inet")
    except AttributeError:
        try:
            connections = process.connections(kind="inet")
        except (psutil.Error, OSError):
            return []
    except (psutil.Error, OSError):
        return []
    listeners = [address for connection in connections if (address := connection_address(connection))]
    unique = {(listener["host"], listener["port"]): listener for listener in listeners}
    return sorted(unique.values(), key=lambda item: (item["host"], item["port"]))


def metadata_listener(metadata: dict[str, Any]) -> dict[str, Any] | None:
    port = metadata.get("port")
    if port is None:
        return None
    try:
        port_number = int(port)
    except (TypeError, ValueError):
        return None
    return {"host": str(metadata.get("host") or ""), "port": port_number}


def add_metadata_listener(listeners: list[dict[str, Any]], metadata: dict[str, Any]) -> list[dict[str, Any]]:
    listener = metadata_listener(metadata)
    if not listener:
        return listeners
    merged = {(item["host"], item["port"]): item for item in listeners}
    merged[(listener["host"], listener["port"])] = listener
    return sorted(merged.values(), key=lambda item: (item["host"], item["port"]))


def server_record(process: Any, *, current: bool, metadata: dict[str, Any] | None = None, shutdown_available: bool = False) -> dict[str, Any]:
    meta = metadata or {}
    listeners = process_listeners(process)
    if current:
        listeners = add_metadata_listener(listeners, meta)
    dataset_path = str(meta.get("dataset_path") or "")
    dataset_name = str(meta.get("dataset_name") or (Path(dataset_path).name if dataset_path else ""))
    display_url = safe_display_url(str(meta.get("display_url") or ""))
    return {
        "pid": process_pid(process),
        "create_time": process_create_time(process),
        "current": current,
        "stoppable": bool(shutdown_available if current else True),
        "command": command_label(process),
        "dataset": dataset_name,
        "dataset_path": dataset_path,
        "display_url": display_url,
        "listeners": listeners,
    }


def lucidum_server_metadata(state: Any) -> dict[str, Any]:
    metadata = getattr(state, "lucidum_server_metadata", None)
    return dict(metadata) if isinstance(metadata, dict) else {}


def list_lucidum_servers(
    state: Any,
    *,
    processes: Iterable[Any] | None = None,
    current: Any | None = None,
) -> list[dict[str, Any]]:
    current_proc = current or current_process()
    current_pid = process_pid(current_proc)
    current_user = process_username(current_proc)
    metadata = lucidum_server_metadata(state)
    shutdown_available = callable(getattr(state, "shutdown_callback", None))
    records = [
        server_record(
            current_proc,
            current=True,
            metadata=metadata,
            shutdown_available=shutdown_available,
        )
    ]
    seen = {current_pid}
    source = processes if processes is not None else psutil.process_iter(["pid", "name", "cmdline", "username", "create_time"])
    for process in source:
        try:
            pid = process_pid(process)
            if not pid or pid in seen:
                continue
            if current_user and process_username(process) != current_user:
                continue
            if not looks_like_lucidum_process(process):
                continue
            records.append(server_record(process, current=False))
            seen.add(pid)
        except (psutil.Error, OSError, TypeError, ValueError):
            continue
    records.sort(key=lambda item: (not item["current"], item["pid"]))
    return records


def create_time_matches(process: Any, expected: float) -> bool:
    actual = process_create_time(process)
    return bool(actual) and abs(actual - float(expected)) <= CREATE_TIME_TOLERANCE_SECONDS


def stop_lucidum_server(
    state: Any,
    pid: int,
    create_time: float,
    *,
    process_factory: Any = psutil.Process,
    current: Any | None = None,
) -> dict[str, Any]:
    target_pid = int(pid)
    current_proc = current or current_process()
    current_pid = process_pid(current_proc)
    try:
        target = process_factory(target_pid)
    except (psutil.Error, OSError) as exc:
        raise ServerStopError("Lucidum server process was not found", 404) from exc
    if not create_time_matches(target, float(create_time)):
        raise ServerStopError("Lucidum server process no longer matches the selected process", 404)

    if target_pid == current_pid:
        shutdown_callback = getattr(state, "shutdown_callback", None)
        if not callable(shutdown_callback):
            raise ServerStopError("Shutdown is only available when launched with the lucidum command", 503)
        threading.Timer(0.2, shutdown_callback).start()
        return {"message": "Current lucidum server is stopping", "pid": target_pid}

    if process_username(target) != process_username(current_proc):
        raise ServerStopError("Cannot stop a lucidum server owned by another user", 403)
    if not looks_like_lucidum_process(target):
        raise ServerStopError("Target process is not a local lucidum server", 400)
    target.terminate()
    return {"message": "Lucidum server is stopping", "pid": target_pid}
