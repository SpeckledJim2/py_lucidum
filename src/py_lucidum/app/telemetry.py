from __future__ import annotations

import re
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from threading import RLock
from typing import Any

import psutil


APP_ACTIONS = {
    ("GET", "/"): "Open app",
    ("GET", "/api/schema"): "Load schema",
    ("POST", "/api/banding/suggestion"): "Band suggestion",
    ("POST", "/api/date-bucket/suggestion"): "Date bucket suggestion",
    ("POST", "/api/filter/row-count"): "Filter row count",
    ("POST", "/api/metrics/summary"): "Metric summary",
    ("POST", "/api/reload"): "Reload dataset",
    ("POST", "/api/shutdown"): "Stop app",
    ("POST", "/api/lucidum-servers/stop"): "Stop Lucidum server",
    ("POST", "/api/column-profile/summary"): "Column profile summary",
    ("POST", "/api/column-profile/detail"): "Column profile detail",
    ("POST", "/api/chart"): "Line/bar chart",
    ("POST", "/api/line-bar/chart"): "Line/bar chart",
    ("POST", "/api/line-bar/table"): "Line/bar table",
    ("POST", "/api/uk-map/summary"): "UK map summary",
    ("GET", "/api/glm/summary"): "GLM summary",
    ("GET", "/api/glm/config"): "GLM config",
    ("GET", "/api/glm/models"): "GLM models",
    ("POST", "/api/glm/validate"): "GLM validate",
    ("POST", "/api/glm/build"): "GLM build",
    ("POST", "/api/glm/tabulations/build"): "GLM tabulation build",
    ("GET", "/api/glm/tabulations/jobs/{job_id}"): "GLM tabulation job",
    ("POST", "/api/glm/tabulations/config"): "GLM tabulation config",
    ("POST", "/api/glm/tabulations/table"): "GLM tabulation table",
    ("POST", "/api/glm/tabulations/plot"): "GLM tabulation plot",
    ("POST", "/api/glm/tabulations/export"): "GLM tabulation export",
    ("GET", "/api/gbm/summary"): "GBM summary",
    ("GET", "/api/gbm/config"): "GBM config",
    ("GET", "/api/gbm/models"): "GBM models",
    ("POST", "/api/gbm/validate"): "GBM validate",
    ("POST", "/api/gbm/train"): "GBM train",
}
HEARTBEAT_PATH = "/api/health"
EXCLUDED_PATHS = {"/api/telemetry", "/api/lucidum-servers"}
UNKNOWN_USER_AGENT = "(unknown)"
BYTES_PER_MB = 1024 * 1024


@dataclass
class TelemetryRequest:
    request_id: int
    client_key: str
    client_ip: str
    user_agent: str
    method: str
    path: str
    action: str | None
    action_label: str
    started_wall: float
    started_perf: float
    heartbeat: bool = False
    static_asset: bool = False


@dataclass
class ClientTelemetry:
    key: str
    client_ip: str
    user_agent: str
    first_seen: float
    last_seen: float
    request_count: int = 0
    app_action_count: int = 0
    error_count: int = 0
    last_status: int | None = None
    last_duration_ms: float | None = None
    last_path: str = ""
    last_app_action: str | None = None
    last_app_path: str | None = None
    current_actions: dict[int, dict[str, Any]] = field(default_factory=dict)


@dataclass
class HeartbeatTelemetry:
    count: int = 0
    error_count: int = 0
    last_seen: float | None = None
    last_status: int | None = None
    last_duration_ms: float | None = None


@dataclass
class DiagnosticTelemetry:
    count: int = 0
    error_count: int = 0
    last_seen: float | None = None
    last_status: int | None = None
    last_duration_ms: float | None = None
    last_path: str | None = None


def iso_timestamp(value: float) -> str:
    return datetime.fromtimestamp(value, timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def header_value(scope: dict[str, Any], name: str) -> str | None:
    name_bytes = name.lower().encode("latin-1")
    for key, value in scope.get("headers") or []:
        if key.lower() == name_bytes:
            return value.decode("latin-1", errors="replace")
    return None


def client_ip_from_scope(scope: dict[str, Any]) -> str:
    client = scope.get("client")
    if isinstance(client, tuple) and client:
        return str(client[0])
    return "unknown"


def app_action_for(method: str, path: str) -> str | None:
    return APP_ACTIONS.get((method.upper(), path))


def browser_version(user_agent: str, token: str, parts: int = 1) -> str | None:
    match = re.search(rf"{re.escape(token)}/([0-9]+(?:\.[0-9]+)*)", user_agent)
    if not match:
        return None
    return ".".join(match.group(1).split(".")[:parts])


def operating_system_label(user_agent: str) -> str | None:
    if "iPad" in user_agent:
        return "iPadOS"
    if "iPhone" in user_agent or "iPod" in user_agent:
        return "iOS"
    if "Android" in user_agent:
        return "Android"
    if "Macintosh" in user_agent or "Mac OS X" in user_agent:
        return "macOS"
    if "Windows" in user_agent:
        return "Windows"
    if "Linux" in user_agent:
        return "Linux"
    return None


def user_agent_label(user_agent: str) -> str:
    if not user_agent or user_agent == UNKNOWN_USER_AGENT:
        return "Unknown client"

    os_label = operating_system_label(user_agent)
    browser_label: str | None = None
    if version := browser_version(user_agent, "Edg"):
        browser_label = f"Edge {version}"
    elif version := browser_version(user_agent, "Firefox"):
        browser_label = f"Firefox {version}"
    elif version := browser_version(user_agent, "CriOS"):
        browser_label = f"Chrome {version}"
    elif version := browser_version(user_agent, "Chrome"):
        browser_label = f"Chrome {version}"
    elif "Safari/" in user_agent and "Chrome/" not in user_agent and "Chromium/" not in user_agent and "Edg/" not in user_agent:
        if os_label in {"iOS", "iPadOS"}:
            browser_label = "Safari"
        elif version := browser_version(user_agent, "Version", parts=2):
            browser_label = f"Safari {version}"
        else:
            browser_label = "Safari"

    if browser_label and os_label:
        return f"{browser_label} · {os_label}"
    if browser_label:
        return browser_label
    if os_label:
        return f"Unknown browser · {os_label}"
    return "Unknown client"


def is_static_asset_request(method: str, path: str) -> bool:
    if method.upper() not in {"GET", "HEAD"}:
        return False
    return path == "/favicon.ico" or path.startswith("/static/") or (path.startswith("/tools/") and "/static/" in path)


def bytes_to_mb(value: int) -> float:
    return round(value / BYTES_PER_MB, 1)


def process_memory_snapshot(process: psutil.Process) -> dict[str, Any]:
    memory = process.memory_info()
    try:
        full_memory = process.memory_full_info()
        uss_bytes = getattr(full_memory, "uss", None)
    except (psutil.AccessDenied, AttributeError):
        uss_bytes = None
    system_memory = psutil.virtual_memory()

    snapshot: dict[str, Any] = {
        "pid": process.pid,
        "rss_bytes": int(memory.rss),
        "rss_mb": bytes_to_mb(int(memory.rss)),
        "vms_bytes": int(memory.vms),
        "vms_mb": bytes_to_mb(int(memory.vms)),
        "memory_percent": round(process.memory_percent(), 2),
        "cpu_percent": round(process.cpu_percent(interval=None), 1),
        "system_cpu_percent": round(psutil.cpu_percent(interval=None), 1),
        "thread_count": process.num_threads(),
        "system_memory": {
            "total_bytes": int(system_memory.total),
            "total_mb": bytes_to_mb(int(system_memory.total)),
            "available_bytes": int(system_memory.available),
            "available_mb": bytes_to_mb(int(system_memory.available)),
            "used_percent": round(float(system_memory.percent), 1),
        },
    }
    if uss_bytes is not None:
        snapshot["uss_bytes"] = int(uss_bytes)
        snapshot["uss_mb"] = bytes_to_mb(int(uss_bytes))
    return snapshot


def recent_error_rate(events: deque[dict[str, Any]]) -> float:
    if not events:
        return 0.0
    error_count = sum(1 for event in events if event.get("error"))
    return round((error_count / len(events)) * 100, 1)


def slowest_recent_action(events: deque[dict[str, Any]]) -> dict[str, Any] | None:
    app_events = [event for event in events if event.get("app_action")]
    if not app_events:
        return None
    event = max(app_events, key=lambda item: float(item.get("duration_ms") or 0))
    return {
        "timestamp": event.get("timestamp"),
        "timestamp_unix": event.get("timestamp_unix"),
        "client_ip": event.get("client_ip"),
        "path": event.get("path"),
        "action": event.get("action"),
        "status": event.get("status"),
        "duration_ms": event.get("duration_ms"),
        "error": event.get("error"),
    }


class TelemetryStore:
    def __init__(
        self,
        active_window_seconds: float = 60,
        max_events: int = 200,
        max_clients: int = 200,
    ) -> None:
        self.active_window_seconds = active_window_seconds
        self.max_clients = max_clients
        self.started_at = time.time()
        self._lock = RLock()
        self._next_request_id = 1
        self._clients: dict[str, ClientTelemetry] = {}
        self._recent: deque[dict[str, Any]] = deque(maxlen=max_events)
        self._status_counts: defaultdict[int, int] = defaultdict(int)
        self._total_requests = 0
        self._app_actions = 0
        self._errors = 0
        self._process = psutil.Process()
        self._process.cpu_percent(interval=None)
        psutil.cpu_percent(interval=None)
        self._peak_rss_bytes = 0
        self._heartbeat = HeartbeatTelemetry()
        self._static_assets = DiagnosticTelemetry()

    def begin(self, scope: dict[str, Any]) -> TelemetryRequest | None:
        if scope.get("type") != "http":
            return None
        method = str(scope.get("method") or "GET").upper()
        path = str(scope.get("path") or "/")
        if path in EXCLUDED_PATHS:
            return None

        now = time.time()
        is_heartbeat = method == "GET" and path == HEARTBEAT_PATH
        is_static_asset = is_static_asset_request(method, path)
        client_ip = client_ip_from_scope(scope)
        user_agent = (header_value(scope, "user-agent") or UNKNOWN_USER_AGENT).strip() or UNKNOWN_USER_AGENT
        user_agent = user_agent[:240]
        client_key = f"{client_ip}\n{user_agent}"
        action = app_action_for(method, path)
        action_label = action or f"{method} {path}"

        with self._lock:
            request_id = self._next_request_id
            self._next_request_id += 1
            if is_heartbeat:
                return TelemetryRequest(
                    request_id=request_id,
                    client_key=client_key,
                    client_ip=client_ip,
                    user_agent=user_agent,
                    method=method,
                    path=path,
                    action=None,
                    action_label=action_label,
                    started_wall=now,
                    started_perf=time.perf_counter(),
                    heartbeat=True,
                )
            if is_static_asset:
                return TelemetryRequest(
                    request_id=request_id,
                    client_key=client_key,
                    client_ip=client_ip,
                    user_agent=user_agent,
                    method=method,
                    path=path,
                    action=None,
                    action_label=action_label,
                    started_wall=now,
                    started_perf=time.perf_counter(),
                    static_asset=True,
                )

            client = self._clients.get(client_key)
            if client is None:
                client = ClientTelemetry(
                    key=client_key,
                    client_ip=client_ip,
                    user_agent=user_agent,
                    first_seen=now,
                    last_seen=now,
                )
                self._clients[client_key] = client
                self._prune_clients_locked()

            self._total_requests += 1
            if action:
                self._app_actions += 1
                client.app_action_count += 1
            client.request_count += 1
            client.last_seen = now
            client.last_path = path
            client.current_actions[request_id] = {"label": action_label, "started_wall": now}

        return TelemetryRequest(
            request_id=request_id,
            client_key=client_key,
            client_ip=client_ip,
            user_agent=user_agent,
            method=method,
            path=path,
            action=action,
            action_label=action_label,
            started_wall=now,
            started_perf=time.perf_counter(),
        )

    def finish(self, request: TelemetryRequest | None, status_code: int | None, failed: bool = False) -> None:
        if request is None:
            return
        now = time.time()
        duration_ms = max(0.0, (time.perf_counter() - request.started_perf) * 1000)
        status = int(status_code or 500)
        is_error = failed or status >= 400

        with self._lock:
            if request.heartbeat:
                self._record_heartbeat_locked(now, status, duration_ms, is_error)
                return
            if request.static_asset:
                self._record_diagnostic_locked(self._static_assets, now, status, duration_ms, is_error, request.path)
                return

            self._status_counts[status] += 1
            if is_error:
                self._errors += 1
            client = self._clients.get(request.client_key)
            if client is not None:
                client.last_seen = now
                client.last_status = status
                client.last_duration_ms = round(duration_ms, 1)
                if request.action:
                    client.last_app_action = request.action
                    client.last_app_path = request.path
                if is_error:
                    client.error_count += 1
                client.current_actions.pop(request.request_id, None)

            self._recent.appendleft({
                "timestamp": iso_timestamp(now),
                "timestamp_unix": now,
                "client_ip": request.client_ip,
                "user_agent": request.user_agent,
                "method": request.method,
                "path": request.path,
                "action": request.action,
                "app_action": request.action is not None,
                "status": status,
                "duration_ms": round(duration_ms, 1),
                "error": is_error,
            })

    def snapshot(self) -> dict[str, Any]:
        now = time.time()
        process = process_memory_snapshot(self._process)
        with self._lock:
            rss_bytes = int(process.get("rss_bytes") or 0)
            self._peak_rss_bytes = max(self._peak_rss_bytes, rss_bytes)
            process["peak_rss_bytes"] = self._peak_rss_bytes
            process["peak_rss_mb"] = bytes_to_mb(self._peak_rss_bytes)
            clients = [self._client_snapshot(client, now) for client in self._clients.values()]
            clients.sort(key=lambda client: (client["in_flight"], client["last_seen_unix"]), reverse=True)
            in_flight = sum(client["in_flight"] for client in clients)
            active_clients = sum(1 for client in clients if client["active"])
            recent_error_count = sum(1 for event in self._recent if event.get("error"))
            return {
                "started_at": iso_timestamp(self.started_at),
                "started_at_unix": self.started_at,
                "now": iso_timestamp(now),
                "now_unix": now,
                "uptime_seconds": round(max(0.0, now - self.started_at), 1),
                "active_window_seconds": self.active_window_seconds,
                "totals": {
                    "total_requests": self._total_requests,
                    "app_actions": self._app_actions,
                    "errors": self._errors,
                    "in_flight": in_flight,
                    "active_clients": active_clients,
                    "clients": len(clients),
                    "status_counts": dict(sorted(self._status_counts.items())),
                    "recent_activity_limit": self._recent.maxlen,
                    "client_limit": self.max_clients,
                    "recent_error_count": recent_error_count,
                    "recent_error_rate": recent_error_rate(self._recent),
                    "slowest_recent_action": slowest_recent_action(self._recent),
                },
                "process": process,
                "clients": clients,
                "recent_activity": list(self._recent),
                "heartbeat": self._heartbeat_snapshot_locked(now),
                "diagnostics": {
                    "static_assets": self._diagnostic_snapshot_locked(self._static_assets, now),
                },
            }

    def _record_heartbeat_locked(self, now: float, status: int, duration_ms: float, is_error: bool) -> None:
        self._heartbeat.count += 1
        self._heartbeat.last_seen = now
        self._heartbeat.last_status = status
        self._heartbeat.last_duration_ms = round(duration_ms, 1)
        if is_error:
            self._heartbeat.error_count += 1

    def _heartbeat_snapshot_locked(self, now: float) -> dict[str, Any]:
        last_seen = self._heartbeat.last_seen
        idle_seconds = max(0.0, now - last_seen) if last_seen is not None else None
        return {
            "count": self._heartbeat.count,
            "error_count": self._heartbeat.error_count,
            "last_seen": iso_timestamp(last_seen) if last_seen is not None else None,
            "last_seen_unix": last_seen,
            "idle_seconds": round(idle_seconds, 1) if idle_seconds is not None else None,
            "status": self._heartbeat.last_status,
            "duration_ms": self._heartbeat.last_duration_ms,
        }

    def _record_diagnostic_locked(
        self,
        diagnostic: DiagnosticTelemetry,
        now: float,
        status: int,
        duration_ms: float,
        is_error: bool,
        path: str,
    ) -> None:
        diagnostic.count += 1
        diagnostic.last_seen = now
        diagnostic.last_status = status
        diagnostic.last_duration_ms = round(duration_ms, 1)
        diagnostic.last_path = path
        if is_error:
            diagnostic.error_count += 1

    def _diagnostic_snapshot_locked(self, diagnostic: DiagnosticTelemetry, now: float) -> dict[str, Any]:
        last_seen = diagnostic.last_seen
        idle_seconds = max(0.0, now - last_seen) if last_seen is not None else None
        return {
            "count": diagnostic.count,
            "error_count": diagnostic.error_count,
            "last_seen": iso_timestamp(last_seen) if last_seen is not None else None,
            "last_seen_unix": last_seen,
            "idle_seconds": round(idle_seconds, 1) if idle_seconds is not None else None,
            "status": diagnostic.last_status,
            "duration_ms": diagnostic.last_duration_ms,
            "path": diagnostic.last_path,
        }

    def _client_snapshot(self, client: ClientTelemetry, now: float) -> dict[str, Any]:
        idle_seconds = max(0.0, now - client.last_seen)
        current_action_entry = next(reversed(client.current_actions.values()), None) if client.current_actions else None
        if isinstance(current_action_entry, dict):
            current_action = current_action_entry.get("label")
            action_started = current_action_entry.get("started_wall")
        else:
            current_action = current_action_entry
            action_started = None
        current_action_seconds = (
            round(max(0.0, now - float(action_started)), 1)
            if action_started is not None
            else None
        )
        return {
            "client_ip": client.client_ip,
            "user_agent": client.user_agent,
            "user_agent_label": user_agent_label(client.user_agent),
            "first_seen": iso_timestamp(client.first_seen),
            "first_seen_unix": client.first_seen,
            "last_seen": iso_timestamp(client.last_seen),
            "last_seen_unix": client.last_seen,
            "idle_seconds": round(idle_seconds, 1),
            "active": bool(client.current_actions) or idle_seconds <= self.active_window_seconds,
            "in_flight": len(client.current_actions),
            "request_count": client.request_count,
            "app_action_count": client.app_action_count,
            "error_count": client.error_count,
            "current_action": current_action,
            "current_action_seconds": current_action_seconds,
            "last_app_action": client.last_app_action,
            "last_app_path": client.last_app_path,
            "last_path": client.last_path,
            "last_status": client.last_status,
            "last_duration_ms": client.last_duration_ms,
        }

    def _prune_clients_locked(self) -> None:
        overflow = len(self._clients) - self.max_clients
        if overflow <= 0:
            return
        inactive_clients = sorted(
            (client for client in self._clients.values() if not client.current_actions),
            key=lambda client: client.last_seen,
        )
        for client in inactive_clients[:overflow]:
            self._clients.pop(client.key, None)


class TelemetryMiddleware:
    def __init__(self, app: Any, store: TelemetryStore) -> None:
        self.app = app
        self.store = store

    async def __call__(self, scope: dict[str, Any], receive: Any, send: Any) -> None:
        request = self.store.begin(scope)
        status_code: int | None = None

        async def telemetry_send(message: dict[str, Any]) -> None:
            nonlocal status_code
            if message.get("type") == "http.response.start":
                status_code = int(message.get("status") or 500)
            await send(message)

        try:
            await self.app(scope, receive, telemetry_send)
        except Exception:
            self.store.finish(request, status_code or 500, failed=True)
            raise
        self.store.finish(request, status_code or 500)
