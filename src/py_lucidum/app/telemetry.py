from __future__ import annotations

import copy
import importlib.metadata
import platform
import re
import sys
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
    ("POST", "/api/line-bar/glm-overlay"): "Line/bar GLM overlay",
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
OPERATION_ID_HEADER = "x-lucidum-operation-id"
OPERATION_ID_RE = re.compile(r"^[A-Za-z0-9._-]{1,80}$")
OPERATION_PHASE_RE = re.compile(r"^[a-z0-9_]{1,40}$")
OPERATION_PHASE_LABELS = {
    "artifacts": "Saving artifacts",
    "awaiting_job_request": "Awaiting job request",
    "coercing_columns": "Coercing response and denominator",
    "constructing_datasets": "Constructing LightGBM datasets",
    "creating_sample": "Creating SAMPLE split",
    "creating_workspace": "Creating model workspace",
    "encoding_categoricals": "Encoding categoricals",
    "fitting": "Fitting model",
    "grid": "Grid search",
    "importing_cffi": "Importing CFFI",
    "importing_lightgbm": "Importing LightGBM",
    "importing_numpy": "Importing NumPy",
    "importing_pandas": "Importing pandas",
    "importing_polars": "Importing Polars",
    "importing_pyarrow": "Importing PyArrow",
    "loading": "Loading training data",
    "loading_data": "Loading selected data",
    "loading_dependencies": "Loading dependencies",
    "preparing": "Preparing model",
    "preparing_init_scores": "Preparing initial scores",
    "queued": "Queued",
    "resolving_features": "Resolving selected features",
    "resolving_parameters": "Resolving parameters",
    "scoring": "Scoring rows",
    "shap": "Calculating SHAP values",
    "splitting_sample": "Applying SAMPLE split",
    "starting": "Starting",
    "tabulating": "Building tabulations",
    "validating_request": "Validating request",
    "waiting_for_dataset": "Waiting for dataset access",
    "writing": "Saving artifacts",
}
OPERATION_METADATA_KEYS = {
    "categorical_feature_count",
    "cells",
    "feature_count",
    "iteration",
    "projection_column_count",
    "scored_rows",
    "scoring_rows",
    "test_rows",
    "total_iterations",
    "training_rows",
    "validation_rows",
    "worker_mode",
}
TERMINAL_OPERATION_PHASES = {"failed", "succeeded"}
PACKAGE_VERSIONS = {
    "lucidum": "py-lucidum",
    "duckdb": "duckdb",
    "lightgbm": "lightgbm",
    "numpy": "numpy",
    "pandas": "pandas",
    "polars": "polars",
    "pyarrow": "pyarrow",
}


@dataclass(frozen=True)
class ResourcePoint:
    perf: float
    cpu_seconds: float
    rss_bytes: int
    thread_count: int


@dataclass
class OperationPhase:
    name: str
    label: str
    started_wall: float
    started: ResourcePoint
    metadata: dict[str, Any] = field(default_factory=dict)
    ended_wall: float | None = None
    ended: ResourcePoint | None = None


@dataclass
class OperationRequest:
    request_id: int
    method: str
    path: str
    label: str
    started_wall: float
    started: ResourcePoint
    status: int | None = None
    ended_wall: float | None = None
    ended: ResourcePoint | None = None


@dataclass
class OperationTelemetry:
    operation_id: str
    tool: str
    label: str
    status: str
    started_wall: float
    started: ResourcePoint
    updated_perf: float
    metadata: dict[str, Any] = field(default_factory=dict)
    phases: list[OperationPhase] = field(default_factory=list)
    requests: list[OperationRequest] = field(default_factory=list)
    in_flight_requests: set[int] = field(default_factory=set)
    current_phase: OperationPhase | None = None
    ended_wall: float | None = None
    ended: ResourcePoint | None = None
    error_type: str | None = None
    observed_peak_rss_bytes: int = 0
    dropped_phase_count: int = 0
    dropped_request_count: int = 0


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
    operation_id: str | None = None
    operation_label: str | None = None
    operation_started: ResourcePoint | None = None


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


def normalise_operation_id(value: Any) -> str | None:
    operation_id = str(value or "").strip()
    return operation_id if OPERATION_ID_RE.fullmatch(operation_id) else None


def request_operation_id(request: Any) -> str | None:
    headers = getattr(request, "headers", {})
    return normalise_operation_id(headers.get(OPERATION_ID_HEADER))


def operation_request_spec(method: str, path: str) -> tuple[str, str, str] | None:
    key = (method.upper(), path)
    exact = {
        ("POST", "/api/gbm/validate"): ("gbm", "GBM Train", "Preflight validation"),
        ("POST", "/api/gbm/train"): ("gbm", "GBM Train", "Train request"),
        ("POST", "/api/glm/validate"): ("glm", "GLM Build", "Preflight validation"),
        ("POST", "/api/glm/build"): ("glm", "GLM Build", "Build request"),
        ("POST", "/api/glm/tabulations/build"): ("glm", "GLM Tabulation", "Tabulation request"),
    }
    if key in exact:
        return exact[key]
    if method.upper() == "GET" and re.fullmatch(r"/api/gbm/jobs/[^/]+", path):
        return ("gbm", "GBM Train", "Job poll")
    if method.upper() == "GET" and re.fullmatch(r"/api/glm/jobs/[^/]+", path):
        return ("glm", "GLM Build", "Job poll")
    if method.upper() == "GET" and re.fullmatch(r"/api/glm/tabulations/jobs/[^/]+", path):
        return ("glm", "GLM Tabulation", "Job poll")
    return None


def package_version(distribution: str) -> str | None:
    try:
        return importlib.metadata.version(distribution)
    except Exception:
        return None


def python_environment_kind() -> str:
    prefix = str(getattr(sys, "prefix", "")).lower()
    if "pipx" in prefix:
        return "pipx"
    if getattr(sys, "prefix", None) != getattr(sys, "base_prefix", None):
        return "virtualenv"
    return "system"


def runtime_environment() -> dict[str, Any]:
    return {
        "python": platform.python_version(),
        "implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "system": platform.system(),
        "release": platform.release(),
        "machine": platform.machine(),
        "executable_name": str(getattr(sys, "executable", "")).rsplit("/", 1)[-1],
        "python_environment": python_environment_kind(),
        "cpu": {
            "logical": psutil.cpu_count(logical=True),
            "physical": psutil.cpu_count(logical=False),
        },
        "packages": {
            label: version
            for label, distribution in PACKAGE_VERSIONS.items()
            if (version := package_version(distribution)) is not None
        },
    }


def operation_metadata(payload: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(payload, dict):
        return {}
    metadata: dict[str, Any] = {}
    for key in OPERATION_METADATA_KEYS:
        value = payload.get(key)
        if isinstance(value, bool):
            continue
        if isinstance(value, (int, float, str)) and str(value).strip():
            metadata[key] = value
    return metadata


def resource_point(process: psutil.Process) -> ResourcePoint:
    perf = time.perf_counter()
    try:
        cpu_times = process.cpu_times()
        cpu_seconds = float(cpu_times.user) + float(cpu_times.system)
    except (psutil.Error, AttributeError, OSError):
        cpu_seconds = 0.0
    try:
        rss_bytes = int(process.memory_info().rss)
    except (psutil.Error, AttributeError, OSError):
        rss_bytes = 0
    try:
        thread_count = int(process.num_threads())
    except (psutil.Error, AttributeError, OSError):
        thread_count = 0
    return ResourcePoint(
        perf=perf,
        cpu_seconds=cpu_seconds,
        rss_bytes=rss_bytes,
        thread_count=thread_count,
    )


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
        max_operations: int = 100,
        max_operation_phases: int = 200,
        max_operation_requests: int = 200,
        orphan_operation_seconds: float = 60,
        environment: dict[str, Any] | None = None,
    ) -> None:
        self.active_window_seconds = active_window_seconds
        self.max_clients = max_clients
        self.max_operations = max(1, int(max_operations))
        self.max_operation_phases = max(2, int(max_operation_phases))
        self.max_operation_requests = max(3, int(max_operation_requests))
        self.orphan_operation_seconds = max(1.0, float(orphan_operation_seconds))
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
        self._active_operations: dict[str, OperationTelemetry] = {}
        self._recent_operations: deque[OperationTelemetry] = deque(maxlen=self.max_operations)
        self._environment = copy.deepcopy(environment) if isinstance(environment, dict) else runtime_environment()

    def update_environment(self, values: dict[str, Any]) -> None:
        if not isinstance(values, dict):
            return
        with self._lock:
            for key, value in values.items():
                self._environment[str(key)] = copy.deepcopy(value)

    def ensure_operation(
        self,
        operation_id: Any,
        *,
        tool: str,
        label: str,
        metadata: dict[str, Any] | None = None,
        _point: ResourcePoint | None = None,
    ) -> str | None:
        normalised = normalise_operation_id(operation_id)
        if not normalised:
            return None
        point = _point or resource_point(self._process)
        now = time.time()
        with self._lock:
            self._expire_orphan_operations_locked(now, point)
            operation = self._active_operations.get(normalised)
            if operation is None:
                completed = self._operation_by_id_locked(normalised)
                if completed is not None:
                    if metadata:
                        completed.metadata.update(operation_metadata(metadata))
                    completed.observed_peak_rss_bytes = max(
                        completed.observed_peak_rss_bytes,
                        point.rss_bytes,
                    )
                    return normalised
                if len(self._active_operations) >= self.max_operations:
                    oldest_id = min(
                        self._active_operations,
                        key=lambda item: self._active_operations[item].updated_perf,
                    )
                    oldest = self._active_operations.pop(oldest_id)
                    self._abandon_operation_locked(oldest, now, point)
                operation = OperationTelemetry(
                    operation_id=normalised,
                    tool=str(tool or ""),
                    label=str(label or "Operation"),
                    status="running",
                    started_wall=now,
                    started=point,
                    updated_perf=point.perf,
                    observed_peak_rss_bytes=point.rss_bytes,
                )
                self._active_operations[normalised] = operation
            if metadata:
                operation.metadata.update(operation_metadata(metadata))
            operation.updated_perf = point.perf
            operation.observed_peak_rss_bytes = max(operation.observed_peak_rss_bytes, point.rss_bytes)
        return normalised

    def update_operation_phase(
        self,
        operation_id: Any,
        *,
        name: str,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        normalised = normalise_operation_id(operation_id)
        phase_name = str(name or "").strip().lower()
        if (
            not normalised
            or not OPERATION_PHASE_RE.fullmatch(phase_name)
            or phase_name in TERMINAL_OPERATION_PHASES
        ):
            return
        phase_label = OPERATION_PHASE_LABELS.get(
            phase_name,
            phase_name.replace("_", " ").title(),
        )
        now = time.time()
        with self._lock:
            operation = self._active_operations.get(normalised)
            if operation is None:
                return
            safe_metadata = operation_metadata(metadata)
            current = operation.current_phase
            if current is not None and current.name == phase_name:
                current.metadata.update(safe_metadata)
                operation.metadata.update(safe_metadata)
                return
            point = resource_point(self._process)
            if current is not None:
                self._finish_phase_locked(operation, current, now, point)
            phase = OperationPhase(
                name=phase_name,
                label=phase_label,
                started_wall=now,
                started=point,
                metadata=safe_metadata,
            )
            operation.phases.append(phase)
            if len(operation.phases) > self.max_operation_phases:
                del operation.phases[1]
                operation.dropped_phase_count += 1
            operation.current_phase = phase
            operation.metadata.update(safe_metadata)
            operation.updated_perf = point.perf
            operation.observed_peak_rss_bytes = max(operation.observed_peak_rss_bytes, point.rss_bytes)

    def finish_operation(
        self,
        operation_id: Any,
        *,
        status: str,
        error_type: str | None = None,
    ) -> None:
        normalised = normalise_operation_id(operation_id)
        if not normalised:
            return
        point = resource_point(self._process)
        now = time.time()
        with self._lock:
            operation = self._active_operations.pop(normalised, None)
            if operation is None:
                return
            if operation.current_phase is not None:
                self._finish_phase_locked(operation, operation.current_phase, now, point)
            operation.status = str(status or "succeeded")
            operation.error_type = str(error_type or "").strip() or None
            operation.ended_wall = now
            operation.ended = point
            operation.updated_perf = point.perf
            operation.observed_peak_rss_bytes = max(operation.observed_peak_rss_bytes, point.rss_bytes)
            self._recent_operations.appendleft(operation)

    def _finish_phase_locked(
        self,
        operation: OperationTelemetry,
        phase: OperationPhase,
        now: float,
        point: ResourcePoint,
    ) -> None:
        if phase.ended is None:
            phase.ended_wall = now
            phase.ended = point
        if operation.current_phase is phase:
            operation.current_phase = None

    def _operation_by_id_locked(self, operation_id: str) -> OperationTelemetry | None:
        operation = self._active_operations.get(operation_id)
        if operation is not None:
            return operation
        return next(
            (
                item
                for item in self._recent_operations
                if item.operation_id == operation_id
            ),
            None,
        )

    def _expire_orphan_operations_locked(
        self,
        now: float,
        point: ResourcePoint,
    ) -> None:
        orphan_ids = [
            operation_id
            for operation_id, operation in self._active_operations.items()
            if (
                operation.current_phase is None
                or operation.current_phase.name == "awaiting_job_request"
            )
            and not operation.in_flight_requests
            and point.perf - operation.updated_perf >= self.orphan_operation_seconds
        ]
        for operation_id in orphan_ids:
            operation = self._active_operations.pop(operation_id)
            self._abandon_operation_locked(operation, now, point)

    def _abandon_operation_locked(
        self,
        operation: OperationTelemetry,
        now: float,
        point: ResourcePoint,
    ) -> None:
        if operation.current_phase is not None:
            self._finish_phase_locked(operation, operation.current_phase, now, point)
        operation.status = "abandoned"
        operation.ended_wall = now
        operation.ended = point
        operation.updated_perf = point.perf
        operation.observed_peak_rss_bytes = max(operation.observed_peak_rss_bytes, point.rss_bytes)
        self._recent_operations.appendleft(operation)

    @staticmethod
    def _resource_delta(start: ResourcePoint, end: ResourcePoint) -> dict[str, Any]:
        wall_seconds = max(0.0, end.perf - start.perf)
        cpu_seconds = max(0.0, end.cpu_seconds - start.cpu_seconds)
        return {
            "duration_ms": round(wall_seconds * 1000, 1),
            "cpu_seconds": round(cpu_seconds, 3),
            "average_cores": round(cpu_seconds / wall_seconds, 2) if wall_seconds > 0 else None,
            "rss_start_mb": bytes_to_mb(start.rss_bytes),
            "rss_end_mb": bytes_to_mb(end.rss_bytes),
            "rss_change_mb": bytes_to_mb(end.rss_bytes - start.rss_bytes),
            "threads_start": start.thread_count,
            "threads_end": end.thread_count,
        }

    def _phase_snapshot(
        self,
        operation: OperationTelemetry,
        phase: OperationPhase,
        point: ResourcePoint,
    ) -> dict[str, Any]:
        end = phase.ended or point
        payload = {
            "kind": "phase",
            "name": phase.name,
            "label": phase.label,
            "status": "running" if phase.ended is None else "completed",
            "started_at": iso_timestamp(phase.started_wall),
            "start_offset_ms": round(max(0.0, phase.started.perf - operation.started.perf) * 1000, 1),
            "metadata": dict(phase.metadata),
            **self._resource_delta(phase.started, end),
        }
        if phase.ended_wall is not None:
            payload["ended_at"] = iso_timestamp(phase.ended_wall)
        return payload

    def _operation_request_snapshot(
        self,
        operation: OperationTelemetry,
        request: OperationRequest,
        point: ResourcePoint,
    ) -> dict[str, Any]:
        end = request.ended or point
        payload = {
            "kind": "request",
            "request_id": request.request_id,
            "method": request.method,
            "path": request.path,
            "name": request.label.lower().replace(" ", "_"),
            "label": request.label,
            "status": request.status if request.status is not None else "running",
            "started_at": iso_timestamp(request.started_wall),
            "start_offset_ms": round(max(0.0, request.started.perf - operation.started.perf) * 1000, 1),
            **self._resource_delta(request.started, end),
        }
        if request.ended_wall is not None:
            payload["ended_at"] = iso_timestamp(request.ended_wall)
        return payload

    def _operation_snapshot(
        self,
        operation: OperationTelemetry,
        point: ResourcePoint,
    ) -> dict[str, Any]:
        end = operation.ended or point
        operation.observed_peak_rss_bytes = max(operation.observed_peak_rss_bytes, point.rss_bytes)
        phases = [
            self._phase_snapshot(operation, phase, point)
            for phase in operation.phases
        ]
        requests = [
            self._operation_request_snapshot(operation, request, point)
            for request in operation.requests
        ]
        timeline = sorted(
            [*requests, *phases],
            key=lambda item: (float(item.get("start_offset_ms") or 0), 0 if item["kind"] == "request" else 1),
        )
        completed_timeline = [
            item
            for item in timeline
            if isinstance(item.get("duration_ms"), (int, float))
        ]
        slowest = (
            max(completed_timeline, key=lambda item: float(item.get("duration_ms") or 0))
            if completed_timeline
            else None
        )
        current_phase = operation.current_phase.label if operation.current_phase is not None else None
        payload = {
            "operation_id": operation.operation_id,
            "tool": operation.tool,
            "label": operation.label,
            "status": operation.status,
            "started_at": iso_timestamp(operation.started_wall),
            "metadata": dict(operation.metadata),
            "current_phase": current_phase,
            "observed_peak_rss_mb": bytes_to_mb(operation.observed_peak_rss_bytes),
            "error_type": operation.error_type,
            "dropped_phase_count": operation.dropped_phase_count,
            "dropped_request_count": operation.dropped_request_count,
            "phases": phases,
            "requests": requests,
            "timeline": timeline,
            "slowest_phase": (
                {
                    "kind": slowest.get("kind"),
                    "name": slowest.get("name"),
                    "label": slowest.get("label"),
                    "duration_ms": slowest.get("duration_ms"),
                }
                if slowest
                else None
            ),
            **self._resource_delta(operation.started, end),
        }
        if operation.ended_wall is not None:
            payload["ended_at"] = iso_timestamp(operation.ended_wall)
        return payload

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
        operation_spec = operation_request_spec(method, path)
        operation_id = (
            normalise_operation_id(header_value(scope, OPERATION_ID_HEADER))
            if operation_spec
            else None
        )
        operation_started = resource_point(self._process) if operation_id else None
        if operation_id and operation_spec:
            tool, operation_name, _ = operation_spec
            self.ensure_operation(
                operation_id,
                tool=tool,
                label=operation_name,
                _point=operation_started,
            )

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
            operation_label: str | None = None
            if operation_id and operation_spec and operation_started is not None:
                operation = self._operation_by_id_locked(operation_id)
                if operation is not None:
                    operation_label = operation_spec[2]
                    operation.requests.append(
                        OperationRequest(
                            request_id=request_id,
                            method=method,
                            path=path,
                            label=operation_label,
                            started_wall=now,
                            started=operation_started,
                        )
                    )
                    if len(operation.requests) > self.max_operation_requests:
                        del operation.requests[2]
                        operation.dropped_request_count += 1
                    if operation.status == "running":
                        operation.in_flight_requests.add(request_id)
                    operation.updated_perf = operation_started.perf
                    operation.observed_peak_rss_bytes = max(
                        operation.observed_peak_rss_bytes,
                        operation_started.rss_bytes,
                    )

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
            operation_id=operation_id,
            operation_label=operation_label,
            operation_started=operation_started,
        )

    def finish(self, request: TelemetryRequest | None, status_code: int | None, failed: bool = False) -> None:
        if request is None:
            return
        now = time.time()
        duration_ms = max(0.0, (time.perf_counter() - request.started_perf) * 1000)
        status = int(status_code or 500)
        is_error = failed or status >= 400
        operation_ended = resource_point(self._process) if request.operation_id else None
        should_fail_operation = bool(
            is_error
            and request.operation_id
            and request.operation_label != "Job poll"
        )

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
                "request_id": request.request_id,
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
                **({"operation_id": request.operation_id} if request.operation_id else {}),
            })
            if request.operation_id and operation_ended is not None:
                operation = self._operation_by_id_locked(request.operation_id)
                if operation is not None:
                    operation_request = next(
                        (
                            item
                            for item in reversed(operation.requests)
                            if item.request_id == request.request_id
                        ),
                        None,
                    )
                    if operation_request is not None:
                        operation_request.status = status
                        operation_request.ended_wall = now
                        operation_request.ended = operation_ended
                    operation.in_flight_requests.discard(request.request_id)
                    operation.updated_perf = operation_ended.perf
                    operation.observed_peak_rss_bytes = max(
                        operation.observed_peak_rss_bytes,
                        operation_ended.rss_bytes,
                    )
                    if operation.ended is not None and operation_ended.perf > operation.ended.perf:
                        operation.ended_wall = now
                        operation.ended = operation_ended
        if should_fail_operation:
            self.finish_operation(
                request.operation_id,
                status="failed",
                error_type=f"HTTP{status}",
            )

    def snapshot(self) -> dict[str, Any]:
        now = time.time()
        process = process_memory_snapshot(self._process)
        operation_point = resource_point(self._process)
        with self._lock:
            self._expire_orphan_operations_locked(now, operation_point)
            rss_bytes = int(process.get("rss_bytes") or 0)
            self._peak_rss_bytes = max(self._peak_rss_bytes, rss_bytes)
            process["peak_rss_bytes"] = self._peak_rss_bytes
            process["peak_rss_mb"] = bytes_to_mb(self._peak_rss_bytes)
            clients = [self._client_snapshot(client, now) for client in self._clients.values()]
            clients.sort(key=lambda client: (client["in_flight"], client["last_seen_unix"]), reverse=True)
            in_flight = sum(client["in_flight"] for client in clients)
            active_clients = sum(1 for client in clients if client["active"])
            recent_error_count = sum(1 for event in self._recent if event.get("error"))
            active_operations = [
                self._operation_snapshot(operation, operation_point)
                for operation in self._active_operations.values()
            ]
            active_operations.sort(key=lambda operation: str(operation.get("started_at") or ""))
            recent_operations = [
                self._operation_snapshot(operation, operation.ended or operation_point)
                for operation in self._recent_operations
            ]
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
                "environment": copy.deepcopy(self._environment),
                "operations": {
                    "active": active_operations,
                    "recent": recent_operations,
                    "active_count": len(active_operations),
                    "recent_limit": self.max_operations,
                },
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
        try:
            request = self.store.begin(scope)
        except Exception:
            request = None
        status_code: int | None = None

        async def telemetry_send(message: dict[str, Any]) -> None:
            nonlocal status_code
            if message.get("type") == "http.response.start":
                status_code = int(message.get("status") or 500)
            await send(message)

        try:
            await self.app(scope, receive, telemetry_send)
        except Exception:
            try:
                self.store.finish(request, status_code or 500, failed=True)
            except Exception:
                pass
            raise
        try:
            self.store.finish(request, status_code or 500)
        except Exception:
            pass
