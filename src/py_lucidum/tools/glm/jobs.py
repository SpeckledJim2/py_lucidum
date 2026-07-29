from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Any
from uuid import uuid4

from py_lucidum.app.telemetry import TelemetryStore, normalise_operation_id
from py_lucidum.core import Dataset

from .store import GlmModelStore
from .tabulation import build_tabulations
from .training import train_model


@dataclass
class GlmJob:
    id: str
    operation_id: str | None = None
    status: str = "queued"
    created_at: str = field(default_factory=lambda: time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    updated_at: str = field(default_factory=lambda: time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    result: dict[str, Any] | None = None
    error: str | None = None
    progress: dict[str, Any] | None = None

    def as_payload(self) -> dict[str, Any]:
        payload = {
            "job_id": self.id,
            "status": self.status,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "result": self.result,
            "error": self.error,
            "progress": self.progress,
        }
        if self.operation_id:
            payload["operation_id"] = self.operation_id
        return payload


class GlmJobManager:
    def __init__(self, telemetry: TelemetryStore | None = None):
        self._jobs: dict[str, GlmJob] = {}
        self._lock = threading.RLock()
        self._telemetry = telemetry

    def start(
        self,
        dataset: Dataset,
        store: GlmModelStore,
        payload: dict[str, Any],
        *,
        operation_id: str | None = None,
    ) -> GlmJob:
        job = GlmJob(id=uuid4().hex, operation_id=normalise_operation_id(operation_id))
        with self._lock:
            self._jobs[job.id] = job
        self._start_operation(job, tool="glm", label="GLM Build")
        thread = threading.Thread(target=self._run, args=(job.id, dataset, store, payload), daemon=True)
        thread.start()
        return job

    def start_tabulations(
        self,
        dataset: Dataset,
        store: GlmModelStore,
        payload: dict[str, Any],
        feature_spec: Any,
        gbm_store: Any = None,
        *,
        operation_id: str | None = None,
    ) -> GlmJob:
        job = GlmJob(id=uuid4().hex, operation_id=normalise_operation_id(operation_id))
        with self._lock:
            self._jobs[job.id] = job
        self._start_operation(job, tool="glm", label="GLM Tabulation")
        thread = threading.Thread(target=self._run_tabulations, args=(job.id, dataset, store, payload, feature_spec, gbm_store), daemon=True)
        thread.start()
        return job

    def get(self, job_id: str) -> GlmJob | None:
        with self._lock:
            return self._jobs.get(job_id)

    def _run(self, job_id: str, dataset: Dataset, store: GlmModelStore, payload: dict[str, Any]) -> None:
        self._update(job_id, status="running")
        try:
            result = train_model(dataset, store, payload, progress_callback=lambda progress: self.update_progress(job_id, progress))
        except Exception as exc:
            self.update_progress(job_id, {"phase": "failed", "message": str(exc)})
            self._update(job_id, status="failed", error=str(exc))
            self._finish_operation(job_id, status="failed", error_type=type(exc).__name__)
            return
        self.update_progress(job_id, {"phase": "succeeded", "message": "GLM training complete", "percent": 100})
        self._update(job_id, status="succeeded", result=result)
        self._finish_operation(job_id, status="succeeded")

    def _run_tabulations(self, job_id: str, dataset: Dataset, store: GlmModelStore, payload: dict[str, Any], feature_spec: Any, gbm_store: Any = None) -> None:
        self._update(job_id, status="running")
        try:
            result = build_tabulations(dataset, store, payload, feature_spec, progress_callback=lambda progress: self.update_progress(job_id, progress), gbm_store=gbm_store)
        except Exception as exc:
            self.update_progress(job_id, {"phase": "failed", "message": str(exc)})
            self._update(job_id, status="failed", error=str(exc))
            self._finish_operation(job_id, status="failed", error_type=type(exc).__name__)
            return
        self.update_progress(job_id, {"phase": "succeeded", "message": "Model tabulation complete", "percent": 100})
        self._update(job_id, status="succeeded", result=result)
        self._finish_operation(job_id, status="succeeded")

    def update_progress(self, job_id: str, progress: dict[str, Any]) -> None:
        with self._lock:
            job = self._jobs[job_id]
            previous = job.progress if isinstance(job.progress, dict) else {}
            job.progress = {**previous, **progress}
            job.updated_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            operation_id = job.operation_id
        phase = str(progress.get("stage") or progress.get("phase") or "").strip().lower()
        if phase and phase not in {"failed", "succeeded"}:
            self._telemetry_call(
                "update_operation_phase",
                operation_id,
                name=phase,
                metadata=progress,
            )

    def _update(
        self,
        job_id: str,
        *,
        status: str,
        result: dict[str, Any] | None = None,
        error: str | None = None,
    ) -> None:
        with self._lock:
            job = self._jobs[job_id]
            job.status = status
            job.result = result
            job.error = error
            job.updated_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    def _start_operation(self, job: GlmJob, *, tool: str, label: str) -> None:
        self._telemetry_call(
            "ensure_operation",
            job.operation_id,
            tool=tool,
            label=label,
        )
        self._telemetry_call(
            "update_operation_phase",
            job.operation_id,
            name="queued",
        )

    def _finish_operation(
        self,
        job_id: str,
        *,
        status: str,
        error_type: str | None = None,
    ) -> None:
        with self._lock:
            operation_id = self._jobs[job_id].operation_id
        self._telemetry_call(
            "finish_operation",
            operation_id,
            status=status,
            error_type=error_type,
        )

    def _telemetry_call(self, method_name: str, *args: Any, **kwargs: Any) -> None:
        if self._telemetry is None:
            return
        try:
            getattr(self._telemetry, method_name)(*args, **kwargs)
        except Exception:
            pass


__all__ = ["GlmJob", "GlmJobManager"]
