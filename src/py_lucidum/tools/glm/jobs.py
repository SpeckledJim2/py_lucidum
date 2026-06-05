from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Any
from uuid import uuid4

from py_lucidum.core import Dataset

from .store import GlmModelStore
from .tabulation import build_tabulations
from .training import train_model


@dataclass
class GlmJob:
    id: str
    status: str = "queued"
    created_at: str = field(default_factory=lambda: time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    updated_at: str = field(default_factory=lambda: time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    result: dict[str, Any] | None = None
    error: str | None = None
    progress: dict[str, Any] | None = None

    def as_payload(self) -> dict[str, Any]:
        return {
            "job_id": self.id,
            "status": self.status,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "result": self.result,
            "error": self.error,
            "progress": self.progress,
        }


class GlmJobManager:
    def __init__(self):
        self._jobs: dict[str, GlmJob] = {}
        self._lock = threading.RLock()

    def start(self, dataset: Dataset, store: GlmModelStore, payload: dict[str, Any]) -> GlmJob:
        job = GlmJob(id=uuid4().hex)
        with self._lock:
            self._jobs[job.id] = job
        thread = threading.Thread(target=self._run, args=(job.id, dataset, store, payload), daemon=True)
        thread.start()
        return job

    def start_tabulations(self, dataset: Dataset, store: GlmModelStore, payload: dict[str, Any], feature_spec: Any) -> GlmJob:
        job = GlmJob(id=uuid4().hex)
        with self._lock:
            self._jobs[job.id] = job
        thread = threading.Thread(target=self._run_tabulations, args=(job.id, dataset, store, payload, feature_spec), daemon=True)
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
            return
        self.update_progress(job_id, {"phase": "succeeded", "message": "GLM training complete", "percent": 100})
        self._update(job_id, status="succeeded", result=result)

    def _run_tabulations(self, job_id: str, dataset: Dataset, store: GlmModelStore, payload: dict[str, Any], feature_spec: Any) -> None:
        self._update(job_id, status="running")
        try:
            result = build_tabulations(dataset, store, payload, feature_spec, progress_callback=lambda progress: self.update_progress(job_id, progress))
        except Exception as exc:
            self.update_progress(job_id, {"phase": "failed", "message": str(exc)})
            self._update(job_id, status="failed", error=str(exc))
            return
        self.update_progress(job_id, {"phase": "succeeded", "message": "GLM tabulation complete", "percent": 100})
        self._update(job_id, status="succeeded", result=result)

    def update_progress(self, job_id: str, progress: dict[str, Any]) -> None:
        with self._lock:
            job = self._jobs[job_id]
            previous = job.progress if isinstance(job.progress, dict) else {}
            job.progress = {**previous, **progress}
            job.updated_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

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


__all__ = ["GlmJob", "GlmJobManager"]
