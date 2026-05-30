from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Any
from uuid import uuid4

from py_lucidum.core import Dataset

from .grid import prepare_grid_run
from .store import GbmModelStore
from .training import train_model


@dataclass
class GbmJob:
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


class GbmJobManager:
    def __init__(self):
        self._jobs: dict[str, GbmJob] = {}
        self._lock = threading.RLock()

    def start(self, dataset: Dataset, store: GbmModelStore, payload: dict[str, Any]) -> GbmJob:
        job = GbmJob(id=uuid4().hex)
        with self._lock:
            self._jobs[job.id] = job
        thread = threading.Thread(target=self._run, args=(job.id, dataset, store, payload), daemon=True)
        thread.start()
        return job

    def get(self, job_id: str) -> GbmJob | None:
        with self._lock:
            return self._jobs.get(job_id)

    def _run(self, job_id: str, dataset: Dataset, store: GbmModelStore, payload: dict[str, Any]) -> None:
        self._update(job_id, status="running")
        try:
            grid_run = prepare_grid_run(dataset, payload, generated_sample_path=store.generated_sample_path)
            if grid_run.enabled:
                result = self._run_grid_search(job_id, dataset, store, payload, grid_run)
            elif grid_run.errors:
                raise ValueError("; ".join(grid_run.errors))
            else:
                result = train_model(dataset, store, payload, progress_callback=lambda progress: self.update_progress(job_id, progress))
        except Exception as exc:  # surfaced through polling endpoint
            self.update_progress(job_id, {"phase": "failed", "message": str(exc)})
            self._update(job_id, status="failed", error=str(exc))
            return
        done_message = "GBM grid search complete" if isinstance(result, dict) and result.get("grid_search_run") else "GBM training complete"
        self.update_progress(job_id, {"phase": "succeeded", "message": done_message, "percent": 100})
        self._update(job_id, status="succeeded", result=result)

    def _run_grid_search(
        self,
        job_id: str,
        dataset: Dataset,
        store: GbmModelStore,
        payload: dict[str, Any],
        grid_run: Any,
    ) -> dict[str, Any]:
        if not grid_run.ok:
            raise ValueError("; ".join(grid_run.errors))
        total_models = len(grid_run.combinations)
        run_id = uuid4().hex
        messages = grid_run.messages or [f"Grid search training {total_models:,} models"]
        self.update_progress(
            job_id,
            {
                "phase": "grid",
                "message": "Training GBM grid search...",
                "percent": 0,
                "grid": grid_run.summary(),
            },
        )
        successful: list[dict[str, Any]] = []
        failed: list[dict[str, Any]] = []
        base_label = str(payload.get("label") or "GBM").strip() or "GBM"
        for model_number, combination in enumerate(grid_run.combinations, start=1):
            combo_payload = {
                **payload,
                "label": f"{base_label} grid {model_number} of {total_models}",
                "parameters": combination.parameters,
            }
            grid_metadata = {
                "run_id": run_id,
                "combo_index": combination.combination_index,
                "model_number": model_number,
                "trained_count": total_models,
                "sampled_count": len(grid_run.sample_indexes),
                "total_combinations": grid_run.grid.total_combinations,
                "skipped_count": len(grid_run.skipped),
                "resolved_parameters": combination.resolved_parameters,
            }
            try:
                result = train_model(
                    dataset,
                    store,
                    combo_payload,
                    progress_callback=self._grid_progress_callback(
                        job_id,
                        model_number,
                        total_models,
                        grid_run.summary(),
                        combination.resolved_parameters,
                    ),
                    activate=False,
                    grid_search=grid_metadata,
                )
            except Exception as exc:
                failed.append(
                    {
                        "model_number": model_number,
                        "combination_index": combination.combination_index,
                        "error": str(exc),
                    }
                )
                self.update_progress(
                    job_id,
                    {
                        "phase": "grid",
                        "message": f"model {model_number}/{total_models} failed: {exc}",
                        "grid": {**grid_run.summary(), "failed_count": len(failed)},
                    },
                )
                continue
            successful.append(result)
        if not successful:
            first_error = failed[0]["error"] if failed else "no models were trained"
            raise ValueError(f"No GBM grid search models completed. First error: {first_error}")
        best = best_grid_model(successful)
        best_id = str(best.get("model_id") or "")
        active = store.activate_model(best_id) if best_id else best
        result = dict(active)
        final_grid_summary = {**grid_run.summary(), "failed_count": len(failed), "completed_count": len(successful), "best_model_id": best_id}
        result["grid_search_run"] = final_grid_summary
        result["grid_search_models"] = [str(model.get("model_id") or "") for model in successful if model.get("model_id")]
        self.update_progress(
            job_id,
            {
                "phase": "succeeded",
                "message": f"GBM grid search complete, best model {best_id}",
                "percent": 100,
                "grid": final_grid_summary,
            },
        )
        return result

    def _grid_progress_callback(
        self,
        job_id: str,
        model_number: int,
        total_models: int,
        grid_summary: dict[str, Any],
        parameters: dict[str, Any],
    ):
        def callback(progress: dict[str, Any]) -> None:
            message = str(progress.get("message") or "").strip()
            prefixed = f"model {model_number}/{total_models}" + (f", {message}" if message else "")
            self.update_progress(
                job_id,
                {
                    **progress,
                    "message": prefixed,
                    "grid": grid_summary,
                    "grid_model_number": model_number,
                    "grid_model_count": total_models,
                    "grid_parameters": key_grid_parameters(parameters),
                },
            )

        return callback

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


HIGHER_IS_BETTER_METRICS = {"auc", "average_precision", "r2"}


def best_grid_model(models: list[dict[str, Any]]) -> dict[str, Any]:
    best = models[-1]
    best_key: tuple[int, float] | None = None
    for model in models:
        metric = str(model.get("metric") or "").strip().lower()
        metrics = model.get("best_metrics") if isinstance(model.get("best_metrics"), dict) else {}
        value = metrics.get("test")
        if value is None:
            value = metrics.get("training")
        try:
            score = float(value)
        except (TypeError, ValueError):
            continue
        signed_score = score if metric in HIGHER_IS_BETTER_METRICS else -score
        key = (1, signed_score)
        if best_key is None or key > best_key:
            best_key = key
            best = model
    return best


KEY_GRID_PARAMETER_LABELS = {
    "objective": "obj",
    "metric": "metric",
    "data_sample_strategy": "sample",
    "num_iterations": "iters",
    "learning_rate": "lr",
    "num_leaves": "leaves",
    "max_depth": "depth",
    "min_data_in_leaf": "min_leaf",
    "feature_fraction": "feature_frac",
    "bagging_fraction": "bagging_frac",
    "bagging_freq": "bag_freq",
}


def key_grid_parameters(parameters: dict[str, Any]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for name, label in KEY_GRID_PARAMETER_LABELS.items():
        value = parameters.get(name)
        if value is None or str(value).strip() == "":
            continue
        rows.append({"name": name, "label": label, "value": str(value)})
    return rows


__all__ = ["GbmJob", "GbmJobManager", "best_grid_model", "key_grid_parameters"]
