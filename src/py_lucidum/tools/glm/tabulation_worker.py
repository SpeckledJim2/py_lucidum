from __future__ import annotations

import json
import os
import sys
import traceback
from pathlib import Path

from py_lucidum.core import Dataset
from py_lucidum.tools.gbm.store import GbmModelStore

from .store import GlmModelStore
from .tabulation import _build_tabulations_impl


def run_worker(request_path: Path, response_path: Path) -> int:
    request = json.loads(request_path.read_text(encoding="utf-8"))
    dataset_path = Path(str(request["dataset_path"]))
    payload = request.get("payload")
    if not isinstance(payload, dict):
        raise ValueError("GLM tabulation worker request payload must be an object")
    dataset = Dataset(dataset_path)
    store = GlmModelStore(dataset_path)
    gbm_store = GbmModelStore(dataset_path) if request.get("gbm_available") else None
    progress_path = Path(str(request["progress_path"])) if request.get("progress_path") else None

    def publish_progress(progress: dict[str, object]) -> None:
        if progress_path is None:
            return
        pending_path = progress_path.with_suffix(f".{os.getpid()}.tmp")
        pending_path.write_text(json.dumps(progress, default=str), encoding="utf-8")
        pending_path.replace(progress_path)

    result = _build_tabulations_impl(
        dataset,
        store,
        payload,
        request.get("feature_spec"),
        progress_callback=publish_progress,
        gbm_store=gbm_store,
    )
    response_path.write_text(json.dumps({"ok": True, "result": result}, default=str), encoding="utf-8")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if len(args) != 2:
        print("Usage: python -m py_lucidum.tools.glm.tabulation_worker REQUEST_JSON RESPONSE_JSON", file=sys.stderr)
        return 2
    response_path = Path(args[1])
    try:
        return run_worker(Path(args[0]), response_path)
    except Exception as exc:  # returned to the parent tabulation job
        response_path.write_text(
            json.dumps(
                {
                    "ok": False,
                    "error": str(exc),
                    "error_type": type(exc).__name__,
                    "traceback": traceback.format_exc(limit=8),
                }
            ),
            encoding="utf-8",
        )
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
