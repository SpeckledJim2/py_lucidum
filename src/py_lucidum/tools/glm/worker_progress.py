from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any


PROGRESS_REPLACE_ATTEMPTS = 8
PROGRESS_REPLACE_DELAY_SECONDS = 0.005


def write_worker_progress(
    path: Path,
    progress: dict[str, Any],
    *,
    pending_path: Path | None = None,
) -> None:
    pending = pending_path or path.with_name(f"{path.name}.tmp")
    pending.write_text(json.dumps(progress, default=str), encoding="utf-8")
    replace_worker_progress(pending, path)


def replace_worker_progress(pending_path: Path, path: Path) -> None:
    for attempt in range(PROGRESS_REPLACE_ATTEMPTS):
        try:
            pending_path.replace(path)
            return
        except PermissionError:
            if attempt == PROGRESS_REPLACE_ATTEMPTS - 1:
                raise
            time.sleep(PROGRESS_REPLACE_DELAY_SECONDS * (attempt + 1))
