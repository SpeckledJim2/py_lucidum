from __future__ import annotations

import json
import sys
import traceback
from pathlib import Path
from typing import Any

from py_lucidum.core import Dataset

from .overlay import (
    _build_glm_partial_dependence_overlay_from_context_impl,
    _build_glm_partial_dependence_overlay_impl,
)


def dataset_signature(path: Path) -> tuple[int, int]:
    stat = path.stat()
    return stat.st_mtime_ns, stat.st_size


def cached_dataset(dataset_path: Path, cache: dict[str, tuple[tuple[int, int], Dataset]] | None) -> Dataset:
    if cache is None:
        return Dataset(dataset_path)
    key = str(dataset_path)
    signature = dataset_signature(dataset_path)
    cached = cache.get(key)
    if cached and cached[0] == signature:
        return cached[1]
    dataset = Dataset(dataset_path)
    cache[key] = (signature, dataset)
    return dataset


def error_response(exc: Exception) -> dict[str, Any]:
    return {
        "ok": False,
        "error": str(exc),
        "error_type": type(exc).__name__,
        "traceback": traceback.format_exc(limit=8),
    }


def run_worker(request_path: Path, response_path: Path, *, dataset_cache: dict[str, tuple[tuple[int, int], Dataset]] | None = None) -> int:
    request = json.loads(request_path.read_text(encoding="utf-8"))
    dataset_path = Path(str(request["dataset_path"]))
    chart_request = dict(request.get("request") or {})
    feature_spec = request.get("feature_spec")
    x_col = str(request.get("x_col") or "")
    x_group_kind = str(request.get("x_group_kind") or "")
    denominator = dict(request.get("denominator") or {})
    result = _build_glm_partial_dependence_overlay_from_context_impl(
        chart_request,
        feature_spec=feature_spec,
        x_col=x_col,
        x_group_kind=x_group_kind,
        denominator=denominator,
        chart_context=request.get("chart_context"),
        model_context=request.get("model_context"),
        source_columns=request.get("source_columns"),
        kinds=request.get("kinds"),
    )
    if result is None:
        dataset = cached_dataset(dataset_path, dataset_cache)
        result = _build_glm_partial_dependence_overlay_impl(
            dataset,
            chart_request,
            feature_spec=feature_spec,
            x_col=x_col,
            x_sql=dict(request.get("x_sql") or {}),
            x_group_kind=x_group_kind,
            denominator=denominator,
        )
    response_path.write_text(json.dumps({"ok": True, "result": result}, default=str), encoding="utf-8")
    return 0


def run_server() -> int:
    dataset_cache: dict[str, tuple[tuple[int, int], Dataset]] = {}
    for line in sys.stdin:
        try:
            message = json.loads(line)
        except json.JSONDecodeError:
            continue
        if str(message.get("command") or "") == "shutdown":
            return 0
        request_id = str(message.get("request_id") or "")
        response_path = Path(str(message.get("response_path") or ""))
        try:
            run_worker(
                Path(str(message.get("request_path") or "")),
                response_path,
                dataset_cache=dataset_cache,
            )
            ack = {"request_id": request_id, "ok": True}
        except Exception as exc:  # returned to the parent chart request
            if response_path:
                response_path.write_text(json.dumps(error_response(exc)), encoding="utf-8")
            ack = {"request_id": request_id, "ok": True}
        print(json.dumps(ack), flush=True)
    return 0


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if args == ["--server"]:
        return run_server()
    if len(args) != 2:
        print("Usage: python -m py_lucidum.tools.glm.overlay_worker REQUEST_JSON RESPONSE_JSON | --server", file=sys.stderr)
        return 2
    response_path = Path(args[1])
    try:
        return run_worker(Path(args[0]), response_path)
    except Exception as exc:  # returned to the parent chart request
        response_path.write_text(json.dumps(error_response(exc)), encoding="utf-8")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
