from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi.responses import FileResponse
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles


NO_STORE_CACHE_CONTROL = "no-store"


def mark_no_store(response: Any) -> Any:
    response.headers["Cache-Control"] = NO_STORE_CACHE_CONTROL
    return response


def no_store_file_response(path: str | Path, **kwargs: Any) -> FileResponse:
    return mark_no_store(FileResponse(path, **kwargs))


def no_store_html_response(content: str, **kwargs: Any) -> HTMLResponse:
    return mark_no_store(HTMLResponse(content, **kwargs))


class NoStoreStaticFiles(StaticFiles):
    async def get_response(self, path: str, scope: Any) -> Any:
        response = await super().get_response(path, scope)
        return mark_no_store(response)
