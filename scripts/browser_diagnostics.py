"""Failure-only Playwright diagnostics for the browser smoke suite."""

from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Callable


def _safe_name(value: str) -> str:
    name = re.sub(r"[^A-Za-z0-9_.-]+", "-", value).strip("-.")
    return (name or "browser-smoke")[-180:]


def _current_test_name() -> str:
    return _safe_name(os.environ.get("PYTEST_CURRENT_TEST", "browser-smoke").removesuffix(" (call)"))


def _safe_url(value: str) -> str:
    return re.sub(r"([?&]token=)[^&]*", r"\1[redacted]", value, flags=re.IGNORECASE)


class _TrackedContext:
    def __init__(self, raw: Any, artifact_dir: Path, label: str) -> None:
        self.raw = raw
        self.artifact_dir = artifact_dir
        self.label = label
        self.pages: list[Any] = []
        self.events: list[str] = []
        self.finalized = False
        self.trace_started = False
        try:
            self.raw.tracing.start(screenshots=True, snapshots=True, sources=True)
            self.trace_started = True
        except Exception as exc:  # pragma: no cover - diagnostic fallback only.
            self.events.append(f"Could not start trace: {exc!r}")

    def track_page(self, page: Any) -> Any:
        if page not in self.pages:
            self.pages.append(page)
            page.on("console", lambda message: self.events.append(f"console.{message.type}: {message.text}"))
            page.on("pageerror", lambda error: self.events.append(f"pageerror: {error}"))
            page.on(
                "requestfailed",
                lambda request: self.events.append(
                    f"requestfailed: {request.method} {_safe_url(request.url)} ({request.failure or 'unknown'})"
                ),
            )
            page.on(
                "response",
                lambda response: self.events.append(
                    f"response: {response.status} {response.request.method} {_safe_url(response.url)}"
                )
                if response.status >= 400
                else None,
            )
        return page

    def finalize(self, failed: bool) -> None:
        if self.finalized:
            return
        self.finalized = True
        test_name = _current_test_name()
        stem = _safe_name(f"{test_name}-{self.label}")
        if failed:
            self.artifact_dir.mkdir(parents=True, exist_ok=True)
            for index, page in enumerate(self.pages, start=1):
                try:
                    if not page.is_closed():
                        page.screenshot(
                            path=str(self.artifact_dir / f"{stem}-page-{index}.png"),
                            full_page=True,
                        )
                except Exception as exc:  # pragma: no cover - diagnostic fallback only.
                    self.events.append(f"Could not capture page {index}: {exc!r}")
            try:
                metadata = {
                    "test": os.environ.get("PYTEST_CURRENT_TEST", ""),
                    "pages": [_safe_url(page.url) for page in self.pages if not page.is_closed()],
                    "events": self.events,
                }
                (self.artifact_dir / f"{stem}.json").write_text(
                    json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
                    encoding="utf-8",
                )
            except Exception:  # pragma: no cover - diagnostic fallback only.
                pass
        if self.trace_started:
            try:
                if failed:
                    self.raw.tracing.stop(path=str(self.artifact_dir / f"{stem}-trace.zip"))
                else:
                    self.raw.tracing.stop()
            except Exception as exc:  # pragma: no cover - diagnostic fallback only.
                if failed:
                    try:
                        (self.artifact_dir / f"{stem}-trace-error.txt").write_text(
                            f"{exc!r}\n",
                            encoding="utf-8",
                        )
                    except Exception:
                        pass


class _ContextProxy:
    def __init__(self, tracked: _TrackedContext) -> None:
        self._tracked = tracked

    def new_page(self, *args: Any, **kwargs: Any) -> Any:
        return self._tracked.track_page(self._tracked.raw.new_page(*args, **kwargs))

    def close(self, *args: Any, **kwargs: Any) -> Any:
        self._tracked.finalize(sys.exc_info()[0] is not None)
        return self._tracked.raw.close(*args, **kwargs)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._tracked.raw, name)


class _BrowserProxy:
    def __init__(self, raw: Any, artifact_dir: Path, browser_name: str) -> None:
        self._raw = raw
        self._artifact_dir = artifact_dir
        self._browser_name = browser_name
        self._contexts: dict[int, _TrackedContext] = {}

    def _track_context(self, raw_context: Any) -> _TrackedContext:
        key = id(raw_context)
        tracked = self._contexts.get(key)
        if tracked is None:
            label = f"{self._browser_name}-{len(self._contexts) + 1}"
            tracked = _TrackedContext(raw_context, self._artifact_dir, label)
            self._contexts[key] = tracked
        return tracked

    def new_page(self, *args: Any, **kwargs: Any) -> Any:
        page = self._raw.new_page(*args, **kwargs)
        return self._track_context(page.context).track_page(page)

    def new_context(self, *args: Any, **kwargs: Any) -> _ContextProxy:
        return _ContextProxy(self._track_context(self._raw.new_context(*args, **kwargs)))

    def finalize(self, failed: bool) -> None:
        for tracked in self._contexts.values():
            tracked.finalize(failed)

    def close(self, *args: Any, **kwargs: Any) -> Any:
        self.finalize(sys.exc_info()[0] is not None)
        return self._raw.close(*args, **kwargs)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._raw, name)


class _BrowserTypeProxy:
    def __init__(self, raw: Any, artifact_dir: Path, browser_name: str, browsers: list[_BrowserProxy]) -> None:
        self._raw = raw
        self._artifact_dir = artifact_dir
        self._browser_name = browser_name
        self._browsers = browsers

    def launch(self, *args: Any, **kwargs: Any) -> _BrowserProxy:
        browser = _BrowserProxy(
            self._raw.launch(*args, **kwargs),
            self._artifact_dir,
            self._browser_name,
        )
        self._browsers.append(browser)
        return browser

    def __getattr__(self, name: str) -> Any:
        return getattr(self._raw, name)


class _PlaywrightProxy:
    def __init__(self, raw: Any, artifact_dir: Path) -> None:
        self._raw = raw
        self._browsers: list[_BrowserProxy] = []
        self.chromium = _BrowserTypeProxy(raw.chromium, artifact_dir, "chromium", self._browsers)
        self.firefox = _BrowserTypeProxy(raw.firefox, artifact_dir, "firefox", self._browsers)
        self.webkit = _BrowserTypeProxy(raw.webkit, artifact_dir, "webkit", self._browsers)

    def finalize(self, failed: bool) -> None:
        for browser in self._browsers:
            browser.finalize(failed)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._raw, name)


class _PlaywrightManager:
    def __init__(self, raw_manager: Any, artifact_dir: Path) -> None:
        self._raw_manager = raw_manager
        self._artifact_dir = artifact_dir
        self._proxy: _PlaywrightProxy | None = None

    def __enter__(self) -> _PlaywrightProxy:
        self._proxy = _PlaywrightProxy(self._raw_manager.__enter__(), self._artifact_dir)
        return self._proxy

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> Any:
        if self._proxy is not None:
            self._proxy.finalize(exc_type is not None)
        return self._raw_manager.__exit__(exc_type, exc_value, traceback)


def instrument_sync_playwright(factory: Callable[[], Any]) -> Callable[[], Any]:
    """Wrap Playwright so CI retains traces and page state only when a test fails."""

    def instrumented() -> Any:
        artifact_dir = os.environ.get("PY_LUCIDUM_BROWSER_ARTIFACT_DIR")
        raw_manager = factory()
        if not artifact_dir:
            return raw_manager
        return _PlaywrightManager(raw_manager, Path(artifact_dir))

    return instrumented
