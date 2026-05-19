from __future__ import annotations

import os
import socket
import threading
import time
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from urllib.request import urlopen

import uvicorn

from py_lucidum.app import create_app


try:
    from playwright.sync_api import sync_playwright
except ImportError:  # pragma: no cover - exercised only without optional test deps.
    sync_playwright = None


RUN_BROWSER_TESTS = os.environ.get("PY_LUCIDUM_RUN_BROWSER_TESTS") == "1"


class BrowserSmokeTests(unittest.TestCase):
    @unittest.skipUnless(RUN_BROWSER_TESTS, "set PY_LUCIDUM_RUN_BROWSER_TESTS=1 to run browser smoke tests")
    @unittest.skipUnless(sync_playwright is not None, "playwright is not installed")
    def test_chart_and_map_tools_load_and_switch_without_extra_api_requests(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            data_path = Path(tmp_dir) / "sample.csv"
            data_path.write_text(
                "PostcodeArea,PostcodeSector,vehicle_age,price,value\n"
                "AB,AB10 1,1,100,10\n"
                "AB,AB10 1,2,200,20\n"
                "AL,AL1 1,3,300,30\n"
                "AL,AL1 2,4,400,40\n",
                encoding="utf-8",
            )
            base_url, server, thread = self.start_app(data_path)
            try:
                self.assert_static_asset(base_url, "/static/app.css", "text/css")
                self.assert_static_asset(base_url, "/static/app.js", "text/javascript")
                self.exercise_browser(base_url)
            finally:
                server.should_exit = True
                thread.join(timeout=5)

    @unittest.skipUnless(RUN_BROWSER_TESTS, "set PY_LUCIDUM_RUN_BROWSER_TESTS=1 to run browser smoke tests")
    @unittest.skipUnless(sync_playwright is not None, "playwright is not installed")
    def test_saved_filter_theme_headings_collapse_their_rows(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            data_path = tmp_path / "sample.csv"
            data_path.write_text(
                "DRIVER_AGE,POSTCODE_AREA,vehicle_age,price,value\n"
                "25,PO,1,100,10\n"
                "45,SO,2,200,20\n"
                "75,B,3,300,30\n",
                encoding="utf-8",
            )
            filters_path = tmp_path / "filters.csv"
            filters_path.write_text(
                "theme,name,expression\n"
                "DRIVER AGE,Young drivers,DRIVER_AGE < 30\n"
                "DRIVER AGE,Older drivers,DRIVER_AGE > 70\n"
                "POSTCODE AREA,Portsmouth,POSTCODE_AREA = 'PO'\n",
                encoding="utf-8",
            )
            base_url, server, thread = self.start_app(data_path, filters_path=filters_path, use_saved_filters=True)
            try:
                self.exercise_saved_filter_theme_collapse(base_url)
            finally:
                server.should_exit = True
                thread.join(timeout=5)

    @staticmethod
    def start_app(
        data_path: Path,
        *,
        filters_path: Path | None = None,
        use_saved_filters: bool = False,
    ) -> tuple[str, uvicorn.Server, threading.Thread]:
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            port = int(sock.getsockname()[1])
        app = create_app(
            data_path,
            defaults={
                "x": "vehicle_age",
                "actual": "price",
                "denominator": "value",
            },
            filters_path=filters_path,
            use_saved_filters=use_saved_filters,
        )
        config = uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning", access_log=False)
        server = uvicorn.Server(config)
        thread = threading.Thread(target=server.run, name="py-lucidum-browser-smoke", daemon=True)
        thread.start()
        for _ in range(100):
            if server.started:
                break
            time.sleep(0.05)
        if not server.started:
            server.should_exit = True
            thread.join(timeout=5)
            raise RuntimeError("Uvicorn did not start for browser smoke test")
        return f"http://127.0.0.1:{port}", server, thread

    @staticmethod
    def assert_static_asset(base_url: str, path: str, expected_content_type: str) -> None:
        with urlopen(f"{base_url}{path}", timeout=5) as response:
            assert response.status == 200
            assert expected_content_type in response.headers.get("content-type", "")

    def exercise_browser(self, base_url: str) -> None:
        assert sync_playwright is not None
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch()
            page = browser.new_page(viewport={"width": 1280, "height": 800})
            page_errors: list[str] = []
            chart_requests = 0
            map_requests = 0

            page.on("pageerror", lambda error: page_errors.append(str(error)))

            def count_request(request: object) -> None:
                nonlocal chart_requests, map_requests
                url = request.url
                if url.endswith("/api/chart"):
                    chart_requests += 1
                elif url.endswith("/api/uk-map/summary"):
                    map_requests += 1

            page.on("request", count_request)
            try:
                page.goto(base_url, wait_until="domcontentloaded")
                page.locator("#datasetMeta").get_by_text("sample.csv").wait_for(timeout=10_000)
                page.locator("#chart:not(.hidden)").wait_for(timeout=10_000)

                page.locator("#ukMapTool").click()
                page.locator("#ukMap:not(.hidden)").wait_for(timeout=20_000)
                page.locator("#mapFloatingControl:not(.hidden)").wait_for(timeout=10_000)
                page.wait_for_function("() => window.L && document.querySelector('#ukMap .leaflet-pane')")

                page.locator("#lineBarTool").click()
                page.locator("#chart:not(.hidden)").wait_for(timeout=10_000)

                page.locator("#ukMapTool").click()
                page.locator("#ukMap:not(.hidden)").wait_for(timeout=10_000)

                self.assertEqual(page_errors, [])
                self.assertEqual(chart_requests, 1)
                self.assertEqual(map_requests, 1)
            finally:
                browser.close()

    def exercise_saved_filter_theme_collapse(self, base_url: str) -> None:
        assert sync_playwright is not None
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch()
            page = browser.new_page(viewport={"width": 1280, "height": 800})
            page_errors: list[str] = []
            page.on("pageerror", lambda error: page_errors.append(str(error)))
            try:
                page.goto(base_url, wait_until="domcontentloaded")
                page.locator("#datasetMeta").get_by_text("sample.csv").wait_for(timeout=10_000)

                driver_heading = page.locator('.saved-filter-theme[data-filter-theme="DRIVER AGE"]')
                driver_rows = page.locator('.saved-filter-option[data-filter-theme="DRIVER AGE"]')
                postcode_rows = page.locator('.saved-filter-option[data-filter-theme="POSTCODE AREA"]')

                driver_heading.wait_for(timeout=10_000)
                self.assertEqual(driver_heading.get_attribute("aria-expanded"), "true")
                self.assertTrue(driver_rows.first.is_visible())
                self.assertTrue(postcode_rows.first.is_visible())

                driver_heading.click()
                self.assertEqual(driver_heading.get_attribute("aria-expanded"), "false")
                self.assertFalse(driver_rows.first.is_visible())
                self.assertTrue(postcode_rows.first.is_visible())

                driver_heading.click()
                self.assertEqual(driver_heading.get_attribute("aria-expanded"), "true")
                self.assertTrue(driver_rows.first.is_visible())
                self.assertEqual(page_errors, [])
            finally:
                browser.close()


if __name__ == "__main__":
    unittest.main()
