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
                "DRIVER AGE,Middle aged drivers,DRIVER_AGE >= 30 AND DRIVER_AGE < 60\n"
                "DRIVER AGE,Older drivers,DRIVER_AGE > 70\n"
                "POSTCODE AREA,Portsmouth,POSTCODE_AREA = 'PO'\n"
                "POSTCODE AREA,Southampton,POSTCODE_AREA = 'SO'\n",
                encoding="utf-8",
            )
            base_url, server, thread = self.start_app(data_path, filters_path=filters_path, use_saved_filters=True)
            try:
                self.exercise_saved_filter_theme_collapse(base_url)
            finally:
                server.should_exit = True
                thread.join(timeout=5)

    @unittest.skipUnless(RUN_BROWSER_TESTS, "set PY_LUCIDUM_RUN_BROWSER_TESTS=1 to run browser smoke tests")
    @unittest.skipUnless(sync_playwright is not None, "playwright is not installed")
    def test_kpi_rows_select_metrics_and_survive_tool_switch(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            data_path = tmp_path / "sample.csv"
            data_path.write_text(
                "PostcodeArea,PostcodeSector,vehicle_age,price,value,rate\n"
                "AB,AB10 1,1,100,10,0.1\n"
                "AB,AB10 1,2,200,20,0.2\n"
                "AL,AL1 1,3,300,30,0.3\n",
                encoding="utf-8",
            )
            kpis_path = tmp_path / "kpi_spec.csv"
            kpis_path.write_text(
                "group,name,actual,denominator,decimals,format\n"
                "PRICE,Price,price,N,2,currency\n"
                "VALUE,Value,value,N,1,number\n"
                "RATE,Rate,rate,N,1,percent\n",
                encoding="utf-8",
            )
            base_url, server, thread = self.start_app(
                data_path,
                defaults={"x": "vehicle_age"},
                kpis_path=kpis_path,
                use_kpis=True,
            )
            try:
                self.exercise_kpi_selection(base_url)
            finally:
                server.should_exit = True
                thread.join(timeout=5)

    @unittest.skipUnless(RUN_BROWSER_TESTS, "set PY_LUCIDUM_RUN_BROWSER_TESTS=1 to run browser smoke tests")
    @unittest.skipUnless(sync_playwright is not None, "playwright is not installed")
    def test_missing_token_boot_error_is_visible(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            data_path = Path(tmp_dir) / "sample.csv"
            data_path.write_text(
                "PostcodeArea,PostcodeSector,vehicle_age,price,value\n"
                "AB,AB10 1,1,100,10\n",
                encoding="utf-8",
            )
            base_url, server, thread = self.start_app(data_path, token="dev-token")
            try:
                self.exercise_missing_token_boot_error(base_url)
            finally:
                server.should_exit = True
                thread.join(timeout=5)

    @staticmethod
    def start_app(
        data_path: Path,
        *,
        filters_path: Path | None = None,
        use_saved_filters: bool = False,
        kpis_path: Path | None = None,
        use_kpis: bool = False,
        token: str | None = None,
        defaults: dict[str, str] | None = None,
    ) -> tuple[str, uvicorn.Server, threading.Thread]:
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            port = int(sock.getsockname()[1])
        app = create_app(
            data_path,
            defaults=defaults or {
                "x": "vehicle_age",
                "actual": "price",
                "denominator": "value",
            },
            filters_path=filters_path,
            use_saved_filters=use_saved_filters,
            kpis_path=kpis_path,
            use_kpis=use_kpis,
            token=token,
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
            profile_requests = 0
            profile_detail_requests = 0
            chart_requests = 0
            map_requests = 0

            page.on("pageerror", lambda error: page_errors.append(str(error)))

            def count_request(request: object) -> None:
                nonlocal profile_requests, profile_detail_requests, chart_requests, map_requests
                url = request.url
                if url.endswith("/api/column-profile/summary"):
                    profile_requests += 1
                elif url.endswith("/api/column-profile/detail"):
                    profile_detail_requests += 1
                elif url.endswith("/api/chart"):
                    chart_requests += 1
                elif url.endswith("/api/uk-map/summary"):
                    map_requests += 1

            page.on("request", count_request)
            try:
                page.goto(base_url, wait_until="domcontentloaded")
                page.locator("#datasetMeta").get_by_text("sample.csv").wait_for(timeout=10_000)
                page.locator("#profileWrap:not(.hidden) .profile-table").wait_for(timeout=10_000)
                page.locator('#profileWrap .profile-summary-row[aria-selected="true"]').wait_for(timeout=10_000)
                page.locator("#profileDetailTitle").get_by_text("PostcodeArea").wait_for(timeout=10_000)
                self.assertEqual(page.locator("#profileFilter").evaluate("node => getComputedStyle(node).fontSize"), "10px")
                page.locator('#profileWrap .profile-summary-row[data-profile-column="vehicle_age"]').click()
                page.locator('#profileWrap .profile-summary-row[data-profile-column="vehicle_age"][aria-selected="true"]').wait_for(timeout=10_000)
                page.locator("#profileDetailTitle").get_by_text("vehicle_age").wait_for(timeout=10_000)
                page.locator('#profileWrap .profile-sort-button[data-profile-sort="distinct"]').click()
                page.wait_for_function(
                    '() => document.querySelector("#profileWrap tbody td.profile-column-name")?.textContent === "PostcodeArea"'
                )
                self.assertEqual(page.locator('#profileWrap .profile-summary-row[data-profile-column="vehicle_age"]').get_attribute("aria-selected"), "true")
                page.locator('#profileWrap .profile-sort-button[data-profile-sort="distinct"]').click()
                page.wait_for_function(
                    '() => document.querySelector("#profileWrap tbody td.profile-column-name")?.textContent === "vehicle_age"'
                )
                self.assertEqual(page.locator('#profileWrap .profile-summary-row[data-profile-column="vehicle_age"]').get_attribute("aria-selected"), "true")
                page.wait_for_function(
                    """
                    () => {
                        const text = document.querySelector("#actionTimingMonitor")?.textContent || "";
                        return /^DuckDB: \\d+(?:ns|us|ms), Profile render: \\d+(?:ns|us|ms)$/.test(text);
                    }
                    """
                )

                page.locator("#sidebarToggleBtn").click()
                self.assertEqual(page.locator("#sidebarToggleBtn").get_attribute("aria-expanded"), "false")
                self.assertIsNone(page.locator("#appSidebar").get_attribute("aria-hidden"))
                self.assertTrue(page.locator("#profileTool").is_visible())
                self.assertTrue(page.locator("#lineBarTool").is_visible())
                self.assertTrue(page.locator("#ukMapTool").is_visible())
                self.assertFalse(page.locator(".sidebar-kpi-section").is_visible())
                self.assertFalse(page.locator(".sidebar-filter-section").is_visible())
                self.assertFalse(page.locator("#sidebarResizer").is_visible())

                page.locator("#reloadBtn").click()
                page.wait_for_function('() => document.querySelector("#sidebarToggleBtn")?.getAttribute("aria-expanded") === "false"')
                self.assertTrue(page.locator("#profileTool").is_visible())
                self.assertTrue(page.locator("#lineBarTool").is_visible())
                self.assertTrue(page.locator("#ukMapTool").is_visible())
                self.assertFalse(page.locator(".sidebar-kpi-section").is_visible())
                self.assertFalse(page.locator(".sidebar-filter-section").is_visible())
                self.assertFalse(page.locator("#sidebarResizer").is_visible())

                page.locator("#ukMapTool").click()
                page.locator("#ukMap:not(.hidden)").wait_for(timeout=20_000)
                page.locator("#mapFloatingControl:not(.hidden)").wait_for(timeout=10_000)
                page.wait_for_function("() => window.L && document.querySelector('#ukMap .leaflet-pane')")
                page.wait_for_function("() => document.querySelector('#ukMap')?.classList.contains('map-bg-light')")
                page.locator("#themeBtn").click()
                page.wait_for_function("() => document.querySelector('#ukMap')?.classList.contains('map-bg-dark')")
                page.wait_for_function(
                    """
                    () => {
                        const text = document.querySelector("#actionTimingMonitor")?.textContent || "";
                        return /^DuckDB: \\d+(?:ns|us|ms), Map render: \\d+(?:ns|us|ms)$/.test(text);
                    }
                    """
                )

                page.locator("#lineBarTool").click()
                page.locator("#chart:not(.hidden)").wait_for(timeout=10_000)
                page.wait_for_function(
                    """
                    () => {
                        const text = document.querySelector("#actionTimingMonitor")?.textContent || "";
                        return /^DuckDB: \\d+(?:ns|us|ms), Chart render: \\d+(?:ns|us|ms)$/.test(text);
                    }
                    """
                )

                page.locator("#sidebarToggleBtn").click()
                self.assertEqual(page.locator("#sidebarToggleBtn").get_attribute("aria-expanded"), "true")
                self.assertTrue(page.locator(".sidebar-kpi-section").is_visible())
                self.assertTrue(page.locator(".sidebar-filter-section").is_visible())

                page.locator("#ukMapTool").click()
                page.locator("#ukMap:not(.hidden)").wait_for(timeout=10_000)

                self.assertEqual(page_errors, [])
                self.assertEqual(profile_requests, 2)
                self.assertEqual(profile_detail_requests, 3)
                self.assertEqual(chart_requests, 1)
                self.assertEqual(map_requests, 1)
            finally:
                browser.close()

    def exercise_missing_token_boot_error(self, base_url: str) -> None:
        assert sync_playwright is not None
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch()
            page = browser.new_page(viewport={"width": 1280, "height": 800})
            page_errors: list[str] = []
            page.on("pageerror", lambda error: page_errors.append(str(error)))
            try:
                page.goto(base_url, wait_until="domcontentloaded")
                page.locator("#datasetMeta").get_by_text("Dataset failed to load").wait_for(timeout=10_000)
                page.locator("#status").get_by_text("Invalid or missing app token").wait_for(timeout=10_000)
                self.assertEqual(page_errors, [])
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

                self.assertEqual(page.locator("#filterCollapseBtn").get_attribute("aria-expanded"), "false")
                self.assertFalse(driver_heading.is_visible())
                page.locator("#filterCollapseBtn").click()
                self.assertEqual(page.locator("#filterCollapseBtn").get_attribute("aria-expanded"), "true")
                driver_heading.wait_for(timeout=10_000)
                self.assertEqual(driver_heading.get_attribute("aria-expanded"), "false")
                self.assertFalse(driver_rows.first.is_visible())
                self.assertFalse(postcode_rows.first.is_visible())
                driver_heading.click()
                page.locator('.saved-filter-theme[data-filter-theme="POSTCODE AREA"]').click()
                self.assertEqual(driver_heading.get_attribute("aria-expanded"), "true")
                self.assertTrue(driver_rows.first.is_visible())
                self.assertTrue(postcode_rows.first.is_visible())

                self.assertTrue(page.locator('.filter-selection-mode button[data-value="single"]').evaluate("node => node.classList.contains('active')"))
                page.locator('.filter-selection-mode button[data-value="multi"]').click()
                driver_rows.first.click()
                postcode_rows.first.click()
                self.assertEqual(driver_rows.first.get_attribute("aria-selected"), "true")
                self.assertEqual(postcode_rows.first.get_attribute("aria-selected"), "true")
                self.assertIn("DRIVER_AGE < 30", page.locator("#filterInput").input_value())
                self.assertIn("POSTCODE_AREA = 'PO'", page.locator("#filterInput").input_value())
                self.assertEqual(page.locator("#filterInput").input_value(), "(DRIVER_AGE < 30) AND (POSTCODE_AREA = 'PO')")

                driver_rows.first.click()
                self.assertEqual(driver_rows.first.get_attribute("aria-selected"), "false")
                self.assertEqual(postcode_rows.first.get_attribute("aria-selected"), "true")
                self.assertNotIn("DRIVER_AGE < 30", page.locator("#filterInput").input_value())
                self.assertIn("POSTCODE_AREA = 'PO'", page.locator("#filterInput").input_value())
                self.assertEqual(page.locator("#filterInput").input_value(), "POSTCODE_AREA = 'PO'")

                postcode_rows.first.click()
                driver_rows.nth(1).click()
                postcode_rows.nth(1).click()
                page.locator('.filter-operator button[data-value="or"]').click()
                self.assertEqual(
                    page.locator("#filterInput").input_value(),
                    "(DRIVER_AGE >= 30 AND DRIVER_AGE < 60) OR (POSTCODE_AREA = 'SO')",
                )

                driver_rows.nth(2).click()
                self.assertEqual(
                    page.locator("#filterInput").input_value(),
                    "(DRIVER_AGE >= 30 AND DRIVER_AGE < 60) OR (DRIVER_AGE > 70) OR (POSTCODE_AREA = 'SO')",
                )
                page.locator('.filter-selection-mode button[data-value="grouped"]').click()
                self.assertFalse(page.locator('.filter-operator button[data-value="and"]').is_visible())
                self.assertEqual(driver_rows.nth(1).get_attribute("aria-selected"), "true")
                self.assertEqual(driver_rows.nth(2).get_attribute("aria-selected"), "true")
                self.assertEqual(postcode_rows.nth(1).get_attribute("aria-selected"), "true")
                self.assertEqual(
                    page.locator("#filterInput").input_value(),
                    "((DRIVER_AGE >= 30 AND DRIVER_AGE < 60) OR (DRIVER_AGE > 70)) AND (POSTCODE_AREA = 'SO')",
                )
                page.locator('.filter-selection-mode button[data-value="multi"]').click()
                self.assertTrue(page.locator('.filter-operator button[data-value="and"]').is_visible())
                self.assertEqual(
                    page.locator("#filterInput").input_value(),
                    "(DRIVER_AGE >= 30 AND DRIVER_AGE < 60) OR (DRIVER_AGE > 70) OR (POSTCODE_AREA = 'SO')",
                )
                page.locator('.filter-selection-mode button[data-value="grouped"]').click()
                page.locator('.filter-selection-mode button[data-value="single"]').click()
                self.assertEqual(driver_rows.nth(1).get_attribute("aria-selected"), "true")
                self.assertEqual(driver_rows.nth(2).get_attribute("aria-selected"), "false")
                self.assertEqual(postcode_rows.nth(1).get_attribute("aria-selected"), "false")
                self.assertEqual(page.locator("#filterInput").input_value(), "DRIVER_AGE >= 30 AND DRIVER_AGE < 60")
                page.locator('.filter-selection-mode button[data-value="grouped"]').click()
                self.assertEqual(page.locator("#filterInput").input_value(), "DRIVER_AGE >= 30 AND DRIVER_AGE < 60")
                page.locator('.filter-selection-mode button[data-value="single"]').click()

                self.assertFalse(page.locator('.filter-operator button[data-value="and"]').is_visible())
                self.assertTrue(page.locator("#filterSidebarClearBtn").is_visible())
                page.locator("#filterCollapseBtn").click()
                self.assertTrue(page.locator("#filterSidebarClearBtn").is_visible())
                page.locator("#filterCollapseBtn").click()
                driver_rows.first.click()
                self.assertEqual(driver_rows.first.get_attribute("aria-selected"), "true")
                self.assertEqual(postcode_rows.first.get_attribute("aria-selected"), "false")
                postcode_rows.first.click()
                self.assertEqual(driver_rows.first.get_attribute("aria-selected"), "false")
                self.assertEqual(postcode_rows.first.get_attribute("aria-selected"), "true")
                page.locator("#filterSidebarClearBtn").click()
                self.assertEqual(postcode_rows.first.get_attribute("aria-selected"), "false")
                self.assertEqual(page.locator("#filterInput").input_value(), "")

                footer_value = page.locator("#filterInput").input_value()
                page.locator("#filterFooterToggleBtn").click()
                self.assertEqual(page.locator("#filterFooter").get_attribute("aria-hidden"), "true")
                self.assertFalse(page.locator("#filterInput").is_visible())
                page.locator("#filterFooterToggleBtn").click()
                self.assertEqual(page.locator("#filterFooter").get_attribute("aria-hidden"), "false")
                self.assertEqual(page.locator("#filterInput").input_value(), footer_value)

                driver_heading.click()
                self.assertEqual(driver_heading.get_attribute("aria-expanded"), "false")
                self.assertFalse(driver_rows.first.is_visible())
                self.assertTrue(postcode_rows.first.is_visible())

                driver_heading.click()
                self.assertEqual(driver_heading.get_attribute("aria-expanded"), "true")
                self.assertTrue(driver_rows.first.is_visible())

                driver_rows.first.click()
                self.assertEqual(driver_rows.first.get_attribute("aria-selected"), "true")
                self.assertEqual(page.locator("#filterInput").input_value(), "DRIVER_AGE < 30")
                page.locator("#reloadBtn").click()
                page.wait_for_function(
                    """
                    () => {
                        const heading = document.querySelector('.saved-filter-theme[data-filter-theme="DRIVER AGE"]');
                        const row = document.querySelector('.saved-filter-option[data-filter-theme="DRIVER AGE"]');
                        return heading &&
                            row &&
                            heading.getAttribute("aria-expanded") === "true" &&
                            row.getAttribute("aria-selected") === "true";
                    }
                    """
                )
                self.assertEqual(page.locator("#filterCollapseBtn").get_attribute("aria-expanded"), "true")
                self.assertTrue(driver_rows.first.is_visible())
                self.assertEqual(driver_rows.first.get_attribute("aria-selected"), "true")
                self.assertEqual(page.locator("#filterInput").input_value(), "DRIVER_AGE < 30")
                self.assertEqual(page_errors, [])
            finally:
                browser.close()

    def exercise_kpi_selection(self, base_url: str) -> None:
        assert sync_playwright is not None
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch()
            page = browser.new_page(viewport={"width": 1280, "height": 800})
            page_errors: list[str] = []
            page.on("pageerror", lambda error: page_errors.append(str(error)))
            try:
                page.goto(base_url, wait_until="domcontentloaded")
                page.locator("#datasetMeta").get_by_text("sample.csv").wait_for(timeout=10_000)
                page.locator("#lineBarTool").click()
                page.locator("#kpiSelect .kpi-theme").first.wait_for(timeout=10_000)
                price_row = page.locator('.kpi-option[data-kpi-group="PRICE"]')
                self.assertEqual(page.locator("#actualNumerator").input_value(), "price")
                self.assertEqual(page.locator("#denominator").input_value(), "__none__")
                self.assertEqual(price_row.get_attribute("aria-selected"), "true")

                value_heading = page.locator('.kpi-theme[data-kpi-group="VALUE"]')
                value_row = page.locator('.kpi-option[data-kpi-group="VALUE"]')
                if value_heading.get_attribute("aria-expanded") == "false":
                    value_heading.click()
                value_row.click()

                self.assertEqual(page.locator("#actualNumerator").input_value(), "value")
                self.assertEqual(page.locator("#denominator").input_value(), "__none__")
                self.assertEqual(value_row.get_attribute("aria-selected"), "true")

                rate_heading = page.locator('.kpi-theme[data-kpi-group="RATE"]')
                rate_row = page.locator('.kpi-option[data-kpi-group="RATE"]')
                if rate_heading.get_attribute("aria-expanded") == "false":
                    rate_heading.click()
                rate_row.click()
                page.locator("#actualMetricTitle").get_by_text("20.0%").wait_for(timeout=10_000)
                self.assertEqual(page.locator("#actualNumerator").input_value(), "rate")
                self.assertEqual(rate_row.get_attribute("aria-selected"), "true")

                page.locator("#ukMapTool").click()
                page.locator("#ukMap:not(.hidden)").wait_for(timeout=20_000)
                page.locator("#mapFloatingControl:not(.hidden)").wait_for(timeout=10_000)
                page.locator("#lineBarTool").click()
                page.locator("#chart:not(.hidden)").wait_for(timeout=10_000)

                self.assertEqual(page_errors, [])
            finally:
                browser.close()


if __name__ == "__main__":
    unittest.main()
