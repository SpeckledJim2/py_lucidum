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
from py_lucidum.tools.gbm.store import GbmModelStore


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
    def test_gbm_tool_loads_feature_grid(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            data_path = tmp_path / "sample.csv"
            data_path.write_text(
                "actualNumerator,denominator,Age,Segment,PostcodeArea,PostcodeSector,PostcodeUnit,lat,long,sample\n"
                "10,100,30,A,AB,AB10 1,AB10 1AA,57.1,-2.1,training\n"
                "20,200,40,B,AB,AB10 1,AB10 1AB,57.2,-2.2,test\n"
                "30,300,50,C,CD20 2,CD20 2AA,CD20 2AA,56.1,-1.1,training\n",
                encoding="utf-8",
            )
            kpis_path = tmp_path / "kpi_spec.csv"
            kpis_path.write_text(
                "group,name,actual,denominator,decimals,format\n"
                "MODEL,Actual numerator,actualNumerator,denominator,2,number\n",
                encoding="utf-8",
            )
            store = GbmModelStore(data_path)
            for model_id, label, learning_rate, created_at in (
                ("browser-smoke-model", "Browser smoke model", 0.11, "2026-05-25T00:00:00Z"),
                ("browser-smoke-model-2", "Second smoke model", 0.22, "2026-05-25T00:00:01Z"),
            ):
                model_dir = store.create_model_dir(model_id)
                store.write_json(
                    model_dir / "manifest.json",
                    {
                        "model_id": model_id,
                        "label": label,
                        "created_at": created_at,
                        "objective": "gamma",
                        "metric": "gamma",
                        "best_iteration": 3,
                        "training_rows": 2,
                        "test_rows": 1,
                        "feature_importance": [],
                        "sources": {},
                    },
                )
                store.write_json(
                    model_dir / "feature_config.json",
                    [
                        {"name": "Age", "kind": "integer", "include": True, "monotonicity": "Increasing", "gain": 5.0}
                        if model_id == "browser-smoke-model"
                        else {"name": "Segment", "kind": "categorical", "include": True, "monotonicity": "", "gain": 6.0}
                    ],
                )
                store.write_json(
                    model_dir / "parameters.json",
                    {
                        "objective": "gamma",
                        "metric": "gamma",
                        "learning_rate": learning_rate,
                        "num_iterations": 123 if model_id.endswith("-2") else 77,
                    },
                )
                store.write_json(
                    model_dir / "training_log.json",
                    {
                        "evaluation": {
                            "training": {"gamma": [7.38, 7.33, 7.31, 7.305, 7.301]},
                            "test": {"gamma": [7.37, 7.325, 7.3022, 7.303, 7.304]},
                        },
                        "warnings": [],
                    },
                )
                store.write_json(model_dir / "tree_dump.json", {"tree_info": []})
            store.activate_model("browser-smoke-model")
            base_url, server, thread = self.start_app(
                data_path,
                tools=["line_bar", "gbm"],
                kpis_path=kpis_path,
                use_kpis=True,
            )
            try:
                self.exercise_gbm_tool(base_url)
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

    @unittest.skipUnless(RUN_BROWSER_TESTS, "set PY_LUCIDUM_RUN_BROWSER_TESTS=1 to run browser smoke tests")
    @unittest.skipUnless(sync_playwright is not None, "playwright is not installed")
    def test_stopped_overlay_uses_cached_favicon_after_shutdown(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            data_path = Path(tmp_dir) / "sample.csv"
            data_path.write_text(
                "PostcodeArea,PostcodeSector,vehicle_age,price,value\n"
                "AB,AB10 1,1,100,10\n",
                encoding="utf-8",
            )
            base_url, server, thread = self.start_app(data_path)
            try:
                self.exercise_stopped_overlay(base_url)
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
        tools: list[str] | None = None,
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
            tools=tools,
        )
        config = uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning", access_log=False)
        server = uvicorn.Server(config)
        app.state.shutdown_callback = lambda: setattr(server, "should_exit", True)
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
                        return /^DuckDB: \\d+(?:ns|us|ms), JSON: \\d+ms, Profile render: \\d+(?:ns|us|ms), Total: \\d+ms$/.test(text);
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
                        return /^DuckDB: \\d+(?:ns|us|ms), JSON: \\d+ms, Map render: \\d+(?:ns|us|ms), Total: \\d+ms$/.test(text);
                    }
                    """
                )

                page.locator("#lineBarTool").click()
                page.locator("#chart:not(.hidden)").wait_for(timeout=10_000)
                page.wait_for_function(
                    """
                    () => {
                        const text = document.querySelector("#actionTimingMonitor")?.textContent || "";
                        return /^DuckDB: \\d+(?:ns|us|ms), JSON: \\d+ms, Chart render: \\d+(?:ns|us|ms), Total: \\d+ms$/.test(text);
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

    def exercise_gbm_tool(self, base_url: str) -> None:
        assert sync_playwright is not None
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch()
            page = browser.new_page(viewport={"width": 1280, "height": 800})
            page_errors: list[str] = []
            page.on("pageerror", lambda error: page_errors.append(str(error)))
            try:
                page.goto(f"{base_url}/?tool=gbm", wait_until="domcontentloaded")
                page.get_by_text("Features and parameters").wait_for(timeout=10_000)
                page.locator("#gbmFeatureGrid").wait_for(timeout=10_000)
                page.get_by_text("Train GBM").wait_for(timeout=10_000)
                page.get_by_text("Gain").first.wait_for(timeout=10_000)
                page.get_by_text("SHAP rows").wait_for(timeout=10_000)
                page.get_by_text("Features", exact=True).wait_for(timeout=10_000)
                page.get_by_text("Parameters", exact=True).wait_for(timeout=10_000)
                page.get_by_text("Evaluation log", exact=True).wait_for(timeout=10_000)
                page.locator("#gbmActiveModelSelect").wait_for(timeout=10_000)
                self.assertEqual(
                    page.locator("#gbmParameterGrid .tabulator-row", has_text="learning_rate").locator(".tabulator-cell[tabulator-field='value']").text_content(),
                    "0.11",
                )
                page.locator("#gbmParameterGrid .tabulator-row", has_text="objective").locator(".tabulator-cell[tabulator-field='value']").click()
                page.locator("#gbmParameterGrid select.gbm-parameter-select").wait_for(timeout=10_000)
                page.keyboard.press("Escape")
                page.locator("#gbmParameterGrid .tabulator-row", has_text="num_iterations").locator(".tabulator-cell[tabulator-field='value']").click()
                page.locator("#gbmParameterGrid input.gbm-parameter-input").wait_for(timeout=10_000)
                page.keyboard.press("Escape")
                gbm_top_before = page.locator(".gbm-tool").evaluate("node => node.getBoundingClientRect().top")
                page.locator("#gbmCreateSampleBtn").click()
                self.assertEqual(page.locator("#gbmCreateSampleBtn").text_content(), "Sample pending")
                self.assertEqual(page.locator("#gbmCreateSampleBtn").get_attribute("aria-pressed"), "true")
                self.assertFalse(page.locator("#gbmNotice").is_visible())
                self.assertTrue(page.locator("#status").evaluate("node => node.classList.contains('hidden')"))
                gbm_top_after_sample = page.locator(".gbm-tool").evaluate("node => node.getBoundingClientRect().top")
                self.assertLessEqual(abs(gbm_top_before - gbm_top_after_sample), 1)
                page.evaluate(
                    """
                    () => {
                      for (const checkbox of document.querySelectorAll("#gbmFeatureGrid .gbm-use-checkbox")) {
                        checkbox.checked = false;
                        checkbox.dispatchEvent(new Event("change", { bubbles: true }));
                      }
                    }
                    """
                )
                page.locator("#gbmTrainBtn").click()
                page.locator("#gbmNotice").get_by_text("Choose at least one usable GBM feature").wait_for(timeout=10_000)
                self.assertTrue(page.locator("#status").evaluate("node => node.classList.contains('hidden')"))
                gbm_top_after_error = page.locator(".gbm-tool").evaluate("node => node.getBoundingClientRect().top")
                self.assertLessEqual(abs(gbm_top_before - gbm_top_after_error), 1)
                self.assertTrue(page.locator(".sidebar-kpi-section").is_visible())
                self.assertTrue(page.locator("#actualNumerator").is_visible())
                self.assertTrue(page.locator("#denominator").is_visible())
                self.assertFalse(page.locator(".sidebar-filter-section").is_visible())
                self.assertFalse(page.locator("#modelToolGroupMeta").is_visible())
                self.assertFalse(page.locator("#modelToolFilter").is_visible())
                page.wait_for_function(
                    """
                    () => {
                      const target = document.querySelector("#gbmEvaluationChart");
                      const chart = target && window.echarts?.getInstanceByDom(target);
                      return chart?.getOption()?.title?.[0]?.text === "evaluation metric: gamma, test metric: 7.3022, best iteration: 3";
                    }
                    """,
                    timeout=10_000,
                )
                chart_options = page.evaluate(
                    """
                    () => {
                      const chart = window.echarts.getInstanceByDom(document.querySelector("#gbmEvaluationChart"));
                      const option = chart.getOption();
                      return {
                        title: option.title[0].text,
                        subtext: option.title[0].subtext || "",
                        titleFontSize: option.title[0].textStyle.fontSize,
                        legendOrient: option.legend[0].orient,
                        legendRight: option.legend[0].right,
                        gridTop: option.grid[0].top,
                        gridRight: option.grid[0].right,
                        gridContainLabel: option.grid[0].containLabel,
                        xType: option.xAxis[0].type,
                        xInterval: option.xAxis[0].interval,
                        xMax: option.xAxis[0].max,
                        yType: option.yAxis[0].type,
                        yScale: option.yAxis[0].scale,
                        seriesNames: option.series.map((series) => series.name),
                        showSymbol: option.series.every((series) => series.showSymbol),
                      };
                    }
                    """
                )
                self.assertEqual(chart_options["title"], "evaluation metric: gamma, test metric: 7.3022, best iteration: 3")
                self.assertEqual(chart_options["subtext"], "")
                self.assertEqual(chart_options["titleFontSize"], 12)
                self.assertEqual(chart_options["legendOrient"], "vertical")
                self.assertEqual(chart_options["legendRight"], 8)
                self.assertEqual(chart_options["gridTop"], 42)
                self.assertEqual(chart_options["gridRight"], 82)
                self.assertTrue(chart_options["gridContainLabel"])
                self.assertEqual(chart_options["xType"], "value")
                self.assertEqual(chart_options["xInterval"], 2)
                self.assertEqual(chart_options["xMax"], 6)
                self.assertEqual(chart_options["yType"], "value")
                self.assertTrue(chart_options["yScale"])
                self.assertEqual(chart_options["seriesNames"], ["train", "test"])
                self.assertTrue(chart_options["showSymbol"])
                page.get_by_text("Model navigator").click()
                page.locator("tr", has_text="Second smoke model").get_by_role("button", name="Activate").click()
                page.get_by_text("Features and parameters").click()
                page.wait_for_function(
                    """
                    () => [...document.querySelectorAll("#gbmParameterGrid .tabulator-row")]
                      .find((row) => row.textContent.includes("learning_rate"))
                      ?.querySelector(".tabulator-cell[tabulator-field='value']")
                      ?.textContent.trim() === "0.22"
                    """,
                    timeout=10_000,
                )
                self.assertEqual(
                    page.locator("#gbmParameterGrid .tabulator-row", has_text="num_iterations").locator(".tabulator-cell[tabulator-field='value']").text_content(),
                    "123",
                )
                feature_state = page.evaluate(
                    """
                    () => {
                      function rowState(name) {
                        const row = [...document.querySelectorAll("#gbmFeatureGrid .tabulator-row")]
                          .find((item) => item.textContent.includes(name));
                        return {
                          checked: Boolean(row?.querySelector(".gbm-use-checkbox")?.checked),
                          monotonicity: row?.querySelector(".tabulator-cell[tabulator-field='monotonicity']")?.textContent.trim() || "",
                          gain: row?.querySelector(".tabulator-cell[tabulator-field='gain']")?.textContent.trim() || "",
                        };
                      }
                      return { age: rowState("Age"), segment: rowState("Segment") };
                    }
                    """
                )
                self.assertFalse(feature_state["age"]["checked"])
                self.assertTrue(feature_state["segment"]["checked"])
                self.assertEqual(feature_state["age"]["monotonicity"], "")
                self.assertEqual(feature_state["segment"]["gain"], "6.000")
                layout = page.evaluate(
                    """
                    () => {
                        const visual = document.querySelector("#visualArea").getBoundingClientRect();
                        const tool = document.querySelector(".gbm-tool").getBoundingClientRect();
                        const grid = document.querySelector("#gbmFeatureGrid").getBoundingClientRect();
                        const right = document.querySelector(".gbm-right-panel").getBoundingClientRect();
                        const firstRow = document.querySelector("#gbmFeatureGrid .tabulator-row");
                        const normalRow = document.querySelector("#gbmFeatureGrid .tabulator-row:not(.gbm-feature-disabled):not(.gbm-feature-warning)");
                        const firstGain = document.querySelector("#gbmFeatureGrid .tabulator-cell[tabulator-field='gain']");
                        const tableHolder = document.querySelector("#gbmFeatureGrid .tabulator-tableholder");
                        const tab = document.querySelector(".gbm-tabs .tab");
                        const shap = document.querySelector("#gbmShapRows");
                        const sample = document.querySelector("#gbmCreateSampleBtn");
                        const train = document.querySelector("#gbmTrainBtn");
                        const parameterGrid = document.querySelector("#gbmParameterGrid");
                        const evaluationChart = document.querySelector("#gbmEvaluationChart");
                        const evaluationPanel = evaluationChart?.parentElement;
                        const featureCell = document.querySelector("#gbmFeatureGrid .tabulator-row .tabulator-cell");
                        const parameterCell = document.querySelector("#gbmParameterGrid .tabulator-row .tabulator-cell");
                        const featureHeader = document.querySelector("#gbmFeatureGrid .tabulator-col");
                        const parameterHeader = document.querySelector("#gbmParameterGrid .tabulator-col");
                        const featureHeaderContent = document.querySelector("#gbmFeatureGrid .tabulator-col .tabulator-col-content");
                        const parameterHeaderContent = document.querySelector("#gbmParameterGrid .tabulator-col .tabulator-col-content");
                        const featureNameCell = document.querySelector("#gbmFeatureGrid .tabulator-row .gbm-feature-name-cell");
                        const featureKind = document.querySelector("#gbmFeatureGrid .gbm-feature-kind");
                        const segmentKind = [...document.querySelectorAll("#gbmFeatureGrid .tabulator-row")]
                          .find((row) => row.textContent.includes("Segment"))
                          ?.querySelector(".gbm-feature-kind");
                        const headerTitles = [...document.querySelectorAll("#gbmFeatureGrid .tabulator-col-title")]
                          .map((node) => node.textContent.trim())
                          .filter(Boolean);
                        return {
                            visualWidth: visual.width,
                            toolWidth: tool.width,
                            gridWidth: grid.width,
                            rightWidth: right.width,
                            shapRadios: document.querySelectorAll("input[name='gbmShapRows']").length,
                            featureCheckboxes: document.querySelectorAll("#gbmFeatureGrid .gbm-use-checkbox").length,
                            disabledFeatureCheckboxes: document.querySelectorAll("#gbmFeatureGrid .gbm-feature-disabled .gbm-use-checkbox").length,
                            rowHeight: firstRow ? firstRow.getBoundingClientRect().height : 0,
                            gainAlign: firstGain ? getComputedStyle(firstGain).textAlign : "",
                            rowBackground: normalRow ? getComputedStyle(normalRow).backgroundColor : "",
                            holderBackground: tableHolder ? getComputedStyle(tableHolder).backgroundColor : "",
                            tabTop: tab ? Math.round(tab.getBoundingClientRect().top) : 0,
                            shapTop: shap ? Math.round(shap.getBoundingClientRect().top) : 0,
                            shapRight: shap ? Math.round(shap.getBoundingClientRect().right) : 0,
                            sampleLeft: sample ? Math.round(sample.getBoundingClientRect().left) : 0,
                            sampleTop: sample ? Math.round(sample.getBoundingClientRect().top) : 0,
                            trainTop: train ? Math.round(train.getBoundingClientRect().top) : 0,
                            parameterGridHeight: parameterGrid ? Math.round(parameterGrid.getBoundingClientRect().height) : 0,
                            evaluationChartHeight: evaluationChart ? Math.round(evaluationChart.getBoundingClientRect().height) : 0,
                            evaluationPanelHeight: evaluationPanel ? Math.round(evaluationPanel.getBoundingClientRect().height) : 0,
                            featureCellFontSize: featureCell ? getComputedStyle(featureCell).fontSize : "",
                            featureCellLineHeight: featureCell ? getComputedStyle(featureCell).lineHeight : "",
                            featureCellDisplay: featureCell ? getComputedStyle(featureCell).display : "",
                            featureCellAlignItems: featureCell ? getComputedStyle(featureCell).alignItems : "",
                            parameterCellFontSize: parameterCell ? getComputedStyle(parameterCell).fontSize : "",
                            parameterCellLineHeight: parameterCell ? getComputedStyle(parameterCell).lineHeight : "",
                            parameterCellDisplay: parameterCell ? getComputedStyle(parameterCell).display : "",
                            parameterCellAlignItems: parameterCell ? getComputedStyle(parameterCell).alignItems : "",
                            featureHeaderFontSize: featureHeader ? getComputedStyle(featureHeader).fontSize : "",
                            featureHeaderJustifyContent: featureHeader ? getComputedStyle(featureHeader).justifyContent : "",
                            featureHeaderContentDisplay: featureHeaderContent ? getComputedStyle(featureHeaderContent).display : "",
                            featureHeaderContentAlignItems: featureHeaderContent ? getComputedStyle(featureHeaderContent).alignItems : "",
                            parameterHeaderFontSize: parameterHeader ? getComputedStyle(parameterHeader).fontSize : "",
                            parameterHeaderJustifyContent: parameterHeader ? getComputedStyle(parameterHeader).justifyContent : "",
                            parameterHeaderContentDisplay: parameterHeaderContent ? getComputedStyle(parameterHeaderContent).display : "",
                            parameterHeaderContentAlignItems: parameterHeaderContent ? getComputedStyle(parameterHeaderContent).alignItems : "",
                            featureNameJustifyContent: featureNameCell ? getComputedStyle(featureNameCell).justifyContent : "",
                            featureKindFontSize: featureKind ? getComputedStyle(featureKind).fontSize : "",
                            segmentKindText: segmentKind ? segmentKind.textContent.trim() : "",
                            featureHeaders: headerTitles,
                        };
                    }
                    """
                )
                self.assertGreater(layout["toolWidth"], layout["visualWidth"] * 0.85)
                self.assertGreater(layout["gridWidth"], 320)
                self.assertGreater(layout["rightWidth"], 300)
                self.assertGreaterEqual(layout["shapRadios"], 3)
                self.assertGreater(layout["featureCheckboxes"], 0)
                self.assertEqual(layout["disabledFeatureCheckboxes"], 0)
                self.assertLess(layout["rowHeight"], 28)
                self.assertEqual(layout["gainAlign"], "center")
                self.assertEqual(layout["rowBackground"], layout["holderBackground"])
                self.assertLess(layout["shapRight"], layout["sampleLeft"])
                self.assertLessEqual(abs(layout["tabTop"] - layout["shapTop"]), 2)
                self.assertLessEqual(abs(layout["tabTop"] - layout["sampleTop"]), 2)
                self.assertLessEqual(abs(layout["tabTop"] - layout["trainTop"]), 2)
                self.assertGreaterEqual(layout["parameterGridHeight"], 260)
                self.assertGreaterEqual(layout["evaluationChartHeight"], 220)
                self.assertLess(layout["evaluationChartHeight"], layout["evaluationPanelHeight"])
                self.assertEqual(layout["featureCellFontSize"], "11px")
                self.assertEqual(layout["parameterCellFontSize"], "11px")
                self.assertEqual(layout["featureCellLineHeight"], layout["parameterCellLineHeight"])
                self.assertEqual(layout["featureCellDisplay"], "inline-flex")
                self.assertEqual(layout["parameterCellDisplay"], "inline-flex")
                self.assertEqual(layout["featureCellAlignItems"], "center")
                self.assertEqual(layout["parameterCellAlignItems"], "center")
                self.assertEqual(layout["featureHeaderFontSize"], "11px")
                self.assertEqual(layout["parameterHeaderFontSize"], "11px")
                self.assertEqual(layout["featureHeaderJustifyContent"], "center")
                self.assertEqual(layout["parameterHeaderJustifyContent"], "center")
                self.assertEqual(layout["featureHeaderContentDisplay"], "flex")
                self.assertEqual(layout["parameterHeaderContentDisplay"], "flex")
                self.assertEqual(layout["featureHeaderContentAlignItems"], "center")
                self.assertEqual(layout["parameterHeaderContentAlignItems"], "center")
                self.assertEqual(layout["featureNameJustifyContent"], "space-between")
                self.assertEqual(layout["featureKindFontSize"], "9px")
                self.assertEqual(layout["segmentKindText"], "categorical (3)")
                self.assertIn("Feature", layout["featureHeaders"])
                self.assertIn("Use", layout["featureHeaders"])
                self.assertIn("Monotonicity", layout["featureHeaders"])
                self.assertIn("Gain", layout["featureHeaders"])
                self.assertNotIn("Type", layout["featureHeaders"])
                self.assertEqual(page_errors, [])
            finally:
                browser.close()

    def exercise_stopped_overlay(self, base_url: str) -> None:
        assert sync_playwright is not None
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch()
            page = browser.new_page(viewport={"width": 1280, "height": 800})
            page_errors: list[str] = []
            page.on("pageerror", lambda error: page_errors.append(str(error)))
            try:
                page.goto(base_url, wait_until="domcontentloaded")
                page.locator("#datasetMeta").get_by_text("sample.csv").wait_for(timeout=10_000)
                page.locator("#stopAppBtn").click()
                page.locator(".stop-confirm-ok").click()
                page.locator(".shutdown-message").get_by_text("lucidum has stopped").wait_for(timeout=10_000)
                icon = page.locator(".shutdown-icon")

                self.assertEqual(icon.evaluate("node => node.tagName.toLowerCase()"), "img")
                icon_src = icon.get_attribute("src")
                self.assertIsNotNone(icon_src)
                assert icon_src is not None
                self.assertTrue(icon_src.startswith("data:image/"))
                self.assertTrue(icon.evaluate("node => node.complete && node.naturalWidth > 0"))
                self.assertEqual(page.locator(".shutdown-icon-fallback").count(), 0)
                self.assertEqual(page_errors, [])
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
