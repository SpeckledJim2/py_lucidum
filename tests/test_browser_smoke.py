from __future__ import annotations

import json
import importlib.util
import math
import os
import re
import socket
import sys
import threading
import time
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any
from unittest.mock import patch
from urllib.request import urlopen

import duckdb
import uvicorn

from py_lucidum import __version__
from py_lucidum.app import create_app
from py_lucidum.core import Dataset, quote_ident, sql_literal
from py_lucidum.tools.gbm.store import GbmModelStore
from py_lucidum.tools.glm.store import GlmModelStore
from py_lucidum.tools.glm.tabulation import build_tabulations
from py_lucidum.tools.glm.training import stop_persistent_glm_fit_worker, train_model


try:
    from playwright.sync_api import TimeoutError as PlaywrightTimeoutError, sync_playwright
except ImportError:  # pragma: no cover - exercised only without optional test deps.
    PlaywrightTimeoutError = TimeoutError
    sync_playwright = None


RUN_BROWSER_TESTS = os.environ.get("PY_LUCIDUM_RUN_BROWSER_TESTS") == "1"


def write_gbm_evaluation(store: GbmModelStore, model_id: str, evaluation: dict[str, dict[str, list[Any]]]) -> None:
    selects: list[str] = []
    for dataset_name, metrics in evaluation.items():
        for metric_name, values in metrics.items():
            for iteration, value in enumerate(values, start=1):
                value_sql = "NULL" if value is None else str(float(value))
                selects.append(
                    f"SELECT {sql_literal(str(dataset_name))} AS dataset, {sql_literal(str(metric_name))} AS metric, "
                    f"{iteration} AS iteration, {value_sql} AS value"
                )
    if not selects:
        return
    con = duckdb.connect(database=":memory:")
    try:
        con.execute(
            f"""
COPY (
  {" UNION ALL ".join(selects)}
) TO {sql_literal(str(store.artifact_path(model_id, "evaluation")))} (FORMAT PARQUET)
"""
        )
    finally:
        con.close()


def sql_scalar(value: Any) -> str:
    if value is None:
        return "NULL"
    if isinstance(value, bool):
        return "TRUE" if value else "FALSE"
    if isinstance(value, (int, float)):
        number = float(value)
        return str(value) if math.isfinite(number) else "NULL"
    return sql_literal(str(value))


def write_gbm_feature_config(store: GbmModelStore, model_id: str, rows: list[dict[str, Any]]) -> None:
    store.write_json(store.artifact_path(model_id, "features"), [str(row["name"]) for row in rows])
    columns = ["name", "kind", "include", "monotonicity", "monotonicity_value", "gain", "mean_abs_shap"]
    selects = [
        "SELECT " + ", ".join(f"{sql_scalar(row.get(column))} AS {quote_ident(column)}" for column in columns)
        for row in rows
    ]
    if not selects:
        return
    con = duckdb.connect(database=":memory:")
    try:
        con.execute(
            f"""
COPY (
  {" UNION ALL ".join(selects)}
) TO {sql_literal(str(store.artifact_path(model_id, "feature_config")))} (FORMAT PARQUET)
"""
        )
    finally:
        con.close()


class BrowserSmokeTests(unittest.TestCase):
    def click_sidebar_favourite_action(self, page: Any, selector: str) -> None:
        if page.locator("#favouritesCollapseBtn").get_attribute("aria-expanded") != "true":
            page.locator("#favouritesCollapseBtn").click()
            page.wait_for_function(
                '() => document.querySelector("#favouritesCollapseBtn")?.getAttribute("aria-expanded") === "true"',
                timeout=10_000,
            )
        page.locator(".sidebar-favourites-header-row").hover()
        page.wait_for_function(
            """
            (selector) => {
              const button = document.querySelector(selector);
              const controls = button?.closest(".sidebar-favourites-controls");
              const header = document.querySelector("#favouritesCollapseBtn");
              if (!controls) return false;
              const style = getComputedStyle(controls);
              return style.display !== "none"
                && style.pointerEvents !== "none"
                && getComputedStyle(button).color === getComputedStyle(header).color;
            }
            """,
            arg=selector,
            timeout=10_000,
        )
        page.locator(selector).click()

    def assert_tool_button_tooltip_right_of_icon(self, page: Any, selector: str, text: str) -> None:
        page.locator(selector).hover()
        page.wait_for_timeout(250)
        self.assertTrue(
            page.evaluate(
                """
                () => {
                  const tooltip = document.querySelector("#toolButtonTooltip");
                  return !tooltip || tooltip.hidden;
                }
                """
            )
        )
        page.wait_for_function(
            """
            ({ selector, text }) => {
              const button = document.querySelector(selector);
              const tooltip = document.querySelector("#toolButtonTooltip");
              if (!button || !tooltip || tooltip.hidden || tooltip.textContent.trim() !== text) return false;
              const iconRect = button.querySelector(".tool-icon")?.getBoundingClientRect();
              const tooltipRect = tooltip.getBoundingClientRect();
              const front = document.elementFromPoint(
                tooltipRect.left + tooltipRect.width / 2,
                tooltipRect.top + tooltipRect.height / 2,
              );
              return tooltipRect.width > 0
                && tooltipRect.height > 0
                && iconRect
                && tooltipRect.left >= iconRect.right
                && Math.abs((tooltipRect.top + tooltipRect.height / 2) - (iconRect.top + iconRect.height / 2)) <= 2
                && (front === tooltip || tooltip.contains(front));
            }
            """,
            arg={"selector": selector, "text": text},
            timeout=2_000,
        )
        self.assertIsNone(page.locator(selector).get_attribute("title"))
        page.mouse.move(640, 400)
        page.wait_for_function(
            '() => document.querySelector("#toolButtonTooltip")?.hidden !== false',
            timeout=2_000,
        )

    def assert_tool_button_tooltip_fronts_map(self, page: Any, selector: str, text: str) -> None:
        page.locator(selector).hover()
        page.wait_for_timeout(250)
        self.assertTrue(
            page.evaluate(
                """
                () => {
                  const tooltip = document.querySelector("#toolButtonTooltip");
                  return !tooltip || tooltip.hidden;
                }
                """
            )
        )
        page.wait_for_function(
            """
            ({ selector, text }) => {
              const button = document.querySelector(selector);
              const tooltip = document.querySelector("#toolButtonTooltip");
              const map = document.querySelector("#ukMap");
              if (!button || !tooltip || !map || tooltip.hidden || tooltip.textContent.trim() !== text) return false;
              const iconRect = button.querySelector(".tool-icon")?.getBoundingClientRect();
              const tooltipRect = tooltip.getBoundingClientRect();
              const mapRect = map.getBoundingClientRect();
              const x = Math.max(mapRect.left + 2, Math.min(tooltipRect.right - 2, tooltipRect.left + tooltipRect.width / 2));
              const y = tooltipRect.top + tooltipRect.height / 2;
              if (x <= tooltipRect.left || x >= tooltipRect.right || y <= mapRect.top || y >= mapRect.bottom) return false;
              const front = document.elementFromPoint(x, y);
              return tooltipRect.width > 0
                && tooltipRect.height > 0
                && iconRect
                && tooltipRect.left >= iconRect.right
                && Math.abs((tooltipRect.top + tooltipRect.height / 2) - (iconRect.top + iconRect.height / 2)) <= 2
                && (front === tooltip || tooltip.contains(front));
            }
            """,
            arg={"selector": selector, "text": text},
            timeout=2_000,
        )
        self.assertIsNone(page.locator(selector).get_attribute("title"))
        page.mouse.move(640, 400)
        page.wait_for_function(
            '() => document.querySelector("#toolButtonTooltip")?.hidden !== false',
            timeout=2_000,
        )

    @unittest.skipUnless(RUN_BROWSER_TESTS, "set PY_LUCIDUM_RUN_BROWSER_TESTS=1 to run browser smoke tests")
    @unittest.skipUnless(sync_playwright is not None, "playwright is not installed")
    def test_chart_and_map_tools_load_and_switch_without_extra_api_requests(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            data_path = Path(tmp_dir) / "sample.csv"
            data_path.write_text(
                "PostcodeArea,PostcodeSector,vehicle_age,price,value,PostcodeUnit,lat,long\n"
                "AB,AB10 1,1,100,10,AB10 1AA,57.1,-2.1\n"
                "AB,AB10 1,2,200,20,AB10 1AB,57.2,-2.2\n"
                "AL,AL1 1,3,300,30,AL1 1AA,51.8,-0.3\n"
                "AL,AL1 2,4,400,40,AL1 2AA,51.7,-0.2\n",
                encoding="utf-8",
            )
            base_url, server, thread = self.start_app(data_path, title_prefix="Lucidum Smoke Dataset")
            try:
                self.assert_static_asset(base_url, "/static/app.css", "text/css")
                self.assert_static_asset(base_url, "/static/app.js", "text/javascript")
                self.exercise_browser(base_url)
            finally:
                server.should_exit = True
                thread.join(timeout=5)

    @unittest.skipUnless(RUN_BROWSER_TESTS, "set PY_LUCIDUM_RUN_BROWSER_TESTS=1 to run browser smoke tests")
    @unittest.skipUnless(sync_playwright is not None, "playwright is not installed")
    def test_mobile_phone_viewport_keeps_default_tools_usable(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            data_path = Path(tmp_dir) / "sample.csv"
            data_path.write_text(
                "PostcodeArea,PostcodeSector,vehicle_age,price,value,PostcodeUnit,lat,long\n"
                "AB,AB10 1,1,100,10,AB10 1AA,57.1,-2.1\n"
                "AB,AB10 1,2,200,20,AB10 1AB,57.2,-2.2\n"
                "AL,AL1 1,3,300,30,AL1 1AA,51.8,-0.3\n"
                "AL,AL1 2,4,400,40,AL1 2AA,51.7,-0.2\n",
                encoding="utf-8",
            )
            base_url, server, thread = self.start_app(data_path, title_prefix="Lucidum Smoke Dataset")
            try:
                assert sync_playwright is not None
                with sync_playwright() as playwright:
                    browser = playwright.chromium.launch()
                    page_errors: list[str] = []

                    def new_mobile_page(width: int, height: int):
                        page = browser.new_page(viewport={"width": width, "height": height})
                        page.on("pageerror", lambda error: page_errors.append(str(error)))
                        page.goto(base_url, wait_until="domcontentloaded")
                        page.locator("#datasetMeta").get_by_text("Lucidum Smoke Dataset").wait_for(timeout=10_000)
                        return page

                    def wait_for_mobile_line_bar(page) -> dict[str, Any]:
                        page.wait_for_function(
                            """
                            () => {
                              const controls = document.querySelector("#chartSideControls");
                              const toolbar = document.querySelector("#lineBarToolbar");
                              const sideToggle = document.querySelector("#lineBarSideControlsToggleBtn");
                              const toolbarToggle = document.querySelector("#lineBarToolbarToggleBtn");
                              const workspace = document.querySelector(".workspace");
                              const chart = document.querySelector("#chart");
                              if (!controls || !toolbar || !sideToggle || !toolbarToggle || !workspace || !chart || chart.classList.contains("hidden")) return false;
                              const workspaceRect = workspace.getBoundingClientRect();
                              const chartRect = chart.getBoundingClientRect();
                              return document.body.classList.contains("sidebar-collapsed")
                                && document.body.scrollWidth <= document.body.clientWidth
                                && sideToggle.getAttribute("aria-expanded") === "false"
                                && toolbarToggle.getAttribute("aria-expanded") === "false"
                                && getComputedStyle(controls).display === "none"
                                && getComputedStyle(toolbar).display === "none"
                                && workspaceRect.width >= 280
                                && chartRect.width >= 260;
                            }
                            """,
                            timeout=10_000,
                        )
                        return page.evaluate(
                            """
                            () => {
                              const rectFor = (selector) => {
                                const rect = document.querySelector(selector).getBoundingClientRect();
                                return { left: rect.left, top: rect.top, right: rect.right, width: rect.width, height: rect.height };
                              };
                              return {
                                bodyScrollWidth: document.body.scrollWidth,
                                bodyClientWidth: document.body.clientWidth,
                                controlsDisplay: getComputedStyle(document.querySelector("#chartSideControls")).display,
                                toolbarDisplay: getComputedStyle(document.querySelector("#lineBarToolbar")).display,
                                sideExpanded: document.querySelector("#lineBarSideControlsToggleBtn").getAttribute("aria-expanded"),
                                toolbarExpanded: document.querySelector("#lineBarToolbarToggleBtn").getAttribute("aria-expanded"),
                                workspace: rectFor(".workspace"),
                                chart: rectFor("#chart"),
                              };
                            }
                            """
                        )

                    def assert_mobile_sidebar_fronts_map(page) -> None:
                        page.locator("#sidebarToggleBtn").click()
                        page.wait_for_function(
                            """
                            () => {
                              const sidebar = document.querySelector("#appSidebar")?.getBoundingClientRect();
                              const main = document.querySelector("main")?.getBoundingClientRect();
                              if (!sidebar || !main) return false;
                              return !document.body.classList.contains("sidebar-collapsed")
                                && sidebar.width > 200
                                && sidebar.right > main.left;
                            }
                            """,
                            timeout=10_000,
                        )
                        page.wait_for_function("() => window.L && document.querySelector('#ukMap .leaflet-control')")
                        front_points = page.evaluate(
                            """
                            () => {
                              const sidebarEl = document.querySelector("#appSidebar");
                              const sidebarRect = sidebarEl.getBoundingClientRect();
                              const mainRect = document.querySelector("main").getBoundingClientRect();
                              const x = Math.round(Math.min(
                                sidebarRect.right - 8,
                                Math.max(mainRect.left + 16, sidebarRect.left + Math.min(260, sidebarRect.width - 20)),
                              ));
                              return [
                                sidebarRect.top + 72,
                                sidebarRect.top + 220,
                                sidebarRect.bottom - 80,
                              ].map((rawY) => {
                                const y = Math.round(Math.max(sidebarRect.top + 8, Math.min(sidebarRect.bottom - 8, rawY)));
                                const element = document.elementFromPoint(x, y);
                                return {
                                  x,
                                  y,
                                  insideSidebar: Boolean(element && sidebarEl.contains(element)),
                                  target: element
                                    ? `${element.tagName.toLowerCase()}#${element.id || ""}.${String(element.className || "").replace(/\\s+/g, ".")}`
                                    : "",
                                };
                              });
                            }
                            """
                        )
                        self.assertTrue(all(point["insideSidebar"] for point in front_points), front_points)
                        page.locator("#sidebarToggleBtn").click()
                        page.wait_for_function("() => document.body.classList.contains('sidebar-collapsed')", timeout=10_000)

                    prepaint_page = browser.new_page(viewport={"width": 390, "height": 844})
                    prepaint_page.route(
                        "**/static/app.js",
                        lambda route: route.fulfill(status=200, content_type="text/javascript", body=""),
                    )
                    try:
                        prepaint_page.goto(base_url, wait_until="domcontentloaded")
                        self.assertTrue(prepaint_page.locator("body").evaluate("body => body.classList.contains('sidebar-collapsed')"))
                        self.assertEqual(prepaint_page.locator("#datasetMeta").inner_text(), "Loading dataset...")
                    finally:
                        prepaint_page.close()

                    page = new_mobile_page(390, 844)
                    try:
                        line_bar_rects = wait_for_mobile_line_bar(page)
                        self.assertLessEqual(line_bar_rects["bodyScrollWidth"], line_bar_rects["bodyClientWidth"])
                        self.assertGreaterEqual(line_bar_rects["chart"]["width"], 260)
                        self.assertGreaterEqual(line_bar_rects["workspace"]["width"], 280)
                        self.assertEqual(line_bar_rects["controlsDisplay"], "none")
                        self.assertEqual(line_bar_rects["toolbarDisplay"], "none")
                        self.assertEqual(line_bar_rects["sideExpanded"], "false")
                        self.assertEqual(line_bar_rects["toolbarExpanded"], "false")

                        page.locator("#sidebarToggleBtn").click()
                        page.wait_for_function(
                            """
                            () => {
                              const sidebar = document.querySelector("#appSidebar")?.getBoundingClientRect();
                              const main = document.querySelector("main")?.getBoundingClientRect();
                              if (!sidebar || !main) return false;
                              return !document.body.classList.contains("sidebar-collapsed")
                                && document.body.scrollWidth <= document.body.clientWidth
                                && sidebar.width > 200
                                && main.width >= 300
                                && sidebar.right > main.left;
                            }
                            """,
                            timeout=10_000,
                        )
                        page.locator("#sidebarToggleBtn").click()
                        page.wait_for_function("() => document.body.classList.contains('sidebar-collapsed')", timeout=10_000)

                        for tool_button, wrapper_selector in [
                            ("#datasetViewerTool", "#datasetViewerWrap"),
                            ("#profileTool", "#profileWrap"),
                            ("#histogramTool", "#histogramWrap"),
                            ("#ukMapTool", "#ukMap"),
                            ("#specsTool", "#specificationsWrap"),
                        ]:
                            page.locator(tool_button).click()
                            page.wait_for_function(
                                """
                                (selector) => {
                                  const wrapper = document.querySelector(selector);
                                  if (!wrapper || wrapper.classList.contains("hidden")) return false;
                                  const rect = wrapper.getBoundingClientRect();
                                  return document.body.scrollWidth <= document.body.clientWidth
                                    && rect.width >= 280
                                    && rect.height >= 260;
                                }
                                """,
                                arg=wrapper_selector,
                                timeout=10_000,
                            )
                            if tool_button == "#datasetViewerTool":
                                page.locator("#datasetViewerWrap:not(.hidden) #datasetViewerGrid .tabulator-row").first.wait_for(timeout=10_000)
                                dataset_viewer_mobile_toolbar = page.evaluate(
                                    """
                                    () => {
                                      const rectFor = (selector) => {
                                        const rect = document.querySelector(selector).getBoundingClientRect();
                                        return { bottom: rect.bottom, left: rect.left, top: rect.top, width: rect.width };
                                      };
                                      return {
                                        alphabetical: rectFor('label:has(#datasetViewerAlphabeticalColumns)'),
                                        search: rectFor(".dataset-viewer-search-row"),
                                        toolbar: rectFor(".dataset-viewer-toolbar"),
                                        transpose: rectFor('label:has(#datasetViewerTranspose)'),
                                      };
                                    }
                                    """
                                )
                                self.assertGreaterEqual(
                                    dataset_viewer_mobile_toolbar["search"]["width"],
                                    dataset_viewer_mobile_toolbar["toolbar"]["width"] - 2,
                                )
                                self.assertGreaterEqual(
                                    dataset_viewer_mobile_toolbar["transpose"]["top"],
                                    dataset_viewer_mobile_toolbar["search"]["bottom"] - 1,
                                )
                                self.assertGreaterEqual(
                                    dataset_viewer_mobile_toolbar["alphabetical"]["top"],
                                    dataset_viewer_mobile_toolbar["search"]["bottom"] - 1,
                                )
                            if tool_button == "#profileTool":
                                page.locator("#profileWrap:not(.hidden) .profile-table").wait_for(timeout=10_000)
                                profile_mobile_layout = page.evaluate(
                                    """
                                    () => {
                                      const content = document.querySelector("#profileWrap .profile-content").getBoundingClientRect();
                                      const summary = document.querySelector("#profileWrap .profile-summary-pane").getBoundingClientRect();
                                      const detail = document.querySelector("#profileDetailPane").getBoundingClientRect();
                                      const contentStyle = getComputedStyle(document.querySelector("#profileWrap .profile-content"));
                                      const detailStyle = getComputedStyle(document.querySelector("#profileDetailPane"));
                                      return {
                                        detailBottomBorder: detailStyle.borderBottomWidth,
                                        detailHeightRatio: detail.height / content.height,
                                        detailLeftBorder: detailStyle.borderLeftWidth,
                                        detailStartsAtTop: Math.abs(detail.top - content.top) <= 2,
                                        detailTop: detail.top,
                                        gridColumns: contentStyle.gridTemplateColumns,
                                        gridRows: contentStyle.gridTemplateRows,
                                        mobile: window.matchMedia("(max-width: 640px)").matches,
                                        summaryBottomAligned: Math.abs(summary.bottom - content.bottom) <= 2,
                                        summaryHeightRatio: summary.height / content.height,
                                        summaryTop: summary.top,
                                      };
                                    }
                                    """
                                )
                                self.assertTrue(profile_mobile_layout["mobile"])
                                self.assertTrue(profile_mobile_layout["detailStartsAtTop"], profile_mobile_layout)
                                self.assertTrue(profile_mobile_layout["summaryBottomAligned"], profile_mobile_layout)
                                self.assertLess(profile_mobile_layout["detailTop"], profile_mobile_layout["summaryTop"])
                                self.assertGreaterEqual(profile_mobile_layout["detailHeightRatio"], 0.38)
                                self.assertGreaterEqual(profile_mobile_layout["summaryHeightRatio"], 0.38)
                                self.assertEqual(profile_mobile_layout["detailLeftBorder"], "0px")
                                self.assertEqual(profile_mobile_layout["detailBottomBorder"], "1px")
                            if tool_button == "#ukMapTool":
                                page.locator("#mapFloatingControl:not(.hidden)").wait_for(timeout=10_000)
                                page.wait_for_function(
                                    """
                                    () => {
                                      const control = document.querySelector("#mapFloatingControl");
                                      const controlButton = document.querySelector("#mapControlReset");
                                      const legend = document.querySelector("#mapLegend");
                                      const legendButton = document.querySelector("#mapLegendToggle");
                                      const legendBody = document.querySelector("#mapLegendBody");
                                      if (!control || !controlButton || !legend || !legendButton || !legendBody) return false;
                                      return control.classList.contains("collapsed")
                                        && controlButton.getAttribute("aria-expanded") === "false"
                                        && !legend.classList.contains("hidden")
                                        && legend.classList.contains("collapsed")
                                        && legendButton.getAttribute("aria-expanded") === "false"
                                        && getComputedStyle(legendBody).display === "none";
                                    }
                                    """,
                                    timeout=20_000,
                                )
                                page.locator("#mapControlReset").click()
                                page.wait_for_function('() => !document.querySelector("#mapFloatingControl")?.classList.contains("collapsed")', timeout=10_000)
                                mobile_map_panel = page.evaluate(
                                    """
                                    () => {
                                      const columnCount = (selector) => {
                                        const columns = getComputedStyle(document.querySelector(selector)).gridTemplateColumns;
                                        return columns.split(" ").filter(Boolean).length;
                                      };
                                      const visibleControls = [...document.querySelectorAll(".map-slider-control")]
                                        .filter((control) => control.offsetParent !== null);
                                      const endpointLabels = visibleControls.flatMap((control) => [
                                        control.querySelector(".slider-scale b:first-child"),
                                        control.querySelector(".slider-scale b:last-child"),
                                      ]).filter(Boolean);
                                      return {
                                        width: document.querySelector("#mapFloatingControl").getBoundingClientRect().width,
                                        baseColumns: columnCount("#mapBaseLayerTiles"),
                                        levelColumns: columnCount("#mapLevelTiles"),
                                        paletteColumns: columnCount(".map-palette-buttons"),
                                        sliderEndpointsVisible: endpointLabels.length > 0 && endpointLabels.every((label) => {
                                          const rect = label.getBoundingClientRect();
                                          return getComputedStyle(label).display !== "none" && rect.width > 0 && rect.height > 0;
                                        }),
                                      };
                                    }
                                    """
                                )
                                self.assertAlmostEqual(mobile_map_panel["width"], 244, delta=2)
                                self.assertEqual(mobile_map_panel["baseColumns"], 3)
                                self.assertEqual(mobile_map_panel["levelColumns"], 3)
                                self.assertEqual(mobile_map_panel["paletteColumns"], 3)
                                self.assertTrue(mobile_map_panel["sliderEndpointsVisible"], mobile_map_panel)
                                page.locator("#mapControlReset").click()
                                page.wait_for_function('() => document.querySelector("#mapFloatingControl")?.classList.contains("collapsed")', timeout=10_000)
                                assert_mobile_sidebar_fronts_map(page)
                    finally:
                        page.close()

                    page = new_mobile_page(430, 932)
                    try:
                        line_bar_rects = wait_for_mobile_line_bar(page)
                        self.assertLessEqual(line_bar_rects["bodyScrollWidth"], line_bar_rects["bodyClientWidth"])
                        self.assertGreaterEqual(line_bar_rects["chart"]["width"], 260)
                    finally:
                        page.close()

                    browser.close()
                    self.assertEqual(page_errors, [])
            finally:
                server.should_exit = True
                thread.join(timeout=5)

    @unittest.skipUnless(RUN_BROWSER_TESTS, "set PY_LUCIDUM_RUN_BROWSER_TESTS=1 to run browser smoke tests")
    @unittest.skipUnless(sync_playwright is not None, "playwright is not installed")
    def test_app_text_is_not_drag_selectable_except_inputs(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            data_path = Path(tmp_dir) / "sample.csv"
            data_path.write_text(
                "PostcodeArea,PostcodeSector,vehicle_age,price,value,PostcodeUnit,lat,long\n"
                "AB,AB10 1,1,100,10,AB10 1AA,57.1,-2.1\n"
                "AB,AB10 1,2,200,20,AB10 1AB,57.2,-2.2\n"
                "AL,AL1 1,3,300,30,AL1 1AA,51.8,-0.3\n",
                encoding="utf-8",
            )
            base_url, server, thread = self.start_app(data_path, tools=["line_bar"])
            try:
                assert sync_playwright is not None
                with sync_playwright() as playwright:
                    browser = playwright.chromium.launch()
                    page = browser.new_page(viewport={"width": 1280, "height": 800})
                    page_errors: list[str] = []
                    page.on("pageerror", lambda error: page_errors.append(str(error)))
                    try:
                        page.goto(base_url, wait_until="domcontentloaded")
                        page.locator("#datasetMeta").get_by_text("sample.csv").wait_for(timeout=10_000)
                        self.assertEqual(page.locator(".dataset-meta-title").count(), 0)
                        self.assertFalse(page.locator("#datasetMeta").evaluate("node => node.classList.contains('dataset-meta-title-only')"))
                        self.assertIn("sample.csv", page.locator(".dataset-meta-details").text_content())
                        page.locator("#lineBarSideControlsToggleBtn").click()
                        page.wait_for_function(
                            """
                            () => document.querySelector("#lineBarSideControlsToggleBtn")?.getAttribute("aria-expanded") === "true"
                              && getComputedStyle(document.querySelector("#chartSideControls")).display !== "none"
                            """,
                            timeout=10_000,
                        )
                        self.assertFalse(page.locator("#toolSelectorSection").is_visible())
                        self.assertEqual(page.locator("#toolSelectorSection .tool-option:not(.hidden)").count(), 0)
                        self.assertTrue(page.locator(".sidebar-metric-section").is_visible())
                        self.assertEqual(page.locator("#favouritesCollapseBtn").get_attribute("aria-expanded"), "true")
                        page.locator("#favouritesCollapseBtn").click()
                        page.wait_for_function(
                            '() => document.querySelector("#favouritesCollapseBtn")?.getAttribute("aria-expanded") === "false"',
                            timeout=10_000,
                        )
                        self.assertTrue(page.locator(".sidebar-metric-section").is_visible())
                        sidebar_box = page.locator("#appSidebar").bounding_box()
                        favourites_box = page.locator(".sidebar-favourites-section").bounding_box()
                        metric_box = page.locator(".sidebar-metric-section").bounding_box()
                        self.assertIsNotNone(sidebar_box)
                        self.assertIsNotNone(favourites_box)
                        self.assertIsNotNone(metric_box)
                        assert sidebar_box is not None
                        assert favourites_box is not None
                        assert metric_box is not None
                        self.assertLessEqual(metric_box["y"] + metric_box["height"], favourites_box["y"] + 1)
                        page.locator("#sidebarToggleBtn").click()
                        page.wait_for_function(
                            """
                            () => document.body.classList.contains("single-tool-mode")
                              && document.body.classList.contains("sidebar-collapsed")
                              && document.querySelector("#appSidebar")
                              && getComputedStyle(document.querySelector("#appSidebar")).display === "none"
                            """,
                            timeout=10_000,
                        )
                        self.assertFalse(page.locator("#sidebarResizer").is_visible())
                        page.locator("#sidebarToggleBtn").click()
                        page.locator(".sidebar-metric-section").wait_for(timeout=10_000)

                        def drag_text(selector: str) -> str:
                            box = page.locator(selector).bounding_box()
                            self.assertIsNotNone(box)
                            assert box is not None
                            y = box["y"] + (box["height"] / 2)
                            start_x = box["x"] + 2
                            end_x = box["x"] + max(4, min(box["width"] - 2, 180))
                            page.evaluate("() => window.getSelection()?.removeAllRanges()")
                            page.mouse.move(start_x, y)
                            page.mouse.down()
                            page.mouse.move(end_x, y, steps=8)
                            page.mouse.up()
                            page.wait_for_timeout(50)
                            return page.evaluate("() => window.getSelection()?.toString() || ''")

                        self.assertEqual(drag_text("#datasetMeta"), "")
                        self.assertEqual(drag_text("#actualMetricTitle"), "")

                        page.locator("#featureSearch").fill("vehicle_age")
                        page.locator("#featureSearch").focus()
                        shortcut = "Meta+A" if sys.platform == "darwin" else "Control+A"
                        page.keyboard.press(shortcut)
                        input_selection = page.evaluate(
                            """
                            () => {
                              const input = document.querySelector("#featureSearch");
                              return {
                                start: input?.selectionStart ?? -1,
                                end: input?.selectionEnd ?? -1,
                                value: input?.value || "",
                                userSelect: getComputedStyle(input).userSelect,
                              };
                            }
                            """
                        )
                        self.assertEqual(input_selection["start"], 0)
                        self.assertEqual(input_selection["end"], len(input_selection["value"]))
                        self.assertEqual(input_selection["userSelect"], "text")
                        self.assertEqual(page_errors, [])
                    finally:
                        browser.close()
            finally:
                server.should_exit = True
                thread.join(timeout=5)

    @unittest.skipUnless(RUN_BROWSER_TESTS, "set PY_LUCIDUM_RUN_BROWSER_TESTS=1 to run browser smoke tests")
    @unittest.skipUnless(sync_playwright is not None, "playwright is not installed")
    def test_hidden_cached_visual_tools_refresh_theme_without_api_requests(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            data_path = Path(tmp_dir) / "sample.csv"
            data_path.write_text(
                "PostcodeArea,PostcodeSector,vehicle_age,price,value,PostcodeUnit,lat,long\n"
                "AB,AB10 1,1,100,10,AB10 1AA,57.1,-2.1\n"
                "AB,AB10 1,2,200,20,AB10 1AB,57.2,-2.2\n"
                "AL,AL1 1,3,300,30,AL1 1AA,51.8,-0.3\n"
                "AL,AL1 2,4,400,40,AL1 2AA,51.7,-0.2\n",
                encoding="utf-8",
            )
            base_url, server, thread = self.start_app(
                data_path,
                defaults={
                    "x": "vehicle_age",
                    "actual": "price",
                    "denominator": "value",
                },
                tools=["line_bar", "uk_map"],
            )
            try:
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
                        if request.url.endswith("/api/chart"):
                            chart_requests += 1
                        elif request.url.endswith("/api/uk-map/summary"):
                            map_requests += 1

                    def wait_for_line_bar_text_theme() -> None:
                        page.wait_for_function(
                            """
                            () => {
                              const chart = window.echarts?.getInstanceByDom(document.querySelector("#chart"));
                              const option = chart?.getOption?.();
                              const xAxis = Array.isArray(option?.xAxis) ? option.xAxis[0] : option?.xAxis;
                              const actual = xAxis?.axisLabel?.color || "";
                              const expected = getComputedStyle(document.body).getPropertyValue("--text").trim();
                              return Boolean(actual && expected && actual === expected);
                            }
                            """,
                            timeout=10_000,
                        )

                    page.on("request", count_request)
                    page.goto(base_url, wait_until="domcontentloaded")
                    page.locator("#datasetMeta").get_by_text("sample.csv").wait_for(timeout=10_000)
                    page.locator("#lineBarTool.active").wait_for(timeout=10_000)
                    page.locator("#chart:not(.hidden)").wait_for(timeout=10_000)
                    wait_for_line_bar_text_theme()
                    self.assertGreaterEqual(chart_requests, 1)

                    page.locator("#ukMapTool").click()
                    page.locator("#ukMap:not(.hidden)").wait_for(timeout=10_000)
                    page.wait_for_function("() => document.querySelector('#ukMap')?.classList.contains('map-bg-light')", timeout=10_000)
                    page.wait_for_function(
                        """
                        () => (document.querySelector("#mapGroupMeta")?.textContent || "").includes("matched")
                        """,
                        timeout=10_000,
                    )
                    self.assertGreaterEqual(map_requests, 1)
                    page.locator("#sidebarToggleBtn").click()
                    page.wait_for_function("() => document.body.classList.contains('sidebar-collapsed')", timeout=10_000)
                    self.assert_tool_button_tooltip_fronts_map(page, "#ukMapTool", "UK mapping")
                    chart_requests_after_initial_render = chart_requests
                    map_requests_after_initial_render = map_requests

                    page.locator("#themeBtn").click()
                    page.wait_for_function("() => document.body.classList.contains('dark')", timeout=10_000)
                    page.wait_for_function("() => document.querySelector('#ukMap')?.classList.contains('map-bg-dark')", timeout=10_000)
                    page.locator("#lineBarTool").click()
                    page.locator("#chart:not(.hidden)").wait_for(timeout=10_000)
                    wait_for_line_bar_text_theme()
                    page.wait_for_timeout(100)
                    self.assertEqual(chart_requests, chart_requests_after_initial_render)
                    self.assertEqual(map_requests, map_requests_after_initial_render)

                    page.locator("#themeBtn").click()
                    page.wait_for_function("() => !document.body.classList.contains('dark')", timeout=10_000)
                    wait_for_line_bar_text_theme()
                    page.locator("#ukMapTool").click()
                    page.locator("#ukMap:not(.hidden)").wait_for(timeout=10_000)
                    page.wait_for_function("() => document.querySelector('#ukMap')?.classList.contains('map-bg-light')", timeout=10_000)
                    page.wait_for_timeout(100)
                    self.assertEqual(chart_requests, chart_requests_after_initial_render)
                    self.assertEqual(map_requests, map_requests_after_initial_render)
                    self.assertEqual(page_errors, [])
                    browser.close()
            finally:
                server.should_exit = True
                thread.join(timeout=5)

    @unittest.skipUnless(RUN_BROWSER_TESTS, "set PY_LUCIDUM_RUN_BROWSER_TESTS=1 to run browser smoke tests")
    @unittest.skipUnless(sync_playwright is not None, "playwright is not installed")
    def test_column_profile_preserves_table_scroll_on_filter_change(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            data_path = Path(tmp_dir) / "many_profile_columns.csv"
            columns = ["vehicle_age", "price", "value"] + [f"feature_{index:03d}" for index in range(1, 91)]
            rows = [",".join(columns)]
            for row_index in range(1, 10):
                values = [str(row_index), str(100 + row_index), str(10 + row_index)]
                values.extend(str((row_index * column_index) % 101) for column_index in range(1, 91))
                rows.append(",".join(values))
            data_path.write_text("\n".join(rows) + "\n", encoding="utf-8")
            base_url, server, thread = self.start_app(data_path, tools=["column_profile"])
            try:
                assert sync_playwright is not None
                with sync_playwright() as playwright:
                    browser = playwright.chromium.launch()
                    page = browser.new_page(viewport={"width": 1280, "height": 800})
                    page_errors: list[str] = []
                    page.on("pageerror", lambda error: page_errors.append(str(error)))
                    try:
                        page.goto(base_url, wait_until="domcontentloaded")
                        page.locator("#datasetMeta").get_by_text("many_profile_columns.csv").wait_for(timeout=10_000)
                        page.locator("#profileWrap:not(.hidden) .profile-table").wait_for(timeout=10_000)
                        profile_surface = page.evaluate(
                            """
                            () => {
                              const main = document.querySelector("main");
                              const visual = document.querySelector("#visualArea");
                              const workspace = document.querySelector(".workspace");
                              const panel = getComputedStyle(document.documentElement).getPropertyValue("--panel").trim();
                              const probe = document.createElement("div");
                              probe.style.background = panel;
                              probe.style.position = "absolute";
                              probe.style.visibility = "hidden";
                              document.body.appendChild(probe);
                              const panelBackground = getComputedStyle(probe).backgroundColor;
                              probe.remove();
                              return {
                                mainBackground: getComputedStyle(main).backgroundColor,
                                panelBackground,
                                profileMode: visual.classList.contains("profile-mode"),
                                workspaceBackground: getComputedStyle(workspace).backgroundColor,
                              };
                            }
                            """
                        )
                        self.assertTrue(profile_surface["profileMode"])
                        self.assertEqual(profile_surface["mainBackground"], profile_surface["panelBackground"])
                        self.assertEqual(profile_surface["workspaceBackground"], profile_surface["panelBackground"])
                        page.locator('#profileWrap [data-profile-sort="missing"]').click()
                        page.locator('#profileWrap th[aria-sort="ascending"] [data-profile-sort="missing"]').wait_for(timeout=10_000)
                        before_scroll = page.evaluate(
                            """
                            () => {
                              const scroll = document.querySelector("#profileWrap .profile-table-scroll");
                              scroll.scrollTop = 720;
                              return scroll.scrollTop;
                            }
                            """
                        )
                        self.assertGreater(before_scroll, 0)
                        with page.expect_response(lambda response: response.url.endswith("/api/column-profile/summary") and response.status == 200, timeout=10_000):
                            page.evaluate(
                                """
                                () => {
                                  document.querySelector("#filterInput").value = "vehicle_age >= 3";
                                  document.querySelector("#filterApplyBtn").click();
                                }
                                """
                            )
                        page.wait_for_function(
                            """
                            expectedScroll => {
                              const scroll = document.querySelector("#profileWrap .profile-table-scroll");
                              const activeSort = document.querySelector('#profileWrap th[aria-sort="ascending"] [data-profile-sort="missing"]');
                              return Boolean(scroll && activeSort && scroll.scrollTop >= Math.max(1, expectedScroll - 2));
                            }
                            """,
                            arg=before_scroll,
                            timeout=10_000,
                        )
                        self.assertEqual(page_errors, [])
                    finally:
                        browser.close()
            finally:
                server.should_exit = True
                thread.join(timeout=5)

    @unittest.skipUnless(RUN_BROWSER_TESTS, "set PY_LUCIDUM_RUN_BROWSER_TESTS=1 to run browser smoke tests")
    @unittest.skipUnless(sync_playwright is not None, "playwright is not installed")
    def test_glm_ace_editor_uses_current_theme_when_opened_from_cache(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            data_path = Path(tmp_dir) / "sample.csv"
            data_path.write_text(
                "vehicle_age,price,value\n"
                "1,100,10\n"
                "2,200,20\n"
                "3,300,30\n",
                encoding="utf-8",
            )
            base_url, server, thread = self.start_app(
                data_path,
                defaults={
                    "x": "vehicle_age",
                    "actual": "price",
                    "denominator": "value",
                },
                tools=["line_bar", "glm"],
            )
            try:
                assert sync_playwright is not None
                with sync_playwright() as playwright:
                    browser = playwright.chromium.launch()
                    page = browser.new_page(viewport={"width": 1280, "height": 800})
                    page_errors: list[str] = []
                    glm_config_requests = 0
                    page.on("pageerror", lambda error: page_errors.append(str(error)))

                    def count_request(request: object) -> None:
                        nonlocal glm_config_requests
                        if request.url.endswith("/api/glm/config"):
                            glm_config_requests += 1

                    def wait_for_ace_theme(theme: str, dark_gutter: bool) -> None:
                        page.wait_for_function(
                            """
                            ({ theme, darkGutter }) => {
                              const editorNode = document.querySelector("#glmFormulaEditor");
                              const editor = editorNode?.env?.editor || null;
                              const gutter = editorNode?.querySelector(".ace_gutter");
                              if (!editor || editor.getTheme() !== theme || !gutter) return false;
                              const match = getComputedStyle(gutter).backgroundColor.match(/\\d+(?:\\.\\d+)?/g) || [];
                              const rgb = match.slice(0, 3).map(Number);
                              if (rgb.length < 3 || rgb.some((value) => !Number.isFinite(value))) return false;
                              const average = (rgb[0] + rgb[1] + rgb[2]) / 3;
                              return darkGutter ? average < 90 : average > 150;
                            }
                            """,
                            arg={"theme": theme, "darkGutter": dark_gutter},
                            timeout=10_000,
                        )

                    page.on("request", count_request)
                    page.goto(base_url, wait_until="domcontentloaded")
                    page.locator("#datasetMeta").get_by_text("sample.csv").wait_for(timeout=10_000)
                    page.locator("#lineBarTool.active").wait_for(timeout=10_000)

                    page.locator("#themeBtn").click()
                    page.wait_for_function("() => document.body.classList.contains('dark')", timeout=10_000)
                    page.locator("#glmTool").click()
                    page.locator("#modelToolWrap:not(.hidden) #glmFormulaEditor").wait_for(timeout=10_000)
                    wait_for_ace_theme("ace/theme/monokai", True)
                    self.assertGreaterEqual(glm_config_requests, 1)
                    glm_requests_after_first_open = glm_config_requests

                    page.locator("#lineBarTool").click()
                    page.locator("#chart:not(.hidden)").wait_for(timeout=10_000)
                    page.locator("#themeBtn").click()
                    page.wait_for_function("() => !document.body.classList.contains('dark')", timeout=10_000)
                    page.locator("#glmTool").click()
                    page.locator("#modelToolWrap:not(.hidden) #glmFormulaEditor").wait_for(timeout=10_000)
                    wait_for_ace_theme("ace/theme/textmate", False)
                    page.wait_for_timeout(100)
                    self.assertEqual(glm_config_requests, glm_requests_after_first_open)
                    self.assertEqual(page_errors, [])
                    browser.close()
            finally:
                server.should_exit = True
                thread.join(timeout=5)

    @unittest.skipUnless(RUN_BROWSER_TESTS, "set PY_LUCIDUM_RUN_BROWSER_TESTS=1 to run browser smoke tests")
    @unittest.skipUnless(sync_playwright is not None, "playwright is not installed")
    def test_dataset_viewer_is_not_loaded_when_tool_is_disabled(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            data_path = Path(tmp_dir) / "sample.csv"
            data_path.write_text(
                "PostcodeArea,PostcodeSector,vehicle_age,price,value,PostcodeUnit,lat,long\n"
                "AB,AB10 1,1,100,10,AB10 1AA,57.1,-2.1\n"
                "AL,AL1 1,3,300,30,AL1 1AA,51.8,-0.3\n",
                encoding="utf-8",
            )
            base_url, server, thread = self.start_app(data_path, tools=["line-bar"])
            try:
                self.exercise_dataset_viewer_disabled(base_url)
            finally:
                server.should_exit = True
                thread.join(timeout=5)

    @unittest.skipUnless(RUN_BROWSER_TESTS, "set PY_LUCIDUM_RUN_BROWSER_TESTS=1 to run browser smoke tests")
    @unittest.skipUnless(sync_playwright is not None, "playwright is not installed")
    def test_dataset_viewer_sidebar_resize_waits_to_redraw_until_release(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            data_path = Path(tmp_dir) / "sample.csv"
            data_path.write_text(
                "PostcodeArea,PostcodeSector,vehicle_age,price,value,PostcodeUnit,lat,long,feature_001,feature_002,feature_003\n"
                "AB,AB10 1,1,100,10,AB10 1AA,57.1,-2.1,11,12,13\n"
                "AB,AB10 1,2,200,20,AB10 1AB,57.2,-2.2,21,22,23\n"
                "AL,AL1 1,3,300,30,AL1 1AA,51.8,-0.3,31,32,33\n",
                encoding="utf-8",
            )
            base_url, server, thread = self.start_app(data_path)
            try:
                self.exercise_dataset_viewer_sidebar_resize(base_url)
            finally:
                server.should_exit = True
                thread.join(timeout=5)

    @unittest.skipUnless(RUN_BROWSER_TESTS, "set PY_LUCIDUM_RUN_BROWSER_TESTS=1 to run browser smoke tests")
    @unittest.skipUnless(sync_playwright is not None, "playwright is not installed")
    def test_line_bar_picker_click_preserves_scroll_position(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            data_path = Path(tmp_dir) / "many_columns.csv"
            columns = ["actual", "expected"] + [f"feature_{index:03d}" for index in range(1, 161)]
            rows = [",".join(columns)]
            for row_index in range(1, 8):
                values = [str(100 + row_index), str(90 + row_index)]
                values.extend(str((row_index * column_index) % 97) for column_index in range(1, 161))
                rows.append(",".join(values))
            data_path.write_text("\n".join(rows) + "\n", encoding="utf-8")
            base_url, server, thread = self.start_app(
                data_path,
                defaults={
                    "x": "feature_001",
                    "actual": "actual",
                    "expected": "",
                    "denominator": "__none__",
                },
                tools=["line_bar"],
            )
            try:
                assert sync_playwright is not None
                with sync_playwright() as playwright:
                    browser = playwright.chromium.launch()
                    page = browser.new_page(viewport={"width": 1280, "height": 800})
                    page_errors: list[str] = []
                    page.on("pageerror", lambda error: page_errors.append(str(error)))
                    page.add_init_script(
                        """
                        localStorage.setItem("py_lucidum_chart_feature_controls_height", "160");
                        """
                    )
                    page.goto(base_url, wait_until="domcontentloaded")
                    page.locator("#datasetMeta").get_by_text("many_columns.csv").wait_for(timeout=10_000)
                    page.locator("#chart:not(.hidden)").wait_for(timeout=10_000)
                    page.locator("#lineBarSideControlsToggleBtn").click()
                    page.wait_for_function(
                        """
                        () => document.querySelector("#lineBarSideControlsToggleBtn")?.getAttribute("aria-expanded") === "true"
                          && getComputedStyle(document.querySelector("#chartSideControls")).display !== "none"
                        """,
                        timeout=10_000,
                    )
                    page.locator("#featureList .line-bar-scroll-region").wait_for(timeout=10_000)
                    page.locator("#expectedList .line-bar-scroll-region").wait_for(state="attached", timeout=10_000)
                    page.wait_for_function(
                        """
                        () => document.querySelector("#chartSideControls")?.classList.contains("chart-expected-collapsed")
                        """,
                        timeout=10_000,
                    )

                    initial_split_state = page.evaluate(
                        """
                        () => {
                          const controls = document.querySelector("#chartSideControls");
                          const feature = document.querySelector(".feature-section");
                          const expected = document.querySelector("#expectedSideSection");
                          const row = document.querySelector("#chartControlHeightRow");
                          const resizer = document.querySelector("#chartControlHeightResizer");
                          const toggle = document.querySelector("#chartExpectedToggle");
                          const icon = document.querySelector(".chart-expected-toggle-icon");
                          const controlsRect = controls?.getBoundingClientRect();
                          const resizerRect = resizer?.getBoundingClientRect();
                          return {
                            collapsed: controls?.classList.contains("chart-expected-collapsed") || false,
                            controlsHeight: controlsRect?.height || 0,
                            featureHeight: feature?.getBoundingClientRect().height || 0,
                            expectedHeight: expected?.getBoundingClientRect().height || 0,
                            expectedHidden: expected?.hidden || false,
                            expectedInert: expected?.hasAttribute("inert") || false,
                            expectedAriaHidden: expected?.getAttribute("aria-hidden") || "",
                            savedHeight: localStorage.getItem("py_lucidum_chart_feature_controls_height") || "",
                            toggleExpanded: toggle?.getAttribute("aria-expanded") || "",
                            toggleLabel: toggle?.getAttribute("aria-label") || "",
                            iconTransform: icon ? getComputedStyle(icon).transform : "",
                            rowWidth: row?.getBoundingClientRect().width || 0,
                            resizerWidth: resizerRect?.width || 0,
                            toggleWidth: toggle?.getBoundingClientRect().width || 0,
                            resizerBottomDelta: controlsRect && resizerRect
                              ? Math.abs(controlsRect.bottom - resizerRect.bottom)
                              : 999,
                          };
                        }
                        """
                    )
                    self.assertTrue(initial_split_state["collapsed"])
                    self.assertGreater(initial_split_state["controlsHeight"], 300)
                    self.assertGreater(initial_split_state["featureHeight"], 96)
                    self.assertLessEqual(initial_split_state["expectedHeight"], 1)
                    self.assertTrue(initial_split_state["expectedHidden"])
                    self.assertTrue(initial_split_state["expectedInert"])
                    self.assertEqual(initial_split_state["expectedAriaHidden"], "true")
                    self.assertEqual(initial_split_state["savedHeight"], "160")
                    self.assertEqual(initial_split_state["toggleExpanded"], "false")
                    self.assertEqual(initial_split_state["toggleLabel"], "Show Expected controls")
                    self.assertLessEqual(initial_split_state["resizerBottomDelta"], 1)
                    self.assertGreater(initial_split_state["rowWidth"], initial_split_state["resizerWidth"])
                    self.assertGreater(initial_split_state["toggleWidth"], 0)
                    self.assertGreaterEqual(
                        initial_split_state["featureHeight"],
                        initial_split_state["controlsHeight"] - 36,
                    )
                    page.locator("#featureSearch").focus()
                    feature_focus_state = page.evaluate(
                        """
                        () => {
                          const style = getComputedStyle(document.querySelector("#featureSearch"));
                          return {
                            boxShadow: style.boxShadow,
                            outlineStyle: style.outlineStyle,
                          };
                        }
                        """
                    )
                    self.assertEqual(feature_focus_state["boxShadow"], "none")
                    self.assertEqual(feature_focus_state["outlineStyle"], "none")

                    page.locator("#filterFooterToggleBtn").click()
                    page.wait_for_function(
                        """
                        () => {
                          const footer = document.querySelector("#filterFooter");
                          const controls = document.querySelector("#chartSideControls");
                          const resizer = document.querySelector("#chartControlHeightResizer");
                          const filterInput = document.querySelector("#filterInput");
                          const timing = document.querySelector("#actionTimingMonitor");
                          const footerRect = footer?.getBoundingClientRect();
                          const controlsRect = controls?.getBoundingClientRect();
                          const resizerRect = resizer?.getBoundingClientRect();
                          const filterRect = filterInput?.getBoundingClientRect();
                          const timingRect = timing?.getBoundingClientRect();
                          return footer?.getAttribute("aria-hidden") === "false"
                            && footerRect
                            && controlsRect
                            && resizerRect
                            && filterRect
                            && timingRect
                            && resizerRect.bottom <= footerRect.top + 1
                            && Math.abs(controlsRect.bottom - resizerRect.bottom) <= 1
                            && Math.abs(timingRect.left - filterRect.right - 10) <= 2;
                        }
                        """,
                        timeout=10_000,
                    )
                    footer_open_split_state = page.evaluate(
                        """
                        () => {
                          const footer = document.querySelector("#filterFooter");
                          const controls = document.querySelector("#chartSideControls");
                          const feature = document.querySelector(".feature-section");
                          const resizer = document.querySelector("#chartControlHeightResizer");
                          const filterInput = document.querySelector("#filterInput");
                          const timing = document.querySelector("#actionTimingMonitor");
                          const footerRect = footer?.getBoundingClientRect();
                          const controlsRect = controls?.getBoundingClientRect();
                          const resizerRect = resizer?.getBoundingClientRect();
                          const filterRect = filterInput?.getBoundingClientRect();
                          const timingRect = timing?.getBoundingClientRect();
                          return {
                            collapsed: controls?.classList.contains("chart-expected-collapsed") || false,
                            footerVisible: footer?.getAttribute("aria-hidden") === "false",
                            controlsHeight: controlsRect?.height || 0,
                            featureHeight: feature?.getBoundingClientRect().height || 0,
                            filterTimingGap: filterRect && timingRect ? timingRect.left - filterRect.right : -999,
                            filterWidth: filterRect?.width || 0,
                            resizerBottomDelta: controlsRect && resizerRect
                              ? Math.abs(controlsRect.bottom - resizerRect.bottom)
                              : 999,
                            resizerFooterGap: footerRect && resizerRect
                              ? footerRect.top - resizerRect.bottom
                              : -999,
                          };
                        }
                        """
                    )
                    self.assertTrue(footer_open_split_state["collapsed"])
                    self.assertTrue(footer_open_split_state["footerVisible"])
                    self.assertLessEqual(footer_open_split_state["resizerBottomDelta"], 1)
                    self.assertGreaterEqual(footer_open_split_state["resizerFooterGap"], -1)
                    self.assertGreater(footer_open_split_state["filterWidth"], 500)
                    self.assertAlmostEqual(footer_open_split_state["filterTimingGap"], 10, delta=2)
                    self.assertGreaterEqual(
                        footer_open_split_state["featureHeight"],
                        footer_open_split_state["controlsHeight"] - 36,
                    )
                    page.locator("#filterFooterToggleBtn").click()
                    page.wait_for_function(
                        """
                        () => document.querySelector("#filterFooter")?.getAttribute("aria-hidden") === "true"
                        """,
                        timeout=10_000,
                    )

                    page.locator("#chartExpectedToggle").click()
                    page.wait_for_function(
                        """
                        () => {
                          const controls = document.querySelector("#chartSideControls");
                          const expected = document.querySelector("#expectedSideSection");
                          return controls
                            && expected
                            && !controls.classList.contains("chart-expected-collapsed")
                            && !expected.hidden
                            && !expected.hasAttribute("inert");
                        }
                        """,
                        timeout=10_000,
                    )
                    expanded_split_state = page.evaluate(
                        """
                        () => {
                          const controls = document.querySelector("#chartSideControls");
                          const feature = document.querySelector(".feature-section");
                          const expected = document.querySelector("#expectedSideSection");
                          const toggle = document.querySelector("#chartExpectedToggle");
                          const icon = document.querySelector(".chart-expected-toggle-icon");
                          return {
                            collapsed: controls?.classList.contains("chart-expected-collapsed") || false,
                            controlsHeight: controls?.getBoundingClientRect().height || 0,
                            featureHeight: feature?.getBoundingClientRect().height || 0,
                            expectedHeight: expected?.getBoundingClientRect().height || 0,
                            expectedHidden: expected?.hidden || false,
                            expectedInert: expected?.hasAttribute("inert") || false,
                            expectedAriaHidden: expected?.getAttribute("aria-hidden") || "",
                            savedHeight: localStorage.getItem("py_lucidum_chart_feature_controls_height") || "",
                            toggleExpanded: toggle?.getAttribute("aria-expanded") || "",
                            toggleLabel: toggle?.getAttribute("aria-label") || "",
                            iconTransform: icon ? getComputedStyle(icon).transform : "",
                          };
                        }
                        """
                    )
                    self.assertFalse(expanded_split_state["collapsed"])
                    self.assertFalse(expanded_split_state["expectedHidden"])
                    self.assertFalse(expanded_split_state["expectedInert"])
                    self.assertEqual(expanded_split_state["expectedAriaHidden"], "")
                    self.assertEqual(expanded_split_state["savedHeight"], "160")
                    self.assertEqual(expanded_split_state["toggleExpanded"], "true")
                    self.assertEqual(expanded_split_state["toggleLabel"], "Hide Expected controls")
                    self.assertNotEqual(expanded_split_state["iconTransform"], initial_split_state["iconTransform"])
                    self.assertGreater(expanded_split_state["expectedHeight"], 96)
                    page.locator("#expectedSearch").focus()
                    expected_focus_state = page.evaluate(
                        """
                        () => {
                          const style = getComputedStyle(document.querySelector("#expectedSearch"));
                          return {
                            boxShadow: style.boxShadow,
                            outlineStyle: style.outlineStyle,
                          };
                        }
                        """
                    )
                    self.assertEqual(expected_focus_state["boxShadow"], "none")
                    self.assertEqual(expected_focus_state["outlineStyle"], "none")

                    page.locator("#chartExpectedToggle").click()
                    page.wait_for_function(
                        """
                        () => document.querySelector("#chartSideControls")?.classList.contains("chart-expected-collapsed")
                          && document.querySelector("#chartExpectedToggle")?.getAttribute("aria-expanded") === "false"
                        """,
                        timeout=10_000,
                    )
                    toggle_collapsed_state = page.evaluate(
                        """
                        () => {
                          const expected = document.querySelector("#expectedSideSection");
                          const toggle = document.querySelector("#chartExpectedToggle");
                          const icon = document.querySelector(".chart-expected-toggle-icon");
                          return {
                            expectedHidden: expected?.hidden || false,
                            expectedInert: expected?.hasAttribute("inert") || false,
                            expectedAriaHidden: expected?.getAttribute("aria-hidden") || "",
                            savedHeight: localStorage.getItem("py_lucidum_chart_feature_controls_height") || "",
                            toggleExpanded: toggle?.getAttribute("aria-expanded") || "",
                            toggleLabel: toggle?.getAttribute("aria-label") || "",
                            iconTransform: icon ? getComputedStyle(icon).transform : "",
                          };
                        }
                        """
                    )
                    self.assertTrue(toggle_collapsed_state["expectedHidden"])
                    self.assertTrue(toggle_collapsed_state["expectedInert"])
                    self.assertEqual(toggle_collapsed_state["expectedAriaHidden"], "true")
                    self.assertEqual(toggle_collapsed_state["savedHeight"], "160")
                    self.assertEqual(toggle_collapsed_state["toggleExpanded"], "false")
                    self.assertEqual(toggle_collapsed_state["toggleLabel"], "Show Expected controls")
                    self.assertEqual(toggle_collapsed_state["iconTransform"], initial_split_state["iconTransform"])

                    page.locator("#chartExpectedToggle").click()
                    page.wait_for_function(
                        """
                        () => !document.querySelector("#chartSideControls")?.classList.contains("chart-expected-collapsed")
                        """,
                        timeout=10_000,
                    )

                    resizer = page.locator("#chartControlHeightResizer")
                    resizer_box = resizer.bounding_box()
                    controls_box = page.locator("#chartSideControls").bounding_box()
                    self.assertIsNotNone(resizer_box)
                    self.assertIsNotNone(controls_box)
                    assert resizer_box is not None
                    assert controls_box is not None
                    resizer_center_x = resizer_box["x"] + resizer_box["width"] / 2
                    page.mouse.move(resizer_center_x, resizer_box["y"] + resizer_box["height"] / 2)
                    page.mouse.down()
                    page.mouse.move(resizer_center_x, controls_box["y"] + controls_box["height"] + 80)
                    page.mouse.up()
                    page.wait_for_function(
                        """
                        () => document.querySelector("#chartSideControls")?.classList.contains("chart-expected-collapsed")
                        """,
                        timeout=10_000,
                    )
                    collapsed_split_state = page.evaluate(
                        """
                        () => {
                          const controls = document.querySelector("#chartSideControls");
                          const feature = document.querySelector(".feature-section");
                          const expected = document.querySelector("#expectedSideSection");
                          const resizer = document.querySelector("#chartControlHeightResizer");
                          const controlsRect = controls?.getBoundingClientRect();
                          const resizerRect = resizer?.getBoundingClientRect();
                          return {
                            collapsed: controls?.classList.contains("chart-expected-collapsed") || false,
                            controlsHeight: controlsRect?.height || 0,
                            featureHeight: feature?.getBoundingClientRect().height || 0,
                            expectedHeight: expected?.getBoundingClientRect().height || 0,
                            expectedHidden: expected?.hidden || false,
                            expectedInert: expected?.hasAttribute("inert") || false,
                            expectedAriaHidden: expected?.getAttribute("aria-hidden") || "",
                            savedHeight: localStorage.getItem("py_lucidum_chart_feature_controls_height") || "",
                            toggleExpanded: document.querySelector("#chartExpectedToggle")?.getAttribute("aria-expanded") || "",
                            resizerBottomDelta: controlsRect && resizerRect
                              ? Math.abs(controlsRect.bottom - resizerRect.bottom)
                              : 999,
                          };
                        }
                        """
                    )
                    self.assertTrue(collapsed_split_state["collapsed"])
                    self.assertTrue(collapsed_split_state["expectedHidden"])
                    self.assertTrue(collapsed_split_state["expectedInert"])
                    self.assertEqual(collapsed_split_state["expectedAriaHidden"], "true")
                    self.assertEqual(collapsed_split_state["savedHeight"], "160")
                    self.assertEqual(collapsed_split_state["toggleExpanded"], "false")
                    self.assertLessEqual(collapsed_split_state["expectedHeight"], 1)
                    self.assertLessEqual(collapsed_split_state["resizerBottomDelta"], 1)
                    self.assertGreater(
                        collapsed_split_state["featureHeight"],
                        expanded_split_state["featureHeight"] + 80,
                    )
                    self.assertGreaterEqual(
                        collapsed_split_state["featureHeight"],
                        collapsed_split_state["controlsHeight"] - 36,
                    )

                    resizer_box = resizer.bounding_box()
                    controls_box = page.locator("#chartSideControls").bounding_box()
                    self.assertIsNotNone(resizer_box)
                    self.assertIsNotNone(controls_box)
                    assert resizer_box is not None
                    assert controls_box is not None
                    resizer_center_x = resizer_box["x"] + resizer_box["width"] / 2
                    page.mouse.move(resizer_center_x, resizer_box["y"] + resizer_box["height"] / 2)
                    page.mouse.down()
                    page.mouse.move(resizer_center_x, controls_box["y"] + controls_box["height"] * 0.55)
                    page.mouse.up()
                    page.wait_for_function(
                        """
                        () => {
                          const controls = document.querySelector("#chartSideControls");
                          const expected = document.querySelector("#expectedSideSection");
                          return controls
                            && expected
                            && !controls.classList.contains("chart-expected-collapsed")
                            && !expected.hidden
                            && !expected.hasAttribute("inert");
                        }
                        """,
                        timeout=10_000,
                    )
                    restored_split_state = page.evaluate(
                        """
                        () => {
                          const controls = document.querySelector("#chartSideControls");
                          const expected = document.querySelector("#expectedSideSection");
                          return {
                            collapsed: controls?.classList.contains("chart-expected-collapsed") || false,
                            expectedHeight: expected?.getBoundingClientRect().height || 0,
                            expectedHidden: expected?.hidden || false,
                            expectedInert: expected?.hasAttribute("inert") || false,
                            expectedAriaHidden: expected?.getAttribute("aria-hidden") || "",
                            savedHeight: localStorage.getItem("py_lucidum_chart_feature_controls_height") || "",
                            toggleExpanded: document.querySelector("#chartExpectedToggle")?.getAttribute("aria-expanded") || "",
                          };
                        }
                        """
                    )
                    self.assertFalse(restored_split_state["collapsed"])
                    self.assertFalse(restored_split_state["expectedHidden"])
                    self.assertFalse(restored_split_state["expectedInert"])
                    self.assertEqual(restored_split_state["expectedAriaHidden"], "")
                    self.assertEqual(restored_split_state["savedHeight"], "160")
                    self.assertEqual(restored_split_state["toggleExpanded"], "true")
                    self.assertGreater(restored_split_state["expectedHeight"], 96)

                    page.locator('#featureList .feature.active[data-value="feature_001"]').wait_for(timeout=10_000)
                    page.locator("#featureSearch").focus()
                    page.keyboard.press("ArrowDown")
                    page.locator('#featureList .feature.active[data-value="feature_002"]').wait_for(timeout=10_000)
                    feature_arrow_down_state = page.evaluate(
                        """
                        () => ({
                          activeValue: document.querySelector("#featureList .feature.active")?.dataset.value || "",
                          focusedId: document.activeElement?.id || "",
                        })
                        """
                    )
                    self.assertEqual(feature_arrow_down_state["activeValue"], "feature_002")
                    self.assertEqual(feature_arrow_down_state["focusedId"], "featureSearch")
                    page.keyboard.press("ArrowUp")
                    page.locator('#featureList .feature.active[data-value="feature_001"]').wait_for(timeout=10_000)
                    feature_arrow_up_state = page.evaluate(
                        """
                        () => ({
                          activeValue: document.querySelector("#featureList .feature.active")?.dataset.value || "",
                          focusedId: document.activeElement?.id || "",
                        })
                        """
                    )
                    self.assertEqual(feature_arrow_up_state["activeValue"], "feature_001")
                    self.assertEqual(feature_arrow_up_state["focusedId"], "featureSearch")

                    page.locator("#expectedList .feature.active.expected-none-option").focus()
                    page.keyboard.press("ArrowDown")
                    page.locator('#expectedList .feature.active[data-value="actual"]').wait_for(timeout=10_000)
                    page.wait_for_function(
                        """
                        () => document.activeElement?.closest?.("#expectedList")
                          && document.activeElement?.dataset?.value === "actual"
                        """,
                        timeout=10_000,
                    )
                    expected_arrow_down_state = page.evaluate(
                        """
                        () => {
                          const active = document.querySelector("#expectedList .feature.active");
                          const focused = document.activeElement;
                          return {
                            activeValue: active?.dataset.value || "",
                            focusedValue: focused?.dataset?.value || "",
                            focusedInExpected: Boolean(focused?.closest?.("#expectedList")),
                          };
                        }
                        """
                    )
                    self.assertEqual(expected_arrow_down_state["activeValue"], "actual")
                    self.assertEqual(expected_arrow_down_state["focusedValue"], "actual")
                    self.assertTrue(expected_arrow_down_state["focusedInExpected"])
                    page.keyboard.press("ArrowUp")
                    page.locator("#expectedList .feature.active.expected-none-option").wait_for(timeout=10_000)
                    page.wait_for_function(
                        """
                        () => document.activeElement?.closest?.("#expectedList")
                          && document.activeElement?.classList?.contains("expected-none-option")
                        """,
                        timeout=10_000,
                    )
                    expected_arrow_up_state = page.evaluate(
                        """
                        () => {
                          const active = document.querySelector("#expectedList .feature.active");
                          const focused = document.activeElement;
                          return {
                            activeValue: active?.dataset.value || "",
                            focusedValue: focused?.dataset?.value || "",
                            focusedInExpected: Boolean(focused?.closest?.("#expectedList")),
                            focusedNone: focused?.classList?.contains("expected-none-option") || false,
                          };
                        }
                        """
                    )
                    self.assertEqual(expected_arrow_up_state["activeValue"], "")
                    self.assertEqual(expected_arrow_up_state["focusedValue"], "")
                    self.assertTrue(expected_arrow_up_state["focusedInExpected"])
                    self.assertTrue(expected_arrow_up_state["focusedNone"])

                    page.locator('#featureList .feature[data-value="feature_003"]').click()
                    page.wait_for_function(
                        """
                        () => document.activeElement?.closest?.("#featureList")
                          && document.activeElement?.dataset?.value === "feature_003"
                        """,
                        timeout=10_000,
                    )
                    page.keyboard.press("ArrowDown")
                    page.wait_for_function(
                        """
                        () => document.querySelector("#featureList .feature.active")?.dataset.value === "feature_004"
                          && document.activeElement?.closest?.("#featureList")
                          && document.activeElement?.dataset?.value === "feature_004"
                        """,
                        timeout=10_000,
                    )
                    feature_click_arrow_state = page.evaluate(
                        """
                        () => {
                          const previous = document.querySelector('#featureList .feature[data-value="feature_003"]');
                          const active = document.querySelector("#featureList .feature.active");
                          return {
                            activeValue: active?.dataset.value || "",
                            activeCount: document.querySelectorAll("#featureList .feature.active").length,
                            focusedValue: document.activeElement?.dataset?.value || "",
                            focusedInFeature: Boolean(document.activeElement?.closest?.("#featureList")),
                            keyboardNavigation: document.querySelector("#featureList")?.classList?.contains("line-bar-keyboard-navigation") || false,
                            previousActive: previous?.classList?.contains("active") || false,
                            previousHovered: previous?.matches(":hover") || false,
                            previousBackground: previous ? getComputedStyle(previous).backgroundColor : "",
                            activeBackground: active ? getComputedStyle(active).backgroundColor : "",
                          };
                        }
                        """
                    )
                    self.assertEqual(feature_click_arrow_state["activeValue"], "feature_004")
                    self.assertEqual(feature_click_arrow_state["activeCount"], 1)
                    self.assertEqual(feature_click_arrow_state["focusedValue"], "feature_004")
                    self.assertTrue(feature_click_arrow_state["focusedInFeature"])
                    self.assertTrue(feature_click_arrow_state["keyboardNavigation"])
                    self.assertFalse(feature_click_arrow_state["previousActive"])
                    self.assertTrue(feature_click_arrow_state["previousHovered"])
                    self.assertEqual(feature_click_arrow_state["previousBackground"], "rgba(0, 0, 0, 0)")
                    self.assertNotEqual(
                        feature_click_arrow_state["previousBackground"],
                        feature_click_arrow_state["activeBackground"],
                    )

                    page.evaluate(
                        """
                        () => document.querySelector("#featureList .feature")?.focus({ preventScroll: true })
                        """
                    )
                    page.keyboard.press("ArrowUp")
                    top_boundary_focus_state = page.evaluate(
                        """
                        () => {
                          const focused = document.activeElement;
                          const style = focused ? getComputedStyle(focused) : null;
                          return {
                            focusedInFeature: Boolean(focused?.closest?.("#featureList")),
                            outlineStyle: style?.outlineStyle || "",
                            outlineWidth: style?.outlineWidth || "",
                          };
                        }
                        """
                    )
                    self.assertTrue(top_boundary_focus_state["focusedInFeature"])
                    self.assertEqual(top_boundary_focus_state["outlineStyle"], "none")
                    self.assertEqual(top_boundary_focus_state["outlineWidth"], "0px")

                    page.evaluate(
                        """
                        () => {
                          const buttons = [...document.querySelectorAll("#featureList .feature")];
                          buttons[buttons.length - 1]?.focus({ preventScroll: true });
                        }
                        """
                    )
                    page.keyboard.press("ArrowDown")
                    bottom_boundary_focus_state = page.evaluate(
                        """
                        () => {
                          const focused = document.activeElement;
                          const style = focused ? getComputedStyle(focused) : null;
                          return {
                            focusedInFeature: Boolean(focused?.closest?.("#featureList")),
                            outlineStyle: style?.outlineStyle || "",
                            outlineWidth: style?.outlineWidth || "",
                          };
                        }
                        """
                    )
                    self.assertTrue(bottom_boundary_focus_state["focusedInFeature"])
                    self.assertEqual(bottom_boundary_focus_state["outlineStyle"], "none")
                    self.assertEqual(bottom_boundary_focus_state["outlineWidth"], "0px")

                    page.locator('#expectedList .feature[data-value="actual"]').click()
                    page.wait_for_function(
                        """
                        () => document.activeElement?.closest?.("#expectedList")
                          && document.activeElement?.dataset?.value === "actual"
                        """,
                        timeout=10_000,
                    )
                    page.keyboard.press("ArrowUp")
                    page.wait_for_function(
                        """
                        () => document.querySelector("#expectedList .feature.active")?.classList?.contains("expected-none-option")
                          && document.activeElement?.closest?.("#expectedList")
                          && document.activeElement?.classList?.contains("expected-none-option")
                        """,
                        timeout=10_000,
                    )
                    expected_click_arrow_state = page.evaluate(
                        """
                        () => ({
                          activeValue: document.querySelector("#expectedList .feature.active")?.dataset.value || "",
                          focusedValue: document.activeElement?.dataset?.value || "",
                          focusedInExpected: Boolean(document.activeElement?.closest?.("#expectedList")),
                          focusedNone: document.activeElement?.classList?.contains("expected-none-option") || false,
                        })
                        """
                    )
                    self.assertEqual(expected_click_arrow_state["activeValue"], "")
                    self.assertEqual(expected_click_arrow_state["focusedValue"], "")
                    self.assertTrue(expected_click_arrow_state["focusedInExpected"])
                    self.assertTrue(expected_click_arrow_state["focusedNone"])

                    def click_scrolled_picker(list_id: str, value: str) -> dict[str, int | str]:
                        return page.evaluate(
                            """
                            ({ listId, value }) => {
                              const list = document.querySelector(`#${listId}`);
                              const beforeRegion = list?.querySelector(".line-bar-scroll-region") || list;
                              const button = [...(beforeRegion?.querySelectorAll("button.feature") || [])]
                                .find((item) => item.dataset.value === value);
                              if (!list || !beforeRegion || !button) throw new Error(`Missing ${listId} ${value}`);
                              button.scrollIntoView({ block: "center" });
                              const before = beforeRegion.scrollTop;
                              button.click();
                              const afterRegion = list.querySelector(".line-bar-scroll-region") || list;
                              return {
                                before,
                                after: afterRegion.scrollTop,
                                activeValue: list.querySelector(".feature.active")?.dataset.value || "",
                              };
                            }
                            """,
                            {"listId": list_id, "value": value},
                        )

                    feature_result = click_scrolled_picker("featureList", "feature_130")
                    expected_result = click_scrolled_picker("expectedList", "feature_145")
                    self.assertEqual(feature_result["activeValue"], "feature_130")
                    self.assertEqual(expected_result["activeValue"], "feature_145")
                    self.assertGreater(feature_result["before"], 0)
                    self.assertGreater(expected_result["before"], 0)
                    self.assertGreater(feature_result["after"], 0)
                    self.assertGreater(expected_result["after"], 0)
                    self.assertLessEqual(abs(feature_result["after"] - feature_result["before"]), 2)
                    self.assertLessEqual(abs(expected_result["after"] - expected_result["before"]), 2)
                    self.assertEqual(page_errors, [])
                    browser.close()
            finally:
                server.should_exit = True
                thread.join(timeout=5)

    @unittest.skipUnless(RUN_BROWSER_TESTS, "set PY_LUCIDUM_RUN_BROWSER_TESTS=1 to run browser smoke tests")
    @unittest.skipUnless(sync_playwright is not None, "playwright is not installed")
    def test_line_bar_collapsed_side_controls_stay_inside_viewport_with_long_header(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            data_path = Path(tmp_dir) / (
                "line_bar_zoom_overflow_regression_dataset_with_a_long_unbroken_name_for_header_truncation.csv"
            )
            columns = ["PostcodeArea", "PostcodeSector", "PostcodeUnit", "vehicle_age", "price", "value"]
            columns.extend(f"feature_{index:03d}" for index in range(1, 31))
            rows = [",".join(columns)]
            for row_index in range(1, 8):
                values = [
                    "AB",
                    "AB10 1",
                    "AB10 1AA",
                    str(row_index),
                    str(100 + row_index * 12),
                    str(10 + row_index),
                ]
                values.extend(str((row_index * column_index) % 97) for column_index in range(1, 31))
                rows.append(",".join(values))
            data_path.write_text("\n".join(rows) + "\n", encoding="utf-8")
            base_url, server, thread = self.start_app(
                data_path,
                defaults={
                    "x": "vehicle_age",
                    "actual": "price",
                    "denominator": "value",
                },
            )
            try:
                assert sync_playwright is not None
                with sync_playwright() as playwright:
                    browser = playwright.chromium.launch()
                    page = browser.new_page(viewport={"width": 760, "height": 720})
                    page_errors: list[str] = []
                    page.on("pageerror", lambda error: page_errors.append(str(error)))
                    page.goto(base_url, wait_until="domcontentloaded")
                    page.locator("#datasetMeta").get_by_text("line_bar_zoom_overflow_regression_dataset").wait_for(
                        timeout=10_000
                    )
                    page.locator("#chart:not(.hidden)").wait_for(timeout=10_000)
                    page.wait_for_function(
                        """
                        () => {
                          const chartNode = document.querySelector("#chart");
                          const chart = window.echarts?.getInstanceByDom(chartNode);
                          return Boolean(chart && chart.getWidth() > 0 && chartNode.querySelector("canvas"));
                        }
                        """,
                        timeout=10_000,
                    )
                    if page.locator("#sidebarToggleBtn").get_attribute("aria-expanded") == "true":
                        page.locator("#sidebarToggleBtn").click()
                        page.wait_for_function(
                            """
                            () => document.querySelector("#sidebarToggleBtn")?.getAttribute("aria-expanded") === "false"
                            """,
                            timeout=10_000,
                        )
                    page.wait_for_function(
                        """
                        () => document.querySelector("#visualArea")?.classList.contains("line-bar-side-controls-collapsed")
                          && document.querySelector("#lineBarSideControlsToggleBtn")?.getAttribute("aria-expanded") === "false"
                        """,
                        timeout=10_000,
                    )
                    page.wait_for_timeout(250)
                    layout = page.evaluate(
                        """
                        () => {
                          const rectFor = (selector) => {
                            const node = document.querySelector(selector);
                            if (!node) return null;
                            const rect = node.getBoundingClientRect();
                            return {
                              left: rect.left,
                              right: rect.right,
                              width: rect.width,
                              clientWidth: node.clientWidth,
                              scrollWidth: node.scrollWidth,
                            };
                          };
                          const canvas = document.querySelector("#chart canvas");
                          const canvasRect = canvas?.getBoundingClientRect();
                          const chartNode = document.querySelector("#chart");
                          const chart = window.echarts?.getInstanceByDom(chartNode);
                          return {
                            innerWidth: window.innerWidth,
                            app: rectFor(".app"),
                            meta: rectFor("#datasetMeta"),
                            rects: {
                              header: rectFor("header"),
                              shell: rectFor(".shell"),
                              visualArea: rectFor("#visualArea"),
                              workspace: rectFor(".workspace"),
                              chart: rectFor("#chart"),
                            },
                            canvas: canvasRect ? {
                              left: canvasRect.left,
                              right: canvasRect.right,
                              width: canvasRect.width,
                            } : null,
                            chartWidth: chart?.getWidth?.() || 0,
                          };
                        }
                        """
                    )
                    self.assertIsNotNone(layout["app"])
                    self.assertIsNotNone(layout["meta"])
                    self.assertIsNotNone(layout["canvas"])
                    self.assertGreater(layout["chartWidth"], 0)
                    self.assertGreater(layout["meta"]["scrollWidth"], layout["meta"]["clientWidth"])
                    self.assertLessEqual(layout["app"]["scrollWidth"], layout["innerWidth"] + 1)
                    for selector, rect in layout["rects"].items():
                        self.assertIsNotNone(rect, selector)
                        self.assertGreaterEqual(rect["left"], -1, selector)
                        self.assertLessEqual(rect["right"], layout["innerWidth"] + 1, selector)
                    self.assertLessEqual(layout["canvas"]["right"], layout["innerWidth"] + 1)
                    self.assertEqual(page_errors, [])
                    browser.close()
            finally:
                server.should_exit = True
                thread.join(timeout=5)

    @unittest.skipUnless(RUN_BROWSER_TESTS, "set PY_LUCIDUM_RUN_BROWSER_TESTS=1 to run browser smoke tests")
    @unittest.skipUnless(sync_playwright is not None, "playwright is not installed")
    def test_line_bar_missing_exclusions_render_in_top_meta_for_chart_and_table(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            data_path = Path(tmp_dir) / "line_bar_missing_exclusions.csv"
            data_path.write_text(
                "MAKE,PREMIUM\n"
                "ALFA ROMEO,100\n"
                "ALFA ROMEO,\n"
                "BMW,300\n",
                encoding="utf-8",
            )
            base_url, server, thread = self.start_app(
                data_path,
                defaults={
                    "x": "MAKE",
                    "actual": "PREMIUM",
                    "denominator": "__none__",
                },
                tools=["line_bar"],
            )
            try:
                assert sync_playwright is not None
                with sync_playwright() as playwright:
                    browser = playwright.chromium.launch()
                    page = browser.new_page(viewport={"width": 1280, "height": 800})
                    page_errors: list[str] = []
                    page.on("pageerror", lambda error: page_errors.append(str(error)))

                    page.goto(base_url, wait_until="domcontentloaded")
                    page.locator("#datasetMeta").get_by_text("line_bar_missing_exclusions.csv").wait_for(timeout=10_000)
                    page.wait_for_function(
                        """
                        () => document.querySelector("#lineBarGroupMeta")?.textContent.trim() ===
                          "2 groups · 3 rows. 1 row excluded due to missing PREMIUM"
                          && !(document.querySelector("#chartMessage")?.textContent || "").includes("missing PREMIUM")
                        """,
                        timeout=10_000,
                    )
                    page.locator("#tableTab").click()
                    page.locator("#tableWrap:not(.hidden) #lineBarTableGrid").wait_for(timeout=10_000)
                    page.wait_for_function(
                        """
                        () => document.querySelector("#lineBarGroupMeta")?.textContent.trim() ===
                          "2 groups · 3 rows. 1 row excluded due to missing PREMIUM"
                          && !(document.querySelector("#chartMessage")?.textContent || "").includes("missing PREMIUM")
                        """,
                        timeout=10_000,
                    )
                    self.assertEqual(page_errors, [])
                    browser.close()
            finally:
                server.should_exit = True
                thread.join(timeout=5)

    @unittest.skipUnless(RUN_BROWSER_TESTS, "set PY_LUCIDUM_RUN_BROWSER_TESTS=1 to run browser smoke tests")
    @unittest.skipUnless(sync_playwright is not None, "playwright is not installed")
    def test_line_bar_table_search_filters_complete_table_client_side(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            data_path = Path(tmp_dir) / "line_bar_table.csv"
            data_path.write_text(
                "MAKE,PREMIUM\n"
                "ALFA ROMEO,100\n"
                "ALFA ROMEO,200\n"
                "BMW,300\n",
                encoding="utf-8",
            )
            base_url, server, thread = self.start_app(
                data_path,
                defaults={
                    "x": "MAKE",
                    "actual": "PREMIUM",
                    "denominator": "__none__",
                },
                tools=["line_bar"],
            )
            try:
                assert sync_playwright is not None
                with sync_playwright() as playwright:
                    browser = playwright.chromium.launch()
                    page = browser.new_page(viewport={"width": 1280, "height": 800})
                    page.emulate_media(color_scheme="light")
                    page_errors: list[str] = []
                    chart_requests = 0
                    table_requests = 0
                    page.on("pageerror", lambda error: page_errors.append(str(error)))

                    def count_request(request: object) -> None:
                        nonlocal chart_requests, table_requests
                        if request.url.endswith("/api/chart"):
                            chart_requests += 1
                        elif request.url.endswith("/api/line-bar/table"):
                            table_requests += 1

                    page.on("request", count_request)
                    page.add_init_script(
                        """
                        window.__lucidumCopiedText = null;
                        window.__lucidumCopiedImage = null;
                        Object.defineProperty(navigator, "clipboard", {
                            configurable: true,
                            value: {
                                write: async (items) => {
                                    const item = items?.[0];
                                    const blob = item ? await item.getType("image/png") : null;
                                    window.__lucidumCopiedImage = {
                                        types: item ? Array.from(item.types || []) : [],
                                        type: blob?.type || "",
                                        size: blob?.size || 0,
                                    };
                                },
                                writeText: async (text) => {
                                    window.__lucidumCopiedText = text;
                                },
                            },
                        });
                        """
                    )

                    def wait_for_total_row_theme(expected_dark: bool) -> dict[str, Any]:
                        page.wait_for_function(
                            """
                            (expectedDark) => {
                              const normalizeColor = (value) => {
                                const probe = document.createElement("span");
                                probe.style.color = value;
                                document.body.append(probe);
                                const color = getComputedStyle(probe).color;
                                probe.remove();
                                return color;
                              };
                              const bodyStyles = getComputedStyle(document.body);
                              const panelColor = normalizeColor(bodyStyles.getPropertyValue("--panel").trim());
                              const textColor = normalizeColor(bodyStyles.getPropertyValue("--text").trim());
                              const holder = document.querySelector("#lineBarTableGrid .tabulator-footer .tabulator-calcs-holder");
                              const row = holder?.querySelector(".tabulator-row.tabulator-calcs");
                              const cell = row?.querySelector(".tabulator-cell");
                              if (!holder || !row || !cell) return false;
                              return document.body.classList.contains("dark") === expectedDark
                                && getComputedStyle(holder).backgroundColor === panelColor
                                && getComputedStyle(row).backgroundColor === panelColor
                                && getComputedStyle(cell).color === textColor;
                            }
                            """,
                            arg=expected_dark,
                            timeout=10_000,
                        )
                        return page.evaluate(
                            """
                            () => {
                              const normalizeColor = (value) => {
                                const probe = document.createElement("span");
                                probe.style.color = value;
                                document.body.append(probe);
                                const color = getComputedStyle(probe).color;
                                probe.remove();
                                return color;
                              };
                              const bodyStyles = getComputedStyle(document.body);
                              const panelColor = normalizeColor(bodyStyles.getPropertyValue("--panel").trim());
                              const textColor = normalizeColor(bodyStyles.getPropertyValue("--text").trim());
                              const holder = document.querySelector("#lineBarTableGrid .tabulator-footer .tabulator-calcs-holder");
                              const row = holder?.querySelector(".tabulator-row.tabulator-calcs");
                              const cell = row?.querySelector(".tabulator-cell");
                              return {
                                dark: document.body.classList.contains("dark"),
                                panelColor,
                                textColor,
                                holderBackground: holder ? getComputedStyle(holder).backgroundColor : "",
                                rowBackground: row ? getComputedStyle(row).backgroundColor : "",
                                cellColor: cell ? getComputedStyle(cell).color : "",
                              };
                            }
                            """
                        )

                    page.goto(base_url, wait_until="domcontentloaded")
                    page.locator("#datasetMeta").get_by_text("line_bar_table.csv").wait_for(timeout=10_000)
                    page.locator("#chart:not(.hidden)").wait_for(timeout=10_000)
                    page.wait_for_function(
                        """
                        () => document.querySelector("#lineBarGroupMeta")?.textContent.includes("2 groups")
                        """
                    )
                    initial_focus_state = page.evaluate(
                        """
                        () => {
                          const rootStyle = getComputedStyle(document.documentElement);
                          const displayFor = (selector) => getComputedStyle(document.querySelector(selector)).display;
                          const chartRect = document.querySelector("#chart").getBoundingClientRect();
                          return {
                            sidebarWidth: rootStyle.getPropertyValue("--sidebar-width").trim(),
                            chartControlsWidth: rootStyle.getPropertyValue("--chart-controls-width").trim(),
                            sidebarExpanded: document.querySelector("#sidebarToggleBtn").getAttribute("aria-expanded"),
                            chartWidth: chartRect.width,
                            toolbarDisplay: displayFor("#lineBarToolbar"),
                            controlsDisplay: displayFor("#chartSideControls"),
                          };
                        }
                        """
                    )
                    self.assertEqual(initial_focus_state["sidebarExpanded"], "true")
                    self.assertEqual(initial_focus_state["toolbarDisplay"], "none")
                    self.assertEqual(initial_focus_state["controlsDisplay"], "none")
                    page.locator("#lineBarCopyBtn").click()
                    page.wait_for_function(
                        """
                        () => window.__lucidumCopiedImage
                          && window.__lucidumCopiedImage.types.includes("image/png")
                          && window.__lucidumCopiedImage.type === "image/png"
                          && window.__lucidumCopiedImage.size > 0
                        """,
                        timeout=10_000,
                    )
                    initial_toggle_state = page.evaluate(
                        """
                        () => ({
                          sideExpanded: document.querySelector("#lineBarSideControlsToggleBtn").getAttribute("aria-expanded"),
                          toolbarExpanded: document.querySelector("#lineBarToolbarToggleBtn").getAttribute("aria-expanded"),
                          sideLabel: document.querySelector("#lineBarSideControlsToggleBtn").getAttribute("aria-label"),
                          toolbarLabel: document.querySelector("#lineBarToolbarToggleBtn").getAttribute("aria-label"),
                          sideRight: document.querySelector("#lineBarSideControlsToggleBtn").getBoundingClientRect().right,
                          toolbarLeft: document.querySelector("#lineBarToolbarToggleBtn").getBoundingClientRect().left,
                          toolbarRight: document.querySelector("#lineBarToolbarToggleBtn").getBoundingClientRect().right,
                          copyLeft: document.querySelector("#lineBarCopyBtn").getBoundingClientRect().left,
                          copyRight: document.querySelector("#lineBarCopyBtn").getBoundingClientRect().right,
                          chartLeft: document.querySelector("#chartTab").getBoundingClientRect().left,
                        })
                        """
                    )
                    self.assertEqual(initial_toggle_state["sideExpanded"], "false")
                    self.assertEqual(initial_toggle_state["toolbarExpanded"], "false")
                    self.assertEqual(initial_toggle_state["sideLabel"], "Show x-axis and Expected controls")
                    self.assertEqual(initial_toggle_state["toolbarLabel"], "Show chart control row")
                    self.assertLessEqual(initial_toggle_state["sideRight"], initial_toggle_state["toolbarLeft"])
                    self.assertLessEqual(initial_toggle_state["toolbarRight"], initial_toggle_state["copyLeft"])
                    self.assertLessEqual(initial_toggle_state["copyRight"], initial_toggle_state["chartLeft"])
                    page.locator("#lineBarSideControlsToggleBtn").click()
                    page.locator("#lineBarToolbarToggleBtn").click()
                    page.wait_for_function(
                        """
                        () => document.querySelector("#lineBarSideControlsToggleBtn")?.getAttribute("aria-expanded") === "true"
                          && document.querySelector("#lineBarToolbarToggleBtn")?.getAttribute("aria-expanded") === "true"
                          && getComputedStyle(document.querySelector("#chartSideControls")).display !== "none"
                          && getComputedStyle(document.querySelector("#lineBarToolbar")).display !== "none"
                        """,
                        timeout=10_000,
                    )
                    expanded_focus_state = page.evaluate(
                        """
                        () => {
                          const rootStyle = getComputedStyle(document.documentElement);
                          const chartRect = document.querySelector("#chart").getBoundingClientRect();
                          return {
                            sidebarWidth: rootStyle.getPropertyValue("--sidebar-width").trim(),
                            chartControlsWidth: rootStyle.getPropertyValue("--chart-controls-width").trim(),
                            chartWidth: chartRect.width,
                          };
                        }
                        """
                    )
                    page.locator("#lineBarSideControlsToggleBtn").click()
                    page.wait_for_function(
                        """
                        () => document.querySelector("#visualArea")?.classList.contains("line-bar-side-controls-collapsed")
                          && document.querySelector("#sidebarToggleBtn")?.getAttribute("aria-expanded") === "true"
                          && getComputedStyle(document.querySelector("#chartSideControls")).display === "none"
                          && getComputedStyle(document.querySelector("#chartControlsResizer")).display === "none"
                          && getComputedStyle(document.querySelector("#lineBarToolbar")).display !== "none"
                          && getComputedStyle(document.querySelector("#chartTab")).display !== "none"
                          && getComputedStyle(document.querySelector("#tableTab")).display !== "none"
                          && getComputedStyle(document.querySelector("#lineBarCopyBtn")).display !== "none"
                          && getComputedStyle(document.querySelector("#lineBarSideControlsToggleBtn")).display !== "none"
                          && getComputedStyle(document.querySelector("#lineBarToolbarToggleBtn")).display !== "none"
                        """,
                        timeout=10_000,
                    )
                    side_collapsed_state = page.evaluate(
                        """
                        () => {
                          const rootStyle = getComputedStyle(document.documentElement);
                          const chartRect = document.querySelector("#chart").getBoundingClientRect();
                          return {
                            sidebarWidth: rootStyle.getPropertyValue("--sidebar-width").trim(),
                            chartControlsWidth: rootStyle.getPropertyValue("--chart-controls-width").trim(),
                            chartWidth: chartRect.width,
                            sideExpanded: document.querySelector("#lineBarSideControlsToggleBtn").getAttribute("aria-expanded"),
                            toolbarExpanded: document.querySelector("#lineBarToolbarToggleBtn").getAttribute("aria-expanded"),
                            sideIconTransform: getComputedStyle(document.querySelector("#lineBarSideControlsToggleBtn .line-bar-chevron-horizontal")).transform,
                          };
                        }
                        """
                    )
                    self.assertEqual(side_collapsed_state["sidebarWidth"], expanded_focus_state["sidebarWidth"])
                    self.assertEqual(side_collapsed_state["chartControlsWidth"], expanded_focus_state["chartControlsWidth"])
                    self.assertEqual(side_collapsed_state["sideExpanded"], "false")
                    self.assertEqual(side_collapsed_state["toolbarExpanded"], "true")
                    self.assertNotEqual(side_collapsed_state["sideIconTransform"], "none")
                    self.assertGreater(side_collapsed_state["chartWidth"], expanded_focus_state["chartWidth"] + 100)
                    page.locator("#lineBarToolbarToggleBtn").click()
                    page.wait_for_function(
                        """
                        () => document.querySelector("#lineBarToolbarToggleBtn")?.getAttribute("aria-expanded") === "false"
                          && document.querySelector("#sidebarToggleBtn")?.getAttribute("aria-expanded") === "true"
                          && getComputedStyle(document.querySelector("#lineBarToolbar")).display === "none"
                          && getComputedStyle(document.querySelector("#chartSideControls")).display === "none"
                          && getComputedStyle(document.querySelector("#chartControlsResizer")).display === "none"
                          && getComputedStyle(document.querySelector("#chartTab")).display !== "none"
                          && getComputedStyle(document.querySelector("#tableTab")).display !== "none"
                          && getComputedStyle(document.querySelector("#lineBarCopyBtn")).display !== "none"
                          && getComputedStyle(document.querySelector("#lineBarSideControlsToggleBtn")).display !== "none"
                          && getComputedStyle(document.querySelector("#lineBarToolbarToggleBtn")).display !== "none"
                        """,
                        timeout=10_000,
                    )
                    toolbar_collapsed_state = page.evaluate(
                        """
                        () => {
                          const rootStyle = getComputedStyle(document.documentElement);
                          const chartRect = document.querySelector("#chart").getBoundingClientRect();
                          return {
                            sidebarWidth: rootStyle.getPropertyValue("--sidebar-width").trim(),
                            chartControlsWidth: rootStyle.getPropertyValue("--chart-controls-width").trim(),
                            chartWidth: chartRect.width,
                            sideExpanded: document.querySelector("#lineBarSideControlsToggleBtn").getAttribute("aria-expanded"),
                            toolbarExpanded: document.querySelector("#lineBarToolbarToggleBtn").getAttribute("aria-expanded"),
                            toolbarIconTransform: getComputedStyle(document.querySelector("#lineBarToolbarToggleBtn .line-bar-chevron-vertical")).transform,
                          };
                        }
                        """
                    )
                    self.assertEqual(toolbar_collapsed_state["sidebarWidth"], expanded_focus_state["sidebarWidth"])
                    self.assertEqual(toolbar_collapsed_state["chartControlsWidth"], expanded_focus_state["chartControlsWidth"])
                    self.assertEqual(toolbar_collapsed_state["sideExpanded"], "false")
                    self.assertEqual(toolbar_collapsed_state["toolbarExpanded"], "false")
                    self.assertNotEqual(toolbar_collapsed_state["toolbarIconTransform"], "none")
                    self.assertGreaterEqual(toolbar_collapsed_state["chartWidth"], side_collapsed_state["chartWidth"] - 5)
                    page.locator("#lineBarToolbarToggleBtn").click()
                    page.wait_for_function(
                        """
                        () => document.querySelector("#lineBarToolbarToggleBtn")?.getAttribute("aria-expanded") === "true"
                          && document.querySelector("#visualArea")?.classList.contains("line-bar-side-controls-collapsed")
                          && getComputedStyle(document.querySelector("#lineBarToolbar")).display !== "none"
                        """,
                        timeout=10_000,
                    )
                    page.locator("#lineBarSideControlsToggleBtn").click()
                    page.wait_for_function(
                        """
                        () => !document.querySelector("#visualArea")?.classList.contains("line-bar-side-controls-collapsed")
                          && document.querySelector("#lineBarSideControlsToggleBtn")?.getAttribute("aria-expanded") === "true"
                          && document.querySelector("#lineBarToolbarToggleBtn")?.getAttribute("aria-expanded") === "true"
                          && document.querySelector("#sidebarToggleBtn")?.getAttribute("aria-expanded") === "true"
                          && getComputedStyle(document.querySelector("#lineBarToolbar")).display !== "none"
                          && getComputedStyle(document.querySelector("#chartSideControls")).display !== "none"
                          && getComputedStyle(document.querySelector("#chartControlsResizer")).display !== "none"
                        """,
                        timeout=10_000,
                    )
                    restored_focus_state = page.evaluate(
                        """
                        () => {
                          const rootStyle = getComputedStyle(document.documentElement);
                          const chartRect = document.querySelector("#chart").getBoundingClientRect();
                          return {
                            sidebarWidth: rootStyle.getPropertyValue("--sidebar-width").trim(),
                            chartControlsWidth: rootStyle.getPropertyValue("--chart-controls-width").trim(),
                            chartWidth: chartRect.width,
                          };
                        }
                        """
                    )
                    self.assertEqual(restored_focus_state["sidebarWidth"], expanded_focus_state["sidebarWidth"])
                    self.assertEqual(restored_focus_state["chartControlsWidth"], expanded_focus_state["chartControlsWidth"])
                    self.assertLess(restored_focus_state["chartWidth"], side_collapsed_state["chartWidth"] - 100)
                    page.locator("#sidebarToggleBtn").click()
                    page.wait_for_function(
                        """
                        () => document.querySelector("#sidebarToggleBtn")?.getAttribute("aria-expanded") === "false"
                        """,
                        timeout=10_000,
                    )
                    page.locator("#lineBarSideControlsToggleBtn").click()
                    page.wait_for_function(
                        """
                        () => document.querySelector("#visualArea")?.classList.contains("line-bar-side-controls-collapsed")
                          && document.querySelector("#sidebarToggleBtn")?.getAttribute("aria-expanded") === "false"
                        """,
                        timeout=10_000,
                    )
                    page.locator("#lineBarSideControlsToggleBtn").click()
                    page.wait_for_function(
                        """
                        () => !document.querySelector("#visualArea")?.classList.contains("line-bar-side-controls-collapsed")
                          && document.querySelector("#sidebarToggleBtn")?.getAttribute("aria-expanded") === "false"
                        """,
                        timeout=10_000,
                    )
                    page.locator("#sidebarToggleBtn").click()
                    page.wait_for_function(
                        """
                        () => document.querySelector("#sidebarToggleBtn")?.getAttribute("aria-expanded") === "true"
                        """,
                        timeout=10_000,
                    )
                    chart_requests_before_search = chart_requests
                    page.locator("#tableTab").click()
                    page.locator("#tableWrap:not(.hidden) #lineBarTableSearch").wait_for(timeout=10_000)
                    page.wait_for_function(
                        """
                        () => {
                            const xCells = [...document.querySelectorAll("#lineBarTableGrid .tabulator-row:not(.tabulator-calcs) .tabulator-cell[tabulator-field='x']")]
                              .map((cell) => cell.textContent.trim())
                              .filter(Boolean);
                            return xCells.length === 2 && xCells[0] === "ALFA ROMEO" && xCells[1] === "BMW";
                        }
                        """,
                        timeout=10_000,
                    )
                    page.locator("#lineBarSideControlsToggleBtn").click()
                    page.wait_for_function(
                        """
                        () => document.querySelector("#visualArea")?.classList.contains("line-bar-side-controls-collapsed")
                          && getComputedStyle(document.querySelector("#chartTab")).display !== "none"
                          && getComputedStyle(document.querySelector("#tableTab")).display !== "none"
                          && getComputedStyle(document.querySelector("#lineBarCopyBtn")).display !== "none"
                          && getComputedStyle(document.querySelector("#lineBarSideControlsToggleBtn")).display !== "none"
                          && getComputedStyle(document.querySelector("#lineBarToolbarToggleBtn")).display !== "none"
                        """,
                        timeout=10_000,
                    )
                    focused_table_spacing = page.evaluate(
                        """
                        () => {
                          const tabs = document.querySelector("#lineBarTabs").getBoundingClientRect();
                          const grid = document.querySelector("#lineBarTableGrid").getBoundingClientRect();
                          return {
                            tabsBottom: tabs.bottom,
                            gridTop: grid.top,
                            paddingTop: getComputedStyle(document.querySelector("#tableWrap")).paddingTop,
                          };
                        }
                        """
                    )
                    self.assertEqual(focused_table_spacing["paddingTop"], "36px")
                    self.assertGreaterEqual(focused_table_spacing["gridTop"], focused_table_spacing["tabsBottom"])
                    page.locator("#chartTab").click()
                    page.locator("#chart:not(.hidden)").wait_for(timeout=10_000)
                    page.locator("#tableTab").click()
                    page.locator("#tableWrap:not(.hidden) #lineBarTableGrid").wait_for(timeout=10_000)
                    page.locator("#lineBarSideControlsToggleBtn").click()
                    page.wait_for_function("() => !document.querySelector('#visualArea')?.classList.contains('line-bar-side-controls-collapsed')", timeout=10_000)
                    chart_requests_before_sort = chart_requests
                    table_requests_before_sort = table_requests
                    page.locator('.segmented[data-control="sort"] button[data-value="actual"]').click()
                    page.wait_for_function(
                        """
                        () => {
                            const xCells = [...document.querySelectorAll("#lineBarTableGrid .tabulator-row:not(.tabulator-calcs) .tabulator-cell[tabulator-field='x']")]
                              .map((cell) => cell.textContent.trim())
                              .filter(Boolean);
                            return xCells.length === 2 && xCells[0] === "BMW" && xCells[1] === "ALFA ROMEO";
                        }
                        """,
                        timeout=10_000,
                    )
                    page.wait_for_timeout(100)
                    self.assertEqual(chart_requests, chart_requests_before_sort)
                    self.assertEqual(table_requests, table_requests_before_sort)
                    table_requests_before_client_search = table_requests
                    page.locator("#lineBarTableSearch").fill("romeo")
                    page.wait_for_function(
                        """
                        () => {
                            const xCells = [...document.querySelectorAll("#lineBarTableGrid .tabulator-row:not(.tabulator-calcs) .tabulator-cell[tabulator-field='x']")]
                              .map((cell) => cell.textContent.trim())
                              .filter(Boolean);
                            return xCells.length === 1 && xCells[0] === "ALFA ROMEO";
                        }
                        """
                    )
                    page.wait_for_timeout(350)
                    self.assertEqual(chart_requests, chart_requests_before_search)
                    self.assertEqual(table_requests, table_requests_before_client_search)
                    table_search_state = page.evaluate(
                        """
                        () => {
                            const rectFor = (selector) => {
                              const rect = document.querySelector(selector)?.getBoundingClientRect();
                              return rect ? { top: rect.top, bottom: rect.bottom } : null;
                            };
                            const calcCells = [...document.querySelectorAll("#lineBarTableGrid .tabulator-row.tabulator-calcs .tabulator-cell")]
                              .map((cell) => cell.textContent.trim());
                            const visibleRows = [...document.querySelectorAll("#lineBarTableGrid .tabulator-row:not(.tabulator-calcs)")]
                              .map((row) => [
                                row.querySelector(".tabulator-cell[tabulator-field='x']")?.textContent.trim(),
                                row.querySelector(".tabulator-cell[tabulator-field='volume']")?.textContent.trim(),
                                row.querySelector(".tabulator-cell[tabulator-field='resp0']")?.textContent.trim(),
                              ])
                              .filter((row) => row[0]);
                            const dataRows = [...document.querySelectorAll("#lineBarTableGrid .tabulator-tableholder .tabulator-row:not(.tabulator-calcs)")].filter((row) => (
                              row.querySelector(".tabulator-cell[tabulator-field='x']")?.textContent.trim()
                            ));
                            const finalDataRowRect = dataRows.at(-1)?.getBoundingClientRect();
                            const totalRowRect = document.querySelector("#lineBarTableGrid .tabulator-footer .tabulator-row.tabulator-calcs")?.getBoundingClientRect();
                            return {
                              footerCells: calcCells,
                              visibleRows,
                              totalGap: finalDataRowRect && totalRowRect ? totalRowRect.top - finalDataRowRect.bottom : null,
                              search: rectFor(".line-bar-table-search-row"),
                              messages: rectFor(".workspace-messages"),
                            };
                        }
                        """
                    )
                    self.assertEqual(table_search_state["visibleRows"][0][0], "ALFA ROMEO")
                    self.assertEqual(table_search_state["footerCells"][0], "Total")
                    self.assertEqual(table_search_state["footerCells"][1], table_search_state["visibleRows"][0][1])
                    self.assertEqual(table_search_state["footerCells"][2], table_search_state["visibleRows"][0][2])
                    self.assertIsNotNone(table_search_state["totalGap"])
                    self.assertGreaterEqual(table_search_state["totalGap"], -1)
                    self.assertLessEqual(table_search_state["totalGap"], 6)
                    self.assertGreaterEqual(table_search_state["search"]["top"], table_search_state["messages"]["bottom"])
                    light_total_theme = wait_for_total_row_theme(False)
                    self.assertEqual(light_total_theme["holderBackground"], light_total_theme["panelColor"])
                    self.assertEqual(light_total_theme["rowBackground"], light_total_theme["panelColor"])
                    self.assertEqual(light_total_theme["cellColor"], light_total_theme["textColor"])
                    page.locator("#themeBtn").click()
                    dark_total_theme = wait_for_total_row_theme(True)
                    self.assertEqual(dark_total_theme["holderBackground"], dark_total_theme["panelColor"])
                    self.assertEqual(dark_total_theme["rowBackground"], dark_total_theme["panelColor"])
                    self.assertEqual(dark_total_theme["cellColor"], dark_total_theme["textColor"])
                    self.assertNotEqual(dark_total_theme["panelColor"], light_total_theme["panelColor"])
                    self.assertNotEqual(dark_total_theme["textColor"], light_total_theme["textColor"])
                    page.locator("#themeBtn").click()
                    light_total_theme_again = wait_for_total_row_theme(False)
                    self.assertEqual(light_total_theme_again["holderBackground"], light_total_theme_again["panelColor"])
                    self.assertEqual(light_total_theme_again["rowBackground"], light_total_theme_again["panelColor"])
                    self.assertEqual(light_total_theme_again["cellColor"], light_total_theme_again["textColor"])
                    self.assertEqual(light_total_theme_again["panelColor"], light_total_theme["panelColor"])
                    self.assertEqual(light_total_theme_again["textColor"], light_total_theme["textColor"])
                    table_requests_before_clear = table_requests
                    page.locator("#lineBarTableSearchClear").click()
                    page.wait_for_function(
                        """
                        () => document.querySelectorAll("#lineBarTableGrid .tabulator-row:not(.tabulator-calcs) .tabulator-cell[tabulator-field='x']").length === 2
                          && document.querySelector("#lineBarTableSearch")?.value === ""
                        """
                    )
                    page.wait_for_timeout(350)
                    self.assertEqual(table_requests, table_requests_before_clear)
                    clear_state = page.evaluate(
                        """
                        () => ({
                          footerCells: [...document.querySelectorAll("#lineBarTableGrid .tabulator-row.tabulator-calcs .tabulator-cell")]
                            .map((cell) => cell.textContent.trim()),
                          xCells: [...document.querySelectorAll("#lineBarTableGrid .tabulator-row:not(.tabulator-calcs) .tabulator-cell[tabulator-field='x']")]
                            .map((cell) => cell.textContent.trim())
                            .filter(Boolean),
                        })
                        """
                    )
                    self.assertEqual(clear_state["footerCells"][0], "Total")
                    self.assertEqual(clear_state["footerCells"][1], "3")
                    self.assertEqual(clear_state["footerCells"][2], "200.00")
                    self.assertEqual(clear_state["xCells"], ["BMW", "ALFA ROMEO"])

                    row_locator = page.locator("#lineBarTableGrid .tabulator-row:not(.tabulator-calcs)")
                    row_locator.nth(0).click()
                    page.wait_for_function(
                        """
                        () => document.querySelectorAll("#lineBarTableGrid .tabulator-row.tabulator-selected:not(.tabulator-calcs)").length === 1
                        """,
                        timeout=10_000,
                    )
                    page.locator('#lineBarTableGrid .tabulator-row:not(.tabulator-calcs) .tabulator-cell[tabulator-field="x"]').first.click(button="right")
                    page.locator("#lineBarTableContextMenu:not([hidden])").get_by_text("Copy cell to clipboard").wait_for(timeout=10_000)
                    page.locator("#lineBarTableContextMenu:not([hidden])").get_by_text("Copy selected row to clipboard").wait_for(timeout=10_000)
                    page.locator("#lineBarTableContextMenu:not([hidden])").get_by_text("Copy table to clipboard").wait_for(timeout=10_000)
                    self.assertNotIn("Copy selected column", page.locator("#lineBarTableContextMenu:not([hidden])").text_content())
                    page.locator("#lineBarTableContextMenu:not([hidden])").get_by_text("Copy cell to clipboard").click()
                    page.wait_for_function("() => window.__lucidumCopiedText === 'BMW'", timeout=10_000)

                    selected_row_csv = page.evaluate(
                        """
                        () => {
                          const fields = ["x", "volume", "resp0"];
                          const headers = fields.map((field) => document.querySelector(`#lineBarTableGrid .tabulator-col[tabulator-field="${field}"] .tabulator-col-title`)?.textContent.trim() || field);
                          const row = document.querySelector("#lineBarTableGrid .tabulator-row.tabulator-selected:not(.tabulator-calcs)");
                          const values = fields.map((field) => row?.querySelector(`.tabulator-cell[tabulator-field="${field}"]`)?.textContent.trim() || "");
                          return [headers.join(","), values.join(",")].join("\\n");
                        }
                        """
                    )
                    page.locator('#lineBarTableGrid .tabulator-row:not(.tabulator-calcs) .tabulator-cell[tabulator-field="x"]').first.click(button="right")
                    page.locator("#lineBarTableContextMenu:not([hidden])").get_by_text("Copy selected row to clipboard").click()
                    page.wait_for_function(
                        "(expected) => window.__lucidumCopiedText === expected",
                        arg=selected_row_csv,
                        timeout=10_000,
                    )

                    row_locator.nth(1).click()
                    page.wait_for_function(
                        """
                        () => document.querySelectorAll("#lineBarTableGrid .tabulator-row.tabulator-selected:not(.tabulator-calcs)").length === 2
                        """,
                        timeout=10_000,
                    )
                    selected_rows_csv = page.evaluate(
                        """
                        () => {
                          const fields = ["x", "volume", "resp0"];
                          const headers = fields.map((field) => document.querySelector(`#lineBarTableGrid .tabulator-col[tabulator-field="${field}"] .tabulator-col-title`)?.textContent.trim() || field);
                          const rows = [...document.querySelectorAll("#lineBarTableGrid .tabulator-row.tabulator-selected:not(.tabulator-calcs)")];
                          const body = rows.map((row) => fields.map((field) => row.querySelector(`.tabulator-cell[tabulator-field="${field}"]`)?.textContent.trim() || "").join(","));
                          return [headers.join(","), ...body].join("\\n");
                        }
                        """
                    )
                    page.locator("#lineBarTableGrid").click(button="right")
                    page.locator("#lineBarTableContextMenu:not([hidden])").get_by_text("Copy selected rows to clipboard").wait_for(timeout=10_000)
                    page.locator("#lineBarTableContextMenu:not([hidden])").get_by_text("Copy selected rows to clipboard").click()
                    page.wait_for_function(
                        "(expected) => window.__lucidumCopiedText === expected",
                        arg=selected_rows_csv,
                        timeout=10_000,
                    )

                    full_table_csv = page.evaluate(
                        """
                        () => {
                          const fields = ["x", "volume", "resp0"];
                          const headers = fields.map((field) => document.querySelector(`#lineBarTableGrid .tabulator-col[tabulator-field="${field}"] .tabulator-col-title`)?.textContent.trim() || field);
                          const rows = [...document.querySelectorAll("#lineBarTableGrid .tabulator-row:not(.tabulator-calcs)")];
                          const body = rows.map((row) => fields.map((field) => row.querySelector(`.tabulator-cell[tabulator-field="${field}"]`)?.textContent.trim() || "").join(","));
                          const footer = fields.map((field) => document.querySelector(`#lineBarTableGrid .tabulator-row.tabulator-calcs .tabulator-cell[tabulator-field="${field}"]`)?.textContent.trim() || "").join(",");
                          return [headers.join(","), ...body, footer].join("\\n");
                        }
                        """
                    )
                    page.locator("#lineBarTableGrid").click(button="right")
                    page.locator("#lineBarTableContextMenu:not([hidden])").get_by_text("Copy table to clipboard").click()
                    page.wait_for_function(
                        "(expected) => window.__lucidumCopiedText === expected",
                        arg=full_table_csv,
                        timeout=10_000,
                    )
                    table_copy_context = page.evaluate(
                        """
                        () => [
                          document.querySelector("#lineBarGroupMeta")?.textContent.trim() || "",
                          document.querySelector("#lineBarFilter")?.textContent.trim() || "",
                        ].filter(Boolean).join("\\n")
                        """
                    )
                    page.locator("#lineBarCopyBtn").click()
                    page.wait_for_function(
                        "(expected) => window.__lucidumCopiedText === expected",
                        arg=f"{table_copy_context}\n\n{full_table_csv}",
                        timeout=10_000,
                    )

                    page.locator("#lineBarTableGrid").click(button="right")
                    page.locator("#lineBarTableContextMenu:not([hidden])").get_by_text("Clear selection").wait_for(timeout=10_000)
                    self.assertEqual(page.locator("#lineBarTableContextMenu:not([hidden]) [role='separator']").count(), 1)
                    page.locator("#lineBarTableContextMenu:not([hidden])").get_by_text("Clear selection").click()
                    page.wait_for_function(
                        """
                        () => document.querySelectorAll("#lineBarTableGrid .tabulator-row.tabulator-selected:not(.tabulator-calcs)").length === 0
                        """,
                        timeout=10_000,
                    )
                    self.assertEqual(page_errors, [])
                    browser.close()
            finally:
                server.should_exit = True
                thread.join(timeout=5)

    @unittest.skipUnless(RUN_BROWSER_TESTS, "set PY_LUCIDUM_RUN_BROWSER_TESTS=1 to run browser smoke tests")
    @unittest.skipUnless(sync_playwright is not None, "playwright is not installed")
    def test_line_bar_table_weighted_shows_row_count_column(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            data_path = Path(tmp_dir) / "weighted_line_bar_table.csv"
            data_path.write_text(
                "MAKE,theft_claims_count,earned_exposure_in_period\n"
                "ALFA ROMEO,100,1\n"
                "ALFA ROMEO,200,2\n"
                "BMW,300,10\n",
                encoding="utf-8",
            )
            base_url, server, thread = self.start_app(
                data_path,
                defaults={
                    "x": "MAKE",
                    "actual": "theft_claims_count",
                    "denominator": "earned_exposure_in_period",
                },
                tools=["line_bar"],
            )
            try:
                assert sync_playwright is not None
                with sync_playwright() as playwright:
                    browser = playwright.chromium.launch()
                    page = browser.new_page(viewport={"width": 1280, "height": 800})
                    page_errors: list[str] = []
                    page.on("pageerror", lambda error: page_errors.append(str(error)))

                    page.goto(base_url, wait_until="domcontentloaded")
                    page.locator("#datasetMeta").get_by_text("weighted_line_bar_table.csv").wait_for(timeout=10_000)
                    page.locator("#tableTab").click()
                    page.locator("#tableWrap:not(.hidden) #lineBarTableGrid").wait_for(timeout=10_000)
                    page.wait_for_function(
                        """
                        () => {
                          const headers = [...document.querySelectorAll("#lineBarTableGrid .tabulator-col[tabulator-field] .tabulator-col-title")]
                            .map((cell) => cell.textContent.trim());
                          return headers.join("|") === "MAKE|Row count|earned_exposure_in_period|theft_claims_count";
                        }
                        """,
                        timeout=10_000,
                    )
                    table_state = page.evaluate(
                        """
                        () => {
                          const fields = ["x", "row_count", "volume", "resp0"];
                          const cellText = (selector) => document.querySelector(selector)?.textContent.trim() || "";
                          const headers = [...document.querySelectorAll("#lineBarTableGrid .tabulator-col[tabulator-field] .tabulator-col-title")]
                            .map((cell) => {
                              const style = getComputedStyle(cell);
                              return {
                                text: cell.textContent.trim(),
                                textOverflow: style.textOverflow,
                                whiteSpace: style.whiteSpace,
                              };
                            });
                          const gridStyle = getComputedStyle(document.querySelector("#lineBarTableGrid"));
                          const rows = [...document.querySelectorAll("#lineBarTableGrid .tabulator-row:not(.tabulator-calcs)")].map((row) => (
                            fields.map((field) => row.querySelector(`.tabulator-cell[tabulator-field="${field}"]`)?.textContent.trim() || "")
                          )).filter((row) => row[0]);
                          const footer = fields.map((field) => cellText(`#lineBarTableGrid .tabulator-row.tabulator-calcs .tabulator-cell[tabulator-field="${field}"]`));
                          const dataRows = [...document.querySelectorAll("#lineBarTableGrid .tabulator-tableholder .tabulator-row:not(.tabulator-calcs)")].filter((row) => (
                            row.querySelector(".tabulator-cell[tabulator-field='x']")?.textContent.trim()
                          ));
                          const finalDataRowRect = dataRows.at(-1)?.getBoundingClientRect();
                          const totalRowRect = document.querySelector("#lineBarTableGrid .tabulator-footer .tabulator-row.tabulator-calcs")?.getBoundingClientRect();
                          return {
                            headers,
                            rows,
                            footer,
                            gridBorderLeftWidth: gridStyle.borderLeftWidth,
                            gridBorderLeftStyle: gridStyle.borderLeftStyle,
                            gridBorderTopWidth: gridStyle.borderTopWidth,
                            gridBorderTopStyle: gridStyle.borderTopStyle,
                            totalGap: finalDataRowRect && totalRowRect ? totalRowRect.top - finalDataRowRect.bottom : null,
                          };
                        }
                        """
                    )
                    self.assertEqual(
                        [header["text"] for header in table_state["headers"]],
                        ["MAKE", "Row count", "earned_exposure_in_period", "theft_claims_count"],
                    )
                    for header in table_state["headers"]:
                        self.assertNotIn("...", header["text"])
                        self.assertNotIn("…", header["text"])
                        self.assertEqual(header["textOverflow"], "clip")
                        self.assertEqual(header["whiteSpace"], "normal")
                    self.assertEqual(table_state["gridBorderLeftWidth"], "1px")
                    self.assertEqual(table_state["gridBorderLeftStyle"], "solid")
                    self.assertEqual(table_state["gridBorderTopWidth"], "1px")
                    self.assertEqual(table_state["gridBorderTopStyle"], "solid")
                    self.assertIsNotNone(table_state["totalGap"])
                    self.assertGreaterEqual(table_state["totalGap"], -1)
                    self.assertLessEqual(table_state["totalGap"], 6)
                    self.assertEqual(table_state["rows"], [
                        ["ALFA ROMEO", "2", "3", "100.00"],
                        ["BMW", "1", "10", "30.00"],
                    ])
                    self.assertEqual(table_state["footer"], ["Total", "3", "13", "46.15"])
                    self.assertEqual(page_errors, [])
                    browser.close()
            finally:
                server.should_exit = True
                thread.join(timeout=5)

    @unittest.skipUnless(RUN_BROWSER_TESTS, "set PY_LUCIDUM_RUN_BROWSER_TESTS=1 to run browser smoke tests")
    @unittest.skipUnless(sync_playwright is not None, "playwright is not installed")
    def test_line_bar_dense_categorical_iso_date_labels_rotate(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            data_path = Path(tmp_dir) / "dense_quote_dates.parquet"
            selects = [
                f"SELECT {sql_literal(f'2026-06-{day:02d}')}::VARCHAR AS quote_date, 1 AS sold"
                for day in range(6, 29)
            ]
            con = duckdb.connect(database=":memory:")
            try:
                con.execute(
                    f"""
COPY (
  {" UNION ALL ".join(selects)}
) TO {sql_literal(str(data_path))} (FORMAT PARQUET)
"""
                )
            finally:
                con.close()
            base_url, server, thread = self.start_app(
                data_path,
                defaults={
                    "x": "quote_date",
                    "actual": "sold",
                    "denominator": "__none__",
                },
                tools=["line_bar"],
            )
            try:
                assert sync_playwright is not None
                with sync_playwright() as playwright:
                    browser = playwright.chromium.launch()
                    page = browser.new_page(viewport={"width": 1280, "height": 800})
                    page.emulate_media(color_scheme="light")
                    page_errors: list[str] = []
                    page.on("pageerror", lambda error: page_errors.append(str(error)))

                    page.goto(base_url, wait_until="domcontentloaded")
                    page.locator("#datasetMeta").get_by_text("dense_quote_dates.parquet").wait_for(timeout=10_000)
                    page.locator("#chart:not(.hidden)").wait_for(timeout=10_000)
                    page.wait_for_function(
                        """
                        () => document.querySelector("#lineBarGroupMeta")?.textContent.includes("23 groups")
                        """,
                        timeout=10_000,
                    )
                    axis_state = page.evaluate(
                        """
                        () => {
                          const chart = window.echarts?.getInstanceByDom(document.querySelector("#chart"));
                          const axis = chart?.getOption?.()?.xAxis?.[0] || {};
                          return {
                            labels: axis.data || [],
                            rotate: axis.axisLabel?.rotate,
                          };
                        }
                        """
                    )
                    self.assertEqual(axis_state["labels"][0], "2026-06-06")
                    self.assertEqual(axis_state["labels"][-1], "2026-06-28")
                    self.assertEqual(len(axis_state["labels"]), 23)
                    self.assertEqual(axis_state["rotate"], 65)
                    self.assertEqual(page_errors, [])
                    browser.close()
            finally:
                server.should_exit = True
                thread.join(timeout=5)

    @unittest.skipUnless(RUN_BROWSER_TESTS, "set PY_LUCIDUM_RUN_BROWSER_TESTS=1 to run browser smoke tests")
    @unittest.skipUnless(sync_playwright is not None, "playwright is not installed")
    def test_line_bar_quantile_ranges_and_band_restore(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            data_path = Path(tmp_dir) / "line_bar_quantiles.csv"
            data_path.write_text(
                "Score,Other,Actual\n"
                "0.123456789,10,10\n"
                "0.234567891,20,20\n"
                "0.345678912,30,30\n"
                "0.456789123,40,40\n",
                encoding="utf-8",
            )
            base_url, server, thread = self.start_app(
                data_path,
                defaults={
                    "x": "Score",
                    "actual": "Actual",
                    "denominator": "__none__",
                },
                tools=["line_bar"],
            )
            try:
                assert sync_playwright is not None
                with sync_playwright() as playwright:
                    browser = playwright.chromium.launch()
                    page = browser.new_page(viewport={"width": 1280, "height": 800})
                    page.emulate_media(color_scheme="light")
                    page_errors: list[str] = []
                    page.on("pageerror", lambda error: page_errors.append(str(error)))

                    page.goto(base_url, wait_until="domcontentloaded")
                    page.locator("#datasetMeta").get_by_text("line_bar_quantiles.csv").wait_for(timeout=10_000)
                    page.locator("#chart:not(.hidden)").wait_for(timeout=10_000)
                    page.wait_for_function(
                        """
                        () => document.querySelector("#lineBarGroupMeta")?.textContent.includes("4 groups")
                        """,
                        timeout=10_000,
                    )
                    page.locator("#lineBarSideControlsToggleBtn").click()
                    page.locator("#lineBarToolbarToggleBtn").click()
                    page.wait_for_function(
                        """
                        () => document.querySelector("#lineBarSideControlsToggleBtn")?.getAttribute("aria-expanded") === "true"
                          && document.querySelector("#lineBarToolbarToggleBtn")?.getAttribute("aria-expanded") === "true"
                          && getComputedStyle(document.querySelector("#chartSideControls")).display !== "none"
                          && getComputedStyle(document.querySelector("#lineBarToolbar")).display !== "none"
                        """,
                        timeout=10_000,
                    )
                    page.locator('#bandControl [data-value="5"]').click()
                    page.wait_for_function(
                        """
                        () => document.querySelector("#bandValue")?.textContent.trim() === "(5)"
                        """,
                        timeout=10_000,
                    )
                    page.locator('#quantileControl [data-value="quantile"]').click()
                    page.locator('#bandControl [data-value="1"]').click()
                    page.locator('#bandControl [data-action="band-up"]').click()
                    page.wait_for_function(
                        """
                        () => document.querySelector("#bandValue")?.textContent.trim() === "(2)"
                        """,
                        timeout=10_000,
                    )
                    page.wait_for_function(
                        """
                        () => {
                          const chart = window.echarts?.getInstanceByDom(document.querySelector("#chart"));
                          const labels = chart?.getOption()?.xAxis?.[0]?.data || [];
                          return labels.includes("Q1\\n0.1235 to 0.2346") && labels.includes("Q2\\n0.3457 to 0.4568");
                        }
                        """,
                        timeout=10_000,
                    )
                    page.locator('#featureList .feature[data-value="Other"]').click()
                    page.wait_for_function(
                        """
                        () => {
                          const chart = window.echarts?.getInstanceByDom(document.querySelector("#chart"));
                          const labels = chart?.getOption()?.xAxis?.[0]?.data || [];
                          return document.querySelector("#bandValue")?.textContent.trim() === "(2)"
                            && labels.includes("Q1\\n10 to 20")
                            && labels.includes("Q2\\n30 to 40");
                        }
                        """,
                        timeout=10_000,
                    )
                    page.locator('#featureList .feature[data-value="Score"]').click()
                    page.wait_for_function(
                        """
                        () => {
                          const chart = window.echarts?.getInstanceByDom(document.querySelector("#chart"));
                          const labels = chart?.getOption()?.xAxis?.[0]?.data || [];
                          return document.querySelector("#bandValue")?.textContent.trim() === "(2)"
                            && labels.includes("Q1\\n0.1235 to 0.2346")
                            && labels.includes("Q2\\n0.3457 to 0.4568");
                        }
                        """,
                        timeout=10_000,
                    )

                    page.locator("#tableTab").click()
                    page.wait_for_function(
                        """
                        () => {
                          const cells = [...document.querySelectorAll("#lineBarTableGrid .tabulator-row:not(.tabulator-calcs) .tabulator-cell[tabulator-field='x']")]
                            .map((cell) => cell.textContent.trim());
                          return cells.includes("Q1: 0.1235 to 0.2346") && cells.includes("Q2: 0.3457 to 0.4568");
                        }
                        """,
                        timeout=10_000,
                    )
                    page.locator('#quantileControl [data-value="off"]').click()
                    page.wait_for_function(
                        """
                        () => document.querySelector("#bandValue")?.textContent.trim() === "(5)"
                          && document.querySelector('#bandControl [data-value="5"]')?.classList.contains("active")
                        """,
                        timeout=10_000,
                    )
                    self.assertEqual(page_errors, [])
                    browser.close()
            finally:
                server.should_exit = True
                thread.join(timeout=5)

    @unittest.skipUnless(RUN_BROWSER_TESTS, "set PY_LUCIDUM_RUN_BROWSER_TESTS=1 to run browser smoke tests")
    @unittest.skipUnless(sync_playwright is not None, "playwright is not installed")
    def test_line_bar_table_search_reaches_beyond_chart_cap(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            data_path = Path(tmp_dir) / "many_line_bar_groups.csv"
            lines = ["Category,Actual"]
            lines.extend(f"G{index:05d},{index}" for index in range(10005))
            data_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
            base_url, server, thread = self.start_app(
                data_path,
                defaults={
                    "x": "Category",
                    "actual": "Actual",
                    "denominator": "__none__",
                },
                tools=["line_bar"],
            )
            try:
                assert sync_playwright is not None
                with sync_playwright() as playwright:
                    browser = playwright.chromium.launch()
                    page = browser.new_page(viewport={"width": 1280, "height": 800})
                    page_errors: list[str] = []
                    chart_requests = 0
                    table_requests = 0
                    page.on("pageerror", lambda error: page_errors.append(str(error)))

                    def count_request(request: object) -> None:
                        nonlocal chart_requests, table_requests
                        if request.url.endswith("/api/chart"):
                            chart_requests += 1
                        elif request.url.endswith("/api/line-bar/table"):
                            table_requests += 1

                    page.on("request", count_request)
                    page.goto(base_url, wait_until="domcontentloaded")
                    page.locator("#datasetMeta").get_by_text("many_line_bar_groups.csv").wait_for(timeout=10_000)
                    page.wait_for_function(
                        """
                        () => document.querySelector("#lineBarGroupMeta")?.textContent.includes("10,005 groups")
                          && document.querySelector("#chartMessage")?.textContent.includes("Use Table view to inspect all groups")
                        """,
                        timeout=20_000,
                    )
                    chart_requests_before_search = chart_requests
                    page.locator("#tableTab").click()
                    page.locator("#tableWrap:not(.hidden) #lineBarTableSearch").wait_for(timeout=10_000)
                    page.wait_for_function(
                        """
                        () => document.querySelector("#tableWrap .table-pagination")?.textContent.includes("1-10,000 of 10,005 groups")
                          && document.querySelector("#lineBarTableGrid .tabulator-tableholder")
                        """,
                        timeout=20_000,
                    )
                    virtual_state = page.evaluate(
                        """
                        () => ({
                          renderedRows: document.querySelectorAll("#lineBarTableGrid .tabulator-row:not(.tabulator-calcs)").length,
                          holderClientHeight: document.querySelector("#lineBarTableGrid .tabulator-tableholder")?.clientHeight || 0,
                          holderScrollHeight: document.querySelector("#lineBarTableGrid .tabulator-tableholder")?.scrollHeight || 0,
                          pager: document.querySelector("#tableWrap .table-pagination")?.textContent || "",
                        })
                        """
                    )
                    self.assertLess(virtual_state["renderedRows"], 500)
                    self.assertGreater(virtual_state["holderScrollHeight"], virtual_state["holderClientHeight"])
                    self.assertIn("1-10,000 of 10,005 groups", virtual_state["pager"])
                    chart_requests_before_sort = chart_requests
                    if page.locator("#lineBarToolbarToggleBtn").get_attribute("aria-expanded") == "false":
                        page.locator("#lineBarToolbarToggleBtn").click()
                        page.wait_for_function(
                            """
                            () => document.querySelector("#lineBarToolbarToggleBtn")?.getAttribute("aria-expanded") === "true"
                              && getComputedStyle(document.querySelector("#lineBarToolbar")).display !== "none"
                            """,
                            timeout=10_000,
                        )
                    with page.expect_response(lambda response: response.url.endswith("/api/line-bar/table") and response.status == 200, timeout=10_000):
                        page.locator('.segmented[data-control="sort"] button[data-value="volume"]').click()
                    self.assertEqual(chart_requests, chart_requests_before_sort)
                    page.locator("#lineBarTableGrid .tabulator-tableholder").wait_for(timeout=10_000)
                    page.evaluate(
                        """
                        () => {
                          const holder = document.querySelector("#lineBarTableGrid .tabulator-tableholder");
                          holder.scrollTop = holder.scrollHeight;
                          holder.dispatchEvent(new Event("scroll"));
                        }
                        """
                    )
                    page.wait_for_function(
                        """
                        () => [...document.querySelectorAll("#lineBarTableGrid .tabulator-row:not(.tabulator-calcs) .tabulator-cell[tabulator-field='x']")]
                          .some((cell) => cell.textContent.trim() === "G09999")
                        """,
                        timeout=20_000,
                    )
                    table_requests_before_table_search = table_requests
                    with page.expect_response(lambda response: response.url.endswith("/api/line-bar/table") and response.status == 200, timeout=10_000):
                        page.locator("#lineBarTableSearch").fill("g10004")
                    page.wait_for_function(
                        """
                        () => {
                            const xCells = [...document.querySelectorAll("#lineBarTableGrid .tabulator-row:not(.tabulator-calcs) .tabulator-cell[tabulator-field='x']")]
                              .map((cell) => cell.textContent.trim())
                              .filter(Boolean);
                            return xCells.length === 1
                              && xCells[0] === "G10004"
                              && document.querySelector("#tableWrap .table-pagination")?.textContent.includes("1-1 of 1 groups");
                        }
                        """,
                        timeout=20_000,
                    )
                    page.wait_for_timeout(100)
                    self.assertEqual(chart_requests, chart_requests_before_search)
                    self.assertGreater(table_requests, table_requests_before_table_search)
                    self.assertGreaterEqual(table_requests, 2)
                    self.assertEqual(page_errors, [])
                    browser.close()
            finally:
                server.should_exit = True
                thread.join(timeout=5)

    @unittest.skipUnless(RUN_BROWSER_TESTS, "set PY_LUCIDUM_RUN_BROWSER_TESTS=1 to run browser smoke tests")
    @unittest.skipUnless(sync_playwright is not None, "playwright is not installed")
    def test_specs_tool_uses_full_workspace_width(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            data_path = tmp_path / "sample.csv"
            data_path.write_text(
                "PostcodeArea,PostcodeSector,vehicle_age,price,value,PostcodeUnit,lat,long\n"
                "AB,AB10 1,1,100,10,AB10 1AA,57.1,-2.1\n"
                "AB,AB10 1,2,200,20,AB10 1AB,57.2,-2.2\n"
                "AL,AL1 1,3,300,30,AL1 1AA,51.8,-0.3\n",
                encoding="utf-8",
            )
            features_path = tmp_path / "feature_spec.csv"
            extra_feature_rows = "".join(
                f"FutureFeature{i},REFERENCE,,,,,\n"
                for i in range(1, 80)
            )
            features_path.write_text(
                "Feature,Grouping,Base,min,max,banding,scenario1\n"
                "vehicle_age,VEHICLE,1,0,10,1,feature\n"
                "PostcodeArea,POSTCODE,AB,,,,feature\n"
                "FutureFeature,REFERENCE,,,,,feature\n"
                + extra_feature_rows,
                encoding="utf-8",
            )
            kpis_path = tmp_path / "kpi_spec.csv"
            kpis_path.write_text(
                "group,name,actual,denominator,decimals,format\n"
                "PRICE,Price,price,Average row value,2,currency\n"
                "VALUE,Value,value,Average row value,0,number\n"
                "COUNT,Count,price,N,0,number\n",
                encoding="utf-8",
            )
            filters_path = tmp_path / "filter_spec.csv"
            filters_path.write_text(
                "theme,name,expression\n"
                "POSTCODE,AB,PostcodeArea = 'AB'\n"
                "VEHICLE,Young vehicle_age,vehicle_age < 3\n"
                "POSTCODE,Broken auto,AutoMissingColumn = 1\n",
                encoding="utf-8",
            )
            base_url, server, thread = self.start_app(
                data_path,
                tools=["line_bar", "histogram", "glm", "gbm", "specs"],
                filters_path=filters_path,
                use_saved_filters=True,
                kpis_path=kpis_path,
                use_kpis=True,
                features_path=features_path,
            )
            try:
                self.exercise_specs_tool_layout(base_url)
            finally:
                server.should_exit = True
                thread.join(timeout=5)

    @unittest.skipUnless(RUN_BROWSER_TESTS, "set PY_LUCIDUM_RUN_BROWSER_TESTS=1 to run browser smoke tests")
    @unittest.skipUnless(sync_playwright is not None, "playwright is not installed")
    def test_specs_tool_shows_generated_starters_without_saving_placeholders(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            data_path = tmp_path / "sample.csv"
            data_path.write_text(
                "vehicle_age,price,value\n"
                "1,100,10\n"
                "2,200,20\n",
                encoding="utf-8",
            )
            kpis_path = tmp_path / "specs" / "kpi_spec.csv"
            base_url, server, thread = self.start_app(data_path, tools=["specs"], kpis_path=kpis_path, use_kpis=False, use_features=False)
            try:
                assert sync_playwright is not None
                with sync_playwright() as playwright:
                    browser = playwright.chromium.launch()
                    page = browser.new_page(viewport={"width": 1280, "height": 800})
                    page_errors: list[str] = []
                    page.on("pageerror", lambda error: page_errors.append(str(error)))
                    page.goto(base_url, wait_until="domcontentloaded")
                    page.locator("#datasetMeta").get_by_text("sample.csv").wait_for(timeout=10_000)

                    def save_button_state() -> dict[str, bool]:
                        return page.locator("#specSaveBtn").evaluate(
                            """
                            node => ({
                              disabled: node.disabled,
                              dirty: node.classList.contains("dirty"),
                              pending: node.classList.contains("pending"),
                            })
                            """
                        )

                    def wait_for_save_button_state(expected: dict[str, bool]) -> None:
                        page.wait_for_function(
                            """
                            expected => {
                              const button = document.querySelector("#specSaveBtn");
                              return button
                                && button.disabled === expected.disabled
                                && button.classList.contains("dirty") === expected.dirty
                                && button.classList.contains("pending") === expected.pending;
                            }
                            """,
                            arg=expected,
                            timeout=10_000,
                        )
                        self.assertEqual(save_button_state(), expected)

                    def edit_kpi_cell(field: str, value: str) -> None:
                        page.locator(f'#specGrid .tabulator-row .tabulator-cell[tabulator-field="{field}"]').first.dblclick()
                        page.locator("#specGrid .tabulator-cell.tabulator-editing input").fill(value)
                        page.keyboard.press("Enter")

                    page.locator("#specFilePath", has_text="Save target:").wait_for(timeout=10_000)
                    page.locator("#specFilePath", has_text=re.compile(r"\((new file|existing file ignored by --no-features)\)")).wait_for(timeout=10_000)
                    page.locator("#specNotice", has_text="Valid feature spec").wait_for(timeout=10_000)
                    self.assertEqual(
                        page.locator('#specGrid .tabulator-row .tabulator-cell[tabulator-field="Feature"]').first.inner_text().strip(),
                        "vehicle_age",
                    )
                    self.assertFalse(kpis_path.exists())
                    page.locator('[data-spec-kind="kpi"]').click()
                    page.locator("#specFilePath", has_text="kpi_spec.csv (new file)").wait_for(timeout=10_000)
                    page.locator("#specNotice", has_text="Valid KPI spec").wait_for(timeout=10_000)
                    page.locator(".spec-cell-placeholder", has_text="Numeric column").wait_for(timeout=10_000)
                    wait_for_save_button_state({"disabled": False, "dirty": False, "pending": True})
                    page.locator("#specSaveBtn").click()
                    page.locator("#specNotice", has_text="KPI spec saved").wait_for(timeout=10_000)
                    wait_for_save_button_state({"disabled": True, "dirty": False, "pending": False})
                    edit_kpi_cell("group", "PRICE")
                    edit_kpi_cell("name", "Price")
                    edit_kpi_cell("actual", "price")
                    edit_kpi_cell("decimals", "2")
                    edit_kpi_cell("format", "currency")
                    page.wait_for_function(
                        """
                        () => {
                          const button = document.querySelector("#specSaveBtn");
                          return button && !button.disabled && button.classList.contains("dirty") && !button.classList.contains("pending");
                        }
                        """,
                        timeout=10_000,
                    )
                    page.locator("#specSaveBtn").click()
                    wait_for_save_button_state({"disabled": True, "dirty": False, "pending": False})
                    self.assertEqual(page_errors, [])
                    browser.close()
            finally:
                server.should_exit = True
                thread.join(timeout=5)

            saved_text = kpis_path.read_text(encoding="utf-8")
            self.assertIn("group,name,actual,denominator,decimals,format", saved_text)
            self.assertNotIn("Numeric column", saved_text)
            self.assertNotIn("number, currency, or percent", saved_text)

    @unittest.skipUnless(RUN_BROWSER_TESTS, "set PY_LUCIDUM_RUN_BROWSER_TESTS=1 to run browser smoke tests")
    @unittest.skipUnless(sync_playwright is not None, "playwright is not installed")
    def test_spec_save_preserves_open_filter_and_kpi_sidebar_groups(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            data_path = tmp_path / "sample.csv"
            data_path.write_text(
                "vehicle_age,price,value\n"
                "1,100,10\n"
                "2,200,20\n"
                "3,300,30\n",
                encoding="utf-8",
            )
            filters_path = tmp_path / "filter_spec.csv"
            filters_path.write_text(
                "theme,name,expression\n"
                "AGE,Young,vehicle_age < 3\n",
                encoding="utf-8",
            )
            kpis_path = tmp_path / "kpi_spec.csv"
            kpis_path.write_text(
                "group,name,actual,denominator,decimals,format\n"
                "PRICE,Price,price,N,0,number\n"
                "VALUE,Value,value,N,0,number\n",
                encoding="utf-8",
            )
            base_url, server, thread = self.start_app(
                data_path,
                tools=["line-bar", "specs"],
                filters_path=filters_path,
                use_saved_filters=True,
                kpis_path=kpis_path,
                use_kpis=True,
            )
            try:
                assert sync_playwright is not None
                with sync_playwright() as playwright:
                    browser = playwright.chromium.launch()
                    page = browser.new_page(viewport={"width": 1280, "height": 800})
                    page_errors: list[str] = []
                    page.on("pageerror", lambda error: page_errors.append(str(error)))
                    page.goto(base_url, wait_until="domcontentloaded")
                    page.locator("#datasetMeta").get_by_text("sample.csv").wait_for(timeout=10_000)

                    page.locator("#filterCollapseBtn").click()
                    page.locator('.saved-filter-theme[data-filter-theme="AGE"]').click()
                    page.wait_for_function(
                        """
                        () => document.querySelector("#filterCollapseBtn")?.getAttribute("aria-expanded") === "true"
                          && document.querySelector('.saved-filter-theme[data-filter-theme="AGE"]')?.getAttribute("aria-expanded") === "true"
                          && !document.querySelector('.saved-filter-option[data-filter-theme="AGE"]')?.hidden
                        """,
                        timeout=10_000,
                    )

                    page.locator("#specsTool:not(.hidden)").click()
                    page.locator('[data-spec-kind="filter"]').click()
                    page.locator('[data-spec-kind="filter"][aria-selected="true"]').wait_for(timeout=10_000)
                    page.locator('#specGrid .tabulator-row .tabulator-cell[tabulator-field="name"]').first.dblclick()
                    page.locator("#specGrid .tabulator-cell.tabulator-editing input").fill("Young updated")
                    page.keyboard.press("Enter")
                    page.locator("#specSaveBtn").click()
                    page.locator("#specNotice", has_text="Filter spec saved").wait_for(timeout=10_000)
                    page.wait_for_function(
                        """
                        () => document.querySelector("#filterCollapseBtn")?.getAttribute("aria-expanded") === "true"
                          && document.querySelector('.saved-filter-theme[data-filter-theme="AGE"]')?.getAttribute("aria-expanded") === "true"
                          && Boolean(document.querySelector('.saved-filter-option[data-filter-name="Young updated"]:not([hidden])'))
                        """,
                        timeout=10_000,
                    )

                    page.locator("#kpiCollapseBtn").click()
                    page.locator('.kpi-theme[data-kpi-group="VALUE"]').click()
                    page.wait_for_function(
                        """
                        () => document.querySelector("#kpiCollapseBtn")?.getAttribute("aria-expanded") === "true"
                          && document.querySelector('.kpi-theme[data-kpi-group="VALUE"]')?.getAttribute("aria-expanded") === "true"
                          && !document.querySelector('.kpi-option[data-kpi-group="VALUE"]')?.hidden
                        """,
                        timeout=10_000,
                    )

                    page.locator('[data-spec-kind="kpi"]').click()
                    page.locator('[data-spec-kind="kpi"][aria-selected="true"]').wait_for(timeout=10_000)
                    page.locator("#specGrid .tabulator-row").nth(1).locator('.tabulator-cell[tabulator-field="name"]').dblclick()
                    page.locator("#specGrid .tabulator-cell.tabulator-editing input").fill("Value updated")
                    page.keyboard.press("Enter")
                    page.locator("#specSaveBtn").click()
                    page.locator("#specNotice", has_text="KPI spec saved").wait_for(timeout=10_000)
                    page.wait_for_function(
                        """
                        () => document.querySelector("#kpiCollapseBtn")?.getAttribute("aria-expanded") === "true"
                          && document.querySelector('.kpi-theme[data-kpi-group="VALUE"]')?.getAttribute("aria-expanded") === "true"
                          && Boolean([...document.querySelectorAll('.kpi-option[data-kpi-group="VALUE"]:not([hidden])')]
                            .some((node) => node.textContent.includes("Value updated")))
                        """,
                        timeout=10_000,
                    )

                    self.assertEqual(page_errors, [])
                    browser.close()
            finally:
                server.should_exit = True
                thread.join(timeout=5)

    @unittest.skipUnless(RUN_BROWSER_TESTS, "set PY_LUCIDUM_RUN_BROWSER_TESTS=1 to run browser smoke tests")
    @unittest.skipUnless(sync_playwright is not None, "playwright is not installed")
    def test_gbm_init_score_long_dropdown_stays_in_viewport(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            data_path = tmp_path / "many_init_score_columns.csv"
            numeric_columns = [f"score_{index:03d}" for index in range(70)]
            headers = ["actualNumerator", "denominator", *numeric_columns, "SAMPLE"]
            rows = [",".join(headers)]
            for row_index, sample in enumerate(("training", "test", "training"), start=1):
                values = [
                    str(row_index * 10),
                    str(row_index * 100),
                    *[str(row_index * 1000 + column_index) for column_index, _ in enumerate(numeric_columns)],
                    sample,
                ]
                rows.append(",".join(values))
            data_path.write_text("\n".join(rows) + "\n", encoding="utf-8")
            base_url, server, thread = self.start_app(
                data_path,
                defaults={"x": "score_000", "actual": "actualNumerator", "denominator": "denominator"},
                tools=["line_bar", "gbm"],
                use_features=False,
                use_kpis=False,
            )
            try:
                assert sync_playwright is not None
                with sync_playwright() as playwright:
                    browser = playwright.chromium.launch()
                    page = browser.new_page(viewport={"width": 1280, "height": 800})
                    page_errors: list[str] = []
                    page.on("pageerror", lambda error: page_errors.append(str(error)))
                    try:
                        page.goto(f"{base_url}/?tool=gbm", wait_until="domcontentloaded")
                        page.get_by_text("Features and parameters").wait_for(timeout=10_000)
                        page.wait_for_function(
                            """
                            () => [...document.querySelectorAll("#gbmParameterGrid .tabulator-row")]
                              .some((row) => row.querySelector(".tabulator-cell[tabulator-field='name']")?.textContent.trim() === "init_score")
                            """,
                            timeout=10_000,
                        )
                        page.locator("#gbmParameterGrid .tabulator-row", has_text="init_score").locator(".tabulator-cell[tabulator-field='value']").click()
                        page.wait_for_function(
                            """
                            () => {
                              const popup = document.querySelector(".tabulator-popup-container.tabulator-edit-list");
                              const rect = popup?.getBoundingClientRect();
                              return Boolean(popup && rect.width > 0 && rect.height > 0);
                            }
                            """,
                            timeout=10_000,
                        )
                        dropdown_state = page.evaluate(
                            """
                            () => {
                              const row = [...document.querySelectorAll("#gbmParameterGrid .tabulator-row")]
                                .find((item) => item.querySelector(".tabulator-cell[tabulator-field='name']")?.textContent.trim() === "init_score");
                              const cell = row?.querySelector(".tabulator-cell[tabulator-field='value']");
                              const input = cell?.querySelector("input.gbm-parameter-list-editor");
                              const popup = document.querySelector(".tabulator-popup-container.tabulator-edit-list");
                              const inputRect = input?.getBoundingClientRect();
                              const popupRect = popup?.getBoundingClientRect();
                              return {
                                editing: Boolean(cell?.classList.contains("tabulator-editing")),
                                inputValue: input?.value || "",
                                input: inputRect ? { top: inputRect.top, bottom: inputRect.bottom } : null,
                                popup: popupRect ? { top: popupRect.top, bottom: popupRect.bottom, height: popupRect.height } : null,
                                viewportHeight: window.innerHeight,
                                popupClientHeight: popup?.clientHeight || 0,
                                popupScrollHeight: popup?.scrollHeight || 0,
                                popupText: popup?.textContent || "",
                                groupTexts: [...(popup?.querySelectorAll(".tabulator-edit-list-group") || [])].map((node) => node.textContent.trim()),
                                itemTexts: [...(popup?.querySelectorAll(".tabulator-edit-list-item") || [])].map((node) => node.textContent.trim()),
                              };
                            }
                            """
                        )
                        self.assertTrue(dropdown_state["editing"])
                        self.assertEqual(dropdown_state["inputValue"], "none")
                        self.assertIsNotNone(dropdown_state["input"])
                        self.assertIsNotNone(dropdown_state["popup"])
                        popup = dropdown_state["popup"]
                        input_rect = dropdown_state["input"]
                        assert popup is not None
                        assert input_rect is not None
                        self.assertGreaterEqual(popup["top"], 0)
                        self.assertLessEqual(popup["bottom"], dropdown_state["viewportHeight"])
                        self.assertGreaterEqual(popup["top"], input_rect["bottom"] - 1)
                        self.assertLessEqual(popup["height"], 362)
                        self.assertGreater(dropdown_state["popupScrollHeight"], dropdown_state["popupClientHeight"] + 1)
                        self.assertIn("DATASET COLUMNS", dropdown_state["groupTexts"])
                        self.assertIn("score_059", dropdown_state["itemTexts"])
                        page.locator("#gbmParameterGrid input.gbm-parameter-list-editor").fill("score_059")
                        page.wait_for_function(
                            """
                            () => document.querySelector("#gbmParameterGrid input.gbm-parameter-list-editor")?.value === "score_059"
                              && document.querySelector(".tabulator-popup-container.tabulator-edit-list")?.textContent.includes("score_059")
                            """,
                            timeout=10_000,
                        )
                        self.assertEqual(page_errors, [])
                    finally:
                        browser.close()
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
                "actualNumerator,denominator,Age,Segment,BadText,PostcodeArea,PostcodeSector,PostcodeUnit,lat,long,SAMPLE\n"
                "10,100,30,A,bad,AB,AB10 1,AB10 1AA,57.1,-2.1,training\n"
                "20,200,40,B,bad,AB,AB10 1,AB10 1AB,57.2,-2.2,test\n"
                "30,300,50,C,bad,CD20 2,CD20 2AA,CD20 2AA,56.1,-1.1,training\n",
                encoding="utf-8",
            )
            kpis_path = tmp_path / "kpi_spec.csv"
            kpis_path.write_text(
                "group,name,actual,denominator,decimals,format\n"
                "MODEL,Actual numerator,actualNumerator,denominator,2,number\n",
                encoding="utf-8",
            )
            features_path = tmp_path / "feature_spec.csv"
            features_path.write_text(
                "Feature,Grouping,scenario1\n"
                "Age,DRIVER,feature\n"
                "Segment,VEHICLE,feature\n",
                encoding="utf-8",
            )
            store = GbmModelStore(data_path)
            for model_id, label, learning_rate, created_at in (
                ("browser-smoke-model", "Browser smoke model", 0.11, "2026-05-25T00:00:00Z"),
                ("browser-smoke-model-2", "Second smoke model", 0.22, "2026-05-25T00:00:01Z"),
                ("browser-smoke-delete-a", "Disposable smoke model A", 0.09, "2026-05-24T00:00:00Z"),
                ("browser-smoke-delete-b", "Disposable smoke model B", 0.08, "2026-05-24T00:00:01Z"),
            ):
                model_dir = store.create_model_dir(model_id)
                store.write_json(
                    model_dir / "manifest.json",
                    {
                        "model_id": model_id,
                        "label": label,
                        "created_at": created_at,
                        "training_mode": "ebm" if model_id.endswith("-2") else "normal",
                        "response_column": "actualNumerator",
                        "offset_column": "denominator",
                        "best_iteration": 3,
                        "training_rows": 2,
                        "test_rows": 1,
                        "scored_rows": 3,
                        "sample_column": "SAMPLE",
                        "sample_source": "dataset",
                        "timings": {"training_seconds": 1.234 if model_id == "browser-smoke-model" else 62.0},
                        "feature_scenario": (
                            {"name": "scenario1", "features": ["Age", "Segment"]}
                            if model_id.endswith("-2")
                            else {"name": "old_scenario", "features": ["Age"]}
                        ),
                        "feature_interaction_constraints": (
                            {"groupings": ["DRIVER"], "groups": [{"grouping": "DRIVER", "features": ["Age"]}]}
                            if model_id.endswith("-2")
                            else (
                                {"mode": "pairs", "pairs": [{"left": "Age", "right": "Segment"}]}
                                if model_id == "browser-smoke-delete-a"
                                else {"groupings": ["OLD"], "groups": [{"grouping": "OLD", "features": ["Age"]}]}
                            )
                        ),
                    },
                )
                feature_config = [{"name": "Age", "kind": "integer", "include": True, "monotonicity": "Increasing", "gain": 5.0}]
                if model_id == "browser-smoke-model":
                    feature_config.append({"name": "lat", "kind": "numeric", "include": True, "monotonicity": "", "gain": 4.0})
                if model_id.endswith("-2") or model_id == "browser-smoke-delete-a":
                    feature_config.append({"name": "Segment", "kind": "categorical", "include": True, "monotonicity": "", "gain": 6.0})
                write_gbm_feature_config(store, model_id, feature_config)
                store.write_json(
                    model_dir / "parameters.json",
                    {
                        "objective": "gamma",
                        "metric": "gamma",
                        "learning_rate": learning_rate,
                        "num_iterations": 123 if model_id.endswith("-2") else 77,
                        "early_stopping_rounds": 25,
                    },
                )
                if model_id == "browser-smoke-model":
                    training_eval = [7.38, 7.33, 7.31, 7.305, 7.301]
                    test_eval = [7.37, 7.325, 7.3022, 7.303, 7.304]
                else:
                    training_eval = [round(0.17 + 0.52 / ((index + 1) ** 0.58), 6) for index in range(3000)]
                    test_eval = [round(0.18 + 0.5 / ((index + 1) ** 0.55), 6) for index in range(3000)]
                write_gbm_evaluation(
                    store,
                    model_id,
                    {
                        "training": {"gamma": training_eval},
                        "test": {"gamma": test_eval},
                    },
                )
                con = duckdb.connect(database=":memory:")
                try:
                    con.execute(
                        f"""
COPY (
  SELECT 0 AS tree_index, 1 AS node_depth, '0-S0' AS node_index, '0-L0' AS left_child, '0-S1' AS right_child,
         NULL AS parent_index, 'Age' AS split_feature, 5.0 AS split_gain, '35' AS threshold,
         NULL AS threshold_label, '<=' AS decision_type, 'left' AS missing_direction, 'None' AS missing_type,
         1.0 AS value, 3.0 AS weight, 3 AS count
  UNION ALL
  SELECT 0, 2, '0-L0', NULL, NULL, '0-S0', NULL, NULL, NULL, NULL, NULL, NULL, NULL, 0.8, 1.0, 1
  UNION ALL
  SELECT 0, 2, '0-S1', '0-L1', '0-L2', '0-S0', 'Segment', 2.0, '0||2', 'A / C', '==', 'right', 'None', 1.2, 2.0, 2
  UNION ALL
  SELECT 0, 3, '0-L1', NULL, NULL, '0-S1', NULL, NULL, NULL, NULL, NULL, NULL, NULL, 1.1, 1.0, 1
  UNION ALL
  SELECT 0, 3, '0-L2', NULL, NULL, '0-S1', NULL, NULL, NULL, NULL, NULL, NULL, NULL, 1.4, 1.0, 1
) TO {sql_literal(str(model_dir / "tree_table.parquet"))} (FORMAT PARQUET)
"""
                    )
                    con.execute(
                        f"""
COPY (
  SELECT 1 AS __lucidum_row_id, 0.20 AS Age, 0.10 AS Segment, 0.40 AS lat
  UNION ALL
  SELECT 2, -0.30, -0.15, -0.20
  UNION ALL
  SELECT 3, 0.05, 0.25, 0.10
) TO {sql_literal(str(model_dir / "shap_values.parquet"))} (FORMAT PARQUET)
"""
                    )
                    con.execute(
                        f"""
COPY (
  SELECT 'Age' AS feature, 0.183 AS mean_abs_shap, -0.017 AS mean_shap, 3 AS row_count
  UNION ALL
  SELECT 'Segment', 0.167, 0.067, 3
  UNION ALL
  SELECT 'lat', 0.233, 0.100, 3
) TO {sql_literal(str(model_dir / "shap_summary.parquet"))} (FORMAT PARQUET)
"""
                    )
                    if model_id == "browser-smoke-model":
                        con.execute(
                            f"""
COPY (
  SELECT 1 AS __lucidum_row_id, 0.11 AS gbm_prediction
  UNION ALL
  SELECT 2, 0.21
  UNION ALL
  SELECT 3, 0.31
) TO {sql_literal(str(model_dir / "predictions.parquet"))} (FORMAT PARQUET)
"""
                        )
                finally:
                    con.close()
            store.activate_model("browser-smoke-model")
            original_probe = Dataset.probe_column_readable

            def fake_probe(dataset: Dataset, column: Any) -> None:
                if column.name == "BadText":
                    raise duckdb.InvalidInputException(
                        'Invalid Input Error: Invalid string encoding found in Parquet file "/tmp/bad.parquet": '
                        'value "bad" is not valid UTF8!'
                    )
                original_probe(dataset, column)

            with patch.object(Dataset, "probe_column_readable", fake_probe):
                base_url, server, thread = self.start_app(
                    data_path,
                    tools=["line_bar", "gbm"],
                    kpis_path=kpis_path,
                    use_kpis=True,
                    features_path=features_path,
                )
                try:
                    self.exercise_gbm_tool(base_url)
                finally:
                    server.should_exit = True
                    thread.join(timeout=5)

    @unittest.skipUnless(RUN_BROWSER_TESTS, "set PY_LUCIDUM_RUN_BROWSER_TESTS=1 to run browser smoke tests")
    @unittest.skipUnless(sync_playwright is not None, "playwright is not installed")
    def test_gbm_sidebar_switch_preserves_profile_but_refreshes_model_chart(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            data_path = Path(tmp_dir) / "sample.csv"
            data_path.write_text(
                "actualNumerator,denominator,Age,Segment,SAMPLE\n"
                "10,100,30,A,training\n"
                "20,200,40,B,test\n"
                "30,300,50,C,training\n",
                encoding="utf-8",
            )
            features_path = Path(tmp_dir) / "feature_spec.csv"
            features_path.write_text(
                "Feature,Grouping,Base,scenario1\n"
                "Age,DRIVER,40,feature\n"
                "Segment,VEHICLE,B,feature\n",
                encoding="utf-8",
            )
            store = GbmModelStore(data_path)
            self.write_gbm_prediction_model(
                store,
                "browser-smoke-model",
                "Browser smoke model",
                "2026-05-25T00:00:00Z",
                [0.11, 0.21, 0.31],
            )
            self.write_gbm_tabulation_artifacts(store, "browser-smoke-model", offset=0.2, tabulated=False)
            self.write_gbm_prediction_model(
                store,
                "browser-smoke-model-2",
                "Second smoke model",
                "2026-05-25T00:00:01Z",
                [0.41, 0.51, 0.61],
            )
            self.write_gbm_tabulation_artifacts(store, "browser-smoke-model-2", offset=0.4, blocked=True)
            self.write_tabulated_prediction_sidecar(
                store.artifact_path("browser-smoke-model-2", "tabulated_predictions"),
                "gbm_tabulated_prediction",
                [0.42, 0.52, 0.62],
            )
            store.activate_model("browser-smoke-model")
            glm_store = GlmModelStore(data_path)
            self.write_glm_prediction_model(
                glm_store,
                "browser-smoke-glm",
                "Browser smoke GLM",
                "2026-05-25T00:00:02Z",
                [0.15, 0.25, 0.35],
                formula="actualNumerator ~ 1 + Age + Segment",
                family="tweedie",
                family_parameter=1.5,
                training_scope="all",
                regularization={"mode": "none"},
            )
            self.write_glm_tabulation_artifacts(glm_store, "browser-smoke-glm", include_segment=True, offset=0.0)
            self.write_tabulated_prediction_sidecar(
                glm_store.artifact_path("browser-smoke-glm", "tabulated_predictions"),
                "glm_tabulated_prediction",
                [0.16, 0.26, 0.36],
            )
            self.write_glm_prediction_model(
                glm_store,
                "browser-smoke-glm-2",
                "Second smoke GLM",
                "2026-05-25T00:00:03Z",
                [0.45, 0.55, 0.65],
                formula="actualNumerator ~ 1 + Age",
                family="normal",
                family_parameter=None,
                training_scope="training",
                regularization={"mode": "manual", "l1_ratio": 1, "alpha": "0.07"},
            )
            self.write_glm_tabulation_artifacts(glm_store, "browser-smoke-glm-2", offset=0.2)
            self.write_glm_prediction_model(
                glm_store,
                "browser-smoke-glm-delete-a",
                "Disposable smoke GLM A",
                "2026-05-25T00:00:04Z",
                [0.18, 0.28, 0.38],
            )
            self.write_glm_prediction_model(
                glm_store,
                "browser-smoke-glm-delete-b",
                "Disposable smoke GLM B",
                "2026-05-25T00:00:05Z",
                [0.19, 0.29, 0.39],
            )
            glm_store.activate_model("browser-smoke-glm")
            base_url, server, thread = self.start_app(data_path, tools=["column_profile", "line_bar", "glm", "gbm"], features_path=features_path)
            try:
                self.exercise_gbm_profile_cache_and_model_chart_refresh(base_url)
            finally:
                server.should_exit = True
                thread.join(timeout=5)

    @unittest.skipUnless(RUN_BROWSER_TESTS, "set PY_LUCIDUM_RUN_BROWSER_TESTS=1 to run browser smoke tests")
    @unittest.skipUnless(sync_playwright is not None, "playwright is not installed")
    def test_gbm_model_navigator_preserves_line_bar_x_feature(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            data_path = Path(tmp_dir) / "sample.csv"
            data_path.write_text(
                "actualNumerator,denominator,Age,Segment,SAMPLE\n"
                "10,100,30,A,training\n"
                "20,200,40,B,test\n"
                "30,300,50,C,training\n",
                encoding="utf-8",
            )
            store = GbmModelStore(data_path)
            self.write_gbm_prediction_model(
                store,
                "browser-smoke-model",
                "Browser smoke model",
                "2026-05-25T00:00:00Z",
                [0.11, 0.21, 0.31],
            )
            self.write_gbm_prediction_model(
                store,
                "browser-smoke-model-2",
                "Second smoke model",
                "2026-05-25T00:00:01Z",
                [0.41, 0.51, 0.61],
            )
            store.activate_model("browser-smoke-model")
            base_url, server, thread = self.start_app(data_path, tools=["line_bar", "gbm"])
            try:
                assert sync_playwright is not None
                with sync_playwright() as playwright:
                    browser = playwright.chromium.launch()
                    page = browser.new_page(viewport={"width": 1280, "height": 800})
                    page_errors: list[str] = []
                    page.on("pageerror", lambda error: page_errors.append(str(error)))
                    try:
                        chart_url = (
                            f"{base_url}/?tool=line_bar&source=gbm%3Abrowser-smoke-model%3Apredictions"
                            "&x=Segment&actual=actualNumerator&denominator=denominator"
                        )
                        page.goto(chart_url, wait_until="domcontentloaded")
                        page.locator("#chart:not(.hidden)").wait_for(timeout=10_000)
                        page.wait_for_function(
                            '() => document.querySelector("#lineBarGroupMeta")?.textContent.includes("groups")',
                            timeout=10_000,
                        )
                        page.locator("#lineBarSideControlsToggleBtn").click()
                        page.wait_for_function(
                            """
                            () => document.querySelector("#lineBarSideControlsToggleBtn")?.getAttribute("aria-expanded") === "true"
                              && getComputedStyle(document.querySelector("#chartSideControls")).display !== "none"
                            """,
                            timeout=10_000,
                        )
                        page.locator("#featureList .feature.active", has_text="Segment").wait_for(timeout=10_000)
                        page.locator("#gbmTool").click()
                        page.get_by_role("button", name="Model navigator").click()
                        page.locator("#gbmModelGrid .tabulator-row", has_text="Second smoke model").click()
                        page.locator("#gbmActivateModelBtn").click()
                        page.wait_for_function(
                            """
                            () => document.querySelector("#gbmModelSelectedMeta")?.textContent.includes("Second smoke model")
                              && document.querySelector("#gbmModelGrid .gbm-model-active-dot")?.closest(".tabulator-row")?.textContent.includes("Second smoke model")
                            """,
                            timeout=10_000,
                        )
                        with page.expect_request(lambda request: request.url.endswith("/api/chart"), timeout=10_000) as chart_request_info:
                            page.locator("#lineBarTool").click()
                        request_body = json.loads(chart_request_info.value.post_data or "{}")
                        self.assertEqual(request_body["x"], "Segment")
                        page.locator("#featureList .feature.active", has_text="Segment").wait_for(timeout=10_000)
                        self.assertEqual(page_errors, [])
                    finally:
                        browser.close()
            finally:
                server.should_exit = True
                thread.join(timeout=5)

    @unittest.skipUnless(RUN_BROWSER_TESTS, "set PY_LUCIDUM_RUN_BROWSER_TESTS=1 to run browser smoke tests")
    @unittest.skipUnless(sync_playwright is not None, "playwright is not installed")
    def test_sidebar_accordion_works_across_tools(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            data_path = Path(tmp_dir) / "sample.csv"
            data_path.write_text(
                "actualNumerator,denominator,Age,Segment,PostcodeArea,PostcodeSector,PostcodeUnit,lat,long,price,value\n"
                "10,100,30,A,AB,AB10 1,AB10 1AA,57.1,-2.1,100,10\n"
                "20,200,40,B,AB,AB10 1,AB10 1AB,57.2,-2.2,200,20\n"
                "30,300,50,C,AL,AL1 1,AL1 1AA,51.8,-0.3,300,30\n"
                "40,400,60,D,AL,AL1 2,AL1 2AA,51.7,-0.2,400,40\n",
                encoding="utf-8",
            )
            store = GbmModelStore(data_path)
            self.write_gbm_prediction_model(
                store,
                "sidebar-accordion-model",
                "Sidebar accordion model",
                "2026-05-26T00:00:00Z",
                [0.12, 0.23, 0.34, 0.45],
            )
            store.activate_model("sidebar-accordion-model")
            base_url, server, thread = self.start_app(data_path, tools=["column_profile", "line_bar", "uk_map", "glm", "gbm"])
            try:
                self.exercise_sidebar_accordion(base_url)
            finally:
                server.should_exit = True
                thread.join(timeout=5)

    @unittest.skipUnless(RUN_BROWSER_TESTS, "set PY_LUCIDUM_RUN_BROWSER_TESTS=1 to run browser smoke tests")
    @unittest.skipUnless(sync_playwright is not None, "playwright is not installed")
    def test_default_sidebar_hides_model_accordions(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            data_path = Path(tmp_dir) / "sample.csv"
            data_path.write_text(
                "PostcodeArea,PostcodeSector,PostcodeUnit,lat,long,vehicle_age,price,value\n"
                "AB,AB10 1,AB10 1AA,57.1,-2.1,1,100,10\n"
                "AL,AL1 1,AL1 1AA,51.8,-0.3,2,200,20\n",
                encoding="utf-8",
            )
            base_url, server, thread = self.start_app(data_path)
            try:
                self.exercise_default_sidebar_hides_model_accordions(base_url)
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
            base_url, server, thread = self.start_app(data_path, buttons=True)
            try:
                self.exercise_stopped_overlay(base_url)
            finally:
                server.should_exit = True
                thread.join(timeout=5)

    @unittest.skipUnless(RUN_BROWSER_TESTS, "set PY_LUCIDUM_RUN_BROWSER_TESTS=1 to run browser smoke tests")
    @unittest.skipUnless(sync_playwright is not None, "playwright is not installed")
    @unittest.skipUnless(importlib.util.find_spec("glum") is not None, "glum is not installed")
    def test_glm_tabulation_rebase_smoke(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            data_path = Path(tmp_dir) / "sample.csv"
            data_path.write_text(
                "actualNumerator,Age,Segment,SAMPLE\n"
                "10,30,A,training\n"
                "20,40,B,test\n"
                "30,50,A,training\n"
                "45,55,B,training\n"
                "60,60,A,test\n"
                "75,65,B,training\n",
                encoding="utf-8",
            )
            features_path = Path(tmp_dir) / "feature_spec.csv"
            features_path.write_text(
                "Feature,Grouping,Base,min,max,banding,scenario1\n"
                "Age,DRIVER,40,30,70,5,feature\n"
                "Segment,DRIVER,A,,,,feature\n",
                encoding="utf-8",
            )
            dataset = Dataset(data_path)
            store = GlmModelStore(data_path)
            result = train_model(
                dataset,
                store,
                {
                    "label": "Browser rebase GLM",
                    "formula": "Age * C(Segment)",
                    "response_column": "actualNumerator",
                    "family": "normal",
                    "training_scope": "all",
                    "regularization": {"mode": "manual", "alpha": 0.1, "l1_ratio": 0.0},
                },
            )
            build_tabulations(
                dataset,
                store,
                {"model_ids": [result["model_id"]]},
                {
                    "rows": [
                        {"feature": "Age", "grouping": "DRIVER", "base": "40", "min": "30", "max": "70", "banding": "5"},
                        {"feature": "Segment", "grouping": "DRIVER", "base": "A"},
                    ]
                },
            )
            base_url, server, thread = self.start_app(
                data_path,
                tools=["line_bar", "column_profile", "glm"],
                features_path=features_path,
                defaults={"x": "Age", "actual": "actualNumerator", "denominator": "__none__"},
            )
            try:
                self.exercise_glm_tabulation_rebase(base_url)
            finally:
                server.should_exit = True
                thread.join(timeout=5)
                stop_persistent_glm_fit_worker()

    @unittest.skipUnless(RUN_BROWSER_TESTS, "set PY_LUCIDUM_RUN_BROWSER_TESTS=1 to run browser smoke tests")
    @unittest.skipUnless(sync_playwright is not None, "playwright is not installed")
    def test_dataset_viewer_preview_transpose_keeps_stop_app_responsive(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            data_path = Path(tmp_dir) / "preview.csv"
            rows = ["PostcodeArea,PostcodeSector,vehicle_age,price,value,PostcodeUnit,lat,long"]
            for index in range(1, 1001):
                rows.append(f"AB,AB10 1,{index % 99},{100 + index},{10 + index},AB10 1AA,57.1,-2.1")
            data_path.write_text("\n".join(rows) + "\n", encoding="utf-8")
            base_url, server, thread = self.start_app(data_path, buttons=True)
            try:
                self.exercise_dataset_viewer_large_transpose(base_url)
            finally:
                server.should_exit = True
                thread.join(timeout=5)
                stop_persistent_glm_fit_worker()

    @unittest.skipUnless(RUN_BROWSER_TESTS, "set PY_LUCIDUM_RUN_BROWSER_TESTS=1 to run browser smoke tests")
    @unittest.skipUnless(sync_playwright is not None, "playwright is not installed")
    def test_dataset_viewer_favourites_save_and_restore_dataset_view(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            data_path = root / "sample.csv"
            data_path.write_text(
                "vehicle_age,segment,price,value,postcode\n"
                "1,A,100,10,AB10 1AA\n"
                "2,A,200,20,AB10 1AB\n"
                "3,B,300,30,AL1 1AA\n"
                "4,B,400,40,AL1 2AA\n",
                encoding="utf-8",
            )
            filters_path = root / "filter_spec.csv"
            filters_path.write_text(
                "theme,name,expression\n"
                "AGE,Older,vehicle_age >= 3\n",
                encoding="utf-8",
            )
            favourites_path = root / "config" / "favourites.json"
            base_url, server, thread = self.start_app(
                data_path,
                filters_path=filters_path,
                use_saved_filters=True,
                line_bar_favourites_path=favourites_path,
                tools=["dataset_viewer", "line_bar"],
            )
            try:
                assert sync_playwright is not None

                def resize_tabulator_column(page: Any, column_selector: str, delta: int = 48) -> dict[str, float]:
                    probe = page.evaluate(
                        """
                        ({ selector }) => {
                          const column = document.querySelector(selector);
                          if (!column) return null;
                          const handles = [...column.parentElement?.querySelectorAll(".tabulator-col-resize-handle") || []];
                          const columnRect = column.getBoundingClientRect();
                          const handle = handles.find((candidate) => Math.abs(candidate.getBoundingClientRect().left - columnRect.right) <= 8) || null;
                          if (!handle) return null;
                          const handleRect = handle.getBoundingClientRect();
                          return {
                            before: columnRect.width,
                            x: handleRect.left + handleRect.width / 2,
                            y: handleRect.top + handleRect.height / 2,
                          };
                        }
                        """,
                        arg={"selector": column_selector},
                    )
                    self.assertIsNotNone(probe)
                    assert probe is not None
                    page.mouse.move(probe["x"], probe["y"])
                    page.mouse.down()
                    page.mouse.move(probe["x"] + delta, probe["y"], steps=6)
                    page.mouse.up()
                    page.wait_for_function(
                        """
                        ({ selector, minimum }) => {
                          const column = document.querySelector(selector);
                          return Boolean(column && column.getBoundingClientRect().width >= minimum);
                        }
                        """,
                        arg={"selector": column_selector, "minimum": probe["before"] + 16},
                        timeout=10_000,
                    )
                    after = page.evaluate(
                        "selector => document.querySelector(selector)?.getBoundingClientRect().width || 0",
                        arg=column_selector,
                    )
                    return {"before": float(probe["before"]), "after": float(after)}

                with sync_playwright() as playwright:
                    browser = playwright.chromium.launch()
                    page = browser.new_page(viewport={"width": 1280, "height": 800})
                    page_errors: list[str] = []
                    page.on("pageerror", lambda error: page_errors.append(str(error)))
                    page.goto(base_url, wait_until="domcontentloaded")
                    page.locator("#datasetViewerTool.active").wait_for(timeout=10_000)
                    page.locator("#datasetViewerWrap:not(.hidden) #datasetViewerGrid .tabulator-row").first.wait_for(timeout=10_000)

                    if page.locator("#filterCollapseBtn").get_attribute("aria-expanded") == "false":
                        page.locator("#filterCollapseBtn").click()
                    age_heading = page.locator('.saved-filter-theme[data-filter-theme="AGE"]')
                    if age_heading.get_attribute("aria-expanded") == "false":
                        age_heading.click()
                    page.locator('.saved-filter-option[data-filter-theme="AGE"]').click()
                    page.wait_for_function("""() => document.querySelector("#filterInput")?.value === "vehicle_age >= 3" """, timeout=10_000)

                    page.locator("#datasetViewerAlphabeticalColumns").check()
                    page.locator('#datasetViewerGrid .tabulator-col[tabulator-field="c4"]').click(button="right")
                    page.locator("#datasetViewerCellContextMenu:not([hidden])").get_by_role("menuitem", name="Pin column").click()
                    page.locator('#datasetViewerGrid .tabulator-col[tabulator-field="c0"]').click(button="right")
                    page.locator("#datasetViewerCellContextMenu:not([hidden])").get_by_role("menuitem", name="Pin column").click()
                    page.wait_for_function(
                        """
                        () => document.querySelector("#datasetViewerCount")?.textContent.includes("postcode, vehicle_age pinned")
                        """,
                        timeout=10_000,
                    )
                    page.locator("#datasetViewerSearch").fill("price, post")
                    page.wait_for_function(
                        """
                        () => {
                          const headers = [...document.querySelectorAll('#datasetViewerGrid .tabulator-col')]
                            .filter((cell) => cell.offsetParent !== null)
                            .map((cell) => cell.getAttribute('tabulator-field'))
                            .filter(Boolean);
                          return headers.join(",") === "c4,c0,c2";
                        }
                        """,
                        timeout=10_000,
                    )
                    page.locator('#datasetViewerGrid .tabulator-col[tabulator-field="c2"] .tabulator-col-sorter').click()
                    page.locator('#datasetViewerGrid .tabulator-col[tabulator-field="c2"] .tabulator-col-sorter').click()
                    page.wait_for_function(
                        """
                        () => document.querySelector('#datasetViewerGrid .tabulator-row .tabulator-cell[tabulator-field="c2"]')?.textContent.trim() === "400"
                        """,
                        timeout=10_000,
                    )
                    normal_width = resize_tabulator_column(page, '#datasetViewerGrid .tabulator-col[tabulator-field="c2"]', 52)
                    page.locator("#datasetViewerTranspose").check()
                    page.wait_for_function(
                        """
                        () => {
                          const grid = document.querySelector('#datasetViewerGrid.dataset-viewer-grid-transposed');
                          const rows = [...grid?.querySelectorAll('.tabulator-row') || []].filter((row) => row.offsetParent !== null);
                          const names = rows.map((row) => {
                            const cell = row.querySelector('.tabulator-cell[tabulator-field="__field"]');
                            return (cell?.querySelector('.dataset-viewer-pinned-field-text') || cell)?.textContent.trim();
                          }).filter(Boolean);
                          return names.slice(0, 3).join(",") === "postcode,vehicle_age,price";
                        }
                        """,
                        timeout=10_000,
                    )
                    transposed_width = resize_tabulator_column(page, '#datasetViewerGrid .tabulator-col[tabulator-field="__field"]', 44)
                    page.locator('#datasetViewerGrid .tabulator-col[tabulator-field="__field"] .dataset-viewer-transposed-sort-button').click()
                    page.wait_for_function(
                        """
                        () => document.querySelector('#datasetViewerGrid .tabulator-col[tabulator-field="__field"] .dataset-viewer-transposed-sort-button')?.dataset.sortDir === "asc"
                        """,
                        timeout=10_000,
                    )

                    if page.locator("#favouritesCollapseBtn").get_attribute("aria-expanded") == "false":
                        page.locator("#favouritesCollapseBtn").click()
                    self.click_sidebar_favourite_action(page, "#sidebarFavouriteAddBtn")
                    page.locator("#sidebarFavouritePopover:not([hidden])").wait_for(timeout=10_000)
                    page.wait_for_function(
                        """() => document.querySelector('input[name="sidebarFavouriteScope"][value="dataset_view"]')?.checked
                          && [...document.querySelectorAll(".sidebar-favourite-scope-option span")]
                            .map((node) => node.textContent.trim()).join("|") === "Dataset view" """,
                        timeout=10_000,
                    )
                    page.locator("#sidebarFavouriteNameInput").fill("Dataset favourite")
                    page.locator('[data-favourite-action="save-add"]').click()
                    page.wait_for_function(
                        """() => [...document.querySelectorAll(".saved-favourite-option")]
                          .some((button) => button.querySelector(".saved-filter-name")?.textContent.trim() === "Dataset favourite"
                            && button.querySelector(".favourite-detail")?.textContent.trim() === "Dataset view"
                            && button.classList.contains("active")) """,
                        timeout=10_000,
                    )
                    saved_payload = json.loads(favourites_path.read_text(encoding="utf-8"))
                    saved_view = saved_payload["favourites"][0]["view"]
                    self.assertEqual(saved_view["scope"], "dataset_view")
                    self.assertEqual(saved_view["filter"], "vehicle_age >= 3")
                    self.assertEqual(saved_view["filterSelectionMode"], "grouped")
                    self.assertEqual(saved_view["savedFilterRows"][0]["name"], "Older")
                    dataset_view = saved_view["datasetView"]
                    self.assertTrue(dataset_view["transpose"])
                    self.assertTrue(dataset_view["alphabeticalColumns"])
                    self.assertEqual(dataset_view["selectColumns"], "price, post")
                    self.assertEqual(dataset_view["pinnedColumns"], ["postcode", "vehicle_age"])
                    self.assertGreaterEqual(dataset_view["columnWidths"]["normal"]["price"], normal_width["after"] - 2)
                    self.assertGreaterEqual(dataset_view["columnWidths"]["transposed"]["__field"], transposed_width["after"] - 2)
                    self.assertEqual(dataset_view["sort"]["normal"], [{"column": "price", "dir": "desc"}])
                    self.assertEqual(dataset_view["sort"]["transposed"], {"field": "__field", "dir": "asc"})

                    page.locator('#datasetViewerGrid .tabulator-col[tabulator-field="__field"] .dataset-viewer-transposed-sort-button').click()
                    page.locator('#datasetViewerGrid .tabulator-col[tabulator-field="__field"] .dataset-viewer-transposed-sort-button').click()
                    page.wait_for_function(
                        """
                        () => document.querySelector('#datasetViewerGrid .tabulator-col[tabulator-field="__field"] .dataset-viewer-transposed-sort-button')?.dataset.sortDir === "none"
                        """,
                        timeout=10_000,
                    )
                    page.locator("#datasetViewerTranspose").uncheck()
                    page.wait_for_function(
                        """
                        () => Boolean(document.querySelector('#datasetViewerGrid:not(.dataset-viewer-grid-transposed) .tabulator-row .tabulator-cell[tabulator-field="c2"]'))
                        """,
                        timeout=10_000,
                    )
                    page.locator('#datasetViewerGrid .tabulator-col[tabulator-field="c2"] .tabulator-col-sorter').click()
                    page.wait_for_function(
                        """
                        () => document.querySelector('#datasetViewerGrid .tabulator-row .tabulator-cell[tabulator-field="c2"]')?.textContent.trim() === "300"
                        """,
                        timeout=10_000,
                    )
                    page.locator("#datasetViewerSearchClear").click()
                    page.locator("#datasetViewerAlphabeticalColumns").uncheck()
                    page.locator("#filterRowClearBtn").click()
                    page.locator("#lineBarTool").click()
                    page.locator("#lineBarTool.active").wait_for(timeout=10_000)
                    page.locator('.saved-favourite-option[data-favourite-scope="dataset_view"]').click()
                    page.wait_for_function(
                        """
                        () => {
                          const grid = document.querySelector('#datasetViewerGrid.dataset-viewer-grid-transposed');
                          const rows = [...grid?.querySelectorAll('.tabulator-row') || []].filter((row) => row.offsetParent !== null);
                          const names = rows.map((row) => {
                            const cell = row.querySelector('.tabulator-cell[tabulator-field="__field"]');
                            return (cell?.querySelector('.dataset-viewer-pinned-field-text') || cell)?.textContent.trim();
                          }).filter(Boolean);
                          return document.querySelector("#datasetViewerTool")?.classList.contains("active")
                            && document.querySelector("#filterInput")?.value === "vehicle_age >= 3"
                            && document.querySelector("#datasetViewerSearch")?.value === "price, post"
                            && document.querySelector("#datasetViewerTranspose")?.checked
                            && document.querySelector("#datasetViewerAlphabeticalColumns")?.checked
                            && document.querySelector('#datasetViewerGrid .tabulator-col[tabulator-field="__field"] .dataset-viewer-transposed-sort-button')?.dataset.sortDir === "asc"
                            && names.slice(0, 3).join(",") === "postcode,vehicle_age,price";
                        }
                        """,
                        timeout=10_000,
                    )
                    restored_transposed_width = page.evaluate(
                        """() => document.querySelector('#datasetViewerGrid .tabulator-col[tabulator-field="__field"]')?.getBoundingClientRect().width || 0"""
                    )
                    self.assertGreaterEqual(float(restored_transposed_width), transposed_width["after"] - 2)
                    page.locator("#datasetViewerTranspose").uncheck()
                    page.wait_for_function(
                        """
                        () => {
                          const grid = document.querySelector('#datasetViewerGrid:not(.dataset-viewer-grid-transposed)');
                          const headers = [...grid?.querySelectorAll('.tabulator-col') || []]
                            .filter((cell) => cell.offsetParent !== null)
                            .map((cell) => cell.getAttribute('tabulator-field'))
                            .filter(Boolean);
                          const priceWidth = grid?.querySelector('.tabulator-col[tabulator-field="c2"]')?.getBoundingClientRect().width || 0;
                          const firstPrice = grid?.querySelector('.tabulator-row .tabulator-cell[tabulator-field="c2"]')?.textContent.trim();
                          return headers.slice(0, 3).join(",") === "c4,c0,c2" && priceWidth > 0 && firstPrice === "400";
                        }
                        """,
                        timeout=10_000,
                    )
                    restored_normal_width = page.evaluate(
                        """() => document.querySelector('#datasetViewerGrid .tabulator-col[tabulator-field="c2"]')?.getBoundingClientRect().width || 0"""
                    )
                    self.assertGreaterEqual(float(restored_normal_width), normal_width["after"] - 2)

                    startup_page = browser.new_page(viewport={"width": 1280, "height": 800})
                    startup_errors: list[str] = []
                    startup_page.on("pageerror", lambda error: startup_errors.append(str(error)))
                    startup_page.goto(f"{base_url}?line_bar_favourite=Dataset%20favourite", wait_until="domcontentloaded")
                    startup_page.wait_for_function(
                        """
                        () => document.querySelector("#datasetViewerTool")?.classList.contains("active")
                          && document.querySelector("#datasetViewerTranspose")?.checked
                          && document.querySelector("#datasetViewerSearch")?.value === "price, post"
                          && document.querySelector("#filterInput")?.value === "vehicle_age >= 3"
                          && document.querySelector('#datasetViewerGrid .tabulator-col[tabulator-field="__field"] .dataset-viewer-transposed-sort-button')?.dataset.sortDir === "asc"
                        """,
                        timeout=10_000,
                    )
                    self.assertEqual(startup_errors, [])
                    startup_page.close()
                    self.assertEqual(page_errors, [])
                    browser.close()
            finally:
                server.should_exit = True
                thread.join(timeout=5)
                stop_persistent_glm_fit_worker()

    @unittest.skipUnless(RUN_BROWSER_TESTS, "set PY_LUCIDUM_RUN_BROWSER_TESTS=1 to run browser smoke tests")
    @unittest.skipUnless(sync_playwright is not None, "playwright is not installed")
    def test_line_bar_favourites_ui_flow(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            data_path = root / "sample.csv"
            data_path.write_text(
                "vehicle_age,segment,price,expected,value\n"
                "1,A,100,90,10\n"
                "2,A,200,210,20\n"
                "3,B,300,290,30\n"
                "4,B,400,410,40\n",
                encoding="utf-8",
            )
            filters_path = root / "filter_spec.csv"
            filters_path.write_text(
                "theme,name,expression\n"
                "AGE,Older,vehicle_age >= 3\n",
                encoding="utf-8",
            )
            kpis_path = root / "kpi_spec.csv"
            kpis_path.write_text(
                "group,name,actual,denominator,decimals,format\n"
                "PRICE,Weighted price,price,value,1,number\n",
                encoding="utf-8",
            )
            favourites_path = root / "config" / "favourites.json"
            base_url, server, thread = self.start_app(
                data_path,
                filters_path=filters_path,
                use_saved_filters=True,
                kpis_path=kpis_path,
                use_kpis=True,
                line_bar_favourites_path=favourites_path,
                defaults={"x": "vehicle_age"},
                tools=["line_bar"],
            )
            try:
                self.exercise_line_bar_favourites(base_url)
                self.assertTrue(favourites_path.exists())
                self.assertFalse((root / ".lucidum").exists())
            finally:
                server.should_exit = True
                thread.join(timeout=5)
                stop_persistent_glm_fit_worker()

    @unittest.skipUnless(RUN_BROWSER_TESTS, "set PY_LUCIDUM_RUN_BROWSER_TESTS=1 to run browser smoke tests")
    @unittest.skipUnless(sync_playwright is not None, "playwright is not installed")
    def test_histogram_favourites_save_and_restore_histogram_view(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            data_path = root / "sample.csv"
            data_path.write_text(
                "vehicle_age,segment,price,value\n"
                "1,A,100,10\n"
                "2,A,200,20\n"
                "3,B,300,30\n"
                "4,B,400,40\n",
                encoding="utf-8",
            )
            filters_path = root / "filter_spec.csv"
            filters_path.write_text(
                "theme,name,expression\n"
                "SEG,B segment,segment = 'B'\n",
                encoding="utf-8",
            )
            favourites_path = root / "config" / "favourites.json"
            base_url, server, thread = self.start_app(
                data_path,
                filters_path=filters_path,
                use_saved_filters=True,
                line_bar_favourites_path=favourites_path,
                defaults={"actual": "price", "denominator": "value"},
                tools=["histogram", "line_bar"],
            )
            try:
                assert sync_playwright is not None
                with sync_playwright() as playwright:
                    browser = playwright.chromium.launch()
                    page = browser.new_page(viewport={"width": 1280, "height": 800})
                    page_errors: list[str] = []
                    page.on("pageerror", lambda error: page_errors.append(str(error)))
                    page.goto(base_url, wait_until="domcontentloaded")
                    page.wait_for_function(
                        """() => document.querySelector("#histogramTool")?.classList.contains("active")
                          && (document.querySelector("#histogramGroupMeta")?.textContent || "").includes("bins")""",
                        timeout=10_000,
                    )

                    if page.locator("#favouritesCollapseBtn").get_attribute("aria-expanded") == "false":
                        page.locator("#favouritesCollapseBtn").click()
                    page.locator("#filterCollapseBtn").click()
                    segment_heading = page.locator('.saved-filter-theme[data-filter-theme="SEG"]')
                    if segment_heading.get_attribute("aria-expanded") == "false":
                        segment_heading.click()
                    page.locator('.saved-filter-option[data-filter-theme="SEG"]').click()
                    page.wait_for_function("""() => document.querySelector("#filterInput")?.value === "segment = 'B'" """, timeout=10_000)

                    page.locator("#histogramBins").fill("3")
                    page.locator('.segmented[data-control="histogramDistribution"] button[data-value="cumulative"]').click()
                    page.locator('.segmented[data-control="histogramYAxis"] button[data-value="probability"]').click()
                    page.locator('.segmented[data-control="histogramLogScale"] button[data-value="y"]').click()
                    page.locator('.segmented[data-control="histogramSampleMode"] button[data-value="all"]').click()

                    self.click_sidebar_favourite_action(page, "#sidebarFavouriteAddBtn")
                    page.locator("#sidebarFavouritePopover:not([hidden])").wait_for(timeout=10_000)
                    page.wait_for_function(
                        """() => document.querySelector('input[name="sidebarFavouriteScope"][value="histogram_view"]')?.checked
                          && [...document.querySelectorAll(".sidebar-favourite-scope-option span")]
                            .map((node) => node.textContent.trim()).join("|") === "Histogram view|Metrics + filter|Metrics" """,
                        timeout=10_000,
                    )
                    page.locator("#sidebarFavouriteNameInput").fill("Histogram view")
                    page.locator('[data-favourite-action="save-add"]').click()
                    page.wait_for_function(
                        """() => [...document.querySelectorAll(".saved-favourite-option")]
                          .some((button) => button.querySelector(".saved-filter-name")?.textContent.trim() === "Histogram view"
                            && button.querySelector(".favourite-detail")?.textContent.trim() === "Histogram view"
                            && button.classList.contains("active")) """,
                        timeout=10_000,
                    )
                    self.assertTrue(favourites_path.exists())
                    saved_payload = json.loads(favourites_path.read_text(encoding="utf-8"))
                    saved_view = saved_payload["favourites"][0]["view"]
                    self.assertEqual(saved_view["scope"], "histogram_view")
                    self.assertEqual(saved_view["filter"], "segment = 'B'")
                    self.assertEqual(
                        saved_view["histogram"],
                        {
                            "bins": "3",
                            "distribution": "cumulative",
                            "yAxis": "probability",
                            "logScale": "y",
                            "sampleMode": "all",
                        },
                    )

                    page.locator("#filterRowClearBtn").click()
                    page.locator("#histogramBins").fill("8")
                    page.locator('.segmented[data-control="histogramDistribution"] button[data-value="incremental"]').click()
                    page.locator('.segmented[data-control="histogramYAxis"] button[data-value="sum"]').click()
                    page.locator('.segmented[data-control="histogramLogScale"] button[data-value="none"]').click()
                    page.locator('.segmented[data-control="histogramSampleMode"] button[data-value="100k"]').click()
                    page.wait_for_function("""() => !document.querySelector(".saved-favourite-option.active")""", timeout=10_000)

                    page.locator(".saved-favourite-option").filter(has_text="Histogram view").click()
                    page.wait_for_function(
                        """() => document.querySelector("#histogramTool")?.classList.contains("active")
                          && document.querySelector("#histogramBins")?.value === "3"
                          && document.querySelector('.segmented[data-control="histogramDistribution"] button[data-value="cumulative"]')?.classList.contains("active")
                          && document.querySelector('.segmented[data-control="histogramYAxis"] button[data-value="probability"]')?.classList.contains("active")
                          && document.querySelector('.segmented[data-control="histogramLogScale"] button[data-value="y"]')?.classList.contains("active")
                          && document.querySelector('.segmented[data-control="histogramSampleMode"] button[data-value="all"]')?.classList.contains("active")
                          && document.querySelector("#filterInput")?.value === "segment = 'B'"
                          && document.querySelector("#actualMetricTitle .metric-value")?.textContent.trim()
                          && document.querySelector("#weightMetricTitle .metric-value")?.textContent.trim()
                          && document.querySelector(".saved-favourite-option.active .saved-filter-name")?.textContent.trim() === "Histogram view" """,
                        timeout=10_000,
                    )
                    restored_metric_titles = page.evaluate(
                        """() => ({
                          actual: document.querySelector("#actualMetricTitle .metric-value")?.textContent.trim() || "",
                          weight: document.querySelector("#weightMetricTitle .metric-value")?.textContent.trim() || "",
                        })"""
                    )
                    self.assertTrue(restored_metric_titles["actual"])
                    self.assertTrue(restored_metric_titles["weight"])
                    page.locator("#lineBarTool").click()
                    page.wait_for_function(
                        """(expected) => document.querySelector("#lineBarTool")?.classList.contains("active")
                          && document.querySelector("#actualMetricTitle .metric-value")?.textContent.trim() === expected.actual
                          && document.querySelector("#weightMetricTitle .metric-value")?.textContent.trim() === expected.weight """,
                        arg=restored_metric_titles,
                        timeout=10_000,
                    )
                    self.assertEqual(page_errors, [])
                    browser.close()
            finally:
                server.should_exit = True
                thread.join(timeout=5)
                stop_persistent_glm_fit_worker()

    @unittest.skipUnless(RUN_BROWSER_TESTS, "set PY_LUCIDUM_RUN_BROWSER_TESTS=1 to run browser smoke tests")
    @unittest.skipUnless(sync_playwright is not None, "playwright is not installed")
    def test_favourites_startup_does_not_show_empty_state_while_loading(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            data_path = root / "sample.csv"
            data_path.write_text(
                "vehicle_age,segment,price,expected,value\n"
                "1,A,100,90,10\n"
                "2,A,200,210,20\n"
                "3,B,300,290,30\n"
                "4,B,400,410,40\n",
                encoding="utf-8",
            )
            favourite_view = {
                "version": 1,
                "source": "dataset",
                "scope": "line_bar_view",
                "view": "chart",
                "x": "vehicle_age",
                "xSource": "dataset",
                "sort": "alpha",
                "lowGroup": "0",
                "labels": "none",
                "bandWidth": "1",
                "quantileMode": "off",
                "dateBucket": "none",
                "transform": "none",
                "sigma": "0",
                "partialDependence": "none",
                "featureSort": "alpha",
                "expectedSort": "alpha",
                "actual": {"value": "price", "sourceId": "dataset", "metricKind": "metric"},
                "denominator": "value",
                "expectedSelections": [],
                "filter": "",
                "filterSelectionMode": "grouped",
                "filterOperator": "and",
                "savedFilterRows": [],
            }
            favourites_path = root / "favourites.json"
            favourites_path.write_text(
                json.dumps(
                    {
                        "version": 1,
                        "favourites": [
                            {
                                "id": "startup-view",
                                "name": "Startup view",
                                "view": favourite_view,
                            },
                        ],
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
            base_url, server, thread = self.start_app(
                data_path,
                line_bar_favourites_path=favourites_path,
                defaults={"x": "segment", "actual": "expected", "denominator": "value"},
                tools=["line_bar"],
            )
            try:
                assert sync_playwright is not None
                with sync_playwright() as playwright:
                    browser = playwright.chromium.launch()
                    try:
                        page = browser.new_page(viewport={"width": 1280, "height": 800})
                        page_errors: list[str] = []
                        page.on("pageerror", lambda error: page_errors.append(str(error)))
                        page.add_init_script(
                            """
                            (() => {
                              const originalFetch = window.fetch.bind(window);
                              window.__lucidumDelayedResolvers = {};
                              window.__lucidumReleaseDelayedFetch = (path) => {
                                const resolve = window.__lucidumDelayedResolvers[path];
                                if (resolve) {
                                  delete window.__lucidumDelayedResolvers[path];
                                  resolve();
                                }
                              };
                              window.fetch = async (...args) => {
                                const input = args[0];
                                const rawUrl = typeof input === "string" ? input : input?.url || "";
                                const path = new URL(rawUrl, window.location.href).pathname;
                                if (path === "/api/line-bar/favourites") {
                                  await new Promise((resolve) => {
                                    window.__lucidumDelayedResolvers[path] = resolve;
                                  });
                                }
                                return originalFetch(...args);
                              };
                            })();
                            """
                        )
                        page.goto(base_url, wait_until="domcontentloaded")
                        page.wait_for_function(
                            """() => Boolean(window.__lucidumDelayedResolvers?.["/api/line-bar/favourites"])""",
                            timeout=10000,
                        )
                        pending_text = page.locator("#favouritesSelect").inner_text().strip()
                        self.assertEqual(pending_text, "")
                        self.assertNotIn("No favourites", pending_text)
                        self.assertTrue(
                            page.locator("#favouritesSelect").evaluate(
                                """(node) => node.hasAttribute("aria-busy")"""
                            )
                        )

                        page.evaluate(
                            """() => window.__lucidumReleaseDelayedFetch("/api/line-bar/favourites")"""
                        )
                        page.wait_for_function(
                            """() => document.querySelector("#favouritesSelect .saved-favourite-option")?.textContent.includes("Startup view")
                              && document.querySelector("#favouritesSelectedMeta")?.textContent === "Startup view" """,
                            timeout=10000,
                        )
                        self.assertFalse(
                            page.locator("#favouritesSelect").evaluate(
                                """(node) => node.hasAttribute("aria-busy")"""
                            )
                        )
                        self.assertNotIn("No favourites", page.locator("#favouritesSelect").inner_text())
                        self.assertEqual(page_errors, [])
                    finally:
                        browser.close()
            finally:
                server.should_exit = True
                thread.join(timeout=5)
                stop_persistent_glm_fit_worker()

    @unittest.skipUnless(RUN_BROWSER_TESTS, "set PY_LUCIDUM_RUN_BROWSER_TESTS=1 to run browser smoke tests")
    @unittest.skipUnless(sync_playwright is not None, "playwright is not installed")
    def test_view_favourites_restore_with_single_data_request_and_no_stale_kpi(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            data_path = root / "sample.csv"
            data_path.write_text(
                "PostcodeArea,PostcodeSector,PostcodeUnit,lat,long,vehicle_age,segment,price,value,alt_price\n"
                "AB,AB10 1,AB10 1AA,57.1,-2.1,1,A,100,10,1000\n"
                "AB,AB10 1,AB10 1AB,57.2,-2.2,2,A,200,20,2000\n"
                "AL,AL1 1,AL1 1AA,51.8,-0.3,3,B,300,30,3000\n"
                "AL,AL1 2,AL1 2AA,51.7,-0.2,4,B,400,40,4000\n",
                encoding="utf-8",
            )
            kpis_path = root / "kpi_spec.csv"
            kpis_path.write_text(
                "group,name,actual,denominator,decimals,format\n"
                "TEST,Price,price,value,1,number\n"
                "TEST,Alt,alt_price,value,1,number\n",
                encoding="utf-8",
            )
            favourite_view = {
                "version": 1,
                "source": "dataset",
                "x": "vehicle_age",
                "xSource": "dataset",
                "sort": "alpha",
                "lowGroup": "0",
                "labels": "none",
                "bandWidth": "1",
                "quantileMode": "off",
                "dateBucket": "none",
                "transform": "none",
                "sigma": "0",
                "partialDependence": "none",
                "featureSort": "alpha",
                "expectedSort": "alpha",
                "actual": {"value": "price", "sourceId": "dataset", "metricKind": "metric"},
                "denominator": "value",
                "expectedSelections": [],
                "filter": "",
                "filterSelectionMode": "grouped",
                "filterOperator": "and",
                "savedFilterRows": [],
            }
            favourites_path = root / "favourites.json"
            favourites_path.write_text(
                json.dumps(
                    {
                        "version": 1,
                        "favourites": [
                            {
                                "id": "chart-fav",
                                "name": "Chart favourite",
                                "view": {**favourite_view, "scope": "line_bar_view", "view": "chart"},
                            },
                            {
                                "id": "table-fav",
                                "name": "Table favourite",
                                "view": {**favourite_view, "scope": "line_bar_view", "view": "table", "x": "segment"},
                            },
                            {
                                "id": "map-fav",
                                "name": "Map favourite",
                                "view": {
                                    **favourite_view,
                                    "scope": "map_view",
                                    "map": {
                                        "level": "sector",
                                        "baseMap": "blank",
                                        "palette": "viridis",
                                        "lineWeight": 1,
                                        "dotSize": 1,
                                        "opacity": 1,
                                        "hotspots": 0,
                                        "labelSize": 0,
                                        "smoothingLevel": 0,
                                    },
                                },
                            },
                        ],
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
            base_url, server, thread = self.start_app(
                data_path,
                kpis_path=kpis_path,
                use_kpis=True,
                line_bar_favourites_path=favourites_path,
                defaults={"x": "vehicle_age", "actual": "alt_price", "denominator": "value"},
                tools=["line_bar", "uk_map"],
            )
            try:
                assert sync_playwright is not None
                with sync_playwright() as playwright:
                    browser = playwright.chromium.launch()
                    page = browser.new_page(viewport={"width": 1280, "height": 800})
                    page_errors: list[str] = []
                    request_counts = {
                        "/api/chart": 0,
                        "/api/line-bar/table": 0,
                        "/api/uk-map/summary": 0,
                        "/api/metrics/summary": 0,
                        "/api/filter/row-count": 0,
                    }
                    page.on("pageerror", lambda error: page_errors.append(str(error)))

                    def count_request(request: object) -> None:
                        path = "/" + request.url.split("/", 3)[3].split("?", 1)[0]
                        if path in request_counts:
                            request_counts[path] += 1

                    page.on("request", count_request)
                    page.add_init_script(
                        """
                        (() => {
                          const originalFetch = window.fetch.bind(window);
                          window.__lucidumDelayPaths = new Set();
                          window.__lucidumDelayedResolvers = {};
                          window.__lucidumReleaseDelayedFetch = (path) => {
                            window.__lucidumDelayPaths.delete(path);
                            const resolve = window.__lucidumDelayedResolvers[path];
                            if (resolve) {
                              delete window.__lucidumDelayedResolvers[path];
                              resolve();
                            }
                          };
                          window.fetch = async (...args) => {
                            const input = args[0];
                            const rawUrl = typeof input === "string" ? input : input?.url || "";
                            const path = new URL(rawUrl, window.location.href).pathname;
                            const responsePromise = originalFetch(...args);
                            if (window.__lucidumDelayPaths.has(path)) {
                              await new Promise((resolve) => {
                                window.__lucidumDelayedResolvers[path] = resolve;
                              });
                            }
                            return responsePromise;
                          };
                        })();
                        """
                    )

                    def reset_counts() -> None:
                        for key in request_counts:
                            request_counts[key] = 0

                    def delay_path(path: str) -> None:
                        page.evaluate("(path) => window.__lucidumDelayPaths.add(path)", path)

                    def release_path(path: str) -> None:
                        page.evaluate("(path) => window.__lucidumReleaseDelayedFetch(path)", path)

                    def click_favourite(name: str) -> None:
                        page.evaluate(
                            """
                            (name) => {
                              const row = [...document.querySelectorAll("#favouritesSelect .saved-favourite-option")]
                                .find((button) => (button.textContent || "").includes(name));
                              if (!row) throw new Error(`Favourite not found: ${name}`);
                              row.click();
                            }
                            """,
                            name,
                        )

                    def select_alt_kpi() -> None:
                        page.evaluate(
                            """
                            () => {
                              const row = [...document.querySelectorAll("#kpiSelect .kpi-option")]
                                .find((button) => (button.textContent || "").includes("Alt"));
                              if (!row) throw new Error("Alt KPI not found");
                              row.click();
                            }
                            """
                        )
                        page.wait_for_function(
                            """() => document.querySelector("#actualNumerator")?.value === "alt_price"
                              && (document.querySelector("#actualMetricTitle")?.textContent || "").includes("100")
                              && (!document.querySelector("#lineBarTool")?.classList.contains("active")
                                || (!((document.querySelector("#lineBarGroupMeta")?.textContent || "").includes("Computing"))
                                  && !((document.querySelector("#lineBarTableContent")?.textContent || "").includes("Loading table"))))
                              && (!document.querySelector("#ukMapTool")?.classList.contains("active")
                                || !((document.querySelector("#mapGroupMeta")?.textContent || "").includes("Computing")))""",
                            timeout=10_000,
                        )

                    def assert_metric_pending() -> None:
                        pending = page.evaluate(
                            """
                            () => ({
                              actual: document.querySelector("#actualNumerator")?.value || "",
                              title: document.querySelector("#actualMetricTitle")?.textContent.trim() || "",
                              hasValue: Boolean(document.querySelector("#actualMetricTitle .metric-value")),
                            })
                            """
                        )
                        self.assertEqual(pending["actual"], "price")
                        self.assertEqual(pending["title"], "Actual")
                        self.assertFalse(pending["hasValue"])

                    page.goto(base_url, wait_until="domcontentloaded")
                    page.locator("#datasetMeta").get_by_text("sample.csv").wait_for(timeout=10_000)
                    page.wait_for_function(
                        """() => document.querySelector("#lineBarTool")?.classList.contains("active")
                          && (document.querySelector("#lineBarGroupMeta")?.textContent || "").includes("groups")""",
                        timeout=10_000,
                    )

                    select_alt_kpi()
                    delay_path("/api/chart")
                    reset_counts()
                    click_favourite("Chart favourite")
                    page.wait_for_function(
                        """() => document.querySelector("#actualNumerator")?.value === "price"
                          && (document.querySelector("#lineBarGroupMeta")?.textContent || "") === "Computing..." """,
                        timeout=10_000,
                    )
                    assert_metric_pending()
                    page.wait_for_function("() => window.__lucidumDelayedResolvers['/api/chart']", timeout=10_000)
                    self.assertEqual(request_counts["/api/chart"], 1)
                    self.assertEqual(request_counts["/api/metrics/summary"], 0)
                    self.assertEqual(request_counts["/api/filter/row-count"], 0)
                    release_path("/api/chart")
                    page.wait_for_function(
                        """() => (document.querySelector("#actualMetricTitle")?.textContent || "").includes("10.0")
                          && (document.querySelector("#lineBarGroupMeta")?.textContent || "").includes("groups")""",
                        timeout=10_000,
                    )

                    select_alt_kpi()
                    delay_path("/api/line-bar/table")
                    reset_counts()
                    click_favourite("Table favourite")
                    page.wait_for_function(
                        """() => document.querySelector("#actualNumerator")?.value === "price"
                          && (document.querySelector("#lineBarTableContent")?.textContent || "").includes("Loading table")""",
                        timeout=10_000,
                    )
                    assert_metric_pending()
                    page.wait_for_function("() => window.__lucidumDelayedResolvers['/api/line-bar/table']", timeout=10_000)
                    self.assertEqual(request_counts["/api/line-bar/table"], 1)
                    self.assertEqual(request_counts["/api/chart"], 0)
                    self.assertEqual(request_counts["/api/metrics/summary"], 0)
                    self.assertEqual(request_counts["/api/filter/row-count"], 0)
                    release_path("/api/line-bar/table")
                    page.wait_for_function(
                        """() => (document.querySelector("#actualMetricTitle")?.textContent || "").includes("10.0")
                          && (document.querySelector("#lineBarGroupMeta")?.textContent || "").includes("groups")
                          && !(document.querySelector("#lineBarTableContent")?.textContent || "").includes("Loading table")""",
                        timeout=10_000,
                    )

                    select_alt_kpi()
                    delay_path("/api/uk-map/summary")
                    reset_counts()
                    click_favourite("Map favourite")
                    page.wait_for_function(
                        """() => document.querySelector("#ukMapTool")?.classList.contains("active")
                          && document.querySelector("#actualNumerator")?.value === "price"
                          && (document.querySelector("#mapGroupMeta")?.textContent || "") === "Computing map..." """,
                        timeout=10_000,
                    )
                    assert_metric_pending()
                    page.wait_for_function("() => window.__lucidumDelayedResolvers['/api/uk-map/summary']", timeout=10_000)
                    self.assertEqual(request_counts["/api/uk-map/summary"], 1)
                    self.assertEqual(request_counts["/api/chart"], 0)
                    self.assertEqual(request_counts["/api/line-bar/table"], 0)
                    self.assertEqual(request_counts["/api/metrics/summary"], 0)
                    self.assertEqual(request_counts["/api/filter/row-count"], 0)
                    release_path("/api/uk-map/summary")
                    page.wait_for_function(
                        """() => (document.querySelector("#actualMetricTitle")?.textContent || "").includes("10.0")
                          && (document.querySelector("#mapGroupMeta")?.textContent || "").includes("sectors matched")""",
                        timeout=10_000,
                    )

                    self.assertEqual(page_errors, [])
                    browser.close()
            finally:
                server.should_exit = True
                thread.join(timeout=5)
                stop_persistent_glm_fit_worker()

    @unittest.skipUnless(RUN_BROWSER_TESTS, "set PY_LUCIDUM_RUN_BROWSER_TESTS=1 to run browser smoke tests")
    @unittest.skipUnless(sync_playwright is not None, "playwright is not installed")
    def test_uk_map_favourites_save_and_restore_map_view(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            data_path = root / "sample.csv"
            data_path.write_text(
                "PostcodeArea,PostcodeSector,vehicle_age,price,value,PostcodeUnit,lat,long\n"
                "AB,AB10 1,1,100,10,AB10 1AA,57.1,-2.1\n"
                "AB,AB10 1,2,200,20,AB10 1AB,57.2,-2.2\n"
                "AL,AL1 1,3,300,30,AL1 1AA,51.8,-0.3\n"
                "AL,AL1 2,4,400,40,AL1 2AA,51.7,-0.2\n",
                encoding="utf-8",
            )
            filters_path = root / "filter_spec.csv"
            filters_path.write_text(
                "theme,name,expression\n"
                "AGE,Older,vehicle_age >= 3\n",
                encoding="utf-8",
            )
            favourites_path = root / "config" / "favourites.json"
            base_url, server, thread = self.start_app(
                data_path,
                filters_path=filters_path,
                use_saved_filters=True,
                line_bar_favourites_path=favourites_path,
                tools=["line_bar", "uk_map"],
                defaults={"x": "vehicle_age", "actual": "price", "denominator": "value"},
            )
            try:
                assert sync_playwright is not None
                with sync_playwright() as playwright:
                    browser = playwright.chromium.launch()
                    page = browser.new_page(viewport={"width": 1280, "height": 800})
                    page_errors: list[str] = []
                    summary_requests: list[str] = []
                    page.on("pageerror", lambda error: page_errors.append(str(error)))
                    page.on(
                        "request",
                        lambda request: summary_requests.append(request.url)
                        if request.url.endswith("/api/uk-map/summary")
                        else None,
                    )
                    page.add_init_script(
                        """
                        (() => {
                          const originalFetch = window.fetch.bind(window);
                          window.__lucidumDelayPaths = new Set();
                          window.__lucidumDelayedResolvers = {};
                          window.__lucidumReleaseDelayedFetch = (path) => {
                            window.__lucidumDelayPaths.delete(path);
                            const resolve = window.__lucidumDelayedResolvers[path];
                            if (resolve) {
                              delete window.__lucidumDelayedResolvers[path];
                              resolve();
                            }
                          };
                          window.fetch = async (...args) => {
                            const input = args[0];
                            const rawUrl = typeof input === "string" ? input : input?.url || "";
                            const path = new URL(rawUrl, window.location.href).pathname;
                            const responsePromise = originalFetch(...args);
                            if (window.__lucidumDelayPaths.has(path)) {
                              await new Promise((resolve) => {
                                window.__lucidumDelayedResolvers[path] = resolve;
                              });
                            }
                            return responsePromise;
                          };
                        })();
                        """
                    )

                    def delay_path(path: str) -> None:
                        page.evaluate("(path) => window.__lucidumDelayPaths.add(path)", path)

                    def release_path(path: str) -> None:
                        page.evaluate("(path) => window.__lucidumReleaseDelayedFetch(path)", path)

                    try:
                        page.goto(base_url, wait_until="domcontentloaded")
                        page.locator("#ukMapTool").click()
                        page.locator("#ukMap:not(.hidden)").wait_for(timeout=20_000)
                        page.locator("#mapFloatingControl:not(.hidden)").wait_for(timeout=10_000)
                        page.wait_for_function('() => document.querySelector("#mapGroupMeta")?.textContent.includes("areas matched")')

                        with page.expect_response(lambda response: response.url.endswith("/api/uk-map/summary") and response.status == 200, timeout=10_000):
                            page.locator('#mapLevelTiles input[name="mapLevel"][value="sector"]').check()
                        page.wait_for_function('() => document.querySelector("#mapGroupMeta")?.textContent.includes("sectors matched")')
                        page.locator('#mapBaseLayerTiles input[name="baseMap"][value="grey"]').check()
                        page.locator('.map-palette-button[data-palette="viridis"]').click()
                        page.wait_for_function(
                            """() => document.querySelector("#mapHotspotsMinLabel")?.textContent.trim() === "Low"
                              && document.querySelector("#mapHotspotsMaxLabel")?.textContent.trim() === "High"
                              && document.querySelector("#mapHotspotsMinLabel")?.style.getPropertyValue("--map-extreme-color") === "#fde725"
                              && document.querySelector("#mapHotspotsMaxLabel")?.style.getPropertyValue("--map-extreme-color") === "#440154" """,
                            timeout=10_000,
                        )
                        page.eval_on_selector("#mapLineWeight", "(input) => { input.value = '3'; input.dispatchEvent(new Event('input', { bubbles: true })); }")
                        page.eval_on_selector("#mapOpacity", "(input) => { input.value = '4'; input.dispatchEvent(new Event('input', { bubbles: true })); }")
                        with page.expect_response(lambda response: response.url.endswith("/api/uk-map/summary") and response.status == 200, timeout=10_000):
                            page.eval_on_selector("#mapSmoothing", "(input) => { input.value = '2'; input.dispatchEvent(new Event('input', { bubbles: true })); }")
                        page.wait_for_function(
                            """() => document.querySelector("#mapLineWeight")?.value === "3"
                              && document.querySelector("#mapOpacity")?.value === "4"
                              && document.querySelector("#mapSmoothing")?.value === "2" """,
                            timeout=10_000,
                        )
                        page.evaluate(
                            """
                            () => {
                              const map = document.querySelector("#ukMap")?._lucidumMap;
                              map.setView([51.5, -0.12], 9, { animate: false });
                            }
                            """
                        )
                        page.wait_for_timeout(50)

                        if page.locator("#favouritesCollapseBtn").get_attribute("aria-expanded") == "false":
                            page.locator("#favouritesCollapseBtn").click()
                            page.wait_for_function(
                                '() => document.querySelector("#favouritesCollapseBtn")?.getAttribute("aria-expanded") === "true"',
                                timeout=10_000,
                            )
                        self.click_sidebar_favourite_action(page, "#sidebarFavouriteAddBtn")
                        page.locator("#sidebarFavouritePopover:not([hidden])").wait_for(timeout=10_000)
                        page.wait_for_function(
                            """() => document.querySelector('input[name="sidebarFavouriteScope"][value="map_view"]')?.checked
                              && [...document.querySelectorAll(".sidebar-favourite-scope-option span")]
                                .map((node) => node.textContent.trim()).join("|") === "Map view|Metrics + filter|Metrics" """,
                            timeout=10_000,
                        )
                        page.locator("#sidebarFavouriteNameInput").fill("Sector map")
                        page.locator('[data-favourite-action="save-add"]').click()
                        page.wait_for_function(
                            """() => [...document.querySelectorAll(".saved-favourite-option")]
                              .some((button) => button.querySelector(".saved-filter-name")?.textContent.trim() === "Sector map"
                                && button.querySelector(".favourite-detail")?.textContent.trim() === "Map view"
                                && button.classList.contains("active")) """,
                            timeout=10_000,
                        )
                        map_favourite_id = page.eval_on_selector(
                            ".saved-favourite-option.active",
                            'button => button?.dataset.favouriteId || ""',
                        )
                        self.assertTrue(map_favourite_id)
                        self.assertTrue(favourites_path.exists())
                        saved_payload = json.loads(favourites_path.read_text(encoding="utf-8"))
                        saved_map_favourite = next(
                            item for item in saved_payload["favourites"]
                            if item["id"] == map_favourite_id
                        )
                        saved_map = saved_map_favourite["view"]["map"]
                        self.assertEqual(saved_map_favourite["view"]["scope"], "map_view")
                        self.assertAlmostEqual(saved_map["center"]["lat"], 51.5, delta=0.01)
                        self.assertAlmostEqual(saved_map["center"]["lng"], -0.12, delta=0.01)
                        self.assertAlmostEqual(saved_map["zoom"], 9, delta=0.01)
                        self.assertNotIn("view", saved_map)

                        def base_tile_id() -> str | None:
                            return page.evaluate(
                                """
                                () => {
                                  const layer = document.querySelector("#ukMap")?._lucidumBaseTileLayer;
                                  if (!layer) return null;
                                  if (!layer._lucidumTestId) layer._lucidumTestId = `tile-${Math.random()}`;
                                  return layer._lucidumTestId;
                                }
                                """
                            )

                        grey_tile_id = base_tile_id()
                        self.assertTrue(grey_tile_id)
                        with page.expect_response(lambda response: response.url.endswith("/api/uk-map/summary") and response.status == 200, timeout=10_000):
                            page.eval_on_selector("#mapSmoothing", "(input) => { input.value = '0'; input.dispatchEvent(new Event('input', { bubbles: true })); }")
                        page.wait_for_function(
                            """() => document.querySelector("#mapSmoothing")?.value === "0"
                              && (document.querySelector("#mapGroupMeta")?.textContent || "").includes("sectors matched") """,
                            timeout=10_000,
                        )
                        unsmoothed_meta = page.locator("#mapGroupMeta").inner_text()
                        self.click_sidebar_favourite_action(page, "#sidebarFavouriteAddBtn")
                        page.locator("#sidebarFavouritePopover:not([hidden])").wait_for(timeout=10_000)
                        page.locator("#sidebarFavouriteNameInput").fill("Sector map N0")
                        page.locator('[data-favourite-action="save-add"]').click()
                        page.wait_for_function(
                            """() => [...document.querySelectorAll(".saved-favourite-option")]
                              .some((button) => button.querySelector(".saved-filter-name")?.textContent.trim() === "Sector map N0"
                                && button.classList.contains("active")) """,
                            timeout=10_000,
                        )
                        unsmoothed_favourite_id = page.eval_on_selector(
                            ".saved-favourite-option.active",
                            'button => button?.dataset.favouriteId || ""',
                        )
                        self.assertTrue(unsmoothed_favourite_id)
                        delay_path("/api/uk-map/summary")
                        summary_count_before_in_place_restore = len(summary_requests)
                        page.locator(f'.saved-favourite-option[data-favourite-id="{map_favourite_id}"]').click()
                        page.wait_for_function(
                            """([id]) => document.querySelector(`.saved-favourite-option[data-favourite-id="${id}"]`)?.classList.contains("active")
                              && document.querySelector("#mapSmoothing")?.value === "2" """,
                            arg=[map_favourite_id],
                            timeout=10_000,
                        )
                        page.wait_for_function("() => window.__lucidumDelayedResolvers['/api/uk-map/summary']", timeout=10_000)
                        self.assertEqual(page.locator("#mapGroupMeta").inner_text(), unsmoothed_meta)
                        self.assertEqual(len(summary_requests), summary_count_before_in_place_restore + 1)
                        release_path("/api/uk-map/summary")
                        page.wait_for_function(
                            """([id]) => document.querySelector(`.saved-favourite-option[data-favourite-id="${id}"]`)?.classList.contains("active")
                              && document.querySelector("#mapSmoothing")?.value === "2"
                              && (document.querySelector("#mapGroupMeta")?.textContent || "").includes("sectors matched") """,
                            arg=[map_favourite_id],
                            timeout=10_000,
                        )
                        self.click_sidebar_favourite_action(page, "#sidebarFavouriteAddBtn")
                        page.locator("#sidebarFavouritePopover:not([hidden])").wait_for(timeout=10_000)
                        page.locator("#sidebarFavouriteNameInput").fill("Sector map copy")
                        page.locator('[data-favourite-action="save-add"]').click()
                        page.wait_for_function(
                            """() => [...document.querySelectorAll(".saved-favourite-option")]
                              .some((button) => button.querySelector(".saved-filter-name")?.textContent.trim() === "Sector map copy"
                                && button.classList.contains("active")) """,
                            timeout=10_000,
                        )
                        map_favourite_copy_id = page.eval_on_selector(
                            ".saved-favourite-option.active",
                            'button => button?.dataset.favouriteId || ""',
                        )
                        self.assertTrue(map_favourite_copy_id)
                        self.assertNotEqual(map_favourite_copy_id, map_favourite_id)
                        summary_count_before_cached_restore = len(summary_requests)
                        page.locator(f'.saved-favourite-option[data-favourite-id="{map_favourite_id}"]').click()
                        page.wait_for_function(
                            """([id]) => document.querySelector(`.saved-favourite-option[data-favourite-id="${id}"]`)?.classList.contains("active")""",
                            arg=[map_favourite_id],
                            timeout=10_000,
                        )
                        page.wait_for_timeout(250)
                        self.assertEqual(len(summary_requests), summary_count_before_cached_restore)
                        self.assertEqual(base_tile_id(), grey_tile_id)

                        page.locator('#mapBaseLayerTiles input[name="baseMap"][value="osm"]').check()
                        page.wait_for_function("""() => document.querySelector('#mapBaseLayerTiles input[name="baseMap"][value="osm"]')?.checked""", timeout=10_000)
                        osm_tile_id = base_tile_id()
                        self.assertTrue(osm_tile_id)
                        self.assertNotEqual(osm_tile_id, grey_tile_id)
                        self.click_sidebar_favourite_action(page, "#sidebarFavouriteAddBtn")
                        page.locator("#sidebarFavouritePopover:not([hidden])").wait_for(timeout=10_000)
                        page.locator("#sidebarFavouriteNameInput").fill("Sector map OSM")
                        page.locator('[data-favourite-action="save-add"]').click()
                        page.wait_for_function(
                            """() => [...document.querySelectorAll(".saved-favourite-option")]
                              .some((button) => button.querySelector(".saved-filter-name")?.textContent.trim() === "Sector map OSM"
                                && button.classList.contains("active")) """,
                            timeout=10_000,
                        )
                        summary_count_before_base_restore = len(summary_requests)
                        page.locator(f'.saved-favourite-option[data-favourite-id="{map_favourite_id}"]').click()
                        page.wait_for_function(
                            """([id]) => document.querySelector(`.saved-favourite-option[data-favourite-id="${id}"]`)?.classList.contains("active")
                              && document.querySelector('#mapBaseLayerTiles input[name="baseMap"][value="grey"]')?.checked """,
                            arg=[map_favourite_id],
                            timeout=10_000,
                        )
                        page.wait_for_timeout(250)
                        self.assertEqual(len(summary_requests), summary_count_before_base_restore)
                        self.assertNotEqual(base_tile_id(), osm_tile_id)

                        page.evaluate(
                            """
                            () => {
                              const map = document.querySelector("#ukMap")?._lucidumMap;
                              map.setView([54.5, -3.2], 6, { animate: false });
                            }
                            """
                        )
                        page.wait_for_timeout(100)
                        current_camera = page.evaluate(
                            """
                            () => {
                              const map = document.querySelector("#ukMap")?._lucidumMap;
                              const center = map.getCenter();
                              return { lat: center.lat, lng: center.lng, zoom: map.getZoom() };
                            }
                            """
                        )
                        self.assertAlmostEqual(current_camera["lat"], 54.5, delta=0.25)
                        self.assertAlmostEqual(current_camera["lng"], -3.2, delta=0.25)
                        self.assertAlmostEqual(current_camera["zoom"], 6, delta=0.5)

                        page.locator("#lineBarTool").click()
                        page.locator("#chart:not(.hidden)").wait_for(timeout=10_000)
                        page.locator("#favouritesCollapseBtn").click()
                        page.wait_for_function(
                            '() => document.querySelector("#favouritesCollapseBtn")?.getAttribute("aria-expanded") === "false"',
                            timeout=10_000,
                        )
                        page.locator(".sidebar-metric-section").wait_for(state="visible", timeout=10_000)
                        page.locator("#actualNumerator").select_option("vehicle_age")
                        page.locator("#denominator").select_option("__none__")
                        page.locator("#favouritesCollapseBtn").click()
                        page.wait_for_function(
                            '() => document.querySelector("#favouritesCollapseBtn")?.getAttribute("aria-expanded") === "true"',
                            timeout=10_000,
                        )
                        page.locator(f'.saved-favourite-option[data-favourite-id="{map_favourite_id}"]').click()
                        page.wait_for_function(
                            """([id]) => document.querySelector("#ukMapTool")?.classList.contains("active")
                              && document.querySelector("#mapGroupMeta")?.textContent.includes("sectors matched")
                              && document.querySelector(`.saved-favourite-option[data-favourite-id="${id}"]`)?.classList.contains("active")
                              && document.querySelector('#mapLevelTiles input[name="mapLevel"][value="sector"]')?.checked
                              && document.querySelector('#mapBaseLayerTiles input[name="baseMap"][value="grey"]')?.checked
                              && document.querySelector('.map-palette-button[data-palette="viridis"]')?.classList.contains("active")
                              && document.querySelector("#mapLineWeight")?.value === "3"
                              && document.querySelector("#mapOpacity")?.value === "4"
                              && document.querySelector("#mapSmoothing")?.value === "2"
                              && document.querySelector("#actualNumerator")?.value === "price"
                              && document.querySelector("#denominator")?.value === "value" """,
                            arg=[map_favourite_id],
                            timeout=10_000,
                        )
                        restored_camera = page.evaluate(
                            """
                            () => {
                              const map = document.querySelector("#ukMap")?._lucidumMap;
                              const center = map.getCenter();
                              return { lat: center.lat, lng: center.lng, zoom: map.getZoom() };
                            }
                            """
                        )
                        self.assertAlmostEqual(restored_camera["lat"], 51.5, delta=0.25)
                        self.assertAlmostEqual(restored_camera["lng"], -0.12, delta=0.25)
                        self.assertAlmostEqual(restored_camera["zoom"], 9, delta=0.5)

                        page.locator('.map-palette-button[data-palette="spectral"]').click()
                        page.wait_for_function(
                            """() => !document.querySelector(".saved-favourite-option.active")
                              && document.querySelector("#mapHotspotsMinLabel")?.style.getPropertyValue("--map-extreme-color") === "#2c7bb6"
                              && document.querySelector("#mapHotspotsMaxLabel")?.style.getPropertyValue("--map-extreme-color") === "#a50026" """,
                            timeout=10_000,
                        )

                        self.click_sidebar_favourite_action(page, "#sidebarFavouriteAddBtn")
                        page.locator("#sidebarFavouriteNameInput").fill("Map metrics")
                        page.locator('[data-favourite-scope-option="metrics"]').click()
                        page.locator('[data-favourite-action="save-add"]').click()
                        page.wait_for_function(
                            """() => [...document.querySelectorAll(".saved-favourite-option .saved-filter-name")]
                              .some((node) => node.textContent.trim() === "Map metrics"
                                && node.closest(".saved-favourite-option")?.querySelector(".favourite-detail")?.textContent.trim() === "Metrics")""",
                            timeout=10_000,
                        )
                        metric_id = page.evaluate(
                            """
                            () => [...document.querySelectorAll(".saved-favourite-option")]
                              .find((button) => button.querySelector(".saved-filter-name")?.textContent.trim() === "Map metrics")
                              ?.dataset.favouriteId || ""
                            """
                        )
                        self.assertTrue(metric_id)
                        page.locator("#lineBarTool").click()
                        page.locator("#favouritesCollapseBtn").click()
                        page.wait_for_function(
                            '() => document.querySelector("#favouritesCollapseBtn")?.getAttribute("aria-expanded") === "false"',
                            timeout=10_000,
                        )
                        page.locator(".sidebar-metric-section").wait_for(state="visible", timeout=10_000)
                        page.locator("#actualNumerator").select_option("vehicle_age")
                        page.locator("#denominator").select_option("__none__")
                        page.locator("#favouritesCollapseBtn").click()
                        page.wait_for_function(
                            '() => document.querySelector("#favouritesCollapseBtn")?.getAttribute("aria-expanded") === "true"',
                            timeout=10_000,
                        )
                        page.locator(f'.saved-favourite-option[data-favourite-id="{metric_id}"]').click()
                        page.wait_for_function(
                            """() => document.querySelector("#lineBarTool")?.classList.contains("active")
                              && document.querySelector("#actualNumerator")?.value === "price"
                              && document.querySelector("#denominator")?.value === "value" """,
                            timeout=10_000,
                        )

                        page.locator("#ukMapTool").click()
                        if page.locator("#filterCollapseBtn").get_attribute("aria-expanded") == "false":
                            page.locator("#filterCollapseBtn").click()
                        age_heading = page.locator('.saved-filter-theme[data-filter-theme="AGE"]')
                        if age_heading.get_attribute("aria-expanded") == "false":
                            age_heading.click()
                        page.locator('.saved-filter-option[data-filter-theme="AGE"]').click()
                        page.locator("#favouritesCollapseBtn").click()
                        self.click_sidebar_favourite_action(page, "#sidebarFavouriteAddBtn")
                        page.locator("#sidebarFavouriteNameInput").fill("Map metric filter")
                        page.locator('[data-favourite-scope-option="metrics_filter"]').click()
                        page.locator('[data-favourite-action="save-add"]').click()
                        page.wait_for_function(
                            """() => [...document.querySelectorAll(".saved-favourite-option .saved-filter-name")]
                              .some((node) => node.textContent.trim() === "Map metric filter"
                                && node.closest(".saved-favourite-option")?.querySelector(".favourite-detail")?.textContent.trim() === "Metrics + filter")""",
                            timeout=10_000,
                        )
                        metric_filter_id = page.evaluate(
                            """
                            () => [...document.querySelectorAll(".saved-favourite-option")]
                              .find((button) => button.querySelector(".saved-filter-name")?.textContent.trim() === "Map metric filter")
                              ?.dataset.favouriteId || ""
                            """
                        )
                        self.assertTrue(metric_filter_id)
                        page.locator("#lineBarTool").click()
                        page.locator("#filterRowClearBtn").click()
                        page.locator(f'.saved-favourite-option[data-favourite-id="{metric_filter_id}"]').click()
                        page.wait_for_function(
                            """() => document.querySelector("#lineBarTool")?.classList.contains("active")
                              && document.querySelector("#filterInput")?.value === "vehicle_age >= 3" """,
                            timeout=10_000,
                        )
                        self.assertEqual(page_errors, [])
                    finally:
                        browser.close()
                self.assertTrue(favourites_path.exists())
            finally:
                server.should_exit = True
                thread.join(timeout=5)
                stop_persistent_glm_fit_worker()

    @unittest.skipUnless(RUN_BROWSER_TESTS, "set PY_LUCIDUM_RUN_BROWSER_TESTS=1 to run browser smoke tests")
    @unittest.skipUnless(sync_playwright is not None, "playwright is not installed")
    def test_line_bar_favourite_keeps_chart_format_while_next_view_loads(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            data_path = root / "sample.csv"
            data_path.write_text(
                "segment,price,ratio,value\n"
                "A,100,0.10,10\n"
                "A,120,0.20,20\n"
                "B,200,0.30,30\n"
                "B,240,0.40,40\n"
                "C,300,0.50,50\n"
                "C,360,0.60,60\n",
                encoding="utf-8",
            )
            kpis_path = root / "kpi_spec.csv"
            kpis_path.write_text(
                "group,name,actual,denominator,decimals,format\n"
                "FIN,Price,price,__none__,2,currency\n"
                "RATIO,Ratio,ratio,__none__,1,percent\n",
                encoding="utf-8",
            )
            base_view = {
                "version": 1,
                "source": "dataset",
                "x": "segment",
                "xSource": "dataset",
                "view": "chart",
                "sort": "alpha",
                "lowGroup": "0",
                "labels": "line",
                "bandWidth": "0",
                "quantileMode": "off",
                "dateBucket": "none",
                "transform": "none",
                "sigma": "0",
                "partialDependence": "none",
                "featureSort": "alpha",
                "expectedSort": "alpha",
                "denominator": "__none__",
                "expectedSelections": [],
                "filter": "",
                "filterSelectionMode": "single",
                "filterOperator": "and",
                "savedFilterRows": [],
            }
            favourites_path = root / "favourites.json"
            favourites_path.write_text(
                json.dumps(
                    {
                        "version": 1,
                        "favourites": [
                            {
                                "id": "ratio-view",
                                "name": "Ratio view",
                                "created_at": "2026-06-29T00:00:00Z",
                                "updated_at": "2026-06-29T00:00:00Z",
                                "view": {
                                    **base_view,
                                    "actual": {"value": "ratio", "sourceId": "dataset", "metricKind": "dataset"},
                                },
                            },
                        ],
                    }
                ),
                encoding="utf-8",
            )
            base_url, server, thread = self.start_app(
                data_path,
                kpis_path=kpis_path,
                use_kpis=True,
                line_bar_favourites_path=favourites_path,
                defaults={"x": "segment", "actual": "price", "denominator": "__none__"},
                tools=["line_bar"],
            )
            held_chart_routes: list[Any] = []

            def release_held_chart_routes() -> None:
                while held_chart_routes:
                    held_chart_routes.pop(0).continue_()

            try:
                assert sync_playwright is not None
                with sync_playwright() as playwright:
                    browser = playwright.chromium.launch()
                    page = browser.new_page(viewport={"width": 1280, "height": 800})
                    page_errors: list[str] = []
                    page.on("pageerror", lambda error: page_errors.append(str(error)))
                    try:
                        page.goto(base_url, wait_until="domcontentloaded")
                        page.wait_for_function(
                            '() => document.querySelector("#lineBarGroupMeta")?.textContent.includes("groups")',
                            timeout=10_000,
                        )
                        page.wait_for_function(
                            """
                            () => document.querySelector("#actualNumerator")?.value === "ratio"
                              && document.querySelector("#favouritesSelectedMeta")?.textContent === "Ratio view"
                            """,
                            timeout=10_000,
                        )
                        page.evaluate(
                            """
                            () => {
                              const select = document.querySelector("#actualNumerator");
                              select.value = "price";
                              select.dispatchEvent(new Event("change", { bubbles: true }));
                            }
                            """
                        )
                        page.wait_for_function(
                            """
                            () => document.querySelector("#actualNumerator")?.value === "price"
                              && !document.querySelector(".saved-favourite-option.active")
                              && document.querySelector("#lineBarGroupMeta")?.textContent.includes("groups")
                              && echarts.getInstanceByDom(document.querySelector("#chart"))?.getOption?.()
                                .series?.find((series) => series.type === "line")?.data?.[0] > 1
                            """,
                            timeout=10_000,
                        )
                        if page.locator("#lineBarToolbarToggleBtn").get_attribute("aria-expanded") == "false":
                            page.locator("#lineBarToolbarToggleBtn").click()
                            page.wait_for_function(
                                """
                                () => document.querySelector("#lineBarToolbarToggleBtn")?.getAttribute("aria-expanded") === "true"
                                  && getComputedStyle(document.querySelector("#lineBarToolbar")).display !== "none"
                                """,
                                timeout=10_000,
                            )
                        page.locator('.segmented[data-control="labels"] button[data-value="line"]').click()
                        page.wait_for_function(
                            """
                            () => {
                              const chart = echarts.getInstanceByDom(document.querySelector("#chart"));
                              const labels = chart?.getZr?.().storage.getDisplayList()
                                .map((item) => item.style?.text || "") || [];
                              return labels.includes("£110.00") && labels.includes("£220.00") && labels.includes("£330.00");
                            }
                            """,
                            timeout=10_000,
                        )

                        def handle_chart_route(route: Any) -> None:
                            if route.request.method == "POST":
                                held_chart_routes.append(route)
                                return
                            route.continue_()

                        page.route("**/api/chart", handle_chart_route)
                        if page.locator("#favouritesCollapseBtn").get_attribute("aria-expanded") == "false":
                            page.locator("#favouritesCollapseBtn").click()
                            page.wait_for_function(
                                '() => document.querySelector("#favouritesCollapseBtn")?.getAttribute("aria-expanded") === "true"',
                                timeout=10_000,
                            )
                        page.locator('.saved-favourite-option[data-favourite-id="ratio-view"]').wait_for(timeout=10_000)
                        page.locator('.saved-favourite-option[data-favourite-id="ratio-view"]').click()
                        page.wait_for_function(
                            """
                            () => document.querySelector("#actualNumerator")?.value === "ratio"
                              && document.querySelector("#favouritesSelectedMeta")?.textContent === "Ratio view"
                            """,
                            timeout=10_000,
                        )
                        for _ in range(50):
                            if held_chart_routes:
                                break
                            page.wait_for_timeout(100)
                        self.assertTrue(held_chart_routes)
                        page.wait_for_timeout(250)

                        pending_state = page.evaluate(
                            """
                            () => {
                              const chart = echarts.getInstanceByDom(document.querySelector("#chart"));
                              const option = chart?.getOption?.() || {};
                              const line = option.series?.find((series) => series.type === "line");
                              const yFormatter = option.yAxis?.[0]?.axisLabel?.formatter;
                              const labelFormatter = line?.label?.formatter;
                              const visibleLabels = chart?.getZr?.().storage.getDisplayList()
                                .map((item) => item.style?.text || "")
                                .filter(Boolean) || [];
                              return {
                                actual: document.querySelector("#actualNumerator")?.value || "",
                                groupMeta: document.querySelector("#lineBarGroupMeta")?.textContent || "",
                                data: line?.data || [],
                                ySample: typeof yFormatter === "function" ? yFormatter(1.23) : "",
                                labelSample: typeof labelFormatter === "function" ? labelFormatter({ value: 123.456 }) : "",
                                visibleLabels,
                              };
                            }
                            """
                        )
                        self.assertEqual(pending_state["actual"], "ratio")
                        self.assertEqual(pending_state["groupMeta"], "Computing...")
                        self.assertEqual(pending_state["data"], [110, 220, 330])
                        self.assertEqual(pending_state["ySample"], "£1.23")
                        self.assertEqual(pending_state["labelSample"], "£123.46")
                        self.assertIn("£110.00", pending_state["visibleLabels"])
                        self.assertNotIn("11,000.0%", pending_state["visibleLabels"])

                        release_held_chart_routes()
                        page.wait_for_function(
                            """
                            () => document.querySelector("#lineBarGroupMeta")?.textContent.includes("groups")
                              && echarts.getInstanceByDom(document.querySelector("#chart"))?.getOption?.()
                                .series?.find((series) => series.type === "line")?.data
                                ?.every((value) => Number(value) < 1)
                            """,
                            timeout=10_000,
                        )
                        loaded_state = page.evaluate(
                            """
                            () => {
                              const chart = echarts.getInstanceByDom(document.querySelector("#chart"));
                              const labels = chart?.getZr?.().storage.getDisplayList()
                                .map((item) => item.style?.text || "")
                                .filter(Boolean) || [];
                              return {
                                labels,
                                data: chart?.getOption?.().series?.find((series) => series.type === "line")?.data || [],
                              };
                            }
                            """
                        )
                        self.assertEqual(loaded_state["data"], [0.15000000000000002, 0.35, 0.55])
                        self.assertIn("15.0%", loaded_state["labels"])
                        self.assertIn("35.0%", loaded_state["labels"])
                        self.assertIn("55.0%", loaded_state["labels"])
                        self.assertEqual(page_errors, [])
                    finally:
                        release_held_chart_routes()
                        browser.close()
            finally:
                server.should_exit = True
                thread.join(timeout=5)
                stop_persistent_glm_fit_worker()

    @unittest.skipUnless(RUN_BROWSER_TESTS, "set PY_LUCIDUM_RUN_BROWSER_TESTS=1 to run browser smoke tests")
    @unittest.skipUnless(sync_playwright is not None, "playwright is not installed")
    def test_line_bar_favourite_restores_model_outputs_against_active_models(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            data_path = root / "sample.csv"
            row_count = 60
            data_rows = ["actualNumerator,denominator,Age,Segment,SAMPLE"]
            for index in range(1, row_count + 1):
                segment = "B" if index % 2 == 0 else "A"
                sample = "validation" if segment == "B" else "training"
                data_rows.append(f"{100 + index},{1000 + index},{20 + index % 40},{segment},{sample}")
            data_path.write_text("\n".join(data_rows) + "\n", encoding="utf-8")
            active_glm_predictions = [100.0 + index for index in range(1, row_count + 1)]
            saved_glm_predictions = [90.0 + index for index in range(1, row_count + 1)]
            active_gbm_predictions = [
                active_glm_predictions[index - 1] * (1.1 + index / 1000)
                for index in range(1, row_count + 1)
            ]
            saved_gbm_predictions = [
                active_glm_predictions[index - 1] * (1.5 + index / 1000)
                for index in range(1, row_count + 1)
            ]
            gbm_store = GbmModelStore(data_path)
            self.write_gbm_prediction_model(
                gbm_store,
                "saved-gbm",
                "Saved GBM",
                "2026-05-25T00:00:00Z",
                saved_gbm_predictions,
            )
            self.write_gbm_prediction_model(
                gbm_store,
                "active-gbm",
                "Active GBM",
                "2026-05-25T00:00:01Z",
                active_gbm_predictions,
            )
            gbm_store.activate_model("active-gbm")
            glm_store = GlmModelStore(data_path)
            self.write_glm_prediction_model(
                glm_store,
                "saved-glm",
                "Saved GLM",
                "2026-05-25T00:00:02Z",
                saved_glm_predictions,
            )
            self.write_glm_prediction_model(
                glm_store,
                "active-glm",
                "Active GLM",
                "2026-05-25T00:00:03Z",
                active_glm_predictions,
            )
            glm_store.activate_model("active-glm")
            saved_ratio_source = "model_ratio:gbm_to_glm_ratio:saved-gbm:saved-glm"
            activated_saved_ratio_source = "model_ratio:gbm_to_glm_ratio:saved-gbm:active-glm"
            saved_gbm_source = "gbm:saved-gbm:predictions"
            saved_glm_source = "glm:saved-glm:predictions"
            active_ratio_source = "model_ratio:gbm_to_glm_ratio:active-gbm:active-glm"
            active_gbm_source = "gbm:active-gbm:predictions"
            active_glm_source = "glm:active-glm:predictions"
            favourites_path = root / "config" / "favourites.json"
            favourites_path.parent.mkdir(parents=True)
            favourites_path.write_text(
                json.dumps(
                    {
                        "version": 1,
                        "favourites": [
                            {
                                "id": "saved-ratio-view",
                                "name": "Saved ratio view",
                                "created_at": "2026-06-28T00:00:00Z",
                                "updated_at": "2026-06-28T00:00:00Z",
                                "view": {
                                    "version": 1,
                                    "source": "dataset",
                                    "x": "gbm_to_glm_ratio",
                                    "xSource": saved_ratio_source,
                                    "view": "chart",
                                    "sort": "alpha",
                                    "lowGroup": "0",
                                    "labels": "none",
                                    "bandWidth": "1",
                                    "quantileMode": "off",
                                    "dateBucket": "none",
                                    "transform": "none",
                                    "sigma": "0",
                                    "partialDependence": "none",
                                    "featureSort": "alpha",
                                    "expectedSort": "alpha",
                                    "actual": {"value": "actualNumerator", "sourceId": "dataset", "metricKind": "dataset"},
                                    "denominator": "__none__",
                                    "expectedSelections": [
                                        {"value": "glm_prediction", "sourceId": saved_glm_source, "metricKind": "prediction"},
                                        {"value": "gbm_prediction", "sourceId": saved_gbm_source, "metricKind": "prediction"},
                                    ],
                                    "filter": "Segment = 'B'",
                                    "filterSelectionMode": "single",
                                    "filterOperator": "and",
                                    "savedFilterRows": [],
                                },
                            },
                        ],
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
            base_url, server, thread = self.start_app(
                data_path,
                line_bar_favourites_path=favourites_path,
                tools=["line_bar", "gbm", "glm"],
                defaults={"x": "Age", "actual": "actualNumerator", "denominator": "__none__"},
            )
            try:
                assert sync_playwright is not None
                with sync_playwright() as playwright:
                    browser = playwright.chromium.launch()
                    page = browser.new_page(viewport={"width": 1280, "height": 800})
                    page_errors: list[str] = []
                    chart_requests: list[dict[str, Any]] = []
                    page.on(
                        "request",
                        lambda request: chart_requests.append(json.loads(request.post_data or "{}"))
                        if request.url.endswith("/api/chart") and request.method == "POST"
                        else None,
                    )
                    page.on("pageerror", lambda error: page_errors.append(str(error)))
                    try:
                        page.goto(base_url, wait_until="domcontentloaded")
                        page.locator("#datasetMeta").get_by_text("sample.csv").wait_for(timeout=10_000)
                        page.wait_for_function(
                            '() => document.querySelector("#lineBarGroupMeta")?.textContent.includes("groups")',
                            timeout=10_000,
                        )
                        if page.locator("#favouritesCollapseBtn").get_attribute("aria-expanded") == "false":
                            page.locator("#favouritesCollapseBtn").click()
                            page.wait_for_function(
                                '() => document.querySelector("#favouritesCollapseBtn")?.getAttribute("aria-expanded") === "true"',
                                timeout=10_000,
                            )
                        page.locator('.saved-favourite-option[data-favourite-id="saved-ratio-view"]').wait_for(timeout=10_000)
                        with page.expect_request(
                            lambda request: request.url.endswith("/api/chart") and "gbm_to_glm_ratio" in (request.post_data or ""),
                            timeout=10_000,
                        ) as chart_request_info:
                            page.locator('.saved-favourite-option[data-favourite-id="saved-ratio-view"]').click()
                        page.wait_for_function(
                            """
                            () => document.querySelector('#featureList .feature.active')?.dataset.sourceId === 'model_ratio:gbm_to_glm_ratio:active-gbm:active-glm'
                            """,
                            timeout=10_000,
                        )
                        request_body = json.loads(chart_request_info.value.post_data or "{}")
                        expected_state = page.evaluate(
                            """
                            () => [...document.querySelectorAll("#expectedList .feature.active")]
                              .map((button) => ({
                                value: button.dataset.value || "",
                                source: button.dataset.sourceId || "",
                              }))
                            """
                        )
                        status_text = page.locator("#status").text_content(timeout=10_000) or ""

                        self.assertEqual(request_body["x"], "gbm_to_glm_ratio")
                        self.assertEqual(request_body["xSource"], active_ratio_source)
                        self.assertEqual(request_body["responses"][1]["source"], active_glm_source)
                        self.assertEqual(request_body["responses"][2]["source"], active_gbm_source)
                        self.assertIn({"value": "glm_prediction", "source": active_glm_source}, expected_state)
                        self.assertIn({"value": "gbm_prediction", "source": active_gbm_source}, expected_state)
                        self.assertNotIn("missing x-axis source", status_text)
                        self.assertNotIn("cannot be used", status_text)

                        page.locator("#gbmTool").click()
                        page.locator('[data-gbm-tab="models"]').click(timeout=10_000)
                        page.locator("#gbmModelGrid .tabulator-row").filter(has_text="Saved GBM").click(timeout=10_000)
                        page.locator("#gbmActivateModelBtn").click(timeout=10_000)
                        with page.expect_response(
                            lambda response: (
                                response.url.endswith("/api/banding/suggestion")
                                and response.status == 200
                                and activated_saved_ratio_source in (response.request.post_data or "")
                            ),
                            timeout=10_000,
                        ) as banding_response_info:
                            page.locator("#lineBarTool").click()
                        banding_payload = banding_response_info.value.json()
                        page.wait_for_function(
                            """
                            ([sourceId]) => {
                              const active = document.querySelector('#featureList .feature.active');
                              const bandValue = document.querySelector('#bandValue')?.textContent || '';
                              return active?.dataset.sourceId === sourceId && bandValue && bandValue !== '(1)';
                            }
                            """,
                            arg=[activated_saved_ratio_source],
                            timeout=10_000,
                        )
                        activated_chart_request = next(
                            request for request in reversed(chart_requests)
                            if request.get("xSource") == activated_saved_ratio_source
                        )
                        status_text = page.locator("#status").text_content(timeout=10_000) or ""

                        self.assertEqual(banding_payload["source"], activated_saved_ratio_source)
                        self.assertGreater(banding_payload["band_suggestion"], 0)
                        self.assertNotEqual(banding_payload["band_suggestion"], 1)
                        self.assertEqual(activated_chart_request["x"], "gbm_to_glm_ratio")
                        self.assertEqual(activated_chart_request["xSource"], activated_saved_ratio_source)
                        self.assertEqual(activated_chart_request["filter"], "Segment = 'B'")
                        self.assertGreater(activated_chart_request["bandWidth"], 0)
                        self.assertNotEqual(activated_chart_request["bandWidth"], 1)
                        self.assertEqual(activated_chart_request["responses"][1]["source"], active_glm_source)
                        self.assertEqual(activated_chart_request["responses"][2]["source"], saved_gbm_source)
                        self.assertNotIn("Banding estimate failed", status_text)
                        self.assertEqual(page_errors, [])
                    finally:
                        browser.close()
            finally:
                server.should_exit = True
                thread.join(timeout=5)
                stop_persistent_glm_fit_worker()

    @staticmethod
    def start_app(
        data_path: Path,
        *,
        filters_path: Path | None = None,
        use_saved_filters: bool = False,
        kpis_path: Path | None = None,
        use_kpis: bool = False,
        features_path: Path | None = None,
        use_features: bool = True,
        line_bar_favourites_path: Path | None = None,
        token: str | None = None,
        defaults: dict[str, str] | None = None,
        tools: list[str] | None = None,
        buttons: bool = False,
        title_prefix: str | None = None,
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
            features_path=features_path,
            use_features=use_features,
            line_bar_favourites_path=line_bar_favourites_path,
            token=token,
            tools=tools,
            header_buttons=buttons,
            title_prefix=title_prefix,
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
    def write_tabulated_prediction_sidecar(path: Path, column_name: str, values: list[float]) -> None:
        rows = "\n  UNION ALL\n  ".join(
            f"SELECT {index + 1} AS __lucidum_row_id, {float(value)} AS {quote_ident(column_name)}"
            for index, value in enumerate(values)
        )
        con = duckdb.connect(database=":memory:")
        try:
            con.execute(
                f"""
COPY (
  {rows}
) TO {sql_literal(str(path))} (FORMAT PARQUET)
"""
            )
        finally:
            con.close()

    @staticmethod
    def write_gbm_prediction_model(
        store: GbmModelStore,
        model_id: str,
        label: str,
        created_at: str,
        predictions: list[float],
    ) -> None:
        model_dir = store.create_model_dir(model_id)
        store.write_json(
            model_dir / "manifest.json",
            {
                "model_id": model_id,
                "label": label,
                "created_at": created_at,
                "training_mode": "normal",
                "response_column": "actualNumerator",
                "offset_column": "denominator",
                "best_iteration": len(predictions),
                "training_rows": 2,
                "test_rows": 1,
                "scored_rows": len(predictions),
                "sample_column": "SAMPLE",
                "sample_source": "dataset",
            },
        )
        write_gbm_feature_config(
            store,
            model_id,
            [{"name": "Age", "kind": "integer", "include": True, "gain": 1.0, "mean_abs_shap": 0.2}],
        )
        store.write_json(model_dir / "parameters.json", {"objective": "gamma", "metric": "gamma", "num_iterations": len(predictions)})
        write_gbm_evaluation(store, model_id, {"training": {"gamma": predictions}, "test": {"gamma": predictions}})
        prediction_rows = "\n  UNION ALL\n  ".join(
            f"SELECT {index + 1} AS __lucidum_row_id, {float(value)} AS gbm_prediction"
            for index, value in enumerate(predictions)
        )
        con = duckdb.connect(database=":memory:")
        try:
            con.execute(
                f"""
COPY (
  {prediction_rows}
) TO {sql_literal(str(model_dir / "predictions.parquet"))} (FORMAT PARQUET)
"""
            )
            shap_rows = "\n  UNION ALL\n  ".join(
                f"SELECT {index + 1} AS __lucidum_row_id, {float(value) / 10.0} AS Age"
                for index, value in enumerate(predictions)
            )
            con.execute(
                f"""
COPY (
  {shap_rows}
) TO {sql_literal(str(model_dir / "shap_values.parquet"))} (FORMAT PARQUET)
"""
            )
            con.execute(
                f"""
COPY (
  SELECT 'Age' AS feature, 0.2 AS mean_abs_shap, 0.2 AS mean_shap, {len(predictions)} AS row_count
) TO {sql_literal(str(model_dir / "shap_summary.parquet"))} (FORMAT PARQUET)
"""
            )
        finally:
            con.close()

    @staticmethod
    def write_glm_prediction_model(
        store: GlmModelStore,
        model_id: str,
        label: str,
        created_at: str,
        predictions: list[float],
        *,
        formula: str = "actualNumerator ~ 1 + Age + Segment",
        family: str = "tweedie",
        family_parameter: float | str | None = 1.5,
        training_scope: str = "all",
        regularization: dict[str, Any] | None = None,
    ) -> None:
        model_dir = store.create_model_dir(model_id)
        diagnostics = {
            "aic": 123.45,
            "deviance": 67.89,
            "dispersion": 1.2,
            "na_in_fitted": 0,
            "training_rows": 2,
            "scored_rows": len(predictions),
        }
        manifest = {
            "model_id": model_id,
            "label": label,
            "created_at": created_at,
            "family": family,
            "link": "auto",
            "response_column": "actualNumerator",
            "denominator_column": "denominator",
            "training_scope": training_scope,
            "regularization": regularization or {"mode": "none"},
        }
        if family_parameter is not None:
            manifest["family_parameter"] = family_parameter
        store.write_json(model_dir / "manifest.json", manifest)
        store.write_json(model_dir / "diagnostics.json", diagnostics)
        (model_dir / "formula.txt").write_text(formula, encoding="utf-8")
        prediction_rows = "\n  UNION ALL\n  ".join(
            f"SELECT {index + 1} AS __lucidum_row_id, {float(value)} AS glm_prediction"
            for index, value in enumerate(predictions)
        )
        con = duckdb.connect(database=":memory:")
        try:
            con.execute(
                f"""
COPY (
  {prediction_rows}
) TO {sql_literal(str(model_dir / "predictions.parquet"))} (FORMAT PARQUET)
"""
            )
            con.execute(
                f"""
COPY (
  SELECT '(Intercept)' AS term, []::VARCHAR[] AS features, 0.1::DOUBLE AS estimate, 0.01::DOUBLE AS std_error, 10.0::DOUBLE AS statistic, 0.001::DOUBLE AS p_value, 0.08::DOUBLE AS ci_lower, 0.12::DOUBLE AS ci_upper
  UNION ALL
  SELECT 'Age' AS term, ['Age']::VARCHAR[] AS features, 0.2::DOUBLE AS estimate, 0.02::DOUBLE AS std_error, 10.0::DOUBLE AS statistic, 0.001::DOUBLE AS p_value, 0.16::DOUBLE AS ci_lower, 0.24::DOUBLE AS ci_upper
  UNION ALL
  SELECT 'Age:Segment[A]' AS term, ['Age', 'Segment']::VARCHAR[] AS features, 0.3::DOUBLE AS estimate, 0.03::DOUBLE AS std_error, 10.0::DOUBLE AS statistic, 0.001::DOUBLE AS p_value, 0.24::DOUBLE AS ci_lower, 0.36::DOUBLE AS ci_upper
) TO {sql_literal(str(model_dir / "coefficients.parquet"))} (FORMAT PARQUET)
"""
            )
        finally:
            con.close()

    @staticmethod
    def write_glm_tabulation_artifacts(store: GlmModelStore, model_id: str, *, include_segment: bool = False, offset: float = 0.0) -> None:
        model_dir = store.model_dir(model_id)
        (model_dir / "estimator.pkl").write_bytes(b"browser smoke estimator placeholder")
        tables = [
            {"table_id": "base", "label": "base", "index": 1, "features": [], "cell_count": 1, "skipped": False, "path": "tabulations/base.parquet", "min": offset, "max": offset},
            {"table_id": "Age", "label": "Age", "index": 2, "features": ["Age"], "cell_count": 3, "skipped": False, "path": "tabulations/Age.parquet", "min": offset, "max": offset + 1.0},
        ]
        if include_segment:
            tables.append(
                {"table_id": "Segment", "label": "Segment", "index": 3, "features": ["Segment"], "cell_count": 3, "skipped": False, "path": "tabulations/Segment.parquet", "min": offset - 0.2, "max": offset + 0.3}
            )
        store.write_json(
            store.artifact_path(model_id, "tabulation_manifest"),
            {
                "model_id": model_id,
                "status": "tabulated",
                "tables": tables,
                "warnings": [],
                "diagnostics": {
                    "mean_linear_error": round(offset / 10, 4),
                    "linear_sd_error": round(0.01 + offset / 20, 4),
                    "tabulated_row_count": 3,
                    "missing_tabulated_prediction_rows": int(offset * 10),
                },
            },
        )
        store.tabulations_dir(model_id).mkdir(parents=True, exist_ok=True)
        con = duckdb.connect(database=":memory:")
        try:
            con.execute(
                f"""
COPY (
  SELECT 'base' AS table_id, 'ok' AS status, {offset} AS tabulated_linear
) TO {sql_literal(str(store.tabulations_dir(model_id) / "base.parquet"))} (FORMAT PARQUET)
"""
            )
            con.execute(
                f"""
COPY (
  SELECT 30 AS Age, 'ok' AS status, {offset} AS tabulated_linear
  UNION ALL SELECT 40, 'ok', {offset + 0.5}
  UNION ALL SELECT 50, 'ok', {offset + 1.0}
) TO {sql_literal(str(store.tabulations_dir(model_id) / "Age.parquet"))} (FORMAT PARQUET)
"""
            )
            if include_segment:
                con.execute(
                    f"""
COPY (
  SELECT 'A' AS Segment, 'ok' AS status, {offset - 0.2} AS tabulated_linear
  UNION ALL SELECT 'B', 'ok', {offset + 0.1}
  UNION ALL SELECT 'C', 'ok', {offset + 0.3}
) TO {sql_literal(str(store.tabulations_dir(model_id) / "Segment.parquet"))} (FORMAT PARQUET)
"""
                )
        finally:
            con.close()

    @staticmethod
    def write_gbm_tabulation_artifacts(store: GbmModelStore, model_id: str, *, offset: float = 0.0, tabulated: bool = True, blocked: bool = False) -> None:
        model_dir = store.model_dir(model_id)
        if blocked:
            warnings = [
                "Tree 0 has 4 leaves; GBM tabulation supports only 2 or 3 leaf trees.",
                "Tree 0 uses 3 features; GBM tabulation supports only 1D and 2D trees.",
            ]
            store.write_json(
                store.artifact_path(model_id, "tabulation_manifest"),
                {
                    "model_id": model_id,
                    "model_kind": "gbm",
                    "model_ref": f"gbm:{model_id}",
                    "status": "not_tabulatable",
                    "tables": [],
                    "warnings": warnings,
                    "diagnostics": {"blocking_warnings": warnings},
                },
            )
            con = duckdb.connect(database=":memory:")
            try:
                con.execute(
                    f"""
COPY (
  SELECT 0 AS tree_index, 1 AS node_depth, '0-S0' AS node_index, '0-S1' AS left_child, '0-S2' AS right_child,
         NULL AS parent_index, 'Age' AS split_feature, 1.0 AS split_gain, '35' AS threshold,
         NULL AS threshold_label, '<=' AS decision_type, 'right' AS missing_direction, 'None' AS missing_type,
         0.0 AS value, 3.0 AS weight, 3 AS count
  UNION ALL SELECT 0, 2, '0-S1', '0-L0', '0-L1', '0-S0', 'Segment', 1.0, '0', 'A', '==', 'right', 'None', 0.0, 1.0, 1
  UNION ALL SELECT 0, 2, '0-S2', '0-L2', '0-L3', '0-S0', 'denominator', 1.0, '1', NULL, '<=', 'right', 'None', 0.0, 2.0, 2
  UNION ALL SELECT 0, 3, '0-L0', NULL, NULL, '0-S1', NULL, NULL, NULL, NULL, NULL, NULL, NULL, 1.0, 1.0, 1
  UNION ALL SELECT 0, 3, '0-L1', NULL, NULL, '0-S1', NULL, NULL, NULL, NULL, NULL, NULL, NULL, 2.0, 1.0, 1
  UNION ALL SELECT 0, 3, '0-L2', NULL, NULL, '0-S2', NULL, NULL, NULL, NULL, NULL, NULL, NULL, 3.0, 1.0, 1
  UNION ALL SELECT 0, 3, '0-L3', NULL, NULL, '0-S2', NULL, NULL, NULL, NULL, NULL, NULL, NULL, 4.0, 1.0, 1
) TO {sql_literal(str(model_dir / "tree_table.parquet"))} (FORMAT PARQUET)
"""
                )
            finally:
                con.close()
            return
        con = duckdb.connect(database=":memory:")
        try:
            con.execute(
                f"""
COPY (
  SELECT 0 AS tree_index, 1 AS node_depth, '0-S0' AS node_index, '0-L0' AS left_child, '0-L1' AS right_child,
         NULL AS parent_index, 'Age' AS split_feature, 1.0 AS split_gain, '35' AS threshold,
         NULL AS threshold_label, '<=' AS decision_type, 'left' AS missing_direction, 'None' AS missing_type,
         0.0 AS value, 3.0 AS weight, 3 AS count
  UNION ALL SELECT 0, 2, '0-L0', NULL, NULL, '0-S0', NULL, NULL, NULL, NULL, NULL, NULL, NULL, 0.0, 1.0, 1
  UNION ALL SELECT 0, 2, '0-L1', NULL, NULL, '0-S0', NULL, NULL, NULL, NULL, NULL, NULL, NULL, 0.8, 2.0, 2
) TO {sql_literal(str(model_dir / "tree_table.parquet"))} (FORMAT PARQUET)
"""
            )
            if not tabulated:
                return
            store.write_json(
                store.artifact_path(model_id, "tabulation_manifest"),
                {
                    "model_id": model_id,
                    "model_kind": "gbm",
                    "model_ref": f"gbm:{model_id}",
                    "status": "tabulated",
                    "tables": [
                        {"table_id": "base", "label": "base", "index": 1, "features": [], "cell_count": 1, "skipped": False, "path": "tabulations/base.parquet", "min": offset, "max": offset},
                        {"table_id": "Age", "label": "Age", "index": 2, "features": ["Age"], "cell_count": 3, "skipped": False, "path": "tabulations/Age.parquet", "min": offset, "max": offset + 0.8},
                    ],
                    "warnings": [],
                    "diagnostics": {
                        "mean_linear_error": round(offset / 10, 4),
                        "linear_sd_error": round(0.02 + offset / 20, 4),
                        "tabulated_row_count": 3,
                        "missing_tabulated_prediction_rows": int(offset * 10),
                    },
                },
            )
            store.tabulations_dir(model_id).mkdir(parents=True, exist_ok=True)
            con.execute(
                f"""
COPY (
  SELECT 'base' AS table_id, 'ok' AS status, {offset} AS tabulated_linear
) TO {sql_literal(str(store.tabulations_dir(model_id) / "base.parquet"))} (FORMAT PARQUET)
"""
            )
            con.execute(
                f"""
COPY (
  SELECT 30 AS Age, 'ok' AS status, {offset} AS tabulated_linear
  UNION ALL SELECT 40, 'ok', {offset + 0.4}
  UNION ALL SELECT 50, 'ok', {offset + 0.8}
) TO {sql_literal(str(store.tabulations_dir(model_id) / "Age.parquet"))} (FORMAT PARQUET)
"""
            )
        finally:
            con.close()

    @staticmethod
    def assert_static_asset(base_url: str, path: str, expected_content_type: str) -> None:
        with urlopen(f"{base_url}{path}", timeout=5) as response:
            assert response.status == 200
            assert expected_content_type in response.headers.get("content-type", "")

    def exercise_glm_tabulation_rebase(self, base_url: str) -> None:
        assert sync_playwright is not None
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch()
            page = browser.new_page(viewport={"width": 1280, "height": 800})
            page_errors: list[str] = []
            page.on("pageerror", lambda error: page_errors.append(str(error)))
            try:
                def right_click_tabulation_cell(script: str, arg: dict[str, object] | None = None) -> None:
                    point = page.wait_for_function(
                        """
                        ({ script, arg }) => {
                          const resolve = new Function(`return (${script});`)();
                          const element = resolve(arg || {});
                          if (!element) return null;
                          element.scrollIntoView({ block: "center", inline: "center" });
                          const rect = element.getBoundingClientRect();
                          return {
                            x: rect.left + Math.min(18, Math.max(4, rect.width / 2)),
                            y: rect.top + Math.min(12, Math.max(4, rect.height / 2)),
                          };
                        }
                        """,
                        arg={"script": script, "arg": arg or {}},
                        timeout=5_000,
                    ).json_value()
                    self.assertIsNotNone(point)
                    page.mouse.click(point["x"], point["y"], button="right")
                    page.wait_for_timeout(50)
                    if page.locator("#glmTabulationContextMenu:not([hidden])").count() == 0:
                        page.evaluate(
                            """
                            ({ script, arg, point }) => {
                              const resolve = new Function(`return (${script});`)();
                              const element = resolve(arg || {});
                              if (!element) return;
                              element.dispatchEvent(new MouseEvent("contextmenu", {
                                bubbles: true,
                                cancelable: true,
                                button: 2,
                                buttons: 2,
                                clientX: point.x,
                                clientY: point.y,
                              }));
                            }
                            """,
                            {"script": script, "arg": arg or {}, "point": point},
                        )

                def click_tabulation_menu_item(name: str) -> None:
                    page.locator("#glmTabulationContextMenu:not([hidden])").wait_for(timeout=5_000)
                    page.get_by_role("menuitem", name=name).click()

                page.goto(base_url, wait_until="domcontentloaded")
                page.locator("#lineBarTool.active").wait_for(timeout=10_000)
                page.wait_for_function(
                    """() => !document.querySelector("#glmTool")?.classList.contains("hidden")""",
                    timeout=10_000,
                )
                page.locator("#glmTool").click()
                page.locator("#modelToolWrap:not(.hidden) .glm-tool").wait_for(timeout=10_000)
                page.get_by_role("button", name="Tabulations").click()
                page.locator("#glmTabulationModelGrid .tabulator-row").first.wait_for(timeout=10_000)
                page.locator("#glmTabulationTableGrid .tabulator-row", has_text="Age × Segment").click()
                page.locator("#glmTabulationTable .tabulator-row").first.wait_for(timeout=10_000)
                page.wait_for_function(
                    """
                    () => document.querySelector("#glmTabulationCrosstab")?.value === "Segment"
                    """,
                    timeout=10_000,
                )
                page.locator('[data-glm-tabulation-view="plot"]').click()
                page.locator("#glmTabulationPlot canvas").first.wait_for(timeout=10_000)
                page.wait_for_function(
                    """
                    () => {
                      const notice = document.querySelector("#glmTabulationNotice")?.textContent || "";
                      return !notice.includes("Choose a feature crosstab");
                    }
                    """,
                    timeout=10_000,
                )
                page.locator('[data-glm-tabulation-view="table"]').click()
                page.locator("#glmTabulationTable .tabulator-row").first.wait_for(timeout=10_000)
                page.locator('[data-glm-tabulation-scale="exp"]').click()
                page.wait_for_function(
                    """
                    () => {
                      const headers = [...document.querySelectorAll("#glmTabulationTable .tabulator-col")];
                      return headers.some((header) => header.textContent.trim() === "B")
                        && document.querySelectorAll("#glmTabulationTable .tabulator-row").length > 0;
                    }
                    """,
                    timeout=10_000,
                )
                right_click_tabulation_cell(
                    """
                    () => document.querySelector('#glmTabulationTable .tabulator-row .tabulator-cell[tabulator-field="Age"]')
                    """
                )
                page.wait_for_timeout(150)
                self.assertEqual(page.locator("#glmTabulationContextMenu:not([hidden])").count(), 0)
                before = page.evaluate(
                    """
                    () => {
                      const headers = [...document.querySelectorAll("#glmTabulationTable .tabulator-col")];
                      const bField = headers.find((header) => header.textContent.trim() === "B")?.getAttribute("tabulator-field");
                      if (!bField) return null;
                      for (const row of document.querySelectorAll("#glmTabulationTable .tabulator-row")) {
                        const age = row.querySelector('.tabulator-cell[tabulator-field="Age"]')?.textContent.trim() || "";
                        const cell = row.querySelector(`.tabulator-cell[tabulator-field="${bField}"]`);
                        const text = cell?.textContent.trim() || "";
                        if (cell && text && text !== "NA" && text !== "1.0000") {
                          return { age, text, bField };
                        }
                      }
                      return null;
                    }
                    """
                )
                self.assertIsNotNone(before)
                right_click_tabulation_cell(
                    """
                    ({ age, bField }) => {
                      const rows = [...document.querySelectorAll("#glmTabulationTable .tabulator-row")];
                      const row = rows.find((candidate) => candidate.querySelector('.tabulator-cell[tabulator-field="Age"]')?.textContent.trim() === age)
                        || rows.find((candidate) => candidate.querySelector(`.tabulator-cell[tabulator-field="${bField}"]`)?.textContent.trim() === "1.0000");
                      return [...(row?.querySelectorAll(".tabulator-cell") || [])]
                        .find((cell) => cell.getAttribute("tabulator-field") === bField) || null;
                    }
                    """,
                    before,
                )
                click_tabulation_menu_item("Rebase Segment=B slice to this cell; offset Segment table")
                page.wait_for_function(
                    """
                    ({ age }) => {
                      const headers = [...document.querySelectorAll("#glmTabulationTable .tabulator-col")];
                      const bField = headers.find((header) => header.textContent.trim() === "B")?.getAttribute("tabulator-field");
                      const row = [...document.querySelectorAll("#glmTabulationTable .tabulator-row")]
                        .find((candidate) => candidate.querySelector('.tabulator-cell[tabulator-field="Age"]')?.textContent.trim() === age);
                      const text = row?.querySelector(`.tabulator-cell[tabulator-field="${bField}"]`)?.textContent.trim() || "";
                      return text === "1.0000"
                        && document.querySelector("#glmTabulationDiagnostics")?.textContent.includes("Rebased");
                    }
                    """,
                    arg={"age": before["age"]},
                    timeout=15_000,
                )
                right_click_tabulation_cell(
                    """
                    ({ age, bField }) => {
                      const row = [...document.querySelectorAll("#glmTabulationTable .tabulator-row")]
                        .find((candidate) => candidate.querySelector('.tabulator-cell[tabulator-field="Age"]')?.textContent.trim() === age);
                      return [...(row?.querySelectorAll(".tabulator-cell") || [])]
                        .find((cell) => cell.getAttribute("tabulator-field") === bField) || null;
                    }
                    """,
                    before,
                )
                click_tabulation_menu_item("Reset rebase")
                page.wait_for_function(
                    """
                    ({ age, text }) => {
                      const headers = [...document.querySelectorAll("#glmTabulationTable .tabulator-col")];
                      const bField = headers.find((header) => header.textContent.trim() === "B")?.getAttribute("tabulator-field");
                      const rows = [...document.querySelectorAll("#glmTabulationTable .tabulator-row")];
                      const row = rows.find((candidate) => candidate.querySelector('.tabulator-cell[tabulator-field="Age"]')?.textContent.trim() === age)
                        || rows.find((candidate) => candidate.querySelector(`.tabulator-cell[tabulator-field="${bField}"]`)?.textContent.trim() === text);
                      const current = row?.querySelector(`.tabulator-cell[tabulator-field="${bField}"]`)?.textContent.trim() || "";
                      return current === text
                        && !document.querySelector("#glmTabulationDiagnostics")?.textContent.includes("Rebased");
                    }
                    """,
                    arg=before,
                    timeout=15_000,
                )
                selected_segment = page.evaluate(
                    """
                    () => {
                      for (const row of document.querySelectorAll("#glmTabulationTableGrid .tabulator-row")) {
                        const name = row.querySelector('.tabulator-cell[tabulator-field="table_name"]')?.textContent.trim() || "";
                        if (name === "Segment") {
                          row.click();
                          return true;
                        }
                      }
                      return false;
                    }
                    """
                )
                self.assertTrue(selected_segment)
                page.wait_for_function(
                    """
                    () => {
                      const headers = [...document.querySelectorAll("#glmTabulationTable .tabulator-col")];
                      return document.querySelector("#glmTabulationCrosstab")?.value === ""
                        && headers.some((header) => header.textContent.trim() === "Segment")
                        && document.querySelectorAll("#glmTabulationTable .tabulator-row").length > 0;
                    }
                    """,
                    timeout=10_000,
                )
                one_way_before = page.evaluate(
                    """
                    () => {
                      const row = [...document.querySelectorAll("#glmTabulationTable .tabulator-row")]
                        .find((candidate) => candidate.querySelector('.tabulator-cell[tabulator-field="Segment"]')?.textContent.trim() === "B");
                      const cell = row?.querySelector(".tabulator-cell.glm-tabulation-rebase-cell") || null;
                      const text = cell?.textContent.trim() || "";
                      if (cell && text && text !== "NA" && text !== "1.0000") {
                        return { text, segment: "B" };
                      }
                      return null;
                    }
                    """
                )
                self.assertIsNotNone(one_way_before)
                right_click_tabulation_cell(
                    """
                    ({ segment }) => {
                      const row = [...document.querySelectorAll("#glmTabulationTable .tabulator-row")]
                        .find((candidate) => candidate.querySelector('.tabulator-cell[tabulator-field="Segment"]')?.textContent.trim() === segment);
                      return row?.querySelector(".tabulator-cell.glm-tabulation-rebase-cell") || null;
                    }
                    """,
                    one_way_before,
                )
                click_tabulation_menu_item("Rebase whole table to this cell; offset base")
                page.wait_for_function(
                    """
                    () => {
                      const row = [...document.querySelectorAll("#glmTabulationTable .tabulator-row")]
                        .find((candidate) => candidate.querySelector('.tabulator-cell[tabulator-field="Segment"]')?.textContent.trim() === "B");
                      const text = row?.querySelector(".tabulator-cell.glm-tabulation-rebase-cell")?.textContent.trim() || "";
                      return text === "1.0000"
                        && document.querySelector("#glmTabulationDiagnostics")?.textContent.includes("Rebased");
                    }
                    """,
                    timeout=15_000,
                )
                page.locator("#glmTabulationTable .tabulator-cell.glm-tabulation-rebase-cell").first.click(button="right", force=True)
                click_tabulation_menu_item("Reset rebase")
                page.wait_for_function(
                    """
                    ({ text, segment }) => {
                      const row = [...document.querySelectorAll("#glmTabulationTable .tabulator-row")]
                        .find((candidate) => candidate.querySelector('.tabulator-cell[tabulator-field="Segment"]')?.textContent.trim() === segment);
                      const current = row?.querySelector(".tabulator-cell.glm-tabulation-rebase-cell")?.textContent.trim() || "";
                      return current === text
                        && !document.querySelector("#glmTabulationDiagnostics")?.textContent.includes("Rebased");
                    }
                    """,
                    arg=one_way_before,
                    timeout=15_000,
                )
                self.assertFalse(page_errors)
            finally:
                browser.close()

    def exercise_sidebar_accordion(self, base_url: str) -> None:
        assert sync_playwright is not None
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch()
            page = browser.new_page(viewport={"width": 1280, "height": 800})
            page_errors: list[str] = []
            page.on("pageerror", lambda error: page_errors.append(str(error)))

            section_buttons = {
                "favourites": "#favouritesCollapseBtn",
                "kpi": "#kpiCollapseBtn",
                "gbm": "#gbmModelCollapseBtn",
                "glm": "#glmModelCollapseBtn",
                "filter": "#filterCollapseBtn",
            }

            section_bodies = {
                "favourites": "#favouritesSelect",
                "kpi": "#kpiSelect",
                "gbm": "#gbmModelSelect",
                "glm": "#glmModelSelect",
                "filter": "#savedFilterSelect",
            }

            def wait_accordion_state(open_section: str | None) -> None:
                page.wait_for_function(
                    """
                    ({ openSection, buttons }) => Object.entries(buttons).every(([section, selector]) => {
                      const button = document.querySelector(selector);
                      return button && button.getAttribute("aria-expanded") === String(section === openSection);
                    })
                    """,
                    arg={"openSection": open_section, "buttons": section_buttons},
                    timeout=10_000,
                )
                for section, selector in section_buttons.items():
                    self.assertEqual(page.locator(selector).get_attribute("aria-expanded"), str(section == open_section).lower())
                for section, selector in section_bodies.items():
                    self.assertEqual(page.locator(selector).is_visible(), section == open_section)
                self.assertEqual(
                    page.locator("#sidebarFavouritesControls").evaluate("node => getComputedStyle(node).display !== 'none'"),
                    open_section == "favourites",
                )
                page.locator(".sidebar-metric-section").wait_for(state="visible", timeout=10_000)
                self.assertTrue(page.locator(".sidebar-metric-section").is_visible())
                self.assertTrue(page.locator("#actualNumerator").is_visible())
                self.assertTrue(page.locator("#denominator").is_visible())
                metric_layout = page.evaluate(
                    """
                    () => {
                      const favourites = document.querySelector(".sidebar-favourites-section").getBoundingClientRect();
                      const metric = document.querySelector(".sidebar-metric-section").getBoundingClientRect();
                      return { favouritesTop: favourites.top, metricBottom: metric.bottom };
                    }
                    """
                )
                self.assertLessEqual(metric_layout["metricBottom"], metric_layout["favouritesTop"] + 1)
                icon_layout = page.evaluate(
                    """
                    (buttons) => {
                      const icons = Object.values(buttons)
                        .map((selector) => document.querySelector(`${selector} .filter-collapse-icon`))
                        .filter(Boolean);
                      const rects = icons.map((icon) => icon.getBoundingClientRect());
                      return {
                        count: rects.length,
                        leftSpread: Math.max(...rects.map((rect) => Math.round(rect.left))) - Math.min(...rects.map((rect) => Math.round(rect.left))),
                        allSameSize: rects.every((rect) => Math.round(rect.width) === 14 && Math.round(rect.height) === 14),
                      };
                    }
                    """,
                    arg=section_buttons,
                )
                self.assertEqual(icon_layout, {"count": 5, "leftSpread": 0, "allSameSize": True})

            def assert_sidebar_headers_visible() -> None:
                for selector in section_buttons.values():
                    self.assertTrue(page.locator(selector).is_visible())
                self.assertEqual(page.locator("#sidebarGbmResizer").count(), 0)
                self.assertEqual(page.locator("#sidebarGlmResizer").count(), 0)
                self.assertEqual(page.locator("#sidebarFilterResizer").count(), 0)
                self.assertTrue(page.locator("#sidebarResizer").is_visible())
                self.assertTrue(
                    page.evaluate(
                        """
                        () => {
                          const sections = [...document.querySelectorAll("[data-sidebar-section]")].map((section) => section.dataset.sidebarSection);
                          return sections.join("|") === "favourites|kpi|gbm|glm|filter";
                        }
                        """
                    )
                )

            def assert_tool_order(expected_order: str) -> None:
                page.wait_for_function(
                    """
                    (expected) => [...document.querySelectorAll(".tool-option:not(.hidden)")]
                      .map((button) => button.dataset.tool)
                      .join("|") === expected
                    """,
                    arg=expected_order,
                    timeout=10_000,
                )

            try:
                page.goto(f"{base_url}?tool=line_bar", wait_until="domcontentloaded")
                page.locator("#datasetMeta").get_by_text("sample.csv").wait_for(timeout=10_000)
                page.locator("#lineBarTool.active").wait_for(timeout=10_000)
                assert_tool_order("column_profile|line_bar|uk_map|glm|gbm")

                page.goto(base_url, wait_until="domcontentloaded")
                page.locator("#datasetMeta").get_by_text("sample.csv").wait_for(timeout=10_000)
                page.locator("#profileTool.active").wait_for(timeout=10_000)
                assert_tool_order("column_profile|line_bar|uk_map|glm|gbm")
                page.locator("#profileWrap:not(.hidden) .profile-table").wait_for(timeout=10_000)
                page.locator("#gbmSidebarPanel").wait_for(timeout=10_000)
                tool_rail_state = page.evaluate(
                    """
                    () => {
                      const buttons = [...document.querySelectorAll("#toolSelectorSection .tool-option:not(.hidden)")];
                      const selector = document.querySelector("#toolSelectorSection .tool-selector");
                      const selectorStyle = getComputedStyle(selector);
                      const sidebarRect = document.querySelector("#appSidebar").getBoundingClientRect();
                      const railRect = document.querySelector("#toolSelectorSection").getBoundingClientRect();
                      const paneRect = document.querySelector("#sidebarControlPane").getBoundingClientRect();
                      const tops = buttons.map((button) => Math.round(button.getBoundingClientRect().top));
                      const lefts = buttons.map((button) => Math.round(button.getBoundingClientRect().left));
                      return {
                        count: buttons.length,
                        selectorDisplay: selectorStyle.display,
                        gap: selectorStyle.gap,
                        overflowX: selectorStyle.overflowX,
                        overflowY: selectorStyle.overflowY,
                        selectorMatchesSidebar: selectorStyle.backgroundColor === getComputedStyle(document.querySelector("#appSidebar")).backgroundColor,
                        vertical: new Set(lefts).size === 1 && new Set(tops).size === buttons.length,
                        railWidth: Math.round(railRect.width),
                        railAtSidebarLeft: Math.round(railRect.left) === Math.round(sidebarRect.left),
                        paneRightOfRail: Math.round(paneRect.left) >= Math.round(railRect.right),
                        allButtonsVisible: buttons.every((button) => {
                          const rect = button.getBoundingClientRect();
                          return rect.width > 0 && rect.height > 0 && getComputedStyle(button).display !== "none";
                        }),
                        allButtonsSquare: buttons.every((button) => {
                          const rect = button.getBoundingClientRect();
                          return Math.round(rect.width) === 36 && Math.round(rect.height) === 36;
                        }),
                        allButtonsBorderless: buttons.every((button) => getComputedStyle(button).borderTopWidth === "0px"),
                        allButtonsTransparentWithActiveAccent: buttons.every((button) => {
                          const style = getComputedStyle(button);
                          const expectedColor = button.classList.contains("active") ? "rgb(34, 118, 210)" : "rgb(102, 112, 133)";
                          return style.backgroundColor === "rgba(0, 0, 0, 0)" && style.color === expectedColor;
                        }),
                        allButtonsFullOpacity: buttons.every((button) => getComputedStyle(button).opacity === "1"),
                        allIconsLarge: buttons.every((button) => {
                          const icon = button.querySelector(".tool-icon");
                          const renderedIcon = button.querySelector(".tool-icon svg, .tool-icon img");
                          const iconRect = icon?.getBoundingClientRect();
                          const renderedRect = renderedIcon?.getBoundingClientRect();
                          return Math.round(iconRect?.width || 0) === 30
                            && Math.round(iconRect?.height || 0) === 30
                            && Math.round(renderedRect?.width || 0) >= 28
                            && Math.round(renderedRect?.height || 0) >= 28;
                        }),
                        labelsHidden: buttons.every((button) => {
                          const label = button.querySelector(".tool-label");
                          if (!label) return false;
                          const style = getComputedStyle(label);
                          return style.position === "absolute" && style.width === "1px" && style.overflow === "hidden";
                        }),
                      };
                    }
                    """
                )
                self.assertEqual(
                    tool_rail_state,
                    {
                        "count": 5,
                        "selectorDisplay": "grid",
                        "gap": "12px",
                        "overflowX": "visible",
                        "overflowY": "visible",
                        "selectorMatchesSidebar": True,
                        "vertical": True,
                        "railWidth": 50,
                        "railAtSidebarLeft": True,
                        "paneRightOfRail": True,
                        "allButtonsVisible": True,
                        "allButtonsSquare": True,
                        "allButtonsBorderless": True,
                        "allButtonsTransparentWithActiveAccent": True,
                        "allButtonsFullOpacity": True,
                        "allIconsLarge": True,
                        "labelsHidden": True,
                    },
                )
                self.assert_tool_button_tooltip_right_of_icon(page, "#profileTool", "Column profile")
                assert_sidebar_headers_visible()
                wait_accordion_state("favourites")

                page.locator("#favouritesCollapseBtn").click()
                wait_accordion_state(None)
                page.locator("#favouritesCollapseBtn").click()
                wait_accordion_state("favourites")
                page.locator("#kpiCollapseBtn").click()
                wait_accordion_state("kpi")
                page.locator("#glmModelCollapseBtn").click()
                wait_accordion_state("glm")
                page.locator("#glmModelCollapseBtn").click()
                wait_accordion_state(None)
                page.locator("#filterCollapseBtn").click()
                wait_accordion_state("filter")

                page.locator("#lineBarTool").click()
                page.locator("#chart:not(.hidden)").wait_for(timeout=10_000)
                self.assertTrue(page.locator("#lineBarTool").is_visible())
                self.assertTrue(page.locator("#profileTool").is_visible())
                self.assertTrue(page.locator("#ukMapTool").is_visible())
                assert_sidebar_headers_visible()
                wait_accordion_state("filter")
                page.locator("#lineBarTool.active").click()
                self.assertEqual(page.locator("#sidebarToggleBtn").get_attribute("aria-expanded"), "false")
                self.assertFalse(page.locator(".sidebar-metric-section").is_visible())

                page.locator("#ukMapTool").click()
                page.locator("#ukMap:not(.hidden)").wait_for(timeout=20_000)
                self.assertEqual(page.locator("#sidebarToggleBtn").get_attribute("aria-expanded"), "false")
                self.assertFalse(page.locator(".sidebar-metric-section").is_visible())

                page.locator("#ukMapTool.active").click()
                self.assertEqual(page.locator("#sidebarToggleBtn").get_attribute("aria-expanded"), "true")
                assert_sidebar_headers_visible()
                wait_accordion_state("filter")

                page.locator("#glmTool").click()
                page.locator("#modelToolWrap:not(.hidden) .glm-tool").wait_for(timeout=10_000)
                assert_sidebar_headers_visible()
                wait_accordion_state("filter")

                page.locator("#gbmTool").click()
                page.locator("#modelToolWrap:not(.hidden) .gbm-tool").wait_for(timeout=10_000)
                assert_sidebar_headers_visible()
                wait_accordion_state("filter")

                page.locator("#filterCollapseBtn").click()
                wait_accordion_state(None)
                page.locator("#sidebarToggleBtn").click()
                self.assertEqual(page.locator("#sidebarToggleBtn").get_attribute("aria-expanded"), "false")
                self.assertFalse(page.locator("#sidebarResizer").is_visible())
                self.assertFalse(page.locator("#favouritesCollapseBtn").is_visible())
                self.assertFalse(page.locator("#kpiCollapseBtn").is_visible())
                page.locator("#sidebarToggleBtn").click()
                self.assertEqual(page.locator("#sidebarToggleBtn").get_attribute("aria-expanded"), "true")
                assert_sidebar_headers_visible()
                wait_accordion_state(None)
                self.assertEqual(page_errors, [])
            finally:
                browser.close()

    def exercise_default_sidebar_hides_model_accordions(self, base_url: str) -> None:
        assert sync_playwright is not None
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch()
            page = browser.new_page(viewport={"width": 1280, "height": 800})
            page_errors: list[str] = []
            page.on("pageerror", lambda error: page_errors.append(str(error)))
            try:
                page.goto(base_url, wait_until="domcontentloaded")
                page.locator("#datasetMeta").get_by_text("sample.csv").wait_for(timeout=10_000)
                page.locator("#lineBarTool.active").wait_for(timeout=10_000)
                page.locator("#chart:not(.hidden)").wait_for(timeout=10_000)
                self.assertFalse(page.locator("#gbmSidebarPanel").is_visible())
                self.assertFalse(page.locator("#glmSidebarPanel").is_visible())
                self.assertFalse(page.locator("#gbmModelCollapseBtn").is_visible())
                self.assertFalse(page.locator("#glmModelCollapseBtn").is_visible())
                self.assertTrue(page.locator("#filterCollapseBtn").is_visible())
                self.assertTrue(page.locator("#favouritesCollapseBtn").is_visible())
                self.assertTrue(page.locator("#kpiCollapseBtn").is_visible())
                self.assertEqual(page.locator("#filterCollapseBtn").get_attribute("aria-expanded"), "false")
                self.assertTrue(
                    page.evaluate(
                        """
                        () => {
                          const sections = [...document.querySelectorAll("[data-sidebar-section]")]
                            .filter((section) => section.offsetParent !== null)
                            .map((section) => section.dataset.sidebarSection);
                          return sections.join("|") === "favourites|kpi|filter";
                        }
                        """
                    )
                )
                page.locator("#filterCollapseBtn").click()
                self.assertEqual(page.locator("#filterCollapseBtn").get_attribute("aria-expanded"), "true")
                self.assertTrue(page.locator("#savedFilterSelect").is_visible())
                page.locator("#filterCollapseBtn").click()
                self.assertEqual(page.locator("#filterCollapseBtn").get_attribute("aria-expanded"), "false")
                self.assertFalse(page.locator("#savedFilterSelect").is_visible())
                self.assertEqual(page_errors, [])
            finally:
                browser.close()

    def exercise_gbm_profile_cache_and_model_chart_refresh(self, base_url: str) -> None:
        assert sync_playwright is not None
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch()
            page = browser.new_page(viewport={"width": 1280, "height": 800})
            page_errors: list[str] = []
            dataset_viewer_requests = 0
            profile_requests = 0
            profile_detail_requests = 0
            chart_requests = 0

            page.on("pageerror", lambda error: page_errors.append(str(error)))

            def count_request(request: object) -> None:
                nonlocal profile_requests, profile_detail_requests, chart_requests
                url = request.url
                if url.endswith("/api/column-profile/summary"):
                    profile_requests += 1
                elif url.endswith("/api/column-profile/detail"):
                    profile_detail_requests += 1
                elif url.endswith("/api/chart"):
                    chart_requests += 1

            page.on("request", count_request)

            def header_top(selector: str) -> float:
                box = page.locator(selector).bounding_box()
                self.assertIsNotNone(box)
                assert box is not None
                return float(box["y"])

            def assert_header_top_stable(selector: str, before: float) -> None:
                page.wait_for_function(
                    """
                    ({ selector, before }) => {
                      const rect = document.querySelector(selector)?.getBoundingClientRect();
                      return Boolean(rect) && Math.abs(rect.top - before) <= 1;
                    }
                    """,
                    arg={"selector": selector, "before": before},
                    timeout=10_000,
                )
                self.assertLessEqual(abs(header_top(selector) - before), 1)

            def open_sidebar_section(button_selector: str) -> None:
                if page.locator(button_selector).get_attribute("aria-expanded") != "true":
                    page.locator(button_selector).click()
                page.wait_for_function(
                    """
                    (selector) => document.querySelector(selector)?.getAttribute("aria-expanded") === "true"
                    """,
                    arg=button_selector,
                    timeout=10_000,
                )

            try:
                page.goto(f"{base_url}?tool=column_profile", wait_until="domcontentloaded")
                page.locator("#profileTool.active").wait_for(timeout=10_000)
                page.locator("#profileWrap:not(.hidden) .profile-table").wait_for(timeout=10_000)
                page.locator("#profileDetailTitle").wait_for(timeout=10_000)
                page.locator("#gbmModelSelect").wait_for(state="attached", timeout=10_000)
                page.locator("#glmModelSelect").wait_for(state="attached", timeout=10_000)
                if page.locator("#favouritesCollapseBtn").get_attribute("aria-expanded") == "true":
                    page.locator("#favouritesCollapseBtn").click()
                    page.wait_for_function(
                        '() => document.querySelector("#favouritesCollapseBtn")?.getAttribute("aria-expanded") === "false"',
                        timeout=10_000,
                    )
                if page.locator("#glmModelCollapseBtn").get_attribute("aria-expanded") == "true":
                    page.locator("#glmModelCollapseBtn").click()
                    self.assertEqual(page.locator("#glmModelCollapseBtn").get_attribute("aria-expanded"), "false")
                glm_header_top = header_top("#glmModelCollapseBtn")
                page.locator("#glmModelCollapseBtn").click()
                self.assertEqual(page.locator("#glmModelCollapseBtn").get_attribute("aria-expanded"), "true")
                assert_header_top_stable("#glmModelCollapseBtn", glm_header_top)
                open_sidebar_section("#gbmModelCollapseBtn")
                startup_model_details = page.evaluate(
                    """
                    () => Object.fromEntries(
                      [...document.querySelectorAll("#gbmModelSelect [data-gbm-model-id]")]
                        .map((button) => [
                          button.dataset.gbmModelId,
                          button.querySelector(".gbm-model-detail")?.textContent || "",
                        ])
                    )
                    """
                )
                self.assertEqual(
                    startup_model_details["browser-smoke-model"],
                    "gamma · iter 3 · train 0.31 · test 0.31",
                )
                self.assertEqual(
                    startup_model_details["browser-smoke-model-2"],
                    "gamma · iter 3 · train 0.61 · test 0.61",
                )

                def glm_builder_state() -> dict[str, Any]:
                    return page.evaluate(
                        """
                        () => {
                          const editorNode = document.querySelector("#glmFormulaEditor");
                          const aceEditor = editorNode?.env?.editor || null;
                          const formula = aceEditor
                            ? aceEditor.getValue()
                            : (document.querySelector("#glmFormulaText")?.value || "");
                          return {
                            formula,
                            family: document.querySelector("#glmFamilySelect")?.value || "",
                            familyParameter: document.querySelector("#glmFamilyParameter")?.value || "",
                            familyParameterDisabled: document.querySelector("#glmFamilyParameter")?.disabled ?? null,
                            scope: document.querySelector("[data-glm-scope].active")?.dataset?.glmScope || "",
                            regularizationMode: document.querySelector("#glmRegularizationMode")?.value || "",
                            regularizationMix: document.querySelector("#glmRegularizationMix")?.value || "",
                            regularizationAlpha: document.querySelector("#glmRegularizationAlpha")?.value || "",
                            regularizationMixDisabled: document.querySelector("#glmRegularizationMix")?.disabled ?? null,
                            regularizationAlphaDisabled: document.querySelector("#glmRegularizationAlpha")?.disabled ?? null,
                          };
                        }
                        """
                    )

                def wait_for_glm_builder_state(expected: dict[str, Any]) -> dict[str, Any]:
                    page.wait_for_function(
                        """
                        (expected) => {
                          const editorNode = document.querySelector("#glmFormulaEditor");
                          const aceEditor = editorNode?.env?.editor || null;
                          const state = {
                            formula: aceEditor ? aceEditor.getValue() : (document.querySelector("#glmFormulaText")?.value || ""),
                            family: document.querySelector("#glmFamilySelect")?.value || "",
                            familyParameter: document.querySelector("#glmFamilyParameter")?.value || "",
                            familyParameterDisabled: document.querySelector("#glmFamilyParameter")?.disabled ?? null,
                            scope: document.querySelector("[data-glm-scope].active")?.dataset?.glmScope || "",
                            regularizationMode: document.querySelector("#glmRegularizationMode")?.value || "",
                            regularizationMix: document.querySelector("#glmRegularizationMix")?.value || "",
                            regularizationAlpha: document.querySelector("#glmRegularizationAlpha")?.value || "",
                            regularizationMixDisabled: document.querySelector("#glmRegularizationMix")?.disabled ?? null,
                            regularizationAlphaDisabled: document.querySelector("#glmRegularizationAlpha")?.disabled ?? null,
                          };
                          return Object.entries(expected).every(([key, value]) => state[key] === value);
                        }
                        """,
                        arg=expected,
                        timeout=10_000,
                    )
                    return glm_builder_state()

                def set_glm_builder_draft(draft: dict[str, Any]) -> None:
                    page.evaluate(
                        """
                        (draft) => {
                          const change = (node) => node?.dispatchEvent(new Event("change", { bubbles: true }));
                          const input = (node) => node?.dispatchEvent(new Event("input", { bubbles: true }));
                          const editorNode = document.querySelector("#glmFormulaEditor");
                          const aceEditor = editorNode?.env?.editor || null;
                          if (aceEditor) aceEditor.setValue(draft.formula, -1);
                          const fallback = document.querySelector("#glmFormulaText");
                          if (fallback) {
                            fallback.value = draft.formula;
                            input(fallback);
                          }
                          const family = document.querySelector("#glmFamilySelect");
                          if (family) {
                            family.value = draft.family;
                            change(family);
                          }
                          const familyParameter = document.querySelector("#glmFamilyParameter");
                          if (familyParameter) familyParameter.value = draft.familyParameter;
                          document.querySelector(`[data-glm-scope="${draft.scope}"]`)?.click();
                          const mode = document.querySelector("#glmRegularizationMode");
                          if (mode) {
                            mode.value = draft.regularizationMode;
                            change(mode);
                          }
                          const mix = document.querySelector("#glmRegularizationMix");
                          if (mix) {
                            mix.value = draft.regularizationMix;
                            change(mix);
                          }
                          const alpha = document.querySelector("#glmRegularizationAlpha");
                          if (alpha) alpha.value = draft.regularizationAlpha;
                        }
                        """,
                        draft,
                    )

                profile_requests_before = profile_requests
                profile_detail_requests_before = profile_detail_requests
                open_sidebar_section("#gbmModelCollapseBtn")
                page.locator('#gbmModelSelect [data-gbm-model-id="browser-smoke-model-2"]').click()
                page.locator("#gbmModelSelectedMeta", has_text="Second smoke model").wait_for(timeout=10_000)
                page.wait_for_function(
                    """
                    () => document.querySelector('#gbmModelSelect [data-gbm-model-id="browser-smoke-model-2"]')
                      ?.classList.contains("active")
                    """,
                    timeout=10_000,
                )
                page.wait_for_timeout(250)
                self.assertEqual(profile_requests, profile_requests_before)
                self.assertEqual(profile_detail_requests, profile_detail_requests_before)
                open_sidebar_section("#gbmModelCollapseBtn")
                page.locator('#gbmModelSelect [data-gbm-model-id="browser-smoke-model"]').click()
                page.locator("#gbmModelSelectedMeta", has_text="Browser smoke model").wait_for(timeout=10_000)
                page.wait_for_timeout(250)
                self.assertEqual(profile_requests, profile_requests_before)
                self.assertEqual(profile_detail_requests, profile_detail_requests_before)

                page.locator("#glmTool").click()
                page.locator("#modelToolWrap:not(.hidden) .glm-tool").wait_for(timeout=10_000)
                page.get_by_role("button", name="Formula builder").click()
                self.assertEqual(
                    wait_for_glm_builder_state(
                        {
                            "formula": "actualNumerator ~ 1 + Age + Segment",
                            "family": "tweedie",
                            "familyParameter": "1.5",
                            "familyParameterDisabled": False,
                            "scope": "all",
                            "regularizationMode": "none",
                            "regularizationMix": "0.5",
                            "regularizationAlpha": "0.01",
                            "regularizationMixDisabled": True,
                            "regularizationAlphaDisabled": True,
                        }
                    ),
                    {
                        "formula": "actualNumerator ~ 1 + Age + Segment",
                        "family": "tweedie",
                        "familyParameter": "1.5",
                        "familyParameterDisabled": False,
                        "scope": "all",
                        "regularizationMode": "none",
                        "regularizationMix": "0.5",
                        "regularizationAlpha": "0.01",
                        "regularizationMixDisabled": True,
                        "regularizationAlphaDisabled": True,
                    },
                )

                page.evaluate(
                    """
                    () => {
                      const editorNode = document.querySelector("#glmFormulaEditor");
                      const aceEditor = editorNode?.env?.editor || null;
                      if (aceEditor) {
                        aceEditor.setValue("", -1);
                        aceEditor.focus();
                      }
                    }
                    """
                )
                page.keyboard.type("Ag")
                page.locator(".glm-formula-autocomplete-row", has_text="Age").wait_for(timeout=10_000)
                page.locator(".glm-formula-autocomplete-row", has_text="Age").first.click()
                page.wait_for_function(
                    """
                    () => document.querySelector("#glmFormulaEditor")?.env?.editor?.getValue() === "Age"
                    """,
                    timeout=10_000,
                )

                closed_editor_top = page.evaluate(
                    """
                    () => document.querySelector("#glmFormulaEditor")?.getBoundingClientRect().top || 0
                    """
                )
                page.evaluate(
                    """
                    () => {
                      const drawer = document.querySelector("#glmFormulaAssistDrawer");
                      if (drawer?.classList.contains("hidden")) document.querySelector("#glmFormulaAssistBtn")?.click();
                    }
                    """
                )
                page.locator("#glmFormulaAssistDrawer:not(.hidden)").wait_for(timeout=10_000)
                page.locator('[data-glm-assist-tab="snippets"]').click()
                self.assertEqual(page.locator("#glmFormulaAssistAppendPlus").count(), 0)
                self.assertEqual(page.locator("#glmFormulaAssistSecondaryFeature").count(), 0)
                page.wait_for_function(
                    """
                    () => {
                      const drawer = document.querySelector("#glmFormulaAssistDrawer");
                      const header = document.querySelector(".glm-formula-panel .glm-panel-header");
                      const actions = document.querySelector(".glm-formula-panel .glm-builder-actions");
                      const controls = document.querySelector(".glm-formula-panel .glm-builder-control-row");
                      const family = document.querySelector("#glmFamilySelect");
                      const editor = document.querySelector("#glmFormulaEditor");
                      const preview = document.querySelector("#glmFormulaAssistPreview");
                      if (!drawer || !header || !actions || !controls || !family || !editor || !preview || drawer.classList.contains("hidden")) return false;
                      const drawerRect = drawer.getBoundingClientRect();
                      const headerRect = header.getBoundingClientRect();
                      const actionsRect = actions.getBoundingClientRect();
                      const editorRect = editor.getBoundingClientRect();
                      const previewRect = preview.getBoundingClientRect();
                      const style = getComputedStyle(drawer);
                      return Math.abs(drawerRect.top - headerRect.bottom) <= 2
                        && drawerRect.height >= 428
                        && previewRect.height >= 78
                        && drawerRect.top >= actionsRect.bottom - 2
                        && controls.getClientRects().length === 0
                        && family.getClientRects().length === 0
                        && editorRect.top >= drawerRect.bottom - 2
                        && style.borderLeftWidth === "0px"
                        && style.borderRightWidth === "0px"
                        && style.borderTopWidth === "1px"
                        && style.borderBottomWidth === "1px";
                    }
                    """,
                    timeout=10_000,
                )
                self.assertGreater(
                    page.evaluate("() => document.querySelector('#glmFormulaEditor')?.getBoundingClientRect().top || 0"),
                    closed_editor_top + 100,
                )
                page.locator("#glmFormulaAssistBtn").click()
                page.wait_for_function(
                    """
                    () => document.querySelector("#glmFormulaAssistDrawer")?.classList.contains("hidden")
                    """,
                    timeout=10_000,
                )
                page.wait_for_function(
                    """
                    (closedTop) => {
                      const controls = document.querySelector(".glm-formula-panel .glm-builder-control-row");
                      const family = document.querySelector("#glmFamilySelect");
                      const editor = document.querySelector("#glmFormulaEditor");
                      if (!controls || !family || !editor) return false;
                      return controls.getClientRects().length > 0
                        && family.getClientRects().length > 0
                        && Math.abs(editor.getBoundingClientRect().top - closedTop) <= 2;
                    }
                    """,
                    arg=closed_editor_top,
                    timeout=10_000,
                )
                page.locator("#glmFormulaAssistBtn").click()
                page.locator("#glmFormulaAssistDrawer:not(.hidden)").wait_for(timeout=10_000)
                page.locator('[data-glm-assist-tab="snippets"]').click()
                page.locator("#glmFormulaAssistFeatureSearch").click()
                page.keyboard.type("Ag")
                page.wait_for_function(
                    """
                    () => {
                      const search = document.querySelector("#glmFormulaAssistFeatureSearch");
                      const options = [...document.querySelectorAll("#glmFormulaAssistFeatureSelect option")].map((option) => option.textContent.trim());
                      return search?.value === "Ag" && options.includes("Age") && options.every((text) => !text.includes("·"));
                    }
                    """,
                    timeout=10_000,
                )
                page.locator("#glmFormulaAssistFeatureSelect").select_option("Age")
                page.wait_for_function(
                    """
                    () => document.querySelector("#glmFormulaAssistInsertBtn")?.textContent.trim() === "Insert at cursor"
                    """,
                    timeout=10_000,
                )
                snippet_order = page.locator("#glmFormulaAssistSnippetList [data-glm-snippet-id]").evaluate_all(
                    "(items) => items.map((item) => item.textContent.trim())"
                )
                page.locator('[data-glm-snippet-id="sqrt"]').click()
                page.wait_for_function(
                    """
                    () => document.querySelector("#glmFormulaAssistPreview")?.textContent.trim() === "+ sqrt(Age)"
                      && document.querySelector('[data-glm-snippet-id="sqrt"]')?.getAttribute("aria-selected") === "true"
                    """,
                    timeout=10_000,
                )
                self.assertEqual(
                    page.locator("#glmFormulaAssistSnippetList [data-glm-snippet-id]").evaluate_all(
                        "(items) => items.map((item) => item.textContent.trim())"
                    ),
                    snippet_order,
                )
                self.assertEqual(
                    page.locator('[data-glm-snippet-id="sqrt"]').evaluate(
                        """
                        (node) => {
                          const style = getComputedStyle(node);
                          return `${style.outlineStyle}|${style.outlineWidth}|${style.boxShadow}`;
                        }
                        """
                    ),
                    "none|0px|none",
                )
                page.locator('[data-glm-snippet-id="identity"]').click()
                page.locator("#glmFormulaAssistIncludeHeader").check()
                page.wait_for_function(
                    """
                    () => document.querySelector("#glmFormulaAssistPreview")?.textContent.trim() === "# Age\\n+ Age"
                    """,
                    timeout=10_000,
                )
                page.evaluate(
                    """
                    () => {
                      const editorNode = document.querySelector("#glmFormulaEditor");
                      const aceEditor = editorNode?.env?.editor || null;
                      if (aceEditor) {
                        aceEditor.setValue("replace me", -1);
                        aceEditor.selectAll();
                        aceEditor.focus();
                      }
                    }
                    """
                )
                page.locator("#glmFormulaAssistInsertBtn").click()
                page.wait_for_function(
                    """
                    () => document.querySelector("#glmFormulaEditor")?.env?.editor?.getValue() === "# Age\\nAge"
                    """,
                    timeout=10_000,
                )
                page.locator("#glmFormulaAssistIncludeHeader").uncheck()
                page.wait_for_function(
                    """
                    () => document.querySelector("#glmFormulaAssistPreview")?.textContent.trim() === "+ Age"
                    """,
                    timeout=10_000,
                )
                page.evaluate(
                    """
                    () => {
                      const editorNode = document.querySelector("#glmFormulaEditor");
                      const aceEditor = editorNode?.env?.editor || null;
                      if (aceEditor) {
                        aceEditor.setValue("actualNumerator ~ 1 ", 1);
                        aceEditor.focus();
                      }
                    }
                    """
                )
                page.locator("#glmFormulaAssistInsertBtn").click()
                page.wait_for_function(
                    """
                    () => document.querySelector("#glmFormulaEditor")?.env?.editor?.getValue()
                      === "actualNumerator ~ 1 + Age\\n"
                    """,
                    timeout=10_000,
                )

                page.locator('[data-glm-assist-tab="piecewise"]').click()
                page.wait_for_function(
                    """
                    () => {
                      const options = [...document.querySelectorAll("#glmFormulaAssistFeatureSelect option")].map((option) => option.textContent.trim());
                      return options.includes("Age") && options.every((text) => !text.includes("·"));
                    }
                    """,
                    timeout=10_000,
                )
                page.locator("#glmFormulaAssistFeatureSelect").select_option("Age")
                page.locator("#glmFormulaAssistBreakpoints").fill("10,20,30")
                page.evaluate(
                    """
                    () => {
                      const editorNode = document.querySelector("#glmFormulaEditor");
                      const aceEditor = editorNode?.env?.editor || null;
                      if (aceEditor) {
                        aceEditor.setValue("", -1);
                        aceEditor.focus();
                      }
                    }
                    """
                )
                page.locator("#glmFormulaAssistInsertPiecewiseBtn").click()
                page.wait_for_function(
                    """
                    () => {
                      const formula = document.querySelector("#glmFormulaEditor")?.env?.editor?.getValue() || "";
                      return formula.startsWith("+ pmin(10, Age)")
                        && formula.includes("+ pmax(10, pmin(20, Age))")
                        && formula.includes("pmax(20, pmin(30, Age))")
                        && formula.endsWith("\\n");
                    }
                    """,
                    timeout=10_000,
                )

                page.locator('[data-glm-assist-tab="levels"]').click()
                page.locator("#glmFormulaAssistFeatureSearch").fill("")
                page.wait_for_function(
                    """
                    () => {
                      const options = [...document.querySelectorAll("#glmFormulaAssistFeatureSelect option")].map((option) => option.textContent.trim());
                      return options.includes("Segment") && options.every((text) => !text.includes("·"));
                    }
                    """,
                    timeout=10_000,
                )
                page.locator("#glmFormulaAssistFeatureSelect").select_option("Segment")
                page.wait_for_function(
                    """
                    () => {
                      const search = document.querySelector("#glmFormulaAssistLevelSearch")?.getBoundingClientRect();
                      const mode = document.querySelector(".glm-formula-assist-level-mode")?.getBoundingClientRect();
                      return search && mode && Math.abs(search.top - mode.top) <= 2 && search.right <= mode.left;
                    }
                    """,
                    timeout=10_000,
                )
                page.locator('#glmFormulaAssistLevelList [data-glm-level-value="A"]').wait_for(timeout=10_000)
                page.locator('#glmFormulaAssistLevelList [data-glm-level-value="A"]').check()
                page.locator('#glmFormulaAssistLevelList [data-glm-level-value="B"]').check()
                page.locator('[data-glm-level-mode="group"]').click()
                page.wait_for_function(
                    """
                    () => document.querySelector("#glmFormulaAssistPreview")?.textContent.trim()
                      === '+ ifelse(np.isin(Segment, ["A", "B"]), 1, 0)'
                    """,
                    timeout=10_000,
                )
                page.evaluate(
                    """
                    () => {
                      const editorNode = document.querySelector("#glmFormulaEditor");
                      const aceEditor = editorNode?.env?.editor || null;
                      if (aceEditor) {
                        aceEditor.setValue("actualNumerator ~ 1 ", 1);
                        aceEditor.focus();
                      }
                    }
                    """
                )
                page.locator("#glmFormulaAssistInsertLevelsBtn").click()
                page.wait_for_function(
                    """
                    () => document.querySelector("#glmFormulaEditor")?.env?.editor?.getValue()
                      === 'actualNumerator ~ 1 + ifelse(np.isin(Segment, ["A", "B"]), 1, 0)\\n'
                    """,
                    timeout=10_000,
                )
                page.locator('[data-glm-level-mode="ind"]').click()
                page.wait_for_function(
                    """
                    () => document.querySelector("#glmFormulaAssistPreview")?.textContent.trim()
                      === '+ ifelse(Segment == "A", 1, 0)\\n+ ifelse(Segment == "B", 1, 0)'
                    """,
                    timeout=10_000,
                )
                page.evaluate(
                    """
                    () => {
                      const editorNode = document.querySelector("#glmFormulaEditor");
                      const aceEditor = editorNode?.env?.editor || null;
                      if (aceEditor) {
                        aceEditor.setValue("actualNumerator ~ 1 ", 1);
                        aceEditor.focus();
                      }
                    }
                    """
                )
                page.locator("#glmFormulaAssistInsertLevelsBtn").click()
                page.wait_for_function(
                    """
                    () => document.querySelector("#glmFormulaEditor")?.env?.editor?.getValue()
                      === 'actualNumerator ~ 1 + ifelse(Segment == "A", 1, 0)\\n+ ifelse(Segment == "B", 1, 0)\\n'
                    """,
                    timeout=10_000,
                )
                set_glm_builder_draft({
                    "formula": "actualNumerator ~ 1 + Age + Segment",
                    "family": "tweedie",
                    "familyParameter": "1.5",
                    "scope": "all",
                    "regularizationMode": "none",
                    "regularizationMix": "0.5",
                    "regularizationAlpha": "0.01",
                })
                page.evaluate(
                    """
                    () => {
                      const drawer = document.querySelector("#glmFormulaAssistDrawer");
                      if (drawer && !drawer.classList.contains("hidden")) document.querySelector("#glmFormulaAssistBtn")?.click();
                    }
                    """
                )
                page.locator("#glmCoefficientTable tbody tr", has_text="(Intercept)").click(button="right")
                page.wait_for_timeout(150)
                self.assertEqual(page.locator("#glmCoefficientContextMenu:not([hidden])").count(), 0)
                page.locator("#glmCoefficientTable tbody tr", has_text="Age").first.click(button="right")
                page.locator("#glmCoefficientContextMenu:not([hidden])").wait_for(timeout=10_000)
                glm_single_coefficient_context_labels = page.locator("#glmCoefficientContextMenu [role='menuitem']").evaluate_all(
                    "(items) => items.map((item) => item.textContent.trim())"
                )
                self.assertEqual(glm_single_coefficient_context_labels, ["Go to Line and Bar (Age)"])
                page.keyboard.press("Escape")
                page.wait_for_function('() => document.querySelector("#glmCoefficientContextMenu")?.hidden === true')
                page.locator("#glmCoefficientTable tbody tr", has_text="Age:Segment[A]").click(button="right")
                page.locator("#glmCoefficientContextMenu:not([hidden])").wait_for(timeout=10_000)
                glm_interaction_coefficient_context_labels = page.locator("#glmCoefficientContextMenu [role='menuitem']").evaluate_all(
                    "(items) => items.map((item) => item.textContent.trim())"
                )
                self.assertEqual(
                    glm_interaction_coefficient_context_labels,
                    ["Go to Line and Bar (Age)", "Go to Line and Bar (Segment)"],
                )
                with page.expect_request(lambda request: request.url.endswith("/api/chart"), timeout=10_000) as glm_coefficient_chart_info:
                    page.locator("#glmCoefficientContextMenu [role='menuitem']", has_text="Go to Line and Bar (Segment)").click()
                glm_coefficient_chart_body = json.loads(glm_coefficient_chart_info.value.post_data or "{}")
                page.locator("#lineBarTool.active").wait_for(timeout=10_000)
                if page.locator("#lineBarSideControlsToggleBtn").get_attribute("aria-expanded") == "false":
                    page.locator("#lineBarSideControlsToggleBtn").click()
                    page.wait_for_function(
                        """
                        () => document.querySelector("#lineBarSideControlsToggleBtn")?.getAttribute("aria-expanded") === "true"
                          && getComputedStyle(document.querySelector("#chartSideControls")).display !== "none"
                        """,
                        timeout=10_000,
                    )
                page.locator("#featureList .feature.active", has_text="Segment").wait_for(timeout=10_000)
                self.assertEqual(glm_coefficient_chart_body["x"], "Segment")
                self.assertEqual(glm_coefficient_chart_body["responses"][1]["numerator"], "glm_prediction")
                self.assertEqual(glm_coefficient_chart_body["responses"][1]["source"], "glm:browser-smoke-glm:predictions")
                page.locator("#glmTool").click()
                page.locator("#modelToolWrap:not(.hidden) .glm-tool").wait_for(timeout=10_000)
                page.get_by_role("button", name="Formula builder").click()
                wait_for_glm_builder_state(
                    {
                        "formula": "actualNumerator ~ 1 + Age + Segment",
                        "family": "tweedie",
                        "familyParameter": "1.5",
                        "scope": "all",
                        "regularizationMode": "none",
                    }
                )
                edited_glm_draft = {
                    "formula": "actualNumerator ~ 1 + Age + C(Segment)",
                    "family": "tweedie",
                    "familyParameter": "1.7",
                    "familyParameterDisabled": False,
                    "scope": "training",
                    "regularizationMode": "manual",
                    "regularizationMix": "1",
                    "regularizationAlpha": "0.09",
                    "regularizationMixDisabled": False,
                    "regularizationAlphaDisabled": False,
                }
                set_glm_builder_draft(edited_glm_draft)
                self.assertEqual(wait_for_glm_builder_state(edited_glm_draft), edited_glm_draft)
                page.locator("#lineBarTool").click()
                page.locator("#chart:not(.hidden)").wait_for(timeout=10_000)
                page.locator("#glmTool").click()
                page.locator("#modelToolWrap:not(.hidden) .glm-tool").wait_for(timeout=10_000)
                page.get_by_role("button", name="Formula builder").click()
                self.assertEqual(wait_for_glm_builder_state(edited_glm_draft), edited_glm_draft)

                open_sidebar_section("#glmModelCollapseBtn")
                page.locator('#glmModelSelect [data-glm-model-id="browser-smoke-glm-2"]').click()
                page.locator("#glmModelSelectedMeta", has_text="Second smoke GLM").wait_for(timeout=10_000)
                self.assertEqual(
                    wait_for_glm_builder_state(
                        {
                            "formula": "actualNumerator ~ 1 + Age",
                            "family": "normal",
                            "familyParameter": "",
                            "familyParameterDisabled": True,
                            "scope": "training",
                            "regularizationMode": "manual",
                            "regularizationMix": "1",
                            "regularizationAlpha": "0.07",
                            "regularizationMixDisabled": False,
                            "regularizationAlphaDisabled": False,
                        }
                    ),
                    {
                        "formula": "actualNumerator ~ 1 + Age",
                        "family": "normal",
                        "familyParameter": "",
                        "familyParameterDisabled": True,
                        "scope": "training",
                        "regularizationMode": "manual",
                        "regularizationMix": "1",
                        "regularizationAlpha": "0.07",
                        "regularizationMixDisabled": False,
                        "regularizationAlphaDisabled": False,
                    },
                )
                page.locator('#glmModelSelect [data-glm-model-id="browser-smoke-glm"]').click()
                page.locator("#glmModelSelectedMeta", has_text="Browser smoke GLM").wait_for(timeout=10_000)
                wait_for_glm_builder_state(
                    {
                        "formula": "actualNumerator ~ 1 + Age + Segment",
                        "family": "tweedie",
                        "familyParameter": "1.5",
                        "scope": "all",
                        "regularizationMode": "none",
                    }
                )

                chart_url = (
                    f"{base_url}/?tool=line_bar&source=gbm%3Abrowser-smoke-model%3Apredictions"
                    "&x=Segment&actual=gbm_prediction&denominator=denominator"
                )
                page.goto(chart_url, wait_until="domcontentloaded")
                page.locator("#chart:not(.hidden)").wait_for(timeout=10_000)
                page.wait_for_function(
                    '() => document.querySelector("#lineBarGroupMeta")?.textContent.includes("groups")',
                    timeout=10_000,
                )
                page.locator("#lineBarSideControlsToggleBtn").click()
                page.wait_for_function(
                    """
                    () => document.querySelector("#lineBarSideControlsToggleBtn")?.getAttribute("aria-expanded") === "true"
                      && getComputedStyle(document.querySelector("#chartSideControls")).display !== "none"
                    """,
                    timeout=10_000,
                )
                open_sidebar_section("#gbmModelCollapseBtn")
                chart_requests_before = chart_requests
                with page.expect_request(lambda request: request.url.endswith("/api/chart"), timeout=10_000) as chart_request_info:
                    page.locator('#gbmModelSelect [data-gbm-model-id="browser-smoke-model-2"]').click()
                request_body = json.loads(chart_request_info.value.post_data or "{}")
                page.locator("#gbmModelSelectedMeta", has_text="Second smoke model").wait_for(timeout=10_000)
                self.assertGreater(chart_requests, chart_requests_before)
                self.assertEqual(request_body["source"], "gbm:browser-smoke-model-2:predictions")
                self.assertEqual(request_body["x"], "Segment")
                self.assertEqual(request_body["responses"][0]["numerator"], "gbm_prediction")
                self.assertEqual(request_body["denominator"], "denominator")
                page.locator("#featureList .feature.active", has_text="Segment").wait_for(timeout=10_000)

                mixed_expected_url = (
                    f"{base_url}/?tool=line_bar&source=glm%3Abrowser-smoke-glm%3Apredictions"
                    "&x=Segment&actual=actualNumerator&expected=glm_prediction&denominator=denominator"
                )
                page.goto(mixed_expected_url, wait_until="domcontentloaded")
                page.locator("#chart:not(.hidden)").wait_for(timeout=10_000)
                page.wait_for_function(
                    '() => document.querySelector("#lineBarGroupMeta")?.textContent.includes("groups")',
                    timeout=10_000,
                )
                page.locator("#lineBarSideControlsToggleBtn").click()
                page.wait_for_function(
                    """
                    () => document.querySelector("#lineBarSideControlsToggleBtn")?.getAttribute("aria-expanded") === "true"
                      && getComputedStyle(document.querySelector("#chartSideControls")).display !== "none"
                    """,
                    timeout=10_000,
                )
                expected_state = page.evaluate(
                    """
                    () => [...document.querySelectorAll("#expectedList .feature")]
                      .map((button) => ({
                        text: button.textContent || "",
                        source: button.dataset.sourceId || "",
                        active: button.classList.contains("active"),
                      }))
                    """
                )
                self.assertIn(
                    {"text": "glm_predictionnumeric", "source": "glm:browser-smoke-glm:predictions", "active": True},
                    expected_state,
                )
                self.assertIn(
                    {"text": "gbm_predictionnumeric", "source": "gbm:browser-smoke-model-2:predictions", "active": False},
                    expected_state,
                )
                expected_pinned_state = page.evaluate(
                    """
                    () => {
                      const pinned = [...document.querySelectorAll("#expectedList > .line-bar-pinned-region > .feature")]
                        .map((button) => ({
                          text: button.textContent || "",
                          value: button.dataset.value || "",
                          source: button.dataset.sourceId || "",
                        }));
                      const scrollValues = [...document.querySelectorAll("#expectedList > .line-bar-scroll-region > .feature")]
                        .map((button) => button.dataset.value || "");
                      const none = document.querySelector("#expectedList > .line-bar-pinned-region > .feature.expected-none-option");
                      const noneKind = none?.querySelector(".kind");
                      const special = document.querySelector('#expectedList > .line-bar-pinned-region > .feature[data-value="glm_prediction"]');
                      return {
                        pinned,
                        scrollValues,
                        noneFontWeight: none ? getComputedStyle(none).fontWeight : "",
                        noneKindFontWeight: noneKind ? getComputedStyle(noneKind).fontWeight : "",
                        noneTextTransform: noneKind ? getComputedStyle(noneKind).textTransform : "",
                        specialBackground: special ? getComputedStyle(special).backgroundColor : "",
                      };
                    }
                    """
                )
                self.assertEqual(
                    [row["value"] for row in expected_pinned_state["pinned"][:7]],
                    ["", "glm_prediction", "gbm_prediction", "glm_prediction_rate", "gbm_prediction_rate", "glm_tabulated_prediction", "gbm_tabulated_prediction"],
                )
                self.assertEqual(expected_pinned_state["pinned"][0]["text"], "No expected lineoff")
                self.assertEqual(expected_pinned_state["noneFontWeight"], "400")
                self.assertEqual(expected_pinned_state["noneKindFontWeight"], "400")
                self.assertEqual(expected_pinned_state["noneTextTransform"], "none")
                self.assertIn("glm_tabulated_prediction", [row["value"] for row in expected_pinned_state["pinned"]])
                self.assertIn("gbm_tabulated_prediction", [row["value"] for row in expected_pinned_state["pinned"]])
                self.assertTrue(expected_pinned_state["specialBackground"])
                page.locator("#chartExpectedToggle").click()
                page.wait_for_function(
                    """
                    () => {
                      const controls = document.querySelector("#chartSideControls");
                      const expected = document.querySelector("#expectedSideSection");
                      return controls
                        && expected
                        && !controls.classList.contains("chart-expected-collapsed")
                        && !expected.hidden
                        && !expected.hasAttribute("inert");
                    }
                    """,
                    timeout=10_000,
                )
                page.locator('.segmented[data-control="expectedSort"] button[data-value="alpha"]').click()
                expected_alpha_state = page.evaluate(
                    """
                    () => ({
                      pinned: [...document.querySelectorAll("#expectedList > .line-bar-pinned-region > .feature")]
                        .map((button) => button.dataset.value || ""),
                      scroll: [...document.querySelectorAll("#expectedList > .line-bar-scroll-region > .feature")]
                        .map((button) => button.dataset.value || ""),
                    })
                    """
                )
                self.assertEqual(
                    expected_alpha_state["pinned"][:7],
                    ["", "glm_prediction", "gbm_prediction", "glm_prediction_rate", "gbm_prediction_rate", "glm_tabulated_prediction", "gbm_tabulated_prediction"],
                )
                self.assertEqual(expected_alpha_state["scroll"], sorted(expected_alpha_state["scroll"], key=str.casefold))
                page.locator('.segmented[data-control="expectedSort"] button[data-value="original"]').click()

                feature_state = page.evaluate(
                    """
                    () => [...document.querySelectorAll("#featureList .feature")]
                      .map((button) => ({
                        text: button.textContent || "",
                        source: button.dataset.sourceId || "",
                        active: button.classList.contains("active"),
                      }))
                    """
                )
                self.assertIn(
                    {"text": "glm_predictionnumeric", "source": "glm:browser-smoke-glm:predictions", "active": False},
                    feature_state,
                )
                self.assertIn(
                    {"text": "gbm_predictionnumeric", "source": "gbm:browser-smoke-model-2:predictions", "active": False},
                    feature_state,
                )
                ratio_source_id = "model_ratio:gbm_to_glm_ratio:browser-smoke-model-2:browser-smoke-glm"
                self.assertIn(
                    {"text": "gbm_to_glm_rationumeric", "source": ratio_source_id, "active": False},
                    feature_state,
                )
                feature_pinned_state = page.evaluate(
                    """
                    () => {
                      const state = () => ({
                        pinned: [...document.querySelectorAll("#featureList > .line-bar-pinned-region > .feature")]
                          .map((button) => button.dataset.value || ""),
                        scroll: [...document.querySelectorAll("#featureList > .line-bar-scroll-region > .feature")]
                          .map((button) => button.dataset.value || ""),
                      });
                      const before = state();
                      const pinnedRegion = document.querySelector("#featureList > .line-bar-pinned-region");
                      const scrollRegion = document.querySelector("#featureList > .line-bar-scroll-region");
                      const previousStyle = scrollRegion?.getAttribute("style");
                      const pinnedTopBefore = Math.round(pinnedRegion?.getBoundingClientRect().top || 0);
                      if (scrollRegion) {
                        scrollRegion.style.flex = "0 0 42px";
                        scrollRegion.style.height = "42px";
                        scrollRegion.scrollTop = scrollRegion.scrollHeight;
                      }
                      const scrollTop = scrollRegion?.scrollTop || 0;
                      const pinnedTopAfter = Math.round(pinnedRegion?.getBoundingClientRect().top || 0);
                      if (scrollRegion) {
                        if (previousStyle === null) scrollRegion.removeAttribute("style");
                        else scrollRegion.setAttribute("style", previousStyle);
                      }
                      return {
                        ...before,
                        scrollTop,
                        pinnedTopBefore,
                        pinnedTopAfter,
                        rootScrollTop: document.querySelector("#featureList")?.scrollTop || 0,
                      };
                    }
                    """
                )
                self.assertEqual(
                    feature_pinned_state["pinned"],
                    ["gbm_to_glm_ratio", "glm_prediction", "gbm_prediction", "glm_prediction_rate", "gbm_prediction_rate", "glm_tabulated_prediction", "gbm_tabulated_prediction"],
                )
                self.assertIn("glm_tabulated_prediction", feature_pinned_state["pinned"])
                self.assertIn("gbm_tabulated_prediction", feature_pinned_state["pinned"])
                self.assertGreater(feature_pinned_state["scrollTop"], 0)
                self.assertEqual(feature_pinned_state["pinnedTopBefore"], feature_pinned_state["pinnedTopAfter"])
                self.assertEqual(feature_pinned_state["rootScrollTop"], 0)

                page.locator('.segmented[data-control="featureSort"] button[data-value="alpha"]').click()
                alpha_feature_state = page.evaluate(
                    """
                    () => ({
                      pinned: [...document.querySelectorAll("#featureList > .line-bar-pinned-region > .feature")]
                        .map((button) => button.dataset.value || ""),
                      scroll: [...document.querySelectorAll("#featureList > .line-bar-scroll-region > .feature")]
                        .map((button) => button.dataset.value || ""),
                    })
                    """
                )
                self.assertEqual(alpha_feature_state["pinned"], feature_pinned_state["pinned"])
                self.assertEqual(alpha_feature_state["scroll"], sorted(alpha_feature_state["scroll"], key=str.casefold))
                self.assertFalse(set(alpha_feature_state["pinned"]) & set(alpha_feature_state["scroll"]))

                page.wait_for_function(
                    '() => !document.querySelector(\'.segmented[data-control="featureSort"] button[data-value="importance"]\')?.classList.contains("hidden")',
                    timeout=10_000,
                )
                page.locator('.segmented[data-control="featureSort"] button[data-value="importance"]').click()
                importance_feature_state = page.evaluate(
                    """
                    () => ({
                      split: document.querySelector("#featureList")?.classList.contains("line-bar-split-list"),
                      specialValues: [...document.querySelectorAll("#featureList .feature")]
                        .map((button) => button.dataset.value || "")
                        .filter((value) => ["gbm_to_glm_ratio", "glm_prediction", "gbm_prediction", "glm_prediction_rate", "gbm_prediction_rate", "glm_tabulated_prediction", "gbm_tabulated_prediction"].includes(value)),
                    })
                    """
                )
                self.assertTrue(importance_feature_state["split"])
                self.assertEqual(importance_feature_state["specialValues"], ["gbm_to_glm_ratio"])
                page.locator('.segmented[data-control="featureSort"] button[data-value="original"]').click()
                page.locator(
                    f'#featureList .feature[data-source-id="{ratio_source_id}"][data-value="gbm_to_glm_ratio"]',
                ).wait_for(timeout=10_000)

                with page.expect_response(
                    lambda response: (
                        response.url.endswith("/api/banding/suggestion")
                        and response.status == 200
                        and ratio_source_id in (response.request.post_data or "")
                    ),
                    timeout=10_000,
                ) as ratio_banding_info:
                    with page.expect_request(
                        lambda request: request.url.endswith("/api/chart") and "gbm_to_glm_ratio" in (request.post_data or ""),
                        timeout=10_000,
                    ) as ratio_chart_info:
                        page.locator(
                            f'#featureList .feature[data-source-id="{ratio_source_id}"][data-value="gbm_to_glm_ratio"]',
                        ).click()
                ratio_banding_body = json.loads(ratio_banding_info.value.request.post_data or "{}")
                ratio_chart_body = json.loads(ratio_chart_info.value.post_data or "{}")
                self.assertEqual(ratio_banding_body["source"], "glm:browser-smoke-glm:predictions")
                self.assertEqual(ratio_banding_body["xSource"], ratio_source_id)
                self.assertEqual(ratio_banding_body["feature"], "gbm_to_glm_ratio")
                self.assertEqual(ratio_banding_body["responses"][1]["source"], "glm:browser-smoke-glm:predictions")
                self.assertEqual(ratio_chart_body["x"], "gbm_to_glm_ratio")
                self.assertEqual(ratio_chart_body["xSource"], ratio_source_id)

                page.locator(
                    '#featureList .feature[data-source-id="glm:browser-smoke-glm:predictions"][data-value="glm_prediction"]',
                ).wait_for(timeout=10_000)

                with page.expect_response(
                    lambda response: response.url.endswith("/api/banding/suggestion") and response.status == 200,
                    timeout=10_000,
                ) as glm_banding_info:
                    page.locator(
                        '#featureList .feature[data-source-id="glm:browser-smoke-glm:predictions"][data-value="glm_prediction"]',
                    ).click()
                glm_banding_body = json.loads(glm_banding_info.value.request.post_data or "{}")
                self.assertEqual(glm_banding_body["source"], "glm:browser-smoke-glm:predictions")
                self.assertEqual(glm_banding_body["xSource"], "glm:browser-smoke-glm:predictions")
                self.assertEqual(glm_banding_body["feature"], "glm_prediction")
                page.wait_for_function(
                    '() => document.querySelector("#lineBarGroupMeta")?.textContent.includes("groups")',
                    timeout=10_000,
                )
                self.assertNotIn("Banding estimate failed", page.locator("#status").text_content(timeout=10_000))

                with page.expect_request(lambda request: request.url.endswith("/api/chart"), timeout=10_000) as gbm_expected_info:
                    page.locator(
                        '#expectedList .feature[data-source-id="gbm:browser-smoke-model-2:predictions"][data-value="gbm_prediction"]',
                    ).click()
                gbm_expected_body = json.loads(gbm_expected_info.value.post_data or "{}")
                self.assertEqual(gbm_expected_body["source"], "glm:browser-smoke-glm:predictions")
                self.assertEqual(gbm_expected_body["responses"][0]["numerator"], "actualNumerator")
                self.assertEqual(gbm_expected_body["responses"][1]["numerator"], "glm_prediction")
                self.assertEqual(gbm_expected_body["responses"][1]["source"], "glm:browser-smoke-glm:predictions")
                self.assertEqual(gbm_expected_body["responses"][2]["numerator"], "gbm_prediction")
                self.assertEqual(gbm_expected_body["responses"][2]["source"], "gbm:browser-smoke-model-2:predictions")
                page.wait_for_function(
                    """
                    () => {
                      const chart = window.echarts.getInstanceByDom(document.querySelector("#chart"));
                      return chart?.getOption().series?.some((series) => series.name === "gbm_prediction");
                    }
                    """,
                    timeout=10_000,
                )
                expected_two_line_state = page.evaluate(
                    """
                    () => {
                      const buttons = [...document.querySelectorAll("#expectedList .feature")];
                      const chart = window.echarts.getInstanceByDom(document.querySelector("#chart"));
                      const series = chart.getOption().series || [];
                      const lineColor = (name) => {
                        const style = series.find((item) => item.name === name)?.lineStyle;
                        return ((Array.isArray(style) ? style[0]?.color : style?.color) || "").toLowerCase();
                      };
                      return {
                        active: buttons.filter((button) => button.classList.contains("active"))
                          .map((button) => `${button.dataset.sourceId || ""}:${button.dataset.value || ""}`),
                        disabledInactiveCount: buttons.filter((button) => button.dataset.value && !button.classList.contains("active") && button.disabled).length,
                        noneDisabled: document.querySelector("#expectedList .expected-none-option")?.disabled || false,
                        glmColor: lineColor("glm_prediction"),
                        gbmColor: lineColor("gbm_prediction"),
                        accent: getComputedStyle(document.body).getPropertyValue("--accent").trim().toLowerCase(),
                      };
                    }
                    """
                )
                self.assertEqual(
                    expected_two_line_state["active"],
                    [
                        "glm:browser-smoke-glm:predictions:glm_prediction",
                        "gbm:browser-smoke-model-2:predictions:gbm_prediction",
                    ],
                )
                self.assertGreater(expected_two_line_state["disabledInactiveCount"], 0)
                self.assertFalse(expected_two_line_state["noneDisabled"])
                self.assertEqual(expected_two_line_state["glmColor"], "#d13f3f")
                self.assertEqual(expected_two_line_state["gbmColor"], expected_two_line_state["accent"])
                feature_state = page.evaluate(
                    """
                    () => [...document.querySelectorAll("#featureList .feature")]
                      .map((button) => ({
                        text: button.textContent || "",
                        source: button.dataset.sourceId || "",
                        active: button.classList.contains("active"),
                      }))
                    """
                )
                self.assertIn(
                    {"text": "glm_predictionnumeric", "source": "glm:browser-smoke-glm:predictions", "active": True},
                    feature_state,
                )
                self.assertIn(
                    {"text": "gbm_predictionnumeric", "source": "gbm:browser-smoke-model-2:predictions", "active": False},
                    feature_state,
                )

                with page.expect_request(lambda request: request.url.endswith("/api/chart"), timeout=10_000) as segment_feature_info:
                    page.locator('#featureList .feature[data-source-id="dataset"]', has_text="Segment").click()
                segment_feature_body = json.loads(segment_feature_info.value.post_data or "{}")
                self.assertEqual(segment_feature_body["source"], "dataset")
                self.assertEqual(segment_feature_body["x"], "Segment")
                self.assertNotIn("xSource", segment_feature_body)
                self.assertEqual(segment_feature_body["responses"][1]["source"], "glm:browser-smoke-glm:predictions")
                self.assertEqual(segment_feature_body["responses"][2]["source"], "gbm:browser-smoke-model-2:predictions")
                page.locator("#featureList .feature.active", has_text="Segment").wait_for(timeout=10_000)

                open_sidebar_section("#glmModelCollapseBtn")
                with page.expect_request(lambda request: request.url.endswith("/api/chart"), timeout=10_000) as glm_to_glm_info:
                    page.locator('#glmModelSelect [data-glm-model-id="browser-smoke-glm-2"]').click()
                glm_to_glm_body = json.loads(glm_to_glm_info.value.post_data or "{}")
                page.locator("#glmModelSelectedMeta", has_text="Second smoke GLM").wait_for(timeout=10_000)
                self.assertEqual(glm_to_glm_body["source"], "dataset")
                self.assertEqual(glm_to_glm_body["x"], "Segment")
                self.assertNotIn("xSource", glm_to_glm_body)
                self.assertEqual(glm_to_glm_body["responses"][0]["numerator"], "actualNumerator")
                self.assertEqual(glm_to_glm_body["responses"][1]["numerator"], "glm_prediction")
                self.assertEqual(glm_to_glm_body["responses"][1]["source"], "glm:browser-smoke-glm-2:predictions")
                self.assertEqual(glm_to_glm_body["responses"][2]["numerator"], "gbm_prediction")
                self.assertEqual(glm_to_glm_body["responses"][2]["source"], "gbm:browser-smoke-model-2:predictions")
                page.locator("#featureList .feature.active", has_text="Segment").wait_for(timeout=10_000)

                open_sidebar_section("#gbmModelCollapseBtn")
                with page.expect_request(lambda request: request.url.endswith("/api/chart"), timeout=10_000) as glm_to_gbm_info:
                    page.locator('#gbmModelSelect [data-gbm-model-id="browser-smoke-model"]').click()
                glm_to_gbm_body = json.loads(glm_to_gbm_info.value.post_data or "{}")
                page.locator("#gbmModelSelectedMeta", has_text="Browser smoke model").wait_for(timeout=10_000)
                self.assertEqual(glm_to_gbm_body["source"], "dataset")
                self.assertEqual(glm_to_gbm_body["x"], "Segment")
                self.assertNotIn("xSource", glm_to_gbm_body)
                self.assertEqual(glm_to_gbm_body["responses"][0]["numerator"], "actualNumerator")
                self.assertEqual(glm_to_gbm_body["responses"][1]["numerator"], "glm_prediction")
                self.assertEqual(glm_to_gbm_body["responses"][1]["source"], "glm:browser-smoke-glm-2:predictions")
                self.assertEqual(glm_to_gbm_body["responses"][2]["numerator"], "gbm_prediction")
                self.assertEqual(glm_to_gbm_body["responses"][2]["source"], "gbm:browser-smoke-model:predictions")
                page.locator("#featureList .feature.active", has_text="Segment").wait_for(timeout=10_000)

                open_sidebar_section("#gbmModelCollapseBtn")
                with page.expect_request(lambda request: request.url.endswith("/api/chart"), timeout=10_000) as gbm_to_gbm_info:
                    page.locator('#gbmModelSelect [data-gbm-model-id="browser-smoke-model-2"]').click()
                gbm_to_gbm_body = json.loads(gbm_to_gbm_info.value.post_data or "{}")
                page.locator("#gbmModelSelectedMeta", has_text="Second smoke model").wait_for(timeout=10_000)
                self.assertEqual(gbm_to_gbm_body["source"], "dataset")
                self.assertEqual(gbm_to_gbm_body["x"], "Segment")
                self.assertNotIn("xSource", gbm_to_gbm_body)
                self.assertEqual(gbm_to_gbm_body["responses"][0]["numerator"], "actualNumerator")
                self.assertEqual(gbm_to_gbm_body["responses"][1]["numerator"], "glm_prediction")
                self.assertEqual(gbm_to_gbm_body["responses"][1]["source"], "glm:browser-smoke-glm-2:predictions")
                self.assertEqual(gbm_to_gbm_body["responses"][2]["numerator"], "gbm_prediction")
                self.assertEqual(gbm_to_gbm_body["responses"][2]["source"], "gbm:browser-smoke-model-2:predictions")
                page.locator("#featureList .feature.active", has_text="Segment").wait_for(timeout=10_000)

                open_sidebar_section("#glmModelCollapseBtn")
                with page.expect_request(lambda request: request.url.endswith("/api/chart"), timeout=10_000) as gbm_to_glm_info:
                    page.locator('#glmModelSelect [data-glm-model-id="browser-smoke-glm"]').click()
                gbm_to_glm_body = json.loads(gbm_to_glm_info.value.post_data or "{}")
                page.locator("#glmModelSelectedMeta", has_text="Browser smoke GLM").wait_for(timeout=10_000)
                self.assertEqual(gbm_to_glm_body["source"], "dataset")
                self.assertEqual(gbm_to_glm_body["x"], "Segment")
                self.assertNotIn("xSource", gbm_to_glm_body)
                self.assertEqual(gbm_to_glm_body["responses"][0]["numerator"], "actualNumerator")
                self.assertEqual(gbm_to_glm_body["responses"][1]["numerator"], "glm_prediction")
                self.assertEqual(gbm_to_glm_body["responses"][1]["source"], "glm:browser-smoke-glm:predictions")
                self.assertEqual(gbm_to_glm_body["responses"][2]["numerator"], "gbm_prediction")
                self.assertEqual(gbm_to_glm_body["responses"][2]["source"], "gbm:browser-smoke-model-2:predictions")
                page.locator("#featureList .feature.active", has_text="Segment").wait_for(timeout=10_000)

                page.locator('#featureList .feature[data-source-id="dataset"]', has_text="Age").click()
                page.wait_for_function(
                    '() => document.querySelector("#lineBarGroupMeta")?.textContent.includes("groups")',
                    timeout=10_000,
                )
                axis_shrink_state = page.evaluate(
                    """
                    () => {
                      const chart = window.echarts.getInstanceByDom(document.querySelector("#chart"));
                      const option = chart.getOption();
                      const candidates = (option.series || [])
                        .filter((series) => series.yAxisIndex === 0 && series.type === "line")
                        .filter((series) => !String(series.name || "").startsWith("SHAP") && series.name !== "GLM")
                        .map((series) => ({
                          name: series.name,
                          max: Math.max(...(series.data || []).map((value) => Array.isArray(value) ? value[1] : value).map(Number).filter(Number.isFinite)),
                        }))
                        .filter((series) => Number.isFinite(series.max))
                        .sort((left, right) => right.max - left.max);
                      const target = candidates[0]?.name || "";
                      const beforeMax = Number(option.yAxis?.[0]?.max);
                      if (target) chart.dispatchAction({ type: "legendUnSelect", name: target });
                      return { target, beforeMax };
                    }
                    """
                )
                self.assertTrue(axis_shrink_state["target"])
                page.wait_for_function(
                    """
                    ({ target, beforeMax }) => {
                      const chart = window.echarts.getInstanceByDom(document.querySelector("#chart"));
                      const option = chart.getOption();
                      const selected = Object.assign({}, ...chart.getOption().legend.map((legend) => legend.selected || {}));
                      return selected[target] === false && Number(option.yAxis?.[0]?.max) < beforeMax;
                    }
                    """,
                    arg=axis_shrink_state,
                    timeout=10_000,
                )
                page.evaluate(
                    """
                    () => {
                      const chart = window.echarts.getInstanceByDom(document.querySelector("#chart"));
                      ["actualNumerator", "glm_prediction", "denominator"].forEach((name) => {
                        chart.dispatchAction({ type: "legendUnSelect", name });
                      });
                    }
                    """
                )
                page.wait_for_function(
                    """
                    () => {
                      const chart = window.echarts.getInstanceByDom(document.querySelector("#chart"));
                      const selected = Object.assign({}, ...chart.getOption().legend.map((legend) => legend.selected || {}));
                      return selected.actualNumerator === false
                        && selected.glm_prediction === false
                        && selected.denominator === false;
                    }
                    """,
                    timeout=10_000,
                )
                page.locator('#featureList .feature[data-source-id="dataset"]', has_text="Segment").click()
                page.wait_for_function(
                    """
                    () => {
                      const meta = document.querySelector("#lineBarGroupMeta")?.textContent || "";
                      const chart = window.echarts.getInstanceByDom(document.querySelector("#chart"));
                      const selected = Object.assign({}, ...chart.getOption().legend.map((legend) => legend.selected || {}));
                      return meta.includes("groups")
                        && selected.actualNumerator === false
                        && selected.glm_prediction === false
                        && selected.denominator === false;
                    }
                    """,
                    timeout=10_000,
                )
                page.locator('#featureList .feature[data-source-id="dataset"]', has_text="Age").click()
                page.wait_for_function(
                    '() => document.querySelector("#lineBarGroupMeta")?.textContent.includes("groups")',
                    timeout=10_000,
                )
                if page.locator("#lineBarToolbarToggleBtn").get_attribute("aria-expanded") == "false":
                    page.locator("#lineBarToolbarToggleBtn").click()
                    page.wait_for_function(
                        """
                        () => document.querySelector("#lineBarToolbarToggleBtn")?.getAttribute("aria-expanded") === "true"
                          && getComputedStyle(document.querySelector("#lineBarToolbar")).display !== "none"
                        """,
                        timeout=10_000,
                    )
                with page.expect_request(lambda request: request.url.endswith("/api/chart"), timeout=10_000):
                    page.locator('.segmented[data-control="transform"] button[data-value="one"]').click()
                page.wait_for_function(
                    """
                    () => {
                      const chart = window.echarts.getInstanceByDom(document.querySelector("#chart"));
                      const option = chart.getOption();
                      const baseColor = getComputedStyle(document.body).getPropertyValue("--base-bar").trim();
                      const barSeries = (option.series || []).find((series) => series.type === "bar");
                      const baseline = (option.series || []).find((series) => series.name === "0% uplift baseline");
                      const axisFormatter = option.yAxis?.[0]?.axisLabel?.formatter;
                      return (barSeries?.data || []).some((item) => item?.itemStyle?.color === baseColor)
                        && baseline?.markLine?.data?.[0]?.yAxis === 1
                        && axisFormatter?.(1) === "0%";
                    }
                    """,
                    timeout=10_000,
                )
                base_emphasis_state = page.evaluate(
                    """
                    () => {
                      const chart = window.echarts.getInstanceByDom(document.querySelector("#chart"));
                      const option = chart.getOption();
                      const baseColor = getComputedStyle(document.body).getPropertyValue("--base-bar").trim();
                      const barColor = getComputedStyle(document.body).getPropertyValue("--bar").trim();
                      const barSeries = (option.series || []).find((series) => series.type === "bar");
                      const baseline = (option.series || []).find((series) => series.name === "0% uplift baseline");
                      const axisFormatter = option.yAxis?.[0]?.axisLabel?.formatter;
                      return {
                        baseColor,
                        barColor,
                        baseColoredBars: (barSeries?.data || []).filter((item) => item?.itemStyle?.color === baseColor).length,
                        baselineY: baseline?.markLine?.data?.[0]?.yAxis,
                        baselineWidth: baseline?.markLine?.lineStyle?.width,
                        baselineSilent: baseline?.markLine?.silent,
                        axisBaseLabel: axisFormatter?.(1) || "",
                        axisMin: Number(option.yAxis?.[0]?.min),
                        axisMax: Number(option.yAxis?.[0]?.max),
                      };
                    }
                    """
                )
                self.assertEqual(base_emphasis_state["baseColoredBars"], 1)
                self.assertNotEqual(base_emphasis_state["baseColor"], base_emphasis_state["barColor"])
                self.assertEqual(base_emphasis_state["baselineY"], 1)
                self.assertGreater(base_emphasis_state["baselineWidth"], 1)
                self.assertTrue(base_emphasis_state["baselineSilent"])
                self.assertEqual(base_emphasis_state["axisBaseLabel"], "0%")
                self.assertLessEqual(base_emphasis_state["axisMin"], 1)
                self.assertGreaterEqual(base_emphasis_state["axisMax"], 1)
                page.locator('.segmented[data-control="partialDependence"] button[data-value="both"]').click()
                page.wait_for_function(
                    """
                    () => {
                      const chart = window.echarts.getInstanceByDom(document.querySelector("#chart"));
                      const option = chart.getOption();
                      return option.series.some((series) => series.name === "SHAP median")
                        && option.legend[1]?.textStyle?.fontWeight === 400
                        && option.legend[1]?.textStyle?.fontSize === 11;
                    }
                    """,
                    timeout=10_000,
                )
                page.evaluate(
                    """
                    () => {
                      const chart = window.echarts.getInstanceByDom(document.querySelector("#chart"));
                      chart.dispatchAction({ type: "legendUnSelect", name: "SHAP median" });
                    }
                    """
                )
                open_sidebar_section("#gbmModelCollapseBtn")
                with page.expect_request(lambda request: request.url.endswith("/api/chart"), timeout=10_000):
                    page.locator('#gbmModelSelect [data-gbm-model-id="browser-smoke-model"]').click()
                page.locator("#gbmModelSelectedMeta", has_text="Browser smoke model").wait_for(timeout=10_000)
                page.wait_for_function(
                    """
                    () => {
                      const chart = window.echarts.getInstanceByDom(document.querySelector("#chart"));
                      const selected = Object.assign({}, ...chart.getOption().legend.map((legend) => legend.selected || {}));
                      return selected["SHAP median"] === false;
                    }
                    """,
                    timeout=10_000,
                )
                with page.expect_request(lambda request: request.url.endswith("/api/chart"), timeout=10_000):
                    page.locator('#gbmModelSelect [data-gbm-model-id="browser-smoke-model-2"]').click()
                page.locator("#gbmModelSelectedMeta", has_text="Second smoke model").wait_for(timeout=10_000)
                page.wait_for_function(
                    """
                    () => {
                      const chart = window.echarts.getInstanceByDom(document.querySelector("#chart"));
                      const selected = Object.assign({}, ...chart.getOption().legend.map((legend) => legend.selected || {}));
                      return selected["SHAP median"] === false;
                    }
                    """,
                    timeout=10_000,
                )

                navigator_line_bar_url = (
                    f"{base_url}/?tool=line_bar&source=glm%3Abrowser-smoke-glm%3Apredictions"
                    "&x=Segment&actual=actualNumerator&expected=glm_prediction&denominator=denominator"
                )
                page.goto(navigator_line_bar_url, wait_until="domcontentloaded")
                page.locator("#chart:not(.hidden)").wait_for(timeout=10_000)
                page.wait_for_function(
                    '() => document.querySelector("#lineBarGroupMeta")?.textContent.includes("groups")',
                    timeout=10_000,
                )
                if page.locator("#lineBarSideControlsToggleBtn").get_attribute("aria-expanded") == "false":
                    page.locator("#lineBarSideControlsToggleBtn").click()
                    page.wait_for_function(
                        """
                        () => document.querySelector("#lineBarSideControlsToggleBtn")?.getAttribute("aria-expanded") === "true"
                          && getComputedStyle(document.querySelector("#chartSideControls")).display !== "none"
                        """,
                        timeout=10_000,
                    )
                page.locator("#featureList .feature.active", has_text="Segment").wait_for(timeout=10_000)
                page.locator("#glmTool").click()
                page.locator(".glm-tool").wait_for(timeout=10_000)
                page.get_by_role("button", name="Model navigator").click()
                page.locator("#glmModelGrid .tabulator-row").first.wait_for(timeout=10_000)
                glm_navigator_state = page.evaluate(
                    """
                    () => {
                      const dot = document.querySelector("#glmModelGrid .glm-model-active-dot");
                      const cell = dot?.closest(".tabulator-cell");
                      let activeDotCenterDelta = null;
                      if (dot && cell) {
                        const dotRect = dot.getBoundingClientRect();
                        const cellRect = cell.getBoundingClientRect();
                        activeDotCenterDelta = Math.abs((dotRect.left + dotRect.width / 2) - (cellRect.left + cellRect.width / 2));
                      }
                      return {
                        headers: [...document.querySelectorAll("#glmModelGrid .tabulator-col-title")]
                          .map((node) => node.textContent.trim()).filter(Boolean),
                        rows: document.querySelectorAll("#glmModelGrid .tabulator-row").length,
                        activeDots: document.querySelectorAll("#glmModelGrid .glm-model-active-dot").length,
                        activeDotRowText: dot?.closest(".tabulator-row")?.textContent || "",
                        activeDotCenterDelta,
                        selectedRows: document.querySelectorAll("#glmModelGrid .tabulator-row.tabulator-selected").length,
                        renameDisabled: document.querySelector("#glmRenameModelBtn")?.disabled,
                        activateDisabled: document.querySelector("#glmActivateModelBtn")?.disabled,
                        deleteDisabled: document.querySelector("#glmDeleteModelBtn")?.disabled,
                      };
                    }
                    """
                )
                self.assertEqual(
                    glm_navigator_state["headers"],
                    ["Model", "Created", "Response", "Weight", "Family", "Deviance", "AIC", "BIC", "Rows"],
                )
                self.assertEqual(glm_navigator_state["rows"], 4)
                self.assertEqual(glm_navigator_state["activeDots"], 1)
                self.assertIn("Browser smoke GLM", glm_navigator_state["activeDotRowText"])
                self.assertIsNotNone(glm_navigator_state["activeDotCenterDelta"])
                self.assertLessEqual(glm_navigator_state["activeDotCenterDelta"], 1.5)
                self.assertEqual(glm_navigator_state["selectedRows"], 0)
                self.assertTrue(glm_navigator_state["renameDisabled"])
                self.assertTrue(glm_navigator_state["activateDisabled"])
                self.assertTrue(glm_navigator_state["deleteDisabled"])
                page.locator("#glmModelGrid .tabulator-row", has_text="Disposable smoke GLM A").click()
                page.locator("#glmModelGrid .tabulator-row", has_text="Disposable smoke GLM B").click()
                plain_glm_selection = page.evaluate(
                    """
                    () => ({
                      selectedRows: document.querySelectorAll("#glmModelGrid .tabulator-row.tabulator-selected").length,
                      selectedText: [...document.querySelectorAll("#glmModelGrid .tabulator-row.tabulator-selected")]
                        .map((row) => row.textContent),
                      renameDisabled: document.querySelector("#glmRenameModelBtn")?.disabled,
                      activateDisabled: document.querySelector("#glmActivateModelBtn")?.disabled,
                      deleteDisabled: document.querySelector("#glmDeleteModelBtn")?.disabled,
                    })
                    """
                )
                self.assertEqual(plain_glm_selection["selectedRows"], 1)
                self.assertTrue(any("Disposable smoke GLM B" in text for text in plain_glm_selection["selectedText"]))
                self.assertFalse(any("Disposable smoke GLM A" in text for text in plain_glm_selection["selectedText"]))
                self.assertFalse(plain_glm_selection["renameDisabled"])
                self.assertFalse(plain_glm_selection["activateDisabled"])
                self.assertFalse(plain_glm_selection["deleteDisabled"])
                page.locator("#glmModelGrid .tabulator-row", has_text="Disposable smoke GLM A").click(modifiers=["Shift"])
                shift_glm_selection = page.evaluate(
                    """
                    () => ({
                      selectedRows: document.querySelectorAll("#glmModelGrid .tabulator-row.tabulator-selected").length,
                      selectedText: [...document.querySelectorAll("#glmModelGrid .tabulator-row.tabulator-selected")]
                        .map((row) => row.textContent),
                      renameDisabled: document.querySelector("#glmRenameModelBtn")?.disabled,
                      activateDisabled: document.querySelector("#glmActivateModelBtn")?.disabled,
                      deleteDisabled: document.querySelector("#glmDeleteModelBtn")?.disabled,
                    })
                    """
                )
                self.assertEqual(shift_glm_selection["selectedRows"], 2)
                self.assertTrue(any("Disposable smoke GLM A" in text for text in shift_glm_selection["selectedText"]))
                self.assertTrue(any("Disposable smoke GLM B" in text for text in shift_glm_selection["selectedText"]))
                self.assertTrue(shift_glm_selection["renameDisabled"])
                self.assertTrue(shift_glm_selection["activateDisabled"])
                self.assertFalse(shift_glm_selection["deleteDisabled"])
                row_selection_modifier = "Meta" if sys.platform == "darwin" else "Control"
                page.locator("#glmModelGrid .tabulator-row", has_text="Second smoke GLM").click(modifiers=[row_selection_modifier])
                command_glm_selection = page.evaluate(
                    """
                    () => ({
                      selectedRows: document.querySelectorAll("#glmModelGrid .tabulator-row.tabulator-selected").length,
                      selectedText: [...document.querySelectorAll("#glmModelGrid .tabulator-row.tabulator-selected")]
                        .map((row) => row.textContent),
                      renameDisabled: document.querySelector("#glmRenameModelBtn")?.disabled,
                      activateDisabled: document.querySelector("#glmActivateModelBtn")?.disabled,
                      deleteDisabled: document.querySelector("#glmDeleteModelBtn")?.disabled,
                    })
                    """
                )
                self.assertEqual(command_glm_selection["selectedRows"], 3)
                self.assertTrue(any("Second smoke GLM" in text for text in command_glm_selection["selectedText"]))
                self.assertTrue(any("Disposable smoke GLM A" in text for text in command_glm_selection["selectedText"]))
                self.assertTrue(any("Disposable smoke GLM B" in text for text in command_glm_selection["selectedText"]))
                self.assertTrue(command_glm_selection["renameDisabled"])
                self.assertTrue(command_glm_selection["activateDisabled"])
                self.assertFalse(command_glm_selection["deleteDisabled"])
                page.locator("#glmModelGrid .tabulator-row", has_text="Second smoke GLM").click(modifiers=[row_selection_modifier])
                toggled_glm_selection = page.evaluate(
                    """
                    () => ({
                      selectedRows: document.querySelectorAll("#glmModelGrid .tabulator-row.tabulator-selected").length,
                      selectedText: [...document.querySelectorAll("#glmModelGrid .tabulator-row.tabulator-selected")]
                        .map((row) => row.textContent),
                      renameDisabled: document.querySelector("#glmRenameModelBtn")?.disabled,
                      activateDisabled: document.querySelector("#glmActivateModelBtn")?.disabled,
                      deleteDisabled: document.querySelector("#glmDeleteModelBtn")?.disabled,
                    })
                    """
                )
                self.assertEqual(toggled_glm_selection["selectedRows"], 2)
                self.assertFalse(any("Second smoke GLM" in text for text in toggled_glm_selection["selectedText"]))
                self.assertTrue(any("Disposable smoke GLM A" in text for text in toggled_glm_selection["selectedText"]))
                self.assertTrue(any("Disposable smoke GLM B" in text for text in toggled_glm_selection["selectedText"]))
                self.assertTrue(toggled_glm_selection["renameDisabled"])
                self.assertTrue(toggled_glm_selection["activateDisabled"])
                self.assertFalse(toggled_glm_selection["deleteDisabled"])
                page.evaluate("() => { window.confirm = () => true; }")
                page.locator("#glmDeleteModelBtn").click()
                page.wait_for_function(
                    """
                    () => document.querySelectorAll("#glmModelGrid .tabulator-row").length === 2
                      && !document.body.textContent.includes("Disposable smoke GLM A")
                      && !document.body.textContent.includes("Disposable smoke GLM B")
                      && document.querySelector("#glmModelSelectedMeta")?.textContent.includes("Browser smoke GLM")
                      && document.querySelector(".dataset-meta-glm-link")?.textContent.trim() === "GLMs (2)"
                    """,
                    timeout=10_000,
                )
                page.locator("#glmModelGrid .tabulator-row", has_text="Second smoke GLM").click()
                selected_glm_navigator_state = page.evaluate(
                    """
                    () => ({
                      selectedRows: document.querySelectorAll("#glmModelGrid .tabulator-row.tabulator-selected").length,
                      renameDisabled: document.querySelector("#glmRenameModelBtn")?.disabled,
                      activateDisabled: document.querySelector("#glmActivateModelBtn")?.disabled,
                      deleteDisabled: document.querySelector("#glmDeleteModelBtn")?.disabled,
                    })
                    """
                )
                self.assertEqual(selected_glm_navigator_state["selectedRows"], 1)
                self.assertFalse(selected_glm_navigator_state["renameDisabled"])
                self.assertFalse(selected_glm_navigator_state["activateDisabled"])
                self.assertFalse(selected_glm_navigator_state["deleteDisabled"])

                page.locator("#glmActivateModelBtn").click()
                page.wait_for_function(
                    """
                    () => document.querySelector("#glmModelSelectedMeta")?.textContent.includes("Second smoke GLM")
                      && document.querySelector("#glmModelGrid .glm-model-active-dot")?.closest(".tabulator-row")?.textContent.includes("Second smoke GLM")
                    """,
                    timeout=10_000,
                )
                with page.expect_request(lambda request: request.url.endswith("/api/chart"), timeout=10_000) as glm_navigator_chart_info:
                    page.locator("#lineBarTool").click()
                glm_navigator_chart_body = json.loads(glm_navigator_chart_info.value.post_data or "{}")
                self.assertEqual(glm_navigator_chart_body["source"], "glm:browser-smoke-glm-2:predictions")
                self.assertEqual(glm_navigator_chart_body["x"], "Segment")
                self.assertEqual(glm_navigator_chart_body["responses"][1]["numerator"], "glm_prediction")
                self.assertEqual(glm_navigator_chart_body["responses"][1]["source"], "glm:browser-smoke-glm-2:predictions")
                if page.locator("#lineBarSideControlsToggleBtn").get_attribute("aria-expanded") == "false":
                    page.locator("#lineBarSideControlsToggleBtn").click()
                    page.wait_for_function(
                        """
                        () => document.querySelector("#lineBarSideControlsToggleBtn")?.getAttribute("aria-expanded") === "true"
                          && getComputedStyle(document.querySelector("#chartSideControls")).display !== "none"
                        """,
                        timeout=10_000,
                    )
                page.locator("#featureList .feature.active", has_text="Segment").wait_for(timeout=10_000)
                page.locator("#glmTool").click()
                page.locator(".glm-tool").wait_for(timeout=10_000)
                page.get_by_role("button", name="Model navigator").click()
                page.locator("#glmModelGrid .tabulator-row", has_text="Browser smoke GLM").click()
                page.locator("#glmActivateModelBtn").click()
                page.wait_for_function(
                    """
                    () => document.querySelector("#glmModelSelectedMeta")?.textContent.includes("Browser smoke GLM")
                      && document.querySelector("#glmModelGrid .glm-model-active-dot")?.closest(".tabulator-row")?.textContent.includes("Browser smoke GLM")
                    """,
                    timeout=10_000,
                )

                if page.evaluate("() => document.body.classList.contains('dark')"):
                    page.locator("#themeBtn").click()
                    page.wait_for_function("() => !document.body.classList.contains('dark')", timeout=10_000)

                page.get_by_role("button", name="Tabulations").click()
                page.locator("#glmTabulationModelGrid .tabulator-row").first.wait_for(timeout=10_000)
                page.locator("#glmTabulationTableGrid .tabulator-row", has_text="Age").click()
                page.locator("#glmTabulationTable .tabulator-row").first.wait_for(timeout=10_000)
                if not page.locator("#glmTabulationColor").is_checked():
                    page.locator("#glmTabulationColor").check()
                    page.locator("#glmTabulationTable .glm-tabulation-colour-cell").first.wait_for(timeout=10_000)
                page.locator('[data-glm-tabulation-scale="exp"]').click()
                page.wait_for_function(
                    """
                    () => {
                      const ageRow = [...document.querySelectorAll("#glmTabulationTableGrid .tabulator-row")]
                        .find((row) => row.textContent.includes("Age"));
                      return ageRow?.querySelector('.tabulator-cell[tabulator-field="display_span"]')?.textContent.trim() === "2.7183";
                    }
                    """,
                    timeout=10_000,
                )
                page.locator("#glmTabulationTable .glm-tabulation-colour-cell").first.wait_for(timeout=10_000)
                tabulation_single_state = page.evaluate(
                    """
                    () => {
                      const ageRow = [...document.querySelectorAll("#glmTabulationTableGrid .tabulator-row")]
                        .find((row) => row.textContent.includes("Age"));
                      const modelRows = [...document.querySelectorAll("#glmTabulationModelGrid .tabulator-row")];
                      const tabulatedGlmRow = modelRows.find((row) => row.textContent.includes("Browser smoke GLM"));
                      const blockedGbmRow = modelRows.find((row) => row.textContent.includes("Second smoke model"));
                      const untabulatedGbmRow = modelRows.find((row) => row.textContent.includes("Browser smoke model"));
                      const selectedModelCell = document.querySelector("#glmTabulationModelGrid .tabulator-row.tabulator-selected .tabulator-cell");
                      const unselectedModelCell = modelRows
                        .find((row) => !row.classList.contains("tabulator-selected"))
                        ?.querySelector(".tabulator-cell");
                      const selectedTableCell = document.querySelector("#glmTabulationTableGrid .tabulator-row.tabulator-selected .tabulator-cell");
                      const unselectedTableCell = [...document.querySelectorAll("#glmTabulationTableGrid .tabulator-row")]
                        .find((row) => !row.classList.contains("tabulator-selected"))
                        ?.querySelector(".tabulator-cell");
                      return {
                        modelHeaders: [...document.querySelectorAll("#glmTabulationModelGrid .tabulator-col-title")]
                          .map((node) => node.textContent.trim()).filter(Boolean),
                        modelTypes: modelRows.map((row) => row.querySelector('.tabulator-cell[tabulator-field="model_type"]')?.textContent.trim()).filter(Boolean),
                        selectedModels: document.querySelectorAll("#glmTabulationModelGrid .tabulator-row.tabulator-selected").length,
                        selectedModelBackground: selectedModelCell ? getComputedStyle(selectedModelCell).backgroundColor : "",
                        unselectedModelBackground: unselectedModelCell ? getComputedStyle(unselectedModelCell).backgroundColor : "",
                        blockedGbmPresent: Boolean(blockedGbmRow),
                        blockedGbmTextPresent: modelRows.some((row) => row.textContent.includes("n/a: >3 leaves")),
                        untabulatedGbmCountText: untabulatedGbmRow?.querySelector('.tabulator-cell[tabulator-field="table_count"]')?.textContent.trim() || "",
                        untabulatedGbmMeanText: untabulatedGbmRow?.querySelector('.tabulator-cell[tabulator-field="mean_error"]')?.textContent.trim() || "",
                        untabulatedGbmSdText: untabulatedGbmRow?.querySelector('.tabulator-cell[tabulator-field="linear_sd_error"]')?.textContent.trim() || "",
                        untabulatedGbmMissingText: untabulatedGbmRow?.querySelector('.tabulator-cell[tabulator-field="missing"]')?.textContent.trim() || "",
                        untabulatedGbmNameColor: untabulatedGbmRow?.querySelector('.tabulator-cell[tabulator-field="model_name"]')
                          ? getComputedStyle(untabulatedGbmRow.querySelector('.tabulator-cell[tabulator-field="model_name"]')).color
                          : "",
                        tabulatedGlmNameColor: tabulatedGlmRow?.querySelector('.tabulator-cell[tabulator-field="model_name"]')
                          ? getComputedStyle(tabulatedGlmRow.querySelector('.tabulator-cell[tabulator-field="model_name"]')).color
                          : "",
                        modelNameColumnWidth: document.querySelector("#glmTabulationModelGrid .tabulator-col[tabulator-field='model_name']")?.getBoundingClientRect().width || 0,
                        missingColumnWidth: document.querySelector("#glmTabulationModelGrid .tabulator-col[tabulator-field='missing']")?.getBoundingClientRect().width || 0,
                        tableHeaders: [...document.querySelectorAll("#glmTabulationTableGrid .tabulator-col-title")]
                          .map((node) => node.textContent.trim()).filter(Boolean),
                        selectedTableBackground: selectedTableCell ? getComputedStyle(selectedTableCell).backgroundColor : "",
                        unselectedTableBackground: unselectedTableCell ? getComputedStyle(unselectedTableCell).backgroundColor : "",
                        tableNameColumnWidth: document.querySelector("#glmTabulationTableGrid .tabulator-col[tabulator-field='table_name']")?.getBoundingClientRect().width || 0,
                        spanColumnWidth: document.querySelector("#glmTabulationTableGrid .tabulator-col[tabulator-field='display_span']")?.getBoundingClientRect().width || 0,
                        ageMin: ageRow?.querySelector('.tabulator-cell[tabulator-field="display_min"]')?.textContent.trim() || "",
                        ageMax: ageRow?.querySelector('.tabulator-cell[tabulator-field="display_max"]')?.textContent.trim() || "",
                        ageSpan: ageRow?.querySelector('.tabulator-cell[tabulator-field="display_span"]')?.textContent.trim() || "",
                        exportText: document.querySelector("#glmExportTabulationsBtn")?.textContent.trim() || "",
                        exportDisabled: document.querySelector("#glmExportTabulationsBtn")?.disabled,
                        diagnosticsHidden: document.querySelector("#glmTabulationDiagnostics")?.classList.contains("hidden"),
                        splitDelta: Math.abs(
                          (document.querySelector(".glm-tabulation-sidebar")?.getBoundingClientRect().width || 0)
                          - (document.querySelector(".glm-tabulation-main")?.getBoundingClientRect().width || 0)
                        ),
                        resultHeaders: [...document.querySelectorAll("#glmTabulationTable .tabulator-col-title")]
                          .map((node) => node.textContent.trim()).filter(Boolean),
                      };
                    }
                    """
                )
                self.assertEqual(
                    tabulation_single_state["modelHeaders"],
                    ["Model name", "Model type", "Number of tables", "Mean error", "linear SD error", "missing"],
                )
                self.assertIn("GLM", tabulation_single_state["modelTypes"])
                self.assertIn("GBM", tabulation_single_state["modelTypes"])
                self.assertEqual(tabulation_single_state["selectedModels"], 1)
                self.assertNotEqual(tabulation_single_state["selectedModelBackground"], tabulation_single_state["unselectedModelBackground"])
                self.assertFalse(tabulation_single_state["blockedGbmPresent"])
                self.assertFalse(tabulation_single_state["blockedGbmTextPresent"])
                self.assertEqual(tabulation_single_state["untabulatedGbmCountText"], "not tabulated")
                self.assertEqual(tabulation_single_state["untabulatedGbmMeanText"], "")
                self.assertEqual(tabulation_single_state["untabulatedGbmSdText"], "")
                self.assertEqual(tabulation_single_state["untabulatedGbmMissingText"], "")
                self.assertNotEqual(tabulation_single_state["untabulatedGbmNameColor"], tabulation_single_state["tabulatedGlmNameColor"])
                self.assertGreater(tabulation_single_state["modelNameColumnWidth"], tabulation_single_state["missingColumnWidth"])
                self.assertEqual(tabulation_single_state["tableHeaders"], ["#", "Table name", "Dim", "Cells", "Min", "Max", "Span"])
                self.assertNotEqual(tabulation_single_state["selectedTableBackground"], tabulation_single_state["unselectedTableBackground"])
                self.assertGreater(tabulation_single_state["tableNameColumnWidth"], tabulation_single_state["spanColumnWidth"])
                self.assertEqual(tabulation_single_state["ageMin"], "1")
                self.assertEqual(tabulation_single_state["ageMax"], "2.7183")
                self.assertEqual(tabulation_single_state["ageSpan"], "2.7183")
                self.assertEqual(tabulation_single_state["exportText"], "Export xlsx")
                self.assertFalse(tabulation_single_state["exportDisabled"])
                self.assertTrue(tabulation_single_state["diagnosticsHidden"])
                self.assertLess(tabulation_single_state["splitDelta"], 36)
                self.assertIn("Age", tabulation_single_state["resultHeaders"])

                tabulation_light_colour = page.evaluate(
                    """
                    () => {
                      const parseColor = (value) => {
                        const text = String(value || "").trim();
                        const rgb = text.match(/^rgba?\\(([^)]+)\\)$/i);
                        if (rgb) {
                          return rgb[1].replace(/\\//g, " ").split(/[\\s,]+/).filter(Boolean).slice(0, 3).map(Number);
                        }
                        const srgb = text.match(/^color\\(srgb\\s+([0-9.]+)\\s+([0-9.]+)\\s+([0-9.]+)/i);
                        if (srgb) return [Number(srgb[1]) * 255, Number(srgb[2]) * 255, Number(srgb[3]) * 255];
                        return [0, 0, 0];
                      };
                      const relativeLuminance = (color) => {
                        const [r, g, b] = parseColor(color).map((channel) => {
                          const value = channel / 255;
                          return value <= 0.03928 ? value / 12.92 : ((value + 0.055) / 1.055) ** 2.4;
                        });
                        return 0.2126 * r + 0.7152 * g + 0.0722 * b;
                      };
                      const contrast = (a, b) => {
                        const first = relativeLuminance(a);
                        const second = relativeLuminance(b);
                        return (Math.max(first, second) + 0.05) / (Math.min(first, second) + 0.05);
                      };
                      const cell = document.querySelector("#glmTabulationTable .glm-tabulation-colour-cell");
                      const styles = cell ? getComputedStyle(cell) : null;
                      return {
                        dark: document.body.classList.contains("dark"),
                        background: styles?.backgroundColor || "",
                        color: styles?.color || "",
                        contrast: styles ? contrast(styles.color, styles.backgroundColor) : 0,
                      };
                    }
                    """
                )
                self.assertFalse(tabulation_light_colour["dark"])
                self.assertGreaterEqual(tabulation_light_colour["contrast"], 4.5)
                page.locator("#themeBtn").click()
                page.wait_for_function(
                    """
                    (lightBackground) => {
                      const cell = document.querySelector("#glmTabulationTable .glm-tabulation-colour-cell");
                      return document.body.classList.contains("dark")
                        && cell
                        && getComputedStyle(cell).backgroundColor !== lightBackground;
                    }
                    """,
                    arg=tabulation_light_colour["background"],
                    timeout=10_000,
                )
                tabulation_dark_colour = page.evaluate(
                    """
                    () => {
                      const parseColor = (value) => {
                        const text = String(value || "").trim();
                        const rgb = text.match(/^rgba?\\(([^)]+)\\)$/i);
                        if (rgb) {
                          return rgb[1].replace(/\\//g, " ").split(/[\\s,]+/).filter(Boolean).slice(0, 3).map(Number);
                        }
                        const srgb = text.match(/^color\\(srgb\\s+([0-9.]+)\\s+([0-9.]+)\\s+([0-9.]+)/i);
                        if (srgb) return [Number(srgb[1]) * 255, Number(srgb[2]) * 255, Number(srgb[3]) * 255];
                        return [0, 0, 0];
                      };
                      const relativeLuminance = (color) => {
                        const [r, g, b] = parseColor(color).map((channel) => {
                          const value = channel / 255;
                          return value <= 0.03928 ? value / 12.92 : ((value + 0.055) / 1.055) ** 2.4;
                        });
                        return 0.2126 * r + 0.7152 * g + 0.0722 * b;
                      };
                      const contrast = (a, b) => {
                        const first = relativeLuminance(a);
                        const second = relativeLuminance(b);
                        return (Math.max(first, second) + 0.05) / (Math.min(first, second) + 0.05);
                      };
                      const cell = document.querySelector("#glmTabulationTable .glm-tabulation-colour-cell");
                      const styles = cell ? getComputedStyle(cell) : null;
                      return {
                        dark: document.body.classList.contains("dark"),
                        background: styles?.backgroundColor || "",
                        color: styles?.color || "",
                        contrast: styles ? contrast(styles.color, styles.backgroundColor) : 0,
                      };
                    }
                    """
                )
                self.assertTrue(tabulation_dark_colour["dark"])
                self.assertNotEqual(tabulation_dark_colour["background"], tabulation_light_colour["background"])
                self.assertNotEqual(tabulation_dark_colour["color"], tabulation_light_colour["color"])
                self.assertGreaterEqual(tabulation_dark_colour["contrast"], 4.5)
                page.locator("#themeBtn").click()
                page.wait_for_function("() => !document.body.classList.contains('dark')", timeout=10_000)

                page.locator("#glmExportTabulationsBtn").click()
                page.locator("#glmTabulationNotice", has_text="Saved XLSX:").wait_for(timeout=10_000)
                export_path = page.locator("#glmTabulationNotice").text_content(timeout=10_000).replace("Saved XLSX:", "", 1).strip()
                self.assertTrue(export_path.endswith("browser-smoke-glm_tabulations_exp.xlsx"))
                self.assertTrue(Path(export_path).exists(), export_path)

                page.locator('[data-glm-tabulation-view="plot"]').click()
                page.locator("#glmTabulationPlot canvas").first.wait_for(timeout=10_000)
                initial_plot_width = page.evaluate(
                    """
                    () => window.echarts.getInstanceByDom(document.querySelector("#glmTabulationPlot"))?.getWidth() || 0
                    """
                )
                self.assertGreater(initial_plot_width, 0)
                page.locator("#sidebarToggleBtn").click()
                page.wait_for_function(
                    """
                    (initialWidth) => {
                      if (document.querySelector("#sidebarToggleBtn")?.getAttribute("aria-expanded") !== "false") return false;
                      const chart = window.echarts.getInstanceByDom(document.querySelector("#glmTabulationPlot"));
                      return chart && chart.getWidth() > initialWidth + 20;
                    }
                    """,
                    arg=initial_plot_width,
                    timeout=10_000,
                )
                collapsed_plot_width = page.evaluate(
                    """
                    () => window.echarts.getInstanceByDom(document.querySelector("#glmTabulationPlot"))?.getWidth() || 0
                    """
                )
                page.locator("#sidebarToggleBtn").click()
                page.wait_for_function(
                    """
                    (collapsedWidth) => {
                      if (document.querySelector("#sidebarToggleBtn")?.getAttribute("aria-expanded") !== "true") return false;
                      const chart = window.echarts.getInstanceByDom(document.querySelector("#glmTabulationPlot"));
                      return chart && chart.getWidth() < collapsedWidth - 20;
                    }
                    """,
                    arg=collapsed_plot_width,
                    timeout=10_000,
                )
                page.locator('[data-glm-tabulation-view="table"]').click()
                page.locator("#glmTabulationTable .tabulator-row").first.wait_for(timeout=10_000)

                page.locator("#glmTabulationModelGrid .tabulator-row", has_text="Second smoke GLM").click(modifiers=[row_selection_modifier])
                page.wait_for_function(
                    """
                    () => document.querySelectorAll("#glmTabulationModelGrid .tabulator-row.tabulator-selected").length === 2
                      && document.querySelectorAll("#glmTabulationCommonTableGrid .tabulator-row").length > 0
                      && document.querySelectorAll("#glmTabulationOtherTableGrid .tabulator-row").length > 0
                    """,
                    timeout=10_000,
                )
                tabulation_multi_state = page.evaluate(
                    """
                    () => ({
                      selectedModels: [...document.querySelectorAll("#glmTabulationModelGrid .tabulator-row.tabulator-selected")]
                        .map((row) => row.textContent),
                      commonHeaders: [...document.querySelectorAll("#glmTabulationCommonTableGrid .tabulator-col-title")]
                        .map((node) => node.textContent.trim()).filter(Boolean),
                      otherHeaders: [...document.querySelectorAll("#glmTabulationOtherTableGrid .tabulator-col-title")]
                        .map((node) => node.textContent.trim()).filter(Boolean),
                      commonRows: [...document.querySelectorAll("#glmTabulationCommonTableGrid .tabulator-row")]
                        .map((row) => row.textContent),
                      otherRows: [...document.querySelectorAll("#glmTabulationOtherTableGrid .tabulator-row")]
                        .map((row) => row.textContent),
                      commonSelectedRows: [...document.querySelectorAll("#glmTabulationCommonTableGrid .tabulator-row.tabulator-selected")]
                        .map((row) => row.textContent),
                      otherSelectedRows: [...document.querySelectorAll("#glmTabulationOtherTableGrid .tabulator-row.tabulator-selected")]
                        .map((row) => row.textContent),
                      exportDisabled: document.querySelector("#glmExportTabulationsBtn")?.disabled,
                    })
                    """
                )
                self.assertTrue(any("Browser smoke GLM" in text for text in tabulation_multi_state["selectedModels"]))
                self.assertTrue(any("Second smoke GLM" in text for text in tabulation_multi_state["selectedModels"]))
                self.assertEqual(tabulation_multi_state["commonHeaders"], ["#", "Table name", "Dim"])
                self.assertEqual(tabulation_multi_state["otherHeaders"], ["#", "Table name", "Dim"])
                self.assertTrue(any("Age" in text for text in tabulation_multi_state["commonRows"]))
                self.assertTrue(any("Segment" in text for text in tabulation_multi_state["otherRows"]))
                self.assertEqual(len(tabulation_multi_state["commonSelectedRows"]), 1)
                self.assertTrue(any("Age" in text for text in tabulation_multi_state["commonSelectedRows"]))
                self.assertEqual(tabulation_multi_state["otherSelectedRows"], [])
                self.assertTrue(tabulation_multi_state["exportDisabled"])

                page.locator("#glmTabulationOtherTableGrid .tabulator-row", has_text="Segment").click()
                page.locator("#glmTabulationTable .tabulator-row").first.wait_for(timeout=10_000)
                page.locator("#glmTabulationNotice", has_text="has no Segment tabulation").wait_for(timeout=10_000)
                tabulation_other_selected_state = page.evaluate(
                    """
                    () => ({
                      commonSelectedRows: [...document.querySelectorAll("#glmTabulationCommonTableGrid .tabulator-row.tabulator-selected")]
                        .map((row) => row.textContent),
                      otherSelectedRows: [...document.querySelectorAll("#glmTabulationOtherTableGrid .tabulator-row.tabulator-selected")]
                        .map((row) => row.textContent),
                    })
                    """
                )
                self.assertEqual(tabulation_other_selected_state["commonSelectedRows"], [])
                self.assertEqual(len(tabulation_other_selected_state["otherSelectedRows"]), 1)
                self.assertTrue(any("Segment" in text for text in tabulation_other_selected_state["otherSelectedRows"]))

                glm_job_succeed = {"value": False}
                glm_build_payload = {"value": None}

                def glm_build_route(route: Any) -> None:
                    glm_build_payload["value"] = json.loads(route.request.post_data or "{}")
                    route.fulfill(
                        status=200,
                        content_type="application/json",
                        body=json.dumps(
                            {
                                "job_id": "glm-live-job",
                                "status": "queued",
                                "created_at": "2026-05-25T00:00:00Z",
                                "updated_at": "2026-05-25T00:00:00Z",
                                "result": None,
                                "error": None,
                                "progress": None,
                            }
                        ),
                    )

                def glm_job_route(route: Any) -> None:
                    if glm_job_succeed["value"]:
                        payload = {
                            "job_id": "glm-live-job",
                            "status": "succeeded",
                            "created_at": "2026-05-25T00:00:00Z",
                            "updated_at": "2026-05-25T00:00:01Z",
                            "result": {"model_id": "browser-smoke-glm", "sources": {}},
                            "error": None,
                            "progress": {
                                "phase": "succeeded",
                                "message": "GLM training complete",
                                "training_rows": 3,
                            },
                        }
                    else:
                        payload = {
                            "job_id": "glm-live-job",
                            "status": "running",
                            "created_at": "2026-05-25T00:00:00Z",
                            "updated_at": "2026-05-25T00:00:01Z",
                            "result": None,
                            "error": None,
                            "progress": {
                                "phase": "fitting",
                                "message": "Fitting GLM",
                                "training_rows": 3,
                            },
                        }
                    route.fulfill(status=200, content_type="application/json", body=json.dumps(payload))

                page.route("**/api/glm/build", glm_build_route)
                page.route("**/api/glm/jobs/glm-live-job", glm_job_route)
                page.get_by_role("button", name="Formula builder").click()
                page.evaluate(
                    """
                    () => {
                      window.__glmBusyPointerMoves = 0;
                      document.addEventListener("pointermove", () => { window.__glmBusyPointerMoves += 1; }, true);
                    }
                    """
                )
                page.locator("#glmBuildBtn").hover()
                page.locator("#glmBuildBtn").click()
                page.locator("#glmBuildStatus").get_by_text("Fitting GLM").wait_for(timeout=10_000)
                self.assertEqual(glm_build_payload["value"]["family"], "tweedie")
                glm_busy_button = page.locator("#glmBuildBtn").evaluate(
                    """
                    (button) => {
                      const style = getComputedStyle(button);
                      const spinner = getComputedStyle(button, "::before");
                      return {
                        text: button.textContent.trim(),
                        disabled: button.disabled,
                        ariaBusy: button.getAttribute("aria-busy"),
                        building: button.classList.contains("building"),
                        cursor: style.cursor,
                        background: style.backgroundColor,
                        spinnerContent: spinner.content,
                        spinnerWidth: spinner.width,
                        spinnerAnimation: spinner.animationName,
                      };
                    }
                    """
                )
                self.assertEqual(glm_busy_button["text"], "Building...")
                self.assertTrue(glm_busy_button["disabled"])
                self.assertEqual(glm_busy_button["ariaBusy"], "true")
                self.assertTrue(glm_busy_button["building"])
                self.assertEqual(glm_busy_button["cursor"], "pointer")
                self.assertEqual(glm_busy_button["background"], "rgb(217, 119, 6)")
                self.assertEqual(glm_busy_button["spinnerContent"], '""')
                self.assertEqual(glm_busy_button["spinnerWidth"], "12px")
                self.assertEqual(glm_busy_button["spinnerAnimation"], "model-busy-button-spin")
                glm_pointer_moves_while_busy = page.evaluate("window.__glmBusyPointerMoves")
                glm_job_succeed["value"] = True
                page.locator("#glmBuildBtn", has_text="Build GLM").wait_for(timeout=10_000)
                page.locator("#startupProgress.ready", has_text="Ready").wait_for(timeout=10_000)
                page.wait_for_function(
                    """
                    () => {
                      const status = document.querySelector("#glmBuildStatus");
                      return status?.classList.contains("hidden")
                        && !status.textContent.includes("GLM training complete")
                        && !status.textContent.includes("training rows");
                    }
                    """,
                    timeout=10_000,
                )
                glm_ready_button = page.locator("#glmBuildBtn").evaluate(
                    """
                    (button) => {
                      const style = getComputedStyle(button);
                      const spinner = getComputedStyle(button, "::before");
                      return {
                        text: button.textContent.trim(),
                        disabled: button.disabled,
                        ariaBusy: button.getAttribute("aria-busy"),
                        building: button.classList.contains("building"),
                        cursor: style.cursor,
                        background: style.backgroundColor,
                        spinnerContent: spinner.content,
                      };
                    }
                    """
                )
                self.assertEqual(glm_ready_button["text"], "Build GLM")
                self.assertFalse(glm_ready_button["disabled"])
                self.assertIsNone(glm_ready_button["ariaBusy"])
                self.assertFalse(glm_ready_button["building"])
                self.assertEqual(glm_ready_button["cursor"], "pointer")
                self.assertEqual(glm_ready_button["background"], "rgb(21, 128, 61)")
                self.assertEqual(glm_ready_button["spinnerContent"], "none")
                self.assertEqual(page.evaluate("window.__glmBusyPointerMoves"), glm_pointer_moves_while_busy)
                page.unroute("**/api/glm/build", glm_build_route)
                page.unroute("**/api/glm/jobs/glm-live-job", glm_job_route)

                shap_url = (
                    f"{base_url}/?tool=line_bar&source=gbm%3Abrowser-smoke-model-2%3Ashap_long"
                    "&x=Age&actual=SHAP__Age&denominator=denominator"
                )
                page.goto(shap_url, wait_until="domcontentloaded")
                page.locator("#chart:not(.hidden)").wait_for(timeout=10_000)
                page.wait_for_function(
                    '() => document.querySelector("#lineBarGroupMeta")?.textContent.includes("groups")',
                    timeout=10_000,
                )
                actual_state = page.evaluate(
                    """
                    () => {
                      const select = document.querySelector("#actualNumerator");
                      return {
                        value: select.value,
                        selectedSource: select.selectedOptions[0]?.dataset.sourceId || "",
                        groups: [...select.querySelectorAll("optgroup")].map((group) => ({
                          label: group.label,
                          options: [...group.querySelectorAll("option")].map((option) => ({
                            text: option.textContent,
                            value: option.value,
                            source: option.dataset.sourceId || "",
                            disabled: option.disabled,
                          })),
                        })),
                      };
                    }
                    """
                )
                self.assertEqual([group["label"] for group in actual_state["groups"]], ["Dataset features", "Model predictions", "SHAP values"])
                for group in actual_state["groups"]:
                    option_texts = [option["text"] for option in group["options"] if not option["disabled"]]
                    self.assertEqual(option_texts, sorted(option_texts, key=str.casefold))
                weight_options = page.evaluate(
                    """
                    () => [...document.querySelectorAll("#denominator option")].map((option) => option.textContent || "")
                    """
                )
                self.assertEqual(weight_options[0], "Average row value")
                self.assertEqual(weight_options[1:], sorted(weight_options[1:], key=str.casefold))
                self.assertEqual(actual_state["value"], "SHAP__Age")
                self.assertEqual(actual_state["selectedSource"], "gbm:browser-smoke-model-2:shap_long")
                shap_options = next(group for group in actual_state["groups"] if group["label"] == "SHAP values")["options"]
                self.assertEqual([option["source"] for option in shap_options if not option["disabled"]], ["gbm:browser-smoke-model-2:shap_long"])
                self.assertIn("Age", [option["text"] for option in shap_options])
                expected_options = page.evaluate(
                    """
                    () => [...document.querySelectorAll("#expectedList .feature")]
                      .map((button) => button.textContent || "")
                    """
                )
                self.assertTrue(any("gbm_prediction" in option for option in expected_options))
                self.assertFalse(any("SHAP__" in option for option in expected_options))

                chart_requests_before = chart_requests
                open_sidebar_section("#gbmModelCollapseBtn")
                with page.expect_request(lambda request: request.url.endswith("/api/chart"), timeout=10_000) as shap_request_info:
                    page.locator('#gbmModelSelect [data-gbm-model-id="browser-smoke-model"]').click()
                shap_request_body = json.loads(shap_request_info.value.post_data or "{}")
                page.locator("#gbmModelSelectedMeta", has_text="Browser smoke model").wait_for(timeout=10_000)
                self.assertGreater(chart_requests, chart_requests_before)
                self.assertEqual(shap_request_body["source"], "gbm:browser-smoke-model:shap_long")
                self.assertEqual(shap_request_body["responses"][0]["numerator"], "SHAP__Age")
                self.assertEqual(page_errors, [])
            finally:
                browser.close()

    def exercise_specs_tool_layout(self, base_url: str) -> None:
        assert sync_playwright is not None
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch()
            page = browser.new_page(viewport={"width": 1280, "height": 800})
            page_errors: list[str] = []
            page.on("pageerror", lambda error: page_errors.append(str(error)))

            def assert_model_workspace_style(tool_selector: str) -> None:
                page.locator(f"#modelToolWrap:not(.hidden) {tool_selector}").wait_for(timeout=10_000)
                style = page.evaluate(
                    """
                    () => {
                      const main = document.querySelector("main");
                      const visual = document.querySelector("#visualArea");
                      const workspace = document.querySelector(".workspace");
                      const modelWrap = document.querySelector("#modelToolWrap");
                      const mainStyle = getComputedStyle(main);
                      const workspaceStyle = getComputedStyle(workspace);
                      const modelWrapStyle = getComputedStyle(modelWrap);
                      const panel = getComputedStyle(document.documentElement).getPropertyValue("--panel").trim();
                      const probe = document.createElement("div");
                      probe.style.background = panel;
                      probe.style.position = "absolute";
                      probe.style.visibility = "hidden";
                      document.body.appendChild(probe);
                      const panelBackground = getComputedStyle(probe).backgroundColor;
                      probe.remove();
                      const borderColors = [
                        workspaceStyle.borderTopColor,
                        workspaceStyle.borderRightColor,
                        workspaceStyle.borderBottomColor,
                        workspaceStyle.borderLeftColor,
                      ];
                      const borderRadii = [
                        workspaceStyle.borderTopLeftRadius,
                        workspaceStyle.borderTopRightRadius,
                        workspaceStyle.borderBottomRightRadius,
                        workspaceStyle.borderBottomLeftRadius,
                      ];
                      const mainPadding = [
                        mainStyle.paddingTop,
                        mainStyle.paddingRight,
                        mainStyle.paddingBottom,
                        mainStyle.paddingLeft,
                      ];
                      const workspacePadding = [
                        workspaceStyle.paddingTop,
                        workspaceStyle.paddingRight,
                        workspaceStyle.paddingBottom,
                        workspaceStyle.paddingLeft,
                      ];
                      const modelWrapPadding = [
                        modelWrapStyle.paddingTop,
                        modelWrapStyle.paddingRight,
                        modelWrapStyle.paddingBottom,
                        modelWrapStyle.paddingLeft,
                      ];
                      return {
                        borderColors,
                        borderRadii,
                        mainPadding,
                        modelWrapPadding,
                        mainBackground: mainStyle.backgroundColor,
                        panelBackground,
                        visualModelMode: visual.classList.contains("model-mode"),
                        workspaceBackground: workspaceStyle.backgroundColor,
                        workspaceBorderTransparent: borderColors.every((color) => color === "transparent" || color === "rgba(0, 0, 0, 0)"),
                        workspaceBoxShadow: workspaceStyle.boxShadow,
                        workspacePadding,
                      };
                    }
                    """
                )
                self.assertTrue(style["visualModelMode"])
                self.assertEqual(style["mainBackground"], style["panelBackground"])
                self.assertEqual(style["workspaceBackground"], style["panelBackground"])
                self.assertTrue(style["workspaceBorderTransparent"], style["borderColors"])
                self.assertEqual(style["borderRadii"], ["0px", "0px", "0px", "0px"])
                self.assertEqual(style["mainPadding"], ["0px", "0px", "0px", "0px"])
                self.assertEqual(style["workspacePadding"], ["0px", "0px", "0px", "0px"])
                self.assertEqual(style["modelWrapPadding"], ["8px", "8px", "8px", "8px"])
                self.assertEqual(style["workspaceBoxShadow"], "none")

            def assert_model_tools_use_compact_workspace() -> None:
                page.locator("#glmTool:not(.hidden)").click()
                page.locator("#modelToolWrap:not(.hidden) .glm-tool").wait_for(timeout=10_000)
                assert_model_workspace_style(".glm-tool")

                page.locator("#gbmTool:not(.hidden)").click()
                page.locator("#modelToolWrap:not(.hidden) .gbm-tool").wait_for(timeout=10_000)
                assert_model_workspace_style(".gbm-tool")

                page.locator("#specsTool:not(.hidden)").click()
                page.locator("#specificationsWrap:not(.hidden) .spec-tool").wait_for(timeout=10_000)
                page.locator("#specGrid .tabulator-row").first.wait_for(timeout=10_000)

            def assert_specs_workspace_style() -> None:
                style = page.evaluate(
                    """
                    () => {
                      const main = document.querySelector("main");
                      const visual = document.querySelector("#visualArea");
                      const workspace = document.querySelector(".workspace");
                      const specsWrap = document.querySelector("#specificationsWrap");
                      const mainStyle = getComputedStyle(main);
                      const workspaceStyle = getComputedStyle(workspace);
                      const specsWrapStyle = getComputedStyle(specsWrap);
                      const panel = getComputedStyle(document.documentElement).getPropertyValue("--panel").trim();
                      const probe = document.createElement("div");
                      probe.style.background = panel;
                      probe.style.position = "absolute";
                      probe.style.visibility = "hidden";
                      document.body.appendChild(probe);
                      const panelBackground = getComputedStyle(probe).backgroundColor;
                      probe.remove();
                      const borderColors = [
                        workspaceStyle.borderTopColor,
                        workspaceStyle.borderRightColor,
                        workspaceStyle.borderBottomColor,
                        workspaceStyle.borderLeftColor,
                      ];
                      return {
                        borderColors,
                        mainBackground: mainStyle.backgroundColor,
                        mainPadding: [
                          mainStyle.paddingTop,
                          mainStyle.paddingRight,
                          mainStyle.paddingBottom,
                          mainStyle.paddingLeft,
                        ],
                        panelBackground,
                        specsMode: visual.classList.contains("specs-mode"),
                        specsWrapPadding: [
                          specsWrapStyle.paddingTop,
                          specsWrapStyle.paddingRight,
                          specsWrapStyle.paddingBottom,
                          specsWrapStyle.paddingLeft,
                        ],
                        workspaceBackground: workspaceStyle.backgroundColor,
                        workspaceBorderTransparent: borderColors.every((color) => color === "transparent" || color === "rgba(0, 0, 0, 0)"),
                        workspaceBoxShadow: workspaceStyle.boxShadow,
                        workspacePadding: [
                          workspaceStyle.paddingTop,
                          workspaceStyle.paddingRight,
                          workspaceStyle.paddingBottom,
                          workspaceStyle.paddingLeft,
                        ],
                        workspaceRadii: [
                          workspaceStyle.borderTopLeftRadius,
                          workspaceStyle.borderTopRightRadius,
                          workspaceStyle.borderBottomRightRadius,
                          workspaceStyle.borderBottomLeftRadius,
                        ],
                      };
                    }
                    """
                )
                self.assertTrue(style["specsMode"])
                self.assertEqual(style["mainBackground"], style["panelBackground"])
                self.assertEqual(style["workspaceBackground"], style["panelBackground"])
                self.assertTrue(style["workspaceBorderTransparent"], style["borderColors"])
                self.assertEqual(style["workspaceRadii"], ["0px", "0px", "0px", "0px"])
                self.assertEqual(style["workspaceBoxShadow"], "none")
                self.assertEqual(style["mainPadding"], ["0px", "0px", "0px", "0px"])
                self.assertEqual(style["workspacePadding"], ["0px", "0px", "0px", "0px"])
                self.assertEqual(style["specsWrapPadding"], ["8px", "8px", "8px", "8px"])

            def assert_specs_full_width() -> None:
                layout = page.evaluate(
                    """
                    () => {
                        const visual = document.querySelector("#visualArea").getBoundingClientRect();
                        const workspace = document.querySelector(".workspace").getBoundingClientRect();
                        const specs = document.querySelector("#specificationsWrap").getBoundingClientRect();
                        return {
                            visualWidth: visual.width,
                            workspaceWidth: workspace.width,
                            specsWidth: specs.width,
                        };
                    }
                    """
                )
                self.assertGreaterEqual(layout["workspaceWidth"], layout["visualWidth"] * 0.8)
                self.assertGreaterEqual(layout["specsWidth"], layout["visualWidth"] * 0.8)

            def assert_specs_table_style() -> None:
                cell_locator = page.locator("#specGrid .tabulator-row .tabulator-cell[tabulator-field]").first
                cell_locator.wait_for(timeout=10_000)
                page.wait_for_function(
                    """
                    () => {
                      const cell = document.querySelector("#specGrid .tabulator-row .tabulator-cell[tabulator-field]");
                      return cell && getComputedStyle(cell).display === "inline-flex";
                    }
                    """,
                    timeout=10_000,
                )
                style = cell_locator.evaluate(
                    """
                    node => {
                      const computed = getComputedStyle(node);
                      return {
                        alignItems: computed.alignItems,
                        display: computed.display,
                        fontSize: computed.fontSize,
                      };
                    }
                    """
                )
                self.assertEqual(style, {"alignItems": "center", "display": "inline-flex", "fontSize": "11px"})
                self.assertEqual(
                    page.locator(".spec-tool").evaluate(
                        """
                        node => {
                          const style = getComputedStyle(node);
                          return {
                            borderTopWidth: style.borderTopWidth,
                            borderTopLeftRadius: style.borderTopLeftRadius,
                            boxShadow: style.boxShadow,
                            paddingLeft: style.paddingLeft,
                            paddingTop: style.paddingTop,
                          };
                        }
                        """
                    ),
                    {
                        "borderTopWidth": "0px",
                        "borderTopLeftRadius": "0px",
                        "boxShadow": "none",
                        "paddingLeft": "0px",
                        "paddingTop": "0px",
                    },
                )
                self.assertEqual(page.locator("#specGrid").evaluate("node => getComputedStyle(node).borderTopLeftRadius"), "6px")
                self.assertEqual(page.locator(".spec-kind-tabs .tab").first.evaluate("node => getComputedStyle(node).fontWeight"), "700")
                topbar_layout = page.evaluate(
                    """
                    () => {
                      const topbar = document.querySelector(".spec-topbar")?.getBoundingClientRect();
                      const tabs = document.querySelector(".spec-kind-tabs")?.getBoundingClientRect();
                      const path = document.querySelector("#specFilePath")?.getBoundingClientRect();
                      const notice = document.querySelector("#specNotice")?.getBoundingClientRect();
                      const save = document.querySelector("#specSaveBtn")?.getBoundingClientRect();
                      const pathStyle = getComputedStyle(document.querySelector("#specFilePath"));
                      const noticeStyle = getComputedStyle(document.querySelector("#specNotice"));
                      return {
                        pathBelowTabs: path.top > tabs.bottom,
                        pathFullWidth: path.width >= topbar.width * 0.95,
                        pathStartsAtLeft: Math.abs(path.left - topbar.left) <= 2,
                        noticeBelowPath: notice.top > path.bottom,
                        noticeRightAligned: Math.abs(notice.right - save.right) <= 2,
                        noticeUnderSave: notice.left <= save.right && notice.right >= save.left,
                        pathWeight: pathStyle.fontWeight,
                        saveRightAligned: Math.abs(save.right - topbar.right) <= 2,
                        noticeTextAlign: noticeStyle.textAlign,
                        noticeWeight: noticeStyle.fontWeight,
                        noticeOverflow: noticeStyle.textOverflow,
                        noticeWhiteSpace: noticeStyle.whiteSpace,
                      };
                    }
                    """
                )
                self.assertTrue(topbar_layout["pathBelowTabs"])
                self.assertTrue(topbar_layout["pathFullWidth"])
                self.assertTrue(topbar_layout["pathStartsAtLeft"])
                self.assertTrue(topbar_layout["noticeBelowPath"])
                self.assertTrue(topbar_layout["noticeRightAligned"])
                self.assertTrue(topbar_layout["noticeUnderSave"])
                self.assertEqual(topbar_layout["pathWeight"], "400")
                self.assertTrue(topbar_layout["saveRightAligned"])
                self.assertEqual(topbar_layout["noticeTextAlign"], "right")
                self.assertEqual(topbar_layout["noticeWeight"], "400")
                self.assertEqual(topbar_layout["noticeOverflow"], "ellipsis")
                self.assertEqual(topbar_layout["noticeWhiteSpace"], "nowrap")

            def assert_spec_row_numbers() -> None:
                page.wait_for_function(
                    """
                    () => {
                      const rows = document.querySelectorAll("#specGrid .tabulator-row");
                      return rows[0]?.querySelector(".tabulator-row-header")?.textContent.trim() === "2"
                        && rows[1]?.querySelector(".tabulator-row-header")?.textContent.trim() === "3";
                    }
                    """,
                    timeout=10_000,
                )
                row_header_style = page.locator("#specGrid .tabulator-row").first.locator(".tabulator-row-header").evaluate(
                    """
                    node => {
                      const style = getComputedStyle(node);
                      return {
                        justifyContent: style.justifyContent,
                        textAlign: style.textAlign,
                        width: Math.round(node.getBoundingClientRect().width),
                      };
                    }
                    """
                )
                self.assertEqual(row_header_style["justifyContent"], "center")
                self.assertEqual(row_header_style["textAlign"], "center")
                self.assertLessEqual(row_header_style["width"], 42)

            def assert_spec_headers_untruncated() -> None:
                clipped = page.evaluate(
                    """
                    () => Array.from(document.querySelectorAll("#specGrid .tabulator-header .tabulator-col[tabulator-field] .tabulator-col-title"))
                      .map((title) => {
                        const column = title.closest(".tabulator-col");
                        return {
                          text: title.textContent.trim(),
                          titleWidth: title.scrollWidth,
                          columnWidth: column ? column.getBoundingClientRect().width : 0,
                        };
                      })
                      .filter((entry) => entry.text && entry.titleWidth > entry.columnWidth + 1)
                    """
                )
                self.assertEqual(clipped, [])

            def spec_cell(field: str, row_index: int = 0):
                return page.locator("#specGrid .tabulator-row").nth(row_index).locator(f'.tabulator-cell[tabulator-field="{field}"]')

            def spec_cell_background(field: str, row_index: int = 0) -> str:
                return spec_cell(field, row_index).evaluate("node => getComputedStyle(node).backgroundColor")

            def assert_global_status_clear() -> None:
                page.wait_for_function(
                    """
                    () => {
                      const status = document.querySelector("#status");
                      return status && status.classList.contains("hidden") && !status.textContent.trim();
                    }
                    """,
                    timeout=10_000,
                )

            def assert_validation_row_highlight(row_index: int, row_number: str = "2") -> None:
                page.wait_for_function(
                    """
                    expected => {
                      const row = document.querySelectorAll("#specGrid .tabulator-row")[expected.rowIndex];
                      const rowHeader = row?.querySelector(".tabulator-row-header");
                      const firstCell = row?.querySelector(".tabulator-cell[tabulator-field]");
                      return row?.classList.contains("spec-validation-issue-row")
                        && rowHeader?.textContent.trim() === expected.rowNumber
                        && firstCell
                        && getComputedStyle(firstCell).backgroundColor === "rgb(255, 248, 215)";
                    }
                    """,
                    arg={"rowIndex": row_index, "rowNumber": row_number},
                    timeout=10_000,
                )

            def assert_no_validation_row_highlight(row_index: int) -> None:
                page.wait_for_function(
                    """
                    rowIndex => {
                      const row = document.querySelectorAll("#specGrid .tabulator-row")[rowIndex];
                      const firstCell = row?.querySelector(".tabulator-cell[tabulator-field]");
                      return row
                        && !row.classList.contains("spec-validation-issue-row")
                        && firstCell
                        && getComputedStyle(firstCell).backgroundColor !== "rgb(255, 248, 215)";
                    }
                    """,
                    arg=row_index,
                    timeout=10_000,
                )

            def assert_notice_right_aligned() -> None:
                notice_layout = page.evaluate(
                    """
                    () => {
                      const path = document.querySelector("#specFilePath")?.getBoundingClientRect();
                      const notice = document.querySelector("#specNotice")?.getBoundingClientRect();
                      const save = document.querySelector("#specSaveBtn")?.getBoundingClientRect();
                      const noticeStyle = getComputedStyle(document.querySelector("#specNotice"));
                      return {
                        noticeBelowPath: notice.top > path.bottom,
                        noticeRightAligned: Math.abs(notice.right - save.right) <= 2,
                        noticeUnderSave: notice.left <= save.right && notice.right >= save.left,
                        noticeTextAlign: noticeStyle.textAlign,
                      };
                    }
                    """
                )
                self.assertTrue(notice_layout["noticeBelowPath"])
                self.assertTrue(notice_layout["noticeRightAligned"])
                self.assertTrue(notice_layout["noticeUnderSave"])
                self.assertEqual(notice_layout["noticeTextAlign"], "right")

            def spec_header(field: str):
                return page.locator(f'#specGrid .tabulator-header .tabulator-col[tabulator-field="{field}"]').first

            def spec_header_title(field: str) -> str:
                return spec_header(field).locator(".tabulator-col-title").first.inner_text().strip()

            def wait_for_spec_header_title(field: str, title: str) -> None:
                page.wait_for_function(
                    """
                    expected => {
                      const header = document.querySelector(`#specGrid .tabulator-header .tabulator-col[tabulator-field="${expected.field}"] .tabulator-col-title`);
                      return header?.textContent.trim() === expected.title;
                    }
                    """,
                    arg={"field": field, "title": title},
                    timeout=10_000,
                )
                self.assertEqual(spec_header_title(field), title)

            def spec_table_scroll_top() -> float:
                return float(page.locator("#specGrid .tabulator-tableholder").evaluate("node => node.scrollTop"))

            def scroll_specs_table_down() -> float:
                state = page.locator("#specGrid .tabulator-tableholder").evaluate(
                    """
                    node => {
                      node.scrollTop = Math.max(0, node.scrollHeight - node.clientHeight - 80);
                      return {
                        scrollTop: node.scrollTop,
                        scrollHeight: node.scrollHeight,
                        clientHeight: node.clientHeight,
                      };
                    }
                    """
                )
                self.assertGreater(state["scrollHeight"], state["clientHeight"])
                page.wait_for_function(
                    "() => document.querySelector('#specGrid .tabulator-tableholder')?.scrollTop > 0",
                    timeout=10_000,
                )
                return spec_table_scroll_top()

            def assert_specs_table_scroll_stable(before: float) -> None:
                current = spec_table_scroll_top()
                self.assertLessEqual(abs(current - before), 2)

            def reset_specs_table_scroll() -> None:
                page.locator("#specGrid .tabulator-tableholder").evaluate("node => { node.scrollTop = 0; }")
                page.wait_for_function(
                    """
                    () => {
                      const holder = document.querySelector('#specGrid .tabulator-tableholder');
                      const first = document.querySelector('#specGrid .tabulator-row .tabulator-cell[tabulator-field="Feature"]');
                      return holder && holder.scrollTop === 0 && first?.textContent.trim() === "vehicle_age";
                    }
                    """,
                    timeout=10_000,
                )

            def click_visible_scenario_checkbox() -> None:
                point = page.evaluate(
                    """
                    () => {
                      const holder = document.querySelector("#specGrid .tabulator-tableholder");
                      if (!holder) return null;
                      const holderRect = holder.getBoundingClientRect();
                      const checkboxes = Array.from(holder.querySelectorAll('.tabulator-cell[tabulator-field="scenario1"] .spec-checkbox-cell'));
                      const candidates = checkboxes
                        .map((checkbox) => {
                          const rect = checkbox.getBoundingClientRect();
                          return {
                            x: rect.left + rect.width / 2,
                            y: rect.top + rect.height / 2,
                            distance: Math.abs((rect.top + rect.height / 2) - (holderRect.top + holderRect.height / 2)),
                            visible: rect.top >= holderRect.top + 16 && rect.bottom <= holderRect.bottom - 16,
                          };
                        })
                        .filter((item) => item.visible)
                        .sort((left, right) => left.distance - right.distance);
                      return candidates[0] || null;
                    }
                    """
                )
                self.assertIsNotNone(point)
                assert point is not None
                page.mouse.click(point["x"], point["y"])

            def delete_visible_missing_feature_row() -> None:
                point = page.evaluate(
                    """
                    () => {
                      const holder = document.querySelector("#specGrid .tabulator-tableholder");
                      if (!holder) return null;
                      const holderRect = holder.getBoundingClientRect();
                      const rows = Array.from(holder.querySelectorAll(".tabulator-row.spec-missing-feature-row"));
                      const candidates = rows
                        .map((row) => {
                          const cell = row.querySelector('.tabulator-cell[tabulator-field="Feature"]');
                          const rect = cell?.getBoundingClientRect();
                          if (!cell || !rect) return null;
                          return {
                            x: rect.left + Math.min(rect.width - 8, 24),
                            y: rect.top + rect.height / 2,
                            text: cell.textContent.trim(),
                            distance: Math.abs((rect.top + rect.height / 2) - (holderRect.top + holderRect.height / 2)),
                            visible: rect.top >= holderRect.top + 16 && rect.bottom <= holderRect.bottom - 16,
                          };
                        })
                        .filter((item) => item && item.visible)
                        .sort((left, right) => left.distance - right.distance);
                      return candidates[0] || null;
                    }
                    """
                )
                self.assertIsNotNone(point)
                assert point is not None
                page.mouse.click(point["x"], point["y"], button="right")
                page.locator("#specContextMenu:not([hidden])").wait_for(timeout=10_000)
                page.locator('#specContextMenu [data-spec-row-action="delete"]').click()

            def spec_header_fields() -> list[str]:
                return page.evaluate(
                    """
                    () => Array.from(document.querySelectorAll('#specGrid .tabulator-header .tabulator-col[tabulator-field]'))
                      .map((column) => column.getAttribute('tabulator-field') || '')
                    """
                )

            def column_menu_labels() -> list[str]:
                return page.locator("#specColumnContextMenu .spec-context-menu-item").evaluate_all(
                    "items => items.map((item) => item.textContent.trim())"
                )

            def click_column_menu_action(action: str, prompt_value: str | None = None) -> None:
                button = page.locator(f'#specColumnContextMenu [data-spec-column-action="{action}"]')
                if prompt_value is None:
                    button.click()
                    return
                dialog_types = []

                def accept_prompt(dialog) -> None:
                    dialog_types.append(dialog.type)
                    dialog.accept(prompt_value)

                page.once("dialog", accept_prompt)
                button.click()
                self.assertEqual(dialog_types, ["prompt"])

            def assert_header_order(before: str, after: str) -> None:
                fields = spec_header_fields()
                self.assertLess(fields.index(before), fields.index(after))

            def wait_for_header_absent(field: str) -> None:
                page.wait_for_function(
                    """
                    field => !document.querySelector(`#specGrid .tabulator-header .tabulator-col[tabulator-field="${field}"]`)
                    """,
                    arg=field,
                    timeout=10_000,
                )

            def specs_selection_state() -> dict:
                return page.evaluate(
                    """
                    () => {
                      const rows = Array.from(document.querySelectorAll("#specGrid .tabulator-row"));
                      let activeField = "";
                      let activeRowIndex = -1;
                      rows.forEach((row, rowIndex) => {
                        const active = row.querySelector(".tabulator-cell.spec-cell-active");
                        if (active) {
                          activeField = active.getAttribute("tabulator-field") || "";
                          activeRowIndex = rowIndex;
                        }
                      });
                      return {
                        activeField,
                        activeRowIndex,
                        selectedCount: document.querySelectorAll("#specGrid .spec-cell-selected").length,
                        nativeSelection: window.getSelection()?.toString() || "",
                      };
                    }
                    """
                )

            def assert_active_cell(field: str, row_index: int, selected_count: int | None = 1) -> None:
                page.wait_for_function(
                    """
                    expected => {
                      const rows = Array.from(document.querySelectorAll("#specGrid .tabulator-row"));
                      let activeField = "";
                      let activeRowIndex = -1;
                      rows.forEach((row, rowIndex) => {
                        const active = row.querySelector(".tabulator-cell.spec-cell-active");
                        if (active) {
                          activeField = active.getAttribute("tabulator-field") || "";
                          activeRowIndex = rowIndex;
                        }
                      });
                      const selectedCount = document.querySelectorAll("#specGrid .spec-cell-selected").length;
                      return activeField === expected.field
                        && activeRowIndex === expected.rowIndex
                        && (expected.selectedCount === null || selectedCount === expected.selectedCount);
                    }
                    """,
                    arg={"field": field, "rowIndex": row_index, "selectedCount": selected_count},
                    timeout=10_000,
                )
                state = specs_selection_state()
                self.assertEqual(state["activeField"], field)
                self.assertEqual(state["activeRowIndex"], row_index)
                if selected_count is not None:
                    self.assertEqual(state["selectedCount"], selected_count)
                self.assertEqual(state["nativeSelection"], "")

            def drag_specs_selection(start, end, expected_count: int) -> None:
                start_box = start.bounding_box()
                end_box = end.bounding_box()
                self.assertIsNotNone(start_box)
                self.assertIsNotNone(end_box)
                assert start_box is not None and end_box is not None
                page.mouse.move(start_box["x"] + 4, start_box["y"] + start_box["height"] / 2)
                page.mouse.down()
                page.mouse.move(end_box["x"] + end_box["width"] - 4, end_box["y"] + end_box["height"] / 2, steps=8)
                page.mouse.up()
                page.wait_for_function(
                    "expected => document.querySelectorAll('#specGrid .spec-cell-selected').length >= expected",
                    arg=expected_count,
                    timeout=10_000,
                )
                self.assertEqual(page.evaluate("() => window.getSelection()?.toString() || ''"), "")

            try:
                page.goto(base_url, wait_until="domcontentloaded")
                page.evaluate(
                    """
                    () => {
                        window.__lucidumClipboardText = "";
                        Object.defineProperty(navigator, "clipboard", {
                            configurable: true,
                            value: {
                                writeText: async (text) => {
                                    window.__lucidumClipboardText = text;
                                },
                                readText: async () => window.__lucidumClipboardText,
                            },
                        });
                    }
                    """
                )
                page.locator("#datasetMeta").get_by_text("sample.csv").wait_for(timeout=10_000)
                assert_model_tools_use_compact_workspace()
                if page.locator("#favouritesCollapseBtn").get_attribute("aria-expanded") == "true":
                    page.locator("#favouritesCollapseBtn").click()
                    page.wait_for_function(
                        '() => document.querySelector("#favouritesCollapseBtn")?.getAttribute("aria-expanded") === "false"',
                        timeout=10_000,
                    )
                self.assertTrue(page.locator(".sidebar-metric-section").is_visible())
                self.assertTrue(page.locator("#actualNumerator").is_visible())
                self.assertTrue(page.locator("#denominator").is_visible())
                self.assertTrue(page.locator("#visualArea").evaluate("node => node.classList.contains('specs-mode')"))
                self.assertFalse(page.locator("#chartSideControls").is_visible())
                self.assertFalse(page.locator("#chartControlsResizer").is_visible())
                assert_specs_workspace_style()
                assert_specs_full_width()
                assert_specs_table_style()
                assert_spec_row_numbers()
                assert_spec_headers_untruncated()
                wait_for_spec_header_title("scenario1", "scenario1 (3)")
                scrolled_top = scroll_specs_table_down()
                initial_dark_mode = bool(page.evaluate("() => document.body.classList.contains('dark')"))
                page.locator("#themeBtn").click()
                page.wait_for_function(
                    "expected => document.body.classList.contains('dark') === expected",
                    arg=not initial_dark_mode,
                    timeout=10_000,
                )
                assert_specs_table_scroll_stable(scrolled_top)
                page.locator("#themeBtn").click()
                page.wait_for_function(
                    "expected => document.body.classList.contains('dark') === expected",
                    arg=initial_dark_mode,
                    timeout=10_000,
                )
                assert_specs_table_scroll_stable(scrolled_top)
                click_visible_scenario_checkbox()
                wait_for_spec_header_title("scenario1", "scenario1 (4)")
                assert_specs_table_scroll_stable(scrolled_top)
                self.assertEqual(specs_selection_state()["activeField"], "scenario1")
                page.keyboard.press("Delete")
                wait_for_spec_header_title("scenario1", "scenario1 (3)")
                assert_specs_table_scroll_stable(scrolled_top)
                self.assertEqual(specs_selection_state()["selectedCount"], 1)
                page.evaluate("() => { window.__lucidumClipboardText = 'feature'; }")
                page.keyboard.press("Control+V")
                wait_for_spec_header_title("scenario1", "scenario1 (4)")
                assert_specs_table_scroll_stable(scrolled_top)
                page.keyboard.press("Delete")
                wait_for_spec_header_title("scenario1", "scenario1 (3)")
                assert_specs_table_scroll_stable(scrolled_top)
                delete_visible_missing_feature_row()
                assert_specs_table_scroll_stable(scrolled_top)
                page.wait_for_function(
                    """
                    () => Boolean(
                      document.querySelector('#specGrid .tabulator-row.spec-missing-feature-row .tabulator-cell[tabulator-field="Feature"]')
                    )
                    """,
                    timeout=10_000,
                )
                reset_specs_table_scroll()
                self.assertEqual(spec_cell_background("Feature", 2), "rgb(255, 248, 215)")
                self.assertNotEqual(spec_cell_background("Feature", 0), "rgb(255, 248, 215)")
                spec_cell("Feature", 2).dblclick()
                page.locator("#specGrid .tabulator-cell.tabulator-editing input").fill("price")
                page.keyboard.press("Enter")
                page.wait_for_function(
                    """
                    () => {
                      const row = document.querySelectorAll("#specGrid .tabulator-row")[2];
                      const cell = row?.querySelector('.tabulator-cell[tabulator-field="Feature"]');
                      return cell?.textContent.trim() === "price" && getComputedStyle(cell).backgroundColor !== "rgb(255, 248, 215)";
                    }
                    """,
                    timeout=10_000,
                )
                spec_cell("Feature", 2).click()
                page.keyboard.press("Delete")
                page.wait_for_function(
                    """
                    () => {
                      const row = document.querySelectorAll("#specGrid .tabulator-row")[2];
                      const cell = row?.querySelector('.tabulator-cell[tabulator-field="Feature"]');
                      return cell?.textContent.trim() === "" && getComputedStyle(cell).backgroundColor !== "rgb(255, 248, 215)";
                    }
                    """,
                    timeout=10_000,
                )
                spec_cell("Feature", 2).click()
                page.evaluate("() => { window.__lucidumClipboardText = 'FutureFeature'; }")
                page.keyboard.press("Control+V")
                page.wait_for_function(
                    """
                    () => {
                      const row = document.querySelectorAll("#specGrid .tabulator-row")[2];
                      const cell = row?.querySelector('.tabulator-cell[tabulator-field="Feature"]');
                      return cell?.textContent.trim() === "FutureFeature" && row?.classList.contains("spec-missing-feature-row");
                    }
                    """,
                    timeout=10_000,
                )
                page.locator("#specNotice", has_text="Valid feature spec").wait_for(timeout=10_000)
                assert_global_status_clear()
                page.locator("#specSaveBtn").click()
                page.locator("#specNotice", has_text="Feature spec saved").wait_for(timeout=10_000)
                assert_global_status_clear()
                self.assertEqual(page.locator("#specScenarioToolbar").count(), 0)
                spec_header("Feature").click(button="right")
                page.locator("#specColumnContextMenu:not([hidden])").wait_for(timeout=10_000)
                self.assertEqual(column_menu_labels(), ["Add scenario"])
                self.assertEqual(page.locator("#specColumnContextMenu").evaluate("node => getComputedStyle(node).padding"), "3px")
                self.assertEqual(page.locator("#specColumnContextMenu .spec-context-menu-item").first.evaluate("node => getComputedStyle(node).fontWeight"), "400")
                click_column_menu_action("add-end", "scenario_from_header")
                spec_header("scenario_from_header").wait_for(timeout=10_000)
                wait_for_spec_header_title("scenario_from_header", "scenario_from_header (0)")
                assert_spec_headers_untruncated()
                assert_header_order("scenario1", "scenario_from_header")
                spec_header("scenario_from_header").click(button="right")
                page.locator("#specColumnContextMenu:not([hidden])").wait_for(timeout=10_000)
                self.assertEqual(column_menu_labels(), ["Add scenario before", "Add scenario after", "Delete scenario", "Rename scenario"])
                click_column_menu_action("delete")
                wait_for_header_absent("scenario_from_header")
                spec_header("scenario1").click(button="right")
                page.locator("#specColumnContextMenu:not([hidden])").wait_for(timeout=10_000)
                self.assertEqual(column_menu_labels(), ["Add scenario before", "Add scenario after", "Delete scenario", "Rename scenario"])
                click_column_menu_action("add-before", "scenario_before")
                spec_header("scenario_before").wait_for(timeout=10_000)
                wait_for_spec_header_title("scenario_before", "scenario_before (0)")
                assert_header_order("scenario_before", "scenario1")
                spec_header("scenario_before").click(button="right")
                page.locator("#specColumnContextMenu:not([hidden])").wait_for(timeout=10_000)
                click_column_menu_action("delete")
                wait_for_header_absent("scenario_before")
                spec_header("scenario1").click(button="right")
                page.locator("#specColumnContextMenu:not([hidden])").wait_for(timeout=10_000)
                click_column_menu_action("add-after", "scenario_after")
                spec_header("scenario_after").wait_for(timeout=10_000)
                wait_for_spec_header_title("scenario_after", "scenario_after (0)")
                assert_header_order("scenario1", "scenario_after")
                spec_header("scenario_after").click(button="right")
                page.locator("#specColumnContextMenu:not([hidden])").wait_for(timeout=10_000)
                click_column_menu_action("rename", "scenario_renamed")
                spec_header("scenario_renamed").wait_for(timeout=10_000)
                wait_for_spec_header_title("scenario_renamed", "scenario_renamed (0)")
                wait_for_header_absent("scenario_after")
                spec_header("scenario_renamed").click(button="right")
                page.locator("#specColumnContextMenu:not([hidden])").wait_for(timeout=10_000)
                click_column_menu_action("delete")
                wait_for_header_absent("scenario_renamed")
                spec_header("scenario1").wait_for(timeout=10_000)
                self.assertEqual(spec_header_fields()[-1], "scenario1")
                scenario_cell = spec_cell("scenario1", 0)
                scenario_checkbox = scenario_cell.locator(".spec-checkbox-cell")
                self.assertTrue(scenario_checkbox.is_checked())
                self.assertIsNone(scenario_checkbox.get_attribute("disabled"))
                self.assertEqual(scenario_checkbox.evaluate("node => getComputedStyle(node).pointerEvents"), "auto")
                self.assertEqual(scenario_checkbox.evaluate("node => getComputedStyle(node).opacity"), "1")
                scenario_cell.dblclick(position={"x": 4, "y": 8})
                page.wait_for_function("() => !document.querySelector('#specGrid .tabulator-cell.tabulator-editing')")
                self.assertNotIn("feature", scenario_cell.inner_text())
                scenario_checkbox.click()
                page.wait_for_function(
                    "() => !document.querySelector('#specGrid .tabulator-row .tabulator-cell[tabulator-field=\"scenario1\"] .spec-checkbox-cell')?.checked",
                    timeout=10_000,
                )
                wait_for_spec_header_title("scenario1", "scenario1 (2)")
                self.assertTrue(page.locator("#specSaveBtn").evaluate("node => node.classList.contains('dirty')"))
                scenario_cell.locator(".spec-checkbox-cell").click()
                page.wait_for_function(
                    "() => Boolean(document.querySelector('#specGrid .tabulator-row .tabulator-cell[tabulator-field=\"scenario1\"] .spec-checkbox-cell')?.checked)",
                    timeout=10_000,
                )
                wait_for_spec_header_title("scenario1", "scenario1 (3)")
                page.evaluate("() => { window.__lucidumClipboardText = 'unset'; }")
                page.keyboard.press("Control+C")
                page.wait_for_function("() => window.__lucidumClipboardText === 'feature'", timeout=10_000)
                scenario_cell.locator(".spec-checkbox-cell").click()
                page.wait_for_function(
                    "() => !document.querySelector('#specGrid .tabulator-row .tabulator-cell[tabulator-field=\"scenario1\"] .spec-checkbox-cell')?.checked",
                    timeout=10_000,
                )
                wait_for_spec_header_title("scenario1", "scenario1 (2)")
                page.evaluate("() => { window.__lucidumClipboardText = 'unset'; }")
                page.keyboard.press("Control+C")
                page.wait_for_function("() => window.__lucidumClipboardText === ''", timeout=10_000)
                scenario_cell.locator(".spec-checkbox-cell").click()
                page.wait_for_function(
                    "() => Boolean(document.querySelector('#specGrid .tabulator-row .tabulator-cell[tabulator-field=\"scenario1\"] .spec-checkbox-cell')?.checked)",
                    timeout=10_000,
                )
                wait_for_spec_header_title("scenario1", "scenario1 (3)")
                drag_specs_selection(spec_cell("scenario1", 0), spec_cell("scenario1", 1), 2)
                page.keyboard.press("Delete")
                page.wait_for_function(
                    """
                    () => Array.from(document.querySelectorAll('#specGrid .tabulator-row .tabulator-cell[tabulator-field="scenario1"] .spec-checkbox-cell'))
                        .slice(0, 2)
                        .every((checkbox) => !checkbox.checked)
                    """,
                    timeout=10_000,
                )
                wait_for_spec_header_title("scenario1", "scenario1 (1)")
                page.evaluate("() => { window.__lucidumClipboardText = 'feature\\nfeature'; }")
                page.keyboard.press("Control+V")
                page.wait_for_function(
                    """
                    () => Array.from(document.querySelectorAll('#specGrid .tabulator-row .tabulator-cell[tabulator-field="scenario1"] .spec-checkbox-cell'))
                        .slice(0, 2)
                        .every((checkbox) => checkbox.checked)
                    """,
                    timeout=10_000,
                )
                wait_for_spec_header_title("scenario1", "scenario1 (3)")
                spec_cell("Feature", 0).click()
                assert_active_cell("Feature", 0)
                page.keyboard.press("ArrowRight")
                assert_active_cell("Grouping", 0)
                page.keyboard.press("ArrowDown")
                assert_active_cell("Grouping", 1)
                page.keyboard.press("ArrowLeft")
                assert_active_cell("Feature", 1)
                page.keyboard.press("ArrowLeft")
                assert_active_cell("Feature", 1)
                page.keyboard.press("ArrowUp")
                assert_active_cell("Feature", 0)
                page.keyboard.press("ArrowUp")
                assert_active_cell("Feature", 0)
                for _ in range(6):
                    page.keyboard.press("ArrowRight")
                assert_active_cell("scenario1", 0)
                page.keyboard.press("ArrowRight")
                assert_active_cell("scenario1", 0)
                page.keyboard.press("ArrowDown")
                assert_active_cell("scenario1", 1)
                page.keyboard.press("ArrowDown")
                assert_active_cell("scenario1", 2)
                page.keyboard.press("ArrowDown")
                assert_active_cell("scenario1", 3)
                spec_cell("Base", 0).click()
                assert_active_cell("Base", 0)
                page.keyboard.press("9")
                page.locator("#specGrid .tabulator-cell.tabulator-editing input").wait_for(timeout=10_000)
                self.assertEqual(page.locator("#specGrid .tabulator-cell.tabulator-editing input").input_value(), "9")
                page.keyboard.press("Enter")
                page.wait_for_function(
                    "() => !document.querySelector('#specGrid .tabulator-cell.tabulator-editing')",
                    timeout=10_000,
                )
                page.locator("#specGrid .tabulator-row").first.locator('.tabulator-cell[tabulator-field="Base"]', has_text="9").wait_for(timeout=10_000)
                assert_active_cell("Base", 0)
                page.keyboard.press("ArrowRight")
                assert_active_cell("min", 0)
                spec_cell("Feature", 0).click()
                assert_active_cell("Feature", 0)
                page.keyboard.press("Shift+ArrowRight")
                assert_active_cell("Feature", 0, 2)
                page.keyboard.press("Shift+ArrowDown")
                assert_active_cell("Feature", 0, 4)
                page.evaluate("() => { window.__lucidumClipboardText = 'unset'; }")
                page.keyboard.press("Control+C")
                page.wait_for_function("() => window.__lucidumClipboardText === 'vehicle_age\\tVEHICLE\\nPostcodeArea\\tPOSTCODE'", timeout=10_000)
                page.keyboard.press("Delete")
                page.wait_for_function(
                    """
                    () => {
                        const rows = document.querySelectorAll("#specGrid .tabulator-row");
                        return rows[0]?.querySelector('.tabulator-cell[tabulator-field="Feature"]')?.textContent.trim() === ""
                            && rows[1]?.querySelector('.tabulator-cell[tabulator-field="Grouping"]')?.textContent.trim() === "";
                    }
                    """,
                    timeout=10_000,
                )
                drag_specs_selection(spec_cell("Feature", 0), spec_cell("Grouping", 1), 4)
                assert_active_cell("Feature", 0, 4)
                page.keyboard.press("Delete")
                page.wait_for_function(
                    """
                    () => {
                        const rows = document.querySelectorAll("#specGrid .tabulator-row");
                        return rows[0]?.querySelector('.tabulator-cell[tabulator-field="Feature"]')?.textContent.trim() === ""
                            && rows[1]?.querySelector('.tabulator-cell[tabulator-field="Grouping"]')?.textContent.trim() === "";
                    }
                    """,
                    timeout=10_000,
                )
                spec_cell("scenario1", 2).click(button="right")
                page.locator("#specContextMenu:not([hidden])").wait_for(timeout=10_000)
                page.locator('#specContextMenu [data-spec-row-action="delete"]').click()
                wait_for_spec_header_title("scenario1", "scenario1 (2)")

                page.locator('[data-spec-kind="kpi"]').click()
                page.locator('[data-spec-kind="kpi"][aria-selected="true"]').wait_for(timeout=10_000)
                page.locator("#specGrid .tabulator-row").first.wait_for(timeout=10_000)
                assert_specs_full_width()
                assert_specs_table_style()
                assert_spec_row_numbers()
                assert_spec_headers_untruncated()
                spec_header("group").click(button="right")
                page.wait_for_timeout(100)
                self.assertTrue(page.locator("#specColumnContextMenu").evaluate("node => node.hidden"))
                page.locator("#specGrid").focus()
                page.keyboard.press("ArrowRight")
                assert_active_cell("name", 0)
                page.keyboard.press("Shift+ArrowDown")
                assert_active_cell("name", 0, 2)
                drag_specs_selection(spec_cell("group", 0), spec_cell("name", 0), 2)
                page.keyboard.press("Control+C")
                page.wait_for_function("() => window.__lucidumClipboardText === 'PRICE\\tPrice'", timeout=10_000)
                drag_specs_selection(spec_cell("group", 1), spec_cell("name", 1), 2)
                page.keyboard.press("Control+V")
                page.wait_for_function(
                    """
                    () => {
                        const row = document.querySelectorAll("#specGrid .tabulator-row")[1];
                        return row?.querySelector('.tabulator-cell[tabulator-field="group"]')?.textContent.trim() === "PRICE"
                            && row?.querySelector('.tabulator-cell[tabulator-field="name"]')?.textContent.trim() === "Price";
                    }
                    """,
                    timeout=10_000,
                )
                assert_active_cell("group", 1, 2)
                page.keyboard.press("ArrowDown")
                assert_active_cell("group", 2, 1)
                spec_cell("actual", 0).dblclick()
                page.locator("#specGrid .tabulator-cell.tabulator-editing input").fill("MissingActual")
                page.keyboard.press("Enter")
                page.locator("#specNotice.error", has_text="kpi_spec.csv row 2 actual column does not exist: MissingActual").wait_for(timeout=10_000)
                assert_validation_row_highlight(0, "2")
                self.assertTrue(page.locator("#specSaveBtn").evaluate("node => node.disabled"))
                self.assertEqual(page.locator("#specSaveBtn").get_attribute("title"), "Fix validation errors before saving")
                assert_notice_right_aligned()
                assert_global_status_clear()
                spec_cell("actual", 0).dblclick()
                page.locator("#specGrid .tabulator-cell.tabulator-editing input").fill("price")
                page.keyboard.press("Enter")
                page.locator("#specNotice", has_text="Valid KPI spec").wait_for(timeout=10_000)
                assert_no_validation_row_highlight(0)
                page.wait_for_function(
                    """
                    () => {
                      const button = document.querySelector("#specSaveBtn");
                      return button && !button.disabled && button.classList.contains("dirty") && button.title === "Save specification";
                    }
                    """,
                    timeout=10_000,
                )

                def reject_kpi_save(route: Any) -> None:
                    route.fulfill(
                        status=400,
                        content_type="application/json",
                        body=json.dumps({"detail": "simulated save rejection"}),
                    )

                page.route("**/api/specs/kpi/save", reject_kpi_save)
                page.locator("#specSaveBtn").click()
                page.locator("#specNotice.error", has_text="Save failed; file was not written: simulated save rejection").wait_for(timeout=10_000)
                page.wait_for_function(
                    """
                    () => {
                      const button = document.querySelector("#specSaveBtn");
                      return button && !button.disabled && button.classList.contains("dirty");
                    }
                    """,
                    timeout=10_000,
                )
                page.unroute("**/api/specs/kpi/save", reject_kpi_save)

                page.locator('[data-spec-kind="filter"]').click()
                page.locator('[data-spec-kind="filter"][aria-selected="true"]').wait_for(timeout=10_000)
                page.locator("#specGrid .tabulator-row").first.wait_for(timeout=10_000)
                assert_specs_full_width()
                assert_specs_table_style()
                assert_spec_row_numbers()
                assert_spec_headers_untruncated()
                page.locator("#specNotice.error", has_text="AutoMissingColumn").wait_for(timeout=10_000)
                assert_validation_row_highlight(2, "4")
                assert_global_status_clear()
                spec_cell("theme", 0).click(button="right")
                page.locator("#specContextMenu:not([hidden])").wait_for(timeout=10_000)
                page.locator('#specContextMenu [data-spec-row-action="delete"]').click()
                page.locator("#specNotice.error", has_text="AutoMissingColumn").wait_for(timeout=10_000)
                assert_validation_row_highlight(1, "3")
                assert_global_status_clear()
                spec_header("theme").click(button="right")
                page.wait_for_timeout(100)
                self.assertTrue(page.locator("#specColumnContextMenu").evaluate("node => node.hidden"))
                page.locator("#specGrid").focus()
                page.keyboard.press("ArrowRight")
                assert_active_cell("name", 0)
                page.keyboard.press("Shift+ArrowDown")
                assert_active_cell("name", 0, 2)
                drag_specs_selection(spec_cell("theme", 0), spec_cell("name", 1), 4)
                assert_active_cell("theme", 0, 4)
                spec_cell("name", 0).dblclick()
                page.locator("#specGrid .tabulator-cell.tabulator-editing input").fill("Edited filter")
                page.keyboard.press("ArrowLeft")
                page.locator("#specGrid .tabulator-cell.tabulator-editing input").wait_for(timeout=10_000)
                page.keyboard.press("Enter")
                page.locator("#specGrid .tabulator-row").first.locator('.tabulator-cell[tabulator-field="name"]', has_text="Edited filter").wait_for(timeout=10_000)
                spec_cell("expression", 0).dblclick()
                page.locator("#specGrid .tabulator-cell.tabulator-editing input").fill("MissingColumn = 1")
                page.keyboard.press("Enter")
                page.locator("#specNotice.error", has_text="MissingColumn").wait_for(timeout=10_000)
                assert_validation_row_highlight(0, "2")
                assert_global_status_clear()
                spec_cell("expression", 0).dblclick()
                page.locator("#specGrid .tabulator-cell.tabulator-editing input").fill("vehicle_age < 3")
                page.keyboard.press("Enter")
                assert_no_validation_row_highlight(0)
                spec_cell("theme", 0).click(button="right")
                page.locator("#specContextMenu:not([hidden])").wait_for(timeout=10_000)
                self.assertEqual(page.locator("#specContextMenu").evaluate("node => getComputedStyle(node).padding"), "3px")
                self.assertEqual(page.locator("#specContextMenu .spec-context-menu-item").first.evaluate("node => getComputedStyle(node).fontWeight"), "400")

                page.evaluate(
                    """
                    () => {
                      window.__lucidumRemoveAllRangesCalls = 0;
                      const proto = Selection.prototype;
                      if (!proto.__lucidumOriginalRemoveAllRanges) {
                        Object.defineProperty(proto, "__lucidumOriginalRemoveAllRanges", {
                          configurable: true,
                          value: proto.removeAllRanges,
                        });
                      }
                      proto.removeAllRanges = function(...args) {
                        window.__lucidumRemoveAllRangesCalls += 1;
                        return proto.__lucidumOriginalRemoveAllRanges.apply(this, args);
                      };
                    }
                    """
                )
                page.locator("#histogramTool:not(.hidden)").click()
                page.locator("#histogramWrap:not(.hidden)").wait_for(timeout=10_000)
                page.evaluate("() => { document.querySelector('#histogramBins').value = ''; }")
                page.locator("#histogramBins").click()
                page.keyboard.type("17")
                page.wait_for_function(
                    """
                    () => document.activeElement?.id === "histogramBins"
                      && document.querySelector("#histogramBins")?.value === "17"
                      && window.__lucidumRemoveAllRangesCalls === 0
                    """,
                    timeout=10_000,
                )

                self.assertEqual(page_errors, [])
            finally:
                browser.close()

    def exercise_dataset_viewer_sidebar_resize(self, base_url: str) -> None:
        assert sync_playwright is not None
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch()
            page = browser.new_page(viewport={"width": 1280, "height": 800})
            page_errors: list[str] = []
            page.on("pageerror", lambda error: page_errors.append(str(error)))
            try:
                page.goto(f"{base_url}?tool=dataset_viewer", wait_until="domcontentloaded")
                page.locator("#datasetViewerTool.active").wait_for(timeout=10_000)
                page.locator("#datasetViewerWrap:not(.hidden) #datasetViewerGrid .tabulator-row").first.wait_for(timeout=10_000)
                dataset_viewer_surface = page.evaluate(
                    """
                    () => {
                      const main = document.querySelector("main");
                      const visual = document.querySelector("#visualArea");
                      const workspace = document.querySelector(".workspace");
                      const panel = getComputedStyle(document.documentElement).getPropertyValue("--panel").trim();
                      const probe = document.createElement("div");
                      probe.style.background = panel;
                      probe.style.position = "absolute";
                      probe.style.visibility = "hidden";
                      document.body.appendChild(probe);
                      const panelBackground = getComputedStyle(probe).backgroundColor;
                      probe.remove();
                      return {
                        datasetViewerMode: visual.classList.contains("dataset-viewer-mode"),
                        mainBackground: getComputedStyle(main).backgroundColor,
                        panelBackground,
                        workspaceBackground: getComputedStyle(workspace).backgroundColor,
                      };
                    }
                    """
                )
                self.assertTrue(dataset_viewer_surface["datasetViewerMode"])
                self.assertEqual(dataset_viewer_surface["mainBackground"], dataset_viewer_surface["panelBackground"])
                self.assertEqual(dataset_viewer_surface["workspaceBackground"], dataset_viewer_surface["panelBackground"])
                page.wait_for_function(
                    """
                    () => typeof window.Tabulator?.prototype?.redraw === "function"
                    """,
                    timeout=10_000,
                )
                page.evaluate(
                    """
                    async () => {
                      await new Promise((resolve) => requestAnimationFrame(() => requestAnimationFrame(resolve)));
                      const proto = window.Tabulator.prototype;
                      window.__datasetViewerRedraws = [];
                      if (!proto.__lucidumOriginalRedraw) {
                        Object.defineProperty(proto, "__lucidumOriginalRedraw", {
                          configurable: true,
                          value: proto.redraw,
                        });
                        proto.redraw = function(force) {
                          (window.__datasetViewerRedraws ||= []).push({
                            force: force === true,
                            value: force === undefined ? "undefined" : String(force),
                          });
                          return proto.__lucidumOriginalRedraw.apply(this, arguments);
                        };
                      }
                    }
                    """
                )
                resizer_box = page.locator("#sidebarResizer").bounding_box()
                self.assertIsNotNone(resizer_box)
                assert resizer_box is not None
                center_x = resizer_box["x"] + resizer_box["width"] / 2
                center_y = resizer_box["y"] + resizer_box["height"] / 2

                page.mouse.move(center_x, center_y)
                page.mouse.down()
                page.mouse.move(center_x + 48, center_y, steps=3)
                page.mouse.move(center_x + 96, center_y, steps=3)
                page.mouse.move(center_x + 144, center_y, steps=3)
                during_drag_redraws = page.evaluate(
                    """
                    async () => {
                      await new Promise((resolve) => requestAnimationFrame(() => requestAnimationFrame(resolve)));
                      return window.__datasetViewerRedraws || [];
                    }
                    """
                )
                self.assertEqual(during_drag_redraws, [])
                page.mouse.up()
                page.wait_for_function(
                    """
                    () => (window.__datasetViewerRedraws || []).some((entry) => entry.force === true)
                    """,
                    timeout=5_000,
                )
                self.assertEqual(page_errors, [])
            finally:
                browser.close()

    def exercise_dataset_viewer_large_transpose(self, base_url: str) -> None:
        assert sync_playwright is not None
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch()
            page = browser.new_page(viewport={"width": 1280, "height": 800})
            page_errors: list[str] = []
            dataset_viewer_requests = 0
            page.on("pageerror", lambda error: page_errors.append(str(error)))

            def count_dataset_viewer_request(request) -> None:
                nonlocal dataset_viewer_requests
                if request.url.endswith("/api/dataset-viewer/table"):
                    dataset_viewer_requests += 1

            page.on("request", count_dataset_viewer_request)
            try:
                page.goto(f"{base_url}?tool=dataset_viewer", wait_until="domcontentloaded")
                page.locator("#datasetViewerWrap:not(.hidden) #datasetViewerGrid .tabulator-row").first.wait_for(timeout=10_000)
                page.wait_for_function(
                    """
                    () => document.querySelector("#datasetViewerCount")?.textContent.trim() === "First 100 shown"
                    """,
                    timeout=10_000,
                )
                page.locator("#datasetViewerTranspose").check(timeout=5_000)
                page.wait_for_function(
                    """
                    () => {
                      const grid = document.querySelector('#datasetViewerGrid.dataset-viewer-grid-transposed');
                      const first = grid?.querySelector('.tabulator-row .tabulator-cell[tabulator-field="__field"]');
                      const renderedCells = grid?.querySelectorAll('.tabulator-cell').length || 0;
                      return Boolean(
                        grid
                        && !grid.querySelector('.dataset-viewer-transposed-table')
                        && first?.textContent.trim() === 'PostcodeArea'
                        && renderedCells > 0
                        && renderedCells < 8 * 101
                      );
                    }
                    """,
                    timeout=10_000,
                )
                requests_before_scroll = dataset_viewer_requests
                page.locator("#datasetViewerGrid .tabulator-tableholder").evaluate(
                    """
                    (node) => {
                      node.scrollLeft = 4000;
                      node.scrollTop = 400;
                      node.dispatchEvent(new Event("scroll", { bubbles: true }));
                    }
                    """,
                )
                page.wait_for_timeout(100)
                self.assertEqual(dataset_viewer_requests, requests_before_scroll)
                page.locator("#stopAppBtn").click(timeout=5_000)
                page.locator(".stop-confirm-cancel").click(timeout=5_000)
                self.assertEqual(page_errors, [])
            finally:
                browser.close()

    def exercise_dataset_viewer_disabled(self, base_url: str) -> None:
        assert sync_playwright is not None
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch()
            page = browser.new_page(viewport={"width": 1280, "height": 800})
            page_errors: list[str] = []
            dataset_viewer_module_requests = 0
            dataset_viewer_css_requests = 0
            dataset_viewer_table_requests = 0

            page.on("pageerror", lambda error: page_errors.append(str(error)))

            def count_request(request: object) -> None:
                nonlocal dataset_viewer_module_requests, dataset_viewer_css_requests, dataset_viewer_table_requests
                url = request.url
                if url.endswith("/static/app/dataset-viewer-tool.js"):
                    dataset_viewer_module_requests += 1
                elif url.endswith("/static/styles/dataset-viewer.css"):
                    dataset_viewer_css_requests += 1
                elif url.endswith("/api/dataset-viewer/table"):
                    dataset_viewer_table_requests += 1

            page.on("request", count_request)

            try:
                page.goto(base_url, wait_until="domcontentloaded")
                page.locator("#datasetMeta").get_by_text("sample.csv").wait_for(timeout=10_000)
                page.wait_for_function(
                    """
                    () => document.querySelector("#lineBarTool")?.classList.contains("active")
                    """,
                    timeout=10_000,
                )
                page.locator("#profileTool").wait_for(state="hidden", timeout=10_000)
                page.wait_for_function(
                    """
                    () => document.querySelector("#datasetViewerTool")?.classList.contains("hidden")
                    """,
                    timeout=10_000,
                )
                self.assertFalse(page.locator("#datasetViewerTool").is_visible())
                self.assertEqual(dataset_viewer_module_requests, 0)
                self.assertEqual(dataset_viewer_css_requests, 0)
                self.assertEqual(dataset_viewer_table_requests, 0)

                page.locator("#chart:not(.hidden)").wait_for(timeout=10_000)
                page.wait_for_timeout(250)

                self.assertEqual(dataset_viewer_module_requests, 0)
                self.assertEqual(dataset_viewer_css_requests, 0)
                self.assertEqual(dataset_viewer_table_requests, 0)
                self.assertEqual(page_errors, [])
            finally:
                browser.close()

    def exercise_browser(self, base_url: str) -> None:
        assert sync_playwright is not None
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch()
            page = browser.new_page(viewport={"width": 1280, "height": 800})
            page_errors: list[str] = []
            dataset_viewer_requests = 0
            profile_requests = 0
            profile_detail_requests = 0
            chart_requests = 0
            histogram_requests = 0
            map_requests = 0

            page.on("pageerror", lambda error: page_errors.append(str(error)))

            def count_request(request: object) -> None:
                nonlocal dataset_viewer_requests, profile_requests, profile_detail_requests, chart_requests, histogram_requests, map_requests
                url = request.url
                if url.endswith("/api/dataset-viewer/table"):
                    dataset_viewer_requests += 1
                elif url.endswith("/api/column-profile/summary"):
                    profile_requests += 1
                elif url.endswith("/api/column-profile/detail"):
                    profile_detail_requests += 1
                elif url.endswith("/api/chart"):
                    chart_requests += 1
                elif url.endswith("/api/histogram/chart"):
                    histogram_requests += 1
                elif url.endswith("/api/uk-map/summary"):
                    map_requests += 1

            page.on("request", count_request)

            def map_view() -> dict[str, float]:
                return page.evaluate(
                    """
                    () => {
                        const map = document.querySelector("#ukMap")?._lucidumMap;
                        if (!map) return null;
                        const center = map.getCenter();
                        return { lat: center.lat, lng: center.lng, zoom: map.getZoom() };
                    }
                    """
                )

            def wait_for_map_view(expected: dict[str, float]) -> None:
                try:
                    page.wait_for_function(
                        """
                        expected => {
                            const map = document.querySelector("#ukMap")?._lucidumMap;
                            if (!map) return false;
                            const center = map.getCenter();
                            return Math.abs(center.lat - expected.lat) < 0.01
                                && Math.abs(center.lng - expected.lng) < 0.01
                                && Math.abs(map.getZoom() - expected.zoom) < 0.01;
                        }
                        """,
                        arg=expected,
                        timeout=10_000,
                    )
                except PlaywrightTimeoutError as exc:
                    debug = page.evaluate(
                        """
                        expected => {
                            const map = document.querySelector("#ukMap")?._lucidumMap;
                            const container = document.querySelector("#ukMap");
                            const rect = container?.getBoundingClientRect();
                            if (!map) return { expected, actual: null };
                            const center = map.getCenter();
                            const actual = { lat: center.lat, lng: center.lng, zoom: map.getZoom() };
                            return {
                                expected,
                                actual,
                                delta: {
                                    lat: actual.lat - expected.lat,
                                    lng: actual.lng - expected.lng,
                                    zoom: actual.zoom - expected.zoom,
                                },
                                hidden: container?.classList.contains("hidden"),
                                activeTools: [...document.querySelectorAll(".tool-option.active")].map((node) => node.id),
                                sidebarExpanded: document.querySelector("#sidebarToggleBtn")?.getAttribute("aria-expanded"),
                                mapMeta: document.querySelector("#mapGroupMeta")?.textContent || "",
                                actionTiming: document.querySelector("#actionTimingMonitor")?.textContent || "",
                                mapBox: rect ? { width: rect.width, height: rect.height, x: rect.x, y: rect.y } : null,
                            };
                        }
                        """,
                        arg=expected,
                    )
                    raise AssertionError(f"UK map view did not settle: {json.dumps(debug, sort_keys=True)}") from exc

            def assert_filter_label_badge(selector: str, applied_class: str, applied: bool) -> None:
                page.wait_for_function(
                    """
                    ({ selector, appliedClass, applied }) => {
                      const label = document.querySelector(selector);
                      if (!label || label.classList.contains(appliedClass) !== applied) return false;
                      if (!applied) return label.textContent.includes("no filter");
                      const probe = document.createElement("span");
                      probe.style.backgroundColor = getComputedStyle(document.body).getPropertyValue("--filter-applied-bg").trim();
                      document.body.append(probe);
                      const expectedBackground = getComputedStyle(probe).backgroundColor;
                      probe.remove();
                      return getComputedStyle(label).backgroundColor === expectedBackground;
                    }
                    """,
                    arg={"selector": selector, "appliedClass": applied_class, "applied": applied},
                    timeout=10_000,
                )

            def assert_filter_badge_clear(clear_selector: str, text_selector: str, visible: bool) -> None:
                page.wait_for_function(
                    """
                    ({ clearSelector, visible }) => {
                      const clear = document.querySelector(clearSelector);
                      if (!clear) return false;
                      return visible
                        ? !clear.hidden && clear.offsetParent !== null
                        : clear.hidden || clear.offsetParent === null;
                    }
                    """,
                    arg={"clearSelector": clear_selector, "visible": visible},
                    timeout=10_000,
                )
                if not visible:
                    return
                layout = page.evaluate(
                    """
                    ({ clearSelector, textSelector }) => {
                      const clear = document.querySelector(clearSelector).getBoundingClientRect();
                      const text = document.querySelector(textSelector).getBoundingClientRect();
                      return {
                        clearLeft: clear.left,
                        clearRight: clear.right,
                        textLeft: text.left,
                        textRight: text.right,
                      };
                    }
                    """,
                    arg={"clearSelector": clear_selector, "textSelector": text_selector},
                )
                self.assertLessEqual(layout["clearLeft"], layout["textLeft"])
                self.assertLessEqual(layout["clearRight"], layout["textLeft"] + 1)
                self.assertGreater(layout["textRight"], layout["textLeft"])

            def unit_point_alpha_pixels() -> int:
                return page.evaluate(
                    """
                    () => {
                        const canvas = document.querySelector("#ukMap .leaflet-unit-point-layer");
                        if (!canvas || canvas.width <= 0 || canvas.height <= 0) return 0;
                        const context = canvas.getContext("2d");
                        if (!context) return 0;
                        const imageData = context.getImageData(0, 0, canvas.width, canvas.height).data;
                        let pixels = 0;
                        for (let index = 3; index < imageData.length; index += 4) {
                            if (imageData[index] > 0) pixels += 1;
                        }
                        return pixels;
                    }
                    """
                )

            def assert_dataset_viewer_hidden() -> None:
                page.wait_for_function(
                    """
                    () => {
                      const wrap = document.querySelector("#datasetViewerWrap");
                      const visualArea = document.querySelector("#visualArea");
                      return Boolean(
                        wrap
                        && wrap.classList.contains("hidden")
                        && getComputedStyle(wrap).display === "none"
                        && visualArea
                        && !visualArea.classList.contains("dataset-viewer-mode")
                      );
                    }
                    """,
                    timeout=10_000,
                )

            def assert_profile_table_unoccluded() -> None:
                page.wait_for_function(
                    """
                    () => {
                      const profile = document.querySelector("#profileWrap");
                      const table = profile?.querySelector(".profile-table");
                      if (!profile || !table || profile.classList.contains("hidden") || getComputedStyle(profile).display === "none") return false;
                      const rect = table.getBoundingClientRect();
                      if (rect.width <= 0 || rect.height <= 0) return false;
                      const top = document.elementFromPoint(rect.left + rect.width / 2, rect.top + rect.height / 2);
                      const datasetViewer = document.querySelector("#datasetViewerWrap");
                      return Boolean(top && profile.contains(top) && !datasetViewer?.contains(top));
                    }
                    """,
                    timeout=10_000,
                )

            def resize_tabulator_column(column_selector: str, delta: int = 36) -> dict[str, float]:
                probe = page.evaluate(
                    """
                    ({ selector }) => {
                      const column = document.querySelector(selector);
                      if (!column) return null;
                      let handle = column.nextElementSibling;
                      if (!handle?.classList?.contains("tabulator-col-resize-handle")) {
                        handle = [...column.parentElement?.querySelectorAll(".tabulator-col-resize-handle") || []]
                          .find((candidate) => candidate.getBoundingClientRect().left >= column.getBoundingClientRect().right - 6) || null;
                      }
                      if (!handle) return null;
                      const columnRect = column.getBoundingClientRect();
                      const handleRect = handle.getBoundingClientRect();
                      return {
                        before: columnRect.width,
                        x: handleRect.left + handleRect.width / 2,
                        y: handleRect.top + handleRect.height / 2,
                      };
                    }
                    """,
                    arg={"selector": column_selector},
                )
                self.assertIsNotNone(probe)
                assert probe is not None
                page.mouse.move(probe["x"], probe["y"])
                page.mouse.down()
                page.mouse.move(probe["x"] + delta, probe["y"], steps=6)
                page.mouse.up()
                page.wait_for_function(
                    """
                    ({ selector, minimum }) => {
                      const column = document.querySelector(selector);
                      return Boolean(column && column.getBoundingClientRect().width >= minimum);
                    }
                    """,
                    arg={"selector": column_selector, "minimum": probe["before"] + 20},
                    timeout=10_000,
                )
                after = page.evaluate(
                    """
                    selector => document.querySelector(selector)?.getBoundingClientRect().width || 0
                    """,
                    arg=column_selector,
                )
                return {"before": float(probe["before"]), "after": float(after)}

            def tabulator_column_width(column_selector: str) -> float:
                width = page.evaluate(
                    """
                    selector => document.querySelector(selector)?.getBoundingClientRect().width || 0
                    """,
                    arg=column_selector,
                )
                return float(width)

            try:
                page.goto(base_url, wait_until="domcontentloaded")
                page.locator("#datasetMeta").get_by_text("sample.csv").wait_for(timeout=10_000)
                page.wait_for_function(
                    """
                    () => document.querySelector(".dataset-meta-column-count")?.textContent.trim() === "8 columns"
                    """
                )
                self.assertEqual(page.locator("header").evaluate("node => getComputedStyle(node).height"), "52px")
                self.assertEqual(page.locator(".dataset-meta-title").text_content().strip(), "Lucidum Smoke Dataset")
                self.assertFalse(page.locator("#datasetMeta").evaluate("node => node.classList.contains('dataset-meta-title-only')"))
                self.assertTrue(page.locator(".dataset-meta-details").is_visible())
                self.assertEqual(
                    page.locator(".dataset-meta-title").evaluate("node => getComputedStyle(node).color"),
                    page.evaluate(
                        """
                        () => {
                          const probe = document.createElement("span");
                          probe.style.color = "var(--text)";
                          document.body.append(probe);
                          const color = getComputedStyle(probe).color;
                          probe.remove();
                          return color;
                        }
                        """
                    ),
                )
                self.assertGreaterEqual(
                    int(page.locator(".dataset-meta-title").evaluate("node => getComputedStyle(node).fontWeight")),
                    700,
                )
                self.assertGreater(
                    int(page.locator(".dataset-meta-title").evaluate("node => getComputedStyle(node).fontWeight")),
                    int(page.locator("#datasetMeta").evaluate("node => getComputedStyle(node).fontWeight")),
                )
                self.assertIn("4 rows", page.locator("#datasetMeta").text_content())
                self.assertEqual(page.locator(".dataset-meta-column-count").text_content().strip(), "8 columns")
                self.assertEqual(
                    page.locator(".dataset-meta-column-count").evaluate("node => getComputedStyle(node).textDecorationLine"),
                    "none",
                )
                page.set_viewport_size({"width": 420, "height": 800})
                page.wait_for_function(
                    """
                    () => document.querySelector("#datasetMeta")?.classList.contains("dataset-meta-title-only")
                    """,
                    timeout=10_000,
                )
                self.assertEqual(page.locator("#datasetMeta").inner_text().strip(), "Lucidum Smoke Dataset")
                self.assertFalse(page.locator(".dataset-meta-details").is_visible())
                page.set_viewport_size({"width": 1280, "height": 800})
                page.wait_for_function(
                    """
                    () => !document.querySelector("#datasetMeta")?.classList.contains("dataset-meta-title-only")
                      && document.querySelector(".dataset-meta-details")
                      && getComputedStyle(document.querySelector(".dataset-meta-details")).display !== "none"
                    """,
                    timeout=10_000,
                )
                if page.locator("#sidebarToggleBtn").get_attribute("aria-expanded") == "false":
                    page.locator("#sidebarToggleBtn").click()
                    page.wait_for_function(
                        """
                        () => document.querySelector("#sidebarToggleBtn")?.getAttribute("aria-expanded") === "true"
                        """,
                        timeout=10_000,
                    )
                self.assertEqual(page.locator(".dataset-meta-column-link").count(), 0)
                self.assertEqual(page.locator(".dataset-meta-uk-map-link").count(), 0)
                self.assertNotIn("Area·Sector·Unit", page.locator("#datasetMeta").text_content())
                self.assertEqual(page.locator(".dataset-meta-uk-map-icon").count(), 0)
                self.assertTrue(page.locator("#ukMapTool img").is_visible())
                self.assertTrue(page.locator("#ukMapTool img").evaluate("node => node.complete && node.naturalWidth > 0"))
                page.locator("#lineBarTool.active").wait_for(timeout=10_000)
                page.locator("#chart:not(.hidden)").wait_for(timeout=10_000)
                self.assertGreaterEqual(chart_requests, 1)
                page.locator("#datasetViewerTool").click()
                page.locator("#datasetViewerTool.active").wait_for(timeout=10_000)
                page.locator("#datasetViewerWrap:not(.hidden) #datasetViewerGrid .tabulator-row").first.wait_for(timeout=10_000)
                page.wait_for_function('() => Boolean(document.querySelector("#datasetViewerStylesheet"))', timeout=10_000)
                self.assertGreaterEqual(dataset_viewer_requests, 1)
                if page.locator("#favouritesCollapseBtn").get_attribute("aria-expanded") == "true":
                    page.locator("#favouritesCollapseBtn").click()
                    page.wait_for_function(
                        '() => document.querySelector("#favouritesCollapseBtn")?.getAttribute("aria-expanded") === "false"',
                        timeout=10_000,
                    )
                self.assertTrue(page.locator(".sidebar-metric-section").is_visible())
                self.assertTrue(page.locator("#actualNumerator").is_visible())
                self.assertTrue(page.locator("#denominator").is_visible())
                self.assertTrue(page.locator("#datasetViewerFilter").is_visible())
                page.wait_for_function(
                    """
                    () => {
                      const toolbar = document.querySelector(".dataset-viewer-toolbar");
                      const meta = document.querySelector("#datasetViewerMeta");
                      const summary = document.querySelector("#datasetViewerSummaryMeta");
                      const count = document.querySelector("#datasetViewerCount");
                      const separator = document.querySelector("#datasetViewerCountSeparator");
                      const group = document.querySelector("#datasetViewerGroupMeta");
                      const filter = document.querySelector("#datasetViewerFilter");
                      const countRect = count?.getBoundingClientRect();
                      const groupRect = group?.getBoundingClientRect();
                      return Boolean(
                        toolbar
                        && !toolbar.hidden
                        && meta
                        && summary
                        && count
                        && separator
                        && toolbar.contains(meta)
                        && meta.contains(summary)
                        && summary.contains(count)
                        && summary.contains(separator)
                        && summary.contains(group)
                        && meta.contains(filter)
                        && (count.compareDocumentPosition(group) & Node.DOCUMENT_POSITION_FOLLOWING)
                        && countRect
                        && groupRect
                        && Math.abs(countRect.top - groupRect.top) < 3
                        && countRect.left < groupRect.left
                        && !document.querySelector("#datasetViewerGrid .dataset-viewer-state")
                      );
                    }
                    """,
                    timeout=10_000,
                )
                dataset_viewer_meta_position = page.evaluate(
                    """
                    () => {
                      const rect = document.querySelector("#datasetViewerMeta")?.getBoundingClientRect();
                      return rect ? { top: rect.top, rightOffset: window.innerWidth - rect.right } : null;
                    }
                    """
                )
                self.assertIsNotNone(dataset_viewer_meta_position)
                page.wait_for_function(
                    """
                    () => document.querySelector("#filterRowMetaText")?.textContent.trim() === "4 rows"
                    """,
                    timeout=10_000,
                )
                page.wait_for_function(
                    """
                    () => document.querySelector("#datasetViewerGroupMeta")?.textContent.trim() === "8 columns · 4 rows"
                    """,
                    timeout=10_000,
                )
                assert_filter_label_badge("#datasetViewerFilter", "dataset-viewer-filter--applied", False)
                assert_filter_badge_clear("#datasetViewerFilterClearBtn", "#datasetViewerFilterText", False)
                self.assertFalse(page.locator("#filterRowMeta").evaluate('node => node.classList.contains("filter-row-meta--applied")'))
                self.assertEqual(page.locator("#collapsedSidebarVersion").inner_text().strip(), f"v{__version__}")
                self.assertFalse(page.locator("#collapsedSidebarVersion").is_visible())
                dataset_requests_before_filter = dataset_viewer_requests
                page.evaluate(
                    """
                    () => {
                        document.querySelector("#filterInput").value = "vehicle_age >= 3";
                        document.querySelector("#filterApplyBtn").click();
                    }
                    """
                )
                page.wait_for_function(
                    """
                    () => document.querySelector("#datasetViewerCount")?.textContent.includes("2 shown")
                    """,
                    timeout=10_000,
                )
                page.wait_for_function(
                    """
                    () => document.querySelector("#filterRowMetaText")?.textContent.trim() === "2 / 4 rows"
                    """,
                    timeout=10_000,
                )
                page.wait_for_function(
                    """
                    () => document.querySelector("#datasetViewerGroupMeta")?.textContent.trim() === "8 columns · 2 / 4 rows"
                      && document.querySelector("#datasetViewerFilter")?.textContent.trim() === "vehicle_age >= 3"
                    """,
                    timeout=10_000,
                )
                assert_filter_label_badge("#datasetViewerFilter", "dataset-viewer-filter--applied", True)
                assert_filter_label_badge("#filterRowMeta", "filter-row-meta--applied", True)
                assert_filter_badge_clear("#datasetViewerFilterClearBtn", "#datasetViewerFilterText", True)
                assert_filter_badge_clear("#filterRowClearBtn", "#filterRowMetaText", True)
                self.assertFalse(page.locator("#collapsedSidebarVersion").is_visible())
                page.locator("#sidebarToggleBtn").click()
                page.wait_for_function(
                    """
                    (expectedText) => {
                      if (document.querySelector("#sidebarToggleBtn")?.getAttribute("aria-expanded") !== "false") return false;
                      const marker = document.querySelector("#collapsedSidebarVersion");
                      if (!marker || marker.hidden || marker.textContent.trim() !== expectedText || marker.offsetParent === null) return false;
                      const probe = document.createElement("span");
                      probe.style.color = getComputedStyle(document.body).getPropertyValue("--muted").trim();
                      document.body.append(probe);
                      const muted = getComputedStyle(probe).color;
                      probe.remove();
                      return getComputedStyle(marker).color === muted;
                    }
                    """,
                    arg=f"v{__version__}",
                    timeout=10_000,
                )
                page.locator("#sidebarToggleBtn").click()
                page.wait_for_function(
                    """
                    () => document.querySelector("#sidebarToggleBtn")?.getAttribute("aria-expanded") === "true"
                      && document.querySelector("#collapsedSidebarVersion")?.offsetParent === null
                    """,
                    timeout=10_000,
                )
                self.assertGreater(dataset_viewer_requests, dataset_requests_before_filter)
                dataset_requests_before_second_filter = dataset_viewer_requests
                page.evaluate(
                    """
                    () => {
                        document.querySelector("#filterInput").value = "vehicle_age >= 2";
                        document.querySelector("#filterApplyBtn").click();
                    }
                    """
                )
                page.wait_for_function(
                    """
                    () => document.querySelector("#datasetViewerCount")?.textContent.includes("3 shown")
                    """,
                    timeout=10_000,
                )
                page.wait_for_function(
                    """
                    () => document.querySelector("#filterRowMetaText")?.textContent.trim() === "3 / 4 rows"
                    """,
                    timeout=10_000,
                )
                page.wait_for_function(
                    """
                    () => document.querySelector("#datasetViewerGroupMeta")?.textContent.trim() === "8 columns · 3 / 4 rows"
                      && document.querySelector("#datasetViewerFilter")?.textContent.trim() === "vehicle_age >= 2"
                    """,
                    timeout=10_000,
                )
                self.assertGreater(dataset_viewer_requests, dataset_requests_before_second_filter)
                page.locator("#lineBarTool").click()
                page.locator("#chart:not(.hidden)").wait_for(timeout=10_000)
                assert_dataset_viewer_hidden()
                page.wait_for_function(
                    """
                    () => document.querySelector("#filterRowMetaText")?.textContent.trim() === "3 / 4 rows"
                    """,
                    timeout=10_000,
                )
                assert_filter_label_badge("#lineBarFilter", "line-bar-filter--applied", True)
                assert_filter_badge_clear("#lineBarFilterClearBtn", "#lineBarFilterText", True)
                legend_spacing = page.evaluate(
                    """
                    () => {
                      const chartNode = document.querySelector("#chart");
                      const filter = document.querySelector("#lineBarFilter");
                      const chart = window.echarts?.getInstanceByDom(chartNode);
                      const option = chart?.getOption?.() || {};
                      const legend = Array.isArray(option.legend) ? option.legend[0] : option.legend;
                      const grid = Array.isArray(option.grid) ? option.grid[0] : option.grid;
                      const chartRect = chartNode?.getBoundingClientRect();
                      const filterRect = filter?.getBoundingClientRect();
                      return {
                        filterVisible: Boolean(filter && getComputedStyle(filter).display !== "none" && filterRect.width > 0),
                        filterBottomOffset: filterRect && chartRect ? filterRect.bottom - chartRect.top : 0,
                        legendTop: Number(legend?.top),
                        gridTop: Number(grid?.top),
                      };
                    }
                    """
                )
                self.assertTrue(legend_spacing["filterVisible"])
                self.assertGreater(legend_spacing["legendTop"], legend_spacing["filterBottomOffset"])
                self.assertGreaterEqual(legend_spacing["gridTop"], legend_spacing["legendTop"] + 40)
                page.locator("#lineBarFilterClearBtn").click()
                page.wait_for_function(
                    """
                    () => document.querySelector("#filterInput")?.value === ""
                      && document.querySelector("#lineBarFilterText")?.textContent.trim() === "no filter"
                      && document.querySelector("#lineBarFilterClearBtn")?.hidden
                    """,
                    timeout=10_000,
                )
                page.evaluate(
                    """
                    () => {
                        document.querySelector("#filterInput").value = "vehicle_age >= 2";
                        document.querySelector("#filterApplyBtn").click();
                    }
                    """
                )
                page.wait_for_function(
                    """
                    () => document.querySelector("#filterRowMetaText")?.textContent.trim() === "3 / 4 rows"
                      && document.querySelector("#lineBarFilterText")?.textContent.trim() === "vehicle_age >= 2"
                    """,
                    timeout=10_000,
                )
                page.locator("#ukMapTool").click()
                page.wait_for_function(
                    """
                    () => (document.querySelector("#mapGroupMeta")?.textContent || "").includes("matched")
                      || (document.querySelector("#mapGroupMeta")?.textContent || "").includes("plotted")
                    """,
                    timeout=10_000,
                )
                assert_dataset_viewer_hidden()
                page.wait_for_function(
                    """
                    () => document.querySelector("#filterRowMetaText")?.textContent.trim() === "3 / 4 rows"
                    """,
                    timeout=10_000,
                )
                map_panel_layout = page.evaluate(
                    """
                    () => {
                      const columnCount = (selector) => {
                        const columns = getComputedStyle(document.querySelector(selector)).gridTemplateColumns;
                        return columns.split(" ").filter(Boolean).length;
                      };
                      const visibleControls = [...document.querySelectorAll(".map-slider-control")]
                        .filter((control) => control.offsetParent !== null);
                      const endpointLabels = visibleControls.flatMap((control) => [
                        control.querySelector(".slider-scale b:first-child"),
                        control.querySelector(".slider-scale b:last-child"),
                      ]).filter(Boolean);
                      return {
                        width: document.querySelector("#mapFloatingControl").getBoundingClientRect().width,
                        groupMeta: document.querySelector("#mapGroupMeta")?.textContent.trim() || "",
                        rowMeta: document.querySelector("#mapRowMeta")?.textContent.trim() || "",
                        baseColumns: columnCount("#mapBaseLayerTiles"),
                        levelColumns: columnCount("#mapLevelTiles"),
                        paletteColumns: columnCount(".map-palette-buttons"),
                        sliderEndpointsVisible: endpointLabels.length > 0 && endpointLabels.every((label) => {
                          const rect = label.getBoundingClientRect();
                          return getComputedStyle(label).display !== "none" && rect.width > 0 && rect.height > 0;
                        }),
                      };
                    }
                    """
                )
                self.assertAlmostEqual(map_panel_layout["width"], 244, delta=2)
                self.assertNotIn("rows", map_panel_layout["groupMeta"])
                self.assertEqual(map_panel_layout["rowMeta"], "3 / 4 rows")
                self.assertEqual(map_panel_layout["baseColumns"], 3)
                self.assertEqual(map_panel_layout["levelColumns"], 3)
                self.assertEqual(map_panel_layout["paletteColumns"], 3)
                self.assertTrue(map_panel_layout["sliderEndpointsVisible"], map_panel_layout)
                assert_filter_label_badge("#mapControlFilter", "map-filter--applied", True)
                assert_filter_badge_clear("#mapControlFilterClearBtn", "#mapControlFilterText", True)
                page.wait_for_function(
                    """
                    () => {
                      const filter = document.querySelector("#mapControlFilter");
                      const header = document.querySelector(".map-floating-header");
                      if (!filter || !header) return false;
                      const filterWidth = filter.getBoundingClientRect().width;
                      const headerWidth = header.getBoundingClientRect().width;
                      return filterWidth > 0 && filterWidth < headerWidth * 0.6;
                    }
                    """,
                    timeout=10_000,
                )
                page.locator("#datasetViewerTool").click()
                page.locator("#datasetViewerWrap:not(.hidden) #datasetViewerGrid .tabulator-row").first.wait_for(timeout=10_000)
                page.wait_for_function(
                    """
                    () => document.querySelector("#datasetViewerGroupMeta")?.textContent.trim() === "8 columns · 3 / 4 rows"
                    """,
                    timeout=10_000,
                )
                dataset_requests_before_clear = dataset_viewer_requests
                page.locator("#datasetViewerFilterClearBtn").click()
                page.wait_for_function(
                    """
                    () => document.querySelector("#datasetViewerCount")?.textContent.includes("4 shown")
                    """,
                    timeout=10_000,
                )
                page.wait_for_function(
                    """
                    () => document.querySelector("#filterRowMetaText")?.textContent.trim() === "4 rows"
                    """,
                    timeout=10_000,
                )
                page.wait_for_function(
                    """
                    () => document.querySelector("#datasetViewerGroupMeta")?.textContent.trim() === "8 columns · 4 rows"
                    """,
                    timeout=10_000,
                )
                self.assertFalse(page.locator("#filterRowMeta").evaluate('node => node.classList.contains("filter-row-meta--applied")'))
                assert_filter_label_badge("#datasetViewerFilter", "dataset-viewer-filter--applied", False)
                assert_filter_label_badge("#lineBarFilter", "line-bar-filter--applied", False)
                assert_filter_label_badge("#histogramFilter", "histogram-filter--applied", False)
                assert_filter_label_badge("#mapControlFilter", "map-filter--applied", False)
                assert_filter_badge_clear("#datasetViewerFilterClearBtn", "#datasetViewerFilterText", False)
                assert_filter_badge_clear("#lineBarFilterClearBtn", "#lineBarFilterText", False)
                assert_filter_badge_clear("#histogramFilterClearBtn", "#histogramFilterText", False)
                assert_filter_badge_clear("#mapControlFilterClearBtn", "#mapControlFilterText", False)
                self.assertFalse(page.locator("#collapsedSidebarVersion").is_visible())
                self.assertGreater(dataset_viewer_requests, dataset_requests_before_clear)
                normal_resize = resize_tabulator_column('#datasetViewerGrid .tabulator-col[tabulator-field="c0"]')
                self.assertAlmostEqual(normal_resize["before"], 180, delta=4)
                self.assertGreater(normal_resize["after"], normal_resize["before"] + 20)
                page.evaluate(
                    """
                    () => {
                        window.__lucidumCopiedText = null;
                        Object.defineProperty(navigator, "clipboard", {
                            configurable: true,
                            value: {
                                writeText: async (text) => {
                                    window.__lucidumCopiedText = text;
                                },
                            },
                        });
                    }
                    """
                )
                page.locator("#datasetViewerGrid .tabulator-row").nth(0).click()
                page.locator('#datasetViewerGrid .tabulator-row .tabulator-cell[tabulator-field="c5"]').first.click(button="right")
                page.locator("#datasetViewerCellContextMenu:not([hidden])").get_by_text("Copy cell to clipboard").wait_for(timeout=10_000)
                page.locator("#datasetViewerCellContextMenu:not([hidden])").get_by_text("Copy selected row to clipboard").wait_for(timeout=10_000)
                page.locator("#datasetViewerCellContextMenu:not([hidden])").get_by_text("Copy cell to clipboard").click()
                page.wait_for_function("() => window.__lucidumCopiedText === 'AB10 1AA'", timeout=10_000)
                page.locator('#datasetViewerGrid .tabulator-row .tabulator-cell[tabulator-field="c5"]').first.click(button="right")
                page.locator("#datasetViewerCellContextMenu:not([hidden])").get_by_text("Copy selected row to clipboard").wait_for(timeout=10_000)
                page.locator("#datasetViewerCellContextMenu:not([hidden])").get_by_text("Copy selected row to clipboard").click()
                page.wait_for_function(
                    """
                    () => (window.__lucidumCopiedText || '') === [
                      'PostcodeArea,PostcodeSector,vehicle_age,price,value,PostcodeUnit,lat,long',
                      'AB,AB10 1,1,100,10,AB10 1AA,57.1,-2.1',
                    ].join('\\n')
                    """,
                    timeout=10_000,
                )
                page.locator("#datasetViewerGrid .tabulator-row").nth(1).click()
                page.locator("#datasetViewerGrid").click(button="right")
                page.locator("#datasetViewerCellContextMenu:not([hidden])").get_by_text("Copy selected rows to clipboard").wait_for(timeout=10_000)
                page.locator("#datasetViewerCellContextMenu:not([hidden])").get_by_text("Copy selected rows to clipboard").click()
                page.wait_for_function(
                    """
                    () => (window.__lucidumCopiedText || '') === [
                      'PostcodeArea,PostcodeSector,vehicle_age,price,value,PostcodeUnit,lat,long',
                      'AB,AB10 1,1,100,10,AB10 1AA,57.1,-2.1',
                      'AB,AB10 1,2,200,20,AB10 1AB,57.2,-2.2',
                    ].join('\\n')
                    """,
                    timeout=10_000,
                )
                page.locator("#datasetViewerGrid").click(button="right")
                page.locator("#datasetViewerCellContextMenu:not([hidden])").get_by_text("Clear selection").wait_for(timeout=10_000)
                self.assertEqual(page.locator("#datasetViewerCellContextMenu:not([hidden]) [role='separator']").count(), 1)
                page.locator("#datasetViewerCellContextMenu:not([hidden])").get_by_text("Clear selection").click()
                page.wait_for_function(
                    """
                    () => document.querySelectorAll('#datasetViewerGrid .tabulator-row.tabulator-selected').length === 0
                      && document.querySelectorAll('#datasetViewerGrid .dataset-viewer-column-selected').length === 0
                    """,
                    timeout=10_000,
                )
                page.locator('#datasetViewerGrid .tabulator-col[tabulator-field="c0"] .dataset-viewer-header-label').click()
                page.wait_for_function(
                    """
                    () => document.querySelectorAll('#datasetViewerGrid .tabulator-row.tabulator-selected').length === 0
                      && document.querySelectorAll('#datasetViewerGrid .dataset-viewer-column-selected').length > 0
                    """,
                    timeout=10_000,
                )
                page.locator('#datasetViewerGrid .tabulator-row .tabulator-cell[tabulator-field="c5"]').first.click(button="right")
                page.locator("#datasetViewerCellContextMenu:not([hidden])").get_by_text("Copy cell to clipboard").wait_for(timeout=10_000)
                page.locator("#datasetViewerCellContextMenu:not([hidden])").get_by_text("Copy selected column to clipboard").wait_for(timeout=10_000)
                page.locator("#datasetViewerCellContextMenu:not([hidden])").get_by_text("Copy cell to clipboard").click()
                page.wait_for_function("() => window.__lucidumCopiedText === 'AB10 1AA'", timeout=10_000)
                page.locator('#datasetViewerGrid .tabulator-row .tabulator-cell[tabulator-field="c5"]').first.click(button="right")
                page.locator("#datasetViewerCellContextMenu:not([hidden])").get_by_text("Copy selected column to clipboard").wait_for(timeout=10_000)
                page.locator("#datasetViewerCellContextMenu:not([hidden])").get_by_text("Copy selected column to clipboard").click()
                page.wait_for_function(
                    """
                    () => (window.__lucidumCopiedText || '') === [
                      'PostcodeArea',
                      'AB',
                      'AB',
                      'AL',
                      'AL',
                    ].join('\\n')
                    """,
                    timeout=10_000,
                )
                page.locator('#datasetViewerGrid .tabulator-col[tabulator-field="c2"] .dataset-viewer-header-label').click()
                page.locator("#datasetViewerGrid").click(button="right")
                page.locator("#datasetViewerCellContextMenu:not([hidden])").get_by_text("Copy selected columns to clipboard").wait_for(timeout=10_000)
                page.locator("#datasetViewerCellContextMenu:not([hidden])").get_by_text("Copy selected columns to clipboard").click()
                page.wait_for_function(
                    """
                    () => (window.__lucidumCopiedText || '') === [
                      'PostcodeArea,vehicle_age',
                      'AB,1',
                      'AB,2',
                      'AL,3',
                      'AL,4',
                    ].join('\\n')
                    """,
                    timeout=10_000,
                )
                page.locator("#datasetViewerGrid").click(button="right")
                page.locator("#datasetViewerCellContextMenu:not([hidden])").get_by_text("Clear selection").wait_for(timeout=10_000)
                self.assertEqual(page.locator("#datasetViewerCellContextMenu:not([hidden]) [role='separator']").count(), 1)
                page.locator("#datasetViewerCellContextMenu:not([hidden])").get_by_text("Clear selection").click()
                page.wait_for_function(
                    """
                    () => document.querySelectorAll('#datasetViewerGrid .tabulator-row.tabulator-selected').length === 0
                      && document.querySelectorAll('#datasetViewerGrid .dataset-viewer-column-selected').length === 0
                    """,
                    timeout=10_000,
                )
                page.locator("#datasetViewerGrid .tabulator-row").nth(0).click()
                page.wait_for_function(
                    """
                    () => document.querySelectorAll('#datasetViewerGrid .tabulator-row.tabulator-selected').length === 1
                      && document.querySelectorAll('#datasetViewerGrid .dataset-viewer-column-selected').length === 0
                    """,
                    timeout=10_000,
                )
                page.locator('#datasetViewerGrid .tabulator-col[tabulator-field="c0"] .dataset-viewer-header-label').click()
                page.locator('#datasetViewerGrid .tabulator-col[tabulator-field="c2"] .dataset-viewer-header-label').click()
                page.wait_for_function(
                    """
                    () => document.querySelector('#datasetViewerGrid .tabulator-col[tabulator-field="c0"]')?.getAttribute('aria-sort') === 'none'
                      && document.querySelector('#datasetViewerGrid .tabulator-row .tabulator-cell[tabulator-field="c0"]')?.textContent.trim() === 'AB'
                      && document.querySelectorAll('#datasetViewerGrid .dataset-viewer-column-selected').length > 0
                    """,
                    timeout=10_000,
                )
                page.wait_for_function(
                    """
                    () => {
                      const header = document.querySelector('#datasetViewerGrid .tabulator-col[tabulator-field="c0"]');
                      const title = header?.querySelector('.tabulator-col-title');
                      const sorter = header?.querySelector('.tabulator-col-sorter');
                      if (!title || !sorter) return false;
                      const titleRect = title.getBoundingClientRect();
                      const sorterRect = sorter.getBoundingClientRect();
                      const titleCenter = titleRect.top + titleRect.height / 2;
                      const sorterCenter = sorterRect.top + sorterRect.height / 2;
                      return sorterRect.left >= titleRect.right - 1
                        && Math.abs(sorterCenter - titleCenter) <= 2;
                    }
                    """,
                    timeout=10_000,
                )
                page.locator('#datasetViewerGrid .tabulator-col[tabulator-field="c0"] .tabulator-col-sorter').click()
                page.wait_for_function(
                    """
                    () => document.querySelector('#datasetViewerGrid .tabulator-col[tabulator-field="c0"]')?.getAttribute('aria-sort') === 'ascending'
                      && document.querySelector('#datasetViewerGrid .tabulator-row .tabulator-cell[tabulator-field="c0"]')?.textContent.trim() === 'AB'
                    """,
                    timeout=10_000,
                )
                page.locator('#datasetViewerGrid .tabulator-col[tabulator-field="c0"] .tabulator-col-sorter').click()
                page.wait_for_function(
                    """
                    () => document.querySelector('#datasetViewerGrid .tabulator-col[tabulator-field="c0"]')?.getAttribute('aria-sort') === 'descending'
                      && document.querySelector('#datasetViewerGrid .tabulator-row .tabulator-cell[tabulator-field="c0"]')?.textContent.trim() === 'AL'
                    """,
                    timeout=10_000,
                )
                page.locator('#datasetViewerGrid .tabulator-col[tabulator-field="c0"] .tabulator-col-sorter').click()
                page.wait_for_function(
                    """
                    () => document.querySelector('#datasetViewerGrid .tabulator-col[tabulator-field="c0"]')?.getAttribute('aria-sort') === 'none'
                      && document.querySelector('#datasetViewerGrid .tabulator-row .tabulator-cell[tabulator-field="c0"]')?.textContent.trim() === 'AB'
                    """,
                    timeout=10_000,
                )
                page.locator('#datasetViewerGrid .tabulator-col[tabulator-field="c0"] .dataset-viewer-header-label').click()
                page.locator('#datasetViewerGrid .tabulator-col[tabulator-field="c2"] .dataset-viewer-header-label').click()
                page.wait_for_function(
                    """
                    () => document.querySelectorAll('#datasetViewerGrid .tabulator-row.tabulator-selected').length === 0
                      && document.querySelectorAll('#datasetViewerGrid .dataset-viewer-column-selected').length === 0
                    """,
                    timeout=10_000,
                )
                page.locator('#datasetViewerGrid .tabulator-row .tabulator-cell[tabulator-field="c5"]').first.click(button="right")
                page.locator("#datasetViewerCellContextMenu:not([hidden])").get_by_text("Copy cell to clipboard").wait_for(timeout=10_000)
                page.locator("#datasetViewerCellContextMenu:not([hidden])").get_by_text("Copy displayed table to clipboard").wait_for(timeout=10_000)
                page.wait_for_function(
                    """
                    () => {
                      const menu = document.querySelector("#datasetViewerCellContextMenu:not([hidden])");
                      const entries = [...menu?.children || []].map((node) => (
                        node.getAttribute("role") === "separator" ? "separator" : node.textContent.trim()
                      ));
                      return entries.slice(0, 4).join("|") === "Pin column|separator|Copy cell to clipboard|Copy displayed table to clipboard";
                    }
                    """,
                    timeout=10_000,
                )
                self.assertEqual(page.locator("#datasetViewerCellContextMenu:not([hidden]) [role='separator']").count(), 1)
                page.locator("#datasetViewerCellContextMenu:not([hidden])").get_by_text("Copy displayed table to clipboard").click()
                page.wait_for_function(
                    """
                    () => (window.__lucidumCopiedText || '') === [
                      'PostcodeArea,PostcodeSector,vehicle_age,price,value,PostcodeUnit,lat,long',
                      'AB,AB10 1,1,100,10,AB10 1AA,57.1,-2.1',
                      'AB,AB10 1,2,200,20,AB10 1AB,57.2,-2.2',
                      'AL,AL1 1,3,300,30,AL1 1AA,51.8,-0.3',
                      'AL,AL1 2,4,400,40,AL1 2AA,51.7,-0.2',
                    ].join('\\n')
                    """,
                    timeout=10_000,
                )
                page.locator('#datasetViewerGrid .tabulator-row .tabulator-cell[tabulator-field="c5"]').first.click(button="right")
                page.locator("#datasetViewerCellContextMenu:not([hidden])").get_by_text("Copy cell to clipboard").wait_for(timeout=10_000)
                page.locator("#datasetViewerCellContextMenu:not([hidden])").get_by_text("Copy cell to clipboard").click()
                page.wait_for_function("() => window.__lucidumCopiedText === 'AB10 1AA'", timeout=10_000)
                pin_requests_before = dataset_viewer_requests
                page.locator('#datasetViewerGrid .tabulator-col[tabulator-field="c2"]').click(button="right")
                page.locator("#datasetViewerCellContextMenu:not([hidden])").get_by_role("menuitem", name="Pin column").click()
                page.wait_for_function(
                    """
                    () => {
                      const headers = [...document.querySelectorAll('#datasetViewerGrid .tabulator-col')]
                        .filter((cell) => cell.offsetParent !== null)
                        .map((cell) => cell.getAttribute('tabulator-field'))
                        .filter(Boolean);
                      const grid = document.querySelector('#datasetViewerGrid');
                      const hasFrozenContent = Boolean(grid?.querySelector('.tabulator-col.tabulator-frozen, .tabulator-cell.tabulator-frozen, .tabulator-frozen-rows-holder .tabulator-row'));
                      return headers[0] === 'c2'
                        && document.querySelector('#datasetViewerGrid .tabulator-col[tabulator-field="c2"] .dataset-viewer-pin-indicator')
                        && document.querySelector('#datasetViewerCount')?.textContent.trim() === '4 shown · vehicle_age pinned'
                        && !hasFrozenContent;
                    }
                    """,
                    timeout=10_000,
                )
                page.locator('#datasetViewerGrid .tabulator-row .tabulator-cell[tabulator-field="c5"]').first.click(button="right")
                page.locator("#datasetViewerCellContextMenu:not([hidden])").get_by_role("menuitem", name="Pin column").click()
                page.wait_for_function(
                    """
                    () => {
                      const grid = document.querySelector('#datasetViewerGrid:not(.dataset-viewer-grid-transposed)');
                      if (!grid) return false;
                      const headers = [...grid.querySelectorAll('.tabulator-col')]
                        .filter((cell) => cell.offsetParent !== null)
                        .map((cell) => cell.getAttribute('tabulator-field'))
                        .filter(Boolean);
                      return headers.slice(0, 4).join(',') === 'c5,c2,c0,c1'
                        && grid.querySelector('.tabulator-col[tabulator-field="c5"] .dataset-viewer-pin-indicator')
                        && grid.querySelector('.tabulator-col[tabulator-field="c2"] .dataset-viewer-pin-indicator')
                        && document.querySelector('#datasetViewerCount')?.textContent.trim() === '4 shown · PostcodeUnit, vehicle_age pinned'
                        && !grid.querySelector('.tabulator-col.tabulator-frozen, .tabulator-cell.tabulator-frozen, .tabulator-frozen-rows-holder .tabulator-row');
                    }
                    """,
                    timeout=10_000,
                )
                page.locator("#datasetViewerGrid .tabulator-tableholder").evaluate(
                    """
                    (node) => {
                      node.scrollLeft = 4000;
                      node.dispatchEvent(new Event("scroll", { bubbles: true }));
                    }
                    """
                )
                page.wait_for_function(
                    """
                    () => {
                      const grid = document.querySelector('#datasetViewerGrid:not(.dataset-viewer-grid-transposed)');
                      const holder = grid?.querySelector('.tabulator-tableholder');
                      if (!grid || !holder) return false;
                      const canScroll = holder.scrollWidth > holder.clientWidth + 1;
                      if (canScroll && holder.scrollLeft <= 0) return false;
                      const holderRect = holder.getBoundingClientRect();
                      const visibleHeaders = [...grid.querySelectorAll('.tabulator-col[tabulator-field]')]
                        .filter((cell) => {
                          const rect = cell.getBoundingClientRect();
                          return rect.right > holderRect.left && rect.left < holderRect.right;
                        })
                        .map((cell) => cell.getAttribute('tabulator-field'))
                        .filter(Boolean);
                      return visibleHeaders.length > 0
                        && (!canScroll || holder.scrollLeft < 180 || visibleHeaders[0] !== 'c5')
                        && !grid.querySelector('.tabulator-col.tabulator-frozen, .tabulator-cell.tabulator-frozen, .tabulator-frozen-rows-holder .tabulator-row');
                    }
                    """,
                    timeout=10_000,
                )
                page.locator("#datasetViewerGrid .tabulator-tableholder").evaluate(
                    """
                    (node) => {
                      node.scrollLeft = 0;
                      node.dispatchEvent(new Event("scroll", { bubbles: true }));
                    }
                    """
                )
                page.locator("#datasetViewerSearch").fill("price")
                page.wait_for_function(
                    """
                    () => {
                      const headers = [...document.querySelectorAll('#datasetViewerGrid .tabulator-col')]
                        .filter((cell) => cell.offsetParent !== null)
                        .map((cell) => cell.getAttribute('tabulator-field'))
                        .filter(Boolean);
                      const row = document.querySelector('#datasetViewerGrid .tabulator-row');
                      return headers.join(',') === 'c5,c2,c3'
                        && row?.querySelectorAll('.tabulator-cell').length === 3
                        && row.querySelector('.tabulator-cell[tabulator-field="c5"]')?.textContent.trim() === 'AB10 1AA'
                        && row.querySelector('.tabulator-cell[tabulator-field="c3"]')?.textContent.trim() === '100';
                    }
                    """,
                    timeout=10_000,
                )
                page.locator("#datasetViewerTranspose").check()
                page.wait_for_function(
                    """
                    () => {
                      const grid = document.querySelector('#datasetViewerGrid.dataset-viewer-grid-transposed');
                      if (!grid) return false;
                      const labelText = (row) => {
                        const cell = row.querySelector('.tabulator-cell[tabulator-field="__field"]');
                        return (cell?.querySelector('.dataset-viewer-pinned-field-text') || cell)?.textContent.trim();
                      };
                      const visibleNames = [...grid.querySelectorAll('.tabulator-tableholder .tabulator-row')]
                        .map(labelText)
                        .filter(Boolean);
                      return visibleNames.slice(0, 3).join(',') === 'PostcodeUnit,vehicle_age,price'
                        && grid.querySelectorAll('.tabulator-tableholder .tabulator-row .dataset-viewer-pin-indicator').length >= 2
                        && !grid.querySelector('.tabulator-col.tabulator-frozen, .tabulator-cell.tabulator-frozen, .tabulator-frozen-rows-holder .tabulator-row');
                    }
                    """,
                    timeout=10_000,
                )
                page.locator("#datasetViewerGrid .tabulator-tableholder").evaluate(
                    """
                    (node) => {
                      node.scrollLeft = 600;
                      node.dispatchEvent(new Event("scroll", { bubbles: true }));
                    }
                    """
                )
                page.wait_for_function(
                    """
                    () => {
                      const grid = document.querySelector('#datasetViewerGrid.dataset-viewer-grid-transposed');
                      const holder = grid?.querySelector('.tabulator-tableholder');
                      if (!grid || !holder) return false;
                      const canScroll = holder.scrollWidth > holder.clientWidth + 1;
                      if (canScroll && holder.scrollLeft <= 0) return false;
                      const holderRect = holder.getBoundingClientRect();
                      const visibleHeaders = [...grid.querySelectorAll('.tabulator-col[tabulator-field]')]
                        .filter((cell) => {
                          const rect = cell.getBoundingClientRect();
                          return rect.right > holderRect.left && rect.left < holderRect.right;
                        })
                        .map((cell) => cell.getAttribute('tabulator-field'))
                        .filter(Boolean);
                      return visibleHeaders.length > 0
                        && (!canScroll || holder.scrollLeft < 300 || visibleHeaders[0] !== '__field')
                        && !grid.querySelector('.tabulator-col.tabulator-frozen, .tabulator-cell.tabulator-frozen, .tabulator-frozen-rows-holder .tabulator-row');
                    }
                    """,
                    timeout=10_000,
                )
                page.locator("#datasetViewerGrid .tabulator-tableholder").evaluate(
                    """
                    (node) => {
                      node.scrollLeft = 0;
                      node.dispatchEvent(new Event("scroll", { bubbles: true }));
                    }
                    """
                )
                page.locator("#datasetViewerGrid .tabulator-tableholder").evaluate(
                    """
                    (node) => {
                      node.scrollTop = 1000;
                      node.dispatchEvent(new Event("scroll", { bubbles: true }));
                    }
                    """
                )
                page.wait_for_function(
                    """
                    () => {
                      const grid = document.querySelector('#datasetViewerGrid.dataset-viewer-grid-transposed');
                      return Boolean(grid) && !grid.querySelector('.tabulator-col.tabulator-frozen, .tabulator-cell.tabulator-frozen, .tabulator-frozen-rows-holder .tabulator-row');
                    }
                    """,
                    timeout=10_000,
                )
                page.locator("#datasetViewerTranspose").uncheck()
                page.wait_for_function(
                    """
                    () => {
                      const grid = document.querySelector('#datasetViewerGrid:not(.dataset-viewer-grid-transposed)');
                      const headers = [...grid.querySelectorAll('.tabulator-col')]
                        .filter((cell) => cell.offsetParent !== null)
                        .map((cell) => cell.getAttribute('tabulator-field'))
                        .filter(Boolean);
                      return headers.join(',') === 'c5,c2,c3';
                    }
                    """,
                    timeout=10_000,
                )
                page.locator('#datasetViewerGrid .tabulator-col[tabulator-field="c5"]').click(button="right")
                page.locator("#datasetViewerCellContextMenu:not([hidden])").get_by_role("menuitem", name="Unpin column").click()
                page.wait_for_function(
                    """
                    () => {
                      const headers = [...document.querySelectorAll('#datasetViewerGrid .tabulator-col')]
                        .filter((cell) => cell.offsetParent !== null)
                        .map((cell) => cell.getAttribute('tabulator-field'))
                        .filter(Boolean);
                      return headers.join(',') === 'c2,c3';
                    }
                    """,
                    timeout=10_000,
                )
                page.locator('#datasetViewerGrid .tabulator-col[tabulator-field="c2"]').click(button="right")
                page.locator("#datasetViewerCellContextMenu:not([hidden])").get_by_role("menuitem", name="Unpin column").click()
                page.wait_for_function(
                    """
                    () => {
                      const headers = [...document.querySelectorAll('#datasetViewerGrid .tabulator-col')]
                        .filter((cell) => cell.offsetParent !== null)
                        .map((cell) => cell.getAttribute('tabulator-field'))
                        .filter(Boolean);
                      return headers.join(',') === 'c3';
                    }
                    """,
                    timeout=10_000,
                )
                page.locator("#datasetViewerSearchClear").click()
                page.wait_for_function(
                    """
                    () => {
                      const search = document.querySelector('#datasetViewerSearch');
                      const headers = [...document.querySelectorAll('#datasetViewerGrid .tabulator-col')]
                        .filter((cell) => cell.offsetParent !== null)
                        .map((cell) => cell.getAttribute('tabulator-field'))
                        .filter(Boolean);
                      return search?.value === ''
                        && headers.join(',') === 'c0,c1,c2,c3,c4,c5,c6,c7'
                        && document.querySelector('#datasetViewerCount')?.textContent.trim() === '4 shown'
                        && !document.querySelector('#datasetViewerGrid .tabulator-col.tabulator-frozen, #datasetViewerGrid .tabulator-cell.tabulator-frozen, #datasetViewerGrid .tabulator-frozen-rows-holder .tabulator-row');
                    }
                    """,
                    timeout=10_000,
                )
                self.assertEqual(dataset_viewer_requests, pin_requests_before)
                self.assertEqual(page.locator("#datasetViewerSearch").get_attribute("placeholder"), "Select columns, separate with commas")
                page.locator("#datasetViewerSearch").fill("vehicle, Postcode")
                page.wait_for_function(
                    """
                    () => {
                      const rows = [...document.querySelectorAll('#datasetViewerGrid .tabulator-row')].filter((row) => row.offsetParent !== null);
                      const headers = [...document.querySelectorAll('#datasetViewerGrid .tabulator-col')]
                        .filter((cell) => cell.offsetParent !== null)
                        .map((cell) => cell.getAttribute('tabulator-field'))
                        .filter(Boolean);
                      return rows.length >= 4
                        && headers.join(',') === 'c2,c0,c1,c5'
                        && rows[0].querySelectorAll('.tabulator-cell').length === 4
                        && rows[0].querySelector('.tabulator-cell[tabulator-field="c5"]')?.textContent.trim() === 'AB10 1AA';
                    }
                    """,
                    timeout=10_000,
                )
                requests_before_orientation_search_toggle = dataset_viewer_requests
                page.locator("#datasetViewerTranspose").check()
                page.wait_for_function(
                    """
                    () => {
                      const grid = document.querySelector('#datasetViewerGrid.dataset-viewer-grid-transposed');
                      if (!grid) return false;
                      const rows = [...grid.querySelectorAll('.tabulator-row')].filter((row) => row.offsetParent !== null);
                      const names = rows.map((row) => row.querySelector('.tabulator-cell[tabulator-field="__field"]')?.textContent.trim()).filter(Boolean);
                      return rows.length === 4 && names.join(',') === 'vehicle_age,PostcodeArea,PostcodeSector,PostcodeUnit';
                    }
                    """,
                    timeout=10_000,
                )
                page.locator("#datasetViewerTranspose").uncheck()
                page.wait_for_function(
                    """
                    () => {
                      const grid = document.querySelector('#datasetViewerGrid:not(.dataset-viewer-grid-transposed)');
                      if (!grid) return false;
                      const rows = [...grid.querySelectorAll('.tabulator-row')].filter((row) => row.offsetParent !== null);
                      const headers = [...grid.querySelectorAll('.tabulator-col')]
                        .filter((cell) => cell.offsetParent !== null)
                        .map((cell) => cell.getAttribute('tabulator-field'))
                        .filter(Boolean);
                      return rows.length >= 4 && headers.join(',') === 'c2,c0,c1,c5';
                    }
                    """,
                    timeout=10_000,
                )
                self.assertEqual(dataset_viewer_requests, requests_before_orientation_search_toggle)
                page.locator("#datasetViewerSearch").fill("  vehicle , , Postcode  ,")
                page.wait_for_function(
                    """
                    () => {
                      const headers = [...document.querySelectorAll('#datasetViewerGrid .tabulator-col')]
                        .filter((cell) => cell.offsetParent !== null)
                        .map((cell) => cell.getAttribute('tabulator-field'))
                        .filter(Boolean);
                      const rows = [...document.querySelectorAll('#datasetViewerGrid .tabulator-row')].filter((row) => row.offsetParent !== null);
                      return rows.length >= 4 && headers.join(',') === 'c2,c0,c1,c5';
                    }
                    """,
                    timeout=10_000,
                )
                page.locator("#datasetViewerSearchClear").click()
                page.wait_for_function(
                    """
                    () => {
                      const search = document.querySelector('#datasetViewerSearch');
                      const rows = [...document.querySelectorAll('#datasetViewerGrid .tabulator-row')].filter((row) => row.offsetParent !== null);
                      const headers = [...document.querySelectorAll('#datasetViewerGrid .tabulator-col')]
                        .filter((cell) => cell.offsetParent !== null)
                        .map((cell) => cell.getAttribute('tabulator-field'))
                        .filter(Boolean);
                      return search?.value === ''
                        && rows.length >= 4
                        && headers.join(',') === 'c0,c1,c2,c3,c4,c5,c6,c7';
                    }
                    """,
                    timeout=10_000,
                )
                page.locator('#datasetViewerGrid .tabulator-col[tabulator-field="c0"] .dataset-viewer-header-label').click()
                page.locator('#datasetViewerGrid .tabulator-col[tabulator-field="c2"] .dataset-viewer-header-label').click()
                page.wait_for_function(
                    """
                    () => document.querySelectorAll('#datasetViewerGrid .dataset-viewer-column-selected').length > 0
                      && document.querySelectorAll('#datasetViewerGrid .tabulator-row.tabulator-selected').length === 0
                    """,
                    timeout=10_000,
                )
                self.assertFalse(page.locator("#datasetViewerAlphabeticalColumns").is_checked())
                page.locator("#datasetViewerAlphabeticalColumns").check()
                page.wait_for_function(
                    """
                    () => {
                      const row = document.querySelector('#datasetViewerGrid .tabulator-row');
                      const firstCell = row?.querySelector('.tabulator-cell');
                      const headers = [...document.querySelectorAll('#datasetViewerGrid .tabulator-col')]
                        .map((cell) => cell.getAttribute('tabulator-field'))
                        .filter(Boolean);
                      return firstCell?.getAttribute('tabulator-field') === 'c6'
                        && firstCell.textContent.trim() === '57.1'
                        && headers.slice(0, 3).join(',') === 'c6,c7,c0';
                    }
                    """,
                    timeout=10_000,
                )
                page.locator("#datasetViewerTranspose").check()
                page.wait_for_function(
                    """
                    () => {
                      const grid = document.querySelector('#datasetViewerGrid.dataset-viewer-grid-transposed');
                      const first = grid?.querySelector('.tabulator-row .tabulator-cell[tabulator-field="__field"]');
                      return Boolean(grid && !grid.querySelector('.dataset-viewer-transposed-table') && first?.textContent.trim() === 'lat');
                    }
                    """,
                    timeout=10_000,
                )
                transposed_resize = resize_tabulator_column('#datasetViewerGrid .tabulator-col[tabulator-field="r0"]')
                self.assertAlmostEqual(transposed_resize["before"], 150, delta=4)
                self.assertGreater(transposed_resize["after"], transposed_resize["before"] + 20)
                page.wait_for_function(
                    """
                    () => {
                      const selectedRows = [...document.querySelectorAll('#datasetViewerGrid .tabulator-row[data-dataset-viewer-column-field].tabulator-selected')]
                        .map((row) => row.querySelector('.tabulator-cell[tabulator-field="__field"]')?.textContent.trim());
                      return selectedRows.length === 2
                        && selectedRows.includes('PostcodeArea')
                        && selectedRows.includes('vehicle_age');
                    }
                    """,
                    timeout=10_000,
                )
                page.locator("#datasetViewerGrid .tabulator-row[data-dataset-viewer-column-field]", has_text="PostcodeArea").locator('.tabulator-cell[tabulator-field="r0"]').click(button="right")
                page.locator("#datasetViewerCellContextMenu:not([hidden])").get_by_text("Copy cell to clipboard").wait_for(timeout=10_000)
                page.locator("#datasetViewerCellContextMenu:not([hidden])").get_by_text("Copy selected rows to clipboard").wait_for(timeout=10_000)
                page.locator("#datasetViewerCellContextMenu:not([hidden])").get_by_text("Copy selected rows to clipboard").click()
                page.wait_for_function(
                    """
                    () => (window.__lucidumCopiedText || '') === [
                      'Column,Row 1,Row 2,Row 3,Row 4',
                      'PostcodeArea,AB,AB,AL,AL',
                      'vehicle_age,1,2,3,4',
                    ].join('\\n')
                    """,
                    timeout=10_000,
                )
                page.locator("#datasetViewerGrid .tabulator-row[data-dataset-viewer-column-field]", has_text="PostcodeArea").locator('.tabulator-cell[tabulator-field="r0"]').click(button="right")
                page.locator("#datasetViewerCellContextMenu:not([hidden])").get_by_text("Clear selection").wait_for(timeout=10_000)
                self.assertEqual(page.locator("#datasetViewerCellContextMenu:not([hidden]) [role='separator']").count(), 2)
                page.locator("#datasetViewerCellContextMenu:not([hidden])").get_by_text("Clear selection").click()
                page.wait_for_function(
                    """
                    () => document.querySelectorAll('#datasetViewerGrid .tabulator-row[data-dataset-viewer-column-field].tabulator-selected').length === 0
                      && document.querySelectorAll('#datasetViewerGrid .tabulator-cell.dataset-viewer-transposed-column-selected').length === 0
                    """,
                    timeout=10_000,
                )
                page.locator('#datasetViewerGrid .tabulator-col[tabulator-field="r0"] .dataset-viewer-transposed-header-label').click()
                page.wait_for_function(
                    """
                    () => document.querySelectorAll('#datasetViewerGrid .tabulator-row[data-dataset-viewer-column-field].tabulator-selected').length === 0
                      && document.querySelectorAll('#datasetViewerGrid .tabulator-cell.dataset-viewer-transposed-column-selected').length > 0
                    """,
                    timeout=10_000,
                )
                page.locator('#datasetViewerGrid .tabulator-row[data-dataset-viewer-column-field]').first.locator('.tabulator-cell[tabulator-field="r0"]').click(button="right")
                page.locator("#datasetViewerCellContextMenu:not([hidden])").get_by_text("Copy cell to clipboard").wait_for(timeout=10_000)
                page.locator("#datasetViewerCellContextMenu:not([hidden])").get_by_text("Copy selected column to clipboard").wait_for(timeout=10_000)
                page.locator("#datasetViewerCellContextMenu:not([hidden])").get_by_text("Copy cell to clipboard").click()
                page.wait_for_function("() => window.__lucidumCopiedText === '57.1'", timeout=10_000)
                page.locator('#datasetViewerGrid .tabulator-row[data-dataset-viewer-column-field]').first.locator('.tabulator-cell[tabulator-field="r0"]').click(button="right")
                page.locator("#datasetViewerCellContextMenu:not([hidden])").get_by_text("Copy selected column to clipboard").wait_for(timeout=10_000)
                page.locator("#datasetViewerCellContextMenu:not([hidden])").get_by_text("Copy selected column to clipboard").click()
                page.wait_for_function(
                    """
                    () => (window.__lucidumCopiedText || '') === [
                      'Column,Row 1',
                      'lat,57.1',
                      'long,-2.1',
                      'PostcodeArea,AB',
                      'PostcodeSector,AB10 1',
                      'PostcodeUnit,AB10 1AA',
                      'price,100',
                      'value,10',
                      'vehicle_age,1',
                    ].join('\\n')
                    """,
                    timeout=10_000,
                )
                page.locator('#datasetViewerGrid .tabulator-col[tabulator-field="r1"] .dataset-viewer-transposed-header-label').click()
                page.wait_for_function(
                    """
                    () => document.querySelectorAll('#datasetViewerGrid .tabulator-cell.dataset-viewer-transposed-column-selected').length > 1
                    """,
                    timeout=10_000,
                )
                page.locator("#datasetViewerGrid").click(button="right")
                page.locator("#datasetViewerCellContextMenu:not([hidden])").get_by_text("Copy selected columns to clipboard").wait_for(timeout=10_000)
                page.locator("#datasetViewerCellContextMenu:not([hidden])").get_by_text("Copy selected columns to clipboard").click()
                page.wait_for_function(
                    """
                    () => (window.__lucidumCopiedText || '') === [
                      'Column,Row 1,Row 2',
                      'lat,57.1,57.2',
                      'long,-2.1,-2.2',
                      'PostcodeArea,AB,AB',
                      'PostcodeSector,AB10 1,AB10 1',
                      'PostcodeUnit,AB10 1AA,AB10 1AB',
                      'price,100,200',
                      'value,10,20',
                      'vehicle_age,1,2',
                    ].join('\\n')
                    """,
                    timeout=10_000,
                )
                page.locator('#datasetViewerGrid .tabulator-col[tabulator-field="r1"] .dataset-viewer-transposed-header-label').click()
                page.wait_for_function(
                    """
                    () => document.querySelectorAll('#datasetViewerGrid .tabulator-cell.dataset-viewer-transposed-column-selected').length > 0
                    """,
                    timeout=10_000,
                )
                page.locator("#datasetViewerAlphabeticalColumns").uncheck()
                page.wait_for_function(
                    """
                    () => document.querySelector('#datasetViewerGrid.dataset-viewer-grid-transposed .tabulator-row .tabulator-cell[tabulator-field="__field"]')?.textContent.trim() === 'PostcodeArea'
                    """
                )
                page.locator('#datasetViewerGrid .tabulator-col[tabulator-field="__field"] .dataset-viewer-transposed-sort-button').click()
                page.wait_for_function(
                    """
                    () => document.querySelector('#datasetViewerGrid.dataset-viewer-grid-transposed .tabulator-row .tabulator-cell[tabulator-field="__field"]')?.textContent.trim() === 'lat'
                      && document.querySelector('#datasetViewerGrid .tabulator-col[tabulator-field="__field"] .dataset-viewer-transposed-sort-button')?.dataset.sortDir === 'asc'
                    """,
                    timeout=10_000,
                )
                page.locator('#datasetViewerGrid .tabulator-col[tabulator-field="__field"] .dataset-viewer-transposed-sort-button').click()
                page.wait_for_function(
                    """
                    () => document.querySelector('#datasetViewerGrid.dataset-viewer-grid-transposed .tabulator-row .tabulator-cell[tabulator-field="__field"]')?.textContent.trim() === 'vehicle_age'
                      && document.querySelector('#datasetViewerGrid .tabulator-col[tabulator-field="__field"] .dataset-viewer-transposed-sort-button')?.dataset.sortDir === 'desc'
                    """,
                    timeout=10_000,
                )
                page.locator('#datasetViewerGrid .tabulator-col[tabulator-field="__field"] .dataset-viewer-transposed-sort-button').click()
                page.wait_for_function(
                    """
                    () => document.querySelector('#datasetViewerGrid.dataset-viewer-grid-transposed .tabulator-row .tabulator-cell[tabulator-field="__field"]')?.textContent.trim() === 'PostcodeArea'
                      && document.querySelector('#datasetViewerGrid .tabulator-col[tabulator-field="__field"] .dataset-viewer-transposed-sort-button')?.dataset.sortDir === 'none'
                    """,
                    timeout=10_000,
                )
                page.locator('#datasetViewerGrid .tabulator-col[tabulator-field="r0"] .dataset-viewer-transposed-header-label').click()
                page.wait_for_function(
                    """
                    () => document.querySelectorAll('#datasetViewerGrid .tabulator-cell.dataset-viewer-transposed-column-selected').length === 0
                    """,
                    timeout=10_000,
                )
                page.locator('#datasetViewerGrid .tabulator-row[data-dataset-viewer-column-field]').first.locator('.tabulator-cell[tabulator-field="r0"]').click(button="right")
                page.locator("#datasetViewerCellContextMenu:not([hidden])").get_by_text("Copy cell to clipboard").wait_for(timeout=10_000)
                page.locator("#datasetViewerCellContextMenu:not([hidden])").get_by_text("Copy displayed table to clipboard").wait_for(timeout=10_000)
                self.assertEqual(page.locator("#datasetViewerCellContextMenu:not([hidden]) [role='separator']").count(), 1)
                page.locator("#datasetViewerCellContextMenu:not([hidden])").get_by_text("Copy displayed table to clipboard").click()
                page.wait_for_function(
                    """
                    () => (window.__lucidumCopiedText || '') === [
                      'Column,Row 1,Row 2,Row 3,Row 4',
                      'PostcodeArea,AB,AB,AL,AL',
                      'PostcodeSector,AB10 1,AB10 1,AL1 1,AL1 2',
                      'vehicle_age,1,2,3,4',
                      'price,100,200,300,400',
                      'value,10,20,30,40',
                      'PostcodeUnit,AB10 1AA,AB10 1AB,AL1 1AA,AL1 2AA',
                      'lat,57.1,57.2,51.8,51.7',
                      'long,-2.1,-2.2,-0.3,-0.2',
                    ].join('\\n')
                    """,
                    timeout=10_000,
                )
                page.locator('#datasetViewerGrid .tabulator-row[data-dataset-viewer-column-field]').first.locator('.tabulator-cell[tabulator-field="r0"]').click(button="right")
                page.locator("#datasetViewerCellContextMenu:not([hidden])").get_by_text("Copy cell to clipboard").wait_for(timeout=10_000)
                page.locator("#datasetViewerCellContextMenu:not([hidden])").get_by_text("Copy cell to clipboard").click()
                page.wait_for_function("() => window.__lucidumCopiedText === 'AB'", timeout=10_000)
                page.locator('#datasetViewerGrid .tabulator-col[tabulator-field="r0"] .dataset-viewer-transposed-header-label').click()
                page.wait_for_function(
                    """
                    () => document.querySelectorAll('#datasetViewerGrid .tabulator-cell.dataset-viewer-transposed-column-selected').length > 0
                    """,
                    timeout=10_000,
                )
                transposed_requests_before_search = dataset_viewer_requests
                page.locator("#datasetViewerGrid .tabulator-tableholder").evaluate(
                    "node => { node.dataset.datasetViewerSearchProbe = 'kept'; }"
                )
                page.locator("#datasetViewerSearch").fill("PostcodeUnit, vehicle")
                page.wait_for_function(
                    """
                    () => {
                      const grid = document.querySelector('#datasetViewerGrid.dataset-viewer-grid-transposed');
                      if (!grid) return false;
                      const holder = grid.querySelector('.tabulator-tableholder');
                      const headers = [...grid.querySelectorAll('.tabulator-col')].map((cell) => cell.textContent.trim());
                      const rows = [...grid.querySelectorAll('.tabulator-row')].filter((row) => row.offsetParent !== null);
                      const names = rows.map((row) => row.querySelector('.tabulator-cell[tabulator-field="__field"]')?.textContent.trim()).filter(Boolean);
                      return holder?.dataset.datasetViewerSearchProbe === 'kept'
                        && headers.length === 5
                        && headers[0] === 'Column'
                        && headers[1] === 'Row 1'
                        && headers[4] === 'Row 4'
                        && rows.length === 2
                        && names.join(',') === 'PostcodeUnit,vehicle_age'
                        && rows.some((row) => row.querySelector('.tabulator-cell[tabulator-field="r0"]')?.textContent.trim() === 'AB10 1AA')
                        && rows.some((row) => row.querySelector('.tabulator-cell[tabulator-field="r0"]')?.textContent.trim() === '1');
                    }
                    """,
                    timeout=10_000,
                )
                self.assertEqual(dataset_viewer_requests, transposed_requests_before_search)
                page.locator("#datasetViewerTranspose").uncheck()
                page.wait_for_function(
                    """
                    () => {
                      const grid = document.querySelector('#datasetViewerGrid:not(.dataset-viewer-grid-transposed)');
                      if (!grid) return false;
                      const rows = [...grid.querySelectorAll('.tabulator-row')].filter((row) => row.offsetParent !== null);
                      const headers = [...grid.querySelectorAll('.tabulator-col')]
                        .filter((cell) => cell.offsetParent !== null)
                        .map((cell) => cell.getAttribute('tabulator-field'))
                        .filter(Boolean);
                      return rows.length >= 4 && headers.join(',') === 'c5,c2';
                    }
                    """,
                    timeout=10_000,
                )
                self.assertEqual(dataset_viewer_requests, transposed_requests_before_search)
                page.locator("#datasetViewerTranspose").check()
                page.wait_for_function(
                    """
                    () => {
                      const grid = document.querySelector('#datasetViewerGrid.dataset-viewer-grid-transposed');
                      if (!grid) return false;
                      const rows = [...grid.querySelectorAll('.tabulator-row')].filter((row) => row.offsetParent !== null);
                      const names = rows.map((row) => row.querySelector('.tabulator-cell[tabulator-field="__field"]')?.textContent.trim()).filter(Boolean);
                      return rows.length === 2 && names.join(',') === 'PostcodeUnit,vehicle_age';
                    }
                    """,
                    timeout=10_000,
                )
                page.locator("#datasetViewerGrid .tabulator-tableholder").evaluate(
                    "node => { node.dataset.datasetViewerSearchProbe = 'kept'; }"
                )
                page.locator("#datasetViewerSearchClear").click()
                page.wait_for_function(
                    """
                    () => document.querySelector('#datasetViewerGrid.dataset-viewer-grid-transposed .tabulator-tableholder')?.dataset.datasetViewerSearchProbe === 'kept'
                      && document.querySelectorAll('#datasetViewerGrid.dataset-viewer-grid-transposed .tabulator-col').length >= 5
                    """,
                    timeout=10_000,
                )
                self.assertEqual(dataset_viewer_requests, transposed_requests_before_search)
                page.locator("#datasetViewerTranspose").uncheck()
                page.wait_for_function(
                    """
                    () => document.querySelector('#datasetViewerGrid .tabulator-row .tabulator-cell[tabulator-field="c0"]')?.textContent.trim() === 'AB'
                    """
                )
                page.wait_for_function(
                    """
                    () => document.querySelectorAll('#datasetViewerGrid .tabulator-row.tabulator-selected').length === 1
                      && document.querySelector('#datasetViewerGrid .tabulator-row.tabulator-selected .tabulator-cell[tabulator-field="c5"]')?.textContent.trim() === 'AB10 1AA'
                    """,
                    timeout=10_000,
                )
                self.assertAlmostEqual(
                    tabulator_column_width('#datasetViewerGrid .tabulator-col[tabulator-field="c0"]'),
                    normal_resize["after"],
                    delta=4,
                )
                page.locator("#datasetViewerTranspose").check()
                page.wait_for_function(
                    """
                    () => {
                      const grid = document.querySelector('#datasetViewerGrid.dataset-viewer-grid-transposed');
                      const first = grid?.querySelector('.tabulator-row .tabulator-cell[tabulator-field="__field"]');
                      return Boolean(grid && first?.textContent.trim() === 'PostcodeArea');
                    }
                    """,
                    timeout=10_000,
                )
                self.assertAlmostEqual(
                    tabulator_column_width('#datasetViewerGrid .tabulator-col[tabulator-field="r0"]'),
                    transposed_resize["after"],
                    delta=4,
                )
                page.locator("#datasetViewerTranspose").uncheck()
                page.wait_for_function(
                    """
                    () => document.querySelector('#datasetViewerGrid .tabulator-row .tabulator-cell[tabulator-field="c0"]')?.textContent.trim() === 'AB'
                    """,
                    timeout=10_000,
                )
                page.wait_for_function(
                    """
                    () => {
                        const text = document.querySelector("#actionTimingMonitor")?.textContent || "";
                        return /^DuckDB: \\d+(?:ns|us|ms), JSON: \\d+ms, Dataset render: \\d+(?:ns|us|ms), Total: \\d+ms$/.test(text);
                    }
                    """
                )
                page.locator("#profileTool").click()
                page.locator("#profileWrap:not(.hidden) .profile-table").wait_for(timeout=10_000)
                assert_dataset_viewer_hidden()
                assert_profile_table_unoccluded()
                page.wait_for_function(
                    """
                    () => {
                      const toolbar = document.querySelector(".profile-toolbar");
                      const meta = document.querySelector("#profileMeta");
                      const group = document.querySelector("#profileGroupMeta");
                      const filter = document.querySelector("#profileFilter");
                      return Boolean(toolbar && meta && toolbar.contains(meta) && meta.contains(group) && meta.contains(filter));
                    }
                    """,
                    timeout=10_000,
                )
                profile_meta_position = page.evaluate(
                    """
                    () => {
                      const rect = document.querySelector("#profileMeta")?.getBoundingClientRect();
                      return rect ? { top: rect.top, rightOffset: window.innerWidth - rect.right } : null;
                    }
                    """
                )
                self.assertIsNotNone(profile_meta_position)
                self.assertAlmostEqual(profile_meta_position["top"], dataset_viewer_meta_position["top"], delta=1)
                self.assertAlmostEqual(profile_meta_position["rightOffset"], dataset_viewer_meta_position["rightOffset"], delta=1)
                page.locator('#profileWrap .profile-summary-row[aria-selected="true"]').wait_for(timeout=10_000)
                page.locator("#profileDetailTitle").get_by_text("PostcodeArea").wait_for(timeout=10_000)
                self.assertEqual(page.locator("#profileFilter").evaluate("node => getComputedStyle(node).fontSize"), "10px")
                self.assertTrue(page.locator('input[name="profileSummaryMode"][value="auto"]').is_checked())
                with page.expect_request(lambda request: request.url.endswith("/api/column-profile/summary"), timeout=10_000) as profile_full_request_info:
                    page.locator(".profile-summary-mode-option", has_text="Use all rows").click()
                profile_full_payload = json.loads(profile_full_request_info.value.post_data or "{}")
                self.assertEqual(profile_full_payload["mode"], "full")
                page.locator('input[name="profileSummaryMode"][value="full"]').wait_for(state="attached", timeout=10_000)
                self.assertTrue(page.locator('input[name="profileSummaryMode"][value="full"]').is_checked())
                page.locator("#profileDetailTitle").get_by_text("PostcodeArea").wait_for(timeout=10_000)
                with page.expect_request(lambda request: request.url.endswith("/api/column-profile/summary"), timeout=10_000) as profile_filter_request_info:
                    page.evaluate(
                        """
                        () => {
                            document.querySelector("#filterInput").value = "vehicle_age >= 1";
                            document.querySelector("#filterApplyBtn").click();
                        }
                        """
                    )
                profile_filter_payload = json.loads(profile_filter_request_info.value.post_data or "{}")
                self.assertEqual(profile_filter_payload["filter"], "vehicle_age >= 1")
                self.assertEqual(profile_filter_payload["mode"], "full")
                page.wait_for_function(
                    """
                    () => {
                      const meta = document.querySelector("#profileFilter");
                      if (!meta?.classList.contains("profile-filter--applied")) return false;
                      const probe = document.createElement("span");
                      probe.style.backgroundColor = getComputedStyle(document.body).getPropertyValue("--filter-applied-bg").trim();
                      document.body.append(probe);
                      const expectedBackground = getComputedStyle(probe).backgroundColor;
                      probe.remove();
                      return getComputedStyle(meta).backgroundColor === expectedBackground;
                    }
                    """,
                    timeout=10_000,
                )
                assert_filter_badge_clear("#profileFilterClearBtn", "#profileFilterText", True)
                page.locator('input[name="profileSummaryMode"][value="full"]').wait_for(state="attached", timeout=10_000)
                self.assertTrue(page.locator('input[name="profileSummaryMode"][value="full"]').is_checked())
                page.locator("#profileDetailTitle").get_by_text("PostcodeArea").wait_for(timeout=10_000)
                with page.expect_request(lambda request: request.url.endswith("/api/column-profile/summary"), timeout=10_000) as profile_auto_request_info:
                    page.locator(".profile-summary-mode-option", has_text="Use 100k").click()
                profile_auto_payload = json.loads(profile_auto_request_info.value.post_data or "{}")
                self.assertEqual(profile_auto_payload["mode"], "auto")
                page.locator('input[name="profileSummaryMode"][value="auto"]').wait_for(state="attached", timeout=10_000)
                self.assertTrue(page.locator('input[name="profileSummaryMode"][value="auto"]').is_checked())
                page.locator("#profileDetailTitle").get_by_text("PostcodeArea").wait_for(timeout=10_000)
                with page.expect_request(lambda request: request.url.endswith("/api/column-profile/summary"), timeout=10_000) as profile_clear_filter_request_info:
                    with page.expect_response(lambda response: response.url.endswith("/api/column-profile/summary") and response.status == 200, timeout=10_000):
                        page.locator("#profileFilterClearBtn").click()
                profile_clear_filter_payload = json.loads(profile_clear_filter_request_info.value.post_data or "{}")
                self.assertEqual(profile_clear_filter_payload["filter"], "")
                self.assertEqual(profile_clear_filter_payload["mode"], "auto")
                page.wait_for_function(
                    """
                    () => {
                      const meta = document.querySelector("#profileFilter");
                      return meta && !meta.classList.contains("profile-filter--applied") && meta.textContent.includes("no filter");
                    }
                    """,
                    timeout=10_000,
                )
                assert_filter_badge_clear("#profileFilterClearBtn", "#profileFilterText", False)
                page.locator("#profileDetailTitle").get_by_text("PostcodeArea").wait_for(timeout=10_000)
                page.locator('#profileWrap .profile-summary-row[data-profile-column="vehicle_age"]').wait_for(timeout=10_000)
                profile_requests_before_search = profile_requests
                page.locator("#profileColumnSearch").fill("Postcode")
                page.wait_for_function(
                    '() => document.querySelector("#profileWrap .profile-summary-row[data-profile-column=\\"vehicle_age\\"]")?.hidden === true'
                )
                postcode_area_row = page.locator('#profileWrap .profile-summary-row[data-profile-column="PostcodeArea"]')
                self.assertEqual(postcode_area_row.get_attribute("aria-selected"), "true")
                self.assertFalse(postcode_area_row.evaluate("node => node.hidden"))
                self.assertEqual(profile_requests, profile_requests_before_search)
                page.locator("#profileColumnSearch").fill("")
                page.wait_for_function(
                    '() => document.querySelector("#profileWrap .profile-summary-row[data-profile-column=\\"vehicle_age\\"]")?.hidden === false'
                )
                self.assertEqual(profile_requests, profile_requests_before_search)
                vehicle_age_row = page.locator('#profileWrap .profile-summary-row[data-profile-column="vehicle_age"]')
                self.assertEqual(vehicle_age_row.evaluate("node => getComputedStyle(node).userSelect"), "none")
                page.evaluate(
                    """
                    () => {
                        window.__lucidumCopiedText = null;
                        Object.defineProperty(navigator, "clipboard", {
                            configurable: true,
                            value: {
                                writeText: async (text) => {
                                    window.__lucidumCopiedText = text;
                                },
                            },
                        });
                    }
                    """
                )
                vehicle_age_row.click(button="right")
                page.locator("#profileColumnContextMenu:not([hidden])").get_by_text("Copy feature to clipboard").wait_for(timeout=10_000)
                self.assertTrue(page.evaluate(
                    """
                    () => {
                      const item = document.querySelector("#profileColumnContextMenu:not([hidden]) [role=menuitem]");
                      if (!item) return false;
                      item.click();
                      return true;
                    }
                    """
                ))
                page.wait_for_function("() => window.__lucidumCopiedText === 'vehicle_age'")
                page.wait_for_function('() => document.querySelector("#profileColumnContextMenu")?.hidden === true')
                page.locator("#clipboardToast").get_by_text("Copied vehicle_age to clipboard").wait_for(timeout=10_000)
                clipboard_toast_style = page.locator("#clipboardToast").evaluate(
                    """
                    node => {
                      const style = getComputedStyle(node);
                      return {
                        position: style.position,
                        top: style.top,
                        right: style.right,
                        bottom: style.bottom,
                      };
                    }
                    """
                )
                self.assertEqual(clipboard_toast_style["position"], "fixed")
                self.assertEqual(clipboard_toast_style["top"], "18px")
                self.assertEqual(clipboard_toast_style["right"], "18px")
                self.assertNotEqual(clipboard_toast_style["bottom"], "18px")
                self.assertEqual(page.locator("#status").text_content(), "")
                self.assertNotEqual(vehicle_age_row.get_attribute("aria-selected"), "true")
                vehicle_age_row.click()
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
                page.wait_for_function(
                    """
                    () => {
                      const version = document.querySelector("#sidebarVersion");
                      return version
                        && version.offsetParent !== null
                        && version.textContent.trim().length > 0;
                    }
                    """,
                    timeout=10_000,
                )
                self.assertEqual(page.locator("#sidebarVersion").inner_text().strip(), f"lucidum v{__version__}")
                self.assertEqual(page.locator("#collapsedSidebarVersion").inner_text().strip(), f"v{__version__}")
                self.assertFalse(page.locator("#collapsedSidebarVersion").is_visible())
                self.assertTrue(page.locator("#toolSelectorSection .tool-option:not(.hidden)").first.is_visible())
                expanded_first_tool_top = page.locator("#toolSelectorSection .tool-option:not(.hidden)").first.bounding_box()
                self.assertIsNotNone(expanded_first_tool_top)
                expanded_tool_state = page.evaluate(
                    """
                    () => {
                      const buttons = [...document.querySelectorAll("#toolSelectorSection .tool-option:not(.hidden)")];
                      const selector = document.querySelector("#toolSelectorSection .tool-selector");
                      const rail = document.querySelector("#toolSelectorSection");
                      const pane = document.querySelector("#sidebarControlPane");
                      const sidebar = document.querySelector("#appSidebar");
                      const sidebarResizer = document.querySelector("#sidebarResizer");
                      const lineBarSidePanel = document.querySelector(".chart-side-section");
                      const railRect = rail.getBoundingClientRect();
                      const paneRect = pane.getBoundingClientRect();
                      const sidebarRect = sidebar.getBoundingClientRect();
                      const sidebarResizerRect = sidebarResizer.getBoundingClientRect();
                      const sidebarResizerStyle = getComputedStyle(sidebarResizer);
                      const sidebarResizerLineStyle = getComputedStyle(sidebarResizer, "::before");
                      const railStyle = getComputedStyle(rail);
                      const lefts = buttons.map((button) => Math.round(button.getBoundingClientRect().left));
                      const tops = buttons.map((button) => Math.round(button.getBoundingClientRect().top));
                      const sidebarBackground = getComputedStyle(sidebar).backgroundColor;
                      return {
                        count: buttons.length,
                        selectorDisplay: getComputedStyle(selector).display,
                        selectorMatchesSidebar: getComputedStyle(selector).backgroundColor === sidebarBackground,
                        sidebarMatchesLineBarSidePanel: sidebarBackground === getComputedStyle(lineBarSidePanel).backgroundColor,
                        gap: getComputedStyle(selector).gap,
                        overflowX: getComputedStyle(selector).overflowX,
                        overflowY: getComputedStyle(selector).overflowY,
                        vertical: new Set(lefts).size === 1 && new Set(tops).size === buttons.length,
                        railWidth: Math.round(railRect.width),
                        railAtSidebarLeft: Math.round(railRect.left) === Math.round(sidebarRect.left),
                        paneRightOfRail: Math.round(paneRect.left) >= Math.round(railRect.right),
                        sidebarResizerFillTransparent: sidebarResizerStyle.backgroundColor === "rgba(0, 0, 0, 0)",
                        siderailLineWidth: Math.round(parseFloat(railStyle.borderRightWidth)),
                        sidebarResizerLineWidth: Math.round(parseFloat(sidebarResizerLineStyle.width)),
                        sidebarResizerMatchesSiderailLine: sidebarResizerLineStyle.backgroundColor === railStyle.borderRightColor
                          && Math.round(parseFloat(sidebarResizerLineStyle.width)) === Math.round(parseFloat(railStyle.borderRightWidth)),
                        sidebarResizerWidth: Math.round(sidebarResizerRect.width),
                        labelsHidden: buttons.every((button) => {
                          const label = button.querySelector(".tool-label");
                          const style = label ? getComputedStyle(label) : null;
                          return style?.position === "absolute" && style.width === "1px" && style.overflow === "hidden";
                        }),
                      };
                    }
                    """
                )
                self.assertEqual(
                    expanded_tool_state,
                    {
                        "count": 6,
                        "selectorDisplay": "grid",
                        "selectorMatchesSidebar": True,
                        "sidebarMatchesLineBarSidePanel": True,
                        "gap": "12px",
                        "overflowX": "visible",
                        "overflowY": "visible",
                        "vertical": True,
                        "railWidth": 50,
                        "railAtSidebarLeft": True,
                        "paneRightOfRail": True,
                        "sidebarResizerFillTransparent": True,
                        "siderailLineWidth": 1,
                        "sidebarResizerLineWidth": 1,
                        "sidebarResizerMatchesSiderailLine": True,
                        "sidebarResizerWidth": 9,
                        "labelsHidden": True,
                    },
                )
                page.locator("#sidebarToggleBtn").click()
                self.assertEqual(page.locator("#sidebarToggleBtn").get_attribute("aria-expanded"), "false")
                collapsed_first_tool_top = page.locator("#toolSelectorSection .tool-option:not(.hidden)").first.bounding_box()
                self.assertIsNotNone(collapsed_first_tool_top)
                assert expanded_first_tool_top is not None
                assert collapsed_first_tool_top is not None
                self.assertLessEqual(abs(collapsed_first_tool_top["y"] - expanded_first_tool_top["y"]), 1)
                self.assertIsNone(page.locator("#appSidebar").get_attribute("aria-hidden"))
                self.assertTrue(page.locator("#profileTool").is_visible())
                self.assertTrue(page.locator("#datasetViewerTool").is_visible())
                self.assertTrue(page.locator("#lineBarTool").is_visible())
                self.assertTrue(page.locator("#histogramTool").is_visible())
                self.assertTrue(page.locator("#ukMapTool").is_visible())
                collapsed_tool_state = page.evaluate(
                    """
                    () => {
                      const buttons = [...document.querySelectorAll("#toolSelectorSection .tool-option:not(.hidden)")];
                      const selector = document.querySelector("#toolSelectorSection .tool-selector");
                      const sidebar = document.querySelector("#appSidebar");
                      const lineBarSidePanel = document.querySelector(".chart-side-section");
                      const lefts = buttons.map((button) => Math.round(button.getBoundingClientRect().left));
                      const tops = buttons.map((button) => Math.round(button.getBoundingClientRect().top));
                      const sidebarBackground = getComputedStyle(sidebar).backgroundColor;
                      return {
                        count: buttons.length,
                        selectorDisplay: getComputedStyle(selector).display,
                        selectorMatchesSidebar: getComputedStyle(selector).backgroundColor === sidebarBackground,
                        sidebarMatchesLineBarSidePanel: sidebarBackground === getComputedStyle(lineBarSidePanel).backgroundColor,
                        gap: getComputedStyle(selector).gap,
                        overflowX: getComputedStyle(selector).overflowX,
                        overflowY: getComputedStyle(selector).overflowY,
                        vertical: new Set(lefts).size === 1 && new Set(tops).size === buttons.length,
                        allButtonsSquare: buttons.every((button) => {
                          const rect = button.getBoundingClientRect();
                          return Math.round(rect.width) === 36 && Math.round(rect.height) === 36;
                        }),
                        allButtonsBorderless: buttons.every((button) => getComputedStyle(button).borderTopWidth === "0px"),
                        allButtonsTransparentWithActiveAccent: buttons.every((button) => {
                          const style = getComputedStyle(button);
                          const expectedColor = button.classList.contains("active") ? "rgb(34, 118, 210)" : "rgb(102, 112, 133)";
                          return style.backgroundColor === "rgba(0, 0, 0, 0)" && style.color === expectedColor;
                        }),
                        allButtonsFullOpacity: buttons.every((button) => getComputedStyle(button).opacity === "1"),
                        allIconsLarge: buttons.every((button) => {
                          const icon = button.querySelector(".tool-icon");
                          const renderedIcon = button.querySelector(".tool-icon svg, .tool-icon img");
                          const iconRect = icon?.getBoundingClientRect();
                          const renderedRect = renderedIcon?.getBoundingClientRect();
                          return Math.round(iconRect?.width || 0) === 30
                            && Math.round(iconRect?.height || 0) === 30
                            && Math.round(renderedRect?.width || 0) >= 28
                            && Math.round(renderedRect?.height || 0) >= 28;
                        }),
                      };
                    }
                    """
                )
                self.assertEqual(
                    collapsed_tool_state,
                    {
                        "count": 6,
                        "selectorDisplay": "grid",
                        "selectorMatchesSidebar": True,
                        "sidebarMatchesLineBarSidePanel": True,
                        "gap": "12px",
                        "overflowX": "visible",
                        "overflowY": "visible",
                        "vertical": True,
                        "allButtonsSquare": True,
                        "allButtonsBorderless": True,
                        "allButtonsTransparentWithActiveAccent": True,
                        "allButtonsFullOpacity": True,
                        "allIconsLarge": True,
                    },
                )
                self.assert_tool_button_tooltip_right_of_icon(page, "#ukMapTool", "UK mapping")
                self.assertFalse(page.locator(".sidebar-metric-section").is_visible())
                self.assertFalse(page.locator(".sidebar-favourites-section").is_visible())
                self.assertFalse(page.locator(".sidebar-filter-section").is_visible())
                self.assertFalse(page.locator("#sidebarVersion").is_visible())
                self.assertTrue(page.locator("#collapsedSidebarVersion").is_visible())
                self.assertEqual(page.locator("#collapsedSidebarVersion").inner_text().strip(), f"v{__version__}")
                self.assertFalse(page.locator("#sidebarResizer").is_visible())

                page.locator("#reloadBtn").click()
                page.wait_for_function('() => document.querySelector("#sidebarToggleBtn")?.getAttribute("aria-expanded") === "false"')
                self.assertTrue(page.locator("#datasetViewerTool").is_visible())
                self.assertTrue(page.locator("#profileTool").is_visible())
                self.assertTrue(page.locator("#lineBarTool").is_visible())
                self.assertTrue(page.locator("#histogramTool").is_visible())
                self.assertTrue(page.locator("#ukMapTool").is_visible())
                self.assertFalse(page.locator(".sidebar-metric-section").is_visible())
                self.assertFalse(page.locator(".sidebar-favourites-section").is_visible())
                self.assertFalse(page.locator(".sidebar-filter-section").is_visible())
                self.assertFalse(page.locator("#sidebarVersion").is_visible())
                self.assertTrue(page.locator("#collapsedSidebarVersion").is_visible())
                self.assertEqual(page.locator("#collapsedSidebarVersion").inner_text().strip(), f"v{__version__}")
                self.assertFalse(page.locator("#sidebarResizer").is_visible())

                page.locator("#ukMapTool").click()
                page.locator("#ukMap:not(.hidden)").wait_for(timeout=20_000)
                page.locator("#mapFloatingControl:not(.hidden)").wait_for(timeout=10_000)
                page.wait_for_function("() => window.L && document.querySelector('#ukMap .leaflet-pane')")
                page.wait_for_function("() => document.querySelector('#ukMap')?.classList.contains('map-bg-light')")
                page.wait_for_function('() => document.querySelector("#mapGroupMeta")?.textContent.includes("areas matched")')
                map_toggle = page.locator("#mapControlReset")
                self.assertEqual(map_toggle.get_attribute("aria-expanded"), "true")
                self.assertEqual(
                    page.evaluate('() => getComputedStyle(document.querySelector("#mapControlReset")).transform'),
                    "matrix(1, 0, 0, 1, 5, -5)",
                )
                expanded_button_box = map_toggle.bounding_box()
                self.assertIsNotNone(expanded_button_box)

                def expected_top_right_button_box() -> dict[str, float]:
                    return page.evaluate(
                        """
                        () => {
                            const panel = document.querySelector("#mapFloatingControl");
                            const button = document.querySelector("#mapControlReset");
                            const container = panel.offsetParent || panel.closest(".workspace");
                            const wasCollapsed = panel.classList.contains("collapsed");
                            const previous = { left: panel.style.left, top: panel.style.top, right: panel.style.right };
                            if (wasCollapsed) panel.classList.remove("collapsed");
                            const rect = container.getBoundingClientRect();
                            const frame = {
                                left: rect.left + container.clientLeft,
                                top: rect.top + container.clientTop,
                                width: container.clientWidth,
                            };
                            const panelLeft = Math.max(8, frame.width - panel.offsetWidth - 8);
                            panel.style.left = `${panelLeft}px`;
                            panel.style.top = "4px";
                            panel.style.right = "auto";
                            const buttonRect = button.getBoundingClientRect();
                            const result = { x: buttonRect.x, y: buttonRect.y };
                            panel.style.left = previous.left;
                            panel.style.top = previous.top;
                            panel.style.right = previous.right;
                            if (wasCollapsed) panel.classList.add("collapsed");
                            return result;
                        }
                        """
                    )

                def wait_for_map_toggle_top_right() -> dict[str, float]:
                    page.wait_for_function(
                        """
                        () => {
                            const panel = document.querySelector("#mapFloatingControl");
                            const button = document.querySelector("#mapControlReset");
                            if (!panel || !button) return false;
                            const container = panel.offsetParent || panel.closest(".workspace");
                            if (!container) return false;
                            const wasCollapsed = panel.classList.contains("collapsed");
                            const previous = { left: panel.style.left, top: panel.style.top, right: panel.style.right };
                            if (wasCollapsed) panel.classList.remove("collapsed");
                            const rect = container.getBoundingClientRect();
                            const frameLeft = rect.left + container.clientLeft;
                            const frameTop = rect.top + container.clientTop;
                            const panelLeft = Math.max(8, container.clientWidth - panel.offsetWidth - 8);
                            panel.style.left = `${panelLeft}px`;
                            panel.style.top = "4px";
                            panel.style.right = "auto";
                            const expectedRect = button.getBoundingClientRect();
                            const expectedX = expectedRect.x;
                            const expectedY = expectedRect.y;
                            panel.style.left = previous.left;
                            panel.style.top = previous.top;
                            panel.style.right = previous.right;
                            if (wasCollapsed) panel.classList.add("collapsed");
                            const buttonRect = button.getBoundingClientRect();
                            return Math.abs(buttonRect.x - expectedX) <= 1
                                && Math.abs(buttonRect.y - expectedY) <= 1
                                && expectedX >= frameLeft
                                && expectedY >= frameTop;
                        }
                        """,
                        timeout=10_000,
                    )
                    box = map_toggle.bounding_box()
                    self.assertIsNotNone(box)
                    return box

                wait_for_map_toggle_top_right()
                map_toggle.click()
                page.wait_for_function('() => document.querySelector("#mapFloatingControl")?.classList.contains("collapsed")')
                self.assertEqual(map_toggle.get_attribute("aria-expanded"), "false")
                self.assertEqual(
                    page.evaluate('() => getComputedStyle(document.querySelector("#mapControlReset")).transform'),
                    "none",
                )
                self.assertFalse(page.locator("#mapLineWeight").is_visible())
                collapsed_button_box = wait_for_map_toggle_top_right()
                map_toggle.click()
                page.wait_for_function('() => !document.querySelector("#mapFloatingControl")?.classList.contains("collapsed")')
                self.assertEqual(map_toggle.get_attribute("aria-expanded"), "true")
                expanded_again_button_box = wait_for_map_toggle_top_right()
                page.locator("#sidebarToggleBtn").click()
                page.wait_for_function('() => document.querySelector("#sidebarToggleBtn")?.getAttribute("aria-expanded") === "true"')
                wait_for_map_toggle_top_right()
                page.locator("#sidebarToggleBtn").click()
                page.wait_for_function('() => document.querySelector("#sidebarToggleBtn")?.getAttribute("aria-expanded") === "false"')
                wait_for_map_toggle_top_right()
                header_box = page.locator(".map-floating-header").bounding_box()
                self.assertIsNotNone(header_box)
                page.mouse.move(header_box["x"] + 12, header_box["y"] + 10)
                page.mouse.down()
                page.mouse.move(header_box["x"] + 12, header_box["y"] + 58, steps=8)
                page.mouse.up()
                page.wait_for_timeout(50)
                dragged_button_box = map_toggle.bounding_box()
                self.assertIsNotNone(dragged_button_box)
                self.assertGreater(dragged_button_box["y"], expected_top_right_button_box()["y"])
                map_toggle.click()
                page.wait_for_function('() => document.querySelector("#mapFloatingControl")?.classList.contains("collapsed")')
                wait_for_map_toggle_top_right()
                page.locator("#sidebarToggleBtn").click()
                page.wait_for_function('() => document.querySelector("#sidebarToggleBtn")?.getAttribute("aria-expanded") === "true"')
                wait_for_map_toggle_top_right()
                sidebar_resizer_box = page.locator("#sidebarResizer").bounding_box()
                self.assertIsNotNone(sidebar_resizer_box)
                sidebar_resizer_x = sidebar_resizer_box["x"] + sidebar_resizer_box["width"] / 2
                page.mouse.move(sidebar_resizer_x, sidebar_resizer_box["y"] + 100)
                page.mouse.down()
                page.mouse.move(sidebar_resizer_x + 80, sidebar_resizer_box["y"] + 100, steps=8)
                page.mouse.up()
                wait_for_map_toggle_top_right()
                page.locator("#sidebarToggleBtn").click()
                page.wait_for_function('() => document.querySelector("#sidebarToggleBtn")?.getAttribute("aria-expanded") === "false"')
                wait_for_map_toggle_top_right()
                map_toggle.click()
                page.wait_for_function('() => !document.querySelector("#mapFloatingControl")?.classList.contains("collapsed")')
                wait_for_map_toggle_top_right()
                self.assertTrue(page.locator('#mapLevelTiles input[name="mapLevel"][value="area"]').is_checked())
                self.assertFalse(page.locator("#mapLineWeightControl").is_hidden())
                self.assertFalse(page.locator("#mapLineWeight").is_disabled())
                self.assertTrue(page.locator("#mapDotSizeControl").is_hidden())
                self.assertTrue(page.locator("#mapDotSize").is_disabled())
                self.assertEqual(page.locator("#mapLineWeightControl > span:first-child").text_content().strip(), "Width")
                self.assertEqual(page.locator("#mapHotspots").get_attribute("min"), "-9")
                self.assertEqual(page.locator("#mapHotspots").get_attribute("max"), "9")
                self.assertEqual(page.locator("#mapHotspots").get_attribute("step"), "1")
                self.assertEqual(page.locator("#mapHotspotsValue").text_content().strip(), "All")
                self.assertEqual(page.locator("#mapHotspotsMinLabel").text_content().strip(), "Low")
                self.assertEqual(page.locator("#mapHotspotsMaxLabel").text_content().strip(), "High")
                self.assertTrue(page.locator("#mapHotspots").evaluate("input => input.classList.contains('map-slider-thumb-centered')"))
                page.evaluate(
                    """
                    () => {
                        const input = document.querySelector("#mapHotspots");
                        input.value = "-1";
                        input.dispatchEvent(new Event("input", { bubbles: true }));
                    }
                    """
                )
                page.wait_for_function('() => document.querySelector("#mapHotspotsValue")?.textContent === "B90"')
                self.assertFalse(page.locator("#mapHotspots").evaluate("input => input.classList.contains('map-slider-thumb-centered')"))
                page.evaluate(
                    """
                    () => {
                        const input = document.querySelector("#mapHotspots");
                        input.value = "1";
                        input.dispatchEvent(new Event("input", { bubbles: true }));
                    }
                    """
                )
                page.wait_for_function('() => document.querySelector("#mapHotspotsValue")?.textContent === "T90"')
                self.assertFalse(page.locator("#mapHotspots").evaluate("input => input.classList.contains('map-slider-thumb-centered')"))
                self.assertNotIn("100", page.locator("#mapHotspotsValue").text_content())
                page.evaluate(
                    """
                    () => {
                        const input = document.querySelector("#mapHotspots");
                        input.value = "0";
                        input.dispatchEvent(new Event("input", { bubbles: true }));
                    }
                    """
                )
                page.wait_for_function('() => document.querySelector("#mapHotspotsValue")?.textContent === "All"')
                self.assertTrue(page.locator("#mapHotspots").evaluate("input => input.classList.contains('map-slider-thumb-centered')"))
                self.assertFalse(page.locator("#mapLabelControl").is_hidden())
                self.assertFalse(page.locator("#mapLabelSize").is_disabled())
                self.assertEqual(page.locator("#mapLabelSize").get_attribute("max"), "10")
                label_states = page.evaluate(
                    """
                    () => {
                        const input = document.querySelector("#mapLabelSize");
                        let labelText = "";
                        const states = [];
                        for (let size = 1; size <= 10; size += 1) {
                            input.value = String(size);
                            input.dispatchEvent(new Event("input", { bubbles: true }));
                            const labels = [...document.querySelectorAll("#ukMap .map-label")];
                            const label = labelText
                                ? labels.find((node) => node.textContent === labelText)
                                : labels[0];
                            if (!label) return null;
                            labelText = label.textContent;
                            const rect = label.getBoundingClientRect();
                            states.push({
                                size,
                                fontSize: getComputedStyle(label).fontSize,
                                centerX: rect.left + rect.width / 2,
                                centerY: rect.top + rect.height / 2,
                                width: rect.width,
                                height: rect.height,
                            });
                        }
                        input.value = "0";
                        input.dispatchEvent(new Event("input", { bubbles: true }));
                        return { states, hiddenCount: document.querySelectorAll("#ukMap .map-label").length };
                    }
                    """
                )
                self.assertIsNotNone(label_states)
                self.assertEqual(len(label_states["states"]), 10)
                self.assertEqual(label_states["states"][0]["fontSize"], "6px")
                self.assertEqual(label_states["states"][-1]["fontSize"], "20px")
                self.assertEqual(label_states["hiddenCount"], 0)
                first_label_state = label_states["states"][0]
                for label_state in label_states["states"][1:]:
                    self.assertLessEqual(abs(label_state["centerX"] - first_label_state["centerX"]), 1)
                    self.assertLessEqual(abs(label_state["centerY"] - first_label_state["centerY"]), 1)
                self.assertGreater(label_states["states"][-1]["width"], first_label_state["width"])
                self.assertGreater(label_states["states"][-1]["height"], first_label_state["height"])
                page.locator("#profileTool").click()
                page.locator("#profileTool.active").wait_for(timeout=10_000)
                page.locator("#profileWrap:not(.hidden)").wait_for(timeout=10_000)
                page.locator("#ukMapTool").click()
                page.locator("#ukMap:not(.hidden)").wait_for(timeout=10_000)
                page.evaluate(
                  """
                    () => {
                        const map = document.querySelector("#ukMap")?._lucidumMap;
                        map.setView([51.5074, -0.1278], 9, { animate: false });
                    }
                    """
                )
                stable_map_view = map_view()
                page.set_viewport_size({"width": 1000, "height": 720})
                wait_for_map_view(stable_map_view)
                page.locator("#themeBtn").click()
                page.wait_for_function("() => document.querySelector('#ukMap')?.classList.contains('map-bg-dark')")
                wait_for_map_view(stable_map_view)
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

                page.locator("#histogramTool").click()
                page.locator("#histogramWrap:not(.hidden)").wait_for(timeout=10_000)
                page.locator("#histogramChart canvas").wait_for(timeout=10_000)
                page.locator("#histogramStatsGrid .tabulator-row").first.wait_for(timeout=10_000)
                histogram_grid_border = page.evaluate(
                    """
                    () => {
                        const gridStyle = getComputedStyle(document.querySelector("#histogramStatsGrid"));
                        const headerStyle = getComputedStyle(
                            document.querySelector('#histogramStatsGrid .tabulator-col[tabulator-field="statistic"]')
                        );
                        const firstCellStyle = getComputedStyle(
                            document.querySelector('#histogramStatsGrid .tabulator-row .tabulator-cell[tabulator-field="statistic"]')
                        );
                        return {
                            firstCellLeft: firstCellStyle.borderLeftWidth,
                            grid: [
                                gridStyle.borderTopWidth,
                                gridStyle.borderRightWidth,
                                gridStyle.borderBottomWidth,
                                gridStyle.borderLeftWidth,
                            ],
                            headerLeft: headerStyle.borderLeftWidth,
                        };
                    }
                    """
                )
                self.assertEqual(histogram_grid_border["grid"], ["1px", "0px", "0px", "0px"])
                self.assertEqual(histogram_grid_border["headerLeft"], "1px")
                self.assertEqual(histogram_grid_border["firstCellLeft"], "1px")
                histogram_reference_colors = page.evaluate(
                    """
                    () => {
                        const normalizeColor = (color) => {
                            if (!color) return "";
                            const probe = document.createElement("span");
                            probe.style.color = color;
                            document.body.appendChild(probe);
                            const normalized = getComputedStyle(probe).color;
                            probe.remove();
                            return normalized;
                        };
                        const chart = echarts.getInstanceByDom(document.querySelector("#histogramChart"));
                        const series = chart?.getOption?.().series || [];
                        const chartColor = (name) => normalizeColor(
                            series.find((item) => item.name === name)?.markLine?.lineStyle?.color || ""
                        );
                        const rowColors = (name) => {
                            const rows = Array.from(document.querySelectorAll("#histogramStatsGrid .tabulator-row"));
                            const row = rows.find((candidate) =>
                                candidate.querySelector('.tabulator-cell[tabulator-field="statistic"]')?.textContent.trim() === name
                            );
                            const statisticCell = row?.querySelector('.tabulator-cell[tabulator-field="statistic"]');
                            const valueCell = row?.querySelector('.tabulator-cell[tabulator-field="value"]');
                            return {
                                statistic: statisticCell ? getComputedStyle(statisticCell).color : "",
                                value: valueCell ? getComputedStyle(valueCell).color : "",
                            };
                        };
                        return {
                            meanChart: chartColor("Mean"),
                            medianChart: chartColor("Median"),
                            meanRow: rowColors("Mean"),
                            medianRow: rowColors("Median"),
                        };
                    }
                    """
                )
                self.assertEqual(histogram_reference_colors["meanChart"], "rgb(209, 63, 63)")
                self.assertEqual(histogram_reference_colors["medianChart"], "rgb(31, 122, 140)")
                self.assertEqual(histogram_reference_colors["meanRow"]["statistic"], histogram_reference_colors["meanChart"])
                self.assertEqual(histogram_reference_colors["meanRow"]["value"], histogram_reference_colors["meanChart"])
                self.assertEqual(histogram_reference_colors["medianRow"]["statistic"], histogram_reference_colors["medianChart"])
                self.assertEqual(histogram_reference_colors["medianRow"]["value"], histogram_reference_colors["medianChart"])
                page.wait_for_function(
                    """
                    () => {
                        const text = document.querySelector("#actionTimingMonitor")?.textContent || "";
                        return /^DuckDB: \\d+(?:ns|us|ms), JSON: \\d+ms, Histogram render: \\d+(?:ns|us|ms), Total: \\d+ms$/.test(text);
                    }
                    """
                )
                page.wait_for_function(
                    """
                    () => {
                        const chart = echarts.getInstanceByDom(document.querySelector("#histogramChart"));
                        const zoom = chart?.getOption?.().dataZoom || [];
                        return zoom.some((item) => item.type === "slider" && item.xAxisIndex === 0);
                    }
                    """,
                    timeout=10_000,
                )

                with page.expect_response(lambda response: response.url.endswith("/api/histogram/chart") and response.status == 200, timeout=10_000):
                    page.locator('.segmented[data-control="histogramDistribution"] button[data-value="cumulative"]').click()
                page.wait_for_function(
                    """
                    () => {
                        const chart = echarts.getInstanceByDom(document.querySelector("#histogramChart"));
                        if (!chart) return false;
                        const series = chart.getOption().series?.[0];
                        const values = (series?.data || [])
                          .map((item) => Number(item.row?.height ?? item.value?.[1]))
                          .filter(Number.isFinite);
                        const maxHeight = Math.max(...values);
                        const yExtent = chart.getModel()?.getComponent("yAxis")?.axis?.scale?.getExtent?.();
                        return Number.isFinite(maxHeight)
                          && maxHeight > 10
                          && Array.isArray(yExtent)
                          && Number(yExtent[1]) >= maxHeight;
                    }
                    """,
                    timeout=10_000,
                )

                with page.expect_response(lambda response: response.url.endswith("/api/histogram/chart") and response.status == 200, timeout=10_000):
                    page.evaluate(
                        """
                        () => {
                            document.querySelector("#actualNumerator").value = "vehicle_age";
                            document.querySelector("#denominator").value = "__none__";
                            document.querySelector("#denominator").dispatchEvent(new Event("change", { bubbles: true }));
                        }
                        """
                    )
                page.wait_for_function(
                    """
                    () => {
                        const chart = echarts.getInstanceByDom(document.querySelector("#histogramChart"));
                        const option = chart?.getOption?.();
                        const axis = Array.isArray(option?.xAxis) ? option.xAxis[0] : option?.xAxis;
                        const formatter = axis?.axisLabel?.formatter;
                        return axis?.minInterval === 1
                          && formatter?.(1) === "1"
                          && formatter?.(1.5) === "";
                    }
                    """,
                    timeout=10_000,
                )

                with page.expect_request(lambda request: request.url.endswith("/api/histogram/chart"), timeout=10_000) as histogram_filter_request_info:
                    with page.expect_response(lambda response: response.url.endswith("/api/histogram/chart") and response.status == 200, timeout=10_000):
                        page.evaluate(
                            """
                            () => {
                                document.querySelector("#filterInput").value = "vehicle_age >= 0";
                                document.querySelector("#filterApplyBtn").click();
                            }
                            """
                        )
                histogram_filter_payload = json.loads(histogram_filter_request_info.value.post_data or "{}")
                self.assertEqual(histogram_filter_payload["filter"], "vehicle_age >= 0")
                assert_filter_label_badge("#histogramFilter", "histogram-filter--applied", True)
                assert_filter_badge_clear("#histogramFilterClearBtn", "#histogramFilterText", True)
                page.set_viewport_size({"width": 420, "height": 800})
                page.wait_for_function(
                    """
                    () => window.matchMedia("(max-width: 900px)").matches
                      && !document.querySelector("#histogramWrap")?.classList.contains("hidden")
                      && document.querySelector("#histogramFilter")?.classList.contains("histogram-filter--applied")
                    """,
                    timeout=10_000,
                )
                histogram_mobile_filter_layout = page.evaluate(
                    """
                    () => {
                      const group = document.querySelector("#histogramGroupMeta").getBoundingClientRect();
                      const filter = document.querySelector("#histogramFilter").getBoundingClientRect();
                      const header = document.querySelector("#histogramStatsGrid .tabulator-header").getBoundingClientRect();
                      const statsPanel = document.querySelector(".histogram-stats-panel").getBoundingClientRect();
                      return {
                        filterBottom: filter.bottom,
                        groupBottom: group.bottom,
                        headerTop: header.top,
                        statsPanelTop: statsPanel.top,
                      };
                    }
                    """
                )
                self.assertGreaterEqual(
                    histogram_mobile_filter_layout["headerTop"],
                    max(
                        histogram_mobile_filter_layout["filterBottom"],
                        histogram_mobile_filter_layout["groupBottom"],
                    ) + 4,
                    histogram_mobile_filter_layout,
                )
                self.assertGreater(
                    histogram_mobile_filter_layout["headerTop"],
                    histogram_mobile_filter_layout["statsPanelTop"] + 18,
                    histogram_mobile_filter_layout,
                )
                page.set_viewport_size({"width": 1280, "height": 800})
                page.wait_for_function(
                    """
                    () => !window.matchMedia("(max-width: 900px)").matches
                      && !document.querySelector("#histogramWrap")?.classList.contains("hidden")
                    """,
                    timeout=10_000,
                )

                with page.expect_request(lambda request: request.url.endswith("/api/histogram/chart"), timeout=10_000) as histogram_clear_filter_request_info:
                    with page.expect_response(lambda response: response.url.endswith("/api/histogram/chart") and response.status == 200, timeout=10_000):
                        page.locator("#histogramFilterClearBtn").click()
                histogram_clear_filter_payload = json.loads(histogram_clear_filter_request_info.value.post_data or "{}")
                self.assertEqual(histogram_clear_filter_payload["filter"], "")
                assert_filter_label_badge("#histogramFilter", "histogram-filter--applied", False)
                assert_filter_badge_clear("#histogramFilterClearBtn", "#histogramFilterText", False)

                with page.expect_request(lambda request: request.url.endswith("/api/histogram/chart"), timeout=10_000) as histogram_refilter_request_info:
                    with page.expect_response(lambda response: response.url.endswith("/api/histogram/chart") and response.status == 200, timeout=10_000):
                        page.evaluate(
                            """
                            () => {
                                document.querySelector("#filterInput").value = "vehicle_age >= 0";
                                document.querySelector("#filterApplyBtn").click();
                            }
                            """
                        )
                histogram_refilter_payload = json.loads(histogram_refilter_request_info.value.post_data or "{}")
                self.assertEqual(histogram_refilter_payload["filter"], "vehicle_age >= 0")
                assert_filter_label_badge("#histogramFilter", "histogram-filter--applied", True)

                with page.expect_request(lambda request: request.url.endswith("/api/histogram/chart"), timeout=10_000) as histogram_metric_request_info:
                    with page.expect_response(lambda response: response.url.endswith("/api/histogram/chart") and response.status == 200, timeout=10_000):
                        page.evaluate(
                            """
                            () => {
                                document.querySelector("#actualNumerator").value = "vehicle_age";
                                document.querySelector("#denominator").value = "price";
                                document.querySelector("#denominator").dispatchEvent(new Event("change", { bubbles: true }));
                            }
                            """
                        )
                histogram_metric_payload = json.loads(histogram_metric_request_info.value.post_data or "{}")
                self.assertEqual(histogram_metric_payload["actual"], "vehicle_age")
                self.assertEqual(histogram_metric_payload["denominator"], "price")
                self.assertEqual(histogram_metric_payload["filter"], "vehicle_age >= 0")
                page.locator("#histogramStatsGrid .tabulator-row").first.wait_for(timeout=10_000)

                with page.expect_request(lambda request: request.url.endswith("/api/histogram/chart"), timeout=10_000) as histogram_bins_request_info:
                    with page.expect_response(lambda response: response.url.endswith("/api/histogram/chart") and response.status == 200, timeout=10_000):
                        page.locator("#histogramBins").fill("5")
                histogram_bins_payload = json.loads(histogram_bins_request_info.value.post_data or "{}")
                self.assertEqual(histogram_bins_payload["bins"], "5")
                page.locator("#histogramStatsGrid .tabulator-row").first.wait_for(timeout=10_000)

                page.evaluate(
                    """
                    () => {
                      ["#favouritesCollapseBtn", "#kpiCollapseBtn", "#gbmModelCollapseBtn", "#glmModelCollapseBtn", "#filterCollapseBtn"]
                        .forEach((selector) => {
                          const button = document.querySelector(selector);
                          if (button?.getAttribute("aria-expanded") === "true") button.click();
                        });
                    }
                    """
                )
                page.wait_for_function(
                    """
                    () => ["#favouritesCollapseBtn", "#kpiCollapseBtn", "#gbmModelCollapseBtn", "#glmModelCollapseBtn", "#filterCollapseBtn"]
                      .every((selector) => {
                        const button = document.querySelector(selector);
                        return !button || button.getAttribute("aria-expanded") === "false";
                      })
                    """,
                    timeout=10_000,
                )
                page.locator("#sidebarToggleBtn").click()
                self.assertEqual(page.locator("#sidebarToggleBtn").get_attribute("aria-expanded"), "true")
                self.assertTrue(page.locator(".sidebar-metric-section").is_visible())
                self.assertTrue(page.locator(".sidebar-favourites-section").is_visible())
                self.assertTrue(page.locator(".sidebar-filter-section").is_visible())
                page.locator("#sidebarToggleBtn").click()
                self.assertEqual(page.locator("#sidebarToggleBtn").get_attribute("aria-expanded"), "false")
                self.assertFalse(page.locator(".sidebar-metric-section").is_visible())
                self.assertFalse(page.locator("#actualNumerator").is_visible())
                self.assertFalse(page.locator("#denominator").is_visible())
                page.locator("#sidebarToggleBtn").click()
                self.assertEqual(page.locator("#sidebarToggleBtn").get_attribute("aria-expanded"), "true")
                self.assertTrue(page.locator(".sidebar-metric-section").is_visible())

                page.locator("#ukMapTool").click()
                page.locator("#ukMap:not(.hidden)").wait_for(timeout=10_000)
                page.evaluate(
                    """
                    () => new Promise((resolve) => {
                        requestAnimationFrame(() => requestAnimationFrame(resolve));
                    })
                    """
                )
                stable_map_view = map_view()
                wait_for_map_view(stable_map_view)

                with page.expect_response(lambda response: response.url.endswith("/api/uk-map/summary") and response.status == 200, timeout=10_000):
                    page.locator("#reloadBtn").click()
                page.locator("#ukMap:not(.hidden)").wait_for(timeout=10_000)
                wait_for_map_view(stable_map_view)

                with page.expect_response(lambda response: response.url.endswith("/api/uk-map/summary") and response.status == 200, timeout=10_000):
                    page.evaluate(
                        """
                        () => {
                            const select = document.querySelector("#actualNumerator");
                            const next = [...select.options].find((option) => option.value && option.value !== select.value);
                            if (!next) throw new Error("No alternate Actual option available");
                            select.value = next.value;
                            select.dispatchEvent(new Event("change", { bubbles: true }));
                        }
                        """
                    )
                wait_for_map_view(stable_map_view)

                with page.expect_response(lambda response: response.url.endswith("/api/uk-map/summary") and response.status == 200, timeout=10_000):
                    page.evaluate(
                        """
                        () => {
                            document.querySelector("#filterInput").value = "vehicle_age >= 1";
                            document.querySelector("#filterApplyBtn").click();
                        }
                        """
                    )
                wait_for_map_view(stable_map_view)

                with page.expect_response(lambda response: response.url.endswith("/api/uk-map/summary") and response.status == 200, timeout=10_000):
                    page.locator('#mapLevelTiles input[name="mapLevel"][value="sector"]').check()
                page.wait_for_function('() => document.querySelector("#mapGroupMeta")?.textContent.includes("sectors matched")')
                self.assertTrue(page.locator('#mapLevelTiles input[name="mapLevel"][value="sector"]').is_checked())
                self.assertFalse(page.locator("#mapLineWeightControl").is_hidden())
                self.assertFalse(page.locator("#mapLineWeight").is_disabled())
                self.assertTrue(page.locator("#mapDotSizeControl").is_hidden())
                self.assertTrue(page.locator("#mapDotSize").is_disabled())
                self.assertEqual(page.locator("#mapLineWeightControl > span:first-child").text_content().strip(), "Width")
                self.assertEqual(page.locator("#mapHotspotsMinLabel").text_content().strip(), "Low")
                self.assertEqual(page.locator("#mapHotspotsMaxLabel").text_content().strip(), "High")
                wait_for_map_view(stable_map_view)

                self.assertTrue(page.locator("#mapLabelControl").is_hidden())
                self.assertTrue(page.locator("#mapLabelSize").is_disabled())
                self.assertFalse(page.locator("#mapSmoothingControl").is_hidden())
                self.assertFalse(page.locator("#mapSmoothing").is_disabled())
                with page.expect_response(lambda response: response.url.endswith("/api/uk-map/summary") and response.status == 200, timeout=10_000):
                    page.evaluate(
                        """
                        () => {
                            const input = document.querySelector("#mapSmoothing");
                            input.value = "2";
                            input.dispatchEvent(new Event("input", { bubbles: true }));
                        }
                        """
                    )
                page.wait_for_function('() => document.querySelector("#mapSmoothingValue")?.textContent === "N2"')
                wait_for_map_view(stable_map_view)

                with page.expect_response(lambda response: response.url.endswith("/api/uk-map/summary") and response.status == 200, timeout=10_000):
                    page.locator('#mapLevelTiles input[name="mapLevel"][value="unit"]').check()
                page.wait_for_function('() => document.querySelector("#mapGroupMeta")?.textContent.includes("units plotted")')
                self.assertTrue(page.locator('#mapLevelTiles input[name="mapLevel"][value="unit"]').is_checked())
                self.assertTrue(page.locator("#mapLineWeightControl").is_hidden())
                self.assertTrue(page.locator("#mapLineWeight").is_disabled())
                self.assertFalse(page.locator("#mapDotSizeControl").is_hidden())
                self.assertFalse(page.locator("#mapDotSize").is_disabled())
                self.assertEqual(page.locator("#mapDotSizeControl > span:first-child").text_content().strip(), "Dot size")
                self.assertEqual(page.locator("#mapHotspotsMinLabel").text_content().strip(), "Low")
                self.assertEqual(page.locator("#mapHotspotsMaxLabel").text_content().strip(), "High")
                self.assertIn("unit-mode", page.locator("#mapSliderGrid").get_attribute("class") or "")
                wait_for_map_view(stable_map_view)

                self.assertTrue(page.locator("#mapLabelControl").is_hidden())
                self.assertTrue(page.locator("#mapLabelSize").is_disabled())
                self.assertTrue(page.locator("#mapSmoothingControl").is_hidden())
                self.assertTrue(page.locator("#mapSmoothing").is_disabled())
                page.locator("#ukMap .leaflet-unit-point-layer").wait_for(timeout=10_000)
                page.evaluate(
                    """
                    () => {
                        const input = document.querySelector("#mapDotSize");
                        input.value = "10";
                        input.dispatchEvent(new Event("input", { bubbles: true }));
                    }
                    """
                )
                page.wait_for_function('() => document.querySelector("#mapDotSizeValue")?.textContent === "10"')
                large_dot_pixels = unit_point_alpha_pixels()
                self.assertGreater(large_dot_pixels, 0)
                page.evaluate(
                    """
                    () => {
                        const input = document.querySelector("#mapOpacity");
                        input.value = "0";
                        input.dispatchEvent(new Event("input", { bubbles: true }));
                    }
                    """
                )
                page.wait_for_function('() => document.querySelector("#mapOpacityValue")?.textContent === "0"')
                self.assertEqual(unit_point_alpha_pixels(), 0)
                page.evaluate(
                    """
                    () => {
                        const input = document.querySelector("#mapOpacity");
                        input.value = "10";
                        input.dispatchEvent(new Event("input", { bubbles: true }));
                    }
                    """
                )
                page.wait_for_function('() => document.querySelector("#mapOpacityValue")?.textContent === "10"')
                restored_dot_pixels = unit_point_alpha_pixels()
                self.assertGreater(restored_dot_pixels, 0)
                page.evaluate(
                    """
                    () => {
                        const input = document.querySelector("#mapDotSize");
                        input.value = "1";
                        input.dispatchEvent(new Event("input", { bubbles: true }));
                    }
                    """
                )
                page.wait_for_function('() => document.querySelector("#mapDotSizeValue")?.textContent === "1"')
                small_dot_pixels = unit_point_alpha_pixels()
                self.assertGreater(small_dot_pixels, 0)
                self.assertGreater(large_dot_pixels, small_dot_pixels)
                wait_for_map_view(stable_map_view)
                assert_filter_label_badge("#mapControlFilter", "map-filter--applied", True)
                assert_filter_badge_clear("#mapControlFilterClearBtn", "#mapControlFilterText", True)
                with page.expect_response(lambda response: response.url.endswith("/api/uk-map/summary") and response.status == 200, timeout=10_000):
                    page.locator("#mapControlFilterClearBtn").click()
                page.wait_for_function(
                    """
                    () => document.querySelector("#filterInput")?.value === ""
                      && document.querySelector("#mapControlFilterText")?.textContent.trim() === "no filter"
                      && document.querySelector("#mapControlFilterClearBtn")?.hidden
                    """,
                    timeout=10_000,
                )

                request_counts = {
                    "profile": profile_requests,
                    "profile_detail": profile_detail_requests,
                    "chart": chart_requests,
                    "histogram": histogram_requests,
                    "map": map_requests,
                }
                self.assertEqual(page_errors, [])
                self.assertEqual(profile_requests, 6, request_counts)
                self.assertEqual(profile_detail_requests, 7, request_counts)
                self.assertEqual(chart_requests, 4, request_counts)
                self.assertEqual(histogram_requests, 8, request_counts)
                self.assertEqual(map_requests, 10, request_counts)
            finally:
                browser.close()

    def exercise_gbm_tool(self, base_url: str) -> None:
        assert sync_playwright is not None
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch()
            page = browser.new_page(viewport={"width": 1280, "height": 800})
            page_errors: list[str] = []
            page.on("pageerror", lambda error: page_errors.append(str(error)))

            def assert_feature_heading_matches_checked(expected_count: int | None = None) -> None:
                page.wait_for_function(
                    """
                    (expectedCount) => {
                      const title = document.querySelector("#gbmFeatureSectionTitle");
                      const checked = document.querySelectorAll("#gbmFeatureGrid .gbm-use-checkbox:checked").length;
                      const count = expectedCount === null ? checked : expectedCount;
                      return title?.textContent.trim() === `Features (${count})`;
                    }
                    """,
                    arg=expected_count,
                    timeout=10_000,
                )

            def feature_scenario_state() -> dict[str, object]:
                return page.evaluate(
                    """
                    () => {
                      const root = document.querySelector("#gbmFeatureScenarioDropdown");
                      return {
                        value: root?.dataset.gbmSelectedFeatureScenario || "",
                        rows: [...document.querySelectorAll("#gbmFeatureScenarioMenu .gbm-feature-scenario-row")]
                          .map((row) => row.textContent.trim()),
                        activeRows: [...document.querySelectorAll("#gbmFeatureScenarioMenu .gbm-feature-scenario-row.active")]
                          .map((row) => row.textContent.trim()),
                        menuHidden: document.querySelector("#gbmFeatureScenarioMenu")?.classList.contains("hidden") ?? true,
                        title: document.querySelector("#gbmFeatureScenarioButton")?.getAttribute("title") || "",
                      };
                    }
                    """
                )

            def choose_feature_scenario(name: str) -> None:
                page.locator("#gbmFeatureScenarioButton").click()
                page.locator(f'[data-gbm-feature-scenario="{name}"]').click()

            try:
                page.goto(f"{base_url}/?tool=gbm", wait_until="domcontentloaded")
                page.get_by_text("Features and parameters").wait_for(timeout=10_000)
                page.locator("#gbmFeatureGrid").wait_for(timeout=10_000)
                page.get_by_text("Train GBM").wait_for(timeout=10_000)
                page.locator("#gbmFeatureMetricToggle").wait_for(timeout=10_000)
                page.get_by_text("Grouping").first.wait_for(timeout=10_000)
                page.get_by_text("SHAP rows").wait_for(timeout=10_000)
                page.get_by_text("Training mode").wait_for(timeout=10_000)
                page.wait_for_function(
                    """
                    () => document.querySelector("#gbmFeatureGrid .tabulator-cell[tabulator-field='mean_abs_shap']")
                      ?.textContent.trim() === "0.2330"
                    """,
                    timeout=10_000,
                )
                default_metric_state = page.evaluate(
                    """
                    () => {
                      const headers = [...document.querySelectorAll("#gbmFeatureGrid .tabulator-col-title")]
                        .map((node) => node.textContent.trim()).filter(Boolean);
                      const rows = [...document.querySelectorAll("#gbmFeatureGrid .tabulator-row")];
                      const firstRow = rows[0];
                      const ageRow = rows.find((row) => row.textContent.includes("Age"));
                      return {
                        headers,
                        checkedMetric: document.querySelector("input[name='gbmFeatureMetric']:checked")?.value || "",
                        metricLabels: [...document.querySelectorAll("#gbmFeatureMetricToggle .gbm-feature-metric-option span")].map((node) => node.textContent.trim()),
                        firstRowText: firstRow?.textContent || "",
                        ageShap: ageRow?.querySelector(".tabulator-cell[tabulator-field='mean_abs_shap']")?.textContent.trim() || "",
                      };
                    }
                    """
                )
                self.assertEqual(default_metric_state["checkedMetric"], "shap")
                self.assertEqual(default_metric_state["metricLabels"], ["Gain", "SHAP"])
                self.assertIn("SHAP", default_metric_state["headers"])
                self.assertNotIn("Gain", default_metric_state["headers"])
                self.assertIn("lat", default_metric_state["firstRowText"])
                self.assertEqual(default_metric_state["ageShap"], "0.1830")
                page.locator("#gbmFeatureGrid .tabulator-row", has_text="Age").locator(".tabulator-cell[tabulator-field='name']").click(button="right")
                page.locator("#gbmFeatureContextMenu:not([hidden])").wait_for(timeout=10_000)
                normal_context_labels = page.locator("#gbmFeatureContextMenu [role='menuitem']").evaluate_all(
                    "(items) => items.map((item) => item.textContent.trim())"
                )
                self.assertEqual(normal_context_labels, ["Toggle interaction constraint", "Go to Line and Bar", "Go to SHAP", "Go to Stacked SHAP"])
                self.assertEqual(page.locator("#gbmFeatureContextMenu [role='separator']").count(), 1)
                page.locator("#gbmFeatureContextMenu [role='menuitem']", has_text="Toggle interaction constraint").click()
                page.wait_for_function(
                    """
                    () => {
                      const row = [...document.querySelectorAll("#gbmFeatureGrid .tabulator-row")]
                        .find((item) => item.textContent.includes("Age"));
                      return (row?.querySelector(".tabulator-cell[tabulator-field='name']")?.textContent || "").includes("\\uD83D\\uDD12");
                    }
                    """,
                    timeout=10_000,
                )
                page.locator("#gbmFeatureGrid .tabulator-row", has_text="Age").locator(".tabulator-cell[tabulator-field='mean_abs_shap']").click(button="right")
                page.locator("#gbmFeatureContextMenu [role='menuitem']", has_text="Copy importance value").wait_for(timeout=10_000)
                page.keyboard.press("Escape")
                page.locator("#gbmFeatureGrid .tabulator-row", has_text="Age").locator(".tabulator-cell[tabulator-field='monotonicity']").click(button="right")
                monotonicity_context_labels = page.locator("#gbmFeatureContextMenu [role='menuitem']").evaluate_all(
                    "(items) => items.map((item) => item.textContent.trim())"
                )
                self.assertEqual(monotonicity_context_labels, ["Clear all monotonicities"])
                page.keyboard.press("Escape")
                page.locator("#gbmFeatureGrid .tabulator-row", has_text="Age").locator(".tabulator-cell[tabulator-field='name']").click(button="right")
                with page.expect_request(lambda request: request.url.endswith("/api/chart"), timeout=10_000) as gbm_context_chart_info:
                    page.locator("#gbmFeatureContextMenu [role='menuitem']", has_text="Go to Line and Bar").click()
                gbm_context_chart_body = json.loads(gbm_context_chart_info.value.post_data or "{}")
                page.locator("#lineBarTool.active").wait_for(timeout=10_000)
                if page.locator("#lineBarSideControlsToggleBtn").get_attribute("aria-expanded") == "false":
                    page.locator("#lineBarSideControlsToggleBtn").click()
                    page.wait_for_function(
                        """
                        () => document.querySelector("#lineBarSideControlsToggleBtn")?.getAttribute("aria-expanded") === "true"
                          && getComputedStyle(document.querySelector("#chartSideControls")).display !== "none"
                        """,
                        timeout=10_000,
                    )
                page.locator("#featureList .feature.active", has_text="Age").wait_for(timeout=10_000)
                self.assertEqual(gbm_context_chart_body["x"], "Age")
                self.assertEqual(gbm_context_chart_body["responses"][1]["numerator"], "gbm_prediction")
                self.assertEqual(gbm_context_chart_body["responses"][1]["source"], "gbm:browser-smoke-model:predictions")
                page.locator("#gbmTool").click()
                page.locator("#gbmFeatureGrid").wait_for(timeout=10_000)
                page.locator("#gbmFeatureGrid .tabulator-row", has_text="Age").locator(".tabulator-cell[tabulator-field='name']").click(button="right")
                with page.expect_response(lambda response: "/api/gbm/models/" in response.url and "/shap/plot" in response.url and response.request.method == "POST", timeout=10_000):
                    page.locator("#gbmFeatureContextMenu [role='menuitem']", has_text="Go to SHAP").click()
                page.wait_for_function(
                    """
                    () => document.querySelector("#gbmShapFeatureList1 .feature.active")?.textContent.includes("Age")
                      && document.querySelector("#gbmShapFeatureList2 .feature.active")?.textContent.includes("None")
                    """,
                    timeout=10_000,
                )
                page.get_by_role("button", name="Features and parameters").click()
                page.locator("#gbmFeatureGrid").wait_for(timeout=10_000)
                page.locator("#gbmFeatureGrid .tabulator-row", has_text="Age").locator(".tabulator-cell[tabulator-field='name']").click(button="right")
                with page.expect_response(lambda response: "/api/gbm/models/" in response.url and "/shap/stacked" in response.url and response.request.method == "POST", timeout=10_000):
                    page.locator("#gbmFeatureContextMenu [role='menuitem']", has_text="Go to Stacked SHAP").click()
                page.wait_for_function(
                    """
                    () => document.querySelector("#gbmStackedShapFeatureList .feature.active")?.textContent.includes("Age")
                    """,
                    timeout=10_000,
                )
                page.get_by_role("button", name="Features and parameters").click()
                page.locator("#gbmFeatureGrid").wait_for(timeout=10_000)
                page.locator("#gbmFeatureGrid .tabulator-row", has_text="PostcodeArea").locator(".tabulator-cell[tabulator-field='name']").click(button="right")
                page.locator("#gbmFeatureContextMenu:not([hidden])").wait_for(timeout=10_000)
                no_shap_context_labels = page.locator("#gbmFeatureContextMenu [role='menuitem']").evaluate_all(
                    "(items) => items.map((item) => item.textContent.trim())"
                )
                self.assertEqual(no_shap_context_labels, ["Toggle interaction constraint", "Go to Line and Bar"])
                page.keyboard.press("Escape")
                page.wait_for_function('() => document.querySelector("#gbmFeatureContextMenu")?.hidden === true')
                page.get_by_role("button", name="SHAP", exact=True).click()
                with page.expect_response(lambda response: "/api/gbm/models/" in response.url and "/shap/plot" in response.url and response.request.method == "POST", timeout=10_000):
                    page.locator("#gbmShapFeatureList1 .feature", has_text="lat").click()
                page.wait_for_function(
                    """
                    () => document.querySelector("#gbmShapFeatureList1 .feature.active")?.textContent.includes("lat")
                      && document.querySelector("#gbmShapFeatureList2 .feature.active")?.textContent.includes("None")
                    """,
                    timeout=10_000,
                )
                page.get_by_role("button", name="Stacked SHAP", exact=True).click()
                with page.expect_response(lambda response: "/api/gbm/models/" in response.url and "/shap/stacked" in response.url and response.request.method == "POST", timeout=10_000):
                    page.locator("#gbmStackedShapFeatureList .feature", has_text="lat").click()
                page.wait_for_function(
                    """
                    () => document.querySelector("#gbmStackedShapFeatureList .feature.active")?.textContent.includes("lat")
                    """,
                    timeout=10_000,
                )
                page.get_by_role("button", name="Features and parameters").click()
                page.locator("#gbmFeatureGrid").wait_for(timeout=10_000)
                page.locator("#gbmFeatureMetricToggle .gbm-feature-metric-option", has_text="Gain").click()
                page.wait_for_function(
                    """
                    () => {
                      const headers = [...document.querySelectorAll("#gbmFeatureGrid .tabulator-col-title")]
                        .map((node) => node.textContent.trim()).filter(Boolean);
                      return document.querySelector("input[name='gbmFeatureMetric']:checked")?.value === "gain"
                        && headers.includes("Gain")
                        && !headers.includes("SHAP");
                    }
                    """,
                    timeout=10_000,
                )
                assert_feature_heading_matches_checked(2)
                if page.locator("#gbmModelCollapseBtn").get_attribute("aria-expanded") != "true":
                    page.locator("#gbmModelCollapseBtn").click()
                page.wait_for_function(
                    '() => document.querySelector("#gbmModelCollapseBtn")?.getAttribute("aria-expanded") === "true"',
                    timeout=10_000,
                )
                page.locator('#gbmModelSelect [data-gbm-model-id="browser-smoke-model-2"]').click()
                page.wait_for_function(
                    """
                    () => {
                      const labels = [...document.querySelectorAll("#gbmFeatureMetricToggle .gbm-feature-metric-option span")]
                        .map((node) => node.textContent.trim());
                      return document.querySelector("#gbmModelSelectedMeta")?.textContent.includes("Second smoke model")
                        && document.querySelector("input[name='gbmFeatureMetric']:checked")?.value === "gain"
                        && labels.join("|") === "EBM Gain|Gain|SHAP";
                    }
                    """,
                    timeout=10_000,
                )
                if page.locator("#gbmModelCollapseBtn").get_attribute("aria-expanded") != "true":
                    page.locator("#gbmModelCollapseBtn").click()
                page.wait_for_function(
                    '() => document.querySelector("#gbmModelCollapseBtn")?.getAttribute("aria-expanded") === "true"',
                    timeout=10_000,
                )
                page.locator('#gbmModelSelect [data-gbm-model-id="browser-smoke-model"]').click()
                page.wait_for_function(
                    """
                    () => {
                      const labels = [...document.querySelectorAll("#gbmFeatureMetricToggle .gbm-feature-metric-option span")]
                        .map((node) => node.textContent.trim());
                      return document.querySelector("#gbmModelSelectedMeta")?.textContent.includes("Browser smoke model")
                        && document.querySelector("input[name='gbmFeatureMetric']:checked")?.value === "gain"
                        && labels.join("|") === "Gain|SHAP";
                    }
                    """,
                    timeout=10_000,
                )
                initial_scenario = feature_scenario_state()
                self.assertEqual(initial_scenario["value"], "")
                self.assertIn("old_scenario (1; trained; missing from spec)", initial_scenario["rows"])
                self.assertIn("old_scenario (1; trained; missing from spec)", initial_scenario["activeRows"])
                self.assertIn("feature scenario", str(initial_scenario["title"]).lower())
                initial_constraints = page.evaluate(
                    """
                    () => {
                      const rows = [...document.querySelectorAll(".gbm-interaction-constraint-row")].map((row) => row.textContent.trim());
                      const ageRow = [...document.querySelectorAll("#gbmFeatureGrid .tabulator-row")]
                        .find((row) => row.textContent.includes("Age"));
                      return {
                        button: document.querySelector("#gbmFeatureInteractionConstraintButton")?.textContent.trim() || "",
                        title: document.querySelector("#gbmFeatureInteractionConstraintButton")?.getAttribute("title") || "",
                        rows,
                        ageGrouping: ageRow?.querySelector(".tabulator-cell[tabulator-field='grouping']")?.textContent.trim() || "",
                      };
                    }
                    """
                )
                self.assertEqual(initial_constraints["button"], "Trained constraint groups (1)")
                self.assertIn("interact within selected groups", initial_constraints["title"])
                self.assertIn("OLD (trained; missing from spec)", initial_constraints["rows"])
                self.assertIn("\U0001f512", initial_constraints["ageGrouping"])
                page.locator("#gbmFeatureInteractionConstraintButton").click()
                page.wait_for_function(
                    """
                    () => {
                      const rows = [...document.querySelectorAll(".gbm-interaction-constraint-row")]
                        .map((row) => row.textContent.trim());
                      return rows.includes("DRIVER (1)") && rows.includes("VEHICLE (0)");
                    }
                    """,
                    timeout=10_000,
                )
                page.locator("#gbmFeatureSectionTitle").click()
                page.wait_for_function(
                    """
                    () => document.querySelector("#gbmFeatureInteractionConstraintMenu")?.classList.contains("hidden")
                      && document.querySelector("#gbmFeatureInteractionConstraintButton")?.getAttribute("aria-expanded") === "false"
                    """,
                    timeout=10_000,
                )
                page.locator("#gbmFeatureGrid .tabulator-row", has_text="Age").locator(".tabulator-cell[tabulator-field='grouping']").click(button="right")
                grouping_context_labels = page.locator("#gbmFeatureContextMenu [role='menuitem']").evaluate_all(
                    "(items) => items.map((item) => item.textContent.trim())"
                )
                self.assertEqual(grouping_context_labels, ["Toggle group interaction constraint"])
                page.locator("#gbmFeatureContextMenu [role='menuitem']", has_text="Toggle group interaction constraint").click()
                page.wait_for_function(
                    """
                    () => document.querySelector("#gbmFeatureInteractionConstraintButton")?.textContent.trim() === "Constraint groups (1)"
                      && [...document.querySelectorAll("#gbmFeatureGrid .tabulator-row")]
                        .find((row) => row.textContent.includes("Age"))
                        ?.querySelector(".tabulator-cell[tabulator-field='grouping']")
                        ?.textContent.includes("\\uD83D\\uDD12")
                    """,
                    timeout=10_000,
                )
                page.get_by_role("button", name="Model navigator").click()
                page.locator(".gbm-model-navigator").get_by_text("Browser smoke model", exact=False).wait_for(timeout=10_000)
                page.get_by_role("button", name="Features and parameters").click()
                page.wait_for_function(
                    """
                    () => document.querySelector("#gbmFeatureInteractionConstraintButton")?.textContent.trim() === "Constraint groups (1)"
                      && [...document.querySelectorAll("#gbmFeatureGrid .tabulator-row")]
                        .find((row) => row.textContent.includes("Age"))
                        ?.querySelector(".tabulator-cell[tabulator-field='grouping']")
                        ?.textContent.includes("\\uD83D\\uDD12")
                    """,
                    timeout=10_000,
                )
                page.locator("#gbmFeatureScenarioButton").click()
                scenario_menu = feature_scenario_state()
                self.assertIn("scenario1 (2)", scenario_menu["rows"])
                page.locator("#gbmFeatureSectionTitle").click()
                page.wait_for_function(
                    """
                    () => document.querySelector("#gbmFeatureScenarioMenu")?.classList.contains("hidden")
                      && document.querySelector("#gbmFeatureScenarioButton")?.getAttribute("aria-expanded") === "false"
                    """,
                    timeout=10_000,
                )
                choose_feature_scenario("scenario1")
                assert_feature_heading_matches_checked(2)
                self.assertEqual(feature_scenario_state()["value"], "scenario1")
                page.locator("#gbmClearFeaturesBtn").click()
                assert_feature_heading_matches_checked(0)
                self.assertEqual(feature_scenario_state()["value"], "")
                page.get_by_text("Parameters", exact=True).wait_for(timeout=10_000)
                page.get_by_text("Evaluation Log", exact=True).wait_for(timeout=10_000)
                page.get_by_role("button", name="SHAP", exact=True).click()
                page.locator("#gbmShapChart").wait_for(timeout=10_000)
                page.wait_for_function(
                    """
                    () => {
                      const chart = window.echarts.getInstanceByDom(document.querySelector("#gbmShapChart"));
                      const option = chart?.getOption();
                      return option?.title?.[0]?.text?.includes("SHAP flame plot: lat")
                        && option.series?.length > 0
                        && document.querySelector("#gbmShapFeatureList1 .feature.active")?.textContent.includes("lat")
                        && document.querySelector("#gbmShapFeatureList2 .feature.active")?.textContent.includes("None");
                    }
                    """,
                    timeout=10_000,
                )
                default_shap_feature_label = page.locator("#gbmShapFeatureList1 .feature.active .kind").text_content()
                self.assertEqual(default_shap_feature_label, "Rank 1 · 0.2330")
                self.assertNotIn("numeric", default_shap_feature_label)
                shap_banding_buttons = page.evaluate(
                    """
                    () => ({
                      feature1: [...document.querySelectorAll(".gbm-shap-feature1-control [data-gbm-shap-band-value]")]
                        .map((button) => button.textContent.trim()),
                      feature2: [...document.querySelectorAll(".gbm-shap-feature2-control [data-gbm-shap-band-value]")]
                        .map((button) => button.textContent.trim()),
                    })
                    """
                )
                self.assertEqual(shap_banding_buttons["feature1"], ["0.01", "0.1", "1", "5", "10"])
                self.assertEqual(shap_banding_buttons["feature2"], ["0.01", "0.1", "1", "5", "10"])
                divider_height_before = page.evaluate(
                    """() => document.querySelector("#gbmShapFeatureList1")?.closest(".gbm-shap-feature-section")?.getBoundingClientRect().height || 0"""
                )
                divider_box = page.locator("#gbmShapChooserDivider").bounding_box()
                self.assertIsNotNone(divider_box)
                assert divider_box is not None
                page.mouse.move(divider_box["x"] + divider_box["width"] / 2, divider_box["y"] + divider_box["height"] / 2)
                page.mouse.down()
                page.mouse.move(divider_box["x"] + divider_box["width"] / 2, divider_box["y"] + divider_box["height"] / 2 + 36, steps=4)
                page.mouse.up()
                page.wait_for_function(
                    """
                    (height) => {
                      const current = document.querySelector("#gbmShapFeatureList1")?.closest(".gbm-shap-feature-section")?.getBoundingClientRect().height || 0;
                      const cssHeight = parseFloat(getComputedStyle(document.documentElement).getPropertyValue("--gbm-shap-feature1-height"));
                      return Number.isFinite(cssHeight) && cssHeight > 0 && Math.abs(current - height) > 8;
                    }
                    """,
                    arg=divider_height_before,
                    timeout=10_000,
                )
                page.locator("#gbmShapFeatureList1 .feature", has_text="Age").click()
                page.wait_for_function(
                    """
                    () => {
                      const chart = window.echarts.getInstanceByDom(document.querySelector("#gbmShapChart"));
                      const option = chart?.getOption();
                      return option?.title?.[0]?.text?.includes("SHAP flame plot: Age")
                        && document.querySelector("#gbmShapFeatureList1 .feature.active")?.textContent.includes("Rank 2 · 0.1830");
                    }
                    """,
                    timeout=10_000,
                )
                initial_shap_state = page.evaluate(
                    """
                    () => {
                      const chart = window.echarts.getInstanceByDom(document.querySelector("#gbmShapChart"));
                      const option = chart?.getOption();
                      const medianIndex = option.series.findIndex((series) => series.name === "Median");
                      const median = option.series[medianIndex];
                      const formatter = option.tooltip?.[0]?.formatter;
                      return {
                        xMin: option.xAxis?.[0]?.min,
                        xMax: option.xAxis?.[0]?.max,
                        xInterval: option.xAxis?.[0]?.interval,
                        firstMedianX: median?.data?.[0]?.[0],
                        lastMedianX: median?.data?.[median.data.length - 1]?.[0],
                        legendLeft: option.legend?.[0]?.left || "",
                        seriesNames: option.series?.map((series) => series.name) || [],
                        hasRibbon: option.series?.some((series) => series.type === "custom"),
                        ribbonColors: option.series
                          ?.filter((series) => series.type === "custom")
                          .map((series) => series.itemStyle?.color || ""),
                        medianColor: option.series?.find((series) => series.name === "Median")?.itemStyle?.color || "",
                        tooltipText: typeof formatter === "function"
                          ? formatter([{ axisValue: median?.data?.[0]?.[0], value: median?.data?.[0] }])
                          : "",
                      };
                    }
                    """
                )
                self.assertAlmostEqual(initial_shap_state["xMin"], initial_shap_state["firstMedianX"])
                self.assertAlmostEqual(initial_shap_state["xMax"], initial_shap_state["lastMedianX"])
                self.assertAlmostEqual(initial_shap_state["xInterval"], 5)
                self.assertTrue(initial_shap_state["hasRibbon"])
                self.assertEqual(initial_shap_state["legendLeft"], "center")
                self.assertNotIn("45-55", initial_shap_state["seriesNames"])
                self.assertTrue(all(color.startswith("rgba(209, 63, 63,") for color in initial_shap_state["ribbonColors"]))
                self.assertEqual(initial_shap_state["medianColor"], "#d13f3f")
                tooltip_text = initial_shap_state["tooltipText"]
                self.assertIn("Median", tooltip_text)
                self.assertNotIn("45-55", tooltip_text)
                self.assertIsNone(re.search(r"\d+\.\d{5,}", tooltip_text))
                with page.expect_response(lambda response: "/api/gbm/models/" in response.url and "/shap/plot" in response.url and response.request.method == "POST"):
                    page.locator('[data-gbm-shap-rescale="1"]').click()
                page.wait_for_function(
                    """
                    () => {
                      const chart = window.echarts.getInstanceByDom(document.querySelector("#gbmShapChart"));
                      const option = chart?.getOption();
                      return option?.title?.[0]?.text?.includes("SHAP flame plot: Age")
                        && option.yAxis?.[0]?.axisLabel?.formatter?.(1) === "0%"
                        && option.yAxis?.[0]?.axisLabel?.formatter?.(1.25) === "+25%";
                    }
                    """,
                    timeout=10_000,
                )
                with page.expect_response(lambda response: "/api/gbm/models/" in response.url and "/shap/plot" in response.url and response.request.method == "POST"):
                    page.locator('[data-gbm-shap-rescale="-"]').click()
                page.wait_for_function(
                    """
                    () => {
                      const chart = window.echarts.getInstanceByDom(document.querySelector("#gbmShapChart"));
                      const option = chart?.getOption();
                      return option?.title?.[0]?.text?.includes("SHAP flame plot: Age")
                        && option.yAxis?.[0]?.axisLabel?.formatter?.(1) === "1";
                    }
                    """,
                    timeout=10_000,
                )
                page.evaluate(
                    """
                    () => {
                      const chart = window.echarts.getInstanceByDom(document.querySelector("#gbmShapChart"));
                      chart?.dispatchAction({ type: "legendUnSelect", name: "5-95" });
                    }
                    """
                )
                with page.expect_response(lambda response: "/api/gbm/models/" in response.url and "/shap/plot" in response.url and response.request.method == "POST"):
                    page.locator('.gbm-shap-feature1-control [data-gbm-shap-band-value="5"]').click()
                page.wait_for_function(
                    """
                    () => {
                      const chart = window.echarts.getInstanceByDom(document.querySelector("#gbmShapChart"));
                      return chart?.getOption()?.legend?.[0]?.selected?.["5-95"] === false;
                    }
                    """,
                    timeout=10_000,
                )
                with page.expect_response(lambda response: "/api/gbm/models/" in response.url and "/shap/plot" in response.url and response.request.method == "POST"):
                    page.locator("#gbmShapFeatureList2 .feature", has_text="lat").click()
                page.wait_for_function(
                    """
                    () => {
                      const chart = window.echarts.getInstanceByDom(document.querySelector("#gbmShapChart"));
                      const option = chart?.getOption();
                      return option?.title?.[0]?.text?.includes("SHAP surface plot: Age x lat")
                        && option.series?.some((series) => series.type === "surface");
                    }
                    """,
                    timeout=10_000,
                )
                with page.expect_response(lambda response: "/api/gbm/models/" in response.url and "/shap/plot" in response.url and response.request.method == "POST"):
                    page.locator("#gbmShapFeatureList2 .feature", has_text="None").click()
                page.wait_for_function(
                    """
                    () => {
                      const chart = window.echarts.getInstanceByDom(document.querySelector("#gbmShapChart"));
                      const option = chart?.getOption();
                      return option?.title?.[0]?.text?.includes("SHAP flame plot: Age")
                        && option.legend?.[0]?.selected?.["5-95"] !== false;
                    }
                    """,
                    timeout=10_000,
                )
                with page.expect_response(lambda response: "/api/gbm/models/" in response.url and "/shap/plot" in response.url and response.request.method == "POST"):
                    page.locator("#gbmShapFeatureList2 .feature", has_text="lat").click()
                page.wait_for_function(
                    """
                    () => {
                      const chart = window.echarts.getInstanceByDom(document.querySelector("#gbmShapChart"));
                      const option = chart?.getOption();
                      return option?.title?.[0]?.text?.includes("SHAP surface plot: Age x lat")
                        && option.series?.some((series) => series.type === "surface");
                    }
                    """,
                    timeout=10_000,
                )
                surface_state = page.evaluate(
                    """
                    () => {
                      const chart = window.echarts.getInstanceByDom(document.querySelector("#gbmShapChart"));
                      const option = chart?.getOption();
                      const surface = option?.series?.find((series) => series.type === "surface");
                      const grid3D = option?.grid3D?.[0] || {};
                      const xAxis3D = option?.xAxis3D?.[0] || {};
                      return {
                        title: option?.title?.[0]?.text || "",
                        invalidZ: Boolean(surface?.data?.some((point) => Number.isNaN(point?.[2]) || point?.[2] == null)),
                        visualMapHover: option?.visualMap?.[0]?.formatter?.(0.123456) || "",
                        gridTop: grid3D.top,
                        gridBottom: grid3D.bottom,
                        boxWidth: grid3D.boxWidth,
                        boxDepth: grid3D.boxDepth,
                        axisLabelFontSize: xAxis3D.axisLabel?.fontSize,
                        axisNameFontSize: xAxis3D.nameTextStyle?.fontSize,
                        noticeHidden: Boolean(document.querySelector("#gbmNotice")?.classList.contains("hidden")),
                        noticeText: document.querySelector("#gbmNotice")?.textContent || "",
                      };
                    }
                    """
                )
                self.assertIn("SHAP surface plot: Age x lat", surface_state["title"])
                self.assertTrue(surface_state["invalidZ"])
                self.assertEqual(surface_state["visualMapHover"], "0.1235")
                self.assertEqual(surface_state["gridTop"], 34)
                self.assertEqual(surface_state["gridBottom"], 54)
                self.assertEqual(surface_state["boxWidth"], 100)
                self.assertEqual(surface_state["boxDepth"], 74)
                self.assertEqual(surface_state["axisLabelFontSize"], 10)
                self.assertEqual(surface_state["axisNameFontSize"], 11)
                self.assertTrue(surface_state["noticeHidden"])
                self.assertNotIn("undefined is not an object", surface_state["noticeText"])
                page.locator('[data-gbm-shap-factor="1"]').check()
                page.locator('[data-gbm-shap-factor="2"]').check()
                page.wait_for_function(
                    """
                    () => {
                      const chart = window.echarts.getInstanceByDom(document.querySelector("#gbmShapChart"));
                      const option = chart?.getOption();
                      return option?.title?.[0]?.text?.includes("SHAP heatmap: Age x lat")
                        && option.series?.some((series) => series.type === "heatmap");
                    }
                    """,
                    timeout=10_000,
                )
                shap_axis_formatting = page.evaluate(
                    """
                    () => {
                      const chart = window.echarts.getInstanceByDom(document.querySelector("#gbmShapChart"));
                      const option = chart?.getOption();
                      function visibleLabels(axis) {
                        const interval = axis?.axisLabel?.interval;
                        const formatter = axis?.axisLabel?.formatter;
                        return (axis?.data || [])
                          .filter((label, index) => typeof interval === "function" ? interval(index, label) : true)
                          .map((label) => typeof formatter === "function" ? formatter(label) : String(label));
                      }
                      function hasNiceSpacing(labels) {
                        const numbers = labels.map(Number).filter(Number.isFinite);
                        if (numbers.length < 2) return true;
                        const diffs = numbers.slice(1).map((value, index) => Math.abs(value - numbers[index])).filter((value) => value > 0);
                        return diffs.every((diff) => {
                          const magnitude = 10 ** Math.floor(Math.log10(diff));
                          const normalised = Number((diff / magnitude).toPrecision(12));
                          return [1, 2, 5, 10].some((candidate) => Math.abs(normalised - candidate) < 1e-9);
                        });
                      }
                      function isMonotoneNumeric(labels) {
                        const numbers = labels.map(Number).filter(Number.isFinite);
                        return numbers.length > 1 && numbers.every((value, index) => index === 0 || value >= numbers[index - 1]);
                      }
                      const xLabels = visibleLabels(option?.xAxis?.[0]);
                      const yLabels = visibleLabels(option?.yAxis?.[0]);
                      return {
                        x: option?.xAxis?.[0]?.axisLabel?.formatter?.("56.800000000000004") || "",
                        y: option?.yAxis?.[0]?.axisLabel?.formatter?.("49.00000000000001") || "",
                        xMonotone: isMonotoneNumeric(option?.xAxis?.[0]?.data || []),
                        yMonotone: isMonotoneNumeric(option?.yAxis?.[0]?.data || []),
                        xIntervalType: typeof option?.xAxis?.[0]?.axisLabel?.interval,
                        yIntervalType: typeof option?.yAxis?.[0]?.axisLabel?.interval,
                        xNiceSpacing: hasNiceSpacing(xLabels),
                        yNiceSpacing: hasNiceSpacing(yLabels),
                        visualMapHover: option?.visualMap?.[0]?.formatter?.(-0.123456) || "",
                        tooltip: option?.tooltip?.[0]?.formatter?.({ value: [0, 0, -0.123456] }) || "",
                      };
                    }
                    """
                )
                self.assertEqual(shap_axis_formatting["x"], "56.8")
                self.assertEqual(shap_axis_formatting["y"], "49")
                self.assertTrue(shap_axis_formatting["xMonotone"])
                self.assertTrue(shap_axis_formatting["yMonotone"])
                self.assertEqual(shap_axis_formatting["xIntervalType"], "function")
                self.assertEqual(shap_axis_formatting["yIntervalType"], "function")
                self.assertTrue(shap_axis_formatting["xNiceSpacing"])
                self.assertTrue(shap_axis_formatting["yNiceSpacing"])
                self.assertEqual(shap_axis_formatting["visualMapHover"], "-0.1235")
                self.assertIsNone(re.search(r"\d+\.\d{5,}", shap_axis_formatting["tooltip"]))
                with page.expect_response(lambda response: "/api/gbm/models/" in response.url and "/shap/plot" in response.url and response.request.method == "POST"):
                    page.locator("#gbmShapFeatureList2 .feature", has_text="None").click()
                with page.expect_response(lambda response: "/api/gbm/models/" in response.url and "/shap/plot" in response.url and response.request.method == "POST"):
                    page.locator("#gbmShapFeatureList1 .feature", has_text="lat").click()
                page.wait_for_function(
                    """
                    () => {
                      const chart = window.echarts.getInstanceByDom(document.querySelector("#gbmShapChart"));
                      const option = chart?.getOption();
                      return option?.title?.[0]?.text?.includes("SHAP box plot: lat")
                        && option?.legend?.[0]?.show === false
                        && document.querySelector('[data-gbm-shap-factor="1"]')?.checked;
                    }
                    """,
                    timeout=10_000,
                )
                page.get_by_role("button", name="Stacked SHAP").click()
                page.locator("#gbmStackedShapChart").wait_for(timeout=10_000)
                page.wait_for_function(
                    """
                    () => {
                      const chart = window.echarts.getInstanceByDom(document.querySelector("#gbmStackedShapChart"));
                      const option = chart?.getOption();
                      return option?.title?.[0]?.text?.includes("SHAP Values by lat")
                        && option.series?.some((series) => series.name === "Sum of SHAP values" && series.type === "scatter")
                        && option.series?.some((series) => series.type === "bar")
                        && option.xAxis?.[0]?.data?.some((label) => !String(label).includes("[") && !String(label).includes(")"))
                        && option.xAxis?.[0]?.axisLabel?.rotate === 0
                        && option.xAxis?.[0]?.axisLabel?.fontSize === 10
                        && document.querySelector('[data-gbm-stacked-shap-feature-sort="importance"]')?.classList.contains("active")
                        && document.querySelector('[data-gbm-stacked-shap-feature-sort="alpha"]')
                        && document.querySelector("#gbmStackedShapFeatureList .feature.active")?.textContent.includes("lat");
                    }
                    """,
                    timeout=10_000,
                )
                stacked_feature_label = page.locator("#gbmStackedShapFeatureList .feature.active .kind").text_content()
                self.assertEqual(stacked_feature_label, "Rank 1 · 0.2330")
                with page.expect_response(lambda response: "/api/gbm/models/" in response.url and "/shap/stacked" in response.url and response.request.method == "POST"):
                    page.locator('[data-gbm-stacked-shap-feature-count="1"]').click()
                stacked_state = page.evaluate(
                    """
                    () => {
                      const chart = window.echarts.getInstanceByDom(document.querySelector("#gbmStackedShapChart"));
                      const option = chart?.getOption();
                      const bars = (option?.series || []).filter((series) => series.type === "bar");
                      const scatter = (option?.series || []).find((series) => series.name === "Sum of SHAP values");
                      const index = 0;
                      const barSum = bars.reduce((sum, series) => sum + Number(series.data?.[index] || 0), 0);
                      const dot = Number(scatter?.data?.[index]);
                      return {
                        title: option?.title?.[0]?.text || "",
                        hasOther: bars.some((series) => series.name === "Other"),
                        barSum,
                        dot,
                        reconciles: Math.abs(barSum - dot) < 1e-9,
                      };
                    }
                    """
                )
                self.assertIn("SHAP Values by lat", stacked_state["title"])
                self.assertTrue(stacked_state["hasOther"])
                self.assertTrue(stacked_state["reconciles"])
                page.get_by_role("button", name="Features and parameters").click()
                page.locator("#gbmModelSelect").wait_for(state="attached", timeout=10_000)
                page.locator("#gbmModelCollapseBtn").wait_for(timeout=10_000)
                page.wait_for_function(
                    """
                    () => [...document.querySelectorAll("#gbmFeatureGrid .tabulator-row")]
                      .some((row) => row.textContent.includes("BadText"))
                    """,
                    timeout=10_000,
                )
                invalid_feature_state = page.evaluate(
                    """
                    () => {
                      const row = [...document.querySelectorAll("#gbmFeatureGrid .tabulator-row")]
                        .find((item) => item.textContent.includes("BadText"));
                      return {
                        found: Boolean(row),
                        invalidClass: Boolean(row?.classList.contains("gbm-feature-invalid")),
                        typeText: row?.querySelector(".gbm-feature-kind")?.textContent.trim() || "",
                        hasCheckbox: Boolean(row?.querySelector(".gbm-use-checkbox")),
                        monotonicity: row?.querySelector(".tabulator-cell[tabulator-field='monotonicity']")?.textContent.trim() || "",
                      };
                    }
                    """
                )
                self.assertTrue(invalid_feature_state["found"])
                self.assertTrue(invalid_feature_state["invalidClass"])
                self.assertEqual(invalid_feature_state["typeText"], "invalid")
                self.assertFalse(invalid_feature_state["hasCheckbox"])
                self.assertEqual(invalid_feature_state["monotonicity"], "")
                self.assertEqual(
                    page.locator("#gbmParameterGrid .tabulator-row", has_text="learning_rate").locator(".tabulator-cell[tabulator-field='value']").text_content(),
                    "0.11",
                )
                page.locator("#gbmParameterGrid .tabulator-row", has_text="objective").locator(".tabulator-cell[tabulator-field='value']").click()
                page.locator("#gbmParameterGrid input.gbm-parameter-list-editor").wait_for(timeout=10_000)
                page.locator(".tabulator-popup-container.tabulator-edit-list").wait_for(timeout=10_000)
                page.keyboard.press("Escape")
                page.wait_for_function(
                    "() => !document.querySelector('#gbmParameterGrid .tabulator-cell.tabulator-editing')",
                    timeout=10_000,
                )
                page.locator("#gbmParameterGrid .tabulator-row", has_text="num_iterations").locator(".tabulator-cell[tabulator-field='value']").click()
                page.locator("#gbmParameterGrid input.gbm-parameter-input-editor").wait_for(timeout=10_000)
                page.keyboard.press("Escape")
                page.get_by_role("button", name="Model navigator").click()
                page.locator("#gbmModelGrid .tabulator-row").first.wait_for(timeout=10_000)
                navigator_state = page.evaluate(
                    """
                    () => ({
                      headers: [...document.querySelectorAll("#gbmModelGrid .tabulator-col-title")]
                        .map((node) => node.textContent.trim()).filter(Boolean),
                      rows: document.querySelectorAll("#gbmModelGrid .tabulator-row").length,
                      activeDots: document.querySelectorAll("#gbmModelGrid .gbm-model-active-dot").length,
                      activeDotRowText: document.querySelector("#gbmModelGrid .gbm-model-active-dot")?.closest(".tabulator-row")?.textContent || "",
                      selectedRows: document.querySelectorAll("#gbmModelGrid .tabulator-row.tabulator-selected").length,
                      renameDisabled: document.querySelector("#gbmRenameModelBtn")?.disabled,
                      activateDisabled: document.querySelector("#gbmActivateModelBtn")?.disabled,
                      deleteDisabled: document.querySelector("#gbmDeleteModelBtn")?.disabled,
                      hasActivateButton: Boolean(document.querySelector("#gbmActivateModelBtn")),
                    })
                    """
                )
                self.assertEqual(
                    navigator_state["headers"],
                    [
                        "Model", "Created", "Response", "Weight", "Objective", "Metric", "Mode", "Constraints", "Train", "Best iter.",
                        "tr@best", "te@best", "n_iter", "lr", "leaves", "depth", "min_leaf", "ES", "Run time", "Sample",
                    ],
                )
                self.assertEqual(navigator_state["rows"], 4)
                self.assertEqual(navigator_state["activeDots"], 1)
                self.assertIn("Browser smoke model", navigator_state["activeDotRowText"])
                self.assertEqual(navigator_state["selectedRows"], 0)
                self.assertTrue(navigator_state["renameDisabled"])
                self.assertTrue(navigator_state["activateDisabled"])
                self.assertTrue(navigator_state["deleteDisabled"])
                self.assertTrue(navigator_state["hasActivateButton"])
                page.locator("#gbmModelGrid .tabulator-row", has_text="Disposable smoke model A").click()
                page.locator("#gbmActivateModelBtn").click()
                page.wait_for_function(
                    """
                    () => document.querySelector("#gbmModelSelectedMeta")?.textContent.includes("Disposable smoke model A")
                      && document.querySelector("#gbmModelGrid .gbm-model-active-dot")?.closest(".tabulator-row")?.textContent.includes("Disposable smoke model A")
                    """,
                    timeout=10_000,
                )
                page.get_by_role("button", name="Features and parameters").click()
                page.wait_for_function(
                    """
                    () => {
                      const lockState = (name) => {
                        const row = [...document.querySelectorAll("#gbmFeatureGrid .tabulator-row")]
                          .find((item) => item.querySelector(".tabulator-cell[tabulator-field='name']")?.textContent.includes(name));
                        const cell = row?.querySelector(".tabulator-cell[tabulator-field='name']");
                        return {
                          pair: Boolean(cell?.querySelector(".gbm-pair-interaction-lock")),
                          singleton: Boolean(cell?.querySelector(".gbm-feature-interaction-lock")),
                          subscript: cell?.querySelector(".gbm-pair-interaction-lock .gbm-interaction-lock-subscript")?.textContent.trim() || "",
                        };
                      };
                      const age = lockState("Age");
                      const segment = lockState("Segment");
                      return document.querySelector("#gbmFeatureInteractionPairButton")?.textContent.trim() === "Interaction pairs (1)"
                        && document.querySelectorAll("[data-gbm-interaction-pair-row]").length === 1
                        && age.pair && !age.singleton && age.subscript === "2"
                        && segment.pair && !segment.singleton && segment.subscript === "2";
                    }
                    """,
                    timeout=10_000,
                )
                pair_validate_payload: dict[str, Any] = {}
                pair_train_payload: dict[str, Any] = {}

                def pair_validate_route(route: Any) -> None:
                    pair_validate_payload["value"] = route.request.post_data_json
                    route.fulfill(
                        status=200,
                        content_type="application/json",
                        body=json.dumps({"ok": True, "errors": [], "warnings": [], "grid": {"messages": []}}),
                    )

                def pair_train_route(route: Any) -> None:
                    pair_train_payload["value"] = route.request.post_data_json
                    route.fulfill(
                        status=200,
                        content_type="application/json",
                        body=json.dumps(
                            {
                                "job_id": "pair-live-job",
                                "status": "queued",
                                "created_at": "2026-05-25T00:00:00Z",
                                "updated_at": "2026-05-25T00:00:00Z",
                                "result": None,
                                "error": None,
                                "progress": None,
                            }
                        ),
                    )

                def pair_job_route(route: Any) -> None:
                    route.fulfill(
                        status=200,
                        content_type="application/json",
                        body=json.dumps(
                            {
                                "job_id": "pair-live-job",
                                "status": "succeeded",
                                "created_at": "2026-05-25T00:00:00Z",
                                "updated_at": "2026-05-25T00:00:01Z",
                                "result": {"sources": {}},
                                "error": None,
                                "progress": {"phase": "succeeded", "message": "GBM training complete", "percent": 100},
                            }
                        ),
                    )

                page.route("**/api/gbm/validate", pair_validate_route)
                page.route("**/api/gbm/train", pair_train_route)
                page.route("**/api/gbm/jobs/pair-live-job", pair_job_route)
                with page.expect_request("**/api/gbm/train", timeout=10_000):
                    page.locator("#gbmTrainBtn").click()
                page.wait_for_function("() => !document.querySelector('#gbmTrainBtn')?.classList.contains('training')", timeout=10_000)
                self.assertEqual(pair_validate_payload["value"]["feature_interaction_pairs"], [{"left": "Age", "right": "Segment"}])
                self.assertEqual(pair_train_payload["value"]["feature_interaction_pairs"], [{"left": "Age", "right": "Segment"}])
                page.unroute("**/api/gbm/validate", pair_validate_route)
                page.unroute("**/api/gbm/train", pair_train_route)
                page.unroute("**/api/gbm/jobs/pair-live-job", pair_job_route)
                page.get_by_role("button", name="Model navigator").click()
                page.locator("#gbmModelGrid .tabulator-row", has_text="Browser smoke model").click()
                page.locator("#gbmActivateModelBtn").click()
                page.wait_for_function(
                    """
                    () => document.querySelector("#gbmModelSelectedMeta")?.textContent.includes("Browser smoke model")
                      && document.querySelector("#gbmModelGrid .gbm-model-active-dot")?.closest(".tabulator-row")?.textContent.includes("Browser smoke model")
                    """,
                    timeout=10_000,
                )
                page.locator("#gbmModelGrid .tabulator-row", has_text="Disposable smoke model A").click()
                page.locator("#gbmModelGrid .tabulator-row", has_text="Disposable smoke model B").click()
                plain_replace_state = page.evaluate(
                    """
                    () => ({
                      selectedRows: document.querySelectorAll("#gbmModelGrid .tabulator-row.tabulator-selected").length,
                      selectedText: [...document.querySelectorAll("#gbmModelGrid .tabulator-row.tabulator-selected")]
                        .map((row) => row.textContent),
                      renameDisabled: document.querySelector("#gbmRenameModelBtn")?.disabled,
                      activateDisabled: document.querySelector("#gbmActivateModelBtn")?.disabled,
                      deleteDisabled: document.querySelector("#gbmDeleteModelBtn")?.disabled,
                    })
                    """
                )
                self.assertEqual(plain_replace_state["selectedRows"], 1)
                self.assertTrue(any("Disposable smoke model B" in text for text in plain_replace_state["selectedText"]))
                self.assertFalse(any("Disposable smoke model A" in text for text in plain_replace_state["selectedText"]))
                self.assertFalse(plain_replace_state["renameDisabled"])
                self.assertFalse(plain_replace_state["activateDisabled"])
                self.assertFalse(plain_replace_state["deleteDisabled"])
                page.locator("#gbmModelGrid .tabulator-row", has_text="Disposable smoke model A").click(modifiers=["Shift"])
                shift_selected_state = page.evaluate(
                    """
                    () => ({
                      selectedRows: document.querySelectorAll("#gbmModelGrid .tabulator-row.tabulator-selected").length,
                      selectedText: [...document.querySelectorAll("#gbmModelGrid .tabulator-row.tabulator-selected")]
                        .map((row) => row.textContent),
                      renameDisabled: document.querySelector("#gbmRenameModelBtn")?.disabled,
                      activateDisabled: document.querySelector("#gbmActivateModelBtn")?.disabled,
                      deleteDisabled: document.querySelector("#gbmDeleteModelBtn")?.disabled,
                    })
                    """
                )
                self.assertEqual(shift_selected_state["selectedRows"], 2)
                self.assertTrue(any("Disposable smoke model A" in text for text in shift_selected_state["selectedText"]))
                self.assertTrue(any("Disposable smoke model B" in text for text in shift_selected_state["selectedText"]))
                self.assertTrue(shift_selected_state["renameDisabled"])
                self.assertTrue(shift_selected_state["activateDisabled"])
                self.assertFalse(shift_selected_state["deleteDisabled"])
                row_selection_modifier = "Meta" if sys.platform == "darwin" else "Control"
                page.locator("#gbmModelGrid .tabulator-row", has_text="Second smoke model").click(modifiers=[row_selection_modifier])
                command_selected_state = page.evaluate(
                    """
                    () => ({
                      selectedRows: document.querySelectorAll("#gbmModelGrid .tabulator-row.tabulator-selected").length,
                      selectedText: [...document.querySelectorAll("#gbmModelGrid .tabulator-row.tabulator-selected")]
                        .map((row) => row.textContent),
                      renameDisabled: document.querySelector("#gbmRenameModelBtn")?.disabled,
                      activateDisabled: document.querySelector("#gbmActivateModelBtn")?.disabled,
                      deleteDisabled: document.querySelector("#gbmDeleteModelBtn")?.disabled,
                    })
                    """
                )
                self.assertEqual(command_selected_state["selectedRows"], 3)
                self.assertTrue(any("Second smoke model" in text for text in command_selected_state["selectedText"]))
                self.assertTrue(any("Disposable smoke model A" in text for text in command_selected_state["selectedText"]))
                self.assertTrue(any("Disposable smoke model B" in text for text in command_selected_state["selectedText"]))
                self.assertTrue(command_selected_state["renameDisabled"])
                self.assertTrue(command_selected_state["activateDisabled"])
                self.assertFalse(command_selected_state["deleteDisabled"])
                page.locator("#gbmModelGrid .tabulator-row", has_text="Second smoke model").click(modifiers=[row_selection_modifier])
                multi_selected_state = page.evaluate(
                    """
                    () => ({
                      selectedRows: document.querySelectorAll("#gbmModelGrid .tabulator-row.tabulator-selected").length,
                      selectedText: [...document.querySelectorAll("#gbmModelGrid .tabulator-row.tabulator-selected")]
                        .map((row) => row.textContent),
                      renameDisabled: document.querySelector("#gbmRenameModelBtn")?.disabled,
                      activateDisabled: document.querySelector("#gbmActivateModelBtn")?.disabled,
                      deleteDisabled: document.querySelector("#gbmDeleteModelBtn")?.disabled,
                    })
                    """
                )
                self.assertEqual(multi_selected_state["selectedRows"], 2)
                self.assertFalse(any("Second smoke model" in text for text in multi_selected_state["selectedText"]))
                self.assertTrue(any("Disposable smoke model A" in text for text in multi_selected_state["selectedText"]))
                self.assertTrue(any("Disposable smoke model B" in text for text in multi_selected_state["selectedText"]))
                self.assertTrue(multi_selected_state["renameDisabled"])
                self.assertTrue(multi_selected_state["activateDisabled"])
                self.assertFalse(multi_selected_state["deleteDisabled"])
                page.evaluate("() => { window.confirm = () => true; }")
                page.locator("#gbmDeleteModelBtn").click()
                page.wait_for_function(
                    """
                    () => document.querySelectorAll("#gbmModelGrid .tabulator-row").length === 2
                      && !document.body.textContent.includes("Disposable smoke model A")
                      && !document.body.textContent.includes("Disposable smoke model B")
                      && document.querySelector("#gbmModelSelectedMeta")?.textContent.includes("Browser smoke model")
                      && document.querySelector(".dataset-meta-gbm-link")?.textContent.trim() === "GBMs (2)"
                    """,
                    timeout=10_000,
                )
                page.locator("#gbmModelGrid .tabulator-row", has_text="Second smoke model").click()
                page.wait_for_function(
                    """
                    () => [...document.querySelectorAll("#gbmModelGrid .tabulator-row.tabulator-selected")]
                      .some((row) => row.textContent.includes("Second smoke model"))
                    """,
                    timeout=10_000,
                )
                selected_navigator_state = page.evaluate(
                    """
                    () => ({
                      activeDotRowText: document.querySelector("#gbmModelGrid .gbm-model-active-dot")?.closest(".tabulator-row")?.textContent || "",
                      selectedRows: document.querySelectorAll("#gbmModelGrid .tabulator-row.tabulator-selected").length,
                      renameDisabled: document.querySelector("#gbmRenameModelBtn")?.disabled,
                      activateDisabled: document.querySelector("#gbmActivateModelBtn")?.disabled,
                      deleteDisabled: document.querySelector("#gbmDeleteModelBtn")?.disabled,
                      sidebarMeta: document.querySelector("#gbmModelSelectedMeta")?.textContent || "",
                    })
                    """
                )
                self.assertIn("Browser smoke model", selected_navigator_state["activeDotRowText"])
                self.assertEqual(selected_navigator_state["selectedRows"], 1)
                self.assertFalse(selected_navigator_state["renameDisabled"])
                self.assertFalse(selected_navigator_state["activateDisabled"])
                self.assertFalse(selected_navigator_state["deleteDisabled"])
                self.assertIn("Browser smoke model", selected_navigator_state["sidebarMeta"])
                page.locator("#gbmActivateModelBtn").click()
                page.wait_for_function(
                    """
                    () => document.querySelector("#gbmModelSelectedMeta")?.textContent.includes("Second smoke model")
                      && document.querySelector("#gbmModelGrid .gbm-model-active-dot")?.closest(".tabulator-row")?.textContent.includes("Second smoke model")
                    """,
                    timeout=10_000,
                )
                activated_navigator_state = page.evaluate(
                    """
                    () => ({
                      activeDotRowText: document.querySelector("#gbmModelGrid .gbm-model-active-dot")?.closest(".tabulator-row")?.textContent || "",
                      selectedRows: document.querySelectorAll("#gbmModelGrid .tabulator-row.tabulator-selected").length,
                      renameDisabled: document.querySelector("#gbmRenameModelBtn")?.disabled,
                      activateDisabled: document.querySelector("#gbmActivateModelBtn")?.disabled,
                      deleteDisabled: document.querySelector("#gbmDeleteModelBtn")?.disabled,
                      sidebarMeta: document.querySelector("#gbmModelSelectedMeta")?.textContent || "",
                    })
                    """
                )
                self.assertIn("Second smoke model", activated_navigator_state["activeDotRowText"])
                self.assertEqual(activated_navigator_state["selectedRows"], 0)
                self.assertTrue(activated_navigator_state["renameDisabled"])
                self.assertTrue(activated_navigator_state["activateDisabled"])
                self.assertTrue(activated_navigator_state["deleteDisabled"])
                self.assertIn("Second smoke model", activated_navigator_state["sidebarMeta"])
                page.get_by_role("button", name="SHAP", exact=True).click()
                page.wait_for_function(
                    """
                    () => {
                      const chart = window.echarts.getInstanceByDom(document.querySelector("#gbmShapChart"));
                      const option = chart?.getOption();
                      return option?.title?.[0]?.text?.includes("SHAP flame plot: Age")
                        && document.querySelector("#gbmShapFeatureList1 .feature.active")?.textContent.includes("Age")
                        && document.querySelector("#gbmShapFeatureList2 .feature.active")?.textContent.includes("None");
                    }
                    """,
                    timeout=10_000,
                )
                page.locator("#gbmShapFeatureList2 .feature", has_text="Segment").click()
                page.wait_for_function(
                    """
                    () => {
                      const chart = window.echarts.getInstanceByDom(document.querySelector("#gbmShapChart"));
                      const option = chart?.getOption();
                      return option?.title?.[0]?.text?.includes("SHAP lines plot: Age x Segment")
                        && option.series?.some((series) => series.type === "line");
                    }
                    """,
                    timeout=10_000,
                )
                self.assertFalse(page.locator("#gbmShapMessage", has_text="undefined is not an object").is_visible())
                page.get_by_role("button", name="Model navigator").click()
                page.locator("#gbmModelGrid .tabulator-row", has_text="Browser smoke model").click()
                page.locator("#gbmActivateModelBtn").click()
                page.wait_for_function(
                    """
                    () => document.querySelector("#gbmModelSelectedMeta")?.textContent.includes("Browser smoke model")
                      && document.querySelector("#gbmModelGrid .gbm-model-active-dot")?.closest(".tabulator-row")?.textContent.includes("Browser smoke model")
                    """,
                    timeout=10_000,
                )
                page.get_by_role("button", name="SHAP", exact=True).click()
                page.wait_for_function(
                    """
                    () => {
                      const chart = window.echarts.getInstanceByDom(document.querySelector("#gbmShapChart"));
                      const option = chart?.getOption();
                      return option?.title?.[0]?.text?.includes("SHAP flame plot: Age")
                        && document.querySelector("#gbmShapFeatureList1 .feature.active")?.textContent.includes("Age")
                        && document.querySelector("#gbmShapFeatureList2 .feature.active")?.textContent.includes("None");
                    }
                    """,
                    timeout=10_000,
                )
                page.evaluate(
                    """
                    () => {
                      const chart = window.echarts.getInstanceByDom(document.querySelector("#gbmShapChart"));
                      chart?.dispatchAction({ type: "legendUnSelect", name: "5-95" });
                    }
                    """
                )
                page.get_by_role("button", name="Model navigator").click()
                page.locator("#gbmModelGrid .tabulator-row", has_text="Second smoke model").click()
                page.locator("#gbmActivateModelBtn").click()
                page.wait_for_function(
                    """
                    () => document.querySelector("#gbmModelSelectedMeta")?.textContent.includes("Second smoke model")
                      && document.querySelector("#gbmModelGrid .gbm-model-active-dot")?.closest(".tabulator-row")?.textContent.includes("Second smoke model")
                    """,
                    timeout=10_000,
                )
                page.get_by_role("button", name="SHAP", exact=True).click()
                page.wait_for_function(
                    """
                    () => {
                      const chart = window.echarts.getInstanceByDom(document.querySelector("#gbmShapChart"));
                      const option = chart?.getOption();
                      return option?.title?.[0]?.text?.includes("SHAP flame plot: Age")
                        && option.legend?.[0]?.selected?.["5-95"] === false
                        && document.querySelector("#gbmShapFeatureList1 .feature.active")?.textContent.includes("Age")
                        && document.querySelector("#gbmShapFeatureList2 .feature.active")?.textContent.includes("None");
                    }
                    """,
                    timeout=10_000,
                )
                page.get_by_role("button", name="Model navigator").click()
                page.locator("#gbmModelGrid .tabulator-row", has_text="Browser smoke model").click()
                page.locator("#gbmActivateModelBtn").click()
                page.wait_for_function(
                    """
                    () => document.querySelector("#gbmModelSelectedMeta")?.textContent.includes("Browser smoke model")
                      && document.querySelector("#gbmModelGrid .gbm-model-active-dot")?.closest(".tabulator-row")?.textContent.includes("Browser smoke model")
                    """,
                    timeout=10_000,
                )
                page.get_by_role("button", name="SHAP", exact=True).click()
                page.wait_for_function(
                    """
                    () => {
                      const chart = window.echarts.getInstanceByDom(document.querySelector("#gbmShapChart"));
                      const option = chart?.getOption();
                      return option?.title?.[0]?.text?.includes("SHAP flame plot: Age")
                        && option.legend?.[0]?.selected?.["5-95"] === false
                        && document.querySelector("#gbmShapFeatureList1 .feature.active")?.textContent.includes("Age");
                    }
                    """,
                    timeout=10_000,
                )
                page.locator('[data-gbm-shap-feature="1"][data-gbm-shap-sort="alpha"]').click()
                page.locator("#gbmShapFeatureList1 .feature", has_text="lat").click()
                page.wait_for_function(
                    """
                    () => {
                      const chart = window.echarts.getInstanceByDom(document.querySelector("#gbmShapChart"));
                      const option = chart?.getOption();
                      return option?.title?.[0]?.text?.includes("SHAP flame plot: lat")
                        && document.querySelector("#gbmShapFeatureList1 .feature.active")?.textContent.includes("lat");
                    }
                    """,
                    timeout=10_000,
                )
                page.get_by_role("button", name="Model navigator").click()
                page.locator("#gbmModelGrid .tabulator-row", has_text="Second smoke model").click()
                page.locator("#gbmActivateModelBtn").click()
                page.wait_for_function(
                    """
                    () => document.querySelector("#gbmModelSelectedMeta")?.textContent.includes("Second smoke model")
                      && document.querySelector("#gbmModelGrid .gbm-model-active-dot")?.closest(".tabulator-row")?.textContent.includes("Second smoke model")
                    """,
                    timeout=10_000,
                )
                page.get_by_role("button", name="SHAP", exact=True).click()
                page.wait_for_function(
                    """
                    () => {
                      const chart = window.echarts.getInstanceByDom(document.querySelector("#gbmShapChart"));
                      const option = chart?.getOption();
                      return option?.title?.[0]?.text?.includes("SHAP flame plot: Age")
                        && document.querySelector("#gbmShapFeatureList1 .feature.active")?.textContent.includes("Age")
                        && document.querySelector("#gbmShapFeatureList2 .feature.active")?.textContent.includes("None")
                        && option.legend?.[0]?.selected?.["5-95"] !== false
                        && document.querySelector('[data-gbm-shap-feature="1"][data-gbm-shap-sort="alpha"]')?.classList.contains("active");
                    }
                    """,
                    timeout=10_000,
                )
                page.get_by_role("button", name="Features and parameters").click()
                self.assertEqual(
                    page.locator("#gbmParameterGrid .tabulator-row", has_text="learning_rate").locator(".tabulator-cell[tabulator-field='value']").text_content(),
                    "0.22",
                )
                self.assertEqual(feature_scenario_state()["value"], "scenario1")
                self.assertEqual(page.locator("#gbmFeatureInteractionConstraintButton").text_content(), "Constraint groups (1)")
                assert_feature_heading_matches_checked(2)
                self.assertTrue(page.locator("input[name='gbmTrainingMode'][value='ebm']").is_checked())
                ebm_metric_state = page.evaluate(
                    """
                    () => ({
                      checkedMetric: document.querySelector("input[name='gbmFeatureMetric']:checked")?.value || "",
                      metricLabels: [...document.querySelectorAll("#gbmFeatureMetricToggle .gbm-feature-metric-option span")].map((node) => node.textContent.trim()),
                    })
                    """
                )
                self.assertEqual(ebm_metric_state["checkedMetric"], "gain")
                self.assertEqual(ebm_metric_state["metricLabels"], ["EBM Gain", "Gain", "SHAP"])
                page.locator("input[name='gbmFeatureMetric'][value='gain_ebm']").check(force=True)
                page.wait_for_function(
                    """
                    () => {
                      const headers = [...document.querySelectorAll("#gbmEbmGainSummaryGrid .tabulator-col-title")]
                        .map((node) => node.textContent.trim()).filter(Boolean);
                      const firstRow = document.querySelector("#gbmEbmGainSummaryGrid .tabulator-row");
                      return document.querySelector("input[name='gbmFeatureMetric']:checked")?.value === "gain_ebm"
                        && document.querySelector("#gbmFeatureGrid")?.classList.contains("hidden")
                        && !document.querySelector("#gbmEbmGainSummaryGrid")?.classList.contains("hidden")
                        && ["Tree features", "Dim", "Trees", "Gain", "% Gain"].every((header) => headers.includes(header))
                        && firstRow?.textContent.includes("Age x Segment")
                        && firstRow?.textContent.includes("100.0%");
                    }
                    """,
                    timeout=10_000,
                )
                ebm_summary_alignment = page.evaluate(
                    """
                    () => Object.fromEntries(["dim", "trees", "gain", "gain_percent"].map((field) => {
                      const cell = document.querySelector(`#gbmEbmGainSummaryGrid .tabulator-cell[tabulator-field='${field}']`);
                      return [field, {
                        textAlign: cell ? getComputedStyle(cell).textAlign : "",
                        justifyContent: cell ? getComputedStyle(cell).justifyContent : "",
                      }];
                    }))
                    """
                )
                for style in ebm_summary_alignment.values():
                    self.assertEqual(style["textAlign"], "center")
                    self.assertEqual(style["justifyContent"], "center")
                page.locator("#gbmEbmGainSummaryGrid .tabulator-row", has_text="Age x Segment").click(button="right")
                page.locator("#gbmFeatureContextMenu:not([hidden])").wait_for(timeout=10_000)
                ebm_dim2_context_labels = page.locator("#gbmFeatureContextMenu [role='menuitem']").evaluate_all(
                    "(items) => items.map((item) => item.textContent.trim())"
                )
                self.assertEqual(ebm_dim2_context_labels, ["Allow interaction pair", "Go to SHAP"])
                page.locator("#gbmFeatureContextMenu [role='menuitem']", has_text="Allow interaction pair").click()
                self.assertEqual(page.locator("#gbmFeatureInteractionPairButton").text_content(), "Interaction pairs (1)")
                page.locator("#gbmEbmGainSummaryGrid .tabulator-row", has_text="Age x Segment").click(button="right")
                page.locator("#gbmFeatureContextMenu:not([hidden])").wait_for(timeout=10_000)
                with page.expect_response(lambda response: "/api/gbm/models/" in response.url and "/shap/plot" in response.url and response.request.method == "POST", timeout=10_000):
                    page.locator("#gbmFeatureContextMenu [role='menuitem']", has_text="Go to SHAP").click()
                page.wait_for_function(
                    """
                    () => document.querySelector("#gbmShapFeatureList1 .feature.active")?.textContent.includes("Age")
                      && document.querySelector("#gbmShapFeatureList2 .feature.active")?.textContent.includes("Segment")
                    """,
                    timeout=10_000,
                )
                page.get_by_role("button", name="Features and parameters").click()
                page.wait_for_function(
                    """
                    () => {
                      const firstRow = document.querySelector("#gbmEbmGainSummaryGrid .tabulator-row");
                      return document.querySelector("input[name='gbmFeatureMetric']:checked")?.value === "gain_ebm"
                        && document.querySelector("#gbmFeatureGrid")?.classList.contains("hidden")
                        && !document.querySelector("#gbmEbmGainSummaryGrid")?.classList.contains("hidden")
                        && firstRow?.textContent.includes("Age x Segment")
                        && firstRow?.textContent.includes("100.0%");
                    }
                    """,
                    timeout=10_000,
                )
                page.locator("input[name='gbmFeatureMetric'][value='gain']").check(force=True)
                page.wait_for_function(
                    """
                    () => {
                      const headers = [...document.querySelectorAll("#gbmFeatureGrid .tabulator-col-title")]
                        .map((node) => node.textContent.trim()).filter(Boolean);
                      return document.querySelector("input[name='gbmFeatureMetric']:checked")?.value === "gain"
                        && !document.querySelector("#gbmFeatureGrid")?.classList.contains("hidden")
                        && document.querySelector("#gbmEbmGainSummaryGrid")?.classList.contains("hidden")
                        && headers.includes("Gain");
                    }
                    """,
                    timeout=10_000,
                )
                assert_feature_heading_matches_checked(2)
                page.wait_for_function(
                    """
                    () => {
                      const chart = window.echarts.getInstanceByDom(document.querySelector("#gbmEvaluationChart"));
                      const option = chart?.getOption();
                      return option?.xAxis?.[0]?.max === 3000
                        && option.xAxis[0].interval === 100
                        && option.series?.every((series) => series.data.length <= 1503)
                        && Number.isFinite(option.yAxis?.[0]?.max)
                        && option.yAxis[0].max < option.series?.[1]?.data?.[0]?.[1];
                    }
                    """,
                    timeout=10_000,
                )
                switched_chart_options = page.evaluate(
                    """
                    () => {
                      const chart = window.echarts.getInstanceByDom(document.querySelector("#gbmEvaluationChart"));
                      const option = chart.getOption();
                      return {
                        allChecked: document.querySelector("input[name='gbmEvaluationViewMode'][value='all']")?.checked,
                        tailChecked: document.querySelector("input[name='gbmEvaluationViewMode'][value='tail']")?.checked,
                        xMin: option.xAxis[0].min,
                        xMax: option.xAxis[0].max,
                        xInterval: option.xAxis[0].interval,
                        xAxisLabel100: option.xAxis[0].axisLabel.formatter(100),
                        xAxisLabel500: option.xAxis[0].axisLabel.formatter(500),
                        xAxisLabel1000: option.xAxis[0].axisLabel.formatter(1000),
                        seriesLengths: option.series.map((series) => series.data.length),
                        firstTestIteration: option.series[1].data[0][0],
                        lastTestIteration: option.series[1].data.at(-1)[0],
                        hasBestIteration: option.series[1].data.some((point) => point[0] === 3),
                        yMax: option.yAxis[0].max,
                        firstTestValue: option.series[1].data[0][1],
                        tooltip: option.tooltip[0].formatter([
                          { axisValue: 2.2, seriesName: "train", marker: "", value: [2.2, 0.12345] },
                          { axisValue: 2.2, seriesName: "test", marker: "", value: [2.2, 0.23456] },
                        ]),
                      };
                    }
                    """
                )
                self.assertTrue(switched_chart_options["allChecked"])
                self.assertFalse(switched_chart_options["tailChecked"])
                self.assertEqual(switched_chart_options["xMin"], 0)
                self.assertEqual(switched_chart_options["xMax"], 3000)
                self.assertEqual(switched_chart_options["xInterval"], 100)
                self.assertEqual(switched_chart_options["xAxisLabel100"], "")
                self.assertEqual(switched_chart_options["xAxisLabel500"], "500")
                self.assertEqual(switched_chart_options["xAxisLabel1000"], "1,000")
                self.assertTrue(all(length <= 1503 for length in switched_chart_options["seriesLengths"]))
                self.assertEqual(switched_chart_options["firstTestIteration"], 1)
                self.assertEqual(switched_chart_options["lastTestIteration"], 3000)
                self.assertTrue(switched_chart_options["hasBestIteration"])
                self.assertLess(switched_chart_options["yMax"], switched_chart_options["firstTestValue"])
                self.assertIn("<strong>Iteration:</strong> 2", switched_chart_options["tooltip"])
                page.locator("input[name='gbmEvaluationViewMode'][value='tail']").check()
                page.wait_for_function(
                    """
                    () => {
                      const chart = window.echarts.getInstanceByDom(document.querySelector("#gbmEvaluationChart"));
                      const option = chart?.getOption();
                      return option?.xAxis?.[0]?.min === 2876
                        && option.xAxis[0].max === 3000
                        && option.series?.[1]?.data?.[0]?.[0] === 2876
                        && option.series[1].data.at(-1)[0] === 3000;
                    }
                    """,
                    timeout=10_000,
                )
                tail_chart_options = page.evaluate(
                    """
                    () => {
                      const chart = window.echarts.getInstanceByDom(document.querySelector("#gbmEvaluationChart"));
                      const option = chart.getOption();
                      const testValues = option.series[1].data.map((point) => point[1]);
                      return {
                        tailChecked: document.querySelector("input[name='gbmEvaluationViewMode'][value='tail']")?.checked,
                        xMin: option.xAxis[0].min,
                        xMax: option.xAxis[0].max,
                        xInterval: option.xAxis[0].interval,
                        seriesLengths: option.series.map((series) => series.data.length),
                        yMin: option.yAxis[0].min,
                        yMax: option.yAxis[0].max,
                        testMin: Math.min(...testValues),
                        testMax: Math.max(...testValues),
                        testRange: Math.max(...testValues) - Math.min(...testValues),
                      };
                    }
                    """
                )
                self.assertTrue(tail_chart_options["tailChecked"])
                self.assertEqual(tail_chart_options["xMin"], 2876)
                self.assertEqual(tail_chart_options["xMax"], 3000)
                self.assertEqual(tail_chart_options["xMax"] - tail_chart_options["xMin"], 124)
                self.assertLessEqual(max(tail_chart_options["seriesLengths"]), 125)
                self.assertLessEqual(tail_chart_options["yMin"], tail_chart_options["testMin"])
                self.assertGreaterEqual(tail_chart_options["yMax"], tail_chart_options["testMax"])
                self.assertLess(
                    tail_chart_options["yMax"] - tail_chart_options["yMin"],
                    tail_chart_options["testRange"] * 1.5,
                )
                page.locator("input[name='gbmEvaluationViewMode'][value='all']").check()
                page.wait_for_function(
                    """
                    () => {
                      const chart = window.echarts.getInstanceByDom(document.querySelector("#gbmEvaluationChart"));
                      const option = chart?.getOption();
                      return option?.xAxis?.[0]?.min === 0 && option.xAxis[0].max === 3000;
                    }
                    """,
                    timeout=10_000,
                )
                page.locator("input[name='gbmEvaluationViewMode'][value='tail']").check()
                page.get_by_role("button", name="Model navigator").click()
                page.locator("#gbmModelGrid .tabulator-row", has_text="Second smoke model").click()
                page.evaluate("() => { window.prompt = () => 'renamed-smoke-model'; }")
                page.locator("#gbmRenameModelBtn").click()
                page.locator("#gbmModelGrid .tabulator-row", has_text="renamed-smoke-model").wait_for(timeout=10_000)
                page.locator("#gbmModelSelectedMeta", has_text="renamed-smoke-model").wait_for(timeout=10_000)
                page.evaluate("() => { window.confirm = () => true; }")
                page.locator("#gbmModelGrid .tabulator-row", has_text="renamed-smoke-model").click()
                page.locator("#gbmDeleteModelBtn").click()
                page.wait_for_function(
                    """
                    () => !document.body.textContent.includes("renamed-smoke-model")
                      && document.querySelector("#gbmModelSelectedMeta")?.textContent.includes("Browser smoke model")
                    """,
                    timeout=10_000,
                )
                self.assertEqual(page.locator("#gbmModelGrid .tabulator-row").count(), 1)
                page.get_by_role("button", name="Features and parameters").click()
                self.assertEqual(
                    page.locator("#gbmParameterGrid .tabulator-row", has_text="learning_rate").locator(".tabulator-cell[tabulator-field='value']").text_content(),
                    "0.11",
                )
                restored_scenario = feature_scenario_state()
                self.assertEqual(restored_scenario["value"], "")
                self.assertIn("old_scenario (1; trained; missing from spec)", restored_scenario["rows"])
                self.assertIn("old_scenario (1; trained; missing from spec)", restored_scenario["activeRows"])
                assert_feature_heading_matches_checked(2)
                self.assertTrue(page.locator("input[name='gbmTrainingMode'][value='normal']").is_checked())
                self.assertTrue(page.locator("input[name='gbmEvaluationViewMode'][value='tail']").is_checked())
                page.locator("input[name='gbmEvaluationViewMode'][value='all']").check()
                live_job_status = {"value": "running"}
                train_payload = {"value": None}

                def train_route(route: Any) -> None:
                    train_payload["value"] = json.loads(route.request.post_data or "{}")
                    route.fulfill(
                        status=200,
                        content_type="application/json",
                        body=json.dumps(
                            {
                                "job_id": "live-job",
                                "status": "queued",
                                "created_at": "2026-05-25T00:00:00Z",
                                "updated_at": "2026-05-25T00:00:00Z",
                                "result": None,
                                "error": None,
                                "progress": None,
                            }
                        ),
                    )

                def job_route(route: Any) -> None:
                    if live_job_status["value"] == "succeeded":
                        payload = {
                            "job_id": "live-job",
                            "status": "succeeded",
                            "created_at": "2026-05-25T00:00:00Z",
                            "updated_at": "2026-05-25T00:00:01Z",
                            "result": {"sources": {}},
                            "error": None,
                            "progress": {
                                "phase": "succeeded",
                                "message": "GBM training complete",
                                "iteration": 2,
                                "total_iterations": 10,
                                "percent": 100,
                                "metric": "gamma",
                                "latest": [{"dataset": "test", "metric": "gamma", "value": 7.2}],
                                "evaluation": {"training": {"gamma": [7.4, 7.3]}, "test": {"gamma": [7.35, 7.2]}},
                            },
                        }
                    elif live_job_status["value"] == "failed":
                        payload = {
                            "job_id": "live-job",
                            "status": "failed",
                            "created_at": "2026-05-25T00:00:00Z",
                            "updated_at": "2026-05-25T00:00:01Z",
                            "result": None,
                            "error": "Synthetic GBM training failure",
                            "progress": {
                                "phase": "failed",
                                "message": "Synthetic GBM training failure",
                            },
                        }
                    else:
                        payload = {
                            "job_id": "live-job",
                            "status": "running",
                            "created_at": "2026-05-25T00:00:00Z",
                            "updated_at": "2026-05-25T00:00:01Z",
                            "result": None,
                            "error": None,
                            "progress": {
                                "phase": "training",
                                "message": "training, tree 2/10, test gamma 7.2",
                                "iteration": 2,
                                "total_iterations": 10,
                                "percent": 18,
                                "metric": "gamma",
                                "grid_model_number": 1,
                                "grid_model_count": 25,
                                "latest": [{"dataset": "test", "metric": "gamma", "value": 7.2}],
                                "evaluation": {"training": {"gamma": [7.4, 7.3]}, "test": {"gamma": [7.35, 7.2]}},
                            },
                        }
                    route.fulfill(status=200, content_type="application/json", body=json.dumps(payload))

                page.route("**/api/gbm/train", train_route)
                page.route("**/api/gbm/jobs/live-job", job_route)
                self.assertEqual(page.locator("#gbmFeatureInteractionPairButton").text_content(), "Interaction pairs")
                self.assertFalse(page.locator("#gbmFeatureInteractionPairButton").is_disabled())
                page.locator("#gbmFeatureInteractionPairButton").click()
                page.locator("#gbmInteractionPairLeft").select_option("Age")
                page.locator("#gbmInteractionPairRight").select_option("Segment")
                page.locator("#gbmInteractionPairAdd").click()
                page.wait_for_function(
                    "() => document.querySelector('#gbmFeatureInteractionPairButton')?.textContent.trim() === 'Interaction pairs (1)'",
                    timeout=10_000,
                )
                page.keyboard.press("Escape")
                page.wait_for_function(
                    "() => document.querySelector('#gbmFeatureInteractionPairMenu')?.classList.contains('hidden')",
                    timeout=10_000,
                )
                page.wait_for_function(
                    """
                    () => {
                      const lockState = (name) => {
                        const row = [...document.querySelectorAll("#gbmFeatureGrid .tabulator-row")]
                          .find((item) => item.querySelector(".tabulator-cell[tabulator-field='name']")?.textContent.includes(name));
                        const cell = row?.querySelector(".tabulator-cell[tabulator-field='name']");
                        return {
                          pair: Boolean(cell?.querySelector(".gbm-pair-interaction-lock")),
                          singleton: Boolean(cell?.querySelector(".gbm-feature-interaction-lock")),
                          subscript: cell?.querySelector(".gbm-pair-interaction-lock .gbm-interaction-lock-subscript")?.textContent.trim() || "",
                        };
                      };
                      const age = lockState("Age");
                      const segment = lockState("Segment");
                      return age.pair && !age.singleton && age.subscript === "2"
                        && segment.pair && !segment.singleton && segment.subscript === "2";
                    }
                    """,
                    timeout=10_000,
                )
                page.locator("#gbmFeatureGrid .tabulator-row", has_text="Segment").locator(".tabulator-cell[tabulator-field='name']").click(button="right")
                page.locator("#gbmFeatureContextMenu [role='menuitem']", has_text="Toggle interaction constraint").click()
                page.wait_for_function(
                    """
                    () => {
                      const rows = [...document.querySelectorAll("#gbmFeatureGrid .tabulator-row")];
                      const featureCell = (name) => rows
                        .find((item) => item.querySelector(".tabulator-cell[tabulator-field='name']")?.textContent.includes(name))
                        ?.querySelector(".tabulator-cell[tabulator-field='name']");
                      const ageCell = featureCell("Age");
                      const segmentCell = featureCell("Segment");
                      return Boolean(ageCell?.querySelector(".gbm-pair-interaction-lock"))
                        && !ageCell?.querySelector(".gbm-feature-interaction-lock")
                        && Boolean(segmentCell?.querySelector(".gbm-feature-interaction-lock"))
                        && !segmentCell?.querySelector(".gbm-pair-interaction-lock");
                    }
                    """,
                    timeout=10_000,
                )
                page.locator("#gbmFeatureGrid .tabulator-row", has_text="Segment").locator(".tabulator-cell[tabulator-field='name']").click(button="right")
                page.locator("#gbmFeatureContextMenu [role='menuitem']", has_text="Toggle interaction constraint").click()
                page.wait_for_function(
                    """
                    () => {
                      const row = [...document.querySelectorAll("#gbmFeatureGrid .tabulator-row")]
                        .find((item) => item.querySelector(".tabulator-cell[tabulator-field='name']")?.textContent.includes("Segment"));
                      const cell = row?.querySelector(".tabulator-cell[tabulator-field='name']");
                      return Boolean(cell?.querySelector(".gbm-pair-interaction-lock"))
                        && !cell?.querySelector(".gbm-feature-interaction-lock")
                        && cell?.querySelector(".gbm-interaction-lock-subscript")?.textContent.trim() === "2";
                    }
                    """,
                    timeout=10_000,
                )
                page.evaluate(
                    """
                    () => {
                      window.__gbmBusyPointerMoves = 0;
                      document.addEventListener("pointermove", () => { window.__gbmBusyPointerMoves += 1; }, true);
                    }
                    """
                )
                page.locator("#gbmTrainBtn").hover()
                page.locator("#gbmTrainBtn").click()
                page.locator("#gbmTrainingStatus").get_by_text("training, tree 2/10, test gamma 7.2").wait_for(timeout=10_000)
                page.locator("#startupProgress.ready", has_text="Training GBM (1/25)...").wait_for(timeout=10_000)
                gbm_busy_button = page.locator("#gbmTrainBtn").evaluate(
                    """
                    (button) => {
                      const style = getComputedStyle(button);
                      const spinner = getComputedStyle(button, "::before");
                      return {
                        text: button.textContent.trim(),
                        disabled: button.disabled,
                        ariaBusy: button.getAttribute("aria-busy"),
                        training: button.classList.contains("training"),
                        cursor: style.cursor,
                        background: style.backgroundColor,
                        spinnerContent: spinner.content,
                        spinnerWidth: spinner.width,
                        spinnerAnimation: spinner.animationName,
                      };
                    }
                    """
                )
                self.assertEqual(gbm_busy_button["text"], "Training...")
                self.assertTrue(gbm_busy_button["disabled"])
                self.assertEqual(gbm_busy_button["ariaBusy"], "true")
                self.assertTrue(gbm_busy_button["training"])
                self.assertEqual(gbm_busy_button["cursor"], "pointer")
                self.assertEqual(gbm_busy_button["background"], "rgb(217, 119, 6)")
                self.assertEqual(gbm_busy_button["spinnerContent"], '""')
                self.assertEqual(gbm_busy_button["spinnerWidth"], "12px")
                self.assertEqual(gbm_busy_button["spinnerAnimation"], "model-busy-button-spin")
                gbm_pointer_moves_while_busy = page.evaluate("window.__gbmBusyPointerMoves")
                self.assertNotIn("feature_scenario", train_payload["value"])
                self.assertEqual(train_payload["value"]["feature_interaction_pairs"], [{"left": "Age", "right": "Segment"}])
                self.assertNotIn("feature_interaction_groupings", train_payload["value"])
                self.assertNotIn("feature_interaction_features", train_payload["value"])
                trained_features = {feature["name"]: feature for feature in train_payload["value"]["features"]}
                self.assertTrue(trained_features["Age"]["include"])
                self.assertTrue(trained_features["Segment"]["include"])
                page.wait_for_function(
                    """
                    () => {
                      const chart = window.echarts.getInstanceByDom(document.querySelector("#gbmEvaluationChart"));
                      const option = chart?.getOption();
                      return option?.title?.[0]?.text === "evaluation metric: gamma, test metric: 7.2, iteration: 2"
                        && option.xAxis?.[0]?.max === 10
                        && document.querySelector("input[name='gbmEvaluationViewMode'][value='all']")?.checked
                        && option.series?.length === 2
                        && option.series[0].data.length === 2;
                    }
                    """,
                    timeout=10_000,
                )
                live_job_status["value"] = "succeeded"
                page.locator("#gbmTrainBtn", has_text="Train GBM").wait_for(timeout=10_000)
                page.locator("#startupProgress.ready", has_text="Ready").wait_for(timeout=10_000)
                gbm_ready_button = page.locator("#gbmTrainBtn").evaluate(
                    """
                    (button) => {
                      const style = getComputedStyle(button);
                      const spinner = getComputedStyle(button, "::before");
                      return {
                        text: button.textContent.trim(),
                        disabled: button.disabled,
                        ariaBusy: button.getAttribute("aria-busy"),
                        training: button.classList.contains("training"),
                        cursor: style.cursor,
                        background: style.backgroundColor,
                        spinnerContent: spinner.content,
                      };
                    }
                    """
                )
                self.assertEqual(gbm_ready_button["text"], "Train GBM")
                self.assertFalse(gbm_ready_button["disabled"])
                self.assertIsNone(gbm_ready_button["ariaBusy"])
                self.assertFalse(gbm_ready_button["training"])
                self.assertEqual(gbm_ready_button["cursor"], "pointer")
                self.assertEqual(gbm_ready_button["background"], "rgb(21, 128, 61)")
                self.assertEqual(gbm_ready_button["spinnerContent"], "none")
                self.assertEqual(page.evaluate("window.__gbmBusyPointerMoves"), gbm_pointer_moves_while_busy)

                live_job_status["value"] = "failed"
                page.locator("#gbmTrainBtn").click()
                page.locator("#gbmNotice", has_text="Synthetic GBM training failure").wait_for(timeout=10_000)
                gbm_failed_button = page.locator("#gbmTrainBtn").evaluate(
                    """
                    (button) => ({
                      text: button.textContent.trim(),
                      disabled: button.disabled,
                      ariaBusy: button.getAttribute("aria-busy"),
                      training: button.classList.contains("training"),
                      spinnerContent: getComputedStyle(button, "::before").content,
                    })
                    """
                )
                self.assertEqual(gbm_failed_button["text"], "Train GBM")
                self.assertFalse(gbm_failed_button["disabled"])
                self.assertIsNone(gbm_failed_button["ariaBusy"])
                self.assertFalse(gbm_failed_button["training"])
                self.assertEqual(gbm_failed_button["spinnerContent"], "none")
                page.evaluate(
                    """
                    () => {
                      const notice = document.querySelector("#gbmNotice");
                      if (!notice) return;
                      notice.textContent = "";
                      notice.classList.add("hidden");
                    }
                    """
                )
                page.unroute("**/api/gbm/train", train_route)
                page.unroute("**/api/gbm/jobs/live-job", job_route)
                gbm_top_before = page.locator(".gbm-tool").evaluate("node => node.getBoundingClientRect().top")
                page.locator("#gbmSampleStatus").get_by_text("SAMPLE column found").wait_for(timeout=10_000)
                self.assertEqual(page.locator("#gbmCreateSampleBtn").count(), 0)
                page.locator("#gbmClearFeaturesBtn").click()
                page.wait_for_function(
                    "() => document.querySelectorAll('#gbmFeatureGrid .gbm-use-checkbox:checked').length === 0",
                    timeout=10_000,
                )
                assert_feature_heading_matches_checked(0)
                page.locator("#gbmSelectFeaturesBtn").click()
                page.wait_for_function(
                    "() => document.querySelectorAll('#gbmFeatureGrid .gbm-use-checkbox:checked').length > 0",
                    timeout=10_000,
                )
                assert_feature_heading_matches_checked()
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
                assert_feature_heading_matches_checked(0)
                page.locator("#gbmTrainBtn").click()
                page.locator("#gbmNotice").get_by_text("Choose at least one usable GBM feature").wait_for(timeout=10_000)
                self.assertTrue(page.locator("#status").evaluate("node => node.classList.contains('hidden')"))
                gbm_top_after_error = page.locator(".gbm-tool").evaluate("node => node.getBoundingClientRect().top")
                self.assertLessEqual(abs(gbm_top_before - gbm_top_after_error), 1)
                self.assertTrue(page.locator(".sidebar-metric-section").is_visible())
                self.assertTrue(page.locator(".sidebar-favourites-section").is_visible())
                self.assertTrue(page.locator("#actualNumerator").is_visible())
                self.assertTrue(page.locator("#denominator").is_visible())
                self.assertTrue(page.locator(".sidebar-filter-section").is_visible())
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
                        gridBottom: option.grid[0].bottom,
                        gridContainLabel: option.grid[0].containLabel,
                        xType: option.xAxis[0].type,
                        xInterval: option.xAxis[0].interval,
                        xMax: option.xAxis[0].max,
                        xAxisLabelHideOverlap: option.xAxis[0].axisLabel.hideOverlap,
                        xAxisLabelMargin: option.xAxis[0].axisLabel.margin,
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
                self.assertEqual(chart_options["gridBottom"], 20)
                self.assertTrue(chart_options["gridContainLabel"])
                self.assertEqual(chart_options["xType"], "value")
                self.assertEqual(chart_options["xInterval"], 1)
                self.assertEqual(chart_options["xMax"], 5)
                self.assertFalse(chart_options["xAxisLabelHideOverlap"])
                self.assertEqual(chart_options["xAxisLabelMargin"], 4)
                self.assertEqual(chart_options["yType"], "value")
                self.assertTrue(chart_options["yScale"])
                self.assertEqual(chart_options["seriesNames"], ["train", "test"])
                self.assertTrue(chart_options["showSymbol"])
                page.get_by_role("button", name="Tree viewer").click()
                page.locator("#gbmTreeSummaryGrid .tabulator-row").first.wait_for(timeout=10_000)
                page.locator("#gbmTreeChart svg.gbm-tree-svg").wait_for(state="attached", timeout=10_000)
                tree_state = page.evaluate(
                    """
                    async () => {
                      const chart = document.querySelector("#gbmTreeChart");
                      const plainFill = chart.querySelector("rect.gbm-tree-split-node")?.getAttribute("fill") || "";
                      const beforeZoom = chart.querySelector(".gbm-tree-viewport")?.getAttribute("transform") || "";
                      document.querySelector('[data-gbm-tree-zoom="in"]').click();
                      await new Promise((resolve) => setTimeout(resolve, 220));
                      const afterZoom = chart.querySelector(".gbm-tree-viewport")?.getAttribute("transform") || "";
                      chart.querySelector("svg.gbm-tree-svg")?.dispatchEvent(new WheelEvent("wheel", {
                        deltaY: -420,
                        bubbles: true,
                        cancelable: true,
                        clientX: chart.getBoundingClientRect().left + chart.getBoundingClientRect().width / 2,
                        clientY: chart.getBoundingClientRect().top + chart.getBoundingClientRect().height / 2,
                      }));
                      await new Promise((resolve) => setTimeout(resolve, 120));
                      const afterWheelZoom = chart.querySelector(".gbm-tree-viewport")?.getAttribute("transform") || "";
                      document.querySelector('[data-gbm-tree-palette="viridis"]').click();
                      await new Promise((resolve) => setTimeout(resolve, 50));
                      const afterPaletteZoom = chart.querySelector(".gbm-tree-viewport")?.getAttribute("transform") || "";
                      const viridisFill = chart.querySelector("rect.gbm-tree-split-node")?.getAttribute("fill") || "";
                      const rootLabel = chart.querySelector(".gbm-tree-node-label");
                      const rootLabelSpans = [...(rootLabel?.querySelectorAll("tspan") || [])];
                      return {
                        rows: document.querySelectorAll("#gbmTreeSummaryGrid .tabulator-row").length,
                        selectedRows: document.querySelectorAll("#gbmTreeSummaryGrid .tabulator-row.tabulator-selected").length,
                        selectedTree: document.querySelector("#gbmTreeSummaryGrid .tabulator-row.tabulator-selected .tabulator-cell[tabulator-field='tree']")?.textContent.trim() || "",
                        detailSummary: document.querySelector("#gbmTreeDetailSummary")?.textContent.replace(/\\s+/g, " ").trim() || "",
                        summaryInsideChart: Boolean(document.querySelector("#gbmTreeChart > #gbmTreeDetailSummary")),
                        summaryWidth: document.querySelector(".gbm-tree-summary-panel")?.getBoundingClientRect().width || 0,
                        treeColumnWidth: document.querySelector("#gbmTreeSummaryGrid .tabulator-col[tabulator-field='tree']")?.getBoundingClientRect().width || 0,
                        dimColumnWidth: document.querySelector("#gbmTreeSummaryGrid .tabulator-col[tabulator-field='dim']")?.getBoundingClientRect().width || 0,
                        headers: [...document.querySelectorAll("#gbmTreeSummaryGrid .tabulator-col-title")]
                          .map((node) => node.textContent.trim()).filter(Boolean),
                        splitNodes: chart.querySelectorAll("rect.gbm-tree-split-node").length,
                        leafNodes: chart.querySelectorAll("ellipse.gbm-tree-leaf-node").length,
                        nodeTitles: chart.querySelectorAll(".gbm-tree-node title").length,
                        edgeLabels: [...chart.querySelectorAll(".gbm-tree-edge-label")].map((node) => node.textContent.trim()),
                        arrowheads: chart.querySelectorAll("marker#gbmTreeArrow").length,
                        zoomButtons: document.querySelectorAll("[data-gbm-tree-zoom]").length,
                        beforeZoom,
                        afterZoom,
                        afterWheelZoom,
                        afterPaletteZoom,
                        textFills: [...chart.querySelectorAll(".gbm-tree-node-label")].map((node) => node.getAttribute("fill")),
                        rootLabelLines: rootLabelSpans.map((node) => node.textContent.trim()),
                        rootLabelWeights: rootLabelSpans.map((node) => node.getAttribute("font-weight")),
                        plainFill,
                        viridisFill,
                      };
                    }
                    """,
                )
                self.assertGreaterEqual(tree_state["rows"], 1)
                self.assertEqual(tree_state["selectedRows"], 1)
                self.assertEqual(tree_state["selectedTree"], "0")
                self.assertIn("Tree 0", tree_state["detailSummary"])
                self.assertIn("Dimensionality: 2", tree_state["detailSummary"])
                self.assertIn("Tree features:", tree_state["detailSummary"])
                self.assertIn("Tree gain: 7", tree_state["detailSummary"])
                self.assertTrue(tree_state["summaryInsideChart"])
                self.assertGreaterEqual(tree_state["summaryWidth"], 420)
                self.assertLessEqual(tree_state["treeColumnWidth"], 50)
                self.assertLessEqual(tree_state["dimColumnWidth"], 46)
                self.assertEqual(tree_state["headers"], ["tree", "dim", "features", "gain"])
                self.assertGreaterEqual(tree_state["splitNodes"], 1)
                self.assertGreaterEqual(tree_state["leafNodes"], 2)
                self.assertEqual(tree_state["nodeTitles"], 0)
                self.assertIn("<= 35", tree_state["edgeLabels"])
                self.assertIn("else", tree_state["edgeLabels"])
                self.assertEqual(tree_state["arrowheads"], 1)
                self.assertEqual(tree_state["zoomButtons"], 3)
                self.assertNotEqual(tree_state["beforeZoom"], tree_state["afterZoom"])
                self.assertNotEqual(tree_state["afterZoom"], tree_state["afterWheelZoom"])
                self.assertEqual(tree_state["afterWheelZoom"], tree_state["afterPaletteZoom"])
                self.assertIn("#111827", tree_state["textFills"])
                self.assertTrue(set(tree_state["textFills"]).issubset({"#111827", "#ffffff"}))
                self.assertEqual(tree_state["rootLabelLines"][:2], ["Tree 0", "Age"])
                self.assertEqual(tree_state["rootLabelWeights"][:2], ["700", "700"])
                self.assertIn("400", tree_state["rootLabelWeights"][2:])
                self.assertNotEqual(tree_state["plainFill"], tree_state["viridisFill"])
                resizer = page.locator("#gbmTreeResizer")
                resizer_box = resizer.bounding_box()
                self.assertIsNotNone(resizer_box)
                summary_width_before = page.locator(".gbm-tree-summary-panel").evaluate("node => node.getBoundingClientRect().width")
                if summary_width_before > 470:
                    page.mouse.move(resizer_box["x"] + resizer_box["width"] / 2, resizer_box["y"] + 24)
                    page.mouse.down()
                    page.mouse.move(resizer_box["x"] + resizer_box["width"] / 2 - 80, resizer_box["y"] + 24)
                    page.mouse.up()
                    summary_width_after = page.locator(".gbm-tree-summary-panel").evaluate("node => node.getBoundingClientRect().width")
                    self.assertLess(summary_width_after, summary_width_before - 40)
                else:
                    self.assertGreaterEqual(summary_width_before, 420)
                page.get_by_role("button", name="Model navigator").click()
                page.locator("#gbmModelGrid .tabulator-row").first.wait_for(timeout=10_000)
                navigator_state = page.evaluate(
                    """
                    () => {
                      const headers = [...document.querySelectorAll("#gbmModelGrid .tabulator-col-title")]
                        .map((node) => node.textContent.trim()).filter(Boolean);
                      const rows = [...document.querySelectorAll("#gbmModelGrid .tabulator-row")];
                      const activeRow = document.querySelector("#gbmModelGrid .gbm-model-active-dot")?.closest(".tabulator-row");
                      const firstRow = rows.find((row) => row.textContent.includes("Browser smoke model"));
                      const firstCells = [...(firstRow?.querySelectorAll(".tabulator-cell") || [])].map((node) => node.textContent.trim());
                      const cell = document.querySelector("#gbmModelGrid .tabulator-cell");
                      return {
                        headers,
                        rowCount: rows.length,
                        activeDots: document.querySelectorAll("#gbmModelGrid .gbm-model-active-dot").length,
                        activeText: activeRow?.textContent || "",
                        firstCells,
                        fontSize: cell ? getComputedStyle(cell).fontSize : "",
                        lineHeight: cell ? getComputedStyle(cell).lineHeight : "",
                        wrapped: Boolean(document.querySelector(".gbm-model-navigator")),
                        activeTab: document.querySelector("[data-gbm-tab].active")?.textContent.trim() || "",
                        fallbackRows: document.querySelectorAll("#gbmModelFallback [data-gbm-model-row]").length,
                        hasDeletedModel: document.body.textContent.includes("renamed-smoke-model"),
                      };
                    }
                    """
                )
                self.assertEqual(
                    navigator_state["headers"],
                    [
                        "Model", "Created", "Response", "Weight", "Objective", "Metric", "Mode", "Constraints", "Train", "Best iter.",
                        "tr@best", "te@best", "n_iter", "lr", "leaves", "depth", "min_leaf", "ES", "Run time", "Sample",
                    ],
                )
                self.assertEqual(navigator_state["rowCount"], 1)
                self.assertEqual(navigator_state["activeDots"], 1)
                self.assertIn("Browser smoke model", navigator_state["activeText"])
                self.assertEqual(navigator_state["fontSize"], "11px")
                self.assertIn("actualNumerator", navigator_state["firstCells"])
                self.assertIn("denominator", navigator_state["firstCells"])
                self.assertIn("Normal", navigator_state["firstCells"])
                self.assertIn("SAMPLE", navigator_state["firstCells"])
                self.assertIn("7.31", navigator_state["firstCells"])
                self.assertIn("7.3022", navigator_state["firstCells"])
                self.assertIn("77", navigator_state["firstCells"])
                self.assertIn("0.11", navigator_state["firstCells"])
                self.assertIn("25", navigator_state["firstCells"])
                self.assertIn("1.2s", navigator_state["firstCells"])
                self.assertTrue(navigator_state["wrapped"])
                self.assertEqual(navigator_state["activeTab"], "Model navigator")
                self.assertEqual(navigator_state["fallbackRows"], 0)
                self.assertFalse(navigator_state["hasDeletedModel"])
                page.locator(".dataset-meta-gbm-link").click()
                page.wait_for_function(
                    """
                    () => document.querySelector("[data-gbm-tab].active")?.textContent.trim() === "Model navigator"
                      && document.querySelectorAll("#gbmModelGrid .tabulator-row").length === 1
                      && document.querySelectorAll("#gbmModelFallback [data-gbm-model-row]").length === 0
                    """,
                    timeout=10_000,
                )
                page.locator("#lineBarTool").click()
                page.locator("#lineBarTool.active").wait_for(timeout=10_000)
                page.locator(".dataset-meta-gbm-link").click()
                page.wait_for_function(
                    """
                    () => document.querySelector("#gbmTool.active")
                      && document.querySelector("[data-gbm-tab].active")?.textContent.trim() === "Model navigator"
                      && document.querySelectorAll("#gbmModelGrid .tabulator-row").length === 1
                      && document.querySelectorAll("#gbmModelFallback [data-gbm-model-row]").length === 0
                    """,
                    timeout=10_000,
                )
                page.locator("#gbmModelGrid .tabulator-row", has_text="Browser smoke model").click()
                page.wait_for_function(
                    """
                    () => [...document.querySelectorAll("#gbmModelGrid .tabulator-row.tabulator-selected")]
                      .some((row) => row.textContent.includes("Browser smoke model"))
                    """,
                    timeout=10_000,
                )
                page.get_by_role("button", name="Features and parameters").click()
                page.wait_for_function(
                    """
                    () => [...document.querySelectorAll("#gbmParameterGrid .tabulator-row")]
                      .find((row) => row.textContent.includes("learning_rate"))
                      ?.querySelector(".tabulator-cell[tabulator-field='value']")
                      ?.textContent.trim() === "0.11"
                    """,
                    timeout=10_000,
                )
                self.assertEqual(
                    page.locator("#gbmParameterGrid .tabulator-row", has_text="num_iterations").locator(".tabulator-cell[tabulator-field='value']").text_content(),
                    "77",
                )
                parameter_dropdown_before = page.evaluate(
                    """
                    () => {
                      const row = [...document.querySelectorAll("#gbmParameterGrid .tabulator-row")]
                        .find((item) => item.querySelector(".tabulator-cell[tabulator-field='name']")?.textContent.trim() === "objective");
                      const cell = row?.querySelector(".tabulator-cell[tabulator-field='value']");
                      if (!cell) return null;
                      const range = document.createRange();
                      range.selectNodeContents(cell);
                      const rect = range.getBoundingClientRect();
                      range.detach();
                      return { text: cell.textContent.trim(), textLeft: rect.left };
                    }
                    """
                )
                self.assertIsNotNone(parameter_dropdown_before)
                assert parameter_dropdown_before is not None
                self.assertEqual(parameter_dropdown_before["text"], "gamma")
                page.locator("#gbmParameterGrid .tabulator-row", has_text="objective").locator(".tabulator-cell[tabulator-field='value']").click()
                page.wait_for_function(
                    """
                    () => {
                      const popup = document.querySelector(".tabulator-popup-container.tabulator-edit-list");
                      const rect = popup?.getBoundingClientRect();
                      return Boolean(popup && rect.width > 0 && rect.height > 0);
                    }
                    """,
                    timeout=10_000,
                )
                parameter_dropdown_after = page.evaluate(
                    """
                    () => {
                      const row = [...document.querySelectorAll("#gbmParameterGrid .tabulator-row")]
                        .find((item) => item.querySelector(".tabulator-cell[tabulator-field='name']")?.textContent.trim() === "objective");
                      const cell = row?.querySelector(".tabulator-cell[tabulator-field='value']");
                      const input = cell?.querySelector("input.gbm-parameter-list-editor");
                      const popup = document.querySelector(".tabulator-popup-container.tabulator-edit-list");
                      const inputRect = input?.getBoundingClientRect();
                      const popupRect = popup?.getBoundingClientRect();
                      const popupStyle = popup ? getComputedStyle(popup) : null;
                      return {
                        editing: Boolean(cell?.classList.contains("tabulator-editing")),
                        inputValue: input?.value || "",
                        inputLeft: inputRect?.left || 0,
                        popupVisible: Boolean(popup && popupRect.width > 0 && popupRect.height > 0),
                        popupClientHeight: popup?.clientHeight || 0,
                        popupScrollHeight: popup?.scrollHeight || 0,
                        popupFontSize: popupStyle?.fontSize || "",
                        popupItems: [...(popup?.querySelectorAll(".tabulator-edit-list-item") || [])]
                          .map((item) => item.textContent.trim()),
                      };
                    }
                    """
                )
                self.assertTrue(parameter_dropdown_after["editing"])
                self.assertTrue(parameter_dropdown_after["popupVisible"])
                self.assertEqual(parameter_dropdown_after["inputValue"], "gamma")
                self.assertLessEqual(abs(parameter_dropdown_after["inputLeft"] - parameter_dropdown_before["textLeft"]), 1)
                self.assertGreaterEqual(parameter_dropdown_after["popupClientHeight"] + 1, parameter_dropdown_after["popupScrollHeight"])
                self.assertEqual(parameter_dropdown_after["popupFontSize"], "11px")
                self.assertEqual(parameter_dropdown_after["popupItems"], sorted(parameter_dropdown_after["popupItems"], key=str.casefold))
                page.keyboard.press("Escape")
                page.wait_for_function(
                    "() => !document.querySelector('#gbmParameterGrid .tabulator-cell.tabulator-editing')",
                    timeout=10_000,
                )
                page.locator("#gbmParameterGrid .tabulator-row", has_text="num_iterations").locator(".tabulator-cell[tabulator-field='value']").click()
                page.wait_for_function(
                    """
                    () => Boolean(document.querySelector("#gbmParameterGrid .tabulator-cell[tabulator-field='value'].tabulator-editing input.gbm-parameter-input-editor"))
                    """,
                    timeout=10_000,
                )
                numeric_parameter_editor = page.evaluate(
                    """
                    () => {
                      const popupVisible = [...document.querySelectorAll(".tabulator-popup-container.tabulator-edit-list")]
                        .some((popup) => {
                          const rect = popup.getBoundingClientRect();
                          return rect.width > 0 && rect.height > 0;
                        });
                      const row = [...document.querySelectorAll("#gbmParameterGrid .tabulator-row")]
                        .find((item) => item.querySelector(".tabulator-cell[tabulator-field='name']")?.textContent.trim() === "num_iterations");
                      const cell = row?.querySelector(".tabulator-cell[tabulator-field='value']");
                      const input = cell?.querySelector("input.gbm-parameter-input-editor");
                      return {
                        editing: Boolean(cell?.classList.contains("tabulator-editing")),
                        inputType: input?.type || "",
                        inputValue: input?.value || "",
                        hasListEditor: Boolean(cell?.querySelector("input.gbm-parameter-list-editor")),
                        popupVisible,
                      };
                    }
                    """
                )
                self.assertTrue(numeric_parameter_editor["editing"])
                self.assertEqual(numeric_parameter_editor["inputType"], "text")
                self.assertEqual(numeric_parameter_editor["inputValue"], "77")
                self.assertFalse(numeric_parameter_editor["hasListEditor"])
                self.assertFalse(numeric_parameter_editor["popupVisible"])
                page.locator("#gbmParameterGrid .tabulator-cell[tabulator-field='value'].tabulator-editing input.gbm-parameter-input-editor").fill("{200, 300}")
                page.keyboard.press("Enter")
                page.wait_for_function(
                    """
                    () => {
                      const root = document.querySelector("#gbmGridSamples");
                      const input = document.querySelector("#gbmGridSampleInput");
                      const rect = root?.getBoundingClientRect();
                      return Boolean(root && input && !root.classList.contains("hidden") && rect.width > 0 && rect.height > 0);
                    }
                    """,
                    timeout=10_000,
                )
                feature_state = page.evaluate(
                    """
                    () => {
                      const metricField = document.querySelector("#gbmFeatureGrid .tabulator-cell[tabulator-field='gain']")
                        ? "gain"
                        : "mean_abs_shap";
                      function rowState(name) {
                        const row = [...document.querySelectorAll("#gbmFeatureGrid .tabulator-row")]
                          .find((item) => item.textContent.includes(name));
                        return {
                          checked: Boolean(row?.querySelector(".gbm-use-checkbox")?.checked),
                          monotonicity: row?.querySelector(".tabulator-cell[tabulator-field='monotonicity']")?.textContent.trim() || "",
                          metric: row?.querySelector(`.tabulator-cell[tabulator-field='${metricField}']`)?.textContent.trim() || "",
                        };
                      }
                      return { metricField, age: rowState("Age"), segment: rowState("Segment") };
                    }
                    """
                )
                self.assertFalse(feature_state["age"]["checked"])
                self.assertFalse(feature_state["segment"]["checked"])
                self.assertEqual(feature_state["age"]["monotonicity"], "Increasing")
                self.assertEqual(feature_state["age"]["metric"], "5.000" if feature_state["metricField"] == "gain" else "0.1830")
                page.evaluate(
                    """
                    () => {
                      const row = [...document.querySelectorAll("#gbmFeatureGrid .tabulator-row")]
                        .find((item) => item.textContent.includes("Segment"));
                      const checkbox = row?.querySelector(".gbm-use-checkbox");
                      if (!checkbox) throw new Error("Segment feature checkbox not found");
                      checkbox.checked = true;
                      checkbox.dispatchEvent(new Event("change", { bubbles: true }));
                    }
                    """
                )
                page.wait_for_function(
                    """
                    () => {
                      const row = [...document.querySelectorAll("#gbmFeatureGrid .tabulator-row")]
                        .find((item) => item.textContent.includes("Segment"));
                      const checked = document.querySelectorAll("#gbmFeatureGrid .gbm-use-checkbox:checked").length;
                      return Boolean(row?.querySelector(".gbm-use-checkbox")?.checked)
                        && document.querySelector("#gbmFeatureSectionTitle")?.textContent.trim() === `Features (${checked})`;
                    }
                    """,
                    timeout=10_000,
                )
                page.locator("#lineBarTool").click()
                page.locator("#lineBarTool.active").wait_for(timeout=10_000)
                page.locator("#gbmTool").click()
                page.locator("#gbmTool.active").wait_for(timeout=10_000)
                page.wait_for_function(
                    """
                    () => {
                      const segmentRow = [...document.querySelectorAll("#gbmFeatureGrid .tabulator-row")]
                        .find((item) => item.textContent.includes("Segment"));
                      const parameterRow = [...document.querySelectorAll("#gbmParameterGrid .tabulator-row")]
                        .find((item) => item.textContent.includes("num_iterations"));
                      const checked = document.querySelectorAll("#gbmFeatureGrid .gbm-use-checkbox:checked").length;
                      return Boolean(segmentRow?.querySelector(".gbm-use-checkbox")?.checked)
                        && parameterRow?.querySelector(".tabulator-cell[tabulator-field='value']")?.textContent.trim() === "{200, 300}"
                        && document.querySelector("#gbmFeatureSectionTitle")?.textContent.trim() === `Features (${checked})`;
                    }
                    """,
                    timeout=10_000,
                )
                layout = page.evaluate(
                    """
                    () => {
                        const visual = document.querySelector("#visualArea").getBoundingClientRect();
                        const tool = document.querySelector(".gbm-tool").getBoundingClientRect();
                        const grid = document.querySelector("#gbmFeatureGrid").getBoundingClientRect();
                        const right = document.querySelector(".gbm-right-panel").getBoundingClientRect();
                        const firstRow = document.querySelector("#gbmFeatureGrid .tabulator-row");
                        const normalRow = document.querySelector("#gbmFeatureGrid .tabulator-row:not(.gbm-feature-disabled):not(.gbm-feature-warning)");
                        const firstMetric = document.querySelector("#gbmFeatureGrid .tabulator-cell[tabulator-field='gain'], #gbmFeatureGrid .tabulator-cell[tabulator-field='mean_abs_shap']");
                        const tableHolder = document.querySelector("#gbmFeatureGrid .tabulator-tableholder");
                        const tab = document.querySelector(".gbm-tabs .tab");
                        const shap = document.querySelector("#gbmShapRows");
                        const shapLabel = document.querySelector("#gbmShapRows .gbm-shap-label");
                        const shapOptions = document.querySelector(".gbm-shap-options");
                        const gridSamples = document.querySelector("#gbmGridSamples");
                        const gridSampleInput = document.querySelector("#gbmGridSampleInput");
                        const firstShapInput = document.querySelector("input[name='gbmShapRows']");
                        const checkedShapOption = document.querySelector(".gbm-shap-option:has(input:checked)");
                        const mode = document.querySelector("#gbmTrainingMode");
                        const modeLabel = document.querySelector("#gbmTrainingMode .gbm-shap-label");
                        const checkedModeOption = document.querySelector(".gbm-mode-option:has(input:checked)");
                        const sampleStatus = document.querySelector("#gbmSampleStatus");
                        const train = document.querySelector("#gbmTrainBtn");
                        const featureTitle = document.querySelector("#gbmFeatureSectionTitle");
                        const controlTitle = document.querySelector(".gbm-parameter-controls-column .gbm-section-title");
                        const parameterTitle = document.querySelector(".gbm-parameter-table-column .gbm-section-title");
                        const parameterLayout = document.querySelector(".gbm-parameter-layout");
                        const parameterTableColumn = document.querySelector(".gbm-parameter-table-column");
                        const parameterControlsColumn = document.querySelector(".gbm-parameter-controls-column");
                        const parameterActions = document.querySelector(".gbm-parameter-controls-column .gbm-actions");
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
                            shapLabels: [...document.querySelectorAll("#gbmShapRows .gbm-shap-option span")].map((node) => node.textContent.trim()),
                            shapOptionsDisplay: shapOptions ? getComputedStyle(shapOptions).display : "",
                            shapOptionsDirection: shapOptions ? getComputedStyle(shapOptions).flexDirection : "",
                            gridSamplesVisible: Boolean(gridSamples && !gridSamples.classList.contains("hidden")),
                            gridSampleValue: gridSampleInput ? gridSampleInput.value : "",
                            gridSamplesTop: gridSamples ? Math.round(gridSamples.getBoundingClientRect().top) : 0,
                            gridSampleInputMin: gridSampleInput ? gridSampleInput.getAttribute("min") : "",
                            gridSampleInputStep: gridSampleInput ? gridSampleInput.getAttribute("step") : "",
                            shapInputOpacity: firstShapInput ? getComputedStyle(firstShapInput).opacity : "",
                            checkedShapBackground: checkedShapOption ? getComputedStyle(checkedShapOption).backgroundColor : "",
                            modeRadios: document.querySelectorAll("input[name='gbmTrainingMode']").length,
                            modeLabels: [...document.querySelectorAll("#gbmTrainingMode .gbm-mode-option span")].map((node) => node.textContent.trim()),
                            checkedModeValue: document.querySelector("input[name='gbmTrainingMode']:checked")?.value || "",
                            modeTitle: mode ? mode.getAttribute("title") : "",
                            checkedModeBackground: checkedModeOption ? getComputedStyle(checkedModeOption).backgroundColor : "",
                            featureCheckboxes: document.querySelectorAll("#gbmFeatureGrid .gbm-use-checkbox").length,
                            disabledFeatureCheckboxes: document.querySelectorAll("#gbmFeatureGrid .gbm-feature-disabled .gbm-use-checkbox").length,
                            rowHeight: firstRow ? firstRow.getBoundingClientRect().height : 0,
                            metricAlign: firstMetric ? getComputedStyle(firstMetric).textAlign : "",
                            metricJustifyContent: firstMetric ? getComputedStyle(firstMetric).justifyContent : "",
                            rowBackground: normalRow ? getComputedStyle(normalRow).backgroundColor : "",
                            holderBackground: tableHolder ? getComputedStyle(tableHolder).backgroundColor : "",
                            tabTop: tab ? Math.round(tab.getBoundingClientRect().top) : 0,
                            shapTop: shap ? Math.round(shap.getBoundingClientRect().top) : 0,
                            shapRight: shap ? Math.round(shap.getBoundingClientRect().right) : 0,
                            sampleStatusText: sampleStatus ? sampleStatus.textContent.trim() : "",
                            sampleTop: sampleStatus ? Math.round(sampleStatus.getBoundingClientRect().top) : 0,
                            trainTop: train ? Math.round(train.getBoundingClientRect().top) : 0,
                            featureTitleTop: featureTitle ? Math.round(featureTitle.getBoundingClientRect().top) : 0,
                            featureTitleFontSize: featureTitle ? getComputedStyle(featureTitle).fontSize : "",
                            featureTitleFontWeight: featureTitle ? getComputedStyle(featureTitle).fontWeight : "",
                            controlTitleTop: controlTitle ? Math.round(controlTitle.getBoundingClientRect().top) : 0,
                            controlTitleText: controlTitle ? controlTitle.textContent.trim() : "",
                            parameterTitleTop: parameterTitle ? Math.round(parameterTitle.getBoundingClientRect().top) : 0,
                            featureGridTop: grid ? Math.round(grid.top) : 0,
                            parameterGridTop: parameterGrid ? Math.round(parameterGrid.getBoundingClientRect().top) : 0,
                            parameterLayoutWidth: parameterLayout ? Math.round(parameterLayout.getBoundingClientRect().width) : 0,
                            parameterTableColumnWidth: parameterTableColumn ? Math.round(parameterTableColumn.getBoundingClientRect().width) : 0,
                            parameterControlsColumnWidth: parameterControlsColumn ? Math.round(parameterControlsColumn.getBoundingClientRect().width) : 0,
                            parameterActionsDirection: parameterActions ? getComputedStyle(parameterActions).flexDirection : "",
                            shapParentInControls: Boolean(shap?.closest(".gbm-parameter-controls-column")),
                            modeParentInControls: Boolean(mode?.closest(".gbm-parameter-controls-column")),
                            sampleParentInControls: Boolean(sampleStatus?.closest(".gbm-parameter-controls-column")),
                            trainParentInControls: Boolean(train?.closest(".gbm-parameter-controls-column")),
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
                            shapLabelFontSize: shapLabel ? getComputedStyle(shapLabel).fontSize : "",
                            shapLabelFontWeight: shapLabel ? getComputedStyle(shapLabel).fontWeight : "",
                            modeLabelFontSize: modeLabel ? getComputedStyle(modeLabel).fontSize : "",
                            modeLabelFontWeight: modeLabel ? getComputedStyle(modeLabel).fontWeight : "",
                        };
                    }
                    """
                )
                self.assertGreater(layout["toolWidth"], layout["visualWidth"] * 0.85)
                self.assertGreater(layout["gridWidth"], 320)
                self.assertGreater(layout["rightWidth"], 300)
                self.assertEqual(layout["shapRadios"], 4)
                self.assertEqual(layout["shapLabels"], ["0", "10k", "100k", "All"])
                self.assertEqual(layout["shapOptionsDisplay"], "flex")
                self.assertEqual(layout["shapOptionsDirection"], "row")
                self.assertTrue(layout["gridSamplesVisible"])
                self.assertEqual(layout["gridSampleValue"], "25")
                self.assertEqual(layout["gridSampleInputMin"], "1")
                self.assertEqual(layout["gridSampleInputStep"], "1")
                self.assertEqual(layout["shapInputOpacity"], "0")
                self.assertNotEqual(layout["checkedShapBackground"], layout["rowBackground"])
                self.assertEqual(layout["modeRadios"], 2)
                self.assertEqual(layout["modeLabels"], ["Normal", "EBM"])
                self.assertEqual(layout["checkedModeValue"], "normal")
                self.assertIn("2-leaf trees at learning rate 0.3", layout["modeTitle"])
                self.assertNotEqual(layout["checkedModeBackground"], layout["rowBackground"])
                self.assertGreater(layout["featureCheckboxes"], 0)
                self.assertEqual(layout["disabledFeatureCheckboxes"], 0)
                self.assertLess(layout["rowHeight"], 28)
                self.assertEqual(layout["metricAlign"], "center")
                self.assertEqual(layout["metricJustifyContent"], "center")
                self.assertEqual(layout["rowBackground"], layout["holderBackground"])
                self.assertTrue(layout["shapParentInControls"])
                self.assertTrue(layout["modeParentInControls"])
                self.assertTrue(layout["sampleParentInControls"])
                self.assertTrue(layout["trainParentInControls"])
                self.assertIn("SAMPLE column found", layout["sampleStatusText"])
                self.assertEqual(layout["controlTitleText"], "Control")
                self.assertLessEqual(abs(layout["featureTitleTop"] - layout["parameterTitleTop"]), 2)
                self.assertLessEqual(abs(layout["featureTitleTop"] - layout["controlTitleTop"]), 2)
                self.assertLessEqual(abs(layout["controlTitleTop"] - layout["parameterTitleTop"]), 2)
                self.assertGreaterEqual(layout["featureGridTop"], layout["parameterGridTop"])
                self.assertLessEqual(layout["featureGridTop"] - layout["parameterGridTop"], 32)
                self.assertEqual(layout["parameterActionsDirection"], "column")
                self.assertGreater(layout["parameterLayoutWidth"], 0)
                parameter_table_ratio = layout["parameterTableColumnWidth"] / layout["parameterLayoutWidth"]
                self.assertGreaterEqual(parameter_table_ratio, 0.55)
                self.assertLessEqual(parameter_table_ratio, 0.72)
                self.assertGreater(layout["parameterTableColumnWidth"], layout["parameterControlsColumnWidth"])
                self.assertGreater(layout["parameterControlsColumnWidth"], 120)
                self.assertLessEqual(abs(layout["trainTop"] - layout["parameterGridTop"]), 2)
                self.assertLess(layout["trainTop"], layout["sampleTop"])
                self.assertLess(layout["sampleTop"], layout["shapTop"])
                self.assertLess(layout["shapTop"], layout["gridSamplesTop"])
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
                self.assertEqual(layout["shapLabelFontSize"], layout["featureTitleFontSize"])
                self.assertEqual(layout["modeLabelFontSize"], layout["featureTitleFontSize"])
                self.assertEqual(layout["shapLabelFontWeight"], layout["featureTitleFontWeight"])
                self.assertEqual(layout["modeLabelFontWeight"], layout["featureTitleFontWeight"])
                self.assertIn("Feature", layout["featureHeaders"])
                self.assertIn("Use", layout["featureHeaders"])
                self.assertIn("Monotonicity", layout["featureHeaders"])
                self.assertEqual(sum(header in layout["featureHeaders"] for header in ("Gain", "SHAP")), 1)
                self.assertNotIn("Type", layout["featureHeaders"])
                page.get_by_role("button", name="Model navigator").click()
                page.locator("#gbmModelGrid .tabulator-row").first.wait_for(timeout=10_000)
                final_model_count = page.locator("#gbmModelGrid .tabulator-row").count()
                self.assertGreater(final_model_count, 0)
                page.locator("#gbmModelGrid .tabulator-row").first.click()
                if final_model_count > 1:
                    page.locator("#gbmModelGrid .tabulator-row").nth(final_model_count - 1).click(modifiers=["Shift"])
                page.evaluate("() => { window.confirm = () => true; }")
                page.locator("#gbmDeleteModelBtn").click()
                page.wait_for_function(
                    """
                    () => document.querySelectorAll("#gbmModelGrid .tabulator-row").length === 0
                      && document.querySelector(".dataset-meta-gbm-link")?.textContent.trim() === "GBMs (0)"
                    """,
                    timeout=10_000,
                )
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

                self.assertTrue(page.locator('.filter-selection-mode button[data-value="grouped"]').evaluate("node => node.classList.contains('active')"))
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
                filter_layout = page.evaluate(
                    """() => {
                      const rectFor = (selector) => {
                        const box = document.querySelector(selector).getBoundingClientRect();
                        return {
                          left: box.left,
                          right: box.right,
                          top: box.top,
                          bottom: box.bottom,
                        };
                      };
                      return {
                        mode: rectFor(".filter-selection-mode"),
                        clear: rectFor("#filterRowClearBtn"),
                        text: rectFor("#filterRowMetaText"),
                        operator: rectFor(".filter-operator"),
                        primary: rectFor(".filter-controls-primary"),
                        header: rectFor(".filter-header"),
                        meta: rectFor("#filterRowMeta"),
                      };
                    }"""
                )
                self.assertGreaterEqual(filter_layout["clear"]["left"], filter_layout["meta"]["left"])
                self.assertLessEqual(filter_layout["clear"]["right"], filter_layout["text"]["left"] + 1)
                self.assertLessEqual(filter_layout["text"]["right"], filter_layout["meta"]["right"])
                self.assertLessEqual(
                    filter_layout["meta"]["right"] - filter_layout["meta"]["left"],
                    filter_layout["text"]["right"] - filter_layout["clear"]["left"] + 14,
                )
                self.assertGreater(filter_layout["operator"]["top"], filter_layout["mode"]["bottom"] - 1)
                self.assertAlmostEqual(filter_layout["operator"]["left"], filter_layout["mode"]["left"], delta=1)
                self.assertAlmostEqual(filter_layout["meta"]["right"], filter_layout["header"]["right"], delta=1)
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
                self.assertTrue(page.locator("#filterRowClearBtn").is_visible())
                page.locator("#filterCollapseBtn").click()
                self.assertTrue(page.locator("#filterRowClearBtn").is_visible())
                page.locator("#filterCollapseBtn").click()
                self.assertTrue(page.locator("#filterRowClearBtn").is_visible())
                driver_rows.first.click()
                self.assertEqual(driver_rows.first.get_attribute("aria-selected"), "true")
                self.assertEqual(postcode_rows.first.get_attribute("aria-selected"), "false")
                postcode_rows.first.click()
                self.assertEqual(driver_rows.first.get_attribute("aria-selected"), "false")
                self.assertEqual(postcode_rows.first.get_attribute("aria-selected"), "true")
                self.assertTrue(page.locator("#filterRowClearBtn").is_visible())
                page.locator("#filterRowClearBtn").click()
                self.assertEqual(postcode_rows.first.get_attribute("aria-selected"), "false")
                self.assertEqual(page.locator("#filterInput").input_value(), "")
                self.assertFalse(page.locator("#filterRowClearBtn").is_visible())

                footer_value = page.locator("#filterInput").input_value()
                self.assertEqual(page.locator("#filterFooter").get_attribute("aria-hidden"), "true")
                page.locator("#filterFooterToggleBtn").click()
                self.assertEqual(page.locator("#filterFooter").get_attribute("aria-hidden"), "false")
                self.assertEqual(page.locator("#filterInput").input_value(), footer_value)
                page.locator("#filterFooterToggleBtn").click()
                self.assertEqual(page.locator("#filterFooter").get_attribute("aria-hidden"), "true")
                self.assertFalse(page.locator("#filterInput").is_visible())

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

    def exercise_line_bar_favourites(self, base_url: str) -> None:
        assert sync_playwright is not None
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch()
            page = browser.new_page(viewport={"width": 1280, "height": 800})
            page_errors: list[str] = []
            dialogs: list[str] = []
            page.on("pageerror", lambda error: page_errors.append(str(error)))
            page.on("dialog", lambda dialog: (dialogs.append(dialog.message), dialog.accept()))
            try:
                page.goto(base_url, wait_until="domcontentloaded")
                page.wait_for_function(
                    """
                    () => document.querySelector("#lineBarTool")?.classList.contains("active")
                    """,
                    timeout=10_000,
                )
                if page.locator("#favouritesCollapseBtn").get_attribute("aria-expanded") == "false":
                    page.locator("#favouritesCollapseBtn").click()
                page.locator("#sidebarFavouriteAddBtn").wait_for(timeout=10_000)

                page.locator("#filterCollapseBtn").click()
                age_heading = page.locator('.saved-filter-theme[data-filter-theme="AGE"]')
                age_row = page.locator('.saved-filter-option[data-filter-theme="AGE"]')
                if age_heading.get_attribute("aria-expanded") == "false":
                    age_heading.click()
                age_row.click()
                self.assertEqual(page.locator("#filterInput").input_value(), "vehicle_age >= 3")

                if page.locator("#lineBarSideControlsToggleBtn").get_attribute("aria-expanded") == "false":
                    page.locator("#lineBarSideControlsToggleBtn").click()
                    page.wait_for_function(
                        """
                        () => document.querySelector("#lineBarSideControlsToggleBtn")?.getAttribute("aria-expanded") === "true"
                          && getComputedStyle(document.querySelector("#chartSideControls")).display !== "none"
                        """,
                        timeout=10_000,
                    )
                if page.locator("#expectedSideSection").get_attribute("aria-hidden") == "true":
                    page.locator("#chartExpectedToggle").click()
                page.locator("#expectedSearch").fill("expected")
                page.locator('#expectedList .feature[data-value="expected"]').click()
                self.assertEqual(page.locator("#denominator").input_value(), "value")
                self.assertEqual(page.locator('.kpi-option[data-kpi-group="PRICE"]').get_attribute("aria-selected"), "true")

                chart_box_before = page.locator("#chart").bounding_box()
                self.assertIsNotNone(chart_box_before)
                page.locator("#favouritesCollapseBtn").click()
                page.wait_for_function(
                    '() => document.querySelector("#favouritesCollapseBtn")?.getAttribute("aria-expanded") === "true"',
                    timeout=10_000,
                )
                self.click_sidebar_favourite_action(page, "#sidebarFavouriteAddBtn")
                page.locator("#sidebarFavouritePopover:not([hidden])").wait_for(timeout=10_000)
                page.wait_for_function(
                    """() => document.querySelector(".sidebar-favourite-scope-title")?.textContent.trim() === "Scope"
                      && document.querySelector('input[name="sidebarFavouriteScope"][value="line_bar_view"]')?.checked
                      && [...document.querySelectorAll(".sidebar-favourite-scope-option span")]
                        .map((node) => node.textContent.trim()).join("|") === "Line/Bar view|Metrics + filter|Metrics" """,
                    timeout=10_000,
                )
                chart_box_after = page.locator("#chart").bounding_box()
                self.assertIsNotNone(chart_box_after)
                assert chart_box_before is not None and chart_box_after is not None
                self.assertLessEqual(abs(chart_box_after["y"] - chart_box_before["y"]), 1)
                long_favourite_name = "Codex Favourites Test With A Long Sidebar Name That Must Truncate"
                page.locator("#sidebarFavouriteNameInput").fill(long_favourite_name)
                page.locator('[data-favourite-action="save-add"]').click()
                page.wait_for_function(
                    """([name]) => [...document.querySelectorAll(".saved-favourite-option")]
                      .some((button) => button.querySelector(".saved-filter-name")?.textContent.trim() === name
                        && button.querySelector(".favourite-detail")?.textContent.trim() === "Line/Bar view"
                        && button.classList.contains("active")) """,
                    arg=[long_favourite_name],
                    timeout=10_000,
                )
                older_id = page.eval_on_selector(
                    ".saved-favourite-option.active",
                    'button => button?.dataset.favouriteId || ""',
                )
                self.assertTrue(older_id)
                page.evaluate("() => document.documentElement.style.setProperty('--sidebar-width', '220px')")
                page.wait_for_function(
                    """
                    ([name]) => {
                      const selectors = ["#favouritesCollapseBtn", "#kpiCollapseBtn", "#filterCollapseBtn"];
                      const icons = selectors.map((selector) => document.querySelector(`${selector} .filter-collapse-icon`));
                      if (icons.some((icon) => !icon)) return false;
                      const lefts = icons.map((icon) => icon.getBoundingClientRect().left);
                      const meta = document.querySelector("#favouritesSelectedMeta");
                      return Math.max(...lefts) - Math.min(...lefts) <= 3
                        && meta?.textContent === name
                        && meta.clientWidth > 0
                        && meta.scrollWidth > meta.clientWidth;
                    }
                    """,
                    arg=[long_favourite_name],
                    timeout=10_000,
                )
                page.evaluate("() => document.documentElement.style.setProperty('--sidebar-width', '300px')")
                self.click_sidebar_favourite_action(page, "#sidebarFavouriteAddBtn")
                page.locator("#sidebarFavouritePopover:not([hidden])").wait_for(timeout=10_000)
                page.evaluate(
                    """
                    ([id]) => document.querySelector(`.saved-favourite-option[data-favourite-id="${id}"]`)
                      ?.dispatchEvent(new MouseEvent("click", { bubbles: true, cancelable: true, view: window }))
                    """,
                    arg=[older_id],
                )
                page.wait_for_function(
                    """() => document.querySelector("#sidebarFavouritePopover")?.hidden === true""",
                    timeout=10_000,
                )

                page.locator("#filterRowClearBtn").click()
                page.locator("#expectedSearch").fill("expected")
                page.locator('#expectedList .feature[data-value="expected"]').click()
                self.assertEqual(page.locator("#filterInput").input_value(), "")
                page.locator(f'.saved-favourite-option[data-favourite-id="{older_id}"]').click()
                page.wait_for_function(
                    """() => document.querySelector("#filterInput")?.value === "vehicle_age >= 3"
                      && document.querySelector('.saved-filter-option[data-filter-theme="AGE"]')?.getAttribute("aria-selected") === "true"
                      && document.querySelector('#expectedList .feature[data-value="expected"]')?.getAttribute("aria-pressed") === "true"
                      && document.querySelector('.kpi-option[data-kpi-group="PRICE"]')?.getAttribute("aria-selected") === "true" """,
                    timeout=10_000,
                )

                self.click_sidebar_favourite_action(page, "#sidebarFavouriteMenuBtn")
                page.locator("#sidebarFavouritePopover:not([hidden])").wait_for(timeout=10_000)
                self.click_sidebar_favourite_action(page, "#sidebarFavouriteMenuBtn")
                page.wait_for_function(
                    """() => document.querySelector("#sidebarFavouritePopover")?.hidden === true""",
                    timeout=10_000,
                )
                self.click_sidebar_favourite_action(page, "#sidebarFavouriteMenuBtn")
                page.locator("#sidebarFavouritePopover:not([hidden])").wait_for(timeout=10_000)
                page.evaluate(
                    """
                    ([id]) => document.querySelector(`.saved-favourite-option[data-favourite-id="${id}"]`)
                      ?.dispatchEvent(new MouseEvent("click", { bubbles: true, cancelable: true, view: window }))
                    """,
                    arg=[older_id],
                )
                page.wait_for_function(
                    """() => document.querySelector("#sidebarFavouritePopover")?.hidden === true""",
                    timeout=10_000,
                )
                self.click_sidebar_favourite_action(page, "#sidebarFavouriteMenuBtn")
                page.locator("#sidebarFavouritePopover:not([hidden])").wait_for(timeout=10_000)
                self.assertTrue(page.locator('[data-favourite-action="move-up"]').is_disabled())
                self.assertTrue(page.locator('[data-favourite-action="move-down"]').is_disabled())
                first_row = page.locator(".line-bar-favourite-row").first
                rename_button = first_row.locator('[data-favourite-action="rename"]')
                self.assertTrue(rename_button.is_disabled())
                first_row.locator("input").fill("Renamed view")
                page.wait_for_function(
                    """() => {
                      const button = document.querySelector('.line-bar-favourite-row [data-favourite-action="rename"]');
                      return Boolean(button && !button.disabled && button.classList.contains("active"));
                    }""",
                    timeout=10_000,
                )
                rename_button.click()
                page.wait_for_function(
                    """() => [...document.querySelectorAll(".saved-favourite-option .saved-filter-name")]
                      .some((node) => node.textContent.trim() === "Renamed view")""",
                    timeout=10_000,
                )

                self.click_sidebar_favourite_action(page, "#sidebarFavouriteAddBtn")
                page.locator("#sidebarFavouriteNameInput").fill("Second view")
                page.locator('[data-favourite-action="save-add"]').click()
                page.wait_for_function(
                    """() => [...document.querySelectorAll(".saved-favourite-option .saved-filter-name")]
                      .some((node) => node.textContent.trim() === "Second view")""",
                    timeout=10_000,
                )
                self.click_sidebar_favourite_action(page, "#sidebarFavouriteMenuBtn")
                page.locator("#sidebarFavouritePopover:not([hidden])").wait_for(timeout=10_000)
                move_up = page.locator('[data-favourite-action="move-up"]')
                move_down = page.locator('[data-favourite-action="move-down"]')
                self.assertTrue(move_up.is_disabled())
                self.assertTrue(move_down.is_disabled())
                page.locator(".line-bar-favourite-row").first.locator("input").click()
                page.wait_for_function(
                    """() => {
                      const rows = [...document.querySelectorAll(".line-bar-favourite-row")];
                      const up = document.querySelector('[data-favourite-action="move-up"]');
                      const down = document.querySelector('[data-favourite-action="move-down"]');
                      return rows[0]?.classList.contains("selected") && up?.disabled && down && !down.disabled;
                    }""",
                    timeout=10_000,
                )
                move_down_elapsed = page.evaluate(
                    """
                    async () => {
                      const button = document.querySelector('[data-favourite-action="move-down"]');
                      const start = performance.now();
                      button?.click();
                      while (performance.now() - start < 1000) {
                        await new Promise((resolve) => requestAnimationFrame(resolve));
                        const labels = [...document.querySelectorAll(".saved-favourite-option .saved-filter-name")].map((node) => node.textContent.trim());
                        const rows = [...document.querySelectorAll(".line-bar-favourite-row")];
                        const up = document.querySelector('[data-favourite-action="move-up"]');
                        const down = document.querySelector('[data-favourite-action="move-down"]');
                        if (labels[0] === "Second view" &&
                          labels[1] === "Renamed view" &&
                          rows[1]?.classList.contains("selected") &&
                          rows[1]?.querySelector("input") === document.activeElement &&
                          up && !up.disabled &&
                          down?.disabled) {
                          return performance.now() - start;
                        }
                      }
                      return -1;
                    }
                    """
                )
                self.assertGreaterEqual(move_down_elapsed, 0)
                self.assertLess(move_down_elapsed, 100)
                move_up_elapsed = page.evaluate(
                    """
                    async () => {
                      const button = document.querySelector('[data-favourite-action="move-up"]');
                      const start = performance.now();
                      button?.click();
                      while (performance.now() - start < 1000) {
                        await new Promise((resolve) => requestAnimationFrame(resolve));
                        const labels = [...document.querySelectorAll(".saved-favourite-option .saved-filter-name")].map((node) => node.textContent.trim());
                        const rows = [...document.querySelectorAll(".line-bar-favourite-row")];
                        const up = document.querySelector('[data-favourite-action="move-up"]');
                        const down = document.querySelector('[data-favourite-action="move-down"]');
                        if (labels[0] === "Renamed view" &&
                          labels[1] === "Second view" &&
                          rows[0]?.classList.contains("selected") &&
                          rows[0]?.querySelector("input") === document.activeElement &&
                          up?.disabled &&
                          down && !down.disabled) {
                          return performance.now() - start;
                        }
                      }
                      return -1;
                    }
                    """
                )
                self.assertGreaterEqual(move_up_elapsed, 0)
                self.assertLess(move_up_elapsed, 100)
                rapid_move_state = page.evaluate(
                    """
                    async () => {
                      const clickAction = (action) => document.querySelector(`[data-favourite-action="${action}"]`)?.click();
                      clickAction("move-down");
                      await new Promise((resolve) => requestAnimationFrame(resolve));
                      clickAction("move-up");
                      await new Promise((resolve) => requestAnimationFrame(resolve));
                      clickAction("move-down");
                      await new Promise((resolve) => requestAnimationFrame(resolve));
                      const labels = [...document.querySelectorAll(".saved-favourite-option .saved-filter-name")].map((node) => node.textContent.trim());
                      const rows = [...document.querySelectorAll(".line-bar-favourite-row")];
                      const up = document.querySelector('[data-favourite-action="move-up"]');
                      const down = document.querySelector('[data-favourite-action="move-down"]');
                      return {
                        labels,
                        selectedBottom: Boolean(rows[1]?.classList.contains("selected")),
                        upEnabled: Boolean(up && !up.disabled),
                        downDisabled: Boolean(down?.disabled),
                      };
                    }
                    """
                )
                self.assertEqual(rapid_move_state["labels"][:2], ["Second view", "Renamed view"])
                self.assertTrue(rapid_move_state["selectedBottom"])
                self.assertTrue(rapid_move_state["upEnabled"])
                self.assertTrue(rapid_move_state["downDisabled"])
                page.wait_for_function(
                    """async () => {
                      const response = await fetch("/api/line-bar/favourites", { headers: { "x-lucidum-token": "" } });
                      const data = await response.json();
                      const names = (data.favourites || []).slice(0, 2).map((favourite) => favourite.name);
                      return names[0] === "Second view" && names[1] === "Renamed view";
                    }""",
                    timeout=10_000,
                )
                self.click_sidebar_favourite_action(page, "#sidebarFavouriteMenuBtn")
                page.wait_for_function(
                    """() => document.querySelector("#sidebarFavouritePopover")?.hidden === true""",
                    timeout=10_000,
                )

                startup_page = browser.new_page(viewport={"width": 1280, "height": 800})
                startup_errors: list[str] = []
                startup_page.on("pageerror", lambda error: startup_errors.append(str(error)))
                startup_page.goto(f"{base_url}?line_bar_favourite=Renamed%20view", wait_until="domcontentloaded")
                startup_page.wait_for_function(
                    """
                    () => document.querySelector("#lineBarTool")?.classList.contains("active")
                    """,
                    timeout=10_000,
                )
                startup_page.wait_for_function(
                    """() => document.querySelector("#filterInput")?.value === "vehicle_age >= 3"
                      && document.querySelector('.saved-filter-option[data-filter-theme="AGE"]')?.getAttribute("aria-selected") === "true"
                      && document.querySelector('#expectedList .feature[data-value="expected"]')?.getAttribute("aria-pressed") === "true" """,
                    timeout=10_000,
                )
                self.assertEqual(startup_errors, [])
                startup_page.close()

                default_startup_page = browser.new_page(viewport={"width": 1280, "height": 800})
                default_startup_errors: list[str] = []
                default_startup_chart_requests: list[dict[str, Any]] = []
                default_startup_page.on("pageerror", lambda error: default_startup_errors.append(str(error)))

                def record_default_startup_chart(route: Any) -> None:
                    if route.request.method == "POST":
                        default_startup_chart_requests.append(json.loads(route.request.post_data or "{}"))
                    route.continue_()

                default_startup_page.route("**/api/chart", record_default_startup_chart)
                default_startup_page.goto(base_url, wait_until="domcontentloaded")
                default_startup_page.wait_for_function(
                    """
                    () => document.querySelector("#lineBarTool")?.classList.contains("active")
                    """,
                    timeout=10_000,
                )
                default_startup_page.wait_for_function(
                    """() => document.querySelector("#favouritesSelectedMeta")?.textContent.trim() === "Second view"
                      && document.querySelector(".saved-favourite-option.active .saved-filter-name")?.textContent.trim() === "Second view"
                      && document.querySelector("#filterInput")?.value === "vehicle_age >= 3"
                      && document.querySelector('.saved-filter-option[data-filter-theme="AGE"]')?.getAttribute("aria-selected") === "true"
                      && document.querySelector('#expectedList .feature[data-value="expected"]')?.getAttribute("aria-pressed") === "true" """,
                    timeout=10_000,
                )
                default_startup_page.wait_for_function(
                    "() => document.querySelector('#lineBarGroupMeta')?.textContent.includes('groups')",
                    timeout=10_000,
                )
                self.assertTrue(default_startup_chart_requests)
                self.assertTrue(all(
                    request.get("filter") == "vehicle_age >= 3"
                    and [response.get("numerator") for response in request.get("responses", [])] == ["price", "expected"]
                    for request in default_startup_chart_requests
                ))
                self.assertEqual(default_startup_errors, [])
                default_startup_page.close()

                self.click_sidebar_favourite_action(page, "#sidebarFavouriteMenuBtn")
                page.locator('.line-bar-favourite-row [data-favourite-action="delete"]').first.click()
                page.wait_for_function(
                    """() => document.querySelectorAll(".saved-favourite-option").length === 1""",
                    timeout=10_000,
                )
                self.click_sidebar_favourite_action(page, "#sidebarFavouriteMenuBtn")
                page.wait_for_function(
                    """() => document.querySelector("#sidebarFavouritePopover")?.hidden === true""",
                    timeout=10_000,
                )
                remaining_favourite = page.eval_on_selector(
                    ".saved-favourite-option",
                    """button => ({
                      id: button?.dataset.favouriteId || "",
                      name: button?.querySelector(".saved-filter-name")?.textContent.trim() || "",
                    })""",
                )
                self.assertTrue(remaining_favourite["id"])
                page.locator(f'.saved-favourite-option[data-favourite-id="{remaining_favourite["id"]}"]').click()
                page.wait_for_function(
                    """([id]) => document.querySelector(`.saved-favourite-option[data-favourite-id="${id}"]`)?.classList.contains("active")""",
                    arg=[remaining_favourite["id"]],
                    timeout=10_000,
                )
                if page.locator("#lineBarToolbarToggleBtn").get_attribute("aria-expanded") == "false":
                    page.locator("#lineBarToolbarToggleBtn").click()
                    page.wait_for_function(
                        """
                        () => document.querySelector("#lineBarToolbarToggleBtn")?.getAttribute("aria-expanded") === "true"
                          && getComputedStyle(document.querySelector("#lineBarToolbar")).display !== "none"
                        """,
                        timeout=10_000,
                    )
                page.locator('.segmented[data-control="labels"] button[data-value="bar"]').click()
                page.wait_for_function(
                    """() => !document.querySelector(".saved-favourite-option.active")""",
                    timeout=10_000,
                )
                page.locator(f'.saved-favourite-option[data-favourite-id="{remaining_favourite["id"]}"]').click()
                page.wait_for_function(
                    """([id]) => document.querySelector(`.saved-favourite-option[data-favourite-id="${id}"]`)?.classList.contains("active")
                      && document.querySelector('.segmented[data-control="labels"] button[data-value="none"]')?.classList.contains("active")""",
                    arg=[remaining_favourite["id"]],
                    timeout=10_000,
                )

                self.click_sidebar_favourite_action(page, "#sidebarFavouriteAddBtn")
                page.locator("#sidebarFavouriteNameInput").fill("Metric only")
                page.locator('[data-favourite-scope-option="metrics"]').click()
                page.wait_for_function(
                    """() => document.querySelector('input[name="sidebarFavouriteScope"][value="metrics"]')?.checked
                      && document.querySelector('[data-favourite-scope-option="metrics"]')?.classList.contains("active")""",
                    timeout=10_000,
                )
                page.locator('[data-favourite-action="save-add"]').click()
                page.wait_for_function(
                    """() => [...document.querySelectorAll(".saved-favourite-option .saved-filter-name")]
                      .some((node) => node.textContent.trim() === "Metric only"
                        && node.closest(".saved-favourite-option")?.querySelector(".favourite-detail")?.textContent.trim() === "Metrics")""",
                    timeout=10_000,
                )
                metric_only_id = page.evaluate(
                    """
                    () => [...document.querySelectorAll(".saved-favourite-option")]
                      .find((button) => button.querySelector(".saved-filter-name")?.textContent.trim() === "Metric only")
                      ?.dataset.favouriteId || ""
                    """
                )
                self.assertTrue(metric_only_id)
                page.locator("#favouritesCollapseBtn").click()
                page.wait_for_function(
                    '() => document.querySelector("#favouritesCollapseBtn")?.getAttribute("aria-expanded") === "false"',
                    timeout=10_000,
                )
                page.locator(".sidebar-metric-section").wait_for(state="visible", timeout=10_000)
                page.locator("#filterRowClearBtn").click()
                page.locator("#actualNumerator").select_option("value")
                page.locator("#denominator").select_option("__none__")
                page.locator("#favouritesCollapseBtn").click()
                page.wait_for_function(
                    '() => document.querySelector("#favouritesCollapseBtn")?.getAttribute("aria-expanded") === "true"',
                    timeout=10_000,
                )
                page.locator(f'.saved-favourite-option[data-favourite-id="{metric_only_id}"]').click()
                page.wait_for_function(
                    """() => document.querySelector("#actualNumerator")?.value === "price"
                      && document.querySelector("#denominator")?.value === "value"
                      && document.querySelector("#filterInput")?.value === "" """,
                    timeout=10_000,
                )

                if page.locator("#filterCollapseBtn").get_attribute("aria-expanded") == "false":
                    page.locator("#filterCollapseBtn").click()
                age_row.click()
                page.locator("#favouritesCollapseBtn").click()
                self.click_sidebar_favourite_action(page, "#sidebarFavouriteAddBtn")
                page.locator("#sidebarFavouriteNameInput").fill("Metric filter")
                page.locator('[data-favourite-scope-option="metrics_filter"]').click()
                page.wait_for_function(
                    """() => document.querySelector('input[name="sidebarFavouriteScope"][value="metrics_filter"]')?.checked
                      && document.querySelector('[data-favourite-scope-option="metrics_filter"]')?.classList.contains("active")""",
                    timeout=10_000,
                )
                page.locator('[data-favourite-action="save-add"]').click()
                page.wait_for_function(
                    """() => [...document.querySelectorAll(".saved-favourite-option .saved-filter-name")]
                      .some((node) => node.textContent.trim() === "Metric filter"
                        && node.closest(".saved-favourite-option")?.querySelector(".favourite-detail")?.textContent.trim() === "Metrics + filter")""",
                    timeout=10_000,
                )
                metric_filter_id = page.evaluate(
                    """
                    () => [...document.querySelectorAll(".saved-favourite-option")]
                      .find((button) => button.querySelector(".saved-filter-name")?.textContent.trim() === "Metric filter")
                      ?.dataset.favouriteId || ""
                    """
                )
                self.assertTrue(metric_filter_id)
                page.locator("#filterRowClearBtn").click()
                page.locator("#tableTab").click()
                page.locator("#tableWrap:not(.hidden)").wait_for(timeout=10_000)
                self.click_sidebar_favourite_action(page, "#sidebarFavouriteAddBtn")
                page.locator("#sidebarFavouritePopover:not([hidden])").wait_for(timeout=10_000)
                page.wait_for_function(
                    """() => document.querySelector('input[name="sidebarFavouriteScope"][value="line_bar_view"]')?.checked
                      && [...document.querySelectorAll(".sidebar-favourite-scope-option span")]
                        .map((node) => node.textContent.trim()).join("|") === "Table view|Metrics + filter|Metrics" """,
                    timeout=10_000,
                )
                page.locator("#sidebarFavouriteNameInput").fill("Table favourite")
                page.locator('[data-favourite-action="save-add"]').click()
                page.wait_for_function(
                    """() => [...document.querySelectorAll(".saved-favourite-option")]
                      .some((button) => button.querySelector(".saved-filter-name")?.textContent.trim() === "Table favourite"
                        && button.querySelector(".favourite-detail")?.textContent.trim() === "Table view"
                        && button.classList.contains("active")) """,
                    timeout=10_000,
                )
                table_favourite_id = page.evaluate(
                    """
                    () => [...document.querySelectorAll(".saved-favourite-option")]
                      .find((button) => button.querySelector(".saved-filter-name")?.textContent.trim() === "Table favourite")
                      ?.dataset.favouriteId || ""
                    """
                )
                self.assertTrue(table_favourite_id)
                self.click_sidebar_favourite_action(page, "#sidebarFavouriteMenuBtn")
                page.locator("#sidebarFavouritePopover:not([hidden])").wait_for(timeout=10_000)
                manage_scroll_state = page.evaluate(
                    """
                    () => {
                      const popover = document.querySelector("#sidebarFavouritePopover");
                      const moveControls = popover?.querySelector(".line-bar-favourite-move-controls");
                      const list = popover?.querySelector(".line-bar-favourite-popover-list");
                      const row = popover?.querySelector(".line-bar-favourite-row");
                      const deleteButton = row?.querySelector('[data-favourite-action="delete"]');
                      const scope = row?.querySelector(".line-bar-favourite-row-scope");
                      const deleteLefts = [...(popover?.querySelectorAll(".line-bar-favourite-delete-button") || [])]
                        .map((button) => button.getBoundingClientRect().left);
                      if (!popover || !moveControls || !list || !deleteButton || !scope) {
                        return { missing: true };
                      }
                      popover.style.maxHeight = "96px";
                      const before = moveControls.getBoundingClientRect();
                      const listBefore = list.getBoundingClientRect();
                      list.scrollTop = list.scrollHeight;
                      const after = moveControls.getBoundingClientRect();
                      const listAfter = list.getBoundingClientRect();
                      return {
                        missing: false,
                        manageMode: popover.classList.contains("line-bar-favourite-popover--manage"),
                        controlsBeforeList: before.bottom <= listBefore.top,
                        controlsStayedPut: Math.abs(before.top - after.top) <= 1,
                        listStayedPut: Math.abs(listBefore.top - listAfter.top) <= 1,
                        listScrolled: list.scrollTop > 0,
                        listScrollable: list.scrollHeight > list.clientHeight + 1,
                        deleteButtonsAligned: deleteLefts.length > 1 && Math.max(...deleteLefts) - Math.min(...deleteLefts) <= 2,
                        scopeAfterDelete: Boolean(deleteButton.compareDocumentPosition(scope) & Node.DOCUMENT_POSITION_FOLLOWING),
                      };
                    }
                    """
                )
                self.assertFalse(manage_scroll_state.get("missing"))
                self.assertTrue(manage_scroll_state["manageMode"])
                self.assertTrue(manage_scroll_state["controlsBeforeList"])
                self.assertTrue(manage_scroll_state["controlsStayedPut"])
                self.assertTrue(manage_scroll_state["listStayedPut"])
                self.assertTrue(manage_scroll_state["listScrollable"])
                self.assertTrue(manage_scroll_state["listScrolled"])
                self.assertTrue(manage_scroll_state["deleteButtonsAligned"])
                self.assertTrue(manage_scroll_state["scopeAfterDelete"])
                self.click_sidebar_favourite_action(page, "#sidebarFavouriteMenuBtn")
                page.wait_for_function(
                    """() => document.querySelector("#sidebarFavouritePopover")?.hidden === true""",
                    timeout=10_000,
                )
                page.locator("#chartTab").click()
                page.locator("#chart:not(.hidden)").wait_for(timeout=10_000)
                page.locator(f'.saved-favourite-option[data-favourite-id="{table_favourite_id}"]').click()
                page.wait_for_function(
                    """([id]) => document.querySelector(`.saved-favourite-option[data-favourite-id="${id}"]`)?.classList.contains("active")
                      && !document.querySelector("#tableWrap")?.classList.contains("hidden")
                      && document.querySelector("#tableTab")?.classList.contains("active") """,
                    arg=[table_favourite_id],
                    timeout=10_000,
                )
                page.locator(f'.saved-favourite-option[data-favourite-id="{metric_filter_id}"]').click()
                page.wait_for_function(
                    """() => document.querySelector("#filterInput")?.value === "vehicle_age >= 3"
                      && !document.querySelector("#tableWrap")?.classList.contains("hidden")
                      && document.querySelector("#lineBarTool")?.classList.contains("active")""",
                    timeout=10_000,
                )
                self.assertEqual(dialogs, [])
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
                page.locator("#lineBarTool.active").wait_for(timeout=10_000)
                page.locator("#kpiCollapseBtn").wait_for(timeout=10_000)
                page.locator("#kpiCollapseBtn").click()
                page.wait_for_function(
                    '() => document.querySelector("#kpiCollapseBtn")?.getAttribute("aria-expanded") === "true"',
                    timeout=10_000,
                )
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
                page.locator("#kpiCollapseBtn").click()
                page.wait_for_function(
                    '() => document.querySelector("#kpiCollapseBtn")?.getAttribute("aria-expanded") === "false"',
                    timeout=10_000,
                )
                page.locator(".sidebar-metric-section").wait_for(state="visible", timeout=10_000)
                page.locator("#actualMetricTitle").get_by_text("20.0%").wait_for(timeout=10_000)
                page.locator("#weightMetricTitle").get_by_text("3").wait_for(timeout=10_000)
                self.assertEqual(page.locator("#actualNumerator").input_value(), "rate")
                self.assertEqual(rate_row.get_attribute("aria-selected"), "true")

                page.locator("#profileTool").click()
                page.locator("#profileWrap:not(.hidden) .profile-table").wait_for(timeout=10_000)
                self.assertTrue(page.locator(".sidebar-metric-section").is_visible())
                self.assertTrue(page.locator("#actualNumerator").is_visible())
                self.assertTrue(page.locator("#denominator").is_visible())
                self.assertIn("20.0%", page.locator("#actualMetricTitle").text_content())
                self.assertIn("3", page.locator("#weightMetricTitle").text_content())

                page.locator("#datasetViewerTool").click()
                page.locator("#datasetViewerWrap:not(.hidden) #datasetViewerGrid .tabulator-row").first.wait_for(timeout=10_000)
                self.assertTrue(page.locator(".sidebar-metric-section").is_visible())
                self.assertTrue(page.locator("#actualNumerator").is_visible())
                self.assertTrue(page.locator("#denominator").is_visible())
                self.assertIn("20.0%", page.locator("#actualMetricTitle").text_content())
                self.assertIn("3", page.locator("#weightMetricTitle").text_content())

                page.locator("#specsTool").click()
                page.locator("#specificationsWrap:not(.hidden)").wait_for(timeout=10_000)
                self.assertTrue(page.locator(".sidebar-metric-section").is_visible())
                self.assertTrue(page.locator("#actualNumerator").is_visible())
                self.assertTrue(page.locator("#denominator").is_visible())
                self.assertIn("20.0%", page.locator("#actualMetricTitle").text_content())
                self.assertIn("3", page.locator("#weightMetricTitle").text_content())

                page.locator("#ukMapTool").click()
                page.locator("#ukMap:not(.hidden)").wait_for(timeout=20_000)
                page.locator("#mapFloatingControl:not(.hidden)").wait_for(timeout=10_000)
                self.assertTrue(page.locator(".sidebar-metric-section").is_visible())
                page.locator("#lineBarTool").click()
                page.locator("#chart:not(.hidden)").wait_for(timeout=10_000)
                self.assertTrue(page.locator(".sidebar-metric-section").is_visible())

                self.assertEqual(page_errors, [])
            finally:
                browser.close()


if __name__ == "__main__":
    unittest.main()
