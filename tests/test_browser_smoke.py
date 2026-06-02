from __future__ import annotations

import json
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

from py_lucidum.app import create_app
from py_lucidum.core import Dataset, sql_literal
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
                        "objective": "gamma",
                        "metric": "gamma",
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
                        "feature_importance": [],
                        "feature_scenario": (
                            {"name": "scenario1", "features": ["Age", "Segment"]}
                            if model_id.endswith("-2")
                            else {"name": "old_scenario", "features": ["Age"]}
                        ),
                        "feature_interaction_constraints": (
                            {"groupings": ["DRIVER"], "groups": [{"grouping": "DRIVER", "features": ["Age"]}]}
                            if model_id.endswith("-2")
                            else {"groupings": ["OLD"], "groups": [{"grouping": "OLD", "features": ["Age"]}]}
                        ),
                        "sources": {},
                    },
                )
                feature_config = [{"name": "Age", "kind": "integer", "include": True, "monotonicity": "Increasing", "gain": 5.0}]
                if model_id == "browser-smoke-model":
                    feature_config.append({"name": "lat", "kind": "numeric", "include": True, "monotonicity": "", "gain": 4.0})
                if model_id.endswith("-2"):
                    feature_config.append({"name": "Segment", "kind": "categorical", "include": True, "monotonicity": "", "gain": 6.0})
                store.write_json(model_dir / "feature_config.json", feature_config)
                store.write_json(
                    model_dir / "parameters.json",
                    {
                        "objective": "gamma",
                        "metric": "gamma",
                        "training_mode": "ebm" if model_id.endswith("-2") else "normal",
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
                store.write_json(
                    model_dir / "training_log.json",
                    {
                        "evaluation": {
                            "training": {"gamma": training_eval},
                            "test": {"gamma": test_eval},
                        },
                        "warnings": [],
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
                self.exercise_gbm_profile_cache_and_model_chart_refresh(base_url)
            finally:
                server.should_exit = True
                thread.join(timeout=5)

    @unittest.skipUnless(RUN_BROWSER_TESTS, "set PY_LUCIDUM_RUN_BROWSER_TESTS=1 to run browser smoke tests")
    @unittest.skipUnless(sync_playwright is not None, "playwright is not installed")
    def test_sidebar_vertical_resizers_work_across_tools(self) -> None:
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
                "sidebar-resizer-model",
                "Sidebar resizer model",
                "2026-05-26T00:00:00Z",
                [0.12, 0.23, 0.34, 0.45],
            )
            store.activate_model("sidebar-resizer-model")
            base_url, server, thread = self.start_app(data_path, tools=["line_bar", "uk_map", "gbm"])
            try:
                self.exercise_sidebar_vertical_resizers(base_url)
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
        features_path: Path | None = None,
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
            features_path=features_path,
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
                "objective": "gamma",
                "metric": "gamma",
                "training_mode": "normal",
                "response_column": "actualNumerator",
                "offset_column": "denominator",
                "best_iteration": len(predictions),
                "training_rows": 2,
                "test_rows": 1,
                "scored_rows": len(predictions),
                "sample_column": "SAMPLE",
                "sample_source": "dataset",
                "source_columns": ["actualNumerator", "denominator", "Age", "Segment"],
                "sources": {"predictions": store.source_id(model_id, "predictions")},
            },
        )
        store.write_json(model_dir / "feature_config.json", [{"name": "Age", "kind": "integer", "include": True, "gain": 1.0}])
        store.write_json(model_dir / "parameters.json", {"objective": "gamma", "metric": "gamma", "num_iterations": len(predictions)})
        store.write_json(
            model_dir / "training_log.json",
            {"evaluation": {"training": {"gamma": predictions}, "test": {"gamma": predictions}}, "warnings": []},
        )
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
        finally:
            con.close()

    @staticmethod
    def assert_static_asset(base_url: str, path: str, expected_content_type: str) -> None:
        with urlopen(f"{base_url}{path}", timeout=5) as response:
            assert response.status == 200
            assert expected_content_type in response.headers.get("content-type", "")

    def exercise_sidebar_vertical_resizers(self, base_url: str) -> None:
        assert sync_playwright is not None
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch()
            page = browser.new_page(viewport={"width": 1280, "height": 800})
            page_errors: list[str] = []
            page.on("pageerror", lambda error: page_errors.append(str(error)))

            def drag_resizer(selector: str, delta_y: int) -> None:
                box = page.locator(selector).bounding_box()
                self.assertIsNotNone(box)
                assert box is not None
                x = box["x"] + box["width"] / 2
                y = box["y"] + box["height"] / 2
                page.mouse.move(x, y)
                page.mouse.down()
                page.mouse.move(x, y + delta_y, steps=5)
                page.mouse.up()

            try:
                page.goto(base_url, wait_until="domcontentloaded")
                page.locator("#datasetMeta").get_by_text("sample.csv").wait_for(timeout=10_000)
                page.locator("#profileWrap:not(.hidden) .profile-table").wait_for(timeout=10_000)
                page.locator("#gbmSidebarPanel:not(.hidden)").wait_for(timeout=10_000)
                self.assertFalse(page.locator(".sidebar-kpi-section").is_visible())

                page.locator("#filterCollapseBtn").click()
                self.assertEqual(page.locator("#filterCollapseBtn").get_attribute("aria-expanded"), "true")
                filter_before = page.evaluate(
                    """
                    () => {
                      const section = document.querySelector(".sidebar-filter-section");
                      const rect = section.getBoundingClientRect();
                      return { top: rect.top, height: rect.height };
                    }
                    """
                )
                drag_resizer("#sidebarFilterResizer", -90)
                page.wait_for_function(
                    """
                    (before) => {
                      const rect = document.querySelector(".sidebar-filter-section")?.getBoundingClientRect();
                      return Number(localStorage.getItem("py_lucidum_sidebar_filter_height")) > 0
                        && rect
                        && rect.height > before.height + 20
                        && rect.top < before.top - 20;
                    }
                    """,
                    arg=filter_before,
                    timeout=10_000,
                )

                gbm_before = page.evaluate(
                    """
                    () => {
                      const rect = document.querySelector(".gbm-sidebar-panel").getBoundingClientRect();
                      return { top: rect.top, bottom: rect.bottom, height: rect.height };
                    }
                    """
                )
                drag_resizer("#sidebarGbmResizer", 70)
                page.wait_for_function(
                    """
                    (before) => {
                      const rect = document.querySelector(".gbm-sidebar-panel")?.getBoundingClientRect();
                      return Number(localStorage.getItem("py_lucidum_sidebar_gbm_height")) > 0
                        && rect
                        && rect.top > before.top + 16
                        && Math.abs(rect.bottom - before.bottom) <= 2
                        && rect.height < before.height - 16;
                    }
                    """,
                    arg=gbm_before,
                    timeout=10_000,
                )

                page.locator("#lineBarTool").click()
                page.locator("#chart:not(.hidden)").wait_for(timeout=10_000)
                page.locator(".sidebar-kpi-section:not(.hidden)").wait_for(timeout=10_000)
                if page.locator("#filterCollapseBtn").get_attribute("aria-expanded") == "true":
                    page.locator("#filterCollapseBtn").click()
                    self.assertEqual(page.locator("#filterCollapseBtn").get_attribute("aria-expanded"), "false")
                if page.locator("#kpiCollapseBtn").get_attribute("aria-expanded") == "true":
                    page.locator("#kpiCollapseBtn").click()
                    self.assertEqual(page.locator("#kpiCollapseBtn").get_attribute("aria-expanded"), "false")
                page.wait_for_timeout(50)
                collapsed_kpi_gbm_before = page.evaluate(
                    """
                    () => {
                      const rect = document.querySelector(".gbm-sidebar-panel").getBoundingClientRect();
                      return { top: rect.top, bottom: rect.bottom, height: rect.height };
                    }
                    """
                )
                drag_resizer("#sidebarGbmResizer", 60)
                page.wait_for_function(
                    """
                    (before) => {
                      const rect = document.querySelector(".gbm-sidebar-panel")?.getBoundingClientRect();
                      return rect
                        && rect.top > before.top + 16
                        && Math.abs(rect.bottom - before.bottom) <= 2
                        && rect.height < before.height - 16;
                    }
                    """,
                    arg=collapsed_kpi_gbm_before,
                    timeout=10_000,
                )
                page.locator("#kpiCollapseBtn").click()
                self.assertEqual(page.locator("#kpiCollapseBtn").get_attribute("aria-expanded"), "true")
                self.assertEqual(page.locator("#sidebarKpiResizer").count(), 0)
                self.assertTrue(page.locator("#sidebarGbmResizer").is_visible())
                kpi_before = page.evaluate(
                    """
                    () => {
                      const rect = document.querySelector(".sidebar-kpi-section").getBoundingClientRect();
                      const gbmRect = document.querySelector(".gbm-sidebar-panel").getBoundingClientRect();
                      return { bottom: rect.bottom, height: rect.height, gbmHeight: gbmRect.height };
                    }
                    """
                )
                kpi_delta = -80 if kpi_before["gbmHeight"] < 100 else 80
                drag_resizer("#sidebarGbmResizer", kpi_delta)
                page.wait_for_function(
                    """
                    ({ before, delta }) => {
                      const rect = document.querySelector(".sidebar-kpi-section")?.getBoundingClientRect();
                      const gbmRect = document.querySelector(".gbm-sidebar-panel")?.getBoundingClientRect();
                      const moved = delta > 0
                        ? rect && rect.height > before.height + 20 && rect.bottom > before.bottom + 20
                        : rect && rect.height < before.height - 20 && rect.bottom < before.bottom - 20;
                      return Number(localStorage.getItem("py_lucidum_sidebar_kpi_height")) > 0
                        && rect
                        && gbmRect
                        && moved
                        && Math.abs(rect.bottom - gbmRect.top) <= 2
                        && !document.querySelector(".sidebar-filter-section")?.classList.contains("hidden");
                    }
                    """,
                    arg={"before": kpi_before, "delta": kpi_delta},
                    timeout=10_000,
                )
                if page.locator("#kpiCollapseBtn").get_attribute("aria-expanded") == "true":
                    page.locator("#kpiCollapseBtn").click()
                    self.assertEqual(page.locator("#kpiCollapseBtn").get_attribute("aria-expanded"), "false")
                if page.locator("#gbmModelCollapseBtn").get_attribute("aria-expanded") == "true":
                    page.locator("#gbmModelCollapseBtn").click()
                    self.assertEqual(page.locator("#gbmModelCollapseBtn").get_attribute("aria-expanded"), "false")
                if page.locator("#filterCollapseBtn").get_attribute("aria-expanded") == "true":
                    page.locator("#filterCollapseBtn").click()
                    self.assertEqual(page.locator("#filterCollapseBtn").get_attribute("aria-expanded"), "false")
                page.wait_for_function(
                    """
                    () => {
                      const kpi = document.querySelector(".kpi-header h2")?.getBoundingClientRect();
                      const gbm = document.querySelector(".gbm-model-header h2")?.getBoundingClientRect();
                      const filter = document.querySelector(".filter-header h2")?.getBoundingClientRect();
                      if (!kpi || !gbm || !filter) return false;
                      const kpiToGbm = gbm.top - kpi.top;
                      const gbmToFilter = filter.top - gbm.top;
                      return Math.abs(kpiToGbm - gbmToFilter) <= 1;
                    }
                    """,
                    timeout=10_000,
                )
                self.assertEqual(page_errors, [])
            finally:
                browser.close()

    def exercise_gbm_profile_cache_and_model_chart_refresh(self, base_url: str) -> None:
        assert sync_playwright is not None
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch()
            page = browser.new_page(viewport={"width": 1280, "height": 800})
            page_errors: list[str] = []
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
            try:
                page.goto(base_url, wait_until="domcontentloaded")
                page.locator("#profileWrap:not(.hidden) .profile-table").wait_for(timeout=10_000)
                page.locator("#profileDetailTitle").wait_for(timeout=10_000)
                page.locator("#gbmModelSelect").wait_for(timeout=10_000)
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

                profile_requests_before = profile_requests
                profile_detail_requests_before = profile_detail_requests
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
                page.locator('#gbmModelSelect [data-gbm-model-id="browser-smoke-model"]').click()
                page.locator("#gbmModelSelectedMeta", has_text="Browser smoke model").wait_for(timeout=10_000)
                page.wait_for_timeout(250)
                self.assertEqual(profile_requests, profile_requests_before)
                self.assertEqual(profile_detail_requests, profile_detail_requests_before)

                chart_url = (
                    f"{base_url}/?tool=line_bar&source=gbm%3Abrowser-smoke-model%3Apredictions"
                    "&x=Age&actual=gbm_prediction&denominator=denominator"
                )
                page.goto(chart_url, wait_until="domcontentloaded")
                page.locator("#chart:not(.hidden)").wait_for(timeout=10_000)
                page.wait_for_function(
                    '() => document.querySelector("#lineBarGroupMeta")?.textContent.includes("groups")',
                    timeout=10_000,
                )
                chart_requests_before = chart_requests
                with page.expect_request(lambda request: request.url.endswith("/api/chart"), timeout=10_000) as chart_request_info:
                    page.locator('#gbmModelSelect [data-gbm-model-id="browser-smoke-model-2"]').click()
                request_body = json.loads(chart_request_info.value.post_data or "{}")
                page.locator("#gbmModelSelectedMeta", has_text="Second smoke model").wait_for(timeout=10_000)
                self.assertGreater(chart_requests, chart_requests_before)
                self.assertEqual(request_body["source"], "gbm:browser-smoke-model-2:predictions")
                self.assertEqual(request_body["responses"][0]["numerator"], "gbm_prediction")
                self.assertEqual(request_body["denominator"], "denominator")
                self.assertEqual(page_errors, [])
            finally:
                browser.close()

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

            try:
                page.goto(base_url, wait_until="domcontentloaded")
                page.locator("#datasetMeta").get_by_text("sample.csv").wait_for(timeout=10_000)
                page.locator("#profileWrap:not(.hidden) .profile-table").wait_for(timeout=10_000)
                page.locator('#profileWrap .profile-summary-row[aria-selected="true"]').wait_for(timeout=10_000)
                page.locator("#profileDetailTitle").get_by_text("PostcodeArea").wait_for(timeout=10_000)
                self.assertEqual(page.locator("#profileFilter").evaluate("node => getComputedStyle(node).fontSize"), "10px")
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
                page.locator("#profileColumnContextMenu [role='menuitem']").click()
                page.wait_for_function("() => window.__lucidumCopiedText === 'vehicle_age'")
                page.wait_for_function('() => document.querySelector("#profileColumnContextMenu")?.hidden === true')
                page.locator("#clipboardToast").get_by_text("Copied vehicle_age to clipboard").wait_for(timeout=10_000)
                self.assertEqual(page.locator("#clipboardToast").evaluate("node => getComputedStyle(node).position"), "fixed")
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
                page.wait_for_function('() => document.querySelector("#mapGroupMeta")?.textContent.includes("areas matched")')
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

                page.locator("#sidebarToggleBtn").click()
                self.assertEqual(page.locator("#sidebarToggleBtn").get_attribute("aria-expanded"), "true")
                self.assertTrue(page.locator(".sidebar-kpi-section").is_visible())
                self.assertTrue(page.locator(".sidebar-filter-section").is_visible())

                page.locator("#ukMapTool").click()
                page.locator("#ukMap:not(.hidden)").wait_for(timeout=10_000)
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
                            document.querySelector("#filterInput").value = "vehicle_age >= 0";
                            document.querySelector("#filterApplyBtn").click();
                        }
                        """
                    )
                wait_for_map_view(stable_map_view)

                with page.expect_response(lambda response: response.url.endswith("/api/uk-map/summary") and response.status == 200, timeout=10_000):
                    page.locator('.map-layer-control input[name="mapLevel"][value="sector"]').check()
                page.wait_for_function('() => document.querySelector("#mapGroupMeta")?.textContent.includes("sectors matched")')
                wait_for_map_view(stable_map_view)

                self.assertEqual(page_errors, [])
                self.assertEqual(profile_requests, 2)
                self.assertEqual(profile_detail_requests, 3)
                self.assertEqual(chart_requests, 1)
                self.assertEqual(map_requests, 5)
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
                page.locator("#gbmFeatureContextMenu [role='menuitem']", has_text="Go to Line and Bar").click()
                page.locator("#lineBarTool.active").wait_for(timeout=10_000)
                page.locator("#featureList .feature.active", has_text="Age").wait_for(timeout=10_000)
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
                page.locator("input[name='gbmFeatureMetric'][value='gain']").check(force=True)
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
                self.assertEqual(initial_constraints["button"], "Trained constraints (1)")
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
                    () => document.querySelector("#gbmFeatureInteractionConstraintButton")?.textContent.trim() === "Constraints (1)"
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
                    () => document.querySelector("#gbmFeatureInteractionConstraintButton")?.textContent.trim() === "Constraints (1)"
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
                    (height) => Number(localStorage.getItem("py_lucidum_gbm_shap_feature1_height")) > 0
                      && Math.abs((document.querySelector("#gbmShapFeatureList1")?.closest(".gbm-shap-feature-section")?.getBoundingClientRect().height || 0) - height) > 8
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
                page.locator("#gbmModelSelect").wait_for(timeout=10_000)
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
                self.assertEqual(page.locator("#gbmFeatureInteractionConstraintButton").text_content(), "Constraints (1)")
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
                self.assertEqual(ebm_dim2_context_labels, ["Go to SHAP"])
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
                live_job_succeed = {"value": False}
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
                    if live_job_succeed["value"]:
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
                choose_feature_scenario("scenario1")
                page.locator("#gbmFeatureGrid .tabulator-row", has_text="Segment").locator(".tabulator-cell[tabulator-field='name']").click(button="right")
                page.locator("#gbmFeatureContextMenu [role='menuitem']", has_text="Toggle interaction constraint").click()
                page.locator("#gbmFeatureInteractionConstraintButton").click()
                page.locator('[data-gbm-interaction-grouping="VEHICLE"]').check()
                self.assertEqual(page.locator("#gbmFeatureInteractionConstraintButton").text_content(), "Constraints (1)")
                page.wait_for_function(
                    """
                    () => {
                      const row = [...document.querySelectorAll("#gbmFeatureGrid .tabulator-row")]
                        .find((item) => item.textContent.includes("Segment"));
                      return (row?.querySelector(".tabulator-cell[tabulator-field='name']")?.textContent || "").includes("\\uD83D\\uDD12")
                        && (row?.querySelector(".tabulator-cell[tabulator-field='grouping']")?.textContent || "").includes("\\uD83D\\uDD12");
                    }
                    """,
                    timeout=10_000,
                )
                page.locator("#gbmTrainBtn").click()
                page.locator("#gbmTrainingStatus").get_by_text("training, tree 2/10, test gamma 7.2").wait_for(timeout=10_000)
                page.locator("#startupProgress.ready", has_text="Training GBM (1/25)...").wait_for(timeout=10_000)
                self.assertEqual(
                    train_payload["value"]["feature_scenario"],
                    {"name": "scenario1", "features": ["Age", "Segment"]},
                )
                self.assertEqual(train_payload["value"]["feature_interaction_groupings"], ["VEHICLE"])
                self.assertEqual(train_payload["value"]["feature_interaction_features"], ["Segment"])
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
                live_job_succeed["value"] = True
                page.locator("#gbmTrainBtn", has_text="Train GBM").wait_for(timeout=10_000)
                page.locator("#startupProgress.ready", has_text="Ready").wait_for(timeout=10_000)
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
                self.assertGreater(tree_state["summaryWidth"], 500)
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
                page.mouse.move(resizer_box["x"] + resizer_box["width"] / 2, resizer_box["y"] + 24)
                page.mouse.down()
                page.mouse.move(resizer_box["x"] + resizer_box["width"] / 2 - 80, resizer_box["y"] + 24)
                page.mouse.up()
                summary_width_after = page.locator(".gbm-tree-summary-panel").evaluate("node => node.getBoundingClientRect().width")
                self.assertLess(summary_width_after, summary_width_before - 40)
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
                self.assertFalse(navigator_state["hasDeletedModel"])
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
                self.assertAlmostEqual(
                    layout["parameterTableColumnWidth"] / layout["parameterLayoutWidth"],
                    0.7,
                    delta=0.08,
                )
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
