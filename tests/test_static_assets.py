from __future__ import annotations

import asyncio
import shutil
import subprocess
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

from py_lucidum.app import create_app


def asgi_get(app: Any, path: str, headers: dict[str, str] | None = None) -> tuple[int, dict[str, str], bytes]:
    messages: list[dict[str, Any]] = []
    raw_headers = [
        (key.lower().encode("latin-1"), value.encode("latin-1"))
        for key, value in (headers or {}).items()
    ]

    async def receive() -> dict[str, Any]:
        return {"type": "http.request", "body": b"", "more_body": False}

    async def send(message: dict[str, Any]) -> None:
        messages.append(message)

    scope = {
        "type": "http",
        "asgi": {"version": "3.0", "spec_version": "2.3"},
        "http_version": "1.1",
        "method": "GET",
        "scheme": "http",
        "path": path,
        "raw_path": path.encode("ascii"),
        "query_string": b"",
        "headers": raw_headers,
        "client": ("127.0.0.1", 12345),
        "server": ("testserver", 80),
    }
    asyncio.run(app(scope, receive, send))

    start = next(message for message in messages if message["type"] == "http.response.start")
    body = b"".join(message.get("body", b"") for message in messages if message["type"] == "http.response.body")
    headers = {key.decode("latin-1").lower(): value.decode("latin-1") for key, value in start["headers"]}
    return start["status"], headers, body


class StaticAssetTests(unittest.TestCase):
    CSS_MODULE_PATHS = [
        "/static/styles/foundations.css",
        "/static/styles/shell.css",
        "/static/styles/controls.css",
        "/static/styles/line-bar.css",
        "/static/styles/uk-map.css",
        "/static/styles/column-profile.css",
        "/static/styles/model-shell.css",
        "/static/styles/gbm.css",
    ]

    def setUp(self) -> None:
        self.tmp = TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.data_path = Path(self.tmp.name) / "sample.csv"
        self.data_path.write_text("PostcodeArea,PostcodeSector,Actual\nAB,AB10 1,100\n", encoding="utf-8")
        self.app = create_app(self.data_path)

    def assert_no_store(self, path: str) -> tuple[dict[str, str], bytes]:
        status, headers, body = asgi_get(self.app, path)

        self.assertEqual(status, 200)
        self.assertEqual(headers.get("cache-control"), "no-store")
        return headers, body

    def app_css_contract(self) -> str:
        module_paths = ["/static/app.css", *self.CSS_MODULE_PATHS]
        return "\n".join(
            self.assert_no_store(path)[1].decode("utf-8")
            for path in module_paths
        )

    def app_js_contract(self) -> str:
        module_paths = [
            "/static/app.js",
            "/static/app/main.js",
            "/static/app/column-profile-tool.js",
            "/static/app/line-bar-tool.js",
            "/static/app/uk-map-tool.js",
            "/static/app/shared/api.js",
            "/static/app/shared/format.js",
            "/static/app/shared/schema.js",
            "/static/app/shared/timing.js",
            "/static/app/gbm-tool.js",
            "/static/app/gbm-shap-tool.js",
            "/static/app/gbm-shap-chart.js",
            "/static/app/gbm-stacked-shap-tool.js",
            "/static/app/gbm-stacked-shap-chart.js",
            "/static/app/gbm-tree-viewer.js",
            "/static/app/model-tool-shell.js",
        ]
        return "\n".join(
            self.assert_no_store(path)[1].decode("utf-8")
            for path in module_paths
        )

    def js_function_source(self, js: str, name: str) -> str:
        starts = [f"export function {name}", f"function {name}"]
        start = next((js.index(prefix) for prefix in starts if prefix in js), -1)
        self.assertGreaterEqual(start, 0, name)
        brace = js.index("{", js.index(")", start))
        depth = 0
        end = None
        for index in range(brace, len(js)):
            if js[index] == "{":
                depth += 1
            elif js[index] == "}":
                depth -= 1
                if depth == 0:
                    end = index + 1
                    break
        self.assertIsNotNone(end)
        return js[start:end].replace("export function", "function", 1)

    def run_node_script(self, script: str) -> None:
        node = shutil.which("node")
        if not node:
            self.skipTest("node is not installed")
        result = subprocess.run(
            [node, "--input-type=module", "-e", script],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr + result.stdout)

    def test_shared_format_helpers_are_importable(self) -> None:
        module = Path("src/py_lucidum/static/app/shared/format.js").resolve().as_uri()
        script = f"""
import {{ createFormatters, escapeHtml }} from "{module}";
const state = {{ activeKpiFormat: null }};
const formatters = createFormatters({{ getActiveKpiFormat: () => state.activeKpiFormat }});
if (escapeHtml("<a&b>") !== "&lt;a&amp;b&gt;") throw new Error("escapeHtml failed");
if (formatters.formatFileSize(1536) !== "1.5Kb") throw new Error("formatFileSize failed");
if (formatters.formatRowMeta(100, 25) !== "25 / 100 rows") throw new Error("formatRowMeta failed");
if (formatters.formatXLabel(1234, "integer") !== "1,234") throw new Error("formatXLabel failed");
state.activeKpiFormat = {{ decimals: 1, format: "percent" }};
if (formatters.formatLineValue(-0.125) !== "-12.5%") throw new Error("KPI percent formatting failed");
"""
        self.run_node_script(script)

    def test_shared_schema_helpers_are_importable(self) -> None:
        module = Path("src/py_lucidum/static/app/shared/schema.js").resolve().as_uri()
        script = f"""
import {{
  dataSourceForId,
  dataSourceHasColumn,
  isModelPredictionColumn,
  isModelTool,
  preferredStartupSource,
  sourceColumns,
  toolEnabled,
}} from "{module}";
const schema = {{
  columns: [{{ name: "Actual", kind: "numeric" }}],
  tools: [{{ id: "line_bar" }}],
  data_sources: [
    {{ id: "dataset", columns: [{{ name: "Actual", kind: "numeric" }}] }},
    {{ id: "gbm:one:predictions", kind: "gbm_predictions", active: true, columns: [{{ name: "gbm_prediction", kind: "numeric" }}] }},
  ],
}};
if (dataSourceForId(schema, "dataset").id !== "dataset") throw new Error("dataSourceForId failed");
if (!dataSourceHasColumn(schema, "gbm:one:predictions", "gbm_prediction")) throw new Error("dataSourceHasColumn failed");
if (sourceColumns(schema, "dataset").length !== 1) throw new Error("sourceColumns failed");
if (!toolEnabled(schema, "line_bar")) throw new Error("toolEnabled failed");
if (!isModelTool("gbm") || isModelTool("line_bar")) throw new Error("isModelTool failed");
if (!isModelPredictionColumn({{ name: "gbm_prediction" }})) throw new Error("isModelPredictionColumn failed");
if (preferredStartupSource(schema.data_sources, "missing") !== "gbm:one:predictions") throw new Error("preferredStartupSource failed");
"""
        self.run_node_script(script)

    def test_shared_timing_controller_is_importable(self) -> None:
        module = Path("src/py_lucidum/static/app/shared/timing.js").resolve().as_uri()
        script = f"""
import {{ createActionTimingController, freshActionTimings }} from "{module}";
const monitor = {{ textContent: "" }};
const state = {{ tool: "line_bar", actionTimings: freshActionTimings() }};
const controller = createActionTimingController({{
  state,
  el: (id) => {{
    if (id !== "actionTimingMonitor") throw new Error(`unexpected element ${{id}}`);
    return monitor;
  }},
  renderLabels: {{ line_bar: "Chart render" }},
  performanceImpl: {{ now: () => 0 }},
  requestAnimationFrameImpl: (callback) => callback(),
}});
controller.startToolTiming("line_bar");
if (monitor.textContent !== "DuckDB: running, JSON: --, Chart render: --, Total: --") throw new Error(monitor.textContent);
controller.setDuckDbTiming("line_bar", {{ duckdb_ns: 2000000 }});
controller.setClientTiming("line_bar", {{ data_ms: 3 }});
controller.setRenderTiming("line_bar", 12);
if (monitor.textContent !== "DuckDB: 2ms, JSON: 3ms, Chart render: 12ms, Total: 17ms") throw new Error(monitor.textContent);
"""
        self.run_node_script(script)

    def test_gbm_shap_selection_helper_maps_active_model_metadata(self) -> None:
        js = self.assert_no_store("/static/app/gbm-tool.js")[1].decode("utf-8")
        script = self.js_function_source(js, "gbmShapSelectionValue") + """
const cases = [
  ["zero", { active_model_id: "m0", models: [{ model_id: "m0", active: true, shap_rows: 0, scored_rows: 50000 }] }, "0"],
  ["ten-k", { active_model_id: "m10", models: [{ model_id: "m10", active: true, shap_rows: 10000, scored_rows: 50000 }] }, "10k"],
  ["hundred-k", { active_model_id: "m100", models: [{ model_id: "m100", active: true, shap_rows: 100000, scored_rows: 500000 }] }, "100k"],
  ["all", { active_model_id: "mall", models: [{ model_id: "mall", active: true, shap_rows: 50000, scored_rows: 50000 }] }, "all"],
  ["missing", { active_model_id: "old", models: [{ model_id: "old", active: true }] }, "0"],
];
for (const [name, data, expected] of cases) {
  const actual = gbmShapSelectionValue(data);
  if (actual !== expected) {
    throw new Error(`${name}: expected ${expected}, got ${actual}`);
  }
}
"""
        self.run_node_script(script)

    def test_gbm_model_detail_label_includes_best_train_and_test_metrics(self) -> None:
        js = self.assert_no_store("/static/app/gbm-tool.js")[1].decode("utf-8")
        helpers = [
            "gbmModelDetailLabel",
            "modelBestMetric",
            "modelNumberOrNull",
            "formatModelMetric",
            "formatEvaluationValue",
        ]
        script = "\n".join(self.js_function_source(js, name) for name in helpers) + """
const cases = [
  ["full", { metric: "mape", best_iteration: 5189, best_metrics: { training: 0.325612, test: 0.336543 } }, "mape · iter 5,189 · train 0.3256 · test 0.3365"],
  ["missing-test", { metric: "mape", best_iteration: 10, best_metrics: { training: 0.123456, test: null } }, "mape · iter 10 · train 0.1235 · test --"],
  ["missing-best-metrics", { metric: "mape", best_iteration: 1516 }, "mape · iter 1,516 · train -- · test --"],
  ["zero", { metric: "poisson", best_iteration: 1, best_metrics: { training: 0, test: 0 } }, "poisson · iter 1 · train 0 · test 0"],
];
for (const [name, model, expected] of cases) {
  const actual = gbmModelDetailLabel(model);
  if (actual !== expected) {
    throw new Error(`${name}: expected ${expected}, got ${actual}`);
  }
}
"""
        self.run_node_script(script)

    def test_gbm_training_ready_badge_label_reports_grid_progress(self) -> None:
        js = self.assert_no_store("/static/app/gbm-tool.js")[1].decode("utf-8")
        helpers = ["modelNumberOrNull", "formatTrainingBadgeCount", "gbmTrainingReadyBadgeLabel"]
        script = "\n".join(self.js_function_source(js, name) for name in helpers) + """
if (gbmTrainingReadyBadgeLabel() !== "Training GBM...") throw new Error("default label failed");
const gridLabel = gbmTrainingReadyBadgeLabel({ grid_model_number: 2, grid_model_count: 25 });
if (gridLabel !== "Training GBM (2/25)...") throw new Error(`grid label failed: ${gridLabel}`);
const pendingGridLabel = gbmTrainingReadyBadgeLabel({ grid: { trainable_count: 25 } });
if (pendingGridLabel !== "Training GBM...") throw new Error(`pending grid label failed: ${pendingGridLabel}`);
"""
        self.run_node_script(script)

    def test_gbm_mean_abs_shap_formatter_uses_four_decimal_places(self) -> None:
        js = self.assert_no_store("/static/app/gbm-tool.js")[1].decode("utf-8")
        script = "\n".join(self.js_function_source(js, name) for name in ["featureNumber", "formatMeanAbsShap"]) + """
const cases = [
  [0.183, "0.1830"],
  ["1.2", "1.2000"],
  [0, "0.0000"],
  [null, ""],
  ["", ""],
];
for (const [value, expected] of cases) {
  const actual = formatMeanAbsShap(value);
  if (actual !== expected) {
    throw new Error(`${value}: expected ${expected}, got ${actual}`);
  }
}
"""
        self.run_node_script(script)

    def test_gbm_shap_banding_steps_match_line_bar_steps(self) -> None:
        line_bar_js = self.assert_no_store("/static/app/main.js")[1].decode("utf-8")
        shap_js = self.assert_no_store("/static/app/gbm-shap-tool.js")[1].decode("utf-8")
        script = "\n".join([
            self.js_function_source(line_bar_js, "makeBandSteps").replace("function makeBandSteps", "function makeLineBarBandSteps", 1),
            self.js_function_source(shap_js, "makeBandSteps").replace("function makeBandSteps", "function makeShapBandSteps", 1),
        ]) + """
const lineBarSteps = makeLineBarBandSteps();
const shapSteps = makeShapBandSteps();
if (JSON.stringify(shapSteps) !== JSON.stringify(lineBarSteps)) {
  throw new Error("SHAP band steps diverged from Line/Bar band steps");
}
for (const value of [4, 7, 12]) {
  if (!shapSteps.includes(value)) throw new Error(`missing special band step ${value}`);
}
"""
        self.run_node_script(script)

    def test_startup_progress_pill_reports_schema_phases(self) -> None:
        _, html_body = self.assert_no_store("/")
        html = html_body.decode("utf-8")
        css = self.app_css_contract()
        js = self.app_js_contract()

        self.assertIn('id="startupProgress"', html)
        self.assertIn('class="startup-progress"', html)
        self.assertIn(".startup-progress", css)
        self.assertIn('setStartupProgress("Requesting schema")', js)
        self.assertIn('setStartupProgress("Schema received")', js)
        self.assertIn('setStartupProgress("Rendering controls")', js)
        self.assertIn('setStartupProgress("Ready", "ready")', js)
        self.assertIn('/api/telemetry', js)
        self.assertIn("current_action_seconds", js)

    def test_gbm_shap_flame_option_uses_exact_domain_without_45_55(self) -> None:
        chart_path = Path(__file__).resolve().parents[1] / "src/py_lucidum/static/app/gbm-shap-chart.js"
        script = f"""
import fs from "node:fs";
const source = fs.readFileSync({str(chart_path)!r}, "utf8").replaceAll("export ", "");
eval(source + "\\nglobalThis.__shapChartOption = shapChartOption;");
const rows = [
  {{ x: 18, p0: 0.1, p5: 0.11, p10: 0.12, p20: 0.13, p30: 0.14, p40: 0.15, p50: 0.16, p60: 0.17, p70: 0.18, p80: 0.19, p90: 0.2, p95: 0.21, p100: 0.22 }},
  {{ x: 83, p0: -0.2, p5: -0.19, p10: -0.18, p20: -0.17, p30: -0.16, p40: -0.15, p50: -0.14, p60: -0.13, p70: -0.12, p80: -0.11, p90: -0.1, p95: -0.09, p100: -0.08 }},
];
const option = globalThis.__shapChartOption({{
  plot_type: "flame",
  title: "SHAP flame plot: Age",
  x_feature: "Age",
  y_label: "SHAP",
  x_domain: [18, 83],
  y_domain: [-0.2, 0.22],
  rows,
}}, {{}});
if (option.xAxis.min !== 18 || option.xAxis.max !== 83) {{
  throw new Error(`expected exact x domain 18..83, got ${{option.xAxis.min}}..${{option.xAxis.max}}`);
}}
const seriesNames = option.series.map((series) => series.name);
if (seriesNames.includes("45-55")) throw new Error("45-55 series should not be rendered");
const tooltip = option.tooltip.formatter([{{ axisValue: 18, value: [18, 0.16] }}]);
if (tooltip.includes("45-55")) throw new Error("45-55 tooltip row should not be rendered");
"""
        self.run_node_script(script)

    def test_gbm_shap_box_axis_title_sits_below_rotated_labels(self) -> None:
        chart_path = Path(__file__).resolve().parents[1] / "src/py_lucidum/static/app/gbm-shap-chart.js"
        script = f"""
import fs from "node:fs";
const source = fs.readFileSync({str(chart_path)!r}, "utf8").replaceAll("export ", "");
eval(source + "\\nglobalThis.__shapChartOption = shapChartOption;");
const rows = Array.from({{ length: 30 }}, (_, index) => ({{
  level: `Make ${{index + 1}}`,
  p0: -0.01,
  p25: -0.005,
  p50: 0,
  p75: 0.005,
  p100: 0.01,
  mean: 0,
}}));
const option = globalThis.__shapChartOption({{
  plot_type: "box",
  title: "SHAP box plot: MAKE",
  x_feature: "MAKE",
  y_label: "SHAP",
  rows,
}}, {{}});
if (option.xAxis.axisLabel.rotate !== 60) throw new Error(`expected rotated labels, got ${{option.xAxis.axisLabel.rotate}}`);
if (option.xAxis.nameGap !== 88) throw new Error(`expected lower x-axis title gap, got ${{option.xAxis.nameGap}}`);
if (option.grid.bottom !== 54) throw new Error(`plot grid bottom should stay unchanged, got ${{option.grid.bottom}}`);
"""
        self.run_node_script(script)

    def test_index_uses_stable_local_asset_urls_and_disables_cache(self) -> None:
        _, body = self.assert_no_store("/")
        html = body.decode("utf-8")
        css = self.app_css_contract()
        js = self.app_js_contract()

        self.assertIn("<title>lucidum · sample.csv</title>", html)
        self.assertIn('href="/favicon.ico"', html)
        self.assertIn('src="/favicon.ico"', html)
        self.assertNotIn("<span>lucidum</span>", html)
        self.assertIn(".mark {\n        width: 48px;\n        height: 48px;", css)
        self.assertIn(".meta {\n        color: var(--muted);\n        font-size: 16px;", css)
        self.assertIn(".dataset-meta-gbm-link {", css)
        self.assertIn("function renderDatasetMeta(", js)
        self.assertIn('const payload = await api("/api/gbm/models", { method: "GET" });', js)
        self.assertIn("button.textContent = `GBMs (${datasetGbmCount.toLocaleString()})`;", js)
        self.assertIn("gbmTool.openModelNavigator();", js)
        self.assertIn("openModelNavigator,", js)
        self.assertIn('window.matchMedia("(prefers-color-scheme: dark)").matches', html)
        self.assertIn('document.body.classList.add("dark");', html)
        self.assertIn('id="sidebarToggleBtn"', html)
        self.assertIn('aria-controls="appSidebar"', html)
        self.assertIn('aria-label="Collapse sidebar"', html)
        self.assertIn('<aside id="appSidebar">', html)
        self.assertIn('id="profileTool" class="tool-option active" type="button" data-tool="column_profile" aria-label="Column profile"', html)
        self.assertIn('class="tool-label">Column profile</span>', html)
        self.assertIn('id="lineBarTool" class="tool-option" type="button" data-tool="line_bar" aria-label="Line and bar"', html)
        self.assertIn('class="tool-label">Line and bar</span>', html)
        self.assertIn('id="ukMapTool" class="tool-option" type="button" data-tool="uk_map" aria-label="UK mapping"', html)
        self.assertIn('class="tool-label">UK mapping</span>', html)
        self.assertIn('class="section sidebar-kpi-section hidden"', html)
        self.assertIn('id="lineBarToolbar" class="toolbar hidden"', html)
        self.assertIn('id="status" class="status main-status hidden"', html)
        self.assertIn('id="visualArea" class="visual-area profile-mode"', html)
        self.assertIn('id="chartSideControls" class="chart-side-controls hidden"', html)
        self.assertIn('id="chartControlsResizer" class="chart-controls-resizer hidden"', html)
        self.assertIn('id="lineBarTabs" class="tabs workspace-tabs hidden"', html)
        self.assertIn('id="lineBarGroupMeta" class="workspace-meta hidden"', html)
        self.assertIn('id="lineBarFilter" class="hidden">no filter</div>', html)
        self.assertIn('id="chart" class="hidden"', html)
        self.assertIn('id="profileWrap" class="profile-wrap"></div>', html)
        self.assertIn('id="themeBtn"', html)
        self.assertIn('aria-label="Switch to dark mode"', html)
        self.assertIn("theme-icon-moon", html)
        self.assertIn("theme-icon-sun", html)
        self.assertIn('id="monitorLink" class="ghost monitor-link header-icon-button" href="/monitor" target="_blank"', html)
        self.assertIn('aria-label="Open monitor" title="Open monitor"', html)
        self.assertIn('id="reloadBtn" class="ghost header-icon-button" type="button" aria-label="Reload dataset" title="Reload dataset"', html)
        self.assertIn('class="header-action-icon"', html)
        self.assertNotIn(">Monitor</a>", html)
        self.assertNotIn(">Reload</button>", html)
        self.assertNotIn('id="themeBtn" class="ghost">Dark</button>', html)
        self.assertNotIn("<h2>Tool</h2>", html)
        self.assertIn('href="/static/app.css"', html)
        self.assertIn('type="module" src="/static/app.js"', html)
        self.assertIn('id="modelToolWrap" class="model-tool-wrap hidden"', html)
        self.assertIn('id="gbmSidebarPanel" class="section gbm-sidebar-panel hidden"', html)
        self.assertIn('id="sidebarGbmResizer"', html)
        self.assertIn('aria-label="Resize KPI and GBM model controls"', html)
        self.assertIn('id="gbmModelCollapseBtn"', html)
        self.assertIn('<h2>GBMs</h2>', html)
        self.assertIn('id="gbmModelSelectedMeta"', html)
        self.assertIn('id="gbmModelSelect" class="feature-list gbm-model-list" role="listbox"', html)
        self.assertIn("No GBMs trained yet", js)
        self.assertIn(".gbm-model-list .gbm-empty-state", css)
        self.assertNotIn('id="gbmActiveModelSelect"', html)
        self.assertNotIn("?v=", html)

    def test_static_app_assets_disable_cache(self) -> None:
        self.assert_no_store("/static/app.js")
        self.assert_no_store("/static/app/main.js")
        self.assert_no_store("/static/app/column-profile-tool.js")
        self.assert_no_store("/static/app/line-bar-tool.js")
        self.assert_no_store("/static/app/uk-map-tool.js")
        self.assert_no_store("/static/app/shared/api.js")
        self.assert_no_store("/static/app/shared/format.js")
        self.assert_no_store("/static/app/shared/schema.js")
        self.assert_no_store("/static/app/shared/timing.js")
        self.assert_no_store("/static/app/gbm-tool.js")
        self.assert_no_store("/static/app/gbm-shap-tool.js")
        self.assert_no_store("/static/app/gbm-shap-chart.js")
        self.assert_no_store("/static/app/gbm-stacked-shap-tool.js")
        self.assert_no_store("/static/app/gbm-stacked-shap-chart.js")
        self.assert_no_store("/static/app/gbm-tree-viewer.js")
        self.assert_no_store("/static/app/model-tool-shell.js")
        self.assert_no_store("/static/vendor/tabulator/tabulator.min.js")
        self.assert_no_store("/static/vendor/tabulator/tabulator.min.css")
        self.assert_no_store("/static/vendor/d3/d3.min.js")
        self.assert_no_store("/static/vendor/echarts-gl/echarts-gl.min.js")
        app_css = self.assert_no_store("/static/app.css")[1].decode("utf-8")
        for path in self.CSS_MODULE_PATHS:
            import_path = f'.{path.removeprefix("/static")}'
            self.assertIn(f'@import url("{import_path}");', app_css)
        for path in self.CSS_MODULE_PATHS:
            self.assert_no_store(path)
        self.assert_no_store("/static/monitor.js")
        self.assert_no_store("/static/monitor.css")

    def test_gbm_frontend_contains_real_tool_contract(self) -> None:
        js = self.app_js_contract()
        css = self.app_css_contract()

        self.assertIn('import { createGbmTool } from "./gbm-tool.js";', js)
        self.assertIn('import { createGbmTreeViewer } from "./gbm-tree-viewer.js";', js)
        self.assertIn('import { createGbmShapTool } from "./gbm-shap-tool.js";', js)
        self.assertIn('import { emptyOption, ensureShapChartLibraries, shapChartOption } from "./gbm-shap-chart.js";', js)
        self.assertIn("export function createGbmTool", js)
        self.assertIn("export function createGbmTreeViewer", js)
        self.assertIn("export function createGbmShapTool", js)
        self.assertIn("function preselectFeatures(feature1, feature2 = \"\")", js)
        self.assertIn("function preselectFeature(value)", js)
        self.assertIn("preselectFeatures,", js)
        self.assertIn("preselectFeature,", js)
        self.assertIn('api("/api/gbm/config"', js)
        self.assertIn('api("/api/gbm/validate"', js)
        self.assertIn('api("/api/gbm/train"', js)
        self.assertIn('const GBM_RUNNING_POLL_MS = 500;', js)
        self.assertIn('const GBM_QUEUED_POLL_MS = 1000;', js)
        self.assertIn("function applyJobProgress(job)", js)
        self.assertIn("function renderLiveProgress(progress)", js)
        self.assertIn("job.progress", js)
        self.assertIn('`/api/gbm/models/${encodeURIComponent(currentModelId)}/trees`', js)
        self.assertIn('`/api/gbm/models/${encodeURIComponent(currentModelId)}/trees/${encodeURIComponent(selectedTree)}`', js)
        self.assertIn('`/api/gbm/models/${encodeURIComponent(modelId)}/shap/config`', js)
        self.assertIn('`/api/gbm/models/${encodeURIComponent(modelId)}/shap/plot`', js)
        self.assertIn('"/static/vendor/tabulator/tabulator.min.js"', js)
        self.assertIn('const D3_SRC = "/static/vendor/d3/d3.min.js";', js)
        self.assertIn('"/static/vendor/echarts-gl/echarts-gl.min.js"', js)
        self.assertIn("function flameRibbonSeries", js)
        self.assertIn('type: "custom"', js)
        self.assertIn("legend: centeredLegend(common.legend)", js)
        self.assertIn("function centeredLegend", js)
        self.assertIn("delete centered.right", js)
        self.assertIn("itemStyle: { color }", js)
        self.assertIn("itemStyle: { color: SHAP_RED }", js)
        self.assertIn("formatter: flameTooltipFormatter(rows, payload)", js)
        self.assertIn('class="gbm-shap-chart-shell"', js)
        self.assertIn("function featureImportanceLabel(feature, rank)", js)
        self.assertIn("formatMeanAbsShap(feature.mean_abs_shap)", js)
        self.assertIn("return features().some((feature) => featureNumber(feature.mean_abs_shap) !== null)", js)
        self.assertIn('? "shap"', js)
        self.assertIn("`Rank ${rank}`", js)
        self.assertNotIn("function featureKindLabel", js)
        self.assertIn(".gbm-shap-chart-shell", css)
        self.assertIn(".gbm-shap-message {\n        background:", css)
        self.assertIn("position: absolute;\n        right: 12px;\n        top: 12px;", css)
        self.assertIn(".gbm-shap-controls .gbm-shap-tail-control {\n        align-items: center;\n        text-align: center;", css)
        self.assertIn(".gbm-shap-tail-control .segmented {\n        justify-content: center;", css)
        self.assertIn(".gbm-shap-chart {\n        background: var(--panel);", css)
        self.assertIn("height: 100%;", css)
        self.assertIn("maximumFractionDigits: 4", js)
        self.assertIn("dataShape", js)
        self.assertIn("Number.NaN", js)
        self.assertIn("function surfaceDataPoint", js)
        self.assertIn("row?.has_data === false", js)
        self.assertIn("loadedSurfaceLibrary", js)
        self.assertIn("await nextAnimationFrame()", js)
        self.assertIn("function isSurfaceLayoutError", js)
        self.assertIn('setNotice("");', js)
        self.assertIn('root.dataset.gbmShapBound === "1"', js)
        self.assertIn(".gbm-shap-feature2-control", css)
        self.assertIn("d3.tree()", js)
        self.assertIn('append("rect")', js)
        self.assertIn('append("ellipse")', js)
        self.assertIn('marker-end', js)
        self.assertIn("edge_label", js)
        self.assertIn("const EDGE_LABEL_WRAP_CHARS = 34;", js)
        self.assertIn("const CATEGORICAL_EDGE_LABEL_WRAP_CHARS = 20;", js)
        self.assertIn("const CATEGORICAL_EDGE_LABEL_X_OFFSET = 32;", js)
        self.assertIn("function edgeLabelPlacement(link)", js)
        self.assertIn("function edgeLabelLines(label)", js)
        self.assertIn("function nodePathIds(node)", js)
        self.assertIn("function updateTreeHighlight(nodeId)", js)
        self.assertIn("gbm-tree-node-highlighted", css)
        self.assertIn("gbm-tree-link-highlighted", css)
        self.assertIn('wrapDelimitedLabel(text, " / ", CATEGORICAL_EDGE_LABEL_WRAP_CHARS)', js)
        self.assertIn("default_branch", js)
        self.assertIn('data-gbm-tree-palette="plain"', js)
        self.assertIn('data-gbm-tree-palette="divergent"', js)
        self.assertIn('data-gbm-tree-palette="spectral"', js)
        self.assertIn('data-gbm-tree-palette="viridis"', js)
        self.assertIn('data-gbm-tree-zoom="out"', js)
        self.assertIn('data-gbm-tree-zoom="reset"', js)
        self.assertIn('data-gbm-tree-zoom="in"', js)
        self.assertIn('id="gbmTreeResizer" class="gbm-tree-resizer"', js)
        self.assertIn("const DEFAULT_SUMMARY_WIDTH = 560;", js)
        self.assertIn("function bindTreeResizer(root)", js)
        self.assertIn("function clampSummaryWidth(root, rawWidth)", js)
        self.assertIn('root.style.setProperty("--gbm-tree-summary-width"', js)
        self.assertIn('{ title: "tree", field: "tree", width: 46, minWidth: 46', js)
        self.assertIn('{ title: "dim", field: "dim", width: 42, minWidth: 42', js)
        self.assertIn("function readableTextColor(fill)", js)
        self.assertIn("function contrastRatio(left, right)", js)
        self.assertIn(".scaleExtent([0.03, 24])", js)
        self.assertIn("function finishSummaryResize()", js)
        self.assertIn("new ResizeObserver(scheduleSvgResize)", js)
        self.assertIn("function updateSvgSize()", js)
        self.assertIn("function fitTree(animated = true)", js)
        self.assertIn("zoomBehavior.scaleBy, scale, [width / 2, height / 2]", js)
        self.assertIn('zoomBehavior.transform, resetTransform', js)
        self.assertIn('divergent: ["#00441b", "#1b7837", "#5aae61", "#a6dba0", "#d9f0d3", "#fddbc7", "#f4a582", "#d6604d", "#b2182b", "#67001f"]', js)
        self.assertIn('spectral: ["#2c7bb6", "#00a6ca", "#00ccbc", "#90eb9d", "#ffff8c", "#f9d057", "#f29e2e", "#e76818", "#d7191c", "#a50026"]', js)
        self.assertIn('viridis: ["#fde725", "#b5de2b", "#6ece58", "#35b779", "#1f9e89", "#26828e", "#31688e", "#3e4989", "#482878", "#440154"]', js)
        self.assertIn('id="gbmTreeViewer" class="gbm-tree-viewer"', js)
        self.assertIn('id="gbmTreeSummaryGrid" class="gbm-grid gbm-tree-summary-grid"', js)
        self.assertIn('id="gbmTreeSearch" class="gbm-tree-search"', js)
        self.assertIn('id="gbmTreeDetailSummary" class="gbm-tree-detail-summary"', js)
        self.assertIn('id="gbmTreeSvgMount" class="gbm-tree-svg-mount"', js)
        self.assertIn("function updateTreeDetailSummary(row)", js)
        self.assertIn("function updateTreePalette()", js)
        self.assertIn('selection.select("rect.gbm-tree-split-node, ellipse.gbm-tree-leaf-node").attr("fill", fill);', js)
        self.assertIn("const NODE_LINE_HEIGHT = 18;", js)
        self.assertIn("const NODE_VERTICAL_PADDING = 14;", js)
        self.assertIn('"1.24em"', js)
        self.assertNotIn('selection.append("title")', js)
        self.assertIn("Tree ${escapeHtml(row.tree)}", js)
        self.assertIn("Dimensionality:", js)
        self.assertIn("Tree features:", js)
        self.assertIn("Tree gain:", js)
        self.assertIn("function isEmphasisLabelLine(node, index)", js)
        self.assertIn('.attr("font-weight", (line) => line.emphasis ? 700 : 400)', js)
        self.assertNotIn('type: "tree"', js)
        self.assertNotIn("function treeNode(node)", js)
        self.assertIn('title: shapMode ? "SHAP" : "Gain"', js)
        self.assertNotIn('{ title: "Type", field: "kind"', js)
        self.assertNotIn("<th>Type</th>", js)
        self.assertIn("formatter: featureNameFormatter", js)
        self.assertIn('cssClass: "gbm-feature-name-cell"', js)
        self.assertIn("function featureNameFormatter(cell)", js)
        self.assertIn("function featureTypeLabel(feature)", js)
        self.assertIn('if (isInvalidFeature(feature) || kind === "invalid") return "invalid";', js)
        self.assertIn("function isInvalidFeature(feature)", js)
        self.assertIn("gbm-feature-invalid", js)
        self.assertIn("return `categorical (${count.toLocaleString()})`;", js)
        self.assertIn('class="gbm-feature-kind kind"', js)
        self.assertIn('id="gbmFeatureScenarioDropdown"', js)
        self.assertIn('id="gbmFeatureScenarioButton"', js)
        self.assertIn('id="gbmFeatureScenarioMenu"', js)
        self.assertIn('id="gbmFeatureInteractionConstraintSelect"', js)
        self.assertIn('data-gbm-feature-menu-root', js)
        self.assertIn('id="gbmFeatureMetricToggle" class="gbm-feature-metric-toggle"', js)
        self.assertIn('value="${escapeHtml(mode)}"', js)
        self.assertIn('if (mode === "shap") return "SHAP";', js)
        self.assertIn('if (mode === "gain_ebm") return "EBM Gain";', js)
        self.assertLess(js.index('modes.push("gain_ebm");'), js.index('modes.push("gain");'))
        self.assertLess(js.index('modes.push("gain");'), js.index('modes.push("shap");'))
        self.assertLess(js.index('modes.push("gain_ebm");'), js.index('modes.push("shap");'))
        self.assertIn("function defaultFeatureMetricMode(modes = [])", js)
        self.assertIn('id="gbmEbmGainSummaryGrid"', js)
        self.assertIn('GET" });', js)
        self.assertIn('/ebm-gain-summary', js)
        self.assertLess(js.index('featureMetricToggleHtml(data.features || [], data)'), js.index("featureInteractionConstraintDropdownHtml"))
        self.assertIn("function featureMetricToggleAvailable(features = [], data = config)", js)
        self.assertIn("function featureMetricColumn()", js)
        self.assertIn("function ebmGainSummaryColumns()", js)
        self.assertIn("function renderEbmGainSummaryFallback(rows)", js)
        self.assertIn('featureTable.on("rowContext", openFeatureContextMenuForTabulatorRow);', js)
        self.assertIn('ebmGainSummaryTable.on("rowContext", openEbmGainContextMenuForTabulatorRow);', js)
        self.assertIn("function openGbmFeatureContextMenu(event, context)", js)
        self.assertIn("function gbmFeatureContextActions(context = {})", js)
        self.assertIn('label: "Toggle interaction constraint"', js)
        self.assertIn('label: "Toggle group interaction constraint"', js)
        self.assertIn('label: "Clear all monotonicities"', js)
        self.assertIn('label: "Copy importance value"', js)
        self.assertIn("function toggleFeatureInteractionConstraint(featureName)", js)
        self.assertIn("function toggleGroupInteractionConstraint(grouping)", js)
        self.assertIn("function clearAllFeatureMonotonicities()", js)
        self.assertIn("function copyFeatureImportanceValue(featureName)", js)
        self.assertIn('divider.className = "gbm-feature-context-menu-divider";', js)
        self.assertIn("canNavigateToLineBarFeature,", js)
        self.assertIn("navigateToLineBarFeature,", js)
        self.assertIn("function canNavigateToLineBarFeature(featureName)", js)
        self.assertIn("function navigateToLineBarFeature(featureName)", js)
        self.assertIn("state.bandFeature = null;", js)
        self.assertIn('label: "Go to Line and Bar"', js)
        self.assertIn('label: "Go to SHAP"', js)
        self.assertIn('label: "Go to Stacked SHAP"', js)
        self.assertIn("function closeGbmFeatureContextMenu()", js)
        self.assertIn('data-gbm-ebm-features="${escapeHtml(JSON.stringify(ebmSummaryFeatures(row)))}"', js)
        self.assertIn("shapTool.preselectFeatures(features[0], features[1] || \"\");", js)
        self.assertIn("stackedShapTool.preselectFeature(name);", js)
        self.assertIn("function formatMeanAbsShap(value)", js)
        self.assertIn("function formatGainPercent(value)", js)
        self.assertIn("let featureDraftState = null;", js)
        self.assertIn("function captureFeatureDraftStateForRender(nextData = config)", js)
        self.assertIn("function applyFeatureDraftStateToData(data = {})", js)
        self.assertIn("function featureInteractionGroupingsEdited(groupings, data = config)", js)
        self.assertIn("function featureScenarioSelectionEdited(name, data = config)", js)
        self.assertIn("config = applyFeatureDraftStateToData(data);", js)
        self.assertIn("function featureInteractionConstraintDropdownHtml(groupings, activeConstraints = null, features = [])", js)
        self.assertIn("const draftGroupings = draft?.interactionGroupingsEdited ? new Set(draft.interactionGroupings || []) : null;", js)
        self.assertIn("function renderedInteractionFeatureNames(features)", js)
        self.assertIn("const groupLocked = renderedInteractionFeatureNames(features);", js)
        self.assertIn("const featureLocked = draft ? selectedFeatureInteractionFeatureNames(features) : activeFeatureInteractionFeatureNames();", js)
        self.assertIn("function currentFeatureInteractionGroupingsPayload()", js)
        self.assertIn("function currentFeatureInteractionFeaturesPayload()", js)
        self.assertIn("function normaliseActiveFeatureInteractionConstraints(activeConstraints)", js)
        self.assertIn("function syncFeatureInteractionControls()", js)
        self.assertIn("function syncFeatureInteractionCounts(features = currentFeatureRows())", js)
        self.assertIn("beforeOpen: () => syncFeatureInteractionCounts(currentFeatureRows()),", js)
        self.assertIn("function featureScenarioDropdownHtml(scenarios, activeScenario = null)", js)
        self.assertIn("function applyFeatureScenario(name)", js)
        self.assertIn("function resetFeatureScenarioSelection()", js)
        self.assertIn("function bindFeatureScenarioActions()", js)
        self.assertIn("function bindGbmFeatureToolbarMenu(root, { beforeOpen = null } = {})", js)
        self.assertIn('document.addEventListener("click", () => closeGbmFeatureToolbarMenus());', js)
        self.assertIn("function normaliseActiveFeatureScenario(activeScenario)", js)
        self.assertIn("function currentFeatureScenarioPayload()", js)
        self.assertIn("payload.feature_scenario = featureScenario;", js)
        self.assertIn("payload.feature_interaction_groupings = featureInteractionGroupings;", js)
        self.assertIn("payload.feature_interaction_features = featureInteractionFeatures;", js)
        self.assertIn("if (currentModelId !== nextModelId) featureDraftState = null;", js)
        self.assertNotIn("config = nextConfig;\n    activeDetail = null;", js)
        self.assertIn("(trained; spec changed)", js)
        self.assertIn("(trained; missing from spec)", js)
        self.assertIn("function groupingFormatter(cell)", js)
        self.assertIn('&#128274;', js)
        self.assertIn('{ title: "Grouping", field: "grouping", formatter: groupingFormatter', js)
        self.assertIn("<th>Grouping</th>", js)
        self.assertIn('width: 120', js)
        self.assertIn('width: 125', js)
        self.assertIn("featureMetricDisplay(feature)", js)
        self.assertIn("formatter: useCheckboxFormatter", js)
        self.assertIn("function useCheckboxFormatter(cell)", js)
        self.assertIn("if (!isFeatureSelectable(rowData)) return \"\";", js)
        self.assertIn('className = "gbm-use-checkbox"', js)
        self.assertNotIn('formatter: "tickCross"', js)
        self.assertIn("initialSort: featureTableInitialSort()", js)
        self.assertIn('Features and parameters', js)
        self.assertIn('Model navigator', js)
        self.assertIn('Tree viewer', js)
        self.assertIn('SHAP</button>', js)
        self.assertIn('class="gbm-toolbar"', js)
        self.assertIn('id="gbmTrainingStatus" class="gbm-training-status', js)
        self.assertIn("setTrainingStatus(progress.message || \"\", progress.phase || \"\", trainingStatusDetail(progress));", js)
        self.assertIn("function trainingStatusDetail(progress)", js)
        self.assertIn("grid_parameters", js)
        self.assertIn("renderEvaluationChart({", js)
        self.assertIn("progress.evaluation", js)
        self.assertIn('pollJob(job.job_id, 0);', js)
        self.assertIn('job.status === "running" ? GBM_RUNNING_POLL_MS : GBM_QUEUED_POLL_MS', js)
        self.assertIn('if (!job.progress) setTrainingStatus("GBM failed", "failed");', js)
        self.assertLess(js.index('id="gbmTrainBtn"'), js.index("sampleStatusHtml(data.sample)"))
        self.assertLess(js.index("sampleStatusHtml(data.sample)"), js.index('id="gbmShapRows"'))
        self.assertLess(js.index('id="gbmShapRows"'), js.index("trainingModeHtml(data.training_mode)"))
        self.assertIn('id="gbmSampleStatus"', js)
        self.assertIn('id="gbmClearFeaturesBtn" class="tab gbm-inline-action-button gbm-icon-action-button" type="button" aria-label="Clear all features" title="Clear all">×</button>', js)
        self.assertIn('id="gbmSelectFeaturesBtn" class="tab gbm-inline-action-button gbm-icon-action-button" type="button" aria-label="Select all features" title="Select all">✓</button>', js)
        self.assertIn('class="gbm-feature-menu-button gbm-interaction-constraint-button${constraintClass}"', js)
        self.assertIn('title="Constrain selected grouped features so they only interact within selected groups"', js)
        self.assertIn('title="Apply a saved feature scenario to the Feature table"', js)
        self.assertIn('button.classList.toggle("has-constraints", selectedCurrentCount + syntheticCount > 0);', js)
        self.assertIn("function createSampleColumn()", js)
        self.assertIn('api("/api/gbm/sample"', js)
        self.assertIn("function setFeatureIncludes(include)", js)
        self.assertIn('id="gbmFeatureSectionTitle" class="gbm-section-title"', js)
        self.assertIn("function featureSectionTitle(features)", js)
        self.assertIn("function syncFeatureSectionTitle()", js)
        self.assertIn('class="gbm-section-title">Parameters</h3>', js)
        self.assertIn('class="gbm-section-title">Control</h3>', js)
        self.assertIn('class="gbm-section-title">Evaluation Log</h3>', js)
        self.assertIn('class="gbm-section-header gbm-evaluation-section-header"', js)
        self.assertIn('id="gbmNotice" class="gbm-notice hidden"', js)
        self.assertIn("function setGbmNotice(message)", js)
        self.assertIn("setGbmNotice(validation.errors.join(\"; \"));", js)
        self.assertIn("setGbmNotice(job.error || \"GBM training failed\");", js)
        self.assertIn("setGbmNotice(error.message);", js)
        self.assertNotIn("A model-local training/test sample will be created when this GBM is trained.", js)
        self.assertNotIn('setStatus("GBM trained.");', js)
        self.assertNotIn('setStatus("GBM model activated.");', js)
        self.assertIn('role="radiogroup" aria-label="SHAP rows"', js)
        self.assertIn('class="gbm-shap-options"', js)
        self.assertIn('type="radio" name="gbmShapRows"', js)
        self.assertIn('{ value: "100k", label: "100k" }', js)
        self.assertIn('document.querySelector("input[name=\'gbmShapRows\']:checked")?.value || "0"', js)
        self.assertIn('id="gbmTrainingMode" class="gbm-shap-rows gbm-mode-rows"', js)
        self.assertIn('role="radiogroup" aria-label="Training mode"', js)
        self.assertIn('type="radio" name="gbmTrainingMode" value="ebm"', js)
        self.assertIn("EBM starts with 2-leaf trees at learning rate 0.3", js)
        self.assertIn("training_mode: currentTrainingMode()", js)
        self.assertNotIn("Zero rows", js)
        self.assertNotIn("10k rows", js)
        self.assertIn("function setTrainingState(active)", js)
        self.assertIn("setTrainingState(true)", js)
        self.assertIn("setTrainingState(false)", js)
        self.assertIn('isTraining ? "Training..." : "Train GBM"', js)
        self.assertIn("if (magnitude >= 1000) return Math.round(number).toLocaleString();", js)
        self.assertIn("let evaluationChart = null;", js)
        self.assertIn("function bindEvaluationResize(target)", js)
        self.assertIn("new ResizeObserver", js)
        self.assertIn("evaluationChart?.resize()", js)
        self.assertIn("function evaluationTitle(rows, primaryMetric, manifest = {}, progress = null)", js)
        self.assertIn("rows.sort(compareEvaluationRows);", js)
        self.assertIn("function compareEvaluationRows(left, right)", js)
        self.assertIn('let evaluationViewMode = "all";', js)
        self.assertIn('id="gbmEvaluationViewMode"', js)
        self.assertIn('name="gbmEvaluationViewMode"', js)
        self.assertIn("function bindEvaluationViewModeActions()", js)
        self.assertIn("function rerenderEvaluationChart()", js)
        self.assertIn("function evaluationTailWindowSize(maxIteration, detail)", js)
        self.assertIn("earlyStoppingRounds * 5", js)
        self.assertIn("Math.ceil(count * 0.2)", js)
        self.assertIn("function evaluationTailYAxisBounds(rows, primaryMetric, xDomain)", js)
        self.assertIn("function evaluationTailYAxisPadding(extent)", js)
        self.assertIn("range * 0.2", js)
        self.assertNotIn("Math.abs(extent.max) * 0.01", js)
        self.assertIn("function evaluationTailFocusRow(rows, primaryMetric)", js)
        self.assertIn("parts.push(`evaluation metric: ${metric}`);", js)
        self.assertIn("test metric: ${formatEvaluationValue(bestValue)}", js)
        self.assertIn("return parts.join(\", \");", js)
        self.assertNotIn("subtext: title.subtext", js)
        self.assertNotIn("subtextStyle:", js)
        self.assertIn("legend: {", js)
        self.assertIn('orient: "vertical"', js)
        self.assertIn("grid: { left: 12, right: 82, top: 42, bottom: 20, containLabel: true }", js)
        self.assertIn('type: "value",', js)
        self.assertIn("interval: xInterval", js)
        self.assertIn("const GBM_EVALUATION_DOWNSAMPLE_THRESHOLD = 2000;", js)
        self.assertIn("const GBM_EVALUATION_MAX_PLOT_POINTS = 1500;", js)
        self.assertIn("function evaluationXAxisMax(maxIteration, progress = null)", js)
        self.assertIn('progress?.phase === "training"', js)
        self.assertIn("function evaluationYAxisMax(rows, maxIteration)", js)
        self.assertIn("const tailStart = evaluationTailStart(maxIteration);", js)
        self.assertIn("function evaluationTooltipFormatter(params)", js)
        self.assertIn("function evaluationSampledIndexes(rows, maxIteration, manifest = {}, progress = null, xDomain = null)", js)
        self.assertIn("function evaluationIndexRange(maxIteration, xDomain = null)", js)
        self.assertIn("function evaluationCompositePoints(rows, startIndex, endIndex)", js)
        self.assertIn("function largestTriangleThreeBuckets(points, threshold)", js)
        self.assertIn("function evaluationSeriesData(values, sampledIndexes)", js)
        self.assertIn("data: evaluationSeriesData(row.values, sampledEvaluationIndexes)", js)
        self.assertIn("<strong>Iteration:</strong>", js)
        self.assertIn("hideOverlap: false", js)
        self.assertIn("margin: 4", js)
        self.assertIn("function niceIterationInterval(maxIteration)", js)
        self.assertIn("function niceIterationLabelInterval(maxIteration)", js)
        self.assertIn("function evaluationIterationAxisLabel(value, labelInterval)", js)
        self.assertIn("Number(maxIteration || 1) / 30", js)
        self.assertIn("Number(maxIteration || 1) / 10", js)
        self.assertIn('const textColor = cssVar("--text", "#3f3f46");', js)
        self.assertIn('const mutedColor = cssVar("--muted", "#4b5563");', js)
        self.assertIn('const lineColor = cssVar("--line", "#e5e7eb");', js)
        self.assertIn('color: ["#ff140f", cssVar("--actual-line", "#050505"), "#2563eb", "#7c3aed"]', js)
        self.assertIn("axisLabel: { color: mutedColor, formatter: (value) => formatEvaluationAxisValue(value) }", js)
        self.assertIn("splitLine: { lineStyle: { color: lineColor } }", js)
        self.assertIn("function cssVar(name, fallback)", js)
        self.assertIn("refreshTheme() {", js)
        self.assertIn("if (liveProgress?.evaluation)", js)
        self.assertIn("treeViewer.refreshTheme();", js)
        self.assertIn("showSymbol: true", js)
        self.assertIn("const GBM_PARAMETER_OPTIONS = {", js)
        self.assertIn('"cross_entropy_lambda"', js)
        self.assertIn('"average_precision"', js)
        self.assertIn('{ title: "Value", field: "value", editor: "adaptable", editorParams: parameterValueEditorParams(), widthGrow: 1 }', js)
        self.assertIn("function parameterValueEditorParams()", js)
        self.assertIn("function parameterValueEditorLookup(cell)", js)
        self.assertIn("function parameterValueEditorParamsLookup(editor, cell)", js)
        self.assertIn("editorLookup: parameterValueEditorLookup", js)
        self.assertIn("paramsLookup: parameterValueEditorParamsLookup", js)
        self.assertIn("class: `gbm-parameter-editor gbm-parameter-${editor}-editor`,", js)
        self.assertIn("function parameterControlHtml(parameter)", js)
        self.assertIn("<select data-gbm-parameter=", js)
        css = self.app_css_contract()
        self.assertIn(".gbm-grid .tabulator-row .tabulator-cell", css)
        self.assertIn(".gbm-grid .tabulator-row.tabulator-row-even", css)
        self.assertIn("background: var(--panel) !important", css)
        self.assertIn(".gbm-tabs .tab {\n        font-size: 12px;\n        font-weight: 700;", css)
        self.assertIn(".gbm-section-title {\n        color: var(--text);\n        flex: 0 0 auto;\n        font-size: 13px;\n        font-weight: 700;", css)
        self.assertIn(".gbm-grid {\n        background: var(--panel);\n        border: 1px solid var(--gbm-table-border);\n        border-radius: 6px;\n        color: var(--text);", css)
        self.assertIn(".gbm-grid.tabulator {\n        border-color: var(--gbm-table-border);", css)
        self.assertIn("color: var(--text) !important", css)
        self.assertIn(".gbm-grid.tabulator .tabulator-header .tabulator-col {\n        justify-content: center;\n        min-height: 20px;\n        font-size: 11px;\n        line-height: 1.15;", css)
        self.assertIn(".gbm-grid.tabulator .tabulator-header .tabulator-col .tabulator-col-content {\n        align-items: center;\n        display: flex;\n        min-height: 20px;\n        padding: 1px 6px;", css)
        self.assertIn(".gbm-grid .tabulator-row .tabulator-cell {\n        align-items: center;\n        background: transparent !important;\n        border-right-color: color-mix(in srgb, var(--line) 80%, transparent);\n        color: var(--text) !important;\n        display: inline-flex;\n        min-height: 20px;\n        padding: 1px 6px;\n        font-size: 11px;\n        line-height: 1.15;", css)
        self.assertIn('#gbmFeatureGrid .tabulator-cell[tabulator-field="include"] {\n        justify-content: center;', css)
        self.assertIn('#gbmFeatureGrid .tabulator-cell[tabulator-field="mean_abs_shap"]', css)
        self.assertIn("#gbmFeatureGrid .tabulator-row .gbm-feature-name-cell,\n      .gbm-feature-name-line {\n        align-items: center;\n        gap: 8px;\n        justify-content: space-between;", css)
        self.assertIn(".gbm-feature-kind {\n        color: var(--muted);\n        flex: 0 0 auto;\n        margin-left: auto;\n        text-align: right;", css)
        self.assertIn("#gbmFeatureGrid .tabulator-row.gbm-feature-disabled", css)
        self.assertIn("#gbmFeatureGrid .tabulator-row.gbm-feature-invalid", css)
        self.assertIn("#gbmFeatureGrid .tabulator-row.gbm-feature-warning", css)
        self.assertIn("body.dark #gbmFeatureGrid .tabulator-row.gbm-feature-disabled", css)
        self.assertIn("body.dark #gbmFeatureGrid .tabulator-row.gbm-feature-invalid", css)
        self.assertIn("color: #fecaca !important;", css)
        self.assertIn("body.dark #gbmFeatureGrid .tabulator-row.gbm-feature-warning", css)
        self.assertIn(".gbm-gain-cell", css)
        self.assertIn(".gbm-train-button.training", css)
        self.assertIn("background: #d97706", css)
        self.assertIn(".gbm-tool {\n        --gbm-table-border: #999;\n        display: flex;", css)
        self.assertIn("position: relative;", css)
        self.assertIn(".gbm-notice", css)
        self.assertIn(".gbm-training-status", css)
        self.assertIn('.gbm-training-status[data-phase="failed"]', css)
        self.assertIn("position: absolute;", css)
        self.assertIn(".gbm-sample-status", css)
        self.assertIn(".gbm-feature-actions", css)
        self.assertIn(".gbm-feature-metric-toggle", css)
        self.assertIn(".gbm-feature-metric-option.active", css)
        self.assertIn(".gbm-feature-metric-cell", css)
        self.assertIn(".gbm-feature-context-menu", css)
        self.assertIn(".gbm-feature-context-menu-item", css)
        self.assertIn(".gbm-feature-context-menu-divider", css)
        self.assertIn(".gbm-interaction-constraint-select", css)
        self.assertIn(".gbm-feature-menu-button", css)
        self.assertIn(".gbm-feature-menu", css)
        self.assertIn(".gbm-interaction-lock", css)
        self.assertIn(".gbm-feature-interaction-lock", css)
        self.assertIn(".gbm-feature-scenario-select", css)
        self.assertIn(".gbm-feature-scenario-row", css)
        self.assertIn(".gbm-evaluation-view-mode", css)
        self.assertIn(".gbm-evaluation-view-option", css)
        self.assertIn("accent-color: var(--accent);", css)
        self.assertIn(".gbm-inline-action-button", css)
        self.assertIn(".gbm-icon-action-button", css)
        self.assertIn(".gbm-feature-menu-button.has-constraints", css)
        self.assertIn(".gbm-parameter-layout {\n        display: grid;", css)
        self.assertIn("grid-template-columns: minmax(0, 7fr) minmax(150px, 3fr);", css)
        self.assertIn(".gbm-parameter-controls-column {\n        display: flex;\n        flex-direction: column;", css)
        self.assertIn(".gbm-section-header {\n        align-items: flex-start;", css)
        self.assertIn(".gbm-parameter-table-column > .gbm-section-title,\n      .gbm-parameter-controls-column > .gbm-section-title", css)
        self.assertIn(".gbm-actions {\n        display: flex;\n        align-items: stretch;\n        flex-direction: column;", css)
        self.assertIn(".gbm-action-button {\n        flex: 0 0 auto;\n        min-height: 24px;\n        width: 100%;", css)
        self.assertIn(".gbm-shap-rows {\n        display: flex;\n        align-items: flex-start;\n        flex: 0 0 auto;\n        flex-direction: column;", css)
        self.assertIn(".gbm-shap-label {\n        color: var(--text);\n        display: block;\n        font-size: 13px;\n        font-weight: 700;", css)
        self.assertIn(".gbm-shap-options {\n        display: flex;\n        flex-wrap: nowrap;", css)
        self.assertIn(".gbm-shap-option {\n        align-items: center;", css)
        self.assertIn(".gbm-shap-option:has(input:checked)", css)
        self.assertIn("flex: 1 1 0;", css)
        self.assertIn("justify-content: center;", css)
        self.assertIn("background: var(--panel);\n        width: 100%;", css)
        self.assertIn(".gbm-shap-option input {\n        block-size: 1px;\n        inline-size: 1px;", css)
        self.assertIn("opacity: 0;\n        pointer-events: none;\n        position: absolute;", css)
        self.assertIn("grid-template-rows: 330px minmax(0, 1fr)", css)
        self.assertIn(".gbm-evaluation-chart {\n        min-height: 220px;\n        border: 1px solid var(--gbm-table-border);", css)
        self.assertIn(".gbm-evaluation-chart {\n        height: 100%;", css)
        self.assertIn(".gbm-tree-viewer {\n        display: grid;", css)
        self.assertIn("grid-template-columns: minmax(420px, var(--gbm-tree-summary-width, 560px)) 7px minmax(360px, 1fr);", css)
        self.assertIn(".gbm-tree-summary-panel,\n      .gbm-tree-diagram-panel {\n        min-height: 0;", css)
        self.assertIn(".gbm-tree-summary-grid {\n        flex: 1 1 auto;", css)
        self.assertIn(".gbm-tree-resizer {\n        align-self: stretch;", css)
        self.assertIn(".gbm-tree-resizer:hover,\n      .gbm-tree-resizer.dragging", css)
        self.assertIn(".gbm-tree-summary-grid .tabulator-row.tabulator-selected", css)
        self.assertIn(".gbm-tree-fallback-table tr.active", css)
        self.assertIn("body.dark .gbm-tree-summary-grid .tabulator-row.tabulator-selected", css)
        self.assertIn(".gbm-tree-detail-summary", css)
        self.assertIn(".gbm-tree-detail-title", css)
        self.assertIn(".gbm-tree-detail-line", css)
        self.assertIn(".gbm-tree-svg-mount", css)
        self.assertIn(".gbm-tree-controls {\n        align-items: center;", css)
        self.assertIn(".gbm-tree-zoom button {\n        min-width: 28px;", css)
        self.assertIn(".gbm-tree-viewer.resizing", css)
        self.assertIn(".gbm-tree-chart {\n        --gbm-tree-split-fill: #dbeafe;", css)
        self.assertIn("body.dark .gbm-tree-chart", css)
        self.assertIn(".gbm-tree-svg", css)
        self.assertIn(".gbm-tree-link-default", css)
        self.assertIn(".gbm-tree-edge-label", css)
        self.assertIn(".gbm-tree-split-node,\n      .gbm-tree-leaf-node", css)
        self.assertIn(".gbm-tree-node-label {\n        font-size: 12px;\n        font-weight: 400;", css)
        self.assertIn(".gbm-model-navigator {\n        background: var(--panel);", css)
        self.assertIn(".gbm-model-actions {\n        align-items: center;", css)
        self.assertIn(".gbm-model-grid .tabulator-row.tabulator-selected", css)
        self.assertIn(".gbm-model-active-dot", css)
        self.assertNotIn("gbm-model-active-row", css)
        self.assertIn("#gbmModelFallback {\n        flex: 1 1 auto;", css)
        self.assertIn("function syncRenderedTab(mount, nextTab)", js)
        self.assertIn("syncRenderedTab(mount, activeTab);\n      scheduleGbmTableRedraws();\n      refreshModelList({ force: true });\n      return;", js)
        self.assertIn(".gbm-model-table {\n        font-size: 11px;\n        line-height: 1.15;", css)
        self.assertIn("min-width: 1620px;", css)
        self.assertIn(".gbm-model-table td {\n        border-right: 1px solid color-mix(in srgb, var(--line) 80%, transparent);", css)
        self.assertNotIn(".gbm-model-activate-button", css)
        self.assertIn(".gbm-parameter-select", css)
        self.assertIn(".gbm-fallback-table select", css)
        self.assertIn('document.querySelector(".sidebar-kpi-section")?.classList.toggle("hidden", tool === "column_profile" || tool === "glm");', js)
        self.assertIn('document.querySelector(".sidebar-filter-section")?.classList.toggle("hidden", isModelTool(tool));', js)
        self.assertIn('el("modelToolGroupMeta").classList.toggle("hidden", !isModelTool(tool) || tool === "gbm");', js)
        self.assertIn('el("modelToolFilter").classList.toggle("hidden", !isModelTool(tool) || tool === "gbm");', js)
        self.assertIn('response: el("actualNumerator")?.value || "actualNumerator"', js)
        self.assertIn('offset: el("denominator")?.value || "denominator"', js)
        self.assertIn("const gbmSourcesAvailable = (state.schema?.data_sources || []).some", js)
        self.assertIn("gbmTool.syncSidebarFromSchema();", js)
        self.assertIn("function syncSidebarFromSchema()", js)
        self.assertIn("function syncSidebarModelChooser(models, activeModelId)", js)
        self.assertIn("function modelGroupLabel(model)", js)
        self.assertIn("const modelsByGroup = new Map();", js)
        self.assertIn("if (!modelsByGroup.has(group)) modelsByGroup.set(group, []);", js)
        self.assertIn("modelsByGroup.get(group).push(model);", js)
        self.assertIn('return `${model.response_column || "actualNumerator"} / ${modelWeightLabel(model.offset_column)}`;', js)
        self.assertIn("function modelDetailLabel(model)", js)
        self.assertIn("return gbmModelDetailLabel(model);", js)
        self.assertIn('parts.push(`train ${formatModelMetric(modelBestMetric(model, "training"))}`);', js)
        self.assertIn('parts.push(`test ${formatModelMetric(modelBestMetric(model, "test"))}`);', js)
        self.assertIn('heading.className = "saved-filter-theme gbm-model-theme";', js)
        self.assertIn('button.className = `feature gbm-model-option${active ? " active" : ""}`;', js)
        self.assertIn('id="gbmRenameModelBtn"', js)
        self.assertIn('id="gbmActivateModelBtn" class="tab gbm-inline-action-button" type="button">Activate</button>', js)
        self.assertIn('id="gbmDeleteModelBtn" class="danger-action gbm-model-delete-button"', js)
        self.assertLess(js.index('id="gbmRenameModelBtn"'), js.index('id="gbmActivateModelBtn"'))
        self.assertLess(js.index('id="gbmActivateModelBtn"'), js.index('id="gbmDeleteModelBtn"'))
        self.assertIn('id="gbmModelGrid" class="gbm-grid gbm-model-grid"', js)
        self.assertIn("function modelRows(models)", js)
        self.assertIn('selectableRows: true,', js)
        self.assertIn('selectableRowsRangeMode: "click",', js)
        self.assertIn('modelTable.on("rowSelectionChanged", syncModelActionButtons);\n      syncModelActionButtons();', js)
        self.assertIn('{ title: "", field: "active", formatter: activeModelDotFormatter', js)
        self.assertIn("function activeModelDotFormatter(cell)", js)
        self.assertIn('class="gbm-model-active-dot"', js)
        self.assertIn('title: "Response", field: "response_column", sorter: "string"', js)
        self.assertIn('title: "Weight", field: "weight_display", sorter: "string"', js)
        self.assertIn('title: "Mode", field: "training_mode_display", sorter: "string"', js)
        self.assertIn('title: "Constraints", field: "constraint_display", sorter: "string"', js)
        self.assertIn("function modelInteractionConstraintLabel(rawConstraints)", js)
        self.assertIn('title: "Train", field: "training_rows", sorter: "number"', js)
        self.assertIn('title: "Best iter.", field: "best_iteration", sorter: "number"', js)
        self.assertIn('layout: "fitDataStretch"', js)
        self.assertIn('title: "tr@best", field: "best_training_metric", sorter: "number"', js)
        self.assertIn('title: "te@best", field: "best_test_metric", sorter: "number"', js)
        self.assertIn('width: 96, headerSort: true, headerTooltip: "Training metric at best iteration"', js)
        self.assertIn('width: 96, headerSort: true, headerTooltip: "Test metric at best iteration"', js)
        self.assertIn('title: "n_iter", field: "param_num_iterations", sorter: "number"', js)
        self.assertIn('title: "lr", field: "param_learning_rate", sorter: "number"', js)
        self.assertIn('title: "leaves", field: "param_num_leaves", sorter: "number"', js)
        self.assertIn('title: "depth", field: "param_max_depth", sorter: "number"', js)
        self.assertIn('title: "min_leaf", field: "param_min_data_in_leaf", sorter: "number"', js)
        self.assertIn('title: "ES", field: "param_early_stopping_rounds", sorter: "number"', js)
        self.assertIn('title: "Run time", field: "runtime_seconds", sorter: "number"', js)
        self.assertIn('title: "Sample", field: "sample_display", sorter: "string"', js)
        self.assertNotIn("<th>Test</th>", js)
        self.assertNotIn("<th>Scored</th>", js)
        self.assertNotIn("data-gbm-activate", js)
        self.assertNotIn('modelTable.on("rowClick"', js)
        self.assertIn("const commandSelection = event.metaKey || event.ctrlKey;", js)
        self.assertIn("if (event.shiftKey) {", js)
        self.assertIn("rows.forEach((candidate) => setSelected(candidate, candidate === row));", js)
        self.assertIn("function formatModelRuntime(model)", js)
        self.assertIn("function formatModelMetric(value)", js)
        self.assertIn("function modelParameterNumber(model, name)", js)
        self.assertIn('model?.timings?.training_seconds', js)
        self.assertIn("function formatModelCreated(value)", js)
        self.assertIn('return `${date.getDate()} ${months[date.getMonth()]} ${hour}:${minute}`;', js)
        self.assertIn("function modelWeightLabel(value)", js)
        self.assertIn('return !text || text === "__none__" || text === "Average row value" ? "N" : text;', js)
        self.assertIn("function formatSampleMode(value, source = \"\")", js)
        self.assertIn('el("gbmActivateModelBtn")?.addEventListener("click", activateSelectedModel);', js)
        self.assertIn('const activate = el("gbmActivateModelBtn");', js)
        self.assertIn("if (activate) activate.disabled = disableActions || selectedCount !== 1;", js)
        self.assertIn("async function activateSelectedModel()", js)
        self.assertIn("if (modelIds.length !== 1) return;", js)
        self.assertIn("await activateModel(modelIds[0]);", js)
        self.assertIn("function renameActiveModel()", js)
        self.assertIn("function deleteActiveModel()", js)
        self.assertIn("function selectedModelIds()", js)
        self.assertIn("const selectedCount = selectedModelIds().length;", js)
        self.assertIn("const [modelId] = selectedModelIds();", js)
        self.assertIn("const modelIds = selectedModelIds();", js)
        self.assertIn("for (const modelId of modelIds)", js)
        self.assertIn('row.classList.toggle("selected", selected);', js)
        self.assertIn('row.setAttribute("aria-selected", String(selected));', js)
        self.assertIn("anchorRow = row;", js)
        self.assertIn('method: "DELETE"', js)
        self.assertIn("button.setAttribute(\"aria-selected\", String(active));", js)
        self.assertIn("if (!active) activateModel(model.model_id);", js)
        self.assertIn("response_column: source.response_column", js)
        self.assertIn("offset_column: source.offset_column", js)
        self.assertIn("training_mode: source.training_mode", js)
        self.assertIn("function sourceColumns()", js)
        self.assertIn("function isModelPredictionColumn(column)", js)
        self.assertIn('return ["gbm_prediction", "glm_prediction"].includes(String(column?.name || ""));', js)
        self.assertIn("function expectedDisplayColumns()", js)
        self.assertIn("const predictionColumns = columns.filter(isModelPredictionColumn);", js)
        self.assertIn("return [...predictionColumns, ...otherColumns];", js)
        self.assertIn("for (const col of expectedDisplayColumns())", js)
        self.assertIn("function preferredStartupSource(availableSources, requestedSource)", js)
        self.assertIn('const activePredictionSource = availableSources.find((source) => source.kind === "gbm_predictions" && source.active);', js)
        self.assertIn('const predictionSource = availableSources.find((source) => source.kind === "gbm_predictions");', js)
        self.assertIn("state.source = preferredStartupSource(availableSources, requestedSource);", js)
        self.assertIn('source: state.source || "dataset"', js)
        self.assertIn('const previousExpected = el("expectedNumerator").value;', js)
        self.assertIn('fillMetricSelect(el("expectedNumerator"), true);', js)
        self.assertIn('el("expectedNumerator").value = numericColumnExists(previousExpected) ? previousExpected : "";', js)

    def test_gbm_model_navigator_incremental_refresh_contract(self) -> None:
        js = self.assert_no_store("/static/app/gbm-tool.js")[1].decode("utf-8")

        self.assertIn("const GBM_MODEL_LIST_POLL_MS = 2000;", js)
        self.assertIn("let modelListRefreshSeq = 0;", js)
        self.assertIn("let modelListLastRefreshAt = 0;", js)
        self.assertIn('if (nextTab === "models") refreshModelList({ force: true });', js)
        self.assertIn("async function refreshModelList({ force = false } = {})", js)
        self.assertIn('await api("/api/gbm/models", { method: "GET", clientTiming: true });', js)
        self.assertIn("async function applyModelListPayload(payload = {})", js)
        self.assertIn("cache.data = { ...cache.data, models, active_model_id: activeModelId };", js)
        self.assertIn("await refreshModelTableRows(modelRows(models));", js)
        self.assertIn("async function refreshModelTableRows(rows)", js)
        self.assertIn("const selectedIds = selectedModelIds();", js)
        self.assertIn("await modelTable.replaceData(rows);", js)
        self.assertIn("restoreModelSelection(preservedIds);", js)
        self.assertIn("function restoreModelSelection(ids)", js)
        self.assertIn('if (activeTab === "models") refreshModelList();', js)
        self.assertIn("const disableActions = isTraining;", js)
        self.assertIn("if (rename) rename.disabled = disableActions || selectedCount !== 1;", js)
        self.assertIn("if (activate) activate.disabled = disableActions || selectedCount !== 1;", js)
        self.assertIn("if (del) del.disabled = disableActions || selectedCount < 1;", js)
        self.assertIn("syncModelActionButtons();\n  }\n\n  function syncTrainingButton()", js)
        self.assertIn("async function activateModel(modelId) {\n    if (isTraining) return;", js)
        self.assertIn("async function activateSelectedModel() {\n    if (isTraining) return;", js)
        self.assertIn("async function renameActiveModel() {\n    if (isTraining) return;", js)
        self.assertIn("async function deleteActiveModel() {\n    if (isTraining) return;", js)

    def test_gbm_auto_training_label_uses_time_only(self) -> None:
        js = self.assert_no_store("/static/app/gbm-tool.js")[1].decode("utf-8")
        helper = self.js_function_source(js, "gbmAutoModelTimeLabel")

        self.assertIn('label: `GBM ${gbmAutoModelTimeLabel()}`', js)
        self.assertNotIn("toISOString().slice(11, 19)", js)
        self.assertNotIn('label: `GBM ${new Date().toISOString().slice(0, 19).replace("T", " ")}`', js)

        self.run_node_script(helper + """
const label = gbmAutoModelTimeLabel(new Date(2026, 6, 1, 18, 12, 59));
if (label !== "18:12:59") throw new Error(`expected local time label, got ${label}`);
""")

    def test_monitor_page_disables_cache(self) -> None:
        headers, body = self.assert_no_store("/monitor")
        _, css_body = self.assert_no_store("/static/monitor.css")
        _, js_body = self.assert_no_store("/static/monitor.js")
        html = body.decode("utf-8")
        css = css_body.decode("utf-8")
        js = js_body.decode("utf-8")

        self.assertEqual(headers.get("cache-control"), "no-store")
        self.assertIn("<title>lucidum monitor · sample.csv</title>", html)
        self.assertIn('window.matchMedia("(prefers-color-scheme: dark)").matches', html)
        self.assertIn('document.body.classList.add("dark");', html)
        self.assertIn('id="serverPanel" class="status-panel server-panel"', html)
        self.assertIn('id="activityPanel" class="status-panel"', html)
        self.assertIn('id="performancePanel" class="status-panel"', html)
        self.assertIn('id="lucidumServersPanel" class="panel lucidum-servers-panel"', html)
        self.assertIn('<h2 id="serverPanelTitle">Server</h2>', html)
        self.assertIn("<h2>Activity</h2>", html)
        self.assertIn("<h2>Performance</h2>", html)
        self.assertIn("<h2>lucidum servers</h2>", html)
        self.assertIn('id="lucidumServersMeta" class="meta"', html)
        self.assertIn('class="table-wrap lucidum-servers-table-wrap"', html)
        self.assertIn("<th>Listeners</th>", html)
        self.assertIn('<tbody id="lucidumServersBody"></tbody>', html)
        self.assertIn("<th>Browser</th>", html)
        self.assertNotIn("<th>User agent</th>", html)
        self.assertIn("<span>RSS</span>", html)
        self.assertIn("<span>USS</span>", html)
        self.assertIn("<span>Peak</span>", html)
        self.assertIn('id="serverRamMetric"', html)
        self.assertIn('id="serverUssMetric"', html)
        self.assertIn('id="peakRamMetric"', html)
        self.assertIn('id="processCpuMetric"', html)
        self.assertIn('id="pauseBtn" class="ghost header-icon-button" type="button" aria-label="Pause polling" title="Pause polling"', html)
        self.assertIn("pause-icon", html)
        self.assertIn("resume-icon", html)
        self.assertIn('id="themeBtn" class="ghost header-icon-button theme-toggle" type="button" aria-label="Switch to dark mode" title="Switch to dark mode"', html)
        self.assertIn("theme-icon-moon", html)
        self.assertIn("theme-icon-sun", html)
        self.assertIn('id="stopAppBtn" class="danger-action" type="button">Stop app</button>', html)
        self.assertNotIn('id="refreshBtn"', html)
        self.assertNotIn(">Refresh</button>", html)
        self.assertNotIn(">Pause</button>", html)
        self.assertIn('id="currentActionMetric"', html)
        self.assertIn('id="slowestActionMetric"', html)
        self.assertIn('id="errorRateMetric"', html)
        self.assertIn('id="heartbeatMeta" class="heartbeat-line"', html)
        self.assertIn('<strong id="totalRequestsMetric">--</strong>', html)
        self.assertIn('id="processMemoryMeta"', html)
        self.assertIn('class="performance-content"', html)
        self.assertIn('class="performance-meta-stack"', html)
        self.assertIn('class="slowest-action-row"', html)
        self.assertIn('id="clientsPanel" class="panel clients-panel"', html)
        self.assertIn('class="table-wrap clients-table-wrap"', html)
        self.assertIn('id="recentActivityPanel" class="panel recent-activity-panel"', html)
        self.assertIn('class="table-wrap recent-activity-table-wrap"', html)
        self.assertNotIn('Total requests <span id="totalRequestsMetric">', html)
        self.assertNotIn("<span>VMS</span>", html)
        self.assertIn("function heartbeatLabel(heartbeat)", js)
        self.assertIn('el("serverPanelTitle").textContent = process.pid ? `Server (PID ${process.pid})` : "Server";', js)
        self.assertIn("function renderLucidumServers(payload)", js)
        self.assertIn('el("lucidumServersMeta").textContent = `${formatNumber(payload?.count ?? servers.length)} running`;', js)
        self.assertIn('row.classList.toggle("current-server-row", Boolean(server.current));', js)
        self.assertIn("function serverHref(server)", js)
        self.assertIn('href.searchParams.set("token", token);', js)
        self.assertIn('link.className = "server-link";', js)
        self.assertIn('link.target = "_blank";', js)
        self.assertIn('link.rel = "noopener noreferrer";', js)
        self.assertIn('button.className = "server-stop-button";', js)
        self.assertIn('button.textContent = "X";', js)
        self.assertIn('async function loadLucidumServers()', js)
        self.assertIn('const response = await fetch("/api/lucidum-servers", {', js)
        self.assertIn("function renderSnapshot(snapshot, serversPayload)", js)
        self.assertIn("const [snapshot, serversPayload] = await Promise.all([loadTelemetry(), loadLucidumServers()]);", js)
        self.assertIn('await postJson("/api/lucidum-servers/stop", { pid: server.pid, create_time: server.create_time });', js)
        self.assertIn('el("lucidumServersBody").addEventListener("click"', js)
        self.assertIn("function formatGigabytesFromMegabytes(value)", js)
        self.assertIn("function syncPauseButton()", js)
        self.assertIn('button.classList.toggle("paused", state.paused);', js)
        self.assertIn("function syncThemeButton()", js)
        self.assertIn('document.body.classList.toggle("dark");', js)
        self.assertIn('let stoppedOverlayShown = false;', js)
        self.assertIn('let faviconDataUrl = "";', js)
        self.assertIn("async function cacheShutdownIcon()", js)
        self.assertIn('const response = await fetch("/favicon.ico", { cache: "force-cache" });', js)
        self.assertIn("reader.readAsDataURL(blob);", js)
        self.assertIn('function confirmStopApp(message = "Stop the local lucidum server?")', js)
        self.assertIn("Stop the local lucidum server?", js)
        self.assertIn('role="dialog" aria-modal="true" aria-labelledby="stopConfirmTitle"', js)
        self.assertIn("function stopApp()", js)
        self.assertIn('if (!(await confirmStopApp())) return;', js)
        self.assertIn('await postJson("/api/shutdown");', js)
        self.assertIn("function showStoppedOverlay()", js)
        self.assertIn('document.body.classList.add("app-stopped");', js)
        self.assertIn('`<img class="shutdown-icon" src="${faviconDataUrl}" alt="">`', js)
        self.assertIn('class="shutdown-icon shutdown-icon-fallback" aria-hidden="true"></span>', js)
        self.assertIn("<h1>lucidum has stopped</h1>", js)
        self.assertIn("cacheShutdownIcon();", js)
        self.assertIn('el("stopAppBtn").addEventListener("click", stopApp);', js)
        self.assertNotIn('el("refreshBtn")', js)
        self.assertNotIn('el("pauseBtn").textContent', js)
        self.assertNotIn('button.textContent = "Stopping...";', js)
        self.assertIn("const heartbeat = snapshot.heartbeat || {};", js)
        self.assertIn("heartbeatNode.textContent = heartbeatLabel(heartbeat);", js)
        self.assertIn('textCell(row, client.user_agent_label || client.user_agent, "client-browser").title = client.user_agent || "";', js)
        self.assertIn("`Total RAM ${formatGigabytesFromMegabytes(systemMemory.total_mb)}`", js)
        self.assertIn('el("processMemoryMeta").textContent = processDetails.length ? processDetails.join(" · ") : "Process RAM --";', js)
        self.assertNotIn("VMS ${formatMegabytes(process.vms_mb)}", js)
        self.assertIn("return `${Math.round(number).toLocaleString()}ms`;", js)
        self.assertNotIn("return `${(number / 1000).toFixed(1)}s`;", js)
        self.assertIn("display: flex;\n  flex-direction: column;", css)
        self.assertIn("body.dark {\n  color-scheme: dark;", css)
        self.assertIn("header {\n  height: 52px;", css)
        self.assertIn("padding: 0 14px;", css)
        self.assertIn(".header-actions {\n  margin-left: auto;\n  display: flex;\n  align-items: center;\n  gap: 14px;", css)
        self.assertIn(".header-icon-button {\n  width: 28px;\n  height: 24px;\n  min-height: 24px;", css)
        self.assertIn(".header-action-icon {\n  width: 17px;", css)
        self.assertIn(".resume-icon,\n.theme-icon-sun {\n  display: none;", css)
        self.assertIn("#pauseBtn.paused .pause-icon,\nbody.dark .theme-icon-moon {\n  display: none;", css)
        self.assertIn("#pauseBtn.paused .resume-icon,\nbody.dark .theme-icon-sun {\n  display: block;", css)
        self.assertIn(
            ".danger-action {\n"
            "  min-height: 24px;\n"
            "  border: 1px solid var(--danger);\n"
            "  border-radius: 5px;\n"
            "  background: var(--danger);\n"
            "  color: white;\n"
            "  padding: 2px 8px;\n"
            "  cursor: pointer;\n"
            "  font-size: 12px;\n"
            "  font-weight: 650;\n"
            "}",
            css,
        )
        self.assertIn("body.dark .danger-action {\n  color: #111827;\n}", css)
        self.assertNotIn(".danger-action:hover", css)
        self.assertNotIn(".danger-action:disabled", css)
        self.assertNotIn("background: #ef4444;", css)
        self.assertNotIn("border-color: #fca5a5;", css)
        self.assertIn(".stop-confirm-overlay {\n  position: fixed;", css)
        self.assertIn(".stop-confirm-content {\n  display: grid;\n  grid-template-columns: 38px minmax(0, 1fr);", css)
        self.assertIn(".stop-confirm-icon {\n  width: 38px;\n  height: 38px;", css)
        self.assertIn(".stop-confirm-actions {\n  display: flex;\n  justify-content: flex-end;", css)
        self.assertIn(".shutdown-overlay {\n  position: fixed;", css)
        self.assertIn(".shutdown-message {\n  width: min(420px, 100%);", css)
        self.assertIn("grid-template-columns: 42px minmax(0, 1fr);", css)
        self.assertIn(".shutdown-icon {\n  width: 42px;\n  height: 42px;", css)
        self.assertIn(".shutdown-icon-fallback {\n  border: 1px solid var(--line);", css)
        self.assertIn("min-height: 132px;", css)
        self.assertIn("grid-template-rows: repeat(4, auto);", css)
        self.assertIn(".server-panel {\n  grid-template-rows: auto auto auto minmax(0, 1fr) auto;", css)
        self.assertIn(".server-panel .diagnostic-line {\n  grid-row: 5;\n  align-self: end;", css)
        self.assertIn(".performance-content {\n  min-width: 0;\n  display: grid;", css)
        self.assertIn("grid-template-columns: minmax(165px, 1fr) minmax(0, max-content);", css)
        self.assertIn(".status:empty {\n  display: none;", css)
        self.assertIn(".performance-content .action-stack strong {\n  overflow: hidden;\n  text-overflow: ellipsis;\n  white-space: nowrap;", css)
        self.assertIn(".slowest-action-row strong {\n  overflow: visible;\n  text-overflow: clip;", css)
        self.assertIn(".performance-meta-stack {\n  min-width: 0;\n  display: grid;\n  justify-items: end;", css)
        self.assertNotIn("grid-template-rows: auto minmax(0, 1fr) auto;", css)
        self.assertIn(".primary-metric strong {\n  font-size: 32px;", css)
        self.assertIn("max-height: min(320px, 40vh);", css)
        self.assertIn(".clients-table-wrap {\n  max-height: min(170px, 24vh);", css)
        self.assertIn(".lucidum-servers-panel {\n  flex: 0 0 auto;", css)
        self.assertIn(".lucidum-servers-table-wrap {\n  max-height: min(135px, 20vh);", css)
        self.assertIn(".current-server-row td {\n  background: color-mix(in srgb, var(--accent) 10%, transparent);", css)
        self.assertIn(".server-link {\n  color: var(--accent);", css)
        self.assertIn(".server-stop-button {\n  width: 16px;", css)
        self.assertIn(".clients-panel th,\n.clients-panel td {\n  padding: 4px 9px;", css)
        self.assertIn(".pill {\n  display: inline-flex;\n  align-items: center;\n  min-height: 16px;", css)
        self.assertIn(".recent-activity-panel {\n  flex: 1 1 auto;\n  min-height: 0;", css)
        self.assertIn(".recent-activity-table-wrap {\n  flex: 1 1 auto;\n  min-height: 0;\n  max-height: none;", css)
        self.assertIn(".client-browser {\n  white-space: nowrap;", css)
        self.assertIn(".recent-activity-panel th,\n.recent-activity-panel td {\n  padding: 4px 8px;", css)
        self.assertNotIn("grid-template-rows: auto auto minmax(0, 1fr) minmax(0, 1fr);", css)
        self.assertIn('href="/static/monitor.css"', html)
        self.assertIn('src="/static/monitor.js"', html)

    def test_line_bar_warnings_render_inside_chart_messages(self) -> None:
        _, html_body = self.assert_no_store("/")
        css = self.app_css_contract()
        html = html_body.decode("utf-8")
        js = self.app_js_contract()

        self.assertLess(html.index('id="lineBarFilter"'), html.index('id="chartMessage"'))
        self.assertLess(html.index('id="chartMessage"'), html.index('id="chart" class="hidden"'))
        self.assertIn('const warnings = [...(data.warnings || [])].filter(Boolean).join(" ");', js)
        self.assertIn('const chartMessage = [warnings, labelMessage].filter(Boolean).join(" ");', js)
        self.assertIn('const displayMessage = message || "";', js)
        self.assertIn('saveToolPresentation("line_bar", { groupMeta, chartMessage });', js)
        self.assertNotIn("setStatus(status);", js)
        self.assertNotIn('replace(/\\.$/, "")', js)
        self.assertNotIn('saveToolPresentation("line_bar", { groupMeta, status, chartMessage: labelMessage });', js)
        self.assertIn("#visualArea:not(.profile-mode):not(.map-mode) .workspace-messages {\n        max-width: min(860px, calc(100% - 150px));", css)
        self.assertIn(".workspace-meta,\n      .chart-message {\n        color: var(--muted);\n        font-size: 10px;", css)

    def test_line_bar_table_summary_row_is_client_computed(self) -> None:
        js = self.app_js_contract()
        css = self.app_css_contract()
        script = f"""
const state = {{ transform: "none" }};
{self.js_function_source(js, "tableNumber")}
{self.js_function_source(js, "transformTableSummaryValue")}
{self.js_function_source(js, "buildTableSummary")}
const data = {{
  rows: [
    {{ volume: 2, resp0_num: 700, resp0_den: 2, resp1_num: 700, resp1_den: 2 }},
    {{ volume: 1, resp0_num: 200, resp0_den: 1, resp1_num: 210, resp1_den: 1 }},
  ],
  responses: [{{}}, {{}}],
}};
let summary = buildTableSummary(data);
if (summary.volume !== 3) throw new Error("volume " + summary.volume);
if (summary.responses[0] !== 300) throw new Error("actual summary " + summary.responses[0]);
if (Math.abs(summary.responses[1] - 910 / 3) > 1e-12) throw new Error("expected summary " + summary.responses[1]);
state.transform = "log";
summary = buildTableSummary(data);
if (Math.abs(summary.responses[0] - Math.log(300)) > 1e-12) throw new Error("log summary " + summary.responses[0]);
state.transform = "zero";
summary = buildTableSummary(data);
if (summary.responses[0] !== 0) throw new Error("zero summary " + summary.responses[0]);
summary = buildTableSummary({{ rows: [{{ volume: 0, resp0_num: null, resp0_den: 0 }}], responses: [{{}}] }});
if (summary.responses[0] !== null) throw new Error("empty denominator summary " + summary.responses[0]);
"""
        self.run_node_script(script)

        self.assertIn("function buildTableSummary(data)", js)
        self.assertIn("const rowNumerator = tableNumber(row[`resp${index}_num`]);", js)
        self.assertIn("const rowDenominator = tableNumber(row[`resp${index}_den`]);", js)
        self.assertIn('const footer = `<tfoot><tr class="line-bar-summary-row"><td>Total</td><td>${formatNumber(summary.volume)}</td>${summaryValues}</tr></tfoot>`;', js)
        self.assertIn("#tableWrap .line-bar-summary-row td {\n        background: var(--panel);\n        border-top: 2px solid var(--line);\n        font-weight: 700;", css)

    def test_stop_app_confirmation_uses_custom_favicon_dialog(self) -> None:
        css = self.app_css_contract()
        js = self.app_js_contract()

        self.assertIn("function confirmStopApp()", js)
        self.assertIn('overlay.className = "stop-confirm-overlay";', js)
        self.assertIn('class="stop-confirm-icon" src="/favicon.ico" alt=""', js)
        self.assertIn("Stop the local lucidum server?", js)
        self.assertNotIn("Stop the local py_lucidum server?", js)
        self.assertIn('role="dialog" aria-modal="true" aria-labelledby="stopConfirmTitle"', js)
        self.assertIn('if (!(await confirmStopApp())) return;', js)
        self.assertNotIn("window.confirm", js)
        self.assertIn(".stop-confirm-content {\n        display: grid;\n        grid-template-columns: 38px minmax(0, 1fr);", css)
        self.assertIn(".stop-confirm-icon {\n        width: 38px;\n        height: 38px;", css)
        self.assertIn(".stop-confirm-actions {\n        display: flex;\n        justify-content: flex-end;", css)

    def test_stopped_overlay_uses_cached_icon_message_layout(self) -> None:
        css = self.app_css_contract()
        js = self.app_js_contract()

        self.assertIn('let faviconDataUrl = "";', js)
        self.assertIn("async function cacheShutdownIcon()", js)
        self.assertIn('const response = await fetch("/favicon.ico", { cache: "force-cache" });', js)
        self.assertIn("reader.readAsDataURL(blob);", js)
        self.assertIn("cacheShutdownIcon();", js)
        self.assertIn('`<img class="shutdown-icon" src="${faviconDataUrl}" alt="">`', js)
        self.assertIn('class="shutdown-icon shutdown-icon-fallback" aria-hidden="true"></span>', js)
        self.assertNotIn('class="shutdown-icon" src="/favicon.ico" alt=""', js)
        self.assertNotIn('class="shutdown-icon" aria-hidden="true">L</span>', js)
        self.assertIn("<h1>lucidum has stopped</h1>", js)
        self.assertNotIn("<h1>py_lucidum has stopped</h1>", js)
        self.assertIn(".shutdown-message {\n        width: min(420px, 100%);", css)
        self.assertIn("display: grid;\n        grid-template-columns: 42px minmax(0, 1fr);", css)
        self.assertIn("text-align: left;", css)
        self.assertIn(".shutdown-icon {\n        width: 42px;\n        height: 42px;", css)
        self.assertIn("object-fit: contain;", css)
        self.assertIn(".shutdown-icon-fallback {\n        border: 1px solid var(--line);", css)
        self.assertIn("display: grid;\n        place-items: center;", css)

    def test_boot_schema_failure_updates_header_and_status(self) -> None:
        js = self.app_js_contract()

        self.assertIn('state.schema = await api("/api/schema");', js)
        self.assertIn('el("datasetMeta").textContent = "Dataset failed to load";', js)
        self.assertIn("setStatus(error.message, true);", js)
        self.assertIn(
            'el("datasetMeta").textContent = "Dataset failed to load";\n          setStatus(error.message, true);',
            js,
        )

    def test_dataset_meta_file_size_uses_one_decimal_place(self) -> None:
        js = self.app_js_contract()

        self.assertIn("function formatFileSize(value)", js)
        self.assertIn("const size = bytes / divisor;", js)
        self.assertIn('return `${(bytes > 0 ? Math.max(0.1, size) : 0).toFixed(1)}${suffix}`;', js)
        self.assertNotIn("Math.round(bytes / divisor)", js)

    def test_feature_picker_rows_are_compact(self) -> None:
        css = self.app_css_contract()

        self.assertIn("min-height: 20px;", css)
        self.assertIn("padding: 1px 6px;", css)
        self.assertIn("font-size: 11px;", css)
        self.assertIn("font-size: 9px;", css)

    def test_saved_filter_select_uses_feature_list_row_spacing(self) -> None:
        _, html_body = self.assert_no_store("/")
        css = self.app_css_contract()
        html = html_body.decode("utf-8")
        js = self.app_js_contract()

        self.assertIn('id="savedFilterSelect" class="feature-list saved-filter-list" role="listbox"', html)
        self.assertIn('aria-multiselectable="true"', html)
        self.assertNotIn("<select id=\"savedFilterSelect\"", html)
        self.assertIn("#savedFilterSelect {\n        flex: 1 1 auto;\n        width: 100%;\n        height: auto;\n        min-height: 64px;\n        margin-bottom: 7px;", css)
        self.assertIn(".saved-filter-list .feature {\n        display: grid;\n        grid-template-columns: fit-content(52%) minmax(96px, 1fr);", css)
        self.assertIn("body:not(.saved-filter-single-mode) .saved-filter-list .saved-filter-option:not(.active):hover {\n        background: transparent;", css)
        self.assertIn("body.saved-filter-single-mode .saved-filter-list .saved-filter-option:not(.active):hover {\n        background: color-mix(in srgb, var(--accent) 15%, transparent);", css)
        self.assertIn(".saved-filter-list .saved-filter-option[hidden] {\n        display: none !important;", css)
        self.assertIn(".saved-filter-theme {\n        -webkit-appearance: none;\n        appearance: none;\n        width: 100%;\n        border-bottom: 1px solid var(--line);", css)
        self.assertIn("--saved-filter-theme-bg: #c8daec;", css)
        self.assertIn("--saved-filter-theme-hover-bg: #b9d0e7;", css)
        self.assertIn("--saved-filter-theme-text: #58677c;", css)
        self.assertIn("--saved-filter-theme-bg: #30415c;", css)
        self.assertIn("--saved-filter-theme-hover-bg: #38506f;", css)
        self.assertIn("--saved-filter-theme-text: #c1ccdc;", css)
        self.assertIn("background: var(--saved-filter-theme-bg);", css)
        self.assertIn("color: var(--saved-filter-theme-text);", css)
        self.assertIn(".saved-filter-theme:hover {\n        background: var(--saved-filter-theme-hover-bg);", css)
        self.assertIn("cursor: pointer;", css)
        self.assertIn(".saved-filter-theme-icon {\n        flex: 0 0 auto;\n        width: 7px;\n        height: 7px;\n        border-right: 1.5px solid currentColor;\n        border-bottom: 1.5px solid currentColor;", css)
        self.assertIn('.saved-filter-theme[aria-expanded="false"] .saved-filter-theme-icon {\n        transform: rotate(-45deg) translate(-1px, 1px);', css)
        self.assertIn(".saved-filter-theme-label {\n        min-width: 0;\n        overflow: hidden;\n        text-overflow: ellipsis;\n        white-space: nowrap;", css)
        self.assertIn("text-transform: uppercase;", css)
        self.assertIn(".saved-filter-expression {\n        min-width: 0;\n        overflow: hidden;\n        text-align: right;\n        text-overflow: ellipsis;\n        white-space: nowrap;\n        color: var(--muted);\n        font-size: 10px;", css)
        self.assertIn("collapsedSavedFilterThemes: new Set()", js)
        self.assertIn("savedFilterThemesInitialised: false", js)
        self.assertIn("function savedFilterSpecSignature(filters = state.schema?.filters || [])", js)
        self.assertIn("function savedFilterSelectionSnapshot()", js)
        self.assertIn("function restoreSavedFilterSelection(selectedKeys)", js)
        self.assertIn('const theme = filter.theme || "General";', js)
        self.assertIn("availableThemes.forEach((theme) => state.collapsedSavedFilterThemes.add(theme));", js)
        self.assertIn("state.savedFilterThemesInitialised = true;", js)
        self.assertIn("state.savedFilterThemesInitialised = false;", js)
        self.assertIn('const collapsed = state.collapsedSavedFilterThemes.has(theme);', js)
        self.assertIn('heading.className = "saved-filter-theme";', js)
        self.assertIn("heading.dataset.filterTheme = theme;", js)
        self.assertIn('heading.setAttribute("aria-expanded", String(!collapsed));', js)
        self.assertIn('heading.innerHTML = `<span class="saved-filter-theme-icon" aria-hidden="true"></span><span class="saved-filter-theme-label">${escapeHtml(theme)}</span>`;', js)
        self.assertIn("heading.addEventListener(\"click\", () => toggleSavedFilterTheme(theme));", js)
        self.assertIn('button.className = "feature saved-filter-option";', js)
        self.assertIn("button.dataset.filterTheme = theme;", js)
        self.assertIn('button.dataset.filterName = filter.name || "";', js)
        self.assertIn("button.hidden = state.collapsedSavedFilterThemes.has(theme);", js)
        self.assertIn('button.setAttribute("role", "option");', js)
        self.assertIn('button.setAttribute("aria-selected", "false");', js)
        self.assertIn('button.innerHTML = `<span class="saved-filter-name">${escapeHtml(filter.name)}</span><span class="saved-filter-expression">${escapeHtml(filter.expression)}</span>`;', js)
        self.assertIn('if (state.filterSelectionMode === "single") {', js)
        self.assertIn("const isClickedOption = option === button;", js)
        self.assertIn('option.classList.toggle("active", isClickedOption);', js)
        self.assertIn("function toggleSavedFilterTheme(theme)", js)
        self.assertIn("state.collapsedSavedFilterThemes.add(theme);", js)
        self.assertIn("state.collapsedSavedFilterThemes.delete(theme);", js)
        self.assertIn('list.querySelectorAll(".saved-filter-theme").forEach((heading) => {', js)
        self.assertIn("if (button.dataset.filterTheme === theme) button.hidden = collapsed;", js)
        self.assertIn('button.addEventListener("click", () => {', js)
        self.assertIn('button.setAttribute("aria-selected", String(!selected));', js)
        self.assertIn('button.classList.toggle("active", !selected);', js)
        self.assertIn("function setFilterSelectionMode(mode, options = {})", js)
        self.assertIn('document.body.classList.toggle("saved-filter-single-mode", nextMode === "single");', js)
        self.assertIn('const group = document.querySelector(\'.segmented[data-control="filterSelectionMode"]\');', js)
        self.assertIn("const keep = selected[0];", js)
        self.assertIn('querySelectorAll(\'.saved-filter-option[aria-selected="true"]\')', js)
        self.assertIn("function selectedSavedFilterRows()", js)
        self.assertIn('theme: button.dataset.filterTheme || "General"', js)
        self.assertIn('name: button.dataset.filterName || ""', js)
        self.assertIn("const previousFilterSignature = savedFilterSpecSignature();", js)
        self.assertIn("const previousSidebarVisible = state.sidebarVisible;", js)
        self.assertIn("const filtersUnchanged = previousFilterSignature === savedFilterSpecSignature(state.schema.filters || []);", js)
        self.assertIn('if (state.tool === "uk_map") ukMapTool.captureView("reload");', js)
        self.assertIn("if (filtersUnchanged) restoreSavedFilterSelection(previousSavedFilterSelection);", js)
        self.assertIn("setSidebarVisible(previousSidebarVisible);", js)
        self.assertIn("function combinedGroupedSavedFilterExpression(rows)", js)
        self.assertIn("if (rows.length === 1) return rows[0].expression;", js)
        self.assertIn("const byTheme = new Map();", js)
        self.assertIn("byTheme.get(row.theme).push(row.expression);", js)
        self.assertIn('const groupedExpressions = expressions.map(wrapFilterExpression).join(" OR ");', js)
        self.assertIn("return expressions.length > 1 ? `(${groupedExpressions})` : groupedExpressions;", js)
        self.assertIn("const combined = groupedExpressions.join(` ${operator} `);", js)

    def test_filter_panel_collapses_and_shows_row_meta(self) -> None:
        _, html_body = self.assert_no_store("/")
        css = self.app_css_contract()
        html = html_body.decode("utf-8")
        js = self.app_js_contract()

        self.assertIn('<section class="section sidebar-filter-section filter-collapsed">', html)
        self.assertIn('aria-label="Resize KPI and filter controls"', html)
        self.assertIn('id="filterCollapseBtn"', html)
        self.assertIn('aria-label="Expand filter"', html)
        self.assertIn('aria-expanded="false"', html)
        self.assertIn("<h2>FILTER</h2>", html)
        self.assertLess(html.index("<h2>FILTER</h2>"), html.index('id="filterRowMeta"'))
        self.assertLess(html.index('id="filterRowMeta"'), html.index('id="filterSidebarClearBtn"'))
        self.assertIn('id="filterRowMeta" class="filter-row-meta"', html)
        self.assertIn(".sidebar-filter-section.filter-collapsed #sidebarFilterResizer,", css)
        self.assertIn(".sidebar-filter-section.filter-collapsed .filter-controls-row,", css)
        self.assertIn(".sidebar-filter-section.filter-collapsed #savedFilterSelect {\n        display: none;", css)
        self.assertIn("display: none;", css)
        self.assertIn(".filter-collapse-icon {\n        width: 9px;", css)
        self.assertIn("border-right: 1.75px solid currentColor;", css)
        self.assertIn(".sidebar-filter-section.filter-collapsed .filter-collapse-icon", css)
        self.assertIn(".filter-header h2 {\n        margin: 0;\n        font-size: 12px;", css)
        self.assertIn(".filter-row-meta {\n        min-width: 0;\n        overflow: hidden;\n        text-align: right;\n        text-overflow: ellipsis;\n        white-space: nowrap;\n        color: var(--muted);\n        font-size: 10px;\n        margin-left: auto;", css)
        self.assertIn(".filter-sidebar-clear {\n        width: 24px;\n        min-height: 24px;\n        font-size: 14px;", css)
        self.assertIn(".sidebar-filter-section.filter-collapsed .filter-sidebar-clear {\n        width: 20px;\n        min-height: 20px;\n        font-size: 13px;", css)
        self.assertIn("filterCollapsed: true", js)
        self.assertIn("function formatRowMeta(rowCount, filteredRowCount = rowCount)", js)
        self.assertIn("? `${total.toLocaleString()} rows`", js)
        self.assertIn(": `${shown.toLocaleString()} / ${total.toLocaleString()} rows`", js)
        self.assertIn("function setFilterCollapsed(collapsed)", js)
        self.assertIn('document.querySelector(".sidebar-filter-section")?.classList.toggle("filter-collapsed", state.filterCollapsed);', js)
        self.assertIn('button.setAttribute("aria-expanded", String(!state.filterCollapsed));', js)
        self.assertIn('el("filterCollapseBtn").addEventListener("click", () => setFilterCollapsed(!state.filterCollapsed));', js)
        self.assertIn("setFilterRowMeta(state.schema.row_count);", js)
        self.assertIn("setFilterRowMeta(data.row_count, data.filtered_row_count);", js)

    def test_kpi_sidebar_section_contains_metric_selects_and_grouped_rows(self) -> None:
        _, html_body = self.assert_no_store("/")
        css = self.app_css_contract()
        html = html_body.decode("utf-8")
        js = self.app_js_contract()

        self.assertIn('<section class="section sidebar-kpi-section hidden">', html)
        self.assertLess(html.index("<h2>KPIs</h2>"), html.index("<h2>FILTER</h2>"))
        self.assertLess(html.index('id="actualNumerator"'), html.index('id="kpiSelect"'))
        self.assertLess(html.index('id="denominator"'), html.index('id="kpiSelect"'))
        self.assertIn('id="kpiCollapseBtn"', html)
        self.assertIn('id="kpiSelectedMeta" class="kpi-selected-meta"', html)
        self.assertIn('id="kpiSelect" class="feature-list kpi-list" role="listbox"', html)
        self.assertNotIn('id="sidebarKpiResizer"', html)
        self.assertIn(".kpi-header h2 {\n        margin: 0;\n        font-size: 12px;", css)
        self.assertIn(".kpi-selected-meta,\n      .gbm-model-selected-meta {\n        min-width: 0;\n        overflow: hidden;\n        text-align: right;", css)
        self.assertIn("color: var(--muted);\n        font-size: 10px;\n        margin-left: auto;", css)
        self.assertIn(".sidebar-kpi-section.kpi-collapsed .kpi-controls,", css)
        self.assertIn(".sidebar-kpi-section.kpi-collapsed #kpiSelect {\n        display: none;", css)
        self.assertIn(".metric-title {\n        display: flex;\n        align-items: baseline;\n        justify-content: space-between;", css)
        self.assertIn("font-weight: 700;\n        font-size: 12px;", css)
        self.assertNotIn(".chart-side-section h2.metric-title", css)
        self.assertIn(".metric-value {\n        color: var(--muted);\n        margin-left: auto;\n        text-align: right;", css)
        self.assertIn("font-size: 10px;\n        font-weight: 400;", css)
        self.assertIn(".sidebar-kpi-section {\n        display: flex;\n        flex-direction: column;\n        flex: 0 0 var(--sidebar-kpi-height, 260px);\n        margin-bottom: 0;\n        min-height: 0;\n        overflow: hidden;", css)
        self.assertIn(".sidebar-kpi-section.kpi-collapsed {\n        flex: 0 0 auto;", css)
        self.assertIn("#kpiSelect {\n        flex: 1 1 auto;\n        width: 100%;\n        height: auto;\n        min-height: 0;", css)
        self.assertIn("max-height: none;", css)
        self.assertNotIn(".sidebar-kpi-resizer", css)
        self.assertIn(".sidebar-filter-section {\n        display: flex;\n        flex-direction: column;\n        flex: 0 0 var(--sidebar-filter-height, 238px);\n        height: auto;\n        margin-top: 0;\n        margin-bottom: 0;\n        min-height: 0;\n        overflow: hidden;", css)
        self.assertIn(".sidebar-kpi-section.hidden ~ .sidebar-filter-section,\n      .sidebar-kpi-section.kpi-collapsed ~ .sidebar-filter-section {\n        margin-top: 0;", css)
        self.assertIn(".kpi-list .feature,\n      .gbm-model-list .feature {\n        display: grid;\n        grid-template-columns: fit-content(52%) minmax(96px, 1fr);", css)
        self.assertIn(".kpi-list .kpi-option.active {\n        background: color-mix(in srgb, #f59e0b 22%, var(--panel));", css)
        self.assertIn(".gbm-sidebar-panel {\n        display: flex;\n        flex-direction: column;\n        flex: 0 0 var(--sidebar-gbm-height, 220px);\n        height: auto;\n        margin-top: 0;\n        margin-bottom: 0;\n        min-height: 0;\n        overflow: hidden;", css)
        self.assertIn("position: relative;", css)
        self.assertIn(".gbm-sidebar-panel.gbm-model-collapsed #gbmModelSelect {\n        display: none;", css)
        self.assertIn(".gbm-sidebar-panel.gbm-model-collapsed #sidebarGbmResizer {\n        display: none;", css)
        self.assertIn(".sidebar-gbm-resizer {\n        background: linear-gradient(to bottom, var(--line), transparent);", css)
        self.assertIn("flex: 0 0 8px;", css)
        self.assertNotIn("top: -8px;", css)
        self.assertIn(".gbm-model-header {\n        display: flex;\n        align-items: center;", css)
        self.assertIn(".gbm-model-selected-meta {\n        min-width: 0;\n        overflow: hidden;", css)
        self.assertIn("#gbmModelSelect {\n        flex: 1 1 auto;\n        width: 100%;", css)
        self.assertIn(".gbm-model-list .feature {\n        display: grid;\n        grid-template-columns: fit-content(52%) minmax(96px, 1fr);", css)
        self.assertIn(".gbm-model-list .gbm-model-option.active {\n        background: color-mix(in srgb, var(--accent) 20%, var(--panel));", css)
        self.assertIn(".gbm-model-list .gbm-model-option {\n        grid-template-columns: fit-content(72%) minmax(0, 1fr);", css)
        self.assertIn(".kpi-detail,\n      .gbm-model-detail {\n        min-width: 0;\n        overflow: hidden;\n        text-align: right;", css)
        self.assertIn(".gbm-model-detail {\n        justify-self: stretch;\n        text-align: right;", css)
        self.assertIn("kpiCollapsed: false", js)
        self.assertIn("collapsedKpiGroups: new Set()", js)
        self.assertIn("gbmModelCollapsed: false", js)
        self.assertIn("collapsedGbmModelGroups: new Set()", js)
        self.assertIn("gbmModelGroupsInitialised: false", js)
        self.assertIn("activeKpiFormat: null", js)
        self.assertIn("function availableKpis()", js)
        self.assertIn("function hasRequestedDefault(name)", js)
        self.assertIn("function applyInitialKpiDefault()", js)
        self.assertIn('if (hasRequestedDefault("actual") || hasRequestedDefault("denominator")) return;', js)
        self.assertIn("const firstKpi = availableKpis()[0];", js)
        self.assertIn('el("actualNumerator").value = firstKpi.actual;', js)
        self.assertIn('el("denominator").value = firstKpi.denominator;', js)
        self.assertIn('const displayNumber = activeKpiFormat.format === "percent" ? number * 100 : number;', js)
        self.assertIn('const sign = displayNumber < 0 ? "-" : "";', js)
        self.assertIn('el("kpiSelectedMeta").textContent = kpi ? kpi.name : "";', js)
        self.assertIn('if (denominator === "__none__") return "N";', js)
        self.assertIn('heading.className = "saved-filter-theme kpi-theme";', js)
        self.assertIn('button.className = `feature kpi-option${active ? " active" : ""}`;', js)
        self.assertIn("function selectKpi(kpi)", js)
        self.assertIn('storageKey: "py_lucidum_sidebar_kpi_height"', js)
        self.assertIn("function clampSidebarFilterHeight()", js)
        self.assertIn("function setGbmModelCollapsed(collapsed)", js)
        self.assertIn("function syncGbmModelCollapseButton()", js)
        self.assertIn("function setupSidebarGbmResize()", js)
        self.assertIn("function setSidebarGbmHeight(rawHeight)", js)
        self.assertIn("function clampSidebarGbmHeight()", js)
        self.assertIn("function setSidebarPanelHeight(key, rawHeight, options = {})", js)
        self.assertIn("function restoreSidebarPanelHeights()", js)
        self.assertIn("function resizeSidebarBoundary(key, delta, startLayout)", js)
        self.assertIn("function sidebarResizableHeightCapacity()", js)
        self.assertIn("requestAnimationFrame(clampSidebarPanelHeights);", js)
        self.assertIn('el("kpiCollapseBtn").addEventListener("click", () => setKpiCollapsed(!state.kpiCollapsed));', js)
        self.assertIn('el("gbmModelCollapseBtn").addEventListener("click", () => setGbmModelCollapsed(!state.gbmModelCollapsed));', js)
        self.assertIn("syncKpiSelectionFromMetrics();", js)

    def test_filter_footer_and_sidebar_filter_controls_contract(self) -> None:
        _, html_body = self.assert_no_store("/")
        css = self.app_css_contract()
        html = html_body.decode("utf-8")
        js = self.app_js_contract()

        self.assertLess(html.index('id="sidebarToggleBtn"'), html.index('id="filterFooterToggleBtn"'))
        self.assertIn('<body class="filter-footer-collapsed">', html)
        self.assertIn('<div class="layout-toggle-group">', html)
        self.assertIn('id="filterFooterToggleBtn" class="footer-toggle" type="button" aria-label="Show filter footer" aria-controls="filterFooter" aria-expanded="false" title="Show filter footer"', html)
        self.assertIn('aria-controls="filterFooter"', html)
        self.assertIn('class="footer-toggle-icon"', html)
        self.assertIn('<footer id="filterFooter" class="filter-footer" aria-hidden="true">', html)
        self.assertIn('<div class="footer-filter-controls">', html)
        self.assertLess(html.index('id="filterClearBtn"'), html.index('id="filterApplyBtn"'))
        self.assertLess(html.index('id="filterApplyBtn"'), html.index('id="filterInput"'))
        self.assertGreater(html.index('id="filterInput"'), html.index('id="filterFooter"'))
        self.assertLess(html.index('id="filterInput"'), html.index('id="actionTimingMonitor"'))
        self.assertIn('id="actionTimingMonitor" class="action-timing-monitor" aria-live="polite"', html)
        self.assertIn("DuckDB: --, JSON: --, Profile render: --, Total: --", html)
        self.assertIn('<div class="filter-controls-row">', html)
        self.assertLess(html.index('data-control="filterSelectionMode"'), html.index('data-control="filterOperator"'))
        self.assertLess(html.index('data-control="filterSelectionMode"'), html.index('id="savedFilterSelect"'))
        self.assertLess(html.index('data-control="filterOperator"'), html.index('id="savedFilterSelect"'))
        self.assertIn('class="segmented filter-operator" data-control="filterOperator"', html)
        self.assertIn('class="segmented filter-selection-mode" data-control="filterSelectionMode"', html)
        self.assertIn('<button data-value="and" class="active" type="button">All</button>', html)
        self.assertIn('<button data-value="or" type="button">Any</button>', html)
        self.assertIn('<button data-value="nand" type="button">Not all</button>', html)
        self.assertIn('<button data-value="nor" type="button">None</button>', html)
        self.assertIn('<button data-value="single" class="active" type="button">Single</button>', html)
        self.assertIn('<button data-value="grouped" type="button">Grouped</button>', html)
        self.assertIn('<button data-value="multi" type="button">Multi</button>', html)
        self.assertLess(html.index('data-value="single"'), html.index('data-value="grouped"'))
        self.assertLess(html.index('data-value="grouped"'), html.index('data-value="multi"'))
        self.assertIn('id="filterSidebarClearBtn" class="filter-action filter-sidebar-clear" type="button" title="Clear filter" aria-label="Clear filter"', html)
        self.assertNotIn("filter-input-row", html)
        self.assertIn("grid-template-rows: auto minmax(0, 1fr) auto;", css)
        self.assertIn(".layout-toggle-group {\n        display: flex;\n        align-items: center;\n        gap: 0;", css)
        self.assertIn(".footer-toggle {\n        width: 28px;\n        height: 24px;\n        min-height: 24px;", css)
        self.assertIn(".footer-toggle-icon {\n        width: 18px;\n        height: 20px;\n        border: 1.5px solid currentColor;", css)
        self.assertIn("bottom: 3.5px;\n        height: 1.5px;", css)
        self.assertIn("body.filter-footer-collapsed .footer-toggle-icon::before {\n        background: transparent;", css)
        self.assertIn(".filter-footer {\n        display: grid;\n        grid-template-columns: minmax(0, 1fr) minmax(0, 1fr);", css)
        self.assertIn(".footer-filter-controls {\n        display: grid;\n        grid-template-columns: 28px 28px minmax(0, 1fr);", css)
        self.assertIn(".action-timing-monitor {\n        min-width: 0;\n        overflow: hidden;\n        color: var(--muted);\n        font-size: 10px;", css)
        self.assertIn("text-align: right;", css)
        self.assertIn("body.filter-footer-collapsed .filter-footer {\n        display: none;", css)
        self.assertIn(".filter-controls-row {\n        display: grid;\n        grid-template-columns: auto minmax(0, 1fr);", css)
        self.assertIn("body.saved-filter-single-mode .filter-operator,\n      body.saved-filter-grouped-mode .filter-operator {\n        visibility: hidden;", css)
        self.assertIn(".filter-operator {\n        justify-self: end;", css)
        self.assertIn("filterFooterCollapsed: true", js)
        self.assertIn('filterSelectionMode: "single"', js)
        self.assertIn('document.body.classList.toggle("saved-filter-grouped-mode", nextMode === "grouped");', js)
        self.assertIn('setFilterSelectionMode(state.filterSelectionMode, { apply: false });', js)
        self.assertIn("function setFilterFooterVisible(visible)", js)
        self.assertIn('document.body.classList.toggle("filter-footer-collapsed", state.filterFooterCollapsed);', js)
        self.assertIn('el("filterFooter").setAttribute("aria-hidden", String(state.filterFooterCollapsed));', js)
        self.assertIn("function syncActionTimingMonitor(tool = state.tool)", js)
        self.assertIn('DuckDB: ${formatDuckDbTimingValue(timing)}, JSON: ${formatClientTimingValue(timing)}, ${renderLabel}: ${formatRenderTimingValue(timing)}, Total: ${formatTotalTimingValue(timing)}', js)
        self.assertIn("const started = performanceImpl.now();", js)
        self.assertIn("requestAnimationFrameImpl(() => {\n        setRenderTiming(tool, performanceImpl.now() - started);", js)
        self.assertIn("function formatActionTimingValue(valueNs, status = \"idle\")", js)
        self.assertIn('if (valueNs === null || valueNs === undefined) return "--";', js)
        self.assertIn("return String(Math.round(number));", js)
        self.assertIn("if (roundedNs < 1000) return `${roundedNs}ns`;", js)
        self.assertIn('if (roundedNs < 1_000_000) return `${formatDurationNumber(roundedNs / 1000)}us`;', js)
        self.assertIn('function formatRenderTimingValue(timing)', js)
        self.assertIn('if (timing.renderStatus === "rendering") return "rendering...";', js)
        self.assertIn("function formatClientTimingValue(timing)", js)
        self.assertIn('if (timing.duckdbStatus === "running") return "--";', js)
        self.assertIn('if (timing.duckdbStatus === "failed") return "--";', js)
        self.assertIn("function formatTotalTimingValue(timing)", js)
        self.assertNotIn("function formatMapTotalTimingValue(timing)", js)
        self.assertIn("function roundedTimingMilliseconds(valueMs)", js)
        self.assertIn("function duckDbTimingMilliseconds(timing)", js)
        self.assertIn("function renderTimingMilliseconds(timing)", js)
        self.assertIn("const duckdbMs = duckDbTimingMilliseconds(timing);", js)
        self.assertIn("const jsonMs = roundedTimingMilliseconds(timing.clientDataMs);", js)
        self.assertIn("const renderMs = renderTimingMilliseconds(timing);", js)
        self.assertIn('if (duckdbMs === null || jsonMs === null || renderMs === null) return "--";', js)
        self.assertIn("return `${formatDurationNumber(duckdbMs + jsonMs + renderMs)}ms`;", js)
        self.assertNotIn("const clientTotalMs = Number(timing.clientTotalMs);", js)
        self.assertIn("function setRenderTimingRunning(tool)", js)
        self.assertIn("setRenderTimingRunning(tool);", js)
        self.assertIn("function formatDuckDbTimingValue(timing)", js)
        self.assertIn("function setClientTiming(tool, timings = {})", js)
        self.assertIn("const data = JSON.parse(text);", js)
        self.assertNotIn("const serverNs = Number(timings.server_ns);", js)
        self.assertNotIn("const serverMs = Number(timings.server_ms);", js)
        self.assertIn("const duckdbNs = Number(timings.duckdb_ns);", js)
        self.assertIn("const duckdbMs = Number(timings.duckdb_ms);", js)
        self.assertNotIn("const appNs = Number(timings.app_ns);", js)
        self.assertNotIn("const appMs = Number(timings.app_ms);", js)
        self.assertIn('return Number.isFinite(duckdbMs) ? `${formatDurationNumber(Math.max(0, duckdbMs))}ms` : "--";', js)
        self.assertIn("syncDuckDbTimingFromData(\"line_bar\", data);", js)
        self.assertIn("syncClientTimingFromData(\"column_profile\", data);", js)
        self.assertIn("syncClientTimingFromData(\"line_bar\", data);", js)
        self.assertIn("syncDuckDbTimingFromData(\"uk_map\", data);", js)
        self.assertIn("syncClientTimingFromData(\"uk_map\", data);", js)
        self.assertIn('api("/api/column-profile/summary", { method: "POST", body: JSON.stringify(request), clientTiming: true })', js)
        self.assertIn('api("/api/column-profile/detail", { method: "POST", body: JSON.stringify(request), clientTiming: true })', js)
        self.assertIn('api("/api/chart", { method: "POST", body: JSON.stringify(request), clientTiming: true })', js)
        self.assertIn('api("/api/uk-map/summary", { method: "POST", body: JSON.stringify(request), clientTiming: true })', js)
        self.assertIn("function syncFilterFooterToggleButton()", js)
        self.assertIn('el("filterFooterToggleBtn").addEventListener("click", () => setFilterFooterVisible(state.filterFooterCollapsed));', js)
        self.assertIn('el("filterSidebarClearBtn").addEventListener("click", clearFilter);', js)

    def test_development_docs_record_performance_timing_semantics(self) -> None:
        readme = Path(__file__).parents[1].joinpath("README.md").read_text(encoding="utf-8")
        development = Path(__file__).parents[1].joinpath("DEVELOPMENT.md").read_text(encoding="utf-8")

        self.assertNotIn("**Performance timings**", readme)
        self.assertIn("**Performance timings**", development)
        self.assertIn("can use `ns`, `us`, or `ms` depending on duration", development)
        self.assertIn("`DuckDB` is measured on the Python server for the active tool API request", development)
        self.assertIn("UK maps use a route-local DuckDB execute/fetch timer", development)
        self.assertIn("This does not include browser-to-server network latency", development)
        self.assertIn("All tools also show `JSON` and `Total`", development)
        self.assertIn("`Total = DuckDB + JSON + render`", development)
        self.assertIn("`Chart render`", development)
        self.assertIn("`Map render`", development)
        self.assertIn("Cached UI rerenders can update render timing without running a new DuckDB query", development)
        self.assertIn("Collapsing the filter footer hides the timing monitor", development)

    def test_chart_search_inputs_have_clear_buttons(self) -> None:
        _, html_body = self.assert_no_store("/")
        css = self.app_css_contract()
        html = html_body.decode("utf-8")
        js = self.app_js_contract()

        self.assertIn('<div class="chart-search-row">', html)
        self.assertIn('id="featureSearchClear"', html)
        self.assertIn('id="expectedSearchClear"', html)
        self.assertIn('class="filter-action" type="button" title="Clear x-axis feature search" aria-label="Clear x-axis feature search"', html)
        self.assertIn('class="filter-action" type="button" title="Clear Expected search" aria-label="Clear Expected search"', html)
        self.assertIn("&times;</button>", html)
        self.assertIn(".chart-search-row {\n        display: grid;\n        grid-template-columns: minmax(0, 1fr) 28px;\n        gap: 5px;\n        margin-bottom: 5px;", css)
        self.assertIn("function clearSearchInput(inputId, render)", js)
        self.assertIn('el("expectedSearchClear").addEventListener("click", () => clearSearchInput("expectedSearch", renderExpectedNumerators));', js)
        self.assertIn('el("featureSearchClear").addEventListener("click", () => clearSearchInput("featureSearch", renderFeatures));', js)
        self.assertIn("input.focus();", js)

    def test_date_x_axis_labels_are_month_aware(self) -> None:
        js = self.app_js_contract()

        self.assertIn("const DATE_AXIS_TARGET_LABELS = 12;", js)
        self.assertIn("const DATE_AXIS_MIN_MONTH_LABELS = 2;", js)
        self.assertIn("const DATE_AXIS_MAX_MONTH_LABELS = 14;", js)
        self.assertIn('return kind === "date" || kind === "datetime";', js)
        self.assertIn("const rawXValues = data.rows.map((r) => r.x);", js)
        self.assertIn("const xLabelPolicy = getXAxisLabelPolicy(labels, data.x_kind, rawXValues);", js)
        self.assertIn("showAllSymbol: true,", js)
        self.assertIn("interval: xLabelPolicy.interval,", js)
        self.assertIn("formatter: xLabelPolicy.formatter,", js)
        self.assertIn("showMinLabel: xLabelPolicy.showMinLabel,", js)
        self.assertIn("showMaxLabel: xLabelPolicy.showMaxLabel,", js)
        self.assertIn('function getXAxisLabelPolicy(labels, kind = "", rawValues = labels)', js)
        self.assertIn("if (isDateKind(kind)) return getDateXAxisLabelPolicy(labels, rawValues);", js)
        self.assertIn("function getDateXAxisLabelPolicy(labels, rawValues)", js)
        self.assertIn("interval: (index) => selectedIndexSet.has(index),", js)
        self.assertIn("formatDateAxisLabel(rawValues[index] ?? value, parsedDates[index])", js)
        self.assertIn(".map((date, index) => (date && date.day === 1 ? index : null))", js)
        self.assertIn("if (monthStartIndexes.length <= DATE_AXIS_MAX_MONTH_LABELS) return monthStartIndexes;", js)
        self.assertIn("return indexes.length >= DATE_AXIS_MIN_MONTH_LABELS ? indexes : sparseDateXAxisLabelIndexes(count);", js)
        self.assertIn("return sparseDateXAxisLabelIndexes(count);", js)
        self.assertIn("const indexes = new Set([0, count - 1]);", js)
        self.assertIn("return Array.from(indexes).sort((a, b) => a - b);", js)
        self.assertIn("function formatDateAxisLabel(value, parsedDate)", js)
        self.assertIn("return `${parsedDate.day} ${month} ${parsedDate.year}`;", js)
        self.assertNotIn("dateAxisYearLabelIndexes", js)
        self.assertIn("formatXLabel(r.x, data.x_kind)", js)

    def test_numeric_x_axis_labels_are_cleaned_defensively(self) -> None:
        js = self.app_js_contract()

        self.assertIn("function formatXLabel(value, kind)", js)
        self.assertIn('if (kind === "numeric") return formatNumericXLabel(value);', js)
        self.assertIn("function formatNumericXLabel(value)", js)
        self.assertIn("const number = Number(text);", js)
        self.assertIn("if (!Number.isFinite(number)) return text;", js)
        self.assertIn("return number.toLocaleString(undefined, { maximumFractionDigits: 12 });", js)
        self.assertIn('if (kind !== "integer") return String(value);', js)

    def test_line_bar_x_axis_title_uses_selected_feature_with_tight_spacing(self) -> None:
        js = self.app_js_contract()

        self.assertIn('textStyle: { color: getCss("--text"), fontWeight: 700 },', js)
        self.assertIn('name: data.x || "",', js)
        self.assertIn('nameLocation: "middle",', js)
        self.assertIn("nameGap: xLabelPolicy.nameGap,", js)
        self.assertIn('nameTextStyle: { color: getCss("--text"), fontSize: 13, fontWeight: 700 },', js)
        self.assertIn("const titleGap = rotate ? Math.max(26, labelSpace - 10) : 26;", js)
        self.assertIn("nameGap: titleGap,", js)
        self.assertIn("bottom: titleGap + 16 + dataZoomSpace,", js)
        self.assertIn("nameGap: 26,", js)
        self.assertIn("bottom: 46 + dataZoomSpace,", js)

    def test_theme_toggle_uses_icons_and_accessible_labels(self) -> None:
        _, html_body = self.assert_no_store("/")
        css = self.app_css_contract()
        html = html_body.decode("utf-8")
        js = self.app_js_contract()

        self.assertIn(".theme-toggle", css)
        self.assertIn("width: 28px;", css)
        self.assertIn("height: 24px;", css)
        self.assertIn("min-height: 24px;", css)
        self.assertIn(".theme-icon-moon", css)
        self.assertIn(".theme-icon-sun", css)
        self.assertIn("body.dark .theme-icon-moon", css)
        self.assertIn("body.dark .theme-icon-sun", css)
        self.assertNotIn("map-background-buttons", html)
        self.assertNotIn("map-background-button", css)
        self.assertNotIn("background-swatch", css)
        self.assertNotIn("mapBackground", js)
        self.assertNotIn("data-map-background", js)
        self.assertIn('label: "Light"', js)
        self.assertLess(js.index('label: "Aerial"'), js.index('label: "Light"'))
        self.assertLess(js.index('label: "Light"'), js.index('label: "Dark"'))
        self.assertIn('url: "https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png"', js)
        self.assertIn('url: "https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png"', js)
        self.assertIn('themePair: { light: "grey", dark: "darkGrey" }', js)
        self.assertIn("function applyMapBackground()", js)
        self.assertIn('const dark = document.body.classList.contains("dark");', js)
        self.assertIn('container.classList.toggle("map-bg-dark", dark);', js)
        self.assertIn('container.classList.toggle("map-bg-light", !dark);', js)
        self.assertIn("function syncCartoBaseMapForTheme()", js)
        self.assertIn("const pair = config?.themePair;", js)
        self.assertIn("if (!pair) return;", js)
        self.assertIn('setBaseMap(document.body.classList.contains("dark") ? pair.dark : pair.light);', js)
        self.assertIn("function syncThemeButton()", js)
        self.assertIn('const label = document.body.classList.contains("dark") ? "Switch to light mode" : "Switch to dark mode";', js)
        self.assertIn('el("themeBtn").setAttribute("aria-label", label);', js)
        self.assertIn('el("themeBtn").title = label;', js)
        self.assertIn('document.body.classList.toggle("dark");\n          syncThemeButton();', js)
        self.assertIn("syncCartoBaseMapForTheme();", js)
        self.assertIn('if (state.tool === "gbm") measureToolRender("gbm", () => gbmTool.refreshTheme());', js)
        self.assertIn("lineBarTool.bindControls();", js)
        self.assertIn("syncThemeButton();", js)
        self.assertIn("applyMapBackground();", js)
        self.assertNotIn('.textContent = document.body.classList.contains("dark") ? "Light" : "Dark"', js)

    def test_line_bar_quantile_control_is_numeric_only(self) -> None:
        _, html_body = self.assert_no_store("/")
        html = html_body.decode("utf-8")
        js = self.app_js_contract()

        self.assertIn('id="bandControl"', html)
        self.assertIn('id="quantileControl"', html)
        self.assertIn('import { createLineBarTool } from "./line-bar-tool.js";', js)
        self.assertIn("export function createLineBarTool", js)
        self.assertLess(html.index('id="bandControl"'), html.index('id="quantileControl"'))
        self.assertIn('<span id="bandLabel">Banding</span>', html)
        self.assertIn("<h3>Quantile</h3>", html)
        self.assertIn('<div class="segmented" data-control="quantileMode">', html)
        self.assertIn('<button data-value="off" class="active">-</button>', html)
        self.assertIn('<button data-value="quantile">Use quantiles</button>', html)
        self.assertNotIn('data-value="auto"', html)
        self.assertNotIn(">Auto<", html)
        self.assertIn('quantileMode: "off"', js)
        self.assertIn('el("bandLabel").textContent = state.quantileMode === "quantile" ? "Quantiles" : "Banding";', js)
        self.assertIn('el("quantileControl").classList.toggle("hidden", !isNumeric);', js)
        self.assertIn('quantileMode: isNumeric ? state.quantileMode : "off"', js)
        self.assertIn('/api/banding/suggestion', js)
        self.assertIn("requestBandSuggestionForSelectedColumn", js)
        self.assertIn('const previousControlValue = state[group.dataset.control];', js)
        self.assertIn('state.quantileMode === "quantile" && previousControlValue !== "quantile"', js)
        self.assertIn('state.bandWidth = "10";', js)
        self.assertIn('function normalizeBandWidthForQuantiles()', js)

    def test_gbm_shap_banding_uses_lazy_suggestion_without_auto_control(self) -> None:
        shap_js = self.assert_no_store("/static/app/gbm-shap-tool.js")[1].decode("utf-8")
        stacked_js = self.assert_no_store("/static/app/gbm-stacked-shap-tool.js")[1].decode("utf-8")

        self.assertIn('/api/banding/suggestion', shap_js)
        self.assertIn("ensureBanding", shap_js)
        self.assertIn("Estimating SHAP banding", shap_js)
        self.assertNotIn(">Auto<", shap_js)
        self.assertNotIn('data-gbm-shap-band-value="auto"', shap_js)
        self.assertIn('/api/banding/suggestion', stacked_js)
        self.assertIn("Estimating Stacked SHAP banding", stacked_js)
        self.assertNotIn(">Auto<", stacked_js)
        self.assertNotIn('data-gbm-stacked-shap-band-value="auto"', stacked_js)

    def test_london_map_button_icon_fills_button(self) -> None:
        css = self.app_css_contract()
        js = self.app_js_contract()

        self.assertIn('class="map-place-icon-london"', js)
        self.assertIn(".map-place-button img.map-place-icon-london", css)
        self.assertIn("width: 30px;", css)
        self.assertIn("height: 30px;", css)
        self.assertIn("body.dark .map-place-button img", css)
        self.assertIn("mix-blend-mode: screen;", css)
        self.assertIn("filter: invert(1) grayscale(1) brightness(1.7) contrast(1.08);", css)

    def test_map_layer_control_uses_distinct_radio_groups(self) -> None:
        _, html_body = self.assert_no_store("/")
        css = self.app_css_contract()
        html = html_body.decode("utf-8")
        js = self.app_js_contract()

        self.assertIn('import { createUkMapTool } from "./uk-map-tool.js";', js)
        self.assertIn("export function createUkMapTool", js)
        self.assertIn("ukMapTool.bindControls();", js)
        self.assertIn("ukMapTool.activate();", js)
        self.assertIn("<span>Line</span>", html)
        self.assertIn("<span>Opacity</span>", html)
        self.assertIn("<span>Extremes</span>", html)
        self.assertIn("<span>Labels</span>", html)
        self.assertIn('<span class="slider-scale"><b>Bot</b><b id="mapHotspotsValue">All</b><b>Top</b></span>', html)
        self.assertIn('<input id="mapHotspots" type="range" min="-1" max="1" step="0.1" value="0" />', html)
        self.assertNotIn("<span>Line thickness</span>", html)
        self.assertNotIn("<span>Top/Bottom %</span>", html)
        self.assertNotIn("<span>Label size</span>", html)
        self.assertNotIn("<span>Max/Min</span>", html)
        self.assertNotIn('<input id="mapHotspots" type="range" min="-0.1" max="0.1" step="0.01" value="0" />', html)
        self.assertNotIn('input id="mapHotspots" type="range" min="-20" max="20" step="5"', html)
        self.assertNotIn("Hot/not-spots", html)
        self.assertIn("--map-floating-right: 19px;", css)
        self.assertIn("--map-floating-right: 11px;", css)
        self.assertNotIn("--map-floating-right: 24px;", css)
        self.assertNotIn("--map-floating-right: 16px;", css)
        self.assertIn("width: min(430px, calc(100% - 190px));", css)
        self.assertIn("grid-template-columns: repeat(4, minmax(0, 1fr));", css)
        self.assertIn("gap: 8px 16px;", css)
        self.assertNotIn("grid-template-columns: repeat(4, 78px);", css)
        self.assertIn("font-size: 10px;", css)
        self.assertIn(".slider-scale b {\n        min-width: 20px;\n        min-height: 18px;", css)
        self.assertIn("padding: 0 2px;\n        font-size: 10px;", css)
        self.assertIn("width: min(390px, calc(100% - 112px));", css)
        self.assertIn("grid-template-columns: repeat(2, minmax(0, 1fr));", css)
        self.assertNotIn("grid-template-columns: repeat(2, 78px);", css)
        self.assertNotIn("justify-content: space-around;", css)
        self.assertIn("function mapHotspotSelection(value = state.mapHotspots)", js)
        self.assertIn("const sliderValue = Math.round(raw * 10) / 10;", js)
        self.assertIn("if (sliderValue === 0) return null;", js)
        self.assertIn("const fraction = Math.min(1, Math.max(0.1, Math.round((1.1 - magnitude) * 10) / 10));", js)
        self.assertIn("if (fraction >= 1) return null;", js)
        self.assertIn("direction: sliderValue > 0 ? -1 : 1,", js)
        self.assertIn("function mapHotspotPercent(value = state.mapHotspots)", js)
        self.assertIn("if (key === null || key === undefined || value === null) continue;", js)
        self.assertIn("const selection = mapHotspotSelection();", js)
        self.assertIn("if (!selection) return null;", js)
        self.assertIn("(a.value - b.value) * selection.direction", js)
        self.assertIn("return a.index - b.index;", js)
        self.assertIn("Math.ceil(validRows.length * selection.fraction)", js)
        self.assertIn("formatHotspotSliderValue(state.mapHotspots)", js)
        self.assertIn("function formatHotspotSliderValue(value)", js)
        self.assertIn('if (sliderValue === 0) return "All";', js)
        self.assertIn('return `${sliderValue < 0 ? "B" : "T"}${mapHotspotPercent(sliderValue)}`;', js)
        self.assertIn("if (mapHotspotSelection())", js)
        self.assertNotIn("formatPercentSliderValue(state.mapHotspots)", js)
        self.assertNotIn("Math.abs(fraction)", js)
        self.assertIn('const rightInset = styles.getPropertyValue("--map-floating-right").trim() || "19px";', js)
        self.assertIn("positionMapFloatingControlTopRight();", js)
        self.assertIn('label: "Aerial"', js)
        self.assertNotIn('label: "Satellite"', js)
        self.assertIn('type="radio" name="baseMap"', js)
        self.assertIn('type="radio" name="mapLevel" value="area"', js)
        self.assertIn('type="radio" name="mapLevel" value="sector"', js)
        self.assertIn('type="radio" name="mapLevel" value="unit"', js)
        self.assertNotIn('name="mapOverlay"', js)
        self.assertIn('target.name === "mapLevel"', js)
        self.assertIn(".uk-map .leaflet-top.leaflet-left .map-place-control", css)
        self.assertIn("grid-column: 3;", css)
        self.assertIn("--map-control-row-gap: calc(var(--map-control-gap) * 2);", css)
        self.assertIn("row-gap: var(--map-control-row-gap);", css)
        self.assertIn(".uk-map .leaflet-control-attribution", css)
        self.assertIn("font-size: 10px;", css)

    def test_sidebar_toggle_contract(self) -> None:
        css = self.app_css_contract()
        js = self.app_js_contract()

        self.assertIn("--sidebar-bg: #dce4ef;", css)
        self.assertIn("--sidebar-bg: #24334b;", css)
        self.assertIn("--sidebar-collapsed-width: 52px;", css)
        self.assertIn("background: var(--sidebar-bg);", css)
        self.assertIn(".sidebar-toggle-icon", css)
        self.assertIn(".sidebar-toggle {\n        width: 28px;\n        height: 24px;\n        min-height: 24px;", css)
        self.assertIn(".sidebar-toggle-icon {\n        width: 18px;\n        height: 20px;\n        border: 1.5px solid currentColor;", css)
        self.assertIn("border: 0;", css)
        self.assertIn("width: 5px;", css)
        self.assertIn(".sidebar-toggle-icon::after", css)
        self.assertIn("left: 3.5px;", css)
        self.assertIn("width: 1.5px;", css)
        self.assertIn("body.sidebar-collapsed .sidebar-toggle-icon::before {\n        background: transparent;", css)
        self.assertNotIn("left: 6px;", css)
        self.assertIn("body.sidebar-collapsed .shell {\n        grid-template-columns: var(--sidebar-collapsed-width) minmax(0, 1fr);", css)
        self.assertIn("body.sidebar-collapsed .sidebar-resizer {\n        display: none;", css)
        self.assertIn("body.sidebar-collapsed #appSidebar {\n        align-items: center;", css)
        self.assertIn("body.sidebar-collapsed #appSidebar > .section:not(#toolSelectorSection) {\n        display: none;", css)
        self.assertIn("#appSidebar {\n        background: var(--sidebar-bg);", css)
        self.assertNotIn("body.sidebar-collapsed aside {\n        align-items: center;", css)
        self.assertNotIn("body.sidebar-collapsed aside > .section:not(#toolSelectorSection)", css)
        self.assertNotIn("aside {\n        background: var(--sidebar-bg);", css)
        self.assertIn("body.sidebar-collapsed .tool-option {\n        width: 36px;", css)
        self.assertIn("height: 30px;\n        min-height: 30px;\n        justify-content: flex-start;", css)
        self.assertIn("padding: 3px 8px;\n        border-radius: 6px;", css)
        self.assertIn("body.sidebar-collapsed .tool-label {\n        position: absolute;", css)
        self.assertNotIn("body.sidebar-collapsed aside,\n      body.sidebar-collapsed .sidebar-resizer", css)
        self.assertIn("sidebarVisible: true", js)
        self.assertIn('document.body.classList.toggle("sidebar-collapsed", !state.sidebarVisible)', js)
        self.assertIn('el("appSidebar").removeAttribute("aria-hidden");', js)
        self.assertNotIn('el("appSidebar").setAttribute("aria-hidden", String(!state.sidebarVisible));', js)
        self.assertIn("function handleToolClick(tool)", js)
        self.assertIn("if (state.tool === tool)", js)
        self.assertIn("setSidebarVisible(!state.sidebarVisible);", js)
        self.assertIn('el("profileTool").addEventListener("click", () => handleToolClick("column_profile"));', js)
        self.assertIn('el("lineBarTool").addEventListener("click", () => handleToolClick("line_bar"));', js)
        self.assertIn('el("ukMapTool").addEventListener("click", () => handleToolClick("uk_map"));', js)
        self.assertNotIn('el("profileTool").addEventListener("click", () => setTool("column_profile"));', js)
        self.assertIn('el("sidebarToggleBtn").addEventListener("click", () => setSidebarVisible(!state.sidebarVisible))', js)
        self.assertIn('const label = state.sidebarVisible ? "Collapse sidebar" : "Expand sidebar";', js)
        self.assertIn('button.setAttribute("aria-expanded", String(state.sidebarVisible));', js)

    def test_tool_selector_aligns_with_main_toolbar(self) -> None:
        css = self.app_css_contract()

        self.assertIn(".tool-selector-section {\n        margin-bottom: 14px;\n        padding-top: 2px;", css)

    def test_column_profile_tool_static_assets_are_registered(self) -> None:
        _, html_body = self.assert_no_store("/")
        css = self.app_css_contract()
        html = html_body.decode("utf-8")
        js = self.app_js_contract()

        self.assertLess(html.index('id="profileTool"'), html.index('id="lineBarTool"'))
        self.assertIn('import { createColumnProfileTool } from "./column-profile-tool.js";', js)
        self.assertIn("export function createColumnProfileTool", js)
        self.assertIn('tool: "column_profile"', js)
        self.assertIn('column_profile: "Profile render"', js)
        self.assertIn('if (tool === "column_profile")', js)
        self.assertIn('api("/api/column-profile/summary"', js)
        self.assertIn('api("/api/column-profile/detail"', js)
        self.assertIn("function renderProfileTable(data, columns = sortedProfileColumns(data.columns || []))", js)
        self.assertIn('profileSummaryMode: "auto"', js)
        self.assertIn('mode: state.profileSummaryMode || "auto"', js)
        self.assertIn("function calculateFullProfile()", js)
        self.assertIn('state.profileSummaryMode = "full";', js)
        self.assertIn('id="profileFullCalcBtn" class="tab profile-full-calc-button" type="button">Calc all rows</button>', js)
        self.assertIn("function profileCalculation(data)", js)
        self.assertIn("function profileCalculationMeta(data)", js)
        self.assertIn("function setProfileFilterMeta(data, calculationMeta = profileCalculationMeta(data))", js)
        self.assertIn("function setProfileGroupMeta(data, groupMeta)", js)
        self.assertIn("function profileSkippedLabel(count)", js)
        self.assertIn("function profileSkippedPopoverHtml(skippedColumns)", js)
        self.assertIn("function toggleProfileSkippedPopover(event)", js)
        self.assertIn("function profileSummaryActionsHtml(data)", js)
        self.assertIn("preview ${calculation.profiledRowCount.toLocaleString()} rows", js)
        self.assertIn('<span class="profile-warning-meta">${escapeHtml(calculationMeta)}</span> · ${escapeHtml(filterLabel)}', js)
        self.assertNotIn("Preview: first ${calculation.profiledRowCount.toLocaleString()} filtered rows", js)
        self.assertIn("function refreshSelectedProfileDetail()", js)
        self.assertIn("function renderProfileDetailLoading(columnName)", js)
        self.assertIn('<h3 id="profileDetailTitle">${escapeHtml(columnName)}</h3>', js)
        self.assertIn('<div class="profile-detail-subtitle"><span>Loading profile...</span></div>', js)
        self.assertNotIn("<strong>${escapeHtml(columnName)}</strong>", js)
        self.assertIn("function renderProfileDetail(data)", js)
        self.assertIn("function profileValueCountsHtml(rows, filteredRowCount)", js)
        self.assertIn("function profileDetailSpecialCountHtml(data)", js)
        self.assertIn("function profileDetailCountBadgeHtml(count, label, flagClass)", js)
        self.assertIn("${profileDetailSpecialCountHtml(data)}", js)
        self.assertIn('return profileDetailCountBadgeHtml(Number(data.zero_count || 0), "zero", "profile-detail-zero");', js)
        self.assertIn('return profileDetailCountBadgeHtml(Number(data.blank_count || 0), "blank", "profile-detail-blank");', js)
        self.assertIn("profileDetailHistogramHtml(data.histogram || [], data.kind)", js)
        self.assertIn("function profileHistogramUsesBinLabels(bins, kind)", js)
        self.assertIn('if (kind !== "integer" || !bins.length || bins.length > 24) return false;', js)
        self.assertIn("function profileHistogramBinIsExact(bin)", js)
        self.assertIn("function profileHistogramBinLabelsHtml(bins)", js)
        self.assertIn("function profileHistogramAxisHtml(bins, kind)", js)
        self.assertIn("function profileHistogramAxisTicks(bins, kind)", js)
        self.assertIn("function profileHistogramAxisTickCount()", js)
        self.assertIn("function formatProfileAxisValue(value, kind)", js)
        self.assertIn("function compactProfileTemporalValue(value)", js)
        self.assertIn("function profileHistogramBinLabel(bin)", js)
        self.assertIn("function profileHistogramLabelFontSize(binCount)", js)
        self.assertNotIn("const showLabels = bins.length < 50;", js)
        self.assertIn('style="--profile-bin-label-size:${profileHistogramLabelFontSize(bins.length)}px"', js)
        self.assertIn('class="profile-detail-bin-label"', js)
        self.assertIn('class="profile-detail-bin-label-row"', js)
        self.assertIn('class="profile-detail-histogram-axis"', js)
        self.assertIn('class="profile-detail-histogram-axis-tick${edgeClass}"', js)
        self.assertIn('data-profile-bin-title="${escapeHtml(label)}"', js)
        self.assertIn('aria-label="${escapeHtml(label)}"', js)
        self.assertNotIn('class="profile-detail-bin" title=', js)
        self.assertIn('profile-detail-histogram-wrap"><div class="profile-detail-histogram" aria-label="Histogram"', js)
        self.assertNotIn("profile-detail-histogram-labelled", js)
        self.assertIn('profileDetailSort: { key: "count", direction: "desc" }', js)
        self.assertIn("function profileDetailSortHeaderHtml(key, label)", js)
        self.assertIn("function setProfileDetailSort(key)", js)
        self.assertIn("function sortedProfileValueCounts(rows)", js)
        self.assertIn('querySelectorAll(".profile-detail-bin[data-profile-bin-title]")', js)
        self.assertIn('bin.addEventListener("pointerenter", showProfileHistogramTooltip);', js)
        self.assertIn('bin.addEventListener("pointermove", positionProfileHistogramTooltip);', js)
        self.assertIn('bin.addEventListener("pointerleave", hideProfileHistogramTooltip);', js)
        self.assertIn('function profileHistogramTooltip()', js)
        self.assertIn('tooltip.id = "profileHistogramTooltip";', js)
        self.assertIn('function showProfileHistogramTooltip(event)', js)
        self.assertIn('function positionProfileHistogramTooltip(event)', js)
        self.assertIn('function hideProfileHistogramTooltip()', js)
        self.assertIn('button.addEventListener("click", () => setProfileDetailSort(button.dataset.profileDetailSort));', js)
        self.assertIn("function syncProfileSelectedRows()", js)
        self.assertIn('row.classList.toggle("selected", selected);', js)
        self.assertIn('row.setAttribute("aria-selected", String(selected));', js)
        self.assertNotIn("state.selectedProfileColumn = columnName;\n        renderProfileTable(state.lastProfileData);", js)
        self.assertIn("function formatProfilePercentFixed(value)", js)
        self.assertIn("return `${(number * 100).toFixed(1)}%`;", js)
        self.assertIn("} else if (distinct > 100) {", js)
        self.assertIn('return `<span class="profile-type"', js)
        self.assertIn("function profileSortHeaderHtml(key, label)", js)
        self.assertIn("function setProfileSort(key)", js)
        self.assertIn('button.addEventListener("click", () => setProfileSort(button.dataset.profileSort));', js)
        self.assertIn('row.addEventListener("click", () => selectProfileColumn(row.dataset.profileColumn || ""));', js)
        self.assertIn('row.addEventListener("contextmenu", openProfileColumnContextMenu);', js)
        self.assertIn("function profileColumnContextMenu()", js)
        self.assertIn("function openProfileColumnContextMenu(event)", js)
        self.assertIn("function closeProfileColumnContextMenu()", js)
        self.assertIn("Copy feature to clipboard", js)
        self.assertIn("navigator.clipboard.writeText(text)", js)
        self.assertIn("function fallbackCopyTextToClipboard(text)", js)
        self.assertIn("function showClipboardToast(message, isError = false)", js)
        self.assertIn('toast.id = "clipboardToast";', js)
        self.assertIn("showClipboardToast(copied ? `Copied ${columnName} to clipboard`", js)
        self.assertNotIn("setStatus(copied ? `Copied ${columnName} to clipboard`", js)
        self.assertIn("closeProfileColumnContextMenu();", js)
        self.assertIn("ensureSelectedProfileColumn(columns);", js)
        self.assertIn('aria-selected="${column.name === state.selectedProfileColumn ? "true" : "false"}"', js)
        self.assertIn('if (toolEnabled("column_profile")) return "column_profile";', js)
        self.assertIn("const skippedColumns = Array.isArray(data.skipped_columns) ? data.skipped_columns : [];", js)
        self.assertIn("const totalColumnCount = columns.length + skippedCount;", js)
        self.assertIn("columns profiled", js)
        self.assertIn("const calculationMeta = profileCalculationMeta(data);", js)
        self.assertIn('const skippedMeta = skippedCount ? profileSkippedLabel(skippedCount) : "";', js)
        self.assertIn('const groupMeta = [columnMeta, skippedMeta, rowMeta].filter(Boolean).join(" · ");', js)
        self.assertIn('const chartMessage = "";', js)
        self.assertIn("setProfileGroupMeta(data, groupMeta);", js)
        self.assertIn("setProfileFilterMeta(data, calculationMeta);", js)
        self.assertIn('saveToolPresentation("column_profile", { groupMeta, chartMessage });', js)
        self.assertIn("#profileFilter,\n      #lineBarFilter {\n        color: var(--muted);\n        font-size: 10px;", css)
        self.assertIn("grid-template-columns: minmax(0, 2fr) minmax(260px, 1fr);", css)
        self.assertIn(".profile-summary-pane,", css)
        self.assertIn(".profile-detail-pane {", css)
        self.assertIn(".profile-summary-actions {", css)
        self.assertIn("top: -32px;\n        left: 0;\n        z-index: 7;", css)
        self.assertIn("#visualArea.profile-mode .workspace-messages {\n        max-width: min(760px, calc(100% - 190px));", css)
        self.assertNotIn(".profile-summary-mode {", css)
        self.assertIn("#profileGroupMeta {", css)
        self.assertIn(".profile-skipped-button {", css)
        self.assertIn("color: var(--danger);\n        cursor: help;", css)
        self.assertIn(".profile-warning-meta {\n        color: var(--danger);", css)
        self.assertIn(".profile-skipped-popover {", css)
        self.assertIn(".profile-skipped-row {", css)
        self.assertIn("display: flex;\n        flex-direction: column;\n        overflow: visible;\n        position: relative;", css)
        self.assertIn(".profile-full-calc-button {", css)
        self.assertIn(".profile-table {", css)
        self.assertIn(".profile-sort-button {", css)
        self.assertIn(".profile-summary-row {\n        cursor: pointer;\n        user-select: none;\n        -webkit-user-select: none;", css)
        self.assertIn(".profile-summary-row.selected td {", css)
        self.assertIn(".profile-context-menu,\n      .gbm-feature-context-menu {", css)
        self.assertIn(".profile-context-menu-item,\n      .gbm-feature-context-menu-item {", css)
        self.assertIn(".clipboard-toast {\n        position: fixed;", css)
        self.assertIn(".clipboard-toast[hidden] {\n        display: none;", css)
        self.assertIn(".gbm-model-grid .tabulator-row,\n      .gbm-tree-summary-grid .tabulator-row,\n      .gbm-model-table tr[data-gbm-model-row],\n      .gbm-tree-fallback-table tr[data-gbm-tree-row] {\n        cursor: pointer;\n        user-select: none;\n        -webkit-user-select: none;", css)
        self.assertNotIn(".table-wrap table {\n        user-select: none;", css)
        self.assertNotIn(".profile-count-table {\n        user-select: none;", css)
        self.assertIn(".profile-badge-warning {", css)
        self.assertIn(".profile-type,\n      .profile-missing-count {", css)
        self.assertIn(".profile-detail-counts .profile-detail-missing,\n      .profile-detail-counts .profile-detail-zero,\n      .profile-detail-counts .profile-detail-blank {", css)
        self.assertIn(".profile-detail-histogram-wrap {", css)
        self.assertIn(".profile-detail-histogram {", css)
        self.assertIn("display: flex;\n        flex-wrap: nowrap;\n        align-items: flex-end;", css)
        self.assertIn("min-width: 0;\n        overflow: hidden;", css)
        profile_histogram_block = css[css.index(".profile-detail-histogram {"):css.index(".profile-detail-bin {")]
        self.assertNotIn("overflow-x: auto;", profile_histogram_block)
        self.assertIn("flex: 1 1 0;\n        position: relative;\n        height: 100%;\n        min-width: 0;", css)
        self.assertNotIn(".profile-detail-histogram-labelled", css)
        self.assertNotIn("padding-bottom: 14px;", css)
        self.assertIn(".profile-detail-bin-label-row {", css)
        self.assertIn(".profile-detail-bin-label {", css)
        self.assertIn("font-size: var(--profile-bin-label-size, 7px);", css)
        self.assertIn("text-align: center;", css)
        self.assertIn(".profile-detail-histogram-axis {", css)
        self.assertIn(".profile-detail-histogram-axis-tick {", css)
        self.assertIn(".profile-detail-histogram-axis-tick-start {", css)
        self.assertIn(".profile-detail-histogram-axis-tick-end {", css)
        self.assertIn(".profile-histogram-tooltip {", css)
        self.assertIn("position: fixed;\n        z-index: 1000;", css)
        self.assertIn("pointer-events: none;", css)
        self.assertIn(".profile-histogram-tooltip[hidden] {\n        display: none;", css)
        self.assertNotIn("grid-template-columns: repeat(20, minmax(3px, 1fr));", css)
        self.assertIn(".profile-stats-table {", css)
        self.assertIn(".profile-count-value {", css)
        self.assertIn(".profile-count-percent {", css)
        self.assertIn(".profile-count-sort-button {", css)
        self.assertIn("grid-template-columns: repeat(auto-fit, minmax(170px, 1fr));", css)
        self.assertIn("overflow: visible;\n        text-overflow: clip;", css)
        self.assertIn(".profile-count-table {", css)
        self.assertNotIn("<th>Sample values</th>", js)
        self.assertNotIn("<th>Distribution</th>", js)
        self.assertNotIn("function profileSamplesHtml", js)
        self.assertNotIn("function profileDistributionHtml", js)

    def test_app_js_contains_unit_point_map_controls(self) -> None:
        js = self.app_js_contract()

        self.assertIn('unitColumn: postcodeColumn("unit")', js)
        self.assertIn('latitudeColumn: latitudeColumn()', js)
        self.assertIn('longitudeColumn: longitudeColumn()', js)
        self.assertIn('compactUnitPoints: state.mapLevel === "unit"', js)
        self.assertIn('aliases: ["PostcodeUnit", "POSTCODE_UNIT"]', js)
        self.assertIn('longitude: ["long", "longitude", "LONGITUDE", "LONGiTUDE"]', js)
        self.assertIn("function unitPointArrays(data)", js)
        self.assertIn("function unitPointEntries(data)", js)
        self.assertIn("function makeUnitPointScale(data)", js)
        self.assertIn("function mapUnitHotspotKeys(data)", js)
        self.assertIn("makeUnitPointLayer", js)
        self.assertIn("setRenderContext(nextScale, nextHotspotKeys)", js)
        self.assertIn("this.scale = nextScale;", js)
        self.assertIn("this.hotspotKeys = nextHotspotKeys;", js)
        self.assertIn("const style = mapPointStyle(entry, this.scale, this.hotspotKeys, pointRadius);", js)
        self.assertIn("unitPointRadiusForZoom", js)
        self.assertIn("unitPointHitRadius(pointRadius)", js)
        self.assertIn("if (pointRadius <= 1)", js)
        self.assertIn("fillRect(point.x - pointRadius", js)
        self.assertIn("if (!ukMapPointLayer?.setRenderContext)", js)
        self.assertIn("ukMapPointLayer.setRenderContext(scale, hotspotKeys);", js)
        self.assertIn('renderMapLegend(scale, state.lastMapData.response?.label || "Actual");', js)
        self.assertIn("<span>Units</span>", js)

    def test_app_js_preserves_map_view_after_layout_resize(self) -> None:
        js = self.app_js_contract()

        self.assertIn("mapView: null", js)
        self.assertIn("mapViewportSyncFrame: null", js)
        self.assertIn("restoringMapView: false", js)
        self.assertIn("function currentMapView()", js)
        self.assertIn('function captureMapView(reason = "")', js)
        self.assertIn("function restoreMapView(view)", js)
        self.assertIn('function scheduleMapViewportSync({ mode = "preserve" } = {})', js)
        self.assertIn("mapResizeObserver = new ResizeObserver", js)
        self.assertIn('scheduleMapViewportSync({ mode: "preserve" });', js)
        self.assertIn('ukMapTool.captureView("tool-switch")', js)
        self.assertIn('captureMapView("map-level-change")', js)
        self.assertIn("zoomSnap: 0.25", js)
        self.assertIn("zoomDelta: 0.5", js)
        self.assertIn("const MAP_INITIAL_FIT_OPTIONS = { animate: false };", js)
        self.assertIn("mapStartupFitDone: false", js)
        self.assertIn("if (!state.mapStartupFitDone)", js)
        self.assertIn("state.mapStartupFitDone = true;", js)
        self.assertIn("fitMapBounds(bounds, level, MAP_INITIAL_FIT_OPTIONS)", js)
        self.assertIn("return { animate: false, ...fitOptions, ...options };", js)
        self.assertIn("if (shouldPreserve && view) restoreMapView(view);", js)

    def test_app_js_reuses_cached_polygon_layers(self) -> None:
        js = self.app_js_contract()

        self.assertIn("mapPolygonLayerCache: {}", js)
        self.assertIn("mapPolygonRenderContext: null", js)
        self.assertIn("function cachedMapPolygonLayer(level, geoJson)", js)
        self.assertIn("if (!state.mapPolygonLayerCache[level])", js)
        self.assertIn("layer: createMapPolygonLayer(level, geoJson)", js)
        self.assertIn("layer._lucidumMapKey = mapPolygonFeatureKey(feature, levelConfig.property);", js)
        self.assertIn("layer.bindTooltip(() => mapPolygonTooltipHtml(layer), { sticky: true });", js)
        self.assertIn("layer.bindPopup(() => mapPolygonPopupHtml(layer));", js)
        self.assertIn("ukMapLayer = cachedPolygonLayer.layer;", js)
        self.assertIn("applyMapPolygonStyles();", js)
        self.assertIn("if (!ukMap.hasLayer(ukMapLayer)) ukMapLayer.addTo(ukMap);", js)
        self.assertIn('if (state.lastMapData?.level === "sector") restyleActiveMapPolygonLayer();', js)
        self.assertNotIn('if (state.lastMapData?.level === "sector") redrawMapInPlace();', js)
        self.assertIn("function restyleActiveMapPolygonLayer()", js)
        self.assertIn("function countMatchedMapPolygonFeatures(layer, summaries)", js)
        self.assertIn("smoothFactor: 1", js)
        self.assertIn("smoothFactor: 0", js)
        self.assertIn("smoothFactor: levelConfig.smoothFactor ?? 1", js)

    def test_uk_map_documents_current_interactivity_contract(self) -> None:
        js = self.app_js_contract()
        development = Path(__file__).parents[1].joinpath("DEVELOPMENT.md").read_text(encoding="utf-8")

        self.assertIn("layer.bindTooltip(() => mapPolygonTooltipHtml(layer), { sticky: true });", js)
        self.assertIn("layer.bindPopup(() => mapPolygonPopupHtml(layer));", js)
        self.assertIn('map.on("mousemove", this.handleMouseMove, this);', js)
        self.assertIn('map.on("click", this.handleClick, this);', js)
        self.assertIn("this.hitGrid = new Map();", js)
        self.assertIn("this.hitGrid.get(gridKey).push({ entry, point });", js)
        self.assertIn("Area and sector geometry use Leaflet GeoJSON with hover tooltips and click popups.", development)
        self.assertIn("Unit points render on a canvas-backed Leaflet layer with a hit grid for hover tooltips and click popups.", development)
        self.assertIn("a geographic viewport prefilter before projection is not part of the current rendering strategy", development)

    def test_monitor_docs_describe_lucidum_server_panel(self) -> None:
        telemetry_doc = Path(__file__).parents[1].joinpath("docs/telemetry-monitor.md").read_text(encoding="utf-8")
        development = Path(__file__).parents[1].joinpath("DEVELOPMENT.md").read_text(encoding="utf-8")

        self.assertIn("The `lucidum servers` table lists local Lucidum-looking server processes", telemetry_doc)
        self.assertIn("highlights that current server row", telemetry_doc)
        self.assertIn("clickable server addresses that open in a new tab", telemetry_doc)
        self.assertIn("Each stoppable row has a compact `X` button", telemetry_doc)
        self.assertIn("stopping a sibling server terminates the matching same-user Lucidum process", telemetry_doc)
        self.assertIn("It is not shared across multiple Lucidum processes. The `lucidum servers` table can discover sibling servers", telemetry_doc)
        self.assertNotIn("open each server's own `/monitor` page", telemetry_doc)
        self.assertIn("GET /api/lucidum-servers", development)
        self.assertIn("POST /api/lucidum-servers/stop", development)

    def test_uk_map_static_assets_disable_cache(self) -> None:
        self.assert_no_store("/tools/uk-map/static/icons/UK.png")

    def test_favicon_disables_cache(self) -> None:
        self.assert_no_store("/favicon.ico")


class HealthEndpointTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.data_path = Path(self.tmp.name) / "sample.csv"
        self.data_path.write_text("PostcodeArea,PostcodeSector,Actual\nAB,AB10 1,100\n", encoding="utf-8")

    def test_health_route_is_registered(self) -> None:
        app = create_app(self.data_path)
        paths = {route.path for route in app.routes}

        self.assertIn("/api/health", paths)

    def test_health_returns_success_without_token_auth(self) -> None:
        app = create_app(self.data_path, token="")
        status, _, body = asgi_get(app, "/api/health")

        self.assertEqual(status, 200)
        self.assertEqual(body, b'{"status":"ok"}')

    def test_health_rejects_missing_or_invalid_token(self) -> None:
        app = create_app(self.data_path, token="dev-token")

        missing_status, _, missing_body = asgi_get(app, "/api/health")
        invalid_status, _, invalid_body = asgi_get(app, "/api/health", headers={"x-lucidum-token": "bad-token"})

        self.assertEqual(missing_status, 401)
        self.assertIn(b"Invalid or missing app token", missing_body)
        self.assertEqual(invalid_status, 401)
        self.assertIn(b"Invalid or missing app token", invalid_body)

    def test_health_accepts_valid_token(self) -> None:
        app = create_app(self.data_path, token="dev-token")
        status, _, body = asgi_get(app, "/api/health", headers={"x-lucidum-token": "dev-token"})

        self.assertEqual(status, 200)
        self.assertEqual(body, b'{"status":"ok"}')


if __name__ == "__main__":
    unittest.main()
