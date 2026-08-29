from __future__ import annotations

import asyncio
import json
import mimetypes
import re
import shutil
import subprocess
import tomllib
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

from py_lucidum import __version__
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
    @classmethod
    def setUpClass(cls) -> None:
        cls.tmp = TemporaryDirectory()
        cls.data_path = Path(cls.tmp.name) / "sample.csv"
        cls.data_path.write_text("PostcodeArea,PostcodeSector,Actual\nAB,AB10 1,100\n", encoding="utf-8")
        cls.app = create_app(cls.data_path)

    @classmethod
    def tearDownClass(cls) -> None:
        cls.tmp.cleanup()

    def assert_no_store(self, path: str) -> tuple[dict[str, str], bytes]:
        status, headers, body = asgi_get(self.app, path)

        self.assertEqual(status, 200)
        self.assertEqual(headers.get("cache-control"), "no-store")
        return headers, body

    def root_html_for_tools(self, tools: list[str] | None, *, header_buttons: bool = False) -> str:
        app = create_app(self.data_path, tools=tools, header_buttons=header_buttons)
        status, headers, body = asgi_get(app, "/")

        self.assertEqual(status, 200)
        self.assertEqual(headers.get("cache-control"), "no-store")
        return body.decode("utf-8")

    def assert_tool_button_visibility(self, html: str, expected_visible_tools: set[str]) -> None:
        all_tools = {"dataset_viewer", "column_profile", "line_bar", "histogram", "uk_map", "glm", "gbm", "specs"}
        for tool_id in all_tools:
            with self.subTest(tool=tool_id):
                match = re.search(rf'<button\b[^>]*\bdata-tool="{re.escape(tool_id)}"[^>]*>', html)
                self.assertIsNotNone(match)
                class_match = re.search(r'\bclass="([^"]*)"', match.group(0))
                self.assertIsNotNone(class_match)
                classes = set(class_match.group(1).split())
                self.assertEqual("hidden" not in classes, tool_id in expected_visible_tools)

    def assert_tool_selector_visibility(self, html: str, expected_visible: bool) -> None:
        match = re.search(r'<section\b[^>]*\bid="toolSelectorSection"[^>]*>', html)
        self.assertIsNotNone(match)
        class_match = re.search(r'\bclass="([^"]*)"', match.group(0))
        self.assertIsNotNone(class_match)
        classes = set(class_match.group(1).split())
        self.assertEqual("hidden" not in classes, expected_visible)

    def assert_tool_button_order(self, html: str, expected_order: list[str]) -> None:
        positions = [html.index(f'data-tool="{tool_id}"') for tool_id in expected_order]
        self.assertEqual(positions, sorted(positions))

    def assert_model_sidebar_panel_visibility(self, html: str, expected_visible_tools: set[str]) -> None:
        panels = {"gbm": "gbmSidebarPanel", "glm": "glmSidebarPanel"}
        for tool_id, panel_id in panels.items():
            with self.subTest(panel=panel_id):
                match = re.search(rf'<section\b[^>]*\bid="{re.escape(panel_id)}"[^>]*>', html)
                self.assertIsNotNone(match)
                tag = match.group(0)
                class_match = re.search(r'\bclass="([^"]*)"', tag)
                self.assertIsNotNone(class_match)
                classes = set(class_match.group(1).split())
                visible = "hidden" not in classes
                self.assertEqual(visible, tool_id in expected_visible_tools)
                self.assertEqual('aria-hidden="true"' in tag, not visible)
                self.assertEqual(bool(re.search(r"\binert\b", tag)), not visible)





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

    def test_api_client_adds_operation_id_only_when_supplied(self) -> None:
        module = Path("src/py_lucidum/static/app/shared/api.js").resolve().as_uri()
        script = f"""
import {{ createApiClient, createOperationId }} from "{module}";
const calls = [];
const fetchImpl = async (path, options) => {{
  calls.push({{ path, options }});
  return {{ ok: true, text: async () => '{{"ok":true}}' }};
}};
let tick = 0;
const api = createApiClient({{
  token: "token",
  fetchImpl,
  performanceImpl: {{ now: () => ++tick }},
}});
const operationId = createOperationId("GBM train");
if (!/^gbm-train-[A-Za-z0-9._-]+$/.test(operationId)) throw new Error(operationId);
await api("/with-operation", {{ method: "POST", operationId }});
await api("/without-operation", {{ method: "GET" }});
if (calls[0].options.headers["x-lucidum-operation-id"] !== operationId) throw new Error("missing operation header");
if ("x-lucidum-operation-id" in calls[1].options.headers) throw new Error("unexpected operation header");
if ("operationId" in calls[0].options) throw new Error("operation option leaked to fetch");
"""
        self.run_node_script(script)

    def test_shared_echarts_target_readiness_helper(self) -> None:
        module = Path("src/py_lucidum/static/app/shared/echarts-gl.js").resolve().as_uri()
        script = f"""
import {{ isEchartsTargetReady }} from "{module}";
const target = (isConnected, clientWidth, clientHeight) => ({{ isConnected, clientWidth, clientHeight }});
if (isEchartsTargetReady(null)) throw new Error("missing target should not be ready");
if (isEchartsTargetReady(target(false, 640, 480))) throw new Error("disconnected target should not be ready");
if (isEchartsTargetReady(target(true, 0, 480))) throw new Error("zero-width target should not be ready");
if (isEchartsTargetReady(target(true, 640, 0))) throw new Error("zero-height target should not be ready");
if (!isEchartsTargetReady(target(true, 640, 480))) throw new Error("positive connected target should be ready");
"""
        self.run_node_script(script)

    def test_shared_line_bar_renderer_supports_static_glm_and_shap_reports(self) -> None:
        module = Path("src/py_lucidum/static/app/line-bar-chart.js").resolve().as_uri()
        script = f"""
import {{ lineBarChartOption }} from "{module}";
const base = {{
  x: "AGE",
  x_kind: "integer",
  x_group_kind: "integer",
  denominator: {{ column: null, bar_label: "Weight" }},
  responses: [
    {{ label: "Actual", numerator: "PREMIUM" }},
    {{ label: "GBM prediction", numerator: "gbm_prediction" }},
  ],
  rows: [
    {{ x: "20", volume: 10, resp0: 100, resp1: 105 }},
    {{ x: "30", volume: 20, resp0: 120, resp1: 118 }},
  ],
}};
const glm = lineBarChartOption({{
  ...base,
  transform: {{ mode: "none" }},
  partial_dependence: {{ mode: "glm", rows: [{{ x: "20", p50: 101 }}, {{ x: "30", p50: 117 }}] }},
}}, {{ content: "actual_expected" }}).option;
const glmNames = glm.series.map((series) => series.name);
if (!glmNames.includes("Actual") || !glmNames.includes("GBM prediction") || !glmNames.includes("Weight") || !glmNames.includes("GLM")) throw new Error(glmNames.join("|"));
if (glm.yAxis.length !== 2) throw new Error("A/E needs a response and weight axis");

const comparison = lineBarChartOption({{
  ...base,
  transform: {{ mode: "none" }},
}}, {{ content: "actual_expected", xAxisTitle: "Challenger / Baseline" }}).option;
if (comparison.xAxis.name !== "Challenger / Baseline") throw new Error("comparison axis title failed");

const currency = lineBarChartOption({{
  ...base,
  transform: {{ mode: "none" }},
}}, {{ content: "actual_expected", labels: "line", kpiFormat: {{ decimals: 2, format: "currency" }} }}).option;
if (currency.yAxis[0].axisLabel.formatter(-12.5) !== "-£12.50") throw new Error("currency y-axis formatting failed");
if (currency.series.find((series) => series.name === "Actual").label.formatter({{ value: 12.5 }}) !== "£12.50") throw new Error("currency line-label formatting failed");
const currencyTooltip = currency.tooltip.formatter([
  {{ axisValueLabel: "20", seriesName: "Actual", value: 12.5, marker: "" }},
  {{ axisValueLabel: "20", seriesName: "Weight", value: 10, marker: "" }},
]);
if (!currencyTooltip.includes("Actual: £12.50") || !currencyTooltip.includes("Weight: 10")) throw new Error(currencyTooltip);
if (currency.yAxis[1].axisLabel.formatter(12.5).includes("£")) throw new Error("Weight must keep generic formatting");

const percent = lineBarChartOption({{
  ...base,
  transform: {{ mode: "none" }},
}}, {{ content: "actual_expected", kpiFormat: {{ decimals: 1, format: "percent" }} }}).option;
if (percent.yAxis[0].axisLabel.formatter(0.125) !== "12.5%") throw new Error("percent y-axis formatting failed");

const upliftWithKpi = lineBarChartOption({{
  ...base,
  transform: {{ mode: "one" }},
}}, {{ content: "actual_expected", kpiFormat: {{ decimals: 2, format: "currency" }} }}).option;
if (upliftWithKpi.yAxis[0].axisLabel.formatter(1.25) !== "+25%") throw new Error("uplift formatting must override KPI formatting");

const shapRows = [
  {{ x: "20", p0: .7, p5: .75, p10: .8, p20: .85, p30: .9, p40: .95, p50: 1, p60: 1.05, p70: 1.1, p80: 1.15, p90: 1.2, p95: 1.25, p100: 1.3 }},
  {{ x: "30", p0: .8, p5: .82, p10: .84, p20: .88, p30: .92, p40: .97, p50: 1.02, p60: 1.07, p70: 1.12, p80: 1.18, p90: 1.24, p95: 1.28, p100: 1.35 }},
];
const shap = lineBarChartOption({{
  ...base,
  transform: {{ mode: "one" }},
  partial_dependence: {{ mode: "shap", rows: shapRows }},
}}, {{ content: "shap_only" }}).option;
const shapNames = shap.series.map((series) => series.name);
if (shapNames.includes("Actual") || shapNames.includes("GBM prediction") || shapNames.includes("Weight")) throw new Error(shapNames.join("|"));
if (!shapNames.includes("SHAP Min-Max") || !shapNames.includes("SHAP median")) throw new Error(shapNames.join("|"));
if (shap.yAxis.length !== 1 || shap.yAxis[0].name !== "SHAP relativity") throw new Error("SHAP-only axis failed");
"""
        self.run_node_script(script)

    def test_line_bar_importance_model_order_follows_expected_model_family(self) -> None:
        module = Path("src/py_lucidum/static/app/line-bar-tool.js").resolve().as_uri()
        script = f"""
import {{ lineBarImportanceModelOrder }} from "{module}";
const order = (selections) => lineBarImportanceModelOrder(selections).join("|");
const dataset = {{ value: "benchmark", sourceId: "dataset", metricKind: "metric" }};
const datasetNamedLikeGlm = {{ value: "glm_prediction", sourceId: "dataset", metricKind: "metric" }};
const glm = (value) => ({{ value, sourceId: "glm:model:predictions", metricKind: "prediction" }});
const gbm = (value) => ({{ value, sourceId: "gbm:model:predictions", metricKind: "prediction" }});
if (order([]) !== "gbm|glm") throw new Error("empty Expected order failed");
if (order([dataset]) !== "gbm|glm") throw new Error("dataset-only Expected order failed");
if (order([datasetNamedLikeGlm]) !== "gbm|glm") throw new Error("dataset column name should not identify a model family");
if (order([dataset, glm("glm_prediction")]) !== "glm|gbm") throw new Error("dataset plus GLM order failed");
if (order([glm("glm_prediction")]) !== "glm|gbm") throw new Error("GLM prediction order failed");
if (order([glm("glm_prediction_rate")]) !== "glm|gbm") throw new Error("GLM rate order failed");
if (order([glm("glm_tabulated_prediction")]) !== "glm|gbm") throw new Error("GLM tabulated order failed");
if (order([gbm("gbm_prediction")]) !== "gbm|glm") throw new Error("GBM prediction order failed");
if (order([gbm("gbm_prediction_rate")]) !== "gbm|glm") throw new Error("GBM rate order failed");
if (order([gbm("gbm_tabulated_prediction")]) !== "gbm|glm") throw new Error("GBM tabulated order failed");
if (order([glm("glm_prediction"), gbm("gbm_prediction")]) !== "gbm|glm") throw new Error("mixed model order failed");
"""
        self.run_node_script(script)

    def test_line_bar_model_switch_policy_aligns_predictions_and_partial_dependence(self) -> None:
        module = Path("src/py_lucidum/static/app/line-bar-tool.js").resolve().as_uri()
        script = f"""
import {{ nextLineBarModelComparisonState }} from "{module}";
const dataset = (value) => ({{ value, sourceId: "dataset", metricKind: "metric" }});
const glm = (value = "glm_prediction", model = "old-glm") => ({{
  value, sourceId: `glm:${{model}}:predictions`, metricKind: "prediction",
}});
const gbm = (value = "gbm_prediction", model = "old-gbm") => ({{
  value, sourceId: `gbm:${{model}}:predictions`, metricKind: "prediction",
}});
const primary = {{ glm: glm("glm_prediction", "new-glm"), gbm: gbm("gbm_prediction", "new-gbm") }};
const defaults = {{
  primaryExpectedSelections: primary,
  predictionAvailable: {{ glm: true, gbm: true }},
  overlayAvailable: {{ glm: true, gbm: true }},
  modelMetricMatches: {{ glm: true, gbm: true }},
  activeModelAvailable: {{ glm: true, gbm: true }},
}};
const transition = (options) => nextLineBarModelComparisonState({{ ...defaults, ...options }});
const values = (result) => result.expectedSelections.map((selection) => selection.value).join("|");
const sources = (result) => result.expectedSelections.map((selection) => selection.sourceId).join("|");

let result = transition({{
  expectedSelections: [], partialDependence: "none", activatedModelKind: "gbm",
}});
if (values(result) !== "gbm_prediction" || result.partialDependence !== "none") throw new Error("empty comparison was not populated");

result = transition({{
  expectedSelections: [dataset("benchmark")], partialDependence: "none", activatedModelKind: "gbm",
}});
if (values(result) !== "gbm_prediction" || result.partialDependence !== "none") throw new Error("dataset comparison was not replaced");

result = transition({{
  expectedSelections: [dataset("benchmark"), dataset("plan")], partialDependence: "none", activatedModelKind: "glm",
}});
if (values(result) !== "glm_prediction" || result.partialDependence !== "none") throw new Error("dataset comparison pair was not replaced");

result = transition({{
  expectedSelections: [glm()], partialDependence: "glm", activatedModelKind: "gbm",
}});
if (values(result) !== "gbm_prediction" || result.partialDependence !== "shap") throw new Error("GLM to GBM switch failed");

result = transition({{
  expectedSelections: [gbm()], partialDependence: "shap", activatedModelKind: "glm",
}});
if (values(result) !== "glm_prediction" || result.partialDependence !== "glm") throw new Error("GBM to GLM switch failed");

result = transition({{
  expectedSelections: [glm(), gbm()], partialDependence: "both", activatedModelKind: "gbm", metricsChanged: false,
}});
if (values(result) !== "glm_prediction|gbm_prediction") throw new Error("compatible prediction pair was not preserved");
if (sources(result) !== "glm:new-glm:predictions|gbm:new-gbm:predictions") throw new Error("compatible prediction pair was not rebound");
if (result.partialDependence !== "both" || result.expectedChanged || result.partialDependenceChanged) throw new Error("compatible comparison state changed");

const explicitPair = [
  {{ ...glm("glm_prediction", "fixed-glm-a"), binding: "explicit" }},
  {{ ...glm("glm_prediction", "fixed-glm-b"), binding: "explicit" }},
];
result = transition({{
  expectedSelections: explicitPair, partialDependence: "none", activatedModelKind: "glm", metricsChanged: true,
}});
if (sources(result) !== "glm:fixed-glm-a:predictions|glm:fixed-glm-b:predictions") throw new Error("explicit comparison was rebound");
if (result.expectedChanged) throw new Error("explicit comparison reported an Expected change");

result = transition({{
  expectedSelections: [glm(), gbm()], partialDependence: "both", activatedModelKind: "gbm", metricsChanged: true,
}});
if (values(result) !== "gbm_prediction" || result.partialDependence !== "shap") throw new Error("changed metric pair was not collapsed");
if (!result.expectedChanged || !result.partialDependenceChanged) throw new Error("changed comparison was not reported");

result = transition({{
  expectedSelections: [glm(), gbm()], partialDependence: "both", activatedModelKind: "glm",
  modelMetricMatches: {{ glm: true, gbm: false }},
}});
if (values(result) !== "glm_prediction" || result.partialDependence !== "glm") throw new Error("incompatible active models were preserved");

result = transition({{
  expectedSelections: [glm("glm_prediction_rate")], partialDependence: "glm", activatedModelKind: "glm",
}});
if (values(result) !== "glm_prediction" || !result.expectedChanged) throw new Error("advanced prediction was not normalized");

result = transition({{
  expectedSelections: [glm()], partialDependence: "glm", activatedModelKind: "gbm",
  overlayAvailable: {{ glm: true, gbm: false }},
}});
if (values(result) !== "gbm_prediction" || result.partialDependence !== "glm") throw new Error("unavailable SHAP fallback failed");
if (!result.activatedOverlayUnavailable) throw new Error("unavailable SHAP was not reported");

result = transition({{
  expectedSelections: [dataset("benchmark")], partialDependence: "none", activatedModelKind: "gbm",
  primaryExpectedSelections: {{ glm: primary.glm, gbm: null }},
  predictionAvailable: {{ glm: true, gbm: false }},
}});
if (values(result) !== "benchmark") throw new Error("unavailable prediction did not preserve dataset comparison");
if (!result.activatedPredictionUnavailable) throw new Error("unavailable prediction was not reported");

result = transition({{
  expectedSelections: [glm(), gbm()], partialDependence: "both", activatedModelKind: "gbm",
  primaryExpectedSelections: {{ glm: primary.glm, gbm: null }},
  predictionAvailable: {{ glm: true, gbm: false }},
  overlayAvailable: {{ glm: true, gbm: false }},
  modelMetricMatches: {{ glm: true, gbm: false }},
  activeModelAvailable: {{ glm: true, gbm: false }},
}});
if (values(result) !== "glm_prediction" || result.partialDependence !== "glm") throw new Error("final model deletion fallback failed");
"""
        self.run_node_script(script)

    def test_line_bar_model_comparison_candidates_filter_defaults_and_labels(self) -> None:
        module = Path("src/py_lucidum/static/app/line-bar-tool.js").resolve().as_uri()
        script = f"""
import {{
  lineBarCompatibleExpectedColumns,
  lineBarDefaultModelComparison,
  lineBarModelComparisonCandidates,
  lineBarOtherColumnToken,
}} from "{module}";
const column = (name) => ({{ name, kind: "numeric" }});
const sources = [
  {{
    id: "glm:old:predictions", kind: "glm_predictions", model_id: "old", model_label: "Pricing",
    active: false, created_at: "2026-01-01", response_column: "Actual", denominator_column: "Weight",
    columns: [column("glm_prediction"), column("glm_prediction_rate"), column("glm_tabulated_prediction")],
  }},
  {{
    id: "glm:new:predictions", kind: "glm_predictions", model_id: "new", model_label: "Pricing",
    active: true, created_at: "2026-02-01", response_column: "Actual", denominator_column: "Weight",
    columns: [column("glm_prediction")],
  }},
  {{
    id: "gbm:current:predictions", kind: "gbm_predictions", model_id: "current", model_label: "Challenger",
    active: true, created_at: "2026-03-01", response_column: "Actual", offset_column: "Weight",
    columns: [column("gbm_prediction")],
  }},
  {{
    id: "gbm:wrong-kpi:predictions", kind: "gbm_predictions", model_id: "wrong-kpi", model_label: "Wrong",
    active: false, created_at: "2026-04-01", response_column: "Other", offset_column: "Weight",
    columns: [column("gbm_prediction")],
  }},
  {{
    id: "gbm:no-primary:predictions", kind: "gbm_predictions", model_id: "no-primary", model_label: "Missing",
    active: false, created_at: "2026-05-01", response_column: "Actual", offset_column: "Weight",
    columns: [column("gbm_prediction_rate")],
  }},
  {{
    id: "gbm:invalid-primary:predictions", kind: "gbm_predictions", model_id: "invalid-primary", model_label: "Invalid",
    active: false, created_at: "2026-05-02", response_column: "Actual", offset_column: "Weight",
    columns: [{{ name: "gbm_prediction", kind: "categorical" }}],
  }},
];
const kpi = {{ numerator: "Actual", numeratorSource: "dataset", denominator: "Weight", denominatorSource: "dataset" }};
let candidates = lineBarModelComparisonCandidates(sources, kpi);
if (candidates.map((candidate) => candidate.sourceId).join("|") !== "glm:old:predictions|glm:new:predictions|gbm:current:predictions") {{
  throw new Error(`candidate filtering failed: ${{JSON.stringify(candidates)}}`);
}}
if (new Set(candidates.map((candidate) => candidate.label)).size !== candidates.length) throw new Error("duplicate picker labels were not disambiguated");
if (!candidates[0].label.includes("(old)") || !candidates[1].label.includes("(new)")) throw new Error("duplicate model ids were not shown");
let pair = lineBarDefaultModelComparison(candidates);
if (pair.baseline.sourceId !== "glm:new:predictions" || pair.challenger.sourceId !== "gbm:current:predictions") {{
  throw new Error("active GLM/GBM order was not preserved");
}}
pair = lineBarDefaultModelComparison(candidates, {{
  baselineSourceId: "glm:old:predictions", challengerSourceId: "glm:new:predictions",
}});
if (pair.baseline.sourceId !== "glm:old:predictions" || pair.challenger.sourceId !== "glm:new:predictions") {{
  throw new Error("existing exact pair was not preserved");
}}
candidates = lineBarModelComparisonCandidates(sources, kpi);
const mismatch = candidates.find((candidate) => candidate.sourceId === "gbm:wrong-kpi:predictions");
if (mismatch) throw new Error("configured KPI mismatch was retained");
const expectedColumns = lineBarCompatibleExpectedColumns(sources, kpi);
const expectedKeys = expectedColumns.map((item) => `${{item.source_id}}:${{item.name}}`).join("|");
if (expectedKeys !== [
  "glm:new:predictions:glm_prediction",
  "glm:old:predictions:glm_prediction",
  "glm:old:predictions:glm_tabulated_prediction",
  "gbm:current:predictions:gbm_prediction",
].join("|")) throw new Error(`compatible Expected ordering failed: ${{expectedKeys}}`);
if (expectedColumns.some((item) => item.name.endsWith("_prediction_rate"))) throw new Error("prediction rate leaked into Expected");
if (expectedColumns[0].label !== "GLM · Pricing (new)" || expectedColumns[1].label !== "GLM · Pricing (old)") {{
  throw new Error(`duplicate Expected labels were not disambiguated: ${{JSON.stringify(expectedColumns)}}`);
}}
if (expectedColumns[2].label !== "GLM · Pricing (old) · tabulated") throw new Error("tabulated label failed");
if (expectedColumns[0].model_binding !== "model") throw new Error("active Expected binding failed");
if (expectedColumns[1].model_binding !== "explicit" || expectedColumns[2].model_binding !== "explicit") {{
  throw new Error("inactive Expected bindings were not exact");
}}
const noDenominatorSource = {{
  id: "glm:none:predictions", kind: "glm_predictions", model_id: "none", model_label: "No denominator",
  active: false, created_at: "2026-06-01", response_column: "Actual", denominator_column: "",
  columns: [column("glm_prediction")],
}};
if (lineBarCompatibleExpectedColumns([noDenominatorSource], {{
  numerator: "Actual", numeratorSource: "dataset", denominator: "Average row value", denominatorSource: "dataset",
}}).length !== 1) throw new Error("no-denominator Expected matching failed");
const otherColumns = [column("Expected Value"), {{ name: "Segment", kind: "categorical" }}, column("Actual")];
candidates = lineBarModelComparisonCandidates([sources[2]], kpi, {{ otherColumns }});
const otherCandidates = candidates.filter((candidate) => candidate.family === "other");
if (otherCandidates.map((candidate) => candidate.label).join("|") !== "Actual|Expected Value") {{
  throw new Error(`OTHER numeric columns were not sorted and filtered: ${{JSON.stringify(otherCandidates)}}`);
}}
if (otherCandidates[1].comparisonId !== `other:${{lineBarOtherColumnToken("Expected Value")}}`) {{
  throw new Error("OTHER column comparison id was not encoded");
}}
pair = lineBarDefaultModelComparison(candidates);
if (pair.baseline.predictionColumn !== "Actual" || pair.challenger.sourceId !== "gbm:current:predictions") {{
  throw new Error("single-model OTHER baseline default failed");
}}
pair = lineBarDefaultModelComparison(candidates, {{
  baselineSourceId: "dataset", baselineColumn: "Expected Value",
  challengerSourceId: "gbm:current:predictions",
}});
if (pair.baseline.predictionColumn !== "Expected Value" || pair.challenger.family !== "gbm") {{
  throw new Error("configured OTHER baseline was not preserved");
}}
pair = lineBarDefaultModelComparison(candidates, {{
  baselineSourceId: "gbm:current:predictions",
  challengerSourceId: "dataset", challengerColumn: "Expected Value",
}});
if (pair.baseline.family !== "gbm" || pair.challenger.predictionColumn !== "Expected Value") {{
  throw new Error("configured OTHER challenger was not preserved");
}}
"""
        self.run_node_script(script)

    def test_line_bar_model_kpi_compatibility_warnings_are_source_aware_and_deduplicated(self) -> None:
        module = Path("src/py_lucidum/static/app/line-bar-tool.js").resolve().as_uri()
        script = f"""
import {{ lineBarModelKpiCompatibilityWarnings }} from "{module}";
const glmA = {{
  id: "glm:glm-a:predictions", kind: "glm_predictions", model_id: "glm-a", active: true,
  response_column: "num_a", denominator_column: "den_a",
}};
const gbmB = {{
  id: "gbm:gbm-b:predictions", kind: "gbm_predictions", model_id: "gbm-b", active: true,
  response_column: "num_b", offset_column: "den_b",
}};
const gbmBShap = {{
  id: "gbm:gbm-b:shap-long", kind: "gbm_shap_long", model_id: "gbm-b", active: true,
  response_column: "num_b", offset_column: "den_b",
}};
const glmN = {{
  id: "glm:glm-n:predictions", kind: "glm_predictions", model_id: "glm-n", active: true,
  response_column: "num_n", denominator_column: "",
}};
const warnings = (request, dataSources = [glmA, gbmB, gbmBShap]) => (
  lineBarModelKpiCompatibilityWarnings({{ request, dataSources }})
);
const request = ({{
  actual = {{ numerator: "num_a" }},
  expected = [],
  denominator = "den_a",
  denominatorSource = "dataset",
  partialDependence = {{ mode: "none" }},
}} = {{}}) => ({{
  responses: [actual, ...expected], denominator, denominatorSource, partialDependence,
}});

let result = warnings(request({{
  expected: [{{ numerator: "glm_prediction", source: glmA.id }}],
  partialDependence: {{ mode: "glm" }},
}}));
if (result.length) throw new Error(`matching GLM produced a warning: ${{result.join(" ")}}`);

result = warnings(request({{
  actual: {{ numerator: "num_b" }},
  expected: [{{ numerator: "glm_prediction", source: glmA.id }}],
  partialDependence: {{ mode: "glm" }},
}}));
if (result.length !== 1) throw new Error("GLM prediction and overlay were not deduplicated");
if (!result[0].includes("GLM prediction and partial dependence were trained for num_a / den_a")) throw new Error("GLM components were not named");
if (!result[0].includes("selected KPI is num_b / den_a")) throw new Error("Numerator mismatch was not described");

result = warnings(request({{
  expected: [{{ numerator: "glm_prediction", source: glmA.id }}], denominator: "den_b",
}}));
if (result.length !== 1 || !result[0].includes("selected KPI is num_a / den_b")) throw new Error("Denominator mismatch was not described");

result = warnings(request({{
  actual: {{ numerator: "num_a", source: "external-source" }},
  expected: [{{ numerator: "glm_prediction", source: glmA.id }}],
}}));
if (result.length !== 1) throw new Error("Numerator source mismatch was not detected");

result = warnings(request({{
  actual: {{ numerator: "num_n" }},
  expected: [{{ numerator: "glm_prediction", source: glmN.id }}],
  denominator: "Average row value", denominatorSource: "external-source",
}}), [glmN]);
if (result.length) throw new Error("row-count denominator aliases did not normalize");

result = warnings(request({{
  actual: {{ numerator: "gbm_prediction_rate", source: gbmB.id }},
  expected: [{{ numerator: "glm_tabulated_prediction", source: glmA.id }}],
  denominator: "den_b",
}}));
if (result.length !== 2) throw new Error("Actual and Expected model outputs were not checked separately");
if (!result.some((warning) => warning.includes("GBM prediction rate"))) throw new Error("Actual prediction rate was not named");
if (!result.some((warning) => warning.includes("GLM tabulated prediction"))) throw new Error("tabulated Expected was not named");

result = warnings(request({{
  partialDependence: {{ mode: "shap", model_id: "gbm-b" }},
}}));
if (result.length !== 1 || !result[0].includes("GBM SHAP was trained for num_b / den_b")) throw new Error("SHAP mismatch was not detected");

result = warnings(request({{
  partialDependence: {{ mode: "both", gbm_model_id: "gbm-b", glm_model_id: "glm-a" }},
}}));
if (result.length !== 1 || !result[0].includes("GBM SHAP was trained for num_b / den_b")) throw new Error("family-specific Both models were not resolved");

result = warnings(request({{
  actual: {{ numerator: "num_b" }}, denominator: "den_b",
  partialDependence: {{ mode: "both", model_id: "gbm-b" }},
}}));
if (result.length !== 1 || !result[0].includes("GLM partial dependence was trained for num_a / den_a")) throw new Error("legacy Both model_id was not treated as GBM-only");

result = warnings(request({{
  expected: [{{ numerator: "glm_prediction", source: "glm:missing:predictions" }}],
  partialDependence: {{ mode: "shap", model_id: "missing" }},
}}), []);
if (result.length) throw new Error("missing metadata produced a compatibility warning");
"""
        self.run_node_script(script)

    def test_line_bar_expected_selection_helper_distinguishes_ordinary_and_additive_clicks(self) -> None:
        module = Path("src/py_lucidum/static/app/line-bar-tool.js").resolve().as_uri()
        script = f"""
import {{ lineBarAdditiveSelectionRequested, nextLineBarExpectedSelections }} from "{module}";
const a = {{ value: "a", sourceId: "dataset", metricKind: "metric" }};
const b = {{ value: "b", sourceId: "dataset", metricKind: "metric" }};
const c = {{ value: "c", sourceId: "dataset", metricKind: "metric" }};
const values = (selections) => selections.map((selection) => selection.value).join("|");
if (values(nextLineBarExpectedSelections([], a)) !== "a") throw new Error("ordinary initial selection failed");
if (values(nextLineBarExpectedSelections([a], b)) !== "b") throw new Error("ordinary replacement failed");
if (values(nextLineBarExpectedSelections([a, b], b)) !== "b") throw new Error("ordinary pair collapse failed");
const sole = [a];
if (nextLineBarExpectedSelections(sole, a) !== sole) throw new Error("ordinary active row should be unchanged");
if (nextLineBarExpectedSelections(sole, a, {{ additive: true }}) !== sole) throw new Error("additive sole removal should be rejected");
const pair = nextLineBarExpectedSelections(sole, b, {{ additive: true }});
if (values(pair) !== "a|b") throw new Error("additive second selection failed");
if (values(nextLineBarExpectedSelections(pair, a, {{ additive: true }})) !== "b") throw new Error("additive first removal failed");
if (nextLineBarExpectedSelections(pair, c, {{ additive: true }}) !== pair) throw new Error("additive third selection should be rejected");
if (values(nextLineBarExpectedSelections(pair, null)) !== "") throw new Error("explicit clear failed");
if (!lineBarAdditiveSelectionRequested({{ metaKey: true }}, "MacIntel")) throw new Error("macOS Command modifier failed");
if (lineBarAdditiveSelectionRequested({{ ctrlKey: true }}, "MacIntel")) throw new Error("macOS Ctrl should not be additive");
if (!lineBarAdditiveSelectionRequested({{ ctrlKey: true }}, "Linux x86_64")) throw new Error("Linux Ctrl modifier failed");
if (!lineBarAdditiveSelectionRequested({{ ctrlKey: true }}, "Win32")) throw new Error("Windows Ctrl modifier failed");
"""
        self.run_node_script(script)

    def test_two_feature_line_bar_chart_options_follow_feature_axis_ordering(self) -> None:
        module = Path("src/py_lucidum/static/app/line-bar-two-feature-chart.js").resolve().as_uri()
        script = f"""
import {{ fitTwoFeatureHeatmapAxes, twoFeatureChartOption }} from "{module}";
const metric = {{ key: "resp0", label: "Actual", format: (value) => String(value) }};
const rows = [
  {{ group0: "1", group0_sort: 1, group0_missing: false, group1: "A", group1_sort: "A", group1_missing: false, resp0: 10, volume: 1 }},
  {{ group0: "2", group0_sort: 2, group0_missing: false, group1: "A", group1_sort: "A", group1_missing: false, resp0: 20, volume: 2 }},
  {{ group0: "1", group0_sort: 1, group0_missing: false, group1: "B", group1_sort: "B", group1_missing: false, resp0: 30, volume: 3 }},
  {{ group0: "2", group0_sort: 2, group0_missing: false, group1: "B", group1_sort: "B", group1_missing: false, resp0: 40, volume: 4 }},
];
const factorGrouping = {{ feature: "Feature 2", kind: "categorical", continuous: false }};
const continuousGrouping = {{ feature: "Feature 1", kind: "numeric", continuous: true }};
const lines = twoFeatureChartOption({{
  plot_type: "lines",
  groupings: [continuousGrouping, factorGrouping],
  denominator: {{ bar_label: "Weight" }},
  rows,
}}, metric, {{
  xAxisLabels: ["1", "2"],
  xAxisLabelPolicy: {{
    show: true,
    interval: 0,
    showMinLabel: true,
    showMaxLabel: true,
    rotate: 65,
    fontSize: 8,
    nameGap: 58,
    bottom: 74,
    dataZoomEnabled: false,
    hideOverlap: false,
  }},
}});
if (lines.xAxis.name !== "Feature 1") throw new Error("continuous feature is not horizontal");
if (lines.xAxis.type !== "category" || lines.xAxis.data.join("|") !== "1|2") {{
  throw new Error("mixed chart must use the shared category labels");
}}
if (lines.xAxis.axisLabel.interval !== 0 || lines.xAxis.axisLabel.rotate !== 65
    || lines.xAxis.axisLabel.fontSize !== 8 || lines.grid.bottom !== 74) {{
  throw new Error("mixed chart did not apply the single-feature label policy");
}}
if (lines.series.map((series) => `${{series.name}}:${{series.type}}`).join("|") !== "A:line|A:bar|B:line|B:bar") {{
  throw new Error("paired factor series are incorrect");
}}
if (lines.series.filter((series) => series.type === "bar").some((series) => series.stack !== "two-feature-volume")) {{
  throw new Error("factor volume bars are not stacked");
}}
if (lines.series[0].data.join("|") !== "10|20" || lines.series[1].data.join("|") !== "1|2") {{
  throw new Error("mixed line/bar data are not aligned to the shared categories");
}}
if (lines.series[0].itemStyle.color !== lines.series[1].itemStyle.color
    || lines.series[2].itemStyle.color !== lines.series[3].itemStyle.color) {{
  throw new Error("line and bar colours do not match");
}}
if (lines.yAxis.length !== 2 || lines.yAxis[0].name !== "Actual" || lines.yAxis[1].name !== "Weight") {{
  throw new Error("mixed chart axes are incorrect");
}}
const modelDenominatorLines = twoFeatureChartOption({{
  plot_type: "lines",
  groupings: [continuousGrouping, factorGrouping],
  denominator: {{ bar_label: "glm_prediction" }},
  rows: rows.map((row, index) => ({{
    ...row,
    volume: [2545000, 896460, 1108800, 124128][index],
  }})),
}}, {{
  ...metric,
  axisLabel: "PREMIUM / glm_prediction",
}}, {{
  measureText: (value, fontSize = 12) => String(value).length * fontSize * 0.56,
}});
if (modelDenominatorLines.yAxis[0].name !== "PREMIUM / glm_prediction"
    || modelDenominatorLines.yAxis[1].name !== "glm_prediction") {{
  throw new Error("mixed chart did not use the numerator / denominator axis title");
}}
if (modelDenominatorLines.yAxis[1].nameGap <= 52
    || modelDenominatorLines.grid.right <= modelDenominatorLines.yAxis[1].nameGap) {{
  throw new Error("mixed chart did not reserve space between right-axis labels and title");
}}
if (lines.legend.data.map((item) => item.name).join("|") !== "A|B") {{
  throw new Error("mixed chart legend must contain one entry per factor group");
}}
const reversedRows = rows.map((row) => ({{
  ...row,
  group0: row.group1,
  group0_sort: row.group1_sort,
  group1: row.group0,
  group1_sort: row.group0_sort,
}}));
const reversedLines = twoFeatureChartOption({{
  plot_type: "lines",
  groupings: [factorGrouping, continuousGrouping],
  denominator: {{ bar_label: "N" }},
  rows: reversedRows,
}}, metric);
if (reversedLines.xAxis.name !== "Feature 1" || reversedLines.yAxis[1].name !== "N") {{
  throw new Error("reversed mixed chart ordering or row-count label is incorrect");
}}
const dateGrouping = {{
  feature: "QUOTE_DATE",
  kind: "date",
  date_bucket: "day",
  continuous: true,
}};
const dateRows = [
  {{ group0: "2024-01-08", group0_sort: "2024-01-08", group0_missing: false, group1: "A", group1_sort: "A", group1_missing: false, resp0: 20, volume: 2 }},
  {{ group0: "2024-01-01", group0_sort: "2024-01-01", group0_missing: false, group1: "A", group1_sort: "A", group1_missing: false, resp0: 10, volume: 1 }},
];
const dateLines = twoFeatureChartOption({{
  plot_type: "lines",
  groupings: [dateGrouping, factorGrouping],
  denominator: {{ bar_label: "N" }},
  rows: dateRows,
}}, metric, {{
  xAxisLabels: ["1 Jan 2024", "8 Jan 2024"],
}});
if (dateLines.xAxis.name !== "QUOTE_DATE"
    || dateLines.xAxis.data.join("|") !== "1 Jan 2024|8 Jan 2024"
    || dateLines.series[0].data.join("|") !== "10|20"
    || dateLines.series[1].data.join("|") !== "1|2") {{
  throw new Error("date mixed chart is not ordered chronologically");
}}
const heatmap = twoFeatureChartOption({{
  plot_type: "heatmap",
  groupings: [{{ ...continuousGrouping, continuous: false }}, factorGrouping],
  rows,
}}, metric);
if (heatmap.xAxis.name !== "Feature 2" || heatmap.yAxis.name !== "Feature 1") {{
  throw new Error("heatmap feature ordering is incorrect");
}}
const labelledHeatmap = twoFeatureChartOption({{
  plot_type: "heatmap",
  groupings: [
    {{ feature: "OVERNIGHT_LOCATION", kind: "categorical", continuous: false }},
    {{ feature: "DRIVER_AGE", kind: "quantile", continuous: false }},
  ],
  rows: [
    {{ group0: "Road", group0_sort: "Road", group1: "Q1", group1_sort: 1, resp0: 10, volume: 1 }},
    {{ group0: "Garage", group0_sort: "Garage", group1: "Q1", group1_sort: 1, resp0: 20, volume: 2 }},
    {{ group0: "Driveway", group0_sort: "Driveway", group1: "Q1", group1_sort: 1, resp0: 30, volume: 3 }},
  ],
}}, metric, {{
  chartWidth: 1200,
  chartHeight: 800,
  measureText: (value, fontSize = 12) => String(value).length * fontSize * (10 / 12),
  xAxisLabels: ["Q1"],
  heatmapLabelMode: "both",
  formatActual: (value) => `£${{Number(value).toFixed(2)}}`,
  formatWeight: (value) => `W${{value}}`,
  xAxisLabelPolicy: {{
    show: true,
    interval: 0,
    showMinLabel: true,
    showMaxLabel: true,
    rotate: 65,
    fontSize: 8,
    nameGap: 58,
    bottom: 74,
    dataZoomEnabled: false,
    hideOverlap: false,
  }},
}});
if (labelledHeatmap.yAxis.nameGap !== 108
    || labelledHeatmap.grid.left !== 128) {{
  throw new Error("heatmap y-axis spacing does not reserve the measured label width");
}}
if (labelledHeatmap.yAxis.axisLabel.rotate !== 0
    || labelledHeatmap.yAxis.axisLabel.align !== "right"
    || labelledHeatmap.yAxis.axisLabel.interval !== 0
    || labelledHeatmap.yAxis.axisLabel.hideOverlap !== false
    || labelledHeatmap.yAxis.axisLabel.fontSize !== 12
    || labelledHeatmap.yAxis.axisLabel.formatter("Driveway") !== "Driveway") {{
  throw new Error("heatmap y-axis labels must remain horizontal and right-aligned");
}}
if (labelledHeatmap.xAxis.axisLabel.interval !== 0
    || labelledHeatmap.xAxis.axisLabel.rotate !== 65
    || labelledHeatmap.grid.bottom !== 74) {{
  throw new Error("heatmap x-axis did not apply the shared label policy");
}}
if (!labelledHeatmap.series[0].label.show
    || labelledHeatmap.series[0].label.position !== "inside"
    || labelledHeatmap.series[0].label.align !== "center"
    || labelledHeatmap.series[0].label.verticalAlign !== "middle"
    || labelledHeatmap.series[0].label.fontSize !== 12
    || labelledHeatmap.series[0].label.textBorderWidth !== 0) {{
  throw new Error("heatmap labels are not centred, legible, and fit to the cells");
}}
const firstHeatmapValue = labelledHeatmap.series[0].data[0];
if (labelledHeatmap.series[0].label.formatter({{ value: firstHeatmapValue }})
    !== "{{heatmapWhitePrimary|£10.00}}\\n{{heatmapWhiteSecondary|W1}}") {{
  throw new Error("heatmap Both labels do not show Actual then Weight");
}}
const middleHeatmapValue = labelledHeatmap.series[0].data[1];
if (labelledHeatmap.series[0].label.formatter({{ value: middleHeatmapValue }})
    !== "{{heatmapDarkPrimary|£20.00}}\\n{{heatmapDarkSecondary|W2}}") {{
  throw new Error("heatmap labels do not adapt to the rendered cell contrast");
}}
const heatmapRich = labelledHeatmap.series[0].label.rich;
if (heatmapRich.heatmapWhitePrimary.color !== "#ffffff"
    || heatmapRich.heatmapWhitePrimary.fontWeight !== 600
    || heatmapRich.heatmapWhitePrimary.textShadowBlur !== 2
    || heatmapRich.heatmapWhiteSecondary.fontWeight !== 500
    || heatmapRich.heatmapDarkPrimary.color !== "#0f172a"
    || heatmapRich.heatmapDarkPrimary.fontWeight !== 600
    || heatmapRich.heatmapDarkPrimary.textShadowBlur !== 0
    || heatmapRich.heatmapDarkSecondary.fontWeight !== 500) {{
  throw new Error("heatmap adaptive label styles are incorrect");
}}
const labelData = {{
  plot_type: "heatmap",
  groupings: [
    {{ feature: "Y", kind: "categorical", continuous: false }},
    {{ feature: "X", kind: "categorical", continuous: false }},
  ],
  rows: [
    {{ group0: "A", group0_sort: "A", group1: "B", group1_sort: "B", resp0: 12.5, volume: 7 }},
  ],
}};
const labelOption = (mode, rows = labelData.rows, dimensions = {{}}) => twoFeatureChartOption(
  {{ ...labelData, rows }},
  metric,
  {{
    chartWidth: dimensions.width || 800,
    chartHeight: dimensions.height || 600,
    heatmapLabelMode: mode,
    formatActual: (value) => `A${{value}}`,
    formatWeight: (value) => `W${{value}}`,
  }},
);
const plainLabelText = (value) => String(value).split("\\n").map((line) => {{
  const delimiter = line.indexOf("|");
  return delimiter < 0 ? line : line.slice(delimiter + 1, -1);
}}).join("\\n");
const labelText = (option) => plainLabelText(
  option.series[0].label.formatter({{ value: option.series[0].data[0] }}),
);
const noCellLabels = labelOption("none");
const actualCellLabels = labelOption("actual");
const weightCellLabels = labelOption("weight");
const bothCellLabels = labelOption("both");
if (noCellLabels.series[0].label.show) throw new Error("heatmap labels should default off");
if (!actualCellLabels.series[0].label.show || labelText(actualCellLabels) !== "A12.5") {{
  throw new Error("heatmap Actual labels are incorrect");
}}
if (!weightCellLabels.series[0].label.show || labelText(weightCellLabels) !== "W7") {{
  throw new Error("heatmap Weight labels are incorrect");
}}
if (!bothCellLabels.series[0].label.show || labelText(bothCellLabels) !== "A12.5\\nW7") {{
  throw new Error("heatmap Both labels are incorrect");
}}
const missingActual = labelOption("both", [
  {{ group0: "A", group0_sort: "A", group1: "B", group1_sort: "B", resp0: null, volume: 7 }},
]);
if (labelText(missingActual) !== "\\nW7") throw new Error("missing Actual must render as a blank label row");
const fittedRows = Array.from({{ length: 100 }}, (_, index) => ({{
  group0: `Y${{Math.floor(index / 10)}}`,
  group0_sort: Math.floor(index / 10),
  group1: `X${{index % 10}}`,
  group1_sort: index % 10,
  resp0: 1234.56,
  volume: 9876,
}}));
const fittedLabels = labelOption("both", fittedRows, {{ width: 800, height: 400 }});
if (!fittedLabels.series[0].label.show
    || fittedLabels.series[0].label.fontSize < 7
    || fittedLabels.series[0].label.fontSize >= 12) {{
  throw new Error("heatmap label font size did not scale to the cell dimensions");
}}
const tooSmallLabels = labelOption("both", fittedRows, {{ width: 300, height: 200 }});
if (tooSmallLabels.series[0].label.show) throw new Error("unreadably small heatmap labels must be suppressed");
const denseRows = Array.from({{ length: 200 }}, (_, index) => ({{
  group0: `Y${{Math.floor(index / 20)}}`,
  group0_sort: Math.floor(index / 20),
  group1: `X${{index % 20}}`,
  group1_sort: index % 20,
  resp0: 1,
  volume: 1,
}}));
const denseLabels = labelOption("both", denseRows, {{ width: 2400, height: 1400 }});
if (!denseLabels.series[0].label.show) {{
  throw new Error("heatmap labels should remain available above 200 cells when they fit");
}}
const narrowHeatmap = twoFeatureChartOption({{
  plot_type: "heatmap",
  groupings: [
    {{ feature: "OVERNIGHT_LOCATION", kind: "categorical", continuous: false }},
    {{ feature: "DRIVER_AGE", kind: "quantile", continuous: false }},
  ],
  rows: [{{
    group0: "A very long factor label",
    group0_sort: "A very long factor label",
    group1: "Q1",
    group1_sort: 1,
    resp0: 10,
  }}],
}}, metric, {{
  chartWidth: 300,
  measureText: (value) => String(value).length * 10,
}});
if (narrowHeatmap.grid.left !== 120
    || !narrowHeatmap.yAxis.axisLabel.formatter("A very long factor label").endsWith("…")) {{
  throw new Error("heatmap y-axis spacing is not capped responsively");
}}
const manyYRows = Array.from({{ length: 60 }}, (_, index) => ({{
  group0: `Factor ${{index + 1}}`,
  group0_sort: index,
  group1: "X",
  group1_sort: "X",
  resp0: index,
}}));
const manyYHeatmap = twoFeatureChartOption({{
  plot_type: "heatmap",
  groupings: [
    {{ feature: "Y", kind: "categorical", continuous: false }},
    {{ feature: "X", kind: "categorical", continuous: false }},
  ],
  rows: manyYRows,
}}, metric, {{ chartHeight: 600 }});
if (manyYHeatmap.yAxis.axisLabel.rotate !== 0
    || manyYHeatmap.yAxis.axisLabel.align !== "right"
    || manyYHeatmap.yAxis.axisLabel.interval !== 0
    || manyYHeatmap.yAxis.axisLabel.hideOverlap !== false
    || manyYHeatmap.yAxis.axisLabel.show !== false
    || manyYHeatmap.yAxis.axisLabel.fontSize !== 8
    || manyYHeatmap.xAxis.axisLabel.show === false
    || manyYHeatmap.yAxis.name !== "Y") {{
  throw new Error("only the unreadable heatmap y-axis labels should be suppressed");
}}
const manyXRows = Array.from({{ length: 80 }}, (_, index) => ({{
  group0: "Y",
  group0_sort: "Y",
  group1: `Factor ${{index + 1}}`,
  group1_sort: index,
  resp0: index,
}}));
const manyXHeatmap = twoFeatureChartOption({{
  plot_type: "heatmap",
  groupings: [
    {{ feature: "Y", kind: "categorical", continuous: false }},
    {{ feature: "X", kind: "categorical", continuous: false }},
  ],
  rows: manyXRows,
}}, metric, {{ chartWidth: 800, chartHeight: 600 }});
if (manyXHeatmap.xAxis.axisLabel.show !== false
    || manyXHeatmap.yAxis.axisLabel.show === false
    || manyXHeatmap.xAxis.name !== "X") {{
  throw new Error("only the unreadable heatmap x-axis labels should be suppressed");
}}
const zoomXAxisPolicy = {{
  show: false,
  interval: 0,
  rotate: 0,
  fontSize: 10,
  nameGap: 22,
  bottom: 74,
  dataZoomEnabled: true,
  hideOverlap: false,
}};
const xAxisLabels = manyXRows.map((row) => row.group1);
const fullHeatmapAxes = fitTwoFeatureHeatmapAxes(xAxisLabels, ["Y"], {{
  chartWidth: 800,
  chartHeight: 600,
  xAxisLabelPolicy: zoomXAxisPolicy,
}});
const zoomedHeatmapAxes = fitTwoFeatureHeatmapAxes(xAxisLabels, ["Y"], {{
  chartWidth: 800,
  chartHeight: 600,
  xAxisLabelPolicy: zoomXAxisPolicy,
  xAxisVisibleRange: {{ startIndex: 0, endIndex: 9 }},
}});
if (fullHeatmapAxes.xAxisLabelPolicy.show !== false
    || zoomedHeatmapAxes.xAxisLabelPolicy.show !== true
    || zoomedHeatmapAxes.xAxisLabelPolicy.fontSize < 8) {{
  throw new Error("heatmap x-axis labels should reappear after sufficient zoom");
}}
const shortHeatmapAxes = fitTwoFeatureHeatmapAxes(["X"], manyYRows.map((row) => row.group0), {{
  chartWidth: 800,
  chartHeight: 600,
}});
const tallHeatmapAxes = fitTwoFeatureHeatmapAxes(["X"], manyYRows.map((row) => row.group0), {{
  chartWidth: 800,
  chartHeight: 1200,
}});
if (shortHeatmapAxes.yAxisLayout.show !== false
    || tallHeatmapAxes.yAxisLayout.show !== true
    || tallHeatmapAxes.yAxisLayout.fontSize < 8) {{
  throw new Error("heatmap y-axis labels should reappear after sufficient resize");
}}
const maximumHeatmapRows = Array.from({{ length: 100000 }}, (_, index) => ({{
  group0: `Y${{Math.floor(index / 400)}}`,
  group0_sort: Math.floor(index / 400),
  group1: `X${{index % 400}}`,
  group1_sort: index % 400,
  resp0: index,
  volume: 1,
}}));
const maximumHeatmap = twoFeatureChartOption({{
  plot_type: "heatmap",
  groupings: [
    {{ feature: "Y", kind: "categorical", continuous: false }},
    {{ feature: "X", kind: "categorical", continuous: false }},
  ],
  rows: maximumHeatmapRows,
}}, metric, {{ chartWidth: 1200, chartHeight: 800 }});
if (maximumHeatmap.series[0].data.length !== 100000
    || maximumHeatmap.visualMap.min !== 0
    || maximumHeatmap.visualMap.max !== 99999) {{
  throw new Error("100,000-cell heatmap option construction failed");
}}
const surfaceRows = rows.map((row) => ({{
  ...row,
  group1: String(row.group1 === "A" ? 10 : 20),
  group1_sort: row.group1 === "A" ? 10 : 20,
}}));
const surface = twoFeatureChartOption({{
  plot_type: "surface",
  groupings: [continuousGrouping, {{ feature: "Feature 2", kind: "numeric", continuous: true }}],
  rows: surfaceRows,
}}, metric);
if (surface.xAxis3D.name !== "Feature 2" || surface.yAxis3D.name !== "Feature 1") {{
  throw new Error("surface feature ordering is incorrect");
}}
if (surface.xAxis3D.min !== 10 || surface.xAxis3D.max !== 20
    || surface.yAxis3D.min !== 1 || surface.yAxis3D.max !== 2) {{
  throw new Error("surface axes do not use the exact plotted domains");
}}
if (surface.grid3D.boxWidth !== 100 || surface.grid3D.boxDepth !== 74) {{
  throw new Error("default surface footprint changed unexpectedly");
}}
const dateSurfaceRows = [
  {{ group0: "1", group0_sort: 1, group0_missing: false, group1: "2024-01-01", group1_sort: "2024-01-01", group1_missing: false, resp0: 10 }},
  {{ group0: "2", group0_sort: 2, group0_missing: false, group1: "2024-01-01", group1_sort: "2024-01-01", group1_missing: false, resp0: 20 }},
  {{ group0: "1", group0_sort: 1, group0_missing: false, group1: "2024-01-08", group1_sort: "2024-01-08", group1_missing: false, resp0: 30 }},
  {{ group0: "2", group0_sort: 2, group0_missing: false, group1: "2024-01-08", group1_sort: "2024-01-08", group1_missing: false, resp0: 40 }},
];
const dateSurface = twoFeatureChartOption({{
  plot_type: "surface",
  groupings: [continuousGrouping, dateGrouping],
  rows: dateSurfaceRows,
}}, metric, {{
  formatDateGroupValue: (value) => `D:${{new Date(value).toISOString().slice(0, 10)}}`,
}});
if (dateSurface.xAxis3D.type !== "time"
    || dateSurface.yAxis3D.type !== "value"
    || dateSurface.xAxis3D.min !== Date.UTC(2024, 0, 1)
    || dateSurface.xAxis3D.max !== Date.UTC(2024, 0, 8)
    || dateSurface.xAxis3D.axisLabel.formatter(Date.UTC(2024, 0, 1)) !== "D:2024-01-01") {{
  throw new Error("date surface axis type, bounds, or formatting is incorrect");
}}
const dateSurfaceTooltip = dateSurface.tooltip.formatter({{
  value: dateSurface.series[0].data[0],
}});
if (!dateSurfaceTooltip.includes("QUOTE_DATE: D:2024-01-01")) {{
  throw new Error("date surface tooltip is not date-aware");
}}
const dualDateSurface = twoFeatureChartOption({{
  plot_type: "surface",
  groupings: [{{ ...dateGrouping, feature: "START_DATE" }}, dateGrouping],
  rows: [
    {{ group0_sort: "2023-01-01", group0_missing: false, group1_sort: "2024-01-01", group1_missing: false, resp0: 1 }},
    {{ group0_sort: "2023-01-08", group0_missing: false, group1_sort: "2024-01-01", group1_missing: false, resp0: 2 }},
    {{ group0_sort: "2023-01-01", group0_missing: false, group1_sort: "2024-01-08", group1_missing: false, resp0: 3 }},
    {{ group0_sort: "2023-01-08", group0_missing: false, group1_sort: "2024-01-08", group1_missing: false, resp0: 4 }},
  ],
}}, metric, {{
  formatDateGroupValue: (value) => new Date(value).toISOString().slice(0, 10),
}});
if (dualDateSurface.xAxis3D.type !== "time" || dualDateSurface.yAxis3D.type !== "time") {{
  throw new Error("date-by-date surface must use two time axes");
}}
const wideSurface = twoFeatureChartOption({{
  plot_type: "surface",
  groupings: [continuousGrouping, {{ feature: "Feature 2", kind: "numeric", continuous: true }}],
  rows: surfaceRows,
}}, metric, {{ chartWidth: 1920, chartHeight: 1000 }});
if (wideSurface.grid3D.boxWidth !== 140 || wideSurface.grid3D.boxDepth !== 92) {{
  throw new Error("surface footprint did not expand for a wide chart");
}}
const shortSurface = twoFeatureChartOption({{
  plot_type: "surface",
  groupings: [continuousGrouping, {{ feature: "Feature 2", kind: "numeric", continuous: true }}],
  rows: surfaceRows,
}}, metric, {{ chartWidth: 1920, chartHeight: 500 }});
if (shortSurface.grid3D.boxWidth !== 100 || shortSurface.grid3D.boxDepth !== 74) {{
  throw new Error("surface footprint must remain contained in a short chart");
}}
if (surface.series[0].dataShape.join("|") !== "2|2") throw new Error("surface grid is not dense");
const sparseSurface = twoFeatureChartOption({{
  plot_type: "surface",
  groupings: [continuousGrouping, {{ feature: "Feature 2", kind: "numeric", continuous: true }}],
  rows: surfaceRows.slice(0, 3),
}}, metric);
if (!sparseSurface.series[0].data.some((point) => Number.isNaN(point[2]))) {{
  throw new Error("missing surface cells must use NaN");
}}
"""
        self.run_node_script(script)

    def test_saved_filter_expression_helpers_preserve_boolean_precedence(self) -> None:
        module = Path("src/py_lucidum/static/app/shared/filter-expression.js").resolve().as_uri()
        script = f"""
import {{
  combineGroupedFilterRows,
  combineLegacyGroupedFilterRows,
  combineSavedFilterRows,
  filterExpressionHasTopLevelOr,
  normaliseLegacyGroupedFilter,
}} from "{module}";

const row = (theme, expression) => ({{ theme, expression }});
const expect = (actual, expected, label) => {{
  if (actual !== expected) throw new Error(`${{label}}: ${{JSON.stringify(actual)}} !== ${{JSON.stringify(expected)}}`);
}};
const onePerGroup = [row("A", "A = 1"), row("B", "B = 2")];
const groupedRows = [row("A", "A = 1"), row("A", "A = 2"), row("B", "B = 3")];

expect(combineGroupedFilterRows([]), "", "empty grouped filter");
expect(combineGroupedFilterRows([row("A", "A = 1")]), "A = 1", "single grouped filter");
expect(combineGroupedFilterRows(onePerGroup), "A = 1 AND B = 2", "one row per group");
expect(combineGroupedFilterRows(groupedRows), "(A = 1 OR A = 2) AND B = 3", "multiple rows in a group");
expect(
  combineGroupedFilterRows([row("A", "A = 1"), row("A", "A = 2")]),
  "A = 1 OR A = 2",
  "single OR group",
);
expect(
  combineGroupedFilterRows([row("A", "A = 1 AND B = 2"), row("B", "C = 3")]),
  "A = 1 AND B = 2 AND C = 3",
  "compound AND row",
);
expect(
  combineGroupedFilterRows([row("A", "A = 1 OR B = 2"), row("B", "C = 3")]),
  "(A = 1 OR B = 2) AND C = 3",
  "top-level OR row",
);
expect(
  combineGroupedFilterRows([row("A", "(A = 1 OR B = 2)"), row("B", "C = 3")]),
  "(A = 1 OR B = 2) AND C = 3",
  "already grouped OR row",
);
expect(
  combineGroupedFilterRows([row("A", "A = 1 AND (B = 2 OR C = 3)"), row("B", "D = 4")]),
  "A = 1 AND (B = 2 OR C = 3) AND D = 4",
  "nested OR row",
);
expect(
  combineGroupedFilterRows([row("A", "CODE = 'A OR B'"), row("B", "C = 3")]),
  "CODE = 'A OR B' AND C = 3",
  "quoted OR text",
);
if (!filterExpressionHasTopLevelOr("A = 1 OR B = 2")) throw new Error("top-level OR was not detected");
if (filterExpressionHasTopLevelOr("(A = 1 OR B = 2)")) throw new Error("nested OR was detected");
if (filterExpressionHasTopLevelOr('"OR" = 1')) throw new Error("quoted identifier OR was detected");
if (filterExpressionHasTopLevelOr("A = $$ OR $$")) throw new Error("dollar-quoted OR was detected");
if (!filterExpressionHasTopLevelOr("A = 'text\\\\' OR B = 2")) throw new Error("OR after a standard string was not detected");
if (filterExpressionHasTopLevelOr("A = E'text\\\\' OR value'")) throw new Error("OR in an escaped string was detected");

expect(combineSavedFilterRows([row("A", "A = 1")], {{ mode: "single" }}), "A = 1", "single filter");
expect(combineSavedFilterRows(onePerGroup, {{ mode: "multi", operator: "and" }}), "(A = 1) AND (B = 2)", "multi all");
expect(combineSavedFilterRows(onePerGroup, {{ mode: "multi", operator: "or" }}), "(A = 1) OR (B = 2)", "multi any");
expect(combineSavedFilterRows(onePerGroup, {{ mode: "multi", operator: "nand" }}), "NOT ((A = 1) AND (B = 2))", "multi not all");
expect(combineSavedFilterRows(onePerGroup, {{ mode: "multi", operator: "nor" }}), "NOT ((A = 1) OR (B = 2))", "multi none");

const legacy = "(A = 1) AND (B = 2)";
expect(combineLegacyGroupedFilterRows(onePerGroup), legacy, "legacy grouped filter");
expect(normaliseLegacyGroupedFilter(legacy, onePerGroup), "A = 1 AND B = 2", "legacy normalization");
const legacyOrGroup = "((A = 1) OR (A = 2)) AND (B = 3)";
expect(combineLegacyGroupedFilterRows(groupedRows), legacyOrGroup, "legacy grouped OR filter");
expect(
  normaliseLegacyGroupedFilter(legacyOrGroup, groupedRows),
  "(A = 1 OR A = 2) AND B = 3",
  "legacy grouped OR normalization",
);
expect(normaliseLegacyGroupedFilter("A = 1 AND B = 2", onePerGroup), "A = 1 AND B = 2", "current filter preservation");
expect(normaliseLegacyGroupedFilter("A = 1", onePerGroup), "A = 1", "mismatched filter preservation");
expect(
  normaliseLegacyGroupedFilter(legacy, onePerGroup, {{ allRowsRestored: false }}),
  legacy,
  "missing-row preservation",
);
"""
        self.run_node_script(script)

    def test_vertical_list_navigation_handles_rerenders_and_focus_ownership(self) -> None:
        module = Path("src/py_lucidum/static/app/shared/list-navigation.js").resolve().as_uri()
        script = f"""
import {{ bindVerticalListNavigation }} from "{module}";

const documentNode = {{ body: {{}}, documentElement: {{}}, activeElement: null }};
const listeners = new Map();
const classNames = new Set();
const list = {{
  ownerDocument: documentNode,
  rows: [],
  classList: {{
    add(name) {{ classNames.add(name); }},
    remove(name) {{ classNames.delete(name); }},
    contains(name) {{ return classNames.has(name); }},
  }},
  addEventListener(type, handler) {{ listeners.set(type, handler); }},
  removeEventListener(type, handler) {{ if (listeners.get(type) === handler) listeners.delete(type); }},
  querySelectorAll(selector) {{
    if (selector !== ".row") throw new Error(`unexpected selector ${{selector}}`);
    return this.rows;
  }},
  contains(item) {{ return this.rows.includes(item); }},
}};

function row(key) {{
  return {{
    dataset: {{ key }},
    disabled: false,
    offsetParent: {{}},
    scrollCount: 0,
    closest(selector) {{ return selector === ".row" ? this : null; }},
    focus() {{ documentNode.activeElement = this; }},
    scrollIntoView() {{ this.scrollCount += 1; }},
  }};
}}

const activations = [];
let releaseActivation = null;
list.rows = [row("a"), row("b"), row("c")];
const navigation = bindVerticalListNavigation({{
  list,
  itemSelector: ".row",
  getItemKey: (item) => item.dataset.key,
  onActivate: async (key) => {{
    activations.push(key);
    if (key === "b") {{
      list.rows = [row("a"), row("b"), row("c")];
      documentNode.activeElement = documentNode.body;
    }}
    if (key === "c") await new Promise((resolve) => {{ releaseActivation = resolve; }});
  }},
}});

function keyEvent(target, key) {{
  return {{
    key,
    target,
    defaultPrevented: false,
    preventDefault() {{ this.defaultPrevented = true; }},
  }};
}}

documentNode.activeElement = documentNode.body;
const clickedRow = list.rows[0];
list.classList.add("list-keyboard-navigation");
await listeners.get("click")({{ target: clickedRow }});
if (documentNode.activeElement !== clickedRow) throw new Error("click should explicitly focus its row");
if (activations.join(",") !== "a") throw new Error("click should activate a");
if (clickedRow.scrollCount !== 1) throw new Error("clicked row was not revealed");
if (list.classList.contains("list-keyboard-navigation")) throw new Error("click should restore pointer hover mode");
activations.length = 0;

const topBoundary = keyEvent(list.rows[0], "ArrowUp");
await listeners.get("keydown")(topBoundary);
if (!topBoundary.defaultPrevented) throw new Error("top boundary should prevent scrolling");
if (activations.length) throw new Error("top boundary should not activate an item");
if (!list.classList.contains("list-keyboard-navigation")) throw new Error("top boundary should enable keyboard mode");
listeners.get("pointermove")();
if (list.classList.contains("list-keyboard-navigation")) throw new Error("pointer movement should restore hover mode after a boundary press");

const down = keyEvent(list.rows[0], "ArrowDown");
await listeners.get("keydown")(down);
if (!down.defaultPrevented) throw new Error("ArrowDown should prevent scrolling");
if (activations.join(",") !== "b") throw new Error("ArrowDown should activate b");
if (documentNode.activeElement !== list.rows[1]) throw new Error("focus was not restored after rerender");
if (list.rows[1].scrollCount !== 1) throw new Error("restored item was not revealed");
if (!list.classList.contains("list-keyboard-navigation")) throw new Error("ArrowDown should enable keyboard mode");
listeners.get("pointermove")();
if (list.classList.contains("list-keyboard-navigation")) throw new Error("pointer movement should restore hover mode");

const bottomBoundary = keyEvent(list.rows[2], "ArrowDown");
await listeners.get("keydown")(bottomBoundary);
if (!bottomBoundary.defaultPrevented) throw new Error("bottom boundary should prevent scrolling");
if (activations.join(",") !== "b") throw new Error("bottom boundary should not activate an item");
if (!list.classList.contains("list-keyboard-navigation")) throw new Error("bottom boundary should enable keyboard mode");
listeners.get("pointermove")();
if (list.classList.contains("list-keyboard-navigation")) throw new Error("pointer movement should restore hover mode after the bottom boundary");

const pendingDown = listeners.get("keydown")(keyEvent(list.rows[1], "ArrowDown"));
const outside = {{}};
documentNode.activeElement = outside;
listeners.get("focusout")({{ relatedTarget: outside }});
releaseActivation();
await pendingDown;
if (documentNode.activeElement !== outside) throw new Error("navigation stole deliberately moved focus");

navigation.destroy();
if (list.classList.contains("list-keyboard-navigation")) throw new Error("destroy should clear keyboard mode");
if (listeners.has("click") || listeners.has("keydown") || listeners.has("focusout") || listeners.has("pointermove")) throw new Error("destroy did not remove listeners");
"""
        self.run_node_script(script)



    def test_package_version_matches_pyproject(self) -> None:
        pyproject = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))

        self.assertEqual(__version__, pyproject["project"]["version"])

    def test_schema_includes_app_version(self) -> None:
        status, _, body = asgi_get(self.app, "/api/schema")
        schema = json.loads(body)

        self.assertEqual(status, 200)
        self.assertEqual(schema["app_version"], __version__)
        self.assertEqual(schema["header_buttons"], False)
        self.assertEqual(schema["title_prefix"], "")

        app = create_app(self.data_path, header_buttons=True, title_prefix=" Portfolio Demo ")
        status, _, body = asgi_get(app, "/api/schema")
        schema = json.loads(body)

        self.assertEqual(status, 200)
        self.assertEqual(schema["header_buttons"], True)
        self.assertEqual(schema["title_prefix"], "Portfolio Demo")

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
    {{ id: "gbm:one:predictions", kind: "gbm_predictions", active: true, columns: [{{ name: "gbm_prediction", kind: "numeric" }}, {{ name: "gbm_prediction_rate", kind: "numeric" }}, {{ name: "gbm_tabulated_prediction", kind: "numeric" }}] }},
    {{ id: "glm:one:predictions", kind: "glm_predictions", active: true, columns: [{{ name: "glm_prediction", kind: "numeric" }}, {{ name: "glm_prediction_rate", kind: "numeric" }}, {{ name: "glm_tabulated_prediction", kind: "numeric" }}] }},
  ],
}};
if (dataSourceForId(schema, "dataset").id !== "dataset") throw new Error("dataSourceForId failed");
if (!dataSourceHasColumn(schema, "gbm:one:predictions", "gbm_prediction")) throw new Error("dataSourceHasColumn failed");
if (sourceColumns(schema, "dataset").length !== 1) throw new Error("sourceColumns failed");
if (!toolEnabled(schema, "line_bar")) throw new Error("toolEnabled failed");
if (!isModelTool("gbm") || !isModelTool("glm") || isModelTool("line_bar")) throw new Error("isModelTool failed");
if (!isModelPredictionColumn({{ name: "gbm_prediction" }}) || !isModelPredictionColumn({{ name: "gbm_prediction_rate" }}) || !isModelPredictionColumn({{ name: "gbm_tabulated_prediction" }}) || !isModelPredictionColumn({{ name: "glm_prediction" }}) || !isModelPredictionColumn({{ name: "glm_prediction_rate" }}) || !isModelPredictionColumn({{ name: "glm_tabulated_prediction" }})) throw new Error("isModelPredictionColumn failed");
if (preferredStartupSource(schema.data_sources, "gbm:one:predictions") !== "gbm:one:predictions") throw new Error("preferredStartupSource explicit source failed");
if (preferredStartupSource(schema.data_sources, "missing") !== "dataset") throw new Error("preferredStartupSource should fall back to dataset");
schema.data_sources[1].active = false;
if (preferredStartupSource(schema.data_sources, "") !== "dataset") throw new Error("preferredStartupSource should ignore active models by default");
"""
        self.run_node_script(script)




    def test_uk_map_shapefile_match_summary_helper(self) -> None:
        module = Path("src/py_lucidum/static/app/uk-map-tool.js").resolve().as_uri()
        script = f"""
import {{ ukMapShapefileMatchSummary }} from "{module}";

const allMatched = ukMapShapefileMatchSummary({{
  level: "area",
  rows: [
    {{ key: "AB", row_count: 2, value: 10 }},
    {{ key: "AL", row_count: 1, value: null }},
  ],
  filteredRowCount: 3,
  shapeKeys: new Set(["AB", "AL"]),
}});
if (allMatched.matchedRowCount !== 3
    || allMatched.unmatchedRowCount !== 0
    || allMatched.eligibleRowCount !== 3
    || allMatched.missingRowCount !== 0
    || allMatched.unmatchedPercentageText !== "0.0"
    || allMatched.missingPercentageText !== "0.0"
    || allMatched.matchText !== "All areas matched"
    || allMatched.missingText !== "no rows missing area"
    || allMatched.matchState !== "complete"
    || allMatched.matchedRows.length !== 2
    || allMatched.unmatchedRows.length !== 0) {{
  throw new Error(`incorrect all-matched summary: ${{JSON.stringify(allMatched)}}`);
}}

const partial = ukMapShapefileMatchSummary({{
  level: "area",
  rows: [
    {{ key: "AB", row_count: 2, value: 10 }},
    {{ key: "ZZ", row_count: 1, value: 999999999 }},
  ],
  filteredRowCount: 4,
  shapeKeys: ["AB", "AL"],
}});
if (partial.matchedRowCount !== 2
    || partial.unmatchedRowCount !== 1
    || partial.eligibleRowCount !== 3
    || partial.missingRowCount !== 1
    || partial.unmatchedPercentageText !== "33.3"
    || partial.matchText !== "1 row unmatched (33.3%)"
    || partial.missingPercentageText !== "25.0"
    || partial.missingText !== "1 row missing area (25.0%)"
    || partial.matchState !== "warning"
    || partial.matchedRows.map((row) => row.key).join("|") !== "AB"
    || partial.unmatchedRows.map((row) => row.key).join("|") !== "ZZ") {{
  throw new Error(`incorrect partial summary: ${{JSON.stringify(partial)}}`);
}}

const plural = ukMapShapefileMatchSummary({{
  level: "sector",
  rows: [
    {{ key: "AB10 1", row_count: 2 }},
    {{ key: "ZZ1 1", row_count: 2 }},
  ],
  filteredRowCount: 6,
  shapeKeys: new Set(["AB10 1"]),
}});
if (plural.matchText !== "2 rows unmatched (50.0%)"
    || plural.missingPercentageText !== "33.3"
    || plural.missingText !== "2 rows missing sector (33.3%)") {{
  throw new Error(`incorrect plural summary: ${{JSON.stringify(plural)}}`);
}}

const noEligibleRows = ukMapShapefileMatchSummary({{
  level: "area",
  rows: [],
  filteredRowCount: 2,
  shapeKeys: new Set(["AB"]),
}});
if (noEligibleRows.matchText !== "no areas to match"
    || noEligibleRows.missingPercentageText !== "100.0"
    || noEligibleRows.missingText !== "2 rows missing area (100.0%)"
    || noEligibleRows.unmatchedPercentageText !== "0.0") {{
  throw new Error(`incorrect empty summary: ${{JSON.stringify(noEligibleRows)}}`);
}}

const smoothedSector = ukMapShapefileMatchSummary({{
  level: "sector",
  rows: [
    {{ key: "AB10 1", row_count: 1, raw_row_count: 1, value: 10 }},
    {{ key: "AB10 2", row_count: 0, raw_row_count: null, value: 10 }},
    {{ key: "ZZ1 1", row_count: 3, raw_row_count: 3, value: 100 }},
  ],
  filteredRowCount: 4,
  shapeKeys: new Set(["AB10 1", "AB10 2"]),
}});
if (smoothedSector.matchedRowCount !== 1
    || smoothedSector.unmatchedRowCount !== 3
    || smoothedSector.eligibleRowCount !== 4
    || smoothedSector.missingRowCount !== 0
    || smoothedSector.unmatchedPercentageText !== "75.0"
    || smoothedSector.missingPercentageText !== "0.0"
    || smoothedSector.missingText !== "no rows missing sector"
    || smoothedSector.matchedRows.length !== 2) {{
  throw new Error(`incorrect smoothed-sector summary: ${{JSON.stringify(smoothedSector)}}`);
}}
"""
        self.run_node_script(script)


    def test_uk_map_postcode_availability_helper(self) -> None:
        module = Path("src/py_lucidum/static/app/uk-map-tool.js").resolve().as_uri()
        area_geojson = Path(
            "src/py_lucidum/tools/uk_map/static/geodata/areas_MappaR.geojson"
        ).resolve().as_uri()
        script = f"""
import {{ readFileSync }} from "node:fs";
import {{
  UK_MAP_POSTCODE_REGIONS,
  combineUkMapPostcodeFilter,
  ukMapPopupContentHtml,
  ukMapPostcodeAvailability,
  ukMapPostcodeFilterClause,
  ukMapPostcodeInFilterClause,
}} from "{module}";
const expectedRegionLabels = [
  "Central London",
  "East Midlands",
  "East of England",
  "North East",
  "North West",
  "Northern Ireland",
  "Outer London",
  "Scotland",
  "South East",
  "South West",
  "Wales",
  "West Midlands",
  "Yorkshire and The Humber",
];
if (UK_MAP_POSTCODE_REGIONS.map((region) => region.label).join("|") !== expectedRegionLabels.join("|")) {{
  throw new Error("postcode regions should retain their requested order");
}}
const mappedAreaCodes = UK_MAP_POSTCODE_REGIONS.flatMap((region) => region.areas);
const geoJsonAreaCodes = JSON.parse(readFileSync(new URL("{area_geojson}"), "utf8"))
  .features.map((feature) => String(feature.properties.PostcodeArea));
if (mappedAreaCodes.length !== 124
    || new Set(mappedAreaCodes).size !== mappedAreaCodes.length
    || [...mappedAreaCodes].sort().join("|") !== [...geoJsonAreaCodes].sort().join("|")) {{
  throw new Error("postcode region mapping should cover every bundled area exactly once");
}}
if (!Object.isFrozen(UK_MAP_POSTCODE_REGIONS)
    || UK_MAP_POSTCODE_REGIONS.some((region) => !Object.isFrozen(region) || !Object.isFrozen(region.areas))) {{
  throw new Error("postcode region mapping should be immutable");
}}
const labels = (schema, query = "") => ukMapPostcodeAvailability({{
  schema,
  locationParams: new URLSearchParams(query),
}}).levels.map((level) => level.label).join("/");
const noPostcodes = {{ columns: [{{ name: "Actual", kind: "numeric" }}], defaults: {{}} }};
if (labels(noPostcodes) !== "") throw new Error("expected no postcode levels");
const areaOnly = {{ columns: [{{ name: "PostcodeArea", kind: "string" }}], defaults: {{}} }};
if (labels(areaOnly) !== "Area") throw new Error("expected area only");
const areaSector = {{ columns: [{{ name: "PostcodeArea", kind: "string" }}, {{ name: "PostcodeSector", kind: "string" }}], defaults: {{}} }};
if (labels(areaSector) !== "Area/Sector") throw new Error("expected area and sector");
const unitWithoutCoordinates = {{
  columns: [
    {{ name: "PostcodeArea", kind: "string" }},
    {{ name: "PostcodeSector", kind: "string" }},
    {{ name: "PostcodeUnit", kind: "string" }},
  ],
  defaults: {{}},
}};
if (labels(unitWithoutCoordinates) !== "Area/Sector") throw new Error("unit should require coordinates");
const unitWithCoordinates = {{
  columns: [
    {{ name: "PostcodeArea", kind: "string" }},
    {{ name: "PostcodeSector", kind: "string" }},
    {{ name: "PostcodeUnit", kind: "string" }},
    {{ name: "lat", kind: "numeric" }},
    {{ name: "long", kind: "numeric" }},
  ],
  defaults: {{}},
}};
if (labels(unitWithCoordinates) !== "Area/Sector/Unit") throw new Error("expected all postcode levels");
const customDefaults = {{
  columns: [
    {{ name: "Area", kind: "string" }},
    {{ name: "Sector", kind: "string" }},
    {{ name: "Unit", kind: "string" }},
    {{ name: "LatCol", kind: "numeric" }},
    {{ name: "LongCol", kind: "numeric" }},
  ],
  defaults: {{
    postcode_area: "Area",
    postcode_sector: "Sector",
    postcode_unit: "Unit",
    latitude: "LatCol",
    longitude: "LongCol",
  }},
}};
if (labels(customDefaults) !== "Area/Sector/Unit") throw new Error("custom defaults should resolve");
if (labels(customDefaults, "postcode_unit=Missing") !== "Area/Sector") throw new Error("invalid explicit unit should hide unit");
const postcodeClause = ukMapPostcodeFilterClause('Post"codeSector', "CA'10 3");
if (postcodeClause !== `"Post""codeSector" = 'CA''10 3'`) throw new Error(`unsafe postcode clause: ${{postcodeClause}}`);
const postcodeGroupClause = ukMapPostcodeInFilterClause('Post"codeArea', ["SE", "CA", "SE", "O'X"]);
if (postcodeGroupClause !== `"Post""codeArea" IN ('CA', 'O''X', 'SE')`) {{
  throw new Error(`unsafe or unstable postcode group clause: ${{postcodeGroupClause}}`);
}}
if (ukMapPostcodeInFilterClause("PostcodeArea", []) !== "") {{
  throw new Error("empty postcode group should not emit a clause");
}}
const combinedFilter = combineUkMapPostcodeFilter("price >= 100", postcodeClause);
if (combinedFilter !== `(price >= 100) AND ("Post""codeSector" = 'CA''10 3')`) {{
  throw new Error(`incorrect combined postcode filter: ${{combinedFilter}}`);
}}
const popupOptions = {{
  title: "CA10 3",
  escapeHtml: (value) => String(value).replace(/[&<>"]/g, (char) => ({{ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }})[char]),
  formatNumber: (value) => Number(value).toLocaleString("en-GB"),
  formatLineValue: (value) => `£${{Number(value).toLocaleString("en-GB")}}`,
}};
const popupText = (html) => html.replace(/<[^>]+>/g, "");
const averagePopup = ukMapPopupContentHtml({{
  ...popupOptions,
  row: {{ value: 505, numerator: 361075, denominator: 715, row_count: 715 }},
  data: {{
    level: "sector",
    response: {{ label: "MARKET_PRICE_1_5" }},
    denominator: {{ column: null, bar_label: "Average row value" }},
  }},
}});
const averagePopupText = popupText(averagePopup);
const averageLabels = ["Average MARKET_PRICE_1_5: £505", "Rows: 715"];
if (!averageLabels.every((label) => averagePopupText.includes(label))
    || averagePopupText.indexOf(averageLabels[0]) > averagePopupText.indexOf(averageLabels[1])
    || averagePopupText.includes("Records:")
    || (averagePopup.match(/class="map-popup-quantity"/g) || []).length !== 2
    || averagePopup.includes("Row count")) {{
  throw new Error(`incorrect average popup: ${{averagePopup}}`);
}}
const selectedAreaPopup = ukMapPopupContentHtml({{
  ...popupOptions,
  title: "CA",
  row: {{ value: 505, row_count: 715 }},
  data: {{
    level: "area",
    response: {{ label: "MARKET_PRICE_1_5" }},
    denominator: {{ column: null }},
  }},
  areaFilterToggle: true,
  areaFilterSelected: true,
}});
if (!selectedAreaPopup.includes('aria-pressed="true"')
    || !selectedAreaPopup.includes("Remove CA from postcode area filter")
    || !selectedAreaPopup.includes("map-popup-action--active")) {{
  throw new Error(`incorrect selected area popup: ${{selectedAreaPopup}}`);
}}
const denominatorPopup = ukMapPopupContentHtml({{
  ...popupOptions,
  row: {{ value: 505, numerator: 361075, denominator: 715, row_count: 720 }},
  data: {{
    level: "sector",
    response: {{ label: "MARKET_PRICE_1_5" }},
    denominator: {{ column: "EXPOSURE", bar_label: "EXPOSURE" }},
  }},
}});
const denominatorLabels = [
  "MARKET_PRICE_1_5 / EXPOSURE: £505",
  "MARKET_PRICE_1_5 total: 361,075",
  "EXPOSURE: 715",
  "Rows: 720",
];
const denominatorPopupText = popupText(denominatorPopup);
if (!denominatorLabels.every((label) => denominatorPopupText.includes(label))
    || denominatorLabels.some((label, index) => index && denominatorPopupText.indexOf(denominatorLabels[index - 1]) > denominatorPopupText.indexOf(label))
    || (denominatorPopup.match(/class="map-popup-quantity"/g) || []).length !== 4) {{
  throw new Error(`incorrect denominator popup: ${{denominatorPopup}}`);
}}
const averageSmoothingPopup = ukMapPopupContentHtml({{
  ...popupOptions,
  row: {{
    value: 500,
    denominator: 1200,
    row_count: 999,
    raw_value: 505,
    raw_row_count: 700,
    smoothing_contributing_sectors: 4,
  }},
  data: {{
    level: "sector",
    response: {{ label: "MARKET_PRICE_1_5" }},
    denominator: {{ column: null }},
    smoothing: {{ applied: true, level: 2 }},
  }},
}});
const averageSmoothingLabels = [
  "Smoothed N2",
  "Smoothed average MARKET_PRICE_1_5: £500",
  "Pooled valid records: 1,200",
  "Contributing sectors: 4",
  "Unsmoothed",
  "Raw average MARKET_PRICE_1_5: £505",
  "Rows: 700",
];
const averageSmoothingPopupText = popupText(averageSmoothingPopup);
for (const label of averageSmoothingLabels) {{
  if (!averageSmoothingPopupText.includes(label)) throw new Error(`missing average smoothing label: ${{label}}`);
}}
if (averageSmoothingLabels.some((label, index) => index && averageSmoothingPopupText.indexOf(averageSmoothingLabels[index - 1]) > averageSmoothingPopupText.indexOf(label))
    || averageSmoothingPopupText.includes("Rows: 999")
    || (averageSmoothingPopup.match(/class="map-popup-quantity"/g) || []).length !== 5
    || (averageSmoothingPopup.match(/map-popup-section/g) || []).length < 4) {{
  throw new Error(`smoothed records should use the raw row count: ${{averageSmoothingPopup}}`);
}}
const denominatorSmoothingPopup = ukMapPopupContentHtml({{
  ...popupOptions,
  row: {{
    value: 500,
    numerator: 600000,
    denominator: 1200,
    row_count: 999,
    raw_value: 505,
    raw_numerator: 361075,
    raw_denominator: 715,
    raw_row_count: 720,
    smoothing_contributing_sectors: 4,
  }},
  data: {{
    level: "sector",
    response: {{ label: "MARKET_PRICE_1_5" }},
    denominator: {{ column: "EXPOSURE", bar_label: "EXPOSURE" }},
    smoothing: {{ applied: true, level: 2 }},
  }},
}});
const denominatorSmoothingLabels = [
  "Smoothed N2",
  "Smoothed MARKET_PRICE_1_5 / EXPOSURE: £500",
  "MARKET_PRICE_1_5 total: 600,000",
  "EXPOSURE: 1,200",
  "Contributing sectors: 4",
  "Unsmoothed",
  "Raw MARKET_PRICE_1_5 / EXPOSURE: £505",
  "Raw MARKET_PRICE_1_5 total: 361,075",
  "Raw EXPOSURE: 715",
  "Rows: 720",
];
const denominatorSmoothingPopupText = popupText(denominatorSmoothingPopup);
for (const label of denominatorSmoothingLabels) {{
  if (!denominatorSmoothingPopupText.includes(label)) throw new Error(`missing denominator smoothing label: ${{label}}`);
}}
if (denominatorSmoothingLabels.some((label, index) => index && denominatorSmoothingPopupText.indexOf(denominatorSmoothingLabels[index - 1]) > denominatorSmoothingPopupText.indexOf(label))
    || (denominatorSmoothingPopup.match(/class="map-popup-quantity"/g) || []).length !== 8) {{
  throw new Error(`incorrect denominator smoothing order: ${{denominatorSmoothingPopup}}`);
}}
const fallbackPopup = ukMapPopupContentHtml({{
  ...popupOptions,
  row: {{ value: 505, row_count: 715, raw_value: 505, raw_row_count: 715, smoothing_contributing_sectors: 0 }},
  data: {{
    level: "sector",
    response: {{ label: "MARKET_PRICE_1_5" }},
    denominator: {{ column: null }},
    smoothing: {{ applied: true, level: 3 }},
  }},
}});
const fallbackLabels = [
  "Smoothed N3",
  "No contributing sectors; unsmoothed value shown.",
  "Unsmoothed",
  "Raw average MARKET_PRICE_1_5: £505",
  "Rows: 715",
];
const fallbackPopupText = popupText(fallbackPopup);
if (!fallbackLabels.every((label) => fallbackPopupText.includes(label))
    || fallbackLabels.some((label, index) => index && fallbackPopupText.indexOf(fallbackLabels[index - 1]) > fallbackPopupText.indexOf(label))
    || (fallbackPopup.match(/class="map-popup-quantity"/g) || []).length !== 2
    || fallbackPopup.includes("Smoothed average")) throw new Error(`incorrect smoothing fallback: ${{fallbackPopup}}`);
const noDataPopup = ukMapPopupContentHtml({{
  ...popupOptions,
  title: "<CA&10>",
  row: null,
  showViewRows: false,
  areaFilterToggle: true,
}});
if (!noDataPopup.includes("&lt;CA&amp;10&gt;")
    || !noDataPopup.includes("No matching data")
    || noDataPopup.includes("view-rows")
    || (noDataPopup.match(/map-popup-action-icon/g) || []).length !== 2
    || !noDataPopup.includes('aria-pressed="false"')
    || !noDataPopup.includes("Add &lt;CA&amp;10&gt; to postcode area filter")
    || !noDataPopup.includes('data-map-popup-action="filter"')) {{
  throw new Error(`incorrect no-data popup or actions: ${{noDataPopup}}`);
}}
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

    def test_shared_model_ui_helpers_are_importable(self) -> None:
        module = Path("src/py_lucidum/static/app/shared/model-ui.js").resolve().as_uri()
        script = f"""
import {{
  bindFallbackModelSelection,
  createSidebarModelHeading,
  createSidebarModelOption,
  emptyStateHtml,
  formatModelCreated,
  formatModelMetric,
  isModelJobPending,
  modelCreatedSort,
  modelGroups,
  modelJobPollDelay,
  modelNumberOrNull,
  observeResize,
  restoreModelSelection,
  selectedModelIdsFromTableOrFallback,
  setInlinePhaseStatus,
  syncCollapsedModelGroups,
  syncModelActionButtons,
  toggleSidebarModelGroup,
}} from "{module}";

if (modelNumberOrNull("") !== null) throw new Error("empty number should be null");
if (modelNumberOrNull("12.5") !== 12.5) throw new Error("numeric string failed");
if (formatModelMetric(null) !== "--") throw new Error("missing metric failed");
if (formatModelMetric(1234.56789) !== "1,234.5679") throw new Error("metric formatting failed");
if (formatModelMetric(-0) !== "0") throw new Error("negative zero metric failed");
if (formatModelMetric(-0.00001) !== "0") throw new Error("rounded negative zero metric failed");
if (formatModelCreated("2026-01-02T03:04:00") !== "2 Jan 03:04") throw new Error("created formatting failed");
if (formatModelCreated("not-a-date") !== "not-a-date") throw new Error("invalid date failed");
if (modelCreatedSort("not-a-date") !== 0) throw new Error("invalid sort failed");
if (!isModelJobPending("queued") || !isModelJobPending("running") || isModelJobPending("failed")) throw new Error("pending status failed");
if (modelJobPollDelay("queued", 1000, 500) !== 1000) throw new Error("queued delay failed");
if (modelJobPollDelay("running", 1000, 500) !== 500) throw new Error("running delay failed");
if (modelJobPollDelay("succeeded", 1000, 500) !== 0) throw new Error("terminal delay failed");

class FakeElement {{
  constructor(tag) {{
    this.tag = tag;
    this.children = [];
    this.dataset = {{}};
    this.attributes = {{}};
    this.listeners = {{}};
    this.className = "";
    this.hidden = false;
    this.title = "";
    this.innerHTML = "";
    this.type = "";
    this.classList = {{
      add: (className) => this.setClass(className, true),
      remove: (className) => this.setClass(className, false),
      toggle: (className, active) => this.setClass(className, active),
      contains: (className) => String(this.className || "").split(/\\s+/).includes(className),
    }};
  }}
  setClass(className, active) {{
    const classes = new Set(String(this.className || "").split(/\\s+/).filter(Boolean));
    if (active) classes.add(className);
    else classes.delete(className);
    this.className = [...classes].join(" ");
  }}
  append(child) {{
    this.children.push(child);
  }}
  setAttribute(name, value) {{
    this.attributes[name] = String(value);
  }}
  getAttribute(name) {{
    return this.attributes[name];
  }}
  addEventListener(name, callback) {{
    this.listeners[name] = this.listeners[name] || [];
    this.listeners[name].push(callback);
  }}
  querySelectorAll(selector) {{
    const className = selector.startsWith(".") ? selector.slice(1) : selector;
    return this.children.filter((child) => String(child.className || "").split(/\\s+/).includes(className));
  }}
}}
const fallbackRows = [];
globalThis.document = {{
  createElement: (tag) => new FakeElement(tag),
  querySelectorAll: (selector) => selector.includes("data-gbm-model-row") ? fallbackRows : [],
}};

const models = [
  {{ model_id: "m1", label: "One", group: "A" }},
  {{ model_id: "m2", label: "Two", group: "B" }},
];
const grouped = modelGroups(models, (model) => model.group);
if (grouped.get("A")[0].model_id !== "m1" || grouped.get("B")[0].model_id !== "m2") throw new Error("model grouping failed");
const collapsedGroups = new Set();
const sync = syncCollapsedModelGroups({{ groups: [...grouped.keys()], collapsedGroups, initialised: false, activeGroup: "B" }});
if (!sync.initialised || !collapsedGroups.has("A") || collapsedGroups.has("B")) throw new Error("collapsed group init failed");

const list = new FakeElement("div");
let toggled = "";
const heading = createSidebarModelHeading({{
  group: "A",
  collapsed: true,
  toolLabel: "GBM",
  className: "gbm-model-theme",
  dataKey: "gbmModelGroup",
  escapeHtml: (value) => String(value),
  onToggle: (group) => {{ toggled = group; }},
}});
list.append(heading);
let activated = "";
const option = createSidebarModelOption({{
  model: models[0],
  group: "A",
  active: false,
  collapsed: true,
  className: "gbm-model-option",
  detailClassName: "gbm-model-detail",
  modelIdDataKey: "gbmModelId",
  groupDataKey: "gbmModelGroup",
  escapeHtml: (value) => String(value),
  modelLabel: (model) => model.label,
  modelDetailLabel: (model) => model.model_id,
  onActivate: (modelId) => {{ activated = modelId; }},
}});
list.append(option);
heading.listeners.click[0]();
if (toggled !== "A") throw new Error("heading click failed");
option.listeners.click[0]();
if (activated !== "m1") throw new Error("option click failed");
toggleSidebarModelGroup({{
  list,
  group: "A",
  collapsedGroups,
  themeClassName: "gbm-model-theme",
  optionClassName: "gbm-model-option",
  groupDataKey: "gbmModelGroup",
  toolLabel: "GBM",
}});
if (collapsedGroups.has("A") || option.hidden) throw new Error("toggle open failed");
if (heading.attributes["aria-expanded"] !== "true") throw new Error("heading expanded failed");
if (emptyStateHtml("Empty", "gbm-empty-state", (value) => String(value)) !== '<div class="gbm-empty-state">Empty</div>') throw new Error("empty state failed");

const status = new FakeElement("div");
setInlinePhaseStatus(status, {{ html: "Working", phase: "running", hidden: false }});
if (status.innerHTML !== "Working" || status.dataset.phase !== "running" || status.classList.contains("hidden")) throw new Error("status set failed");
setInlinePhaseStatus(status, {{ hidden: true }});
if (!status.classList.contains("hidden")) throw new Error("status hide failed");

const rowA = new FakeElement("tr");
rowA.dataset.gbmModelRow = "m1";
const rowB = new FakeElement("tr");
rowB.dataset.gbmModelRow = "m2";
fallbackRows.push(rowA, rowB);
let selectionChanges = 0;
bindFallbackModelSelection(fallbackRows, () => {{ selectionChanges += 1; }});
rowA.listeners.click[0]({{ metaKey: false, ctrlKey: false, shiftKey: false }});
if (rowA.attributes["aria-selected"] !== "true" || rowB.attributes["aria-selected"] !== "false") throw new Error("single row select failed");
rowB.listeners.click[0]({{ metaKey: true, ctrlKey: false, shiftKey: false }});
if (rowA.attributes["aria-selected"] !== "true" || rowB.attributes["aria-selected"] !== "true") throw new Error("command row select failed");
if (selectionChanges < 3) throw new Error("selection change callback failed");

const fallbackIds = selectedModelIdsFromTableOrFallback({{
  table: null,
  fallbackSelector: "#gbmModelFallback [data-gbm-model-row]",
  rowDataKey: "gbmModelRow",
}});
if (fallbackIds.join(",") !== "m1,m2") throw new Error(`fallback ids failed: ${{fallbackIds.join(",")}}`);
const tableIds = selectedModelIdsFromTableOrFallback({{
  table: {{ initialized: true, getSelectedData: () => [{{ model_id: "m3" }}, {{ model_id: "m3" }}, {{ model_id: "" }}] }},
  fallbackSelector: "#gbmModelFallback [data-gbm-model-row]",
  rowDataKey: "gbmModelRow",
}});
if (tableIds.join(",") !== "m3") throw new Error(`table ids failed: ${{tableIds.join(",")}}`);
let unreadySelectionReads = 0;
const unreadyIds = selectedModelIdsFromTableOrFallback({{
  table: {{
    initialized: false,
    getSelectedData: () => {{
      unreadySelectionReads += 1;
      return [{{ model_id: "unready" }}];
    }},
  }},
  fallbackSelector: "#gbmModelFallback [data-gbm-model-row]",
  rowDataKey: "gbmModelRow",
}});
if (unreadySelectionReads !== 0 || unreadyIds.join(",") !== "m1,m2") throw new Error("unready selection read failed");
let unreadyRowReads = 0;
restoreModelSelection({{
  table: {{
    initialized: false,
    getRows: () => {{
      unreadyRowReads += 1;
      return [];
    }},
  }},
  fallbackSelector: "#gbmModelFallback [data-gbm-model-row]",
  rowDataKey: "gbmModelRow",
  ids: ["m2"],
}});
if (unreadyRowReads !== 0) throw new Error("unready row read failed");
if (rowA.attributes["aria-selected"] !== "false" || rowB.attributes["aria-selected"] !== "true") throw new Error("restore fallback failed");
let readySelected = 0;
let readyDeselected = 0;
restoreModelSelection({{
  table: {{
    initialized: true,
    getRows: () => [
      {{ getData: () => ({{ model_id: "m1" }}), select: () => {{ readySelected += 1; }}, deselect: () => {{ readyDeselected += 1; }} }},
      {{ getData: () => ({{ model_id: "m2" }}), select: () => {{ readySelected += 1; }}, deselect: () => {{ readyDeselected += 1; }} }},
    ],
  }},
  fallbackSelector: "#gbmModelFallback [data-gbm-model-row]",
  rowDataKey: "gbmModelRow",
  ids: ["m2"],
}});
if (readySelected !== 1 || readyDeselected !== 1) throw new Error("ready table restore failed");

const activate = {{}};
const rename = {{}};
const deleteButton = {{}};
const openFolder = {{}};
syncModelActionButtons({{ selectedCount: 1, disabled: false, openFolder, activate, rename, deleteButton }});
if (openFolder.disabled || activate.disabled || rename.disabled || deleteButton.disabled) throw new Error("enabled action state failed");
syncModelActionButtons({{ selectedCount: 2, disabled: false, openFolder, activate, rename, deleteButton }});
if (!openFolder.disabled || !activate.disabled || !rename.disabled || deleteButton.disabled) throw new Error("multiple action state failed");
syncModelActionButtons({{ selectedCount: 1, disabled: false, openFolder, openFolderPending: true, activate, rename, deleteButton }});
if (!openFolder.disabled || activate.disabled || rename.disabled || deleteButton.disabled) throw new Error("folder pending state failed");
syncModelActionButtons({{ selectedCount: 2, disabled: true, openFolder, activate, rename, deleteButton }});
if (!openFolder.disabled || !activate.disabled || !rename.disabled || !deleteButton.disabled) throw new Error("disabled action state failed");

let observedCount = 0;
class FakeResizeObserver {{
  constructor(callback) {{ this.callback = callback; }}
  observe(target) {{ if (target) observedCount += 1; }}
  disconnect() {{}}
}}
globalThis.window = {{ ResizeObserver: FakeResizeObserver }};
globalThis.ResizeObserver = FakeResizeObserver;
const resizeObserver = observeResize([rowA, null, rowB], () => {{}});
if (!resizeObserver || observedCount !== 2) throw new Error("resize observer failed");
"""
        self.run_node_script(script)

    def test_model_navigator_fallback_metadata_contract(self) -> None:
        glm_module = Path("src/py_lucidum/static/app/glm-model-navigator.js").resolve().as_uri()
        gbm_module = Path("src/py_lucidum/static/app/gbm-model-navigator.js").resolve().as_uri()
        script = f"""
import {{ createGlmModelNavigator, glmTrainingScopeLabel }} from "{glm_module}";
import {{ createGbmModelNavigator }} from "{gbm_module}";

const target = () => ({{ innerHTML: "", querySelectorAll: () => [] }});
const escapeHtml = (value) => String(value ?? "");
const glmNavigator = createGlmModelNavigator({{
  bindFallbackModelSelection: () => {{}},
  emptyStateHtml: (message) => message,
  escapeHtml,
  formatModelCreated: (value) => String(value || ""),
  formatModelMetric: (value) => value === null || value === undefined ? "--" : String(value),
  modelCreatedSort: () => 0,
  modelLabel: (model) => model.label || model.model_id,
  modelNumberOrNull: (value) => value === null || value === undefined || value === "" ? null : Number(value),
  modelWeightLabel: (value) => String(value || "N"),
  normaliseModels: (models) => models,
  selectedModelIds: () => new Set(),
  onFallbackSelectionChange: () => {{}},
}});
const glmTarget = target();
glmNavigator.renderFallback(glmTarget, [
  {{ model_id: "new", label: "New GLM", training_scope: "training_test", n_terms: 3, n_features: 2, n_interactions: 1, tabulated: true, timings: {{ fit_ms: 1200, elapsed_ms: 2500 }}, diagnostics: {{ bic: 123.45, gini_tr: 0.81234, gini_te: -0.25, gini_vl: null }} }},
  {{ model_id: "legacy", label: "Legacy GLM", tabulated: false, diagnostics: {{}} }},
]);
if (!glmTarget.innerHTML.includes("<th>Name</th>")) throw new Error("GLM fallback Name heading missing");
if (!glmTarget.innerHTML.includes("<th>Terms</th>")) throw new Error("GLM fallback Terms heading missing");
if (!glmTarget.innerHTML.includes("<th>Features</th>")) throw new Error("GLM fallback Features heading missing");
if (!glmTarget.innerHTML.includes("<th>Interactions</th>")) throw new Error("GLM fallback Interactions heading missing");
if (!glmTarget.innerHTML.includes("<th>Tabulated</th>")) throw new Error("GLM fallback Tabulated heading missing");
if (!glmTarget.innerHTML.includes(">gini_tr</th>")) throw new Error("GLM fallback gini_tr heading missing");
if (!glmTarget.innerHTML.includes(">gini_te</th>")) throw new Error("GLM fallback gini_te heading missing");
if (!glmTarget.innerHTML.includes(">gini_vl</th>")) throw new Error("GLM fallback gini_vl heading missing");
if (!glmTarget.innerHTML.includes("<th>Rows</th>\\n            <th>Scope</th>\\n            <th>Fit time</th>")) throw new Error("GLM fallback scope heading order failed");
if (!glmTarget.innerHTML.includes('<td class="numeric">0.81234</td>')) throw new Error("GLM gini_tr value missing");
if (!glmTarget.innerHTML.includes('<td class="numeric">-0.25</td>')) throw new Error("GLM gini_te value missing");
if (!glmTarget.innerHTML.includes('<td class="numeric">--</td>')) throw new Error("GLM null Gini value missing");
if (!glmTarget.innerHTML.includes('<td class="numeric">123.45</td>')) throw new Error("GLM BIC value missing");
if (!glmTarget.innerHTML.includes('<td class="numeric">1.2s</td>')) throw new Error("GLM fit time missing");
if (!glmTarget.innerHTML.includes('<td class="numeric">2.5s</td>')) throw new Error("GLM overall time missing");
if (!glmTarget.innerHTML.includes('<td class="numeric">3</td>\\n        <td class="numeric">2</td>\\n        <td class="numeric">1</td>\\n        <td>Yes</td>')) throw new Error("GLM captured metadata missing");
if (!glmTarget.innerHTML.includes('<td class="numeric"></td>\\n        <td class="numeric"></td>\\n        <td class="numeric"></td>\\n        <td>-</td>')) throw new Error("GLM legacy metadata fallback failed");
if (!glmTarget.innerHTML.includes("<td>Training + Test</td>")) throw new Error("GLM training + test scope missing");
if (!glmTarget.innerHTML.includes("<td>All</td>")) throw new Error("GLM legacy scope fallback missing");
if (glmNavigator.optionalCount(null) !== "" || glmNavigator.optionalCount(0) !== "0") throw new Error("GLM optional count formatting failed");
if (glmTrainingScopeLabel("all") !== "All") throw new Error("GLM all scope mapping failed");
if (glmTrainingScopeLabel("training") !== "Training") throw new Error("GLM training scope mapping failed");
if (glmTrainingScopeLabel("training_test") !== "Training + Test") throw new Error("GLM training + test scope mapping failed");
if (glmTrainingScopeLabel(undefined) !== "All" || glmTrainingScopeLabel("   ") !== "All") throw new Error("GLM missing legacy scope mapping failed");
if (glmTrainingScopeLabel("future_scope") !== "--") throw new Error("GLM invalid scope mapping failed");

const gbmNavigator = createGbmModelNavigator({{
  escapeHtml,
  formatModelMetric: (value) => value === null || value === undefined ? "--" : String(value),
  modelInteractionConstraintLabel: () => "No",
  modelLabel: (model) => model.label || model.model_id,
  normaliseModel: (model) => model,
  uniqueModels: (models) => models,
  onFallbackSelectionChange: () => {{}},
}});
const gbmTarget = target();
gbmNavigator.renderFallback(gbmTarget, [{{ model_id: "gbm", label: "GBM", gini_tr: 0.75, gini_te: null, gini_vl: -0.125 }}]);
if (!gbmTarget.innerHTML.includes("<th>Name</th>")) throw new Error("GBM fallback Name heading missing");
if (gbmTarget.innerHTML.includes("<th>Model</th>")) throw new Error("GBM fallback kept Model heading");
if (!gbmTarget.innerHTML.includes(">gini_tr</th>")) throw new Error("GBM fallback gini_tr heading missing");
if (!gbmTarget.innerHTML.includes(">gini_te</th>")) throw new Error("GBM fallback gini_te heading missing");
if (!gbmTarget.innerHTML.includes(">gini_vl</th>")) throw new Error("GBM fallback gini_vl heading missing");
if (!gbmTarget.innerHTML.includes('<td class="numeric">0.75</td>')) throw new Error("GBM gini_tr value missing");
if (!gbmTarget.innerHTML.includes('<td class="numeric">--</td>')) throw new Error("GBM null Gini value missing");
if (!gbmTarget.innerHTML.includes('<td class="numeric">-0.125</td>')) throw new Error("GBM gini_vl value missing");
"""
        self.run_node_script(script)

    def test_glm_formula_builder_panel_state_defaults_and_migrates(self) -> None:
        module = Path("src/py_lucidum/static/app/glm-formula-builder.js").resolve().as_uri()
        script = f"""
import {{ createGlmFormulaBuilder }} from "{module}";

function createStorage(initial = {{}}) {{
  const values = new Map(Object.entries(initial));
  globalThis.localStorage = {{
    getItem: (key) => values.has(key) ? values.get(key) : null,
    setItem: (key, value) => values.set(key, String(value)),
    removeItem: (key) => values.delete(key),
  }};
}}

function builder(initial = {{}}) {{
  createStorage(initial);
  return createGlmFormulaBuilder({{
    el: () => null,
    escapeHtml: (value) => String(value ?? ""),
  }});
}}

const fresh = builder();
if (!fresh.parametersOpen || fresh.formulaAssistOpen) throw new Error("fresh builder should show parameters");
const legacyFormula = builder({{ py_lucidum_glm_formula_assist_open: "true" }});
if (!legacyFormula.formulaAssistOpen || legacyFormula.parametersOpen) throw new Error("legacy formula drawer state was not migrated");
const storedFormula = builder({{ py_lucidum_glm_builder_panel: "formula" }});
if (!storedFormula.formulaAssistOpen || storedFormula.parametersOpen) throw new Error("stored formula panel failed");
const storedParameters = builder({{ py_lucidum_glm_builder_panel: "parameters" }});
if (storedParameters.formulaAssistOpen || !storedParameters.parametersOpen) throw new Error("stored parameters panel failed");
const storedNone = builder({{ py_lucidum_glm_builder_panel: "none" }});
if (storedNone.formulaAssistOpen || storedNone.parametersOpen) throw new Error("stored editor-only panel failed");
"""
        self.run_node_script(script)






    def test_glm_formula_assist_helpers(self) -> None:
        module = Path("src/py_lucidum/static/app/glm-formula-assist.js").resolve().as_uri()
        script = f"""
import {{
  GLM_FORMULA_SNIPPETS,
  buildGroupedLevelsFormula,
  buildIndividualLevelsFormula,
  buildPiecewiseFormula,
  buildSnippetFormula,
  formulaColumnSuggestions,
  formulaCompletionContext,
  formatDrawerInsertion,
  formulaStringLiteral,
  parseBreakpoints,
  quoteFormulaName,
  rankFormulaSuggestions,
  withFormulaHeader,
}} from "{module}";

if (quoteFormulaName("Age") !== "Age") throw new Error("simple quote failed");
if (quoteFormulaName("vehicle age") !== "`vehicle age`") throw new Error("space quote failed");
if (quoteFormulaName("a`b") !== "`a\\\\`b`") throw new Error("backtick quote failed");
if (formulaStringLiteral('A"B\\\\C') !== '"A\\\\\\"B\\\\\\\\C"') throw new Error("string literal failed");

const grouped = buildGroupedLevelsFormula("PostcodeArea", ["BA", "BH", "BR"]);
if (grouped !== 'ifelse(np.isin(PostcodeArea, ["BA", "BH", "BR"]), 1, 0)') throw new Error(grouped);

const piecewise = buildPiecewiseFormula("EngineCC", [10, 20, 30]);
const expectedPiecewise = [
  "+ pmin(10, EngineCC)",
  "+ pmax(10, pmin(20, EngineCC))",
  "+ pmax(20, pmin(30, EngineCC))",
  "+ pmax(30, EngineCC)",
].join("\\n");
if (piecewise !== expectedPiecewise) throw new Error(piecewise);
const individual = buildIndividualLevelsFormula("MAKE", ["ALFA ROMEO", "AUDI"]);
if (individual !== '+ ifelse(MAKE == "ALFA ROMEO", 1, 0)\\n+ ifelse(MAKE == "AUDI", 1, 0)') throw new Error(individual);

if (buildSnippetFormula("bs4", "Age") !== "bs(Age, df=4)") throw new Error("bs snippet failed");
if (buildSnippetFormula("ns4", "Age") !== 'ns(Age, df=4, constraints="center")') throw new Error("ns snippet failed");
if (buildSnippetFormula("clamp", "Age") !== "pmax(lower_bound, pmin(upper_bound, Age))") throw new Error("clamp snippet failed");
if (buildSnippetFormula("positive_offset", "Age", {{ denominator: "Weight" }}) !== "offset(log(pmax(Weight, 1)))") throw new Error("offset snippet failed");
if (GLM_FORMULA_SNIPPETS.some((item) => item.id === "interaction" || item.id === "factor")) throw new Error("non-numeric snippet listed");
if (formatDrawerInsertion("Age", "") !== "Age\\n") throw new Error("empty insertion format failed");
if (formatDrawerInsertion("Age", "actual ~ 1") !== "+ Age\\n") throw new Error("plus insertion format failed");
if (formatDrawerInsertion("Age", "actual ~ ") !== "Age\\n") throw new Error("tilde insertion format failed");
if (formatDrawerInsertion("Age", "actual ~ 1 + ") !== "Age\\n") throw new Error("operator insertion format failed");
if (formatDrawerInsertion("+ Age", "actual ~ 1") !== "+ Age\\n") throw new Error("double plus insertion format failed");
if (formatDrawerInsertion("# Age\\nAge", "actual ~ 1") !== "# Age\\n+ Age\\n") throw new Error("header insertion format failed");
if (formatDrawerInsertion("Age", "actual ~ 1", {{ replaceSelection: true }}) !== "Age") throw new Error("selection insertion format failed");
if (withFormulaHeader("Age", "Driver Age", true) !== "# Driver Age\\nAge") throw new Error("header wrap failed");
if (withFormulaHeader("Age", "Driver Age", false) !== "Age") throw new Error("header disabled failed");

const parsed = parseBreakpoints("10, 20 30");
if (parsed.error || parsed.values.join(",") !== "10,20,30") throw new Error("break parse failed");
if (!parseBreakpoints("10, 10").error) throw new Error("break order failed");

const columns = [
  {{ name: "Age", kind: "integer" }},
  {{ name: "PostcodeArea", kind: "categorical" }},
  {{ name: "Vehicle Value", kind: "numeric" }},
];
const ranked = rankFormulaSuggestions(formulaColumnSuggestions(columns), "Po");
if (ranked[0].caption !== "PostcodeArea" || ranked[0].value !== "PostcodeArea") throw new Error("rank failed");
const context = formulaCompletionContext("np.isin(PostcodeArea, [B", 0, "np.isin(PostcodeArea, [B".length);
if (context.type !== "levels" || context.feature !== "PostcodeArea" || context.prefix !== "B") throw new Error(JSON.stringify(context));
const formulaContext = formulaCompletionContext("Post", 0, 4);
if (formulaContext.type !== "formula" || formulaContext.prefix !== "Post" || formulaContext.replaceStartColumn !== 0) throw new Error("formula context failed");
if (formulaCompletionContext("# Post", 0, 6).type !== "none") throw new Error("comment context failed");
"""
        self.run_node_script(script)








    def test_gbm_shap_flame_option_uses_line_bar_category_label_density(self) -> None:
        chart_path = Path(__file__).resolve().parents[1] / "src/py_lucidum/static/app/gbm-shap-chart.js"
        script = f"""
import fs from "node:fs";
const source = fs.readFileSync({str(chart_path)!r}, "utf8").replaceAll("export ", "");
eval(source + "\\nglobalThis.__shapChartOption = shapChartOption;\\nglobalThis.__shapFlameXAxisDensityMessage = shapFlameXAxisDensityMessage;\\nglobalThis.__formatUpliftPercent = formatUpliftPercent;");
function flameRow(x) {{
  return {{ x, p0: -0.2, p5: -0.19, p10: -0.18, p20: -0.17, p30: -0.16, p40: -0.15, p50: -0.14, p60: -0.13, p70: -0.12, p80: -0.11, p90: -0.1, p95: -0.09, p100: -0.08 }};
}}
function flameOptionForCount(count, chartWidth = 1200) {{
  const denseRows = Array.from({{ length: count }}, (_, index) => flameRow(index + 1));
  return globalThis.__shapChartOption({{
    plot_type: "flame",
    title: "SHAP flame plot: Category",
    x_feature: "Category",
    y_label: "SHAP",
    x_domain: [1, count],
    y_domain: [-0.2, -0.08],
    banding: 1,
    rows: denseRows,
  }}, {{}}, {{ chartWidth }});
}}
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
}}, {{}}, {{ chartWidth: 1200 }});
if (option.xAxis.type !== "category" || option.xAxis.data.join("|") !== "18|83") {{
  throw new Error(`expected ordered x categories 18|83, got ${{option.xAxis.data.join("|")}}`);
}}
const seriesNames = option.series.map((series) => series.name);
if (seriesNames.includes("45-55")) throw new Error("45-55 series should not be rendered");
const tooltip = option.tooltip.formatter([{{ axisValue: 18, value: [18, 0.16] }}]);
if (tooltip.includes("45-55")) throw new Error("45-55 tooltip row should not be rendered");
const demoOption = globalThis.__shapChartOption({{
  plot_type: "flame",
  title: "SHAP flame plot: POSTCODE_CATEGORY",
  x_feature: "POSTCODE_CATEGORY",
  y_label: "SHAP",
  x_domain: [3, 38],
  y_domain: [-0.4, 0.8],
  banding: 1,
  rows: Array.from({{ length: 36 }}, (_, index) => flameRow(index + 3)),
}}, {{}}, {{ chartWidth: 1200 }});
if (demoOption.xAxis.data.length !== 36 || demoOption.xAxis.data[0] !== 3 || demoOption.xAxis.data.at(-1) !== 38) {{
  throw new Error(`demo categories were not preserved: ${{demoOption.xAxis.data.join("|")}}`);
}}
if (demoOption.xAxis.axisLabel.show !== true
    || demoOption.xAxis.axisLabel.interval !== 0
    || demoOption.xAxis.axisLabel.rotate !== 65
    || demoOption.xAxis.axisLabel.fontSize !== 10
    || demoOption.xAxis.axisLabel.showMinLabel !== true
    || demoOption.xAxis.axisLabel.showMaxLabel !== true
    || demoOption.xAxis.axisLabel.hideOverlap !== false) {{
  throw new Error(`unexpected demo label policy: ${{JSON.stringify(demoOption.xAxis.axisLabel)}}`);
}}
if (demoOption.grid.containLabel !== false || demoOption.grid.bottom !== 64 || demoOption.xAxis.nameGap !== 48) {{
  throw new Error(`unexpected demo axis spacing: ${{JSON.stringify({{ grid: demoOption.grid, nameGap: demoOption.xAxis.nameGap }})}}`);
}}
if (demoOption.dataZoom.length !== 0) throw new Error("36 categories should not enable data zoom");
const demoMedian = demoOption.series.find((series) => series.name === "Median");
if (demoMedian.data.length !== 36 || demoMedian.data.some((value) => Array.isArray(value))) {{
  throw new Error("flame median values must align to category indexes");
}}
const demoRibbon = demoOption.series.find((series) => series.type === "custom");
const ribbonCoordinates = [];
demoRibbon.renderItem({{ dataIndex: 0 }}, {{
  coord: (value) => {{ ribbonCoordinates.push(value); return value; }},
}});
const ribbonIndexes = new Set(ribbonCoordinates.map((value) => value[0]));
if (Math.min(...ribbonIndexes) !== 0 || Math.max(...ribbonIndexes) !== 35 || ribbonIndexes.size !== 36) {{
  throw new Error(`flame ribbons must use category indexes: ${{[...ribbonIndexes].join("|")}}`);
}}
if (flameOptionForCount(50).xAxis.axisLabel.fontSize !== 10) throw new Error("50 categories should retain 10px labels");
if (flameOptionForCount(51).xAxis.axisLabel.fontSize !== 8) throw new Error("51 categories should use 8px labels");
if (flameOptionForCount(120).dataZoom.length !== 0) throw new Error("120 categories should not enable data zoom");
if (flameOptionForCount(121).dataZoom.length !== 2) throw new Error("121 categories should enable inside and slider zoom");
if (flameOptionForCount(199).xAxis.axisLabel.show !== true) throw new Error("199 category labels should remain visible");
const hiddenOption = flameOptionForCount(200);
if (hiddenOption.xAxis.axisLabel.show !== false || hiddenOption.dataZoom.length !== 2) {{
  throw new Error("200 categories should hide labels and retain zoom");
}}
if (!globalThis.__shapFlameXAxisDensityMessage({{ rows: Array(200).fill({{}}) }}).includes("200 or more")) {{
  throw new Error("dense flame plots should explain hidden x-axis labels");
}}
const wideShortOption = flameOptionForCount(10, 1200);
const narrowShortOption = flameOptionForCount(10, 200);
if (wideShortOption.xAxis.axisLabel.rotate !== 0 || narrowShortOption.xAxis.axisLabel.rotate !== 65) {{
  throw new Error("flame label rotation should respond to available chart width");
}}
const responseOption = globalThis.__shapChartOption({{
  plot_type: "flame",
  title: "SHAP flame plot: Age",
  x_feature: "Age",
  y_label: "SHAP",
  x_domain: [18, 83],
  y_domain: [-0.2, 1.2],
  rescale: {{ mode: "1" }},
  rows: [
    {{ x: 18, p0: 0.8, p5: 0.9, p10: 0.95, p20: 1, p30: 1.1, p40: 1.2, p50: 1.25, p60: 1.3, p70: 1.35, p80: 1.4, p90: 1.45, p95: 1.5, p100: 1.6 }},
    {{ x: 83, p0: 0.7, p5: 0.75, p10: 0.8, p20: 0.85, p30: 0.9, p40: 0.95, p50: 1, p60: 1.05, p70: 1.1, p80: 1.15, p90: 1.2, p95: 1.25, p100: 1.3 }},
  ],
}}, {{}}, {{ chartWidth: 1200 }});
const median = responseOption.series.find((series) => series.name === "Median");
if (!median?.markLine?.data || median.markLine.data[0].yAxis !== 1) {{
  throw new Error("rescale=1 SHAP mark line should be drawn at 1");
}}
if (globalThis.__formatUpliftPercent(1) !== "0%") throw new Error("base uplift should be 0%");
if (globalThis.__formatUpliftPercent(1.25) !== "+25%") throw new Error("positive uplift should include plus sign");
if (globalThis.__formatUpliftPercent(0.9) !== "-10%") throw new Error("negative uplift should be shown from base");
if (responseOption.yAxis.interval !== 0.1) throw new Error(`expected finer y-axis ticks, got ${{responseOption.yAxis.interval}}`);
if (responseOption.yAxis.axisLabel.formatter(1) !== "0%") throw new Error("rescale y-axis should show uplift percent at base");
if (responseOption.yAxis.axisLabel.formatter(1.25) !== "+25%") throw new Error("rescale y-axis should show positive uplift");
const responseTooltip = responseOption.tooltip.formatter([{{ axisValue: 18, value: [18, 1.25] }}]);
if (!responseTooltip.includes("+25%")) throw new Error(`rescale tooltip should show uplift percent: ${{responseTooltip}}`);
"""
        self.run_node_script(script)


    def test_gbm_stacked_shap_uses_native_bar_stack_without_rescale(self) -> None:
        chart_path = Path(__file__).resolve().parents[1] / "src/py_lucidum/static/app/gbm-stacked-shap-chart.js"
        script = f"""
import fs from "node:fs";
const source = fs.readFileSync({str(chart_path)!r}, "utf8").replaceAll("export ", "");
eval(source + "\\nglobalThis.__stackedShapChartOption = stackedShapChartOption;");
const option = globalThis.__stackedShapChartOption({{
  plot_type: "stacked_shap",
  title: "Stacked SHAP",
  model_feature: {{ name: "Age" }},
  display_features: ["A", "B"],
  y_domain: [-0.1, 0.2],
  rows: [
    {{ x: 40, row_count: 10, total_shap: 0.1, contributions: {{ A: 0.2, B: -0.1 }} }},
  ],
}}, {{}});
const aSeries = option.series.find((series) => series.name === "A");
const bSeries = option.series.find((series) => series.name === "B");
if (aSeries?.type !== "bar" || bSeries?.type !== "bar") throw new Error("stack segments should use native bars");
if (aSeries.stack !== "shap" || bSeries.stack !== "shap") throw new Error("stack segments should share the shap stack");
if (aSeries.markLine.data[0].yAxis !== 0) throw new Error("stacked mark line should be drawn at 0");
if (aSeries.data[0] !== 0.2 || bSeries.data[0] !== -0.1) throw new Error("stacked values should stay on the linear SHAP scale");
if (option.yAxis.name !== "SHAP Contribution (Linear Predictor Scale)") throw new Error(`unexpected y axis title: ${{option.yAxis.name}}`);
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


    def test_initial_tool_buttons_match_enabled_tools(self) -> None:
        cases = [
            (
                None,
                {"dataset_viewer", "column_profile", "line_bar", "histogram", "uk_map", "specs"},
                True,
                ["line_bar", "dataset_viewer", "column_profile", "histogram", "uk_map", "specs"],
            ),
            (["line-bar"], set(), False, []),
            (["gbm", "line-bar"], {"gbm", "line_bar"}, True, ["gbm", "line_bar"]),
            (["specs"], set(), False, []),
            (["dataset-viewer", "line-bar"], {"dataset_viewer", "line_bar"}, True, ["dataset_viewer", "line_bar"]),
        ]
        for tools, expected_visible_tools, selector_visible, expected_order in cases:
            with self.subTest(tools=tools):
                html = self.root_html_for_tools(tools)
                self.assert_tool_button_visibility(html, expected_visible_tools)
                self.assert_tool_selector_visibility(html, selector_visible)
                if expected_order:
                    self.assert_tool_button_order(html, expected_order)

    def test_header_buttons_are_opt_in(self) -> None:
        hidden_html = self.root_html_for_tools(None)
        visible_html = self.root_html_for_tools(None, header_buttons=True)

        self.assertIn('id="monitorLink" class="ghost monitor-link header-icon-button hidden"', hidden_html)
        self.assertIn('id="stopAppBtn" class="danger-action hidden" aria-hidden="true" inert>Stop app</button>', hidden_html)
        self.assertIn('id="monitorLink" class="ghost monitor-link header-icon-button" href="/monitor"', visible_html)
        self.assertIn('id="stopAppBtn" class="danger-action">Stop app</button>', visible_html)
        self.assertNotIn('id="monitorLink" class="ghost monitor-link header-icon-button hidden"', visible_html)
        self.assertNotIn('id="stopAppBtn" class="danger-action hidden"', visible_html)

    def test_initial_model_sidebar_panels_match_enabled_tools(self) -> None:
        cases = [
            (None, set()),
            (["line-bar"], set()),
            (["gbm", "line-bar"], {"gbm"}),
            (["glm", "line-bar"], {"glm"}),
            (["glm", "gbm", "line-bar"], {"glm", "gbm"}),
        ]
        for tools, expected_visible_tools in cases:
            with self.subTest(tools=tools):
                html = self.root_html_for_tools(tools)
                self.assert_model_sidebar_panel_visibility(html, expected_visible_tools)
                self.assertLess(html.index('data-sidebar-section="favourites"'), html.index('data-sidebar-section="kpi"'))
                self.assertLess(html.index('data-sidebar-section="kpi"'), html.index('data-sidebar-section="gbm"'))
                self.assertLess(html.index('data-sidebar-section="favourites"'), html.index('data-sidebar-section="gbm"'))
                self.assertLess(html.index('data-sidebar-section="gbm"'), html.index('data-sidebar-section="glm"'))
                self.assertLess(html.index('data-sidebar-section="glm"'), html.index('data-sidebar-section="filter"'))

    def test_sidebar_renames_metrics_and_model_denominator_contract_is_source_aware(self) -> None:
        html = self.root_html_for_tools(["line-bar", "histogram", "uk-map", "glm", "gbm"])
        main_source = Path("src/py_lucidum/static/app/main.js").read_text(encoding="utf-8")
        line_bar_source = Path("src/py_lucidum/static/app/line-bar-tool.js").read_text(encoding="utf-8")
        histogram_source = Path("src/py_lucidum/static/app/histogram-tool.js").read_text(encoding="utf-8")
        map_source = Path("src/py_lucidum/static/app/uk-map-tool.js").read_text(encoding="utf-8")
        glm_source = Path("src/py_lucidum/static/app/glm-tool.js").read_text(encoding="utf-8")
        gbm_source = Path("src/py_lucidum/static/app/gbm-tool.js").read_text(encoding="utf-8")

        self.assertIn('<div id="actualMetricTitle" class="metric-title">Numerator</div>', html)
        self.assertIn('<div id="weightMetricTitle" class="metric-title">Denominator</div>', html)
        self.assertIn('predictionGroup.label = "Model predictions"', main_source)
        self.assertIn('column.name === primaryName', main_source)
        self.assertIn("denominatorSource: selectedDenominator.sourceId", main_source)
        for source in (line_bar_source, histogram_source, map_source):
            self.assertIn("denominatorSource:", source)
        self.assertIn("getDenominatorSelection().metricKind === \"prediction\"", glm_source)
        self.assertIn("getDenominatorSelection().metricKind === \"prediction\"", gbm_source)
        self.assertIn("denominator_source:", glm_source)
        self.assertIn("offset_source:", gbm_source)

    def test_index_references_only_local_no_store_assets(self) -> None:
        _, body = self.assert_no_store("/")
        html = body.decode("utf-8")
        asset_urls = re.findall(r'<(?:link|script|img)\b[^>]*(?:href|src)="([^"]+)"', html)

        self.assertTrue(asset_urls)
        self.assertFalse(any(url.startswith(("http://", "https://", "//")) for url in asset_urls))
        self.assertIn("/static/vendor/maplibre-gl/maplibre-gl.css", asset_urls)
        self.assertFalse(any("leaflet" in url.lower() for url in asset_urls))
        for url in asset_urls:
            if url.startswith("/"):
                with self.subTest(url=url):
                    self.assert_no_store(url)

    def test_uk_map_lazy_loads_the_local_maplibre_module(self) -> None:
        adapter = Path("src/py_lucidum/static/app/maplibre-adapter.js").read_text(encoding="utf-8")
        index = Path("src/py_lucidum/static/index.html").read_text(encoding="utf-8")
        main_source = Path("src/py_lucidum/static/app/main.js").read_text(encoding="utf-8")
        map_source = Path("src/py_lucidum/static/app/uk-map-tool.js").read_text(encoding="utf-8")
        map_styles = Path("src/py_lucidum/static/styles/uk-map.css").read_text(encoding="utf-8")

        self.assertIn('import("../vendor/maplibre-gl/maplibre-gl.mjs")', adapter)
        self.assertIn('/static/vendor/maplibre-gl/maplibre-gl.css', index)
        self.assertIn('type: "canvas"', adapter)
        self.assertIn('animate: false', adapter)
        self.assertIn('refresh({ afterRender = null } = {})', adapter)
        self.assertIn('isZooming()', adapter)
        self.assertIn('replaceStyle(style = null, { foregroundLayerPredicate = null', adapter)
        self.assertIn('bringStyleForegroundToFront()', adapter)
        self.assertIn('restoreStyleResources(map = this.map)', adapter)
        self.assertIn('"fill-antialias": this.options.fillAntialias !== false', adapter)
        self.assertIn('getBoundsZoom(bounds, options = {})', adapter)
        self.assertIn('cameraForBounds(bounds.raw, nextOptions)', adapter)
        self.assertIn('getBearing()', adapter)
        self.assertIn('setBearing(bearing)', adapter)
        self.assertNotIn('light_nolabels/', map_source)
        self.assertNotIn('dark_nolabels/', map_source)
        self.assertIn('https://tiles.openfreemap.org/styles/positron', map_source)
        self.assertIn('https://tiles.openfreemap.org/styles/dark', map_source)
        self.assertIn('openFreeMapForegroundLayer', map_source)
        self.assertIn('bringBaseLabelsToFront()', map_source)
        self.assertIn('fillAntialias: false', map_source)
        self.assertIn('value="openFreeMapPositron"', index)
        self.assertIn('value="openFreeMapDark"', index)
        self.assertNotIn('value="grey"', index)
        self.assertNotIn('value="darkGrey"', index)
        self.assertIn('grey: "openFreeMapPositron"', map_source)
        self.assertIn('darkGrey: "openFreeMapDark"', map_source)
        self.assertIn('base-openfreemap-positron.png', index)
        self.assertIn('base-openfreemap-dark.png', index)
        self.assertIn('id="mapDotSizeMode"', index)
        self.assertIn('role="group" aria-label="Dot size"', index)
        self.assertIn('id="mapDotSizeMin"', index)
        self.assertIn('data-map-dot-size-mode="min"', index)
        self.assertIn('aria-pressed="false">Min</button>', index)
        self.assertIn('id="mapDotSizeAdaptive"', index)
        self.assertIn('data-map-dot-size-mode="adaptive"', index)
        self.assertIn('aria-pressed="true">Adaptive</button>', index)
        self.assertNotIn('id="mapDotSize" type="range"', index)
        self.assertIn('id="mapAreaLabels"', index)
        self.assertIn('role="group" aria-label="Area labels"', index)
        self.assertIn('id="mapAreaLabelsOff"', index)
        self.assertIn('data-map-area-labels="off"', index)
        self.assertIn('aria-pressed="true">Off</button>', index)
        self.assertIn('id="mapAreaLabelsOn"', index)
        self.assertIn('data-map-area-labels="on"', index)
        self.assertIn('aria-pressed="false">On</button>', index)
        self.assertNotIn('id="mapLabelSize" type="range"', index)
        self.assertIn('mapDotSizeMode: "adaptive"', main_source)
        self.assertIn('mapAreaLabels: "off"', main_source)
        self.assertIn('const MAP_UNIT_ADAPTIVE_DENSE_COUNT = 500_000;', map_source)
        self.assertIn('const MAP_UNIT_ADAPTIVE_SPARSE_COUNT = 100;', map_source)
        self.assertIn('const MAP_UNIT_ADAPTIVE_ULTRA_DENSE_COUNT = 1_500_000;', map_source)
        self.assertIn('const MAP_UNIT_ADAPTIVE_ULTRA_DENSE_MAX_DIAMETER = 8;', map_source)
        self.assertIn('const MAP_UNIT_ADAPTIVE_ZOOM_RANGE = 6;', map_source)
        self.assertIn('function unitPointAdaptiveDiameter(', map_source)
        self.assertIn('normaliseMapDotSizeMode(payload.dotSizeMode)', map_source)
        self.assertIn('normaliseMapAreaLabels(payload.areaLabels, payload.labelSize)', map_source)
        self.assertIn('normaliseMapBearing(payload.bearing)', map_source)
        self.assertIn('new L.maplibregl.NavigationControl({', map_source)
        self.assertIn('showCompass: true', map_source)
        self.assertIn(
            'mapCompassControlButton = navigationContainer.querySelector(".maplibregl-ctrl-compass")',
            map_source,
        )
        self.assertIn('mapCompassControlButton.id = "mapCompass"', map_source)
        self.assertIn('function mapCompassPointerAngle(', map_source)
        self.assertIn('Math.atan2(offsetY, offsetX)', map_source)
        self.assertIn('grid-template-rows: repeat(6, 34px);', map_styles)
        self.assertIn('.map-native-navigation {', map_styles)
        self.assertIn('display: contents;', map_styles)
        self.assertIn('function approximateMapAreaBoundsArea(', map_source)
        self.assertIn('function prepareMapAreaLabelZoomOffsets(', map_source)
        self.assertIn('function scheduleMapAreaLabelSizeUpdate(', map_source)
        self.assertIn('window.visualViewport?.addEventListener("resize"', map_source)
        self.assertIn('context.fillRect(Math.round(pointX * ratio), Math.round(pointY * ratio), 1, 1);', map_source)
        self.assertIn('.map-strip-mode-control .segmented button:disabled', map_styles)
        self.assertIn('.map-strip-option input:focus-visible + span,', map_styles)
        self.assertIn('.map-palette-button:focus-visible', map_styles)
        self.assertIn('.map-strip-option input:checked + span,', map_styles)
        self.assertIn('.map-palette-button.active', map_styles)
        self.assertIn('var(--map-area-label-base-size, 9px)', map_styles)
        self.assertIn('border: 0;', map_styles)
        self.assertNotIn("leaflet", index.lower())

    def test_javascript_modules_are_served_with_a_javascript_media_type(self) -> None:
        headers, body = self.assert_no_store("/static/vendor/maplibre-gl/maplibre-gl.mjs")

        self.assertTrue(body)
        self.assertEqual(mimetypes.guess_type("module.mjs")[0], "text/javascript")
        self.assertTrue(headers.get("content-type", "").startswith("text/javascript"), headers)

    def test_non_vendored_static_modules_are_served_no_store(self) -> None:
        static_root = Path(__file__).resolve().parents[1] / "src/py_lucidum/static"
        paths = [
            static_root / "app.css",
            static_root / "app.js",
            static_root / "monitor.css",
            static_root / "monitor.js",
            *sorted((static_root / "app").rglob("*.js")),
            *sorted((static_root / "styles").rglob("*.css")),
        ]

        for path in paths:
            url = f"/static/{path.relative_to(static_root).as_posix()}"
            with self.subTest(url=url):
                self.assert_no_store(url)

    def test_tool_screen_navigation_uses_a_distinct_shared_component(self) -> None:
        static_root = Path(__file__).resolve().parents[1] / "src/py_lucidum/static"
        controls = (static_root / "styles/controls.css").read_text(encoding="utf-8")
        helper = (static_root / "app/shared/tool-screen-nav.js").read_text(encoding="utf-8")
        gbm = (static_root / "app/gbm-tab-orchestration.js").read_text(encoding="utf-8")
        glm = (static_root / "app/glm-tool.js").read_text(encoding="utf-8")
        specs = (static_root / "app/specifications-tool.js").read_text(encoding="utf-8")
        index = (static_root / "index.html").read_text(encoding="utf-8")

        self.assertIn(".tool-screen-nav-item {", controls)
        self.assertIn("height: var(--app-tool-row-height);", controls)
        self.assertIn("background: var(--tool-screen-nav-bg);", controls)
        self.assertIn(".tool-screen-nav-item:first-child", controls)
        self.assertIn(".tool-screen-nav-item:last-child", controls)
        self.assertIn(".tool-screen-nav-item.active::after", controls)
        self.assertIn("@media (max-width: 900px)", controls)
        self.assertIn(".tool-screen-nav-label", controls)
        self.assertIn('role="tab"', helper)
        self.assertIn('aria-selected="${active}"', helper)
        self.assertIn('["ArrowLeft", "ArrowRight", "Home", "End"]', helper)
        self.assertIn('class="gbm-tabs tool-screen-nav"', (static_root / "app/gbm-tool.js").read_text(encoding="utf-8"))
        self.assertIn('class="glm-tabs tool-screen-nav"', glm)
        self.assertIn('class="tool-screen-nav spec-kind-tabs"', specs)
        self.assertIn('<div class="spec-file-row app-control-strip">', specs)
        self.assertLess(specs.index('<div class="spec-file-row app-control-strip">'), specs.index('id="specSaveBtn"'))
        self.assertIn("toolScreenNavButtonHtml", gbm)
        self.assertIn('id="lineBarWorkspaceControls" class="line-bar-workspace-controls hidden"', index)
        self.assertIn('id="lineBarTabs" class="tabs hidden"', index)
        self.assertNotIn('id="lineBarTabs" class="tool-screen-nav', index)

    def test_glm_tabulation_reports_table_and_row_scoring_progress(self) -> None:
        glm = (
            Path(__file__).resolve().parents[1]
            / "src/py_lucidum/static/app/glm-tool.js"
        ).read_text(encoding="utf-8")

        self.assertIn("export function glmTabulationReadyBadgeLabel", glm)
        self.assertIn('liveProgress = { phase: "queued", message: "Tabulating GLM..." };', glm)
        self.assertIn('? "Scoring tabulations"', glm)
        self.assertIn(': "Tabulating GLM";', glm)
        self.assertIn(
            "setAppReadyStatus(glmTabulationReadyBadgeLabel(progress), { elapsedStartedAt: tabulationElapsedStartedAt });",
            glm,
        )

    def test_app_status_badge_times_models_and_glm_tabulation(self) -> None:
        static_root = Path(__file__).resolve().parents[1] / "src/py_lucidum/static"
        index = (static_root / "index.html").read_text(encoding="utf-8")
        shell = (static_root / "styles/shell.css").read_text(encoding="utf-8")
        main = (static_root / "app/main.js").read_text(encoding="utf-8")
        glm = (static_root / "app/glm-tool.js").read_text(encoding="utf-8")
        gbm = (static_root / "app/gbm-tool.js").read_text(encoding="utf-8")

        self.assertIn('id="appStatusBadge" class="app-status-badge"', index)
        self.assertIn('class="app-status-badge-elapsed" aria-hidden="true"', index)
        self.assertIn(".app-status-badge.busy {", shell)
        self.assertIn("function setAppStatusBadge(message, stateClass = \"\", options = {})", main)
        self.assertIn("appStatusBadgeElapsedStartedAt !== elapsedStartedAt", main)
        self.assertIn("buildElapsedStartedAt = performance.now();", glm)
        self.assertIn("elapsedStartedAt: buildElapsedStartedAt", glm)
        self.assertLess(glm.index("buildElapsedStartedAt = performance.now();"), glm.index('api("/api/glm/validate"'))
        self.assertIn("tabulationElapsedStartedAt = performance.now();", glm)
        self.assertIn("elapsedStartedAt: tabulationElapsedStartedAt", glm)
        self.assertIn("trainingElapsedStartedAt = performance.now();", gbm)
        self.assertIn("elapsedStartedAt: trainingElapsedStartedAt", gbm)
        self.assertLess(gbm.index("trainingElapsedStartedAt = performance.now();"), gbm.index('api("/api/gbm/validate"'))
        self.assertIn('if (phase === "scoring") return "Scoring GBM";', gbm)
        self.assertIn('if (phase === "shap") return "Calculating GBM SHAP";', gbm)
        self.assertIn('if (phase === "artifacts") return "Saving GBM";', gbm)
        self.assertIn('if (phase === "succeeded") return "Finalising GBM";', gbm)
        self.assertNotIn("startupProgress", index + shell + main)
        self.assertNotIn("startup-progress", index + shell + main)

    def test_sidebar_rail_and_control_pane_use_distinct_theme_tokens(self) -> None:
        static_root = Path(__file__).resolve().parents[1] / "src/py_lucidum/static"
        foundations = (static_root / "styles/foundations.css").read_text(encoding="utf-8")
        shell = (static_root / "styles/shell.css").read_text(encoding="utf-8")
        controls = (static_root / "styles/controls.css").read_text(encoding="utf-8")

        self.assertIn("--sidebar-bg: #dde7f1;", foundations)
        self.assertIn("--sidebar-muted: #526174;", foundations)
        self.assertIn("--sidebar-rail-bg: #24364b;", foundations)
        self.assertIn("--sidebar-rail-text: #b8c4d3;", foundations)
        self.assertIn("--sidebar-rail-active: #72b7ff;", foundations)
        self.assertIn("--sidebar-bg: #1e2c42;", foundations)
        self.assertIn("--sidebar-rail-bg: #0b1220;", foundations)
        self.assertIn("--tool-screen-nav-bg: var(--panel-2);", foundations)
        self.assertIn("background: var(--sidebar-bg);", shell)
        self.assertIn("--muted: var(--sidebar-muted);", shell)
        self.assertIn("color: var(--sidebar-rail-text);", shell)
        self.assertIn("background: var(--sidebar-rail-bg);", controls)
        self.assertIn("color: var(--sidebar-rail-active);", controls)
        self.assertIn("color: var(--sidebar-rail-hover);", controls)
        self.assertIn(".sidebar-control-pane .feature-list {", controls)
        self.assertIn("background: var(--tool-screen-nav-bg);", controls)

    def test_sidebar_tool_icons_use_thin_shared_svg_strokes(self) -> None:
        static_root = Path(__file__).resolve().parents[1] / "src/py_lucidum/static"
        index = (static_root / "index.html").read_text(encoding="utf-8")
        controls = (static_root / "styles/controls.css").read_text(encoding="utf-8")

        tool_selector = index[
            index.index('<div class="tool-selector">') : index.index(
                '<div id="collapsedSidebarVersion"'
            )
        ]
        self.assertIn("stroke-width: 1.25;", controls)
        self.assertNotIn("stroke-width=", tool_selector)
        self.assertIn('<img src="/tools/uk-map/static/icons/UK.png" alt="">', tool_selector)

    def test_sidebar_accordion_selected_rows_share_theme_token(self) -> None:
        static_root = Path(__file__).resolve().parents[1] / "src/py_lucidum/static"
        foundations = (static_root / "styles/foundations.css").read_text(encoding="utf-8")
        controls = (static_root / "styles/controls.css").read_text(encoding="utf-8")

        self.assertEqual(
            foundations.count(
                "--sidebar-selected-row-bg: color-mix(in srgb, #f59e0b 22%, var(--panel));"
            ),
            2,
        )
        self.assertIn(
            """.favourites-list .saved-favourite-option.active,
      .kpi-list .kpi-option.active,
      .gbm-model-list .gbm-model-option.active,
      .glm-model-list .glm-model-option.active,
      .saved-filter-list .saved-filter-option.active {
        background: var(--sidebar-selected-row-bg);
      }""",
            controls,
        )

    def test_sidebar_filter_buttons_explain_their_behavior(self) -> None:
        index = (
            Path(__file__).resolve().parents[1] / "src/py_lucidum/static/index.html"
        ).read_text(encoding="utf-8")

        for tooltip in (
            "Select one saved filter at a time",
            "Filters in the same group use OR; different groups use AND",
            "Select multiple saved filters and choose how to combine them",
            "Require all selected filters to match",
            "Require at least one selected filter to match",
            "Require that not all selected filters match",
            "Require none of the selected filters to match",
        ):
            self.assertIn(f'title="{tooltip}"', index)

    def test_line_bar_two_feature_swap_command_contract(self) -> None:
        static_root = Path(__file__).resolve().parents[1] / "src/py_lucidum/static"
        index = (static_root / "index.html").read_text(encoding="utf-8")
        line_bar = (static_root / "app/line-bar-tool.js").read_text(encoding="utf-8")
        styles = (static_root / "styles/line-bar.css").read_text(encoding="utf-8")

        swap_tag = re.search(r'<button\b[^>]*\bid="lineBarSwapFeaturesBtn"[^>]*>', index)
        self.assertIsNotNone(swap_tag)
        self.assertIn('aria-label="Swap Feature 1 and Feature 2"', swap_tag.group(0))
        self.assertIn(" hidden", swap_tag.group(0))
        self.assertNotRegex(swap_tag.group(0), r"\btitle=")
        self.assertLess(index.index("<h2>x-axis features</h2>"), index.index('id="lineBarSwapFeaturesBtn"'))
        self.assertLess(index.index('id="lineBarSwapFeaturesBtn"'), index.index('data-control="featureSort"'))
        self.assertIn("function swapLineBarGroupingFeatures()", line_bar)
        self.assertIn('["bandWidth", "bandWidth2"]', line_bar)
        self.assertIn('["quantileMode", "quantileMode2"]', line_bar)
        self.assertIn('["dateBucketManualKey", "dateBucketManualKey2"]', line_bar)
        self.assertIn("state.bandSuggestionRequestSeq2 = (state.bandSuggestionRequestSeq2 || 0) + 1;", line_bar)
        self.assertIn("state.dateBucketSuggestionRequestSeq2 = (state.dateBucketSuggestionRequestSeq2 || 0) + 1;", line_bar)
        swap_start = line_bar.index("function swapLineBarGroupingFeatures()")
        self.assertLess(
            line_bar.index("invalidateGroupingSuggestions();", swap_start),
            line_bar.index('["x", "x2"]', swap_start),
        )
        self.assertIn('el("lineBarSwapFeaturesBtn")?.addEventListener("click", swapLineBarGroupingFeatures);', line_bar)
        self.assertIn(".line-bar-feature-heading {", styles)
        self.assertIn(".line-bar-feature-swap {", styles)
        self.assertIn("stroke-width: 1;", styles)

    def test_line_bar_glm_overlay_reuses_only_the_current_chart(self) -> None:
        line_bar = (
            Path(__file__).resolve().parents[1] / "src/py_lucidum/static/app/line-bar-tool.js"
        ).read_text(encoding="utf-8")

        self.assertIn('api("/api/line-bar/glm-overlay"', line_bar)
        self.assertIn("data.glm_overlay_context?.eligible", line_bar)
        self.assertIn("data._lineBarBaseRequestKey === baseChartRequestKey(request)", line_bar)
        self.assertIn("delete data.partial_dependence;", line_bar)
        self.assertNotIn("glmOverlayCache", line_bar)

    def test_line_bar_latest_intent_and_stable_pending_output_contract(self) -> None:
        static_root = Path(__file__).resolve().parents[1] / "src/py_lucidum/static"
        main = (static_root / "app/main.js").read_text(encoding="utf-8")
        line_bar = (static_root / "app/line-bar-tool.js").read_text(encoding="utf-8")
        glm = (static_root / "app/glm-tool.js").read_text(encoding="utf-8")
        gbm = (static_root / "app/gbm-tool.js").read_text(encoding="utf-8")
        index = (static_root / "index.html").read_text(encoding="utf-8")

        self.assertIn("function beginLineBarIntent(options = {})", line_bar)
        self.assertIn("function freezeLineBarSnapshotValue(value)", line_bar)
        self.assertIn("Object.values(value).forEach(freezeLineBarSnapshotValue);", line_bar)
        self.assertIn("abortLineBarController(chartAbortController);", line_bar)
        self.assertIn("abortLineBarController(tableAbortController);", line_bar)
        self.assertIn("function lineBarIntentIsCurrent(intent)", line_bar)
        self.assertIn("cache.requestSnapshot = snapshot;", line_bar)
        self.assertIn("data._lineBarRequestSnapshot = snapshot", line_bar)
        self.assertIn("if (!lineBarTable && !tableCacheData && !content?.children.length)", line_bar)
        self.assertNotIn("setChartPendingHidden(true);", line_bar)
        refresh_start = main.index("async function refreshTool(tool, options = {})")
        self.assertLess(
            main.index('const earlyIntent = tool === "line_bar"', refresh_start),
            main.index("const handler = await toolHandler(tool);", refresh_start),
        )
        self.assertLess(
            main.index("const intent = earlyIntent || handler.beginIntent?.(options) || null;", refresh_start),
            main.index("const request = handler.buildRequest();", refresh_start),
        )
        self.assertIn("if (earlyIntent && !lineBarTool.intentIsCurrent(earlyIntent)) return null;", main)
        self.assertIn("lineBarTool.invalidate({ pending: state.tool === \"line_bar\" });", main)
        self.assertIn("lineBarSchemaRequestSeq: 0", main)
        self.assertIn("if (requestSeq !== state.lineBarSchemaRequestSeq) return false;", main)
        self.assertNotIn(
            "|| columns.find((column) => column.name === state.x)",
            main,
        )
        self.assertEqual(index.count('id="lineBarGroupMeta"'), 1)
        self.assertNotIn("lineBarPending", index)
        for model_tool in (glm, gbm):
            self.assertIn('let queuedActivationModelId = "";', model_tool)
            self.assertIn("while (queuedActivationModelId)", model_tool)
            self.assertIn("if (queuedActivationModelId) continue;", model_tool)

    def test_gbm_parameter_json_is_lightgbm_compatible(self) -> None:
        module = Path("src/py_lucidum/static/app/gbm-feature-parameter-controls.js").resolve().as_uri()
        script = f"""
import {{ GBM_PARAMETER_GRID_COPY_ERROR, gbmParametersJson }} from "{module}";

const json = gbmParametersJson([
  {{ name: "init_score", value: "Baseline", important: true }},
  {{ name: "objective", value: "poisson", important: true }},
  {{ name: "tweedie_variance_power", value: "1.5", important: true }},
  {{ name: "num_iterations", value: "250", important: true }},
  {{ name: "learning_rate", value: " 5e-2 " }},
  {{ name: "force_col_wise", value: "TRUE" }},
  {{ name: "force_row_wise", value: "false" }},
  {{ name: "monotone_constraints", value: [-1, 0, 1] }},
  {{ name: "interaction_constraints", value: [[0, 1], [2]] }},
]);
const params = JSON.parse(json);
if (Object.keys(params).join("|") !== "objective|tweedie_variance_power|num_iterations|learning_rate|force_col_wise|force_row_wise|monotone_constraints") throw new Error("parameter order or metadata filtering failed");
if ("init_score" in params) throw new Error("init_score was copied");
if ("interaction_constraints" in params) throw new Error("generated interaction constraints were copied");
if (params.tweedie_variance_power !== 1.5 || params.num_iterations !== 250 || params.learning_rate !== 0.05) throw new Error("numeric coercion failed");
if (params.force_col_wise !== true || params.force_row_wise !== false) throw new Error("boolean coercion failed");
if (JSON.stringify(params.monotone_constraints) !== "[-1,0,1]") throw new Error("list value changed");
if (!json.includes('\\n  "objective": "poisson"')) throw new Error("JSON was not pretty printed");

let message = "";
try {{
  gbmParametersJson([{{ name: "learning_rate", value: "{{0.05, 0.1}}" }}]);
}} catch (error) {{
  message = error.message;
}}
if (message !== GBM_PARAMETER_GRID_COPY_ERROR) throw new Error("grid-search copy was not blocked");
"""
        self.run_node_script(script)

    def test_gbm_mape_evaluation_values_are_formatted_as_percentages(self) -> None:
        module = Path("src/py_lucidum/static/app/gbm-evaluation-chart-options.js").resolve().as_uri()
        script = f"""
import {{ gbmEvaluationChartOption }} from "{module}";

const mape = gbmEvaluationChartOption({{
  metric: "mape",
  manifest: {{ best_iteration: 2 }},
  evaluation: {{
    training: {{ mape: [0.4, 0.1254] }},
    test: {{ mape: [0.5, 0.206] }},
  }},
}});
if (mape.title.text !== "evaluation metric: mape, test metric: 20.6%, best iteration: 2") {{
  throw new Error(`MAPE title was not formatted as a percentage: ${{mape.title.text}}`);
}}
if (mape.yAxis.axisLabel.formatter(0.1254) !== "12.5%") throw new Error("MAPE axis was not formatted as a percentage");
const tooltip = mape.tooltip.formatter([{{ axisValue: 2, seriesIndex: 1, seriesName: "test", value: [2, 0.206] }}]);
if (!tooltip.includes("20.6%")) throw new Error(`MAPE tooltip was not formatted as a percentage: ${{tooltip}}`);

const l2 = gbmEvaluationChartOption({{
  metric: "l2",
  manifest: {{ best_iteration: 1 }},
  evaluation: {{ test: {{ l2: [0.206] }} }},
}});
if (l2.title.text.includes("%") || l2.yAxis.axisLabel.formatter(0.206).includes("%")) {{
  throw new Error("non-MAPE metric was formatted as a percentage");
}}
"""
        self.run_node_script(script)

    def test_dataset_viewer_sanitizes_tabulator_column_definitions(self) -> None:
        dataset_viewer = (
            Path(__file__).resolve().parents[1]
            / "src/py_lucidum/static/app/dataset-viewer-tool.js"
        ).read_text(encoding="utf-8")

        self.assertIn("function datasetViewerTabulatorColumns(columns) {", dataset_viewer)
        for field in ("copyTitle", "datasetRowId", "name", "sortField"):
            self.assertIn(f"delete definition.{field};", dataset_viewer)
        self.assertEqual(
            dataset_viewer.count("columns: datasetViewerTabulatorColumns(currentColumns),"),
            2,
        )
        self.assertIn(
            "datasetTable.setColumns(datasetViewerTabulatorColumns(search.columns));",
            dataset_viewer,
        )

    def test_column_profile_displays_boolean_columns_as_logical(self) -> None:
        module = Path("src/py_lucidum/static/app/column-profile-tool.js").resolve().as_uri()
        script = f"""
import {{ columnProfileTypeLabel }} from "{module}";

const cases = [
  [{{ kind: "categorical", duckdb_type: "BOOLEAN" }}, "logical"],
  [{{ kind: "categorical", duckdb_type: "bool" }}, "logical"],
  [{{ kind: "categorical", duckdb_type: "VARCHAR" }}, "categorical"],
  [{{ kind: "integer", duckdb_type: "BIGINT" }}, "integer"],
  [{{}}, "unknown"],
];
for (const [column, expected] of cases) {{
  const actual = columnProfileTypeLabel(column);
  if (actual !== expected) throw new Error(`expected ${{expected}}, received ${{actual}}`);
}}
"""
        self.run_node_script(script)

    def test_glm_tabulation_x_axis_presentation_is_dense_and_responsive(self) -> None:
        module = Path("src/py_lucidum/static/app/glm-tabulations.js").resolve().as_uri()
        script = f"""
import {{ createGlmTabulations }} from "{module}";

const tabulations = createGlmTabulations({{
  el: () => null,
  modelNumberOrNull: (value) => Number.isFinite(Number(value)) ? Number(value) : null,
  scheduleResize: () => {{}},
}});
const denseValues = Array.from({{ length: 61 }}, (_, index) => index + 1);
const dense = tabulations.xAxisPresentation({{
  features: ["POSTCODE_CATEGORY"],
  crosstab: "",
  x_axis: denseValues,
}}, 900, {{ text: "#111111", line: "#222222" }});
if (dense.xAxis.name !== "POSTCODE_CATEGORY") throw new Error("one-way x-axis title missing");
if (dense.xAxis.axisLabel.interval !== 0 || dense.xAxis.axisLabel.fontSize !== 8) throw new Error("dense labels were thinned");
if (dense.xAxis.axisLabel.rotate !== 65 || dense.xAxis.axisLabel.show !== true) throw new Error("dense labels did not rotate");
if (dense.xAxis.axisLine.onZero !== true || dense.xAxis.axisLine.lineStyle.width !== 2) throw new Error("zero axis is not prominent");

const interaction = tabulations.xAxisPresentation({{
  features: ["Age", "Segment"],
  crosstab: "Segment",
  x_axis: [30, 40, 50],
}}, 900);
if (interaction.xAxis.name !== "Age") throw new Error("interaction x-axis title is incorrect");

const expBaseline = tabulations.baselineMarkLine({{ scale: "exp" }}, {{ line: "#222222" }});
if (expBaseline?.data?.[0]?.yAxis !== 1 || expBaseline?.lineStyle?.width !== 2) throw new Error("Exp 0% baseline is not prominent");
if (tabulations.baselineMarkLine({{ scale: "linear" }}) !== null) throw new Error("linear plot received an Exp baseline");
const expAxis = tabulations.yAxisOptions({{ scale: "exp", min: 1.2, max: 1.8 }});
if (!(expAxis.min <= 1 && expAxis.max >= 1)) throw new Error("Exp axis omitted the 0% baseline");

const resizePayload = {{
  features: ["Feature"],
  crosstab: "",
  x_axis: Array.from({{ length: 10 }}, (_, index) => `v${{String(index).padStart(4, "0")}}`),
}};
const narrow = tabulations.xAxisPresentation(resizePayload, 300);
const wide = tabulations.xAxisPresentation(resizePayload, 1000);
if (narrow.xAxis.axisLabel.rotate !== 65 || wide.xAxis.axisLabel.rotate !== 0) throw new Error("width-aware rotation failed");

const zoomed = tabulations.xAxisPresentation({{ ...resizePayload, x_axis: Array.from({{ length: 121 }}, (_, index) => index) }}, 900);
if (zoomed.dataZoom.length !== 2) throw new Error("dense x-axis zoom missing");
const hidden = tabulations.xAxisPresentation({{ ...resizePayload, x_axis: Array.from({{ length: 200 }}, (_, index) => index) }}, 900);
if (hidden.xAxis.axisLabel.show !== false) throw new Error("density limit failed");
"""
        self.run_node_script(script)

    def test_glm_tabulation_rebase_ui_exposes_explicit_modes_and_clear_scopes(self) -> None:
        static_root = Path(__file__).resolve().parents[1] / "src/py_lucidum/static"
        glm = (static_root / "app/glm-tool.js").read_text(encoding="utf-8")
        glm_css = (static_root / "styles/glm.css").read_text(encoding="utf-8")

        for contract in (
            'mode: "cell_to_base"',
            'mode: "feature_level_to_one_way"',
            'return "Rebase to this cell; adjust base";',
            'return `Set ${anchorFeature}=${anchorValue} slice to ${baseline}; adjust ${targetFeature} table`;',
            'label: "Clear rebasing involving this table"',
            'label: "Clear all rebasing"',
            'divider.className = "glm-tabulation-context-menu-divider";',
            'item.danger ? " glm-tabulation-context-menu-item--danger" : ""',
            'scope === "table" ? selectedTabulationTableId : ""',
            'rules.map((rule) =>',
            'generatedTables.map((table) =>',
            'Created ${tableId} one-way adjustment table for rebasing',
        ):
            self.assertIn(contract, glm)
        self.assertNotIn("rules.slice(0, 3)", glm)
        self.assertIn(".glm-tabulation-rebase-table", glm_css)
        self.assertIn(".glm-tabulation-context-menu-divider", glm_css)
        self.assertIn("color: var(--danger);", glm_css)
        self.assertIn("background: color-mix(in srgb, var(--danger) 12%, var(--panel));", glm_css)
        self.assertNotIn("glmRecalculateTabulationsBtn", glm)
        self.assertNotIn("/api/glm/tabulations/recalculate", glm)

    def test_glm_tabulation_model_switch_reuses_loaded_config(self) -> None:
        static_root = Path(__file__).resolve().parents[1] / "src/py_lucidum/static"
        glm = (static_root / "app/glm-tool.js").read_text(encoding="utf-8")
        gbm = (static_root / "app/gbm-tool.js").read_text(encoding="utf-8")
        main = (static_root / "app/main.js").read_text(encoding="utf-8")

        for contract in (
            "function tabulationSelectionConfigFromCache(",
            "function registerTabulationModelMutation(",
            "function reconcileTabulationModelRefs(",
            "active: modelRef === activatedRef",
            "async function refreshTabulationSelectionFromCache()",
            "await renderTabulationSelectorTables();",
            "if (changed) refreshTabulationSelectionFromCache();",
            "const changed = previousKey !== tabulationSelectionKey();",
            "if (refreshSeq !== tabulationSelectionRefreshSeq) return;",
            "const modelRefs = reconcileTabulationModelRefs(response, requestedRefs);",
        ):
            self.assertIn(contract, glm)
        self.assertIn("onExternalModelMutation = async () => false", gbm)
        self.assertIn("await onExternalModelMutation({", gbm)
        self.assertIn("glmTool.handleExternalModelMutation(mutation)", main)
        self.assertNotIn("handleExternalModelActivation", glm + main)
        self.assertNotIn(
            "if (selectTabulationModel(modelId, event)) refreshTabulationConfig({ force: true });",
            glm,
        )

    def test_app_control_strips_use_shared_height_tokens(self) -> None:
        static_root = Path(__file__).resolve().parents[1] / "src/py_lucidum/static"
        foundations = (static_root / "styles/foundations.css").read_text(encoding="utf-8")
        controls = (static_root / "styles/controls.css").read_text(encoding="utf-8")
        dataset_viewer = (static_root / "app/dataset-viewer-tool.js").read_text(encoding="utf-8")
        profile = (static_root / "app/column-profile-tool.js").read_text(encoding="utf-8")
        model_shell = (static_root / "styles/model-shell.css").read_text(encoding="utf-8")
        glm_css = (static_root / "styles/glm.css").read_text(encoding="utf-8")
        gbm_css = (static_root / "styles/gbm.css").read_text(encoding="utf-8")
        glm = (static_root / "app/glm-tool.js").read_text(encoding="utf-8")
        glm_tabulations = (static_root / "app/glm-tabulations.js").read_text(encoding="utf-8")
        glm_formula_builder = (static_root / "app/glm-formula-builder.js").read_text(encoding="utf-8")
        gbm = (static_root / "app/gbm-tool.js").read_text(encoding="utf-8")
        gbm_evaluation_chart = (static_root / "app/gbm-evaluation-chart.js").read_text(encoding="utf-8")
        gbm_feature_controls = (static_root / "app/gbm-feature-parameter-controls.js").read_text(encoding="utf-8")
        gbm_tree_viewer = (static_root / "app/gbm-tree-viewer.js").read_text(encoding="utf-8")
        specs = (static_root / "app/specifications-tool.js").read_text(encoding="utf-8")
        index = (static_root / "index.html").read_text(encoding="utf-8")
        line_bar = (static_root / "app/line-bar-tool.js").read_text(encoding="utf-8")
        histogram = (static_root / "app/histogram-tool.js").read_text(encoding="utf-8")
        settings_strip = (static_root / "app/shared/settings-strip.js").read_text(encoding="utf-8")
        line_bar_css = (static_root / "styles/line-bar.css").read_text(encoding="utf-8")

        self.assertIn("--app-tool-row-height: 50px;", foundations)
        self.assertIn("--app-control-strip-height: var(--app-tool-row-height);", foundations)
        self.assertIn("--app-control-button-height: 28px;", foundations)
        self.assertIn(".app-control-strip {", controls)
        self.assertIn("min-height: var(--app-control-strip-height);", controls)
        self.assertIn(".app-control-strip-row {", controls)
        self.assertIn(".app-control-strip--titled {", controls)
        self.assertIn(".app-settings-strip.toolbar {", controls)
        self.assertIn("scrollbar-width: none;", controls)
        self.assertIn(".app-settings-strip.toolbar::-webkit-scrollbar {", controls)
        self.assertIn(".app-settings-strip .segmented button.active {", controls)
        self.assertIn(".app-control-button,", controls)
        self.assertIn("height: var(--app-control-button-height);", controls)
        self.assertIn(".app-command-button {", controls)
        self.assertIn(".app-command-button--danger {", controls)
        self.assertIn(".app-control-input {", controls)
        self.assertIn('class="dataset-viewer-toolbar app-control-strip app-control-strip-row"', dataset_viewer)
        self.assertIn('class="dataset-viewer-toolbar-group dataset-viewer-columns-control"', dataset_viewer)
        self.assertIn('id="datasetViewerColumnsLabel" class="dataset-viewer-toolbar-label"', dataset_viewer)
        self.assertIn(
            'id="datasetViewerPinnedMoveControls" class="dataset-viewer-pinned-move-controls" role="group"',
            dataset_viewer,
        )
        self.assertIn('class="dataset-viewer-pinned-move-label">move pinned column</span>', dataset_viewer)
        self.assertIn('id="datasetViewerPinnedMovePrevious"', dataset_viewer)
        self.assertIn('id="datasetViewerPinnedMoveNext"', dataset_viewer)
        self.assertIn('class="search dataset-viewer-search app-control-input"', dataset_viewer)
        self.assertIn(
            'class="dataset-viewer-search-clear app-control-button app-command-button"',
            dataset_viewer,
        )
        self.assertIn(
            'id="datasetViewerColumnsResizer" class="dataset-viewer-columns-resizer app-resizer app-resizer--vertical" role="separator"',
            dataset_viewer,
        )
        self.assertIn('aria-label="Resize Dataset Viewer Columns controls"', dataset_viewer)
        self.assertIn('class="dataset-viewer-toolbar-group dataset-viewer-view-control"', dataset_viewer)
        self.assertIn('class="dataset-viewer-view-divider" aria-hidden="true"', dataset_viewer)
        self.assertIn(
            'aria-label="Alphabetical columns" title="Alphabetical columns" aria-pressed="false" data-stable-label="A-Z">A-Z</button>',
            dataset_viewer,
        )
        self.assertIn('const DATASET_VIEWER_META_MIN_WIDTH = 220;', dataset_viewer)
        self.assertIn('const DATASET_VIEWER_TOOLBAR_RESIZE_STEP = 24;', dataset_viewer)
        self.assertIn('class="profile-toolbar app-control-strip app-control-strip-row"', profile)
        self.assertIn('class="profile-toolbar-group profile-columns-control"', profile)
        self.assertIn('id="profileColumnsLabel" class="profile-toolbar-label">Columns</h3>', profile)
        self.assertIn('class="search profile-column-search app-control-input"', profile)
        self.assertIn('aria-labelledby="profileColumnsLabel"', profile)
        self.assertIn(
            'id="profileColumnSearchClear" class="profile-column-search-clear app-control-button app-command-button"',
            profile,
        )
        self.assertIn('class="profile-toolbar-group-divider" aria-hidden="true"', profile)
        self.assertIn('class="profile-toolbar-group profile-rows-control"', profile)
        self.assertIn('class="profile-toolbar-label">Rows</h3>', profile)
        self.assertIn(
            'data-profile-summary-mode="full" data-stable-label="Use all" aria-label="Use all rows" title="Use all rows"',
            profile,
        )
        self.assertIn('class="profile-toolbar-meta-divider" aria-hidden="true"', profile)
        self.assertIn('id="profilePaneResizer"', profile)
        self.assertIn('role="separator"', profile)
        self.assertIn('aria-orientation="vertical"', profile)
        self.assertIn('import { bindVerticalListNavigation } from "./shared/list-navigation.js";', profile)
        self.assertIn('itemSelector: "[data-profile-column]"', profile)
        self.assertIn('row.tabIndex = selected ? 0 : -1;', profile)
        self.assertIn('class="toolbar line-bar-settings-strip app-control-strip app-settings-strip hidden"', index)
        self.assertIn('class="toolbar histogram-settings-strip app-control-strip app-settings-strip hidden"', index)
        self.assertIn('data-control="histogramBinMode"', index)
        self.assertIn('data-control="histogramLabels"', index)
        self.assertIn('id="histogramBinValueLabel"', index)
        self.assertIn('class="line-bar-workspace-controls hidden"', index)
        self.assertIn('id="histogramWorkspaceControls" class="histogram-workspace-controls hidden"', index)
        self.assertIn('id="histogramSplitResizer"', index)
        self.assertIn('class="line-bar-table-search-row app-control-strip"', line_bar)
        self.assertIn('import { bindSettingsStripOverflowCue } from "./shared/settings-strip.js";', line_bar)
        self.assertIn('import { bindSettingsStripOverflowCue } from "./shared/settings-strip.js";', histogram)
        self.assertIn('binMode: histogramBinMode()', histogram)
        self.assertIn('HISTOGRAM_Y_AXIS_TARGET_INTERVALS = 6', histogram)
        self.assertIn('niceHistogramYAxisInterval', histogram)
        self.assertIn('HISTOGRAM_BIN_OUTLINE_LIMIT = 200', histogram)
        self.assertIn('HISTOGRAM_MEDIAN_LABEL_OFFSET = 14', histogram)
        self.assertIn('classList.toggle("app-settings-overflow-left"', settings_strip)
        self.assertIn('classList.toggle(\n      "app-settings-overflow-right"', settings_strip)
        self.assertIn('toolbar.addEventListener("wheel", handleWheel, { passive: false });', settings_strip)
        self.assertIn('toolbar.removeEventListener("wheel", handleWheel);', settings_strip)
        self.assertIn("event.target?.closest?.(SETTINGS_STRIP_EDITABLE_SELECTOR)", settings_strip)
        self.assertIn("event.preventDefault();", settings_strip)
        self.assertIn("height: var(--app-control-strip-height);", line_bar_css)
        self.assertIn("grid-template-columns: var(--chart-controls-width, 340px) minmax(0, 1fr);", line_bar_css)
        self.assertIn("inset-inline-start: calc(var(--chart-controls-width, 340px) - 6px);", line_bar_css)
        self.assertNotIn(".model-control-strip", model_shell)
        self.assertNotIn("glm-builder-control-strip", glm_css)
        self.assertNotIn("model-control-strip", glm + gbm)
        self.assertNotIn("model-control-button", glm + gbm)
        self.assertIn("app-control-strip app-control-strip-row app-control-strip--titled", glm)
        self.assertIn("app-control-strip app-control-strip-row app-control-strip--actions", glm)
        self.assertIn("app-control-strip app-control-strip-row app-control-strip--actions", gbm)
        self.assertIn('id="glmFormulaAssistBtn" class="app-control-button glm-builder-option-button', glm)
        self.assertIn('aria-controls="glmFormulaAssistDrawer"', glm)
        self.assertIn('id="glmModelParametersBtn" class="app-control-button glm-builder-option-button', glm)
        self.assertIn('aria-label="Model parameters" title="Model parameters"', glm)
        self.assertIn('aria-controls="glmBuilderParametersPanel"', glm)
        self.assertIn('class="glm-model-parameters-icon" viewBox="0 0 24 24"', glm)
        self.assertIn('id="glmBuilderParametersPanel"', glm)
        self.assertIn('class="glm-editor-font-controls" role="group" aria-label="Formula editor controls"', glm)
        self.assertIn('id="glmCopyFormulaBtn" class="glm-editor-font-button"', glm)
        self.assertIn('aria-label="Copy formula" title="Copy formula"', glm)
        self.assertIn('class="glm-editor-copy-icon" viewBox="0 0 24 24"', glm)
        self.assertIn('id="glmCopyCoefficientsBtn" class="app-control-button glm-coefficient-action-button"', glm)
        self.assertIn('aria-label="Copy coefficients" title="Copy coefficients"', glm)
        self.assertIn('id="glmDownloadCoefficientsBtn" class="app-control-button glm-coefficient-action-button"', glm)
        self.assertIn('aria-label="Download coefficients" title="Download coefficients"', glm)
        self.assertEqual(glm.count('class="glm-coefficient-action-icon" viewBox="0 0 24 24"'), 2)
        self.assertNotIn('id="glmCopyCoefficientsBtn" class="tab ', glm)
        self.assertNotIn('id="glmDownloadCoefficientsBtn" class="tab ', glm)
        self.assertIn('class="glm-tabulation-option-group glm-tabulation-view-toggle"', glm)
        self.assertIn('data-glm-tabulation-view="table" data-stable-label="Table" class="app-control-button glm-tabulation-option-button', glm)
        self.assertIn('data-glm-tabulation-view="plot" data-stable-label="Plot" class="app-control-button glm-tabulation-option-button', glm)
        self.assertEqual(glm.count('data-glm-tabulation-scale="exp"'), 1)
        self.assertNotIn('data-glm-tabulation-scale="linear"', glm)
        self.assertIn('id="glmTabulationExpBtn" type="button" data-glm-tabulation-scale="exp" data-stable-label="Exp"', glm)
        self.assertIn('aria-label="Exponential scale" aria-pressed=', glm)
        self.assertIn('id="glmTabulationColorBtn" type="button" data-stable-label="Colour"', glm)
        self.assertIn('aria-label="Colour cells" aria-pressed=', glm)
        self.assertNotIn('<input id="glmTabulationColor"', glm)
        self.assertIn('id="glmExportTabulationsBtn" class="app-control-button model-busy-button glm-tabulation-export-button', glm)
        self.assertIn('class="glm-tabulation-export-icon" viewBox="0 0 24 24"', glm)
        self.assertIn('aria-label="${isExportingTabulations ? "Exporting XLSX" : "Export XLSX"}"', glm)
        self.assertNotIn('id="glmExportTabulationsBtn" class="tab ', glm)
        self.assertIn('id="glmTabulationCopyBtn" class="app-control-button glm-tabulation-copy-button"', glm)
        self.assertIn('aria-label="Copy tabulation chart" title="Copy tabulation chart" disabled', glm)
        self.assertIn('class="glm-tabulation-copy-icon" viewBox="0 0 24 24"', glm)
        self.assertIn('new window.ClipboardItem({ "image/png": blob })', glm)
        self.assertIn('pixelRatio: 2,', glm)
        self.assertNotIn("exportButton.textContent", glm)
        self.assertIn('tabulationScale = tabulationScale === "exp" ? "linear" : "exp";', glm)
        self.assertIn('tabulationColor = !tabulationColor;', glm)
        self.assertIn('button.setAttribute("aria-pressed", active ? "true" : "false");', glm)
        self.assertIn('button.disabled = view === "plot" && features.length > 2;', glm)
        self.assertIn('colorButton.setAttribute("aria-pressed", tabulationColor ? "true" : "false");', glm)
        self.assertIn('const label = isExportingTabulations ? "Exporting XLSX" : "Export XLSX";', glm)
        self.assertIn(".glm-tabulation-option-button,\n      .glm-tabulation-export-button {", glm_css)
        self.assertIn(".glm-tabulation-option-button.active {", glm_css)
        self.assertIn(".glm-tabulation-option-button::after {", glm_css)
        self.assertIn(".glm-tabulation-export-icon {", glm_css)
        self.assertIn('.glm-tabulation-export-button[aria-busy="true"] .glm-tabulation-export-icon {', glm_css)
        self.assertIn(".glm-tabulation-export-button {\n        flex: 0 0 28px;", glm_css)
        self.assertIn(".glm-tabulation-copy-button {", glm_css)
        self.assertIn(".glm-tabulation-copy-icon {", glm_css)
        self.assertIn("grid-template-columns: minmax(420px, var(--glm-tabulation-sidebar-width, 1fr)) 1px minmax(420px, 1fr);", glm_css)
        self.assertIn("#glmTabulationResizer {\n        align-self: stretch;\n        height: 100%;\n        justify-self: center;\n        width: 12px;", glm_css)
        self.assertIn("const GLM_TABULATION_X_AXIS_LABEL_DENSITY_LIMIT = 200;", glm_tabulations)
        self.assertIn("const fontSize = count > 50 ? 8 : 10;", glm_tabulations)
        self.assertIn("const rotate = count > 30 || maxLength > 10 || horizontalFootprint > slotWidth ? 65 : 0;", glm_tabulations)
        self.assertIn('lineStyle: { color: theme.line || "#cbd5e1", width: 2 },', glm_tabulations)
        self.assertIn('data: [{ yAxis: 1 }],', glm_tabulations)
        self.assertIn("xAxisFeature(data)", glm_tabulations)
        self.assertIn(".glm-tabulation-crosstab-group {\n        align-items: center;\n        display: inline-flex;\n        flex: 1 1 auto;", glm_css)
        self.assertIn(".glm-tabulation-crosstab {\n        background: var(--panel);", glm_css)
        self.assertIn("flex: 1 1 auto;\n        font-size: 12px;", glm_css)
        self.assertNotIn("max-width: 240px;", glm_css)
        self.assertNotIn("width: clamp(150px, 18vw, 240px);", glm_css)
        self.assertNotIn(".glm-tabulation-check,", glm_css)
        self.assertIn('id="glmTrainingScope" class="glm-training-scope-select" aria-label="Rows to fit"', glm)
        self.assertIn('<option value="all"', glm)
        self.assertIn('<option value="training"', glm)
        self.assertIn('<option value="training_test"', glm)
        self.assertIn('>Training + Test</option>', glm)
        self.assertIn('id="glmBuildBtn" class="tab app-control-button model-busy-button glm-build-button', glm)
        self.assertNotIn('id="glmFormulaAssistBtn" class="tab ', glm)
        self.assertIn(".glm-builder-actions .glm-builder-option-button.active,", glm_css)
        self.assertIn(".glm-training-scope-select {\n        flex: 0 0 118px;", glm_css)
        self.assertIn("margin-inline-end: var(--app-control-strip-gap);", glm_css)
        self.assertIn(".glm-builder-control-row {\n        align-items: center;\n        background: var(--sidebar-bg);\n        border-bottom: 1px solid var(--line);", glm_css)
        self.assertIn(".glm-formula-assist-drawer {\n        background: var(--sidebar-bg);", glm_css)
        self.assertIn(".glm-model-parameters-icon {", glm_css)
        self.assertIn(".glm-editor-font-controls {\n        align-items: stretch;\n        backdrop-filter: blur(2px);\n        background: color-mix(in srgb, var(--panel) 88%, transparent);\n        border: 0;", glm_css)
        self.assertIn("flex-direction: column;\n        gap: 3px;\n        padding: 2px;", glm_css)
        self.assertIn("font-weight: 400;", glm_css)
        self.assertIn(".glm-editor-copy-icon {", glm_css)
        self.assertIn(".glm-editor-shell {\n        border-top: 0;", glm_css)
        self.assertIn(".glm-coefficient-action-button {", glm_css)
        self.assertIn(".glm-coefficient-action-button:focus-visible {", glm_css)
        self.assertIn(".glm-coefficient-action-icon {", glm_css)
        self.assertNotIn("#glmFontLargerBtn {", glm_css)
        self.assertNotIn("#glmClearFormulaBtn {", glm_css)
        self.assertIn('localStorage.getItem("py_lucidum_glm_builder_panel")', glm_formula_builder)
        self.assertIn('localStorage.setItem("py_lucidum_glm_builder_panel", builderPanel);', glm_formula_builder)
        self.assertIn('toggleBuilderPanel("formula")', glm_formula_builder)
        self.assertIn('toggleBuilderPanel("parameters")', glm_formula_builder)
        self.assertIn('onCopyFormula = () => {},', glm_formula_builder)
        self.assertIn('onCopyFormula(getFormulaText())', glm_formula_builder)
        for tool, prefix in ((glm, "glm"), (gbm, "gbm")):
            self.assertIn(
                f'id="{prefix}OpenModelFolderBtn" class="app-control-button app-command-button"',
                tool,
            )
            self.assertIn(
                f'id="{prefix}RenameModelBtn" class="app-control-button app-command-button"',
                tool,
            )
            self.assertIn(
                f'id="{prefix}ActivateModelBtn" class="app-control-button app-command-button"',
                tool,
            )
            self.assertIn(
                f'id="{prefix}DeleteModelBtn" class="app-control-button app-command-button app-command-button--danger"',
                tool,
            )
            self.assertNotIn(f'id="{prefix}RenameModelBtn" class="tab ', tool)
            self.assertNotIn(f'id="{prefix}ActivateModelBtn" class="tab ', tool)
            self.assertNotIn(f'id="{prefix}DeleteModelBtn" class="danger-action ', tool)
            self.assertLess(tool.index(f'id="{prefix}OpenModelFolderBtn"'), tool.index(f'id="{prefix}RenameModelBtn"'))
            self.assertIn("state.schema?.capabilities?.open_model_folders", tool)
        self.assertIn('class="gbm-feature-main-control-strip app-control-strip"', gbm)
        self.assertIn('id="gbmFeatureSetupBtn" class="app-control-button gbm-feature-option-button gbm-feature-setup-button', gbm)
        self.assertIn('aria-label="Feature setup" title="Feature setup" aria-controls="gbmFeatureSetupPanel" aria-expanded=', gbm)
        self.assertIn('class="gbm-feature-setup-icon" viewBox="0 0 24 24"', gbm)
        self.assertIn('id="gbmFeatureSetupPanel" class="gbm-feature-setup-panel', gbm)
        self.assertIn(
            "${featureScenarioDropdownHtml(data.feature_scenarios || [], data.active_feature_scenario || null)}\n"
            "                  ${featureInteractionConstraintDropdownHtml(data.feature_interaction_groupings || [], data.active_feature_interaction_constraints || null, data.features || [], selectedShapRows)}\n"
            "                  ${featureInteractionPairsDropdownHtml(data.active_feature_interaction_constraints || null, data.features || [])}",
            gbm,
        )
        self.assertIn("Create constraint group LightGBM model .txt file(s)", gbm)
        self.assertIn("data-gbm-create-interaction-group-models", gbm)
        self.assertIn("create_feature_interaction_group_models: currentCreateFeatureInteractionGroupModels()", gbm)
        self.assertIn("if (!enabled) checkbox.checked = false;", gbm)
        self.assertNotIn("zeroShap.disabled", gbm)
        self.assertIn('Error ${escapeHtml(error)}', gbm)
        self.assertIn(">No trees</span>", gbm)
        self.assertIn('id="gbmClearFeaturesBtn" class="app-control-button app-command-button gbm-feature-command-button"', gbm)
        self.assertIn('id="gbmSelectFeaturesBtn" class="app-control-button app-command-button gbm-feature-command-button"', gbm)
        self.assertIn('class="gbm-parameter-control-cell app-control-strip-row app-control-strip--titled"', gbm)
        self.assertIn('id="gbmCopyParametersBtn" class="app-control-button gbm-parameter-copy-button"', gbm)
        self.assertIn('aria-label="Copy GBM parameters as JSON" title="Copy GBM parameters as JSON"', gbm)
        self.assertIn('class="gbm-parameter-copy-icon" viewBox="0 0 24 24"', gbm)
        self.assertIn('void copyGbmParameters();', gbm)
        self.assertIn('showClipboardToast("GBM parameters copied");', gbm)
        self.assertIn('showClipboardToast("Could not copy GBM parameters", true);', gbm)
        self.assertNotIn('id="gbmClearFeaturesBtn" class="tab ', gbm)
        self.assertNotIn('id="gbmSelectFeaturesBtn" class="tab ', gbm)
        self.assertIn('data-stable-label="${escapeHtml(featureMetricModeLabel(mode))}"', gbm)
        self.assertIn('localStorage.getItem("py_lucidum_gbm_feature_setup_open") === "true"', gbm)
        self.assertIn('localStorage.setItem("py_lucidum_gbm_feature_setup_open", String(featureSetupOpen));', gbm)
        self.assertIn('button.setAttribute("aria-expanded", featureSetupOpen ? "true" : "false");', gbm)
        self.assertIn('if (!featureSetupOpen) closeGbmFeatureToolbarMenus();', gbm)
        self.assertIn('scheduleGbmTableRedraws([featureTable, ebmGainSummaryTable]);', gbm)
        self.assertIn('label: mainEffectOnly ? "Remove main-effect-only constraint" : "Constrain to main effect only (1D)"', gbm)
        self.assertIn('label: "Add pair interaction (2D)…"', gbm)
        self.assertIn('label: `Remove ${pair.left} × ${pair.right} pairwise interaction`', gbm)
        self.assertIn('class="gbm-interaction-pair-section-title">Add pair interaction</div>', gbm)
        self.assertIn('return `Allowed pair interactions (${count})`;', gbm)
        self.assertIn('No pair interactions added', gbm)
        self.assertIn('${escapeHtml(pair.left)} × ${escapeHtml(pair.right)}', gbm)
        self.assertIn('data-gbm-remove-interaction-pair>Remove</button>', gbm)
        self.assertNotIn('data-gbm-remove-interaction-pair>×</button>', gbm)
        self.assertIn('title="Main effect only — cannot interact with other features"', gbm)
        self.assertIn('class="gbm-interaction-lock gbm-group-interaction-lock"', gbm)
        self.assertIn('return `Singleton — ${constraint.feature} only`;', gbm_tree_viewer)
        self.assertIn('return `Pairwise — ${constraint.left} × ${constraint.right}`;', gbm_tree_viewer)
        self.assertIn('return `Group — ${constraint.grouping}`;', gbm_tree_viewer)
        self.assertIn('let palette = "divergent";', gbm_tree_viewer)
        self.assertIn('values.length > 1 ? "Constraints applied:" : "Constraint applied:"', gbm_tree_viewer)
        self.assertIn('values.length ? values.join("; ") : "None"', gbm_tree_viewer)
        self.assertIn('class="gbm-tree-detail-line gbm-tree-detail-constraint" title=', gbm_tree_viewer)
        self.assertIn(".gbm-feature-metric-option::after {", gbm_css)
        self.assertIn(".gbm-feature-actions {\n        flex: 1 1 260px;\n        flex-wrap: nowrap;\n        justify-content: flex-end;", gbm_css)
        self.assertIn(".gbm-feature-option-button.active {", gbm_css)
        self.assertIn(".gbm-feature-setup-button {\n        margin-left: 28px;", gbm_css)
        self.assertIn(".gbm-feature-setup-icon {", gbm_css)
        self.assertIn(".gbm-parameter-copy-button,", gbm_css)
        self.assertIn(".gbm-parameter-copy-icon,", gbm_css)
        self.assertIn(".gbm-feature-setup-panel {\n        background: var(--sidebar-bg);", gbm_css)
        self.assertIn(".gbm-feature-setup-controls {", gbm_css)
        self.assertIn(".gbm-feature-menu-button:focus-visible {", gbm_css)
        self.assertIn(".gbm-interaction-constraint-row.disabled {", gbm_css)
        self.assertIn(".gbm-interaction-pair-section-title {", gbm_css)
        self.assertIn("@media (max-width: 520px) {", gbm_css)
        self.assertIn(".gbm-feature-context-menu-item:disabled {", gbm_css)
        self.assertIn(".gbm-tree-detail-constraint {\n        pointer-events: auto;", gbm_css)
        self.assertIn(".gbm-sample-status-detail {\n        display: block;\n        font-weight: 400;", gbm_css)
        self.assertIn("#gbmShapRows > .gbm-shap-label,\n      #gbmTrainingMode > .gbm-shap-label {\n        color: var(--sidebar-muted);", gbm_css)
        self.assertIn(".gbm-shap-option:hover,\n      .gbm-shap-option:has(input:focus-visible) {", gbm_css)
        self.assertIn(".gbm-shap-option:has(input:checked) {\n        background: transparent;\n        border: 0;\n        color: var(--accent);", gbm_css)
        self.assertNotIn(".gbm-icon-action-button {", gbm_css)
        self.assertIn('class="gbm-evaluation-control-strip app-control-strip"', gbm)
        self.assertIn('id="gbmFeatureResizer"', gbm)
        self.assertIn('id="gbmParameterControlDivider"', gbm)
        self.assertNotIn('id="gbmParameterControlResizer"', gbm)
        self.assertIn('id="gbmEvaluationResizer"', gbm)
        self.assertIn('id="gbmEvaluationViewMode" class="gbm-evaluation-view-mode" role="group" aria-label="Evaluation Log controls"', gbm)
        self.assertIn('id="gbmEvaluationTailBtn" class="app-control-button gbm-evaluation-option-button', gbm)
        self.assertIn('aria-pressed="${String(tailActive)}">Zoom tail</button>', gbm)
        self.assertIn('id="gbmEvaluationCopyBtn" class="app-control-button gbm-evaluation-copy-button"', gbm)
        self.assertIn('aria-label="Copy Evaluation Log chart" title="Copy Evaluation Log chart"', gbm)
        self.assertIn('class="gbm-evaluation-copy-icon" viewBox="0 0 24 24"', gbm)
        self.assertNotIn('name="gbmEvaluationViewMode"', gbm)
        self.assertIn('tailButton.setAttribute("aria-pressed", String(tailActive));', gbm)
        self.assertIn('void evaluationChart.copyToClipboard();', gbm)
        self.assertIn(".gbm-evaluation-option-button.active {", gbm_css)
        self.assertIn(".gbm-evaluation-copy-icon {", gbm_css)
        self.assertIn('showClipboardToast("Evaluation Log chart image copied");', gbm_evaluation_chart)
        self.assertIn('new window.ClipboardItem({ "image/png": blob })', gbm_evaluation_chart)
        self.assertGreaterEqual(gbm.count('role="separator"'), 3)
        self.assertGreaterEqual(gbm.count('aria-orientation="vertical"'), 2)
        self.assertIn('aria-orientation="horizontal"', gbm)
        self.assertIn("createGbmFeatureParameterLayout", gbm_feature_controls)
        self.assertIn("const GBM_DIVIDER_TRACK_WIDTH = 1;", gbm_feature_controls)
        self.assertNotIn("resizeParameter", gbm_feature_controls)
        self.assertIn("#gbmParameterGrid .tabulator-row .tabulator-cell:not(:has(~ .tabulator-cell))", gbm_css)
        self.assertIn("clamp(280px, 42dvh, 360px)", gbm_css)
        self.assertIn("clamp(240px, 38dvh, 320px)", gbm_css)
        self.assertIn("clamp(220px, 35dvh, 300px)", gbm_css)
        self.assertIn(".gbm-feature-main-control-strip,\n        .gbm-feature-control-layout {\n          display: contents;", gbm_css)
        self.assertIn(
            "#gbmShapRows,\n"
            "        #gbmTrainingMode {\n"
            "          align-items: center;\n"
            "          column-gap: 12px;\n"
            "          display: grid;\n"
            "          grid-template-columns: max-content minmax(0, 1fr);",
            gbm_css,
        )
        self.assertIn("#gbmTrainingMode .gbm-mode-option {\n          min-width: 60px;", gbm_css)
        self.assertIn(
            ".gbm-feature-column-resizer,\n"
            "        .gbm-parameter-control-divider,\n"
            "        .gbm-evaluation-resizer {\n"
            "          display: none !important;",
            gbm_css,
        )
        self.assertIn('resizer.addEventListener("pointercancel", finishDrag);', gbm_feature_controls)
        self.assertIn('resizer.addEventListener("keydown"', gbm_feature_controls)
        self.assertIn('resizer.setAttribute("aria-valuenow"', gbm_feature_controls)
        self.assertIn('class="spec-file-row app-control-strip"', specs)
        self.assertIn('class="spec-save-button app-control-button"', specs)

    def test_responsive_sidebar_divider_is_owned_by_shared_shell(self) -> None:
        static_root = Path(__file__).resolve().parents[1] / "src/py_lucidum/static"
        shell = (static_root / "styles/shell.css").read_text(encoding="utf-8")
        gbm = (static_root / "styles/gbm.css").read_text(encoding="utf-8")

        self.assertIn("@media (min-width: 641px) and (max-width: 1180px)", shell)
        self.assertIn(
            "grid-template-columns: var(--sidebar-width, 280px) 1px minmax(0, 1fr);",
            shell,
        )
        self.assertNotIn("grid-template-columns: var(--sidebar-width, 280px) 3px 1fr;", gbm)

    def test_uk_map_control_strip_markup_starts_expanded(self) -> None:
        static_root = Path(__file__).resolve().parents[1] / "src/py_lucidum/static"
        index = (static_root / "index.html").read_text(encoding="utf-8")
        main_source = (static_root / "app/main.js").read_text(encoding="utf-8")
        map_source = (static_root / "app/uk-map-tool.js").read_text(encoding="utf-8")
        map_styles = (static_root / "styles/uk-map.css").read_text(encoding="utf-8")

        self.assertIn(
            'id="mapToolbar" class="map-settings-strip app-control-strip hidden"',
            index,
        )
        self.assertIn(
            'id="mapToolbarScroll" class="map-settings-scroll toolbar app-settings-strip"',
            index,
        )
        self.assertIn('role="radiogroup" aria-label="Base map layer"', index)
        self.assertIn('role="radiogroup" aria-label="Map resolution"', index)
        self.assertIn('role="group" aria-label="Choropleth colour scheme"', index)
        self.assertNotIn('<h3>Base</h3>', index)
        self.assertNotIn('<h3>Resolution</h3>', index)
        self.assertNotIn('<h3>Palette</h3>', index)
        self.assertEqual(index.count('name="baseMap"'), 6)
        self.assertEqual(index.count('name="mapLevel"'), 3)
        self.assertEqual(index.count('class="map-palette-button'), 3)
        self.assertEqual(index.count('data-map-smoothing="'), 6)
        self.assertNotIn('id="mapSmoothingSaveControl"', index)
        self.assertIn('class="map-smoothing-heading"', index)
        self.assertIn('id="mapSaveSmoothingBtn"', index)
        self.assertIn('aria-busy="false">-&gt; .parquet</button>', index)
        self.assertIn('aria-label="Save N1-N5 sector smoothing Parquet"', index)
        self.assertEqual(index.count('data-map-line-weight="'), 3)
        self.assertIn('id="mapSmoothing" class="segmented map-smoothing-mode"', index)
        self.assertNotIn('id="mapSmoothingValue"', index)
        self.assertIn('id="mapLineWeight" class="segmented map-border-mode"', index)
        self.assertNotIn('id="mapLineWeightValue"', index)
        self.assertIn('<h3>Strength</h3>', index)
        self.assertIn('id="mapOpacity" class="segmented map-strength-mode"', index)
        self.assertEqual(index.count('data-map-opacity="'), 3)
        self.assertIn('data-map-opacity="0.2" data-stable-label="Faint"', index)
        self.assertIn('data-map-opacity="0.6" data-stable-label="Medium"', index)
        self.assertIn('data-map-opacity="1" data-stable-label="Solid"', index)
        self.assertNotIn('id="mapOpacity" type="range"', index)
        self.assertNotIn('id="mapOpacityValue"', index)
        self.assertIn('data-map-line-weight="0" data-stable-label="Off"', index)
        self.assertIn('data-map-line-weight="1" data-stable-label="Thin"', index)
        self.assertIn('data-map-line-weight="3" data-stable-label="Bold"', index)
        self.assertIn('.map-smoothing-mode {', map_styles)
        self.assertIn('.map-smoothing-heading {', map_styles)
        self.assertIn('.map-sector-save-button {', map_styles)
        self.assertIn('api("/api/uk-map/sector-smoothing"', map_source)
        self.assertIn('showClipboardToast(error.message, true);', map_source)
        self.assertIn('.map-strength-mode button {', map_styles)
        self.assertIn('--map-slider-width: 108px;', map_styles)
        self.assertIn('padding-inline-start: var(--app-control-button-padding-inline);', map_styles)
        self.assertIn('<h3>Search</h3>', index)
        self.assertNotIn('<h3>Postcode</h3>', index)
        self.assertIn('padding: 0 5px 0 0;', map_styles)
        self.assertIn('.uk-map .maplibregl-popup-tip {\n        background: transparent;', map_styles)
        self.assertIn('.uk-map .maplibre-tooltip .maplibregl-popup-content {\n        padding: 6px 10px;', map_styles)
        self.assertIn('.uk-map .maplibre-tooltip .maplibregl-popup-tip {\n        border-width: 7px;', map_styles)
        self.assertIn('border-bottom-color: var(--panel);', map_styles)
        self.assertIn('border-top-color: var(--panel);', map_styles)
        self.assertIn('border-right-color: var(--panel);', map_styles)
        self.assertIn('border-left-color: var(--panel);', map_styles)
        self.assertNotIn('--map-extreme-color', map_styles)
        self.assertNotIn('class="map-extreme-label"', index)
        self.assertIn('padding-inline: 6px;', map_styles)
        self.assertIn('--map-info-separator-spacing: 0.35em;', map_styles)
        self.assertGreaterEqual(map_styles.count('margin-left: var(--map-info-separator-spacing);'), 2)
        self.assertGreaterEqual(map_styles.count('margin-right: var(--map-info-separator-spacing);'), 2)
        self.assertIn('data-palette="spectral" title="Macaw"', index)
        self.assertIn('<strong id="mapControlMetric" hidden></strong>', index)
        self.assertIn('id="mapMetricSeparator" class="map-info-separator" aria-hidden="true" hidden>·</span>', index)
        self.assertEqual(index.count('class="map-info-separator"'), 2)
        self.assertIn('el("mapMetricSeparator").hidden = !kpiName;', map_source)
        self.assertIn('id="mapInfoStrip" class="map-info-strip hidden"', index)
        self.assertIn(
            'id="mapToolbarStatus" class="map-toolbar-status" aria-label="Map status"', index,
        )
        self.assertNotIn('id="mapFloatingControl"', index)
        self.assertIn('mapToolbarCollapsed: false', main_source)
        self.assertIn('aria-controls="mapToolbar mapInfoStrip" aria-expanded="true"', map_source)
        self.assertLess(map_source.index('id="mapControlReset"'), map_source.index('.id = "mapZoomIn"'))
        self.assertIn('bindSettingsStripOverflowCue(el("mapToolbarScroll"));', map_source)
        self.assertIn('scheduleMapViewportSync({ mode: "preserve" });', map_source)
        self.assertNotIn('setupMapFloatingControlDrag', map_source)
        self.assertIn('height: var(--app-control-strip-height);', map_styles)
        self.assertIn('flex: 0 0 16px;', map_styles)
        self.assertIn('height: 16px;', map_styles)
        self.assertIn('[toolbar, infoStrip].forEach', map_source)
        self.assertIn('grid-template-rows: 32px 17px;', map_styles)
        self.assertIn('justify-content: flex-start;', map_styles)
        self.assertIn('padding: 0 2px;', map_styles)
        self.assertIn('gap: 2px;', map_styles)
        self.assertNotIn('box-shadow: inset 0 -2px 0 var(--accent);', map_styles)
        self.assertIn('border-radius: 0;', map_styles)
        self.assertIn('id="mapLegend" class="map-legend hidden collapsed"', index)
        self.assertIn(
            'id="mapLegendToggle" class="map-legend-toggle" type="button" title="Expand legend" '
            'aria-label="Expand legend" aria-controls="mapLegendBody" aria-expanded="false"',
            index,
        )
        self.assertIn(
            'id="mapMatchLiveStatus" class="map-match-live-status" role="status" aria-live="polite"',
            index,
        )

    def test_monitor_entrypoints_disable_cache(self) -> None:
        _, body = self.assert_no_store("/monitor")
        html = body.decode("utf-8")

        self.assertIn("<title>lucidum monitor · sample.csv</title>", html)
        self.assert_no_store("/static/monitor.css")
        self.assert_no_store("/static/monitor.js")


















































    def test_uk_map_static_assets_disable_cache(self) -> None:
        self.assert_no_store("/tools/uk-map/static/icons/UK.png")
        self.assert_no_store("/tools/uk-map/static/icons/map-controls/base-blank.png")
        self.assert_no_store("/tools/uk-map/static/geodata/sector_adjacency.json")

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
