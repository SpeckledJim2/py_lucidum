from __future__ import annotations

import asyncio
import json
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
    CSS_MODULE_PATHS = [
        "/static/styles/foundations.css",
        "/static/styles/shell.css",
        "/static/styles/controls.css",
        "/static/styles/dataset-viewer.css",
        "/static/styles/line-bar.css",
        "/static/styles/histogram.css",
        "/static/styles/uk-map.css",
        "/static/styles/column-profile.css",
        "/static/styles/model-shell.css",
        "/static/styles/specifications.css",
        "/static/styles/glm.css",
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
            "/static/app/dataset-viewer-tool.js",
            "/static/app/column-profile-tool.js",
            "/static/app/line-bar-tool.js",
            "/static/app/histogram-tool.js",
            "/static/app/uk-map-tool.js",
            "/static/app/glm-tool.js",
            "/static/app/glm-formula-assist.js",
            "/static/app/glm-formula-builder.js",
            "/static/app/glm-model-navigator.js",
            "/static/app/glm-tabulations.js",
            "/static/app/specifications-tool.js",
            "/static/app/shared/api.js",
            "/static/app/shared/format.js",
            "/static/app/shared/model-ui.js",
            "/static/app/shared/schema.js",
            "/static/app/shared/tabulator.js",
            "/static/app/shared/timing.js",
            "/static/app/gbm-tool.js",
            "/static/app/gbm-evaluation-chart.js",
            "/static/app/gbm-feature-parameter-controls.js",
            "/static/app/gbm-model-navigator.js",
            "/static/app/gbm-tab-orchestration.js",
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

    def shared_model_ui_source(self, names: list[str] | tuple[str, ...]) -> str:
        js = self.assert_no_store("/static/app/shared/model-ui.js")[1].decode("utf-8")
        aliases = {
            "formatModelCreated": "sharedFormatModelCreated",
            "formatModelMetric": "sharedFormatModelMetric",
            "modelCreatedSort": "sharedModelCreatedSort",
            "modelNumberOrNull": "sharedModelNumberOrNull",
        }
        parts = []
        if "formatModelCreated" in names:
            parts.append('const MODEL_MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];')
        for name in names:
            source = self.js_function_source(js, name)
            alias = aliases.get(name)
            if alias:
                source = source.replace(f"function {name}", f"function {alias}", 1)
                for dependency, dependency_alias in aliases.items():
                    if dependency != name:
                        source = source.replace(f"{dependency}(", f"{dependency_alias}(")
            parts.append(source)
        return "\n".join(parts)

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

    def test_line_bar_x_fallback_preserves_feature_name_when_source_changes(self) -> None:
        js = self.assert_no_store("/static/app/main.js")[1].decode("utf-8")
        helper = "\n".join(
            [
                self.js_function_source(js, "lineBarFeatureSourceForName"),
                self.js_function_source(js, "syncLineBarXFallback"),
            ]
        )
        script = helper + """
let columns = [];
const state = { source: "glm:model-b:predictions", x: "Segment", xSource: "glm:model-a:predictions" };

function lineBarFeatureColumns() {
  return columns;
}

function lineBarColumnSourceId(column) {
  return column?.source_id || state.source || "dataset";
}

function lineBarColumnExists(name, sourceId = "") {
  const columnName = String(name || "");
  if (!columnName) return false;
  return lineBarFeatureColumns().some((column) => (
    column.name === columnName && (!sourceId || lineBarColumnSourceId(column) === sourceId)
  ));
}

columns = [
  { name: "Age", source_id: "glm:model-b:predictions" },
  { name: "Segment", source_id: "glm:model-b:predictions" },
  { name: "glm_prediction", source_id: "glm:model-b:predictions" },
];
syncLineBarXFallback();
if (state.x !== "Segment" || state.xSource !== "glm:model-b:predictions") {
  throw new Error(`expected current-source feature preservation, got ${state.x} from ${state.xSource}`);
}

state.source = "gbm:model-b:predictions";
state.x = "Segment";
state.xSource = "gbm:model-a:predictions";
columns = [
  { name: "Age", source_id: "gbm:model-b:predictions" },
  { name: "Segment", source_id: "dataset" },
  { name: "gbm_prediction", source_id: "gbm:model-b:predictions" },
];
syncLineBarXFallback();
if (state.x !== "Segment" || state.xSource !== "dataset") {
  throw new Error(`expected any-source feature preservation, got ${state.x} from ${state.xSource}`);
}
if (state.source !== "dataset") {
  throw new Error(`expected dataset source reset, got ${state.source}`);
}

state.source = "dataset";
state.x = "Missing";
state.xSource = "old";
columns = [
  { name: "Age", source_id: "dataset" },
  { name: "Segment", source_id: "dataset" },
];
if (lineBarFeatureSourceForName("Missing", "dataset") !== "") {
  throw new Error("missing feature should not resolve to the current source");
}
syncLineBarXFallback();
if (state.x !== "Age" || state.xSource !== "dataset") {
  throw new Error(`expected first-feature fallback, got ${state.x} from ${state.xSource}`);
}

state.x = "Missing";
state.xSource = "old";
columns = [];
syncLineBarXFallback();
        if (state.x !== null || state.xSource !== "") {
          throw new Error(`expected empty fallback, got ${state.x} from ${state.xSource}`);
        }
        """
        self.run_node_script(script)

    def test_line_bar_expected_selection_array_is_ordered_and_capped(self) -> None:
        js = self.assert_no_store("/static/app/main.js")[1].decode("utf-8")
        helper = "\n".join(
            self.js_function_source(js, name)
            for name in [
                "expectedSelectionKey",
                "expectedOptionForSelection",
                "expectedSelectionFromOption",
                "normaliseExpectedSelections",
                "syncExpectedSelectToSelections",
                "expectedSelections",
                "setExpectedSelections",
            ]
        )
        script = helper + """
const state = { source: "dataset", expectedSelections: [] };
const expectedSelect = {
  value: "",
  options: [
    { value: "", disabled: false, dataset: {}, selected: false },
    { value: "glm_prediction", disabled: false, dataset: { sourceId: "glm:model:predictions", metricKind: "prediction" }, selected: false },
    { value: "gbm_prediction", disabled: false, dataset: { sourceId: "gbm:model:predictions", metricKind: "prediction" }, selected: false },
    { value: "Expected", disabled: false, dataset: { sourceId: "dataset", metricKind: "metric" }, selected: false },
    { value: "Ignored", disabled: false, dataset: { sourceId: "dataset", metricKind: "metric" }, selected: false },
  ],
};
function el(id) {
  if (id !== "expectedNumerator") throw new Error(id);
  return expectedSelect;
}
const kept = normaliseExpectedSelections([
  { value: "glm_prediction", sourceId: "glm:model:predictions" },
  { value: "gbm_prediction", sourceId: "gbm:model:predictions" },
  { value: "Expected", sourceId: "dataset" },
]);
if (kept.length !== 2) throw new Error(`expected cap at 2, got ${kept.length}`);
if (kept[0].value !== "glm_prediction" || kept[1].value !== "gbm_prediction") throw new Error("expected selection order was not preserved");
if (kept[0].metricKind !== "prediction" || kept[1].sourceId !== "gbm:model:predictions") throw new Error("expected source metadata was not preserved");
const ok = setExpectedSelections(kept);
if (!ok) throw new Error("expected exact selections to restore");
if (state.expectedSelections.length !== 2) throw new Error("expected two selected rows");
if (!expectedSelect.options[1].selected) throw new Error("first expected selection should mirror to hidden select");
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

    def test_line_bar_feature_columns_keep_dataset_fields_on_dataset_source(self) -> None:
        js = self.assert_no_store("/static/app/main.js")[1].decode("utf-8")
        helper = self.js_function_source(js, "lineBarFeatureColumns")
        script = helper + """
const LINE_BAR_RATIO_COLUMN = "gbm_to_glm_ratio";
const state = { source: "gbm:model:predictions" };
function currentDataSource() { return { kind: "gbm_predictions" }; }
function dataSourceColumns(sourceId) {
  if (sourceId !== "dataset") throw new Error(`unexpected dataset source ${sourceId}`);
  return [
    { name: "PostcodeArea", kind: "categorical" },
    { name: "PostcodeSector", kind: "categorical" },
    { name: "Actual", kind: "numeric" },
  ];
}
function sourceColumns() {
  return [
    { name: "PostcodeArea", kind: "categorical" },
    { name: "Actual", kind: "numeric" },
    { name: "gbm_prediction", kind: "numeric" },
  ];
}
function isModelPredictionColumn(column) {
  return ["gbm_prediction", "glm_prediction"].includes(String(column?.name || ""));
}
function isGbmShapValueColumn(column) { return column?.source_role === "gbm_shap_value"; }
function activeModelRatioColumns() { return []; }
function activePredictionColumns() {
  return [{ name: "gbm_prediction", kind: "numeric", source_id: "gbm:model:predictions" }];
}
const columns = lineBarFeatureColumns();
const postcodeArea = columns.find((column) => column.name === "PostcodeArea");
if (!postcodeArea || postcodeArea.source_id !== "dataset") {
  throw new Error(`expected PostcodeArea to stay on dataset, got ${postcodeArea?.source_id}`);
}
if (columns.some((column) => column.name === "PostcodeArea" && column.source_id === "gbm:model:predictions")) {
  throw new Error("raw dataset feature leaked onto model source");
}
const prediction = columns.find((column) => column.name === "gbm_prediction");
if (!prediction || prediction.source_id !== "gbm:model:predictions") {
  throw new Error("model prediction column lost its model source");
}
"""
        self.run_node_script(script)

    def test_line_bar_actual_prediction_does_not_switch_global_source(self) -> None:
        js = self.assert_no_store("/static/app/main.js")[1].decode("utf-8")
        helper = "\n".join(
            [
                self.js_function_source(js, "actualSelectionSourceId"),
                self.js_function_source(js, "syncActualSourceFromSelection"),
            ]
        )
        script = helper + """
const state = { source: "dataset" };
let invalidated = false;
let selectedOption = { dataset: { sourceId: "gbm:model:predictions", metricKind: "prediction" } };
function el(id) {
  if (id !== "actualNumerator") throw new Error(id);
  return { selectedOptions: [selectedOption] };
}
function invalidateLineBarDateBucketSuggestion() { invalidated = true; }
if (syncActualSourceFromSelection()) throw new Error("prediction actual should not switch global source");
if (state.source !== "dataset" || invalidated) throw new Error("prediction actual mutated source state");

selectedOption = { dataset: { sourceId: "gbm:model:shap_long", metricKind: "shap" } };
if (!syncActualSourceFromSelection()) throw new Error("SHAP actual should switch to SHAP source");
if (state.source !== "gbm:model:shap_long" || !invalidated) throw new Error("SHAP source switch failed");
"""
        self.run_node_script(script)

    def test_line_bar_chart_request_keeps_dataset_base_for_prediction_response(self) -> None:
        js = self.assert_no_store("/static/app/line-bar-tool.js")[1].decode("utf-8")
        helper = self.js_function_source(js, "buildChartRequest")
        script = helper + """
const state = {
  schema: {},
  source: "dataset",
  xSource: "dataset",
  x: "PostcodeArea",
  sort: "alpha",
  lowGroup: "0",
  bandWidth: "0",
  quantileMode: "off",
  dateBucket: "none",
  transform: "none",
  partialDependence: "none",
  activeFilter: "",
  sigma: "0",
};
let selected = { name: "PostcodeArea", kind: "categorical", source_id: "dataset" };
function selectedColumn() { return selected; }
function isNumericKind(kind) { return kind === "numeric" || kind === "integer"; }
function isModelPredictionColumn(column) {
  return ["gbm_prediction", "glm_prediction"].includes(String(column?.name || ""));
}
function currentBandFeatureKey() { return "band"; }
function normalizeBandWidthForQuantiles() { throw new Error("unexpected quantile normalization"); }
function requestBandSuggestionForSelectedColumn() { throw new Error("unexpected banding request"); }
function currentDateBucketFeatureKey() { return "date"; }
function requestDateBucketSuggestionForSelectedColumn() { throw new Error("unexpected date bucket request"); }
function setGroupMeta() { throw new Error("unexpected pending state"); }
function selectedPartialDependenceMode() { return "none"; }
function selectedFeatureBase() { return ""; }
function currentResponses() {
  return [{ label: "gbm_prediction", numerator: "gbm_prediction", source: "gbm:model:predictions" }];
}
function el(id) {
  if (id !== "denominator") throw new Error(id);
  return { value: "__none__" };
}
let request = buildChartRequest();
if (request.source !== "dataset") throw new Error(`expected dataset base source, got ${request.source}`);
if ("xSource" in request) throw new Error(`dataset x-axis should not carry xSource: ${request.xSource}`);
if (request.responses[0].source !== "gbm:model:predictions") throw new Error("prediction response source was not retained");

selected = { name: "gbm_prediction", kind: "numeric", source_id: "gbm:model:predictions" };
state.x = "gbm_prediction";
state.xSource = "gbm:model:predictions";
state.bandWidth = "1";
state.bandFeature = "band";
request = buildChartRequest();
if (request.source !== "dataset" || request.xSource !== "gbm:model:predictions") {
  throw new Error("model x-axis field should retain model xSource while keeping dataset base");
}
"""
        self.run_node_script(script)

    def test_uk_map_postcode_availability_helper(self) -> None:
        module = Path("src/py_lucidum/static/app/uk-map-tool.js").resolve().as_uri()
        script = f"""
import {{ ukMapPostcodeAvailability }} from "{module}";
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
  table: {{ getSelectedData: () => [{{ model_id: "m3" }}, {{ model_id: "m3" }}, {{ model_id: "" }}] }},
  fallbackSelector: "#gbmModelFallback [data-gbm-model-row]",
  rowDataKey: "gbmModelRow",
}});
if (tableIds.join(",") !== "m3") throw new Error(`table ids failed: ${{tableIds.join(",")}}`);
restoreModelSelection({{ table: null, fallbackSelector: "#gbmModelFallback [data-gbm-model-row]", rowDataKey: "gbmModelRow", ids: ["m2"] }});
if (rowA.attributes["aria-selected"] !== "false" || rowB.attributes["aria-selected"] !== "true") throw new Error("restore fallback failed");

const activate = {{}};
const rename = {{}};
const deleteButton = {{}};
syncModelActionButtons({{ selectedCount: 1, disabled: false, activate, rename, deleteButton }});
if (activate.disabled || rename.disabled || deleteButton.disabled) throw new Error("enabled action state failed");
syncModelActionButtons({{ selectedCount: 2, disabled: true, activate, rename, deleteButton }});
if (!activate.disabled || !rename.disabled || !deleteButton.disabled) throw new Error("disabled action state failed");

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

    def test_gbm_shap_selection_helper_maps_active_model_metadata(self) -> None:
        js = self.assert_no_store("/static/app/gbm-tool.js")[1].decode("utf-8")
        script = self.js_function_source(js, "gbmShapSelectionValue") + """
const cases = [
  ["startup", { active_model_id: "", models: [] }, "100k"],
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
        script = self.shared_model_ui_source(["modelNumberOrNull", "formatModelMetric"]) + "\n"
        script += "\n".join(self.js_function_source(js, name) for name in helpers) + """
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

    def test_glm_model_detail_label_only_includes_family_and_aic(self) -> None:
        js = self.assert_no_store("/static/app/glm-tool.js")[1].decode("utf-8")
        helpers = ["glmModelDetailLabel", "modelNumberOrNull", "formatModelMetric"]
        script = self.shared_model_ui_source(["modelNumberOrNull", "formatModelMetric"]) + "\n"
        script += "\n".join(self.js_function_source(js, name) for name in helpers) + """
const model = {
  family: "gamma",
  diagnostics: { aic: 747117.3116, deviance: 20984.7818 },
  training_rows: 50000,
};
const actual = glmModelDetailLabel(model);
if (actual !== "gamma · AIC 747,117.3116") throw new Error(actual);
"""
        self.run_node_script(script)

    def test_glm_family_parameter_guard(self) -> None:
        js = self.assert_no_store("/static/app/glm-formula-builder.js")[1].decode("utf-8")
        script = "const config = { families: [] };\nfunction getFamilies() { return config.families; }\n" + "\n".join(self.js_function_source(js, name) for name in ["familyParameterConfig", "validateFamilyParameter"]) + """
if (validateFamilyParameter("normal", "") !== "") throw new Error("normal should not validate parameter");
if (validateFamilyParameter("tweedie", "1") !== "") throw new Error("lower bound failed");
if (validateFamilyParameter("tweedie", "2") !== "") throw new Error("upper bound failed");
if (!validateFamilyParameter("tweedie", "0.99")) throw new Error("low invalid failed");
if (!validateFamilyParameter("tweedie", "2.01")) throw new Error("high invalid failed");
if (!validateFamilyParameter("tweedie", "abc")) throw new Error("non-numeric invalid failed");
if (validateFamilyParameter("negative.binomial", "1") !== "") throw new Error("negative binomial valid failed");
if (!validateFamilyParameter("negative.binomial", "0")) throw new Error("negative binomial invalid failed");
"""
        self.run_node_script(script)

    def test_glm_regularization_parameter_guard(self) -> None:
        js = self.assert_no_store("/static/app/glm-formula-builder.js")[1].decode("utf-8")
        script = self.js_function_source(js, "validateRegularizationParameter") + """
if (validateRegularizationParameter({ mode: "none" }) !== "") throw new Error("none should not validate parameter");
if (validateRegularizationParameter({ mode: "auto" }) !== "") throw new Error("auto should not validate parameter");
if (validateRegularizationParameter({ mode: "manual", alpha: "0.1", l1_ratio: 0.5 }) !== "") throw new Error("manual valid failed");
if (!validateRegularizationParameter({ mode: "manual", alpha: "0", l1_ratio: 0.5 })) throw new Error("alpha invalid failed");
if (!validateRegularizationParameter({ mode: "manual", alpha: "0.1", l1_ratio: 1.5 })) throw new Error("mix invalid failed");
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

    def test_glm_coefficient_table_pvalue_styling_contract(self) -> None:
        js = self.assert_no_store("/static/app/glm-tool.js")[1].decode("utf-8")
        css = self.assert_no_store("/static/styles/glm.css")[1].decode("utf-8")
        self.assertIn('<th class="numeric">estimate</th>', js)
        self.assertIn('<th class="numeric">std.error</th>', js)
        self.assertIn('<th class="numeric">p.value</th>', js)
        self.assertIn('class="${penalized ? "" : glmCoefficientPValueClass(row.p_value)}"', js)
        self.assertIn('${penalized ? "" : escapeHtml(formatModelMetric(row.std_error))}', js)
        self.assertIn("#glmCoefficientTable tbody tr.glm-coefficient-pvalue-low", css)
        self.assertIn("#glmCoefficientTable tbody tr.glm-coefficient-pvalue-medium", css)
        self.assertIn("#glmCoefficientTable tbody tr.glm-coefficient-pvalue-high", css)
        script = self.shared_model_ui_source(["modelNumberOrNull"]) + "\n"
        script += "\n".join(self.js_function_source(js, name) for name in ["modelNumberOrNull", "glmCoefficientPValueClass"]) + """
const cases = [
  [0.0099, "glm-coefficient-pvalue-low"],
  [0.01, "glm-coefficient-pvalue-medium"],
  [0.05, "glm-coefficient-pvalue-medium"],
  [0.0501, "glm-coefficient-pvalue-high"],
  [null, ""],
  ["abc", ""],
];
for (const [value, expected] of cases) {
  const actual = glmCoefficientPValueClass(value);
  if (actual !== expected) {
    throw new Error(`${value}: expected ${expected}, got ${actual}`);
  }
}
"""
        self.run_node_script(script)

    def test_gbm_training_ready_badge_label_reports_grid_progress(self) -> None:
        js = self.assert_no_store("/static/app/gbm-tool.js")[1].decode("utf-8")
        helpers = ["modelNumberOrNull", "formatTrainingBadgeCount", "gbmTrainingReadyBadgeLabel"]
        script = self.shared_model_ui_source(["modelNumberOrNull"]) + "\n"
        script += "\n".join(self.js_function_source(js, name) for name in helpers) + """
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

    def test_line_bar_favourites_static_contract(self) -> None:
        html = self.root_html_for_tools(["line_bar"])
        css = self.app_css_contract()
        main_js = self.app_js_contract()
        line_bar_js = self.assert_no_store("/static/app/line-bar-tool.js")[1].decode("utf-8")
        favourites_py = (Path(__file__).resolve().parents[1] / "src/py_lucidum/tools/line_bar/favourites.py").read_text(encoding="utf-8")

        self.assertIn('class="section sidebar-favourites-section sidebar-accordion-section sidebar-section-closed"', html)
        self.assertIn('class="section sidebar-kpi-section sidebar-accordion-section sidebar-section-closed"', html)
        self.assertIn('id="favouritesCollapseBtn"', html)
        self.assertIn('id="sidebarFavouriteAddBtn"', html)
        self.assertIn('id="sidebarFavouriteMenuBtn"', html)
        self.assertIn('id="favouritesSelect"', html)
        self.assertIn('id="sidebarFavouritePopover"', html)
        self.assertIn('class="section sidebar-kpi-section sidebar-accordion-section sidebar-section-closed"', html)
        self.assertIn('id="kpiCollapseBtn"', html)
        self.assertIn('id="kpiSelect"', html)
        self.assertNotIn('id="lineBarFavouritesControl"', html)
        self.assertNotIn('id="lineBarFavouriteSelect"', html)
        favourites_start = html.index('data-sidebar-section="favourites"')
        favourites_header = html.index('<div class="sidebar-favourites-header-row">', favourites_start)
        favourites_controls = html.index('id="sidebarFavouritesControls"', favourites_header)
        favourites_meta = html.index('id="favouritesSelectedMeta"', favourites_header)
        favourites_body = html.index('<div class="sidebar-section-body">', favourites_header)
        self.assertLess(html.index('id="favouritesCollapseBtn"', favourites_header), favourites_controls)
        self.assertLess(favourites_controls, favourites_meta)
        self.assertLess(favourites_meta, favourites_body)
        toolbar_start = html.index('id="lineBarToolbar"')
        first_control = html.index('class="control', toolbar_start)
        self.assertNotIn("Favourite", html[toolbar_start:first_control])
        self.assertIn(".sidebar-favourites-header-row {\n        display: flex;\n        align-items: center;\n        gap: 2px;", css)
        self.assertIn(".sidebar-favourites-header-row .favourites-header.sidebar-section-header {\n        flex: 0 0 auto;\n        width: auto;", css)
        self.assertIn(".sidebar-favourites-header-row:hover .favourites-header.sidebar-section-header,\n      .sidebar-favourites-header-row:focus-within .favourites-header.sidebar-section-header,\n      .sidebar-favourites-header-row:hover .favourites-selected-meta,\n      .sidebar-favourites-header-row:focus-within .favourites-selected-meta {\n        color: var(--accent);", css)
        self.assertIn(".sidebar-favourites-controls", css)
        self.assertIn(".line-bar-favourite-popover", css)
        self.assertIn("gap: 0;\n        grid-template-columns: max-content max-content;", css)
        self.assertIn("grid-template-columns: max-content max-content;", css)
        self.assertIn("opacity: 0;\n        padding: 0;\n        pointer-events: none;\n        transition: opacity 120ms ease;", css)
        self.assertIn(".sidebar-favourites-header-row:hover .sidebar-favourites-controls,\n      .sidebar-favourites-header-row:focus-within .sidebar-favourites-controls {\n        opacity: 1;\n        pointer-events: auto;", css)
        self.assertIn(".sidebar-favourite-icon-button.line-bar-favourite-icon-button {\n        width: 18px;\n        height: 20px;\n        min-height: 20px;\n        border: 0;\n        border-radius: 0;\n        background: transparent;\n        color: var(--accent);", css)
        self.assertIn("font-weight: 400;\n        line-height: 1;\n        transform: translateY(-1px);", css)
        self.assertIn(".sidebar-favourite-icon-button.line-bar-favourite-icon-button:hover,\n      .sidebar-favourite-icon-button.line-bar-favourite-icon-button:focus-visible {\n        color: var(--accent);", css)
        self.assertIn('id="sidebarFavouriteAddBtn" class="line-bar-favourite-icon-button sidebar-favourite-icon-button" type="button" title="Add favourite" aria-label="Add favourite">+</button>', html)
        self.assertIn('id="sidebarFavouriteMenuBtn" class="line-bar-favourite-icon-button sidebar-favourite-icon-button" type="button" title="Edit favourites" aria-label="Edit favourites">...</button>', html)
        self.assertLess(html.index('id="sidebarFavouriteAddBtn"'), html.index('id="sidebarFavouriteMenuBtn"'))
        self.assertIn("position: fixed;", css)
        self.assertIn("line_bar_favourite", main_js)
        self.assertIn('const FAVOURITE_SCOPES = new Set(["metrics", "metrics_filter", "line_bar_view", "map_view"]);', main_js)
        self.assertIn('FAVOURITE_SCOPES = {"metrics", "metrics_filter", "line_bar_view", "map_view"}', favourites_py)
        self.assertIn('if scope in {"metrics_filter", "line_bar_view", "map_view"}:', favourites_py)
        self.assertIn('map_view: "Map view",', main_js)
        self.assertIn("captureLineBarFavouriteView", main_js)
        self.assertIn("applyLineBarFavouriteView", main_js)
        self.assertIn("applyMapFavouriteView", main_js)
        self.assertIn("applyMapFavouriteStateOnly", main_js)
        self.assertIn("startupFavouriteForRestore", main_js)
        self.assertIn("startupToolForFavourite", main_js)
        self.assertIn("applyStartupFavouriteState", main_js)
        self.assertIn("ukMapTool.applyFavouriteState(view.map || {})", main_js)
        self.assertIn("view.map = ukMapTool.captureFavouriteState();", main_js)
        self.assertIn("formatLineValueForFormat", main_js)
        self.assertIn('/api/line-bar/favourites', main_js)
        self.assertIn("placeFavouritePopover", main_js)
        self.assertIn("chartResponseFormatter", line_bar_js)
        self.assertIn("formatLineValueForFormat(value, renderKpiFormat)", line_bar_js)
        self.assertIn("formatter: (value) => formatChartResponseValue(value)", line_bar_js)
        self.assertIn("updateFavouriteRenameButton", main_js)
        self.assertIn("function clearActiveFavouriteSelectionForScope(change)", main_js)
        self.assertIn("line-bar-favourite-action-button", main_js)
        self.assertIn('data-favourite-action="move-up"', main_js)
        self.assertIn('data-favourite-action="move-down"', main_js)
        self.assertIn('class="sidebar-favourite-scope-row" role="radiogroup" aria-label="Favourite scope"', main_js)
        self.assertIn('class="sidebar-favourite-scope-title">Scope</div>', main_js)
        self.assertIn('class="sidebar-favourite-scope-options"', main_js)
        self.assertIn('class="sidebar-favourite-scope-radio" type="radio" name="sidebarFavouriteScope"', main_js)
        self.assertIn('data-favourite-scope-option="${escapeHtml(scope)}"', main_js)
        self.assertIn('const addButton = el("sidebarFavouriteAddBtn");', main_js)
        self.assertIn('const menuButton = el("sidebarFavouriteMenuBtn");', main_js)
        self.assertIn("function toolSupportsFavouriteAdd(tool = state.tool)", main_js)
        self.assertIn('return tool === "line_bar" || tool === "uk_map";', main_js)
        self.assertIn('addButton.classList.toggle("hidden", !canAdd);', main_js)
        self.assertIn('if (!toolSupportsFavouriteAdd()) return;', main_js)
        self.assertIn('popover.classList.toggle("line-bar-favourite-popover--manage", mode !== "add");', main_js)
        self.assertIn('if (popover.contains(event.target) || addButton?.contains(event.target) || menuButton?.contains(event.target)) return;', main_js)
        self.assertIn("function defaultFavouriteAddScope()", main_js)
        self.assertIn('return state.tool === "uk_map" ? "map_view" : DEFAULT_FAVOURITE_SCOPE;', main_js)
        self.assertIn("function lineBarCurrentViewScopeLabel()", main_js)
        self.assertIn('return state.tool === "line_bar" && state.view === "table" ? "Table view" : FAVOURITE_SCOPE_LABELS.line_bar_view;', main_js)
        self.assertIn("function favouriteTypeLabel(favourite)", main_js)
        self.assertIn('if (scope === "line_bar_view" && favourite?.view?.view === "table") return "Table view";', main_js)
        self.assertIn("LINE_BAR_FAVOURITE_SCOPE_OPTIONS", main_js)
        self.assertIn("MAP_FAVOURITE_SCOPE_OPTIONS", main_js)
        self.assertIn('["line_bar_view", FAVOURITE_SCOPE_LABELS.line_bar_view]', main_js)
        self.assertIn('["map_view", FAVOURITE_SCOPE_LABELS.map_view]', main_js)
        self.assertIn('["metrics_filter", FAVOURITE_SCOPE_LABELS.metrics_filter]', main_js)
        self.assertIn('["metrics", FAVOURITE_SCOPE_LABELS.metrics]', main_js)
        self.assertNotIn('id="sidebarFavouriteScopeSelect"', main_js)
        self.assertNotIn('id="sidebarFavouriteScopeBtn"', main_js)
        self.assertNotIn('id="sidebarFavouriteScopeMenu"', main_js)
        self.assertIn("&uarr;", main_js)
        self.assertIn("&darr;", main_js)
        self.assertIn("&#10003;", main_js)
        self.assertIn("&times;", main_js)
        self.assertIn("data-original-name", main_js)
        add_start = main_js.index('data-favourite-action="save-add"')
        add_block = main_js[add_start:main_js.index("function favouritePopoverRow", add_start)]
        add_save_index = main_js.index('data-favourite-action="save-add"', add_start)
        add_input_index = main_js.index('id="sidebarFavouriteNameInput"', add_start)
        add_scope_row_index = main_js.index('class="sidebar-favourite-scope-row"', add_start)
        self.assertLess(add_save_index, add_input_index)
        self.assertLess(add_input_index, add_scope_row_index)
        self.assertNotIn("sidebarFavouriteScopeSelect", add_block)
        self.assertNotIn("toggle-scope", add_block)
        row_start = main_js.index('<div class="line-bar-favourite-row')
        favourite_block = main_js[row_start:main_js.index("function favouritePopoverRow", row_start)]
        delete_index = main_js.index('data-favourite-action="delete"', row_start)
        rename_index = main_js.index('data-favourite-action="rename"', row_start)
        input_index = main_js.index('class="line-bar-favourite-name-input"', row_start)
        scope_index = main_js.index('class="line-bar-favourite-row-scope"', row_start)
        row_message_index = main_js.index('class="line-bar-favourite-row-message"', row_start)
        self.assertLess(rename_index, input_index)
        self.assertLess(input_index, delete_index)
        self.assertLess(delete_index, scope_index)
        self.assertLess(scope_index, row_message_index)
        manage_block_start = main_js.index('const rows = lineBarFavourites.map')
        manage_block = main_js[manage_block_start:main_js.index('popover.querySelectorAll(".line-bar-favourite-row")', manage_block_start)]
        self.assertLess(manage_block.index('class="line-bar-favourite-move-controls"'), manage_block.index('class="line-bar-favourite-popover-list"'))
        move_block_start = main_js.index('if (action === "move-up" || action === "move-down")')
        move_block = main_js[move_block_start:main_js.index("const row = favouritePopoverRow", move_block_start)]
        self.assertIn("lineBarFavourites = ordered;", move_block)
        self.assertIn('renderFavouritePopover("manage");', move_block)
        self.assertIn("queueFavouriteOrderSave", move_block)
        self.assertNotIn('await refreshFavourites({ renderPopover: true });', move_block)
        self.assertNotIn('data-favourite-action="up"', favourite_block)
        self.assertNotIn('data-favourite-action="down"', favourite_block)
        self.assertNotIn(">Save</button>", favourite_block)
        self.assertNotIn(">Up</button>", favourite_block)
        self.assertNotIn(">Down</button>", favourite_block)
        self.assertNotIn(">Delete</button>", favourite_block)
        self.assertNotIn(">Add</button>", favourite_block)
        self.assertNotIn(">Close</button>", favourite_block)
        self.assertIn(".line-bar-favourite-action-button", css)
        self.assertIn(".line-bar-favourite-move-controls", css)
        self.assertIn(".line-bar-favourite-popover--manage {\n        display: flex;\n        flex-direction: column;\n        overflow: hidden;", css)
        self.assertIn(".line-bar-favourite-popover--manage .line-bar-favourite-popover-list {\n        flex: 1 1 auto;\n        gap: 4px;\n        min-height: 0;\n        overflow-y: auto;", css)
        self.assertIn(".line-bar-favourite-popover--manage .line-bar-favourite-move-controls {\n        flex: 0 0 auto;", css)
        self.assertIn(".line-bar-favourite-popover-head.sidebar-favourite-popover-head", css)
        self.assertIn("grid-template-columns: 28px minmax(180px, 1fr);", css)
        self.assertIn("grid-template-columns: 28px 28px;", css)
        self.assertIn("grid-template-columns: 28px minmax(120px, 1fr) 28px 104px;", css)
        self.assertNotIn("grid-template-columns: 28px minmax(120px, 1fr) 28px max-content;", css)
        self.assertIn(".line-bar-favourite-row-scope {\n        align-self: center;\n        min-width: 0;\n        overflow: hidden;\n        text-overflow: ellipsis;\n        white-space: nowrap;", css)
        self.assertIn(".sidebar-favourite-scope-row", css)
        self.assertIn(".sidebar-favourite-scope-title", css)
        self.assertIn(".sidebar-favourite-scope-options", css)
        self.assertIn(".sidebar-favourite-scope-option", css)
        self.assertIn(".sidebar-favourite-scope-radio", css)
        self.assertNotIn(".sidebar-favourite-scope-select", css)
        self.assertNotIn(".sidebar-favourite-scope-menu", css)
        self.assertNotIn(".sidebar-favourite-scope-button", css)
        self.assertNotIn("grid-template-columns: minmax(130px, 1fr) minmax(118px, max-content) 28px;", css)
        self.assertNotIn("grid-template-columns: 28px minmax(140px, 1fr) max-content minmax(64px, max-content);", css)
        self.assertNotIn("async function applyStartupFavourite()", main_js)
        self.assertIn("const startupFavouriteResult = await applyStartupFavouriteState();", main_js)
        self.assertIn("if (startupFavouriteResult.filterApplied) await refreshFilterRowCountMeta();", main_js)
        self.assertIn("state.tool = startupFavouriteResult.applied", main_js)
        self.assertIn("lineBarTool.setView(state.view, { refresh });", main_js)
        boot_start = main_js.index("export async function boot()")
        boot_block = main_js[boot_start:main_js.index("} catch (error)", boot_start)]
        self.assertLess(boot_block.index("await refreshFavourites();"), boot_block.index("const startupFavouriteResult = await applyStartupFavouriteState();"))
        self.assertLess(boot_block.index("const startupFavouriteResult = await applyStartupFavouriteState();"), boot_block.index("setTool(state.tool, false);"))
        self.assertLess(boot_block.index("setTool(state.tool, false);"), boot_block.index("await refreshMetricSummary({ force: true });"))
        self.assertLess(boot_block.index("await refreshMetricSummary({ force: true });"), boot_block.index("await refreshActiveTool({ force: true });"))
        self.assertNotIn('/api/line-bar/favourites', line_bar_js)
        self.assertIn('button.className = `feature favourite-option saved-favourite-option${active ? " active" : ""}${invalid ? " favourite-option-invalid" : ""}`;', main_js)
        self.assertIn('button.innerHTML = `<span class="saved-filter-name">${escapeHtml(String(favourite.name || "") + suffix)}</span><span class="favourite-detail">${escapeHtml(favouriteTypeLabel(favourite))}</span>`;', main_js)
        self.assertIn('heading.className = "saved-filter-theme kpi-theme";', main_js)
        self.assertNotIn('heading.className = "saved-filter-theme favourite-theme";', main_js)

    def test_favourite_add_button_tool_allowlist(self) -> None:
        js = self.assert_no_store("/static/app/main.js")[1].decode("utf-8")
        helper = self.js_function_source(js, "toolSupportsFavouriteAdd")
        self.run_node_script(helper + """
if (!toolSupportsFavouriteAdd("line_bar")) throw new Error("line_bar should allow favourite adds");
if (!toolSupportsFavouriteAdd("uk_map")) throw new Error("uk_map should allow favourite adds");
for (const tool of ["column_profile", "dataset_viewer", "histogram", "glm", "gbm", "specs"]) {
  if (toolSupportsFavouriteAdd(tool)) throw new Error(`${tool} should not allow favourite adds`);
}
""")

    def test_gbm_shap_flame_option_uses_exact_domain_without_45_55(self) -> None:
        chart_path = Path(__file__).resolve().parents[1] / "src/py_lucidum/static/app/gbm-shap-chart.js"
        script = f"""
import fs from "node:fs";
const source = fs.readFileSync({str(chart_path)!r}, "utf8").replaceAll("export ", "");
eval(source + "\\nglobalThis.__shapChartOption = shapChartOption;\\nglobalThis.__formatUpliftPercent = formatUpliftPercent;");
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
}}, {{}});
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

    def test_line_bar_uplift_formatting_and_visible_axis_bounds(self) -> None:
        js = self.app_js_contract()
        script = f"""
const state = {{ transform: "none" }};
const css = {{
  "--bar": "#bar",
  "--base-bar": "#base",
  "--tail": "#tail",
  "--text": "#text",
}};
function getCss(name) {{ return css[name] || name; }}
const RESPONSE_AXIS_PADDING = 0.08;
const RESPONSE_AXIS_TARGET_INTERVALS = 15;
const SHAP_RIBBON_SERIES = [
  ["p0", "p100", "SHAP Min-Max"],
  ["p5", "p95", "SHAP 5-95"],
  ["p10", "p90", "SHAP 10-90"],
  ["p20", "p80", "SHAP 20-80"],
  ["p30", "p70", "SHAP 30-70"],
  ["p40", "p60", "SHAP 40-60"],
];
{self.js_function_source(js, "formatUpliftPercent")}
{self.js_function_source(js, "isUpliftTransform")}
{self.js_function_source(js, "isBaseReferenceTransform")}
{self.js_function_source(js, "isBaseWeightBar")}
{self.js_function_source(js, "weightBarColor")}
{self.js_function_source(js, "upliftBaselineSeries")}
{self.js_function_source(js, "responseAxisOptions")}
{self.js_function_source(js, "withUpliftBaselineExtent")}
{self.js_function_source(js, "partialDependenceOverlayEntries")}
{self.js_function_source(js, "responseAxisExtent")}
{self.js_function_source(js, "responseAxisSpan")}
{self.js_function_source(js, "niceAxisStep")}
{self.js_function_source(js, "roundAxisValue")}
{self.js_function_source(js, "responseAxisBounds")}
{self.js_function_source(js, "matchingLegendSelection")}
{self.js_function_source(js, "legendEntryName")}
if (formatUpliftPercent(1) !== "0%") throw new Error("base uplift should be 0%");
if (formatUpliftPercent(1.25) !== "+25%") throw new Error("positive uplift should include plus sign");
if (formatUpliftPercent(0.9) !== "-10%") throw new Error("negative uplift should be shown from base");
const data = {{
  rows: [
    {{ resp0: 100, resp1: 10 }},
    {{ resp0: 200, resp1: 20 }},
  ],
  responses: [{{ label: "Actual" }}, {{ label: "Expected" }}],
}};
const full = responseAxisOptions(data, {{ Actual: true, Expected: true }});
const expectedOnly = responseAxisOptions(data, {{ Actual: false, Expected: true }});
const fallback = responseAxisOptions(data, {{ Actual: false, Expected: false }});
if (!(expectedOnly.max < full.max)) throw new Error(`expected hidden response to shrink y-axis ${{expectedOnly.max}} vs ${{full.max}}`);
if (fallback.max !== full.max || fallback.min !== full.min) throw new Error("all-hidden axis should fall back to full extent");
state.transform = "one";
const upliftBounds = responseAxisOptions({{
  rows: [{{ resp0: 1.2 }}, {{ resp0: 1.3 }}],
  responses: [{{ label: "Actual" }}],
}}, {{ Actual: true }});
if (upliftBounds.min > 1 || upliftBounds.max <= 1) throw new Error(`uplift axis should include raw baseline 1: ${{upliftBounds.min}}..${{upliftBounds.max}}`);
const baseline = upliftBaselineSeries({{ rows: [{{}}, {{}}, {{}}] }});
if (baseline?.markLine?.data?.[0]?.yAxis !== 1) throw new Error("uplift baseline should be drawn at raw y=1");
if ((baseline?.markLine?.lineStyle?.width || 0) <= 1) throw new Error("uplift baseline should be thicker than grid lines");
const baseData = {{ transform: {{ reference: "base", base_x: "40" }} }};
if (!isBaseWeightBar(baseData, {{ x: 40 }})) throw new Error("numeric base x should match string metadata");
if (weightBarColor(baseData, {{ x: "40" }}) !== "#base") throw new Error("base row should use base bar color");
if (weightBarColor(baseData, {{ x: "40", is_tail: true }}) !== "#tail") throw new Error("tail color should still take precedence");
state.transform = "zero";
if (weightBarColor(baseData, {{ x: "40" }}) !== "#base") throw new Error("zero transform should also use base bar color");
state.transform = "none";
if (isBaseWeightBar(baseData, {{ x: "40" }})) throw new Error("base bar should only apply to base transforms");
const persisted = matchingLegendSelection({{ legend: [{{ selected: {{ Actual: false, Weight: false }} }}] }}, [
  {{ name: "Actual" }},
  {{ name: "Weight" }},
]);
if (persisted.Actual !== false || persisted.Weight !== false) throw new Error("legend selection should persist by matching name");
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
        self.assertIn(".dataset-meta-title {\n        color: var(--text);\n        font-weight: 700;\n      }", css)
        self.assertIn(".dataset-meta-details {\n        color: var(--muted);\n      }", css)
        self.assertIn(".dataset-meta-title-only .dataset-meta-details {\n        display: none;\n      }", css)
        self.assertIn(".dataset-meta-gbm-link,\n      .dataset-meta-glm-link {", css)
        self.assertIn(".dataset-meta-column-count {\n        color: var(--muted);\n      }", css)
        self.assertNotIn("dataset-meta-column-link", css)
        self.assertNotIn("dataset-meta-uk-map-link", css)
        self.assertIn(".tool-icon svg {\n        width: 28px;\n        height: 28px;", css)
        self.assertIn("stroke-width: 1.5;", css)
        self.assertIn(".tool-icon img {\n        width: 30px;\n        height: 30px;", css)
        self.assertIn("filter: grayscale(1) opacity(0.62);", css)
        self.assertIn(".tool-option.active .tool-icon img {\n        filter: brightness(0) saturate(100%) invert(39%) sepia(70%)", css)
        self.assertIn("body.dark .tool-icon img {\n        filter: brightness(0) saturate(100%) invert(76%) sepia(10%)", css)
        self.assertIn("body.dark .tool-option.active .tool-icon img {\n        filter: brightness(0) saturate(100%) invert(72%) sepia(62%)", css)
        self.assertNotIn("dataset-meta-uk-map-icon", css)
        self.assertIn("text-decoration-skip-ink: none;", css)
        self.assertIn("text-decoration-thickness: 1px;", css)
        self.assertIn("function renderDatasetMeta(", js)
        self.assertNotIn("dataset-meta-uk-map-icon", js)
        self.assertNotIn("renderDatasetPostcodeMeta", js)
        self.assertNotIn("dataset-meta-column-link", js)
        self.assertNotIn("dataset-meta-uk-map-link", js)
        self.assertNotIn("function openColumnProfile()", js)
        self.assertNotIn("ukMapTool.setMapLevel(level", js)
        self.assertIn("const titlePrefix = String(state.schema?.title_prefix || \"\").trim();", js)
        self.assertIn('title.className = "dataset-meta-title";', js)
        self.assertIn("title.textContent = titlePrefix;", js)
        self.assertNotIn("if (titlePrefix) parts.push(titlePrefix);", js)
        self.assertIn("function scheduleDatasetMetaCompactCheck()", js)
        self.assertIn("function updateDatasetMetaCompactMode()", js)
        self.assertIn('details.className = "dataset-meta-details";', js)
        self.assertIn("const overflows = target.scrollWidth > target.clientWidth + 1;", js)
        self.assertIn('target.classList.toggle("dataset-meta-title-only", overflows);', js)
        self.assertIn("scheduleDatasetMetaCompactCheck();", js)
        self.assertIn('columnCount.className = "dataset-meta-column-count";', js)
        self.assertIn('const payload = await api("/api/glm/models", { method: "GET" });', js)
        self.assertIn('const payload = await api("/api/gbm/models", { method: "GET" });', js)
        self.assertIn("button.textContent = `GLMs (${datasetGlmCount.toLocaleString()})`;", js)
        self.assertIn("button.textContent = `GBMs (${datasetGbmCount.toLocaleString()})`;", js)
        self.assertIn("glmTool.openModelNavigator();", js)
        self.assertIn("gbmTool.openModelNavigator();", js)
        self.assertIn("openModelNavigator,", js)
        self.assertIn('window.matchMedia("(prefers-color-scheme: dark)").matches', html)
        self.assertIn('document.body.classList.add("dark");', html)
        self.assertIn('id="sidebarToggleBtn"', html)
        self.assertIn('aria-controls="appSidebar"', html)
        self.assertIn('aria-label="Collapse sidebar"', html)
        self.assertIn('<aside id="appSidebar">', html)
        self.assertIn('id="toolSelectorSection" class="section tool-selector-section"', html)
        self.assertIn('<div class="tool-selector">', html)
        self.assertIn('id="sidebarControlPane" class="sidebar-control-pane"', html)
        self.assertNotIn("toolSelectorToggleBtn", html)
        self.assertNotIn("toolSelectorActiveLabel", html)
        self.assertNotIn("toolSelectorList", html)
        self.assertIn('id="sidebarVersion" class="sidebar-version" aria-label="Lucidum version" hidden', html)
        self.assertIn('id="collapsedSidebarVersion" class="collapsed-sidebar-version" aria-label="Lucidum version" hidden', html)
        self.assertIn(".sidebar-version {\n        margin-top: auto;\n        color: var(--muted);\n        font-size: 10px;", css)
        self.assertIn("body.sidebar-collapsed .sidebar-version {\n        display: none;", css)
        self.assertIn(".collapsed-sidebar-version {\n        display: none;\n        margin-top: auto;\n        color: var(--muted);", css)
        self.assertIn("body.sidebar-collapsed .collapsed-sidebar-version:not([hidden]) {\n        display: block;", css)
        self.assertIn("function renderSidebarVersion()", js)
        self.assertNotIn("toolSelectorExpanded", js)
        self.assertNotIn("syncToolSelectorDisclosure", js)
        self.assertNotIn("setToolSelectorExpanded", js)
        self.assertNotIn("toolSelectorToggleBtn", js)
        self.assertIn("function bindToolButtonTooltips()", js)
        self.assertIn("function positionToolButtonTooltip(target = toolButtonTooltipTarget)", js)
        self.assertIn("const TOOL_BUTTON_TOOLTIP_DELAY_MS = 500;", js)
        self.assertIn("let toolButtonTooltipPendingTarget = null;", js)
        self.assertIn("let toolButtonTooltipTimer = null;", js)
        self.assertIn("function scheduleToolButtonTooltip(button)", js)
        self.assertIn('button.removeAttribute("title");', js)
        self.assertIn('button.addEventListener("pointerenter", () => scheduleToolButtonTooltip(button));', js)
        self.assertIn('button.addEventListener("focus", () => scheduleToolButtonTooltip(button));', js)
        self.assertNotIn('button.addEventListener("pointerenter", () => showToolButtonTooltip(button));', js)
        self.assertIn("bindToolButtonTooltips();", js)
        self.assertIn('const version = String(state.schema?.app_version || "").trim();', js)
        self.assertIn("target.textContent = version ? `lucidum v${version}` : \"\";", js)
        self.assertIn("collapsedTarget.textContent = version ? `v${version}` : \"\";", js)
        self.assertIn('id="lineBarTool" class="tool-option" type="button" data-tool="line_bar" aria-label="Line and bar"', html)
        self.assertIn('class="tool-label">Line and bar</span>', html)
        self.assertIn('id="datasetViewerTool" class="tool-option" type="button" data-tool="dataset_viewer" aria-label="Dataset viewer"', html)
        self.assertNotIn('id="datasetViewerTool" class="tool-option active"', html)
        self.assertIn('class="tool-label">Dataset viewer</span>', html)
        self.assertIn('id="profileTool" class="tool-option" type="button" data-tool="column_profile" aria-label="Column profile"', html)
        self.assertIn('class="tool-label">Column profile</span>', html)
        self.assertLess(html.index('id="lineBarTool"'), html.index('id="datasetViewerTool"'))
        self.assertLess(html.index('id="lineBarTool"'), html.index('id="profileTool"'))
        self.assertIn('id="histogramTool" class="tool-option" type="button" data-tool="histogram" aria-label="Histogram"', html)
        self.assertIn('class="tool-label">Histogram</span>', html)
        self.assertIn('id="ukMapTool" class="tool-option" type="button" data-tool="uk_map" aria-label="UK mapping"', html)
        self.assertIn('<img src="/tools/uk-map/static/icons/UK.png" alt="">', html)
        self.assertNotIn("dataset-meta-uk-map-icon", html)
        self.assertIn('class="tool-label">UK mapping</span>', html)
        self.assertIn('class="section sidebar-favourites-section sidebar-accordion-section sidebar-section-closed"', html)
        self.assertIn('id="lineBarToolbar" class="toolbar hidden"', html)
        self.assertIn('id="histogramToolbar" class="toolbar hidden"', html)
        self.assertIn('id="status" class="status main-status hidden"', html)
        self.assertIn('id="visualArea" class="visual-area startup-mode"', html)
        self.assertNotIn('id="visualArea" class="visual-area dataset-viewer-mode"', html)
        self.assertIn('id="chartSideControls" class="chart-side-controls hidden"', html)
        self.assertIn('<button data-value="original">Original</button>\n                    <button data-value="alpha" class="active">A-Z</button>', html)
        self.assertIn('id="chartControlHeightRow" class="chart-control-height-row"', html)
        self.assertIn('id="chartControlHeightResizer" class="chart-control-height-resizer app-resizer app-resizer--horizontal"', html)
        self.assertIn('id="chartExpectedToggle" class="chart-expected-toggle" type="button" aria-controls="expectedSideSection" aria-expanded="true" aria-label="Hide Expected controls"', html)
        self.assertIn('id="expectedSideSection" class="chart-side-section"', html)
        self.assertIn('id="chartControlsResizer" class="chart-controls-resizer app-resizer app-resizer--vertical hidden"', html)
        self.assertIn(".chart-side-controls.chart-expected-collapsed {\n        grid-template-rows: var(--chart-feature-controls-height, minmax(0, 1fr)) 18px;", css)
        self.assertIn(".chart-side-controls.chart-expected-collapsed #expectedSideSection {\n        display: none;", css)
        self.assertIn(".chart-control-height-row {\n        align-items: stretch;", css)
        self.assertIn(".chart-side-controls.chart-expected-collapsed .chart-expected-toggle-icon {\n        transform: translateY(2px) rotate(225deg);", css)
        self.assertIn('const CHART_FEATURE_CONTROLS_HEIGHT_COLLAPSED = "collapsed";', js)
        self.assertIn("featureSort: \"alpha\",", js)
        self.assertIn("expectedSort: \"alpha\",", js)
        self.assertIn("function applyStartupChartExpectedCollapse()", js)
        self.assertIn("function syncChartControlHeightToAvailableSpace()", js)
        self.assertIn("syncChartControlHeightToAvailableSpace();\n          lineBarTool.resize();", js)
        self.assertIn("setChartFeatureControlsHeight(CHART_FEATURE_CONTROLS_HEIGHT_COLLAPSED);", js)
        self.assertIn('toggle?.addEventListener("click", () => toggleExpectedSideSection());', js)
        self.assertIn("function defaultChartFeatureControlsHeight()", js)
        self.assertNotIn("py_lucidum_chart_feature_controls_height", js)
        self.assertIn('expectedSection.toggleAttribute("inert", collapsed);', js)
        self.assertIn("setChartFeatureControlsHeight(height, { allowCollapse: false });", js)
        self.assertIn('id="lineBarTabs" class="tabs workspace-tabs hidden"', html)
        self.assertIn('id="lineBarSideControlsToggleBtn" class="tab line-bar-icon-tab line-bar-side-controls-toggle"', html)
        self.assertIn('id="lineBarToolbarToggleBtn" class="tab line-bar-icon-tab line-bar-toolbar-toggle"', html)
        self.assertIn('id="lineBarCopyBtn" class="tab line-bar-icon-tab"', html)
        self.assertLess(html.index('id="lineBarSideControlsToggleBtn"'), html.index('id="lineBarToolbarToggleBtn"'))
        self.assertLess(html.index('id="lineBarToolbarToggleBtn"'), html.index('id="lineBarCopyBtn"'))
        self.assertLess(html.index('id="lineBarCopyBtn"'), html.index('id="chartTab"'))
        self.assertLess(html.index('id="chartTab"'), html.index('id="tableTab"'))
        self.assertNotIn("lineBarExpandBtn", html)
        self.assertIn("lineBarSideControlsCollapsed: true", js)
        self.assertIn("lineBarToolbarCollapsed: true", js)
        self.assertNotIn("lineBarFocusMode", js)
        self.assertNotIn("toggleLineBarFocusMode", js)
        self.assertIn('el("lineBarSideControlsToggleBtn")?.addEventListener("click", toggleLineBarSideControls);', js)
        self.assertIn('el("lineBarToolbarToggleBtn")?.addEventListener("click", toggleLineBarToolbar);', js)
        self.assertIn('visualArea.classList.toggle("line-bar-side-controls-collapsed", sideCollapsed);', js)
        self.assertIn('toolbar.classList.toggle("hidden", !lineBarActive || toolbarCollapsed);', js)
        self.assertIn(".visual-area.line-bar-side-controls-collapsed {\n        grid-template-columns: minmax(0, 1fr);", css)
        self.assertIn(".visual-area.line-bar-side-controls-collapsed #chartSideControls,\n      .visual-area.line-bar-side-controls-collapsed #chartControlsResizer {\n        display: none !important;", css)
        self.assertIn("#lineBarSideControlsToggleBtn[aria-expanded=\"false\"] .line-bar-chevron-horizontal,\n      #lineBarToolbarToggleBtn[aria-expanded=\"false\"] .line-bar-chevron-vertical {\n        transform: rotate(180deg);", css)
        self.assertIn('id="datasetViewerGroupMeta" class="workspace-meta hidden"', html)
        self.assertIn('id="datasetViewerFilter" class="hidden"><button id="datasetViewerFilterClearBtn" class="filter-row-clear" type="button" title="Clear filter" aria-label="Clear filter" hidden></button><span id="datasetViewerFilterText" class="filter-badge-text">no filter</span></div>', html)
        self.assertIn('id="profileGroupMeta" class="workspace-meta"', html)
        self.assertIn('id="profileFilter" class="hidden"><button id="profileFilterClearBtn" class="filter-row-clear" type="button" title="Clear filter" aria-label="Clear filter" hidden></button><span id="profileFilterText" class="filter-badge-text">no filter</span></div>', html)
        self.assertIn('id="lineBarGroupMeta" class="workspace-meta hidden"', html)
        self.assertIn('id="lineBarFilter" class="hidden"><button id="lineBarFilterClearBtn" class="filter-row-clear" type="button" title="Clear filter" aria-label="Clear filter" hidden></button><span id="lineBarFilterText" class="filter-badge-text">no filter</span></div>', html)
        self.assertIn('id="histogramGroupMeta" class="workspace-meta hidden"', html)
        self.assertIn('id="histogramFilter" class="hidden"><button id="histogramFilterClearBtn" class="filter-row-clear" type="button" title="Clear filter" aria-label="Clear filter" hidden></button><span id="histogramFilterText" class="filter-badge-text">no filter</span></div>', html)
        self.assertIn('id="chart" class="hidden"', html)
        self.assertIn('id="histogramWrap" class="histogram-wrap hidden"', html)
        self.assertIn('id="datasetViewerWrap" class="dataset-viewer-wrap hidden"></div>', html)
        self.assertIn('id="profileWrap" class="profile-wrap hidden"></div>', html)
        self.assertIn('id="themeBtn"', html)
        self.assertIn('aria-label="Switch to dark mode"', html)
        self.assertIn("theme-icon-moon", html)
        self.assertIn("theme-icon-sun", html)
        self.assertIn('id="monitorLink" class="ghost monitor-link header-icon-button hidden" href="/monitor" target="_blank"', html)
        self.assertIn('aria-label="Open monitor" title="Open monitor"', html)
        self.assertIn('id="stopAppBtn" class="danger-action hidden" aria-hidden="true" inert>Stop app</button>', html)
        self.assertIn('id="reloadBtn" class="ghost header-icon-button" type="button" aria-label="Reload dataset" title="Reload dataset"', html)
        self.assertIn('class="header-action-icon"', html)
        self.assertNotIn(">Monitor</a>", html)
        self.assertNotIn(">Reload</button>", html)
        self.assertNotIn('id="themeBtn" class="ghost">Dark</button>', html)
        self.assertNotIn("<h2>Tool</h2>", html)
        self.assertIn('href="/static/app.css"', html)
        self.assertIn('href="/static/vendor/leaflet/leaflet.css"', html)
        self.assertIn('src="/static/vendor/echarts/echarts.min.js"', html)
        self.assertIn('src="/static/vendor/leaflet/leaflet.js"', html)
        self.assertIn('type="module" src="/static/app.js"', html)
        self.assertNotIn("cdn.jsdelivr.net", html)
        self.assertNotIn("unpkg.com", html)
        self.assertIn('id="modelToolWrap" class="model-tool-wrap hidden"', html)
        self.assert_model_sidebar_panel_visibility(html, set())
        self.assertNotIn('id="sidebarGlmResizer"', html)
        self.assertNotIn('aria-label="Resize KPI and GLM model controls"', html)
        self.assertIn('id="glmModelCollapseBtn"', html)
        self.assertIn('<span class="sidebar-section-title">GLMs</span>', html)
        self.assertIn('id="glmModelSelectedMeta"', html)
        self.assertIn('id="glmModelSelect" class="feature-list glm-model-list" role="listbox"', html)
        self.assertNotIn('id="sidebarGbmResizer"', html)
        self.assertNotIn('aria-label="Resize KPI and GBM model controls"', html)
        self.assertIn('id="gbmModelCollapseBtn"', html)
        self.assertIn('<span class="sidebar-section-title">GBMs</span>', html)
        self.assertIn('id="gbmModelSelectedMeta"', html)
        self.assertIn('id="gbmModelSelect" class="feature-list gbm-model-list" role="listbox"', html)
        self.assertIn("No GLMs built yet", js)
        self.assertIn("No GBMs trained yet", js)
        self.assertIn(".glm-empty-state", css)
        self.assertIn(".gbm-model-list .gbm-empty-state", css)
        self.assertNotIn('id="gbmActiveModelSelect"', html)
        self.assertNotIn("?v=", html)

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

    def test_static_app_assets_disable_cache(self) -> None:
        self.assert_no_store("/static/app.js")
        self.assert_no_store("/static/app/main.js")
        self.assert_no_store("/static/app/dataset-viewer-tool.js")
        self.assert_no_store("/static/app/column-profile-tool.js")
        self.assert_no_store("/static/app/line-bar-tool.js")
        self.assert_no_store("/static/app/histogram-tool.js")
        self.assert_no_store("/static/app/uk-map-tool.js")
        self.assert_no_store("/static/app/glm-tool.js")
        self.assert_no_store("/static/app/glm-formula-assist.js")
        self.assert_no_store("/static/app/glm-formula-builder.js")
        self.assert_no_store("/static/app/glm-model-navigator.js")
        self.assert_no_store("/static/app/glm-tabulations.js")
        self.assert_no_store("/static/app/specifications-tool.js")
        self.assert_no_store("/static/app/shared/api.js")
        self.assert_no_store("/static/app/shared/format.js")
        self.assert_no_store("/static/app/shared/model-ui.js")
        self.assert_no_store("/static/app/shared/schema.js")
        self.assert_no_store("/static/app/shared/tabulator.js")
        self.assert_no_store("/static/app/shared/timing.js")
        self.assert_no_store("/static/app/gbm-tool.js")
        self.assert_no_store("/static/app/gbm-evaluation-chart.js")
        self.assert_no_store("/static/app/gbm-feature-parameter-controls.js")
        self.assert_no_store("/static/app/gbm-model-navigator.js")
        self.assert_no_store("/static/app/gbm-tab-orchestration.js")
        self.assert_no_store("/static/app/gbm-shap-tool.js")
        self.assert_no_store("/static/app/gbm-shap-chart.js")
        self.assert_no_store("/static/app/gbm-stacked-shap-tool.js")
        self.assert_no_store("/static/app/gbm-stacked-shap-chart.js")
        self.assert_no_store("/static/app/gbm-tree-viewer.js")
        self.assert_no_store("/static/app/model-tool-shell.js")
        self.assert_no_store("/static/vendor/tabulator/tabulator.min.js")
        self.assert_no_store("/static/vendor/tabulator/tabulator.min.css")
        self.assert_no_store("/static/vendor/d3/d3.min.js")
        self.assert_no_store("/static/vendor/echarts/echarts.min.js")
        self.assert_no_store("/static/vendor/echarts-gl/echarts-gl.min.js")
        self.assert_no_store("/static/vendor/leaflet/leaflet.css")
        self.assert_no_store("/static/vendor/leaflet/leaflet.js")
        self.assert_no_store("/static/vendor/leaflet/images/layers.png")
        self.assert_no_store("/static/vendor/leaflet/images/layers-2x.png")
        self.assert_no_store("/static/vendor/leaflet/images/marker-icon.png")
        self.assert_no_store("/static/vendor/leaflet/images/marker-icon-2x.png")
        self.assert_no_store("/static/vendor/leaflet/images/marker-shadow.png")
        self.assert_no_store("/static/vendor/ace/ace.js")
        self.assert_no_store("/static/vendor/ace/mode-r.js")
        self.assert_no_store("/static/vendor/ace/theme-textmate.js")
        self.assert_no_store("/static/vendor/ace/theme-monokai.js")
        app_css = self.assert_no_store("/static/app.css")[1].decode("utf-8")
        for path in self.CSS_MODULE_PATHS:
            if path == "/static/styles/dataset-viewer.css":
                continue
            import_path = f'.{path.removeprefix("/static")}'
            self.assertIn(f'@import url("{import_path}");', app_css)
        for path in self.CSS_MODULE_PATHS:
            self.assert_no_store(path)
        foundations_css = self.assert_no_store("/static/styles/foundations.css")[1].decode("utf-8")
        shell_css = self.assert_no_store("/static/styles/shell.css")[1].decode("utf-8")
        line_bar_css = self.assert_no_store("/static/styles/line-bar.css")[1].decode("utf-8")
        gbm_css = self.assert_no_store("/static/styles/gbm.css")[1].decode("utf-8")
        dataset_css = self.assert_no_store("/static/styles/dataset-viewer.css")[1].decode("utf-8")
        self.assertIn(".app {\n        user-select: none;\n        -webkit-user-select: none;\n      }", foundations_css)
        self.assertIn(
            ".app input,\n      .app textarea,\n      .app [contenteditable],\n      .app .tabulator-editing,",
            foundations_css,
        )
        self.assertIn(".app .ace_editor,\n      .app .ace_editor *,", foundations_css)
        self.assertIn(".hidden {\n        display: none !important;\n      }", foundations_css)
        self.assertNotRegex(gbm_css, r"(?m)^\s*\.hidden\s*\{\s*display:\s*none")
        self.assertIn(".visual-area.startup-mode {\n        grid-template-columns: minmax(0, 1fr);", line_bar_css)
        self.assertIn(".visual-area.dataset-viewer-mode,\n      .visual-area.profile-mode {\n        grid-template-columns: minmax(0, 1fr);", shell_css)
        self.assertIn(".visual-area.dataset-viewer-mode .workspace,\n      .visual-area.profile-mode .workspace {\n        background: transparent;\n        border: 0;", shell_css)
        self.assertIn(".dataset-viewer-wrap {\n        display: flex;", dataset_css)
        self.assertNotIn(".visual-area.dataset-viewer-mode", dataset_css)
        self.assertNotIn(".dataset-viewer-wrap.hidden", dataset_css)
        self.assert_no_store("/static/monitor.js")
        self.assert_no_store("/static/monitor.css")

    def test_glm_frontend_contains_real_tool_contract(self) -> None:
        js = self.app_js_contract()
        glm_tool_js = self.assert_no_store("/static/app/glm-tool.js")[1].decode("utf-8")
        glm_formula_js = self.assert_no_store("/static/app/glm-formula-builder.js")[1].decode("utf-8")
        glm_model_js = self.assert_no_store("/static/app/glm-model-navigator.js")[1].decode("utf-8")
        glm_tabulation_js = self.assert_no_store("/static/app/glm-tabulations.js")[1].decode("utf-8")
        glm_js = "\n".join([glm_tool_js, glm_formula_js, glm_model_js, glm_tabulation_js])
        css = self.app_css_contract()

        self.assertIn('import { createGlmTool } from "./glm-tool.js";', js)
        self.assertIn("export function createGlmTool", js)
        self.assertIn('api("/api/glm/config"', js)
        self.assertIn('api("/api/glm/validate"', js)
        self.assertIn('api("/api/glm/build"', js)
        self.assertIn('api("/api/glm/tabulations/config"', js)
        self.assertIn('api("/api/glm/tabulations/build"', js)
        self.assertIn('api("/api/glm/tabulations/table"', js)
        self.assertIn('api("/api/glm/tabulations/plot"', js)
        self.assertIn('api("/api/glm/tabulations/export"', js)
        self.assertIn('api("/api/glm/tabulations/rebase"', js)
        self.assertIn('api("/api/glm/tabulations/rebase/reset"', js)
        self.assertIn('api("/api/glm/formula/levels"', js)
        self.assertIn('`/api/glm/tabulations/jobs/${encodeURIComponent(jobId)}`', js)
        self.assertIn('`/api/glm/jobs/${encodeURIComponent(jobId)}`', js)
        self.assertIn('`/api/glm/models/${encodeURIComponent(modelId)}/activate`', js)
        self.assertIn('`/api/glm/models/${encodeURIComponent(modelId)}/rename`', js)
        self.assertIn('`/api/glm/models/${encodeURIComponent(modelId)}`', js)
        self.assertIn('from "./glm-formula-assist.js";', glm_formula_js)
        self.assertIn('import { loadTabulator } from "./shared/tabulator.js";', glm_js)
        self.assertIn('const ACE_BASE_PATH = "/static/vendor/ace";', js)
        self.assertIn('aceEditor.session.setMode("ace/mode/r");', js)
        self.assertIn('data-glm-tab="builder">Formula builder', js)
        self.assertIn('data-glm-tab="models">Model navigator', js)
        self.assertIn('data-glm-tab="tabulations">Tabulations', js)
        self.assertIn('<h3 class="glm-panel-title">GLM formula</h3>', js)
        self.assertNotIn("Formula and family", js)
        self.assertIn('id="glmFormulaEditor"', js)
        self.assertIn('id="glmFormulaAssistBtn"', js)
        self.assertIn('id="glmFormulaAssistDrawer"', js)
        self.assertIn('["snippets", "Numeric"]', glm_formula_js)
        self.assertIn('["piecewise", "Piecewise linear"]', glm_formula_js)
        self.assertIn('["levels", "Categorical"]', glm_formula_js)
        self.assertIn(">Insert at cursor</button>", glm_formula_js)
        self.assertIn("formatDrawerInsertion(output, editorTextBeforeInsertion(), { replaceSelection: editorHasSelection() })", glm_formula_js)
        self.assertIn('id="glmFormulaAssistIncludeHeader"', glm_formula_js)
        self.assertIn("include header", glm_formula_js)
        self.assertIn("data-glm-level-mode=\"group\"", glm_formula_js)
        self.assertIn("data-glm-level-mode=\"ind\"", glm_formula_js)
        self.assertIn("buildIndividualLevelsFormula", glm_formula_js)
        self.assertIn('id="glmFormulaAssistSnippetList"', glm_formula_js)
        self.assertIn("data-glm-snippet-id", glm_formula_js)
        self.assertIn("function refreshFormulaAssistSnippetSelection()", glm_formula_js)
        self.assertNotIn('id="glmFormulaAssistSnippetSelect"', glm_formula_js)
        self.assertNotIn("positionFormulaAssistDrawer", glm_formula_js)
        self.assertNotIn("glmFormulaAssistAppendPlus", glm_formula_js)
        self.assertNotIn("glmFormulaAssistSecondaryFeature", glm_formula_js)
        self.assertNotIn("append +", glm_formula_js)
        self.assertNotIn("Other</label>", glm_formula_js)
        self.assertNotIn("· ${escapeHtml(column.kind", glm_formula_js)
        self.assertIn('id="glmFamilySelect"', js)
        self.assertIn('id="glmFamilyParameter"', js)
        self.assertIn('for="glmFamilySelect">Family</label>', js)
        self.assertIn('placeholder="family.parameter"', js)
        self.assertIn('input.placeholder = "family.parameter";', glm_js)
        self.assertIn('id="glmRegularizationMode"', js)
        self.assertIn('id="glmRegularizationMix"', js)
        self.assertIn('id="glmRegularizationAlpha"', js)
        self.assertIn('class="glm-penalty-manual ${formulaBuilder.selectedRegularizationMode === "manual" ? "" : "disabled"}"', glm_js)
        self.assertIn('manual.classList.toggle("disabled", !isManual);', glm_js)
        self.assertNotIn('manual.classList.toggle("hidden", !isManual);', glm_js)
        self.assertIn('type="text" inputmode="decimal"', glm_js)
        self.assertIn("function syncAceGutterWidth()", glm_js)
        self.assertIn("function currentAceTheme()", glm_js)
        self.assertIn('return document.body.classList.contains("dark") ? "ace/theme/monokai" : "ace/theme/textmate";', glm_js)
        self.assertIn("aceEditor.setTheme(currentAceTheme());", glm_js)
        self.assertNotIn('aceEditor.setTheme("ace/theme/textmate");', glm_js)
        self.assertIn('container.style.setProperty("--glm-ace-gutter-width", value);', glm_js)
        self.assertIn('class="glm-builder-control-row glm-builder-control-stack"', glm_js)
        self.assertIn('class="glm-control-line"', glm_js)
        self.assertIn("glm-header-scope-control", glm_js)
        self.assertIn('id="glmBuilderResizer" class="glm-builder-resizer app-resizer app-resizer--vertical"', glm_js)
        self.assertIn("function bindBuilderResizer()", glm_js)
        self.assertNotIn("py_lucidum_glm_tabulation_sidebar_width_v2", glm_js)
        self.assertIn('const GLM_TABULATION_MODEL_CROSSTAB = "__model__";', glm_js)
        self.assertIn('id="glmTabulationResizer" class="glm-builder-resizer glm-tabulation-resizer app-resizer app-resizer--vertical"', glm_js)
        self.assertIn("function bindTabulationResizer()", glm_js)
        self.assertIn('layout.style.setProperty("--glm-tabulation-sidebar-width"', glm_js)
        self.assertIn("let tabulationResizeObserver = null;", glm_js)
        self.assertIn("function scheduleTabulationResize()", glm_js)
        self.assertIn("function observeTabulationLayoutResize()", glm_js)
        self.assertIn("tabulationResizeObserver = observeResize([main], scheduleTabulationResize);", glm_js)
        self.assertIn("function resize()", glm_js)
        self.assertIn('data-glm-scope="training"', js)
        self.assertIn('id="glmBuildBtn"', js)
        self.assertIn('id="glmCoefficientTable"', js)
        self.assertIn("selectExpectedPredictionForModelKind: (modelKind) => setExpectedPredictionSelectionForModelKind(modelKind)", js)
        self.assertIn("canNavigateToLineBarFeature = () => false", glm_js)
        self.assertIn("navigateToLineBarFeature = () => false", glm_js)
        self.assertIn("selectExpectedPredictionForModelKind = () => false", glm_js)
        self.assertIn("function openGlmCoefficientContextMenuForRow(event)", glm_js)
        self.assertIn("function coefficientContextFeatureNames(row = {})", glm_js)
        self.assertIn(".glm-formula-assist-drawer", css)
        drawer_block = css.split(".glm-formula-assist-drawer {", 1)[1].split("}", 1)[0]
        self.assertIn("border-bottom: 1px solid var(--line);", drawer_block)
        self.assertIn("border-top: 1px solid var(--line);", drawer_block)
        self.assertIn("flex: 0 0 429px;", drawer_block)
        self.assertNotIn("border: 1px solid", drawer_block)
        self.assertNotIn("box-shadow", drawer_block)
        self.assertNotIn("position: absolute", drawer_block)
        self.assertNotIn("height: 429px;", drawer_block)
        self.assertIn(".glm-formula-assist-drawer:not(.hidden) + .glm-builder-control-row", css)
        self.assertIn(".glm-formula-assist-snippet-list", css)
        self.assertIn(".glm-formula-assist-snippet-row.active", css)
        self.assertIn(".glm-formula-assist-header-toggle", css)
        self.assertIn(".glm-formula-assist-level-search-row", css)
        self.assertIn(".glm-formula-assist-level-mode", css)
        self.assertIn("min-height: 82px;", css)
        self.assertIn(".glm-formula-assist-feature-select:focus", css)
        self.assertIn("function modelProblemMessages(diagnostics = {}, model = {}, coefficients = [])", glm_js)
        self.assertIn("Selected model issue:", glm_js)
        self.assertIn("glm-coefficient-meta-warning", glm_js)
        self.assertIn(".glm-coefficient-meta-warning", css)
        self.assertIn(".glm-formula-autocomplete", css)
        self.assertIn("function goToLineBarCoefficientFeature(featureName)", glm_js)
        self.assertIn('button.textContent = `Go to Line and Bar (${feature})`;', glm_js)
        self.assertIn('row.addEventListener("contextmenu", openGlmCoefficientContextMenuForRow);', glm_js)
        self.assertIn('selectExpectedPredictionForModelKind("glm");', glm_js)
        self.assertIn('id="glmModelGrid" class="glm-grid glm-model-grid"', glm_js)
        self.assertIn('id="glmModelFallback" class="glm-model-fallback"', glm_js)
        self.assertNotIn('id="glmModelTable"', glm_js)
        self.assertIn('label: `GLM ${glmAutoModelTimeLabel()}`', glm_js)
        self.assertNotIn('label: `GLM ${actual} ${glmAutoModelTimeLabel()}`', glm_js)
        self.assertIn("function validateFamilyParameter(family, rawValue)", glm_js)
        self.assertIn("function validateRegularizationParameter(regularization = {})", glm_js)
        self.assertIn("function regularizationLabel(regularization = {})", glm_js)
        self.assertIn("regularization: buildRegularizationPayload()", glm_js)
        self.assertIn("function setBuildFailure(message)", glm_js)
        self.assertIn('setBuildFailure(job.error || progress.message || "GLM training did not save a model");', glm_js)
        self.assertIn('setAppReadyStatus("Ready");', glm_js)
        self.assertNotIn('setAppReadyStatus("GLM built")', glm_js)
        self.assertNotIn('setGlmNotice(job.error || "GLM build failed");', glm_js)
        self.assertIn("function syncBuilderFromModelDetail(detail = {}, options = {})", glm_js)
        self.assertIn("syncBuilderFromModelDetail(activeDetail, { syncBuilderDraft });", glm_js)
        self.assertIn('class="glm-model-active-dot"', glm_js)
        self.assertIn('class="glm-model-name-cell"><span class="glm-model-name-main"', glm_js)
        self.assertIn("<th>created</th>", glm_js)
        self.assertIn('selectableRows: true,', glm_js)
        self.assertIn('selectableRowsRangeMode: "click",', glm_js)
        self.assertIn('modelTable.on("rowSelectionChanged", syncSelectedModelsFromTable);', glm_js)
        self.assertIn("function renderModelFallback(models = modelRows, activeModelId = config?.active_model_id)", glm_js)
        self.assertIn("function restoreModelSelection(ids)", glm_js)
        self.assertIn("bindFallbackModelSelection(fallbackRows, onFallbackSelectionChange);", glm_js)
        self.assertIn('id="glmTabulationColor" type="checkbox"', glm_js)
        self.assertIn('id="glmTabulationModelGrid"', glm_js)
        self.assertIn('id="glmTabulationModelFallback"', glm_js)
        self.assertIn('id="glmTabulationSelectorResizer" class="glm-tabulation-selector-resizer app-resizer app-resizer--horizontal"', glm_js)
        self.assertNotIn("py_lucidum_glm_tabulation_model_list_height", glm_js)
        self.assertIn('id="glmTabulationTableSections"', glm_js)
        self.assertIn('id="glmTabulationTableGrid"', glm_js)
        self.assertIn('id="glmTabulationCommonTableGrid"', glm_js)
        self.assertIn('id="glmTabulationOtherTableGrid"', glm_js)
        self.assertIn('id="glmTabulationCrosstab"', glm_js)
        self.assertIn('id="glmBuildTabulationsBtn"', glm_js)
        self.assertIn('id="glmExportTabulationsBtn"', glm_js)
        self.assertNotIn('id="glmTabulationRebaseBtn"', glm_js)
        self.assertNotIn('id="glmTabulationResetRebaseBtn"', glm_js)
        self.assertNotIn('id="glmTabulationRebaseSelection"', glm_js)
        self.assertIn("glm-tabulation-controls-row glm-tabulation-controls-primary", glm_js)
        self.assertNotIn("glm-tabulation-controls-row glm-tabulation-controls-rebase-row", glm_js)
        self.assertNotIn('let selectedTabulationRebaseCell = null;', glm_js)
        self.assertIn("function activeTabulationRebaseTransferFeature()", glm_js)
        self.assertIn('return features.length >= 2 && tabulationCrosstab', glm_js)
        self.assertIn(': "";', glm_js)
        self.assertIn("function tabulationRebaseContextForCell(row = {}, column = {})", glm_js)
        self.assertIn("function tabulationRebaseActionLabel(rebaseContext = {})", glm_js)
        self.assertIn("Rebase ${slice} slice to this cell; offset ${transferFeature} table", glm_js)
        self.assertIn("Rebase whole table to this cell; offset base", glm_js)
        self.assertIn("function openGlmTabulationContextMenu(event, rebaseContext = null)", glm_js)
        self.assertIn("function closeGlmTabulationContextMenu()", glm_js)
        self.assertIn("function applyTabulationRebaseContext(rebaseContext = {})", glm_js)
        self.assertIn("function resetSelectedTabulationRebase()", glm_js)
        self.assertIn('tabulationTable.on("cellContext", openGlmTabulationContextMenuForTabulatorCell);', glm_js)
        self.assertIn("function openGlmTabulationContextMenuForTabulatorCell(event, cell)", glm_js)
        self.assertIn('data-glm-tabulation-fallback-cell="true"', glm_js)
        self.assertIn('transfer_feature: rebaseContext.transfer_feature || ""', glm_js)
        self.assertNotIn("function applySelectedTabulationRebase()", glm_js)
        self.assertNotIn("function selectTabulationRebaseCell(cell, column)", glm_js)
        self.assertIn("clearCachesAfterGlmModelSourceChange();", glm_js)
        self.assertIn('${isTabulating ? "Tabulating..." : "Tabulate"}</button>', glm_js)
        self.assertIn('tabulationButton.textContent = isTabulating ? "Tabulating..." : "Tabulate";', glm_js)
        self.assertIn('${isExportingTabulations ? "Exporting..." : "Export xlsx"}</button>', glm_js)
        self.assertIn('exportButton.textContent = isExportingTabulations ? "Exporting..." : "Export xlsx";', glm_js)
        self.assertIn('function canExportSelectedTabulations()', glm_js)
        self.assertIn('model?.tabulated && !isTabulating && !isExportingTabulations', glm_js)
        self.assertIn('body: JSON.stringify({ model_refs: modelRefs, scale: tabulationScale })', glm_js)
        self.assertIn('setInlineTabulationNotice(["Saving XLSX..."]);', glm_js)
        self.assertIn('`Saved XLSX: ${result.path || result.filename || "XLSX saved"}`', glm_js)
        self.assertNotIn('${isTabulating ? "Building..." : "Build"}</button>', glm_js)
        self.assertNotIn('tabulationButton.textContent = isTabulating ? "Building..." : "Build";', glm_js)
        self.assertIn('let tabulationCrosstab = "";', glm_js)
        self.assertIn('let tabulationCrosstabManualKey = "";', glm_js)
        self.assertIn('const tabulationCrosstabDefaultCache = new Map();', glm_js)
        self.assertNotIn('localStorage.getItem("py_lucidum_glm_tabulation_crosstab")', glm_js)
        self.assertIn('const options = [{ value: "", label: "No crosstab" }];', glm_js)
        self.assertIn('options.push({ value: GLM_TABULATION_MODEL_CROSSTAB, label: "Model" });', glm_js)
        self.assertIn("features.forEach((feature) => options.push({ value: feature, label: feature }));", glm_js)
        self.assertIn("function ensureDefaultTabulationCrosstab(modelIds = tabulationSelectedModelIds(), tableId = selectedTabulationTableId)", glm_js)
        self.assertIn("return counts[0] < counts[1] ? features[0] : features[1];", glm_js)
        self.assertIn("tabulationCrosstabManualKey = tabulationSelectionKey();", glm_js)
        self.assertIn('if (!values.has(tabulationCrosstab)) tabulationCrosstab = "";', glm_js)
        self.assertIn("await ensureDefaultTabulationCrosstab(model_ids, table_id);", glm_js)
        self.assertIn("const payload = { model_refs: model_ids, table_id, scale: tabulationScale, crosstab: tabulationCrosstab };", glm_js)
        self.assertIn("function tabulationModelRows(models = tabulationAvailableModels())", glm_js)
        self.assertNotIn("function tabulationBlockedModelMessage", glm_js)
        self.assertNotIn("function tabulationModelIsBlocked", glm_js)
        self.assertNotIn("n/a: >3 leaves", glm_js)
        self.assertIn('if (!row.tabulated) return "not tabulated";', glm_js)
        self.assertIn('if (!cell.getRow().getData()?.tabulated) return "";', glm_js)
        self.assertNotIn("Tabulations are limited to GBMs with <=3 leaves.", glm_js)
        self.assertIn("function tabulationTableCountFormatter(cell)", glm_js)
        self.assertIn("function tabulationModelMetricFormatter(cell)", glm_js)
        self.assertIn("function tabulationTableGroups()", glm_js)
        self.assertIn("Common tables", glm_js)
        self.assertIn("Other tables", glm_js)
        self.assertIn('data-glm-tabulation-model-id', glm_js)
        self.assertIn('data-glm-tabulation-table-id', glm_js)
        self.assertIn('title: "Model name"', glm_js)
        self.assertIn('title: "Model type"', glm_js)
        self.assertIn('title: "Number of tables"', glm_js)
        self.assertIn('formatter: tabulationTableCountFormatter', glm_js)
        self.assertIn('title: "Mean error"', glm_js)
        self.assertIn('formatter: tabulationModelMetricFormatter', glm_js)
        self.assertIn('title: "linear SD error"', glm_js)
        self.assertIn('title: "missing"', glm_js)
        self.assertIn('title: "#"', glm_js)
        self.assertIn('field: "table_index"', glm_js)
        self.assertIn('title: "Table name"', glm_js)
        self.assertIn('title: "Dim"', glm_js)
        self.assertIn('title: "Cells"', glm_js)
        self.assertIn('title: "Min"', glm_js)
        self.assertIn('title: "Max"', glm_js)
        self.assertIn('title: "Span"', glm_js)
        self.assertIn('glm-tabulation-control-group glm-tabulation-control-middle', glm_js)
        self.assertIn('glm-tabulation-crosstab-group', glm_js)
        self.assertIn("function selectTabulationModel(modelId, event = {})", glm_js)
        self.assertIn("const commandSelection = Boolean(event.metaKey || event.ctrlKey);", glm_js)
        self.assertIn("if (event.shiftKey) {", glm_js)
        self.assertIn("tabulationSelectionAnchorModelId = modelRef;", glm_js)
        self.assertIn("tabulationConfig?.all_models", glm_js)
        self.assertIn("let tabulationPayload = null;", glm_js)
        self.assertIn("let tabulationModelTable = null;", glm_js)
        self.assertIn("let tabulationCommonTable = null;", glm_js)
        self.assertIn("let tabulationOtherTable = null;", glm_js)
        self.assertIn("function renderTabulationSelectorTables(options = {})", glm_js)
        self.assertIn('layout: "fitColumns"', glm_js)
        self.assertNotIn("selectableRowsCheck: (row) => !row.getData()?.tabulation_blocked_message", glm_js)
        self.assertIn("rowFormatter: formatTabulationModelSelectorRow", glm_js)
        self.assertNotIn("function showTabulationBlockedPopover", glm_js)
        self.assertNotIn('data-glm-tabulation-blocked', glm_js)
        self.assertIn("function tabulationDisplayTableValue(value)", glm_js)
        self.assertIn("function tabulationDisplayTableSpan(min, max)", glm_js)
        self.assertIn('diagnostics.classList.toggle("hidden", !html);', glm_js)
        self.assertIn("function niceTabulationAxisStep(span)", glm_js)
        self.assertIn("function roundTabulationAxisValue(value, step)", glm_js)
        self.assertIn("function formatTabulationUpliftPercent(value)", glm_js)
        self.assertIn('function formatTabulationAxisTick(value, scale = "linear")', glm_js)
        self.assertIn("function tabulationYAxisOptions(data = {})", glm_js)
        self.assertIn("const GLM_TABULATION_Y_AXIS_TARGET_INTERVALS = 15;", glm_js)
        self.assertIn("yAxis: tabulationYAxisOptions(data)", glm_js)
        self.assertIn("scale: true,", glm_js)
        self.assertIn("splitNumber: GLM_TABULATION_Y_AXIS_TARGET_INTERVALS", glm_js)
        self.assertIn("min: roundAxisValue(axisMin, step)", glm_js)
        self.assertIn("max: roundAxisValue(axisMax, step)", glm_js)
        self.assertIn("interval: roundAxisValue(step, step)", glm_js)
        self.assertIn("axisLabel: { formatter: (value) => formatAxisTick(value, data.scale) }", glm_js)
        self.assertIn('tooltip: { trigger: "axis", valueFormatter: (value) => formatTabulationAxisTick(value, data.scale) }', glm_js)
        self.assertNotIn('["mean error", diagnostics.mean_linear_error]', glm_js)
        self.assertNotIn('["linear SD error", diagnostics.linear_sd_error]', glm_js)
        self.assertNotIn('["span", tabulationSpanValue(tabulationPayload)]', glm_js)
        self.assertNotIn("${models.length.toLocaleString()} models selected", glm_js)
        self.assertNotIn("setTabulationWarnings", glm_js)
        self.assertNotIn("setTabulationWarnings(tabulationConfig?.warnings", glm_js)
        self.assertIn('element.style.setProperty("background", color, "important");', glm_js)
        self.assertIn("Boolean(column.tabulation_value)", glm_js)
        self.assertIn("function formatTabulationValue(value)", glm_js)
        self.assertIn("number.toFixed(4)", glm_js)
        self.assertIn("tabulationValue ? formatTabulationValue(value) : formatModelMetric(value)", glm_js)
        self.assertIn("tabulationValue ? formatTabulationValue(value) : (numeric ? formatModelMetric(value) : value)", glm_js)
        self.assertIn("color-mix(in srgb, hsl(${hue} 78% 50%) 28%, var(--panel))", glm_js)
        self.assertNotIn("hsl(${hue} 78% 88%)", glm_js)
        self.run_node_script(f"""
{self.shared_model_ui_source(["modelNumberOrNull", "formatModelMetric"])}
function modelNumberOrNull(value) {{ return sharedModelNumberOrNull(value); }}
function formatModelMetric(value) {{ return sharedFormatModelMetric(value); }}
{self.js_function_source(glm_tool_js, "formatTabulationValue")}
if (formatTabulationValue(-0) !== "0") throw new Error("fixed negative zero failed");
if (formatTabulationValue(-0.00001) !== "0") throw new Error("fixed rounded negative zero failed");
if (formatTabulationValue(0.25) !== "0.2500") throw new Error("fixed positive value failed");
{self.js_function_source(glm_tool_js, "tabulationCellColor")}
const lowColour = tabulationCellColor(0, 0, 10);
const highColour = tabulationCellColor(10, 0, 10);
if (!lowColour.includes("color-mix(in srgb, hsl(130 78% 50%) 28%, var(--panel))")) throw new Error(`low colour failed: ${{lowColour}}`);
if (!highColour.includes("color-mix(in srgb, hsl(0 78% 50%) 28%, var(--panel))")) throw new Error(`high colour failed: ${{highColour}}`);
if (tabulationCellColor(1, 1, 1) !== "") throw new Error("flat colour range failed");
""")
        self.assertIn('const statusField = String(column.status_field || `__status__${field}`);', glm_js)
        self.assertIn('glm-tabulation-colour-cell', glm_js)
        self.assertIn('glm-tabulation-rebase-cell', glm_js)
        self.assertNotIn('glm-tabulation-rebase-cell-selected', glm_js)
        self.assertIn('label: `${tabulationRebaseAnchorLabel(anchorCell, features)} -> ${transferFeature || "base"}`', glm_js)
        self.assertNotIn('id="glmRefreshTabulationsBtn"', glm_js)
        self.assertNotIn('>Refresh</button>', glm_js)
        self.assertNotIn('["Training rows", diagnostics.training_rows]', glm_js)
        self.assertNotIn('["Null deviance", diagnostics.null_deviance]', glm_js)
        self.assertNotIn('["BIC", diagnostics.bic]', glm_js)
        self.run_node_script(f"""
const GLM_TABULATION_Y_AXIS_TARGET_INTERVALS = 15;
let tabulationScale = "linear";
{self.shared_model_ui_source(["modelNumberOrNull"])}
function modelNumberOrNull(value) {{ return sharedModelNumberOrNull(value); }}
{self.js_function_source(glm_tabulation_js, "displayTableValue")}
{self.js_function_source(glm_tabulation_js, "displayTableSpan")}
{self.js_function_source(glm_tabulation_js, "niceAxisStep")}
{self.js_function_source(glm_tabulation_js, "roundAxisValue")}
{self.js_function_source(glm_tabulation_js, "formatUpliftPercent")}
{self.js_function_source(glm_tabulation_js, "formatAxisTick")}
{self.js_function_source(glm_tabulation_js, "yAxisOptions")}
function tabulationDisplayTableValue(value) {{ return displayTableValue(value, tabulationScale); }}
function tabulationDisplayTableSpan(min, max) {{ return displayTableSpan(min, max, tabulationScale); }}
function tabulationYAxisOptions(data) {{ return yAxisOptions(data); }}
function formatTabulationAxisTick(value, scale = "linear") {{ return formatAxisTick(value, scale); }}
	if (tabulationDisplayTableValue(0.5) !== 0.5) throw new Error("linear table min/max should stay linear");
	if (tabulationDisplayTableSpan(0.5, 1.25) !== 0.75) throw new Error("linear table span should be a difference");
	tabulationScale = "exp";
	if (Math.abs(tabulationDisplayTableValue(0) - 1) > 1e-9) throw new Error("exp table min/max should exponentiate");
	if (Math.abs(tabulationDisplayTableSpan(0.5, 1.25) - Math.exp(0.75)) > 1e-9) throw new Error("exp table span should be a ratio");
	const expAxis = tabulationYAxisOptions({{ scale: "exp", min: 1, max: 2.2114 }});
if (expAxis.interval !== 0.1) throw new Error(`expected 0.1 exp interval, got ${{expAxis.interval}}`);
if (expAxis.min !== 1 || expAxis.max !== 2.3) throw new Error(`unexpected exp bounds ${{expAxis.min}}..${{expAxis.max}}`);
if (expAxis.axisLabel.formatter(1) !== "0%") throw new Error("exp base tick should be 0%");
if (expAxis.axisLabel.formatter(1.25) !== "+25%") throw new Error("exp tick should be uplift percent");
if (expAxis.axisLabel.formatter(0.9) !== "-10%") throw new Error("negative uplift should be shown from base");
if (formatTabulationAxisTick(1.2, "linear") !== "1.2") throw new Error("linear ticks should stay numeric");
if (formatTabulationAxisTick(-0.0000001, "linear") !== "0") throw new Error("linear negative zero tick should render as zero");
""")
        self.assertIn("syncSidebarModelChooser", js)
        self.assertIn("glm_prediction", js)
        self.assertNotIn("GLM modelling will be added in a later slice", js)
        self.assertIn(".glm-tool", css)
        self.assertIn(".glm-builder-layout", css)
        self.assertIn(".glm-tabulations-panel", css)
        self.assertIn("grid-template-columns: minmax(420px, var(--glm-tabulation-sidebar-width, 1fr)) 12px minmax(420px, 1fr);", css)
        self.assertIn(".glm-grid.glm-tabulation-selector-grid", css)
        self.assertIn(".glm-tabulation-model-region", css)
        self.assertIn(".glm-tabulation-selector-resizer", css)
        self.assertIn(".glm-tabulation-table-sections", css)
        self.assertIn(".glm-tabulation-section-title", css)
        self.assertNotIn(".glm-tabulation-blocked-cell", css)
        self.assertNotIn(".glm-tabulation-blocked-message", css)
        self.assertNotIn(".glm-tabulation-blocked-fallback-cell", css)
        self.assertNotIn(".glm-tabulation-blocked-popover", css)
        self.assertNotIn(".glm-tabulation-model-list .tabulator-row.glm-tabulation-model-blocked", css)
        self.assertIn(".glm-grid.glm-tabulation-selector-grid.glm-tabulation-model-list .tabulator-row.glm-tabulation-model-untabulated .tabulator-cell", css)
        self.assertIn(".glm-grid.glm-tabulation-selector-grid.glm-tabulation-model-list .tabulator-row.tabulator-selected", css)
        self.assertIn(".glm-grid.glm-tabulation-selector-grid.glm-tabulation-table-list .tabulator-row.tabulator-selected", css)
        self.assertIn(".glm-tabulation-selector-table tr.selected", css)
        self.assertIn(".glm-tabulation-control-group", css)
        self.assertIn(".glm-tabulation-controls-row", css)
        self.assertIn("column-gap: 20px;", css)
        self.assertIn("row-gap: 8px;", css)
        self.assertIn("flex-direction: column;", css)
        self.assertIn(".glm-tabulation-control-middle", css)
        self.assertIn(".glm-tabulation-crosstab-group", css)
        self.assertIn("#glmExportTabulationsBtn", css)
        self.assertIn(".glm-coefficient-context-menu", css)
        self.assertIn(".glm-coefficient-context-menu-item", css)
        self.assertIn(".glm-tabulation-context-menu", css)
        self.assertIn(".glm-tabulation-context-menu-item", css)
        self.assertIn("max-width: min(440px, calc(100vw - 16px));", css)
        self.assertIn("white-space: normal;", css)
        self.assertNotIn(".glm-tabulation-rebase-controls", css)
        self.assertNotIn(".glm-tabulation-rebase-cell-selected", css)
        self.assertIn("--glm-tabulation-sidebar-width", css)
        self.assertIn(".glm-tabulation-resizer", css)
        self.assertIn(".glm-formula-editor", css)
        self.assertIn(".glm-coefficient-panel", css)
        self.assertIn(".glm-builder-control-stack", css)
        self.assertIn(".glm-control-line", css)
        self.assertIn(".glm-control-label", css)
        self.assertIn(".glm-family-row > .glm-control-label,\n      .glm-penalty-row > .glm-control-label", css)
        self.assertIn("flex: 0 0 42px;", css)
        self.assertIn("height: 21px;", css)
        self.assertIn("line-height: 19px;", css)
        self.assertIn("padding: 0 5px;", css)
        self.assertIn(".glm-family-parameter {\n        flex: 0 0 132px;\n        width: 132px;", css)
        self.assertIn(".glm-header-scope-control", css)
        self.assertIn(".glm-penalty-row", css)
        self.assertIn(".glm-penalty-alpha", css)
        self.assertIn(".glm-penalty-manual.disabled", css)
        self.assertIn(".glm-formula-editor {\n        background: var(--panel);", css)
        self.assertIn(".glm-formula-text {\n        background: var(--panel);\n        border: 0;\n        color: var(--text);", css)
        self.assertIn(".glm-formula-editor .ace_gutter", css)
        self.assertIn(".glm-formula-editor .ace_scroller", css)
        self.assertIn(".glm-panel-title {\n        color: var(--text);\n        flex: 0 0 auto;\n        font-size: 13px;", css)
        self.assertIn(".glm-builder-resizer", css)
        self.assertIn('.glm-build-status[data-phase="failed"] .glm-build-status-main', css)
        self.assertIn(".glm-build-status[data-phase=\"failed\"] {\n        color: var(--danger);\n        flex: 1 1 auto;", css)
        self.assertIn("max-width: none;", css)
        self.assertIn("overflow-wrap: anywhere;", css)
        self.assertIn(".glm-coefficient-actions {\n        position: absolute;", css)
        self.assertIn(".glm-coefficient-meta {\n        color: var(--muted);\n        display: flex;\n        flex-direction: column;", css)
        self.assertIn("font-weight: 500;", css)
        self.assertIn(".glm-grid.tabulator {\n        border-color: var(--glm-table-border);", css)
        self.assertIn(".glm-model-grid .tabulator-row.tabulator-selected", css)
        self.assertIn('.glm-model-grid .tabulator-cell[tabulator-field="active"] {\n        justify-content: center;', css)
        self.assertIn(".glm-model-active-dot", css)
        self.assertIn(".glm-table tbody tr.selected td", css)
        self.assertIn(".glm-model-detail {\n        color: var(--muted);\n        font-size: 10px;\n        font-weight: 400;", css)
        self.assertIn(".glm-model-list .glm-model-option.active", css)

    def test_glm_active_model_detail_syncs_builder_controls(self) -> None:
        js = self.assert_no_store("/static/app/glm-formula-builder.js")[1].decode("utf-8")
        script = "const config = { families: [] };\nfunction getFamilies() { return config.families; }\n" + "\n".join(self.js_function_source(js, name) for name in ["familyParameterConfig", "syncRegularizationControls", "syncFromModelDetail"]) + r"""
let formulaText = "";
let selectedFamily = "";
let selectedTrainingScope = "";
let selectedRegularizationMode = "none";
let selectedRegularizationMix = "0.5";
let selectedRegularizationAlpha = "0.01";
const storage = new Map();
const localStorage = {
  getItem: (key) => storage.get(key) || null,
  setItem: (key, value) => storage.set(key, String(value)),
};
const nodes = {
  glmFamilySelect: { value: "normal" },
  glmFamilyParameter: { disabled: false, value: "" },
  glmRegularizationMode: { value: "none" },
  glmRegularizationMix: { disabled: false, value: "0.5" },
  glmRegularizationAlpha: { disabled: false, value: "0.01" },
  glmRegularizationManualControls: { disabledClass: false, hidden: false, classList: { toggle: (name, active) => { if (name === "disabled") nodes.glmRegularizationManualControls.disabledClass = Boolean(active); if (name === "hidden") nodes.glmRegularizationManualControls.hidden = Boolean(active); } } },
};
function el(id) {
  return nodes[id] || null;
}
function setFormulaText(value) {
  formulaText = String(value || "");
  localStorage.setItem("py_lucidum_glm_formula", formulaText);
}
function syncFamilyParameterControl() {
  nodes.glmFamilyParameter.disabled = false;
  nodes.glmFamilyParameter.value = localStorage.getItem(`py_lucidum_glm_family_parameter_${nodes.glmFamilySelect.value}`) || "";
}
function makeScopeButton(scope) {
  const button = { dataset: { glmScope: scope }, active: false };
  button.classList = {
    toggle: (name, active) => {
      if (name === "active") button.active = Boolean(active);
    },
  };
  return button;
}
const allButton = makeScopeButton("all");
const trainingButton = makeScopeButton("training");
const document = {
  querySelectorAll: (selector) => selector === "[data-glm-scope]" ? [allButton, trainingButton] : [],
};
syncFromModelDetail({
  formula: "Age + C(Segment)",
  manifest: {
    family: "tweedie",
    family_parameter: 1.3,
    training_scope: "training",
    regularization: { mode: "manual", l1_ratio: 0.25, alpha: "0.07" },
    formula: { raw: "fallback formula" },
  },
}, { syncBuilderDraft: true });
if (formulaText !== "Age + C(Segment)") throw new Error(`formula ${formulaText}`);
if (selectedFamily !== "tweedie") throw new Error(`family ${selectedFamily}`);
if (nodes.glmFamilySelect.value !== "tweedie") throw new Error(`select ${nodes.glmFamilySelect.value}`);
if (nodes.glmFamilyParameter.value !== "1.3") throw new Error(`parameter ${nodes.glmFamilyParameter.value}`);
if (selectedTrainingScope !== "training") throw new Error(`scope ${selectedTrainingScope}`);
if (allButton.active || !trainingButton.active) throw new Error("scope buttons not synced");
if (selectedRegularizationMode !== "manual") throw new Error(`mode ${selectedRegularizationMode}`);
if (nodes.glmRegularizationMode.value !== "manual") throw new Error(`mode select ${nodes.glmRegularizationMode.value}`);
if (selectedRegularizationMix !== "0.25") throw new Error(`mix ${selectedRegularizationMix}`);
if (nodes.glmRegularizationMix.value !== "0.25") throw new Error(`mix input ${nodes.glmRegularizationMix.value}`);
if (selectedRegularizationAlpha !== "0.07") throw new Error(`alpha ${selectedRegularizationAlpha}`);
if (nodes.glmRegularizationAlpha.value !== "0.07") throw new Error(`alpha input ${nodes.glmRegularizationAlpha.value}`);
if (nodes.glmRegularizationMix.disabled) throw new Error("mix should be enabled for manual");
if (nodes.glmRegularizationAlpha.disabled) throw new Error("alpha should be enabled for manual");
if (nodes.glmRegularizationManualControls.disabledClass) throw new Error("manual controls should not be muted for manual");
if (nodes.glmRegularizationManualControls.hidden) throw new Error("manual controls should remain visible");
if (localStorage.getItem("py_lucidum_glm_formula") !== "Age + C(Segment)") throw new Error("formula storage failed");
if (localStorage.getItem("py_lucidum_glm_family") !== "tweedie") throw new Error("family storage failed");
if (localStorage.getItem("py_lucidum_glm_family_parameter_tweedie") !== "1.3") throw new Error("parameter storage failed");
if (localStorage.getItem("py_lucidum_glm_training_scope") !== "training") throw new Error("scope storage failed");
if (localStorage.getItem("py_lucidum_glm_regularization_mode") !== "manual") throw new Error("mode storage failed");
if (localStorage.getItem("py_lucidum_glm_regularization_mix") !== "0.25") throw new Error("mix storage failed");
if (localStorage.getItem("py_lucidum_glm_regularization_alpha") !== "0.07") throw new Error("alpha storage failed");

setFormulaText("draft formula");
selectedFamily = "negative.binomial";
nodes.glmFamilySelect.value = "negative.binomial";
nodes.glmFamilyParameter.value = "2.7";
localStorage.setItem("py_lucidum_glm_family", selectedFamily);
localStorage.setItem("py_lucidum_glm_family_parameter_negative.binomial", "2.7");
selectedTrainingScope = "all";
allButton.active = true;
trainingButton.active = false;
localStorage.setItem("py_lucidum_glm_training_scope", selectedTrainingScope);
selectedRegularizationMode = "manual";
selectedRegularizationMix = "1";
selectedRegularizationAlpha = "0.09";
nodes.glmRegularizationMode.value = "manual";
nodes.glmRegularizationMix.value = "1";
nodes.glmRegularizationAlpha.value = "0.09";
localStorage.setItem("py_lucidum_glm_regularization_mode", selectedRegularizationMode);
localStorage.setItem("py_lucidum_glm_regularization_mix", selectedRegularizationMix);
localStorage.setItem("py_lucidum_glm_regularization_alpha", selectedRegularizationAlpha);
syncFromModelDetail({
  formula: "different model formula",
  manifest: {
    family: "gamma",
    family_parameter: null,
    training_scope: "training",
    regularization: { mode: "none" },
  },
}, { syncBuilderDraft: false });
if (formulaText !== "draft formula") throw new Error(`draft formula overwritten: ${formulaText}`);
if (selectedFamily !== "negative.binomial") throw new Error(`draft family overwritten: ${selectedFamily}`);
if (nodes.glmFamilySelect.value !== "negative.binomial") throw new Error(`draft select overwritten: ${nodes.glmFamilySelect.value}`);
if (nodes.glmFamilyParameter.value !== "2.7") throw new Error(`draft parameter overwritten: ${nodes.glmFamilyParameter.value}`);
if (selectedTrainingScope !== "all") throw new Error(`draft scope overwritten: ${selectedTrainingScope}`);
if (!allButton.active || trainingButton.active) throw new Error("draft scope buttons overwritten");
if (selectedRegularizationMode !== "manual") throw new Error(`draft mode overwritten: ${selectedRegularizationMode}`);
if (selectedRegularizationMix !== "1") throw new Error(`draft mix overwritten: ${selectedRegularizationMix}`);
if (selectedRegularizationAlpha !== "0.09") throw new Error(`draft alpha overwritten: ${selectedRegularizationAlpha}`);
if (localStorage.getItem("py_lucidum_glm_formula") !== "draft formula") throw new Error("draft formula storage overwritten");
if (localStorage.getItem("py_lucidum_glm_family") !== "negative.binomial") throw new Error("draft family storage overwritten");
if (localStorage.getItem("py_lucidum_glm_family_parameter_negative.binomial") !== "2.7") throw new Error("draft parameter storage overwritten");
if (localStorage.getItem("py_lucidum_glm_training_scope") !== "all") throw new Error("draft scope storage overwritten");
if (localStorage.getItem("py_lucidum_glm_regularization_mode") !== "manual") throw new Error("draft mode storage overwritten");
if (localStorage.getItem("py_lucidum_glm_regularization_mix") !== "1") throw new Error("draft mix storage overwritten");
if (localStorage.getItem("py_lucidum_glm_regularization_alpha") !== "0.09") throw new Error("draft alpha storage overwritten");
"""
        self.run_node_script(script)

    def test_glm_build_failure_unlocks_button_and_uses_inline_status(self) -> None:
        js = self.assert_no_store("/static/app/glm-tool.js")[1].decode("utf-8")
        helpers = ["buildStatusHtml", "renderLiveProgress", "setBuildFailure"]
        script = "\n".join(self.js_function_source(js, name) for name in helpers) + r"""
let isBuilding = true;
let liveProgress = null;
let pollTimer = 17;
let clearedTimer = null;
let noticeText = null;
const window = {
  clearTimeout: (value) => { clearedTimer = value; },
};
function escapeHtml(value) {
  return String(value || "");
}
function makeClassList() {
  const values = new Set();
  return {
    values,
    toggle: (name, enabled) => {
      if (enabled) values.add(name);
      else values.delete(name);
    },
  };
}
const status = { innerHTML: "", dataset: {}, classList: makeClassList() };
const button = { disabled: true, textContent: "Building...", classList: makeClassList() };
function el(id) {
  if (id === "glmBuildStatus") return status;
  if (id === "glmBuildBtn") return button;
  return null;
}
function setGlmNotice(text) {
  noticeText = text;
}
setBuildFailure("Unable to evaluate factor bs(BAD, df=2)");
if (clearedTimer !== 17) throw new Error(`timer not cleared: ${clearedTimer}`);
if (pollTimer !== null) throw new Error(`poll timer not reset: ${pollTimer}`);
if (isBuilding) throw new Error("building flag still set");
if (button.disabled) throw new Error("button still disabled");
if (button.textContent !== "Build GLM") throw new Error(`button text ${button.textContent}`);
if (button.classList.values.has("building")) throw new Error("button still has building class");
if (status.dataset.phase !== "failed") throw new Error(`phase ${status.dataset.phase}`);
if (status.classList.values.has("hidden")) throw new Error("inline status hidden");
if (!status.innerHTML.includes("Unable to evaluate factor")) throw new Error(status.innerHTML);
if (noticeText !== "") throw new Error(`notice should be cleared, got ${noticeText}`);
renderLiveProgress(null);
if (!status.classList.values.has("hidden")) throw new Error("cleared status should be hidden");
if (status.dataset.phase !== "") throw new Error(`cleared phase ${status.dataset.phase}`);
if (button.disabled) throw new Error("cleared button disabled");
if (button.textContent !== "Build GLM") throw new Error(`cleared button text ${button.textContent}`);
"""
        self.run_node_script(script)

    def test_glm_tabulation_table_selection_uses_targeted_redraws(self) -> None:
        js = self.assert_no_store("/static/app/glm-tool.js")[1].decode("utf-8")
        select_source = self.js_function_source(js, "selectTabulationTable")
        bind_source = self.js_function_source(js, "bindTabulationControls")

        self.assertNotIn("renderTabulationsPanel", select_source)
        self.assertIn("if (nextTableId === selectedTabulationTableId) return;", select_source)
        self.assertIn("syncTabulationTableSelectorSelection();", select_source)
        self.assertIn("syncTabulationControls();", select_source)
        self.assertIn("loadTabulationView();", select_source)
        self.assertNotIn("renderTabulationsPanel", bind_source)
        self.assertIn("renderTabulationSelectorTables({ forceTables: true });", bind_source)
        self.assertIn("renderTabulationTable(tabulationPayload);", bind_source)
        self.assertIn("let tabulationModelSelectorSignature = \"\";", js)
        self.assertIn("let tabulationTableSelectorSignature = \"\";", js)
        self.assertIn("function syncTabulationControls()", js)

    def test_model_activation_paths_mark_activation_only(self) -> None:
        glm_js = self.assert_no_store("/static/app/glm-tool.js")[1].decode("utf-8")
        gbm_js = self.assert_no_store("/static/app/gbm-tool.js")[1].decode("utf-8")
        main_js = self.assert_no_store("/static/app/main.js")[1].decode("utf-8")

        self.assertIn("await applyModelMutationResult(result, { activationOnly: true });", self.js_function_source(glm_js, "activateModel"))
        self.assertIn("async function applyActivationOnlyTabulationUpdate(nextConfig)", glm_js)
        self.assertIn("handleExternalModelActivation,", glm_js)
        self.assertIn("onExternalModelActivation = async () => false", gbm_js)
        self.assertIn("await applyModelMutationResult(result, { activationOnly: true });", self.js_function_source(gbm_js, "activateModel"))
        self.assertIn('if (!(options?.activationOnly && await onExternalModelActivation("gbm"))) {', gbm_js)
        self.assertIn("onExternalModelActivation: (modelKind) => glmTool.handleExternalModelActivation(modelKind),", main_js)

    def test_gbm_frontend_contains_real_tool_contract(self) -> None:
        js = self.app_js_contract()
        gbm_js = self.assert_no_store("/static/app/gbm-tool.js")[1].decode("utf-8")
        css = self.app_css_contract()

        self.assertIn('import { createGbmTool } from "./gbm-tool.js";', js)
        self.assertIn('import { createGbmTreeViewer } from "./gbm-tree-viewer.js";', js)
        self.assertIn('import { createGbmShapTool } from "./gbm-shap-tool.js";', js)
        self.assertIn('import { emptyOption, ensureShapChartLibraries, shapChartOption } from "./gbm-shap-chart.js";', js)
        self.assertIn('import { loadTabulator } from "./shared/tabulator.js";', gbm_js)
        self.assertNotIn("function loadTabulator()", gbm_js)
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
        self.assertIn("function renderLiveProgress(progress, job = null)", js)
        self.assertIn("job.progress", js)
        self.assertIn('`/api/gbm/models/${encodeURIComponent(currentModelId)}/trees`', js)
        self.assertIn('`/api/gbm/models/${encodeURIComponent(currentModelId)}/trees/${encodeURIComponent(selectedTree)}`', js)
        self.assertIn('`/api/gbm/models/${encodeURIComponent(modelId)}/shap/config`', js)
        self.assertIn('`/api/gbm/models/${encodeURIComponent(modelId)}/shap/plot`', js)
        self.assertIn('"/static/vendor/tabulator/tabulator.min.js"', js)
        self.assertIn('data-gbm-shap-rescale="${value}"', js)
        self.assertNotIn('data-gbm-stacked-shap-rescale="${value}"', js)
        self.assertIn('id="gbmShapChooserDivider" class="gbm-shap-chooser-divider app-resizer app-resizer--horizontal"', js)
        self.assertIn('id="gbmShapMainResizer" class="gbm-shap-main-resizer app-resizer app-resizer--vertical"', js)
        self.assertIn('id="gbmStackedShapMainResizer" class="gbm-stacked-shap-main-resizer app-resizer app-resizer--vertical"', js)
        self.assertNotIn("py_lucidum_gbm_shap_side_width", js)
        self.assertNotIn("py_lucidum_gbm_stacked_shap_side_width", js)
        self.assertIn("rescale: state.rescale", js)
        self.assertIn("referenceLineValue(payload)", js)
        shap_control_order = [
            "${bandingControlHtml(1, feature1)}",
            "${bandingControlHtml(2, feature2)}",
            "${tailControlHtml()}",
            "${rescaleControlHtml()}",
            "${factorControlHtml(1, feature1)}",
            "${factorControlHtml(2, feature2)}",
        ]
        shap_control_positions = [js.index(item) for item in shap_control_order]
        self.assertEqual(shap_control_positions, sorted(shap_control_positions))
        tab_order = [
            '{ id: "features", label: "Features and parameters" }',
            '{ id: "models", label: "Model navigator" }',
            '{ id: "shap", label: "SHAP" }',
            '{ id: "stacked-shap", label: "Stacked SHAP" }',
            '{ id: "trees", label: "Tree viewer" }',
        ]
        tab_positions = [js.index(item) for item in tab_order]
        self.assertEqual(tab_positions, sorted(tab_positions))
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
        self.assertIn("Treat Feature ${index} as factor", js)
        self.assertIn('? "shap"', js)
        self.assertIn("`Rank ${rank}`", js)
        self.assertNotIn("function featureKindLabel", js)
        self.assertIn(".gbm-shap-chart-shell", css)
        self.assertIn("grid-template-columns: minmax(240px, var(--gbm-shap-side-width, 320px)) 12px minmax(0, 1fr);", css)
        self.assertIn("grid-template-columns: minmax(240px, var(--gbm-stacked-shap-side-width, 320px)) 12px minmax(0, 1fr);", css)
        self.assertIn(".gbm-shap-message {\n        background:", css)
        self.assertIn("position: absolute;\n        right: 12px;\n        top: 12px;", css)
        self.assertIn(".gbm-shap-controls {\n        display: grid;\n        grid-template-columns: repeat(4, max-content);\n        gap: 6px 12px;\n        justify-content: flex-start;", css)
        self.assertIn(".gbm-shap-controls .gbm-shap-feature2-control {\n        align-items: flex-start;\n        grid-column: 2;\n        grid-row: 1;\n        text-align: left;", css)
        self.assertIn(".gbm-shap-controls .gbm-shap-tail-control {\n        align-items: flex-start;\n        grid-column: 3;\n        grid-row: 1;\n        text-align: left;", css)
        self.assertIn(".gbm-shap-controls .gbm-shap-rescale-control {\n        align-items: flex-start;\n        grid-column: 4;\n        grid-row: 1;\n        text-align: left;", css)
        self.assertIn(".gbm-shap-feature2-control .segmented {\n        justify-content: flex-start;", css)
        self.assertIn(".gbm-shap-tail-control .segmented {\n        justify-content: flex-start;", css)
        self.assertIn(".gbm-shap-rescale-control .segmented {\n        justify-content: flex-start;", css)
        self.assertIn(".gbm-shap-feature1-factor {\n        grid-column: 1;\n        grid-row: 2;\n        justify-self: start;", css)
        self.assertIn(".gbm-shap-feature2-factor {\n        grid-column: 2;\n        grid-row: 2;\n        justify-self: start;\n        text-align: left;", css)
        self.assertIn(".gbm-stacked-shap-banding-control {\n        align-items: flex-start;\n        text-align: left;", css)
        self.assertNotIn(".gbm-stacked-shap-rescale-control", css)
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
        self.assertIn('id="gbmTreeResizer" class="gbm-tree-resizer app-resizer app-resizer--vertical"', js)
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
        self.assertIn("selectExpectedPredictionForModelKind = () => false", gbm_js)
        self.assertGreaterEqual(
            js.count("selectExpectedPredictionForModelKind: (modelKind) => setExpectedPredictionSelectionForModelKind(modelKind)"),
            2,
        )
        self.assertIn("function canNavigateToLineBarFeature(featureName)", js)
        self.assertIn("function navigateToLineBarFeature(featureName)", js)
        self.assertIn("state.bandFeature = null;", js)
        self.assertIn('selectExpectedPredictionForModelKind("gbm");', js)
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
        self.assertIn("function featureInteractionPairsDropdownHtml(activeConstraints = null, features = [])", js)
        self.assertIn('id="gbmFeatureInteractionPairButton"', js)
        self.assertIn("function currentFeatureInteractionPairsPayload()", js)
        self.assertIn('label: "Allow interaction pair"', js)
        self.assertIn("const draftGroupings = draft?.interactionGroupingsEdited ? new Set(draft.interactionGroupings || []) : null;", js)
        self.assertIn("function renderedInteractionFeatureNames(features)", js)
        self.assertIn("function renderedPairInteractionFeatureNames(features)", js)
        self.assertIn("const groupLocked = renderedInteractionFeatureNames(features);", js)
        self.assertIn("const featureLocked = draft ? selectedFeatureInteractionFeatureNames(features) : activeFeatureInteractionFeatureNames();", js)
        self.assertIn("const pairLocked = renderedPairInteractionFeatureNames(features);", js)
        self.assertIn("function selectedPairInteractionFeatureNames(features = currentFeatureRows())", js)
        self.assertIn("pair_interaction_locked: !featureLocked.has(feature.name) && pairLocked.has(feature.name)", js)
        self.assertIn("gbm-pair-interaction-lock", js)
        self.assertIn("gbm-interaction-lock-subscript", js)
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
        self.assertIn("payload.feature_interaction_pairs = featureInteractionPairs;", js)
        self.assertIn("if (currentModelId !== nextModelId) {\n      featureDraftState = null;\n      featureInteractionPairEditModelId = \"\";", js)
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
        self.assertIn('{ id: "shap", label: "SHAP" }', js)
        self.assertIn('class="gbm-toolbar"', js)
        self.assertIn('id="gbmTrainingStatus" class="gbm-training-status', js)
        self.assertIn("setTrainingStatus(progress.message || \"\", progress.phase || \"\", trainingStatusDetail(progress, job));", js)
        self.assertIn("function trainingStatusDetail(progress, job = null)", js)
        self.assertIn("function formatGbmElapsedDuration(value)", js)
        self.assertIn("grid_parameters", js)
        self.assertIn("renderEvaluationChart({", js)
        self.assertIn("progress.evaluation", js)
        self.assertIn('pollJob(job.job_id, 0);', js)
        self.assertIn("isModelJobPending(job.status)", js)
        self.assertIn("modelJobPollDelay(job.status, GBM_QUEUED_POLL_MS, GBM_RUNNING_POLL_MS)", js)
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
        self.assertIn('document.querySelector("input[name=\'gbmShapRows\']:checked")?.value || "100k"', js)
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
        self.assertIn('import { createGbmEvaluationChart } from "./gbm-evaluation-chart.js";', js)
        self.assertIn("const evaluationChart = createGbmEvaluationChart({ escapeHtml, formatEvaluationValue });", js)
        self.assertIn("let chart = null;", js)
        self.assertIn("function bindResize(target)", js)
        self.assertIn("resizeObserver = observeResize([target, target.parentElement]", js)
        self.assertIn("chart?.resize()", js)
        self.assertIn("function evaluationTitle(rows, primaryMetric, manifest = {}, progress = null)", js)
        self.assertIn("rows.sort(compareEvaluationRows);", js)
        self.assertIn("function compareEvaluationRows(left, right)", js)
        self.assertIn('let viewMode = "all";', js)
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
        self.assertIn('{ title: "Parameter", field: "name", widthGrow: 1 }', js)
        self.assertIn('{ title: "Value", field: "value", formatter: parameterValueFormatter, editor: "adaptable", editorParams: parameterValueEditorParams(), widthGrow: 2 }', js)
        self.assertIn("function initScoreOptions(options)", js)
        self.assertIn("function groupedInitScoreOptions(options)", js)
        self.assertIn('"GLM PREDICTIONS"', js)
        self.assertIn('"DATASET COLUMNS"', js)
        self.assertIn("values: editorValues(rowData.name)", js)
        self.assertIn("function parameterValueEditorParams()", js)
        self.assertIn("function parameterValueFormatter(cell)", js)
        self.assertIn("function parameterValueEditorLookup(cell)", js)
        self.assertIn("function parameterValueEditorParamsLookup(editor, cell)", js)
        self.assertIn("editorLookup: valueEditorLookup", js)
        self.assertIn("paramsLookup: valueEditorParamsLookup", js)
        self.assertIn("class: `gbm-parameter-editor gbm-parameter-${editor}-editor`,", js)
        self.assertIn("function parameterControlHtml(parameter)", js)
        self.assertIn("function parameterSelectOptionsHtml(name, options, value)", js)
        self.assertIn("function parameterOptgroupHtml(label, options, value)", js)
        self.assertIn("<select data-gbm-parameter=", js)
        self.assertIn("<optgroup label=", js)
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
        self.assertIn(".gbm-pair-interaction-lock", css)
        self.assertIn(".gbm-interaction-lock-subscript", css)
        self.assertIn(".gbm-feature-scenario-select", css)
        self.assertIn(".gbm-feature-scenario-row", css)
        self.assertIn(".gbm-evaluation-view-mode", css)
        self.assertIn(".gbm-evaluation-view-option", css)
        self.assertIn("accent-color: var(--accent);", css)
        self.assertIn(".gbm-inline-action-button", css)
        self.assertIn(".gbm-icon-action-button", css)
        self.assertIn(".gbm-feature-menu-button.has-constraints", css)
        self.assertIn(".gbm-interaction-pair-menu", css)
        self.assertIn("width: min(620px, calc(100vw - 32px));", css)
        self.assertIn("grid-template-columns: minmax(180px, 1fr) auto minmax(180px, 1fr) auto;", css)
        self.assertIn(".gbm-interaction-pair-row", css)
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
        self.assertIn(".app-resizer:hover::before,\n      .app-resizer.dragging::before", css)
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
        self.assertIn("function syncDatasetGbmCountFromConfig(data = {})", js)
        self.assertIn("syncDatasetGbmCountFromConfig(nextConfig);", js)
        self.assertIn(".gbm-model-table {\n        font-size: 11px;\n        line-height: 1.15;", css)
        self.assertIn("min-width: 1620px;", css)
        self.assertIn(".gbm-model-table td {\n        border-right: 1px solid color-mix(in srgb, var(--line) 80%, transparent);", css)
        self.assertNotIn(".gbm-model-activate-button", css)
        self.assertIn(".gbm-parameter-select", css)
        self.assertIn(".gbm-fallback-table select", css)
        self.assertNotIn('document.querySelector(".sidebar-metric-section")?.classList.toggle("hidden", tool === "dataset_viewer" || tool === "column_profile" || tool === "specs");', js)
        self.assertIn("function toolUsesMetricControls(tool = state.tool)", js)
        self.assertIn("async function refreshMetricSummary(options = {})", js)
        self.assertIn('api("/api/metrics/summary", {', js)
        self.assertNotIn("function toolRendersMetricSummaries(tool = state.tool)", js)
        self.assertIn("function refreshActiveToolForMetricChange()", js)
        self.assertIn("syncSidebarAccordion();", js)
        self.assertNotIn('document.querySelector(".sidebar-kpi-section")?.classList.toggle("hidden", tool === "column_profile");', js)
        self.assertNotIn('document.querySelector(".sidebar-filter-section")?.classList.toggle("hidden", isModelTool(tool));', js)
        self.assertIn('el("modelToolGroupMeta").classList.toggle("hidden", !isModelTool(tool) || tool === "gbm");', js)
        self.assertIn('el("modelToolFilter").classList.add("hidden");', js)
        self.assertIn('response: el("actualNumerator")?.value || "actualNumerator"', js)
        self.assertIn('offset: el("denominator")?.value || "denominator"', js)
        self.assertNotIn("const glmSourcesAvailable = (state.schema?.data_sources || []).some", js)
        self.assertIn("glmTool.syncSidebarFromSchema();", js)
        self.assertNotIn("const gbmSourcesAvailable = (state.schema?.data_sources || []).some", js)
        self.assertIn("gbmTool.syncSidebarFromSchema();", js)
        self.assertIn("function syncSidebarFromSchema()", js)
        self.assertIn("function syncSidebarModelChooser(models, activeModelId)", js)
        self.assertIn("function modelGroupLabel(model)", js)
        self.assertIn("const modelsByGroup = modelGroups(normalisedModels, modelGroupLabel);", js)
        self.assertIn("state.gbmModelGroupsInitialised = syncCollapsedModelGroups({", js)
        self.assertIn('return `${model.response_column || "actualNumerator"} / ${modelWeightLabel(model.offset_column)}`;', js)
        self.assertIn("function modelDetailLabel(model)", js)
        self.assertIn("return gbmModelDetailLabel(model);", js)
        self.assertIn('parts.push(`train ${formatModelMetric(modelBestMetric(model, "training"))}`);', js)
        self.assertIn('parts.push(`test ${formatModelMetric(modelBestMetric(model, "test"))}`);', js)
        self.assertIn("createSidebarModelHeading({", js)
        self.assertIn('className: "gbm-model-theme"', js)
        self.assertIn("createSidebarModelOption({", js)
        self.assertIn('className: "gbm-model-option"', js)
        self.assertIn("toggleSidebarModelGroup({", js)
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
        self.assertIn("bindFallbackModelSelection(fallbackRows, onFallbackSelectionChange);", js)
        self.assertIn("function formatModelRuntime(model)", js)
        self.assertIn("function formatModelMetric(value)", js)
        self.assertIn("function modelParameterNumber(model, name)", js)
        self.assertIn('model?.timings?.training_seconds', js)
        self.assertIn("function formatModelCreated(value)", js)
        self.assertIn('formatModelCreated as sharedFormatModelCreated', js)
        self.assertIn('return `${date.getDate()} ${MODEL_MONTHS[date.getMonth()]} ${hour}:${minute}`;', js)
        self.assertIn("function modelWeightLabel(value)", js)
        self.assertIn('return !text || text === "__none__" || text === "Average row value" ? "N" : text;', js)
        self.assertIn("function formatSampleMode(value, source = \"\")", js)
        self.assertIn('el("gbmActivateModelBtn")?.addEventListener("click", activateSelectedModel);', js)
        self.assertIn("syncSharedModelActionButtons({", js)
        self.assertIn('activate: el("gbmActivateModelBtn")', js)
        self.assertIn("async function activateSelectedModel()", js)
        self.assertIn("if (modelIds.length !== 1) return;", js)
        self.assertIn("await activateModel(modelIds[0]);", js)
        self.assertIn("function renameActiveModel()", js)
        self.assertIn("function deleteActiveModel()", js)
        self.assertIn("function selectedModelIds()", js)
        self.assertIn("selectedCount: selectedModelIds().length,", js)
        self.assertIn("const [modelId] = selectedModelIds();", js)
        self.assertIn("const modelIds = selectedModelIds();", js)
        self.assertIn("for (const modelId of modelIds)", js)
        self.assertIn("selectedModelIdsFromTableOrFallback({", js)
        self.assertIn("restoreSharedModelSelection({", js)
        self.assertIn('method: "DELETE"', js)
        self.assertIn("button.setAttribute(\"aria-selected\", String(active));", js)
        self.assertIn("onActivate: activateModel", js)
        self.assertIn("if (!active) onActivate(model.model_id, model);", js)
        self.assertIn("response_column: source.response_column", js)
        self.assertIn("offset_column: source.offset_column", js)
        self.assertIn("training_mode: source.training_mode", js)
        self.assertIn("function sourceColumns()", js)
        self.assertIn("function isModelPredictionColumn(column)", js)
        self.assertIn('return ["gbm_prediction", "gbm_prediction_rate", "gbm_tabulated_prediction", "glm_prediction", "glm_prediction_rate", "glm_tabulated_prediction"].includes(String(column?.name || ""));', js)
        self.assertIn("function expectedColumns()", js)
        self.assertIn("function expectedPredictionColumns()", js)
        self.assertIn("option.dataset.sourceId = col.source_id || state.source || \"dataset\";", js)
        self.assertIn("option.dataset.metricKind = isModelPredictionColumn(col) ? \"prediction\" : \"metric\";", js)
        self.assertIn('const expectedKeys = ["token", "tool", "source", "x", "xSource", "actual", "expected", "expected2", "denominator"', js)
        self.assertIn("expectedSelections: [],", js)
        self.assertIn("function normaliseExpectedSelections(selections = [], options = {})", js)
        self.assertIn("function setExpectedSelections(selections = [], options = {})", js)
        self.assertIn("function setExpectedSelection(value, sourceId = \"\", options = {})", js)
        self.assertIn("function predictionColumnNamesForModelKind(modelKind)", js)
        self.assertIn('if (modelKind === "glm") return ["glm_prediction", "glm_prediction_rate"];', js)
        self.assertIn('if (modelKind === "gbm") return ["gbm_prediction", "gbm_prediction_rate"];', js)
        self.assertIn("function syncExpectedSourceFromSelection({ expectedValue = \"\", expectedSource = \"\", expectedSelections: nextSelections = null } = {})", js)
        self.assertIn("syncControlsForSourceChange({", js)
        self.assertIn("const disabled = Boolean(value && maxSelected && !isActive);", js)
        self.assertIn('button.setAttribute("aria-pressed", String(isActive));', js)
        self.assertIn("function expectedDisplayColumns()", js)
        self.assertIn('expectedColumns().filter((column) => column.source_role !== "gbm_shap_value")', js)
        self.assertIn("function compareMetricColumns(a, b)", js)
        self.assertIn("function sortedMetricColumns(columns = [])", js)
        self.assertIn("for (const col of sortedMetricColumns(numericColumns()))", js)
        self.assertIn("function sortedDenominatorColumns()", js)
        self.assertIn("return sortedMetricColumns(numericColumns().map((column) => ({ ...column, label: column.name })));", js)
        self.assertIn("for (const col of sortedDenominatorColumns())", js)
        self.assertIn('select.append(new Option("Average row value", "__none__"));', js)
        self.assertIn('const LINE_BAR_SPECIAL_COLUMN_NAMES = [', js)
        self.assertIn('"gbm_to_glm_ratio",', js)
        self.assertIn('"glm_tabulated_prediction",', js)
        self.assertIn('"gbm_tabulated_prediction",', js)
        self.assertIn("function isLineBarSpecialColumn(column)", js)
        self.assertIn("function isGbmGlmRatioColumn(column)", js)
        self.assertIn("function orderedLineBarSpecialColumns(columns)", js)
        self.assertIn("function lineBarSpecialColumnOrder(column)", js)
        self.assertIn("function activeModelRatioColumns()", js)
        self.assertIn("for (const column of [...currentModelColumns, ...activeModelRatioColumns(), ...activePredictionColumns()])", js)
        self.assertIn("const ratioColumns = orderedLineBarSpecialColumns([...sourceColumns()]).filter", js)
        self.assertIn('const xSource = column && (isModelPredictionColumn(column) || sourceId !== (state.source || "dataset")) ? sourceId : "";', js)
        self.assertIn('if (state.expectedSort === "alpha") {', js)
        self.assertIn('otherColumns.sort((a, b) => a.name.localeCompare(b.name, undefined, { sensitivity: "base" }));', js)
        self.assertIn("return { specialColumns, otherColumns };", js)
        self.assertIn('addExpectedButton(pinned, "No expected line", "", "off", "", "expected-none-option line-bar-special-row");', js)
        self.assertIn('addExpectedButton(pinned, col.name, col.name, col.kind, col.source_id || state.source || "dataset", "line-bar-special-row");', js)
        self.assertIn("function resetLineBarPickerList(list, split)", js)
        self.assertIn('pinned.className = "line-bar-pinned-region";', js)
        self.assertIn('scroll.className = "line-bar-scroll-region";', js)
        self.assertIn("button.dataset.sourceId = sourceId;", js)
        self.assertIn("function addLineBarFeatureButton(list, col, extraClass = \"\")", js)
        self.assertIn('if (isRawDatasetFeature) state.source = "dataset";', js)
        self.assertIn("addLineBarFeatureButton(scroll, col);", js)
        self.assertIn("button.dataset.value = value;", js)
        self.assertIn("button.dataset.value = label;", js)
        self.assertIn("const sourceChanged = syncExpectedSourceFromSelection({", js)
        self.assertIn("for (const selection of expectedSelections())", js)
        self.assertIn('const secondExpectedColor = getCss("--accent") || "#2276d2";', js)
        self.assertIn("const responseColors = [actualColor, expectedColor, secondExpectedColor];", js)
        self.assertIn('classList.add("metric-value--first-expected");', js)
        self.assertIn('valueSpan.className = "metric-value metric-value--second-expected";', js)
        self.assertIn("for (const col of specialColumns)", js)
        self.assertIn("for (const col of otherColumns)", js)
        self.assertIn("function preferredStartupSource(availableSources, requestedSource)", js)
        self.assertIn('const requested = String(requestedSource || "").trim();', js)
        self.assertIn('return "dataset";', js)
        self.assertIn("state.source = preferredStartupSource(availableSources, requestedSource);", js)
        self.assertIn('source: state.source || "dataset"', js)
        self.assertIn("const previousExpectedSelections = expectedSelectionsSnapshot();", js)
        self.assertIn('fillMetricSelect(el("expectedNumerator"), true);', js)
        self.assertIn("restoreExpectedSelectionsAfterModelMutation(previousExpectedSelections, modelKind);", js)

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
        self.assertIn("disabled: isTraining,", js)
        self.assertIn('rename: el("gbmRenameModelBtn")', js)
        self.assertIn('deleteButton: el("gbmDeleteModelBtn")', js)
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
        self.assertIn(".visual-area.specs-mode {\n        grid-template-columns: minmax(0, 1fr);", css)
        self.assertIn(".workspace-meta,\n      .chart-message {\n        color: var(--muted);\n        font-size: 10px;", css)

    def test_specifications_table_uses_glm_model_navigator_style(self) -> None:
        js = self.app_js_contract()
        spec_js = self.assert_no_store("/static/app/specifications-tool.js")[1].decode("utf-8")
        css = self.app_css_contract()

        self.assertNotIn("selectableRange", js)
        self.assertNotIn("clipboardCopyRowRange", js)
        self.assertNotIn("clipboardPasteParser", js)
        self.assertNotIn("clipboardPasteAction", js)
        self.assertNotIn("setStatus", spec_js)
        self.assertIn('clearGlobalStatus: () => setStatus("")', js)
        self.assertIn("datasetColumnNames,", js)
        self.assertIn("function datasetColumnNames()", js)
        self.assertIn("datasetColumnNames = () => []", spec_js)
        self.assertIn("function featureRowMissingDatasetFeature", spec_js)
        self.assertIn("function applyMissingFeatureRowClasses()", spec_js)
        self.assertIn('classList.toggle("spec-missing-feature-row"', spec_js)
        self.assertIn("rowIssues: normaliseRowIssues(payload.row_issues)", spec_js)
        self.assertIn("function applyValidationResultRowIssues(result, spec = specs.get(activeKind))", spec_js)
        self.assertIn("const AUTO_VALIDATION_DELAY_MS = 250;", spec_js)
        self.assertIn("let validationTimer = null;", spec_js)
        self.assertIn("let validationRequestId = 0;", spec_js)
        self.assertIn("async function validateSpecOnLoad(spec)", spec_js)
        self.assertIn("async function validateSpecDraft(spec)", spec_js)
        self.assertIn("function scheduleValidationForActiveSpec()", spec_js)
        self.assertIn("function cancelPendingValidation()", spec_js)
        self.assertIn("function storeValidationResult(spec, result)", spec_js)
        self.assertIn("function showValidationNotice(result)", spec_js)
        self.assertIn("validationPending: false,", spec_js)
        self.assertIn("function specSaveButtonTitle(spec, canSave)", spec_js)
        self.assertIn("if (spec.validationPending || spec.validationResult?.valid === false) return false;", spec_js)
        self.assertIn('return "Checking specification before saving";', spec_js)
        self.assertIn('return "Fix validation errors before saving";', spec_js)
        self.assertIn('return canSave ? "Save specification" : "";', spec_js)
        self.assertIn("button.title = specSaveButtonTitle(spec, canSave);", spec_js)
        self.assertIn("spec.validationPending = true;", spec_js)
        self.assertIn("spec.validationPending = false;", spec_js)
        self.assertIn("Save failed; file was not written:", spec_js)
        self.assertIn("await validateSpecOnLoad(spec);", spec_js)
        self.assertIn("await validateSpecOnLoad(cached);", spec_js)
        self.assertNotIn("syncGenerationNotice", spec_js)
        self.assertNotIn("generation_message", spec_js)
        self.assertIn("function specCanSave(spec)", spec_js)
        self.assertIn("return Boolean(spec.dirty || (spec.generated && !spec.exists));", spec_js)
        self.assertIn("button.disabled = loading || !canSave;", spec_js)
        self.assertIn('button.classList.toggle("pending", !dirty && canSave);', spec_js)
        self.assertIn("if (loading || !specCanSave(specs.get(activeKind))) return;", spec_js)
        self.assertIn("if (!spec || spec.dirty || spec.autoValidated) return;", spec_js)
        self.assertIn("return `Save target: ${path} (new file)`;", spec_js)
        self.assertIn("return `Save target: ${path} (existing file ignored by ${disabledSpecFlag(spec.kind)})`;", spec_js)
        self.assertIn("return `Editing: ${path}`;", spec_js)
        self.assertIn("function clearValidationRowIssuesForActiveSpec()", spec_js)
        self.assertIn("function applyValidationRowIssueClasses()", spec_js)
        self.assertIn('classList.toggle("spec-validation-issue-row"', spec_js)
        self.assertIn("placeholders: payload.placeholders && typeof payload.placeholders === \"object\" ? payload.placeholders : {},", spec_js)
        self.assertIn('class="spec-cell-placeholder"', spec_js)
        self.assertIn("formatter: (cell) => textFormatter(cell, spec)", spec_js)
        self.assertIn('class="spec-control-row"', js)
        self.assertNotIn('class="spec-title-stack"', js)
        self.assertIn('class="tabs spec-kind-tabs"', js)
        self.assertNotIn("specValidateBtn", js)
        self.assertNotIn(">Validate</button>", js)
        self.assertIn('<span id="specFilePath" class="spec-file-path"></span>', js)
        self.assertIn('<div id="specNotice" class="spec-notice spec-notice-empty" role="status" aria-live="polite"></div>', js)
        self.assertNotIn("specGenerationNotice", js)
        self.assertNotIn("spec-generation-notice", js)
        self.assertLess(js.index('<div class="spec-file-actions">'), js.index('<span id="specFilePath" class="spec-file-path"></span>'))
        self.assertLess(js.index('<span id="specFilePath" class="spec-file-path"></span>'), js.index('<div id="specNotice" class="spec-notice spec-notice-empty" role="status" aria-live="polite"></div>'))
        self.assertLess(js.index('<button id="specSaveBtn"'), js.index('<span id="specFilePath"'))
        self.assertIn('class="tab ${kind.id === activeKind ? "active" : ""}"', js)
        self.assertNotIn(".spec-generation-notice", css)
        self.assertIn(".spec-save-button.dirty,\n      .spec-save-button.pending {", css)
        self.assertIn("align-self: flex-end;", css)
        self.assertIn("text-align: right;", css)
        self.assertIn("width: auto;", css)
        self.assertIn(".spec-cell-placeholder", css)
        self.assertIn('id="specContextMenu" class="spec-context-menu" role="menu" hidden', js)
        self.assertIn('id="specColumnContextMenu" class="spec-context-menu" role="menu" hidden', js)
        self.assertIn('class="spec-context-menu-item" type="button" role="menuitem"', js)
        self.assertNotIn("specScenarioToolbar", js)
        self.assertNotIn("specScenarioSelect", js)
        self.assertNotIn("specAddScenarioBtn", js)
        self.assertNotIn("specRenameScenarioBtn", js)
        self.assertNotIn("specRemoveScenarioBtn", js)
        self.assertNotIn(".spec-scenario-toolbar", css)
        self.assertIn('table.on("headerContext", openColumnContextMenu);', js)
        self.assertIn('["add-before", "Add scenario before"]', js)
        self.assertIn('["add-after", "Add scenario after"]', js)
        self.assertIn('["delete", "Delete scenario"]', js)
        self.assertIn('["rename", "Rename scenario"]', js)
        self.assertIn('[["add-end", "Add scenario"]]', js)
        self.assertIn('function addScenarioAt(referenceField = "", position = "end")', js)
        self.assertIn("function renameScenarioField(oldName)", js)
        self.assertIn("function deleteScenarioField(name)", js)
        self.assertIn("const title = columnTitle(spec, field);", spec_js)
        self.assertIn('layout: "fitDataStretch"', spec_js)
        self.assertNotIn('layout: "fitColumns"', spec_js)
        self.assertIn("rowHeader: specRowHeader(),", spec_js)
        self.assertIn("function specRowHeader()", spec_js)
        self.assertIn('field: "_spec_row_number"', spec_js)
        self.assertIn("formatter: rowNumberFormatter", spec_js)
        self.assertIn("cssClass: \"spec-row-number-cell\"", spec_js)
        self.assertIn("function rowNumberFormatter(cell)", spec_js)
        self.assertIn("next._spec_row_number = index + 2;", spec_js)
        self.assertIn("function renumberSpecRows(spec)", spec_js)
        self.assertIn("const title = columnTitle(spec, field);", spec_js)
        self.assertIn("minWidth: columnMinWidth(spec, field, title)", spec_js)
        self.assertIn("function columnMinWidth(spec, field, title = columnTitle(spec, field))", spec_js)
        self.assertIn("function headerMinWidth(title)", spec_js)
        self.assertNotIn("column.width = 116", spec_js)
        self.assertNotIn("column.width = 92", spec_js)
        self.assertIn("function scenarioHeaderTitle(spec, field)", js)
        self.assertIn("return `${field} (${scenarioSelectionCount(spec, field)})`;", js)
        self.assertIn("function scenarioSelectionCount(spec, field)", js)
        self.assertIn("return scenarioCountRows(spec).filter((row) => scenarioCellSelected(row?.[field])).length;", js)
        self.assertIn('const titleElement = header.querySelector(".tabulator-col-title");', js)
        self.assertIn("if (titleElement) titleElement.textContent = title;", js)
        self.assertNotIn("updateColumnDefinition", spec_js)
        self.assertNotIn("setColumns", spec_js)
        self.assertNotIn("if (isScenarioField(cell.getField())) table.redraw(true);", spec_js)
        self.assertIn("let pendingScrollRestore = null;", spec_js)
        self.assertIn("function refreshTheme() {\n    const scrollPosition = captureSpecTableScroll();\n    table?.redraw?.(false);\n    scheduleSpecTableScrollRestore(scrollPosition);", spec_js)
        self.assertIn("function scheduleSpecTableScrollRestore(position)", spec_js)
        self.assertIn("function restorePendingSpecTableScroll()", spec_js)
        self.assertIn("restorePendingSpecTableScroll();", spec_js)
        self.assertIn("renderSpec(spec, { preserveScroll: true });", spec_js)
        delete_row_source = spec_js[spec_js.index("async function deleteContextRow()"):spec_js.index("function newRow", spec_js.index("async function deleteContextRow()"))]
        self.assertIn("renderSpec(spec, { preserveScroll: true });", delete_row_source)
        self.assertIn("await validateSpecDraft(spec);", delete_row_source)
        self.assertNotIn("clearValidationRowIssuesForSpec", delete_row_source)
        self.assertIn('const field = column?.getField?.() || "";', js)
        self.assertIn(".glm-tabulation-context-menu,\n      .spec-context-menu {", css)
        self.assertIn(".glm-tabulation-context-menu-item,\n      .spec-context-menu-item {", css)
        spec_tool_css = css[css.index(".spec-tool {"):css.index("      .spec-topbar", css.index(".spec-tool {"))]
        self.assertIn("background: var(--panel);", spec_tool_css)
        self.assertIn("border: 0;", spec_tool_css)
        self.assertIn("border-radius: 0;", spec_tool_css)
        self.assertIn("box-shadow: none;", spec_tool_css)
        self.assertIn("padding: 0;", spec_tool_css)
        self.assertNotIn("border: 1px solid var(--line);", spec_tool_css)
        self.assertNotIn("box-shadow: var(--shadow);", spec_tool_css)
        self.assertIn(".spec-topbar {\n        align-items: stretch;\n        display: flex;\n        flex: 0 0 auto;\n        flex-direction: column;", css)
        self.assertIn(".spec-control-row {\n        align-items: flex-start;\n        display: flex;\n        flex: 0 0 auto;\n        gap: 12px;\n        justify-content: space-between;", css)
        self.assertNotIn(".spec-title-stack", css)
        self.assertIn(".spec-kind-tabs .tab {\n        font-weight: 700;", css)
        self.assertIn(".spec-file-actions {\n        align-items: center;\n        display: flex;\n        flex: 0 0 auto;\n        gap: 8px;\n        justify-content: flex-end;\n        max-width: none;", css)
        self.assertIn(".spec-file-path {\n        align-self: stretch;\n        color: var(--muted);\n        display: block;\n        flex: 0 0 auto;\n        font-size: 12px;\n        font-weight: 400;", css)
        self.assertIn("text-overflow: ellipsis;\n        white-space: nowrap;\n        width: 100%;", css)
        self.assertIn(".spec-notice {\n        align-self: flex-end;\n        background: transparent;\n        border: 0;\n        color: #15803d;\n        display: block;\n        flex: 0 0 auto;\n        font-size: 12px;\n        font-weight: 400;", css)
        self.assertIn("line-height: 1.25;\n        margin: 0;\n        max-width: 100%;\n        min-height: 15px;", css)
        self.assertIn("text-align: right;\n        text-overflow: ellipsis;\n        white-space: nowrap;\n        width: auto;", css)
        self.assertIn(".spec-notice.error {\n        color: var(--danger);", css)
        self.assertIn(".spec-notice-empty {\n        visibility: hidden;", css)
        self.assertNotIn(".spec-notice.hidden", css)
        self.assertNotIn(".spec-notice.warning", css)
        self.assertNotIn(".spec-notice strong", css)
        self.assertNotIn(".spec-notice ul", css)
        self.assertIn("--spec-table-border: #999;", css)
        self.assertIn(".spec-grid {\n        --spec-table-border: #999;\n        background: var(--panel);\n        border: 1px solid var(--spec-table-border);\n        border-radius: 6px;", css)
        self.assertIn("user-select: none;\n        -webkit-user-select: none;", css)
        self.assertIn(".spec-grid.tabulator {\n        border-color: var(--spec-table-border);", css)
        self.assertIn(".spec-grid .tabulator-header {\n        background: var(--panel-2) !important;", css)
        self.assertIn(".spec-grid .tabulator-header .tabulator-col .tabulator-col-title {\n        overflow: visible;\n        text-overflow: clip;\n        white-space: nowrap;", css)
        self.assertIn(".glm-grid.tabulator .tabulator-header .tabulator-col {\n        font-size: 11px;", css)
        self.assertIn(".glm-grid .tabulator-row .tabulator-cell {\n        align-items: center;", css)
        self.assertIn("font-size: 11px;\n        line-height: 1.15;\n        min-height: 20px;\n        padding: 1px 6px;", css)
        self.assertIn(".spec-grid .tabulator-col,\n      .spec-grid .tabulator-header .tabulator-col {\n        background: var(--panel-2) !important;\n        border-color: var(--line) !important;\n        color: var(--text) !important;\n        font-size: 11px;\n        justify-content: center;\n        line-height: 1.15;\n        min-height: 20px;", css)
        self.assertIn(".spec-grid .tabulator-row .tabulator-cell {\n        align-items: center;\n        background: transparent !important;\n        border-right-color: color-mix(in srgb, var(--line) 80%, transparent);\n        color: var(--text) !important;\n        display: inline-flex;\n        font-size: 11px;\n        line-height: 1.15;\n        min-height: 20px;\n        padding: 1px 6px;", css)
        self.assertIn(".spec-grid .tabulator-row .tabulator-cell.spec-row-number-cell {\n        align-items: center;\n        color: var(--muted) !important;\n        display: inline-flex !important;\n        justify-content: center;\n        padding: 1px 4px;\n        text-align: center;", css)
        self.assertIn(".spec-grid .tabulator-row.spec-missing-feature-row .tabulator-cell,\n      .spec-grid .tabulator-row.spec-validation-issue-row .tabulator-cell {\n        background: #fff8d7 !important;\n        color: #1f2937 !important;", css)
        self.assertIn(".spec-grid .tabulator-row .tabulator-cell.spec-cell-selected {\n        background: color-mix(in srgb, var(--accent) 12%, var(--panel)) !important;\n        color: var(--text) !important;", css)
        self.assertIn("column.cssClass = \"spec-scenario-cell\";", js)
        self.assertIn("delete column.editor;", js)
        self.assertIn("column.editable = false;", js)
        self.assertIn('table.on("cellClick", handleSpecCellClick);', js)
        self.assertIn("function toggleScenarioCheckbox(event, cell)", js)
        self.assertIn("function specGridOwnsKeyboardEvent(event)", js)
        self.assertIn('document.addEventListener("mouseup", handleSpecDocumentMouseup, true);', spec_js)
        self.assertNotIn('document.addEventListener("mouseup", endSelectionDrag, true);', spec_js)
        self.assertIn("function handleSpecDocumentMouseup(event) {\n    if (!selectionDragging) return;\n    endSelectionDrag(event);\n  }", spec_js)
        self.assertIn("function endSelectionDrag(event = null) {\n    if (!selectionDragging) return;", spec_js)
        self.assertIn("clearNativeSelection(event);\n  }\n\n  function startSelectionFromCell", spec_js)
        self.assertIn("function moveSelectionWithArrow(key, extend)", js)
        self.assertIn("function scrollSelectionPointIntoView(point, rows = displayedRows())", js)
        self.assertIn("function activeSingleCell()", js)
        self.assertIn("function isPrintableEditKey(event)", js)
        self.assertIn("function startEditingActiveCell(initialText)", js)
        self.assertIn("function restoreSelectionAfterCellEdit(cell)", js)
        self.assertIn("restoreSelectionAfterCellEdit(cell);", js)
        self.assertIn("if (!specToolVisible() || isEditableTarget(event.target)) return false;", js)
        self.assertIn("if (!specGridOwnsKeyboardEvent(event)) return false;", js)
        self.assertIn('["ArrowUp", "ArrowDown", "ArrowLeft", "ArrowRight"].includes(event.key) && !shortcut && !event.altKey', js)
        self.assertIn("moveSelectionWithArrow(event.key, event.shiftKey);", js)
        self.assertIn("const currentPoint = extend && pointInGrid(selection?.focus, rowIds, columns) ? selection.focus : selection?.active;", js)
        self.assertIn("const active = extend && pointInGrid(selection?.active, rowIds, columns) ? selection.active : next;", js)
        self.assertIn("selection = { anchor, focus: next, active };", js)
        self.assertIn("const origin = { rowId: bounds.rowIds[bounds.top], field: bounds.columns[bounds.left] };", js)
        self.assertIn("selection = { anchor: origin, focus, active: origin };", js)
        self.assertNotIn("selection.active = point;", js)
        self.assertIn("if (isPrintableEditKey(event)) {", js)
        self.assertIn("if (!startEditingActiveCell(event.key)) return false;", js)
        self.assertIn("if (!point || isScenarioField(point.field)) return false;", js)
        self.assertIn("input.setSelectionRange?.(initialText.length, initialText.length);", js)
        self.assertIn('cell?.getElement?.()?.scrollIntoView?.({ block: "nearest", inline: "nearest" });', js)
        self.assertIn('return `<input class="spec-checkbox-cell" type="checkbox" tabindex="-1" aria-label="${escapeHtml(cell.getField())}" ${checked ? "checked" : ""}>`;', js)
        self.assertNotIn('class="spec-checkbox-cell" type="checkbox" tabindex="-1" ${checked ? "checked" : ""} disabled', js)
        self.assertIn(".spec-grid .tabulator-row .tabulator-cell.spec-scenario-cell {\n        justify-content: center;", css)
        self.assertIn(".spec-checkbox-cell {\n        accent-color: var(--accent);\n        cursor: pointer;", css)
        checkbox_block = css[css.index(".spec-checkbox-cell {"):css.index(".spec-empty-state {")]
        self.assertNotIn("pointer-events: none;", checkbox_block)
        self.assertIn(".spec-grid .tabulator-row .tabulator-cell.spec-cell-selected", css)
        self.assertIn(".spec-grid .tabulator-row .tabulator-cell.spec-cell-active", css)
        self.assertNotIn(".spec-grid .tabulator-row .tabulator-cell.tabulator-range-selected", css)
        self.assertNotIn(".spec-context-menu button", css)

    def test_line_bar_table_search_uses_complete_table_client_filter(self) -> None:
        js = self.app_js_contract()
        css = self.app_css_contract()

        self.assertIn('import { loadTabulator } from "./shared/tabulator.js";', js)
        self.assertIn("const lineBarTool = createLineBarTool({\n        api,\n        el,\n        state,\n        echartsImpl: echarts,\n        escapeHtml,\n        isModelPredictionColumn,\n        copyTextToClipboard,", js)
        self.assertIn("saveToolPresentation,\n        showClipboardToast,\n        stableRequestKey,", js)
        self.assertIn("const TABLE_PAGE_SIZE = 10000;", js)
        self.assertIn("const TABLE_SEARCH_DEBOUNCE_MS = 250;", js)
        self.assertIn("stableRequestKey = (request) => JSON.stringify(request),", js)
        self.assertIn("stableRequestKey,", js)
        self.assertIn("copyTextToClipboard = () => Promise.resolve(false),", js)
        self.assertIn("showClipboardToast = () => {},", js)
        self.assertIn("let lineBarTable = null;", js)
        self.assertIn("let lineBarTableCopyRows = [];", js)
        self.assertIn("let lineBarTableCopyColumns = [];", js)
        self.assertIn("let lineBarTableCopyFooterRow = null;", js)
        self.assertIn("let tableRenderToken = 0;", js)
        self.assertIn("let completeTableCacheKey = \"\";", js)
        self.assertIn("let completeTableCacheData = null;", js)
        self.assertIn("function buildTableRequest()", js)
        self.assertIn("tableSearch: state.lineBarTableSearch || \"\",", js)
        self.assertIn("tablePage: state.tablePage || 1,", js)
        self.assertIn("tablePageSize: TABLE_PAGE_SIZE,", js)
        self.assertIn("function applyClientLineBarTableFilter(options = {})", js)
        self.assertIn("function filteredLineBarTableDataFromSource(sourceData, search)", js)
        self.assertIn("function buildClientLineBarTableSummary(rows, data)", js)
        self.assertIn("function lineBarTableRowMatchesSearch(row, search, xKind)", js)
        self.assertIn("function formatLineBarTableNumericSearchLabel(value)", js)
        self.assertIn("&& (!Number.isFinite(page) || page === 1);", js)
        self.assertIn("if (applyClientLineBarTableFilter()) return;", js)
        self.assertIn("if (!options.forceServer && applyClientLineBarTableFilter({ requestKey })) return tableCacheData;", js)
        self.assertIn("rememberCompleteLineBarTableSource(data, request);", js)
        self.assertIn('api("/api/line-bar/table", { method: "POST", body: JSON.stringify(request), clientTiming: true })', js)
        self.assertIn("scheduleLineBarTableRefresh();", js)
        self.assertIn("refreshLineBarTable({ force: true });", js)
        self.assertIn("const summaryResponses = Array.isArray(data.summary?.responses) ? data.summary.responses : [];", js)
        self.assertIn('content.innerHTML = `<div id="lineBarTableGrid" class="line-bar-table-grid"></div>${pager}`;', js)
        self.assertIn("loadTabulator().then((Tabulator) => {", js)
        self.assertIn('renderVertical: "virtual",', js)
        self.assertIn("rowHeight: 22,", js)
        self.assertIn("selectableRows: true,", js)
        self.assertIn('target.addEventListener("contextmenu", handleLineBarTableContextMenu);', js)
        self.assertIn("function handleLineBarTableContextMenu(event)", js)
        self.assertIn("function openLineBarTableContextMenu(event, actions = [])", js)
        self.assertIn("function copyLineBarTableContextValue(event)", js)
        self.assertIn("function copyVisibleLineBarView()", js)
        self.assertIn("function copyVisibleLineBarTable()", js)
        self.assertIn("function copyVisibleLineBarChart()", js)
        self.assertIn("function lineBarChartClipboardBlob(dataUrl)", js)
        self.assertIn("const blobPromise = lineBarChartClipboardBlob(dataUrl);", js)
        self.assertIn("navigator.clipboard.write([new window.ClipboardItem({ \"image/png\": blobPromise })]);", js)
        self.assertIn("function lineBarRowsToCsv(rows, options = {})", js)
        self.assertIn("Copy cell to clipboard", js)
        self.assertIn("Copy selected row", js)
        self.assertIn("Copy selected rows", js)
        self.assertIn("Copy table to clipboard", js)
        self.assertIn("Clear selection", js)
        line_bar_context_source = self.js_function_source(js, "handleLineBarTableContextMenu")
        self.assertNotIn("Copy selected column", line_bar_context_source)
        self.assertNotIn("Copy selected columns", line_bar_context_source)
        self.assertIn('bottomCalc: () => "Total",', js)
        self.assertIn("bottomCalc: () => formatNumber(summaryVolume),", js)
        self.assertIn("bottomCalc: () => formatResponseValue(summaryResponses[responseIndex]),", js)
        self.assertIn("const pager = `<div class=\"table-pagination\">", js)
        self.assertNotIn("<tbody>", self.js_function_source(js, "renderLineBarTableContents"))
        self.assertNotIn("<tfoot>", self.js_function_source(js, "renderLineBarTableContents"))
        self.assertNotIn("lineBarTableSearch", self.js_function_source(js, "buildChartRequest"))
        self.assertIn('id="lineBarTableSearch"', js)
        self.assertIn('id="lineBarTableSearchClear"', js)
        self.assertIn("No matching rows", js)
        self.assertIn(".line-bar-table-search-row {", css)
        self.assertIn(".line-bar-table-content {", css)
        self.assertIn(".line-bar-table-grid {", css)
        self.assertIn(".line-bar-table-context-menu {", css)
        self.assertIn(".line-bar-table-context-menu-item:hover,\n      .line-bar-table-context-menu-item:focus-visible {", css)
        self.assertIn(".line-bar-table-context-menu-divider {", css)
        self.assertIn(".line-bar-table-grid.tabulator .tabulator-row.tabulator-selected:not(.tabulator-calcs),", css)
        self.assertIn(".line-bar-table-grid.tabulator .tabulator-row.tabulator-calcs", css)
        self.assertIn(".line-bar-table-grid.tabulator .tabulator-header .tabulator-frozen.tabulator-frozen-left,\n      .line-bar-table-grid.tabulator .tabulator-row .tabulator-cell.tabulator-frozen.tabulator-frozen-left {\n        border-right: 1px solid color-mix(in srgb, var(--line) 80%, transparent);", css)
        self.assertIn(".line-bar-table-grid.tabulator .tabulator-tableholder .tabulator-table .tabulator-row.tabulator-calcs.tabulator-calcs-bottom {\n        border-top: 1px solid var(--line);", css)
        self.assertIn(".line-bar-table-grid.tabulator .tabulator-footer {\n        background: var(--panel);", css)
        self.assertIn(".line-bar-table-grid.tabulator .tabulator-footer .tabulator-calcs-holder {\n        background: var(--panel) !important;", css)
        self.assertIn(".line-bar-table-grid.tabulator .tabulator-footer .tabulator-calcs-holder .tabulator-row {\n        background: var(--panel) !important;", css)
        self.assertIn(".line-bar-table-grid.tabulator .tabulator-footer .tabulator-calcs-holder .tabulator-row .tabulator-cell {\n        border-right-color: color-mix(in srgb, var(--line) 80%, transparent);", css)
        self.assertIn(".line-bar-table-grid.tabulator .tabulator-footer .tabulator-calcs-holder .tabulator-row .tabulator-cell.tabulator-frozen.tabulator-frozen-left {\n        border-right: 1px solid color-mix(in srgb, var(--line) 80%, transparent);", css)
        self.assertIn(".line-bar-table-state {", css)
        self.assertIn(".line-bar-table-grid.tabulator .tabulator-placeholder {", css)

    def test_line_bar_sort_and_overlarge_chart_contract(self) -> None:
        js = self.app_js_contract()

        self.assertIn("function applyClientLineBarSort(options = {})", js)
        self.assertIn('if (!data || data.x_group_kind !== "categorical") return false;', js)
        self.assertIn("function applyClientLineBarTableSort()", js)
        self.assertIn('&& pageCount === 1', js)
        self.assertIn("&& rows.length === matchCount", js)
        self.assertIn("&& normaliseLineBarTableSearch(table.search) === normaliseLineBarTableSearch(state.lineBarTableSearch)", js)
        self.assertIn('&& (state.sort !== "shap" || shapMedians !== null);', js)
        self.assertIn('if (data?.groups_truncated && !(data.rows || []).length) {', js)
        self.assertIn("chart.clear();", js)
        self.assertIn("data.rows = [...(data.rows || [])].sort(compareLineBarRowsForSort(state.sort, shapMedianMap(data)));", js)
        self.assertIn("data.rows = [...(data.rows || [])].sort(compareLineBarRowsForSort(state.sort, shapMedians || new Map()));", js)
        self.assertIn("orderPartialDependenceRowsForChart(data);", js)
        self.assertIn("cache.requestKey = stableRequestKey(request);", js)
        self.assertIn('if (group.dataset.control === "sort") {', js)
        self.assertIn("if (!applyClientLineBarSort({ render: false })) lineBarChartDirty = true;", js)
        self.assertIn("if (applyClientLineBarTableSort()) return;", js)
        self.assertIn("refreshLineBarTable({ force: true });", js)
        self.assertIn("if (applyClientLineBarSort()) return;", js)

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
        self.assertIn('state.schema?.source_kind === "parquet_folder"', js)
        self.assertIn('path = `${path} (${fileCount.toLocaleString()} ${fileCount === 1 ? "file" : "files"})`;', js)
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
        self.assertIn(".saved-filter-theme.saved-filter-theme--selected {\n        color: var(--danger);", css)
        self.assertIn("cursor: pointer;", css)
        self.assertIn(".saved-filter-theme-icon {\n        color: inherit;\n        flex: 0 0 10px;\n        height: 10px;", css)
        self.assertIn(".saved-filter-theme-icon::before,\n      .saved-filter-theme-icon::after {\n        background: currentColor;", css)
        self.assertIn("height: 1.35px;\n        position: absolute;\n        top: 4.5px;", css)
        self.assertIn(".saved-filter-theme-icon::before {\n        left: 0.5px;\n        transform: rotate(45deg);", css)
        self.assertIn(".saved-filter-theme-icon::after {\n        right: 0.5px;\n        transform: rotate(-45deg);", css)
        self.assertIn('.saved-filter-theme[aria-expanded="false"] .saved-filter-theme-icon {\n        transform: rotate(-90deg);', css)
        self.assertIn(".saved-filter-theme-label {\n        min-width: 0;\n        overflow: hidden;\n        text-overflow: ellipsis;\n        white-space: nowrap;", css)
        self.assertIn("text-transform: uppercase;", css)
        self.assertIn(".saved-filter-expression {\n        min-width: 0;\n        overflow: hidden;\n        text-align: right;\n        text-overflow: ellipsis;\n        white-space: nowrap;\n        color: var(--muted);\n        font-size: 10px;", css)
        self.assertIn("collapsedSavedFilterThemes: new Set()", js)
        self.assertIn("savedFilterThemesInitialised: false", js)
        self.assertIn("function savedFilterSpecSignature(filters = state.schema?.filters || [])", js)
        self.assertIn("function savedFilterSelectionSnapshot()", js)
        self.assertIn("function syncSavedFilterThemeSelectionState()", js)
        self.assertIn('list.querySelectorAll(".saved-filter-theme").forEach((heading) => {', js)
        self.assertIn('heading.classList.toggle("saved-filter-theme--selected", selectedThemes.has(heading.dataset.filterTheme || "General"));', js)
        self.assertIn("function restoreSavedFilterSelection(selectedKeys)", js)
        self.assertIn("syncSavedFilterThemeSelectionState();\n          return;", js)
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
        self.assertIn("if (button.dataset.filterTheme === theme) button.hidden = collapsed;", js)
        self.assertIn('button.addEventListener("click", () => {', js)
        self.assertIn('button.setAttribute("aria-selected", String(!selected));', js)
        self.assertIn('button.classList.toggle("active", !selected);', js)
        self.assertIn("syncSavedFilterThemeSelectionState();\n            applySavedFilters();", js)
        self.assertIn("list.append(button);\n        }\n        syncSavedFilterThemeSelectionState();", js)
        self.assertIn("function setFilterSelectionMode(mode, options = {})", js)
        self.assertIn('document.body.classList.toggle("saved-filter-single-mode", nextMode === "single");', js)
        self.assertIn('const group = document.querySelector(\'.segmented[data-control="filterSelectionMode"]\');', js)
        self.assertIn("const keep = selected[0];", js)
        self.assertIn('querySelectorAll(\'.saved-filter-option[aria-selected="true"]\')', js)
        self.assertIn("syncSavedFilterThemeSelectionState();\n        if (options.apply !== false) {\n          applySavedFilters();\n        }", js)
        self.assertIn('Array.from(el("savedFilterSelect").querySelectorAll(".saved-filter-option")).forEach((button) => {\n          button.setAttribute("aria-selected", "false");\n          button.classList.remove("active");\n        });\n        syncSavedFilterThemeSelectionState();', js)
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

    def test_sidebar_filter_accordion_header_and_row_meta_contract(self) -> None:
        _, html_body = self.assert_no_store("/")
        css = self.app_css_contract()
        html = html_body.decode("utf-8")
        js = self.app_js_contract()

        self.assertIn('<section class="section sidebar-filter-section sidebar-accordion-section sidebar-section-closed" data-sidebar-section="filter">', html)
        self.assertNotIn('aria-label="Resize KPI and filter controls"', html)
        self.assertIn('<div class="filter-header">', html)
        self.assertIn('id="filterCollapseBtn" class="filter-collapse-button sidebar-section-header" type="button" aria-label="Expand FILTER" aria-expanded="false" title="Expand FILTER"', html)
        self.assertIn('<span class="sidebar-section-title">FILTER</span>', html)
        filter_start = html.index('data-sidebar-section="filter"')
        self.assertLess(html.index('<span class="sidebar-section-title">FILTER</span>', filter_start), html.index('id="filterRowMeta"', filter_start))
        self.assertLess(html.index('id="filterRowMeta"', filter_start), html.index('<div class="sidebar-section-body">', filter_start))
        self.assertIn('id="filterRowMeta" class="filter-row-meta"', html)
        self.assertIn('id="filterRowClearBtn" class="filter-row-clear" type="button" title="Clear filter" aria-label="Clear filter" hidden></button>', html)
        self.assertIn('id="filterRowMetaText"', html)
        self.assertIn('id="filterRowMetaText" class="filter-badge-text"', html)
        self.assertLess(html.index('id="filterRowClearBtn"', filter_start), html.index('id="filterRowMetaText"', filter_start))
        self.assertLess(html.index('id="filterRowMetaText"', filter_start), html.index('<div class="sidebar-section-body">', filter_start))
        self.assertNotIn("filterSidebarClearBtn", html)
        self.assertIn('id="collapsedSidebarVersion" class="collapsed-sidebar-version" aria-label="Lucidum version" hidden', html)
        self.assertNotIn("FILTER ACTIVE", html)
        self.assertIn(".sidebar-section-closed .sidebar-section-body {\n        display: none;", css)
        self.assertIn(".sidebar-section-closed .filter-collapse-icon", css)
        self.assertIn(".sidebar-accordion-section.sidebar-section-open {\n        flex: 1 1 auto;", css)
        self.assertIn(".filter-header {\n        display: flex;\n        align-items: center;\n        gap: 5px;", css)
        self.assertIn(".sidebar-section-header {\n        width: 100%;", css)
        self.assertIn(".filter-collapse-button.sidebar-section-header {\n        flex: 1 1 auto;\n        width: auto;", css)
        self.assertIn(".filter-header:hover .filter-collapse-button.sidebar-section-header {\n        color: var(--accent);", css)
        self.assertIn(".sidebar-section-header:hover .favourites-selected-meta,\n      .sidebar-section-header:hover .kpi-selected-meta,\n      .sidebar-section-header:hover .gbm-model-selected-meta,\n      .sidebar-section-header:hover .glm-model-selected-meta,\n      .filter-header:hover .filter-row-meta {\n        color: var(--accent);", css)
        self.assertIn(".filter-collapse-icon {\n        width: 14px;", css)
        self.assertIn("flex: 0 0 14px;", css)
        self.assertIn(".filter-collapse-icon::before,\n      .filter-collapse-icon::after {\n        content: \"\";", css)
        self.assertIn("width: 8px;\n        height: 1.5px;", css)
        self.assertIn("background: currentColor;", css)
        self.assertIn(".filter-collapse-icon::before {\n        left: 0.75px;\n        transform: rotate(45deg);", css)
        self.assertIn(".filter-collapse-icon::after {\n        right: 0.75px;\n        transform: rotate(-45deg);", css)
        self.assertIn(".sidebar-section-closed .filter-collapse-icon {\n        transform: rotate(-90deg);", css)
        self.assertIn(".favourites-selected-meta,\n      .kpi-selected-meta,\n      .gbm-model-selected-meta,\n      .glm-model-selected-meta,\n      .filter-row-meta {\n        min-width: 0;\n        overflow: hidden;", css)
        self.assertIn("margin-left: auto;\n        flex: 1 1 auto;", css)
        self.assertIn(".filter-row-meta {\n        background: transparent;\n        border: 0;\n        border-radius: 4px;", css)
        self.assertIn("display: inline-flex;\n        align-items: center;\n        justify-content: flex-end;\n        gap: 4px;", css)
        self.assertIn("padding: 0;\n        flex: 0 1 auto;", css)
        self.assertIn(".filter-row-meta.filter-row-meta--applied {\n        background: var(--filter-applied-bg);\n        border: 1px solid var(--filter-applied-border);\n        color: var(--filter-applied-text);\n        padding: 1px 4px;", css)
        self.assertIn(".filter-row-clear {\n        appearance: none;\n        border: 0;\n        background: transparent;\n        color: currentColor;", css)
        self.assertIn("font-weight: 800;\n        line-height: 1;", css)
        self.assertIn('.filter-row-clear::before {\n        content: "x";', css)
        self.assertIn(".filter-badge-text {\n        min-width: 0;\n        overflow: hidden;\n        text-overflow: ellipsis;", css)
        self.assertIn(".collapsed-sidebar-version {\n        display: none;\n        margin-top: auto;\n        color: var(--muted);", css)
        self.assertIn("body.sidebar-collapsed .collapsed-sidebar-version:not([hidden]) {\n        display: block;", css)
        self.assertNotIn(".collapsed-filter-indicator", css)
        self.assertNotIn(".filter-sidebar-clear", css)
        self.assertIn('openSidebarSection: "favourites"', js)
        self.assertIn("function formatRowMeta(rowCount, filteredRowCount = rowCount)", js)
        self.assertIn("? `${total.toLocaleString()} rows`", js)
        self.assertIn(": `${shown.toLocaleString()} / ${total.toLocaleString()} rows`", js)
        self.assertIn("function filterIsApplied()", js)
        self.assertIn('return Boolean(String(state.activeFilter || "").trim());', js)
        self.assertIn("function syncActiveFilterIndicator()", js)
        self.assertIn('el("filterRowClearBtn").hidden = !applied;', js)
        self.assertIn('el("datasetViewerFilterClearBtn").hidden = !applied;', js)
        self.assertIn('el("profileFilterClearBtn").hidden = !applied;', js)
        self.assertIn('el("lineBarFilterClearBtn").hidden = !applied;', js)
        self.assertIn('el("histogramFilterClearBtn").hidden = !applied;', js)
        self.assertIn('el("mapControlFilterClearBtn").hidden = !applied;', js)
        self.assertIn('el("filterRowMeta").classList.toggle("filter-row-meta--applied", applied);', js)
        self.assertIn('el("filterRowMetaText").textContent = meta || "";', js)
        self.assertIn('el("datasetViewerFilter").classList.toggle("dataset-viewer-filter--applied", applied);', js)
        self.assertIn('el("profileFilter").classList.toggle("profile-filter--applied", applied);', js)
        self.assertIn('el("lineBarFilter").classList.toggle("line-bar-filter--applied", applied);', js)
        self.assertIn('el("histogramFilter").classList.toggle("histogram-filter--applied", applied);', js)
        self.assertIn('el("mapControlFilter").classList.toggle("map-filter--applied", applied);', js)
        self.assertNotIn("collapsedFilterIndicator", js)
        self.assertIn("const SIDEBAR_ACCORDION_SECTIONS = {", js)
        self.assertIn("function toggleSidebarSection(section)", js)
        self.assertIn("function syncSidebarAccordion()", js)
        self.assertIn('button.setAttribute("aria-expanded", String(open));', js)
        self.assertIn('el("filterCollapseBtn").addEventListener("click", () => toggleSidebarSection("filter"));', js)
        self.assertIn('api("/api/filter/row-count"', js)
        self.assertIn("filterRowCountRequestSeq: 0", js)
        self.assertIn("filterRowCountMeta: null", js)
        self.assertIn("async function refreshFilterRowCountMeta()", js)
        self.assertIn('setFilterRowMetaText("updating...");', js)
        self.assertIn("resetFilterRowMetaToSchema();", js)
        self.assertIn("setFilterRowMeta(data.row_count, data.filtered_row_count);", js)
        self.assertIn("state.filterRowCountMeta = {", js)
        self.assertNotIn("setFilterRowMeta(state.schema.row_count);", js)
        static_app = Path(__file__).parents[1] / "src" / "py_lucidum" / "static" / "app"
        for module_name in ("column-profile-tool.js", "line-bar-tool.js", "histogram-tool.js", "uk-map-tool.js"):
            with self.subTest(module=module_name):
                module_js = (static_app / module_name).read_text(encoding="utf-8")
                self.assertNotIn("setFilterRowMeta", module_js)
        self.assertNotIn("filterCollapsed", js)
        self.assertNotIn("function setFilterCollapsed", js)
        self.assertNotIn("filter-collapsed", html + css)

    def test_sidebar_metric_section_sits_below_filter_accordion(self) -> None:
        _, html_body = self.assert_no_store("/")
        css = self.app_css_contract()
        html = html_body.decode("utf-8")
        js = self.app_js_contract()

        self.assertIn('<section class="section sidebar-metric-section">', html)
        self.assertNotIn('<section class="section sidebar-metric-section hidden">', html)
        self.assertIn('<section class="section sidebar-favourites-section sidebar-accordion-section sidebar-section-closed" data-sidebar-section="favourites">', html)
        self.assertIn('<section class="section sidebar-kpi-section sidebar-accordion-section sidebar-section-closed" data-sidebar-section="kpi">', html)
        filter_start = html.index('data-sidebar-section="filter"')
        metric_start = html.index('class="section sidebar-metric-section"')
        self.assertLess(html.index('data-sidebar-section="favourites"'), html.index('data-sidebar-section="kpi"'))
        self.assertLess(html.index('data-sidebar-section="kpi"'), html.index('data-sidebar-section="gbm"'))
        self.assertLess(html.index('data-sidebar-section="gbm"'), html.index('data-sidebar-section="glm"'))
        self.assertLess(html.index('data-sidebar-section="glm"'), filter_start)
        self.assertLess(filter_start, metric_start)
        self.assertLess(metric_start, html.index('id="sidebarVersion"'))
        self.assertLess(metric_start, html.index('id="actualNumerator"'))
        self.assertLess(html.index('id="actualNumerator"'), html.index('id="denominator"'))
        self.assertLess(html.index('id="favouritesSelect"'), html.index('id="kpiSelect"'))
        self.assertIn('id="favouritesCollapseBtn" class="favourites-header sidebar-section-header" type="button" aria-label="Expand FAVOURITES" aria-expanded="false" title="Expand FAVOURITES"', html)
        self.assertIn('id="favouritesSelectedMeta" class="favourites-selected-meta"', html)
        self.assertIn('id="favouritesSelect" class="feature-list favourites-list" role="listbox"', html)
        self.assertIn('id="sidebarFavouriteAddBtn"', html)
        self.assertIn('id="sidebarFavouriteMenuBtn"', html)
        self.assertIn('id="kpiCollapseBtn" class="kpi-header sidebar-section-header" type="button" aria-label="Expand KPIs" aria-expanded="false" title="Expand KPIs"', html)
        self.assertIn('id="kpiSelectedMeta" class="kpi-selected-meta"', html)
        self.assertIn('id="kpiSelect" class="feature-list kpi-list" role="listbox"', html)
        self.assertIn('id="gbmModelCollapseBtn" class="gbm-model-header sidebar-section-header" type="button" aria-label="Expand GBMs" aria-expanded="false" title="Expand GBMs"', html)
        self.assertIn('id="glmModelCollapseBtn" class="glm-model-header sidebar-section-header" type="button" aria-label="Expand GLMs" aria-expanded="false" title="Expand GLMs"', html)
        self.assertNotIn('id="sidebarKpiResizer"', html)
        self.assertNotIn('id="sidebarGbmResizer"', html)
        self.assertNotIn('id="sidebarGlmResizer"', html)
        self.assertNotIn('id="sidebarFilterResizer"', html)
        self.assertIn('id="sidebarResizer" class="sidebar-resizer" role="separator"', html)
        self.assertIn(".sidebar-metric-section {\n        flex: 0 0 auto;", css)
        self.assertIn(".sidebar-accordion-section {\n        display: flex;\n        flex-direction: column;\n        flex: 0 0 auto;", css)
        self.assertIn(".sidebar-accordion-section.sidebar-section-open {\n        flex: 1 1 auto;", css)
        self.assertIn(".sidebar-section-body {\n        display: flex;\n        flex: 1 1 auto;", css)
        self.assertIn(".sidebar-section-closed .sidebar-section-body {\n        display: none;", css)
        self.assertIn(".sidebar-section-header {\n        width: 100%;", css)
        self.assertIn(".sidebar-section-title {\n        margin: 0;\n        font-size: 12px;", css)
        self.assertIn("font-weight: 700;\n        flex: 0 0 auto;", css)
        self.assertIn(".favourites-list .feature,\n      .kpi-list .feature,\n      .gbm-model-list .feature,\n      .glm-model-list .feature {\n        display: grid;\n        grid-template-columns: fit-content(52%) minmax(96px, 1fr);", css)
        self.assertIn(".gbm-model-list .feature,\n      .glm-model-list .feature {\n        display: grid;\n        grid-template-columns: fit-content(52%) minmax(96px, 1fr);", css)
        self.assertIn(".kpi-list .kpi-option.active {\n        background: color-mix(in srgb, #f59e0b 22%, var(--panel));", css)
        self.assertIn(".favourites-list .saved-favourite-option.active {\n        background: color-mix(in srgb, var(--accent) 20%, var(--panel));", css)
        self.assertIn(".favourites-list .saved-favourite-option {\n        column-gap: 8px;\n        grid-template-columns: minmax(0, 1fr) max-content;", css)
        self.assertIn(".gbm-model-list .gbm-model-option.active,\n      .glm-model-list .glm-model-option.active {\n        background: color-mix(in srgb, var(--accent) 20%, var(--panel));", css)
        self.assertIn(".favourite-detail,\n      .kpi-detail,\n      .gbm-model-detail,\n      .glm-model-detail {\n        min-width: 0;\n        overflow: hidden;\n        text-align: right;", css)
        self.assertNotIn("sidebar-kpi-height", css + js)
        self.assertNotIn("sidebar-gbm-height", css + js)
        self.assertNotIn("sidebar-glm-height", css + js)
        self.assertNotIn("sidebar-filter-height", css + js)
        self.assertNotIn(".sidebar-gbm-resizer", css)
        self.assertNotIn(".sidebar-glm-resizer", css)
        self.assertNotIn(".sidebar-filter-resizer", css)
        self.assertIn('openSidebarSection: "favourites"', js)
        self.assertIn("collapsedKpiGroups: new Set()", js)
        self.assertIn("kpiGroupsInitialised: false", js)
        self.assertIn("collapsedGlmModelGroups: new Set()", js)
        self.assertIn("glmModelGroupsInitialised: false", js)
        self.assertIn("collapsedGbmModelGroups: new Set()", js)
        self.assertIn("gbmModelGroupsInitialised: false", js)
        self.assertIn("activeKpiFormat: null", js)
        self.assertIn("function availableKpis()", js)
        self.assertIn("function applyInitialKpiDefault()", js)
        self.assertIn("function renderFavourites()", js)
        self.assertIn("function renderKpis()", js)
        self.assertNotIn("function favouriteRows()", js)
        favourites_start = js.index("function renderFavourites()")
        favourites_end = js.index("function renderKpis()", favourites_start)
        favourites_js = js[favourites_start:favourites_end]
        self.assertNotIn("kpi-option", favourites_js)
        self.assertNotIn("favourite-theme", favourites_js)
        self.assertIn("const favouritesLoading = !lineBarFavouritesLoaded && !favouriteLoadError;", favourites_js)
        self.assertIn('list.toggleAttribute("aria-busy", favouritesLoading);', favourites_js)
        self.assertIn("if (!favouriteLoadError && lineBarFavouritesLoaded) {", favourites_js)
        self.assertIn('button.className = `feature favourite-option saved-favourite-option${active ? " active" : ""}${invalid ? " favourite-option-invalid" : ""}`;', favourites_js)
        self.assertIn('heading.className = "saved-filter-theme kpi-theme";', js)
        self.assertIn('button.className = `feature kpi-option${active ? " active" : ""}`;', js)
        self.assertIn("function selectKpi(kpi)", js)
        self.assertIn("function toggleSidebarSection(section)", js)
        self.assertIn("function setOpenSidebarSection(section)", js)
        self.assertIn("function setModelSidebarPanelVisibility(panelId, enabled)", js)
        self.assertIn('setModelSidebarPanelVisibility("gbmSidebarPanel", gbmEnabled);', js)
        self.assertIn('setModelSidebarPanelVisibility("glmSidebarPanel", glmEnabled);', js)
        self.assertIn('state.openSidebarSection === "gbm" && !gbmEnabled', js)
        self.assertIn('state.openSidebarSection === "glm" && !glmEnabled', js)
        self.assertIn("function syncSidebarAccordion()", js)
        self.assertIn('document.querySelector(".sidebar-metric-section")?.classList.toggle("hidden", state.openSidebarSection !== null);', js)
        self.assertIn('panel?.classList.toggle("sidebar-section-open", open);', js)
        self.assertIn('panel?.classList.toggle("sidebar-section-closed", !open);', js)
        self.assertIn('el("favouritesCollapseBtn").addEventListener("click", () => toggleSidebarSection("favourites"));', js)
        self.assertIn('el("kpiCollapseBtn").addEventListener("click", () => toggleSidebarSection("kpi"));', js)
        self.assertIn('el("glmModelCollapseBtn").addEventListener("click", () => toggleSidebarSection("glm"));', js)
        self.assertIn('el("gbmModelCollapseBtn").addEventListener("click", () => toggleSidebarSection("gbm"));', js)
        self.assertIn("syncKpiSelectionFromMetrics();", js)
        self.assertIn("const previousCollapsedSavedFilterThemes = new Set(state.collapsedSavedFilterThemes);", js)
        self.assertIn("const previousSavedFilterThemesInitialised = state.savedFilterThemesInitialised;", js)
        self.assertIn("const previousCollapsedKpiGroups = new Set(state.collapsedKpiGroups);", js)
        self.assertIn("const previousKpiGroupsInitialised = state.kpiGroupsInitialised;", js)
        self.assertIn("state.collapsedSavedFilterThemes = previousCollapsedSavedFilterThemes;", js)
        self.assertIn("state.savedFilterThemesInitialised = previousSavedFilterThemesInitialised;", js)
        self.assertIn("state.collapsedKpiGroups = previousCollapsedKpiGroups;", js)
        self.assertIn("state.kpiGroupsInitialised = previousKpiGroupsInitialised;", js)
        self.assertNotIn("kpiCollapsed", js)
        self.assertNotIn("glmModelCollapsed", js)
        self.assertNotIn("gbmModelCollapsed", js)
        self.assertNotIn("filterCollapsed", js)
        self.assertNotIn("py_lucidum_sidebar_kpi_height", js)
        self.assertNotIn("py_lucidum_sidebar_gbm_height", js)
        self.assertNotIn("py_lucidum_sidebar_glm_height", js)
        self.assertNotIn("py_lucidum_sidebar_filter_height", js)
        self.assertNotIn("function restoreSidebarPanelHeights()", js)
        self.assertNotIn("function resizeSidebarBoundary(key, delta, startLayout)", js)

    def test_actual_selector_groups_dataset_predictions_and_shap_values(self) -> None:
        js = self.app_js_contract()

        self.assertIn("function fillActualMetricSelect(select)", js)
        self.assertIn('"Dataset features"', js)
        self.assertIn('"Model predictions"', js)
        self.assertIn('"SHAP values"', js)
        self.assertIn('"No trained models"', js)
        self.assertIn('"No predictions for selected model"', js)
        self.assertIn('"No SHAP values for selected model"', js)
        self.assertIn("for (const column of sortedMetricColumns(columns))", js)
        self.assertIn("option.dataset.sourceId = column.source_id || sourceId;", js)
        self.assertIn("option.dataset.metricKind = kind;", js)
        self.assertIn('column?.source_role === "gbm_shap_value"', js)
        self.assertIn('String(column?.name || "").startsWith("SHAP__")', js)
        self.assertIn("function syncActualSourceFromSelection()", js)

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
        self.assertIn("DuckDB: --, JSON: --, Dataset render: --, Total: --", html)
        self.assertIn('<div class="filter-controls-row">', html)
        self.assertIn('<div class="filter-controls-primary">', html)
        self.assertLess(html.index('data-control="filterSelectionMode"'), html.index('data-control="filterOperator"'))
        self.assertLess(html.index('data-control="filterSelectionMode"'), html.index('id="savedFilterSelect"'))
        self.assertLess(html.index('data-control="filterOperator"'), html.index('id="savedFilterSelect"'))
        self.assertIn('class="segmented filter-operator" data-control="filterOperator"', html)
        self.assertIn('class="segmented filter-selection-mode" data-control="filterSelectionMode"', html)
        self.assertIn('<button data-value="and" class="active" type="button">All</button>', html)
        self.assertIn('<button data-value="or" type="button">Any</button>', html)
        self.assertIn('<button data-value="nand" type="button">Not all</button>', html)
        self.assertIn('<button data-value="nor" type="button">None</button>', html)
        self.assertIn('<button data-value="single" type="button">Single</button>', html)
        self.assertIn('<button data-value="grouped" class="active" type="button">Grouped</button>', html)
        self.assertIn('<button data-value="multi" type="button">Multi</button>', html)
        self.assertLess(html.index('data-value="single"'), html.index('data-value="grouped"'))
        self.assertLess(html.index('data-value="grouped"'), html.index('data-value="multi"'))
        self.assertNotIn('id="filterSidebarClearBtn"', html)
        self.assertNotIn("filter-input-row", html)
        self.assertIn("grid-template-rows: auto minmax(0, 1fr) auto;", css)
        self.assertIn(".layout-toggle-group {\n        display: flex;\n        align-items: center;\n        gap: 0;", css)
        self.assertIn(".footer-toggle {\n        width: 28px;\n        height: 24px;\n        min-height: 24px;", css)
        self.assertIn(".footer-toggle-icon {\n        width: 18px;\n        height: 20px;\n        border: 1.5px solid currentColor;", css)
        self.assertIn("bottom: 3.5px;\n        height: 1.5px;", css)
        self.assertIn("body.filter-footer-collapsed .footer-toggle-icon::before {\n        background: transparent;", css)
        self.assertIn(".filter-footer {\n        display: grid;\n        grid-template-columns: minmax(0, 1fr) max-content;", css)
        self.assertIn(".footer-filter-controls {\n        display: grid;\n        grid-template-columns: 28px 28px minmax(0, 1fr);", css)
        self.assertIn(".action-timing-monitor {\n        min-width: 0;\n        overflow: hidden;\n        color: var(--muted);\n        font-size: 10px;", css)
        self.assertIn("justify-self: end;\n        line-height: 1.1;\n        max-width: min(520px, 45vw);", css)
        self.assertIn("text-align: right;", css)
        self.assertIn("body.filter-footer-collapsed .filter-footer {\n        display: none;", css)
        self.assertIn(".filter-controls-row {\n        display: flex;\n        flex-direction: column;\n        gap: 6px;", css)
        self.assertIn(".filter-controls-primary {\n        display: flex;\n        align-items: center;\n        gap: 6px;\n        min-width: 0;", css)
        self.assertIn("body.saved-filter-single-mode .filter-operator,\n      body.saved-filter-grouped-mode .filter-operator {\n        display: none;", css)
        self.assertIn(".filter-operator {\n        align-self: flex-start;", css)
        self.assertIn("filterFooterCollapsed: true", js)
        self.assertIn('filterSelectionMode: "grouped"', js)
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
        self.assertIn("syncDuckDbTimingFromData(\"histogram\", data);", js)
        self.assertIn("syncClientTimingFromData(\"column_profile\", data);", js)
        self.assertIn("syncClientTimingFromData(\"line_bar\", data);", js)
        self.assertIn("syncClientTimingFromData(\"histogram\", data);", js)
        self.assertIn("syncDuckDbTimingFromData(\"uk_map\", data);", js)
        self.assertIn("syncClientTimingFromData(\"uk_map\", data);", js)
        self.assertIn('api("/api/column-profile/summary", { method: "POST", body: JSON.stringify(request), clientTiming: true })', js)
        self.assertIn('api("/api/column-profile/detail", { method: "POST", body: JSON.stringify(request), clientTiming: true })', js)
        self.assertIn('api("/api/chart", { method: "POST", body: JSON.stringify(request), clientTiming: true })', js)
        self.assertIn('api("/api/histogram/chart", { method: "POST", body: JSON.stringify(request), clientTiming: true })', js)
        self.assertIn('api("/api/uk-map/summary", { method: "POST", body: JSON.stringify(request), clientTiming: true })', js)
        self.assertIn("function syncFilterFooterToggleButton()", js)
        self.assertIn('el("filterFooterToggleBtn").addEventListener("click", () => setFilterFooterVisible(state.filterFooterCollapsed));', js)
        self.assertIn('el("filterRowClearBtn").addEventListener("click", clearFilter);', js)
        self.assertIn('el("datasetViewerFilterClearBtn").addEventListener("click", clearFilter);', js)
        self.assertIn('el("profileFilterClearBtn").addEventListener("click", clearFilter);', js)
        self.assertIn('el("lineBarFilterClearBtn").addEventListener("click", clearFilter);', js)
        self.assertIn('el("histogramFilterClearBtn").addEventListener("click", clearFilter);', js)
        self.assertIn('el("mapControlFilterClearBtn").addEventListener("click", clearFilter);', js)

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
        self.assertIn(".app input:not([type]):focus,\n      .app input:not([type]):focus-visible,", css)
        self.assertIn(".app input[type=\"search\"]:focus,\n      .app input[type=\"search\"]:focus-visible,", css)
        self.assertIn(".app textarea:focus,\n      .app textarea:focus-visible {\n        border-color: var(--line) !important;\n        box-shadow: none !important;\n        outline: none !important;", css)
        self.assertIn("function clearSearchInput(inputId, render)", js)
        self.assertIn('el("expectedSearchClear").addEventListener("click", () => clearSearchInput("expectedSearch", renderExpectedNumerators));', js)
        self.assertIn('el("featureSearchClear").addEventListener("click", () => clearSearchInput("featureSearch", renderFeatures));', js)
        self.assertIn("input.focus();", js)

    def test_line_bar_feature_importance_sort_control_contract(self) -> None:
        _, html_body = self.assert_no_store("/")
        html = html_body.decode("utf-8")
        js = self.app_js_contract()
        css = self.app_css_contract()

        self.assertIn('<button data-value="importance" class="hidden">Imp</button>', html)
        feature_sort_start = html.index('data-control="featureSort"')
        importance_index = html.index('data-value="importance"', feature_sort_start)
        original_index = html.index('<button data-value="original">Original</button>', feature_sort_start)
        alpha_index = html.index('<button data-value="alpha" class="active">A-Z</button>', feature_sort_start)
        self.assertLess(importance_index, original_index)
        self.assertLess(original_index, alpha_index)
        self.assertIn('/api/line-bar/feature-importance', js)
        self.assertIn('function renderFeatureImportanceRows(query, list)', js)
        self.assertIn('function featureImportanceDetail(row, metric)', js)
        self.assertIn('addFeatureListHeader(list, importanceGroupLabel(label, model));', js)
        self.assertIn('addFeatureListHeader(list, "Not used");', js)
        self.assertIn('return value ? `${prefix} · ${value}` : prefix;', js)
        self.assertIn('state.source = "dataset";', js)
        self.assertIn('No active GBM or GLM importances are available.', js)
        self.assertIn("--line-bar-pinned-bg:", css)
        self.assertIn("--line-bar-pinned-hover-bg:", css)
        self.assertIn("--line-bar-pinned-kind:", css)
        self.assertIn("#featureList.line-bar-split-list", css)
        self.assertIn("#expectedList.line-bar-split-list", css)
        self.assertIn("#expectedList .feature:disabled {", css)
        self.assertIn(".line-bar-pinned-region {", css)
        self.assertIn(".line-bar-scroll-region {", css)
        self.assertIn(".feature.line-bar-special-row {", css)
        self.assertIn(".metric-value--first-expected {", css)
        self.assertIn("color: #d13f3f;", css)
        self.assertIn(".metric-value--second-expected {", css)
        self.assertIn('.feature-list-section-header {', css)
        self.assertIn('.feature-list-message {', css)
        self.assertIn('.line-bar-importance-row .kind {', css)

    def test_date_x_axis_labels_are_fit_based(self) -> None:
        js = self.app_js_contract()

        self.assertIn("const DATE_AXIS_FONT_SIZES = [10, 9, 8, 7];", js)
        self.assertIn("const DATE_AXIS_ROTATION = 60;", js)
        self.assertIn("const DATE_AXIS_HORIZONTAL_LABEL_LIMIT = 10;", js)
        self.assertIn("const DATE_AXIS_YEAR_HORIZONTAL_LABEL_LIMIT = 25;", js)
        self.assertIn("const DATE_AXIS_VISIBLE_LABEL_LIMIT = 60;", js)
        self.assertIn('const DATE_AXIS_WEEKDAYS = ["Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat"];', js)
        self.assertIn('return kind === "date" || kind === "datetime";', js)
        self.assertIn("const rawXValues = data.rows.map((r) => r.x);", js)
        self.assertIn("const dateBucket = normaliseDateBucket(data.date_bucket);", js)
        self.assertIn("const displayKind = data.x_group_kind || data.x_kind;", js)
        self.assertIn('const xLabelPolicy = getXAxisLabelPolicy(labels, displayKind, rawXValues, dateBucket, chart.getWidth?.() || el("chart").clientWidth);', js)
        self.assertIn("showAllSymbol: true,", js)
        self.assertIn("interval: xLabelPolicy.interval,", js)
        self.assertIn("formatter: xLabelPolicy.formatter,", js)
        self.assertIn("showMinLabel: xLabelPolicy.showMinLabel,", js)
        self.assertIn("showMaxLabel: xLabelPolicy.showMaxLabel,", js)
        self.assertIn("dataZoom: xLabelPolicy.dataZoomEnabled ? lineBarDataZoomOptions() : [],", js)
        self.assertIn('chart.on("datazoom", scheduleDateXAxisLabelRefresh);', js)
        self.assertIn("function getDateXAxisLabelPolicy(labels, rawValues, dateBucket = \"none\", chartWidth = 0, visibleRange = null)", js)
        self.assertIn("function dateXAxisLabelFit(formattedLabels, chartWidth = 0, visibleRange = null, dateBucket = \"none\")", js)
        self.assertIn("function dateXAxisLabelRotation(dateBucket, visibleCount)", js)
        self.assertIn("rotate: visibleFit.show ? visibleFit.rotate : 0,", js)
        self.assertIn("function refreshDateXAxisLabelsForCurrentZoom()", js)
        self.assertIn("function formatDateAxisLabel(value, parsedDate, dateBucket = \"none\")", js)
        self.assertIn('if (dateBucket === "hour") return `${dateLabel} ${padDateAxisTime(parsedDate.hour)}:${padDateAxisTime(parsedDate.minute)}`;', js)
        self.assertIn('if (dateBucket === "day") return `${DATE_AXIS_WEEKDAYS[parsedDate.weekday]} ${dateLabel}`;', js)
        self.assertIn('if (dateBucket === "year") return String(parsedDate.year);', js)
        self.assertNotIn("dateXAxisLabelIndexes", js)
        self.assertNotIn("sparseDateXAxisLabelIndexes", js)
        self.assertNotIn("dateAxisYearLabelIndexes", js)
        self.assertIn("formatChartXLabel(r, data)", js)

        line_bar_js = self.assert_no_store("/static/app/line-bar-tool.js")[1].decode("utf-8")
        helper = "\n".join([
            'const DATE_AXIS_MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];',
            'const DATE_AXIS_WEEKDAYS = ["Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat"];',
            "const DATE_AXIS_FONT_SIZES = [10, 9, 8, 7];",
            "const DATE_AXIS_LABEL_WIDTH_FACTOR = 0.56;",
            "const DATE_AXIS_LABEL_PADDING = 8;",
            "const DATE_AXIS_HORIZONTAL_LABEL_LIMIT = 10;",
            "const DATE_AXIS_YEAR_HORIZONTAL_LABEL_LIMIT = 25;",
            "const DATE_AXIS_VISIBLE_LABEL_LIMIT = 60;",
            "const DATE_AXIS_ROTATION = 60;",
            self.js_function_source(line_bar_js, "normaliseDateXAxisVisibleRange"),
            self.js_function_source(line_bar_js, "dateXAxisLabelFit"),
            self.js_function_source(line_bar_js, "dateXAxisLabelRotation"),
            self.js_function_source(line_bar_js, "dateXAxisLabelFootprint"),
            self.js_function_source(line_bar_js, "dateXAxisLabelSpace"),
            self.js_function_source(line_bar_js, "dateXAxisPlotWidth"),
            self.js_function_source(line_bar_js, "estimateDateAxisLabelWidth"),
            self.js_function_source(line_bar_js, "parseDateCategory"),
            self.js_function_source(line_bar_js, "formatDateAxisLabel"),
            self.js_function_source(line_bar_js, "padDateAxisTime"),
            self.js_function_source(line_bar_js, "getDateXAxisLabelPolicy"),
        ])
        script = helper + """
function isoDate(offset) {
  const date = new Date(Date.UTC(2024, 0, 1 + offset, 13, 0, 0));
  return date.toISOString().replace("T", " ").slice(0, 19);
}
function isoHour(offset) {
  const date = new Date(Date.UTC(2024, 0, 1, offset, 0, 0));
  return date.toISOString().replace("T", " ").slice(0, 19);
}
const parsed = parseDateCategory("2024-01-02 13:00:00");
if (formatDateAxisLabel("x", parsed, "hour") !== "2 Jan 2024 13:00") throw new Error("hour label failed");
if (formatDateAxisLabel("x", parsed, "day") !== "Tue 2 Jan 2024") throw new Error("day weekday label failed");
if (formatDateAxisLabel("x", parsed, "week") !== "2 Jan 2024") throw new Error("week label failed");
if (formatDateAxisLabel("x", parsed, "year") !== "2024") throw new Error("year label failed");
if (formatDateAxisLabel("bad", null, "hour") !== "bad") throw new Error("raw fallback failed");

const small = [isoDate(0), isoDate(1), isoDate(2)];
const smallPolicy = getDateXAxisLabelPolicy(small, small, "day", 900);
if (!smallPolicy.show) throw new Error("small date labels should show");
if (smallPolicy.dataZoomEnabled) throw new Error("small date labels should not enable zoom");
if (smallPolicy.fontSize !== 10) throw new Error(`small font changed: ${smallPolicy.fontSize}`);
if (smallPolicy.rotate !== 0) throw new Error(`small date labels should be horizontal: ${smallPolicy.rotate}`);
if (smallPolicy.formatter("", 0) !== "Mon 1 Jan 2024") throw new Error("small formatter failed");

const medium = Array.from({ length: 20 }, (_, index) => isoDate(index));
const mediumPolicy = getDateXAxisLabelPolicy(medium, medium, "day", 1200);
if (!mediumPolicy.show) throw new Error("rotated medium date labels should show");
if (mediumPolicy.dataZoomEnabled) throw new Error("medium date labels should not need zoom when rotated");
if (mediumPolicy.rotate !== 60) throw new Error("medium labels should be rotated");

const years24 = Array.from({ length: 24 }, (_, index) => `${2000 + index}-01-01 00:00:00`);
const yearPolicy = getDateXAxisLabelPolicy(years24, years24, "year", 900);
if (!yearPolicy.show) throw new Error("24 year labels should show");
if (yearPolicy.dataZoomEnabled) throw new Error("24 year labels should not enable zoom");
if (yearPolicy.rotate !== 0) throw new Error("fewer than 25 year labels should be horizontal");
if (yearPolicy.formatter("", 0) !== "2000") throw new Error("year formatter failed");

const years25 = Array.from({ length: 25 }, (_, index) => `${2000 + index}-01-01 00:00:00`);
const yearRotatedPolicy = getDateXAxisLabelPolicy(years25, years25, "year", 900);
if (!yearRotatedPolicy.show) throw new Error("25 year labels should show");
if (yearRotatedPolicy.rotate !== 60) throw new Error("25 year labels should be rotated");

const quoteWeeks = Array.from({ length: 29 }, (_, index) => isoDate(index * 7));
const quoteWeekPolicy = getDateXAxisLabelPolicy(quoteWeeks, quoteWeeks, "week", 900);
if (!quoteWeekPolicy.show) throw new Error("QUOTE_DATE-style week labels should show on startup");
if (quoteWeekPolicy.dataZoomEnabled) throw new Error("QUOTE_DATE-style week labels should not enable zoom");
if (quoteWeekPolicy.rotate !== 60) throw new Error("QUOTE_DATE-style week labels should be rotated");

const hours59 = Array.from({ length: 59 }, (_, index) => isoHour(index));
const hours59Policy = getDateXAxisLabelPolicy(hours59, hours59, "hour", 300);
if (!hours59Policy.show) throw new Error("fewer than 60 hour labels should always show");
if (hours59Policy.dataZoomEnabled) throw new Error("fewer than 60 hour labels should not enable zoom");
if (hours59Policy.rotate !== 60) throw new Error("fewer than 60 dense hour labels should be rotated");
if (hours59Policy.fontSize !== 7) throw new Error(`fewer than 60 dense hour labels should use smallest font: ${hours59Policy.fontSize}`);

const months59 = Array.from({ length: 59 }, (_, index) => isoDate(index * 31));
const months59Policy = getDateXAxisLabelPolicy(months59, months59, "month", 300);
if (!months59Policy.show) throw new Error("fewer than 60 month labels should always show");
if (months59Policy.dataZoomEnabled) throw new Error("fewer than 60 month labels should not enable zoom");

const dense = Array.from({ length: 80 }, (_, index) => isoDate(index));
const densePolicy = getDateXAxisLabelPolicy(dense, dense, "day", 500);
if (densePolicy.show) throw new Error("dense date labels should hide");
if (!densePolicy.dataZoomEnabled) throw new Error("dense date labels should enable zoom");
if (densePolicy.fontSize !== 7) throw new Error(`dense fallback font changed: ${densePolicy.fontSize}`);
if (densePolicy.rotate !== 0) throw new Error("hidden dense labels should not rotate");

const zoomedPolicy = getDateXAxisLabelPolicy(dense, dense, "day", 500, { startIndex: 0, endIndex: 3 });
if (!zoomedPolicy.show) throw new Error("zoomed date labels should reappear");
if (!zoomedPolicy.dataZoomEnabled) throw new Error("zoom should remain enabled after zooming");
if (zoomedPolicy.rotate !== 0) throw new Error("zoomed fewer-than-10 labels should be horizontal");
"""
        self.run_node_script(script)

    def test_line_bar_date_bucket_suggestion_contract(self) -> None:
        js = self.app_js_contract()

        self.assertIn('/api/date-bucket/suggestion', js)
        self.assertIn("function requestDateBucketSuggestionForSelectedColumn", js)
        self.assertIn("function currentDateBucketFeatureKey()", js)
        self.assertIn('return JSON.stringify([sourceId, state.x || "", state.activeFilter || ""]);', js)
        self.assertIn('setGroupMeta("line_bar", "Estimating date bucket...");', js)
        self.assertIn('state.dateBucket = "year";', js)
        self.assertIn("state.dateBucketManualKey = state.dateBucketFeature;", js)
        self.assertIn("resetDateBucketSuggestionState();", js)
        self.assertIn("dateBucketSuggestionRequestSeq: 0", js)
        self.assertIn("function invalidateLineBarDateBucketSuggestion()", js)

    def test_numeric_x_axis_labels_are_cleaned_defensively(self) -> None:
        js = self.app_js_contract()

        self.assertIn("function formatXLabel(value, kind)", js)
        self.assertIn('if (kind === "numeric") return formatNumericXLabel(value);', js)
        self.assertIn("function formatNumericXLabel(value)", js)
        self.assertIn("const number = Number(text);", js)
        self.assertIn("if (!Number.isFinite(number)) return text;", js)
        self.assertIn("return number.toLocaleString(undefined, { maximumFractionDigits: 12 });", js)
        self.assertIn('if (kind !== "integer") return String(value);', js)
        self.assertIn("const CATEGORICAL_AXIS_LABEL_PADDING = 8;", js)
        self.assertIn("const plotWidth = dateXAxisPlotWidth(chartWidth);", js)
        self.assertIn("const slotWidth = plotWidth / Math.max(1, labels.length);", js)
        self.assertIn("const horizontalFootprint = estimatedTextWidth + CATEGORICAL_AXIS_LABEL_PADDING;", js)
        self.assertIn("const rotate = labels.length > 30 || maxLength > 10 || horizontalFootprint > slotWidth ? 65 : 0;", js)

        line_bar_js = self.assert_no_store("/static/app/line-bar-tool.js")[1].decode("utf-8")
        script = "\n".join([
            "const LABEL_DENSITY_LIMIT = 200;",
            "const CATEGORICAL_AXIS_LABEL_PADDING = 8;",
            "function isDateKind() { return false; }",
            self.js_function_source(line_bar_js, "dateXAxisPlotWidth"),
            self.js_function_source(line_bar_js, "getXAxisLabelPolicy"),
            """
const labels30 = Array.from({ length: 30 }, (_, index) => String(index));
const labels31 = Array.from({ length: 31 }, (_, index) => String(index));
const policy30 = getXAxisLabelPolicy(labels30, "integer", labels30, "none", 900);
const policy31 = getXAxisLabelPolicy(labels31, "integer", labels31, "none", 900);
if (policy30.rotate !== 0) throw new Error(`30 short labels should stay horizontal, got ${policy30.rotate}`);
if (policy31.rotate !== 65) throw new Error(`31 short labels should rotate, got ${policy31.rotate}`);
const isoDates23 = Array.from({ length: 23 }, (_, index) => `2026-06-${String(index + 6).padStart(2, "0")}`);
const denseIsoPolicy = getXAxisLabelPolicy(isoDates23, "categorical", isoDates23, "none", 900);
if (denseIsoPolicy.rotate !== 65) throw new Error(`dense 10-character categorical labels should rotate, got ${denseIsoPolicy.rotate}`);
const longPolicy = getXAxisLabelPolicy(["short", "long-label-x"], "categorical", ["short", "long-label-x"], "none", 900);
if (longPolicy.rotate !== 65) throw new Error("long labels should still rotate");
""",
        ])
        self.run_node_script(script)

    def test_line_bar_x_axis_title_uses_selected_feature_with_tight_spacing(self) -> None:
        js = self.app_js_contract()

        self.assertIn('const textStyle = { color: getCss("--text"), fontWeight: 700, fontSize: 13 };', js)
        self.assertIn('name: data.x || "",', js)
        self.assertIn('nameLocation: "middle",', js)
        self.assertIn("nameGap: xLabelPolicy.nameGap,", js)
        self.assertIn('nameTextStyle: { color: getCss("--text"), fontSize: 13, fontWeight: 700 },', js)
        self.assertIn("const titleGap = rotate ? Math.max(26, labelSpace - 10) : 26;", js)
        self.assertIn("nameGap: titleGap,", js)
        self.assertIn("bottom: titleGap + 16 + dataZoomSpace,", js)
        self.assertIn("const titleGap = visibleFit.show ? Math.max(26, labelSpace - 10) : 22;", js)
        self.assertIn("nameGap: titleGap,", js)
        self.assertIn("bottom: titleGap + 16 + dataZoomSpace,", js)

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
        self.assertIn("function syncBaseMapVisualState()", js)
        self.assertIn("const sameBaseMap = nextBaseMap === state.baseMap;", js)
        self.assertIn("if (sameBaseMap && tileLayerMatches) {\n      syncBaseMapVisualState();\n      syncMapControls();\n      return;\n    }", js)
        self.assertIn("container._lucidumBaseTileLayer = baseTileLayer;", js)
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
        self.assertIn("function currentThemeKey()", js)
        self.assertIn('return document.body.classList.contains("dark") ? "dark" : "light";', js)
        self.assertIn("function markToolCacheThemeSynced(tool)", js)
        self.assertIn("toolCache(tool).themeKey = currentThemeKey();", js)
        self.assertIn("const themeChanged = cache.themeKey !== currentThemeKey();", js)
        self.assertIn("renderIfCached: options.renderIfCached || themeChanged", js)
        self.assertIn("function syncActiveToolTheme()", js)
        self.assertIn("syncActiveToolTheme();", js)
        self.assertIn("syncCartoBaseMapForTheme();", js)
        self.assertIn('if (activeTool === "gbm") {\n          measureToolRender("gbm", () => gbmTool.refreshTheme());', js)
        self.assertIn("lineBarTool.bindControls();", js)
        self.assertIn("syncThemeButton();", js)
        self.assertIn("applyMapBackground();", js)
        self.assertIn("async function useCachedMapData(cache, options = {})", js)
        self.assertIn("if (options.renderIfCached) {\n      refreshTheme();", js)
        self.assertIn("useCached: (cache, options) => ukMapTool.useCached(cache, options),", js)
        self.assertIn("canUseCached: canUseCachedMapData,", js)
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
        self.assertIn("previousBandWidthsByFeature: {}", js)
        self.assertIn('el("bandLabel").textContent = state.quantileMode === "quantile" ? "Quantiles" : "Banding";', js)
        self.assertIn('el("quantileControl").classList.toggle("hidden", !isNumeric);', js)
        self.assertIn('quantileMode: isNumeric ? state.quantileMode : "off"', js)
        self.assertIn('/api/banding/suggestion', js)
        self.assertIn("requestBandSuggestionForSelectedColumn", js)
        self.assertIn('if (state.quantileMode === "quantile") {\n      state.bandFeature = bandFeatureKey;\n      clearPendingBandSuggestion();\n      syncBandingControl();\n      return;\n    }', js)
        self.assertIn('if (isNumeric && state.quantileMode !== "quantile" && state.tool === "line_bar" && state.bandFeature !== bandFeatureKey)', js)
        self.assertIn('const previousControlValue = state[group.dataset.control];', js)
        self.assertIn('state.quantileMode === "quantile" && previousControlValue !== "quantile"', js)
        self.assertIn("rememberNonQuantileBandWidthForCurrentFeature();", js)
        self.assertIn("restoreNonQuantileBandWidthForCurrentFeature();", js)
        self.assertIn('state.bandWidth = "10";', js)
        self.assertIn('function normalizeBandWidthForQuantiles()', js)

    def test_line_bar_quantile_range_label_contract(self) -> None:
        js = self.app_js_contract()

        self.assertIn("function formatChartXLabel(row, data)", js)
        self.assertIn('const rangeLabel = formatQuantileRangeLabel(row, data, "\\n");', js)
        self.assertIn("function formatTableXLabel(row, data)", js)
        self.assertIn('const rangeLabel = formatQuantileRangeLabel(row, data, ": ");', js)
        self.assertIn('if ((data?.x_group_kind || data?.x_kind) !== "quantile") return "";', js)
        self.assertIn('return start === end\n      ? `${prefix}${separator}${start}`\n      : `${prefix}${separator}${start} to ${end}`;', js)
        self.assertIn("return formatNumber(value);", js)
        self.assertIn("const labels = data.rows.map((r) => formatChartXLabel(r, data));", js)
        self.assertIn("const displayKind = data.x_group_kind || data.x_kind;", js)
        self.assertIn("x: formatTableXLabel(row, data),", js)
        self.assertIn('if (kind === "quantile") return getQuantileXAxisLabelPolicy(labels, chartWidth);', js)
        self.assertIn("hideOverlap: Boolean(xLabelPolicy.hideOverlap),", js)
        self.assertIn("return chartDensityMessage(labels.length, !xLabelPolicy.show, !dataLabelsAllowed && labelMode !== \"-\", xLabelPolicy.hiddenReason, Boolean(xLabelPolicy.hideOverlap));", js)

    def test_line_bar_toolbar_shared_controls_precede_type_specific_controls(self) -> None:
        _, html_body = self.assert_no_store("/")
        html = html_body.decode("utf-8")
        js = self.app_js_contract()
        shared_controls = [
            "<h3>Group low weights</h3>",
            "<h3>Labels</h3>",
            "<h3>Partial dependence</h3>",
            "<h3>Response transform</h3>",
            "<h3>Sigma bars</h3>",
        ]
        type_specific_controls = [
            'id="sortControl"',
            'id="dateControl"',
            'id="bandControl"',
            'id="quantileControl"',
        ]

        self.assertIn('id="partialDependenceControl" class="control hidden"', html)
        self.assertIn('id="responseTransformControl" class="control hidden"', html)
        self.assertIn('id="sigmaControl" class="control hidden"', html)
        self.assertIn('const modelControlsAvailable = toolEnabled("gbm") || toolEnabled("glm");', js)
        self.assertIn('el("partialDependenceControl").classList.toggle("hidden", !modelControlsAvailable);', js)
        self.assertIn('el("responseTransformControl").classList.toggle("hidden", !modelControlsAvailable);', js)
        self.assertIn('el("sigmaControl").classList.toggle("hidden", !hasExpected);', js)
        for before, after in zip(shared_controls, shared_controls[1:]):
            self.assertLess(html.index(before), html.index(after))
        for shared_control in shared_controls:
            for type_specific_control in type_specific_controls:
                self.assertLess(html.index(shared_control), html.index(type_specific_control))
        self.assertLess(html.index('id="sortControl"'), html.index('id="dateControl"'))
        self.assertLess(html.index('id="dateControl"'), html.index('id="bandControl"'))
        self.assertLess(html.index('id="bandControl"'), html.index('id="quantileControl"'))

    def test_line_bar_partial_dependence_shap_contract(self) -> None:
        _, html_body = self.assert_no_store("/")
        html = html_body.decode("utf-8")
        js = self.app_js_contract()

        self.assertIn("<h3>Partial dependence</h3>", html)
        self.assertIn('<div class="segmented" data-control="partialDependence">', html)
        self.assertIn('<button data-value="none" class="active">None</button>', html)
        self.assertIn('<button data-value="shap">SHAP</button>', html)
        self.assertIn('<button data-value="glm">GLM</button>', html)
        self.assertIn('<button data-value="both">Both</button>', html)
        self.assertIn('id="shapSortButton" data-value="shap" class="hidden"', html)
        self.assertLess(html.index("<h3>Labels</h3>"), html.index("<h3>Partial dependence</h3>"))
        self.assertLess(html.index("<h3>Partial dependence</h3>"), html.index('id="bandControl"'))
        self.assertIn('partialDependence: "none"', js)
        self.assertIn('function selectedPartialDependenceMode()', js)
        self.assertIn('function shapPartialDependenceVisible()', js)
        self.assertIn('partialDependence: { mode: selectedPartialDependenceMode() }', js)
        self.assertIn('el("shapSortButton")?.classList.toggle("hidden", !shapSortAvailable);', js)
        self.assertIn('"partialDependence"', js)
        self.assertIn('const GLM_LINE_COLOR = "#1f7a8c";', js)
        self.assertIn("const LINE_BAR_MAIN_LEGEND_TOP = 52;", js)
        self.assertIn("const LINE_BAR_OVERLAY_LEGEND_TOP = 78;", js)
        self.assertIn("const LINE_BAR_GRID_TOP = 112;", js)
        self.assertIn("const LINE_BAR_OVERLAY_GRID_TOP = 140;", js)
        self.assertIn("function partialDependenceOverlay(data, key)", js)
        self.assertIn("function shapPartialDependenceSeries(data)", js)
        self.assertIn("function glmPartialDependenceSeries(data)", js)
        self.assertIn("function shapRibbonSeries(rows, lowKey, highKey, label, color)", js)
        self.assertIn("function lineBarLegendOptions(legendData, mainLegendSelection, overlayLegendData, overlayLegendSelection)", js)
        self.assertIn("const textStyle = { color: getCss(\"--text\"), fontWeight: 700, fontSize: 13 };", js)
        self.assertIn("const overlayTextStyle = { color: getCss(\"--text\"), fontWeight: 400, fontSize: 11 };", js)
        self.assertIn("top: LINE_BAR_MAIN_LEGEND_TOP", js)
        self.assertIn("top: LINE_BAR_OVERLAY_LEGEND_TOP", js)
        self.assertIn("--base-bar:", self.app_css_contract())
        self.assertIn("selected: mainLegendSelection", js)
        self.assertNotIn("selectedMode: false", js)
        self.assertIn("function legendEntryName(entry)", js)
        self.assertIn("const names = entries.map(legendEntryName).filter(Boolean);", js)
        self.assertIn("matchingLegendSelection(previousOption, legendData)", js)
        self.assertIn("matchingLegendSelection(previousOption, overlayLegendData)", js)
        self.assertIn("series: [barSeries, ...shapSeries, ...glmSeries, ...lineSeries, ...(upliftBaseline ? [upliftBaseline] : []), ...customSeries]", js)
        self.assertIn("grid: { left: 72, right: 76, top: hasOverlaySeries ? LINE_BAR_OVERLAY_GRID_TOP : LINE_BAR_GRID_TOP", js)

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

        self.assertIn('class="map-viewport-icon-london"', js)
        self.assertIn(".map-viewport-button img.map-viewport-icon-london", css)
        self.assertIn("width: 30px;", css)
        self.assertIn("height: 30px;", css)
        self.assertIn("body.dark .map-viewport-button img", css)
        self.assertIn("mix-blend-mode: screen;", css)
        self.assertIn("filter: invert(1) grayscale(1) brightness(1.7) contrast(1.08);", css)

    def test_map_layer_control_uses_distinct_radio_groups(self) -> None:
        _, html_body = self.assert_no_store("/")
        css = self.app_css_contract()
        html = html_body.decode("utf-8")
        js = self.app_js_contract()

        self.assertIn('import { createUkMapTool } from "./uk-map-tool.js";', js)
        self.assertIn("export function createUkMapTool", js)
        self.assertIn("clearActiveFavouriteSelection = () => {},", js)
        self.assertIn("function captureFavouriteState()", js)
        self.assertIn("function applyFavouriteState(map = {})", js)
        self.assertIn("view: normaliseMapView({ center: payload.center, zoom: payload.zoom }),", js)
        self.assertIn("const view = normaliseMapView(currentMapView() || state.mapView) || normaliseMapView(MAP_DEFAULT_VIEW);", js)
        self.assertIn("center: view?.center || null,", js)
        self.assertIn("zoom: view?.zoom ?? null,", js)
        self.assertIn("state.mapViewRestorePending = next.view;", js)
        self.assertIn("const restoreView = state.mapViewRestorePending || state.mapView;", js)
        self.assertIn("if (restored && state.mapViewRestorePending) state.mapViewRestorePending = null;", js)
        self.assertIn("cancelMapRequests({ preservePendingRestore: true });", js)
        self.assertIn("async function refreshMapFavouriteViewData()", js)
        self.assertIn("const canUseCachedMap = Boolean(", js)
        self.assertIn("&& ukMapTool.canUseCached(cache)", js)
        self.assertIn("return refreshUkMap({ renderIfCached: true });", js)
        self.assertIn("const data = await handler.fetch(request, requestKey, options);", js)
        self.assertIn("fetch: (request, requestKey, options) => ukMapTool.fetchData(request, requestKey, options),", js)
        self.assertIn("async function fetchMapData(request, requestKey, options = {})", js)
        self.assertIn("const quietPending = Boolean(options.preserveRenderedMap && options.suppressPendingMeta);", js)
        self.assertIn('if (!quietPending) setGroupMeta("uk_map", "Computing map...");', js)
        self.assertIn("function canRefreshMapInPlace(request)", js)
        self.assertIn("canRefreshInPlace: canRefreshMapInPlace,", js)
        self.assertIn("return refreshUkMap({ force: true, preserveRenderedMap: true, suppressPendingMeta: true });", js)
        self.assertLess(
            js.index("return refreshUkMap({ force: true, preserveRenderedMap: true, suppressPendingMeta: true });"),
            js.index("ukMapTool.showPendingRestore();"),
        )
        self.assertIn("captureFavouriteState,", js)
        self.assertIn("applyFavouriteState,", js)
        self.assertIn('clearActiveFavouriteSelection: () => clearActiveFavouriteSelectionForScope("map_view"),', js)
        self.assertIn("ukMapTool.bindControls();", js)
        self.assertIn("ukMapTool.activate();", js)
        self.assertIn('<div id="mapSliderGrid" class="map-slider-grid">', html)
        self.assertIn('<label id="mapLineWeightControl" class="map-slider-control">', html)
        self.assertIn("<span>Line width</span>", html)
        self.assertIn('<span class="slider-scale"><b>0</b><b id="mapLineWeightValue">1</b><b>10</b></span>', html)
        self.assertIn('<input id="mapLineWeight" type="range" min="0" max="10" step="1" value="1" />', html)
        self.assertIn("lineWeight: clampMapNumber(payload.lineWeight, 1, 0, 10, { integer: true }),", js)
        self.assertIn("function mapStrokeWeightForLineWeight(value = state.mapLineWeight)", js)
        self.assertIn("return Math.min(10, sliderValue) / 2;", js)
        self.assertIn("const baseWeight = mapStrokeWeightForLineWeight();", js)
        self.assertNotIn("lineWeight: clampMapNumber(payload.lineWeight, 1, 0, 5, { integer: true }),", js)
        self.assertIn('<label id="mapDotSizeControl" class="map-slider-control" hidden>', html)
        self.assertIn("<span>Dot size</span>", html)
        self.assertIn('<span class="slider-scale"><b>1</b><b id="mapDotSizeValue">1</b><b>10</b></span>', html)
        self.assertIn('<input id="mapDotSize" type="range" min="1" max="10" step="1" value="1" />', html)
        self.assertIn("dotSize: clampMapNumber(payload.dotSize, 1, 1, 10, { integer: true }),", js)
        self.assertNotIn("dotSize: clampMapNumber(payload.dotSize, 1, 0, 5, { integer: true }),", js)
        self.assertIn("<span>Opacity</span>", html)
        self.assertIn('<span class="slider-scale"><b>0</b><b id="mapOpacityValue">10</b><b>10</b></span>', html)
        self.assertIn('<input id="mapOpacity" type="range" min="0" max="10" step="1" value="10" />', html)
        self.assertIn("formatOpacitySliderValue(state.mapOpacity)", js)
        self.assertIn("function formatOpacitySliderValue(value)", js)
        self.assertIn("function opacitySliderValue(value = state.mapOpacity)", js)
        self.assertIn("return Math.round(number * 10);", js)
        self.assertIn("function opacityFromSliderValue(value)", js)
        self.assertIn("return number / 10;", js)
        self.assertIn('state[stateKey] = id === "mapOpacity" ? opacityFromSliderValue(event.target.value) : Number(event.target.value);', js)
        self.assertNotIn('id="mapOpacity" type="range" min="0" max="1" step="0.2"', html)
        self.assertNotIn('id="mapOpacity" type="range" min="0" max="1" step="0.1"', html)
        self.assertIn("<span>Extremes</span>", html)
        self.assertIn("<span>Labels</span>", html)
        self.assertIn("<span>Smooth</span>", html)
        self.assertIn('<label id="mapHotspotsControl" class="map-slider-control">', html)
        self.assertIn('<span class="slider-scale"><b id="mapHotspotsMinLabel" class="map-extreme-label">Low</b><b id="mapHotspotsValue">All</b><b id="mapHotspotsMaxLabel" class="map-extreme-label">High</b></span>', html)
        self.assertIn('<input id="mapHotspots" type="range" min="-9" max="9" step="1" value="0" />', html)
        self.assertIn('<label id="mapLabelControl" class="map-slider-control">', html)
        self.assertIn('<span class="slider-scale"><b>0</b><b id="mapLabelSizeValue">0</b><b>10</b></span>', html)
        self.assertIn('<input id="mapLabelSize" type="range" min="0" max="10" step="1" value="0" />', html)
        self.assertIn('<label id="mapSmoothingControl" class="map-slider-control" hidden>', html)
        self.assertIn('<span class="slider-scale"><b>N0</b><b id="mapSmoothingValue">None</b><b>N5</b></span>', html)
        self.assertIn('<input id="mapSmoothing" type="range" min="0" max="5" step="1" value="0" />', html)
        self.assertIn('<button id="mapControlReset" class="map-control-reset" type="button" title="Collapse map controls" aria-label="Collapse map controls" aria-controls="mapFloatingControl" aria-expanded="true">', html)
        self.assertNotIn("Reset map controls position", html)
        self.assertNotIn("<span>Line thickness</span>", html)
        self.assertNotIn("<span>Top/Bottom %</span>", html)
        self.assertNotIn("<span>Label size</span>", html)
        self.assertNotIn("<span>Max/Min</span>", html)
        self.assertNotIn('<input id="mapHotspots" type="range" min="-1" max="1" step="0.1" value="0" />', html)
        self.assertNotIn('<input id="mapHotspots" type="range" min="-0.1" max="0.1" step="0.01" value="0" />', html)
        self.assertNotIn('input id="mapHotspots" type="range" min="-20" max="20" step="5"', html)
        self.assertNotIn("Hot/not-spots", html)
        self.assertIn('id="mapControlFilter"><button id="mapControlFilterClearBtn" class="filter-row-clear" type="button" title="Clear filter" aria-label="Clear filter" hidden></button><span id="mapControlFilterText" class="filter-badge-text">no filter</span></span>', html)
        self.assertIn("--map-floating-right: 19px;", css)
        self.assertIn("--map-floating-right: 11px;", css)
        self.assertNotIn("--map-floating-right: 24px;", css)
        self.assertNotIn("--map-floating-right: 16px;", css)
        self.assertIn('id="mapBaseLayerTiles" class="map-tile-grid map-base-layer-tiles" aria-label="Base map layer"', html)
        self.assertIn('id="mapLevelTiles" class="map-tile-grid map-level-tiles" aria-label="Map level"', html)
        self.assertNotIn('id="mapViewportControls" class="map-viewport-controls" aria-label="Map viewport"', html)
        self.assertIn('class="map-palette-buttons map-palette-tiles"', html)
        self.assertIn('data-palette="divergent" title="Split"', html)
        self.assertIn('<span class="map-control-tile-text">Split</span>', html)
        self.assertIn('<span class="map-control-tile-text">Spectral</span>', html)
        self.assertIn('<span class="map-control-tile-text">Viridis</span>', html)
        self.assertIn('/tools/uk-map/static/icons/map-controls/base-blank.png', html)
        self.assertIn('/tools/uk-map/static/icons/map-controls/base-esri.png', html)
        self.assertIn('/tools/uk-map/static/icons/map-controls/base-osm.png', html)
        self.assertIn('/tools/uk-map/static/icons/map-controls/base-aerial.png', html)
        self.assertIn('/tools/uk-map/static/icons/map-controls/base-light.png', html)
        self.assertIn('/tools/uk-map/static/icons/map-controls/base-dark.png', html)
        self.assertIn('/tools/uk-map/static/icons/map-controls/level-area.png', html)
        self.assertIn('/tools/uk-map/static/icons/map-controls/level-sector.png', html)
        self.assertIn('/tools/uk-map/static/icons/map-controls/level-unit.png', html)
        self.assertIn('<button id="mapZoomIn" class="map-viewport-button" type="button" title="Zoom in" aria-label="Zoom in">+</button>', js)
        self.assertIn('<button id="mapZoomOut" class="map-viewport-button" type="button" title="Zoom out" aria-label="Zoom out">&minus;</button>', js)
        self.assertIn('<button id="mapFitUk" class="map-viewport-button" type="button" title="Fit UK map layer" aria-label="Fit UK map layer">', js)
        self.assertIn('<button id="mapZoomLondon" class="map-viewport-button" type="button" title="Zoom to London" aria-label="Zoom to London">', js)
        self.assertIn("width: min(400px, calc(100% - 64px));", css)
        self.assertIn(".map-floating-control.collapsed {\n        width: 36px;\n        height: 36px;", css)
        self.assertIn("background: transparent;\n        box-shadow: none;\n        overflow: visible;", css)
        self.assertIn(".map-floating-control.collapsed .map-floating-header strong,\n      .map-floating-control.collapsed .map-floating-header #mapGroupMeta,\n      .map-floating-control.collapsed .map-floating-header #mapControlFilter,\n      .map-floating-control.collapsed .map-control-panel-body,\n      .map-floating-control.collapsed .map-floating-row,\n      .map-floating-control.collapsed .map-slider-grid", css)
        self.assertIn("#mapControlFilter {\n        background: transparent;\n        border: 1px solid transparent;\n        border-radius: 4px;\n        display: inline-flex;\n        align-items: center;\n        gap: 4px;", css)
        self.assertIn("font-size: 10px;\n        justify-self: start;", css)
        self.assertIn("max-width: 100%;\n        min-width: 0;\n        padding: 1px 4px;\n        text-align: left;\n        width: fit-content;", css)
        self.assertIn("#mapControlFilter:not(.map-filter--applied) {\n        justify-content: flex-start;\n        padding-left: 0;\n        padding-right: 0;\n        width: 100%;", css)
        self.assertIn("#mapControlFilter .filter-badge-text {\n        font-size: inherit;\n        line-height: inherit;", css)
        self.assertIn("#mapControlFilter.map-filter--applied {\n        background: var(--filter-applied-bg);\n        border-color: var(--filter-applied-border);\n        color: var(--filter-applied-text);", css)
        self.assertIn(".map-floating-control.collapsed .map-control-reset {\n        grid-column: auto;\n        grid-row: auto;", css)
        self.assertIn("width: 36px;\n        height: 36px;", css)
        self.assertIn(".map-slider-grid {\n        display: grid;\n        grid-template-columns: minmax(0, 1fr);\n        gap: 7px;", css)
        self.assertNotIn("gap: 8px 16px;", css)
        self.assertIn("--map-control-tile-size: 58px;", css)
        self.assertIn(".map-base-layer-tiles {\n        grid-template-columns: repeat(6, minmax(0, var(--map-control-tile-size)));", css)
        self.assertIn(".map-level-tiles {\n        grid-template-columns: repeat(3, minmax(0, var(--map-control-tile-size)));", css)
        self.assertIn(".map-level-palette-row {\n        display: grid;\n        grid-template-columns: repeat(2, max-content);", css)
        self.assertIn("align-items: start;\n        gap: 6px;", css)
        self.assertIn(".map-palette-buttons {\n        display: grid;\n        grid-template-columns: repeat(3, minmax(0, var(--map-control-tile-size)));", css)
        self.assertIn(".map-palette-button.active {", css)
        self.assertIn(".map-control-tile input:checked + .map-control-tile-card", css)
        self.assertIn(".map-control-tile input:disabled + .map-control-tile-card", css)
        self.assertIn(".map-viewport-control {\n        display: grid;\n        grid-template-rows: repeat(4, 34px);", css)
        self.assertNotIn(".map-viewport-controls", css)
        self.assertNotIn(".map-control-strip", css)
        self.assertNotIn("grid-template-columns: repeat(4, 78px);", css)
        self.assertNotIn("grid-template-columns: repeat(5, minmax(0, 1fr));", css)
        self.assertIn(".map-slider-control[hidden] {\n        display: none;", css)
        self.assertNotIn(".map-slider-grid.unit-mode #mapHotspotsControl", css)
        self.assertIn(".map-slider-control.disabled > span,\n      .map-slider-control input[type=\"range\"]:disabled", css)
        self.assertIn("--map-slider-active: color-mix(in srgb, var(--accent) 88%, #0f4f9e);", css)
        self.assertIn("--map-slider-end-label: color-mix(in srgb, var(--text) 74%, transparent);", css)
        self.assertIn("--map-slider-track: color-mix(in srgb, var(--text) 24%, var(--panel));", css)
        self.assertIn("--map-slider-thumb-size: 22px;", css)
        self.assertIn("--map-slider-track-height: 6px;", css)
        self.assertIn("body.dark .map-slider-control {\n        --map-slider-active: color-mix(in srgb, var(--accent) 92%, white);", css)
        self.assertIn(".map-slider-control {\n        --map-slider-active: color-mix(in srgb, var(--accent) 88%, #0f4f9e);", css)
        self.assertIn("display: grid;\n        grid-template-columns: 66px max-content minmax(0, 1fr) max-content;", css)
        self.assertIn(".slider-scale {\n        display: contents;", css)
        self.assertIn("color: var(--map-slider-end-label);", css)
        self.assertIn("font-weight: 600;", css)
        self.assertIn(".slider-scale b:first-child {\n        grid-column: 2;\n        grid-row: 1;", css)
        self.assertIn(".slider-scale b:nth-child(2) {\n        display: none;", css)
        self.assertIn(".slider-scale b:last-child {\n        grid-column: 4;\n        grid-row: 1;", css)
        self.assertIn(".slider-scale b.map-extreme-label {\n        color: var(--map-extreme-color, var(--map-slider-end-label));\n        font-weight: 700;", css)
        self.assertIn(".map-slider-control input[type=\"range\"] {\n        appearance: none;\n        -webkit-appearance: none;", css)
        self.assertIn("grid-column: 3;\n        grid-row: 1;", css)
        self.assertIn("accent-color: var(--map-slider-active);", css)
        self.assertNotIn("accent-color: #8bb8ff;", css)
        self.assertIn("var(--map-slider-active) var(--map-slider-progress, 0%),", css)
        self.assertIn("var(--map-slider-track) var(--map-slider-progress, 0%),", css)
        self.assertIn(") center / calc(100% - var(--map-slider-thumb-size)) var(--map-slider-track-height) no-repeat;", css)
        self.assertIn("height: var(--map-slider-track-height);\n        border-radius: 999px;", css)
        self.assertIn(".map-slider-control input[type=\"range\"]::-moz-range-progress {\n        background: var(--map-slider-active);", css)
        self.assertIn("margin-top: calc((var(--map-slider-track-height) - var(--map-slider-thumb-size)) / 2);", css)
        self.assertIn("border: 1px solid var(--map-slider-thumb-border);", css)
        self.assertIn("#mapHotspots.map-slider-thumb-centered::-webkit-slider-thumb", css)
        self.assertIn("#mapHotspots.map-slider-thumb-centered::-moz-range-thumb", css)
        self.assertIn("background: var(--map-slider-active);", css)
        self.assertIn("#mapHotspotsControl .slider-scale b:first-child,\n      #mapHotspotsControl .slider-scale b:last-child", css)
        self.assertIn("function updateMapSliderProgress(input)", js)
        self.assertIn('input.style.setProperty("--map-slider-progress", `${Math.max(0, Math.min(100, progress))}%`);', js)
        self.assertIn('input.classList.toggle("map-slider-thumb-centered", value === 0);', js)
        self.assertIn("function syncMapSliderProgressStyles()", js)
        self.assertIn('["mapLineWeight", "mapDotSize", "mapOpacity", "mapHotspots", "mapLabelSize", "mapSmoothing"]', js)
        self.assertIn("updateMapSliderProgress(event.target);", js)
        self.assertIn("width: min(400px, calc(100% - 24px));", css)
        self.assertNotIn("width: min(440px, calc(100% - 24px));", css)
        self.assertIn("        .map-base-layer-tiles {\n          grid-template-columns: repeat(3, minmax(0, var(--map-control-tile-size)));", css)
        self.assertIn("        .map-level-palette-row {\n          grid-template-columns: max-content;", css)
        self.assertIn("width: min(208px, calc(100% - 16px));", css)
        self.assertIn("        .map-control-panel-body {\n          --map-control-tile-size: 58px;", css)
        self.assertIn("        .postcode-search {\n          grid-template-columns: minmax(0, 1fr) 28px 28px;", css)
        self.assertIn("        .map-slider-control {\n          grid-template-columns: 62px minmax(0, 1fr);", css)
        self.assertNotIn("        .map-slider-control > span:first-child {\n          display: none;", css)
        self.assertIn("        .slider-scale b:first-child,\n        .slider-scale b:last-child {\n          display: none;", css)
        self.assertIn("        .map-slider-control input[type=\"range\"] {\n          grid-column: 2;", css)
        self.assertNotIn("grid-template-columns: repeat(2, 78px);", css)
        self.assertNotIn("justify-content: space-around;", css)
        self.assertIn(".map-label-icon {\n        background: transparent;\n        border: 0;", css)
        self.assertIn('className: "map-label-icon",', js)
        self.assertIn("display: inline-block;\n        font-weight: 650;", css)
        self.assertIn("left: 0;\n        line-height: 1.1;\n        position: absolute;\n        top: 0;", css)
        self.assertIn("transform: translate(-50%, -50%);\n        transform-origin: center;", css)
        self.assertIn("function normaliseMapHotspotNotch(value = state.mapHotspots)", js)
        self.assertIn("return Math.max(-9, Math.min(9, Math.round(number)));", js)
        self.assertIn("function mapHotspotSelection(value = state.mapHotspots)", js)
        self.assertIn("if (notch === 0) return null;", js)
        self.assertIn("direction: notch > 0 ? -1 : 1,", js)
        self.assertIn("fraction: mapHotspotPercent(notch) / 100,", js)
        self.assertIn("function mapHotspotPercent(value = state.mapHotspots)", js)
        self.assertIn("return 100 - (Math.abs(notch) * 10);", js)
        self.assertIn("if (key === null || key === undefined || value === null) continue;", js)
        self.assertIn("const selection = mapHotspotSelection();", js)
        self.assertIn("if (!selection) return null;", js)
        self.assertIn("(a.value - b.value) * selection.direction", js)
        self.assertIn("return a.index - b.index;", js)
        self.assertIn("Math.ceil(validRows.length * selection.fraction)", js)
        self.assertIn("function sectorLineWeightScaleForZoom(zoom)", js)
        self.assertIn("return baseWeight * sectorLineWeightScaleForZoom(ukMap.getZoom());", js)
        self.assertNotIn("Math.min(baseWeight, 0.15)", js)
        self.assertNotIn("Math.min(baseWeight, 0.85)", js)
        self.assertIn("formatHotspotSliderValue(state.mapHotspots)", js)
        self.assertIn("function formatHotspotSliderValue(value)", js)
        self.assertIn('if (notch === 0) return "All";', js)
        self.assertIn('return `${notch < 0 ? "B" : "T"}${mapHotspotPercent(notch)}`;', js)
        self.assertIn("function activeMapExtremeColors()", js)
        self.assertIn("low: palette[0] || MAP_MISSING_COLOR,", js)
        self.assertIn("high: palette[palette.length - 1] || MAP_MISSING_COLOR,", js)
        self.assertIn("function syncMapExtremeLabels()", js)
        self.assertIn('lowLabel.textContent = "Low";', js)
        self.assertIn('highLabel.textContent = "High";', js)
        self.assertIn('lowLabel.style.setProperty("--map-extreme-color", colors.low);', js)
        self.assertIn('highLabel.style.setProperty("--map-extreme-color", colors.high);', js)
        self.assertIn("syncMapExtremeLabels();", js)
        self.assertNotIn("B100", js)
        self.assertNotIn("T100", js)
        self.assertNotIn("if (fraction >= 1) return null;", js)
        self.assertNotIn("mapLineWeightLabel", js)
        self.assertNotIn("mapLineWeightLabel", html)
        self.assertIn('const unitMode = state.mapLevel === "unit";', js)
        self.assertIn('el("mapDotSize").value = String(state.mapDotSize);', js)
        self.assertIn('el("mapDotSizeValue").textContent = String(state.mapDotSize);', js)
        self.assertNotIn('el("mapHotspotsMinLabel").textContent = "Bottom 10%";', js)
        self.assertNotIn('el("mapHotspotsMaxLabel").textContent = "Top 10%";', js)
        self.assertIn('el("mapSliderGrid").classList.toggle("unit-mode", unitMode);', js)
        self.assertIn('const lineWeightControl = el("mapLineWeightControl") || el("mapLineWeight").closest(".map-slider-control");', js)
        self.assertIn('if (lineWeightControl) lineWeightControl.hidden = unitMode;', js)
        self.assertIn('el("mapLineWeight").disabled = unitMode;', js)
        self.assertIn('const dotSizeControl = el("mapDotSizeControl") || el("mapDotSize").closest(".map-slider-control");', js)
        self.assertIn('if (dotSizeControl) dotSizeControl.hidden = !unitMode;', js)
        self.assertIn('el("mapDotSize").disabled = !unitMode;', js)
        self.assertIn('["mapDotSize", "mapDotSize"],', js)
        self.assertIn('const labelHidden = state.mapLevel !== "area";', js)
        self.assertIn('if (labelControl) labelControl.hidden = labelHidden;', js)
        self.assertIn('el("mapLabelSize").disabled = labelHidden;', js)
        self.assertIn('if (data.level === "area") renderMapLabels(data, summaries, hotspotKeys);', js)
        self.assertIn("const MAP_LABEL_MIN_FONT_SIZE = 6;", js)
        self.assertIn("const MAP_LABEL_MAX_FONT_SIZE = 20;", js)
        self.assertIn("function mapLabelFontSize(value = state.mapLabelSize)", js)
        self.assertIn("const sliderValue = Math.max(0, Math.min(10, Math.round(number)));", js)
        self.assertIn("if (sliderValue <= 0) return 0;", js)
        self.assertIn("return MAP_LABEL_MIN_FONT_SIZE + (((sliderValue - 1) / 9) * (MAP_LABEL_MAX_FONT_SIZE - MAP_LABEL_MIN_FONT_SIZE));", js)
        self.assertIn("const fontSize = mapLabelFontSize(state.mapLabelSize);", js)
        self.assertIn('if (data.level !== "area" || !Number.isFinite(fontSize) || fontSize <= 0 || !ukMapLayer) return;', js)
        self.assertIn('(id === "mapLineWeight" && state.mapLevel === "unit")', js)
        self.assertIn('|| (id === "mapDotSize" && state.mapLevel !== "unit")', js)
        self.assertIn('|| (id === "mapLabelSize" && state.mapLevel !== "area")', js)
        self.assertIn("mapSmoothingLevel: 0", js)
        self.assertIn('smoothingLevel: state.mapLevel === "sector" ? state.mapSmoothingLevel : 0', js)
        self.assertIn("formatSmoothingLevel(state.mapSmoothingLevel)", js)
        self.assertIn("function formatSmoothingLevel(value)", js)
        self.assertIn('return level <= 0 ? "None" : `N${level}`;', js)
        self.assertIn('const smoothingHidden = state.mapLevel !== "sector";', js)
        self.assertIn('if (smoothingControl) smoothingControl.hidden = smoothingHidden;', js)
        self.assertIn('el("mapSmoothing").disabled = smoothingHidden;', js)
        self.assertIn('el("mapSmoothing").addEventListener("input"', js)
        self.assertIn('if (state.mapLevel !== "sector") return;', js)
        self.assertIn('captureMapView("smoothing-change");', js)
        self.assertIn("function mapSmoothingApplied(data, row)", js)
        self.assertIn('data?.level === "sector"', js)
        self.assertIn("Raw ${escapeHtml(responseLabel)}", js)
        self.assertIn("Raw ${escapeHtml(weightLabel)}", js)
        self.assertIn("Neighbours: ${escapeHtml(formatNumber(row.smoothing_contributing_sectors))}", js)
        self.assertNotIn("smoothing_contributing_points", js)
        self.assertNotIn("nearest_unit_weighted_numerator", js)
        self.assertIn("if (mapHotspotSelection())", js)
        self.assertNotIn("formatPercentSliderValue(state.mapHotspots)", js)
        self.assertNotIn("Math.abs(fraction)", js)
        self.assertIn("const topMargin = 4;", js)
        self.assertIn("const top = Math.min(Math.max(rawTop, topMargin), maxTop);", js)
        self.assertIn("function mapFloatingPositionFrame()", js)
        self.assertIn("left: rect.left + container.clientLeft,", js)
        self.assertIn("top: rect.top + container.clientTop,", js)
        self.assertIn("panel.style.left = `${left}px`;", js)
        self.assertIn("panel.style.top = `${top}px`;", js)
        self.assertNotIn("panel.style.top = `${Math.round(top)}px`;", js)
        self.assertIn("function mapFloatingTopRightPanelPosition()", js)
        self.assertIn("function mapFloatingTopRightButtonPosition()", js)
        self.assertIn("const position = mapFloatingTopRightPanelPosition();", js)
        self.assertIn("function positionCollapsedMapFloatingControlTopRight()", js)
        self.assertIn("if (state.mapControlCollapsed) {\n      positionCollapsedMapFloatingControlTopRight();", js)
        self.assertIn("const position = mapFloatingTopRightButtonPosition();", js)
        self.assertIn("if (wasCollapsed) panel.classList.remove(\"collapsed\");", js)
        self.assertIn("left: panelPosition.left + buttonOffset.left,", js)
        self.assertIn("top: panelPosition.top + buttonOffset.top,", js)
        self.assertIn("setMapFloatingPosition(position.left, position.top, { updateState: false });", js)
        self.assertNotIn("function mapFloatingExpandedTopRightPosition()", js)
        self.assertIn("left: Math.max(margin, frame.width - panel.offsetWidth - margin),\n      top: topMargin,", js)
        self.assertIn("const dragThreshold = 3;", js)
        self.assertIn("if (!dragMoved && Math.hypot(deltaX, deltaY) < dragThreshold) return;", js)
        self.assertIn("dragMoved = true;\n      state.mapControlMoved = true;", js)
        self.assertIn("state.mapControlMoved = false;\n    state.mapControlPosition = null;", js)
        self.assertIn("positionCollapsedMapFloatingControlTopRight();", js)
        self.assertNotIn("topRight.button", js)
        self.assertNotIn('getPropertyValue("--map-floating-right")', js)
        self.assertIn("positionMapFloatingControlTopRight();", js)
        self.assertIn("function initialMobileLayoutActive()", js)
        self.assertIn("return window.innerWidth <= MOBILE_LAYOUT_MAX_WIDTH;", js)
        self.assertIn("const startedInMobileLayout = initialMobileLayoutActive();", js)
        self.assertIn("mapControlCollapsed: startedInMobileLayout", js)
        self.assertIn("mapControlCollapsedPosition: null", js)
        self.assertIn("mapLegendCollapsed: startedInMobileLayout", js)
        self.assertIn("function syncMapFloatingControlCollapsedState()", js)
        self.assertIn('el("mapFloatingControl").classList.toggle("collapsed", Boolean(state.mapControlCollapsed));', js)
        self.assertIn("const MAP_CONTROL_COLLAPSED_ICON", js)
        self.assertIn("function toggleMapFloatingControlCollapsed()", js)
        self.assertIn('button.setAttribute("aria-expanded", String(!collapsed));', js)
        self.assertIn('el("mapControlReset").addEventListener("click", toggleMapFloatingControlCollapsed);', js)
        self.assertIn(".map-floating-control:not(.collapsed) .map-control-reset {\n        transform: translate(5px, -5px);", css)
        self.assertNotIn("function resetMapFloatingControlPosition()", js)
        self.assertIn('label: "Aerial"', js)
        self.assertNotIn('label: "Satellite"', js)
        self.assertIn('type="radio" name="baseMap"', html)
        self.assertIn('type="radio" name="mapLevel" value="area"', html)
        self.assertIn('type="radio" name="mapLevel" value="sector"', html)
        self.assertIn('type="radio" name="mapLevel" value="unit"', html)
        self.assertNotIn('name="mapOverlay"', js)
        self.assertIn('target.name === "mapLevel"', js)
        self.assertIn('el("mapBaseLayerTiles").addEventListener("change", handleMapLayerControlChange);', js)
        self.assertIn('el("mapLevelTiles").addEventListener("change", handleMapLayerControlChange);', js)
        self.assertIn('container.querySelector("#mapZoomIn").addEventListener("click", () => zoomMapBy(Number(ukMap?.options?.zoomDelta) || 1));', js)
        self.assertIn('container.querySelector("#mapFitUk").addEventListener("click", () => fitMapToLayer());', js)
        self.assertIn("function zoomMapToLondon()", js)
        self.assertNotIn("map-layer-control", js)
        self.assertNotIn("map-layer-control", css)
        self.assertNotIn("map-place-control", js)
        self.assertNotIn("map-place-control", css)
        self.assertNotIn("leaflet-control-zoom", css)
        self.assertIn(".uk-map .leaflet-control-attribution", css)
        self.assertIn("font-size: 10px;", css)

    def test_sidebar_toggle_contract(self) -> None:
        css = self.app_css_contract()
        js = self.app_js_contract()

        self.assertIn("--sidebar-bg: #dce4ef;", css)
        self.assertIn("--sidebar-bg: #24334b;", css)
        self.assertIn("--sidebar-collapsed-width: 50px;", css)
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
        self.assertIn("body.single-tool-mode.sidebar-collapsed .shell {\n        grid-template-columns: minmax(0, 1fr);", css)
        self.assertIn("body.sidebar-collapsed .sidebar-resizer {\n        display: none;", css)
        self.assertIn("body.single-tool-mode.sidebar-collapsed #appSidebar {\n        display: none;", css)
        self.assertIn("body.sidebar-collapsed #appSidebar {\n        grid-template-columns: var(--sidebar-collapsed-width);", css)
        self.assertIn("body.sidebar-collapsed .sidebar-control-pane {\n        display: none;", css)
        self.assertIn("#appSidebar {\n        background: var(--sidebar-bg);", css)
        self.assertNotIn("body.sidebar-collapsed aside {\n        align-items: center;", css)
        self.assertNotIn("body.sidebar-collapsed aside > .section:not(#toolSelectorSection)", css)
        self.assertNotIn("aside {\n        background: var(--sidebar-bg);", css)
        self.assertIn("grid-template-columns: var(--sidebar-collapsed-width) minmax(0, 1fr);", css)
        self.assertIn(".shell {\n        display: grid;\n        grid-template-columns: var(--sidebar-width, 400px) 3px minmax(0, 1fr);", css)
        self.assertIn(".sidebar-control-pane {\n        border-left: 1px solid var(--line);\n        display: flex;\n        flex-direction: column;\n        grid-column: 2;", css)
        self.assertIn("body.single-tool-mode .sidebar-control-pane {\n        border-left: 0;\n        grid-column: 1;", css)
        self.assertIn(".sidebar-resizer {\n        background: linear-gradient(to right, var(--line), transparent);\n        cursor: col-resize;\n        min-width: 3px;", css)
        self.assertIn(".tool-option {\n        width: 36px;", css)
        self.assertIn("height: 36px;\n        min-height: 36px;\n        flex: 0 0 36px;\n        border: 0;", css)
        self.assertIn("border-radius: 0;\n        background: transparent;", css)
        self.assertIn("justify-content: center;\n        padding: 0;", css)
        self.assertIn(".tool-icon {\n        width: 30px;\n        height: 30px;", css)
        self.assertIn(".tool-icon svg {\n        width: 28px;\n        height: 28px;", css)
        self.assertIn(".tool-icon img {\n        width: 30px;\n        height: 30px;", css)
        self.assertIn(".tool-label {\n        position: absolute;", css)
        self.assertIn(".tool-button-tooltip {\n        position: fixed;\n        z-index: 1200;", css)
        self.assertIn(".tool-button-tooltip[hidden] {\n        display: none;", css)
        self.assertIn("body.dark .tool-button-tooltip {\n        background: color-mix(in srgb, white 92%, var(--panel));", css)
        self.assertIn(".tool-selector {\n        display: grid;", css)
        self.assertNotIn("tool-selector-toggle", css)
        self.assertNotIn("tool-selector-collapsed", css)
        self.assertNotIn("body.sidebar-collapsed aside,\n      body.sidebar-collapsed .sidebar-resizer", css)
        self.assertIn("function initialSidebarVisible()", js)
        self.assertIn('return !document.body.classList.contains("sidebar-collapsed");', js)
        self.assertIn("sidebarVisible: initialSidebarVisible()", js)
        self.assertNotIn("sidebarVisible: true", js)
        self.assertIn('document.body.classList.toggle("single-tool-mode", enabledTools.length === 1);', js)
        self.assertIn('document.body.classList.toggle("sidebar-collapsed", !state.sidebarVisible)', js)
        self.assertIn('el("appSidebar").removeAttribute("aria-hidden");', js)
        self.assertNotIn('el("appSidebar").setAttribute("aria-hidden", String(!state.sidebarVisible));', js)
        self.assertIn("function handleToolClick(tool)", js)
        self.assertIn("if (state.tool === tool) {\n          setSidebarVisible(!state.sidebarVisible);", js)
        self.assertNotIn("if (state.tool === tool) return;", js)
        self.assertIn("function scheduleActiveToolResize({ hard = true } = {})", js)
        self.assertIn("function resizeActiveTool({ hard = true } = {})", js)
        self.assertIn("glmTool.resize();", js)
        self.assertIn("scheduleActiveToolResize({ hard: true });", js)
        self.assertNotIn("requestAnimationFrame(resizeActiveTool);", js)
        self.assertIn('el("visualArea").classList.remove("startup-mode");', js)
        self.assertIn('el("profileTool").addEventListener("click", () => handleToolClick("column_profile"));', js)
        self.assertIn('el("datasetViewerTool").addEventListener("click", () => handleToolClick("dataset_viewer"));', js)
        self.assertIn('el("lineBarTool").addEventListener("click", () => handleToolClick("line_bar"));', js)
        self.assertIn('el("histogramTool").addEventListener("click", () => handleToolClick("histogram"));', js)
        self.assertIn('el("ukMapTool").addEventListener("click", () => handleToolClick("uk_map"));', js)
        self.assertNotIn('el("profileTool").addEventListener("click", () => setTool("column_profile"));', js)
        self.assertIn('el("sidebarToggleBtn").addEventListener("click", () => setSidebarVisible(!state.sidebarVisible))', js)
        self.assertIn('const label = state.sidebarVisible ? "Collapse sidebar" : "Expand sidebar";', js)
        self.assertIn('button.setAttribute("aria-expanded", String(state.sidebarVisible));', js)

    def test_tool_selector_aligns_with_main_toolbar(self) -> None:
        css = self.app_css_contract()

        self.assertIn(".tool-selector-section {\n        align-items: center;\n        display: flex;\n        flex-direction: column;", css)
        self.assertIn("grid-column: 1;\n        margin-bottom: 0;\n        min-height: 0;", css)
        self.assertIn("padding: 8px 0;\n        width: var(--sidebar-collapsed-width);", css)
        self.assertIn(".tool-selector {\n        display: grid;\n        align-content: start;\n        flex: 1 1 auto;\n        grid-template-columns: 1fr;", css)
        self.assertIn("min-width: 0;\n        width: 100%;", css)
        self.assertIn("border-radius: 4px;\n        background: var(--sidebar-bg);", css)
        self.assertIn("overflow: visible;", css)
        self.assertIn(".tool-option {\n        width: 36px;\n        height: 36px;\n        min-height: 36px;\n        flex: 0 0 36px;\n        border: 0;\n        border-radius: 0;\n        background: transparent;\n        color: var(--muted);", css)
        self.assertIn("opacity: 1;\n        cursor: pointer;", css)
        self.assertIn("font-weight: 650;\n        justify-self: center;", css)
        self.assertIn(".tool-option.active {\n        background: transparent;\n        color: var(--accent);\n        opacity: 1;", css)
        self.assertIn(".tool-option:not(.active):hover,\n      .tool-option:not(.active):focus-visible {\n        color: var(--accent);", css)
        self.assertIn(".tool-option.active::before {\n        content: \"\";\n        position: absolute;\n        left: -7px;\n        top: -6px;\n        bottom: -6px;\n        width: 3px;\n        background: var(--accent);", css)
        self.assertIn("body.dark .tool-option.active {\n        color: var(--accent);", css)
        self.assertIn(".tool-label {\n        position: absolute;\n        width: 1px;", css)
        self.assertIn("overflow: hidden;\n        pointer-events: auto;\n        text-overflow: ellipsis;\n        white-space: nowrap;", css)

    def test_mobile_layout_static_contract(self) -> None:
        html = self.assert_no_store("/")[1].decode("utf-8")
        css = self.app_css_contract()
        js = self.app_js_contract()

        self.assertIn('window.matchMedia("(max-width: 640px)").matches', html)
        self.assertIn('document.body.classList.add("sidebar-collapsed");', html)
        self.assertIn("@media (max-width: 640px) {\n        header {", css)
        self.assertIn(
            ".shell,\n"
            "        body.sidebar-collapsed .shell {\n"
            "          grid-template-columns: var(--sidebar-collapsed-width) minmax(0, 1fr) !important;",
            css,
        )
        self.assertIn("box-shadow: 12px 0 28px rgb(15 23 42 / 18%);", css)
        self.assertIn("body.sidebar-collapsed #appSidebar {\n          box-shadow: none;", css)
        self.assertIn("z-index: 1100;", css)
        self.assertIn(".sidebar-resizer {\n          display: none;", css)
        self.assertIn(
            "main {\n"
            "          padding: 8px;\n"
            "          position: relative;\n"
            "          z-index: 0;",
            css,
        )
        self.assertIn(
            ".filter-footer {\n"
            "          align-items: stretch;\n"
            "          grid-template-columns: minmax(0, 1fr);",
            css,
        )
        self.assertIn(
            ".visual-area {\n"
            "          grid-template-columns: minmax(0, 1fr) !important;\n"
            "          grid-template-rows: auto minmax(0, 1fr);",
            css,
        )
        self.assertIn(
            ".visual-area.dataset-viewer-mode,\n"
            "        .visual-area.map-mode,\n"
            "        .visual-area.profile-mode,\n"
            "        .visual-area.histogram-mode,\n"
            "        .visual-area.model-mode,\n"
            "        .visual-area.specs-mode,\n"
            "        .visual-area.line-bar-side-controls-collapsed {\n"
            "          grid-template-columns: minmax(0, 1fr) !important;\n"
            "          grid-template-rows: minmax(0, 1fr);",
            css,
        )
        self.assertIn(".chart-controls-resizer {\n          display: none !important;", css)
        self.assertIn(
            ".chart-side-controls {\n"
            "          grid-template-rows: minmax(0, 1fr) 18px minmax(0, 1fr);\n"
            "          height: min(34dvh, 280px);",
            css,
        )
        self.assertIn(
            ".chart-side-controls.chart-expected-collapsed {\n"
            "          grid-template-rows: minmax(0, 1fr) 18px;",
            css,
        )

        self.assertIn("const MOBILE_LAYOUT_MAX_WIDTH = 640;", js)
        self.assertIn("let mobileLayoutActive = null;", js)
        self.assertIn("function syncMobileSidebarLayout({ initial = false } = {})", js)
        self.assertIn("const enteredMobile = mobile && mobileLayoutActive !== true;", js)
        self.assertIn("if ((initial || enteredMobile) && mobile && state.sidebarVisible)", js)
        self.assertIn("syncMobileSidebarLayout();\n          scheduleDatasetMetaCompactCheck();", js)
        self.assertIn("syncHeaderButtons();\n        syncSidebarToggleButton();", js)
        self.assertIn("setTool(state.tool, false);\n          syncMobileSidebarLayout({ initial: true });", js)

    def test_dataset_viewer_tool_static_assets_are_registered(self) -> None:
        _, html_body = self.assert_no_store("/")
        css = self.app_css_contract()
        app_css = self.assert_no_store("/static/app.css")[1].decode("utf-8")
        html = html_body.decode("utf-8")
        js = self.app_js_contract()
        main_js = self.assert_no_store("/static/app/main.js")[1].decode("utf-8")
        dataset_js = self.assert_no_store("/static/app/dataset-viewer-tool.js")[1].decode("utf-8")
        shell_css = self.assert_no_store("/static/styles/shell.css")[1].decode("utf-8")
        dataset_css = self.assert_no_store("/static/styles/dataset-viewer.css")[1].decode("utf-8")

        self.assertNotIn('import { createDatasetViewerTool } from "./dataset-viewer-tool.js";', main_js)
        self.assertIn('import("./dataset-viewer-tool.js")', main_js)
        self.assertNotIn('styles/dataset-viewer.css', app_css)
        self.assertIn('link.href = "/static/styles/dataset-viewer.css";', dataset_js)
        self.assertIn("export function createDatasetViewerTool", js)
        self.assertIn('dataset_viewer: "Dataset render"', js)
        self.assertIn('tool: ""', js)
        self.assertIn("function enabledToolIds()", js)
        self.assertIn('const requested = locationParams.get("tool");', js)
        self.assertIn('if (requested && toolEnabled(requested)) return requested;', js)
        self.assertIn('if (requestedDefault("line_bar_favourite") && toolEnabled("line_bar")) return "line_bar";', js)
        self.assertIn('const firstEnabled = enabledToolIds()[0] || "";', js)
        self.assertIn("if (firstEnabled && toolEnabled(firstEnabled)) return firstEnabled;", js)
        self.assertIn('api("/api/dataset-viewer/table"', js)
        self.assertIn('id="datasetViewerWrap" class="dataset-viewer-wrap hidden"', html)
        self.assertIn('id="datasetViewerSearch"', js)
        self.assertIn('placeholder="Select columns, separate with commas"', dataset_js)
        self.assertIn('id="datasetViewerTranspose"', js)
        self.assertIn('id="datasetViewerAlphabeticalColumns"', js)
        self.assertIn('id="datasetViewerMeta"', dataset_js)
        self.assertIn('renderDatasetViewerState("Reading data...");', dataset_js)
        self.assertIn('renderDatasetViewerState("Preparing table...", { transposed: Boolean(state.datasetViewerTranspose) });', dataset_js)
        self.assertIn("function setDatasetViewerToolbarHidden(hidden)", dataset_js)
        self.assertIn("function renderDatasetViewerState(message, { error = false, transposed = false } = {})", dataset_js)
        self.assertIn("setDatasetViewerToolbarHidden(true);", dataset_js)
        self.assertIn("setDatasetViewerToolbarHidden(false);", dataset_js)
        self.assertIn("DATASET_VIEWER_STATE_INLINE_STYLE", dataset_js)
        self.assertNotIn("Loading dataset...", dataset_js)
        self.assertIn("function attachDatasetViewerMeta()", dataset_js)
        self.assertIn('if (groupMeta.parentElement !== meta) meta.append(groupMeta);', dataset_js)
        self.assertIn('if (filter.parentElement !== meta) meta.append(filter);', dataset_js)
        self.assertNotIn('id="datasetViewerResetSort"', js)
        self.assertNotIn("function resetSort()", dataset_js)
        self.assertNotIn('id="datasetViewerCopySelected"', js)
        self.assertNotIn("datasetViewerCopySelected", dataset_js)
        self.assertIn("selectableRows: true", js)
        self.assertIn("autoResize: false", dataset_js)
        self.assertIn('renderHorizontal: "virtual"', dataset_js)
        self.assertIn('layout: "fitData"', dataset_js)
        self.assertNotIn('layout: "fitDataStretch"', dataset_js)
        self.assertIn('headerSortClickElement: "icon"', dataset_js)
        self.assertIn("headerSortTristate: true", dataset_js)
        self.assertIn("function datasetViewerColumnWidth(column)", dataset_js)
        self.assertIn('headerHozAlign: "left"', dataset_js)
        self.assertNotIn('headerHozAlign: column.kind === "numeric"', dataset_js)
        self.assertIn('let renderedWidthMode = "";', dataset_js)
        self.assertIn("let renderedPinnedColumns = null;", dataset_js)
        self.assertIn("let normalColumnWidths = new Map();", dataset_js)
        self.assertIn("let transposedColumnWidths = new Map();", dataset_js)
        self.assertIn('datasetTable.on("columnResized", rememberNormalColumnWidth);', dataset_js)
        self.assertIn('datasetTable.on("columnResized", rememberTransposedColumnWidth);', dataset_js)
        self.assertIn("function snapshotDatasetViewerColumnWidths()", dataset_js)
        self.assertIn("function datasetViewerColumnWidthWithSaved(mode, field, defaults)", dataset_js)
        self.assertIn("{ width: 112, minWidth: 72 }", dataset_js)
        self.assertIn("{ width: 144, minWidth: 104 }", dataset_js)
        self.assertIn("{ width: 180, minWidth: 96 }", dataset_js)
        self.assertIn("...datasetViewerColumnWidth(column)", dataset_js)
        self.assertNotIn("maxInitialWidth", dataset_js)
        self.assertIn("resizable: true", dataset_js)
        self.assertIn("function orderedDatasetViewerColumns(data)", dataset_js)
        self.assertIn("function datasetViewerPinnedColumnFields()", dataset_js)
        self.assertIn("function datasetViewerPinnedColumnSet()", dataset_js)
        self.assertIn("function isDatasetViewerColumnPinned(field)", dataset_js)
        self.assertIn("function datasetViewerPinnedColumnsKey()", dataset_js)
        self.assertIn("function datasetViewerPinnedColumnNames()", dataset_js)
        self.assertIn("function pruneDatasetViewerPinnedColumns(columns = currentDatasetColumns)", dataset_js)
        self.assertIn("function compareDatasetViewerColumnNames(left, right)", dataset_js)
        self.assertIn("function compareDatasetViewerValues(left, right)", dataset_js)
        self.assertIn("left.index - right.index", dataset_js)
        self.assertIn("const pinnedColumns = indexedColumns", dataset_js)
        self.assertIn("return [...pinnedColumns, ...unpinnedColumns].map((entry) => entry.column);", dataset_js)
        self.assertIn("renderedAlphabeticalColumns", dataset_js)
        self.assertIn("function renderTransposedGrid(token, requestKey)", dataset_js)
        self.assertNotIn("const frozenRows = transposedFrozenRowCount(currentRenderedRows);", dataset_js)
        self.assertNotIn("frozenRows", dataset_js)
        self.assertIn("selectableRows: false", dataset_js)
        self.assertIn("headerSort: false", dataset_js)
        self.assertIn("formatter: formatTransposedCell", dataset_js)
        self.assertIn('renderedWidthMode = "normal";', dataset_js)
        self.assertIn('renderedWidthMode = "transposed";', dataset_js)
        self.assertIn("...datasetViewerTransposedColumnWidth(\"__field\", { width: 300, minWidth: 170 })", dataset_js)
        self.assertNotIn("frozen: hasDatasetViewerPinnedColumns()", dataset_js)
        self.assertIn("minWidth: 170", dataset_js)
        self.assertIn("...datasetViewerTransposedColumnWidth(field, { width: 150, minWidth: 72 })", dataset_js)
        self.assertIn("minWidth: 72", dataset_js)
        self.assertIn("rowFormatter: formatTransposedRow", dataset_js)
        self.assertIn("function formatTransposedRow(row)", dataset_js)
        self.assertIn("function formatTransposedCell(cell)", dataset_js)
        self.assertIn("function datasetViewerPinnedFieldHtml(rowData)", dataset_js)
        self.assertIn("function datasetViewerPinIndicator(field)", dataset_js)
        self.assertIn('class="dataset-viewer-pin-indicator"', dataset_js)
        self.assertIn("&#128204;", dataset_js)
        self.assertIn("function transposedCellValue(rowData, field", dataset_js)
        self.assertIn("function datasetViewerSearchTerms()", dataset_js)
        self.assertIn(".split(\",\")", dataset_js)
        self.assertIn("const search = normalColumnsForSearch(datasetViewerSearchTerms(), sourceColumns);", dataset_js)
        self.assertIn("function normalColumnsForSearch(terms, sourceColumns = currentDatasetColumns)", dataset_js)
        self.assertIn("const visibleDatasetColumns = orderedNormalColumnsForSearch(terms, sourceColumns);", dataset_js)
        self.assertIn("const hiddenDatasetColumns = sourceColumns.filter((column) => !visibleFields.has(column.field));", dataset_js)
        self.assertIn("const renderedColumns = terms.length ? [...visibleDatasetColumns, ...hiddenDatasetColumns] : sourceColumns;", dataset_js)
        self.assertIn("function normalColumnDefinition(column, visibleFields)", dataset_js)
        self.assertIn("visible: visibleFields.has(column.field)", dataset_js)
        self.assertNotIn("frozen: pinned", dataset_js)
        self.assertNotIn("hasDatasetViewerPinnedColumns", dataset_js)
        self.assertIn("function datasetViewerColumnMatchesSearch(column, terms)", dataset_js)
        self.assertIn("function datasetViewerColumnSearchTermIndex(column, terms)", dataset_js)
        self.assertIn("function orderedNormalColumnsForSearch(terms, sourceColumns = currentDatasetColumns)", dataset_js)
        self.assertIn("const pinnedColumns = sourceColumns.filter((column) => pinnedFields.has(column.field));", dataset_js)
        self.assertIn("if (pinnedFields.has(column.field)) return;", dataset_js)
        self.assertIn("const groupedColumns = terms.map(() => []);", dataset_js)
        self.assertIn("groupedColumns[matchIndex].push(column);", dataset_js)
        self.assertIn("return [...pinnedColumns, ...groupedColumns.flat()];", dataset_js)
        self.assertIn("function syncNormalColumnVisibilityAndOrder(visibleColumns, visibleFields)", dataset_js)
        self.assertIn("withDatasetViewerRedrawBlocked(() => {", dataset_js)
        self.assertIn("function withDatasetViewerRedrawBlocked(callback)", dataset_js)
        self.assertIn("typeof datasetTable.blockRedraw === \"function\"", dataset_js)
        self.assertIn("typeof datasetTable.restoreRedraw === \"function\"", dataset_js)
        self.assertIn("datasetTable.blockRedraw();", dataset_js)
        self.assertIn("} finally {\n      datasetTable.restoreRedraw();", dataset_js)
        self.assertIn("function reorderNormalColumns(visibleColumns, componentByField, tableColumns, visibleFields)", dataset_js)
        self.assertIn("const stableFields = longestInOrderFields(currentVisibleFields, desiredFields);", dataset_js)
        self.assertIn("function longestInOrderFields(currentFields, desiredFields)", dataset_js)
        self.assertIn("column.move(nextColumn, false);", dataset_js)
        self.assertIn("column.move(previousColumn, true);", dataset_js)
        self.assertIn("function applySearch({ mark = true } = {})", dataset_js)
        self.assertIn("function applyNormalColumnSearch(terms, { mark = true } = {})", dataset_js)
        self.assertIn("const search = normalColumnsForSearch(terms);", dataset_js)
        self.assertIn("currentVisibleDatasetColumns = search.visibleDatasetColumns;", dataset_js)
        self.assertIn("if (!replaceNormalColumns(search))", dataset_js)
        self.assertIn("const NORMAL_SEARCH_REPLACE_MOVE_THRESHOLD = 16;", dataset_js)
        self.assertIn("function replaceNormalColumns(search)", dataset_js)
        self.assertIn("function normalSearchNeedsColumnReplacement(visibleColumns, visibleFields)", dataset_js)
        self.assertIn("return desiredFields.length - stableFields.size > NORMAL_SEARCH_REPLACE_MOVE_THRESHOLD;", dataset_js)
        self.assertIn("snapshotDatasetViewerColumnWidths();", dataset_js)
        self.assertIn("datasetTable.setColumns(search.columns);", dataset_js)
        self.assertIn("function normalTableSorters()", dataset_js)
        self.assertIn("function restoreNormalTableSorters(sorters)", dataset_js)
        normal_search = re.search(
            r"function applyNormalColumnSearch\(terms, \{ mark = true \} = \{\}\) \{(?P<body>.*?)\n  \}\n\n  function transposedRowSearchTermIndex",
            dataset_js,
            re.S,
        )
        self.assertIsNotNone(normal_search)
        self.assertNotIn("clearFilter", normal_search.group("body"))
        self.assertIn("function transposedRowSearchTermIndex(row, terms)", dataset_js)
        self.assertIn("function orderedTransposedRowsForSearch(terms, sourceRows = currentRows)", dataset_js)
        self.assertIn("const pinnedRows = sourceRows.filter((row) => isDatasetViewerColumnPinned(row?.__column_field));", dataset_js)
        self.assertIn("if (isDatasetViewerColumnPinned(row?.__column_field)) return;", dataset_js)
        self.assertIn("const groupedRows = terms.map(() => []);", dataset_js)
        self.assertIn("groupedRows[matchIndex].push(row);", dataset_js)
        self.assertIn("return [...pinnedRows, ...groupedRows.flat()];", dataset_js)
        self.assertNotIn("transposedFrozenRowCount", dataset_js)
        self.assertIn("function applyTransposedSearch(terms, { mark = true } = {})", dataset_js)
        self.assertIn("const rows = orderedTransposedRowsForSearch(terms);", dataset_js)
        self.assertIn("currentRenderedRows = rows;", dataset_js)
        self.assertIn("replaceTransposedRows(rows);", dataset_js)
        self.assertIn("function replaceTransposedRows(rows)", dataset_js)
        self.assertIn("datasetTable.replaceData(rows)", dataset_js)
        self.assertIn("function datasetColumnsFromTransposedRows(rows, columns = currentDatasetColumns)", dataset_js)
        self.assertIn("currentVisibleDatasetColumns = datasetColumnsFromTransposedRows(rows);", dataset_js)
        self.assertNotIn("transposedColumnMatchesSearch", dataset_js)
        self.assertNotIn("datasetTable.setFilter((row) => transposedColumnMatchesSearch(row, terms));", dataset_js)
        self.assertIn("function reconcileRenderedSearch(token, requestKey, syncSelection)", dataset_js)
        self.assertIn("reconcileRenderedSearch(token, requestKey, syncNormalRenderedSelection);", dataset_js)
        self.assertIn("reconcileRenderedSearch(token, requestKey, syncTransposedRenderedSelection);", dataset_js)
        self.assertIn("applySearch({ mark: false });", dataset_js)
        self.assertIn("markRendered(requestKey);", dataset_js)
        self.assertNotIn("fields.some((field) => formatCellValue(row[field])", dataset_js)
        self.assertIn("function syncTransposedVisibleColumnsFromActiveRows()", dataset_js)
        self.assertIn("function datasetViewerCellValue(cell)", dataset_js)
        self.assertNotIn("sourceRows.forEach((sourceRow, rowIndex)", dataset_js)
        self.assertNotIn("row[`r${rowIndex}`]", dataset_js)
        self.assertNotIn("dataset-viewer-transposed-table", dataset_js)
        self.assertNotIn("dataset-viewer-transposed-hover", dataset_js)
        self.assertNotIn("function wireTransposedHover(target)", dataset_js)
        self.assertIn("selectedColumnFields", dataset_js)
        self.assertIn("selectedRowIds", dataset_js)
        self.assertIn("selectionAxis", dataset_js)
        self.assertIn("function toggleColumnSelection(field)", dataset_js)
        self.assertIn("function toggleRowSelection(rowId)", dataset_js)
        self.assertIn("function syncTransposedRenderedSelection()", dataset_js)
        self.assertIn(".tabulator-row[data-dataset-viewer-column-field]", dataset_js)
        self.assertIn(".tabulator-col[tabulator-field], .tabulator-cell[tabulator-field]", dataset_js)
        self.assertIn("function clearTransposedColumnSelectionClasses", dataset_js)
        self.assertIn("function clearTransposedRowSelectionClasses", dataset_js)
        self.assertIn("function datasetViewerCellContextMenu()", dataset_js)
        self.assertIn('menu.id = "datasetViewerCellContextMenu";', dataset_js)
        self.assertIn("function selectedCopyLabel()", dataset_js)
        self.assertIn("const screenAxis = state.datasetViewerTranspose", dataset_js)
        self.assertIn("Copy selected row", dataset_js)
        self.assertIn("Copy selected column", dataset_js)
        self.assertIn("Copy cell to clipboard", dataset_js)
        self.assertIn("Pin column", dataset_js)
        self.assertIn("Unpin column", dataset_js)
        context_menu = re.search(
            r"function handleDatasetViewerGridContextMenu\(event\) \{(?P<body>.*?)\n  \}\n\n  function datasetViewerContextColumnField",
            dataset_js,
            re.S,
        )
        self.assertIsNotNone(context_menu)
        context_menu_body = context_menu.group("body")
        self.assertLess(context_menu_body.find('mode: "toggle-pin"'), context_menu_body.find('mode: "cell"'))
        self.assertLess(context_menu_body.find('actions.push({ divider: true });'), context_menu_body.find('mode: "cell"'))
        self.assertIn("Copy displayed table to clipboard", dataset_js)
        self.assertIn("function copyDisplayedTable()", dataset_js)
        self.assertIn("function normalSelectionToCsv()", dataset_js)
        self.assertIn("function transposedSelectionToCsv()", dataset_js)
        self.assertIn("state.datasetViewerTranspose ? transposedSelectionToCsv() : normalSelectionToCsv()", dataset_js)
        self.assertIn("function activeTransposedRowsForCopy()", dataset_js)
        self.assertIn("function transposedRowsToCsv(rows, columns)", dataset_js)
        self.assertIn('button.dataset.copyMode === "displayed-table"', dataset_js)
        self.assertIn("Clear selection", dataset_js)
        self.assertIn("function clearDatasetViewerSelection()", dataset_js)
        self.assertIn('button.dataset.copyMode === "clear-selection"', dataset_js)
        self.assertIn('button.dataset.copyMode === "toggle-pin"', dataset_js)
        self.assertIn("function datasetViewerContextColumnField(event, grid)", dataset_js)
        self.assertIn("function validDatasetViewerColumnField(field)", dataset_js)
        self.assertIn("function toggleDatasetViewerPinnedColumn(field)", dataset_js)
        self.assertIn('divider.setAttribute("role", "separator");', dataset_js)
        self.assertIn("button.dataset.copyMode", dataset_js)
        self.assertIn("button.dataset.copyValue", dataset_js)
        self.assertIn("button.dataset.columnField", dataset_js)
        self.assertIn("openDatasetViewerContextMenu(event, actions)", dataset_js)
        self.assertIn('{ mode: "selection", label: selectionLabel }', dataset_js)
        self.assertIn('actions.push({ mode: "displayed-table", label: "Copy displayed table to clipboard" });', dataset_js)
        self.assertIn('actions.push({ mode: "clear-selection", label: "Clear selection" });', dataset_js)
        self.assertNotIn("menu.dataset.copyMode", dataset_js)
        self.assertNotIn("menu.dataset.copyValue", dataset_js)
        self.assertNotIn("headerTooltip", dataset_js)
        self.assertNotIn('title="${escapeHtml(column.name)}"', dataset_js)
        self.assertNotIn('title="${escapeHtml(column.headerTooltip || title)}"', dataset_js)
        self.assertIn("copyDatasetViewerContextValue", dataset_js)
        self.assertIn(": rowsToCsv(rowsForColumnCopy(), currentVisibleDatasetColumns);", dataset_js)
        self.assertIn("function cacheIsRendered(cache)", dataset_js)
        self.assertIn("requestAnimationFrame(() => resize());", dataset_js)
        self.assertIn("let resizeFrame = null;", dataset_js)
        self.assertIn("function resize({ hard = true } = {})", dataset_js)
        self.assertIn("datasetTable.redraw(shouldHard);", dataset_js)
        self.assertNotIn("datasetTable.redraw(true);", dataset_js)
        self.assertIn("function scheduleActiveToolResize({ hard = true } = {})", main_js)
        self.assertIn("resizeActiveTool({ hard: shouldHard });", main_js)
        self.assertIn("if (!hard) return;", main_js)
        self.assertIn("setSidebarWidth(event.clientX - bounds.left, { hard: false });", main_js)
        self.assertIn("scheduleActiveToolResize({ hard: true });", main_js)
        self.assertIn("datasetViewerSearch: \"\"", js)
        self.assertIn("datasetViewerTranspose: false", js)
        self.assertIn("datasetViewerAlphabeticalColumns: false", js)
        self.assertIn("datasetViewerPinnedColumns: []", js)
        self.assertIn("datasetViewerColumnCount: null", js)
        self.assertIn("function syncDatasetViewerMeta()", main_js)
        self.assertIn('el("datasetViewerFilterText").textContent = label;', main_js)
        self.assertIn('el("datasetViewerGroupMeta").classList.toggle("hidden", tool !== "dataset_viewer");', main_js)
        self.assertIn('el("datasetViewerFilter").classList.toggle("hidden", tool !== "dataset_viewer");', main_js)
        self.assertIn("syncDatasetViewerMeta = () => {},", dataset_js)
        self.assertIn("state.datasetViewerColumnCount = Array.isArray(data?.columns) ? data.columns.length : null;", dataset_js)
        self.assertIn("syncDatasetViewerMeta();", dataset_js)
        self.assertIn("loadedRows > MAX_ROWS", dataset_js)
        self.assertIn('const displayMeta = data?.has_more ? `First ${shownMeta}` : shownMeta;', dataset_js)
        self.assertNotIn("more available", dataset_js)
        self.assertIn('return `${displayMeta} · ${pinnedNames.join(", ")} pinned`;', dataset_js)
        self.assertIn("shown", dataset_js)
        self.assertNotIn("setGroupMeta", dataset_js)
        self.assertNotIn("setFilterRowMeta", dataset_js)
        self.assertIn('if (tool === "dataset_viewer") return;', js)
        self.assertIn(".visual-area.dataset-viewer-mode,\n      .visual-area.profile-mode {\n        grid-template-columns: minmax(0, 1fr);", shell_css)
        self.assertIn(".visual-area.dataset-viewer-mode .workspace,\n      .visual-area.profile-mode .workspace {\n        background: transparent;\n        border: 0;", shell_css)
        self.assertIn(".visual-area.dataset-viewer-mode .workspace-messages,\n      .visual-area.profile-mode .workspace-messages {\n        display: none;", shell_css)
        self.assertNotIn(".visual-area.dataset-viewer-mode", dataset_css)
        self.assertNotIn("padding-top: 36px;", dataset_css)
        self.assertIn("flex: 0 1 312px;", dataset_css)
        self.assertIn(".dataset-viewer-meta {", dataset_css)
        self.assertIn("flex-direction: column;\n        gap: 4px;", dataset_css)
        self.assertIn("max-width: min(440px, 46%);", dataset_css)
        self.assertIn(".dataset-viewer-meta .workspace-meta,\n      .dataset-viewer-meta #datasetViewerFilter {\n        max-width: 100%;\n        overflow: hidden;\n        text-overflow: ellipsis;\n        white-space: nowrap;", dataset_css)
        self.assertIn(".dataset-viewer-grid .tabulator-row.tabulator-selected", css)
        self.assertIn(".dataset-viewer-header-label", css)
        self.assertIn(".dataset-viewer-header-text,\n      .dataset-viewer-pinned-field-text", dataset_css)
        self.assertIn(".dataset-viewer-pin-indicator", dataset_css)
        self.assertIn(".dataset-viewer-pinned-field-label", dataset_css)
        self.assertIn(".dataset-viewer-column-selected", css)
        self.assertNotIn(".dataset-viewer-grid.tabulator .tabulator-header .tabulator-frozen", dataset_css)
        self.assertNotIn(".dataset-viewer-grid.tabulator .tabulator-row .tabulator-cell.tabulator-frozen", dataset_css)
        self.assertNotIn(".tabulator-frozen-rows-holder", dataset_css)
        self.assertIn(".dataset-viewer-grid-transposed.tabulator", css)
        self.assertIn('.dataset-viewer-grid-transposed .tabulator-row .tabulator-cell[tabulator-field="__field"]', css)
        self.assertNotIn(".dataset-viewer-transposed-table", css)
        self.assertNotIn(".dataset-viewer-transposed-hover", css)
        self.assertIn(".dataset-viewer-grid.tabulator .tabulator-header .tabulator-col {\n        font-size: 11px;\n        justify-content: flex-start;", dataset_css)
        self.assertIn(".dataset-viewer-grid.tabulator .tabulator-header .tabulator-col .tabulator-col-content {\n        align-items: center;\n        display: flex;\n        justify-content: flex-start;", dataset_css)
        self.assertIn("text-align: left;\n        text-overflow: ellipsis;\n        white-space: nowrap;\n        width: 100%;", dataset_css)
        self.assertIn(".dataset-viewer-transposed-header-content {\n        align-items: center;\n        display: flex;\n        gap: 4px;\n        justify-content: flex-start;", dataset_css)
        self.assertIn(".tabulator-col[aria-sort=\"ascending\"]", css)
        self.assertIn(".tabulator-col[aria-sort=\"descending\"]", css)
        self.assertIn("flex-flow: row nowrap;", css)
        self.assertIn(".tabulator-col-title-holder {\n        align-items: center;\n        display: flex;", css)
        self.assertIn(".tabulator-col.tabulator-sortable .tabulator-col-content .tabulator-col-title", css)
        self.assertIn("padding-right: 0;", css)
        self.assertIn(".tabulator-col-sorter .tabulator-arrow {\n        display: none;", css)
        self.assertIn(".dataset-viewer-transposed-sort-button", css)
        self.assertIn(".dataset-viewer-context-menu", css)
        self.assertIn(".dataset-viewer-context-menu-item", css)
        self.assertIn(".dataset-viewer-context-menu-divider", css)
        self.assertNotIn(".dataset-viewer-transposed-table tbody tr:hover,\n      .dataset-viewer-transposed-table tbody tr:hover td", css)

    def test_column_profile_tool_static_assets_are_registered(self) -> None:
        _, html_body = self.assert_no_store("/")
        css = self.app_css_contract()
        html = html_body.decode("utf-8")
        js = self.app_js_contract()

        self.assertLess(html.index('id="lineBarTool"'), html.index('id="datasetViewerTool"'))
        self.assertLess(html.index('id="datasetViewerTool"'), html.index('id="profileTool"'))
        self.assertIn('import { createColumnProfileTool } from "./column-profile-tool.js";', js)
        self.assertIn("export function createColumnProfileTool", js)
        self.assertIn("column_profile: freshProfileCache()", js)
        self.assertIn('column_profile: "Profile render"', js)
        self.assertIn('if (tool === "column_profile")', js)
        self.assertIn('api("/api/column-profile/summary"', js)
        self.assertIn('api("/api/column-profile/detail"', js)
        self.assertIn("function renderProfileTable(data, columns = sortedProfileColumns(data.columns || []))", js)
        self.assertIn("const tableScroll = captureProfileTableScroll();", js)
        self.assertIn("function captureProfileTableScroll()", js)
        self.assertIn("function restoreProfileTableScroll(position)", js)
        self.assertIn("restoreProfileTableScroll(tableScroll);", js)
        self.assertNotIn('.profile-table-scroll").addEventListener("scroll"', js)
        self.assertIn('profileSummaryMode: "auto"', js)
        self.assertIn('profileColumnSearch: ""', js)
        self.assertIn('mode: state.profileSummaryMode || "auto"', js)
        self.assertIn("function bindProfileSummaryModeControl()", js)
        self.assertIn("async function handleProfileSummaryModeChange(event)", js)
        self.assertIn("function profileSummaryMode()", js)
        self.assertIn("function normaliseProfileSummaryMode(value)", js)
        self.assertIn("function syncProfileSummaryModeControl(disabled = false)", js)
        self.assertIn("state.profileSummaryMode = nextMode;", js)
        self.assertIn('id="profileSummaryMode" class="profile-summary-mode" role="radiogroup" aria-label="Profile calculation rows"', js)
        self.assertIn('type="radio" name="profileSummaryMode" value="auto"', js)
        self.assertIn('type="radio" name="profileSummaryMode" value="full"', js)
        self.assertIn("<span>Use 100k</span>", js)
        self.assertIn("<span>Use all rows</span>", js)
        self.assertNotIn("function calculateFullProfile()", js)
        self.assertNotIn("profileFullCalcBtn", js)
        self.assertNotIn("Calc all rows", js)
        self.assertNotIn("resetSummaryMode", js)
        self.assertIn('id="profileColumnSearch" class="search profile-column-search" type="search"', js)
        self.assertIn('el("profileColumnSearch")?.addEventListener("input", handleProfileColumnSearch);', js)
        self.assertIn("function handleProfileColumnSearch(event)", js)
        self.assertIn("function applyProfileColumnSearch()", js)
        self.assertIn('row.hidden = !profileColumnMatchesSearch(row.dataset.profileColumn || "");', js)
        self.assertIn("function searchedProfileColumns(columns)", js)
        self.assertIn("function profileColumnMatchesSearch(columnName)", js)
        self.assertIn("No columns match the search.", js)
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
        self.assertIn("function enabledToolIds()", js)
        self.assertIn("if (firstEnabled && toolEnabled(firstEnabled)) return firstEnabled;", js)
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
        self.assertIn('<div class="profile-toolbar">', js)
        self.assertIn('id="profileMeta" class="profile-meta"', js)
        self.assertIn('<div class="profile-content">', js)
        self.assertIn('const profileGroupMeta = document.getElementById("profileGroupMeta");', js)
        self.assertIn('const profileFilter = document.getElementById("profileFilter");', js)
        self.assertIn("profileGroupMeta?.remove();", js)
        self.assertIn("profileFilter?.remove();", js)
        self.assertIn("attachProfileMeta(profileGroupMeta, profileFilter);", js)
        self.assertIn("function attachProfileMeta(profileGroupMeta = null, profileFilter = null)", js)
        self.assertIn('const meta = document.getElementById("profileMeta");', js)
        self.assertIn("if (groupMeta.parentElement !== meta) meta.append(groupMeta);", js)
        self.assertIn("if (filter.parentElement !== meta) meta.append(filter);", js)
        self.assertIn('const filterText = el("profileFilterText");', js)
        self.assertIn("filterText.textContent = filterLabel;", js)
        self.assertIn('filterText.innerHTML = `<span class="profile-warning-meta">${escapeHtml(calculationMeta)}</span> · ${escapeHtml(filterLabel)}`;', js)
        self.assertIn("--filter-applied-bg: #fee2e2;", css)
        self.assertIn("--filter-applied-border: #ef4444;", css)
        self.assertIn("--filter-applied-text: #991b1b;", css)
        self.assertIn("--filter-applied-bg: #450a0a;", css)
        self.assertIn("--filter-applied-border: #f87171;", css)
        self.assertIn("--filter-applied-text: #fecaca;", css)
        self.assertIn("#datasetViewerFilter,\n      #profileFilter,\n      #histogramFilter,\n      #lineBarFilter {\n        background: transparent;\n        border: 1px solid transparent;\n        border-radius: 4px;\n        color: var(--muted);", css)
        self.assertIn("display: inline-flex;\n        align-items: center;\n        gap: 4px;\n        font-size: 10px;", css)
        self.assertIn("padding: 1px 4px;\n        text-overflow: ellipsis;\n        white-space: nowrap;", css)
        self.assertIn("#datasetViewerFilter.dataset-viewer-filter--applied,\n      #profileFilter.profile-filter--applied,\n      #lineBarFilter.line-bar-filter--applied,\n      #histogramFilter.histogram-filter--applied {\n        background: var(--filter-applied-bg);\n        border-color: var(--filter-applied-border);\n        color: var(--filter-applied-text);", css)
        self.assertIn(".visual-area.dataset-viewer-mode,\n      .visual-area.profile-mode {\n        grid-template-columns: minmax(0, 1fr);", css)
        self.assertIn(".visual-area.dataset-viewer-mode .workspace,\n      .visual-area.profile-mode .workspace {\n        background: transparent;\n        border: 0;", css)
        self.assertIn(".visual-area.dataset-viewer-mode .workspace-messages,\n      .visual-area.profile-mode .workspace-messages {\n        display: none;", css)
        self.assertIn(".profile-wrap {\n        display: flex;\n        flex-direction: column;\n        gap: 8px;", css)
        self.assertIn(".profile-toolbar {\n        align-items: center;\n        display: flex;", css)
        self.assertIn(".profile-meta {\n        align-items: flex-end;\n        display: flex;", css)
        self.assertIn("flex-direction: column;\n        gap: 4px;", css)
        self.assertIn("max-width: min(440px, 46%);", css)
        self.assertIn(".profile-meta .workspace-meta,\n      .profile-meta #profileFilter {\n        max-width: 100%;\n        overflow: hidden;\n        text-overflow: ellipsis;\n        white-space: nowrap;", css)
        self.assertIn(".profile-content {\n        display: grid;", css)
        self.assertIn("grid-template-columns: minmax(0, 2fr) minmax(260px, 1fr);", css)
        self.assertIn(".profile-summary-pane,", css)
        self.assertIn(".profile-detail-pane {", css)
        self.assertIn(".profile-summary-actions {", css)
        self.assertIn(".profile-summary-actions {\n        flex: 0 0 auto;", css)
        self.assertNotIn("top: -32px;\n        left: 0;\n        z-index: 7;", css)
        self.assertNotIn("#visualArea.profile-mode .workspace-messages {\n        max-width: min(760px, calc(100% - 190px));", css)
        self.assertNotIn("padding-top: 36px;\n      }\n\n      .profile-summary-pane,\n      .profile-detail-pane", css)
        self.assertIn(".profile-column-search-row {", css)
        self.assertIn(".profile-column-search {", css)
        self.assertIn(".profile-summary-mode {", css)
        self.assertIn(".profile-summary-mode-option {", css)
        self.assertIn(".profile-summary-mode-option.active {", css)
        self.assertIn("#profileGroupMeta {", css)
        self.assertIn(".profile-skipped-button {", css)
        self.assertIn("color: var(--danger);\n        cursor: help;", css)
        self.assertIn(".profile-warning-meta {\n        color: var(--danger);", css)
        self.assertIn(".profile-skipped-popover {", css)
        self.assertIn(".profile-skipped-row {", css)
        self.assertIn("display: flex;\n        flex-direction: column;\n        overflow: hidden;\n        position: relative;", css)
        self.assertNotIn(".profile-full-calc-button {", css)
        self.assertIn(".profile-table {", css)
        self.assertIn(".profile-sort-button {", css)
        self.assertIn(".profile-summary-row {\n        cursor: pointer;\n        user-select: none;\n        -webkit-user-select: none;", css)
        self.assertIn(".profile-summary-row.selected td {", css)
        self.assertIn(".profile-context-menu,\n      .gbm-feature-context-menu,\n      .glm-coefficient-context-menu,\n      .glm-tabulation-context-menu,\n      .spec-context-menu {", css)
        self.assertIn(".profile-context-menu-item,\n      .gbm-feature-context-menu-item,\n      .glm-coefficient-context-menu-item,\n      .glm-tabulation-context-menu-item,\n      .spec-context-menu-item {", css)
        clipboard_toast_block = css[css.index(".clipboard-toast {"):css.index(".clipboard-toast.error {")]
        self.assertIn(".clipboard-toast {\n        position: fixed;\n        right: 18px;\n        top: 18px;", css)
        self.assertNotIn("bottom: 18px;", clipboard_toast_block)
        self.assertIn("border: 1px solid color-mix(in srgb, var(--accent) 45%, var(--line));", clipboard_toast_block)
        self.assertIn("background: color-mix(in srgb, var(--accent) 9%, var(--panel));", clipboard_toast_block)
        self.assertIn("background: color-mix(in srgb, var(--danger) 8%, var(--panel));", css)
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
        self.assertIn("line-height: 1.15;", css)
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

    def test_histogram_tool_static_assets_are_registered(self) -> None:
        _, html_body = self.assert_no_store("/")
        css = self.app_css_contract()
        html = html_body.decode("utf-8")
        js = self.app_js_contract()

        self.assertLess(html.index('id="lineBarTool"'), html.index('id="histogramTool"'))
        self.assertLess(html.index('id="histogramTool"'), html.index('id="ukMapTool"'))
        self.assertIn('import { createHistogramTool } from "./histogram-tool.js";', js)
        self.assertIn("export function createHistogramTool", js)
        self.assertIn('histogram: "Histogram render"', js)
        self.assertIn('if (tool === "histogram")', js)
        self.assertIn('api("/api/histogram/chart"', js)
        self.assertIn("const invalidCount = Math.max(0, Number(data.filtered_row_count || 0) - Number(data.valid_count || 0));", js)
        self.assertIn('const invalidLabel = invalidCount > 0 ? `${formatNumber(invalidCount)} invalid` : "";', js)
        self.assertIn('const sampledLabel = data.sampled_valid_count && data.sampled_valid_count !== data.valid_count', js)
        self.assertIn('const groupMeta = `${formatNumber(data.bins)} bins - ${rowMeta}`;', js)
        self.assertIn('class="histogram-invalid-badge">${escapeHtml(invalidLabel)}</span>', js)
        self.assertIn('class="histogram-sample-badge">${escapeHtml(sampledLabel)}</span>', js)
        self.assertIn('setGroupMeta("histogram", groupMetaHtml || groupMeta, { html: Boolean(groupMetaHtml) });', js)
        self.assertIn('saveToolPresentation("histogram", { groupMeta, groupMetaHtml, chartMessage: warnings });', js)
        self.assertIn("groupMetaHtml: presentation.groupMetaHtml || \"\",", js)
        self.assertIn("setGroupMeta(tool, presentation.groupMetaHtml || presentation.groupMeta, { html: Boolean(presentation.groupMetaHtml) });", js)
        self.assertIn("const HISTOGRAM_BINS_REFRESH_DELAY_MS = 250;", js)
        self.assertIn('const DEFAULT_HISTOGRAM_MEAN_COLOR = "#d13f3f";', js)
        self.assertIn('const DEFAULT_HISTOGRAM_MEDIAN_COLOR = "#1f7a8c";', js)
        self.assertIn('const HISTOGRAM_MEAN_ROW_CLASS = "histogram-stat-mean-row";', js)
        self.assertIn('const HISTOGRAM_MEDIAN_ROW_CLASS = "histogram-stat-median-row";', js)
        self.assertIn('value: [row.bin_mid, row.height, row.bin_lower, row.bin_upper]', js)
        self.assertIn("grid: { left: 72, right: 30, top: 42, bottom: 92, containLabel: false },", js)
        self.assertIn('dimensions: ["bin_mid", "height", "bin_lower", "bin_upper"],', js)
        self.assertIn("encode: { x: 0, y: 1 },", js)
        self.assertIn("dataZoom: [", js)
        self.assertIn('{ type: "inside", xAxisIndex: 0, filterMode: "none" },', js)
        self.assertIn('type: "slider",', js)
        self.assertIn("bottom: 12,", js)
        self.assertIn("fillerColor: alphaColor(barColor, 0.18),", js)
        self.assertIn("function alphaColor(color, alpha)", js)
        self.assertIn("const lower = Number(api.value(2));", js)
        self.assertIn("const upper = Number(api.value(3));", js)
        self.assertIn("const height = Number(api.value(1));", js)
        self.assertIn("const x = Math.floor(leftPx - 0.5);", js)
        self.assertIn("const width = Math.max(1, Math.ceil(rightPx + 0.5) - x);", js)
        self.assertIn("function histogramXAxisPolicy(data, _rows, xLog, chartWidth, formatContinuousValue)", js)
        self.assertIn("function formatHistogramXAxisValue(value, binning, formatContinuousValue)", js)
        self.assertIn("minInterval: 1,", js)
        self.assertIn("maxInterval: step,", js)
        self.assertIn("hideOverlap: true", js)
        self.assertIn("rowFormatter: formatHistogramStatRow", js)
        self.assertIn('const meanColor = getCss("--histogram-mean-color") || DEFAULT_HISTOGRAM_MEAN_COLOR;', js)
        self.assertIn('const medianColor = getCss("--histogram-median-color") || DEFAULT_HISTOGRAM_MEDIAN_COLOR;', js)
        self.assertIn('referenceLine("Mean", mean, meanColor, xLog)', js)
        self.assertIn('referenceLine("Median", median, medianColor, xLog)', js)
        self.assertNotIn("Math.abs(end[0] - start[0]) - 1", js)
        self.assertIn('el("histogramBins").addEventListener("input", () => scheduleHistogramBinsRefresh());', js)
        self.assertIn("window.setTimeout(() => {\n      histogramBinsRefreshTimer = null;\n      refreshHistogram();", js)
        self.assertIn('data-control="histogramDistribution"', html)
        self.assertIn('data-control="histogramYAxis"', html)
        self.assertIn('data-control="histogramLogScale"', html)
        self.assertIn('data-control="histogramSampleMode"', html)
        self.assertIn('id="histogramStatsGrid" class="histogram-grid"', html)
        self.assertNotIn("histogramCopyBtn", html)
        self.assertNotIn("histogramDownloadBtn", html)
        self.assertNotIn("histogramDownloadMenu", html)
        self.assertNotIn("data-histogram-download", html)
        self.assertNotIn("histogramCopyBtn", js)
        self.assertNotIn("histogramDownloadBtn", js)
        self.assertNotIn("histogramDownloadMenu", js)
        self.assertNotIn("histogram-export", css)
        self.assertNotIn("histogram-action", css)
        self.assertNotIn("histogram-download", css)
        self.assertIn("--histogram-mean-color: #d13f3f;", css)
        self.assertIn("--histogram-median-color: #1f7a8c;", css)
        self.assertIn(".visual-area.histogram-mode", css)
        self.assertIn("#histogramGroupMeta:not(.hidden) {\n        align-items: center;\n        display: inline-flex;", css)
        self.assertIn(".histogram-invalid-badge,\n      .histogram-sample-badge {\n        align-items: center;\n        background: var(--filter-applied-bg);", css)
        self.assertIn("border: 1px solid var(--filter-applied-border);\n        border-radius: 4px;\n        color: var(--filter-applied-text);", css)
        self.assertIn("grid-template-columns: minmax(240px, 310px) minmax(0, 1fr);", css)
        self.assertIn(".histogram-grid {\n        background: var(--panel);\n        border: solid var(--line);\n        border-width: 1px 0 0 1px;", css)
        self.assertIn(".histogram-grid.tabulator {\n        border: solid var(--line);\n        border-width: 1px 0 0 1px;", css)
        self.assertIn(".histogram-grid.tabulator .tabulator-header .tabulator-col {\n        font-size: 11px;", css)
        self.assertIn('headerHozAlign: "right"', js)
        self.assertIn('.histogram-grid.tabulator .tabulator-header .tabulator-col[tabulator-field="value"] .tabulator-col-content {\n        justify-content: flex-end;\n        text-align: right;', css)
        self.assertIn(".histogram-grid .tabulator-row .tabulator-cell {\n        align-items: center;", css)
        self.assertIn(".histogram-grid .tabulator-row.histogram-stat-mean-row .tabulator-cell {\n        color: var(--histogram-mean-color, #d13f3f) !important;", css)
        self.assertIn(".histogram-grid .tabulator-row.histogram-stat-median-row .tabulator-cell {\n        color: var(--histogram-median-color, #1f7a8c) !important;", css)

    def test_histogram_axis_policy_formats_integer_ticks(self) -> None:
        js = self.assert_no_store("/static/app/histogram-tool.js")[1].decode("utf-8")
        helper = "\n".join(
            [
                "const HISTOGRAM_X_AXIS_MIN_LABELS = 8;",
                "const HISTOGRAM_X_AXIS_MAX_LABELS = 20;",
                "const HISTOGRAM_X_AXIS_LABEL_PX = 48;",
                self.js_function_source(js, "histogramXAxisPolicy"),
                self.js_function_source(js, "targetHistogramXAxisLabelCount"),
                self.js_function_source(js, "niceIntegerAxisStep"),
                self.js_function_source(js, "formatHistogramXAxisValue"),
            ]
        )
        script = helper + """
const continuous = (value) => `numeric:${value}`;
const integerPolicy = histogramXAxisPolicy(
  { binning: { mode: "integer", kind: "integer", min: 17, max: 96, step: 1 } },
  [],
  false,
  1000,
  continuous,
);
if (integerPolicy.axisOptions.minInterval !== 1) throw new Error("integer minInterval failed");
if (integerPolicy.axisOptions.maxInterval !== 5) throw new Error(`expected 5-year labels, got ${integerPolicy.axisOptions.maxInterval}`);
if (integerPolicy.axisLabel.hideOverlap !== true) throw new Error("integer hideOverlap failed");
if (integerPolicy.axisLabel.formatter(20) !== "20") throw new Error("integer formatter failed");
if (integerPolicy.axisLabel.formatter(16.5) !== "") throw new Error("half-step tick should be blank");
if (integerPolicy.axisLabel.formatter(1234) !== "1,234") throw new Error("integer grouping failed");

const continuousPolicy = histogramXAxisPolicy(
  { binning: { mode: "continuous", kind: "numeric", min: 0, max: 1, step: null } },
  [],
  false,
  1000,
  continuous,
);
if (continuousPolicy.axisLabel.formatter(16.5) !== "numeric:16.5") throw new Error("continuous formatter changed");
if (continuousPolicy.axisOptions.minInterval !== undefined) throw new Error("continuous axis should not force integer intervals");
"""
        self.run_node_script(script)

    def test_app_js_contains_unit_point_map_controls(self) -> None:
        js = self.app_js_contract()
        _, html_body = self.assert_no_store("/")
        html = html_body.decode("utf-8")
        css = Path("src/py_lucidum/static/styles/uk-map.css").read_text(encoding="utf-8")

        self.assertIn('unitColumn: postcodeColumn("unit")', js)
        self.assertIn('latitudeColumn: latitudeColumn()', js)
        self.assertIn('longitudeColumn: longitudeColumn()', js)
        self.assertIn('compactUnitPoints: state.mapLevel === "unit"', js)
        self.assertIn('smoothingLevel: state.mapLevel === "sector" ? state.mapSmoothingLevel : 0', js)
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
        self.assertIn("const MAP_UNIT_POINT_RADIUS_MULTIPLIER = 0.85;", js)
        self.assertIn("const MAP_UNIT_POINT_MIN_RADIUS = 0.5;", js)
        self.assertIn("const MAP_UNIT_POINT_MAX_RADIUS_MULTIPLIER = MAP_UNIT_POINT_RADIUS_MULTIPLIER * 4;", js)
        self.assertIn("unitPointRadiusForZoom", js)
        self.assertIn("const sliderValue = Math.max(1, Math.min(10, Number(state.mapDotSize)));", js)
        self.assertIn("if (!Number.isFinite(sliderValue) || sliderValue <= 1) return MAP_UNIT_POINT_MIN_RADIUS;", js)
        self.assertIn("const maxRadius = unitPointRadiusForZoom(zoom) * MAP_UNIT_POINT_MAX_RADIUS_MULTIPLIER;", js)
        self.assertIn("const progress = (sliderValue - 1) / 9;", js)
        self.assertIn("return MAP_UNIT_POINT_MIN_RADIUS + ((maxRadius - MAP_UNIT_POINT_MIN_RADIUS) * progress);", js)
        self.assertNotIn("function unitPointRadiusScale(value = state.mapDotSize)", js)
        self.assertIn("const pointRadius = unitPointRadiusForCurrentStyle(this.map.getZoom());", js)
        self.assertIn("unitPointHitRadius(pointRadius)", js)
        self.assertIn("const opacityValue = Number(state.mapOpacity);", js)
        self.assertIn("const mapOpacity = Number.isFinite(opacityValue) ? Math.max(0, Math.min(1, opacityValue)) : 1;", js)
        self.assertIn("const baseStrokeOpacity = radius < 2 ? 0 : (radius < 3 ? 0.35 : 0.65);", js)
        self.assertIn("const strokeOpacity = (muted ? Math.min(baseStrokeOpacity, 0.25) : baseStrokeOpacity) * mapOpacity;", js)
        self.assertIn("fillOpacity: muted ? Math.min(mapOpacity, 0.28) : mapOpacity,", js)
        self.assertIn("if (pointRadius <= 1)", js)
        self.assertIn("fillRect(point.x - pointRadius", js)
        self.assertIn("if (!ukMapPointLayer?.setRenderContext)", js)
        self.assertIn("ukMapPointLayer.setRenderContext(scale, hotspotKeys);", js)
        self.assertIn('renderMapLegend(scale, state.lastMapData.response?.label || "Actual");', js)
        self.assertIn("mapLegendCollapsed: startedInMobileLayout", js)
        self.assertIn('id="mapLegendToggle"', html)
        self.assertIn('aria-controls="mapLegendBody"', html)
        self.assertIn('id="mapLegendBody" class="map-legend-body"', html)
        self.assertIn('const legendBody = el("mapLegendBody");', js)
        self.assertIn('legendBody.innerHTML = rows.join("");', js)
        self.assertIn('el("mapLegendToggle").addEventListener("click", toggleMapLegendCollapsed);', js)
        self.assertIn('button.setAttribute("aria-expanded", String(!collapsed));', js)
        self.assertIn(".map-legend.collapsed .map-legend-body", css)
        self.assertIn('<span class="map-control-tile-text">Units</span>', html)

    def test_app_js_preserves_map_view_after_layout_resize(self) -> None:
        js = self.app_js_contract()

        self.assertIn("mapView: null", js)
        self.assertIn("mapViewRestorePending: null", js)
        self.assertIn("mapViewportSyncFrame: null", js)
        self.assertIn("restoringMapView: false", js)
        self.assertIn("function currentMapView()", js)
        self.assertIn('function captureMapView(reason = "")', js)
        self.assertIn("function restoreMapView(view)", js)
        self.assertIn('function scheduleMapViewportSync({ mode = "preserve" } = {})', js)
        self.assertIn("const pendingRestoreView = state.mapViewRestorePending || null;", js)
        self.assertIn("let view = shouldPreserve ? pendingRestoreView || state.mapView : null;", js)
        self.assertIn("if (shouldPreserve && !pendingRestoreView && (state.mapStartupFitDone || state.mapView))", js)
        self.assertIn("mapResizeObserver = new ResizeObserver", js)
        self.assertIn('scheduleMapViewportSync({ mode: "preserve" });', js)
        self.assertIn('ukMapTool.captureView("tool-switch")', js)
        self.assertIn('captureMapView("map-level-change")', js)
        self.assertIn("zoomSnap: 0.25", js)
        self.assertIn("zoomDelta: 0.5", js)
        self.assertIn("const MAP_DEFAULT_VIEW = { center: { lat: 54.5, lng: -3.2 }, zoom: 6 };", js)
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
