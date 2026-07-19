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

    def test_model_navigator_fallback_metadata_contract(self) -> None:
        glm_module = Path("src/py_lucidum/static/app/glm-model-navigator.js").resolve().as_uri()
        gbm_module = Path("src/py_lucidum/static/app/gbm-model-navigator.js").resolve().as_uri()
        script = f"""
import {{ createGlmModelNavigator }} from "{glm_module}";
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
  {{ model_id: "new", label: "New GLM", n_terms: 3, n_features: 2, n_interactions: 1, tabulated: true, diagnostics: {{}} }},
  {{ model_id: "legacy", label: "Legacy GLM", tabulated: false, diagnostics: {{}} }},
]);
if (!glmTarget.innerHTML.includes("<th>Name</th>")) throw new Error("GLM fallback Name heading missing");
if (!glmTarget.innerHTML.includes("<th>Terms</th>")) throw new Error("GLM fallback Terms heading missing");
if (!glmTarget.innerHTML.includes("<th>Features</th>")) throw new Error("GLM fallback Features heading missing");
if (!glmTarget.innerHTML.includes("<th>Interactions</th>")) throw new Error("GLM fallback Interactions heading missing");
if (!glmTarget.innerHTML.includes("<th>Tabulated</th>")) throw new Error("GLM fallback Tabulated heading missing");
if (!glmTarget.innerHTML.includes('<td class="numeric">3</td>\\n        <td class="numeric">2</td>\\n        <td class="numeric">1</td>\\n        <td>Yes</td>')) throw new Error("GLM captured metadata missing");
if (!glmTarget.innerHTML.includes('<td class="numeric"></td>\\n        <td class="numeric"></td>\\n        <td class="numeric"></td>\\n        <td>-</td>')) throw new Error("GLM legacy metadata fallback failed");
if (glmNavigator.optionalCount(null) !== "" || glmNavigator.optionalCount(0) !== "0") throw new Error("GLM optional count formatting failed");

const gbmNavigator = createGbmModelNavigator({{
  escapeHtml,
  formatModelMetric: (value) => String(value ?? ""),
  modelInteractionConstraintLabel: () => "No",
  modelLabel: (model) => model.label || model.model_id,
  normaliseModel: (model) => model,
  uniqueModels: (models) => models,
  onFallbackSelectionChange: () => {{}},
}});
const gbmTarget = target();
gbmNavigator.renderFallback(gbmTarget, [{{ model_id: "gbm", label: "GBM" }}]);
if (!gbmTarget.innerHTML.includes("<th>Name</th>")) throw new Error("GBM fallback Name heading missing");
if (gbmTarget.innerHTML.includes("<th>Model</th>")) throw new Error("GBM fallback kept Model heading");
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

    def test_index_references_only_local_no_store_assets(self) -> None:
        _, body = self.assert_no_store("/")
        html = body.decode("utf-8")
        asset_urls = re.findall(r'<(?:link|script|img)\b[^>]*(?:href|src)="([^"]+)"', html)

        self.assertTrue(asset_urls)
        self.assertFalse(any(url.startswith(("http://", "https://", "//")) for url in asset_urls))
        for url in asset_urls:
            if url.startswith("/"):
                with self.subTest(url=url):
                    self.assert_no_store(url)

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
        glm_formula_builder = (static_root / "app/glm-formula-builder.js").read_text(encoding="utf-8")
        gbm = (static_root / "app/gbm-tool.js").read_text(encoding="utf-8")
        gbm_evaluation_chart = (static_root / "app/gbm-evaluation-chart.js").read_text(encoding="utf-8")
        gbm_feature_controls = (static_root / "app/gbm-feature-parameter-controls.js").read_text(encoding="utf-8")
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
        self.assertIn(".app-settings-strip .segmented button.active {", controls)
        self.assertIn(".app-control-button,", controls)
        self.assertIn("height: var(--app-control-button-height);", controls)
        self.assertIn(".app-command-button {", controls)
        self.assertIn(".app-command-button--danger {", controls)
        self.assertIn(".app-control-input {", controls)
        self.assertIn('class="dataset-viewer-toolbar app-control-strip app-control-strip-row"', dataset_viewer)
        self.assertIn('class="search dataset-viewer-search app-control-input"', dataset_viewer)
        self.assertIn('class="filter-action app-control-button"', dataset_viewer)
        self.assertIn('class="profile-toolbar app-control-strip app-control-strip-row"', profile)
        self.assertIn('class="search profile-column-search app-control-input"', profile)
        self.assertIn('id="profilePaneResizer"', profile)
        self.assertIn('role="separator"', profile)
        self.assertIn('aria-orientation="vertical"', profile)
        self.assertIn('import { bindVerticalListNavigation } from "./shared/list-navigation.js";', profile)
        self.assertIn('itemSelector: "[data-profile-column]"', profile)
        self.assertIn('row.tabIndex = selected ? 0 : -1;', profile)
        self.assertIn('class="toolbar line-bar-settings-strip app-control-strip app-settings-strip hidden"', index)
        self.assertIn('class="toolbar histogram-settings-strip app-control-strip app-settings-strip hidden"', index)
        self.assertIn('class="line-bar-workspace-controls hidden"', index)
        self.assertIn('id="histogramWorkspaceControls" class="histogram-workspace-controls hidden"', index)
        self.assertIn('id="histogramSplitResizer"', index)
        self.assertIn('class="line-bar-table-search-row app-control-strip"', line_bar)
        self.assertIn('import { bindSettingsStripOverflowCue } from "./shared/settings-strip.js";', line_bar)
        self.assertIn('import { bindSettingsStripOverflowCue } from "./shared/settings-strip.js";', histogram)
        self.assertIn('classList.toggle("app-settings-overflow-left"', settings_strip)
        self.assertIn('classList.toggle(\n      "app-settings-overflow-right"', settings_strip)
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
        self.assertIn(".glm-tabulation-crosstab-group {\n        align-items: center;\n        display: inline-flex;\n        flex: 1 1 auto;", glm_css)
        self.assertIn(".glm-tabulation-crosstab {\n        background: var(--panel);", glm_css)
        self.assertIn("flex: 1 1 auto;\n        font-size: 12px;", glm_css)
        self.assertNotIn("max-width: 240px;", glm_css)
        self.assertNotIn("width: clamp(150px, 18vw, 240px);", glm_css)
        self.assertNotIn(".glm-tabulation-check,", glm_css)
        self.assertIn('data-glm-scope="all" data-stable-label="All" class="app-control-button glm-builder-option-button', glm)
        self.assertIn('data-glm-scope="training" data-stable-label="Training" class="app-control-button glm-builder-option-button', glm)
        self.assertIn('id="glmBuildBtn" class="tab app-control-button model-busy-button glm-build-button', glm)
        self.assertNotIn('id="glmFormulaAssistBtn" class="tab ', glm)
        self.assertIn(".glm-builder-actions .glm-builder-option-button.active,", glm_css)
        self.assertIn(".glm-header-scope-control {\n        gap: 0;", glm_css)
        self.assertIn(".glm-header-scope-control button.glm-builder-option-button {\n        padding-inline: 6px;", glm_css)
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
        self.assertIn('class="gbm-feature-main-control-strip app-control-strip"', gbm)
        self.assertIn('id="gbmFeatureSetupBtn" class="app-control-button gbm-feature-option-button gbm-feature-setup-button', gbm)
        self.assertIn('aria-label="Feature setup" title="Feature setup" aria-controls="gbmFeatureSetupPanel" aria-expanded=', gbm)
        self.assertIn('class="gbm-feature-setup-icon" viewBox="0 0 24 24"', gbm)
        self.assertIn('id="gbmFeatureSetupPanel" class="gbm-feature-setup-panel', gbm)
        self.assertIn(
            "${featureScenarioDropdownHtml(data.feature_scenarios || [], data.active_feature_scenario || null)}\n"
            "                  ${featureInteractionConstraintDropdownHtml(data.feature_interaction_groupings || [], data.active_feature_interaction_constraints || null, data.features || [])}\n"
            "                  ${featureInteractionPairsDropdownHtml(data.active_feature_interaction_constraints || null, data.features || [])}",
            gbm,
        )
        self.assertIn('id="gbmClearFeaturesBtn" class="app-control-button app-command-button gbm-feature-command-button"', gbm)
        self.assertIn('id="gbmSelectFeaturesBtn" class="app-control-button app-command-button gbm-feature-command-button"', gbm)
        self.assertNotIn('id="gbmClearFeaturesBtn" class="tab ', gbm)
        self.assertNotIn('id="gbmSelectFeaturesBtn" class="tab ', gbm)
        self.assertIn('data-stable-label="${escapeHtml(featureMetricModeLabel(mode))}"', gbm)
        self.assertIn('localStorage.getItem("py_lucidum_gbm_feature_setup_open") === "true"', gbm)
        self.assertIn('localStorage.setItem("py_lucidum_gbm_feature_setup_open", String(featureSetupOpen));', gbm)
        self.assertIn('button.setAttribute("aria-expanded", featureSetupOpen ? "true" : "false");', gbm)
        self.assertIn('if (!featureSetupOpen) closeGbmFeatureToolbarMenus();', gbm)
        self.assertIn('scheduleGbmTableRedraws([featureTable, ebmGainSummaryTable]);', gbm)
        self.assertIn(".gbm-feature-metric-option::after {", gbm_css)
        self.assertIn(".gbm-feature-actions {\n        flex: 1 1 260px;\n        flex-wrap: nowrap;\n        justify-content: flex-end;", gbm_css)
        self.assertIn(".gbm-feature-option-button.active {", gbm_css)
        self.assertIn(".gbm-feature-setup-button {\n        margin-left: 28px;", gbm_css)
        self.assertIn(".gbm-feature-setup-icon {", gbm_css)
        self.assertIn(".gbm-feature-setup-panel {\n        background: var(--sidebar-bg);", gbm_css)
        self.assertIn(".gbm-feature-setup-controls {", gbm_css)
        self.assertIn(".gbm-feature-menu-button:focus-visible {", gbm_css)
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

    def test_uk_map_overlay_markup_starts_collapsed(self) -> None:
        index = (
            Path(__file__).resolve().parents[1] / "src/py_lucidum/static/index.html"
        ).read_text(encoding="utf-8")

        self.assertIn(
            'id="mapFloatingControl" class="map-floating-control hidden collapsed"',
            index,
        )
        self.assertIn(
            'id="mapControlReset" class="map-control-reset" type="button" title="Expand map controls" '
            'aria-label="Expand map controls" aria-controls="mapFloatingControl" aria-expanded="false"',
            index,
        )
        self.assertIn('id="mapLegend" class="map-legend hidden collapsed"', index)
        self.assertIn(
            'id="mapLegendToggle" class="map-legend-toggle" type="button" title="Expand legend" '
            'aria-label="Expand legend" aria-controls="mapLegendBody" aria-expanded="false"',
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
