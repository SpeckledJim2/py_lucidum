from __future__ import annotations

import asyncio
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

    def test_index_uses_stable_local_asset_urls_and_disables_cache(self) -> None:
        _, body = self.assert_no_store("/")
        html = body.decode("utf-8")

        self.assertIn("<title>lucidum · sample.csv</title>", html)
        self.assertIn('href="/favicon.ico"', html)
        self.assertIn('src="/favicon.ico"', html)
        self.assertIn('id="sidebarToggleBtn"', html)
        self.assertIn('aria-controls="appSidebar"', html)
        self.assertIn('aria-label="Collapse sidebar"', html)
        self.assertIn('<aside id="appSidebar">', html)
        self.assertIn('id="lineBarTool" class="tool-option active" type="button" data-tool="line_bar" aria-label="Line and bar"', html)
        self.assertIn('class="tool-label">Line and bar</span>', html)
        self.assertIn('id="ukMapTool" class="tool-option" type="button" data-tool="uk_map" aria-label="UK mapping"', html)
        self.assertIn('class="tool-label">UK mapping</span>', html)
        self.assertIn('id="themeBtn"', html)
        self.assertIn('aria-label="Switch to dark mode"', html)
        self.assertIn("theme-icon-moon", html)
        self.assertIn("theme-icon-sun", html)
        self.assertNotIn('id="themeBtn" class="ghost">Dark</button>', html)
        self.assertNotIn("<h2>Tool</h2>", html)
        self.assertIn('href="/static/app.css"', html)
        self.assertIn('src="/static/app.js"', html)
        self.assertNotIn("?v=", html)

    def test_static_app_assets_disable_cache(self) -> None:
        self.assert_no_store("/static/app.js")
        self.assert_no_store("/static/app.css")

    def test_feature_picker_rows_are_compact(self) -> None:
        _, body = self.assert_no_store("/static/app.css")
        css = body.decode("utf-8")

        self.assertIn("min-height: 20px;", css)
        self.assertIn("padding: 1px 6px;", css)
        self.assertIn("font-size: 11px;", css)
        self.assertIn("font-size: 9px;", css)

    def test_saved_filter_select_uses_feature_list_row_spacing(self) -> None:
        _, html_body = self.assert_no_store("/")
        _, css_body = self.assert_no_store("/static/app.css")
        _, js_body = self.assert_no_store("/static/app.js")
        html = html_body.decode("utf-8")
        css = css_body.decode("utf-8")
        js = js_body.decode("utf-8")

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
        self.assertIn('const preserveMapViewOnReload = state.tool === "uk_map" && Boolean(ukMap);', js)
        self.assertIn("if (preserveMapViewOnReload) state.preserveMapView = true;", js)
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
        _, css_body = self.assert_no_store("/static/app.css")
        _, js_body = self.assert_no_store("/static/app.js")
        html = html_body.decode("utf-8")
        css = css_body.decode("utf-8")
        js = js_body.decode("utf-8")

        self.assertIn('<section class="section sidebar-filter-section filter-collapsed">', html)
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
        self.assertIn(".filter-row-meta {\n        min-width: 0;\n        overflow: hidden;\n        text-align: right;\n        text-overflow: ellipsis;\n        white-space: nowrap;\n        color: var(--muted);\n        font-size: 11px;\n        margin-left: auto;", css)
        self.assertIn(".filter-sidebar-clear {\n        width: 24px;\n        min-height: 24px;\n        font-size: 14px;", css)
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
        _, css_body = self.assert_no_store("/static/app.css")
        _, js_body = self.assert_no_store("/static/app.js")
        html = html_body.decode("utf-8")
        css = css_body.decode("utf-8")
        js = js_body.decode("utf-8")

        self.assertIn('<section class="section sidebar-kpi-section">', html)
        self.assertLess(html.index("<h2>KPIs</h2>"), html.index("<h2>FILTER</h2>"))
        self.assertLess(html.index('id="actualNumerator"'), html.index('id="kpiSelect"'))
        self.assertLess(html.index('id="denominator"'), html.index('id="kpiSelect"'))
        self.assertIn('id="kpiCollapseBtn"', html)
        self.assertIn('id="kpiSelectedMeta" class="kpi-selected-meta"', html)
        self.assertIn('id="kpiSelect" class="feature-list kpi-list" role="listbox"', html)
        self.assertIn(".kpi-header h2 {\n        margin: 0;\n        font-size: 12px;", css)
        self.assertIn(".kpi-selected-meta {\n        min-width: 0;\n        overflow: hidden;\n        text-align: right;", css)
        self.assertIn(".sidebar-kpi-section.kpi-collapsed .kpi-controls,", css)
        self.assertIn(".sidebar-kpi-section.kpi-collapsed #kpiSelect {\n        display: none;", css)
        self.assertIn(".metric-title {\n        display: flex;\n        align-items: baseline;\n        justify-content: space-between;", css)
        self.assertIn("font-weight: 400;\n        font-size: 12px;", css)
        self.assertIn(".chart-side-section h2.metric-title {\n        font-weight: 400;", css)
        self.assertIn(".metric-value {\n        color: var(--muted);\n        margin-left: auto;\n        text-align: right;", css)
        self.assertIn("#kpiSelect {\n        flex: 0 0 auto;\n        width: 100%;\n        height: auto;\n        min-height: 0;", css)
        self.assertIn(".kpi-list .kpi-option.active {\n        background: color-mix(in srgb, #f59e0b 22%, var(--panel));", css)
        self.assertIn(".kpi-detail {\n        min-width: 0;\n        overflow: hidden;\n        text-align: right;", css)
        self.assertIn("kpiCollapsed: false", js)
        self.assertIn("collapsedKpiGroups: new Set()", js)
        self.assertIn("activeKpiFormat: null", js)
        self.assertIn("function availableKpis()", js)
        self.assertIn('el("kpiSelectedMeta").textContent = kpi ? kpi.name : "";', js)
        self.assertIn('if (denominator === "__none__") return "N";', js)
        self.assertIn('heading.className = "saved-filter-theme kpi-theme";', js)
        self.assertIn('button.className = `feature kpi-option${active ? " active" : ""}`;', js)
        self.assertIn("function selectKpi(kpi)", js)
        self.assertIn('el("kpiCollapseBtn").addEventListener("click", () => setKpiCollapsed(!state.kpiCollapsed));', js)
        self.assertIn("syncKpiSelectionFromMetrics();", js)

    def test_filter_footer_and_sidebar_filter_controls_contract(self) -> None:
        _, html_body = self.assert_no_store("/")
        _, css_body = self.assert_no_store("/static/app.css")
        _, js_body = self.assert_no_store("/static/app.js")
        html = html_body.decode("utf-8")
        css = css_body.decode("utf-8")
        js = js_body.decode("utf-8")

        self.assertLess(html.index('id="sidebarToggleBtn"'), html.index('id="filterFooterToggleBtn"'))
        self.assertIn('<div class="layout-toggle-group">', html)
        self.assertIn('id="filterFooterToggleBtn" class="footer-toggle"', html)
        self.assertIn('aria-controls="filterFooter"', html)
        self.assertIn('class="footer-toggle-icon"', html)
        self.assertIn('<footer id="filterFooter" class="filter-footer">', html)
        self.assertLess(html.index('id="filterClearBtn"'), html.index('id="filterApplyBtn"'))
        self.assertLess(html.index('id="filterApplyBtn"'), html.index('id="filterInput"'))
        self.assertGreater(html.index('id="filterInput"'), html.index('id="filterFooter"'))
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
        self.assertIn(".footer-toggle-icon {\n        width: 20px;\n        height: 22px;", css)
        self.assertIn("body.filter-footer-collapsed .footer-toggle-icon::before {\n        background: transparent;", css)
        self.assertIn(".filter-footer {\n        display: grid;\n        grid-template-columns: 28px 28px minmax(0, 1fr);", css)
        self.assertIn("body.filter-footer-collapsed .filter-footer {\n        display: none;", css)
        self.assertIn(".filter-controls-row {\n        display: grid;\n        grid-template-columns: auto minmax(0, 1fr);", css)
        self.assertIn("body.saved-filter-single-mode .filter-operator,\n      body.saved-filter-grouped-mode .filter-operator {\n        visibility: hidden;", css)
        self.assertIn(".filter-operator {\n        justify-self: end;", css)
        self.assertIn("filterFooterCollapsed: false", js)
        self.assertIn('filterSelectionMode: "single"', js)
        self.assertIn('document.body.classList.toggle("saved-filter-grouped-mode", nextMode === "grouped");', js)
        self.assertIn('setFilterSelectionMode(state.filterSelectionMode, { apply: false });', js)
        self.assertIn("function setFilterFooterVisible(visible)", js)
        self.assertIn('document.body.classList.toggle("filter-footer-collapsed", state.filterFooterCollapsed);', js)
        self.assertIn('el("filterFooter").setAttribute("aria-hidden", String(state.filterFooterCollapsed));', js)
        self.assertIn("function syncFilterFooterToggleButton()", js)
        self.assertIn('el("filterFooterToggleBtn").addEventListener("click", () => setFilterFooterVisible(state.filterFooterCollapsed));', js)
        self.assertIn('el("filterSidebarClearBtn").addEventListener("click", clearFilter);', js)

    def test_chart_search_inputs_have_clear_buttons(self) -> None:
        _, html_body = self.assert_no_store("/")
        _, css_body = self.assert_no_store("/static/app.css")
        _, js_body = self.assert_no_store("/static/app.js")
        html = html_body.decode("utf-8")
        css = css_body.decode("utf-8")
        js = js_body.decode("utf-8")

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
        _, js_body = self.assert_no_store("/static/app.js")
        js = js_body.decode("utf-8")

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

    def test_theme_toggle_uses_icons_and_accessible_labels(self) -> None:
        _, css_body = self.assert_no_store("/static/app.css")
        _, js_body = self.assert_no_store("/static/app.js")
        css = css_body.decode("utf-8")
        js = js_body.decode("utf-8")

        self.assertIn(".theme-toggle", css)
        self.assertIn("width: 28px;", css)
        self.assertIn("height: 24px;", css)
        self.assertIn("min-height: 24px;", css)
        self.assertIn(".theme-icon-moon", css)
        self.assertIn(".theme-icon-sun", css)
        self.assertIn("body.dark .theme-icon-moon", css)
        self.assertIn("body.dark .theme-icon-sun", css)
        self.assertIn('const label = document.body.classList.contains("dark") ? "Switch to light mode" : "Switch to dark mode";', js)
        self.assertIn('el("themeBtn").setAttribute("aria-label", label);', js)
        self.assertIn('el("themeBtn").title = label;', js)
        self.assertNotIn('.textContent = document.body.classList.contains("dark") ? "Light" : "Dark"', js)

    def test_line_bar_quantile_control_is_numeric_only(self) -> None:
        _, html_body = self.assert_no_store("/")
        _, js_body = self.assert_no_store("/static/app.js")
        html = html_body.decode("utf-8")
        js = js_body.decode("utf-8")

        self.assertIn('id="bandControl"', html)
        self.assertIn('id="quantileControl"', html)
        self.assertLess(html.index('id="bandControl"'), html.index('id="quantileControl"'))
        self.assertIn('<span id="bandLabel">Banding</span>', html)
        self.assertIn("<h3>Quantile</h3>", html)
        self.assertIn('<div class="segmented" data-control="quantileMode">', html)
        self.assertIn('<button data-value="off" class="active">-</button>', html)
        self.assertIn('<button data-value="quantile">Use quantiles</button>', html)
        self.assertIn('quantileMode: "off"', js)
        self.assertIn('el("bandLabel").textContent = state.quantileMode === "quantile" ? "Quantiles" : "Banding";', js)
        self.assertIn('el("quantileControl").classList.toggle("hidden", !isNumeric);', js)
        self.assertIn('quantileMode: isNumeric ? state.quantileMode : "off"', js)
        self.assertIn('const previousControlValue = state[group.dataset.control];', js)
        self.assertIn('state.quantileMode === "quantile" && previousControlValue !== "quantile"', js)
        self.assertIn('state.bandWidth = "10";', js)
        self.assertIn('function normalizeBandWidthForQuantiles()', js)

    def test_london_map_button_icon_fills_button(self) -> None:
        _, css_body = self.assert_no_store("/static/app.css")
        _, js_body = self.assert_no_store("/static/app.js")
        css = css_body.decode("utf-8")
        js = js_body.decode("utf-8")

        self.assertIn('class="map-place-icon-london"', js)
        self.assertIn(".map-place-button img.map-place-icon-london", css)
        self.assertIn("width: 30px;", css)
        self.assertIn("height: 30px;", css)
        self.assertIn("body.dark .map-place-button img", css)
        self.assertIn("mix-blend-mode: screen;", css)
        self.assertIn("filter: invert(1) grayscale(1) brightness(1.7) contrast(1.08);", css)

    def test_map_layer_control_uses_distinct_radio_groups(self) -> None:
        _, css_body = self.assert_no_store("/static/app.css")
        _, js_body = self.assert_no_store("/static/app.js")
        css = css_body.decode("utf-8")
        js = js_body.decode("utf-8")

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
        _, css_body = self.assert_no_store("/static/app.css")
        _, js_body = self.assert_no_store("/static/app.js")
        css = css_body.decode("utf-8")
        js = js_body.decode("utf-8")

        self.assertIn("--sidebar-bg: #dce4ef;", css)
        self.assertIn("--sidebar-bg: #24334b;", css)
        self.assertIn("--sidebar-collapsed-width: 52px;", css)
        self.assertIn("background: var(--sidebar-bg);", css)
        self.assertIn(".sidebar-toggle-icon", css)
        self.assertIn("border: 0;", css)
        self.assertIn("width: 6px;", css)
        self.assertIn(".sidebar-toggle-icon::after", css)
        self.assertIn("left: 4px;", css)
        self.assertIn("width: 2px;", css)
        self.assertIn("body.sidebar-collapsed .sidebar-toggle-icon::before {\n        background: transparent;", css)
        self.assertNotIn("left: 6px;", css)
        self.assertIn("body.sidebar-collapsed .shell {\n        grid-template-columns: var(--sidebar-collapsed-width) minmax(0, 1fr);", css)
        self.assertIn("body.sidebar-collapsed .sidebar-resizer {\n        display: none;", css)
        self.assertIn("body.sidebar-collapsed aside {\n        align-items: center;", css)
        self.assertIn("body.sidebar-collapsed aside > .section:not(#toolSelectorSection) {\n        display: none;", css)
        self.assertIn("body.sidebar-collapsed .tool-option {\n        width: 36px;", css)
        self.assertIn("height: 30px;\n        min-height: 30px;\n        justify-content: flex-start;", css)
        self.assertIn("padding: 3px 8px;\n        border-radius: 6px;", css)
        self.assertIn("body.sidebar-collapsed .tool-label {\n        position: absolute;", css)
        self.assertNotIn("body.sidebar-collapsed aside,\n      body.sidebar-collapsed .sidebar-resizer", css)
        self.assertIn("sidebarVisible: true", js)
        self.assertIn('document.body.classList.toggle("sidebar-collapsed", !state.sidebarVisible)', js)
        self.assertIn('el("appSidebar").removeAttribute("aria-hidden");', js)
        self.assertNotIn('el("appSidebar").setAttribute("aria-hidden", String(!state.sidebarVisible));', js)
        self.assertIn('el("sidebarToggleBtn").addEventListener("click", () => setSidebarVisible(!state.sidebarVisible))', js)
        self.assertIn('const label = state.sidebarVisible ? "Collapse sidebar" : "Expand sidebar";', js)
        self.assertIn('button.setAttribute("aria-expanded", String(state.sidebarVisible));', js)

    def test_tool_selector_aligns_with_main_toolbar(self) -> None:
        _, css_body = self.assert_no_store("/static/app.css")
        css = css_body.decode("utf-8")

        self.assertIn(".tool-selector-section {\n        margin-bottom: 14px;\n        padding-top: 2px;", css)

    def test_app_js_contains_unit_point_map_controls(self) -> None:
        _, body = self.assert_no_store("/static/app.js")
        js = body.decode("utf-8")

        self.assertIn('unitColumn: postcodeColumn("unit")', js)
        self.assertIn('latitudeColumn: latitudeColumn()', js)
        self.assertIn('longitudeColumn: longitudeColumn()', js)
        self.assertIn('aliases: ["PostcodeUnit", "POSTCODE_UNIT"]', js)
        self.assertIn('longitude: ["long", "longitude", "LONGITUDE", "LONGiTUDE"]', js)
        self.assertIn("makeUnitPointLayer", js)
        self.assertIn("unitPointRadiusForZoom", js)
        self.assertIn("unitPointHitRadius(pointRadius)", js)
        self.assertIn("if (pointRadius <= 1)", js)
        self.assertIn("fillRect(point.x - pointRadius", js)
        self.assertIn("<span>Units</span>", js)

    def test_app_js_refits_map_after_layout_resize(self) -> None:
        _, body = self.assert_no_store("/static/app.js")
        js = body.decode("utf-8")

        self.assertIn("function scheduleMapResize({ refit = false } = {})", js)
        self.assertIn("fitMapToLayer({ animate: false })", js)
        self.assertIn("zoomSnap: 0.25", js)
        self.assertIn("zoomDelta: 0.5", js)
        self.assertIn("const MAP_INITIAL_FIT_OPTIONS = { animate: false };", js)
        self.assertIn("mapStartupFitDone: false", js)
        self.assertIn("if (!state.mapStartupFitDone)", js)
        self.assertIn("state.mapStartupFitDone = true;", js)
        self.assertIn("fitMapBounds(bounds, data.level, MAP_INITIAL_FIT_OPTIONS)", js)
        self.assertIn("scheduleMapResize({ refit: didFitLayer });", js)

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
