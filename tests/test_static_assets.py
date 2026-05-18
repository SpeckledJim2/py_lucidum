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
        self.assertIn('<aside id="appSidebar">', html)
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
        self.assertIn(".saved-filter-list .feature {\n        font-size: 12px;\n        min-height: 22px;\n        padding: 2px 6px;\n        justify-content: flex-start;", css)
        self.assertIn('button.className = "feature saved-filter-option";', js)
        self.assertIn('button.setAttribute("role", "option");', js)
        self.assertIn('button.setAttribute("aria-selected", "false");', js)
        self.assertIn("const multiSelect = event.metaKey || event.ctrlKey;", js)
        self.assertIn('list.querySelectorAll(".saved-filter-option").forEach((option) => {', js)
        self.assertIn("const isClickedOption = option === button;", js)
        self.assertIn('option.classList.toggle("active", isClickedOption);', js)
        self.assertIn('button.classList.toggle("active", !selected);', js)
        self.assertIn('querySelectorAll(\'.saved-filter-option[aria-selected="true"]\')', js)

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
        self.assertIn("background: var(--sidebar-bg);", css)
        self.assertIn(".sidebar-toggle-icon", css)
        self.assertIn("border: 0;", css)
        self.assertIn("width: 6px;", css)
        self.assertIn(".sidebar-toggle-icon::after", css)
        self.assertIn("left: 4px;", css)
        self.assertIn("width: 2px;", css)
        self.assertIn("body.sidebar-collapsed .sidebar-toggle-icon::before {\n        background: transparent;", css)
        self.assertNotIn("left: 6px;", css)
        self.assertIn("body.sidebar-collapsed .shell", css)
        self.assertIn("body.sidebar-collapsed aside", css)
        self.assertIn("sidebarVisible: true", js)
        self.assertIn('document.body.classList.toggle("sidebar-collapsed", !state.sidebarVisible)', js)
        self.assertIn('el("sidebarToggleBtn").addEventListener("click", () => setSidebarVisible(!state.sidebarVisible))', js)
        self.assertIn('button.setAttribute("aria-expanded", String(state.sidebarVisible));', js)

    def test_tool_selector_aligns_with_main_toolbar(self) -> None:
        _, css_body = self.assert_no_store("/static/app.css")
        css = css_body.decode("utf-8")

        self.assertIn(".tool-selector-section {\n        margin-bottom: 28px;\n        padding-top: 2px;", css)

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
