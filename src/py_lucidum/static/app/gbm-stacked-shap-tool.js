import { stackedShapChartOption } from "./gbm-stacked-shap-chart.js";
import { bindSettingsStripOverflowCue } from "./shared/settings-strip.js";

const BAND_STEPS = makeBandSteps();
const BAND_BUTTONS = [0.01, 0.1, 1, 5, 10, 100];
const TAIL_OPTIONS = [
  { value: 0, label: "-" },
  { value: 0.1, label: "0.1%" },
  { value: 0.5, label: "0.5%" },
  { value: 1, label: "1%" },
  { value: 5, label: "5%" },
];
const FEATURE_COUNT_OPTIONS = [
  { value: "1", label: "1" },
  { value: "2", label: "2" },
  { value: "3", label: "3" },
  { value: "5", label: "5" },
  { value: "10", label: "10" },
  { value: "all", label: "All" },
];
const STACKED_SHAP_SIDE_DEFAULT_WIDTH = 320;
const STACKED_SHAP_SIDE_MIN_WIDTH = 240;
const STACKED_SHAP_SIDE_MAX_WIDTH = 560;
const STACKED_SHAP_CHART_MIN_WIDTH = 420;
const STACKED_SHAP_SPLITTER_KEY_STEP = 10;
const STACKED_SHAP_STACKED_MEDIA = "(max-width: 900px)";

export function createGbmStackedShapTool({ api, escapeHtml, setNotice }) {
  let modelId = "";
  let lastModelId = "";
  let config = null;
  let chart = null;
  let resizeObserver = null;
  let lastPayload = null;
  let configSeq = 0;
  let plotSeq = 0;
  let sidePanelWidth = STACKED_SHAP_SIDE_DEFAULT_WIDTH;
  let controlsCollapsed = true;
  let sidePanelCollapsed = false;
  let chartResizeFrame = null;
  let chartResizeFlush = false;
  let settledObserverSize = null;
  let layoutMediaQuery = null;
  let layoutMediaListener = null;
  let settingsOverflowCleanup = null;
  const state = {
    modelFeature: "",
    featureSort: "importance",
    xSort: "alpha",
    tailPercent: 0,
    numFeatures: "all",
    banding: 1,
    bandingKey: "",
    bandingPendingKey: "",
    bandingSeq: 0,
  };

  function shellHtml() {
    const toolbarHidden = controlsCollapsed ? " hidden inert aria-hidden=\"true\"" : "";
    const sideHidden = sidePanelCollapsed ? " hidden inert aria-hidden=\"true\"" : "";
    return `
      <div id="gbmStackedShapRoot" class="gbm-stacked-shap-view">
        <div id="gbmStackedShapControls" class="gbm-stacked-shap-controls toolbar app-control-strip app-settings-strip${controlsCollapsed ? " hidden" : ""}"${toolbarHidden}></div>
        <div class="gbm-stacked-shap-workspace">
          <aside id="gbmStackedShapSide" class="gbm-stacked-shap-side${sidePanelCollapsed ? " hidden" : ""}"${sideHidden}>
            <section class="gbm-stacked-shap-feature-section chart-side-section">
              <div class="section-title-row">
                <h2>Model feature</h2>
                <div class="segmented gbm-stacked-shap-feature-sort" role="group" aria-label="Stacked SHAP model feature sort">
                  <button type="button" data-gbm-stacked-shap-feature-sort="importance" data-stable-label="Importance">Importance</button>
                  <button type="button" data-gbm-stacked-shap-feature-sort="alpha" data-stable-label="A-Z">A-Z</button>
                </div>
              </div>
              <div id="gbmStackedShapFeatureList" class="feature-list gbm-stacked-shap-feature-list" role="listbox" aria-label="Stacked SHAP model feature"></div>
            </section>
          </aside>
          <div id="gbmStackedShapMainResizer" class="gbm-stacked-shap-main-resizer app-resizer app-resizer--vertical" role="separator" aria-orientation="vertical" aria-label="Resize Stacked SHAP feature chooser" tabindex="0"></div>
          <section class="gbm-stacked-shap-main">
            <div class="gbm-stacked-shap-workspace-controls">
              <button id="gbmStackedShapSideToggle" class="gbm-stacked-shap-overlay-button app-control-button" type="button" aria-controls="gbmStackedShapSide" aria-expanded="${String(!sidePanelCollapsed)}">
                <svg class="gbm-stacked-shap-toggle-icon gbm-stacked-shap-chevron-horizontal" viewBox="0 0 24 24" aria-hidden="true" focusable="false">
                  <path d="m15 18-6-6 6-6"></path>
                </svg>
              </button>
              <button id="gbmStackedShapToolbarToggle" class="gbm-stacked-shap-overlay-button app-control-button" type="button" aria-controls="gbmStackedShapControls" aria-expanded="${String(!controlsCollapsed)}">
                <svg class="gbm-stacked-shap-toggle-icon gbm-stacked-shap-chevron-vertical" viewBox="0 0 24 24" aria-hidden="true" focusable="false">
                  <path d="m18 15-6-6-6 6"></path>
                </svg>
              </button>
            </div>
            <div class="gbm-stacked-shap-chart-shell">
              <div id="gbmStackedShapMessage" class="gbm-shap-message hidden"></div>
              <div id="gbmStackedShapChart" class="gbm-stacked-shap-chart" aria-label="GBM Stacked SHAP plot"></div>
            </div>
          </section>
        </div>
      </div>
    `;
  }

  async function render(nextModelId) {
    modelId = String(nextModelId || "");
    const root = rootNode();
    if (!root) return;
    bindStaticEvents(root);
    syncLayoutVisibility(root, { resize: false });
    if (!modelId) {
      config = null;
      renderEmpty("No active GBM selected");
      return;
    }
    if (modelId !== lastModelId) config = null;
    const seq = ++configSeq;
    renderLoading("Loading Stacked SHAP...");
    try {
      const nextConfig = await api(`/api/gbm/models/${encodeURIComponent(modelId)}/shap/config`, { method: "GET" });
      if (seq !== configSeq || modelId !== String(nextModelId || "")) return;
      config = nextConfig;
      syncStateWithConfig();
      renderControlsAndLists();
      if (!config.has_shap) {
        renderEmpty(config.warnings?.[0] || "This GBM has no saved SHAP rows");
        return;
      }
      await refreshPlot();
    } catch (error) {
      if (seq !== configSeq) return;
      setNotice(error.message);
      renderEmpty("Stacked SHAP could not load");
    }
  }

  function bindStaticEvents(root) {
    if (root.dataset.gbmStackedShapBound === "1") return;
    root.dataset.gbmStackedShapBound = "1";
    root.addEventListener("click", handleClick);
    settingsOverflowCleanup = bindSettingsStripOverflowCue(root.querySelector("#gbmStackedShapControls"));
    setupMainDividerResize(root);
    layoutMediaQuery = window.matchMedia(STACKED_SHAP_STACKED_MEDIA);
    layoutMediaListener = () => syncLayoutVisibility(root);
    layoutMediaQuery.addEventListener?.("change", layoutMediaListener);
  }

  function handleClick(event) {
    const button = event.target.closest("button");
    if (!button || !rootNode()?.contains(button)) return;
    if (button.id === "gbmStackedShapToolbarToggle") {
      controlsCollapsed = !controlsCollapsed;
      syncLayoutVisibility(rootNode());
      return;
    }
    if (button.id === "gbmStackedShapSideToggle") {
      sidePanelCollapsed = !sidePanelCollapsed;
      syncLayoutVisibility(rootNode());
      return;
    }
    if (button.dataset.gbmStackedShapFeatureValue !== undefined) {
      selectFeature(button.dataset.gbmStackedShapFeatureValue);
      return;
    }
    if (button.dataset.gbmStackedShapFeatureSort) {
      state.featureSort = normaliseFeatureSort(button.dataset.gbmStackedShapFeatureSort);
      renderFeatureList();
      return;
    }
    if (button.dataset.gbmStackedShapSort) {
      state.xSort = normaliseXSort(button.dataset.gbmStackedShapSort);
      renderControls();
      refreshPlot();
      return;
    }
    if (button.dataset.gbmStackedShapTail !== undefined) {
      state.tailPercent = Number(button.dataset.gbmStackedShapTail);
      renderControls();
      refreshPlot();
      return;
    }
    if (button.dataset.gbmStackedShapFeatureCount) {
      state.numFeatures = normaliseFeatureCount(button.dataset.gbmStackedShapFeatureCount);
      renderControls();
      refreshPlot();
      return;
    }
    if (button.dataset.gbmStackedShapBandAction) {
      stepBanding(button.dataset.gbmStackedShapBandAction === "down" ? -1 : 1);
      return;
    }
    if (button.dataset.gbmStackedShapBandValue) {
      setBanding(Number(button.dataset.gbmStackedShapBandValue));
      return;
    }
  }

  function selectFeature(value) {
    const previous = state.modelFeature;
    state.modelFeature = String(value || "");
    if (!state.modelFeature) state.modelFeature = config?.default_feature_1 || features()[0]?.name || "";
    if (previous !== state.modelFeature) resetBanding();
    renderControlsAndLists();
    refreshPlot();
  }

  function preselectFeature(value) {
    const nextFeature = String(value || "");
    if (!nextFeature) return;
    const changed = state.modelFeature !== nextFeature;
    state.modelFeature = nextFeature;
    if (changed) resetBanding();
  }

  function setBanding(value) {
    clearPendingBanding();
    state.banding = normaliseBanding(value);
    state.bandingKey = currentBandingKey();
    renderControls();
    refreshPlot();
  }

  function stepBanding(direction) {
    clearPendingBanding();
    const current = Number(state.banding) || defaultBanding();
    const next = direction < 0
      ? [...BAND_STEPS].reverse().find((step) => step < current) || current
      : BAND_STEPS.find((step) => step > current) || current;
    setBanding(next);
  }

  async function refreshPlot() {
    if (!config?.has_shap || !state.modelFeature) return;
    const featureKey = state.modelFeature;
    const ensured = await ensureBanding();
    if (!ensured || featureKey !== state.modelFeature) return;
    const seq = ++plotSeq;
    setMessage("Computing Stacked SHAP...");
    try {
      const payload = await api(`/api/gbm/models/${encodeURIComponent(modelId)}/shap/stacked`, {
        method: "POST",
        body: JSON.stringify({
          model_feature: state.modelFeature,
          x_sort: state.xSort,
          tail_percent: Number(state.tailPercent),
          num_features: state.numFeatures,
          banding: Number(state.banding),
        }),
      });
      if (seq !== plotSeq) return;
      renderChart(payload);
      setNotice("");
      setMessage((payload.warnings || []).join(" "));
    } catch (error) {
      if (seq !== plotSeq) return;
      setNotice(error.message);
      setMessage("");
      renderEmpty("Choose a valid Stacked SHAP plot");
    }
  }

  function renderChart(payload) {
    const target = document.getElementById("gbmStackedShapChart");
    if (!target) return;
    ensureChart(target);
    const option = stackedShapChartOption(payload, chartTheme());
    chart.setOption(option, true);
    lastPayload = payload;
    scheduleChartResize({ flush: true });
  }

  function ensureChart(target) {
    if (chart) return;
    chart = window.echarts.init(target);
    if (typeof ResizeObserver === "function") {
      resizeObserver = new ResizeObserver((entries) => {
        const nextSize = resizeObserverEntrySize(entries?.[0]);
        if (sameChartSize(nextSize, settledObserverSize)) {
          settledObserverSize = null;
          return;
        }
        settledObserverSize = null;
        scheduleChartResize();
      });
      resizeObserver.observe(target);
    }
  }

  function renderControlsAndLists() {
    renderControls();
    renderFeatureList();
  }

  function renderControls() {
    const target = document.getElementById("gbmStackedShapControls");
    if (!target) return;
    target.innerHTML = `
      ${xSortControlHtml()}
      ${tailControlHtml()}
      ${featureCountControlHtml()}
      ${bandingControlHtml(selectedFeature())}
    `;
  }

  function renderFeatureList() {
    const list = document.getElementById("gbmStackedShapFeatureList");
    if (!list) return;
    syncFeatureSortControls();
    const ranks = featureRankMap();
    list.innerHTML = sortedFeatures(state.featureSort).map((feature) => featureButtonHtml(feature, ranks.get(feature.name))).join("");
  }

  function syncStateWithConfig() {
    const names = new Set(features().map((feature) => feature.name));
    if (modelId !== lastModelId) {
      lastModelId = modelId;
      state.featureSort = "importance";
      state.xSort = "alpha";
      state.tailPercent = 0;
      state.numFeatures = "all";
    }
    if (!names.has(state.modelFeature)) state.modelFeature = config?.default_feature_1 || features()[0]?.name || "";
    state.banding = normaliseBanding(state.banding);
  }

  function resetBanding() {
    clearPendingBanding();
    state.banding = defaultBanding();
    state.bandingKey = "";
  }

  function renderLoading(message) {
    renderControlsAndLists();
    renderEmpty(message);
  }

  function renderEmpty(message) {
    disposeChart();
    const target = document.getElementById("gbmStackedShapChart");
    if (!target) return;
    target.innerHTML = "";
    ensureChart(target);
    chart.setOption(stackedShapChartOption(null, chartTheme()), true);
    chart.setOption({ title: { text: message } }, false);
    setMessage(config?.warnings?.join(" ") || "");
  }

  function setMessage(message) {
    const node = document.getElementById("gbmStackedShapMessage");
    if (!node) return;
    const text = String(message || "");
    node.textContent = text;
    node.classList.toggle("hidden", !text);
  }

  function dispose() {
    configSeq += 1;
    plotSeq += 1;
    if (layoutMediaQuery && layoutMediaListener) {
      layoutMediaQuery.removeEventListener?.("change", layoutMediaListener);
    }
    layoutMediaQuery = null;
    layoutMediaListener = null;
    settingsOverflowCleanup?.();
    settingsOverflowCleanup = null;
    disposeChart();
    config = null;
  }

  function disposeChart() {
    if (chartResizeFrame !== null) cancelAnimationFrame(chartResizeFrame);
    chartResizeFrame = null;
    chartResizeFlush = false;
    settledObserverSize = null;
    resizeObserver?.disconnect();
    resizeObserver = null;
    lastPayload = null;
    if (chart) {
      chart.dispose();
      chart = null;
    }
  }

  function refreshTheme() {
    if (!chart || !lastPayload) return;
    chart.setOption(stackedShapChartOption(lastPayload, chartTheme()), true);
    resizeChart({ flush: true });
  }

  function xSortControlHtml() {
    return `
      <div class="control gbm-stacked-shap-sort-control">
        <h3>x-axis sort order</h3>
        <div class="segmented" role="group" aria-label="Stacked SHAP x-axis sort order">
          <button type="button" data-gbm-stacked-shap-sort="alpha" data-stable-label="A-Z" class="${state.xSort === "alpha" ? "active" : ""}">A-Z</button>
          <button type="button" data-gbm-stacked-shap-sort="descending" data-stable-label="Descending" class="${state.xSort === "descending" ? "active" : ""}">Descending</button>
        </div>
      </div>
    `;
  }

  function tailControlHtml() {
    return `
      <div class="control gbm-stacked-shap-tail-control">
        <h3>Tail grouping</h3>
        <div class="segmented" role="group" aria-label="Stacked SHAP tail grouping">
          ${TAIL_OPTIONS.map((option) => `
            <button type="button" data-gbm-stacked-shap-tail="${option.value}" data-stable-label="${escapeHtml(option.label)}" class="${Number(state.tailPercent) === option.value ? "active" : ""}">${escapeHtml(option.label)}</button>
          `).join("")}
        </div>
      </div>
    `;
  }

  function featureCountControlHtml() {
    return `
      <div class="control gbm-stacked-shap-feature-count-control">
        <h3>Num features to display</h3>
        <div class="segmented" role="group" aria-label="Stacked SHAP number of features to display">
          ${FEATURE_COUNT_OPTIONS.map((option) => `
            <button type="button" data-gbm-stacked-shap-feature-count="${option.value}" data-stable-label="${escapeHtml(option.label)}" class="${state.numFeatures === option.value ? "active" : ""}">${escapeHtml(option.label)}</button>
          `).join("")}
        </div>
      </div>
    `;
  }

  function bandingControlHtml(feature) {
    const numeric = feature && isNumericKind(feature.kind);
    const disabled = numeric ? "" : " disabled";
    const current = formatBanding(state.banding);
    return `
      <div class="control gbm-stacked-shap-banding-control ${numeric ? "" : "disabled"}">
        <h3>Banding ${numeric ? `<span>(${escapeHtml(current)})</span>` : ""}</h3>
        <div class="segmented" role="group" aria-label="Stacked SHAP numeric banding">
          <button type="button" class="gbm-stacked-shap-band-step" data-gbm-stacked-shap-band-action="down" data-stable-label="&lt;"${disabled}>&lt;</button>
          ${BAND_BUTTONS.map((value) => `
            <button type="button" data-gbm-stacked-shap-band-value="${value}" data-stable-label="${value}" class="${Number(state.banding) === value ? "active" : ""}"${disabled}>${value}</button>
          `).join("")}
          <button type="button" class="gbm-stacked-shap-band-step" data-gbm-stacked-shap-band-action="up" data-stable-label="&gt;"${disabled}>&gt;</button>
        </div>
      </div>
    `;
  }

  function featureButtonHtml(feature, rank) {
    const active = state.modelFeature === feature.name;
    return `
      <button class="feature ${active ? "active" : ""}" type="button" data-gbm-stacked-shap-feature-value="${escapeHtml(feature.name)}">
        <span>${escapeHtml(feature.name)}</span><span class="kind">${escapeHtml(featureDetailLabel(feature, rank))}</span>
      </button>
    `;
  }

  function featureDetailLabel(feature, rank) {
    const prefix = Number.isFinite(Number(rank)) ? `Rank ${rank}` : "";
    const meanAbsShap = formatMeanAbsShap(feature?.mean_abs_shap);
    return prefix && meanAbsShap ? `${prefix} · ${meanAbsShap}` : (prefix || meanAbsShap || "");
  }

  function syncFeatureSortControls() {
    rootNode()?.querySelectorAll("[data-gbm-stacked-shap-feature-sort]").forEach((button) => {
      button.classList.toggle("active", button.dataset.gbmStackedShapFeatureSort === state.featureSort);
    });
  }

  function sortedFeatures(sortMode) {
    const rows = [...features()];
    if (sortMode === "alpha") {
      rows.sort((a, b) => a.name.localeCompare(b.name, undefined, { sensitivity: "base" }));
    } else {
      rows.sort((a, b) => (
        featureImportanceValue(b) - featureImportanceValue(a)
      ) || a.name.localeCompare(b.name, undefined, { sensitivity: "base" }));
    }
    return rows;
  }

  function featureRankMap() {
    return new Map(sortedFeatures("importance").map((feature, index) => [feature.name, index + 1]));
  }

  function featureImportanceValue(feature) {
    const value = Number(feature?.mean_abs_shap);
    return Number.isFinite(value) ? value : 0;
  }

  function features() {
    return Array.isArray(config?.features) ? config.features : [];
  }

  function selectedFeature() {
    return features().find((feature) => feature.name === state.modelFeature) || null;
  }

  function defaultBanding() {
    return normaliseBanding(null);
  }

  function normaliseBanding(value) {
    const number = Number(value);
    if (Number.isFinite(number) && number > 0) return Number(number.toPrecision(12));
    return 1;
  }

  function currentBandingKey() {
    const feature = selectedFeature();
    return feature && isNumericKind(feature.kind) ? `${modelId}:${feature.name}` : "";
  }

  function clearPendingBanding() {
    state.bandingPendingKey = "";
  }

  async function ensureBanding() {
    const feature = selectedFeature();
    if (!feature || !isNumericKind(feature.kind)) return true;
    const key = currentBandingKey();
    if (!key || state.bandingKey === key) return true;
    if (state.bandingPendingKey === key) return false;
    const seq = (state.bandingSeq || 0) + 1;
    state.bandingSeq = seq;
    state.bandingPendingKey = key;
    renderControls();
    setMessage("Estimating Stacked SHAP banding...");
    try {
      const data = await api("/api/banding/suggestion", {
        method: "POST",
        body: JSON.stringify({ source: "dataset", feature: feature.name }),
      });
      if (state.bandingSeq !== seq || state.bandingPendingKey !== key || currentBandingKey() !== key) return false;
      state.banding = normaliseBanding(data.band_suggestion);
      state.bandingKey = key;
      clearPendingBanding();
      renderControls();
      return true;
    } catch (error) {
      if (state.bandingSeq !== seq || state.bandingPendingKey !== key || currentBandingKey() !== key) return false;
      state.banding = defaultBanding();
      state.bandingKey = key;
      clearPendingBanding();
      renderControls();
      setMessage(`Banding estimate failed; using ${formatBanding(state.banding)}. ${error.message}`);
      return true;
    }
  }

  function normaliseXSort(value) {
    return String(value || "").toLowerCase() === "descending" ? "descending" : "alpha";
  }

  function normaliseFeatureSort(value) {
    return String(value || "").toLowerCase() === "alpha" ? "alpha" : "importance";
  }

  function normaliseFeatureCount(value) {
    const text = String(value || "all").toLowerCase();
    return FEATURE_COUNT_OPTIONS.some((option) => option.value === text) ? text : "all";
  }

  function isNumericKind(kind) {
    return kind === "numeric" || kind === "integer";
  }

  function formatBanding(value) {
    const number = Number(value);
    if (!Number.isFinite(number)) return "1";
    return Number(number.toPrecision(12)).toString();
  }

  function formatMeanAbsShap(value) {
    const number = Number(value);
    return Number.isFinite(number) ? number.toFixed(4) : "";
  }

  function chartTheme() {
    const body = document.body;
    const style = getComputedStyle(body);
    return {
      panel: style.getPropertyValue("--panel").trim() || "transparent",
      text: style.getPropertyValue("--text").trim() || "#334155",
      muted: style.getPropertyValue("--muted").trim() || "#64748b",
      line: style.getPropertyValue("--line").trim() || "#cbd5e1",
      grid: body.classList.contains("dark") ? "#243044" : "#e5e7eb",
      zero: body.classList.contains("dark") ? "#cbd5e1" : "#334155",
    };
  }

  function rootNode() {
    return document.getElementById("gbmStackedShapRoot");
  }

  function syncLayoutVisibility(root = rootNode(), { resize = true } = {}) {
    if (!root) return;
    const toolbar = root.querySelector("#gbmStackedShapControls");
    const toolbarToggle = root.querySelector("#gbmStackedShapToolbarToggle");
    const side = root.querySelector("#gbmStackedShapSide");
    const sideToggle = root.querySelector("#gbmStackedShapSideToggle");
    const workspace = root.querySelector(".gbm-stacked-shap-workspace");
    const resizer = root.querySelector("#gbmStackedShapMainResizer");
    const stacked = window.matchMedia(STACKED_SHAP_STACKED_MEDIA).matches;

    if (controlsCollapsed && toolbar?.contains(document.activeElement)) toolbarToggle?.focus();
    if (sidePanelCollapsed && side?.contains(document.activeElement)) sideToggle?.focus();

    setElementCollapsed(toolbar, controlsCollapsed);
    setElementCollapsed(side, sidePanelCollapsed);
    workspace?.classList.toggle("gbm-stacked-shap-side-collapsed", sidePanelCollapsed);

    const splitterHidden = sidePanelCollapsed || stacked;
    if (resizer) {
      resizer.hidden = splitterHidden;
      resizer.toggleAttribute("inert", splitterHidden);
      resizer.tabIndex = splitterHidden ? -1 : 0;
      resizer.setAttribute("aria-hidden", String(splitterHidden));
    }

    syncToggleButton(
      toolbarToggle,
      !controlsCollapsed,
      controlsCollapsed ? "Show Stacked SHAP control row" : "Hide Stacked SHAP control row",
    );
    syncToggleButton(
      sideToggle,
      !sidePanelCollapsed,
      sidePanelCollapsed ? "Show Stacked SHAP feature chooser" : "Hide Stacked SHAP feature chooser",
    );
    if (!stacked) setMainSideWidth(root, sidePanelWidth, { resize: false });
    if (resize) scheduleChartResize({ flush: true });
  }

  function setElementCollapsed(element, collapsed) {
    if (!element) return;
    element.classList.toggle("hidden", collapsed);
    element.hidden = collapsed;
    element.toggleAttribute("inert", collapsed);
    element.setAttribute("aria-hidden", String(collapsed));
  }

  function syncToggleButton(button, expanded, label) {
    if (!button) return;
    button.setAttribute("aria-expanded", String(expanded));
    button.setAttribute("aria-label", label);
    button.title = label;
  }

  function resizeChart({ flush = false } = {}) {
    if (!chart) return;
    chart.resize();
    if (flush) {
      chart.getZr?.().flush?.();
      settledObserverSize = currentChartSize();
    }
  }

  function currentChartSize() {
    const target = document.getElementById("gbmStackedShapChart");
    return target ? { width: target.clientWidth, height: target.clientHeight } : null;
  }

  function resizeObserverEntrySize(entry) {
    if (!entry) return null;
    const box = entry.contentBoxSize;
    const size = Array.isArray(box) ? box[0] : box;
    if (size) return { width: size.inlineSize, height: size.blockSize };
    return { width: entry.contentRect?.width, height: entry.contentRect?.height };
  }

  function sameChartSize(left, right) {
    if (!left || !right) return false;
    return Math.abs(Number(left.width) - Number(right.width)) < 0.5
      && Math.abs(Number(left.height) - Number(right.height)) < 0.5;
  }

  function scheduleChartResize({ flush = false } = {}) {
    chartResizeFlush ||= Boolean(flush);
    if (chartResizeFrame !== null) return;
    chartResizeFrame = requestAnimationFrame(() => {
      chartResizeFrame = null;
      const shouldFlush = chartResizeFlush;
      chartResizeFlush = false;
      resizeChart({ flush: shouldFlush });
    });
  }

  function resize() {
    const root = rootNode();
    if (root && !window.matchMedia(STACKED_SHAP_STACKED_MEDIA).matches) {
      setMainSideWidth(root, sidePanelWidth, { resize: false });
    }
    resizeChart({ flush: true });
  }

  function setupMainDividerResize(root) {
    const side = root.querySelector(".gbm-stacked-shap-side");
    const resizer = root.querySelector("#gbmStackedShapMainResizer");
    if (!side || !resizer) return;
    let dragging = false;
    let startX = 0;
    let startWidth = sidePanelWidth;

    resizer.addEventListener("pointerdown", (event) => {
      if (window.matchMedia(STACKED_SHAP_STACKED_MEDIA).matches || sidePanelCollapsed) return;
      event.preventDefault();
      dragging = true;
      startX = event.clientX;
      startWidth = side.getBoundingClientRect().width || sidePanelWidth;
      resizer.classList.add("dragging");
      document.body.classList.add("resizing-chart-controls");
      resizer.setPointerCapture(event.pointerId);
      window.getSelection()?.removeAllRanges();
    });
    resizer.addEventListener("pointermove", (event) => {
      if (!dragging) return;
      event.preventDefault();
      setMainSideWidth(root, startWidth + event.clientX - startX);
    });
    const finishDrag = (event) => {
      if (!dragging) return;
      dragging = false;
      resizer.classList.remove("dragging");
      document.body.classList.remove("resizing-chart-controls");
      window.getSelection()?.removeAllRanges();
      if (event.pointerId !== undefined) {
        try {
          resizer.releasePointerCapture(event.pointerId);
        } catch (_) {
        }
      }
      scheduleChartResize({ flush: true });
    };
    resizer.addEventListener("pointerup", finishDrag);
    resizer.addEventListener("pointercancel", finishDrag);

    resizer.addEventListener("keydown", (event) => {
      if (window.matchMedia(STACKED_SHAP_STACKED_MEDIA).matches || sidePanelCollapsed) return;
      const bounds = mainSideWidthBounds(root);
      let nextWidth = null;
      if (event.key === "ArrowLeft") nextWidth = sidePanelWidth - STACKED_SHAP_SPLITTER_KEY_STEP;
      if (event.key === "ArrowRight") nextWidth = sidePanelWidth + STACKED_SHAP_SPLITTER_KEY_STEP;
      if (event.key === "Home") nextWidth = bounds.min;
      if (event.key === "End") nextWidth = bounds.max;
      if (nextWidth === null) return;
      event.preventDefault();
      setMainSideWidth(root, nextWidth, { flush: true });
    });
    setMainSideWidth(root, sidePanelWidth, { resize: false });
  }

  function mainSideWidthBounds(root = rootNode()) {
    const workspace = root?.querySelector(".gbm-stacked-shap-workspace");
    const availableWidth = workspace?.getBoundingClientRect().width || window.innerWidth;
    const max = Math.max(
      STACKED_SHAP_SIDE_MIN_WIDTH,
      Math.min(STACKED_SHAP_SIDE_MAX_WIDTH, availableWidth - STACKED_SHAP_CHART_MIN_WIDTH),
    );
    return { min: STACKED_SHAP_SIDE_MIN_WIDTH, max };
  }

  function setMainSideWidth(root, rawWidth, { resize = true, flush = false } = {}) {
    const resizer = root?.querySelector("#gbmStackedShapMainResizer");
    const bounds = mainSideWidthBounds(root);
    const width = Math.min(
      Math.max(Number(rawWidth) || STACKED_SHAP_SIDE_DEFAULT_WIDTH, bounds.min),
      bounds.max,
    );
    sidePanelWidth = width;
    root.style.setProperty("--gbm-stacked-shap-side-width", `${Math.round(width)}px`);
    resizer?.setAttribute("aria-valuemin", String(Math.round(bounds.min)));
    resizer?.setAttribute("aria-valuemax", String(Math.round(bounds.max)));
    resizer?.setAttribute("aria-valuenow", String(Math.round(width)));
    if (resize) scheduleChartResize({ flush });
    return width;
  }

  return {
    dispose,
    preselectFeature,
    refreshTheme,
    render,
    resize,
    shellHtml,
  };
}

function makeBandSteps() {
  const steps = [];
  for (let exponent = -8; exponent <= 12; exponent += 1) {
    const multiplier = 10 ** exponent;
    steps.push(1 * multiplier, 2 * multiplier, 5 * multiplier);
  }
  steps.push(4, 7, 12);
  return [...new Set(steps)].sort((a, b) => a - b);
}
