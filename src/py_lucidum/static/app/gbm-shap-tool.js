import { emptyOption, ensureShapChartLibraries, shapChartOption } from "./gbm-shap-chart.js";
import { isEchartsTargetReady } from "./shared/echarts-gl.js";
import { bindSettingsStripOverflowCue } from "./shared/settings-strip.js";

const BAND_STEPS = makeBandSteps();
const BAND_BUTTONS = [0.01, 0.1, 1, 5, 10];
const TAIL_OPTIONS = [
  { value: 0, label: "-" },
  { value: 0.1, label: "0.1%" },
  { value: 0.5, label: "0.5%" },
  { value: 1, label: "1%" },
  { value: 2, label: "2%" },
  { value: 5, label: "5%" },
];
const SHAP_SIDE_DEFAULT_WIDTH = 320;
const SHAP_SIDE_MIN_WIDTH = 240;
const SHAP_SIDE_MAX_WIDTH = 560;
const SHAP_CHART_MIN_WIDTH = 420;
const SHAP_SPLITTER_KEY_STEP = 10;
const SHAP_CHOOSER_MIN_HEIGHT = 96;
const SHAP_CHOOSER_DIVIDER_HEIGHT = 18;
const SHAP_STACKED_MEDIA = "(max-width: 900px)";

export function createGbmShapTool({ api, escapeHtml, setNotice, showClipboardToast = () => {} }) {
  let modelId = "";
  let lastModelId = "";
  let config = null;
  let chart = null;
  let resizeObserver = null;
  let lastPayload = null;
  let configSeq = 0;
  let plotSeq = 0;
  let pendingLegendState = null;
  let chooserFeatureHeight = null;
  let sidePanelWidth = SHAP_SIDE_DEFAULT_WIDTH;
  let controlsCollapsed = true;
  let sidePanelCollapsed = false;
  let feature2Collapsed = true;
  let chartResizeFrame = null;
  let chartResizeFlush = false;
  let settledObserverSize = null;
  let observedChartTarget = null;
  let pendingChartRender = null;
  let pendingEmptyMessage = null;
  let chartRenderPending = false;
  let layoutMediaQuery = null;
  let layoutMediaListener = null;
  let settingsOverflowCleanup = null;
  const state = {
    feature1: "",
    feature2: "",
    sort1: "importance",
    sort2: "importance",
    search1: "",
    search2: "",
    banding1: 1,
    banding2: 1,
    bandingKey1: "",
    bandingKey2: "",
    bandingPendingKey1: "",
    bandingPendingKey2: "",
    bandingSeq1: 0,
    bandingSeq2: 0,
    factor1: false,
    factor2: false,
    tailPercent: 1,
    rescale: "-",
  };

  function shellHtml() {
    const toolbarHidden = controlsCollapsed ? " hidden inert aria-hidden=\"true\"" : "";
    const sideHidden = sidePanelCollapsed ? " hidden inert aria-hidden=\"true\"" : "";
    const feature2Hidden = feature2Collapsed ? " hidden inert aria-hidden=\"true\"" : "";
    return `
      <div id="gbmShapRoot" class="gbm-shap-view">
        <div id="gbmShapControls" class="gbm-shap-controls toolbar app-control-strip app-settings-strip${controlsCollapsed ? " hidden" : ""}"${toolbarHidden}></div>
        <div class="gbm-shap-workspace${sidePanelCollapsed ? " gbm-shap-side-collapsed" : ""}${feature2Collapsed ? " gbm-shap-feature2-collapsed" : ""}">
          <aside id="gbmShapSide" class="gbm-shap-side${sidePanelCollapsed ? " hidden" : ""}"${sideHidden}>
            ${featureChooserHtml(1, "Feature 1")}
            <div class="gbm-shap-chooser-divider-row">
              <div id="gbmShapChooserDivider" class="gbm-shap-chooser-divider app-resizer app-resizer--horizontal" role="separator" aria-orientation="horizontal" aria-label="Resize SHAP feature choosers" tabindex="0"></div>
              <button id="gbmShapFeature2Toggle" class="gbm-shap-feature2-toggle" type="button" aria-controls="gbmShapFeatureSection2" aria-expanded="${String(!feature2Collapsed)}">
                <span class="gbm-shap-feature2-toggle-icon" aria-hidden="true"></span>
              </button>
            </div>
            ${featureChooserHtml(2, "Feature 2", feature2Hidden)}
          </aside>
          <div id="gbmShapMainResizer" class="gbm-shap-main-resizer app-resizer app-resizer--vertical" role="separator" aria-orientation="vertical" aria-label="Resize SHAP feature chooser" tabindex="0"></div>
          <section class="gbm-shap-main">
            <div class="gbm-shap-workspace-controls">
              <button id="gbmShapSideToggle" class="gbm-shap-overlay-button app-control-button" type="button" aria-controls="gbmShapSide" aria-expanded="${String(!sidePanelCollapsed)}">
                <svg class="gbm-shap-toggle-icon gbm-shap-chevron-horizontal" viewBox="0 0 24 24" aria-hidden="true" focusable="false">
                  <path d="m15 18-6-6 6-6"></path>
                </svg>
              </button>
              <button id="gbmShapToolbarToggle" class="gbm-shap-overlay-button app-control-button" type="button" aria-controls="gbmShapControls" aria-expanded="${String(!controlsCollapsed)}">
                <svg class="gbm-shap-toggle-icon gbm-shap-chevron-vertical" viewBox="0 0 24 24" aria-hidden="true" focusable="false">
                  <path d="m18 15-6-6-6 6"></path>
                </svg>
              </button>
              <button id="gbmShapCopyButton" class="gbm-shap-overlay-button app-control-button" type="button" aria-label="Copy SHAP chart" title="Copy SHAP chart">
                <svg class="gbm-shap-toggle-icon" viewBox="0 0 24 24" aria-hidden="true" focusable="false">
                  <rect x="8" y="8" width="10" height="10" rx="1.5"></rect>
                  <path d="M6 14H5.5A1.5 1.5 0 0 1 4 12.5v-7A1.5 1.5 0 0 1 5.5 4h7A1.5 1.5 0 0 1 14 5.5V6"></path>
                </svg>
              </button>
            </div>
            <div class="gbm-shap-chart-shell">
              <div id="gbmShapMessage" class="gbm-shap-message hidden"></div>
              <div id="gbmShapChart" class="gbm-shap-chart" aria-label="GBM SHAP plot"></div>
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
    observeChartTarget(document.getElementById("gbmShapChart"));
    syncLayoutVisibility(root, { resize: false });
    if (!modelId) {
      config = null;
      clearPendingLegendState();
      renderEmpty("No active GBM selected");
      return;
    }
    if (modelId !== lastModelId) config = null;
    const seq = ++configSeq;
    renderLoading("Loading SHAP...");
    try {
      const nextConfig = await api(`/api/gbm/models/${encodeURIComponent(modelId)}/shap/config`, { method: "GET" });
      if (seq !== configSeq || modelId !== String(nextModelId || "")) return;
      config = nextConfig;
      const syncResult = syncStateWithConfig();
      if (syncResult.featureFallback) clearPendingLegendState();
      renderControlsAndLists();
      if (!config.has_shap) {
        clearPendingLegendState();
        renderEmpty(config.warnings?.[0] || "This GBM has no saved SHAP rows");
        return;
      }
      await refreshPlot();
    } catch (error) {
      if (seq !== configSeq) return;
      clearPendingLegendState();
      setNotice(error.message);
      renderEmpty("SHAP could not load");
    }
  }

  function bindStaticEvents(root) {
    if (root.dataset.gbmShapBound === "1") return;
    root.dataset.gbmShapBound = "1";
    root.addEventListener("click", handleClick);
    root.addEventListener("input", handleInput);
    settingsOverflowCleanup = bindSettingsStripOverflowCue(root.querySelector("#gbmShapControls"));
    setupChooserDividerResize(root);
    setupMainDividerResize(root);
    layoutMediaQuery = window.matchMedia(SHAP_STACKED_MEDIA);
    layoutMediaListener = () => syncLayoutVisibility(root);
    layoutMediaQuery.addEventListener?.("change", layoutMediaListener);
  }

  function handleClick(event) {
    const button = event.target.closest("button");
    if (!button || !rootNode()?.contains(button)) return;
    if (button.id === "gbmShapToolbarToggle") {
      controlsCollapsed = !controlsCollapsed;
      syncLayoutVisibility(rootNode());
      return;
    }
    if (button.id === "gbmShapSideToggle") {
      sidePanelCollapsed = !sidePanelCollapsed;
      syncLayoutVisibility(rootNode());
      return;
    }
    if (button.id === "gbmShapCopyButton") {
      copyChartToClipboard();
      return;
    }
    if (button.id === "gbmShapFeature2Toggle") {
      feature2Collapsed = !feature2Collapsed;
      syncLayoutVisibility(rootNode());
      return;
    }
    if (button.dataset.gbmShapSort) {
      state[`sort${button.dataset.gbmShapFeature}`] = button.dataset.gbmShapSort;
      renderFeatureLists();
      return;
    }
    if (button.dataset.gbmShapSearchClear) {
      const index = button.dataset.gbmShapSearchClear;
      state[`search${index}`] = "";
      const input = document.getElementById(`gbmShapFeatureSearch${index}`);
      if (input) input.value = "";
      renderFeatureLists();
      input?.focus();
      return;
    }
    if (button.dataset.gbmShapFeatureValue !== undefined) {
      selectFeature(Number(button.dataset.gbmShapFeature), button.dataset.gbmShapFeatureValue);
      return;
    }
    if (button.dataset.gbmShapBandAction) {
      stepBanding(Number(button.dataset.gbmShapFeature), button.dataset.gbmShapBandAction === "down" ? -1 : 1);
      return;
    }
    if (button.dataset.gbmShapBandValue) {
      setBanding(Number(button.dataset.gbmShapFeature), Number(button.dataset.gbmShapBandValue));
      return;
    }
    if (button.dataset.gbmShapTail !== undefined) {
      state.tailPercent = Number(button.dataset.gbmShapTail);
      renderControls();
      refreshPlot();
      return;
    }
    if (button.dataset.gbmShapRescale !== undefined) {
      state.rescale = normaliseRescale(button.dataset.gbmShapRescale);
      renderControls();
      refreshPlot();
      return;
    }
    if (button.dataset.gbmShapFactor) {
      const index = button.dataset.gbmShapFactor;
      state[`factor${index}`] = !state[`factor${index}`];
      clearPendingLegendState();
      renderControls();
      refreshPlot();
    }
  }

  function handleInput(event) {
    const input = event.target;
    if (input?.dataset?.gbmShapSearch) {
      state[`search${input.dataset.gbmShapSearch}`] = input.value;
      renderFeatureLists();
    }
  }

  function selectFeature(index, value) {
    const key = `feature${index}`;
    const previous = state[key];
    state[key] = String(value || "");
    if (index === 1 && !state.feature1) {
      state.feature1 = config?.default_feature_1 || features()[0]?.name || "";
    }
    if (previous !== state[key]) {
      clearPendingLegendState();
      resetBanding(index);
    }
    renderControlsAndLists();
    refreshPlot();
  }

  function preselectFeatures(feature1, feature2 = "") {
    const nextFeature1 = String(feature1 || "");
    const nextFeature2 = String(feature2 || "");
    if (!nextFeature1) return;
    const changed = state.feature1 !== nextFeature1 || state.feature2 !== nextFeature2;
    state.feature1 = nextFeature1;
    state.feature2 = nextFeature2;
    sidePanelCollapsed = false;
    if (nextFeature2) feature2Collapsed = false;
    state.search1 = "";
    state.search2 = "";
    if (changed) {
      clearPendingLegendState();
      resetBanding(1);
      resetBanding(2);
    }
  }

  function setBanding(index, value) {
    clearPendingBanding(index);
    state[`banding${index}`] = normaliseBanding(value);
    state[`bandingKey${index}`] = currentBandingKey(index);
    renderControls();
    refreshPlot();
  }

  function stepBanding(index, direction) {
    clearPendingBanding(index);
    const current = Number(state[`banding${index}`]) || defaultBanding();
    const next = direction < 0
      ? [...BAND_STEPS].reverse().find((step) => step < current) || current
      : BAND_STEPS.find((step) => step > current) || current;
    setBanding(index, next);
  }

  async function refreshPlot() {
    if (!config?.has_shap || !state.feature1) return;
    const featureKey = `${state.feature1}\n${state.feature2 || ""}`;
    const ensured = await Promise.all([ensureBanding(1), ensureBanding(2)]);
    if (!ensured.every(Boolean) || featureKey !== `${state.feature1}\n${state.feature2 || ""}`) return;
    const seq = ++plotSeq;
    setMessage("Computing SHAP plot...");
    try {
      const payload = await api(`/api/gbm/models/${encodeURIComponent(modelId)}/shap/plot`, {
        method: "POST",
        body: JSON.stringify({
          feature_1: state.feature1,
          feature_2: state.feature2 || "None",
          banding_1: Number(state.banding1),
          banding_2: Number(state.banding2),
          tail_percent: Number(state.tailPercent),
          factor_1: Boolean(state.factor1),
          factor_2: Boolean(state.factor2),
          rescale: state.rescale,
        }),
      });
      if (seq !== plotSeq) return;
      await renderChart(payload, seq);
      if (seq !== plotSeq) return;
      setNotice("");
      setMessage((payload.warnings || []).join(" "));
    } catch (error) {
      if (seq !== plotSeq) return;
      setNotice(error.message);
      setMessage("");
      renderEmpty("Choose a valid SHAP plot");
    }
  }

  async function renderChart(payload, seq) {
    const target = document.getElementById("gbmShapChart");
    if (!target) return;
    const loadedSurfaceLibrary = await ensureShapChartLibraries(payload.plot_type);
    if (seq !== plotSeq) return;
    pendingEmptyMessage = null;
    pendingChartRender = { payload, seq, loadedSurfaceLibrary };
    observeChartTarget(target);
    await flushPendingChartRender();
  }

  async function flushPendingChartRender() {
    if (chartRenderPending || !pendingChartRender) return false;
    const work = pendingChartRender;
    const target = document.getElementById("gbmShapChart");
    if (
      !target
      || work.seq !== plotSeq
      || target !== observedChartTarget
      || !isEchartsTargetReady(target)
    ) {
      return false;
    }
    pendingChartRender = null;
    chartRenderPending = true;
    const { payload, seq, loadedSurfaceLibrary } = work;
    const isSurface = payload.plot_type === "surface";
    const previousPlotType = lastPayload?.plot_type || "";
    const previousOption = chart?.getOption?.();
    const previousLegendEntries = legendEntryNames(previousOption);
    const previousLegendSelection = legendSelection(previousOption, previousLegendEntries);
    try {
      if (isSurface && (loadedSurfaceLibrary || previousPlotType !== "surface")) {
        disposeChart();
        await nextAnimationFrame();
        if (!chartWorkIsCurrent(target, seq)) {
          retainPendingChartRender(work);
          return false;
        }
      }
      if (!ensureChart(target)) {
        retainPendingChartRender(work);
        return false;
      }
      const option = shapChartOption(payload, chartTheme());
      const nextLegendEntries = legendEntryNames(option);
      const pendingLegendSelection = pendingLegendSelectionForPayload(payload, nextLegendEntries);
      if (previousPlotType === payload.plot_type && sameEntries(previousLegendEntries, nextLegendEntries)) {
        applyLegendSelection(option, previousLegendSelection, nextLegendEntries);
        clearPendingLegendState();
      } else if (pendingLegendSelection) {
        applyLegendSelection(option, pendingLegendSelection, nextLegendEntries);
        clearPendingLegendState();
      } else if (pendingLegendState) {
        clearPendingLegendState();
      }
      try {
        chart.setOption(option, true);
      } catch (error) {
        if (!isSurface || !isSurfaceLayoutError(error)) throw error;
        disposeChart();
        await nextAnimationFrame();
        if (!chartWorkIsCurrent(target, seq) || !ensureChart(target)) {
          retainPendingChartRender(work);
          return false;
        }
        await nextAnimationFrame();
        if (!chartWorkIsCurrent(target, seq)) {
          retainPendingChartRender(work);
          return false;
        }
        chart.setOption(option, true);
      }
      lastPayload = payload;
      scheduleChartResize({ flush: true });
      return true;
    } finally {
      chartRenderPending = false;
      if ((pendingChartRender || pendingEmptyMessage !== null) && isEchartsTargetReady(observedChartTarget)) {
        scheduleChartResize({ flush: true });
      }
    }
  }

  function chartWorkIsCurrent(target, seq) {
    return seq === plotSeq
      && target === document.getElementById("gbmShapChart")
      && target === observedChartTarget
      && isEchartsTargetReady(target);
  }

  function retainPendingChartRender(work) {
    if (!work || work.seq !== plotSeq) return;
    if (!pendingChartRender || pendingChartRender.seq <= work.seq) {
      pendingChartRender = work;
    }
  }

  function resumePendingChartRender() {
    void flushPendingChartRender().catch((error) => {
      setNotice(error.message);
      setMessage("");
      renderEmpty("Choose a valid SHAP plot");
    });
  }

  function ensureChart(target) {
    observeChartTarget(target);
    if (chart) return true;
    if (!isEchartsTargetReady(target)) return false;
    chart = window.echarts.init(target);
    return true;
  }

  function observeChartTarget(target) {
    if (!target || observedChartTarget === target) return;
    resizeObserver?.disconnect();
    resizeObserver = null;
    observedChartTarget = target;
    if (typeof ResizeObserver !== "function") return;
    resizeObserver = new ResizeObserver((entries) => {
      const nextSize = resizeObserverEntrySize(entries?.[0]);
      if (sameChartSize(nextSize, settledObserverSize)) {
        settledObserverSize = null;
        return;
      }
      settledObserverSize = null;
      if (isEchartsTargetReady(observedChartTarget)) scheduleChartResize();
    });
    resizeObserver.observe(target);
  }

  function disconnectChartTargetObserver() {
    resizeObserver?.disconnect();
    resizeObserver = null;
    observedChartTarget = null;
  }

  function renderPendingEmptyState() {
    if (pendingEmptyMessage === null) return false;
    const target = document.getElementById("gbmShapChart");
    if (!target || target !== observedChartTarget || !isEchartsTargetReady(target)) return false;
    const message = pendingEmptyMessage;
    pendingEmptyMessage = null;
    if (!ensureChart(target)) {
      pendingEmptyMessage = message;
      return false;
    }
    chart.setOption(emptyOption(message, chartTheme()), true);
    scheduleChartResize({ flush: true });
    return true;
  }

  function renderControlsAndLists() {
    renderControls();
    renderFeatureLists();
  }

  function renderControls() {
    const target = document.getElementById("gbmShapControls");
    if (!target) return;
    const feature1 = selectedFeature(1);
    const feature2 = selectedFeature(2);
    target.innerHTML = `
      ${bandingControlHtml(1, feature1)}
      ${bandingControlHtml(2, feature2)}
      ${tailControlHtml()}
      ${rescaleControlHtml()}
      ${factorControlHtml(feature1, feature2)}
    `;
  }

  function renderFeatureLists() {
    syncChooserControls(1);
    syncChooserControls(2);
    renderFeatureList(1);
    renderFeatureList(2);
  }

  function renderFeatureList(index) {
    const list = document.getElementById(`gbmShapFeatureList${index}`);
    if (!list) return;
    const selected = state[`feature${index}`];
    const query = String(state[`search${index}`] || "").trim().toLowerCase();
    const rows = sortedFeatures(state[`sort${index}`]);
    const ranks = featureRankMap();
    const buttons = [];
    if (index === 2 && (!query || "none".includes(query) || "no second feature".includes(query) || "off".includes(query))) {
      buttons.push(featureButtonHtml(index, "None", "", "off", "expected-none-option", !selected));
    }
    for (const feature of rows) {
      if (query && !feature.name.toLowerCase().includes(query)) continue;
      buttons.push(
        featureButtonHtml(
          index,
          feature.name,
          feature.name,
          featureImportanceLabel(feature, ranks.get(feature.name)),
          "",
          selected === feature.name,
        )
      );
    }
    list.innerHTML = buttons.join("");
  }

  function syncChooserControls(index) {
    const input = document.getElementById(`gbmShapFeatureSearch${index}`);
    if (input && input.value !== state[`search${index}`]) input.value = state[`search${index}`];
    rootNode()?.querySelectorAll(`[data-gbm-shap-feature="${index}"][data-gbm-shap-sort]`).forEach((button) => {
      button.classList.toggle("active", button.dataset.gbmShapSort === state[`sort${index}`]);
    });
  }

  function syncStateWithConfig() {
    const nextFeatures = features();
    const names = new Set(nextFeatures.map((feature) => feature.name));
    let featureFallback = false;
    if (modelId !== lastModelId) {
      state.factor1 = false;
      state.factor2 = false;
      state.search1 = "";
      state.search2 = "";
      lastModelId = modelId;
    }
    if (!names.has(state.feature1)) {
      state.feature1 = firstFeatureNameForChooser(1);
      featureFallback = true;
    }
    if (state.feature2 && !names.has(state.feature2)) {
      state.feature2 = "";
      featureFallback = true;
    }
    state.banding1 = normaliseBanding(state.banding1);
    state.banding2 = normaliseBanding(state.banding2);
    return { featureFallback };
  }

  function resetBanding(index) {
    clearPendingBanding(index);
    state[`banding${index}`] = defaultBanding();
    state[`bandingKey${index}`] = "";
  }

  function firstFeatureNameForChooser(index) {
    return sortedFeatures(state[`sort${index}`])[0]?.name || "";
  }

  function renderLoading(message) {
    renderControlsAndLists();
    renderEmpty(message);
  }

  function renderEmpty(message) {
    pendingChartRender = null;
    disposeChart();
    const target = document.getElementById("gbmShapChart");
    if (!target) return;
    target.innerHTML = "";
    pendingEmptyMessage = String(message || "");
    observeChartTarget(target);
    if (isEchartsTargetReady(target)) renderPendingEmptyState();
    setMessage(config?.warnings?.join(" ") || "");
  }

  function setMessage(message) {
    const node = document.getElementById("gbmShapMessage");
    if (!node) return;
    const text = String(message || "");
    node.textContent = text;
    node.classList.toggle("hidden", !text);
  }

  function dispose() {
    configSeq += 1;
    plotSeq += 1;
    snapshotLegendState();
    if (layoutMediaQuery && layoutMediaListener) {
      layoutMediaQuery.removeEventListener?.("change", layoutMediaListener);
    }
    layoutMediaQuery = null;
    layoutMediaListener = null;
    settingsOverflowCleanup?.();
    settingsOverflowCleanup = null;
    pendingChartRender = null;
    pendingEmptyMessage = null;
    disposeChart();
    disconnectChartTargetObserver();
    config = null;
  }

  function disposeChart() {
    if (chartResizeFrame !== null) cancelAnimationFrame(chartResizeFrame);
    chartResizeFrame = null;
    chartResizeFlush = false;
    settledObserverSize = null;
    lastPayload = null;
    if (chart) {
      chart.dispose();
      chart = null;
    }
  }

  function refreshTheme() {
    if (!lastPayload) return;
    pendingEmptyMessage = null;
    pendingChartRender = { payload: lastPayload, seq: plotSeq, loadedSurfaceLibrary: false };
    scheduleChartResize({ flush: true });
  }

  function featureChooserHtml(index, title, attributes = "") {
    return `
      <section id="gbmShapFeatureSection${index}" class="gbm-shap-feature-section chart-side-section${index === 2 && feature2Collapsed ? " hidden" : ""}"${attributes}>
        <div class="section-title-row">
          <h2>${escapeHtml(title)}</h2>
          <div class="segmented gbm-shap-sort" role="group" aria-label="${escapeHtml(title)} sort">
            <button type="button" data-gbm-shap-feature="${index}" data-gbm-shap-sort="importance" data-stable-label="Importance">Importance</button>
            <button type="button" data-gbm-shap-feature="${index}" data-gbm-shap-sort="alpha" data-stable-label="A-Z">A-Z</button>
          </div>
        </div>
        <div class="chart-search-row gbm-shap-picker-search-row">
          <input id="gbmShapFeatureSearch${index}" class="search app-control-input gbm-shap-picker-search-input" data-gbm-shap-search="${index}" placeholder="search" aria-label="Search ${escapeHtml(title)}" />
          <button class="app-control-button app-command-button gbm-shap-picker-search-clear" type="button" data-gbm-shap-search-clear="${index}" title="Clear ${escapeHtml(title)} search" aria-label="Clear ${escapeHtml(title)} search">&times;</button>
        </div>
        <div id="gbmShapFeatureList${index}" class="feature-list gbm-shap-feature-list" role="listbox" aria-label="${escapeHtml(title)}"></div>
      </section>
    `;
  }

  function featureButtonHtml(index, label, value, detail, extraClass, active) {
    return `
      <button class="feature ${extraClass || ""} ${active ? "active" : ""}" type="button" data-gbm-shap-feature="${index}" data-gbm-shap-feature-value="${escapeHtml(value)}">
        <span>${escapeHtml(label)}</span><span class="kind">${escapeHtml(detail)}</span>
      </button>
    `;
  }

  function bandingControlHtml(index, feature) {
    const numeric = feature && isNumericKind(feature.kind);
    const disabled = numeric ? "" : " disabled";
    const current = formatBanding(state[`banding${index}`]);
    return `
      <div class="control gbm-shap-banding-control gbm-shap-feature${index}-control ${numeric ? "" : "disabled"}">
        <h3>Feature ${index} banding ${numeric ? `<span>(${escapeHtml(current)})</span>` : ""}</h3>
        <div class="segmented" role="group" aria-label="Feature ${index} banding">
          <button type="button" class="gbm-shap-band-step" data-gbm-shap-feature="${index}" data-gbm-shap-band-action="down" data-stable-label="&lt;"${disabled}>&lt;</button>
          ${BAND_BUTTONS.map((value) => `
            <button type="button" data-gbm-shap-feature="${index}" data-gbm-shap-band-value="${value}" data-stable-label="${value}" class="${Number(state[`banding${index}`]) === value ? "active" : ""}"${disabled}>${value}</button>
          `).join("")}
          <button type="button" class="gbm-shap-band-step" data-gbm-shap-feature="${index}" data-gbm-shap-band-action="up" data-stable-label="&gt;"${disabled}>&gt;</button>
        </div>
      </div>
    `;
  }

  function tailControlHtml() {
    return `
      <div class="control gbm-shap-tail-control">
        <h3>Tail grouping</h3>
        <div class="segmented" role="group" aria-label="Tail percent to group">
          ${TAIL_OPTIONS.map((option) => `
            <button type="button" data-gbm-shap-tail="${option.value}" data-stable-label="${escapeHtml(option.label)}" class="${Number(state.tailPercent) === option.value ? "active" : ""}">${escapeHtml(option.label)}</button>
          `).join("")}
        </div>
      </div>
    `;
  }

  function factorControlHtml(feature1, feature2) {
    return `
      <div class="control gbm-shap-factor-control">
        <h3>Treat as factor</h3>
        <div class="segmented" role="group" aria-label="Treat SHAP features as factors">
          ${factorButtonHtml(1, feature1)}
          ${factorButtonHtml(2, feature2)}
        </div>
      </div>
    `;
  }

  function factorButtonHtml(index, feature) {
    const enabled = Boolean(feature && isNumericKind(feature.kind));
    const pressed = Boolean(state[`factor${index}`]);
    const label = `Feature ${index}`;
    return `
      <button type="button" data-gbm-shap-factor="${index}" data-stable-label="${label}" class="${pressed ? "active" : ""}" aria-pressed="${String(pressed)}" aria-label="Treat Feature ${index} as factor"${enabled ? "" : " disabled"}>${label}</button>
    `;
  }

  function rescaleControlHtml() {
    return `
      <div class="control gbm-shap-rescale-control">
        <h3>Rescale</h3>
        <div class="segmented" role="group" aria-label="SHAP rescale">
          ${["-", "0", "1"].map((value) => `
            <button type="button" data-gbm-shap-rescale="${value}" data-stable-label="${value}" class="${state.rescale === value ? "active" : ""}">${value}</button>
          `).join("")}
        </div>
      </div>
    `;
  }

  function rootNode() {
    return document.getElementById("gbmShapRoot");
  }

  function features() {
    return Array.isArray(config?.features) ? config.features : [];
  }

  function selectedFeature(index) {
    const name = state[`feature${index}`];
    return features().find((feature) => feature.name === name) || null;
  }

  function sortedFeatures(sortMode) {
    const rows = [...features()];
    if (sortMode === "alpha") {
      rows.sort((a, b) => a.name.localeCompare(b.name, undefined, { sensitivity: "base" }));
    } else {
      const metric = featureImportanceMetric();
      rows.sort((a, b) => (
        featureImportanceValue(b, metric) - featureImportanceValue(a, metric)
      ) || a.name.localeCompare(b.name, undefined, { sensitivity: "base" }));
    }
    return rows;
  }

  function featureRankMap() {
    const metric = featureImportanceMetric();
    return new Map(sortedFeaturesByImportance(metric).map((feature, index) => [feature.name, index + 1]));
  }

  function sortedFeaturesByImportance(metric) {
    const rows = [...features()];
    rows.sort((a, b) => (
      featureImportanceValue(b, metric) - featureImportanceValue(a, metric)
    ) || a.name.localeCompare(b.name, undefined, { sensitivity: "base" }));
    return rows;
  }

  function defaultBanding() {
    return normaliseBanding(null);
  }

  function normaliseBanding(value) {
    const number = Number(value);
    if (Number.isFinite(number) && number > 0) return Number(number.toPrecision(12));
    return 1;
  }

  function normaliseRescale(value) {
    const text = String(value || "-").trim();
    return text === "0" || text === "1" ? text : "-";
  }

  function currentBandingKey(index) {
    const feature = selectedFeature(index);
    return feature && isNumericKind(feature.kind) ? `${modelId}:${feature.name}` : "";
  }

  function clearPendingBanding(index) {
    state[`bandingPendingKey${index}`] = "";
  }

  async function ensureBanding(index) {
    const feature = selectedFeature(index);
    if (!feature || !isNumericKind(feature.kind)) return true;
    const key = currentBandingKey(index);
    if (!key || state[`bandingKey${index}`] === key) return true;
    if (state[`bandingPendingKey${index}`] === key) return false;
    const seqKey = `bandingSeq${index}`;
    const seq = (state[seqKey] || 0) + 1;
    state[seqKey] = seq;
    state[`bandingPendingKey${index}`] = key;
    renderControls();
    setMessage("Estimating SHAP banding...");
    try {
      const data = await api("/api/banding/suggestion", {
        method: "POST",
        body: JSON.stringify({ source: "dataset", feature: feature.name }),
      });
      if (state[seqKey] !== seq || state[`bandingPendingKey${index}`] !== key || currentBandingKey(index) !== key) return false;
      state[`banding${index}`] = normaliseBanding(data.band_suggestion);
      state[`bandingKey${index}`] = key;
      clearPendingBanding(index);
      renderControls();
      return true;
    } catch (error) {
      if (state[seqKey] !== seq || state[`bandingPendingKey${index}`] !== key || currentBandingKey(index) !== key) return false;
      state[`banding${index}`] = defaultBanding();
      state[`bandingKey${index}`] = key;
      clearPendingBanding(index);
      renderControls();
      setMessage(`Banding estimate failed; using ${formatBanding(state[`banding${index}`])}. ${error.message}`);
      return true;
    }
  }

  function featureImportanceLabel(feature, rank) {
    if (!feature) return "";
    const metric = featureImportanceMetric();
    const value = metric === "shap" ? formatMeanAbsShap(feature.mean_abs_shap) : formatGain(feature.gain);
    const prefix = Number.isFinite(Number(rank)) ? `Rank ${rank}` : "";
    return value ? `${prefix} · ${value}` : prefix;
  }

  function featureImportanceMetric() {
    return features().some((feature) => featureNumber(feature.mean_abs_shap) !== null)
      ? "shap"
      : "gain";
  }

  function featureImportanceValue(feature, metric) {
    const value = metric === "shap"
      ? featureNumber(feature?.mean_abs_shap)
      : featureNumber(feature?.gain);
    return value ?? 0;
  }

  function featureNumber(value) {
    const number = Number(value);
    return Number.isFinite(number) ? number : null;
  }

  function isNumericKind(kind) {
    return kind === "numeric" || kind === "integer";
  }

  function formatBanding(value) {
    const number = Number(value);
    if (!Number.isFinite(number)) return "1";
    return Number(number.toPrecision(12)).toString();
  }

  function formatGain(value) {
    const number = Number(value || 0);
    if (!Number.isFinite(number)) return "0.000";
    const magnitude = Math.abs(number);
    if (magnitude < 0.0005) return "0.000";
    if (magnitude >= 1000) return Math.round(number).toLocaleString();
    if (magnitude >= 10) return number.toFixed(1);
    return number.toFixed(3);
  }

  function formatMeanAbsShap(value) {
    const number = featureNumber(value);
    return number === null ? "" : number.toFixed(4);
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

  async function copyChartToClipboard() {
    if (!chart || !navigator.clipboard?.write || typeof window.ClipboardItem !== "function") {
      showClipboardToast("Could not copy SHAP chart image", true);
      return;
    }
    try {
      const dataUrl = chart.getDataURL({
        type: "png",
        pixelRatio: 2,
        backgroundColor: chartTheme().panel || "#fff",
      });
      const blob = await fetch(dataUrl).then((response) => response.blob());
      await navigator.clipboard.write([new window.ClipboardItem({ "image/png": blob })]);
      showClipboardToast("SHAP chart image copied");
    } catch (_) {
      showClipboardToast("Could not copy SHAP chart image", true);
    }
  }

  function syncLayoutVisibility(root = rootNode(), { resize = true } = {}) {
    if (!root) return;
    const toolbar = root.querySelector("#gbmShapControls");
    const toolbarToggle = root.querySelector("#gbmShapToolbarToggle");
    const side = root.querySelector("#gbmShapSide");
    const sideToggle = root.querySelector("#gbmShapSideToggle");
    const feature2 = root.querySelector("#gbmShapFeatureSection2");
    const feature2Toggle = root.querySelector("#gbmShapFeature2Toggle");
    const chooserResizer = root.querySelector("#gbmShapChooserDivider");
    const mainResizer = root.querySelector("#gbmShapMainResizer");
    const workspace = root.querySelector(".gbm-shap-workspace");
    const stacked = window.matchMedia(SHAP_STACKED_MEDIA).matches;

    if (controlsCollapsed && toolbar?.contains(document.activeElement)) toolbarToggle?.focus();
    if (sidePanelCollapsed && side?.contains(document.activeElement)) sideToggle?.focus();
    if (feature2Collapsed && feature2?.contains(document.activeElement)) feature2Toggle?.focus();

    setElementCollapsed(toolbar, controlsCollapsed);
    setElementCollapsed(side, sidePanelCollapsed);
    setElementCollapsed(feature2, feature2Collapsed);
    workspace?.classList.toggle("gbm-shap-side-collapsed", sidePanelCollapsed);
    workspace?.classList.toggle("gbm-shap-feature2-collapsed", feature2Collapsed);

    const mainSplitterHidden = sidePanelCollapsed || stacked;
    syncSplitterVisibility(mainResizer, mainSplitterHidden);
    syncSplitterVisibility(chooserResizer, sidePanelCollapsed || feature2Collapsed);

    syncToggleButton(
      toolbarToggle,
      !controlsCollapsed,
      controlsCollapsed ? "Show SHAP control row" : "Hide SHAP control row",
    );
    syncToggleButton(
      sideToggle,
      !sidePanelCollapsed,
      sidePanelCollapsed ? "Show SHAP feature choosers" : "Hide SHAP feature choosers",
    );
    syncToggleButton(
      feature2Toggle,
      !feature2Collapsed,
      feature2Collapsed ? "Show Feature 2 chooser" : "Hide Feature 2 chooser",
    );

    if (!stacked) setMainSideWidth(root, sidePanelWidth, { resize: false });
    if (!feature2Collapsed) {
      setChooserFeatureHeight(root, chooserFeatureHeight ?? defaultChooserFeatureHeight(root));
    }
    if (resize) scheduleChartResize({ flush: true });
  }

  function setElementCollapsed(element, collapsed) {
    if (!element) return;
    element.classList.toggle("hidden", collapsed);
    element.hidden = collapsed;
    element.toggleAttribute("inert", collapsed);
    element.setAttribute("aria-hidden", String(collapsed));
  }

  function syncSplitterVisibility(resizer, hidden) {
    if (!resizer) return;
    resizer.hidden = hidden;
    resizer.toggleAttribute("inert", hidden);
    resizer.tabIndex = hidden ? -1 : 0;
    resizer.setAttribute("aria-hidden", String(hidden));
  }

  function syncToggleButton(button, expanded, label) {
    if (!button) return;
    button.setAttribute("aria-expanded", String(expanded));
    button.setAttribute("aria-label", label);
    button.title = label;
  }

  function resizeChart({ flush = false } = {}) {
    const target = document.getElementById("gbmShapChart");
    if (!chart || target !== observedChartTarget || !isEchartsTargetReady(target)) return;
    chart.resize();
    if (flush) {
      chart.getZr?.().flush?.();
      settledObserverSize = currentChartSize();
    }
  }

  function currentChartSize() {
    const target = document.getElementById("gbmShapChart");
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
      if (!isEchartsTargetReady(observedChartTarget)) return;
      if (pendingChartRender) {
        resumePendingChartRender();
        return;
      }
      if (pendingEmptyMessage !== null) {
        renderPendingEmptyState();
        return;
      }
      const shouldFlush = chartResizeFlush;
      chartResizeFlush = false;
      resizeChart({ flush: shouldFlush });
    });
  }

  function resize() {
    const root = rootNode();
    if (root && !window.matchMedia(SHAP_STACKED_MEDIA).matches) {
      setMainSideWidth(root, sidePanelWidth, { resize: false });
    }
    if (root && !feature2Collapsed) {
      setChooserFeatureHeight(root, chooserFeatureHeight ?? defaultChooserFeatureHeight(root));
    }
    scheduleChartResize({ flush: true });
  }

  function setupChooserDividerResize(root) {
    const side = root.querySelector(".gbm-shap-side");
    const firstPanel = side?.querySelector(".gbm-shap-feature-section");
    const resizer = root.querySelector("#gbmShapChooserDivider");
    if (!side || !firstPanel || !resizer) return;
    if (Number.isFinite(chooserFeatureHeight) && chooserFeatureHeight > 0) setChooserFeatureHeight(root, chooserFeatureHeight);

    let dragging = false;
    let startY = 0;
    let startHeight = 0;
    resizer.addEventListener("pointerdown", (event) => {
      if (feature2Collapsed || sidePanelCollapsed) return;
      event.preventDefault();
      dragging = true;
      startY = event.clientY;
      startHeight = firstPanel.getBoundingClientRect().height || 0;
      resizer.classList.add("dragging");
      document.body.classList.add("resizing-chart-control-heights");
      resizer.setPointerCapture(event.pointerId);
      window.getSelection()?.removeAllRanges();
    });
    resizer.addEventListener("pointermove", (event) => {
      if (!dragging) return;
      event.preventDefault();
      setChooserFeatureHeight(root, startHeight + event.clientY - startY);
    });
    function finishDrag(event) {
      if (!dragging) return;
      dragging = false;
      resizer.classList.remove("dragging");
      document.body.classList.remove("resizing-chart-control-heights");
      window.getSelection()?.removeAllRanges();
      if (event.pointerId !== undefined) {
        try {
          resizer.releasePointerCapture(event.pointerId);
        } catch (_) {
        }
      }
    }
    resizer.addEventListener("pointerup", finishDrag);
    resizer.addEventListener("pointercancel", finishDrag);
    resizer.addEventListener("keydown", (event) => {
      if (feature2Collapsed || sidePanelCollapsed) return;
      const bounds = chooserHeightBounds(root);
      let nextHeight = null;
      if (event.key === "ArrowUp") nextHeight = (chooserFeatureHeight ?? firstPanel.getBoundingClientRect().height) - SHAP_SPLITTER_KEY_STEP;
      if (event.key === "ArrowDown") nextHeight = (chooserFeatureHeight ?? firstPanel.getBoundingClientRect().height) + SHAP_SPLITTER_KEY_STEP;
      if (event.key === "Home") nextHeight = bounds.min;
      if (event.key === "End") nextHeight = bounds.max;
      if (nextHeight === null) return;
      event.preventDefault();
      setChooserFeatureHeight(root, nextHeight);
    });
  }

  function defaultChooserFeatureHeight(root = rootNode()) {
    const bounds = chooserHeightBounds(root);
    return Math.round((bounds.min + bounds.max) / 2);
  }

  function chooserHeightBounds(root = rootNode()) {
    const side = root?.querySelector(".gbm-shap-side");
    const availableHeight = side?.getBoundingClientRect().height || window.innerHeight;
    const max = Math.max(
      SHAP_CHOOSER_MIN_HEIGHT,
      availableHeight - SHAP_CHOOSER_DIVIDER_HEIGHT - SHAP_CHOOSER_MIN_HEIGHT,
    );
    return { min: SHAP_CHOOSER_MIN_HEIGHT, max };
  }

  function setChooserFeatureHeight(root, rawHeight) {
    const resizer = root?.querySelector("#gbmShapChooserDivider");
    const bounds = chooserHeightBounds(root);
    const height = Math.min(Math.max(Number(rawHeight) || bounds.min, bounds.min), bounds.max);
    chooserFeatureHeight = height;
    root?.style.setProperty("--gbm-shap-feature1-height", `${Math.round(height)}px`);
    document.documentElement.style.setProperty("--gbm-shap-feature1-height", `${Math.round(height)}px`);
    resizer?.setAttribute("aria-valuemin", String(bounds.min));
    resizer?.setAttribute("aria-valuemax", String(Math.round(bounds.max)));
    resizer?.setAttribute("aria-valuenow", String(Math.round(height)));
    return height;
  }

  function setupMainDividerResize(root) {
    const side = root.querySelector(".gbm-shap-side");
    const resizer = root.querySelector("#gbmShapMainResizer");
    if (!side || !resizer) return;
    let dragging = false;
    let startX = 0;
    let startWidth = sidePanelWidth;
    resizer.addEventListener("pointerdown", (event) => {
      if (window.matchMedia(SHAP_STACKED_MEDIA).matches || sidePanelCollapsed) return;
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
      if (window.matchMedia(SHAP_STACKED_MEDIA).matches || sidePanelCollapsed) return;
      const bounds = mainSideWidthBounds(root);
      let nextWidth = null;
      if (event.key === "ArrowLeft") nextWidth = sidePanelWidth - SHAP_SPLITTER_KEY_STEP;
      if (event.key === "ArrowRight") nextWidth = sidePanelWidth + SHAP_SPLITTER_KEY_STEP;
      if (event.key === "Home") nextWidth = bounds.min;
      if (event.key === "End") nextWidth = bounds.max;
      if (nextWidth === null) return;
      event.preventDefault();
      setMainSideWidth(root, nextWidth, { flush: true });
    });
    setMainSideWidth(root, sidePanelWidth, { resize: false });
  }

  function mainSideWidthBounds(root = rootNode()) {
    const workspace = root?.querySelector(".gbm-shap-workspace");
    const availableWidth = workspace?.getBoundingClientRect().width || window.innerWidth;
    const max = Math.max(
      SHAP_SIDE_MIN_WIDTH,
      Math.min(SHAP_SIDE_MAX_WIDTH, availableWidth - SHAP_CHART_MIN_WIDTH),
    );
    return { min: SHAP_SIDE_MIN_WIDTH, max };
  }

  function setMainSideWidth(root, rawWidth, { resize = true, flush = false } = {}) {
    const resizer = root?.querySelector("#gbmShapMainResizer");
    const bounds = mainSideWidthBounds(root);
    const width = Math.min(
      Math.max(Number(rawWidth) || SHAP_SIDE_DEFAULT_WIDTH, bounds.min),
      bounds.max,
    );
    sidePanelWidth = width;
    root.style.setProperty("--gbm-shap-side-width", `${Math.round(width)}px`);
    resizer?.setAttribute("aria-valuemin", String(Math.round(bounds.min)));
    resizer?.setAttribute("aria-valuemax", String(Math.round(bounds.max)));
    resizer?.setAttribute("aria-valuenow", String(Math.round(width)));
    if (resize) scheduleChartResize({ flush });
    return width;
  }

  function legendEntryNames(option) {
    const series = Array.isArray(option?.series) ? option.series : [];
    return series.map((item) => String(item?.name || "")).filter(Boolean);
  }

  function legendSelection(option, entries) {
    const selected = Array.isArray(option?.legend)
      ? option.legend[0]?.selected
      : option?.legend?.selected;
    return Object.fromEntries(entries.map((entry) => [entry, selected?.[entry] !== false]));
  }

  function applyLegendSelection(option, selection, entries) {
    const legend = Array.isArray(option?.legend) ? option.legend[0] : option?.legend;
    if (!legend || !entries.length) return;
    legend.selected = {
      ...(legend.selected || {}),
      ...Object.fromEntries(entries.map((entry) => [entry, selection?.[entry] !== false])),
    };
  }

  function snapshotLegendState() {
    if (!chart || !lastPayload) return;
    const option = chart.getOption?.();
    const entries = legendEntryNames(option);
    if (!entries.length) return;
    pendingLegendState = {
      entries,
      feature1: state.feature1 || "",
      feature2: state.feature2 || "",
      plotType: lastPayload.plot_type || "",
      selection: legendSelection(option, entries),
    };
  }

  function pendingLegendSelectionForPayload(payload, entries) {
    if (!pendingLegendState) return null;
    if (pendingLegendState.plotType !== payload.plot_type) return null;
    if (!sameEntries(pendingLegendState.entries, entries)) return null;
    if ((pendingLegendState.feature1 || "") !== (state.feature1 || "")) return null;
    if ((pendingLegendState.feature2 || "") !== (state.feature2 || "")) return null;
    return pendingLegendState.selection;
  }

  function clearPendingLegendState() {
    pendingLegendState = null;
  }

  function sameEntries(left, right) {
    if (!Array.isArray(left) || !Array.isArray(right) || left.length !== right.length) return false;
    return left.every((entry, index) => entry === right[index]);
  }

  return {
    dispose,
    preselectFeatures,
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

function nextAnimationFrame() {
  return new Promise((resolve) => {
    if (typeof requestAnimationFrame === "function") {
      requestAnimationFrame(() => resolve());
    } else {
      setTimeout(resolve, 0);
    }
  });
}

function isSurfaceLayoutError(error) {
  const message = String(error?.message || error || "");
  return (message.includes("undefined") && message.includes("length")) || message.includes("o.length");
}
