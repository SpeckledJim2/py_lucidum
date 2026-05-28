import { emptyOption, ensureShapChartLibraries, shapChartOption } from "./gbm-shap-chart.js";

const BAND_STEPS = makeBandSteps();
const TAIL_OPTIONS = [
  { value: 0, label: "-" },
  { value: 0.1, label: "0.1%" },
  { value: 0.5, label: "0.5%" },
  { value: 1, label: "1%" },
  { value: 2, label: "2%" },
  { value: 5, label: "5%" },
];

export function createGbmShapTool({ api, escapeHtml, setNotice }) {
  let modelId = "";
  let lastModelId = "";
  let config = null;
  let chart = null;
  let resizeObserver = null;
  let lastPayload = null;
  let configSeq = 0;
  let plotSeq = 0;
  const state = {
    feature1: "",
    feature2: "",
    sort1: "importance",
    sort2: "importance",
    search1: "",
    search2: "",
    banding1: 1,
    banding2: 1,
    factor1: false,
    factor2: false,
    tailPercent: 1,
  };

  function shellHtml() {
    return `
      <div id="gbmShapRoot" class="gbm-shap-view">
        <aside class="gbm-shap-side">
          ${featureChooserHtml(1, "Feature 1")}
          <div id="gbmShapChooserDivider" class="gbm-shap-chooser-divider"></div>
          ${featureChooserHtml(2, "Feature 2")}
        </aside>
        <section class="gbm-shap-main">
          <div id="gbmShapControls" class="gbm-shap-controls"></div>
          <div class="gbm-shap-chart-shell">
            <div id="gbmShapMessage" class="gbm-shap-message hidden"></div>
            <div id="gbmShapChart" class="gbm-shap-chart" aria-label="GBM SHAP plot"></div>
          </div>
        </section>
      </div>
    `;
  }

  async function render(nextModelId) {
    modelId = String(nextModelId || "");
    const root = rootNode();
    if (!root) return;
    bindStaticEvents(root);
    if (!modelId) {
      config = null;
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
      renderEmpty("SHAP could not load");
    }
  }

  function bindStaticEvents(root) {
    if (root.dataset.gbmShapBound === "1") return;
    root.dataset.gbmShapBound = "1";
    root.addEventListener("click", handleClick);
    root.addEventListener("input", handleInput);
    root.addEventListener("change", handleChange);
  }

  function handleClick(event) {
    const button = event.target.closest("button");
    if (!button || !rootNode()?.contains(button)) return;
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
    }
  }

  function handleInput(event) {
    const input = event.target;
    if (input?.dataset?.gbmShapSearch) {
      state[`search${input.dataset.gbmShapSearch}`] = input.value;
      renderFeatureLists();
    }
  }

  function handleChange(event) {
    const input = event.target;
    if (input?.dataset?.gbmShapFactor) {
      state[`factor${input.dataset.gbmShapFactor}`] = Boolean(input.checked);
      renderControls();
      refreshPlot();
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
      state[`banding${index}`] = defaultBanding(selectedFeature(index));
      state[`factor${index}`] = false;
    }
    renderControlsAndLists();
    refreshPlot();
  }

  function setBanding(index, value) {
    state[`banding${index}`] = normaliseBanding(value, selectedFeature(index));
    renderControls();
    refreshPlot();
  }

  function stepBanding(index, direction) {
    const current = Number(state[`banding${index}`]) || defaultBanding(selectedFeature(index));
    const next = direction < 0
      ? [...BAND_STEPS].reverse().find((step) => step < current) || current
      : BAND_STEPS.find((step) => step > current) || current;
    setBanding(index, next);
  }

  async function refreshPlot() {
    if (!config?.has_shap || !state.feature1) return;
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
    const isSurface = payload.plot_type === "surface";
    const previousPlotType = lastPayload?.plot_type || "";
    const loadedSurfaceLibrary = await ensureShapChartLibraries(payload.plot_type);
    if (seq !== plotSeq) return;
    if (isSurface && (loadedSurfaceLibrary || previousPlotType !== "surface")) {
      disposeChart();
      await nextAnimationFrame();
      if (seq !== plotSeq) return;
    }
    ensureChart(target);
    const option = shapChartOption(payload, chartTheme());
    try {
      chart.setOption(option, true);
    } catch (error) {
      if (!isSurface || !isSurfaceLayoutError(error)) throw error;
      disposeChart();
      await nextAnimationFrame();
      if (seq !== plotSeq) return;
      ensureChart(target);
      await nextAnimationFrame();
      if (seq !== plotSeq) return;
      chart.setOption(option, true);
    }
    lastPayload = payload;
  }

  function ensureChart(target) {
    if (chart) return;
    chart = window.echarts.init(target);
    resizeObserver = new ResizeObserver(() => chart?.resize());
    resizeObserver.observe(target);
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
      ${tailControlHtml()}
      ${bandingControlHtml(2, feature2)}
      ${factorControlHtml(1, feature1)}
      <div></div>
      ${factorControlHtml(2, feature2)}
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
    if (modelId !== lastModelId) {
      state.factor1 = false;
      state.factor2 = false;
      state.search1 = "";
      state.search2 = "";
      lastModelId = modelId;
    }
    if (!names.has(state.feature1)) state.feature1 = firstFeatureNameForChooser(1);
    if (state.feature2 && !names.has(state.feature2)) state.feature2 = "";
    state.banding1 = normaliseBanding(state.banding1, selectedFeature(1));
    state.banding2 = normaliseBanding(state.banding2, selectedFeature(2));
  }

  function firstFeatureNameForChooser(index) {
    return sortedFeatures(state[`sort${index}`])[0]?.name || "";
  }

  function renderLoading(message) {
    renderControlsAndLists();
    renderEmpty(message);
  }

  function renderEmpty(message) {
    disposeChart();
    const target = document.getElementById("gbmShapChart");
    if (!target) return;
    target.innerHTML = "";
    chart = window.echarts.init(target);
    resizeObserver = new ResizeObserver(() => chart?.resize());
    resizeObserver.observe(target);
    chart.setOption(emptyOption(message, chartTheme()), true);
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
    disposeChart();
    config = null;
  }

  function disposeChart() {
    resizeObserver?.disconnect();
    resizeObserver = null;
    lastPayload = null;
    if (chart) {
      chart.dispose();
      chart = null;
    }
  }

  function refreshTheme() {
    if (chart && lastPayload) {
      chart.setOption(shapChartOption(lastPayload, chartTheme()), true);
      chart.resize();
    }
  }

  function featureChooserHtml(index, title) {
    return `
      <section class="gbm-shap-feature-section chart-side-section">
        <div class="section-title-row">
          <h2>${escapeHtml(title)}</h2>
          <div class="segmented gbm-shap-sort" role="group" aria-label="${escapeHtml(title)} sort">
            <button type="button" data-gbm-shap-feature="${index}" data-gbm-shap-sort="importance">Importance</button>
            <button type="button" data-gbm-shap-feature="${index}" data-gbm-shap-sort="alpha">A-Z</button>
          </div>
        </div>
        <div class="chart-search-row">
          <input id="gbmShapFeatureSearch${index}" class="search" data-gbm-shap-search="${index}" placeholder="search" />
          <button class="filter-action" type="button" data-gbm-shap-search-clear="${index}" title="Clear ${escapeHtml(title)} search" aria-label="Clear ${escapeHtml(title)} search">&times;</button>
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
    const label = numeric ? `Feature ${index}` : (feature ? "Factor" : `Feature ${index}`);
    const disabled = numeric ? "" : " disabled";
    const current = formatBanding(state[`banding${index}`]);
    return `
      <div class="control gbm-shap-banding-control gbm-shap-feature${index}-control ${numeric ? "" : "disabled"}">
        <h3>${escapeHtml(label)} ${numeric ? `<span>(${escapeHtml(current)})</span>` : ""}</h3>
        <div class="segmented" role="group" aria-label="Feature ${index} banding">
          <button type="button" data-gbm-shap-feature="${index}" data-gbm-shap-band-action="down"${disabled}>&lt;</button>
          ${[0.01, 0.1, 1, 5, 10, 100].map((value) => `
            <button type="button" data-gbm-shap-feature="${index}" data-gbm-shap-band-value="${value}" class="${Number(state[`banding${index}`]) === value ? "active" : ""}"${disabled}>${value}</button>
          `).join("")}
          <button type="button" data-gbm-shap-feature="${index}" data-gbm-shap-band-action="up"${disabled}>&gt;</button>
        </div>
      </div>
    `;
  }

  function tailControlHtml() {
    return `
      <div class="control gbm-shap-tail-control">
        <h3>Tail % to group</h3>
        <div class="segmented" role="group" aria-label="Tail percent to group">
          ${TAIL_OPTIONS.map((option) => `
            <button type="button" data-gbm-shap-tail="${option.value}" class="${Number(state.tailPercent) === option.value ? "active" : ""}">${escapeHtml(option.label)}</button>
          `).join("")}
        </div>
      </div>
    `;
  }

  function factorControlHtml(index, feature) {
    const disabled = feature && isNumericKind(feature.kind) ? "" : " disabled";
    return `
      <label class="gbm-shap-factor-control gbm-shap-feature${index}-factor ${disabled ? "disabled" : ""}">
        <input type="checkbox" data-gbm-shap-factor="${index}" ${state[`factor${index}`] ? "checked" : ""}${disabled} />
        <span>Treat as factor</span>
      </label>
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

  function defaultBanding(feature) {
    return normaliseBanding(feature?.band_suggestion, feature);
  }

  function normaliseBanding(value, feature) {
    const number = Number(value);
    if (Number.isFinite(number) && number > 0) return Number(number.toPrecision(12));
    const fallback = Number(feature?.band_suggestion);
    if (Number.isFinite(fallback) && fallback > 0) return Number(fallback.toPrecision(12));
    return 1;
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

  return {
    dispose,
    refreshTheme,
    render,
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
