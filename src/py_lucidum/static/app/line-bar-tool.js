import { loadTabulator } from "./shared/tabulator.js";

const LINE_BAR_SPECIAL_COLUMN_NAMES = [
  "gbm_to_glm_ratio",
  "glm_prediction",
  "gbm_prediction",
  "glm_prediction_rate",
  "gbm_prediction_rate",
  "glm_tabulated_prediction",
  "gbm_tabulated_prediction",
];

export function createLineBarTool({
  api,
  el,
  state,
  echartsImpl,
  escapeHtml,
  isModelPredictionColumn,
  copyTextToClipboard = () => Promise.resolve(false),
  formatNumber,
  formatChartLabel,
  formatLineLabel,
  formatLineValue,
  formatLineValueForFormat = formatLineValue,
  formatXLabel,
  formatRowMeta,
  measureToolRender,
  startToolTiming,
  setToolTimingFailed,
  syncDuckDbTimingFromData,
  syncClientTimingFromData,
  setStatus,
  setChartMessage,
  setGroupMeta,
  applyToolPresentation,
  saveToolPresentation,
  showClipboardToast = () => {},
  stableRequestKey = (request) => JSON.stringify(request),
  toolCache,
  sourceColumns,
  expectedColumns = numericColumns,
  selectedColumn,
  numericColumns,
  dataSourceForId,
  dataSourceHasColumn,
  syncExpectedSourceFromSelection = () => false,
  toolEnabled,
  setTool,
  renderMetricTitle,
  getCss,
  bandSteps,
  refreshLineBar,
  clearActiveFavouriteSelection = () => false,
}) {
  const TABLE_PAGE_SIZE = 10000;
  const TABLE_SEARCH_DEBOUNCE_MS = 250;
  const LABEL_DENSITY_LIMIT = 200;
  const DATE_AXIS_MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];
  const DATE_AXIS_WEEKDAYS = ["Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat"];
  const DATE_AXIS_FONT_SIZES = [10, 9, 8, 7];
  const DATE_AXIS_LABEL_WIDTH_FACTOR = 0.56;
  const DATE_AXIS_LABEL_PADDING = 8;
  const DATE_AXIS_HORIZONTAL_LABEL_LIMIT = 10;
  const DATE_AXIS_YEAR_HORIZONTAL_LABEL_LIMIT = 25;
  const DATE_AXIS_VISIBLE_LABEL_LIMIT = 60;
  const DATE_AXIS_ROTATION = 60;
  const QUANTILE_AXIS_FONT_SIZES = [10, 9, 8, 7];
  const QUANTILE_AXIS_ROTATIONS = [0, 45, 65, 75];
  const QUANTILE_AXIS_LABEL_WIDTH_FACTOR = 0.54;
  const QUANTILE_AXIS_LABEL_PADDING = 8;
  const CATEGORICAL_AXIS_LABEL_PADDING = 8;
  const DATE_BUCKET_VALUES = new Set(["none", "hour", "day", "week", "month", "year"]);
  const RESPONSE_AXIS_PADDING = 0.08;
  const RESPONSE_AXIS_TARGET_INTERVALS = 15;
  const LINE_BAR_MAIN_LEGEND_TOP = 52;
  const LINE_BAR_OVERLAY_LEGEND_TOP = 78;
  const LINE_BAR_GRID_TOP = 112;
  const LINE_BAR_OVERLAY_GRID_TOP = 140;
  const SHAP_RIBBON_SERIES = [
    ["p0", "p100", "SHAP Min-Max", "rgba(209, 63, 63, 0.10)"],
    ["p5", "p95", "SHAP 5-95", "rgba(209, 63, 63, 0.16)"],
    ["p10", "p90", "SHAP 10-90", "rgba(209, 63, 63, 0.20)"],
    ["p20", "p80", "SHAP 20-80", "rgba(209, 63, 63, 0.24)"],
    ["p30", "p70", "SHAP 30-70", "rgba(209, 63, 63, 0.28)"],
    ["p40", "p60", "SHAP 40-60", "rgba(209, 63, 63, 0.34)"],
  ];
  const SHAP_LINE_COLOR = "#d13f3f";
  const GLM_LINE_COLOR = "#1f7a8c";
  const chart = echartsImpl.init(el("chart"));
  let featureImportanceData = null;
  let featureImportanceKey = "";
  let featureImportancePendingKey = "";
  let featureImportanceRequestSeq = 0;
  let featureImportanceError = "";
  let tableRequestSeq = 0;
  let tableSearchTimer = null;
  let tableCacheKey = "";
  let tableCacheData = null;
  let completeTableCacheKey = "";
  let completeTableCacheData = null;
  let tableRenderToken = 0;
  let lineBarTable = null;
  let lineBarTableCopyRows = [];
  let lineBarTableCopyColumns = [];
  let lineBarTableCopyFooterRow = null;
  let dateXAxisContext = null;
  let dateXAxisRefreshFrame = null;
  let chartRenderTransform = "none";
  let lineBarChartDirty = false;

  function isNumericKind(kind) {
    return kind === "numeric" || kind === "integer";
  }

  function isDateKind(kind) {
    return kind === "date" || kind === "datetime";
  }

  function isLineBarSpecialColumn(column) {
    return LINE_BAR_SPECIAL_COLUMN_NAMES.includes(String(column?.name || ""));
  }

  function isGbmGlmRatioColumn(column) {
    return String(column?.name || "") === "gbm_to_glm_ratio";
  }

  function lineBarSpecialColumnOrder(column) {
    const index = LINE_BAR_SPECIAL_COLUMN_NAMES.indexOf(String(column?.name || ""));
    return index >= 0 ? index : LINE_BAR_SPECIAL_COLUMN_NAMES.length;
  }

  function orderedLineBarSpecialColumns(columns) {
    return columns
      .filter(isLineBarSpecialColumn)
      .sort((a, b) => (
        lineBarSpecialColumnOrder(a) - lineBarSpecialColumnOrder(b)
        || String(a.source_id || "").localeCompare(String(b.source_id || ""), undefined, { sensitivity: "base" })
      ));
  }

  function lineBarFeatureTargetSource(featureName) {
    if (!lineBarToolAvailable()) return "";
    const name = String(featureName || "");
    if (!name) return "";
    const match = sourceColumns().find((column) => column.name === name);
    if (match) return match.source_id || state.source || "dataset";
    return dataSourceHasColumn("dataset", name) ? "dataset" : "";
  }

  function lineBarToolAvailable() {
    if (toolEnabled("line_bar")) return true;
    const button = el("lineBarTool");
    return Boolean(button && !button.disabled && !button.classList.contains("hidden"));
  }

  function canNavigateToLineBarFeature(featureName) {
    return Boolean(lineBarFeatureTargetSource(featureName));
  }

  function navigateToLineBarFeature(featureName) {
    const name = String(featureName || "");
    const targetSource = lineBarFeatureTargetSource(name);
    if (!targetSource) return false;
    state.source = targetSource;
    state.x = name;
    state.xSource = targetSource;
    state.bandFeature = null;
    resetDateBucketSuggestionState();
    renderFeatures();
    renderExpectedNumerators();
    updateAxisControls();
    setTool("line_bar");
    return state.tool === "line_bar";
  }

  function expectedDisplayColumns() {
    const columns = expectedColumns().filter((column) => column.source_role !== "gbm_shap_value");
    const specialColumns = orderedLineBarSpecialColumns(columns);
    const otherColumns = columns.filter((column) => !isLineBarSpecialColumn(column));
    if (state.expectedSort === "alpha") {
      otherColumns.sort((a, b) => a.name.localeCompare(b.name, undefined, { sensitivity: "base" }));
    }
    return { specialColumns, otherColumns };
  }

  function syncSegmented(control, value) {
    const group = document.querySelector(`.segmented[data-control="${control}"]`);
    if (!group) return;
    group.querySelectorAll("button").forEach((button) => {
      button.classList.toggle("active", button.dataset.value === value);
    });
  }

  function featureImportanceCacheKey() {
    const sources = state.schema?.data_sources || [];
    const activeSourceId = (kind) => sources.find((source) => source.kind === kind && source.active)?.model_id || "";
    const dataset = dataSourceForId("dataset") || {};
    return JSON.stringify([
      activeSourceId("gbm_predictions"),
      activeSourceId("glm_predictions"),
      dataset.row_count || state.schema?.row_count || 0,
      state.schema?.path || "",
    ]);
  }

  function currentFeatureImportanceData() {
    return featureImportanceKey && featureImportanceKey === featureImportanceCacheKey() ? featureImportanceData : null;
  }

  function featureImportanceHasRows(data = currentFeatureImportanceData()) {
    const gbmRows = data?.models?.gbm?.rows;
    const glmRows = data?.models?.glm?.rows;
    return Boolean((Array.isArray(gbmRows) && gbmRows.length) || (Array.isArray(glmRows) && glmRows.length));
  }

  function syncFeatureImportanceButton(data = currentFeatureImportanceData()) {
    const button = document.querySelector('.segmented[data-control="featureSort"] button[data-value="importance"]');
    if (!button) return;
    const visible = featureImportanceHasRows(data) || state.featureSort === "importance";
    button.classList.toggle("hidden", !visible);
    button.classList.toggle("active", state.featureSort === "importance");
  }

  async function ensureFeatureImportance() {
    if (!state.schema) return null;
    const key = featureImportanceCacheKey();
    if (featureImportanceKey === key && featureImportanceData) {
      syncFeatureImportanceButton(featureImportanceData);
      return featureImportanceData;
    }
    if (featureImportancePendingKey === key) return null;
    const requestSeq = featureImportanceRequestSeq + 1;
    featureImportanceRequestSeq = requestSeq;
    featureImportancePendingKey = key;
    featureImportanceError = "";
    syncFeatureImportanceButton(null);
    try {
      const data = await api("/api/line-bar/feature-importance");
      if (requestSeq !== featureImportanceRequestSeq || featureImportancePendingKey !== key) return null;
      featureImportanceData = data;
      featureImportanceKey = key;
      featureImportancePendingKey = "";
      featureImportanceError = "";
      syncFeatureImportanceButton(data);
      if (state.featureSort === "importance") renderFeatures();
      return data;
    } catch (error) {
      if (requestSeq !== featureImportanceRequestSeq || featureImportancePendingKey !== key) return null;
      featureImportanceData = null;
      featureImportanceKey = key;
      featureImportancePendingKey = "";
      featureImportanceError = error.message || "Feature importances unavailable.";
      syncFeatureImportanceButton(null);
      if (state.featureSort === "importance") renderFeatures();
      return null;
    }
  }

  function selectedPartialDependenceMode() {
    const mode = String(state.partialDependence || "none");
    return ["none", "shap", "glm", "both"].includes(mode) ? mode : "none";
  }

  function shapPartialDependenceVisible() {
    return selectedPartialDependenceMode() === "shap" || selectedPartialDependenceMode() === "both";
  }

  function formatBandWidth(value) {
    const number = Number(value);
    if (!Number.isFinite(number) || number <= 0) return "0";
    return Number(number.toPrecision(12)).toString();
  }

  function previousBandWidthsByFeature() {
    if (!state.previousBandWidthsByFeature || typeof state.previousBandWidthsByFeature !== "object" || Array.isArray(state.previousBandWidthsByFeature)) {
      state.previousBandWidthsByFeature = {};
    }
    return state.previousBandWidthsByFeature;
  }

  function rememberNonQuantileBandWidthForCurrentFeature() {
    previousBandWidthsByFeature()[currentBandFeatureKey()] = String(state.bandWidth ?? "0");
  }

  function restoreNonQuantileBandWidthForCurrentFeature() {
    const saved = previousBandWidthsByFeature()[currentBandFeatureKey()];
    if (saved === undefined) return false;
    state.bandWidth = String(saved);
    return true;
  }

  function syncBandingControl() {
    syncSegmented("bandWidth", state.bandWidth);
    el("bandLabel").textContent = state.quantileMode === "quantile" ? "Quantiles" : "Banding";
    const display = Number(state.bandWidth) > 0 ? state.bandWidth : "auto off";
    el("bandValue").textContent = `(${display})`;
  }

  function quantileCountForBandWidth(value = state.bandWidth) {
    const number = Number(value);
    if (!Number.isFinite(number) || number <= 0) return 1;
    return Math.min(1000, Math.max(1, Math.round(number)));
  }

  function syncQuantileControl() {
    syncSegmented("quantileMode", state.quantileMode);
  }

  function normalizeBandWidthForQuantiles() {
    state.bandWidth = String(quantileCountForBandWidth());
    state.bandFeature = currentBandFeatureKey();
    syncBandingControl();
  }

  function currentBandFeatureKey() {
    const sourceId = selectedColumn()?.source_id || state.xSource || state.source || "dataset";
    return JSON.stringify([sourceId, state.x || ""]);
  }

  function normaliseDateBucket(value) {
    const bucket = String(value || "none").toLowerCase();
    return DATE_BUCKET_VALUES.has(bucket) ? bucket : "none";
  }

  function syncDateBucketControl() {
    syncSegmented("dateBucket", normaliseDateBucket(state.dateBucket));
  }

  function currentDateBucketFeatureKey() {
    const sourceId = selectedColumn()?.source_id || state.xSource || state.source || "dataset";
    return JSON.stringify([sourceId, state.x || "", state.activeFilter || ""]);
  }

  function clearPendingDateBucketSuggestion() {
    state.dateBucketSuggestionPendingKey = null;
  }

  function resetDateBucketSuggestionState() {
    state.dateBucketFeature = null;
    state.dateBucketManualKey = null;
    clearPendingDateBucketSuggestion();
  }

  function resetDateBucketSuggestionIfKeyChanged(previousKey) {
    if (previousKey !== currentDateBucketFeatureKey()) resetDateBucketSuggestionState();
  }

  function fallbackBandWidthForSelectedColumn() {
    return isNumericKind(selectedColumn()?.kind) ? "1" : "0";
  }

  function clearPendingBandSuggestion() {
    state.bandSuggestionPendingKey = null;
  }

  async function requestBandSuggestionForSelectedColumn(bandFeatureKey = currentBandFeatureKey()) {
    if (!state.schema || !state.x || state.bandSuggestionPendingKey === bandFeatureKey) return;
    const column = selectedColumn();
    if (!isNumericKind(column?.kind)) return;
    if (state.quantileMode === "quantile") {
      state.bandFeature = bandFeatureKey;
      clearPendingBandSuggestion();
      syncBandingControl();
      return;
    }
    const requestSeq = (state.bandSuggestionRequestSeq || 0) + 1;
    state.bandSuggestionRequestSeq = requestSeq;
    state.bandSuggestionPendingKey = bandFeatureKey;
    syncBandingControl();
    try {
      const sourceId = selectedColumn()?.source_id || state.xSource || state.source || "dataset";
      const baseSource = state.source || "dataset";
      const requestPayload = {
        source: baseSource,
        feature: state.x,
        filter: state.activeFilter,
        responses: currentResponses(),
      };
      if (sourceId && (sourceId !== baseSource || isModelPredictionColumn(column))) {
        requestPayload.xSource = sourceId;
      }
      const data = await api("/api/banding/suggestion", {
        method: "POST",
        body: JSON.stringify(requestPayload),
      });
      if (requestSeq !== state.bandSuggestionRequestSeq || state.bandSuggestionPendingKey !== bandFeatureKey) return;
      if (currentBandFeatureKey() !== bandFeatureKey) return;
      const formatted = formatBandWidth(data.band_suggestion);
      state.bandWidth = formatted === "0" ? fallbackBandWidthForSelectedColumn() : formatted;
      state.bandFeature = bandFeatureKey;
      clearPendingBandSuggestion();
      if (state.quantileMode === "quantile") {
        normalizeBandWidthForQuantiles();
      } else {
        syncBandingControl();
      }
      if (state.tool === "line_bar") refreshChart({ force: true });
    } catch (error) {
      if (requestSeq !== state.bandSuggestionRequestSeq || state.bandSuggestionPendingKey !== bandFeatureKey) return;
      if (currentBandFeatureKey() !== bandFeatureKey) return;
      state.bandWidth = fallbackBandWidthForSelectedColumn();
      state.bandFeature = bandFeatureKey;
      clearPendingBandSuggestion();
      syncBandingControl();
      const warning = `Banding estimate failed; using ${state.bandWidth}. ${error.message}`;
      if (state.tool === "line_bar") {
        refreshChart({ force: true }).then(() => setStatus(warning, false));
      } else {
        setStatus(warning, false);
      }
    }
  }

  async function requestDateBucketSuggestionForSelectedColumn(dateBucketKey = currentDateBucketFeatureKey()) {
    if (!state.schema || !state.x || state.dateBucketSuggestionPendingKey === dateBucketKey) return;
    const column = selectedColumn();
    if (!isDateKind(column?.kind) || state.dateBucketManualKey === dateBucketKey) return;
    const requestSeq = (state.dateBucketSuggestionRequestSeq || 0) + 1;
    state.dateBucketSuggestionRequestSeq = requestSeq;
    state.dateBucketSuggestionPendingKey = dateBucketKey;
    syncDateBucketControl();
    try {
      const sourceId = selectedColumn()?.source_id || state.xSource || state.source || "dataset";
      const data = await api("/api/date-bucket/suggestion", {
        method: "POST",
        body: JSON.stringify({
          source: sourceId,
          xSource: sourceId,
          feature: state.x,
          filter: state.activeFilter,
        }),
      });
      if (requestSeq !== state.dateBucketSuggestionRequestSeq || state.dateBucketSuggestionPendingKey !== dateBucketKey) return;
      if (currentDateBucketFeatureKey() !== dateBucketKey || state.dateBucketManualKey === dateBucketKey) return;
      state.dateBucket = normaliseDateBucket(data.date_bucket);
      state.dateBucketFeature = dateBucketKey;
      clearPendingDateBucketSuggestion();
      syncDateBucketControl();
      if (state.tool === "line_bar") refreshChart({ force: true });
    } catch (error) {
      if (requestSeq !== state.dateBucketSuggestionRequestSeq || state.dateBucketSuggestionPendingKey !== dateBucketKey) return;
      if (currentDateBucketFeatureKey() !== dateBucketKey || state.dateBucketManualKey === dateBucketKey) return;
      state.dateBucket = "year";
      state.dateBucketFeature = dateBucketKey;
      clearPendingDateBucketSuggestion();
      syncDateBucketControl();
      const warning = `Date bucket estimate failed; using Year. ${error.message}`;
      if (state.tool === "line_bar") {
        refreshChart({ force: true }).then(() => setStatus(warning, false));
      } else {
        setStatus(warning, false);
      }
    }
  }

  function stepBandWidth(direction) {
    clearPendingBandSuggestion();
    const current = Number(state.bandWidth) > 0 ? Number(state.bandWidth) : Number(fallbackBandWidthForSelectedColumn()) || 1;
    let next = current;
    if (direction < 0) {
      const smallerSteps = bandSteps.filter((step) => step < current);
      next = smallerSteps.length ? smallerSteps[smallerSteps.length - 1] : current;
    } else {
      next = bandSteps.find((step) => step > current) || current;
    }
    state.bandWidth = state.quantileMode === "quantile" ? String(quantileCountForBandWidth(next)) : formatBandWidth(next);
    state.bandFeature = currentBandFeatureKey();
    syncBandingControl();
    refreshChart();
  }

  function updateAxisControls() {
    const kind = selectedColumn()?.kind;
    const isDate = kind === "date" || kind === "datetime";
    const isNumeric = isNumericKind(kind);
    const isCategorical = kind === "categorical";
    const hasExpected = expectedSelections().length > 0;
    const modelControlsAvailable = toolEnabled("gbm") || toolEnabled("glm");
    const shapSortAvailable = isCategorical && shapPartialDependenceVisible() && shapOverlayAvailableForSelectedColumn();
    el("partialDependenceControl").classList.toggle("hidden", !modelControlsAvailable);
    el("responseTransformControl").classList.toggle("hidden", !modelControlsAvailable);
    el("sigmaControl").classList.toggle("hidden", !hasExpected);
    el("sortControl").classList.toggle("hidden", !isCategorical);
    el("expectedSortButton").classList.toggle("hidden", !hasExpected);
    el("shapSortButton")?.classList.toggle("hidden", !shapSortAvailable);
    el("dateControl").classList.toggle("hidden", !isDate);
    el("bandControl").classList.toggle("hidden", !isNumeric);
    el("quantileControl").classList.toggle("hidden", !isNumeric);
    syncSegmented("lowGroup", state.lowGroup);
    syncSegmented("labels", state.labels);
    syncSegmented("transform", state.transform);
    syncSegmented("sigma", state.sigma);
    syncSegmented("partialDependence", selectedPartialDependenceMode());
    syncSegmented("featureSort", state.featureSort);
    syncSegmented("expectedSort", state.expectedSort);
    const bandFeatureKey = currentBandFeatureKey();
    const dateBucketKey = currentDateBucketFeatureKey();
    if (isNumeric && state.quantileMode === "quantile" && state.bandFeature !== bandFeatureKey) {
      state.bandFeature = bandFeatureKey;
      clearPendingBandSuggestion();
    }
    if (isNumeric && state.quantileMode !== "quantile" && state.tool === "line_bar" && state.bandFeature !== bandFeatureKey) {
      requestBandSuggestionForSelectedColumn(bandFeatureKey);
    }
    if (isDate && state.tool === "line_bar" && state.dateBucketManualKey !== dateBucketKey && state.dateBucketFeature !== dateBucketKey) {
      requestDateBucketSuggestionForSelectedColumn(dateBucketKey);
    }
    if (isNumeric && state.quantileMode === "quantile") {
      normalizeBandWidthForQuantiles();
    }
    if (!isCategorical || (state.sort === "expected" && !hasExpected) || (state.sort === "shap" && !shapSortAvailable)) {
      state.sort = "alpha";
      syncSegmented("sort", "alpha");
    } else {
      syncSegmented("sort", state.sort);
    }
    if (!isDate) {
      state.dateBucket = "none";
      resetDateBucketSuggestionState();
      syncDateBucketControl();
    } else {
      syncDateBucketControl();
    }
    if (!isNumeric) {
      state.bandWidth = "0";
      state.quantileMode = "off";
      state.bandFeature = bandFeatureKey;
      clearPendingBandSuggestion();
      syncSegmented("bandWidth", "0");
    }
    syncBandingControl();
    syncQuantileControl();
  }

  function expectedSelections() {
    return Array.isArray(state.expectedSelections) ? state.expectedSelections : [];
  }

  function expectedSelectionKey(value, sourceId = "") {
    return `${sourceId || ""}\u0000${value || ""}`;
  }

  function expectedButtonSelection(value, sourceId = "") {
    return expectedSelections().find((selection) => (
      selection.value === value && (selection.sourceId || "") === (sourceId || "")
    )) || null;
  }

  function expectedSelectionOption(value, sourceId = "") {
    const select = el("expectedNumerator");
    const options = Array.from(select.options);
    return options.find((option) => (
      !option.disabled &&
      option.value === value &&
      (!sourceId || option.dataset.sourceId === sourceId)
    )) || options.find((option) => !option.disabled && option.value === value) || null;
  }

  function syncExpectedSelectToFirstSelection() {
    const first = expectedSelections()[0];
    const option = first ? expectedSelectionOption(first.value, first.sourceId) : null;
    if (option) option.selected = true;
    else el("expectedNumerator").value = "";
  }

  function toggleExpectedSelection(value, sourceId = "") {
    const current = expectedSelections();
    if (!value) {
      if (!current.length) return false;
      state.expectedSelections = [];
      syncExpectedSelectToFirstSelection();
      return true;
    }
    const existingIndex = current.findIndex((selection) => (
      selection.value === value && (selection.sourceId || "") === (sourceId || "")
    ));
    if (existingIndex >= 0) {
      state.expectedSelections = current.filter((_, index) => index !== existingIndex);
      syncExpectedSelectToFirstSelection();
      return true;
    }
    if (current.length >= 2) return false;
    const option = expectedSelectionOption(value, sourceId);
    const nextSelection = {
      value,
      sourceId: option?.dataset.sourceId || sourceId || state.source || "dataset",
      metricKind: option?.dataset.metricKind || "metric",
    };
    state.expectedSelections = [...current, nextSelection];
    syncExpectedSelectToFirstSelection();
    return true;
  }

  function replaceExpectedSelection(value, sourceId = "") {
    const current = expectedSelections();
    if (!value) {
      if (!current.length) return false;
      state.expectedSelections = [];
      syncExpectedSelectToFirstSelection();
      return true;
    }
    const option = expectedSelectionOption(value, sourceId);
    const nextSelection = {
      value,
      sourceId: option?.dataset.sourceId || sourceId || state.source || "dataset",
      metricKind: option?.dataset.metricKind || "metric",
    };
    const unchanged = current.length === 1
      && expectedSelectionKey(current[0].value, current[0].sourceId) === expectedSelectionKey(nextSelection.value, nextSelection.sourceId);
    if (unchanged) return false;
    state.expectedSelections = [nextSelection];
    syncExpectedSelectToFirstSelection();
    return true;
  }

  function activateExpectedKeyboardSelection(button) {
    const value = button?.dataset.value || "";
    const sourceId = button?.dataset.sourceId || "";
    if (!replaceExpectedSelection(value, sourceId)) return;
    const sourceChanged = syncExpectedSourceFromSelection({
      expectedValue: value,
      expectedSource: sourceId,
      expectedSelections: state.expectedSelections,
    });
    if (!sourceChanged) {
      renderExpectedNumerators({ preserveScroll: true });
      updateAxisControls();
    }
    clearActiveFavouriteSelection();
    refreshChart({ force: sourceChanged });
  }

  function renderExpectedNumerators(options = {}) {
    const query = el("expectedSearch").value.trim().toLowerCase();
    const list = el("expectedList");
    const scrollPosition = captureLineBarPickerScroll(list, options.preserveScroll);
    const { pinned, scroll } = resetLineBarPickerList(list, true);
    const selections = expectedSelections();
    const selectedKeys = new Set(selections.map((selection) => expectedSelectionKey(selection.value, selection.sourceId)));
    const maxSelected = selections.length >= 2;

    function addExpectedButton(target, label, value, kind, sourceId = "", extraClass = "") {
      const isActive = value
        ? selectedKeys.has(expectedSelectionKey(value, sourceId))
        : selections.length === 0;
      const disabled = Boolean(value && maxSelected && !isActive);
      const button = document.createElement("button");
      button.type = "button";
      button.className = `feature ${extraClass} ${isActive ? "active" : ""}`.trim();
      button.disabled = disabled;
      button.setAttribute("aria-pressed", String(isActive));
      if (sourceId) button.dataset.sourceId = sourceId;
      button.dataset.value = value;
      button.innerHTML = `<span>${escapeHtml(label)}</span><span class="kind">${escapeHtml(kind)}</span>`;
      button.addEventListener("click", (event) => {
        const previousSelections = expectedSelections().map((selection) => ({ ...selection }));
        const changed = toggleExpectedSelection(value, sourceId);
        if (!changed) return;
        const sourceChanged = syncExpectedSourceFromSelection({
          expectedValue: value,
          expectedSource: sourceId,
          expectedSelections: state.expectedSelections,
        });
        if (!sourceChanged) {
          renderExpectedNumerators({ preserveScroll: true });
          updateAxisControls();
        }
        clearActiveFavouriteSelection();
        refreshChart({ force: sourceChanged });
        if (event.isTrusted) {
          const remainingSelection = expectedButtonSelection(value, sourceId);
          const fallback = remainingSelection || state.expectedSelections[0] || previousSelections[0] || { value: "", sourceId: "" };
          focusLineBarPickerButton(list, { value: fallback.value, sourceId: fallback.sourceId || "", index: 0 });
        }
      });
      target.append(button);
    }

    if (!query || "none".includes(query) || "no expected line".includes(query) || "off".includes(query)) {
      addExpectedButton(pinned, "No expected line", "", "off", "", "expected-none-option line-bar-special-row");
    }

    const { specialColumns, otherColumns } = expectedDisplayColumns();
    for (const col of specialColumns) {
      if (query && !col.name.toLowerCase().includes(query)) continue;
      addExpectedButton(pinned, col.name, col.name, col.kind, col.source_id || state.source || "dataset", "line-bar-special-row");
    }
    pinned.hidden = pinned.childElementCount === 0;

    for (const col of otherColumns) {
      if (query && !col.name.toLowerCase().includes(query)) continue;
      addExpectedButton(scroll, col.name, col.name, col.kind, col.source_id || state.source || "dataset");
    }
    restoreLineBarPickerScroll(list, scrollPosition);
  }

  function renderFeatures(options = {}) {
    const query = el("featureSearch").value.trim().toLowerCase();
    const list = el("featureList");
    const scrollPosition = captureLineBarPickerScroll(list, options.preserveScroll);
    syncFeatureImportanceButton();
    ensureFeatureImportance();
    if (state.featureSort === "importance") {
      const ratioColumns = orderedLineBarSpecialColumns([...sourceColumns()]).filter((column) => (
        isGbmGlmRatioColumn(column) && featureMatchesQuery(column.name, query)
      ));
      if (ratioColumns.length) {
        const { pinned, scroll } = resetLineBarPickerList(list, true);
        for (const col of ratioColumns) {
          addLineBarFeatureButton(pinned, col, "line-bar-special-row");
        }
        renderFeatureImportanceRows(query, scroll);
      } else {
        resetLineBarPickerList(list, false);
        renderFeatureImportanceRows(query, list);
      }
      restoreLineBarPickerScroll(list, scrollPosition);
      return;
    }
    const columns = [...sourceColumns()];
    const specialColumns = orderedLineBarSpecialColumns(columns).filter((column) => featureMatchesQuery(column.name, query));
    const otherColumns = columns.filter((column) => !isLineBarSpecialColumn(column));
    if (state.featureSort === "alpha") {
      otherColumns.sort((a, b) => a.name.localeCompare(b.name, undefined, { sensitivity: "base" }));
    }
    const { pinned, scroll } = resetLineBarPickerList(list, true);
    for (const col of specialColumns) {
      addLineBarFeatureButton(pinned, col, "line-bar-special-row");
    }
    pinned.hidden = pinned.childElementCount === 0;

    for (const col of otherColumns) {
      if (!featureMatchesQuery(col.name, query)) continue;
      addLineBarFeatureButton(scroll, col);
    }
    restoreLineBarPickerScroll(list, scrollPosition);
  }

  function addLineBarFeatureButton(list, col, extraClass = "") {
    const sourceId = col.source_id || state.source || "dataset";
    const active = col.name === state.x && (!state.xSource || state.xSource === sourceId);
    const isRawDatasetFeature = sourceId === "dataset" && !isModelPredictionColumn(col) && !isGbmGlmRatioColumn(col);
    addFeatureButton(list, {
      label: col.name,
      detail: col.kind,
      sourceId,
      extraClass,
      active,
      onClick: () => {
        const previousDateBucketKey = currentDateBucketFeatureKey();
        const changed = state.x !== col.name || state.xSource !== sourceId;
        if (isRawDatasetFeature) state.source = "dataset";
        state.x = col.name;
        state.xSource = sourceId;
        resetDateBucketSuggestionIfKeyChanged(previousDateBucketKey);
        renderFeatures({ preserveScroll: true });
        updateAxisControls();
        if (changed) clearActiveFavouriteSelection();
        refreshChart();
      },
    });
  }

  function renderFeatureImportanceRows(query, list) {
    const data = currentFeatureImportanceData();
    if (!data) {
      addFeatureListMessage(list, featureImportanceError || "Loading feature importances...");
      return;
    }
    const datasetColumns = datasetFeatureColumns(data);
    const datasetByName = new Map(datasetColumns.map((column) => [column.name, column]));
    const usedFeatures = new Set();
    const renderedAny = [
      renderImportanceGroup(list, "GBM", data.models?.gbm, query, datasetByName, usedFeatures),
      renderImportanceGroup(list, "GLM", data.models?.glm, query, datasetByName, usedFeatures),
    ].some(Boolean);
    const notUsed = datasetColumns
      .filter((column) => !isLineBarSpecialColumn(column))
      .filter((column) => !usedFeatures.has(column.name))
      .map((column) => ({ feature: column.name, importance: null, kind: column.kind }));
    renderNotUsedGroup(list, notUsed, query);
    if (!renderedAny && !featureImportanceHasRows(data)) {
      list.innerHTML = "";
      addFeatureListMessage(list, "No active GBM or GLM importances are available.");
    }
  }

  function datasetFeatureColumns(data) {
    const rows = Array.isArray(data?.dataset_features) ? data.dataset_features : [];
    if (rows.length) return rows.map((column) => ({ ...column, source_id: "dataset" }));
    return (dataSourceForId("dataset")?.columns || []).map((column) => ({ ...column, source_id: "dataset" }));
  }

  function renderImportanceGroup(list, label, model, query, datasetByName, usedFeatures) {
    const rows = Array.isArray(model?.rows) ? model.rows : [];
    rows.forEach((row) => {
      if (isLineBarSpecialColumn({ name: row?.feature })) return;
      if (row?.feature) usedFeatures.add(String(row.feature));
    });
    const filtered = rows.filter((row) => !isLineBarSpecialColumn({ name: row?.feature }) && featureMatchesQuery(row.feature, query));
    const message = String(model?.message || "");
    if (!filtered.length && (!message || query)) return false;
    addFeatureListHeader(list, importanceGroupLabel(label, model));
    if (!filtered.length && message) {
      addFeatureListMessage(list, message);
      return true;
    }
    for (const row of filtered) {
      const column = datasetByName.get(String(row.feature)) || { name: String(row.feature), kind: String(row.kind || "") };
      addFeatureButton(list, {
        label: column.name,
        detail: featureImportanceDetail(row, model?.metric),
        sourceId: "dataset",
        extraClass: "line-bar-importance-row",
        onClick: () => selectDatasetFeature(column.name),
      });
    }
    return true;
  }

  function renderNotUsedGroup(list, rows, query) {
    const filtered = rows.filter((row) => featureMatchesQuery(row.feature, query));
    if (!filtered.length) return;
    addFeatureListHeader(list, "Not used");
    for (const row of filtered) {
      addFeatureButton(list, {
        label: row.feature,
        detail: row.kind || "",
        sourceId: "dataset",
        extraClass: "line-bar-not-used-row",
        onClick: () => selectDatasetFeature(row.feature),
      });
    }
  }

  function importanceGroupLabel(label, model) {
    const metric = String(model?.metric_label || "");
    return metric ? `${label} (${metric})` : label;
  }

  function addFeatureListHeader(list, label) {
    const header = document.createElement("div");
    header.className = "feature-list-section-header";
    header.textContent = label;
    list.append(header);
  }

  function addFeatureListMessage(list, message) {
    const item = document.createElement("div");
    item.className = "feature-list-message";
    item.textContent = message;
    list.append(item);
  }

  function resetLineBarPickerList(list, split) {
    list.innerHTML = "";
    list.classList.toggle("line-bar-split-list", Boolean(split));
    if (!split) return { pinned: list, scroll: list };
    const pinned = document.createElement("div");
    pinned.className = "line-bar-pinned-region";
    const scroll = document.createElement("div");
    scroll.className = "line-bar-scroll-region";
    list.append(pinned, scroll);
    return { pinned, scroll };
  }

  function lineBarPickerScrollNode(list) {
    return list.querySelector(".line-bar-scroll-region") || list;
  }

  function captureLineBarPickerScroll(list, preserveScroll = false) {
    if (!preserveScroll) return null;
    const scrollNode = lineBarPickerScrollNode(list);
    return { top: scrollNode.scrollTop };
  }

  function restoreLineBarPickerScroll(list, position) {
    if (!position) return;
    const scrollNode = lineBarPickerScrollNode(list);
    const maxTop = Math.max(0, scrollNode.scrollHeight - scrollNode.clientHeight);
    scrollNode.scrollTop = Math.min(position.top, maxTop);
  }

  function lineBarPickerButtons(list, { includeDisabled = false } = {}) {
    return Array.from(list.querySelectorAll("button.feature"))
      .filter((button) => (includeDisabled || !button.disabled) && button.offsetParent !== null);
  }

  function currentLineBarPickerButton(list, buttons) {
    const focused = document.activeElement;
    if (focused instanceof HTMLButtonElement && list.contains(focused) && focused.matches("button.feature")) {
      return focused;
    }
    return buttons.find((button) => button.classList.contains("active")) || null;
  }

  function focusLineBarPickerButton(list, targetState) {
    requestAnimationFrame(() => {
      const buttons = lineBarPickerButtons(list);
      if (!buttons.length) return;
      const target = buttons.find((button) => (
        (button.dataset.value || "") === targetState.value
        && (button.dataset.sourceId || "") === targetState.sourceId
      )) || buttons.find((button) => button.classList.contains("active")) || buttons[Math.min(targetState.index, buttons.length - 1)];
      target.focus({ preventScroll: true });
      target.scrollIntoView({ block: "nearest" });
    });
  }

  function handleLineBarPickerKeydown(event, searchInputId, listId) {
    if (!["ArrowUp", "ArrowDown"].includes(event.key)) return;
    const list = el(listId);
    const isExpectedPicker = listId === "expectedList";
    const buttons = lineBarPickerButtons(list, { includeDisabled: isExpectedPicker });
    if (!buttons.length) return;
    const currentButton = currentLineBarPickerButton(list, buttons);
    const currentIndex = currentButton ? buttons.indexOf(currentButton) : -1;
    const direction = event.key === "ArrowDown" ? 1 : -1;
    const nextIndex = currentIndex < 0
      ? (direction > 0 ? 0 : buttons.length - 1)
      : Math.min(Math.max(currentIndex + direction, 0), buttons.length - 1);
    event.preventDefault();
    if (nextIndex === currentIndex) return;
    const target = buttons[nextIndex];
    const targetState = {
      value: target.dataset.value || "",
      sourceId: target.dataset.sourceId || "",
      index: nextIndex,
    };
    const startedFromButton = event.target instanceof HTMLButtonElement && list.contains(event.target);
    list.classList.add("line-bar-keyboard-navigation");
    if (isExpectedPicker) activateExpectedKeyboardSelection(target);
    else target.click();
    if (startedFromButton) {
      focusLineBarPickerButton(list, targetState);
    } else {
      requestAnimationFrame(() => el(searchInputId).focus({ preventScroll: true }));
    }
  }

  function bindLineBarPickerKeyboard(searchInputId, listId) {
    const handler = (event) => handleLineBarPickerKeydown(event, searchInputId, listId);
    el(searchInputId).addEventListener("keydown", handler);
    const list = el(listId);
    list.addEventListener("keydown", handler);
    list.addEventListener("pointermove", () => {
      list.classList.remove("line-bar-keyboard-navigation");
    });
  }

  function addFeatureButton(list, { label, detail, sourceId, extraClass = "", active = null, onClick }) {
    const activeSource = state.xSource || state.source || "dataset";
    const isActive = active === null ? label === state.x && activeSource === sourceId : Boolean(active);
    const button = document.createElement("button");
    button.className = `feature ${extraClass} ${isActive ? "active" : ""}`.trim();
    button.dataset.sourceId = sourceId;
    button.dataset.value = label;
    button.innerHTML = `<span>${escapeHtml(label)}</span><span class="kind">${escapeHtml(detail)}</span>`;
    button.addEventListener("click", (event) => {
      const pickerList = list.closest("#featureList") || list;
      onClick();
      if (event.isTrusted) {
        focusLineBarPickerButton(pickerList, { value: label, sourceId, index: 0 });
      }
    });
    list.append(button);
  }

  function selectDatasetFeature(featureName) {
    const previousDateBucketKey = currentDateBucketFeatureKey();
    state.source = "dataset";
    state.x = featureName;
    state.xSource = "dataset";
    resetDateBucketSuggestionIfKeyChanged(previousDateBucketKey);
    renderFeatures({ preserveScroll: true });
    renderExpectedNumerators();
    updateAxisControls();
    refreshChart({ force: true });
  }

  function featureMatchesQuery(featureName, query) {
    return !query || String(featureName || "").toLowerCase().includes(query);
  }

  function formatFeatureImportance(value, metric) {
    const number = Number(value);
    if (!Number.isFinite(number)) return "";
    if (metric === "gain") return formatGain(number);
    return formatCompactImportance(number);
  }

  function featureImportanceDetail(row, metric) {
    const value = formatFeatureImportance(row?.importance, metric);
    const rank = Number(row?.rank);
    const prefix = Number.isFinite(rank) ? `Rank ${rank}` : "";
    return value ? `${prefix} · ${value}` : prefix;
  }

  function formatCompactImportance(number) {
    const magnitude = Math.abs(number);
    if (magnitude < 0.00005) return "0.0000";
    if (magnitude >= 1000) return Math.round(number).toLocaleString();
    if (magnitude >= 10) return number.toFixed(1);
    if (magnitude >= 1) return number.toFixed(3);
    return number.toFixed(4);
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

  function clearSearchInput(inputId, render) {
    const input = el(inputId);
    if (input.value) {
      input.value = "";
      render();
    }
    input.focus();
  }

  function currentResponses() {
    const responses = [];
    if (el("actualNumerator").value) {
      const option = el("actualNumerator").selectedOptions[0];
      const source = option?.dataset.metricKind === "prediction" ? option.dataset.sourceId || "" : "";
      responses.push({
        label: el("actualNumerator").value,
        numerator: el("actualNumerator").value,
        ...(source ? { source } : {}),
      });
    }
    for (const selection of expectedSelections()) {
      const source = selection.metricKind === "prediction" ? selection.sourceId || "" : "";
      responses.push({
        label: selection.value,
        numerator: selection.value,
        ...(source ? { source } : {}),
      });
    }
    return responses;
  }

  function buildChartRequest() {
    if (!state.schema || !state.x) return null;
    const kind = selectedColumn()?.kind;
    const isDate = kind === "date" || kind === "datetime";
    const isNumeric = isNumericKind(kind);
    const bandFeatureKey = currentBandFeatureKey();
    const dateBucketKey = currentDateBucketFeatureKey();
    if (isNumeric && state.quantileMode === "quantile" && state.bandFeature !== bandFeatureKey) {
      normalizeBandWidthForQuantiles();
    }
    if (isNumeric && state.quantileMode !== "quantile" && state.bandFeature !== bandFeatureKey) {
      requestBandSuggestionForSelectedColumn(bandFeatureKey);
      return null;
    }
    if (isNumeric && state.quantileMode !== "quantile" && state.bandSuggestionPendingKey === bandFeatureKey) {
      setGroupMeta("line_bar", "Estimating banding...");
      return null;
    }
    if (isDate && state.dateBucketManualKey !== dateBucketKey && state.dateBucketFeature !== dateBucketKey) {
      requestDateBucketSuggestionForSelectedColumn(dateBucketKey);
      return null;
    }
    if (isDate && state.dateBucketSuggestionPendingKey === dateBucketKey) {
      setGroupMeta("line_bar", "Estimating date bucket...");
      return null;
    }
    const column = selectedColumn();
    const sourceId = column?.source_id || state.xSource || state.source || "dataset";
    const xSource = column && (isModelPredictionColumn(column) || sourceId !== (state.source || "dataset")) ? sourceId : "";
    return {
      source: state.source || "dataset",
      ...(xSource ? { xSource } : {}),
      x: state.x,
      sort: state.sort,
      lowGroup: state.lowGroup,
      bandWidth: isNumeric ? Number(state.bandWidth) : 0,
      quantileMode: isNumeric ? state.quantileMode : "off",
      dateBucket: isDate ? state.dateBucket : "none",
      transform: state.transform,
      partialDependence: { mode: selectedPartialDependenceMode() },
      base: selectedFeatureBase(),
      sigma: Number(state.sigma),
      filter: state.activeFilter,
      denominator: el("denominator").value,
      responses: currentResponses(),
      maxGroups: 10000,
    };
  }

  async function refreshChart(options = {}) {
    return refreshLineBar(options);
  }

  async function fetchChartData(request, requestKey) {
    const requestSeq = state.chartRequestSeq + 1;
    state.chartRequestSeq = requestSeq;
    setStatus("");
    setChartMessage("");
    setGroupMeta("line_bar", "Computing...");
    startToolTiming("line_bar");
    updateAxisControls();
    try {
      const data = await api("/api/chart", { method: "POST", body: JSON.stringify(request), clientTiming: true });
      if (requestSeq !== state.chartRequestSeq) return;
      const cache = toolCache("line_bar");
      cache.requestKey = requestKey;
      cache.data = data;
      syncDuckDbTimingFromData("line_bar", data);
      syncClientTimingFromData("line_bar", data);
      measureToolRender("line_bar", () => renderChartData(data, { resetTablePage: true }));
      return data;
    } catch (error) {
      if (requestSeq !== state.chartRequestSeq) return;
      setToolTimingFailed("line_bar");
      setGroupMeta("line_bar", "Query failed");
      setChartMessage("");
      setStatus(error.message, true);
    }
  }

  function cancelLineBarRequests() {
    state.chartRequestSeq += 1;
    tableRequestSeq += 1;
  }

  function lineBarExclusionWarnings(data) {
    return Array.isArray(data?.exclusion_warnings) ? data.exclusion_warnings.filter(Boolean) : [];
  }

  function lineBarDisplayWarnings(data) {
    const exclusionSet = new Set(lineBarExclusionWarnings(data));
    return [...(data?.warnings || [])].filter((warning) => warning && !exclusionSet.has(warning));
  }

  function lineBarGroupMetaWithExclusions(groupLabel, rowMeta, data) {
    const base = `${groupLabel} · ${rowMeta}`;
    const exclusions = lineBarExclusionWarnings(data);
    return exclusions.length ? `${base}. ${exclusions.join(". ")}` : base;
  }

  function syncLineBarTablePresentation(data) {
    const tableMeta = data?.table || {};
    const groupCount = Math.max(0, Number(tableMeta.group_count ?? data?.rows?.length ?? 0) || 0);
    const matchCount = Math.max(0, Number(tableMeta.match_count ?? groupCount) || 0);
    const rowMeta = formatRowMeta(data?.row_count, data?.filtered_row_count);
    const groupLabel = state.lineBarTableSearch && matchCount !== groupCount
      ? `${matchCount.toLocaleString()} of ${groupCount.toLocaleString()} groups`
      : `${groupCount.toLocaleString()} groups`;
    const groupMeta = lineBarGroupMetaWithExclusions(groupLabel, rowMeta, data);
    const chartMessage = lineBarDisplayWarnings(data).join(" ");
    setGroupMeta("line_bar", groupMeta);
    setStatus("");
    setChartMessage(chartMessage);
    saveToolPresentation("line_bar", { groupMeta, chartMessage });
  }

  function setChartPendingHidden(hidden) {
    el("chart").style.visibility = hidden ? "hidden" : "";
  }

  function renderChartData(data, options = {}) {
    setChartPendingHidden(false);
    state.lastData = data;
    if (options.resetTablePage) {
      state.tablePage = 1;
    }
    invalidateLineBarTableCache();
    updateMetricTitles(data);
    const labelMessage = renderChart(data);
    renderTableShell();
    if (state.view === "table") refreshLineBarTable({ force: true });
    const rowMeta = formatRowMeta(data.row_count, data.filtered_row_count);
    const groupCount = Number.isFinite(Number(data.group_count)) ? Number(data.group_count) : data.rows.length;
    const groupLabel = `${groupCount.toLocaleString()} groups`;
    const groupMeta = lineBarGroupMetaWithExclusions(groupLabel, rowMeta, data);
    const warnings = lineBarDisplayWarnings(data).join(" ");
    const chartMessage = [warnings, labelMessage].filter(Boolean).join(" ");
    setGroupMeta("line_bar", groupMeta);
    setStatus("");
    setChartMessage(chartMessage);
    saveToolPresentation("line_bar", { groupMeta, chartMessage });
  }

  function useCachedChartData(cache, options = {}) {
    state.lastData = cache.data;
    if (options.renderIfCached) {
      measureToolRender("line_bar", () => renderChartData(cache.data));
      return;
    }
    measureToolRender("line_bar", () => {
      updateMetricTitles(cache.data);
      applyToolPresentation("line_bar");
      requestAnimationFrame(() => {
        chart.resize();
        refreshDateXAxisLabelsForCurrentZoom();
      });
    });
  }

  function renderChart(data) {
    if (data?.groups_truncated && !(data.rows || []).length) {
      dateXAxisContext = null;
      chart.clear();
      return "";
    }
    const labels = data.rows.map((r) => formatChartXLabel(r, data));
    const labelMode = state.labels;
    const renderTransform = String(state.transform || "none");
    const formatChartResponseValue = chartResponseFormatter(renderTransform);
    const rawXValues = data.rows.map((r) => r.x);
    const dateBucket = normaliseDateBucket(data.date_bucket);
    const displayKind = data.x_group_kind || data.x_kind;
    const xLabelPolicy = getXAxisLabelPolicy(labels, displayKind, rawXValues, dateBucket, chart.getWidth?.() || el("chart").clientWidth);
    dateXAxisContext = isDateKind(data.x_kind)
      ? { labels, rawXValues, dateBucket, xKind: data.x_kind }
      : null;
    const dataLabelsAllowed = labels.length < LABEL_DENSITY_LIMIT;
    const showBarLabels = dataLabelsAllowed && (labelMode === "bar" || labelMode === "all");
    const showLineLabels = dataLabelsAllowed && (labelMode === "line" || labelMode === "all");
    const barLayout = getBarLayout(labels.length);
    const shapSeries = shapPartialDependenceSeries(data);
    const glmSeries = glmPartialDependenceSeries(data);
    const overlayLegendData = [...shapSeries, ...glmSeries].map((series) => series.name);
    const hasOverlaySeries = overlayLegendData.length > 0;
    const previousOption = chart.getOption();
    const actualColor = getCss("--actual-line");
    const expectedColor = "#d13f3f";
    const secondExpectedColor = getCss("--accent") || "#2276d2";
    const responseColors = [actualColor, expectedColor, secondExpectedColor];
    const nColor = getCss("--bar");
    const weightLabel = data.denominator?.bar_label || "Weight";
    const sigmaColor = "#8a94a6";
    const legendData = [
      ...data.responses.map((response) => response.label),
      { name: weightLabel, icon: "roundRect", itemStyle: { color: nColor, borderColor: nColor } },
    ];
    const mainLegendSelection = matchingLegendSelection(previousOption, legendData);
    const overlayLegendSelection = matchingLegendSelection(previousOption, overlayLegendData);
    const responseAxis = responseAxisOptions(data, { ...mainLegendSelection, ...overlayLegendSelection }, renderTransform);
    const barSeries = {
      name: weightLabel,
      type: "bar",
      yAxisIndex: 1,
      z: 1,
      legendHoverLink: true,
      itemStyle: { color: nColor },
      animation: false,
      animationDuration: 0,
      animationDurationUpdate: 0,
      data: data.rows.map((r) => ({
        value: r.volume,
        itemStyle: { color: weightBarColor(data, r) },
      })),
      label: { show: showBarLabels, position: "top", fontSize: 10, formatter: formatChartLabel, ...lineBarChartLabelStyle() },
      barWidth: barLayout.width,
      barMaxWidth: barLayout.maxWidth,
      barCategoryGap: barLayout.categoryGap,
    };
    const lineSeries = data.responses.map((response, index) => ({
      name: response.label,
      type: "line",
      yAxisIndex: 0,
      z: 3,
      animation: false,
      animationDuration: 0,
      animationDurationUpdate: 0,
      smooth: false,
      showSymbol: data.rows.length < 250,
      symbolSize: 5,
      lineStyle: { color: responseColors[index] || actualColor },
      itemStyle: { color: responseColors[index] || actualColor },
      data: data.rows.map((r) => r[`resp${index}`]),
      showAllSymbol: true,
      label: { show: showLineLabels, fontSize: 10, formatter: (params) => formatResponseLabel(params, formatChartResponseValue), ...lineBarChartLabelStyle() },
    }));
    const upliftBaseline = upliftBaselineSeries(data, renderTransform);

    const customSeries = [];
    if (Number(state.sigma) > 0 && data.responses.length >= 2) {
      customSeries.push({
        name: "sigma",
        type: "custom",
        yAxisIndex: 0,
        z: 5,
        legendHoverLink: false,
        animation: false,
        animationDuration: 0,
        animationDurationUpdate: 0,
        renderItem: function (params, api) {
          const x = api.coord([api.value(0), api.value(1)])[0];
          const low = api.coord([api.value(0), api.value(2)])[1];
          const high = api.coord([api.value(0), api.value(3)])[1];
          if (!Number.isFinite(low) || !Number.isFinite(high)) return;
          return {
            type: "group",
            children: [
              { type: "line", shape: { x1: x, y1: low, x2: x, y2: high }, style: { stroke: sigmaColor, lineWidth: 1.5 } },
              { type: "line", shape: { x1: x - 4, y1: low, x2: x + 4, y2: low }, style: { stroke: sigmaColor, lineWidth: 1.5 } },
              { type: "line", shape: { x1: x - 4, y1: high, x2: x + 4, y2: high }, style: { stroke: sigmaColor, lineWidth: 1.5 } },
            ],
          };
        },
        data: data.rows.map((r, i) => [i, r.resp1, r.resp1_low, r.resp1_high]).filter((r) => r.every((v) => v !== null && v !== undefined)),
        encode: { x: 0, y: [2, 3] },
        tooltip: { show: false },
      });
    }

    chart.setOption(
      {
        animation: false,
        animationDuration: 0,
        animationDurationUpdate: 0,
        stateAnimation: { duration: 0 },
        backgroundColor: "transparent",
        color: [actualColor, expectedColor, secondExpectedColor, nColor],
        tooltip: {
          trigger: "axis",
          formatter: (params) => formatChartTooltip(params, weightLabel, formatChartResponseValue),
        },
        legend: lineBarLegendOptions(legendData, mainLegendSelection, overlayLegendData, overlayLegendSelection),
        grid: { left: 72, right: 76, top: hasOverlaySeries ? LINE_BAR_OVERLAY_GRID_TOP : LINE_BAR_GRID_TOP, bottom: xLabelPolicy.bottom, containLabel: false },
        xAxis: {
          type: "category",
          name: data.x || "",
          nameLocation: "middle",
          nameGap: xLabelPolicy.nameGap,
          nameTextStyle: { color: getCss("--text"), fontSize: 13, fontWeight: 700 },
          data: labels,
          axisLabel: {
            show: xLabelPolicy.show,
            color: getCss("--text"),
            interval: xLabelPolicy.interval,
            formatter: xLabelPolicy.formatter,
            hideOverlap: Boolean(xLabelPolicy.hideOverlap),
            showMinLabel: xLabelPolicy.showMinLabel,
            showMaxLabel: xLabelPolicy.showMaxLabel,
            rotate: xLabelPolicy.rotate,
            fontSize: xLabelPolicy.fontSize,
            margin: 8,
          },
          axisLine: { lineStyle: { color: getCss("--line") } },
        },
        yAxis: [
          { type: "value", scale: true, splitNumber: RESPONSE_AXIS_TARGET_INTERVALS, min: responseAxis.min, max: responseAxis.max, interval: responseAxis.interval, axisLabel: { color: getCss("--text"), formatter: (value) => formatChartResponseValue(value) }, splitLine: { lineStyle: { color: getCss("--line") } } },
          { type: "value", axisLabel: { color: getCss("--text"), formatter: (value) => formatNumber(value) }, splitLine: { show: false } },
        ],
        dataZoom: xLabelPolicy.dataZoomEnabled ? lineBarDataZoomOptions() : [],
        series: [barSeries, ...shapSeries, ...glmSeries, ...lineSeries, ...(upliftBaseline ? [upliftBaseline] : []), ...customSeries],
      },
      true,
    );
    chartRenderTransform = renderTransform;
    requestAnimationFrame(() => {
      chart.resize();
      refreshDateXAxisLabelsForCurrentZoom();
    });
    return chartDensityMessage(labels.length, !xLabelPolicy.show, !dataLabelsAllowed && labelMode !== "-", xLabelPolicy.hiddenReason, Boolean(xLabelPolicy.hideOverlap));
  }

  function lineBarDataZoomOptions() {
    return [{ type: "inside" }, { type: "slider", height: 18, bottom: 18 }];
  }

  function scheduleDateXAxisLabelRefresh() {
    if (dateXAxisRefreshFrame !== null) return;
    dateXAxisRefreshFrame = requestAnimationFrame(() => {
      dateXAxisRefreshFrame = null;
      refreshDateXAxisLabelsForCurrentZoom();
    });
  }

  function refreshDateXAxisLabelsForCurrentZoom() {
    if (!dateXAxisContext) return;
    const range = currentDateXAxisVisibleRange(dateXAxisContext.labels.length);
    const policy = getDateXAxisLabelPolicy(
      dateXAxisContext.labels,
      dateXAxisContext.rawXValues,
      dateXAxisContext.dateBucket,
      chart.getWidth?.() || el("chart").clientWidth,
      range,
    );
    const currentZoomEnabled = Array.isArray(chart.getOption?.()?.dataZoom) && chart.getOption().dataZoom.length > 0;
    const option = {
      grid: { bottom: policy.bottom },
      xAxis: {
        nameGap: policy.nameGap,
        axisLabel: {
          show: policy.show,
          interval: policy.interval,
          formatter: policy.formatter,
          showMinLabel: policy.showMinLabel,
          showMaxLabel: policy.showMaxLabel,
          rotate: policy.rotate,
          fontSize: policy.fontSize,
        },
      },
    };
    if (currentZoomEnabled !== policy.dataZoomEnabled) {
      option.dataZoom = policy.dataZoomEnabled ? lineBarDataZoomOptions() : [];
    }
    chart.setOption(option);
  }

  function currentDateXAxisVisibleRange(count) {
    const zooms = Array.isArray(chart.getOption?.()?.dataZoom) ? chart.getOption().dataZoom : [];
    const zoom = zooms.find((item) => item && (Number.isFinite(Number(item.start)) || Number.isFinite(Number(item.end)) || item.startValue !== undefined || item.endValue !== undefined));
    if (!zoom) return normaliseDateXAxisVisibleRange(null, count);
    const lastIndex = Math.max(0, count - 1);
    if (zoom.startValue !== undefined || zoom.endValue !== undefined) {
      return normaliseDateXAxisVisibleRange({
        startIndex: dateXAxisZoomValueIndex(zoom.startValue, count, 0),
        endIndex: dateXAxisZoomValueIndex(zoom.endValue, count, lastIndex),
      }, count);
    }
    return normaliseDateXAxisVisibleRange({
      startIndex: Math.floor((Number(zoom.start ?? 0) / 100) * lastIndex),
      endIndex: Math.ceil((Number(zoom.end ?? 100) / 100) * lastIndex),
    }, count);
  }

  function dateXAxisZoomValueIndex(value, count, fallback) {
    if (value === undefined || value === null || value === "") return fallback;
    const numeric = Number(value);
    if (Number.isFinite(numeric)) return Math.round(numeric);
    const labelIndex = dateXAxisContext?.labels?.indexOf(String(value)) ?? -1;
    return labelIndex >= 0 ? labelIndex : fallback;
  }

  function partialDependenceOverlay(data, key) {
    const partial = data.partial_dependence || {};
    if (partial.overlays && partial.overlays[key]) return partial.overlays[key];
    return partial.mode === key ? partial : {};
  }

  function indexedPartialDependenceRows(data, rows) {
    if (!rows.length) return [];
    const labelIndex = new Map((data.rows || []).map((row, index) => [String(row.x), index]));
    return rows
      .map((row) => ({ ...row, index: labelIndex.get(String(row.x)) }))
      .filter((row) => Number.isInteger(row.index));
  }

  function shapPartialDependenceSeries(data) {
    const partial = partialDependenceOverlay(data, "shap");
    const rows = Array.isArray(partial?.rows) ? partial.rows : [];
    const indexedRows = indexedPartialDependenceRows(data, rows);
    if (!indexedRows.length) return [];
    const series = [];
    SHAP_RIBBON_SERIES.forEach(([lowKey, highKey, label, color]) => {
      const ribbon = shapRibbonSeries(indexedRows, lowKey, highKey, label, color);
      if (ribbon) series.push(ribbon);
    });
    series.push({
      name: "SHAP median",
      type: "line",
      yAxisIndex: 0,
      z: 2.8,
      animation: false,
      animationDuration: 0,
      animationDurationUpdate: 0,
      smooth: false,
      showSymbol: (data.rows || []).length < 250,
      symbolSize: 4,
      lineStyle: { color: SHAP_LINE_COLOR, width: 1.8, type: "dashed" },
      itemStyle: { color: SHAP_LINE_COLOR },
      data: (data.rows || []).map((row) => {
        const match = rows.find((partialRow) => String(partialRow.x) === String(row.x));
        const value = Number(match?.p50);
        return Number.isFinite(value) ? value : null;
      }),
      label: { show: false },
    });
    return series;
  }

  function glmPartialDependenceSeries(data) {
    const partial = partialDependenceOverlay(data, "glm");
    const rows = Array.isArray(partial?.rows) ? partial.rows : [];
    const indexedRows = indexedPartialDependenceRows(data, rows);
    if (!indexedRows.length) return [];
    return [
      {
        name: "GLM",
        type: "line",
        yAxisIndex: 0,
        z: 2.9,
        animation: false,
        animationDuration: 0,
        animationDurationUpdate: 0,
        smooth: false,
        showSymbol: (data.rows || []).length < 250,
        symbolSize: 4,
        lineStyle: { color: GLM_LINE_COLOR, width: 2, type: "dashed" },
        itemStyle: { color: GLM_LINE_COLOR },
        data: (data.rows || []).map((row) => {
          const match = rows.find((partialRow) => String(partialRow.x) === String(row.x));
          const value = Number(match?.p50);
          return Number.isFinite(value) ? value : null;
        }),
        label: { show: false },
      },
    ];
  }

  function shapRibbonSeries(rows, lowKey, highKey, label, color) {
    const segments = shapRibbonSegments(rows, lowKey, highKey);
    if (!segments.length) return null;
    return {
      name: label,
      type: "custom",
      coordinateSystem: "cartesian2d",
      yAxisIndex: 0,
      data: segments.map((_, index) => index),
      itemStyle: { color },
      silent: true,
      z: 2,
      animation: false,
      animationDuration: 0,
      animationDurationUpdate: 0,
      renderItem: (params, api) => {
        const segment = segments[params.dataIndex] || [];
        const upper = segment.map((row) => api.coord([row.index, row.high]));
        const lower = [...segment].reverse().map((row) => api.coord([row.index, row.low]));
        return {
          type: "polygon",
          shape: { points: [...upper, ...lower] },
          style: { fill: color, stroke: "none" },
        };
      },
    };
  }

  function shapRibbonSegments(rows, lowKey, highKey) {
    const points = [];
    rows.forEach((row) => {
      const index = Number(row.index);
      const low = Number(row[lowKey]);
      const high = Number(row[highKey]);
      if (!Number.isInteger(index) || !Number.isFinite(low) || !Number.isFinite(high)) return;
      points.push({ index, low, high });
    });
    points.sort((a, b) => a.index - b.index);
    const segments = [];
    let current = [];
    points.forEach((point) => {
      const previous = current[current.length - 1];
      if (previous && point.index !== previous.index + 1) {
        if (current.length > 1) segments.push(current);
        current = [];
      }
      current.push(point);
    });
    if (current.length > 1) segments.push(current);
    return segments;
  }

  function lineBarLegendOptions(legendData, mainLegendSelection, overlayLegendData, overlayLegendSelection) {
    const textStyle = { color: getCss("--text"), fontWeight: 700, fontSize: 13 };
    const overlayTextStyle = { color: getCss("--text"), fontWeight: 400, fontSize: 11 };
    const mainLegend = {
      top: LINE_BAR_MAIN_LEGEND_TOP,
      data: legendData,
      selected: mainLegendSelection,
      textStyle,
    };
    if (!overlayLegendData.length) return mainLegend;
    return [
      mainLegend,
      {
        top: LINE_BAR_OVERLAY_LEGEND_TOP,
        left: "center",
        type: "scroll",
        data: overlayLegendData,
        selected: overlayLegendSelection,
        textStyle: overlayTextStyle,
        pageIconColor: getCss("--text"),
        pageIconInactiveColor: getCss("--muted"),
        pageTextStyle: { color: getCss("--muted") },
      },
    ];
  }

  function matchingLegendSelection(option, entries) {
    const names = entries.map(legendEntryName).filter(Boolean);
    const defaults = Object.fromEntries(names.map((entry) => [entry, true]));
    if (!entries.length) return defaults;
    const legends = Array.isArray(option?.legend) ? option.legend : (option?.legend ? [option.legend] : []);
    const previous = Object.assign({}, ...legends.map((legend) => legend?.selected || {}));
    names.forEach((entry) => {
      if (Object.prototype.hasOwnProperty.call(previous, entry)) {
        defaults[entry] = previous[entry] !== false;
      }
    });
    return defaults;
  }

  function legendEntryName(entry) {
    if (typeof entry === "string") return entry;
    if (entry && typeof entry === "object") return String(entry.name || "");
    return "";
  }

  function chartDensityMessage(groupCount, xLabelsHidden, chartLabelsHidden, xLabelReason = "", xLabelsSuppressed = false) {
    if (xLabelsSuppressed && !chartLabelsHidden) return xLabelReason ? `Some X-axis labels hidden ${xLabelReason}.` : "Some X-axis labels hidden to avoid overlap.";
    if (!xLabelsHidden && !chartLabelsHidden) return "";
    if (xLabelsHidden && xLabelReason && !chartLabelsHidden) return `X-axis labels hidden ${xLabelReason}.`;
    const labelTarget = xLabelsHidden && chartLabelsHidden
      ? "X-axis and chart labels"
      : xLabelsHidden ? "X-axis labels" : "Chart labels";
    return `${labelTarget} hidden as >${LABEL_DENSITY_LIMIT.toLocaleString()} categories.`;
  }

  function finiteNumberOrNull(value) {
    if (value === null || value === undefined || value === "") return null;
    const number = Number(value);
    return Number.isFinite(number) ? number : null;
  }

  function rawResponseValue(row, index) {
    const numerator = finiteNumberOrNull(row?.[`resp${index}_num`]);
    const denominator = finiteNumberOrNull(row?.[`resp${index}_den`]);
    if (numerator === null || denominator === null || denominator === 0) return null;
    return numerator / denominator;
  }

  function compareLineBarLabels(left, right) {
    const leftLabel = String(left?.x ?? "").toLowerCase();
    const rightLabel = String(right?.x ?? "").toLowerCase();
    if (leftLabel < rightLabel) return -1;
    if (leftLabel > rightLabel) return 1;
    return 0;
  }

  function compareNullableDescending(left, right) {
    if (left === null && right === null) return 0;
    if (left === null) return 1;
    if (right === null) return -1;
    return right - left;
  }

  function shapMedianMap(data) {
    const partial = partialDependenceOverlay(data, "shap");
    const rows = Array.isArray(partial?.rows) ? partial.rows : [];
    return new Map(rows.map((row) => [String(row.x), finiteNumberOrNull(row.p50)]));
  }

  function compareLineBarRowsForSort(sort, shapMedians) {
    return (left, right) => {
      if (sort === "volume") {
        const tailCompare = Number(!left?.is_tail) - Number(!right?.is_tail);
        if (tailCompare) return tailCompare;
        const volumeCompare = (finiteNumberOrNull(right?.volume) || 0) - (finiteNumberOrNull(left?.volume) || 0);
        if (volumeCompare) return volumeCompare;
        return compareLineBarLabels(left, right);
      }
      if (sort === "actual" || sort === "response") {
        const responseCompare = compareNullableDescending(rawResponseValue(left, 0), rawResponseValue(right, 0));
        if (responseCompare) return responseCompare;
        return compareLineBarLabels(left, right);
      }
      if (sort === "expected") {
        const responseCompare = compareNullableDescending(rawResponseValue(left, 1), rawResponseValue(right, 1));
        if (responseCompare) return responseCompare;
        return compareLineBarLabels(left, right);
      }
      if (sort === "shap") {
        const tailCompare = Number(Boolean(left?.is_tail)) - Number(Boolean(right?.is_tail));
        if (tailCompare) return tailCompare;
        const medianCompare = compareNullableDescending(shapMedians.get(String(left?.x)), shapMedians.get(String(right?.x)));
        if (medianCompare) return medianCompare;
        return compareLineBarLabels(left, right);
      }
      return compareLineBarLabels(left, right);
    };
  }

  function orderPartialDependenceRowsForChart(data) {
    const rowOrder = new Map((data.rows || []).map((row, index) => [String(row.x), index]));
    function orderRows(partial) {
      if (!partial || typeof partial !== "object") return;
      if (partial.overlays && typeof partial.overlays === "object") {
        Object.values(partial.overlays).forEach(orderRows);
      }
      if (!Array.isArray(partial.rows)) return;
      partial.rows.sort((left, right) => {
        const leftIndex = rowOrder.get(String(left?.x));
        const rightIndex = rowOrder.get(String(right?.x));
        if (leftIndex === undefined && rightIndex === undefined) return 0;
        if (leftIndex === undefined) return 1;
        if (rightIndex === undefined) return -1;
        return leftIndex - rightIndex;
      });
    }
    orderRows(data.partial_dependence);
  }

  function applyClientLineBarSort(options = {}) {
    const cache = toolCache("line_bar");
    const data = state.lastData || cache.data;
    if (!data || data.x_group_kind !== "categorical") return false;
    if (!data.groups_truncated) {
      data.rows = [...(data.rows || [])].sort(compareLineBarRowsForSort(state.sort, shapMedianMap(data)));
      orderPartialDependenceRowsForChart(data);
    }
    const request = buildChartRequest();
    if (request) {
      cache.requestKey = stableRequestKey(request);
      cache.data = data;
    }
    state.lastData = data;
    if (options.render !== false) {
      measureToolRender("line_bar", () => renderChartData(data, { resetTablePage: true }));
      lineBarChartDirty = false;
    } else {
      lineBarChartDirty = true;
    }
    return true;
  }

  function shapMediansForCompleteTableSort(data) {
    if (state.sort !== "shap") return new Map();
    const chartData = state.lastData || toolCache("line_bar").data;
    if (!chartData || chartData.groups_truncated) return null;
    const medians = shapMedianMap(chartData);
    for (const row of data.rows || []) {
      const median = medians.get(String(row?.x));
      if (median === null || median === undefined) return null;
    }
    return medians;
  }

  function canSortLineBarTableClientSide(data, shapMedians) {
    const table = data?.table || {};
    const rows = Array.isArray(data?.rows) ? data.rows : [];
    const pageCount = Number(table.page_count);
    const matchCount = Number(table.match_count);
    return data?.x_group_kind === "categorical"
      && Number.isFinite(pageCount)
      && pageCount === 1
      && Number.isFinite(matchCount)
      && rows.length === matchCount
      && normaliseLineBarTableSearch(table.search) === normaliseLineBarTableSearch(state.lineBarTableSearch)
      && (state.sort !== "shap" || shapMedians !== null);
  }

  function applyClientLineBarTableSort() {
    const sourceData = completeTableCacheData;
    const sourceShapMedians = shapMediansForCompleteTableSort(sourceData || {});
    if (canUseCompleteLineBarTableSource(sourceData) && (state.sort !== "shap" || sourceShapMedians !== null)) {
      sourceData.rows = [...(sourceData.rows || [])].sort(compareLineBarRowsForSort(state.sort, sourceShapMedians || new Map()));
      completeTableCacheKey = stableRequestKey(buildCompleteTableSourceRequest());
      completeTableCacheData = sourceData;
      return applyClientLineBarTableFilter();
    }
    const data = tableCacheData;
    const shapMedians = shapMediansForCompleteTableSort(data || {});
    if (!canSortLineBarTableClientSide(data, shapMedians)) return false;
    data.rows = [...(data.rows || [])].sort(compareLineBarRowsForSort(state.sort, shapMedians || new Map()));
    const request = buildTableRequest();
    if (request) tableCacheKey = stableRequestKey(request);
    tableCacheData = data;
    rememberCompleteLineBarTableSource(data);
    measureToolRender("line_bar", () => renderLineBarTableContents(data));
    return true;
  }

  function normaliseLineBarTableSearch(value = state.lineBarTableSearch) {
    return String(value || "").trim();
  }

  function cloneLineBarTableData(data, rows = data?.rows || []) {
    return {
      ...(data || {}),
      rows: rows.map((row) => ({ ...row })),
      responses: (data?.responses || []).map((response) => ({ ...response })),
      denominator: { ...(data?.denominator || {}) },
      summary: {
        ...(data?.summary || {}),
        responses: Array.isArray(data?.summary?.responses) ? [...data.summary.responses] : [],
      },
      table: { ...(data?.table || {}) },
      transform: {
        ...(data?.transform || {}),
        values: Array.isArray(data?.transform?.values) ? [...data.transform.values] : data?.transform?.values,
      },
      warnings: Array.isArray(data?.warnings) ? [...data.warnings] : data?.warnings,
      exclusion_warnings: Array.isArray(data?.exclusion_warnings) ? [...data.exclusion_warnings] : data?.exclusion_warnings,
    };
  }

  function buildCompleteTableSourceRequest() {
    const request = buildChartRequest();
    if (!request) return null;
    return {
      ...request,
      tableSearch: "",
      tablePage: 1,
      tablePageSize: TABLE_PAGE_SIZE,
    };
  }

  function lineBarTableComplete(data) {
    const table = data?.table || {};
    const rows = Array.isArray(data?.rows) ? data.rows : [];
    const pageCount = Number(table.page_count);
    const matchCount = Number(table.match_count);
    const page = Number(table.page);
    return Number.isFinite(pageCount)
      && pageCount === 1
      && Number.isFinite(matchCount)
      && rows.length === matchCount
      && (!Number.isFinite(page) || page === 1);
  }

  function rememberCompleteLineBarTableSource(data, request = null) {
    const table = data?.table || {};
    if (!lineBarTableComplete(data)) return;
    if (normaliseLineBarTableSearch(table.search) !== "") return;
    const sourceRequest = request
      ? { ...request, tableSearch: "", tablePage: 1, tablePageSize: TABLE_PAGE_SIZE }
      : buildCompleteTableSourceRequest();
    if (!sourceRequest) return;
    completeTableCacheKey = stableRequestKey(sourceRequest);
    completeTableCacheData = cloneLineBarTableData(data);
  }

  function canUseCompleteLineBarTableSource(data = completeTableCacheData) {
    const sourceRequest = buildCompleteTableSourceRequest();
    if (!sourceRequest || !data || !completeTableCacheKey || !lineBarTableComplete(data)) return false;
    return completeTableCacheKey === stableRequestKey(sourceRequest)
      && normaliseLineBarTableSearch(data.table?.search) === "";
  }

  function formatLineBarTableNumericSearchLabel(value) {
    const number = finiteNumberOrNull(value);
    if (number === null) return null;
    if (Number.isInteger(number)) {
      return number.toLocaleString("en-US", { maximumFractionDigits: 0 });
    }
    let formatted = number.toLocaleString("en-US", {
      minimumFractionDigits: 12,
      maximumFractionDigits: 12,
    }).replace(/(\.\d*?)0+$/, "$1");
    if (formatted.endsWith(".")) formatted = formatted.slice(0, -1);
    return formatted === "-0" ? "0" : formatted;
  }

  function lineBarTableRowMatchesSearch(row, search, xKind) {
    const needle = normaliseLineBarTableSearch(search).toLowerCase();
    if (!needle) return true;
    const rawCandidates = [row?.x, row?.x_sort];
    const candidates = [...rawCandidates];
    if (xKind === "integer" || xKind === "numeric") {
      rawCandidates.forEach((value) => candidates.push(formatLineBarTableNumericSearchLabel(value)));
    }
    return candidates.some((value) => value !== null && value !== undefined && String(value).toLowerCase().includes(needle));
  }

  function transformLineBarTableSummaryValue(value, data, responseIndex) {
    const number = finiteNumberOrNull(value);
    if (number === null) return null;
    const transform = String(data?.transform?.mode || state.transform || "none");
    const references = Array.isArray(data?.transform?.values) ? data.transform.values : [];
    const reference = finiteNumberOrNull(references[responseIndex]);
    if (transform === "log") return number > 0 ? Math.log(number) : null;
    if (transform === "exp") {
      const transformed = Math.exp(number);
      return Number.isFinite(transformed) ? transformed : null;
    }
    if (transform === "logit") return number > 0 && number < 1 ? Math.log(number / (1 - number)) : null;
    if (transform === "zero") return reference !== null ? number - reference : null;
    if (transform === "one") return reference !== null && reference !== 0 ? number / reference : null;
    return number;
  }

  function buildClientLineBarTableSummary(rows, data) {
    const responseCount = Array.isArray(data?.responses) ? data.responses.length : 0;
    const summary = {
      volume: rows.reduce((total, row) => total + (finiteNumberOrNull(row?.volume) || 0), 0),
      row_count: rows.reduce((total, row) => {
        const rowCount = finiteNumberOrNull(row?.row_count);
        return total + (rowCount === null ? (finiteNumberOrNull(row?.volume) || 0) : rowCount);
      }, 0),
      responses: [],
    };
    for (let responseIndex = 0; responseIndex < responseCount; responseIndex += 1) {
      const numerator = rows.reduce((total, row) => total + (finiteNumberOrNull(row?.[`resp${responseIndex}_num`]) || 0), 0);
      const denominator = rows.reduce((total, row) => total + (finiteNumberOrNull(row?.[`resp${responseIndex}_den`]) || 0), 0);
      const average = denominator ? numerator / denominator : null;
      summary.responses.push(transformLineBarTableSummaryValue(average, data, responseIndex));
    }
    return summary;
  }

  function filteredLineBarTableDataFromSource(sourceData, search) {
    const tableSearch = normaliseLineBarTableSearch(search);
    const rows = (sourceData.rows || []).filter((row) => lineBarTableRowMatchesSearch(row, tableSearch, sourceData.x_kind));
    const data = cloneLineBarTableData(sourceData, rows);
    data.summary = buildClientLineBarTableSummary(rows, sourceData);
    data.table = {
      ...(sourceData.table || {}),
      search: tableSearch,
      page: 1,
      page_count: 1,
      match_count: rows.length,
    };
    return data;
  }

  function applyClientLineBarTableFilter(options = {}) {
    if (!canUseCompleteLineBarTableSource()) return false;
    const request = buildTableRequest();
    if (!request) return false;
    const data = filteredLineBarTableDataFromSource(completeTableCacheData, state.lineBarTableSearch);
    tableCacheKey = options.requestKey || stableRequestKey(request);
    tableCacheData = data;
    state.tablePage = 1;
    measureToolRender("line_bar", () => renderLineBarTableContents(data));
    return true;
  }

  function selectedFeatureBase() {
    return String(state.schema?.feature_bases?.[state.x] || "").trim();
  }

  function shapOverlayAvailableForSelectedColumn() {
    const feature = String(state.x || "");
    if (!feature) return false;
    const source = (state.schema?.data_sources || []).find((item) => item.kind === "gbm_shap_long" && item.active);
    if (!source) return false;
    return (source.columns || []).some((column) => (
      column?.source_role === "gbm_shap_value"
      && String(column.artifact_column || column.label || "") === feature
    ));
  }

  function formatChartXLabel(row, data) {
    const rangeLabel = formatQuantileRangeLabel(row, data, "\n");
    return rangeLabel || formatXLabel(row?.x, data.x_kind);
  }

  function formatTableXLabel(row, data) {
    const rangeLabel = formatQuantileRangeLabel(row, data, ": ");
    return rangeLabel || formatXLabel(row?.x, data.x_kind);
  }

  function formatQuantileRangeLabel(row, data, separator) {
    if ((data?.x_group_kind || data?.x_kind) !== "quantile") return "";
    if (!row || row.is_tail || row.x === "Missing") return "";
    const start = formatQuantileEndpoint(row.x_start);
    const end = formatQuantileEndpoint(row.x_end);
    if (!start || !end) return "";
    const prefix = String(row.x ?? "");
    return start === end
      ? `${prefix}${separator}${start}`
      : `${prefix}${separator}${start} to ${end}`;
  }

  function formatQuantileEndpoint(value) {
    if (value === null || value === undefined) return "";
    return formatNumber(value);
  }

  function formatChartTooltip(params, weightLabel, responseValueFormatter = formatResponseValue) {
    const items = Array.isArray(params) ? params : [params];
    if (!items.length) return "";
    const lines = [escapeHtml(items[0].axisValueLabel ?? items[0].name ?? "")];
    items.forEach((item) => {
      const value = Array.isArray(item.value) ? item.value[1] : item.value;
      const formatter = item.seriesName === weightLabel ? formatNumber : responseValueFormatter;
      lines.push(`${item.marker || ""}${escapeHtml(item.seriesName)}: ${escapeHtml(formatter(value))}`);
    });
    return lines.join("<br/>");
  }

  function updateMetricTitles(data) {
    const summaries = data.response_summaries || [];
    renderMetricTitle(el("expectedMetricTitle"), "Expected", summaries[1]?.value);
    el("expectedMetricTitle").querySelector(".metric-value")?.classList.add("metric-value--first-expected");
    if (summaries[2]?.value === null || summaries[2]?.value === undefined) return;
    const valueSpan = document.createElement("span");
    valueSpan.className = "metric-value metric-value--second-expected";
    valueSpan.textContent = formatResponseValue(summaries[2].value);
    el("expectedMetricTitle").append(valueSpan);
  }

  function formatResponseLabel(params, responseValueFormatter = formatResponseValue) {
    const value = Array.isArray(params.value) ? params.value[1] : params.value;
    return responseValueFormatter(value);
  }

  function lineBarChartLabelStyle() {
    if (!document.body.classList.contains("dark")) return {};
    return {
      color: "#ffffff",
      textBorderWidth: 0,
      textShadowBlur: 0,
    };
  }

  function formatResponseValue(value) {
    return isUpliftTransform() ? formatUpliftPercent(value) : formatLineValue(value);
  }

  function chartResponseFormatter(transform = state.transform, kpiFormat = state.activeKpiFormat) {
    const renderTransform = String(transform || "none");
    const renderKpiFormat = kpiFormat
      ? {
          decimals: Number(kpiFormat.decimals),
          format: String(kpiFormat.format || "number").toLowerCase(),
        }
      : null;
    return (value) => (
      isUpliftTransform(renderTransform)
        ? formatUpliftPercent(value)
        : formatLineValueForFormat(value, renderKpiFormat)
    );
  }

  function isUpliftTransform(transform = state.transform) {
    return String(transform || "none") === "one";
  }

  function isBaseReferenceTransform(transform = state.transform) {
    return ["zero", "one"].includes(String(transform || "none"));
  }

  function isBaseWeightBar(data, row, transform = state.transform) {
    if (!isBaseReferenceTransform(transform)) return false;
    if (String(data?.transform?.reference || "") !== "base") return false;
    const baseX = data?.transform?.base_x;
    if (baseX === null || baseX === undefined) return false;
    return String(row?.x) === String(baseX);
  }

  function weightBarColor(data, row) {
    if (row?.is_tail) return getCss("--tail");
    return isBaseWeightBar(data, row) ? getCss("--base-bar") : getCss("--bar");
  }

  function upliftBaselineSeries(data, transform = state.transform) {
    if (!isUpliftTransform(transform)) return null;
    return {
      name: "0% uplift baseline",
      type: "line",
      yAxisIndex: 0,
      z: 2.7,
      silent: true,
      legendHoverLink: false,
      animation: false,
      animationDuration: 0,
      animationDurationUpdate: 0,
      showSymbol: false,
      symbolSize: 0,
      lineStyle: { opacity: 0 },
      itemStyle: { opacity: 0 },
      tooltip: { show: false },
      data: (data.rows || []).map(() => 1),
      markLine: {
        silent: true,
        symbol: "none",
        label: { show: false },
        lineStyle: { color: getCss("--text"), width: 2, type: "solid", opacity: 0.5 },
        data: [{ yAxis: 1 }],
      },
    };
  }

  function formatUpliftPercent(value) {
    if (value === null || value === undefined || Number.isNaN(value)) return "";
    const number = Number(value);
    if (!Number.isFinite(number)) return "";
    let percent = Number(((number - 1) * 100).toFixed(10));
    if (Math.abs(percent) < 1e-9) percent = 0;
    const abs = Math.abs(percent);
    let fractionDigits = 0;
    if (abs !== 0 && abs < 0.01) fractionDigits = 4;
    else if (abs !== 0 && abs < 1) fractionDigits = 2;
    else if (abs !== 0 && abs < 10) fractionDigits = 1;
    const formatted = percent.toLocaleString(undefined, {
      minimumFractionDigits: fractionDigits,
      maximumFractionDigits: fractionDigits,
    });
    const sign = percent > 0 ? "+" : "";
    return `${sign}${formatted}%`;
  }

  function responseAxisOptions(data, selected = null, transform = state.transform) {
    const extent = withUpliftBaselineExtent(
      responseAxisExtent(data.rows, data.responses, data.partial_dependence, selected)
        || responseAxisExtent(data.rows, data.responses, data.partial_dependence),
      transform,
    );
    return responseAxisBounds(extent) || {};
  }

  function withUpliftBaselineExtent(extent, transform = state.transform) {
    if (!extent || !isUpliftTransform(transform)) return extent;
    return {
      min: Math.min(Number(extent.min), 1),
      max: Math.max(Number(extent.max), 1),
    };
  }

  function partialDependenceOverlayEntries(partialDependence) {
    if (!partialDependence) return [];
    if (partialDependence.overlays) {
      return Object.entries(partialDependence.overlays).filter((entry) => entry[1] && typeof entry[1] === "object");
    }
    return [[partialDependence.mode || "", partialDependence]];
  }

  function responseAxisExtent(rows, responses, partialDependence = null, selected = null) {
    let min = Infinity;
    let max = -Infinity;
    const responseList = Array.isArray(responses)
      ? responses
      : Array.from({ length: Number(responses) || 0 }, (_, index) => ({ label: `resp${index}` }));
    const selectedVisible = (name) => !selected || selected[String(name)] !== false;
    const addValue = (rawValue) => {
      const value = Number(rawValue);
      if (!Number.isFinite(value)) return;
      min = Math.min(min, value);
      max = Math.max(max, value);
    };
    rows.forEach((row) => {
      responseList.forEach((response, index) => {
        if (!selectedVisible(response?.label)) return;
        addValue(row[`resp${index}`]);
      });
    });
    partialDependenceOverlayEntries(partialDependence).forEach(([key, overlay]) => {
      const overlayKey = String(key || overlay?.mode || "");
      (overlay?.rows || []).forEach((row) => {
        if (overlayKey === "shap") {
          SHAP_RIBBON_SERIES.forEach(([lowKey, highKey, label]) => {
            if (!selectedVisible(label)) return;
            [row?.[lowKey], row?.[highKey]].forEach(addValue);
          });
          if (selectedVisible("SHAP median")) addValue(row?.p50);
          return;
        }
        if (overlayKey === "glm") {
          if (selectedVisible("GLM")) addValue(row?.p50);
          return;
        }
        SHAP_RIBBON_SERIES.forEach(([lowKey, highKey, label]) => {
          if (!selectedVisible(label)) return;
          [row?.[lowKey], row?.[highKey]].forEach(addValue);
        });
        if (selectedVisible("SHAP median") || selectedVisible("GLM")) addValue(row?.p50);
      });
    });
    return Number.isFinite(min) && Number.isFinite(max) ? { min, max } : null;
  }

  function legendSelectionFromOption(option) {
    const legends = Array.isArray(option?.legend) ? option.legend : (option?.legend ? [option.legend] : []);
    return Object.assign({}, ...legends.map((legend) => legend?.selected || {}));
  }

  function updateResponseAxisForLegendSelection() {
    if (!state.lastData) return;
    const responseAxis = responseAxisOptions(state.lastData, legendSelectionFromOption(chart.getOption()), chartRenderTransform);
    chart.setOption({
      yAxis: [{
        min: responseAxis.min,
        max: responseAxis.max,
        interval: responseAxis.interval,
      }],
    });
  }

  function responseAxisSpan(value) {
    const min = Number(value?.min);
    const max = Number(value?.max);
    if (!Number.isFinite(min) || !Number.isFinite(max)) return null;
    if (max > min) return max - min;
    return Math.max(Math.abs(max), Math.abs(min), 1);
  }

  function niceAxisStep(span) {
    if (!Number.isFinite(span) || span <= 0) return 1;
    const roughStep = span / RESPONSE_AXIS_TARGET_INTERVALS;
    const magnitude = 10 ** Math.floor(Math.log10(roughStep));
    const normalized = roughStep / magnitude;
    const multiplier = [1, 2, 5, 10].find((candidate) => normalized <= candidate) || 10;
    return multiplier * magnitude;
  }

  function roundAxisValue(value, step) {
    if (!Number.isFinite(value)) return value;
    const precision = Math.min(12, Math.max(0, Math.ceil(-Math.log10(Math.abs(step))) + 3));
    return Number(value.toFixed(precision));
  }

  function responseAxisBounds(value) {
    const min = Number(value?.min);
    const max = Number(value?.max);
    const span = responseAxisSpan(value);
    if (!Number.isFinite(min) || !Number.isFinite(max) || span === null) return null;
    const paddedMin = min - span * RESPONSE_AXIS_PADDING;
    const paddedMax = max + span * RESPONSE_AXIS_PADDING;
    const step = niceAxisStep(paddedMax - paddedMin);
    let axisMin = Math.floor(paddedMin / step) * step;
    let axisMax = Math.ceil(paddedMax / step) * step;
    if (min >= 0) axisMin = Math.max(0, axisMin);
    if (axisMax <= axisMin) axisMax = axisMin + step;
    return {
      min: roundAxisValue(axisMin, step),
      max: roundAxisValue(axisMax, step),
      interval: step,
    };
  }

  function getXAxisLabelPolicy(labels, kind = "", rawValues = labels, dateBucket = "none", chartWidth = 0) {
    if (isDateKind(kind)) return getDateXAxisLabelPolicy(labels, rawValues, dateBucket, chartWidth);
    if (kind === "quantile") return getQuantileXAxisLabelPolicy(labels, chartWidth);
    const maxLength = labels.reduce((longest, label) => Math.max(longest, String(label).length), 0);
    const tooMany = labels.length >= LABEL_DENSITY_LIMIT;
    const dataZoomSpace = labels.length > 120 ? 36 : 0;
    if (tooMany) {
      return {
        show: false,
        interval: 0,
        formatter: undefined,
        showMinLabel: true,
        showMaxLabel: true,
        rotate: 0,
        fontSize: 10,
        nameGap: 22,
        bottom: 38 + dataZoomSpace,
        dataZoomEnabled: labels.length > 120,
        hideOverlap: false,
        hiddenReason: `as >${LABEL_DENSITY_LIMIT.toLocaleString()} categories`,
      };
    }
    const fontSize = labels.length > 50 ? 8 : 10;
    const estimatedTextWidth = maxLength * fontSize * 0.5;
    const plotWidth = dateXAxisPlotWidth(chartWidth);
    const slotWidth = plotWidth / Math.max(1, labels.length);
    const horizontalFootprint = estimatedTextWidth + CATEGORICAL_AXIS_LABEL_PADDING;
    const rotate = labels.length > 30 || maxLength > 10 || horizontalFootprint > slotWidth ? 65 : 0;
    const rotatedHeight = estimatedTextWidth * Math.sin((rotate * Math.PI) / 180) + fontSize * Math.cos((rotate * Math.PI) / 180);
    const labelSpace = rotate ? Math.min(140, Math.max(58, Math.ceil(rotatedHeight) + 18)) : 38;
    const titleGap = rotate ? Math.max(26, labelSpace - 10) : 26;
    return {
      show: true,
      interval: 0,
      formatter: undefined,
      showMinLabel: true,
      showMaxLabel: true,
      rotate,
      fontSize,
      nameGap: titleGap,
      bottom: titleGap + 16 + dataZoomSpace,
      dataZoomEnabled: labels.length > 120,
      hideOverlap: false,
      hiddenReason: "",
    };
  }

  function getQuantileXAxisLabelPolicy(labels, chartWidth = 0) {
    const count = labels.length;
    const tooMany = count >= LABEL_DENSITY_LIMIT;
    const dataZoomSpace = count > 120 ? 36 : 0;
    if (tooMany) {
      return {
        show: false,
        interval: 0,
        formatter: undefined,
        showMinLabel: true,
        showMaxLabel: true,
        rotate: 0,
        fontSize: 10,
        nameGap: 22,
        bottom: 38 + dataZoomSpace,
        dataZoomEnabled: count > 120,
        hideOverlap: false,
        hiddenReason: `as >${LABEL_DENSITY_LIMIT.toLocaleString()} categories`,
      };
    }
    const visibleCount = Math.max(1, count);
    const slotWidth = dateXAxisPlotWidth(chartWidth) / visibleCount;
    let fallback = null;
    for (const rotate of QUANTILE_AXIS_ROTATIONS) {
      for (const fontSize of QUANTILE_AXIS_FONT_SIZES) {
        const size = quantileAxisLabelMaxSize(labels, fontSize);
        const footprint = quantileAxisLabelFootprint(size.width, size.height, rotate);
        const labelSpace = quantileAxisLabelSpace(size.width, size.height, rotate);
        const policy = {
          show: true,
          interval: 0,
          formatter: undefined,
          showMinLabel: true,
          showMaxLabel: true,
          rotate,
          fontSize,
          nameGap: Math.max(26, labelSpace - 10),
          bottom: Math.max(26, labelSpace - 10) + 16 + dataZoomSpace,
          dataZoomEnabled: count > 120,
          hideOverlap: false,
          hiddenReason: "",
        };
        fallback = policy;
        if (visibleCount <= 1 || footprint <= slotWidth) return policy;
      }
    }
    return {
      ...(fallback || {
        show: true,
        interval: 0,
        formatter: undefined,
        showMinLabel: true,
        showMaxLabel: true,
        rotate: 75,
        fontSize: 7,
        nameGap: 82,
        bottom: 98 + dataZoomSpace,
        dataZoomEnabled: count > 120,
      }),
      hideOverlap: true,
      hiddenReason: "because quantile labels would overlap",
    };
  }

  function quantileAxisLabelMaxSize(labels, fontSize) {
    return labels.reduce((maxSize, label) => {
      const lines = String(label || "").split("\n");
      const width = lines.reduce((lineWidth, line) => Math.max(lineWidth, line.length * fontSize * QUANTILE_AXIS_LABEL_WIDTH_FACTOR), 0);
      const height = Math.max(fontSize, lines.length * fontSize * 1.15);
      return {
        width: Math.max(maxSize.width, width),
        height: Math.max(maxSize.height, height),
      };
    }, { width: 0, height: fontSize });
  }

  function quantileAxisLabelFootprint(labelWidth, labelHeight, rotate) {
    if (!rotate) return labelWidth + QUANTILE_AXIS_LABEL_PADDING;
    const radians = (rotate * Math.PI) / 180;
    return labelWidth * Math.cos(radians) + labelHeight * Math.sin(radians) + QUANTILE_AXIS_LABEL_PADDING;
  }

  function quantileAxisLabelSpace(labelWidth, labelHeight, rotate) {
    if (!rotate) return Math.max(38, Math.ceil(labelHeight) + 18);
    const radians = (rotate * Math.PI) / 180;
    const rotatedHeight = labelWidth * Math.sin(radians) + labelHeight * Math.cos(radians);
    return Math.min(190, Math.max(70, Math.ceil(rotatedHeight) + 18));
  }

  function getDateXAxisLabelPolicy(labels, rawValues, dateBucket = "none", chartWidth = 0, visibleRange = null) {
    const formattedLabels = rawValues.map((value) => formatDateAxisLabel(value, parseDateCategory(value), dateBucket));
    const fullFit = dateXAxisLabelFit(formattedLabels, chartWidth, normaliseDateXAxisVisibleRange(null, labels.length), dateBucket);
    const visibleFit = dateXAxisLabelFit(formattedLabels, chartWidth, normaliseDateXAxisVisibleRange(visibleRange, labels.length), dateBucket);
    const dataZoomEnabled = !fullFit.show;
    const dataZoomSpace = dataZoomEnabled ? 36 : 0;
    const labelSpace = visibleFit.show ? visibleFit.labelSpace : 38;
    const titleGap = visibleFit.show ? Math.max(26, labelSpace - 10) : 22;
    return {
      show: visibleFit.show,
      interval: 0,
      formatter: (_value, index) => formattedLabels[index] ?? String(_value),
      showMinLabel: visibleFit.show ? true : undefined,
      showMaxLabel: visibleFit.show ? true : undefined,
      rotate: visibleFit.show ? visibleFit.rotate : 0,
      fontSize: visibleFit.fontSize,
      nameGap: titleGap,
      bottom: titleGap + 16 + dataZoomSpace,
      dataZoomEnabled,
      hiddenReason: dataZoomEnabled ? "because date labels would overlap; use zoom to inspect labels" : "",
    };
  }

  function normaliseDateXAxisVisibleRange(range, count) {
    const lastIndex = Math.max(0, count - 1);
    const startIndex = Math.max(0, Math.min(lastIndex, Math.floor(Number(range?.startIndex ?? 0))));
    const endIndex = Math.max(startIndex, Math.min(lastIndex, Math.ceil(Number(range?.endIndex ?? lastIndex))));
    return { startIndex, endIndex };
  }

  function dateXAxisLabelFit(formattedLabels, chartWidth = 0, visibleRange = null, dateBucket = "none") {
    const count = formattedLabels.length;
    if (count <= 0) return { show: false, fontSize: DATE_AXIS_FONT_SIZES[0], rotate: 0 };
    const { startIndex, endIndex } = normaliseDateXAxisVisibleRange(visibleRange, count);
    const visibleCount = Math.max(1, endIndex - startIndex + 1);
    const plotWidth = dateXAxisPlotWidth(chartWidth);
    const slotWidth = plotWidth / visibleCount;
    const rotate = dateXAxisLabelRotation(dateBucket, visibleCount);
    const visibleLabels = formattedLabels.slice(startIndex, endIndex + 1);
    for (const fontSize of DATE_AXIS_FONT_SIZES) {
      const maxWidth = visibleLabels
        .reduce((width, label) => Math.max(width, estimateDateAxisLabelWidth(label, fontSize)), 0);
      const labelWidth = dateXAxisLabelFootprint(maxWidth, fontSize, rotate);
      if (visibleCount <= 1 || labelWidth <= slotWidth) {
        return {
          show: true,
          fontSize,
          rotate,
          labelSpace: dateXAxisLabelSpace(maxWidth, fontSize, rotate),
        };
      }
    }
    if (visibleCount < DATE_AXIS_VISIBLE_LABEL_LIMIT) {
      const fontSize = visibleCount > 40 ? 7 : 8;
      const maxWidth = visibleLabels
        .reduce((width, label) => Math.max(width, estimateDateAxisLabelWidth(label, fontSize)), 0);
      return {
        show: true,
        fontSize,
        rotate,
        labelSpace: dateXAxisLabelSpace(maxWidth, fontSize, rotate),
      };
    }
    return { show: false, fontSize: DATE_AXIS_FONT_SIZES[DATE_AXIS_FONT_SIZES.length - 1], rotate: 0, labelSpace: 38 };
  }

  function dateXAxisLabelRotation(dateBucket, visibleCount) {
    const horizontalLimit = dateBucket === "year" ? DATE_AXIS_YEAR_HORIZONTAL_LABEL_LIMIT : DATE_AXIS_HORIZONTAL_LABEL_LIMIT;
    return visibleCount < horizontalLimit ? 0 : DATE_AXIS_ROTATION;
  }

  function dateXAxisLabelFootprint(labelWidth, fontSize, rotate) {
    if (!rotate) return labelWidth + DATE_AXIS_LABEL_PADDING;
    const radians = (rotate * Math.PI) / 180;
    return labelWidth * Math.cos(radians) + fontSize * Math.sin(radians) + DATE_AXIS_LABEL_PADDING;
  }

  function dateXAxisLabelSpace(labelWidth, fontSize, rotate) {
    if (!rotate) return 38;
    const radians = (rotate * Math.PI) / 180;
    const rotatedHeight = labelWidth * Math.sin(radians) + fontSize * Math.cos(radians);
    return Math.min(190, Math.max(70, Math.ceil(rotatedHeight) + 18));
  }

  function dateXAxisPlotWidth(chartWidth = 0) {
    const width = Number(chartWidth);
    return Math.max(120, (Number.isFinite(width) && width > 0 ? width : 900) - 72 - 76);
  }

  function estimateDateAxisLabelWidth(label, fontSize) {
    return String(label || "").length * fontSize * DATE_AXIS_LABEL_WIDTH_FACTOR;
  }

  function parseDateCategory(value) {
    if (value === null || value === undefined) return null;
    const match = String(value).trim().match(/^(\d{4})-(\d{2})-(\d{2})(?:[ T](\d{2}):(\d{2})(?::(\d{2})(?:\.\d+)?)?)?/);
    if (!match) return null;
    const year = Number(match[1]);
    const month = Number(match[2]);
    const day = Number(match[3]);
    const hour = Number(match[4] || 0);
    const minute = Number(match[5] || 0);
    const second = Number(match[6] || 0);
    if (!Number.isInteger(year) || !Number.isInteger(month) || !Number.isInteger(day)) return null;
    if (![hour, minute, second].every((valuePart) => Number.isInteger(valuePart))) return null;
    const checked = new Date(Date.UTC(year, month - 1, day, hour, minute, second));
    if (
      checked.getUTCFullYear() !== year
      || checked.getUTCMonth() !== month - 1
      || checked.getUTCDate() !== day
      || checked.getUTCHours() !== hour
      || checked.getUTCMinutes() !== minute
      || checked.getUTCSeconds() !== second
    ) {
      return null;
    }
    return { year, month, day, hour, minute, second, weekday: checked.getUTCDay() };
  }

  function formatDateAxisLabel(value, parsedDate, dateBucket = "none") {
    if (!parsedDate) return String(value);
    const month = DATE_AXIS_MONTHS[parsedDate.month - 1];
    const dateLabel = `${parsedDate.day} ${month} ${parsedDate.year}`;
    if (dateBucket === "hour") return `${dateLabel} ${padDateAxisTime(parsedDate.hour)}:${padDateAxisTime(parsedDate.minute)}`;
    if (dateBucket === "day") return `${DATE_AXIS_WEEKDAYS[parsedDate.weekday]} ${dateLabel}`;
    if (dateBucket === "year") return String(parsedDate.year);
    return dateLabel;
  }

  function padDateAxisTime(value) {
    return String(value).padStart(2, "0");
  }

  function getBarLayout(count) {
    if (count <= 3) {
      return { width: "62%", maxWidth: 240, categoryGap: "18%" };
    }
    if (count <= 8) {
      return { width: "56%", maxWidth: 180, categoryGap: "24%" };
    }
    if (count <= 20) {
      return { width: "46%", maxWidth: 90, categoryGap: "34%" };
    }
    if (count <= 60) {
      return { width: "68%", maxWidth: 34, categoryGap: "28%" };
    }
    return { width: null, maxWidth: 18, categoryGap: "30%" };
  }

  function invalidateLineBarTableCache() {
    tableRequestSeq += 1;
    tableRenderToken += 1;
    tableCacheKey = "";
    tableCacheData = null;
    completeTableCacheKey = "";
    completeTableCacheData = null;
    clearLineBarTable();
    if (tableSearchTimer) {
      window.clearTimeout(tableSearchTimer);
      tableSearchTimer = null;
    }
  }

  function buildTableRequest() {
    const request = buildChartRequest();
    if (!request) return null;
    return {
      ...request,
      tableSearch: state.lineBarTableSearch || "",
      tablePage: state.tablePage || 1,
      tablePageSize: TABLE_PAGE_SIZE,
    };
  }

  function scheduleLineBarTableRefresh() {
    if (tableSearchTimer) window.clearTimeout(tableSearchTimer);
    tableSearchTimer = window.setTimeout(() => {
      tableSearchTimer = null;
      if (applyClientLineBarTableFilter()) return;
      refreshLineBarTable({ force: true });
    }, TABLE_SEARCH_DEBOUNCE_MS);
  }

  function renderTableShell() {
    const tableWrap = el("tableWrap");
    if (document.getElementById("lineBarTableSearch") && document.getElementById("lineBarTableContent")) {
      const input = el("lineBarTableSearch");
      if (document.activeElement !== input && input.value !== (state.lineBarTableSearch || "")) {
        input.value = state.lineBarTableSearch || "";
      }
      return;
    }
    tableWrap.innerHTML = `
      <div class="line-bar-table-search-row app-control-strip">
        <input id="lineBarTableSearch" class="search app-control-input" placeholder="search table" />
        <button id="lineBarTableSearchClear" class="filter-action app-control-button" type="button" title="Clear table search" aria-label="Clear table search">&times;</button>
      </div>
      <div id="lineBarTableContent" class="line-bar-table-content"></div>`;
    const searchInput = el("lineBarTableSearch");
    searchInput.value = state.lineBarTableSearch || "";
    searchInput.addEventListener("input", () => {
      state.lineBarTableSearch = searchInput.value;
      state.tablePage = 1;
      scheduleLineBarTableRefresh();
    });
    el("lineBarTableSearchClear").addEventListener("click", () => {
      if (state.lineBarTableSearch || searchInput.value) {
        state.lineBarTableSearch = "";
        searchInput.value = "";
        state.tablePage = 1;
        if (applyClientLineBarTableFilter()) {
          searchInput.focus();
          return;
        }
        refreshLineBarTable({ force: true });
      }
      searchInput.focus();
    });
  }

  function clearLineBarTable() {
    closeLineBarTableContextMenu();
    lineBarTableCopyRows = [];
    lineBarTableCopyColumns = [];
    lineBarTableCopyFooterRow = null;
    if (!lineBarTable) return;
    try {
      lineBarTable.destroy();
    } catch (_) {
      // Tabulator may already have been removed by a stale render.
    }
    lineBarTable = null;
  }

  function lineBarCsvCell(value) {
    const text = value === null || value === undefined ? "" : String(value);
    return /[",\n\r]/.test(text) ? `"${text.replace(/"/g, '""')}"` : text;
  }

  function lineBarRowsToCsv(rows, options = {}) {
    if (!lineBarTableCopyColumns.length) return "";
    const header = lineBarTableCopyColumns.map((column) => lineBarCsvCell(column.title)).join(",");
    const body = rows.map((row) => lineBarTableCopyColumns.map((column) => lineBarCsvCell(row[column.field])).join(","));
    if (options.includeFooter && lineBarTableCopyFooterRow) {
      body.push(lineBarTableCopyColumns.map((column) => lineBarCsvCell(lineBarTableCopyFooterRow[column.field])).join(","));
    }
    return [header, ...body].join("\n");
  }

  function visibleLineBarCopyContextItems(options = {}) {
    const includeMessage = Boolean(options.includeMessage);
    const ids = includeMessage
      ? ["lineBarGroupMeta", "lineBarFilter", "chartMessage"]
      : ["lineBarGroupMeta", "lineBarFilter"];
    return ids.map((id) => {
      const node = el(id);
      const text = String(node?.textContent || "").trim();
      if (!node || !text || node.classList.contains("hidden")) return null;
      const style = window.getComputedStyle(node);
      if (style.display === "none" || style.visibility === "hidden") return null;
      return { text, color: style.color || getCss("--muted") || "#64748b" };
    }).filter(Boolean);
  }

  function visibleLineBarCopyContextText(options = {}) {
    return visibleLineBarCopyContextItems(options).map((item) => item.text).join("\n");
  }

  async function copyVisibleLineBarView() {
    if (state.view === "table") {
      await copyVisibleLineBarTable();
      return;
    }
    await copyVisibleLineBarChart();
  }

  async function copyVisibleLineBarTable() {
    const csv = lineBarRowsToCsv(lineBarTableCopyRows, { includeFooter: true });
    const context = visibleLineBarCopyContextText();
    const text = [context, csv].filter(Boolean).join("\n\n");
    const copied = text ? await copyTextToClipboard(text) : false;
    showClipboardToast(copied ? "Table copied" : "Could not copy table", !copied);
  }

  async function copyVisibleLineBarChart() {
    if (!navigator.clipboard?.write || typeof window.ClipboardItem !== "function") {
      showClipboardToast("Could not copy chart image", true);
      return;
    }
    try {
      const dataUrl = chart.getDataURL({
        type: "png",
        pixelRatio: 2,
        backgroundColor: getCss("--panel") || "#fff",
      });
      const blobPromise = lineBarChartClipboardBlob(dataUrl);
      await navigator.clipboard.write([new window.ClipboardItem({ "image/png": blobPromise })]);
      showClipboardToast("Chart image copied");
    } catch (_) {
      showClipboardToast("Could not copy chart image", true);
    }
  }

  async function lineBarChartClipboardBlob(dataUrl) {
    const image = await loadLineBarCopyImage(dataUrl);
    const chartNode = el("chart");
    const scale = Math.max(1, image.naturalWidth / Math.max(1, chartNode?.clientWidth || image.naturalWidth));
    const contextItems = visibleLineBarCopyContextItems({ includeMessage: true });
    const canvas = document.createElement("canvas");
    const drawContext = canvas.getContext("2d");
    if (!drawContext) throw new Error("Canvas is unavailable");
    const padding = Math.round(12 * scale);
    const fontSize = Math.round(10 * scale);
    const lineHeight = Math.round(15 * scale);
    drawContext.font = `${fontSize}px system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif`;
    const maxTextWidth = Math.max(1, image.naturalWidth - padding * 2);
    const textLines = contextItems.flatMap((item) => (
      wrapLineBarCanvasText(drawContext, item.text, maxTextWidth).map((text) => ({ text, color: item.color }))
    ));
    const headerHeight = textLines.length ? padding * 2 + textLines.length * lineHeight : 0;
    canvas.width = image.naturalWidth;
    canvas.height = image.naturalHeight + headerHeight;
    drawContext.fillStyle = getCss("--panel") || "#fff";
    drawContext.fillRect(0, 0, canvas.width, canvas.height);
    if (textLines.length) {
      drawContext.font = `${fontSize}px system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif`;
      drawContext.textAlign = "right";
      drawContext.textBaseline = "top";
      textLines.forEach((line, index) => {
        drawContext.fillStyle = line.color || getCss("--muted") || "#64748b";
        drawContext.fillText(line.text, canvas.width - padding, padding + index * lineHeight);
      });
    }
    drawContext.drawImage(image, 0, headerHeight);
    return new Promise((resolve, reject) => {
      canvas.toBlob((blob) => {
        if (blob) resolve(blob);
        else reject(new Error("PNG export failed"));
      }, "image/png");
    });
  }

  function loadLineBarCopyImage(src) {
    return new Promise((resolve, reject) => {
      const image = new Image();
      image.onload = () => resolve(image);
      image.onerror = () => reject(new Error("Chart image export failed"));
      image.src = src;
    });
  }

  function wrapLineBarCanvasText(context, text, maxWidth) {
    const words = String(text || "").split(/\s+/).filter(Boolean);
    if (!words.length) return [];
    const lines = [];
    let line = "";
    words.forEach((word) => {
      const nextLine = line ? `${line} ${word}` : word;
      if (line && context.measureText(nextLine).width > maxWidth) {
        lines.push(line);
        line = word;
      } else {
        line = nextLine;
      }
    });
    if (line) lines.push(line);
    return lines;
  }

  function selectedLineBarTableRowsForCopy() {
    if (!lineBarTable) return [];
    let selectedRows = [];
    try {
      selectedRows = typeof lineBarTable.getSelectedData === "function" ? lineBarTable.getSelectedData() : [];
    } catch (_) {
      selectedRows = [];
    }
    const selectedIds = new Set(selectedRows.map((row) => row?.__id).filter(Boolean));
    if (!selectedIds.size) return [];
    return lineBarTableCopyRows.filter((row) => selectedIds.has(row.__id));
  }

  function selectedLineBarTableCopyLabel() {
    const count = selectedLineBarTableRowsForCopy().length;
    if (!count) return "";
    return count === 1 ? "Copy selected row to clipboard" : "Copy selected rows to clipboard";
  }

  function handleLineBarTableContextMenu(event) {
    const grid = document.getElementById("lineBarTableGrid");
    if (!grid || !grid.contains(event.target)) return;
    const cell = event.target?.closest?.(".tabulator-cell[tabulator-field]");
    const selectionLabel = selectedLineBarTableCopyLabel();
    const actions = [];
    if (cell && grid.contains(cell)) {
      actions.push({ mode: "cell", label: "Copy cell to clipboard", value: cell.textContent || "" });
    }
    if (selectionLabel) actions.push({ mode: "selection", label: selectionLabel });
    actions.push({ mode: "table", label: "Copy table to clipboard" });
    if (selectionLabel) {
      actions.push({ divider: true });
      actions.push({ mode: "clear-selection", label: "Clear selection" });
    }
    event.preventDefault();
    event.stopPropagation();
    openLineBarTableContextMenu(event, actions);
  }

  function lineBarTableContextMenu() {
    let menu = document.getElementById("lineBarTableContextMenu");
    if (menu) return menu;
    menu = document.createElement("div");
    menu.id = "lineBarTableContextMenu";
    menu.className = "line-bar-table-context-menu";
    menu.hidden = true;
    menu.setAttribute("role", "menu");
    menu.addEventListener("click", copyLineBarTableContextValue);
    document.body.append(menu);
    return menu;
  }

  function openLineBarTableContextMenu(event, actions = []) {
    closeLineBarTableContextMenu();
    if (!actions.length) return;
    const menu = lineBarTableContextMenu();
    actions.forEach((action) => {
      if (action.divider) {
        const divider = document.createElement("div");
        divider.className = "line-bar-table-context-menu-divider";
        divider.setAttribute("role", "separator");
        menu.append(divider);
        return;
      }
      const button = document.createElement("button");
      button.className = "line-bar-table-context-menu-item";
      button.type = "button";
      button.setAttribute("role", "menuitem");
      button.dataset.copyMode = action.mode || "cell";
      button.dataset.copyValue = action.value || "";
      button.textContent = action.label || "Copy cell to clipboard";
      menu.append(button);
    });
    menu.hidden = false;
    positionLineBarTableContextMenu(menu, event.clientX, event.clientY);
    menu.querySelector("button")?.focus({ preventScroll: true });
    window.addEventListener("pointerdown", handleLineBarTableContextPointerDown, true);
    window.addEventListener("keydown", handleLineBarTableContextKeydown, true);
    window.addEventListener("resize", closeLineBarTableContextMenu, true);
    window.addEventListener("scroll", closeLineBarTableContextMenu, true);
  }

  function positionLineBarTableContextMenu(menu, clientX, clientY) {
    const margin = 8;
    menu.style.left = "0px";
    menu.style.top = "0px";
    const rect = menu.getBoundingClientRect();
    const maxLeft = Math.max(margin, window.innerWidth - rect.width - margin);
    const maxTop = Math.max(margin, window.innerHeight - rect.height - margin);
    const left = Math.min(Math.max(margin, clientX || margin), maxLeft);
    const top = Math.min(Math.max(margin, clientY || margin), maxTop);
    menu.style.left = `${left}px`;
    menu.style.top = `${top}px`;
  }

  async function copyLineBarTableContextValue(event) {
    const button = event.target?.closest?.("button[data-copy-mode]");
    const menu = document.getElementById("lineBarTableContextMenu");
    if (!button || !menu?.contains(button)) return;
    event.preventDefault();
    const mode = button.dataset.copyMode || "cell";
    if (mode === "selection") {
      const rows = selectedLineBarTableRowsForCopy();
      const csv = lineBarRowsToCsv(rows);
      const copied = csv ? await copyTextToClipboard(csv) : false;
      showClipboardToast(copied ? `Selected row${rows.length === 1 ? "" : "s"} copied` : "Could not copy selected rows", !copied);
      closeLineBarTableContextMenu();
      return;
    }
    if (mode === "table") {
      const csv = lineBarRowsToCsv(lineBarTableCopyRows, { includeFooter: true });
      const copied = csv ? await copyTextToClipboard(csv) : false;
      showClipboardToast(copied ? "Table copied" : "Could not copy table", !copied);
      closeLineBarTableContextMenu();
      return;
    }
    if (mode === "clear-selection") {
      clearLineBarTableSelection();
      closeLineBarTableContextMenu();
      return;
    }
    const value = button.dataset.copyValue || "";
    const copied = await copyTextToClipboard(value);
    showClipboardToast(copied ? "Cell copied to clipboard" : "Could not copy cell", !copied);
    closeLineBarTableContextMenu();
  }

  function clearLineBarTableSelection() {
    try {
      lineBarTable?.deselectRow?.();
    } catch (_) {
      // Ignore stale Tabulator instances.
    }
  }

  function handleLineBarTableContextPointerDown(event) {
    const menu = document.getElementById("lineBarTableContextMenu");
    if (!menu || menu.hidden || menu.contains(event.target)) return;
    closeLineBarTableContextMenu();
  }

  function handleLineBarTableContextKeydown(event) {
    if (event.key !== "Escape") return;
    event.preventDefault();
    closeLineBarTableContextMenu();
  }

  function closeLineBarTableContextMenu() {
    const menu = document.getElementById("lineBarTableContextMenu");
    if (menu) {
      menu.hidden = true;
      menu.replaceChildren();
    }
    window.removeEventListener("pointerdown", handleLineBarTableContextPointerDown, true);
    window.removeEventListener("keydown", handleLineBarTableContextKeydown, true);
    window.removeEventListener("resize", closeLineBarTableContextMenu, true);
    window.removeEventListener("scroll", closeLineBarTableContextMenu, true);
  }

  function renderLineBarTableLoading() {
    const content = document.getElementById("lineBarTableContent");
    if (!content) return;
    tableRenderToken += 1;
    clearLineBarTable();
    content.innerHTML = `<div class="line-bar-table-state">Loading table...</div>`;
  }

  function renderLineBarTableError(message) {
    const content = document.getElementById("lineBarTableContent");
    if (!content) return;
    tableRenderToken += 1;
    clearLineBarTable();
    content.innerHTML = `<div class="line-bar-table-state line-bar-table-state-error">${escapeHtml(message || "Table query failed")}</div>`;
  }

  function lineBarTableMaxHeight() {
    const content = document.getElementById("lineBarTableContent");
    if (!content) return 240;
    const contentRect = content.getBoundingClientRect();
    const contentStyle = window.getComputedStyle(content);
    const padding = (parseFloat(contentStyle.paddingTop) || 0) + (parseFloat(contentStyle.paddingBottom) || 0);
    const pager = content.querySelector(".table-pagination");
    let pagerSpace = 0;
    if (pager) {
      const pagerRect = pager.getBoundingClientRect();
      const pagerStyle = window.getComputedStyle(pager);
      pagerSpace = pagerRect.height
        + (parseFloat(pagerStyle.marginTop) || 0)
        + (parseFloat(pagerStyle.marginBottom) || 0);
    }
    const available = Math.floor((contentRect.height || content.clientHeight || 0) - padding - pagerSpace);
    return Math.max(140, available || 240);
  }

  function syncLineBarTableMaxHeight() {
    const target = document.getElementById("lineBarTableGrid");
    if (!target) return 240;
    const maxHeight = lineBarTableMaxHeight();
    target.style.maxHeight = `${maxHeight}px`;
    return maxHeight;
  }

  function lineBarTableHeaderMinWidth(label, minimum) {
    const text = String(label || "");
    const estimated = Math.ceil(text.length * 7 + 18);
    return Math.max(minimum, Math.min(240, estimated));
  }

  async function refreshLineBarTable(options = {}) {
    if (state.tool !== "line_bar" || state.view !== "table") return null;
    renderTableShell();
    const request = buildTableRequest();
    if (!request) return null;
    const requestKey = stableRequestKey(request);
    if (!options.force && tableCacheData && tableCacheKey === requestKey) {
      measureToolRender("line_bar", () => renderLineBarTableContents(tableCacheData));
      return tableCacheData;
    }
    if (!options.forceServer && applyClientLineBarTableFilter({ requestKey })) return tableCacheData;
    const requestSeq = tableRequestSeq + 1;
    tableRequestSeq = requestSeq;
    renderLineBarTableLoading();
    try {
      const data = await api("/api/line-bar/table", { method: "POST", body: JSON.stringify(request), clientTiming: true });
      if (requestSeq !== tableRequestSeq) return null;
      tableCacheKey = requestKey;
      tableCacheData = data;
      rememberCompleteLineBarTableSource(data, request);
      syncDuckDbTimingFromData("line_bar", data);
      syncClientTimingFromData("line_bar", data);
      measureToolRender("line_bar", () => renderLineBarTableContents(data));
      return data;
    } catch (error) {
      if (requestSeq !== tableRequestSeq) return null;
      renderLineBarTableError(error.message);
      setStatus(error.message, true);
      return null;
    }
  }

  function renderLineBarTableContents(data) {
    const rowsData = Array.isArray(data.rows) ? data.rows : [];
    const weightLabel = data.denominator?.bar_label || "Weight";
    const weightedMode = Boolean(data.denominator?.column);
    const summaryResponses = Array.isArray(data.summary?.responses) ? data.summary.responses : [];
    const summaryVolume = Number.isFinite(Number(data.summary?.volume)) ? Number(data.summary.volume) : 0;
    const summaryRowCount = Number.isFinite(Number(data.summary?.row_count)) ? Number(data.summary.row_count) : summaryVolume;
    const tableMeta = data.table || {};
    const page = Math.max(1, Number(tableMeta.page) || 1);
    const pageSize = Math.max(1, Number(tableMeta.page_size) || TABLE_PAGE_SIZE);
    const pageCount = Math.max(1, Number(tableMeta.page_count) || 1);
    const matchCount = Math.max(0, Number(tableMeta.match_count) || 0);
    state.tablePage = page;
    updateMetricTitles(data);
    syncLineBarTablePresentation(data);
    const start = matchCount ? (page - 1) * pageSize + 1 : 0;
    const end = matchCount ? Math.min((page - 1) * pageSize + rowsData.length, matchCount) : 0;
    const pager = `<div class="table-pagination">
        <span>${start.toLocaleString()}-${end.toLocaleString()} of ${matchCount.toLocaleString()} groups</span>
        <button id="tablePrevBtn" type="button"${page <= 1 ? " disabled" : ""}>Previous</button>
        <span>Page ${page.toLocaleString()} of ${pageCount.toLocaleString()}</span>
        <button id="tableNextBtn" type="button"${page >= pageCount ? " disabled" : ""}>Next</button>
      </div>`;
    const content = document.getElementById("lineBarTableContent");
    if (!content) return;
    const renderToken = tableRenderToken + 1;
    tableRenderToken = renderToken;
    clearLineBarTable();
    content.innerHTML = `<div id="lineBarTableGrid" class="line-bar-table-grid"></div>${pager}`;
    document.getElementById("tablePrevBtn")?.addEventListener("click", () => {
      if (state.tablePage <= 1) return;
      state.tablePage -= 1;
      refreshLineBarTable({ force: true });
    });
    document.getElementById("tableNextBtn")?.addEventListener("click", () => {
      if (state.tablePage >= pageCount) return;
      state.tablePage += 1;
      refreshLineBarTable({ force: true });
    });
    const tableRows = rowsData.map((row, index) => {
      const rowCount = finiteNumberOrNull(row.row_count);
      const displayRow = {
        __id: `${page}:${index}`,
        x: formatTableXLabel(row, data),
        row_count: formatNumber(rowCount === null ? row.volume : rowCount),
        volume: formatNumber(row.volume),
      };
      data.responses.forEach((_, responseIndex) => {
        displayRow[`resp${responseIndex}`] = formatResponseValue(row[`resp${responseIndex}`]);
      });
      return displayRow;
    });
    lineBarTableCopyRows = tableRows.map((row) => ({ ...row }));
    lineBarTableCopyColumns = [
      { title: data.x, field: "x" },
      ...(weightedMode ? [{ title: "Row count", field: "row_count" }] : []),
      { title: weightLabel, field: "volume" },
      ...data.responses.map((response, responseIndex) => ({ title: response.label, field: `resp${responseIndex}` })),
    ];
    lineBarTableCopyFooterRow = {
      x: "Total",
      ...(weightedMode ? { row_count: formatNumber(summaryRowCount) } : {}),
      volume: formatNumber(summaryVolume),
    };
    data.responses.forEach((_, responseIndex) => {
      lineBarTableCopyFooterRow[`resp${responseIndex}`] = formatResponseValue(summaryResponses[responseIndex]);
    });
    const columns = [
      {
        title: data.x,
        field: "x",
        headerSort: false,
        frozen: true,
        minWidth: lineBarTableHeaderMinWidth(data.x, 130),
        widthGrow: 2,
        hozAlign: "left",
        bottomCalc: () => "Total",
      },
      ...(weightedMode ? [{
        title: "Row count",
        field: "row_count",
        headerSort: false,
        headerHozAlign: "right",
        hozAlign: "right",
        minWidth: lineBarTableHeaderMinWidth("Row count", 90),
        widthGrow: 0.7,
        bottomCalc: () => formatNumber(summaryRowCount),
      }] : []),
      {
        title: weightLabel,
        field: "volume",
        headerSort: false,
        headerHozAlign: "right",
        hozAlign: "right",
        minWidth: lineBarTableHeaderMinWidth(weightLabel, 90),
        widthGrow: 0.7,
        bottomCalc: () => formatNumber(summaryVolume),
      },
      ...data.responses.map((response, responseIndex) => ({
        title: response.label,
        field: `resp${responseIndex}`,
        headerSort: false,
        headerHozAlign: "right",
        hozAlign: "right",
        minWidth: lineBarTableHeaderMinWidth(response.label, 110),
        widthGrow: 0.8,
        bottomCalc: () => formatResponseValue(summaryResponses[responseIndex]),
      })),
    ];
    loadTabulator().then((Tabulator) => {
      if (renderToken !== tableRenderToken) return;
      const target = document.getElementById("lineBarTableGrid");
      if (!target) return;
      const maxHeight = syncLineBarTableMaxHeight();
      lineBarTable = new Tabulator(target, {
        data: tableRows,
        index: "__id",
        maxHeight,
        layout: "fitColumns",
        placeholder: "No matching rows",
        reactiveData: false,
        selectableRows: true,
        renderVertical: "virtual",
        rowHeight: 22,
        columnDefaults: {
          resizable: false,
          headerSort: false,
          headerWordWrap: true,
          formatter: (cell) => escapeHtml(cell.getValue() ?? ""),
        },
        columns,
      });
      target.addEventListener("contextmenu", handleLineBarTableContextMenu);
    }).catch((error) => {
      if (renderToken !== tableRenderToken) return;
      renderLineBarTableError(error.message || String(error));
    });
  }

  function setView(view, options = {}) {
    const shouldRefresh = options.refresh !== false;
    state.view = view;
    if (state.tool !== "line_bar") return;
    el("chartTab").classList.toggle("active", view === "chart");
    el("tableTab").classList.toggle("active", view === "table");
    el("chart").classList.toggle("hidden", view !== "chart");
    el("tableWrap").classList.toggle("hidden", view !== "table");
    el("ukMap").classList.add("hidden");
    el("mapLegend").classList.add("hidden");
    el("chartMessage").classList.toggle("hidden", view !== "chart" || !el("chartMessage").textContent);
    if (view === "chart") {
      if (!shouldRefresh) {
        chart.resize();
        refreshDateXAxisLabelsForCurrentZoom();
        return;
      }
      if (lineBarChartDirty) {
        lineBarChartDirty = false;
        if (!applyClientLineBarSort()) refreshChart();
        return;
      }
      chart.resize();
      refreshDateXAxisLabelsForCurrentZoom();
    } else {
      renderTableShell();
      if (shouldRefresh) refreshLineBarTable();
    }
  }

  function showPendingRestore(view) {
    cancelLineBarRequests();
    setView(view, { refresh: false });
    setStatus("");
    setChartMessage("");
    renderMetricTitle(el("expectedMetricTitle"), "Expected");
    if (view === "table") {
      setGroupMeta("line_bar", "Loading table...");
      renderTableShell();
      renderLineBarTableLoading();
      return;
    }
    setGroupMeta("line_bar", "Computing...");
    setChartPendingHidden(true);
  }

  function bindControls() {
    chart.on("legendselectchanged", updateResponseAxisForLegendSelection);
    chart.on("datazoom", scheduleDateXAxisLabelRefresh);
    const lineBarControls = new Set(["sort", "lowGroup", "labels", "bandWidth", "quantileMode", "dateBucket", "transform", "sigma", "partialDependence", "featureSort", "expectedSort"]);
    document.querySelectorAll(".segmented").forEach((group) => {
      if (!lineBarControls.has(group.dataset.control)) return;
      group.addEventListener("click", (event) => {
        if (event.target.tagName !== "BUTTON") return;
        if (group.dataset.control === "bandWidth" && event.target.dataset.action) {
          stepBandWidth(event.target.dataset.action === "band-down" ? -1 : 1);
          clearActiveFavouriteSelection();
          return;
        }
        group.querySelectorAll("button").forEach((button) => button.classList.remove("active"));
        event.target.classList.add("active");
        const previousControlValue = state[group.dataset.control];
        state[group.dataset.control] = event.target.dataset.value;
        if (state[group.dataset.control] !== previousControlValue) clearActiveFavouriteSelection();
        if (group.dataset.control === "featureSort") {
          renderFeatures();
          return;
        }
        if (group.dataset.control === "expectedSort") {
          renderExpectedNumerators();
          return;
        }
        if (group.dataset.control === "sort") {
          state.tablePage = 1;
          if (state.view === "table") {
            if (!applyClientLineBarSort({ render: false })) lineBarChartDirty = true;
            if (applyClientLineBarTableSort()) return;
            refreshLineBarTable({ force: true });
            return;
          }
          if (applyClientLineBarSort()) return;
        }
        if (group.dataset.control === "bandWidth") {
          clearPendingBandSuggestion();
          state.bandFeature = currentBandFeatureKey();
          if (state.quantileMode === "quantile") {
            normalizeBandWidthForQuantiles();
          } else {
            syncBandingControl();
          }
        }
        if (group.dataset.control === "quantileMode") {
          if (state.quantileMode === "quantile" && previousControlValue !== "quantile") {
            clearPendingBandSuggestion();
            rememberNonQuantileBandWidthForCurrentFeature();
            state.bandWidth = "10";
            state.bandFeature = currentBandFeatureKey();
            syncBandingControl();
          } else if (state.quantileMode === "quantile") {
            normalizeBandWidthForQuantiles();
          } else {
            restoreNonQuantileBandWidthForCurrentFeature();
            syncBandingControl();
          }
          syncQuantileControl();
        }
        if (group.dataset.control === "dateBucket") {
          clearPendingDateBucketSuggestion();
          state.dateBucket = normaliseDateBucket(state.dateBucket);
          state.dateBucketFeature = currentDateBucketFeatureKey();
          state.dateBucketManualKey = state.dateBucketFeature;
          syncDateBucketControl();
        }
        if (group.dataset.control === "partialDependence") {
          updateAxisControls();
        }
        refreshChart({ renderIfCached: group.dataset.control === "labels" });
      });
    });
    el("expectedNumerator").addEventListener("change", () => {
      const option = el("expectedNumerator").selectedOptions[0];
      state.expectedSelections = option?.value
        ? [{
            value: option.value,
            sourceId: option.dataset.sourceId || state.source || "dataset",
            metricKind: option.dataset.metricKind || "metric",
          }]
        : [];
      const sourceChanged = syncExpectedSourceFromSelection();
      if (!sourceChanged) {
        renderExpectedNumerators();
        updateAxisControls();
      }
      clearActiveFavouriteSelection();
      refreshChart({ force: sourceChanged });
    });
    el("expectedSearch").addEventListener("input", renderExpectedNumerators);
    el("featureSearch").addEventListener("input", renderFeatures);
    bindLineBarPickerKeyboard("expectedSearch", "expectedList");
    bindLineBarPickerKeyboard("featureSearch", "featureList");
    el("expectedSearchClear").addEventListener("click", () => clearSearchInput("expectedSearch", renderExpectedNumerators));
    el("featureSearchClear").addEventListener("click", () => clearSearchInput("featureSearch", renderFeatures));
    el("lineBarCopyBtn")?.addEventListener("click", copyVisibleLineBarView);
    el("chartTab").addEventListener("click", () => {
      const changed = state.view !== "chart";
      setView("chart");
      if (changed) clearActiveFavouriteSelection();
    });
    el("tableTab").addEventListener("click", () => {
      const changed = state.view !== "table";
      setView("table");
      if (changed) clearActiveFavouriteSelection();
    });
  }

  function resize() {
    if (state.view === "table") {
      syncLineBarTableMaxHeight();
      lineBarTable?.redraw?.(true);
      return;
    }
    chart.resize();
    refreshDateXAxisLabelsForCurrentZoom();
  }

  function refreshTheme() {
    if (state.lastData) measureToolRender("line_bar", () => renderChart(state.lastData));
  }

  return {
    buildRequest: buildChartRequest,
    fetchData: fetchChartData,
    useCached: useCachedChartData,
    render: renderChartData,
    refreshTable: refreshLineBarTable,
    showPendingRestore,
    cancelRequests: cancelLineBarRequests,
    bindControls,
    setView,
    resize,
    refreshTheme,
    renderFeatures,
    renderExpectedNumerators,
    updateAxisControls,
    canNavigateToFeature: canNavigateToLineBarFeature,
    navigateToFeature: navigateToLineBarFeature,
  };
}
