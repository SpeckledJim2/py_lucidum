import { loadTabulator } from "./shared/tabulator.js";

const LINE_BAR_SPECIAL_COLUMN_NAMES = [
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
  formatNumber,
  formatChartLabel,
  formatLineLabel,
  formatLineValue,
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
  const DATE_BUCKET_VALUES = new Set(["none", "hour", "day", "week", "month", "year"]);
  const RESPONSE_AXIS_PADDING = 0.08;
  const RESPONSE_AXIS_TARGET_INTERVALS = 15;
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
  let tableRenderToken = 0;
  let lineBarTable = null;
  let dateXAxisContext = null;
  let dateXAxisRefreshFrame = null;

  function isNumericKind(kind) {
    return kind === "numeric" || kind === "integer";
  }

  function isDateKind(kind) {
    return kind === "date" || kind === "datetime";
  }

  function isLineBarSpecialColumn(column) {
    return LINE_BAR_SPECIAL_COLUMN_NAMES.includes(String(column?.name || ""));
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
    const requestSeq = (state.bandSuggestionRequestSeq || 0) + 1;
    state.bandSuggestionRequestSeq = requestSeq;
    state.bandSuggestionPendingKey = bandFeatureKey;
    syncBandingControl();
    try {
      const sourceId = selectedColumn()?.source_id || state.xSource || state.source || "dataset";
      const data = await api("/api/banding/suggestion", {
        method: "POST",
        body: JSON.stringify({
          source: sourceId,
          xSource: sourceId,
          feature: state.x,
          filter: state.activeFilter,
        }),
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
    const hasExpected = Boolean(el("expectedNumerator").value);
    const shapSortAvailable = isCategorical && shapPartialDependenceVisible() && shapOverlayAvailableForSelectedColumn();
    el("sortControl").classList.toggle("hidden", !isCategorical);
    el("expectedSortButton").classList.toggle("hidden", !hasExpected);
    el("shapSortButton")?.classList.toggle("hidden", !shapSortAvailable);
    el("dateControl").classList.toggle("hidden", !isDate);
    el("bandControl").classList.toggle("hidden", !isNumeric);
    el("quantileControl").classList.toggle("hidden", !isNumeric);
    const bandFeatureKey = currentBandFeatureKey();
    const dateBucketKey = currentDateBucketFeatureKey();
    if (isNumeric && state.tool === "line_bar" && state.bandFeature !== bandFeatureKey) {
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

  function renderExpectedNumerators(options = {}) {
    const query = el("expectedSearch").value.trim().toLowerCase();
    const select = el("expectedNumerator");
    const list = el("expectedList");
    const scrollPosition = captureLineBarPickerScroll(list, options.preserveScroll);
    const { pinned, scroll } = resetLineBarPickerList(list, true);

    function addExpectedButton(target, label, value, kind, sourceId = "", extraClass = "") {
      const button = document.createElement("button");
      button.type = "button";
      button.className = `feature ${extraClass} ${value === select.value ? "active" : ""}`.trim();
      if (sourceId) button.dataset.sourceId = sourceId;
      button.dataset.value = value;
      button.innerHTML = `<span>${escapeHtml(label)}</span><span class="kind">${escapeHtml(kind)}</span>`;
      button.addEventListener("click", (event) => {
        const changed = select.value !== value;
        select.value = value;
        const sourceChanged = syncExpectedSourceFromSelection({
          expectedValue: value,
          expectedSource: sourceId,
        });
        if (!sourceChanged) {
          renderExpectedNumerators({ preserveScroll: true });
          updateAxisControls();
        }
        if (changed || sourceChanged) refreshChart({ force: sourceChanged });
        if (event.isTrusted) {
          focusLineBarPickerButton(list, { value, sourceId, index: 0 });
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
      resetLineBarPickerList(list, false);
      renderFeatureImportanceRows(query, list);
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
    addFeatureButton(list, {
      label: col.name,
      detail: col.kind,
      sourceId,
      extraClass,
      active,
      onClick: () => {
        const previousDateBucketKey = currentDateBucketFeatureKey();
        state.x = col.name;
        state.xSource = sourceId;
        resetDateBucketSuggestionIfKeyChanged(previousDateBucketKey);
        renderFeatures({ preserveScroll: true });
        updateAxisControls();
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

  function lineBarPickerButtons(list) {
    return Array.from(list.querySelectorAll("button.feature"))
      .filter((button) => !button.disabled && button.offsetParent !== null);
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
    const buttons = lineBarPickerButtons(list);
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
    target.click();
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
    if (el("expectedNumerator").value) {
      const option = el("expectedNumerator").selectedOptions[0];
      const source = option?.dataset.metricKind === "prediction" ? option.dataset.sourceId || "" : "";
      responses.push({
        label: el("expectedNumerator").value,
        numerator: el("expectedNumerator").value,
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
    if (isNumeric && state.bandFeature !== bandFeatureKey) {
      requestBandSuggestionForSelectedColumn(bandFeatureKey);
      return null;
    }
    if (isNumeric && state.bandSuggestionPendingKey === bandFeatureKey) {
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
    const xSource = column && isModelPredictionColumn(column) ? column.source_id || state.xSource || state.source || "dataset" : "";
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

  function renderChartData(data, options = {}) {
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
    const groupMeta = `${groupCount.toLocaleString()} groups · ${rowMeta}`;
    const warnings = [...(data.warnings || [])].filter(Boolean).join(" ");
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
    const labels = data.rows.map((r) => formatXLabel(r.x, data.x_kind));
    const labelMode = state.labels;
    const rawXValues = data.rows.map((r) => r.x);
    const dateBucket = normaliseDateBucket(data.date_bucket);
    const xLabelPolicy = getXAxisLabelPolicy(labels, data.x_kind, rawXValues, dateBucket, chart.getWidth?.() || el("chart").clientWidth);
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
    const responseColors = [actualColor, expectedColor];
    const nColor = getCss("--bar");
    const weightLabel = data.denominator?.bar_label || "Weight";
    const sigmaColor = "#8a94a6";
    const legendData = [
      ...data.responses.map((response) => response.label),
      { name: weightLabel, icon: "roundRect", itemStyle: { color: nColor, borderColor: nColor } },
    ];
    const mainLegendSelection = matchingLegendSelection(previousOption, legendData);
    const overlayLegendSelection = matchingLegendSelection(previousOption, overlayLegendData);
    const responseAxis = responseAxisOptions(data, { ...mainLegendSelection, ...overlayLegendSelection });
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
      label: { show: showBarLabels, position: "top", fontSize: 10, formatter: formatChartLabel },
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
      label: { show: showLineLabels, fontSize: 10, formatter: formatResponseLabel },
    }));
    const upliftBaseline = upliftBaselineSeries(data);

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
        color: [actualColor, expectedColor, nColor],
        tooltip: {
          trigger: "axis",
          formatter: (params) => formatChartTooltip(params, weightLabel),
        },
        legend: lineBarLegendOptions(legendData, mainLegendSelection, overlayLegendData, overlayLegendSelection),
        grid: { left: 72, right: 76, top: hasOverlaySeries ? 82 : 56, bottom: xLabelPolicy.bottom, containLabel: false },
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
            hideOverlap: false,
            showMinLabel: xLabelPolicy.showMinLabel,
            showMaxLabel: xLabelPolicy.showMaxLabel,
            rotate: xLabelPolicy.rotate,
            fontSize: xLabelPolicy.fontSize,
            margin: 8,
          },
          axisLine: { lineStyle: { color: getCss("--line") } },
        },
        yAxis: [
          { type: "value", scale: true, splitNumber: RESPONSE_AXIS_TARGET_INTERVALS, min: responseAxis.min, max: responseAxis.max, interval: responseAxis.interval, axisLabel: { color: getCss("--text"), formatter: (value) => formatResponseValue(value) }, splitLine: { lineStyle: { color: getCss("--line") } } },
          { type: "value", axisLabel: { color: getCss("--text"), formatter: (value) => formatNumber(value) }, splitLine: { show: false } },
        ],
        dataZoom: xLabelPolicy.dataZoomEnabled ? lineBarDataZoomOptions() : [],
        series: [barSeries, ...shapSeries, ...glmSeries, ...lineSeries, ...(upliftBaseline ? [upliftBaseline] : []), ...customSeries],
      },
      true,
    );
    requestAnimationFrame(() => {
      chart.resize();
      refreshDateXAxisLabelsForCurrentZoom();
    });
    return chartDensityMessage(labels.length, !xLabelPolicy.show, !dataLabelsAllowed && labelMode !== "-", xLabelPolicy.hiddenReason);
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
      top: 0,
      data: legendData,
      selected: mainLegendSelection,
      textStyle,
    };
    if (!overlayLegendData.length) return mainLegend;
    return [
      mainLegend,
      {
        top: 26,
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

  function chartDensityMessage(groupCount, xLabelsHidden, chartLabelsHidden, xLabelReason = "") {
    if (!xLabelsHidden && !chartLabelsHidden) return "";
    if (xLabelsHidden && xLabelReason && !chartLabelsHidden) return `X-axis labels hidden ${xLabelReason}.`;
    const labelTarget = xLabelsHidden && chartLabelsHidden
      ? "X-axis and chart labels"
      : xLabelsHidden ? "X-axis labels" : "Chart labels";
    return `${labelTarget} hidden as >${LABEL_DENSITY_LIMIT.toLocaleString()} categories.`;
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

  function formatChartTooltip(params, weightLabel) {
    const items = Array.isArray(params) ? params : [params];
    if (!items.length) return "";
    const lines = [escapeHtml(items[0].axisValueLabel ?? items[0].name ?? "")];
    items.forEach((item) => {
      const value = Array.isArray(item.value) ? item.value[1] : item.value;
      const formatter = item.seriesName === weightLabel ? formatNumber : formatResponseValue;
      lines.push(`${item.marker || ""}${escapeHtml(item.seriesName)}: ${escapeHtml(formatter(value))}`);
    });
    return lines.join("<br/>");
  }

  function updateMetricTitles(data) {
    const summaries = data.response_summaries || [];
    renderMetricTitle(el("expectedMetricTitle"), "Expected", summaries[1]?.value);
  }

  function formatResponseLabel(params) {
    const value = Array.isArray(params.value) ? params.value[1] : params.value;
    return formatResponseValue(value);
  }

  function formatResponseValue(value) {
    return isUpliftTransform() ? formatUpliftPercent(value) : formatLineValue(value);
  }

  function isUpliftTransform() {
    return String(state.transform || "none") === "one";
  }

  function isBaseReferenceTransform() {
    return ["zero", "one"].includes(String(state.transform || "none"));
  }

  function isBaseWeightBar(data, row) {
    if (!isBaseReferenceTransform()) return false;
    if (String(data?.transform?.reference || "") !== "base") return false;
    const baseX = data?.transform?.base_x;
    if (baseX === null || baseX === undefined) return false;
    return String(row?.x) === String(baseX);
  }

  function weightBarColor(data, row) {
    if (row?.is_tail) return getCss("--tail");
    return isBaseWeightBar(data, row) ? getCss("--base-bar") : getCss("--bar");
  }

  function upliftBaselineSeries(data) {
    if (!isUpliftTransform()) return null;
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

  function responseAxisOptions(data, selected = null) {
    const extent = withUpliftBaselineExtent(
      responseAxisExtent(data.rows, data.responses, data.partial_dependence, selected)
        || responseAxisExtent(data.rows, data.responses, data.partial_dependence),
    );
    return responseAxisBounds(extent) || {};
  }

  function withUpliftBaselineExtent(extent) {
    if (!extent || !isUpliftTransform()) return extent;
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
    const responseAxis = responseAxisOptions(state.lastData, legendSelectionFromOption(chart.getOption()));
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
        hiddenReason: `as >${LABEL_DENSITY_LIMIT.toLocaleString()} categories`,
      };
    }
    const rotate = labels.length > 30 || maxLength > 10 ? 65 : 0;
    const fontSize = labels.length > 50 ? 8 : 10;
    const estimatedTextWidth = maxLength * fontSize * 0.5;
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
      hiddenReason: "",
    };
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
      <div class="line-bar-table-search-row">
        <input id="lineBarTableSearch" class="search" placeholder="search table" />
        <button id="lineBarTableSearchClear" class="filter-action" type="button" title="Clear table search" aria-label="Clear table search">&times;</button>
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
        refreshLineBarTable({ force: true });
      }
      searchInput.focus();
    });
  }

  function clearLineBarTable() {
    if (!lineBarTable) return;
    try {
      lineBarTable.destroy();
    } catch (_) {
      // Tabulator may already have been removed by a stale render.
    }
    lineBarTable = null;
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
    const requestSeq = tableRequestSeq + 1;
    tableRequestSeq = requestSeq;
    renderLineBarTableLoading();
    try {
      const data = await api("/api/line-bar/table", { method: "POST", body: JSON.stringify(request), clientTiming: true });
      if (requestSeq !== tableRequestSeq) return null;
      tableCacheKey = requestKey;
      tableCacheData = data;
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
    const summaryResponses = Array.isArray(data.summary?.responses) ? data.summary.responses : [];
    const summaryVolume = Number.isFinite(Number(data.summary?.volume)) ? Number(data.summary.volume) : 0;
    const tableMeta = data.table || {};
    const page = Math.max(1, Number(tableMeta.page) || 1);
    const pageSize = Math.max(1, Number(tableMeta.page_size) || TABLE_PAGE_SIZE);
    const pageCount = Math.max(1, Number(tableMeta.page_count) || 1);
    const matchCount = Math.max(0, Number(tableMeta.match_count) || 0);
    state.tablePage = page;
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
      const displayRow = {
        __id: `${page}:${index}`,
        x: formatXLabel(row.x, data.x_kind),
        volume: formatNumber(row.volume),
      };
      data.responses.forEach((_, responseIndex) => {
        displayRow[`resp${responseIndex}`] = formatResponseValue(row[`resp${responseIndex}`]);
      });
      return displayRow;
    });
    const columns = [
      {
        title: data.x,
        field: "x",
        headerSort: false,
        frozen: true,
        minWidth: 130,
        widthGrow: 2,
        hozAlign: "left",
        bottomCalc: () => "Total",
      },
      {
        title: weightLabel,
        field: "volume",
        headerSort: false,
        headerHozAlign: "right",
        hozAlign: "right",
        minWidth: 90,
        widthGrow: 0.7,
        bottomCalc: () => formatNumber(summaryVolume),
      },
      ...data.responses.map((response, responseIndex) => ({
        title: response.label,
        field: `resp${responseIndex}`,
        headerSort: false,
        headerHozAlign: "right",
        hozAlign: "right",
        minWidth: 110,
        widthGrow: 0.8,
        bottomCalc: () => formatResponseValue(summaryResponses[responseIndex]),
      })),
    ];
    loadTabulator().then((Tabulator) => {
      if (renderToken !== tableRenderToken) return;
      const target = document.getElementById("lineBarTableGrid");
      if (!target) return;
      lineBarTable = new Tabulator(target, {
        data: tableRows,
        index: "__id",
        height: "100%",
        layout: "fitColumns",
        placeholder: "No matching rows",
        reactiveData: false,
        selectable: false,
        renderVertical: "virtual",
        rowHeight: 22,
        columnDefaults: {
          resizable: false,
          headerSort: false,
          formatter: (cell) => escapeHtml(cell.getValue() ?? ""),
        },
        columns,
      });
    }).catch((error) => {
      if (renderToken !== tableRenderToken) return;
      renderLineBarTableError(error.message || String(error));
    });
  }

  function setView(view) {
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
      chart.resize();
      refreshDateXAxisLabelsForCurrentZoom();
    } else {
      renderTableShell();
      refreshLineBarTable();
    }
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
          return;
        }
        group.querySelectorAll("button").forEach((button) => button.classList.remove("active"));
        event.target.classList.add("active");
        const previousControlValue = state[group.dataset.control];
        state[group.dataset.control] = event.target.dataset.value;
        if (group.dataset.control === "featureSort") {
          renderFeatures();
          return;
        }
        if (group.dataset.control === "expectedSort") {
          renderExpectedNumerators();
          return;
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
            state.bandWidth = "10";
            state.bandFeature = currentBandFeatureKey();
            syncBandingControl();
          } else if (state.quantileMode === "quantile") {
            normalizeBandWidthForQuantiles();
          } else {
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
      const sourceChanged = syncExpectedSourceFromSelection();
      if (!sourceChanged) {
        renderExpectedNumerators();
        updateAxisControls();
      }
      refreshChart({ force: sourceChanged });
    });
    el("expectedSearch").addEventListener("input", renderExpectedNumerators);
    el("featureSearch").addEventListener("input", renderFeatures);
    bindLineBarPickerKeyboard("expectedSearch", "expectedList");
    bindLineBarPickerKeyboard("featureSearch", "featureList");
    el("expectedSearchClear").addEventListener("click", () => clearSearchInput("expectedSearch", renderExpectedNumerators));
    el("featureSearchClear").addEventListener("click", () => clearSearchInput("featureSearch", renderFeatures));
    el("chartTab").addEventListener("click", () => setView("chart"));
    el("tableTab").addEventListener("click", () => setView("table"));
  }

  function resize() {
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
