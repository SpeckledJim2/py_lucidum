import { createGbmTreeViewer } from "./gbm-tree-viewer.js";
import { createGbmShapTool } from "./gbm-shap-tool.js";
import { createGbmStackedShapTool } from "./gbm-stacked-shap-tool.js";

const GBM_PARAMETER_OPTIONS = {
  objective: [
    "regression",
    "regression_l1",
    "huber",
    "fair",
    "poisson",
    "quantile",
    "mape",
    "gamma",
    "tweedie",
    "binary",
    "cross_entropy",
    "cross_entropy_lambda",
  ],
  metric: [
    "l1",
    "l2",
    "rmse",
    "quantile",
    "mape",
    "huber",
    "fair",
    "poisson",
    "gamma",
    "gamma_deviance",
    "tweedie",
    "auc",
    "average_precision",
    "binary_logloss",
    "binary_error",
    "cross_entropy",
    "cross_entropy_lambda",
    "kullback_leibler",
    "r2",
  ],
  data_sample_strategy: [
    "bagging",
    "goss",
  ],
};

const GBM_RUNNING_POLL_MS = 500;
const GBM_QUEUED_POLL_MS = 1000;
const GBM_MODEL_LIST_POLL_MS = 2000;
const GBM_EVALUATION_DOWNSAMPLE_THRESHOLD = 2000;
const GBM_EVALUATION_MAX_PLOT_POINTS = 1500;
const GBM_GRID_SAMPLE_DEFAULT = 25;

function gbmAutoModelTimeLabel(date = new Date()) {
  const hour = String(date.getHours()).padStart(2, "0");
  const minute = String(date.getMinutes()).padStart(2, "0");
  const second = String(date.getSeconds()).padStart(2, "0");
  return `${hour}:${minute}:${second}`;
}

export function gbmShapSelectionValue(data = {}) {
  const models = Array.isArray(data?.models) ? data.models : [];
  const activeModelId = String(data?.active_model_id || "");
  const model = models.find((item) => String(item?.model_id || "") === activeModelId) || models.find((item) => Boolean(item?.active));
  if (!model) return "100k";
  const shapRows = Number(model?.shap_rows);
  const scoredRows = Number(model?.scored_rows);
  if (!Number.isFinite(shapRows) || shapRows <= 0) return "0";
  if (Number.isFinite(scoredRows) && scoredRows > 0 && shapRows >= scoredRows) return "all";
  if (shapRows >= 100000) return "100k";
  if (shapRows >= 10000) return "10k";
  return "0";
}

export function gbmModelDetailLabel(model = {}) {
  const parts = [];
  if (model.metric) parts.push(model.metric);
  const bestIteration = Number(model.best_iteration || 0);
  if (Number.isFinite(bestIteration) && bestIteration > 0) parts.push(`iter ${bestIteration.toLocaleString()}`);
  parts.push(`train ${formatModelMetric(modelBestMetric(model, "training"))}`);
  parts.push(`test ${formatModelMetric(modelBestMetric(model, "test"))}`);
  return parts.join(" · ");
}

function modelBestMetric(model, name) {
  const metrics = model?.best_metrics && typeof model.best_metrics === "object" ? model.best_metrics : {};
  return modelNumberOrNull(metrics[name]);
}

function modelNumberOrNull(value) {
  if (value === null || value === undefined || String(value).trim() === "") return null;
  const number = Number(value);
  return Number.isFinite(number) ? number : null;
}

function formatTrainingBadgeCount(value) {
  const number = modelNumberOrNull(value);
  if (number === null || number <= 0) return "";
  return Math.trunc(number).toLocaleString();
}

export function gbmTrainingReadyBadgeLabel(progress = null) {
  const current = formatTrainingBadgeCount(progress?.grid_model_number);
  const total = formatTrainingBadgeCount(progress?.grid_model_count ?? progress?.grid?.trainable_count);
  if (current && total) return `Training GBM (${current}/${total})...`;
  return "Training GBM...";
}

function formatModelMetric(value) {
  const number = modelNumberOrNull(value);
  return number === null ? "--" : formatEvaluationValue(number) || "--";
}

function formatEvaluationValue(value) {
  const number = Number(value);
  if (!Number.isFinite(number)) return "";
  return number.toLocaleString(undefined, { maximumFractionDigits: 4 });
}

export function createGbmTool({
  api,
  clearToolCaches,
  el,
  escapeHtml,
  measureToolRender,
  renderExpectedNumerators,
  renderFeatures,
  saveToolPresentation,
  setChartMessage,
  setClientTiming,
  setDuckDbTiming,
  setGroupMeta,
  setRenderTiming,
  setStatus,
  setAppReadyStatus = () => {},
  setToolTimingFailed,
  setDatasetGbmCount = () => {},
  startToolTiming,
  state,
  canNavigateToLineBarFeature,
  navigateToLineBarFeature,
  syncClientTimingFromData,
  syncDuckDbTimingFromData,
  toolCache,
  updateAxisControls,
  refreshActiveTool,
  reloadSchema,
}) {
  const tool = "gbm";
  let tabulatorPromise = null;
  let featureTable = null;
  let parameterTable = null;
  let modelTable = null;
  let ebmGainSummaryTable = null;
  let ebmGainSummaryRows = null;
  let ebmGainSummaryModelId = "";
  let ebmGainSummaryRequestSeq = 0;
  let activeTab = "features";
  let config = null;
  let activeDetail = null;
  let pollTimer = null;
  let modelListRefreshSeq = 0;
  let modelListLastRefreshAt = 0;
  let isTraining = false;
  let liveProgress = null;
  let liveEvaluationParameters = null;
  let gridSampleValue = GBM_GRID_SAMPLE_DEFAULT;
  let gridTrainingNotice = "";
  let evaluationChart = null;
  let evaluationResizeObserver = null;
  let evaluationViewMode = "all";
  let featureMetricMode = "gain";
  let featureMetricModelId = "";
  let featureToolbarOutsideClickBound = false;
  let featureDraftState = null;
  const treeViewer = createGbmTreeViewer({ api, escapeHtml, loadTabulator, setGbmNotice });
  const shapTool = createGbmShapTool({ api, escapeHtml, setNotice: setGbmNotice });
  const stackedShapTool = createGbmStackedShapTool({ api, escapeHtml, setNotice: setGbmNotice });

  function buildRequest() {
    if (!state.schema) return null;
    return {
      tool,
      source: state.source || "dataset",
    };
  }

  async function fetchData(request, requestKey) {
    const requestSeq = state.gbmRequestSeq + 1;
    state.gbmRequestSeq = requestSeq;
    setStatus("");
    setChartMessage("");
    setGroupMeta(tool, "Loading GBM...");
    startToolTiming(tool);
    try {
      const data = await api("/api/gbm/config", { method: "GET", clientTiming: true });
      if (requestSeq !== state.gbmRequestSeq) return null;
      const cache = toolCache(tool);
      cache.requestKey = requestKey;
      cache.data = data;
      setDatasetGbmCount(Array.isArray(data?.models) ? data.models.length : null);
      syncDuckDbTimingFromData(tool, data);
      syncClientTimingFromData(tool, data);
      measureToolRender(tool, () => render(data));
      return data;
    } catch (error) {
      if (requestSeq !== state.gbmRequestSeq) return null;
      setToolTimingFailed(tool);
      setGroupMeta(tool, "GBM failed");
      setChartMessage("");
      setGbmNotice(error.message);
      return null;
    }
  }

  function useCached(cache) {
    measureToolRender(tool, () => {
      if (canReuseRenderedGbmShell(cache?.data)) {
        config = cache.data;
        syncFeatureMetricMode(cache.data);
        closeGbmFeatureContextMenu();
        syncSidebarModelChooser(config.models || [], config.active_model_id);
        applyPresentation();
        scheduleGbmTableRedraws();
        return;
      }
      render(cache.data);
      applyPresentation();
    });
  }

  function canReuseRenderedGbmShell(data) {
    const mount = el("modelToolWrap");
    return Boolean(
      data
      && data.tool === tool
      && config
      && mount?.querySelector(".gbm-tool")
      && featureMetricModelIdFromData(data) === featureMetricModelIdFromData(config)
    );
  }

  function scheduleGbmTableRedraws() {
    const redraw = () => {
      for (const table of [featureTable, parameterTable, modelTable, ebmGainSummaryTable]) {
        if (table && typeof table.redraw === "function") table.redraw(true);
      }
    };
    requestAnimationFrame(() => {
      redraw();
      requestAnimationFrame(redraw);
    });
  }

  function applyPresentation() {
    const presentation = toolCache(tool).presentation;
    if (!presentation) return;
    setGroupMeta(tool, presentation.groupMeta);
    setChartMessage(presentation.chartMessage);
  }

  function render(data = {}) {
    captureFeatureDraftStateForRender(data);
    config = applyFeatureDraftStateToData(data);
    data = config;
    syncFeatureMetricMode(data);
    const groupMeta = "";
    setGroupMeta(tool, groupMeta);
    setStatus("");
    setChartMessage("");
    const mount = el("modelToolWrap");
    if (!mount) return;
    closeGbmFeatureContextMenu();
    disposeEvaluationChart();
    treeViewer.dispose();
    shapTool.dispose();
    stackedShapTool.dispose();
    mount.innerHTML = `
      <div class="gbm-tool">
        <div id="gbmNotice" class="gbm-notice hidden" role="alert" aria-live="polite"></div>
        <div class="gbm-toolbar">
          <div class="gbm-tabs tabs workspace-tabs">
            <button class="tab ${activeTab === "features" ? "active" : ""}" type="button" data-gbm-tab="features">Features and parameters</button>
            <button class="tab ${activeTab === "models" ? "active" : ""}" type="button" data-gbm-tab="models">Model navigator</button>
            <button class="tab ${activeTab === "shap" ? "active" : ""}" type="button" data-gbm-tab="shap">SHAP</button>
            <button class="tab ${activeTab === "stacked-shap" ? "active" : ""}" type="button" data-gbm-tab="stacked-shap">Stacked SHAP</button>
            <button class="tab ${activeTab === "trees" ? "active" : ""}" type="button" data-gbm-tab="trees">Tree viewer</button>
          </div>
          <div id="gbmTrainingStatus" class="gbm-training-status ${liveProgress ? "" : "hidden"}" aria-live="polite">${trainingStatusHtml(liveProgress)}</div>
        </div>
        <div class="gbm-tab-panel ${activeTab === "features" ? "" : "hidden"}" data-gbm-panel="features">
          <div class="gbm-feature-layout">
            <section class="gbm-panel-section gbm-grid-panel">
              <div class="gbm-section-header gbm-feature-section-header">
                <h3 id="gbmFeatureSectionTitle" class="gbm-section-title">${escapeHtml(featureSectionTitle(data.features || []))}</h3>
                <div class="gbm-feature-actions" role="group" aria-label="Feature selection">
                  ${featureMetricToggleHtml(data.features || [], data)}
                  ${featureInteractionConstraintDropdownHtml(data.feature_interaction_groupings || [], data.active_feature_interaction_constraints || null, data.features || [])}
                  ${featureScenarioDropdownHtml(data.feature_scenarios || [], data.active_feature_scenario || null)}
                  <button id="gbmClearFeaturesBtn" class="tab gbm-inline-action-button gbm-icon-action-button" type="button" aria-label="Clear all features" title="Clear all">×</button>
                  <button id="gbmSelectFeaturesBtn" class="tab gbm-inline-action-button gbm-icon-action-button" type="button" aria-label="Select all features" title="Select all">✓</button>
                </div>
              </div>
              <div id="gbmFeatureGrid" class="gbm-grid"></div>
              <div id="gbmFeatureFallback" class="gbm-fallback-table"></div>
              <div id="gbmEbmGainSummaryGrid" class="gbm-grid gbm-ebm-gain-summary-grid hidden"></div>
              <div id="gbmEbmGainSummaryFallback" class="gbm-fallback-table hidden"></div>
            </section>
            <section class="gbm-right-panel">
              <section class="gbm-panel-section gbm-parameter-section">
                <div class="gbm-parameter-layout">
                  <div class="gbm-parameter-table-column">
                    <h3 class="gbm-section-title">Parameters</h3>
                    <div id="gbmParameterGrid" class="gbm-grid gbm-parameter-grid"></div>
                    <div id="gbmParameterFallback" class="gbm-fallback-table"></div>
                  </div>
                  <div class="gbm-parameter-controls-column">
                    <h3 class="gbm-section-title">Control</h3>
                    <div class="gbm-actions">
                      <button id="gbmTrainBtn" class="tab gbm-action-button gbm-train-button ${isTraining ? "training" : ""}" type="button" ${isTraining ? "disabled aria-busy=\"true\"" : ""}>${isTraining ? "Training..." : "Train GBM"}</button>
                      ${sampleStatusHtml(data.sample)}
                      <div id="gbmShapRows" class="gbm-shap-rows" role="radiogroup" aria-label="SHAP rows">
                        <span class="gbm-shap-label">SHAP rows</span>
                        <div class="gbm-shap-options">
                          ${shapOptionsHtml(data.shap_options || [], gbmShapSelectionValue(data))}
                        </div>
                      </div>
                      ${gridSampleHtml(data.parameters || [])}
                      ${data.ebm_available ? trainingModeHtml(data.training_mode) : ""}
                      ${shouldShowCreateSampleButton(data.sample) ? '<button id="gbmCreateSampleBtn" class="tab gbm-action-button gbm-sample-button" type="button">Create sample column</button>' : ""}
                    </div>
                  </div>
                </div>
              </section>
              <section class="gbm-panel-section">
                <div class="gbm-section-header gbm-evaluation-section-header">
                  <h3 class="gbm-section-title">Evaluation Log</h3>
                  ${evaluationViewModeHtml()}
                </div>
                <div id="gbmEvaluationChart" class="gbm-evaluation-chart"></div>
              </section>
            </section>
          </div>
        </div>
        <div class="gbm-tab-panel ${activeTab === "models" ? "" : "hidden"}" data-gbm-panel="models">
          <div class="gbm-model-navigator">
            <div class="gbm-model-actions" role="group" aria-label="GBM model actions">
              <button id="gbmRenameModelBtn" class="tab gbm-inline-action-button" type="button">Rename</button>
              <button id="gbmActivateModelBtn" class="tab gbm-inline-action-button" type="button">Activate</button>
              <button id="gbmDeleteModelBtn" class="danger-action gbm-model-delete-button" type="button">Delete</button>
            </div>
            <div id="gbmModelGrid" class="gbm-grid gbm-model-grid"></div>
            <div id="gbmModelFallback" class="gbm-fallback-table"></div>
          </div>
        </div>
        <div class="gbm-tab-panel ${activeTab === "trees" ? "" : "hidden"}" data-gbm-panel="trees">
          <div id="gbmTreeViewer" class="gbm-tree-viewer">
            <section class="gbm-panel-section gbm-tree-summary-panel">
              <div class="gbm-tree-section-header">
                <h3 class="gbm-section-title">Select tree</h3>
                <input id="gbmTreeSearch" class="gbm-tree-search" type="search" placeholder="Search" aria-label="Search trees" />
              </div>
              <div id="gbmTreeSummaryGrid" class="gbm-grid gbm-tree-summary-grid"></div>
              <div id="gbmTreeSummaryFallback" class="gbm-fallback-table"></div>
            </section>
            <div id="gbmTreeResizer" class="gbm-tree-resizer" role="separator" aria-orientation="vertical" aria-label="Resize tree selector"></div>
            <section class="gbm-panel-section gbm-tree-diagram-panel">
              <div class="gbm-tree-diagram-header">
                <h3 class="gbm-section-title">Tree viewer</h3>
                <div class="gbm-tree-controls">
                  <div class="gbm-tree-zoom segmented" role="group" aria-label="Tree zoom">
                    <button type="button" data-gbm-tree-zoom="out" aria-label="Zoom out">-</button>
                    <button type="button" data-gbm-tree-zoom="reset" aria-label="Reset zoom">Reset</button>
                    <button type="button" data-gbm-tree-zoom="in" aria-label="Zoom in">+</button>
                  </div>
                  <div class="gbm-tree-palette segmented" role="group" aria-label="Tree colour mode">
                    <button type="button" data-gbm-tree-palette="plain" aria-pressed="true">Plain</button>
                    <button type="button" data-gbm-tree-palette="divergent" aria-pressed="false">Divergent</button>
                    <button type="button" data-gbm-tree-palette="spectral" aria-pressed="false">Spectral</button>
                    <button type="button" data-gbm-tree-palette="viridis" aria-pressed="false">Viridis</button>
                  </div>
                </div>
              </div>
              <div id="gbmTreeChart" class="gbm-tree-chart" aria-label="GBM tree diagram">
                <div id="gbmTreeDetailSummary" class="gbm-tree-detail-summary">
                  <h3 class="gbm-section-title">Tree viewer</h3>
                </div>
                <div id="gbmTreeSvgMount" class="gbm-tree-svg-mount"></div>
              </div>
            </section>
          </div>
        </div>
        <div class="gbm-tab-panel ${activeTab === "shap" ? "" : "hidden"}" data-gbm-panel="shap">
          ${shapTool.shellHtml()}
        </div>
        <div class="gbm-tab-panel ${activeTab === "stacked-shap" ? "" : "hidden"}" data-gbm-panel="stacked-shap">
          ${stackedShapTool.shellHtml()}
        </div>
      </div>
    `;
    bindTabs(mount);
    bindModelActions();
    syncSidebarModelChooser(data.models || [], data.active_model_id);
    bindFeatureActions();
    bindEvaluationViewModeActions();
    renderTables(data);
    if (data.active_model_id) loadModelDetail(data.active_model_id);
    if (activeTab === "trees") treeViewer.render(data.active_model_id || "");
    if (activeTab === "shap") shapTool.render(data.active_model_id || "");
    if (activeTab === "stacked-shap") stackedShapTool.render(data.active_model_id || "");
    if (liveProgress) renderLiveProgress(liveProgress);
    setDuckDbTiming(tool, data.timings || {});
    setClientTiming(tool, data.client_timings || {});
    setRenderTiming(tool, 0);
    saveToolPresentation(tool, { groupMeta, chartMessage: "" });
  }

  function shapOptionsHtml(options, selectedValue = "100k") {
    const rows = options.length ? options : [
      { value: "0", label: "0" },
      { value: "10k", label: "10k" },
      { value: "100k", label: "100k" },
      { value: "all", label: "All" },
    ];
    const selected = String(selectedValue || "100k").trim().toLowerCase();
    const selectedRowValue = rows.some((row) => String(row.value || "").trim().toLowerCase() === selected) ? selected : "100k";
    return rows.map((row) => {
      const value = String(row.value || "");
      const checked = value.trim().toLowerCase() === selectedRowValue ? "checked" : "";
      return `
      <label class="gbm-shap-option">
        <input type="radio" name="gbmShapRows" value="${escapeHtml(value)}" ${checked} />
        <span>${escapeHtml(row.label)}</span>
      </label>
    `;
    }).join("");
  }

  function trainingModeHtml(mode) {
    const selected = normaliseTrainingMode(mode);
    const ebmTitle = "EBM starts with 2-leaf trees at learning rate 0.3, then uses the configured learning rate for 3+ leaves.";
    return `
      <div id="gbmTrainingMode" class="gbm-shap-rows gbm-mode-rows" role="radiogroup" aria-label="Training mode" title="${escapeHtml(ebmTitle)}">
        <span class="gbm-shap-label gbm-mode-label">Training mode</span>
        <div class="gbm-shap-options gbm-mode-options">
          <label class="gbm-shap-option gbm-mode-option">
            <input type="radio" name="gbmTrainingMode" value="normal" ${selected === "normal" ? "checked" : ""} />
            <span>Normal</span>
          </label>
          <label class="gbm-shap-option gbm-mode-option" title="${escapeHtml(ebmTitle)}">
            <input type="radio" name="gbmTrainingMode" value="ebm" ${selected === "ebm" ? "checked" : ""} />
            <span>EBM</span>
          </label>
        </div>
      </div>
    `;
  }

  function gridSampleHtml(parameters = []) {
    const hidden = hasGridParameters(parameters) ? "" : " hidden";
    return `
      <label id="gbmGridSamples" class="gbm-grid-samples${hidden}">
        <span class="gbm-shap-label">Grid samples</span>
        <input id="gbmGridSampleInput" class="gbm-grid-sample-input" type="number" min="1" step="1" value="${escapeHtml(String(currentGridSampleValue()))}" aria-label="Grid samples" />
      </label>
    `;
  }

  function trainingStatusHtml(progress) {
    if (!progress) return "";
    return trainingStatusContentHtml(progress.message || "", trainingStatusDetail(progress));
  }

  function trainingStatusContentHtml(message, detail = "") {
    const main = String(message || "");
    const sub = String(detail || "");
    if (!main && !sub) return "";
    return `
      <span class="gbm-training-status-main">${escapeHtml(main)}</span>
      ${sub ? `<span class="gbm-training-status-detail">${escapeHtml(sub)}</span>` : ""}
    `;
  }

  function evaluationViewModeHtml() {
    const selected = normaliseEvaluationViewMode(evaluationViewMode);
    return `
      <div id="gbmEvaluationViewMode" class="gbm-evaluation-view-mode" role="radiogroup" aria-label="Evaluation Log view">
        <label class="gbm-evaluation-view-option">
          <input type="radio" name="gbmEvaluationViewMode" value="all" ${selected === "all" ? "checked" : ""} />
          <span>All</span>
        </label>
        <label class="gbm-evaluation-view-option">
          <input type="radio" name="gbmEvaluationViewMode" value="tail" ${selected === "tail" ? "checked" : ""} />
          <span>Tail</span>
        </label>
      </div>
    `;
  }

  function normaliseEvaluationViewMode(value) {
    return String(value || "").trim().toLowerCase() === "tail" ? "tail" : "all";
  }

  function shouldShowCreateSampleButton(sample) {
    return !sample?.has_dataset_sample && !sample?.has_generated_sample;
  }

  function sampleStatusHtml(sample) {
    const info = normaliseSampleInfo(sample);
    const levels = sampleLevelRowsHtml(info.levels);
    if (info.source === "dataset") {
      return `
        <div id="gbmSampleStatus" class="gbm-sample-status gbm-sample-status-ok" role="status">
          <span class="gbm-sample-status-title">SAMPLE column found</span>
          <span class="gbm-sample-status-levels">${levels}</span>
        </div>
      `;
    }
    if (info.source === "generated") {
      return `
        <div id="gbmSampleStatus" class="gbm-sample-status gbm-sample-status-warning" role="status">
          <span class="gbm-sample-status-title">Generated SAMPLE</span>
          <span class="gbm-sample-status-levels">${levels}</span>
          <span class="gbm-sample-status-warning-text">${escapeHtml(info.warning)}</span>
        </div>
      `;
    }
    return `
      <div id="gbmSampleStatus" class="gbm-sample-status gbm-sample-status-missing" role="status">
        <span class="gbm-sample-status-title">No SAMPLE column</span>
        <span class="gbm-sample-status-detail">${escapeHtml(info.warning)}</span>
      </div>
    `;
  }

  function normaliseSampleInfo(sample) {
    return {
      column: sample?.column || "",
      source: sample?.source || "none",
      levels: Array.isArray(sample?.levels) ? sample.levels : [],
      has_dataset_sample: Boolean(sample?.has_dataset_sample),
      has_generated_sample: Boolean(sample?.has_generated_sample),
      warning: String(sample?.warning || ""),
    };
  }

  function sampleLevelRowsHtml(levels) {
    const byName = new Map((levels || []).map((level) => [String(level.name || "").toLowerCase(), level]));
    return ["training", "test", "validation"].map((name) => {
      const level = byName.get(name) || {};
      const percent = Number(level.percent || 0);
      const count = Number(level.row_count || 0);
      return `<span class="gbm-sample-status-detail">${escapeHtml(name)} ${escapeHtml(formatSamplePercent(percent))} (${escapeHtml(Number.isFinite(count) ? Math.round(count).toLocaleString() : "0")})</span>`;
    }).join("");
  }

  function formatSamplePercent(value) {
    const number = Number(value);
    if (!Number.isFinite(number)) return "0%";
    return `${number.toLocaleString(undefined, { maximumFractionDigits: 1 })}%`;
  }

  function bindTabs(mount) {
    for (const button of mount.querySelectorAll("[data-gbm-tab]")) {
      button.addEventListener("click", () => {
        closeGbmFeatureContextMenu();
        const nextTab = button.dataset.gbmTab;
        activeTab = nextTab;
        render(config || {});
        if (nextTab === "models") refreshModelList({ force: true });
      });
    }
  }

  function syncRenderedTab(mount, nextTab) {
    for (const button of mount.querySelectorAll("[data-gbm-tab]")) {
      button.classList.toggle("active", button.dataset.gbmTab === nextTab);
    }
    for (const panel of mount.querySelectorAll("[data-gbm-panel]")) {
      panel.classList.toggle("hidden", panel.dataset.gbmPanel !== nextTab);
    }
  }

  function openModelNavigator() {
    closeGbmFeatureContextMenu();
    activeTab = "models";
    const mount = el("modelToolWrap");
    if (mount?.querySelector(".gbm-tool")) {
      syncRenderedTab(mount, activeTab);
      scheduleGbmTableRedraws();
      refreshModelList({ force: true });
      return;
    }
    const cachedData = config || toolCache(tool)?.data || null;
    if (cachedData) render(cachedData);
    refreshModelList({ force: true });
  }

  function featureDraftModelId(data = config) {
    return featureMetricModelIdFromData(data || {}) || "__new_gbm__";
  }

  function captureFeatureDraftStateForRender(nextData = config) {
    const mount = el("modelToolWrap");
    if (!mount?.querySelector(".gbm-tool") || !config) {
      if (featureDraftState && featureDraftState.modelId !== featureDraftModelId(nextData)) featureDraftState = null;
      return;
    }
    const currentModelId = featureDraftModelId(config);
    const nextModelId = featureDraftModelId(nextData);
    if (currentModelId !== nextModelId) {
      featureDraftState = null;
      return;
    }
    const rows = currentFeatureRows();
    if (!rows.length) return;
    const interactionGroupings = currentFeatureInteractionGroupings();
    const scenarioName = el("gbmFeatureScenarioDropdown")?.dataset.gbmSelectedFeatureScenario || "";
    featureDraftState = {
      modelId: currentModelId,
      features: rows.map((feature) => ({
        name: feature.name,
        include: Boolean(feature.include),
        monotonicity: feature.monotonicity || "",
        feature_interaction_locked: Boolean(feature.feature_interaction_locked),
      })),
      interactionGroupings,
      interactionGroupingsEdited: featureInteractionGroupingsEdited(interactionGroupings, config),
      scenarioName,
      scenarioEdited: featureScenarioSelectionEdited(scenarioName, config),
    };
  }

  function featureDraftForData(data = config) {
    return featureDraftState && featureDraftState.modelId === featureDraftModelId(data) ? featureDraftState : null;
  }

  function applyFeatureDraftStateToData(data = {}) {
    const draft = featureDraftForData(data);
    if (!draft) return data;
    const draftFeatures = new Map(draft.features.map((feature) => [feature.name, feature]));
    return {
      ...data,
      features: (data.features || []).map((feature) => {
        const draftFeature = draftFeatures.get(feature.name);
        return draftFeature
          ? {
              ...feature,
              include: draftFeature.include,
              monotonicity: draftFeature.monotonicity,
              feature_interaction_locked: draftFeature.feature_interaction_locked,
            }
          : feature;
      }),
    };
  }

  function sameStringSet(leftValues = [], rightValues = []) {
    const left = new Set(leftValues.map((value) => String(value || "").trim()).filter(Boolean));
    const right = new Set(rightValues.map((value) => String(value || "").trim()).filter(Boolean));
    if (left.size !== right.size) return false;
    for (const value of left) {
      if (!right.has(value)) return false;
    }
    return true;
  }

  function bindFeatureActions() {
    bindFeatureToolbarOutsideClicks();
    bindFeatureMetricActions();
    bindFeatureInteractionActions();
    bindFeatureScenarioActions();
    el("gbmClearFeaturesBtn")?.addEventListener("click", () => setFeatureIncludes(false));
    el("gbmSelectFeaturesBtn")?.addEventListener("click", () => setFeatureIncludes(true));
    el("gbmCreateSampleBtn")?.addEventListener("click", createSampleColumn);
    el("gbmTrainBtn")?.addEventListener("click", train);
    bindGridSampleInput();
    syncTrainingButton();
  }

  function syncFeatureMetricMode(data = {}) {
    const features = data.features || [];
    const modelId = featureMetricModelIdFromData(data);
    if (modelId !== featureMetricModelId) {
      const previousModelId = featureMetricModelId;
      featureMetricModelId = modelId;
      ebmGainSummaryModelId = "";
      ebmGainSummaryRows = null;
      const modes = featureMetricModes(features, data);
      featureMetricMode = previousModelId && modes.includes(featureMetricMode)
        ? featureMetricMode
        : defaultFeatureMetricMode(modes);
      return;
    }
    featureMetricMode = normaliseFeatureMetricMode(featureMetricMode, features, data);
  }

  function featureMetricModelIdFromData(data = {}) {
    const activeModelId = String(data.active_model_id || "");
    if (activeModelId) return activeModelId;
    const activeModel = (data.models || []).find((model) => model.active);
    return String(activeModel?.model_id || "");
  }

  function activeModelFromData(data = config) {
    const models = Array.isArray(data?.models) ? data.models : [];
    const activeModelId = String(data?.active_model_id || "");
    return models.find((model) => String(model?.model_id || "") === activeModelId) || models.find((model) => Boolean(model?.active)) || null;
  }

  function activeModelUsesEbm(data = config) {
    const model = activeModelFromData(data);
    return String(model?.training_mode || data?.training_mode || "").trim().toLowerCase() === "ebm" && Boolean(featureMetricModelIdFromData(data || {}));
  }

  function featureMetricModes(features = [], data = config) {
    const rows = Array.isArray(features) ? features : [];
    const hasGain = rows.some((feature) => featureNumber(feature?.gain) !== null);
    const hasShap = rows.some((feature) => featureMeanAbsShap(feature) !== null);
    const modes = [];
    if (hasGain && activeModelUsesEbm(data)) modes.push("gain_ebm");
    if (hasGain) modes.push("gain");
    if (hasGain && hasShap) modes.push("shap");
    return modes;
  }

  function defaultFeatureMetricMode(modes = []) {
    if (modes.includes("shap")) return "shap";
    if (modes.includes("gain")) return "gain";
    return modes[0] || "gain";
  }

  function featureMetricToggleAvailable(features = [], data = config) {
    return featureMetricModes(features, data).length > 1;
  }

  function normaliseFeatureMetricMode(value, features = config?.features || [], data = config) {
    const mode = String(value || "").toLowerCase();
    const available = featureMetricModes(features, data);
    return available.includes(mode) ? mode : defaultFeatureMetricMode(available);
  }

  function featureMetricToggleHtml(features = [], data = config) {
    const modes = featureMetricModes(features, data);
    if (modes.length < 2) return "";
    const selected = normaliseFeatureMetricMode(featureMetricMode, features, data);
    return `
      <div id="gbmFeatureMetricToggle" class="gbm-feature-metric-toggle" role="radiogroup" aria-label="Feature table metric">
        ${modes.map((mode) => `
          <label class="gbm-feature-metric-option${selected === mode ? " active" : ""}">
            <input type="radio" name="gbmFeatureMetric" value="${escapeHtml(mode)}" ${selected === mode ? "checked" : ""} />
            <span>${escapeHtml(featureMetricModeLabel(mode))}</span>
          </label>
        `).join("")}
      </div>
    `;
  }

  function featureMetricModeLabel(mode) {
    if (mode === "shap") return "SHAP";
    if (mode === "gain_ebm") return "EBM Gain";
    return "Gain";
  }

  function bindFeatureMetricActions() {
    for (const input of document.querySelectorAll("input[name='gbmFeatureMetric']")) {
      input.addEventListener("change", () => {
        if (!input.checked) return;
        setFeatureMetricMode(input.value);
      });
    }
  }

  function setFeatureMetricMode(mode) {
    const features = currentFeatureRows();
    featureMetricMode = normaliseFeatureMetricMode(mode, features);
    syncFeatureMetricToggle();
    updateFeatureMetricView(features);
  }

  function syncFeatureMetricToggle() {
    for (const input of document.querySelectorAll("input[name='gbmFeatureMetric']")) {
      input.checked = input.value === featureMetricMode;
      input.closest(".gbm-feature-metric-option")?.classList.toggle("active", input.checked);
    }
  }

  function updateFeatureMetricView(features = currentFeatureRows(), options = {}) {
    const useEbmGainSummary = featureMetricMode === "gain_ebm";
    el("gbmFeatureGrid")?.classList.toggle("hidden", useEbmGainSummary);
    el("gbmFeatureFallback")?.classList.toggle("hidden", useEbmGainSummary);
    el("gbmEbmGainSummaryGrid")?.classList.toggle("hidden", !useEbmGainSummary);
    el("gbmEbmGainSummaryFallback")?.classList.toggle("hidden", !useEbmGainSummary);
    if (useEbmGainSummary) {
      loadEbmGainSummary(features);
      return;
    }
    if (featureTable && typeof featureTable.setColumns === "function") {
      if (options.refreshColumns !== false) {
        featureTable.setColumns(featureTableColumns());
        featureTable.setSort(featureTableInitialSort());
      }
      return;
    }
    renderFeatureFallback(features);
  }

  async function loadEbmGainSummary(features = currentFeatureRows()) {
    const modelId = currentActiveModelId();
    const modeFeatures = Array.isArray(features) && features.length ? features : (config?.features || []);
    if (!modelId || !featureMetricModes(modeFeatures).includes("gain_ebm")) {
      renderEbmGainSummaryRows([]);
      return;
    }
    if (ebmGainSummaryModelId === modelId && Array.isArray(ebmGainSummaryRows)) {
      renderEbmGainSummaryRows(ebmGainSummaryRows);
      return;
    }
    const requestSeq = ebmGainSummaryRequestSeq + 1;
    ebmGainSummaryRequestSeq = requestSeq;
    ebmGainSummaryModelId = modelId;
    ebmGainSummaryRows = null;
    renderEbmGainSummaryLoading();
    try {
      const payload = await api(`/api/gbm/models/${encodeURIComponent(modelId)}/ebm-gain-summary`, { method: "GET" });
      if (requestSeq !== ebmGainSummaryRequestSeq || modelId !== currentActiveModelId()) return;
      ebmGainSummaryRows = Array.isArray(payload?.rows) ? payload.rows : [];
      renderEbmGainSummaryRows(ebmGainSummaryRows);
    } catch (error) {
      if (requestSeq !== ebmGainSummaryRequestSeq || modelId !== currentActiveModelId()) return;
      ebmGainSummaryRows = [];
      renderEbmGainSummaryRows([]);
      setGbmNotice(error.message);
    }
  }

  function renderEbmGainSummaryLoading() {
    if (ebmGainSummaryTable && typeof ebmGainSummaryTable.setData === "function") {
      setEbmGainSummaryTableRows([]);
      return;
    }
    renderEbmGainSummaryFallback(null);
  }

  function renderEbmGainSummaryRows(rows) {
    const summaryRows = Array.isArray(rows) ? rows : [];
    if (ebmGainSummaryTable && typeof ebmGainSummaryTable.setData === "function") {
      setEbmGainSummaryTableRows(summaryRows);
      return;
    }
    renderEbmGainSummaryFallback(summaryRows);
  }

  function cachedEbmGainSummaryRowsForActiveModel() {
    return ebmGainSummaryModelId === currentActiveModelId() && Array.isArray(ebmGainSummaryRows)
      ? ebmGainSummaryRows
      : [];
  }

  function setEbmGainSummaryTableRows(rows) {
    const table = ebmGainSummaryTable;
    if (!table || typeof table.setData !== "function") return;
    const applyRows = () => {
      if (table !== ebmGainSummaryTable) return;
      table.setData(rows);
    };
    if (table.initialized) {
      applyRows();
      return;
    }
    if (typeof table.on === "function") table.on("tableBuilt", applyRows);
    window.setTimeout(applyRows, 0);
  }

  function featureInteractionConstraintDropdownHtml(groupings, activeConstraints = null, features = []) {
    const rows = featureInteractionGroupingRows(groupings);
    const active = normaliseActiveFeatureInteractionConstraints(activeConstraints);
    const draft = featureDraftForData(config);
    const draftGroupings = draft?.interactionGroupingsEdited ? new Set(draft.interactionGroupings || []) : null;
    const currentNames = new Set(rows.map((row) => row.name));
    const selectedCurrent = draftGroupings
      ? new Set(rows.filter((row) => draftGroupings.has(row.name)).map((row) => row.name))
      : new Set(
          active.groups
            .filter((group) => group.status === "current" && currentNames.has(group.grouping))
            .map((group) => group.grouping)
        );
    const synthetic = draftGroupings
      ? []
      : active.groups.filter((group) => group.status !== "current" || !currentNames.has(group.grouping));
    const counts = selectedFeatureCountsByGrouping(features);
    const hasOptions = rows.length || synthetic.length;
    const hidden = hasOptions ? "" : " hidden";
    const disabled = hasOptions ? "" : " disabled";
    const constraintClass = selectedCurrent.size + synthetic.length > 0 ? " has-constraints" : "";
    return `
      <div id="gbmFeatureInteractionConstraintSelect" class="gbm-interaction-constraint-select${hidden}" data-gbm-feature-menu-root>
        <button id="gbmFeatureInteractionConstraintButton" class="gbm-feature-menu-button gbm-interaction-constraint-button${constraintClass}" type="button" aria-haspopup="true" aria-expanded="false" aria-label="Interaction constraints" title="Constrain selected grouped features so they only interact within selected groups" data-gbm-feature-menu-button${disabled}>${escapeHtml(featureInteractionButtonLabel(selectedCurrent.size, synthetic.length))}</button>
        <div id="gbmFeatureInteractionConstraintMenu" class="gbm-feature-menu gbm-interaction-constraint-menu hidden" role="menu" data-gbm-feature-menu>
          ${synthetic.map((group) => `
            <label class="gbm-interaction-constraint-row gbm-interaction-constraint-row-trained" data-gbm-trained-interaction-row="${escapeHtml(group.grouping)}">
              <input type="checkbox" checked disabled />
              <span>${escapeHtml(trainedFeatureInteractionLabel(group))}</span>
            </label>
          `).join("")}
          ${rows.map((row) => `
            <label class="gbm-interaction-constraint-row">
              <input type="checkbox" value="${escapeHtml(row.name)}" data-gbm-interaction-grouping="${escapeHtml(row.name)}" ${selectedCurrent.has(row.name) ? "checked" : ""} />
              <span data-gbm-interaction-count-label="${escapeHtml(row.name)}">${escapeHtml(featureInteractionGroupingLabel(row.name, counts))}</span>
            </label>
          `).join("")}
        </div>
      </div>
    `;
  }

  function bindFeatureInteractionActions() {
    const root = el("gbmFeatureInteractionConstraintSelect");
    if (!root) return;
    bindGbmFeatureToolbarMenu(root, {
      beforeOpen: () => syncFeatureInteractionCounts(currentFeatureRows()),
    });
    for (const checkbox of root.querySelectorAll("[data-gbm-interaction-grouping]")) {
      checkbox.addEventListener("change", () => {
        clearTrainedInteractionConstraintRows();
        syncFeatureInteractionControls();
      });
    }
  }

  function featureInteractionGroupingRows(groupings) {
    if (!Array.isArray(groupings)) return [];
    const seen = new Set();
    const rows = [];
    for (const item of groupings) {
      const name = String(item || "").trim();
      const key = name.toLowerCase();
      if (!name || seen.has(key)) continue;
      rows.push({ name });
      seen.add(key);
    }
    return rows.sort((left, right) => left.name.localeCompare(right.name));
  }

  function normaliseActiveFeatureInteractionConstraints(activeConstraints) {
    if (!activeConstraints || typeof activeConstraints !== "object") return { groups: [], features: [] };
    const groups = Array.isArray(activeConstraints.groups) ? activeConstraints.groups : [];
    return {
      features: scenarioFeatureList(activeConstraints.features),
      groups: groups
        .map((group) => {
          const grouping = String(group?.grouping || "").trim();
          const status = String(group?.status || "current").trim().toLowerCase();
          const features = Array.isArray(group?.features) ? group.features.map((feature) => String(feature || "").trim()).filter(Boolean) : [];
          return {
            grouping,
            status: ["current", "stale", "missing"].includes(status) ? status : "current",
            features,
          };
        })
        .filter((group) => group.grouping && group.features.length),
    };
  }

  function activeCurrentFeatureInteractionGroupings(data = config) {
    const currentNames = new Set(featureInteractionGroupingRows(data?.feature_interaction_groupings || []).map((row) => row.name));
    return normaliseActiveFeatureInteractionConstraints(data?.active_feature_interaction_constraints)
      .groups
      .filter((group) => group.status === "current" && currentNames.has(group.grouping))
      .map((group) => group.grouping);
  }

  function hasSyntheticActiveFeatureInteractionConstraints(data = config) {
    const currentNames = new Set(featureInteractionGroupingRows(data?.feature_interaction_groupings || []).map((row) => row.name));
    return normaliseActiveFeatureInteractionConstraints(data?.active_feature_interaction_constraints)
      .groups
      .some((group) => group.status !== "current" || !currentNames.has(group.grouping));
  }

  function featureInteractionGroupingsEdited(groupings, data = config) {
    if (!sameStringSet(groupings, activeCurrentFeatureInteractionGroupings(data))) return true;
    return hasSyntheticActiveFeatureInteractionConstraints(data) && !document.querySelector("[data-gbm-trained-interaction-row]");
  }

  function trainedFeatureInteractionLabel(group) {
    if (group.status === "stale") return `${group.grouping} (trained; spec changed)`;
    if (group.status === "missing") return `${group.grouping} (trained; missing from spec)`;
    return group.grouping;
  }

  function featureInteractionButtonLabel(selectedCurrentCount, syntheticCount = 0) {
    if (selectedCurrentCount > 0) return `Constraints (${selectedCurrentCount})`;
    if (syntheticCount > 0) return `Trained constraints (${syntheticCount})`;
    return "Constraints";
  }

  function featureInteractionGroupingLabel(grouping, counts) {
    return `${grouping} (${Number(counts.get(grouping) || 0).toLocaleString()})`;
  }

  function selectedFeatureCountsByGrouping(features) {
    const counts = new Map();
    for (const feature of features || []) {
      const grouping = String(feature?.grouping || "").trim();
      if (!grouping || !feature?.include || !isFeatureSelectable(feature)) continue;
      counts.set(grouping, (counts.get(grouping) || 0) + 1);
    }
    return counts;
  }

  function currentFeatureInteractionGroupings() {
    return [...document.querySelectorAll("[data-gbm-interaction-grouping]:checked")]
      .map((checkbox) => String(checkbox.value || "").trim())
      .filter(Boolean);
  }

  function currentFeatureInteractionGroupingsPayload() {
    const valid = new Set(featureInteractionGroupingRows(config?.feature_interaction_groupings || []).map((row) => row.name));
    const groupings = currentFeatureInteractionGroupings().filter((grouping) => valid.has(grouping));
    return groupings.length ? groupings : null;
  }

  function currentFeatureInteractionFeaturesPayload() {
    const features = currentFeatureRows()
      .filter((feature) => feature?.feature_interaction_locked && feature?.include && isFeatureSelectable(feature))
      .map((feature) => String(feature.name || "").trim())
      .filter(Boolean);
    return features.length ? [...new Set(features)] : null;
  }

  function clearTrainedInteractionConstraintRows() {
    for (const row of document.querySelectorAll("[data-gbm-trained-interaction-row]")) {
      row.remove();
    }
  }

  function syncFeatureInteractionControls() {
    const root = el("gbmFeatureInteractionConstraintSelect");
    if (!root) return;
    const features = currentFeatureRows();
    syncFeatureInteractionCounts(features);
    syncFeatureInteractionLocks(features);
  }

  function syncFeatureInteractionCounts(features = currentFeatureRows()) {
    const root = el("gbmFeatureInteractionConstraintSelect");
    if (!root) return;
    const counts = selectedFeatureCountsByGrouping(features);
    for (const label of root.querySelectorAll("[data-gbm-interaction-count-label]")) {
      const grouping = label.getAttribute("data-gbm-interaction-count-label") || "";
      label.textContent = featureInteractionGroupingLabel(grouping, counts);
    }
    const selectedCurrentCount = currentFeatureInteractionGroupings().length;
    const syntheticCount = root.querySelectorAll("[data-gbm-trained-interaction-row]").length;
    const button = el("gbmFeatureInteractionConstraintButton");
    if (button) {
      button.textContent = featureInteractionButtonLabel(selectedCurrentCount, syntheticCount);
      button.classList.toggle("has-constraints", selectedCurrentCount + syntheticCount > 0);
    }
  }

  function selectedInteractionFeatureNames(features) {
    const selectedGroupings = new Set(currentFeatureInteractionGroupings());
    const trainedFeatures = trainedGroupInteractionFeatureNames();
    const locked = new Set();
    for (const feature of features || []) {
      if (!feature?.include || !isFeatureSelectable(feature)) continue;
      const grouping = String(feature.grouping || "").trim();
      if ((grouping && selectedGroupings.has(grouping)) || trainedFeatures.has(feature.name)) {
        locked.add(feature.name);
      }
    }
    return locked;
  }

  function renderedInteractionFeatureNames(features) {
    const draft = featureDraftForData(config);
    const selectedGroupings = new Set(
      draft?.interactionGroupingsEdited
        ? draft.interactionGroupings || []
        : activeCurrentFeatureInteractionGroupings(config)
    );
    const trainedFeatures = new Set();
    if (!draft?.interactionGroupingsEdited) {
      for (const group of normaliseActiveFeatureInteractionConstraints(config?.active_feature_interaction_constraints).groups) {
        if (group.status !== "current") {
          for (const feature of group.features) trainedFeatures.add(feature);
        }
      }
    }
    const locked = new Set();
    for (const feature of features || []) {
      if (!feature?.include || !isFeatureSelectable(feature)) continue;
      const grouping = String(feature.grouping || "").trim();
      if ((grouping && selectedGroupings.has(grouping)) || trainedFeatures.has(feature.name)) {
        locked.add(feature.name);
      }
    }
    return locked;
  }

  function trainedGroupInteractionFeatureNames() {
    if (!document.querySelector("[data-gbm-trained-interaction-row]")) return new Set();
    const active = normaliseActiveFeatureInteractionConstraints(config?.active_feature_interaction_constraints);
    const names = new Set();
    for (const group of active.groups) {
      for (const feature of group.features) names.add(feature);
    }
    return names;
  }

  function activeFeatureInteractionFeatureNames() {
    return new Set(normaliseActiveFeatureInteractionConstraints(config?.active_feature_interaction_constraints).features);
  }

  function selectedFeatureInteractionFeatureNames(features = currentFeatureRows()) {
    const locked = new Set();
    for (const feature of features || []) {
      if (feature?.feature_interaction_locked) locked.add(feature.name);
    }
    return locked;
  }

  function applyInteractionLocksToFeatures(features) {
    const draft = featureDraftForData(config);
    const groupLocked = renderedInteractionFeatureNames(features);
    const featureLocked = draft ? selectedFeatureInteractionFeatureNames(features) : activeFeatureInteractionFeatureNames();
    return (features || []).map((feature) => ({
      ...feature,
      interaction_locked: groupLocked.has(feature.name),
      feature_interaction_locked: featureLocked.has(feature.name),
    }));
  }

  function syncFeatureInteractionLocks(features = currentFeatureRows()) {
    const groupLocked = selectedInteractionFeatureNames(features);
    const featureLocked = selectedFeatureInteractionFeatureNames(features);
    if (featureTable) {
      for (const row of featureTable.getRows()) {
        const data = row.getData();
        const interactionLocked = groupLocked.has(data.name);
        const featureInteractionLocked = featureLocked.has(data.name);
        row.update({ interaction_locked: interactionLocked, feature_interaction_locked: featureInteractionLocked });
        const featureCell = typeof row.getCell === "function" ? row.getCell("name") : null;
        const groupingCell = typeof row.getCell === "function" ? row.getCell("grouping") : null;
        if (featureCell) featureCell.getElement().innerHTML = featureNameHtml({ ...data, feature_interaction_locked: featureInteractionLocked });
        if (groupingCell) groupingCell.getElement().innerHTML = groupingHtml({ ...data, interaction_locked: interactionLocked });
      }
      return;
    }
    if (el("gbmFeatureFallback")) {
      renderFeatureFallback(applyInteractionLocksToFeatures(features));
    }
  }

  function syncFeatureInteractionControlsAfter(updates) {
    const pending = (updates || []).filter((update) => update && typeof update.then === "function");
    if (!pending.length) {
      syncFeatureInteractionControls();
      return;
    }
    Promise.all(pending).then(syncFeatureInteractionControls, syncFeatureInteractionControls);
  }

  function featureScenarioDropdownHtml(scenarios, activeScenario = null) {
    const rows = featureScenarioRows(scenarios);
    const active = normaliseActiveFeatureScenario(activeScenario);
    const draft = featureDraftForData(config);
    const hasScenarioDraft = Boolean(draft?.scenarioEdited);
    const draftScenarioName = hasScenarioDraft ? String(draft.scenarioName || "") : null;
    const activeSelectedCurrent = active?.status === "current" && rows.some((scenario) => scenario.name === active.name);
    const draftSelectedCurrent = Boolean(draftScenarioName) && rows.some((scenario) => scenario.name === draftScenarioName);
    const selectedCurrent = hasScenarioDraft ? draftSelectedCurrent : activeSelectedCurrent;
    const synthetic = !hasScenarioDraft && active && !activeSelectedCurrent
      ? { label: trainedFeatureScenarioLabel(active) }
      : null;
    const hasOptions = rows.length || synthetic;
    const hidden = hasOptions ? "" : " hidden";
    const disabled = hasOptions ? "" : " disabled";
    const scenarioClass = selectedCurrent || synthetic ? " has-scenario" : "";
    const selectedName = selectedCurrent ? (hasScenarioDraft ? draftScenarioName : active.name) : "";
    return `
      <div id="gbmFeatureScenarioDropdown" class="gbm-feature-scenario-select${hidden}" data-gbm-selected-feature-scenario="${escapeHtml(selectedName)}" data-gbm-feature-menu-root>
        <button id="gbmFeatureScenarioButton" class="gbm-feature-menu-button gbm-feature-scenario-button${scenarioClass}" type="button" aria-haspopup="true" aria-expanded="false" aria-label="Features" title="Apply a saved feature scenario to the Feature table" data-gbm-feature-menu-button${disabled}>Features</button>
        <div id="gbmFeatureScenarioMenu" class="gbm-feature-menu gbm-feature-scenario-menu hidden" role="menu" data-gbm-feature-menu>
          ${synthetic ? `<div class="gbm-feature-scenario-row gbm-feature-scenario-row-trained active" role="menuitem" aria-disabled="true" data-gbm-trained-feature-scenario-row>${escapeHtml(synthetic.label)}</div>` : ""}
          ${rows.map((scenario) => `
            <button class="gbm-feature-scenario-row${selectedCurrent && scenario.name === selectedName ? " active" : ""}" type="button" role="menuitemradio" aria-checked="${selectedCurrent && scenario.name === selectedName ? "true" : "false"}" data-gbm-feature-scenario="${escapeHtml(scenario.name)}">${escapeHtml(featureScenarioLabel(scenario))}</button>
          `).join("")}
        </div>
      </div>
    `;
  }

  function featureScenarioRows(scenarios) {
    if (!Array.isArray(scenarios)) return [];
    return scenarios
      .map((scenario) => ({
        name: String(scenario?.name || "").trim(),
        features: scenarioFeatureList(scenario?.features),
      }))
      .filter((scenario) => scenario.name);
  }

  function scenarioFeatureList(features) {
    if (!Array.isArray(features)) return [];
    return features.map((feature) => String(feature || "").trim()).filter(Boolean);
  }

  function featureScenarioLabel(scenario) {
    return `${scenario.name} (${scenario.features.length.toLocaleString()})`;
  }

  function normaliseActiveFeatureScenario(activeScenario) {
    if (!activeScenario || typeof activeScenario !== "object") return null;
    const name = String(activeScenario.name || "").trim();
    if (!name) return null;
    const status = String(activeScenario.status || "").trim().toLowerCase();
    if (!["current", "stale", "missing"].includes(status)) return null;
    return {
      name,
      status,
      features: scenarioFeatureList(activeScenario.features),
      current_features: scenarioFeatureList(activeScenario.current_features),
    };
  }

  function activeCurrentFeatureScenarioName(data = config) {
    const active = normaliseActiveFeatureScenario(data?.active_feature_scenario);
    if (!active || active.status !== "current") return "";
    return featureScenarioRows(data?.feature_scenarios || []).some((scenario) => scenario.name === active.name) ? active.name : "";
  }

  function hasSyntheticActiveFeatureScenario(data = config) {
    const active = normaliseActiveFeatureScenario(data?.active_feature_scenario);
    return Boolean(active && !activeCurrentFeatureScenarioName(data));
  }

  function featureScenarioSelectionEdited(name, data = config) {
    const selected = String(name || "");
    if (selected !== activeCurrentFeatureScenarioName(data)) return true;
    return hasSyntheticActiveFeatureScenario(data) && !document.querySelector("[data-gbm-trained-feature-scenario-row]");
  }

  function trainedFeatureScenarioLabel(activeScenario) {
    const count = activeScenario.features.length.toLocaleString();
    if (activeScenario.status === "stale") return `${activeScenario.name} (${count}; trained; spec changed)`;
    if (activeScenario.status === "missing") return `${activeScenario.name} (${count}; trained; missing from spec)`;
    return `${activeScenario.name} (${count})`;
  }

  function featureScenarioByName(name) {
    const target = String(name || "");
    return featureScenarioRows(config?.feature_scenarios || []).find((scenario) => scenario.name === target) || null;
  }

  function currentFeatureScenarioPayload() {
    const selected = el("gbmFeatureScenarioDropdown")?.dataset.gbmSelectedFeatureScenario || "";
    const scenario = featureScenarioByName(selected);
    return scenario ? { name: scenario.name, features: scenario.features } : null;
  }

  function resetFeatureScenarioSelection() {
    setFeatureScenarioSelection("");
  }

  function setFeatureScenarioSelection(name) {
    const root = el("gbmFeatureScenarioDropdown");
    if (!root) return;
    const selected = String(name || "");
    root.dataset.gbmSelectedFeatureScenario = selected;
    const button = el("gbmFeatureScenarioButton");
    button?.classList.toggle("has-scenario", Boolean(selected));
    for (const row of root.querySelectorAll("[data-gbm-feature-scenario]")) {
      const active = row.getAttribute("data-gbm-feature-scenario") === selected;
      row.classList.toggle("active", active);
      row.setAttribute("aria-checked", active ? "true" : "false");
    }
    for (const row of root.querySelectorAll("[data-gbm-trained-feature-scenario-row]")) {
      row.classList.toggle("active", !selected);
    }
  }

  function bindFeatureScenarioActions() {
    const root = el("gbmFeatureScenarioDropdown");
    if (!root) return;
    bindGbmFeatureToolbarMenu(root);
    for (const button of root.querySelectorAll("[data-gbm-feature-scenario]")) {
      button.addEventListener("click", () => applyFeatureScenario(button.getAttribute("data-gbm-feature-scenario") || ""));
    }
  }

  function bindFeatureToolbarOutsideClicks() {
    if (featureToolbarOutsideClickBound) return;
    document.addEventListener("click", () => closeGbmFeatureToolbarMenus());
    featureToolbarOutsideClickBound = true;
  }

  function bindGbmFeatureToolbarMenu(root, { beforeOpen = null } = {}) {
    const button = root.querySelector("[data-gbm-feature-menu-button]");
    const menu = root.querySelector("[data-gbm-feature-menu]");
    if (!button || !menu) return;
    button.addEventListener("click", (event) => {
      event.stopPropagation();
      if (button.disabled) return;
      const opening = menu.classList.contains("hidden");
      closeGbmFeatureToolbarMenus(root);
      if (opening) {
        if (beforeOpen) beforeOpen();
        menu.classList.remove("hidden");
        button.setAttribute("aria-expanded", "true");
      } else {
        closeGbmFeatureToolbarMenu(root);
      }
    });
    root.addEventListener("keydown", (event) => {
      if (event.key !== "Escape") return;
      closeGbmFeatureToolbarMenu(root, { focus: true });
    });
    menu.addEventListener("click", (event) => event.stopPropagation());
  }

  function closeGbmFeatureToolbarMenus(exceptRoot = null) {
    for (const root of document.querySelectorAll("[data-gbm-feature-menu-root]")) {
      if (exceptRoot && root === exceptRoot) continue;
      closeGbmFeatureToolbarMenu(root);
    }
  }

  function closeGbmFeatureToolbarMenu(root, { focus = false } = {}) {
    const button = root?.querySelector("[data-gbm-feature-menu-button]");
    const menu = root?.querySelector("[data-gbm-feature-menu]");
    if (!button || !menu) return;
    menu.classList.add("hidden");
    button.setAttribute("aria-expanded", "false");
    if (focus) button.focus();
  }

  function bindEvaluationViewModeActions() {
    for (const input of document.querySelectorAll("input[name='gbmEvaluationViewMode']")) {
      input.addEventListener("change", () => {
        if (!input.checked) return;
        evaluationViewMode = normaliseEvaluationViewMode(input.value);
        rerenderEvaluationChart();
      });
    }
  }

  function rerenderEvaluationChart() {
    if (liveProgress?.evaluation) {
      renderLiveProgress(liveProgress);
      return;
    }
    renderEvaluationChart();
  }

  async function createSampleColumn() {
    const button = el("gbmCreateSampleBtn");
    if (!button) return;
    button.disabled = true;
    button.textContent = "Creating...";
    setGbmNotice("");
    try {
      const result = await api("/api/gbm/sample", { method: "POST", body: "{}", clientTiming: true });
      clearToolCaches();
      config = result.config;
      measureToolRender(tool, () => render(result.config));
    } catch (error) {
      button.disabled = false;
      button.textContent = "Create sample column";
      setGbmNotice(error.message);
    }
  }

  function setFeatureIncludes(include) {
    resetFeatureScenarioSelection();
    if (featureTable) {
      const updates = [];
      for (const row of featureTable.getRows()) {
        const data = row.getData();
        if (isFeatureSelectable(data)) updates.push(row.update({ include }));
      }
      syncFeatureSectionTitleAfter(updates);
      syncFeatureInteractionControlsAfter(updates);
      return;
    }
    for (const checkbox of document.querySelectorAll("[data-gbm-feature]")) {
      const name = checkbox.getAttribute("data-gbm-feature") || "";
      const feature = (config?.features || []).find((item) => item.name === name);
      if (!feature || !isFeatureSelectable(feature)) continue;
      checkbox.checked = include;
      checkbox.dispatchEvent(new Event("change", { bubbles: true }));
    }
    syncFeatureSectionTitle();
    syncFeatureInteractionControls();
  }

  function applyFeatureScenario(name) {
    const scenario = featureScenarioByName(name);
    if (!scenario) return;
    setFeatureScenarioSelection(scenario.name);
    closeGbmFeatureToolbarMenu(el("gbmFeatureScenarioDropdown"));
    const selected = new Set(scenario.features);
    if (featureTable) {
      const updates = [];
      for (const row of featureTable.getRows()) {
        const data = row.getData();
        updates.push(row.update({ include: isFeatureSelectable(data) && selected.has(data.name) }));
      }
      syncFeatureSectionTitleAfter(updates);
      syncFeatureInteractionControlsAfter(updates);
      return;
    }
    for (const checkbox of document.querySelectorAll("[data-gbm-feature]")) {
      const name = checkbox.getAttribute("data-gbm-feature") || "";
      const feature = (config?.features || []).find((item) => item.name === name);
      checkbox.checked = Boolean(feature && isFeatureSelectable(feature) && selected.has(name));
    }
    syncFeatureSectionTitle();
    syncFeatureInteractionControls();
  }

  function featureSectionTitle(features) {
    return `Features (${selectedFeatureCount(features)})`;
  }

  function selectedFeatureCount(features) {
    return (features || []).filter((feature) => feature?.include && isFeatureSelectable(feature)).length;
  }

  function syncFeatureSectionTitle() {
    const title = el("gbmFeatureSectionTitle");
    if (!title) return;
    title.textContent = featureSectionTitle(currentFeatureRows());
  }

  function syncFeatureSectionTitleAfter(updates) {
    const pending = (updates || []).filter((update) => update && typeof update.then === "function");
    if (!pending.length) {
      syncFeatureSectionTitle();
      return;
    }
    Promise.all(pending).then(syncFeatureSectionTitle, syncFeatureSectionTitle);
  }

  function setTrainingState(active) {
    isTraining = Boolean(active);
    syncTrainingButton();
    syncModelActionButtons();
  }

  function syncTrainingButton() {
    const button = el("gbmTrainBtn");
    if (!button) return;
    button.textContent = isTraining ? "Training..." : "Train GBM";
    button.classList.toggle("training", isTraining);
    button.disabled = isTraining;
    if (isTraining) {
      button.setAttribute("aria-busy", "true");
    } else {
      button.removeAttribute("aria-busy");
    }
  }

  function setGbmNotice(message) {
    const text = String(message || "");
    let notice = el("gbmNotice");
    if (!notice) {
      const mount = el("modelToolWrap");
      let toolNode = mount?.querySelector(".gbm-tool");
      if (!toolNode && mount && text) {
        mount.innerHTML = `<div class="gbm-tool gbm-tool-error-shell"></div>`;
        toolNode = mount.querySelector(".gbm-tool");
      }
      if (!toolNode) return;
      notice = document.createElement("div");
      notice.id = "gbmNotice";
      notice.className = "gbm-notice hidden";
      notice.setAttribute("role", "alert");
      notice.setAttribute("aria-live", "polite");
      toolNode.prepend(notice);
    }
    notice.textContent = text;
    notice.classList.toggle("hidden", !text);
  }

  function setTrainingStatus(message, phase = "", detail = "") {
    const status = el("gbmTrainingStatus");
    if (!status) return;
    const text = String(message || "");
    const detailText = String(detail || "");
    status.innerHTML = trainingStatusContentHtml(text, detailText);
    status.dataset.phase = String(phase || "");
    status.classList.toggle("hidden", !text && !detailText);
  }

  function syncSidebarModelChooser(models, activeModelId) {
    const list = el("gbmModelSelect");
    const meta = el("gbmModelSelectedMeta");
    if (!list) return;
    const normalisedModels = uniqueModels(models.map(normaliseModel).filter((model) => model.model_id));
    const activeModel = normalisedModels.find((model) => model.model_id === activeModelId) || null;
    if (meta) meta.textContent = activeModel ? modelLabel(activeModel) : "No active model";
    const modelsByGroup = new Map();
    for (const model of normalisedModels) {
      const group = modelGroupLabel(model);
      if (!modelsByGroup.has(group)) modelsByGroup.set(group, []);
      modelsByGroup.get(group).push(model);
    }
    const groups = [...modelsByGroup.keys()];
    if (!state.gbmModelGroupsInitialised) {
      groups.forEach((group) => state.collapsedGbmModelGroups.add(group));
      const openGroup = activeModel ? modelGroupLabel(activeModel) : groups[0];
      if (openGroup) state.collapsedGbmModelGroups.delete(openGroup);
      state.gbmModelGroupsInitialised = true;
    }
    for (const group of state.collapsedGbmModelGroups) {
      if (!groups.includes(group)) state.collapsedGbmModelGroups.delete(group);
    }
    list.innerHTML = "";
    if (!normalisedModels.length) {
      list.innerHTML = `<div class="gbm-empty-state">No GBMs trained yet</div>`;
      return;
    }
    for (const group of groups) {
      const collapsed = state.collapsedGbmModelGroups.has(group);
      const heading = document.createElement("button");
      heading.type = "button";
      heading.className = "saved-filter-theme gbm-model-theme";
      heading.dataset.gbmModelGroup = group;
      heading.setAttribute("aria-expanded", String(!collapsed));
      heading.setAttribute("aria-label", `${collapsed ? "Expand" : "Collapse"} ${group} GBM models`);
      heading.title = `${collapsed ? "Expand" : "Collapse"} ${group} GBM models`;
      heading.innerHTML = `<span class="saved-filter-theme-icon" aria-hidden="true"></span><span class="saved-filter-theme-label">${escapeHtml(group)}</span>`;
      heading.addEventListener("click", () => toggleGbmModelGroup(group));
      list.append(heading);
      for (const model of modelsByGroup.get(group) || []) {
        const active = model.model_id === activeModelId;
        const button = document.createElement("button");
        button.type = "button";
        button.className = `feature gbm-model-option${active ? " active" : ""}`;
        button.dataset.gbmModelId = model.model_id;
        button.dataset.gbmModelGroup = group;
        button.hidden = collapsed;
        button.setAttribute("role", "option");
        button.setAttribute("aria-selected", String(active));
        button.innerHTML = `<span class="saved-filter-name">${escapeHtml(modelLabel(model))}</span><span class="gbm-model-detail">${escapeHtml(modelDetailLabel(model))}</span>`;
        button.addEventListener("click", () => {
          if (!active) activateModel(model.model_id);
        });
        list.append(button);
      }
    }
  }

  function toggleGbmModelGroup(group) {
    const collapsed = !state.collapsedGbmModelGroups.has(group);
    if (collapsed) {
      state.collapsedGbmModelGroups.add(group);
    } else {
      state.collapsedGbmModelGroups.delete(group);
    }
    const list = el("gbmModelSelect");
    list.querySelectorAll(".gbm-model-theme").forEach((heading) => {
      if (heading.dataset.gbmModelGroup !== group) return;
      heading.setAttribute("aria-expanded", String(!collapsed));
      heading.setAttribute("aria-label", `${collapsed ? "Expand" : "Collapse"} ${group} GBM models`);
      heading.title = `${collapsed ? "Expand" : "Collapse"} ${group} GBM models`;
    });
    list.querySelectorAll(".gbm-model-option").forEach((button) => {
      if (button.dataset.gbmModelGroup === group) button.hidden = collapsed;
    });
  }

  function normaliseModel(model) {
    return {
      ...model,
      model_id: String(model?.model_id || ""),
      label: String(model?.label || ""),
      response_column: String(model?.response_column || "actualNumerator"),
      offset_column: model?.offset_column ? String(model.offset_column) : "",
      objective: String(model?.objective || ""),
      metric: String(model?.metric || ""),
      training_mode: normaliseTrainingMode(model?.training_mode),
      best_iteration: Number(model?.best_iteration || 0),
      best_metrics: model?.best_metrics,
      parameters: model?.parameters || {},
      training_rows: Number(model?.training_rows || 0),
      sample_column: String(model?.sample_column || ""),
      sample_source: String(model?.sample_source || ""),
      timings: model?.timings || {},
      sources: model?.sources || {},
      created_at: String(model?.created_at || ""),
      active: Boolean(model?.active),
    };
  }

  function normaliseTrainingMode(value) {
    return String(value || "normal").trim().toLowerCase() === "ebm" ? "ebm" : "normal";
  }

  function uniqueModels(models) {
    const seen = new Set();
    return models.filter((model) => {
      if (seen.has(model.model_id)) return false;
      seen.add(model.model_id);
      return true;
    });
  }

  function modelGroupLabel(model) {
    return `${model.response_column || "actualNumerator"} / ${modelWeightLabel(model.offset_column)}`;
  }

  function modelLabel(model) {
    return model.label || model.model_id;
  }

  function modelDetailLabel(model) {
    return gbmModelDetailLabel(model);
  }

  function syncSidebarFromSchema() {
    const sources = state.schema?.data_sources || [];
    const models = [];
    const seen = new Set();
    for (const source of sources) {
      if (source.kind !== "gbm_predictions" || !source.model_id || seen.has(source.model_id)) continue;
      seen.add(source.model_id);
      models.push({
        model_id: source.model_id,
        label: String(source.label || source.model_id).replace(/\s+-\s+(Predictions|SHAP values|SHAP summary)$/i, ""),
        active: Boolean(source.active),
        response_column: source.response_column,
        offset_column: source.offset_column,
        created_at: source.created_at,
        objective: source.objective,
        metric: source.metric,
        training_mode: source.training_mode,
        best_iteration: source.best_iteration,
        best_metrics: source.best_metrics,
      });
    }
    const activeModel = models.find((model) => model.active)?.model_id || "";
    syncSidebarModelChooser(models, activeModel);
  }

  async function renderTables(data) {
    featureTable = null;
    parameterTable = null;
    modelTable = null;
    ebmGainSummaryTable = null;
    const features = applyInteractionLocksToFeatures(data.features || []);
    const parameters = data.parameters || [];
    try {
      const Tabulator = await loadTabulator();
      if (!config || data !== config) return;
      const models = modelRows(config.models || data.models || []);
      const modelFallback = el("gbmModelFallback");
      if (modelFallback) modelFallback.innerHTML = "";
      modelTable = new Tabulator("#gbmModelGrid", {
        data: models,
        height: "100%",
        layout: "fitDataStretch",
        placeholder: "No GBMs trained yet",
        initialSort: [{ column: "created_sort", dir: "desc" }],
        selectableRows: true,
        selectableRowsRangeMode: "click",
        columns: [
          { title: "", field: "active", formatter: activeModelDotFormatter, hozAlign: "center", headerHozAlign: "center", width: 28, minWidth: 28, headerSort: false, resizable: false },
          { title: "Model", field: "model_label", sorter: "string", formatter: modelNameFormatter, widthGrow: 3, headerSort: true },
          { title: "Created", field: "created_sort", sorter: "number", formatter: (cell) => escapeHtml(cell.getRow().getData().created_display), width: 105, headerSort: true },
          { title: "Response", field: "response_column", sorter: "string", widthGrow: 1.6, headerSort: true },
          { title: "Weight", field: "weight_display", sorter: "string", widthGrow: 1.2, headerSort: true },
          { title: "Objective", field: "objective", sorter: "string", widthGrow: 1.1, headerSort: true },
          { title: "Metric", field: "metric", sorter: "string", widthGrow: 1.1, headerSort: true },
          { title: "Mode", field: "training_mode_display", sorter: "string", width: 70, headerSort: true },
          { title: "Constraints", field: "constraint_display", sorter: "string", widthGrow: 1.2, headerSort: true },
          { title: "Train", field: "training_rows", sorter: "number", formatter: (cell) => formatModelCount(cell.getValue()), hozAlign: "right", headerHozAlign: "right", width: 86, headerSort: true },
          { title: "Best iter.", field: "best_iteration", sorter: "number", formatter: (cell) => formatModelCount(cell.getValue()), hozAlign: "right", headerHozAlign: "right", width: 92, headerSort: true },
          { title: "tr@best", field: "best_training_metric", sorter: "number", formatter: (cell) => escapeHtml(formatModelMetric(cell.getValue())), hozAlign: "right", headerHozAlign: "right", width: 96, headerSort: true, headerTooltip: "Training metric at best iteration" },
          { title: "te@best", field: "best_test_metric", sorter: "number", formatter: (cell) => escapeHtml(formatModelMetric(cell.getValue())), hozAlign: "right", headerHozAlign: "right", width: 96, headerSort: true, headerTooltip: "Test metric at best iteration" },
          { title: "n_iter", field: "param_num_iterations", sorter: "number", formatter: (cell) => escapeHtml(formatModelInteger(cell.getValue())), hozAlign: "right", headerHozAlign: "right", width: 58, headerSort: true, headerTooltip: "num_iterations" },
          { title: "lr", field: "param_learning_rate", sorter: "number", formatter: (cell) => escapeHtml(formatModelDecimal(cell.getValue())), hozAlign: "right", headerHozAlign: "right", width: 50, headerSort: true, headerTooltip: "learning_rate" },
          { title: "leaves", field: "param_num_leaves", sorter: "number", formatter: (cell) => escapeHtml(formatModelInteger(cell.getValue())), hozAlign: "right", headerHozAlign: "right", width: 58, headerSort: true, headerTooltip: "num_leaves" },
          { title: "depth", field: "param_max_depth", sorter: "number", formatter: (cell) => escapeHtml(formatModelInteger(cell.getValue())), hozAlign: "right", headerHozAlign: "right", width: 56, headerSort: true, headerTooltip: "max_depth" },
          { title: "min_leaf", field: "param_min_data_in_leaf", sorter: "number", formatter: (cell) => escapeHtml(formatModelInteger(cell.getValue())), hozAlign: "right", headerHozAlign: "right", width: 64, headerSort: true, headerTooltip: "min_data_in_leaf" },
          { title: "ES", field: "param_early_stopping_rounds", sorter: "number", formatter: (cell) => escapeHtml(formatModelInteger(cell.getValue())), hozAlign: "right", headerHozAlign: "right", width: 46, headerSort: true, headerTooltip: "early_stopping_rounds" },
          { title: "Run time", field: "runtime_seconds", sorter: "number", formatter: (cell) => escapeHtml(cell.getRow().getData().runtime_display), hozAlign: "right", headerHozAlign: "right", width: 84, headerSort: true },
          { title: "Sample", field: "sample_display", sorter: "string", widthGrow: 1.1, headerSort: true },
        ],
      });
      modelTable.on("rowSelectionChanged", syncModelActionButtons);
      syncModelActionButtons();
      featureTable = new Tabulator("#gbmFeatureGrid", {
        data: features,
        height: "100%",
        layout: "fitColumns",
        initialSort: featureTableInitialSort(),
        columns: featureTableColumns(),
        rowFormatter: (row) => {
          const data = row.getData();
          const element = row.getElement();
          element.classList.toggle("gbm-feature-invalid", isInvalidFeature(data));
          element.classList.toggle("gbm-feature-disabled", !isFeatureSelectable(data));
          element.classList.toggle("gbm-feature-warning", isFeatureSelectable(data) && Boolean(data.high_cardinality));
        },
      });
      featureTable.on("rowContext", openFeatureContextMenuForTabulatorRow);
      ebmGainSummaryTable = new Tabulator("#gbmEbmGainSummaryGrid", {
        data: cachedEbmGainSummaryRowsForActiveModel(),
        height: "100%",
        layout: "fitColumns",
        placeholder: "No EBM gain summary available",
        initialSort: [{ column: "gain", dir: "desc" }],
        columns: ebmGainSummaryColumns(),
      });
      ebmGainSummaryTable.on("rowContext", openEbmGainContextMenuForTabulatorRow);
      parameterTable = new Tabulator("#gbmParameterGrid", {
        data: parameters,
        height: "100%",
        layout: "fitColumns",
        initialSort: [{ column: "important", dir: "desc" }],
        columns: [
          { title: "Parameter", field: "name", widthGrow: 2 },
          { title: "Value", field: "value", editor: "adaptable", editorParams: parameterValueEditorParams(), widthGrow: 1 },
        ],
      });
      parameterTable.on("cellEdited", syncGridSampleControl);
    } catch (_) {
      renderModelFallback(modelRows(config?.models || data.models || []));
      renderFeatureFallback(features);
      renderEbmGainSummaryFallback([]);
      renderParameterFallback(parameters);
      bindFallbackParameterGridSearchControls();
    }
    syncFeatureInteractionControls();
    syncGridSampleControl();
    updateFeatureMetricView(features, { refreshColumns: false });
  }

  function featureTableColumns() {
    return [
      { title: "Feature", field: "name", formatter: featureNameFormatter, cssClass: "gbm-feature-name-cell", widthGrow: 3, headerSort: true },
      { title: "Grouping", field: "grouping", formatter: groupingFormatter, widthGrow: 1.1, headerSort: true },
      {
        title: "Use",
        field: "include",
        formatter: useCheckboxFormatter,
        hozAlign: "center",
        headerHozAlign: "center",
        width: 58,
        headerSort: false,
        cellClick: (event) => event.stopPropagation(),
      },
      { title: "Monotonicity", field: "monotonicity", editor: "list", editable: (cell) => isFeatureSelectable(cell.getRow().getData()), editorParams: { values: ["", "Increasing", "Decreasing", "1", "-1"] }, width: 120 },
      featureMetricColumn(),
    ];
  }

  function featureMetricColumn() {
    const shapMode = featureMetricMode === "shap";
    return {
      title: shapMode ? "SHAP" : "Gain",
      field: featureMetricField(),
      formatter: (cell) => shapMode ? formatMeanAbsShap(cell.getValue()) : formatGain(cell.getValue()),
      sorter: shapMode ? featureMetricSorter : "number",
      hozAlign: "center",
      headerHozAlign: "center",
      cssClass: "gbm-feature-metric-cell",
      width: 125,
    };
  }

  function ebmGainSummaryColumns() {
    return [
      { title: "Tree features", field: "tree_features", sorter: "string", widthGrow: 3, headerSort: true },
      { title: "Dim", field: "dim", sorter: "number", formatter: (cell) => escapeHtml(formatModelInteger(cell.getValue())), hozAlign: "center", headerHozAlign: "center", width: 70, headerSort: true },
      { title: "Trees", field: "trees", sorter: "number", formatter: (cell) => escapeHtml(formatModelInteger(cell.getValue())), hozAlign: "center", headerHozAlign: "center", width: 82, headerSort: true },
      { title: "Gain", field: "gain", sorter: "number", formatter: (cell) => escapeHtml(formatEbmSummaryGain(cell.getValue())), hozAlign: "center", headerHozAlign: "center", width: 96, headerSort: true },
      { title: "% Gain", field: "gain_percent", sorter: "number", formatter: (cell) => escapeHtml(formatGainPercent(cell.getValue())), hozAlign: "center", headerHozAlign: "center", width: 96, headerSort: true },
    ];
  }

  function featureTableInitialSort() {
    return [{ column: featureMetricField(), dir: "desc" }];
  }

  function featureMetricField() {
    return featureMetricMode === "shap" ? "mean_abs_shap" : "gain";
  }

  function featureMetricSorter(a, b) {
    const left = featureNumber(a);
    const right = featureNumber(b);
    if (left === null && right === null) return 0;
    if (left === null) return -1;
    if (right === null) return 1;
    return left - right;
  }

  function featureNumber(value) {
    if (value === null || value === undefined || String(value).trim() === "") return null;
    const number = Number(value);
    return Number.isFinite(number) ? number : null;
  }

  function featureMeanAbsShap(feature) {
    return featureNumber(feature?.mean_abs_shap);
  }

  function activeModelDotFormatter(cell) {
    return cell.getValue() ? '<span class="gbm-model-active-dot" title="Active model" aria-label="Active model"></span>' : "";
  }

  function modelNameFormatter(cell) {
    return `<span class="gbm-model-name-main">${escapeHtml(cell.getValue() || "")}</span>`;
  }

  function featureNameFormatter(cell) {
    return featureNameHtml(cell.getRow().getData());
  }

  function featureNameHtml(feature) {
    const lock = feature?.feature_interaction_locked
      ? `<span class="gbm-interaction-lock gbm-feature-interaction-lock" title="Feature isolated from all other features" aria-label="Feature isolated from all other features">&#128274;</span>`
      : "";
    return `
      <span class="gbm-feature-name-line">
        <span class="gbm-feature-name-main">${escapeHtml(feature.name)}${lock}</span>
        <span class="gbm-feature-kind kind">${escapeHtml(featureTypeLabel(feature))}</span>
      </span>
    `;
  }

  function groupingFormatter(cell) {
    return groupingHtml(cell.getRow().getData());
  }

  function groupingHtml(feature) {
    const grouping = String(feature?.grouping || "");
    if (!grouping) return "";
    const lock = feature?.interaction_locked
      ? `<span class="gbm-interaction-lock" title="Feature interaction constrained" aria-label="Feature interaction constrained">&#128274;</span>`
      : "";
    return `<span class="gbm-grouping-value">${escapeHtml(grouping)}${lock}</span>`;
  }

  function featureTypeLabel(feature) {
    const kind = String(feature?.kind || "");
    if (isInvalidFeature(feature) || kind === "invalid") return "invalid";
    if (kind === "categorical") {
      const count = Number(feature?.distinct_count);
      if (Number.isFinite(count)) return `categorical (${count.toLocaleString()})`;
    }
    return kind;
  }

  function useCheckboxFormatter(cell) {
    const rowData = cell.getRow().getData();
    if (!isFeatureSelectable(rowData)) return "";
    const checkbox = document.createElement("input");
    checkbox.type = "checkbox";
    checkbox.className = "gbm-use-checkbox";
    checkbox.checked = Boolean(cell.getValue());
    checkbox.setAttribute("aria-label", `Use ${rowData.name}`);
    checkbox.addEventListener("click", (event) => event.stopPropagation());
    checkbox.addEventListener("change", () => {
      resetFeatureScenarioSelection();
      const updates = [cell.getRow().update({ include: checkbox.checked })];
      syncFeatureSectionTitleAfter(updates);
      syncFeatureInteractionControlsAfter(updates);
    });
    return checkbox;
  }

  function isFeatureSelectable(feature) {
    if (!feature.usable) return false;
    const reserved = currentReservedFeatureNames();
    return !reserved.has(feature.name);
  }

  function isInvalidFeature(feature) {
    return Boolean(feature?.invalid) || String(feature?.kind || "") === "invalid";
  }

  function featureRowClasses(feature) {
    return [
      isInvalidFeature(feature) ? "gbm-feature-invalid" : "",
      isFeatureSelectable(feature) ? "" : "gbm-feature-disabled",
      isFeatureSelectable(feature) && feature.high_cardinality ? "gbm-feature-warning" : "",
    ].filter(Boolean).join(" ");
  }

  function currentReservedFeatureNames() {
    const response = el("actualNumerator")?.value || "actualNumerator";
    const offset = el("denominator")?.value || "denominator";
    const sample = config?.sample?.source === "dataset" ? config.sample.column : "";
    return new Set([response, offset, sample].filter((value) => value && value !== "__none__"));
  }

  function renderFeatureFallback(features) {
    const target = el("gbmFeatureFallback");
    if (!target) return;
    const metricTitle = featureMetricMode === "shap" ? "SHAP" : "Gain";
    target.innerHTML = `
      <table>
        <thead><tr><th>Feature</th><th>Grouping</th><th>Use</th><th>Monotonicity</th><th>${metricTitle}</th></tr></thead>
        <tbody>
          ${sortedFeatureRowsForMetric(features).map((feature) => `
            <tr class="${featureRowClasses(feature)}" data-gbm-feature-row data-gbm-feature-name="${escapeHtml(feature.name)}" data-gbm-feature-interaction-locked="${feature.feature_interaction_locked ? "true" : "false"}">
              <td data-gbm-context-field="name">${featureNameHtml(feature)}</td>
              <td data-gbm-context-field="grouping">${groupingHtml(feature)}</td>
              <td class="gbm-use-cell" data-gbm-context-field="include">${isFeatureSelectable(feature) ? `<input type="checkbox" data-gbm-feature="${escapeHtml(feature.name)}" ${feature.include ? "checked" : ""} />` : ""}</td>
              <td data-gbm-context-field="monotonicity"><input data-gbm-monotonicity="${escapeHtml(feature.name)}" value="${escapeHtml(feature.monotonicity || "")}" ${isFeatureSelectable(feature) ? "" : "disabled"} /></td>
              <td class="numeric gbm-feature-metric-cell" data-gbm-context-field="${escapeHtml(featureMetricField())}">${featureMetricDisplay(feature)}</td>
            </tr>
          `).join("")}
        </tbody>
      </table>
    `;
    for (const checkbox of target.querySelectorAll("[data-gbm-feature]")) {
      checkbox.addEventListener("change", () => {
        resetFeatureScenarioSelection();
        syncFeatureSectionTitle();
        syncFeatureInteractionControls();
      });
    }
    for (const row of target.querySelectorAll("[data-gbm-feature-row]")) {
      row.addEventListener("contextmenu", openFeatureContextMenuForFallbackRow);
    }
    syncFeatureSectionTitle();
  }

  function renderEbmGainSummaryFallback(rows) {
    const target = el("gbmEbmGainSummaryFallback");
    if (!target) return;
    if (rows === null) {
      target.innerHTML = '<div class="gbm-empty-state">Loading EBM gain summary...</div>';
      return;
    }
    const summaryRows = Array.isArray(rows) ? rows : [];
    if (!summaryRows.length) {
      target.innerHTML = '<div class="gbm-empty-state">No EBM gain summary available</div>';
      return;
    }
    target.innerHTML = `
      <table>
        <thead><tr><th>Tree features</th><th>Dim</th><th>Trees</th><th>Gain</th><th>% Gain</th></tr></thead>
        <tbody>
          ${summaryRows.map((row) => `
            <tr data-gbm-ebm-gain-row data-gbm-ebm-features="${escapeHtml(JSON.stringify(ebmSummaryFeatures(row)))}" data-gbm-ebm-dim="${escapeHtml(row.dim)}">
              <td>${escapeHtml(row.tree_features || "")}</td>
              <td class="numeric">${escapeHtml(formatModelInteger(row.dim))}</td>
              <td class="numeric">${escapeHtml(formatModelInteger(row.trees))}</td>
              <td class="numeric gbm-gain-cell">${escapeHtml(formatEbmSummaryGain(row.gain))}</td>
              <td class="numeric gbm-gain-cell">${escapeHtml(formatGainPercent(row.gain_percent))}</td>
            </tr>
          `).join("")}
        </tbody>
      </table>
    `;
    for (const row of target.querySelectorAll("[data-gbm-ebm-gain-row]")) {
      row.addEventListener("contextmenu", openEbmGainContextMenuForFallbackRow);
    }
  }

  function openFeatureContextMenuForTabulatorRow(event, row) {
    const feature = row?.getData?.() || {};
    const field = event.target?.closest?.(".tabulator-cell")?.getAttribute("tabulator-field") || "";
    openGbmFeatureContextMenu(event, { features: [feature.name], field });
  }

  function openEbmGainContextMenuForTabulatorRow(event, row) {
    const data = row?.getData?.() || {};
    openGbmFeatureContextMenu(event, { features: ebmSummaryFeatures(data), ebmDim: data.dim });
  }

  function openFeatureContextMenuForFallbackRow(event) {
    const field = event.target?.closest?.("[data-gbm-context-field]")?.getAttribute("data-gbm-context-field") || "";
    openGbmFeatureContextMenu(event, { features: [event.currentTarget?.dataset?.gbmFeatureName || ""], field });
  }

  function openEbmGainContextMenuForFallbackRow(event) {
    const row = event.currentTarget;
    openGbmFeatureContextMenu(event, {
      features: parseFallbackEbmFeatures(row?.dataset?.gbmEbmFeatures || "[]"),
      ebmDim: row?.dataset?.gbmEbmDim,
    });
  }

  function openGbmFeatureContextMenu(event, context) {
    const actions = gbmFeatureContextActions(context);
    closeGbmFeatureContextMenu();
    if (!actions.length) return;
    event.preventDefault();
    event.stopPropagation();
    const menu = gbmFeatureContextMenu();
    menu.innerHTML = "";
    for (const action of actions) {
      if (action.divider) {
        const divider = document.createElement("div");
        divider.className = "gbm-feature-context-menu-divider";
        divider.setAttribute("role", "separator");
        menu.append(divider);
        continue;
      }
      const button = document.createElement("button");
      button.className = "gbm-feature-context-menu-item";
      button.type = "button";
      button.setAttribute("role", "menuitem");
      button.textContent = action.label;
      button.addEventListener("click", () => {
        closeGbmFeatureContextMenu();
        action.run();
      });
      menu.append(button);
    }
    menu.hidden = false;
    const rowRect = event.currentTarget?.getElement?.()?.getBoundingClientRect?.()
      || event.currentTarget?.getBoundingClientRect?.()
      || { left: event.clientX, top: event.clientY, height: 18 };
    const clientX = event.clientX || rowRect.left + 12;
    const clientY = event.clientY || rowRect.top + Math.min(18, Math.max(8, rowRect.height / 2));
    positionGbmFeatureContextMenu(menu, clientX, clientY);
    menu.querySelector("button")?.focus({ preventScroll: true });
    window.addEventListener("pointerdown", handleGbmFeatureContextMenuPointerDown, true);
    window.addEventListener("keydown", handleGbmFeatureContextMenuKeydown, true);
    window.addEventListener("resize", closeGbmFeatureContextMenu, true);
    window.addEventListener("scroll", closeGbmFeatureContextMenu, true);
  }

  function gbmFeatureContextMenu() {
    let menu = document.getElementById("gbmFeatureContextMenu");
    if (menu) return menu;
    menu = document.createElement("div");
    menu.id = "gbmFeatureContextMenu";
    menu.className = "gbm-feature-context-menu";
    menu.hidden = true;
    menu.setAttribute("role", "menu");
    document.body.append(menu);
    return menu;
  }

  function gbmFeatureContextActions(context = {}) {
    const names = normaliseContextFeatureNames(context.features);
    const dim = Number(context.ebmDim || names.length || 0);
    if (context.ebmDim !== undefined && ![1, 2].includes(dim)) return [];
    if (dim === 2) {
      if (names.length !== 2 || !featuresHaveSavedShap(names)) return [];
      return [{ label: "Go to SHAP", run: () => goToGbmShap(names) }];
    }
    if (names.length !== 1) return [];
    const featureName = names[0];
    const field = String(context.field || "name");
    const feature = featureByName(featureName);
    if (field === "grouping") {
      return feature?.grouping && isFeatureSelectable(feature)
        ? [{ label: "Toggle group interaction constraint", run: () => toggleGroupInteractionConstraint(feature.grouping) }]
        : [];
    }
    if (field === "monotonicity") {
      return [{ label: "Clear all monotonicities", run: clearAllFeatureMonotonicities }];
    }
    if (field === featureMetricField()) {
      return feature ? [{ label: "Copy importance value", run: () => copyFeatureImportanceValue(featureName) }] : [];
    }
    if (field !== "name") return [];
    const navigationActions = [];
    if (canNavigateFeatureToLineBar(featureName)) {
      navigationActions.push({ label: "Go to Line and Bar", run: () => goToLineBarFeature(featureName) });
    }
    if (featuresHaveSavedShap([featureName])) {
      navigationActions.push({ label: "Go to SHAP", run: () => goToGbmShap([featureName]) });
      navigationActions.push({ label: "Go to Stacked SHAP", run: () => goToGbmStackedShap(featureName) });
    }
    if (!feature || !isFeatureSelectable(feature)) return navigationActions;
    return [
      { label: "Toggle interaction constraint", run: () => toggleFeatureInteractionConstraint(featureName) },
      ...(navigationActions.length ? [{ divider: true }, ...navigationActions] : []),
    ];
  }

  function toggleFeatureInteractionConstraint(featureName) {
    if (!featureName) return;
    if (featureTable) {
      const row = featureTable.getRows().find((item) => item.getData()?.name === featureName);
      if (!row) return;
      const data = row.getData();
      row.update({ feature_interaction_locked: !data.feature_interaction_locked }).then(
        () => syncFeatureInteractionLocks(currentFeatureRows()),
        () => syncFeatureInteractionLocks(currentFeatureRows())
      );
      return;
    }
    const row = document.querySelector(`[data-gbm-feature-row][data-gbm-feature-name="${cssEscape(featureName)}"]`);
    if (!row) return;
    const locked = row.dataset.gbmFeatureInteractionLocked !== "true";
    row.dataset.gbmFeatureInteractionLocked = locked ? "true" : "false";
    const feature = { ...(featureByName(featureName) || {}), feature_interaction_locked: locked };
    const featureCell = row.querySelector('[data-gbm-context-field="name"]');
    if (featureCell) featureCell.innerHTML = featureNameHtml(feature);
    syncFeatureInteractionControls();
  }

  function toggleGroupInteractionConstraint(grouping) {
    const name = String(grouping || "").trim();
    if (!name) return;
    const checkbox = document.querySelector(`[data-gbm-interaction-grouping="${cssEscape(name)}"]`);
    if (!checkbox) return;
    clearTrainedInteractionConstraintRows();
    checkbox.checked = !checkbox.checked;
    checkbox.dispatchEvent(new Event("change", { bubbles: true }));
  }

  function clearAllFeatureMonotonicities() {
    if (featureTable) {
      const updates = featureTable.getRows().map((row) => row.update({ monotonicity: "" }));
      Promise.all(updates).then(syncFeatureInteractionControls, syncFeatureInteractionControls);
      return;
    }
    for (const input of document.querySelectorAll("[data-gbm-monotonicity]")) input.value = "";
  }

  function copyFeatureImportanceValue(featureName) {
    const feature = featureByName(featureName);
    if (!feature) return;
    const metric = featureMetricMode === "shap" ? "SHAP" : "Gain";
    const value = featureMetricDisplay(feature);
    copyTextToClipboard(`${featureName}\t${metric}\t${value}`, "Copied importance value");
  }

  function copyTextToClipboard(text, successMessage) {
    let copy;
    try {
      copy = navigator.clipboard?.writeText
        ? navigator.clipboard.writeText(text)
        : fallbackCopyTextToClipboard(text);
    } catch (error) {
      copy = Promise.reject(error);
    }
    Promise.resolve(copy).then(
      () => setGbmNotice(successMessage),
      () => setGbmNotice("Unable to copy importance value")
    );
  }

  function fallbackCopyTextToClipboard(text) {
    const input = document.createElement("textarea");
    input.value = text;
    input.setAttribute("readonly", "");
    input.style.position = "fixed";
    input.style.left = "-9999px";
    document.body.append(input);
    input.select();
    const copied = document.execCommand("copy");
    input.remove();
    if (!copied) throw new Error("copy failed");
  }

  function normaliseContextFeatureNames(features) {
    const names = Array.isArray(features) ? features : [features];
    const seen = new Set();
    const result = [];
    for (const value of names) {
      const name = String(value || "").trim();
      if (!name || seen.has(name)) continue;
      seen.add(name);
      result.push(name);
    }
    return result;
  }

  function ebmSummaryFeatures(row = {}) {
    const features = Array.isArray(row.features) ? row.features : [];
    const names = normaliseContextFeatureNames(features);
    if (names.length) return names;
    return normaliseContextFeatureNames(String(row.tree_features || "").split(/\s+x\s+/));
  }

  function parseFallbackEbmFeatures(value) {
    try {
      return normaliseContextFeatureNames(JSON.parse(value));
    } catch (_) {
      return [];
    }
  }

  function featuresHaveSavedShap(names) {
    return names.length > 0 && names.every((name) => featureMeanAbsShap(featureByName(name)) !== null);
  }

  function featureByName(name) {
    return (config?.features || []).find((feature) => feature?.name === name) || null;
  }

  function canNavigateFeatureToLineBar(name) {
    return Boolean(lineBarFeatureTargetSource(name));
  }

  function lineBarFeatureTargetSource(name) {
    const featureName = String(name || "");
    if (!featureName || !lineBarToolAvailable()) return "";
    const currentSource = state.source || "dataset";
    if (sourceHasFeature(currentSource, featureName)) return currentSource;
    if (sourceHasFeature("dataset", featureName)) return "dataset";
    const feature = featureByName(featureName);
    return feature && !isInvalidFeature(feature) ? "dataset" : "";
  }

  function lineBarToolAvailable() {
    if ((state.schema?.tools || []).some((item) => item?.id === "line_bar")) return true;
    const button = document.getElementById("lineBarTool");
    return Boolean(button && !button.disabled && !button.classList.contains("hidden"));
  }

  function sourceHasFeature(sourceId, featureName) {
    const source = dataSourceById(sourceId);
    return Boolean(source?.columns?.some((column) => column?.name === featureName));
  }

  function dataSourceById(sourceId) {
    const id = String(sourceId || "dataset");
    const sources = state.schema?.data_sources || [];
    const source = sources.find((item) => item?.id === id);
    if (source) return source;
    return id === "dataset" ? { columns: state.schema?.columns || [] } : null;
  }

  function goToLineBarFeature(name) {
    if (typeof navigateToLineBarFeature === "function" && navigateToLineBarFeature(name)) return;
    setGbmNotice(`Feature ${name} is not available in Line and Bar`);
  }

  function goToGbmShap(names) {
    const features = normaliseContextFeatureNames(names);
    if (!features.length) return;
    shapTool.preselectFeatures(features[0], features[1] || "");
    activeTab = "shap";
    render(config || {});
  }

  function goToGbmStackedShap(name) {
    if (!name) return;
    stackedShapTool.preselectFeature(name);
    activeTab = "stacked-shap";
    render(config || {});
  }

  function positionGbmFeatureContextMenu(menu, clientX, clientY) {
    const margin = 8;
    menu.style.left = "0px";
    menu.style.top = "0px";
    const rect = menu.getBoundingClientRect();
    const maxLeft = Math.max(margin, window.innerWidth - rect.width - margin);
    const maxTop = Math.max(margin, window.innerHeight - rect.height - margin);
    const left = Math.min(Math.max(margin, clientX), maxLeft);
    const top = Math.min(Math.max(margin, clientY), maxTop);
    menu.style.left = `${left}px`;
    menu.style.top = `${top}px`;
  }

  function handleGbmFeatureContextMenuPointerDown(event) {
    const menu = document.getElementById("gbmFeatureContextMenu");
    if (!menu || menu.hidden || menu.contains(event.target)) return;
    closeGbmFeatureContextMenu();
  }

  function handleGbmFeatureContextMenuKeydown(event) {
    if (event.key !== "Escape") return;
    event.preventDefault();
    closeGbmFeatureContextMenu();
  }

  function closeGbmFeatureContextMenu() {
    const menu = document.getElementById("gbmFeatureContextMenu");
    if (menu) menu.hidden = true;
    window.removeEventListener("pointerdown", handleGbmFeatureContextMenuPointerDown, true);
    window.removeEventListener("keydown", handleGbmFeatureContextMenuKeydown, true);
    window.removeEventListener("resize", closeGbmFeatureContextMenu, true);
    window.removeEventListener("scroll", closeGbmFeatureContextMenu, true);
  }

  function sortedFeatureRowsForMetric(features = []) {
    const field = featureMetricField();
    return [...(features || [])].sort((left, right) => {
      const leftValue = featureNumber(left?.[field]);
      const rightValue = featureNumber(right?.[field]);
      if (leftValue === null && rightValue === null) return String(left?.name || "").localeCompare(String(right?.name || ""), undefined, { sensitivity: "base" });
      if (leftValue === null) return 1;
      if (rightValue === null) return -1;
      return (rightValue - leftValue) || String(left?.name || "").localeCompare(String(right?.name || ""), undefined, { sensitivity: "base" });
    });
  }

  function featureMetricDisplay(feature) {
    return featureMetricMode === "shap" ? formatMeanAbsShap(feature?.mean_abs_shap) : formatGain(feature?.gain);
  }

  function renderParameterFallback(parameters) {
    const target = el("gbmParameterFallback");
    if (!target) return;
    target.innerHTML = `
      <table>
        <thead><tr><th>Parameter</th><th>Value</th></tr></thead>
        <tbody>
          ${parameters.map((parameter) => `
            <tr>
              <td>${escapeHtml(parameter.name)}</td>
              <td>${parameterControlHtml(parameter)}</td>
            </tr>
          `).join("")}
        </tbody>
      </table>
    `;
  }

  function parameterOptionsForName(name) {
    const parameterName = String(name || "");
    const configured = config?.parameter_options?.[parameterName];
    const options = Array.isArray(configured) && configured.length
      ? configured
      : GBM_PARAMETER_OPTIONS[parameterName] || [];
    return options.map((value) => String(value)).sort(compareParameterOption);
  }

  function compareParameterOption(left, right) {
    return left.localeCompare(right, undefined, { sensitivity: "base" });
  }

  function parameterValueEditorParams() {
    return {
      editorLookup: parameterValueEditorLookup,
      paramsLookup: parameterValueEditorParamsLookup,
    };
  }

  function parameterValueEditorLookup(cell) {
    const rowData = cell.getRow().getData();
    return parameterOptionsForName(rowData.name).length ? "list" : "input";
  }

  function parameterValueEditorParamsLookup(editor, cell) {
    const rowData = cell.getRow().getData();
    const label = String(rowData.name || "Parameter value");
    const elementAttributes = {
      "aria-label": label,
      class: `gbm-parameter-editor gbm-parameter-${editor}-editor`,
    };
    if (editor === "list") {
      return {
        values: parameterOptionsForName(rowData.name),
        autocomplete: true,
        freetext: true,
        listOnEmpty: true,
        elementAttributes,
      };
    }
    return {
      selectContents: true,
      elementAttributes,
    };
  }

  function parameterControlHtml(parameter) {
    const name = String(parameter.name || "");
    const value = String(parameter.value ?? "");
    const options = parameterOptionsForName(name);
    if (!options.length) {
      return `<input data-gbm-parameter="${escapeHtml(name)}" value="${escapeHtml(value)}" />`;
    }
    return `
      <select data-gbm-parameter="${escapeHtml(name)}" aria-label="${escapeHtml(name)}">
        ${options.map((option) => `<option value="${escapeHtml(option)}" ${option === value ? "selected" : ""}>${escapeHtml(option)}</option>`).join("")}
      </select>
    `;
  }

  function bindGridSampleInput() {
    const input = el("gbmGridSampleInput");
    if (!input) return;
    input.addEventListener("input", () => {
      gridSampleValue = currentGridSampleValue();
    });
  }

  function bindFallbackParameterGridSearchControls() {
    for (const input of document.querySelectorAll("[data-gbm-parameter]")) {
      input.addEventListener("input", syncGridSampleControl);
      input.addEventListener("change", syncGridSampleControl);
    }
  }

  function syncGridSampleControl() {
    const root = el("gbmGridSamples");
    const input = el("gbmGridSampleInput");
    if (!root) return;
    const show = hasGridParameters(currentParameters());
    root.classList.toggle("hidden", !show);
    if (input) input.value = String(currentGridSampleValue());
  }

  function hasGridParameters(parameters = []) {
    return parameters.some((parameter) => isGridParameterValue(parameter?.value));
  }

  function isGridParameterValue(value) {
    const text = String(value ?? "").trim();
    return /^\{[^{}]+\}$/.test(text);
  }

  function currentGridSampleValue() {
    const input = el("gbmGridSampleInput");
    const source = input ? input.value : gridSampleValue;
    const number = Number.parseInt(String(source ?? ""), 10);
    const value = Number.isFinite(number) && number > 0 ? number : GBM_GRID_SAMPLE_DEFAULT;
    gridSampleValue = value;
    return value;
  }

  function modelRows(models) {
    return uniqueModels(models.map(normaliseModel).filter((model) => model.model_id)).map((model) => ({
      ...model,
      model_label: modelLabel(model),
      created_sort: modelCreatedSort(model.created_at),
      created_display: formatModelCreated(model.created_at),
      weight_display: modelWeightLabel(model.offset_column),
      training_mode_display: model.training_mode === "ebm" ? "EBM" : "Normal",
      constraint_display: modelInteractionConstraintLabel(model.feature_interaction_constraints),
      best_training_metric: modelBestMetric(model, "training"),
      best_test_metric: modelBestMetric(model, "test"),
      param_num_iterations: modelParameterNumber(model, "num_iterations"),
      param_learning_rate: modelParameterNumber(model, "learning_rate"),
      param_num_leaves: modelParameterNumber(model, "num_leaves"),
      param_max_depth: modelParameterNumber(model, "max_depth"),
      param_min_data_in_leaf: modelParameterNumber(model, "min_data_in_leaf"),
      param_early_stopping_rounds: modelParameterNumber(model, "early_stopping_rounds"),
      runtime_seconds: modelRuntimeSeconds(model),
      runtime_display: formatModelRuntime(model),
      sample_display: formatSampleMode(model.sample_column, model.sample_source),
    }));
  }

  function modelInteractionConstraintLabel(rawConstraints) {
    const active = normaliseActiveFeatureInteractionConstraints(rawConstraints);
    const groupings = active.groups.map((group) => group.grouping);
    return groupings.length ? groupings.join(", ") : "No";
  }

  function renderModelFallback(models) {
    const target = el("gbmModelFallback");
    if (!target) return;
    if (!models.length) {
      target.innerHTML = `<div class="gbm-empty-state">No GBMs trained yet</div>`;
      return;
    }
    target.innerHTML = `
      <table class="gbm-model-table">
        <thead>
          <tr>
            <th class="gbm-model-active-heading" aria-label="Active model"></th>
            <th>Model</th>
            <th>Created</th>
            <th>Response</th>
            <th>Weight</th>
            <th>Objective</th>
            <th>Metric</th>
            <th>Mode</th>
            <th>Constraints</th>
            <th class="numeric">Train</th>
            <th class="numeric">Best iter.</th>
            <th class="numeric compact" title="Training metric at best iteration">tr@best</th>
            <th class="numeric compact" title="Test metric at best iteration">te@best</th>
            <th class="numeric compact" title="num_iterations">n_iter</th>
            <th class="numeric compact" title="learning_rate">lr</th>
            <th class="numeric compact" title="num_leaves">leaves</th>
            <th class="numeric compact" title="max_depth">depth</th>
            <th class="numeric compact" title="min_data_in_leaf">min_leaf</th>
            <th class="numeric compact" title="early_stopping_rounds">ES</th>
            <th class="numeric">Run time</th>
            <th>Sample</th>
          </tr>
        </thead>
        <tbody>
          ${models.map((model) => `
            <tr data-gbm-model-row="${escapeHtml(model.model_id)}" aria-selected="false">
              <td class="gbm-model-active-cell">
                ${model.active ? '<span class="gbm-model-active-dot" title="Active model" aria-label="Active model"></span>' : ""}
              </td>
              <td class="gbm-model-name-cell">
                <span class="gbm-model-name-main">${escapeHtml(model.model_label)}</span>
              </td>
              <td>${escapeHtml(model.created_display)}</td>
              <td>${escapeHtml(model.response_column)}</td>
              <td>${escapeHtml(model.weight_display)}</td>
              <td>${escapeHtml(model.objective || "")}</td>
              <td>${escapeHtml(model.metric || "")}</td>
              <td>${escapeHtml(model.training_mode_display)}</td>
              <td>${escapeHtml(model.constraint_display)}</td>
              <td class="numeric">${formatModelCount(model.training_rows)}</td>
              <td class="numeric">${formatModelCount(model.best_iteration)}</td>
              <td class="numeric">${escapeHtml(formatModelMetric(model.best_training_metric))}</td>
              <td class="numeric">${escapeHtml(formatModelMetric(model.best_test_metric))}</td>
              <td class="numeric">${escapeHtml(formatModelInteger(model.param_num_iterations))}</td>
              <td class="numeric">${escapeHtml(formatModelDecimal(model.param_learning_rate))}</td>
              <td class="numeric">${escapeHtml(formatModelInteger(model.param_num_leaves))}</td>
              <td class="numeric">${escapeHtml(formatModelInteger(model.param_max_depth))}</td>
              <td class="numeric">${escapeHtml(formatModelInteger(model.param_min_data_in_leaf))}</td>
              <td class="numeric">${escapeHtml(formatModelInteger(model.param_early_stopping_rounds))}</td>
              <td class="numeric">${escapeHtml(model.runtime_display)}</td>
              <td>${escapeHtml(model.sample_display)}</td>
            </tr>
          `).join("")}
        </tbody>
      </table>
    `;
    const rows = Array.from(target.querySelectorAll("[data-gbm-model-row]"));
    let anchorRow = null;
    const setSelected = (row, selected) => {
      row.classList.toggle("selected", selected);
      row.setAttribute("aria-selected", String(selected));
    };
    for (const row of rows) {
      row.addEventListener("click", (event) => {
        const commandSelection = event.metaKey || event.ctrlKey;
        if (event.shiftKey) {
          event.preventDefault();
          const anchor = anchorRow && rows.includes(anchorRow) ? anchorRow : row;
          const start = rows.indexOf(anchor);
          const end = rows.indexOf(row);
          const min = Math.min(start, end);
          const max = Math.max(start, end);
          if (!commandSelection) rows.forEach((candidate) => setSelected(candidate, false));
          for (let index = min; index <= max; index += 1) {
            const candidate = rows[index];
            if (commandSelection && candidate !== anchor) {
              setSelected(candidate, candidate.getAttribute("aria-selected") !== "true");
            } else {
              setSelected(candidate, true);
            }
          }
        } else if (commandSelection) {
          setSelected(row, row.getAttribute("aria-selected") !== "true");
        } else {
          rows.forEach((candidate) => setSelected(candidate, candidate === row));
        }
        anchorRow = row;
        syncModelActionButtons();
      });
    }
    syncModelActionButtons();
  }

  function formatModelCount(value) {
    const number = Number(value);
    return Number.isFinite(number) && number > 0 ? Math.round(number).toLocaleString() : "0";
  }

  function modelParameterNumber(model, name) {
    const parameters = model?.parameters && typeof model.parameters === "object" ? model.parameters : {};
    return modelNumberOrNull(parameters[name]);
  }

  function formatModelInteger(value) {
    const number = Number(value);
    return Number.isFinite(number) ? Math.round(number).toLocaleString() : "--";
  }

  function formatModelDecimal(value) {
    const number = Number(value);
    if (!Number.isFinite(number)) return "--";
    return number.toLocaleString(undefined, { maximumSignificantDigits: 4 });
  }

  function modelRuntimeSeconds(model) {
    const seconds = Number(model?.timings?.training_seconds ?? model?.training_seconds);
    return Number.isFinite(seconds) && seconds >= 0 ? seconds : -1;
  }

  function formatModelRuntime(model) {
    const seconds = modelRuntimeSeconds(model);
    if (seconds < 0) return "--";
    if (seconds < 1) return `${Math.round(seconds * 1000).toLocaleString()}ms`;
    if (seconds < 60) return `${seconds.toLocaleString(undefined, { maximumFractionDigits: 1 })}s`;
    const minutes = Math.floor(seconds / 60);
    const remainder = Math.round(seconds % 60);
    return `${minutes}m ${remainder.toString().padStart(2, "0")}s`;
  }

  function formatModelCreated(value) {
    if (!value) return "";
    const date = new Date(value);
    if (!Number.isFinite(date.getTime())) return String(value);
    const months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];
    const hour = String(date.getHours()).padStart(2, "0");
    const minute = String(date.getMinutes()).padStart(2, "0");
    return `${date.getDate()} ${months[date.getMonth()]} ${hour}:${minute}`;
  }

  function modelCreatedSort(value) {
    const time = new Date(value || "").getTime();
    return Number.isFinite(time) ? time : 0;
  }

  function formatSampleMode(value, source = "") {
    const text = String(value || "").trim();
    if (!text) return "All rows";
    if (String(source || "").trim() === "generated") return "Generated 60/20/20";
    return text;
  }

  function modelWeightLabel(value) {
    const text = String(value || "").trim();
    return !text || text === "__none__" || text === "Average row value" ? "N" : text;
  }

  function bindModelActions() {
    el("gbmRenameModelBtn")?.addEventListener("click", renameActiveModel);
    el("gbmActivateModelBtn")?.addEventListener("click", activateSelectedModel);
    el("gbmDeleteModelBtn")?.addEventListener("click", deleteActiveModel);
    syncModelActionButtons();
  }

  function syncModelActionButtons() {
    const selectedCount = selectedModelIds().length;
    const disableActions = isTraining;
    const rename = el("gbmRenameModelBtn");
    const activate = el("gbmActivateModelBtn");
    const del = el("gbmDeleteModelBtn");
    if (rename) rename.disabled = disableActions || selectedCount !== 1;
    if (activate) activate.disabled = disableActions || selectedCount !== 1;
    if (del) del.disabled = disableActions || selectedCount < 1;
  }

  async function refreshModelList({ force = false } = {}) {
    const now = Date.now();
    if (!force && now - modelListLastRefreshAt < GBM_MODEL_LIST_POLL_MS) return;
    modelListLastRefreshAt = now;
    const requestSeq = modelListRefreshSeq + 1;
    modelListRefreshSeq = requestSeq;
    try {
      const payload = await api("/api/gbm/models", { method: "GET", clientTiming: true });
      if (requestSeq !== modelListRefreshSeq) return;
      await applyModelListPayload(payload);
    } catch (error) {
      if (force) setGbmNotice(error.message);
    }
  }

  async function applyModelListPayload(payload = {}) {
    const activeModelId = String(payload?.active_model_id || "");
    const models = Array.isArray(payload?.models)
      ? payload.models.map((model) => ({
        ...model,
        active: Boolean(model?.active) || String(model?.model_id || "") === activeModelId,
      }))
      : [];
    config = config || {};
    config.models = models;
    config.active_model_id = activeModelId;
    setDatasetGbmCount(models.length);
    const cache = toolCache(tool);
    if (cache?.data) {
      cache.data = { ...cache.data, models, active_model_id: activeModelId };
    }
    syncSidebarModelChooser(models, activeModelId);
    if (activeTab === "models") {
      await refreshModelTableRows(modelRows(models));
    }
  }

  async function refreshModelTableRows(rows) {
    const selectedIds = selectedModelIds();
    const availableIds = new Set(rows.map((row) => String(row?.model_id || "")).filter(Boolean));
    const preservedIds = selectedIds.filter((id) => availableIds.has(id));
    if (modelTable && typeof modelTable.replaceData === "function") {
      await modelTable.replaceData(rows);
      restoreModelSelection(preservedIds);
      syncModelActionButtons();
      return;
    }
    renderModelFallback(rows);
    restoreModelSelection(preservedIds);
    syncModelActionButtons();
  }

  function restoreModelSelection(ids) {
    const selected = new Set((ids || []).map((id) => String(id || "")).filter(Boolean));
    if (modelTable && typeof modelTable.getRows === "function") {
      for (const row of modelTable.getRows()) {
        const rowId = String(row.getData()?.model_id || "");
        if (selected.has(rowId)) {
          row.select();
        } else {
          row.deselect();
        }
      }
      return;
    }
    for (const row of document.querySelectorAll("#gbmModelFallback [data-gbm-model-row]")) {
      const rowId = String(row.dataset.gbmModelRow || "");
      const active = selected.has(rowId);
      row.classList.toggle("selected", active);
      row.setAttribute("aria-selected", String(active));
    }
  }

  function selectedModelIds() {
    const ids = modelTable && typeof modelTable.getSelectedData === "function"
      ? modelTable.getSelectedData().map((row) => row?.model_id)
      : Array.from(document.querySelectorAll('#gbmModelFallback [data-gbm-model-row][aria-selected="true"]'))
        .map((row) => row.dataset.gbmModelRow);
    return [...new Set(ids.map((id) => String(id || "")).filter(Boolean))];
  }

  function currentActiveModelId() {
    return String(config?.active_model_id || (config?.models || []).find((model) => model.active)?.model_id || "");
  }

  function currentFeatureRows() {
    const reserved = currentReservedFeatureNames();
    function applyReserved(feature) {
      return reserved.has(feature.name) ? { ...feature, include: false } : feature;
    }
    if (featureTable) return featureTable.getData().map(applyReserved);
    return (config?.features || []).map((feature) => {
      const checkbox = document.querySelector(`[data-gbm-feature="${cssEscape(feature.name)}"]`);
      const monotonicity = document.querySelector(`[data-gbm-monotonicity="${cssEscape(feature.name)}"]`);
      const featureRow = document.querySelector(`[data-gbm-feature-row][data-gbm-feature-name="${cssEscape(feature.name)}"]`);
      return applyReserved({
        ...feature,
        include: checkbox ? checkbox.checked : feature.include,
        monotonicity: monotonicity ? monotonicity.value : feature.monotonicity,
        feature_interaction_locked: featureRow ? featureRow.dataset.gbmFeatureInteractionLocked === "true" : Boolean(feature.feature_interaction_locked),
      });
    });
  }

  function currentParameters() {
    if (parameterTable) return parameterTable.getData();
    return (config?.parameters || []).map((parameter) => {
      const input = document.querySelector(`[data-gbm-parameter="${cssEscape(parameter.name)}"]`);
      return { ...parameter, value: input ? input.value : parameter.value };
    });
  }

  function currentTrainingMode() {
    if (!config?.ebm_available) return "normal";
    return normaliseTrainingMode(document.querySelector("input[name='gbmTrainingMode']:checked")?.value || config?.training_mode);
  }

  async function train() {
    if (isTraining) return;
    setStatus("");
    setChartMessage("");
    const featureScenario = currentFeatureScenarioPayload();
    const featureInteractionGroupings = currentFeatureInteractionGroupingsPayload();
    const featureInteractionFeatures = currentFeatureInteractionFeaturesPayload();
    const payload = {
      label: `GBM ${gbmAutoModelTimeLabel()}`,
      response: el("actualNumerator")?.value || "actualNumerator",
      offset: el("denominator")?.value || "denominator",
      features: currentFeatureRows(),
      parameters: currentParameters(),
      shap_rows: document.querySelector("input[name='gbmShapRows']:checked")?.value || "100k",
      training_mode: currentTrainingMode(),
      sample_column: config?.sample?.column || config?.sample_column || "",
      sample_source: config?.sample?.source || "none",
      create_sample: false,
    };
    if (hasGridParameters(payload.parameters)) payload.grid_samples = currentGridSampleValue();
    if (featureScenario) payload.feature_scenario = featureScenario;
    if (featureInteractionGroupings) payload.feature_interaction_groupings = featureInteractionGroupings;
    if (featureInteractionFeatures) payload.feature_interaction_features = featureInteractionFeatures;
    try {
      const validation = await api("/api/gbm/validate", { method: "POST", body: JSON.stringify(payload) });
      if (!validation.ok) {
        setGbmNotice(validation.errors.join("; "));
        return;
      }
      gridTrainingNotice = gridValidationNotice(validation);
      setGbmNotice("");
      setGroupMeta(tool, "Training GBM...");
      startToolTiming(tool);
      setTrainingState(true);
      setAppReadyStatus(gbmTrainingReadyBadgeLabel());
      modelListLastRefreshAt = 0;
      liveEvaluationParameters = payload.parameters;
      liveProgress = null;
      setTrainingStatus("Training GBM...", "queued", gridTrainingNotice);
      const job = await api("/api/gbm/train", { method: "POST", body: JSON.stringify(payload), clientTiming: true });
      applyJobProgress(job);
      pollJob(job.job_id, 0);
    } catch (error) {
      setTrainingState(false);
      setAppReadyStatus("Ready");
      liveEvaluationParameters = null;
      gridTrainingNotice = "";
      setToolTimingFailed(tool);
      setTrainingStatus("");
      setGbmNotice(error.message);
    }
  }

  function gridValidationNotice(validation) {
    const messages = Array.isArray(validation?.grid?.messages) ? validation.grid.messages : [];
    return messages.join(" ");
  }

  function pollJob(jobId, delay = GBM_QUEUED_POLL_MS) {
    clearTimeout(pollTimer);
    pollTimer = setTimeout(async () => {
      try {
        const job = await api(`/api/gbm/jobs/${encodeURIComponent(jobId)}`, { method: "GET", clientTiming: true });
        applyJobProgress(job);
        if (job.status === "queued" || job.status === "running") {
          if (!job.progress) {
            const fallback = job.status === "queued" ? "GBM queued..." : "Training GBM...";
            setTrainingStatus(fallback, job.status, gridTrainingNotice);
            setGroupMeta(tool, fallback);
          }
          if (activeTab === "models") refreshModelList();
          pollJob(jobId, job.status === "running" ? GBM_RUNNING_POLL_MS : GBM_QUEUED_POLL_MS);
          return;
        }
        if (job.status === "failed") {
          modelListRefreshSeq += 1;
          setTrainingState(false);
          setAppReadyStatus("Ready");
          liveEvaluationParameters = null;
          gridTrainingNotice = "";
          setToolTimingFailed(tool);
          if (!job.progress) setTrainingStatus("GBM failed", "failed");
          setGbmNotice(job.error || "GBM training failed");
          setGroupMeta(tool, "GBM failed");
          return;
        }
        modelListRefreshSeq += 1;
        liveProgress = null;
        liveEvaluationParameters = null;
        gridTrainingNotice = "";
        await reloadSchema(job.result?.sources?.predictions, { modelKind: "gbm" });
        const preserveProfile = clearCachesAfterGbmModelSourceChange();
        const data = await api("/api/gbm/config", { method: "GET", clientTiming: true });
        const cache = toolCache(tool);
        cache.requestKey = stableConfigKey();
        cache.data = data;
        setTrainingState(false);
        setAppReadyStatus("Ready");
        setTrainingStatus("");
        measureToolRender(tool, () => render(data));
        if (!preserveProfile) refreshActiveTool({ force: true });
      } catch (error) {
        setTrainingState(false);
        setAppReadyStatus("Ready");
        liveEvaluationParameters = null;
        gridTrainingNotice = "";
        setToolTimingFailed(tool);
        setGbmNotice(error.message);
      }
    }, Math.max(0, delay));
  }

  function applyJobProgress(job) {
    if (!job?.progress) return;
    renderLiveProgress(job.progress);
  }

  function renderLiveProgress(progress) {
    liveProgress = progress;
    setAppReadyStatus(gbmTrainingReadyBadgeLabel(progress));
    setTrainingStatus(progress.message || "", progress.phase || "", trainingStatusDetail(progress));
    if (progress.message) setGroupMeta(tool, progress.message);
    if (progress.evaluation) {
      renderEvaluationChart({
        evaluation: progress.evaluation,
        progress,
        metric: progress.metric,
        parameters: liveEvaluationParameters || currentParameters(),
        manifest: {
          metric: progress.metric,
          best_iteration: progress.iteration,
        },
      });
    }
  }

  function trainingStatusDetail(progress) {
    const parts = [];
    const gridMessages = Array.isArray(progress?.grid?.messages) ? progress.grid.messages : [];
    if (gridMessages.length) parts.push(gridMessages.join(" "));
    const parameters = Array.isArray(progress?.grid_parameters) ? progress.grid_parameters : [];
    if (parameters.length) {
      parts.push(parameters.map((row) => `${row.label || row.name}=${row.value}`).join(" · "));
    }
    if (!parts.length && gridTrainingNotice) parts.push(gridTrainingNotice);
    return parts.join("  ");
  }

  async function activateModel(modelId) {
    if (isTraining) return;
    if (!modelId) return;
    try {
      const result = await api(`/api/gbm/models/${encodeURIComponent(modelId)}/activate`, { method: "POST", body: "{}" });
      await applyModelMutationResult(result);
    } catch (error) {
      setGbmNotice(error.message);
    }
  }

  async function activateSelectedModel() {
    if (isTraining) return;
    const modelIds = selectedModelIds();
    if (modelIds.length !== 1) return;
    await activateModel(modelIds[0]);
  }

  async function renameActiveModel() {
    if (isTraining) return;
    const [modelId] = selectedModelIds();
    if (!modelId) return;
    const newModelId = window.prompt("Rename GBM model", modelId);
    if (newModelId === null) return;
    const trimmed = newModelId.trim();
    if (!trimmed || trimmed === modelId) return;
    try {
      const result = await api(`/api/gbm/models/${encodeURIComponent(modelId)}/rename`, {
        method: "POST",
        body: JSON.stringify({ new_model_id: trimmed }),
      });
      await applyModelMutationResult(result);
    } catch (error) {
      setGbmNotice(error.message);
    }
  }

  async function deleteActiveModel() {
    if (isTraining) return;
    const modelIds = selectedModelIds();
    if (!modelIds.length) return;
    const label = modelIds.length === 1 ? `GBM model "${modelIds[0]}"` : `${modelIds.length} GBM models`;
    const confirmed = confirm(`Delete ${label}? This deletes the selected .lucidum model folder${modelIds.length === 1 ? "" : "s"}.`);
    if (!confirmed) return;
    let result = null;
    let deletedCount = 0;
    try {
      for (const modelId of modelIds) {
        result = await api(`/api/gbm/models/${encodeURIComponent(modelId)}`, { method: "DELETE", body: "{}" });
        deletedCount += 1;
      }
      await applyModelMutationResult(result);
    } catch (error) {
      try {
        const latest = await api("/api/gbm/config", { method: "GET", clientTiming: true });
        await applyModelMutationResult({ config: latest });
      } catch (_) {
        // Keep the original delete error visible when the refresh also fails.
      }
      const prefix = deletedCount > 0 ? `${deletedCount} deleted. ` : "";
      setGbmNotice(`${prefix}${error.message}`);
    }
  }

  async function applyModelMutationResult(result) {
    const nextConfig = result.config || config || {};
    await reloadSchema(preferredModelSource(result, nextConfig), { modelKind: "gbm" });
    const preserveProfile = clearCachesAfterGbmModelSourceChange();
    const currentModelId = featureDraftModelId(config);
    const nextModelId = featureDraftModelId(nextConfig);
    if (currentModelId !== nextModelId) featureDraftState = null;
    activeDetail = null;
    setGbmNotice("");
    if (state.tool === tool) {
      measureToolRender(tool, () => render(nextConfig));
    } else if (preserveProfile) {
      config = nextConfig;
      syncSidebarModelChooser(nextConfig?.models || [], nextConfig?.active_model_id);
    } else {
      config = nextConfig;
      syncSidebarModelChooser(nextConfig?.models || [], nextConfig?.active_model_id);
      await refreshActiveTool({ force: true });
    }
  }

  function clearCachesAfterGbmModelSourceChange() {
    const preserveProfile = state.tool === "column_profile";
    // GBM model changes update model-output sources. Source-scoped tools must refresh,
    // but Column Profile is raw-dataset + filter only, so its cache can survive.
    clearToolCaches(preserveProfile ? { preserve: ["column_profile"] } : {});
    return preserveProfile;
  }

  function preferredModelSource(result, data) {
    const currentKind = dataSourceById(state.source)?.kind || "";
    const configActiveModel = (data?.models || []).find((item) => item.active);
    const activeModel = result?.deleted_model_id
      ? null
      : (result?.model || configActiveModel);
    const shapSource = activeModel?.sources?.shap_long || configActiveModel?.sources?.shap_long || "";
    const predictionSource = activeModel?.sources?.predictions || configActiveModel?.sources?.predictions || "";
    if (currentKind === "gbm_shap_long") return shapSource || predictionSource || "dataset";
    if (currentKind === "gbm_predictions") return predictionSource || "dataset";
    if (!dataSourceById(state.source) && predictionSource) return predictionSource;
    const fallbackModel = (data?.models || []).find((item) => item.active);
    return dataSourceById(state.source) ? state.source || "dataset" : fallbackModel?.sources?.predictions || "dataset";
  }

  async function loadModelDetail(modelId) {
    try {
      activeDetail = await api(`/api/gbm/models/${encodeURIComponent(modelId)}`, { method: "GET" });
      renderEvaluationChart();
    } catch (_) {
      activeDetail = null;
    }
  }

  function renderEvaluationChart(source = null) {
    const target = el("gbmEvaluationChart");
    const detail = source || activeDetail;
    const evaluation = detail?.training_log?.evaluation || detail?.evaluation;
    if (!target || !window.echarts || !evaluation) return;
    const rows = [];
    for (const [datasetName, metrics] of Object.entries(evaluation)) {
      for (const [metricName, values] of Object.entries(metrics)) {
        rows.push({ datasetName, metricName, values });
      }
    }
    if (!rows.length) return;
    rows.sort(compareEvaluationRows);
    const metricNames = new Set(rows.map((row) => row.metricName));
    const primaryMetric = String(detail?.manifest?.metric || detail?.metric || rows[0]?.metricName || "metric");
    const maxIteration = Math.max(1, ...rows.map((row) => row.values.length));
    const xMax = evaluationXAxisMax(maxIteration, detail?.progress || null);
    const xDomain = evaluationXDomain(maxIteration, detail, xMax);
    const xInterval = niceIterationInterval(evaluationXDomainSpan(xDomain));
    const xLabelInterval = niceIterationLabelInterval(evaluationXDomainSpan(xDomain));
    const yAxisBounds = evaluationYAxisBounds(rows, maxIteration, primaryMetric, detail, xDomain);
    const sampledEvaluationIndexes = evaluationSampledIndexes(rows, maxIteration, detail?.manifest || {}, detail?.progress || null, xDomain);
    const title = evaluationTitle(rows, primaryMetric, detail?.manifest || {}, detail?.progress || null);
    const textColor = cssVar("--text", "#3f3f46");
    const mutedColor = cssVar("--muted", "#4b5563");
    const lineColor = cssVar("--line", "#e5e7eb");
    const panelColor = cssVar("--panel", "#ffffff");
    if (!evaluationChart) {
      evaluationChart = window.echarts.init(target);
      bindEvaluationResize(target);
    }
    evaluationChart.setOption({
      animation: false,
      color: ["#ff140f", cssVar("--actual-line", "#050505"), "#2563eb", "#7c3aed"],
      title: {
        text: title,
        left: "center",
        top: 8,
        textStyle: { color: textColor, fontSize: 12, fontWeight: 800, lineHeight: 15 },
      },
      legend: {
        orient: "vertical",
        right: 8,
        top: "middle",
        itemWidth: 10,
        itemHeight: 10,
        textStyle: { color: textColor, fontSize: 12 },
      },
      tooltip: {
        trigger: "axis",
        backgroundColor: panelColor,
        borderColor: lineColor,
        textStyle: { color: textColor },
        formatter: (params) => evaluationTooltipFormatter(params),
      },
      grid: { left: 12, right: 82, top: 42, bottom: 20, containLabel: true },
      xAxis: {
        type: "value",
        min: xDomain.min,
        max: xDomain.max,
        interval: xInterval,
        axisLabel: { color: mutedColor, hideOverlap: false, margin: 4, formatter: (value) => evaluationIterationAxisLabel(value, xLabelInterval) },
        axisLine: { lineStyle: { color: mutedColor } },
        splitLine: { lineStyle: { color: lineColor } },
      },
      yAxis: {
        type: "value",
        scale: true,
        ...yAxisBounds,
        splitNumber: 4,
        axisLabel: { color: mutedColor, formatter: (value) => formatEvaluationAxisValue(value) },
        axisLine: { show: true, lineStyle: { color: mutedColor } },
        splitLine: { lineStyle: { color: lineColor } },
      },
      series: rows.map((row) => ({
        name: evaluationSeriesName(row, metricNames.size > 1),
        type: "line",
        showSymbol: true,
        symbol: "circle",
        symbolSize: 4,
        lineStyle: { width: 2 },
        data: evaluationSeriesData(row.values, sampledEvaluationIndexes),
      })),
    }, true);
    requestAnimationFrame(() => evaluationChart?.resize());
  }

  function evaluationSeriesName(row, includeMetric) {
    const dataset = String(row.datasetName || "").toLowerCase() === "training" ? "train" : String(row.datasetName || "series");
    return includeMetric ? `${dataset} ${row.metricName}` : dataset;
  }

  function compareEvaluationRows(left, right) {
    const leftOrder = evaluationDatasetOrder(left.datasetName);
    const rightOrder = evaluationDatasetOrder(right.datasetName);
    if (leftOrder !== rightOrder) return leftOrder - rightOrder;
    return String(left.metricName).localeCompare(String(right.metricName));
  }

  function evaluationDatasetOrder(datasetName) {
    const name = String(datasetName || "").toLowerCase();
    if (name === "training" || name === "train") return 0;
    if (name === "test" || name === "validation" || name === "valid") return 1;
    return 2;
  }

  function evaluationTitle(rows, primaryMetric, manifest = {}, progress = null) {
    const bestIteration = Math.max(0, Number(manifest.best_iteration || 0));
    const metric = primaryMetric || rows[0]?.metricName || "metric";
    const testRow = rows.find((row) => row.datasetName === "test" && row.metricName === metric)
      || rows.find((row) => row.datasetName === "test")
      || rows.find((row) => row.metricName === metric)
      || rows[0];
    const livePoint = progress ? preferredLiveMetric(progress.latest || [], metric) : null;
    const liveValue = Number(livePoint?.value);
    const bestValue = Number.isFinite(liveValue) ? liveValue : valueAtIteration(testRow?.values || [], bestIteration) ?? lastFiniteValue(testRow?.values || []);
    const parts = [];
    parts.push(`evaluation metric: ${metric}`);
    if (bestValue !== null) parts.push(`test metric: ${formatEvaluationValue(bestValue)}`);
    if (progress?.iteration) {
      parts.push(`iteration: ${Number(progress.iteration).toLocaleString()}`);
    } else if (bestIteration) {
      parts.push(`best iteration: ${bestIteration.toLocaleString()}`);
    }
    return parts.join(", ");
  }

  function preferredLiveMetric(latest, metric) {
    if (!Array.isArray(latest) || !latest.length) return null;
    return [...latest].sort((left, right) => liveMetricSortKey(left, metric).localeCompare(liveMetricSortKey(right, metric)))[0];
  }

  function liveMetricSortKey(item, metric) {
    const dataset = String(item?.dataset || "").toLowerCase();
    const datasetRank = dataset === "test" ? "0" : ["validation", "valid"].includes(dataset) ? "1" : ["training", "train"].includes(dataset) ? "2" : "3";
    const metricRank = String(item?.metric || "") === String(metric || "") ? "0" : "1";
    return `${datasetRank}:${metricRank}:${item?.metric || ""}`;
  }

  function valueAtIteration(values, iteration) {
    if (!iteration || iteration < 1) return null;
    const value = values[iteration - 1];
    return Number.isFinite(Number(value)) ? Number(value) : null;
  }

  function lastFiniteValue(values) {
    for (let index = values.length - 1; index >= 0; index -= 1) {
      const value = Number(values[index]);
      if (Number.isFinite(value)) return value;
    }
    return null;
  }

  function evaluationXAxisMax(maxIteration, progress = null) {
    const liveTotal = progress?.phase === "training" ? Number(progress.total_iterations) : NaN;
    if (Number.isFinite(liveTotal) && liveTotal > 0) return Math.max(1, Math.round(liveTotal));
    return Math.max(1, Math.round(Number(maxIteration || 1)));
  }

  function evaluationXDomain(maxIteration, detail, xMax) {
    if (evaluationViewMode !== "tail") return { min: 0, max: xMax };
    const tailMax = Math.max(1, Math.round(Number(maxIteration || 1)));
    const width = evaluationTailWindowSize(tailMax, detail);
    return { min: Math.max(1, tailMax - width + 1), max: tailMax };
  }

  function evaluationXDomainSpan(domain) {
    return Math.max(1, Number(domain?.max || 1) - Number(domain?.min || 0));
  }

  function evaluationTailWindowSize(maxIteration, detail) {
    const count = Math.max(1, Math.round(Number(maxIteration || 1)));
    const bestIteration = Math.round(Number(detail?.manifest?.best_iteration || 0));
    const earlyStoppingRounds = evaluationEarlyStoppingRounds(detail);
    if (bestIteration >= 1 && earlyStoppingRounds > 0 && count - bestIteration >= earlyStoppingRounds) {
      return Math.min(count, Math.max(50, earlyStoppingRounds * 5));
    }
    return Math.min(count, Math.max(50, Math.ceil(count * 0.2)));
  }

  function evaluationEarlyStoppingRounds(detail) {
    const value = evaluationParameterValue(detail?.parameters, "early_stopping_rounds");
    const rounds = Math.round(Number(value));
    return Number.isFinite(rounds) && rounds > 0 ? rounds : 0;
  }

  function evaluationParameterValue(parameters, name) {
    if (Array.isArray(parameters)) {
      return parameters.find((parameter) => String(parameter?.name || "") === name)?.value;
    }
    if (parameters && typeof parameters === "object") return parameters[name];
    return null;
  }

  function evaluationYAxisBounds(rows, maxIteration, primaryMetric, detail, xDomain) {
    if (evaluationViewMode === "tail") return evaluationTailYAxisBounds(rows, primaryMetric, xDomain);
    const yMax = evaluationYAxisMax(rows, maxIteration);
    return yMax !== null ? { max: yMax } : {};
  }

  function evaluationTailYAxisBounds(rows, primaryMetric, xDomain) {
    const row = evaluationTailFocusRow(rows, primaryMetric);
    const values = Array.isArray(row?.values) ? row.values : [];
    const startIndex = Math.max(0, Math.ceil(Number(xDomain?.min || 1)) - 1);
    const endIndex = Math.min(values.length - 1, Math.floor(Number(xDomain?.max || values.length)) - 1);
    const extent = evaluationEmptyExtent();
    for (let index = startIndex; index <= endIndex; index += 1) {
      const value = Number(values[index]);
      if (Number.isFinite(value)) updateEvaluationExtent(extent, value);
    }
    if (!extent.count) return {};
    const padding = evaluationTailYAxisPadding(extent);
    return { min: extent.min - padding, max: extent.max + padding };
  }

  function evaluationTailYAxisPadding(extent) {
    const range = Math.max(0, extent.max - extent.min);
    if (range > 0) return Math.max(range * 0.2, 1e-9);
    return Math.max(Math.abs(extent.max), 1) * 0.0001;
  }

  function evaluationTailFocusRow(rows, primaryMetric) {
    const metric = String(primaryMetric || "");
    return rows.find((row) => String(row.datasetName || "").toLowerCase() === "test" && String(row.metricName || "") === metric)
      || rows.find((row) => String(row.datasetName || "").toLowerCase() === "test")
      || rows.find((row) => ["training", "train"].includes(String(row.datasetName || "").toLowerCase()) && String(row.metricName || "") === metric)
      || rows.find((row) => ["training", "train"].includes(String(row.datasetName || "").toLowerCase()))
      || rows.find((row) => String(row.metricName || "") === metric)
      || rows[0];
  }

  function evaluationYAxisMax(rows, maxIteration) {
    if (maxIteration < 50) return null;
    const tailStart = evaluationTailStart(maxIteration);
    const initialExtent = evaluationEmptyExtent();
    const tailExtent = evaluationEmptyExtent();
    for (const row of rows) {
      const values = Array.isArray(row.values) ? row.values : [];
      values.forEach((value, index) => {
        const number = Number(value);
        if (!Number.isFinite(number)) return;
        if (index < tailStart) {
          updateEvaluationExtent(initialExtent, number);
        } else {
          updateEvaluationExtent(tailExtent, number);
        }
      });
    }
    if (!initialExtent.count || !tailExtent.count) return null;
    const initialMax = initialExtent.max;
    const tailMin = tailExtent.min;
    const tailMax = tailExtent.max;
    const tailRange = Math.max(0, tailMax - tailMin);
    const materialGap = Math.max(tailRange * 2, Math.abs(tailMax) * 0.03, 1e-9);
    if (initialMax <= tailMax + materialGap) return null;
    const padding = Math.max(tailRange * 0.12, Math.abs(tailMax) * 0.01, 1e-9);
    return tailMax + padding;
  }

  function evaluationTailStart(maxIteration) {
    return Math.max(10, Math.min(300, Math.floor(Number(maxIteration || 0) * 0.08)));
  }

  function evaluationTooltipFormatter(params) {
    const items = Array.isArray(params) ? params : [params].filter(Boolean);
    const first = items[0] || {};
    const rawIteration = Number(first.axisValue ?? (Array.isArray(first.value) ? first.value[0] : null));
    const iteration = Number.isFinite(rawIteration) ? Math.round(rawIteration).toLocaleString() : "";
    const lines = [`<strong>Iteration:</strong> ${escapeHtml(iteration)}`];
    for (const item of items) {
      const value = Array.isArray(item?.value) ? item.value[1] : item?.value;
      lines.push(`${item?.marker || ""}${escapeHtml(item?.seriesName || "series")}: ${escapeHtml(formatEvaluationValue(value))}`);
    }
    return lines.join("<br/>");
  }

  function evaluationSampledIndexes(rows, maxIteration, manifest = {}, progress = null, xDomain = null) {
    const count = Math.max(1, Math.round(Number(maxIteration || 1)));
    const range = evaluationIndexRange(count, xDomain);
    const visibleCount = range.end - range.start + 1;
    if (visibleCount <= GBM_EVALUATION_DOWNSAMPLE_THRESHOLD) return sequentialIndexes(range.start, range.end);
    const requiredIndexes = requiredEvaluationIndexes(count, manifest, progress, range);
    const compositePoints = evaluationCompositePoints(rows, range.start, range.end);
    if (compositePoints.length <= GBM_EVALUATION_MAX_PLOT_POINTS) {
      return mergeEvaluationIndexes(compositePoints.map((point) => point.index), requiredIndexes);
    }
    const samplingLimit = Math.max(2, GBM_EVALUATION_MAX_PLOT_POINTS - requiredIndexes.size);
    const sampled = largestTriangleThreeBuckets(compositePoints, samplingLimit).map((point) => point.index);
    return mergeEvaluationIndexes(sampled, requiredIndexes);
  }

  function evaluationIndexRange(maxIteration, xDomain = null) {
    const end = Math.max(0, Math.min(maxIteration - 1, Math.floor(Number(xDomain?.max || maxIteration)) - 1));
    const start = Math.max(0, Math.min(end, Math.ceil(Number(xDomain?.min || 1)) - 1));
    return { start, end };
  }

  function sequentialIndexes(start, end) {
    const count = Math.max(0, end - start + 1);
    return Array.from({ length: count }, (_value, offset) => start + offset);
  }

  function requiredEvaluationIndexes(maxIteration, manifest = {}, progress = null, range = null) {
    const required = new Set([0, Math.max(0, maxIteration - 1)]);
    const bestIteration = Math.round(Number(manifest?.best_iteration || 0));
    const liveIteration = Math.round(Number(progress?.iteration || 0));
    for (const iteration of [bestIteration, liveIteration]) {
      if (iteration >= 1 && iteration <= maxIteration) required.add(iteration - 1);
    }
    if (!range) return required;
    return new Set([...required].filter((index) => index >= range.start && index <= range.end));
  }

  function evaluationCompositePoints(rows, startIndex, endIndex) {
    const stats = rows.map((row) => evaluationRowStats(row.values));
    const points = [];
    for (let index = startIndex; index <= endIndex; index += 1) {
      let total = 0;
      let count = 0;
      rows.forEach((row, rowIndex) => {
        const value = Number(row.values?.[index]);
        const stat = stats[rowIndex];
        if (!Number.isFinite(value) || !stat) return;
        total += stat.range > 0 ? (value - stat.min) / stat.range : 0.5;
        count += 1;
      });
      if (count) points.push({ index, x: index, y: total / count });
    }
    return points;
  }

  function evaluationRowStats(values) {
    const extent = evaluationEmptyExtent();
    for (const value of Array.isArray(values) ? values : []) {
      const number = Number(value);
      if (Number.isFinite(number)) updateEvaluationExtent(extent, number);
    }
    if (!extent.count) return null;
    return { min: extent.min, max: extent.max, range: extent.max - extent.min };
  }

  function evaluationEmptyExtent() {
    return { count: 0, min: Infinity, max: -Infinity };
  }

  function updateEvaluationExtent(extent, value) {
    extent.count += 1;
    if (value < extent.min) extent.min = value;
    if (value > extent.max) extent.max = value;
  }

  function largestTriangleThreeBuckets(points, threshold) {
    if (threshold >= points.length || threshold <= 2) return points.slice();
    const sampled = [points[0]];
    let anchorIndex = 0;
    const bucketSize = (points.length - 2) / (threshold - 2);
    for (let bucket = 0; bucket < threshold - 2; bucket += 1) {
      const bucketStart = Math.floor((bucket + 0) * bucketSize) + 1;
      const bucketEnd = Math.floor((bucket + 1) * bucketSize) + 1;
      const nextBucketStart = Math.floor((bucket + 1) * bucketSize) + 1;
      const nextBucketEnd = Math.floor((bucket + 2) * bucketSize) + 1;
      const average = averageEvaluationPoint(points.slice(nextBucketStart, Math.min(nextBucketEnd, points.length)));
      const anchor = points[anchorIndex];
      let maxArea = -1;
      let nextAnchorIndex = bucketStart;
      for (let index = bucketStart; index < Math.min(bucketEnd, points.length - 1); index += 1) {
        const point = points[index];
        const area = Math.abs((anchor.x - average.x) * (point.y - anchor.y) - (anchor.x - point.x) * (average.y - anchor.y)) * 0.5;
        if (area > maxArea) {
          maxArea = area;
          nextAnchorIndex = index;
        }
      }
      sampled.push(points[nextAnchorIndex]);
      anchorIndex = nextAnchorIndex;
    }
    sampled.push(points[points.length - 1]);
    return sampled;
  }

  function averageEvaluationPoint(points) {
    if (!points.length) return { x: 0, y: 0 };
    const totals = points.reduce((accumulator, point) => ({
      x: accumulator.x + point.x,
      y: accumulator.y + point.y,
    }), { x: 0, y: 0 });
    return { x: totals.x / points.length, y: totals.y / points.length };
  }

  function mergeEvaluationIndexes(sampledIndexes, requiredIndexes) {
    return [...new Set([...sampledIndexes, ...requiredIndexes])]
      .filter((index) => Number.isInteger(index) && index >= 0)
      .sort((left, right) => left - right);
  }

  function evaluationSeriesData(values, sampledIndexes) {
    const seriesValues = Array.isArray(values) ? values : [];
    return sampledIndexes
      .filter((index) => index < seriesValues.length)
      .map((index) => [index + 1, seriesValues[index]]);
  }

  function niceIterationInterval(maxIteration) {
    return niceIterationStep(Math.max(1, Number(maxIteration || 1) / 30));
  }

  function niceIterationLabelInterval(maxIteration) {
    return niceIterationStep(Math.max(1, Number(maxIteration || 1) / 10));
  }

  function niceIterationStep(rawStep) {
    const raw = Math.max(1, Number(rawStep || 1));
    const magnitude = 10 ** Math.floor(Math.log10(raw));
    for (const step of [1, 2, 5, 10]) {
      const interval = step * magnitude;
      if (interval >= raw) return interval;
    }
    return 10 * magnitude;
  }

  function evaluationIterationAxisLabel(value, labelInterval) {
    const number = Number(value);
    if (!Number.isFinite(number)) return "";
    const iteration = Math.round(number);
    if (Math.abs(number - iteration) > 1e-6) return "";
    const interval = Math.max(1, Math.round(Number(labelInterval || 1)));
    return iteration % interval === 0 ? iteration.toLocaleString() : "";
  }

  function formatEvaluationAxisValue(value) {
    const number = Number(value);
    if (!Number.isFinite(number)) return "";
    const abs = Math.abs(number);
    if (abs >= 1000) return Math.round(number).toLocaleString();
    if (abs >= 10) return number.toLocaleString(undefined, { maximumFractionDigits: 1 });
    if (abs >= 1) return number.toLocaleString(undefined, { maximumFractionDigits: 3 });
    return number.toLocaleString(undefined, { maximumFractionDigits: 4 });
  }

  function bindEvaluationResize(target) {
    if (!window.ResizeObserver) return;
    evaluationResizeObserver = new ResizeObserver(() => {
      evaluationChart?.resize();
    });
    evaluationResizeObserver.observe(target);
    if (target.parentElement) evaluationResizeObserver.observe(target.parentElement);
  }

  function disposeEvaluationChart() {
    evaluationResizeObserver?.disconnect();
    evaluationResizeObserver = null;
    if (evaluationChart) {
      evaluationChart.dispose();
      evaluationChart = null;
    }
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

  function formatEbmSummaryGain(value) {
    const number = Number(value || 0);
    if (!Number.isFinite(number)) return "0";
    return Math.round(number).toLocaleString();
  }

  function formatGainPercent(value) {
    const number = Number(value || 0);
    if (!Number.isFinite(number)) return "0.0%";
    return `${number.toFixed(1)}%`;
  }

  function formatMeanAbsShap(value) {
    const number = featureNumber(value);
    return number === null ? "" : number.toFixed(4);
  }

  function stableConfigKey() {
    return JSON.stringify(buildRequest());
  }

  function cssEscape(value) {
    return window.CSS?.escape ? window.CSS.escape(value) : String(value).replace(/"/g, '\\"');
  }

  function cssVar(name, fallback) {
    return getComputedStyle(document.body).getPropertyValue(name).trim() || fallback;
  }

  function loadTabulator() {
    if (window.Tabulator) return Promise.resolve(window.Tabulator);
    if (tabulatorPromise) return tabulatorPromise;
    tabulatorPromise = new Promise((resolve, reject) => {
      const cssHref = "/static/vendor/tabulator/tabulator.min.css";
      if (![...document.styleSheets].some((sheet) => sheet.href?.endsWith(cssHref))) {
        const link = document.createElement("link");
        link.rel = "stylesheet";
        link.href = cssHref;
        document.head.append(link);
      }
      const script = document.createElement("script");
      script.src = "/static/vendor/tabulator/tabulator.min.js";
      script.onload = () => window.Tabulator ? resolve(window.Tabulator) : reject(new Error("Tabulator did not load"));
      script.onerror = () => reject(new Error("Tabulator did not load"));
      document.head.append(script);
    });
    return tabulatorPromise;
  }

  return {
    buildRequest,
    fetchData,
    openModelNavigator,
    render,
    refreshTheme() {
      if (liveProgress?.evaluation) {
        renderLiveProgress(liveProgress);
      } else {
        renderEvaluationChart();
      }
      treeViewer.refreshTheme();
      shapTool.refreshTheme();
      stackedShapTool.refreshTheme();
    },
    syncSidebarFromSchema,
    useCached,
  };
}
