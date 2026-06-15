import { createGbmTreeViewer } from "./gbm-tree-viewer.js";
import { createGbmEvaluationChart } from "./gbm-evaluation-chart.js";
import { createGbmParameterControls } from "./gbm-feature-parameter-controls.js";
import { createGbmModelNavigator } from "./gbm-model-navigator.js";
import { createGbmShapTool } from "./gbm-shap-tool.js";
import { createGbmStackedShapTool } from "./gbm-stacked-shap-tool.js";
import { bindGbmTabs, gbmPanelClass, gbmTabsHtml, syncGbmRenderedTab } from "./gbm-tab-orchestration.js";
import { loadTabulator } from "./shared/tabulator.js";
import {
  createSidebarModelHeading,
  createSidebarModelOption,
  emptyStateHtml,
  formatModelMetric as sharedFormatModelMetric,
  isModelJobPending,
  modelGroups,
  modelJobPollDelay,
  modelNumberOrNull as sharedModelNumberOrNull,
  restoreModelSelection as restoreSharedModelSelection,
  selectedModelIdsFromTableOrFallback,
  setInlinePhaseStatus,
  syncCollapsedModelGroups,
  syncModelActionButtons as syncSharedModelActionButtons,
  toggleSidebarModelGroup,
} from "./shared/model-ui.js";

const GBM_RUNNING_POLL_MS = 500;
const GBM_QUEUED_POLL_MS = 1000;
const GBM_MODEL_LIST_POLL_MS = 2000;
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
  return sharedModelNumberOrNull(value);
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
  return sharedFormatModelMetric(value);
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
  selectExpectedPredictionForModelKind = () => false,
  syncClientTimingFromData,
  syncDuckDbTimingFromData,
  toolCache,
  updateAxisControls,
  refreshActiveTool,
  reloadSchema,
  onExternalModelActivation = async () => false,
}) {
  const tool = "gbm";
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
  const evaluationChart = createGbmEvaluationChart({ escapeHtml, formatEvaluationValue });
  const parameterControls = createGbmParameterControls({
    escapeHtml,
    parameterOptions: () => config?.parameter_options || {},
  });
  const modelNavigator = createGbmModelNavigator({
    escapeHtml,
    formatModelMetric,
    modelInteractionConstraintLabel,
    modelLabel,
    normaliseModel,
    uniqueModels,
    onFallbackSelectionChange: () => syncModelActionButtons(),
  });
  let featureMetricMode = "gain";
  let featureMetricModelId = "";
  let featureToolbarOutsideClickBound = false;
  let featureDraftState = null;
  let featureInteractionPairEditModelId = "";
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
      syncDatasetGbmCountFromConfig(data);
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

  function syncDatasetGbmCountFromConfig(data = {}) {
    setDatasetGbmCount(Array.isArray(data?.models) ? data.models.length : null);
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
    evaluationChart.dispose();
    treeViewer.dispose();
    shapTool.dispose();
    stackedShapTool.dispose();
    mount.innerHTML = `
      <div class="gbm-tool">
        <div id="gbmNotice" class="gbm-notice hidden" role="alert" aria-live="polite"></div>
        <div class="gbm-toolbar">
          <div class="gbm-tabs tabs workspace-tabs">
            ${gbmTabsHtml(activeTab)}
          </div>
          <div id="gbmTrainingStatus" class="gbm-training-status ${liveProgress ? "" : "hidden"}" aria-live="polite">${trainingStatusHtml(liveProgress)}</div>
        </div>
        <div class="${gbmPanelClass(activeTab, "features")}" data-gbm-panel="features">
          <div class="gbm-feature-layout">
            <section class="gbm-panel-section gbm-grid-panel">
              <div class="gbm-section-header gbm-feature-section-header">
                <h3 id="gbmFeatureSectionTitle" class="gbm-section-title">${escapeHtml(featureSectionTitle(data.features || []))}</h3>
                <div class="gbm-feature-actions" role="group" aria-label="Feature selection">
                  ${featureMetricToggleHtml(data.features || [], data)}
                  ${featureInteractionConstraintDropdownHtml(data.feature_interaction_groupings || [], data.active_feature_interaction_constraints || null, data.features || [])}
                  ${featureInteractionPairsDropdownHtml(data.active_feature_interaction_constraints || null, data.features || [])}
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
        <div class="${gbmPanelClass(activeTab, "models")}" data-gbm-panel="models">
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
        <div class="${gbmPanelClass(activeTab, "trees")}" data-gbm-panel="trees">
          <div id="gbmTreeViewer" class="gbm-tree-viewer">
            <section class="gbm-panel-section gbm-tree-summary-panel">
              <div class="gbm-tree-section-header">
                <h3 class="gbm-section-title">Select tree</h3>
                <input id="gbmTreeSearch" class="gbm-tree-search" type="search" placeholder="Search" aria-label="Search trees" />
              </div>
              <div id="gbmTreeSummaryGrid" class="gbm-grid gbm-tree-summary-grid"></div>
              <div id="gbmTreeSummaryFallback" class="gbm-fallback-table"></div>
            </section>
            <div id="gbmTreeResizer" class="gbm-tree-resizer app-resizer app-resizer--vertical" role="separator" aria-orientation="vertical" aria-label="Resize tree selector"></div>
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
        <div class="${gbmPanelClass(activeTab, "shap")}" data-gbm-panel="shap">
          ${shapTool.shellHtml()}
        </div>
        <div class="${gbmPanelClass(activeTab, "stacked-shap")}" data-gbm-panel="stacked-shap">
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
    const selected = normaliseEvaluationViewMode(evaluationChart.getViewMode());
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
    bindGbmTabs(mount, selectTab);
  }

  function selectTab(nextTab) {
    closeGbmFeatureContextMenu();
    activeTab = nextTab;
    render(config || {});
    if (nextTab === "models") refreshModelList({ force: true });
  }

  function syncRenderedTab(mount, nextTab) {
    syncGbmRenderedTab(mount, nextTab);
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
      if (featureDraftState && featureDraftState.modelId !== featureDraftModelId(nextData)) {
        featureDraftState = null;
        featureInteractionPairEditModelId = "";
      }
      return;
    }
    const currentModelId = featureDraftModelId(config);
    const nextModelId = featureDraftModelId(nextData);
    if (currentModelId !== nextModelId) {
      featureDraftState = null;
      featureInteractionPairEditModelId = "";
      return;
    }
    const rows = currentFeatureRows();
    if (!rows.length) return;
    const interactionGroupings = currentFeatureInteractionGroupings();
    const interactionPairs = currentFeatureInteractionPairs();
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
      interactionPairs,
      interactionPairsEdited: featureInteractionPairsEdited(interactionPairs, config),
      scenarioName,
      scenarioEdited: featureScenarioSelectionEdited(scenarioName, config),
    };
  }

  function featureDraftForData(data = config) {
    return featureDraftState && featureDraftState.modelId === featureDraftModelId(data) ? featureDraftState : null;
  }

  function markFeatureInteractionPairsEdited(pairs, features = currentFeatureRows()) {
    const existing = featureDraftForData(config);
    const rows = Array.isArray(features) && features.length ? features : currentFeatureRows();
    const interactionGroupings = existing?.interactionGroupings || currentFeatureInteractionGroupings();
    const scenarioName = existing?.scenarioName ?? (el("gbmFeatureScenarioDropdown")?.dataset.gbmSelectedFeatureScenario || "");
    featureDraftState = {
      modelId: featureDraftModelId(config),
      features: rows.map((feature) => ({
        name: feature.name,
        include: Boolean(feature.include),
        monotonicity: feature.monotonicity || "",
        feature_interaction_locked: Boolean(feature.feature_interaction_locked),
      })),
      interactionGroupings,
      interactionGroupingsEdited: existing?.interactionGroupingsEdited ?? featureInteractionGroupingsEdited(interactionGroupings, config),
      interactionPairs: normaliseFeatureInteractionPairs(pairs),
      interactionPairsEdited: true,
      scenarioName,
      scenarioEdited: existing?.scenarioEdited ?? featureScenarioSelectionEdited(scenarioName, config),
    };
    featureInteractionPairEditModelId = featureDraftModelId(config);
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
    bindFeatureInteractionPairActions();
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
        <button id="gbmFeatureInteractionConstraintButton" class="gbm-feature-menu-button gbm-interaction-constraint-button${constraintClass}" type="button" aria-haspopup="true" aria-expanded="false" aria-label="Constraint groups" title="Constrain selected grouped features so they only interact within selected groups" data-gbm-feature-menu-button${disabled}>${escapeHtml(featureInteractionButtonLabel(selectedCurrent.size, synthetic.length))}</button>
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

  function featureInteractionPairsDropdownHtml(activeConstraints = null, features = []) {
    const active = normaliseActiveFeatureInteractionConstraints(activeConstraints);
    const draft = featureDraftForData(config);
    const pairFeatures = pairCandidateFeatureRows(features);
    const sourcePairs = featureInteractionPairsUserEdited(config) ? draft.interactionPairs || [] : active.pairs;
    const pairs = normaliseFeatureInteractionPairs(sourcePairs);
    const pairClass = pairs.length ? " has-constraints" : "";
    return `
      <div id="gbmFeatureInteractionPairSelect" class="gbm-interaction-pair-select" data-gbm-feature-menu-root>
        <button id="gbmFeatureInteractionPairButton" class="gbm-feature-menu-button gbm-interaction-pair-button${pairClass}" type="button" aria-haspopup="true" aria-expanded="false" aria-label="Interaction pairs" title="Allow only these selected feature pairs to interact" data-gbm-feature-menu-button>${escapeHtml(featureInteractionPairButtonLabel(pairs.length))}</button>
        <div id="gbmFeatureInteractionPairMenu" class="gbm-feature-menu gbm-interaction-pair-menu hidden" role="menu" data-gbm-feature-menu>
          <div class="gbm-interaction-pair-builder">
            <select id="gbmInteractionPairLeft" class="gbm-interaction-pair-feature" aria-label="Interaction pair feature A">
              ${featureInteractionPairOptions(pairFeatures, pairFeatures[0]?.name || "")}
            </select>
            <span class="gbm-interaction-pair-times">x</span>
            <select id="gbmInteractionPairRight" class="gbm-interaction-pair-feature" aria-label="Interaction pair feature B">
              ${featureInteractionPairOptions(pairFeatures, pairFeatures[1]?.name || pairFeatures[0]?.name || "")}
            </select>
            <button id="gbmInteractionPairAdd" class="tab gbm-inline-action-button gbm-interaction-pair-add" type="button">Add pair</button>
          </div>
          <div id="gbmInteractionPairRows" class="gbm-interaction-pair-rows">
            ${featureInteractionPairRowsHtml(pairs)}
          </div>
        </div>
      </div>
    `;
  }

  function featureInteractionPairOptions(features = [], selected = "") {
    return pairCandidateFeatureRows(features)
      .map((feature) => `<option value="${escapeHtml(feature.name)}" ${feature.name === selected ? "selected" : ""}>${escapeHtml(feature.name)}</option>`)
      .join("");
  }

  function featureInteractionPairRowsHtml(pairs = []) {
    const rows = normaliseFeatureInteractionPairs(pairs);
    if (!rows.length) return '<div class="gbm-interaction-pair-empty">No interaction pairs</div>';
    return rows.map((pair) => `
      <div class="gbm-interaction-pair-row" data-gbm-interaction-pair-row data-gbm-interaction-pair-left="${escapeHtml(pair.left)}" data-gbm-interaction-pair-right="${escapeHtml(pair.right)}">
        <span class="gbm-interaction-pair-label">${escapeHtml(pair.left)} x ${escapeHtml(pair.right)}</span>
        <button class="gbm-interaction-pair-remove" type="button" aria-label="Remove ${escapeHtml(pair.left)} x ${escapeHtml(pair.right)}" title="Remove pair" data-gbm-remove-interaction-pair>×</button>
      </div>
    `).join("");
  }

  function bindFeatureInteractionPairActions() {
    const root = el("gbmFeatureInteractionPairSelect");
    if (!root) return;
    bindGbmFeatureToolbarMenu(root, {
      beforeOpen: () => syncFeatureInteractionPairControls(currentFeatureRows()),
    });
    root.addEventListener("click", (event) => {
      const add = event.target?.closest?.("#gbmInteractionPairAdd");
      if (add) {
        addFeatureInteractionPair(el("gbmInteractionPairLeft")?.value || "", el("gbmInteractionPairRight")?.value || "");
        return;
      }
      const remove = event.target?.closest?.("[data-gbm-remove-interaction-pair]");
      if (!remove) return;
      const row = remove.closest("[data-gbm-interaction-pair-row]");
      const key = featureInteractionPairKey(row?.dataset?.gbmInteractionPairLeft || "", row?.dataset?.gbmInteractionPairRight || "");
      setFeatureInteractionPairs(currentFeatureInteractionPairs().filter((pair) => featureInteractionPairKey(pair.left, pair.right) !== key));
    }, true);
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
    if (!activeConstraints || typeof activeConstraints !== "object") return { mode: "", groups: [], features: [], pairs: [] };
    const groups = Array.isArray(activeConstraints.groups) ? activeConstraints.groups : [];
    const pairs = normaliseFeatureInteractionPairs(activeConstraints.pairs);
    return {
      mode: String(activeConstraints.mode || "").trim().toLowerCase(),
      features: scenarioFeatureList(activeConstraints.features),
      pairs,
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

  function normaliseFeatureInteractionPairs(rawPairs) {
    if (!Array.isArray(rawPairs)) return [];
    const pairs = [];
    const seen = new Set();
    for (const item of rawPairs) {
      const left = String(item?.left || "").trim();
      const right = String(item?.right || "").trim();
      if (!left || !right || left === right) continue;
      const key = featureInteractionPairKey(left, right);
      if (seen.has(key)) continue;
      pairs.push({ left, right });
      seen.add(key);
    }
    return pairs;
  }

  function featureInteractionPairKey(left, right) {
    return [String(left || "").trim(), String(right || "").trim()]
      .filter(Boolean)
      .sort((a, b) => a.localeCompare(b, undefined, { sensitivity: "base" }) || a.localeCompare(b))
      .join("\u0000");
  }

  function selectedPairFeatureRows(features = currentFeatureRows()) {
    return (features || [])
      .filter((feature) => feature?.include && isFeatureSelectable(feature))
      .sort((left, right) => String(left?.name || "").localeCompare(String(right?.name || ""), undefined, { sensitivity: "base" }));
  }

  function pairCandidateFeatureRows(features = currentFeatureRows()) {
    return (features || [])
      .filter((feature) => isFeaturePairCandidate(feature))
      .sort((left, right) => String(left?.name || "").localeCompare(String(right?.name || ""), undefined, { sensitivity: "base" }));
  }

  function isFeaturePairCandidate(feature) {
    if (!feature || isInvalidFeature(feature) || feature.usable === false) return false;
    const kind = String(feature.kind || "");
    if (!["integer", "numeric", "categorical"].includes(kind)) return false;
    return !currentReservedFeatureNames().has(feature.name);
  }

  function currentFeatureInteractionPairFeatureNames() {
    const names = new Set();
    for (const pair of currentFeatureInteractionPairs()) {
      names.add(pair.left);
      names.add(pair.right);
    }
    return names;
  }

  function pruneFeatureInteractionPairs(pairs = [], features = currentFeatureRows(), { requireSelected = true } = {}) {
    const allowedRows = requireSelected ? selectedPairFeatureRows(features) : pairCandidateFeatureRows(features);
    const allowed = new Set(allowedRows.map((feature) => feature.name));
    return normaliseFeatureInteractionPairs(pairs).filter((pair) => allowed.has(pair.left) && allowed.has(pair.right));
  }

  function activeFeatureInteractionPairs(data = config) {
    return normaliseActiveFeatureInteractionConstraints(data?.active_feature_interaction_constraints).pairs;
  }

  function sameFeatureInteractionPairs(leftPairs = [], rightPairs = []) {
    const left = new Set(normaliseFeatureInteractionPairs(leftPairs).map((pair) => featureInteractionPairKey(pair.left, pair.right)));
    const right = new Set(normaliseFeatureInteractionPairs(rightPairs).map((pair) => featureInteractionPairKey(pair.left, pair.right)));
    if (left.size !== right.size) return false;
    for (const key of left) {
      if (!right.has(key)) return false;
    }
    return true;
  }

  function featureInteractionPairsEdited(pairs, data = config) {
    return !sameFeatureInteractionPairs(pairs, activeFeatureInteractionPairs(data));
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
    if (selectedCurrentCount > 0) return `Constraint groups (${selectedCurrentCount})`;
    if (syntheticCount > 0) return `Trained constraint groups (${syntheticCount})`;
    return "Constraint groups";
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

  function renderedFeatureInteractionPairs() {
    return normaliseFeatureInteractionPairs(
      [...document.querySelectorAll("[data-gbm-interaction-pair-row]")]
        .map((row) => ({
          left: row.getAttribute("data-gbm-interaction-pair-left") || "",
          right: row.getAttribute("data-gbm-interaction-pair-right") || "",
        }))
    );
  }

  function currentFeatureInteractionPairs(data = config) {
    const draft = featureDraftForData(data);
    if (featureInteractionPairsUserEdited(data)) return normaliseFeatureInteractionPairs(draft.interactionPairs || []);
    const activePairs = activeFeatureInteractionPairs(data);
    return activePairs.length ? activePairs : renderedFeatureInteractionPairs();
  }

  function featureInteractionPairsUserEdited(data = config) {
    return Boolean(
      featureInteractionPairEditModelId
      && featureInteractionPairEditModelId === featureDraftModelId(data)
      && featureDraftForData(data)?.interactionPairsEdited
    );
  }

  function currentFeatureInteractionPairsPayload() {
    const pairs = pruneFeatureInteractionPairs(currentFeatureInteractionPairs(), currentFeatureRows());
    return pairs.length ? pairs : null;
  }

  function featureInteractionPairButtonLabel(count) {
    return count > 0 ? `Interaction pairs (${count})` : "Interaction pairs";
  }

  function setFeatureInteractionPairs(pairs, features = currentFeatureRows()) {
    markFeatureInteractionPairsEdited(pairs, features);
    syncFeatureInteractionPairControls(features, pairs);
    syncFeatureInteractionLocks(currentFeatureRows());
  }

  function addFeatureInteractionPair(left, right) {
    const features = currentFeatureRows();
    const allowed = new Set(pairCandidateFeatureRows(features).map((feature) => feature.name));
    const cleanLeft = String(left || "").trim();
    const cleanRight = String(right || "").trim();
    if (!cleanLeft || !cleanRight || cleanLeft === cleanRight || !allowed.has(cleanLeft) || !allowed.has(cleanRight)) return;
    resetFeatureScenarioSelection();
    const pairFeatures = new Set([cleanLeft, cleanRight]);
    const nextFeatures = features.map((feature) => pairFeatures.has(feature.name) ? { ...feature, include: true } : feature);
    const nextPairs = [...currentFeatureInteractionPairs(), { left: cleanLeft, right: cleanRight }];
    if (featureTable) {
      const updates = [];
      for (const row of featureTable.getRows()) {
        const data = row.getData();
        if (pairFeatures.has(data.name) && !data.include) updates.push(row.update({ include: true }));
      }
      syncFeatureSectionTitleAfter(updates);
      syncFeatureInteractionPairControls(nextFeatures, nextPairs);
      syncFeatureInteractionLocks(nextFeatures);
      Promise.all(updates).then(
        () => {
          syncFeatureInteractionPairControls(currentFeatureRows(), nextPairs);
          syncFeatureInteractionLocks(currentFeatureRows());
        },
        () => {
          syncFeatureInteractionPairControls(currentFeatureRows(), nextPairs);
          syncFeatureInteractionLocks(currentFeatureRows());
        }
      );
      return;
    }
    for (const checkbox of document.querySelectorAll("[data-gbm-feature]")) {
      const name = checkbox.getAttribute("data-gbm-feature") || "";
      if (pairFeatures.has(name)) checkbox.checked = true;
    }
    syncFeatureSectionTitle();
    syncFeatureInteractionPairControls(nextFeatures, nextPairs);
    syncFeatureInteractionLocks(nextFeatures);
  }

  function syncFeatureInteractionPairControls(features = currentFeatureRows(), nextPairs = null) {
    const root = el("gbmFeatureInteractionPairSelect");
    if (!root) return;
    const selectedFeatures = pairCandidateFeatureRows(features);
    const pairs = normaliseFeatureInteractionPairs(nextPairs === null ? currentFeatureInteractionPairs() : nextPairs);
    const leftSelect = el("gbmInteractionPairLeft");
    const rightSelect = el("gbmInteractionPairRight");
    const leftValue = selectedFeatures.some((feature) => feature.name === leftSelect?.value) ? leftSelect.value : selectedFeatures[0]?.name || "";
    const rightValue = selectedFeatures.some((feature) => feature.name === rightSelect?.value) ? rightSelect.value : selectedFeatures.find((feature) => feature.name !== leftValue)?.name || selectedFeatures[1]?.name || "";
    if (leftSelect) leftSelect.innerHTML = featureInteractionPairOptions(selectedFeatures, leftValue);
    if (rightSelect) rightSelect.innerHTML = featureInteractionPairOptions(selectedFeatures, rightValue);
    const rows = el("gbmInteractionPairRows");
    if (rows) rows.innerHTML = featureInteractionPairRowsHtml(pairs);
    const button = el("gbmFeatureInteractionPairButton");
    if (button) {
      button.textContent = featureInteractionPairButtonLabel(pairs.length);
      button.classList.toggle("has-constraints", pairs.length > 0);
    }
    const add = el("gbmInteractionPairAdd");
    if (add) add.disabled = selectedFeatures.length < 2;
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
    syncFeatureInteractionPairControls(features);
    syncFeatureInteractionLocks(currentFeatureRows());
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

  function renderedPairInteractionFeatureNames(features) {
    const draft = featureDraftForData(config);
    const sourcePairs = featureInteractionPairsUserEdited(config)
      ? draft.interactionPairs || []
      : activeFeatureInteractionPairs(config);
    const locked = new Set();
    for (const pair of normaliseFeatureInteractionPairs(sourcePairs)) {
      locked.add(pair.left);
      locked.add(pair.right);
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

  function selectedPairInteractionFeatureNames(features = currentFeatureRows()) {
    const locked = new Set();
    for (const pair of normaliseFeatureInteractionPairs(currentFeatureInteractionPairs())) {
      locked.add(pair.left);
      locked.add(pair.right);
    }
    return locked;
  }

  function applyInteractionLocksToFeatures(features) {
    const draft = featureDraftForData(config);
    const groupLocked = renderedInteractionFeatureNames(features);
    const featureLocked = draft ? selectedFeatureInteractionFeatureNames(features) : activeFeatureInteractionFeatureNames();
    const pairLocked = renderedPairInteractionFeatureNames(features);
    return (features || []).map((feature) => ({
      ...feature,
      interaction_locked: groupLocked.has(feature.name),
      feature_interaction_locked: featureLocked.has(feature.name),
      pair_interaction_locked: !featureLocked.has(feature.name) && pairLocked.has(feature.name),
    }));
  }

  function syncFeatureInteractionLocks(features = currentFeatureRows()) {
    const groupLocked = selectedInteractionFeatureNames(features);
    const featureLocked = selectedFeatureInteractionFeatureNames(features);
    const pairLocked = selectedPairInteractionFeatureNames(features);
    if (featureTable) {
      for (const row of featureTable.getRows()) {
        const data = row.getData();
        const interactionLocked = groupLocked.has(data.name);
        const featureInteractionLocked = featureLocked.has(data.name);
        const pairInteractionLocked = !featureInteractionLocked && pairLocked.has(data.name);
        const update = {};
        if (Boolean(data.interaction_locked) !== interactionLocked) update.interaction_locked = interactionLocked;
        if (Boolean(data.feature_interaction_locked) !== featureInteractionLocked) update.feature_interaction_locked = featureInteractionLocked;
        if (Boolean(data.pair_interaction_locked) !== pairInteractionLocked) update.pair_interaction_locked = pairInteractionLocked;
        if (Object.keys(update).length) row.update(update);
        const featureCell = typeof row.getCell === "function" ? row.getCell("name") : null;
        const groupingCell = typeof row.getCell === "function" ? row.getCell("grouping") : null;
        if (featureCell) {
          featureCell.getElement().innerHTML = featureNameHtml({
            ...data,
            feature_interaction_locked: featureInteractionLocked,
            pair_interaction_locked: pairInteractionLocked,
          });
        }
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
        evaluationChart.setViewMode(normaliseEvaluationViewMode(input.value));
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
    setInlinePhaseStatus(status, {
      html: trainingStatusContentHtml(text, detailText),
      phase,
      hidden: !text && !detailText,
    });
  }

  function syncSidebarModelChooser(models, activeModelId) {
    const list = el("gbmModelSelect");
    const meta = el("gbmModelSelectedMeta");
    if (!list) return;
    const normalisedModels = uniqueModels(models.map(normaliseModel).filter((model) => model.model_id));
    const activeModel = normalisedModels.find((model) => model.model_id === activeModelId) || null;
    if (meta) meta.textContent = activeModel ? modelLabel(activeModel) : "No active model";
    const modelsByGroup = modelGroups(normalisedModels, modelGroupLabel);
    const groups = [...modelsByGroup.keys()];
    state.gbmModelGroupsInitialised = syncCollapsedModelGroups({
      groups,
      collapsedGroups: state.collapsedGbmModelGroups,
      initialised: state.gbmModelGroupsInitialised,
      activeGroup: activeModel ? modelGroupLabel(activeModel) : "",
    }).initialised;
    list.innerHTML = "";
    if (!normalisedModels.length) {
      list.innerHTML = emptyStateHtml("No GBMs trained yet", "gbm-empty-state", escapeHtml);
      return;
    }
    for (const group of groups) {
      const collapsed = state.collapsedGbmModelGroups.has(group);
      list.append(createSidebarModelHeading({
        group,
        collapsed,
        toolLabel: "GBM",
        className: "gbm-model-theme",
        dataKey: "gbmModelGroup",
        escapeHtml,
        onToggle: toggleGbmModelGroup,
      }));
      for (const model of modelsByGroup.get(group) || []) {
        const active = model.model_id === activeModelId;
        list.append(createSidebarModelOption({
          model,
          group,
          active,
          collapsed,
          className: "gbm-model-option",
          detailClassName: "gbm-model-detail",
          modelIdDataKey: "gbmModelId",
          groupDataKey: "gbmModelGroup",
          escapeHtml,
          modelLabel,
          modelDetailLabel,
          onActivate: activateModel,
        }));
      }
    }
  }

  function toggleGbmModelGroup(group) {
    toggleSidebarModelGroup({
      list: el("gbmModelSelect"),
      group,
      collapsedGroups: state.collapsedGbmModelGroups,
      themeClassName: "gbm-model-theme",
      optionClassName: "gbm-model-option",
      groupDataKey: "gbmModelGroup",
      toolLabel: "GBM",
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
          { title: "Parameter", field: "name", widthGrow: 1 },
          { title: "Value", field: "value", formatter: parameterValueFormatter, editor: "adaptable", editorParams: parameterValueEditorParams(), widthGrow: 2 },
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
      : feature?.pair_interaction_locked
        ? `<span class="gbm-interaction-lock gbm-pair-interaction-lock" title="Feature participates in an allowed pair interaction" aria-label="Feature participates in an allowed pair interaction">&#128274;<sub class="gbm-interaction-lock-subscript">2</sub></span>`
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
      if (names.length !== 2) return [];
      const actions = [];
      if (canAddFeatureInteractionPair(names)) {
        actions.push({ label: "Allow interaction pair", run: () => addFeatureInteractionPair(names[0], names[1]) });
      }
      if (featuresHaveSavedShap(names)) {
        actions.push({ label: "Go to SHAP", run: () => goToGbmShap(names) });
      }
      return actions;
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

  function canAddFeatureInteractionPair(names = []) {
    const pairNames = normaliseContextFeatureNames(names);
    if (pairNames.length !== 2 || pairNames[0] === pairNames[1]) return false;
    const candidates = new Set(pairCandidateFeatureRows(currentFeatureRows()).map((feature) => feature.name));
    return pairNames.every((name) => candidates.has(name));
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
    const featureName = String(name || "").trim();
    if (!featureName) return;
    selectExpectedPredictionForModelKind("gbm");
    if (typeof navigateToLineBarFeature === "function" && navigateToLineBarFeature(featureName)) {
      renderExpectedNumerators();
      updateAxisControls();
      return;
    }
    setGbmNotice(`Feature ${featureName} is not available in Line and Bar`);
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
    return parameterControls.optionsForName(name);
  }

  function parameterEditorValues(name) {
    return parameterControls.editorValues(name);
  }

  function parameterOptionByValue(name, value) {
    return parameterControls.optionByValue(name, value);
  }

  function parameterValueDisplay(name, value) {
    return parameterControls.valueDisplay(name, value);
  }

  function parameterValueFormatter(cell) {
    return parameterControls.valueFormatter(cell);
  }

  function parameterValueEditorParams() {
    return parameterControls.valueEditorParams();
  }

  function parameterValueEditorLookup(cell) {
    return parameterControls.valueEditorLookup(cell);
  }

  function parameterValueEditorParamsLookup(editor, cell) {
    return parameterControls.valueEditorParamsLookup(editor, cell);
  }

  function parameterControlHtml(parameter) {
    return parameterControls.controlHtml(parameter);
  }

  function parameterSelectOptionsHtml(name, options, value) {
    return parameterControls.selectOptionsHtml(name, options, value);
  }

  function parameterOptgroupHtml(label, options, value) {
    return parameterControls.optgroupHtml(label, options, value);
  }

  function parameterOptionHtml(option, value) {
    return parameterControls.optionHtml(option, value);
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
    return parameters.some((parameter) => String(parameter?.name || "") !== "init_score" && isGridParameterValue(parameter?.value));
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
    return modelNavigator.rows(models);
  }

  function modelInteractionConstraintLabel(rawConstraints) {
    const active = normaliseActiveFeatureInteractionConstraints(rawConstraints);
    if (active.mode === "pairs" || active.pairs.length) return active.pairs.length ? `Pairs (${active.pairs.length})` : "No";
    const groupings = active.groups.map((group) => group.grouping);
    return groupings.length ? groupings.join(", ") : "No";
  }

  function renderModelFallback(models) {
    modelNavigator.renderFallback(el("gbmModelFallback"), models);
  }

  function formatModelCount(value) {
    return modelNavigator.count(value);
  }

  function modelParameterNumber(model, name) {
    return modelNavigator.parameterNumber(model, name);
  }

  function formatModelInteger(value) {
    return modelNavigator.integer(value);
  }

  function formatModelDecimal(value) {
    return modelNavigator.decimal(value);
  }

  function modelRuntimeSeconds(model) {
    return modelNavigator.runtimeSeconds(model);
  }

  function formatModelRuntime(model) {
    return modelNavigator.runtime(model);
  }

  function formatModelCreated(value) {
    return modelNavigator.created(value);
  }

  function modelCreatedSort(value) {
    return modelNavigator.createdSort(value);
  }

  function formatSampleMode(value, source = "") {
    return modelNavigator.sampleMode(value, source);
  }

  function modelWeightLabel(value) {
    return modelNavigator.weightLabel(value);
  }

  function bindModelActions() {
    el("gbmRenameModelBtn")?.addEventListener("click", renameActiveModel);
    el("gbmActivateModelBtn")?.addEventListener("click", activateSelectedModel);
    el("gbmDeleteModelBtn")?.addEventListener("click", deleteActiveModel);
    syncModelActionButtons();
  }

  function syncModelActionButtons() {
    syncSharedModelActionButtons({
      selectedCount: selectedModelIds().length,
      disabled: isTraining,
      rename: el("gbmRenameModelBtn"),
      activate: el("gbmActivateModelBtn"),
      deleteButton: el("gbmDeleteModelBtn"),
    });
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
    syncDatasetGbmCountFromConfig({ models });
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
    restoreSharedModelSelection({
      table: modelTable,
      fallbackSelector: "#gbmModelFallback [data-gbm-model-row]",
      rowDataKey: "gbmModelRow",
      ids,
    });
  }

  function selectedModelIds() {
    return selectedModelIdsFromTableOrFallback({
      table: modelTable,
      fallbackSelector: "#gbmModelFallback [data-gbm-model-row]",
      rowDataKey: "gbmModelRow",
    });
  }

  function currentActiveModelId() {
    return String(config?.active_model_id || (config?.models || []).find((model) => model.active)?.model_id || "");
  }

  function currentFeatureRows() {
    const reserved = currentReservedFeatureNames();
    const pairFeatureNames = currentFeatureInteractionPairFeatureNames();
    function applyReserved(feature) {
      if (reserved.has(feature.name)) return { ...feature, include: false };
      return pairFeatureNames.has(feature.name) ? { ...feature, include: true } : feature;
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
    const featureInteractionPairs = currentFeatureInteractionPairsPayload();
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
    if (featureInteractionPairs) payload.feature_interaction_pairs = featureInteractionPairs;
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
        if (isModelJobPending(job.status)) {
          if (!job.progress) {
            const fallback = job.status === "queued" ? "GBM queued..." : "Training GBM...";
            setTrainingStatus(fallback, job.status, gridTrainingNotice);
            setGroupMeta(tool, fallback);
          }
          if (activeTab === "models") refreshModelList();
          pollJob(jobId, modelJobPollDelay(job.status, GBM_QUEUED_POLL_MS, GBM_RUNNING_POLL_MS));
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
        syncDatasetGbmCountFromConfig(data);
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
    renderLiveProgress(job.progress, job);
  }

  function renderLiveProgress(progress, job = null) {
    liveProgress = progress;
    setAppReadyStatus(gbmTrainingReadyBadgeLabel(progress));
    setTrainingStatus(progress.message || "", progress.phase || "", trainingStatusDetail(progress, job));
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

  function trainingStatusDetail(progress, job = null) {
    const parts = [];
    const elapsed = formatGbmElapsedDuration(job?.elapsed_seconds ?? progress?.elapsed_seconds);
    if (elapsed) parts.push(`elapsed ${elapsed}`);
    const gridMessages = Array.isArray(progress?.grid?.messages) ? progress.grid.messages : [];
    if (gridMessages.length) parts.push(gridMessages.join(" "));
    const parameters = Array.isArray(progress?.grid_parameters) ? progress.grid_parameters : [];
    if (parameters.length) {
      parts.push(parameters.map((row) => `${row.label || row.name}=${row.value}`).join(" · "));
    }
    if (!parts.length && gridTrainingNotice) parts.push(gridTrainingNotice);
    return parts.join("  ");
  }

  function formatGbmElapsedDuration(value) {
    const seconds = Math.max(0, Math.floor(Number(value)));
    if (!Number.isFinite(seconds)) return "";
    if (seconds < 60) return `${seconds}s`;
    const minutes = Math.floor(seconds / 60);
    const remainingSeconds = seconds % 60;
    if (minutes < 60) return `${minutes}m ${String(remainingSeconds).padStart(2, "0")}s`;
    const hours = Math.floor(minutes / 60);
    const remainingMinutes = minutes % 60;
    return `${hours}h ${String(remainingMinutes).padStart(2, "0")}m`;
  }

  async function activateModel(modelId) {
    if (isTraining) return;
    if (!modelId) return;
    try {
      const result = await api(`/api/gbm/models/${encodeURIComponent(modelId)}/activate`, { method: "POST", body: "{}" });
      await applyModelMutationResult(result, { activationOnly: true });
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

  async function applyModelMutationResult(result, options = {}) {
    const nextConfig = result.config || config || {};
    await reloadSchema(preferredModelSource(result, nextConfig), { modelKind: "gbm" });
    const preserveProfile = clearCachesAfterGbmModelSourceChange();
    syncDatasetGbmCountFromConfig(nextConfig);
    const currentModelId = featureDraftModelId(config);
    const nextModelId = featureDraftModelId(nextConfig);
    if (currentModelId !== nextModelId) {
      featureDraftState = null;
      featureInteractionPairEditModelId = "";
    }
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
      if (!(options?.activationOnly && await onExternalModelActivation("gbm"))) {
        await refreshActiveTool({ force: true });
      }
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
    evaluationChart.render(source || activeDetail);
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
