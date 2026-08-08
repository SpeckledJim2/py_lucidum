import { createGlmFormulaBuilder } from "./glm-formula-builder.js";
import { createGlmModelNavigator } from "./glm-model-navigator.js";
import { createGlmTabulations, GLM_TABULATION_MODEL_CROSSTAB } from "./glm-tabulations.js";
import { createOperationId } from "./shared/api.js";
import { loadTabulator } from "./shared/tabulator.js";
import {
  bindToolScreenNavigation,
  syncToolScreenNavigation,
  toolScreenNavButtonHtml,
} from "./shared/tool-screen-nav.js";
import {
  bindFallbackModelSelection,
  createSidebarModelHeading,
  createSidebarModelOption,
  emptyStateHtml,
  formatModelCreated as sharedFormatModelCreated,
  formatModelMetric as sharedFormatModelMetric,
  isModelJobPending,
  modelCreatedSort as sharedModelCreatedSort,
  modelGroups,
  modelJobPollDelay,
  modelNumberOrNull as sharedModelNumberOrNull,
  observeResize,
  restoreModelSelection as restoreSharedModelSelection,
  selectedModelIdsFromTableOrFallback,
  syncCollapsedModelGroups,
  syncModelActionButtons as syncSharedModelActionButtons,
  toggleSidebarModelGroup,
} from "./shared/model-ui.js";

const GLM_RUNNING_POLL_MS = 500;
const GLM_QUEUED_POLL_MS = 1000;
const GLM_MODEL_LIST_POLL_MS = 2000;
const GLM_TABS = [
  { id: "builder", label: "Formula builder", icon: "formula" },
  { id: "models", label: "Model navigator", icon: "models" },
  { id: "tabulations", label: "Tabulations", icon: "table" },
];

function glmAutoModelTimeLabel(date = new Date()) {
  const hour = String(date.getHours()).padStart(2, "0");
  const minute = String(date.getMinutes()).padStart(2, "0");
  const second = String(date.getSeconds()).padStart(2, "0");
  return `${hour}:${minute}:${second}`;
}

export function glmBuildReadyBadgeLabel(progress = null) {
  const phase = String(progress?.phase || "").trim().toLowerCase();
  return phase === "scoring" || phase === "writing" || phase === "succeeded"
    ? "Scoring GLM"
    : "Training GLM";
}

export function glmTabulationReadyBadgeLabel(progress = null) {
  const phase = String(progress?.phase || "").trim().toLowerCase();
  return phase === "scoring" || phase === "writing" || phase === "succeeded"
    ? "Scoring tabulations"
    : "Tabulating GLM";
}

function modelNumberOrNull(value) {
  return sharedModelNumberOrNull(value);
}

function formatModelMetric(value) {
  return sharedFormatModelMetric(value);
}

function formatModelCreated(value) {
  return sharedFormatModelCreated(value);
}

function csvEscape(value) {
  const text = String(value ?? "");
  return /[",\n]/.test(text) ? `"${text.replaceAll('"', '""')}"` : text;
}

function escapeRegExp(value) {
  return String(value || "").replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}

function downloadText(filename, text, type = "text/plain") {
  const blob = new Blob([text], { type });
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = filename;
  document.body.append(link);
  link.click();
  link.remove();
  URL.revokeObjectURL(url);
}

export function glmModelDetailLabel(model = {}) {
  const metrics = model.diagnostics || model.metrics || {};
  const parts = [];
  if (model.family) parts.push(String(model.family));
  if (metrics.aic !== undefined) parts.push(`AIC ${formatModelMetric(metrics.aic)}`);
  return parts.join(" · ");
}

export function createGlmTool({
  api,
  clearToolCaches,
  copyTextToClipboard,
  el,
  escapeHtml,
  measureToolRender,
  renderExpectedNumerators,
  renderFeatures,
  saveToolPresentation,
  setChartMessage,
  setClientTiming,
  setGlmModelCount = () => {},
  setDuckDbTiming,
  setGroupMeta,
  setRenderTiming,
  setStatus,
  setAppReadyStatus = () => {},
  setToolTimingFailed,
  showClipboardToast = () => {},
  startToolTiming,
  state,
  syncClientTimingFromData,
  syncDuckDbTimingFromData,
  toolCache,
  canNavigateToLineBarFeature = () => false,
  navigateToLineBarFeature = () => false,
  selectExpectedPredictionForModelKind = () => false,
  updateAxisControls,
  refreshActiveTool,
  reloadSchema,
  invalidateLineBar = () => {},
  getDenominatorSelection = () => ({ value: "__none__", sourceId: "dataset", metricKind: "dataset" }),
}) {
  const tool = "glm";
  let activeTab = "builder";
  let config = null;
  let activeDetail = null;
  let coefficientRows = [];
  let coefficientSort = { key: "index", direction: "asc" };
  let modelTable = null;
  let modelTableReady = false;
  let modelTableRenderSeq = 0;
  let modelRows = [];
  let selectedModelIds = new Set();
  let selectedTabulationModelIds = new Set();
  let tabulationSelectionAnchorModelId = "";
  let pollTimer = null;
  let tabulationPollTimer = null;
  let buildOperationId = "";
  let tabulationOperationId = "";
  let modelListRefreshSeq = 0;
  let modelListLastRefreshAt = 0;
  let modelStateGeneration = 0;
  let modelDetailRequestSeq = 0;
  let modelMutationPending = 0;
  let queuedActivationModelId = "";
  let activationPromise = null;
  let isBuilding = false;
  let modelFolderOpenPending = false;
  let buildElapsedStartedAt = null;
  let isTabulating = false;
  let tabulationElapsedStartedAt = null;
  let isExportingTabulations = false;
  let liveProgress = null;
  let tabulationConfig = null;
  let tabulationModelTable = null;
  let tabulationCommonTable = null;
  let tabulationOtherTable = null;
  let tabulationTable = null;
  let tabulationChart = null;
  let tabulationPayload = null;
  let tabulationRenderSeq = 0;
  let tabulationSelectionRefreshSeq = 0;
  let tabulationSelectorRenderSeq = 0;
  let tabulationModelSelectorSignature = "";
  let tabulationTableSelectorSignature = "";
  let tabulationPanelModeSignature = "";
  let tabulationResizeObserver = null;
  let tabulationResizeFrame = null;
  let tabulationFallbackRows = [];
  let tabulationFallbackColumns = [];
  let glmCoefficientContextMenuListeners = null;
  let glmTabulationContextMenuListeners = null;
  let selectedTabulationTableId = localStorage.getItem("py_lucidum_glm_tabulation_table") || "base";
  let tabulationView = localStorage.getItem("py_lucidum_glm_tabulation_view") || "table";
  let tabulationScale = localStorage.getItem("py_lucidum_glm_tabulation_scale") || "linear";
  let tabulationColor = localStorage.getItem("py_lucidum_glm_tabulation_color") === "true";
  let tabulationCrosstab = "";
  let tabulationCrosstabManualKey = "";
  let tabulationCrosstabDefaultKey = "";
  const tabulationCrosstabDefaultCache = new Map();
  let isRebasing = false;
  let builderDraftSourceModelId = "";
  const formulaBuilder = createGlmFormulaBuilder({
    api,
    el,
    escapeHtml,
    getColumns: schemaDatasetColumns,
    getDenominator: () => el("denominator")?.value || "__none__",
    getFamilies: () => config?.families || [],
    onBuildModel: buildModel,
    onCoefficientSearch: () => renderCoefficientTable(coefficientRows),
    onCopyCoefficients: copyCoefficients,
    onCopyFormula: copyFormula,
    onDownloadCoefficients: downloadCoefficients,
  });
  const modelNavigator = createGlmModelNavigator({
    bindFallbackModelSelection,
    emptyStateHtml,
    escapeHtml,
    formatModelCreated,
    formatModelMetric,
    modelCreatedSort: sharedModelCreatedSort,
    modelLabel,
    modelNumberOrNull,
    modelWeightLabel,
    normaliseModels,
    selectedModelIds: () => selectedModelIds,
    onFallbackSelectionChange: syncSelectedModelsFromTable,
  });
  const tabulations = createGlmTabulations({
    el,
    modelNumberOrNull,
    scheduleResize: scheduleTabulationResize,
  });

  function buildRequest() {
    if (!state.schema) return null;
    return {
      tool,
      source: state.source || "dataset",
    };
  }

  function advanceModelStateGeneration() {
    modelStateGeneration += 1;
    modelListRefreshSeq += 1;
    modelDetailRequestSeq += 1;
    return modelStateGeneration;
  }

  function modelStateIsCurrent(generation) {
    return generation === modelStateGeneration;
  }

  function beginModelMutation() {
    modelMutationPending += 1;
    updateModelActionButtons();
  }

  function endModelMutation() {
    modelMutationPending = Math.max(0, modelMutationPending - 1);
    updateModelActionButtons();
  }

  async function fetchData(request, requestKey) {
    const requestSeq = state.glmRequestSeq + 1;
    const generation = modelStateGeneration;
    state.glmRequestSeq = requestSeq;
    setStatus("");
    setChartMessage("");
    setGroupMeta(tool, "Loading GLM...");
    startToolTiming(tool);
    try {
      const data = await api("/api/glm/config", { method: "GET", clientTiming: true });
      if (requestSeq !== state.glmRequestSeq || !modelStateIsCurrent(generation)) return null;
      const cache = toolCache(tool);
      cache.requestKey = requestKey;
      cache.data = data;
      setGlmModelCount(Array.isArray(data?.models) ? data.models.length : null);
      syncDuckDbTimingFromData(tool, data);
      syncClientTimingFromData(tool, data);
      measureToolRender(tool, () => render(data));
      return data;
    } catch (error) {
      if (requestSeq !== state.glmRequestSeq || !modelStateIsCurrent(generation)) return null;
      setToolTimingFailed(tool);
      setGroupMeta(tool, "GLM failed");
      setChartMessage("");
      setGlmNotice(error.message);
      return null;
    }
  }

  function useCached(cache) {
    measureToolRender(tool, () => {
      render(cache.data);
      applyPresentation();
    });
  }

  function applyPresentation() {
    const presentation = toolCache(tool).presentation;
    if (!presentation) return;
    setGroupMeta(tool, presentation.groupMeta);
    setChartMessage(presentation.chartMessage);
  }

  function render(data = {}) {
    formulaBuilder.captureDraft();
    config = data;
    modelRows = normaliseModels(data.models || []);
    const activeModelId = currentActiveModelId(data);
    if (!activeModelId) builderDraftSourceModelId = "";
    const availableModelIds = new Set(modelRows.map((model) => model.model_id));
    selectedModelIds = new Set(Array.from(selectedModelIds).filter((modelId) => availableModelIds.has(modelId)));
    const availableTabulationRefs = new Set(tabulationAvailableModels().map((model) => tabulationModelRef(model)).filter(Boolean));
    selectedTabulationModelIds = new Set(Array.from(selectedTabulationModelIds).map(normaliseTabulationRef).filter((modelRef) => availableTabulationRefs.has(modelRef)));
    if (tabulationSelectionAnchorModelId && !availableTabulationRefs.has(normaliseTabulationRef(tabulationSelectionAnchorModelId))) tabulationSelectionAnchorModelId = "";
    if (!selectedTabulationModelIds.size && data.active_model_id) selectedTabulationModelIds.add(`glm:${data.active_model_id}`);
    const groupMeta = "";
    setGroupMeta(tool, groupMeta);
    setStatus("");
    setChartMessage("");
    formulaBuilder.disposeEditor();
    disconnectTabulationResizeObserver();
    disposeTabulationChart();
    disposeTabulationTable();
    disposeTabulationSelectorTables();
    tabulationPanelModeSignature = "";
    modelTable = null;
    modelTableReady = false;
    const mount = el("modelToolWrap");
    if (!mount) return;
    mount.innerHTML = shellHtml(data);
    bindTabs(mount);
    bindBuilderControls();
    bindBuilderResizer();
    bindModelActions();
    bindTabulationControls();
    bindTabulationResizer();
    renderModelTable(modelRows, data.active_model_id);
    syncSidebarModelChooser(modelRows, data.active_model_id);
    renderCoefficientTable(coefficientRowsForActiveModel(data.active_model_id));
    formulaBuilder.initEditor();
    if (activeModelId) loadModelDetail(activeModelId);
    if (liveProgress) renderLiveProgress(liveProgress);
    if (activeTab === "tabulations") refreshTabulationConfig({ force: true });
    setDuckDbTiming(tool, data.timings || {});
    setClientTiming(tool, data.client_timings || {});
    setRenderTiming(tool, 0);
    saveToolPresentation(tool, { groupMeta, chartMessage: "" });
  }

  function shellHtml(data = {}) {
    const trainingDisabled = formulaBuilder.ensureTrainingScope(data);
    const predictionDenominator = getDenominatorSelection().metricKind === "prediction";
    const activeModel = modelForActiveModel(data.active_model_id);
    const diagnostics = activeModel?.diagnostics || activeModel?.metrics || {};
    const splitStyle = formulaBuilder.savedSplitWidthStyle();
    return `
      <div class="glm-tool">
        <div id="glmNotice" class="glm-notice hidden" role="alert" aria-live="polite"></div>
        <div class="glm-toolbar">
          <div class="glm-tabs tool-screen-nav" role="tablist" aria-label="GLM screens">
            ${GLM_TABS.map((tab) => toolScreenNavButtonHtml({
              active: activeTab === tab.id,
              buttonId: `glm-screen-tab-${tab.id}`,
              controlsId: `glm-screen-panel-${tab.id}`,
              icon: tab.icon,
              label: tab.label,
              targetId: tab.id,
              toolDataAttribute: "glm-tab",
            })).join("")}
          </div>
          <div id="glmBuildStatus" class="glm-build-status ${liveProgress ? "" : "hidden"}" aria-live="polite">${buildStatusHtml(liveProgress)}</div>
        </div>
        <div id="glm-screen-panel-builder" class="glm-tab-panel ${activeTab === "builder" ? "" : "hidden"}" data-glm-panel="builder" role="tabpanel" aria-labelledby="glm-screen-tab-builder">
          <div class="glm-builder-layout"${splitStyle ? ` style="${splitStyle}"` : ""}>
            <section class="glm-formula-panel">
              <div class="glm-panel-header app-control-strip app-control-strip-row app-control-strip--titled">
                <h3 class="glm-panel-title">GLM formula</h3>
                <div class="glm-builder-actions">
                  <div class="glm-builder-panel-toggles" role="group" aria-label="Formula builder panels">
                    <button id="glmFormulaAssistBtn" class="app-control-button glm-builder-option-button glm-icon-action-button ${formulaBuilder.formulaAssistOpen ? "active" : ""}" type="button" aria-label="Formula tools" title="Formula tools" aria-controls="glmFormulaAssistDrawer" aria-expanded="${formulaBuilder.formulaAssistOpen ? "true" : "false"}">f(x)</button>
                    <button id="glmModelParametersBtn" class="app-control-button glm-builder-option-button glm-icon-action-button ${formulaBuilder.parametersOpen ? "active" : ""}" type="button" aria-label="Model parameters" title="Model parameters" aria-controls="glmBuilderParametersPanel" aria-expanded="${formulaBuilder.parametersOpen ? "true" : "false"}">
                      <svg class="glm-model-parameters-icon" viewBox="0 0 24 24" aria-hidden="true" focusable="false">
                        <path d="M4 6h3m4 0h9M9 4v4M4 12h9m4 0h3M15 10v4M4 18h5m4 0h7M11 16v4"></path>
                      </svg>
                    </button>
                  </div>
                  <div class="segmented glm-scope-control glm-header-scope-control" role="group" aria-label="Rows to fit">
                    <button type="button" data-glm-scope="all" data-stable-label="All" class="app-control-button glm-builder-option-button ${formulaBuilder.selectedTrainingScope === "all" ? "active" : ""}" aria-pressed="${formulaBuilder.selectedTrainingScope === "all" ? "true" : "false"}">All</button>
                    <button type="button" data-glm-scope="training" data-stable-label="Training" class="app-control-button glm-builder-option-button ${formulaBuilder.selectedTrainingScope === "training" ? "active" : ""}" aria-pressed="${formulaBuilder.selectedTrainingScope === "training" ? "true" : "false"}" ${trainingDisabled ? "disabled" : ""}>Training</button>
                  </div>
                  <button id="glmBuildBtn" class="tab app-control-button model-busy-button glm-build-button ${isBuilding ? "building" : ""}" type="button" ${isBuilding || predictionDenominator ? "disabled" : ""} ${isBuilding ? "aria-busy=\"true\"" : ""}>${isBuilding ? "Building..." : "Build GLM"}</button>
                </div>
              </div>
              <div id="glmModelDenominatorBuildNotice" class="glm-model-denominator-build-notice ${predictionDenominator ? "" : "hidden"}">Building is unavailable while Denominator is a model prediction. Use GBM init_score for prediction chaining.</div>
              ${formulaBuilder.formulaAssistDrawerHtml()}
              <div id="glmBuilderParametersPanel" class="glm-builder-control-row glm-builder-control-stack ${formulaBuilder.parametersOpen ? "" : "hidden"}">
                <div class="glm-control-line">
                  <div class="glm-family-row">
                    <label class="glm-control-label" for="glmFamilySelect">Family</label>
                    <select id="glmFamilySelect" aria-label="GLM family">${formulaBuilder.familyOptionsHtml(data.families || [])}</select>
                    <input id="glmFamilyParameter" class="glm-family-parameter" type="text" inputmode="decimal" placeholder="family.parameter" value="${escapeHtml(String(formulaBuilder.familyParameterDefault(data.families || [])))}" aria-label="GLM family parameter" />
                  </div>
                </div>
                <div class="glm-control-line">
                  <div class="glm-penalty-row">
                    <label class="glm-control-label" for="glmRegularizationMode">Penalty</label>
                    <select id="glmRegularizationMode" class="glm-penalty-mode" aria-label="GLM penalty">${formulaBuilder.regularizationModeOptionsHtml(data.regularization)}</select>
                    <div id="glmRegularizationManualControls" class="glm-penalty-manual ${formulaBuilder.selectedRegularizationMode === "manual" ? "" : "disabled"}">
                      <label class="glm-control-label" for="glmRegularizationMix">Mix</label>
                      <select id="glmRegularizationMix" class="glm-penalty-mix" aria-label="GLM penalty mix">${formulaBuilder.regularizationMixOptionsHtml(data.regularization)}</select>
                      <label class="glm-control-label" for="glmRegularizationAlpha">Alpha</label>
                      <input id="glmRegularizationAlpha" class="glm-penalty-alpha" type="text" inputmode="decimal" value="${escapeHtml(formulaBuilder.selectedRegularizationAlpha)}" aria-label="GLM penalty alpha" />
                    </div>
                  </div>
                </div>
              </div>
              <div class="glm-editor-shell">
                <div id="glmFormulaEditor" class="glm-formula-editor"></div>
                <textarea id="glmFormulaText" class="glm-formula-text" spellcheck="false">${escapeHtml(formulaBuilder.formulaDraft)}</textarea>
                <div class="glm-editor-font-controls" role="group" aria-label="Formula editor controls">
                  <button id="glmClearFormulaBtn" class="glm-editor-font-button" type="button" aria-label="Clear formula" title="Clear formula">×</button>
                  <button id="glmFontSmallerBtn" class="glm-editor-font-button" type="button" aria-label="Decrease formula font size" title="Decrease font size">A-</button>
                  <button id="glmFontLargerBtn" class="glm-editor-font-button" type="button" aria-label="Increase formula font size" title="Increase font size">A+</button>
                  <button id="glmCopyFormulaBtn" class="glm-editor-font-button" type="button" aria-label="Copy formula" title="Copy formula">
                    <svg class="glm-editor-copy-icon" viewBox="0 0 24 24" aria-hidden="true" focusable="false">
                      <rect x="9" y="9" width="11" height="11" rx="2"></rect>
                      <path d="M15 9V6a2 2 0 0 0-2-2H6a2 2 0 0 0-2 2v7a2 2 0 0 0 2 2h3"></path>
                    </svg>
                  </button>
                </div>
              </div>
            </section>
            <div id="glmBuilderResizer" class="glm-builder-resizer app-resizer app-resizer--vertical" role="separator" aria-orientation="vertical" aria-label="Resize GLM formula and coefficients panels" tabindex="0"></div>
            <section class="glm-coefficient-panel">
              <div class="glm-panel-header glm-coefficient-header app-control-strip app-control-strip-row app-control-strip--titled">
                <h3 class="glm-panel-title">Coefficients</h3>
                <div class="glm-coefficient-actions">
                  <button id="glmCopyCoefficientsBtn" class="app-control-button glm-coefficient-action-button" type="button" aria-label="Copy coefficients" title="Copy coefficients">
                    <svg class="glm-coefficient-action-icon" viewBox="0 0 24 24" aria-hidden="true" focusable="false">
                      <rect x="9" y="9" width="11" height="11" rx="2"></rect>
                      <path d="M15 9V6a2 2 0 0 0-2-2H6a2 2 0 0 0-2 2v7a2 2 0 0 0 2 2h3"></path>
                    </svg>
                  </button>
                  <button id="glmDownloadCoefficientsBtn" class="app-control-button glm-coefficient-action-button" type="button" aria-label="Download coefficients" title="Download coefficients">
                    <svg class="glm-coefficient-action-icon" viewBox="0 0 24 24" aria-hidden="true" focusable="false">
                      <path d="M12 3v12m0 0 4-4m-4 4-4-4M5 20h14"></path>
                    </svg>
                  </button>
                </div>
              </div>
              <div id="glmCoefficientMeta" class="glm-coefficient-meta">${diagnosticsHtml(diagnostics, activeModel, coefficientRowsForActiveModel(data.active_model_id))}</div>
              <div class="glm-table-tools">
                <label>Search: <input id="glmCoefficientSearch" class="search" type="search" /></label>
              </div>
              <div class="glm-coefficient-table-wrap">
                <table class="glm-table" id="glmCoefficientTable"></table>
              </div>
            </section>
          </div>
        </div>
        <div id="glm-screen-panel-models" class="glm-tab-panel ${activeTab === "models" ? "" : "hidden"}" data-glm-panel="models" role="tabpanel" aria-labelledby="glm-screen-tab-models">
          <div class="glm-model-navigator">
            <div class="glm-model-actions app-control-strip app-control-strip-row app-control-strip--actions" role="group" aria-label="GLM model actions">
              ${state.schema?.capabilities?.open_model_folders ? '<button id="glmOpenModelFolderBtn" class="app-control-button app-command-button" type="button" title="Open the selected model sidecar folder">Open folder</button>' : ""}
              <button id="glmRenameModelBtn" class="app-control-button app-command-button" type="button">Rename</button>
              <button id="glmActivateModelBtn" class="app-control-button app-command-button" type="button">Activate</button>
              <button id="glmDeleteModelBtn" class="app-control-button app-command-button app-command-button--danger" type="button">Delete</button>
            </div>
            <div id="glmModelGrid" class="glm-grid glm-model-grid"></div>
            <div id="glmModelFallback" class="glm-model-fallback"></div>
          </div>
        </div>
        <div id="glm-screen-panel-tabulations" class="glm-tab-panel ${activeTab === "tabulations" ? "" : "hidden"}" data-glm-panel="tabulations" role="tabpanel" aria-labelledby="glm-screen-tab-tabulations">
          <div id="glmTabulationsPanel" class="glm-tabulations-panel">
            ${tabulationsPanelHtml()}
          </div>
        </div>
      </div>
    `;
  }

  function tabulationsPanelHtml() {
    const selectedIds = tabulationSelectedModelIds();
    const availableModels = tabulationAvailableModels();
    const tables = Array.isArray(tabulationConfig?.tables) ? tabulationConfig.tables : [];
    if (tables.length && !tables.some((table) => String(table.table_id || "") === selectedTabulationTableId)) {
      selectedTabulationTableId = String(tables[0]?.table_id || "base");
    }
    const activeTable = activeTabulationTable();
    const features = Array.isArray(activeTable?.features) ? activeTable.features : [];
    const crosstabOptions = tabulationCrosstabOptions(features, selectedIds);
    normaliseTabulationCrosstab(crosstabOptions);
    const diagnostics = tabulationDiagnosticsHtml();
    return `
      <div class="glm-tabulation-layout" style="${savedTabulationSplitWidthStyle()}">
        <section class="glm-tabulation-sidebar">
          <div class="glm-panel-header app-control-strip app-control-strip-row app-control-strip--titled">
            <h3 class="glm-panel-title">Tabulations</h3>
            <button id="glmBuildTabulationsBtn" class="tab app-control-button model-busy-button glm-build-button ${isTabulating ? "building" : ""}" type="button" ${isTabulating || !availableModels.length ? "disabled" : ""} ${isTabulating ? "aria-busy=\"true\"" : ""}>${isTabulating ? "Tabulating..." : "Tabulate"}</button>
          </div>
          <label id="glmTabulationModelLabel" class="glm-tabulation-label">Select models</label>
          <div class="glm-tabulation-model-region">
            <div id="glmTabulationModelGrid" class="glm-grid glm-tabulation-selector-grid glm-tabulation-model-list" aria-labelledby="glmTabulationModelLabel"></div>
            <div id="glmTabulationModelFallback" class="glm-tabulation-selector-fallback"></div>
          </div>
          <div id="glmTabulationSelectorResizer" class="glm-tabulation-selector-resizer app-resizer app-resizer--horizontal" role="separator" aria-orientation="horizontal" aria-label="Resize GLM model and table selectors" tabindex="0"></div>
          <label id="glmTabulationTableLabel" class="glm-tabulation-label">Select table</label>
          <div id="glmTabulationTableSections" class="glm-tabulation-table-sections ${selectedIds.length > 1 ? "multi" : "single"}" aria-labelledby="glmTabulationTableLabel">
            ${tabulationTableSelectorShellHtml(selectedIds)}
          </div>
          <div id="glmTabulationDiagnostics" class="glm-tabulation-diagnostics ${diagnostics ? "" : "hidden"}">${diagnostics}</div>
        </section>
        <div id="glmTabulationResizer" class="glm-builder-resizer glm-tabulation-resizer app-resizer app-resizer--vertical" role="separator" aria-orientation="vertical" aria-label="Resize GLM tabulations and table panels" tabindex="0"></div>
        <section class="glm-tabulation-main">
          <div class="glm-tabulation-controls app-control-strip app-control-strip-row">
            <div class="glm-tabulation-controls-row glm-tabulation-controls-primary">
              <div class="glm-tabulation-control-group glm-tabulation-control-left">
                <div class="glm-tabulation-option-group glm-tabulation-view-toggle" role="group" aria-label="Tabulation view">
                  <button type="button" data-glm-tabulation-view="table" data-stable-label="Table" class="app-control-button glm-tabulation-option-button ${tabulationView === "table" ? "active" : ""}" aria-pressed="${tabulationView === "table" ? "true" : "false"}">Table</button>
                  <button type="button" data-glm-tabulation-view="plot" data-stable-label="Plot" class="app-control-button glm-tabulation-option-button ${tabulationView === "plot" ? "active" : ""}" aria-pressed="${tabulationView === "plot" ? "true" : "false"}" ${features.length > 2 ? "disabled" : ""}>Plot</button>
                </div>
              </div>
              <div class="glm-tabulation-control-group glm-tabulation-control-middle">
                <div class="glm-tabulation-option-group glm-tabulation-scale-toggle" role="group" aria-label="Tabulation display scale">
                  <button id="glmTabulationExpBtn" type="button" data-glm-tabulation-scale="exp" data-stable-label="Exp" class="app-control-button glm-tabulation-option-button ${tabulationScale === "exp" ? "active" : ""}" aria-label="Exponential scale" aria-pressed="${tabulationScale === "exp" ? "true" : "false"}">Exp</button>
                </div>
                <button id="glmTabulationColorBtn" type="button" data-stable-label="Colour" class="app-control-button glm-tabulation-option-button ${tabulationColor ? "active" : ""}" aria-label="Colour cells" aria-pressed="${tabulationColor ? "true" : "false"}">Colour</button>
              </div>
              <div class="glm-tabulation-control-group glm-tabulation-control-right">
                <div class="glm-tabulation-crosstab-group">
                  <label class="glm-tabulation-crosstab-label" for="glmTabulationCrosstab">crosstab</label>
                  <select id="glmTabulationCrosstab" class="glm-tabulation-crosstab" ${crosstabOptions.length > 1 ? "" : "disabled"}>
                    ${tabulationCrosstabOptionsHtml(crosstabOptions)}
                  </select>
                </div>
                <button id="glmExportTabulationsBtn" class="app-control-button model-busy-button glm-tabulation-export-button ${isExportingTabulations ? "building" : ""}" type="button" aria-label="${isExportingTabulations ? "Exporting XLSX" : "Export XLSX"}" title="${isExportingTabulations ? "Exporting XLSX" : "Export XLSX"}" ${canExportSelectedTabulations() ? "" : "disabled"} ${isExportingTabulations ? "aria-busy=\"true\"" : ""}>
                  <svg class="glm-tabulation-export-icon" viewBox="0 0 24 24" aria-hidden="true" focusable="false">
                    <path d="M4 3h10l5 5v3M14 3v5h5M5 12h8v8H5zM5 16h8M9 12v8M18 13v7m0 0 3-3m-3 3-3-3"></path>
                  </svg>
                </button>
              </div>
            </div>
          </div>
          <div id="glmTabulationNotice" class="glm-tabulation-inline-notice"></div>
          <div class="glm-tabulation-view-shell ${tabulationView === "table" ? "" : "hidden"}" data-glm-tabulation-view-panel="table">
            <div id="glmTabulationTable" class="glm-grid glm-tabulation-grid"></div>
            <div id="glmTabulationFallback" class="glm-tabulation-fallback"></div>
          </div>
          <div class="glm-tabulation-view-shell ${tabulationView === "plot" ? "" : "hidden"}" data-glm-tabulation-view-panel="plot">
            <button id="glmTabulationCopyBtn" class="app-control-button glm-tabulation-copy-button" type="button" aria-label="Copy tabulation chart" title="Copy tabulation chart" disabled>
              <svg class="glm-tabulation-copy-icon" viewBox="0 0 24 24" aria-hidden="true" focusable="false">
                <rect x="8" y="8" width="10" height="10" rx="1.5"></rect>
                <path d="M6 14H5.5A1.5 1.5 0 0 1 4 12.5v-7A1.5 1.5 0 0 1 5.5 4h7A1.5 1.5 0 0 1 14 5.5V6"></path>
              </svg>
            </button>
            <div id="glmTabulationPlot" class="glm-tabulation-plot"></div>
          </div>
        </section>
      </div>
    `;
  }

  function tabulationSelectedModelIds() {
    const availableModels = tabulationAvailableModels();
    const availableModelIds = new Set(availableModels.map((model) => tabulationModelRef(model)).filter(Boolean));
    const ids = Array.from(selectedTabulationModelIds).map(normaliseTabulationRef).filter((modelId) => availableModelIds.has(modelId));
    if (ids.length) return [...new Set(ids)];
    const active = availableModels.find((model) => model.active) || (config?.active_model_id ? { model_kind: "glm", model_id: config.active_model_id } : null) || availableModels[0] || null;
    const activeRef = active ? tabulationModelRef(active) : "";
    return activeRef ? [activeRef] : [];
  }

  function normaliseTabulationRef(value) {
    return tabulations.normaliseRef(value);
  }

  function tabulationModelRef(model = {}) {
    return tabulations.modelRef(model);
  }

  function tabulationAvailableModels() {
    const allModels = Array.isArray(tabulationConfig?.all_models) ? tabulationConfig.all_models : [];
    if (allModels.length) return allModels;
    return modelRows.map((model) => ({ ...model, model_kind: "glm", model_ref: `glm:${model.model_id}` }));
  }

  function tabulationTableSelectorShellHtml(selectedIds = []) {
    return tabulations.tableSelectorShellHtml(selectedIds);
  }

  function activeTabulationTable() {
    return (tabulationConfig?.tables || []).find((table) => String(table.table_id || "") === selectedTabulationTableId) || null;
  }

  function tabulationTableLabel(table = {}) {
    return String(table.label || table.table_id || "");
  }

  function tabulationModelLabel(model = {}) {
    const kind = String(model.model_kind || "glm").toUpperCase();
    return `${kind} · ${modelLabel(model)}`;
  }

  function tabulationConfigModel(modelId) {
    const models = Array.isArray(tabulationConfig?.all_models) ? tabulationConfig.all_models : (tabulationConfig?.models || []);
    const ref = normaliseTabulationRef(modelId);
    return models.find((model) => tabulationModelRef(model) === ref || String(model.model_id || "") === String(modelId || "")) || null;
  }

  function tabulationSelectionConfigFromCache(modelRefs = tabulationSelectedModelIds()) {
    const allModels = Array.isArray(tabulationConfig?.all_models) ? tabulationConfig.all_models : [];
    if (!allModels.length || !modelRefs.length) return null;
    const byRef = new Map(allModels.map((model) => [tabulationModelRef(model), model]));
    const models = modelRefs.map((modelRef) => byRef.get(normaliseTabulationRef(modelRef))).filter(Boolean);
    if (models.length !== modelRefs.length) return null;
    const tablesById = new Map();
    const warnings = [];
    models.forEach((model) => {
      modelTabulationTables(model).forEach((table) => {
        const tableId = String(table.table_id || "");
        if (tableId && !tablesById.has(tableId)) tablesById.set(tableId, table);
      });
      (Array.isArray(model.warnings) ? model.warnings : []).forEach((warning) => {
        if (warning) warnings.push(String(warning));
      });
    });
    const tables = Array.from(tablesById.values()).sort((left, right) => (
      (tabulationTableIndex(left) - tabulationTableIndex(right))
      || String(left.table_id || "").localeCompare(String(right.table_id || ""))
    ));
    return { ...tabulationConfig, models, tables, warnings };
  }

  function selectedTabulationModel() {
    const refs = tabulationSelectedModelIds();
    return refs.length === 1 ? tabulationConfigModel(refs[0]) : null;
  }

  function canExportSelectedTabulations() {
    const refs = tabulationSelectedModelIds();
    const model = refs.length === 1 ? tabulationConfigModel(refs[0]) : null;
    return Boolean(refs.length === 1 && model?.tabulated && !isTabulating && !isExportingTabulations);
  }

  function selectedTabulationRebaseRules() {
    const model = selectedTabulationModel();
    const rules = model?.rebasing?.rules;
    return Array.isArray(rules) ? rules : [];
  }

  function selectedTabulationGeneratedTables() {
    const model = selectedTabulationModel();
    const generated = model?.rebasing?.generated_tables;
    return Array.isArray(generated) ? generated : [];
  }

  function tabulationRebaseRuleTarget(rule = {}) {
    return String(rule.target_table_id || rule.transfer_feature || "base");
  }

  function selectedTableHasRebaseInvolvement() {
    const tableId = String(selectedTabulationTableId || "");
    return Boolean(tableId && selectedTabulationRebaseRules().some((rule) => (
      String(rule.table_id || "") === tableId || tabulationRebaseRuleTarget(rule) === tableId
    )));
  }

  function canRebaseSelectedTable() {
    const model = selectedTabulationModel();
    const modelRef = model ? tabulationModelRef(model) : "";
    const table = activeTabulationTable();
    const features = Array.isArray(table?.features) ? table.features : [];
    return Boolean(
      model
      && modelRef
      && String(model.model_kind || "glm").toLowerCase() === "glm"
      && selectedTabulationTableId !== "base"
      && features.length >= 1
      && tabulationSelectedModelIds().length === 1
    );
  }

  function canRebaseActiveTabulation() {
    return tabulationView === "table" && canRebaseSelectedTable();
  }

  function tabulationRebaseUnavailableMessage() {
    const refs = tabulationSelectedModelIds();
    const table = activeTabulationTable();
    const features = Array.isArray(table?.features) ? table.features : [];
    if (refs.length !== 1) return "Choose one GLM model to rebase.";
    const model = selectedTabulationModel();
    if (!model || String(model.model_kind || "glm").toLowerCase() !== "glm") return "Only GLM tabulations can be rebased.";
    if (!table || selectedTabulationTableId === "base" || !features.length) return "Choose a non-base GLM tabulation table to rebase.";
    if (tabulationView !== "table") return "Switch to table view before rebasing.";
    return "Right-click an OK numeric table cell to choose a rebase action.";
  }

  function tabulationRebaseAnchorLabel(anchorCell = {}, features = []) {
    return features.map((feature) => `${feature}=${anchorCell[feature]}`).join(", ");
  }

  function tabulationRebaseContextsForCell(row = {}, column = {}) {
    if (!canRebaseActiveTabulation() || !column?.tabulation_value) return [];
    const field = String(column.field || "");
    const statusField = String(column.status_field || `__status__${field}`);
    const status = row[statusField] || "ok";
    const value = row[field];
    if (status !== "ok" || value === null || value === undefined || !Number.isFinite(Number(value))) return [];
    const table = activeTabulationTable();
    const features = Array.isArray(table?.features) ? table.features : [];
    const anchorCell = {};
    features.forEach((feature) => {
      anchorCell[feature] = feature === tabulationCrosstab ? column.title : row[feature];
    });
    if (features.some((feature) => anchorCell[feature] === undefined || anchorCell[feature] === null)) return [];
    const modelRef = tabulationSelectedModelIds()[0] || "";
    if (!modelRef) return [];
    const contexts = [{
      model_ref: modelRef,
      table_id: selectedTabulationTableId,
      anchor_cell: anchorCell,
      mode: "cell_to_base",
      anchor_feature: "",
    }];
    if (features.length === 2) {
      features.forEach((anchorFeature) => {
        contexts.push({
          model_ref: modelRef,
          table_id: selectedTabulationTableId,
          anchor_cell: anchorCell,
          mode: "feature_level_to_one_way",
          anchor_feature: anchorFeature,
          target_feature: features.find((feature) => feature !== anchorFeature) || "",
        });
      });
    }
    return contexts;
  }

  function tabulationRebaseContextValueLabel(value) {
    if (value === null || value === undefined) return "";
    return String(value);
  }

  function tabulationRebaseActionLabel(rebaseContext = {}) {
    if (rebaseContext.mode === "feature_level_to_one_way") {
      const anchorFeature = String(rebaseContext.anchor_feature || "");
      const anchorValue = tabulationRebaseContextValueLabel(rebaseContext.anchor_cell?.[anchorFeature]);
      const targetFeature = String(rebaseContext.target_feature || "");
      const baseline = tabulationScale === "exp" ? "1.0000" : "0";
      return `Set ${anchorFeature}=${anchorValue} slice to ${baseline}; adjust ${targetFeature} table`;
    }
    return "Rebase to this cell; adjust base";
  }

  function glmTabulationContextMenu() {
    let menu = document.getElementById("glmTabulationContextMenu");
    if (menu) return menu;
    menu = document.createElement("div");
    menu.id = "glmTabulationContextMenu";
    menu.className = "glm-tabulation-context-menu";
    menu.setAttribute("role", "menu");
    menu.hidden = true;
    document.body.appendChild(menu);
    return menu;
  }

  function closeGlmTabulationContextMenu() {
    if (glmTabulationContextMenuListeners) {
      document.removeEventListener("pointerdown", glmTabulationContextMenuListeners.pointerdown, true);
      document.removeEventListener("keydown", glmTabulationContextMenuListeners.keydown, true);
      window.removeEventListener("resize", glmTabulationContextMenuListeners.viewport, true);
      window.removeEventListener("scroll", glmTabulationContextMenuListeners.viewport, true);
      glmTabulationContextMenuListeners = null;
    }
    const menu = document.getElementById("glmTabulationContextMenu");
    if (!menu) return;
    menu.hidden = true;
    menu.innerHTML = "";
  }

  function positionGlmTabulationContextMenu(menu, event = {}) {
    const margin = 8;
    const x = Number(event.clientX) || margin;
    const y = Number(event.clientY) || margin;
    menu.style.left = `${margin}px`;
    menu.style.top = `${margin}px`;
    menu.hidden = false;
    const rect = menu.getBoundingClientRect();
    const left = Math.max(margin, Math.min(x, window.innerWidth - rect.width - margin));
    const top = Math.max(margin, Math.min(y, window.innerHeight - rect.height - margin));
    menu.style.left = `${left}px`;
    menu.style.top = `${top}px`;
  }

  function bindGlmTabulationContextMenuDismissal() {
    const pointerdown = (event) => {
      const menu = document.getElementById("glmTabulationContextMenu");
      if (!menu || menu.hidden || menu.contains(event.target)) return;
      closeGlmTabulationContextMenu();
    };
    const keydown = (event) => {
      if (event.key === "Escape") closeGlmTabulationContextMenu();
    };
    const viewport = () => closeGlmTabulationContextMenu();
    glmTabulationContextMenuListeners = { pointerdown, keydown, viewport };
    document.addEventListener("pointerdown", pointerdown, true);
    document.addEventListener("keydown", keydown, true);
    window.addEventListener("resize", viewport, true);
    window.addEventListener("scroll", viewport, true);
  }

  function openGlmTabulationContextMenu(event, rebaseContexts = []) {
    const actions = [];
    rebaseContexts.forEach((rebaseContext) => {
      actions.push({
        label: tabulationRebaseActionLabel(rebaseContext),
        action: () => applyTabulationRebaseContext(rebaseContext),
      });
    });
    if (selectedTabulationRebaseRules().length) {
      if (selectedTableHasRebaseInvolvement()) {
        actions.push({
          label: "Clear rebasing involving this table",
          separatorBefore: true,
          action: () => resetSelectedTabulationRebase("table"),
        });
      }
      actions.push({
        label: "Clear all rebasing",
        separatorBefore: true,
        danger: true,
        action: () => resetSelectedTabulationRebase("all"),
      });
    }
    if (!actions.length) {
      closeGlmTabulationContextMenu();
      return false;
    }
    event?.preventDefault?.();
    event?.stopPropagation?.();
    closeGlmTabulationContextMenu();
    const menu = glmTabulationContextMenu();
    menu.innerHTML = "";
    actions.forEach((item) => {
      if (item.separatorBefore) {
        const divider = document.createElement("div");
        divider.className = "glm-tabulation-context-menu-divider";
        divider.setAttribute("role", "separator");
        menu.appendChild(divider);
      }
      const button = document.createElement("button");
      button.type = "button";
      button.className = `glm-tabulation-context-menu-item${item.danger ? " glm-tabulation-context-menu-item--danger" : ""}`;
      button.setAttribute("role", "menuitem");
      button.textContent = item.label;
      button.addEventListener("click", () => {
        closeGlmTabulationContextMenu();
        item.action();
      });
      menu.appendChild(button);
    });
    positionGlmTabulationContextMenu(menu, event);
    bindGlmTabulationContextMenuDismissal();
    menu.querySelector("button")?.focus?.();
    return true;
  }

  function tabulationModelRows(models = tabulationAvailableModels()) {
    return models.map((model) => {
      const modelRef = tabulationModelRef(model);
      const kind = String(model.model_kind || "glm").toLowerCase() === "gbm" ? "GBM" : "GLM";
      const diagnostics = model.diagnostics || {};
      const tables = Array.isArray(model.tables) ? model.tables : [];
      return {
        model_ref: modelRef,
        model_name: modelLabel(model),
        model_type: kind,
        table_count: tables.length,
        mean_error: modelNumberOrNull(diagnostics.mean_linear_error),
        linear_sd_error: modelNumberOrNull(diagnostics.linear_sd_error),
        missing: modelNumberOrNull(diagnostics.missing_tabulated_prediction_rows),
        tabulated: Boolean(model.tabulated),
        tabulatable: Boolean(model.tabulatable),
      };
    }).filter((row) => row.model_ref);
  }

  function tabulationTableGroups() {
    const models = Array.isArray(tabulationConfig?.models) ? tabulationConfig.models : [];
    if (models.length <= 1) {
      const tables = models.length ? modelTabulationTables(models[0]) : (Array.isArray(tabulationConfig?.tables) ? tabulationConfig.tables : []);
      return {
        single: sortedTabulationTables(tables).map((table) => tabulationTableSelectorRow(table, "single")),
        common: [],
        other: [],
      };
    }
    const byTable = new Map();
    models.forEach((model) => {
      const seen = new Set();
      modelTabulationTables(model).forEach((table) => {
        const tableId = String(table.table_id || "");
        if (!tableId || seen.has(tableId)) return;
        seen.add(tableId);
        const entry = byTable.get(tableId) || { table: table, count: 0, dims: new Set(), indexes: [] };
        entry.count += 1;
        entry.dims.add(tabulationTableDim(table));
        entry.indexes.push(tabulationTableIndex(table));
        byTable.set(tableId, entry);
      });
    });
    const rows = Array.from(byTable.entries()).map(([, entry]) => {
      const dim = entry.dims.size === 1 ? Array.from(entry.dims)[0] : null;
      const index = Math.min(...entry.indexes.filter((value) => Number.isFinite(value)));
      const tableIndex = Number.isFinite(index) ? index : null;
      return {
        ...tabulationTableSelectorRow(entry.table, entry.count === models.length && dim !== null ? "common" : "other"),
        table_index: tableIndex,
        dim,
        sort_index: tableIndex === null ? tabulationTableIndex(entry.table) : tableIndex,
      };
    });
    rows.sort(compareTabulationSelectorRows);
    return {
      single: [],
      common: rows.filter((row) => row.section === "common"),
      other: rows.filter((row) => row.section === "other"),
    };
  }

  function modelTabulationTables(model = {}) {
    return Array.isArray(model.tables) ? model.tables : [];
  }

  function sortedTabulationTables(tables = []) {
    return [...tables].sort((left, right) => compareTabulationSelectorRows(tabulationTableSelectorRow(left), tabulationTableSelectorRow(right)));
  }

  function compareTabulationSelectorRows(left, right) {
    return (left.sort_index - right.sort_index) || String(left.table_name || "").localeCompare(String(right.table_name || ""));
  }

  function tabulationTableSelectorRow(table = {}, section = "single") {
    const min = modelNumberOrNull(table.min);
    const max = modelNumberOrNull(table.max);
    const span = min !== null && max !== null ? max - min : null;
    const tableIndex = tabulationTableDisplayIndex(table);
    return {
      table_id: String(table.table_id || ""),
      table_index: tableIndex,
      table_name: tabulationTableLabel(table),
      dim: tabulationTableDim(table),
      cells: Number(table.cell_count || 0),
      min,
      max,
      span,
      skipped: Boolean(table.skipped),
      section,
      sort_index: tableIndex === null ? tabulationTableIndex(table) : tableIndex,
    };
  }

  function tabulationTableDim(table = {}) {
    return Array.isArray(table.features) ? table.features.length : 0;
  }

  function tabulationTableIndex(table = {}) {
    const index = Number(table.index);
    return Number.isFinite(index) ? index : 9999;
  }

  function tabulationTableDisplayIndex(table = {}) {
    const index = Number(table.index);
    return Number.isFinite(index) ? index : null;
  }

  function tabulationCrosstabOptions(features = [], modelIds = tabulationSelectedModelIds()) {
    return tabulations.crosstabOptions(features, modelIds);
  }

  function tabulationSelectionKey(modelIds = tabulationSelectedModelIds(), tableId = selectedTabulationTableId) {
    return `${modelIds.map(normaliseTabulationRef).join("\u001f")}\u001e${String(tableId || "base")}`;
  }

  function resetTabulationCrosstabDefault() {
    tabulationCrosstab = "";
    tabulationCrosstabManualKey = "";
    tabulationCrosstabDefaultKey = "";
  }

  function syncTabulationCrosstabSelect() {
    const select = el("glmTabulationCrosstab");
    if (select) select.value = tabulationCrosstab;
  }

  function tabulationDistinctValueKey(value) {
    if (value === null || value === undefined) return "__lucidum_missing__";
    try {
      return JSON.stringify(value);
    } catch (_) {
      return String(value);
    }
  }

  function tabulationDefaultCrosstabFromPayload(data = {}, features = []) {
    if (features.length !== 2) return "";
    const rows = Array.isArray(data.rows) ? data.rows : [];
    if (!rows.length) return features[1] || "";
    const counts = features.map((feature) => {
      const values = new Set();
      rows.forEach((row) => values.add(tabulationDistinctValueKey(row?.[feature])));
      return values.size;
    });
    return counts[0] < counts[1] ? features[0] : features[1];
  }

  async function defaultTabulationCrosstabForTable(modelIds = tabulationSelectedModelIds(), tableId = selectedTabulationTableId, features = []) {
    if (features.length !== 2) return "";
    const key = tabulationSelectionKey(modelIds, tableId);
    if (tabulationCrosstabDefaultCache.has(key)) return await tabulationCrosstabDefaultCache.get(key);
    const promise = api("/api/glm/tabulations/table", {
      method: "POST",
      body: JSON.stringify({ model_refs: modelIds, table_id: tableId, scale: tabulationScale, crosstab: "" }),
    })
      .then((data) => tabulationDefaultCrosstabFromPayload(data, features))
      .catch(() => "");
    tabulationCrosstabDefaultCache.set(key, promise);
    const result = await promise;
    tabulationCrosstabDefaultCache.set(key, result);
    return result;
  }

  async function ensureDefaultTabulationCrosstab(modelIds = tabulationSelectedModelIds(), tableId = selectedTabulationTableId) {
    const table = activeTabulationTable();
    const features = Array.isArray(table?.features) ? table.features.map((feature) => String(feature || "")).filter(Boolean) : [];
    const key = tabulationSelectionKey(modelIds, tableId);
    if (tabulationCrosstabManualKey === key) return;
    if (features.length !== 2) {
      tabulationCrosstab = "";
      tabulationCrosstabDefaultKey = "";
      syncTabulationCrosstabSelect();
      return;
    }
    if (tabulationCrosstabDefaultKey === key && features.includes(tabulationCrosstab)) return;
    const defaultCrosstab = await defaultTabulationCrosstabForTable(modelIds, tableId, features);
    if (key !== tabulationSelectionKey()) return;
    if (tabulationCrosstabManualKey === key) return;
    tabulationCrosstab = features.includes(defaultCrosstab) ? defaultCrosstab : "";
    tabulationCrosstabDefaultKey = tabulationCrosstab ? key : "";
    syncTabulationCrosstabSelect();
  }

  function normaliseTabulationCrosstab(options = []) {
    const values = new Set(options.map((option) => option.value));
    if (!values.has(tabulationCrosstab)) tabulationCrosstab = "";
  }

  function tabulationCrosstabOptionsHtml(options = []) {
    return options.map((option) => `<option value="${escapeHtml(option.value)}" ${option.value === tabulationCrosstab ? "selected" : ""}>${escapeHtml(option.label)}</option>`).join("");
  }

  function tabulationDiagnosticsHtml() {
    const models = Array.isArray(tabulationConfig?.models) ? tabulationConfig.models : [];
    const rows = models
      .map((model) => {
        const warnings = Array.isArray(model.warnings) ? model.warnings.filter(Boolean) : [];
        const rules = Array.isArray(model?.rebasing?.rules) ? model.rebasing.rules : [];
        const generatedTables = Array.isArray(model?.rebasing?.generated_tables) ? model.rebasing.generated_tables : [];
        if (!model.tabulatable && !warnings.length) warnings.push("rebuild required");
        if (!warnings.length && !rules.length && !generatedTables.length) return "";
        return `
          <div class="glm-tabulation-model-diagnostic">
            <strong>${escapeHtml(tabulationModelLabel(model))}</strong>
            ${warnings.slice(0, 3).map((warning) => `<span class="glm-tabulation-warning">${escapeHtml(warning)}</span>`).join("")}
            ${rules.map((rule) => `<span class="glm-tabulation-rebase-rule">${escapeHtml(tabulationRebaseRuleLabel(rule))}</span>`).join("")}
            ${generatedTables.map((table) => `<span class="glm-tabulation-rebase-table">${escapeHtml(tabulationGeneratedTableLabel(table))}</span>`).join("")}
          </div>
        `;
      })
      .filter(Boolean)
      .join("");
    return rows;
  }

  function tabulationRebaseRuleLabel(rule = {}) {
    const anchor = rule.anchor_cell || {};
    const anchorLabel = Object.entries(anchor).map(([feature, value]) => `${feature}=${value}`).join(", ");
    const table = String(rule.table_id || "").replaceAll("|", " × ");
    if (rule.mode === "feature_level_to_one_way") {
      const anchorFeature = String(rule.anchor_feature || "");
      const anchorValue = rule.anchor_level ?? anchor[anchorFeature];
      return `Rebased ${table}: ${anchorFeature}=${anchorValue} slice`;
    }
    if (rule.mode === "legacy_slice_to_one_way" || (!rule.mode && rule.transfer_feature)) {
      return `Legacy rebase ${table} at ${anchorLabel || "selected cell"}`;
    }
    return `Rebased ${table} at ${anchorLabel || "selected cell"}`;
  }

  function tabulationGeneratedTableLabel(table = {}) {
    const tableId = String(table.label || table.table_id || "");
    return `Created ${tableId} one-way adjustment table for rebasing`;
  }

  function refreshTabulationDiagnostics() {
    const diagnostics = el("glmTabulationDiagnostics");
    if (!diagnostics) return;
    const html = tabulationDiagnosticsHtml();
    diagnostics.innerHTML = html;
    diagnostics.classList.toggle("hidden", !html);
  }

  function tabulationPanelModeSignatureValue(modelIds = tabulationSelectedModelIds()) {
    return modelIds.length > 1 ? "multi" : "single";
  }

  function tabulationModelSelectorSignatureValue(rows = tabulationModelRows()) {
    return JSON.stringify(rows.map((row) => [
      row.model_ref,
      row.model_name,
      row.model_type,
      row.table_count,
      row.mean_error,
      row.linear_sd_error,
      row.missing,
      row.tabulated,
      row.tabulatable,
    ]));
  }

  function tabulationTableSelectorSignatureValue(groups = tabulationTableGroups()) {
    const displayRows = (rows = []) => tabulationDisplayTableRows(rows).map((row) => [
      row.table_id,
      row.table_index,
      row.table_name,
      row.dim,
      row.cells,
      row.display_min,
      row.display_max,
      row.display_span,
    ]);
    return JSON.stringify({
      mode: tabulationPanelModeSignatureValue(),
      scale: tabulationScale,
      single: displayRows(groups.single),
      common: displayRows(groups.common),
      other: displayRows(groups.other),
    });
  }

  function renderTabulationsPanel(options = {}) {
    renderTabulationShell({ ...options, force: options.force !== false });
  }

  function renderTabulationShell(options = {}) {
    const panel = el("glmTabulationsPanel");
    if (!panel) return false;
    const nextModeSignature = tabulationPanelModeSignatureValue();
    const hasShell = Boolean(panel.querySelector(".glm-tabulation-layout"));
    if (!options.force && hasShell && tabulationPanelModeSignature === nextModeSignature) return false;
    disconnectTabulationResizeObserver();
    disposeTabulationChart();
    disposeTabulationTable();
    disposeTabulationSelectorTables();
    panel.innerHTML = tabulationsPanelHtml();
    tabulationPanelModeSignature = nextModeSignature;
    tabulationModelSelectorSignature = "";
    tabulationTableSelectorSignature = "";
    bindTabulationControls();
    bindTabulationResizer();
    observeTabulationLayoutResize();
    renderTabulationSelectorTables({ force: true });
    syncTabulationControls();
    return true;
  }

  function ensureTabulationShell() {
    return renderTabulationShell({ force: false });
  }

  function syncTabulationControls() {
    const selectedIds = tabulationSelectedModelIds();
    const activeTable = activeTabulationTable();
    const features = Array.isArray(activeTable?.features) ? activeTable.features : [];
    if (features.length > 2 && tabulationView === "plot") {
      tabulationView = "table";
      localStorage.setItem("py_lucidum_glm_tabulation_view", tabulationView);
    }
    document.querySelectorAll("[data-glm-tabulation-view]").forEach((button) => {
      const view = button.dataset.glmTabulationView || "table";
      const active = view === tabulationView;
      button.classList.toggle("active", active);
      button.setAttribute("aria-pressed", active ? "true" : "false");
      button.disabled = view === "plot" && features.length > 2;
    });
    document.querySelectorAll("[data-glm-tabulation-view-panel]").forEach((panel) => {
      panel.classList.toggle("hidden", panel.dataset.glmTabulationViewPanel !== tabulationView);
    });
    document.querySelectorAll("[data-glm-tabulation-scale]").forEach((button) => {
      const active = tabulationScale === "exp";
      button.classList.toggle("active", active);
      button.setAttribute("aria-pressed", active ? "true" : "false");
    });
    const colorButton = el("glmTabulationColorBtn");
    if (colorButton) {
      colorButton.classList.toggle("active", tabulationColor);
      colorButton.setAttribute("aria-pressed", tabulationColor ? "true" : "false");
    }
    const crosstabOptions = tabulationCrosstabOptions(features, selectedIds);
    normaliseTabulationCrosstab(crosstabOptions);
    const crosstabSelect = el("glmTabulationCrosstab");
    if (crosstabSelect) {
      const crosstabHtml = tabulationCrosstabOptionsHtml(crosstabOptions);
      if (crosstabSelect.innerHTML !== crosstabHtml) crosstabSelect.innerHTML = crosstabHtml;
      crosstabSelect.disabled = crosstabOptions.length <= 1;
      crosstabSelect.value = tabulationCrosstab;
    }
    const sections = el("glmTabulationTableSections");
    if (sections) {
      const multi = selectedIds.length > 1;
      sections.classList.toggle("multi", multi);
      sections.classList.toggle("single", !multi);
    }
    const buildButton = el("glmBuildTabulationsBtn");
    if (buildButton) {
      const availableModels = tabulationAvailableModels();
      buildButton.disabled = isTabulating || !availableModels.length;
      buildButton.classList.toggle("building", isTabulating);
      buildButton.textContent = isTabulating ? "Tabulating..." : "Tabulate";
      syncButtonBusyState(buildButton, isTabulating);
    }
    const exportButton = el("glmExportTabulationsBtn");
    if (exportButton) syncTabulationExportButton(exportButton);
    refreshTabulationDiagnostics();
  }

  function selectTabulationModel(modelId, event = {}) {
    const previousKey = tabulationSelectionKey();
    const modelRef = normaliseTabulationRef(modelId);
    const orderedIds = tabulationAvailableModels().map((model) => tabulationModelRef(model)).filter(Boolean);
    if (!orderedIds.includes(modelRef)) return false;
    const current = new Set(tabulationSelectedModelIds());
    const commandSelection = Boolean(event.metaKey || event.ctrlKey);
    let next;
    if (event.shiftKey) {
      const anchor = orderedIds.includes(tabulationSelectionAnchorModelId)
        ? tabulationSelectionAnchorModelId
        : (Array.from(current).find((candidate) => orderedIds.includes(candidate)) || modelRef);
      const start = orderedIds.indexOf(anchor);
      const end = orderedIds.indexOf(modelRef);
      const min = Math.min(start, end);
      const max = Math.max(start, end);
      const range = orderedIds.slice(min, max + 1);
      next = commandSelection ? new Set(current) : new Set();
      range.forEach((candidate) => next.add(candidate));
    } else if (commandSelection) {
      next = new Set(current);
      if (next.has(modelRef)) next.delete(modelRef);
      else next.add(modelRef);
    } else {
      next = new Set([modelRef]);
    }
    if (!next.size) next.add(modelRef);
    selectedTabulationModelIds = next;
    tabulationSelectionAnchorModelId = modelRef;
    const changed = previousKey !== tabulationSelectionKey();
    if (changed) resetTabulationCrosstabDefault();
    closeGlmTabulationContextMenu();
    return changed;
  }

  async function refreshTabulationSelectionFromCache() {
    const seq = tabulationSelectionRefreshSeq + 1;
    tabulationSelectionRefreshSeq = seq;
    const modelRefs = tabulationSelectedModelIds();
    const nextConfig = tabulationSelectionConfigFromCache(modelRefs);
    if (!nextConfig) {
      await refreshTabulationConfig({ force: false });
      return;
    }
    tabulationConfig = nextConfig;
    tabulationPayload = null;
    const previousTableId = selectedTabulationTableId;
    const tables = Array.isArray(tabulationConfig.tables) ? tabulationConfig.tables : [];
    if (tables.length && !tables.some((table) => String(table.table_id || "") === selectedTabulationTableId)) {
      selectedTabulationTableId = String(tables[0]?.table_id || "base");
    }
    if (previousTableId !== selectedTabulationTableId) resetTabulationCrosstabDefault();
    setGlmNotice("");
    const shellRebuilt = ensureTabulationShell();
    if (!shellRebuilt) {
      await renderTabulationSelectorTables();
      if (seq !== tabulationSelectionRefreshSeq) return;
      syncTabulationControls();
    }
    if (seq !== tabulationSelectionRefreshSeq) return;
    await loadTabulationView();
  }

  function bindTabulationControls() {
    bindTabulationFallbackSelectors();
    document.querySelectorAll("[data-glm-tabulation-view]").forEach((button) => {
      button.addEventListener("click", () => {
        if (button.disabled) return;
        const nextView = button.dataset.glmTabulationView || "table";
        if (nextView === tabulationView) {
          syncTabulationControls();
          return;
        }
        tabulationView = nextView;
        localStorage.setItem("py_lucidum_glm_tabulation_view", tabulationView);
        syncTabulationControls();
        loadTabulationView();
      });
    });
    document.querySelectorAll("[data-glm-tabulation-scale]").forEach((button) => {
      button.addEventListener("click", () => {
        tabulationScale = tabulationScale === "exp" ? "linear" : "exp";
        localStorage.setItem("py_lucidum_glm_tabulation_scale", tabulationScale);
        syncTabulationControls();
        renderTabulationSelectorTables({ forceTables: true });
        loadTabulationView();
      });
    });
    el("glmTabulationColorBtn")?.addEventListener("click", () => {
      tabulationColor = !tabulationColor;
      localStorage.setItem("py_lucidum_glm_tabulation_color", String(tabulationColor));
      syncTabulationControls();
      if (tabulationView === "table" && Array.isArray(tabulationPayload?.rows) && Array.isArray(tabulationPayload?.columns)) {
        renderTabulationTable(tabulationPayload);
      } else {
        loadTabulationView();
      }
    });
    el("glmTabulationCrosstab")?.addEventListener("change", (event) => {
      tabulationCrosstab = event.target.value || "";
      tabulationCrosstabManualKey = tabulationSelectionKey();
      tabulationCrosstabDefaultKey = "";
      closeGlmTabulationContextMenu();
      syncTabulationControls();
      loadTabulationView();
    });
    el("glmBuildTabulationsBtn")?.addEventListener("click", buildSelectedTabulations);
    el("glmExportTabulationsBtn")?.addEventListener("click", exportSelectedTabulations);
    el("glmTabulationCopyBtn")?.addEventListener("click", copyTabulationChartToClipboard);
  }

  function bindTabulationFallbackSelectors() {
    document.querySelectorAll("[data-glm-tabulation-model-id]").forEach((row) => {
      row.addEventListener("click", (event) => {
        const modelId = String(row.dataset.glmTabulationModelId || "");
        if (!modelId) return;
        if (selectTabulationModel(modelId, event)) refreshTabulationSelectionFromCache();
      });
      row.addEventListener("keydown", (event) => {
        if (event.key !== "Enter" && event.key !== " ") return;
        event.preventDefault();
        const modelId = String(row.dataset.glmTabulationModelId || "");
        if (!modelId) return;
        if (selectTabulationModel(modelId, event)) refreshTabulationSelectionFromCache();
      });
    });
    document.querySelectorAll("[data-glm-tabulation-table-id]").forEach((row) => {
      row.addEventListener("click", () => selectTabulationTable(row.dataset.glmTabulationTableId || "base"));
      row.addEventListener("keydown", (event) => {
        if (event.key !== "Enter" && event.key !== " ") return;
        event.preventDefault();
        selectTabulationTable(row.dataset.glmTabulationTableId || "base");
      });
    });
  }

  function selectTabulationTable(tableId) {
    const nextTableId = String(tableId || "base") || "base";
    if (nextTableId === selectedTabulationTableId) return;
    const previousKey = tabulationSelectionKey();
    selectedTabulationTableId = nextTableId;
    localStorage.setItem("py_lucidum_glm_tabulation_table", selectedTabulationTableId);
    if (previousKey !== tabulationSelectionKey()) resetTabulationCrosstabDefault();
    closeGlmTabulationContextMenu();
    const table = activeTabulationTable();
    const features = Array.isArray(table?.features) ? table.features : [];
    if (features.length > 2 && tabulationView === "plot") {
      tabulationView = "table";
      localStorage.setItem("py_lucidum_glm_tabulation_view", tabulationView);
    }
    syncTabulationTableSelectorSelection();
    syncTabulationControls();
    loadTabulationView();
  }

  async function renderTabulationSelectorTables(options = {}) {
    const modelRows = tabulationModelRows();
    const tableGroups = tabulationTableGroups();
    const nextModelSignature = tabulationModelSelectorSignatureValue(modelRows);
    const nextTableSignature = tabulationTableSelectorSignatureValue(tableGroups);
    const shouldRenderModels = Boolean(options.force || options.forceModel || nextModelSignature !== tabulationModelSelectorSignature);
    const shouldRenderTables = Boolean(options.force || options.forceTables || nextTableSignature !== tabulationTableSelectorSignature);
    if (!shouldRenderModels && !shouldRenderTables) {
      syncTabulationModelSelectorSelection();
      syncTabulationTableSelectorSelection();
      return;
    }
    const seq = tabulationSelectorRenderSeq + 1;
    tabulationSelectorRenderSeq = seq;
    if (shouldRenderModels) clearTabulationModelSelectorFallback();
    if (shouldRenderTables) clearTabulationTableSelectorFallbacks();
    try {
      const Tabulator = await loadTabulator();
      if (seq !== tabulationSelectorRenderSeq) return;
      if (shouldRenderModels) {
        disposeTabulationModelSelectorTable();
        renderTabulationModelSelectorGrid(Tabulator, modelRows);
        tabulationModelSelectorSignature = nextModelSignature;
      }
      if (shouldRenderTables) {
        disposeTabulationTableSelectorTables();
        renderTabulationTableSelectorGrids(Tabulator, tableGroups);
        tabulationTableSelectorSignature = nextTableSignature;
      }
    } catch (_) {
      if (seq !== tabulationSelectorRenderSeq) return;
      if (shouldRenderModels) {
        disposeTabulationModelSelectorTable();
        renderTabulationModelSelectorFallback(modelRows);
        tabulationModelSelectorSignature = nextModelSignature;
      }
      if (shouldRenderTables) {
        disposeTabulationTableSelectorTables();
        renderTabulationTableSelectorFallbacks(tableGroups);
        tabulationTableSelectorSignature = nextTableSignature;
      }
    }
  }

  function renderTabulationModelSelectorGrid(Tabulator, rows = []) {
    const target = el("glmTabulationModelGrid");
    if (!target) return;
    target.classList.remove("hidden");
    tabulationModelTable = new Tabulator(target, {
      data: rows,
      index: "model_ref",
      height: "100%",
      layout: "fitColumns",
      placeholder: "No models",
      selectableRows: true,
      rowFormatter: formatTabulationModelSelectorRow,
      columns: [
        { title: "Model name", field: "model_name", sorter: "string", formatter: tabulationTextFormatter, minWidth: 150, widthGrow: 1 },
        { title: "Model type", field: "model_type", sorter: "string", formatter: tabulationTextFormatter, width: 78 },
        { title: "Number of tables", field: "table_count", sorter: "number", formatter: tabulationTableCountFormatter, hozAlign: "right", headerHozAlign: "right", width: 112 },
        { title: "Mean error", field: "mean_error", sorter: "number", formatter: tabulationModelMetricFormatter, hozAlign: "right", headerHozAlign: "right", width: 96 },
        { title: "linear SD error", field: "linear_sd_error", sorter: "number", formatter: tabulationModelMetricFormatter, hozAlign: "right", headerHozAlign: "right", width: 112 },
        { title: "missing", field: "missing", sorter: "number", formatter: tabulationModelIntegerFormatter, hozAlign: "right", headerHozAlign: "right", width: 78 },
      ],
    });
    tabulationModelTable.on("rowClick", (event, row) => {
      const data = row.getData() || {};
      const modelRef = String(data.model_ref || "");
      if (!modelRef) return;
      const changed = selectTabulationModel(modelRef, event);
      syncTabulationModelSelectorSelection();
      if (changed) refreshTabulationSelectionFromCache();
    });
    tabulationModelTable.on("tableBuilt", syncTabulationModelSelectorSelection);
    syncTabulationModelSelectorSelection();
    window.setTimeout(syncTabulationModelSelectorSelection, 0);
  }

  function formatTabulationModelSelectorRow(row) {
    const data = row.getData() || {};
    const element = row.getElement();
    element.classList.toggle("glm-tabulation-model-untabulated", !data.tabulated);
  }

  function renderTabulationTableSelectorGrids(Tabulator, groups = tabulationTableGroups()) {
    if (tabulationSelectedModelIds().length > 1) {
      tabulationCommonTable = renderTabulationTableSelectorGrid(Tabulator, "glmTabulationCommonTableGrid", tabulationDisplayTableRows(groups.common), true, "No common tables");
      tabulationOtherTable = renderTabulationTableSelectorGrid(Tabulator, "glmTabulationOtherTableGrid", tabulationDisplayTableRows(groups.other), true, "No other tables");
    } else {
      tabulationCommonTable = renderTabulationTableSelectorGrid(Tabulator, "glmTabulationTableGrid", tabulationDisplayTableRows(groups.single), false, "No tabulations built");
      tabulationOtherTable = null;
    }
    syncTabulationTableSelectorSelection();
  }

  function renderTabulationTableSelectorGrid(Tabulator, elementId, rows = [], multiModel = false, placeholder = "No tables") {
    const target = el(elementId);
    if (!target) return null;
    target.classList.remove("hidden");
    const indexColumn = {
      title: "#",
      field: "table_index",
      sorter: "number",
      formatter: tabulationIntegerFormatter,
      hozAlign: "right",
      headerHozAlign: "right",
      width: 42,
      minWidth: 42,
      widthGrow: 0,
    };
    const columns = multiModel
      ? [
        indexColumn,
        { title: "Table name", field: "table_name", sorter: "string", formatter: tabulationTextFormatter, minWidth: 180, widthGrow: 2 },
        { title: "Dim", field: "dim", sorter: "number", formatter: tabulationDimFormatter, hozAlign: "right", headerHozAlign: "right", width: 54 },
      ]
      : [
        indexColumn,
        { title: "Table name", field: "table_name", sorter: "string", formatter: tabulationTextFormatter, minWidth: 180, widthGrow: 2 },
        { title: "Dim", field: "dim", sorter: "number", formatter: tabulationDimFormatter, hozAlign: "right", headerHozAlign: "right", width: 54 },
        { title: "Cells", field: "cells", sorter: "number", formatter: tabulationIntegerFormatter, hozAlign: "right", headerHozAlign: "right", width: 78 },
        { title: "Min", field: "display_min", sorter: "number", formatter: tabulationMetricFormatter, hozAlign: "right", headerHozAlign: "right", width: 86 },
        { title: "Max", field: "display_max", sorter: "number", formatter: tabulationMetricFormatter, hozAlign: "right", headerHozAlign: "right", width: 86 },
        { title: "Span", field: "display_span", sorter: "number", formatter: tabulationMetricFormatter, hozAlign: "right", headerHozAlign: "right", width: 86 },
      ];
    const table = new Tabulator(target, {
      data: rows,
      index: "table_id",
      height: "100%",
      layout: "fitColumns",
      placeholder,
      selectableRows: 1,
      columns,
    });
    table.on("rowClick", (_, row) => {
      const tableId = String(row.getData()?.table_id || "");
      if (tableId) selectTabulationTable(tableId);
    });
    table.on("tableBuilt", syncTabulationTableSelectorSelection);
    window.setTimeout(syncTabulationTableSelectorSelection, 0);
    return table;
  }

  function tabulationDisplayTableRows(rows = []) {
    return rows.map((row) => ({
      ...row,
      display_min: tabulationDisplayTableValue(row.min),
      display_max: tabulationDisplayTableValue(row.max),
      display_span: tabulationDisplayTableSpan(row.min, row.max),
    }));
  }

  function tabulationDisplayTableValue(value) {
    return tabulations.displayTableValue(value, tabulationScale);
  }

  function tabulationDisplayTableSpan(min, max) {
    return tabulations.displayTableSpan(min, max, tabulationScale);
  }

  function syncTabulationModelSelectorSelection() {
    const selected = new Set(tabulationSelectedModelIds());
    syncTabulationFallbackModelSelectorSelection(selected);
    if (!tabulatorReady(tabulationModelTable)) return;
    try {
      tabulationModelTable.deselectRow();
      selected.forEach((modelRef) => tabulationModelTable.selectRow(modelRef));
    } catch (_) {
    }
  }

  function syncTabulationTableSelectorSelection() {
    syncTabulationFallbackTableSelectorSelection();
    [tabulationCommonTable, tabulationOtherTable].forEach((table) => {
      if (!tabulatorReady(table)) return;
      try {
        table.deselectRow();
        const row = tabulationTableSelectorRowForId(table, selectedTabulationTableId);
        if (row) row.select();
      } catch (_) {
      }
    });
  }

  function tabulationTableSelectorRowForId(table, tableId) {
    const target = String(tableId || "");
    if (!target) return null;
    try {
      return table.getRows().find((row) => String(row.getData()?.table_id || "") === target) || null;
    } catch (_) {
      return null;
    }
  }

  function syncTabulationFallbackModelSelectorSelection(selected = new Set(tabulationSelectedModelIds())) {
    document.querySelectorAll("[data-glm-tabulation-model-id]").forEach((row) => {
      const isSelected = selected.has(String(row.dataset.glmTabulationModelId || ""));
      row.classList.toggle("selected", isSelected);
      row.setAttribute("aria-selected", isSelected ? "true" : "false");
    });
  }

  function syncTabulationFallbackTableSelectorSelection() {
    document.querySelectorAll("[data-glm-tabulation-table-id]").forEach((row) => {
      const isSelected = String(row.dataset.glmTabulationTableId || "base") === selectedTabulationTableId;
      row.classList.toggle("selected", isSelected);
      row.setAttribute("aria-selected", isSelected ? "true" : "false");
    });
  }

  function clearTabulationModelSelectorFallback() {
    const target = el("glmTabulationModelFallback");
    if (target) target.innerHTML = "";
  }

  function clearTabulationTableSelectorFallbacks() {
    ["glmTabulationTableFallback", "glmTabulationCommonTableFallback", "glmTabulationOtherTableFallback"].forEach((id) => {
      const target = el(id);
      if (target) target.innerHTML = "";
    });
  }

  function renderTabulationModelSelectorFallback(rows = []) {
    const grid = el("glmTabulationModelGrid");
    const target = el("glmTabulationModelFallback");
    if (grid) grid.classList.add("hidden");
    if (!target) return;
    if (!rows.length) {
      target.innerHTML = '<div class="glm-empty-state">No models</div>';
      return;
    }
    const selected = new Set(tabulationSelectedModelIds());
    target.innerHTML = `
      <table class="glm-table glm-tabulation-selector-table">
        <thead><tr><th>Model name</th><th>Model type</th><th class="numeric">Number of tables</th><th class="numeric">Mean error</th><th class="numeric">linear SD error</th><th class="numeric">missing</th></tr></thead>
        <tbody>
          ${rows.map((row) => `
            <tr data-glm-tabulation-model-id="${escapeHtml(row.model_ref)}" class="${!row.tabulated ? "untabulated " : ""}${selected.has(row.model_ref) ? "selected" : ""}" tabindex="0" aria-selected="${selected.has(row.model_ref) ? "true" : "false"}">
              <td>${escapeHtml(row.model_name)}</td>
              <td>${escapeHtml(row.model_type)}</td>
              <td class="numeric">${row.tabulated ? escapeHtml(formatTabulationInteger(row.table_count)) : "not tabulated"}</td>
              <td class="numeric">${row.tabulated ? escapeHtml(formatModelMetric(row.mean_error)) : ""}</td>
              <td class="numeric">${row.tabulated ? escapeHtml(formatModelMetric(row.linear_sd_error)) : ""}</td>
              <td class="numeric">${row.tabulated ? escapeHtml(formatTabulationInteger(row.missing)) : ""}</td>
            </tr>
          `).join("")}
        </tbody>
      </table>
    `;
    bindTabulationFallbackSelectors();
  }

  function renderTabulationTableSelectorFallbacks(groups = tabulationTableGroups()) {
    if (tabulationSelectedModelIds().length > 1) {
      renderTabulationTableSelectorFallback("glmTabulationCommonTableGrid", "glmTabulationCommonTableFallback", tabulationDisplayTableRows(groups.common), true, "No common tables");
      renderTabulationTableSelectorFallback("glmTabulationOtherTableGrid", "glmTabulationOtherTableFallback", tabulationDisplayTableRows(groups.other), true, "No other tables");
      return;
    }
    renderTabulationTableSelectorFallback("glmTabulationTableGrid", "glmTabulationTableFallback", tabulationDisplayTableRows(groups.single), false, "No tabulations built");
  }

  function renderTabulationTableSelectorFallback(gridId, fallbackId, rows = [], multiModel = false, emptyText = "No tables") {
    const grid = el(gridId);
    const target = el(fallbackId);
    if (grid) grid.classList.add("hidden");
    if (!target) return;
    if (!rows.length) {
      target.innerHTML = `<div class="glm-empty-state">${escapeHtml(emptyText)}</div>`;
      return;
    }
    const headers = multiModel
      ? "<th class=\"numeric\">#</th><th>Table name</th><th class=\"numeric\">Dim</th>"
      : "<th class=\"numeric\">#</th><th>Table name</th><th class=\"numeric\">Dim</th><th class=\"numeric\">Cells</th><th class=\"numeric\">Min</th><th class=\"numeric\">Max</th><th class=\"numeric\">Span</th>";
    target.innerHTML = `
      <table class="glm-table glm-tabulation-selector-table">
        <thead><tr>${headers}</tr></thead>
        <tbody>
          ${rows.map((row) => `
            <tr data-glm-tabulation-table-id="${escapeHtml(row.table_id)}" class="${row.table_id === selectedTabulationTableId ? "selected" : ""}" tabindex="0" aria-selected="${row.table_id === selectedTabulationTableId ? "true" : "false"}">
              <td class="numeric">${escapeHtml(formatTabulationInteger(row.table_index))}</td>
              <td>${escapeHtml(row.table_name)}</td>
              <td class="numeric">${escapeHtml(formatTabulationDim(row.dim))}</td>
              ${multiModel ? "" : `
                <td class="numeric">${escapeHtml(formatTabulationInteger(row.cells))}</td>
                <td class="numeric">${escapeHtml(formatModelMetric(row.display_min))}</td>
                <td class="numeric">${escapeHtml(formatModelMetric(row.display_max))}</td>
                <td class="numeric">${escapeHtml(formatModelMetric(row.display_span))}</td>
              `}
            </tr>
          `).join("")}
        </tbody>
      </table>
    `;
    bindTabulationFallbackSelectors();
  }

  function tabulationTextFormatter(cell) {
    return escapeHtml(cell.getValue() ?? "");
  }

  function tabulationTableCountFormatter(cell) {
    const row = cell.getRow().getData() || {};
    if (!row.tabulated) return "not tabulated";
    return escapeHtml(formatTabulationInteger(cell.getValue()));
  }

  function tabulationModelMetricFormatter(cell) {
    if (!cell.getRow().getData()?.tabulated) return "";
    return tabulationMetricFormatter(cell);
  }

  function tabulationModelIntegerFormatter(cell) {
    if (!cell.getRow().getData()?.tabulated) return "";
    return tabulationIntegerFormatter(cell);
  }

  function tabulationMetricFormatter(cell) {
    return escapeHtml(formatModelMetric(cell.getValue()));
  }

  function tabulationIntegerFormatter(cell) {
    return escapeHtml(formatTabulationInteger(cell.getValue()));
  }

  function tabulationDimFormatter(cell) {
    return escapeHtml(formatTabulationDim(cell.getValue()));
  }

  function formatTabulationInteger(value) {
    const number = modelNumberOrNull(value);
    return number === null ? "--" : Math.round(number).toLocaleString();
  }

  function formatTabulationDim(value) {
    const number = modelNumberOrNull(value);
    return number === null ? "--" : Math.round(number).toLocaleString();
  }

  async function refreshTabulationConfig(options = {}) {
    tabulationSelectionRefreshSeq += 1;
    let model_ids = tabulationSelectedModelIds();
    const previousKey = tabulationSelectionKey(model_ids, selectedTabulationTableId);
    selectedTabulationModelIds = new Set(model_ids);
    tabulationPayload = null;
    try {
      if (!model_ids.length) {
        tabulationConfig = await api("/api/glm/tabulations/config", { method: "POST", body: JSON.stringify({ model_refs: [] }) });
        const discoveredModels = Array.isArray(tabulationConfig?.all_models) ? tabulationConfig.all_models : [];
        const defaultModel = discoveredModels.find((model) => model.active) || discoveredModels[0] || null;
        const defaultModelRef = defaultModel ? tabulationModelRef(defaultModel) : "";
        if (!defaultModelRef) {
          resetTabulationCrosstabDefault();
          const shellRebuilt = ensureTabulationShell();
          if (!shellRebuilt) {
            await renderTabulationSelectorTables({ force: Boolean(options.force) });
            syncTabulationControls();
          }
          return;
        }
        model_ids = [defaultModelRef];
        selectedTabulationModelIds = new Set(model_ids);
      }
      tabulationConfig = await api("/api/glm/tabulations/config", { method: "POST", body: JSON.stringify({ model_refs: model_ids }) });
      tabulationCrosstabDefaultCache.clear();
      const tables = Array.isArray(tabulationConfig?.tables) ? tabulationConfig.tables : [];
      if (tables.length && !tables.some((table) => String(table.table_id || "") === selectedTabulationTableId)) {
        selectedTabulationTableId = String(tables[0]?.table_id || "base");
      }
      if (previousKey !== tabulationSelectionKey(tabulationSelectedModelIds(), selectedTabulationTableId)) resetTabulationCrosstabDefault();
      setGlmNotice("");
      const shellRebuilt = ensureTabulationShell();
      if (!shellRebuilt) {
        await renderTabulationSelectorTables({
          force: Boolean(options.force),
          forceModel: Boolean(options.forceModel),
          forceTables: Boolean(options.forceTables),
        });
        syncTabulationControls();
      }
      await loadTabulationView();
    } catch (error) {
      setGlmNotice(error.message);
    }
  }

  async function buildSelectedTabulations() {
    if (isTabulating) return;
    tabulationElapsedStartedAt = performance.now();
    const model_ids = tabulationSelectedModelIds();
    if (!model_ids.length) {
      tabulationElapsedStartedAt = null;
      setGlmNotice("Choose at least one model to tabulate");
      return;
    }
    isTabulating = true;
    tabulationOperationId = createOperationId("glm-tabulation");
    liveProgress = { phase: "queued", message: "Tabulating GLM..." };
    renderLiveProgress(liveProgress);
    renderTabulationsPanel();
    try {
      const job = await api("/api/glm/tabulations/build", {
        method: "POST",
        body: JSON.stringify({ model_refs: model_ids }),
        operationId: tabulationOperationId,
      });
      pollTabulationJob(job.job_id);
    } catch (error) {
      setTabulationFailure(error.message);
    }
  }

  async function exportSelectedTabulations() {
    if (isExportingTabulations) return;
    if (!canExportSelectedTabulations()) {
      setInlineTabulationNotice(["Choose exactly one tabulated model to export."]);
      return;
    }
    const modelRefs = tabulationSelectedModelIds();
    isExportingTabulations = true;
    setInlineTabulationNotice(["Saving XLSX..."]);
    syncTabulationControls();
    try {
      const result = await api("/api/glm/tabulations/export", {
        method: "POST",
        body: JSON.stringify({ model_refs: modelRefs, scale: tabulationScale }),
      });
      setInlineTabulationNotice([`Saved XLSX: ${result.path || result.filename || "XLSX saved"}`]);
    } catch (error) {
      setInlineTabulationNotice([error.message]);
    } finally {
      isExportingTabulations = false;
      syncTabulationControls();
    }
  }

  function pollTabulationJob(jobId) {
    if (tabulationPollTimer) window.clearTimeout(tabulationPollTimer);
    const poll = async () => {
      try {
        const job = await api(`/api/glm/tabulations/jobs/${encodeURIComponent(jobId)}`, {
          method: "GET",
          operationId: tabulationOperationId,
        });
        const progress = job.progress || { phase: job.status, message: job.status };
        liveProgress = progress;
        renderLiveProgress(liveProgress);
        if (isModelJobPending(job.status)) {
          tabulationPollTimer = window.setTimeout(poll, modelJobPollDelay(job.status, GLM_QUEUED_POLL_MS, GLM_RUNNING_POLL_MS));
          return;
        }
        tabulationPollTimer = null;
        isTabulating = false;
        if (job.status === "succeeded") {
          liveProgress = null;
          await reloadSchema(state.source || "dataset", {});
          clearCachesAfterGlmModelSourceChange();
          renderExpectedNumerators();
          renderFeatures();
          updateAxisControls();
          const latest = await api("/api/glm/config", { method: "GET", clientTiming: true });
          config = latest;
          modelRows = normaliseModels(latest.models || []);
          setGlmModelCount(modelRows.length);
          await refreshTabulationConfig({ force: true });
          renderLiveProgress(liveProgress);
          setAppReadyStatus("Ready");
          tabulationElapsedStartedAt = null;
          tabulationOperationId = "";
        } else {
          setTabulationFailure(job.error || progress.message || "Model tabulation failed");
        }
      } catch (error) {
        setTabulationFailure(error.message);
      }
    };
    poll();
  }

  function setTabulationFailure(message) {
    if (tabulationPollTimer) {
      window.clearTimeout(tabulationPollTimer);
      tabulationPollTimer = null;
    }
    isTabulating = false;
    setAppReadyStatus("Ready");
    tabulationElapsedStartedAt = null;
    tabulationOperationId = "";
    liveProgress = { phase: "failed", message: String(message || "Model tabulation failed") };
    renderLiveProgress(liveProgress);
    renderTabulationsPanel();
  }

  async function refreshAfterTabulationArtifactMutation() {
    await reloadSchema(state.source || "dataset", {});
    clearCachesAfterGlmModelSourceChange();
    renderExpectedNumerators();
    renderFeatures();
    updateAxisControls();
    await refreshTabulationConfig({ force: true });
    setAppReadyStatus("Ready");
  }

  async function applyTabulationRebaseContext(rebaseContext = {}) {
    if (isRebasing) return;
    if (!rebaseContext.model_ref || !rebaseContext.table_id || !rebaseContext.anchor_cell) {
      setInlineTabulationNotice([tabulationRebaseUnavailableMessage()]);
      return;
    }
    isRebasing = true;
    try {
      await api("/api/glm/tabulations/rebase", {
        method: "POST",
        body: JSON.stringify({
          model_ref: rebaseContext.model_ref,
          table_id: rebaseContext.table_id,
          anchor_cell: rebaseContext.anchor_cell,
          mode: rebaseContext.mode || "cell_to_base",
          anchor_feature: rebaseContext.anchor_feature || "",
        }),
      });
      await refreshAfterTabulationArtifactMutation();
    } catch (error) {
      setInlineTabulationNotice([error.message]);
    } finally {
      isRebasing = false;
    }
  }

  async function resetSelectedTabulationRebase(scope = "all") {
    if (isRebasing) return;
    const modelRef = tabulationSelectedModelIds()[0] || "";
    if (!modelRef || !selectedTabulationRebaseRules().length) return;
    isRebasing = true;
    try {
      await api("/api/glm/tabulations/rebase/reset", {
        method: "POST",
        body: JSON.stringify({
          model_ref: modelRef,
          scope,
          table_id: scope === "table" ? selectedTabulationTableId : "",
        }),
      });
      await refreshAfterTabulationArtifactMutation();
    } catch (error) {
      setInlineTabulationNotice([error.message]);
    } finally {
      isRebasing = false;
    }
  }

  async function loadTabulationView() {
    if (activeTab !== "tabulations") return;
    const model_ids = tabulationSelectedModelIds();
    const table_id = selectedTabulationTableId || "base";
    if (!model_ids.length || !table_id || !(tabulationConfig?.tables || []).length) {
      tabulationPayload = null;
      refreshTabulationDiagnostics();
      renderTabulationEmpty("Build tabulations to view rating tables");
      return;
    }
    tabulationPayload = null;
    refreshTabulationDiagnostics();
    const seq = tabulationRenderSeq + 1;
    tabulationRenderSeq = seq;
    try {
      await ensureDefaultTabulationCrosstab(model_ids, table_id);
      if (seq !== tabulationRenderSeq) return;
      syncTabulationControls();
      const payload = { model_refs: model_ids, table_id, scale: tabulationScale, crosstab: tabulationCrosstab };
      if (tabulationView === "plot") {
        const data = await api("/api/glm/tabulations/plot", { method: "POST", body: JSON.stringify(payload) });
        if (seq !== tabulationRenderSeq) return;
        tabulationPayload = { ...data, crosstab: tabulationCrosstab };
        refreshTabulationDiagnostics();
        renderTabulationPlot(tabulationPayload);
      } else {
        const data = await api("/api/glm/tabulations/table", { method: "POST", body: JSON.stringify(payload) });
        if (seq !== tabulationRenderSeq) return;
        tabulationPayload = data;
        refreshTabulationDiagnostics();
        await renderTabulationTable(data, seq);
      }
    } catch (error) {
      tabulationPayload = null;
      refreshTabulationDiagnostics();
      renderTabulationEmpty(error.message);
    }
  }

  function renderTabulationEmpty(message) {
    const fallback = el("glmTabulationFallback");
    const grid = el("glmTabulationTable");
    const plot = el("glmTabulationPlot");
    tabulationPayload = null;
    refreshTabulationDiagnostics();
    disposeTabulationTable();
    disposeTabulationChart();
    tabulationFallbackRows = [];
    tabulationFallbackColumns = [];
    if (grid) grid.innerHTML = "";
    if (fallback) fallback.innerHTML = `<div class="glm-empty-state">${escapeHtml(message)}</div>`;
    if (plot) plot.innerHTML = `<div class="glm-empty-state">${escapeHtml(message)}</div>`;
    syncTabulationCopyButton();
  }

  async function renderTabulationTable(data = {}, renderSeq = null) {
    const grid = el("glmTabulationTable");
    const fallback = el("glmTabulationFallback");
    if (!grid || !fallback) return;
    const rows = Array.isArray(data.rows) ? data.rows : [];
    const columns = Array.isArray(data.columns) ? data.columns : [];
    if (renderSeq !== null && renderSeq !== tabulationRenderSeq) return;
    setInlineTabulationNotice(data.notices || []);
    if (!rows.length || !columns.length) {
      renderTabulationEmpty("No rows for this tabulation");
      return;
    }
    try {
      const Tabulator = await loadTabulator();
      if (renderSeq !== null && renderSeq !== tabulationRenderSeq) return;
      disposeTabulationTable();
      grid.innerHTML = "";
      fallback.innerHTML = "";
      const columnMetadataByField = new Map(
        columns.map((column) => [String(column.field || ""), column]),
      );
      tabulationTable = new Tabulator("#glmTabulationTable", {
        data: rows,
        height: "100%",
        layout: "fitDataStretch",
        placeholder: "No rows",
        columns: columns.map((column) => tabulationColumnDefinition(column, data)),
      });
      tabulationTable.on(
        "cellContext",
        (event, cell) => openGlmTabulationContextMenuForTabulatorCell(event, cell, columnMetadataByField),
      );
    } catch (_) {
      if (renderSeq !== null && renderSeq !== tabulationRenderSeq) return;
      disposeTabulationTable();
      grid.innerHTML = "";
      fallback.innerHTML = "";
      renderTabulationFallbackTable(columns, rows, data);
    }
  }

  function openGlmTabulationContextMenuForTabulatorCell(event, cell, columnMetadataByField) {
    const row = cell?.getRow?.().getData?.() || {};
    const field = String(cell?.getField?.() || "");
    const column = columnMetadataByField?.get(field) || {};
    openGlmTabulationContextMenu(event, tabulationRebaseContextsForCell(row, column));
  }

  function tabulationColumnDefinition(column, data = {}) {
    const field = String(column.field || "");
    const tabulationValue = Boolean(column.tabulation_value);
    const numeric = tabulationValue || tabulationSelectedModelIds().includes(field);
    const statusField = String(column.status_field || `__status__${field}`);
    return {
      title: column.title ?? field,
      field,
      formatter: (cell) => {
        if (!numeric) return escapeHtml(cell.getValue() ?? "");
        const row = cell.getRow().getData();
        const status = row[statusField] || "ok";
        const value = cell.getValue();
        const element = cell.getElement();
        element.classList.toggle("glm-tabulation-na-cell", status !== "ok" || value === null || value === undefined);
        element.classList.toggle("glm-tabulation-rebase-cell", canRebaseActiveTabulation() && tabulationValue && status === "ok" && value !== null && value !== undefined);
        if (tabulationColor && status === "ok" && value !== null && value !== undefined) {
          const color = tabulationCellColor(value, data.min, data.max);
          if (color) {
            element.classList.add("glm-tabulation-colour-cell");
            element.style.setProperty("--glm-tabulation-cell-bg", color);
            element.style.setProperty("background", color, "important");
            return value === null || value === undefined ? "NA" : escapeHtml(tabulationValue ? formatTabulationValue(value) : formatModelMetric(value));
          }
        }
        element.classList.remove("glm-tabulation-colour-cell");
        element.style.removeProperty("--glm-tabulation-cell-bg");
        element.style.removeProperty("background");
        return value === null || value === undefined ? "NA" : escapeHtml(tabulationValue ? formatTabulationValue(value) : formatModelMetric(value));
      },
      hozAlign: numeric ? "right" : "left",
    };
  }

  function formatTabulationValue(value) {
    const number = Number(value);
    if (!Number.isFinite(number)) return formatModelMetric(value);
    if (Object.is(number, -0)) return "0";
    const formatted = number.toFixed(4);
    return /^-0(?:\.0+)?$/.test(formatted) ? "0" : formatted;
  }

  function tabulationCellColor(value, min, max) {
    const number = Number(value);
    const lo = Number(min);
    const hi = Number(max);
    if (!Number.isFinite(number) || !Number.isFinite(lo) || !Number.isFinite(hi) || hi <= lo) return "";
    const ratio = Math.max(0, Math.min(1, (number - lo) / (hi - lo)));
    const hue = 130 - ratio * 130;
    return `color-mix(in srgb, hsl(${hue} 78% 50%) 28%, var(--panel))`;
  }

  function niceTabulationAxisStep(span) {
    return tabulations.niceAxisStep(span);
  }

  function roundTabulationAxisValue(value, step) {
    return tabulations.roundAxisValue(value, step);
  }

  function formatTabulationUpliftPercent(value) {
    return tabulations.formatUpliftPercent(value);
  }

  function formatTabulationAxisTick(value, scale = "linear") {
    return tabulations.formatAxisTick(value, scale);
  }

  function tabulationYAxisOptions(data = {}) {
    return tabulations.yAxisOptions(data);
  }

  function renderTabulationFallbackTable(columns, rows, data = {}) {
    const fallback = el("glmTabulationFallback");
    if (!fallback) return;
    tabulationFallbackRows = rows;
    tabulationFallbackColumns = columns;
    fallback.innerHTML = `
      <table class="glm-table glm-tabulation-fallback-table">
        <thead><tr>${columns.map((column) => `<th>${escapeHtml(column.title || column.field)}</th>`).join("")}</tr></thead>
        <tbody>${rows.map((row, rowIndex) => `<tr data-glm-tabulation-fallback-row-index="${rowIndex}">${columns.map((column, columnIndex) => {
          const value = row[column.field];
          const tabulationValue = Boolean(column.tabulation_value);
          const numeric = tabulationValue || tabulationSelectedModelIds().includes(column.field);
          const status = row[column.status_field || `__status__${column.field}`] || "ok";
          const rebasable = tabulationRebaseContextsForCell(row, column).length > 0;
          const style = numeric && tabulationColor && status === "ok" ? ` style="background:${tabulationCellColor(value, data.min, data.max)}"` : "";
          return `<td data-glm-tabulation-fallback-cell="true" data-glm-tabulation-fallback-row-index="${rowIndex}" data-glm-tabulation-fallback-column-index="${columnIndex}" class="${numeric ? "numeric" : ""}${status !== "ok" ? " glm-tabulation-na-cell" : ""}${rebasable ? " glm-tabulation-rebase-cell" : ""}"${style}>${value === null || value === undefined ? "NA" : escapeHtml(tabulationValue ? formatTabulationValue(value) : (numeric ? formatModelMetric(value) : value))}</td>`;
        }).join("")}</tr>`).join("")}</tbody>
      </table>
    `;
    fallback.querySelectorAll("[data-glm-tabulation-fallback-cell]").forEach((cell) => {
      cell.addEventListener("contextmenu", openGlmTabulationContextMenuForFallbackCell);
    });
  }

  function openGlmTabulationContextMenuForFallbackCell(event) {
    const target = event.currentTarget;
    const rowIndex = Number(target?.dataset?.glmTabulationFallbackRowIndex);
    const columnIndex = Number(target?.dataset?.glmTabulationFallbackColumnIndex);
    const row = Number.isInteger(rowIndex) ? tabulationFallbackRows[rowIndex] || {} : {};
    const column = Number.isInteger(columnIndex) ? tabulationFallbackColumns[columnIndex] || {} : {};
    openGlmTabulationContextMenu(event, tabulationRebaseContextsForCell(row, column));
  }

  function renderTabulationPlot(data = {}) {
    const plot = el("glmTabulationPlot");
    if (!plot) return;
    setInlineTabulationNotice(data.notices || []);
    disposeTabulationChart();
    if (!data.plottable || !Array.isArray(data.series) || !data.series.length) {
      plot.innerHTML = (data.notices || []).filter(Boolean).length
        ? ""
        : '<div class="glm-empty-state">Plot is unavailable for this table</div>';
      return;
    }
    if (!window.echarts) {
      plot.innerHTML = `<div class="glm-empty-state">ECharts is not available</div>`;
      return;
    }
    plot.innerHTML = "";
    tabulationChart = window.echarts.init(plot);
    const theme = tabulationChartTheme();
    const xAxisPresentation = tabulations.xAxisPresentation(
      data,
      tabulationChart.getWidth?.() || plot.clientWidth,
      theme,
    );
    const series = data.series.map((item) => ({ ...item }));
    const baselineMarkLine = tabulations.baselineMarkLine(data, theme);
    if (baselineMarkLine && series.length) {
      series[0].markLine = baselineMarkLine;
    }
    tabulationChart.setOption({
      animation: false,
      tooltip: { trigger: "axis", valueFormatter: (value) => formatTabulationAxisTick(value, data.scale) },
      legend: { type: "scroll", top: 4, right: 8 },
      grid: { left: 54, right: 24, top: 48, ...xAxisPresentation.grid },
      xAxis: xAxisPresentation.xAxis,
      yAxis: tabulationYAxisOptions(data),
      dataZoom: xAxisPresentation.dataZoom,
      series,
    });
    syncTabulationCopyButton();
  }

  function tabulationChartTheme() {
    const style = window.getComputedStyle(document.body);
    return {
      panel: style.getPropertyValue("--panel").trim() || "#fff",
      text: style.getPropertyValue("--text").trim() || "#334155",
      line: style.getPropertyValue("--line").trim() || "#cbd5e1",
    };
  }

  async function copyTabulationChartToClipboard() {
    if (!tabulationChart || !navigator.clipboard?.write || typeof window.ClipboardItem !== "function") {
      showClipboardToast("Could not copy tabulation chart image", true);
      return;
    }
    try {
      const dataUrl = tabulationChart.getDataURL({
        type: "png",
        pixelRatio: 2,
        backgroundColor: tabulationChartTheme().panel,
      });
      const blob = await fetch(dataUrl).then((response) => response.blob());
      await navigator.clipboard.write([new window.ClipboardItem({ "image/png": blob })]);
      showClipboardToast("Tabulation chart image copied");
    } catch (_) {
      showClipboardToast("Could not copy tabulation chart image", true);
    }
  }

  function syncTabulationCopyButton() {
    const button = el("glmTabulationCopyBtn");
    if (button) button.disabled = !tabulationChart;
  }

  function refreshTabulationChartXAxis() {
    const plot = el("glmTabulationPlot");
    if (!tabulationChart || !tabulationPayload || !plot) return;
    const presentation = tabulations.xAxisPresentation(
      tabulationPayload,
      tabulationChart.getWidth?.() || plot.clientWidth,
      tabulationChartTheme(),
    );
    tabulationChart.setOption({
      grid: presentation.grid,
      xAxis: presentation.xAxis,
    });
  }

  function disposeTabulationChart() {
    if (!tabulationChart) return;
    try {
      tabulationChart.dispose();
    } catch (_) {
    }
    tabulationChart = null;
    syncTabulationCopyButton();
  }

  function scheduleTabulationResize() {
    if (tabulationResizeFrame) return;
    tabulationResizeFrame = window.requestAnimationFrame(() => {
      tabulationResizeFrame = null;
      tabulationChart?.resize?.();
      refreshTabulationChartXAxis();
      safeTabulatorRedraw(tabulationModelTable);
      safeTabulatorRedraw(tabulationCommonTable);
      safeTabulatorRedraw(tabulationOtherTable);
      safeTabulatorRedraw(tabulationTable);
    });
  }

  function tabulatorReady(table) {
    const element = table?.element || table?.rowManager?.element?.closest?.(".tabulator") || null;
    return Boolean(table?.initialized && element?.isConnected);
  }

  function safeTabulatorRedraw(table) {
    if (!tabulatorReady(table) || typeof table.redraw !== "function") return;
    try {
      table.redraw(true);
    } catch (_) {
    }
  }

  function observeTabulationLayoutResize() {
    disconnectTabulationResizeObserver();
    if (!window.ResizeObserver) return;
    const main = document.querySelector(".glm-tabulation-main");
    if (!main) return;
    tabulationResizeObserver = observeResize([main], scheduleTabulationResize);
  }

  function disconnectTabulationResizeObserver() {
    tabulationResizeObserver?.disconnect?.();
    tabulationResizeObserver = null;
    if (tabulationResizeFrame) {
      window.cancelAnimationFrame(tabulationResizeFrame);
      tabulationResizeFrame = null;
    }
  }

  function disposeTabulationTable() {
    closeGlmTabulationContextMenu();
    tabulationFallbackRows = [];
    tabulationFallbackColumns = [];
    if (tabulationTable) {
      try {
        tabulationTable.destroy();
      } catch (_) {
      }
    }
    tabulationTable = null;
  }

  function disposeTabulationSelectorTables() {
    tabulationSelectorRenderSeq += 1;
    disposeTabulationModelSelectorTable();
    disposeTabulationTableSelectorTables();
    tabulationModelSelectorSignature = "";
    tabulationTableSelectorSignature = "";
  }

  function disposeTabulationModelSelectorTable() {
    if (!tabulationModelTable) return;
    try {
      tabulationModelTable.destroy();
    } catch (_) {
    }
    tabulationModelTable = null;
  }

  function disposeTabulationTableSelectorTables() {
    [tabulationCommonTable, tabulationOtherTable].forEach((table) => {
      if (!table) return;
      try {
        table.destroy();
      } catch (_) {
      }
    });
    tabulationCommonTable = null;
    tabulationOtherTable = null;
  }

  function setInlineTabulationNotice(notices = []) {
    const notice = el("glmTabulationNotice");
    if (!notice) return;
    const text = notices.filter(Boolean).join(" ");
    notice.textContent = text;
    notice.classList.toggle("hidden", !text);
  }

  function bindTabs(mount) {
    bindToolScreenNavigation(mount.querySelector(".glm-tabs"), (nextTab) => {
      activeTab = nextTab;
      syncToolScreenNavigation(mount.querySelector(".glm-tabs"), activeTab);
      mount.querySelectorAll("[data-glm-panel]").forEach((panel) => panel.classList.toggle("hidden", panel.dataset.glmPanel !== activeTab));
      if (activeTab === "builder") {
        formulaBuilder.resize();
        void syncBuilderToActiveModel();
      }
      if (activeTab === "models") refreshModelListIfNeeded();
      if (activeTab === "tabulations") refreshTabulationConfig({ force: true });
    });
  }

  function bindBuilderControls() {
    formulaBuilder.bindControls();
  }

  function savedTabulationSplitWidthStyle() {
    return tabulations.savedSplitWidthStyle();
  }

  function bindBuilderResizer() {
    formulaBuilder.bindResizer();
  }

  function bindTabulationResizer() {
    tabulations.bindResizer();
  }

  function bindModelActions() {
    el("glmOpenModelFolderBtn")?.addEventListener("click", openSelectedModelFolder);
    el("glmActivateModelBtn")?.addEventListener("click", activateSelectedModel);
    el("glmRenameModelBtn")?.addEventListener("click", renameSelectedModel);
    el("glmDeleteModelBtn")?.addEventListener("click", deleteSelectedModels);
    updateModelActionButtons();
  }

  function buildPayload() {
    const actual = el("actualNumerator")?.value || "";
    const denominatorSelection = getDenominatorSelection();
    return {
      ...formulaBuilder.buildPayload({
      actual,
      denominator: denominatorSelection.value,
      label: `GLM ${glmAutoModelTimeLabel()}`,
      }),
      denominator_source: denominatorSelection.sourceId || "dataset",
    };
  }

  function buildRegularizationPayload() {
    return formulaBuilder.buildRegularizationPayload();
  }

  async function buildModel() {
    if (isBuilding) return;
    if (getDenominatorSelection().metricKind === "prediction") {
      setBuildFailure("Building is unavailable while Denominator is a model prediction. Use GBM init_score for prediction chaining.");
      return;
    }
    buildElapsedStartedAt = performance.now();
    const payload = buildPayload();
    const familyError = validateFamilyParameter(payload.family, payload.family_parameter);
    if (familyError) {
      setBuildFailure(familyError);
      return;
    }
    const regularizationError = validateRegularizationParameter(payload.regularization);
    if (regularizationError) {
      setBuildFailure(regularizationError);
      return;
    }
    if (!payload.response_column && !String(payload.formula || "").includes("~")) {
      setBuildFailure("Choose an Actual metric or enter a full response ~ terms formula");
      return;
    }
    buildOperationId = createOperationId("glm-build");
    try {
      const validation = await api("/api/glm/validate", {
        method: "POST",
        body: JSON.stringify(payload),
        operationId: buildOperationId,
      });
      if (Array.isArray(validation.errors) && validation.errors.length) {
        setBuildFailure(validation.errors.join("; "));
        return;
      }
    } catch (error) {
      setBuildFailure(error.message);
      return;
    }
    setBuildingState(true);
    liveProgress = { phase: "queued", message: "Starting GLM build" };
    renderLiveProgress(liveProgress);
    setGlmNotice("");
    try {
      const job = await api("/api/glm/build", {
        method: "POST",
        body: JSON.stringify(payload),
        operationId: buildOperationId,
      });
      pollBuildJob(job.job_id);
    } catch (error) {
      setBuildFailure(error.message);
    }
  }

  function setBuildFailure(message) {
    if (pollTimer) {
      window.clearTimeout(pollTimer);
      pollTimer = null;
    }
    setBuildingState(false);
    setAppReadyStatus("Ready");
    buildElapsedStartedAt = null;
    buildOperationId = "";
    liveProgress = { phase: "failed", message: String(message || "GLM training did not save a model") };
    renderLiveProgress(liveProgress);
    setGlmNotice("");
  }

  function setBuildingState(active) {
    isBuilding = Boolean(active);
    updateModelActionButtons();
  }

  function validateFamilyParameter(family, rawValue) {
    return formulaBuilder.validateFamilyParameter(family, rawValue);
  }

  function validateRegularizationParameter(regularization = {}) {
    return formulaBuilder.validateRegularizationParameter(regularization);
  }

  function pollBuildJob(jobId) {
    if (pollTimer) window.clearTimeout(pollTimer);
    const poll = async () => {
      try {
        const job = await api(`/api/glm/jobs/${encodeURIComponent(jobId)}`, {
          method: "GET",
          operationId: buildOperationId,
        });
        const progress = job.progress || { phase: job.status, message: job.status };
        liveProgress = progress;
        renderLiveProgress(liveProgress);
        if (isModelJobPending(job.status)) {
          pollTimer = window.setTimeout(poll, modelJobPollDelay(job.status, GLM_QUEUED_POLL_MS, GLM_RUNNING_POLL_MS));
          return;
        }
        pollTimer = null;
        if (job.status === "succeeded") {
          const generation = advanceModelStateGeneration();
          setAppReadyStatus("Finalising GLM", { elapsedStartedAt: buildElapsedStartedAt });
          const latest = await api("/api/glm/config", { method: "GET", clientTiming: true });
          if (!modelStateIsCurrent(generation)) return;
          liveProgress = null;
          await applyModelMutationResult({ model: job.result, config: latest }, {
            modelStateGeneration: generation,
          });
          if (!modelStateIsCurrent(generation)) return;
          setBuildingState(false);
          renderLiveProgress(liveProgress);
          setAppReadyStatus("Ready");
          buildElapsedStartedAt = null;
          buildOperationId = "";
        } else {
          setBuildFailure(job.error || progress.message || "GLM training did not save a model");
        }
      } catch (error) {
        setBuildFailure(error.message);
      }
    };
    poll();
  }

  function buildStatusHtml(progress) {
    if (!progress) return "";
    const main = String(progress.message || progress.phase || "");
    const phase = String(progress.phase || "").trim().toLowerCase();
    const postFit = phase === "scoring" || phase === "writing" || phase === "succeeded";
    const rows = Number(postFit ? (progress.scoring_rows || 0) : (progress.training_rows || 0));
    const cells = Number(progress.cells || 0);
    const detail = rows
      ? `${rows.toLocaleString()} ${postFit ? "rows to score" : "training rows"}`
      : (cells ? `${cells.toLocaleString()} cells` : "");
    return `<span class="glm-build-status-main">${escapeHtml(main)}</span>${detail ? `<span class="glm-build-status-detail">${escapeHtml(detail)}</span>` : ""}`;
  }

  function syncButtonBusyState(button, active) {
    if (!button) return;
    if (active) {
      button.setAttribute("aria-busy", "true");
    } else {
      button.removeAttribute("aria-busy");
    }
  }

  function syncTabulationExportButton(button) {
    if (!button) return;
    const label = isExportingTabulations ? "Exporting XLSX" : "Export XLSX";
    button.disabled = !canExportSelectedTabulations();
    button.classList.toggle("building", isExportingTabulations);
    button.setAttribute("aria-label", label);
    button.setAttribute("title", label);
    syncButtonBusyState(button, isExportingTabulations);
  }

  function renderLiveProgress(progress) {
    if (isBuilding) {
      setAppReadyStatus(glmBuildReadyBadgeLabel(progress), { elapsedStartedAt: buildElapsedStartedAt });
    } else if (isTabulating) {
      setAppReadyStatus(glmTabulationReadyBadgeLabel(progress), { elapsedStartedAt: tabulationElapsedStartedAt });
    }
    const status = el("glmBuildStatus");
    if (!status) return;
    status.innerHTML = buildStatusHtml(progress);
    status.dataset.phase = String(progress?.phase || "");
    status.classList.toggle("hidden", !progress);
    const button = el("glmBuildBtn");
    if (button) {
      button.disabled = isBuilding || getDenominatorSelection().metricKind === "prediction";
      button.classList.toggle("building", isBuilding);
      button.textContent = isBuilding ? "Building..." : "Build GLM";
      syncButtonBusyState(button, isBuilding);
    }
    const tabulationButton = el("glmBuildTabulationsBtn");
    if (tabulationButton) {
      tabulationButton.disabled = isTabulating || !tabulationAvailableModels().length;
      tabulationButton.classList.toggle("building", isTabulating);
      tabulationButton.textContent = isTabulating ? "Tabulating..." : "Tabulate";
      syncButtonBusyState(tabulationButton, isTabulating);
    }
    const exportButton = el("glmExportTabulationsBtn");
    if (exportButton) syncTabulationExportButton(exportButton);
  }

  function syncDenominatorBuildState() {
    const blocked = getDenominatorSelection().metricKind === "prediction";
    const button = el("glmBuildBtn");
    if (button) button.disabled = isBuilding || blocked;
    el("glmModelDenominatorBuildNotice")?.classList.toggle("hidden", !blocked);
  }

  function diagnosticsForActiveModel(activeModelId) {
    const active = modelForActiveModel(activeModelId);
    return active?.diagnostics || active?.metrics || {};
  }

  function modelForActiveModel(activeModelId) {
    if (activeDetail?.manifest && (!activeModelId || activeDetail.manifest.model_id === activeModelId)) return activeDetail.manifest;
    return modelRows.find((model) => model.model_id === activeModelId) || modelRows.find((model) => model.active) || null;
  }

  function activeModelIsPenalized() {
    const model = modelForActiveModel(config?.active_model_id);
    return regularizationMode(model?.regularization) !== "none";
  }

  function regularizationMode(regularization = {}) {
    return String(regularization?.mode || "none").trim().toLowerCase() || "none";
  }

  function regularizationMixLabel(value) {
    const number = modelNumberOrNull(value);
    if (number === null) return "";
    if (number === 0) return "ridge";
    if (number === 1) return "lasso";
    return "elastic net";
  }

  function regularizationLabel(regularization = {}) {
    const mode = regularizationMode(regularization);
    if (mode === "none") return "";
    const alpha = modelNumberOrNull(regularization.selected_alpha ?? regularization.alpha);
    const l1Ratio = regularization.selected_l1_ratio ?? regularization.l1_ratio;
    const mix = Array.isArray(l1Ratio) ? "" : regularizationMixLabel(l1Ratio);
    return [mode, mix, alpha === null ? "" : `alpha=${formatModelMetric(alpha)}`].filter(Boolean).join(" ");
  }

  function nonzeroCoefficientLabel(regularization = {}, diagnostics = {}) {
    const nonzero = modelNumberOrNull(regularization.nonzero_coefficients ?? diagnostics.nonzero_coefficients);
    const total = modelNumberOrNull(regularization.coefficient_count ?? diagnostics.coefficient_count);
    if (nonzero === null) return "";
    return total === null ? String(nonzero) : `${nonzero.toLocaleString()} / ${total.toLocaleString()}`;
  }

  function modelProblemMessages(diagnostics = {}, model = {}, coefficients = []) {
    const messages = [
      ...(Array.isArray(model?.warnings) ? model.warnings : []),
      ...(Array.isArray(diagnostics?.warnings) ? diagnostics.warnings : []),
      ...(Array.isArray(diagnostics?.blocking_warnings) ? diagnostics.blocking_warnings : []),
    ].map((warning) => String(warning || "").trim()).filter(Boolean);
    const penalized = regularizationMode(model?.regularization) !== "none";
    const hasInferenceWarning = messages.some((message) => /coefficient inference|standard errors/i.test(message));
    if (!penalized && !hasInferenceWarning && Array.isArray(coefficients) && coefficients.some((row) => row && row.std_error === null)) {
      messages.push("GLM coefficient inference was not fully available because one or more standard errors were non-finite. Simplify rank-deficient terms, use centered/no-intercept spline syntax, or use ridge/auto regularization.");
    }
    const seen = new Set();
    return messages.filter((message) => {
      if (seen.has(message)) return false;
      seen.add(message);
      return true;
    });
  }

  function diagnosticsHtml(diagnostics = {}, model = {}, coefficients = []) {
    const regularization = model?.regularization || {};
    const penalty = regularizationLabel(regularization);
    const nonzero = nonzeroCoefficientLabel(regularization, diagnostics);
    const problems = modelProblemMessages(diagnostics, model, coefficients);
    const primary = [
      ["Deviance", diagnostics.deviance],
      ["AIC", diagnostics.aic],
      ["Dispersion", diagnostics.dispersion],
    ].filter(([, value]) => value !== undefined && value !== null && value !== "");
    const secondary = [
      ["Penalty", penalty],
      ["Nonzero", nonzero],
      ["NAs in fitted", diagnostics.fitted_na_rows],
    ].filter(([, value]) => value !== undefined && value !== null && value !== "");
    const itemValue = (value) => {
      const number = modelNumberOrNull(value);
      return number === null ? String(value) : formatModelMetric(number);
    };
    const itemHtml = ([label, value]) => `<span><strong>${escapeHtml(label)}:</strong> ${escapeHtml(itemValue(value))}</span>`;
    if (!primary.length && !secondary.length && !problems.length) return "No active model";
    return [
      problems.length ? `<span class="glm-coefficient-meta-row glm-coefficient-meta-warning"><strong>Selected model issue:</strong> ${escapeHtml(problems.join(" "))}</span>` : "",
      primary.length ? `<span class="glm-coefficient-meta-row">${primary.map(itemHtml).join("")}</span>` : "",
      secondary.length ? `<span class="glm-coefficient-meta-row glm-coefficient-meta-row-secondary">${secondary.map(itemHtml).join("")}</span>` : "",
    ].filter(Boolean).join("");
  }

  function syncBuilderFromModelDetail(detail = {}, options = {}) {
    formulaBuilder.syncFromModelDetail(detail, options);
  }

  function coefficientRowsForActiveModel(activeModelId) {
    if (!activeDetail?.manifest) return [];
    if (activeModelId && activeDetail.manifest.model_id !== activeModelId) return [];
    return Array.isArray(activeDetail.coefficients) ? activeDetail.coefficients : [];
  }

  function renderCoefficientTable(rows = []) {
    coefficientRows = rows;
    const table = el("glmCoefficientTable");
    if (!table) return;
    closeGlmCoefficientContextMenu();
    const penalized = activeModelIsPenalized();
    const query = String(el("glmCoefficientSearch")?.value || "").trim().toLowerCase();
    const indexedRows = rows.map((row, index) => ({ row, index }));
    const filtered = query
      ? indexedRows.filter(({ row }) => Object.values(row).some((value) => String(value ?? "").toLowerCase().includes(query)))
      : indexedRows;
    if (!filtered.length) {
      table.innerHTML = `<tbody><tr><td class="glm-empty-cell" colspan="5">No coefficients to show</td></tr></tbody>`;
      return;
    }
    const visibleRows = sortedCoefficientRows(filtered);
    table.innerHTML = `
      <thead>
        <tr>
          ${coefficientSortHeaderHtml("index", "#", true)}
          ${coefficientSortHeaderHtml("term", "term")}
          ${coefficientSortHeaderHtml("estimate", "estimate", true)}
          ${coefficientSortHeaderHtml("std_error", "std.error", true)}
          ${coefficientSortHeaderHtml("p_value", "p.value", true)}
        </tr>
      </thead>
      <tbody>
        ${visibleRows.map(({ row, index }) => `
          <tr class="${penalized ? "" : glmCoefficientPValueClass(row.p_value)}" data-glm-coefficient-index="${index}">
            <td class="numeric">${(index + 1).toLocaleString()}</td>
            <td>${escapeHtml(row.term)}</td>
            <td class="numeric">${escapeHtml(formatModelMetric(row.estimate))}</td>
            <td class="numeric">${penalized ? "" : escapeHtml(formatModelMetric(row.std_error))}</td>
            <td class="numeric">${penalized ? "" : escapeHtml(formatPValue(row.p_value))}</td>
          </tr>
        `).join("")}
      </tbody>
    `;
    table.querySelectorAll("[data-glm-coefficient-sort]").forEach((button) => {
      button.addEventListener("click", () => setCoefficientSort(button.dataset.glmCoefficientSort));
    });
    table.querySelectorAll("[data-glm-coefficient-index]").forEach((row) => {
      row.addEventListener("contextmenu", openGlmCoefficientContextMenuForRow);
    });
  }

  function coefficientSortHeaderHtml(key, label, numeric = false) {
    const active = coefficientSort.key === key;
    const direction = active ? coefficientSort.direction : "";
    const ariaSort = active ? (direction === "desc" ? "descending" : "ascending") : "none";
    return `<th class="glm-coefficient-sort-header${numeric ? " numeric" : ""}" aria-sort="${ariaSort}">
      <button class="glm-coefficient-sort-button" type="button" data-glm-coefficient-sort="${key}" aria-label="Sort by ${escapeHtml(label)}">
        <span>${escapeHtml(label)}</span><span class="glm-coefficient-sort-indicator" aria-hidden="true"></span>
      </button>
    </th>`;
  }

  function setCoefficientSort(key) {
    if (!["index", "term", "estimate", "std_error", "p_value"].includes(key)) return;
    coefficientSort = coefficientSort.key === key
      ? { key, direction: coefficientSort.direction === "asc" ? "desc" : "asc" }
      : { key, direction: "asc" };
    renderCoefficientTable(coefficientRows);
  }

  function sortedCoefficientRows(rows) {
    const direction = coefficientSort.direction === "desc" ? -1 : 1;
    const key = coefficientSort.key;
    return [...rows].sort((left, right) => {
      if (key === "index") return direction * (left.index - right.index);
      if (key === "term") {
        const compared = String(left.row.term || "").localeCompare(String(right.row.term || ""), undefined, { numeric: true, sensitivity: "base" });
        return compared ? direction * compared : left.index - right.index;
      }
      const leftValue = modelNumberOrNull(left.row[key]);
      const rightValue = modelNumberOrNull(right.row[key]);
      if (leftValue === null && rightValue === null) return left.index - right.index;
      if (leftValue === null) return 1;
      if (rightValue === null) return -1;
      const compared = leftValue - rightValue;
      return compared ? direction * compared : left.index - right.index;
    });
  }

  function isGlmInterceptTerm(term) {
    const text = String(term || "").trim().toLowerCase();
    return text === "intercept" || text === "(intercept)";
  }

  function schemaDatasetFeatureNames() {
    return schemaDatasetColumns().map((column) => String(column?.name || "")).filter(Boolean);
  }

  function schemaDatasetColumns() {
    const datasetSource = (state.schema?.data_sources || []).find((source) => source?.id === "dataset");
    return datasetSource?.columns || state.schema?.columns || [];
  }

  function coefficientStoredFeatureNames(row = {}) {
    if (Array.isArray(row.features)) return row.features;
    if (typeof row.features !== "string") return [];
    const text = row.features.trim();
    if (!text) return [];
    try {
      const parsed = JSON.parse(text);
      return Array.isArray(parsed) ? parsed : [];
    } catch (_) {
      return [];
    }
  }

  function coefficientFallbackFeatureNames(row = {}) {
    const term = String(row.term || "");
    if (!term || isGlmInterceptTerm(term)) return [];
    return schemaDatasetFeatureNames().filter((feature) => {
      const pattern = new RegExp(`(^|[^A-Za-z0-9_])${escapeRegExp(feature)}($|[^A-Za-z0-9_])`);
      return pattern.test(term);
    });
  }

  function normaliseCoefficientFeatureNames(names = []) {
    const seen = new Set();
    const features = [];
    for (const value of names) {
      const name = String(value || "").trim();
      if (!name || seen.has(name) || !canNavigateToLineBarFeature(name)) continue;
      seen.add(name);
      features.push(name);
    }
    return features;
  }

  function coefficientContextFeatureNames(row = {}) {
    if (isGlmInterceptTerm(row?.term)) return [];
    const storedFeatures = normaliseCoefficientFeatureNames(coefficientStoredFeatureNames(row));
    if (storedFeatures.length) return storedFeatures;
    return normaliseCoefficientFeatureNames(coefficientFallbackFeatureNames(row));
  }

  function openGlmCoefficientContextMenuForRow(event) {
    closeGlmCoefficientContextMenu();
    const index = Number(event.currentTarget?.dataset?.glmCoefficientIndex);
    const row = Number.isInteger(index) && index >= 0 ? coefficientRows[index] : null;
    const features = coefficientContextFeatureNames(row || {});
    if (!features.length) return;
    event.preventDefault();
    event.stopPropagation();
    const menu = glmCoefficientContextMenu();
    menu.innerHTML = "";
    features.forEach((feature) => {
      const button = document.createElement("button");
      button.type = "button";
      button.className = "glm-coefficient-context-menu-item";
      button.setAttribute("role", "menuitem");
      button.textContent = `Go to Line and Bar (${feature})`;
      button.addEventListener("click", () => {
        closeGlmCoefficientContextMenu();
        goToLineBarCoefficientFeature(feature);
      });
      menu.append(button);
    });
    menu.hidden = false;
    const rowRect = event.currentTarget?.getBoundingClientRect?.() || { left: event.clientX, top: event.clientY, height: 18 };
    const clientX = event.clientX || rowRect.left + 12;
    const clientY = event.clientY || rowRect.top + Math.min(18, Math.max(8, rowRect.height / 2));
    positionGlmCoefficientContextMenu(menu, clientX, clientY);
    menu.querySelector("button")?.focus({ preventScroll: true });
    bindGlmCoefficientContextMenuDismissal();
  }

  function glmCoefficientContextMenu() {
    let menu = document.getElementById("glmCoefficientContextMenu");
    if (menu) return menu;
    menu = document.createElement("div");
    menu.id = "glmCoefficientContextMenu";
    menu.className = "glm-coefficient-context-menu";
    menu.hidden = true;
    menu.setAttribute("role", "menu");
    document.body.append(menu);
    return menu;
  }

  function positionGlmCoefficientContextMenu(menu, clientX, clientY) {
    const margin = 8;
    menu.style.left = "0px";
    menu.style.top = "0px";
    const rect = menu.getBoundingClientRect();
    const maxLeft = Math.max(margin, window.innerWidth - rect.width - margin);
    const maxTop = Math.max(margin, window.innerHeight - rect.height - margin);
    menu.style.left = `${Math.max(margin, Math.min(clientX, maxLeft))}px`;
    menu.style.top = `${Math.max(margin, Math.min(clientY, maxTop))}px`;
  }

  function bindGlmCoefficientContextMenuDismissal() {
    const pointerdown = (event) => {
      const menu = document.getElementById("glmCoefficientContextMenu");
      if (!menu || menu.hidden || menu.contains(event.target)) return;
      closeGlmCoefficientContextMenu();
    };
    const keydown = (event) => {
      if (event.key !== "Escape") return;
      event.preventDefault();
      closeGlmCoefficientContextMenu();
    };
    const viewport = () => closeGlmCoefficientContextMenu();
    glmCoefficientContextMenuListeners = { pointerdown, keydown, viewport };
    document.addEventListener("pointerdown", pointerdown, true);
    document.addEventListener("keydown", keydown, true);
    window.addEventListener("resize", viewport, true);
    window.addEventListener("scroll", viewport, true);
  }

  function closeGlmCoefficientContextMenu() {
    if (glmCoefficientContextMenuListeners) {
      document.removeEventListener("pointerdown", glmCoefficientContextMenuListeners.pointerdown, true);
      document.removeEventListener("keydown", glmCoefficientContextMenuListeners.keydown, true);
      window.removeEventListener("resize", glmCoefficientContextMenuListeners.viewport, true);
      window.removeEventListener("scroll", glmCoefficientContextMenuListeners.viewport, true);
      glmCoefficientContextMenuListeners = null;
    }
    const menu = document.getElementById("glmCoefficientContextMenu");
    if (!menu) return;
    menu.hidden = true;
    menu.innerHTML = "";
  }

  function goToLineBarCoefficientFeature(featureName) {
    const name = String(featureName || "").trim();
    if (!name) return;
    selectExpectedPredictionForModelKind("glm");
    if (typeof navigateToLineBarFeature === "function" && navigateToLineBarFeature(name)) {
      renderExpectedNumerators();
      updateAxisControls();
      return;
    }
    setGlmNotice(`Feature ${name} is not available in Line and Bar`);
  }

  function glmCoefficientPValueClass(value) {
    const number = modelNumberOrNull(value);
    if (number === null) return "";
    if (number < 0.01) return "glm-coefficient-pvalue-low";
    if (number <= 0.05) return "glm-coefficient-pvalue-medium";
    return "glm-coefficient-pvalue-high";
  }

  function formatPValue(value) {
    const number = modelNumberOrNull(value);
    if (number === null) return "--";
    return `${(number * 100).toLocaleString(undefined, { maximumFractionDigits: 2 })}%`;
  }

  function coefficientsAsTsv(rows = coefficientRows) {
    const headers = ["term", "estimate", "std_error", "statistic", "p_value", "ci_lower", "ci_upper"];
    return [headers.join("\t"), ...rows.map((row) => headers.map((header) => String(row[header] ?? "")).join("\t"))].join("\n");
  }

  function copyFormula(formula) {
    copyTextToClipboard(String(formula ?? "")).then((ok) => {
      showClipboardToast(ok ? "Copied formula" : "Copy failed", !ok);
    });
  }

  function copyCoefficients() {
    if (!coefficientRows.length) return;
    copyTextToClipboard(coefficientsAsTsv()).then((ok) => showClipboardToast(ok ? "Copied coefficients" : "Copy failed", !ok));
  }

  function downloadCoefficients() {
    if (!coefficientRows.length) return;
    const headers = ["term", "estimate", "std_error", "statistic", "p_value", "ci_lower", "ci_upper"];
    const csv = [headers.join(","), ...coefficientRows.map((row) => headers.map((header) => csvEscape(row[header])).join(","))].join("\n");
    downloadText("glm_coefficients.csv", csv, "text/csv");
  }

  async function loadModelDetail(modelId) {
    const requestedModelId = String(modelId || "");
    const requestSeq = modelDetailRequestSeq + 1;
    modelDetailRequestSeq = requestSeq;
    try {
      const detail = await api(`/api/glm/models/${encodeURIComponent(requestedModelId)}`, { method: "GET" });
      const detailModelId = String(detail?.manifest?.model_id || requestedModelId);
      if (
        requestSeq !== modelDetailRequestSeq
        || requestedModelId !== currentActiveModelId()
        || detailModelId !== requestedModelId
      ) return;
      activeDetail = detail;
      const syncBuilderDraft = detailModelId !== builderDraftSourceModelId;
      syncBuilderFromModelDetail(activeDetail, { syncBuilderDraft });
      if (syncBuilderDraft) builderDraftSourceModelId = detailModelId;
      const diagnostics = activeDetail?.diagnostics || {};
      const meta = el("glmCoefficientMeta");
      const coefficients = Array.isArray(activeDetail?.coefficients) ? activeDetail.coefficients : [];
      if (meta) meta.innerHTML = diagnosticsHtml(diagnostics, activeDetail?.manifest || {}, coefficients);
      renderCoefficientTable(coefficients);
    } catch (error) {
      if (requestSeq === modelDetailRequestSeq && requestedModelId === currentActiveModelId()) {
        activeDetail = null;
        setGlmNotice(error.message);
      }
    }
  }

  async function syncBuilderToActiveModel() {
    const activeModelId = currentActiveModelId();
    if (!activeModelId || activeModelId === builderDraftSourceModelId) return;
    await loadModelDetail(activeModelId);
  }

  async function renderModelTable(models = modelRows, activeModelId = config?.active_model_id) {
    const grid = el("glmModelGrid");
    const fallback = el("glmModelFallback");
    const preservedIds = Array.from(selectedModelIds);
    const renderSeq = modelTableRenderSeq + 1;
    modelTableRenderSeq = renderSeq;
    const previousTable = modelTable;
    modelTable = null;
    modelTableReady = false;
    previousTable?.destroy();
    if (!grid || !fallback) {
      updateModelActionButtons();
      return;
    }
    grid.innerHTML = "";
    fallback.innerHTML = "";
    const rows = modelNavigator.rows(models, activeModelId);
    try {
      const Tabulator = await loadTabulator();
      if (renderSeq !== modelTableRenderSeq || !grid.isConnected) return;
      const renderedTable = new Tabulator("#glmModelGrid", {
        data: rows,
        height: "100%",
        layout: "fitDataStretch",
        placeholder: "No GLMs built yet",
        initialSort: [{ column: "created_sort", dir: "desc" }],
        selectableRows: true,
        selectableRowsRangeMode: "click",
        columns: [
          { title: "", field: "active", formatter: modelNavigator.activeDotFormatter, hozAlign: "center", headerHozAlign: "center", width: 28, minWidth: 28, headerSort: false, resizable: false },
          { title: "Name", field: "model_label", sorter: "string", formatter: modelNavigator.nameFormatter, widthGrow: 3, headerSort: true },
          { title: "Created", field: "created_sort", sorter: "number", formatter: (cell) => escapeHtml(cell.getRow().getData().created_display), width: 105, headerSort: true },
          { title: "Response", field: "response_column", sorter: "string", formatter: (cell) => escapeHtml(cell.getValue() || ""), widthGrow: 1.4, headerSort: true },
          { title: "Weight", field: "weight_display", sorter: "string", formatter: (cell) => escapeHtml(cell.getValue() || ""), widthGrow: 1.1, headerSort: true },
          { title: "Family", field: "family", sorter: "string", formatter: (cell) => escapeHtml(cell.getValue() || ""), widthGrow: 1.1, headerSort: true },
          { title: "Terms", field: "n_terms", sorter: "number", formatter: (cell) => escapeHtml(modelNavigator.optionalCount(cell.getValue())), hozAlign: "right", headerHozAlign: "right", width: 68, headerSort: true },
          { title: "Features", field: "n_features", sorter: "number", formatter: (cell) => escapeHtml(modelNavigator.optionalCount(cell.getValue())), hozAlign: "right", headerHozAlign: "right", width: 76, headerSort: true },
          { title: "Interactions", field: "n_interactions", sorter: "number", formatter: (cell) => escapeHtml(modelNavigator.optionalCount(cell.getValue())), hozAlign: "right", headerHozAlign: "right", width: 100, headerSort: true },
          { title: "Tabulated", field: "tabulated", sorter: "boolean", formatter: (cell) => cell.getValue() ? "Yes" : "-", width: 82, headerSort: true },
          { title: "Deviance", field: "deviance", sorter: "number", formatter: (cell) => escapeHtml(formatModelMetric(cell.getValue())), hozAlign: "right", headerHozAlign: "right", width: 96, headerSort: true },
          { title: "AIC", field: "aic", sorter: "number", formatter: (cell) => escapeHtml(formatModelMetric(cell.getValue())), hozAlign: "right", headerHozAlign: "right", width: 96, headerSort: true },
          { title: "BIC", field: "bic", sorter: "number", formatter: (cell) => escapeHtml(formatModelMetric(cell.getValue())), hozAlign: "right", headerHozAlign: "right", width: 96, headerSort: true },
          { title: "Rows", field: "training_rows", sorter: "number", formatter: (cell) => Number(cell.getValue() || 0).toLocaleString(), hozAlign: "right", headerHozAlign: "right", width: 86, headerSort: true },
          { title: "Fit time", field: "fit_ms", sorter: "number", formatter: (cell) => escapeHtml(cell.getRow().getData().fit_display), hozAlign: "right", headerHozAlign: "right", width: 84, headerSort: true, tooltip: "Glum coefficient fitting time" },
          { title: "Overall time", field: "elapsed_ms", sorter: "number", formatter: (cell) => escapeHtml(cell.getRow().getData().elapsed_display), hozAlign: "right", headerHozAlign: "right", width: 104, headerSort: true, tooltip: "Full GLM build time, including fitting, scoring, diagnostics, feature importance, and artifact writing" },
        ],
      });
      modelTable = renderedTable;
      renderedTable.on("rowSelectionChanged", syncSelectedModelsFromTable);
      renderedTable.on("tableBuilt", () => {
        if (renderSeq !== modelTableRenderSeq || modelTable !== renderedTable) return;
        modelTableReady = true;
        restoreModelSelection(preservedIds);
        syncSelectedModelsFromTable();
      });
      updateModelActionButtons();
    } catch (_) {
      if (renderSeq !== modelTableRenderSeq) return;
      const failedTable = modelTable;
      modelTable = null;
      modelTableReady = false;
      try {
        failedTable?.destroy?.();
      } catch (_) {
      }
      renderModelFallback(models, activeModelId);
      restoreModelSelection(preservedIds);
      updateModelActionButtons();
    }
  }

  function renderModelFallback(models = modelRows, activeModelId = config?.active_model_id) {
    const target = el("glmModelFallback");
    modelNavigator.renderFallback(target, models, activeModelId);
  }

  function syncSelectedModelsFromTable() {
    selectedModelIds = new Set(selectedModelIdList());
    updateModelActionButtons();
  }

  function selectedModelIdList() {
    return selectedModelIdsFromTableOrFallback({
      table: modelTable,
      fallbackSelector: "#glmModelFallback [data-glm-model-row]",
      rowDataKey: "glmModelRow",
    });
  }

  function restoreModelSelection(ids) {
    selectedModelIds = restoreSharedModelSelection({
      table: modelTable,
      fallbackSelector: "#glmModelFallback [data-glm-model-row]",
      rowDataKey: "glmModelRow",
      ids,
    });
  }

  function updateModelActionButtons() {
    syncSharedModelActionButtons({
      selectedCount: selectedModelIdList().length,
      disabled: isBuilding || Boolean(modelTable && !modelTableReady),
      openFolder: el("glmOpenModelFolderBtn"),
      openFolderPending: modelFolderOpenPending || modelMutationPending > 0,
      activate: el("glmActivateModelBtn"),
      rename: el("glmRenameModelBtn"),
      deleteButton: el("glmDeleteModelBtn"),
    });
  }

  async function refreshModelListIfNeeded(options = {}) {
    if (isBuilding || modelMutationPending > 0) return;
    const now = Date.now();
    if (!options.force && now - modelListLastRefreshAt < GLM_MODEL_LIST_POLL_MS) return;
    modelListLastRefreshAt = now;
    const seq = modelListRefreshSeq + 1;
    modelListRefreshSeq = seq;
    const generation = modelStateGeneration;
    try {
      const data = await api("/api/glm/config", { method: "GET", clientTiming: true });
      if (seq !== modelListRefreshSeq || !modelStateIsCurrent(generation)) return;
      const previousActiveModelId = currentActiveModelId(config);
      const nextActiveModelId = currentActiveModelId(data);
      if (previousActiveModelId !== nextActiveModelId) {
        await applyModelMutationResult({ config: data }, {
          activationOnly: true,
          syncModelMetrics: true,
          modelStateGeneration: generation,
        });
        return;
      }
      config = data;
      modelRows = normaliseModels(data.models || []);
      const cache = toolCache(tool);
      if (cache) cache.data = data;
      setGlmModelCount(modelRows.length);
      renderModelTable(modelRows, data.active_model_id);
      syncSidebarModelChooser(modelRows, data.active_model_id);
    } catch (error) {
      if (seq === modelListRefreshSeq && modelStateIsCurrent(generation)) setGlmNotice(error.message);
    }
  }

  async function activateModel(modelId) {
    if (isBuilding || !modelId) return;
    queuedActivationModelId = modelId;
    advanceModelStateGeneration();
    invalidateLineBar({ pending: state.tool === "line_bar" });
    if (activationPromise) return activationPromise;
    beginModelMutation();
    activationPromise = (async () => {
      while (queuedActivationModelId) {
        const targetModelId = queuedActivationModelId;
        queuedActivationModelId = "";
        const generation = modelStateGeneration;
        try {
          const result = await api(`/api/glm/models/${encodeURIComponent(targetModelId)}/activate`, {
            method: "POST",
            body: "{}",
          });
          if (queuedActivationModelId) continue;
          if (!modelStateIsCurrent(generation)) continue;
          // Explicit activation is authoritative even when the tool's cached config
          // still names the target model and the schema-backed sidebar does not.
          await applyModelMutationResult(result, {
            activationOnly: true,
            syncModelMetrics: true,
            modelStateGeneration: generation,
          });
        } catch (error) {
          if (!queuedActivationModelId) setGlmNotice(error.message);
        }
      }
    })().finally(() => {
      activationPromise = null;
      endModelMutation();
      if (queuedActivationModelId) void activateModel(queuedActivationModelId);
    });
    return activationPromise;
  }

  async function activateSelectedModel() {
    const [modelId] = selectedModelIdList();
    if (modelId) await activateModel(modelId);
  }

  async function openSelectedModelFolder() {
    if (isBuilding || modelFolderOpenPending || modelMutationPending > 0) return;
    const modelIds = selectedModelIdList();
    if (modelIds.length !== 1) return;
    const button = el("glmOpenModelFolderBtn");
    modelFolderOpenPending = true;
    button?.setAttribute("aria-busy", "true");
    updateModelActionButtons();
    try {
      await api(`/api/glm/models/${encodeURIComponent(modelIds[0])}/open-folder`, {
        method: "POST",
        body: "{}",
      });
      setGlmNotice("");
    } catch (error) {
      setGlmNotice(error.message);
    } finally {
      modelFolderOpenPending = false;
      button?.removeAttribute("aria-busy");
      updateModelActionButtons();
    }
  }

  async function renameSelectedModel() {
    if (isBuilding) return;
    const [modelId] = selectedModelIdList();
    if (!modelId) return;
    const newModelId = window.prompt("Rename GLM model", modelId);
    if (newModelId === null) return;
    const trimmed = newModelId.trim();
    if (!trimmed || trimmed === modelId) return;
    invalidateLineBar({ pending: state.tool === "line_bar" });
    const generation = advanceModelStateGeneration();
    beginModelMutation();
    try {
      const result = await api(`/api/glm/models/${encodeURIComponent(modelId)}/rename`, {
        method: "POST",
        body: JSON.stringify({ new_model_id: trimmed }),
      });
      if (!modelStateIsCurrent(generation)) return;
      await applyModelMutationResult(result, {
        renamedFrom: modelId,
        modelStateGeneration: generation,
      });
    } catch (error) {
      if (modelStateIsCurrent(generation)) setGlmNotice(error.message);
    } finally {
      endModelMutation();
    }
  }

  async function deleteSelectedModels() {
    if (isBuilding) return;
    const modelIds = selectedModelIdList();
    if (!modelIds.length) return;
    const label = modelIds.length === 1 ? `GLM model "${modelIds[0]}"` : `${modelIds.length} GLM models`;
    const confirmed = confirm(`Delete ${label}? This deletes the selected .lucidum model folder${modelIds.length === 1 ? "" : "s"}.`);
    if (!confirmed) return;
    invalidateLineBar({ pending: state.tool === "line_bar" });
    const generation = advanceModelStateGeneration();
    beginModelMutation();
    const activeModelIdBeforeDelete = currentActiveModelId();
    let result = null;
    let deletedCount = 0;
    try {
      for (const modelId of modelIds) {
        result = await api(`/api/glm/models/${encodeURIComponent(modelId)}`, { method: "DELETE", body: "{}" });
        deletedCount += 1;
      }
      if (!modelStateIsCurrent(generation)) return;
      await applyModelMutationResult(result, {
        syncModelMetrics: modelIds.includes(activeModelIdBeforeDelete),
        modelStateGeneration: generation,
      });
    } catch (error) {
      try {
        const latest = await api("/api/glm/config", { method: "GET", clientTiming: true });
        if (modelStateIsCurrent(generation)) {
          await applyModelMutationResult({ config: latest }, {
            syncModelMetrics: modelIds.slice(0, deletedCount).includes(activeModelIdBeforeDelete),
            modelStateGeneration: generation,
          });
        }
      } catch (_) {
      }
      const prefix = deletedCount > 0 ? `${deletedCount} deleted. ` : "";
      if (modelStateIsCurrent(generation)) setGlmNotice(`${prefix}${error.message}`);
    } finally {
      endModelMutation();
    }
  }

  function tabulationActivationNeedsConfigRefresh() {
    const explicitRefs = Array.from(selectedTabulationModelIds).map(normaliseTabulationRef).filter(Boolean);
    if (!explicitRefs.length) return true;
    const availableRefs = new Set(tabulationAvailableModels().map((model) => tabulationModelRef(model)).filter(Boolean));
    if (explicitRefs.some((modelRef) => !availableRefs.has(modelRef))) return true;
    const tables = Array.isArray(tabulationConfig?.tables) ? tabulationConfig.tables : [];
    return Boolean(tables.length && !tables.some((table) => String(table.table_id || "") === selectedTabulationTableId));
  }

  async function applyActivationOnlyTabulationUpdate(nextConfig) {
    if (state.tool !== tool || activeTab !== "tabulations" || !el("glmTabulationsPanel")) return false;
    config = nextConfig;
    modelRows = normaliseModels(nextConfig?.models || []);
    syncSidebarModelChooser(modelRows, nextConfig?.active_model_id);
    if (!tabulationConfig || tabulationActivationNeedsConfigRefresh()) {
      await refreshTabulationConfig({ force: false });
      return true;
    }
    ensureTabulationShell();
    await renderTabulationSelectorTables();
    syncTabulationModelSelectorSelection();
    syncTabulationTableSelectorSelection();
    syncTabulationControls();
    return true;
  }

  async function handleExternalModelActivation() {
    if (state.tool !== tool || activeTab !== "tabulations" || !el("glmTabulationsPanel")) return false;
    if (!tabulationConfig || tabulationActivationNeedsConfigRefresh()) {
      await refreshTabulationConfig({ force: false });
      return true;
    }
    ensureTabulationShell();
    await renderTabulationSelectorTables();
    syncTabulationModelSelectorSelection();
    syncTabulationTableSelectorSelection();
    syncTabulationControls();
    return true;
  }

  async function applyModelMutationResult(result, options = {}) {
    const generation = options?.modelStateGeneration ?? modelStateGeneration;
    if (!modelStateIsCurrent(generation)) return false;
    modelDetailRequestSeq += 1;
    invalidateLineBar({ pending: state.tool === "line_bar" });
    formulaBuilder.captureDraft();
    const nextConfig = result.config || config || {};
    const renamedFrom = String(options?.renamedFrom || "");
    const renamedTo = String(result?.model?.model_id || "");
    if (renamedFrom && renamedTo && builderDraftSourceModelId === renamedFrom) {
      builderDraftSourceModelId = renamedTo;
    }
    const activeMetricModel = options?.syncModelMetrics ? activeModelFromConfig(nextConfig) : null;
    const schemaResult = await reloadSchema(preferredModelSource(result, nextConfig), {
      modelKind: "glm",
      activeModel: activeMetricModel,
      activeModelChanged: Boolean(options?.syncModelMetrics),
    });
    if (schemaResult === false || !modelStateIsCurrent(generation)) return false;
    const chartReady = schemaResult?.chartReady !== false;
    const preserveProfile = clearCachesAfterGlmModelSourceChange();
    if (!currentActiveModelId(nextConfig)) builderDraftSourceModelId = "";
    activeDetail = null;
    coefficientRows = [];
    setGlmModelCount(Array.isArray(nextConfig?.models) ? nextConfig.models.length : null);
    setGlmNotice("");
    if (options?.activationOnly && await applyActivationOnlyTabulationUpdate(nextConfig)) {
      // The visible GLM Tabulations selection did not change, so keep the mounted table UI intact.
      await syncBuilderToActiveModel();
    } else if (state.tool === tool) {
      measureToolRender(tool, () => render(nextConfig));
    } else if (preserveProfile) {
      config = nextConfig;
      syncSidebarModelChooser(nextConfig?.models || [], nextConfig?.active_model_id);
    } else {
      config = nextConfig;
      syncSidebarModelChooser(nextConfig?.models || [], nextConfig?.active_model_id);
      if (chartReady) await refreshActiveTool({ force: true });
    }
    renderExpectedNumerators();
    renderFeatures();
    updateAxisControls();
    return modelStateIsCurrent(generation);
  }

  function clearCachesAfterGlmModelSourceChange() {
    const preserveProfile = state.tool === "column_profile";
    clearToolCaches(preserveProfile ? { preserve: ["column_profile"] } : {});
    return preserveProfile;
  }

  function preferredModelSource(result, data) {
    const currentKind = dataSourceById(state.source)?.kind || "";
    const configActiveModel = (data?.models || []).find((item) => item.active);
    const activeModel = configActiveModel;
    const predictionSource = activeModel?.sources?.predictions || configActiveModel?.sources?.predictions || "";
    if (currentKind === "glm_predictions") return predictionSource || "dataset";
    if (!dataSourceById(state.source) && predictionSource) return predictionSource;
    const fallbackModel = (data?.models || []).find((item) => item.active);
    return dataSourceById(state.source) ? state.source || "dataset" : fallbackModel?.sources?.predictions || "dataset";
  }

  function dataSourceById(sourceId) {
    return (state.schema?.data_sources || []).find((source) => source.id === sourceId) || null;
  }

  function currentActiveModelId(data = config) {
    return String(data?.active_model_id || (data?.models || []).find((model) => model.active)?.model_id || "");
  }

  function activeModelFromConfig(data = config) {
    const activeModelId = currentActiveModelId(data);
    return (data?.models || []).find((model) => String(model?.model_id || "") === activeModelId) || null;
  }

  function normaliseModels(models = []) {
    const seen = new Set();
    return models
      .map((model) => ({
        ...model,
        model_id: String(model?.model_id || ""),
        label: String(model?.label || ""),
        response_column: String(model?.response_column || ""),
        denominator_column: String(model?.denominator_column || model?.offset_column || ""),
        family: String(model?.family || ""),
        link: String(model?.link || "auto"),
        training_scope: String(model?.training_scope || "all"),
        training_rows: Number(model?.training_rows || model?.diagnostics?.training_rows || 0),
        diagnostics: model?.diagnostics || model?.metrics || {},
        regularization: model?.regularization || { mode: "none" },
        sources: model?.sources || {},
        created_at: String(model?.created_at || ""),
        active: Boolean(model?.active),
      }))
      .filter((model) => {
        if (!model.model_id || seen.has(model.model_id)) return false;
        seen.add(model.model_id);
        return true;
      });
  }

  function modelGroupLabel(model) {
    return `${model.response_column || "response"} / ${modelWeightLabel(model.denominator_column)}`;
  }

  function modelWeightLabel(value) {
    const text = String(value || "").trim();
    return text || "N";
  }

  function modelLabel(model) {
    return model.label || model.model_id;
  }

  function syncSidebarModelChooser(models, activeModelId) {
    const list = el("glmModelSelect");
    const meta = el("glmModelSelectedMeta");
    if (!list) return;
    const normalisedModels = normaliseModels(models);
    const activeModel = normalisedModels.find((model) => model.model_id === activeModelId) || null;
    if (meta) meta.textContent = activeModel ? modelLabel(activeModel) : "No active model";
    const modelsByGroup = modelGroups(normalisedModels, modelGroupLabel);
    const groups = [...modelsByGroup.keys()];
    state.glmModelGroupsInitialised = syncCollapsedModelGroups({
      groups,
      collapsedGroups: state.collapsedGlmModelGroups,
      initialised: state.glmModelGroupsInitialised,
      activeGroup: activeModel ? modelGroupLabel(activeModel) : "",
    }).initialised;
    list.innerHTML = "";
    if (!normalisedModels.length) {
      list.innerHTML = emptyStateHtml("No GLMs built yet", "glm-empty-state", escapeHtml);
      return;
    }
    for (const group of groups) {
      const collapsed = state.collapsedGlmModelGroups.has(group);
      list.append(createSidebarModelHeading({
        group,
        collapsed,
        toolLabel: "GLM",
        className: "glm-model-theme",
        dataKey: "glmModelGroup",
        escapeHtml,
        onToggle: toggleGlmModelGroup,
      }));
      for (const model of modelsByGroup.get(group) || []) {
        const active = model.model_id === activeModelId;
        list.append(createSidebarModelOption({
          model,
          group,
          active,
          collapsed,
          className: "glm-model-option",
          detailClassName: "glm-model-detail",
          modelIdDataKey: "glmModelId",
          groupDataKey: "glmModelGroup",
          escapeHtml,
          modelLabel,
          modelDetailLabel: glmModelDetailLabel,
          onActivate: activateModel,
        }));
      }
    }
  }

  function toggleGlmModelGroup(group) {
    toggleSidebarModelGroup({
      list: el("glmModelSelect"),
      group,
      collapsedGroups: state.collapsedGlmModelGroups,
      themeClassName: "glm-model-theme",
      optionClassName: "glm-model-option",
      groupDataKey: "glmModelGroup",
      toolLabel: "GLM",
    });
  }

  function syncSidebarFromSchema() {
    const sources = state.schema?.data_sources || [];
    const models = [];
    const seen = new Set();
    for (const source of sources) {
      if (source.kind !== "glm_predictions" || !source.model_id || seen.has(source.model_id)) continue;
      seen.add(source.model_id);
      models.push({
        model_id: source.model_id,
        label: String(source.label || source.model_id).replace(/\s+-\s+Predictions$/i, ""),
        active: Boolean(source.active),
        response_column: source.response_column,
        denominator_column: source.denominator_column || source.offset_column,
        created_at: source.created_at,
        family: source.family,
        link: source.link,
        training_scope: source.training_scope,
        diagnostics: source.metrics || {},
      });
    }
    const activeModel = models.find((model) => model.active)?.model_id || "";
    syncSidebarModelChooser(models, activeModel);
  }

  function setGlmNotice(text) {
    let notice = el("glmNotice");
    if (!notice) {
      const mount = el("modelToolWrap");
      let toolNode = mount?.querySelector(".glm-tool");
      if (!toolNode && mount && text) {
        mount.innerHTML = `<div class="glm-tool glm-tool-error-shell"></div>`;
        toolNode = mount.querySelector(".glm-tool");
      }
      if (!toolNode) return;
      notice = document.createElement("div");
      notice.id = "glmNotice";
      notice.className = "glm-notice hidden";
      notice.setAttribute("role", "alert");
      notice.setAttribute("aria-live", "polite");
      toolNode.prepend(notice);
    }
    notice.textContent = text || "";
    notice.classList.toggle("hidden", !text);
  }

  function openModelNavigator() {
    activeTab = "models";
    if (config) render(config);
  }

  function refreshTheme() {
    formulaBuilder.refreshTheme();
  }

  function resize() {
    scheduleTabulationResize();
    formulaBuilder.resize();
  }

  return {
    buildRequest,
    fetchData,
    handleExternalModelActivation,
    openModelNavigator,
    render,
    refreshTheme,
    resize,
    syncDenominatorBuildState,
    syncSidebarFromSchema,
    useCached,
  };
}
