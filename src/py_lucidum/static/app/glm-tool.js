import { loadTabulator } from "./shared/tabulator.js";

const GLM_RUNNING_POLL_MS = 500;
const GLM_QUEUED_POLL_MS = 1000;
const GLM_MODEL_LIST_POLL_MS = 2000;
const ACE_BASE_PATH = "/static/vendor/ace";
const GLM_BUILDER_SPLIT_STORAGE_KEY = "py_lucidum_glm_formula_panel_width";
const GLM_TABULATION_SPLIT_STORAGE_KEY = "py_lucidum_glm_tabulation_sidebar_width";
const GLM_TABULATION_MODEL_CROSSTAB = "__model__";

function glmAutoModelTimeLabel(date = new Date()) {
  const hour = String(date.getHours()).padStart(2, "0");
  const minute = String(date.getMinutes()).padStart(2, "0");
  const second = String(date.getSeconds()).padStart(2, "0");
  return `${hour}:${minute}:${second}`;
}

function modelNumberOrNull(value) {
  if (value === null || value === undefined || String(value).trim() === "") return null;
  const number = Number(value);
  return Number.isFinite(number) ? number : null;
}

function formatModelMetric(value) {
  const number = modelNumberOrNull(value);
  if (number === null) return "--";
  return number.toLocaleString(undefined, { maximumFractionDigits: 4 });
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

function csvEscape(value) {
  const text = String(value ?? "");
  return /[",\n]/.test(text) ? `"${text.replaceAll('"', '""')}"` : text;
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

function loadScript(src) {
  return new Promise((resolve, reject) => {
    const existing = document.querySelector(`script[src="${src}"]`);
    if (existing) {
      existing.addEventListener("load", resolve, { once: true });
      existing.addEventListener("error", reject, { once: true });
      if (existing.dataset.loaded === "true") resolve();
      return;
    }
    const script = document.createElement("script");
    script.src = src;
    script.async = true;
    script.dataset.loaded = "false";
    script.addEventListener("load", () => {
      script.dataset.loaded = "true";
      resolve();
    }, { once: true });
    script.addEventListener("error", reject, { once: true });
    document.head.append(script);
  });
}

let aceLoaderPromise = null;

async function loadAce() {
  if (window.ace) return window.ace;
  if (!aceLoaderPromise) {
    aceLoaderPromise = loadScript(`${ACE_BASE_PATH}/ace.js`).then(() => window.ace);
  }
  const ace = await aceLoaderPromise;
  if (!ace) throw new Error("Ace editor did not load");
  ace.config.set("basePath", ACE_BASE_PATH);
  return ace;
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
  setDatasetGlmCount = () => {},
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
  updateAxisControls,
  refreshActiveTool,
  reloadSchema,
}) {
  const tool = "glm";
  let activeTab = "builder";
  let config = null;
  let activeDetail = null;
  let coefficientRows = [];
  let modelTable = null;
  let modelTableRenderSeq = 0;
  let modelRows = [];
  let selectedModelIds = new Set();
  let selectedTabulationModelIds = new Set();
  let tabulationSelectionAnchorModelId = "";
  let pollTimer = null;
  let tabulationPollTimer = null;
  let modelListRefreshSeq = 0;
  let modelListLastRefreshAt = 0;
  let isBuilding = false;
  let isTabulating = false;
  let liveProgress = null;
  let tabulationConfig = null;
  let tabulationTable = null;
  let tabulationChart = null;
  let tabulationRenderSeq = 0;
  let selectedTabulationTableId = localStorage.getItem("py_lucidum_glm_tabulation_table") || "base";
  let tabulationView = localStorage.getItem("py_lucidum_glm_tabulation_view") || "table";
  let tabulationScale = localStorage.getItem("py_lucidum_glm_tabulation_scale") || "linear";
  let tabulationColor = localStorage.getItem("py_lucidum_glm_tabulation_color") === "true";
  let tabulationCrosstab = "";
  let aceEditor = null;
  let editorInitialisedFor = null;
  let editorFontSize = Number(localStorage.getItem("py_lucidum_glm_font_size")) || 14;
  let selectedFamily = localStorage.getItem("py_lucidum_glm_family") || "normal";
  let selectedTrainingScope = localStorage.getItem("py_lucidum_glm_training_scope") || "all";
  let selectedRegularizationMode = localStorage.getItem("py_lucidum_glm_regularization_mode") || "none";
  let selectedRegularizationMix = localStorage.getItem("py_lucidum_glm_regularization_mix") || "0.5";
  let selectedRegularizationAlpha = localStorage.getItem("py_lucidum_glm_regularization_alpha") || "0.01";
  let formulaDraft = localStorage.getItem("py_lucidum_glm_formula")
    || "# GLM formula\n# Enter RHS terms, or response ~ terms\n";

  function buildRequest() {
    if (!state.schema) return null;
    return {
      tool,
      source: state.source || "dataset",
    };
  }

  async function fetchData(request, requestKey) {
    const requestSeq = state.glmRequestSeq + 1;
    state.glmRequestSeq = requestSeq;
    setStatus("");
    setChartMessage("");
    setGroupMeta(tool, "Loading GLM...");
    startToolTiming(tool);
    try {
      const data = await api("/api/glm/config", { method: "GET", clientTiming: true });
      if (requestSeq !== state.glmRequestSeq) return null;
      const cache = toolCache(tool);
      cache.requestKey = requestKey;
      cache.data = data;
      setDatasetGlmCount(Array.isArray(data?.models) ? data.models.length : null);
      syncDuckDbTimingFromData(tool, data);
      syncClientTimingFromData(tool, data);
      measureToolRender(tool, () => render(data));
      return data;
    } catch (error) {
      if (requestSeq !== state.glmRequestSeq) return null;
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
    config = data;
    modelRows = normaliseModels(data.models || []);
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
    disposeEditor();
    disposeTabulationChart();
    modelTable = null;
    tabulationTable = null;
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
    initEditor();
    if (data.active_model_id) loadModelDetail(data.active_model_id);
    if (liveProgress) renderLiveProgress(liveProgress);
    if (activeTab === "tabulations") refreshTabulationConfig({ force: true });
    setDuckDbTiming(tool, data.timings || {});
    setClientTiming(tool, data.client_timings || {});
    setRenderTiming(tool, 0);
    saveToolPresentation(tool, { groupMeta, chartMessage: "" });
  }

  function shellHtml(data = {}) {
    const sample = data.sample || {};
    const trainingDisabled = !sample.available || !Number(sample.training_rows || 0);
    if (trainingDisabled && selectedTrainingScope === "training" && !data.active_model_id) selectedTrainingScope = "all";
    const activeModel = modelForActiveModel(data.active_model_id);
    const diagnostics = activeModel?.diagnostics || activeModel?.metrics || {};
    const splitStyle = savedBuilderSplitWidthStyle();
    return `
      <div class="glm-tool">
        <div id="glmNotice" class="glm-notice hidden" role="alert" aria-live="polite"></div>
        <div class="glm-toolbar">
          <div class="glm-tabs tabs workspace-tabs">
            <button class="tab ${activeTab === "builder" ? "active" : ""}" type="button" data-glm-tab="builder">Formula builder</button>
            <button class="tab ${activeTab === "models" ? "active" : ""}" type="button" data-glm-tab="models">Model navigator</button>
            <button class="tab ${activeTab === "tabulations" ? "active" : ""}" type="button" data-glm-tab="tabulations">Tabulations</button>
          </div>
          <div id="glmBuildStatus" class="glm-build-status ${liveProgress ? "" : "hidden"}" aria-live="polite">${buildStatusHtml(liveProgress)}</div>
        </div>
        <div class="glm-tab-panel ${activeTab === "builder" ? "" : "hidden"}" data-glm-panel="builder">
          <div class="glm-builder-layout"${splitStyle ? ` style="${splitStyle}"` : ""}>
            <section class="glm-formula-panel">
              <div class="glm-panel-header">
                <h3 class="glm-panel-title">GLM formula</h3>
                <div class="glm-builder-actions">
                  <button id="glmClearFormulaBtn" class="tab glm-inline-action-button" type="button" title="Clear formula">× clear</button>
                  <button id="glmFontSmallerBtn" class="tab glm-icon-action-button" type="button" aria-label="Decrease formula font size" title="Decrease font size">A-</button>
                  <button id="glmFontLargerBtn" class="tab glm-icon-action-button" type="button" aria-label="Increase formula font size" title="Increase font size">A+</button>
                  <div class="segmented glm-scope-control glm-header-scope-control" role="group" aria-label="Rows to fit">
                    <button type="button" data-glm-scope="all" class="${selectedTrainingScope === "all" ? "active" : ""}">All</button>
                    <button type="button" data-glm-scope="training" class="${selectedTrainingScope === "training" ? "active" : ""}" ${trainingDisabled ? "disabled" : ""}>Training</button>
                  </div>
                  <button id="glmBuildBtn" class="tab glm-build-button ${isBuilding ? "building" : ""}" type="button" ${isBuilding ? "disabled aria-busy=\"true\"" : ""}>${isBuilding ? "Building..." : "Build GLM"}</button>
                </div>
              </div>
              <div class="glm-builder-control-row glm-builder-control-stack">
                <div class="glm-control-line">
                  <div class="glm-family-row">
                    <label class="glm-control-label" for="glmFamilySelect">Family</label>
                    <select id="glmFamilySelect" aria-label="GLM family">${familyOptionsHtml(data.families || [])}</select>
                    <input id="glmFamilyParameter" class="glm-family-parameter" type="text" inputmode="decimal" placeholder="family.parameter" value="${escapeHtml(String(familyParameterDefault(data.families || [])))}" aria-label="GLM family parameter" />
                  </div>
                </div>
                <div class="glm-control-line">
                  <div class="glm-penalty-row">
                    <label class="glm-control-label" for="glmRegularizationMode">Penalty</label>
                    <select id="glmRegularizationMode" class="glm-penalty-mode" aria-label="GLM penalty">${regularizationModeOptionsHtml(data.regularization)}</select>
                    <div id="glmRegularizationManualControls" class="glm-penalty-manual ${selectedRegularizationMode === "manual" ? "" : "disabled"}">
                      <label class="glm-control-label" for="glmRegularizationMix">Mix</label>
                      <select id="glmRegularizationMix" class="glm-penalty-mix" aria-label="GLM penalty mix">${regularizationMixOptionsHtml(data.regularization)}</select>
                      <label class="glm-control-label" for="glmRegularizationAlpha">Alpha</label>
                      <input id="glmRegularizationAlpha" class="glm-penalty-alpha" type="text" inputmode="decimal" value="${escapeHtml(selectedRegularizationAlpha)}" aria-label="GLM penalty alpha" />
                    </div>
                  </div>
                </div>
              </div>
              <div class="glm-editor-shell">
                <div id="glmFormulaEditor" class="glm-formula-editor"></div>
                <textarea id="glmFormulaText" class="glm-formula-text" spellcheck="false">${escapeHtml(formulaDraft)}</textarea>
              </div>
            </section>
            <div id="glmBuilderResizer" class="glm-builder-resizer" role="separator" aria-orientation="vertical" aria-label="Resize GLM formula and coefficients panels" tabindex="0"></div>
            <section class="glm-coefficient-panel">
              <div class="glm-panel-header glm-coefficient-header">
                <div>
                  <h3 class="glm-panel-title">Coefficients</h3>
                  <div id="glmCoefficientMeta" class="glm-coefficient-meta">${diagnosticsHtml(diagnostics, activeModel)}</div>
                </div>
                <div class="glm-coefficient-actions">
                  <button id="glmCopyCoefficientsBtn" class="tab glm-inline-action-button" type="button">Copy</button>
                  <button id="glmDownloadCoefficientsBtn" class="tab glm-inline-action-button" type="button">Download</button>
                </div>
              </div>
              <div class="glm-table-tools">
                <label>Search: <input id="glmCoefficientSearch" class="search" type="search" /></label>
              </div>
              <div class="glm-coefficient-table-wrap">
                <table class="glm-table" id="glmCoefficientTable"></table>
              </div>
            </section>
          </div>
        </div>
        <div class="glm-tab-panel ${activeTab === "models" ? "" : "hidden"}" data-glm-panel="models">
          <div class="glm-model-navigator">
            <div class="glm-model-actions" role="group" aria-label="GLM model actions">
              <button id="glmRenameModelBtn" class="tab glm-inline-action-button" type="button">Rename</button>
              <button id="glmActivateModelBtn" class="tab glm-inline-action-button" type="button">Activate</button>
              <button id="glmDeleteModelBtn" class="danger-action glm-model-delete-button" type="button">Delete</button>
            </div>
            <div id="glmModelGrid" class="glm-grid glm-model-grid"></div>
            <div id="glmModelFallback" class="glm-model-fallback"></div>
          </div>
        </div>
        <div class="glm-tab-panel ${activeTab === "tabulations" ? "" : "hidden"}" data-glm-panel="tabulations">
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
          <div class="glm-panel-header">
            <h3 class="glm-panel-title">Tabulations</h3>
            <button id="glmBuildTabulationsBtn" class="tab glm-build-button ${isTabulating ? "building" : ""}" type="button" ${isTabulating || !availableModels.length ? "disabled" : ""}>${isTabulating ? "Tabulating..." : "Tabulate"}</button>
          </div>
          <label class="glm-tabulation-label" for="glmTabulationModelSelect">Select models</label>
          <div id="glmTabulationModelSelect" class="glm-tabulation-list glm-tabulation-model-list" role="listbox" aria-label="Select models" aria-multiselectable="true">
            ${tabulationModelListHtml(availableModels, selectedIds)}
          </div>
          <label class="glm-tabulation-label" for="glmTabulationTableSelect">Select table</label>
          <div id="glmTabulationTableSelect" class="glm-tabulation-list glm-tabulation-table-list" role="listbox" aria-label="Select tabulated table">
            ${tables.length ? tables.map((table) => tabulationTableRowHtml(table, String(table.table_id) === selectedTabulationTableId)).join("") : '<div class="glm-empty-state">No tabulations built</div>'}
          </div>
          <div id="glmTabulationDiagnostics" class="glm-tabulation-diagnostics">${diagnostics}</div>
        </section>
        <div id="glmTabulationResizer" class="glm-builder-resizer glm-tabulation-resizer" role="separator" aria-orientation="vertical" aria-label="Resize GLM tabulations and table panels" tabindex="0"></div>
        <section class="glm-tabulation-main">
          <div class="glm-tabulation-controls">
            <div class="glm-tabulation-control-group glm-tabulation-control-left">
              <div class="segmented glm-tabulation-view-toggle" role="group" aria-label="Tabulation view">
                <button type="button" data-glm-tabulation-view="table" class="${tabulationView === "table" ? "active" : ""}">Table</button>
                <button type="button" data-glm-tabulation-view="plot" class="${tabulationView === "plot" ? "active" : ""}" ${features.length > 2 ? "disabled" : ""}>Plot</button>
              </div>
            </div>
            <div class="glm-tabulation-control-group glm-tabulation-control-middle">
              <div class="segmented glm-tabulation-scale-toggle" role="group" aria-label="Tabulation display scale">
                <button type="button" data-glm-tabulation-scale="linear" class="${tabulationScale === "linear" ? "active" : ""}">linear</button>
                <button type="button" data-glm-tabulation-scale="exp" class="${tabulationScale === "exp" ? "active" : ""}">exp</button>
              </div>
              <label class="glm-tabulation-check"><input id="glmTabulationColor" type="checkbox" ${tabulationColor ? "checked" : ""} /> colour</label>
            </div>
            <div class="glm-tabulation-control-group glm-tabulation-control-right">
              <div class="glm-tabulation-crosstab-group">
                <label class="glm-tabulation-crosstab-label" for="glmTabulationCrosstab">crosstab</label>
                <select id="glmTabulationCrosstab" class="glm-tabulation-crosstab" ${crosstabOptions.length > 1 ? "" : "disabled"}>
                  ${tabulationCrosstabOptionsHtml(crosstabOptions)}
                </select>
              </div>
            </div>
          </div>
          <div id="glmTabulationNotice" class="glm-tabulation-inline-notice"></div>
          <div class="glm-tabulation-view-shell ${tabulationView === "table" ? "" : "hidden"}" data-glm-tabulation-view-panel="table">
            <div id="glmTabulationTable" class="glm-grid glm-tabulation-grid"></div>
            <div id="glmTabulationFallback" class="glm-tabulation-fallback"></div>
          </div>
          <div class="glm-tabulation-view-shell ${tabulationView === "plot" ? "" : "hidden"}" data-glm-tabulation-view-panel="plot">
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
    const text = String(value || "").trim();
    if (!text) return "";
    if (text.startsWith("glm:") || text.startsWith("gbm:")) {
      const parts = text.split(":");
      return parts.length >= 2 ? `${parts[0]}:${parts[1]}` : text;
    }
    return `glm:${text}`;
  }

  function tabulationModelRef(model = {}) {
    const explicit = String(model.model_ref || "").trim();
    if (explicit) return normaliseTabulationRef(explicit);
    const kind = String(model.model_kind || "glm").toLowerCase() === "gbm" ? "gbm" : "glm";
    const modelId = String(model.model_id || "").trim();
    return modelId ? `${kind}:${modelId}` : "";
  }

  function tabulationAvailableModels() {
    const allModels = Array.isArray(tabulationConfig?.all_models) ? tabulationConfig.all_models : [];
    if (allModels.length) return allModels;
    return modelRows.map((model) => ({ ...model, model_kind: "glm", model_ref: `glm:${model.model_id}` }));
  }

  function tabulationModelListHtml(models = [], selectedIds = []) {
    const groups = [
      { kind: "glm", label: "GLMs", models: models.filter((model) => String(model.model_kind || "glm").toLowerCase() !== "gbm") },
      { kind: "gbm", label: "GBMs", models: models.filter((model) => String(model.model_kind || "glm").toLowerCase() === "gbm") },
    ];
    const html = groups
      .filter((group) => group.models.length)
      .map((group) => `
        <div class="saved-filter-theme glm-tabulation-model-group" data-glm-tabulation-model-group="${group.kind}" role="presentation">
          <span class="saved-filter-theme-label">${escapeHtml(group.label)}</span>
        </div>
        ${group.models.map((model) => tabulationModelRowHtml(model, selectedIds.includes(tabulationModelRef(model)))).join("")}
      `)
      .join("");
    return html || '<div class="glm-empty-state">No models</div>';
  }

  function activeTabulationTable() {
    return (tabulationConfig?.tables || []).find((table) => String(table.table_id || "") === selectedTabulationTableId) || null;
  }

  function tabulationTableLabel(table = {}) {
    return String(table.label || table.table_id || "");
  }

  function tabulationModelRowHtml(model, selected) {
    const modelId = tabulationModelRef(model);
    return `
      <button type="button" class="glm-tabulation-list-row ${selected ? "selected" : ""}" data-glm-tabulation-model-id="${escapeHtml(modelId)}" role="option" aria-selected="${selected ? "true" : "false"}">
        <span class="glm-tabulation-row-main">${escapeHtml(tabulationModelLabel(model))}</span>
        <span class="glm-tabulation-row-meta">${escapeHtml(tabulationModelStatus(modelId))}</span>
      </button>
    `;
  }

  function tabulationModelLabel(model = {}) {
    const kind = String(model.model_kind || "glm").toUpperCase();
    return `${kind} · ${modelLabel(model)}`;
  }

  function tabulationTableRowHtml(table, selected) {
    const tableId = String(table?.table_id || "");
    return `
      <button type="button" class="glm-tabulation-list-row ${selected ? "selected" : ""}" data-glm-tabulation-table-id="${escapeHtml(tableId)}" role="option" aria-selected="${selected ? "true" : "false"}">
        <span class="glm-tabulation-row-main">${escapeHtml(tabulationTableLabel(table))}</span>
        <span class="glm-tabulation-row-meta">${escapeHtml(tabulationTableMeta(table))}</span>
      </button>
    `;
  }

  function tabulationConfigModel(modelId) {
    const models = Array.isArray(tabulationConfig?.all_models) ? tabulationConfig.all_models : (tabulationConfig?.models || []);
    const ref = normaliseTabulationRef(modelId);
    return models.find((model) => tabulationModelRef(model) === ref || String(model.model_id || "") === String(modelId || "")) || null;
  }

  function tabulationModelStatus(modelId) {
    const model = tabulationConfigModel(modelId);
    if (!model) return "Untabulated";
    if (!model.tabulatable) return "Rebuild required";
    if (!model.tabulated) return "Untabulated";
    const tables = Array.isArray(model.tables) ? model.tables.length : 0;
    return `Tabulated · ${tables.toLocaleString()} ${tables === 1 ? "table" : "tables"}`;
  }

  function tabulationTableMeta(table = {}) {
    const dim = Array.isArray(table.features) ? table.features.length : 0;
    const cells = Number(table.cell_count || 0);
    const base = `dim ${dim} · cells ${cells.toLocaleString()}`;
    return table.skipped ? `${base} · skipped` : base;
  }

  function tabulationCrosstabOptions(features = [], modelIds = tabulationSelectedModelIds()) {
    const options = [{ value: "", label: "No crosstab" }];
    if (modelIds.length > 1) options.push({ value: GLM_TABULATION_MODEL_CROSSTAB, label: "Model" });
    features.forEach((feature) => options.push({ value: feature, label: feature }));
    return options;
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
    if (!models.length) return '<div class="glm-empty-state">No tabulations loaded</div>';
    return models.map((model) => {
      const diagnostics = model.diagnostics || {};
      const warnings = Array.isArray(model.warnings) ? model.warnings : [];
      const rows = [
        ["linear SD error", diagnostics.linear_sd_error],
        ["tabulated rows", diagnostics.tabulated_row_count],
        ["missing", diagnostics.missing_tabulated_prediction_rows],
      ].filter(([, value]) => value !== undefined && value !== null && value !== "");
      return `
        <div class="glm-tabulation-model-diagnostic">
          <strong>${escapeHtml(tabulationModelLabel(model))}</strong>
          ${rows.map(([label, value]) => `<span>${escapeHtml(label)}: ${escapeHtml(formatModelMetric(value))}</span>`).join("")}
          ${model.tabulatable ? "" : '<span class="glm-tabulation-warning">rebuild required</span>'}
          ${warnings.slice(0, 3).map((warning) => `<span class="glm-tabulation-warning">${escapeHtml(warning)}</span>`).join("")}
        </div>
      `;
    }).join("");
  }

  function renderTabulationsPanel() {
    const panel = el("glmTabulationsPanel");
    if (!panel) return;
    disposeTabulationChart();
    tabulationTable = null;
    panel.innerHTML = tabulationsPanelHtml();
    bindTabulationControls();
    bindTabulationResizer();
  }

  function selectTabulationModel(modelId, event = {}) {
    const modelRef = normaliseTabulationRef(modelId);
    const orderedIds = tabulationAvailableModels().map((model) => tabulationModelRef(model)).filter(Boolean);
    if (!orderedIds.includes(modelRef)) return;
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
  }

  function bindTabulationControls() {
    document.querySelectorAll("[data-glm-tabulation-model-id]").forEach((button) => {
      button.addEventListener("click", (event) => {
        const modelId = String(button.dataset.glmTabulationModelId || "");
        if (!modelId) return;
        selectTabulationModel(modelId, event);
        refreshTabulationConfig({ force: true });
      });
      button.addEventListener("keydown", (event) => {
        if (event.key !== "Enter" && event.key !== " ") return;
        event.preventDefault();
        const modelId = String(button.dataset.glmTabulationModelId || "");
        if (!modelId) return;
        selectTabulationModel(modelId, event);
        refreshTabulationConfig({ force: true });
      });
    });
    document.querySelectorAll("[data-glm-tabulation-table-id]").forEach((button) => {
      button.addEventListener("click", () => {
        selectedTabulationTableId = button.dataset.glmTabulationTableId || "base";
        localStorage.setItem("py_lucidum_glm_tabulation_table", selectedTabulationTableId);
        const table = activeTabulationTable();
        const features = Array.isArray(table?.features) ? table.features : [];
        if (features.length > 2) tabulationView = "table";
        renderTabulationsPanel();
        loadTabulationView();
      });
      button.addEventListener("keydown", (event) => {
        if (event.key !== "Enter" && event.key !== " ") return;
        event.preventDefault();
        button.click();
      });
    });
    document.querySelectorAll("[data-glm-tabulation-view]").forEach((button) => {
      button.addEventListener("click", () => {
        if (button.disabled) return;
        tabulationView = button.dataset.glmTabulationView || "table";
        localStorage.setItem("py_lucidum_glm_tabulation_view", tabulationView);
        renderTabulationsPanel();
        loadTabulationView();
      });
    });
    document.querySelectorAll("[data-glm-tabulation-scale]").forEach((button) => {
      button.addEventListener("click", () => {
        tabulationScale = button.dataset.glmTabulationScale || "linear";
        localStorage.setItem("py_lucidum_glm_tabulation_scale", tabulationScale);
        renderTabulationsPanel();
        loadTabulationView();
      });
    });
    el("glmTabulationColor")?.addEventListener("change", (event) => {
      tabulationColor = Boolean(event.target.checked);
      localStorage.setItem("py_lucidum_glm_tabulation_color", String(tabulationColor));
      loadTabulationView();
    });
    el("glmTabulationCrosstab")?.addEventListener("change", (event) => {
      tabulationCrosstab = event.target.value || "";
      loadTabulationView();
    });
    el("glmBuildTabulationsBtn")?.addEventListener("click", buildSelectedTabulations);
  }

  async function refreshTabulationConfig() {
    const model_ids = tabulationSelectedModelIds();
    selectedTabulationModelIds = new Set(model_ids);
    if (!model_ids.length) {
      tabulationConfig = { models: [], all_models: [], tables: [], warnings: [] };
      renderTabulationsPanel();
      return;
    }
    try {
      tabulationConfig = await api("/api/glm/tabulations/config", { method: "POST", body: JSON.stringify({ model_refs: model_ids }) });
      const tables = Array.isArray(tabulationConfig?.tables) ? tabulationConfig.tables : [];
      if (tables.length && !tables.some((table) => String(table.table_id || "") === selectedTabulationTableId)) {
        selectedTabulationTableId = String(tables[0]?.table_id || "base");
      }
      renderTabulationsPanel();
      setTabulationWarnings(tabulationConfig?.warnings || []);
      await loadTabulationView();
    } catch (error) {
      setGlmNotice(error.message);
    }
  }

  function setTabulationWarnings(warnings = []) {
    const message = warnings.slice(0, 4).join(" ");
    setGlmNotice(message);
  }

  async function buildSelectedTabulations() {
    if (isTabulating) return;
    const model_ids = tabulationSelectedModelIds();
    if (!model_ids.length) {
      setGlmNotice("Choose at least one model to tabulate");
      return;
    }
    isTabulating = true;
    liveProgress = { phase: "queued", message: "Starting model tabulations" };
    renderLiveProgress(liveProgress);
    renderTabulationsPanel();
    try {
      const job = await api("/api/glm/tabulations/build", { method: "POST", body: JSON.stringify({ model_refs: model_ids }) });
      pollTabulationJob(job.job_id);
    } catch (error) {
      setTabulationFailure(error.message);
    }
  }

  function pollTabulationJob(jobId) {
    if (tabulationPollTimer) window.clearTimeout(tabulationPollTimer);
    const poll = async () => {
      try {
        const job = await api(`/api/glm/tabulations/jobs/${encodeURIComponent(jobId)}`, { method: "GET" });
        const progress = job.progress || { phase: job.status, message: job.status };
        liveProgress = progress;
        renderLiveProgress(liveProgress);
        if (job.status === "queued" || job.status === "running") {
          tabulationPollTimer = window.setTimeout(poll, job.status === "queued" ? GLM_QUEUED_POLL_MS : GLM_RUNNING_POLL_MS);
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
          setDatasetGlmCount(modelRows.length);
          await refreshTabulationConfig({ force: true });
          renderLiveProgress(liveProgress);
          setAppReadyStatus("Ready");
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
    liveProgress = { phase: "failed", message: String(message || "Model tabulation failed") };
    renderLiveProgress(liveProgress);
    renderTabulationsPanel();
  }

  async function loadTabulationView() {
    if (activeTab !== "tabulations") return;
    const model_ids = tabulationSelectedModelIds();
    const table_id = selectedTabulationTableId || "base";
    if (!model_ids.length || !table_id || !(tabulationConfig?.tables || []).length) {
      renderTabulationEmpty("Build tabulations to view rating tables");
      return;
    }
    const seq = tabulationRenderSeq + 1;
    tabulationRenderSeq = seq;
    const payload = { model_refs: model_ids, table_id, scale: tabulationScale, crosstab: tabulationCrosstab };
    try {
      if (tabulationView === "plot") {
        const data = await api("/api/glm/tabulations/plot", { method: "POST", body: JSON.stringify(payload) });
        if (seq !== tabulationRenderSeq) return;
        renderTabulationPlot(data);
      } else {
        const data = await api("/api/glm/tabulations/table", { method: "POST", body: JSON.stringify(payload) });
        if (seq !== tabulationRenderSeq) return;
        renderTabulationTable(data);
      }
    } catch (error) {
      renderTabulationEmpty(error.message);
    }
  }

  function renderTabulationEmpty(message) {
    const fallback = el("glmTabulationFallback");
    const grid = el("glmTabulationTable");
    const plot = el("glmTabulationPlot");
    if (grid) grid.innerHTML = "";
    if (fallback) fallback.innerHTML = `<div class="glm-empty-state">${escapeHtml(message)}</div>`;
    if (plot) plot.innerHTML = `<div class="glm-empty-state">${escapeHtml(message)}</div>`;
  }

  async function renderTabulationTable(data = {}) {
    const grid = el("glmTabulationTable");
    const fallback = el("glmTabulationFallback");
    if (!grid || !fallback) return;
    grid.innerHTML = "";
    fallback.innerHTML = "";
    setInlineTabulationNotice(data.notices || []);
    const rows = Array.isArray(data.rows) ? data.rows : [];
    const columns = Array.isArray(data.columns) ? data.columns : [];
    if (!rows.length || !columns.length) {
      renderTabulationEmpty("No rows for this tabulation");
      return;
    }
    try {
      const Tabulator = await loadTabulator();
      tabulationTable = new Tabulator("#glmTabulationTable", {
        data: rows,
        height: "100%",
        layout: "fitDataStretch",
        placeholder: "No rows",
        columns: columns.map((column) => tabulationColumnDefinition(column, data)),
      });
    } catch (_) {
      renderTabulationFallbackTable(columns, rows, data);
    }
  }

  function tabulationColumnDefinition(column, data = {}) {
    const field = String(column.field || "");
    const tabulationValue = Boolean(column.tabulation_value);
    const numeric = tabulationValue || tabulationSelectedModelIds().includes(field);
    const statusField = String(column.status_field || `__status__${field}`);
    return {
      ...column,
      formatter: (cell) => {
        if (!numeric) return escapeHtml(cell.getValue() ?? "");
        const row = cell.getRow().getData();
        const status = row[statusField] || "ok";
        const value = cell.getValue();
        const element = cell.getElement();
        element.classList.toggle("glm-tabulation-na-cell", status !== "ok" || value === null || value === undefined);
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
    return Number.isFinite(number) ? number.toFixed(4) : formatModelMetric(value);
  }

  function tabulationCellColor(value, min, max) {
    const number = Number(value);
    const lo = Number(min);
    const hi = Number(max);
    if (!Number.isFinite(number) || !Number.isFinite(lo) || !Number.isFinite(hi) || hi <= lo) return "";
    const ratio = Math.max(0, Math.min(1, (number - lo) / (hi - lo)));
    const hue = 130 - ratio * 130;
    return `hsl(${hue} 78% 88%)`;
  }

  function renderTabulationFallbackTable(columns, rows, data = {}) {
    const fallback = el("glmTabulationFallback");
    if (!fallback) return;
    fallback.innerHTML = `
      <table class="glm-table glm-tabulation-fallback-table">
        <thead><tr>${columns.map((column) => `<th>${escapeHtml(column.title || column.field)}</th>`).join("")}</tr></thead>
        <tbody>${rows.map((row) => `<tr>${columns.map((column) => {
          const value = row[column.field];
          const tabulationValue = Boolean(column.tabulation_value);
          const numeric = tabulationValue || tabulationSelectedModelIds().includes(column.field);
          const status = row[column.status_field || `__status__${column.field}`] || "ok";
          const style = numeric && tabulationColor && status === "ok" ? ` style="background:${tabulationCellColor(value, data.min, data.max)}"` : "";
          return `<td class="${numeric ? "numeric" : ""}${status !== "ok" ? " glm-tabulation-na-cell" : ""}"${style}>${value === null || value === undefined ? "NA" : escapeHtml(tabulationValue ? formatTabulationValue(value) : (numeric ? formatModelMetric(value) : value))}</td>`;
        }).join("")}</tr>`).join("")}</tbody>
      </table>
    `;
  }

  function renderTabulationPlot(data = {}) {
    const plot = el("glmTabulationPlot");
    if (!plot) return;
    setInlineTabulationNotice(data.notices || []);
    disposeTabulationChart();
    if (!data.plottable || !Array.isArray(data.series) || !data.series.length) {
      plot.innerHTML = `<div class="glm-empty-state">${escapeHtml((data.notices || [])[0] || "Plot is unavailable for this table")}</div>`;
      return;
    }
    if (!window.echarts) {
      plot.innerHTML = `<div class="glm-empty-state">ECharts is not available</div>`;
      return;
    }
    plot.innerHTML = "";
    tabulationChart = window.echarts.init(plot);
    tabulationChart.setOption({
      animation: false,
      tooltip: { trigger: "axis" },
      legend: { type: "scroll", top: 4, right: 8 },
      grid: { left: 54, right: 24, top: 48, bottom: 52 },
      xAxis: { type: "category", data: data.x_axis || [], axisLabel: { hideOverlap: true } },
      yAxis: { type: "value", name: data.scale === "exp" ? "exp(tabulated)" : "tabulated" },
      series: data.series,
    });
  }

  function disposeTabulationChart() {
    if (!tabulationChart) return;
    try {
      tabulationChart.dispose();
    } catch (_) {
    }
    tabulationChart = null;
  }

  function setInlineTabulationNotice(notices = []) {
    const notice = el("glmTabulationNotice");
    if (!notice) return;
    const text = notices.filter(Boolean).join(" ");
    notice.textContent = text;
    notice.classList.toggle("hidden", !text);
  }

  function familyOptionsHtml(families = []) {
    const rows = families.length ? families : [
      { value: "normal", label: "Normal" },
      { value: "poisson", label: "Poisson" },
      { value: "gamma", label: "Gamma" },
      { value: "tweedie", label: "Tweedie" },
      { value: "binomial", label: "Binomial" },
      { value: "inverse.gaussian", label: "Inverse Gaussian" },
      { value: "negative.binomial", label: "Negative Binomial" },
    ];
    if (!rows.some((row) => row.value === selectedFamily)) selectedFamily = rows[0]?.value || "normal";
    return rows.map((row) => `<option value="${escapeHtml(row.value)}" ${row.value === selectedFamily ? "selected" : ""}>${escapeHtml(row.label || row.value)}</option>`).join("");
  }

  function familyParameterDefault(families = []) {
    const parameter = familyParameterConfig(selectedFamily, families);
    if (!parameter) return "";
    const key = `py_lucidum_glm_family_parameter_${selectedFamily}`;
    return localStorage.getItem(key) || parameter.default || "";
  }

  function familyParameterConfig(familyValue, families = config?.families || []) {
    const value = String(familyValue || "").trim();
    const family = (families || []).find((row) => row.value === value);
    if (family?.parameter) return family.parameter;
    if (value === "tweedie") return { label: "var.power", default: "1.5", min: 1, max: 2 };
    if (value === "negative.binomial") return { label: "theta", default: "1", min: 0.000001 };
    return null;
  }

  function syncFamilyParameterControl() {
    const select = el("glmFamilySelect");
    const input = el("glmFamilyParameter");
    if (!select || !input) return;
    const parameter = familyParameterConfig(select.value);
    input.disabled = !parameter;
    input.placeholder = "family.parameter";
    input.value = parameter ? (localStorage.getItem(`py_lucidum_glm_family_parameter_${select.value}`) || parameter.default || "") : "";
  }

  function regularizationModeOptionsHtml(regularization = {}) {
    const rows = Array.isArray(regularization?.modes) && regularization.modes.length
      ? regularization.modes
      : [
        { value: "none", label: "None" },
        { value: "auto", label: "Auto" },
        { value: "manual", label: "Manual" },
      ];
    if (!rows.some((row) => row.value === selectedRegularizationMode)) selectedRegularizationMode = "none";
    return rows.map((row) => `<option value="${escapeHtml(row.value)}" ${row.value === selectedRegularizationMode ? "selected" : ""}>${escapeHtml(row.label || row.value)}</option>`).join("");
  }

  function regularizationMixOptionsHtml(regularization = {}) {
    const rows = Array.isArray(regularization?.mixes) && regularization.mixes.length
      ? regularization.mixes
      : [
        { value: "0", label: "Ridge" },
        { value: "0.5", label: "Elastic net" },
        { value: "1", label: "Lasso" },
      ];
    if (!rows.some((row) => String(row.value) === String(selectedRegularizationMix))) selectedRegularizationMix = "0.5";
    return rows.map((row) => `<option value="${escapeHtml(row.value)}" ${String(row.value) === String(selectedRegularizationMix) ? "selected" : ""}>${escapeHtml(row.label || row.value)}</option>`).join("");
  }

  function syncRegularizationControls() {
    const mode = el("glmRegularizationMode");
    const manual = el("glmRegularizationManualControls");
    const mix = el("glmRegularizationMix");
    const alpha = el("glmRegularizationAlpha");
    const isManual = (mode?.value || selectedRegularizationMode) === "manual";
    if (manual) manual.classList.toggle("disabled", !isManual);
    if (mix) mix.disabled = !isManual;
    if (alpha) alpha.disabled = !isManual;
  }

  function bindTabs(mount) {
    mount.querySelectorAll("[data-glm-tab]").forEach((button) => {
      button.addEventListener("click", () => {
        activeTab = button.dataset.glmTab;
        mount.querySelectorAll("[data-glm-tab]").forEach((item) => item.classList.toggle("active", item === button));
        mount.querySelectorAll("[data-glm-panel]").forEach((panel) => panel.classList.toggle("hidden", panel.dataset.glmPanel !== activeTab));
        if (activeTab === "models") refreshModelListIfNeeded();
        if (activeTab === "tabulations") refreshTabulationConfig({ force: true });
      });
    });
  }

  function bindBuilderControls() {
    syncFamilyParameterControl();
    syncRegularizationControls();
    el("glmFamilySelect")?.addEventListener("change", (event) => {
      selectedFamily = event.target.value;
      localStorage.setItem("py_lucidum_glm_family", selectedFamily);
      syncFamilyParameterControl();
    });
    el("glmFamilyParameter")?.addEventListener("change", (event) => {
      if (familyParameterConfig(selectedFamily)) localStorage.setItem(`py_lucidum_glm_family_parameter_${selectedFamily}`, event.target.value.trim());
    });
    el("glmRegularizationMode")?.addEventListener("change", (event) => {
      selectedRegularizationMode = event.target.value || "none";
      localStorage.setItem("py_lucidum_glm_regularization_mode", selectedRegularizationMode);
      syncRegularizationControls();
    });
    el("glmRegularizationMix")?.addEventListener("change", (event) => {
      selectedRegularizationMix = event.target.value || "0.5";
      localStorage.setItem("py_lucidum_glm_regularization_mix", selectedRegularizationMix);
    });
    el("glmRegularizationAlpha")?.addEventListener("change", (event) => {
      selectedRegularizationAlpha = event.target.value.trim() || "0.01";
      localStorage.setItem("py_lucidum_glm_regularization_alpha", selectedRegularizationAlpha);
    });
    document.querySelectorAll("[data-glm-scope]").forEach((button) => {
      button.addEventListener("click", () => {
        if (button.disabled) return;
        selectedTrainingScope = button.dataset.glmScope || "all";
        localStorage.setItem("py_lucidum_glm_training_scope", selectedTrainingScope);
        document.querySelectorAll("[data-glm-scope]").forEach((item) => item.classList.toggle("active", item === button));
      });
    });
    el("glmClearFormulaBtn")?.addEventListener("click", () => setFormulaText(""));
    el("glmFontSmallerBtn")?.addEventListener("click", () => adjustFontSize(-1));
    el("glmFontLargerBtn")?.addEventListener("click", () => adjustFontSize(1));
    el("glmBuildBtn")?.addEventListener("click", buildModel);
    el("glmCopyCoefficientsBtn")?.addEventListener("click", copyCoefficients);
    el("glmDownloadCoefficientsBtn")?.addEventListener("click", downloadCoefficients);
    el("glmCoefficientSearch")?.addEventListener("input", () => renderCoefficientTable(coefficientRows));
  }

  function savedBuilderSplitWidthStyle() {
    const width = Number(localStorage.getItem(GLM_BUILDER_SPLIT_STORAGE_KEY));
    return Number.isFinite(width) && width > 0 ? `--glm-formula-panel-width: ${Math.round(width)}px;` : "";
  }

  function savedTabulationSplitWidthStyle() {
    const width = Number(localStorage.getItem(GLM_TABULATION_SPLIT_STORAGE_KEY));
    return Number.isFinite(width) && width > 0 ? `--glm-tabulation-sidebar-width: ${Math.round(width)}px;` : "";
  }

  function bindBuilderResizer() {
    const layout = document.querySelector(".glm-builder-layout");
    const resizer = el("glmBuilderResizer");
    if (!layout || !resizer) return;

    const resizeTo = (width, persist = true) => {
      const layoutRect = layout.getBoundingClientRect();
      const resizerWidth = resizer.getBoundingClientRect().width || 0;
      const minLeft = 320;
      const minRight = 360;
      const maxLeft = Math.max(minLeft, layoutRect.width - resizerWidth - minRight);
      const clamped = Math.max(minLeft, Math.min(maxLeft, width));
      layout.style.setProperty("--glm-formula-panel-width", `${Math.round(clamped)}px`);
      if (persist) localStorage.setItem(GLM_BUILDER_SPLIT_STORAGE_KEY, String(Math.round(clamped)));
      if (aceEditor) aceEditor.resize();
    };

    const resizeFromClientX = (clientX) => {
      const layoutRect = layout.getBoundingClientRect();
      resizeTo(clientX - layoutRect.left);
    };

    resizer.addEventListener("pointerdown", (event) => {
      event.preventDefault();
      resizer.classList.add("dragging");
      document.body.classList.add("glm-builder-resizing");
      resizer.setPointerCapture?.(event.pointerId);
      const onMove = (moveEvent) => resizeFromClientX(moveEvent.clientX);
      const onUp = () => {
        resizer.classList.remove("dragging");
        document.body.classList.remove("glm-builder-resizing");
        window.removeEventListener("pointermove", onMove);
        window.removeEventListener("pointerup", onUp);
      };
      window.addEventListener("pointermove", onMove);
      window.addEventListener("pointerup", onUp, { once: true });
    });

    resizer.addEventListener("keydown", (event) => {
      if (!["ArrowLeft", "ArrowRight"].includes(event.key)) return;
      event.preventDefault();
      const formulaPanel = layout.querySelector(".glm-formula-panel");
      const current = formulaPanel?.getBoundingClientRect().width || 0;
      resizeTo(current + (event.key === "ArrowRight" ? 24 : -24));
    });
  }

  function bindTabulationResizer() {
    const layout = document.querySelector(".glm-tabulation-layout");
    const resizer = el("glmTabulationResizer");
    if (!layout || !resizer) return;

    const resizeTo = (width, persist = true) => {
      const layoutRect = layout.getBoundingClientRect();
      const resizerWidth = resizer.getBoundingClientRect().width || 0;
      const minLeft = 240;
      const minRight = 420;
      const maxLeft = Math.max(minLeft, layoutRect.width - resizerWidth - minRight);
      const clamped = Math.max(minLeft, Math.min(maxLeft, width));
      layout.style.setProperty("--glm-tabulation-sidebar-width", `${Math.round(clamped)}px`);
      if (persist) localStorage.setItem(GLM_TABULATION_SPLIT_STORAGE_KEY, String(Math.round(clamped)));
      tabulationChart?.resize?.();
      tabulationTable?.redraw?.(true);
    };

    const resizeFromClientX = (clientX) => {
      const layoutRect = layout.getBoundingClientRect();
      resizeTo(clientX - layoutRect.left);
    };

    resizer.addEventListener("pointerdown", (event) => {
      event.preventDefault();
      resizer.classList.add("dragging");
      document.body.classList.add("glm-builder-resizing");
      resizer.setPointerCapture?.(event.pointerId);
      const onMove = (moveEvent) => resizeFromClientX(moveEvent.clientX);
      const onUp = () => {
        resizer.classList.remove("dragging");
        document.body.classList.remove("glm-builder-resizing");
        window.removeEventListener("pointermove", onMove);
        window.removeEventListener("pointerup", onUp);
      };
      window.addEventListener("pointermove", onMove);
      window.addEventListener("pointerup", onUp, { once: true });
    });

    resizer.addEventListener("keydown", (event) => {
      if (!["ArrowLeft", "ArrowRight"].includes(event.key)) return;
      event.preventDefault();
      const sidebar = layout.querySelector(".glm-tabulation-sidebar");
      const current = sidebar?.getBoundingClientRect().width || 0;
      resizeTo(current + (event.key === "ArrowRight" ? 24 : -24));
    });
  }

  function bindModelActions() {
    el("glmActivateModelBtn")?.addEventListener("click", activateSelectedModel);
    el("glmRenameModelBtn")?.addEventListener("click", renameSelectedModel);
    el("glmDeleteModelBtn")?.addEventListener("click", deleteSelectedModels);
    updateModelActionButtons();
  }

  function syncAceGutterWidth() {
    if (!aceEditor?.session || !aceEditor?.container) return;
    const lineCount = Math.max(1, aceEditor.session.getLength());
    const digits = String(lineCount).length;
    const width = Math.max(24, Math.ceil((digits * editorFontSize * 0.62) + 10));
    const value = `${width}px`;
    const container = aceEditor.container;
    container.style.setProperty("--glm-ace-gutter-width", value);
    const gutter = container.querySelector(".ace_gutter");
    const scroller = container.querySelector(".ace_scroller");
    if (gutter) gutter.style.width = value;
    if (scroller) scroller.style.left = value;
    aceEditor.resize();
  }

  async function initEditor() {
    const mount = el("glmFormulaEditor");
    const fallback = el("glmFormulaText");
    if (!mount || !fallback) return;
    fallback.value = formulaDraft;
    fallback.style.fontSize = `${editorFontSize}px`;
    try {
      const ace = await loadAce();
      if (!document.body.contains(mount) || editorInitialisedFor === mount) return;
      aceEditor = ace.edit(mount);
      editorInitialisedFor = mount;
      aceEditor.setTheme("ace/theme/textmate");
      aceEditor.session.setMode("ace/mode/r");
      aceEditor.session.setUseWorker(false);
      aceEditor.setOptions({
        fontSize: `${editorFontSize}px`,
        tabSize: 2,
        useSoftTabs: true,
        showPrintMargin: false,
        wrap: true,
      });
      aceEditor.setValue(formulaDraft, -1);
      syncAceGutterWidth();
      aceEditor.session.on("change", () => {
        formulaDraft = aceEditor.getValue();
        localStorage.setItem("py_lucidum_glm_formula", formulaDraft);
        syncAceGutterWidth();
      });
      fallback.classList.add("hidden");
      mount.classList.remove("fallback");
    } catch (_) {
      mount.classList.add("fallback");
      fallback.classList.remove("hidden");
      fallback.addEventListener("input", () => {
        formulaDraft = fallback.value;
        localStorage.setItem("py_lucidum_glm_formula", formulaDraft);
      });
    }
  }

  function disposeEditor() {
    if (!aceEditor) return;
    formulaDraft = aceEditor.getValue();
    try {
      aceEditor.destroy();
    } catch (_) {
    }
    aceEditor = null;
    editorInitialisedFor = null;
  }

  function getFormulaText() {
    if (aceEditor) formulaDraft = aceEditor.getValue();
    else if (el("glmFormulaText")) formulaDraft = el("glmFormulaText").value;
    localStorage.setItem("py_lucidum_glm_formula", formulaDraft);
    return formulaDraft;
  }

  function setFormulaText(value) {
    formulaDraft = String(value || "");
    localStorage.setItem("py_lucidum_glm_formula", formulaDraft);
    if (aceEditor) {
      aceEditor.setValue(formulaDraft, -1);
      syncAceGutterWidth();
    }
    if (el("glmFormulaText")) el("glmFormulaText").value = formulaDraft;
  }

  function adjustFontSize(delta) {
    editorFontSize = Math.max(10, Math.min(24, editorFontSize + delta));
    localStorage.setItem("py_lucidum_glm_font_size", String(editorFontSize));
    if (aceEditor) {
      aceEditor.setFontSize(`${editorFontSize}px`);
      syncAceGutterWidth();
    }
    if (el("glmFormulaText")) el("glmFormulaText").style.fontSize = `${editorFontSize}px`;
  }

  function buildPayload() {
    const actual = el("actualNumerator")?.value || "";
    const denominator = el("denominator")?.value || "__none__";
    const family = el("glmFamilySelect")?.value || selectedFamily || "normal";
    const familyParameter = familyParameterConfig(family) ? (el("glmFamilyParameter")?.value.trim() || "") : "";
    return {
      formula: getFormulaText(),
      family,
      family_parameter: familyParameter,
      regularization: buildRegularizationPayload(),
      training_scope: selectedTrainingScope,
      response_column: actual,
      denominator_column: denominator === "__none__" ? "" : denominator,
      label: `GLM ${glmAutoModelTimeLabel()}`,
    };
  }

  function buildRegularizationPayload() {
    const mode = el("glmRegularizationMode")?.value || selectedRegularizationMode || "none";
    const payload = { mode };
    if (mode === "manual") {
      payload.l1_ratio = Number(el("glmRegularizationMix")?.value || selectedRegularizationMix || 0.5);
      payload.alpha = el("glmRegularizationAlpha")?.value.trim() || selectedRegularizationAlpha || "";
    }
    return payload;
  }

  async function buildModel() {
    if (isBuilding) return;
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
    isBuilding = true;
    liveProgress = { phase: "queued", message: "Starting GLM build" };
    renderLiveProgress(liveProgress);
    setGlmNotice("");
    try {
      const job = await api("/api/glm/build", { method: "POST", body: JSON.stringify(payload) });
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
    isBuilding = false;
    liveProgress = { phase: "failed", message: String(message || "GLM build failed") };
    renderLiveProgress(liveProgress);
    setGlmNotice("");
  }

  function validateFamilyParameter(family, rawValue) {
    const parameter = familyParameterConfig(family);
    if (!parameter) return "";
    const text = String(rawValue || "").trim();
    const value = Number(text);
    const min = Number(parameter.min);
    const max = Number(parameter.max);
    if (!text || !Number.isFinite(value) || (Number.isFinite(min) && value < min) || (Number.isFinite(max) && value > max)) {
      const label = parameter.label || "family parameter";
      if (Number.isFinite(min) && Number.isFinite(max)) return `Choose ${label} from ${min} to ${max}`;
      if (Number.isFinite(min)) return `Choose ${label} of at least ${min}`;
      return `Choose a numeric ${label}`;
    }
    return "";
  }

  function validateRegularizationParameter(regularization = {}) {
    if (regularization.mode !== "manual") return "";
    const alpha = Number(regularization.alpha);
    const l1Ratio = Number(regularization.l1_ratio);
    if (!Number.isFinite(alpha) || alpha <= 0) {
      return "Choose a positive GLM regularization alpha";
    }
    if (!Number.isFinite(l1Ratio) || l1Ratio < 0 || l1Ratio > 1) {
      return "Choose a GLM regularization mix from 0 to 1";
    }
    return "";
  }

  function pollBuildJob(jobId) {
    if (pollTimer) window.clearTimeout(pollTimer);
    const poll = async () => {
      try {
        const job = await api(`/api/glm/jobs/${encodeURIComponent(jobId)}`, { method: "GET" });
        const progress = job.progress || { phase: job.status, message: job.status };
        liveProgress = progress;
        renderLiveProgress(liveProgress);
        if (job.status === "queued" || job.status === "running") {
          pollTimer = window.setTimeout(poll, job.status === "queued" ? GLM_QUEUED_POLL_MS : GLM_RUNNING_POLL_MS);
          return;
        }
        pollTimer = null;
        isBuilding = false;
        if (job.status === "succeeded") {
          const latest = await api("/api/glm/config", { method: "GET", clientTiming: true });
          liveProgress = null;
          await applyModelMutationResult({ model: job.result, config: latest });
          renderLiveProgress(liveProgress);
          setAppReadyStatus("Ready");
        } else {
          setBuildFailure(job.error || progress.message || "GLM build failed");
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
    const rows = Number(progress.training_rows || 0);
    const cells = Number(progress.cells || 0);
    const detail = rows ? `${rows.toLocaleString()} training rows` : (cells ? `${cells.toLocaleString()} cells` : "");
    return `<span class="glm-build-status-main">${escapeHtml(main)}</span>${detail ? `<span class="glm-build-status-detail">${escapeHtml(detail)}</span>` : ""}`;
  }

  function renderLiveProgress(progress) {
    const status = el("glmBuildStatus");
    if (!status) return;
    status.innerHTML = buildStatusHtml(progress);
    status.dataset.phase = String(progress?.phase || "");
    status.classList.toggle("hidden", !progress);
    const button = el("glmBuildBtn");
    if (button) {
      button.disabled = isBuilding;
      button.classList.toggle("building", isBuilding);
      button.textContent = isBuilding ? "Building..." : "Build GLM";
    }
    const tabulationButton = el("glmBuildTabulationsBtn");
    if (tabulationButton) {
      tabulationButton.disabled = isTabulating || !tabulationAvailableModels().length;
      tabulationButton.classList.toggle("building", isTabulating);
      tabulationButton.textContent = isTabulating ? "Tabulating..." : "Tabulate";
    }
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

  function diagnosticsHtml(diagnostics = {}, model = {}) {
    const regularization = model?.regularization || {};
    const penalty = regularizationLabel(regularization);
    const nonzero = nonzeroCoefficientLabel(regularization, diagnostics);
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
    if (!primary.length && !secondary.length) return "No active model";
    return [
      primary.length ? `<span class="glm-coefficient-meta-row">${primary.map(itemHtml).join("")}</span>` : "",
      secondary.length ? `<span class="glm-coefficient-meta-row glm-coefficient-meta-row-secondary">${secondary.map(itemHtml).join("")}</span>` : "",
    ].filter(Boolean).join("");
  }

  function syncBuilderFromModelDetail(detail = {}) {
    const manifest = detail?.manifest || {};
    const rawFormula = detail?.formula ?? manifest?.formula?.raw;
    if (rawFormula !== undefined && rawFormula !== null) setFormulaText(String(rawFormula));

    const family = String(manifest.family || "").trim();
    if (family) {
      selectedFamily = family;
      localStorage.setItem("py_lucidum_glm_family", selectedFamily);
      const select = el("glmFamilySelect");
      if (select) select.value = selectedFamily;
    }

    const familyParameter = manifest.family_parameter;
    if (familyParameterConfig(family) && familyParameter !== undefined && familyParameter !== null && String(familyParameter).trim() !== "") {
      localStorage.setItem(`py_lucidum_glm_family_parameter_${family}`, String(familyParameter));
    }
    syncFamilyParameterControl();
    const input = el("glmFamilyParameter");
    if (input && !input.disabled && familyParameter !== undefined && familyParameter !== null) {
      input.value = String(familyParameter);
    }

    const trainingScope = String(manifest.training_scope || "").trim().toLowerCase();
    if (trainingScope === "all" || trainingScope === "training") {
      selectedTrainingScope = trainingScope;
      localStorage.setItem("py_lucidum_glm_training_scope", selectedTrainingScope);
      document.querySelectorAll("[data-glm-scope]").forEach((button) => {
        button.classList.toggle("active", button.dataset.glmScope === selectedTrainingScope);
      });
    }

    const regularization = manifest.regularization || {};
    const regularizationMode = String(regularization.mode || "none").trim().toLowerCase();
    if (["none", "auto", "manual"].includes(regularizationMode)) {
      selectedRegularizationMode = regularizationMode;
      localStorage.setItem("py_lucidum_glm_regularization_mode", selectedRegularizationMode);
      const mode = el("glmRegularizationMode");
      if (mode) mode.value = selectedRegularizationMode;
    }
    if (regularization.l1_ratio !== undefined && regularization.l1_ratio !== null && !Array.isArray(regularization.l1_ratio)) {
      selectedRegularizationMix = String(regularization.l1_ratio);
      localStorage.setItem("py_lucidum_glm_regularization_mix", selectedRegularizationMix);
      const mix = el("glmRegularizationMix");
      if (mix) mix.value = selectedRegularizationMix;
    }
    if (regularization.alpha !== undefined && regularization.alpha !== null && String(regularization.alpha).trim() !== "") {
      selectedRegularizationAlpha = String(regularization.alpha);
      localStorage.setItem("py_lucidum_glm_regularization_alpha", selectedRegularizationAlpha);
      const alpha = el("glmRegularizationAlpha");
      if (alpha) alpha.value = selectedRegularizationAlpha;
    }
    syncRegularizationControls();
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
    const penalized = activeModelIsPenalized();
    const query = String(el("glmCoefficientSearch")?.value || "").trim().toLowerCase();
    const filtered = query
      ? rows.filter((row) => Object.values(row).some((value) => String(value ?? "").toLowerCase().includes(query)))
      : rows;
    if (!filtered.length) {
      table.innerHTML = `<tbody><tr><td class="glm-empty-cell">No coefficients to show</td></tr></tbody>`;
      return;
    }
    table.innerHTML = `
      <thead>
        <tr>
          <th>term</th>
          <th class="numeric">estimate</th>
          <th class="numeric">std.error</th>
          <th class="numeric">p.value</th>
        </tr>
      </thead>
      <tbody>
        ${filtered.map((row) => `
          <tr class="${penalized ? "" : glmCoefficientPValueClass(row.p_value)}">
            <td>${escapeHtml(row.term)}</td>
            <td class="numeric">${escapeHtml(formatModelMetric(row.estimate))}</td>
            <td class="numeric">${penalized ? "" : escapeHtml(formatModelMetric(row.std_error))}</td>
            <td class="numeric">${penalized ? "" : escapeHtml(formatPValue(row.p_value))}</td>
          </tr>
        `).join("")}
      </tbody>
    `;
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
    try {
      activeDetail = await api(`/api/glm/models/${encodeURIComponent(modelId)}`, { method: "GET" });
      const detailModelId = String(activeDetail?.manifest?.model_id || modelId || "");
      if (detailModelId !== String(config?.active_model_id || modelId || "")) return;
      syncBuilderFromModelDetail(activeDetail);
      const diagnostics = activeDetail?.diagnostics || activeDetail?.manifest?.diagnostics || {};
      const meta = el("glmCoefficientMeta");
      if (meta) meta.innerHTML = diagnosticsHtml(diagnostics, activeDetail?.manifest || {});
      renderCoefficientTable(Array.isArray(activeDetail?.coefficients) ? activeDetail.coefficients : []);
    } catch (error) {
      activeDetail = null;
      setGlmNotice(error.message);
    }
  }

  async function renderModelTable(models = modelRows, activeModelId = config?.active_model_id) {
    const grid = el("glmModelGrid");
    const fallback = el("glmModelFallback");
    const preservedIds = Array.from(selectedModelIds);
    const renderSeq = modelTableRenderSeq + 1;
    modelTableRenderSeq = renderSeq;
    modelTable = null;
    if (!grid || !fallback) {
      updateModelActionButtons();
      return;
    }
    grid.innerHTML = "";
    fallback.innerHTML = "";
    const rows = modelTableRows(models, activeModelId);
    try {
      const Tabulator = await loadTabulator();
      if (renderSeq !== modelTableRenderSeq || !grid.isConnected) return;
      modelTable = new Tabulator("#glmModelGrid", {
        data: rows,
        height: "100%",
        layout: "fitDataStretch",
        placeholder: "No GLMs built yet",
        initialSort: [{ column: "created_sort", dir: "desc" }],
        selectableRows: true,
        selectableRowsRangeMode: "click",
        columns: [
          { title: "", field: "active", formatter: activeModelDotFormatter, hozAlign: "center", headerHozAlign: "center", width: 28, minWidth: 28, headerSort: false, resizable: false },
          { title: "Model", field: "model_label", sorter: "string", formatter: modelNameFormatter, widthGrow: 3, headerSort: true },
          { title: "Created", field: "created_sort", sorter: "number", formatter: (cell) => escapeHtml(cell.getRow().getData().created_display), width: 105, headerSort: true },
          { title: "Response", field: "response_column", sorter: "string", formatter: (cell) => escapeHtml(cell.getValue() || ""), widthGrow: 1.4, headerSort: true },
          { title: "Weight", field: "weight_display", sorter: "string", formatter: (cell) => escapeHtml(cell.getValue() || ""), widthGrow: 1.1, headerSort: true },
          { title: "Family", field: "family", sorter: "string", formatter: (cell) => escapeHtml(cell.getValue() || ""), widthGrow: 1.1, headerSort: true },
          { title: "Deviance", field: "deviance", sorter: "number", formatter: (cell) => escapeHtml(formatModelMetric(cell.getValue())), hozAlign: "right", headerHozAlign: "right", width: 96, headerSort: true },
          { title: "AIC", field: "aic", sorter: "number", formatter: (cell) => escapeHtml(formatModelMetric(cell.getValue())), hozAlign: "right", headerHozAlign: "right", width: 96, headerSort: true },
          { title: "BIC", field: "bic", sorter: "number", formatter: (cell) => escapeHtml(formatModelMetric(cell.getValue())), hozAlign: "right", headerHozAlign: "right", width: 96, headerSort: true },
          { title: "Rows", field: "training_rows", sorter: "number", formatter: (cell) => Number(cell.getValue() || 0).toLocaleString(), hozAlign: "right", headerHozAlign: "right", width: 86, headerSort: true },
        ],
      });
      modelTable.on("rowSelectionChanged", syncSelectedModelsFromTable);
      restoreModelSelection(preservedIds);
      updateModelActionButtons();
    } catch (_) {
      if (renderSeq !== modelTableRenderSeq) return;
      renderModelFallback(models, activeModelId);
      restoreModelSelection(preservedIds);
      updateModelActionButtons();
    }
  }

  function renderModelFallback(models = modelRows, activeModelId = config?.active_model_id) {
    const target = el("glmModelFallback");
    if (!target) return;
    if (!models.length) {
      target.innerHTML = `<div class="glm-empty-state">No GLMs built yet</div>`;
      return;
    }
    target.innerHTML = `
      <table class="glm-table glm-model-table">
        <thead>
          <tr>
            <th class="glm-model-active-heading" aria-label="Active model"></th>
            <th>model</th>
            <th>created</th>
            <th>response</th>
            <th>weight</th>
            <th>family</th>
            <th>deviance</th>
            <th>AIC</th>
            <th>BIC</th>
            <th>rows</th>
          </tr>
        </thead>
        <tbody>
          ${models.map((model) => modelTableRowHtml(model, activeModelId)).join("")}
        </tbody>
      </table>
    `;
    const rows = Array.from(target.querySelectorAll("[data-glm-model-row]"));
    let anchorRow = null;
    const setSelected = (row, selected) => {
      row.classList.toggle("selected", selected);
      row.setAttribute("aria-selected", String(selected));
    };
    rows.forEach((row) => {
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
        syncSelectedModelsFromTable();
      });
    });
    syncSelectedModelsFromTable();
  }

  function modelTableRowHtml(model, activeModelId) {
    const active = model.model_id === activeModelId;
    const selected = selectedModelIds.has(model.model_id);
    const diagnostics = model.diagnostics || model.metrics || {};
    return `
      <tr data-glm-model-row="${escapeHtml(model.model_id)}" class="${active ? "active" : ""}${selected ? " selected" : ""}" aria-selected="${selected ? "true" : "false"}">
        <td class="glm-model-active-cell">
          ${active ? '<span class="glm-model-active-dot" title="Active model" aria-label="Active model"></span>' : ""}
        </td>
        <td class="glm-model-name-cell"><span class="glm-model-name-main">${escapeHtml(model.label || model.model_id)}</span></td>
        <td>${escapeHtml(formatModelCreated(model.created_at))}</td>
        <td>${escapeHtml(model.response_column || "")}</td>
        <td>${escapeHtml(modelWeightLabel(model.denominator_column || model.offset_column))}</td>
        <td>${escapeHtml(model.family || "")}</td>
        <td class="numeric">${escapeHtml(formatModelMetric(diagnostics.deviance))}</td>
        <td class="numeric">${escapeHtml(formatModelMetric(diagnostics.aic))}</td>
        <td class="numeric">${escapeHtml(formatModelMetric(diagnostics.bic))}</td>
        <td class="numeric">${Number(model.training_rows || diagnostics.training_rows || 0).toLocaleString()}</td>
      </tr>
    `;
  }

  function modelTableRows(models = modelRows, activeModelId = config?.active_model_id) {
    return normaliseModels(models).map((model) => {
      const diagnostics = model.diagnostics || model.metrics || {};
      return {
        ...model,
        active: model.model_id === activeModelId || Boolean(model.active),
        model_label: modelLabel(model),
        created_sort: modelCreatedSort(model.created_at),
        created_display: formatModelCreated(model.created_at),
        weight_display: modelWeightLabel(model.denominator_column || model.offset_column),
        deviance: modelNumberOrNull(diagnostics.deviance),
        aic: modelNumberOrNull(diagnostics.aic),
        bic: modelNumberOrNull(diagnostics.bic),
        training_rows: Number(model.training_rows || diagnostics.training_rows || 0),
      };
    });
  }

  function modelCreatedSort(value) {
    const time = new Date(value || "").getTime();
    return Number.isFinite(time) ? time : 0;
  }

  function activeModelDotFormatter(cell) {
    return cell.getValue() ? '<span class="glm-model-active-dot" title="Active model" aria-label="Active model"></span>' : "";
  }

  function modelNameFormatter(cell) {
    return `<span class="glm-model-name-main">${escapeHtml(cell.getValue() || "")}</span>`;
  }

  function syncSelectedModelsFromTable() {
    selectedModelIds = new Set(selectedModelIdList());
    updateModelActionButtons();
  }

  function selectedModelIdList() {
    const ids = modelTable && typeof modelTable.getSelectedData === "function"
      ? modelTable.getSelectedData().map((row) => row?.model_id)
      : Array.from(document.querySelectorAll('#glmModelFallback [data-glm-model-row][aria-selected="true"]'))
        .map((row) => row.dataset.glmModelRow);
    return [...new Set(ids.map((id) => String(id || "")).filter(Boolean))];
  }

  function restoreModelSelection(ids) {
    const selected = new Set((ids || []).map((id) => String(id || "")).filter(Boolean));
    selectedModelIds = selected;
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
    for (const row of document.querySelectorAll("#glmModelFallback [data-glm-model-row]")) {
      const rowId = String(row.dataset.glmModelRow || "");
      const active = selected.has(rowId);
      row.classList.toggle("selected", active);
      row.setAttribute("aria-selected", String(active));
    }
  }

  function updateModelActionButtons() {
    const selectedCount = selectedModelIdList().length;
    const disableActions = isBuilding;
    const activate = el("glmActivateModelBtn");
    const rename = el("glmRenameModelBtn");
    const deleteButton = el("glmDeleteModelBtn");
    if (activate) activate.disabled = disableActions || selectedCount !== 1;
    if (rename) rename.disabled = disableActions || selectedCount !== 1;
    if (deleteButton) deleteButton.disabled = disableActions || selectedCount < 1;
  }

  async function refreshModelListIfNeeded(options = {}) {
    if (isBuilding) return;
    const now = Date.now();
    if (!options.force && now - modelListLastRefreshAt < GLM_MODEL_LIST_POLL_MS) return;
    modelListLastRefreshAt = now;
    const seq = modelListRefreshSeq + 1;
    modelListRefreshSeq = seq;
    try {
      const data = await api("/api/glm/config", { method: "GET", clientTiming: true });
      if (seq !== modelListRefreshSeq) return;
      config = data;
      modelRows = normaliseModels(data.models || []);
      setDatasetGlmCount(modelRows.length);
      renderModelTable(modelRows, data.active_model_id);
      syncSidebarModelChooser(modelRows, data.active_model_id);
    } catch (error) {
      setGlmNotice(error.message);
    }
  }

  async function activateModel(modelId) {
    if (isBuilding || !modelId) return;
    try {
      const result = await api(`/api/glm/models/${encodeURIComponent(modelId)}/activate`, { method: "POST", body: "{}" });
      await applyModelMutationResult(result);
    } catch (error) {
      setGlmNotice(error.message);
    }
  }

  async function activateSelectedModel() {
    const [modelId] = selectedModelIdList();
    if (modelId) await activateModel(modelId);
  }

  async function renameSelectedModel() {
    if (isBuilding) return;
    const [modelId] = selectedModelIdList();
    if (!modelId) return;
    const newModelId = window.prompt("Rename GLM model", modelId);
    if (newModelId === null) return;
    const trimmed = newModelId.trim();
    if (!trimmed || trimmed === modelId) return;
    try {
      const result = await api(`/api/glm/models/${encodeURIComponent(modelId)}/rename`, {
        method: "POST",
        body: JSON.stringify({ new_model_id: trimmed }),
      });
      await applyModelMutationResult(result);
    } catch (error) {
      setGlmNotice(error.message);
    }
  }

  async function deleteSelectedModels() {
    if (isBuilding) return;
    const modelIds = selectedModelIdList();
    if (!modelIds.length) return;
    const label = modelIds.length === 1 ? `GLM model "${modelIds[0]}"` : `${modelIds.length} GLM models`;
    const confirmed = confirm(`Delete ${label}? This deletes the selected .lucidum model folder${modelIds.length === 1 ? "" : "s"}.`);
    if (!confirmed) return;
    let result = null;
    let deletedCount = 0;
    try {
      for (const modelId of modelIds) {
        result = await api(`/api/glm/models/${encodeURIComponent(modelId)}`, { method: "DELETE", body: "{}" });
        deletedCount += 1;
      }
      await applyModelMutationResult(result);
    } catch (error) {
      try {
        const latest = await api("/api/glm/config", { method: "GET", clientTiming: true });
        await applyModelMutationResult({ config: latest });
      } catch (_) {
      }
      const prefix = deletedCount > 0 ? `${deletedCount} deleted. ` : "";
      setGlmNotice(`${prefix}${error.message}`);
    }
  }

  async function applyModelMutationResult(result) {
    const nextConfig = result.config || config || {};
    await reloadSchema(preferredModelSource(result, nextConfig), { modelKind: "glm" });
    const preserveProfile = clearCachesAfterGlmModelSourceChange();
    activeDetail = null;
    coefficientRows = [];
    setDatasetGlmCount(Array.isArray(nextConfig?.models) ? nextConfig.models.length : null);
    setGlmNotice("");
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
    renderExpectedNumerators();
    renderFeatures();
    updateAxisControls();
  }

  function clearCachesAfterGlmModelSourceChange() {
    const preserveProfile = state.tool === "column_profile";
    clearToolCaches(preserveProfile ? { preserve: ["column_profile"] } : {});
    return preserveProfile;
  }

  function preferredModelSource(result, data) {
    const currentKind = dataSourceById(state.source)?.kind || "";
    const configActiveModel = (data?.models || []).find((item) => item.active);
    const activeModel = result?.deleted_model_id ? null : (result?.model || configActiveModel);
    const predictionSource = activeModel?.sources?.predictions || configActiveModel?.sources?.predictions || "";
    if (currentKind === "glm_predictions") return predictionSource || "dataset";
    if (!dataSourceById(state.source) && predictionSource) return predictionSource;
    const fallbackModel = (data?.models || []).find((item) => item.active);
    return dataSourceById(state.source) ? state.source || "dataset" : fallbackModel?.sources?.predictions || "dataset";
  }

  function dataSourceById(sourceId) {
    return (state.schema?.data_sources || []).find((source) => source.id === sourceId) || null;
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
    const modelsByGroup = new Map();
    for (const model of normalisedModels) {
      const group = modelGroupLabel(model);
      if (!modelsByGroup.has(group)) modelsByGroup.set(group, []);
      modelsByGroup.get(group).push(model);
    }
    const groups = [...modelsByGroup.keys()];
    if (!state.glmModelGroupsInitialised) {
      groups.forEach((group) => state.collapsedGlmModelGroups.add(group));
      const openGroup = activeModel ? modelGroupLabel(activeModel) : groups[0];
      if (openGroup) state.collapsedGlmModelGroups.delete(openGroup);
      state.glmModelGroupsInitialised = true;
    }
    for (const group of state.collapsedGlmModelGroups) {
      if (!groups.includes(group)) state.collapsedGlmModelGroups.delete(group);
    }
    list.innerHTML = "";
    if (!normalisedModels.length) {
      list.innerHTML = `<div class="glm-empty-state">No GLMs built yet</div>`;
      return;
    }
    for (const group of groups) {
      const collapsed = state.collapsedGlmModelGroups.has(group);
      const heading = document.createElement("button");
      heading.type = "button";
      heading.className = "saved-filter-theme glm-model-theme";
      heading.dataset.glmModelGroup = group;
      heading.setAttribute("aria-expanded", String(!collapsed));
      heading.setAttribute("aria-label", `${collapsed ? "Expand" : "Collapse"} ${group} GLM models`);
      heading.title = `${collapsed ? "Expand" : "Collapse"} ${group} GLM models`;
      heading.innerHTML = `<span class="saved-filter-theme-icon" aria-hidden="true"></span><span class="saved-filter-theme-label">${escapeHtml(group)}</span>`;
      heading.addEventListener("click", () => toggleGlmModelGroup(group));
      list.append(heading);
      for (const model of modelsByGroup.get(group) || []) {
        const active = model.model_id === activeModelId;
        const button = document.createElement("button");
        button.type = "button";
        button.className = `feature glm-model-option${active ? " active" : ""}`;
        button.dataset.glmModelId = model.model_id;
        button.dataset.glmModelGroup = group;
        button.hidden = collapsed;
        button.setAttribute("role", "option");
        button.setAttribute("aria-selected", String(active));
        button.innerHTML = `<span class="saved-filter-name">${escapeHtml(modelLabel(model))}</span><span class="glm-model-detail">${escapeHtml(glmModelDetailLabel(model))}</span>`;
        button.addEventListener("click", () => {
          if (!active) activateModel(model.model_id);
        });
        list.append(button);
      }
    }
  }

  function toggleGlmModelGroup(group) {
    const collapsed = !state.collapsedGlmModelGroups.has(group);
    if (collapsed) state.collapsedGlmModelGroups.add(group);
    else state.collapsedGlmModelGroups.delete(group);
    const list = el("glmModelSelect");
    list.querySelectorAll(".glm-model-theme").forEach((heading) => {
      if (heading.dataset.glmModelGroup !== group) return;
      heading.setAttribute("aria-expanded", String(!collapsed));
      heading.setAttribute("aria-label", `${collapsed ? "Expand" : "Collapse"} ${group} GLM models`);
      heading.title = `${collapsed ? "Expand" : "Collapse"} ${group} GLM models`;
    });
    list.querySelectorAll(".glm-model-option").forEach((button) => {
      if (button.dataset.glmModelGroup === group) button.hidden = collapsed;
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
    if (aceEditor) {
      aceEditor.setTheme(document.body.classList.contains("dark") ? "ace/theme/monokai" : "ace/theme/textmate");
    }
  }

  return {
    buildRequest,
    fetchData,
    openModelNavigator,
    render,
    refreshTheme,
    syncSidebarFromSchema,
    useCached,
  };
}
