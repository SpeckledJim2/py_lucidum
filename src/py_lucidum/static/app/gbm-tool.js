import { createGbmTreeViewer } from "./gbm-tree-viewer.js";

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
};

const GBM_RUNNING_POLL_MS = 500;
const GBM_QUEUED_POLL_MS = 1000;

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
  setToolTimingFailed,
  startToolTiming,
  state,
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
  let activeTab = "features";
  let config = null;
  let activeDetail = null;
  let pollTimer = null;
  let isTraining = false;
  let liveProgress = null;
  let evaluationChart = null;
  let evaluationResizeObserver = null;
  const treeViewer = createGbmTreeViewer({ api, escapeHtml, loadTabulator, setGbmNotice });

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
    const groupMeta = "";
    setGroupMeta(tool, groupMeta);
    setStatus("");
    setChartMessage("");
    const mount = el("modelToolWrap");
    if (!mount) return;
    disposeEvaluationChart();
    treeViewer.dispose();
    mount.innerHTML = `
      <div class="gbm-tool">
        <div id="gbmNotice" class="gbm-notice hidden" role="alert" aria-live="polite"></div>
        <div class="gbm-toolbar">
          <div class="gbm-tabs tabs workspace-tabs">
            <button class="tab ${activeTab === "features" ? "active" : ""}" type="button" data-gbm-tab="features">Features and parameters</button>
            <button class="tab ${activeTab === "models" ? "active" : ""}" type="button" data-gbm-tab="models">Model navigator</button>
            <button class="tab ${activeTab === "trees" ? "active" : ""}" type="button" data-gbm-tab="trees">Tree viewer</button>
          </div>
          <div id="gbmTrainingStatus" class="gbm-training-status ${liveProgress ? "" : "hidden"}" aria-live="polite">${escapeHtml(liveProgress?.message || "")}</div>
        </div>
        <div class="gbm-tab-panel ${activeTab === "features" ? "" : "hidden"}" data-gbm-panel="features">
          <div class="gbm-feature-layout">
            <section class="gbm-panel-section gbm-grid-panel">
              <div class="gbm-section-header gbm-feature-section-header">
                <h3 class="gbm-section-title">Features</h3>
                <div class="gbm-feature-actions" role="group" aria-label="Feature selection">
                  <button id="gbmClearFeaturesBtn" class="tab gbm-inline-action-button" type="button">Clear all</button>
                  <button id="gbmSelectFeaturesBtn" class="tab gbm-inline-action-button" type="button">Select all</button>
                </div>
              </div>
              <div id="gbmFeatureGrid" class="gbm-grid"></div>
              <div id="gbmFeatureFallback" class="gbm-fallback-table"></div>
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
                          ${shapOptionsHtml(data.shap_options || [])}
                        </div>
                      </div>
                      ${shouldShowCreateSampleButton(data.sample) ? '<button id="gbmCreateSampleBtn" class="tab gbm-action-button gbm-sample-button" type="button">Create sample column</button>' : ""}
                    </div>
                  </div>
                </div>
              </section>
              <section class="gbm-panel-section">
                <h3 class="gbm-section-title">Evaluation log</h3>
                <div id="gbmEvaluationChart" class="gbm-evaluation-chart"></div>
              </section>
            </section>
          </div>
        </div>
        <div class="gbm-tab-panel ${activeTab === "models" ? "" : "hidden"}" data-gbm-panel="models">
          <div class="gbm-model-table-wrap">${modelTableHtml(data.models || [])}</div>
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
      </div>
    `;
    bindTabs(mount);
    bindModelTable(mount);
    syncSidebarModelChooser(data.models || [], data.active_model_id);
    bindFeatureActions();
    renderTables(data);
    if (data.active_model_id) loadModelDetail(data.active_model_id);
    if (activeTab === "trees") treeViewer.render(data.active_model_id || "");
    if (liveProgress) renderLiveProgress(liveProgress);
    setDuckDbTiming(tool, data.timings || {});
    setClientTiming(tool, data.client_timings || {});
    setRenderTiming(tool, 0);
    saveToolPresentation(tool, { groupMeta, chartMessage: "" });
  }

  function shapOptionsHtml(options) {
    const rows = options.length ? options : [
      { value: "0", label: "0" },
      { value: "10k", label: "10k" },
      { value: "100k", label: "100k" },
      { value: "all", label: "All" },
    ];
    return rows.map((row, index) => `
      <label class="gbm-shap-option">
        <input type="radio" name="gbmShapRows" value="${escapeHtml(row.value)}" ${index === 0 ? "checked" : ""} />
        <span>${escapeHtml(row.label)}</span>
      </label>
    `).join("");
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
        activeTab = button.dataset.gbmTab;
        render(config || {});
      });
    }
  }

  function bindFeatureActions() {
    el("gbmClearFeaturesBtn")?.addEventListener("click", () => setFeatureIncludes(false));
    el("gbmSelectFeaturesBtn")?.addEventListener("click", () => setFeatureIncludes(true));
    el("gbmCreateSampleBtn")?.addEventListener("click", createSampleColumn);
    el("gbmTrainBtn")?.addEventListener("click", train);
    syncTrainingButton();
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
    if (featureTable) {
      for (const row of featureTable.getRows()) {
        const data = row.getData();
        if (isFeatureSelectable(data)) row.update({ include });
      }
      return;
    }
    for (const checkbox of document.querySelectorAll("[data-gbm-feature]")) {
      const name = checkbox.getAttribute("data-gbm-feature") || "";
      const feature = (config?.features || []).find((item) => item.name === name);
      if (!feature || !isFeatureSelectable(feature)) continue;
      checkbox.checked = include;
      checkbox.dispatchEvent(new Event("change", { bubbles: true }));
    }
  }

  function setTrainingState(active) {
    isTraining = Boolean(active);
    syncTrainingButton();
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

  function setTrainingStatus(message, phase = "") {
    const status = el("gbmTrainingStatus");
    if (!status) return;
    const text = String(message || "");
    status.textContent = text;
    status.dataset.phase = String(phase || "");
    status.classList.toggle("hidden", !text);
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
      list.innerHTML = `<div class="gbm-empty-state">No GBM models have been trained yet.</div>`;
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
      metric: String(model?.metric || ""),
      best_iteration: Number(model?.best_iteration || 0),
      created_at: String(model?.created_at || ""),
      active: Boolean(model?.active),
    };
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
    return `${model.response_column || "actualNumerator"} / ${model.offset_column || "Average row value"}`;
  }

  function modelLabel(model) {
    return model.label || model.model_id;
  }

  function modelDetailLabel(model) {
    const parts = [];
    if (model.metric) parts.push(model.metric);
    if (model.best_iteration) parts.push(`iter ${model.best_iteration.toLocaleString()}`);
    return parts.join(" · ");
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
        best_iteration: source.best_iteration,
      });
    }
    const activeModel = models.find((model) => model.active)?.model_id || "";
    syncSidebarModelChooser(models, activeModel);
  }

  async function renderTables(data) {
    featureTable = null;
    parameterTable = null;
    const features = data.features || [];
    const parameters = data.parameters || [];
    try {
      const Tabulator = await loadTabulator();
      if (!config || data !== config) return;
      featureTable = new Tabulator("#gbmFeatureGrid", {
        data: features,
        height: "100%",
        layout: "fitColumns",
        initialSort: [{ column: "gain", dir: "desc" }],
        columns: [
          { title: "Feature", field: "name", formatter: featureNameFormatter, cssClass: "gbm-feature-name-cell", widthGrow: 3, headerSort: true },
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
          { title: "Gain", field: "gain", formatter: (cell) => formatGain(cell.getValue()), sorter: "number", hozAlign: "center", headerHozAlign: "center", width: 125 },
        ],
        rowFormatter: (row) => {
          const data = row.getData();
          const element = row.getElement();
          element.classList.toggle("gbm-feature-invalid", isInvalidFeature(data));
          element.classList.toggle("gbm-feature-disabled", !isFeatureSelectable(data));
          element.classList.toggle("gbm-feature-warning", isFeatureSelectable(data) && Boolean(data.high_cardinality));
        },
      });
      parameterTable = new Tabulator("#gbmParameterGrid", {
        data: parameters,
        height: "100%",
        layout: "fitColumns",
        initialSort: [{ column: "important", dir: "desc" }],
        columns: [
          { title: "Parameter", field: "name", widthGrow: 2 },
          { title: "Value", field: "value", editor: parameterValueEditor, widthGrow: 1 },
        ],
      });
    } catch (_) {
      renderFeatureFallback(features);
      renderParameterFallback(parameters);
    }
  }

  function featureNameFormatter(cell) {
    const feature = cell.getRow().getData();
    return `
      <span class="gbm-feature-name-main">${escapeHtml(feature.name)}</span>
      <span class="gbm-feature-kind kind">${escapeHtml(featureTypeLabel(feature))}</span>
    `;
  }

  function featureNameHtml(feature) {
    return `
      <span class="gbm-feature-name-line">
        <span class="gbm-feature-name-main">${escapeHtml(feature.name)}</span>
        <span class="gbm-feature-kind kind">${escapeHtml(featureTypeLabel(feature))}</span>
      </span>
    `;
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
      cell.getRow().update({ include: checkbox.checked });
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
    target.innerHTML = `
      <table>
        <thead><tr><th>Feature</th><th>Use</th><th>Monotonicity</th><th>Gain</th></tr></thead>
        <tbody>
          ${features.map((feature) => `
            <tr class="${featureRowClasses(feature)}">
              <td>${featureNameHtml(feature)}</td>
              <td class="gbm-use-cell">${isFeatureSelectable(feature) ? `<input type="checkbox" data-gbm-feature="${escapeHtml(feature.name)}" ${feature.include ? "checked" : ""} />` : ""}</td>
              <td><input data-gbm-monotonicity="${escapeHtml(feature.name)}" value="${escapeHtml(feature.monotonicity || "")}" ${isFeatureSelectable(feature) ? "" : "disabled"} /></td>
              <td class="numeric gbm-gain-cell">${formatGain(feature.gain)}</td>
            </tr>
          `).join("")}
        </tbody>
      </table>
    `;
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
    if (Array.isArray(configured) && configured.length) return configured.map((value) => String(value));
    return GBM_PARAMETER_OPTIONS[parameterName] || [];
  }

  function parameterValueEditor(cell, onRendered, success, cancel) {
    const rowData = cell.getRow().getData();
    const options = parameterOptionsForName(rowData.name);
    let submitted = false;
    const submit = (value) => {
      if (submitted) return;
      submitted = true;
      success(value);
    };
    if (options.length) {
      const select = document.createElement("select");
      select.className = "gbm-parameter-select";
      select.setAttribute("aria-label", rowData.name);
      for (const option of options) {
        select.append(new Option(option, option));
      }
      select.value = String(cell.getValue() ?? "");
      select.addEventListener("change", () => submit(select.value));
      select.addEventListener("blur", () => submit(select.value));
      select.addEventListener("keydown", (event) => {
        if (event.key === "Enter") submit(select.value);
        if (event.key === "Escape") cancel();
      });
      onRendered(() => select.focus());
      return select;
    }

    const input = document.createElement("input");
    input.type = "text";
    input.className = "gbm-parameter-input";
    input.value = String(cell.getValue() ?? "");
    input.addEventListener("change", () => submit(input.value));
    input.addEventListener("blur", () => submit(input.value));
    input.addEventListener("keydown", (event) => {
      if (event.key === "Enter") submit(input.value);
      if (event.key === "Escape") cancel();
    });
    onRendered(() => {
      input.focus();
      input.select();
    });
    return input;
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

  function modelTableHtml(models) {
    if (!models.length) return `<div class="gbm-empty-state">No GBM models have been trained yet.</div>`;
    return `
      <table class="gbm-model-table">
        <thead>
          <tr>
            <th>Model</th>
            <th>Created</th>
            <th>Response</th>
            <th>Weight</th>
            <th>Objective</th>
            <th>Metric</th>
            <th>Train</th>
            <th>Test</th>
            <th>Scored</th>
            <th>Best iter.</th>
            <th>Run time</th>
            <th>Sample</th>
            <th></th>
          </tr>
        </thead>
        <tbody>
          ${models.map((model) => `
            <tr class="${model.active ? "active" : ""}">
              <td class="gbm-model-name-cell">
                <span class="gbm-model-name-main">${escapeHtml(model.label || model.model_id)}</span>
              </td>
              <td>${escapeHtml(formatModelCreated(model.created_at))}</td>
              <td>${escapeHtml(model.response_column || "actualNumerator")}</td>
              <td>${escapeHtml(model.offset_column || "Average row value")}</td>
              <td>${escapeHtml(model.objective || "")}</td>
              <td>${escapeHtml(model.metric || "")}</td>
              <td class="numeric">${formatModelCount(model.training_rows)}</td>
              <td class="numeric">${formatModelCount(model.test_rows)}</td>
              <td class="numeric">${formatModelCount(model.scored_rows)}</td>
              <td class="numeric">${formatModelCount(model.best_iteration)}</td>
              <td class="numeric">${escapeHtml(formatModelRuntime(model))}</td>
              <td>${escapeHtml(formatSampleMode(model.sample_column, model.sample_source))}</td>
              <td><button class="gbm-model-activate-button" type="button" data-gbm-activate="${escapeHtml(model.model_id)}">${model.active ? "Active" : "Activate"}</button></td>
            </tr>
          `).join("")}
        </tbody>
      </table>
    `;
  }

  function formatModelCount(value) {
    const number = Number(value);
    return Number.isFinite(number) && number > 0 ? Math.round(number).toLocaleString() : "0";
  }

  function formatModelRuntime(model) {
    const seconds = Number(model?.timings?.training_seconds ?? model?.training_seconds);
    if (!Number.isFinite(seconds) || seconds < 0) return "--";
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
    return date.toLocaleString(undefined, {
      day: "2-digit",
      hour: "2-digit",
      minute: "2-digit",
      month: "short",
    });
  }

  function formatSampleMode(value, source = "") {
    const text = String(value || "").trim();
    if (!text) return "All rows";
    if (String(source || "").trim() === "generated") return "Generated 60/20/20";
    return text;
  }

  function bindModelTable(mount) {
    for (const button of mount.querySelectorAll("[data-gbm-activate]")) {
      button.addEventListener("click", () => activateModel(button.dataset.gbmActivate));
    }
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
      return applyReserved({
        ...feature,
        include: checkbox ? checkbox.checked : feature.include,
        monotonicity: monotonicity ? monotonicity.value : feature.monotonicity,
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

  async function train() {
    if (isTraining) return;
    setStatus("");
    setChartMessage("");
    const payload = {
      label: `GBM ${new Date().toISOString().slice(0, 19).replace("T", " ")}`,
      response: el("actualNumerator")?.value || "actualNumerator",
      offset: el("denominator")?.value || "denominator",
      features: currentFeatureRows(),
      parameters: currentParameters(),
      shap_rows: document.querySelector("input[name='gbmShapRows']:checked")?.value || "0",
      sample_column: config?.sample?.column || config?.sample_column || "",
      sample_source: config?.sample?.source || "none",
      create_sample: false,
    };
    try {
      const validation = await api("/api/gbm/validate", { method: "POST", body: JSON.stringify(payload) });
      if (!validation.ok) {
        setGbmNotice(validation.errors.join("; "));
        return;
      }
      setGbmNotice("");
      setGroupMeta(tool, "Training GBM...");
      startToolTiming(tool);
      setTrainingState(true);
      liveProgress = null;
      setTrainingStatus("Training GBM...", "queued");
      const job = await api("/api/gbm/train", { method: "POST", body: JSON.stringify(payload), clientTiming: true });
      applyJobProgress(job);
      pollJob(job.job_id, 0);
    } catch (error) {
      setTrainingState(false);
      setToolTimingFailed(tool);
      setTrainingStatus("");
      setGbmNotice(error.message);
    }
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
            setTrainingStatus(fallback, job.status);
            setGroupMeta(tool, fallback);
          }
          pollJob(jobId, job.status === "running" ? GBM_RUNNING_POLL_MS : GBM_QUEUED_POLL_MS);
          return;
        }
        if (job.status === "failed") {
          setTrainingState(false);
          setToolTimingFailed(tool);
          if (!job.progress) setTrainingStatus("GBM failed", "failed");
          setGbmNotice(job.error || "GBM training failed");
          setGroupMeta(tool, "GBM failed");
          return;
        }
        liveProgress = null;
        await reloadSchema(job.result?.sources?.predictions);
        clearToolCaches();
        const data = await api("/api/gbm/config", { method: "GET", clientTiming: true });
        const cache = toolCache(tool);
        cache.requestKey = stableConfigKey();
        cache.data = data;
        setTrainingState(false);
        setTrainingStatus("");
        measureToolRender(tool, () => render(data));
        refreshActiveTool({ force: true });
      } catch (error) {
        setTrainingState(false);
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
    setTrainingStatus(progress.message || "", progress.phase || "");
    if (progress.message) setGroupMeta(tool, progress.message);
    if (progress.evaluation) {
      renderEvaluationChart({
        evaluation: progress.evaluation,
        progress,
        metric: progress.metric,
        manifest: {
          metric: progress.metric,
          best_iteration: progress.iteration,
        },
      });
    }
  }

  async function activateModel(modelId) {
    if (!modelId) return;
    try {
      const result = await api(`/api/gbm/models/${encodeURIComponent(modelId)}/activate`, { method: "POST", body: "{}" });
      await reloadSchema(result.model?.sources?.predictions);
      clearToolCaches();
      config = result.config;
      if (state.tool === tool) {
        measureToolRender(tool, () => render(result.config));
      } else {
        syncSidebarModelChooser(result.config?.models || [], result.config?.active_model_id);
        await refreshActiveTool({ force: true });
      }
    } catch (error) {
      setGbmNotice(error.message);
    }
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
    const xInterval = niceIterationInterval(maxIteration);
    const xMax = Math.ceil(maxIteration / xInterval) * xInterval;
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
        valueFormatter: (value) => formatEvaluationValue(value),
      },
      grid: { left: 12, right: 82, top: 42, bottom: 38, containLabel: true },
      xAxis: {
        type: "value",
        min: 0,
        max: xMax,
        interval: xInterval,
        axisLabel: { color: mutedColor, formatter: (value) => String(Math.round(Number(value))) },
        axisLine: { lineStyle: { color: mutedColor } },
        splitLine: { lineStyle: { color: lineColor } },
      },
      yAxis: {
        type: "value",
        scale: true,
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
        data: row.values.map((value, index) => [index + 1, value]),
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

  function niceIterationInterval(maxIteration) {
    const raw = Math.max(1, Number(maxIteration || 1) / 4);
    const magnitude = 10 ** Math.floor(Math.log10(raw));
    for (const step of [1, 2, 5, 10]) {
      const interval = step * magnitude;
      if (interval >= raw) return interval;
    }
    return 10 * magnitude;
  }

  function formatEvaluationValue(value) {
    const number = Number(value);
    if (!Number.isFinite(number)) return "";
    return number.toLocaleString(undefined, { maximumFractionDigits: 4 });
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
    render,
    refreshTheme() {
      if (liveProgress?.evaluation) {
        renderLiveProgress(liveProgress);
      } else {
        renderEvaluationChart();
      }
      treeViewer.refreshTheme();
    },
    syncSidebarFromSchema,
    useCached,
  };
}
