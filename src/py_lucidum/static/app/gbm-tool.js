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
  let evaluationChart = null;
  let evaluationResizeObserver = null;

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
    mount.innerHTML = `
      <div class="gbm-tool">
        <div id="gbmNotice" class="gbm-notice hidden" role="alert" aria-live="polite"></div>
        <div class="gbm-toolbar">
          <div class="gbm-tabs tabs workspace-tabs">
            <button class="tab ${activeTab === "features" ? "active" : ""}" type="button" data-gbm-tab="features">Features and parameters</button>
            <button class="tab ${activeTab === "models" ? "active" : ""}" type="button" data-gbm-tab="models">Model navigator</button>
            <button class="tab ${activeTab === "trees" ? "active" : ""}" type="button" data-gbm-tab="trees">Tree viewer</button>
          </div>
          <div class="gbm-actions">
            <div id="gbmShapRows" class="gbm-shap-rows" role="radiogroup" aria-label="SHAP rows">
              <span class="gbm-shap-label">SHAP rows</span>
              ${shapOptionsHtml(data.shap_options || [])}
            </div>
            <button id="gbmCreateSampleBtn" class="tab gbm-action-button gbm-sample-button ${state.gbmCreateSample ? "active" : ""}" type="button" aria-pressed="${state.gbmCreateSample ? "true" : "false"}">${state.gbmCreateSample ? "Sample pending" : "Create sample column"}</button>
            <button id="gbmTrainBtn" class="tab gbm-action-button gbm-train-button ${isTraining ? "training" : ""}" type="button" ${isTraining ? "disabled aria-busy=\"true\"" : ""}>${isTraining ? "Training..." : "Train GBM"}</button>
          </div>
        </div>
        <div class="gbm-tab-panel ${activeTab === "features" ? "" : "hidden"}" data-gbm-panel="features">
          <div class="gbm-feature-layout">
            <section class="gbm-panel-section gbm-grid-panel">
              <h3 class="gbm-section-title">Features</h3>
              <div id="gbmFeatureGrid" class="gbm-grid"></div>
              <div id="gbmFeatureFallback" class="gbm-fallback-table"></div>
            </section>
            <section class="gbm-right-panel">
              <section class="gbm-panel-section">
                <h3 class="gbm-section-title">Parameters</h3>
                <div id="gbmParameterGrid" class="gbm-grid gbm-parameter-grid"></div>
                <div id="gbmParameterFallback" class="gbm-fallback-table"></div>
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
          <div class="gbm-tree-toolbar">
            <select id="gbmTreeSelect" aria-label="GBM tree"></select>
          </div>
          <div id="gbmTreeChart" class="gbm-tree-chart"></div>
        </div>
      </div>
    `;
    bindTabs(mount);
    bindModelTable(mount);
    syncSidebarModelSelect(data.models || [], data.active_model_id);
    bindFeatureActions();
    renderTables(data);
    if (data.active_model_id) loadModelDetail(data.active_model_id);
    setDuckDbTiming(tool, data.timings || {});
    setClientTiming(tool, data.client_timings || {});
    setRenderTiming(tool, 0);
    saveToolPresentation(tool, { groupMeta, chartMessage: "" });
  }

  function shapOptionsHtml(options) {
    const rows = options.length ? options : [
      { value: "zero", label: "Zero rows" },
      { value: "10k", label: "10k rows" },
      { value: "all", label: "All rows" },
    ];
    return rows.map((row, index) => `
      <label class="gbm-shap-option">
        <input type="radio" name="gbmShapRows" value="${escapeHtml(row.value)}" ${index === 0 ? "checked" : ""} />
        <span>${escapeHtml(row.label)}</span>
      </label>
    `).join("");
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
    el("gbmCreateSampleBtn")?.addEventListener("click", () => {
      state.gbmCreateSample = true;
      syncSampleButton();
      setGbmNotice("");
    });
    el("gbmTrainBtn")?.addEventListener("click", train);
    syncSampleButton();
    syncTrainingButton();
  }

  function syncSampleButton() {
    const button = el("gbmCreateSampleBtn");
    if (!button) return;
    button.textContent = state.gbmCreateSample ? "Sample pending" : "Create sample column";
    button.classList.toggle("active", Boolean(state.gbmCreateSample));
    button.setAttribute("aria-pressed", state.gbmCreateSample ? "true" : "false");
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

  function syncSidebarModelSelect(models, activeModelId) {
    const select = el("gbmActiveModelSelect");
    if (!select) return;
    select.innerHTML = "";
    select.append(new Option("No active model", ""));
    for (const model of models) {
      select.append(new Option(model.label || model.model_id, model.model_id));
    }
    select.value = activeModelId || "";
    select.onchange = () => {
      if (select.value) activateModel(select.value);
    };
  }

  function syncSidebarFromSchema() {
    const sources = state.schema?.data_sources || [];
    const models = [];
    const seen = new Set();
    for (const source of sources) {
      if (!String(source.id || "").startsWith("gbm:") || !source.model_id || seen.has(source.model_id)) continue;
      seen.add(source.model_id);
      models.push({
        model_id: source.model_id,
        label: String(source.label || source.model_id).replace(/\s+-\s+(Predictions|SHAP values|SHAP summary)$/i, ""),
        active: Boolean(source.active),
      });
    }
    const activeModel = models.find((model) => model.active)?.model_id || "";
    syncSidebarModelSelect(models, activeModel);
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

  function featureRowClasses(feature) {
    return [
      isFeatureSelectable(feature) ? "" : "gbm-feature-disabled",
      isFeatureSelectable(feature) && feature.high_cardinality ? "gbm-feature-warning" : "",
    ].filter(Boolean).join(" ");
  }

  function currentReservedFeatureNames() {
    const response = el("actualNumerator")?.value || "actualNumerator";
    const offset = el("denominator")?.value || "denominator";
    return new Set([response, offset].filter((value) => value && value !== "__none__"));
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
        <thead><tr><th>Model</th><th>Objective</th><th>Metric</th><th>Train</th><th>Test</th><th>Best iter.</th><th></th></tr></thead>
        <tbody>
          ${models.map((model) => `
            <tr class="${model.active ? "active" : ""}">
              <td>${escapeHtml(model.label || model.model_id)}</td>
              <td>${escapeHtml(model.objective || "")}</td>
              <td>${escapeHtml(model.metric || "")}</td>
              <td>${Number(model.training_rows || 0).toLocaleString()}</td>
              <td>${Number(model.test_rows || 0).toLocaleString()}</td>
              <td>${Number(model.best_iteration || 0).toLocaleString()}</td>
              <td><button type="button" data-gbm-activate="${escapeHtml(model.model_id)}">${model.active ? "Active" : "Activate"}</button></td>
            </tr>
          `).join("")}
        </tbody>
      </table>
    `;
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
      shap_rows: document.querySelector("input[name='gbmShapRows']:checked")?.value || "zero",
      sample_column: config?.sample_column || "",
      create_sample: Boolean(state.gbmCreateSample),
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
      const job = await api("/api/gbm/train", { method: "POST", body: JSON.stringify(payload), clientTiming: true });
      pollJob(job.job_id);
    } catch (error) {
      setTrainingState(false);
      setToolTimingFailed(tool);
      setGbmNotice(error.message);
    }
  }

  function pollJob(jobId) {
    clearTimeout(pollTimer);
    pollTimer = setTimeout(async () => {
      try {
        const job = await api(`/api/gbm/jobs/${encodeURIComponent(jobId)}`, { method: "GET", clientTiming: true });
        if (job.status === "queued" || job.status === "running") {
          setGroupMeta(tool, job.status === "queued" ? "GBM queued..." : "Training GBM...");
          pollJob(jobId);
          return;
        }
        if (job.status === "failed") {
          setTrainingState(false);
          setToolTimingFailed(tool);
          setGbmNotice(job.error || "GBM training failed");
          setGroupMeta(tool, "GBM failed");
          return;
        }
        await reloadSchema(job.result?.sources?.predictions);
        clearToolCaches();
        state.gbmCreateSample = false;
        const data = await api("/api/gbm/config", { method: "GET", clientTiming: true });
        const cache = toolCache(tool);
        cache.requestKey = stableConfigKey();
        cache.data = data;
        setTrainingState(false);
        measureToolRender(tool, () => render(data));
        refreshActiveTool({ force: true });
      } catch (error) {
        setTrainingState(false);
        setToolTimingFailed(tool);
        setGbmNotice(error.message);
      }
    }, 1000);
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
        syncSidebarModelSelect(result.config?.models || [], result.config?.active_model_id);
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
      renderTreeViewer();
    } catch (_) {
      activeDetail = null;
    }
  }

  function renderEvaluationChart() {
    const target = el("gbmEvaluationChart");
    if (!target || !window.echarts || !activeDetail?.training_log?.evaluation) return;
    disposeEvaluationChart();
    const rows = [];
    const evaluation = activeDetail.training_log.evaluation;
    for (const [datasetName, metrics] of Object.entries(evaluation)) {
      for (const [metricName, values] of Object.entries(metrics)) {
        rows.push({ datasetName, metricName, values });
      }
    }
    if (!rows.length) return;
    rows.sort(compareEvaluationRows);
    const metricNames = new Set(rows.map((row) => row.metricName));
    const primaryMetric = String(activeDetail?.manifest?.metric || rows[0]?.metricName || "metric");
    const maxIteration = Math.max(1, ...rows.map((row) => row.values.length));
    const xInterval = niceIterationInterval(maxIteration);
    const xMax = Math.ceil(maxIteration / xInterval) * xInterval;
    const title = evaluationTitle(rows, primaryMetric);
    evaluationChart = window.echarts.init(target);
    evaluationChart.setOption({
      animation: false,
      color: ["#ff140f", "#050505", "#2563eb", "#7c3aed"],
      title: {
        text: title,
        left: "center",
        top: 8,
        textStyle: { color: "#3f3f46", fontSize: 12, fontWeight: 800, lineHeight: 15 },
      },
      legend: {
        orient: "vertical",
        right: 8,
        top: "middle",
        itemWidth: 10,
        itemHeight: 10,
        textStyle: { color: "#3f3f46", fontSize: 12 },
      },
      tooltip: {
        trigger: "axis",
        valueFormatter: (value) => formatEvaluationValue(value),
      },
      grid: { left: 12, right: 82, top: 42, bottom: 38, containLabel: true },
      xAxis: {
        type: "value",
        min: 0,
        max: xMax,
        interval: xInterval,
        axisLabel: { color: "#4b5563", formatter: (value) => String(Math.round(Number(value))) },
        axisLine: { lineStyle: { color: "#4b5563" } },
        splitLine: { lineStyle: { color: "#e5e7eb" } },
      },
      yAxis: {
        type: "value",
        scale: true,
        splitNumber: 4,
        axisLabel: { color: "#4b5563", formatter: (value) => formatEvaluationAxisValue(value) },
        axisLine: { show: true, lineStyle: { color: "#4b5563" } },
        splitLine: { lineStyle: { color: "#e5e7eb" } },
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
    });
    bindEvaluationResize(target);
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

  function evaluationTitle(rows, primaryMetric) {
    const manifest = activeDetail?.manifest || {};
    const bestIteration = Math.max(0, Number(manifest.best_iteration || 0));
    const metric = primaryMetric || rows[0]?.metricName || "metric";
    const testRow = rows.find((row) => row.datasetName === "test" && row.metricName === metric)
      || rows.find((row) => row.datasetName === "test")
      || rows.find((row) => row.metricName === metric)
      || rows[0];
    const bestValue = valueAtIteration(testRow?.values || [], bestIteration) ?? lastFiniteValue(testRow?.values || []);
    const parts = [];
    parts.push(`evaluation metric: ${metric}`);
    if (bestValue !== null) parts.push(`test metric: ${formatEvaluationValue(bestValue)}`);
    if (bestIteration) parts.push(`best iteration: ${bestIteration.toLocaleString()}`);
    return parts.join(", ");
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

  function renderTreeViewer() {
    const select = el("gbmTreeSelect");
    const target = el("gbmTreeChart");
    const trees = activeDetail?.tree_dump?.tree_info || [];
    if (!select || !target) return;
    select.innerHTML = "";
    trees.forEach((tree, index) => select.append(new Option(`Tree ${index + 1}`, String(index))));
    select.onchange = () => renderTree(Number(select.value || 0));
    renderTree(0);
  }

  function renderTree(index) {
    const target = el("gbmTreeChart");
    if (!target || !window.echarts) return;
    const tree = activeDetail?.tree_dump?.tree_info?.[index]?.tree_structure;
    const chart = window.echarts.init(target);
    chart.setOption({
      animation: false,
      tooltip: { trigger: "item" },
      series: [{
        type: "tree",
        data: [treeNode(tree)],
        top: 20,
        bottom: 20,
        left: 40,
        right: 160,
        symbolSize: 8,
        label: { position: "left", verticalAlign: "middle", align: "right", fontSize: 11 },
        leaves: { label: { position: "right", align: "left" } },
        expandAndCollapse: true,
      }],
    });
  }

  function treeNode(node) {
    if (!node) return { name: "No tree" };
    if (node.leaf_index !== undefined) {
      return { name: `leaf ${node.leaf_index}: ${formatGain(node.leaf_value)}` };
    }
    return {
      name: `${node.split_feature} <= ${node.threshold}`,
      children: [treeNode(node.left_child), treeNode(node.right_child)],
    };
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
    syncSidebarFromSchema,
    useCached,
  };
}
