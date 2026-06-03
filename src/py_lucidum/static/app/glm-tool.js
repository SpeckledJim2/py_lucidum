import { loadTabulator } from "./shared/tabulator.js";

const GLM_RUNNING_POLL_MS = 500;
const GLM_QUEUED_POLL_MS = 1000;
const GLM_MODEL_LIST_POLL_MS = 2000;
const ACE_BASE_PATH = "/static/vendor/ace";
const GLM_BUILDER_SPLIT_STORAGE_KEY = "py_lucidum_glm_formula_panel_width";

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
  let pollTimer = null;
  let modelListRefreshSeq = 0;
  let modelListLastRefreshAt = 0;
  let isBuilding = false;
  let liveProgress = null;
  let aceEditor = null;
  let editorInitialisedFor = null;
  let editorFontSize = Number(localStorage.getItem("py_lucidum_glm_font_size")) || 14;
  let selectedFamily = localStorage.getItem("py_lucidum_glm_family") || "normal";
  let selectedTrainingScope = localStorage.getItem("py_lucidum_glm_training_scope") || "all";
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
    const groupMeta = "";
    setGroupMeta(tool, groupMeta);
    setStatus("");
    setChartMessage("");
    disposeEditor();
    modelTable = null;
    const mount = el("modelToolWrap");
    if (!mount) return;
    mount.innerHTML = shellHtml(data);
    bindTabs(mount);
    bindBuilderControls();
    bindBuilderResizer();
    bindModelActions();
    renderModelTable(modelRows, data.active_model_id);
    syncSidebarModelChooser(modelRows, data.active_model_id);
    renderCoefficientTable(coefficientRowsForActiveModel(data.active_model_id));
    initEditor();
    if (data.active_model_id) loadModelDetail(data.active_model_id);
    if (liveProgress) renderLiveProgress(liveProgress);
    setDuckDbTiming(tool, data.timings || {});
    setClientTiming(tool, data.client_timings || {});
    setRenderTiming(tool, 0);
    saveToolPresentation(tool, { groupMeta, chartMessage: "" });
  }

  function shellHtml(data = {}) {
    const sample = data.sample || {};
    const trainingDisabled = !sample.available || !Number(sample.training_rows || 0);
    if (trainingDisabled && selectedTrainingScope === "training" && !data.active_model_id) selectedTrainingScope = "all";
    const diagnostics = diagnosticsForActiveModel(data.active_model_id);
    const splitStyle = savedBuilderSplitWidthStyle();
    return `
      <div class="glm-tool">
        <div id="glmNotice" class="glm-notice hidden" role="alert" aria-live="polite"></div>
        <div class="glm-toolbar">
          <div class="glm-tabs tabs workspace-tabs">
            <button class="tab ${activeTab === "builder" ? "active" : ""}" type="button" data-glm-tab="builder">Formula builder</button>
            <button class="tab ${activeTab === "models" ? "active" : ""}" type="button" data-glm-tab="models">Model navigator</button>
          </div>
          <div id="glmBuildStatus" class="glm-build-status ${liveProgress ? "" : "hidden"}" aria-live="polite">${buildStatusHtml(liveProgress)}</div>
        </div>
        <div class="glm-tab-panel ${activeTab === "builder" ? "" : "hidden"}" data-glm-panel="builder">
          <div class="glm-builder-layout"${splitStyle ? ` style="${splitStyle}"` : ""}>
            <section class="glm-formula-panel">
              <div class="glm-panel-header">
                <h3 class="glm-panel-title">Formula and family</h3>
                <div class="glm-builder-actions">
                  <button id="glmClearFormulaBtn" class="tab glm-inline-action-button" type="button" title="Clear formula">× clear</button>
                  <button id="glmFontSmallerBtn" class="tab glm-icon-action-button" type="button" aria-label="Decrease formula font size" title="Decrease font size">A-</button>
                  <button id="glmFontLargerBtn" class="tab glm-icon-action-button" type="button" aria-label="Increase formula font size" title="Increase font size">A+</button>
                  <button id="glmBuildBtn" class="tab glm-build-button ${isBuilding ? "building" : ""}" type="button" ${isBuilding ? "disabled aria-busy=\"true\"" : ""}>${isBuilding ? "Building..." : "Build GLM"}</button>
                </div>
              </div>
              <div class="glm-builder-control-row">
                <div class="segmented glm-scope-control" role="group" aria-label="Rows to fit">
                  <button type="button" data-glm-scope="all" class="${selectedTrainingScope === "all" ? "active" : ""}">All</button>
                  <button type="button" data-glm-scope="training" class="${selectedTrainingScope === "training" ? "active" : ""}" ${trainingDisabled ? "disabled" : ""}>Training</button>
                </div>
                <div class="glm-family-row">
                  <input id="glmFamilyParameter" class="glm-family-parameter" type="text" inputmode="decimal" value="${escapeHtml(String(familyParameterDefault(data.families || [])))}" aria-label="GLM family parameter" />
                  <select id="glmFamilySelect" aria-label="GLM family">${familyOptionsHtml(data.families || [])}</select>
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
                  <div id="glmCoefficientMeta" class="glm-coefficient-meta">${diagnosticsHtml(diagnostics)}</div>
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
      </div>
    `;
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
    if (String(familyValue || "").trim() !== "tweedie") return null;
    const family = (families || []).find((row) => row.value === "tweedie");
    return family?.parameter || { label: "Tweedie power", default: "1.5" };
  }

  function syncFamilyParameterControl() {
    const select = el("glmFamilySelect");
    const input = el("glmFamilyParameter");
    if (!select || !input) return;
    const parameter = familyParameterConfig(select.value);
    input.disabled = !parameter;
    input.placeholder = parameter ? (parameter.label || "Tweedie power") : "";
    input.value = parameter ? (localStorage.getItem(`py_lucidum_glm_family_parameter_${select.value}`) || parameter.default || "") : "";
  }

  function bindTabs(mount) {
    mount.querySelectorAll("[data-glm-tab]").forEach((button) => {
      button.addEventListener("click", () => {
        activeTab = button.dataset.glmTab;
        mount.querySelectorAll("[data-glm-tab]").forEach((item) => item.classList.toggle("active", item === button));
        mount.querySelectorAll("[data-glm-panel]").forEach((panel) => panel.classList.toggle("hidden", panel.dataset.glmPanel !== activeTab));
        if (activeTab === "models") refreshModelListIfNeeded();
      });
    });
  }

  function bindBuilderControls() {
    syncFamilyParameterControl();
    el("glmFamilySelect")?.addEventListener("change", (event) => {
      selectedFamily = event.target.value;
      localStorage.setItem("py_lucidum_glm_family", selectedFamily);
      syncFamilyParameterControl();
    });
    el("glmFamilyParameter")?.addEventListener("change", (event) => {
      if (selectedFamily === "tweedie") localStorage.setItem(`py_lucidum_glm_family_parameter_${selectedFamily}`, event.target.value.trim());
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

  function bindModelActions() {
    el("glmActivateModelBtn")?.addEventListener("click", activateSelectedModel);
    el("glmRenameModelBtn")?.addEventListener("click", renameSelectedModel);
    el("glmDeleteModelBtn")?.addEventListener("click", deleteSelectedModels);
    updateModelActionButtons();
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
      aceEditor.session.on("change", () => {
        formulaDraft = aceEditor.getValue();
        localStorage.setItem("py_lucidum_glm_formula", formulaDraft);
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
    }
    if (el("glmFormulaText")) el("glmFormulaText").value = formulaDraft;
  }

  function adjustFontSize(delta) {
    editorFontSize = Math.max(10, Math.min(24, editorFontSize + delta));
    localStorage.setItem("py_lucidum_glm_font_size", String(editorFontSize));
    if (aceEditor) aceEditor.setFontSize(`${editorFontSize}px`);
    if (el("glmFormulaText")) el("glmFormulaText").style.fontSize = `${editorFontSize}px`;
  }

  function buildPayload() {
    const actual = el("actualNumerator")?.value || "";
    const denominator = el("denominator")?.value || "__none__";
    const family = el("glmFamilySelect")?.value || selectedFamily || "normal";
    return {
      formula: getFormulaText(),
      family,
      family_parameter: family === "tweedie" ? (el("glmFamilyParameter")?.value.trim() || "") : "",
      training_scope: selectedTrainingScope,
      response_column: actual,
      denominator_column: denominator === "__none__" ? "" : denominator,
      label: `GLM ${glmAutoModelTimeLabel()}`,
    };
  }

  async function buildModel() {
    if (isBuilding) return;
    const payload = buildPayload();
    const familyError = validateFamilyParameter(payload.family, payload.family_parameter);
    if (familyError) {
      setBuildFailure(familyError);
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
    if (family !== "tweedie") return "";
    const text = String(rawValue || "").trim();
    const value = Number(text);
    if (!text || !Number.isFinite(value) || value < 1 || value > 2) {
      return "Choose a Tweedie power from 1 to 2";
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
    const detail = rows ? `${rows.toLocaleString()} training rows` : "";
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
  }

  function diagnosticsForActiveModel(activeModelId) {
    const active = modelRows.find((model) => model.model_id === activeModelId) || modelRows.find((model) => model.active);
    return active?.diagnostics || active?.metrics || {};
  }

  function diagnosticsHtml(diagnostics = {}) {
    const primary = [
      ["Deviance", diagnostics.deviance],
      ["AIC", diagnostics.aic],
      ["Dispersion", diagnostics.dispersion],
    ].filter(([, value]) => value !== undefined && value !== null && value !== "");
    const secondary = [
      ["NAs in fitted", diagnostics.fitted_na_rows],
    ].filter(([, value]) => value !== undefined && value !== null && value !== "");
    const itemHtml = ([label, value]) => `<span><strong>${escapeHtml(label)}:</strong> ${escapeHtml(formatModelMetric(value))}</span>`;
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
    if (family === "tweedie" && familyParameter !== undefined && familyParameter !== null && String(familyParameter).trim() !== "") {
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
          <tr class="${glmCoefficientPValueClass(row.p_value)}">
            <td>${escapeHtml(row.term)}</td>
            <td class="numeric">${escapeHtml(formatModelMetric(row.estimate))}</td>
            <td class="numeric">${escapeHtml(formatModelMetric(row.std_error))}</td>
            <td class="numeric">${escapeHtml(formatPValue(row.p_value))}</td>
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
      if (meta) meta.innerHTML = diagnosticsHtml(diagnostics);
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
