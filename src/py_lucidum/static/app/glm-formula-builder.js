import {
  GLM_FORMULA_SNIPPETS,
  buildGroupedLevelsFormula,
  buildIndividualLevelsFormula,
  buildPiecewiseFormula,
  buildSnippetFormula,
  formulaColumnSuggestions,
  formulaCompletionContext,
  formulaFunctionSuggestions,
  formulaLiteral,
  formatDrawerInsertion,
  parseBreakpoints,
  rankFormulaSuggestions,
  withFormulaHeader,
} from "./glm-formula-assist.js";

const ACE_BASE_PATH = "/static/vendor/ace";

let aceLoaderPromise = null;

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

export function createGlmFormulaBuilder({
  api = null,
  el,
  escapeHtml,
  getColumns = () => [],
  getDenominator = () => "",
  getFamilies = () => [],
  onBuildModel = () => {},
  onCoefficientSearch = () => {},
  onCopyCoefficients = () => {},
  onCopyFormula = () => {},
  onDownloadCoefficients = () => {},
}) {
  let aceEditor = null;
  let editorInitialisedFor = null;
  let editorFontSize = Number(localStorage.getItem("py_lucidum_glm_font_size")) || 11;
  let selectedFamily = localStorage.getItem("py_lucidum_glm_family") || "normal";
  let selectedTrainingScope = localStorage.getItem("py_lucidum_glm_training_scope") || "all";
  let selectedRegularizationMode = localStorage.getItem("py_lucidum_glm_regularization_mode") || "none";
  let selectedRegularizationMix = localStorage.getItem("py_lucidum_glm_regularization_mix") || "0.5";
  let selectedRegularizationAlpha = localStorage.getItem("py_lucidum_glm_regularization_alpha") || "0.01";
  let formulaDraft = localStorage.getItem("py_lucidum_glm_formula")
    || "# GLM formula\n# Enter RHS terms, or response ~ terms\n";
  const storedBuilderPanel = localStorage.getItem("py_lucidum_glm_builder_panel");
  let builderPanel = ["formula", "parameters", "none"].includes(storedBuilderPanel)
    ? storedBuilderPanel
    : (localStorage.getItem("py_lucidum_glm_formula_assist_open") === "true" ? "formula" : "parameters");
  let formulaAssistOpen = builderPanel === "formula";
  let formulaAssistTab = localStorage.getItem("py_lucidum_glm_formula_assist_tab") || "snippets";
  let formulaAssistFeatureSearch = "";
  let formulaAssistSelectedFeature = localStorage.getItem("py_lucidum_glm_formula_assist_feature") || "";
  let formulaAssistSelectedSnippet = localStorage.getItem("py_lucidum_glm_formula_assist_snippet") || "identity";
  let formulaAssistBreakpoints = localStorage.getItem("py_lucidum_glm_formula_assist_breakpoints") || "";
  let formulaAssistLevelSearch = "";
  let formulaAssistSelectedLevelKeys = new Set();
  let formulaAssistIncludeHeader = localStorage.getItem("py_lucidum_glm_formula_assist_include_header") === "true";
  let formulaAssistCategoricalMode = localStorage.getItem("py_lucidum_glm_formula_assist_categorical_mode") || "group";
  let splitPanelWidth = null;
  if (!["group", "ind"].includes(formulaAssistCategoricalMode)) formulaAssistCategoricalMode = "group";
  let formulaAssistLevelRequestSeq = 0;
  const formulaAssistLevelCache = new Map();
  let autocompletePopup = null;
  let autocompleteItems = [];
  let autocompleteContext = null;
  let autocompleteIndex = 0;

  function ensureTrainingScope(data = {}) {
    const sample = data.sample || {};
    const trainingDisabled = !sample.available || !Number(sample.training_rows || 0);
    if (trainingDisabled && selectedTrainingScope === "training" && !data.active_model_id) selectedTrainingScope = "all";
    return trainingDisabled;
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

  function familyParameterConfig(familyValue, families = getFamilies()) {
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

  function formulaColumns() {
    return (getColumns() || []).filter((column) => column?.name);
  }

  function numericFormulaColumns() {
    return formulaColumns().filter((column) => ["integer", "numeric"].includes(String(column.kind || "")));
  }

  function categoricalFormulaColumns() {
    return formulaColumns().filter((column) => !["integer", "numeric"].includes(String(column.kind || "")));
  }

  function ensureFormulaAssistFeature(preferredKind = "") {
    const columns = preferredKind === "numeric"
      ? numericFormulaColumns()
      : (preferredKind === "categorical" ? categoricalFormulaColumns() : formulaColumns());
    if (columns.some((column) => column.name === formulaAssistSelectedFeature)) return formulaAssistSelectedFeature;
    formulaAssistSelectedFeature = columns[0]?.name || "";
    localStorage.setItem("py_lucidum_glm_formula_assist_feature", formulaAssistSelectedFeature);
    return formulaAssistSelectedFeature;
  }

  function filteredAssistColumns(columns = formulaColumns()) {
    const query = formulaAssistFeatureSearch.trim().toLowerCase();
    if (!query) return columns;
    return columns.filter((column) => String(column.name || "").toLowerCase().includes(query));
  }

  function formulaAssistColumnsForTab() {
    if (formulaAssistTab === "levels") return categoricalFormulaColumns();
    return numericFormulaColumns();
  }

  function numericFormulaSnippets() {
    return GLM_FORMULA_SNIPPETS;
  }

  function ensureFormulaAssistSnippet() {
    const snippets = numericFormulaSnippets();
    if (snippets.some((item) => item.id === formulaAssistSelectedSnippet)) return formulaAssistSelectedSnippet;
    formulaAssistSelectedSnippet = snippets[0]?.id || "identity";
    localStorage.setItem("py_lucidum_glm_formula_assist_snippet", formulaAssistSelectedSnippet);
    return formulaAssistSelectedSnippet;
  }

  function formulaAssistDrawerHtml() {
    const hidden = formulaAssistOpen ? "" : "hidden";
    return `<div id="glmFormulaAssistDrawer" class="glm-formula-assist-drawer ${hidden}">
      ${formulaAssistContentHtml()}
    </div>`;
  }

  function formulaAssistContentHtml() {
    const tabs = [
      ["snippets", "Numeric"],
      ["piecewise", "Piecewise linear"],
      ["levels", "Categorical"],
    ];
    return `
      <div class="glm-formula-assist-top">
        <div class="segmented glm-formula-assist-tabs" role="group" aria-label="Formula tools">
          ${tabs.map(([id, label]) => `<button type="button" data-glm-assist-tab="${id}" class="${formulaAssistTab === id ? "active" : ""}">${escapeHtml(label)}</button>`).join("")}
        </div>
        <label class="glm-formula-assist-header-toggle"><input id="glmFormulaAssistIncludeHeader" type="checkbox" ${formulaAssistIncludeHeader ? "checked" : ""} /> include header</label>
      </div>
      ${formulaAssistTab === "piecewise" ? formulaAssistPiecewiseHtml() : (formulaAssistTab === "levels" ? formulaAssistLevelsHtml() : formulaAssistSnippetsHtml())}
    `;
  }

  function formulaAssistFeaturePickerHtml(columns = formulaColumns(), selected = formulaAssistSelectedFeature) {
    return `
      <div class="glm-formula-assist-feature-block">
        <label class="glm-formula-assist-label" for="glmFormulaAssistFeatureSearch">Feature</label>
        <input id="glmFormulaAssistFeatureSearch" class="search glm-formula-assist-search" type="search" value="${escapeHtml(formulaAssistFeatureSearch)}" />
        <select id="glmFormulaAssistFeatureSelect" class="glm-formula-assist-feature-select" size="7">
          ${formulaAssistFeatureOptionsHtml(columns, selected)}
        </select>
      </div>
    `;
  }

  function formulaAssistFeatureOptionsHtml(columns = formulaAssistColumnsForTab(), selected = formulaAssistSelectedFeature) {
    const rows = filteredAssistColumns(columns);
    const safeSelected = rows.some((column) => column.name === selected) ? selected : (rows[0]?.name || "");
    return rows.map((column) => `<option value="${escapeHtml(column.name)}" ${column.name === safeSelected ? "selected" : ""}>${escapeHtml(column.name)}</option>`).join("");
  }

  function formulaAssistSnippetsHtml() {
    ensureFormulaAssistFeature("numeric");
    ensureFormulaAssistSnippet();
    const selected = formulaAssistSelectedFeature;
    const snippets = numericFormulaSnippets();
    const snippet = snippets.find((item) => item.id === formulaAssistSelectedSnippet) || snippets[0];
    const formula = buildSnippetFormula(snippet?.id || "identity", selected, {
      denominator: denominatorForOffsetSnippet(),
    });
    const preview = formulaAssistPreviewDisplayText(formula, selected);
    return `
      <div class="glm-formula-assist-grid">
        ${formulaAssistFeaturePickerHtml(numericFormulaColumns())}
        <div class="glm-formula-assist-main">
          <label class="glm-formula-assist-label">Snippet</label>
          <div id="glmFormulaAssistSnippetList" class="glm-formula-assist-snippet-list" role="listbox" aria-label="Snippet">
            ${snippets.map((item) => `<button type="button" class="glm-formula-assist-snippet-row ${item.id === formulaAssistSelectedSnippet ? "active" : ""}" data-glm-snippet-id="${escapeHtml(item.id)}" role="option" aria-selected="${item.id === formulaAssistSelectedSnippet ? "true" : "false"}">${escapeHtml(item.label)}</button>`).join("")}
          </div>
          <pre id="glmFormulaAssistPreview" class="glm-formula-assist-preview">${escapeHtml(preview)}</pre>
          <button id="glmFormulaAssistInsertBtn" class="tab glm-inline-action-button" type="button" ${preview ? "" : "disabled"}>Insert at cursor</button>
        </div>
      </div>
    `;
  }

  function formulaAssistPiecewiseHtml() {
    ensureFormulaAssistFeature("numeric");
    const parsed = parseBreakpoints(formulaAssistBreakpoints);
    const formula = !parsed.error ? buildPiecewiseFormula(formulaAssistSelectedFeature, parsed.values) : "";
    const preview = formulaAssistPreviewDisplayText(formula, formulaAssistSelectedFeature);
    return `
      <div class="glm-formula-assist-grid">
        ${formulaAssistFeaturePickerHtml(numericFormulaColumns())}
        <div class="glm-formula-assist-main">
          <label class="glm-formula-assist-label" for="glmFormulaAssistBreakpoints">Breaks</label>
          <input id="glmFormulaAssistBreakpoints" class="glm-formula-assist-breakpoints" type="text" inputmode="decimal" value="${escapeHtml(formulaAssistBreakpoints)}" />
          <pre id="glmFormulaAssistPreview" class="glm-formula-assist-preview">${escapeHtml(preview || parsed.error)}</pre>
          <button id="glmFormulaAssistInsertPiecewiseBtn" class="tab glm-inline-action-button" type="button" ${preview ? "" : "disabled"}>Insert at cursor</button>
        </div>
      </div>
    `;
  }

  function formulaAssistLevelsHtml() {
    ensureFormulaAssistFeature("categorical");
    const levelRows = levelCacheEntry(formulaAssistSelectedFeature, formulaAssistLevelSearch)?.values || [];
    const selectedValues = selectedFormulaAssistLevelValues();
    const formula = selectedValues.length
      ? (formulaAssistCategoricalMode === "ind"
        ? buildIndividualLevelsFormula(formulaAssistSelectedFeature, selectedValues)
        : buildGroupedLevelsFormula(formulaAssistSelectedFeature, selectedValues))
      : "";
    const preview = formulaAssistPreviewDisplayText(formula, formulaAssistSelectedFeature);
    return `
      <div class="glm-formula-assist-grid">
        ${formulaAssistFeaturePickerHtml(categoricalFormulaColumns())}
        <div class="glm-formula-assist-main">
          <label class="glm-formula-assist-label" for="glmFormulaAssistLevelSearch">Levels</label>
          <div class="glm-formula-assist-level-search-row">
            <input id="glmFormulaAssistLevelSearch" class="search glm-formula-assist-search" type="search" value="${escapeHtml(formulaAssistLevelSearch)}" />
            <div class="segmented glm-formula-assist-level-mode" role="radiogroup" aria-label="Categorical formula mode">
              <button type="button" data-glm-level-mode="group" role="radio" aria-checked="${formulaAssistCategoricalMode === "group" ? "true" : "false"}" class="${formulaAssistCategoricalMode === "group" ? "active" : ""}">group</button>
              <button type="button" data-glm-level-mode="ind" role="radio" aria-checked="${formulaAssistCategoricalMode === "ind" ? "true" : "false"}" class="${formulaAssistCategoricalMode === "ind" ? "active" : ""}">ind</button>
            </div>
          </div>
          <div id="glmFormulaAssistLevelList" class="glm-formula-assist-level-list">
            ${formulaAssistLevelRowsHtml(levelRows)}
          </div>
          <pre id="glmFormulaAssistPreview" class="glm-formula-assist-preview">${escapeHtml(preview)}</pre>
          <button id="glmFormulaAssistInsertLevelsBtn" class="tab glm-inline-action-button" type="button" ${preview ? "" : "disabled"}>Insert at cursor</button>
        </div>
      </div>
    `;
  }

  function denominatorForOffsetSnippet() {
    const denominator = String(getDenominator() || "").trim();
    if (!denominator || denominator === "__none__") return formulaAssistSelectedFeature;
    return denominator;
  }

  function renderFormulaAssistDrawer() {
    const drawer = el("glmFormulaAssistDrawer");
    if (!drawer) return;
    drawer.classList.toggle("hidden", !formulaAssistOpen);
    drawer.innerHTML = formulaAssistContentHtml();
    bindFormulaAssistDrawerControls();
    const parametersOpen = builderPanel === "parameters";
    el("glmBuilderParametersPanel")?.classList.toggle("hidden", !parametersOpen);
    syncBuilderPanelButton(el("glmFormulaAssistBtn"), formulaAssistOpen);
    syncBuilderPanelButton(el("glmModelParametersBtn"), parametersOpen);
    if (formulaAssistOpen && formulaAssistTab === "levels") loadFormulaAssistLevels();
    if (aceEditor) window.requestAnimationFrame(() => aceEditor?.resize());
  }

  function syncBuilderPanelButton(button, active) {
    if (!button) return;
    button.classList.toggle("active", active);
    button.setAttribute("aria-expanded", active ? "true" : "false");
  }

  function toggleBuilderPanel(panel) {
    builderPanel = builderPanel === panel ? "none" : panel;
    formulaAssistOpen = builderPanel === "formula";
    localStorage.setItem("py_lucidum_glm_builder_panel", builderPanel);
    localStorage.setItem("py_lucidum_glm_formula_assist_open", formulaAssistOpen ? "true" : "false");
    renderFormulaAssistDrawer();
  }

  function refreshFormulaAssistFeatureOptions() {
    const select = el("glmFormulaAssistFeatureSelect");
    if (!select) return;
    const columns = formulaAssistColumnsForTab();
    const rows = filteredAssistColumns(columns);
    if (!rows.some((column) => column.name === formulaAssistSelectedFeature)) {
      formulaAssistSelectedFeature = rows[0]?.name || "";
      localStorage.setItem("py_lucidum_glm_formula_assist_feature", formulaAssistSelectedFeature);
      if (formulaAssistTab === "levels") formulaAssistSelectedLevelKeys = new Set();
    }
    select.innerHTML = formulaAssistFeatureOptionsHtml(columns);
    select.value = formulaAssistSelectedFeature;
    refreshFormulaAssistPreview();
    if (formulaAssistTab === "levels" && formulaAssistSelectedFeature) loadFormulaAssistLevels();
    else if (formulaAssistTab === "levels") refreshFormulaAssistLevelRows([]);
  }

  function refreshFormulaAssistPreview() {
    const preview = el("glmFormulaAssistPreview");
    if (!preview) return;
    const text = formulaAssistPreviewText();
    preview.textContent = text;
    const button = formulaAssistInsertButton();
    if (button) button.disabled = !text || Boolean(formulaAssistPreviewError());
  }

  function formulaAssistPreviewText() {
    const error = formulaAssistPreviewError();
    if (error) return error;
    const text = formulaAssistGeneratedFormulaText();
    return formulaAssistPreviewDisplayText(text, formulaAssistSelectedFeature);
  }

  function formulaAssistPreviewDisplayText(text, feature) {
    const value = withFormulaHeader(text, feature, formulaAssistIncludeHeader);
    return formatDrawerInsertion(value, "__preview__").trimEnd();
  }

  function formulaAssistGeneratedFormulaText() {
    if (formulaAssistTab === "piecewise") {
      const parsed = parseBreakpoints(formulaAssistBreakpoints);
      if (parsed.error) return parsed.error;
      return buildPiecewiseFormula(formulaAssistSelectedFeature, parsed.values);
    }
    if (formulaAssistTab === "levels") {
      const values = selectedFormulaAssistLevelValues();
      if (!values.length) return "";
      return formulaAssistCategoricalMode === "ind"
        ? buildIndividualLevelsFormula(formulaAssistSelectedFeature, values)
        : buildGroupedLevelsFormula(formulaAssistSelectedFeature, values);
    }
    return buildSnippetFormula(formulaAssistSelectedSnippet, formulaAssistSelectedFeature, {
      denominator: denominatorForOffsetSnippet(),
    });
  }

  function formulaAssistPreviewError() {
    if (formulaAssistTab !== "piecewise") return "";
    return parseBreakpoints(formulaAssistBreakpoints).error;
  }

  function formulaAssistInsertButton() {
    if (formulaAssistTab === "piecewise") return el("glmFormulaAssistInsertPiecewiseBtn");
    if (formulaAssistTab === "levels") return el("glmFormulaAssistInsertLevelsBtn");
    return el("glmFormulaAssistInsertBtn");
  }

  function formulaAssistLevelRowsHtml(rows = []) {
    return rows.length ? rows.map((row) => {
      const value = String(row.value ?? "");
      const key = formulaAssistLevelValueKey(row.value);
      const checked = formulaAssistSelectedLevelKeys.has(key) ? "checked" : "";
      return `<label class="glm-formula-assist-level-row"><input type="checkbox" data-glm-level-key="${escapeHtml(key)}" data-glm-level-value="${escapeHtml(value)}" ${checked} /> <span>${escapeHtml(row.label || value)}</span><small>${Number(row.count || 0).toLocaleString()}</small></label>`;
    }).join("") : `<div class="glm-formula-assist-empty">No levels</div>`;
  }

  function refreshFormulaAssistLevelRows(rows = []) {
    const list = el("glmFormulaAssistLevelList");
    if (!list) return;
    list.innerHTML = formulaAssistLevelRowsHtml(rows);
    bindFormulaAssistLevelControls();
    refreshFormulaAssistPreview();
  }

  function bindFormulaAssistLevelControls() {
    document.querySelectorAll("[data-glm-level-value]").forEach((checkbox) => {
      checkbox.addEventListener("change", () => {
        const key = checkbox.dataset.glmLevelKey || "";
        if (checkbox.checked) formulaAssistSelectedLevelKeys.add(key);
        else formulaAssistSelectedLevelKeys.delete(key);
        refreshFormulaAssistPreview();
      });
    });
  }

  function refreshFormulaAssistSnippetSelection() {
    document.querySelectorAll("[data-glm-snippet-id]").forEach((button) => {
      const active = button.dataset.glmSnippetId === formulaAssistSelectedSnippet;
      button.classList.toggle("active", active);
      button.setAttribute("aria-selected", active ? "true" : "false");
    });
  }

  function bindFormulaAssistDrawerControls() {
    document.querySelectorAll("[data-glm-assist-tab]").forEach((button) => {
      button.addEventListener("click", () => {
        formulaAssistTab = button.dataset.glmAssistTab || "snippets";
        localStorage.setItem("py_lucidum_glm_formula_assist_tab", formulaAssistTab);
        renderFormulaAssistDrawer();
      });
    });
    el("glmFormulaAssistFeatureSearch")?.addEventListener("input", (event) => {
      formulaAssistFeatureSearch = event.target.value || "";
      refreshFormulaAssistFeatureOptions();
    });
    el("glmFormulaAssistFeatureSelect")?.addEventListener("change", (event) => {
      formulaAssistSelectedFeature = event.target.value || "";
      formulaAssistSelectedLevelKeys = new Set();
      localStorage.setItem("py_lucidum_glm_formula_assist_feature", formulaAssistSelectedFeature);
      refreshFormulaAssistPreview();
      if (formulaAssistTab === "levels") loadFormulaAssistLevels();
    });
    document.querySelectorAll("[data-glm-snippet-id]").forEach((button) => {
      button.addEventListener("click", () => {
        formulaAssistSelectedSnippet = button.dataset.glmSnippetId || "identity";
        localStorage.setItem("py_lucidum_glm_formula_assist_snippet", formulaAssistSelectedSnippet);
        refreshFormulaAssistSnippetSelection();
        refreshFormulaAssistPreview();
      });
    });
    el("glmFormulaAssistIncludeHeader")?.addEventListener("change", (event) => {
      formulaAssistIncludeHeader = Boolean(event.target.checked);
      localStorage.setItem("py_lucidum_glm_formula_assist_include_header", formulaAssistIncludeHeader ? "true" : "false");
      refreshFormulaAssistPreview();
    });
    el("glmFormulaAssistBreakpoints")?.addEventListener("input", (event) => {
      formulaAssistBreakpoints = event.target.value || "";
      localStorage.setItem("py_lucidum_glm_formula_assist_breakpoints", formulaAssistBreakpoints);
      refreshFormulaAssistPreview();
    });
    el("glmFormulaAssistLevelSearch")?.addEventListener("input", (event) => {
      formulaAssistLevelSearch = event.target.value || "";
      loadFormulaAssistLevels();
    });
    document.querySelectorAll("[data-glm-level-mode]").forEach((button) => {
      button.addEventListener("click", () => {
        formulaAssistCategoricalMode = button.dataset.glmLevelMode === "ind" ? "ind" : "group";
        localStorage.setItem("py_lucidum_glm_formula_assist_categorical_mode", formulaAssistCategoricalMode);
        document.querySelectorAll("[data-glm-level-mode]").forEach((item) => {
          const active = item.dataset.glmLevelMode === formulaAssistCategoricalMode;
          item.classList.toggle("active", active);
          item.setAttribute("aria-checked", active ? "true" : "false");
        });
        refreshFormulaAssistPreview();
      });
    });
    bindFormulaAssistLevelControls();
    el("glmFormulaAssistInsertBtn")?.addEventListener("click", insertSelectedSnippet);
    el("glmFormulaAssistInsertPiecewiseBtn")?.addEventListener("click", insertPiecewiseFormula);
    el("glmFormulaAssistInsertLevelsBtn")?.addEventListener("click", insertGroupedLevelsFormula);
  }

  function insertSelectedSnippet() {
    insertFormulaAtCursor(buildSnippetFormula(formulaAssistSelectedSnippet, formulaAssistSelectedFeature, {
      denominator: denominatorForOffsetSnippet(),
    }));
  }

  function insertPiecewiseFormula() {
    const parsed = parseBreakpoints(formulaAssistBreakpoints);
    if (parsed.error) return;
    insertFormulaAtCursor(buildPiecewiseFormula(formulaAssistSelectedFeature, parsed.values));
  }

  function insertGroupedLevelsFormula() {
    const values = selectedFormulaAssistLevelValues();
    if (!values.length) return;
    insertFormulaAtCursor(formulaAssistCategoricalMode === "ind"
      ? buildIndividualLevelsFormula(formulaAssistSelectedFeature, values)
      : buildGroupedLevelsFormula(formulaAssistSelectedFeature, values));
  }

  function insertFormulaAtCursor(text) {
    const value = String(text || "");
    if (!value) return;
    const output = withFormulaHeader(value, formulaAssistSelectedFeature, formulaAssistIncludeHeader);
    insertEditorText(formatDrawerInsertion(output, editorTextBeforeInsertion(), { replaceSelection: editorHasSelection() }));
  }

  function editorHasSelection() {
    if (aceEditor) {
      const range = aceEditor.getSelectionRange?.();
      if (!range?.start || !range?.end) return false;
      return range.start.row !== range.end.row || range.start.column !== range.end.column;
    }
    const fallback = el("glmFormulaText");
    if (!fallback) return false;
    const start = fallback.selectionStart ?? fallback.value.length;
    const end = fallback.selectionEnd ?? start;
    return end > start;
  }

  function editorTextBeforeInsertion() {
    if (aceEditor) {
      const range = aceEditor.getSelectionRange?.();
      const position = range?.start || aceEditor.getCursorPosition();
      const value = aceEditor.getValue();
      return textBeforeAcePosition(value, position);
    }
    const fallback = el("glmFormulaText");
    if (!fallback) return "";
    const start = fallback.selectionStart ?? fallback.value.length;
    return fallback.value.slice(0, start);
  }

  function textBeforeAcePosition(text, position = {}) {
    const row = Math.max(0, Number(position.row || 0));
    const column = Math.max(0, Number(position.column || 0));
    const lines = String(text || "").split("\n");
    return [...lines.slice(0, row), (lines[row] || "").slice(0, column)].join("\n");
  }

  function insertEditorText(text) {
    if (aceEditor) {
      aceEditor.focus();
      aceEditor.insert(String(text || ""));
      captureDraft();
      hideAutocomplete();
      return;
    }
    const fallback = el("glmFormulaText");
    if (!fallback) return;
    const start = fallback.selectionStart ?? fallback.value.length;
    const end = fallback.selectionEnd ?? start;
    fallback.value = `${fallback.value.slice(0, start)}${text}${fallback.value.slice(end)}`;
    fallback.selectionStart = fallback.selectionEnd = start + String(text).length;
    formulaDraft = fallback.value;
    localStorage.setItem("py_lucidum_glm_formula", formulaDraft);
  }

  function levelCacheKey(column, search) {
    return `${column}\u0000${String(search || "").toLowerCase()}`;
  }

  function levelCacheEntry(column, search) {
    return formulaAssistLevelCache.get(levelCacheKey(column, search));
  }

  function formulaAssistLevelValueKey(value) {
    return JSON.stringify(value ?? "");
  }

  function selectedFormulaAssistLevelValues() {
    return [...formulaAssistSelectedLevelKeys].map((key) => {
      try {
        return JSON.parse(key);
      } catch (_) {
        return key;
      }
    });
  }

  async function loadFormulaAssistLevels({ force = false } = {}) {
    if (!api || !formulaAssistSelectedFeature) return null;
    const key = levelCacheKey(formulaAssistSelectedFeature, formulaAssistLevelSearch);
    if (!force && formulaAssistLevelCache.has(key)) {
      const payload = formulaAssistLevelCache.get(key);
      refreshFormulaAssistLevelRows(payload?.values || []);
      return payload;
    }
    const requestSeq = formulaAssistLevelRequestSeq + 1;
    formulaAssistLevelRequestSeq = requestSeq;
    try {
      const payload = await api("/api/glm/formula/levels", {
        method: "POST",
        body: JSON.stringify({ column: formulaAssistSelectedFeature, search: formulaAssistLevelSearch, limit: 500 }),
      });
      if (requestSeq !== formulaAssistLevelRequestSeq) return null;
      formulaAssistLevelCache.set(key, payload);
      refreshFormulaAssistLevelRows(payload.values || []);
      return payload;
    } catch (_) {
      if (requestSeq === formulaAssistLevelRequestSeq) {
        formulaAssistLevelCache.set(key, { values: [] });
        refreshFormulaAssistLevelRows([]);
      }
      return null;
    }
  }

  function bindControls() {
    syncFamilyParameterControl();
    syncRegularizationControls();
    renderFormulaAssistDrawer();
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
        document.querySelectorAll("[data-glm-scope]").forEach((item) => {
          const active = item === button;
          item.classList.toggle("active", active);
          item.setAttribute("aria-pressed", active ? "true" : "false");
        });
      });
    });
    el("glmClearFormulaBtn")?.addEventListener("click", () => setFormulaText(""));
    el("glmFormulaAssistBtn")?.addEventListener("click", () => toggleBuilderPanel("formula"));
    el("glmModelParametersBtn")?.addEventListener("click", () => toggleBuilderPanel("parameters"));
    el("glmFontSmallerBtn")?.addEventListener("click", () => adjustFontSize(-1));
    el("glmFontLargerBtn")?.addEventListener("click", () => adjustFontSize(1));
    el("glmCopyFormulaBtn")?.addEventListener("click", () => onCopyFormula(getFormulaText()));
    el("glmBuildBtn")?.addEventListener("click", onBuildModel);
    el("glmCopyCoefficientsBtn")?.addEventListener("click", onCopyCoefficients);
    el("glmDownloadCoefficientsBtn")?.addEventListener("click", onDownloadCoefficients);
    el("glmCoefficientSearch")?.addEventListener("input", onCoefficientSearch);
  }

  function savedSplitWidthStyle() {
    return Number.isFinite(splitPanelWidth) && splitPanelWidth > 0 ? `--glm-formula-panel-width: ${Math.round(splitPanelWidth)}px;` : "";
  }

  function bindResizer() {
    const layout = document.querySelector(".glm-builder-layout");
    const resizer = el("glmBuilderResizer");
    if (!layout || !resizer) return;

    const resizeTo = (width) => {
      const layoutRect = layout.getBoundingClientRect();
      const resizerWidth = resizer.getBoundingClientRect().width || 0;
      const minLeft = 320;
      const minRight = 360;
      const maxLeft = Math.max(minLeft, layoutRect.width - resizerWidth - minRight);
      const clamped = Math.max(minLeft, Math.min(maxLeft, width));
      splitPanelWidth = clamped;
      layout.style.setProperty("--glm-formula-panel-width", `${Math.round(clamped)}px`);
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

  function bindAutocomplete() {
    if (!aceEditor) return;
    aceEditor.commands.addCommand({
      name: "glmFormulaAutocomplete",
      bindKey: { win: "Ctrl-Space", mac: "Command-Space|Ctrl-Space" },
      exec: () => showAutocomplete({ manual: true }),
    });
    aceEditor.container.addEventListener("keydown", handleAutocompleteKeydown, true);
    aceEditor.session.on("change", () => window.setTimeout(() => showAutocomplete({ manual: false }), 0));
    aceEditor.selection.on("changeCursor", () => hideAutocomplete());
    document.addEventListener("mousedown", handleAutocompleteDocumentMouseDown);
  }

  function handleAutocompleteDocumentMouseDown(event) {
    if (autocompletePopup && !autocompletePopup.contains(event.target) && !aceEditor?.container.contains(event.target)) hideAutocomplete();
  }

  function handleAutocompleteKeydown(event) {
    if (!autocompletePopup) return;
    if (event.key === "ArrowDown") {
      event.preventDefault();
      event.stopPropagation();
      moveAutocompleteSelection(1);
    } else if (event.key === "ArrowUp") {
      event.preventDefault();
      event.stopPropagation();
      moveAutocompleteSelection(-1);
    } else if (event.key === "Enter" || event.key === "Tab") {
      event.preventDefault();
      event.stopPropagation();
      insertAutocompleteItem(autocompleteItems[autocompleteIndex]);
    } else if (event.key === "Escape") {
      event.preventDefault();
      event.stopPropagation();
      hideAutocomplete();
    }
  }

  async function showAutocomplete({ manual = false } = {}) {
    if (!aceEditor) return;
    const cursor = aceEditor.getCursorPosition();
    const context = formulaCompletionContext(aceEditor.getValue(), cursor.row, cursor.column);
    if (context.type === "none") {
      hideAutocomplete();
      return;
    }
    if (!manual && context.type === "formula" && context.prefix.length < 1) {
      hideAutocomplete();
      return;
    }
    autocompleteContext = context;
    let items = [];
    if (context.type === "levels") {
      items = await autocompleteLevelItems(context);
    } else {
      items = rankFormulaSuggestions([
        ...formulaColumnSuggestions(formulaColumns()),
        ...formulaFunctionSuggestions(),
      ], context.prefix).slice(0, 80);
    }
    if (!items.length) {
      hideAutocomplete();
      return;
    }
    autocompleteItems = items;
    autocompleteIndex = 0;
    renderAutocompletePopup();
  }

  async function autocompleteLevelItems(context) {
    if (!api || !context.feature) return [];
    const search = context.prefix || "";
    const key = levelCacheKey(context.feature, search);
    let payload = formulaAssistLevelCache.get(key);
    if (!payload) {
      try {
        payload = await api("/api/glm/formula/levels", {
          method: "POST",
          body: JSON.stringify({ column: context.feature, search, limit: 80 }),
        });
        formulaAssistLevelCache.set(key, payload);
      } catch (_) {
        return [];
      }
    }
    return rankFormulaSuggestions((payload.values || []).map((row) => ({
      type: "level",
      caption: String(row.label || row.value || ""),
      value: formulaLiteral(row.value),
      meta: `${Number(row.count || 0).toLocaleString()} rows`,
    })), search).slice(0, 80);
  }

  function renderAutocompletePopup() {
    const shell = document.querySelector(".glm-editor-shell");
    if (!shell || !aceEditor) return;
    if (!autocompletePopup) {
      autocompletePopup = document.createElement("div");
      autocompletePopup.className = "glm-formula-autocomplete";
      shell.append(autocompletePopup);
    }
    autocompletePopup.innerHTML = autocompleteItems.map((item, index) => `
      <button type="button" class="glm-formula-autocomplete-row ${index === autocompleteIndex ? "active" : ""}" data-glm-autocomplete-index="${index}">
        <span>${escapeHtml(item.caption || item.value || "")}</span>
        <small>${escapeHtml(item.meta || "")}</small>
      </button>
    `).join("");
    autocompletePopup.querySelectorAll("[data-glm-autocomplete-index]").forEach((button) => {
      button.addEventListener("mousedown", (event) => {
        event.preventDefault();
        insertAutocompleteItem(autocompleteItems[Number(button.dataset.glmAutocompleteIndex || 0)]);
      });
    });
    positionAutocompletePopup();
  }

  function positionAutocompletePopup() {
    if (!autocompletePopup || !aceEditor) return;
    const shell = document.querySelector(".glm-editor-shell");
    if (!shell) return;
    const cursor = aceEditor.getCursorPosition();
    const coords = aceEditor.renderer.textToScreenCoordinates(cursor.row, cursor.column);
    const shellRect = shell.getBoundingClientRect();
    const left = Math.max(4, Math.min(shellRect.width - 260, coords.pageX - shellRect.left));
    const top = Math.max(4, Math.min(shellRect.height - 220, coords.pageY - shellRect.top + 18));
    autocompletePopup.style.left = `${left}px`;
    autocompletePopup.style.top = `${top}px`;
  }

  function moveAutocompleteSelection(delta) {
    if (!autocompleteItems.length) return;
    autocompleteIndex = (autocompleteIndex + delta + autocompleteItems.length) % autocompleteItems.length;
    renderAutocompletePopup();
  }

  function insertAutocompleteItem(item) {
    if (!item || !aceEditor || !autocompleteContext) return;
    const ace = window.ace;
    const cursor = aceEditor.getCursorPosition();
    const Range = ace?.Range || ace?.require?.("ace/range")?.Range;
    const startColumn = Math.max(0, Number(autocompleteContext.replaceStartColumn ?? cursor.column));
    const selectionRange = aceEditor.getSelectionRange?.();
    if (selectionRange && !selectionRange.isEmpty?.()) {
      aceEditor.session.replace(selectionRange, item.value || item.caption || "");
    } else if (Range) {
      aceEditor.session.replace(new Range(cursor.row, startColumn, cursor.row, cursor.column), item.value || item.caption || "");
    } else {
      aceEditor.insert(item.value || item.caption || "");
    }
    captureDraft();
    hideAutocomplete();
    aceEditor.focus();
  }

  function hideAutocomplete() {
    if (autocompletePopup) autocompletePopup.remove();
    autocompletePopup = null;
    autocompleteItems = [];
    autocompleteContext = null;
    autocompleteIndex = 0;
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

  function currentAceTheme() {
    return document.body.classList.contains("dark") ? "ace/theme/monokai" : "ace/theme/textmate";
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
      aceEditor.setTheme(currentAceTheme());
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
      bindAutocomplete();
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
    localStorage.setItem("py_lucidum_glm_formula", formulaDraft);
    hideAutocomplete();
    aceEditor.container.removeEventListener("keydown", handleAutocompleteKeydown, true);
    document.removeEventListener("mousedown", handleAutocompleteDocumentMouseDown);
    try {
      aceEditor.destroy();
    } catch (_) {
    }
    aceEditor = null;
    editorInitialisedFor = null;
  }

  function getFormulaText() {
    captureDraft();
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

  function captureDraft() {
    if (aceEditor) formulaDraft = aceEditor.getValue();
    else if (el("glmFormulaText")) formulaDraft = el("glmFormulaText").value;
    localStorage.setItem("py_lucidum_glm_formula", formulaDraft);

    const familySelect = el("glmFamilySelect");
    if (familySelect?.value) {
      selectedFamily = familySelect.value;
      localStorage.setItem("py_lucidum_glm_family", selectedFamily);
    }
    const familyParameter = el("glmFamilyParameter");
    if (familyParameter && familyParameterConfig(selectedFamily)) {
      localStorage.setItem(`py_lucidum_glm_family_parameter_${selectedFamily}`, familyParameter.value.trim());
    }

    const activeScope = document.querySelector("[data-glm-scope].active")?.dataset?.glmScope;
    if (activeScope === "all" || activeScope === "training") {
      selectedTrainingScope = activeScope;
      localStorage.setItem("py_lucidum_glm_training_scope", selectedTrainingScope);
    }

    const regularizationMode = el("glmRegularizationMode");
    if (regularizationMode?.value) {
      selectedRegularizationMode = regularizationMode.value || "none";
      localStorage.setItem("py_lucidum_glm_regularization_mode", selectedRegularizationMode);
    }
    const regularizationMix = el("glmRegularizationMix");
    if (regularizationMix?.value) {
      selectedRegularizationMix = regularizationMix.value || "0.5";
      localStorage.setItem("py_lucidum_glm_regularization_mix", selectedRegularizationMix);
    }
    const regularizationAlpha = el("glmRegularizationAlpha");
    if (regularizationAlpha) {
      selectedRegularizationAlpha = regularizationAlpha.value.trim();
      localStorage.setItem("py_lucidum_glm_regularization_alpha", selectedRegularizationAlpha);
    }
  }

  function buildPayload({ actual = "", denominator = "__none__", label = "" } = {}) {
    captureDraft();
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
      label,
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

  function syncFromModelDetail(detail = {}, options = {}) {
    if (options.syncBuilderDraft === false) return;
    const manifest = detail?.manifest || {};
    const rawFormula = detail?.formula;
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
        const active = button.dataset.glmScope === selectedTrainingScope;
        button.classList.toggle("active", active);
        button.setAttribute("aria-pressed", active ? "true" : "false");
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

  function refreshTheme() {
    if (aceEditor) {
      aceEditor.setTheme(currentAceTheme());
      syncAceGutterWidth();
    }
  }

  function resize() {
    if (aceEditor) aceEditor.resize();
  }

  return {
    bindControls,
    bindResizer,
    buildPayload,
    buildRegularizationPayload,
    captureDraft,
    disposeEditor,
    ensureTrainingScope,
    familyOptionsHtml,
    formulaAssistDrawerHtml,
    familyParameterConfig,
    familyParameterDefault,
    getFormulaText,
    initEditor,
    refreshTheme,
    regularizationMixOptionsHtml,
    regularizationModeOptionsHtml,
    resize,
    savedSplitWidthStyle,
    setFormulaText,
    syncFamilyParameterControl,
    syncFromModelDetail,
    syncRegularizationControls,
    validateFamilyParameter,
    validateRegularizationParameter,
    get formulaDraft() {
      return formulaDraft;
    },
    get formulaAssistOpen() {
      return formulaAssistOpen;
    },
    get parametersOpen() {
      return builderPanel === "parameters";
    },
    get selectedRegularizationAlpha() {
      return selectedRegularizationAlpha;
    },
    get selectedRegularizationMix() {
      return selectedRegularizationMix;
    },
    get selectedRegularizationMode() {
      return selectedRegularizationMode;
    },
    get selectedTrainingScope() {
      return selectedTrainingScope;
    },
  };
}
