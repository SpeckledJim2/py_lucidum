const ACE_BASE_PATH = "/static/vendor/ace";
const GLM_BUILDER_SPLIT_STORAGE_KEY = "py_lucidum_glm_formula_panel_width";

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
  el,
  escapeHtml,
  getFamilies = () => [],
  onBuildModel = () => {},
  onCoefficientSearch = () => {},
  onCopyCoefficients = () => {},
  onDownloadCoefficients = () => {},
}) {
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

  function bindControls() {
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
    el("glmBuildBtn")?.addEventListener("click", onBuildModel);
    el("glmCopyCoefficientsBtn")?.addEventListener("click", onCopyCoefficients);
    el("glmDownloadCoefficientsBtn")?.addEventListener("click", onDownloadCoefficients);
    el("glmCoefficientSearch")?.addEventListener("input", onCoefficientSearch);
  }

  function savedSplitWidthStyle() {
    const width = Number(localStorage.getItem(GLM_BUILDER_SPLIT_STORAGE_KEY));
    return Number.isFinite(width) && width > 0 ? `--glm-formula-panel-width: ${Math.round(width)}px;` : "";
  }

  function bindResizer() {
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
    localStorage.setItem("py_lucidum_glm_formula", formulaDraft);
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

  function refreshTheme() {
    if (aceEditor) {
      aceEditor.setTheme(document.body.classList.contains("dark") ? "ace/theme/monokai" : "ace/theme/textmate");
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
