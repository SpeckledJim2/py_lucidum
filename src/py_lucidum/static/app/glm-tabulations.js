export const GLM_TABULATION_MODEL_CROSSTAB = "__model__";
const GLM_TABULATION_SPLIT_STORAGE_KEY = "py_lucidum_glm_tabulation_sidebar_width_v2";
const GLM_TABULATION_MODEL_LIST_HEIGHT_KEY = "py_lucidum_glm_tabulation_model_list_height";
const GLM_TABULATION_Y_AXIS_TARGET_INTERVALS = 15;

export function createGlmTabulations({ el, modelNumberOrNull, scheduleResize }) {
  function savedSplitWidthStyle() {
    const width = Number(localStorage.getItem(GLM_TABULATION_SPLIT_STORAGE_KEY));
    return Number.isFinite(width) && width > 0 ? `--glm-tabulation-sidebar-width: ${Math.round(width)}px;` : "";
  }

  function bindResizer() {
    const layout = document.querySelector(".glm-tabulation-layout");
    const resizer = el("glmTabulationResizer");
    if (!layout || !resizer) return;

    const resizeTo = (width, persist = true) => {
      const layoutRect = layout.getBoundingClientRect();
      const resizerWidth = resizer.getBoundingClientRect().width || 0;
      const minLeft = 420;
      const minRight = 420;
      const maxLeft = Math.max(minLeft, layoutRect.width - resizerWidth - minRight);
      const clamped = Math.max(minLeft, Math.min(maxLeft, width));
      layout.style.setProperty("--glm-tabulation-sidebar-width", `${Math.round(clamped)}px`);
      if (persist) localStorage.setItem(GLM_TABULATION_SPLIT_STORAGE_KEY, String(Math.round(clamped)));
      scheduleResize();
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

    bindSelectorResizer(layout);
  }

  function bindSelectorResizer(layout) {
    const sidebar = layout.querySelector(".glm-tabulation-sidebar");
    const modelRegion = layout.querySelector(".glm-tabulation-model-region");
    const resizer = el("glmTabulationSelectorResizer");
    if (!sidebar || !modelRegion || !resizer || resizer.dataset.bound === "true") return;
    resizer.dataset.bound = "true";

    const resizeTo = (height, persist = true) => {
      const minModelHeight = 96;
      const minTableHeight = 120;
      const fixedNodes = [
        sidebar.querySelector(".glm-panel-header"),
        el("glmTabulationModelLabel"),
        el("glmTabulationTableLabel"),
        resizer,
        el("glmTabulationDiagnostics"),
      ];
      const fixedHeight = fixedNodes.reduce((total, node) => {
        if (!node || node.classList?.contains("hidden")) return total;
        return total + (node.getBoundingClientRect().height || 0);
      }, 48);
      const availableHeight = sidebar.getBoundingClientRect().height || window.innerHeight;
      const maxHeight = Math.max(minModelHeight, availableHeight - fixedHeight - minTableHeight);
      const clamped = Math.max(minModelHeight, Math.min(maxHeight, height));
      sidebar.style.setProperty("--glm-tabulation-model-list-height", `${Math.round(clamped)}px`);
      resizer.setAttribute("aria-valuemin", String(minModelHeight));
      resizer.setAttribute("aria-valuemax", String(Math.round(maxHeight)));
      resizer.setAttribute("aria-valuenow", String(Math.round(clamped)));
      if (persist) localStorage.setItem(GLM_TABULATION_MODEL_LIST_HEIGHT_KEY, String(Math.round(clamped)));
      scheduleResize();
    };

    const savedHeight = Number(localStorage.getItem(GLM_TABULATION_MODEL_LIST_HEIGHT_KEY));
    if (Number.isFinite(savedHeight) && savedHeight > 0) {
      resizeTo(savedHeight, false);
    }

    resizer.addEventListener("pointerdown", (event) => {
      event.preventDefault();
      const startY = event.clientY;
      const startHeight = modelRegion.getBoundingClientRect().height || 0;
      resizer.classList.add("dragging");
      document.body.classList.add("resizing-chart-control-heights");
      resizer.setPointerCapture?.(event.pointerId);
      window.getSelection()?.removeAllRanges();
      const onMove = (moveEvent) => resizeTo(startHeight + moveEvent.clientY - startY);
      const onUp = () => {
        resizer.classList.remove("dragging");
        document.body.classList.remove("resizing-chart-control-heights");
        window.removeEventListener("pointermove", onMove);
        window.removeEventListener("pointerup", onUp);
        window.getSelection()?.removeAllRanges();
      };
      window.addEventListener("pointermove", onMove);
      window.addEventListener("pointerup", onUp, { once: true });
    });

    resizer.addEventListener("keydown", (event) => {
      if (!["ArrowUp", "ArrowDown"].includes(event.key)) return;
      event.preventDefault();
      const current = modelRegion.getBoundingClientRect().height || 0;
      resizeTo(current + (event.key === "ArrowDown" ? 24 : -24));
    });
  }

  function normaliseRef(value) {
    const text = String(value || "").trim();
    if (!text) return "";
    if (text.startsWith("glm:") || text.startsWith("gbm:")) {
      const parts = text.split(":");
      return parts.length >= 2 ? `${parts[0]}:${parts[1]}` : text;
    }
    return `glm:${text}`;
  }

  function modelRef(model = {}) {
    const explicit = String(model.model_ref || "").trim();
    if (explicit) return normaliseRef(explicit);
    const kind = String(model.model_kind || "glm").toLowerCase() === "gbm" ? "gbm" : "glm";
    const modelId = String(model.model_id || "").trim();
    return modelId ? `${kind}:${modelId}` : "";
  }

  function tableSelectorShellHtml(selectedIds = []) {
    if (selectedIds.length > 1) {
      return `
        <section class="glm-tabulation-table-section">
          <div class="glm-tabulation-section-title">Common tables</div>
          <div id="glmTabulationCommonTableGrid" class="glm-grid glm-tabulation-selector-grid glm-tabulation-table-list"></div>
          <div id="glmTabulationCommonTableFallback" class="glm-tabulation-selector-fallback"></div>
        </section>
        <section class="glm-tabulation-table-section">
          <div class="glm-tabulation-section-title">Other tables</div>
          <div id="glmTabulationOtherTableGrid" class="glm-grid glm-tabulation-selector-grid glm-tabulation-table-list"></div>
          <div id="glmTabulationOtherTableFallback" class="glm-tabulation-selector-fallback"></div>
        </section>
      `;
    }
    return `
      <div id="glmTabulationTableGrid" class="glm-grid glm-tabulation-selector-grid glm-tabulation-table-list"></div>
      <div id="glmTabulationTableFallback" class="glm-tabulation-selector-fallback"></div>
    `;
  }

  function crosstabOptions(features = [], modelIds = []) {
    const options = [{ value: "", label: "No crosstab" }];
    if (modelIds.length > 1) options.push({ value: GLM_TABULATION_MODEL_CROSSTAB, label: "Model" });
    features.forEach((feature) => options.push({ value: feature, label: feature }));
    return options;
  }

  function displayTableValue(value, scale = "linear") {
    const number = modelNumberOrNull(value);
    if (number === null) return null;
    return scale === "exp" ? Math.exp(number) : number;
  }

  function displayTableSpan(min, max, scale = "linear") {
    const lo = modelNumberOrNull(min);
    const hi = modelNumberOrNull(max);
    if (lo === null || hi === null) return null;
    return scale === "exp" ? Math.exp(hi - lo) : hi - lo;
  }

  function niceAxisStep(span) {
    if (!Number.isFinite(span) || span <= 0) return 1;
    const roughStep = span / GLM_TABULATION_Y_AXIS_TARGET_INTERVALS;
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

  function formatAxisTick(value, scale = "linear") {
    const number = Number(value);
    if (!Number.isFinite(number)) return "";
    if (scale === "exp") return formatUpliftPercent(number);
    const formatted = number.toLocaleString(undefined, { maximumFractionDigits: 6 });
    return /^-0(?:[.,]0+)?$/.test(formatted) ? formatted.slice(1) : formatted;
  }

  function yAxisOptions(data = {}) {
    const name = data.scale === "exp" ? "exp(tabulated)" : "tabulated";
    let min = Number(data.min);
    let max = Number(data.max);
    if (!Number.isFinite(min) || !Number.isFinite(max)) {
      return {
        type: "value",
        name,
        scale: true,
        splitNumber: GLM_TABULATION_Y_AXIS_TARGET_INTERVALS,
        axisLabel: { formatter: (value) => formatAxisTick(value, data.scale) },
      };
    }
    if (max < min) [min, max] = [max, min];
    const dataMin = min;
    const dataMax = max;
    if (min === max) {
      const pad = data.scale === "exp"
        ? Math.max(Math.abs(min) * 0.02, 0.01)
        : Math.max(Math.abs(min) * 0.02, 0.1);
      min -= pad;
      max += pad;
    }
    const span = Math.max(max - min, Number.EPSILON);
    const paddedMin = min - span * 0.06;
    const paddedMax = max + span * 0.06;
    const step = niceAxisStep(paddedMax - paddedMin);
    let axisMin = Math.floor(paddedMin / step) * step;
    let axisMax = Math.ceil(paddedMax / step) * step;
    if (data.scale === "exp") {
      axisMin = Math.floor(dataMin / step) * step;
      if (axisMin > min) axisMin = Math.max(step, axisMin - step);
      if (axisMin <= 0) axisMin = Math.max(step, Math.floor(dataMin / step) * step);
    } else if (dataMin > 0 && axisMin <= 0) {
      axisMin = Math.floor(dataMin / step) * step;
      if (axisMin <= 0) axisMin = step;
    } else if (dataMax < 0 && axisMax >= 0) {
      axisMax = Math.ceil(dataMax / step) * step;
      if (axisMax >= 0) axisMax = dataMax;
    }
    return {
      type: "value",
      name,
      scale: true,
      splitNumber: GLM_TABULATION_Y_AXIS_TARGET_INTERVALS,
      min: roundAxisValue(axisMin, step),
      max: roundAxisValue(axisMax, step),
      interval: roundAxisValue(step, step),
      axisLabel: { formatter: (value) => formatAxisTick(value, data.scale) },
    };
  }

  return {
    bindResizer,
    crosstabOptions,
    displayTableSpan,
    displayTableValue,
    formatAxisTick,
    formatUpliftPercent,
    modelRef,
    niceAxisStep,
    normaliseRef,
    roundAxisValue,
    savedSplitWidthStyle,
    tableSelectorShellHtml,
    yAxisOptions,
  };
}
