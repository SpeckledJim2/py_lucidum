export const GLM_TABULATION_MODEL_CROSSTAB = "__model__";
const GLM_TABULATION_Y_AXIS_TARGET_INTERVALS = 15;
const GLM_TABULATION_X_AXIS_LABEL_DENSITY_LIMIT = 200;
const GLM_TABULATION_X_AXIS_ZOOM_THRESHOLD = 120;
const GLM_TABULATION_X_AXIS_LABEL_PADDING = 8;

export function createGlmTabulations({ el, modelNumberOrNull, scheduleResize }) {
  let splitSidebarWidth = null;
  let modelListHeight = null;

  function savedSplitWidthStyle() {
    return Number.isFinite(splitSidebarWidth) && splitSidebarWidth > 0 ? `--glm-tabulation-sidebar-width: ${Math.round(splitSidebarWidth)}px;` : "";
  }

  function bindResizer() {
    const layout = document.querySelector(".glm-tabulation-layout");
    const resizer = el("glmTabulationResizer");
    if (!layout || !resizer) return;

    const resizeTo = (width) => {
      const layoutRect = layout.getBoundingClientRect();
      const resizerWidth = resizer.getBoundingClientRect().width || 0;
      const minLeft = 420;
      const minRight = 420;
      const maxLeft = Math.max(minLeft, layoutRect.width - resizerWidth - minRight);
      const clamped = Math.max(minLeft, Math.min(maxLeft, width));
      splitSidebarWidth = clamped;
      layout.style.setProperty("--glm-tabulation-sidebar-width", `${Math.round(clamped)}px`);
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

    const resizeTo = (height) => {
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
      modelListHeight = clamped;
      sidebar.style.setProperty("--glm-tabulation-model-list-height", `${Math.round(clamped)}px`);
      resizer.setAttribute("aria-valuemin", String(minModelHeight));
      resizer.setAttribute("aria-valuemax", String(Math.round(maxHeight)));
      resizer.setAttribute("aria-valuenow", String(Math.round(clamped)));
      scheduleResize();
    };

    if (Number.isFinite(modelListHeight) && modelListHeight > 0) {
      resizeTo(modelListHeight);
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
    if (data.scale === "exp") {
      min = Math.min(min, 1);
      max = Math.max(max, 1);
    }
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

  function xAxisFeature(data = {}) {
    const features = Array.isArray(data.features)
      ? data.features.map((feature) => String(feature || "")).filter(Boolean)
      : [];
    if (features.length === 1) return features[0];
    const crosstab = String(data.crosstab || "");
    if (features.length === 2 && features.includes(crosstab)) {
      return features.find((feature) => feature !== crosstab) || "";
    }
    return "";
  }

  function xAxisLabelPolicy(labels = [], chartWidth = 0) {
    const count = labels.length;
    const dataZoomEnabled = count > GLM_TABULATION_X_AXIS_ZOOM_THRESHOLD;
    const dataZoomSpace = dataZoomEnabled ? 36 : 0;
    if (count >= GLM_TABULATION_X_AXIS_LABEL_DENSITY_LIMIT) {
      return {
        show: false,
        interval: 0,
        rotate: 0,
        fontSize: 10,
        nameGap: 22,
        bottom: 38 + dataZoomSpace,
        dataZoomEnabled,
      };
    }
    const fontSize = count > 50 ? 8 : 10;
    const maxLength = labels.reduce((longest, label) => Math.max(longest, String(label ?? "").length), 0);
    const estimatedTextWidth = maxLength * fontSize * 0.5;
    const width = Number(chartWidth);
    const plotWidth = Math.max(120, (Number.isFinite(width) && width > 0 ? width : 900) - 128);
    const slotWidth = plotWidth / Math.max(1, count);
    const horizontalFootprint = estimatedTextWidth + GLM_TABULATION_X_AXIS_LABEL_PADDING;
    const rotate = count > 30 || maxLength > 10 || horizontalFootprint > slotWidth ? 65 : 0;
    const radians = (rotate * Math.PI) / 180;
    const rotatedHeight = estimatedTextWidth * Math.sin(radians) + fontSize * Math.cos(radians);
    const labelSpace = rotate ? Math.min(140, Math.max(58, Math.ceil(rotatedHeight) + 18)) : 38;
    const nameGap = rotate ? Math.max(26, labelSpace - 10) : 26;
    return {
      show: count > 0,
      interval: 0,
      rotate,
      fontSize,
      nameGap,
      bottom: nameGap + 16 + dataZoomSpace,
      dataZoomEnabled,
    };
  }

  function xAxisPresentation(data = {}, chartWidth = 0, theme = {}) {
    const values = Array.isArray(data.x_axis) ? data.x_axis : [];
    const labels = values.map((value) => String(value ?? ""));
    const policy = xAxisLabelPolicy(labels, chartWidth);
    return {
      grid: { bottom: policy.bottom },
      xAxis: {
        type: "category",
        data: values,
        name: xAxisFeature(data),
        nameLocation: "middle",
        nameGap: policy.nameGap,
        nameTextStyle: { color: theme.text || "#334155", fontSize: 13, fontWeight: 700 },
        axisLine: {
          onZero: true,
          lineStyle: { color: theme.line || "#cbd5e1", width: 2 },
        },
        axisLabel: {
          show: policy.show,
          color: theme.text || "#334155",
          interval: policy.interval,
          hideOverlap: false,
          showMinLabel: true,
          showMaxLabel: true,
          rotate: policy.rotate,
          fontSize: policy.fontSize,
          margin: 8,
        },
      },
      dataZoom: policy.dataZoomEnabled
        ? [{ type: "inside" }, { type: "slider", height: 18, bottom: 18 }]
        : [],
    };
  }

  function baselineMarkLine(data = {}, theme = {}) {
    if (data.scale !== "exp") return null;
    return {
      silent: true,
      symbol: "none",
      label: { show: false },
      lineStyle: { color: theme.line || "#cbd5e1", type: "solid", width: 2 },
      data: [{ yAxis: 1 }],
    };
  }

  return {
    baselineMarkLine,
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
    xAxisPresentation,
    yAxisOptions,
  };
}
