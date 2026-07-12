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
  data_sample_strategy: [
    "bagging",
    "goss",
  ],
};

const GBM_FEATURE_MIN_WIDTH = 360;
const GBM_PARAMETER_MIN_WIDTH = 240;
const GBM_CONTROL_MIN_WIDTH = 162;
const GBM_PANE_MIN_HEIGHT = 120;
const GBM_DIVIDER_TRACK_WIDTH = 1;
const GBM_CONTROL_STRIP_HEIGHT = 50;
const GBM_DEFAULT_UPPER_BOUNDARY = 330;
const GBM_RESIZE_STEP = 24;

export function createGbmFeatureParameterLayout({ onResize = () => {} } = {}) {
  let parameterWidth = null;
  let upperHeight = null;
  let resizeObserver = null;
  let fallbackResizeHandler = null;
  let resizeFrame = null;
  let observedSize = null;

  function bind(root = document) {
    disposeBindings();
    const workspace = root.querySelector?.("#gbmFeatureWorkspace");
    const featureResizer = root.querySelector?.("#gbmFeatureResizer");
    const evaluationResizer = root.querySelector?.("#gbmEvaluationResizer");
    if (!workspace || !featureResizer || !evaluationResizer) return;
    if (workspace.closest("[data-gbm-panel]")?.classList.contains("hidden")) return;

    applySavedLayout(workspace);
    bindAxisResizer({
      resizer: featureResizer,
      axis: "x",
      bodyClass: "gbm-feature-column-resizing",
      currentValue: () => paneWidth(workspace, ".gbm-grid-panel"),
      resizeTo: (value) => resizeFeature(workspace, value),
    });
    bindAxisResizer({
      resizer: evaluationResizer,
      axis: "y",
      bodyClass: "gbm-feature-row-resizing",
      currentValue: () => paneHeight(workspace, ".gbm-parameter-section"),
      resizeTo: (value) => resizeUpper(workspace, value),
    });

    observedSize = workspaceSize(workspace);
    const handleContainerResize = () => {
      const nextSize = workspaceSize(workspace);
      if (nextSize.width === observedSize?.width && nextSize.height === observedSize?.height) return;
      observedSize = nextSize;
      applySavedLayout(workspace);
    };
    if (window.ResizeObserver) {
      resizeObserver = new ResizeObserver(handleContainerResize);
      resizeObserver.observe(workspace);
    } else {
      fallbackResizeHandler = handleContainerResize;
      window.addEventListener("resize", fallbackResizeHandler);
    }
  }

  function bindAxisResizer({ resizer, axis, bodyClass, currentValue, resizeTo }) {
    let pointerId = null;
    let startPoint = 0;
    let startValue = 0;
    const coordinate = (event) => axis === "x" ? event.clientX : event.clientY;

    resizer.addEventListener("pointerdown", (event) => {
      if (event.button !== undefined && event.button !== 0) return;
      event.preventDefault();
      pointerId = event.pointerId;
      startPoint = coordinate(event);
      startValue = currentValue();
      resizer.classList.add("dragging");
      document.body.classList.add(bodyClass);
      resizer.setPointerCapture?.(event.pointerId);
      window.getSelection()?.removeAllRanges();
    });
    resizer.addEventListener("pointermove", (event) => {
      if (pointerId !== event.pointerId) return;
      resizeTo(startValue + coordinate(event) - startPoint);
    });
    const finishDrag = (event) => {
      if (pointerId !== event.pointerId) return;
      pointerId = null;
      resizer.classList.remove("dragging");
      document.body.classList.remove(bodyClass);
      if (resizer.hasPointerCapture?.(event.pointerId)) resizer.releasePointerCapture(event.pointerId);
      window.getSelection()?.removeAllRanges();
    };
    resizer.addEventListener("pointerup", finishDrag);
    resizer.addEventListener("pointercancel", finishDrag);
    resizer.addEventListener("keydown", (event) => {
      const negativeKey = axis === "x" ? "ArrowLeft" : "ArrowUp";
      const positiveKey = axis === "x" ? "ArrowRight" : "ArrowDown";
      if (![negativeKey, positiveKey].includes(event.key)) return;
      event.preventDefault();
      resizeTo(currentValue() + (event.key === positiveKey ? GBM_RESIZE_STEP : -GBM_RESIZE_STEP));
    });
  }

  function applySavedLayout(workspace) {
    const contentWidth = availableColumnWidth(workspace);
    const minimumContentWidth = GBM_FEATURE_MIN_WIDTH + GBM_PARAMETER_MIN_WIDTH + GBM_CONTROL_MIN_WIDTH;
    if (contentWidth >= minimumContentWidth) {
      const desiredParameter = parameterWidth ?? paneWidth(workspace, ".gbm-parameter-section");
      const maxParameter = Math.max(
        GBM_PARAMETER_MIN_WIDTH,
        contentWidth - GBM_FEATURE_MIN_WIDTH - GBM_CONTROL_MIN_WIDTH,
      );
      parameterWidth = clamp(desiredParameter, GBM_PARAMETER_MIN_WIDTH, maxParameter);
      workspace.style.setProperty("--gbm-parameter-pane-width", `${Math.round(parameterWidth)}px`);
    } else {
      if (parameterWidth !== null) {
        workspace.style.setProperty("--gbm-parameter-pane-width", `${Math.round(parameterWidth)}px`);
      }
    }

    const desiredUpper = upperHeight ?? Math.max(
      GBM_PANE_MIN_HEIGHT,
      GBM_DEFAULT_UPPER_BOUNDARY - GBM_CONTROL_STRIP_HEIGHT,
    );
    if (upperHeight === null) upperHeight = desiredUpper;
    const appliedUpperHeight = clampUpperHeight(workspace, desiredUpper);
    workspace.style.setProperty("--gbm-parameter-pane-height", `${Math.round(appliedUpperHeight)}px`);
    syncAria(workspace);
    scheduleResize();
  }

  function resizeFeature(workspace, width) {
    const currentFeature = paneWidth(workspace, ".gbm-grid-panel");
    const currentParameter = paneWidth(workspace, ".gbm-parameter-section");
    const combinedWidth = currentFeature + currentParameter;
    const maxWidth = Math.max(
      GBM_FEATURE_MIN_WIDTH,
      combinedWidth - GBM_PARAMETER_MIN_WIDTH,
    );
    const featureWidth = clamp(width, GBM_FEATURE_MIN_WIDTH, maxWidth);
    parameterWidth = combinedWidth - featureWidth;
    workspace.style.setProperty("--gbm-parameter-pane-width", `${Math.round(parameterWidth)}px`);
    syncAria(workspace);
    scheduleResize();
  }

  function resizeUpper(workspace, height) {
    upperHeight = clampUpperHeight(workspace, height);
    workspace.style.setProperty("--gbm-parameter-pane-height", `${Math.round(upperHeight)}px`);
    syncAria(workspace);
    scheduleResize();
  }

  function syncAria(workspace) {
    const contentWidth = availableColumnWidth(workspace);
    const currentFeature = paneWidth(workspace, ".gbm-grid-panel");
    const featureMax = Math.max(
      GBM_FEATURE_MIN_WIDTH,
      contentWidth - GBM_PARAMETER_MIN_WIDTH - GBM_CONTROL_MIN_WIDTH,
    );
    setAriaRange(workspace.querySelector("#gbmFeatureResizer"), GBM_FEATURE_MIN_WIDTH, featureMax, currentFeature);
    setAriaRange(
      workspace.querySelector("#gbmEvaluationResizer"),
      GBM_PANE_MIN_HEIGHT,
      upperHeightLimit(workspace),
      paneHeight(workspace, ".gbm-parameter-section"),
    );
  }

  function scheduleResize() {
    if (resizeFrame !== null) return;
    resizeFrame = window.requestAnimationFrame(() => {
      resizeFrame = null;
      onResize();
    });
  }

  function disposeBindings() {
    resizeObserver?.disconnect();
    resizeObserver = null;
    if (fallbackResizeHandler) window.removeEventListener("resize", fallbackResizeHandler);
    fallbackResizeHandler = null;
    observedSize = null;
    document.body.classList.remove("gbm-feature-column-resizing", "gbm-feature-row-resizing");
    if (resizeFrame !== null) window.cancelAnimationFrame(resizeFrame);
    resizeFrame = null;
  }

  function dispose() {
    disposeBindings();
  }

  return { bind, dispose };
}

function availableColumnWidth(workspace) {
  return Math.max(0, workspace.getBoundingClientRect().width - (GBM_DIVIDER_TRACK_WIDTH * 2));
}

function workspaceSize(workspace) {
  const rect = workspace.getBoundingClientRect();
  return { width: Math.round(rect.width), height: Math.round(rect.height) };
}

function paneWidth(workspace, selector) {
  return workspace.querySelector(selector)?.getBoundingClientRect().width || 0;
}

function paneHeight(workspace, selector) {
  return workspace.querySelector(selector)?.getBoundingClientRect().height || 0;
}

function upperHeightLimit(workspace) {
  const available = workspace.getBoundingClientRect().height - (GBM_CONTROL_STRIP_HEIGHT * 2);
  return Math.max(GBM_PANE_MIN_HEIGHT, available - GBM_PANE_MIN_HEIGHT);
}

function clampUpperHeight(workspace, height) {
  return clamp(height, GBM_PANE_MIN_HEIGHT, upperHeightLimit(workspace));
}

function setAriaRange(resizer, min, max, now) {
  if (!resizer) return;
  resizer.setAttribute("aria-valuemin", String(Math.round(min)));
  resizer.setAttribute("aria-valuemax", String(Math.round(max)));
  resizer.setAttribute("aria-valuenow", String(Math.round(now)));
}

function clamp(value, min, max) {
  const number = Number(value);
  const finite = Number.isFinite(number) ? number : min;
  return Math.max(min, Math.min(max, finite));
}

export function createGbmParameterControls({ escapeHtml, parameterOptions }) {
  function optionsForName(name) {
    const parameterName = String(name || "");
    const configured = parameterOptions()?.[parameterName];
    const options = Array.isArray(configured) && configured.length
      ? configured
      : GBM_PARAMETER_OPTIONS[parameterName] || [];
    const normalised = options.map(normaliseOption);
    return parameterName === "init_score"
      ? initScoreOptions(normalised)
      : normalised.sort(compareOption);
  }

  function editorValues(name) {
    const parameterName = String(name || "");
    const options = optionsForName(parameterName).filter((option) => !option.disabled);
    if (parameterName === "init_score") return groupedInitScoreOptions(options);
    const values = {};
    for (const option of options) values[option.value] = option.label;
    return values;
  }

  function optionByValue(name, value) {
    const text = String(value ?? "");
    return optionsForName(name).find((option) => option.value === text) || null;
  }

  function valueDisplay(name, value) {
    return optionByValue(name, value)?.label || String(value ?? "");
  }

  function valueFormatter(cell) {
    const rowData = cell.getRow().getData();
    return escapeHtml(valueDisplay(rowData.name, cell.getValue()));
  }

  function valueEditorParams() {
    return {
      editorLookup: valueEditorLookup,
      paramsLookup: valueEditorParamsLookup,
    };
  }

  function valueEditorLookup(cell) {
    const rowData = cell.getRow().getData();
    return optionsForName(rowData.name).filter((option) => !option.disabled).length ? "list" : "input";
  }

  function valueEditorParamsLookup(editor, cell) {
    const rowData = cell.getRow().getData();
    const label = String(rowData.name || "Parameter value");
    const elementAttributes = {
      "aria-label": label,
      class: `gbm-parameter-editor gbm-parameter-${editor}-editor`,
    };
    if (editor === "list") {
      return {
        values: editorValues(rowData.name),
        autocomplete: true,
        freetext: true,
        listOnEmpty: true,
        elementAttributes,
      };
    }
    return {
      selectContents: true,
      elementAttributes,
    };
  }

  function controlHtml(parameter) {
    const name = String(parameter.name || "");
    const value = String(parameter.value ?? "");
    const options = optionsForName(name);
    if (!options.length) {
      return `<input data-gbm-parameter="${escapeHtml(name)}" value="${escapeHtml(value)}" />`;
    }
    const hasCurrentValue = options.some((option) => option.value === value);
    const renderedOptions = hasCurrentValue
      ? options
      : [{ value, label: `${value} (missing)`, disabled: true }, ...options];
    return `
      <select data-gbm-parameter="${escapeHtml(name)}" aria-label="${escapeHtml(name)}">
        ${selectOptionsHtml(name, renderedOptions, value)}
      </select>
    `;
  }

  function selectOptionsHtml(name, options, value) {
    if (String(name || "") !== "init_score") {
      return options.map((option) => optionHtml(option, value)).join("");
    }
    const none = options.filter((option) => option.value === "none");
    const glms = options.filter((option) => option.value !== "none" && option.kind === "glm_prediction").sort(compareOption);
    const columns = options.filter((option) => option.kind === "dataset_column").sort(compareOption);
    const other = options
      .filter((option) => option.value !== "none" && option.kind !== "glm_prediction" && option.kind !== "dataset_column")
      .sort(compareOption);
    return [
      ...none.map((option) => optionHtml(option, value)),
      optgroupHtml("GLM PREDICTIONS", glms, value),
      optgroupHtml("DATASET COLUMNS", columns, value),
      ...other.map((option) => optionHtml(option, value)),
    ].filter(Boolean).join("");
  }

  function optgroupHtml(label, options, value) {
    if (!options.length) return "";
    return `<optgroup label="${escapeHtml(label)}">${options.map((option) => optionHtml(option, value)).join("")}</optgroup>`;
  }

  function optionHtml(option, value) {
    return `<option value="${escapeHtml(option.value)}" ${option.value === value ? "selected" : ""} ${option.disabled ? "disabled" : ""}>${escapeHtml(option.label)}</option>`;
  }

  return {
    controlHtml,
    editorValues,
    optionByValue,
    optionHtml,
    optionsForName,
    optgroupHtml,
    selectOptionsHtml,
    valueDisplay,
    valueEditorLookup,
    valueEditorParams,
    valueEditorParamsLookup,
    valueFormatter,
  };
}

function normaliseOption(option) {
  if (option && typeof option === "object") {
    const value = String(option.value ?? "");
    return {
      value,
      label: String(option.label ?? value),
      kind: String(option.kind ?? ""),
      disabled: Boolean(option.disabled),
    };
  }
  const value = String(option);
  return { value, label: value, kind: "", disabled: false };
}

function compareOption(left, right) {
  if (left.value === "none") return -1;
  if (right.value === "none") return 1;
  return left.label.localeCompare(right.label, undefined, { sensitivity: "base" });
}

function initScoreOptions(options) {
  const none = options.filter((option) => option.value === "none").sort(compareOption);
  const glms = options.filter((option) => option.value !== "none" && option.kind === "glm_prediction").sort(compareOption);
  const columns = options.filter((option) => option.kind === "dataset_column").sort(compareOption);
  const other = options
    .filter((option) => option.value !== "none" && option.kind !== "glm_prediction" && option.kind !== "dataset_column")
    .sort(compareOption);
  return [...none, ...glms, ...columns, ...other];
}

function groupedInitScoreOptions(options) {
  const enabled = options.filter((option) => !option.disabled);
  const none = enabled.filter((option) => option.value === "none");
  const glms = enabled.filter((option) => option.value !== "none" && option.kind === "glm_prediction").sort(compareOption);
  const columns = enabled.filter((option) => option.kind === "dataset_column").sort(compareOption);
  const groups = [...none];
  if (glms.length) groups.push({ label: "GLM PREDICTIONS", options: glms });
  if (columns.length) groups.push({ label: "DATASET COLUMNS", options: columns });
  return groups;
}
