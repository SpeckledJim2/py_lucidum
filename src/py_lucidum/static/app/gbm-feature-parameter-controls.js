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
