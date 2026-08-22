export function createGlmModelNavigator({
  bindFallbackModelSelection,
  emptyStateHtml,
  escapeHtml,
  formatModelCreated,
  formatModelMetric,
  modelCreatedSort,
  modelLabel,
  modelNumberOrNull,
  modelWeightLabel,
  normaliseModels,
  selectedModelIds,
  onFallbackSelectionChange,
}) {
  function rows(models = [], activeModelId = "") {
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
        gini_tr: modelNumberOrNull(model.gini_tr ?? diagnostics.gini_tr),
        gini_te: modelNumberOrNull(model.gini_te ?? diagnostics.gini_te),
        gini_vl: modelNumberOrNull(model.gini_vl ?? diagnostics.gini_vl),
        n_terms: modelNumberOrNull(model.n_terms ?? diagnostics.n_terms),
        n_features: modelNumberOrNull(model.n_features ?? diagnostics.n_features),
        n_interactions: modelNumberOrNull(model.n_interactions ?? diagnostics.n_interactions),
        training_rows: Number(model.training_rows || diagnostics.training_rows || 0),
        scope_display: glmTrainingScopeLabel(model.training_scope),
        fit_ms: timingMilliseconds(model, "fit_ms"),
        fit_display: timingDisplay(model, "fit_ms"),
        elapsed_ms: timingMilliseconds(model, "elapsed_ms"),
        elapsed_display: timingDisplay(model, "elapsed_ms"),
      };
    });
  }

  function renderFallback(target, models = [], activeModelId = "") {
    if (!target) return;
    if (!models.length) {
      target.innerHTML = emptyStateHtml("No GLMs built yet", "glm-empty-state", escapeHtml);
      return;
    }
    target.innerHTML = `
      <table class="glm-table glm-model-table">
        <thead>
          <tr>
            <th class="glm-model-active-heading" aria-label="Active model"></th>
            <th>Name</th>
            <th>created</th>
            <th>response</th>
            <th>weight</th>
            <th>family</th>
            <th>Terms</th>
            <th>Features</th>
            <th>Interactions</th>
            <th>Tabulated</th>
            <th>deviance</th>
            <th>AIC</th>
            <th>BIC</th>
            <th class="numeric" title="Normalized Gini for SAMPLE = training">gini_tr</th>
            <th class="numeric" title="Normalized Gini for SAMPLE = test">gini_te</th>
            <th class="numeric" title="Normalized Gini for SAMPLE = validation">gini_vl</th>
            <th>Rows</th>
            <th>Scope</th>
            <th>Fit time</th>
            <th>Overall time</th>
          </tr>
        </thead>
        <tbody>
          ${models.map((model) => rowHtml(model, activeModelId)).join("")}
        </tbody>
      </table>
    `;
    const fallbackRows = Array.from(target.querySelectorAll("[data-glm-model-row]"));
    bindFallbackModelSelection(fallbackRows, onFallbackSelectionChange);
  }

  function rowHtml(model, activeModelId) {
    const active = model.model_id === activeModelId;
    const selected = selectedModelIds().has(model.model_id);
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
        <td class="numeric">${escapeHtml(optionalCount(model.n_terms ?? diagnostics.n_terms))}</td>
        <td class="numeric">${escapeHtml(optionalCount(model.n_features ?? diagnostics.n_features))}</td>
        <td class="numeric">${escapeHtml(optionalCount(model.n_interactions ?? diagnostics.n_interactions))}</td>
        <td>${model.tabulated ? "Yes" : "-"}</td>
        <td class="numeric">${escapeHtml(formatModelMetric(diagnostics.deviance))}</td>
        <td class="numeric">${escapeHtml(formatModelMetric(diagnostics.aic))}</td>
        <td class="numeric">${escapeHtml(formatModelMetric(diagnostics.bic))}</td>
        <td class="numeric">${escapeHtml(formatModelMetric(model.gini_tr ?? diagnostics.gini_tr))}</td>
        <td class="numeric">${escapeHtml(formatModelMetric(model.gini_te ?? diagnostics.gini_te))}</td>
        <td class="numeric">${escapeHtml(formatModelMetric(model.gini_vl ?? diagnostics.gini_vl))}</td>
        <td class="numeric">${Number(model.training_rows || diagnostics.training_rows || 0).toLocaleString()}</td>
        <td>${escapeHtml(glmTrainingScopeLabel(model.training_scope))}</td>
        <td class="numeric">${escapeHtml(timingDisplay(model, "fit_ms"))}</td>
        <td class="numeric">${escapeHtml(timingDisplay(model, "elapsed_ms"))}</td>
      </tr>
    `;
  }

  function activeDotFormatter(cell) {
    return cell.getValue() ? '<span class="glm-model-active-dot" title="Active model" aria-label="Active model"></span>' : "";
  }

  function nameFormatter(cell) {
    return `<span class="glm-model-name-main">${escapeHtml(cell.getValue() || "")}</span>`;
  }

  return {
    activeDotFormatter,
    nameFormatter,
    optionalCount,
    renderFallback,
    rowHtml,
    rows,
  };
}

export function glmTrainingScopeLabel(value) {
  const scope = String(value ?? "").trim().toLowerCase();
  if (!scope || scope === "all") return "All";
  if (scope === "training") return "Training";
  if (scope === "training_test") return "Training + Test";
  return "--";
}

function optionalCount(value) {
  if (value === null || value === undefined || String(value).trim() === "") return "";
  const number = Number(value);
  return Number.isFinite(number) ? Math.round(number).toLocaleString() : "";
}

function timingMilliseconds(model, name) {
  const milliseconds = Number(model?.timings?.[name]);
  return Number.isFinite(milliseconds) && milliseconds >= 0 ? milliseconds : -1;
}

function timingDisplay(model, name) {
  return formatMilliseconds(timingMilliseconds(model, name));
}

function formatMilliseconds(value) {
  const milliseconds = Number(value);
  if (!Number.isFinite(milliseconds) || milliseconds < 0) return "--";
  if (milliseconds < 1000) return `${Math.round(milliseconds).toLocaleString()}ms`;
  const seconds = milliseconds / 1000;
  if (seconds < 60) return `${seconds.toLocaleString(undefined, { maximumFractionDigits: 1 })}s`;
  const roundedSeconds = Math.round(seconds);
  const minutes = Math.floor(roundedSeconds / 60);
  const remainder = roundedSeconds % 60;
  return `${minutes}m ${remainder.toString().padStart(2, "0")}s`;
}
