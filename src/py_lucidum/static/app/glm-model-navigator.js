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
        training_rows: Number(model.training_rows || diagnostics.training_rows || 0),
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
        <td class="numeric">${escapeHtml(formatModelMetric(diagnostics.deviance))}</td>
        <td class="numeric">${escapeHtml(formatModelMetric(diagnostics.aic))}</td>
        <td class="numeric">${escapeHtml(formatModelMetric(diagnostics.bic))}</td>
        <td class="numeric">${Number(model.training_rows || diagnostics.training_rows || 0).toLocaleString()}</td>
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
    renderFallback,
    rowHtml,
    rows,
  };
}
