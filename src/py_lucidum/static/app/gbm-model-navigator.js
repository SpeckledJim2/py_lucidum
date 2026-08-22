import {
  bindFallbackModelSelection,
  emptyStateHtml,
  formatModelCreated,
  modelCreatedSort,
  modelNumberOrNull,
} from "./shared/model-ui.js";

export function createGbmModelNavigator({
  escapeHtml,
  formatModelMetric,
  modelInteractionConstraintLabel,
  modelLabel,
  normaliseModel,
  uniqueModels,
  onFallbackSelectionChange,
}) {
  function rows(models) {
    return uniqueModels(models.map(normaliseModel).filter((model) => model.model_id)).map((model) => ({
      ...model,
      model_label: modelLabel(model),
      created_sort: modelCreatedSort(model.created_at),
      created_display: formatModelCreated(model.created_at),
      weight_display: weightLabel(model.offset_column),
      training_mode_display: model.training_mode === "ebm" ? "EBM" : "Normal",
      constraint_display: modelInteractionConstraintLabel(model.feature_interaction_constraints),
      best_training_metric: bestMetric(model, "training"),
      best_test_metric: bestMetric(model, "test"),
      gini_tr: modelNumberOrNull(model.gini_tr),
      gini_te: modelNumberOrNull(model.gini_te),
      gini_vl: modelNumberOrNull(model.gini_vl),
      param_num_iterations: parameterNumber(model, "num_iterations"),
      param_learning_rate: parameterNumber(model, "learning_rate"),
      param_num_leaves: parameterNumber(model, "num_leaves"),
      param_max_depth: parameterNumber(model, "max_depth"),
      param_min_data_in_leaf: parameterNumber(model, "min_data_in_leaf"),
      param_early_stopping_rounds: parameterNumber(model, "early_stopping_rounds"),
      runtime_seconds: runtimeSeconds(model),
      runtime_display: runtime(model),
      sample_display: sampleMode(model.sample_column, model.sample_source),
    }));
  }

  function renderFallback(target, models) {
    if (!target) return;
    if (!models.length) {
      target.innerHTML = emptyStateHtml("No GBMs trained yet", "gbm-empty-state", escapeHtml);
      return;
    }
    target.innerHTML = `
      <table class="gbm-model-table">
        <thead>
          <tr>
            <th class="gbm-model-active-heading" aria-label="Active model"></th>
            <th>Name</th>
            <th>Created</th>
            <th>Response</th>
            <th>Weight</th>
            <th>Objective</th>
            <th>Metric</th>
            <th>Mode</th>
            <th>Constraints</th>
            <th class="numeric">Train</th>
            <th class="numeric">Best iter.</th>
            <th class="numeric compact" title="Training metric at best iteration">tr@best</th>
            <th class="numeric compact" title="Test metric at best iteration">te@best</th>
            <th class="numeric compact" title="Normalized Gini for SAMPLE = training">gini_tr</th>
            <th class="numeric compact" title="Normalized Gini for SAMPLE = test">gini_te</th>
            <th class="numeric compact" title="Normalized Gini for SAMPLE = validation">gini_vl</th>
            <th class="numeric compact" title="num_iterations">n_iter</th>
            <th class="numeric compact" title="learning_rate">lr</th>
            <th class="numeric compact" title="num_leaves">leaves</th>
            <th class="numeric compact" title="max_depth">depth</th>
            <th class="numeric compact" title="min_data_in_leaf">min_leaf</th>
            <th class="numeric compact" title="early_stopping_rounds">ES</th>
            <th class="numeric">Run time</th>
            <th>Sample</th>
          </tr>
        </thead>
        <tbody>
          ${models.map((model) => `
            <tr data-gbm-model-row="${escapeHtml(model.model_id)}" aria-selected="false">
              <td class="gbm-model-active-cell">
                ${model.active ? '<span class="gbm-model-active-dot" title="Active model" aria-label="Active model"></span>' : ""}
              </td>
              <td class="gbm-model-name-cell">
                <span class="gbm-model-name-main">${escapeHtml(model.model_label)}</span>
              </td>
              <td>${escapeHtml(model.created_display)}</td>
              <td>${escapeHtml(model.response_column)}</td>
              <td>${escapeHtml(model.weight_display)}</td>
              <td>${escapeHtml(model.objective || "")}</td>
              <td>${escapeHtml(model.metric || "")}</td>
              <td>${escapeHtml(model.training_mode_display)}</td>
              <td>${escapeHtml(model.constraint_display)}</td>
              <td class="numeric">${count(model.training_rows)}</td>
              <td class="numeric">${count(model.best_iteration)}</td>
              <td class="numeric">${escapeHtml(formatModelMetric(model.best_training_metric))}</td>
              <td class="numeric">${escapeHtml(formatModelMetric(model.best_test_metric))}</td>
              <td class="numeric">${escapeHtml(formatModelMetric(model.gini_tr))}</td>
              <td class="numeric">${escapeHtml(formatModelMetric(model.gini_te))}</td>
              <td class="numeric">${escapeHtml(formatModelMetric(model.gini_vl))}</td>
              <td class="numeric">${escapeHtml(integer(model.param_num_iterations))}</td>
              <td class="numeric">${escapeHtml(decimal(model.param_learning_rate))}</td>
              <td class="numeric">${escapeHtml(integer(model.param_num_leaves))}</td>
              <td class="numeric">${escapeHtml(integer(model.param_max_depth))}</td>
              <td class="numeric">${escapeHtml(integer(model.param_min_data_in_leaf))}</td>
              <td class="numeric">${escapeHtml(integer(model.param_early_stopping_rounds))}</td>
              <td class="numeric">${escapeHtml(model.runtime_display)}</td>
              <td>${escapeHtml(model.sample_display)}</td>
            </tr>
          `).join("")}
        </tbody>
      </table>
    `;
    const fallbackRows = Array.from(target.querySelectorAll("[data-gbm-model-row]"));
    bindFallbackModelSelection(fallbackRows, onFallbackSelectionChange);
  }

  function activeDotFormatter(cell) {
    return cell.getValue() ? '<span class="gbm-model-active-dot" title="Active model" aria-label="Active model"></span>' : "";
  }

  function nameFormatter(cell) {
    return `<span class="gbm-model-name-main">${escapeHtml(cell.getValue() || "")}</span>`;
  }

  return {
    activeDotFormatter,
    count,
    created: formatModelCreated,
    createdSort: modelCreatedSort,
    decimal,
    integer,
    nameFormatter,
    parameterNumber,
    renderFallback,
    rows,
    runtime,
    runtimeSeconds,
    sampleMode,
    weightLabel,
  };
}

function bestMetric(model, name) {
  const metrics = model?.best_metrics && typeof model.best_metrics === "object" ? model.best_metrics : {};
  return modelNumberOrNull(metrics[name]);
}

function count(value) {
  const number = Number(value);
  return Number.isFinite(number) && number > 0 ? Math.round(number).toLocaleString() : "0";
}

function parameterNumber(model, name) {
  const parameters = model?.parameters && typeof model.parameters === "object" ? model.parameters : {};
  return modelNumberOrNull(parameters[name]);
}

function integer(value) {
  const number = Number(value);
  return Number.isFinite(number) ? Math.round(number).toLocaleString() : "--";
}

function decimal(value) {
  const number = Number(value);
  if (!Number.isFinite(number)) return "--";
  return number.toLocaleString(undefined, { maximumSignificantDigits: 4 });
}

function runtimeSeconds(model) {
  const seconds = Number(model?.timings?.training_seconds ?? model?.training_seconds);
  return Number.isFinite(seconds) && seconds >= 0 ? seconds : -1;
}

function runtime(model) {
  const seconds = runtimeSeconds(model);
  if (seconds < 0) return "--";
  if (seconds < 1) return `${Math.round(seconds * 1000).toLocaleString()}ms`;
  if (seconds < 60) return `${seconds.toLocaleString(undefined, { maximumFractionDigits: 1 })}s`;
  const minutes = Math.floor(seconds / 60);
  const remainder = Math.round(seconds % 60);
  return `${minutes}m ${remainder.toString().padStart(2, "0")}s`;
}

function sampleMode(value, source = "") {
  const text = String(value || "").trim();
  if (!text) return "All rows";
  if (String(source || "").trim() === "generated") return "Generated 60/20/20";
  return text;
}

function weightLabel(value) {
  const text = String(value || "").trim();
  return !text || text === "__none__" || text === "Average row value" ? "N" : text;
}
