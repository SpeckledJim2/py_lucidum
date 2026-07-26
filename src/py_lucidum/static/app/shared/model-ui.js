const MODEL_MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];

export function modelNumberOrNull(value) {
  if (value === null || value === undefined || String(value).trim() === "") return null;
  const number = Number(value);
  return Number.isFinite(number) ? number : null;
}

export function formatModelMetric(value) {
  const number = modelNumberOrNull(value);
  if (number === null) return "--";
  const formatted = number.toLocaleString(undefined, { maximumFractionDigits: 4 });
  return /^-0(?:[.,]0+)?$/.test(formatted) ? formatted.slice(1) : formatted;
}

export function formatModelCreated(value) {
  if (!value) return "";
  const date = new Date(value);
  if (!Number.isFinite(date.getTime())) return String(value);
  const hour = String(date.getHours()).padStart(2, "0");
  const minute = String(date.getMinutes()).padStart(2, "0");
  return `${date.getDate()} ${MODEL_MONTHS[date.getMonth()]} ${hour}:${minute}`;
}

export function modelCreatedSort(value) {
  const time = new Date(value || "").getTime();
  return Number.isFinite(time) ? time : 0;
}

export function isModelJobPending(status) {
  return status === "queued" || status === "running";
}

export function modelJobPollDelay(status, queuedMs, runningMs) {
  if (status === "queued") return queuedMs;
  if (status === "running") return runningMs;
  return 0;
}

export function modelGroups(models = [], groupLabel) {
  const groups = new Map();
  for (const model of models) {
    const group = String(groupLabel(model) || "").trim() || "Models";
    if (!groups.has(group)) groups.set(group, []);
    groups.get(group).push(model);
  }
  return groups;
}

export function syncCollapsedModelGroups({ groups = [], collapsedGroups, initialised = false, activeGroup = "" } = {}) {
  const groupNames = groups.map(String);
  if (!collapsedGroups) return { initialised, groups: groupNames };
  let nextInitialised = Boolean(initialised);
  if (!nextInitialised) {
    groupNames.forEach((group) => collapsedGroups.add(group));
    const openGroup = activeGroup || groupNames[0];
    if (openGroup) collapsedGroups.delete(openGroup);
    nextInitialised = true;
  }
  for (const group of [...collapsedGroups]) {
    if (!groupNames.includes(group)) collapsedGroups.delete(group);
  }
  return { initialised: nextInitialised, groups: groupNames };
}

export function createSidebarModelHeading({
  group,
  collapsed,
  toolLabel,
  className,
  dataKey,
  escapeHtml,
  onToggle,
}) {
  const heading = document.createElement("button");
  heading.type = "button";
  heading.className = `saved-filter-theme ${className}`;
  heading.dataset[dataKey] = group;
  heading.setAttribute("aria-expanded", String(!collapsed));
  heading.setAttribute("aria-label", `${collapsed ? "Expand" : "Collapse"} ${group} ${toolLabel} models`);
  heading.title = `${collapsed ? "Expand" : "Collapse"} ${group} ${toolLabel} models`;
  heading.innerHTML = `<span class="saved-filter-theme-icon" aria-hidden="true"></span><span class="saved-filter-theme-label">${escapeHtml(group)}</span>`;
  heading.addEventListener("click", () => onToggle(group));
  return heading;
}

export function createSidebarModelOption({
  model,
  group,
  active,
  collapsed,
  className,
  detailClassName,
  modelIdDataKey,
  groupDataKey,
  escapeHtml,
  modelLabel,
  modelDetailLabel,
  onActivate,
}) {
  const button = document.createElement("button");
  button.type = "button";
  button.className = `feature ${className}${active ? " active" : ""}`;
  button.dataset[modelIdDataKey] = model.model_id;
  button.dataset[groupDataKey] = group;
  button.hidden = collapsed;
  button.setAttribute("role", "option");
  button.setAttribute("aria-selected", String(active));
  button.innerHTML = `<span class="saved-filter-name">${escapeHtml(modelLabel(model))}</span><span class="${detailClassName}">${escapeHtml(modelDetailLabel(model))}</span>`;
  button.addEventListener("click", () => {
    if (!active) onActivate(model.model_id, model);
  });
  return button;
}

export function emptyStateHtml(message, className, escapeHtml) {
  return `<div class="${className}">${escapeHtml(message)}</div>`;
}

export function toggleSidebarModelGroup({
  list,
  group,
  collapsedGroups,
  themeClassName,
  optionClassName,
  groupDataKey,
  toolLabel,
}) {
  if (!list || !collapsedGroups) return false;
  const collapsed = !collapsedGroups.has(group);
  if (collapsed) collapsedGroups.add(group);
  else collapsedGroups.delete(group);
  list.querySelectorAll(`.${themeClassName}`).forEach((heading) => {
    if (heading.dataset[groupDataKey] !== group) return;
    heading.setAttribute("aria-expanded", String(!collapsed));
    heading.setAttribute("aria-label", `${collapsed ? "Expand" : "Collapse"} ${group} ${toolLabel} models`);
    heading.title = `${collapsed ? "Expand" : "Collapse"} ${group} ${toolLabel} models`;
  });
  list.querySelectorAll(`.${optionClassName}`).forEach((button) => {
    if (button.dataset[groupDataKey] === group) button.hidden = collapsed;
  });
  return collapsed;
}

export function bindFallbackModelSelection(rows, onChange) {
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
      onChange();
    });
  });
  onChange();
}

export function selectedModelIdsFromTableOrFallback({ table, fallbackSelector, rowDataKey }) {
  const ids = table?.initialized === true && typeof table.getSelectedData === "function"
    ? table.getSelectedData().map((row) => row?.model_id)
    : Array.from(document.querySelectorAll(`${fallbackSelector}[aria-selected="true"]`))
      .map((row) => row.dataset[rowDataKey]);
  return [...new Set(ids.map((id) => String(id || "")).filter(Boolean))];
}

export function restoreModelSelection({ table, fallbackSelector, rowDataKey, ids = [] }) {
  const selected = new Set((ids || []).map((id) => String(id || "")).filter(Boolean));
  if (table?.initialized === true && typeof table.getRows === "function") {
    for (const row of table.getRows()) {
      const rowId = String(row.getData()?.model_id || "");
      if (selected.has(rowId)) {
        row.select();
      } else {
        row.deselect();
      }
    }
    return selected;
  }
  for (const row of document.querySelectorAll(fallbackSelector)) {
    const rowId = String(row.dataset[rowDataKey] || "");
    const active = selected.has(rowId);
    row.classList.toggle("selected", active);
    row.setAttribute("aria-selected", String(active));
  }
  return selected;
}

export function syncModelActionButtons({ selectedCount, disabled = false, activate, rename, deleteButton }) {
  if (activate) activate.disabled = disabled || selectedCount !== 1;
  if (rename) rename.disabled = disabled || selectedCount !== 1;
  if (deleteButton) deleteButton.disabled = disabled || selectedCount < 1;
}

export function observeResize(targets, onResize) {
  if (!window.ResizeObserver) return null;
  const observer = new ResizeObserver(onResize);
  for (const target of targets) {
    if (target) observer.observe(target);
  }
  return observer;
}

export function setInlinePhaseStatus(element, { html = "", phase = "", hidden = false } = {}) {
  if (!element) return;
  element.innerHTML = String(html || "");
  element.dataset.phase = String(phase || "");
  element.classList.toggle("hidden", Boolean(hidden));
}
