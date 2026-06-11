import { loadTabulator } from "./shared/tabulator.js";

const SPEC_KINDS = [
  { id: "feature", label: "Feature spec" },
  { id: "kpi", label: "KPI spec" },
  { id: "filter", label: "Filter spec" },
];

const FEATURE_METADATA_COLUMNS = new Set(["Base", "min", "max", "banding"]);
const FIELD_TITLES = {
  feature: {
    Feature: "Feature",
    Grouping: "Grouping",
    Base: "Base",
    min: "min",
    max: "max",
    banding: "banding",
  },
  kpi: {
    group: "Group",
    name: "Name",
    actual: "Actual",
    denominator: "Denominator",
    decimals: "Decimals",
    format: "Format",
  },
  filter: {
    theme: "Group",
    name: "Name",
    expression: "Expression",
  },
};

export function createSpecificationsTool({
  api,
  clearGlobalStatus = () => {},
  datasetColumnNames = () => [],
  el,
  escapeHtml,
  measureToolRender,
  reloadSchemaAfterSpecsSave,
  showClipboardToast,
}) {
  let rendered = false;
  let table = null;
  let activeKind = "feature";
  let contextRowId = "";
  let contextColumnField = "";
  let rowIdCounter = 0;
  let loading = false;
  let suppressDirty = false;
  let selection = null;
  let selectionDragging = false;
  let pendingScrollRestore = null;
  const specs = new Map();

  function renderShell() {
    if (rendered) return;
    el("specificationsWrap").innerHTML = `
      <div class="spec-tool">
        <div class="spec-topbar">
          <div class="spec-control-row">
            <div class="tabs spec-kind-tabs" role="tablist" aria-label="Specification type">
              ${SPEC_KINDS.map((kind) => `<button class="tab ${kind.id === activeKind ? "active" : ""}" type="button" role="tab" data-spec-kind="${kind.id}" aria-selected="${kind.id === activeKind ? "true" : "false"}">${escapeHtml(kind.label)}</button>`).join("")}
            </div>
            <div class="spec-file-actions">
              <div id="specNotice" class="spec-notice hidden" role="status" aria-live="polite"></div>
              <button id="specValidateBtn" class="ghost spec-action-button" type="button">Validate</button>
              <button id="specSaveBtn" class="spec-save-button" type="button">Save</button>
            </div>
          </div>
          <div id="specGenerationNotice" class="spec-generation-notice hidden" role="status" aria-live="polite"></div>
          <span id="specFilePath" class="spec-file-path"></span>
        </div>
        <div id="specGrid" class="spec-grid" tabindex="0"></div>
        <div id="specContextMenu" class="spec-context-menu" role="menu" hidden>
          <button class="spec-context-menu-item" type="button" role="menuitem" data-spec-row-action="above">Add row above</button>
          <button class="spec-context-menu-item" type="button" role="menuitem" data-spec-row-action="below">Add row below</button>
          <button class="spec-context-menu-item" type="button" role="menuitem" data-spec-row-action="delete">Delete row</button>
        </div>
        <div id="specColumnContextMenu" class="spec-context-menu" role="menu" hidden></div>
      </div>
    `;
    bindShell();
    rendered = true;
  }

  function bindShell() {
    el("specificationsWrap").querySelectorAll("[data-spec-kind]").forEach((button) => {
      button.addEventListener("click", () => selectKind(button.dataset.specKind));
    });
    el("specValidateBtn").addEventListener("click", validateCurrentSpec);
    el("specSaveBtn").addEventListener("click", saveCurrentSpec);
    el("specContextMenu").addEventListener("click", (event) => {
      const button = event.target.closest("[data-spec-row-action]");
      if (!button) return;
      event.preventDefault();
      const action = button.dataset.specRowAction;
      closeMenus();
      if (action === "above") addRowRelative("above");
      if (action === "below") addRowRelative("below");
      if (action === "delete") void deleteContextRow();
    });
    el("specColumnContextMenu").addEventListener("click", (event) => {
      const button = event.target.closest("[data-spec-column-action]");
      if (!button) return;
      event.preventDefault();
      const action = button.dataset.specColumnAction;
      const field = contextColumnField;
      closeMenus();
      if (action === "add-end") addScenarioAt("", "end");
      if (action === "add-before") addScenarioAt(field, "before");
      if (action === "add-after") addScenarioAt(field, "after");
      if (action === "delete") deleteScenarioField(field);
      if (action === "rename") renameScenarioField(field);
    });
    document.addEventListener("click", (event) => {
      if (!el("specContextMenu")?.contains(event.target) && !el("specColumnContextMenu")?.contains(event.target)) closeMenus();
    });
    document.addEventListener("keydown", (event) => {
      if (handleSpecKeydown(event)) return;
      if (event.key === "Escape") closeMenus();
    });
    document.addEventListener("copy", handleSpecCopy);
    document.addEventListener("paste", handleSpecPaste);
    document.addEventListener("mouseup", endSelectionDrag, true);
  }

  async function activate() {
    renderShell();
    clearGlobalStatus();
    await loadKind(activeKind);
  }

  async function refresh(options = {}) {
    renderShell();
    await loadKind(activeKind, options);
  }

  function resize() {
    table?.redraw?.(true);
  }

  function refreshTheme() {
    const scrollPosition = captureSpecTableScroll();
    table?.redraw?.(false);
    scheduleSpecTableScrollRestore(scrollPosition);
  }

  async function selectKind(kind) {
    const nextKind = SPEC_KINDS.some((entry) => entry.id === kind) ? kind : "feature";
    if (nextKind === activeKind) return;
    saveActiveDraft();
    clearValidationRowIssuesForSpec(specs.get(activeKind));
    activeKind = nextKind;
    syncKindTabs();
    await loadKind(activeKind);
  }

  async function loadKind(kind, options = {}) {
    if (loading) return;
    const cached = specs.get(kind);
    if (cached?.dirty || (cached && !options.force)) {
      renderSpec(cached);
      if (cached && !cached.dirty) await validateSpecOnLoad(cached);
      return;
    }
    loading = true;
    syncButtons();
    showNotice({ message: "Loading specification", valid: true, warnings: [], errors: [] });
    try {
      const payload = await api(`/api/specs/${kind}`, { method: "GET" });
      const spec = cacheSpec(payload, false);
      if (kind === activeKind) {
        renderSpec(spec);
        if (!spec.generated) await validateSpecOnLoad(spec);
      }
    } catch (error) {
      showNotice({ valid: false, errors: [error.message], warnings: [], message: error.message });
    } finally {
      loading = false;
      syncButtons();
    }
  }

  function cacheSpec(payload, dirty) {
    const kind = payload.kind || activeKind;
    const columns = Array.isArray(payload.columns) && payload.columns.length ? payload.columns.map(String) : ["name"];
    const rows = rowsWithIds(Array.isArray(payload.rows) ? payload.rows : [], columns);
    const spec = {
      ...payload,
      kind,
      columns,
      rows,
      placeholders: payload.placeholders && typeof payload.placeholders === "object" ? payload.placeholders : {},
      rowIssues: normaliseRowIssues(payload.row_issues),
      dirty: Boolean(dirty),
    };
    specs.set(kind, spec);
    return spec;
  }

  function renderSpec(spec, options = {}) {
    if (!spec || spec.kind !== activeKind) return;
    if (options.preserveScroll) pendingScrollRestore = captureSpecTableScroll();
    renumberSpecRows(spec);
    clearGlobalStatus();
    syncKindTabs();
    syncFilePath(spec);
    syncGenerationNotice(spec);
    showValidationNotice(spec.validationResult || null, { showValid: false });
    measureToolRender("specs", () => renderTable(spec));
    syncButtons();
  }

  function renderTable(spec) {
    const target = el("specGrid");
    resetSelection();
    table?.destroy?.();
    table = null;
    loadTabulator()
      .then((Tabulator) => {
        if (spec.kind !== activeKind) return;
        suppressDirty = true;
        table = new Tabulator(target, {
          data: spec.rows,
          height: "100%",
          index: "_row_id",
          layout: "fitDataStretch",
          placeholder: "No specification rows",
          editTriggerEvent: "dblclick",
          rowHeader: specRowHeader(),
          columns: tabulatorColumns(spec),
        });
        table.on("tableBuilt", () => {
          suppressDirty = false;
          table?.redraw?.(true);
          applyTableDecorations();
          restorePendingSpecTableScroll();
        });
        table.on("rowContext", openRowContextMenu);
        if (spec.kind === "feature") table.on("headerContext", openColumnContextMenu);
        table.on("cellMouseDown", startSelectionFromCell);
        table.on("cellMouseOver", extendSelectionToCell);
        table.on("cellClick", handleSpecCellClick);
        table.on("cellDblClick", handleSpecCellDblClick);
        table.on("renderComplete", applyTableDecorations);
        table.on("dataSorted", applyTableDecorations);
        table.on("cellEdited", (cell) => {
          if (!suppressDirty) {
            clearValidationRowIssuesForActiveSpec();
            markDirty();
            restoreSelectionAfterCellEdit(cell);
          }
          applyTableDecorations();
        });
        table.on("dataChanged", () => {
          if (!suppressDirty) markDirty();
        });
        window.setTimeout(() => {
          suppressDirty = false;
        }, 0);
      })
      .catch((error) => {
        target.innerHTML = `<div class="spec-empty-state">${escapeHtml(error.message || "Specification table failed to load")}</div>`;
      });
  }

  function tabulatorColumns(spec) {
    return spec.columns.map((field) => {
      const title = columnTitle(spec, field);
      const column = {
        title,
        field,
        editor: "input",
        headerSort: true,
        formatter: (cell) => textFormatter(cell, spec),
        minWidth: columnMinWidth(spec, field, title),
        widthGrow: columnGrow(spec.kind, field),
      };
      if (spec.kind === "feature" && isScenarioField(field, spec)) {
        column.cssClass = "spec-scenario-cell";
        delete column.editor;
        column.editable = false;
        column.formatter = scenarioFormatter;
        column.hozAlign = "center";
        column.headerHozAlign = "center";
      }
      if (spec.kind === "kpi" && field === "decimals") {
        column.widthGrow = 0;
      }
      if (["Base", "min", "max", "banding", "format"].includes(field)) {
        column.widthGrow = 0.8;
      }
      return column;
    });
  }

  function specRowHeader() {
    return {
      title: "",
      field: "_spec_row_number",
      formatter: rowNumberFormatter,
      headerSort: false,
      hozAlign: "center",
      headerHozAlign: "center",
      width: 38,
      minWidth: 38,
      resizable: false,
      frozen: true,
      cssClass: "spec-row-number-cell",
    };
  }

  function rowNumberFormatter(cell) {
    return escapeHtml(cell.getValue() ?? "");
  }

  function fieldTitle(kind, field) {
    return FIELD_TITLES[kind]?.[field] || field;
  }

  function columnTitle(spec, field) {
    if (spec?.kind === "feature" && isScenarioField(field, spec)) {
      return scenarioHeaderTitle(spec, field);
    }
    return fieldTitle(spec?.kind, field);
  }

  function columnMinWidth(spec, field, title = columnTitle(spec, field)) {
    const headerWidth = headerMinWidth(title);
    if (spec?.kind === "feature" && field === "Feature") return Math.max(headerWidth, 180);
    if (spec?.kind === "feature" && field === "Grouping") return Math.max(headerWidth, 112);
    if (spec?.kind === "feature" && isScenarioField(field, spec)) return Math.max(headerWidth, 116);
    if (spec?.kind === "filter" && field === "expression") return Math.max(headerWidth, 360);
    if (spec?.kind === "filter" && field === "name") return Math.max(headerWidth, 220);
    if (spec?.kind === "kpi" && field === "name") return Math.max(headerWidth, 180);
    if (spec?.kind === "kpi" && field === "denominator") return Math.max(headerWidth, 160);
    if (["Base", "min", "max", "banding", "decimals", "format"].includes(field)) return Math.max(headerWidth, 86);
    return headerWidth;
  }

  function headerMinWidth(title) {
    return Math.max(64, Math.ceil(String(title || "").length * 7.5) + 34);
  }

  function scenarioHeaderTitle(spec, field) {
    return `${field} (${scenarioSelectionCount(spec, field)})`;
  }

  function scenarioSelectionCount(spec, field) {
    return scenarioCountRows(spec).filter((row) => scenarioCellSelected(row?.[field])).length;
  }

  function scenarioCountRows(spec) {
    if (spec?.kind === activeKind && table) {
      try {
        return table.getData?.() || spec.rows || [];
      } catch (_) {
        return spec.rows || [];
      }
    }
    return spec?.rows || [];
  }

  function refreshScenarioHeaderCounts() {
    const spec = specs.get(activeKind);
    if (!spec || spec.kind !== "feature" || !table) return;
    const fields = scenarioFields(spec);
    if (!fields.length) return;
    fields.forEach((field) => {
      const title = scenarioHeaderTitle(spec, field);
      el("specGrid")?.querySelectorAll(".tabulator-header .tabulator-col[tabulator-field]").forEach((header) => {
        if (header.getAttribute("tabulator-field") !== field) return;
        const titleElement = header.querySelector(".tabulator-col-title");
        if (titleElement) titleElement.textContent = title;
      });
    });
  }

  function captureSpecTableScroll() {
    const holder = el("specGrid")?.querySelector(".tabulator-tableholder");
    if (!holder) return null;
    return { holder, kind: activeKind, left: holder.scrollLeft, top: holder.scrollTop };
  }

  function restoreSpecTableScroll(position) {
    if (!position || (position.kind && position.kind !== activeKind)) return;
    const holder = position.holder?.isConnected
      ? position.holder
      : el("specGrid")?.querySelector(".tabulator-tableholder");
    if (!holder) return;
    holder.scrollLeft = position.left;
    holder.scrollTop = position.top;
  }

  function scheduleSpecTableScrollRestore(position) {
    restoreSpecTableScroll(position);
    window.requestAnimationFrame(() => {
      restoreSpecTableScroll(position);
      window.setTimeout(() => restoreSpecTableScroll(position), 0);
    });
  }

  function restorePendingSpecTableScroll() {
    const position = pendingScrollRestore;
    pendingScrollRestore = null;
    scheduleSpecTableScrollRestore(position);
  }

  function columnGrow(kind, field) {
    if (kind === "filter" && field === "expression") return 3;
    if (kind === "feature" && field === "Feature") return 4;
    if (kind === "kpi" && field === "name") return 2.4;
    if (kind === "kpi" && field === "actual") return 2;
    return 1;
  }

  function textFormatter(cell, spec) {
    const value = cell.getValue() ?? "";
    if (String(value) === "" && spec?.generated && spec?.placeholders?.[cell.getField()]) {
      return `<span class="spec-cell-placeholder">${escapeHtml(spec.placeholders[cell.getField()])}</span>`;
    }
    return escapeHtml(value);
  }

  function scenarioFormatter(cell) {
    const checked = scenarioCellSelected(cell.getValue());
    return `<input class="spec-checkbox-cell" type="checkbox" tabindex="-1" aria-label="${escapeHtml(cell.getField())}" ${checked ? "checked" : ""}>`;
  }

  function scenarioCellSelected(value) {
    const text = String(value || "").trim().toLowerCase();
    return text === "feature" || text === "true" || text === "yes" || text === "1" || text === "x" || text === "checked";
  }

  function normaliseScenarioValue(value) {
    const text = String(value || "").trim();
    if (!text) return "";
    if (scenarioCellSelected(text)) return "feature";
    return "";
  }

  function syncKindTabs() {
    if (!rendered) return;
    el("specificationsWrap").querySelectorAll("[data-spec-kind]").forEach((button) => {
      const active = button.dataset.specKind === activeKind;
      button.classList.toggle("active", active);
      button.setAttribute("aria-selected", String(active));
    });
  }

  function syncFilePath(spec = specs.get(activeKind)) {
    const target = el("specFilePath");
    if (!target) return;
    const path = String(spec?.path || "");
    target.textContent = path;
    target.title = path;
  }

  function syncGenerationNotice(spec = specs.get(activeKind)) {
    const target = el("specGenerationNotice");
    if (!target) return;
    const message = spec?.generated ? String(spec.generation_message || "") : "";
    target.textContent = message;
    target.title = message;
    target.classList.toggle("hidden", !message);
  }

  function syncButtons() {
    if (!rendered) return;
    const dirty = Boolean(specs.get(activeKind)?.dirty);
    el("specValidateBtn").disabled = loading;
    el("specSaveBtn").disabled = loading;
    el("specSaveBtn").classList.toggle("dirty", dirty);
  }

  function showNotice(result, isError = false) {
    const notice = el("specNotice");
    if (!notice) return;
    if (!result) {
      notice.textContent = "";
      notice.title = "";
      notice.classList.add("hidden");
      notice.classList.remove("error", "warning");
      return;
    }
    const errors = Array.isArray(result.errors) ? result.errors : [];
    const warnings = Array.isArray(result.warnings) ? result.warnings : [];
    const items = [...errors, ...warnings].slice(0, 8);
    notice.classList.toggle("error", isError || errors.length > 0 || result.valid === false);
    notice.classList.toggle("warning", !errors.length && warnings.length > 0);
    notice.classList.remove("hidden");
    const messageParts = [];
    [result.message || "", ...items].forEach((item) => {
      if (item && !messageParts.includes(item)) messageParts.push(item);
    });
    const message = messageParts.join("; ");
    notice.textContent = message;
    notice.title = message;
  }

  function showErrorNotice(message) {
    showNotice({ valid: false, errors: [message], warnings: [], message }, true);
  }

  function showValidationNotice(result, options = {}) {
    if (!result) {
      showNotice(null);
      return;
    }
    const errors = Array.isArray(result.errors) ? result.errors : [];
    const warnings = Array.isArray(result.warnings) ? result.warnings : [];
    if (options.showValid || errors.length || warnings.length || result.valid === false) {
      showNotice(result);
    } else {
      showNotice(null);
    }
  }

  function saveActiveDraft() {
    const spec = specs.get(activeKind);
    if (!spec || !table) return spec;
    spec.rows = rowsWithIds(table.getData().map((row) => cleanRow(row, spec.columns)), spec.columns);
    return spec;
  }

  function activePayload() {
    const spec = saveActiveDraft();
    if (!spec) return null;
    return {
      columns: spec.columns,
      rows: rowsForApi(spec),
    };
  }

  function rowsForApi(spec) {
    return spec.rows.map((row) => {
      const next = {};
      spec.columns.forEach((column) => {
        const value = row[column] ?? "";
        next[column] = spec.kind === "feature" && isScenarioField(column, spec)
          ? normaliseScenarioValue(value)
          : String(value);
      });
      return next;
    });
  }

  async function validateCurrentSpec() {
    const payload = activePayload();
    if (!payload) return;
    loading = true;
    syncButtons();
    try {
      const result = await api(`/api/specs/${activeKind}/validate`, {
        method: "POST",
        body: JSON.stringify(payload),
      });
      storeValidationResult(specs.get(activeKind), result, { showValid: true });
    } catch (error) {
      showNotice({ valid: false, errors: [error.message], warnings: [], message: error.message }, true);
    } finally {
      loading = false;
      syncButtons();
    }
  }

  async function saveCurrentSpec() {
    const payload = activePayload();
    if (!payload) return;
    loading = true;
    syncButtons();
    try {
      const result = await api(`/api/specs/${activeKind}/save`, {
        method: "POST",
        body: JSON.stringify(payload),
      });
      const spec = cacheSpec(result.spec, false);
      renderSpec(spec);
      await reloadSchemaAfterSpecsSave(activeKind);
      showNotice(result);
      showClipboardToast(`${result.message}: ${result.path || spec.path}`);
    } catch (error) {
      showNotice({ valid: false, errors: [error.message], warnings: [], message: error.message }, true);
    } finally {
      loading = false;
      syncButtons();
    }
  }

  function markDirty() {
    if (suppressDirty) return;
    const spec = specs.get(activeKind);
    if (spec) spec.dirty = true;
    syncButtons();
  }

  function normaliseRowIssues(rowIssues) {
    if (!Array.isArray(rowIssues)) return [];
    return rowIssues
      .map((issue) => ({
        rowNumber: Number(issue?.row_number),
        severity: String(issue?.severity || ""),
        message: String(issue?.message || ""),
      }))
      .filter((issue) => Number.isFinite(issue.rowNumber) && issue.rowNumber >= 2);
  }

  async function validateSpecOnLoad(spec) {
    if (!spec || spec.generated || spec.dirty || spec.autoValidated) return;
    await validateSpecDraft(spec, { showValid: false });
  }

  async function validateSpecDraft(spec, options = {}) {
    if (!spec) return;
    try {
      const result = await api(`/api/specs/${spec.kind}/validate`, {
        method: "POST",
        body: JSON.stringify(payloadForSpec(spec)),
      });
      storeValidationResult(spec, result, options);
    } catch (error) {
      const result = { valid: false, errors: [error.message], warnings: [], row_issues: [], message: error.message };
      storeValidationResult(spec, result, options);
    }
  }

  function payloadForSpec(spec) {
    return {
      columns: spec.columns,
      rows: rowsForApi(spec),
    };
  }

  function storeValidationResult(spec, result, options = {}) {
    if (!spec || !result) return;
    spec.validationResult = result;
    spec.autoValidated = true;
    applyValidationResultRowIssues(result, spec);
    if (spec.kind === activeKind) showValidationNotice(result, options);
  }

  function applyValidationResultRowIssues(result, spec = specs.get(activeKind)) {
    if (!spec) return;
    spec.rowIssues = normaliseRowIssues(result?.row_issues);
    if (spec.kind === activeKind) applyValidationRowIssueClasses();
  }

  function clearValidationRowIssuesForActiveSpec() {
    const spec = specs.get(activeKind);
    clearValidationRowIssuesForSpec(spec);
  }

  function clearValidationRowIssuesForSpec(spec) {
    if (!spec) return;
    const hadValidationState = Boolean(spec.validationResult || spec.autoValidated || (Array.isArray(spec.rowIssues) && spec.rowIssues.length));
    if (!hadValidationState) return;
    if (["kpi", "filter"].includes(spec.kind)) spec.rowIssues = [];
    spec.validationResult = null;
    spec.autoValidated = false;
    if (spec.kind === activeKind) showNotice(null);
    applyValidationRowIssueClasses();
  }

  function specToolVisible() {
    return rendered && !el("specificationsWrap")?.classList.contains("hidden");
  }

  function isEditableTarget(target) {
    return Boolean(target?.closest?.("input, textarea, select, [contenteditable='true']"));
  }

  function specGridOwnsKeyboardEvent(event) {
    const grid = el("specGrid");
    if (!grid) return false;
    const target = event?.target;
    const active = document.activeElement;
    return Boolean(
      (target?.nodeType && grid.contains(target))
      || active === grid
      || (active?.nodeType && grid.contains(active)),
    );
  }

  function clearNativeSelection(event) {
    if (isEditableTarget(event?.target)) return;
    window.getSelection?.()?.removeAllRanges();
  }

  function handleSpecCellClick(event, cell) {
    if (toggleScenarioCheckbox(event, cell)) return;
    clearNativeSelection(event);
  }

  function handleSpecCellDblClick(event, cell) {
    endSelectionDrag();
    if (!isScenarioField(cell?.getField?.())) return;
    event.preventDefault();
    clearNativeSelection(event);
  }

  function toggleScenarioCheckbox(event, cell) {
    const checkbox = event.target?.closest?.(".spec-checkbox-cell");
    if (!checkbox || !isScenarioField(cell?.getField?.())) return false;
    event.preventDefault();
    event.stopPropagation();
    const point = selectionPointForCell(cell);
    if (!point) return true;
    const nextValue = scenarioCellSelected(cell.getValue?.()) ? "" : "feature";
    selection = { anchor: point, focus: point, active: point };
    el("specGrid")?.focus?.({ preventScroll: true });
    applyCellUpdates([{ rowId: point.rowId, field: point.field, value: nextValue }]);
    clearNativeSelection();
    return true;
  }

  function resetSelection() {
    clearSelectionClasses();
    selection = null;
    selectionDragging = false;
    el("specGrid")?.classList.remove("spec-selecting");
  }

  function endSelectionDrag() {
    selectionDragging = false;
    el("specGrid")?.classList.remove("spec-selecting");
    clearNativeSelection();
  }

  function startSelectionFromCell(event, cell) {
    if (!specToolVisible() || event.button !== 0 || isEditableTarget(event.target)) return;
    const point = selectionPointForCell(cell);
    if (!point) return;
    event.preventDefault();
    closeMenus();
    clearNativeSelection(event);
    el("specGrid")?.focus?.({ preventScroll: true });
    const anchor = event.shiftKey && selection?.anchor ? selection.anchor : point;
    const active = event.shiftKey && selection?.active ? selection.active : anchor;
    selection = { anchor, focus: point, active };
    selectionDragging = true;
    el("specGrid")?.classList.add("spec-selecting");
    applySelectionClasses();
  }

  function extendSelectionToCell(event, cell) {
    if (!selectionDragging || !selection || isEditableTarget(event.target)) return;
    const point = selectionPointForCell(cell);
    if (!point) return;
    event.preventDefault();
    clearNativeSelection(event);
    selection.focus = point;
    applySelectionClasses();
  }

  function selectionPointForCell(cell) {
    const field = cell?.getField?.();
    const data = cell?.getRow?.()?.getData?.();
    if (!field || !data?._row_id) return null;
    const spec = specs.get(activeKind);
    if (!spec?.columns.includes(field)) return null;
    return { rowId: String(data._row_id), field };
  }

  function displayedRows() {
    if (!table) return [];
    try {
      const activeRows = table.getRows?.("active");
      if (Array.isArray(activeRows) && activeRows.length) return activeRows;
    } catch (_) {
      // Fall back to the default row order for older Tabulator builds.
    }
    try {
      return table.getRows?.() || [];
    } catch (_) {
      return [];
    }
  }

  function displayedRowIds() {
    const rows = displayedRows()
      .map((row) => String(row.getData?.()?._row_id || ""))
      .filter(Boolean);
    if (rows.length) return rows;
    return (specs.get(activeKind)?.rows || []).map((row) => String(row._row_id || "")).filter(Boolean);
  }

  function navigationGrid() {
    const rows = displayedRows();
    const rowIds = rows
      .map((row) => String(row.getData?.()?._row_id || ""))
      .filter(Boolean);
    const columns = specs.get(activeKind)?.columns || [];
    return { rows, rowIds, columns };
  }

  function pointInGrid(point, rowIds, columns) {
    return Boolean(point && rowIds.includes(point.rowId) && columns.includes(point.field));
  }

  function selectionPointIndexes(point, rowIds, columns) {
    return {
      row: rowIds.indexOf(point?.rowId),
      column: columns.indexOf(point?.field),
    };
  }

  function moveSelectionWithArrow(key, extend) {
    const { rows, rowIds, columns } = navigationGrid();
    if (!rowIds.length || !columns.length) return;
    const currentPoint = extend && pointInGrid(selection?.focus, rowIds, columns) ? selection.focus : selection?.active;
    const current = pointInGrid(currentPoint, rowIds, columns)
      ? currentPoint
      : { rowId: rowIds[0], field: columns[0] };
    const currentIndex = selectionPointIndexes(current, rowIds, columns);
    const delta = {
      ArrowUp: [-1, 0],
      ArrowDown: [1, 0],
      ArrowLeft: [0, -1],
      ArrowRight: [0, 1],
    }[key] || [0, 0];
    const nextRow = Math.max(0, Math.min(rowIds.length - 1, currentIndex.row + delta[0]));
    const nextColumn = Math.max(0, Math.min(columns.length - 1, currentIndex.column + delta[1]));
    const next = { rowId: rowIds[nextRow], field: columns[nextColumn] };
    const anchor = extend && pointInGrid(selection?.anchor, rowIds, columns) ? selection.anchor : next;
    const active = extend && pointInGrid(selection?.active, rowIds, columns) ? selection.active : next;
    selection = { anchor, focus: next, active };
    clearNativeSelection();
    applySelectionClasses();
    scrollSelectionPointIntoView(next, rows);
  }

  function scrollSelectionPointIntoView(point, rows = displayedRows()) {
    const row = rows.find((entry) => String(entry.getData?.()?._row_id || "") === point?.rowId);
    const cell = row?.getCell?.(point?.field);
    cell?.getElement?.()?.scrollIntoView?.({ block: "nearest", inline: "nearest" });
  }

  function activeSingleCell() {
    const bounds = selectionBounds();
    if (!bounds || bounds.top !== bounds.bottom || bounds.left !== bounds.right) return null;
    return {
      rowId: bounds.rowIds[bounds.top],
      field: bounds.columns[bounds.left],
    };
  }

  function isPrintableEditKey(event) {
    return event.key.length === 1 && !event.metaKey && !event.ctrlKey && !event.altKey;
  }

  function startEditingActiveCell(initialText) {
    const point = activeSingleCell();
    if (!point || isScenarioField(point.field)) return false;
    try {
      const cell = table?.getRow?.(point.rowId)?.getCell?.(point.field);
      if (!cell?.edit) return false;
      cell.edit(true);
      if (!setEditorInitialText(cell, initialText)) {
        window.requestAnimationFrame(() => {
          setEditorInitialText(cell, initialText);
        });
      }
      return true;
    } catch (_) {
      return false;
    }
  }

  function setEditorInitialText(cell, initialText) {
    const input = cell.getElement?.()?.querySelector?.("input, textarea");
    if (!input) return false;
    input.value = initialText;
    input.dispatchEvent(new Event("input", { bubbles: true }));
    input.focus?.();
    input.setSelectionRange?.(initialText.length, initialText.length);
    return true;
  }

  function restoreSelectionAfterCellEdit(cell) {
    const point = selectionPointForCell(cell);
    if (point) {
      selection = { anchor: point, focus: point, active: point };
    }
    window.setTimeout(() => {
      if (!specToolVisible() || isEditableTarget(document.activeElement)) return;
      el("specGrid")?.focus?.({ preventScroll: true });
      applySelectionClasses();
      if (point) scrollSelectionPointIntoView(point);
    }, 0);
  }

  function selectionBounds() {
    if (!selection) return null;
    const rowIds = displayedRowIds();
    const columns = specs.get(activeKind)?.columns || [];
    const anchorRow = rowIds.indexOf(selection.anchor.rowId);
    const focusRow = rowIds.indexOf(selection.focus.rowId);
    const anchorCol = columns.indexOf(selection.anchor.field);
    const focusCol = columns.indexOf(selection.focus.field);
    if (anchorRow < 0 || focusRow < 0 || anchorCol < 0 || focusCol < 0) return null;
    return {
      rowIds,
      columns,
      top: Math.min(anchorRow, focusRow),
      bottom: Math.max(anchorRow, focusRow),
      left: Math.min(anchorCol, focusCol),
      right: Math.max(anchorCol, focusCol),
    };
  }

  function clearSelectionClasses() {
    const target = el("specGrid");
    if (!target) return;
    target.querySelectorAll(".spec-cell-selected, .spec-cell-active").forEach((cell) => {
      cell.classList.remove("spec-cell-selected", "spec-cell-active");
    });
  }

  function applyTableDecorations() {
    applyMissingFeatureRowClasses();
    applyValidationRowIssueClasses();
    applySelectionClasses();
  }

  function datasetFeatureNameSet() {
    try {
      return new Set((datasetColumnNames() || []).map((name) => String(name || "")).filter(Boolean));
    } catch (_) {
      return new Set();
    }
  }

  function featureRowMissingDatasetFeature(rowData, spec = specs.get(activeKind), datasetNames = datasetFeatureNameSet()) {
    const feature = String(rowData?.Feature || "").trim();
    return Boolean(spec?.kind === "feature" && feature && !datasetNames.has(feature));
  }

  function applyMissingFeatureRowClasses() {
    const spec = specs.get(activeKind);
    const datasetNames = datasetFeatureNameSet();
    displayedRows().forEach((row) => {
      row.getElement?.()?.classList.toggle("spec-missing-feature-row", featureRowMissingDatasetFeature(row.getData?.(), spec, datasetNames));
    });
  }

  function validationRowIssueNumberSet(spec = specs.get(activeKind)) {
    if (!spec || !["kpi", "filter"].includes(spec.kind)) return new Set();
    return new Set((spec.rowIssues || []).map((issue) => Number(issue.rowNumber)).filter(Number.isFinite));
  }

  function rowHasValidationIssue(rowData, spec = specs.get(activeKind), issueNumbers = validationRowIssueNumberSet(spec)) {
    return Boolean(["kpi", "filter"].includes(spec?.kind) && issueNumbers.has(Number(rowData?._spec_row_number)));
  }

  function applyValidationRowIssueClasses() {
    const spec = specs.get(activeKind);
    const issueNumbers = validationRowIssueNumberSet(spec);
    displayedRows().forEach((row) => {
      row.getElement?.()?.classList.toggle("spec-validation-issue-row", rowHasValidationIssue(row.getData?.(), spec, issueNumbers));
    });
  }

  function applySelectionClasses() {
    clearSelectionClasses();
    const bounds = selectionBounds();
    if (!bounds || !table) return;
    displayedRows().forEach((row) => {
      const rowId = String(row.getData?.()?._row_id || "");
      const rowIndex = bounds.rowIds.indexOf(rowId);
      if (rowIndex < bounds.top || rowIndex > bounds.bottom) return;
      row.getCells?.().forEach((cell) => {
        const field = cell.getField?.();
        const colIndex = bounds.columns.indexOf(field);
        if (colIndex < bounds.left || colIndex > bounds.right) return;
        const element = cell.getElement?.();
        if (!element) return;
        element.classList.add("spec-cell-selected");
        if (selection.active?.rowId === rowId && selection.active?.field === field) {
          element.classList.add("spec-cell-active");
        }
      });
    });
  }

  function selectedTsv() {
    const bounds = selectionBounds();
    if (!bounds) return null;
    const lines = [];
    for (let rowIndex = bounds.top; rowIndex <= bounds.bottom; rowIndex += 1) {
      const row = rowDataById(bounds.rowIds[rowIndex]);
      const values = [];
      for (let colIndex = bounds.left; colIndex <= bounds.right; colIndex += 1) {
        const field = bounds.columns[colIndex];
        values.push(storedCellValue(field, row?.[field]));
      }
      lines.push(values.join("\t"));
    }
    return lines.join("\n");
  }

  function rowDataById(rowId) {
    try {
      return table?.getRow?.(rowId)?.getData?.() || null;
    } catch (_) {
      return (specs.get(activeKind)?.rows || []).find((row) => String(row._row_id || "") === String(rowId)) || null;
    }
  }

  function parseClipboardText(text) {
    const normalised = String(text ?? "").replace(/\r\n/g, "\n").replace(/\r/g, "\n");
    const lines = normalised.split("\n");
    if (lines.length > 1 && lines[lines.length - 1] === "") lines.pop();
    return lines.map((line) => line.split("\t"));
  }

  function storedCellValue(field, value) {
    const spec = specs.get(activeKind);
    if (spec?.kind === "feature" && isScenarioField(field, spec)) {
      return normaliseScenarioValue(value);
    }
    return String(value ?? "");
  }

  function applyCellUpdates(updates) {
    const normalisedUpdates = updates.map((update) => ({
      ...update,
      value: storedCellValue(update.field, update.value),
    }));
    const changed = normalisedUpdates.filter((update) => rowDataById(update.rowId)?.[update.field] !== update.value);
    if (!changed.length) return;
    clearValidationRowIssuesForActiveSpec();
    const scenarioChanged = changed.some((update) => isScenarioField(update.field));
    const scrollPosition = scenarioChanged ? captureSpecTableScroll() : null;
    suppressDirty = true;
    try {
      changed.forEach(({ rowId, field, value }) => {
        const nextValue = String(value ?? "");
        try {
          const row = table.getRow(rowId);
          const cell = row?.getCell?.(field);
          if (cell) {
            cell.setValue(nextValue, false);
          } else {
            const data = row?.getData?.();
            if (data) data[field] = nextValue;
          }
        } catch (_) {
          const row = rowDataById(rowId);
          if (row) row[field] = nextValue;
        }
      });
    } finally {
      suppressDirty = false;
    }
    markDirty();
    if (scenarioChanged) {
      restoreSpecTableScroll(scrollPosition);
      refreshScenarioHeaderCounts();
    }
    window.requestAnimationFrame(() => {
      applyTableDecorations();
      if (scenarioChanged) restoreSpecTableScroll(scrollPosition);
    });
  }

  function clearSelectedCells() {
    const bounds = selectionBounds();
    if (!bounds) return;
    const updates = [];
    for (let rowIndex = bounds.top; rowIndex <= bounds.bottom; rowIndex += 1) {
      for (let colIndex = bounds.left; colIndex <= bounds.right; colIndex += 1) {
        updates.push({ rowId: bounds.rowIds[rowIndex], field: bounds.columns[colIndex], value: "" });
      }
    }
    applyCellUpdates(updates);
  }

  function pasteTextIntoSelection(text) {
    const bounds = selectionBounds();
    if (!bounds) return;
    const rows = parseClipboardText(text);
    const updates = [];
    let focus = null;
    rows.forEach((values, rowOffset) => {
      const rowIndex = bounds.top + rowOffset;
      if (rowIndex >= bounds.rowIds.length) return;
      values.forEach((value, colOffset) => {
        const colIndex = bounds.left + colOffset;
        if (colIndex >= bounds.columns.length) return;
        focus = { rowId: bounds.rowIds[rowIndex], field: bounds.columns[colIndex] };
        updates.push({ ...focus, value });
      });
    });
    if (!updates.length) return;
    const origin = { rowId: bounds.rowIds[bounds.top], field: bounds.columns[bounds.left] };
    applyCellUpdates(updates);
    selection = { anchor: origin, focus, active: origin };
    window.requestAnimationFrame(applyTableDecorations);
  }

  async function copySelectionToSystemClipboard() {
    const text = selectedTsv();
    if (text === null) return;
    try {
      await navigator.clipboard?.writeText?.(text);
      showClipboardToast("Copied selection");
    } catch (error) {
      showErrorNotice(`Copy failed: ${error.message}`);
    }
  }

  async function pasteSelectionFromSystemClipboard() {
    try {
      const text = await navigator.clipboard?.readText?.();
      if (text !== undefined) pasteTextIntoSelection(text);
    } catch (error) {
      showErrorNotice(`Paste failed: ${error.message}`);
    }
  }

  function handleSpecCopy(event) {
    if (!specToolVisible() || isEditableTarget(event.target) || !selection) return;
    const text = selectedTsv();
    if (text === null) return;
    event.preventDefault();
    event.clipboardData?.setData("text/plain", text);
  }

  function handleSpecPaste(event) {
    if (!specToolVisible() || isEditableTarget(event.target) || !selection) return;
    const text = event.clipboardData?.getData("text/plain");
    if (text === undefined) return;
    event.preventDefault();
    pasteTextIntoSelection(text);
  }

  function handleSpecKeydown(event) {
    if (!specToolVisible() || isEditableTarget(event.target)) return false;
    if (!specGridOwnsKeyboardEvent(event)) return false;
    const key = event.key.toLowerCase();
    const shortcut = event.metaKey || event.ctrlKey;
    if (["ArrowUp", "ArrowDown", "ArrowLeft", "ArrowRight"].includes(event.key) && !shortcut && !event.altKey) {
      event.preventDefault();
      moveSelectionWithArrow(event.key, event.shiftKey);
      return true;
    }
    if (!selection) return false;
    if (isPrintableEditKey(event)) {
      if (!startEditingActiveCell(event.key)) return false;
      event.preventDefault();
      return true;
    }
    if (shortcut && key === "c") {
      event.preventDefault();
      void copySelectionToSystemClipboard();
      return true;
    }
    if (shortcut && key === "v") {
      event.preventDefault();
      void pasteSelectionFromSystemClipboard();
      return true;
    }
    if (!shortcut && (event.key === "Delete" || event.key === "Backspace")) {
      event.preventDefault();
      clearSelectedCells();
      return true;
    }
    return false;
  }

  function rowsWithIds(rows, columns) {
    return rows.map((row, index) => {
      const next = cleanRow(row, columns);
      next._row_id = row._row_id || `spec-row-${++rowIdCounter}`;
      next._spec_row_number = index + 2;
      return next;
    });
  }

  function renumberSpecRows(spec) {
    if (!spec) return;
    spec.rows = rowsWithIds(Array.isArray(spec.rows) ? spec.rows : [], spec.columns || []);
  }

  function cleanRow(row, columns) {
    const next = {};
    columns.forEach((column) => {
      next[column] = String(row?.[column] ?? "");
    });
    if (row?._row_id) next._row_id = row._row_id;
    return next;
  }

  function scenarioFields(spec = specs.get(activeKind)) {
    if (!spec || spec.kind !== "feature") return [];
    return spec.columns.slice(featureScenarioStart(spec.columns));
  }

  function isScenarioField(field, spec = specs.get(activeKind)) {
    return Boolean(spec?.kind === "feature" && scenarioFields(spec).includes(field));
  }

  function featureScenarioStart(columns) {
    let index = 2;
    const seen = new Set();
    while (index < columns.length) {
      const column = columns[index];
      if (!FEATURE_METADATA_COLUMNS.has(column) || seen.has(column)) break;
      seen.add(column);
      index += 1;
    }
    return index;
  }

  function addScenarioAt(referenceField = "", position = "end") {
    const spec = saveActiveDraft();
    if (!spec || spec.kind !== "feature") return;
    const promptValue = window.prompt("Scenario name", nextScenarioName(spec));
    if (promptValue === null) return;
    const name = uniqueScenarioName(promptValue, spec);
    if (!name) return;
    let insertAt = spec.columns.length;
    if ((position === "before" || position === "after") && isScenarioField(referenceField, spec)) {
      const referenceIndex = spec.columns.indexOf(referenceField);
      if (referenceIndex >= 0) insertAt = referenceIndex + (position === "after" ? 1 : 0);
    }
    spec.columns.splice(insertAt, 0, name);
    spec.rows.forEach((row) => {
      row[name] = "";
    });
    spec.dirty = true;
    renderSpec(spec);
  }

  function renameScenarioField(oldName) {
    const spec = saveActiveDraft();
    if (!spec || !isScenarioField(oldName, spec)) return;
    const promptValue = window.prompt("Scenario name", oldName);
    if (promptValue === null) return;
    const nextName = uniqueScenarioName(promptValue, spec, oldName);
    if (!nextName || nextName === oldName) return;
    spec.columns = spec.columns.map((column) => column === oldName ? nextName : column);
    spec.rows.forEach((row) => {
      row[nextName] = row[oldName] || "";
      delete row[oldName];
    });
    spec.dirty = true;
    renderSpec(spec);
  }

  function deleteScenarioField(name) {
    const spec = saveActiveDraft();
    if (!spec || !isScenarioField(name, spec)) return;
    spec.columns = spec.columns.filter((column) => column !== name);
    spec.rows.forEach((row) => delete row[name]);
    spec.dirty = true;
    renderSpec(spec);
  }

  function nextScenarioName(spec) {
    let index = scenarioFields(spec).length + 1;
    while (spec.columns.includes(`scenario${index}`)) index += 1;
    return `scenario${index}`;
  }

  function uniqueScenarioName(rawName, spec, existing = "") {
    const name = String(rawName || "").trim();
    if (!name || name === existing) return name;
    if (["Feature", "Grouping", "Base", "min", "max", "banding"].includes(name)) {
      showErrorNotice(`Scenario name is reserved: ${name}`);
      return "";
    }
    if (spec.columns.includes(name)) {
      showErrorNotice(`Scenario already exists: ${name}`);
      return "";
    }
    return name;
  }

  function openRowContextMenu(event, row) {
    event.preventDefault();
    contextRowId = row.getData()?._row_id || "";
    const menu = el("specContextMenu");
    el("specColumnContextMenu").hidden = true;
    positionContextMenu(menu, event);
  }

  function openColumnContextMenu(event, column) {
    event.preventDefault();
    const spec = specs.get(activeKind);
    const field = column?.getField?.() || "";
    if (!spec || spec.kind !== "feature" || !field) {
      closeMenus();
      return;
    }
    contextColumnField = field;
    const scenario = isScenarioField(field, spec);
    const actions = scenario
      ? [
          ["add-before", "Add scenario before"],
          ["add-after", "Add scenario after"],
          ["delete", "Delete scenario"],
          ["rename", "Rename scenario"],
        ]
      : [["add-end", "Add scenario"]];
    const menu = el("specColumnContextMenu");
    menu.innerHTML = actions.map(([action, label]) => (
      `<button class="spec-context-menu-item" type="button" role="menuitem" data-spec-column-action="${action}">${escapeHtml(label)}</button>`
    )).join("");
    el("specContextMenu").hidden = true;
    positionContextMenu(menu, event);
  }

  function positionContextMenu(menu, event) {
    if (!menu) return;
    menu.hidden = false;
    const maxLeft = window.innerWidth - menu.offsetWidth - 8;
    const maxTop = window.innerHeight - menu.offsetHeight - 8;
    menu.style.left = `${Math.max(8, Math.min(event.clientX, maxLeft))}px`;
    menu.style.top = `${Math.max(8, Math.min(event.clientY, maxTop))}px`;
  }

  function closeMenus() {
    const rowMenu = el("specContextMenu");
    const columnMenu = el("specColumnContextMenu");
    if (rowMenu) rowMenu.hidden = true;
    if (columnMenu) columnMenu.hidden = true;
    contextColumnField = "";
  }

  function addRowRelative(position) {
    const spec = saveActiveDraft();
    if (!spec || !contextRowId) return;
    const rows = spec.rows.slice();
    const index = rows.findIndex((row) => row._row_id === contextRowId);
    const insertAt = index < 0 ? rows.length : index + (position === "below" ? 1 : 0);
    rows.splice(insertAt, 0, newRow(spec));
    spec.rows = rows;
    clearValidationRowIssuesForSpec(spec);
    spec.dirty = true;
    renderSpec(spec);
  }

  async function deleteContextRow() {
    const spec = saveActiveDraft();
    if (!spec || !contextRowId) return;
    spec.rows = spec.rows.filter((row) => row._row_id !== contextRowId);
    spec.dirty = true;
    renderSpec(spec, { preserveScroll: true });
    await validateSpecDraft(spec, { showValid: false });
  }

  function newRow(spec) {
    const row = {};
    spec.columns.forEach((column) => {
      row[column] = "";
    });
    row[spec.columns[0]] = nextNewRowLabel(spec);
    row._row_id = `spec-row-${++rowIdCounter}`;
    return row;
  }

  function nextNewRowLabel(spec) {
    const firstColumn = spec.columns[0];
    const existing = new Set(spec.rows.map((row) => String(row[firstColumn] || "")));
    let index = 1;
    while (existing.has(`New row ${index}`)) index += 1;
    return `New row ${index}`;
  }

  return {
    activate,
    closeMenus,
    refresh,
    refreshTheme,
    resize,
  };
}
