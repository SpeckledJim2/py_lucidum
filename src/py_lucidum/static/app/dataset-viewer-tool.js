import { loadTabulator } from "./shared/tabulator.js";

const TOOL_ID = "dataset_viewer";
const MAX_ROWS = 100;
const STYLESHEET_ID = "datasetViewerStylesheet";

function ensureDatasetViewerStyles() {
  if (document.getElementById(STYLESHEET_ID)) return;
  const link = document.createElement("link");
  link.id = STYLESHEET_ID;
  link.rel = "stylesheet";
  link.href = "/static/styles/dataset-viewer.css";
  document.head.appendChild(link);
}

export function createDatasetViewerTool({
  api,
  copyTextToClipboard,
  el,
  escapeHtml,
  measureToolRender,
  saveToolPresentation,
  setChartMessage,
  setStatus,
  setToolTimingFailed,
  showClipboardToast,
  stableRequestKey,
  startToolTiming,
  state,
  syncDatasetViewerMeta = () => {},
  syncClientTimingFromData,
  syncDuckDbTimingFromData,
  toolCache,
}) {
  ensureDatasetViewerStyles();

  let datasetTable = null;
  let renderToken = 0;
  let currentRows = [];
  let currentColumns = [];
  let currentFields = [];
  let currentDatasetRows = [];
  let currentDatasetColumns = [];
  let currentDatasetColumnByField = new Map();
  let currentVisibleDatasetColumns = [];
  let selectedRowIds = new Set();
  let selectedColumnFields = new Set();
  let selectionAxis = "";
  let suppressRowSelectionSync = false;
  let transposedSort = { field: "", dir: "none" };
  let renderedRequestKey = null;
  let renderedSearch = null;
  let renderedTranspose = null;
  let renderedAlphabeticalColumns = null;
  let resizeFrame = null;
  let resizeHard = false;

  function buildRequest() {
    return {
      filter: state.activeFilter,
      limit: MAX_ROWS,
    };
  }

  async function fetchData(request, requestKey) {
    const requestSeq = (state.datasetViewerRequestSeq || 0) + 1;
    state.datasetViewerRequestSeq = requestSeq;
    startToolTiming(TOOL_ID);
    setStatus("");
    setChartMessage("");
    renderLoading();
    try {
      const data = await api("/api/dataset-viewer/table", { method: "POST", body: JSON.stringify(request), clientTiming: true });
      if (requestSeq !== state.datasetViewerRequestSeq) return null;
      const cache = toolCache(TOOL_ID);
      cache.requestKey = requestKey;
      cache.data = data;
      syncDuckDbTimingFromData(TOOL_ID, data);
      syncClientTimingFromData(TOOL_ID, data);
      measureToolRender(TOOL_ID, () => renderData(data, requestKey));
      return data;
    } catch (error) {
      if (requestSeq !== state.datasetViewerRequestSeq) return null;
      setToolTimingFailed(TOOL_ID);
      renderError(error.message);
      setStatus(error.message, true);
      return null;
    }
  }

  function useCached(cache) {
    if (cacheIsRendered(cache)) {
      setStatus("");
      setChartMessage("");
      requestAnimationFrame(() => resize());
      return Promise.resolve(cache.data);
    }
    measureToolRender(TOOL_ID, () => renderData(cache.data, cache.requestKey));
    return Promise.resolve(cache.data);
  }

  function renderShell() {
    const wrap = el("datasetViewerWrap");
    if (!wrap) return null;
    if (!document.getElementById("datasetViewerGrid")) {
      wrap.innerHTML = `
        <div class="dataset-viewer-toolbar">
          <div class="dataset-viewer-search-row">
            <input id="datasetViewerSearch" class="search dataset-viewer-search" placeholder="search table" />
            <button id="datasetViewerSearchClear" class="filter-action" type="button" title="Clear table search" aria-label="Clear table search">&times;</button>
          </div>
          <label class="dataset-viewer-checkbox">
            <input id="datasetViewerTranspose" type="checkbox" />
            <span>Transpose</span>
          </label>
          <label class="dataset-viewer-checkbox">
            <input id="datasetViewerAlphabeticalColumns" type="checkbox" />
            <span>Alphabetical columns</span>
          </label>
          <div id="datasetViewerCount" class="dataset-viewer-count"></div>
          <div id="datasetViewerMeta" class="dataset-viewer-meta"></div>
        </div>
        <div id="datasetViewerGrid" class="dataset-viewer-grid"></div>`;
      attachDatasetViewerMeta();
      el("datasetViewerSearch").addEventListener("input", () => {
        state.datasetViewerSearch = el("datasetViewerSearch").value;
        applySearch();
      });
      el("datasetViewerSearchClear").addEventListener("click", () => {
        state.datasetViewerSearch = "";
        el("datasetViewerSearch").value = "";
        applySearch();
        el("datasetViewerSearch").focus();
      });
      el("datasetViewerTranspose").addEventListener("change", () => {
        state.datasetViewerTranspose = el("datasetViewerTranspose").checked;
        rerenderCachedData();
      });
      el("datasetViewerAlphabeticalColumns").addEventListener("change", () => {
        state.datasetViewerAlphabeticalColumns = el("datasetViewerAlphabeticalColumns").checked;
        rerenderCachedData();
      });
      el("datasetViewerGrid").addEventListener("click", handleDatasetViewerGridClick);
      el("datasetViewerGrid").addEventListener("contextmenu", handleDatasetViewerGridContextMenu);
    }
    attachDatasetViewerMeta();
    const search = el("datasetViewerSearch");
    if (document.activeElement !== search && search.value !== (state.datasetViewerSearch || "")) {
      search.value = state.datasetViewerSearch || "";
    }
    el("datasetViewerTranspose").checked = Boolean(state.datasetViewerTranspose);
    el("datasetViewerAlphabeticalColumns").checked = Boolean(state.datasetViewerAlphabeticalColumns);
    return wrap;
  }

  function attachDatasetViewerMeta() {
    const meta = document.getElementById("datasetViewerMeta");
    const groupMeta = document.getElementById("datasetViewerGroupMeta");
    const filter = document.getElementById("datasetViewerFilter");
    if (!meta || !groupMeta || !filter) return;
    if (groupMeta.parentElement !== meta) meta.append(groupMeta);
    if (filter.parentElement !== meta) meta.append(filter);
  }

  function syncTransposeControl(data) {
    const loadedRows = Array.isArray(data?.rows) ? data.rows.length : 0;
    const disabled = loadedRows > MAX_ROWS;
    const input = el("datasetViewerTranspose");
    if (!input) return;
    input.disabled = disabled;
    input.title = disabled ? `Transpose is available for ${MAX_ROWS.toLocaleString()} loaded rows or fewer` : "";
    if (disabled && state.datasetViewerTranspose) {
      state.datasetViewerTranspose = false;
      input.checked = false;
    }
  }

  function clearTable({ resetSelection = false } = {}) {
    closeDatasetViewerCellContextMenu();
    if (resizeFrame !== null) {
      cancelAnimationFrame(resizeFrame);
      resizeFrame = null;
      resizeHard = false;
    }
    if (datasetTable) {
      try {
        datasetTable.destroy();
      } catch (_) {
        // Tabulator may already have been removed by a stale render.
      }
    }
    datasetTable = null;
    if (resetSelection) {
      resetSelections();
      transposedSort = { field: "", dir: "none" };
    }
    currentRows = [];
    currentColumns = [];
    currentFields = [];
    currentDatasetRows = [];
    currentDatasetColumns = [];
    currentDatasetColumnByField = new Map();
    currentVisibleDatasetColumns = [];
    renderedRequestKey = null;
    renderedSearch = null;
    renderedTranspose = null;
    renderedAlphabeticalColumns = null;
  }

  function renderLoading() {
    const wrap = renderShell();
    if (!wrap) return;
    renderToken += 1;
    clearTable({ resetSelection: true });
    el("datasetViewerCount").textContent = "";
    el("datasetViewerGrid").classList.remove("dataset-viewer-grid-transposed");
    el("datasetViewerGrid").innerHTML = `<div class="dataset-viewer-state">Loading dataset...</div>`;
  }

  function renderError(message) {
    const wrap = renderShell();
    if (!wrap) return;
    renderToken += 1;
    clearTable({ resetSelection: true });
    el("datasetViewerCount").textContent = "";
    el("datasetViewerGrid").classList.remove("dataset-viewer-grid-transposed");
    el("datasetViewerGrid").innerHTML = `<div class="dataset-viewer-state dataset-viewer-state-error">${escapeHtml(message || "Dataset viewer query failed")}</div>`;
  }

  function renderData(data, requestKey = null) {
    const wrap = renderShell();
    if (!wrap) return;
    const token = renderToken + 1;
    renderToken = token;
    clearTable();
    syncTransposeControl(data);
    const tableData = Boolean(state.datasetViewerTranspose)
      ? transposedTableData(data)
      : normalTableData(data);
    currentRows = tableData.rows;
    currentColumns = tableData.columns;
    currentFields = tableData.fields;
    currentDatasetRows = tableData.datasetRows;
    currentDatasetColumns = tableData.datasetColumns;
    currentDatasetColumnByField = new Map(currentDatasetColumns.map((column) => [column.field, column]));
    currentVisibleDatasetColumns = tableData.visibleDatasetColumns || tableData.datasetColumns;
    state.datasetViewerColumnCount = Array.isArray(data?.columns) ? data.columns.length : null;
    syncDatasetViewerMeta();
    pruneSelectionsForCurrentData();
    const meta = countMeta(data);
    setStatus("");
    setChartMessage("");
    el("datasetViewerCount").textContent = meta;
    saveToolPresentation(TOOL_ID, {
      groupMeta: "",
      status: "",
      chartMessage: "",
    });
    el("datasetViewerGrid").innerHTML = "";
    el("datasetViewerGrid").classList.toggle("dataset-viewer-grid-transposed", Boolean(state.datasetViewerTranspose));
    if (state.datasetViewerTranspose) {
      renderTransposedGrid(token, requestKey);
      return;
    }
    loadTabulator().then((Tabulator) => {
      if (token !== renderToken) return;
      const target = el("datasetViewerGrid");
      if (!target) return;
      datasetTable = new Tabulator(target, {
        data: currentRows,
        index: "__row_id",
        autoResize: false,
        height: "100%",
        layout: "fitData",
        placeholder: "No matching rows",
        reactiveData: false,
        renderHorizontal: "virtual",
        renderVertical: "virtual",
        rowHeight: 22,
        selectableRows: true,
        selectableRowsPersistence: true,
        headerSortClickElement: "icon",
        columnDefaults: {
          headerSort: true,
          headerSortTristate: true,
          resizable: true,
          formatter: (cell) => escapeHtml(formatCellValue(cell.getValue())),
        },
        columns: currentColumns,
      });
      if (typeof datasetTable.on === "function") {
        datasetTable.on("rowSelectionChanged", handleNormalRowSelectionChanged);
        datasetTable.on("renderComplete", syncNormalRenderedSelection);
        datasetTable.on("dataSorted", syncNormalRenderedSelection);
        datasetTable.on("dataFiltered", syncNormalRenderedSelection);
      }
      applySearch();
      syncNormalRenderedSelection();
      markRendered(requestKey);
    }).catch((error) => {
      if (token !== renderToken) return;
      renderError(error.message || String(error));
    });
  }

  function renderTransposedGrid(token, requestKey) {
    loadTabulator().then((Tabulator) => {
      if (token !== renderToken) return;
      const target = el("datasetViewerGrid");
      if (!target) return;
      datasetTable = new Tabulator(target, {
        data: currentRows,
        index: "__row_id",
        autoResize: false,
        height: "100%",
        layout: "fitData",
        placeholder: "No matching columns",
        reactiveData: false,
        renderHorizontal: "virtual",
        renderVertical: "virtual",
        rowHeight: 22,
        selectableRows: false,
        columnDefaults: {
          headerSort: false,
          resizable: true,
          formatter: formatTransposedCell,
        },
        rowFormatter: formatTransposedRow,
        columns: currentColumns,
      });
      if (typeof datasetTable.on === "function") {
        datasetTable.on("renderComplete", syncTransposedRenderedSelection);
        datasetTable.on("dataFiltered", syncTransposedRenderedSelection);
      }
      applySearch();
      syncTransposedRenderedSelection();
      markRendered(requestKey);
    }).catch((error) => {
      if (token !== renderToken) return;
      renderError(error.message || String(error));
    });
  }

  function normalTableData(data) {
    const sourceColumns = orderedDatasetViewerColumns(data);
    const rows = Array.isArray(data?.rows) ? data.rows : [];
    const columns = sourceColumns.map((column) => ({
      title: normalHeaderHtml(column),
      copyTitle: column.name,
      name: column.name,
      field: column.field,
      headerTooltip: column.name,
      ...datasetViewerColumnWidth(column),
      hozAlign: column.kind === "numeric" || column.kind === "integer" ? "right" : "left",
      headerHozAlign: column.kind === "numeric" || column.kind === "integer" ? "right" : "left",
    }));
    return {
      columns,
      rows,
      fields: sourceColumns.map((column) => column.field),
      datasetRows: rows,
      datasetColumns: sourceColumns,
      visibleDatasetColumns: sourceColumns,
    };
  }

  function datasetViewerColumnWidth(column) {
    const kind = String(column?.kind || "");
    if (kind === "integer" || kind === "numeric") return { width: 96, minWidth: 72 };
    if (kind === "date" || kind === "datetime") return { width: 128, minWidth: 104 };
    return { width: 150, minWidth: 96 };
  }

  function transposedTableData(data) {
    const sourceColumns = orderedDatasetViewerColumns(data);
    const sourceRows = Array.isArray(data?.rows) ? data.rows : [];
    const sortedColumns = sortedTransposedColumns(sourceColumns, sourceRows);
    const rows = sortedColumns.map((column) => ({
      __row_id: column.field,
      __column_field: column.field,
      __field: column.name,
    }));
    const columns = [
      {
        title: transposedHeaderHtml({
          title: "Column",
          copyTitle: "Column",
          field: "__field",
          sortField: "__field",
        }),
        copyTitle: "Column",
        field: "__field",
        sortField: "__field",
        width: 220,
        minWidth: 170,
        headerSort: false,
        resizable: true,
      },
      ...sourceRows.map((row, index) => ({
        title: transposedHeaderHtml({
          title: `Row ${index + 1}`,
          copyTitle: `Row ${index + 1}`,
          field: `r${index}`,
          sortField: `r${index}`,
          datasetRowId: row.__row_id,
          headerTooltip: row.__row_id ? `Dataset row ${row.__row_id}` : `Row ${index + 1}`,
        }),
        copyTitle: `Row ${index + 1}`,
        field: `r${index}`,
        sortField: `r${index}`,
        datasetRowId: row.__row_id,
        headerTooltip: row.__row_id ? `Dataset row ${row.__row_id}` : `Row ${index + 1}`,
        headerSort: false,
        resizable: true,
        width: 100,
        minWidth: 72,
      })),
    ];
    return {
      columns,
      rows,
      fields: ["__field", ...sourceRows.map((_, index) => `r${index}`)],
      datasetRows: sourceRows,
      datasetColumns: sourceColumns,
      visibleDatasetColumns: sortedColumns,
    };
  }

  function normalHeaderHtml(column) {
    return `<button class="dataset-viewer-header-label" type="button" data-dataset-viewer-column-field="${escapeHtml(column.field)}" title="${escapeHtml(column.name)}">${escapeHtml(column.name)}</button>`;
  }

  function transposedHeaderHtml(column) {
    const title = column.copyTitle || column.title || column.field;
    const sortDir = transposedSort.field === column.sortField ? transposedSort.dir : "none";
    const selected = column.datasetRowId !== undefined && selectedRowIds.has(Number(column.datasetRowId));
    const label = column.datasetRowId !== undefined
      ? `<button class="dataset-viewer-transposed-header-label" type="button" data-dataset-viewer-row-id="${escapeHtml(String(column.datasetRowId))}" title="${escapeHtml(column.headerTooltip || title)}">${escapeHtml(title)}</button>`
      : `<span class="dataset-viewer-transposed-header-label-static">${escapeHtml(title)}</span>`;
    return `<div class="dataset-viewer-transposed-header-content" data-dataset-viewer-transposed-field="${escapeHtml(column.sortField || column.field)}" data-sort-dir="${escapeHtml(sortDir)}" aria-sort="${transposedAriaSort(sortDir)}"${selected ? ' data-selected-row="true"' : ""}>
        ${label}
        <button class="dataset-viewer-transposed-sort-button" type="button" data-dataset-viewer-transposed-sort="${escapeHtml(column.sortField || column.field)}" data-sort-dir="${escapeHtml(sortDir)}" aria-label="Sort ${escapeHtml(title)}"></button>
      </div>`;
  }

  function formatTransposedRow(row) {
    const data = typeof row?.getData === "function" ? row.getData() : {};
    const element = typeof row?.getElement === "function" ? row.getElement() : null;
    if (!element) return;
    const field = String(data.__column_field || "");
    if (field) {
      element.dataset.datasetViewerColumnField = field;
    } else {
      delete element.dataset.datasetViewerColumnField;
    }
    element.classList.toggle("tabulator-selected", selectionAxis === "columns" && selectedColumnFields.has(field));
  }

  function formatTransposedCell(cell) {
    const rowData = typeof cell?.getData === "function" ? cell.getData() : {};
    const field = typeof cell?.getField === "function" ? cell.getField() : "";
    return escapeHtml(formatCellValue(transposedCellValue(rowData, field)));
  }

  function transposedCellValue(rowData, field, sourceRows = currentDatasetRows) {
    if (field === "__field") return rowData?.__field;
    const rowIndex = transposedRowIndexFromField(field);
    if (rowIndex < 0 || rowIndex >= sourceRows.length) return "";
    const columnField = rowData?.__column_field;
    return columnField ? sourceRows[rowIndex]?.[columnField] : "";
  }

  function transposedAriaSort(dir) {
    if (dir === "asc") return "ascending";
    if (dir === "desc") return "descending";
    return "none";
  }

  function sortedTransposedColumns(columns, sourceRows) {
    if (!transposedSort.field || transposedSort.dir === "none") return columns;
    const direction = transposedSort.dir === "desc" ? -1 : 1;
    return columns
      .map((column, index) => ({ column, index }))
      .sort((left, right) => {
        const comparison = compareDatasetViewerValues(
          transposedSortValue(left.column, transposedSort.field, sourceRows),
          transposedSortValue(right.column, transposedSort.field, sourceRows),
        );
        return (comparison || left.index - right.index) * direction;
      })
      .map((entry) => entry.column);
  }

  function transposedSortValue(column, field, sourceRows = currentDatasetRows) {
    return transposedCellValue({ __column_field: column?.field, __field: column?.name }, field, sourceRows);
  }

  function transposedRowIndexFromField(field) {
    const match = String(field || "").match(/^r(\d+)$/);
    return match ? Number(match[1]) : -1;
  }

  function orderedDatasetViewerColumns(data) {
    const sourceColumns = Array.isArray(data?.columns) ? data.columns : [];
    if (!state.datasetViewerAlphabeticalColumns) return sourceColumns;
    return sourceColumns
      .map((column, index) => ({ column, index }))
      .sort((left, right) => (
        compareDatasetViewerColumnNames(left.column, right.column) || left.index - right.index
      ))
      .map((entry) => entry.column);
  }

  function compareDatasetViewerColumnNames(left, right) {
    return String(left?.name || "").localeCompare(String(right?.name || ""), undefined, {
      numeric: true,
      sensitivity: "base",
    });
  }

  function compareDatasetViewerValues(left, right) {
    const leftMissing = left === null || left === undefined || left === "";
    const rightMissing = right === null || right === undefined || right === "";
    if (leftMissing && rightMissing) return 0;
    if (leftMissing) return 1;
    if (rightMissing) return -1;
    const leftNumber = Number(left);
    const rightNumber = Number(right);
    if (Number.isFinite(leftNumber) && Number.isFinite(rightNumber)) {
      return leftNumber - rightNumber;
    }
    return formatCellValue(left).localeCompare(formatCellValue(right), undefined, {
      numeric: true,
      sensitivity: "base",
    });
  }

  function handleDatasetViewerGridClick(event) {
    const grid = el("datasetViewerGrid");
    if (!grid || !grid.contains(event.target)) return;
    const normalHeaderLabel = event.target?.closest?.(".dataset-viewer-header-label[data-dataset-viewer-column-field]");
    if (normalHeaderLabel && grid.contains(normalHeaderLabel)) {
      event.preventDefault();
      event.stopPropagation();
      toggleColumnSelection(normalHeaderLabel.dataset.datasetViewerColumnField || "");
      return;
    }
    const transposedSortButton = event.target?.closest?.("[data-dataset-viewer-transposed-sort]");
    if (transposedSortButton && grid.contains(transposedSortButton)) {
      event.preventDefault();
      event.stopPropagation();
      cycleTransposedSort(transposedSortButton.dataset.datasetViewerTransposedSort || "");
      return;
    }
    const transposedHeaderLabel = event.target?.closest?.(".dataset-viewer-transposed-header-label[data-dataset-viewer-row-id]");
    if (transposedHeaderLabel && grid.contains(transposedHeaderLabel)) {
      event.preventDefault();
      event.stopPropagation();
      toggleRowSelection(Number(transposedHeaderLabel.dataset.datasetViewerRowId));
      return;
    }
    if (!state.datasetViewerTranspose) return;
    const transposedRow = event.target?.closest?.(".tabulator-row[data-dataset-viewer-column-field]");
    if (!transposedRow || !grid.contains(transposedRow)) return;
    event.preventDefault();
    toggleColumnSelection(transposedRow.dataset.datasetViewerColumnField || "");
  }

  function handleDatasetViewerGridContextMenu(event) {
    const grid = el("datasetViewerGrid");
    if (!grid || !grid.contains(event.target)) return;
    const cell = event.target?.closest?.(".tabulator-cell[tabulator-field]");
    const selectionLabel = selectedCopyLabel();
    const actions = [];
    if (cell && grid.contains(cell)) {
      actions.push({ mode: "cell", label: "Copy cell to clipboard", value: datasetViewerCellValue(cell) });
    }
    if (selectionLabel) actions.push({ mode: "selection", label: selectionLabel });
    actions.push({ mode: "displayed-table", label: "Copy displayed table to clipboard" });
    if (selectionLabel) {
      actions.push({ divider: true });
      actions.push({ mode: "clear-selection", label: "Clear selection" });
    }
    event.preventDefault();
    event.stopPropagation();
    openDatasetViewerContextMenu(event, actions);
  }

  function cycleTransposedSort(field) {
    if (!field) return;
    const current = transposedSort.field === field ? transposedSort.dir : "none";
    const next = current === "none" ? "asc" : current === "asc" ? "desc" : "none";
    transposedSort = next === "none" ? { field: "", dir: "none" } : { field, dir: next };
    rerenderCachedData();
  }

  function resetSelections() {
    selectedRowIds = new Set();
    selectedColumnFields = new Set();
    selectionAxis = "";
  }

  function pruneSelectionsForCurrentData() {
    const validRowIds = new Set(currentDatasetRows.map((row) => Number(row.__row_id)));
    const validColumnFields = new Set(currentDatasetColumns.map((column) => column.field));
    selectedRowIds = new Set([...selectedRowIds].filter((rowId) => validRowIds.has(Number(rowId))));
    selectedColumnFields = new Set([...selectedColumnFields].filter((field) => validColumnFields.has(field)));
    if (selectionAxis === "rows" && !selectedRowIds.size) selectionAxis = "";
    if (selectionAxis === "columns" && !selectedColumnFields.size) selectionAxis = "";
    if (!selectionAxis) {
      selectedRowIds = new Set();
      selectedColumnFields = new Set();
    }
  }

  function toggleColumnSelection(field) {
    if (!field || String(field).startsWith("__")) return;
    clearRowSelection({ syncTable: true });
    if (selectionAxis !== "columns") selectedColumnFields = new Set();
    if (selectedColumnFields.has(field)) {
      selectedColumnFields.delete(field);
    } else {
      selectedColumnFields.add(field);
    }
    selectionAxis = selectedColumnFields.size ? "columns" : "";
    syncNormalColumnSelectionClasses();
    syncTransposedRenderedSelection();
  }

  function toggleRowSelection(rowId) {
    if (!Number.isFinite(rowId)) return;
    clearColumnSelection();
    if (selectionAxis !== "rows") selectedRowIds = new Set();
    if (selectedRowIds.has(rowId)) {
      selectedRowIds.delete(rowId);
    } else {
      selectedRowIds.add(rowId);
    }
    selectionAxis = selectedRowIds.size ? "rows" : "";
    if (state.datasetViewerTranspose) {
      syncTransposedRenderedSelection();
    } else {
      restoreNormalRowSelection();
    }
  }

  function clearRowSelection({ syncTable = false } = {}) {
    const hadRows = selectedRowIds.size > 0;
    selectedRowIds = new Set();
    if (selectionAxis === "rows") selectionAxis = "";
    if (state.datasetViewerTranspose) {
      if (hadRows) clearTransposedRowSelectionClasses();
      syncTransposedRenderedSelection();
      return;
    }
    if (!syncTable || !datasetTable || typeof datasetTable.deselectRow !== "function") return;
    suppressRowSelectionSync = true;
    try {
      datasetTable.deselectRow();
    } catch (_) {
      // Ignore stale Tabulator instances.
    } finally {
      suppressRowSelectionSync = false;
    }
  }

  function clearColumnSelection() {
    const hadColumns = selectedColumnFields.size > 0;
    selectedColumnFields = new Set();
    if (selectionAxis === "columns") selectionAxis = "";
    if (state.datasetViewerTranspose) {
      if (hadColumns) clearTransposedColumnSelectionClasses();
      syncTransposedRenderedSelection();
      return;
    }
    syncNormalColumnSelectionClasses();
  }

  function clearDatasetViewerSelection() {
    clearRowSelection({ syncTable: true });
    clearColumnSelection();
  }

  function handleNormalRowSelectionChanged(rows = []) {
    if (suppressRowSelectionSync) return;
    const rowIds = new Set((rows || [])
      .map((row) => Number(row?.__row_id))
      .filter((rowId) => Number.isFinite(rowId)));
    if (rowIds.size) {
      selectedRowIds = rowIds;
      selectedColumnFields = new Set();
      selectionAxis = "rows";
      syncNormalColumnSelectionClasses();
    } else if (selectionAxis === "rows") {
      selectedRowIds = new Set();
      selectionAxis = "";
    }
  }

  function restoreNormalRowSelection() {
    if (!datasetTable || typeof datasetTable.deselectRow !== "function") return;
    suppressRowSelectionSync = true;
    try {
      datasetTable.deselectRow();
      if (selectionAxis === "rows" && selectedRowIds.size && typeof datasetTable.selectRow === "function") {
        datasetTable.selectRow([...selectedRowIds]);
      }
    } catch (_) {
      // Ignore stale Tabulator instances.
    } finally {
      suppressRowSelectionSync = false;
    }
  }

  function syncNormalColumnSelectionClasses() {
    const grid = document.getElementById("datasetViewerGrid");
    if (!grid || state.datasetViewerTranspose) return;
    const selected = selectionAxis === "columns" ? selectedColumnFields : new Set();
    grid.querySelectorAll(".tabulator-col[tabulator-field], .tabulator-cell[tabulator-field]").forEach((node) => {
      const field = node.getAttribute("tabulator-field") || "";
      node.classList.toggle("dataset-viewer-column-selected", selected.has(field));
    });
  }

  function syncNormalRenderedSelection() {
    if (selectionAxis === "rows") restoreNormalRowSelection();
    syncNormalColumnSelectionClasses();
  }

  function syncTransposedRenderedSelection() {
    const grid = document.getElementById("datasetViewerGrid");
    if (!grid || !state.datasetViewerTranspose) return;
    const selectedRows = selectionAxis === "rows" ? selectedRowIds : new Set();
    const selectedRowFields = new Set(
      currentColumns
        .filter((column) => column.datasetRowId !== undefined && selectedRows.has(Number(column.datasetRowId)))
        .map((column) => column.field),
    );
    const selectedColumns = selectionAxis === "columns" ? selectedColumnFields : new Set();
    grid.querySelectorAll(".tabulator-row[data-dataset-viewer-column-field]").forEach((row) => {
      const field = row.getAttribute("data-dataset-viewer-column-field") || "";
      row.classList.toggle("tabulator-selected", selectedColumns.has(field));
    });
    grid.querySelectorAll(".tabulator-col[tabulator-field], .tabulator-cell[tabulator-field]").forEach((node) => {
      const field = node.getAttribute("tabulator-field") || "";
      node.classList.toggle("dataset-viewer-transposed-column-selected", selectedRowFields.has(field));
    });
  }

  function clearTransposedColumnSelectionClasses(grid = document.getElementById("datasetViewerGrid")) {
    if (!grid || !state.datasetViewerTranspose) return;
    grid.querySelectorAll(".tabulator-row.tabulator-selected").forEach((row) => {
      row.classList.remove("tabulator-selected");
    });
  }

  function clearTransposedRowSelectionClasses(grid = document.getElementById("datasetViewerGrid")) {
    if (!grid || !state.datasetViewerTranspose) return;
    grid.querySelectorAll(".dataset-viewer-transposed-column-selected").forEach((node) => {
      node.classList.remove("dataset-viewer-transposed-column-selected");
    });
  }

  function applySearch() {
    const query = String(state.datasetViewerSearch || "").trim().toLowerCase();
    if (state.datasetViewerTranspose) {
      applyTransposedSearch(query);
      return;
    }
    if (!datasetTable) return;
    try {
      if (!query) {
        datasetTable.clearFilter();
        markRendered(toolCache(TOOL_ID).requestKey);
        return;
      }
      const fields = [...currentFields];
      datasetTable.setFilter((row) => fields.some((field) => formatCellValue(row[field]).toLowerCase().includes(query)));
      markRendered(toolCache(TOOL_ID).requestKey);
    } catch (_) {
      // Search is client-only convenience; stale Tabulator instances can be ignored.
    }
  }

  function transposedColumnMatchesSearch(column, query) {
    return formatCellValue(column?.__field ?? column?.name).toLowerCase().includes(query);
  }

  function applyTransposedSearch(query) {
    if (!datasetTable) return;
    try {
      if (!query) {
        datasetTable.clearFilter();
        syncTransposedVisibleColumnsFromRows(currentRows);
      } else {
        datasetTable.setFilter((row) => transposedColumnMatchesSearch(row, query));
        syncTransposedVisibleColumnsFromActiveRows();
      }
      syncTransposedRenderedSelection();
      markRendered(toolCache(TOOL_ID).requestKey);
    } catch (_) {
      // Search is client-only convenience; stale Tabulator instances can be ignored.
    }
  }

  function syncTransposedVisibleColumnsFromActiveRows() {
    if (!state.datasetViewerTranspose) return;
    let rows = currentRows;
    try {
      if (datasetTable && typeof datasetTable.getData === "function") rows = datasetTable.getData("active");
    } catch (_) {
      rows = currentRows;
    }
    syncTransposedVisibleColumnsFromRows(rows);
  }

  function syncTransposedVisibleColumnsFromRows(rows) {
    currentVisibleDatasetColumns = (rows || [])
      .map((row) => currentDatasetColumnByField.get(row?.__column_field))
      .filter(Boolean);
  }

  function datasetViewerCellValue(cell) {
    if (!state.datasetViewerTranspose) return cell.textContent || "";
    const field = cell.getAttribute("tabulator-field") || "";
    const row = cell.closest(".tabulator-row[data-dataset-viewer-column-field]");
    const columnField = row?.dataset?.datasetViewerColumnField || "";
    if (!field || !columnField) return cell.textContent || "";
    const column = currentDatasetColumnByField.get(columnField);
    return formatCellValue(transposedCellValue({
      __column_field: columnField,
      __field: column?.name || "",
    }, field));
  }

  async function copySelectedRows() {
    if (!selectionAxis) return;
    const csv = state.datasetViewerTranspose ? transposedSelectionToCsv() : normalSelectionToCsv();
    if (!csv) return;
    const copied = await copyTextToClipboard(csv);
    const subject = selectionAxis === "columns" ? "columns" : "rows";
    showClipboardToast(copied ? `Selected ${subject} copied` : `Could not copy selected ${subject}`, !copied);
  }

  function normalSelectionToCsv() {
    const rows = selectionAxis === "columns" ? rowsForColumnCopy() : selectedRowsForCopy();
    const columns = selectionAxis === "columns" ? selectedColumnsForCopy() : columnsForRowCopy();
    if (!rows.length || !columns.length) return "";
    return rowsToCsv(rows, columns);
  }

  function transposedSelectionToCsv() {
    if (selectionAxis === "columns") {
      const rows = activeTransposedRowsForCopy().filter((row) => selectedColumnFields.has(row.__column_field));
      return rows.length ? transposedRowsToCsv(rows, currentColumns) : "";
    }
    if (selectionAxis === "rows") {
      const columns = currentColumns.filter((column) => (
        column.field === "__field"
        || (column.datasetRowId !== undefined && selectedRowIds.has(Number(column.datasetRowId)))
      ));
      return columns.length > 1 ? transposedRowsToCsv(activeTransposedRowsForCopy(), columns) : "";
    }
    return "";
  }

  async function copyDisplayedTable() {
    const csv = state.datasetViewerTranspose
      ? transposedRowsToCsv(activeTransposedRowsForCopy(), currentColumns)
      : rowsToCsv(rowsForColumnCopy(), currentDatasetColumns);
    const copied = await copyTextToClipboard(csv);
    showClipboardToast(copied ? "Displayed table copied" : "Could not copy displayed table", !copied);
  }

  function selectedRowsForCopy() {
    const selected = rowsForColumnCopy().filter((row) => selectedRowIds.has(Number(row.__row_id)));
    if (selected.length) return selected;
    return currentDatasetRows.filter((row) => selectedRowIds.has(Number(row.__row_id)));
  }

  function selectedColumnsForCopy() {
    const visibleSelected = currentVisibleDatasetColumns.filter((column) => selectedColumnFields.has(column.field));
    const visibleFields = new Set(visibleSelected.map((column) => column.field));
    const hiddenSelected = currentDatasetColumns.filter((column) => selectedColumnFields.has(column.field) && !visibleFields.has(column.field));
    return [...visibleSelected, ...hiddenSelected];
  }

  function columnsForRowCopy() {
    return state.datasetViewerTranspose ? currentVisibleDatasetColumns : currentDatasetColumns;
  }

  function rowsForColumnCopy() {
    if (state.datasetViewerTranspose) return currentDatasetRows;
    if (!datasetTable) return currentDatasetRows;
    try {
      if (typeof datasetTable.getRows === "function") {
        return datasetTable.getRows("active")
          .map((row) => typeof row.getData === "function" ? row.getData() : null)
          .filter(Boolean);
      }
    } catch (_) {
      // Fall through to the simpler public data getter.
    }
    try {
      if (typeof datasetTable.getData === "function") return datasetTable.getData("active");
    } catch (_) {
      // Ignore stale Tabulator instances.
    }
    return currentDatasetRows;
  }

  function activeTransposedRowsForCopy() {
    if (!datasetTable) return currentRows;
    try {
      if (typeof datasetTable.getRows === "function") {
        return datasetTable.getRows("active")
          .map((row) => typeof row.getData === "function" ? row.getData() : null)
          .filter(Boolean);
      }
    } catch (_) {
      // Fall through to the simpler public data getter.
    }
    try {
      if (typeof datasetTable.getData === "function") return datasetTable.getData("active");
    } catch (_) {
      // Ignore stale Tabulator instances.
    }
    return currentRows;
  }

  function rowsToCsv(rows, columns) {
    const visibleColumns = columns.filter((column) => column.field && !String(column.field).startsWith("__"));
    const header = visibleColumns.map((column) => csvCell(column.name || column.copyTitle || column.title || column.field)).join(",");
    const body = rows.map((row) => visibleColumns.map((column) => csvCell(row[column.field])).join(","));
    return [header, ...body].join("\n");
  }

  function transposedRowsToCsv(rows, columns) {
    const visibleColumns = columns.filter((column) => column.field);
    const header = visibleColumns.map((column) => csvCell(column.copyTitle || column.title || column.field)).join(",");
    const body = rows.map((row) => visibleColumns.map((column) => csvCell(transposedCellValue(row, column.field))).join(","));
    return [header, ...body].join("\n");
  }

  function csvCell(value) {
    const text = formatCellValue(value);
    return /[",\n\r]/.test(text) ? `"${text.replace(/"/g, '""')}"` : text;
  }

  function formatCellValue(value) {
    if (value === null || value === undefined) return "";
    return String(value);
  }

  function selectedCopyCount() {
    if (selectionAxis === "columns") return selectedColumnFields.size;
    if (selectionAxis === "rows") return selectedRowIds.size;
    return 0;
  }

  function selectedCopyLabel() {
    const count = selectedCopyCount();
    const screenAxis = state.datasetViewerTranspose
      ? selectionAxis === "rows" ? "columns" : selectionAxis === "columns" ? "rows" : ""
      : selectionAxis;
    if (screenAxis === "rows" && count > 0) return `Copy selected row${count === 1 ? "" : "s"} to clipboard`;
    if (screenAxis === "columns" && count > 0) return `Copy selected column${count === 1 ? "" : "s"} to clipboard`;
    return "";
  }

  function datasetViewerCellContextMenu() {
    let menu = document.getElementById("datasetViewerCellContextMenu");
    if (menu) return menu;
    menu = document.createElement("div");
    menu.id = "datasetViewerCellContextMenu";
    menu.className = "dataset-viewer-context-menu";
    menu.hidden = true;
    menu.setAttribute("role", "menu");
    menu.addEventListener("click", copyDatasetViewerContextValue);
    document.body.append(menu);
    return menu;
  }

  function openDatasetViewerContextMenu(event, actions = []) {
    closeDatasetViewerCellContextMenu();
    if (!actions.length) return;
    const menu = datasetViewerCellContextMenu();
    actions.forEach((action) => {
      if (action.divider) {
        const divider = document.createElement("div");
        divider.className = "dataset-viewer-context-menu-divider";
        divider.setAttribute("role", "separator");
        menu.append(divider);
        return;
      }
      const button = document.createElement("button");
      button.className = "dataset-viewer-context-menu-item";
      button.type = "button";
      button.setAttribute("role", "menuitem");
      button.dataset.copyMode = action.mode || "cell";
      button.dataset.copyValue = action.value || "";
      button.textContent = action.label || "Copy cell to clipboard";
      menu.append(button);
    });
    menu.hidden = false;
    positionDatasetViewerContextMenu(menu, event.clientX, event.clientY);
    menu.querySelector("button")?.focus({ preventScroll: true });
    window.addEventListener("pointerdown", handleDatasetViewerContextPointerDown, true);
    window.addEventListener("keydown", handleDatasetViewerContextKeydown, true);
    window.addEventListener("resize", closeDatasetViewerCellContextMenu, true);
    window.addEventListener("scroll", closeDatasetViewerCellContextMenu, true);
  }

  function positionDatasetViewerContextMenu(menu, clientX, clientY) {
    const margin = 8;
    menu.style.left = "0px";
    menu.style.top = "0px";
    const rect = menu.getBoundingClientRect();
    const maxLeft = Math.max(margin, window.innerWidth - rect.width - margin);
    const maxTop = Math.max(margin, window.innerHeight - rect.height - margin);
    const left = Math.min(Math.max(margin, clientX || margin), maxLeft);
    const top = Math.min(Math.max(margin, clientY || margin), maxTop);
    menu.style.left = `${left}px`;
    menu.style.top = `${top}px`;
  }

  async function copyDatasetViewerContextValue(event) {
    const button = event.target?.closest?.("button[data-copy-mode]");
    const menu = document.getElementById("datasetViewerCellContextMenu");
    if (!button || !menu?.contains(button)) return;
    event.preventDefault();
    if (button.dataset.copyMode === "selection") {
      await copySelectedRows();
      closeDatasetViewerCellContextMenu();
      return;
    }
    if (button.dataset.copyMode === "displayed-table") {
      await copyDisplayedTable();
      closeDatasetViewerCellContextMenu();
      return;
    }
    if (button.dataset.copyMode === "clear-selection") {
      clearDatasetViewerSelection();
      closeDatasetViewerCellContextMenu();
      return;
    }
    const value = button.dataset.copyValue || "";
    const copied = await copyTextToClipboard(value);
    showClipboardToast(copied ? "Cell copied to clipboard" : "Could not copy cell", !copied);
    closeDatasetViewerCellContextMenu();
  }

  function handleDatasetViewerContextPointerDown(event) {
    const menu = document.getElementById("datasetViewerCellContextMenu");
    if (!menu || menu.hidden || menu.contains(event.target)) return;
    closeDatasetViewerCellContextMenu();
  }

  function handleDatasetViewerContextKeydown(event) {
    if (event.key !== "Escape") return;
    event.preventDefault();
    closeDatasetViewerCellContextMenu();
  }

  function closeDatasetViewerCellContextMenu() {
    const menu = document.getElementById("datasetViewerCellContextMenu");
    if (menu) {
      menu.hidden = true;
      menu.replaceChildren();
    }
    window.removeEventListener("pointerdown", handleDatasetViewerContextPointerDown, true);
    window.removeEventListener("keydown", handleDatasetViewerContextKeydown, true);
    window.removeEventListener("resize", closeDatasetViewerCellContextMenu, true);
    window.removeEventListener("scroll", closeDatasetViewerCellContextMenu, true);
  }

  function countMeta(data) {
    const displayed = Number(data?.displayed_row_count || 0);
    const displayMeta = `${displayed.toLocaleString()} shown`;
    return data?.has_more ? `${displayMeta} · more available` : displayMeta;
  }

  function currentSearchKey() {
    return String(state.datasetViewerSearch || "");
  }

  function markRendered(requestKey) {
    renderedRequestKey = requestKey || null;
    renderedSearch = currentSearchKey();
    renderedTranspose = Boolean(state.datasetViewerTranspose);
    renderedAlphabeticalColumns = Boolean(state.datasetViewerAlphabeticalColumns);
  }

  function cacheIsRendered(cache) {
    const grid = document.getElementById("datasetViewerGrid");
    return Boolean(
      cache?.data
        && cache.requestKey
        && renderedRequestKey === cache.requestKey
        && renderedSearch === currentSearchKey()
        && renderedTranspose === Boolean(state.datasetViewerTranspose)
        && renderedAlphabeticalColumns === Boolean(state.datasetViewerAlphabeticalColumns)
        && grid
        && grid.children.length
    );
  }

  function rerenderCachedData() {
    const cache = toolCache(TOOL_ID);
    if (cache.data) measureToolRender(TOOL_ID, () => renderData(cache.data, cache.requestKey));
  }

  function resize({ hard = true } = {}) {
    if (!datasetTable) return;
    resizeHard = resizeHard || hard;
    if (resizeFrame !== null) return;
    resizeFrame = requestAnimationFrame(() => {
      const shouldHard = resizeHard;
      resizeFrame = null;
      resizeHard = false;
      try {
        datasetTable.redraw(shouldHard);
      } catch (_) {
        // Ignore stale Tabulator instances.
      }
    });
  }

  function refreshTheme() {
    resize();
  }

  return {
    buildRequest,
    fetchData,
    refreshTheme,
    resize,
    useCached,
    requestKey: () => stableRequestKey(buildRequest()),
  };
}
