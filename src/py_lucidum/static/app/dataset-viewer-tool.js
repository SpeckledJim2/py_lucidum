import { loadTabulator } from "./shared/tabulator.js";

const TOOL_ID = "dataset_viewer";
const MAX_ROWS = 1000;
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
  let transposedHoverRow = null;
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
    hideTransposedHover();
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
      renderTransposedGrid();
      markRendered(requestKey);
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
        layout: "fitDataStretch",
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

  function renderTransposedGrid() {
    const target = el("datasetViewerGrid");
    if (!target) return;
    target.innerHTML = `
      <div class="dataset-viewer-transposed-scroll">
        <div class="dataset-viewer-transposed-hover" hidden aria-hidden="true"></div>
        <table class="dataset-viewer-transposed-table">
          <thead>
            <tr>${currentColumns.map((column) => transposedHeaderHtml(column)).join("")}</tr>
          </thead>
          <tbody>
            ${currentRows.map((row) => `
              <tr data-dataset-viewer-column-field="${escapeHtml(String(row.__column_field || ""))}"${selectedColumnFields.has(row.__column_field) ? ' class="tabulator-selected"' : ""}>
                ${currentColumns.map((column) => transposedCellHtml(row, column)).join("")}
              </tr>
            `).join("")}
          </tbody>
        </table>
      </div>`;
    wireTransposedHover(target);
  }

  function wireTransposedHover(target) {
    const scroll = target.querySelector(".dataset-viewer-transposed-scroll");
    const hover = target.querySelector(".dataset-viewer-transposed-hover");
    if (!scroll || !hover) return;
    scroll.addEventListener("pointerover", (event) => {
      const row = event.target?.closest?.("tbody tr[data-dataset-viewer-column-field]");
      if (!row || !scroll.contains(row)) return;
      if (row === transposedHoverRow) return;
      transposedHoverRow = row;
      hover.style.width = `${Math.max(scroll.scrollWidth, scroll.clientWidth)}px`;
      hover.style.height = `${row.offsetHeight}px`;
      hover.style.transform = `translateY(${row.offsetTop}px)`;
      hover.hidden = false;
    });
    scroll.addEventListener("pointerleave", hideTransposedHover);
    scroll.addEventListener("scroll", () => {
      if (!transposedHoverRow || !transposedHoverRow.isConnected || hover.hidden) return;
      hover.style.width = `${Math.max(scroll.scrollWidth, scroll.clientWidth)}px`;
    });
  }

  function hideTransposedHover() {
    const hover = document.querySelector("#datasetViewerGrid .dataset-viewer-transposed-hover");
    transposedHoverRow = null;
    if (hover) hover.hidden = true;
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
      minWidth: column.kind === "numeric" || column.kind === "integer" ? 90 : 130,
      maxInitialWidth: 320,
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

  function transposedTableData(data) {
    const sourceColumns = orderedDatasetViewerColumns(data);
    const sourceRows = Array.isArray(data?.rows) ? data.rows : [];
    const query = String(state.datasetViewerSearch || "").trim().toLowerCase();
    const visibleColumns = query
      ? sourceColumns.filter((column) => transposedColumnMatchesSearch(column, query))
      : sourceColumns;
    const sortedColumns = sortedTransposedColumns(visibleColumns, sourceRows);
    const rows = sortedColumns.map((column) => {
      const row = {
        __row_id: column.field,
        __column_field: column.field,
        __field: column.name,
      };
      sourceRows.forEach((sourceRow, rowIndex) => {
        row[`r${rowIndex}`] = sourceRow[column.field];
      });
      return row;
    });
    const columns = [
      {
        title: "Column",
        copyTitle: "Column",
        field: "__field",
        frozen: true,
        headerSort: true,
        sortField: "__field",
        minWidth: 170,
        width: 220,
      },
      ...sourceRows.map((row, index) => ({
        title: `Row ${index + 1}`,
        copyTitle: `Row ${index + 1}`,
        field: `r${index}`,
        sortField: `r${index}`,
        datasetRowId: row.__row_id,
        headerTooltip: row.__row_id ? `Dataset row ${row.__row_id}` : `Row ${index + 1}`,
        headerSort: true,
        minWidth: 90,
        maxInitialWidth: 220,
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
    return `<th data-dataset-viewer-transposed-field="${escapeHtml(column.sortField || column.field)}" data-sort-dir="${escapeHtml(sortDir)}" aria-sort="${transposedAriaSort(sortDir)}"${selected ? ' class="dataset-viewer-transposed-column-selected"' : ""}>
      <div class="dataset-viewer-transposed-header-content">
        ${label}
        <button class="dataset-viewer-transposed-sort-button" type="button" data-dataset-viewer-transposed-sort="${escapeHtml(column.sortField || column.field)}" data-sort-dir="${escapeHtml(sortDir)}" aria-label="Sort ${escapeHtml(title)}"></button>
      </div>
    </th>`;
  }

  function transposedCellHtml(row, column) {
    const selected = column.datasetRowId !== undefined && selectedRowIds.has(Number(column.datasetRowId));
    const value = row[column.field];
    return `<td data-dataset-viewer-cell-field="${escapeHtml(column.field)}"${selected ? ' class="dataset-viewer-transposed-column-selected"' : ""}>${escapeHtml(formatCellValue(value))}</td>`;
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
    if (field === "__field") return column?.name;
    const rowIndex = transposedRowIndexFromField(field);
    if (rowIndex < 0 || rowIndex >= sourceRows.length) return "";
    return sourceRows[rowIndex]?.[column.field];
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
    const transposedRow = event.target?.closest?.("tbody tr[data-dataset-viewer-column-field]");
    if (!transposedRow || !grid.contains(transposedRow)) return;
    event.preventDefault();
    toggleColumnSelection(transposedRow.dataset.datasetViewerColumnField || "");
  }

  function handleDatasetViewerGridContextMenu(event) {
    const grid = el("datasetViewerGrid");
    if (!grid || !grid.contains(event.target)) return;
    const selectionLabel = selectedCopyLabel();
    if (selectionLabel) {
      event.preventDefault();
      event.stopPropagation();
      openDatasetViewerContextMenu(event, { mode: "selection", label: selectionLabel });
      return;
    }
    const cell = event.target?.closest?.(".tabulator-cell[tabulator-field], .dataset-viewer-transposed-table tbody td[data-dataset-viewer-cell-field]");
    if (!cell || !grid.contains(cell)) return;
    event.preventDefault();
    event.stopPropagation();
    openDatasetViewerContextMenu(event, { mode: "cell", label: "Copy to clipboard", value: cell.textContent || "" });
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
    setTransposedColumnRowSelected(field, selectionAxis === "columns" && selectedColumnFields.has(field));
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
    restoreNormalRowSelection();
    setTransposedRowColumnSelected(transposedFieldForDatasetRowId(rowId), selectionAxis === "rows" && selectedRowIds.has(rowId));
  }

  function clearRowSelection({ syncTable = false } = {}) {
    const hadRows = selectedRowIds.size > 0;
    selectedRowIds = new Set();
    if (selectionAxis === "rows") selectionAxis = "";
    if (hadRows && state.datasetViewerTranspose) clearTransposedRowSelectionClasses();
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
    if (hadColumns && state.datasetViewerTranspose) clearTransposedColumnSelectionClasses();
    syncNormalColumnSelectionClasses();
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

  function setTransposedColumnRowSelected(field, selected, grid = document.getElementById("datasetViewerGrid")) {
    if (!grid || !state.datasetViewerTranspose || !field) return;
    grid.querySelector(`tbody tr[data-dataset-viewer-column-field="${cssEscape(field)}"]`)?.classList.toggle("tabulator-selected", selected);
  }

  function clearTransposedColumnSelectionClasses(grid = document.getElementById("datasetViewerGrid")) {
    if (!grid || !state.datasetViewerTranspose) return;
    grid.querySelectorAll("tbody tr.tabulator-selected").forEach((row) => {
      row.classList.remove("tabulator-selected");
    });
  }

  function transposedFieldForDatasetRowId(rowId) {
    const column = currentColumns.find((candidate) => candidate.datasetRowId !== undefined && Number(candidate.datasetRowId) === Number(rowId));
    return column?.field || "";
  }

  function setTransposedRowColumnSelected(field, selected, grid = document.getElementById("datasetViewerGrid")) {
    if (!grid || !state.datasetViewerTranspose || !field) return;
    const escapedField = cssEscape(field);
    grid.querySelector(`thead th[data-dataset-viewer-transposed-field="${escapedField}"]`)?.classList.toggle("dataset-viewer-transposed-column-selected", selected);
    grid.querySelectorAll(`tbody td[data-dataset-viewer-cell-field="${escapedField}"]`).forEach((cell) => {
      cell.classList.toggle("dataset-viewer-transposed-column-selected", selected);
    });
  }

  function clearTransposedRowSelectionClasses(grid = document.getElementById("datasetViewerGrid")) {
    if (!grid || !state.datasetViewerTranspose) return;
    grid.querySelectorAll(".dataset-viewer-transposed-column-selected").forEach((node) => {
      node.classList.remove("dataset-viewer-transposed-column-selected");
    });
  }

  function cssEscape(value) {
    if (window.CSS?.escape) return window.CSS.escape(String(value));
    return String(value).replace(/["\\]/g, "\\$&");
  }

  function applySearch() {
    const query = String(state.datasetViewerSearch || "").trim().toLowerCase();
    if (!datasetTable && state.datasetViewerTranspose) {
      const cache = toolCache(TOOL_ID);
      if (cache.data) measureToolRender(TOOL_ID, () => renderData(cache.data, cache.requestKey));
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
    return formatCellValue(column?.name).toLowerCase().includes(query);
  }

  async function copySelectedRows() {
    if (!selectionAxis) return;
    const rows = selectionAxis === "columns" ? rowsForColumnCopy() : selectedRowsForCopy();
    const columns = selectionAxis === "columns" ? selectedColumnsForCopy() : columnsForRowCopy();
    if (!rows.length || !columns.length) return;
    const csv = rowsToCsv(rows, columns);
    const copied = await copyTextToClipboard(csv);
    const subject = selectionAxis === "columns" ? "columns" : "rows";
    showClipboardToast(copied ? `Selected ${subject} copied` : `Could not copy selected ${subject}`, !copied);
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

  function rowsToCsv(rows, columns) {
    const visibleColumns = columns.filter((column) => column.field && !String(column.field).startsWith("__"));
    const header = visibleColumns.map((column) => csvCell(column.name || column.copyTitle || column.title || column.field)).join(",");
    const body = rows.map((row) => visibleColumns.map((column) => csvCell(row[column.field])).join(","));
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
    if (selectionAxis === "rows" && count > 0) return `Copy row${count === 1 ? "" : "s"} to clipboard`;
    if (selectionAxis === "columns" && count > 0) return `Copy column${count === 1 ? "" : "s"} to clipboard`;
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
    menu.innerHTML = '<button class="dataset-viewer-context-menu-item" type="button" role="menuitem">Copy to clipboard</button>';
    menu.querySelector("button")?.addEventListener("click", copyDatasetViewerContextValue);
    document.body.append(menu);
    return menu;
  }

  function openDatasetViewerContextMenu(event, { mode = "cell", label = "Copy to clipboard", value = "" } = {}) {
    closeDatasetViewerCellContextMenu();
    const menu = datasetViewerCellContextMenu();
    menu.dataset.copyMode = mode;
    menu.dataset.copyValue = value;
    const button = menu.querySelector("button");
    if (button) button.textContent = label;
    menu.hidden = false;
    positionDatasetViewerContextMenu(menu, event.clientX, event.clientY);
    button?.focus({ preventScroll: true });
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

  async function copyDatasetViewerContextValue() {
    const menu = document.getElementById("datasetViewerCellContextMenu");
    if (menu?.dataset?.copyMode === "selection") {
      await copySelectedRows();
      closeDatasetViewerCellContextMenu();
      return;
    }
    const value = menu?.dataset?.copyValue || "";
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
      menu.dataset.copyMode = "";
      menu.dataset.copyValue = "";
      const button = menu.querySelector("button");
      if (button) button.textContent = "Copy to clipboard";
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
