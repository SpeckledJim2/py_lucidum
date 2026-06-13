import { loadTabulator } from "./shared/tabulator.js";

const TOOL_ID = "dataset_viewer";
const MAX_ROWS = 1000;

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
  syncClientTimingFromData,
  syncDuckDbTimingFromData,
  toolCache,
}) {
  let datasetTable = null;
  let renderToken = 0;
  let currentRows = [];
  let currentColumns = [];
  let currentFields = [];
  let transposedSelectedIds = new Set();
  let renderedRequestKey = null;
  let renderedSearch = null;
  let renderedTranspose = null;
  let transposedHoverRow = null;

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
          <button id="datasetViewerResetSort" class="ghost dataset-viewer-action" type="button">Reset sort</button>
          <button id="datasetViewerCopySelected" class="ghost dataset-viewer-action" type="button" disabled>Copy selected</button>
          <div id="datasetViewerCount" class="dataset-viewer-count"></div>
        </div>
        <div id="datasetViewerGrid" class="dataset-viewer-grid"></div>`;
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
        const cache = toolCache(TOOL_ID);
        if (cache.data) measureToolRender(TOOL_ID, () => renderData(cache.data, cache.requestKey));
      });
      el("datasetViewerResetSort").addEventListener("click", resetSort);
      el("datasetViewerCopySelected").addEventListener("click", copySelectedRows);
    }
    const search = el("datasetViewerSearch");
    if (document.activeElement !== search && search.value !== (state.datasetViewerSearch || "")) {
      search.value = state.datasetViewerSearch || "";
    }
    el("datasetViewerTranspose").checked = Boolean(state.datasetViewerTranspose);
    return wrap;
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

  function clearTable() {
    hideTransposedHover();
    if (datasetTable) {
      try {
        datasetTable.destroy();
      } catch (_) {
        // Tabulator may already have been removed by a stale render.
      }
    }
    datasetTable = null;
    transposedSelectedIds = new Set();
    currentRows = [];
    currentColumns = [];
    currentFields = [];
    renderedRequestKey = null;
    renderedSearch = null;
    renderedTranspose = null;
    updateCopyButton();
  }

  function renderLoading() {
    const wrap = renderShell();
    if (!wrap) return;
    renderToken += 1;
    clearTable();
    el("datasetViewerCount").textContent = "";
    el("datasetViewerGrid").classList.remove("dataset-viewer-grid-transposed");
    el("datasetViewerGrid").innerHTML = `<div class="dataset-viewer-state">Loading dataset...</div>`;
  }

  function renderError(message) {
    const wrap = renderShell();
    if (!wrap) return;
    renderToken += 1;
    clearTable();
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
        height: "100%",
        layout: "fitDataStretch",
        placeholder: "No matching rows",
        reactiveData: false,
        renderHorizontal: "virtual",
        renderVertical: "virtual",
        rowHeight: 22,
        selectableRows: true,
        selectableRowsPersistence: true,
        columnDefaults: {
          headerSort: true,
          resizable: true,
          formatter: (cell) => escapeHtml(formatCellValue(cell.getValue())),
        },
        columns: currentColumns,
      });
      if (typeof datasetTable.on === "function") {
        datasetTable.on("rowSelectionChanged", updateCopyButton);
      }
      applySearch();
      updateCopyButton();
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
            <tr>${currentColumns.map((column) => `<th>${escapeHtml(column.copyTitle || column.title || column.field)}</th>`).join("")}</tr>
          </thead>
          <tbody>
            ${currentRows.map((row) => `
              <tr data-dataset-viewer-row-id="${escapeHtml(String(row.__row_id))}">
                ${currentColumns.map((column) => `<td>${escapeHtml(formatCellValue(row[column.field]))}</td>`).join("")}
              </tr>
            `).join("")}
          </tbody>
        </table>
      </div>`;
    target.querySelector("tbody")?.addEventListener("click", (event) => {
      const row = event.target?.closest?.("tr[data-dataset-viewer-row-id]");
      if (!row) return;
      const rowId = Number(row.dataset.datasetViewerRowId);
      if (transposedSelectedIds.has(rowId)) {
        transposedSelectedIds.delete(rowId);
        row.classList.remove("tabulator-selected");
      } else {
        transposedSelectedIds.add(rowId);
        row.classList.add("tabulator-selected");
      }
      updateCopyButton();
    });
    wireTransposedHover(target);
    updateCopyButton();
  }

  function wireTransposedHover(target) {
    const scroll = target.querySelector(".dataset-viewer-transposed-scroll");
    const hover = target.querySelector(".dataset-viewer-transposed-hover");
    if (!scroll || !hover) return;
    scroll.addEventListener("pointerover", (event) => {
      const row = event.target?.closest?.("tbody tr[data-dataset-viewer-row-id]");
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
    const sourceColumns = Array.isArray(data?.columns) ? data.columns : [];
    const rows = Array.isArray(data?.rows) ? data.rows : [];
    const columns = sourceColumns.map((column) => ({
      title: escapeHtml(column.name),
      copyTitle: column.name,
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
    };
  }

  function transposedTableData(data) {
    const sourceColumns = Array.isArray(data?.columns) ? data.columns : [];
    const sourceRows = filteredSourceRows(data, sourceColumns);
    const rows = sourceColumns.map((column, columnIndex) => {
      const row = {
        __row_id: columnIndex + 1,
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
        minWidth: 170,
        width: 220,
      },
      ...sourceRows.map((row, index) => ({
        title: `Row ${index + 1}`,
        copyTitle: `Row ${index + 1}`,
        field: `r${index}`,
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
    };
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

  function filteredSourceRows(data, sourceColumns) {
    const rows = Array.isArray(data?.rows) ? data.rows : [];
    const query = String(state.datasetViewerSearch || "").trim().toLowerCase();
    if (!query) return rows;
    return rows.filter((row) => rowMatchesSearch(row, sourceColumns, query));
  }

  function rowMatchesSearch(row, sourceColumns, query) {
    return sourceColumns.some((column) => formatCellValue(row[column.field]).toLowerCase().includes(query));
  }

  function resetSort() {
    if (!datasetTable) {
      if (state.datasetViewerTranspose) {
        const cache = toolCache(TOOL_ID);
        if (cache.data) measureToolRender(TOOL_ID, () => renderData(cache.data, cache.requestKey));
      }
      return;
    }
    try {
      datasetTable.clearSort();
    } catch (_) {
      // Ignore stale Tabulator instances.
    }
  }

  async function copySelectedRows() {
    const selected = datasetTable && typeof datasetTable.getSelectedData === "function"
      ? datasetTable.getSelectedData()
      : currentRows.filter((row) => transposedSelectedIds.has(Number(row.__row_id)));
    if (!selected.length) return;
    const csv = rowsToCsv(selected, currentColumns);
    const copied = await copyTextToClipboard(csv);
    showClipboardToast(copied ? "Selected rows copied" : "Could not copy selected rows", !copied);
  }

  function rowsToCsv(rows, columns) {
    const visibleColumns = columns.filter((column) => column.field && !String(column.field).startsWith("__row_"));
    const header = visibleColumns.map((column) => csvCell(column.copyTitle || column.title || column.field)).join(",");
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

  function updateCopyButton() {
    const button = document.getElementById("datasetViewerCopySelected");
    if (!button) return;
    const selectedCount = datasetTable && typeof datasetTable.getSelectedData === "function"
      ? datasetTable.getSelectedData().length
      : transposedSelectedIds.size;
    button.disabled = selectedCount <= 0;
    button.textContent = selectedCount > 0 ? `Copy selected (${selectedCount.toLocaleString()})` : "Copy selected";
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
  }

  function cacheIsRendered(cache) {
    const grid = document.getElementById("datasetViewerGrid");
    return Boolean(
      cache?.data
        && cache.requestKey
        && renderedRequestKey === cache.requestKey
        && renderedSearch === currentSearchKey()
        && renderedTranspose === Boolean(state.datasetViewerTranspose)
        && grid
        && grid.children.length
    );
  }

  function resize() {
    if (!datasetTable) return;
    requestAnimationFrame(() => {
      try {
        datasetTable.redraw(true);
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
