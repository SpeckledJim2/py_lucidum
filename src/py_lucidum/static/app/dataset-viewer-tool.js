import { loadTabulator } from "./shared/tabulator.js";

const TOOL_ID = "dataset_viewer";
const MAX_ROWS = 100;
const STYLESHEET_ID = "datasetViewerStylesheet";
const NORMAL_SEARCH_REPLACE_MOVE_THRESHOLD = 16;
const DATASET_VIEWER_STATE_INLINE_STYLE = "align-items:center;display:flex;font:12px system-ui,-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;height:100%;justify-content:center;min-height:220px;";

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
  let currentRenderedRows = [];
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
  let renderedPinnedColumns = null;
  let renderedWidthMode = "";
  let normalColumnWidths = new Map();
  let transposedColumnWidths = new Map();
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
            <input id="datasetViewerSearch" class="search dataset-viewer-search" placeholder="Select columns, separate with commas" />
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
    snapshotDatasetViewerColumnWidths();
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
    currentRenderedRows = [];
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
    renderedPinnedColumns = null;
    renderedWidthMode = "";
  }

  function renderLoading() {
    const wrap = renderShell();
    if (!wrap) return;
    renderToken += 1;
    clearTable({ resetSelection: true });
    el("datasetViewerCount").textContent = "";
    setDatasetViewerToolbarHidden(true);
    renderDatasetViewerState("Reading data...");
  }

  function renderError(message) {
    const wrap = renderShell();
    if (!wrap) return;
    renderToken += 1;
    clearTable({ resetSelection: true });
    el("datasetViewerCount").textContent = "";
    setDatasetViewerToolbarHidden(true);
    renderDatasetViewerState(message || "Dataset viewer query failed", { error: true });
  }

  function setDatasetViewerToolbarHidden(hidden) {
    const toolbar = el("datasetViewerWrap")?.querySelector(".dataset-viewer-toolbar");
    if (toolbar) toolbar.hidden = Boolean(hidden);
  }

  function renderDatasetViewerState(message, { error = false, transposed = false } = {}) {
    const grid = el("datasetViewerGrid");
    if (!grid) return;
    grid.classList.toggle("dataset-viewer-grid-transposed", Boolean(transposed));
    const classes = `dataset-viewer-state${error ? " dataset-viewer-state-error" : ""}`;
    const color = error ? "var(--danger)" : "var(--muted)";
    grid.innerHTML = `<div class="${classes}" style="${DATASET_VIEWER_STATE_INLINE_STYLE}color:${color};">${escapeHtml(message || "")}</div>`;
  }

  function renderData(data, requestKey = null) {
    const wrap = renderShell();
    if (!wrap) return;
    const token = renderToken + 1;
    renderToken = token;
    clearTable();
    setDatasetViewerToolbarHidden(true);
    syncTransposeControl(data);
    const tableData = Boolean(state.datasetViewerTranspose)
      ? transposedTableData(data)
      : normalTableData(data);
    currentRows = tableData.rows;
    currentRenderedRows = tableData.renderRows || tableData.rows;
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
    renderDatasetViewerState("Preparing table...", { transposed: Boolean(state.datasetViewerTranspose) });
    if (state.datasetViewerTranspose) {
      renderTransposedGrid(token, requestKey);
      return;
    }
    loadTabulator().then((Tabulator) => {
      if (token !== renderToken) return;
      const target = el("datasetViewerGrid");
      if (!target) return;
      datasetTable = new Tabulator(target, {
        data: currentRenderedRows,
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
      renderedWidthMode = "normal";
      let searchReconciled = false;
      const reconcileSearch = () => {
        if (searchReconciled) return;
        searchReconciled = true;
        reconcileRenderedSearch(token, requestKey, syncNormalRenderedSelection);
      };
      if (typeof datasetTable.on === "function") {
        datasetTable.on("rowSelectionChanged", handleNormalRowSelectionChanged);
        datasetTable.on("renderComplete", () => {
          if (token !== renderToken) return;
          setDatasetViewerToolbarHidden(false);
          syncNormalRenderedSelection();
          reconcileSearch();
        });
        datasetTable.on("dataSorted", syncNormalRenderedSelection);
        datasetTable.on("dataFiltered", syncNormalRenderedSelection);
        datasetTable.on("columnResized", rememberNormalColumnWidth);
      }
      applySearch({ mark: false });
      syncNormalRenderedSelection();
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
        data: currentRenderedRows,
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
      renderedWidthMode = "transposed";
      let searchReconciled = false;
      const reconcileSearch = () => {
        if (searchReconciled) return;
        searchReconciled = true;
        reconcileRenderedSearch(token, requestKey, syncTransposedRenderedSelection);
      };
      if (typeof datasetTable.on === "function") {
        datasetTable.on("renderComplete", () => {
          if (token !== renderToken) return;
          setDatasetViewerToolbarHidden(false);
          syncTransposedRenderedSelection();
          reconcileSearch();
        });
        datasetTable.on("dataFiltered", syncTransposedRenderedSelection);
        datasetTable.on("columnResized", rememberTransposedColumnWidth);
      }
      applySearch({ mark: false });
      syncTransposedRenderedSelection();
    }).catch((error) => {
      if (token !== renderToken) return;
      renderError(error.message || String(error));
    });
  }

  function normalTableData(data) {
    const sourceColumns = orderedDatasetViewerColumns(data);
    const rows = Array.isArray(data?.rows) ? data.rows : [];
    const search = normalColumnsForSearch(datasetViewerSearchTerms(), sourceColumns);
    return {
      columns: search.columns,
      rows,
      fields: sourceColumns.map((column) => column.field),
      datasetRows: rows,
      datasetColumns: sourceColumns,
      visibleDatasetColumns: search.visibleDatasetColumns,
    };
  }

  function normalColumnsForSearch(terms, sourceColumns = currentDatasetColumns) {
    const visibleDatasetColumns = orderedNormalColumnsForSearch(terms, sourceColumns);
    const visibleFields = new Set(visibleDatasetColumns.map((column) => column.field));
    const hiddenDatasetColumns = sourceColumns.filter((column) => !visibleFields.has(column.field));
    const renderedColumns = terms.length ? [...visibleDatasetColumns, ...hiddenDatasetColumns] : sourceColumns;
    return {
      columns: renderedColumns.map((column) => normalColumnDefinition(column, visibleFields)),
      visibleDatasetColumns,
      visibleFields,
    };
  }

  function normalColumnDefinition(column, visibleFields) {
    return {
      title: normalHeaderHtml(column),
      copyTitle: column.name,
      name: column.name,
      field: column.field,
      visible: visibleFields.has(column.field),
      ...datasetViewerColumnWidth(column),
      hozAlign: column.kind === "numeric" || column.kind === "integer" ? "right" : "left",
      headerHozAlign: "left",
    };
  }

  function datasetViewerColumnWidth(column) {
    const kind = String(column?.kind || "");
    if (kind === "integer" || kind === "numeric") {
      return datasetViewerColumnWidthWithSaved("normal", column?.field, { width: 112, minWidth: 72 });
    }
    if (kind === "date" || kind === "datetime") {
      return datasetViewerColumnWidthWithSaved("normal", column?.field, { width: 144, minWidth: 104 });
    }
    return datasetViewerColumnWidthWithSaved("normal", column?.field, { width: 180, minWidth: 96 });
  }

  function datasetViewerTransposedColumnWidth(field, defaults) {
    return datasetViewerColumnWidthWithSaved("transposed", field, defaults);
  }

  function datasetViewerColumnWidthWithSaved(mode, field, defaults) {
    const key = String(field || "");
    const widths = datasetViewerColumnWidthMap(mode);
    if (!key || !widths.has(key)) return defaults;
    const width = Number(widths.get(key));
    if (!Number.isFinite(width) || width <= 0) return defaults;
    const minWidth = Number(defaults?.minWidth);
    return {
      ...defaults,
      width: Math.max(Number.isFinite(minWidth) ? minWidth : 0, Math.round(width)),
    };
  }

  function datasetViewerColumnWidthMap(mode) {
    return mode === "transposed" ? transposedColumnWidths : normalColumnWidths;
  }

  function rememberNormalColumnWidth(column) {
    rememberDatasetViewerColumnWidth("normal", column);
  }

  function rememberTransposedColumnWidth(column) {
    rememberDatasetViewerColumnWidth("transposed", column);
  }

  function rememberDatasetViewerColumnWidth(mode, column) {
    rememberDatasetViewerFieldWidth(mode, datasetViewerColumnField(column), datasetViewerColumnRenderedWidth(column));
  }

  function rememberDatasetViewerFieldWidth(mode, field, width) {
    const key = String(field || "");
    const value = Math.round(Number(width));
    if (!key || !Number.isFinite(value) || value <= 0) return;
    datasetViewerColumnWidthMap(mode).set(key, value);
  }

  function datasetViewerColumnField(column) {
    if (column && typeof column.getField === "function") {
      try {
        return String(column.getField() || "");
      } catch (_) {
        // Fall back to the rendered header element below.
      }
    }
    try {
      const element = typeof column?.getElement === "function" ? column.getElement() : null;
      return String(element?.getAttribute?.("tabulator-field") || "");
    } catch (_) {
      return "";
    }
  }

  function datasetViewerColumnRenderedWidth(column) {
    if (column && typeof column.getWidth === "function") {
      try {
        const width = Number(column.getWidth());
        if (Number.isFinite(width) && width > 0) return width;
      } catch (_) {
        // Fall back to the rendered header element below.
      }
    }
    try {
      const element = typeof column?.getElement === "function" ? column.getElement() : null;
      const width = Number(element?.getBoundingClientRect?.().width);
      return Number.isFinite(width) ? width : 0;
    } catch (_) {
      return 0;
    }
  }

  function snapshotDatasetViewerColumnWidths() {
    const mode = renderedWidthMode;
    if (!datasetTable || !mode) return;
    let captured = false;
    try {
      if (typeof datasetTable.getColumns === "function") {
        const columns = datasetTable.getColumns();
        if (Array.isArray(columns)) {
          columns.forEach((column) => rememberDatasetViewerColumnWidth(mode, column));
          captured = columns.length > 0;
        }
      }
    } catch (_) {
      captured = false;
    }
    if (captured) return;
    const grid = document.getElementById("datasetViewerGrid");
    if (!grid) return;
    grid.querySelectorAll(".tabulator-col[tabulator-field]").forEach((node) => {
      rememberDatasetViewerFieldWidth(
        mode,
        node.getAttribute("tabulator-field") || "",
        node.getBoundingClientRect().width,
      );
    });
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
    const renderRows = orderedTransposedRowsForSearch(datasetViewerSearchTerms(), rows);
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
        ...datasetViewerTransposedColumnWidth("__field", { width: 300, minWidth: 170 }),
        headerSort: false,
        resizable: true,
      },
      ...sourceRows.map((row, index) => {
        const field = `r${index}`;
        return {
          title: transposedHeaderHtml({
            title: `Row ${index + 1}`,
            copyTitle: `Row ${index + 1}`,
            field,
            sortField: field,
            datasetRowId: row.__row_id,
          }),
          copyTitle: `Row ${index + 1}`,
          field,
          sortField: field,
          datasetRowId: row.__row_id,
          headerSort: false,
          resizable: true,
          ...datasetViewerTransposedColumnWidth(field, { width: 150, minWidth: 72 }),
        };
      }),
    ];
    return {
      columns,
      rows,
      renderRows,
      fields: ["__field", ...sourceRows.map((_, index) => `r${index}`)],
      datasetRows: sourceRows,
      datasetColumns: sourceColumns,
      visibleDatasetColumns: datasetColumnsFromTransposedRows(renderRows, sourceColumns),
    };
  }

  function datasetColumnsFromTransposedRows(rows, columns = currentDatasetColumns) {
    const columnByField = new Map(columns.map((column) => [column.field, column]));
    return (rows || [])
      .map((row) => columnByField.get(row?.__column_field))
      .filter(Boolean);
  }

  function normalHeaderHtml(column) {
    return `<button class="dataset-viewer-header-label" type="button" data-dataset-viewer-column-field="${escapeHtml(column.field)}">
        <span class="dataset-viewer-header-text">${escapeHtml(column.name)}</span>
        ${datasetViewerPinIndicator(column.field)}
      </button>`;
  }

  function transposedHeaderHtml(column) {
    const title = column.copyTitle || column.title || column.field;
    const sortDir = transposedSort.field === column.sortField ? transposedSort.dir : "none";
    const selected = column.datasetRowId !== undefined && selectedRowIds.has(Number(column.datasetRowId));
    const label = column.datasetRowId !== undefined
      ? `<button class="dataset-viewer-transposed-header-label" type="button" data-dataset-viewer-row-id="${escapeHtml(String(column.datasetRowId))}">${escapeHtml(title)}</button>`
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
    if (field === "__field") return datasetViewerPinnedFieldHtml(rowData);
    return escapeHtml(formatCellValue(transposedCellValue(rowData, field)));
  }

  function datasetViewerPinnedFieldHtml(rowData) {
    const label = escapeHtml(formatCellValue(rowData?.__field));
    if (!isDatasetViewerColumnPinned(rowData?.__column_field)) return label;
    return `<span class="dataset-viewer-pinned-field-label">
        <span class="dataset-viewer-pinned-field-text">${label}</span>
        ${datasetViewerPinIndicator(rowData?.__column_field)}
      </span>`;
  }

  function datasetViewerPinIndicator(field) {
    if (!isDatasetViewerColumnPinned(field)) return "";
    return `<span class="dataset-viewer-pin-indicator" aria-hidden="true">&#128204;</span>`;
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
    const pinnedFields = datasetViewerPinnedColumnSet();
    const pinnedColumns = columns.filter((column) => pinnedFields.has(column.field));
    const unpinnedColumns = columns.filter((column) => !pinnedFields.has(column.field));
    if (!transposedSort.field || transposedSort.dir === "none") return [...pinnedColumns, ...unpinnedColumns];
    const direction = transposedSort.dir === "desc" ? -1 : 1;
    const sortedUnpinnedColumns = unpinnedColumns
      .map((column, index) => ({ column, index }))
      .sort((left, right) => {
        const comparison = compareDatasetViewerValues(
          transposedSortValue(left.column, transposedSort.field, sourceRows),
          transposedSortValue(right.column, transposedSort.field, sourceRows),
        );
        return (comparison || left.index - right.index) * direction;
      })
      .map((entry) => entry.column);
    return [...pinnedColumns, ...sortedUnpinnedColumns];
  }

  function transposedSortValue(column, field, sourceRows = currentDatasetRows) {
    return transposedCellValue({ __column_field: column?.field, __field: column?.name }, field, sourceRows);
  }

  function transposedRowIndexFromField(field) {
    const match = String(field || "").match(/^r(\d+)$/);
    return match ? Number(match[1]) : -1;
  }

  function datasetViewerPinnedColumnFields() {
    const raw = Array.isArray(state.datasetViewerPinnedColumns) ? state.datasetViewerPinnedColumns : [];
    const fields = [];
    const seen = new Set();
    raw.forEach((field) => {
      const key = String(field || "");
      if (!key || key.startsWith("__") || seen.has(key)) return;
      seen.add(key);
      fields.push(key);
    });
    if (fields.length !== raw.length || fields.some((field, index) => field !== raw[index])) {
      state.datasetViewerPinnedColumns = fields;
    }
    return fields;
  }

  function datasetViewerPinnedColumnSet() {
    return new Set(datasetViewerPinnedColumnFields());
  }

  function isDatasetViewerColumnPinned(field) {
    return datasetViewerPinnedColumnSet().has(String(field || ""));
  }

  function datasetViewerPinnedColumnsKey() {
    return datasetViewerPinnedColumnFields().slice().sort().join("|");
  }

  function datasetViewerPinnedColumnNames() {
    return datasetViewerPinnedColumnFields()
      .map((field) => currentDatasetColumnByField.get(field))
      .filter(Boolean)
      .sort((left, right) => compareDatasetViewerColumnNames(left, right))
      .map((column) => column.name || column.field);
  }

  function pruneDatasetViewerPinnedColumns(columns = currentDatasetColumns) {
    const validFields = new Set((columns || []).map((column) => String(column?.field || "")).filter(Boolean));
    const currentFields = datasetViewerPinnedColumnFields();
    const fields = currentFields.filter((field) => validFields.has(field));
    if (fields.length !== currentFields.length) state.datasetViewerPinnedColumns = fields;
  }

  function orderedDatasetViewerColumns(data) {
    const sourceColumns = Array.isArray(data?.columns) ? data.columns : [];
    pruneDatasetViewerPinnedColumns(sourceColumns);
    const pinnedFields = datasetViewerPinnedColumnSet();
    const indexedColumns = sourceColumns.map((column, index) => ({ column, index }));
    const pinnedColumns = indexedColumns
      .filter((entry) => pinnedFields.has(entry.column.field))
      .sort((left, right) => (
        compareDatasetViewerColumnNames(left.column, right.column) || left.index - right.index
      ));
    const unpinnedColumns = indexedColumns.filter((entry) => !pinnedFields.has(entry.column.field));
    if (state.datasetViewerAlphabeticalColumns) {
      unpinnedColumns.sort((left, right) => (
        compareDatasetViewerColumnNames(left.column, right.column) || left.index - right.index
      ));
    }
    return [...pinnedColumns, ...unpinnedColumns].map((entry) => entry.column);
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
    const pinField = datasetViewerContextColumnField(event, grid);
    const selectionLabel = selectedCopyLabel();
    const actions = [];
    if (pinField) {
      actions.push({
        mode: "toggle-pin",
        label: isDatasetViewerColumnPinned(pinField) ? "Unpin column" : "Pin column",
        field: pinField,
      });
      actions.push({ divider: true });
    }
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

  function datasetViewerContextColumnField(event, grid) {
    if (state.datasetViewerTranspose) {
      const row = event.target?.closest?.(".tabulator-row[data-dataset-viewer-column-field]");
      const field = row && grid.contains(row) ? row.dataset.datasetViewerColumnField || "" : "";
      return validDatasetViewerColumnField(field) ? field : "";
    }
    const cell = event.target?.closest?.(".tabulator-cell[tabulator-field]");
    const header = event.target?.closest?.(".tabulator-col[tabulator-field]");
    const field = cell && grid.contains(cell)
      ? cell.getAttribute("tabulator-field") || ""
      : header && grid.contains(header)
        ? header.getAttribute("tabulator-field") || ""
        : "";
    return validDatasetViewerColumnField(field) ? field : "";
  }

  function validDatasetViewerColumnField(field) {
    const key = String(field || "");
    return Boolean(key && !key.startsWith("__") && currentDatasetColumnByField.has(key));
  }

  function toggleDatasetViewerPinnedColumn(field) {
    const key = String(field || "");
    if (!validDatasetViewerColumnField(key)) return;
    const fields = datasetViewerPinnedColumnFields();
    state.datasetViewerPinnedColumns = fields.includes(key)
      ? fields.filter((candidate) => candidate !== key)
      : [...fields, key];
    rerenderCachedData();
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

  function applySearch({ mark = true } = {}) {
    const terms = datasetViewerSearchTerms();
    if (state.datasetViewerTranspose) {
      applyTransposedSearch(terms, { mark });
      return;
    }
    applyNormalColumnSearch(terms, { mark });
  }

  function datasetViewerSearchTerms() {
    return String(state.datasetViewerSearch || "")
      .split(",")
      .map((term) => term.trim().toLowerCase())
      .filter(Boolean);
  }

  function datasetViewerColumnMatchesSearch(column, terms) {
    if (!terms.length) return true;
    return datasetViewerColumnSearchTermIndex(column, terms) !== -1;
  }

  function datasetViewerColumnSearchTermIndex(column, terms) {
    const name = formatCellValue(column?.name ?? column?.__field ?? column?.title).toLowerCase();
    return terms.findIndex((term) => name.includes(term));
  }

  function orderedNormalColumnsForSearch(terms, sourceColumns = currentDatasetColumns) {
    if (!terms.length) return sourceColumns;
    const pinnedFields = datasetViewerPinnedColumnSet();
    const pinnedColumns = sourceColumns.filter((column) => pinnedFields.has(column.field));
    const groupedColumns = terms.map(() => []);
    sourceColumns.forEach((column) => {
      if (pinnedFields.has(column.field)) return;
      const matchIndex = datasetViewerColumnSearchTermIndex(column, terms);
      if (matchIndex !== -1) groupedColumns[matchIndex].push(column);
    });
    return [...pinnedColumns, ...groupedColumns.flat()];
  }

  function syncNormalColumnVisibilityAndOrder(visibleColumns, visibleFields) {
    if (!datasetTable || typeof datasetTable.getColumns !== "function") return;
    withDatasetViewerRedrawBlocked(() => {
      const tableColumns = datasetTable.getColumns();
      const componentByField = new Map();
      tableColumns.forEach((column) => {
        const field = datasetViewerColumnField(column);
        if (!field) return;
        componentByField.set(field, column);
        const visible = visibleFields.has(field);
        if (visible && typeof column.show === "function") column.show();
        if (!visible && typeof column.hide === "function") column.hide();
      });
      reorderNormalColumns(visibleColumns, componentByField, tableColumns, visibleFields);
    });
  }

  function withDatasetViewerRedrawBlocked(callback) {
    const canBlock = datasetTable
      && datasetTable.initialized === true
      && typeof datasetTable.blockRedraw === "function"
      && typeof datasetTable.restoreRedraw === "function";
    if (!canBlock) {
      callback();
      return;
    }
    datasetTable.blockRedraw();
    try {
      callback();
    } finally {
      datasetTable.restoreRedraw();
    }
  }

  function reorderNormalColumns(visibleColumns, componentByField, tableColumns, visibleFields) {
    const desiredFields = visibleColumns.map((column) => column.field).filter((field) => componentByField.has(field));
    if (desiredFields.length < 2) return;
    const currentVisibleFields = tableColumns
      .map((column) => datasetViewerColumnField(column))
      .filter((field) => visibleFields.has(field));
    const alreadyOrdered = desiredFields.length === currentVisibleFields.length
      && desiredFields.every((field, index) => field === currentVisibleFields[index]);
    if (alreadyOrdered) return;
    const stableFields = longestInOrderFields(currentVisibleFields, desiredFields);
    let previousField = "";
    desiredFields.forEach((field, index) => {
      if (stableFields.has(field)) {
        previousField = field;
        return;
      }
      const column = componentByField.get(field);
      if (!column || typeof column.move !== "function") {
        previousField = field;
        return;
      }
      const nextStableField = desiredFields.slice(index + 1).find((candidate) => stableFields.has(candidate));
      try {
        if (nextStableField) {
          const nextColumn = componentByField.get(nextStableField);
          if (nextColumn && nextColumn !== column) column.move(nextColumn, false);
        } else if (previousField) {
          const previousColumn = componentByField.get(previousField);
          if (previousColumn && previousColumn !== column) column.move(previousColumn, true);
        }
      } catch (_) {
        // Reordering is cosmetic; preserve visibility and copy order if Tabulator rejects a stale component.
      }
      previousField = field;
    });
  }

  function longestInOrderFields(currentFields, desiredFields) {
    const desiredIndexByField = new Map(desiredFields.map((field, index) => [field, index]));
    const indexedFields = currentFields
      .map((field) => ({ field, index: desiredIndexByField.get(field) }))
      .filter((entry) => Number.isInteger(entry.index));
    if (!indexedFields.length) return new Set();
    const lengths = indexedFields.map(() => 1);
    const previous = indexedFields.map(() => -1);
    let best = 0;
    for (let index = 1; index < indexedFields.length; index += 1) {
      for (let candidate = 0; candidate < index; candidate += 1) {
        if (indexedFields[candidate].index >= indexedFields[index].index) continue;
        if (lengths[candidate] + 1 <= lengths[index]) continue;
        lengths[index] = lengths[candidate] + 1;
        previous[index] = candidate;
      }
      if (lengths[index] > lengths[best]) best = index;
    }
    const fields = new Set();
    for (let index = best; index !== -1; index = previous[index]) {
      fields.add(indexedFields[index].field);
    }
    return fields;
  }

  function applyNormalColumnSearch(terms, { mark = true } = {}) {
    if (!datasetTable) return;
    try {
      const search = normalColumnsForSearch(terms);
      currentColumns = search.columns;
      currentVisibleDatasetColumns = search.visibleDatasetColumns;
      if (!replaceNormalColumns(search)) {
        syncNormalColumnVisibilityAndOrder(search.visibleDatasetColumns, search.visibleFields);
      }
      syncNormalRenderedSelection();
      if (mark) markRendered(toolCache(TOOL_ID).requestKey);
    } catch (_) {
      // Search is client-only convenience; stale Tabulator instances can be ignored.
    }
  }

  function replaceNormalColumns(search) {
    if (!datasetTable || datasetTable.initialized !== true || typeof datasetTable.setColumns !== "function") return false;
    if (!normalSearchNeedsColumnReplacement(search.visibleDatasetColumns, search.visibleFields)) return false;
    snapshotDatasetViewerColumnWidths();
    const sorters = normalTableSorters();
    try {
      datasetTable.setColumns(search.columns);
      restoreNormalTableSorters(sorters);
      return true;
    } catch (_) {
      return false;
    }
  }

  function normalSearchNeedsColumnReplacement(visibleColumns, visibleFields) {
    if (!datasetTable || typeof datasetTable.getColumns !== "function") return false;
    let tableColumns = [];
    try {
      tableColumns = datasetTable.getColumns();
    } catch (_) {
      return false;
    }
    const desiredFields = visibleColumns.map((column) => column.field).filter(Boolean);
    const currentVisibleFields = tableColumns
      .map((column) => datasetViewerColumnField(column))
      .filter((field) => visibleFields.has(field));
    if (desiredFields.length < 2 || desiredFields.length !== currentVisibleFields.length) return false;
    const stableFields = longestInOrderFields(currentVisibleFields, desiredFields);
    return desiredFields.length - stableFields.size > NORMAL_SEARCH_REPLACE_MOVE_THRESHOLD;
  }

  function normalTableSorters() {
    if (!datasetTable || typeof datasetTable.getSorters !== "function") return [];
    try {
      const sorters = datasetTable.getSorters();
      return Array.isArray(sorters) ? sorters : [];
    } catch (_) {
      return [];
    }
  }

  function restoreNormalTableSorters(sorters) {
    if (!sorters.length || !datasetTable || typeof datasetTable.setSort !== "function") return;
    try {
      datasetTable.setSort(sorters);
    } catch (_) {
      // Search should remain useful even if a stale sorter references a replaced column.
    }
  }

  function transposedRowSearchTermIndex(row, terms) {
    return datasetViewerColumnSearchTermIndex({ name: row?.__field ?? row?.name }, terms);
  }

  function orderedTransposedRowsForSearch(terms, sourceRows = currentRows) {
    if (!terms.length) return sourceRows;
    const pinnedRows = sourceRows.filter((row) => isDatasetViewerColumnPinned(row?.__column_field));
    const groupedRows = terms.map(() => []);
    sourceRows.forEach((row) => {
      if (isDatasetViewerColumnPinned(row?.__column_field)) return;
      const matchIndex = transposedRowSearchTermIndex(row, terms);
      if (matchIndex !== -1) groupedRows[matchIndex].push(row);
    });
    return [...pinnedRows, ...groupedRows.flat()];
  }

  function applyTransposedSearch(terms, { mark = true } = {}) {
    if (!datasetTable) return;
    try {
      const rows = orderedTransposedRowsForSearch(terms);
      currentRenderedRows = rows;
      replaceTransposedRows(rows);
      syncTransposedVisibleColumnsFromRows(rows);
      syncTransposedRenderedSelection();
      if (mark) markRendered(toolCache(TOOL_ID).requestKey);
    } catch (_) {
      // Search is client-only convenience; stale Tabulator instances can be ignored.
    }
  }

  function replaceTransposedRows(rows) {
    if (!datasetTable || datasetTable.initialized !== true) return false;
    try {
      const replace = typeof datasetTable.replaceData === "function"
        ? datasetTable.replaceData(rows)
        : typeof datasetTable.setData === "function"
          ? datasetTable.setData(rows)
          : null;
      if (replace && typeof replace.then === "function") {
        replace.then(syncTransposedRenderedSelection).catch(() => {});
      }
      return Boolean(replace);
    } catch (_) {
      return false;
    }
  }

  function reconcileRenderedSearch(token, requestKey, syncSelection) {
    if (token !== renderToken || !datasetTable) return;
    applySearch({ mark: false });
    if (typeof syncSelection === "function") syncSelection();
    markRendered(requestKey);
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
    currentVisibleDatasetColumns = datasetColumnsFromTransposedRows(rows);
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
      : rowsToCsv(rowsForColumnCopy(), currentVisibleDatasetColumns);
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
    return currentVisibleDatasetColumns;
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
    if (!datasetTable) return currentRenderedRows;
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
    return currentRenderedRows;
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
      button.dataset.columnField = action.field || "";
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
    if (button.dataset.copyMode === "toggle-pin") {
      toggleDatasetViewerPinnedColumn(button.dataset.columnField || "");
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
    const shownMeta = `${displayed.toLocaleString()} shown`;
    const displayMeta = data?.has_more ? `First ${shownMeta}` : shownMeta;
    const pinnedNames = datasetViewerPinnedColumnNames();
    if (!pinnedNames.length) return displayMeta;
    return `${displayMeta} · ${pinnedNames.join(", ")} pinned`;
  }

  function currentSearchKey() {
    return String(state.datasetViewerSearch || "");
  }

  function markRendered(requestKey) {
    renderedRequestKey = requestKey || null;
    renderedSearch = currentSearchKey();
    renderedTranspose = Boolean(state.datasetViewerTranspose);
    renderedAlphabeticalColumns = Boolean(state.datasetViewerAlphabeticalColumns);
    renderedPinnedColumns = datasetViewerPinnedColumnsKey();
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
        && renderedPinnedColumns === datasetViewerPinnedColumnsKey()
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
