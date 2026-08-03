import { bindVerticalListNavigation } from "./shared/list-navigation.js";

const PROFILE_SUMMARY_MIN_WIDTH = 320;
const PROFILE_DETAIL_MIN_WIDTH = 260;
const PROFILE_RESIZE_STEP = 24;

export function columnProfileTypeLabel(column = {}) {
  const kind = column.kind || "unknown";
  const duckdbType = String(column.duckdb_type || "").toUpperCase();
  return duckdbType.includes("BOOL") ? "logical" : kind;
}

export function createColumnProfileTool({
  api,
  el,
  state,
  escapeHtml,
  formatNumber,
  formatRowMeta,
  measureToolRender,
  startToolTiming,
  setToolTimingFailed,
  syncDuckDbTimingFromData,
  syncClientTimingFromData,
  setStatus,
  setChartMessage,
  setGroupMeta,
  saveToolPresentation,
  stableRequestKey,
  toolCache,
  clearProfileDetailCache,
  copyTextToClipboard,
  showClipboardToast,
  activeFilterLabel,
}) {
  let profileSummaryPaneWidth = null;
  let profileTableNavigation = null;

  function isNumericKind(kind) {
    return kind === "numeric" || kind === "integer";
  }

  function setProfileGroupMeta(data, groupMeta) {
    const meta = el("profileGroupMeta");
    const skippedColumns = Array.isArray(data?.skipped_columns) ? data.skipped_columns : [];
    if (!skippedColumns.length) {
      meta.textContent = groupMeta || "";
      return;
    }
    const skippedLabel = profileSkippedLabel(skippedColumns.length);
    const detailHtml = profileSkippedPopoverHtml(skippedColumns);
    meta.innerHTML = escapeHtml(groupMeta || "").replace(
      escapeHtml(skippedLabel),
      `<button id="profileSkippedBtn" class="profile-skipped-button" type="button" aria-expanded="false" aria-controls="profileSkippedPopover">${escapeHtml(skippedLabel)}</button>${detailHtml}`,
    );
    const button = el("profileSkippedBtn");
    button?.addEventListener("click", toggleProfileSkippedPopover);
  }

  function buildProfileRequest() {
    if (!state.schema) return null;
    return {
      filter: state.activeFilter,
      mode: state.profileSummaryMode || "auto",
    };
  }

  function buildProfileDetailRequest(columnName = state.selectedProfileColumn) {
    if (!state.schema || !columnName) return null;
    return {
      column: columnName,
      filter: state.activeFilter,
    };
  }

  async function fetchProfileData(request, requestKey) {
    const requestSeq = state.profileRequestSeq + 1;
    state.profileRequestSeq = requestSeq;
    state.profileDetailRequestSeq += 1;
    setStatus("");
    setChartMessage("");
    setGroupMeta("column_profile", "Computing profile...");
    startToolTiming("column_profile");
    try {
      const data = await api("/api/column-profile/summary", { method: "POST", body: JSON.stringify(request), clientTiming: true });
      if (requestSeq !== state.profileRequestSeq) return;
      const cache = toolCache("column_profile");
      if (cache.requestKey !== requestKey) cache.details = new Map();
      cache.requestKey = requestKey;
      cache.data = data;
      syncDuckDbTimingFromData("column_profile", data);
      syncClientTimingFromData("column_profile", data);
      measureToolRender("column_profile", () => renderProfileData(data));
      return data;
    } catch (error) {
      if (requestSeq !== state.profileRequestSeq) return;
      setToolTimingFailed("column_profile");
      setGroupMeta("column_profile", "Profile failed");
      setChartMessage("");
      setStatus(error.message, true);
    }
  }

  function useCachedProfileData(cache) {
    state.lastProfileData = cache.data;
    measureToolRender("column_profile", () => {
      renderProfileData(cache.data);
    });
  }

  function renderProfileData(data) {
    state.lastProfileData = data;
    const columns = sortedProfileColumns(data.columns || []);
    const skippedColumns = Array.isArray(data.skipped_columns) ? data.skipped_columns : [];
    const skippedCount = skippedColumns.length;
    const totalColumnCount = columns.length + skippedCount;
    ensureSelectedProfileColumn(columns);
    renderProfileTable(data, columns);
    const rowMeta = formatRowMeta(data.row_count, data.filtered_row_count);
    const calculationMeta = profileCalculationMeta(data);
    const skippedMeta = skippedCount ? profileSkippedLabel(skippedCount) : "";
    const columnMeta = skippedCount
      ? `${columns.length.toLocaleString()} / ${totalColumnCount.toLocaleString()} columns profiled`
      : `${columns.length.toLocaleString()} columns`;
    const groupMeta = [columnMeta, skippedMeta, rowMeta].filter(Boolean).join(" · ");
    const chartMessage = "";
    setProfileGroupMeta(data, groupMeta);
    setProfileFilterMeta(data, calculationMeta);
    setStatus("");
    setChartMessage(chartMessage);
    saveToolPresentation("column_profile", { groupMeta, chartMessage });
    if (state.selectedProfileColumn) {
      renderProfileDetailLoading(state.selectedProfileColumn);
      scheduleProfileDetailRefresh(state.selectedProfileColumn);
    } else {
      renderProfileDetailEmpty(profileEmptyDetailMessage());
    }
  }

  function renderProfileTable(data, columns = sortedProfileColumns(data.columns || [])) {
    const tableScroll = captureProfileTableScroll();
    profileTableNavigation?.destroy();
    profileTableNavigation = null;
    closeProfileColumnContextMenu();
    ensureSelectedProfileColumn(columns);
    const visibleColumns = searchedProfileColumns(columns);
    const rows = columns.map((column) => `
      <tr class="profile-summary-row${column.name === state.selectedProfileColumn ? " selected" : ""}" data-profile-column="${escapeHtml(column.name)}" tabindex="${column.name === state.selectedProfileColumn ? "0" : "-1"}" aria-selected="${column.name === state.selectedProfileColumn ? "true" : "false"}"${profileColumnMatchesSearch(column.name) ? "" : " hidden"}>
        <td class="profile-column-name">${escapeHtml(column.name)}</td>
        <td>${profileTypeBadgeHtml(column)}</td>
        <td>${profileMissingHtml(column)}</td>
        <td>${profileDistinctHtml(column, data.filtered_row_count)}</td>
        <td>${profileRangeHtml(column)}</td>
      </tr>
    `).join("");
    const empty = profileTableEmptyHtml(columns, visibleColumns);
    const currentDetail = el("profileDetailPane")?.innerHTML || profileDetailEmptyHtml("Select a column to view details.");
    const profileGroupMeta = document.getElementById("profileGroupMeta");
    const profileFilter = document.getElementById("profileFilter");
    profileGroupMeta?.remove();
    profileFilter?.remove();
    const splitStyle = Number.isFinite(profileSummaryPaneWidth) && profileSummaryPaneWidth > 0
      ? ` style="--profile-summary-pane-width: ${Math.round(profileSummaryPaneWidth)}px"`
      : "";
    el("profileWrap").innerHTML = `
      <div class="profile-toolbar app-control-strip app-control-strip-row">
        <div class="profile-toolbar-group profile-columns-control">
          <h3 id="profileColumnsLabel" class="profile-toolbar-label">Columns</h3>
          <div class="profile-column-search-row">
            <input id="profileColumnSearch" class="search profile-column-search app-control-input" type="search" placeholder="Search columns" aria-labelledby="profileColumnsLabel" autocomplete="off" value="${escapeHtml(state.profileColumnSearch || "")}" />
            <button id="profileColumnSearchClear" class="profile-column-search-clear app-control-button app-command-button" type="button" title="Clear column search" aria-label="Clear column search">&times;</button>
          </div>
        </div>
        <div class="profile-toolbar-group-divider" aria-hidden="true"></div>
        ${profileSummaryActionsHtml()}
        <div class="profile-toolbar-meta-divider" aria-hidden="true"></div>
        <div id="profileMeta" class="profile-meta"></div>
      </div>
      <div class="profile-content"${splitStyle}>
        <div class="profile-summary-pane">
          <div class="profile-table-scroll">
            <table class="profile-table">
              <thead>
                <tr>
                  ${profileSortHeaderHtml("name", "Column")}
                  ${profileSortHeaderHtml("type", "Type")}
                  ${profileSortHeaderHtml("missing", "Missing")}
                  ${profileSortHeaderHtml("distinct", "Distinct")}
                  <th>Min / Max</th>
                </tr>
              </thead>
              <tbody>${rows}</tbody>
            </table>
          </div>
          ${empty}
        </div>
        <div id="profilePaneResizer" class="profile-pane-resizer app-resizer app-resizer--vertical" role="separator" aria-orientation="vertical" aria-label="Resize column summary and profile detail panels" tabindex="0"></div>
        <aside id="profileDetailPane" class="profile-detail-pane" aria-live="polite">${currentDetail}</aside>
      </div>
    `;
    attachProfileMeta(profileGroupMeta, profileFilter);
    bindProfileTable();
    restoreProfileTableScroll(tableScroll);
  }

  function attachProfileMeta(profileGroupMeta = null, profileFilter = null) {
    const meta = document.getElementById("profileMeta");
    const groupMeta = profileGroupMeta || document.getElementById("profileGroupMeta");
    const filter = profileFilter || document.getElementById("profileFilter");
    if (!meta || !groupMeta || !filter) return;
    if (groupMeta.parentElement !== meta) meta.append(groupMeta);
    if (filter.parentElement !== meta) meta.append(filter);
  }

  function captureProfileTableScroll() {
    const scroll = el("profileWrap")?.querySelector(".profile-table-scroll");
    return {
      left: Number(scroll?.scrollLeft || 0),
      top: Number(scroll?.scrollTop || 0),
    };
  }

  function restoreProfileTableScroll(position) {
    const scroll = el("profileWrap")?.querySelector(".profile-table-scroll");
    if (!scroll || !position) return;
    scroll.scrollLeft = Math.max(0, Number(position.left || 0));
    scroll.scrollTop = Math.max(0, Number(position.top || 0));
  }

  function bindProfileTable() {
    bindProfileSummaryModeControl();
    bindProfilePaneResizer();
    el("profileColumnSearch")?.addEventListener("input", handleProfileColumnSearch);
    el("profileColumnSearchClear")?.addEventListener("click", clearProfileColumnSearch);
    el("profileWrap").querySelectorAll("[data-profile-sort]").forEach((button) => {
      button.addEventListener("click", () => setProfileSort(button.dataset.profileSort));
    });
    const tableBody = el("profileWrap")?.querySelector(".profile-table tbody");
    if (tableBody) {
      profileTableNavigation = bindVerticalListNavigation({
        list: tableBody,
        itemSelector: "[data-profile-column]",
        getItemKey: (row) => row.dataset.profileColumn || "",
        onActivate: (columnName) => selectProfileColumn(columnName),
      });
    }
    el("profileWrap").querySelectorAll("[data-profile-column]").forEach((row) => {
      row.addEventListener("contextmenu", openProfileColumnContextMenu);
      row.addEventListener("keydown", (event) => {
        if (event.key !== "Enter" && event.key !== " ") return;
        event.preventDefault();
        selectProfileColumn(row.dataset.profileColumn || "");
      });
    });
  }

  function bindProfilePaneResizer() {
    const content = el("profileWrap")?.querySelector(".profile-content");
    const summary = content?.querySelector(".profile-summary-pane");
    const resizer = el("profilePaneResizer");
    if (!content || !summary || !resizer) return;

    const splitLimits = () => {
      const contentWidth = content.getBoundingClientRect().width || 0;
      const resizerWidth = resizer.getBoundingClientRect().width || 0;
      return {
        min: PROFILE_SUMMARY_MIN_WIDTH,
        max: Math.max(PROFILE_SUMMARY_MIN_WIDTH, contentWidth - resizerWidth - PROFILE_DETAIL_MIN_WIDTH),
      };
    };

    const syncAccessibility = (width) => {
      const limits = splitLimits();
      resizer.setAttribute("aria-valuemin", String(limits.min));
      resizer.setAttribute("aria-valuemax", String(Math.round(limits.max)));
      resizer.setAttribute("aria-valuenow", String(Math.round(width)));
    };

    const resizeTo = (width) => {
      if (window.matchMedia("(max-width: 640px)").matches) return;
      const limits = splitLimits();
      const clamped = Math.max(limits.min, Math.min(limits.max, width));
      profileSummaryPaneWidth = clamped;
      content.style.setProperty("--profile-summary-pane-width", `${Math.round(clamped)}px`);
      syncAccessibility(clamped);
    };

    const resizeFromClientX = (clientX) => {
      const contentRect = content.getBoundingClientRect();
      const resizerWidth = resizer.getBoundingClientRect().width || 0;
      resizeTo(clientX - contentRect.left - resizerWidth / 2);
    };

    const currentWidth = summary.getBoundingClientRect().width || PROFILE_SUMMARY_MIN_WIDTH;
    if (Number.isFinite(profileSummaryPaneWidth) && profileSummaryPaneWidth > 0) {
      resizeTo(profileSummaryPaneWidth);
    } else {
      syncAccessibility(currentWidth);
    }

    resizer.addEventListener("pointerdown", (event) => {
      if (window.matchMedia("(max-width: 640px)").matches) return;
      event.preventDefault();
      resizer.classList.add("dragging");
      document.body.classList.add("profile-pane-resizing");
      resizer.setPointerCapture?.(event.pointerId);
      const onMove = (moveEvent) => resizeFromClientX(moveEvent.clientX);
      const onUp = () => {
        resizer.classList.remove("dragging");
        document.body.classList.remove("profile-pane-resizing");
        window.removeEventListener("pointermove", onMove);
        window.removeEventListener("pointerup", onUp);
        window.removeEventListener("pointercancel", onUp);
      };
      window.addEventListener("pointermove", onMove);
      window.addEventListener("pointerup", onUp, { once: true });
      window.addEventListener("pointercancel", onUp, { once: true });
    });

    resizer.addEventListener("keydown", (event) => {
      if (!["ArrowLeft", "ArrowRight"].includes(event.key)) return;
      event.preventDefault();
      const width = summary.getBoundingClientRect().width || PROFILE_SUMMARY_MIN_WIDTH;
      resizeTo(width + (event.key === "ArrowRight" ? PROFILE_RESIZE_STEP : -PROFILE_RESIZE_STEP));
    });
  }

  function profileTableEmptyHtml(columns, visibleColumns) {
    if (!columns.length) {
      return '<div class="profile-empty">No columns were found in the loaded dataset.</div>';
    }
    return `<div id="profileSearchEmpty" class="profile-empty"${visibleColumns.length ? " hidden" : ""}>No columns match the search.</div>`;
  }

  function handleProfileColumnSearch(event) {
    state.profileColumnSearch = event.target.value;
    applyProfileColumnSearch();
  }

  function clearProfileColumnSearch() {
    const search = el("profileColumnSearch");
    if (!search) return;
    search.value = "";
    state.profileColumnSearch = "";
    applyProfileColumnSearch();
    search.focus();
  }

  function applyProfileColumnSearch() {
    if (!state.lastProfileData) return;
    const columns = sortedProfileColumns(state.lastProfileData.columns || []);
    const selectedBefore = state.selectedProfileColumn;
    ensureSelectedProfileColumn(columns);
    el("profileWrap").querySelectorAll("[data-profile-column]").forEach((row) => {
      row.hidden = !profileColumnMatchesSearch(row.dataset.profileColumn || "");
    });
    syncProfileSelectedRows();
    syncProfileSearchEmptyState(columns);
    if (state.selectedProfileColumn === selectedBefore) return;
    if (!state.selectedProfileColumn) {
      renderProfileDetailEmpty("No columns match the search.");
      return;
    }
    renderProfileDetailLoading(state.selectedProfileColumn);
    refreshSelectedProfileDetail();
  }

  function syncProfileSearchEmptyState(columns) {
    const empty = el("profileSearchEmpty");
    if (!empty) return;
    empty.hidden = Boolean(searchedProfileColumns(columns).length);
  }

  function bindProfileSummaryModeControl() {
    const group = el("profileSummaryMode");
    if (!group) return;
    group.querySelectorAll("[data-profile-summary-mode]").forEach((button) => {
      button.addEventListener("click", handleProfileSummaryModeChange);
    });
    syncProfileSummaryModeControl(false);
  }

  async function handleProfileSummaryModeChange(event) {
    const nextMode = normaliseProfileSummaryMode(event.currentTarget?.dataset.profileSummaryMode);
    if (nextMode === profileSummaryMode()) {
      syncProfileSummaryModeControl(false);
      return;
    }
    state.profileSummaryMode = nextMode;
    syncProfileSummaryModeControl(true);
    const request = buildProfileRequest();
    if (!request) {
      syncProfileSummaryModeControl(false);
      return;
    }
    await fetchProfileData(request, stableRequestKey(request));
    syncProfileSummaryModeControl(false);
  }

  function profileSummaryMode() {
    return normaliseProfileSummaryMode(state.profileSummaryMode);
  }

  function normaliseProfileSummaryMode(value) {
    return String(value || "auto").trim().toLowerCase() === "full" ? "full" : "auto";
  }

  function syncProfileSummaryModeControl(disabled = false) {
    const group = el("profileSummaryMode");
    if (!group) return;
    const selected = profileSummaryMode();
    group.setAttribute("aria-disabled", String(Boolean(disabled)));
    group.querySelectorAll("[data-profile-summary-mode]").forEach((button) => {
      const active = normaliseProfileSummaryMode(button.dataset.profileSummaryMode) === selected;
      button.disabled = Boolean(disabled);
      button.classList.toggle("active", active);
      button.setAttribute("aria-pressed", String(active));
    });
  }

  function profileCalculation(data) {
    const calculation = data?.calculation || {};
    const fullRowCount = Number(calculation.full_row_count ?? data?.filtered_row_count ?? data?.row_count ?? 0);
    const profiledRowCount = Number(calculation.profiled_row_count ?? fullRowCount);
    const exact = calculation.exact !== false || profiledRowCount >= fullRowCount;
    return {
      exact,
      fullAvailable: Boolean(calculation.full_available) && !exact,
      fullRowCount: Number.isFinite(fullRowCount) ? Math.max(0, fullRowCount) : 0,
      profiledRowCount: Number.isFinite(profiledRowCount) ? Math.max(0, profiledRowCount) : 0,
    };
  }

  function profileCalculationMeta(data) {
    const calculation = profileCalculation(data);
    if (calculation.exact) return "";
    return `preview ${calculation.profiledRowCount.toLocaleString()} rows`;
  }

  function setProfileFilterMeta(data, calculationMeta = profileCalculationMeta(data)) {
    const filterLabel = activeFilterLabel();
    const filterText = el("profileFilterText");
    if (!calculationMeta) {
      filterText.textContent = filterLabel;
      return;
    }
    filterText.innerHTML = `<span class="profile-warning-meta">${escapeHtml(calculationMeta)}</span> · ${escapeHtml(filterLabel)}`;
  }

  function profileSkippedLabel(count) {
    const safeCount = Math.max(0, Number(count) || 0);
    return `${safeCount.toLocaleString()} skipped`;
  }

  function profileSkippedPopoverHtml(skippedColumns) {
    const rows = skippedColumns.map((column) => `
      <div class="profile-skipped-row">
        <strong>${escapeHtml(column.name || "")}</strong>
        <span>${escapeHtml(column.error || "DuckDB could not read this column.")}</span>
      </div>
    `).join("");
    return `<div id="profileSkippedPopover" class="profile-skipped-popover" hidden>${rows}</div>`;
  }

  function toggleProfileSkippedPopover(event) {
    event.preventDefault();
    event.stopPropagation();
    const popover = el("profileSkippedPopover");
    const button = el("profileSkippedBtn");
    if (!popover || !button) return;
    const show = popover.hidden;
    popover.hidden = !show;
    button.setAttribute("aria-expanded", String(show));
    if (show) {
      window.addEventListener("pointerdown", closeProfileSkippedPopoverOnPointerDown, true);
      window.addEventListener("keydown", closeProfileSkippedPopoverOnEscape, true);
    } else {
      removeProfileSkippedPopoverListeners();
    }
  }

  function closeProfileSkippedPopover() {
    const popover = el("profileSkippedPopover");
    const button = el("profileSkippedBtn");
    if (popover) popover.hidden = true;
    if (button) button.setAttribute("aria-expanded", "false");
    removeProfileSkippedPopoverListeners();
  }

  function closeProfileSkippedPopoverOnPointerDown(event) {
    const popover = el("profileSkippedPopover");
    const button = el("profileSkippedBtn");
    if (!popover || popover.hidden || popover.contains(event.target) || button?.contains(event.target)) return;
    closeProfileSkippedPopover();
  }

  function closeProfileSkippedPopoverOnEscape(event) {
    if (event.key !== "Escape") return;
    closeProfileSkippedPopover();
  }

  function removeProfileSkippedPopoverListeners() {
    window.removeEventListener("pointerdown", closeProfileSkippedPopoverOnPointerDown, true);
    window.removeEventListener("keydown", closeProfileSkippedPopoverOnEscape, true);
  }

  function profileSummaryActionsHtml() {
    const selected = profileSummaryMode();
    return `
      <div class="profile-toolbar-group profile-rows-control">
        <h3 class="profile-toolbar-label">Rows</h3>
        <div class="profile-summary-actions">
          <div id="profileSummaryMode" class="profile-summary-mode" role="group" aria-label="Profile calculation rows">
            <button type="button" class="profile-summary-mode-option app-control-button ${selected === "auto" ? "active" : ""}" data-profile-summary-mode="auto" data-stable-label="Use 100k" aria-pressed="${String(selected === "auto")}">Use 100k</button>
            <button type="button" class="profile-summary-mode-option app-control-button ${selected === "full" ? "active" : ""}" data-profile-summary-mode="full" data-stable-label="Use all" aria-label="Use all rows" title="Use all rows" aria-pressed="${String(selected === "full")}">Use all</button>
          </div>
        </div>
      </div>
    `;
  }

  function profileColumnContextMenu() {
    let menu = document.getElementById("profileColumnContextMenu");
    if (menu) return menu;
    menu = document.createElement("div");
    menu.id = "profileColumnContextMenu";
    menu.className = "profile-context-menu";
    menu.hidden = true;
    menu.setAttribute("role", "menu");
    menu.innerHTML = '<button class="profile-context-menu-item" type="button" role="menuitem">Copy feature to clipboard</button>';
    menu.querySelector("button").addEventListener("click", copyProfileContextMenuFeature);
    document.body.append(menu);
    return menu;
  }

  function openProfileColumnContextMenu(event) {
    const columnName = event.currentTarget?.dataset?.profileColumn || "";
    if (!columnName) return;
    event.preventDefault();
    event.stopPropagation();
    closeProfileColumnContextMenu();
    const menu = profileColumnContextMenu();
    menu.dataset.profileColumn = columnName;
    menu.hidden = false;
    const rowRect = event.currentTarget.getBoundingClientRect();
    const clientX = event.clientX || rowRect.left + 12;
    const clientY = event.clientY || rowRect.top + Math.min(18, Math.max(8, rowRect.height / 2));
    positionProfileColumnContextMenu(menu, clientX, clientY);
    menu.querySelector("button")?.focus({ preventScroll: true });
    window.addEventListener("pointerdown", handleProfileColumnContextMenuPointerDown, true);
    window.addEventListener("keydown", handleProfileColumnContextMenuKeydown, true);
    window.addEventListener("resize", closeProfileColumnContextMenu, true);
    window.addEventListener("scroll", closeProfileColumnContextMenu, true);
  }

  function positionProfileColumnContextMenu(menu, clientX, clientY) {
    const margin = 8;
    menu.style.left = "0px";
    menu.style.top = "0px";
    const rect = menu.getBoundingClientRect();
    const maxLeft = Math.max(margin, window.innerWidth - rect.width - margin);
    const maxTop = Math.max(margin, window.innerHeight - rect.height - margin);
    const left = Math.min(Math.max(margin, clientX), maxLeft);
    const top = Math.min(Math.max(margin, clientY), maxTop);
    menu.style.left = `${left}px`;
    menu.style.top = `${top}px`;
  }

  async function copyProfileContextMenuFeature() {
    const menu = document.getElementById("profileColumnContextMenu");
    const columnName = menu?.dataset?.profileColumn || "";
    if (!columnName) {
      closeProfileColumnContextMenu();
      return;
    }
    const copied = await copyTextToClipboard(columnName);
    showClipboardToast(copied ? `Copied ${columnName} to clipboard` : "Could not copy feature to clipboard", !copied);
    closeProfileColumnContextMenu();
  }

  function handleProfileColumnContextMenuPointerDown(event) {
    const menu = document.getElementById("profileColumnContextMenu");
    if (!menu || menu.hidden || menu.contains(event.target)) return;
    closeProfileColumnContextMenu();
  }

  function handleProfileColumnContextMenuKeydown(event) {
    if (event.key !== "Escape") return;
    event.preventDefault();
    closeProfileColumnContextMenu();
  }

  function closeProfileColumnContextMenu() {
    const menu = document.getElementById("profileColumnContextMenu");
    if (menu) {
      menu.hidden = true;
      menu.dataset.profileColumn = "";
    }
    window.removeEventListener("pointerdown", handleProfileColumnContextMenuPointerDown, true);
    window.removeEventListener("keydown", handleProfileColumnContextMenuKeydown, true);
    window.removeEventListener("resize", closeProfileColumnContextMenu, true);
    window.removeEventListener("scroll", closeProfileColumnContextMenu, true);
  }

  function profileSortHeaderHtml(key, label) {
    const active = state.profileSort.key === key;
    const direction = active ? state.profileSort.direction : "";
    const ariaSort = active ? (direction === "desc" ? "descending" : "ascending") : "none";
    const indicator = active ? (direction === "desc" ? "v" : "^") : "";
    return `<th aria-sort="${ariaSort}">
      <button class="profile-sort-button" type="button" data-profile-sort="${key}">
        <span>${escapeHtml(label)}</span><span class="profile-sort-indicator" aria-hidden="true">${indicator}</span>
      </button>
    </th>`;
  }

  function setProfileSort(key) {
    if (!["name", "type", "missing", "distinct"].includes(key)) return;
    if (state.profileSort.key === key) {
      state.profileSort.direction = state.profileSort.direction === "asc" ? "desc" : "asc";
    } else {
      state.profileSort = { key, direction: "asc" };
    }
    if (state.lastProfileData) {
      const columns = sortedProfileColumns(state.lastProfileData.columns || []);
      ensureSelectedProfileColumn(columns);
      renderProfileTable(state.lastProfileData, columns);
    }
  }

  function ensureSelectedProfileColumn(columns) {
    const visibleColumns = searchedProfileColumns(columns);
    if (!visibleColumns.length) {
      state.selectedProfileColumn = "";
      return;
    }
    if (!visibleColumns.some((column) => column.name === state.selectedProfileColumn)) {
      state.selectedProfileColumn = visibleColumns[0].name;
    }
  }

  function searchedProfileColumns(columns) {
    return columns.filter((column) => profileColumnMatchesSearch(column.name));
  }

  function profileColumnMatchesSearch(columnName) {
    const query = profileColumnSearchQuery();
    if (!query) return true;
    return String(columnName || "").toLowerCase().includes(query);
  }

  function profileColumnSearchQuery() {
    return String(state.profileColumnSearch || "").trim().toLowerCase();
  }

  function profileEmptyDetailMessage() {
    return profileColumnSearchQuery() ? "No columns match the search." : "Select a column to view details.";
  }

  function selectProfileColumn(columnName) {
    if (!state.lastProfileData || !columnName) return;
    const exists = (state.lastProfileData.columns || []).some((column) => column.name === columnName);
    if (!exists) return;
    const changed = state.selectedProfileColumn !== columnName;
    if (!changed) return;
    state.selectedProfileColumn = columnName;
    syncProfileSelectedRows();
    renderProfileDetailLoading(columnName);
    refreshSelectedProfileDetail();
  }

  function syncProfileSelectedRows() {
    el("profileWrap").querySelectorAll("[data-profile-column]").forEach((row) => {
      const selected = row.dataset.profileColumn === state.selectedProfileColumn;
      row.classList.toggle("selected", selected);
      row.setAttribute("aria-selected", String(selected));
      row.tabIndex = selected ? 0 : -1;
    });
  }

  function scheduleProfileDetailRefresh(columnName) {
    window.setTimeout(() => {
      if (state.selectedProfileColumn === columnName) refreshSelectedProfileDetail();
    }, 0);
  }

  async function refreshSelectedProfileDetail() {
    const request = buildProfileDetailRequest();
    const columnName = request?.column || "";
    const detailRequestSeq = state.profileDetailRequestSeq + 1;
    state.profileDetailRequestSeq = detailRequestSeq;
    if (!request || !columnName) {
      renderProfileDetailEmpty("Select a column to view details.");
      return null;
    }
    const requestKey = stableRequestKey(request);
    const cache = toolCache("column_profile");
    const cached = cache.details.get(requestKey);
    if (cached) {
      measureToolRender("column_profile", () => renderProfileDetail(cached));
      return cached;
    }
    renderProfileDetailLoading(columnName);
    startToolTiming("column_profile");
    try {
      const data = await api("/api/column-profile/detail", { method: "POST", body: JSON.stringify(request), clientTiming: true });
      if (detailRequestSeq !== state.profileDetailRequestSeq || state.selectedProfileColumn !== columnName) return null;
      cache.details.set(requestKey, data);
      syncDuckDbTimingFromData("column_profile", data);
      syncClientTimingFromData("column_profile", data);
      measureToolRender("column_profile", () => renderProfileDetail(data));
      return data;
    } catch (error) {
      if (detailRequestSeq !== state.profileDetailRequestSeq) return null;
      setToolTimingFailed("column_profile");
      renderProfileDetailError(error.message);
      setStatus(error.message, true);
      return null;
    }
  }

  function profileDetailPane() {
    return el("profileDetailPane");
  }

  function profileDetailEmptyHtml(message) {
    return `<div class="profile-detail-state">${escapeHtml(message)}</div>`;
  }

  function renderProfileDetailEmpty(message) {
    const pane = profileDetailPane();
    if (pane) pane.innerHTML = profileDetailEmptyHtml(message);
  }

  function renderProfileDetailLoading(columnName) {
    const pane = profileDetailPane();
    if (!pane) return;
    pane.innerHTML = `
      <div class="profile-detail-header">
        <div>
          <h3 id="profileDetailTitle">${escapeHtml(columnName)}</h3>
          <div class="profile-detail-subtitle"><span>Loading profile...</span></div>
        </div>
      </div>
    `;
  }

  function renderProfileDetailError(message) {
    const pane = profileDetailPane();
    if (!pane) return;
    pane.innerHTML = `<div class="profile-detail-state profile-detail-error">${escapeHtml(message || "Profile detail failed")}</div>`;
  }

  function renderProfileDetail(data) {
    state.lastProfileDetailData = data;
    const isNumeric = isNumericKind(data.kind);
    const isTemporal = data.kind === "date" || data.kind === "datetime";
    const body = isNumeric || isTemporal
      ? `${profileDetailHistogramHtml(data.histogram || [], data.kind)}${profileStatsTableHtml(data.stats || {}, profileDetailStatKeys(data.kind))}`
      : profileValueCountsHtml(data.value_counts || [], data.filtered_row_count);
    const pane = profileDetailPane();
    if (!pane) return;
    pane.innerHTML = `
      <div class="profile-detail-header">
        <div>
          <h3 id="profileDetailTitle">${escapeHtml(data.name)}</h3>
          <div class="profile-detail-subtitle">${profileTypeBadgeHtml(data)} <span>${escapeHtml(data.duckdb_type || data.kind || "")}</span></div>
        </div>
      </div>
      ${profileDetailCountsHtml(data)}
      ${body}
    `;
    bindProfileDetail();
  }

  function profileDetailCountsHtml(data) {
    const filtered = Number(data.filtered_row_count || 0);
    const nonMissing = Number(data.non_missing_count || 0);
    const missing = Number(data.missing_count || 0);
    const distinct = Number(data.distinct_count || 0);
    const missingRate = filtered ? missing / filtered : 0;
    return `
      <div class="profile-detail-counts">
        <span><strong>${nonMissing.toLocaleString()}</strong> non-missing</span>
        <span><strong>${distinct.toLocaleString()}</strong> distinct</span>
        <span class="${missing > 0 ? "profile-detail-missing" : ""}"><strong>${missing.toLocaleString()}</strong> missing${missing > 0 ? ` (${formatProfilePercent(missingRate)})` : ""}</span>
        ${profileDetailSpecialCountHtml(data)}
      </div>
    `;
  }

  function profileDetailSpecialCountHtml(data) {
    if (isNumericKind(data.kind)) {
      return profileDetailCountBadgeHtml(Number(data.zero_count || 0), "zero", "profile-detail-zero");
    }
    if (data.kind === "categorical") {
      return profileDetailCountBadgeHtml(Number(data.blank_count || 0), "blank", "profile-detail-blank");
    }
    return "";
  }

  function profileDetailCountBadgeHtml(count, label, flagClass) {
    const safeCount = Number.isFinite(count) ? Math.max(0, count) : 0;
    const className = safeCount > 0 ? ` class="${flagClass}"` : "";
    return `<span${className}><strong>${safeCount.toLocaleString()}</strong> ${escapeHtml(label)}</span>`;
  }

  function profileDetailHistogramHtml(histogram, kind) {
    const bins = Array.isArray(histogram) ? histogram : [];
    const maxCount = Math.max(0, ...bins.map((bin) => Number(bin.count || 0)));
    if (!bins.length) return '<div class="profile-detail-empty">No non-missing values.</div>';
    const showBinLabels = profileHistogramUsesBinLabels(bins, kind);
    const bars = bins.map((bin) => {
      const count = Number(bin.count || 0);
      const height = maxCount ? Math.max(2, Math.round((count / maxCount) * 100)) : 0;
      const lower = formatProfileValue(bin.lower);
      const upper = formatProfileValue(bin.upper);
      const range = lower === upper ? lower : `${lower} to ${upper}`;
      const label = `${range}: ${count.toLocaleString()}`;
      return `<div class="profile-detail-bin" data-profile-bin-title="${escapeHtml(label)}" aria-label="${escapeHtml(label)}"><span class="profile-detail-bin-bar" style="height:${height}%"></span></div>`;
    }).join("");
    const guide = showBinLabels ? profileHistogramBinLabelsHtml(bins) : profileHistogramAxisHtml(bins, kind);
    return `<div class="profile-detail-histogram-wrap"><div class="profile-detail-histogram" aria-label="Histogram">${bars}</div>${guide}</div>`;
  }

  function profileHistogramUsesBinLabels(bins, kind) {
    if (kind !== "integer" || !bins.length || bins.length > 24) return false;
    if (!bins.every(profileHistogramBinIsExact)) return false;
    return bins.every((bin) => profileHistogramBinLabel(bin).length <= 4);
  }

  function profileHistogramBinIsExact(bin) {
    return bin.lower === bin.upper;
  }

  function profileHistogramBinLabelsHtml(bins) {
    const labelStyle = ` style="--profile-bin-label-size:${profileHistogramLabelFontSize(bins.length)}px"`;
    const labels = bins.map((bin) => `<span class="profile-detail-bin-label">${escapeHtml(profileHistogramBinLabel(bin))}</span>`).join("");
    return `<div class="profile-detail-bin-label-row"${labelStyle} aria-hidden="true">${labels}</div>`;
  }

  function profileHistogramAxisHtml(bins, kind) {
    const ticks = profileHistogramAxisTicks(bins, kind);
    const tickHtml = ticks.map((tick, index) => {
      const edgeClass = index === 0
        ? " profile-detail-histogram-axis-tick-start"
        : (index === ticks.length - 1 ? " profile-detail-histogram-axis-tick-end" : "");
      return `<span class="profile-detail-histogram-axis-tick${edgeClass}" style="left:${tick.position}%">${escapeHtml(tick.label)}</span>`;
    }).join("");
    return `<div class="profile-detail-histogram-axis" aria-hidden="true">${tickHtml}</div>`;
  }

  function profileHistogramAxisTicks(bins, kind) {
    if (!bins.length) return [];
    const targetCount = Math.min(bins.length + 1, profileHistogramAxisTickCount());
    const seenLabels = new Set();
    const ticks = [];
    for (let index = 0; index < targetCount; index += 1) {
      const position = targetCount === 1 ? 0 : Number(((index / (targetCount - 1)) * 100).toFixed(2));
      const label = formatProfileAxisValue(profileHistogramAxisValue(bins, index, targetCount), kind);
      if (!label || seenLabels.has(label)) continue;
      seenLabels.add(label);
      ticks.push({ position, label });
    }
    return ticks;
  }

  function profileHistogramAxisTickCount() {
    const width = profileDetailPane()?.clientWidth || 720;
    if (width >= 680) return 5;
    if (width >= 460) return 4;
    return 3;
  }

  function profileHistogramAxisValue(bins, index, tickCount) {
    const lastBin = bins[bins.length - 1];
    if (index === tickCount - 1) return lastBin.upper ?? lastBin.lower;
    const binIndex = tickCount === 1 ? 0 : Math.min(bins.length - 1, Math.round(((bins.length - 1) * index) / (tickCount - 1)));
    return bins[binIndex]?.lower ?? bins[binIndex]?.upper;
  }

  function formatProfileAxisValue(value, kind) {
    const formatted = formatProfileValue(value);
    if (kind !== "date" && kind !== "datetime") return formatted;
    return compactProfileTemporalValue(formatted);
  }

  function compactProfileTemporalValue(value) {
    const text = String(value || "");
    const midnight = text.match(/^(\d{4}-\d{2}-\d{2})T00:00(?::00(?:\.0+)?)?$/);
    if (midnight) return midnight[1];
    return text.replace("T", " ");
  }

  function profileHistogramBinLabel(bin) {
    return formatProfileValue(bin.lower);
  }

  function profileHistogramLabelFontSize(binCount) {
    const count = Math.max(1, Number(binCount) || 1);
    return Math.max(8, Math.min(9, 220 / count)).toFixed(2);
  }

  function profileDetailStatKeys(kind) {
    if (kind === "date" || kind === "datetime") return ["min", "p25", "median", "p75", "max"];
    return ["min", "p1", "p5", "p25", "median", "mean", "p75", "p95", "p99", "max", "sd"];
  }

  function profileStatsTableHtml(stats, keys) {
    const rows = keys.map((key) => `
      <tr>
        <th>${escapeHtml(profileStatLabel(key))}</th>
        <td>${escapeHtml(formatProfileValue(stats[key]))}</td>
      </tr>
    `).join("");
    return `<table class="profile-stats-table"><tbody>${rows}</tbody></table>`;
  }

  function profileStatLabel(key) {
    return {
      p1: "P1",
      p5: "P5",
      p25: "P25",
      p75: "P75",
      p95: "P95",
      p99: "P99",
      sd: "SD",
    }[key] || key.charAt(0).toUpperCase() + key.slice(1);
  }

  function profileValueCountsHtml(rows, filteredRowCount) {
    if (!rows.length) return '<div class="profile-detail-empty">No non-missing values.</div>';
    const filtered = Number(filteredRowCount || 0);
    const tableRows = sortedProfileValueCounts(rows).map((row) => {
      const count = Number(row.count || 0);
      const percent = filtered ? formatProfilePercentFixed(count / filtered) : "";
      return `
        <tr>
          <td>${escapeHtml(formatProfileValue(row.value))}</td>
          <td><span class="profile-count-value">${count.toLocaleString()}</span><span class="profile-count-percent">${escapeHtml(percent)}</span></td>
        </tr>
      `;
    }).join("");
    return `
      <div class="profile-count-table-scroll">
        <table class="profile-count-table">
          <thead><tr>${profileDetailSortHeaderHtml("value", "Value")}${profileDetailSortHeaderHtml("count", "Rows")}</tr></thead>
          <tbody>${tableRows}</tbody>
        </table>
      </div>
    `;
  }

  function profileDetailSortHeaderHtml(key, label) {
    const active = state.profileDetailSort.key === key;
    const direction = active ? state.profileDetailSort.direction : "";
    const ariaSort = active ? (direction === "desc" ? "descending" : "ascending") : "none";
    const indicator = active ? (direction === "desc" ? "v" : "^") : "";
    return `<th aria-sort="${ariaSort}">
      <button class="profile-count-sort-button" type="button" data-profile-detail-sort="${key}">
        <span>${escapeHtml(label)}</span><span class="profile-sort-indicator" aria-hidden="true">${indicator}</span>
      </button>
    </th>`;
  }

  function bindProfileDetail() {
    profileDetailPane()?.querySelectorAll(".profile-detail-bin[data-profile-bin-title]").forEach((bin) => {
      bin.addEventListener("pointerenter", showProfileHistogramTooltip);
      bin.addEventListener("pointermove", positionProfileHistogramTooltip);
      bin.addEventListener("pointerleave", hideProfileHistogramTooltip);
      bin.addEventListener("pointercancel", hideProfileHistogramTooltip);
    });
    profileDetailPane()?.querySelectorAll("[data-profile-detail-sort]").forEach((button) => {
      button.addEventListener("click", () => setProfileDetailSort(button.dataset.profileDetailSort));
    });
  }

  function profileHistogramTooltip() {
    let tooltip = document.getElementById("profileHistogramTooltip");
    if (!tooltip) {
      tooltip = document.createElement("div");
      tooltip.id = "profileHistogramTooltip";
      tooltip.className = "profile-histogram-tooltip";
      tooltip.hidden = true;
      document.body.appendChild(tooltip);
    }
    return tooltip;
  }

  function showProfileHistogramTooltip(event) {
    const label = event.currentTarget?.dataset?.profileBinTitle || "";
    if (!label) return;
    const tooltip = profileHistogramTooltip();
    tooltip.textContent = label;
    tooltip.hidden = false;
    tooltip.classList.add("visible");
    positionProfileHistogramTooltip(event);
  }

  function positionProfileHistogramTooltip(event) {
    const tooltip = document.getElementById("profileHistogramTooltip");
    if (!tooltip || tooltip.hidden) return;
    const offset = 10;
    const margin = 8;
    const rect = tooltip.getBoundingClientRect();
    const maxLeft = Math.max(margin, window.innerWidth - rect.width - margin);
    let left = event.clientX + offset;
    let top = event.clientY - rect.height - offset;
    if (top < margin) top = event.clientY + offset;
    left = Math.min(Math.max(margin, left), maxLeft);
    tooltip.style.left = `${left}px`;
    tooltip.style.top = `${Math.max(margin, top)}px`;
  }

  function hideProfileHistogramTooltip() {
    const tooltip = document.getElementById("profileHistogramTooltip");
    if (!tooltip) return;
    tooltip.hidden = true;
    tooltip.classList.remove("visible");
  }

  function setProfileDetailSort(key) {
    if (!["value", "count"].includes(key)) return;
    if (state.profileDetailSort.key === key) {
      state.profileDetailSort.direction = state.profileDetailSort.direction === "asc" ? "desc" : "asc";
    } else {
      state.profileDetailSort = { key, direction: key === "count" ? "desc" : "asc" };
    }
    if (state.lastProfileDetailData && !isNumericKind(state.lastProfileDetailData.kind)) {
      measureToolRender("column_profile", () => renderProfileDetail(state.lastProfileDetailData));
    }
  }

  function sortedProfileValueCounts(rows) {
    const key = state.profileDetailSort.key || "count";
    const direction = state.profileDetailSort.direction === "asc" ? 1 : -1;
    return [...rows]
      .map((row, index) => ({ row, index }))
      .sort((left, right) => {
        let compared = 0;
        if (key === "value") {
          compared = compareProfileText(formatProfileValue(left.row.value), formatProfileValue(right.row.value));
        } else {
          compared = Number(left.row.count || 0) - Number(right.row.count || 0);
        }
        return compared ? compared * direction : left.index - right.index;
      })
      .map((entry) => entry.row);
  }

  function sortedProfileColumns(columns) {
    const key = state.profileSort.key;
    if (!key) return [...columns];
    const direction = state.profileSort.direction === "desc" ? -1 : 1;
    return columns
      .map((column, index) => ({ column, index }))
      .sort((left, right) => {
        const compared = compareProfileColumns(left.column, right.column, key);
        return compared ? compared * direction : left.index - right.index;
      })
      .map((entry) => entry.column);
  }

  function compareProfileColumns(left, right, key) {
    if (key === "missing" || key === "distinct") {
      const field = key === "missing" ? "missing_count" : "distinct_count";
      const leftValue = Number(left[field] || 0);
      const rightValue = Number(right[field] || 0);
      return leftValue === rightValue ? compareProfileText(left.name, right.name) : leftValue - rightValue;
    }
    if (key === "type") {
      const leftType = `${columnProfileTypeLabel(left)}\u0000${left.duckdb_type || ""}`;
      const rightType = `${columnProfileTypeLabel(right)}\u0000${right.duckdb_type || ""}`;
      const compared = compareProfileText(leftType, rightType);
      return compared || compareProfileText(left.name, right.name);
    }
    return compareProfileText(left.name, right.name);
  }

  function compareProfileText(left, right) {
    return String(left || "").localeCompare(String(right || ""), undefined, { sensitivity: "base", numeric: true });
  }

  function profileTypeBadgeHtml(column) {
    const label = columnProfileTypeLabel(column);
    return `<span class="profile-type" title="${escapeHtml(column.duckdb_type || label)}">${escapeHtml(label)}</span>`;
  }

  function profileMissingHtml(column) {
    const missing = Number(column.missing_count || 0);
    if (!Number.isFinite(missing) || missing <= 0) {
      return '<span class="profile-missing-count">0</span>';
    }
    const rate = Number(column.missing_rate || 0);
    const percent = Number.isFinite(rate) ? ` (${formatProfilePercent(rate)})` : "";
    return `<span class="profile-badge profile-badge-warning profile-missing-count">${missing.toLocaleString()}${percent}</span>`;
  }

  function profileDistinctHtml(column, filteredRowCount) {
    const distinct = Number(column.distinct_count || 0);
    const filtered = Number(filteredRowCount || 0);
    const classes = ["profile-badge", "profile-badge-neutral"];
    let label = Number.isFinite(distinct) ? distinct.toLocaleString() : "0";
    if (distinct === 0) {
      classes.push("profile-badge-empty");
      label = "empty";
    } else if (distinct === 1) {
      classes.push("profile-badge-constant");
      label = `${label} constant`;
    } else if (distinct > 100) {
      classes.push("profile-badge-cardinality");
      label = `${label} high`;
    }
    return `<span class="${classes.join(" ")}">${escapeHtml(label)}</span>`;
  }

  function profileRangeHtml(column) {
    if (column.min === null || column.min === undefined || column.max === null || column.max === undefined) {
      return '<span class="profile-muted">-</span>';
    }
    return `<span class="profile-range">${escapeHtml(formatProfileValue(column.min))} <span>to</span> ${escapeHtml(formatProfileValue(column.max))}</span>`;
  }

  function formatProfileValue(value) {
    if (value === null || value === undefined) return "-";
    if (value === "") return '""';
    if (typeof value === "number") return formatNumber(value);
    if (typeof value === "boolean") return value ? "true" : "false";
    return String(value);
  }

  function formatProfilePercent(value) {
    const number = Number(value);
    if (!Number.isFinite(number)) return "";
    if (number > 0 && number < 0.001) return "<0.1%";
    return `${Number((number * 100).toFixed(1))}%`;
  }

  function formatProfilePercentFixed(value) {
    const number = Number(value);
    if (!Number.isFinite(number)) return "";
    return `${(number * 100).toFixed(1)}%`;
  }

  function closeMenus() {
    closeProfileSkippedPopover();
    closeProfileColumnContextMenu();
  }

  return {
    buildRequest: buildProfileRequest,
    fetchData: fetchProfileData,
    useCached: useCachedProfileData,
    render: renderProfileData,
    refreshSelectedDetail: refreshSelectedProfileDetail,
    closeMenus,
  };
}
