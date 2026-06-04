      import { createColumnProfileTool } from "./column-profile-tool.js";
      import { createLineBarTool } from "./line-bar-tool.js";
      import { createUkMapTool } from "./uk-map-tool.js";
      import { createGlmTool } from "./glm-tool.js";
      import { createGbmTool } from "./gbm-tool.js";
      import { createApiClient, monitorPath } from "./shared/api.js";
      import { createFormatters, escapeHtml } from "./shared/format.js";
      import {
        currentDataSource as schemaCurrentDataSource,
        dataSourceColumns as schemaDataSourceColumns,
        dataSourceForId as schemaDataSourceForId,
        dataSourceHasColumn as schemaDataSourceHasColumn,
        isModelPredictionColumn,
        isModelTool,
        preferredStartupSource as schemaPreferredStartupSource,
        sourceColumns as schemaSourceColumns,
        toolEnabled as schemaToolEnabled,
      } from "./shared/schema.js";
      import { createActionTimingController, freshActionTimings } from "./shared/timing.js";

      function paramsFromLocation() {
        const standardParams = new URLSearchParams(location.search);
        const expectedKeys = ["token", "tool", "source", "x", "xSource", "actual", "expected", "denominator", "postcode_area", "postcode_sector", "postcode_unit", "latitude", "longitude"];
        if (expectedKeys.some((key) => standardParams.has(key))) return standardParams;
        const rawSearch = location.search.startsWith("?") ? location.search.slice(1) : location.search;
        try {
          const decodedParams = new URLSearchParams(decodeURIComponent(rawSearch));
          return expectedKeys.some((key) => decodedParams.has(key)) ? decodedParams : standardParams;
        } catch (_) {
          return standardParams;
        }
      }

      const locationParams = paramsFromLocation();
      const token = locationParams.get("token") || "";
      const ACTION_RENDER_LABELS = {
        column_profile: "Profile render",
        line_bar: "Chart render",
        uk_map: "Map render",
        glm: "GLM render",
        gbm: "GBM render",
      };
      const state = {
        schema: null,
        x: null,
        xSource: "",
        sort: "alpha",
        lowGroup: "0",
        labels: "none",
        sidebarVisible: true,
        bandWidth: "0",
        quantileMode: "off",
        dateBucket: "none",
        transform: "none",
        sigma: "0",
        partialDependence: "none",
        source: locationParams.get("source") || "dataset",
        tool: "column_profile",
        view: "chart",
        mapLevel: "area",
        baseMap: "blank",
        mapPalette: "divergent",
        mapLineWeight: 1,
        mapOpacity: 1,
        mapHotspots: 0,
        mapLabelSize: 0,
        featureSort: "original",
        expectedSort: "original",
        openSidebarSection: null,
        collapsedKpiGroups: new Set(),
        kpiGroupsInitialised: false,
        activeKpiKey: "",
        activeKpiFormat: null,
        collapsedGlmModelGroups: new Set(),
        glmModelGroupsInitialised: false,
        collapsedGbmModelGroups: new Set(),
        gbmModelGroupsInitialised: false,
        filterOperator: "and",
        filterFooterCollapsed: true,
        filterSelectionMode: "single",
        collapsedSavedFilterThemes: new Set(),
        savedFilterThemesInitialised: false,
        activeFilter: "",
        profileSort: { key: "", direction: "asc" },
        profileDetailSort: { key: "count", direction: "desc" },
        profileSummaryMode: "auto",
        selectedProfileColumn: "",
        lastProfileData: null,
        lastProfileDetailData: null,
        lastData: null,
        lastMapData: null,
        toolCache: {
          column_profile: freshProfileCache(),
          line_bar: { requestKey: null, data: null, presentation: null },
          uk_map: { requestKey: null, data: null, presentation: null },
          glm: { requestKey: null, data: null, presentation: null },
          gbm: { requestKey: null, data: null, presentation: null },
        },
        actionTimings: freshActionTimings(),
        mapGeoJsonCache: {},
        mapPolygonLayerCache: {},
        mapPolygonRenderContext: null,
        mapStartupFitDone: false,
        renderedMapLevel: null,
        mapView: null,
        mapViewportSyncFrame: null,
        restoringMapView: false,
        pendingMapZoom: null,
        mapControlPosition: null,
        mapControlMoved: false,
        tablePage: 1,
        bandFeature: null,
        bandSuggestionPendingKey: null,
        bandSuggestionRequestSeq: 0,
        profileRequestSeq: 0,
        profileDetailRequestSeq: 0,
        chartRequestSeq: 0,
        mapRequestSeq: 0,
        glmRequestSeq: 0,
        gbmRequestSeq: 0,
      };

      const BAND_STEPS = makeBandSteps();
      let serverHeartbeatTimer = null;
      let startupTelemetryTimer = null;
      let startupProgressStartedAt = 0;
      let clipboardToastTimer = null;
      let stoppedOverlayShown = false;
      let faviconDataUrl = "";
      let datasetMetaBase = "";
      let datasetGlmCount = null;
      let datasetGbmCount = null;
      const el = (id) => document.getElementById(id);
      const api = createApiClient({ token });
      const {
        formatNumber,
        formatChartLabel,
        formatLineLabel,
        formatLineValue,
        formatWeightValue,
        formatFileSize,
        formatXLabel,
        formatRowMeta,
      } = createFormatters({ getActiveKpiFormat: () => state.activeKpiFormat });
      const {
        syncActionTimingMonitor,
        startToolTiming,
        setDuckDbTiming,
        setToolTimingFailed,
        setRenderTiming,
        measureToolRender,
        syncDuckDbTimingFromData,
        setClientTiming,
        syncClientTimingFromData,
      } = createActionTimingController({
        state,
        el,
        renderLabels: ACTION_RENDER_LABELS,
      });
      const columnProfileTool = createColumnProfileTool({
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
        setFilterRowMeta,
        setGroupMeta,
        saveToolPresentation,
        stableRequestKey,
        toolCache,
        clearProfileDetailCache,
        copyTextToClipboard,
        showClipboardToast,
        activeFilterLabel,
      });
      const lineBarTool = createLineBarTool({
        api,
        el,
        state,
        echartsImpl: echarts,
        escapeHtml,
        isModelPredictionColumn,
        formatNumber,
        formatChartLabel,
        formatLineLabel,
        formatLineValue,
        formatWeightValue,
        formatXLabel,
        formatRowMeta,
        measureToolRender,
        startToolTiming,
        setToolTimingFailed,
        syncDuckDbTimingFromData,
        syncClientTimingFromData,
        setStatus,
        setChartMessage,
        setFilterRowMeta,
        setGroupMeta,
        applyToolPresentation,
        saveToolPresentation,
        toolCache,
        sourceColumns: lineBarFeatureColumns,
        expectedColumns,
        selectedColumn: selectedLineBarColumn,
        numericColumns,
        dataSourceForId,
        dataSourceHasColumn,
        syncExpectedSourceFromSelection,
        toolEnabled,
        setTool,
        renderMetricTitle,
        getCss,
        bandSteps: BAND_STEPS,
        refreshLineBar,
      });
      const ukMapTool = createUkMapTool({
        api,
        el,
        state,
        leafletImpl: L,
        locationParams,
        escapeHtml,
        formatNumber,
        formatLineValue,
        formatWeightValue,
        formatRowMeta,
        measureToolRender,
        startToolTiming,
        setToolTimingFailed,
        syncDuckDbTimingFromData,
        syncClientTimingFromData,
        setStatus,
        setChartMessage,
        setFilterRowMeta,
        setGroupMeta,
        applyToolPresentation,
        saveToolPresentation,
        toolCache,
        syncActiveFilterLabels,
        renderMetricTitle,
        columnExists,
        numericColumnExists,
        refreshUkMap,
      });
      const glmTool = createGlmTool({
        api,
        clearToolCaches,
        copyTextToClipboard,
        el,
        escapeHtml,
        measureToolRender,
        renderExpectedNumerators: () => lineBarTool.renderExpectedNumerators(),
        renderFeatures: () => lineBarTool.renderFeatures(),
        saveToolPresentation,
        setChartMessage,
        setClientTiming,
        setDatasetGlmCount,
        setDuckDbTiming,
        setGroupMeta,
        setRenderTiming,
        setStatus,
        setAppReadyStatus: setReadyBadge,
        setToolTimingFailed,
        showClipboardToast,
        startToolTiming,
        state,
        syncClientTimingFromData,
        syncDuckDbTimingFromData,
        toolCache,
        updateAxisControls: () => lineBarTool.updateAxisControls(),
        refreshActiveTool,
        reloadSchema: reloadSchemaAfterModelMutation,
      });
      const gbmTool = createGbmTool({
        api,
        clearToolCaches,
        el,
        escapeHtml,
        measureToolRender,
        renderExpectedNumerators: () => lineBarTool.renderExpectedNumerators(),
        renderFeatures: () => lineBarTool.renderFeatures(),
        saveToolPresentation,
        setChartMessage,
        setClientTiming,
        setDuckDbTiming,
        setGroupMeta,
        setRenderTiming,
        setStatus,
        setAppReadyStatus: setReadyBadge,
        setToolTimingFailed,
        startToolTiming,
        state,
        canNavigateToLineBarFeature: (featureName) => lineBarTool.canNavigateToFeature(featureName),
        navigateToLineBarFeature: (featureName) => lineBarTool.navigateToFeature(featureName),
        syncClientTimingFromData,
        syncDuckDbTimingFromData,
        toolCache,
        updateAxisControls: () => lineBarTool.updateAxisControls(),
        refreshActiveTool,
        setDatasetGbmCount,
        reloadSchema: reloadSchemaAfterModelMutation,
      });

      function monitorUrl() {
        return monitorPath({ token, href: location.href });
      }

      function syncMonitorLink() {
        const link = el("monitorLink");
        if (link) link.href = monitorUrl();
      }

      function startServerHeartbeat() {
        if (serverHeartbeatTimer) return;
        serverHeartbeatTimer = window.setInterval(checkServerHealth, 2000);
      }

      function stopServerHeartbeat() {
        if (!serverHeartbeatTimer) return;
        window.clearInterval(serverHeartbeatTimer);
        serverHeartbeatTimer = null;
      }

      function startupElapsedSeconds() {
        if (!startupProgressStartedAt) return 0;
        return Math.max(0, Math.round((performance.now() - startupProgressStartedAt) / 1000));
      }

      function setStartupProgress(message, stateClass = "") {
        const node = el("startupProgress");
        if (!node) return;
        node.textContent = message || "";
        node.classList.toggle("ready", stateClass === "ready");
        node.classList.toggle("error", stateClass === "error");
        node.classList.toggle("hidden", !message);
      }

      function setReadyBadge(message = "Ready") {
        setStartupProgress(message || "Ready", "ready");
      }

      function currentTelemetryAction(snapshot) {
        const clients = Array.isArray(snapshot?.clients) ? snapshot.clients : [];
        return clients.find((client) => client.current_action) || null;
      }

      function startStartupTelemetryPolling(fallbackLabel) {
        stopStartupTelemetryPolling();
        startupProgressStartedAt = performance.now();
        startupTelemetryTimer = window.setInterval(async () => {
          try {
            const snapshot = await api("/api/telemetry", { method: "GET" });
            const current = currentTelemetryAction(snapshot);
            const elapsed = current?.current_action_seconds ?? startupElapsedSeconds();
            if (current?.current_action) {
              setStartupProgress(`${current.current_action} · ${Math.round(elapsed)}s`);
            } else {
              setStartupProgress(`${fallbackLabel} · ${startupElapsedSeconds()}s`);
            }
          } catch (_) {
            setStartupProgress(`${fallbackLabel} · ${startupElapsedSeconds()}s`);
          }
        }, 1000);
      }

      function stopStartupTelemetryPolling() {
        if (!startupTelemetryTimer) return;
        window.clearInterval(startupTelemetryTimer);
        startupTelemetryTimer = null;
      }

      async function cacheShutdownIcon() {
        try {
          const response = await fetch("/favicon.ico", { cache: "force-cache" });
          if (!response.ok) return;
          const blob = await response.blob();
          faviconDataUrl = await new Promise((resolve, reject) => {
            const reader = new FileReader();
            reader.addEventListener("load", () => resolve(String(reader.result || "")), { once: true });
            reader.addEventListener("error", () => reject(reader.error), { once: true });
            reader.readAsDataURL(blob);
          });
        } catch (_) {
          faviconDataUrl = "";
        }
      }

      async function checkServerHealth() {
        if (stoppedOverlayShown) return;
        try {
          await fetch("/api/health", {
            cache: "no-store",
            headers: {
              "x-lucidum-token": token,
            },
          });
        } catch (_) {
          showStoppedOverlay();
        }
      }

      function setStatus(message, isError = false) {
        el("status").textContent = message || "";
        el("status").classList.toggle("error", isError);
        el("status").classList.toggle("hidden", !message);
      }

      function copyTextToClipboard(text) {
        if (navigator.clipboard?.writeText) {
          return navigator.clipboard.writeText(text)
            .then(() => true)
            .catch(() => fallbackCopyTextToClipboard(text));
        }
        return Promise.resolve(fallbackCopyTextToClipboard(text));
      }

      function fallbackCopyTextToClipboard(text) {
        const input = document.createElement("textarea");
        input.value = text;
        input.setAttribute("readonly", "");
        input.style.position = "fixed";
        input.style.left = "-9999px";
        input.style.top = "0";
        document.body.append(input);
        input.select();
        try {
          return document.execCommand("copy");
        } catch (_) {
          return false;
        } finally {
          input.remove();
        }
      }

      function showClipboardToast(message, isError = false) {
        let toast = document.getElementById("clipboardToast");
        if (!toast) {
          toast = document.createElement("div");
          toast.id = "clipboardToast";
          toast.className = "clipboard-toast";
          toast.hidden = true;
          toast.setAttribute("role", "status");
          toast.setAttribute("aria-live", "polite");
          document.body.append(toast);
        }
        if (clipboardToastTimer) {
          window.clearTimeout(clipboardToastTimer);
          clipboardToastTimer = null;
        }
        toast.textContent = message || "";
        toast.classList.toggle("error", isError);
        toast.hidden = !message;
        if (!message) return;
        clipboardToastTimer = window.setTimeout(() => {
          toast.hidden = true;
          clipboardToastTimer = null;
        }, 1800);
      }

      function setChartMessage(message) {
        const displayMessage = message || "";
        el("chartMessage").textContent = displayMessage;
        const hiddenForView = state.tool === "line_bar" && state.view !== "chart";
        el("chartMessage").classList.toggle("hidden", !displayMessage || hiddenForView);
      }

      function setGroupMeta(tool, message) {
        const id = tool === "uk_map"
          ? "mapGroupMeta"
          : (tool === "column_profile" ? "profileGroupMeta" : (isModelTool(tool) ? "modelToolGroupMeta" : "lineBarGroupMeta"));
        el(id).textContent = message || "";
      }

      function activeFilterLabel() {
        return state.activeFilter || "no filter";
      }

      function syncActiveFilterLabels() {
        const label = activeFilterLabel();
        el("profileFilter").textContent = label;
        el("lineBarFilter").textContent = label;
        el("modelToolFilter").textContent = label;
        el("mapControlFilter").textContent = label;
      }

      function setFilterRowMeta(rowCount, filteredRowCount = rowCount) {
        const meta = formatRowMeta(rowCount, filteredRowCount);
        if (meta) el("filterRowMeta").textContent = meta;
      }

      function makeBandSteps() {
        const steps = [];
        for (let exponent = -8; exponent <= 12; exponent += 1) {
          const multiplier = 10 ** exponent;
          steps.push(1 * multiplier, 2 * multiplier, 5 * multiplier);
        }
        steps.push(4, 7, 12);
        return [...new Set(steps)].sort((a, b) => a - b);
      }

      function numericColumns() {
        return sourceColumns().filter((c) => isNumericKind(c.kind));
      }

      function selectedColumn() {
        return sourceColumns().find((c) => c.name === state.x);
      }

      function lineBarFeatureColumns() {
        const currentSource = state.source || "dataset";
        const currentKind = currentDataSource()?.kind || "";
        const columns = sourceColumns().map((column) => ({
          ...column,
          source_id: column.source_id || currentSource,
        }));
        if (currentKind === "gbm_shap_long") return columns;
        const seen = new Set(columns.map((column) => `${column.source_id || currentSource}\u0000${column.name}`));
        for (const column of activePredictionColumns()) {
          const sourceId = column.source_id || "";
          const key = `${sourceId}\u0000${column.name}`;
          if (!sourceId || seen.has(key)) continue;
          columns.push(column);
          seen.add(key);
        }
        return columns;
      }

      function lineBarColumnSourceId(column) {
        return column?.source_id || state.source || "dataset";
      }

      function selectedLineBarColumn() {
        const sourceId = state.xSource || state.source || "dataset";
        const columns = lineBarFeatureColumns();
        return columns.find((column) => column.name === state.x && lineBarColumnSourceId(column) === sourceId)
          || columns.find((column) => column.name === state.x)
          || null;
      }

      function lineBarColumnExists(name, sourceId = "") {
        const columnName = String(name || "");
        if (!columnName) return false;
        return lineBarFeatureColumns().some((column) => (
          column.name === columnName && (!sourceId || lineBarColumnSourceId(column) === sourceId)
        ));
      }

      function lineBarFeatureSourceForName(name, preferredSource = "") {
        const columnName = String(name || "");
        const columns = lineBarFeatureColumns();
        const preferred = String(preferredSource || "");
        const match = columns.find((column) => (
          column.name === columnName && (!preferred || lineBarColumnSourceId(column) === preferred)
        )) || columns.find((column) => column.name === columnName);
        return lineBarColumnSourceId(match);
      }

      function syncLineBarXFallback() {
        if (lineBarColumnExists(state.x, state.xSource)) return;
        const first = lineBarFeatureColumns()[0] || null;
        state.x = first?.name || null;
        state.xSource = lineBarColumnSourceId(first);
      }

      function currentDataSource() {
        return schemaCurrentDataSource(state.schema, state.source || "dataset");
      }

      function sourceColumns() {
        return schemaSourceColumns(state.schema, state.source || "dataset");
      }

      function dataSourceForId(sourceId) {
        return schemaDataSourceForId(state.schema, sourceId);
      }

      function dataSourceColumns(sourceId) {
        return schemaDataSourceColumns(state.schema, sourceId);
      }

      function dataSourceHasColumn(sourceId, columnName) {
        return schemaDataSourceHasColumn(state.schema, sourceId, columnName);
      }

      function preferredStartupSource(availableSources, requestedSource) {
        return schemaPreferredStartupSource(availableSources, requestedSource);
      }

      function toolEnabled(id) {
        return schemaToolEnabled(state.schema, id);
      }

      function freshProfileCache() {
        return { requestKey: null, data: null, presentation: null, details: new Map() };
      }

      function freshToolCache() {
        return {
          column_profile: freshProfileCache(),
          line_bar: { requestKey: null, data: null, presentation: null },
          uk_map: { requestKey: null, data: null, presentation: null },
          glm: { requestKey: null, data: null, presentation: null },
          gbm: { requestKey: null, data: null, presentation: null },
        };
      }

      function clearToolCaches(options = {}) {
        const preserve = new Set(Array.isArray(options.preserve) ? options.preserve : []);
        const previousToolCache = state.toolCache || {};
        const previousActionTimings = state.actionTimings || {};
        state.toolCache = freshToolCache();
        state.actionTimings = freshActionTimings();
        preserve.forEach((tool) => {
          if (previousToolCache[tool]) state.toolCache[tool] = previousToolCache[tool];
          if (previousActionTimings[tool]) state.actionTimings[tool] = previousActionTimings[tool];
        });
        if (!preserve.has("column_profile")) {
          columnProfileTool.resetSummaryMode();
          state.lastProfileData = null;
          clearProfileDetailCache();
        }
        state.lastData = null;
        ukMapTool.resetRenderState();
        syncActionTimingMonitor();
      }

      function clearProfileDetailCache() {
        const cache = toolCache("column_profile");
        cache.details = new Map();
        state.lastProfileDetailData = null;
        state.profileDetailRequestSeq += 1;
      }

      function toolCache(tool) {
        if (!state.toolCache[tool]) {
          state.toolCache[tool] = tool === "column_profile" ? freshProfileCache() : { requestKey: null, data: null, presentation: null };
        }
        if (tool === "column_profile" && !(state.toolCache[tool].details instanceof Map)) {
          state.toolCache[tool].details = new Map();
        }
        return state.toolCache[tool];
      }

      function normaliseForRequestKey(value) {
        if (Array.isArray(value)) {
          return value.map(normaliseForRequestKey);
        }
        if (value && typeof value === "object") {
          return Object.keys(value).sort().reduce((result, key) => {
            result[key] = normaliseForRequestKey(value[key]);
            return result;
          }, {});
        }
        return value;
      }

      function stableRequestKey(request) {
        return JSON.stringify(normaliseForRequestKey(request));
      }

      function saveToolPresentation(tool, presentation) {
        toolCache(tool).presentation = {
          groupMeta: presentation.groupMeta || "",
          status: presentation.status || "",
          statusError: Boolean(presentation.statusError),
          chartMessage: presentation.chartMessage || "",
        };
      }

      function applyToolPresentation(tool) {
        const presentation = toolCache(tool).presentation;
        if (!presentation) return;
        setGroupMeta(tool, presentation.groupMeta);
        setStatus(presentation.status, presentation.statusError);
        setChartMessage(presentation.chartMessage);
      }

      function toolHandler(tool) {
        if (tool === "column_profile") {
          return {
            buildRequest: () => columnProfileTool.buildRequest(),
            fetch: (request, requestKey) => columnProfileTool.fetchData(request, requestKey),
            useCached: (cache) => columnProfileTool.useCached(cache),
          };
        }
        if (tool === "line_bar") {
          return {
            buildRequest: () => lineBarTool.buildRequest(),
            fetch: (request, requestKey) => lineBarTool.fetchData(request, requestKey),
            useCached: (cache, options) => lineBarTool.useCached(cache, options),
          };
        }
        if (tool === "uk_map") {
          return {
            buildRequest: () => ukMapTool.buildRequest(),
            fetch: (request, requestKey) => ukMapTool.fetchData(request, requestKey),
            useCached: (cache) => ukMapTool.useCached(cache),
            handleMissingRequest: () => ukMapTool.showMissingRequest(),
          };
        }
        if (tool === "glm") {
          return {
            buildRequest: () => glmTool.buildRequest(),
            fetch: (request, requestKey) => glmTool.fetchData(request, requestKey),
            useCached: (cache, options) => glmTool.useCached(cache, options),
          };
        }
        if (tool === "gbm") {
          return {
            buildRequest: () => gbmTool.buildRequest(),
            fetch: (request, requestKey) => gbmTool.fetchData(request, requestKey),
            useCached: (cache, options) => gbmTool.useCached(cache, options),
          };
        }
        return {
          buildRequest: () => lineBarTool.buildRequest(),
          fetch: (request, requestKey) => lineBarTool.fetchData(request, requestKey),
          useCached: (cache, options) => lineBarTool.useCached(cache, options),
        };
      }

      async function refreshTool(tool, options = {}) {
        const handler = toolHandler(tool);
        const request = handler.buildRequest();
        if (!request) {
          handler.handleMissingRequest?.();
          return null;
        }
        const requestKey = stableRequestKey(request);
        const cache = toolCache(tool);
        if (!options.force && cache.data && cache.requestKey === requestKey) {
          await handler.useCached(cache, options);
          return cache.data;
        }
        return handler.fetch(request, requestKey);
      }

      function refreshActiveTool(options = {}) {
        return refreshTool(state.tool, options);
      }

      function refreshLineBar(options = {}) {
        return refreshTool("line_bar", options);
      }

      function refreshUkMap(options = {}) {
        return refreshTool("uk_map", options);
      }

      function chooseDefaultTool() {
        const requested = locationParams.get("tool");
        if (requested && toolEnabled(requested)) return requested;
        if (toolEnabled("column_profile")) return "column_profile";
        if (toolEnabled("line_bar")) return "line_bar";
        if (toolEnabled("uk_map")) return "uk_map";
        if (toolEnabled("glm")) return "glm";
        if (toolEnabled("gbm")) return "gbm";
        return "column_profile";
      }

      function renderToolSelector() {
        const profileEnabled = toolEnabled("column_profile");
        const lineBarEnabled = toolEnabled("line_bar");
        const ukMapEnabled = toolEnabled("uk_map");
        const glmEnabled = toolEnabled("glm");
        const gbmEnabled = toolEnabled("gbm");
        el("profileTool").disabled = !profileEnabled;
        el("lineBarTool").disabled = !lineBarEnabled;
        el("ukMapTool").disabled = !ukMapEnabled;
        el("glmTool").disabled = !glmEnabled;
        el("gbmTool").disabled = !gbmEnabled;
        el("profileTool").classList.toggle("hidden", !profileEnabled);
        el("lineBarTool").classList.toggle("hidden", !lineBarEnabled);
        el("ukMapTool").classList.toggle("hidden", !ukMapEnabled);
        el("glmTool").classList.toggle("hidden", !glmEnabled);
        el("gbmTool").classList.toggle("hidden", !gbmEnabled);
        el("toolSelectorSection").classList.toggle("hidden", !(profileEnabled || lineBarEnabled || ukMapEnabled || glmEnabled || gbmEnabled));
      }

      function schemaFileMeta() {
        const path = state.schema?.path?.split(/[\\/]/).pop() || "";
        const fileSize = formatFileSize(state.schema?.file_size);
        return fileSize ? `${path} · ${fileSize}` : path;
      }

      function renderDatasetMeta(fileMeta = datasetMetaBase, gbmCount = datasetGbmCount, glmCount = datasetGlmCount) {
        datasetMetaBase = String(fileMeta || "");
        const numericCount = Number(gbmCount);
        datasetGbmCount = Number.isFinite(numericCount) && numericCount >= 0 ? Math.trunc(numericCount) : null;
        const numericGlmCount = Number(glmCount);
        datasetGlmCount = Number.isFinite(numericGlmCount) && numericGlmCount >= 0 ? Math.trunc(numericGlmCount) : null;
        const target = el("datasetMeta");
        const rows = Number(state.schema?.row_count || 0).toLocaleString();
        const columns = Number(state.schema?.columns?.length || 0).toLocaleString();
        target.textContent = "";
        target.append(document.createTextNode(`${datasetMetaBase} · ${rows} rows · ${columns} columns`));
        if (datasetGlmCount !== null && toolEnabled("glm")) {
          target.append(document.createTextNode(" · "));
          const button = document.createElement("button");
          button.type = "button";
          button.className = "dataset-meta-gbm-link dataset-meta-glm-link";
          button.textContent = `GLMs (${datasetGlmCount.toLocaleString()})`;
          button.title = "Open GLM Model navigator";
          button.setAttribute("aria-label", `Open saved GLMs, ${datasetGlmCount.toLocaleString()} models`);
          button.addEventListener("click", openGlmModelNavigator);
          target.append(button);
        }
        if (datasetGbmCount === null || !toolEnabled("gbm")) return;
        target.append(document.createTextNode(" · "));
        const button = document.createElement("button");
        button.type = "button";
        button.className = "dataset-meta-gbm-link";
        button.textContent = `GBMs (${datasetGbmCount.toLocaleString()})`;
        button.title = "Open GBM Model navigator";
        button.setAttribute("aria-label", `Open saved GBMs, ${datasetGbmCount.toLocaleString()} models`);
        button.addEventListener("click", openGbmModelNavigator);
        target.append(button);
      }

      async function refreshDatasetGlmCount() {
        if (!toolEnabled("glm")) {
          renderDatasetMeta(datasetMetaBase, datasetGbmCount, null);
          return;
        }
        try {
          const payload = await api("/api/glm/models", { method: "GET" });
          const models = Array.isArray(payload?.models) ? payload.models : [];
          renderDatasetMeta(datasetMetaBase, datasetGbmCount, models.length);
        } catch (_) {
          renderDatasetMeta(datasetMetaBase, datasetGbmCount, null);
        }
      }

      async function refreshDatasetGbmCount() {
        if (!toolEnabled("gbm")) {
          renderDatasetMeta(datasetMetaBase, null, datasetGlmCount);
          return;
        }
        try {
          const payload = await api("/api/gbm/models", { method: "GET" });
          const models = Array.isArray(payload?.models) ? payload.models : [];
          renderDatasetMeta(datasetMetaBase, models.length, datasetGlmCount);
        } catch (_) {
          renderDatasetMeta(datasetMetaBase, null, datasetGlmCount);
        }
      }

      function setDatasetGbmCount(count) {
        if (!toolEnabled("gbm")) return;
        renderDatasetMeta(datasetMetaBase, count, datasetGlmCount);
      }

      function setDatasetGlmCount(count) {
        if (!toolEnabled("glm")) return;
        renderDatasetMeta(datasetMetaBase, datasetGbmCount, count);
      }

      function openGlmModelNavigator() {
        if (!toolEnabled("glm")) return;
        glmTool.openModelNavigator();
        setTool("glm");
      }

      function openGbmModelNavigator() {
        if (!toolEnabled("gbm")) return;
        gbmTool.openModelNavigator();
        setTool("gbm");
      }

      function setTool(tool, refresh = true) {
        if (!toolEnabled(tool)) return;
        const previousTool = state.tool;
        if (previousTool === "uk_map" && tool !== "uk_map") ukMapTool.captureView("tool-switch");
        if (previousTool === "column_profile" && tool !== "column_profile") columnProfileTool.closeMenus();
        state.tool = tool;
        el("profileTool").classList.toggle("active", tool === "column_profile");
        el("lineBarTool").classList.toggle("active", tool === "line_bar");
        el("ukMapTool").classList.toggle("active", tool === "uk_map");
        el("glmTool").classList.toggle("active", tool === "glm");
        el("gbmTool").classList.toggle("active", tool === "gbm");
        document.querySelector(".sidebar-metric-section")?.classList.toggle("hidden", tool === "column_profile");
        glmTool.syncSidebarFromSchema();
        gbmTool.syncSidebarFromSchema();
        syncSidebarAccordion();
        el("lineBarToolbar").classList.toggle("hidden", tool !== "line_bar");
        el("visualArea").classList.toggle("map-mode", tool === "uk_map");
        el("visualArea").classList.toggle("profile-mode", tool === "column_profile");
        el("visualArea").classList.toggle("model-mode", isModelTool(tool));
        el("chartSideControls").classList.toggle("hidden", tool !== "line_bar");
        el("chartControlsResizer").classList.toggle("hidden", tool !== "line_bar");
        el("lineBarTabs").classList.toggle("hidden", tool !== "line_bar");
        el("profileGroupMeta").classList.toggle("hidden", tool !== "column_profile");
        el("profileFilter").classList.toggle("hidden", tool !== "column_profile");
        el("lineBarGroupMeta").classList.toggle("hidden", tool !== "line_bar");
        el("lineBarFilter").classList.toggle("hidden", tool !== "line_bar");
        el("modelToolGroupMeta").classList.toggle("hidden", !isModelTool(tool) || tool === "gbm");
        el("modelToolFilter").classList.add("hidden");
        el("mapFloatingControl").classList.toggle("hidden", tool !== "uk_map");
        el("mapLegend").classList.toggle("hidden", tool !== "uk_map" || !el("mapLegend").textContent);
        el("profileWrap").classList.toggle("hidden", tool !== "column_profile");
        el("modelToolWrap").classList.toggle("hidden", !isModelTool(tool));
        syncActiveFilterLabels();
        syncActionTimingMonitor(tool);
        setStatus("");
        setChartMessage("");
        if (tool === "line_bar") {
          el("profileWrap").classList.add("hidden");
          el("modelToolWrap").classList.add("hidden");
          el("ukMap").classList.add("hidden");
          el("mapLegend").classList.add("hidden");
          lineBarTool.setView(state.view);
          lineBarTool.updateAxisControls();
          requestAnimationFrame(() => lineBarTool.resize());
        } else if (tool === "uk_map") {
          el("profileWrap").classList.add("hidden");
          el("modelToolWrap").classList.add("hidden");
          el("chart").classList.add("hidden");
          el("tableWrap").classList.add("hidden");
          el("ukMap").classList.remove("hidden");
          ukMapTool.activate();
        } else {
          el("chart").classList.add("hidden");
          el("tableWrap").classList.add("hidden");
          el("ukMap").classList.add("hidden");
          el("mapLegend").classList.add("hidden");
          if (isModelTool(tool)) {
            el("profileWrap").classList.add("hidden");
            el("modelToolWrap").classList.remove("hidden");
          } else {
            el("modelToolWrap").classList.add("hidden");
            el("profileWrap").classList.remove("hidden");
          }
        }
        if (refresh && state.schema) refreshActiveTool();
      }

      function handleToolClick(tool) {
        if (state.tool === tool) {
          setSidebarVisible(!state.sidebarVisible);
          return;
        }
        setTool(tool);
      }

      function setSidebarVisible(visible) {
        state.sidebarVisible = Boolean(visible);
        document.body.classList.toggle("sidebar-collapsed", !state.sidebarVisible);
        el("appSidebar").removeAttribute("aria-hidden");
        syncSidebarToggleButton();
        requestAnimationFrame(() => {
          if (state.tool === "line_bar") {
            lineBarTool.resize();
          } else if (state.tool === "uk_map") {
            ukMapTool.syncViewport({ mode: "preserve" });
          }
        });
      }

      function syncSidebarToggleButton() {
        const button = el("sidebarToggleBtn");
        const label = state.sidebarVisible ? "Collapse sidebar" : "Expand sidebar";
        button.setAttribute("aria-expanded", String(state.sidebarVisible));
        button.setAttribute("aria-label", label);
        button.title = label;
      }

      function setFilterFooterVisible(visible) {
        state.filterFooterCollapsed = !visible;
        document.body.classList.toggle("filter-footer-collapsed", state.filterFooterCollapsed);
        el("filterFooter").setAttribute("aria-hidden", String(state.filterFooterCollapsed));
        syncFilterFooterToggleButton();
        requestAnimationFrame(() => {
          if (state.tool === "line_bar") {
            lineBarTool.resize();
          } else if (state.tool === "uk_map") {
            ukMapTool.syncViewport({ mode: "preserve" });
          }
        });
      }

      function syncFilterFooterToggleButton() {
        const button = el("filterFooterToggleBtn");
        const visible = !state.filterFooterCollapsed;
        const label = visible ? "Hide filter footer" : "Show filter footer";
        button.setAttribute("aria-expanded", String(visible));
        button.setAttribute("aria-label", label);
        button.title = label;
      }

      const SIDEBAR_ACCORDION_SECTIONS = {
        kpi: { sectionSelector: ".sidebar-kpi-section", buttonId: "kpiCollapseBtn", label: "KPIs" },
        gbm: { sectionSelector: ".gbm-sidebar-panel", buttonId: "gbmModelCollapseBtn", label: "GBMs" },
        glm: { sectionSelector: ".glm-sidebar-panel", buttonId: "glmModelCollapseBtn", label: "GLMs" },
        filter: { sectionSelector: ".sidebar-filter-section", buttonId: "filterCollapseBtn", label: "FILTER" },
      };

      function toggleSidebarSection(section) {
        setOpenSidebarSection(state.openSidebarSection === section ? null : section);
      }

      function setOpenSidebarSection(section) {
        state.openSidebarSection = Object.prototype.hasOwnProperty.call(SIDEBAR_ACCORDION_SECTIONS, section) ? section : null;
        syncSidebarAccordion();
      }

      function syncSidebarAccordion() {
        Object.entries(SIDEBAR_ACCORDION_SECTIONS).forEach(([section, config]) => {
          const open = state.openSidebarSection === section;
          const panel = document.querySelector(config.sectionSelector);
          panel?.classList.toggle("sidebar-section-open", open);
          panel?.classList.toggle("sidebar-section-closed", !open);
          const button = el(config.buttonId);
          if (!button) return;
          const label = `${open ? "Collapse" : "Expand"} ${config.label}`;
          button.setAttribute("aria-expanded", String(open));
          button.setAttribute("aria-label", label);
          button.title = label;
        });
      }

      function isNumericKind(kind) {
        return kind === "numeric" || kind === "integer";
      }

      function fillMetricSelect(select, includeNone = false) {
        if (select.id === "actualNumerator" && !includeNone) {
          fillActualMetricSelect(select);
          return;
        }
        if (select.id === "expectedNumerator") {
          fillExpectedMetricSelect(select, includeNone);
          return;
        }
        select.innerHTML = "";
        if (includeNone) {
          select.append(new Option("None", ""));
        }
        for (const col of numericColumns()) {
          select.append(new Option(col.name, col.name));
        }
      }

      function fillExpectedMetricSelect(select, includeNone = false) {
        select.innerHTML = "";
        if (includeNone) {
          select.append(new Option("None", ""));
        }
        for (const col of expectedColumns()) {
          const option = new Option(metricColumnLabel(col), col.name);
          option.dataset.sourceId = col.source_id || state.source || "dataset";
          option.dataset.metricKind = isModelPredictionColumn(col) ? "prediction" : "metric";
          select.append(option);
        }
      }

      function fillActualMetricSelect(select) {
        select.innerHTML = "";
        appendActualMetricGroup(select, "Dataset features", numericColumnsForSource("dataset"), "dataset", "dataset", "No numeric dataset features");
        const predictionColumns = activePredictionColumns();
        const trainedModels = modelPredictionSourcesExist();
        appendActualMetricGroup(
          select,
          "Model predictions",
          predictionColumns,
          "",
          "prediction",
          trainedModels ? "No predictions for selected model" : "No trained models",
        );
        const shapSource = activeModelSource("gbm_shap_long");
        const shapColumns = shapSource
          ? numericColumnsForSource(shapSource.id).filter(isGbmShapValueColumn)
          : [];
        appendActualMetricGroup(
          select,
          "SHAP values",
          shapColumns,
          shapSource?.id || "",
          "shap",
          trainedModels ? "No SHAP values for selected model" : "No trained models",
        );
      }

      function appendActualMetricGroup(select, label, columns, sourceId, kind, emptyLabel) {
        const group = document.createElement("optgroup");
        group.label = label;
        if (!columns.length) {
          const option = new Option(emptyLabel, "");
          option.disabled = true;
          group.append(option);
        } else {
          for (const column of columns) {
            const option = new Option(metricColumnLabel(column), column.name);
            option.dataset.sourceId = column.source_id || sourceId;
            option.dataset.metricKind = kind;
            group.append(option);
          }
        }
        select.append(group);
      }

      function activeModelSource(kind) {
        return (state.schema?.data_sources || []).find((source) => source.kind === kind && source.active) || null;
      }

      function activePredictionColumns() {
        return activeModelSources(["glm_predictions", "gbm_predictions"]).flatMap((source) => (
          numericColumnsForSource(source.id)
            .filter(isModelPredictionColumn)
            .map((column) => ({
              ...column,
              label: `${source.kind === "glm_predictions" ? "GLM" : "GBM"} · ${metricColumnLabel(column)}`,
              source_id: source.id,
            }))
        ));
      }

      function expectedPredictionColumns() {
        return activeModelSources(["glm_predictions", "gbm_predictions"]).flatMap((source) => (
          numericColumnsForSource(source.id)
            .filter(isModelPredictionColumn)
            .map((column) => ({
              ...column,
              label: metricColumnLabel(column),
              source_id: source.id,
            }))
        ));
      }

      function expectedColumns() {
        const predictionColumns = expectedPredictionColumns();
        const currentSourceColumns = numericColumns()
          .filter((column) => column.source_role !== "gbm_shap_value" && !isModelPredictionColumn(column))
          .map((column) => ({
            ...column,
            source_id: state.source || "dataset",
          }));
        return [...predictionColumns, ...currentSourceColumns];
      }

      function activeModelSources(kinds) {
        const desiredKinds = new Set(kinds);
        return (state.schema?.data_sources || []).filter((source) => desiredKinds.has(source.kind) && source.active);
      }

      function modelPredictionSourcesExist() {
        return (state.schema?.data_sources || []).some((source) => source.kind === "gbm_predictions" || source.kind === "glm_predictions");
      }

      function gbmModelSourcesExist() {
        return (state.schema?.data_sources || []).some((source) => source.kind === "gbm_predictions");
      }

      function numericColumnsForSource(sourceId) {
        return dataSourceColumns(sourceId).filter((c) => isNumericKind(c.kind));
      }

      function isGbmShapValueColumn(column) {
        return column?.source_role === "gbm_shap_value" || String(column?.name || "").startsWith("SHAP__");
      }

      function metricColumnLabel(column) {
        return String(column?.label || column?.name || "");
      }

      function setActualSelection(value, sourceId = "") {
        const select = el("actualNumerator");
        const name = String(value || "");
        if (!name) return false;
        const options = Array.from(select.options);
        const option = options.find((item) => (
          !item.disabled &&
          item.value === name &&
          (!sourceId || item.dataset.sourceId === sourceId)
        )) || options.find((item) => !item.disabled && item.value === name);
        if (!option) return false;
        option.selected = true;
        return true;
      }

      function setExpectedSelection(value, sourceId = "", options = {}) {
        const select = el("expectedNumerator");
        const name = String(value || "");
        if (!name) {
          select.value = "";
          return true;
        }
        const allowAnySource = options.allowAnySource !== false;
        const selectOptions = Array.from(select.options);
        const option = selectOptions.find((item) => (
          !item.disabled &&
          item.value === name &&
          (!sourceId || item.dataset.sourceId === sourceId)
        )) || (allowAnySource ? selectOptions.find((item) => !item.disabled && item.value === name) : null);
        if (!option) return false;
        option.selected = true;
        return true;
      }

      function chooseFirstActualSelection() {
        const option = Array.from(el("actualNumerator").options).find((item) => !item.disabled && item.value);
        if (!option) return "";
        option.selected = true;
        return option.value;
      }

      function fillDenominatorSelect(select) {
        select.innerHTML = "";
        select.append(new Option("Average row value", "__none__"));
        for (const col of numericColumns()) {
          select.append(new Option(col.name, col.name));
        }
      }

      function columnExists(name) {
        return Boolean(name && sourceColumns().some((col) => col.name === name));
      }

      function numericColumnExists(name) {
        return Boolean(name && numericColumns().some((col) => col.name === name));
      }

      function selectedExpectedIsPrediction() {
        return el("expectedNumerator").selectedOptions[0]?.dataset.metricKind === "prediction";
      }

      function expectedSelectionSourceId() {
        return el("expectedNumerator").selectedOptions[0]?.dataset.sourceId || "";
      }

      function predictionColumnNameForModelKind(modelKind) {
        if (modelKind === "glm") return "glm_prediction";
        if (modelKind === "gbm") return "gbm_prediction";
        return "";
      }

      function activePredictionSourceForModelKind(modelKind) {
        if (modelKind === "glm") return activeModelSource("glm_predictions");
        if (modelKind === "gbm") return activeModelSource("gbm_predictions");
        return null;
      }

      function setExpectedPredictionSelectionForModelKind(modelKind) {
        const predictionColumn = predictionColumnNameForModelKind(modelKind);
        const predictionSource = activePredictionSourceForModelKind(modelKind);
        if (!predictionColumn || !predictionSource?.id) return false;
        return setExpectedSelection(predictionColumn, predictionSource.id, { allowAnySource: false });
      }

      function actualSelectionSourceId() {
        return el("actualNumerator").selectedOptions[0]?.dataset.sourceId || "";
      }

      function syncActualSourceFromSelection() {
        const targetSource = actualSelectionSourceId();
        if (!targetSource || targetSource === state.source) return false;
        state.source = targetSource;
        return true;
      }

      function syncExpectedSourceFromSelection({ expectedValue = "", expectedSource = "" } = {}) {
        const targetSource = expectedSource || expectedSelectionSourceId();
        if (selectedExpectedIsPrediction()) return false;
        if (!targetSource || targetSource === state.source) return false;
        const selectedExpected = expectedValue || el("expectedNumerator").value;
        state.source = targetSource;
        syncControlsForSourceChange({
          expectedValue: selectedExpected,
          expectedSource: targetSource,
          expectedIsPrediction: true,
        });
        return true;
      }

      function syncControlsForSourceChange({ actualValue = "", actualSource = "", expectedValue = "", expectedSource = "", expectedIsPrediction = null } = {}) {
        const previousActual = actualValue || el("actualNumerator").value;
        const previousActualSource = actualSource || actualSelectionSourceId();
        const previousExpected = expectedValue || el("expectedNumerator").value;
        const previousExpectedSource = expectedSource || expectedSelectionSourceId();
        const previousExpectedIsPrediction = expectedIsPrediction === null ? selectedExpectedIsPrediction() : Boolean(expectedIsPrediction);
        const previousDenominator = el("denominator").value;
        syncLineBarXFallback();
        fillMetricSelect(el("actualNumerator"));
        if (previousActual && !setActualSelection(previousActual, previousActualSource)) {
          el("actualNumerator").value = numericColumnExists(previousActual) ? previousActual : numericColumns()[0]?.name || "";
        }
        fillMetricSelect(el("expectedNumerator"), true);
        fillDenominatorSelect(el("denominator"));
        if (!setExpectedSelection(previousExpected, previousExpectedSource, { allowAnySource: !previousExpectedIsPrediction })) {
          el("expectedNumerator").value = "";
        }
        el("denominator").value = numericColumnExists(previousDenominator) ? previousDenominator : "__none__";
        syncLineBarXFallback();
        lineBarTool.renderExpectedNumerators();
        lineBarTool.renderFeatures();
        lineBarTool.updateAxisControls();
      }

      async function reloadSchemaAfterModelMutation(preferredSource, options = {}) {
        const previousX = state.x;
        const previousXSource = state.xSource;
        const previousActual = el("actualNumerator").value;
        const previousActualSource = actualSelectionSourceId();
        const previousExpected = el("expectedNumerator").value;
        const previousExpectedSource = expectedSelectionSourceId();
        const previousExpectedIsPrediction = selectedExpectedIsPrediction();
        const previousDenominator = el("denominator").value;
        state.schema = await api("/api/schema");
        const modelKind = String(options?.modelKind || "");
        if (preferredSource) state.source = preferredSource;
        state.x = previousX;
        state.xSource = previousXSource;
        if (modelKind && previousX === predictionColumnNameForModelKind(modelKind)) {
          const predictionSource = activePredictionSourceForModelKind(modelKind);
          if (predictionSource?.id) state.xSource = predictionSource.id;
        }
        syncLineBarXFallback();
        fillMetricSelect(el("actualNumerator"));
        fillMetricSelect(el("expectedNumerator"), true);
        fillDenominatorSelect(el("denominator"));
        if (!setActualSelection(previousActual, previousActualSource)) {
          el("actualNumerator").value = numericColumnExists(previousActual) ? previousActual : numericColumns()[0]?.name || "";
        }
        if (previousExpectedIsPrediction && modelKind) {
          if (!setExpectedPredictionSelectionForModelKind(modelKind)) el("expectedNumerator").value = "";
        } else if (!setExpectedSelection(previousExpected, previousExpectedSource, { allowAnySource: !previousExpectedIsPrediction })) {
          el("expectedNumerator").value = "";
        }
        el("denominator").value = numericColumnExists(previousDenominator) ? previousDenominator : "__none__";
        syncLineBarXFallback();
        lineBarTool.renderExpectedNumerators();
        lineBarTool.renderFeatures();
        lineBarTool.updateAxisControls();
      }

      function requestedDefault(name) {
        return locationParams.get(name) || state.schema.defaults?.[name] || "";
      }

      function hasRequestedDefault(name) {
        return locationParams.has(name) || Boolean(state.schema.defaults?.[name]);
      }

      function normaliseKpiDenominator(value) {
        const denominator = String(value || "").trim();
        if (!denominator || denominator.toLowerCase() === "n" || denominator.toLowerCase() === "average row value" || denominator === "__none__") {
          return "__none__";
        }
        return denominator;
      }

      function kpiKey(kpi) {
        if (!kpi) return "";
        return `${kpi.actual}\u0000${normaliseKpiDenominator(kpi.denominator)}`;
      }

      function denominatorDisplayName(value) {
        const denominator = normaliseKpiDenominator(value);
        if (denominator === "__none__") return "N";
        return denominator;
      }

      function availableKpis() {
        return (state.schema?.kpis || [])
          .map((kpi) => ({
            group: String(kpi.group || "General").trim() || "General",
            name: String(kpi.name || "").trim(),
            actual: String(kpi.actual || "").trim(),
            denominator: normaliseKpiDenominator(kpi.denominator),
            decimals: Number(kpi.decimals),
            format: String(kpi.format || "number").toLowerCase(),
          }))
          .filter((kpi) => (
            kpi.name &&
            numericColumnExists(kpi.actual) &&
            (kpi.denominator === "__none__" || numericColumnExists(kpi.denominator)) &&
            Number.isInteger(kpi.decimals) &&
            kpi.decimals >= 0 &&
            ["number", "currency", "percent"].includes(kpi.format)
          ));
      }

      function selectedKpiForCurrentMetric() {
        const actual = el("actualNumerator").value;
        const denominator = normaliseKpiDenominator(el("denominator").value);
        return availableKpis().find((kpi) => kpi.actual === actual && kpi.denominator === denominator) || null;
      }

      function setActiveKpiState(kpi) {
        state.activeKpiKey = kpi ? kpiKey(kpi) : "";
        state.activeKpiFormat = kpi ? { decimals: kpi.decimals, format: kpi.format } : null;
        el("kpiSelectedMeta").textContent = kpi ? kpi.name : "";
      }

      function syncKpiSelectionFromMetrics() {
        setActiveKpiState(selectedKpiForCurrentMetric());
        syncKpiActiveRows();
      }

      function syncKpiActiveRows() {
        el("kpiSelect").querySelectorAll(".kpi-option").forEach((button) => {
          const active = button.dataset.kpiKey === state.activeKpiKey;
          button.classList.toggle("active", active);
          button.setAttribute("aria-selected", String(active));
        });
      }

      function renderKpis() {
        const list = el("kpiSelect");
        const kpis = availableKpis();
        const availableGroups = new Set(kpis.map((kpi) => kpi.group));
        const selected = selectedKpiForCurrentMetric();
        setActiveKpiState(selected);
        if (!state.kpiGroupsInitialised) {
          availableGroups.forEach((group) => state.collapsedKpiGroups.add(group));
          const openGroup = selected?.group || kpis[0]?.group;
          if (openGroup) state.collapsedKpiGroups.delete(openGroup);
          state.kpiGroupsInitialised = true;
        }
        for (const group of state.collapsedKpiGroups) {
          if (!availableGroups.has(group)) state.collapsedKpiGroups.delete(group);
        }
        list.innerHTML = "";
        let currentGroup = "";
        for (const kpi of kpis) {
          const group = kpi.group || "General";
          if (group !== currentGroup) {
            const collapsed = state.collapsedKpiGroups.has(group);
            const heading = document.createElement("button");
            heading.type = "button";
            heading.className = "saved-filter-theme kpi-theme";
            heading.dataset.kpiGroup = group;
            heading.setAttribute("aria-expanded", String(!collapsed));
            heading.setAttribute("aria-label", `${collapsed ? "Expand" : "Collapse"} ${group} KPIs`);
            heading.title = `${collapsed ? "Expand" : "Collapse"} ${group} KPIs`;
            heading.innerHTML = `<span class="saved-filter-theme-icon" aria-hidden="true"></span><span class="saved-filter-theme-label">${escapeHtml(group)}</span>`;
            heading.addEventListener("click", () => toggleKpiGroup(group));
            list.append(heading);
            currentGroup = group;
          }
          const button = document.createElement("button");
          const key = kpiKey(kpi);
          const active = key === state.activeKpiKey;
          button.type = "button";
          button.className = `feature kpi-option${active ? " active" : ""}`;
          button.dataset.kpiKey = key;
          button.dataset.kpiGroup = group;
          button.hidden = state.collapsedKpiGroups.has(group);
          button.setAttribute("role", "option");
          button.setAttribute("aria-selected", String(active));
          button.innerHTML = `<span class="saved-filter-name">${escapeHtml(kpi.name)}</span><span class="kpi-detail">${escapeHtml(`${kpi.actual} / ${denominatorDisplayName(kpi.denominator)}`)}</span>`;
          button.addEventListener("click", () => selectKpi(kpi));
          list.append(button);
        }
      }

      function toggleKpiGroup(group) {
        const collapsed = !state.collapsedKpiGroups.has(group);
        if (collapsed) {
          state.collapsedKpiGroups.add(group);
        } else {
          state.collapsedKpiGroups.delete(group);
        }
        const list = el("kpiSelect");
        list.querySelectorAll(".kpi-theme").forEach((heading) => {
          if (heading.dataset.kpiGroup !== group) return;
          heading.setAttribute("aria-expanded", String(!collapsed));
          heading.setAttribute("aria-label", `${collapsed ? "Expand" : "Collapse"} ${group} KPIs`);
          heading.title = `${collapsed ? "Expand" : "Collapse"} ${group} KPIs`;
        });
        list.querySelectorAll(".kpi-option").forEach((button) => {
          if (button.dataset.kpiGroup === group) button.hidden = collapsed;
        });
      }

      function selectKpi(kpi) {
        const actual = el("actualNumerator");
        const denominator = el("denominator");
        const nextDenominator = normaliseKpiDenominator(kpi.denominator);
        const changed = actual.value !== kpi.actual || denominator.value !== nextDenominator;
        actual.value = kpi.actual;
        denominator.value = nextDenominator;
        setActiveKpiState(kpi);
        renderKpis();
        if (changed) {
          refreshActiveTool();
        }
      }

      function savedFilterSpecSignature(filters = state.schema?.filters || []) {
        return JSON.stringify((filters || []).map((filter) => ({
          theme: String(filter.theme || "General"),
          name: String(filter.name || ""),
          expression: String(filter.expression || "").trim(),
        })));
      }

      function savedFilterRowKey(row) {
        return [row.theme || "General", row.name || "", row.expression || ""].map(String).join("\u0000");
      }

      function savedFilterButtonKey(button) {
        return savedFilterRowKey({
          theme: button.dataset.filterTheme || "General",
          name: button.dataset.filterName || "",
          expression: button.dataset.expression || "",
        });
      }

      function savedFilterSelectionSnapshot() {
        return new Set(selectedSavedFilterRows().map(savedFilterRowKey));
      }

      function restoreSavedFilterSelection(selectedKeys) {
        if (!selectedKeys?.size) return;
        el("savedFilterSelect").querySelectorAll(".saved-filter-option").forEach((button) => {
          const active = selectedKeys.has(savedFilterButtonKey(button));
          button.setAttribute("aria-selected", String(active));
          button.classList.toggle("active", active);
        });
      }

      function renderSavedFilters() {
        const list = el("savedFilterSelect");
        const filters = state.schema.filters || [];
        const availableThemes = new Set(filters.map((filter) => filter.theme || "General"));
        if (!state.savedFilterThemesInitialised) {
          availableThemes.forEach((theme) => state.collapsedSavedFilterThemes.add(theme));
          state.savedFilterThemesInitialised = true;
        }
        for (const theme of state.collapsedSavedFilterThemes) {
          if (!availableThemes.has(theme)) state.collapsedSavedFilterThemes.delete(theme);
        }
        list.innerHTML = "";
        let currentTheme = "";
        for (const filter of filters) {
          const theme = filter.theme || "General";
          if (theme !== currentTheme) {
            const collapsed = state.collapsedSavedFilterThemes.has(theme);
            const heading = document.createElement("button");
            heading.type = "button";
            heading.className = "saved-filter-theme";
            heading.dataset.filterTheme = theme;
            heading.setAttribute("aria-expanded", String(!collapsed));
            heading.setAttribute("aria-label", `${collapsed ? "Expand" : "Collapse"} ${theme} saved filters`);
            heading.title = `${collapsed ? "Expand" : "Collapse"} ${theme} saved filters`;
            heading.innerHTML = `<span class="saved-filter-theme-icon" aria-hidden="true"></span><span class="saved-filter-theme-label">${escapeHtml(theme)}</span>`;
            heading.addEventListener("click", () => toggleSavedFilterTheme(theme));
            list.append(heading);
            currentTheme = theme;
          }
          const button = document.createElement("button");
          button.type = "button";
          button.className = "feature saved-filter-option";
          button.dataset.expression = filter.expression || "";
          button.dataset.filterTheme = theme;
          button.dataset.filterName = filter.name || "";
          button.hidden = state.collapsedSavedFilterThemes.has(theme);
          button.setAttribute("role", "option");
          button.setAttribute("aria-selected", "false");
          button.innerHTML = `<span class="saved-filter-name">${escapeHtml(filter.name)}</span><span class="saved-filter-expression">${escapeHtml(filter.expression)}</span>`;
          button.addEventListener("click", () => {
            const selected = button.getAttribute("aria-selected") === "true";
            if (state.filterSelectionMode === "single") {
              list.querySelectorAll(".saved-filter-option").forEach((option) => {
                const isClickedOption = option === button;
                option.setAttribute("aria-selected", String(isClickedOption));
                option.classList.toggle("active", isClickedOption);
              });
            } else {
              button.setAttribute("aria-selected", String(!selected));
              button.classList.toggle("active", !selected);
            }
            applySavedFilters();
          });
          list.append(button);
        }
      }

      function setFilterSelectionMode(mode, options = {}) {
        const nextMode = mode === "single" || mode === "grouped" ? mode : "multi";
        state.filterSelectionMode = nextMode;
        document.body.classList.toggle("saved-filter-single-mode", nextMode === "single");
        document.body.classList.toggle("saved-filter-grouped-mode", nextMode === "grouped");
        const group = document.querySelector('.segmented[data-control="filterSelectionMode"]');
        group?.querySelectorAll("button").forEach((button) => {
          button.classList.toggle("active", button.dataset.value === nextMode);
        });
        if (nextMode === "single") {
          const filterOptions = Array.from(el("savedFilterSelect").querySelectorAll(".saved-filter-option"));
          const selected = filterOptions.filter((button) => button.getAttribute("aria-selected") === "true");
          if (selected.length > 1) {
            const keep = selected[0];
            filterOptions.forEach((button) => {
              const active = button === keep;
              button.setAttribute("aria-selected", String(active));
              button.classList.toggle("active", active);
            });
          }
        }
        if (options.apply !== false) {
          applySavedFilters();
        }
      }

      function toggleSavedFilterTheme(theme) {
        const collapsed = !state.collapsedSavedFilterThemes.has(theme);
        if (collapsed) {
          state.collapsedSavedFilterThemes.add(theme);
        } else {
          state.collapsedSavedFilterThemes.delete(theme);
        }
        const list = el("savedFilterSelect");
        list.querySelectorAll(".saved-filter-theme").forEach((heading) => {
          if (heading.dataset.filterTheme !== theme) return;
          heading.setAttribute("aria-expanded", String(!collapsed));
          heading.setAttribute("aria-label", `${collapsed ? "Expand" : "Collapse"} ${theme} saved filters`);
          heading.title = `${collapsed ? "Expand" : "Collapse"} ${theme} saved filters`;
        });
        list.querySelectorAll(".saved-filter-option").forEach((button) => {
          if (button.dataset.filterTheme === theme) button.hidden = collapsed;
        });
      }

      function selectedSavedFilterRows() {
        return Array.from(el("savedFilterSelect").querySelectorAll('.saved-filter-option[aria-selected="true"]'))
          .map((button) => ({
            theme: button.dataset.filterTheme || "General",
            name: button.dataset.filterName || "",
            expression: button.dataset.expression.trim(),
          }))
          .filter((row) => row.expression);
      }

      function selectedSavedFilterExpressions() {
        return selectedSavedFilterRows().map((row) => row.expression);
      }

      function wrapFilterExpression(expression) {
        return `(${expression})`;
      }

      function combinedFlatSavedFilterExpression(rows) {
        const expressions = rows.map((row) => row.expression);
        if (!expressions.length) return "";
        const operator = state.filterOperator === "or" || state.filterOperator === "nor" ? "OR" : "AND";
        const groupedExpressions = expressions.length > 1 ? expressions.map(wrapFilterExpression) : expressions;
        const combined = groupedExpressions.join(` ${operator} `);
        return state.filterOperator === "nand" || state.filterOperator === "nor" ? `NOT (${combined})` : combined;
      }

      function combinedGroupedSavedFilterExpression(rows) {
        if (rows.length === 1) return rows[0].expression;
        const groups = [];
        const byTheme = new Map();
        rows.forEach((row) => {
          if (!byTheme.has(row.theme)) {
            const group = [];
            byTheme.set(row.theme, group);
            groups.push(group);
          }
          byTheme.get(row.theme).push(row.expression);
        });
        return groups
          .map((expressions) => {
            const groupedExpressions = expressions.map(wrapFilterExpression).join(" OR ");
            return expressions.length > 1 ? `(${groupedExpressions})` : groupedExpressions;
          })
          .join(" AND ");
      }

      function combinedSavedFilterExpression() {
        const rows = selectedSavedFilterRows();
        if (!rows.length) return "";
        return state.filterSelectionMode === "grouped"
          ? combinedGroupedSavedFilterExpression(rows)
          : combinedFlatSavedFilterExpression(rows);
      }

      function applySavedFilters() {
        el("filterInput").value = combinedSavedFilterExpression();
        applyFilter();
      }

      function chooseDefaults() {
        const requestedSource = requestedDefault("source");
        const availableSources = state.schema.data_sources || [];
        state.source = preferredStartupSource(availableSources, requestedSource);
        const requestedX = requestedDefault("x");
        const requestedXSource = requestedDefault("xSource") || state.source;
        if (lineBarColumnExists(requestedX, requestedXSource) || lineBarColumnExists(requestedX)) {
          state.x = requestedX;
          state.xSource = lineBarFeatureSourceForName(requestedX, requestedXSource);
        } else {
          syncLineBarXFallback();
        }
        fillMetricSelect(el("actualNumerator"));
        fillMetricSelect(el("expectedNumerator"), true);
        fillDenominatorSelect(el("denominator"));
        const requestedActual = requestedDefault("actual");
        const requestedExpected = requestedDefault("expected");
        const requestedDenominator = requestedDefault("denominator");
        if (!setActualSelection(requestedActual, state.source)) {
          el("actualNumerator").value = numericColumnExists(requestedActual) ? requestedActual : numericColumns()[0]?.name || "";
        }
        if (!el("actualNumerator").value) chooseFirstActualSelection();
        if (!setExpectedSelection(requestedExpected)) el("expectedNumerator").value = "";
        el("denominator").value = numericColumnExists(requestedDenominator) ? requestedDenominator : "__none__";
        applyInitialKpiDefault();
      }

      function applyInitialKpiDefault() {
        if (hasRequestedDefault("actual") || hasRequestedDefault("denominator")) return;
        const firstKpi = availableKpis()[0];
        if (!firstKpi) return;
        el("actualNumerator").value = firstKpi.actual;
        el("denominator").value = firstKpi.denominator;
      }

      function applyFilter() {
        const nextFilter = el("filterInput").value.trim();
        if (nextFilter === state.activeFilter) {
          syncActiveFilterLabels();
          refreshActiveTool();
          return;
        }
        state.activeFilter = nextFilter;
        columnProfileTool.resetSummaryMode();
        clearProfileDetailCache();
        syncActiveFilterLabels();
        refreshActiveTool();
      }

      function clearFilter() {
        el("filterInput").value = "";
        Array.from(el("savedFilterSelect").querySelectorAll(".saved-filter-option")).forEach((button) => {
          button.setAttribute("aria-selected", "false");
          button.classList.remove("active");
        });
        if (state.activeFilter === "") {
          syncActiveFilterLabels();
          refreshActiveTool();
          return;
        }
        state.activeFilter = "";
        columnProfileTool.resetSummaryMode();
        clearProfileDetailCache();
        syncActiveFilterLabels();
        refreshActiveTool();
      }

      function confirmStopApp() {
        return new Promise((resolve) => {
          const overlay = document.createElement("div");
          overlay.className = "stop-confirm-overlay";
          overlay.innerHTML = `
            <div class="stop-confirm-dialog" role="dialog" aria-modal="true" aria-labelledby="stopConfirmTitle">
              <div class="stop-confirm-content">
                <img class="stop-confirm-icon" src="/favicon.ico" alt="">
                <p id="stopConfirmTitle">Stop the local lucidum server?</p>
              </div>
              <div class="stop-confirm-actions">
                <button class="ghost stop-confirm-cancel" type="button">Cancel</button>
                <button class="ghost stop-confirm-ok" type="button">OK</button>
              </div>
            </div>
          `;
          const cancelButton = overlay.querySelector(".stop-confirm-cancel");
          const okButton = overlay.querySelector(".stop-confirm-ok");
          let closed = false;
          const close = (confirmed) => {
            if (closed) return;
            closed = true;
            window.removeEventListener("keydown", handleKeydown);
            overlay.remove();
            resolve(confirmed);
          };
          function handleKeydown(event) {
            if (event.key === "Escape") close(false);
          }
          cancelButton.addEventListener("click", () => close(false));
          okButton.addEventListener("click", () => close(true));
          window.addEventListener("keydown", handleKeydown);
          document.body.append(overlay);
          cancelButton.focus();
        });
      }

      async function stopApp() {
        if (!(await confirmStopApp())) return;
        const button = el("stopAppBtn");
        button.disabled = true;
        button.textContent = "Stopping...";
        setStatus("Stopping app...");
        try {
          await api("/api/shutdown", { method: "POST" });
          showStoppedOverlay();
        } catch (error) {
          button.disabled = false;
          button.textContent = "Stop app";
          setStatus(error.message, true);
        }
      }

      function showStoppedOverlay() {
        if (stoppedOverlayShown) return;
        stoppedOverlayShown = true;
        stopServerHeartbeat();
        document.body.classList.add("app-stopped");
        const shutdownIcon = faviconDataUrl
          ? `<img class="shutdown-icon" src="${faviconDataUrl}" alt="">`
          : '<span class="shutdown-icon shutdown-icon-fallback" aria-hidden="true"></span>';
        const overlay = document.createElement("div");
        overlay.className = "shutdown-overlay";
        overlay.innerHTML = `
          <div class="shutdown-message" role="status" aria-live="polite">
            ${shutdownIcon}
            <div>
              <h1>lucidum has stopped</h1>
              <p>The local server is no longer running. You can close this browser tab.</p>
            </div>
          </div>
        `;
        document.body.append(overlay);
      }



      function renderMetricTitle(target, label, value, formatter = formatLineValue) {
        const formatted = formatter(value);
        target.textContent = label;
        if (!formatted) return;
        target.append(" ");
        const valueSpan = document.createElement("span");
        valueSpan.className = "metric-value";
        valueSpan.textContent = formatted;
        target.append(valueSpan);
      }

      function setupSidebarResize() {
        const shell = document.querySelector(".shell");
        const resizer = el("sidebarResizer");
        const savedWidth = Number(localStorage.getItem("py_lucidum_sidebar_width"));
        if (Number.isFinite(savedWidth) && savedWidth > 0) {
          setSidebarWidth(savedWidth);
        }

        let dragging = false;
        resizer.addEventListener("pointerdown", (event) => {
          event.preventDefault();
          dragging = true;
          resizer.classList.add("dragging");
          document.body.classList.add("resizing-sidebar");
          resizer.setPointerCapture(event.pointerId);
          window.getSelection()?.removeAllRanges();
        });
        resizer.addEventListener("pointermove", (event) => {
          if (!dragging) return;
          event.preventDefault();
          const bounds = shell.getBoundingClientRect();
          setSidebarWidth(event.clientX - bounds.left);
        });
        function finishDrag(event) {
          if (!dragging) return;
          dragging = false;
          resizer.classList.remove("dragging");
          document.body.classList.remove("resizing-sidebar");
          window.getSelection()?.removeAllRanges();
          const width = parseFloat(getComputedStyle(document.documentElement).getPropertyValue("--sidebar-width"));
          if (Number.isFinite(width)) {
            localStorage.setItem("py_lucidum_sidebar_width", String(Math.round(width)));
          }
          if (event.pointerId !== undefined) {
            try {
              resizer.releasePointerCapture(event.pointerId);
            } catch (_) {
            }
          }
          if (state.tool === "uk_map") {
            ukMapTool.syncViewport({ mode: "preserve" });
          } else {
            lineBarTool.resize();
          }
        }
        resizer.addEventListener("pointerup", finishDrag);
        resizer.addEventListener("pointercancel", finishDrag);
      }

      function setSidebarWidth(rawWidth) {
        const viewportLimit = Math.max(260, window.innerWidth - 520);
        const width = Math.min(Math.max(rawWidth, 220), Math.min(560, viewportLimit));
        document.documentElement.style.setProperty("--sidebar-width", `${Math.round(width)}px`);
        requestAnimationFrame(() => {
          if (state.tool === "uk_map") {
            ukMapTool.syncViewport({ mode: "preserve" });
          } else {
            lineBarTool.resize();
          }
        });
      }

      function setupChartControlsResize() {
        const visualArea = document.querySelector(".visual-area");
        const resizer = el("chartControlsResizer");
        const savedWidth = Number(localStorage.getItem("py_lucidum_chart_controls_width"));
        if (Number.isFinite(savedWidth) && savedWidth > 0) {
          setChartControlsWidth(savedWidth);
        }

        let dragging = false;
        resizer.addEventListener("pointerdown", (event) => {
          event.preventDefault();
          dragging = true;
          resizer.classList.add("dragging");
          document.body.classList.add("resizing-chart-controls");
          resizer.setPointerCapture(event.pointerId);
          window.getSelection()?.removeAllRanges();
        });
        resizer.addEventListener("pointermove", (event) => {
          if (!dragging) return;
          event.preventDefault();
          const bounds = visualArea.getBoundingClientRect();
          setChartControlsWidth(event.clientX - bounds.left);
        });
        function finishDrag(event) {
          if (!dragging) return;
          dragging = false;
          resizer.classList.remove("dragging");
          document.body.classList.remove("resizing-chart-controls");
          window.getSelection()?.removeAllRanges();
          const width = parseFloat(getComputedStyle(document.documentElement).getPropertyValue("--chart-controls-width"));
          if (Number.isFinite(width)) {
            localStorage.setItem("py_lucidum_chart_controls_width", String(Math.round(width)));
          }
          if (event.pointerId !== undefined) {
            try {
              resizer.releasePointerCapture(event.pointerId);
            } catch (_) {
            }
          }
          lineBarTool.resize();
        }
        resizer.addEventListener("pointerup", finishDrag);
        resizer.addEventListener("pointercancel", finishDrag);
      }

      function setChartControlsWidth(rawWidth) {
        const visualArea = document.querySelector(".visual-area");
        const availableWidth = visualArea?.getBoundingClientRect().width || window.innerWidth;
        const minWidth = 280;
        const maxWidth = Math.max(minWidth, Math.min(560, availableWidth - 420));
        const width = Math.min(Math.max(rawWidth, minWidth), maxWidth);
        document.documentElement.style.setProperty("--chart-controls-width", `${Math.round(width)}px`);
        requestAnimationFrame(() => lineBarTool.resize());
      }

      function setupChartControlHeightsResize() {
        const controls = document.querySelector(".chart-side-controls");
        const firstPanel = controls?.querySelector(".chart-side-section");
        const resizer = el("chartControlHeightResizer");
        const savedHeight = Number(localStorage.getItem("py_lucidum_chart_feature_controls_height"));
        if (Number.isFinite(savedHeight) && savedHeight > 0) {
          setChartFeatureControlsHeight(savedHeight);
        }

        let dragging = false;
        let startY = 0;
        let startHeight = 0;
        resizer.addEventListener("pointerdown", (event) => {
          event.preventDefault();
          dragging = true;
          startY = event.clientY;
          startHeight = firstPanel?.getBoundingClientRect().height || 0;
          resizer.classList.add("dragging");
          document.body.classList.add("resizing-chart-control-heights");
          resizer.setPointerCapture(event.pointerId);
          window.getSelection()?.removeAllRanges();
        });
        resizer.addEventListener("pointermove", (event) => {
          if (!dragging) return;
          event.preventDefault();
          setChartFeatureControlsHeight(startHeight + event.clientY - startY);
        });
        function finishDrag(event) {
          if (!dragging) return;
          dragging = false;
          resizer.classList.remove("dragging");
          document.body.classList.remove("resizing-chart-control-heights");
          window.getSelection()?.removeAllRanges();
          const height = parseFloat(getComputedStyle(document.documentElement).getPropertyValue("--chart-feature-controls-height"));
          if (Number.isFinite(height)) {
            localStorage.setItem("py_lucidum_chart_feature_controls_height", String(Math.round(height)));
          }
          if (event.pointerId !== undefined) {
            try {
              resizer.releasePointerCapture(event.pointerId);
            } catch (_) {
            }
          }
        }
        resizer.addEventListener("pointerup", finishDrag);
        resizer.addEventListener("pointercancel", finishDrag);
      }

      function setChartFeatureControlsHeight(rawHeight) {
        const controls = document.querySelector(".chart-side-controls");
        const availableHeight = controls?.getBoundingClientRect().height || window.innerHeight;
        const splitterSpace = 22;
        const minPanelHeight = 96;
        const maxHeight = Math.max(minPanelHeight, availableHeight - splitterSpace - minPanelHeight);
        const height = Math.min(Math.max(rawHeight, minPanelHeight), maxHeight);
        document.documentElement.style.setProperty("--chart-feature-controls-height", `${Math.round(height)}px`);
      }

      function bindControls() {
        setupSidebarResize();
        setupChartControlsResize();
        setupChartControlHeightsResize();
        ukMapTool.bindControls();
        syncSidebarToggleButton();
        syncSidebarAccordion();
        syncFilterFooterToggleButton();
        syncActionTimingMonitor();
        setFilterSelectionMode(state.filterSelectionMode, { apply: false });
        lineBarTool.bindControls();
        document.querySelectorAll('.segmented[data-control="filterOperator"], .segmented[data-control="filterSelectionMode"]').forEach((group) => {
          group.addEventListener("click", (event) => {
            if (event.target.tagName !== "BUTTON") return;
            group.querySelectorAll("button").forEach((button) => button.classList.remove("active"));
            event.target.classList.add("active");
            state[group.dataset.control] = event.target.dataset.value;
            if (group.dataset.control === "filterOperator") {
              applySavedFilters();
              return;
            }
            if (group.dataset.control === "filterSelectionMode") {
              setFilterSelectionMode(event.target.dataset.value);
              return;
            }
          });
        });
        el("actualNumerator").addEventListener("change", () => {
          const actualValue = el("actualNumerator").value;
          const actualSource = actualSelectionSourceId();
          if (syncActualSourceFromSelection()) {
            syncControlsForSourceChange({ actualValue, actualSource });
          }
          syncKpiSelectionFromMetrics();
          refreshActiveTool();
        });
        el("denominator").addEventListener("change", () => {
          syncKpiSelectionFromMetrics();
          refreshActiveTool();
        });
        el("filterApplyBtn").addEventListener("click", applyFilter);
        el("filterClearBtn").addEventListener("click", clearFilter);
        el("filterInput").addEventListener("keydown", (event) => {
          if (event.key === "Enter") {
            event.preventDefault();
            applyFilter();
          }
        });
        el("profileTool").addEventListener("click", () => handleToolClick("column_profile"));
        el("lineBarTool").addEventListener("click", () => handleToolClick("line_bar"));
        el("ukMapTool").addEventListener("click", () => handleToolClick("uk_map"));
        el("glmTool").addEventListener("click", () => handleToolClick("glm"));
        el("gbmTool").addEventListener("click", () => handleToolClick("gbm"));
        el("sidebarToggleBtn").addEventListener("click", () => setSidebarVisible(!state.sidebarVisible));
        el("filterFooterToggleBtn").addEventListener("click", () => setFilterFooterVisible(state.filterFooterCollapsed));
        el("kpiCollapseBtn").addEventListener("click", () => toggleSidebarSection("kpi"));
        el("glmModelCollapseBtn").addEventListener("click", () => toggleSidebarSection("glm"));
        el("gbmModelCollapseBtn").addEventListener("click", () => toggleSidebarSection("gbm"));
        el("filterCollapseBtn").addEventListener("click", () => toggleSidebarSection("filter"));
        el("filterSidebarClearBtn").addEventListener("click", clearFilter);
        el("stopAppBtn").addEventListener("click", stopApp);
        el("themeBtn").addEventListener("click", () => {
          document.body.classList.toggle("dark");
          syncThemeButton();
          if (state.tool === "line_bar") lineBarTool.refreshTheme();
          if (state.tool === "uk_map") ukMapTool.refreshTheme();
          if (state.tool === "glm") measureToolRender("glm", () => glmTool.refreshTheme());
          if (state.tool === "gbm") measureToolRender("gbm", () => gbmTool.refreshTheme());
        });
        el("reloadBtn").addEventListener("click", async () => {
          setStatus("");
          setGroupMeta(state.tool, "Reloading...");
          const previousFilterSignature = savedFilterSpecSignature();
          const previousSavedFilterSelection = savedFilterSelectionSnapshot();
          const previousCollapsedSavedFilterThemes = new Set(state.collapsedSavedFilterThemes);
          const previousSavedFilterThemesInitialised = state.savedFilterThemesInitialised;
          const previousSidebarVisible = state.sidebarVisible;
          if (state.tool === "uk_map") ukMapTool.captureView("reload");
          state.schema = await api("/api/reload", { method: "POST" });
          const filtersUnchanged = previousFilterSignature === savedFilterSpecSignature(state.schema.filters || []);
          state.bandFeature = null;
          state.bandSuggestionPendingKey = null;
          state.bandSuggestionRequestSeq = (state.bandSuggestionRequestSeq || 0) + 1;
          clearToolCaches();
          renderDatasetMeta(schemaFileMeta(), datasetGbmCount, datasetGlmCount);
          refreshDatasetGlmCount();
          refreshDatasetGbmCount();
          setFilterRowMeta(state.schema.row_count);
          if (filtersUnchanged) {
            state.collapsedSavedFilterThemes = previousCollapsedSavedFilterThemes;
            state.savedFilterThemesInitialised = previousSavedFilterThemesInitialised;
          } else {
            state.collapsedSavedFilterThemes = new Set();
            state.savedFilterThemesInitialised = false;
          }
          state.kpiGroupsInitialised = false;
          renderSavedFilters();
          if (filtersUnchanged) restoreSavedFilterSelection(previousSavedFilterSelection);
          renderKpis();
          renderToolSelector();
          if (!toolEnabled(state.tool)) {
            state.tool = chooseDefaultTool();
          }
          lineBarTool.renderExpectedNumerators();
          lineBarTool.renderFeatures();
          lineBarTool.updateAxisControls();
          setTool(state.tool, false);
          setSidebarVisible(previousSidebarVisible);
          refreshActiveTool({ force: true });
        });
        window.addEventListener("resize", () => {
          if (state.tool === "line_bar") {
            const controls = document.querySelector(".chart-side-controls");
            if (controls) setChartControlsWidth(controls.getBoundingClientRect().width);
            const firstPanel = controls?.querySelector(".chart-side-section");
            if (firstPanel) setChartFeatureControlsHeight(firstPanel.getBoundingClientRect().height);
            lineBarTool.resize();
          } else if (state.tool === "uk_map") {
            ukMapTool.syncViewport({ mode: "preserve" });
          }
        });
      }



      function getCss(name) {
        return getComputedStyle(document.body).getPropertyValue(name).trim();
      }

      function syncThemeButton() {
        const label = document.body.classList.contains("dark") ? "Switch to light mode" : "Switch to dark mode";
        el("themeBtn").setAttribute("aria-label", label);
        el("themeBtn").title = label;
      }

      export async function boot() {
        bindControls();
        syncThemeButton();
        syncMonitorLink();
        cacheShutdownIcon();
        try {
          setStartupProgress("Requesting schema");
          startStartupTelemetryPolling("Requesting schema");
          state.schema = await api("/api/schema");
          stopStartupTelemetryPolling();
          setStartupProgress("Schema received");
          const path = state.schema.path.split(/[\\/]/).pop();
          const fileMeta = schemaFileMeta();
          document.title = path ? `lucidum · ${path}` : "lucidum";
          setFilterRowMeta(state.schema.row_count);
          setStartupProgress("Rendering controls");
          chooseDefaults();
          renderKpis();
          renderToolSelector();
          renderDatasetMeta(fileMeta);
          refreshDatasetGlmCount();
          refreshDatasetGbmCount();
          state.tool = chooseDefaultTool();
          renderSavedFilters();
          lineBarTool.renderExpectedNumerators();
          lineBarTool.renderFeatures();
          lineBarTool.updateAxisControls();
          setTool(state.tool, false);
          setStartupProgress("Loading initial profile");
          await refreshActiveTool({ force: true });
          setStartupProgress("Ready", "ready");
          startServerHeartbeat();
        } catch (error) {
          stopStartupTelemetryPolling();
          setStartupProgress("Startup failed", "error");
          el("datasetMeta").textContent = "Dataset failed to load";
          setStatus(error.message, true);
        }
      }
