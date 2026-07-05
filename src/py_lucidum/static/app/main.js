      import { createColumnProfileTool } from "./column-profile-tool.js";
      import { createLineBarTool } from "./line-bar-tool.js";
      import { createHistogramTool } from "./histogram-tool.js";
      import { createUkMapTool } from "./uk-map-tool.js";
      import { createGlmTool } from "./glm-tool.js";
      import { createGbmTool } from "./gbm-tool.js";
      import { createSpecificationsTool } from "./specifications-tool.js";
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
        const expectedKeys = ["token", "tool", "source", "x", "xSource", "actual", "expected", "expected2", "denominator", "line_bar_favourite", "postcode_area", "postcode_sector", "postcode_unit", "latitude", "longitude"];
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
        dataset_viewer: "Dataset render",
        column_profile: "Profile render",
        line_bar: "Chart render",
        histogram: "Histogram render",
        uk_map: "Map render",
        glm: "GLM render",
        gbm: "GBM render",
        specs: "Specs render",
      };
      const TOOL_BUTTON_IDS = {
        line_bar: "lineBarTool",
        dataset_viewer: "datasetViewerTool",
        column_profile: "profileTool",
        histogram: "histogramTool",
        uk_map: "ukMapTool",
        glm: "glmTool",
        gbm: "gbmTool",
        specs: "specsTool",
      };
      const TOOL_IDS = Object.keys(TOOL_BUTTON_IDS);
      const CHART_FEATURE_CONTROLS_HEIGHT_COLLAPSED = "collapsed";
      const MOBILE_LAYOUT_MAX_WIDTH = 640;
      const TOOL_BUTTON_TOOLTIP_DELAY_MS = 500;
      const LINE_BAR_RATIO_COLUMN = "gbm_to_glm_ratio";
      const GLM_PREDICTION_COLUMNS = ["glm_prediction", "glm_prediction_rate", "glm_tabulated_prediction"];
      const GBM_PREDICTION_COLUMNS = ["gbm_prediction", "gbm_prediction_rate", "gbm_tabulated_prediction"];
      const MODEL_RATIO_SOURCE_RE = /^model_ratio:gbm_to_glm_ratio:[A-Za-z0-9_.-]+:[A-Za-z0-9_.-]+$/;
      const GLM_PREDICTION_SOURCE_RE = /^glm:[A-Za-z0-9_.-]+:predictions$/;
      const GBM_PREDICTION_SOURCE_RE = /^gbm:[A-Za-z0-9_.-]+:predictions$/;
      const FAVOURITE_SCOPES = new Set(["metrics", "metrics_filter", "line_bar_view", "map_view"]);
      const DEFAULT_FAVOURITE_SCOPE = "line_bar_view";
      const FAVOURITE_SCOPE_LABELS = {
        metrics: "Metrics",
        metrics_filter: "Metrics + filter",
        line_bar_view: "Line/Bar view",
        map_view: "Map view",
      };
      const LINE_BAR_FAVOURITE_SCOPE_OPTIONS = [
        ["line_bar_view", FAVOURITE_SCOPE_LABELS.line_bar_view],
        ["metrics_filter", FAVOURITE_SCOPE_LABELS.metrics_filter],
        ["metrics", FAVOURITE_SCOPE_LABELS.metrics],
      ];
      const MAP_FAVOURITE_SCOPE_OPTIONS = [
        ["map_view", FAVOURITE_SCOPE_LABELS.map_view],
        ["metrics_filter", FAVOURITE_SCOPE_LABELS.metrics_filter],
        ["metrics", FAVOURITE_SCOPE_LABELS.metrics],
      ];

      function initialSidebarVisible() {
        return !document.body.classList.contains("sidebar-collapsed");
      }

      const state = {
        schema: null,
        x: null,
        xSource: "",
        sort: "alpha",
        lowGroup: "0",
        labels: "none",
        sidebarVisible: initialSidebarVisible(),
        lineBarSideControlsCollapsed: true,
        lineBarToolbarCollapsed: true,
        bandWidth: "0",
        quantileMode: "off",
        previousBandWidthsByFeature: {},
        dateBucket: "none",
        dateBucketFeature: null,
        dateBucketManualKey: null,
        transform: "none",
        sigma: "0",
        partialDependence: "none",
        histogramDistribution: "incremental",
        histogramYAxis: "sum",
        histogramLogScale: "none",
        histogramSampleMode: "100k",
        source: locationParams.get("source") || "dataset",
        tool: "",
        view: "chart",
        mapLevel: "area",
        baseMap: "blank",
        mapPalette: "divergent",
        mapLineWeight: 1,
        mapDotSize: 1,
        mapOpacity: 1,
        mapHotspots: 0,
        mapLabelSize: 0,
        mapSmoothingLevel: 0,
        featureSort: "alpha",
        expectedSort: "alpha",
        expectedSelections: [],
        openSidebarSection: "favourites",
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
        filterSelectionMode: "grouped",
        collapsedSavedFilterThemes: new Set(),
        savedFilterThemesInitialised: false,
        activeFilter: "",
        activeLineBarFavouriteId: "",
        filterRowCountMeta: null,
        datasetViewerSearch: "",
        datasetViewerTranspose: false,
        datasetViewerAlphabeticalColumns: false,
        datasetViewerPinnedColumns: [],
        datasetViewerColumnCount: null,
        profileSort: { key: "", direction: "asc" },
        profileColumnSearch: "",
        lineBarTableSearch: "",
        profileDetailSort: { key: "count", direction: "desc" },
        profileSummaryMode: "auto",
        selectedProfileColumn: "",
        lastDatasetViewerData: null,
        lastProfileData: null,
        lastProfileDetailData: null,
        lastData: null,
        lastHistogramData: null,
        lastMapData: null,
        toolCache: {
          dataset_viewer: { requestKey: null, data: null, presentation: null },
          column_profile: freshProfileCache(),
          line_bar: { requestKey: null, data: null, presentation: null },
          histogram: { requestKey: null, data: null, presentation: null },
          uk_map: { requestKey: null, data: null, presentation: null },
          glm: { requestKey: null, data: null, presentation: null },
          gbm: { requestKey: null, data: null, presentation: null },
          specs: { requestKey: null, data: null, presentation: null },
        },
        actionTimings: freshActionTimings(),
        mapGeoJsonCache: {},
        mapPolygonLayerCache: {},
        mapPolygonRenderContext: null,
        mapStartupFitDone: false,
        renderedMapLevel: null,
        mapView: null,
        mapFavouriteRestoreInProgress: false,
        mapViewportSyncFrame: null,
        restoringMapView: false,
        pendingMapZoom: null,
        mapControlPosition: null,
        mapControlMoved: false,
        mapControlCollapsed: false,
        mapControlCollapsedPosition: null,
        tablePage: 1,
        bandFeature: null,
        bandSuggestionPendingKey: null,
        bandSuggestionRequestSeq: 0,
        dateBucketSuggestionPendingKey: null,
        dateBucketSuggestionRequestSeq: 0,
        profileRequestSeq: 0,
        profileDetailRequestSeq: 0,
        chartRequestSeq: 0,
        histogramRequestSeq: 0,
        mapRequestSeq: 0,
        datasetViewerRequestSeq: 0,
        filterRowCountRequestSeq: 0,
        metricSummaryRequestSeq: 0,
        metricSummaryRequestKey: null,
        metricSummaryData: null,
        glmRequestSeq: 0,
        gbmRequestSeq: 0,
      };

      const BAND_STEPS = makeBandSteps();
      let serverHeartbeatTimer = null;
      let startupTelemetryTimer = null;
      let startupProgressStartedAt = 0;
      let clipboardToastTimer = null;
      let toolButtonTooltip = null;
      let toolButtonTooltipTarget = null;
      let toolButtonTooltipPendingTarget = null;
      let toolButtonTooltipTimer = null;
      let toolButtonTooltipKeydownBound = false;
      let stoppedOverlayShown = false;
      let faviconDataUrl = "";
      let datasetMetaBase = "";
      let datasetGlmCount = null;
      let datasetGbmCount = null;
      let datasetMetaCompactFrame = null;
      const el = (id) => document.getElementById(id);
      const api = createApiClient({ token });
      const {
        formatNumber,
        formatChartLabel,
        formatLineLabel,
        formatLineValue,
        formatLineValueForFormat,
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
      let datasetViewerTool = null;
      let datasetViewerToolPromise = null;
      let chartFeatureControlsExpandedHeight = null;
      let chartExpectedStartupCollapseApplied = false;
      let mobileLayoutActive = null;
      let lineBarFavourites = [];
      let lineBarFavouritesLoaded = false;
      let favouritePopoverMode = "manage";
      let favouriteStartupApplied = false;
      let selectedFavouriteManageId = "";
      let favouriteLoadError = "";
      let favouriteOrderSaveSequence = 0;
      let favouriteOrderSaveInFlight = false;
      let pendingFavouriteOrderSave = null;

      async function ensureDatasetViewerTool() {
        if (!toolEnabled("dataset_viewer")) return null;
        if (datasetViewerTool) return datasetViewerTool;
        if (!datasetViewerToolPromise) {
          datasetViewerToolPromise = import("./dataset-viewer-tool.js")
            .then(({ createDatasetViewerTool }) => {
              datasetViewerTool = createDatasetViewerTool({
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
                syncDatasetViewerMeta,
                syncClientTimingFromData,
                syncDuckDbTimingFromData,
                toolCache,
              });
              return datasetViewerTool;
            })
            .catch((error) => {
              datasetViewerToolPromise = null;
              throw error;
            });
        }
        return datasetViewerToolPromise;
      }

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
        copyTextToClipboard,
        formatNumber,
        formatChartLabel,
        formatLineLabel,
        formatLineValue,
        formatLineValueForFormat,
        formatXLabel,
        formatRowMeta,
        measureToolRender,
        startToolTiming,
        setToolTimingFailed,
        syncDuckDbTimingFromData,
        syncClientTimingFromData,
        setStatus,
        setChartMessage,
        setGroupMeta,
        applyToolPresentation,
        saveToolPresentation,
        showClipboardToast,
        stableRequestKey,
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
        clearActiveFavouriteSelection: () => clearActiveFavouriteSelectionForScope("line_bar_view"),
      });
      const histogramTool = createHistogramTool({
        api,
        el,
        state,
        echartsImpl: echarts,
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
        setGroupMeta,
        applyToolPresentation,
        saveToolPresentation,
        toolCache,
        getCss,
        refreshActiveTool,
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
        formatRowMeta,
        measureToolRender,
        startToolTiming,
        setToolTimingFailed,
        syncDuckDbTimingFromData,
        syncClientTimingFromData,
        setStatus,
        setChartMessage,
        setGroupMeta,
        applyToolPresentation,
        saveToolPresentation,
        toolCache,
        syncActiveFilterLabels,
        columnExists,
        numericColumnExists,
        refreshUkMap,
        clearActiveFavouriteSelection: () => clearActiveFavouriteSelectionForScope("map_view"),
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
        canNavigateToLineBarFeature: (featureName) => lineBarTool.canNavigateToFeature(featureName),
        navigateToLineBarFeature: (featureName) => lineBarTool.navigateToFeature(featureName),
        selectExpectedPredictionForModelKind: (modelKind) => setExpectedPredictionSelectionForModelKind(modelKind),
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
        selectExpectedPredictionForModelKind: (modelKind) => setExpectedPredictionSelectionForModelKind(modelKind),
        syncClientTimingFromData,
        syncDuckDbTimingFromData,
        toolCache,
        updateAxisControls: () => lineBarTool.updateAxisControls(),
        refreshActiveTool,
        setDatasetGbmCount,
        reloadSchema: reloadSchemaAfterModelMutation,
        onExternalModelActivation: (modelKind) => glmTool.handleExternalModelActivation(modelKind),
      });
      const specificationsTool = createSpecificationsTool({
        api,
        clearGlobalStatus: () => setStatus(""),
        datasetColumnNames,
        el,
        escapeHtml,
        measureToolRender,
        reloadSchemaAfterSpecsSave,
        showClipboardToast,
      });

      function monitorUrl() {
        return monitorPath({ token, href: location.href });
      }

      function syncMonitorLink() {
        const link = el("monitorLink");
        if (link) link.href = monitorUrl();
      }

      function setHeaderButtonVisible(node, visible) {
        if (!node) return;
        node.classList.toggle("hidden", !visible);
        if (visible) {
          node.removeAttribute("aria-hidden");
          node.removeAttribute("inert");
        } else {
          node.setAttribute("aria-hidden", "true");
          node.setAttribute("inert", "");
        }
      }

      function syncHeaderButtons() {
        syncMonitorLink();
        const visible = state.schema
          ? Boolean(state.schema.header_buttons)
          : !el("monitorLink")?.classList.contains("hidden");
        setHeaderButtonVisible(el("monitorLink"), visible);
        setHeaderButtonVisible(el("stopAppBtn"), visible);
        scheduleDatasetMetaCompactCheck();
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
        scheduleDatasetMetaCompactCheck();
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

      function ensureToolButtonTooltip() {
        if (!toolButtonTooltip) {
          toolButtonTooltip = document.createElement("div");
          toolButtonTooltip.id = "toolButtonTooltip";
          toolButtonTooltip.className = "tool-button-tooltip";
          toolButtonTooltip.hidden = true;
          toolButtonTooltip.setAttribute("role", "tooltip");
          document.body.append(toolButtonTooltip);
        }
        return toolButtonTooltip;
      }

      function toolButtonTooltipText(button) {
        return String(
          button?.dataset?.tooltip
          || button?.getAttribute("aria-label")
          || button?.querySelector(".tool-label")?.textContent
          || "",
        ).trim();
      }

      function positionToolButtonTooltip(target = toolButtonTooltipTarget) {
        const tooltip = toolButtonTooltip;
        if (!tooltip || tooltip.hidden || !target) return;
        const targetRect = target.getBoundingClientRect();
        const margin = 6;
        const tooltipWidth = tooltip.offsetWidth;
        const tooltipHeight = tooltip.offsetHeight;
        const left = Math.max(
          margin,
          Math.min(
            targetRect.left + targetRect.width / 2 - tooltipWidth / 2,
            window.innerWidth - tooltipWidth - margin,
          ),
        );
        let top = targetRect.top - tooltipHeight - margin;
        if (top < margin) {
          top = Math.min(window.innerHeight - tooltipHeight - margin, targetRect.bottom + margin);
        }
        tooltip.style.left = `${Math.round(left)}px`;
        tooltip.style.top = `${Math.round(Math.max(margin, top))}px`;
      }

      function toolButtonTooltipButtonAvailable(button) {
        return Boolean(button && button.isConnected && !button.disabled && !button.classList.contains("hidden"));
      }

      function clearToolButtonTooltipTimer(target = null) {
        if (target && toolButtonTooltipPendingTarget && target !== toolButtonTooltipPendingTarget) return;
        if (toolButtonTooltipTimer) {
          window.clearTimeout(toolButtonTooltipTimer);
          toolButtonTooltipTimer = null;
        }
        if (!target || target === toolButtonTooltipPendingTarget) {
          toolButtonTooltipPendingTarget = null;
        }
      }

      function showToolButtonTooltip(button) {
        if (!toolButtonTooltipButtonAvailable(button)) return;
        const text = toolButtonTooltipText(button);
        if (!text) return;
        const tooltip = ensureToolButtonTooltip();
        toolButtonTooltipTarget = button;
        tooltip.textContent = text;
        tooltip.hidden = false;
        positionToolButtonTooltip(button);
      }

      function scheduleToolButtonTooltip(button) {
        if (!toolButtonTooltipButtonAvailable(button)) return;
        const text = toolButtonTooltipText(button);
        if (!text) return;
        if (toolButtonTooltipTarget === button && toolButtonTooltip && !toolButtonTooltip.hidden) {
          positionToolButtonTooltip(button);
          return;
        }
        hideToolButtonTooltip();
        toolButtonTooltipPendingTarget = button;
        toolButtonTooltipTimer = window.setTimeout(() => {
          toolButtonTooltipTimer = null;
          if (toolButtonTooltipPendingTarget !== button) return;
          toolButtonTooltipPendingTarget = null;
          showToolButtonTooltip(button);
        }, TOOL_BUTTON_TOOLTIP_DELAY_MS);
      }

      function hideToolButtonTooltip(target = null) {
        const pendingMatches = toolButtonTooltipPendingTarget && (!target || target === toolButtonTooltipPendingTarget);
        const visibleMatches = toolButtonTooltipTarget && (!target || target === toolButtonTooltipTarget);
        if (target && !pendingMatches && !visibleMatches) return;
        if (!target || pendingMatches) clearToolButtonTooltipTimer(target);
        if (!target || visibleMatches) {
          if (toolButtonTooltip) toolButtonTooltip.hidden = true;
          if (!target || target === toolButtonTooltipTarget) toolButtonTooltipTarget = null;
        }
      }

      function bindToolButtonTooltips() {
        document.querySelectorAll("#toolSelectorSection .tool-option").forEach((button) => {
          if (button.dataset.tooltipBound === "true") return;
          const tooltipText = String(button.getAttribute("title") || "").trim() || toolButtonTooltipText(button);
          if (tooltipText) button.dataset.tooltip = tooltipText;
          button.removeAttribute("title");
          button.dataset.tooltipBound = "true";
          button.addEventListener("pointerenter", () => scheduleToolButtonTooltip(button));
          button.addEventListener("pointerleave", () => hideToolButtonTooltip(button));
          button.addEventListener("pointerdown", () => hideToolButtonTooltip(button));
          button.addEventListener("focus", () => scheduleToolButtonTooltip(button));
          button.addEventListener("blur", () => hideToolButtonTooltip(button));
        });
        if (!toolButtonTooltipKeydownBound) {
          document.addEventListener("keydown", (event) => {
            if (event.key === "Escape") hideToolButtonTooltip();
          });
          toolButtonTooltipKeydownBound = true;
        }
      }

      function setChartMessage(message) {
        const displayMessage = message || "";
        el("chartMessage").textContent = displayMessage;
        const hiddenForView = state.tool === "line_bar" && state.view !== "chart";
        el("chartMessage").classList.toggle("hidden", !displayMessage || hiddenForView);
      }

      function setGroupMeta(tool, message, options = {}) {
        if (tool === "dataset_viewer") return;
        const id = tool === "uk_map"
            ? "mapGroupMeta"
            : (tool === "column_profile" ? "profileGroupMeta" : (tool === "histogram" ? "histogramGroupMeta" : (isModelTool(tool) ? "modelToolGroupMeta" : "lineBarGroupMeta")));
        const target = el(id);
        if (options.html) {
          target.innerHTML = message || "";
          return;
        }
        target.textContent = message || "";
      }

      function activeFilterLabel() {
        return state.activeFilter || "no filter";
      }

      function datasetViewerReadableColumnCount() {
        const explicitCount = Number(state.datasetViewerColumnCount);
        if (Number.isFinite(explicitCount) && explicitCount >= 0) return explicitCount;
        const schemaCount = Number(state.schema?.columns?.length || 0);
        return Number.isFinite(schemaCount) && schemaCount >= 0 ? schemaCount : 0;
      }

      function syncDatasetViewerMeta() {
        const columnMeta = `${datasetViewerReadableColumnCount().toLocaleString()} columns`;
        const rowMeta = state.filterRowCountMeta?.text || formatRowMeta(state.schema?.row_count || 0);
        el("datasetViewerGroupMeta").textContent = [columnMeta, rowMeta].filter(Boolean).join(" · ");
      }

      function syncActiveFilterLabels() {
        const label = activeFilterLabel();
        el("datasetViewerFilterText").textContent = label;
        el("profileFilterText").textContent = label;
        el("lineBarFilterText").textContent = label;
        el("histogramFilterText").textContent = label;
        el("modelToolFilter").textContent = label;
        el("mapControlFilterText").textContent = label;
      }

      function filterIsApplied() {
        return Boolean(String(state.activeFilter || "").trim());
      }

      function syncActiveFilterIndicator() {
        const applied = filterIsApplied();
        el("filterRowClearBtn").hidden = !applied;
        el("datasetViewerFilterClearBtn").hidden = !applied;
        el("profileFilterClearBtn").hidden = !applied;
        el("lineBarFilterClearBtn").hidden = !applied;
        el("histogramFilterClearBtn").hidden = !applied;
        el("mapControlFilterClearBtn").hidden = !applied;
        el("filterRowMeta").classList.toggle("filter-row-meta--applied", applied);
        el("datasetViewerFilter").classList.toggle("dataset-viewer-filter--applied", applied);
        el("profileFilter").classList.toggle("profile-filter--applied", applied);
        el("lineBarFilter").classList.toggle("line-bar-filter--applied", applied);
        el("histogramFilter").classList.toggle("histogram-filter--applied", applied);
        el("mapControlFilter").classList.toggle("map-filter--applied", applied);
      }

      function setFilterRowMeta(rowCount, filteredRowCount = rowCount) {
        const meta = formatRowMeta(rowCount, filteredRowCount);
        state.filterRowCountMeta = {
          text: meta,
          rowCount,
          filteredRowCount,
          filter: state.activeFilter || "",
        };
        el("filterRowMetaText").textContent = meta || "";
        syncActiveFilterIndicator();
        syncDatasetViewerMeta();
      }

      function setFilterRowMetaText(message) {
        state.filterRowCountMeta = {
          text: message || "",
          filter: state.activeFilter || "",
        };
        el("filterRowMetaText").textContent = message || "";
        syncActiveFilterIndicator();
        syncDatasetViewerMeta();
      }

      function cancelFilterRowCountRequests() {
        state.filterRowCountRequestSeq = (state.filterRowCountRequestSeq || 0) + 1;
      }

      function resetFilterRowMetaToSchema() {
        cancelFilterRowCountRequests();
        setFilterRowMeta(state.schema?.row_count || 0);
      }

      async function refreshFilterRowCountMeta() {
        const filter = state.activeFilter || "";
        if (!filter) {
          resetFilterRowMetaToSchema();
          return;
        }
        const requestSeq = (state.filterRowCountRequestSeq || 0) + 1;
        state.filterRowCountRequestSeq = requestSeq;
        setFilterRowMetaText("updating...");
        try {
          const data = await api("/api/filter/row-count", {
            method: "POST",
            body: JSON.stringify({ filter }),
          });
          if (requestSeq !== state.filterRowCountRequestSeq || filter !== (state.activeFilter || "")) return;
          setFilterRowMeta(data.row_count, data.filtered_row_count);
        } catch (_) {
          if (requestSeq !== state.filterRowCountRequestSeq || filter !== (state.activeFilter || "")) return;
          setFilterRowMetaText("count unavailable");
        }
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

      function isModelRatioSourceId(sourceId) {
        return MODEL_RATIO_SOURCE_RE.test(String(sourceId || ""));
      }

      function lineBarFeatureColumns() {
        const currentSource = state.source || "dataset";
        const currentKind = currentDataSource()?.kind || "";
        const columns = dataSourceColumns("dataset").map((column) => ({
          ...column,
          source_id: "dataset",
        }));
        const currentModelColumns = sourceColumns()
          .filter((column) => (
            isModelPredictionColumn(column)
            || isGbmShapValueColumn(column)
            || (currentKind === "model_ratio" && String(column?.name || "") === LINE_BAR_RATIO_COLUMN)
          ))
          .map((column) => ({
            ...column,
            source_id: column.source_id || currentSource,
          }));
        const seen = new Set(columns.map((column) => `${column.source_id || "dataset"}\u0000${column.name}`));
        for (const column of [...currentModelColumns, ...activeModelRatioColumns(), ...activePredictionColumns()]) {
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
        return match ? lineBarColumnSourceId(match) : "";
      }

      function syncLineBarXFallback() {
        if (lineBarColumnExists(state.x, state.xSource)) return;
        const currentFeature = String(state.x || "");
        if (currentFeature) {
          const currentSource = state.source || "dataset";
          const preservedSource = lineBarFeatureSourceForName(currentFeature, currentSource)
            || lineBarFeatureSourceForName(currentFeature);
          if (preservedSource) {
            if (preservedSource === "dataset") state.source = "dataset";
            state.x = currentFeature;
            state.xSource = preservedSource;
            return;
          }
        }
        const first = lineBarFeatureColumns()[0] || null;
        if (first && lineBarColumnSourceId(first) === "dataset") state.source = "dataset";
        state.x = first?.name || null;
        state.xSource = first ? lineBarColumnSourceId(first) : "";
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

      function datasetColumnNames() {
        const readable = (state.schema?.columns || []).map((column) => column.name);
        const invalid = (state.schema?.invalid_columns || []).map((column) => column.name);
        return [...readable, ...invalid].filter(Boolean);
      }

      function preferredStartupSource(availableSources, requestedSource) {
        return schemaPreferredStartupSource(availableSources, requestedSource);
      }

      function toolEnabled(id) {
        return schemaToolEnabled(state.schema, id);
      }

      function enabledToolIds() {
        return (state.schema?.tools || [])
          .map((tool) => String(tool?.id || ""))
          .filter((toolId) => toolId && Object.prototype.hasOwnProperty.call(TOOL_BUTTON_IDS, toolId));
      }

      function freshProfileCache() {
        return { requestKey: null, data: null, presentation: null, themeKey: null, details: new Map() };
      }

      function freshToolCache() {
        return {
          dataset_viewer: { requestKey: null, data: null, presentation: null, themeKey: null },
          column_profile: freshProfileCache(),
          line_bar: { requestKey: null, data: null, presentation: null, themeKey: null },
          histogram: { requestKey: null, data: null, presentation: null, themeKey: null },
          uk_map: { requestKey: null, data: null, presentation: null, themeKey: null },
          glm: { requestKey: null, data: null, presentation: null, themeKey: null },
          gbm: { requestKey: null, data: null, presentation: null, themeKey: null },
          specs: { requestKey: null, data: null, presentation: null, themeKey: null },
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
          state.lastProfileData = null;
          clearProfileDetailCache();
        }
        state.lastDatasetViewerData = null;
        state.lastData = null;
        state.lastHistogramData = null;
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
          state.toolCache[tool] = tool === "column_profile" ? freshProfileCache() : { requestKey: null, data: null, presentation: null, themeKey: null };
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

      function currentThemeKey() {
        return document.body.classList.contains("dark") ? "dark" : "light";
      }

      function markToolCacheThemeSynced(tool) {
        toolCache(tool).themeKey = currentThemeKey();
      }

      function metricSummaryRequest() {
        const actual = el("actualNumerator")?.value || "";
        if (!state.schema || !actual) return null;
        const actualOption = el("actualNumerator")?.selectedOptions?.[0] || null;
        return {
          source: actualOption?.dataset.sourceId || state.source || "dataset",
          actual,
          denominator: el("denominator")?.value || "__none__",
          filter: state.activeFilter || "",
        };
      }

      function resetMetricSummaryTitles() {
        renderMetricTitle(el("actualMetricTitle"), "Actual");
        renderMetricTitle(el("weightMetricTitle"), "Weight", null, formatWeightValue);
      }

      function renderMetricSummary(data) {
        const summaries = Array.isArray(data?.response_summaries) ? data.response_summaries : [];
        renderMetricTitle(el("actualMetricTitle"), "Actual", summaries[0]?.value);
        renderMetricTitle(el("weightMetricTitle"), "Weight", data?.denominator?.value, formatWeightValue);
      }

      function cancelMetricSummaryRequests() {
        state.metricSummaryRequestSeq = (state.metricSummaryRequestSeq || 0) + 1;
      }

      function toolResponseSummaries(data) {
        if (Array.isArray(data?.response_summaries)) return data.response_summaries;
        if (Array.isArray(data?.summary?.responses)) {
          return data.summary.responses.map((value) => ({ value }));
        }
        if (data?.response && Object.prototype.hasOwnProperty.call(data.response, "value")) {
          return [{
            value: data.response.value,
            numerator: data.response.numerator_total,
            denominator: data.response.denominator,
          }];
        }
        return [];
      }

      function syncMetricSummaryFromToolData(data) {
        if (!data) return;
        const summary = {
          response_summaries: toolResponseSummaries(data),
          denominator: data.denominator || {},
        };
        const request = metricSummaryRequest();
        state.metricSummaryRequestKey = request ? stableRequestKey(request) : null;
        state.metricSummaryData = summary;
        renderMetricSummary(summary);
      }

      function syncFilterRowMetaFromToolData(data) {
        const rowCount = Number(data?.row_count);
        const filteredRowCount = Number(data?.filtered_row_count);
        if (!Number.isFinite(rowCount)) return;
        setFilterRowMeta(rowCount, Number.isFinite(filteredRowCount) ? filteredRowCount : rowCount);
      }

      function beginFavouriteViewRestore() {
        cancelMetricSummaryRequests();
        cancelFilterRowCountRequests();
        state.metricSummaryRequestKey = null;
        state.metricSummaryData = null;
        resetMetricSummaryTitles();
        setStatus("");
        setChartMessage("");
      }

      function syncSidebarSummariesFromToolData(data) {
        syncMetricSummaryFromToolData(data);
        syncFilterRowMetaFromToolData(data);
      }

      async function refreshMetricSummary(options = {}) {
        const request = metricSummaryRequest();
        if (!request) {
          cancelMetricSummaryRequests();
          state.metricSummaryRequestKey = null;
          state.metricSummaryData = null;
          resetMetricSummaryTitles();
          return null;
        }
        const requestKey = stableRequestKey(request);
        if (!options.force && state.metricSummaryData && state.metricSummaryRequestKey === requestKey) {
          renderMetricSummary(state.metricSummaryData);
          return state.metricSummaryData;
        }
        const requestSeq = (state.metricSummaryRequestSeq || 0) + 1;
        state.metricSummaryRequestSeq = requestSeq;
        try {
          const data = await api("/api/metrics/summary", {
            method: "POST",
            body: JSON.stringify(request),
          });
          const latestRequest = metricSummaryRequest();
          const latestRequestKey = latestRequest ? stableRequestKey(latestRequest) : null;
          if (requestSeq !== state.metricSummaryRequestSeq || latestRequestKey !== requestKey) return null;
          state.metricSummaryRequestKey = requestKey;
          state.metricSummaryData = data;
          renderMetricSummary(data);
          return data;
        } catch (_) {
          const latestRequest = metricSummaryRequest();
          const latestRequestKey = latestRequest ? stableRequestKey(latestRequest) : null;
          if (requestSeq !== state.metricSummaryRequestSeq || latestRequestKey !== requestKey) return null;
          state.metricSummaryRequestKey = null;
          state.metricSummaryData = null;
          resetMetricSummaryTitles();
          return null;
        }
      }

      function saveToolPresentation(tool, presentation) {
        toolCache(tool).presentation = {
          groupMeta: presentation.groupMeta || "",
          groupMetaHtml: presentation.groupMetaHtml || "",
          status: presentation.status || "",
          statusError: Boolean(presentation.statusError),
          chartMessage: presentation.chartMessage || "",
        };
      }

      function applyToolPresentation(tool) {
        const presentation = toolCache(tool).presentation;
        if (!presentation) return;
        setGroupMeta(tool, presentation.groupMetaHtml || presentation.groupMeta, { html: Boolean(presentation.groupMetaHtml) });
        setStatus(presentation.status, presentation.statusError);
        setChartMessage(presentation.chartMessage);
      }

      async function toolHandler(tool) {
        if (tool === "dataset_viewer") {
          const loadedDatasetViewerTool = await ensureDatasetViewerTool();
          if (!loadedDatasetViewerTool) return null;
          return {
            buildRequest: () => loadedDatasetViewerTool.buildRequest(),
            fetch: (request, requestKey) => loadedDatasetViewerTool.fetchData(request, requestKey),
            useCached: (cache) => loadedDatasetViewerTool.useCached(cache),
          };
        }
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
        if (tool === "histogram") {
          return {
            buildRequest: () => histogramTool.buildRequest(),
            fetch: (request, requestKey) => histogramTool.fetchData(request, requestKey),
            useCached: (cache, options) => histogramTool.useCached(cache, options),
          };
        }
        if (tool === "uk_map") {
          return {
            buildRequest: () => ukMapTool.buildRequest(),
            fetch: (request, requestKey) => ukMapTool.fetchData(request, requestKey),
            useCached: (cache, options) => ukMapTool.useCached(cache, options),
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
        const handler = await toolHandler(tool);
        if (!handler) return null;
        const request = handler.buildRequest();
        if (!request) {
          handler.handleMissingRequest?.();
          return null;
        }
        const requestKey = stableRequestKey(request);
        const cache = toolCache(tool);
        if (!options.force && cache.data && cache.requestKey === requestKey) {
          const themeChanged = cache.themeKey !== currentThemeKey();
          await handler.useCached(cache, { ...options, renderIfCached: options.renderIfCached || themeChanged });
          markToolCacheThemeSynced(tool);
          return cache.data;
        }
        const data = await handler.fetch(request, requestKey);
        if (data && toolCache(tool).requestKey === requestKey) markToolCacheThemeSynced(tool);
        return data;
      }

      function refreshActiveTool(options = {}) {
        if (state.tool === "specs") return specificationsTool.refresh(options);
        return refreshTool(state.tool, options);
      }

      function refreshLineBar(options = {}) {
        return refreshTool("line_bar", options);
      }

      function invalidateLineBarDateBucketSuggestion() {
        state.dateBucketFeature = null;
        state.dateBucketManualKey = null;
        state.dateBucketSuggestionPendingKey = null;
        state.dateBucketSuggestionRequestSeq = (state.dateBucketSuggestionRequestSeq || 0) + 1;
      }

      function refreshUkMap(options = {}) {
        return refreshTool("uk_map", options);
      }

      function toolUsesMetricControls(tool = state.tool) {
        return ["line_bar", "histogram", "uk_map", "glm", "gbm"].includes(tool);
      }

      function refreshActiveToolForMetricChange() {
        if (toolUsesMetricControls()) refreshActiveTool();
      }

      function syncActiveToolTheme() {
        const activeTool = state.tool;
        if (activeTool === "line_bar") {
          lineBarTool.refreshTheme();
          markToolCacheThemeSynced(activeTool);
          return;
        }
        if (activeTool === "dataset_viewer") {
          ensureDatasetViewerTool()
            .then((loadedDatasetViewerTool) => loadedDatasetViewerTool?.refreshTheme())
            .finally(() => markToolCacheThemeSynced(activeTool));
          return;
        }
        if (activeTool === "histogram") {
          histogramTool.refreshTheme();
          markToolCacheThemeSynced(activeTool);
          return;
        }
        if (activeTool === "uk_map") {
          ukMapTool.refreshTheme();
          markToolCacheThemeSynced(activeTool);
          return;
        }
        if (activeTool === "glm") {
          measureToolRender("glm", () => glmTool.refreshTheme());
          markToolCacheThemeSynced(activeTool);
          return;
        }
        if (activeTool === "gbm") {
          measureToolRender("gbm", () => gbmTool.refreshTheme());
          markToolCacheThemeSynced(activeTool);
          return;
        }
        if (activeTool === "specs") {
          measureToolRender("specs", () => specificationsTool.refreshTheme());
          markToolCacheThemeSynced(activeTool);
        }
      }

      function chooseDefaultTool() {
        const requested = locationParams.get("tool");
        if (requested && toolEnabled(requested)) return requested;
        if (requestedDefault("line_bar_favourite") && toolEnabled("line_bar")) return "line_bar";
        const firstEnabled = enabledToolIds()[0] || "";
        if (firstEnabled && toolEnabled(firstEnabled)) return firstEnabled;
        for (const toolId of TOOL_IDS) {
          if (toolEnabled(toolId)) return toolId;
        }
        return "column_profile";
      }

      function setModelSidebarPanelVisibility(panelId, enabled) {
        const panel = el(panelId);
        if (!panel) return;
        panel.classList.toggle("hidden", !enabled);
        if (enabled) {
          panel.removeAttribute("aria-hidden");
        } else {
          panel.setAttribute("aria-hidden", "true");
        }
        panel.toggleAttribute("inert", !enabled);
      }

      function renderToolSelector() {
        const enabledTools = enabledToolIds();
        const enabledSet = new Set(enabledTools);
        const selector = document.querySelector("#toolSelectorSection .tool-selector");
        if (selector) {
          [...enabledTools, ...TOOL_IDS.filter((toolId) => !enabledSet.has(toolId))].forEach((toolId) => {
            const button = el(TOOL_BUTTON_IDS[toolId]);
            if (button) selector.append(button);
          });
        }
        const showSelector = enabledTools.length > 1;
        document.body.classList.toggle("single-tool-mode", enabledTools.length === 1);
        TOOL_IDS.forEach((toolId) => {
          const button = el(TOOL_BUTTON_IDS[toolId]);
          const enabled = enabledSet.has(toolId);
          if (!button) return;
          button.disabled = !enabled;
          button.classList.toggle("hidden", !showSelector || !enabled);
        });
        el("toolSelectorSection").classList.toggle("hidden", !showSelector);
        if (
          (toolButtonTooltipTarget && !toolButtonTooltipButtonAvailable(toolButtonTooltipTarget))
          || (toolButtonTooltipPendingTarget && !toolButtonTooltipButtonAvailable(toolButtonTooltipPendingTarget))
        ) {
          hideToolButtonTooltip();
        }
        const glmEnabled = enabledSet.has("glm");
        const gbmEnabled = enabledSet.has("gbm");
        setModelSidebarPanelVisibility("gbmSidebarPanel", gbmEnabled);
        setModelSidebarPanelVisibility("glmSidebarPanel", glmEnabled);
        if ((state.openSidebarSection === "gbm" && !gbmEnabled) || (state.openSidebarSection === "glm" && !glmEnabled)) {
          state.openSidebarSection = null;
        }
        syncSidebarAccordion();
      }

      function schemaFileMeta() {
        const parts = [];
        let path = state.schema?.path?.split(/[\\/]/).pop() || "";
        if (state.schema?.source_kind === "parquet_folder") {
          const fileCount = Number(state.schema?.file_count || 0);
          if (Number.isFinite(fileCount) && fileCount > 0) {
            path = `${path} (${fileCount.toLocaleString()} ${fileCount === 1 ? "file" : "files"})`;
          }
        }
        if (path) parts.push(path);
        const fileSize = formatFileSize(state.schema?.file_size);
        if (fileSize) parts.push(fileSize);
        return parts.join(" · ");
      }

      function renderSidebarVersion() {
        const target = el("sidebarVersion");
        const collapsedTarget = el("collapsedSidebarVersion");
        if (!target || !collapsedTarget) return;
        const version = String(state.schema?.app_version || "").trim();
        target.textContent = version ? `lucidum v${version}` : "";
        collapsedTarget.textContent = version ? `v${version}` : "";
        target.hidden = !version;
        collapsedTarget.hidden = !version;
      }

      function scheduleDatasetMetaCompactCheck() {
        if (datasetMetaCompactFrame !== null) return;
        datasetMetaCompactFrame = requestAnimationFrame(() => {
          datasetMetaCompactFrame = null;
          updateDatasetMetaCompactMode();
        });
      }

      function updateDatasetMetaCompactMode() {
        const target = el("datasetMeta");
        if (!target) return;
        const title = target.querySelector(".dataset-meta-title");
        const details = target.querySelector(".dataset-meta-details");
        if (!title || !details) {
          target.classList.remove("dataset-meta-title-only");
          return;
        }
        target.classList.remove("dataset-meta-title-only");
        const overflows = target.scrollWidth > target.clientWidth + 1;
        target.classList.toggle("dataset-meta-title-only", overflows);
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
        const titlePrefix = String(state.schema?.title_prefix || "").trim();
        if (titlePrefix) {
          const title = document.createElement("span");
          title.className = "dataset-meta-title";
          title.textContent = titlePrefix;
          target.append(title);
        }
        const details = document.createElement("span");
        details.className = "dataset-meta-details";
        target.append(details);
        const leadingParts = [datasetMetaBase, `${rows} rows`].filter(Boolean);
        if (leadingParts.length) {
          details.append(document.createTextNode(`${titlePrefix ? " · " : ""}${leadingParts.join(" · ")} · `));
        }
        const columnCount = document.createElement("span");
        columnCount.className = "dataset-meta-column-count";
        columnCount.textContent = `${columns} columns`;
        details.append(columnCount);
        if (datasetGlmCount !== null && toolEnabled("glm")) {
          details.append(document.createTextNode(" · "));
          const button = document.createElement("button");
          button.type = "button";
          button.className = "dataset-meta-glm-link";
          button.textContent = `GLMs (${datasetGlmCount.toLocaleString()})`;
          button.title = "Open GLM Model navigator";
          button.setAttribute("aria-label", `Open saved GLMs, ${datasetGlmCount.toLocaleString()} models`);
          button.addEventListener("click", openGlmModelNavigator);
          details.append(button);
        }
        if (datasetGbmCount !== null && toolEnabled("gbm")) {
          details.append(document.createTextNode(" · "));
          const button = document.createElement("button");
          button.type = "button";
          button.className = "dataset-meta-gbm-link";
          button.textContent = `GBMs (${datasetGbmCount.toLocaleString()})`;
          button.title = "Open GBM Model navigator";
          button.setAttribute("aria-label", `Open saved GBMs, ${datasetGbmCount.toLocaleString()} models`);
          button.addEventListener("click", openGbmModelNavigator);
          details.append(button);
        }
        scheduleDatasetMetaCompactCheck();
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
        hideToolButtonTooltip();
        if (!toolEnabled(tool)) return;
        const previousTool = state.tool;
        if (previousTool === "uk_map" && tool !== "uk_map") ukMapTool.captureView("tool-switch");
        if (previousTool === "column_profile" && tool !== "column_profile") columnProfileTool.closeMenus();
        if (previousTool === "specs" && tool !== "specs") specificationsTool.closeMenus();
        state.tool = tool;
        if (previousTool && previousTool !== tool) {
          if (previousTool === "line_bar") clearActiveFavouriteSelectionForScope("line_bar_view");
          if (previousTool === "uk_map") clearActiveFavouriteSelectionForScope("map_view");
        }
        el("visualArea").classList.remove("startup-mode");
        el("datasetViewerTool").classList.toggle("active", tool === "dataset_viewer");
        el("profileTool").classList.toggle("active", tool === "column_profile");
        el("lineBarTool").classList.toggle("active", tool === "line_bar");
        el("histogramTool").classList.toggle("active", tool === "histogram");
        el("ukMapTool").classList.toggle("active", tool === "uk_map");
        el("glmTool").classList.toggle("active", tool === "glm");
        el("gbmTool").classList.toggle("active", tool === "gbm");
        el("specsTool").classList.toggle("active", tool === "specs");
        glmTool.syncSidebarFromSchema();
        gbmTool.syncSidebarFromSchema();
        syncSidebarAccordion();
        syncFavouriteActionButtons();
        el("histogramToolbar").classList.toggle("hidden", tool !== "histogram");
        el("visualArea").classList.toggle("map-mode", tool === "uk_map");
        el("visualArea").classList.toggle("dataset-viewer-mode", tool === "dataset_viewer");
        el("visualArea").classList.toggle("profile-mode", tool === "column_profile");
        el("visualArea").classList.toggle("histogram-mode", tool === "histogram");
        el("visualArea").classList.toggle("specs-mode", tool === "specs");
        el("visualArea").classList.toggle("model-mode", isModelTool(tool));
        el("lineBarTabs").classList.toggle("hidden", tool !== "line_bar");
        syncLineBarLayoutVisibility();
        el("datasetViewerGroupMeta").classList.toggle("hidden", tool !== "dataset_viewer");
        el("datasetViewerFilter").classList.toggle("hidden", tool !== "dataset_viewer");
        el("profileGroupMeta").classList.toggle("hidden", tool !== "column_profile");
        el("profileFilter").classList.toggle("hidden", tool !== "column_profile");
        el("lineBarGroupMeta").classList.toggle("hidden", tool !== "line_bar");
        el("lineBarFilter").classList.toggle("hidden", tool !== "line_bar");
        el("histogramGroupMeta").classList.toggle("hidden", tool !== "histogram");
        el("histogramFilter").classList.toggle("hidden", tool !== "histogram");
        el("modelToolGroupMeta").classList.toggle("hidden", !isModelTool(tool) || tool === "gbm");
        el("modelToolFilter").classList.add("hidden");
        el("mapFloatingControl").classList.toggle("hidden", tool !== "uk_map");
        el("mapLegend").classList.toggle("hidden", tool !== "uk_map" || !el("mapLegend").textContent);
        el("datasetViewerWrap").classList.toggle("hidden", tool !== "dataset_viewer");
        el("profileWrap").classList.toggle("hidden", tool !== "column_profile");
        el("modelToolWrap").classList.toggle("hidden", !isModelTool(tool));
        el("specificationsWrap").classList.toggle("hidden", tool !== "specs");
        syncActiveFilterLabels();
        syncDatasetViewerMeta();
        syncActionTimingMonitor(tool);
        setStatus("");
        setChartMessage("");
        if (tool === "dataset_viewer") {
          el("chart").classList.add("hidden");
          el("tableWrap").classList.add("hidden");
          el("histogramWrap").classList.add("hidden");
          el("ukMap").classList.add("hidden");
          el("mapLegend").classList.add("hidden");
          el("profileWrap").classList.add("hidden");
          el("modelToolWrap").classList.add("hidden");
          el("specificationsWrap").classList.add("hidden");
          el("datasetViewerWrap").classList.remove("hidden");
          scheduleActiveToolResize({ hard: true });
        } else if (tool === "line_bar") {
          el("datasetViewerWrap").classList.add("hidden");
          el("profileWrap").classList.add("hidden");
          el("modelToolWrap").classList.add("hidden");
          el("specificationsWrap").classList.add("hidden");
          el("histogramWrap").classList.add("hidden");
          el("ukMap").classList.add("hidden");
          el("mapLegend").classList.add("hidden");
          lineBarTool.setView(state.view, { refresh });
          lineBarTool.updateAxisControls();
          requestAnimationFrame(() => {
            applyStartupChartExpectedCollapse();
            lineBarTool.resize();
          });
        } else if (tool === "histogram") {
          el("datasetViewerWrap").classList.add("hidden");
          el("profileWrap").classList.add("hidden");
          el("modelToolWrap").classList.add("hidden");
          el("specificationsWrap").classList.add("hidden");
          el("chart").classList.add("hidden");
          el("tableWrap").classList.add("hidden");
          el("ukMap").classList.add("hidden");
          el("mapLegend").classList.add("hidden");
          el("histogramWrap").classList.remove("hidden");
          histogramTool.activate();
        } else if (tool === "uk_map") {
          el("datasetViewerWrap").classList.add("hidden");
          el("profileWrap").classList.add("hidden");
          el("modelToolWrap").classList.add("hidden");
          el("specificationsWrap").classList.add("hidden");
          el("chart").classList.add("hidden");
          el("tableWrap").classList.add("hidden");
          el("histogramWrap").classList.add("hidden");
          el("ukMap").classList.remove("hidden");
          ukMapTool.activate();
        } else if (tool === "specs") {
          el("datasetViewerWrap").classList.add("hidden");
          el("chart").classList.add("hidden");
          el("tableWrap").classList.add("hidden");
          el("histogramWrap").classList.add("hidden");
          el("ukMap").classList.add("hidden");
          el("mapLegend").classList.add("hidden");
          el("profileWrap").classList.add("hidden");
          el("modelToolWrap").classList.add("hidden");
          el("specificationsWrap").classList.remove("hidden");
        } else {
          el("datasetViewerWrap").classList.add("hidden");
          el("chart").classList.add("hidden");
          el("tableWrap").classList.add("hidden");
          el("histogramWrap").classList.add("hidden");
          el("ukMap").classList.add("hidden");
          el("mapLegend").classList.add("hidden");
          el("specificationsWrap").classList.add("hidden");
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

      let activeToolResizeFrame = null;
      let activeToolResizeHard = false;

      function scheduleActiveToolResize({ hard = true } = {}) {
        activeToolResizeHard = activeToolResizeHard || hard;
        if (activeToolResizeFrame !== null) return;
        activeToolResizeFrame = requestAnimationFrame(() => {
          const shouldHard = activeToolResizeHard;
          activeToolResizeFrame = null;
          activeToolResizeHard = false;
          resizeActiveTool({ hard: shouldHard });
        });
      }

      function resizeActiveTool({ hard = true } = {}) {
        const resizeOptions = { hard };
        if (state.tool === "dataset_viewer") {
          if (!hard) return;
          if (datasetViewerTool) {
            datasetViewerTool.resize(resizeOptions);
          } else {
            ensureDatasetViewerTool().then((loadedDatasetViewerTool) => loadedDatasetViewerTool?.resize(resizeOptions));
          }
        } else if (state.tool === "uk_map") {
          ukMapTool.syncViewport({ mode: "preserve" });
        } else if (state.tool === "histogram") {
          histogramTool.resize();
        } else if (state.tool === "glm") {
          glmTool.resize();
        } else if (state.tool === "gbm") {
          gbmTool.resize?.();
        } else if (state.tool === "specs") {
          specificationsTool.resize();
        } else {
          syncChartControlHeightToAvailableSpace();
          lineBarTool.resize();
        }
      }

      function handleToolClick(tool) {
        hideToolButtonTooltip();
        if (state.tool === tool) {
          setSidebarVisible(!state.sidebarVisible);
          return;
        }
        setTool(tool);
      }

      function setSidebarVisible(visible) {
        hideToolButtonTooltip();
        state.sidebarVisible = Boolean(visible);
        document.body.classList.toggle("sidebar-collapsed", !state.sidebarVisible);
        el("appSidebar").removeAttribute("aria-hidden");
        syncSidebarToggleButton();
        scheduleActiveToolResize({ hard: true });
      }

      function syncMobileSidebarLayout({ initial = false } = {}) {
        const mobile = window.innerWidth <= MOBILE_LAYOUT_MAX_WIDTH;
        const enteredMobile = mobile && mobileLayoutActive !== true;
        mobileLayoutActive = mobile;
        if ((initial || enteredMobile) && mobile && state.sidebarVisible) {
          setSidebarVisible(false);
        }
      }

      function toggleLineBarSideControls() {
        setLineBarSideControlsCollapsed(!state.lineBarSideControlsCollapsed);
      }

      function setLineBarSideControlsCollapsed(collapsed) {
        const nextCollapsed = Boolean(collapsed);
        if (state.lineBarSideControlsCollapsed === nextCollapsed) {
          syncLineBarLayoutVisibility();
          scheduleLineBarLayoutResize();
          return;
        }
        state.lineBarSideControlsCollapsed = nextCollapsed;
        syncLineBarLayoutVisibility();
        scheduleLineBarLayoutResize();
      }

      function toggleLineBarToolbar() {
        setLineBarToolbarCollapsed(!state.lineBarToolbarCollapsed);
      }

      function setLineBarToolbarCollapsed(collapsed) {
        const nextCollapsed = Boolean(collapsed);
        if (state.lineBarToolbarCollapsed === nextCollapsed) {
          syncLineBarLayoutVisibility();
          scheduleLineBarLayoutResize();
          return;
        }
        state.lineBarToolbarCollapsed = nextCollapsed;
        syncLineBarLayoutVisibility();
        scheduleLineBarLayoutResize();
      }

      function setupLineBarLayoutToggles() {
        el("lineBarSideControlsToggleBtn")?.addEventListener("click", toggleLineBarSideControls);
        el("lineBarToolbarToggleBtn")?.addEventListener("click", toggleLineBarToolbar);
        syncLineBarLayoutVisibility();
      }

      function syncLineBarLayoutVisibility() {
        const lineBarActive = state.tool === "line_bar";
        const sideCollapsed = lineBarActive && state.lineBarSideControlsCollapsed;
        const toolbarCollapsed = lineBarActive && state.lineBarToolbarCollapsed;
        const toolbar = el("lineBarToolbar");
        const visualArea = el("visualArea");
        const sideControls = el("chartSideControls");
        const resizer = el("chartControlsResizer");

        toolbar.classList.toggle("hidden", !lineBarActive || toolbarCollapsed);
        toolbar.toggleAttribute("inert", !lineBarActive || toolbarCollapsed);
        if (toolbarCollapsed && toolbar.contains(document.activeElement)) {
          document.activeElement?.blur?.();
        }

        sideControls.classList.toggle("hidden", !lineBarActive);
        resizer.classList.toggle("hidden", !lineBarActive);
        visualArea.classList.toggle("line-bar-side-controls-collapsed", sideCollapsed);
        sideControls.toggleAttribute("inert", !lineBarActive || sideCollapsed);
        resizer.toggleAttribute("inert", !lineBarActive || sideCollapsed);
        if (sideCollapsed && sideControls.contains(document.activeElement)) {
          document.activeElement?.blur?.();
        }
        if (!lineBarActive || sideCollapsed) {
          sideControls.setAttribute("aria-hidden", "true");
          resizer.setAttribute("aria-hidden", "true");
        } else {
          sideControls.removeAttribute("aria-hidden");
          resizer.removeAttribute("aria-hidden");
        }

        syncLineBarSideControlsToggle();
        syncLineBarToolbarToggle();
      }

      function syncLineBarSideControlsToggle() {
        const button = el("lineBarSideControlsToggleBtn");
        if (!button) return;
        const collapsed = state.tool === "line_bar" && state.lineBarSideControlsCollapsed;
        const label = collapsed ? "Show x-axis and Expected controls" : "Hide x-axis and Expected controls";
        button.setAttribute("aria-expanded", String(!collapsed));
        button.setAttribute("aria-label", label);
        button.title = label;
      }

      function syncLineBarToolbarToggle() {
        const button = el("lineBarToolbarToggleBtn");
        if (!button) return;
        const collapsed = state.tool === "line_bar" && state.lineBarToolbarCollapsed;
        const label = collapsed ? "Show chart control row" : "Hide chart control row";
        button.setAttribute("aria-expanded", String(!collapsed));
        button.setAttribute("aria-label", label);
        button.title = label;
      }

      function scheduleLineBarLayoutResize() {
        requestAnimationFrame(() => {
          syncChartControlHeightToAvailableSpace();
          lineBarTool.resize();
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
        scheduleActiveToolResize({ hard: true });
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
        favourites: { sectionSelector: ".sidebar-favourites-section", buttonId: "favouritesCollapseBtn", label: "FAVOURITES" },
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
        document.querySelector(".sidebar-metric-section")?.classList.toggle("hidden", state.openSidebarSection !== null);
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
        for (const col of sortedMetricColumns(numericColumns())) {
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
          for (const column of sortedMetricColumns(columns)) {
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

      function activeModelRatioColumns() {
        return activeModelSources(["model_ratio"]).flatMap((source) => (
          numericColumnsForSource(source.id)
            .map((column) => ({
              ...column,
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
        const currentSourceColumns = numericColumnsForSource("dataset")
          .filter((column) => column.source_role !== "gbm_shap_value" && !isModelPredictionColumn(column))
          .map((column) => ({
            ...column,
            source_id: "dataset",
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

      function compareMetricColumns(a, b) {
        const compareOptions = { sensitivity: "base", numeric: true };
        return metricColumnLabel(a).localeCompare(metricColumnLabel(b), undefined, compareOptions)
          || String(a?.name || "").localeCompare(String(b?.name || ""), undefined, compareOptions)
          || String(a?.source_id || "").localeCompare(String(b?.source_id || ""), undefined, compareOptions);
      }

      function sortedMetricColumns(columns = []) {
        return [...columns].sort(compareMetricColumns);
      }

      function sortedDenominatorColumns() {
        return sortedMetricColumns(numericColumns().map((column) => ({ ...column, label: column.name })));
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

      function expectedSelectionKey(selection) {
        return `${selection?.sourceId || ""}\u0000${selection?.value || ""}`;
      }

      function expectedOptionForSelection(value, sourceId = "", options = {}) {
        const name = String(value || "");
        if (!name) return null;
        const allowAnySource = options.allowAnySource !== false;
        const selectOptions = Array.from(el("expectedNumerator").options);
        return selectOptions.find((item) => (
          !item.disabled &&
          item.value === name &&
          (!sourceId || item.dataset.sourceId === sourceId)
        )) || (allowAnySource ? selectOptions.find((item) => !item.disabled && item.value === name) : null) || null;
      }

      function expectedSelectionFromOption(option) {
        if (!option?.value) return null;
        return {
          value: option.value,
          sourceId: option.dataset.sourceId || state.source || "dataset",
          metricKind: option.dataset.metricKind || "metric",
        };
      }

      function modelKindForPredictionColumn(columnName = "") {
        const name = String(columnName || "");
        if (GLM_PREDICTION_COLUMNS.includes(name)) return "glm";
        if (GBM_PREDICTION_COLUMNS.includes(name)) return "gbm";
        return "";
      }

      function modelKindForPredictionSource(sourceId = "") {
        const source = String(sourceId || "");
        if (GLM_PREDICTION_SOURCE_RE.test(source)) return "glm";
        if (GBM_PREDICTION_SOURCE_RE.test(source)) return "gbm";
        return "";
      }

      function resolveFavouriteSourceId(columnName = "", sourceId = "") {
        const source = String(sourceId || "");
        if (String(columnName || "") === LINE_BAR_RATIO_COLUMN || isModelRatioSourceId(source)) {
          return activeModelSource("model_ratio")?.id || source;
        }
        const modelKind = modelKindForPredictionColumn(columnName) || modelKindForPredictionSource(source);
        if (modelKind) return activePredictionSourceForModelKind(modelKind)?.id || source;
        return source;
      }

      function normaliseExpectedSelections(selections = [], options = {}) {
        const allowAnySource = options.allowAnySource !== false;
        const seen = new Set();
        const normalised = [];
        for (const selection of selections) {
          const value = String(selection?.value || selection?.column || "");
          if (!value) continue;
          const requestedSource = String(selection?.sourceId || selection?.source || "");
          const targetSource = typeof resolveFavouriteSourceId === "function" ? resolveFavouriteSourceId(value, requestedSource) : requestedSource;
          const option = expectedOptionForSelection(value, targetSource, { allowAnySource });
          const next = expectedSelectionFromOption(option);
          if (!next) continue;
          const key = expectedSelectionKey(next);
          if (seen.has(key)) continue;
          seen.add(key);
          normalised.push(next);
          if (normalised.length >= 2) break;
        }
        return normalised;
      }

      function syncExpectedSelectToSelections() {
        const select = el("expectedNumerator");
        const first = expectedSelections()[0];
        if (!first) {
          select.value = "";
          return;
        }
        const option = expectedOptionForSelection(first.value, first.sourceId, { allowAnySource: false });
        if (option) option.selected = true;
        else select.value = "";
      }

      function expectedSelections() {
        return Array.isArray(state.expectedSelections) ? state.expectedSelections : [];
      }

      function setExpectedSelections(selections = [], options = {}) {
        const normalised = normaliseExpectedSelections(selections, options);
        state.expectedSelections = normalised;
        syncExpectedSelectToSelections();
        return normalised.length === selections.filter((selection) => String(selection?.value || selection?.column || "")).slice(0, 2).length;
      }

      function clearExpectedSelections() {
        state.expectedSelections = [];
        syncExpectedSelectToSelections();
      }

      function setExpectedSelection(value, sourceId = "", options = {}) {
        const name = String(value || "");
        if (!name) {
          clearExpectedSelections();
          return true;
        }
        const selections = options.append
          ? [...expectedSelections(), { value: name, sourceId }]
          : [{ value: name, sourceId }];
        return setExpectedSelections(selections, options);
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
        for (const col of sortedDenominatorColumns()) {
          select.append(new Option(col.name, col.name));
        }
      }

      function columnExists(name) {
        return Boolean(name && sourceColumns().some((col) => col.name === name));
      }

      function numericColumnExists(name) {
        return Boolean(name && numericColumns().some((col) => col.name === name));
      }

      function datasetNumericColumnExists(name) {
        return Boolean(name && numericColumnsForSource("dataset").some((col) => col.name === name));
      }

      function selectedExpectedIsPrediction() {
        return expectedSelections()[0]?.metricKind === "prediction";
      }

      function expectedSelectionSourceId() {
        return expectedSelections()[0]?.sourceId || "";
      }

      function predictionColumnNamesForModelKind(modelKind) {
        if (modelKind === "glm") return ["glm_prediction", "glm_prediction_rate"];
        if (modelKind === "gbm") return ["gbm_prediction", "gbm_prediction_rate"];
        return [];
      }

      function predictionColumnNameForModelKind(modelKind) {
        return predictionColumnNamesForModelKind(modelKind)[0] || "";
      }

      function isPredictionColumnForModelKind(columnName, modelKind) {
        return predictionColumnNamesForModelKind(modelKind).includes(String(columnName || ""));
      }

      function activePredictionSourceForModelKind(modelKind) {
        if (modelKind === "glm") return activeModelSource("glm_predictions");
        if (modelKind === "gbm") return activeModelSource("gbm_predictions");
        return null;
      }

      function setExpectedPredictionSelectionForModelKind(modelKind, preferredColumn = "") {
        const predictionSource = activePredictionSourceForModelKind(modelKind);
        if (!predictionSource?.id) return false;
        const candidates = predictionColumnNamesForModelKind(modelKind);
        const ordered = [
          ...(isPredictionColumnForModelKind(preferredColumn, modelKind) ? [String(preferredColumn)] : []),
          ...candidates,
        ].filter((name, index, values) => name && values.indexOf(name) === index);
        return ordered.some((predictionColumn) => setExpectedSelection(predictionColumn, predictionSource.id, { allowAnySource: false }));
      }

      function expectedSelectionsSnapshot() {
        return expectedSelections().map((selection) => ({ ...selection }));
      }

      function restoreExpectedSelectionsAfterModelMutation(selections, modelKind = "") {
        const resolved = [];
        const activePredictionSource = activePredictionSourceForModelKind(modelKind);
        for (const selection of selections) {
          if (modelKind && selection?.metricKind === "prediction" && isPredictionColumnForModelKind(selection.value, modelKind)) {
            if (activePredictionSource?.id) {
              resolved.push({ ...selection, sourceId: activePredictionSource.id });
            }
          } else {
            resolved.push(selection);
          }
        }
        return setExpectedSelections(resolved, { allowAnySource: true });
      }

      function actualSelectionSourceId() {
        return el("actualNumerator").selectedOptions[0]?.dataset.sourceId || "";
      }

      function syncActualSourceFromSelection() {
        const option = el("actualNumerator").selectedOptions[0];
        if (option?.dataset.metricKind === "prediction") return false;
        const targetSource = actualSelectionSourceId();
        if (!targetSource || targetSource === state.source) return false;
        state.source = targetSource;
        invalidateLineBarDateBucketSuggestion();
        return true;
      }

      function syncExpectedSourceFromSelection({ expectedValue = "", expectedSource = "", expectedSelections: nextSelections = null } = {}) {
        const selections = nextSelections ? normaliseExpectedSelections(nextSelections, { allowAnySource: true }) : expectedSelections();
        const firstSelection = selections[0] || null;
        const targetSource = firstSelection?.sourceId || expectedSource || expectedSelectionSourceId();
        if (firstSelection?.metricKind === "prediction" || (!nextSelections && selectedExpectedIsPrediction())) return false;
        if (!targetSource || targetSource === state.source) return false;
        const selectedExpected = firstSelection?.value || expectedValue || el("expectedNumerator").value;
        state.source = targetSource;
        invalidateLineBarDateBucketSuggestion();
        syncControlsForSourceChange({
          expectedSelections: selections.length ? selections : [{ value: selectedExpected, sourceId: targetSource }],
        });
        refreshMetricSummary();
        return true;
      }

      function syncControlsForSourceChange({ actualValue = "", actualSource = "", expectedValue = "", expectedSource = "", expectedSelections: previousExpectedSelections = null } = {}) {
        const previousActual = actualValue || el("actualNumerator").value;
        const previousActualSource = actualSource || actualSelectionSourceId();
        const previousExpected = expectedValue || el("expectedNumerator").value;
        const previousExpectedSource = expectedSource || expectedSelectionSourceId();
        const expectedSnapshot = previousExpectedSelections || expectedSelectionsSnapshot();
        if (!expectedSnapshot.length && previousExpected) {
          expectedSnapshot.push({ value: previousExpected, sourceId: previousExpectedSource });
        }
        const previousDenominator = el("denominator").value;
        syncLineBarXFallback();
        fillMetricSelect(el("actualNumerator"));
        if (previousActual && !setActualSelection(previousActual, previousActualSource)) {
          el("actualNumerator").value = numericColumnExists(previousActual) ? previousActual : numericColumns()[0]?.name || "";
        }
        fillMetricSelect(el("expectedNumerator"), true);
        fillDenominatorSelect(el("denominator"));
        setExpectedSelections(expectedSnapshot, { allowAnySource: true });
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
        const previousExpectedSelections = expectedSelectionsSnapshot();
        const previousDenominator = el("denominator").value;
        state.schema = await api("/api/schema");
        state.datasetViewerColumnCount = null;
        renderSidebarVersion();
        const modelKind = String(options?.modelKind || "");
        if (preferredSource) state.source = preferredSource;
        state.x = previousX;
        state.xSource = previousXSource;
        if (modelKind && isPredictionColumnForModelKind(previousX, modelKind)) {
          const predictionSource = activePredictionSourceForModelKind(modelKind);
          if (predictionSource?.id) {
            state.xSource = predictionSource.id;
            if (!lineBarColumnExists(state.x, predictionSource.id)) {
              const fallbackPredictionColumn = predictionColumnNameForModelKind(modelKind);
              if (lineBarColumnExists(fallbackPredictionColumn, predictionSource.id)) state.x = fallbackPredictionColumn;
            }
          }
        }
        syncLineBarXFallback();
        fillMetricSelect(el("actualNumerator"));
        fillMetricSelect(el("expectedNumerator"), true);
        fillDenominatorSelect(el("denominator"));
        if (!setActualSelection(previousActual, previousActualSource)) {
          el("actualNumerator").value = numericColumnExists(previousActual) ? previousActual : numericColumns()[0]?.name || "";
        }
        restoreExpectedSelectionsAfterModelMutation(previousExpectedSelections, modelKind);
        el("denominator").value = numericColumnExists(previousDenominator) ? previousDenominator : "__none__";
        syncLineBarXFallback();
        lineBarTool.renderExpectedNumerators();
        lineBarTool.renderFeatures();
        lineBarTool.updateAxisControls();
        syncKpiSelectionFromMetrics();
        renderKpis();
        renderFavourites();
        await refreshMetricSummary({ force: true });
      }

      async function reloadSchemaAfterSpecsSave() {
        const previousFilterSignature = savedFilterSpecSignature();
        const previousSavedFilterSelection = savedFilterSelectionSnapshot();
        const previousCollapsedSavedFilterThemes = new Set(state.collapsedSavedFilterThemes);
        const previousSavedFilterThemesInitialised = state.savedFilterThemesInitialised;
        const previousCollapsedKpiGroups = new Set(state.collapsedKpiGroups);
        const previousKpiGroupsInitialised = state.kpiGroupsInitialised;
        state.schema = await api("/api/schema");
        state.datasetViewerColumnCount = null;
        renderSidebarVersion();
        const filtersUnchanged = previousFilterSignature === savedFilterSpecSignature(state.schema.filters || []);
        clearToolCaches({ preserve: ["specs"] });
        renderDatasetMeta(schemaFileMeta(), datasetGbmCount, datasetGlmCount);
        await refreshFilterRowCountMeta();
        state.collapsedSavedFilterThemes = previousCollapsedSavedFilterThemes;
        state.savedFilterThemesInitialised = previousSavedFilterThemesInitialised;
        state.collapsedKpiGroups = previousCollapsedKpiGroups;
        state.kpiGroupsInitialised = previousKpiGroupsInitialised;
        renderSavedFilters();
        if (filtersUnchanged) restoreSavedFilterSelection(previousSavedFilterSelection);
        renderKpis();
        renderFavourites();
        await refreshFavourites();
        syncKpiSelectionFromMetrics();
        lineBarTool.renderExpectedNumerators();
        lineBarTool.renderFeatures();
        lineBarTool.updateAxisControls();
        glmTool.syncSidebarFromSchema();
        gbmTool.syncSidebarFromSchema();
        await refreshMetricSummary({ force: true });
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
            datasetNumericColumnExists(kpi.actual) &&
            (kpi.denominator === "__none__" || datasetNumericColumnExists(kpi.denominator)) &&
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
        const meta = el("kpiSelectedMeta");
        if (meta) meta.textContent = kpi ? kpi.name : "";
      }

      function syncKpiSelectionFromMetrics() {
        setActiveKpiState(selectedKpiForCurrentMetric());
        syncKpiActiveRows();
      }

      function syncKpiActiveRows() {
        const list = el("kpiSelect");
        if (!list) return;
        list.querySelectorAll(".kpi-option").forEach((button) => {
          const active = button.dataset.kpiKey === state.activeKpiKey;
          button.classList.toggle("active", active);
          button.setAttribute("aria-selected", String(active));
        });
      }

      function favouriteScope(favourite) {
        const scope = String(favourite?.view?.scope || "").trim();
        return FAVOURITE_SCOPES.has(scope) ? scope : DEFAULT_FAVOURITE_SCOPE;
      }

      function favouriteScopeLabel(scope) {
        return FAVOURITE_SCOPE_LABELS[scope] || FAVOURITE_SCOPE_LABELS[DEFAULT_FAVOURITE_SCOPE];
      }

      function lineBarCurrentViewScopeLabel() {
        return state.tool === "line_bar" && state.view === "table" ? "Table view" : FAVOURITE_SCOPE_LABELS.line_bar_view;
      }

      function favouriteTypeLabel(favourite) {
        const scope = favouriteScope(favourite);
        if (scope === "line_bar_view" && favourite?.view?.view === "table") return "Table view";
        return favouriteScopeLabel(scope);
      }

      function favouriteScopeIncludesChange(scope, change) {
        if (change === "metrics") return true;
        if (change === "filter") return scope === "metrics_filter" || scope === "line_bar_view" || scope === "map_view";
        if (change === "line_bar_view") return scope === "line_bar_view";
        if (change === "map_view") return scope === "map_view";
        return false;
      }

      function favouriteById(id) {
        return lineBarFavourites.find((favourite) => favourite.id === id) || null;
      }

      function activeSavedFavourite() {
        return favouriteById(state.activeLineBarFavouriteId);
      }

      function clearActiveFavouriteSelectionForScope(change) {
        const favourite = activeSavedFavourite();
        if (!favourite || !favouriteScopeIncludesChange(favouriteScope(favourite), change)) return false;
        state.activeLineBarFavouriteId = "";
        renderFavourites();
        return true;
      }

      function favouriteByStartupKey(key) {
        const target = String(key || "").trim();
        if (!target) return null;
        const byId = favouriteById(target);
        if (byId) return byId;
        const folded = target.toLowerCase();
        return lineBarFavourites.find((favourite) => String(favourite.name || "").toLowerCase() === folded) || null;
      }

      function favouriteValidationErrors(favourite) {
        const errors = favourite?.validation?.errors;
        return Array.isArray(errors) ? errors.filter(Boolean) : [];
      }

      function favouriteValidationWarnings(favourite) {
        const warnings = favourite?.validation?.warnings;
        return Array.isArray(warnings) ? warnings.filter(Boolean) : [];
      }

      function favouriteMessage(favourite) {
        return [...favouriteValidationErrors(favourite), ...favouriteValidationWarnings(favourite)].join(" ");
      }

      function syncFavouriteHeaderMeta() {
        const meta = el("favouritesSelectedMeta");
        if (!meta) return;
        const favourite = activeSavedFavourite();
        meta.textContent = favourite ? favourite.name || "" : "";
      }

      function syncFavouriteActiveRows() {
        const list = el("favouritesSelect");
        if (!list) return;
        list.querySelectorAll(".saved-favourite-option").forEach((button) => {
          const active = button.dataset.favouriteId === state.activeLineBarFavouriteId;
          button.classList.toggle("active", active);
          button.setAttribute("aria-selected", String(active));
        });
        syncFavouriteHeaderMeta();
      }

      function renderFavourites() {
        const list = el("favouritesSelect");
        if (!list) return;
        const favouritesLoading = !lineBarFavouritesLoaded && !favouriteLoadError;
        list.toggleAttribute("aria-busy", favouritesLoading);
        list.innerHTML = "";
        if (favouriteLoadError) {
          const message = document.createElement("div");
          message.className = "favourites-list-message";
          message.textContent = favouriteLoadError;
          list.append(message);
        }
        if (!lineBarFavourites.length) {
          if (!favouriteLoadError && lineBarFavouritesLoaded) {
            const empty = document.createElement("div");
            empty.className = "favourites-list-message";
            empty.textContent = "No favourites";
            list.append(empty);
          }
          syncFavouriteHeaderMeta();
          syncFavouriteActionButtons();
          return;
        }
        for (const favourite of lineBarFavourites) {
          const scope = favouriteScope(favourite);
          const invalid = favouriteValidationErrors(favourite).length > 0;
          const active = favourite.id === state.activeLineBarFavouriteId;
          const button = document.createElement("button");
          button.type = "button";
          button.className = `feature favourite-option saved-favourite-option${active ? " active" : ""}${invalid ? " favourite-option-invalid" : ""}`;
          button.dataset.favouriteId = favourite.id;
          button.dataset.favouriteScope = scope;
          button.setAttribute("role", "option");
          button.setAttribute("aria-selected", String(active));
          if (favouriteMessage(favourite)) button.title = favouriteMessage(favourite);
          const suffix = invalid ? " (invalid)" : "";
          button.innerHTML = `<span class="saved-filter-name">${escapeHtml(String(favourite.name || "") + suffix)}</span><span class="favourite-detail">${escapeHtml(favouriteTypeLabel(favourite))}</span>`;
          button.addEventListener("click", () => applySavedFavourite(favourite, { refresh: true }));
          list.append(button);
        }
        syncFavouriteHeaderMeta();
        syncFavouriteActionButtons();
      }

      function renderKpis() {
        const list = el("kpiSelect");
        if (!list) return;
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

      function applyMetricValues(actualValue, actualSource, denominatorValue) {
        const previousActual = el("actualNumerator").value;
        const previousActualSource = actualSelectionSourceId();
        const previousDenominator = normaliseKpiDenominator(el("denominator").value);
        const resolvedActualSource = resolveFavouriteSourceId(actualValue, actualSource || state.source || "dataset");
        if (!setActualSelection(actualValue, resolvedActualSource)) {
          if (!setActualSelection(actualValue)) chooseFirstActualSelection();
        }
        const selectedActual = el("actualNumerator").value;
        const selectedActualSource = actualSelectionSourceId();
        const sourceChanged = syncActualSourceFromSelection();
        if (sourceChanged) {
          syncControlsForSourceChange({ actualValue: selectedActual, actualSource: selectedActualSource });
        }
        const nextDenominator = normaliseKpiDenominator(denominatorValue);
        el("denominator").value = numericColumnExists(nextDenominator) ? nextDenominator : "__none__";
        syncKpiSelectionFromMetrics();
        return previousActual !== el("actualNumerator").value
          || previousActualSource !== actualSelectionSourceId()
          || previousDenominator !== normaliseKpiDenominator(el("denominator").value);
      }

      function selectKpi(kpi) {
        const changed = applyMetricValues(kpi.actual, "dataset", kpi.denominator);
        setActiveKpiState(kpi);
        syncKpiActiveRows();
        if (changed) clearActiveFavouriteSelectionForScope("metrics");
        refreshMetricSummary();
        if (changed) {
          refreshActiveToolForMetricChange();
        }
      }

      function applyFavouriteMetricState(favourite) {
        const view = favourite?.view || {};
        const actual = view.actual && typeof view.actual === "object" ? view.actual : {};
        return applyMetricValues(actual.value, actual.sourceId || view.source || "dataset", view.denominator || "__none__");
      }

      function applyFavouriteFilterState(view) {
        state.filterSelectionMode = String(view.filterSelectionMode || "grouped");
        state.filterOperator = String(view.filterOperator || "and");
        state.activeFilter = String(view.filter || "").trim();
        el("filterInput").value = state.activeFilter;
        setFilterSelectionMode(state.filterSelectionMode, { apply: false });
        syncFilterOperatorControl();
        restoreSavedFilterRows(view.savedFilterRows);
        invalidateLineBarDateBucketSuggestion();
        clearProfileDetailCache();
        syncActiveFilterLabels();
      }

      function waitForFrame() {
        return new Promise((resolve) => requestAnimationFrame(resolve));
      }

      async function finishMapFavouriteRestore() {
        await waitForFrame();
        await waitForFrame();
        state.mapFavouriteRestoreInProgress = false;
      }

      async function applyMapFavouriteView(favourite) {
        if (!toolEnabled("uk_map")) {
          throw new Error("UK Mapping is not enabled for this app.");
        }
        const view = favourite?.view || {};
        state.activeLineBarFavouriteId = favourite?.id || "";
        applyFavouriteMetricState(favourite);
        applyFavouriteFilterState(view);
        state.mapFavouriteRestoreInProgress = true;
        try {
          ukMapTool.applyFavouriteState(view.map || {});
          renderFavourites();
          setTool("uk_map", false);
          beginFavouriteViewRestore();
          ukMapTool.showPendingRestore();
          const data = await refreshUkMap({ force: true });
          syncSidebarSummariesFromToolData(data);
        } finally {
          await finishMapFavouriteRestore();
        }
      }

      function applyMapFavouriteStateOnly(favourite) {
        if (!toolEnabled("uk_map")) {
          throw new Error("UK Mapping is not enabled for this app.");
        }
        const view = favourite?.view || {};
        state.activeLineBarFavouriteId = favourite?.id || "";
        applyFavouriteMetricState(favourite);
        applyFavouriteFilterState(view);
        ukMapTool.applyFavouriteState(view.map || {});
        renderFavourites();
      }

      async function refreshFavourites(options = {}) {
        try {
          const data = await api("/api/line-bar/favourites");
          lineBarFavourites = Array.isArray(data.favourites) ? data.favourites : [];
          lineBarFavouritesLoaded = true;
          favouriteLoadError = "";
          renderFavourites();
          if (options.renderPopover) renderFavouritePopover(favouritePopoverMode);
          return lineBarFavourites;
        } catch (error) {
          lineBarFavourites = [];
          lineBarFavouritesLoaded = false;
          favouriteLoadError = error.message || "Favourites unavailable";
          renderFavourites();
          return [];
        }
      }

      async function applySavedFavourite(favourite, options = {}) {
        if (!favourite) return "";
        const errors = favouriteValidationErrors(favourite);
        if (errors.length) {
          const message = `Favourite "${favourite.name}" cannot be used. ${errors.join(" ")}`;
          setStatus(message, true);
          renderFavourites();
          return message;
        }
        const scope = favouriteScope(favourite);
        try {
          if (scope === "map_view") {
            await applyMapFavouriteView(favourite);
          } else if (scope === "line_bar_view") {
            if (toolEnabled("line_bar")) setTool("line_bar", false);
            await applyLineBarFavouriteView(favourite, options);
          } else {
            state.activeLineBarFavouriteId = favourite.id || "";
            const metricsChanged = applyFavouriteMetricState(favourite);
            if (scope === "metrics_filter") {
              applyFavouriteFilterState(favourite.view || {});
              await refreshFilterRowCountMeta();
            }
            renderFavourites();
            await refreshMetricSummary({ force: true });
            if (scope === "metrics_filter") {
              await refreshActiveTool({ force: true });
            } else if (metricsChanged) {
              refreshActiveToolForMetricChange();
            }
          }
          const warnings = favouriteValidationWarnings(favourite);
          const message = warnings.length ? warnings.join(" ") : "";
          setStatus(message, Boolean(message));
          renderFavourites();
          return "";
        } catch (error) {
          const message = error.message || "Favourite could not be restored.";
          setStatus(message, true);
          renderFavourites();
          return message;
        }
      }

      function startupFavouriteForRestore() {
        const target = String(requestedDefault("line_bar_favourite") || "").trim();
        if (target) {
          return {
            explicit: true,
            target,
            favourite: favouriteByStartupKey(target),
          };
        }
        return {
          explicit: false,
          target: "",
          favourite: lineBarFavourites[0] || null,
        };
      }

      function startupToolForFavourite(favourite, fallbackTool) {
        const scope = favouriteScope(favourite);
        if (scope === "line_bar_view" && toolEnabled("line_bar")) return "line_bar";
        if (scope === "map_view" && toolEnabled("uk_map")) return "uk_map";
        return fallbackTool;
      }

      async function applyStartupFavouriteState() {
        if (favouriteStartupApplied) {
          return { applied: false, favourite: null, filterApplied: false, message: "", statusError: false };
        }
        favouriteStartupApplied = true;
        if (!lineBarFavouritesLoaded) await refreshFavourites();
        const { explicit, target, favourite } = startupFavouriteForRestore();
        if (!favourite) {
          return {
            applied: false,
            favourite: null,
            filterApplied: false,
            message: explicit ? `Favourite not found: ${target}` : "",
            statusError: explicit,
          };
        }
        const errors = favouriteValidationErrors(favourite);
        if (errors.length) {
          return {
            applied: false,
            favourite,
            filterApplied: false,
            message: `Favourite "${favourite.name}" cannot be used. ${errors.join(" ")}`,
            statusError: true,
          };
        }
        try {
          const scope = favouriteScope(favourite);
          if (scope === "map_view") {
            applyMapFavouriteStateOnly(favourite);
          } else if (scope === "line_bar_view") {
            if (!toolEnabled("line_bar")) {
              throw new Error("Line/Bar is not enabled for this app.");
            }
            await applyLineBarFavouriteView(favourite, { refresh: false });
          } else {
            state.activeLineBarFavouriteId = favourite.id || "";
            applyFavouriteMetricState(favourite);
            if (scope === "metrics_filter") {
              applyFavouriteFilterState(favourite.view || {});
            }
            renderFavourites();
          }
          const warnings = favouriteValidationWarnings(favourite);
          const message = warnings.length ? warnings.join(" ") : "";
          return {
            applied: true,
            favourite,
            filterApplied: Boolean(state.activeFilter),
            message,
            statusError: Boolean(message),
          };
        } catch (error) {
          return {
            applied: false,
            favourite,
            filterApplied: false,
            message: error.message || "Favourite could not be restored.",
            statusError: true,
          };
        }
      }

      function syncFavouriteActionButtons() {
        const addButton = el("sidebarFavouriteAddBtn");
        const menuButton = el("sidebarFavouriteMenuBtn");
        const canAdd = toolSupportsFavouriteAdd();
        if (addButton) {
          addButton.classList.toggle("hidden", !canAdd);
          addButton.toggleAttribute("aria-hidden", !canAdd);
          addButton.disabled = !canAdd || !lineBarFavouritesLoaded;
          if (!canAdd && addButton.contains(document.activeElement)) document.activeElement?.blur?.();
        }
        if (menuButton) menuButton.disabled = !lineBarFavouritesLoaded;
        const popover = el("sidebarFavouritePopover");
        if (!canAdd && popover && !popover.hidden && favouritePopoverMode === "add") closeFavouritePopover();
      }

      function closeFavouritePopover() {
        const popover = el("sidebarFavouritePopover");
        if (!popover) return;
        popover.hidden = true;
        popover.innerHTML = "";
        selectedFavouriteManageId = "";
      }

      function placeFavouritePopover() {
        const control = el("sidebarFavouritesControls");
        const popover = el("sidebarFavouritePopover");
        if (!control || !popover) return;
        const rect = control.getBoundingClientRect();
        const top = Math.max(8, Math.round(rect.bottom + 6));
        const popoverWidth = Math.min(520, Math.max(0, window.innerWidth - 32));
        const left = Math.max(8, Math.min(Math.round(rect.left), Math.round(window.innerWidth - popoverWidth - 8)));
        const maxHeight = Math.max(160, Math.round(window.innerHeight - top - 12));
        popover.style.setProperty("--line-bar-favourite-popover-top", `${top}px`);
        popover.style.setProperty("--line-bar-favourite-popover-left", `${left}px`);
        popover.style.setProperty("--line-bar-favourite-popover-max-height", `${maxHeight}px`);
      }

      function openFavouritePopover(mode = "manage") {
        favouritePopoverMode = mode;
        if (mode === "manage") selectedFavouriteManageId = "";
        renderFavouritePopover(mode);
        const popover = el("sidebarFavouritePopover");
        if (!popover) return;
        placeFavouritePopover();
        popover.hidden = false;
        const input = popover.querySelector("input");
        if (mode === "add" && input) input.focus();
      }

      function favouriteManageRowById(id) {
        const popover = el("sidebarFavouritePopover");
        if (!popover || !id) return null;
        return [...popover.querySelectorAll(".line-bar-favourite-row")]
          .find((row) => row.dataset.favouriteId === id) || null;
      }

      function selectedFavouriteManageIndex() {
        if (!selectedFavouriteManageId) return -1;
        return lineBarFavourites.findIndex((favourite) => favourite.id === selectedFavouriteManageId);
      }

      function focusSelectedFavouriteManageInput() {
        const input = favouriteManageRowById(selectedFavouriteManageId)?.querySelector(".line-bar-favourite-name-input");
        input?.focus();
      }

      function focusSelectedFavouriteManageInputAfterRender() {
        requestAnimationFrame(() => focusSelectedFavouriteManageInput());
      }

      function queueFavouriteOrderSave(ids) {
        pendingFavouriteOrderSave = { ids: [...ids], sequence: favouriteOrderSaveSequence + 1 };
        favouriteOrderSaveSequence = pendingFavouriteOrderSave.sequence;
        void flushFavouriteOrderSaves();
      }

      async function flushFavouriteOrderSaves() {
        if (favouriteOrderSaveInFlight) return;
        favouriteOrderSaveInFlight = true;
        try {
          while (pendingFavouriteOrderSave) {
            const save = pendingFavouriteOrderSave;
            pendingFavouriteOrderSave = null;
            try {
              await api("/api/line-bar/favourites/order", {
                method: "PUT",
                body: JSON.stringify({ ids: save.ids }),
              });
            } catch (error) {
              if (save.sequence !== favouriteOrderSaveSequence || pendingFavouriteOrderSave) continue;
              const popover = el("sidebarFavouritePopover");
              const showManageError = Boolean(popover && !popover.hidden && favouritePopoverMode === "manage");
              await refreshFavourites({ renderPopover: showManageError });
              if (showManageError) {
                renderFavouritePopover("manage", error.message || "Favourite order could not be saved", true);
                focusSelectedFavouriteManageInputAfterRender();
              }
            }
          }
        } finally {
          favouriteOrderSaveInFlight = false;
          if (pendingFavouriteOrderSave) void flushFavouriteOrderSaves();
        }
      }

      function updateFavouriteMoveControls() {
        const popover = el("sidebarFavouritePopover");
        if (!popover) return;
        if (selectedFavouriteManageIndex() < 0) selectedFavouriteManageId = "";
        const selectedIndex = selectedFavouriteManageIndex();
        popover.querySelectorAll(".line-bar-favourite-row").forEach((row) => {
          const selected = Boolean(selectedFavouriteManageId) && row.dataset.favouriteId === selectedFavouriteManageId;
          row.classList.toggle("selected", selected);
          row.setAttribute("aria-selected", selected ? "true" : "false");
        });
        const upButton = popover.querySelector('[data-favourite-action="move-up"]');
        const downButton = popover.querySelector('[data-favourite-action="move-down"]');
        if (upButton) upButton.disabled = selectedIndex <= 0;
        if (downButton) downButton.disabled = selectedIndex < 0 || selectedIndex >= lineBarFavourites.length - 1;
      }

      function selectFavouriteManageRow(row) {
        const id = row?.dataset.favouriteId || "";
        if (!id) return;
        selectedFavouriteManageId = id;
        updateFavouriteMoveControls();
      }

      function updateFavouriteRenameButton(row) {
        const input = row?.querySelector(".line-bar-favourite-name-input");
        const button = row?.querySelector('[data-favourite-action="rename"]');
        if (!input || !button) return;
        const original = String(row.dataset.originalName || "").trim();
        const current = String(input.value || "").trim();
        const dirty = Boolean(current) && current !== original;
        button.disabled = !dirty;
        button.classList.toggle("active", dirty);
      }

      function updateFavouriteAddButton() {
        const input = el("sidebarFavouriteNameInput");
        const button = el("sidebarFavouritePopover")?.querySelector('[data-favourite-action="save-add"]');
        if (!input || !button) return;
        const ready = Boolean(String(input.value || "").trim());
        button.disabled = !ready;
        button.classList.toggle("active", ready);
      }

      function toolSupportsFavouriteAdd(tool = state.tool) {
        return tool === "line_bar" || tool === "uk_map";
      }

      function defaultFavouriteAddScope() {
        return state.tool === "uk_map" ? "map_view" : DEFAULT_FAVOURITE_SCOPE;
      }

      function favouriteScopeOptionsForAdd() {
        if (defaultFavouriteAddScope() === "map_view") return MAP_FAVOURITE_SCOPE_OPTIONS;
        return LINE_BAR_FAVOURITE_SCOPE_OPTIONS.map(([scope, label]) => (
          scope === "line_bar_view" ? [scope, lineBarCurrentViewScopeLabel()] : [scope, label]
        ));
      }

      function renderFavouriteScopeOptions(selectedScope = defaultFavouriteAddScope()) {
        return favouriteScopeOptionsForAdd()
          .map(([scope, label]) => `
            <label class="sidebar-favourite-scope-option${scope === selectedScope ? " active" : ""}" data-favourite-scope-option="${escapeHtml(scope)}">
              <input class="sidebar-favourite-scope-radio" type="radio" name="sidebarFavouriteScope" value="${escapeHtml(scope)}"${scope === selectedScope ? " checked" : ""} />
              <span>${escapeHtml(label)}</span>
            </label>
          `)
          .join("");
      }

      function selectedFavouriteAddScope() {
        const popover = el("sidebarFavouritePopover");
        const fallback = defaultFavouriteAddScope();
        const scope = String(popover?.dataset.favouriteScope || fallback);
        return FAVOURITE_SCOPES.has(scope) ? scope : fallback;
      }

      function setFavouriteAddScope(scope) {
        const popover = el("sidebarFavouritePopover");
        if (!popover) return;
        const fallback = defaultFavouriteAddScope();
        const nextScope = FAVOURITE_SCOPES.has(scope) ? scope : fallback;
        popover.dataset.favouriteScope = nextScope;
        popover.querySelectorAll("[data-favourite-scope-option]").forEach((option) => {
          const active = option.dataset.favouriteScopeOption === nextScope;
          option.classList.toggle("active", active);
          const input = option.querySelector('input[type="radio"]');
          if (input) input.checked = active;
        });
      }

      function renderFavouritePopover(mode = "manage", message = "", isError = false) {
        const popover = el("sidebarFavouritePopover");
        if (!popover) return;
        popover.classList.toggle("line-bar-favourite-popover--manage", mode !== "add");
        if (mode === "add") {
          const defaultScope = defaultFavouriteAddScope();
          popover.dataset.favouriteScope = defaultScope;
          popover.innerHTML = `
            <div class="line-bar-favourite-popover-head sidebar-favourite-popover-head">
              <button class="line-bar-favourite-action-button line-bar-favourite-save" type="button" data-favourite-action="save-add" aria-label="Save favourite" title="Save favourite" disabled>&#10003;</button>
              <input id="sidebarFavouriteNameInput" class="line-bar-favourite-name-input" type="text" maxlength="120" placeholder="Name" />
            </div>
            <div class="sidebar-favourite-scope-row" role="radiogroup" aria-label="Favourite scope">
              <div class="sidebar-favourite-scope-title">Scope</div>
              <div class="sidebar-favourite-scope-options">
                ${renderFavouriteScopeOptions(defaultScope)}
              </div>
            </div>
            <div class="line-bar-favourite-popover-message${isError ? " error" : ""}">${escapeHtml(message)}</div>
          `;
          updateFavouriteAddButton();
          placeFavouritePopover();
          return;
        }
        if (selectedFavouriteManageId && !favouriteById(selectedFavouriteManageId)) selectedFavouriteManageId = "";
        const rows = lineBarFavourites.map((favourite) => `
          <div class="line-bar-favourite-row${favourite.id === selectedFavouriteManageId ? " selected" : ""}" data-favourite-id="${escapeHtml(favourite.id)}" data-original-name="${escapeHtml(favourite.name)}" aria-selected="${favourite.id === selectedFavouriteManageId ? "true" : "false"}">
            <button class="line-bar-favourite-action-button line-bar-favourite-rename-button" type="button" data-favourite-action="rename" aria-label="Save name change" title="Save name change" disabled>&#10003;</button>
            <input class="line-bar-favourite-name-input" type="text" maxlength="120" value="${escapeHtml(favourite.name)}" aria-label="Favourite name" />
            <button class="line-bar-favourite-action-button line-bar-favourite-delete-button" type="button" data-favourite-action="delete" aria-label="Delete ${escapeHtml(favourite.name)}" title="Delete">&times;</button>
            <div class="line-bar-favourite-row-scope">${escapeHtml(favouriteScopeLabel(favouriteScope(favourite)))}</div>
          </div>
          ${favouriteMessage(favourite) ? `<div class="line-bar-favourite-row-message">${escapeHtml(favouriteMessage(favourite))}</div>` : ""}
        `).join("");
        popover.innerHTML = `
          <div class="line-bar-favourite-move-controls" aria-label="Move selected favourite">
            <button class="line-bar-favourite-action-button" type="button" data-favourite-action="move-up" aria-label="Move selected favourite up" title="Move selected favourite up" disabled>&uarr;</button>
            <button class="line-bar-favourite-action-button" type="button" data-favourite-action="move-down" aria-label="Move selected favourite down" title="Move selected favourite down" disabled>&darr;</button>
          </div>
          <div class="line-bar-favourite-popover-list">${rows || '<div class="line-bar-favourite-empty">No favourites</div>'}</div>
          <div class="line-bar-favourite-popover-message${isError ? " error" : ""}">${escapeHtml(message)}</div>
        `;
        popover.querySelectorAll(".line-bar-favourite-row").forEach(updateFavouriteRenameButton);
        updateFavouriteMoveControls();
        placeFavouritePopover();
      }

      function favouritePopoverRow(button) {
        return button.closest(".line-bar-favourite-row");
      }

      async function handleFavouritePopoverAction(action, button) {
        try {
          if (action === "save-add") {
            const input = el("sidebarFavouriteNameInput");
            const scope = selectedFavouriteAddScope();
            const name = input?.value.trim() || "";
            if (!name) return;
            const view = captureLineBarFavouriteView({ scope });
            const data = await api("/api/line-bar/favourites", { method: "POST", body: JSON.stringify({ name, view }) });
            state.activeLineBarFavouriteId = data?.favourite?.id || "";
            await refreshFavourites();
            closeFavouritePopover();
            return;
          }
          if (action === "move-up" || action === "move-down") {
            const favouriteId = selectedFavouriteManageId;
            const index = lineBarFavourites.findIndex((favourite) => favourite.id === favouriteId);
            const nextIndex = action === "move-up" ? index - 1 : index + 1;
            if (index < 0 || nextIndex < 0 || nextIndex >= lineBarFavourites.length) return;
            const ordered = [...lineBarFavourites];
            [ordered[index], ordered[nextIndex]] = [ordered[nextIndex], ordered[index]];
            lineBarFavourites = ordered;
            selectedFavouriteManageId = favouriteId;
            renderFavourites();
            renderFavouritePopover("manage");
            focusSelectedFavouriteManageInputAfterRender();
            queueFavouriteOrderSave(ordered.map((favourite) => favourite.id));
            return;
          }
          const row = favouritePopoverRow(button);
          const favouriteId = row?.dataset.favouriteId || "";
          if (!favouriteId) return;
          if (action === "rename") {
            const name = row.querySelector("input")?.value.trim() || "";
            if (!name || button.disabled) return;
            selectedFavouriteManageId = favouriteId;
            await api(`/api/line-bar/favourites/${encodeURIComponent(favouriteId)}`, { method: "PATCH", body: JSON.stringify({ name }) });
            await refreshFavourites({ renderPopover: true });
            focusSelectedFavouriteManageInput();
            return;
          }
          if (action === "delete") {
            await api(`/api/line-bar/favourites/${encodeURIComponent(favouriteId)}`, { method: "DELETE" });
            if (state.activeLineBarFavouriteId === favouriteId) state.activeLineBarFavouriteId = "";
            if (selectedFavouriteManageId === favouriteId) selectedFavouriteManageId = "";
            await refreshFavourites({ renderPopover: true });
          }
        } catch (error) {
          renderFavouritePopover(favouritePopoverMode, error.message || "Favourite action failed", true);
        }
      }

      function bindFavouriteControls() {
        el("sidebarFavouriteAddBtn")?.addEventListener("click", () => {
          if (!toolSupportsFavouriteAdd()) return;
          setOpenSidebarSection("favourites");
          openFavouritePopover("add");
        });
        el("sidebarFavouriteMenuBtn")?.addEventListener("click", () => {
          setOpenSidebarSection("favourites");
          const popover = el("sidebarFavouritePopover");
          if (popover && !popover.hidden && favouritePopoverMode === "manage") {
            closeFavouritePopover();
            return;
          }
          openFavouritePopover("manage");
        });
        el("sidebarFavouritePopover")?.addEventListener("click", (event) => {
          event.stopPropagation();
          const scopeButton = event.target.closest("[data-favourite-scope-option]");
          if (scopeButton) {
            setFavouriteAddScope(scopeButton.dataset.favouriteScopeOption || defaultFavouriteAddScope());
            return;
          }
          const button = event.target.closest("button[data-favourite-action]");
          if (!button) return;
          handleFavouritePopoverAction(button.dataset.favouriteAction, button);
        });
        el("sidebarFavouritePopover")?.addEventListener("focusin", (event) => {
          const row = event.target.closest?.(".line-bar-favourite-row");
          if (!row || !event.target.classList?.contains("line-bar-favourite-name-input")) return;
          selectFavouriteManageRow(row);
        });
        el("sidebarFavouritePopover")?.addEventListener("input", (event) => {
          if (event.target.id === "sidebarFavouriteNameInput") {
            updateFavouriteAddButton();
            return;
          }
          if (event.target.name === "sidebarFavouriteScope") {
            setFavouriteAddScope(event.target.value || defaultFavouriteAddScope());
            return;
          }
          const row = event.target.closest(".line-bar-favourite-row");
          if (!row) return;
          selectFavouriteManageRow(row);
          updateFavouriteRenameButton(row);
        });
        el("sidebarFavouritePopover")?.addEventListener("keydown", (event) => {
          if (event.key === "Escape") {
            closeFavouritePopover();
          }
          if (event.key === "Enter" && favouritePopoverMode === "add" && event.target.id === "sidebarFavouriteNameInput") {
            const saveButton = el("sidebarFavouritePopover").querySelector('[data-favourite-action="save-add"]');
            saveButton?.click();
          }
        });
        document.addEventListener("click", (event) => {
          const popover = el("sidebarFavouritePopover");
          if (!popover || popover.hidden) return;
          const addButton = el("sidebarFavouriteAddBtn");
          const menuButton = el("sidebarFavouriteMenuBtn");
          if (popover.contains(event.target) || addButton?.contains(event.target) || menuButton?.contains(event.target)) return;
          closeFavouritePopover();
        });
        window.addEventListener("resize", () => {
          const popover = el("sidebarFavouritePopover");
          if (!popover || popover.hidden) return;
          placeFavouritePopover();
        });
        el("appSidebar")?.addEventListener("scroll", () => {
          const popover = el("sidebarFavouritePopover");
          if (!popover || popover.hidden) return;
          placeFavouritePopover();
        });
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

      function syncSavedFilterThemeSelectionState() {
        const list = el("savedFilterSelect");
        const selectedThemes = new Set(
          Array.from(list.querySelectorAll('.saved-filter-option[aria-selected="true"]'))
            .map((button) => button.dataset.filterTheme || "General")
        );
        list.querySelectorAll(".saved-filter-theme").forEach((heading) => {
          heading.classList.toggle("saved-filter-theme--selected", selectedThemes.has(heading.dataset.filterTheme || "General"));
        });
      }

      function restoreSavedFilterSelection(selectedKeys) {
        if (!selectedKeys?.size) {
          syncSavedFilterThemeSelectionState();
          return;
        }
        el("savedFilterSelect").querySelectorAll(".saved-filter-option").forEach((button) => {
          const active = selectedKeys.has(savedFilterButtonKey(button));
          button.setAttribute("aria-selected", String(active));
          button.classList.toggle("active", active);
        });
        syncSavedFilterThemeSelectionState();
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
            syncSavedFilterThemeSelectionState();
            applySavedFilters();
          });
          list.append(button);
        }
        syncSavedFilterThemeSelectionState();
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
        syncSavedFilterThemeSelectionState();
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

      function currentKpiSnapshot() {
        const kpi = selectedKpiForCurrentMetric();
        return kpi
          ? {
              group: kpi.group,
              name: kpi.name,
              actual: kpi.actual,
              denominator: kpi.denominator,
            }
          : null;
      }

      function captureLineBarFavouriteView(options = {}) {
        const actualOption = el("actualNumerator").selectedOptions[0];
        const requestedScope = String(options.scope || DEFAULT_FAVOURITE_SCOPE);
        const scope = FAVOURITE_SCOPES.has(requestedScope) ? requestedScope : DEFAULT_FAVOURITE_SCOPE;
        const view = {
          version: 1,
          scope,
          source: state.source || "dataset",
          x: state.x || "",
          xSource: state.xSource || "",
          view: state.view === "table" ? "table" : "chart",
          sort: state.sort,
          lowGroup: state.lowGroup,
          labels: state.labels,
          bandWidth: state.bandWidth,
          quantileMode: state.quantileMode,
          dateBucket: state.dateBucket,
          transform: state.transform,
          sigma: state.sigma,
          partialDependence: state.partialDependence,
          featureSort: state.featureSort,
          expectedSort: state.expectedSort,
          actual: {
            value: el("actualNumerator").value,
            sourceId: actualOption?.dataset.sourceId || state.source || "dataset",
            metricKind: actualOption?.dataset.metricKind || "metric",
          },
          denominator: el("denominator").value || "__none__",
          expectedSelections: expectedSelectionsSnapshot(),
          kpi: currentKpiSnapshot(),
          filter: state.activeFilter || "",
          filterSelectionMode: state.filterSelectionMode,
          filterOperator: state.filterOperator,
          savedFilterRows: selectedSavedFilterRows(),
        };
        if (scope === "map_view") {
          view.map = ukMapTool.captureFavouriteState();
        }
        return view;
      }

      function syncFilterOperatorControl() {
        const group = document.querySelector('.segmented[data-control="filterOperator"]');
        group?.querySelectorAll("button").forEach((button) => {
          button.classList.toggle("active", button.dataset.value === state.filterOperator);
        });
      }

      function restoreSavedFilterRows(rows) {
        const selectedRows = Array.isArray(rows) ? rows.filter((row) => row && typeof row === "object") : [];
        const selectedThemes = new Set(selectedRows.map((row) => row.theme || "General"));
        selectedThemes.forEach((theme) => state.collapsedSavedFilterThemes.delete(theme));
        renderSavedFilters();
        const selectedKeys = new Set(selectedRows.map(savedFilterRowKey));
        el("savedFilterSelect").querySelectorAll(".saved-filter-option").forEach((button) => {
          const active = selectedKeys.has(savedFilterButtonKey(button));
          button.setAttribute("aria-selected", String(active));
          button.classList.toggle("active", active);
        });
        syncSavedFilterThemeSelectionState();
      }

      function setLineBarManualGroupingKeys() {
        const sourceId = state.xSource || state.source || "dataset";
        const feature = state.x || "";
        state.bandFeature = JSON.stringify([sourceId, feature]);
        state.dateBucketFeature = JSON.stringify([sourceId, feature, state.activeFilter || ""]);
        state.dateBucketManualKey = state.dateBucketFeature;
        state.bandSuggestionPendingKey = null;
        state.dateBucketSuggestionPendingKey = null;
      }

      async function applyLineBarFavouriteView(favourite, options = {}) {
        const validation = favourite?.validation || {};
        if (Array.isArray(validation.errors) && validation.errors.length) {
          throw new Error(validation.errors.join(" "));
        }
        const view = favourite?.view || {};
        state.activeLineBarFavouriteId = favourite?.id || "";
        state.source = resolveFavouriteSourceId("", view.source || "dataset") || "dataset";
        state.x = String(view.x || state.x || "");
        state.xSource = resolveFavouriteSourceId(state.x, view.xSource || state.source || "dataset") || state.source || "dataset";
        state.view = view.view === "table" ? "table" : "chart";
        state.sort = String(view.sort || "alpha");
        state.lowGroup = String(view.lowGroup || "0");
        state.labels = String(view.labels || "none");
        state.bandWidth = String(view.bandWidth ?? "0");
        state.quantileMode = view.quantileMode === "quantile" ? "quantile" : "off";
        state.dateBucket = String(view.dateBucket || "none");
        state.transform = String(view.transform || "none");
        state.sigma = String(view.sigma ?? "0");
        state.partialDependence = String(view.partialDependence || "none");
        state.featureSort = String(view.featureSort || "alpha");
        state.expectedSort = String(view.expectedSort || "alpha");
        state.filterSelectionMode = String(view.filterSelectionMode || "grouped");
        state.filterOperator = String(view.filterOperator || "and");
        state.activeFilter = String(view.filter || "").trim();
        el("filterInput").value = state.activeFilter;
        setFilterSelectionMode(state.filterSelectionMode, { apply: false });
        syncFilterOperatorControl();
        restoreSavedFilterRows(view.savedFilterRows);
        fillMetricSelect(el("actualNumerator"));
        fillMetricSelect(el("expectedNumerator"), true);
        fillDenominatorSelect(el("denominator"));
        const actual = view.actual && typeof view.actual === "object" ? view.actual : {};
        const actualSource = resolveFavouriteSourceId(actual.value, actual.sourceId || state.source || "dataset");
        if (!setActualSelection(actual.value, actualSource)) {
          chooseFirstActualSelection();
        }
        const expectedSelections = Array.isArray(view.expectedSelections) ? view.expectedSelections : [];
        setExpectedSelections(expectedSelections, { allowAnySource: true });
        const denominator = String(view.denominator || "__none__");
        el("denominator").value = numericColumnExists(denominator) ? denominator : "__none__";
        syncKpiSelectionFromMetrics();
        syncLineBarXFallback();
        setLineBarManualGroupingKeys();
        invalidateLineBarDateBucketSuggestion();
        setLineBarManualGroupingKeys();
        clearProfileDetailCache();
        syncActiveFilterLabels();
        lineBarTool.renderExpectedNumerators();
        lineBarTool.renderFeatures();
        lineBarTool.updateAxisControls();
        renderFavourites();
        if (options.refresh !== false) {
          beginFavouriteViewRestore();
          lineBarTool.showPendingRestore(state.view);
          const data = state.view === "table"
            ? await lineBarTool.refreshTable({ force: true, forceServer: true })
            : await refreshLineBar({ force: true });
          syncSidebarSummariesFromToolData(data);
        } else {
          lineBarTool.setView(state.view, { refresh: false });
        }
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
        const requestedExpected2 = requestedDefault("expected2");
        const requestedDenominator = requestedDefault("denominator");
        if (!setActualSelection(requestedActual, state.source)) {
          el("actualNumerator").value = numericColumnExists(requestedActual) ? requestedActual : numericColumns()[0]?.name || "";
        }
        if (!el("actualNumerator").value) chooseFirstActualSelection();
        setExpectedSelections([
          { value: requestedExpected, sourceId: "" },
          { value: requestedExpected2, sourceId: "" },
        ]);
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
          refreshMetricSummary();
          refreshActiveTool();
          refreshFilterRowCountMeta();
          return;
        }
        state.activeFilter = nextFilter;
        clearActiveFavouriteSelectionForScope("filter");
        invalidateLineBarDateBucketSuggestion();
        clearProfileDetailCache();
        syncActiveFilterLabels();
        refreshMetricSummary();
        refreshActiveTool();
        refreshFilterRowCountMeta();
      }

      function clearFilter() {
        el("filterInput").value = "";
        Array.from(el("savedFilterSelect").querySelectorAll(".saved-filter-option")).forEach((button) => {
          button.setAttribute("aria-selected", "false");
          button.classList.remove("active");
        });
        syncSavedFilterThemeSelectionState();
        if (state.activeFilter === "") {
          syncActiveFilterLabels();
          refreshMetricSummary();
          refreshActiveTool();
          refreshFilterRowCountMeta();
          return;
        }
        state.activeFilter = "";
        clearActiveFavouriteSelectionForScope("filter");
        invalidateLineBarDateBucketSuggestion();
        clearProfileDetailCache();
        syncActiveFilterLabels();
        refreshMetricSummary();
        refreshActiveTool();
        refreshFilterRowCountMeta();
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
        if (button) {
          button.disabled = true;
          button.textContent = "Stopping...";
        }
        setStatus("Stopping app...");
        try {
          await api("/api/shutdown", { method: "POST" });
          showStoppedOverlay();
        } catch (error) {
          if (button) {
            button.disabled = false;
            button.textContent = "Stop app";
          }
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
          setSidebarWidth(event.clientX - bounds.left, { hard: false });
        });
        function finishDrag(event) {
          if (!dragging) return;
          dragging = false;
          resizer.classList.remove("dragging");
          document.body.classList.remove("resizing-sidebar");
          window.getSelection()?.removeAllRanges();
          if (event.pointerId !== undefined) {
            try {
              resizer.releasePointerCapture(event.pointerId);
            } catch (_) {
            }
          }
          scheduleActiveToolResize({ hard: true });
        }
        resizer.addEventListener("pointerup", finishDrag);
        resizer.addEventListener("pointercancel", finishDrag);
      }

      function setSidebarWidth(rawWidth, { hard = true } = {}) {
        const viewportLimit = Math.max(260, window.innerWidth - 520);
        const width = Math.min(Math.max(rawWidth, 220), Math.min(560, viewportLimit));
        document.documentElement.style.setProperty("--sidebar-width", `${Math.round(width)}px`);
        scheduleActiveToolResize({ hard });
      }

      function setupChartControlsResize() {
        const visualArea = document.querySelector(".visual-area");
        const resizer = el("chartControlsResizer");

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
        const toggle = el("chartExpectedToggle");
        syncChartExpectedToggle();

        let dragging = false;
        let startY = 0;
        let startHeight = 0;
        let dragStartExpandedHeight = null;
        resizer.addEventListener("pointerdown", (event) => {
          event.preventDefault();
          dragging = true;
          startY = event.clientY;
          startHeight = firstPanel?.getBoundingClientRect().height || 0;
          dragStartExpandedHeight = controls?.classList.contains("chart-expected-collapsed")
            ? null
            : startHeight;
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
          if (controls?.classList.contains("chart-expected-collapsed")) {
            if (Number.isFinite(dragStartExpandedHeight) && dragStartExpandedHeight > 0) {
              chartFeatureControlsExpandedHeight = dragStartExpandedHeight;
            }
          } else {
            rememberExpandedChartFeatureControlsHeight();
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
        toggle?.addEventListener("click", () => toggleExpectedSideSection());
      }

      function applyStartupChartExpectedCollapse() {
        if (chartExpectedStartupCollapseApplied) return;
        chartExpectedStartupCollapseApplied = true;
        setChartFeatureControlsHeight(CHART_FEATURE_CONTROLS_HEIGHT_COLLAPSED);
      }

      function rememberExpandedChartFeatureControlsHeight() {
        const controls = document.querySelector(".chart-side-controls");
        if (controls?.classList.contains("chart-expected-collapsed")) return;
        const height = parseFloat(getComputedStyle(document.documentElement).getPropertyValue("--chart-feature-controls-height"));
        if (Number.isFinite(height) && height > 0) {
          chartFeatureControlsExpandedHeight = height;
        }
      }

      function defaultChartFeatureControlsHeight() {
        const controls = document.querySelector(".chart-side-controls");
        const availableHeight = controls?.getBoundingClientRect().height || window.innerHeight;
        const splitterHeight = el("chartControlHeightRow")?.getBoundingClientRect().height || 18;
        const gridGap = 8;
        const minFeaturePanelHeight = 96;
        const usableHeight = Math.max(minFeaturePanelHeight, availableHeight - splitterHeight - gridGap * 2);
        return Math.max(minFeaturePanelHeight, Math.round(usableHeight / 2));
      }

      function toggleExpectedSideSection() {
        const controls = document.querySelector(".chart-side-controls");
        const collapsed = controls?.classList.contains("chart-expected-collapsed");
        if (collapsed) {
          const height = Number.isFinite(chartFeatureControlsExpandedHeight)
            ? chartFeatureControlsExpandedHeight
            : defaultChartFeatureControlsHeight();
          setChartFeatureControlsHeight(height, { allowCollapse: false });
        } else {
          rememberExpandedChartFeatureControlsHeight();
          setChartFeatureControlsHeight(CHART_FEATURE_CONTROLS_HEIGHT_COLLAPSED);
        }
      }

      function setExpectedSideCollapsed(collapsed) {
        const controls = document.querySelector(".chart-side-controls");
        const expectedSection = el("expectedSideSection");
        controls?.classList.toggle("chart-expected-collapsed", collapsed);
        syncChartExpectedToggle();
        if (!expectedSection) return;
        if (collapsed && expectedSection.contains(document.activeElement)) {
          document.activeElement?.blur?.();
        }
        expectedSection.hidden = collapsed;
        expectedSection.toggleAttribute("inert", collapsed);
        if (collapsed) {
          expectedSection.setAttribute("aria-hidden", "true");
        } else {
          expectedSection.removeAttribute("aria-hidden");
        }
      }

      function syncChartExpectedToggle() {
        const toggle = el("chartExpectedToggle");
        if (!toggle) return;
        const collapsed = document.querySelector(".chart-side-controls")?.classList.contains("chart-expected-collapsed") || false;
        const label = collapsed ? "Show Expected controls" : "Hide Expected controls";
        toggle.setAttribute("aria-expanded", String(!collapsed));
        toggle.setAttribute("aria-label", label);
        toggle.title = label;
      }

      function syncChartControlHeightToAvailableSpace() {
        const controls = document.querySelector(".chart-side-controls");
        if (!controls || controls.classList.contains("hidden") || state.lineBarSideControlsCollapsed) return;
        const firstPanel = controls.querySelector(".chart-side-section");
        if (controls.classList.contains("chart-expected-collapsed")) {
          setChartFeatureControlsHeight(CHART_FEATURE_CONTROLS_HEIGHT_COLLAPSED);
        } else if (firstPanel) {
          setChartFeatureControlsHeight(firstPanel.getBoundingClientRect().height, { allowCollapse: false });
        }
      }

      function setChartFeatureControlsHeight(rawHeight, options = {}) {
        const controls = document.querySelector(".chart-side-controls");
        const availableHeight = controls?.getBoundingClientRect().height || window.innerHeight;
        const splitterHeight = el("chartControlHeightRow")?.getBoundingClientRect().height || 18;
        const gridGap = 8;
        const expandedSplitterSpace = splitterHeight + gridGap * 2;
        const collapsedSplitterSpace = splitterHeight + gridGap;
        const minFeaturePanelHeight = 96;
        const minExpectedPanelHeight = 0;
        const numericHeight = Number(rawHeight);
        const maxExpandedHeight = Math.max(
          minFeaturePanelHeight,
          availableHeight - expandedSplitterSpace - minExpectedPanelHeight,
        );
        const collapseThreshold = maxExpandedHeight;
        const shouldCollapse = rawHeight === CHART_FEATURE_CONTROLS_HEIGHT_COLLAPSED
          || (
            options.allowCollapse !== false
            && Number.isFinite(numericHeight)
            && numericHeight >= collapseThreshold
          );
        if (shouldCollapse) {
          const height = Math.max(minFeaturePanelHeight, availableHeight - collapsedSplitterSpace);
          setExpectedSideCollapsed(true);
          document.documentElement.style.setProperty("--chart-feature-controls-height", `${Math.round(height)}px`);
          return;
        }
        setExpectedSideCollapsed(false);
        const height = Math.min(
          Math.max(Number.isFinite(numericHeight) ? numericHeight : minFeaturePanelHeight, minFeaturePanelHeight),
          maxExpandedHeight,
        );
        document.documentElement.style.setProperty("--chart-feature-controls-height", `${Math.round(height)}px`);
        chartFeatureControlsExpandedHeight = height;
      }

      function bindControls() {
        setupSidebarResize();
        setupChartControlsResize();
        setupChartControlHeightsResize();
        setupLineBarLayoutToggles();
        ukMapTool.bindControls();
        histogramTool.bindControls();
        syncSidebarToggleButton();
        syncSidebarAccordion();
        syncFilterFooterToggleButton();
        syncActionTimingMonitor();
        setFilterSelectionMode(state.filterSelectionMode, { apply: false });
        lineBarTool.bindControls();
        bindFavouriteControls();
        bindToolButtonTooltips();
        document.querySelectorAll('.segmented[data-control="filterOperator"], .segmented[data-control="filterSelectionMode"]').forEach((group) => {
          group.addEventListener("click", (event) => {
            if (event.target.tagName !== "BUTTON") return;
            group.querySelectorAll("button").forEach((button) => button.classList.remove("active"));
            event.target.classList.add("active");
            const previousValue = state[group.dataset.control];
            state[group.dataset.control] = event.target.dataset.value;
            if (state[group.dataset.control] !== previousValue) clearActiveFavouriteSelectionForScope("filter");
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
          clearActiveFavouriteSelectionForScope("metrics");
          refreshMetricSummary();
          refreshActiveToolForMetricChange();
        });
        el("denominator").addEventListener("change", () => {
          syncKpiSelectionFromMetrics();
          clearActiveFavouriteSelectionForScope("metrics");
          refreshMetricSummary();
          refreshActiveToolForMetricChange();
        });
        el("filterApplyBtn").addEventListener("click", applyFilter);
        el("filterClearBtn").addEventListener("click", clearFilter);
        el("filterInput").addEventListener("keydown", (event) => {
          if (event.key === "Enter") {
            event.preventDefault();
            applyFilter();
          }
        });
        el("datasetViewerTool").addEventListener("click", () => handleToolClick("dataset_viewer"));
        el("profileTool").addEventListener("click", () => handleToolClick("column_profile"));
        el("lineBarTool").addEventListener("click", () => handleToolClick("line_bar"));
        el("histogramTool").addEventListener("click", () => handleToolClick("histogram"));
        el("ukMapTool").addEventListener("click", () => handleToolClick("uk_map"));
        el("glmTool").addEventListener("click", () => handleToolClick("glm"));
        el("gbmTool").addEventListener("click", () => handleToolClick("gbm"));
        el("specsTool").addEventListener("click", () => handleToolClick("specs"));
        el("sidebarToggleBtn").addEventListener("click", () => setSidebarVisible(!state.sidebarVisible));
        el("filterFooterToggleBtn").addEventListener("click", () => setFilterFooterVisible(state.filterFooterCollapsed));
        el("favouritesCollapseBtn").addEventListener("click", () => toggleSidebarSection("favourites"));
        el("kpiCollapseBtn").addEventListener("click", () => toggleSidebarSection("kpi"));
        el("glmModelCollapseBtn").addEventListener("click", () => toggleSidebarSection("glm"));
        el("gbmModelCollapseBtn").addEventListener("click", () => toggleSidebarSection("gbm"));
        el("filterCollapseBtn").addEventListener("click", () => toggleSidebarSection("filter"));
        el("filterRowClearBtn").addEventListener("click", clearFilter);
        el("datasetViewerFilterClearBtn").addEventListener("click", clearFilter);
        el("profileFilterClearBtn").addEventListener("click", clearFilter);
        el("lineBarFilterClearBtn").addEventListener("click", clearFilter);
        el("histogramFilterClearBtn").addEventListener("click", clearFilter);
        el("mapControlFilterClearBtn").addEventListener("click", clearFilter);
        el("stopAppBtn")?.addEventListener("click", stopApp);
        el("themeBtn").addEventListener("click", () => {
          document.body.classList.toggle("dark");
          syncThemeButton();
          syncActiveToolTheme();
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
          state.datasetViewerPinnedColumns = [];
          state.datasetViewerColumnCount = null;
          renderSidebarVersion();
          syncHeaderButtons();
          const filtersUnchanged = previousFilterSignature === savedFilterSpecSignature(state.schema.filters || []);
          state.bandFeature = null;
          state.previousBandWidthsByFeature = {};
          state.bandSuggestionPendingKey = null;
          state.bandSuggestionRequestSeq = (state.bandSuggestionRequestSeq || 0) + 1;
          invalidateLineBarDateBucketSuggestion();
          clearToolCaches();
          renderDatasetMeta(schemaFileMeta(), datasetGbmCount, datasetGlmCount);
          refreshDatasetGlmCount();
          refreshDatasetGbmCount();
          await refreshFilterRowCountMeta();
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
          renderFavourites();
          renderToolSelector();
          if (!toolEnabled(state.tool)) {
            state.tool = chooseDefaultTool();
          }
          lineBarTool.renderExpectedNumerators();
          lineBarTool.renderFeatures();
          lineBarTool.updateAxisControls();
          await refreshFavourites();
          setTool(state.tool, false);
          setSidebarVisible(previousSidebarVisible);
          await refreshMetricSummary({ force: true });
          refreshActiveTool({ force: true });
        });
        window.addEventListener("resize", () => {
          hideToolButtonTooltip();
          syncMobileSidebarLayout();
          scheduleDatasetMetaCompactCheck();
          if (state.tool === "line_bar") {
            const controls = document.querySelector(".chart-side-controls");
            if (controls) setChartControlsWidth(controls.getBoundingClientRect().width);
            syncChartControlHeightToAvailableSpace();
            lineBarTool.resize();
          } else if (state.tool === "histogram") {
            histogramTool.resize();
          } else {
            scheduleActiveToolResize({ hard: true });
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
        syncHeaderButtons();
        syncSidebarToggleButton();
        cacheShutdownIcon();
        try {
          setStartupProgress("Requesting schema");
          startStartupTelemetryPolling("Requesting schema");
          state.schema = await api("/api/schema");
          state.datasetViewerColumnCount = null;
          renderSidebarVersion();
          syncHeaderButtons();
          stopStartupTelemetryPolling();
          setStartupProgress("Schema received");
          const path = state.schema.path.split(/[\\/]/).pop();
          const fileMeta = schemaFileMeta();
          document.title = path ? `lucidum · ${path}` : "lucidum";
          resetFilterRowMetaToSchema();
          setStartupProgress("Rendering controls");
          chooseDefaults();
          renderKpis();
          renderFavourites();
          renderToolSelector();
          renderDatasetMeta(fileMeta);
          refreshDatasetGlmCount();
          refreshDatasetGbmCount();
          renderSavedFilters();
          lineBarTool.renderExpectedNumerators();
          lineBarTool.renderFeatures();
          lineBarTool.updateAxisControls();
          await refreshFavourites();
          const defaultStartupTool = chooseDefaultTool();
          const startupFavouriteResult = await applyStartupFavouriteState();
          if (startupFavouriteResult.filterApplied) await refreshFilterRowCountMeta();
          state.tool = startupFavouriteResult.applied
            ? startupToolForFavourite(startupFavouriteResult.favourite, defaultStartupTool)
            : defaultStartupTool;
          setTool(state.tool, false);
          syncMobileSidebarLayout({ initial: true });
          setStartupProgress("Loading initial dataset");
          await refreshMetricSummary({ force: true });
          await refreshActiveTool({ force: true });
          if (startupFavouriteResult.message) setStatus(startupFavouriteResult.message, startupFavouriteResult.statusError);
          setStartupProgress("Ready", "ready");
          startServerHeartbeat();
        } catch (error) {
          stopStartupTelemetryPolling();
          setStartupProgress("Startup failed", "error");
          el("datasetMeta").textContent = "Dataset failed to load";
          setStatus(error.message, true);
        }
      }
