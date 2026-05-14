      function paramsFromLocation() {
        const standardParams = new URLSearchParams(location.search);
        const expectedKeys = ["token", "tool", "x", "actual", "expected", "denominator", "postcode_area", "postcode_sector"];
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
      const state = {
        schema: null,
        x: null,
        sort: "alpha",
        lowGroup: "0",
        labels: "none",
        bandWidth: "0",
        dateBucket: "none",
        transform: "none",
        sigma: "0",
        tool: "line_bar",
        view: "chart",
        mapLevel: "area",
        baseMap: "blank",
        mapBackground: "dark",
        mapPalette: "divergent",
        mapLineWeight: 1,
        mapOpacity: 1,
        mapHotspots: 0,
        mapLabelSize: 0,
        featureSort: "original",
        expectedSort: "original",
        filterOperator: "and",
        activeFilter: "",
        lastData: null,
        lastMapData: null,
        toolCache: {
          line_bar: { requestKey: null, data: null, presentation: null },
          uk_map: { requestKey: null, data: null, presentation: null },
        },
        mapGeoJsonCache: {},
        mapFitLevel: null,
        renderedMapLevel: null,
        preserveMapView: false,
        pendingMapZoom: null,
        mapControlPosition: null,
        tablePage: 1,
        bandFeature: null,
        chartRequestSeq: 0,
        mapRequestSeq: 0,
      };

      const BAND_STEPS = makeBandSteps();
      const TABLE_PAGE_SIZE = 1000;
      const LABEL_DENSITY_LIMIT = 200;
      const RESPONSE_AXIS_PADDING = 0.08;
      const RESPONSE_AXIS_TARGET_INTERVALS = 15;
      const MAP_LEVELS = {
        area: {
          label: "areas",
          singular: "area",
          property: "PostcodeArea",
          url: "/tools/uk-map/static/geodata/areas_MappaR.geojson",
          defaultColumn: "PostcodeArea",
        },
        sector: {
          label: "sectors",
          singular: "sector",
          property: "PostcodeSector",
          url: "/tools/uk-map/static/geodata/sectors_MappaR.geojson",
          defaultColumn: "PostcodeSector",
        },
      };
      const MAP_PALETTES = {
        divergent: ["#00441b", "#1b7837", "#5aae61", "#a6dba0", "#d9f0d3", "#fddbc7", "#f4a582", "#d6604d", "#b2182b", "#67001f"],
        spectral: ["#2c7bb6", "#00a6ca", "#00ccbc", "#90eb9d", "#ffff8c", "#f9d057", "#f29e2e", "#e76818", "#d7191c", "#a50026"],
        viridis: ["#fde725", "#b5de2b", "#6ece58", "#35b779", "#1f9e89", "#26828e", "#31688e", "#3e4989", "#482878", "#440154"],
      };
      const MAP_COLOR_BUCKETS = 100;
      const MAP_LEGEND_BUCKETS = 10;
      const MAP_MISSING_COLOR = "#e5e7eb";
      const MAP_MUTED_COLOR = "#cbd5e1";
      const MAP_CONTROL_POSITION_VERSION = "3";
      const MAP_CONTROL_POSITION_KEYS = {
        left: "py_lucidum_map_control_left",
        top: "py_lucidum_map_control_top",
        version: "py_lucidum_map_control_version",
      };
      const MAP_BASE_LAYERS = {
        blank: { label: "Blank" },
        esri: {
          label: "Esri",
          url: "https://server.arcgisonline.com/ArcGIS/rest/services/World_Street_Map/MapServer/tile/{z}/{y}/{x}",
          attribution: "Tiles &copy; Esri",
        },
        grey: {
          label: "Grey",
          url: "https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png",
          attribution: "&copy; OpenStreetMap contributors &copy; CARTO",
        },
        osm: {
          label: "OSM",
          url: "https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png",
          attribution: "&copy; OpenStreetMap contributors",
        },
        satellite: {
          label: "Satellite",
          url: "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
          attribution: "Tiles &copy; Esri",
        },
      };

      const chart = echarts.init(document.getElementById("chart"));
      let ukMap = null;
      let ukMapLayer = null;
      let ukMapLabelLayer = null;
      let baseTileLayer = null;
      let mapLayerControl = null;
      let mapZoomControl = null;
      let mapHomeControl = null;
      const el = (id) => document.getElementById(id);

      async function api(path, options = {}) {
        const response = await fetch(path, {
          ...options,
          headers: {
            "Content-Type": "application/json",
            "x-lucidum-token": token,
            ...(options.headers || {}),
          },
        });
        if (!response.ok) {
          const text = await response.text();
          let message = text;
          try {
            message = JSON.parse(text).detail || text;
          } catch (_) {
          }
          throw new Error(message);
        }
        return response.json();
      }

      function setStatus(message, isError = false) {
        el("status").textContent = message || "";
        el("status").classList.toggle("error", isError);
        el("status").classList.toggle("hidden", !message);
      }

      function setChartMessage(message) {
        el("chartMessage").textContent = message || "";
        const hiddenForView = state.tool === "line_bar" && state.view !== "chart";
        el("chartMessage").classList.toggle("hidden", !message || hiddenForView);
      }

      function setGroupMeta(message) {
        el("groupMeta").textContent = message || "";
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
        return state.schema.columns.filter((c) => isNumericKind(c.kind));
      }

      function selectedColumn() {
        return state.schema?.columns.find((c) => c.name === state.x);
      }

      function toolEnabled(id) {
        return Boolean((state.schema?.tools || []).some((tool) => tool.id === id));
      }

      function freshToolCache() {
        return {
          line_bar: { requestKey: null, data: null, presentation: null },
          uk_map: { requestKey: null, data: null, presentation: null },
        };
      }

      function clearToolCaches() {
        state.toolCache = freshToolCache();
        state.lastData = null;
        state.lastMapData = null;
        state.renderedMapLevel = null;
      }

      function toolCache(tool) {
        if (!state.toolCache[tool]) {
          state.toolCache[tool] = { requestKey: null, data: null, presentation: null };
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
        setGroupMeta(presentation.groupMeta);
        setStatus(presentation.status, presentation.statusError);
        setChartMessage(presentation.chartMessage);
      }

      function toolHandler(tool) {
        if (tool === "uk_map") {
          return {
            buildRequest: buildMapRequest,
            fetch: fetchMapData,
            useCached: useCachedMapData,
            handleMissingRequest: showMapMissingNumerator,
          };
        }
        return {
          buildRequest: buildChartRequest,
          fetch: fetchChartData,
          useCached: useCachedChartData,
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

      function chooseDefaultTool() {
        const requested = locationParams.get("tool");
        if (requested && toolEnabled(requested)) return requested;
        if (toolEnabled("line_bar")) return "line_bar";
        if (toolEnabled("uk_map")) return "uk_map";
        return "line_bar";
      }

      function renderToolSelector() {
        const lineBarEnabled = toolEnabled("line_bar");
        const ukMapEnabled = toolEnabled("uk_map");
        el("lineBarTool").disabled = !lineBarEnabled;
        el("ukMapTool").disabled = !ukMapEnabled;
        el("lineBarTool").classList.toggle("hidden", !lineBarEnabled);
        el("ukMapTool").classList.toggle("hidden", !ukMapEnabled);
        el("toolSelectorSection").classList.toggle("hidden", !(lineBarEnabled || ukMapEnabled));
      }

      function setTool(tool, refresh = true) {
        if (!toolEnabled(tool)) return;
        state.tool = tool;
        el("lineBarTool").classList.toggle("active", tool === "line_bar");
        el("ukMapTool").classList.toggle("active", tool === "uk_map");
        el("lineBarToolbar").classList.toggle("hidden", tool !== "line_bar");
        el("visualArea").classList.toggle("map-mode", tool === "uk_map");
        el("chartSideControls").classList.toggle("hidden", tool !== "line_bar");
        el("chartControlsResizer").classList.toggle("hidden", tool !== "line_bar");
        el("lineBarTabs").classList.toggle("hidden", tool !== "line_bar");
        el("mapFloatingControl").classList.toggle("hidden", tool !== "uk_map");
        el("mapLegend").classList.toggle("hidden", tool !== "uk_map" || !el("mapLegend").textContent);
        setStatus("");
        setChartMessage("");
        if (tool === "line_bar") {
          el("ukMap").classList.add("hidden");
          el("mapLegend").classList.add("hidden");
          setView(state.view);
          updateAxisControls();
          requestAnimationFrame(() => chart.resize());
        } else {
          el("chart").classList.add("hidden");
          el("tableWrap").classList.add("hidden");
          el("ukMap").classList.remove("hidden");
          initMap();
          syncFloatingMapControl();
          syncMapControls();
          requestAnimationFrame(() => {
            clampMapFloatingControl();
            resizeMap();
          });
        }
        if (refresh && state.schema) refreshActiveTool();
      }

      function isNumericKind(kind) {
        return kind === "numeric" || kind === "integer";
      }

      function syncSegmented(control, value) {
        const group = document.querySelector(`.segmented[data-control="${control}"]`);
        if (!group) return;
        group.querySelectorAll("button").forEach((button) => {
          button.classList.toggle("active", button.dataset.value === value);
        });
      }

      function formatBandWidth(value) {
        const number = Number(value);
        if (!Number.isFinite(number) || number <= 0) return "0";
        return Number(number.toPrecision(12)).toString();
      }

      function syncBandingControl() {
        syncSegmented("bandWidth", state.bandWidth);
        const display = Number(state.bandWidth) > 0 ? state.bandWidth : "auto off";
        el("bandValue").textContent = `(${display})`;
      }

      function autoBandWidthForSelectedColumn() {
        const suggestion = selectedColumn()?.band_suggestion;
        return suggestion ? formatBandWidth(suggestion) : "0";
      }

      function stepBandWidth(direction) {
        const current = Number(state.bandWidth) > 0 ? Number(state.bandWidth) : Number(autoBandWidthForSelectedColumn()) || 1;
        let next = current;
        if (direction < 0) {
          const smallerSteps = BAND_STEPS.filter((step) => step < current);
          next = smallerSteps.length ? smallerSteps[smallerSteps.length - 1] : current;
        } else {
          next = BAND_STEPS.find((step) => step > current) || current;
        }
        state.bandWidth = formatBandWidth(next);
        state.bandFeature = state.x;
        syncBandingControl();
        refreshChart();
      }

      function updateAxisControls() {
        const kind = selectedColumn()?.kind;
        const isDate = kind === "date" || kind === "datetime";
        const isNumeric = isNumericKind(kind);
        const isCategorical = kind === "categorical";
        const hasExpected = Boolean(el("expectedNumerator").value);
        el("sortControl").classList.toggle("hidden", !isCategorical);
        el("expectedSortButton").classList.toggle("hidden", !hasExpected);
        el("dateControl").classList.toggle("hidden", !isDate);
        el("bandControl").classList.toggle("hidden", !isNumeric);
        if (isNumeric && state.bandFeature !== state.x) {
          state.bandWidth = autoBandWidthForSelectedColumn();
          state.bandFeature = state.x;
        }
        if (!isCategorical || (state.sort === "expected" && !hasExpected)) {
          state.sort = "alpha";
          syncSegmented("sort", "alpha");
        } else {
          syncSegmented("sort", state.sort);
        }
        if (!isDate) {
          state.dateBucket = "none";
          syncSegmented("dateBucket", "none");
        }
        if (!isNumeric) {
          state.bandWidth = "0";
          state.bandFeature = state.x;
          syncSegmented("bandWidth", "0");
        }
        syncBandingControl();
      }

      function fillMetricSelect(select, includeNone = false) {
        select.innerHTML = "";
        if (includeNone) {
          select.append(new Option("None", ""));
        }
        for (const col of numericColumns()) {
          select.append(new Option(col.name, col.name));
        }
      }

      function fillDenominatorSelect(select) {
        select.innerHTML = "";
        select.append(new Option("Average row value", "__none__"));
        for (const col of numericColumns()) {
          select.append(new Option(col.name, col.name));
        }
      }

      function columnExists(name) {
        return Boolean(name && state.schema.columns.some((col) => col.name === name));
      }

      function numericColumnExists(name) {
        return Boolean(name && numericColumns().some((col) => col.name === name));
      }

      function requestedDefault(name) {
        return locationParams.get(name) || state.schema.defaults?.[name] || "";
      }

      function renderSavedFilters() {
        const select = el("savedFilterSelect");
        const filters = state.schema.filters || [];
        select.innerHTML = "";
        for (const filter of filters) {
          select.append(new Option(filter.name, filter.expression));
        }
        select.disabled = filters.length === 0;
      }

      function selectedSavedFilterExpressions() {
        return Array.from(el("savedFilterSelect").selectedOptions)
          .map((option) => option.value.trim())
          .filter(Boolean);
      }

      function combinedSavedFilterExpression() {
        const expressions = selectedSavedFilterExpressions();
        if (!expressions.length) return "";
        const operator = state.filterOperator === "or" || state.filterOperator === "nor" ? "OR" : "AND";
        const combined = expressions.join(` ${operator} `);
        return state.filterOperator === "nand" || state.filterOperator === "nor" ? `NOT (${combined})` : combined;
      }

      function applySavedFilters() {
        el("filterInput").value = combinedSavedFilterExpression();
        applyFilter();
      }

      function chooseDefaults() {
        const requestedX = requestedDefault("x");
        state.x = columnExists(requestedX) ? requestedX : state.schema.columns[0]?.name || null;
        fillMetricSelect(el("actualNumerator"));
        fillMetricSelect(el("expectedNumerator"), true);
        fillDenominatorSelect(el("denominator"));
        const requestedActual = requestedDefault("actual");
        const requestedExpected = requestedDefault("expected");
        const requestedDenominator = requestedDefault("denominator");
        el("actualNumerator").value = numericColumnExists(requestedActual) ? requestedActual : numericColumns()[0]?.name || "";
        el("expectedNumerator").value = numericColumnExists(requestedExpected) ? requestedExpected : "";
        el("denominator").value = numericColumnExists(requestedDenominator) ? requestedDenominator : "__none__";
      }

      function renderExpectedNumerators() {
        const query = el("expectedSearch").value.trim().toLowerCase();
        const select = el("expectedNumerator");
        const list = el("expectedList");
        list.innerHTML = "";

        function addExpectedButton(label, value, kind, extraClass = "") {
          const button = document.createElement("button");
          button.type = "button";
          button.className = `feature ${extraClass} ${value === select.value ? "active" : ""}`.trim();
          button.innerHTML = `<span>${escapeHtml(label)}</span><span class="kind">${escapeHtml(kind)}</span>`;
          button.addEventListener("click", () => {
            const changed = select.value !== value;
            select.value = value;
            renderExpectedNumerators();
            updateAxisControls();
            if (changed) refreshChart();
          });
          list.append(button);
        }

        if (!query || "none".includes(query) || "no expected line".includes(query) || "off".includes(query)) {
          addExpectedButton("No expected line", "", "off", "expected-none-option");
        }

        const columns = [...numericColumns()];
        if (state.expectedSort === "alpha") {
          columns.sort((a, b) => a.name.localeCompare(b.name, undefined, { sensitivity: "base" }));
        }
        for (const col of columns) {
          if (query && !col.name.toLowerCase().includes(query)) continue;
          addExpectedButton(col.name, col.name, col.kind);
        }
      }

      function renderFeatures() {
        const query = el("featureSearch").value.trim().toLowerCase();
        const list = el("featureList");
        list.innerHTML = "";
        const columns = [...state.schema.columns];
        if (state.featureSort === "alpha") {
          columns.sort((a, b) => a.name.localeCompare(b.name, undefined, { sensitivity: "base" }));
        }
        for (const col of columns) {
          if (query && !col.name.toLowerCase().includes(query)) continue;
          const button = document.createElement("button");
          button.className = `feature ${col.name === state.x ? "active" : ""}`;
          button.innerHTML = `<span>${escapeHtml(col.name)}</span><span class="kind">${col.kind}</span>`;
          button.addEventListener("click", () => {
            state.x = col.name;
            renderFeatures();
            updateAxisControls();
            refreshChart();
          });
          list.append(button);
        }
      }

      function currentResponses() {
        const responses = [];
        if (el("actualNumerator").value) {
          responses.push({
            label: el("actualNumerator").value,
            numerator: el("actualNumerator").value,
          });
        }
        if (el("expectedNumerator").value) {
          responses.push({
            label: el("expectedNumerator").value,
            numerator: el("expectedNumerator").value,
          });
        }
        return responses;
      }

      function applyFilter() {
        const nextFilter = el("filterInput").value.trim();
        if (nextFilter === state.activeFilter) {
          refreshActiveTool();
          return;
        }
        state.activeFilter = nextFilter;
        refreshActiveTool();
      }

      function clearFilter() {
        el("filterInput").value = "";
        Array.from(el("savedFilterSelect").options).forEach((option) => {
          option.selected = false;
        });
        if (state.activeFilter === "") {
          refreshActiveTool();
          return;
        }
        state.activeFilter = "";
        refreshActiveTool();
      }

      async function stopApp() {
        if (!window.confirm("Stop the local py_lucidum server?")) return;
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
        document.body.classList.add("app-stopped");
        const overlay = document.createElement("div");
        overlay.className = "shutdown-overlay";
        overlay.innerHTML = `
          <div class="shutdown-message" role="status" aria-live="polite">
            <h1>py_lucidum has stopped</h1>
            <p>The local server is no longer running. You can close this browser tab.</p>
          </div>
        `;
        document.body.append(overlay);
      }

      function buildChartRequest() {
        if (!state.schema || !state.x) return null;
        const kind = selectedColumn()?.kind;
        const isDate = kind === "date" || kind === "datetime";
        const isNumeric = isNumericKind(kind);
        return {
          x: state.x,
          sort: state.sort,
          lowGroup: state.lowGroup,
          bandWidth: isNumeric ? Number(state.bandWidth) : 0,
          dateBucket: isDate ? state.dateBucket : "none",
          transform: state.transform,
          sigma: Number(state.sigma),
          filter: state.activeFilter,
          denominator: el("denominator").value,
          responses: currentResponses(),
          maxGroups: 10000,
        };
      }

      async function refreshChart(options = {}) {
        return refreshTool("line_bar", options);
      }

      async function fetchChartData(request, requestKey) {
        const requestSeq = state.chartRequestSeq + 1;
        state.chartRequestSeq = requestSeq;
        setStatus("");
        setChartMessage("");
        setGroupMeta("Computing...");
        updateAxisControls();
        try {
          const data = await api("/api/chart", { method: "POST", body: JSON.stringify(request) });
          if (requestSeq !== state.chartRequestSeq) return;
          const cache = toolCache("line_bar");
          cache.requestKey = requestKey;
          cache.data = data;
          renderChartData(data, { resetTablePage: true });
          return data;
        } catch (error) {
          if (requestSeq !== state.chartRequestSeq) return;
          setGroupMeta("Query failed");
          setChartMessage("");
          setStatus(error.message, true);
        }
      }

      function renderChartData(data, options = {}) {
        state.lastData = data;
        if (options.resetTablePage) {
          state.tablePage = 1;
        }
        updateMetricTitles(data);
        const labelMessage = renderChart(data);
        renderTable(data);
        const filteredRows = data.filtered_row_count ?? data.row_count;
        const rowMeta = filteredRows === data.row_count
          ? `${data.row_count.toLocaleString()} rows`
          : `${filteredRows.toLocaleString()} / ${data.row_count.toLocaleString()} rows`;
        const groupMeta = `${data.rows.length.toLocaleString()} groups · ${rowMeta}`;
        const status = [...(data.warnings || [])].filter(Boolean).join(" ");
        setGroupMeta(groupMeta);
        setStatus(status);
        setChartMessage(labelMessage);
        saveToolPresentation("line_bar", { groupMeta, status, chartMessage: labelMessage });
      }

      function useCachedChartData(cache, options = {}) {
        state.lastData = cache.data;
        if (options.renderIfCached) {
          renderChartData(cache.data);
          return;
        }
        updateMetricTitles(cache.data);
        applyToolPresentation("line_bar");
        requestAnimationFrame(() => chart.resize());
      }

      function buildMapRequest() {
        if (!state.schema) return null;
        const numerator = el("actualNumerator").value;
        if (!numerator) return null;
        return {
          level: state.mapLevel,
          numerator,
          denominator: el("denominator").value,
          filter: state.activeFilter,
          areaColumn: postcodeColumn("area"),
          sectorColumn: postcodeColumn("sector"),
        };
      }

      function showMapMissingNumerator() {
        setGroupMeta("Choose an Actual column");
        setChartMessage("UK mapping needs a numeric Actual column.");
      }

      async function refreshMap(options = {}) {
        return refreshTool("uk_map", options);
      }

      async function fetchMapData(request, requestKey) {
        const requestSeq = state.mapRequestSeq + 1;
        state.mapRequestSeq = requestSeq;
        setStatus("");
        setChartMessage("");
        setGroupMeta("Computing map...");
        try {
          const [data, geoJson] = await Promise.all([
            api("/api/uk-map/summary", { method: "POST", body: JSON.stringify(request) }),
            loadMapGeoJson(request.level),
          ]);
          if (requestSeq !== state.mapRequestSeq) return;
          const cache = toolCache("uk_map");
          cache.requestKey = requestKey;
          cache.data = data;
          updateMapMetricTitles(data);
          renderMap(data, geoJson);
          return data;
        } catch (error) {
          if (requestSeq !== state.mapRequestSeq) return;
          state.pendingMapZoom = null;
          setGroupMeta("Map failed");
          setChartMessage(error.message);
        }
      }

      async function useCachedMapData(cache) {
        state.lastMapData = cache.data;
        updateMapMetricTitles(cache.data);
        syncFloatingMapControl();
        applyToolPresentation("uk_map");
        const geoJson = state.mapGeoJsonCache[cache.data.level];
        if (!ukMapLayer || state.renderedMapLevel !== cache.data.level || state.pendingMapZoom) {
          if (geoJson) {
            state.preserveMapView = !state.pendingMapZoom;
            renderMap(cache.data, geoJson);
          } else {
            const loadedGeoJson = await loadMapGeoJson(cache.data.level);
            state.preserveMapView = !state.pendingMapZoom;
            renderMap(cache.data, loadedGeoJson);
          }
          return;
        }
        requestAnimationFrame(() => resizeMap());
      }

      function postcodeColumn(level) {
        const key = level === "sector" ? "postcode_sector" : "postcode_area";
        const fallback = MAP_LEVELS[level].defaultColumn;
        return locationParams.get(key) || state.schema.defaults?.[key] || fallback;
      }

      async function loadMapGeoJson(level) {
        if (state.mapGeoJsonCache[level]) return state.mapGeoJsonCache[level];
        const config = MAP_LEVELS[level];
        const response = await fetch(config.url);
        if (!response.ok) {
          throw new Error(`Could not load ${config.label} GeoJSON`);
        }
        const geoJson = await response.json();
        const firstFeature = geoJson.features?.[0];
        if (!firstFeature?.properties || !(config.property in firstFeature.properties)) {
          throw new Error(`${config.label} GeoJSON is missing ${config.property}`);
        }
        state.mapGeoJsonCache[level] = geoJson;
        return geoJson;
      }

      function initMap() {
        if (ukMap) return;
        ukMap = L.map("ukMap", {
          preferCanvas: true,
          zoomControl: false,
        }).setView([54.5, -3.2], 6);
        ukMap.on("zoomend", () => {
          if (state.lastMapData?.level === "sector") redrawMapInPlace();
        });
        setBaseMap(state.baseMap);
        addMapLayerControl();
        addMapZoomControl();
        addMapHomeControl();
      }

      function resizeMap() {
        if (!ukMap) return;
        ukMap.invalidateSize();
      }

      function setBaseMap(baseMap) {
        state.baseMap = MAP_BASE_LAYERS[baseMap] ? baseMap : "blank";
        if (!ukMap) return;
        if (baseTileLayer) {
          ukMap.removeLayer(baseTileLayer);
          baseTileLayer = null;
        }
        const config = MAP_BASE_LAYERS[state.baseMap];
        if (config.url) {
          baseTileLayer = L.tileLayer(config.url, {
            maxZoom: 19,
            attribution: config.attribution || "",
          }).addTo(ukMap);
          baseTileLayer.bringToBack();
        }
        ukMap.getContainer().classList.toggle("blank-base", state.baseMap === "blank");
        applyMapBackground();
        syncMapControls();
      }

      function applyMapBackground() {
        const container = ukMap?.getContainer();
        if (!container) return;
        container.classList.toggle("map-bg-dark", state.mapBackground === "dark");
        container.classList.toggle("map-bg-light", state.mapBackground !== "dark");
      }

      function addMapLayerControl() {
        if (!ukMap || mapLayerControl) return;
        const LayerControl = L.Control.extend({
          options: { position: "topleft" },
          onAdd() {
            const container = L.DomUtil.create("div", "map-layer-control leaflet-control");
            container.innerHTML = `
              ${Object.entries(MAP_BASE_LAYERS).map(([value, config]) => `
                <label>
                  <input type="radio" name="baseMap" value="${escapeHtml(value)}">
                  <span>${escapeHtml(config.label)}</span>
                </label>
              `).join("")}
              <div class="map-layer-separator"></div>
              <label>
                <input type="checkbox" name="mapOverlay" value="area">
                <span>Area</span>
              </label>
              <label>
                <input type="checkbox" name="mapOverlay" value="sector">
                <span>Sector</span>
              </label>
              <label>
                <input type="checkbox" name="mapOverlay" value="unit" disabled>
                <span>Unit</span>
              </label>
            `;
            L.DomEvent.disableClickPropagation(container);
            L.DomEvent.disableScrollPropagation(container);
            container.addEventListener("change", handleMapLayerControlChange);
            return container;
          },
        });
        mapLayerControl = new LayerControl();
        mapLayerControl.addTo(ukMap);
        syncMapControls();
      }

      function handleMapLayerControlChange(event) {
        const target = event.target;
        if (!target || target.tagName !== "INPUT") return;
        if (target.name === "baseMap") {
          setBaseMap(target.value);
          return;
        }
        if (target.name === "mapOverlay" && (target.value === "area" || target.value === "sector")) {
          if (!target.checked && target.value === state.mapLevel) {
            target.checked = true;
            return;
          }
          if (target.checked && target.value !== state.mapLevel) {
            state.mapLevel = target.value;
            state.preserveMapView = true;
            syncMapControls();
            refreshMap();
          }
        }
      }

      function syncMapControls() {
        const container = document.querySelector(".map-layer-control");
        if (!container) return;
        container.querySelectorAll('input[name="baseMap"]').forEach((input) => {
          input.checked = input.value === state.baseMap;
        });
        container.querySelectorAll('input[name="mapOverlay"]').forEach((input) => {
          input.checked = input.value === state.mapLevel;
        });
      }

      function addMapZoomControl() {
        if (!ukMap || mapZoomControl) return;
        mapZoomControl = L.control.zoom({ position: "topleft" });
        mapZoomControl.addTo(ukMap);
      }

      function addMapHomeControl() {
        if (!ukMap || mapHomeControl) return;
        const HomeControl = L.Control.extend({
          options: { position: "topleft" },
          onAdd() {
            const container = L.DomUtil.create("div", "map-place-control leaflet-control");
            const ukButton = L.DomUtil.create("button", "map-place-button", container);
            ukButton.type = "button";
            ukButton.title = "Fit UK map layer";
            ukButton.setAttribute("aria-label", "Fit UK map layer");
            ukButton.innerHTML = '<img src="/tools/uk-map/static/icons/UK.png" alt="">';
            const londonButton = L.DomUtil.create("button", "map-place-button", container);
            londonButton.type = "button";
            londonButton.title = "Zoom to London";
            londonButton.setAttribute("aria-label", "Zoom to London");
            londonButton.innerHTML = '<img src="/tools/uk-map/static/icons/London.png" alt="">';
            L.DomEvent.disableClickPropagation(container);
            ukButton.addEventListener("click", (event) => {
              event.preventDefault();
              fitMapToLayer();
            });
            londonButton.addEventListener("click", (event) => {
              event.preventDefault();
              ukMap?.setView([51.5074, -0.1278], 10);
            });
            return container;
          },
        });
        mapHomeControl = new HomeControl();
        mapHomeControl.addTo(ukMap);
      }

      function fitMapToLayer() {
        if (!ukMapLayer) {
          ukMap?.setView([54.5, -3.2], 6);
          return;
        }
        const bounds = ukMapLayer.getBounds();
        if (bounds.isValid()) {
          ukMap.fitBounds(bounds, { padding: [14, 14] });
        }
      }

      function activeMapPalette() {
        return MAP_PALETTES[state.mapPalette] || MAP_PALETTES.viridis;
      }

      function hexToRgb(hex) {
        const match = String(hex || "").trim().match(/^#?([0-9a-f]{6})$/i);
        if (!match) return null;
        const value = Number.parseInt(match[1], 16);
        return {
          r: (value >> 16) & 255,
          g: (value >> 8) & 255,
          b: value & 255,
        };
      }

      function rgbToHex({ r, g, b }) {
        return `#${[r, g, b].map((channel) => {
          const value = Math.min(255, Math.max(0, Math.round(channel)));
          return value.toString(16).padStart(2, "0");
        }).join("")}`;
      }

      function interpolateMapPalette(basePalette, count) {
        const colors = basePalette.map(hexToRgb).filter(Boolean);
        if (!colors.length || count <= 0) return [];
        if (count === 1 || colors.length === 1) return [rgbToHex(colors[0])];
        return Array.from({ length: count }, (_, index) => {
          const position = (index * (colors.length - 1)) / (count - 1);
          const lowerIndex = Math.floor(position);
          const upperIndex = Math.min(colors.length - 1, Math.ceil(position));
          const ratio = position - lowerIndex;
          const lower = colors[lowerIndex];
          const upper = colors[upperIndex];
          return rgbToHex({
            r: lower.r + (upper.r - lower.r) * ratio,
            g: lower.g + (upper.g - lower.g) * ratio,
            b: lower.b + (upper.b - lower.b) * ratio,
          });
        });
      }

      function averageHexColors(colors) {
        const rgbs = colors.map(hexToRgb).filter(Boolean);
        if (!rgbs.length) return MAP_MISSING_COLOR;
        const total = rgbs.reduce((sum, color) => ({
          r: sum.r + color.r,
          g: sum.g + color.g,
          b: sum.b + color.b,
        }), { r: 0, g: 0, b: 0 });
        return rgbToHex({
          r: total.r / rgbs.length,
          g: total.g / rgbs.length,
          b: total.b / rgbs.length,
        });
      }

      function legendPaletteFromMapPalette(mapPalette) {
        return Array.from({ length: MAP_LEGEND_BUCKETS }, (_, index) => {
          const start = Math.floor((index * mapPalette.length) / MAP_LEGEND_BUCKETS);
          const end = Math.max(start + 1, Math.floor(((index + 1) * mapPalette.length) / MAP_LEGEND_BUCKETS));
          return averageHexColors(mapPalette.slice(start, end));
        });
      }

      function quantileThresholds(values, bucketCount) {
        const thresholds = [];
        if (!values.length || bucketCount <= 1) return thresholds;
        for (let index = 1; index < bucketCount; index += 1) {
          thresholds.push(values[Math.min(values.length - 1, Math.ceil((values.length * index) / bucketCount) - 1)]);
        }
        return thresholds;
      }

      function mapHotspotKeys(rows) {
        const count = Number(state.mapHotspots);
        if (!Number.isFinite(count) || count === 0) return null;
        const rankedRows = rows
          .filter((row) => row.key !== null && row.key !== undefined && finiteNumber(row.value) !== null)
          .sort((a, b) => finiteNumber(a.value) - finiteNumber(b.value));
        if (count > 0) rankedRows.reverse();
        return new Set(rankedRows.slice(0, Math.abs(count)).map((row) => String(row.key)));
      }

      function mapLineWeightForLevel(level) {
        const baseWeight = Number(state.mapLineWeight);
        if (!Number.isFinite(baseWeight) || baseWeight <= 0) return 0;
        if (level !== "sector" || !ukMap) return baseWeight;
        const zoom = ukMap.getZoom();
        if (zoom <= 6) return Math.min(baseWeight, 0.15);
        if (zoom <= 7) return Math.min(baseWeight, 0.25);
        if (zoom <= 8) return Math.min(baseWeight, 0.4);
        if (zoom <= 9) return Math.min(baseWeight, 0.65);
        if (zoom <= 10) return Math.min(baseWeight, 0.85);
        return baseWeight;
      }

      function mapFeatureStyle(row, scale, hotspotKeys, level = state.mapLevel) {
        const value = finiteNumber(row?.value);
        const hasValue = value !== null;
        const selected = hasValue && (!hotspotKeys || hotspotKeys.has(String(row.key)));
        const muted = hasValue && !selected;
        const lineWeight = mapLineWeightForLevel(level);
        return {
          color: "#000000",
          opacity: lineWeight > 0 ? (muted ? 0.35 : 0.75) : 0,
          weight: lineWeight,
          fillColor: hasValue ? (muted ? MAP_MUTED_COLOR : scale.color(value)) : MAP_MISSING_COLOR,
          fillOpacity: hasValue ? (muted ? Math.min(Number(state.mapOpacity), 0.22) : Number(state.mapOpacity)) : Math.min(Number(state.mapOpacity), 0.35),
        };
      }

      function renderMap(data, geoJson) {
        state.lastMapData = data;
        state.renderedMapLevel = data.level;
        initMap();
        syncFloatingMapControl();
        const levelConfig = MAP_LEVELS[data.level] || MAP_LEVELS.area;
        const summaries = new Map((data.rows || []).map((row) => [String(row.key), row]));
        const scale = makeQuantileScale(data.rows || []);
        const hotspotKeys = mapHotspotKeys(data.rows || []);
        const featureCount = geoJson.features?.length || 0;
        const matchedFeatureCount = (geoJson.features || []).reduce((count, feature) => {
          const row = summaries.get(String(feature.properties?.[data.join_property] ?? ""));
          return count + (finiteNumber(row?.value) === null ? 0 : 1);
        }, 0);
        if (ukMapLayer) {
          ukMap.removeLayer(ukMapLayer);
          ukMapLayer = null;
        }
        if (ukMapLabelLayer) {
          ukMap.removeLayer(ukMapLabelLayer);
          ukMapLabelLayer = null;
        }
        ukMapLayer = L.geoJSON(geoJson, {
          style: (feature) => {
            const row = summaries.get(String(feature.properties?.[data.join_property] ?? ""));
            return mapFeatureStyle(row, scale, hotspotKeys, data.level);
          },
          onEachFeature: (feature, layer) => {
            const key = String(feature.properties?.[data.join_property] ?? "");
            const row = summaries.get(key);
            const title = key || "Unknown";
            const value = finiteNumber(row?.value);
            layer.bindTooltip(`${title}: ${value === null ? "No data" : formatLineValue(value)}`, { sticky: true });
            layer.bindPopup(mapPopupHtml(title, row, data));
          },
        }).addTo(ukMap);
        renderMapLabels(data, summaries, hotspotKeys);

        let searchWarning = "";
        if (state.pendingMapZoom && state.pendingMapZoom.level === data.level) {
          const zoomed = zoomToMapKey(state.pendingMapZoom.level, state.pendingMapZoom.key);
          if (!zoomed) {
            searchWarning = `Postcode ${state.pendingMapZoom.label} was not found.`;
          }
          state.pendingMapZoom = null;
          state.mapFitLevel = data.level;
          state.preserveMapView = false;
        } else if (state.preserveMapView) {
          state.mapFitLevel = data.level;
          state.preserveMapView = false;
        } else if (state.mapFitLevel !== data.level) {
          const bounds = ukMapLayer.getBounds();
          if (bounds.isValid()) {
            ukMap.fitBounds(bounds, { padding: [14, 14] });
            state.mapFitLevel = data.level;
          }
        }
        renderMapLegend(scale, data.response?.label || "Actual");
        const filteredRows = data.filtered_row_count ?? data.row_count;
        const rowMeta = filteredRows === data.row_count
          ? `${data.row_count.toLocaleString()} rows`
          : `${filteredRows.toLocaleString()} / ${data.row_count.toLocaleString()} rows`;
        const groupMeta = `${matchedFeatureCount.toLocaleString()} / ${featureCount.toLocaleString()} ${levelConfig.label} matched · ${rowMeta}`;
        setGroupMeta(groupMeta);
        const warnings = [...(data.warnings || [])];
        if (searchWarning) {
          warnings.push(searchWarning);
        }
        if (matchedFeatureCount === 0 && (data.rows || []).length) {
          warnings.push(`No ${levelConfig.label} matched the GeoJSON ${levelConfig.property} values.`);
        }
        const chartMessage = warnings.filter(Boolean).join(" ");
        setChartMessage(chartMessage);
        saveToolPresentation("uk_map", { groupMeta, chartMessage });
        requestAnimationFrame(() => resizeMap());
      }

      function updateMapMetricTitles(data) {
        renderMetricTitle(el("actualMetricTitle"), "Actual", data.response?.value);
        renderMetricTitle(el("weightMetricTitle"), "Weight", data.denominator?.value, formatWeightValue);
      }

      function renderMapLabels(data, summaries, hotspotKeys) {
        const fontSize = Number(state.mapLabelSize);
        if (!Number.isFinite(fontSize) || fontSize <= 0 || !ukMapLayer) return;
        ukMapLabelLayer = L.layerGroup().addTo(ukMap);
        ukMapLayer.eachLayer((layer) => {
          const key = String(layer.feature?.properties?.[data.join_property] ?? "");
          const row = summaries.get(key);
          const value = finiteNumber(row?.value);
          if (value === null) return;
          if (hotspotKeys && !hotspotKeys.has(key)) return;
          const bounds = layer.getBounds?.();
          if (!bounds?.isValid()) return;
          const html = `<div class="map-label" style="font-size:${fontSize}px">${escapeHtml(key)}<br>${escapeHtml(formatLineValue(value))}</div>`;
          L.marker(bounds.getCenter(), {
            interactive: false,
            icon: L.divIcon({
              className: "",
              html,
              iconSize: [0, 0],
              iconAnchor: [0, 0],
            }),
          }).addTo(ukMapLabelLayer);
        });
      }

      function zoomToMapKey(level, key) {
        if (!ukMapLayer) return false;
        let targetLayer = null;
        const property = MAP_LEVELS[level]?.property;
        ukMapLayer.eachLayer((layer) => {
          if (targetLayer) return;
          if (String(layer.feature?.properties?.[property] ?? "") === key) {
            targetLayer = layer;
          }
        });
        if (!targetLayer) return false;
        const bounds = targetLayer.getBounds?.();
        if (bounds?.isValid()) {
          ukMap.fitBounds(bounds, { padding: [30, 30], maxZoom: level === "sector" ? 13 : 9 });
          return true;
        }
        return false;
      }

      function redrawMapInPlace() {
        syncFloatingMapControl();
        if (state.tool !== "uk_map" || !state.lastMapData) return;
        const geoJson = state.mapGeoJsonCache[state.lastMapData.level];
        if (!geoJson) return;
        state.preserveMapView = true;
        renderMap(state.lastMapData, geoJson);
      }

      function syncFloatingMapControl() {
        const actualLabel = el("actualNumerator").selectedOptions[0]?.textContent || el("actualNumerator").value || "Actual";
        const denominatorValue = el("denominator").value;
        const denominatorLabel = denominatorValue && denominatorValue !== "__none__"
          ? (el("denominator").selectedOptions[0]?.textContent || denominatorValue)
          : "";
        el("mapControlMetric").textContent = denominatorLabel ? `${actualLabel} / ${denominatorLabel}` : actualLabel;
        el("mapControlFilter").textContent = state.activeFilter || "no filter";
        document.querySelectorAll(".map-palette-button").forEach((button) => {
          button.classList.toggle("active", button.dataset.palette === state.mapPalette);
        });
        document.querySelectorAll(".map-background-button").forEach((button) => {
          button.classList.toggle("active", button.dataset.mapBackground === state.mapBackground);
        });
        el("mapLineWeight").value = String(state.mapLineWeight);
        el("mapOpacity").value = String(state.mapOpacity);
        el("mapHotspots").value = String(state.mapHotspots);
        el("mapLabelSize").value = String(state.mapLabelSize);
        el("mapLineWeightValue").textContent = String(state.mapLineWeight);
        el("mapOpacityValue").textContent = formatCompactSliderValue(state.mapOpacity);
        el("mapHotspotsValue").textContent = String(state.mapHotspots);
        el("mapLabelSizeValue").textContent = String(state.mapLabelSize);
      }

      function formatCompactSliderValue(value) {
        const number = Number(value);
        if (!Number.isFinite(number)) return "";
        return Number.isInteger(number) ? String(number) : String(Number(number.toFixed(1)));
      }

      function normalisePostcodeSearch(raw) {
        const value = String(raw || "").trim().toUpperCase();
        const compact = value.replace(/[^A-Z0-9]/g, "");
        if (!compact) return null;
        const areaMatch = compact.match(/^[A-Z]{1,2}/);
        if (!areaMatch) return null;
        const area = areaMatch[0];
        if (/^[A-Z]{1,2}$/.test(compact)) {
          return { level: "area", key: area, label: area };
        }
        const parts = value.replace(/[^A-Z0-9 ]/g, " ").trim().split(/\s+/).filter(Boolean);
        let sector = "";
        if (parts.length >= 2 && /\d/.test(parts[0]) && /^\d/.test(parts[1])) {
          sector = `${parts[0]} ${parts[1][0]}`;
        } else if (compact.length >= 5 && /\d[A-Z]{2}$/.test(compact)) {
          sector = `${compact.slice(0, -3)} ${compact.slice(-3, -2)}`;
        } else if (/\d$/.test(compact) && /\d/.test(compact.slice(0, -1))) {
          sector = `${compact.slice(0, -1)} ${compact.slice(-1)}`;
        }
        if (sector) return { level: "sector", key: sector, label: sector };
        return { level: "area", key: area, label: area };
      }

      async function searchMapPostcode() {
        const search = normalisePostcodeSearch(el("mapPostcodeInput").value);
        if (!search) {
          setChartMessage("Enter a postcode area or sector.");
          return;
        }
        el("mapPostcodeInput").value = search.label;
        setChartMessage("");
        state.pendingMapZoom = search;
        if (state.mapLevel !== search.level) {
          state.mapLevel = search.level;
          syncMapControls();
          await refreshMap();
          return;
        }
        if (!state.lastMapData || state.lastMapData.level !== search.level) {
          await refreshMap();
          return;
        }
        const zoomed = zoomToMapKey(search.level, search.key);
        if (!zoomed) {
          setChartMessage(`Postcode ${search.label} was not found.`);
        }
        state.pendingMapZoom = null;
      }

      function mapPopupHtml(title, row, data) {
        if (!row) {
          return `<div class="map-popup"><strong>${escapeHtml(title)}</strong><div>No matching data</div></div>`;
        }
        const weightLabel = data.denominator?.bar_label || "Weight";
        return `<div class="map-popup">
          <strong>${escapeHtml(title)}</strong>
          <div>${escapeHtml(data.response?.label || "Actual")}: ${escapeHtml(formatLineValue(row.value) || "No data")}</div>
          <div>${escapeHtml(weightLabel)}: ${escapeHtml(formatNumber(row.denominator))}</div>
          <div>Rows: ${escapeHtml(formatNumber(row.row_count))}</div>
        </div>`;
      }

      function finiteNumber(value) {
        const number = Number(value);
        return Number.isFinite(number) ? number : null;
      }

      function makeQuantileScale(rows) {
        const palette = interpolateMapPalette(activeMapPalette(), MAP_COLOR_BUCKETS);
        const values = rows
          .map((row) => finiteNumber(row.value))
          .filter((value) => value !== null)
          .sort((a, b) => a - b);
        const thresholds = quantileThresholds(values, palette.length);
        return {
          palette,
          legendPalette: legendPaletteFromMapPalette(palette),
          values,
          thresholds,
          legendThresholds: quantileThresholds(values, MAP_LEGEND_BUCKETS),
          color(value) {
            if (value === null) return MAP_MISSING_COLOR;
            let bucket = 0;
            while (bucket < thresholds.length && value > thresholds[bucket]) bucket += 1;
            return palette[Math.min(bucket, palette.length - 1)];
          },
        };
      }

      function renderMapLegend(scale, title) {
        const legend = el("mapLegend");
        if (state.tool !== "uk_map" || !scale.values.length) {
          legend.classList.add("hidden");
          legend.innerHTML = "";
          return;
        }
        const rows = [];
        let lower = null;
        for (let index = 0; index < scale.legendPalette.length; index += 1) {
          const upper = scale.legendThresholds[index] ?? null;
          const label = mapLegendLabel(lower, upper, index === scale.legendPalette.length - 1);
          rows.push(`<div class="map-legend-row"><span class="map-swatch" style="background:${scale.legendPalette[index]}"></span><span>${escapeHtml(label)}</span></div>`);
          lower = upper;
        }
        if (Number(state.mapHotspots) !== 0) {
          rows.push(`<div class="map-legend-row"><span class="map-swatch" style="background:${MAP_MUTED_COLOR}"></span><span>Not selected</span></div>`);
        }
        rows.push(`<div class="map-legend-row"><span class="map-swatch" style="background:${MAP_MISSING_COLOR}"></span><span>No data</span></div>`);
        legend.innerHTML = rows.join("");
        legend.classList.remove("hidden");
      }

      function mapLegendLabel(lower, upper, isLast) {
        if (lower === null && upper === null) return "All values";
        if (lower === null) return `≤ ${formatLineValue(upper)}`;
        if (upper === null || isLast) return `> ${formatLineValue(lower)}`;
        return `${formatLineValue(lower)}–${formatLineValue(upper)}`;
      }

      function renderChart(data) {
        const labels = data.rows.map((r) => formatXLabel(r.x, data.x_kind));
        const labelMode = state.labels;
        const xLabelPolicy = getXAxisLabelPolicy(labels);
        const dataLabelsAllowed = labels.length < LABEL_DENSITY_LIMIT;
        const showBarLabels = dataLabelsAllowed && (labelMode === "bar" || labelMode === "all");
        const showLineLabels = dataLabelsAllowed && (labelMode === "line" || labelMode === "all");
        const barLayout = getBarLayout(labels.length);
        const responseAxis = responseAxisOptions(data);
        const actualColor = getCss("--actual-line");
        const expectedColor = "#d13f3f";
        const responseColors = [actualColor, expectedColor];
        const nColor = getCss("--bar");
        const weightLabel = data.denominator?.bar_label || "Weight";
        const sigmaColor = "#8a94a6";
        const legendData = [
          ...data.responses.map((response) => response.label),
          { name: weightLabel, icon: "roundRect", itemStyle: { color: nColor, borderColor: nColor } },
        ];
        const barSeries = {
          name: weightLabel,
          type: "bar",
          yAxisIndex: 1,
          z: 1,
          legendHoverLink: true,
          itemStyle: { color: nColor },
          animation: false,
          animationDuration: 0,
          animationDurationUpdate: 0,
          data: data.rows.map((r) => ({
            value: r.volume,
            itemStyle: { color: r.is_tail ? getCss("--tail") : getCss("--bar") },
          })),
          label: { show: showBarLabels, position: "top", fontSize: 10, formatter: formatChartLabel },
          barWidth: barLayout.width,
          barMaxWidth: barLayout.maxWidth,
          barCategoryGap: barLayout.categoryGap,
        };
        const lineSeries = data.responses.map((response, index) => ({
          name: response.label,
          type: "line",
          yAxisIndex: 0,
          z: 3,
          animation: false,
          animationDuration: 0,
          animationDurationUpdate: 0,
          smooth: false,
          showSymbol: data.rows.length < 250,
          symbolSize: 5,
          lineStyle: { color: responseColors[index] || actualColor },
          itemStyle: { color: responseColors[index] || actualColor },
          data: data.rows.map((r) => r[`resp${index}`]),
          label: { show: showLineLabels, fontSize: 10, formatter: formatLineLabel },
        }));

        const customSeries = [];
        if (Number(state.sigma) > 0 && data.responses.length >= 2) {
          customSeries.push({
            name: "sigma",
            type: "custom",
            yAxisIndex: 0,
            z: 5,
            legendHoverLink: false,
            animation: false,
            animationDuration: 0,
            animationDurationUpdate: 0,
            renderItem: function (params, api) {
              const x = api.coord([api.value(0), api.value(1)])[0];
              const low = api.coord([api.value(0), api.value(2)])[1];
              const high = api.coord([api.value(0), api.value(3)])[1];
              if (!Number.isFinite(low) || !Number.isFinite(high)) return;
              return {
                type: "group",
                children: [
                  { type: "line", shape: { x1: x, y1: low, x2: x, y2: high }, style: { stroke: sigmaColor, lineWidth: 1.5 } },
                  { type: "line", shape: { x1: x - 4, y1: low, x2: x + 4, y2: low }, style: { stroke: sigmaColor, lineWidth: 1.5 } },
                  { type: "line", shape: { x1: x - 4, y1: high, x2: x + 4, y2: high }, style: { stroke: sigmaColor, lineWidth: 1.5 } },
                ],
              };
            },
            data: data.rows.map((r, i) => [i, r.resp1, r.resp1_low, r.resp1_high]).filter((r) => r.every((v) => v !== null && v !== undefined)),
            encode: { x: 0, y: [2, 3] },
            tooltip: { show: false },
          });
        }

        chart.setOption(
          {
            animation: false,
            animationDuration: 0,
            animationDurationUpdate: 0,
            stateAnimation: { duration: 0 },
            backgroundColor: "transparent",
            color: [actualColor, expectedColor, nColor],
            tooltip: {
              trigger: "axis",
              valueFormatter: (value) => formatNumber(value),
            },
            legend: {
              top: 8,
              data: legendData,
              selectedMode: false,
              textStyle: { color: getCss("--text") },
            },
            grid: { left: 72, right: 76, top: 64, bottom: xLabelPolicy.bottom, containLabel: false },
            xAxis: {
              type: "category",
              data: labels,
              axisLabel: {
                show: xLabelPolicy.show,
                color: getCss("--text"),
                interval: 0,
                hideOverlap: false,
                showMinLabel: true,
                showMaxLabel: true,
                rotate: xLabelPolicy.rotate,
                fontSize: xLabelPolicy.fontSize,
                margin: 8,
              },
              axisLine: { lineStyle: { color: getCss("--line") } },
            },
            yAxis: [
              { type: "value", scale: true, splitNumber: RESPONSE_AXIS_TARGET_INTERVALS, min: responseAxis.min, max: responseAxis.max, interval: responseAxis.interval, axisLabel: { color: getCss("--text"), formatter: (value) => formatNumber(value) }, splitLine: { lineStyle: { color: getCss("--line") } } },
              { type: "value", axisLabel: { color: getCss("--text"), formatter: (value) => formatNumber(value) }, splitLine: { show: false } },
            ],
            dataZoom: labels.length > 120 ? [{ type: "inside" }, { type: "slider", height: 18, bottom: 18 }] : [],
            series: [barSeries, ...lineSeries, ...customSeries],
          },
          true,
        );
        requestAnimationFrame(() => chart.resize());
        return chartDensityMessage(labels.length, !xLabelPolicy.show, !dataLabelsAllowed && labelMode !== "-");
      }

      function chartDensityMessage(groupCount, xLabelsHidden, chartLabelsHidden) {
        if (!xLabelsHidden && !chartLabelsHidden) return "";
        const labelTarget = xLabelsHidden && chartLabelsHidden
          ? "X-axis and chart labels"
          : xLabelsHidden ? "X-axis labels" : "Chart labels";
        return `${labelTarget} hidden as >${LABEL_DENSITY_LIMIT.toLocaleString()} categories.`;
      }

      function updateMetricTitles(data) {
        const summaries = data.response_summaries || [];
        renderMetricTitle(el("actualMetricTitle"), "Actual", summaries[0]?.value);
        renderMetricTitle(el("expectedMetricTitle"), "Expected", summaries[1]?.value);
        renderMetricTitle(el("weightMetricTitle"), "Weight", data.denominator?.value, formatWeightValue);
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

      function responseAxisOptions(data) {
        return responseAxisBounds(responseAxisExtent(data.rows, data.responses.length)) || {};
      }

      function responseAxisExtent(rows, responseCount) {
        let min = Infinity;
        let max = -Infinity;
        rows.forEach((row) => {
          for (let index = 0; index < responseCount; index += 1) {
            const value = Number(row[`resp${index}`]);
            if (!Number.isFinite(value)) continue;
            min = Math.min(min, value);
            max = Math.max(max, value);
          }
        });
        return Number.isFinite(min) && Number.isFinite(max) ? { min, max } : null;
      }

      function responseAxisSpan(value) {
        const min = Number(value?.min);
        const max = Number(value?.max);
        if (!Number.isFinite(min) || !Number.isFinite(max)) return null;
        if (max > min) return max - min;
        return Math.max(Math.abs(max), Math.abs(min), 1);
      }

      function niceAxisStep(span) {
        if (!Number.isFinite(span) || span <= 0) return 1;
        const roughStep = span / RESPONSE_AXIS_TARGET_INTERVALS;
        const magnitude = 10 ** Math.floor(Math.log10(roughStep));
        const normalized = roughStep / magnitude;
        const multiplier = [1, 2, 5, 10].find((candidate) => normalized <= candidate) || 10;
        return multiplier * magnitude;
      }

      function roundAxisValue(value, step) {
        if (!Number.isFinite(value)) return value;
        const precision = Math.min(12, Math.max(0, Math.ceil(-Math.log10(Math.abs(step))) + 3));
        return Number(value.toFixed(precision));
      }

      function responseAxisBounds(value) {
        const min = Number(value?.min);
        const max = Number(value?.max);
        const span = responseAxisSpan(value);
        if (!Number.isFinite(min) || !Number.isFinite(max) || span === null) return null;
        const paddedMin = min - span * RESPONSE_AXIS_PADDING;
        const paddedMax = max + span * RESPONSE_AXIS_PADDING;
        const step = niceAxisStep(paddedMax - paddedMin);
        let axisMin = Math.floor(paddedMin / step) * step;
        let axisMax = Math.ceil(paddedMax / step) * step;
        if (min >= 0) axisMin = Math.max(0, axisMin);
        if (axisMax <= axisMin) axisMax = axisMin + step;
        return {
          min: roundAxisValue(axisMin, step),
          max: roundAxisValue(axisMax, step),
          interval: step,
        };
      }

      function getXAxisLabelPolicy(labels) {
        const maxLength = labels.reduce((longest, label) => Math.max(longest, String(label).length), 0);
        const tooMany = labels.length >= LABEL_DENSITY_LIMIT;
        if (tooMany) {
          return {
            show: false,
            rotate: 0,
            fontSize: 10,
            bottom: 44,
          };
        }
        const rotate = labels.length > 18 || maxLength > 10 ? 65 : 0;
        const fontSize = labels.length > 50 ? 8 : 10;
        const dataZoomSpace = labels.length > 120 ? 36 : 0;
        const estimatedTextWidth = maxLength * fontSize * 0.5;
        const rotatedHeight = estimatedTextWidth * Math.sin((rotate * Math.PI) / 180) + fontSize * Math.cos((rotate * Math.PI) / 180);
        const labelSpace = rotate ? Math.min(140, Math.max(58, Math.ceil(rotatedHeight) + 18)) : 38;
        return {
          show: true,
          rotate,
          fontSize,
          bottom: labelSpace + dataZoomSpace,
        };
      }

      function getBarLayout(count) {
        if (count <= 3) {
          return { width: "62%", maxWidth: 240, categoryGap: "18%" };
        }
        if (count <= 8) {
          return { width: "56%", maxWidth: 180, categoryGap: "24%" };
        }
        if (count <= 20) {
          return { width: "46%", maxWidth: 90, categoryGap: "34%" };
        }
        if (count <= 60) {
          return { width: "68%", maxWidth: 34, categoryGap: "28%" };
        }
        return { width: null, maxWidth: 18, categoryGap: "30%" };
      }

      function renderTable(data) {
        const responseHeaders = data.responses.map((r, i) => `<th>${escapeHtml(r.label)}</th>`).join("");
        const weightLabel = data.denominator?.bar_label || "Weight";
        const needsPagination = data.rows.length > TABLE_PAGE_SIZE;
        const pageCount = needsPagination ? Math.ceil(data.rows.length / TABLE_PAGE_SIZE) : 1;
        state.tablePage = Math.min(Math.max(state.tablePage, 1), pageCount);
        const start = needsPagination ? (state.tablePage - 1) * TABLE_PAGE_SIZE : 0;
        const pageRows = needsPagination ? data.rows.slice(start, start + TABLE_PAGE_SIZE) : data.rows;
        const rows = pageRows
          .map((r) => {
            const values = data.responses.map((_, i) => `<td>${formatNumber(r[`resp${i}`])}</td>`).join("");
            return `<tr><td>${escapeHtml(formatXLabel(r.x, data.x_kind))}</td><td>${formatNumber(r.volume)}</td>${values}</tr>`;
          })
          .join("");
        const pager = needsPagination
          ? `<div class="table-pagination">
              <span>${(start + 1).toLocaleString()}-${(start + pageRows.length).toLocaleString()} of ${data.rows.length.toLocaleString()} rows</span>
              <button id="tablePrevBtn" type="button"${state.tablePage === 1 ? " disabled" : ""}>Previous</button>
              <span>Page ${state.tablePage.toLocaleString()} of ${pageCount.toLocaleString()}</span>
              <button id="tableNextBtn" type="button"${state.tablePage === pageCount ? " disabled" : ""}>Next</button>
            </div>`
          : "";
        el("tableWrap").innerHTML = `<div class="table-scroll"><table><thead><tr><th>${escapeHtml(data.x)}</th><th>${escapeHtml(weightLabel)}</th>${responseHeaders}</tr></thead><tbody>${rows}</tbody></table></div>${pager}`;
        if (needsPagination) {
          el("tablePrevBtn").addEventListener("click", () => {
            state.tablePage -= 1;
            renderTable(data);
          });
          el("tableNextBtn").addEventListener("click", () => {
            state.tablePage += 1;
            renderTable(data);
          });
        }
      }

      function setView(view) {
        state.view = view;
        if (state.tool !== "line_bar") return;
        el("chartTab").classList.toggle("active", view === "chart");
        el("tableTab").classList.toggle("active", view === "table");
        el("chart").classList.toggle("hidden", view !== "chart");
        el("tableWrap").classList.toggle("hidden", view !== "table");
        el("ukMap").classList.add("hidden");
        el("mapLegend").classList.add("hidden");
        el("chartMessage").classList.toggle("hidden", view !== "chart" || !el("chartMessage").textContent);
        if (view === "chart") chart.resize();
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
          chart.resize();
        }
        resizer.addEventListener("pointerup", finishDrag);
        resizer.addEventListener("pointercancel", finishDrag);
      }

      function setSidebarWidth(rawWidth) {
        const viewportLimit = Math.max(260, window.innerWidth - 520);
        const width = Math.min(Math.max(rawWidth, 220), Math.min(560, viewportLimit));
        document.documentElement.style.setProperty("--sidebar-width", `${Math.round(width)}px`);
        requestAnimationFrame(() => chart.resize());
      }

      function setupSidebarFilterResize() {
        const section = document.querySelector(".sidebar-filter-section");
        const resizer = el("sidebarFilterResizer");
        const savedHeight = Number(localStorage.getItem("py_lucidum_sidebar_filter_height"));
        if (Number.isFinite(savedHeight) && savedHeight > 0) {
          setSidebarFilterHeight(savedHeight);
        }

        let dragging = false;
        let startY = 0;
        let startHeight = 0;
        resizer.addEventListener("pointerdown", (event) => {
          event.preventDefault();
          dragging = true;
          startY = event.clientY;
          startHeight = section?.getBoundingClientRect().height || 0;
          resizer.classList.add("dragging");
          document.body.classList.add("resizing-sidebar-filter");
          resizer.setPointerCapture(event.pointerId);
          window.getSelection()?.removeAllRanges();
        });
        resizer.addEventListener("pointermove", (event) => {
          if (!dragging) return;
          event.preventDefault();
          setSidebarFilterHeight(startHeight + startY - event.clientY);
        });
        function finishDrag(event) {
          if (!dragging) return;
          dragging = false;
          resizer.classList.remove("dragging");
          document.body.classList.remove("resizing-sidebar-filter");
          window.getSelection()?.removeAllRanges();
          const height = parseFloat(getComputedStyle(document.documentElement).getPropertyValue("--sidebar-filter-height"));
          if (Number.isFinite(height)) {
            localStorage.setItem("py_lucidum_sidebar_filter_height", String(Math.round(height)));
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

      function setSidebarFilterHeight(rawHeight) {
        const section = document.querySelector(".sidebar-filter-section");
        const aside = section?.closest("aside");
        const occupiedHeight = Array.from(aside?.children || [])
          .filter((child) => child !== section)
          .reduce((total, child) => total + child.getBoundingClientRect().height, 0);
        const availableHeight = (aside?.getBoundingClientRect().height || window.innerHeight) - occupiedHeight - 8;
        const minHeight = 168;
        const maxHeight = Math.max(minHeight, availableHeight);
        const height = Math.min(Math.max(rawHeight, minHeight), maxHeight);
        document.documentElement.style.setProperty("--sidebar-filter-height", `${Math.round(height)}px`);
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
          chart.resize();
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
        requestAnimationFrame(() => chart.resize());
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
        setupSidebarFilterResize();
        setupChartControlsResize();
        setupChartControlHeightsResize();
        setupMapFloatingControlDrag();
        bindMapFloatingControls();
        document.querySelectorAll(".segmented, .filter-operator").forEach((group) => {
          group.addEventListener("click", (event) => {
            if (event.target.tagName !== "BUTTON") return;
            if (group.dataset.control === "bandWidth" && event.target.dataset.action) {
              stepBandWidth(event.target.dataset.action === "band-down" ? -1 : 1);
              return;
            }
            group.querySelectorAll("button").forEach((button) => button.classList.remove("active"));
            event.target.classList.add("active");
            state[group.dataset.control] = event.target.dataset.value;
            if (group.dataset.control === "featureSort") {
              renderFeatures();
              return;
            }
            if (group.dataset.control === "expectedSort") {
              renderExpectedNumerators();
              return;
            }
            if (group.dataset.control === "filterOperator") {
              applySavedFilters();
              return;
            }
            if (group.dataset.control === "bandWidth") {
              state.bandFeature = state.x;
              syncBandingControl();
            }
            refreshChart({ renderIfCached: group.dataset.control === "labels" });
          });
        });
        ["actualNumerator", "denominator"].forEach((id) => {
          el(id).addEventListener("change", refreshActiveTool);
        });
        el("expectedNumerator").addEventListener("change", () => {
          renderExpectedNumerators();
          updateAxisControls();
          refreshChart();
        });
        el("expectedSearch").addEventListener("input", renderExpectedNumerators);
        el("featureSearch").addEventListener("input", renderFeatures);
        el("filterApplyBtn").addEventListener("click", applyFilter);
        el("filterClearBtn").addEventListener("click", clearFilter);
        el("filterInput").addEventListener("keydown", (event) => {
          if (event.key === "Enter") {
            event.preventDefault();
            applyFilter();
          }
        });
        el("savedFilterSelect").addEventListener("change", applySavedFilters);
        el("chartTab").addEventListener("click", () => setView("chart"));
        el("tableTab").addEventListener("click", () => setView("table"));
        el("lineBarTool").addEventListener("click", () => setTool("line_bar"));
        el("ukMapTool").addEventListener("click", () => setTool("uk_map"));
        el("stopAppBtn").addEventListener("click", stopApp);
        el("themeBtn").addEventListener("click", () => {
          document.body.classList.toggle("dark");
          el("themeBtn").textContent = document.body.classList.contains("dark") ? "Light" : "Dark";
          if (state.lastData) renderChart(state.lastData);
          if (state.tool === "uk_map") resizeMap();
        });
        el("reloadBtn").addEventListener("click", async () => {
          setStatus("");
          setGroupMeta("Reloading...");
          state.schema = await api("/api/reload", { method: "POST" });
          state.bandFeature = null;
          state.mapFitLevel = null;
          clearToolCaches();
          renderSavedFilters();
          renderToolSelector();
          if (!toolEnabled(state.tool)) {
            state.tool = chooseDefaultTool();
          }
          renderExpectedNumerators();
          renderFeatures();
          updateAxisControls();
          setTool(state.tool, false);
          refreshActiveTool({ force: true });
        });
        window.addEventListener("resize", () => {
          const filterSection = document.querySelector(".sidebar-filter-section");
          if (filterSection) setSidebarFilterHeight(filterSection.getBoundingClientRect().height);
          if (state.tool === "line_bar") {
            const controls = document.querySelector(".chart-side-controls");
            if (controls) setChartControlsWidth(controls.getBoundingClientRect().width);
            const firstPanel = controls?.querySelector(".chart-side-section");
            if (firstPanel) setChartFeatureControlsHeight(firstPanel.getBoundingClientRect().height);
            chart.resize();
          } else {
            clampMapFloatingControl();
            resizeMap();
          }
        });
      }

      function setupMapFloatingControlDrag() {
        const panel = el("mapFloatingControl");
        const saved = restoreMapFloatingPosition();
        if (saved) {
          requestAnimationFrame(() => setMapFloatingPosition(saved.left, saved.top));
        }

        let dragging = false;
        let startX = 0;
        let startY = 0;
        let startLeft = 0;
        let startTop = 0;
        panel.addEventListener("pointerdown", (event) => {
          if (event.button !== 0 || isMapFloatingInteractiveTarget(event.target)) return;
          event.preventDefault();
          dragging = true;
          startX = event.clientX;
          startY = event.clientY;
          startLeft = panel.offsetLeft;
          startTop = panel.offsetTop;
          panel.classList.add("dragging");
          document.body.classList.add("dragging-map-control");
          panel.setPointerCapture(event.pointerId);
          window.getSelection()?.removeAllRanges();
        });
        panel.addEventListener("pointermove", (event) => {
          if (!dragging) return;
          event.preventDefault();
          setMapFloatingPosition(startLeft + event.clientX - startX, startTop + event.clientY - startY);
        });
        function finishDrag(event) {
          if (!dragging) return;
          dragging = false;
          panel.classList.remove("dragging");
          document.body.classList.remove("dragging-map-control");
          window.getSelection()?.removeAllRanges();
          persistMapFloatingPosition();
          if (event.pointerId !== undefined) {
            try {
              panel.releasePointerCapture(event.pointerId);
            } catch (_) {
            }
          }
        }
        panel.addEventListener("pointerup", finishDrag);
        panel.addEventListener("pointercancel", finishDrag);
      }

      function isMapFloatingInteractiveTarget(target) {
        return Boolean(target?.closest?.("button, input, select, textarea, label, a"));
      }

      function restoreMapFloatingPosition() {
        if (localStorage.getItem(MAP_CONTROL_POSITION_KEYS.version) !== MAP_CONTROL_POSITION_VERSION) {
          clearMapFloatingPosition();
          return null;
        }
        const left = Number(localStorage.getItem(MAP_CONTROL_POSITION_KEYS.left));
        const top = Number(localStorage.getItem(MAP_CONTROL_POSITION_KEYS.top));
        if (!Number.isFinite(left) || !Number.isFinite(top)) return null;
        return { left, top };
      }

      function persistMapFloatingPosition() {
        const position = state.mapControlPosition;
        if (!position) return;
        localStorage.setItem(MAP_CONTROL_POSITION_KEYS.left, String(Math.round(position.left)));
        localStorage.setItem(MAP_CONTROL_POSITION_KEYS.top, String(Math.round(position.top)));
        localStorage.setItem(MAP_CONTROL_POSITION_KEYS.version, MAP_CONTROL_POSITION_VERSION);
      }

      function clearMapFloatingPosition() {
        localStorage.removeItem(MAP_CONTROL_POSITION_KEYS.left);
        localStorage.removeItem(MAP_CONTROL_POSITION_KEYS.top);
        localStorage.removeItem(MAP_CONTROL_POSITION_KEYS.version);
      }

      function setMapFloatingPosition(rawLeft, rawTop) {
        const panel = el("mapFloatingControl");
        const workspace = panel.closest(".workspace");
        const workspaceRect = workspace?.getBoundingClientRect();
        if (!workspaceRect) return;
        const margin = 8;
        const maxLeft = Math.max(margin, workspaceRect.width - panel.offsetWidth - margin);
        const maxTop = Math.max(margin, workspaceRect.height - panel.offsetHeight - margin);
        const left = Math.min(Math.max(rawLeft, margin), maxLeft);
        const top = Math.min(Math.max(rawTop, margin), maxTop);
        panel.style.left = `${Math.round(left)}px`;
        panel.style.top = `${Math.round(top)}px`;
        panel.style.right = "auto";
        state.mapControlPosition = { left, top };
      }

      function clampMapFloatingControl() {
        const panel = el("mapFloatingControl");
        if (state.mapControlPosition) {
          setMapFloatingPosition(state.mapControlPosition.left, state.mapControlPosition.top);
          return;
        }
        if (panel.style.left || panel.style.top) {
          setMapFloatingPosition(panel.offsetLeft, panel.offsetTop);
          return;
        }
        positionMapFloatingControlTopRight();
      }

      function positionMapFloatingControlTopRight() {
        const panel = el("mapFloatingControl");
        const workspace = panel.closest(".workspace");
        const workspaceRect = workspace?.getBoundingClientRect();
        if (!workspaceRect) return;
        const rightInset = 24;
        const topInset = 50;
        setMapFloatingPosition(workspaceRect.width - panel.offsetWidth - rightInset, topInset);
      }

      function bindMapFloatingControls() {
        document.querySelectorAll(".map-palette-button").forEach((button) => {
          button.addEventListener("click", () => {
            state.mapPalette = button.dataset.palette || "viridis";
            redrawMapInPlace();
          });
        });
        document.querySelectorAll(".map-background-button").forEach((button) => {
          button.addEventListener("click", () => {
            state.mapBackground = button.dataset.mapBackground === "dark" ? "dark" : "light";
            applyMapBackground();
            syncFloatingMapControl();
          });
        });
        [
          ["mapLineWeight", "mapLineWeight"],
          ["mapOpacity", "mapOpacity"],
          ["mapHotspots", "mapHotspots"],
          ["mapLabelSize", "mapLabelSize"],
        ].forEach(([id, stateKey]) => {
          el(id).addEventListener("input", (event) => {
            state[stateKey] = Number(event.target.value);
            redrawMapInPlace();
          });
        });
        el("mapPostcodeSearch").addEventListener("click", searchMapPostcode);
        el("mapPostcodeClear").addEventListener("click", () => {
          el("mapPostcodeInput").value = "";
          setChartMessage("");
        });
        el("mapPostcodeInput").addEventListener("keydown", (event) => {
          if (event.key === "Enter") {
            event.preventDefault();
            searchMapPostcode();
          }
        });
      }

      function escapeHtml(value) {
        return String(value).replace(/[&<>"']/g, (char) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#039;" })[char]);
      }

      function formatNumber(value) {
        if (value === null || value === undefined || Number.isNaN(value)) return "";
        const number = Number(value);
        if (!Number.isFinite(number)) return "";
        const abs = Math.abs(number);
        let maximumFractionDigits = 0;
        if (abs !== 0 && abs < 0.01) maximumFractionDigits = 6;
        else if (abs < 1) maximumFractionDigits = 4;
        else if (abs < 10) maximumFractionDigits = 3;
        else if (abs < 1000) maximumFractionDigits = 2;
        else maximumFractionDigits = 1;
        return number.toLocaleString(undefined, {
          minimumFractionDigits: 0,
          maximumFractionDigits,
        });
      }

      function formatChartLabel(params) {
        const value = Array.isArray(params.value) ? params.value[1] : params.value;
        return formatNumber(value);
      }

      function formatLineLabel(params) {
        const value = Array.isArray(params.value) ? params.value[1] : params.value;
        return formatLineValue(value);
      }

      function formatLineValue(value) {
        if (value === null || value === undefined || Number.isNaN(value)) return "";
        const number = Number(value);
        if (!Number.isFinite(number)) return "";
        const abs = Math.abs(number);
        let fractionDigits = 2;
        if (abs !== 0 && abs < 0.01) fractionDigits = 6;
        else if (abs < 1) fractionDigits = 4;
        return number.toLocaleString(undefined, {
          minimumFractionDigits: fractionDigits,
          maximumFractionDigits: fractionDigits,
        });
      }

      function formatWeightValue(value) {
        if (value === null || value === undefined || Number.isNaN(value)) return "";
        const number = Number(value);
        if (!Number.isFinite(number)) return "";
        const abs = Math.abs(number);
        if (abs >= 10 || Number.isInteger(number)) {
          return number.toLocaleString(undefined, {
            minimumFractionDigits: 0,
            maximumFractionDigits: 0,
          });
        }
        return formatNumber(number);
      }

      function formatXLabel(value, kind) {
        if (kind !== "integer") return String(value);
        const number = Number(value);
        if (!Number.isFinite(number) || !Number.isInteger(number)) return String(value);
        return number.toLocaleString(undefined, { maximumFractionDigits: 0 });
      }

      function getCss(name) {
        return getComputedStyle(document.body).getPropertyValue(name).trim();
      }

      async function boot() {
        bindControls();
        try {
          state.schema = await api("/api/schema");
          const path = state.schema.path.split("/").pop();
          el("datasetMeta").textContent = `${path} · ${state.schema.row_count.toLocaleString()} rows · ${state.schema.columns.length} columns`;
          chooseDefaults();
          renderToolSelector();
          state.tool = chooseDefaultTool();
          renderSavedFilters();
          renderExpectedNumerators();
          renderFeatures();
          updateAxisControls();
          setTool(state.tool, false);
          await refreshActiveTool({ force: true });
        } catch (error) {
          setStatus(error.message, true);
        }
      }

      boot();
