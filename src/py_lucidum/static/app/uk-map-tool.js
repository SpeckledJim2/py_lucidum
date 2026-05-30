export function createUkMapTool({
  api,
  el,
  state,
  leafletImpl,
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
}) {
  const L = leafletImpl;

  const MAP_LEVELS = {
    area: {
      label: "areas",
      singular: "area",
      property: "PostcodeArea",
      url: "/tools/uk-map/static/geodata/areas_MappaR.geojson",
      defaultColumn: "PostcodeArea",
      aliases: ["PostcodeArea", "POSTCODE_AREA"],
      smoothFactor: 1,
    },
    sector: {
      label: "sectors",
      singular: "sector",
      property: "PostcodeSector",
      url: "/tools/uk-map/static/geodata/sectors_MappaR.geojson",
      defaultColumn: "PostcodeSector",
      aliases: ["PostcodeSector", "POSTCODE_SECTOR"],
      smoothFactor: 0,
    },
    unit: {
      label: "units",
      singular: "unit",
      property: "PostcodeUnit",
      defaultColumn: "PostcodeUnit",
      aliases: ["PostcodeUnit", "POSTCODE_UNIT"],
    },
  };
  const COORDINATE_COLUMN_ALIASES = {
    latitude: ["lat", "latitude", "LATITUDE"],
    longitude: ["long", "longitude", "LONGITUDE", "LONGiTUDE"],
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
  const MAP_POINT_GRID_SIZE = 18;
  const MAP_FIT_PADDING = [8, 8];
  const MAP_UNIT_FIT_PADDING = [18, 18];
  const MAP_INITIAL_FIT_OPTIONS = { animate: false };
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
    osm: {
      label: "OSM",
      url: "https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png",
      attribution: "&copy; OpenStreetMap contributors",
    },
    satellite: {
      label: "Aerial",
      url: "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
      attribution: "Tiles &copy; Esri",
    },
    grey: {
      label: "Light",
      url: "https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png",
      attribution: "&copy; OpenStreetMap contributors &copy; CARTO",
      themePair: { light: "grey", dark: "darkGrey" },
    },
    darkGrey: {
      label: "Dark",
      url: "https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png",
      attribution: "&copy; OpenStreetMap contributors &copy; CARTO",
      themePair: { light: "grey", dark: "darkGrey" },
    },
  };

  let ukMap = null;
  let ukMapLayer = null;
  let ukMapPointLayer = null;
  let ukMapLabelLayer = null;
  let baseTileLayer = null;
  let mapLayerControl = null;
  let mapZoomControl = null;
  let mapHomeControl = null;
  let mapResizeObserver = null;

  function buildMapRequest() {
    if (!state.schema) return null;
    const numerator = el("actualNumerator").value;
    if (!numerator) return null;
    if (state.mapLevel === "unit" && !mapLevelSelectable("unit")) return null;
    return {
      level: state.mapLevel,
      source: state.source || "dataset",
      numerator,
      denominator: el("denominator").value,
      filter: state.activeFilter,
      areaColumn: postcodeColumn("area"),
      sectorColumn: postcodeColumn("sector"),
      unitColumn: postcodeColumn("unit"),
      latitudeColumn: latitudeColumn(),
      longitudeColumn: longitudeColumn(),
      compactUnitPoints: state.mapLevel === "unit",
    };
  }

  function showMapMissingNumerator() {
    setGroupMeta("uk_map", "Choose an Actual column");
    setChartMessage("UK mapping needs a numeric Actual column.");
  }

  async function refreshMap(options = {}) {
    return refreshUkMap(options);
  }

  async function fetchMapData(request, requestKey) {
    const requestSeq = state.mapRequestSeq + 1;
    state.mapRequestSeq = requestSeq;
    setStatus("");
    setChartMessage("");
    setGroupMeta("uk_map", "Computing map...");
    startToolTiming("uk_map");
    try {
      const [data, geoJson] = await Promise.all([
        api("/api/uk-map/summary", { method: "POST", body: JSON.stringify(request), clientTiming: true }),
        request.level === "unit" ? Promise.resolve(null) : loadMapGeoJson(request.level),
      ]);
      if (requestSeq !== state.mapRequestSeq) return;
      const cache = toolCache("uk_map");
      cache.requestKey = requestKey;
      cache.data = data;
      syncDuckDbTimingFromData("uk_map", data);
      syncClientTimingFromData("uk_map", data);
      updateMapMetricTitles(data);
      renderMap(data, geoJson);
      return data;
    } catch (error) {
      if (requestSeq !== state.mapRequestSeq) return;
      setToolTimingFailed("uk_map");
      state.pendingMapZoom = null;
      setGroupMeta("uk_map", "Map failed");
      setChartMessage(error.message);
    }
  }

  async function useCachedMapData(cache) {
    state.lastMapData = cache.data;
    updateMapMetricTitles(cache.data);
    syncFloatingMapControl();
    applyToolPresentation("uk_map");
    const geoJson = state.mapGeoJsonCache[cache.data.level];
    const activeLayer = cache.data.level === "unit" ? ukMapPointLayer : ukMapLayer;
    if (!activeLayer || state.renderedMapLevel !== cache.data.level || state.pendingMapZoom) {
      if (cache.data.level === "unit") {
        renderMap(cache.data, null);
        return;
      }
      if (geoJson) {
        renderMap(cache.data, geoJson);
      } else {
        const loadedGeoJson = await loadMapGeoJson(cache.data.level);
        renderMap(cache.data, loadedGeoJson);
      }
      return;
    }
    measureToolRender("uk_map", () => scheduleMapViewportSync({ mode: "preserve" }));
  }

  function postcodeColumn(level) {
    const key = level === "sector" ? "postcode_sector" : (level === "unit" ? "postcode_unit" : "postcode_area");
    const fallback = MAP_LEVELS[level].defaultColumn;
    return configuredColumn(key) || resolveColumnAlias(fallback, MAP_LEVELS[level].aliases);
  }

  function latitudeColumn() {
    return configuredColumn("latitude") || resolveColumnAlias("lat", COORDINATE_COLUMN_ALIASES.latitude);
  }

  function longitudeColumn() {
    return configuredColumn("longitude") || resolveColumnAlias("long", COORDINATE_COLUMN_ALIASES.longitude);
  }

  function configuredColumn(key) {
    return locationParams.get(key) || state.schema.defaults?.[key] || "";
  }

  function resolveColumnAlias(requested, aliases) {
    if (columnExists(requested)) return requested;
    return aliases.find((alias) => columnExists(alias)) || requested;
  }

  function configuredDefaultExists(key) {
    return locationParams.has(key) || Object.prototype.hasOwnProperty.call(state.schema?.defaults || {}, key);
  }

  function unitPointColumnsExplicitlyConfigured() {
    return ["postcode_unit", "latitude", "longitude"].some(configuredDefaultExists);
  }

  function unitPointColumnsAvailable() {
    return columnExists(postcodeColumn("unit")) && numericColumnExists(latitudeColumn()) && numericColumnExists(longitudeColumn());
  }

  function mapLevelSelectable(level) {
    if (level !== "unit") return true;
    return unitPointColumnsAvailable() || unitPointColumnsExplicitlyConfigured();
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
      zoomDelta: 0.5,
      zoomSnap: 0.25,
    }).setView([54.5, -3.2], 6);
    ukMap.getContainer()._lucidumMap = ukMap;
    ukMap.on("moveend zoomend", () => captureMapView("leaflet"));
    ukMap.on("zoomend", () => {
      if (state.lastMapData?.level === "sector") restyleActiveMapPolygonLayer();
    });
    setBaseMap(state.baseMap);
    addMapLayerControl();
    addMapZoomControl();
    addMapHomeControl();
    observeMapResize();
  }

  function mapContainerVisible() {
    const container = ukMap?.getContainer?.() || el("ukMap");
    if (!container || container.classList.contains("hidden")) return false;
    const rect = container.getBoundingClientRect();
    return rect.width > 0 && rect.height > 0;
  }

  function normaliseMapView(view) {
    const center = view?.center || {};
    const lat = Number(Array.isArray(center) ? center[0] : center.lat);
    const lng = Number(Array.isArray(center) ? center[1] : center.lng);
    const zoom = Number(view?.zoom);
    if (!Number.isFinite(lat) || !Number.isFinite(lng) || !Number.isFinite(zoom)) return null;
    return { center: { lat, lng }, zoom };
  }

  function currentMapView() {
    if (!ukMap || !mapContainerVisible()) return null;
    const center = ukMap.getCenter();
    return normaliseMapView({ center, zoom: ukMap.getZoom() });
  }

  function captureMapView(reason = "") {
    if (!ukMap || state.restoringMapView) return null;
    const view = currentMapView();
    if (!view) return null;
    if (!state.mapStartupFitDone && !state.mapView && reason !== "startup-fit" && reason !== "explicit") {
      return null;
    }
    state.mapView = view;
    return view;
  }

  function restoreMapView(view) {
    const nextView = normaliseMapView(view);
    if (!ukMap || !nextView || !mapContainerVisible()) return false;
    state.restoringMapView = true;
    state.mapView = nextView;
    try {
      ukMap.setView([nextView.center.lat, nextView.center.lng], nextView.zoom, { animate: false });
      return true;
    } finally {
      requestAnimationFrame(() => {
        state.restoringMapView = false;
      });
    }
  }

  function resizeMap() {
    if (!ukMap || !mapContainerVisible()) return;
    ukMap.invalidateSize({ pan: false });
  }

  function scheduleMapViewportSync({ mode = "preserve" } = {}) {
    if (!ukMap) return;
    const shouldPreserve = mode === "preserve";
    let view = shouldPreserve ? state.mapView : null;
    if (shouldPreserve && (state.mapStartupFitDone || state.mapView)) {
      view = captureMapView("viewport-sync") || view;
    }
    if (state.mapViewportSyncFrame) cancelAnimationFrame(state.mapViewportSyncFrame);
    state.mapViewportSyncFrame = requestAnimationFrame(() => {
      state.mapViewportSyncFrame = null;
      if (!ukMap || !mapContainerVisible()) return;
      resizeMap();
      if (shouldPreserve && view) restoreMapView(view);
    });
  }

  function observeMapResize() {
    if (mapResizeObserver || !window.ResizeObserver) return;
    const mapElement = el("ukMap");
    const target = mapElement?.closest(".workspace") || mapElement;
    if (!target) return;
    mapResizeObserver = new ResizeObserver(() => {
      if (state.tool !== "uk_map") return;
      clampMapFloatingControl();
      scheduleMapViewportSync({ mode: "preserve" });
    });
    mapResizeObserver.observe(target);
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
    const dark = document.body.classList.contains("dark");
    container.classList.toggle("map-bg-dark", dark);
    container.classList.toggle("map-bg-light", !dark);
  }

  function syncCartoBaseMapForTheme() {
    const config = MAP_BASE_LAYERS[state.baseMap];
    const pair = config?.themePair;
    if (!pair) return;
    setBaseMap(document.body.classList.contains("dark") ? pair.dark : pair.light);
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
            <input type="radio" name="mapLevel" value="area">
            <span>Area</span>
          </label>
          <label>
            <input type="radio" name="mapLevel" value="sector">
            <span>Sector</span>
          </label>
          <label>
            <input type="radio" name="mapLevel" value="unit">
            <span>Units</span>
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
      scheduleMapViewportSync({ mode: "preserve" });
      return;
    }
    if (target.name === "mapLevel" && target.checked && mapLevelSelectable(target.value) && target.value !== state.mapLevel) {
      captureMapView("map-level-change");
      state.mapLevel = target.value;
      syncMapControls();
      refreshMap();
    }
  }

  function syncMapControls() {
    const container = document.querySelector(".map-layer-control");
    if (!container) return;
    container.querySelectorAll('input[name="baseMap"]').forEach((input) => {
      input.checked = input.value === state.baseMap;
    });
    container.querySelectorAll('input[name="mapLevel"]').forEach((input) => {
      input.disabled = !mapLevelSelectable(input.value);
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
        londonButton.innerHTML = '<img class="map-place-icon-london" src="/tools/uk-map/static/icons/London.png" alt="">';
        L.DomEvent.disableClickPropagation(container);
        ukButton.addEventListener("click", (event) => {
          event.preventDefault();
          fitMapToLayer();
        });
        londonButton.addEventListener("click", (event) => {
          event.preventDefault();
          if (!ukMap) return;
          ukMap.setView([51.5074, -0.1278], 10, { animate: false });
          state.mapStartupFitDone = true;
          captureMapView("explicit");
        });
        return container;
      },
    });
    mapHomeControl = new HomeControl();
    mapHomeControl.addTo(ukMap);
  }

  function fitMapToLayer(options = {}) {
    const bounds = activeMapBounds();
    if (!bounds) {
      if (!ukMap) return;
      ukMap.setView([54.5, -3.2], 6, { animate: false });
      state.mapStartupFitDone = true;
      captureMapView("explicit");
      return;
    }
    fitMapBounds(bounds, state.renderedMapLevel, options);
  }

  function fitMapBounds(bounds, level = state.renderedMapLevel, options = {}) {
    if (!ukMap || !bounds?.isValid?.()) return false;
    ukMap.fitBounds(bounds, mapFitOptions(level, options));
    state.mapStartupFitDone = true;
    captureMapView("startup-fit");
    return true;
  }

  function mapFitOptions(level, options = {}) {
    const fitOptions = level === "unit"
      ? { padding: MAP_UNIT_FIT_PADDING, maxZoom: 13 }
      : { padding: MAP_FIT_PADDING };
    return { animate: false, ...fitOptions, ...options };
  }

  function activeMapBounds() {
    const layer = state.renderedMapLevel === "unit" ? ukMapPointLayer : ukMapLayer;
    const bounds = layer?.getBounds?.();
    return bounds?.isValid?.() ? bounds : null;
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

  function mapHotspotSelection(value = state.mapHotspots) {
    const raw = Number(value);
    if (!Number.isFinite(raw)) return null;
    const sliderValue = Math.round(raw * 10) / 10;
    if (sliderValue === 0) return null;
    const magnitude = Math.min(1, Math.max(0.1, Math.abs(sliderValue)));
    const fraction = Math.min(1, Math.max(0.1, Math.round((1.1 - magnitude) * 10) / 10));
    if (fraction >= 1) return null;
    return {
      direction: sliderValue > 0 ? -1 : 1,
      fraction,
    };
  }

  function mapHotspotPercent(value = state.mapHotspots) {
    const raw = Number(value);
    if (!Number.isFinite(raw)) return 0;
    const sliderValue = Math.round(raw * 10) / 10;
    if (sliderValue === 0) return 0;
    const magnitude = Math.min(1, Math.max(0.1, Math.abs(sliderValue)));
    return Math.round(Math.min(1, Math.max(0.1, Math.round((1.1 - magnitude) * 10) / 10)) * 100);
  }

  function mapHotspotKeys(rows) {
    const selection = mapHotspotSelection();
    if (!selection) return null;
    const validRows = rows
      .map((row, index) => ({ row, index, value: finiteNumber(row.value) }))
      .filter(({ row, value }) => row.key !== null && row.key !== undefined && value !== null);
    if (!validRows.length) return null;
    validRows.sort((a, b) => {
      if (a.value !== b.value) return (a.value - b.value) * selection.direction;
      return a.index - b.index;
    });
    const count = Math.min(validRows.length, Math.max(1, Math.ceil(validRows.length * selection.fraction)));
    return new Set(validRows.slice(0, count).map(({ row }) => String(row.key)));
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

  function mapPolygonFeatureKey(feature, property) {
    return String(feature?.properties?.[property] ?? "");
  }

  function activeMapPolygonContext() {
    return state.mapPolygonRenderContext;
  }

  function mapPolygonLayerKey(layer, context = activeMapPolygonContext()) {
    return String(layer?._lucidumMapKey ?? mapPolygonFeatureKey(layer?.feature, context?.joinProperty));
  }

  function mapPolygonLayerRow(layer, context = activeMapPolygonContext()) {
    const key = mapPolygonLayerKey(layer, context);
    return {
      key,
      row: context?.summaries?.get(key) || null,
      data: context?.data || null,
    };
  }

  function mapPolygonTooltipHtml(layer) {
    const { key, row } = mapPolygonLayerRow(layer);
    const title = key || "Unknown";
    const value = finiteNumber(row?.value);
    return `${title}: ${value === null ? "No data" : formatLineValue(value)}`;
  }

  function mapPolygonPopupHtml(layer) {
    const { key, row, data } = mapPolygonLayerRow(layer);
    return mapPopupHtml(key || "Unknown", row, data || {});
  }

  function mapPolygonFeatureStyle(feature) {
    const context = activeMapPolygonContext();
    if (!context) return mapFeatureStyle(null, makeQuantileScale([]), null);
    const key = mapPolygonFeatureKey(feature, context.joinProperty);
    const row = context.summaries.get(key);
    return mapFeatureStyle(row, context.scale, context.hotspotKeys, context.data.level);
  }

  function createMapPolygonLayer(level, geoJson) {
    const levelConfig = MAP_LEVELS[level] || MAP_LEVELS.area;
    return L.geoJSON(geoJson, {
      smoothFactor: levelConfig.smoothFactor ?? 1,
      style: mapPolygonFeatureStyle,
      onEachFeature: (feature, layer) => {
        layer._lucidumMapKey = mapPolygonFeatureKey(feature, levelConfig.property);
        layer.bindTooltip(() => mapPolygonTooltipHtml(layer), { sticky: true });
        layer.bindPopup(() => mapPolygonPopupHtml(layer));
      },
    });
  }

  function cachedMapPolygonLayer(level, geoJson) {
    if (!state.mapPolygonLayerCache[level]) {
      state.mapPolygonLayerCache[level] = {
        layer: createMapPolygonLayer(level, geoJson),
        featureCount: geoJson.features?.length || 0,
      };
    }
    return state.mapPolygonLayerCache[level];
  }

  function countMatchedMapPolygonFeatures(layer, summaries) {
    let count = 0;
    layer.eachLayer((featureLayer) => {
      const row = summaries.get(mapPolygonLayerKey(featureLayer));
      if (finiteNumber(row?.value) !== null) count += 1;
    });
    return count;
  }

  function applyMapPolygonStyles() {
    if (!ukMapLayer) return;
    ukMapLayer.setStyle(mapPolygonFeatureStyle);
  }

  function restyleActiveMapPolygonLayer() {
    if (state.tool !== "uk_map" || !state.lastMapData || state.lastMapData.level === "unit") return;
    applyMapPolygonStyles();
  }

  function unitPointRadiusForZoom(zoom) {
    const value = Number(zoom);
    if (!Number.isFinite(value)) return 2.5;
    if (value <= 5) return 1;
    if (value <= 6) return 1.25;
    if (value <= 7) return 1.75;
    if (value <= 8) return 2.5;
    if (value <= 10) return 3.25;
    return 4;
  }

  function unitPointHitRadius(radius) {
    return Math.max(radius + 4, 6);
  }

  function mapPointStyle(row, scale, hotspotKeys, radius) {
    const value = finiteNumber(row?.value);
    const selected = value !== null && (!hotspotKeys || hotspotKeys.has(String(row.key)));
    const muted = value !== null && !selected;
    const strokeOpacity = radius < 2 ? 0 : (radius < 3 ? 0.35 : 0.65);
    return {
      fillColor: muted ? MAP_MUTED_COLOR : scale.color(value),
      fillOpacity: muted ? Math.min(Number(state.mapOpacity), 0.28) : Number(state.mapOpacity),
      strokeOpacity: muted ? Math.min(strokeOpacity, 0.25) : strokeOpacity,
    };
  }

  function unitPointArrays(data) {
    const points = data?.unit_points;
    return points && Array.isArray(points.key) ? points : null;
  }

  function unitPointCount(data) {
    const points = unitPointArrays(data);
    return points ? points.key.length : (data?.rows || []).length;
  }

  function unitPointEntries(data) {
    const bounds = L.latLngBounds([]);
    const entries = [];
    const points = unitPointArrays(data);
    if (points) {
      const count = points.key.length;
      for (let index = 0; index < count; index += 1) {
        const latitude = Number(points.latitude?.[index]);
        const longitude = Number(points.longitude?.[index]);
        if (!Number.isFinite(latitude) || !Number.isFinite(longitude)) continue;
        const latLng = L.latLng(latitude, longitude);
        bounds.extend(latLng);
        entries.push({
          key: points.key[index],
          row_count: points.row_count?.[index],
          numerator: points.numerator?.[index],
          denominator: points.denominator?.[index],
          volume: points.volume?.[index],
          value: points.value?.[index],
          latitude,
          longitude,
          latLng,
        });
      }
      return { entries, bounds };
    }
    for (const row of data?.rows || []) {
      const latitude = Number(row.latitude);
      const longitude = Number(row.longitude);
      if (!Number.isFinite(latitude) || !Number.isFinite(longitude)) continue;
      const latLng = L.latLng(latitude, longitude);
      bounds.extend(latLng);
      entries.push({
        ...row,
        latitude,
        longitude,
        latLng,
      });
    }
    return { entries, bounds };
  }

  function makeUnitPointScale(data) {
    const points = unitPointArrays(data);
    if (!points) return makeQuantileScale(data.rows || []);
    const values = (points.value || [])
      .map(finiteNumber)
      .filter((value) => value !== null);
    return makeQuantileScaleFromValues(values);
  }

  function mapUnitHotspotKeys(data) {
    const points = unitPointArrays(data);
    if (!points) return mapHotspotKeys(data.rows || []);
    const selection = mapHotspotSelection();
    if (!selection) return null;
    const validRows = [];
    for (let index = 0; index < points.key.length; index += 1) {
      const key = points.key[index];
      const value = finiteNumber(points.value?.[index]);
      if (key === null || key === undefined || value === null) continue;
      validRows.push({ key, value, index });
    }
    if (!validRows.length) return null;
    validRows.sort((a, b) => {
      if (a.value !== b.value) return (a.value - b.value) * selection.direction;
      return a.index - b.index;
    });
    const count = Math.min(validRows.length, Math.max(1, Math.ceil(validRows.length * selection.fraction)));
    return new Set(validRows.slice(0, count).map((row) => String(row.key)));
  }

  function makeUnitPointLayer(data, scale, hotspotKeys) {
    return new (L.Layer.extend({
      initialize(mapData, initialScale, initialHotspotKeys) {
        const prepared = unitPointEntries(mapData);
        this.data = mapData;
        this.rows = prepared.entries;
        this.bounds = prepared.bounds;
        this.scale = initialScale;
        this.hotspotKeys = initialHotspotKeys;
        this.tooltip = null;
      },
      onAdd(map) {
        this.map = map;
        this.canvas = L.DomUtil.create("canvas", "leaflet-unit-point-layer");
        this.canvas.style.pointerEvents = "none";
        const pane = map.getPanes().overlayPane;
        pane.appendChild(this.canvas);
        map.on("moveend zoomend resize viewreset", this.reset, this);
        map.on("mousemove", this.handleMouseMove, this);
        map.on("mouseout", this.closeTooltip, this);
        map.on("click", this.handleClick, this);
        this.reset();
      },
      onRemove(map) {
        this.closeTooltip();
        map.off("moveend zoomend resize viewreset", this.reset, this);
        map.off("mousemove", this.handleMouseMove, this);
        map.off("mouseout", this.closeTooltip, this);
        map.off("click", this.handleClick, this);
        this.canvas?.remove();
        this.canvas = null;
        this.map = null;
      },
      getBounds() {
        return this.bounds;
      },
      setRenderContext(nextScale, nextHotspotKeys) {
        this.scale = nextScale;
        this.hotspotKeys = nextHotspotKeys;
        this.reset();
      },
      reset() {
        if (!this.map || !this.canvas) return;
        const size = this.map.getSize();
        const topLeft = this.map.containerPointToLayerPoint([0, 0]);
        const ratio = window.devicePixelRatio || 1;
        L.DomUtil.setPosition(this.canvas, topLeft);
        this.canvas.width = Math.max(1, Math.round(size.x * ratio));
        this.canvas.height = Math.max(1, Math.round(size.y * ratio));
        this.canvas.style.width = `${size.x}px`;
        this.canvas.style.height = `${size.y}px`;
        const context = this.canvas.getContext("2d");
        context.setTransform(ratio, 0, 0, ratio, 0, 0);
        context.clearRect(0, 0, size.x, size.y);
        this.hitGrid = new Map();
        const pointRadius = unitPointRadiusForZoom(this.map.getZoom());
        const hitRadius = unitPointHitRadius(pointRadius);
        this.hitRadius = hitRadius;
        for (const entry of this.rows) {
          const point = this.map.latLngToLayerPoint(entry.latLng).subtract(topLeft);
          if (point.x < -hitRadius || point.y < -hitRadius || point.x > size.x + hitRadius || point.y > size.y + hitRadius) {
            continue;
          }
          const gridKey = `${Math.floor(point.x / MAP_POINT_GRID_SIZE)},${Math.floor(point.y / MAP_POINT_GRID_SIZE)}`;
          if (!this.hitGrid.has(gridKey)) {
            this.hitGrid.set(gridKey, []);
          }
          this.hitGrid.get(gridKey).push({ entry, point });
          const style = mapPointStyle(entry, this.scale, this.hotspotKeys, pointRadius);
          context.globalAlpha = Math.max(0, Math.min(1, style.fillOpacity));
          context.fillStyle = style.fillColor;
          if (pointRadius <= 1) {
            const sizePx = pointRadius * 2;
            context.fillRect(point.x - pointRadius, point.y - pointRadius, sizePx, sizePx);
          } else {
            context.beginPath();
            context.arc(point.x, point.y, pointRadius, 0, Math.PI * 2);
            context.fill();
            if (style.strokeOpacity > 0) {
              context.globalAlpha = Math.max(0, Math.min(1, style.strokeOpacity));
              context.strokeStyle = "#000000";
              context.lineWidth = pointRadius < 3 ? 0.5 : 0.75;
              context.stroke();
            }
          }
        }
        context.globalAlpha = 1;
      },
      findNearest(containerPoint) {
        if (!this.map || !this.hitGrid) return null;
        const hitRadius = this.hitRadius || unitPointHitRadius(unitPointRadiusForZoom(this.map.getZoom()));
        const radiusSquared = hitRadius * hitRadius;
        let nearest = null;
        let nearestDistance = radiusSquared;
        const gridX = Math.floor(containerPoint.x / MAP_POINT_GRID_SIZE);
        const gridY = Math.floor(containerPoint.y / MAP_POINT_GRID_SIZE);
        for (let dxCell = -1; dxCell <= 1; dxCell += 1) {
          for (let dyCell = -1; dyCell <= 1; dyCell += 1) {
            const entries = this.hitGrid.get(`${gridX + dxCell},${gridY + dyCell}`) || [];
            for (const candidate of entries) {
              const dx = candidate.point.x - containerPoint.x;
              const dy = candidate.point.y - containerPoint.y;
              const distance = dx * dx + dy * dy;
              if (distance <= nearestDistance) {
                nearest = candidate.entry;
                nearestDistance = distance;
              }
            }
          }
        }
        return nearest;
      },
      handleMouseMove(event) {
        const nearest = this.findNearest(event.containerPoint);
        if (!nearest) {
          this.closeTooltip();
          return;
        }
        const value = finiteNumber(nearest.value);
        const text = `${nearest.key}: ${value === null ? "No data" : formatLineValue(value)}`;
        if (!this.tooltip) {
          this.tooltip = L.tooltip({ sticky: true, direction: "top", opacity: 0.9 });
        }
        this.tooltip.setLatLng(event.latlng).setContent(text);
        if (!this.map.hasLayer(this.tooltip)) {
          this.tooltip.addTo(this.map);
        }
      },
      handleClick(event) {
        const nearest = this.findNearest(event.containerPoint);
        if (!nearest) return;
        L.popup()
          .setLatLng(nearest.latLng)
          .setContent(mapPopupHtml(String(nearest.key || "Unknown"), nearest, this.data))
          .openOn(this.map);
      },
      closeTooltip() {
        if (this.tooltip && this.map?.hasLayer(this.tooltip)) {
          this.map.removeLayer(this.tooltip);
        }
      },
    }))(data, scale, hotspotKeys);
  }

  function renderMap(data, geoJson) {
    return measureToolRender("uk_map", () => renderMapContents(data, geoJson));
  }

  function applyRenderedMapCamera(level, bounds) {
    let searchWarning = "";
    let explicitMove = false;
    if (state.pendingMapZoom && state.pendingMapZoom.level === level) {
      const zoomed = zoomToMapKey(state.pendingMapZoom.level, state.pendingMapZoom.key);
      if (!zoomed) {
        searchWarning = `Postcode ${state.pendingMapZoom.label} was not found.`;
      }
      explicitMove = zoomed;
      state.pendingMapZoom = null;
    }
    if (!explicitMove) {
      if (state.mapView) {
        restoreMapView(state.mapView);
      } else if (!state.mapStartupFitDone) {
        fitMapBounds(bounds, level, MAP_INITIAL_FIT_OPTIONS);
      }
    }
    return searchWarning;
  }

  function renderMapContents(data, geoJson) {
    if (data.level === "unit") {
      renderUnitMap(data);
      return;
    }
    state.lastMapData = data;
    state.renderedMapLevel = data.level;
    initMap();
    syncFloatingMapControl();
    const levelConfig = MAP_LEVELS[data.level] || MAP_LEVELS.area;
    const summaries = new Map((data.rows || []).map((row) => [String(row.key), row]));
    const scale = makeQuantileScale(data.rows || []);
    const hotspotKeys = mapHotspotKeys(data.rows || []);
    state.mapPolygonRenderContext = {
      data,
      joinProperty: data.join_property,
      summaries,
      scale,
      hotspotKeys,
    };
    const cachedPolygonLayer = cachedMapPolygonLayer(data.level, geoJson);
    const featureCount = cachedPolygonLayer.featureCount;
    const matchedFeatureCount = countMatchedMapPolygonFeatures(cachedPolygonLayer.layer, summaries);
    if (ukMapLayer && ukMapLayer !== cachedPolygonLayer.layer) {
      ukMap.removeLayer(ukMapLayer);
    }
    if (ukMapPointLayer) {
      ukMap.removeLayer(ukMapPointLayer);
      ukMapPointLayer = null;
    }
    if (ukMapLabelLayer) {
      ukMap.removeLayer(ukMapLabelLayer);
      ukMapLabelLayer = null;
    }
    ukMapLayer = cachedPolygonLayer.layer;
    applyMapPolygonStyles();
    if (!ukMap.hasLayer(ukMapLayer)) ukMapLayer.addTo(ukMap);
    renderMapLabels(data, summaries, hotspotKeys);

    const searchWarning = applyRenderedMapCamera(data.level, ukMapLayer.getBounds());
    renderMapLegend(scale, data.response?.label || "Actual");
    const rowMeta = formatRowMeta(data.row_count, data.filtered_row_count);
    const groupMeta = `${matchedFeatureCount.toLocaleString()} / ${featureCount.toLocaleString()} ${levelConfig.label} matched · ${rowMeta}`;
    setFilterRowMeta(data.row_count, data.filtered_row_count);
    setGroupMeta("uk_map", groupMeta);
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
    scheduleMapViewportSync({ mode: "preserve" });
  }

  function renderUnitMap(data) {
    state.lastMapData = data;
    state.renderedMapLevel = data.level;
    initMap();
    syncFloatingMapControl();
    const scale = makeUnitPointScale(data);
    const hotspotKeys = mapUnitHotspotKeys(data);
    if (ukMapLayer) {
      ukMap.removeLayer(ukMapLayer);
      ukMapLayer = null;
    }
    state.mapPolygonRenderContext = null;
    if (ukMapLabelLayer) {
      ukMap.removeLayer(ukMapLabelLayer);
      ukMapLabelLayer = null;
    }
    if (ukMapPointLayer) {
      ukMap.removeLayer(ukMapPointLayer);
      ukMapPointLayer = null;
    }
    ukMapPointLayer = makeUnitPointLayer(data, scale, hotspotKeys).addTo(ukMap);

    applyRenderedMapCamera(data.level, ukMapPointLayer.getBounds());
    renderMapLegend(scale, data.response?.label || "Actual");
    const rowMeta = formatRowMeta(data.row_count, data.filtered_row_count);
    const pointSummary = data.point_summary || {};
    const summaryCount = Number(pointSummary.summary_count ?? unitPointCount(data));
    const plottedCount = Number(pointSummary.plotted_count ?? unitPointCount(data));
    const groupMeta = `${plottedCount.toLocaleString()} / ${summaryCount.toLocaleString()} units plotted · ${rowMeta}`;
    setFilterRowMeta(data.row_count, data.filtered_row_count);
    setGroupMeta("uk_map", groupMeta);
    const warnings = [...(data.warnings || [])];
    const missingValueCount = Number(pointSummary.missing_value_count || 0);
    const missingCoordinateCount = Number(pointSummary.missing_coordinate_count || 0);
    if (missingValueCount) {
      warnings.push(`${missingValueCount.toLocaleString()} ${missingValueCount === 1 ? "unit has" : "units have"} no plottable KPI value.`);
    }
    if (missingCoordinateCount) {
      warnings.push(`${missingCoordinateCount.toLocaleString()} ${missingCoordinateCount === 1 ? "unit has" : "units have"} no valid coordinates.`);
    }
    const chartMessage = warnings.filter(Boolean).join(" ");
    setChartMessage(chartMessage);
    saveToolPresentation("uk_map", { groupMeta, chartMessage });
    scheduleMapViewportSync({ mode: "preserve" });
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
      const key = mapPolygonLayerKey(layer);
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
    ukMapLayer.eachLayer((layer) => {
      if (targetLayer) return;
      if (mapPolygonLayerKey(layer) === key) {
        targetLayer = layer;
      }
    });
    if (!targetLayer) return false;
    const bounds = targetLayer.getBounds?.();
    if (bounds?.isValid()) {
      return fitMapBounds(bounds, level, { padding: [30, 30], maxZoom: level === "sector" ? 13 : 9 });
    }
    return false;
  }

  function redrawMapInPlace() {
    syncFloatingMapControl();
    if (state.tool !== "uk_map" || !state.lastMapData) return;
    if (state.lastMapData.level === "unit") {
      if (!ukMapPointLayer?.setRenderContext) {
        captureMapView("redraw");
        renderMap(state.lastMapData, null);
        return;
      }
      measureToolRender("uk_map", () => {
        const scale = makeUnitPointScale(state.lastMapData);
        const hotspotKeys = mapUnitHotspotKeys(state.lastMapData);
        ukMapPointLayer.setRenderContext(scale, hotspotKeys);
        renderMapLegend(scale, state.lastMapData.response?.label || "Actual");
      });
      return;
    }
    const geoJson = state.mapGeoJsonCache[state.lastMapData.level];
    if (!geoJson) return;
    captureMapView("redraw");
    renderMap(state.lastMapData, geoJson);
  }

  function syncFloatingMapControl() {
    const actualLabel = el("actualNumerator").selectedOptions[0]?.textContent || el("actualNumerator").value || "Actual";
    const denominatorValue = el("denominator").value;
    const denominatorLabel = denominatorValue && denominatorValue !== "__none__"
      ? (el("denominator").selectedOptions[0]?.textContent || denominatorValue)
      : "";
    el("mapControlMetric").textContent = denominatorLabel ? `${actualLabel} / ${denominatorLabel}` : actualLabel;
    syncActiveFilterLabels();
    document.querySelectorAll(".map-palette-button").forEach((button) => {
      button.classList.toggle("active", button.dataset.palette === state.mapPalette);
    });
    el("mapLineWeight").value = String(state.mapLineWeight);
    el("mapOpacity").value = String(state.mapOpacity);
    el("mapHotspots").value = String(state.mapHotspots);
    el("mapLabelSize").value = String(state.mapLabelSize);
    el("mapLineWeightValue").textContent = String(state.mapLineWeight);
    el("mapOpacityValue").textContent = formatCompactSliderValue(state.mapOpacity);
    el("mapHotspotsValue").textContent = formatHotspotSliderValue(state.mapHotspots);
    el("mapLabelSizeValue").textContent = String(state.mapLabelSize);
  }

  function formatCompactSliderValue(value) {
    const number = Number(value);
    if (!Number.isFinite(number)) return "";
    return Number.isInteger(number) ? String(number) : String(Number(number.toFixed(1)));
  }

  function formatHotspotSliderValue(value) {
    const raw = Number(value);
    if (!Number.isFinite(raw)) return "";
    const sliderValue = Math.round(raw * 10) / 10;
    if (sliderValue === 0) return "All";
    return `${sliderValue < 0 ? "B" : "T"}${mapHotspotPercent(sliderValue)}`;
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
    const values = rows
      .map((row) => finiteNumber(row.value))
      .filter((value) => value !== null);
    return makeQuantileScaleFromValues(values);
  }

  function makeQuantileScaleFromValues(rawValues) {
    const palette = interpolateMapPalette(activeMapPalette(), MAP_COLOR_BUCKETS);
    const values = [...rawValues].sort((a, b) => a - b);
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
    if (mapHotspotSelection()) {
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

  function setupMapFloatingControlDrag() {
    const panel = el("mapFloatingControl");
    const saved = restoreMapFloatingPosition();
    if (saved) {
      state.mapControlMoved = true;
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
      if (!state.mapControlMoved) {
        state.mapControlMoved = true;
        setMapFloatingPosition(panel.offsetLeft, panel.offsetTop);
      }
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
    if (state.mapControlMoved) {
      if (state.mapControlPosition) {
        setMapFloatingPosition(state.mapControlPosition.left, state.mapControlPosition.top);
      } else {
        setMapFloatingPosition(panel.offsetLeft, panel.offsetTop);
      }
      return;
    }
    positionMapFloatingControlTopRight();
  }

  function positionMapFloatingControlTopRight() {
    const panel = el("mapFloatingControl");
    const styles = getComputedStyle(panel);
    const rightInset = styles.getPropertyValue("--map-floating-right").trim() || "19px";
    const topInset = styles.getPropertyValue("--map-floating-top").trim() || "16px";
    panel.style.left = "auto";
    panel.style.right = rightInset;
    panel.style.top = topInset;
    state.mapControlPosition = null;
  }

  function resetMapFloatingControlPosition() {
    clearMapFloatingPosition();
    state.mapControlPosition = null;
    state.mapControlMoved = false;
    positionMapFloatingControlTopRight();
  }

  function bindMapFloatingControls() {
    el("mapControlReset").addEventListener("click", resetMapFloatingControlPosition);
    document.querySelectorAll(".map-palette-button").forEach((button) => {
      button.addEventListener("click", () => {
        state.mapPalette = button.dataset.palette || "viridis";
        redrawMapInPlace();
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

  function activate() {
    initMap();
    syncFloatingMapControl();
    syncMapControls();
    requestAnimationFrame(() => {
      clampMapFloatingControl();
      scheduleMapViewportSync({ mode: "preserve" });
    });
  }

  function bindControls() {
    setupMapFloatingControlDrag();
    bindMapFloatingControls();
  }

  function syncViewport(options = {}) {
    clampMapFloatingControl();
    scheduleMapViewportSync(options);
  }

  function resize() {
    resizeMap();
  }

  function refreshTheme() {
    syncCartoBaseMapForTheme();
    applyMapBackground();
    if (state.tool === "uk_map") measureToolRender("uk_map", () => scheduleMapViewportSync({ mode: "preserve" }));
  }

  function resetRenderState() {
    state.lastMapData = null;
    state.mapStartupFitDone = false;
    state.renderedMapLevel = null;
  }

  return {
    buildRequest: buildMapRequest,
    fetchData: fetchMapData,
    useCached: useCachedMapData,
    showMissingRequest: showMapMissingNumerator,
    activate,
    bindControls,
    captureView: captureMapView,
    syncViewport,
    resize,
    refreshTheme,
    resetRenderState,
  };
}
