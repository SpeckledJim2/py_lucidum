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

const POSTCODE_LEVELS = [
  { level: "area", label: "Area" },
  { level: "sector", label: "Sector" },
  { level: "unit", label: "Unit" },
];

const MAP_PENDING_META_DELAY_MS = 500;

function ukMapPositiveRowCount(value) {
  const count = Number(value);
  return Number.isFinite(count) && count > 0 ? count : 0;
}

function ukMapShapeKeySet(shapeKeys) {
  return shapeKeys instanceof Set
    ? shapeKeys
    : new Set(Array.from(shapeKeys || [], (key) => String(key)));
}

export function ukMapShapefileMatchSummary({
  level = "area",
  rows = [],
  filteredRowCount = 0,
  shapeKeys = [],
} = {}) {
  const levelConfig = MAP_LEVELS[level] || MAP_LEVELS.area;
  const validShapeKeys = ukMapShapeKeySet(shapeKeys);
  const matchedRows = [];
  const unmatchedRows = [];
  let matchedRowCount = 0;
  let unmatchedRowCount = 0;

  for (const row of rows || []) {
    const key = row?.key === null || row?.key === undefined ? "" : String(row.key).trim();
    if (!key) continue;
    const rowCount = ukMapPositiveRowCount(row?.row_count);
    if (validShapeKeys.has(key)) {
      matchedRows.push(row);
      matchedRowCount += rowCount;
    } else {
      unmatchedRows.push(row);
      unmatchedRowCount += rowCount;
    }
  }

  const eligibleRowCount = matchedRowCount + unmatchedRowCount;
  const filteredCount = Math.max(0, Number(filteredRowCount) || 0);
  const missingRowCount = Math.max(0, filteredCount - eligibleRowCount);
  const unmatchedPercentage = eligibleRowCount > 0
    ? (unmatchedRowCount / eligibleRowCount) * 100
    : 0;
  const missingPercentage = filteredCount > 0
    ? (missingRowCount / filteredCount) * 100
    : 0;
  const unmatchedPercentageText = unmatchedPercentage.toFixed(1);
  const missingPercentageText = missingPercentage.toFixed(1);
  let matchText = "";
  let matchState = "warning";
  if (eligibleRowCount <= 0) {
    matchText = `No ${levelConfig.label} to match`;
  } else if (unmatchedRowCount <= 0) {
    matchText = `All ${levelConfig.label} matched`;
    matchState = "complete";
  } else {
    matchText = `${unmatchedRowCount.toLocaleString()} ${unmatchedRowCount === 1 ? "row" : "rows"} unmatched (${unmatchedPercentageText}%)`;
  }
  const missingText = missingRowCount > 0
    ? `${missingRowCount.toLocaleString()} ${missingRowCount === 1 ? "row" : "rows"} missing ${levelConfig.singular} (${missingPercentageText}%)`
    : `No rows missing ${levelConfig.singular}`;

  return {
    matchedRows,
    unmatchedRows,
    matchedRowCount,
    unmatchedRowCount,
    eligibleRowCount,
    missingRowCount,
    unmatchedPercentage,
    unmatchedPercentageText,
    missingPercentage,
    missingPercentageText,
    matchText,
    missingText,
    matchState,
  };
}

export const UK_MAP_POSTCODE_REGIONS = Object.freeze([
  Object.freeze({ label: "Central London", areas: Object.freeze(["E", "EC", "N", "NW", "SE", "SW", "W", "WC"]) }),
  Object.freeze({ label: "East Midlands", areas: Object.freeze(["DE", "LE", "LN", "NG", "NN"]) }),
  Object.freeze({ label: "East of England", areas: Object.freeze(["AL", "CB", "CM", "CO", "IP", "LU", "NR", "PE", "SG", "SS"]) }),
  Object.freeze({ label: "North East", areas: Object.freeze(["DH", "DL", "NE", "SR", "TS"]) }),
  Object.freeze({ label: "North West", areas: Object.freeze(["BB", "BL", "CA", "CH", "CW", "FY", "IM", "L", "LA", "M", "OL", "PR", "SK", "WA", "WN"]) }),
  Object.freeze({ label: "Northern Ireland", areas: Object.freeze(["BT"]) }),
  Object.freeze({ label: "Outer London", areas: Object.freeze(["BR", "CR", "DA", "EN", "HA", "IG", "KT", "RM", "SM", "TW", "UB", "WD"]) }),
  Object.freeze({ label: "Scotland", areas: Object.freeze(["AB", "DD", "DG", "EH", "FK", "G", "HS", "IV", "KA", "KW", "KY", "ML", "PA", "PH", "TD", "ZE"]) }),
  Object.freeze({ label: "South East", areas: Object.freeze(["BN", "CT", "GU", "HP", "ME", "MK", "OX", "PO", "RG", "RH", "SL", "SO", "SP", "TN"]) }),
  Object.freeze({ label: "South West", areas: Object.freeze(["BA", "BH", "BS", "DT", "EX", "GL", "GY", "JE", "PL", "SN", "TA", "TQ", "TR"]) }),
  Object.freeze({ label: "Wales", areas: Object.freeze(["CF", "LD", "LL", "NP", "SA", "SY"]) }),
  Object.freeze({ label: "West Midlands", areas: Object.freeze(["B", "CV", "DY", "HR", "ST", "TF", "WR", "WS", "WV"]) }),
  Object.freeze({ label: "Yorkshire and The Humber", areas: Object.freeze(["BD", "DN", "HD", "HG", "HU", "HX", "LS", "S", "WF", "YO"]) }),
]);

const UK_MAP_POSTCODE_AREA_CODES = Object.freeze(
  UK_MAP_POSTCODE_REGIONS.flatMap((region) => region.areas).sort(),
);
const UK_MAP_POSTCODE_AREA_CODE_SET = new Set(UK_MAP_POSTCODE_AREA_CODES);
const MAP_FILTER_ICON_SVG = `
  <svg class="map-popup-action-icon" viewBox="0 0 24 24" aria-hidden="true" focusable="false">
    <path d="M4 5h16l-6.5 7.2v5.3L10.5 20v-7.8z"></path>
  </svg>
`;

function locationParamValue(locationParams, key) {
  if (!locationParams) return "";
  if (typeof locationParams.get === "function") return locationParams.get(key) || "";
  return String(locationParams[key] || "");
}

function schemaDefaultValue(schema, key) {
  return String(schema?.defaults?.[key] || "");
}

function schemaColumnMap(schema) {
  return new Map((schema?.columns || []).map((column) => [column.name, column]));
}

function schemaColumnExists(columnsByName, name) {
  return Boolean(name && columnsByName.has(name));
}

function schemaNumericColumnExists(columnsByName, name) {
  const kind = columnsByName.get(name)?.kind;
  return kind === "numeric" || kind === "integer";
}

function configuredSchemaColumn(schema, locationParams, key) {
  return locationParamValue(locationParams, key) || schemaDefaultValue(schema, key);
}

function firstExistingColumn(columnsByName, names) {
  const seen = new Set();
  for (const name of names) {
    if (!name || seen.has(name)) continue;
    seen.add(name);
    if (schemaColumnExists(columnsByName, name)) return name;
  }
  return "";
}

function resolvedSchemaColumn(columnsByName, schema, locationParams, key, fallback, aliases) {
  const configured = configuredSchemaColumn(schema, locationParams, key);
  if (configured) return schemaColumnExists(columnsByName, configured) ? configured : "";
  return firstExistingColumn(columnsByName, [fallback, ...aliases]);
}

export function ukMapPostcodeAvailability({ schema, locationParams } = {}) {
  const columnsByName = schemaColumnMap(schema);
  const areaColumn = resolvedSchemaColumn(columnsByName, schema, locationParams, "postcode_area", MAP_LEVELS.area.defaultColumn, MAP_LEVELS.area.aliases);
  const sectorColumn = resolvedSchemaColumn(columnsByName, schema, locationParams, "postcode_sector", MAP_LEVELS.sector.defaultColumn, MAP_LEVELS.sector.aliases);
  const unitColumn = resolvedSchemaColumn(columnsByName, schema, locationParams, "postcode_unit", MAP_LEVELS.unit.defaultColumn, MAP_LEVELS.unit.aliases);
  const latitudeColumn = resolvedSchemaColumn(columnsByName, schema, locationParams, "latitude", "lat", COORDINATE_COLUMN_ALIASES.latitude);
  const longitudeColumn = resolvedSchemaColumn(columnsByName, schema, locationParams, "longitude", "long", COORDINATE_COLUMN_ALIASES.longitude);
  const unitAvailable = Boolean(
    unitColumn
      && schemaNumericColumnExists(columnsByName, latitudeColumn)
      && schemaNumericColumnExists(columnsByName, longitudeColumn)
  );
  const columnsByLevel = {
    area: areaColumn,
    sector: sectorColumn,
    unit: unitAvailable ? unitColumn : "",
  };
  const levels = POSTCODE_LEVELS
    .filter((item) => columnsByLevel[item.level])
    .map((item) => ({ ...item, column: columnsByLevel[item.level] }));
  return {
    levels,
    areaColumn,
    sectorColumn,
    unitColumn: unitAvailable ? unitColumn : "",
    latitudeColumn: unitAvailable ? latitudeColumn : "",
    longitudeColumn: unitAvailable ? longitudeColumn : "",
    hasAny: levels.length > 0,
  };
}

export function ukMapPostcodeFilterClause(column, key) {
  const quotedColumn = `"${String(column || "").replaceAll('"', '""')}"`;
  const quotedKey = `'${String(key || "").replaceAll("'", "''")}'`;
  return `${quotedColumn} = ${quotedKey}`;
}

export function ukMapPostcodeInFilterClause(column, keys) {
  const selectedKeys = Array.from(new Set(
    (Array.isArray(keys) ? keys : [])
      .map((key) => String(key || ""))
      .filter(Boolean),
  )).sort();
  if (!selectedKeys.length) return "";
  const quotedColumn = `"${String(column || "").replaceAll('"', '""')}"`;
  const quotedKeys = selectedKeys
    .map((key) => `'${key.replaceAll("'", "''")}'`)
    .join(", ");
  return `${quotedColumn} IN (${quotedKeys})`;
}

export function combineUkMapPostcodeFilter(baseFilter, postcodeClause) {
  const base = String(baseFilter || "").trim();
  const clause = String(postcodeClause || "").trim();
  if (!base) return clause ? `(${clause})` : "";
  if (!clause) return base;
  return `(${base}) AND (${clause})`;
}

export function ukMapPopupContentHtml({
  title,
  row,
  data = {},
  escapeHtml,
  formatNumber,
  formatLineValue,
  showViewRows = true,
  areaFilterToggle = false,
  areaFilterSelected = false,
  postcodeLevel = "",
  postcodeJoinColumn = "",
}) {
  const safeTitle = escapeHtml(title);
  const responseLabel = data.response?.label || "Actual";
  const denominatorLabel = data.denominator?.bar_label || "Weight";
  const averageMode = !data.denominator?.column;
  const formattedLineValue = (value) => formatLineValue(value) || "No data";
  const formattedNumber = (value) => formatNumber(value) || "No data";
  const quantity = (value) => `<strong class="map-popup-quantity">${escapeHtml(value)}</strong>`;
  const actionIcons = {
    copy: `
      <svg class="map-popup-action-icon" viewBox="0 0 24 24" aria-hidden="true" focusable="false">
        <rect x="9" y="9" width="11" height="11" rx="2"></rect>
        <path d="M15 9V6a2 2 0 0 0-2-2H6a2 2 0 0 0-2 2v7a2 2 0 0 0 2 2h3"></path>
      </svg>
    `,
    filter: MAP_FILTER_ICON_SVG,
  };
  const filterActionLabel = areaFilterToggle
    ? `${areaFilterSelected ? "Remove" : "Add"} ${title} ${areaFilterSelected ? "from" : "to"} postcode area filter`
    : `Filter dataset to ${title}`;
  const actionLabels = {
    copy: `Copy ${title} to clipboard`,
    zoom: `Zoom to ${title}`,
    filter: filterActionLabel,
    "view-rows": `View rows for ${title}`,
  };
  const action = (name, label, extraClass = "") => {
    const accessibleLabel = actionLabels[name] || `${label} ${title}`;
    const icon = actionIcons[name];
    const iconClass = icon ? " map-popup-action--icon" : "";
    const activeClass = name === "filter" && areaFilterToggle && areaFilterSelected
      ? " map-popup-action--active"
      : "";
    const customClass = extraClass ? ` ${extraClass}` : "";
    const pressed = name === "filter" && areaFilterToggle
      ? ` aria-pressed="${areaFilterSelected ? "true" : "false"}"`
      : "";
    return `<button class="map-popup-action app-control-button app-command-button${iconClass}${activeClass}${customClass}" type="button" data-map-popup-action="${name}" title="${escapeHtml(accessibleLabel)}" aria-label="${escapeHtml(accessibleLabel)}"${pressed}>${icon || escapeHtml(label)}</button>`;
  };
  const actions = [];
  if (showViewRows) actions.push(action("view-rows", "View rows"));
  actions.push(
    action("zoom", "Zoom"),
    action("filter", "Filter"),
  );

  const lines = [
    `<strong>${safeTitle}</strong>`,
    action("copy", "Copy", "map-popup-header-copy"),
  ];
  if (!row) {
    lines.push('<div class="map-popup-metrics"><div>No matching data</div></div>');
  } else {
    const rowsLabel = row.raw_row_count ?? row.row_count;
    if (mapPopupSmoothingApplied(data, row)) {
      const smoothingLevel = `N${Math.max(0, Math.round(Number(data.smoothing.level) || 0))}`;
      const contributingSectors = Number(row.smoothing_contributing_sectors) || 0;
      const smoothedLines = [`<div class="map-popup-section-title">Smoothed ${escapeHtml(smoothingLevel)}</div>`];
      if (contributingSectors <= 0) {
        smoothedLines.push("<div>No contributing sectors; unsmoothed value shown.</div>");
      } else if (averageMode) {
        smoothedLines.push(`<div>Smoothed average ${escapeHtml(responseLabel)}: ${quantity(formattedLineValue(row.value))}</div>`);
        smoothedLines.push(`<div>Pooled valid records: ${quantity(formattedNumber(row.denominator))}</div>`);
        smoothedLines.push(`<div>Contributing sectors: ${quantity(formattedNumber(contributingSectors))}</div>`);
      } else {
        smoothedLines.push(`<div>Smoothed ${escapeHtml(responseLabel)} / ${escapeHtml(denominatorLabel)}: ${quantity(formattedLineValue(row.value))}</div>`);
        smoothedLines.push(`<div>${escapeHtml(responseLabel)} total: ${quantity(formattedNumber(row.numerator))}</div>`);
        smoothedLines.push(`<div>${escapeHtml(denominatorLabel)}: ${quantity(formattedNumber(row.denominator))}</div>`);
        smoothedLines.push(`<div>Contributing sectors: ${quantity(formattedNumber(contributingSectors))}</div>`);
      }
      lines.push(`<div class="map-popup-section map-popup-section--smoothed">${smoothedLines.join("")}</div>`);

      const unsmoothedLines = ['<div class="map-popup-section-title">Unsmoothed</div>'];
      if (averageMode) {
        unsmoothedLines.push(`<div>Raw average ${escapeHtml(responseLabel)}: ${quantity(formattedLineValue(row.raw_value))}</div>`);
      } else {
        unsmoothedLines.push(`<div>Raw ${escapeHtml(responseLabel)} / ${escapeHtml(denominatorLabel)}: ${quantity(formattedLineValue(row.raw_value))}</div>`);
        unsmoothedLines.push(`<div>Raw ${escapeHtml(responseLabel)} total: ${quantity(formattedNumber(row.raw_numerator))}</div>`);
        unsmoothedLines.push(`<div>Raw ${escapeHtml(denominatorLabel)}: ${quantity(formattedNumber(row.raw_denominator))}</div>`);
      }
      unsmoothedLines.push(`<div>Rows: ${quantity(formattedNumber(rowsLabel))}</div>`);
      lines.push(`<div class="map-popup-section map-popup-section--unsmoothed">${unsmoothedLines.join("")}</div>`);
    } else {
      const metrics = [];
      if (averageMode) {
        metrics.push(`<div>Average ${escapeHtml(responseLabel)}: ${quantity(formattedLineValue(row.value))}</div>`);
      } else {
        metrics.push(`<div>${escapeHtml(responseLabel)} / ${escapeHtml(denominatorLabel)}: ${quantity(formattedLineValue(row.value))}</div>`);
        metrics.push(`<div>${escapeHtml(responseLabel)} total: ${quantity(formattedNumber(row.numerator))}</div>`);
        metrics.push(`<div>${escapeHtml(denominatorLabel)}: ${quantity(formattedNumber(row.denominator))}</div>`);
      }
      metrics.push(`<div>Rows: ${quantity(formattedNumber(rowsLabel))}</div>`);
      lines.push(`<div class="map-popup-metrics">${metrics.join("")}</div>`);
    }
  }
  lines.push(`<div class="map-popup-actions" role="group" aria-label="${escapeHtml(`${title} postcode actions`)}">${actions.join("")}</div>`);
  return `<div class="map-popup" data-map-postcode-key="${safeTitle}" data-map-postcode-level="${escapeHtml(postcodeLevel)}" data-map-postcode-column="${escapeHtml(postcodeJoinColumn)}">${lines.join("")}</div>`;
}

function mapPopupSmoothingApplied(data, row) {
  return Boolean(
    data?.level === "sector"
      && data?.smoothing?.applied
      && Number(data?.smoothing?.level) > 0
      && row
      && Object.prototype.hasOwnProperty.call(row, "raw_value")
  );
}

export function createUkMapTool({
  api,
  el,
  state,
  leafletImpl,
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
  getCss = () => "",
  refreshUkMap,
  copyTextToClipboard = () => Promise.resolve(false),
  showClipboardToast = () => {},
  applyMapPostcodeFilter = () => {},
  applyMapAreaGroupFilter = () => false,
  openMapPostcodeRows = () => {},
  isMapPostcodeSelected = () => false,
  getMapSelectedAreas = () => null,
  canOpenDatasetViewer = () => false,
  clearActiveFavouriteSelection = () => {},
}) {
  const L = leafletImpl;
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
  const MAP_POPUP_MAX_WIDTH = 440;
  const MAP_UNIT_POINT_RADIUS_MULTIPLIER = 0.85;
  const MAP_UNIT_POINT_MIN_RADIUS = 0.5;
  const MAP_UNIT_POINT_MAX_RADIUS_MULTIPLIER = MAP_UNIT_POINT_RADIUS_MULTIPLIER * 4;
  const MAP_DEFAULT_VIEW = { center: { lat: 54.5, lng: -3.2 }, zoom: 6 };
  const MAP_LABEL_MIN_FONT_SIZE = 6;
  const MAP_LABEL_MAX_FONT_SIZE = 20;
  const MAP_INITIAL_FIT_OPTIONS = { animate: false };
  const MAP_CONTROL_EXPANDED_ICON = '<path d="M7 17 17 7"></path><path d="M10 7h7v7"></path>';
  const MAP_CONTROL_COLLAPSED_ICON = '<path d="M17 7 7 17"></path><path d="M14 17H7v-7"></path>';
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
  let mapViewportControl = null;
  let mapResizeObserver = null;
  let activeMapPopupSelection = null;
  let mapRegionFilterPanel = null;
  let stagedMapRegionAreas = null;
  let mapRegionFilterReturnFocus = null;
  let mapRegionDismissClickPending = false;
  let mapPendingMetaTimer = null;
  let mapPendingMetaRequestSeq = null;

  function clearActiveMapFavourite(options = {}) {
    if (state.mapFavouriteRestoreInProgress && !options.force) return;
    clearActiveFavouriteSelection();
  }

  function buildMapRequest() {
    if (!state.schema) return null;
    const numeratorOption = el("actualNumerator")?.selectedOptions?.[0] || null;
    const numerator = numeratorOption?.value || "";
    if (!numerator) return null;
    const denominatorOption = el("denominator")?.selectedOptions?.[0] || null;
    if (denominatorOption?.dataset.unavailable === "true") {
      setStatus("The selected model prediction Denominator is unavailable because there is no active model.", true);
      return null;
    }
    if (state.mapLevel === "unit" && !mapLevelSelectable("unit")) return null;
    return {
      level: state.mapLevel,
      source: numeratorOption?.dataset.sourceId || state.source || "dataset",
      numerator,
      denominator: denominatorOption?.value || "__none__",
      denominatorSource: denominatorOption?.dataset.sourceId || "dataset",
      filter: state.activeFilter,
      areaColumn: postcodeColumn("area"),
      sectorColumn: postcodeColumn("sector"),
      unitColumn: postcodeColumn("unit"),
      latitudeColumn: latitudeColumn(),
      longitudeColumn: longitudeColumn(),
      compactUnitPoints: state.mapLevel === "unit",
      smoothingLevel: state.mapLevel === "sector" ? state.mapSmoothingLevel : 0,
    };
  }

  function showMapMissingNumerator() {
    setGroupMeta("uk_map", "Choose an Actual column");
    setMapRowMeta("");
    setMapMatchLiveStatus("");
    setChartMessage("UK mapping needs a numeric Actual column.");
  }

  function setMapRowMeta(message) {
    el("mapRowMeta").textContent = message || "";
  }

  function setMapMatchLiveStatus(message, { persist = true } = {}) {
    const liveStatus = String(message || "");
    el("mapMatchLiveStatus").textContent = liveStatus;
    if (persist) toolCache("uk_map").mapMatchLiveStatus = liveStatus;
  }

  function mapRowMetaForData(data) {
    return formatRowMeta(data?.row_count, data?.filtered_row_count);
  }

  function cancelMapPendingMeta(requestSeq = null) {
    if (requestSeq !== null && mapPendingMetaRequestSeq !== requestSeq) return;
    if (mapPendingMetaTimer !== null) window.clearTimeout(mapPendingMetaTimer);
    mapPendingMetaTimer = null;
    mapPendingMetaRequestSeq = null;
  }

  function scheduleMapPendingMeta(requestSeq) {
    cancelMapPendingMeta();
    mapPendingMetaRequestSeq = requestSeq;
    mapPendingMetaTimer = window.setTimeout(() => {
      mapPendingMetaTimer = null;
      mapPendingMetaRequestSeq = null;
      if (requestSeq !== state.mapRequestSeq) return;
      setGroupMeta("uk_map", "Computing map...");
      setMapRowMeta("");
      setMapMatchLiveStatus("");
    }, MAP_PENDING_META_DELAY_MS);
  }

  async function refreshMap(options = {}) {
    return refreshUkMap(options);
  }

  async function fetchMapData(request, requestKey, options = {}) {
    const requestSeq = state.mapRequestSeq + 1;
    state.mapRequestSeq = requestSeq;
    const quietPending = Boolean(options.preserveRenderedMap && options.suppressPendingMeta);
    cancelMapPendingMeta();
    setStatus("");
    setChartMessage("");
    if (!quietPending) scheduleMapPendingMeta(requestSeq);
    startToolTiming("uk_map");
    try {
      const [data, geoJson] = await Promise.all([
        api("/api/uk-map/summary", { method: "POST", body: JSON.stringify(request), clientTiming: true }),
        request.level === "unit" ? Promise.resolve(null) : loadMapGeoJson(request.level),
      ]);
      if (requestSeq !== state.mapRequestSeq) return;
      cancelMapPendingMeta(requestSeq);
      const cache = toolCache("uk_map");
      cache.requestKey = requestKey;
      cache.data = data;
      syncDuckDbTimingFromData("uk_map", data);
      syncClientTimingFromData("uk_map", data);
      renderMap(data, geoJson);
      return data;
    } catch (error) {
      if (requestSeq !== state.mapRequestSeq) return;
      cancelMapPendingMeta(requestSeq);
      setToolTimingFailed("uk_map");
      state.pendingMapZoom = null;
      state.mapViewRestorePending = null;
      setGroupMeta("uk_map", "Map failed");
      setMapRowMeta("");
      setMapMatchLiveStatus("");
      setChartMessage(error.message);
    }
  }

  function cancelMapRequests(options = {}) {
    cancelMapPendingMeta();
    state.mapRequestSeq += 1;
    if (!options.preservePendingRestore) state.mapViewRestorePending = null;
  }

  function clearRenderedMap() {
    state.lastMapData = null;
    state.renderedMapLevel = null;
    state.mapPolygonRenderContext = null;
    if (ukMapLayer && ukMap) {
      ukMap.removeLayer(ukMapLayer);
      ukMapLayer = null;
    }
    if (ukMapPointLayer && ukMap) {
      ukMap.removeLayer(ukMapPointLayer);
      ukMapPointLayer = null;
    }
    if (ukMapLabelLayer && ukMap) {
      ukMap.removeLayer(ukMapLabelLayer);
      ukMapLabelLayer = null;
    }
    el("mapLegendBody").textContent = "";
    el("mapLegend").classList.add("hidden");
    setMapRowMeta("");
    setMapMatchLiveStatus("");
  }

  function showPendingRestore() {
    cancelMapRequests({ preservePendingRestore: true });
    setStatus("");
    setChartMessage("");
    setGroupMeta("uk_map", "");
    setMapRowMeta("");
    setMapMatchLiveStatus("");
    clearRenderedMap();
    syncFloatingMapControl();
  }

  async function useCachedMapData(cache, options = {}) {
    state.lastMapData = cache.data;
    syncFloatingMapControl();
    applyToolPresentation("uk_map");
    setMapRowMeta(mapRowMetaForData(cache.data));
    setMapMatchLiveStatus(cache.mapMatchLiveStatus || "", { persist: false });
    const geoJson = state.mapGeoJsonCache[cache.data.level];
    const activeLayer = cache.data.level === "unit" ? ukMapPointLayer : ukMapLayer;
    if (options.renderIfCached) {
      refreshTheme();
      if (cache.data.level === "unit") {
        renderMap(cache.data, null);
      } else if (geoJson) {
        renderMap(cache.data, geoJson);
      } else {
        const loadedGeoJson = await loadMapGeoJson(cache.data.level);
        renderMap(cache.data, loadedGeoJson);
      }
      return;
    }
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
    }).setView([MAP_DEFAULT_VIEW.center.lat, MAP_DEFAULT_VIEW.center.lng], MAP_DEFAULT_VIEW.zoom);
    ukMap.getContainer()._lucidumMap = ukMap;
    ukMap.on("moveend zoomend", () => captureMapView("leaflet"));
    ukMap.on("zoomend", () => {
      if (state.lastMapData?.level === "sector") restyleActiveMapPolygonLayer();
    });
    ukMap.on("popupclose", (event) => {
      if (!activeMapPopupSelection?.popup || activeMapPopupSelection.popup === event.popup) {
        activeMapPopupSelection = null;
      }
    });
    setBaseMap(state.baseMap);
    addMapViewportControl();
    observeMapResize();
  }

  function mapRegionFilterIsOpen() {
    return Boolean(mapRegionFilterPanel && !mapRegionFilterPanel.hidden);
  }

  function ensureMapRegionFilterPanel() {
    if (mapRegionFilterPanel) return mapRegionFilterPanel;
    const regionRows = UK_MAP_POSTCODE_REGIONS.map((region, index) => `
      <label class="map-region-filter-option" for="mapRegionFilterOption${index}">
        <input id="mapRegionFilterOption${index}" type="checkbox" data-map-region-index="${index}">
        <span>${escapeHtml(region.label)}</span>
      </label>
    `).join("");
    mapRegionFilterPanel = document.createElement("div");
    mapRegionFilterPanel.id = "mapRegionFilterPanel";
    mapRegionFilterPanel.className = "map-region-filter-panel";
    mapRegionFilterPanel.hidden = true;
    mapRegionFilterPanel.setAttribute("role", "dialog");
    mapRegionFilterPanel.setAttribute("aria-modal", "false");
    mapRegionFilterPanel.setAttribute("aria-labelledby", "mapRegionFilterTitle");
    mapRegionFilterPanel.tabIndex = -1;
    mapRegionFilterPanel.innerHTML = `
      <div class="map-region-filter-header">
        <div class="map-region-filter-shortcuts" role="group" aria-label="Postcode region selection shortcuts">
          <button type="button" data-map-region-action="select-all">Select all</button>
          <button type="button" data-map-region-action="deselect-all">Deselect all</button>
        </div>
        <button class="map-region-filter-apply" type="button" data-map-region-action="apply" title="Apply postcode region filter" aria-label="Apply postcode region filter">
          ${MAP_FILTER_ICON_SVG}
        </button>
      </div>
      <div id="mapRegionFilterTitle" class="map-region-filter-title">Only show:</div>
      <div class="map-region-filter-options">${regionRows}</div>
    `;
    mapRegionFilterPanel.addEventListener("click", handleMapRegionFilterPanelClick);
    mapRegionFilterPanel.addEventListener("change", handleMapRegionFilterCheckboxChange);
    document.body.appendChild(mapRegionFilterPanel);
    return mapRegionFilterPanel;
  }

  function syncMapRegionFilterPanel() {
    if (!mapRegionFilterPanel || !stagedMapRegionAreas) return;
    mapRegionFilterPanel.querySelectorAll("[data-map-region-index]").forEach((checkbox) => {
      const region = UK_MAP_POSTCODE_REGIONS[Number(checkbox.dataset.mapRegionIndex)];
      if (!region) return;
      const selectedCount = region.areas.reduce(
        (count, area) => count + (stagedMapRegionAreas.has(area) ? 1 : 0),
        0,
      );
      checkbox.checked = selectedCount === region.areas.length;
      checkbox.indeterminate = selectedCount > 0 && selectedCount < region.areas.length;
    });
    const applyButton = mapRegionFilterPanel.querySelector('[data-map-region-action="apply"]');
    if (applyButton) applyButton.disabled = stagedMapRegionAreas.size === 0;
  }

  function positionMapRegionFilterPanel(clientX, clientY) {
    if (!mapRegionFilterPanel) return;
    const margin = 8;
    mapRegionFilterPanel.style.left = `${Math.round(clientX)}px`;
    mapRegionFilterPanel.style.top = `${Math.round(clientY)}px`;
    const bounds = mapRegionFilterPanel.getBoundingClientRect();
    const maxLeft = Math.max(margin, window.innerWidth - bounds.width - margin);
    const maxTop = Math.max(margin, window.innerHeight - bounds.height - margin);
    const left = Math.max(margin, Math.min(Math.round(clientX), maxLeft));
    const top = Math.max(margin, Math.min(Math.round(clientY), maxTop));
    mapRegionFilterPanel.style.left = `${left}px`;
    mapRegionFilterPanel.style.top = `${top}px`;
  }

  function closeMapRegionFilterPanel({ restoreFocus = false } = {}) {
    if (!mapRegionFilterPanel || mapRegionFilterPanel.hidden) return false;
    mapRegionFilterPanel.hidden = true;
    mapRegionFilterPanel.style.left = "";
    mapRegionFilterPanel.style.top = "";
    stagedMapRegionAreas = null;
    const returnFocus = mapRegionFilterReturnFocus;
    mapRegionFilterReturnFocus = null;
    if (restoreFocus && returnFocus?.isConnected && typeof returnFocus.focus === "function") {
      returnFocus.focus({ preventScroll: true });
    }
    return true;
  }

  function openMapRegionFilterPanel({ clientX, clientY, returnFocus } = {}) {
    if (state.tool !== "uk_map" || !mapContainerVisible()) return false;
    closeMapRegionFilterPanel();
    const panel = ensureMapRegionFilterPanel();
    const joinColumn = postcodeColumn("area");
    const selectedAreas = getMapSelectedAreas({ joinColumn });
    stagedMapRegionAreas = new Set(
      Array.isArray(selectedAreas)
        ? selectedAreas.filter((area) => UK_MAP_POSTCODE_AREA_CODE_SET.has(String(area)))
        : UK_MAP_POSTCODE_AREA_CODES,
    );
    mapRegionFilterReturnFocus = returnFocus || ukMap?.getContainer?.() || el("ukMap");
    panel.dataset.mapAreaColumn = joinColumn;
    panel.hidden = false;
    syncMapRegionFilterPanel();
    positionMapRegionFilterPanel(clientX, clientY);
    panel.focus({ preventScroll: true });
    return true;
  }

  function handleMapRegionFilterCheckboxChange(event) {
    const checkbox = event.target.closest?.("[data-map-region-index]");
    if (!checkbox || !stagedMapRegionAreas) return;
    const region = UK_MAP_POSTCODE_REGIONS[Number(checkbox.dataset.mapRegionIndex)];
    if (!region) return;
    region.areas.forEach((area) => {
      if (checkbox.checked) {
        stagedMapRegionAreas.add(area);
      } else {
        stagedMapRegionAreas.delete(area);
      }
    });
    syncMapRegionFilterPanel();
  }

  function handleMapRegionFilterPanelClick(event) {
    const button = event.target.closest?.("[data-map-region-action]");
    if (!button || !mapRegionFilterPanel?.contains(button) || !stagedMapRegionAreas) return;
    const action = button.dataset.mapRegionAction;
    if (action === "select-all") {
      stagedMapRegionAreas = new Set(UK_MAP_POSTCODE_AREA_CODES);
      syncMapRegionFilterPanel();
      return;
    }
    if (action === "deselect-all") {
      stagedMapRegionAreas.clear();
      syncMapRegionFilterPanel();
      return;
    }
    if (action !== "apply" || button.disabled || !stagedMapRegionAreas.size) return;
    const selectedAreas = Array.from(stagedMapRegionAreas).sort();
    const allSelected = selectedAreas.length === UK_MAP_POSTCODE_AREA_CODES.length;
    const joinColumn = String(mapRegionFilterPanel.dataset.mapAreaColumn || postcodeColumn("area"));
    closeMapRegionFilterPanel();
    const changed = Boolean(applyMapAreaGroupFilter({ joinColumn, selectedAreas, allSelected }));
    restyleActiveMapPolygonLayer();
    if (allSelected) {
      showClipboardToast(changed ? "Postcode region filter cleared" : "All postcode regions already shown");
    } else {
      showClipboardToast(changed ? "Postcode region filter applied" : "Postcode region filter unchanged");
    }
  }

  function mapRegionFilterIgnoresTarget(target) {
    return Boolean(target?.closest?.(
      "button, input, select, textarea, a, [contenteditable], .leaflet-control, .leaflet-popup, .map-floating-control",
    ));
  }

  function handleMapContextMenu(event) {
    if (state.tool !== "uk_map" || mapRegionFilterIgnoresTarget(event.target)) return;
    event.preventDefault();
    event.stopPropagation();
    ukMap?.closePopup?.();
    const activeElement = document.activeElement;
    openMapRegionFilterPanel({
      clientX: Number(event.clientX) || 0,
      clientY: Number(event.clientY) || 0,
      returnFocus: activeElement && el("ukMap").contains(activeElement)
        ? activeElement
        : ukMap?.getContainer?.(),
    });
  }

  function handleMapContextMenuKeydown(event) {
    const contextMenuKey = event.key === "ContextMenu" || (event.key === "F10" && event.shiftKey);
    if (!contextMenuKey || state.tool !== "uk_map" || mapRegionFilterIgnoresTarget(event.target)) return;
    event.preventDefault();
    event.stopPropagation();
    ukMap?.closePopup?.();
    const bounds = el("ukMap").getBoundingClientRect();
    openMapRegionFilterPanel({
      clientX: bounds.left + (bounds.width / 2),
      clientY: bounds.top + (bounds.height / 2),
      returnFocus: document.activeElement || ukMap?.getContainer?.(),
    });
  }

  function handleMapRegionFilterDocumentPointerDown(event) {
    if (!mapRegionFilterIsOpen() || mapRegionFilterPanel.contains(event.target)) return;
    mapRegionDismissClickPending = Boolean(
      event.button === 0
        && el("ukMap")?.contains(event.target)
        && !mapRegionFilterIgnoresTarget(event.target),
    );
    closeMapRegionFilterPanel();
  }

  function handleMapRegionFilterDocumentClick(event) {
    if (!mapRegionDismissClickPending) return;
    mapRegionDismissClickPending = false;
    if (!el("ukMap")?.contains(event.target) || mapRegionFilterIgnoresTarget(event.target)) return;
    event.preventDefault();
    event.stopImmediatePropagation();
  }

  function handleMapRegionFilterDocumentPointerEnd() {
    if (!mapRegionDismissClickPending) return;
    window.setTimeout(() => {
      mapRegionDismissClickPending = false;
    }, 0);
  }

  function handleMapRegionFilterDocumentKeydown(event) {
    if (!mapRegionFilterIsOpen() || event.key !== "Escape") return;
    event.preventDefault();
    event.stopPropagation();
    closeMapRegionFilterPanel({ restoreFocus: true });
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
    if (reason === "leaflet" && state.tool === "uk_map") clearActiveMapFavourite();
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
    const pendingRestoreView = state.mapViewRestorePending || null;
    let view = shouldPreserve ? pendingRestoreView || state.mapView : null;
    if (shouldPreserve && !pendingRestoreView && (state.mapStartupFitDone || state.mapView)) {
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
    const nextBaseMap = MAP_BASE_LAYERS[baseMap] ? baseMap : "blank";
    const sameBaseMap = nextBaseMap === state.baseMap;
    state.baseMap = nextBaseMap;
    if (!ukMap) return;
    const config = MAP_BASE_LAYERS[nextBaseMap];
    const tileLayerMatches = config.url
      ? Boolean(baseTileLayer && ukMap.hasLayer(baseTileLayer))
      : !baseTileLayer;
    if (sameBaseMap && tileLayerMatches) {
      syncBaseMapVisualState();
      syncMapControls();
      return;
    }
    if (baseTileLayer) {
      ukMap.removeLayer(baseTileLayer);
      baseTileLayer = null;
    }
    if (config.url) {
      baseTileLayer = L.tileLayer(config.url, {
        maxZoom: 19,
        attribution: config.attribution || "",
      }).addTo(ukMap);
      baseTileLayer.bringToBack();
    }
    syncBaseMapVisualState();
    syncMapControls();
  }

  function syncBaseMapVisualState() {
    if (!ukMap) return;
    const container = ukMap.getContainer();
    container._lucidumBaseMap = state.baseMap;
    container._lucidumBaseTileLayer = baseTileLayer;
    container.classList.toggle("blank-base", state.baseMap === "blank");
    applyMapBackground();
  }

  function canUseRenderedMapLevel(level) {
    if (!level || state.pendingMapZoom || state.renderedMapLevel !== level) return false;
    return Boolean(level === "unit" ? ukMapPointLayer : ukMapLayer);
  }

  function canUseCachedMapData(cache) {
    return canUseRenderedMapLevel(cache?.data?.level);
  }

  function canRefreshMapInPlace(request) {
    return canUseRenderedMapLevel(request?.level);
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

  function handleMapLayerControlChange(event) {
    const target = event.target;
    if (!target || target.tagName !== "INPUT") return;
    if (target.name === "baseMap") {
      setBaseMap(target.value);
      clearActiveMapFavourite({ force: true });
      scheduleMapViewportSync({ mode: "preserve" });
      return;
    }
    if (target.name === "mapLevel" && target.checked && mapLevelSelectable(target.value) && target.value !== state.mapLevel) {
      closeMapRegionFilterPanel();
      captureMapView("map-level-change");
      state.mapLevel = target.value;
      clearActiveMapFavourite({ force: true });
      syncMapControls();
      syncFloatingMapControl();
      refreshMap();
    }
  }

  function setMapLevel(level, options = {}) {
    const nextLevel = String(level || "");
    if (!Object.prototype.hasOwnProperty.call(MAP_LEVELS, nextLevel) || !mapLevelSelectable(nextLevel)) return false;
    const changed = state.mapLevel !== nextLevel;
    if (changed) {
      closeMapRegionFilterPanel();
      if (state.tool === "uk_map") captureMapView("map-level-change");
      state.mapLevel = nextLevel;
      if (options.clearFavourite !== false) clearActiveMapFavourite();
    }
    syncMapControls();
    syncFloatingMapControl();
    if (changed && options.refresh !== false && state.tool === "uk_map") {
      refreshMap(options.refreshOptions || {});
    }
    return true;
  }

  function clampMapNumber(value, fallback, min, max, options = {}) {
    const number = Number(value);
    const finite = Number.isFinite(number) ? number : fallback;
    const clamped = Math.max(min, Math.min(max, finite));
    return options.integer ? Math.round(clamped) : clamped;
  }

  function normaliseFavouriteMapLevel(level) {
    const requested = String(level || "");
    if (Object.prototype.hasOwnProperty.call(MAP_LEVELS, requested) && mapLevelSelectable(requested)) return requested;
    if (Object.prototype.hasOwnProperty.call(MAP_LEVELS, state.mapLevel) && mapLevelSelectable(state.mapLevel)) return state.mapLevel;
    return Object.keys(MAP_LEVELS).find((candidate) => mapLevelSelectable(candidate)) || "area";
  }

  function normaliseFavouriteMapState(map = {}) {
    const payload = map && typeof map === "object" ? map : {};
    const level = normaliseFavouriteMapLevel(payload.level);
    const baseMap = MAP_BASE_LAYERS[payload.baseMap] ? String(payload.baseMap) : "blank";
    const palette = MAP_PALETTES[payload.palette] ? String(payload.palette) : "divergent";
    return {
      level,
      baseMap,
      palette,
      lineWeight: clampMapNumber(payload.lineWeight, 1, 0, 10, { integer: true }),
      dotSize: clampMapNumber(payload.dotSize, 1, 1, 10, { integer: true }),
      opacity: clampMapNumber(payload.opacity, 1, 0, 1),
      hotspots: clampMapNumber(payload.hotspots, 0, -9, 9, { integer: true }),
      labelSize: clampMapNumber(payload.labelSize, 0, 0, 10, { integer: true }),
      smoothingLevel: clampMapNumber(payload.smoothingLevel, 0, 0, 5, { integer: true }),
      view: normaliseMapView({ center: payload.center, zoom: payload.zoom }),
    };
  }

  function captureFavouriteState() {
    const view = normaliseMapView(currentMapView() || state.mapView) || normaliseMapView(MAP_DEFAULT_VIEW);
    if (view) state.mapView = view;
    return {
      level: state.mapLevel,
      baseMap: state.baseMap,
      palette: state.mapPalette,
      lineWeight: Number(state.mapLineWeight),
      dotSize: Number(state.mapDotSize),
      opacity: Number(state.mapOpacity),
      hotspots: Number(state.mapHotspots),
      labelSize: Number(state.mapLabelSize),
      smoothingLevel: Number(state.mapSmoothingLevel),
      center: view?.center || null,
      zoom: view?.zoom ?? null,
    };
  }

  function applyFavouriteState(map = {}) {
    const next = normaliseFavouriteMapState(map);
    state.mapLevel = next.level;
    state.mapPalette = next.palette;
    state.mapLineWeight = next.lineWeight;
    state.mapDotSize = next.dotSize;
    state.mapOpacity = next.opacity;
    state.mapHotspots = next.hotspots;
    state.mapLabelSize = next.labelSize;
    state.mapSmoothingLevel = next.smoothingLevel;
    state.mapView = next.view;
    state.mapViewRestorePending = next.view;
    state.pendingMapZoom = null;
    state.mapStartupFitDone = Boolean(next.view);
    setBaseMap(next.baseMap);
    syncMapControls();
    syncFloatingMapControl();
    return next;
  }

  function syncMapControls() {
    document.querySelectorAll('input[name="baseMap"]').forEach((input) => {
      input.checked = input.value === state.baseMap;
    });
    document.querySelectorAll('input[name="mapLevel"]').forEach((input) => {
      input.disabled = !mapLevelSelectable(input.value);
      input.checked = input.value === state.mapLevel;
    });
  }

  function zoomMapBy(delta) {
    if (!ukMap) return;
    ukMap.setZoom(ukMap.getZoom() + delta, { animate: false });
    state.mapStartupFitDone = true;
    captureMapView("explicit");
  }

  function zoomMapToLondon() {
    if (!ukMap) return;
    ukMap.setView([51.5074, -0.1278], 10, { animate: false });
    state.mapStartupFitDone = true;
    captureMapView("explicit");
  }

  function addMapViewportControl() {
    if (!ukMap || mapViewportControl) return;
    const ViewportControl = L.Control.extend({
      options: { position: "topleft" },
      onAdd() {
        const container = L.DomUtil.create("div", "map-viewport-control leaflet-control");
        container.innerHTML = `
          <button id="mapZoomIn" class="map-viewport-button" type="button" title="Zoom in" aria-label="Zoom in">+</button>
          <button id="mapZoomOut" class="map-viewport-button" type="button" title="Zoom out" aria-label="Zoom out">&minus;</button>
          <button id="mapFitUk" class="map-viewport-button" type="button" title="Fit UK map layer" aria-label="Fit UK map layer">
            <img src="/tools/uk-map/static/icons/UK.png" alt="">
          </button>
          <button id="mapZoomLondon" class="map-viewport-button" type="button" title="Zoom to London" aria-label="Zoom to London">
            <img class="map-viewport-icon-london" src="/tools/uk-map/static/icons/London.png" alt="">
          </button>
        `;
        L.DomEvent.disableClickPropagation(container);
        L.DomEvent.disableScrollPropagation(container);
        container.querySelector("#mapZoomIn").addEventListener("click", () => zoomMapBy(Number(ukMap?.options?.zoomDelta) || 1));
        container.querySelector("#mapZoomOut").addEventListener("click", () => zoomMapBy(-(Number(ukMap?.options?.zoomDelta) || 1)));
        container.querySelector("#mapFitUk").addEventListener("click", () => fitMapToLayer());
        container.querySelector("#mapZoomLondon").addEventListener("click", () => zoomMapToLondon());
        return container;
      },
    });
    mapViewportControl = new ViewportControl();
    mapViewportControl.addTo(ukMap);
  }

  function fitMapToLayer(options = {}) {
    const bounds = activeMapBounds();
    if (!bounds) {
      if (!ukMap) return;
      ukMap.setView([MAP_DEFAULT_VIEW.center.lat, MAP_DEFAULT_VIEW.center.lng], MAP_DEFAULT_VIEW.zoom, { animate: false });
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

  function activeMapExtremeColors() {
    const palette = activeMapPalette();
    return {
      low: palette[0] || MAP_MISSING_COLOR,
      high: palette[palette.length - 1] || MAP_MISSING_COLOR,
    };
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

  function normaliseMapHotspotNotch(value = state.mapHotspots) {
    const number = Number(value);
    if (!Number.isFinite(number)) return 0;
    return Math.max(-9, Math.min(9, Math.round(number)));
  }

  function mapHotspotSelection(value = state.mapHotspots) {
    const notch = normaliseMapHotspotNotch(value);
    if (notch === 0) return null;
    return {
      direction: notch > 0 ? -1 : 1,
      fraction: mapHotspotPercent(notch) / 100,
    };
  }

  function mapHotspotPercent(value = state.mapHotspots) {
    const notch = normaliseMapHotspotNotch(value);
    if (notch === 0) return 0;
    return 100 - (Math.abs(notch) * 10);
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

  function sectorLineWeightScaleForZoom(zoom) {
    const value = Number(zoom);
    if (!Number.isFinite(value)) return 1;
    if (value <= 6) return 0.15;
    if (value <= 7) return 0.25;
    if (value <= 8) return 0.4;
    if (value <= 9) return 0.65;
    if (value <= 10) return 0.85;
    return 1;
  }

  function mapStrokeWeightForLineWeight(value = state.mapLineWeight) {
    const sliderValue = Number(value);
    if (!Number.isFinite(sliderValue) || sliderValue <= 0) return 0;
    return Math.min(10, sliderValue) / 2;
  }

  function mapLineWeightForLevel(level) {
    const baseWeight = mapStrokeWeightForLineWeight();
    if (baseWeight <= 0) return 0;
    if (level !== "sector" || !ukMap) return baseWeight;
    return baseWeight * sectorLineWeightScaleForZoom(ukMap.getZoom());
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
    activeMapPopupSelection = {
      key: key || "Unknown",
      level: data?.level || state.mapLevel,
      joinColumn: data?.join_column || postcodeColumn(data?.level || state.mapLevel),
      layer,
      popup: layer.getPopup?.() || null,
    };
    return mapPopupHtml(key || "Unknown", row, data || {});
  }

  function mapPolygonFeatureStyle(feature) {
    const context = activeMapPolygonContext();
    if (!context) return mapFeatureStyle(null, makeQuantileScale([]), null);
    const key = mapPolygonFeatureKey(feature, context.joinProperty);
    const row = context.summaries.get(key);
    const style = mapFeatureStyle(row, context.scale, context.hotspotKeys, context.data.level);
    const selected = context.data.level === "area" && isMapPostcodeSelected({
      key,
      level: "area",
      joinColumn: context.data.join_column || postcodeColumn("area"),
    });
    if (!selected) return style;
    return {
      ...style,
      color: getCss("--accent") || "#2276d2",
      opacity: 1,
    };
  }

  function mapPostcodeSelectionForPolygon(level, layer) {
    const key = mapPolygonLayerKey(layer);
    return {
      key,
      level,
      joinColumn: activeMapPolygonContext()?.data?.join_column || postcodeColumn(level),
    };
  }

  function syncActiveAreaFilterButton(selected, postcode) {
    const button = el("ukMap").querySelector('.map-popup-action[data-map-popup-action="filter"][aria-pressed]');
    const key = String(postcode?.key || activeMapPopupSelection?.key || "");
    if (!button || !key) return;
    const label = `${selected ? "Remove" : "Add"} ${key} ${selected ? "from" : "to"} postcode area filter`;
    button.classList.toggle("map-popup-action--active", selected);
    button.setAttribute("aria-pressed", String(selected));
    button.setAttribute("aria-label", label);
    button.title = label;
  }

  function toggleMapAreaSelection(postcode, options = {}) {
    if (postcode?.level !== "area" || !postcode.key || !postcode.joinColumn) return false;
    const wasSelected = Boolean(isMapPostcodeSelected(postcode));
    applyMapPostcodeFilter(postcode);
    const selected = !wasSelected;
    restyleActiveMapPolygonLayer();
    syncActiveAreaFilterButton(selected, postcode);
    if (options.feedback !== false) {
      showClipboardToast(`${selected ? "Added" : "Removed"} ${postcode.key} ${selected ? "to" : "from"} postcode area filter`);
    }
    return selected;
  }

  function createMapPolygonLayer(level, geoJson) {
    const levelConfig = MAP_LEVELS[level] || MAP_LEVELS.area;
    return L.geoJSON(geoJson, {
      smoothFactor: levelConfig.smoothFactor ?? 1,
      style: mapPolygonFeatureStyle,
      onEachFeature: (feature, layer) => {
        layer._lucidumMapKey = mapPolygonFeatureKey(feature, levelConfig.property);
        layer.bindTooltip(() => mapPolygonTooltipHtml(layer), { sticky: true });
        layer._lucidumPopupContent = () => mapPolygonPopupHtml(layer);
        layer.bindPopup(layer._lucidumPopupContent, { maxWidth: MAP_POPUP_MAX_WIDTH });
      },
    });
  }

  function cachedMapPolygonLayer(level, geoJson) {
    if (!state.mapPolygonLayerCache[level]) {
      const levelConfig = MAP_LEVELS[level] || MAP_LEVELS.area;
      state.mapPolygonLayerCache[level] = {
        layer: createMapPolygonLayer(level, geoJson),
        featureCount: geoJson.features?.length || 0,
        shapeKeys: new Set(
          (geoJson.features || []).map((feature) => mapPolygonFeatureKey(feature, levelConfig.property)),
        ),
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
    if (activeMapPolygonContext()?.data?.level !== "area") return;
    ukMapLayer.eachLayer((layer) => {
      const postcode = mapPostcodeSelectionForPolygon("area", layer);
      if (isMapPostcodeSelected(postcode)) layer.bringToFront?.();
    });
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

  function unitPointRadiusForCurrentStyle(zoom) {
    const sliderValue = Math.max(1, Math.min(10, Number(state.mapDotSize)));
    if (!Number.isFinite(sliderValue) || sliderValue <= 1) return MAP_UNIT_POINT_MIN_RADIUS;
    const maxRadius = unitPointRadiusForZoom(zoom) * MAP_UNIT_POINT_MAX_RADIUS_MULTIPLIER;
    const progress = (sliderValue - 1) / 9;
    return MAP_UNIT_POINT_MIN_RADIUS + ((maxRadius - MAP_UNIT_POINT_MIN_RADIUS) * progress);
  }

  function unitPointHitRadius(radius) {
    return Math.max(radius + 4, 6);
  }

  function mapPointStyle(row, scale, hotspotKeys, radius) {
    const value = finiteNumber(row?.value);
    const selected = value !== null && (!hotspotKeys || hotspotKeys.has(String(row.key)));
    const muted = value !== null && !selected;
    const opacityValue = Number(state.mapOpacity);
    const mapOpacity = Number.isFinite(opacityValue) ? Math.max(0, Math.min(1, opacityValue)) : 1;
    const baseStrokeOpacity = radius < 2 ? 0 : (radius < 3 ? 0.35 : 0.65);
    const strokeOpacity = (muted ? Math.min(baseStrokeOpacity, 0.25) : baseStrokeOpacity) * mapOpacity;
    return {
      fillColor: muted ? MAP_MUTED_COLOR : scale.color(value),
      fillOpacity: muted ? Math.min(mapOpacity, 0.28) : mapOpacity,
      strokeOpacity,
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
        const pointRadius = unitPointRadiusForCurrentStyle(this.map.getZoom());
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
        const hitRadius = this.hitRadius || unitPointHitRadius(unitPointRadiusForCurrentStyle(this.map.getZoom()));
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
        const popup = L.popup({ maxWidth: MAP_POPUP_MAX_WIDTH })
          .setLatLng(nearest.latLng)
          .setContent(mapPopupHtml(String(nearest.key || "Unknown"), nearest, this.data));
        activeMapPopupSelection = {
          key: String(nearest.key || "Unknown"),
          level: "unit",
          joinColumn: this.data?.join_column || postcodeColumn("unit"),
          latLng: nearest.latLng,
          popup,
        };
        popup.openOn(this.map);
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

  function refreshOpenMapPopup(data) {
    const selection = activeMapPopupSelection;
    const popup = ukMap?._popup;
    if (!selection || !popup || !ukMap.hasLayer(popup) || selection.level !== data?.level) return false;
    let row = null;
    if (data.level === "unit") {
      row = ukMapPointLayer?.rows?.find((candidate) => String(candidate.key) === selection.key) || null;
    } else {
      row = activeMapPolygonContext()?.summaries?.get(selection.key) || null;
    }
    selection.joinColumn = data.join_column || postcodeColumn(data.level);
    if (data.level !== "unit" && selection.layer?._lucidumPopupContent) {
      popup.setContent(selection.layer._lucidumPopupContent);
    } else {
      popup.setContent(mapPopupHtml(selection.key, row, data));
    }
    popup.update?.();
    return true;
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
      const restoreView = state.mapViewRestorePending || state.mapView;
      if (restoreView) {
        const restored = restoreMapView(restoreView);
        if (restored && state.mapViewRestorePending) state.mapViewRestorePending = null;
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
    const cachedPolygonLayer = cachedMapPolygonLayer(data.level, geoJson);
    const matchSummary = ukMapShapefileMatchSummary({
      level: data.level,
      rows: data.rows || [],
      filteredRowCount: data.filtered_row_count,
      shapeKeys: cachedPolygonLayer.shapeKeys,
    });
    const scale = makeQuantileScale(matchSummary.matchedRows);
    const hotspotKeys = mapHotspotKeys(matchSummary.matchedRows);
    state.mapPolygonRenderContext = {
      data,
      joinProperty: data.join_property,
      summaries,
      scale,
      hotspotKeys,
    };
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
    if (data.level === "area") renderMapLabels(data, summaries, hotspotKeys);

    const searchWarning = applyRenderedMapCamera(data.level, ukMapLayer.getBounds());
    renderMapLegend(scale, data.response?.label || "Actual");
    const rowMeta = mapRowMetaForData(data);
    const smoothingLevel = Number(data.smoothing?.level) || 0;
    const groupMeta = data.level === "sector" && smoothingLevel > 0
      ? `${formatSmoothingLevel(smoothingLevel)} sector smoothing`
      : `${matchedFeatureCount.toLocaleString()} / ${featureCount.toLocaleString()} ${levelConfig.label} matched`;
    const matchWarningClass = matchSummary.matchState === "warning"
      ? " map-shapefile-match-status--warning"
      : "";
    const missingWarningClass = matchSummary.missingRowCount > 0
      ? " map-shapefile-match-status--warning"
      : "";
    const missingStatusHtml = `<div class="map-shapefile-missing-status${missingWarningClass}">${escapeHtml(matchSummary.missingText)}</div>`;
    const groupMetaHtml = `
      <div class="map-group-meta-count">${escapeHtml(groupMeta)}</div>
      <div class="map-shapefile-match-status${matchWarningClass}">${escapeHtml(matchSummary.matchText)}</div>
      ${missingStatusHtml}
    `;
    setGroupMeta("uk_map", groupMetaHtml, { html: true });
    setMapRowMeta(rowMeta);
    setMapMatchLiveStatus(
      [groupMeta, matchSummary.matchText, matchSummary.missingText].filter(Boolean).join(". "),
    );
    const warnings = [...(data.warnings || [])];
    if (searchWarning) {
      warnings.push(searchWarning);
    }
    if (matchedFeatureCount === 0 && (data.rows || []).length) {
      warnings.push(`No ${levelConfig.label} matched the GeoJSON ${levelConfig.property} values.`);
    }
    const chartMessage = warnings.filter(Boolean).join(" ");
    setChartMessage(chartMessage);
    saveToolPresentation("uk_map", { groupMeta, groupMetaHtml, chartMessage });
    refreshOpenMapPopup(data);
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
    const rowMeta = mapRowMetaForData(data);
    const pointSummary = data.point_summary || {};
    const summaryCount = Number(pointSummary.summary_count ?? unitPointCount(data));
    const plottedCount = Number(pointSummary.plotted_count ?? unitPointCount(data));
    const missingValueCount = Number(pointSummary.missing_value_count || 0);
    const missingCoordinateCount = Number(pointSummary.missing_coordinate_count || 0);
    const groupMeta = `${plottedCount.toLocaleString()} / ${summaryCount.toLocaleString()} units plotted`;
    const missingValueText = missingValueCount > 0
      ? `${missingValueCount.toLocaleString()} ${missingValueCount === 1 ? "unit" : "units"} missing KPI value`
      : "No units missing KPI value";
    const missingCoordinateText = missingCoordinateCount > 0
      ? `${missingCoordinateCount.toLocaleString()} ${missingCoordinateCount === 1 ? "unit" : "units"} missing coordinates`
      : "No units missing coordinates";
    const missingValueWarningClass = missingValueCount > 0
      ? " map-shapefile-match-status--warning"
      : "";
    const missingCoordinateWarningClass = missingCoordinateCount > 0
      ? " map-shapefile-match-status--warning"
      : "";
    const groupMetaHtml = `
      <div class="map-group-meta-count">${escapeHtml(groupMeta)}</div>
      <div class="map-unit-status${missingValueWarningClass}">${escapeHtml(missingValueText)}</div>
      <div class="map-unit-status${missingCoordinateWarningClass}">${escapeHtml(missingCoordinateText)}</div>
    `;
    setGroupMeta("uk_map", groupMetaHtml, { html: true });
    setMapRowMeta(rowMeta);
    setMapMatchLiveStatus([groupMeta, missingValueText, missingCoordinateText].join(". "));
    const warnings = [...(data.warnings || [])];
    if (missingValueCount) {
      warnings.push(`${missingValueCount.toLocaleString()} ${missingValueCount === 1 ? "unit has" : "units have"} no plottable KPI value.`);
    }
    if (missingCoordinateCount) {
      warnings.push(`${missingCoordinateCount.toLocaleString()} ${missingCoordinateCount === 1 ? "unit has" : "units have"} no valid coordinates.`);
    }
    const chartMessage = warnings.filter(Boolean).join(" ");
    setChartMessage(chartMessage);
    saveToolPresentation("uk_map", { groupMeta, groupMetaHtml, chartMessage });
    refreshOpenMapPopup(data);
    scheduleMapViewportSync({ mode: "preserve" });
  }

  function mapLabelFontSize(value = state.mapLabelSize) {
    const number = Number(value);
    if (!Number.isFinite(number)) return 0;
    const sliderValue = Math.max(0, Math.min(10, Math.round(number)));
    if (sliderValue <= 0) return 0;
    return MAP_LABEL_MIN_FONT_SIZE + (((sliderValue - 1) / 9) * (MAP_LABEL_MAX_FONT_SIZE - MAP_LABEL_MIN_FONT_SIZE));
  }

  function renderMapLabels(data, summaries, hotspotKeys) {
    const fontSize = mapLabelFontSize(state.mapLabelSize);
    if (data.level !== "area" || !Number.isFinite(fontSize) || fontSize <= 0 || !ukMapLayer) return;
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
          className: "map-label-icon",
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

  function updateMapSliderProgress(input) {
    if (!input) return;
    const min = Number(input.min || 0);
    const max = Number(input.max || 100);
    const value = Number(input.value || 0);
    const span = max - min;
    const progress = span ? ((value - min) / span) * 100 : 0;
    input.style.setProperty("--map-slider-progress", `${Math.max(0, Math.min(100, progress))}%`);
    if (input.id === "mapHotspots") {
      input.classList.toggle("map-slider-thumb-centered", value === 0);
    }
  }

  function syncMapSliderProgressStyles() {
    ["mapLineWeight", "mapDotSize", "mapOpacity", "mapHotspots", "mapLabelSize", "mapSmoothing"]
      .forEach((id) => updateMapSliderProgress(el(id)));
  }

  function syncMapExtremeLabels() {
    const colors = activeMapExtremeColors();
    const lowLabel = el("mapHotspotsMinLabel");
    const highLabel = el("mapHotspotsMaxLabel");
    lowLabel.textContent = "Low";
    highLabel.textContent = "High";
    lowLabel.style.setProperty("--map-extreme-color", colors.low);
    highLabel.style.setProperty("--map-extreme-color", colors.high);
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
    const unitMode = state.mapLevel === "unit";
    el("mapLineWeight").value = String(state.mapLineWeight);
    el("mapDotSize").value = String(state.mapDotSize);
    el("mapOpacity").value = String(opacitySliderValue(state.mapOpacity));
    el("mapHotspots").value = String(state.mapHotspots);
    el("mapLabelSize").value = String(state.mapLabelSize);
    el("mapSmoothing").value = String(state.mapSmoothingLevel);
    el("mapLineWeightValue").textContent = String(state.mapLineWeight);
    el("mapDotSizeValue").textContent = String(state.mapDotSize);
    el("mapOpacityValue").textContent = formatOpacitySliderValue(state.mapOpacity);
    el("mapHotspotsValue").textContent = formatHotspotSliderValue(state.mapHotspots);
    syncMapExtremeLabels();
    el("mapLabelSizeValue").textContent = String(state.mapLabelSize);
    el("mapSmoothingValue").textContent = formatSmoothingLevel(state.mapSmoothingLevel);
    el("mapSliderGrid").classList.toggle("unit-mode", unitMode);
    const lineWeightControl = el("mapLineWeightControl") || el("mapLineWeight").closest(".map-slider-control");
    if (lineWeightControl) lineWeightControl.hidden = unitMode;
    el("mapLineWeight").disabled = unitMode;
    lineWeightControl?.classList.toggle("disabled", unitMode);
    const dotSizeControl = el("mapDotSizeControl") || el("mapDotSize").closest(".map-slider-control");
    if (dotSizeControl) dotSizeControl.hidden = !unitMode;
    el("mapDotSize").disabled = !unitMode;
    dotSizeControl?.classList.toggle("disabled", !unitMode);
    const labelHidden = state.mapLevel !== "area";
    const labelControl = el("mapLabelControl") || el("mapLabelSize").closest(".map-slider-control");
    if (labelControl) labelControl.hidden = labelHidden;
    el("mapLabelSize").disabled = labelHidden;
    labelControl?.classList.toggle("disabled", labelHidden);
    const smoothingHidden = state.mapLevel !== "sector";
    const smoothingControl = el("mapSmoothingControl") || el("mapSmoothing").closest(".map-slider-control");
    if (smoothingControl) smoothingControl.hidden = smoothingHidden;
    el("mapSmoothing").disabled = smoothingHidden;
    smoothingControl?.classList.toggle("disabled", smoothingHidden);
    syncMapSliderProgressStyles();
  }

  function formatCompactSliderValue(value) {
    const number = Number(value);
    if (!Number.isFinite(number)) return "";
    return Number.isInteger(number) ? String(number) : String(Number(number.toFixed(1)));
  }

  function formatOpacitySliderValue(value) {
    return String(opacitySliderValue(value));
  }

  function opacitySliderValue(value = state.mapOpacity) {
    const number = Math.max(0, Math.min(1, Number(value) || 0));
    return Math.round(number * 10);
  }

  function opacityFromSliderValue(value) {
    const number = Math.max(0, Math.min(10, Number(value) || 0));
    return number / 10;
  }

  function formatHotspotSliderValue(value) {
    const notch = normaliseMapHotspotNotch(value);
    if (notch === 0) return "All";
    return `${notch < 0 ? "B" : "T"}${mapHotspotPercent(notch)}`;
  }

  function formatSmoothingLevel(value) {
    const level = Math.max(0, Math.min(5, Math.round(Number(value) || 0)));
    return level <= 0 ? "None" : `N${level}`;
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
      syncFloatingMapControl();
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
    const postcode = {
      key: String(title || ""),
      level: data?.level || state.mapLevel,
      joinColumn: data?.join_column || postcodeColumn(data?.level || state.mapLevel),
    };
    const areaFilterToggle = postcode.level === "area";
    return ukMapPopupContentHtml({
      title,
      row,
      data,
      escapeHtml,
      formatNumber,
      formatLineValue,
      showViewRows: canOpenDatasetViewer(),
      areaFilterToggle,
      areaFilterSelected: areaFilterToggle && isMapPostcodeSelected(postcode),
      postcodeLevel: postcode.level,
      postcodeJoinColumn: postcode.joinColumn,
    });
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
    const legendBody = el("mapLegendBody");
    if (state.tool !== "uk_map" || !scale.values.length) {
      legend.classList.add("hidden");
      legendBody.innerHTML = "";
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
      rows.push(`<div class="map-legend-row"><span class="map-swatch" style="background:${MAP_MUTED_COLOR}"></span><span>Hidden</span></div>`);
    }
    rows.push(`<div class="map-legend-row"><span class="map-swatch" style="background:${MAP_MISSING_COLOR}"></span><span>No data</span></div>`);
    legendBody.innerHTML = rows.join("");
    syncMapLegendCollapseButton();
    legend.classList.remove("hidden");
  }

  function mapLegendLabel(lower, upper, isLast) {
    if (lower === null && upper === null) return "All values";
    if (lower === null) return `≤ ${formatLineValue(upper)}`;
    if (upper === null || isLast) return `> ${formatLineValue(lower)}`;
    return `${formatLineValue(lower)}–${formatLineValue(upper)}`;
  }

  function syncMapLegendCollapseButton() {
    const collapsed = Boolean(state.mapLegendCollapsed);
    const legend = el("mapLegend");
    const button = el("mapLegendToggle");
    const label = collapsed ? "Expand legend" : "Collapse legend";
    legend.classList.toggle("collapsed", collapsed);
    button.title = label;
    button.setAttribute("aria-label", label);
    button.setAttribute("aria-expanded", String(!collapsed));
  }

  function toggleMapLegendCollapsed() {
    state.mapLegendCollapsed = !state.mapLegendCollapsed;
    syncMapLegendCollapseButton();
  }

  function setupMapFloatingControlDrag() {
    const panel = el("mapFloatingControl");
    const dragThreshold = 3;

    let dragging = false;
    let dragMoved = false;
    let startX = 0;
    let startY = 0;
    let startLeft = 0;
    let startTop = 0;
    panel.addEventListener("pointerdown", (event) => {
      if (state.mapControlCollapsed || event.button !== 0 || isMapFloatingInteractiveTarget(event.target)) return;
      event.preventDefault();
      dragging = true;
      dragMoved = false;
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
      const deltaX = event.clientX - startX;
      const deltaY = event.clientY - startY;
      if (!dragMoved && Math.hypot(deltaX, deltaY) < dragThreshold) return;
      dragMoved = true;
      state.mapControlMoved = true;
      setMapFloatingPosition(startLeft + deltaX, startTop + deltaY);
    });
    function finishDrag(event) {
      if (!dragging) return;
      dragging = false;
      panel.classList.remove("dragging");
      document.body.classList.remove("dragging-map-control");
      window.getSelection()?.removeAllRanges();
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

  function setMapFloatingPosition(rawLeft, rawTop, { updateState = true } = {}) {
    const panel = el("mapFloatingControl");
    const frame = mapFloatingPositionFrame();
    if (!frame) return null;
    const margin = 8;
    const topMargin = 4;
    const maxLeft = Math.max(margin, frame.width - panel.offsetWidth - margin);
    const maxTop = Math.max(topMargin, frame.height - panel.offsetHeight - margin);
    const left = Math.min(Math.max(rawLeft, margin), maxLeft);
    const top = Math.min(Math.max(rawTop, topMargin), maxTop);
    panel.style.left = `${left}px`;
    panel.style.top = `${top}px`;
    panel.style.right = "auto";
    if (updateState) state.mapControlPosition = { left, top };
    return { left, top };
  }

  function clampMapFloatingControl() {
    const panel = el("mapFloatingControl");
    if (state.mapControlCollapsed) {
      positionCollapsedMapFloatingControlTopRight();
      return;
    }
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

  function positionCollapsedMapFloatingControlTopRight() {
    const position = mapFloatingTopRightButtonPosition();
    if (position) setMapFloatingCollapsedPosition(position.left, position.top);
  }

  function positionMapFloatingControlTopRight() {
    const position = mapFloatingTopRightPanelPosition();
    if (position) {
      setMapFloatingPosition(position.left, position.top, { updateState: false });
    }
    state.mapControlPosition = null;
  }

  function mapFloatingPositionFrame() {
    const panel = el("mapFloatingControl");
    const container = panel.offsetParent || panel.closest(".workspace");
    const rect = container?.getBoundingClientRect();
    if (!container || !rect) return null;
    return {
      left: rect.left + container.clientLeft,
      top: rect.top + container.clientTop,
      width: container.clientWidth,
      height: container.clientHeight,
    };
  }

  function mapFloatingButtonPosition() {
    const frame = mapFloatingPositionFrame();
    if (!frame) return null;
    const buttonRect = el("mapControlReset").getBoundingClientRect();
    return {
      left: buttonRect.left - frame.left,
      top: buttonRect.top - frame.top,
    };
  }

  function mapFloatingButtonOffset() {
    const panelRect = el("mapFloatingControl").getBoundingClientRect();
    const buttonRect = el("mapControlReset").getBoundingClientRect();
    return {
      left: buttonRect.left - panelRect.left,
      top: buttonRect.top - panelRect.top,
    };
  }

  function mapFloatingTopRightPanelPosition() {
    const panel = el("mapFloatingControl");
    const frame = mapFloatingPositionFrame();
    if (!frame) return null;
    const margin = 8;
    const topMargin = 4;
    return {
      left: Math.max(margin, frame.width - panel.offsetWidth - margin),
      top: topMargin,
    };
  }

  function mapFloatingTopRightButtonPosition() {
    const panel = el("mapFloatingControl");
    if (!panel) return null;
    const wasCollapsed = panel.classList.contains("collapsed");
    const previous = {
      left: panel.style.left,
      top: panel.style.top,
      right: panel.style.right,
    };
    if (wasCollapsed) panel.classList.remove("collapsed");
    const panelPosition = mapFloatingTopRightPanelPosition();
    let buttonPosition = null;
    if (panelPosition) {
      setMapFloatingPosition(panelPosition.left, panelPosition.top, { updateState: false });
      const buttonOffset = mapFloatingButtonOffset();
      buttonPosition = {
        left: panelPosition.left + buttonOffset.left,
        top: panelPosition.top + buttonOffset.top,
      };
    }
    panel.style.left = previous.left;
    panel.style.top = previous.top;
    panel.style.right = previous.right;
    if (wasCollapsed) panel.classList.add("collapsed");
    return buttonPosition;
  }

  function setMapFloatingCollapsedPosition(rawLeft, rawTop) {
    const position = setMapFloatingPosition(rawLeft, rawTop, { updateState: false });
    if (position) state.mapControlCollapsedPosition = position;
  }

  function syncMapControlCollapseButton() {
    const button = el("mapControlReset");
    const collapsed = Boolean(state.mapControlCollapsed);
    const label = collapsed ? "Expand map controls" : "Collapse map controls";
    button.title = label;
    button.setAttribute("aria-label", label);
    button.setAttribute("aria-expanded", String(!collapsed));
    button.innerHTML = `<svg viewBox="0 0 24 24" aria-hidden="true" focusable="false">${collapsed ? MAP_CONTROL_COLLAPSED_ICON : MAP_CONTROL_EXPANDED_ICON}</svg>`;
  }

  function syncMapFloatingControlCollapsedState() {
    el("mapFloatingControl").classList.toggle("collapsed", Boolean(state.mapControlCollapsed));
    syncMapControlCollapseButton();
  }

  function collapseMapFloatingControl() {
    const panelPosition = mapFloatingTopRightPanelPosition();
    if (!panelPosition) return;
    state.mapControlMoved = false;
    state.mapControlPosition = null;
    state.mapControlCollapsed = true;
    el("mapFloatingControl").classList.add("collapsed");
    syncMapControlCollapseButton();
    positionCollapsedMapFloatingControlTopRight();
  }

  function expandMapFloatingControl() {
    const buttonPosition = mapFloatingButtonPosition();
    if (!buttonPosition) return;
    state.mapControlCollapsed = false;
    state.mapControlCollapsedPosition = null;
    el("mapFloatingControl").classList.remove("collapsed");
    syncMapControlCollapseButton();
    if (!state.mapControlMoved) {
      positionMapFloatingControlTopRight();
      return;
    }
    const buttonOffset = mapFloatingButtonOffset();
    setMapFloatingPosition(buttonPosition.left - buttonOffset.left, buttonPosition.top - buttonOffset.top);
  }

  function toggleMapFloatingControlCollapsed() {
    if (state.mapControlCollapsed) {
      expandMapFloatingControl();
    } else {
      collapseMapFloatingControl();
    }
  }

  function bindMapFloatingControls() {
    syncMapFloatingControlCollapsedState();
    syncMapLegendCollapseButton();
    el("mapControlReset").addEventListener("click", toggleMapFloatingControlCollapsed);
    el("mapLegendToggle").addEventListener("click", toggleMapLegendCollapsed);
    el("mapBaseLayerTiles").addEventListener("change", handleMapLayerControlChange);
    el("mapLevelTiles").addEventListener("change", handleMapLayerControlChange);
    document.querySelectorAll(".map-palette-button").forEach((button) => {
      button.addEventListener("click", () => {
        state.mapPalette = button.dataset.palette || "viridis";
        clearActiveMapFavourite({ force: true });
        redrawMapInPlace();
      });
    });
    [
      ["mapLineWeight", "mapLineWeight"],
      ["mapDotSize", "mapDotSize"],
      ["mapOpacity", "mapOpacity"],
      ["mapHotspots", "mapHotspots"],
      ["mapLabelSize", "mapLabelSize"],
    ].forEach(([id, stateKey]) => {
      el(id).addEventListener("input", (event) => {
        state[stateKey] = id === "mapOpacity" ? opacityFromSliderValue(event.target.value) : Number(event.target.value);
        updateMapSliderProgress(event.target);
        clearActiveMapFavourite({ force: true });
        if (
          (id === "mapLineWeight" && state.mapLevel === "unit")
          || (id === "mapDotSize" && state.mapLevel !== "unit")
          || (id === "mapLabelSize" && state.mapLevel !== "area")
        ) {
          syncFloatingMapControl();
          return;
        }
        redrawMapInPlace();
      });
    });
    el("mapSmoothing").addEventListener("input", (event) => {
      state.mapSmoothingLevel = Number(event.target.value);
      updateMapSliderProgress(event.target);
      clearActiveMapFavourite({ force: true });
      syncFloatingMapControl();
      if (state.mapLevel !== "sector") return;
      captureMapView("smoothing-change");
      refreshMap();
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
    syncMapFloatingControlCollapsedState();
    syncFloatingMapControl();
    syncMapControls();
    clampMapFloatingControl();
    requestAnimationFrame(() => {
      clampMapFloatingControl();
      scheduleMapViewportSync({ mode: "preserve" });
    });
  }

  async function handleMapPopupAction(event) {
    const button = event.target.closest?.("[data-map-popup-action]");
    if (!button || !el("ukMap").contains(button)) return;
    const popupContent = button.closest(".map-popup");
    const postcode = {
      key: String(popupContent?.dataset.mapPostcodeKey || activeMapPopupSelection?.key || ""),
      level: String(popupContent?.dataset.mapPostcodeLevel || activeMapPopupSelection?.level || ""),
      joinColumn: String(popupContent?.dataset.mapPostcodeColumn || activeMapPopupSelection?.joinColumn || ""),
    };
    if (!postcode.key || !postcode.joinColumn) return;
    const selection = activeMapPopupSelection?.key === postcode.key
      ? activeMapPopupSelection
      : null;
    event.preventDefault();
    event.stopPropagation();
    const action = button.dataset.mapPopupAction;
    if (action === "copy") {
      const copied = await copyTextToClipboard(postcode.key);
      showClipboardToast(copied ? `Copied ${postcode.key}` : `Could not copy ${postcode.key}`, !copied);
      return;
    }
    if (action === "zoom") {
      zoomToMapPopupSelection(selection);
      return;
    }
    if (action === "filter") {
      if (postcode.level === "area") {
        toggleMapAreaSelection(postcode);
      } else {
        applyMapPostcodeFilter(postcode);
      }
    } else if (action === "view-rows" && canOpenDatasetViewer()) {
      openMapPostcodeRows(postcode);
    }
  }

  function zoomToMapPopupSelection(selection) {
    if (!ukMap || !selection) return false;
    ukMap.closePopup();
    if (selection.level === "unit" && selection.latLng) {
      ukMap.setView(selection.latLng, Math.max(Number(ukMap.getZoom()) || 0, 13), { animate: false });
      state.mapStartupFitDone = true;
      captureMapView("explicit");
      return true;
    }
    const bounds = selection.layer?.getBounds?.();
    if (!bounds?.isValid?.()) return false;
    return fitMapBounds(bounds, selection.level, {
      padding: [30, 30],
      maxZoom: selection.level === "sector" ? 13 : 9,
    });
  }

  function bindControls() {
    setupMapFloatingControlDrag();
    bindMapFloatingControls();
    el("ukMap").addEventListener("click", handleMapPopupAction, true);
    el("ukMap").addEventListener("contextmenu", handleMapContextMenu, true);
    el("ukMap").addEventListener("keydown", handleMapContextMenuKeydown, true);
    document.addEventListener("pointerdown", handleMapRegionFilterDocumentPointerDown, true);
    document.addEventListener("click", handleMapRegionFilterDocumentClick, true);
    document.addEventListener("pointerup", handleMapRegionFilterDocumentPointerEnd, true);
    document.addEventListener("pointercancel", handleMapRegionFilterDocumentPointerEnd, true);
    document.addEventListener("keydown", handleMapRegionFilterDocumentKeydown, true);
    window.addEventListener("resize", closeMapRegionFilterPanel);
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
    cancelMapPendingMeta();
    closeMapRegionFilterPanel();
    state.lastMapData = null;
    state.mapStartupFitDone = false;
    state.renderedMapLevel = null;
    activeMapPopupSelection = null;
    setMapMatchLiveStatus("");
  }

  return {
    buildRequest: buildMapRequest,
    fetchData: fetchMapData,
    useCached: useCachedMapData,
    canUseCached: canUseCachedMapData,
    canRefreshInPlace: canRefreshMapInPlace,
    showMissingRequest: showMapMissingNumerator,
    activate,
    bindControls,
    captureView: captureMapView,
    captureFavouriteState,
    applyFavouriteState,
    showPendingRestore,
    cancelRequests: cancelMapRequests,
    setMapLevel,
    syncViewport,
    resize,
    refreshTheme,
    resetRenderState,
    closeMenus: closeMapRegionFilterPanel,
  };
}
