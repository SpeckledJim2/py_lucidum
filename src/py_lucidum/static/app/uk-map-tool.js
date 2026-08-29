import { bindSettingsStripOverflowCue } from "./shared/settings-strip.js";

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
    matchText = `no ${levelConfig.label} to match`;
  } else if (unmatchedRowCount <= 0) {
    matchText = `All ${levelConfig.label} matched`;
    matchState = "complete";
  } else {
    matchText = `${unmatchedRowCount.toLocaleString()} ${unmatchedRowCount === 1 ? "row" : "rows"} unmatched (${unmatchedPercentageText}%)`;
  }
  const missingText = missingRowCount > 0
    ? `${missingRowCount.toLocaleString()} ${missingRowCount === 1 ? "row" : "rows"} missing ${levelConfig.singular} (${missingPercentageText}%)`
    : `no rows missing ${levelConfig.singular}`;

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
  loadMapAdapter,
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
  getSelectedKpi = () => null,
  clearActiveFavouriteSelection = () => {},
}) {
  let L = null;
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
  const MAP_UNIT_SPATIAL_GRID_SIZE = 256;
  const MAP_UNIT_QUANTILE_SAMPLE_SIZE = 100_000;
  const MAP_MERCATOR_MAX_LATITUDE = 85.0511287798;
  const MAP_FIT_PADDING = [8, 8];
  const MAP_UNIT_FIT_PADDING = [18, 18];
  const MAP_POPUP_MAX_WIDTH = 440;
  const MAP_UNIT_ADAPTIVE_DENSE_COUNT = 500_000;
  const MAP_UNIT_ADAPTIVE_SPARSE_COUNT = 100;
  const MAP_UNIT_ADAPTIVE_DENSE_BASE_DIAMETER = 1;
  const MAP_UNIT_ADAPTIVE_DENSE_MAX_DIAMETER = 10;
  const MAP_UNIT_ADAPTIVE_ULTRA_DENSE_COUNT = 1_500_000;
  const MAP_UNIT_ADAPTIVE_ULTRA_DENSE_MAX_DIAMETER = 8;
  const MAP_UNIT_ADAPTIVE_SPARSE_DIAMETER = 6;
  const MAP_UNIT_ADAPTIVE_ZOOM_RANGE = 6;
  const MAP_UNIT_DOT_SIZE_MODES = new Set(["min", "adaptive"]);
  const MAP_OPACITY_PRESETS = [0.2, 0.6, 1];
  const MAP_DEFAULT_VIEW = { center: { lat: 54.5, lng: -3.2 }, zoom: 6 };
  const MAP_AREA_LABEL_MODES = new Set(["off", "on"]);
  const MAP_AREA_LABEL_MIN_FONT_SIZE = 6;
  const MAP_AREA_LABEL_BASE_FONT_SIZE = 9;
  const MAP_AREA_LABEL_MAX_FONT_SIZE = 20;
  const MAP_AREA_LABEL_GROWTH_PER_ZOOM = 2;
  const MAP_AREA_LABEL_MIN_ZOOM_OFFSET = -3;
  const MAP_AREA_LABEL_MAX_ZOOM_OFFSET = 1.5;
  const MAP_VECTOR_LABEL_COLOR = "#1f2937";
  const MAP_VECTOR_LABEL_HALO_COLOR = "rgba(255, 255, 255, 0.96)";
  const MAP_VECTOR_LABEL_HALO_WIDTH = 1.75;
  const MAP_VECTOR_LABEL_HALO_BLUR = 0.25;
  const MAP_VECTOR_ROAD_WIDTH_SCALE = 0.7;
  const MAP_INITIAL_FIT_OPTIONS = { animate: false };
  const MAP_TOOLBAR_CHEVRON_ICON = '<path d="m18 15-6-6-6 6"></path>';
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
    openFreeMapPositron: {
      label: "Light",
      styleUrl: "https://tiles.openfreemap.org/styles/positron",
      themePair: { light: "openFreeMapPositron", dark: "openFreeMapDark" },
    },
    openFreeMapDark: {
      label: "Dark",
      styleUrl: "https://tiles.openfreemap.org/styles/dark",
      themePair: { light: "openFreeMapPositron", dark: "openFreeMapDark" },
    },
  };
  const MAP_LEGACY_BASE_LAYERS = {
    grey: "openFreeMapPositron",
    darkGrey: "openFreeMapDark",
  };

  let ukMap = null;
  let ukMapLayer = null;
  let ukMapPointLayer = null;
  let ukMapLabelLayer = null;
  let baseTileLayer = null;
  let baseLabelLayer = null;
  let renderedBaseMap = "blank";
  let baseMapChangeGeneration = 0;
  let baseMapChangePending = false;
  let mapViewportControl = null;
  let mapNavigationControl = null;
  let mapCompassControlButton = null;
  let finishMapCompassUnitRotation = null;
  let mapCompassGestureCleanup = null;
  let mapCompassSuppressClick = false;
  let mapCompassSuppressClickTimer = null;
  let mapResizeObserver = null;
  let activeMapPopupSelection = null;
  let mapRegionFilterPanel = null;
  let stagedMapRegionAreas = null;
  let mapRegionFilterReturnFocus = null;
  let mapRegionDismissClickPending = false;
  let mapPendingMetaTimer = null;
  let mapPendingMetaRequestSeq = null;
  let mapInitPromise = null;
  let mapAreaLabelSizeFrame = null;
  let sectorSmoothingSavePending = false;

  function clearActiveMapFavourite(options = {}) {
    if (state.mapFavouriteRestoreInProgress && !options.force) return;
    clearActiveFavouriteSelection();
  }

  function normaliseUnitBounds(raw) {
    const south = Number(raw?.south);
    const west = Number(raw?.west);
    const north = Number(raw?.north);
    const east = Number(raw?.east);
    if (![south, west, north, east].every(Number.isFinite)) return null;
    if (south >= north || west >= east) return null;
    return { south, west, north, east };
  }

  function unitBoundsFromMap(bounds) {
    if (!bounds?.isValid?.()) return null;
    const rounded = (value) => Number(Number(value).toFixed(6));
    return normaliseUnitBounds({
      south: rounded(bounds.getSouth()),
      west: rounded(bounds.getWest()),
      north: rounded(bounds.getNorth()),
      east: rounded(bounds.getEast()),
    });
  }

  function unitBoundsContains(container, candidate) {
    const outer = normaliseUnitBounds(container);
    const inner = normaliseUnitBounds(candidate);
    if (!outer || !inner) return false;
    const epsilon = 1e-6;
    return outer.south <= inner.south + epsilon
      && outer.west <= inner.west + epsilon
      && outer.north >= inner.north - epsilon
      && outer.east >= inner.east - epsilon;
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
    const request = {
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
    if (request.level === "unit") {
      request.unitViewportBounds = null;
      request.compactUnitMetrics = "minimal";
      const geometryKey = unitGeometryRequestKey(request);
      request.reuseUnitGeometry = Boolean(
        state.renderedMapLevel === "unit"
        && ukMapPointLayer?.geometryKey === geometryKey
      );
    }
    return request;
  }

  function smoothingArtifactFilename(path) {
    const parts = String(path || "").split(/[\\/]/);
    return parts[parts.length - 1] || "sector-smoothing.parquet";
  }

  function compactSmoothingArtifactFilename(path, maxLength = 42) {
    const filename = smoothingArtifactFilename(path);
    if (filename.length <= maxLength) return filename;
    const suffixLength = 19;
    const prefixLength = Math.max(1, maxLength - suffixLength - 1);
    return `${filename.slice(0, prefixLength)}…${filename.slice(-suffixLength)}`;
  }

  async function saveSectorSmoothingParquet() {
    if (sectorSmoothingSavePending || state.mapLevel !== "sector") return;
    const request = buildMapRequest();
    if (!request) return;
    const button = el("mapSaveSmoothingBtn");
    sectorSmoothingSavePending = true;
    button.disabled = true;
    button.setAttribute("aria-busy", "true");
    button.textContent = "… .parquet";
    button.title = "Saving sector smoothing Parquet";
    button.setAttribute("aria-label", "Saving sector smoothing Parquet");
    try {
      const result = await api("/api/uk-map/sector-smoothing", {
        method: "POST",
        body: JSON.stringify(request),
      });
      const verb = result.replaced ? "Replaced" : "Saved";
      const filename = compactSmoothingArtifactFilename(result.path);
      showClipboardToast(
        `${verb} ${formatNumber(result.row_count)} sectors · ${filename}`,
        false,
        result.path,
      );
    } catch (error) {
      showClipboardToast(error.message, true);
    } finally {
      sectorSmoothingSavePending = false;
      button.setAttribute("aria-busy", "false");
      button.textContent = "-> .parquet";
      button.title = "Save N1-N5 sector smoothing Parquet";
      button.setAttribute("aria-label", "Save N1-N5 sector smoothing Parquet");
      button.disabled = state.mapLevel !== "sector";
    }
  }

  function unitGeometryRequestKey(request) {
    return JSON.stringify([
      request?.level || "",
      request?.source || "dataset",
      request?.filter || "",
      request?.unitColumn || "",
      request?.latitudeColumn || "",
      request?.longitudeColumn || "",
      normaliseUnitBounds(request?.unitViewportBounds),
    ]);
  }

  function hydrateReusedUnitGeometry(data, request) {
    if (data?.level !== "unit" || data?.unit_geometry?.included !== false) return true;
    const geometryKey = unitGeometryRequestKey(request);
    const geometry = ukMapPointLayer?.geometryPoints;
    const metrics = unitPointArrays(data);
    if (
      ukMapPointLayer?.geometryKey !== geometryKey
      || !geometry
      || !metrics
      || metrics.value?.length !== geometry.key?.length
    ) {
      return false;
    }
    data.unit_points = { ...geometry, ...metrics };
    data._unitGeometryReused = true;
    return true;
  }

  function showMapMissingNumerator() {
    setGroupMeta("uk_map", "Choose an Actual column");
    setMapRowMeta("");
    setMapMatchLiveStatus("");
    setChartMessage("UK mapping needs a numeric Actual column.");
  }

  function setMapRowMeta(message) {
    const target = el("mapRowMeta");
    target.textContent = message || "";
    target.title = target.textContent.trim();
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
      data._unitGeometryKey = request.level === "unit" ? unitGeometryRequestKey(request) : "";
      if (!hydrateReusedUnitGeometry(data, request)) {
        throw new Error("The cached postcode-unit geometry is no longer available. Refresh the map and try again.");
      }
      const cache = toolCache("uk_map");
      cache.requestKey = requestKey;
      cache.data = data;
      syncDuckDbTimingFromData("uk_map", data);
      syncClientTimingFromData("uk_map", data);
      await renderMap(data, geoJson);
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

  function clearMapLabelLayer() {
    if (mapAreaLabelSizeFrame !== null) cancelAnimationFrame(mapAreaLabelSizeFrame);
    mapAreaLabelSizeFrame = null;
    if (ukMapLabelLayer && ukMap) ukMap.removeLayer(ukMapLabelLayer);
    ukMapLabelLayer = null;
    const container = ukMap?.getContainer?.();
    container?.style.removeProperty("--map-area-label-min-size");
    container?.style.removeProperty("--map-area-label-base-size");
    container?.style.removeProperty("--map-area-label-max-size");
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
    clearMapLabelLayer();
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
        await renderMap(cache.data, null);
      } else if (geoJson) {
        await renderMap(cache.data, geoJson);
      } else {
        const loadedGeoJson = await loadMapGeoJson(cache.data.level);
        await renderMap(cache.data, loadedGeoJson);
      }
      return;
    }
    if (!activeLayer || state.renderedMapLevel !== cache.data.level || state.pendingMapZoom) {
      if (cache.data.level === "unit") {
        await renderMap(cache.data, null);
        return;
      }
      if (geoJson) {
        await renderMap(cache.data, geoJson);
      } else {
        const loadedGeoJson = await loadMapGeoJson(cache.data.level);
        await renderMap(cache.data, loadedGeoJson);
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

  async function initMap() {
    if (ukMap) return ukMap;
    if (mapInitPromise) return mapInitPromise;
    mapInitPromise = (async () => {
      L = await loadMapAdapter();
      ukMap = L.map("ukMap", {
        zoomControl: false,
        zoomDelta: 0.5,
        zoomSnap: 0.25,
      }).setView([MAP_DEFAULT_VIEW.center.lat, MAP_DEFAULT_VIEW.center.lng], MAP_DEFAULT_VIEW.zoom);
      ukMap.getContainer()._lucidumMap = ukMap;
      await ukMap.whenReady?.();
      ukMap.setBearing(state.mapBearing);
      ukMap.on("moveend zoomend rotateend", () => {
        captureMapView("maplibre");
      });
      ukMap.on("zoomend", () => {
        if (state.lastMapData?.level === "sector") restyleActiveMapPolygonLayer();
      });
      ukMap.on("zoomend resize", scheduleMapAreaLabelSizeUpdate);
      ukMap.on("popupclose", (event) => {
        if (!activeMapPopupSelection?.popup || activeMapPopupSelection.popup === event.popup) {
          activeMapPopupSelection = null;
        }
      });
      await setBaseMap(state.baseMap);
      addMapViewportControl();
      observeMapResize();
      return ukMap;
    })();
    try {
      return await mapInitPromise;
    } catch (error) {
      mapInitPromise = null;
      ukMap = null;
      throw error;
    }
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
      "button, input, select, textarea, a, [contenteditable], .maplibregl-ctrl, .maplibregl-popup",
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
    return { center: { lat, lng }, zoom, bearing: normaliseMapBearing(view?.bearing) };
  }

  function currentMapView() {
    if (!ukMap || !mapContainerVisible()) return null;
    const center = ukMap.getCenter();
    return normaliseMapView({ center, zoom: ukMap.getZoom(), bearing: ukMap.getBearing() });
  }

  function captureMapView(reason = "") {
    if (!ukMap || state.restoringMapView) return null;
    const view = currentMapView();
    if (!view) return null;
    if (!state.mapStartupFitDone && !state.mapView && reason !== "startup-fit" && reason !== "explicit") {
      return null;
    }
    state.mapView = view;
    state.mapBearing = view.bearing;
    if (reason === "maplibre" && state.tool === "uk_map") clearActiveMapFavourite();
    return view;
  }

  function restoreMapView(view) {
    const nextView = normaliseMapView(view);
    if (!ukMap || !nextView || !mapContainerVisible()) return false;
    state.restoringMapView = true;
    state.mapView = nextView;
    state.mapBearing = nextView.bearing;
    try {
      ukMap.setView([nextView.center.lat, nextView.center.lng], nextView.zoom, {
        animate: false,
        bearing: nextView.bearing,
      });
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
      scheduleMapViewportSync({ mode: "preserve" });
    });
    mapResizeObserver.observe(target);
  }

  function openFreeMapForegroundLayer(layer) {
    return layer?.type === "line" || layer?.type === "symbol";
  }

  function isOpenFreeMapRoadLayer(layer) {
    if (layer?.type !== "line") return false;
    const layerId = String(layer.id || "").toLowerCase();
    return layerId.includes("road")
      || layerId.includes("highway")
      || layerId.includes("tunnel");
  }

  function applyOpenFreeMapRoadWidths() {
    if (!MAP_BASE_LAYERS[renderedBaseMap]?.styleUrl) return;
    const rawMap = ukMap?.raw;
    const foregroundLayerIds = new Set(ukMap?.getStyleForegroundLayerIds?.() || []);
    (rawMap?.getStyle?.()?.layers || [])
      .filter((layer) => foregroundLayerIds.has(layer.id) && isOpenFreeMapRoadLayer(layer))
      .forEach((layer) => {
        const sourceWidth = rawMap.getPaintProperty(layer.id, "line-width");
        if (sourceWidth === undefined || sourceWidth === null) return;
        rawMap.setPaintProperty(
          layer.id,
          "line-width",
          ["*", sourceWidth, MAP_VECTOR_ROAD_WIDTH_SCALE],
        );
      });
  }

  function applyOpenFreeMapLabelContrast() {
    if (!MAP_BASE_LAYERS[renderedBaseMap]?.styleUrl) return;
    const rawMap = ukMap?.raw;
    const foregroundLayerIds = new Set(ukMap?.getStyleForegroundLayerIds?.() || []);
    (rawMap?.getStyle?.()?.layers || [])
      .filter((layer) => (
        foregroundLayerIds.has(layer.id)
        && layer.type === "symbol"
        && layer.layout?.["text-field"] !== undefined
      ))
      .forEach((layer) => {
        rawMap.setPaintProperty(layer.id, "text-color", MAP_VECTOR_LABEL_COLOR);
        rawMap.setPaintProperty(layer.id, "text-opacity", 1);
        rawMap.setPaintProperty(layer.id, "text-halo-color", MAP_VECTOR_LABEL_HALO_COLOR);
        rawMap.setPaintProperty(layer.id, "text-halo-width", MAP_VECTOR_LABEL_HALO_WIDTH);
        rawMap.setPaintProperty(layer.id, "text-halo-blur", MAP_VECTOR_LABEL_HALO_BLUR);
      });
  }

  function removeRasterBaseLayers() {
    if (baseLabelLayer) {
      ukMap.removeLayer(baseLabelLayer);
      baseLabelLayer = null;
    }
    if (baseTileLayer) {
      ukMap.removeLayer(baseTileLayer);
      baseTileLayer = null;
    }
  }

  function renderedBaseMatches(baseMap) {
    if (!ukMap || renderedBaseMap !== baseMap) return false;
    const config = MAP_BASE_LAYERS[baseMap];
    if (Boolean(config.styleUrl) !== Boolean(ukMap.usesExternalStyle?.())) return false;
    const tileLayerMatches = config.url
      ? Boolean(baseTileLayer && ukMap.hasLayer(baseTileLayer))
      : !baseTileLayer;
    const labelLayerMatches = config.labelUrl
      ? Boolean(baseLabelLayer && ukMap.hasLayer(baseLabelLayer))
      : !baseLabelLayer;
    return tileLayerMatches && labelLayerMatches;
  }

  async function installBaseMap(baseMap, generation) {
    const config = MAP_BASE_LAYERS[baseMap];
    removeRasterBaseLayers();
    if (config.styleUrl || ukMap.usesExternalStyle?.()) {
      const replaced = await ukMap.replaceStyle(config.styleUrl || null, {
        foregroundLayerPredicate: config.styleUrl ? openFreeMapForegroundLayer : null,
      });
      if (!replaced || generation !== baseMapChangeGeneration) return false;
    }
    if (generation !== baseMapChangeGeneration) return false;
    if (config.url) {
      baseTileLayer = L.tileLayer(config.url, {
        maxZoom: 19,
        attribution: config.attribution || "",
      }).addTo(ukMap);
      baseTileLayer.bringToBack();
    }
    if (config.labelUrl) {
      baseLabelLayer = L.tileLayer(config.labelUrl, {
        maxZoom: 19,
      }).addTo(ukMap);
      baseLabelLayer.bringToFront();
    }
    renderedBaseMap = baseMap;
    applyOpenFreeMapRoadWidths();
    applyOpenFreeMapLabelContrast();
    bringBaseLabelsToFront();
    syncBaseMapVisualState();
    syncMapControls();
    return true;
  }

  async function setBaseMap(baseMap) {
    const requestedBaseMap = MAP_LEGACY_BASE_LAYERS[baseMap] || baseMap;
    const nextBaseMap = MAP_BASE_LAYERS[requestedBaseMap] ? requestedBaseMap : "blank";
    const previousBaseMap = renderedBaseMap;
    state.baseMap = nextBaseMap;
    syncMapControls();
    if (!ukMap) return false;
    const hadPendingBaseMapChange = baseMapChangePending;
    const generation = ++baseMapChangeGeneration;
    if (renderedBaseMatches(nextBaseMap) && !hadPendingBaseMapChange) {
      syncBaseMapVisualState();
      return true;
    }
    baseMapChangePending = true;
    try {
      return await installBaseMap(nextBaseMap, generation);
    } catch (error) {
      if (generation !== baseMapChangeGeneration) return false;
      const failedLabel = MAP_BASE_LAYERS[nextBaseMap]?.label || "base map";
      const fallbackBaseMap = previousBaseMap === nextBaseMap ? "blank" : previousBaseMap;
      showClipboardToast(`Could not load ${failedLabel}; restored ${MAP_BASE_LAYERS[fallbackBaseMap].label}.`, true);
      state.baseMap = fallbackBaseMap;
      try {
        if (await installBaseMap(fallbackBaseMap, generation)) return false;
      } catch (_) {
      }
      if (fallbackBaseMap !== "blank" && generation === baseMapChangeGeneration) {
        state.baseMap = "blank";
        try {
          await installBaseMap("blank", generation);
        } catch (_) {
        }
      }
      syncBaseMapVisualState();
      syncMapControls();
      return false;
    } finally {
      if (generation === baseMapChangeGeneration) baseMapChangePending = false;
    }
  }

  function bringBaseLabelsToFront() {
    const rawMap = ukMap?.raw;
    const foregroundLayerIds = ukMap?.getStyleForegroundLayerIds?.() || [];
    const foregroundLayerIdSet = new Set(foregroundLayerIds);
    const styleLayers = rawMap?.getStyle?.()?.layers || [];
    const foregroundLayers = styleLayers
      .filter((layer) => foregroundLayerIdSet.has(layer.id));
    const lineLayerIds = foregroundLayers
      .filter((layer) => layer.type === "line")
      .map((layer) => layer.id);
    const symbolLayerIds = foregroundLayers
      .filter((layer) => layer.type === "symbol")
      .map((layer) => layer.id);

    const analysisFillLayerId = state.renderedMapLevel === "unit"
      ? null
      : ukMapLayer?.fillLayerId;
    const analysisOutlineLayerId = state.renderedMapLevel === "unit"
      ? ukMapPointLayer?.canvasMapLayer?.layerId
      : ukMapLayer?.lineLayerId;
    const desiredLayerIds = [
      analysisFillLayerId,
      ...lineLayerIds,
      analysisOutlineLayerId,
      ...symbolLayerIds,
    ].filter((layerId) => layerId && rawMap?.getLayer?.(layerId));
    const desiredLayerIdSet = new Set(desiredLayerIds);
    const currentLayerIds = styleLayers
      .map((layer) => layer.id)
      .filter((layerId) => desiredLayerIdSet.has(layerId));
    const layerOrderChanged = desiredLayerIds.length !== currentLayerIds.length
      || desiredLayerIds.some((layerId, index) => layerId !== currentLayerIds[index]);
    if (layerOrderChanged) {
      desiredLayerIds.forEach((layerId) => rawMap.moveLayer(layerId));
    }

    baseLabelLayer?.bringToFront?.();
    const container = ukMap?.getContainer?.();
    if (container) {
      container._lucidumBaseStyleForegroundLayerIds = foregroundLayerIds;
    }
  }

  function syncBaseMapVisualState() {
    if (!ukMap) return;
    const container = ukMap.getContainer();
    container._lucidumBaseMap = renderedBaseMap;
    container._lucidumRequestedBaseMap = state.baseMap;
    container._lucidumBaseTileLayer = baseTileLayer;
    container._lucidumBaseLabelLayer = baseLabelLayer;
    container._lucidumBaseStyleForegroundLayerIds = ukMap.getStyleForegroundLayerIds?.() || [];
    container.classList.toggle("blank-base", renderedBaseMap === "blank");
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

  function syncBaseMapForTheme() {
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

  function normaliseMapBearing(value) {
    const bearing = Number(value);
    if (!Number.isFinite(bearing)) return 0;
    const wrapped = ((bearing + 180) % 360 + 360) % 360 - 180;
    return Object.is(wrapped, -0) ? 0 : wrapped;
  }

  function normaliseFavouriteMapLevel(level) {
    const requested = String(level || "");
    if (Object.prototype.hasOwnProperty.call(MAP_LEVELS, requested) && mapLevelSelectable(requested)) return requested;
    if (Object.prototype.hasOwnProperty.call(MAP_LEVELS, state.mapLevel) && mapLevelSelectable(state.mapLevel)) return state.mapLevel;
    return Object.keys(MAP_LEVELS).find((candidate) => mapLevelSelectable(candidate)) || "area";
  }

  function normaliseMapDotSizeMode(value) {
    const mode = String(value || "").toLowerCase();
    return MAP_UNIT_DOT_SIZE_MODES.has(mode) ? mode : "adaptive";
  }

  function normaliseMapAreaLabels(value, legacyLabelSize = 0) {
    const mode = String(value || "").toLowerCase();
    if (MAP_AREA_LABEL_MODES.has(mode)) return mode;
    return Number(legacyLabelSize) > 0 ? "on" : "off";
  }

  function normaliseMapBorderWeight(value) {
    const weight = Number(value);
    if (!Number.isFinite(weight)) return 1;
    if (weight <= 0) return 0;
    return weight <= 2 ? 1 : 3;
  }

  function normaliseMapOpacity(value) {
    const opacity = clampMapNumber(value, 1, 0, 1);
    if (opacity < 0.4) return MAP_OPACITY_PRESETS[0];
    if (opacity < 0.8) return MAP_OPACITY_PRESETS[1];
    return MAP_OPACITY_PRESETS[2];
  }

  function normaliseFavouriteMapState(map = {}) {
    const payload = map && typeof map === "object" ? map : {};
    const level = normaliseFavouriteMapLevel(payload.level);
    const requestedBaseMap = MAP_LEGACY_BASE_LAYERS[payload.baseMap] || String(payload.baseMap || "");
    const baseMap = MAP_BASE_LAYERS[requestedBaseMap] ? requestedBaseMap : "blank";
    const palette = MAP_PALETTES[payload.palette] ? String(payload.palette) : "divergent";
    const bearing = normaliseMapBearing(payload.bearing);
    return {
      level,
      baseMap,
      palette,
      lineWeight: normaliseMapBorderWeight(payload.lineWeight),
      dotSizeMode: normaliseMapDotSizeMode(payload.dotSizeMode),
      opacity: normaliseMapOpacity(payload.opacity),
      hotspots: clampMapNumber(payload.hotspots, 0, -9, 9, { integer: true }),
      areaLabels: normaliseMapAreaLabels(payload.areaLabels, payload.labelSize),
      smoothingLevel: clampMapNumber(payload.smoothingLevel, 0, 0, 5, { integer: true }),
      bearing,
      view: normaliseMapView({ center: payload.center, zoom: payload.zoom, bearing }),
    };
  }

  function captureFavouriteState() {
    const view = normaliseMapView(currentMapView() || state.mapView) || normaliseMapView(MAP_DEFAULT_VIEW);
    if (view) state.mapView = view;
    return {
      level: state.mapLevel,
      baseMap: state.baseMap,
      palette: state.mapPalette,
      lineWeight: normaliseMapBorderWeight(state.mapLineWeight),
      dotSizeMode: normaliseMapDotSizeMode(state.mapDotSizeMode),
      opacity: normaliseMapOpacity(state.mapOpacity),
      hotspots: Number(state.mapHotspots),
      areaLabels: normaliseMapAreaLabels(state.mapAreaLabels),
      smoothingLevel: Number(state.mapSmoothingLevel),
      center: view?.center || null,
      zoom: view?.zoom ?? null,
      bearing: view?.bearing ?? normaliseMapBearing(state.mapBearing),
    };
  }

  function applyFavouriteState(map = {}) {
    const next = normaliseFavouriteMapState(map);
    state.mapLevel = next.level;
    state.mapPalette = next.palette;
    state.mapLineWeight = next.lineWeight;
    state.mapDotSizeMode = next.dotSizeMode;
    state.mapOpacity = next.opacity;
    state.mapHotspots = next.hotspots;
    state.mapAreaLabels = next.areaLabels;
    state.mapSmoothingLevel = next.smoothingLevel;
    state.mapBearing = next.bearing;
    state.mapView = next.view;
    state.mapViewRestorePending = next.view;
    state.pendingMapZoom = null;
    state.mapStartupFitDone = Boolean(next.view);
    if (ukMap && !next.view) ukMap.setBearing(next.bearing);
    const baseMapPromise = setBaseMap(next.baseMap);
    syncMapControls();
    syncFloatingMapControl();
    return baseMapPromise;
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

  function zoomMapToLondon() {
    if (!ukMap) return;
    ukMap.setView([51.5074, -0.1278], 10, { animate: false });
    state.mapStartupFitDone = true;
    captureMapView("explicit");
  }

  function endMapCompassUnitRotation() {
    if (!finishMapCompassUnitRotation) return;
    const finish = finishMapCompassUnitRotation;
    finishMapCompassUnitRotation = null;
    finish();
  }

  function beginMapCompassUnitRotation() {
    if (finishMapCompassUnitRotation) return;
    const unitLayer = ukMapPointLayer;
    if (!unitLayer?.beginRotation?.()) return;
    finishMapCompassUnitRotation = () => unitLayer.endRotation?.();
  }

  function mapCompassPointerAngle(clientX, clientY) {
    const bounds = mapCompassControlButton?.getBoundingClientRect?.();
    if (!bounds) return null;
    const offsetX = Number(clientX) - (bounds.left + (bounds.width / 2));
    const offsetY = Number(clientY) - (bounds.top + (bounds.height / 2));
    if (!Number.isFinite(offsetX) || !Number.isFinite(offsetY) || Math.hypot(offsetX, offsetY) < 2) {
      return null;
    }
    return Math.atan2(offsetY, offsetX);
  }

  function finishMapCompassGesture({ moved = false } = {}) {
    const cleanup = mapCompassGestureCleanup;
    mapCompassGestureCleanup = null;
    cleanup?.();
    endMapCompassUnitRotation();
    if (!moved) return;
    mapCompassSuppressClick = true;
    window.clearTimeout(mapCompassSuppressClickTimer);
    mapCompassSuppressClickTimer = window.setTimeout(() => {
      mapCompassSuppressClick = false;
      mapCompassSuppressClickTimer = null;
    }, 500);
    state.mapStartupFitDone = true;
    captureMapView("explicit");
  }

  function beginMapCompassGesture(clientX, clientY) {
    if (!ukMap) return null;
    if (mapCompassGestureCleanup) finishMapCompassGesture();
    beginMapCompassUnitRotation();
    let lastAngle = mapCompassPointerAngle(clientX, clientY);
    let totalRotation = 0;
    return {
      move(nextClientX, nextClientY) {
        const nextAngle = mapCompassPointerAngle(nextClientX, nextClientY);
        if (nextAngle === null) {
          lastAngle = null;
          return;
        }
        if (lastAngle === null) {
          lastAngle = nextAngle;
          return;
        }
        const angleDelta = Math.atan2(
          Math.sin(nextAngle - lastAngle),
          Math.cos(nextAngle - lastAngle),
        );
        lastAngle = nextAngle;
        const bearingDelta = (angleDelta * 180) / Math.PI;
        if (Math.abs(bearingDelta) < 0.01) return;
        totalRotation += Math.abs(bearingDelta);
        ukMap.setBearing(ukMap.getBearing() - bearingDelta);
      },
      finish() {
        finishMapCompassGesture({ moved: totalRotation >= 0.5 });
      },
    };
  }

  function handleMapCompassMouseDown(event) {
    if (event.button !== 0) return;
    event.preventDefault();
    event.stopImmediatePropagation();
    const gesture = beginMapCompassGesture(event.clientX, event.clientY);
    if (!gesture) return;
    const handleMove = (moveEvent) => {
      moveEvent.preventDefault();
      gesture.move(moveEvent.clientX, moveEvent.clientY);
    };
    const handleEnd = () => gesture.finish();
    mapCompassGestureCleanup = () => {
      window.removeEventListener("mousemove", handleMove);
      window.removeEventListener("mouseup", handleEnd);
    };
    window.addEventListener("mousemove", handleMove, { passive: false });
    window.addEventListener("mouseup", handleEnd, { once: true });
  }

  function handleMapCompassTouchStart(event) {
    if (event.touches.length !== 1) return;
    event.preventDefault();
    event.stopImmediatePropagation();
    const touch = event.touches[0];
    const gesture = beginMapCompassGesture(touch.clientX, touch.clientY);
    if (!gesture) return;
    const handleMove = (moveEvent) => {
      if (moveEvent.touches.length !== 1) return;
      moveEvent.preventDefault();
      const nextTouch = moveEvent.touches[0];
      gesture.move(nextTouch.clientX, nextTouch.clientY);
    };
    const handleEnd = () => gesture.finish();
    mapCompassGestureCleanup = () => {
      window.removeEventListener("touchmove", handleMove);
      window.removeEventListener("touchend", handleEnd);
      window.removeEventListener("touchcancel", handleEnd);
    };
    window.addEventListener("touchmove", handleMove, { passive: false });
    window.addEventListener("touchend", handleEnd, { once: true });
    window.addEventListener("touchcancel", handleEnd, { once: true });
  }

  function handleMapCompassClick(event) {
    if (!mapCompassSuppressClick || event.detail === 0) return;
    event.preventDefault();
    event.stopImmediatePropagation();
    mapCompassSuppressClick = false;
    window.clearTimeout(mapCompassSuppressClickTimer);
    mapCompassSuppressClickTimer = null;
  }

  function addMapViewportControl() {
    if (!ukMap || mapViewportControl) return;
    const ViewportControl = L.Control.extend({
      options: { position: "topleft" },
      onAdd() {
        const container = L.DomUtil.create("div", "map-viewport-control maplibregl-ctrl");
        container.innerHTML = `
          <button id="mapControlReset" class="map-viewport-button map-toolbar-toggle" type="button" title="Collapse map controls" aria-label="Collapse map controls" aria-controls="mapToolbar mapInfoStrip" aria-expanded="true">
            <svg viewBox="0 0 24 24" aria-hidden="true" focusable="false">${MAP_TOOLBAR_CHEVRON_ICON}</svg>
          </button>
        `;
        mapNavigationControl = new L.maplibregl.NavigationControl({
          showCompass: true,
          showZoom: true,
          visualizePitch: false,
        });
        const navigationContainer = mapNavigationControl.onAdd(ukMap.raw);
        navigationContainer.classList.add("map-native-navigation");
        navigationContainer.querySelector(".maplibregl-ctrl-zoom-in").id = "mapZoomIn";
        navigationContainer.querySelector(".maplibregl-ctrl-zoom-out").id = "mapZoomOut";
        mapCompassControlButton = navigationContainer.querySelector(".maplibregl-ctrl-compass");
        mapCompassControlButton.id = "mapCompass";
        mapCompassControlButton.addEventListener("mousedown", handleMapCompassMouseDown, true);
        mapCompassControlButton.addEventListener("touchstart", handleMapCompassTouchStart, {
          capture: true,
          passive: false,
        });
        mapCompassControlButton.addEventListener("click", handleMapCompassClick, true);
        container.append(navigationContainer);
        container.insertAdjacentHTML("beforeend", `
          <button id="mapFitUk" class="map-viewport-button" type="button" title="Fit UK map layer" aria-label="Fit UK map layer">
            <img src="/tools/uk-map/static/icons/UK.png" alt="">
          </button>
          <button id="mapZoomLondon" class="map-viewport-button" type="button" title="Zoom to London" aria-label="Zoom to London">
            <img class="map-viewport-icon-london" src="/tools/uk-map/static/icons/London.png" alt="">
          </button>
        `);
        L.DomEvent.disableClickPropagation(container);
        L.DomEvent.disableScrollPropagation(container);
        container.querySelector("#mapControlReset").addEventListener("click", toggleMapToolbarCollapsed);
        container.querySelector("#mapFitUk").addEventListener("click", () => fitMapToLayer());
        container.querySelector("#mapZoomLondon").addEventListener("click", () => zoomMapToLondon());
        syncMapToolbarVisibility();
        return container;
      },
      onRemove() {
        finishMapCompassGesture();
        mapCompassControlButton?.removeEventListener("mousedown", handleMapCompassMouseDown, true);
        mapCompassControlButton?.removeEventListener("touchstart", handleMapCompassTouchStart, true);
        mapCompassControlButton?.removeEventListener("click", handleMapCompassClick, true);
        window.clearTimeout(mapCompassSuppressClickTimer);
        mapCompassSuppressClickTimer = null;
        mapCompassSuppressClick = false;
        mapCompassControlButton = null;
        mapNavigationControl?.onRemove();
        mapNavigationControl = null;
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
      // MapLibre antialiases each fill feature independently. Adjacent postcode
      // polygons can therefore expose pale hairline seams on dark basemaps even
      // when the explicit border layer is disabled. Borders are rendered by the
      // separate line layer below, so the fills themselves should stay crisp.
      fillAntialias: false,
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

  function mapUnitAdaptiveScarcity(pointCount) {
    const count = Math.max(1, Number(pointCount) || 1);
    const denseLog = Math.log10(MAP_UNIT_ADAPTIVE_DENSE_COUNT);
    const sparseLog = Math.log10(MAP_UNIT_ADAPTIVE_SPARSE_COUNT);
    return Math.max(0, Math.min(1, (denseLog - Math.log10(count)) / (denseLog - sparseLog)));
  }

  function mapUnitUltraDenseProgress(pointCount) {
    const count = Math.max(1, Number(pointCount) || 1);
    if (count <= MAP_UNIT_ADAPTIVE_DENSE_COUNT) return 0;
    const progress = Math.log(count / MAP_UNIT_ADAPTIVE_DENSE_COUNT)
      / Math.log(MAP_UNIT_ADAPTIVE_ULTRA_DENSE_COUNT / MAP_UNIT_ADAPTIVE_DENSE_COUNT);
    return Math.max(0, Math.min(1, progress));
  }

  function unitPointAdaptiveDiameter(zoom, fittedZoom, pointCount) {
    const scarcity = mapUnitAdaptiveScarcity(pointCount);
    const ultraDenseProgress = mapUnitUltraDenseProgress(pointCount);
    const denseMaximumDiameter = MAP_UNIT_ADAPTIVE_DENSE_MAX_DIAMETER
      + ((MAP_UNIT_ADAPTIVE_ULTRA_DENSE_MAX_DIAMETER - MAP_UNIT_ADAPTIVE_DENSE_MAX_DIAMETER)
        * ultraDenseProgress);
    const baselineDiameter = MAP_UNIT_ADAPTIVE_DENSE_BASE_DIAMETER
      + ((MAP_UNIT_ADAPTIVE_SPARSE_DIAMETER - MAP_UNIT_ADAPTIVE_DENSE_BASE_DIAMETER) * scarcity);
    const maximumDiameter = denseMaximumDiameter
      + ((MAP_UNIT_ADAPTIVE_SPARSE_DIAMETER - denseMaximumDiameter) * scarcity);
    const currentZoom = Number(zoom);
    const baselineZoom = Number(fittedZoom);
    const rawProgress = Number.isFinite(currentZoom) && Number.isFinite(baselineZoom)
      ? (currentZoom - baselineZoom) / MAP_UNIT_ADAPTIVE_ZOOM_RANGE
      : 0;
    const progress = Math.max(0, Math.min(1, rawProgress));
    const smoothProgress = progress * progress * (3 - (2 * progress));
    return baselineDiameter + ((maximumDiameter - baselineDiameter) * smoothProgress);
  }

  function unitPointRenderStyle({ zoom, fittedZoom, pointCount, pixelRatio }) {
    const ratio = Number.isFinite(Number(pixelRatio)) && Number(pixelRatio) > 0
      ? Number(pixelRatio)
      : 1;
    if (normaliseMapDotSizeMode(state.mapDotSizeMode) === "min") {
      return { diameter: 1 / ratio, radius: 0.5 / ratio, singleDevicePixel: true };
    }
    const diameter = unitPointAdaptiveDiameter(zoom, fittedZoom, pointCount);
    if (diameter <= MAP_UNIT_ADAPTIVE_DENSE_BASE_DIAMETER) {
      return { diameter: 1 / ratio, radius: 0.5 / ratio, singleDevicePixel: true };
    }
    return { diameter, radius: diameter / 2, singleDevicePixel: false };
  }

  function unitPointHitRadius(radius) {
    return Math.max(radius + 4, 6);
  }

  function unitPointArrays(data) {
    const points = data?.unit_points;
    return points && Array.isArray(points.value) ? points : null;
  }

  function unitPointCount(data) {
    const points = unitPointArrays(data);
    if (!points) return (data?.rows || []).length;
    let count = 0;
    for (const rawValue of points.value || []) {
      if (finiteNumber(rawValue) !== null) count += 1;
    }
    return count;
  }

  function normaliseUnitPointColumns(data) {
    const points = unitPointArrays(data);
    if (points) return points;
    const rows = data?.rows || [];
    return {
      key: rows.map((row) => row.key),
      row_count: rows.map((row) => row.row_count),
      numerator: rows.map((row) => row.numerator),
      denominator: rows.map((row) => row.denominator),
      value: rows.map((row) => row.value),
      latitude: rows.map((row) => row.latitude),
      longitude: rows.map((row) => row.longitude),
    };
  }

  function makeUnitPointScale(data) {
    const points = normaliseUnitPointColumns(data);
    const rawValues = points.value || [];
    const sampleCapacity = Math.min(rawValues.length, MAP_UNIT_QUANTILE_SAMPLE_SIZE);
    const values = new Float64Array(sampleCapacity);
    const stride = rawValues.length > sampleCapacity ? rawValues.length / sampleCapacity : 1;
    let nextSampleIndex = rawValues.length > sampleCapacity ? stride / 2 : 0;
    let valueCount = 0;
    let minimum = Infinity;
    let maximum = -Infinity;
    for (let index = 0; index < rawValues.length; index += 1) {
      const value = finiteNumber(rawValues[index]);
      if (value !== null) {
        minimum = Math.min(minimum, value);
        maximum = Math.max(maximum, value);
      }
      if (value !== null && index >= nextSampleIndex && valueCount < sampleCapacity) {
        values[valueCount] = value;
        valueCount += 1;
        nextSampleIndex += stride;
      }
    }
    const sampledValues = values.subarray(0, valueCount);
    sampledValues.sort();
    if (sampledValues.length) {
      sampledValues[0] = minimum;
      sampledValues[sampledValues.length - 1] = maximum;
    }
    return makeQuantileScaleFromValues(sampledValues, { sorted: true });
  }

  function mapUnitHotspotIndexes(data) {
    const points = normaliseUnitPointColumns(data);
    const selection = mapHotspotSelection();
    if (!selection) return null;
    const validRows = [];
    for (let index = 0; index < (points.value || []).length; index += 1) {
      const value = finiteNumber(points.value?.[index]);
      if (value === null) continue;
      validRows.push({ value, index });
    }
    if (!validRows.length) return null;
    validRows.sort((a, b) => {
      if (a.value !== b.value) return (a.value - b.value) * selection.direction;
      return a.index - b.index;
    });
    const count = Math.min(validRows.length, Math.max(1, Math.ceil(validRows.length * selection.fraction)));
    return new Set(validRows.slice(0, count).map((row) => row.index));
  }

  function makeUnitPointLayer(data, scale, hotspotIndexes, geometryKey) {
    return new (L.Layer.extend({
      initialize(mapData, initialScale, initialHotspotIndexes, initialGeometryKey) {
        this.geometryKey = initialGeometryKey;
        this.coverageBounds = normaliseUnitBounds(mapData?.unit_viewport?.bounds);
        this.zoomGeneration = 0;
        this.zooming = false;
        this.zoomRefreshPending = false;
        this.zoomFallbackFrame = null;
        this.displayResizeFrame = null;
        this.handleDisplayResizeBound = () => this.handleDisplayResize();
        this.setGeometry(normaliseUnitPointColumns(mapData));
        this.setData(mapData, initialScale, initialHotspotIndexes);
        this.tooltip = null;
      },
      setGeometry(points) {
        const keys = points?.key || [];
        const rowCounts = points?.row_count || [];
        const latitudes = points?.latitude || [];
        const longitudes = points?.longitude || [];
        const count = Math.min(keys.length, latitudes.length, longitudes.length);
        this.geometryPoints = {
          key: keys,
          row_count: rowCounts,
          latitude: latitudes,
          longitude: longitudes,
        };
        this.pointCount = count;
        this.worldX = new Float64Array(count);
        this.worldY = new Float64Array(count);
        let minLatitude = Infinity;
        let maxLatitude = -Infinity;
        let minLongitude = Infinity;
        let maxLongitude = -Infinity;
        for (let index = 0; index < count; index += 1) {
          const latitude = Number(latitudes[index]);
          const longitude = Number(longitudes[index]);
          if (!Number.isFinite(latitude) || !Number.isFinite(longitude)) {
            this.worldX[index] = Number.NaN;
            this.worldY[index] = Number.NaN;
            continue;
          }
          const clampedLatitude = Math.max(-MAP_MERCATOR_MAX_LATITUDE, Math.min(MAP_MERCATOR_MAX_LATITUDE, latitude));
          const sinLatitude = Math.sin((clampedLatitude * Math.PI) / 180);
          this.worldX[index] = ((longitude + 180) / 360) * 256;
          this.worldY[index] = (0.5 - (Math.log((1 + sinLatitude) / (1 - sinLatitude)) / (4 * Math.PI))) * 256;
          minLatitude = Math.min(minLatitude, latitude);
          maxLatitude = Math.max(maxLatitude, latitude);
          minLongitude = Math.min(minLongitude, longitude);
          maxLongitude = Math.max(maxLongitude, longitude);
        }
        this.spatialBounds = {
          minLatitude,
          maxLatitude,
          minLongitude,
          maxLongitude,
        };
        this.bounds = Number.isFinite(minLatitude)
          ? L.latLngBounds([[minLatitude, minLongitude], [maxLatitude, maxLongitude]])
          : L.latLngBounds([]);
        this.buildSpatialIndex();
      },
      spatialCell(latitude, longitude) {
        const bounds = this.spatialBounds;
        const longitudeRange = Math.max(Number.EPSILON, bounds.maxLongitude - bounds.minLongitude);
        const latitudeRange = Math.max(Number.EPSILON, bounds.maxLatitude - bounds.minLatitude);
        const x = Math.max(0, Math.min(
          MAP_UNIT_SPATIAL_GRID_SIZE - 1,
          Math.floor(((longitude - bounds.minLongitude) / longitudeRange) * MAP_UNIT_SPATIAL_GRID_SIZE),
        ));
        const y = Math.max(0, Math.min(
          MAP_UNIT_SPATIAL_GRID_SIZE - 1,
          Math.floor(((latitude - bounds.minLatitude) / latitudeRange) * MAP_UNIT_SPATIAL_GRID_SIZE),
        ));
        return { x, y, index: (y * MAP_UNIT_SPATIAL_GRID_SIZE) + x };
      },
      buildSpatialIndex() {
        const cellCount = MAP_UNIT_SPATIAL_GRID_SIZE * MAP_UNIT_SPATIAL_GRID_SIZE;
        const counts = new Uint32Array(cellCount);
        for (let index = 0; index < this.pointCount; index += 1) {
          const latitude = Number(this.geometryPoints.latitude[index]);
          const longitude = Number(this.geometryPoints.longitude[index]);
          if (!Number.isFinite(latitude) || !Number.isFinite(longitude)) continue;
          counts[this.spatialCell(latitude, longitude).index] += 1;
        }
        const offsets = new Uint32Array(cellCount + 1);
        for (let cell = 0; cell < cellCount; cell += 1) {
          offsets[cell + 1] = offsets[cell] + counts[cell];
        }
        const cursors = offsets.slice(0, cellCount);
        const indexes = new Uint32Array(offsets[cellCount]);
        for (let index = 0; index < this.pointCount; index += 1) {
          const latitude = Number(this.geometryPoints.latitude[index]);
          const longitude = Number(this.geometryPoints.longitude[index]);
          if (!Number.isFinite(latitude) || !Number.isFinite(longitude)) continue;
          const cell = this.spatialCell(latitude, longitude).index;
          indexes[cursors[cell]] = index;
          cursors[cell] += 1;
        }
        this.spatialOffsets = offsets;
        this.spatialIndexes = indexes;
      },
      prepareColorBuckets() {
        const buckets = new Uint8Array(this.pointCount);
        buckets.fill(255);
        this.colorBuckets = buckets;
      },
      setData(mapData, nextScale, nextHotspotIndexes) {
        const points = normaliseUnitPointColumns(mapData);
        if ((points.value || []).length !== this.pointCount) return false;
        this.data = mapData;
        this.coverageBounds = normaliseUnitBounds(mapData?.unit_viewport?.bounds);
        this.metricPoints = {
          numerator: points.numerator || [],
          denominator: points.denominator || [],
          value: points.value || [],
        };
        this.scale = nextScale;
        this.hotspotIndexes = nextHotspotIndexes;
        const responseCount = Number(mapData?.point_summary?.plotted_count);
        this.plottedPointCount = Number.isFinite(responseCount) && responseCount >= 0
          ? responseCount
          : unitPointCount(mapData);
        this.prepareColorBuckets();
        this.reset();
        return true;
      },
      onAdd(map) {
        this.map = map;
        this.zooming = Boolean(map.isZooming?.());
        this.zoomRefreshPending = this.zooming;
        if (this.zooming) this.zoomGeneration += 1;
        this.canvas = L.DomUtil.create("canvas", "maplibre-unit-point-layer");
        this.canvas.style.pointerEvents = "none";
        map.getContainer().appendChild(this.canvas);
        map.on("zoomstart", this.handleZoomStart, this);
        map.on("zoomend", this.handleZoomEnd, this);
        map.on("moveend", this.handleMoveEnd, this);
        map.on("resize viewreset", this.reset, this);
        map.on("mousemove", this.handleMouseMove, this);
        map.on("mouseout", this.closeTooltip, this);
        map.on("click", this.handleClick, this);
        window.addEventListener("resize", this.handleDisplayResizeBound);
        window.visualViewport?.addEventListener("resize", this.handleDisplayResizeBound);
        this.reset();
        this.canvasMapLayer = L.canvasLayer(this.canvas, this.canvasCoordinates());
        if (this.zooming) this.canvasMapLayer.setVisible(false);
        this.canvasMapLayer.addTo(map);
        this.canvasMapLayer.refresh();
      },
      onRemove(map) {
        this.closeTooltip();
        this.zoomGeneration += 1;
        if (this.zoomFallbackFrame !== null) cancelAnimationFrame(this.zoomFallbackFrame);
        this.zoomFallbackFrame = null;
        if (this.displayResizeFrame !== null) cancelAnimationFrame(this.displayResizeFrame);
        this.displayResizeFrame = null;
        map.off("zoomstart", this.handleZoomStart, this);
        map.off("zoomend", this.handleZoomEnd, this);
        map.off("moveend", this.handleMoveEnd, this);
        map.off("resize viewreset", this.reset, this);
        map.off("mousemove", this.handleMouseMove, this);
        map.off("mouseout", this.closeTooltip, this);
        map.off("click", this.handleClick, this);
        window.removeEventListener("resize", this.handleDisplayResizeBound);
        window.visualViewport?.removeEventListener("resize", this.handleDisplayResizeBound);
        this.canvasMapLayer?.remove();
        this.canvasMapLayer = null;
        this.canvas?.remove();
        this.canvas = null;
        this.map = null;
      },
      handleDisplayResize() {
        if (!this.map || this.displayResizeFrame !== null) return;
        this.displayResizeFrame = requestAnimationFrame(() => {
          this.displayResizeFrame = null;
          if (!this.map || this.zooming) return;
          this.map.invalidateSize();
          this.reset();
        });
      },
      handleZoomStart() {
        this.zoomGeneration += 1;
        this.zooming = true;
        this.zoomRefreshPending = true;
        if (this.zoomFallbackFrame !== null) cancelAnimationFrame(this.zoomFallbackFrame);
        this.zoomFallbackFrame = null;
        this.hitGrid = new Map();
        this.closeTooltip();
      },
      handleZoomEnd() {
        this.zooming = false;
        const generation = this.zoomGeneration;
        if (this.zoomFallbackFrame !== null) cancelAnimationFrame(this.zoomFallbackFrame);
        this.zoomFallbackFrame = requestAnimationFrame(() => {
          this.zoomFallbackFrame = null;
          if (
            !this.map
            || this.zooming
            || !this.zoomRefreshPending
            || generation !== this.zoomGeneration
          ) {
            return;
          }
          this.zoomRefreshPending = false;
          this.reset({ zoomGeneration: generation });
        });
      },
      handleMoveEnd() {
        if (!this.map || this.zooming || this.rotating) return;
        if (this.zoomRefreshPending) {
          const generation = this.zoomGeneration;
          this.zoomRefreshPending = false;
          if (this.zoomFallbackFrame !== null) cancelAnimationFrame(this.zoomFallbackFrame);
          this.zoomFallbackFrame = null;
          this.reset({ zoomGeneration: generation });
          return;
        }
        this.reset();
      },
      beginRotation() {
        if (!this.map || this.rotating) return false;
        this.rotating = true;
        this.hitGrid = new Map();
        this.closeTooltip();
        return true;
      },
      endRotation() {
        if (!this.map || !this.rotating) return;
        this.rotating = false;
        this.reset();
      },
      getBounds() {
        return this.bounds;
      },
      setRenderContext(nextScale, nextHotspotIndexes) {
        this.scale = nextScale;
        this.hotspotIndexes = nextHotspotIndexes;
        this.prepareColorBuckets();
        this.reset();
      },
      visibleCellRange(hitRadius, size) {
        if (!Number.isFinite(this.spatialBounds.minLatitude)) return null;
        const corners = [
          [-hitRadius, -hitRadius],
          [size.x + hitRadius, -hitRadius],
          [size.x + hitRadius, size.y + hitRadius],
          [-hitRadius, size.y + hitRadius],
        ].map((point) => this.map.containerPointToLatLng(point));
        const longitudes = corners.map((corner) => corner.lng).filter(Number.isFinite);
        const latitudes = corners.map((corner) => corner.lat).filter(Number.isFinite);
        if (longitudes.length !== 4 || latitudes.length !== 4) return null;
        const west = Math.max(this.spatialBounds.minLongitude, Math.min(...longitudes));
        const east = Math.min(this.spatialBounds.maxLongitude, Math.max(...longitudes));
        const south = Math.max(this.spatialBounds.minLatitude, Math.min(...latitudes));
        const north = Math.min(this.spatialBounds.maxLatitude, Math.max(...latitudes));
        if (west > east || south > north) return null;
        const minimum = this.spatialCell(south, west);
        const maximum = this.spatialCell(north, east);
        return {
          minX: Math.min(minimum.x, maximum.x),
          maxX: Math.max(minimum.x, maximum.x),
          minY: Math.min(minimum.y, maximum.y),
          maxY: Math.max(minimum.y, maximum.y),
        };
      },
      projectUnitPoint(index, size = this.map.getSize()) {
        const unrotatedX = (this.worldX[index] * this.projectionScale) - this.projectionMinX;
        const unrotatedY = (this.worldY[index] * this.projectionScale) - this.projectionMinY;
        const offsetX = unrotatedX - (size.x / 2);
        const offsetY = unrotatedY - (size.y / 2);
        const cosine = Number.isFinite(this.projectionBearingCos) ? this.projectionBearingCos : 1;
        const sine = Number.isFinite(this.projectionBearingSin) ? this.projectionBearingSin : 0;
        return {
          x: (size.x / 2) + (offsetX * cosine) + (offsetY * sine),
          y: (size.y / 2) - (offsetX * sine) + (offsetY * cosine),
        };
      },
      canvasCoordinates() {
        const size = this.map.getSize();
        if (!(size.x > 0) || !(size.y > 0)) return null;
        const coordinates = [
          [0, 0],
          [size.x, 0],
          [size.x, size.y],
          [0, size.y],
        ].map((point) => {
          const latLng = this.map.containerPointToLatLng(point);
          return [latLng.lng, latLng.lat];
        });
        return coordinates.every((coordinate) => coordinate.every(Number.isFinite))
          ? coordinates
          : null;
      },
      syncCanvasMapLayer({
        visible = true,
        revealAfterRender = false,
        zoomGeneration = this.zoomGeneration,
      } = {}) {
        if (!this.canvasMapLayer) return;
        const coordinates = this.canvasCoordinates();
        if (!coordinates) {
          this.canvasMapLayer.setVisible(false);
          return;
        }
        const deferReveal = Boolean(visible && revealAfterRender);
        this.canvasMapLayer
          .setVisible(deferReveal ? false : visible)
          .setCoordinates(coordinates)
          .refresh({
            afterRender: deferReveal
              ? () => {
                if (
                  !this.map
                  || !this.canvasMapLayer
                  || this.zooming
                  || zoomGeneration !== this.zoomGeneration
                  || this.canvas?.style.visibility === "hidden"
                ) {
                  return;
                }
                this.canvasMapLayer.setVisible(true);
                bringBaseLabelsToFront();
              }
              : null,
          });
        bringBaseLabelsToFront();
      },
      reset(options = {}) {
        if (!this.map || !this.canvas) return;
        const revealAfterRender = Boolean(
          options?.revealAfterRender
          || this.zooming
          || this.zoomRefreshPending
        );
        const zoomGeneration = Number.isInteger(options?.zoomGeneration)
          ? options.zoomGeneration
          : this.zoomGeneration;
        const visibleBounds = unitBoundsFromMap(this.map.getBounds());
        if (
          this.coverageBounds
          && visibleBounds
          && !unitBoundsContains(this.coverageBounds, visibleBounds)
        ) {
          this.canvas.style.visibility = "hidden";
          this.hitGrid = new Map();
          this.closeTooltip();
          this.syncCanvasMapLayer({ visible: false });
          return;
        }
        this.canvas.style.visibility = "";
        const size = this.map.getSize();
        const ratio = window.devicePixelRatio || 1;
        this.canvas.width = Math.max(1, Math.round(size.x * ratio));
        this.canvas.height = Math.max(1, Math.round(size.y * ratio));
        this.canvas.style.width = `${size.x}px`;
        this.canvas.style.height = `${size.y}px`;
        const context = this.canvas.getContext("2d");
        context.setTransform(1, 0, 0, 1, 0, 0);
        context.clearRect(0, 0, this.canvas.width, this.canvas.height);
        this.hitGrid = new Map();
        const fittedZoom = this.map.getBoundsZoom(this.bounds, mapFitOptions("unit"));
        const renderStyle = unitPointRenderStyle({
          zoom: this.map.getZoom(),
          fittedZoom,
          pointCount: this.plottedPointCount,
          pixelRatio: ratio,
        });
        const pointRadius = renderStyle.radius;
        this.fittedZoom = fittedZoom;
        this.pointDiameter = renderStyle.diameter;
        this.pointRadius = pointRadius;
        this.singleDevicePixel = renderStyle.singleDevicePixel;
        if (!renderStyle.singleDevicePixel) context.setTransform(ratio, 0, 0, ratio, 0, 0);
        const hitRadius = unitPointHitRadius(pointRadius);
        this.hitRadius = hitRadius;
        const cellRange = this.visibleCellRange(hitRadius, size);
        if (!cellRange) {
          this.syncCanvasMapLayer({ revealAfterRender, zoomGeneration });
          return;
        }
        const pixelBounds = this.map.getPixelBounds();
        this.projectionScale = 2 ** this.map.getZoom();
        this.projectionMinX = pixelBounds.min.x;
        this.projectionMinY = pixelBounds.min.y;
        const bearingRadians = normaliseMapBearing(this.map.getBearing()) * (Math.PI / 180);
        this.projectionBearingCos = Math.cos(bearingRadians);
        this.projectionBearingSin = Math.sin(bearingRadians);
        this.hitGridStride = Math.ceil(size.x / MAP_POINT_GRID_SIZE) + 4;
        const opacityValue = Number(state.mapOpacity);
        const mapOpacity = Number.isFinite(opacityValue) ? Math.max(0, Math.min(1, opacityValue)) : 1;
        const baseStrokeOpacity = pointRadius < 2 ? 0 : (pointRadius < 3 ? 0.35 : 0.65);
        let activeFillColor = "";
        let activeFillOpacity = -1;
        for (let cellY = cellRange.minY; cellY <= cellRange.maxY; cellY += 1) {
          for (let cellX = cellRange.minX; cellX <= cellRange.maxX; cellX += 1) {
            const cell = (cellY * MAP_UNIT_SPATIAL_GRID_SIZE) + cellX;
            const start = this.spatialOffsets[cell];
            const end = this.spatialOffsets[cell + 1];
            for (let offset = start; offset < end; offset += 1) {
              const index = this.spatialIndexes[offset];
              let bucket = this.colorBuckets[index];
              if (bucket === 255) {
                const value = finiteNumber(this.metricPoints?.value?.[index]);
                bucket = value === null ? 254 : this.scale.bucket(value);
                this.colorBuckets[index] = bucket;
              }
              if (bucket === 254) continue;
              const projectedPoint = this.projectUnitPoint(index, size);
              const pointX = projectedPoint.x;
              const pointY = projectedPoint.y;
              if (pointX < -hitRadius || pointY < -hitRadius || pointX > size.x + hitRadius || pointY > size.y + hitRadius) {
                continue;
              }
              const gridX = Math.floor(pointX / MAP_POINT_GRID_SIZE) + 2;
              const gridY = Math.floor(pointY / MAP_POINT_GRID_SIZE) + 2;
              const gridKey = (gridY * this.hitGridStride) + gridX;
              if (!this.hitGrid.has(gridKey)) this.hitGrid.set(gridKey, []);
              this.hitGrid.get(gridKey).push(index);
              const muted = Boolean(this.hotspotIndexes && !this.hotspotIndexes.has(index));
              const fillColor = muted ? MAP_MUTED_COLOR : this.scale.palette[bucket];
              const fillOpacity = muted ? Math.min(mapOpacity, 0.28) : mapOpacity;
              if (fillOpacity !== activeFillOpacity) {
                context.globalAlpha = fillOpacity;
                activeFillOpacity = fillOpacity;
              }
              if (fillColor !== activeFillColor) {
                context.fillStyle = fillColor;
                activeFillColor = fillColor;
              }
              if (renderStyle.singleDevicePixel) {
                context.fillRect(Math.round(pointX * ratio), Math.round(pointY * ratio), 1, 1);
              } else if (pointRadius <= 1) {
                context.fillRect(pointX - pointRadius, pointY - pointRadius, renderStyle.diameter, renderStyle.diameter);
              } else {
                context.beginPath();
                context.arc(pointX, pointY, pointRadius, 0, Math.PI * 2);
                context.fill();
                const strokeOpacity = (muted ? Math.min(baseStrokeOpacity, 0.25) : baseStrokeOpacity) * mapOpacity;
                if (strokeOpacity > 0) {
                  context.globalAlpha = strokeOpacity;
                  context.strokeStyle = "#000000";
                  context.lineWidth = pointRadius < 3 ? 0.5 : 0.75;
                  context.stroke();
                  activeFillOpacity = -1;
                }
              }
            }
          }
        }
        context.globalAlpha = 1;
        this.syncCanvasMapLayer({ revealAfterRender, zoomGeneration });
      },
      pointRow(index) {
        return {
          key: this.geometryPoints.key[index],
          row_count: this.geometryPoints.row_count?.[index],
          numerator: this.metricPoints.numerator?.[index],
          denominator: this.metricPoints.denominator?.[index],
          value: this.metricPoints.value?.[index],
          latitude: Number(this.geometryPoints.latitude[index]),
          longitude: Number(this.geometryPoints.longitude[index]),
        };
      },
      rowForKey(rawKey) {
        const key = String(rawKey);
        for (let index = 0; index < this.pointCount; index += 1) {
          if (String(this.geometryPoints.key[index]) === key) return this.pointRow(index);
        }
        return null;
      },
      findNearest(containerPoint) {
        if (!this.map || !this.hitGrid) return null;
        const size = this.map.getSize();
        const hitRadius = this.hitRadius || 6;
        const radiusSquared = hitRadius * hitRadius;
        let nearest = null;
        let nearestDistance = radiusSquared;
        const gridX = Math.floor(containerPoint.x / MAP_POINT_GRID_SIZE);
        const gridY = Math.floor(containerPoint.y / MAP_POINT_GRID_SIZE);
        for (let dxCell = -1; dxCell <= 1; dxCell += 1) {
          for (let dyCell = -1; dyCell <= 1; dyCell += 1) {
            const key = ((gridY + dyCell + 2) * this.hitGridStride) + gridX + dxCell + 2;
            const indexes = this.hitGrid.get(key) || [];
            for (const index of indexes) {
              const projectedPoint = this.projectUnitPoint(index, size);
              const pointX = projectedPoint.x;
              const pointY = projectedPoint.y;
              const dx = pointX - containerPoint.x;
              const dy = pointY - containerPoint.y;
              const distance = dx * dx + dy * dy;
              if (distance <= nearestDistance) {
                nearest = index;
                nearestDistance = distance;
              }
            }
          }
        }
        return nearest === null ? null : this.pointRow(nearest);
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
        const latLng = L.latLng(nearest.latitude, nearest.longitude);
        const popup = L.popup({ maxWidth: MAP_POPUP_MAX_WIDTH })
          .setLatLng(latLng)
          .setContent(mapPopupHtml(String(nearest.key || "Unknown"), nearest, this.data));
        activeMapPopupSelection = {
          key: String(nearest.key || "Unknown"),
          level: "unit",
          joinColumn: this.data?.join_column || postcodeColumn("unit"),
          latLng,
          popup,
        };
        popup.openOn(this.map);
      },
      closeTooltip() {
        if (this.tooltip && this.map?.hasLayer(this.tooltip)) {
          this.map.removeLayer(this.tooltip);
        }
      },
    }))(data, scale, hotspotIndexes, geometryKey);
  }

  async function renderMap(data, geoJson) {
    return measureToolRender("uk_map", () => renderMapContents(data, geoJson));
  }

  function refreshOpenMapPopup(data) {
    const selection = activeMapPopupSelection;
    const popup = ukMap?._popup;
    if (!selection || !popup || !ukMap.hasLayer(popup) || selection.level !== data?.level) return false;
    let row = null;
    if (data.level === "unit") {
      row = ukMapPointLayer?.rowForKey?.(selection.key) || null;
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

  async function renderMapContents(data, geoJson) {
    if (data.level === "unit") {
      await renderUnitMap(data);
      return;
    }
    state.lastMapData = data;
    state.renderedMapLevel = data.level;
    if (!ukMap) await initMap();
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
    clearMapLabelLayer();
    ukMapLayer = cachedPolygonLayer.layer;
    applyMapPolygonStyles();
    if (!ukMap.hasLayer(ukMapLayer)) ukMapLayer.addTo(ukMap);
    if (data.level === "area") renderMapLabels(data, summaries, hotspotKeys);
    bringBaseLabelsToFront();

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
    await ukMap.whenRenderComplete?.();
  }

  async function renderUnitMap(data) {
    state.lastMapData = data;
    state.renderedMapLevel = data.level;
    if (!ukMap) await initMap();
    syncFloatingMapControl();
    const scale = makeUnitPointScale(data);
    const hotspotIndexes = mapUnitHotspotIndexes(data);
    if (ukMapLayer) {
      ukMap.removeLayer(ukMapLayer);
      ukMapLayer = null;
    }
    state.mapPolygonRenderContext = null;
    clearMapLabelLayer();
    const geometryKey = data._unitGeometryKey || "";
    const reuseLayer = Boolean(
      data._unitGeometryReused
      && ukMapPointLayer?.geometryKey === geometryKey
      && ukMapPointLayer?.setData(data, scale, hotspotIndexes)
    );
    if (!reuseLayer) {
      if (ukMapPointLayer) ukMap.removeLayer(ukMapPointLayer);
      ukMapPointLayer = makeUnitPointLayer(data, scale, hotspotIndexes, geometryKey).addTo(ukMap);
    }
    bringBaseLabelsToFront();

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
      : "no units missing KPI value";
    const missingCoordinateText = missingCoordinateCount > 0
      ? `${missingCoordinateCount.toLocaleString()} ${missingCoordinateCount === 1 ? "unit" : "units"} missing coordinates`
      : "no units missing coordinates";
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
    await ukMap.whenRenderComplete?.();
  }

  function approximateMapAreaBoundsArea(bounds) {
    if (!bounds?.isValid?.()) return null;
    const south = Number(bounds.getSouth());
    const west = Number(bounds.getWest());
    const north = Number(bounds.getNorth());
    const east = Number(bounds.getEast());
    if (![south, west, north, east].every(Number.isFinite)) return null;
    const latitude = ((south + north) / 2) * (Math.PI / 180);
    const width = Math.abs(east - west) * Math.max(0, Math.cos(latitude));
    const height = Math.abs(north - south);
    const area = width * height;
    return Number.isFinite(area) && area > 0 ? area : null;
  }

  function prepareMapAreaLabelZoomOffsets() {
    if (!ukMapLayer || ukMapLayer._lucidumAreaLabelOffsetsPrepared) return;
    const entries = [];
    ukMapLayer.eachLayer((layer) => {
      const area = approximateMapAreaBoundsArea(layer.getBounds?.());
      if (area !== null) entries.push({ layer, area });
    });
    const sortedAreas = entries.map((entry) => entry.area).sort((left, right) => left - right);
    const middle = Math.floor(sortedAreas.length / 2);
    const medianArea = sortedAreas.length % 2
      ? sortedAreas[middle]
      : ((sortedAreas[middle - 1] || 0) + (sortedAreas[middle] || 0)) / 2;
    entries.forEach(({ layer, area }) => {
      const rawOffset = medianArea > 0 ? 0.5 * Math.log2(area / medianArea) : 0;
      layer._lucidumAreaLabelZoomOffset = Math.max(
        MAP_AREA_LABEL_MIN_ZOOM_OFFSET,
        Math.min(MAP_AREA_LABEL_MAX_ZOOM_OFFSET, rawOffset),
      );
    });
    ukMapLayer._lucidumAreaLabelOffsetsPrepared = true;
  }

  function updateMapAreaLabelSize() {
    if (!ukMap || !ukMapLayer || !ukMapLabelLayer || state.mapAreaLabels !== "on") return;
    const fittedZoom = ukMap.getBoundsZoom(ukMapLayer.getBounds(), mapFitOptions("area"));
    const currentZoom = Number(ukMap.getZoom());
    if (!Number.isFinite(fittedZoom) || !Number.isFinite(currentZoom)) return;
    const baseFontSize = MAP_AREA_LABEL_BASE_FONT_SIZE
      + (MAP_AREA_LABEL_GROWTH_PER_ZOOM * (currentZoom - fittedZoom));
    const container = ukMap.getContainer();
    container.style.setProperty("--map-area-label-min-size", `${MAP_AREA_LABEL_MIN_FONT_SIZE}px`);
    container.style.setProperty("--map-area-label-base-size", `${baseFontSize}px`);
    container.style.setProperty("--map-area-label-max-size", `${MAP_AREA_LABEL_MAX_FONT_SIZE}px`);
  }

  function scheduleMapAreaLabelSizeUpdate() {
    if (mapAreaLabelSizeFrame !== null) return;
    mapAreaLabelSizeFrame = requestAnimationFrame(() => {
      mapAreaLabelSizeFrame = null;
      updateMapAreaLabelSize();
    });
  }

  function renderMapLabels(data, summaries, hotspotKeys) {
    if (data.level !== "area" || state.mapAreaLabels !== "on" || !ukMapLayer) return;
    prepareMapAreaLabelZoomOffsets();
    ukMapLabelLayer = L.layerGroup().addTo(ukMap);
    ukMapLayer.eachLayer((layer) => {
      const key = mapPolygonLayerKey(layer);
      const row = summaries.get(key);
      const value = finiteNumber(row?.value);
      if (value === null) return;
      if (hotspotKeys && !hotspotKeys.has(key)) return;
      const bounds = layer.getBounds?.();
      if (!bounds?.isValid()) return;
      const zoomOffset = Number(layer._lucidumAreaLabelZoomOffset) || 0;
      const sizeOffset = MAP_AREA_LABEL_GROWTH_PER_ZOOM * zoomOffset;
      const html = `<div class="map-label" data-map-area-key="${escapeHtml(key)}" data-map-area-zoom-offset="${zoomOffset}" style="--map-area-label-size-offset:${sizeOffset}px">${escapeHtml(key)}<br>${escapeHtml(formatLineValue(value))}</div>`;
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
    updateMapAreaLabelSize();
  }

  function redrawMapLabelsInPlace() {
    clearMapLabelLayer();
    const context = state.mapPolygonRenderContext;
    if (state.lastMapData?.level !== "area" || !context) return;
    renderMapLabels(context.data, context.summaries, context.hotspotKeys);
    bringBaseLabelsToFront();
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
        const hotspotIndexes = mapUnitHotspotIndexes(state.lastMapData);
        ukMapPointLayer.setRenderContext(scale, hotspotIndexes);
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
    updateMapSliderProgress(el("mapHotspots"));
  }

  function syncFloatingMapControl() {
    const metric = el("mapControlMetric");
    const kpiName = String(getSelectedKpi()?.name || "").trim();
    metric.textContent = kpiName;
    metric.title = kpiName;
    metric.hidden = !kpiName;
    el("mapMetricSeparator").hidden = !kpiName;
    syncActiveFilterLabels();
    document.querySelectorAll(".map-palette-button").forEach((button) => {
      button.classList.toggle("active", button.dataset.palette === state.mapPalette);
    });
    const unitMode = state.mapLevel === "unit";
    el("mapHotspots").value = String(state.mapHotspots);
    el("mapHotspotsValue").textContent = formatHotspotSliderValue(state.mapHotspots);
    document.querySelectorAll("[data-map-opacity]").forEach((button) => {
      const active = Number(button.dataset.mapOpacity) === normaliseMapOpacity(state.mapOpacity);
      button.classList.toggle("active", active);
      button.setAttribute("aria-pressed", String(active));
    });
    const lineWeightControl = el("mapLineWeightControl");
    if (lineWeightControl) lineWeightControl.hidden = unitMode;
    document.querySelectorAll("[data-map-line-weight]").forEach((button) => {
      const active = Number(button.dataset.mapLineWeight) === normaliseMapBorderWeight(state.mapLineWeight);
      button.classList.toggle("active", active);
      button.setAttribute("aria-pressed", String(active));
      button.disabled = unitMode;
    });
    lineWeightControl?.classList.toggle("disabled", unitMode);
    const dotSizeControl = el("mapDotSizeControl");
    if (dotSizeControl) dotSizeControl.hidden = !unitMode;
    document.querySelectorAll("[data-map-dot-size-mode]").forEach((button) => {
      const active = button.dataset.mapDotSizeMode === normaliseMapDotSizeMode(state.mapDotSizeMode);
      button.classList.toggle("active", active);
      button.setAttribute("aria-pressed", String(active));
      button.disabled = !unitMode;
    });
    const labelHidden = state.mapLevel !== "area";
    const labelControl = el("mapLabelControl");
    if (labelControl) labelControl.hidden = labelHidden;
    document.querySelectorAll("[data-map-area-labels]").forEach((button) => {
      const active = button.dataset.mapAreaLabels === normaliseMapAreaLabels(state.mapAreaLabels);
      button.classList.toggle("active", active);
      button.setAttribute("aria-pressed", String(active));
      button.disabled = labelHidden;
    });
    const smoothingHidden = state.mapLevel !== "sector";
    const smoothingControl = el("mapSmoothingControl");
    if (smoothingControl) smoothingControl.hidden = smoothingHidden;
    const smoothingSaveButton = el("mapSaveSmoothingBtn");
    if (smoothingSaveButton) smoothingSaveButton.disabled = smoothingHidden || sectorSmoothingSavePending;
    document.querySelectorAll("[data-map-smoothing]").forEach((button) => {
      const active = Number(button.dataset.mapSmoothing) === Number(state.mapSmoothingLevel);
      button.classList.toggle("active", active);
      button.setAttribute("aria-pressed", String(active));
      button.disabled = smoothingHidden;
    });
    smoothingControl?.classList.toggle("disabled", smoothingHidden);
    syncMapSliderProgressStyles();
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

  function makeQuantileScaleFromValues(rawValues, { sorted = false } = {}) {
    const palette = interpolateMapPalette(activeMapPalette(), MAP_COLOR_BUCKETS);
    const values = sorted ? rawValues : rawValues.sort((a, b) => a - b);
    const thresholds = quantileThresholds(values, palette.length);
    const bucket = (value) => {
      let low = 0;
      let high = thresholds.length;
      while (low < high) {
        const middle = (low + high) >> 1;
        if (value > thresholds[middle]) low = middle + 1;
        else high = middle;
      }
      return Math.min(low, palette.length - 1);
    };
    return {
      palette,
      legendPalette: legendPaletteFromMapPalette(palette),
      values,
      thresholds,
      legendThresholds: quantileThresholds(values, MAP_LEGEND_BUCKETS),
      bucket,
      color(value) {
        if (value === null) return MAP_MISSING_COLOR;
        return palette[bucket(value)];
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

  function syncMapToolbarCollapseButton() {
    const button = el("mapControlReset");
    if (!button) return;
    const collapsed = Boolean(state.mapToolbarCollapsed);
    const label = collapsed ? "Expand map controls" : "Collapse map controls";
    button.title = label;
    button.setAttribute("aria-label", label);
    button.setAttribute("aria-expanded", String(!collapsed));
  }

  function syncMapToolbarVisibility() {
    const toolbar = el("mapToolbar");
    const infoStrip = el("mapInfoStrip");
    const hidden = state.tool !== "uk_map" || Boolean(state.mapToolbarCollapsed);
    [toolbar, infoStrip].forEach((region) => {
      region.classList.toggle("hidden", hidden);
      region.toggleAttribute("inert", hidden);
    });
    if (hidden) {
      toolbar.setAttribute("aria-hidden", "true");
      infoStrip.setAttribute("aria-hidden", "true");
      if (toolbar.contains(document.activeElement) || infoStrip.contains(document.activeElement)) {
        document.activeElement?.blur?.();
      }
    } else {
      toolbar.removeAttribute("aria-hidden");
      infoStrip.removeAttribute("aria-hidden");
    }
    syncMapToolbarCollapseButton();
  }

  function setMapToolbarCollapsed(collapsed) {
    const nextCollapsed = Boolean(collapsed);
    if (state.mapToolbarCollapsed === nextCollapsed) {
      syncMapToolbarVisibility();
      return;
    }
    state.mapToolbarCollapsed = nextCollapsed;
    syncMapToolbarVisibility();
    scheduleMapViewportSync({ mode: "preserve" });
  }

  function toggleMapToolbarCollapsed() {
    setMapToolbarCollapsed(!state.mapToolbarCollapsed);
  }

  function bindMapToolbarControls() {
    syncMapToolbarVisibility();
    syncMapLegendCollapseButton();
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
    document.querySelectorAll("[data-map-dot-size-mode]").forEach((button) => {
      button.addEventListener("click", () => {
        const mode = normaliseMapDotSizeMode(button.dataset.mapDotSizeMode);
        if (mode === state.mapDotSizeMode) return;
        state.mapDotSizeMode = mode;
        clearActiveMapFavourite({ force: true });
        syncFloatingMapControl();
        if (state.mapLevel === "unit") redrawMapInPlace();
      });
    });
    document.querySelectorAll("[data-map-area-labels]").forEach((button) => {
      button.addEventListener("click", () => {
        const mode = normaliseMapAreaLabels(button.dataset.mapAreaLabels);
        if (mode === state.mapAreaLabels) return;
        state.mapAreaLabels = mode;
        clearActiveMapFavourite({ force: true });
        syncFloatingMapControl();
        if (state.mapLevel === "area") redrawMapLabelsInPlace();
      });
    });
    document.querySelectorAll("[data-map-line-weight]").forEach((button) => {
      button.addEventListener("click", () => {
        const weight = normaliseMapBorderWeight(button.dataset.mapLineWeight);
        if (weight === state.mapLineWeight) return;
        state.mapLineWeight = weight;
        clearActiveMapFavourite({ force: true });
        syncFloatingMapControl();
        if (state.mapLevel !== "unit") redrawMapInPlace();
      });
    });
    document.querySelectorAll("[data-map-opacity]").forEach((button) => {
      button.addEventListener("click", () => {
        const opacity = normaliseMapOpacity(button.dataset.mapOpacity);
        if (opacity === state.mapOpacity) return;
        state.mapOpacity = opacity;
        clearActiveMapFavourite({ force: true });
        syncFloatingMapControl();
        redrawMapInPlace();
      });
    });
    el("mapHotspots").addEventListener("input", (event) => {
      state.mapHotspots = Number(event.target.value);
      updateMapSliderProgress(event.target);
      clearActiveMapFavourite({ force: true });
      redrawMapInPlace();
    });
    document.querySelectorAll("[data-map-smoothing]").forEach((button) => {
      button.addEventListener("click", () => {
        const level = Math.max(0, Math.min(5, Math.round(Number(button.dataset.mapSmoothing) || 0)));
        if (level === state.mapSmoothingLevel) return;
        state.mapSmoothingLevel = level;
        clearActiveMapFavourite({ force: true });
        syncFloatingMapControl();
        if (state.mapLevel !== "sector") return;
        captureMapView("smoothing-change");
        refreshMap();
      });
    });
    el("mapSaveSmoothingBtn").addEventListener("click", saveSectorSmoothingParquet);
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
    initMap().then(() => {
      syncMapToolbarVisibility();
      syncFloatingMapControl();
      syncMapControls();
      requestAnimationFrame(() => {
        scheduleMapViewportSync({ mode: "preserve" });
      });
    }).catch((error) => {
      setStatus("MapLibre needs WebGL2 support in the current browser.", true);
      setChartMessage(error?.message || "MapLibre could not initialise the map.");
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
    bindSettingsStripOverflowCue(el("mapToolbarScroll"));
    bindMapToolbarControls();
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
    syncMapToolbarVisibility();
    scheduleMapViewportSync(options);
  }

  function resize() {
    resizeMap();
  }

  function refreshTheme() {
    syncBaseMapForTheme();
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
