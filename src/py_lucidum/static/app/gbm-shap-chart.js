const SHAP_RED = "#d13f3f";
const SHAP_RED_RIBBONS = [
  "rgba(209, 63, 63, 0.08)",
  "rgba(209, 63, 63, 0.12)",
  "rgba(209, 63, 63, 0.16)",
  "rgba(209, 63, 63, 0.20)",
  "rgba(209, 63, 63, 0.25)",
  "rgba(209, 63, 63, 0.31)",
];
const LINE_COLORS = ["#4fb99f", "#ff7f50", "#8aa1d6", "#b779d6", "#e7b84b", "#5aa2d6", "#d96a8a", "#84b547"];
const AXIS_TARGET_INTERVALS = 6;
const SHAP_VALUE_AXIS_TARGET_INTERVALS = 20;
const SURFACE_AXIS_LABEL_FONT_SIZE = 10;
const SURFACE_AXIS_NAME_FONT_SIZE = 11;
const SURFACE_BOX_WIDTH = 100;
const SURFACE_BOX_DEPTH = 74;
let echartsGlPromise = null;

export async function ensureShapChartLibraries(plotType) {
  if (plotType !== "surface") return false;
  if (window.__lucidumEchartsGlLoaded) return false;
  if (!echartsGlPromise) {
    echartsGlPromise = new Promise((resolve, reject) => {
      const script = document.createElement("script");
      script.src = "/static/vendor/echarts-gl/echarts-gl.min.js";
      script.onload = () => {
        window.__lucidumEchartsGlLoaded = true;
        resolve();
      };
      script.onerror = () => reject(new Error("ECharts GL did not load"));
      document.head.append(script);
    });
  }
  await echartsGlPromise;
  return true;
}

export function shapChartOption(payload, theme = {}) {
  if (!payload || typeof payload !== "object") return emptyOption("Select a SHAP feature", theme);
  const common = commonOption(payload, theme);
  if (payload.plot_type === "flame") return flameOption(payload, common, theme);
  if (payload.plot_type === "box") return boxOption(payload, common, theme);
  if (payload.plot_type === "surface") return surfaceOption(payload, common, theme);
  if (payload.plot_type === "lines") return linesOption(payload, common, theme);
  if (payload.plot_type === "heatmap") return heatmapOption(payload, common, theme);
  return emptyOption("Select a SHAP feature", theme);
}

export function emptyOption(message, theme = {}) {
  return {
    backgroundColor: theme.panel || "transparent",
    title: {
      text: message,
      left: "center",
      top: "middle",
      textStyle: { color: theme.muted || "#64748b", fontSize: 14, fontWeight: 700 },
    },
  };
}

function commonOption(payload, theme) {
  return {
    backgroundColor: theme.panel || "transparent",
    animation: false,
    color: LINE_COLORS,
    title: {
      text: payload.title || "SHAP",
      left: "center",
      top: 10,
      textStyle: { color: theme.text || "#334155", fontSize: 14, fontWeight: 800 },
    },
    tooltip: { trigger: "axis", confine: true },
    legend: {
      top: 34,
      right: 12,
      textStyle: { color: theme.text || "#334155", fontSize: 11 },
      type: "scroll",
    },
    grid: {
      top: 72,
      right: 64,
      bottom: 54,
      left: 64,
      containLabel: true,
    },
  };
}

function flameOption(payload, common, theme) {
  const rows = payload.rows || [];
  const ribbons = [
    ["p0", "p100", "Min-Max"],
    ["p5", "p95", "5-95"],
    ["p10", "p90", "10-90"],
    ["p20", "p80", "20-80"],
    ["p30", "p70", "30-70"],
    ["p40", "p60", "40-60"],
  ];
  const series = [];
  ribbons.forEach(([lowKey, highKey, label], index) => {
    const ribbon = flameRibbonSeries(rows, lowKey, highKey, label, SHAP_RED_RIBBONS[index]);
    if (ribbon) series.push(ribbon);
  });
  series.push({
    name: "Median",
    type: "line",
    data: rows.map((row) => [row.x, numberOrNull(row.p50)]),
    symbol: "none",
    itemStyle: { color: SHAP_RED },
    lineStyle: { color: SHAP_RED, width: 2, type: "dashed" },
    markLine: referenceMarkLine(theme, referenceLineValue(payload)),
  });
  return {
    ...common,
    legend: centeredLegend(common.legend),
    tooltip: { trigger: "axis", confine: true, formatter: flameTooltipFormatter(rows, payload) },
    xAxis: valueAxis(payload.x_feature, theme, payload.x_domain, { exactDomain: true }),
    yAxis: shapValueAxis(payload, payload.y_label || "SHAP", theme, payload.y_domain),
    series,
  };
}

function boxOption(payload, common, theme) {
  const rows = payload.rows || [];
  const labels = rows.map((row) => String(row.level));
  const feature = featureInfoForName(payload, payload.x_feature);
  return {
    ...common,
    legend: { show: false },
    grid: { ...common.grid, top: 56 },
    tooltip: { trigger: "item", confine: true, formatter: (params) => boxTooltip(params, rows, feature, payload) },
    xAxis: categoryAxis(labels, payload.x_feature, theme, feature, { axis: "x" }),
    yAxis: shapValueAxis(payload, payload.y_label || "SHAP", theme, payload.y_domain),
    dataZoom: labels.length > 60 ? [{ type: "inside" }, { type: "slider", height: 18, bottom: 18 }] : [],
    series: [
      {
        name: "SHAP",
        type: "boxplot",
        data: rows.map((row) => [row.p0, row.p25, row.p50, row.p75, row.p100]),
        itemStyle: { borderColor: SHAP_RED, color: "rgba(209, 63, 63, 0.16)" },
        markLine: referenceMarkLine(theme, referenceLineValue(payload)),
      },
      {
        name: "Mean",
        type: "scatter",
        data: rows.map((row, index) => [index, numberOrNull(row.mean)]),
        symbolSize: 4,
        itemStyle: { color: SHAP_RED },
      },
    ],
  };
}

function surfaceOption(payload, common, theme) {
  const rows = payload.rows || [];
  const dataShape = surfaceDataShape(payload, rows);
  if (!rows.length || dataShape[0] < 2 || dataShape[1] < 2) return emptyOption("No SHAP surface data", theme);
  const zValues = rows.map((row) => numberOrNull(row.z)).filter((value) => value !== null);
  if (!zValues.length) return emptyOption("No SHAP surface data", theme);
  const extent = numericExtent(zValues);
  return {
    backgroundColor: common.backgroundColor,
    animation: false,
    title: common.title,
    tooltip: { confine: true, formatter: (params) => surfaceTooltip(params, payload) },
    visualMap: {
      min: extent.min,
      max: extent.max,
      calculable: true,
      formatter: (value) => formatVisualMapValue(value, payload),
      right: 10,
      top: 80,
      inRange: { color: ["#1d4ed8", "#f8fafc", "#b91c1c"] },
      textStyle: { color: theme.text || "#334155" },
    },
    grid3D: {
      top: 34,
      left: 0,
      right: 0,
      bottom: 54,
      boxWidth: SURFACE_BOX_WIDTH,
      boxDepth: SURFACE_BOX_DEPTH,
      viewControl: { projection: "perspective", autoRotate: false },
      axisPointer: { show: true },
      splitLine: { lineStyle: { color: theme.line || "#e2e8f0" } },
    },
    xAxis3D: axis3D(payload.x_feature, theme, payload.x_domain),
    yAxis3D: axis3D(payload.y_feature, theme, payload.y_domain),
    zAxis3D: axis3D(payload.z_label || "SHAP", theme, null, { formatter: valueFormatterForPayload(payload), targetIntervals: SHAP_VALUE_AXIS_TARGET_INTERVALS }),
    series: [
      {
        type: "surface",
        data: rows.map(surfaceDataPoint),
        dataShape,
        shading: "lambert",
        itemStyle: { opacity: 0.96 },
      },
    ],
  };
}

function linesOption(payload, common, theme) {
  const seriesRows = new Map();
  for (const row of payload.rows || []) {
    const key = String(row.series);
    if (!seriesRows.has(key)) seriesRows.set(key, []);
    seriesRows.get(key).push([row.x, numberOrNull(row.y)]);
  }
  return {
    ...common,
    tooltip: { trigger: "axis", confine: true, valueFormatter: (value) => formatShapValue(value, payload) },
    xAxis: valueAxis(payload.x_feature, theme, payload.x_domain),
    yAxis: shapValueAxis(payload, payload.y_label || "SHAP", theme, payload.y_domain),
    series: [...seriesRows.entries()].map(([name, data]) => ({
      name,
      type: "line",
      data,
      symbol: "none",
      lineStyle: { width: 2 },
      markLine: name === [...seriesRows.keys()][0] ? referenceMarkLine(theme, referenceLineValue(payload)) : undefined,
    })),
  };
}

function heatmapOption(payload, common, theme) {
  const rows = payload.rows || [];
  const xFeature = featureInfoForName(payload, payload.x_feature);
  const yFeature = featureInfoForName(payload, payload.y_feature);
  const xLabels = sortedCategoryLabels(rows, "x", "x_sort", xFeature);
  const yLabels = sortedCategoryLabels(rows, "y", "y_sort", yFeature);
  const values = rows.map((row) => numberOrNull(row.z)).filter((value) => value !== null);
  const extent = numericExtent(values);
  const xIndex = new Map(xLabels.map((label, index) => [label, index]));
  const yIndex = new Map(yLabels.map((label, index) => [label, index]));
  return {
    ...common,
    tooltip: {
      trigger: "item",
      confine: true,
      formatter: (params) => heatmapTooltip(params, payload, xLabels, yLabels, xFeature, yFeature),
    },
    xAxis: categoryAxis(xLabels, payload.x_feature, theme, xFeature, { axis: "x" }),
    yAxis: categoryAxis(yLabels, payload.y_feature, theme, yFeature),
    visualMap: {
      min: extent.min,
      max: extent.max,
      calculable: true,
      formatter: (value) => formatVisualMapValue(value, payload),
      orient: "vertical",
      right: 8,
      top: 90,
      inRange: { color: ["#16a34a", "#f8fafc", "#dc2626"] },
      textStyle: { color: theme.text || "#334155" },
    },
    series: [
      {
        name: "SHAP",
        type: "heatmap",
        data: rows.map((row) => [xIndex.get(String(row.x)), yIndex.get(String(row.y)), numberOrNull(row.z)]),
        emphasis: { itemStyle: { borderColor: theme.text || "#334155", borderWidth: 1 } },
      },
    ],
  };
}

function valueAxis(name, theme, domain = null, options = {}) {
  const axis = {
    type: "value",
    name,
    scale: true,
    nameLocation: "middle",
    nameGap: 34,
    axisLine: { lineStyle: { color: theme.line || "#cbd5e1" } },
    axisLabel: { color: theme.text || "#334155", formatter: options.formatter || compactNumber },
    splitLine: { lineStyle: { color: theme.grid || "#e5e7eb" } },
    nameTextStyle: { color: theme.text || "#334155", fontWeight: 700 },
  };
  const bounds = options.exactDomain
    ? exactAxisBounds(domain, options.targetIntervals)
    : niceAxisBounds(domain, options.targetIntervals);
  if (bounds) {
    axis.min = bounds.min;
    axis.max = bounds.max;
    axis.interval = bounds.interval;
  }
  return axis;
}

function shapValueAxis(payload, name, theme, domain = null) {
  return valueAxis(name, theme, domain, {
    formatter: valueFormatterForPayload(payload),
    targetIntervals: SHAP_VALUE_AXIS_TARGET_INTERVALS,
  });
}

function categoryAxis(labels, name, theme, feature = null, options = {}) {
  const numericTickPolicy = numericCategoryTickPolicy(labels, feature);
  const rotate = labels.length > 25 ? 60 : 0;
  return {
    type: "category",
    data: labels,
    name,
    nameLocation: "middle",
    nameGap: categoryAxisNameGap(rotate, options),
    axisLine: { lineStyle: { color: theme.line || "#cbd5e1" } },
    axisLabel: {
      color: theme.text || "#334155",
      interval: numericTickPolicy?.interval || (labels.length > 80 ? "auto" : 0),
      rotate,
      formatter: (value) => formatCategoryLabel(value, feature),
    },
    splitLine: { show: true, lineStyle: { color: theme.grid || "#e5e7eb" } },
    nameTextStyle: { color: theme.text || "#334155", fontWeight: 700 },
  };
}

function categoryAxisNameGap(rotate, options = {}) {
  return options.axis === "x" && Number(rotate) > 0 ? 88 : 34;
}

function axis3D(name, theme, domain = null, options = {}) {
  const axis = {
    type: "value",
    name,
    nameTextStyle: { color: theme.text || "#334155", fontWeight: 700, fontSize: SURFACE_AXIS_NAME_FONT_SIZE },
    axisLabel: { color: theme.text || "#334155", fontSize: SURFACE_AXIS_LABEL_FONT_SIZE, formatter: options.formatter || compactNumber },
    axisLine: { lineStyle: { color: theme.line || "#cbd5e1" } },
    splitLine: { lineStyle: { color: theme.grid || "#e5e7eb" } },
  };
  const bounds = niceAxisBounds(domain, options.targetIntervals);
  if (bounds) {
    axis.min = bounds.min;
    axis.max = bounds.max;
    axis.interval = bounds.interval;
  }
  return axis;
}

function flameRibbonSeries(rows, lowKey, highKey, label, color) {
  const segments = ribbonSegments(rows, lowKey, highKey);
  if (!segments.length) return null;
  return {
    name: label,
    type: "custom",
    coordinateSystem: "cartesian2d",
    data: segments.map((_, index) => index),
    itemStyle: { color },
    silent: true,
    z: 1,
    renderItem: (params, api) => {
      const segment = segments[params.dataIndex] || [];
      const upper = segment.map((row) => api.coord([row.x, row.high]));
      const lower = [...segment].reverse().map((row) => api.coord([row.x, row.low]));
      return {
        type: "polygon",
        shape: { points: [...upper, ...lower] },
        style: { fill: color, stroke: "none" },
      };
    },
  };
}

function ribbonSegments(rows, lowKey, highKey) {
  const points = [];
  for (const row of rows || []) {
    const x = numberOrNull(row.x);
    const low = numberOrNull(row[lowKey]);
    const high = numberOrNull(row[highKey]);
    if (x === null || low === null || high === null) continue;
    points.push({ x, low, high });
  }
  points.sort((a, b) => a.x - b.x);
  return points.length > 1 ? [points] : [];
}

function centeredLegend(legend) {
  const centered = { ...(legend || {}), left: "center" };
  delete centered.right;
  return centered;
}

function flameTooltipFormatter(rows, payload) {
  return (params) => {
    const items = Array.isArray(params) ? params : [params];
    const axisValue = numberOrNull(items[0]?.axisValue ?? items[0]?.value?.[0]);
    const row = nearestRow(rows, axisValue);
    if (!row) return "";
    return [
      `${payload.x_feature}: ${formatTooltipNumber(row.x)}`,
      `Median: ${formatShapValue(row.p50, payload)}`,
      `40-60: ${formatTooltipRange(row.p40, row.p60, payload)}`,
      `5-95: ${formatTooltipRange(row.p5, row.p95, payload)}`,
      `Min-Max: ${formatTooltipRange(row.p0, row.p100, payload)}`,
    ].join("<br>");
  };
}

function boxTooltip(params, rows, feature = null, payload = null) {
  const row = rows?.[params?.dataIndex];
  if (!row) return "";
  const label = formatCategoryLabel(row.level, feature);
  if (params?.seriesName === "Mean") {
    return `${label}<br>Mean: ${formatShapValue(row.mean, payload)}`;
  }
  return [
    `${label}`,
    `min: ${formatShapValue(row.p0, payload)}`,
    `Q1: ${formatShapValue(row.p25, payload)}`,
    `median: ${formatShapValue(row.p50, payload)}`,
    `Q3: ${formatShapValue(row.p75, payload)}`,
    `max: ${formatShapValue(row.p100, payload)}`,
    `mean: ${formatShapValue(row.mean, payload)}`,
  ].join("<br>");
}

function heatmapTooltip(params, payload, xLabels, yLabels, xFeature = null, yFeature = null) {
  const value = Array.isArray(params?.value) ? params.value : [];
  const xLabel = formatCategoryLabel(xLabels[value[0]] ?? "", xFeature);
  const yLabel = formatCategoryLabel(yLabels[value[1]] ?? "", yFeature);
  return `${payload.x_feature}: ${xLabel}<br>${payload.y_feature}: ${yLabel}<br>SHAP: ${formatShapValue(value[2], payload)}`;
}

function surfaceDataShape(payload, rows) {
  const shape = payload?.grid?.data_shape;
  if (Array.isArray(shape) && shape.length === 2) {
    const yCount = Number(shape[0]);
    const xCount = Number(shape[1]);
    if (Number.isInteger(yCount) && Number.isInteger(xCount) && yCount > 0 && xCount > 0) return [yCount, xCount];
  }
  const xCount = unique((rows || []).map((row) => numberOrNull(row.x)).filter((value) => value !== null)).length;
  const yCount = unique((rows || []).map((row) => numberOrNull(row.y)).filter((value) => value !== null)).length;
  return [yCount, xCount];
}

function surfaceDataPoint(row) {
  const z = numberOrNull(row?.z);
  return [
    numberOrNull(row?.x),
    numberOrNull(row?.y),
    row?.has_data === false || z === null ? Number.NaN : z,
  ];
}

function nearestRow(rows, xValue) {
  if (!rows?.length) return null;
  if (xValue === null) return rows[0];
  let best = null;
  let bestDistance = Infinity;
  for (const row of rows) {
    const x = numberOrNull(row.x);
    if (x === null) continue;
    const distance = Math.abs(x - xValue);
    if (distance < bestDistance) {
      best = row;
      bestDistance = distance;
    }
  }
  return best;
}

function formatTooltipRange(low, high, payload = null) {
  return `${formatShapValue(low, payload)} to ${formatShapValue(high, payload)}`;
}

function referenceLineValue(payload) {
  return payload?.rescale?.mode === "1" ? 1 : 0;
}

function referenceMarkLine(theme, value) {
  return {
    symbol: "none",
    data: [{ yAxis: value }],
    lineStyle: { color: theme.zero || "#334155", width: 1, opacity: 0.75 },
    label: { show: false },
  };
}

function numberOrNull(value) {
  const number = Number(value);
  return Number.isFinite(number) ? number : null;
}

function numericExtent(values) {
  if (!values.length) return { min: 0, max: 1 };
  let min = Math.min(...values);
  let max = Math.max(...values);
  if (min === max) {
    const pad = Math.max(0.1, Math.abs(min) * 0.1);
    min -= pad;
    max += pad;
  }
  return { min, max };
}

function exactAxisBounds(domain, targetIntervals = AXIS_TARGET_INTERVALS) {
  if (!Array.isArray(domain) || domain.length < 2) return null;
  const min = numberOrNull(domain[0]);
  const max = numberOrNull(domain[1]);
  if (min === null || max === null) return null;
  if (min === max) return niceAxisBounds(domain, targetIntervals);
  const step = niceAxisStep(max - min, targetIntervals);
  return {
    min: roundAxisValue(min, step),
    max: roundAxisValue(max, step),
    interval: roundAxisValue(step, step),
  };
}

function niceAxisBounds(domain, targetIntervals = AXIS_TARGET_INTERVALS) {
  if (!Array.isArray(domain) || domain.length < 2) return null;
  const min = numberOrNull(domain[0]);
  const max = numberOrNull(domain[1]);
  if (min === null || max === null) return null;
  let lower = min;
  let upper = max;
  if (min === max) {
    const pad = Math.max(0.5, Math.abs(min) * 0.02);
    lower = min - pad;
    upper = max + pad;
  }
  const step = niceAxisStep(upper - lower, targetIntervals);
  let axisMin = Math.floor(lower / step) * step;
  let axisMax = Math.ceil(upper / step) * step;
  if (min >= 0) axisMin = Math.max(0, axisMin);
  if (axisMax <= axisMin) axisMax = axisMin + step;
  return {
    min: roundAxisValue(axisMin, step),
    max: roundAxisValue(axisMax, step),
    interval: roundAxisValue(step, step),
  };
}

function niceAxisStep(span, targetIntervals = AXIS_TARGET_INTERVALS) {
  if (!Number.isFinite(span) || span <= 0) return 1;
  const roughStep = span / Math.max(1, targetIntervals);
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

function numericCategoryTickPolicy(labels, feature = null) {
  if (!isNumericFeature(feature)) return null;
  const numericLabels = labels
    .map((label, index) => ({ index, value: numberOrNull(label) }))
    .filter((row) => row.value !== null)
    .sort((a, b) => a.value - b.value);
  if (!numericLabels.length) return null;
  const min = numericLabels[0].value;
  const max = numericLabels[numericLabels.length - 1].value;
  const step = niceAxisStep(max - min);
  const selected = new Set();
  const tolerance = Math.max(Math.abs(step) * 1e-7, 1e-12);
  let tick = Math.ceil(min / step) * step;
  const maxTick = max + tolerance;
  while (tick <= maxTick) {
    const roundedTick = roundAxisValue(tick, step);
    const match = numericLabels.find((row) => Math.abs(row.value - roundedTick) <= tolerance);
    if (match) selected.add(match.index);
    tick += step;
  }
  if (!selected.size) {
    selected.add(numericLabels[0].index);
    selected.add(numericLabels[numericLabels.length - 1].index);
  }
  return {
    interval: (index) => selected.has(index),
    step: roundAxisValue(step, step),
  };
}

function sortedCategoryLabels(rows, valueKey, sortKey, feature = null) {
  const categories = new Map();
  for (const row of rows || []) {
    const label = String(row?.[valueKey] ?? "");
    const rawSort = row?.[sortKey];
    const sortValue = isNumericFeature(feature) ? numberOrNull(rawSort) : rawSort;
    const current = categories.get(label);
    if (!current || compareCategorySort(sortValue, label, current.sortValue, current.label, feature) < 0) {
      categories.set(label, { label, sortValue });
    }
  }
  return [...categories.values()]
    .sort((a, b) => compareCategorySort(a.sortValue, a.label, b.sortValue, b.label, feature))
    .map((item) => item.label);
}

function compareCategorySort(leftSort, leftLabel, rightSort, rightLabel, feature = null) {
  const leftMissing = leftLabel === "(missing)";
  const rightMissing = rightLabel === "(missing)";
  if (leftMissing !== rightMissing) return leftMissing ? 1 : -1;
  if (isNumericFeature(feature)) {
    const leftNumber = numberOrNull(leftSort ?? leftLabel);
    const rightNumber = numberOrNull(rightSort ?? rightLabel);
    if (leftNumber !== null && rightNumber !== null && leftNumber !== rightNumber) return leftNumber - rightNumber;
    if (leftNumber !== null && rightNumber === null) return -1;
    if (leftNumber === null && rightNumber !== null) return 1;
  } else {
    const leftText = String(leftSort ?? leftLabel);
    const rightText = String(rightSort ?? rightLabel);
    const sortCompare = leftText.localeCompare(rightText, undefined, { numeric: true, sensitivity: "base" });
    if (sortCompare) return sortCompare;
  }
  return String(leftLabel).localeCompare(String(rightLabel), undefined, { numeric: true, sensitivity: "base" });
}

function unique(values) {
  const seen = new Set();
  const result = [];
  for (const value of values) {
    if (seen.has(value)) continue;
    seen.add(value);
    result.push(value);
  }
  return result;
}

function featureInfoForName(payload, name) {
  return [payload?.feature_1, payload?.feature_2].find((feature) => feature?.name === name) || null;
}

function formatCategoryLabel(value, feature = null) {
  if (!isNumericFeature(feature)) return String(value);
  const text = String(value);
  const number = Number(text);
  if (!Number.isFinite(number)) return text;
  return number.toLocaleString(undefined, { maximumFractionDigits: 12 });
}

function isNumericFeature(feature) {
  return feature?.kind === "numeric" || feature?.kind === "integer";
}

function compactNumber(value) {
  const number = Number(value);
  if (!Number.isFinite(number)) return "";
  const abs = Math.abs(number);
  if (abs >= 1000000) return `${Number((number / 1000000).toFixed(1))}m`;
  if (abs >= 1000) return `${Number((number / 1000).toFixed(1))}k`;
  if (abs > 0 && abs < 0.001) return number.toExponential(1);
  return Number(number.toPrecision(4)).toString();
}

function valueFormatterForPayload(payload) {
  return isUpliftRescale(payload) ? formatUpliftPercent : compactNumber;
}

function formatShapValue(value, payload = null) {
  return isUpliftRescale(payload) ? formatUpliftPercent(value) : formatTooltipNumber(value);
}

function formatVisualMapValue(value, payload = null) {
  return isUpliftRescale(payload) ? formatUpliftPercent(value) : formatVisualMapNumber(value);
}

function isUpliftRescale(payload) {
  return payload?.rescale?.mode === "1";
}

function formatUpliftPercent(value) {
  const number = Number(value);
  if (!Number.isFinite(number)) return "-";
  let percent = Number(((number - 1) * 100).toFixed(10));
  if (Math.abs(percent) < 1e-9) percent = 0;
  const abs = Math.abs(percent);
  let fractionDigits = 0;
  if (abs !== 0 && abs < 0.01) fractionDigits = 4;
  else if (abs !== 0 && abs < 1) fractionDigits = 2;
  else if (abs !== 0 && abs < 10) fractionDigits = 1;
  const formatted = percent.toLocaleString(undefined, {
    minimumFractionDigits: fractionDigits,
    maximumFractionDigits: fractionDigits,
  });
  const sign = percent > 0 ? "+" : "";
  return `${sign}${formatted}%`;
}

function formatTooltipNumber(value) {
  const number = Number(value);
  if (!Number.isFinite(number)) return "-";
  const rounded = Math.abs(number) < 0.00005 ? 0 : Number(number.toFixed(4));
  return rounded.toLocaleString(undefined, { maximumFractionDigits: 4 });
}

function formatVisualMapNumber(value) {
  const number = Number(value);
  if (!Number.isFinite(number)) return "-";
  const rounded = Math.abs(number) < 0.00005 ? 0 : number;
  return rounded.toLocaleString(undefined, { minimumFractionDigits: 4, maximumFractionDigits: 4 });
}

function surfaceTooltip(params, payload) {
  const value = Array.isArray(params?.value) ? params.value : [];
  const z = numberOrNull(value[2]);
  return `${payload.x_feature}: ${formatTooltipNumber(value[0])}<br>${payload.y_feature}: ${formatTooltipNumber(value[1])}<br>SHAP: ${z === null ? "No data" : formatShapValue(z, payload)}`;
}
