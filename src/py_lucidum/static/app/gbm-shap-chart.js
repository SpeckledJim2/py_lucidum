const SHAP_RED = "#d13f3f";
const SHAP_RED_RIBBONS = [
  "rgba(209, 63, 63, 0.08)",
  "rgba(209, 63, 63, 0.12)",
  "rgba(209, 63, 63, 0.16)",
  "rgba(209, 63, 63, 0.20)",
  "rgba(209, 63, 63, 0.25)",
  "rgba(209, 63, 63, 0.31)",
  "rgba(209, 63, 63, 0.38)",
];
const LINE_COLORS = ["#4fb99f", "#ff7f50", "#8aa1d6", "#b779d6", "#e7b84b", "#5aa2d6", "#d96a8a", "#84b547"];
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
    ["p45", "p55", "45-55"],
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
    markLine: zeroMarkLine(theme),
  });
  return {
    ...common,
    legend: centeredLegend(common.legend),
    tooltip: { trigger: "axis", confine: true, formatter: flameTooltipFormatter(rows, payload) },
    xAxis: valueAxis(payload.x_feature, theme, payload.x_domain),
    yAxis: valueAxis(payload.y_label || "SHAP", theme, payload.y_domain),
    series,
  };
}

function boxOption(payload, common, theme) {
  const rows = payload.rows || [];
  const labels = rows.map((row) => String(row.level));
  return {
    ...common,
    tooltip: { trigger: "item", confine: true, formatter: (params) => boxTooltip(params, rows) },
    xAxis: categoryAxis(labels, payload.x_feature, theme),
    yAxis: valueAxis(payload.y_label || "SHAP", theme),
    dataZoom: labels.length > 60 ? [{ type: "inside" }, { type: "slider", height: 18, bottom: 18 }] : [],
    series: [
      {
        name: "SHAP",
        type: "boxplot",
        data: rows.map((row) => [row.p0, row.p25, row.p50, row.p75, row.p100]),
        itemStyle: { borderColor: SHAP_RED, color: "rgba(209, 63, 63, 0.16)" },
        markLine: zeroMarkLine(theme),
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
      right: 10,
      top: 80,
      inRange: { color: ["#1d4ed8", "#f8fafc", "#b91c1c"] },
      textStyle: { color: theme.text || "#334155" },
    },
    grid3D: {
      top: 52,
      left: 0,
      right: 0,
      bottom: 0,
      boxWidth: 120,
      boxDepth: 90,
      viewControl: { projection: "perspective", autoRotate: false },
      axisPointer: { show: true },
      splitLine: { lineStyle: { color: theme.line || "#e2e8f0" } },
    },
    xAxis3D: axis3D(payload.x_feature, theme, payload.x_domain),
    yAxis3D: axis3D(payload.y_feature, theme, payload.y_domain),
    zAxis3D: axis3D(payload.z_label || "SHAP", theme),
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
    tooltip: { trigger: "axis", confine: true, valueFormatter: formatTooltipNumber },
    xAxis: valueAxis(payload.x_feature, theme, payload.x_domain),
    yAxis: valueAxis(payload.y_label || "SHAP", theme, payload.y_domain),
    series: [...seriesRows.entries()].map(([name, data]) => ({
      name,
      type: "line",
      data,
      symbol: "none",
      lineStyle: { width: 2 },
      markLine: name === [...seriesRows.keys()][0] ? zeroMarkLine(theme) : undefined,
    })),
  };
}

function heatmapOption(payload, common, theme) {
  const rows = payload.rows || [];
  const xLabels = unique(rows.map((row) => String(row.x)));
  const yLabels = unique(rows.map((row) => String(row.y)));
  const values = rows.map((row) => numberOrNull(row.z)).filter((value) => value !== null);
  const extent = numericExtent(values);
  const xIndex = new Map(xLabels.map((label, index) => [label, index]));
  const yIndex = new Map(yLabels.map((label, index) => [label, index]));
  return {
    ...common,
    tooltip: {
      trigger: "item",
      confine: true,
      formatter: (params) => heatmapTooltip(params, payload, xLabels, yLabels),
    },
    xAxis: categoryAxis(xLabels, payload.x_feature, theme),
    yAxis: categoryAxis(yLabels, payload.y_feature, theme),
    visualMap: {
      min: extent.min,
      max: extent.max,
      calculable: true,
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

function valueAxis(name, theme, domain = null) {
  const axis = {
    type: "value",
    name,
    scale: true,
    nameLocation: "middle",
    nameGap: 34,
    axisLine: { lineStyle: { color: theme.line || "#cbd5e1" } },
    axisLabel: { color: theme.text || "#334155", formatter: compactNumber },
    splitLine: { lineStyle: { color: theme.grid || "#e5e7eb" } },
    nameTextStyle: { color: theme.text || "#334155", fontWeight: 700 },
  };
  const bounds = normaliseDomain(domain);
  if (bounds) {
    axis.min = bounds[0];
    axis.max = bounds[1];
  }
  return axis;
}

function categoryAxis(labels, name, theme) {
  return {
    type: "category",
    data: labels,
    name,
    nameLocation: "middle",
    nameGap: 34,
    axisLine: { lineStyle: { color: theme.line || "#cbd5e1" } },
    axisLabel: { color: theme.text || "#334155", interval: labels.length > 80 ? "auto" : 0, rotate: labels.length > 25 ? 60 : 0 },
    splitLine: { show: true, lineStyle: { color: theme.grid || "#e5e7eb" } },
    nameTextStyle: { color: theme.text || "#334155", fontWeight: 700 },
  };
}

function axis3D(name, theme, domain = null) {
  const axis = {
    type: "value",
    name,
    nameTextStyle: { color: theme.text || "#334155", fontWeight: 700 },
    axisLabel: { color: theme.text || "#334155", formatter: compactNumber },
    axisLine: { lineStyle: { color: theme.line || "#cbd5e1" } },
    splitLine: { lineStyle: { color: theme.grid || "#e5e7eb" } },
  };
  const bounds = normaliseDomain(domain);
  if (bounds) {
    axis.min = bounds[0];
    axis.max = bounds[1];
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
      `Median: ${formatTooltipNumber(row.p50)}`,
      `45-55: ${formatTooltipRange(row.p45, row.p55)}`,
      `40-60: ${formatTooltipRange(row.p40, row.p60)}`,
      `5-95: ${formatTooltipRange(row.p5, row.p95)}`,
      `Min-Max: ${formatTooltipRange(row.p0, row.p100)}`,
    ].join("<br>");
  };
}

function boxTooltip(params, rows) {
  const row = rows?.[params?.dataIndex];
  if (!row) return "";
  if (params?.seriesName === "Mean") {
    return `${row.level}<br>Mean: ${formatTooltipNumber(row.mean)}`;
  }
  return [
    `${row.level}`,
    `min: ${formatTooltipNumber(row.p0)}`,
    `Q1: ${formatTooltipNumber(row.p25)}`,
    `median: ${formatTooltipNumber(row.p50)}`,
    `Q3: ${formatTooltipNumber(row.p75)}`,
    `max: ${formatTooltipNumber(row.p100)}`,
    `mean: ${formatTooltipNumber(row.mean)}`,
  ].join("<br>");
}

function heatmapTooltip(params, payload, xLabels, yLabels) {
  const value = Array.isArray(params?.value) ? params.value : [];
  const xLabel = xLabels[value[0]] ?? "";
  const yLabel = yLabels[value[1]] ?? "";
  return `${payload.x_feature}: ${xLabel}<br>${payload.y_feature}: ${yLabel}<br>SHAP: ${formatTooltipNumber(value[2])}`;
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

function formatTooltipRange(low, high) {
  return `${formatTooltipNumber(low)} to ${formatTooltipNumber(high)}`;
}

function zeroMarkLine(theme) {
  return {
    symbol: "none",
    data: [{ yAxis: 0 }],
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

function normaliseDomain(domain) {
  if (!Array.isArray(domain) || domain.length < 2) return null;
  const min = numberOrNull(domain[0]);
  const max = numberOrNull(domain[1]);
  if (min === null || max === null) return null;
  if (min === max) {
    const pad = Math.max(0.5, Math.abs(min) * 0.02);
    return [min - pad, max + pad];
  }
  return [min, max];
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

function compactNumber(value) {
  const number = Number(value);
  if (!Number.isFinite(number)) return "";
  const abs = Math.abs(number);
  if (abs >= 1000000) return `${Number((number / 1000000).toFixed(1))}m`;
  if (abs >= 1000) return `${Number((number / 1000).toFixed(1))}k`;
  if (abs > 0 && abs < 0.001) return number.toExponential(1);
  return Number(number.toPrecision(4)).toString();
}

function formatTooltipNumber(value) {
  const number = Number(value);
  if (!Number.isFinite(number)) return "-";
  const rounded = Math.abs(number) < 0.00005 ? 0 : Number(number.toFixed(4));
  return rounded.toLocaleString(undefined, { maximumFractionDigits: 4 });
}

function surfaceTooltip(params, payload) {
  const value = Array.isArray(params?.value) ? params.value : [];
  const z = numberOrNull(value[2]);
  return `${payload.x_feature}: ${formatTooltipNumber(value[0])}<br>${payload.y_feature}: ${formatTooltipNumber(value[1])}<br>SHAP: ${z === null ? "No data" : formatTooltipNumber(z)}`;
}
