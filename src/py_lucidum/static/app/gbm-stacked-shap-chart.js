const SUM_SERIES = "Sum of SHAP values";
const SUM_RED = "#ff0000";
const OTHER_COLOR = "#9ca3af";
const STACK_COLORS = [
  "#e5f5a9",
  "#80b1d3",
  "#d9a0ae",
  "#d8d2d3",
  "#8dd3c7",
  "#d9d2b6",
  "#d6c84f",
  "#ed8077",
  "#e7bf79",
  "#ffe866",
  "#b8d57a",
  "#c6a5c8",
  "#b3de69",
  "#fccde5",
  "#bc80bd",
];

export function stackedShapChartOption(payload, theme = {}) {
  if (!payload || typeof payload !== "object") return emptyOption("Select a model feature", theme);
  const rows = Array.isArray(payload.rows) ? payload.rows : [];
  const featureNames = Array.isArray(payload.display_features) ? payload.display_features : [];
  if (!rows.length || !featureNames.length) return emptyOption("No Stacked SHAP data", theme);
  const labels = rows.map((row) => String(row.x ?? ""));
  const labelPolicy = xAxisLabelPolicy(labels);
  const series = featureNames.map((name, index) => ({
    name,
    type: "bar",
    stack: "shap",
    barMaxWidth: 34,
    data: rows.map((row) => numberOrNull(row?.contributions?.[name]) ?? 0),
    itemStyle: { color: featureColor(name, index) },
    emphasis: { focus: "series" },
  }));
  if (series.length) {
    series[0].markLine = zeroMarkLine(theme);
  }
  series.push({
    name: SUM_SERIES,
    type: "scatter",
    data: rows.map((row) => numberOrNull(row.total_shap)),
    symbolSize: 8,
    itemStyle: { color: SUM_RED },
    z: 8,
  });
  return {
    backgroundColor: theme.panel || "transparent",
    animation: false,
    color: [SUM_RED, ...featureNames.map(featureColor)],
    title: {
      text: payload.title || "Stacked SHAP",
      left: "center",
      top: 10,
      textStyle: { color: theme.text || "#334155", fontSize: 14, fontWeight: 800 },
    },
    tooltip: {
      trigger: "axis",
      axisPointer: { type: "shadow" },
      confine: true,
      formatter: (params) => tooltipHtml(params, rows, payload),
    },
    legend: {
      data: [SUM_SERIES, ...featureNames],
      type: "scroll",
      orient: "vertical",
      right: 10,
      top: 58,
      bottom: 18,
      textStyle: { color: theme.text || "#334155", fontSize: 11 },
    },
    grid: {
      top: 70,
      right: 220,
      bottom: labelPolicy.rotate ? 66 : 34,
      left: 72,
      containLabel: true,
    },
    xAxis: {
      type: "category",
      name: payload.model_feature?.name || payload.model_feature || "",
      nameLocation: "middle",
      nameGap: labelPolicy.rotate ? 50 : 24,
      data: labels,
      axisLine: { lineStyle: { color: theme.line || "#cbd5e1" } },
      axisLabel: {
        color: theme.text || "#334155",
        fontSize: labelPolicy.fontSize,
        hideOverlap: true,
        interval: labelPolicy.interval,
        rotate: labelPolicy.rotate,
      },
      splitLine: { show: false },
      nameTextStyle: { color: theme.text || "#334155", fontWeight: 700 },
    },
    yAxis: valueAxis(payload.y_domain, theme),
    series,
  };
}

function xAxisLabelPolicy(labels) {
  const maxLength = labels.reduce((max, label) => Math.max(max, String(label || "").length), 0);
  const longLabels = maxLength > 12;
  return {
    fontSize: 10,
    interval: labels.length > 90 ? "auto" : 0,
    rotate: longLabels ? 45 : 0,
  };
}

function emptyOption(message, theme = {}) {
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

function valueAxis(domain, theme) {
  const axis = {
    type: "value",
    name: "SHAP Contribution (Linear Predictor Scale)",
    nameLocation: "middle",
    nameGap: 52,
    scale: true,
    axisLine: { lineStyle: { color: theme.line || "#cbd5e1" } },
    axisLabel: { color: theme.text || "#334155", formatter: compactNumber },
    splitLine: { lineStyle: { color: theme.grid || "#e5e7eb" } },
    nameTextStyle: { color: theme.text || "#334155", fontWeight: 700 },
  };
  const bounds = niceAxisBounds(domain);
  if (bounds) {
    axis.min = bounds.min;
    axis.max = bounds.max;
    axis.interval = bounds.interval;
  }
  return axis;
}

function tooltipHtml(params, rows, payload) {
  const items = Array.isArray(params) ? params : [params];
  const dataIndex = Number(items[0]?.dataIndex ?? 0);
  const row = rows[dataIndex];
  if (!row) return "";
  const featureName = payload.model_feature?.name || payload.model_feature || "Feature";
  const lines = [
    `${escapeHtml(featureName)}: ${escapeHtml(row.x ?? "")}`,
    `Rows: ${Number(row.row_count || 0).toLocaleString()}`,
    `${SUM_SERIES}: ${formatTooltipNumber(row.total_shap)}`,
  ];
  const byName = new Map(items.map((item) => [String(item.seriesName || ""), item]));
  for (const name of payload.display_features || []) {
    const value = numberOrNull(row?.contributions?.[name]);
    if (value === null) continue;
    const marker = byName.get(name)?.marker || "";
    lines.push(`${marker}${escapeHtml(name)}: ${formatTooltipNumber(value)}`);
  }
  return lines.join("<br>");
}

function featureColor(name, index = 0) {
  return name === "Other" ? OTHER_COLOR : STACK_COLORS[index % STACK_COLORS.length];
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

function niceAxisBounds(domain) {
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
  const step = niceAxisStep(upper - lower);
  let axisMin = Math.floor(lower / step) * step;
  let axisMax = Math.ceil(upper / step) * step;
  if (axisMax <= axisMin) axisMax = axisMin + step;
  return {
    min: roundAxisValue(axisMin, step),
    max: roundAxisValue(axisMax, step),
    interval: roundAxisValue(step, step),
  };
}

function niceAxisStep(span, targetIntervals = 6) {
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

function escapeHtml(value) {
  return String(value ?? "")
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#39;");
}
