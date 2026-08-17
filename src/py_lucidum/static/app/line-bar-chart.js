const LABEL_DENSITY_LIMIT = 200;
const RESPONSE_AXIS_PADDING = 0.08;
const RESPONSE_AXIS_TARGET_INTERVALS = 15;
const LINE_BAR_MAIN_LEGEND_TOP = 52;
const LINE_BAR_OVERLAY_LEGEND_TOP = 78;
const LINE_BAR_GRID_TOP = 112;
const LINE_BAR_OVERLAY_GRID_TOP = 140;
const SHAP_RIBBON_SERIES = [
  ["p0", "p100", "SHAP Min-Max", "rgba(209, 63, 63, 0.10)"],
  ["p5", "p95", "SHAP 5-95", "rgba(209, 63, 63, 0.16)"],
  ["p10", "p90", "SHAP 10-90", "rgba(209, 63, 63, 0.20)"],
  ["p20", "p80", "SHAP 20-80", "rgba(209, 63, 63, 0.24)"],
  ["p30", "p70", "SHAP 30-70", "rgba(209, 63, 63, 0.28)"],
  ["p40", "p60", "SHAP 40-60", "rgba(209, 63, 63, 0.34)"],
];
const DEFAULT_THEME = {
  text: "#1f2937",
  muted: "#667085",
  line: "#d7dde7",
  actual: "#222222",
  expected: "#d13f3f",
  secondExpected: "#2276d2",
  bar: "#bfeefa",
  missingBar: "#d9dee7",
  baseBar: "#4fb6d2",
  tailBar: "#72c7df",
  shap: "#d13f3f",
  glm: "#1f7a8c",
  sigma: "#8a94a6",
};

export function lineBarChartOption(data, options = {}) {
  const rows = Array.isArray(data?.rows) ? data.rows : [];
  const responses = Array.isArray(data?.responses) ? data.responses : [];
  const presentation = { ...(options.presentation || {}), ...options };
  const content = presentation.content === "shap_only" ? "shap_only" : "actual_expected";
  const theme = { ...DEFAULT_THEME, ...(presentation.themeColors || {}) };
  const transform = String(data?.transform?.mode || presentation.transform || "none");
  const formatNumber = presentation.formatNumber || defaultNumberFormatter;
  const formatResponse = presentation.formatResponse || responseFormatter(transform, presentation.kpiFormat);
  const escapeHtml = presentation.escapeHtml || escapeHtmlText;
  const labels = presentation.labelsArray || rows.map((row) => formatXValue(row, data, formatNumber));
  const labelMode = String(presentation.labels || "none");
  const xLabelPolicy = presentation.xLabelPolicy || defaultXAxisLabelPolicy(
    labels,
    data?.x_group_kind || data?.x_kind,
    Number(presentation.chartWidth) || 900,
  );
  const labelsAllowed = labels.length < LABEL_DENSITY_LIMIT;
  const showBarLabels = labelsAllowed && ["bar", "all"].includes(labelMode);
  const showLineLabels = labelsAllowed && ["line", "all"].includes(labelMode);
  const shapSeries = shapPartialDependenceSeries(data, theme);
  const glmSeries = glmPartialDependenceSeries(data, theme);
  const overlaySeries = [...shapSeries, ...glmSeries];
  const overlayLegendData = overlaySeries.map((series) => series.name);
  const showMainSeries = content !== "shap_only";
  const weightLabel = data?.denominator?.bar_label || "Weight";
  const mainLegendData = showMainSeries
    ? [
        ...responses.map((response) => response.label),
        { name: weightLabel, icon: "roundRect", itemStyle: { color: theme.bar, borderColor: theme.bar } },
      ]
    : [];
  const previousOption = presentation.previousOption || {};
  const mainLegendSelection = matchingLegendSelection(previousOption, mainLegendData);
  const overlayLegendSelection = matchingLegendSelection(previousOption, overlayLegendData);
  const selected = { ...mainLegendSelection, ...overlayLegendSelection };
  const responseAxis = responseAxisOptions(data, selected, transform, content);
  const measureText = presentation.measureText;
  const responseAxisLayout = verticalAxisLayout(
    [responseAxis.min, responseAxis.max],
    formatResponse,
    measureText,
  );
  const volumeAxisLayout = verticalAxisLayout(
    rows.map((row) => row.volume),
    formatNumber,
    measureText,
  );
  const barLayout = barLayoutForCount(labels.length);
  const labelStyle = presentation.labelStyle || {};
  const responseColors = [theme.actual, theme.expected, theme.secondExpected];
  const lineSeries = showMainSeries
    ? responses.map((response, index) => ({
        name: response.label,
        type: "line",
        yAxisIndex: 0,
        z: 3,
        animation: false,
        animationDuration: 0,
        animationDurationUpdate: 0,
        smooth: false,
        showSymbol: rows.length < 250,
        symbolSize: 5,
        lineStyle: { color: responseColors[index] || theme.actual },
        itemStyle: { color: responseColors[index] || theme.actual },
        data: rows.map((row) => row[`resp${index}`]),
        showAllSymbol: true,
        label: {
          show: showLineLabels,
          fontSize: 10,
          formatter: (params) => formatResponse(seriesValue(params)),
          ...labelStyle,
        },
      }))
    : [];
  const barSeries = showMainSeries
    ? [{
        name: weightLabel,
        type: "bar",
        yAxisIndex: 1,
        z: 1,
        legendHoverLink: true,
        itemStyle: { color: theme.bar },
        animation: false,
        animationDuration: 0,
        animationDurationUpdate: 0,
        data: rows.map((row) => ({ value: row.volume, itemStyle: { color: weightBarColor(data, row, theme) } })),
        label: {
          show: showBarLabels,
          position: "top",
          fontSize: 10,
          formatter: presentation.formatChartLabel || ((params) => formatNumber(seriesValue(params))),
          ...labelStyle,
        },
        barWidth: barLayout.width,
        barMaxWidth: barLayout.maxWidth,
        barCategoryGap: barLayout.categoryGap,
      }]
    : [];
  const sigmaSeries = showMainSeries
    ? buildSigmaSeries(data, Number(presentation.sigma) || 0, theme)
    : [];
  const baseline = transform === "one" ? upliftBaselineSeries(rows, theme) : null;
  const primaryResponseLabel = String(responses[0]?.label || "");
  const displayLabels = primaryResponseLabel
    ? { [primaryResponseLabel]: responseMetricAxisLabel(data, 0) }
    : {};
  const hasOverlaySeries = overlaySeries.length > 0;
  const legend = legendOptions(
    mainLegendData,
    mainLegendSelection,
    overlayLegendData,
    overlayLegendSelection,
    displayLabels,
    theme,
    content,
  );
  const yAxisName = content === "shap_only"
    ? (transform === "one" ? "SHAP relativity" : transform === "zero" ? "SHAP difference" : "SHAP")
    : responseMetricAxisLabel(data, 0);
  const yAxes = [{
    type: "value",
    name: yAxisName,
    nameLocation: "middle",
    nameGap: responseAxisLayout.nameGap,
    nameTextStyle: { color: theme.text, fontWeight: 700 },
    scale: true,
    splitNumber: RESPONSE_AXIS_TARGET_INTERVALS,
    min: responseAxis.min,
    max: responseAxis.max,
    interval: responseAxis.interval,
    axisLabel: { color: theme.text, formatter: (value) => formatResponse(value) },
    splitLine: { lineStyle: { color: theme.line } },
  }];
  if (showMainSeries) {
    yAxes.push({
      type: "value",
      name: weightLabel,
      nameLocation: "middle",
      nameGap: volumeAxisLayout.nameGap,
      nameTextStyle: { color: theme.text, fontWeight: 700 },
      position: "right",
      axisLabel: { color: theme.text, formatter: (value) => formatNumber(value) },
      splitLine: { show: false },
    });
  }
  const option = {
    animation: false,
    animationDuration: 0,
    animationDurationUpdate: 0,
    stateAnimation: { duration: 0 },
    backgroundColor: "transparent",
    color: [theme.actual, theme.expected, theme.secondExpected, theme.bar],
    tooltip: {
      trigger: "axis",
      formatter: (params) => chartTooltip(params, weightLabel, formatResponse, formatNumber, escapeHtml),
    },
    legend,
    grid: {
      left: responseAxisLayout.gridMargin,
      right: showMainSeries ? volumeAxisLayout.gridMargin : 36,
      top: hasOverlaySeries ? LINE_BAR_OVERLAY_GRID_TOP : LINE_BAR_GRID_TOP,
      bottom: xLabelPolicy.bottom,
      containLabel: false,
    },
    xAxis: {
      type: "category",
      name: data?.x || "",
      nameLocation: "middle",
      nameGap: xLabelPolicy.nameGap,
      nameTextStyle: { color: theme.text, fontSize: 13, fontWeight: 700 },
      data: labels,
      axisLabel: {
        show: xLabelPolicy.show,
        color: theme.text,
        interval: xLabelPolicy.interval,
        formatter: xLabelPolicy.formatter,
        hideOverlap: Boolean(xLabelPolicy.hideOverlap),
        showMinLabel: xLabelPolicy.showMinLabel,
        showMaxLabel: xLabelPolicy.showMaxLabel,
        rotate: xLabelPolicy.rotate,
        fontSize: xLabelPolicy.fontSize,
        margin: 8,
      },
      axisLine: { lineStyle: { color: theme.line } },
    },
    yAxis: yAxes,
    dataZoom: xLabelPolicy.dataZoomEnabled ? lineBarDataZoomOptions() : [],
    series: [
      ...barSeries,
      ...overlaySeries,
      ...lineSeries,
      ...(baseline ? [baseline] : []),
      ...sigmaSeries,
    ],
  };
  const messages = [];
  if (!xLabelPolicy.show) messages.push(xLabelPolicy.hiddenReason || "X-axis labels hidden to avoid overlap.");
  if (!labelsAllowed && labelMode !== "none") messages.push(`Chart labels hidden as >${LABEL_DENSITY_LIMIT.toLocaleString()} categories.`);
  return { option, messages, xLabelPolicy };
}

export function bindLineBarChartInteractions(chart, data, presentation = {}) {
  if (!chart?.on) return;
  chart.on("legendselectchanged", () => {
    const option = chart.getOption?.() || {};
    const selected = legendSelectionFromOption(option);
    const transform = String(data?.transform?.mode || presentation.transform || "none");
    const content = presentation.content === "shap_only" ? "shap_only" : "actual_expected";
    const responseAxis = responseAxisOptions(data, selected, transform, content);
    chart.setOption({ yAxis: [{ min: responseAxis.min, max: responseAxis.max, interval: responseAxis.interval }] });
  });
}

function partialDependenceOverlay(data, key) {
  const partial = data?.partial_dependence || {};
  if (partial.overlays && partial.overlays[key]) return partial.overlays[key];
  return partial.mode === key ? partial : {};
}

function indexedPartialDependenceRows(data, rows) {
  const index = new Map((data?.rows || []).map((row, position) => [String(row.x), position]));
  return rows
    .map((row) => ({ ...row, index: index.get(String(row.x)) }))
    .filter((row) => Number.isInteger(row.index));
}

function shapPartialDependenceSeries(data, theme) {
  const partial = partialDependenceOverlay(data, "shap");
  const rows = Array.isArray(partial?.rows) ? partial.rows : [];
  const indexedRows = indexedPartialDependenceRows(data, rows);
  if (!indexedRows.length) return [];
  const series = SHAP_RIBBON_SERIES
    .map(([lowKey, highKey, label, color]) => shapRibbonSeries(indexedRows, lowKey, highKey, label, color))
    .filter(Boolean);
  series.push({
    name: "SHAP median",
    type: "line",
    yAxisIndex: 0,
    z: 2.8,
    animation: false,
    animationDuration: 0,
    animationDurationUpdate: 0,
    smooth: false,
    showSymbol: (data?.rows || []).length < 250,
    symbolSize: 4,
    lineStyle: { color: theme.shap, width: 1.8, type: "dashed" },
    itemStyle: { color: theme.shap },
    data: (data?.rows || []).map((row) => {
      const match = rows.find((partialRow) => String(partialRow.x) === String(row.x));
      return finiteNumberOrNull(match?.p50);
    }),
    label: { show: false },
  });
  return series;
}

function glmPartialDependenceSeries(data, theme) {
  const partial = partialDependenceOverlay(data, "glm");
  const rows = Array.isArray(partial?.rows) ? partial.rows : [];
  if (!indexedPartialDependenceRows(data, rows).length) return [];
  return [{
    name: "GLM",
    type: "line",
    yAxisIndex: 0,
    z: 2.9,
    animation: false,
    animationDuration: 0,
    animationDurationUpdate: 0,
    smooth: false,
    showSymbol: (data?.rows || []).length < 250,
    symbolSize: 4,
    lineStyle: { color: theme.glm, width: 2, type: "dashed" },
    itemStyle: { color: theme.glm },
    data: (data?.rows || []).map((row) => {
      const match = rows.find((partialRow) => String(partialRow.x) === String(row.x));
      return finiteNumberOrNull(match?.p50);
    }),
    label: { show: false },
  }];
}

function shapRibbonSeries(rows, lowKey, highKey, label, color) {
  const segments = shapRibbonSegments(rows, lowKey, highKey);
  if (!segments.length) return null;
  return {
    name: label,
    type: "custom",
    coordinateSystem: "cartesian2d",
    yAxisIndex: 0,
    data: segments.map((_, index) => index),
    itemStyle: { color },
    silent: true,
    z: 2,
    animation: false,
    animationDuration: 0,
    animationDurationUpdate: 0,
    renderItem: (params, api) => {
      const segment = segments[params.dataIndex] || [];
      const upper = segment.map((row) => api.coord([row.index, row.high]));
      const lower = [...segment].reverse().map((row) => api.coord([row.index, row.low]));
      return { type: "polygon", shape: { points: [...upper, ...lower] }, style: { fill: color, stroke: "none" } };
    },
  };
}

function shapRibbonSegments(rows, lowKey, highKey) {
  const points = rows
    .map((row) => ({ index: Number(row.index), low: Number(row[lowKey]), high: Number(row[highKey]) }))
    .filter((row) => Number.isInteger(row.index) && Number.isFinite(row.low) && Number.isFinite(row.high))
    .sort((left, right) => left.index - right.index);
  const segments = [];
  let current = [];
  points.forEach((point) => {
    const previous = current[current.length - 1];
    if (previous && point.index !== previous.index + 1) {
      if (current.length > 1) segments.push(current);
      current = [];
    }
    current.push(point);
  });
  if (current.length > 1) segments.push(current);
  return segments;
}

function buildSigmaSeries(data, sigma, theme) {
  if (!(sigma > 0) || (data?.responses || []).length < 2) return [];
  return [{
    name: "sigma",
    type: "custom",
    yAxisIndex: 0,
    z: 5,
    legendHoverLink: false,
    animation: false,
    animationDuration: 0,
    animationDurationUpdate: 0,
    renderItem: (params, api) => {
      const x = api.coord([api.value(0), api.value(1)])[0];
      const low = api.coord([api.value(0), api.value(2)])[1];
      const high = api.coord([api.value(0), api.value(3)])[1];
      if (!Number.isFinite(low) || !Number.isFinite(high)) return undefined;
      return {
        type: "group",
        children: [
          { type: "line", shape: { x1: x, y1: low, x2: x, y2: high }, style: { stroke: theme.sigma, lineWidth: 1.5 } },
          { type: "line", shape: { x1: x - 4, y1: low, x2: x + 4, y2: low }, style: { stroke: theme.sigma, lineWidth: 1.5 } },
          { type: "line", shape: { x1: x - 4, y1: high, x2: x + 4, y2: high }, style: { stroke: theme.sigma, lineWidth: 1.5 } },
        ],
      };
    },
    data: (data?.rows || [])
      .map((row, index) => [index, row.resp1, row.resp1_low, row.resp1_high])
      .filter((row) => row.every((value) => value !== null && value !== undefined)),
    encode: { x: 0, y: [2, 3] },
    tooltip: { show: false },
  }];
}

function legendOptions(mainData, mainSelected, overlayData, overlaySelected, displayLabels, theme, content) {
  const mainLegend = {
    top: LINE_BAR_MAIN_LEGEND_TOP,
    data: mainData,
    selected: mainSelected,
    formatter: (name) => displayLabels[name] || name,
    textStyle: { color: theme.text, fontWeight: 700, fontSize: 13 },
  };
  const overlayLegend = {
    top: content === "shap_only" ? LINE_BAR_MAIN_LEGEND_TOP : LINE_BAR_OVERLAY_LEGEND_TOP,
    left: "center",
    type: "scroll",
    data: overlayData,
    selected: overlaySelected,
    textStyle: { color: theme.text, fontWeight: 400, fontSize: 11 },
    pageIconColor: theme.text,
    pageIconInactiveColor: theme.muted,
    pageTextStyle: { color: theme.muted },
  };
  if (!mainData.length) return overlayLegend;
  return overlayData.length ? [mainLegend, overlayLegend] : mainLegend;
}

function matchingLegendSelection(option, entries) {
  const names = entries.map(legendEntryName).filter(Boolean);
  const selected = Object.fromEntries(names.map((name) => [name, true]));
  const legends = Array.isArray(option?.legend) ? option.legend : option?.legend ? [option.legend] : [];
  const previous = Object.assign({}, ...legends.map((legend) => legend?.selected || {}));
  names.forEach((name) => {
    if (Object.prototype.hasOwnProperty.call(previous, name)) selected[name] = previous[name] !== false;
  });
  return selected;
}

function legendEntryName(entry) {
  if (typeof entry === "string") return entry;
  return entry && typeof entry === "object" ? String(entry.name || "") : "";
}

function legendSelectionFromOption(option) {
  const legends = Array.isArray(option?.legend) ? option.legend : option?.legend ? [option.legend] : [];
  return Object.assign({}, ...legends.map((legend) => legend?.selected || {}));
}

function responseAxisOptions(data, selected, transform, content) {
  const extent = responseAxisExtent(data, selected, content) || responseAxisExtent(data, null, content);
  if (!extent) return {};
  if (transform === "one") {
    extent.min = Math.min(extent.min, 1);
    extent.max = Math.max(extent.max, 1);
  }
  return responseAxisBounds(extent) || {};
}

function responseAxisExtent(data, selected, content) {
  let min = Infinity;
  let max = -Infinity;
  const visible = (name) => !selected || selected[String(name)] !== false;
  const add = (raw) => {
    const value = finiteNumberOrNull(raw);
    if (value === null) return;
    min = Math.min(min, value);
    max = Math.max(max, value);
  };
  if (content !== "shap_only") {
    (data?.rows || []).forEach((row) => (data?.responses || []).forEach((response, index) => {
      if (visible(response?.label)) add(row[`resp${index}`]);
    }));
  }
  partialDependenceOverlayEntries(data?.partial_dependence).forEach(([key, overlay]) => {
    (overlay?.rows || []).forEach((row) => {
      if (key === "shap") {
        SHAP_RIBBON_SERIES.forEach(([low, high, label]) => {
          if (visible(label)) { add(row[low]); add(row[high]); }
        });
        if (visible("SHAP median")) add(row.p50);
      } else if (key === "glm" && content !== "shap_only" && visible("GLM")) {
        add(row.p50);
      }
    });
  });
  return Number.isFinite(min) && Number.isFinite(max) ? { min, max } : null;
}

function partialDependenceOverlayEntries(partial) {
  if (!partial) return [];
  if (partial.overlays) return Object.entries(partial.overlays).filter(([, overlay]) => overlay && typeof overlay === "object");
  return [[String(partial.mode || ""), partial]];
}

function responseAxisBounds(extent) {
  const min = Number(extent?.min);
  const max = Number(extent?.max);
  if (!Number.isFinite(min) || !Number.isFinite(max)) return null;
  const span = max > min ? max - min : Math.max(Math.abs(max), Math.abs(min), 1);
  const paddedMin = min - span * RESPONSE_AXIS_PADDING;
  const paddedMax = max + span * RESPONSE_AXIS_PADDING;
  const step = niceAxisStep(paddedMax - paddedMin);
  let axisMin = Math.floor(paddedMin / step) * step;
  let axisMax = Math.ceil(paddedMax / step) * step;
  if (min >= 0) axisMin = Math.max(0, axisMin);
  if (axisMax <= axisMin) axisMax = axisMin + step;
  return { min: roundAxisValue(axisMin, step), max: roundAxisValue(axisMax, step), interval: step };
}

function niceAxisStep(span) {
  const rough = span / RESPONSE_AXIS_TARGET_INTERVALS;
  const magnitude = 10 ** Math.floor(Math.log10(rough > 0 ? rough : 1));
  const normalized = rough / magnitude;
  return ([1, 2, 5, 10].find((candidate) => normalized <= candidate) || 10) * magnitude;
}

function roundAxisValue(value, step) {
  const precision = Math.min(12, Math.max(0, Math.ceil(-Math.log10(Math.abs(step))) + 3));
  return Number(value.toFixed(precision));
}

function verticalAxisLayout(values, formatter, measureText) {
  const labels = (values || []).map(finiteNumberOrNull).filter((value) => value !== null).map((value) => String(formatter(value)));
  if (!labels.includes("0")) labels.push("0");
  const width = labels.reduce((maximum, label) => {
    const measured = typeof measureText === "function" ? Number(measureText(label, 12)) : NaN;
    return Math.max(maximum, Number.isFinite(measured) ? measured : label.length * 6.72);
  }, 0);
  const nameGap = Math.max(52, Math.ceil(width + 18));
  return { nameGap, gridMargin: Math.max(76, nameGap + 28) };
}

function defaultXAxisLabelPolicy(labels, kind, chartWidth) {
  const count = labels.length;
  const tooMany = count >= LABEL_DENSITY_LIMIT;
  const zoom = count > 120;
  if (tooMany) return { show: false, interval: 0, rotate: 0, fontSize: 10, nameGap: 22, bottom: zoom ? 74 : 38, dataZoomEnabled: zoom, hideOverlap: false, hiddenReason: `X-axis labels hidden as >${LABEL_DENSITY_LIMIT.toLocaleString()} categories.` };
  const maxLength = labels.reduce((maximum, label) => Math.max(maximum, String(label).length), 0);
  const fontSize = count > 50 ? 8 : 10;
  const slot = Math.max(120, chartWidth - 148) / Math.max(1, count);
  const rotate = count > 30 || maxLength > 10 || maxLength * fontSize * 0.5 + 8 > slot ? 65 : 0;
  const labelSpace = rotate ? Math.min(140, Math.max(58, Math.ceil(maxLength * fontSize * 0.5 * Math.sin(65 * Math.PI / 180)) + 18)) : 38;
  const nameGap = rotate ? Math.max(26, labelSpace - 10) : 26;
  return { show: true, interval: 0, rotate, fontSize, nameGap, bottom: nameGap + 16 + (zoom ? 36 : 0), dataZoomEnabled: zoom, hideOverlap: kind === "quantile" && count > 80, showMinLabel: true, showMaxLabel: true, hiddenReason: "" };
}

function formatXValue(row, data, formatNumber) {
  if ((data?.x_group_kind || data?.x_kind) === "quantile" && row && !row.is_tail && row.x !== "Missing") {
    const start = row.x_start === null || row.x_start === undefined ? "" : formatNumber(row.x_start);
    const end = row.x_end === null || row.x_end === undefined ? "" : formatNumber(row.x_end);
    if (start && end) return start === end ? `${row.x}\n${start}` : `${row.x}\n${start} to ${end}`;
  }
  return String(row?.x ?? "");
}

function responseMetricAxisLabel(data, index) {
  const response = data?.responses?.[index];
  const numerator = String(response?.numerator || response?.label || "").trim();
  const denominator = String(data?.denominator?.column || "").trim();
  if (!numerator) return denominator ? `Actual / ${denominator}` : "Actual";
  return denominator ? `${numerator} / ${denominator}` : numerator;
}

function weightBarColor(data, row, theme) {
  const label = String(row?.x ?? "").trim().toLowerCase();
  if (["missing", "(missing)"].includes(label)) return theme.missingBar;
  if (row?.is_tail) return theme.tailBar;
  const transform = String(data?.transform?.mode || "none");
  const base = data?.transform?.base_x;
  if (["zero", "one"].includes(transform) && base !== null && base !== undefined && String(row?.x) === String(base)) return theme.baseBar;
  return theme.bar;
}

function upliftBaselineSeries(rows, theme) {
  return {
    name: "0% uplift baseline",
    type: "line",
    yAxisIndex: 0,
    z: 2.7,
    silent: true,
    legendHoverLink: false,
    animation: false,
    showSymbol: false,
    symbolSize: 0,
    lineStyle: { opacity: 0 },
    itemStyle: { opacity: 0 },
    tooltip: { show: false },
    data: rows.map(() => 1),
    markLine: { silent: true, symbol: "none", label: { show: false }, lineStyle: { color: theme.text, width: 2, type: "solid", opacity: 0.5 }, data: [{ yAxis: 1 }] },
  };
}

function barLayoutForCount(count) {
  if (count <= 3) return { width: "62%", maxWidth: 240, categoryGap: "18%" };
  if (count <= 8) return { width: "56%", maxWidth: 180, categoryGap: "24%" };
  if (count <= 20) return { width: "46%", maxWidth: 90, categoryGap: "34%" };
  if (count <= 60) return { width: "68%", maxWidth: 34, categoryGap: "28%" };
  return { width: null, maxWidth: 18, categoryGap: "30%" };
}

function lineBarDataZoomOptions() {
  return [{ type: "inside" }, { type: "slider", height: 18, bottom: 18 }];
}

function chartTooltip(params, weightLabel, formatResponse, formatNumber, escapeHtml) {
  const items = Array.isArray(params) ? params : [params];
  if (!items.length) return "";
  const lines = [escapeHtml(items[0].axisValueLabel ?? items[0].name ?? "")];
  items.forEach((item) => {
    const value = seriesValue(item);
    const formatter = item.seriesName === weightLabel ? formatNumber : formatResponse;
    lines.push(`${item.marker || ""}${escapeHtml(item.seriesName)}: ${escapeHtml(formatter(value))}`);
  });
  return lines.join("<br/>");
}

function seriesValue(params) {
  return Array.isArray(params?.value) ? params.value[1] : params?.value;
}

function responseFormatter(transform, kpiFormat = null) {
  if (transform === "one") return formatUpliftPercent;
  return kpiFormat ? (value) => formatKpiValue(value, kpiFormat) : defaultNumberFormatter;
}

function formatKpiValue(value, kpiFormat) {
  if (value === null || value === undefined || Number.isNaN(value)) return "";
  const number = Number(value);
  if (!Number.isFinite(number)) return "";
  const decimals = Number(kpiFormat?.decimals);
  const fractionDigits = Number.isInteger(decimals) ? Math.max(0, Math.min(12, decimals)) : 2;
  const valueFormat = String(kpiFormat?.format || "number").toLowerCase();
  const displayNumber = valueFormat === "percent" ? number * 100 : number;
  const formatted = Math.abs(displayNumber).toLocaleString(undefined, {
    minimumFractionDigits: fractionDigits,
    maximumFractionDigits: fractionDigits,
  });
  const sign = displayNumber < 0 ? "-" : "";
  if (valueFormat === "currency") return `${sign}£${formatted}`;
  const signed = `${sign}${formatted}`;
  return valueFormat === "percent" ? `${signed}%` : signed;
}

function formatUpliftPercent(value) {
  const number = Number(value);
  if (!Number.isFinite(number)) return "";
  const percent = Math.abs(number - 1) < 1e-12 ? 0 : (number - 1) * 100;
  const digits = Math.abs(percent) < 1 ? 2 : Math.abs(percent) < 10 ? 1 : 0;
  return `${percent > 0 ? "+" : ""}${percent.toLocaleString(undefined, { maximumFractionDigits: digits })}%`;
}

function defaultNumberFormatter(value) {
  const number = Number(value);
  if (!Number.isFinite(number)) return "";
  const abs = Math.abs(number);
  const digits = abs > 0 && abs < 1 ? 3 : abs < 100 ? 2 : 0;
  return number.toLocaleString(undefined, { maximumFractionDigits: digits });
}

function escapeHtmlText(value) {
  return String(value ?? "").replace(/[&<>"']/g, (character) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[character]));
}

function finiteNumberOrNull(value) {
  if (value === null || value === undefined || value === "") return null;
  const number = Number(value);
  return Number.isFinite(number) ? number : null;
}
