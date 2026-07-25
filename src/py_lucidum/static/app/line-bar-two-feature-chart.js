const SURFACE_BOX_MIN_WIDTH = 100;
const SURFACE_BOX_MAX_WIDTH = 140;
const SURFACE_BOX_MIN_DEPTH = 74;
const SURFACE_BOX_MAX_DEPTH = 92;
const HEATMAP_Y_AXIS_MIN_LEFT = 88;
const HEATMAP_Y_AXIS_MAX_LEFT = 280;
const HEATMAP_Y_AXIS_MAX_WIDTH_RATIO = 0.4;
const HEATMAP_Y_AXIS_LABEL_MARGIN = 8;
const HEATMAP_Y_AXIS_LABEL_MEASURE_PADDING = 6;
const HEATMAP_Y_AXIS_TITLE_CLEARANCE = 14;
const HEATMAP_Y_AXIS_OUTER_PADDING = 20;
const HEATMAP_Y_AXIS_MAX_FONT_SIZE = 12;
const HEATMAP_Y_AXIS_MIN_FONT_SIZE = 6;
const HEATMAP_Y_AXIS_FONT_HEIGHT_RATIO = 0.78;
const HEATMAP_LABEL_MAX_FONT_SIZE = 12;
const HEATMAP_LABEL_MIN_FONT_SIZE = 7;
const HEATMAP_LABEL_HORIZONTAL_PADDING = 8;
const HEATMAP_LABEL_VERTICAL_PADDING = 6;
const HEATMAP_LABEL_LINE_HEIGHT_FACTOR = 1.2;
const HEATMAP_DIVERGING_COLORS = ["#1d4ed8", "#f8fafc", "#b91c1c"];
const HEATMAP_LABEL_DARK_COLOR = "#0f172a";
const HEATMAP_LABEL_LIGHT_COLOR = "#ffffff";
const LINE_COLORS = ["#4fb99f", "#ff7f50", "#8aa1d6", "#b779d6", "#e7b84b", "#5aa2d6", "#d96a8a", "#84b547"];

export function twoFeatureChartOption(data, metric, options = {}) {
  const rows = Array.isArray(data?.rows) ? data.rows : [];
  const groupings = Array.isArray(data?.groupings) ? data.groupings : [];
  if (groupings.length !== 2) return emptyOption("Choose two grouping features", options);
  if (!rows.length) {
    const message = data?.dense_grid_too_large
      ? "Two-feature surface is too large to plot"
      : "No two-feature chart data";
    return emptyOption(message, options);
  }
  if (data.plot_type === "surface") return surfaceOption(data, metric, options);
  if (data.plot_type === "lines") return linesOption(data, metric, options);
  return heatmapOption(data, metric, options);
}

function surfaceOption(data, metric, options) {
  const rows = data.rows.filter((row) => !row.group0_missing && !row.group1_missing);
  const xGrouping = data.groupings[1];
  const yGrouping = data.groupings[0];
  const xValues = sortedContinuousValues(rows, "group1_sort", xGrouping);
  const yValues = sortedContinuousValues(rows, "group0_sort", yGrouping);
  if (xValues.length < 2 || yValues.length < 2) return emptyOption("No two-feature surface data", options);
  const byCell = new Map(rows.map((row) => [
    cellKey(
      continuousCoordinate(row.group1_sort, xGrouping),
      continuousCoordinate(row.group0_sort, yGrouping),
    ),
    metricValue(row, metric),
  ]));
  const points = [];
  for (const y of yValues) {
    for (const x of xValues) {
      const value = byCell.get(cellKey(x, y));
      points.push([x, y, value === null || value === undefined ? Number.NaN : value]);
    }
  }
  const values = points.map((point) => finiteNumber(point[2])).filter((value) => value !== null);
  const extent = numericExtent(values);
  const theme = chartTheme(options);
  const box = surfaceBoxSize(options);
  return {
    backgroundColor: "transparent",
    animation: false,
    tooltip: {
      confine: true,
      formatter: (params) => {
        const value = Array.isArray(params?.value) ? params.value : [];
        return `${escapeText(xGrouping.feature)}: ${escapeText(formatContinuousGroupValue(value[0], xGrouping, options))}<br>`
          + `${escapeText(yGrouping.feature)}: ${escapeText(formatContinuousGroupValue(value[1], yGrouping, options))}<br>`
          + `${escapeText(metric.label)}: ${escapeText(metric.format(value[2]))}`;
      },
    },
    visualMap: {
      min: extent.min,
      max: extent.max,
      calculable: true,
      formatter: (value) => metric.format(value),
      right: 10,
      top: 80,
      inRange: { color: [...HEATMAP_DIVERGING_COLORS] },
      textStyle: { color: theme.text },
    },
    grid3D: {
      top: 34,
      left: 0,
      right: 0,
      bottom: 54,
      boxWidth: box.width,
      boxDepth: box.depth,
      viewControl: { projection: "perspective", autoRotate: false },
      axisPointer: { show: true },
      splitLine: { lineStyle: { color: theme.line } },
    },
    xAxis3D: axis3D(xGrouping.feature, theme, (value) => (
      formatContinuousGroupValue(value, xGrouping, options)
    ), {
      min: xValues[0],
      max: xValues[xValues.length - 1],
    }, continuousAxisType(xGrouping)),
    yAxis3D: axis3D(yGrouping.feature, theme, (value) => (
      formatContinuousGroupValue(value, yGrouping, options)
    ), {
      min: yValues[0],
      max: yValues[yValues.length - 1],
    }, continuousAxisType(yGrouping)),
    zAxis3D: axis3D(metric.axisLabel || metric.label, theme, metric.format),
    series: [{
      type: "surface",
      data: points,
      dataShape: [yValues.length, xValues.length],
      shading: "lambert",
      itemStyle: { opacity: 0.96 },
    }],
  };
}

function surfaceBoxSize(options = {}) {
  const chartWidth = finiteNumber(options.chartWidth) || 1200;
  const chartHeight = finiteNumber(options.chartHeight) || 800;
  const fittedWidth = Math.round(Math.min(chartWidth / 12, chartHeight / 5));
  const width = Math.max(SURFACE_BOX_MIN_WIDTH, Math.min(SURFACE_BOX_MAX_WIDTH, fittedWidth));
  const widthProgress = (width - SURFACE_BOX_MIN_WIDTH)
    / (SURFACE_BOX_MAX_WIDTH - SURFACE_BOX_MIN_WIDTH);
  const depth = Math.round(
    SURFACE_BOX_MIN_DEPTH
      + widthProgress * (SURFACE_BOX_MAX_DEPTH - SURFACE_BOX_MIN_DEPTH),
  );
  return { width, depth };
}

function linesOption(data, metric, options) {
  const continuousIndex = data.groupings[0].continuous ? 0 : 1;
  const seriesIndex = continuousIndex === 0 ? 1 : 0;
  const continuousGrouping = data.groupings[continuousIndex];
  const xKey = `group${continuousIndex}_sort`;
  const xMissingKey = `group${continuousIndex}_missing`;
  const seriesKey = `group${seriesIndex}`;
  const grouped = new Map();
  const xValues = sortedContinuousValues(
    data.rows.filter((row) => !row[xMissingKey]),
    xKey,
    continuousGrouping,
  );
  for (const row of data.rows) {
    if (row[xMissingKey]) continue;
    const coordinate = continuousCoordinate(row[xKey], continuousGrouping);
    if (coordinate === null) continue;
    const name = String(row[seriesKey]);
    if (!grouped.has(name)) grouped.set(name, { responseByX: new Map(), volumeByX: new Map() });
    grouped.get(name).responseByX.set(coordinate, metricValue(row, metric));
    grouped.get(name).volumeByX.set(coordinate, finiteNumber(row.volume));
  }
  for (const values of grouped.values()) {
    values.response = xValues.map((x) => values.responseByX.get(x) ?? null);
    values.volume = xValues.map((x) => values.volumeByX.get(x) ?? null);
  }
  const theme = chartTheme(options);
  const volumeLabel = data.denominator?.bar_label || "Weight";
  const responseAxisLayout = verticalValueAxisLayout(
    data.rows.map((row) => metricValue(row, metric)),
    metric.format,
    options,
  );
  const volumeAxisLayout = verticalValueAxisLayout(
    data.rows.map((row) => row.volume),
    compactNumber,
    options,
  );
  const groupEntries = [...grouped.entries()];
  const xLabels = Array.isArray(options.xAxisLabels) && options.xAxisLabels.length === xValues.length
    ? options.xAxisLabels
    : xValues.map((value) => formatGroupValue(value, options));
  const xAxis = categoryAxis(
    xLabels,
    data.groupings[continuousIndex].feature,
    theme,
    options.xAxisLabelPolicy,
  );
  const series = [];
  groupEntries.forEach(([name, values], index) => {
    const color = LINE_COLORS[index % LINE_COLORS.length];
    series.push({
      name,
      type: "line",
      yAxisIndex: 0,
      z: 3,
      data: values.response,
      symbol: values.response.length < 250 ? "circle" : "none",
      symbolSize: 5,
      animation: false,
      lineStyle: { color, width: 2 },
      itemStyle: { color },
    });
    series.push({
      name,
      type: "bar",
      yAxisIndex: 1,
      z: 1,
      stack: "two-feature-volume",
      data: values.volume,
      animation: false,
      barMaxWidth: 28,
      itemStyle: { color, opacity: 0.34 },
      emphasis: { itemStyle: { color, opacity: 0.54 } },
    });
  });
  return {
    backgroundColor: "transparent",
    animation: false,
    color: LINE_COLORS,
    tooltip: {
      trigger: "axis",
      confine: true,
      formatter: (params) => {
        const items = Array.isArray(params) ? params : [params];
        if (!items.length) return "";
        const lines = [escapeText(items[0]?.axisValueLabel ?? items[0]?.axisValue ?? "")];
        items.forEach((item) => {
          const value = item?.value;
          const label = item?.seriesType === "bar" ? volumeLabel : metric.label;
          const formatter = item?.seriesType === "bar" ? compactNumber : metric.format;
          lines.push(
            `${item?.marker || ""}${escapeText(item?.seriesName)} ${escapeText(label)}: `
            + `${escapeText(formatter(value))}`
          );
        });
        return lines.join("<br>");
      },
    },
    legend: {
      type: "scroll",
      top: 30,
      left: 24,
      right: 24,
      data: groupEntries.map(([name], index) => ({
        name,
        icon: "circle",
        itemStyle: { color: LINE_COLORS[index % LINE_COLORS.length] },
      })),
      textStyle: { color: theme.text },
    },
    grid: {
      left: responseAxisLayout.gridMargin,
      right: volumeAxisLayout.gridMargin,
      top: 82,
      bottom: options.xAxisLabelPolicy?.bottom ?? 62,
    },
    xAxis,
    yAxis: [
      valueAxis(metric.axisLabel || metric.label, theme, metric.format, responseAxisLayout),
      {
        ...valueAxis(volumeLabel, theme, compactNumber, volumeAxisLayout),
        position: "right",
        scale: false,
        min: 0,
        splitLine: { show: false },
      },
    ],
    dataZoom: options.xAxisLabelPolicy?.dataZoomEnabled
      ? (options.dataZoomOptions || [])
      : [],
    series,
  };
}

function heatmapOption(data, metric, options) {
  const rows = data.rows;
  const xValues = sortedGroupValues(rows, 1);
  const yValues = sortedGroupValues(rows, 0);
  const xIndex = new Map(xValues.map((item, index) => [item.label, index]));
  const yIndex = new Map(yValues.map((item, index) => [item.label, index]));
  const values = rows.map((row) => metricValue(row, metric)).filter((value) => value !== null);
  const extent = numericExtent(values);
  const theme = chartTheme(options);
  const xLabels = Array.isArray(options.xAxisLabels) && options.xAxisLabels.length === xValues.length
    ? options.xAxisLabels
    : xValues.map((item) => item.label);
  const xAxisLabelPolicy = options.xAxisLabelPolicy || null;
  const yLabels = yValues.map((item) => item.label);
  const yAxisLayout = heatmapYAxisLayout(yLabels, options);
  const heatmapLabels = options.heatmapLabelConfig || twoFeatureHeatmapLabelConfig(data, options);
  const yAxis = categoryAxis(yLabels, data.groupings[0].feature, theme);
  yAxis.nameGap = yAxisLayout.nameGap;
  yAxis.axisLabel = {
    ...yAxis.axisLabel,
    align: "right",
    rotate: 0,
    interval: 0,
    hideOverlap: false,
    fontSize: yAxisLayout.fontSize,
    formatter: (value) => truncateHeatmapYAxisLabel(
      value,
      yAxisLayout.labelWidth,
      options,
      yAxisLayout.fontSize,
    ),
  };
  return {
    backgroundColor: "transparent",
    animation: false,
    tooltip: {
      trigger: "item",
      confine: true,
      formatter: (params) => {
        const value = Array.isArray(params?.value) ? params.value : [];
        return `${escapeText(data.groupings[1].feature)}: ${escapeText(xValues[value[0]]?.label || "")}<br>`
          + `${escapeText(data.groupings[0].feature)}: ${escapeText(yValues[value[1]]?.label || "")}<br>`
          + `${escapeText(metric.label)}: ${escapeText(metric.format(value[2]))}`;
      },
    },
    grid: {
      left: yAxisLayout.gridLeft,
      right: 96,
      top: 42,
      bottom: xAxisLabelPolicy?.bottom ?? 84,
    },
    xAxis: categoryAxis(xLabels, data.groupings[1].feature, theme, xAxisLabelPolicy),
    yAxis,
    visualMap: {
      min: extent.min,
      max: extent.max,
      calculable: true,
      formatter: (value) => metric.format(value),
      orient: "vertical",
      right: 8,
      top: 70,
      inRange: { color: [...HEATMAP_DIVERGING_COLORS] },
      textStyle: { color: theme.text },
    },
    dataZoom: xAxisLabelPolicy?.dataZoomEnabled
      ? (options.dataZoomOptions || [])
      : [],
    series: [{
      name: metric.label,
      type: "heatmap",
      data: rows.map((row) => [
        xIndex.get(String(row.group1)),
        yIndex.get(String(row.group0)),
        metricValue(row, metric),
      ]),
      label: {
        show: heatmapLabels.show,
        position: "inside",
        align: "center",
        verticalAlign: "middle",
        fontSize: heatmapLabels.fontSize,
        lineHeight: heatmapLabels.lineHeight,
        textBorderWidth: 0,
        rich: heatmapLabelRichStyles(heatmapLabels.fontSize, heatmapLabels.lineHeight),
        formatter: (params) => {
          const value = Array.isArray(params?.value) ? params.value : [];
          return heatmapRichLabel(
            heatmapLabels.formatCell(value[0], value[1]),
            value[2],
            extent,
            heatmapLabels.mode,
          );
        },
      },
      emphasis: { itemStyle: { borderColor: theme.text, borderWidth: 1 } },
    }],
  };
}

function heatmapLabelRichStyles(fontSize, lineHeight) {
  const shared = { fontSize, lineHeight, align: "center" };
  return {
    heatmapWhitePrimary: {
      ...shared,
      color: HEATMAP_LABEL_LIGHT_COLOR,
      fontWeight: 600,
      textShadowColor: "rgba(15, 23, 42, 0.28)",
      textShadowBlur: 2,
      textShadowOffsetY: 1,
    },
    heatmapWhiteSecondary: {
      ...shared,
      color: HEATMAP_LABEL_LIGHT_COLOR,
      fontWeight: 500,
      textShadowColor: "rgba(15, 23, 42, 0.28)",
      textShadowBlur: 2,
      textShadowOffsetY: 1,
    },
    heatmapDarkPrimary: {
      ...shared,
      color: HEATMAP_LABEL_DARK_COLOR,
      fontWeight: 600,
      textShadowBlur: 0,
    },
    heatmapDarkSecondary: {
      ...shared,
      color: HEATMAP_LABEL_DARK_COLOR,
      fontWeight: 500,
      textShadowBlur: 0,
    },
  };
}

function heatmapRichLabel(text, value, extent, mode) {
  const tone = heatmapLabelTone(value, extent);
  return String(text ?? "").split("\n").map((line, index) => {
    const hierarchy = mode === "both" && index > 0 ? "Secondary" : "Primary";
    return `{heatmap${tone}${hierarchy}|${escapeRichText(line)}}`;
  }).join("\n");
}

function heatmapLabelTone(value, extent) {
  const cellColor = heatmapInterpolatedColor(value, extent);
  const whiteContrast = contrastRatio(cellColor, hexToRgb(HEATMAP_LABEL_LIGHT_COLOR));
  const darkContrast = contrastRatio(cellColor, hexToRgb(HEATMAP_LABEL_DARK_COLOR));
  return whiteContrast >= darkContrast ? "White" : "Dark";
}

function heatmapInterpolatedColor(value, extent) {
  const numeric = finiteNumber(value);
  const minimum = finiteNumber(extent?.min);
  const maximum = finiteNumber(extent?.max);
  if (numeric === null || minimum === null || maximum === null || maximum <= minimum) {
    return hexToRgb(HEATMAP_DIVERGING_COLORS[1]);
  }
  const position = Math.max(0, Math.min(1, (numeric - minimum) / (maximum - minimum)));
  if (position <= 0.5) {
    return interpolateRgb(
      hexToRgb(HEATMAP_DIVERGING_COLORS[0]),
      hexToRgb(HEATMAP_DIVERGING_COLORS[1]),
      position * 2,
    );
  }
  return interpolateRgb(
    hexToRgb(HEATMAP_DIVERGING_COLORS[1]),
    hexToRgb(HEATMAP_DIVERGING_COLORS[2]),
    (position - 0.5) * 2,
  );
}

function hexToRgb(value) {
  const hex = String(value || "").replace(/^#/, "");
  const numeric = Number.parseInt(hex, 16);
  return {
    red: (numeric >> 16) & 255,
    green: (numeric >> 8) & 255,
    blue: numeric & 255,
  };
}

function interpolateRgb(start, end, amount) {
  const channel = (name) => Math.round(start[name] + (end[name] - start[name]) * amount);
  return {
    red: channel("red"),
    green: channel("green"),
    blue: channel("blue"),
  };
}

function contrastRatio(left, right) {
  const leftLuminance = relativeLuminance(left);
  const rightLuminance = relativeLuminance(right);
  const lighter = Math.max(leftLuminance, rightLuminance);
  const darker = Math.min(leftLuminance, rightLuminance);
  return (lighter + 0.05) / (darker + 0.05);
}

function relativeLuminance(color) {
  const channel = (value) => {
    const srgb = value / 255;
    return srgb <= 0.04045
      ? srgb / 12.92
      : ((srgb + 0.055) / 1.055) ** 2.4;
  };
  return (
    0.2126 * channel(color.red)
    + 0.7152 * channel(color.green)
    + 0.0722 * channel(color.blue)
  );
}

function escapeRichText(value) {
  return String(value ?? "").replace(/([\\{}])/g, "\\$1");
}

export function twoFeatureHeatmapLabelConfig(data, options = {}) {
  const rows = Array.isArray(data?.rows) ? data.rows : [];
  const xValues = sortedGroupValues(rows, 1);
  const yValues = sortedGroupValues(rows, 0);
  const xIndex = new Map(xValues.map((item, index) => [item.label, index]));
  const yIndex = new Map(yValues.map((item, index) => [item.label, index]));
  const rowByCell = new Map(rows.map((row) => [
    cellKey(xIndex.get(String(row.group1)), yIndex.get(String(row.group0))),
    row,
  ]));
  const mode = normaliseHeatmapLabelMode(options.heatmapLabelMode);
  const formatActual = typeof options.formatActual === "function" ? options.formatActual : String;
  const formatWeight = typeof options.formatWeight === "function" ? options.formatWeight : String;
  const yAxisLayout = heatmapYAxisLayout(yValues.map((item) => item.label), options);
  const grid = {
    left: yAxisLayout.gridLeft,
    right: 96,
    top: 42,
    bottom: options.xAxisLabelPolicy?.bottom ?? 84,
  };
  const chartWidth = finiteNumber(options.chartWidth) || 1200;
  const chartHeight = finiteNumber(options.chartHeight) || 800;
  const cellLimit = finiteNumber(options.heatmapLabelCellLimit);
  const cellWidth = Math.max(0, chartWidth - grid.left - grid.right) / Math.max(1, xValues.length);
  const cellHeight = Math.max(0, chartHeight - grid.top - grid.bottom) / Math.max(1, yValues.length);
  const denseCellCount = xValues.length * yValues.length;
  const withinCellLimit = cellLimit === null || cellLimit <= 0 || denseCellCount < cellLimit;
  const fittedFontSizes = Object.fromEntries(
    ["actual", "weight", "both"].map((labelMode) => [
      labelMode,
      heatmapLabelFontSize(
        rows,
        labelMode,
        cellWidth,
        cellHeight,
        formatActual,
        formatWeight,
        options,
      ),
    ]),
  );
  const availableModes = Object.fromEntries(
    Object.entries(fittedFontSizes).map(([labelMode, fitted]) => [
      labelMode,
      Boolean(rows.length && withinCellLimit && fitted >= HEATMAP_LABEL_MIN_FONT_SIZE),
    ]),
  );
  const available = Object.values(availableModes).some(Boolean);
  const fittedFontSize = fittedFontSizes[mode] ?? HEATMAP_LABEL_MAX_FONT_SIZE;
  const fontSize = availableModes[mode]
    ? Math.max(HEATMAP_LABEL_MIN_FONT_SIZE, Math.min(HEATMAP_LABEL_MAX_FONT_SIZE, fittedFontSize))
    : HEATMAP_LABEL_MIN_FONT_SIZE;
  return {
    available,
    availableModes,
    show: mode !== "none" && Boolean(availableModes[mode]),
    mode,
    denseCellCount,
    fontSize,
    lineHeight: Math.ceil(fontSize * HEATMAP_LABEL_LINE_HEIGHT_FACTOR),
    formatCell: (x, y) => {
      const row = rowByCell.get(cellKey(x, y));
      return row ? heatmapCellLabel(row, mode, formatActual, formatWeight) : "";
    },
  };
}

function normaliseHeatmapLabelMode(value) {
  const mode = String(value || "none").toLowerCase();
  return ["actual", "weight", "both"].includes(mode) ? mode : "none";
}

function heatmapLabelFontSize(rows, mode, cellWidth, cellHeight, formatActual, formatWeight, options) {
  const lineCount = mode === "both" ? 2 : 1;
  const labels = rows.flatMap((row) => heatmapCellLabelLines(row, mode, formatActual, formatWeight));
  const widest = labels.reduce(
    (maximum, label) => Math.max(
      maximum,
      measureHeatmapLabelWidth(label, HEATMAP_LABEL_MAX_FONT_SIZE, options),
    ),
    0,
  );
  const usableWidth = Math.max(0, cellWidth - HEATMAP_LABEL_HORIZONTAL_PADDING);
  const usableHeight = Math.max(0, cellHeight - HEATMAP_LABEL_VERTICAL_PADDING);
  const widthScale = widest > 0 ? usableWidth / widest : 1;
  const baseHeight = lineCount * HEATMAP_LABEL_MAX_FONT_SIZE * HEATMAP_LABEL_LINE_HEIGHT_FACTOR;
  const heightScale = usableHeight / Math.max(1, baseHeight);
  return Math.floor(
    HEATMAP_LABEL_MAX_FONT_SIZE * Math.min(1, widthScale, heightScale),
  );
}

function heatmapCellLabel(row, mode, formatActual, formatWeight) {
  return heatmapCellLabelLines(row, mode, formatActual, formatWeight).join("\n");
}

function heatmapCellLabelLines(row, mode, formatActual, formatWeight) {
  const actual = optionalFiniteNumber(row?.resp0);
  const weight = optionalFiniteNumber(row?.volume);
  if (mode === "actual") return [actual === null ? "" : String(formatActual(actual) ?? "")];
  if (mode === "weight") return [weight === null ? "" : String(formatWeight(weight) ?? "")];
  if (mode === "both") {
    return [
      actual === null ? "" : String(formatActual(actual) ?? ""),
      weight === null ? "" : String(formatWeight(weight) ?? ""),
    ];
  }
  return [""];
}

function optionalFiniteNumber(value) {
  if (value === null || value === undefined || value === "") return null;
  return finiteNumber(value);
}

function measureHeatmapLabelWidth(value, fontSize, options) {
  const text = String(value ?? "");
  if (typeof options?.measureText === "function") {
    const measured = finiteNumber(options.measureText(text, fontSize));
    if (measured !== null && measured >= 0) return measured;
  }
  return text.length * fontSize * 0.58;
}

function heatmapYAxisLayout(labels, options) {
  const fontSize = heatmapYAxisFontSize(labels.length, options);
  const measuredWidth = labels.reduce(
    (maximum, label) => Math.max(maximum, measureLabelWidth(label, options, fontSize)),
    0,
  );
  const chartWidth = finiteNumber(options?.chartWidth) || 1200;
  const maximumLeft = Math.max(
    HEATMAP_Y_AXIS_MIN_LEFT,
    Math.min(HEATMAP_Y_AXIS_MAX_LEFT, Math.floor(chartWidth * HEATMAP_Y_AXIS_MAX_WIDTH_RATIO)),
  );
  const maximumLabelWidth = Math.max(
    24,
    maximumLeft
      - HEATMAP_Y_AXIS_LABEL_MARGIN
      - HEATMAP_Y_AXIS_TITLE_CLEARANCE
      - HEATMAP_Y_AXIS_OUTER_PADDING,
  );
  const labelWidth = Math.min(
    Math.ceil(measuredWidth) + HEATMAP_Y_AXIS_LABEL_MEASURE_PADDING,
    maximumLabelWidth,
  );
  const nameGap = labelWidth + HEATMAP_Y_AXIS_LABEL_MARGIN + HEATMAP_Y_AXIS_TITLE_CLEARANCE;
  const gridLeft = Math.max(
    HEATMAP_Y_AXIS_MIN_LEFT,
    Math.min(maximumLeft, nameGap + HEATMAP_Y_AXIS_OUTER_PADDING),
  );
  return { gridLeft, labelWidth, nameGap, fontSize };
}

function heatmapYAxisFontSize(categoryCount, options) {
  const chartHeight = finiteNumber(options?.chartHeight) || 800;
  const gridBottom = finiteNumber(options?.xAxisLabelPolicy?.bottom) ?? 84;
  const plotHeight = Math.max(0, chartHeight - 42 - gridBottom);
  const categoryHeight = plotHeight / Math.max(1, categoryCount);
  const fitted = Math.floor(categoryHeight * HEATMAP_Y_AXIS_FONT_HEIGHT_RATIO);
  return Math.max(
    HEATMAP_Y_AXIS_MIN_FONT_SIZE,
    Math.min(HEATMAP_Y_AXIS_MAX_FONT_SIZE, fitted),
  );
}

function truncateHeatmapYAxisLabel(value, maximumWidth, options, fontSize) {
  const text = String(value ?? "");
  if (measureLabelWidth(text, options, fontSize) <= maximumWidth) return text;
  const ellipsis = "…";
  if (measureLabelWidth(ellipsis, options, fontSize) >= maximumWidth) return ellipsis;
  const characters = [...text];
  let lower = 0;
  let upper = characters.length;
  while (lower < upper) {
    const middle = Math.ceil((lower + upper) / 2);
    const candidate = `${characters.slice(0, middle).join("")}${ellipsis}`;
    if (measureLabelWidth(candidate, options, fontSize) <= maximumWidth) lower = middle;
    else upper = middle - 1;
  }
  return `${characters.slice(0, lower).join("")}${ellipsis}`;
}

function measureLabelWidth(value, options, fontSize = 12) {
  const text = String(value ?? "");
  if (typeof options?.measureText === "function") {
    const measured = finiteNumber(options.measureText(text, fontSize));
    if (measured !== null && measured >= 0) return measured;
  }
  return text.length * fontSize * 0.58;
}

function metricValue(row, metric) {
  return finiteNumber(row?.[metric.key]);
}

function sortedContinuousValues(rows, key, grouping) {
  return [...new Set(rows.map((row) => continuousCoordinate(row[key], grouping)).filter((value) => value !== null))]
    .sort((left, right) => left - right);
}

function continuousCoordinate(value, grouping) {
  if (!isDateGrouping(grouping)) return finiteNumber(value);
  if (value instanceof Date) {
    const timestamp = value.getTime();
    return Number.isFinite(timestamp) ? timestamp : null;
  }
  const text = String(value ?? "").trim();
  const match = text.match(
    /^(\d{4})-(\d{2})-(\d{2})(?:[ T](\d{2}):(\d{2})(?::(\d{2})(?:\.\d+)?)?)?/,
  );
  const timestamp = match
    ? Date.UTC(
        Number(match[1]),
        Number(match[2]) - 1,
        Number(match[3]),
        Number(match[4] || 0),
        Number(match[5] || 0),
        Number(match[6] || 0),
      )
    : Date.parse(text);
  return Number.isFinite(timestamp) ? timestamp : null;
}

function continuousAxisType(grouping) {
  return isDateGrouping(grouping) ? "time" : "value";
}

function isDateGrouping(grouping) {
  return grouping?.kind === "date" || grouping?.kind === "datetime";
}

function formatContinuousGroupValue(value, grouping, options) {
  if (isDateGrouping(grouping) && typeof options.formatDateGroupValue === "function") {
    return options.formatDateGroupValue(value, grouping);
  }
  return formatGroupValue(value, options);
}

function sortedGroupValues(rows, index) {
  const byLabel = new Map();
  for (const row of rows) {
    const label = String(row[`group${index}`]);
    if (!byLabel.has(label)) byLabel.set(label, row[`group${index}_sort`]);
  }
  return [...byLabel.entries()]
    .map(([label, sort]) => ({ label, sort }))
    .sort((left, right) => compareSortValues(left.sort, right.sort) || left.label.localeCompare(right.label));
}

function compareSortValues(left, right) {
  const leftNumber = finiteNumber(left);
  const rightNumber = finiteNumber(right);
  if (leftNumber !== null && rightNumber !== null) return leftNumber - rightNumber;
  return String(left ?? "").localeCompare(String(right ?? ""), undefined, { numeric: true, sensitivity: "base" });
}

function numericExtent(values) {
  const finite = values.map(finiteNumber).filter((value) => value !== null);
  if (!finite.length) return { min: 0, max: 1 };
  const min = Math.min(...finite);
  const max = Math.max(...finite);
  if (min === max) {
    const padding = Math.abs(min || 1) * 0.1;
    return { min: min - padding, max: max + padding };
  }
  return { min, max };
}

function axis3D(name, theme, formatter = null, domain = null, type = "value") {
  return {
    type,
    name,
    ...(domain ? { min: domain.min, max: domain.max } : {}),
    nameTextStyle: { color: theme.text, fontSize: 11, fontWeight: 700 },
    axisLabel: { color: theme.text, fontSize: 10, formatter: formatter || compactNumber },
    axisLine: { lineStyle: { color: theme.line } },
    splitLine: { lineStyle: { color: theme.grid } },
  };
}

function valueAxis(name, theme, formatter = null, layout = null) {
  return {
    type: "value",
    name,
    scale: true,
    nameLocation: "middle",
    nameGap: layout?.nameGap ?? 52,
    nameTextStyle: { color: theme.text, fontWeight: 700 },
    axisLabel: { color: theme.text, formatter: formatter || compactNumber },
    axisLine: { lineStyle: { color: theme.line } },
    splitLine: { lineStyle: { color: theme.grid } },
  };
}

function verticalValueAxisLayout(values, formatter, options = {}) {
  const formatted = values
    .map(finiteNumber)
    .filter((value) => value !== null)
    .flatMap((value) => {
      try {
        return [String((formatter || compactNumber)(value) ?? "")];
      } catch (_error) {
        return [String(value)];
      }
    });
  if (!formatted.includes("0")) formatted.push("0");
  const measureText = typeof options.measureText === "function"
    ? options.measureText
    : (value, fontSize) => String(value).length * fontSize * 0.56;
  const labelWidth = formatted.reduce(
    (maximum, value) => Math.max(maximum, finiteNumber(measureText(value, 12)) || 0),
    0,
  );
  const nameGap = Math.max(52, Math.ceil(labelWidth + 18));
  return {
    nameGap,
    gridMargin: Math.max(76, nameGap + 28),
  };
}

function categoryAxis(labels, name, theme, policy = null) {
  return {
    type: "category",
    data: labels,
    name,
    nameLocation: "middle",
    nameGap: policy?.nameGap ?? (labels.length > 25 ? 62 : 38),
    nameTextStyle: { color: theme.text, fontWeight: 700 },
    axisLabel: policy ? {
      show: policy.show,
      color: theme.text,
      interval: policy.interval,
      formatter: policy.formatter,
      hideOverlap: Boolean(policy.hideOverlap),
      showMinLabel: policy.showMinLabel,
      showMaxLabel: policy.showMaxLabel,
      rotate: policy.rotate,
      fontSize: policy.fontSize,
      margin: 8,
    } : {
      color: theme.text,
      rotate: labels.length > 25 ? 60 : 0,
    },
    axisLine: { lineStyle: { color: theme.line } },
  };
}

function emptyOption(message, options) {
  const theme = chartTheme(options);
  return {
    backgroundColor: "transparent",
    title: {
      text: message,
      left: "center",
      top: "middle",
      textStyle: { color: theme.muted, fontSize: 14, fontWeight: 500 },
    },
    series: [],
  };
}

function chartTheme(options) {
  return {
    text: options.text || "#334155",
    muted: options.muted || "#64748b",
    line: options.line || "#cbd5e1",
    grid: options.grid || "#e5e7eb",
  };
}

function finiteNumber(value) {
  const number = Number(value);
  return Number.isFinite(number) ? number : null;
}

function compactNumber(value) {
  const number = finiteNumber(value);
  if (number === null) return "";
  return new Intl.NumberFormat("en-GB", { maximumSignificantDigits: 5 }).format(number);
}

function formatGroupValue(value, options) {
  return options.formatGroupValue ? options.formatGroupValue(value) : compactNumber(value);
}

function cellKey(x, y) {
  return `${x}\u0000${y}`;
}

function escapeText(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#39;");
}
