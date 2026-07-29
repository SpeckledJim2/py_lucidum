import { loadTabulator } from "./shared/tabulator.js";
import { bindSettingsStripOverflowCue } from "./shared/settings-strip.js";

const HISTOGRAM_BINS_REFRESH_DELAY_MS = 250;
const DEFAULT_HISTOGRAM_MEAN_COLOR = "#d13f3f";
const DEFAULT_HISTOGRAM_MEDIAN_COLOR = "#1f7a8c";
const HISTOGRAM_MEAN_ROW_CLASS = "histogram-stat-mean-row";
const HISTOGRAM_MEDIAN_ROW_CLASS = "histogram-stat-median-row";
const HISTOGRAM_X_AXIS_MIN_LABELS = 2;
const HISTOGRAM_X_AXIS_MAX_LABELS = 100;
const HISTOGRAM_X_AXIS_FONT_SIZE = 10;
const HISTOGRAM_X_AXIS_LABEL_WIDTH_FACTOR = 0.56;
const HISTOGRAM_X_AXIS_LABEL_PADDING = 10;
const HISTOGRAM_GRID_LEFT_MIN = 72;
const HISTOGRAM_GRID_RIGHT = 30;
const HISTOGRAM_GRID_TOP = 40;
const HISTOGRAM_GRID_TOP_WITH_BIN_LABELS = 56;
const HISTOGRAM_X_AXIS_ROTATION = 65;
const HISTOGRAM_Y_AXIS_TARGET_INTERVALS = 6;
const HISTOGRAM_Y_AXIS_FONT_SIZE = 12;
const HISTOGRAM_Y_AXIS_LABEL_WIDTH_FACTOR = 0.56;
const HISTOGRAM_Y_AXIS_LABEL_MARGIN = 8;
const HISTOGRAM_Y_AXIS_OUTER_PADDING = 8;
const HISTOGRAM_Y_AXIS_NAME_GAP = 19;
const HISTOGRAM_AXIS_NAME_MIN_WIDTH = 40;
const HISTOGRAM_BIN_OUTLINE_LIMIT = 200;
const HISTOGRAM_BIN_LABEL_MIN_FONT_SIZE = 7;
const HISTOGRAM_BIN_LABEL_MAX_FONT_SIZE = 10;
const HISTOGRAM_BIN_LABEL_WIDTH_FACTOR = 0.56;
const HISTOGRAM_MEDIAN_LABEL_OFFSET = 14;
const HISTOGRAM_STATS_DEFAULT_WIDTH = 240;
const HISTOGRAM_STATS_MIN_WIDTH = 240;
const HISTOGRAM_STATS_MAX_WIDTH = 560;
const HISTOGRAM_CHART_MIN_WIDTH = 420;
const HISTOGRAM_SPLITTER_KEY_STEP = 10;
const HISTOGRAM_STACKED_MEDIA = "(max-width: 900px)";

export function createHistogramTool({
  api,
  el,
  state,
  echartsImpl,
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
  clearActiveFavouriteSelection = () => {},
}) {
  const chart = echartsImpl.init(el("histogramChart"));
  let statsTable = null;
  let statsRenderToken = 0;
  let histogramBinsRefreshTimer = null;
  let histogramStatsWidth = HISTOGRAM_STATS_DEFAULT_WIDTH;
  let histogramChartResizeFrame = null;
  let histogramAxisChartWidth = 0;
  const histogramBinValues = { count: "auto", width: "" };

  function syncSegmented(control, value) {
    const group = document.querySelector(`.segmented[data-control="${control}"]`);
    if (!group) return;
    group.querySelectorAll("button").forEach((button) => {
      button.classList.toggle("active", button.dataset.value === value);
    });
  }

  function histogramBinMode() {
    return state.histogramBinMode === "width" ? "width" : "count";
  }

  function histogramBinsValue() {
    return normaliseHistogramBinsValue(histogramBinValues.count);
  }

  function histogramBinWidthValue() {
    return String(histogramBinValues.width || "").trim();
  }

  function normaliseHistogramBinsValue(value) {
    const raw = String(value ?? "auto").trim();
    return raw || "auto";
  }

  function normaliseHistogramBinWidthValue(value) {
    const raw = String(value ?? "").trim();
    const number = Number(raw.replaceAll(",", ""));
    return Number.isFinite(number) && number > 0 ? raw : "";
  }

  function formatHistogramBinInputNumber(value) {
    const number = Number(value);
    if (!Number.isFinite(number)) return "";
    return String(Number(number.toPrecision(10)));
  }

  function suggestedHistogramBinWidth(data = state.lastHistogramData) {
    const direct = Number(data?.bin_width);
    if (Number.isFinite(direct) && direct > 0) return formatHistogramBinInputNumber(direct);
    const step = Number(data?.binning?.step);
    if (Number.isFinite(step) && step > 0) return formatHistogramBinInputNumber(step);
    const minimum = Number(data?.binning?.min);
    const maximum = Number(data?.binning?.max);
    const bins = Number(data?.bins);
    const derived = (maximum - minimum) / bins;
    return Number.isFinite(derived) && derived > 0 ? formatHistogramBinInputNumber(derived) : "1";
  }

  function syncHistogramBinInput() {
    const mode = histogramBinMode();
    const input = el("histogramBins");
    const label = el("histogramBinValueLabel");
    if (!input) return;
    if (mode === "width") {
      label.textContent = "Bin width";
      input.value = histogramBinWidthValue();
      input.placeholder = "Enter width";
      input.inputMode = "decimal";
      input.setAttribute("aria-label", "Histogram bin width");
    } else {
      label.textContent = "No. bins";
      input.value = histogramBinsValue() === "auto" ? "" : histogramBinsValue();
      input.placeholder = "Auto";
      input.inputMode = "numeric";
      input.setAttribute("aria-label", "Histogram number of bins");
    }
    input.removeAttribute("aria-invalid");
  }

  function captureHistogramBinInput() {
    const raw = String(el("histogramBins")?.value || "").trim();
    if (histogramBinMode() === "width") histogramBinValues.width = raw;
    else histogramBinValues.count = raw || "auto";
    return raw;
  }

  function histogramBinInputIsValid({ report = false } = {}) {
    if (histogramBinMode() !== "width") return true;
    const input = el("histogramBins");
    const value = Number(histogramBinWidthValue().replaceAll(",", ""));
    const valid = Number.isFinite(value) && value > 0;
    input?.setAttribute("aria-invalid", valid ? "false" : "true");
    if (!valid && report) setStatus("Bin width must be a positive number", true);
    return valid;
  }

  function normaliseHistogramControlValue(value, allowed, fallback) {
    const raw = String(value || fallback).trim();
    return allowed.includes(raw) ? raw : fallback;
  }

  function captureFavouriteState() {
    return {
      bins: histogramBinsValue(),
      binMode: histogramBinMode(),
      binWidth: histogramBinWidthValue(),
      distribution: state.histogramDistribution || "incremental",
      yAxis: state.histogramYAxis || "sum",
      labels: state.histogramLabels || "none",
      logScale: state.histogramLogScale || "none",
      sampleMode: state.histogramSampleMode || "100k",
    };
  }

  function applyFavouriteState(payload = {}) {
    if (histogramBinsRefreshTimer) {
      window.clearTimeout(histogramBinsRefreshTimer);
      histogramBinsRefreshTimer = null;
    }
    histogramBinValues.count = normaliseHistogramBinsValue(payload.bins);
    histogramBinValues.width = normaliseHistogramBinWidthValue(payload.binWidth);
    const restoredBinMode = payload.binMode === "width" && histogramBinValues.width ? "width" : "count";
    setSegmentedValue("histogramBinMode", restoredBinMode);
    syncHistogramBinInput();
    setSegmentedValue(
      "histogramDistribution",
      normaliseHistogramControlValue(payload.distribution, ["incremental", "cumulative"], "incremental"),
    );
    setSegmentedValue(
      "histogramYAxis",
      normaliseHistogramControlValue(payload.yAxis, ["sum", "probability"], "sum"),
    );
    setSegmentedValue(
      "histogramLabels",
      normaliseHistogramControlValue(payload.labels, ["none", "bins"], "none"),
    );
    setSegmentedValue(
      "histogramLogScale",
      normaliseHistogramControlValue(payload.logScale, ["none", "x", "y", "both"], "none"),
    );
    setSegmentedValue(
      "histogramSampleMode",
      normaliseHistogramControlValue(payload.sampleMode, ["100k", "all"], "100k"),
    );
  }

  function buildHistogramRequest() {
    const actualOption = el("actualNumerator")?.selectedOptions?.[0] || null;
    if (!state.schema || !actualOption?.value) return null;
    const denominatorOption = el("denominator")?.selectedOptions?.[0] || null;
    if (denominatorOption?.dataset.unavailable === "true") {
      setStatus("The selected model prediction Denominator is unavailable because there is no active model.", true);
      return null;
    }
    const request = {
      source: actualOption?.dataset.sourceId || state.source || "dataset",
      actual: actualOption.value,
      denominator: denominatorOption?.value || "__none__",
      denominatorSource: denominatorOption?.dataset.sourceId || "dataset",
      bins: histogramBinsValue(),
      binMode: histogramBinMode(),
      distribution: state.histogramDistribution || "incremental",
      yAxis: state.histogramYAxis || "sum",
      logScale: state.histogramLogScale || "none",
      sampleMode: state.histogramSampleMode || "100k",
      filter: state.activeFilter,
    };
    if (histogramBinMode() === "width") request.binWidth = histogramBinWidthValue();
    return request;
  }

  async function fetchHistogramData(request, requestKey) {
    const requestSeq = state.histogramRequestSeq + 1;
    state.histogramRequestSeq = requestSeq;
    setStatus("");
    setChartMessage("");
    setGroupMeta("histogram", "Computing...");
    startToolTiming("histogram");
    try {
      const data = await api("/api/histogram/chart", { method: "POST", body: JSON.stringify(request), clientTiming: true });
      if (requestSeq !== state.histogramRequestSeq) return;
      const cache = toolCache("histogram");
      cache.requestKey = requestKey;
      cache.data = data;
      syncDuckDbTimingFromData("histogram", data);
      syncClientTimingFromData("histogram", data);
      measureToolRender("histogram", () => renderHistogramData(data));
      return data;
    } catch (error) {
      if (requestSeq !== state.histogramRequestSeq) return;
      setToolTimingFailed("histogram");
      setGroupMeta("histogram", "Query failed");
      setChartMessage("");
      setStatus(error.message, true);
    }
  }

  function renderHistogramData(data) {
    state.lastHistogramData = data;
    renderChart(data);
    renderStatsTable(data);
    const rowMeta = formatRowMeta(data.row_count, data.filtered_row_count);
    const invalidCount = Math.max(0, Number(data.filtered_row_count || 0) - Number(data.valid_count || 0));
    const invalidLabel = invalidCount > 0 ? `${formatNumber(invalidCount)} invalid` : "";
    const sampledLabel = data.sampled_valid_count && data.sampled_valid_count !== data.valid_count
      ? `${formatNumber(data.sampled_valid_count)} sampled`
      : "";
    const groupMeta = data.bin_mode === "width"
      ? `Width ${formatHistogramBinInputNumber(data.bin_width)} · ${formatNumber(data.bins)} bins - ${rowMeta}`
      : `${formatNumber(data.bins)} bins - ${rowMeta}`;
    const groupMetaBadges = [
      invalidLabel ? `<span class="histogram-invalid-badge">${escapeHtml(invalidLabel)}</span>` : "",
      sampledLabel ? `<span class="histogram-sample-badge">${escapeHtml(sampledLabel)}</span>` : "",
    ].filter(Boolean).join("");
    const groupMetaHtml = groupMetaBadges
      ? `<span class="histogram-meta-text">${escapeHtml(groupMeta)}</span>${groupMetaBadges}`
      : "";
    const warnings = [...(data.warnings || [])].filter(Boolean).join(" ");
    setGroupMeta("histogram", groupMetaHtml || groupMeta, { html: Boolean(groupMetaHtml) });
    setStatus("");
    setChartMessage(warnings);
    saveToolPresentation("histogram", { groupMeta, groupMetaHtml, chartMessage: warnings });
  }

  function useCachedHistogramData(cache, options = {}) {
    state.lastHistogramData = cache.data;
    if (options.renderIfCached) {
      measureToolRender("histogram", () => renderHistogramData(cache.data));
      return;
    }
    measureToolRender("histogram", () => {
      applyToolPresentation("histogram");
      requestAnimationFrame(() => {
        resizeHistogramChart();
        statsTable?.redraw?.(true);
      });
    });
  }

  function renderStatsTable(data) {
    const rows = Array.isArray(data.stats) ? data.stats : [];
    const token = statsRenderToken + 1;
    statsRenderToken = token;
    const target = el("histogramStatsGrid");
    target.classList.add("histogram-grid");
    loadTabulator().then((Tabulator) => {
      if (token !== statsRenderToken) return;
      const tableRows = rows.map((row) => ({
        statistic: row.statistic,
        statisticClass: histogramStatisticRowClass(row.statistic),
        value: formatMetricValue(row.value, row.statistic),
      }));
      if (statsTable) {
        statsTable.replaceData(tableRows);
        statsTable.redraw(true);
        return;
      }
      target.innerHTML = "";
      statsTable = new Tabulator(target, {
        data: tableRows,
        layout: "fitColumns",
        height: "100%",
        reactiveData: false,
        selectable: false,
        rowFormatter: formatHistogramStatRow,
        columns: [
          { title: "Statistic", field: "statistic", headerSort: false, widthGrow: 1.1 },
          { title: "Value", field: "value", headerSort: false, headerHozAlign: "right", hozAlign: "right", widthGrow: 0.9 },
        ],
      });
    }).catch((error) => {
      target.innerHTML = `<div class="feature-list-message">Metrics table failed to load: ${escapeHtml(error.message || String(error))}</div>`;
    });
  }

  function formatMetricValue(value, statistic = "") {
    if (value === null || value === undefined || value === "") return "";
    const label = String(statistic || "").toLowerCase();
    if (label.includes("count")) return formatNumber(value);
    if (label === "weight sum") return formatWeightValue(value);
    return formatLineValue(value);
  }

  function histogramStatisticRowClass(statistic) {
    if (statistic === "Mean") return HISTOGRAM_MEAN_ROW_CLASS;
    if (statistic === "Median") return HISTOGRAM_MEDIAN_ROW_CLASS;
    return "";
  }

  function formatHistogramStatRow(row) {
    const element = row.getElement();
    element.classList.remove(HISTOGRAM_MEAN_ROW_CLASS, HISTOGRAM_MEDIAN_ROW_CLASS);
    const statisticClass = row.getData()?.statisticClass;
    if (statisticClass) element.classList.add(statisticClass);
  }

  function renderChart(data, options = {}) {
    const rows = Array.isArray(data.rows) ? data.rows : [];
    const xLog = data.log_scale === "x" || data.log_scale === "both";
    const yLog = data.log_scale === "y" || data.log_scale === "both";
    const yLabel = yAxisLabel(data);
    const yValues = rows.map((row) => Number(row.height)).filter((value) => Number.isFinite(value) && value > 0);
    const yBaseline = yLog ? (yValues.length ? Math.max(Math.min(...yValues) / 10, 1e-12) : 1e-12) : 0;
    const yAxisPolicy = histogramYAxisPolicy(yValues, yLog);
    const xBounds = axisBounds(rows, xLog);
    const chartWidth = Number(chart.getWidth?.()) || el("histogramChart")?.clientWidth || 800;
    histogramAxisChartWidth = chartWidth;
    const yAxisLayout = histogramYAxisLayout(yValues, yLog, yBaseline, yAxisPolicy, data.y_axis);
    const horizontalPadding = yAxisLayout.gridLeft + HISTOGRAM_GRID_RIGHT;
    const xAxisPolicy = histogramXAxisPolicy(
      data,
      rows,
      xLog,
      chartWidth,
      horizontalPadding,
      formatAxisValue,
    );
    const axisNameMaxWidth = Math.max(HISTOGRAM_AXIS_NAME_MIN_WIDTH, chartWidth - horizontalPadding);
    const showBinLabels = state.histogramLabels === "bins";
    const barColor = getCss("--bar") || "#5bc0de";
    const binOutlineColor = getCss("--histogram-bin-outline") || "#4b5563";
    const lineColor = getCss("--line") || "#d7dde7";
    const textColor = getCss("--text") || "#1f2937";
    const chartPanelColor = getCss("--panel") || "#ffffff";
    const mutedColor = getCss("--muted") || "#6b7280";
    const panelColor = getCss("--panel-2") || "#f3f4f6";
    const dataRows = rows.map((row) => ({
      value: [row.bin_mid, row.height, row.bin_lower, row.bin_upper],
      label: showBinLabels ? formatYAxisValue(row.height, data.y_axis) : "",
      row,
    }));
    const referenceSeries = referenceLineSeries(data, xLog, chartPanelColor);

    chart.setOption(
      {
        animation: false,
        animationDuration: 0,
        animationDurationUpdate: 0,
        stateAnimation: { duration: 0 },
        backgroundColor: "transparent",
        tooltip: {
          trigger: "item",
          formatter: histogramTooltip,
        },
        grid: {
          left: yAxisLayout.gridLeft,
          right: HISTOGRAM_GRID_RIGHT,
          top: showBinLabels ? HISTOGRAM_GRID_TOP_WITH_BIN_LABELS : HISTOGRAM_GRID_TOP,
          bottom: xAxisPolicy.gridBottom,
          containLabel: false,
        },
        xAxis: {
          type: xLog ? "log" : "value",
          name: histogramXAxisTitle(data),
          nameLocation: "middle",
          nameGap: xAxisPolicy.nameGap,
          nameTruncate: {
            maxWidth: axisNameMaxWidth,
            ellipsis: "…",
          },
          min: xBounds.min,
          max: xBounds.max,
          scale: true,
          ...xAxisPolicy.axisOptions,
          axisLabel: { color: textColor, ...xAxisPolicy.axisLabel },
          axisLine: { lineStyle: { color: lineColor } },
          splitLine: { show: false, lineStyle: { color: lineColor } },
          nameTextStyle: { color: textColor, fontSize: 13, fontWeight: 700 },
        },
        dataZoom: [
          { type: "inside", xAxisIndex: 0, filterMode: "none" },
          {
            type: "slider",
            xAxisIndex: 0,
            filterMode: "none",
            height: 18,
            bottom: 12,
            borderColor: lineColor,
            fillerColor: alphaColor(barColor, 0.18),
            backgroundColor: panelColor,
            dataBackground: {
              lineStyle: { color: mutedColor },
              areaStyle: { color: alphaColor(barColor, 0.12) },
            },
            selectedDataBackground: {
              lineStyle: { color: barColor },
              areaStyle: { color: alphaColor(barColor, 0.22) },
            },
            handleStyle: { color: panelColor, borderColor: mutedColor },
            moveHandleStyle: { color: mutedColor },
            textStyle: { color: mutedColor },
          },
        ],
        yAxis: {
          type: yLog ? "log" : "value",
          name: yLabel,
          nameGap: HISTOGRAM_Y_AXIS_NAME_GAP,
          nameTruncate: {
            maxWidth: axisNameMaxWidth,
            ellipsis: "…",
          },
          min: yLog ? yBaseline : 0,
          ...yAxisPolicy,
          axisLabel: {
            color: textColor,
            fontSize: HISTOGRAM_Y_AXIS_FONT_SIZE,
            margin: HISTOGRAM_Y_AXIS_LABEL_MARGIN,
            formatter: (value) => formatYAxisValue(value, data.y_axis),
          },
          axisLine: { lineStyle: { color: lineColor } },
          splitLine: { lineStyle: { color: lineColor } },
          nameTextStyle: { color: textColor, fontSize: 12, fontWeight: 700, align: "left" },
        },
        series: [
          {
            name: yLabel,
            type: "custom",
            coordinateSystem: "cartesian2d",
            dimensions: ["bin_mid", "height", "bin_lower", "bin_upper"],
            encode: { x: 0, y: 1 },
            data: dataRows,
            itemStyle: { color: barColor },
            animation: false,
            renderItem: (params, api) => renderHistogramBar(params, api, {
              yBaseline,
              yLog,
              color: barColor,
              outlineColor: Number(data.bins) <= HISTOGRAM_BIN_OUTLINE_LIMIT ? binOutlineColor : "",
              label: dataRows[params.dataIndex]?.label || "",
              labelColor: textColor,
            }),
          },
          ...referenceSeries,
        ],
      },
      true,
    );
    if (options.scheduleResize !== false) requestAnimationFrame(() => resizeHistogramChart());
  }

  function axisBounds(rows, xLog) {
    const lowers = rows.map((row) => Number(row.bin_lower)).filter((value) => Number.isFinite(value) && (!xLog || value > 0));
    const uppers = rows.map((row) => Number(row.bin_upper)).filter((value) => Number.isFinite(value) && (!xLog || value > 0));
    if (!lowers.length || !uppers.length) return {};
    return { min: Math.min(...lowers), max: Math.max(...uppers) };
  }

  function histogramYAxisPolicy(values, yLog) {
    if (yLog) return {};
    const maximum = Math.max(0, ...values);
    if (!Number.isFinite(maximum) || maximum <= 0) return {};
    const interval = niceHistogramYAxisInterval(maximum / HISTOGRAM_Y_AXIS_TARGET_INTERVALS);
    if (!Number.isFinite(interval) || interval <= 0) return {};
    const tolerance = Math.max(1e-12, Math.abs(maximum) * 1e-12);
    return {
      interval,
      max: Math.ceil((maximum - tolerance) / interval) * interval,
    };
  }

  function niceHistogramYAxisInterval(rawInterval) {
    const value = Math.max(Number.MIN_VALUE, Number(rawInterval) || 0);
    const magnitude = 10 ** Math.floor(Math.log10(value));
    const normalised = value / magnitude;
    const multiplier = normalised <= 1.5 ? 1 : (normalised <= 3.5 ? 2 : (normalised <= 7.5 ? 5 : 10));
    return multiplier * magnitude;
  }

  function histogramYAxisLayout(values, yLog, yBaseline, policy, yAxis) {
    const labelWidth = histogramYAxisTickCandidates(values, yLog, yBaseline, policy)
      .map((value) => formatYAxisValue(value, yAxis))
      .reduce((maximum, label) => Math.max(maximum, measureHistogramYAxisLabel(label)), 0);
    return {
      gridLeft: Math.max(
        HISTOGRAM_GRID_LEFT_MIN,
        Math.ceil(labelWidth + HISTOGRAM_Y_AXIS_LABEL_MARGIN + HISTOGRAM_Y_AXIS_OUTER_PADDING),
      ),
    };
  }

  function histogramYAxisTickCandidates(values, yLog, yBaseline, policy) {
    const finiteValues = (Array.isArray(values) ? values : [])
      .map(Number)
      .filter((value) => Number.isFinite(value) && value > 0);
    if (!yLog) {
      const interval = Number(policy?.interval);
      const maximum = Number(policy?.max);
      if (!Number.isFinite(interval) || interval <= 0 || !Number.isFinite(maximum) || maximum <= 0) {
        return [0, ...finiteValues];
      }
      const tickCount = Math.max(1, Math.ceil(maximum / interval));
      return Array.from({ length: tickCount + 1 }, (_unused, index) => (
        index === tickCount ? maximum : index * interval
      ));
    }
    const positiveBaseline = Number(yBaseline);
    const candidates = [
      ...(Number.isFinite(positiveBaseline) && positiveBaseline > 0 ? [positiveBaseline] : []),
      ...finiteValues,
    ];
    if (!candidates.length) return [1];
    const minimum = Math.min(...candidates);
    const maximum = Math.max(...candidates);
    const minimumExponent = Math.floor(Math.log10(minimum));
    const maximumExponent = Math.ceil(Math.log10(maximum));
    for (let exponent = minimumExponent; exponent <= maximumExponent; exponent += 1) {
      const value = 10 ** exponent;
      if (Number.isFinite(value) && value > 0) candidates.push(value);
    }
    return candidates;
  }

  function measureHistogramYAxisLabel(label) {
    const text = String(label || "");
    const measured = echartsImpl.format?.getTextRect?.(
      text,
      `${HISTOGRAM_Y_AXIS_FONT_SIZE}px sans-serif`,
    )?.width;
    return Number.isFinite(Number(measured))
      ? Number(measured)
      : text.length * HISTOGRAM_Y_AXIS_FONT_SIZE * HISTOGRAM_Y_AXIS_LABEL_WIDTH_FACTOR;
  }

  function histogramXAxisPolicy(
    data,
    _rows,
    xLog,
    chartWidth,
    horizontalPadding,
    formatContinuousValue,
  ) {
    const axisLabel = {
      formatter: (value) => formatHistogramXAxisValue(value, data?.binning, formatContinuousValue),
    };
    const plotWidth = Math.max(
      120,
      (Number(chartWidth) || 800) - Math.max(0, Number(horizontalPadding) || 0),
    );
    if (xLog || data?.binning?.mode !== "integer") {
      const samples = [data?.binning?.min, data?.binning?.max]
        .map((value) => formatContinuousValue(value))
        .filter(Boolean);
      const labelWidth = estimatedHistogramAxisLabelWidth(samples);
      const targetLabels = targetHistogramXAxisLabelCount(plotWidth, labelWidth);
      return {
        axisOptions: { splitNumber: targetLabels },
        axisLabel: { ...axisLabel, hideOverlap: true },
        nameGap: 34,
        gridBottom: 92,
      };
    }
    const minimum = Number(data.binning.min);
    const maximum = Number(data.binning.max);
    const range = maximum - minimum;
    if (!Number.isFinite(range) || range <= 0) {
      return {
        axisOptions: { minInterval: 1, splitNumber: 1 },
        axisLabel: { ...axisLabel, hideOverlap: true },
        nameGap: 34,
        gridBottom: 92,
      };
    }
    const widestLabel = estimatedHistogramAxisLabelWidth([
      formatHistogramXAxisValue(minimum, data?.binning, formatContinuousValue),
      formatHistogramXAxisValue(maximum, data?.binning, formatContinuousValue),
    ]);
    const integerLevelCount = Math.floor(range) + 1;
    const horizontalTarget = targetHistogramXAxisLabelCount(plotWidth, widestLabel);
    const rawTextWidth = Math.max(1, widestLabel - HISTOGRAM_X_AXIS_LABEL_PADDING);
    const radians = (HISTOGRAM_X_AXIS_ROTATION * Math.PI) / 180;
    const rotatedFootprint = rawTextWidth * Math.cos(radians)
      + HISTOGRAM_X_AXIS_FONT_SIZE * Math.sin(radians)
      + HISTOGRAM_X_AXIS_LABEL_PADDING;
    const rotatedTarget = targetHistogramXAxisLabelCount(plotWidth, rotatedFootprint);
    const useRotatedLabels = integerLevelCount > horizontalTarget && rotatedTarget > horizontalTarget;
    const targetLabels = useRotatedLabels ? rotatedTarget : horizontalTarget;
    const step = integerLevelCount <= targetLabels
      ? 1
      : niceIntegerAxisStep(range / Math.max(1, targetLabels - 1));
    const rotatedHeight = useRotatedLabels
      ? rawTextWidth * Math.sin(radians) + HISTOGRAM_X_AXIS_FONT_SIZE * Math.cos(radians)
      : 0;
    const nameGap = useRotatedLabels ? Math.max(34, Math.ceil(rotatedHeight) + 12) : 34;
    return {
      axisOptions: {
        minInterval: step,
        maxInterval: step,
        splitNumber: targetLabels,
      },
      axisLabel: {
        ...axisLabel,
        rotate: useRotatedLabels ? HISTOGRAM_X_AXIS_ROTATION : 0,
        hideOverlap: false,
      },
      nameGap,
      gridBottom: 58 + nameGap,
    };
  }

  function estimatedHistogramAxisLabelWidth(labels) {
    const maximumLength = labels.reduce((longest, label) => Math.max(longest, String(label || "").length), 1);
    return Math.max(
      14,
      maximumLength * HISTOGRAM_X_AXIS_FONT_SIZE * HISTOGRAM_X_AXIS_LABEL_WIDTH_FACTOR
        + HISTOGRAM_X_AXIS_LABEL_PADDING,
    );
  }

  function targetHistogramXAxisLabelCount(plotWidth, labelWidth) {
    const fitted = Math.floor(plotWidth / Math.max(1, labelWidth));
    return Math.max(HISTOGRAM_X_AXIS_MIN_LABELS, Math.min(HISTOGRAM_X_AXIS_MAX_LABELS, fitted));
  }

  function niceIntegerAxisStep(rawStep) {
    const value = Math.max(1, Number(rawStep) || 1);
    const exponent = Math.floor(Math.log10(value));
    const base = 10 ** exponent;
    const normalized = value / base;
    const multiplier = normalized <= 1 ? 1 : normalized <= 2 ? 2 : normalized <= 5 ? 5 : 10;
    return Math.max(1, multiplier * base);
  }

  function formatHistogramXAxisValue(value, binning, formatContinuousValue) {
    if (binning?.mode !== "integer") return formatContinuousValue(value);
    const number = Number(value);
    if (!Number.isFinite(number)) return "";
    const rounded = Math.round(number);
    if (Math.abs(number - rounded) > 1e-9) return "";
    return rounded.toLocaleString(undefined, { maximumFractionDigits: 0 });
  }

  function renderHistogramBar(_params, api, options) {
    const { yBaseline, yLog, color, outlineColor, label, labelColor } = options;
    const lower = Number(api.value(2));
    const upper = Number(api.value(3));
    const height = Number(api.value(1));
    if (!Number.isFinite(lower) || !Number.isFinite(upper) || !Number.isFinite(height)) return null;
    if (height <= 0 || (yLog && height <= 0)) return null;
    const start = api.coord([lower, yLog ? yBaseline : 0]);
    const end = api.coord([upper, height]);
    const leftPx = Math.min(start[0], end[0]);
    const rightPx = Math.max(start[0], end[0]);
    const topPx = Math.min(start[1], end[1]);
    const bottomPx = Math.max(start[1], end[1]);
    const shape = echartsImpl.graphic.clipRectByRect(
      {
        x: leftPx,
        y: topPx,
        width: Math.max(0.5, rightPx - leftPx),
        height: Math.max(0.5, bottomPx - topPx),
      },
      {
        x: _params.coordSys.x,
        y: _params.coordSys.y,
        width: _params.coordSys.width,
        height: _params.coordSys.height,
      },
    );
    if (!shape) return null;
    const children = [{
      type: "rect",
      shape,
      style: {
        fill: color,
        stroke: outlineColor || null,
        lineWidth: outlineColor ? 0.5 : 0,
      },
    }];
    const labelFontSize = histogramBinLabelFontSize(label, shape.width);
    if (labelFontSize) {
      children.push({
        type: "text",
        x: shape.x + shape.width / 2,
        y: shape.y - 3,
        style: {
          text: label,
          fill: labelColor,
          font: `${labelFontSize}px sans-serif`,
          align: "center",
          verticalAlign: "bottom",
        },
        silent: true,
      });
    }
    return { type: "group", children };
  }

  function histogramBinLabelFontSize(label, binWidth) {
    const text = String(label || "");
    if (!text || !Number.isFinite(binWidth) || binWidth <= 0) return 0;
    const availableWidth = Math.max(0, binWidth - 4);
    const fitted = Math.floor(availableWidth / (text.length * HISTOGRAM_BIN_LABEL_WIDTH_FACTOR));
    return Math.max(
      HISTOGRAM_BIN_LABEL_MIN_FONT_SIZE,
      Math.min(HISTOGRAM_BIN_LABEL_MAX_FONT_SIZE, fitted),
    );
  }

  function referenceLineSeries(data, xLog, labelBackgroundColor) {
    const stats = data.stats || [];
    const mean = stats.find((row) => row.statistic === "Mean")?.value;
    const median = stats.find((row) => row.statistic === "Median")?.value;
    const meanColor = getCss("--histogram-mean-color") || DEFAULT_HISTOGRAM_MEAN_COLOR;
    const medianColor = getCss("--histogram-median-color") || DEFAULT_HISTOGRAM_MEDIAN_COLOR;
    return [
      referenceLine("Mean", mean, meanColor, xLog),
      referenceLine(
        "Median",
        median,
        medianColor,
        xLog,
        HISTOGRAM_MEDIAN_LABEL_OFFSET,
        labelBackgroundColor,
      ),
    ].filter(Boolean);
  }

  function referenceLine(name, rawValue, color, xLog, labelOffset = 0, labelBackgroundColor = "") {
    const value = Number(rawValue);
    if (!Number.isFinite(value) || (xLog && value <= 0)) return null;
    const formattedValue = formatMetricValue(value, name);
    return {
      name,
      type: "line",
      data: [],
      silent: true,
      animation: false,
      markLine: {
        symbol: "none",
        lineStyle: { color, width: 1.5, type: "dashed" },
        label: {
          color,
          formatter: formattedValue ? `${name} ${formattedValue}` : name,
          fontSize: 11,
          offset: [0, labelOffset],
          backgroundColor: labelBackgroundColor || "transparent",
          padding: labelBackgroundColor ? [0, 2, 4, 2] : 0,
        },
        data: [{ xAxis: value }],
      },
    };
  }

  function yAxisLabel(data) {
    const base = data.y_axis === "probability" ? "Probability" : (data.denominator?.bar_label || "Weight");
    return data.distribution === "cumulative" ? `Cumulative ${base}` : base;
  }

  function histogramXAxisTitle(data) {
    const responseLabel = String(data.response?.label || data.actual || "Actual").trim() || "Actual";
    const numerator = String(data.response?.numerator || data.actual || responseLabel).trim() || responseLabel;
    const denominator = String(data.denominator?.column || "").trim();
    return denominator ? `${numerator} / ${denominator}` : responseLabel;
  }

  function histogramTooltip(params) {
    const row = params?.data?.row || {};
    const heightLabel = yAxisLabel(state.lastHistogramData || {});
    return [
      escapeHtml(row.bin_label || ""),
      `${params.marker || ""}${escapeHtml(heightLabel)}: ${escapeHtml(formatYAxisValue(row.height, state.lastHistogramData?.y_axis))}`,
      `Rows: ${escapeHtml(formatNumber(row.row_count))}`,
      `Volume: ${escapeHtml(formatWeightValue(row.volume))}`,
      `Probability: ${escapeHtml(formatPercent(row.probability))}`,
    ].join("<br/>");
  }

  function formatAxisValue(value) {
    return formatLineValue(value);
  }

  function formatYAxisValue(value, yAxis) {
    return yAxis === "probability" ? formatPercent(value) : formatWeightValue(value);
  }

  function formatPercent(value) {
    const number = Number(value);
    if (!Number.isFinite(number)) return "";
    const percent = number * 100;
    const abs = Math.abs(percent);
    const digits = abs === 0 || abs >= 10 ? 1 : abs >= 1 ? 2 : 3;
    return `${percent.toLocaleString(undefined, { maximumFractionDigits: digits })}%`;
  }

  function alphaColor(color, alpha) {
    const text = String(color || "").trim();
    const rgb = text.match(/^rgba?\(([^)]+)\)$/i);
    if (rgb) {
      const parts = rgb[1].match(/[\d.]+/g) || [];
      if (parts.length >= 3) return `rgba(${parts[0]}, ${parts[1]}, ${parts[2]}, ${alpha})`;
    }
    const hex = text.match(/^#([0-9a-f]{3}|[0-9a-f]{6})$/i);
    if (hex) {
      const raw = hex[1].length === 3
        ? hex[1].split("").map((part) => part + part).join("")
        : hex[1];
      const r = parseInt(raw.slice(0, 2), 16);
      const g = parseInt(raw.slice(2, 4), 16);
      const b = parseInt(raw.slice(4, 6), 16);
      return `rgba(${r}, ${g}, ${b}, ${alpha})`;
    }
    return `rgba(91, 192, 222, ${alpha})`;
  }

  function setSegmentedValue(control, value) {
    state[control] = value;
    syncSegmented(control, value);
  }

  function bindControls() {
    bindSettingsStripOverflowCue(el("histogramToolbar"));
    bindHistogramSplitResize();
    ["histogramDistribution", "histogramYAxis", "histogramLogScale", "histogramSampleMode"].forEach((control) => {
      const group = document.querySelector(`.segmented[data-control="${control}"]`);
      if (!group) return;
      group.addEventListener("click", (event) => {
        if (event.target.tagName !== "BUTTON") return;
        const previousValue = state[control];
        setSegmentedValue(control, event.target.dataset.value);
        if (state[control] !== previousValue) clearActiveFavouriteSelection();
        refreshHistogram();
      });
    });
    document.querySelector('.segmented[data-control="histogramBinMode"]')?.addEventListener("click", (event) => {
      if (event.target.tagName !== "BUTTON") return;
      captureHistogramBinInput();
      const nextMode = event.target.dataset.value === "width" ? "width" : "count";
      if (nextMode === histogramBinMode()) return;
      if (nextMode === "width" && !histogramBinWidthValue()) {
        histogramBinValues.width = suggestedHistogramBinWidth();
      }
      setSegmentedValue("histogramBinMode", nextMode);
      syncHistogramBinInput();
      clearActiveFavouriteSelection();
      refreshHistogram();
    });
    document.querySelector('.segmented[data-control="histogramLabels"]')?.addEventListener("click", (event) => {
      if (event.target.tagName !== "BUTTON") return;
      const previousValue = state.histogramLabels;
      setSegmentedValue("histogramLabels", event.target.dataset.value === "bins" ? "bins" : "none");
      if (state.histogramLabels === previousValue) return;
      clearActiveFavouriteSelection();
      if (state.lastHistogramData) measureToolRender("histogram", () => renderChart(state.lastHistogramData));
    });
    el("histogramBins").addEventListener("input", () => {
      captureHistogramBinInput();
      clearActiveFavouriteSelection();
      if (!histogramBinInputIsValid()) {
        cancelHistogramBinsRefresh();
        return;
      }
      scheduleHistogramBinsRefresh();
    });
    el("histogramBins").addEventListener("keydown", (event) => {
      if (event.key !== "Enter") return;
      event.preventDefault();
      captureHistogramBinInput();
      if (!histogramBinInputIsValid({ report: true })) return;
      scheduleHistogramBinsRefresh({ immediate: true });
    });
    el("histogramBins").addEventListener("blur", () => {
      captureHistogramBinInput();
      if (!histogramBinInputIsValid({ report: true })) return;
      scheduleHistogramBinsRefresh({ immediate: true });
    });
    syncHistogramBinInput();
  }

  function bindHistogramSplitResize() {
    const resizer = el("histogramSplitResizer");
    if (!resizer) return;
    let dragging = false;
    let startX = 0;
    let startWidth = histogramStatsWidth;

    resizer.addEventListener("pointerdown", (event) => {
      if (window.matchMedia(HISTOGRAM_STACKED_MEDIA).matches) return;
      event.preventDefault();
      dragging = true;
      startX = event.clientX;
      startWidth = histogramStatsWidth;
      resizer.classList.add("dragging");
      document.body.classList.add("resizing-chart-controls");
      resizer.setPointerCapture(event.pointerId);
      window.getSelection()?.removeAllRanges();
    });
    resizer.addEventListener("pointermove", (event) => {
      if (!dragging) return;
      event.preventDefault();
      setHistogramStatsWidth(startWidth - (event.clientX - startX));
    });
    const finishDrag = (event) => {
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
      scheduleHistogramChartResize();
    };
    resizer.addEventListener("pointerup", finishDrag);
    resizer.addEventListener("pointercancel", finishDrag);
    resizer.addEventListener("keydown", (event) => {
      if (window.matchMedia(HISTOGRAM_STACKED_MEDIA).matches) return;
      const bounds = histogramStatsWidthBounds();
      let nextWidth = null;
      if (event.key === "ArrowLeft") nextWidth = histogramStatsWidth + HISTOGRAM_SPLITTER_KEY_STEP;
      if (event.key === "ArrowRight") nextWidth = histogramStatsWidth - HISTOGRAM_SPLITTER_KEY_STEP;
      if (event.key === "Home") nextWidth = bounds.min;
      if (event.key === "End") nextWidth = bounds.max;
      if (nextWidth === null) return;
      event.preventDefault();
      setHistogramStatsWidth(nextWidth);
    });
    setHistogramStatsWidth(histogramStatsWidth, { resize: false });
  }

  function histogramStatsWidthBounds() {
    const availableWidth = el("histogramWrap")?.getBoundingClientRect().width || window.innerWidth;
    const max = Math.max(
      HISTOGRAM_STATS_MIN_WIDTH,
      Math.min(
        HISTOGRAM_STATS_MAX_WIDTH,
        availableWidth - HISTOGRAM_CHART_MIN_WIDTH,
      ),
    );
    return { min: HISTOGRAM_STATS_MIN_WIDTH, max };
  }

  function setHistogramStatsWidth(rawWidth, { resize = true } = {}) {
    const bounds = histogramStatsWidthBounds();
    histogramStatsWidth = Math.min(Math.max(Number(rawWidth) || HISTOGRAM_STATS_DEFAULT_WIDTH, bounds.min), bounds.max);
    document.documentElement.style.setProperty("--histogram-stats-width", `${Math.round(histogramStatsWidth)}px`);
    const resizer = el("histogramSplitResizer");
    resizer?.setAttribute("aria-valuemin", String(Math.round(bounds.min)));
    resizer?.setAttribute("aria-valuemax", String(Math.round(bounds.max)));
    resizer?.setAttribute("aria-valuenow", String(Math.round(histogramStatsWidth)));
    if (resize) scheduleHistogramChartResize();
    return histogramStatsWidth;
  }

  function redrawStatsLayoutFallback() {
    if (typeof ResizeObserver !== "function") statsTable?.redraw?.(true);
  }

  function resizeHistogramChart({ flush = false } = {}) {
    chart.resize();
    const resizedWidth = Number(chart.getWidth?.()) || 0;
    if (state.lastHistogramData && Math.abs(resizedWidth - histogramAxisChartWidth) >= 1) {
      renderChart(state.lastHistogramData, { scheduleResize: false });
    }
    if (flush) chart.getZr?.().flush?.();
    redrawStatsLayoutFallback();
  }

  function scheduleHistogramChartResize() {
    if (histogramChartResizeFrame !== null) return;
    histogramChartResizeFrame = requestAnimationFrame(() => {
      histogramChartResizeFrame = null;
      resizeHistogramChart();
    });
  }

  function refreshHistogram(options = {}) {
    if (state.tool !== "histogram") return;
    return refreshActiveTool(options);
  }

  function scheduleHistogramBinsRefresh(options = {}) {
    cancelHistogramBinsRefresh();
    if (options.immediate) {
      return refreshHistogram();
    }
    histogramBinsRefreshTimer = window.setTimeout(() => {
      histogramBinsRefreshTimer = null;
      refreshHistogram();
    }, HISTOGRAM_BINS_REFRESH_DELAY_MS);
  }

  function cancelHistogramBinsRefresh() {
    if (!histogramBinsRefreshTimer) return;
    window.clearTimeout(histogramBinsRefreshTimer);
    histogramBinsRefreshTimer = null;
  }

  function activate() {
    syncSegmented("histogramBinMode", state.histogramBinMode);
    syncSegmented("histogramDistribution", state.histogramDistribution);
    syncSegmented("histogramYAxis", state.histogramYAxis);
    syncSegmented("histogramLabels", state.histogramLabels);
    syncSegmented("histogramLogScale", state.histogramLogScale);
    syncSegmented("histogramSampleMode", state.histogramSampleMode);
    syncHistogramBinInput();
    requestAnimationFrame(resize);
  }

  function resize() {
    if (!window.matchMedia(HISTOGRAM_STACKED_MEDIA).matches) {
      setHistogramStatsWidth(histogramStatsWidth, { resize: false });
    }
    resizeHistogramChart({ flush: true });
  }

  function refreshTheme() {
    if (state.lastHistogramData) {
      renderChart(state.lastHistogramData);
      statsTable?.redraw?.(true);
    }
  }

  return {
    buildRequest: buildHistogramRequest,
    fetchData: fetchHistogramData,
    useCached: useCachedHistogramData,
    render: renderHistogramData,
    bindControls,
    activate,
    resize,
    refreshTheme,
    captureFavouriteState,
    applyFavouriteState,
  };
}
