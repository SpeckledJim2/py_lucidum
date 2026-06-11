import { loadTabulator } from "./shared/tabulator.js";

const HISTOGRAM_BINS_REFRESH_DELAY_MS = 250;

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
  setFilterRowMeta,
  setGroupMeta,
  applyToolPresentation,
  saveToolPresentation,
  toolCache,
  renderMetricTitle,
  getCss,
  refreshActiveTool,
}) {
  const chart = echartsImpl.init(el("histogramChart"));
  let statsTable = null;
  let statsRenderToken = 0;
  let histogramBinsRefreshTimer = null;

  function syncSegmented(control, value) {
    const group = document.querySelector(`.segmented[data-control="${control}"]`);
    if (!group) return;
    group.querySelectorAll("button").forEach((button) => {
      button.classList.toggle("active", button.dataset.value === value);
    });
  }

  function histogramBinsValue() {
    const raw = String(el("histogramBins")?.value || "").trim();
    return raw || "auto";
  }

  function buildHistogramRequest() {
    if (!state.schema || !el("actualNumerator")?.value) return null;
    return {
      source: state.source || "dataset",
      actual: el("actualNumerator").value,
      denominator: el("denominator").value,
      bins: histogramBinsValue(),
      distribution: state.histogramDistribution || "incremental",
      yAxis: state.histogramYAxis || "sum",
      logScale: state.histogramLogScale || "none",
      sampleMode: state.histogramSampleMode || "100k",
      filter: state.activeFilter,
    };
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
    updateMetricTitles(data);
    renderChart(data);
    renderStatsTable(data);
    const rowMeta = formatRowMeta(data.row_count, data.filtered_row_count);
    const valid = formatNumber(data.valid_count);
    const sampled = data.sampled_valid_count && data.sampled_valid_count !== data.valid_count
      ? ` - ${formatNumber(data.sampled_valid_count)} sampled`
      : "";
    const groupMeta = `${formatNumber(data.bins)} bins - ${valid} valid${sampled} - ${rowMeta}`;
    const warnings = [...(data.warnings || [])].filter(Boolean).join(" ");
    setFilterRowMeta(data.row_count, data.filtered_row_count);
    setGroupMeta("histogram", groupMeta);
    setStatus("");
    setChartMessage(warnings);
    saveToolPresentation("histogram", { groupMeta, chartMessage: warnings });
  }

  function useCachedHistogramData(cache, options = {}) {
    state.lastHistogramData = cache.data;
    if (options.renderIfCached) {
      measureToolRender("histogram", () => renderHistogramData(cache.data));
      return;
    }
    measureToolRender("histogram", () => {
      updateMetricTitles(cache.data);
      applyToolPresentation("histogram");
      requestAnimationFrame(() => {
        chart.resize();
        statsTable?.redraw?.(true);
      });
    });
  }

  function updateMetricTitles(data) {
    const meanRow = (data.stats || []).find((row) => row.statistic === "Mean");
    renderMetricTitle(el("actualMetricTitle"), "Actual", meanRow?.value);
    renderMetricTitle(el("weightMetricTitle"), "Weight", data.denominator?.value, formatWeightValue);
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

  function renderChart(data) {
    const rows = Array.isArray(data.rows) ? data.rows : [];
    const xLog = data.log_scale === "x" || data.log_scale === "both";
    const yLog = data.log_scale === "y" || data.log_scale === "both";
    const yLabel = yAxisLabel(data);
    const yValues = rows.map((row) => Number(row.height)).filter((value) => Number.isFinite(value) && value > 0);
    const yBaseline = yLog ? (yValues.length ? Math.max(Math.min(...yValues) / 10, 1e-12) : 1e-12) : 0;
    const xBounds = axisBounds(rows, xLog);
    const barColor = getCss("--bar") || "#5bc0de";
    const lineColor = getCss("--line") || "#d7dde7";
    const textColor = getCss("--text") || "#1f2937";
    const mutedColor = getCss("--muted") || "#6b7280";
    const panelColor = getCss("--panel-2") || "#f3f4f6";
    const dataRows = rows.map((row) => ({
      value: [row.bin_mid, row.height, row.bin_lower, row.bin_upper],
      row,
    }));
    const referenceSeries = referenceLineSeries(data, xLog);

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
        grid: { left: 72, right: 30, top: 42, bottom: 92, containLabel: false },
        xAxis: {
          type: xLog ? "log" : "value",
          name: data.response?.label || data.actual || "Actual",
          nameLocation: "middle",
          nameGap: 34,
          min: xBounds.min,
          max: xBounds.max,
          scale: true,
          axisLabel: { color: textColor, formatter: (value) => formatAxisValue(value) },
          axisLine: { lineStyle: { color: lineColor } },
          splitLine: { lineStyle: { color: lineColor } },
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
          min: yLog ? yBaseline : 0,
          axisLabel: { color: textColor, formatter: (value) => formatYAxisValue(value, data.y_axis) },
          axisLine: { lineStyle: { color: lineColor } },
          splitLine: { lineStyle: { color: lineColor } },
          nameTextStyle: { color: textColor, fontSize: 12, fontWeight: 700 },
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
            renderItem: (params, api) => renderHistogramBar(params, api, yBaseline, yLog, barColor),
          },
          ...referenceSeries,
        ],
      },
      true,
    );
    requestAnimationFrame(() => chart.resize());
  }

  function axisBounds(rows, xLog) {
    const lowers = rows.map((row) => Number(row.bin_lower)).filter((value) => Number.isFinite(value) && (!xLog || value > 0));
    const uppers = rows.map((row) => Number(row.bin_upper)).filter((value) => Number.isFinite(value) && (!xLog || value > 0));
    if (!lowers.length || !uppers.length) return {};
    return { min: Math.min(...lowers), max: Math.max(...uppers) };
  }

  function renderHistogramBar(_params, api, yBaseline, yLog, color) {
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
    const x = Math.floor(leftPx);
    const y = Math.floor(topPx);
    const width = Math.max(1, Math.ceil(rightPx) - x);
    const barHeight = Math.max(1, Math.ceil(bottomPx) - y);
    const shape = echartsImpl.graphic.clipRectByRect(
      { x, y, width, height: barHeight },
      {
        x: _params.coordSys.x,
        y: _params.coordSys.y,
        width: _params.coordSys.width,
        height: _params.coordSys.height,
      },
    );
    return shape ? { type: "rect", shape, style: { fill: color } } : null;
  }

  function referenceLineSeries(data, xLog) {
    const stats = data.stats || [];
    const mean = stats.find((row) => row.statistic === "Mean")?.value;
    const median = stats.find((row) => row.statistic === "Median")?.value;
    return [
      referenceLine("Mean", mean, "#d13f3f", xLog),
      referenceLine("Median", median, "#1f7a8c", xLog),
    ].filter(Boolean);
  }

  function referenceLine(name, rawValue, color, xLog) {
    const value = Number(rawValue);
    if (!Number.isFinite(value) || (xLog && value <= 0)) return null;
    return {
      name,
      type: "line",
      data: [],
      silent: true,
      animation: false,
      markLine: {
        symbol: "none",
        lineStyle: { color, width: 1.5, type: "dashed" },
        label: { color, formatter: name, fontSize: 11 },
        data: [{ xAxis: value }],
      },
    };
  }

  function yAxisLabel(data) {
    const base = data.y_axis === "probability" ? "Probability" : (data.denominator?.bar_label || "Weight");
    return data.distribution === "cumulative" ? `Cumulative ${base}` : base;
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
    ["histogramDistribution", "histogramYAxis", "histogramLogScale", "histogramSampleMode"].forEach((control) => {
      const group = document.querySelector(`.segmented[data-control="${control}"]`);
      if (!group) return;
      group.addEventListener("click", (event) => {
        if (event.target.tagName !== "BUTTON") return;
        setSegmentedValue(control, event.target.dataset.value);
        refreshHistogram();
      });
    });
    el("histogramBins").addEventListener("input", () => scheduleHistogramBinsRefresh());
    el("histogramBins").addEventListener("keydown", (event) => {
      if (event.key !== "Enter") return;
      event.preventDefault();
      scheduleHistogramBinsRefresh({ immediate: true });
    });
    el("histogramBins").addEventListener("blur", () => scheduleHistogramBinsRefresh({ immediate: true }));
  }

  function refreshHistogram(options = {}) {
    if (state.tool !== "histogram") return;
    return refreshActiveTool(options);
  }

  function scheduleHistogramBinsRefresh(options = {}) {
    if (histogramBinsRefreshTimer) {
      window.clearTimeout(histogramBinsRefreshTimer);
      histogramBinsRefreshTimer = null;
    }
    if (options.immediate) {
      return refreshHistogram();
    }
    histogramBinsRefreshTimer = window.setTimeout(() => {
      histogramBinsRefreshTimer = null;
      refreshHistogram();
    }, HISTOGRAM_BINS_REFRESH_DELAY_MS);
  }

  function activate() {
    syncSegmented("histogramDistribution", state.histogramDistribution);
    syncSegmented("histogramYAxis", state.histogramYAxis);
    syncSegmented("histogramLogScale", state.histogramLogScale);
    syncSegmented("histogramSampleMode", state.histogramSampleMode);
    requestAnimationFrame(resize);
  }

  function resize() {
    chart.resize();
    statsTable?.redraw?.(true);
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
  };
}
