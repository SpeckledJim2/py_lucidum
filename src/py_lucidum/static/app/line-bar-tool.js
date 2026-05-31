export function createLineBarTool({
  api,
  el,
  state,
  echartsImpl,
  escapeHtml,
  isModelPredictionColumn,
  formatNumber,
  formatChartLabel,
  formatLineLabel,
  formatLineValue,
  formatWeightValue,
  formatXLabel,
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
  sourceColumns,
  selectedColumn,
  numericColumns,
  dataSourceForId,
  dataSourceHasColumn,
  toolEnabled,
  setTool,
  renderMetricTitle,
  getCss,
  bandSteps,
  refreshLineBar,
}) {
  const TABLE_PAGE_SIZE = 1000;
  const LABEL_DENSITY_LIMIT = 200;
  const DATE_AXIS_TARGET_LABELS = 12;
  const DATE_AXIS_MIN_MONTH_LABELS = 2;
  const DATE_AXIS_MAX_MONTH_LABELS = 14;
  const DATE_AXIS_MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];
  const RESPONSE_AXIS_PADDING = 0.08;
  const RESPONSE_AXIS_TARGET_INTERVALS = 15;
  const chart = echartsImpl.init(el("chart"));

  function isNumericKind(kind) {
    return kind === "numeric" || kind === "integer";
  }

  function isDateKind(kind) {
    return kind === "date" || kind === "datetime";
  }

  function lineBarFeatureTargetSource(featureName) {
    if (!lineBarToolAvailable()) return "";
    const name = String(featureName || "");
    if (!name) return "";
    const currentSource = state.source || "dataset";
    if (dataSourceHasColumn(currentSource, name)) return currentSource;
    return "dataset";
  }

  function lineBarToolAvailable() {
    if (toolEnabled("line_bar")) return true;
    const button = el("lineBarTool");
    return Boolean(button && !button.disabled && !button.classList.contains("hidden"));
  }

  function canNavigateToLineBarFeature(featureName) {
    return Boolean(lineBarFeatureTargetSource(featureName));
  }

  function navigateToLineBarFeature(featureName) {
    const name = String(featureName || "");
    const targetSource = lineBarFeatureTargetSource(name);
    if (!targetSource) return false;
    state.source = targetSource;
    state.x = name;
    state.bandFeature = null;
    renderFeatures();
    updateAxisControls();
    setTool("line_bar");
    return state.tool === "line_bar";
  }

  function expectedDisplayColumns() {
    const columns = [...numericColumns()];
    if (state.expectedSort === "alpha") {
      columns.sort((a, b) => a.name.localeCompare(b.name, undefined, { sensitivity: "base" }));
    }
    const predictionColumns = columns.filter(isModelPredictionColumn);
    const otherColumns = columns.filter((column) => !isModelPredictionColumn(column));
    return [...predictionColumns, ...otherColumns];
  }

  function syncSegmented(control, value) {
    const group = document.querySelector(`.segmented[data-control="${control}"]`);
    if (!group) return;
    group.querySelectorAll("button").forEach((button) => {
      button.classList.toggle("active", button.dataset.value === value);
    });
  }

  function formatBandWidth(value) {
    const number = Number(value);
    if (!Number.isFinite(number) || number <= 0) return "0";
    return Number(number.toPrecision(12)).toString();
  }

  function syncBandingControl() {
    syncSegmented("bandWidth", state.bandWidth);
    el("bandLabel").textContent = state.quantileMode === "quantile" ? "Quantiles" : "Banding";
    const display = Number(state.bandWidth) > 0 ? state.bandWidth : "auto off";
    el("bandValue").textContent = `(${display})`;
  }

  function quantileCountForBandWidth(value = state.bandWidth) {
    const number = Number(value);
    if (!Number.isFinite(number) || number <= 0) return 1;
    return Math.min(1000, Math.max(1, Math.round(number)));
  }

  function syncQuantileControl() {
    syncSegmented("quantileMode", state.quantileMode);
  }

  function normalizeBandWidthForQuantiles() {
    state.bandWidth = String(quantileCountForBandWidth());
    state.bandFeature = currentBandFeatureKey();
    syncBandingControl();
  }

  function currentBandFeatureKey() {
    return JSON.stringify([state.source || "dataset", state.x || ""]);
  }

  function fallbackBandWidthForSelectedColumn() {
    return isNumericKind(selectedColumn()?.kind) ? "1" : "0";
  }

  function clearPendingBandSuggestion() {
    state.bandSuggestionPendingKey = null;
  }

  async function requestBandSuggestionForSelectedColumn(bandFeatureKey = currentBandFeatureKey()) {
    if (!state.schema || !state.x || state.bandSuggestionPendingKey === bandFeatureKey) return;
    const column = selectedColumn();
    if (!isNumericKind(column?.kind)) return;
    const requestSeq = (state.bandSuggestionRequestSeq || 0) + 1;
    state.bandSuggestionRequestSeq = requestSeq;
    state.bandSuggestionPendingKey = bandFeatureKey;
    syncBandingControl();
    try {
      const data = await api("/api/banding/suggestion", {
        method: "POST",
        body: JSON.stringify({
          source: state.source || "dataset",
          feature: state.x,
          filter: state.activeFilter,
        }),
      });
      if (requestSeq !== state.bandSuggestionRequestSeq || state.bandSuggestionPendingKey !== bandFeatureKey) return;
      if (currentBandFeatureKey() !== bandFeatureKey) return;
      const formatted = formatBandWidth(data.band_suggestion);
      state.bandWidth = formatted === "0" ? fallbackBandWidthForSelectedColumn() : formatted;
      state.bandFeature = bandFeatureKey;
      clearPendingBandSuggestion();
      if (state.quantileMode === "quantile") {
        normalizeBandWidthForQuantiles();
      } else {
        syncBandingControl();
      }
      if (state.tool === "line_bar") refreshChart({ force: true });
    } catch (error) {
      if (requestSeq !== state.bandSuggestionRequestSeq || state.bandSuggestionPendingKey !== bandFeatureKey) return;
      if (currentBandFeatureKey() !== bandFeatureKey) return;
      state.bandWidth = fallbackBandWidthForSelectedColumn();
      state.bandFeature = bandFeatureKey;
      clearPendingBandSuggestion();
      syncBandingControl();
      const warning = `Banding estimate failed; using ${state.bandWidth}. ${error.message}`;
      if (state.tool === "line_bar") {
        refreshChart({ force: true }).then(() => setStatus(warning, false));
      } else {
        setStatus(warning, false);
      }
    }
  }

  function stepBandWidth(direction) {
    clearPendingBandSuggestion();
    const current = Number(state.bandWidth) > 0 ? Number(state.bandWidth) : Number(fallbackBandWidthForSelectedColumn()) || 1;
    let next = current;
    if (direction < 0) {
      const smallerSteps = bandSteps.filter((step) => step < current);
      next = smallerSteps.length ? smallerSteps[smallerSteps.length - 1] : current;
    } else {
      next = bandSteps.find((step) => step > current) || current;
    }
    state.bandWidth = state.quantileMode === "quantile" ? String(quantileCountForBandWidth(next)) : formatBandWidth(next);
    state.bandFeature = currentBandFeatureKey();
    syncBandingControl();
    refreshChart();
  }

  function updateAxisControls() {
    const kind = selectedColumn()?.kind;
    const isDate = kind === "date" || kind === "datetime";
    const isNumeric = isNumericKind(kind);
    const isCategorical = kind === "categorical";
    const hasExpected = Boolean(el("expectedNumerator").value);
    el("sortControl").classList.toggle("hidden", !isCategorical);
    el("expectedSortButton").classList.toggle("hidden", !hasExpected);
    el("dateControl").classList.toggle("hidden", !isDate);
    el("bandControl").classList.toggle("hidden", !isNumeric);
    el("quantileControl").classList.toggle("hidden", !isNumeric);
    const bandFeatureKey = currentBandFeatureKey();
    if (isNumeric && state.tool === "line_bar" && state.bandFeature !== bandFeatureKey) {
      requestBandSuggestionForSelectedColumn(bandFeatureKey);
    }
    if (isNumeric && state.quantileMode === "quantile") {
      normalizeBandWidthForQuantiles();
    }
    if (!isCategorical || (state.sort === "expected" && !hasExpected)) {
      state.sort = "alpha";
      syncSegmented("sort", "alpha");
    } else {
      syncSegmented("sort", state.sort);
    }
    if (!isDate) {
      state.dateBucket = "none";
      syncSegmented("dateBucket", "none");
    }
    if (!isNumeric) {
      state.bandWidth = "0";
      state.quantileMode = "off";
      state.bandFeature = bandFeatureKey;
      clearPendingBandSuggestion();
      syncSegmented("bandWidth", "0");
    }
    syncBandingControl();
    syncQuantileControl();
  }

  function renderExpectedNumerators() {
    const query = el("expectedSearch").value.trim().toLowerCase();
    const select = el("expectedNumerator");
    const list = el("expectedList");
    list.innerHTML = "";

    function addExpectedButton(label, value, kind, extraClass = "") {
      const button = document.createElement("button");
      button.type = "button";
      button.className = `feature ${extraClass} ${value === select.value ? "active" : ""}`.trim();
      button.innerHTML = `<span>${escapeHtml(label)}</span><span class="kind">${escapeHtml(kind)}</span>`;
      button.addEventListener("click", () => {
        const changed = select.value !== value;
        select.value = value;
        renderExpectedNumerators();
        updateAxisControls();
        if (changed) refreshChart();
      });
      list.append(button);
    }

    if (!query || "none".includes(query) || "no expected line".includes(query) || "off".includes(query)) {
      addExpectedButton("No expected line", "", "off", "expected-none-option");
    }

    for (const col of expectedDisplayColumns()) {
      if (query && !col.name.toLowerCase().includes(query)) continue;
      addExpectedButton(col.name, col.name, col.kind);
    }
  }

  function renderFeatures() {
    const query = el("featureSearch").value.trim().toLowerCase();
    const list = el("featureList");
    list.innerHTML = "";
    const columns = [...sourceColumns()];
    if (state.featureSort === "alpha") {
      columns.sort((a, b) => a.name.localeCompare(b.name, undefined, { sensitivity: "base" }));
    }
    for (const col of columns) {
      if (query && !col.name.toLowerCase().includes(query)) continue;
      const button = document.createElement("button");
      button.className = `feature ${col.name === state.x ? "active" : ""}`;
      button.innerHTML = `<span>${escapeHtml(col.name)}</span><span class="kind">${col.kind}</span>`;
      button.addEventListener("click", () => {
        state.x = col.name;
        renderFeatures();
        updateAxisControls();
        refreshChart();
      });
      list.append(button);
    }
  }

  function clearSearchInput(inputId, render) {
    const input = el(inputId);
    if (input.value) {
      input.value = "";
      render();
    }
    input.focus();
  }

  function currentResponses() {
    const responses = [];
    if (el("actualNumerator").value) {
      responses.push({
        label: el("actualNumerator").value,
        numerator: el("actualNumerator").value,
      });
    }
    if (el("expectedNumerator").value) {
      responses.push({
        label: el("expectedNumerator").value,
        numerator: el("expectedNumerator").value,
      });
    }
    return responses;
  }

  function buildChartRequest() {
    if (!state.schema || !state.x) return null;
    const kind = selectedColumn()?.kind;
    const isDate = kind === "date" || kind === "datetime";
    const isNumeric = isNumericKind(kind);
    const bandFeatureKey = currentBandFeatureKey();
    if (isNumeric && state.bandFeature !== bandFeatureKey) {
      requestBandSuggestionForSelectedColumn(bandFeatureKey);
      return null;
    }
    if (isNumeric && state.bandSuggestionPendingKey === bandFeatureKey) {
      setGroupMeta("line_bar", "Estimating banding...");
      return null;
    }
    return {
      source: state.source || "dataset",
      x: state.x,
      sort: state.sort,
      lowGroup: state.lowGroup,
      bandWidth: isNumeric ? Number(state.bandWidth) : 0,
      quantileMode: isNumeric ? state.quantileMode : "off",
      dateBucket: isDate ? state.dateBucket : "none",
      transform: state.transform,
      sigma: Number(state.sigma),
      filter: state.activeFilter,
      denominator: el("denominator").value,
      responses: currentResponses(),
      maxGroups: 10000,
    };
  }

  async function refreshChart(options = {}) {
    return refreshLineBar(options);
  }

  async function fetchChartData(request, requestKey) {
    const requestSeq = state.chartRequestSeq + 1;
    state.chartRequestSeq = requestSeq;
    setStatus("");
    setChartMessage("");
    setGroupMeta("line_bar", "Computing...");
    startToolTiming("line_bar");
    updateAxisControls();
    try {
      const data = await api("/api/chart", { method: "POST", body: JSON.stringify(request), clientTiming: true });
      if (requestSeq !== state.chartRequestSeq) return;
      const cache = toolCache("line_bar");
      cache.requestKey = requestKey;
      cache.data = data;
      syncDuckDbTimingFromData("line_bar", data);
      syncClientTimingFromData("line_bar", data);
      measureToolRender("line_bar", () => renderChartData(data, { resetTablePage: true }));
      return data;
    } catch (error) {
      if (requestSeq !== state.chartRequestSeq) return;
      setToolTimingFailed("line_bar");
      setGroupMeta("line_bar", "Query failed");
      setChartMessage("");
      setStatus(error.message, true);
    }
  }

  function renderChartData(data, options = {}) {
    state.lastData = data;
    if (options.resetTablePage) {
      state.tablePage = 1;
    }
    updateMetricTitles(data);
    const labelMessage = renderChart(data);
    renderTable(data);
    const rowMeta = formatRowMeta(data.row_count, data.filtered_row_count);
    const groupMeta = `${data.rows.length.toLocaleString()} groups · ${rowMeta}`;
    const warnings = [...(data.warnings || [])].filter(Boolean).join(" ");
    const chartMessage = [warnings, labelMessage].filter(Boolean).join(" ");
    setFilterRowMeta(data.row_count, data.filtered_row_count);
    setGroupMeta("line_bar", groupMeta);
    setStatus("");
    setChartMessage(chartMessage);
    saveToolPresentation("line_bar", { groupMeta, chartMessage });
  }

  function useCachedChartData(cache, options = {}) {
    state.lastData = cache.data;
    if (options.renderIfCached) {
      measureToolRender("line_bar", () => renderChartData(cache.data));
      return;
    }
    measureToolRender("line_bar", () => {
      updateMetricTitles(cache.data);
      applyToolPresentation("line_bar");
      requestAnimationFrame(() => chart.resize());
    });
  }

  function renderChart(data) {
    const labels = data.rows.map((r) => formatXLabel(r.x, data.x_kind));
    const labelMode = state.labels;
    const rawXValues = data.rows.map((r) => r.x);
    const xLabelPolicy = getXAxisLabelPolicy(labels, data.x_kind, rawXValues);
    const dataLabelsAllowed = labels.length < LABEL_DENSITY_LIMIT;
    const showBarLabels = dataLabelsAllowed && (labelMode === "bar" || labelMode === "all");
    const showLineLabels = dataLabelsAllowed && (labelMode === "line" || labelMode === "all");
    const barLayout = getBarLayout(labels.length);
    const responseAxis = responseAxisOptions(data);
    const actualColor = getCss("--actual-line");
    const expectedColor = "#d13f3f";
    const responseColors = [actualColor, expectedColor];
    const nColor = getCss("--bar");
    const weightLabel = data.denominator?.bar_label || "Weight";
    const sigmaColor = "#8a94a6";
    const legendData = [
      ...data.responses.map((response) => response.label),
      { name: weightLabel, icon: "roundRect", itemStyle: { color: nColor, borderColor: nColor } },
    ];
    const barSeries = {
      name: weightLabel,
      type: "bar",
      yAxisIndex: 1,
      z: 1,
      legendHoverLink: true,
      itemStyle: { color: nColor },
      animation: false,
      animationDuration: 0,
      animationDurationUpdate: 0,
      data: data.rows.map((r) => ({
        value: r.volume,
        itemStyle: { color: r.is_tail ? getCss("--tail") : getCss("--bar") },
      })),
      label: { show: showBarLabels, position: "top", fontSize: 10, formatter: formatChartLabel },
      barWidth: barLayout.width,
      barMaxWidth: barLayout.maxWidth,
      barCategoryGap: barLayout.categoryGap,
    };
    const lineSeries = data.responses.map((response, index) => ({
      name: response.label,
      type: "line",
      yAxisIndex: 0,
      z: 3,
      animation: false,
      animationDuration: 0,
      animationDurationUpdate: 0,
      smooth: false,
      showSymbol: data.rows.length < 250,
      symbolSize: 5,
      lineStyle: { color: responseColors[index] || actualColor },
      itemStyle: { color: responseColors[index] || actualColor },
      data: data.rows.map((r) => r[`resp${index}`]),
      showAllSymbol: true,
      label: { show: showLineLabels, fontSize: 10, formatter: formatLineLabel },
    }));

    const customSeries = [];
    if (Number(state.sigma) > 0 && data.responses.length >= 2) {
      customSeries.push({
        name: "sigma",
        type: "custom",
        yAxisIndex: 0,
        z: 5,
        legendHoverLink: false,
        animation: false,
        animationDuration: 0,
        animationDurationUpdate: 0,
        renderItem: function (params, api) {
          const x = api.coord([api.value(0), api.value(1)])[0];
          const low = api.coord([api.value(0), api.value(2)])[1];
          const high = api.coord([api.value(0), api.value(3)])[1];
          if (!Number.isFinite(low) || !Number.isFinite(high)) return;
          return {
            type: "group",
            children: [
              { type: "line", shape: { x1: x, y1: low, x2: x, y2: high }, style: { stroke: sigmaColor, lineWidth: 1.5 } },
              { type: "line", shape: { x1: x - 4, y1: low, x2: x + 4, y2: low }, style: { stroke: sigmaColor, lineWidth: 1.5 } },
              { type: "line", shape: { x1: x - 4, y1: high, x2: x + 4, y2: high }, style: { stroke: sigmaColor, lineWidth: 1.5 } },
            ],
          };
        },
        data: data.rows.map((r, i) => [i, r.resp1, r.resp1_low, r.resp1_high]).filter((r) => r.every((v) => v !== null && v !== undefined)),
        encode: { x: 0, y: [2, 3] },
        tooltip: { show: false },
      });
    }

    chart.setOption(
      {
        animation: false,
        animationDuration: 0,
        animationDurationUpdate: 0,
        stateAnimation: { duration: 0 },
        backgroundColor: "transparent",
        color: [actualColor, expectedColor, nColor],
        tooltip: {
          trigger: "axis",
          formatter: (params) => formatChartTooltip(params, weightLabel),
        },
        legend: {
          top: 0,
          data: legendData,
          selectedMode: false,
          textStyle: { color: getCss("--text"), fontWeight: 700 },
        },
        grid: { left: 72, right: 76, top: 56, bottom: xLabelPolicy.bottom, containLabel: false },
        xAxis: {
          type: "category",
          name: data.x || "",
          nameLocation: "middle",
          nameGap: xLabelPolicy.nameGap,
          nameTextStyle: { color: getCss("--text"), fontSize: 13, fontWeight: 700 },
          data: labels,
          axisLabel: {
            show: xLabelPolicy.show,
            color: getCss("--text"),
            interval: xLabelPolicy.interval,
            formatter: xLabelPolicy.formatter,
            hideOverlap: false,
            showMinLabel: xLabelPolicy.showMinLabel,
            showMaxLabel: xLabelPolicy.showMaxLabel,
            rotate: xLabelPolicy.rotate,
            fontSize: xLabelPolicy.fontSize,
            margin: 8,
          },
          axisLine: { lineStyle: { color: getCss("--line") } },
        },
        yAxis: [
          { type: "value", scale: true, splitNumber: RESPONSE_AXIS_TARGET_INTERVALS, min: responseAxis.min, max: responseAxis.max, interval: responseAxis.interval, axisLabel: { color: getCss("--text"), formatter: (value) => formatLineValue(value) }, splitLine: { lineStyle: { color: getCss("--line") } } },
          { type: "value", axisLabel: { color: getCss("--text"), formatter: (value) => formatNumber(value) }, splitLine: { show: false } },
        ],
        dataZoom: labels.length > 120 ? [{ type: "inside" }, { type: "slider", height: 18, bottom: 18 }] : [],
        series: [barSeries, ...lineSeries, ...customSeries],
      },
      true,
    );
    requestAnimationFrame(() => chart.resize());
    return chartDensityMessage(labels.length, !xLabelPolicy.show, !dataLabelsAllowed && labelMode !== "-");
  }

  function chartDensityMessage(groupCount, xLabelsHidden, chartLabelsHidden) {
    if (!xLabelsHidden && !chartLabelsHidden) return "";
    const labelTarget = xLabelsHidden && chartLabelsHidden
      ? "X-axis and chart labels"
      : xLabelsHidden ? "X-axis labels" : "Chart labels";
    return `${labelTarget} hidden as >${LABEL_DENSITY_LIMIT.toLocaleString()} categories.`;
  }

  function formatChartTooltip(params, weightLabel) {
    const items = Array.isArray(params) ? params : [params];
    if (!items.length) return "";
    const lines = [escapeHtml(items[0].axisValueLabel ?? items[0].name ?? "")];
    items.forEach((item) => {
      const value = Array.isArray(item.value) ? item.value[1] : item.value;
      const formatter = item.seriesName === weightLabel ? formatNumber : formatLineValue;
      lines.push(`${item.marker || ""}${escapeHtml(item.seriesName)}: ${escapeHtml(formatter(value))}`);
    });
    return lines.join("<br/>");
  }

  function updateMetricTitles(data) {
    const summaries = data.response_summaries || [];
    renderMetricTitle(el("actualMetricTitle"), "Actual", summaries[0]?.value);
    renderMetricTitle(el("expectedMetricTitle"), "Expected", summaries[1]?.value);
    renderMetricTitle(el("weightMetricTitle"), "Weight", data.denominator?.value, formatWeightValue);
  }

  function responseAxisOptions(data) {
    return responseAxisBounds(responseAxisExtent(data.rows, data.responses.length)) || {};
  }

  function responseAxisExtent(rows, responseCount) {
    let min = Infinity;
    let max = -Infinity;
    rows.forEach((row) => {
      for (let index = 0; index < responseCount; index += 1) {
        const value = Number(row[`resp${index}`]);
        if (!Number.isFinite(value)) continue;
        min = Math.min(min, value);
        max = Math.max(max, value);
      }
    });
    return Number.isFinite(min) && Number.isFinite(max) ? { min, max } : null;
  }

  function responseAxisSpan(value) {
    const min = Number(value?.min);
    const max = Number(value?.max);
    if (!Number.isFinite(min) || !Number.isFinite(max)) return null;
    if (max > min) return max - min;
    return Math.max(Math.abs(max), Math.abs(min), 1);
  }

  function niceAxisStep(span) {
    if (!Number.isFinite(span) || span <= 0) return 1;
    const roughStep = span / RESPONSE_AXIS_TARGET_INTERVALS;
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

  function responseAxisBounds(value) {
    const min = Number(value?.min);
    const max = Number(value?.max);
    const span = responseAxisSpan(value);
    if (!Number.isFinite(min) || !Number.isFinite(max) || span === null) return null;
    const paddedMin = min - span * RESPONSE_AXIS_PADDING;
    const paddedMax = max + span * RESPONSE_AXIS_PADDING;
    const step = niceAxisStep(paddedMax - paddedMin);
    let axisMin = Math.floor(paddedMin / step) * step;
    let axisMax = Math.ceil(paddedMax / step) * step;
    if (min >= 0) axisMin = Math.max(0, axisMin);
    if (axisMax <= axisMin) axisMax = axisMin + step;
    return {
      min: roundAxisValue(axisMin, step),
      max: roundAxisValue(axisMax, step),
      interval: step,
    };
  }

  function getXAxisLabelPolicy(labels, kind = "", rawValues = labels) {
    if (isDateKind(kind)) return getDateXAxisLabelPolicy(labels, rawValues);
    const maxLength = labels.reduce((longest, label) => Math.max(longest, String(label).length), 0);
    const tooMany = labels.length >= LABEL_DENSITY_LIMIT;
    const dataZoomSpace = labels.length > 120 ? 36 : 0;
    if (tooMany) {
      return {
        show: false,
        interval: 0,
        formatter: undefined,
        showMinLabel: true,
        showMaxLabel: true,
        rotate: 0,
        fontSize: 10,
        nameGap: 22,
        bottom: 38 + dataZoomSpace,
      };
    }
    const rotate = labels.length > 18 || maxLength > 10 ? 65 : 0;
    const fontSize = labels.length > 50 ? 8 : 10;
    const estimatedTextWidth = maxLength * fontSize * 0.5;
    const rotatedHeight = estimatedTextWidth * Math.sin((rotate * Math.PI) / 180) + fontSize * Math.cos((rotate * Math.PI) / 180);
    const labelSpace = rotate ? Math.min(140, Math.max(58, Math.ceil(rotatedHeight) + 18)) : 38;
    const titleGap = rotate ? Math.max(26, labelSpace - 10) : 26;
    return {
      show: true,
      interval: 0,
      formatter: undefined,
      showMinLabel: true,
      showMaxLabel: true,
      rotate,
      fontSize,
      nameGap: titleGap,
      bottom: titleGap + 16 + dataZoomSpace,
    };
  }

  function getDateXAxisLabelPolicy(labels, rawValues) {
    const parsedDates = rawValues.map(parseDateCategory);
    const selectedIndexes = dateXAxisLabelIndexes(parsedDates, labels.length);
    const selectedIndexSet = new Set(selectedIndexes);
    const dataZoomSpace = labels.length > 120 ? 36 : 0;
    return {
      show: selectedIndexes.length > 0,
      interval: (index) => selectedIndexSet.has(index),
      formatter: (value, index) => formatDateAxisLabel(rawValues[index] ?? value, parsedDates[index]),
      showMinLabel: selectedIndexSet.has(0) ? true : undefined,
      showMaxLabel: selectedIndexSet.has(labels.length - 1) ? true : undefined,
      rotate: 0,
      fontSize: 10,
      nameGap: 26,
      bottom: 46 + dataZoomSpace,
    };
  }

  function dateXAxisLabelIndexes(parsedDates, count) {
    if (count <= 0) return [];
    const monthStartIndexes = parsedDates
      .map((date, index) => (date && date.day === 1 ? index : null))
      .filter((index) => index !== null);
    if (monthStartIndexes.length >= DATE_AXIS_MIN_MONTH_LABELS) {
      if (monthStartIndexes.length <= DATE_AXIS_MAX_MONTH_LABELS) return monthStartIndexes;
      const stride = Math.ceil(monthStartIndexes.length / DATE_AXIS_TARGET_LABELS);
      const indexes = monthStartIndexes.filter((_, position) => position % stride === 0);
      return indexes.length >= DATE_AXIS_MIN_MONTH_LABELS ? indexes : sparseDateXAxisLabelIndexes(count);
    }
    return sparseDateXAxisLabelIndexes(count);
  }

  function sparseDateXAxisLabelIndexes(count) {
    if (count <= DATE_AXIS_TARGET_LABELS) return Array.from({ length: count }, (_, index) => index);
    const indexes = new Set([0, count - 1]);
    const step = Math.ceil((count - 1) / (DATE_AXIS_TARGET_LABELS - 1));
    for (let index = 0; index < count; index += step) {
      indexes.add(index);
    }
    return Array.from(indexes).sort((a, b) => a - b);
  }

  function parseDateCategory(value) {
    if (value === null || value === undefined) return null;
    const match = String(value).trim().match(/^(\d{4})-(\d{2})-(\d{2})/);
    if (!match) return null;
    const year = Number(match[1]);
    const month = Number(match[2]);
    const day = Number(match[3]);
    if (!Number.isInteger(year) || !Number.isInteger(month) || !Number.isInteger(day)) return null;
    const checked = new Date(Date.UTC(year, month - 1, day));
    if (checked.getUTCFullYear() !== year || checked.getUTCMonth() !== month - 1 || checked.getUTCDate() !== day) return null;
    return { year, month, day };
  }

  function formatDateAxisLabel(value, parsedDate) {
    if (!parsedDate) return String(value);
    const month = DATE_AXIS_MONTHS[parsedDate.month - 1];
    return `${parsedDate.day} ${month} ${parsedDate.year}`;
  }

  function getBarLayout(count) {
    if (count <= 3) {
      return { width: "62%", maxWidth: 240, categoryGap: "18%" };
    }
    if (count <= 8) {
      return { width: "56%", maxWidth: 180, categoryGap: "24%" };
    }
    if (count <= 20) {
      return { width: "46%", maxWidth: 90, categoryGap: "34%" };
    }
    if (count <= 60) {
      return { width: "68%", maxWidth: 34, categoryGap: "28%" };
    }
    return { width: null, maxWidth: 18, categoryGap: "30%" };
  }

  function renderTable(data) {
    const responseHeaders = data.responses.map((r, i) => `<th>${escapeHtml(r.label)}</th>`).join("");
    const weightLabel = data.denominator?.bar_label || "Weight";
    const needsPagination = data.rows.length > TABLE_PAGE_SIZE;
    const pageCount = needsPagination ? Math.ceil(data.rows.length / TABLE_PAGE_SIZE) : 1;
    state.tablePage = Math.min(Math.max(state.tablePage, 1), pageCount);
    const start = needsPagination ? (state.tablePage - 1) * TABLE_PAGE_SIZE : 0;
    const pageRows = needsPagination ? data.rows.slice(start, start + TABLE_PAGE_SIZE) : data.rows;
    const rows = pageRows
      .map((r) => {
        const values = data.responses.map((_, i) => `<td>${formatLineValue(r[`resp${i}`])}</td>`).join("");
        return `<tr><td>${escapeHtml(formatXLabel(r.x, data.x_kind))}</td><td>${formatNumber(r.volume)}</td>${values}</tr>`;
      })
      .join("");
    const pager = needsPagination
      ? `<div class="table-pagination">
          <span>${(start + 1).toLocaleString()}-${(start + pageRows.length).toLocaleString()} of ${data.rows.length.toLocaleString()} rows</span>
          <button id="tablePrevBtn" type="button"${state.tablePage === 1 ? " disabled" : ""}>Previous</button>
          <span>Page ${state.tablePage.toLocaleString()} of ${pageCount.toLocaleString()}</span>
          <button id="tableNextBtn" type="button"${state.tablePage === pageCount ? " disabled" : ""}>Next</button>
        </div>`
      : "";
    el("tableWrap").innerHTML = `<div class="table-scroll"><table><thead><tr><th>${escapeHtml(data.x)}</th><th>${escapeHtml(weightLabel)}</th>${responseHeaders}</tr></thead><tbody>${rows}</tbody></table></div>${pager}`;
    if (needsPagination) {
      el("tablePrevBtn").addEventListener("click", () => {
        state.tablePage -= 1;
        renderTable(data);
      });
      el("tableNextBtn").addEventListener("click", () => {
        state.tablePage += 1;
        renderTable(data);
      });
    }
  }

  function setView(view) {
    state.view = view;
    if (state.tool !== "line_bar") return;
    el("chartTab").classList.toggle("active", view === "chart");
    el("tableTab").classList.toggle("active", view === "table");
    el("chart").classList.toggle("hidden", view !== "chart");
    el("tableWrap").classList.toggle("hidden", view !== "table");
    el("ukMap").classList.add("hidden");
    el("mapLegend").classList.add("hidden");
    el("chartMessage").classList.toggle("hidden", view !== "chart" || !el("chartMessage").textContent);
    if (view === "chart") chart.resize();
  }

  function bindControls() {
    const lineBarControls = new Set(["sort", "lowGroup", "labels", "bandWidth", "quantileMode", "dateBucket", "transform", "sigma", "featureSort", "expectedSort"]);
    document.querySelectorAll(".segmented").forEach((group) => {
      if (!lineBarControls.has(group.dataset.control)) return;
      group.addEventListener("click", (event) => {
        if (event.target.tagName !== "BUTTON") return;
        if (group.dataset.control === "bandWidth" && event.target.dataset.action) {
          stepBandWidth(event.target.dataset.action === "band-down" ? -1 : 1);
          return;
        }
        group.querySelectorAll("button").forEach((button) => button.classList.remove("active"));
        event.target.classList.add("active");
        const previousControlValue = state[group.dataset.control];
        state[group.dataset.control] = event.target.dataset.value;
        if (group.dataset.control === "featureSort") {
          renderFeatures();
          return;
        }
        if (group.dataset.control === "expectedSort") {
          renderExpectedNumerators();
          return;
        }
        if (group.dataset.control === "bandWidth") {
          clearPendingBandSuggestion();
          state.bandFeature = currentBandFeatureKey();
          if (state.quantileMode === "quantile") {
            normalizeBandWidthForQuantiles();
          } else {
            syncBandingControl();
          }
        }
        if (group.dataset.control === "quantileMode") {
          if (state.quantileMode === "quantile" && previousControlValue !== "quantile") {
            clearPendingBandSuggestion();
            state.bandWidth = "10";
            state.bandFeature = currentBandFeatureKey();
            syncBandingControl();
          } else if (state.quantileMode === "quantile") {
            normalizeBandWidthForQuantiles();
          } else {
            syncBandingControl();
          }
          syncQuantileControl();
        }
        refreshChart({ renderIfCached: group.dataset.control === "labels" });
      });
    });
    el("expectedNumerator").addEventListener("change", () => {
      renderExpectedNumerators();
      updateAxisControls();
      refreshChart();
    });
    el("expectedSearch").addEventListener("input", renderExpectedNumerators);
    el("featureSearch").addEventListener("input", renderFeatures);
    el("expectedSearchClear").addEventListener("click", () => clearSearchInput("expectedSearch", renderExpectedNumerators));
    el("featureSearchClear").addEventListener("click", () => clearSearchInput("featureSearch", renderFeatures));
    el("chartTab").addEventListener("click", () => setView("chart"));
    el("tableTab").addEventListener("click", () => setView("table"));
  }

  function resize() {
    chart.resize();
  }

  function refreshTheme() {
    if (state.lastData) measureToolRender("line_bar", () => renderChart(state.lastData));
  }

  return {
    buildRequest: buildChartRequest,
    fetchData: fetchChartData,
    useCached: useCachedChartData,
    render: renderChartData,
    bindControls,
    setView,
    resize,
    refreshTheme,
    renderFeatures,
    renderExpectedNumerators,
    updateAxisControls,
    canNavigateToFeature: canNavigateToLineBarFeature,
    navigateToFeature: navigateToLineBarFeature,
  };
}
