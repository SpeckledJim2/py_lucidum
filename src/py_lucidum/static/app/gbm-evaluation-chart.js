import { observeResize } from "./shared/model-ui.js";

const GBM_EVALUATION_DOWNSAMPLE_THRESHOLD = 2000;
const GBM_EVALUATION_MAX_PLOT_POINTS = 1500;

export function createGbmEvaluationChart({ escapeHtml, formatEvaluationValue }) {
  let chart = null;
  let resizeObserver = null;
  let viewMode = "all";

  function render(source = null) {
    const target = document.getElementById("gbmEvaluationChart");
    const detail = source || {};
    const evaluation = detail?.evaluation;
    if (!target || !window.echarts || !evaluation) return;
    const rows = [];
    for (const [datasetName, metrics] of Object.entries(evaluation)) {
      for (const [metricName, values] of Object.entries(metrics)) {
        rows.push({ datasetName, metricName, values });
      }
    }
    if (!rows.length) return;
    rows.sort(compareEvaluationRows);
    const metricNames = new Set(rows.map((row) => row.metricName));
    const primaryMetric = String(detail?.metric || rows[0]?.metricName || "metric");
    const maxIteration = Math.max(1, ...rows.map((row) => row.values.length));
    const xMax = evaluationXAxisMax(maxIteration, detail?.progress || null);
    const xDomain = evaluationXDomain(maxIteration, detail, xMax);
    const xInterval = niceIterationInterval(evaluationXDomainSpan(xDomain));
    const xLabelInterval = niceIterationLabelInterval(evaluationXDomainSpan(xDomain));
    const yAxisBounds = evaluationYAxisBounds(rows, maxIteration, primaryMetric, detail, xDomain);
    const sampledEvaluationIndexes = evaluationSampledIndexes(rows, maxIteration, detail?.manifest || {}, detail?.progress || null, xDomain);
    const title = evaluationTitle(rows, primaryMetric, detail?.manifest || {}, detail?.progress || null);
    const textColor = cssVar("--text", "#3f3f46");
    const mutedColor = cssVar("--muted", "#4b5563");
    const lineColor = cssVar("--line", "#e5e7eb");
    const panelColor = cssVar("--panel", "#ffffff");
    if (!chart) {
      chart = window.echarts.init(target);
      bindResize(target);
    }
    chart.setOption({
      animation: false,
      color: ["#ff140f", cssVar("--actual-line", "#050505"), "#2563eb", "#7c3aed"],
      title: {
        text: title,
        left: "center",
        top: 8,
        textStyle: { color: textColor, fontSize: 12, fontWeight: 800, lineHeight: 15 },
      },
      legend: {
        orient: "vertical",
        right: 8,
        top: "middle",
        itemWidth: 10,
        itemHeight: 10,
        textStyle: { color: textColor, fontSize: 12 },
      },
      tooltip: {
        trigger: "axis",
        backgroundColor: panelColor,
        borderColor: lineColor,
        textStyle: { color: textColor },
        formatter: (params) => evaluationTooltipFormatter(params),
      },
      grid: { left: 12, right: 82, top: 42, bottom: 20, containLabel: true },
      xAxis: {
        type: "value",
        min: xDomain.min,
        max: xDomain.max,
        interval: xInterval,
        axisLabel: { color: mutedColor, hideOverlap: false, margin: 4, formatter: (value) => evaluationIterationAxisLabel(value, xLabelInterval) },
        axisLine: { lineStyle: { color: mutedColor } },
        splitLine: { lineStyle: { color: lineColor } },
      },
      yAxis: {
        type: "value",
        scale: true,
        ...yAxisBounds,
        splitNumber: 4,
        axisLabel: { color: mutedColor, formatter: (value) => formatEvaluationAxisValue(value) },
        axisLine: { show: true, lineStyle: { color: mutedColor } },
        splitLine: { lineStyle: { color: lineColor } },
      },
      series: rows.map((row) => ({
        name: evaluationSeriesName(row, metricNames.size > 1),
        type: "line",
        showSymbol: true,
        symbol: "circle",
        symbolSize: 4,
        lineStyle: { width: 2 },
        data: evaluationSeriesData(row.values, sampledEvaluationIndexes),
      })),
    }, true);
    requestAnimationFrame(() => chart?.resize());
  }

  function setViewMode(mode) {
    viewMode = mode === "tail" ? "tail" : "all";
  }

  function getViewMode() {
    return viewMode;
  }

  function dispose() {
    resizeObserver?.disconnect();
    resizeObserver = null;
    if (chart) {
      chart.dispose();
      chart = null;
    }
  }

  function bindResize(target) {
    resizeObserver = observeResize([target, target.parentElement], () => {
      chart?.resize();
    });
  }

  function evaluationSeriesName(row, includeMetric) {
    const dataset = String(row.datasetName || "").toLowerCase() === "training" ? "train" : String(row.datasetName || "series");
    return includeMetric ? `${dataset} ${row.metricName}` : dataset;
  }

  function compareEvaluationRows(left, right) {
    const leftOrder = evaluationDatasetOrder(left.datasetName);
    const rightOrder = evaluationDatasetOrder(right.datasetName);
    if (leftOrder !== rightOrder) return leftOrder - rightOrder;
    return String(left.metricName).localeCompare(String(right.metricName));
  }

  function evaluationDatasetOrder(datasetName) {
    const name = String(datasetName || "").toLowerCase();
    if (name === "training" || name === "train") return 0;
    if (name === "test" || name === "validation" || name === "valid") return 1;
    return 2;
  }

  function evaluationTitle(rows, primaryMetric, manifest = {}, progress = null) {
    const bestIteration = Math.max(0, Number(manifest.best_iteration || 0));
    const metric = primaryMetric || rows[0]?.metricName || "metric";
    const testRow = rows.find((row) => row.datasetName === "test" && row.metricName === metric)
      || rows.find((row) => row.datasetName === "test")
      || rows.find((row) => row.metricName === metric)
      || rows[0];
    const livePoint = progress ? preferredLiveMetric(progress.latest || [], metric) : null;
    const liveValue = Number(livePoint?.value);
    const bestValue = Number.isFinite(liveValue) ? liveValue : valueAtIteration(testRow?.values || [], bestIteration) ?? lastFiniteValue(testRow?.values || []);
    const parts = [];
    parts.push(`evaluation metric: ${metric}`);
    if (bestValue !== null) parts.push(`test metric: ${formatEvaluationValue(bestValue)}`);
    if (progress?.iteration) {
      parts.push(`iteration: ${Number(progress.iteration).toLocaleString()}`);
    } else if (bestIteration) {
      parts.push(`best iteration: ${bestIteration.toLocaleString()}`);
    }
    return parts.join(", ");
  }

  function preferredLiveMetric(latest, metric) {
    if (!Array.isArray(latest) || !latest.length) return null;
    return [...latest].sort((left, right) => liveMetricSortKey(left, metric).localeCompare(liveMetricSortKey(right, metric)))[0];
  }

  function liveMetricSortKey(item, metric) {
    const dataset = String(item?.dataset || "").toLowerCase();
    const datasetRank = dataset === "test" ? "0" : ["validation", "valid"].includes(dataset) ? "1" : ["training", "train"].includes(dataset) ? "2" : "3";
    const metricRank = String(item?.metric || "") === String(metric || "") ? "0" : "1";
    return `${datasetRank}:${metricRank}:${item?.metric || ""}`;
  }

  function valueAtIteration(values, iteration) {
    if (!iteration || iteration < 1) return null;
    const value = values[iteration - 1];
    return Number.isFinite(Number(value)) ? Number(value) : null;
  }

  function lastFiniteValue(values) {
    for (let index = values.length - 1; index >= 0; index -= 1) {
      const value = Number(values[index]);
      if (Number.isFinite(value)) return value;
    }
    return null;
  }

  function evaluationXAxisMax(maxIteration, progress = null) {
    const liveTotal = progress?.phase === "training" ? Number(progress.total_iterations) : NaN;
    if (Number.isFinite(liveTotal) && liveTotal > 0) return Math.max(1, Math.round(liveTotal));
    return Math.max(1, Math.round(Number(maxIteration || 1)));
  }

  function evaluationXDomain(maxIteration, detail, xMax) {
    if (viewMode !== "tail") return { min: 0, max: xMax };
    const tailMax = Math.max(1, Math.round(Number(maxIteration || 1)));
    const width = evaluationTailWindowSize(tailMax, detail);
    return { min: Math.max(1, tailMax - width + 1), max: tailMax };
  }

  function evaluationXDomainSpan(domain) {
    return Math.max(1, Number(domain?.max || 1) - Number(domain?.min || 0));
  }

  function evaluationTailWindowSize(maxIteration, detail) {
    const count = Math.max(1, Math.round(Number(maxIteration || 1)));
    const bestIteration = Math.round(Number(detail?.manifest?.best_iteration || 0));
    const earlyStoppingRounds = evaluationEarlyStoppingRounds(detail);
    if (bestIteration >= 1 && earlyStoppingRounds > 0 && count - bestIteration >= earlyStoppingRounds) {
      return Math.min(count, Math.max(50, earlyStoppingRounds * 5));
    }
    return Math.min(count, Math.max(50, Math.ceil(count * 0.2)));
  }

  function evaluationEarlyStoppingRounds(detail) {
    const value = evaluationParameterValue(detail?.parameters, "early_stopping_rounds");
    const rounds = Math.round(Number(value));
    return Number.isFinite(rounds) && rounds > 0 ? rounds : 0;
  }

  function evaluationParameterValue(parameters, name) {
    if (Array.isArray(parameters)) {
      return parameters.find((parameter) => String(parameter?.name || "") === name)?.value;
    }
    if (parameters && typeof parameters === "object") return parameters[name];
    return null;
  }

  function evaluationYAxisBounds(rows, maxIteration, primaryMetric, detail, xDomain) {
    if (viewMode === "tail") return evaluationTailYAxisBounds(rows, primaryMetric, xDomain);
    const yMax = evaluationYAxisMax(rows, maxIteration);
    return yMax !== null ? { max: yMax } : {};
  }

  function evaluationTailYAxisBounds(rows, primaryMetric, xDomain) {
    const row = evaluationTailFocusRow(rows, primaryMetric);
    const values = Array.isArray(row?.values) ? row.values : [];
    const startIndex = Math.max(0, Math.ceil(Number(xDomain?.min || 1)) - 1);
    const endIndex = Math.min(values.length - 1, Math.floor(Number(xDomain?.max || values.length)) - 1);
    const extent = evaluationEmptyExtent();
    for (let index = startIndex; index <= endIndex; index += 1) {
      const value = Number(values[index]);
      if (Number.isFinite(value)) updateEvaluationExtent(extent, value);
    }
    if (!extent.count) return {};
    const padding = evaluationTailYAxisPadding(extent);
    return { min: extent.min - padding, max: extent.max + padding };
  }

  function evaluationTailYAxisPadding(extent) {
    const range = Math.max(0, extent.max - extent.min);
    if (range > 0) return Math.max(range * 0.2, 1e-9);
    return Math.max(Math.abs(extent.max), 1) * 0.0001;
  }

  function evaluationTailFocusRow(rows, primaryMetric) {
    const metric = String(primaryMetric || "");
    return rows.find((row) => String(row.datasetName || "").toLowerCase() === "test" && String(row.metricName || "") === metric)
      || rows.find((row) => String(row.datasetName || "").toLowerCase() === "test")
      || rows.find((row) => ["training", "train"].includes(String(row.datasetName || "").toLowerCase()) && String(row.metricName || "") === metric)
      || rows.find((row) => ["training", "train"].includes(String(row.datasetName || "").toLowerCase()))
      || rows.find((row) => String(row.metricName || "") === metric)
      || rows[0];
  }

  function evaluationYAxisMax(rows, maxIteration) {
    if (maxIteration < 50) return null;
    const tailStart = evaluationTailStart(maxIteration);
    const initialExtent = evaluationEmptyExtent();
    const tailExtent = evaluationEmptyExtent();
    for (const row of rows) {
      const values = Array.isArray(row.values) ? row.values : [];
      values.forEach((value, index) => {
        const number = Number(value);
        if (!Number.isFinite(number)) return;
        if (index < tailStart) {
          updateEvaluationExtent(initialExtent, number);
        } else {
          updateEvaluationExtent(tailExtent, number);
        }
      });
    }
    if (!initialExtent.count || !tailExtent.count) return null;
    const initialMax = initialExtent.max;
    const tailMax = tailExtent.max;
    const tailRange = Math.max(0, tailExtent.max - tailExtent.min);
    const materialGap = Math.max(tailRange * 2, Math.abs(tailMax) * 0.03, 1e-9);
    if (initialMax <= tailMax + materialGap) return null;
    const padding = Math.max(tailRange * 0.12, Math.abs(tailMax) * 0.01, 1e-9);
    return tailMax + padding;
  }

  function evaluationTailStart(maxIteration) {
    return Math.max(10, Math.min(300, Math.floor(Number(maxIteration || 0) * 0.08)));
  }

  function evaluationTooltipFormatter(params) {
    const items = Array.isArray(params) ? params : [params].filter(Boolean);
    const first = items[0] || {};
    const rawIteration = Number(first.axisValue ?? (Array.isArray(first.value) ? first.value[0] : null));
    const iteration = Number.isFinite(rawIteration) ? Math.round(rawIteration).toLocaleString() : "";
    const lines = [`<strong>Iteration:</strong> ${escapeHtml(iteration)}`];
    for (const item of items) {
      const value = Array.isArray(item?.value) ? item.value[1] : item?.value;
      lines.push(`${item?.marker || ""}${escapeHtml(item?.seriesName || "series")}: ${escapeHtml(formatEvaluationValue(value))}`);
    }
    return lines.join("<br/>");
  }

  function evaluationSampledIndexes(rows, maxIteration, manifest = {}, progress = null, xDomain = null) {
    const count = Math.max(1, Math.round(Number(maxIteration || 1)));
    const range = evaluationIndexRange(count, xDomain);
    const visibleCount = range.end - range.start + 1;
    if (visibleCount <= GBM_EVALUATION_DOWNSAMPLE_THRESHOLD) return sequentialIndexes(range.start, range.end);
    const requiredIndexes = requiredEvaluationIndexes(count, manifest, progress, range);
    const compositePoints = evaluationCompositePoints(rows, range.start, range.end);
    if (compositePoints.length <= GBM_EVALUATION_MAX_PLOT_POINTS) {
      return mergeEvaluationIndexes(compositePoints.map((point) => point.index), requiredIndexes);
    }
    const samplingLimit = Math.max(2, GBM_EVALUATION_MAX_PLOT_POINTS - requiredIndexes.size);
    const sampled = largestTriangleThreeBuckets(compositePoints, samplingLimit).map((point) => point.index);
    return mergeEvaluationIndexes(sampled, requiredIndexes);
  }

  function evaluationIndexRange(maxIteration, xDomain = null) {
    const end = Math.max(0, Math.min(maxIteration - 1, Math.floor(Number(xDomain?.max || maxIteration)) - 1));
    const start = Math.max(0, Math.min(end, Math.ceil(Number(xDomain?.min || 1)) - 1));
    return { start, end };
  }

  function sequentialIndexes(start, end) {
    const count = Math.max(0, end - start + 1);
    return Array.from({ length: count }, (_value, offset) => start + offset);
  }

  function requiredEvaluationIndexes(maxIteration, manifest = {}, progress = null, range = null) {
    const required = new Set([0, Math.max(0, maxIteration - 1)]);
    const bestIteration = Math.round(Number(manifest?.best_iteration || 0));
    const liveIteration = Math.round(Number(progress?.iteration || 0));
    for (const iteration of [bestIteration, liveIteration]) {
      if (iteration >= 1 && iteration <= maxIteration) required.add(iteration - 1);
    }
    if (!range) return required;
    return new Set([...required].filter((index) => index >= range.start && index <= range.end));
  }

  function evaluationCompositePoints(rows, startIndex, endIndex) {
    const stats = rows.map((row) => evaluationRowStats(row.values));
    const points = [];
    for (let index = startIndex; index <= endIndex; index += 1) {
      let total = 0;
      let count = 0;
      rows.forEach((row, rowIndex) => {
        const value = Number(row.values?.[index]);
        const stat = stats[rowIndex];
        if (!Number.isFinite(value) || !stat) return;
        total += stat.range > 0 ? (value - stat.min) / stat.range : 0.5;
        count += 1;
      });
      if (count) points.push({ index, x: index, y: total / count });
    }
    return points;
  }

  function evaluationRowStats(values) {
    const extent = evaluationEmptyExtent();
    for (const value of Array.isArray(values) ? values : []) {
      const number = Number(value);
      if (Number.isFinite(number)) updateEvaluationExtent(extent, number);
    }
    if (!extent.count) return null;
    return { min: extent.min, max: extent.max, range: extent.max - extent.min };
  }

  function evaluationEmptyExtent() {
    return { count: 0, min: Infinity, max: -Infinity };
  }

  function updateEvaluationExtent(extent, value) {
    extent.count += 1;
    if (value < extent.min) extent.min = value;
    if (value > extent.max) extent.max = value;
  }

  function largestTriangleThreeBuckets(points, threshold) {
    if (threshold >= points.length || threshold <= 2) return points.slice();
    const sampled = [points[0]];
    let anchorIndex = 0;
    const bucketSize = (points.length - 2) / (threshold - 2);
    for (let bucket = 0; bucket < threshold - 2; bucket += 1) {
      const bucketStart = Math.floor((bucket + 0) * bucketSize) + 1;
      const bucketEnd = Math.floor((bucket + 1) * bucketSize) + 1;
      const nextBucketStart = Math.floor((bucket + 1) * bucketSize) + 1;
      const nextBucketEnd = Math.floor((bucket + 2) * bucketSize) + 1;
      const average = averageEvaluationPoint(points.slice(nextBucketStart, Math.min(nextBucketEnd, points.length)));
      const anchor = points[anchorIndex];
      let maxArea = -1;
      let nextAnchorIndex = bucketStart;
      for (let index = bucketStart; index < Math.min(bucketEnd, points.length - 1); index += 1) {
        const point = points[index];
        const area = Math.abs((anchor.x - average.x) * (point.y - anchor.y) - (anchor.x - point.x) * (average.y - anchor.y)) * 0.5;
        if (area > maxArea) {
          maxArea = area;
          nextAnchorIndex = index;
        }
      }
      sampled.push(points[nextAnchorIndex]);
      anchorIndex = nextAnchorIndex;
    }
    sampled.push(points[points.length - 1]);
    return sampled;
  }

  function averageEvaluationPoint(points) {
    if (!points.length) return { x: 0, y: 0 };
    const totals = points.reduce((accumulator, point) => ({
      x: accumulator.x + point.x,
      y: accumulator.y + point.y,
    }), { x: 0, y: 0 });
    return { x: totals.x / points.length, y: totals.y / points.length };
  }

  function mergeEvaluationIndexes(sampledIndexes, requiredIndexes) {
    return [...new Set([...sampledIndexes, ...requiredIndexes])]
      .filter((index) => Number.isInteger(index) && index >= 0)
      .sort((left, right) => left - right);
  }

  function evaluationSeriesData(values, sampledIndexes) {
    const seriesValues = Array.isArray(values) ? values : [];
    return sampledIndexes
      .filter((index) => index < seriesValues.length)
      .map((index) => [index + 1, seriesValues[index]]);
  }

  function niceIterationInterval(maxIteration) {
    return niceIterationStep(Math.max(1, Number(maxIteration || 1) / 30));
  }

  function niceIterationLabelInterval(maxIteration) {
    return niceIterationStep(Math.max(1, Number(maxIteration || 1) / 10));
  }

  function niceIterationStep(rawStep) {
    const raw = Math.max(1, Number(rawStep || 1));
    const magnitude = 10 ** Math.floor(Math.log10(raw));
    for (const step of [1, 2, 5, 10]) {
      const interval = step * magnitude;
      if (interval >= raw) return interval;
    }
    return 10 * magnitude;
  }

  function evaluationIterationAxisLabel(value, labelInterval) {
    const number = Number(value);
    if (!Number.isFinite(number)) return "";
    const iteration = Math.round(number);
    if (Math.abs(number - iteration) > 1e-6) return "";
    const interval = Math.max(1, Math.round(Number(labelInterval || 1)));
    return iteration % interval === 0 ? iteration.toLocaleString() : "";
  }

  function formatEvaluationAxisValue(value) {
    const number = Number(value);
    if (!Number.isFinite(number)) return "";
    const abs = Math.abs(number);
    if (abs >= 1000) return Math.round(number).toLocaleString();
    if (abs >= 10) return number.toLocaleString(undefined, { maximumFractionDigits: 1 });
    if (abs >= 1) return number.toLocaleString(undefined, { maximumFractionDigits: 3 });
    return number.toLocaleString(undefined, { maximumFractionDigits: 4 });
  }

  function cssVar(name, fallback) {
    return getComputedStyle(document.body).getPropertyValue(name).trim() || fallback;
  }

  return {
    dispose,
    getViewMode,
    render,
    setViewMode,
  };
}
