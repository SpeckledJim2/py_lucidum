const GBM_EVALUATION_DOWNSAMPLE_THRESHOLD = 2000;
const GBM_EVALUATION_MAX_PLOT_POINTS = 1500;

export function gbmEvaluationChartOption(detail = {}, options = {}) {
  const rows = evaluationRows(detail?.evaluation);
  if (!rows.length) return null;
  const viewMode = options.viewMode === "tail" ? "tail" : "all";
  const fallbackFormatValue = options.formatValue || defaultEvaluationValue;
  const escapeHtml = options.escapeHtml || defaultEscapeHtml;
  const colors = {
    text: "#3f3f46",
    muted: "#4b5563",
    line: "#e5e7eb",
    panel: "#ffffff",
    actual: "#050505",
    ...(options.colors || {}),
  };
  const metricNames = new Set(rows.map((row) => row.metricName));
  const primaryMetric = String(detail?.metric || rows[0]?.metricName || "metric");
  const formatValue = (value, metric = primaryMetric) => (
    formatEvaluationMetricValue(value, metric, fallbackFormatValue)
  );
  const maxIteration = Math.max(1, ...rows.map((row) => row.values.length));
  const xMax = evaluationXAxisMax(maxIteration, detail?.progress || null);
  const xDomain = evaluationXDomain(maxIteration, detail, xMax, viewMode);
  const domainSpan = evaluationXDomainSpan(xDomain);
  const sampledIndexes = evaluationSampledIndexes(
    rows,
    maxIteration,
    detail?.manifest || {},
    detail?.progress || null,
    xDomain,
  );

  return {
    animation: false,
    color: ["#ff140f", colors.actual, "#2563eb", "#7c3aed"],
    title: {
      text: evaluationTitle(rows, primaryMetric, detail?.manifest || {}, detail?.progress || null, formatValue),
      left: "center",
      top: 8,
      textStyle: { color: colors.text, fontSize: 12, fontWeight: 800, lineHeight: 15 },
    },
    legend: {
      orient: "vertical",
      right: 8,
      top: "middle",
      itemWidth: 10,
      itemHeight: 10,
      textStyle: { color: colors.text, fontSize: 12 },
    },
    tooltip: {
      trigger: "axis",
      backgroundColor: colors.panel,
      borderColor: colors.line,
      textStyle: { color: colors.text },
      formatter: (params) => evaluationTooltipFormatter(params, escapeHtml, formatValue, rows, primaryMetric),
    },
    grid: { left: 12, right: 82, top: 42, bottom: 20, containLabel: true },
    xAxis: {
      type: "value",
      min: xDomain.min,
      max: xDomain.max,
      interval: niceIterationInterval(domainSpan),
      axisLabel: {
        color: colors.muted,
        hideOverlap: false,
        margin: 4,
        formatter: (value) => evaluationIterationAxisLabel(value, niceIterationLabelInterval(domainSpan)),
      },
      axisLine: { lineStyle: { color: colors.muted } },
      splitLine: { lineStyle: { color: colors.line } },
    },
    yAxis: {
      type: "value",
      scale: true,
      ...evaluationYAxisBounds(rows, maxIteration, primaryMetric, detail, xDomain, viewMode),
      splitNumber: 4,
      axisLabel: {
        color: colors.muted,
        formatter: (value) => (
          isMapeMetric(primaryMetric)
            ? formatEvaluationPercentage(value)
            : formatEvaluationAxisValue(value)
        ),
      },
      axisLine: { show: true, lineStyle: { color: colors.muted } },
      splitLine: { lineStyle: { color: colors.line } },
    },
    series: rows.map((row) => ({
      name: evaluationSeriesName(row, metricNames.size > 1),
      type: "line",
      showSymbol: true,
      symbol: "circle",
      symbolSize: 4,
      lineStyle: { width: 2 },
      data: evaluationSeriesData(row.values, sampledIndexes),
    })),
  };
}

function evaluationRows(evaluation) {
  const rows = [];
  for (const [datasetName, metrics] of Object.entries(evaluation || {})) {
    for (const [metricName, values] of Object.entries(metrics || {})) {
      if (Array.isArray(values)) rows.push({ datasetName, metricName, values });
    }
  }
  return rows.sort(compareEvaluationRows);
}

function evaluationSeriesName(row, includeMetric) {
  const dataset = String(row.datasetName || "").toLowerCase() === "training"
    ? "train"
    : String(row.datasetName || "series");
  return includeMetric ? `${dataset} ${row.metricName}` : dataset;
}

function compareEvaluationRows(left, right) {
  const order = evaluationDatasetOrder(left.datasetName) - evaluationDatasetOrder(right.datasetName);
  return order || String(left.metricName).localeCompare(String(right.metricName));
}

function evaluationDatasetOrder(datasetName) {
  const name = String(datasetName || "").toLowerCase();
  if (name === "training" || name === "train") return 0;
  if (name === "test" || name === "validation" || name === "valid") return 1;
  return 2;
}

function evaluationTitle(rows, primaryMetric, manifest, progress, formatValue) {
  const bestIteration = Math.max(0, Number(manifest.best_iteration || 0));
  const metric = primaryMetric || rows[0]?.metricName || "metric";
  const testRow = rows.find((row) => row.datasetName === "test" && row.metricName === metric)
    || rows.find((row) => row.datasetName === "test")
    || rows.find((row) => row.metricName === metric)
    || rows[0];
  const livePoint = progress ? preferredLiveMetric(progress.latest || [], metric) : null;
  const liveValue = Number(livePoint?.value);
  const bestValue = Number.isFinite(liveValue)
    ? liveValue
    : valueAtIteration(testRow?.values || [], bestIteration) ?? lastFiniteValue(testRow?.values || []);
  const parts = [`evaluation metric: ${metric}`];
  if (bestValue !== null) parts.push(`test metric: ${formatValue(bestValue, metric)}`);
  if (progress?.iteration) parts.push(`iteration: ${Number(progress.iteration).toLocaleString()}`);
  else if (bestIteration) parts.push(`best iteration: ${bestIteration.toLocaleString()}`);
  return parts.join(", ");
}

function preferredLiveMetric(latest, metric) {
  if (!Array.isArray(latest) || !latest.length) return null;
  return [...latest].sort((left, right) => (
    liveMetricSortKey(left, metric).localeCompare(liveMetricSortKey(right, metric))
  ))[0];
}

function liveMetricSortKey(item, metric) {
  const dataset = String(item?.dataset || "").toLowerCase();
  const datasetRank = dataset === "test"
    ? "0"
    : ["validation", "valid"].includes(dataset)
      ? "1"
      : ["training", "train"].includes(dataset) ? "2" : "3";
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

function evaluationXAxisMax(maxIteration, progress) {
  const liveTotal = progress?.phase === "training" ? Number(progress.total_iterations) : NaN;
  if (Number.isFinite(liveTotal) && liveTotal > 0) return Math.max(1, Math.round(liveTotal));
  return Math.max(1, Math.round(Number(maxIteration || 1)));
}

function evaluationXDomain(maxIteration, detail, xMax, viewMode) {
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

function evaluationYAxisBounds(rows, maxIteration, primaryMetric, detail, xDomain, viewMode) {
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
  const range = Math.max(0, extent.max - extent.min);
  const padding = range > 0
    ? Math.max(range * 0.2, 1e-9)
    : Math.max(Math.abs(extent.max), 1) * 0.0001;
  return { min: extent.min - padding, max: extent.max + padding };
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
  const tailStart = Math.max(10, Math.min(300, Math.floor(Number(maxIteration || 0) * 0.08)));
  const initialExtent = evaluationEmptyExtent();
  const tailExtent = evaluationEmptyExtent();
  for (const row of rows) {
    const values = Array.isArray(row.values) ? row.values : [];
    values.forEach((value, index) => {
      const number = Number(value);
      if (!Number.isFinite(number)) return;
      updateEvaluationExtent(index < tailStart ? initialExtent : tailExtent, number);
    });
  }
  if (!initialExtent.count || !tailExtent.count) return null;
  const tailRange = Math.max(0, tailExtent.max - tailExtent.min);
  const materialGap = Math.max(tailRange * 2, Math.abs(tailExtent.max) * 0.03, 1e-9);
  if (initialExtent.max <= tailExtent.max + materialGap) return null;
  const padding = Math.max(tailRange * 0.12, Math.abs(tailExtent.max) * 0.01, 1e-9);
  return tailExtent.max + padding;
}

function evaluationTooltipFormatter(params, escapeHtml, formatValue, rows, primaryMetric) {
  const items = Array.isArray(params) ? params : [params].filter(Boolean);
  const first = items[0] || {};
  const rawIteration = Number(first.axisValue ?? (Array.isArray(first.value) ? first.value[0] : null));
  const iteration = Number.isFinite(rawIteration) ? Math.round(rawIteration).toLocaleString() : "";
  const lines = [`<strong>Iteration:</strong> ${escapeHtml(iteration)}`];
  for (const item of items) {
    const value = Array.isArray(item?.value) ? item.value[1] : item?.value;
    const metric = rows[Number(item?.seriesIndex)]?.metricName || primaryMetric;
    lines.push(`${item?.marker || ""}${escapeHtml(item?.seriesName || "series")}: ${escapeHtml(formatValue(value, metric))}`);
  }
  return lines.join("<br/>");
}

function evaluationSampledIndexes(rows, maxIteration, manifest, progress, xDomain) {
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

function evaluationIndexRange(maxIteration, xDomain) {
  const end = Math.max(0, Math.min(maxIteration - 1, Math.floor(Number(xDomain?.max || maxIteration)) - 1));
  const start = Math.max(0, Math.min(end, Math.ceil(Number(xDomain?.min || 1)) - 1));
  return { start, end };
}

function sequentialIndexes(start, end) {
  return Array.from({ length: Math.max(0, end - start + 1) }, (_value, offset) => start + offset);
}

function requiredEvaluationIndexes(maxIteration, manifest, progress, range) {
  const required = new Set([0, Math.max(0, maxIteration - 1)]);
  for (const iteration of [manifest?.best_iteration, progress?.iteration]) {
    const value = Math.round(Number(iteration || 0));
    if (value >= 1 && value <= maxIteration) required.add(value - 1);
  }
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
  return extent.count ? { min: extent.min, max: extent.max, range: extent.max - extent.min } : null;
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
    const bucketStart = Math.floor(bucket * bucketSize) + 1;
    const bucketEnd = Math.floor((bucket + 1) * bucketSize) + 1;
    const nextStart = Math.floor((bucket + 1) * bucketSize) + 1;
    const nextEnd = Math.floor((bucket + 2) * bucketSize) + 1;
    const average = averageEvaluationPoint(points.slice(nextStart, Math.min(nextEnd, points.length)));
    const anchor = points[anchorIndex];
    let maxArea = -1;
    let nextAnchorIndex = bucketStart;
    for (let index = bucketStart; index < Math.min(bucketEnd, points.length - 1); index += 1) {
      const point = points[index];
      const area = Math.abs(
        (anchor.x - average.x) * (point.y - anchor.y)
          - (anchor.x - point.x) * (average.y - anchor.y),
      ) * 0.5;
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
  const totals = points.reduce((sum, point) => ({ x: sum.x + point.x, y: sum.y + point.y }), { x: 0, y: 0 });
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

function defaultEvaluationValue(value) {
  const number = Number(value);
  return Number.isFinite(number)
    ? number.toLocaleString(undefined, { maximumFractionDigits: 4 })
    : "";
}

function formatEvaluationMetricValue(value, metric, fallback) {
  return isMapeMetric(metric) ? formatEvaluationPercentage(value) : fallback(value);
}

function isMapeMetric(metric) {
  return String(metric || "").trim().toLowerCase() === "mape";
}

function formatEvaluationPercentage(value) {
  const number = Number(value);
  return Number.isFinite(number)
    ? `${(number * 100).toLocaleString(undefined, {
      minimumFractionDigits: 1,
      maximumFractionDigits: 1,
    })}%`
    : "";
}

function defaultEscapeHtml(value) {
  return String(value ?? "").replace(/[&<>"']/g, (character) => ({
    "&": "&amp;",
    "<": "&lt;",
    ">": "&gt;",
    '"': "&quot;",
    "'": "&#39;",
  })[character]);
}
