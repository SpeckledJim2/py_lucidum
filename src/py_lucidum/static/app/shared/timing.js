export function freshActionTiming() {
  return {
    duckdbNs: null,
    duckdbMs: null,
    duckdbStatus: "idle",
    clientResponseMs: null,
    clientBodyMs: null,
    clientParseMs: null,
    clientDataMs: null,
    clientTotalMs: null,
    renderNs: null,
    renderStatus: "idle",
  };
}

export function freshActionTimings() {
  return {
    column_profile: freshActionTiming(),
    line_bar: freshActionTiming(),
    histogram: freshActionTiming(),
    uk_map: freshActionTiming(),
    glm: freshActionTiming(),
    gbm: freshActionTiming(),
  };
}

export function createActionTimingController({
  state,
  el,
  renderLabels,
  performanceImpl = performance,
  requestAnimationFrameImpl = requestAnimationFrame,
}) {
  function actionTiming(tool) {
    if (!state.actionTimings[tool]) {
      state.actionTimings[tool] = freshActionTiming();
    }
    return state.actionTimings[tool];
  }

  function formatDurationNumber(value) {
    const number = Number(value);
    if (!Number.isFinite(number)) return "";
    return String(Math.round(number));
  }

  function roundedTimingMilliseconds(valueMs) {
    const number = Number(valueMs);
    return Number.isFinite(number) ? Math.max(0, Math.round(number)) : null;
  }

  function formatActionTimingValue(valueNs, status = "idle") {
    if (status === "running") return "running";
    if (status === "failed") return "failed";
    if (valueNs === null || valueNs === undefined) return "--";
    const ns = Number(valueNs);
    if (!Number.isFinite(ns)) return "--";
    const roundedNs = Math.max(0, Math.round(ns));
    if (roundedNs < 1000) return `${roundedNs}ns`;
    if (roundedNs < 1_000_000) return `${formatDurationNumber(roundedNs / 1000)}us`;
    return `${formatDurationNumber(roundedNs / 1_000_000)}ms`;
  }

  function formatDuckDbTimingValue(timing) {
    if (timing.duckdbStatus === "running") return "running";
    if (timing.duckdbStatus === "failed") return "failed";
    const duckdbNs = Number(timing.duckdbNs);
    if (Number.isFinite(duckdbNs)) return formatActionTimingValue(duckdbNs);
    const duckdbMs = Number(timing.duckdbMs);
    return Number.isFinite(duckdbMs) ? `${formatDurationNumber(Math.max(0, duckdbMs))}ms` : "--";
  }

  function duckDbTimingMilliseconds(timing) {
    const duckdbNs = Number(timing.duckdbNs);
    if (Number.isFinite(duckdbNs)) return roundedTimingMilliseconds(duckdbNs / 1_000_000);
    const duckdbMs = Number(timing.duckdbMs);
    return roundedTimingMilliseconds(duckdbMs);
  }

  function formatRenderTimingValue(timing) {
    if (timing.renderStatus === "rendering") return "rendering...";
    return formatActionTimingValue(timing.renderNs);
  }

  function renderTimingMilliseconds(timing) {
    if (timing.renderStatus === "rendering") return null;
    const renderNs = Number(timing.renderNs);
    return Number.isFinite(renderNs) ? roundedTimingMilliseconds(renderNs / 1_000_000) : null;
  }

  function formatClientTimingValue(timing) {
    if (timing.duckdbStatus === "running") return "--";
    if (timing.duckdbStatus === "failed") return "--";
    const valueMs = timing.clientDataMs;
    const number = Number(valueMs);
    return Number.isFinite(number) ? `${formatDurationNumber(Math.max(0, number))}ms` : "--";
  }

  function formatTotalTimingValue(timing) {
    if (timing.duckdbStatus === "running") return "--";
    if (timing.duckdbStatus === "failed") return "failed";
    const duckdbMs = duckDbTimingMilliseconds(timing);
    const jsonMs = roundedTimingMilliseconds(timing.clientDataMs);
    const renderMs = renderTimingMilliseconds(timing);
    if (duckdbMs === null || jsonMs === null || renderMs === null) return "--";
    return `${formatDurationNumber(duckdbMs + jsonMs + renderMs)}ms`;
  }

  function syncActionTimingMonitor(tool = state.tool) {
    const timing = actionTiming(tool);
    const renderLabel = renderLabels[tool] || "Render";
    el("actionTimingMonitor").textContent = `DuckDB: ${formatDuckDbTimingValue(timing)}, JSON: ${formatClientTimingValue(timing)}, ${renderLabel}: ${formatRenderTimingValue(timing)}, Total: ${formatTotalTimingValue(timing)}`;
  }

  function startToolTiming(tool) {
    const timing = actionTiming(tool);
    timing.duckdbNs = null;
    timing.duckdbMs = null;
    timing.duckdbStatus = "running";
    timing.clientResponseMs = null;
    timing.clientBodyMs = null;
    timing.clientParseMs = null;
    timing.clientDataMs = null;
    timing.clientTotalMs = null;
    timing.renderNs = null;
    timing.renderStatus = "idle";
    if (state.tool === tool) syncActionTimingMonitor(tool);
  }

  function setDuckDbTiming(tool, timings = {}) {
    const timing = actionTiming(tool);
    const duckdbNs = Number(timings.duckdb_ns);
    const duckdbMs = Number(timings.duckdb_ms);
    timing.duckdbNs = Number.isFinite(duckdbNs) ? Math.max(0, Math.round(duckdbNs)) : null;
    timing.duckdbMs = timing.duckdbNs === null && Number.isFinite(duckdbMs) ? Math.max(0, duckdbMs) : null;
    timing.duckdbStatus = "idle";
    if (state.tool === tool) syncActionTimingMonitor(tool);
  }

  function setToolTimingFailed(tool) {
    const timing = actionTiming(tool);
    timing.duckdbNs = null;
    timing.duckdbMs = null;
    timing.duckdbStatus = "failed";
    timing.clientResponseMs = null;
    timing.clientBodyMs = null;
    timing.clientParseMs = null;
    timing.clientDataMs = null;
    timing.clientTotalMs = null;
    timing.renderNs = null;
    timing.renderStatus = "idle";
    if (state.tool === tool) syncActionTimingMonitor(tool);
  }

  function setRenderTimingRunning(tool) {
    const timing = actionTiming(tool);
    timing.renderNs = null;
    timing.renderStatus = "rendering";
    if (state.tool === tool) syncActionTimingMonitor(tool);
  }

  function setRenderTiming(tool, valueMs) {
    const timing = actionTiming(tool);
    const number = Number(valueMs);
    timing.renderNs = Number.isFinite(number) ? Math.max(0, Math.round(number * 1_000_000)) : null;
    timing.renderStatus = "idle";
    if (state.tool === tool) syncActionTimingMonitor(tool);
  }

  function measureToolRender(tool, renderCallback) {
    const started = performanceImpl.now();
    setRenderTimingRunning(tool);
    try {
      const result = renderCallback();
      requestAnimationFrameImpl(() => {
        setRenderTiming(tool, performanceImpl.now() - started);
      });
      return result;
    } catch (error) {
      setRenderTiming(tool, null);
      throw error;
    }
  }

  function syncDuckDbTimingFromData(tool, data) {
    setDuckDbTiming(tool, data?.timings || {});
  }

  function setClientTiming(tool, timings = {}) {
    const timing = actionTiming(tool);
    const responseMs = Number(timings.response_ms);
    const bodyMs = Number(timings.body_ms);
    const parseMs = Number(timings.parse_ms);
    const dataMs = Number(timings.data_ms);
    const totalMs = Number(timings.total_ms);
    timing.clientResponseMs = Number.isFinite(responseMs) ? Math.max(0, responseMs) : null;
    timing.clientBodyMs = Number.isFinite(bodyMs) ? Math.max(0, bodyMs) : null;
    timing.clientParseMs = Number.isFinite(parseMs) ? Math.max(0, parseMs) : null;
    timing.clientDataMs = Number.isFinite(dataMs) ? Math.max(0, dataMs) : null;
    timing.clientTotalMs = Number.isFinite(totalMs) ? Math.max(0, totalMs) : null;
    if (state.tool === tool) syncActionTimingMonitor(tool);
  }

  function syncClientTimingFromData(tool, data) {
    setClientTiming(tool, data?.client_timings || {});
  }

  return {
    actionTiming,
    syncActionTimingMonitor,
    startToolTiming,
    setDuckDbTiming,
    setToolTimingFailed,
    setRenderTimingRunning,
    setRenderTiming,
    measureToolRender,
    syncDuckDbTimingFromData,
    setClientTiming,
    syncClientTimingFromData,
  };
}
