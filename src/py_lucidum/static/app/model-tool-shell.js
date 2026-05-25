export function createModelToolShell({
  api,
  el,
  escapeHtml,
  measureToolRender,
  saveToolPresentation,
  setChartMessage,
  setClientTiming,
  setDuckDbTiming,
  setGroupMeta,
  setRenderTiming,
  setStatus,
  setToolTimingFailed,
  startToolTiming,
  state,
  syncClientTimingFromData,
  syncDuckDbTimingFromData,
  toolCache,
}) {
  const labels = {
    glm: "GLM",
    gbm: "GBM",
  };

  function labelFor(tool) {
    return labels[tool] || tool.toUpperCase();
  }

  function buildRequest(tool) {
    if (!state.schema) return null;
    return {
      tool,
      source: state.source || "dataset",
      filter: state.activeFilter,
    };
  }

  async function fetchData(tool, request, requestKey) {
    const requestSeqKey = tool === "glm" ? "glmRequestSeq" : "gbmRequestSeq";
    const requestSeq = state[requestSeqKey] + 1;
    state[requestSeqKey] = requestSeq;
    setStatus("");
    setChartMessage("");
    setGroupMeta(tool, `Loading ${labelFor(tool)}...`);
    startToolTiming(tool);
    try {
      const data = await api(`/api/${tool}/summary`, { method: "GET", clientTiming: true });
      if (requestSeq !== state[requestSeqKey]) return null;
      const cache = toolCache(tool);
      cache.requestKey = requestKey;
      cache.data = data;
      syncDuckDbTimingFromData(tool, data);
      syncClientTimingFromData(tool, data);
      measureToolRender(tool, () => render(tool, data));
      return data;
    } catch (error) {
      if (requestSeq !== state[requestSeqKey]) return null;
      setToolTimingFailed(tool);
      setGroupMeta(tool, `${labelFor(tool)} failed`);
      setChartMessage("");
      setStatus(error.message, true);
      return null;
    }
  }

  function useCached(tool, cache) {
    measureToolRender(tool, () => {
      render(tool, cache.data);
      applyPresentation(tool);
    });
  }

  function applyPresentation(tool) {
    const presentation = toolCache(tool).presentation;
    if (!presentation) return;
    setGroupMeta(tool, presentation.groupMeta);
    setStatus(presentation.status, presentation.statusError);
    setChartMessage(presentation.chartMessage);
  }

  function render(tool, data = {}) {
    const label = labelFor(tool);
    const groupMeta = `${label} setup`;
    const chartMessage = data.message || `${label} modelling will be added in a later slice.`;
    setGroupMeta(tool, groupMeta);
    setStatus("");
    setChartMessage(chartMessage);
    const mount = el("modelToolWrap");
    if (mount) {
      mount.innerHTML = `
        <div class="model-shell">
          <div class="model-shell-header">
            <h2>${escapeHtml(label)}</h2>
            <span>${escapeHtml(state.schema?.path?.split(/[\\\\/]/).pop() || "Dataset")}</span>
          </div>
          <div class="model-shell-body">
            <p>${escapeHtml(chartMessage)}</p>
            <dl>
              <div><dt>Source</dt><dd>${escapeHtml(state.source || "dataset")}</dd></div>
              <div><dt>Rows</dt><dd>${Number(state.schema?.row_count || 0).toLocaleString()}</dd></div>
              <div><dt>Columns</dt><dd>${Number(state.schema?.columns?.length || 0).toLocaleString()}</dd></div>
            </dl>
          </div>
        </div>
      `;
    }
    setDuckDbTiming(tool, data.timings || {});
    setClientTiming(tool, data.client_timings || {});
    setRenderTiming(tool, 0);
    saveToolPresentation(tool, { groupMeta, chartMessage });
  }

  return {
    buildRequest,
    fetchData,
    labelFor,
    render,
    useCached,
  };
}
