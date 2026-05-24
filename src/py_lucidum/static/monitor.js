function paramsFromLocation() {
  const standardParams = new URLSearchParams(location.search);
  if (standardParams.has("token")) return standardParams;
  const rawSearch = location.search.startsWith("?") ? location.search.slice(1) : location.search;
  try {
    const decodedParams = new URLSearchParams(decodeURIComponent(rawSearch));
    return decodedParams.has("token") ? decodedParams : standardParams;
  } catch (_) {
    return standardParams;
  }
}

const token = paramsFromLocation().get("token") || "";
const state = {
  paused: false,
  timer: null,
};
const el = (id) => document.getElementById(id);

function formatNumber(value) {
  const number = Number(value);
  return Number.isFinite(number) ? number.toLocaleString() : "--";
}

function formatDuration(value) {
  const number = Number(value);
  if (!Number.isFinite(number)) return "--";
  if (number < 1) return `${number.toFixed(1)}ms`;
  return `${Math.round(number).toLocaleString()}ms`;
}

function formatMegabytes(value) {
  const number = Number(value);
  if (!Number.isFinite(number)) return "--";
  if (number >= 1000) return `${Math.round(number).toLocaleString()} MB`;
  return `${number.toLocaleString(undefined, { minimumFractionDigits: 1, maximumFractionDigits: 1 })} MB`;
}

function formatGigabytesFromMegabytes(value) {
  const number = Number(value);
  if (!Number.isFinite(number)) return "--";
  const gigabytes = number / 1024;
  const maximumFractionDigits = gigabytes >= 10 ? 0 : 1;
  return `${gigabytes.toLocaleString(undefined, { maximumFractionDigits })} GB`;
}

function formatPercent(value) {
  const number = Number(value);
  if (!Number.isFinite(number)) return "--";
  return `${number.toLocaleString(undefined, { minimumFractionDigits: 0, maximumFractionDigits: 1 })}%`;
}

function formatSeconds(value) {
  const seconds = Number(value);
  if (!Number.isFinite(seconds)) return "--";
  if (seconds < 60) return `${Math.round(seconds)}s`;
  if (seconds < 3600) return `${Math.floor(seconds / 60)}m ${Math.round(seconds % 60)}s`;
  return `${Math.floor(seconds / 3600)}h ${Math.floor((seconds % 3600) / 60)}m`;
}

function formatTime(value) {
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return "--";
  return date.toLocaleTimeString(undefined, { hour: "2-digit", minute: "2-digit", second: "2-digit" });
}

function setStatus(message, isError = false) {
  const node = el("status");
  node.textContent = message || "";
  node.classList.toggle("error", isError);
}

function setPanelState(panelId, stateId, label, tone = "") {
  const panel = el(panelId);
  panel.classList.toggle("active", tone === "active");
  panel.classList.toggle("warning", tone === "warning");
  panel.classList.toggle("error", tone === "error");
  el(stateId).textContent = label;
}

function activityLabel(event) {
  if (!event) return "--";
  return event.action || [event.method, event.path].filter(Boolean).join(" ") || "--";
}

function heartbeatLabel(heartbeat) {
  if (!heartbeat || !Number(heartbeat.count)) return "Heartbeat --";
  const parts = [`Heartbeat ${heartbeat.status || "--"}`];
  if (Number.isFinite(Number(heartbeat.duration_ms))) parts.push(formatDuration(heartbeat.duration_ms));
  if (Number.isFinite(Number(heartbeat.idle_seconds))) parts.push(`${formatSeconds(heartbeat.idle_seconds)} ago`);
  return parts.join(" · ");
}

function currentActionLabel(clients) {
  const activeClient = clients.find((client) => client.current_action);
  return activeClient?.current_action || "idle";
}

function textCell(row, value, className = "") {
  const cell = document.createElement("td");
  cell.textContent = value === null || value === undefined || value === "" ? "--" : String(value);
  if (className) cell.className = className;
  row.append(cell);
  return cell;
}

function pillCell(row, label, className = "") {
  const cell = document.createElement("td");
  const pill = document.createElement("span");
  pill.className = `pill ${className}`.trim();
  pill.textContent = label || "--";
  cell.append(pill);
  row.append(cell);
  return cell;
}

function emptyRow(colspan, message) {
  const row = document.createElement("tr");
  const cell = document.createElement("td");
  cell.className = "empty-cell";
  cell.colSpan = colspan;
  cell.textContent = message;
  row.append(cell);
  return row;
}

function renderMetrics(snapshot) {
  const totals = snapshot.totals || {};
  const process = snapshot.process || {};
  const systemMemory = process.system_memory || {};
  const heartbeat = snapshot.heartbeat || {};
  const clients = snapshot.clients || [];
  const recentActivity = snapshot.recent_activity || [];
  const lastEvent = recentActivity[0] || null;
  const slowestAction = totals.slowest_recent_action || null;
  const recentErrorCount = Number(totals.recent_error_count || 0);
  const inFlight = Number(totals.in_flight || 0);
  const systemMemoryUsed = Number(systemMemory.used_percent);

  el("serverRamMetric").textContent = formatMegabytes(process.rss_mb);
  el("serverUssMetric").textContent = formatMegabytes(process.uss_mb);
  el("peakRamMetric").textContent = formatMegabytes(process.peak_rss_mb);
  el("processCpuMetric").textContent = formatPercent(process.cpu_percent);
  el("systemCpuMetric").textContent = `system ${formatPercent(process.system_cpu_percent)}`;
  el("activeClientsMetric").textContent = formatNumber(totals.active_clients);
  el("inFlightMetric").textContent = formatNumber(inFlight);
  el("appActionsMetric").textContent = formatNumber(totals.app_actions);
  el("totalRequestsMetric").textContent = formatNumber(totals.total_requests);
  el("errorsMetric").textContent = formatNumber(totals.errors);
  el("currentActionMetric").textContent = currentActionLabel(clients);
  el("lastActionMetric").textContent = activityLabel(lastEvent);
  el("lastDurationMetric").textContent = lastEvent ? `${formatDuration(lastEvent.duration_ms)} · ${lastEvent.status || "--"}` : "--";
  el("slowestActionMetric").textContent = slowestAction ? `${activityLabel(slowestAction)} · ${formatDuration(slowestAction.duration_ms)}` : "--";
  el("errorRateMetric").textContent = formatPercent(totals.recent_error_rate);
  el("threadCountMetric").textContent = formatNumber(process.thread_count);
  const heartbeatNode = el("heartbeatMeta");
  heartbeatNode.textContent = heartbeatLabel(heartbeat);
  heartbeatNode.classList.toggle("error", Number(heartbeat.status) >= 400);
  const processDetails = [
    process.pid ? `PID ${process.pid}` : "",
    Number.isFinite(Number(process.memory_percent)) ? `Process RAM ${formatPercent(process.memory_percent)}` : "",
    Number.isFinite(Number(systemMemory.used_percent)) ? `System RAM ${formatPercent(systemMemory.used_percent)}` : "",
    Number.isFinite(Number(systemMemory.total_mb)) ? `Total RAM ${formatGigabytesFromMegabytes(systemMemory.total_mb)}` : "",
  ].filter(Boolean);
  el("processMemoryMeta").textContent = processDetails.length ? processDetails.join(" · ") : "PID --";
  if (systemMemoryUsed >= 95) {
    setPanelState("serverPanel", "serverState", "critical", "error");
  } else if (systemMemoryUsed >= 85) {
    setPanelState("serverPanel", "serverState", "memory high", "warning");
  } else {
    setPanelState("serverPanel", "serverState", "steady");
  }
  setPanelState("activityPanel", "activityState", inFlight ? "running" : "idle", inFlight ? "active" : "");
  setPanelState("performancePanel", "performanceState", recentErrorCount ? "errors" : "ok", recentErrorCount ? "error" : "");
  el("uptimeMeta").textContent = `Uptime ${formatSeconds(snapshot.uptime_seconds)} · Updated ${formatTime(snapshot.now)}`;
  el("clientMeta").textContent = `${formatNumber(totals.clients)} tracked · ${formatNumber(totals.active_clients)} active`;
  el("activityMeta").textContent = `${formatNumber(recentActivity.length)} shown`;
}

function renderClients(clients) {
  const body = el("clientsBody");
  body.replaceChildren();
  if (!clients.length) {
    body.append(emptyRow(10, "No clients yet"));
    return;
  }

  clients.forEach((client) => {
    const row = document.createElement("tr");
    textCell(row, client.client_ip);
    textCell(row, client.user_agent_label || client.user_agent, "client-browser").title = client.user_agent || "";
    const stateLabel = client.in_flight ? `${client.in_flight} running` : (client.active ? "active" : "idle");
    pillCell(row, stateLabel, client.in_flight || client.active ? "active" : "");
    textCell(row, formatNumber(client.request_count));
    textCell(row, formatNumber(client.error_count));
    textCell(row, client.current_action);
    textCell(row, client.last_app_action);
    pillCell(row, client.last_status, Number(client.last_status) >= 400 ? "error" : "ok");
    textCell(row, formatDuration(client.last_duration_ms));
    textCell(row, `${formatSeconds(client.idle_seconds)} ago`);
    body.append(row);
  });
}

function renderActivity(activity) {
  const body = el("activityBody");
  body.replaceChildren();
  if (!activity.length) {
    body.append(emptyRow(7, "No requests yet"));
    return;
  }

  activity.forEach((event) => {
    const row = document.createElement("tr");
    textCell(row, formatTime(event.timestamp));
    textCell(row, event.client_ip);
    textCell(row, event.method);
    textCell(row, event.path);
    textCell(row, event.action);
    pillCell(row, event.status, event.error ? "error" : "ok");
    textCell(row, formatDuration(event.duration_ms));
    body.append(row);
  });
}

function renderSnapshot(snapshot) {
  renderMetrics(snapshot);
  renderClients(snapshot.clients || []);
  renderActivity(snapshot.recent_activity || []);
}

async function loadTelemetry() {
  const response = await fetch("/api/telemetry", {
    headers: {
      "Content-Type": "application/json",
      "x-lucidum-token": token,
    },
  });
  const text = await response.text();
  if (!response.ok) {
    let message = text;
    try {
      message = JSON.parse(text).detail || text;
    } catch (_) {
    }
    throw new Error(message);
  }
  return JSON.parse(text);
}

async function refreshTelemetry() {
  try {
    const snapshot = await loadTelemetry();
    renderSnapshot(snapshot);
    setStatus(state.paused ? "Paused" : "");
  } catch (error) {
    setStatus(error.message, true);
  }
}

function setPaused(paused) {
  state.paused = paused;
  el("pauseBtn").textContent = paused ? "Resume" : "Pause";
  if (paused && state.timer) {
    window.clearInterval(state.timer);
    state.timer = null;
  } else if (!paused && !state.timer) {
    state.timer = window.setInterval(refreshTelemetry, 1000);
    refreshTelemetry();
  }
  if (paused) setStatus("Paused");
}

function bindControls() {
  el("refreshBtn").addEventListener("click", refreshTelemetry);
  el("pauseBtn").addEventListener("click", () => setPaused(!state.paused));
}

bindControls();
setPaused(false);
