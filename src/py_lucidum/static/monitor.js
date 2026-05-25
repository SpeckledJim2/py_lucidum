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
let stoppedOverlayShown = false;
let faviconDataUrl = "";
const el = (id) => document.getElementById(id);

async function cacheShutdownIcon() {
  try {
    const response = await fetch("/favicon.ico", { cache: "force-cache" });
    if (!response.ok) return;
    const blob = await response.blob();
    faviconDataUrl = await new Promise((resolve, reject) => {
      const reader = new FileReader();
      reader.addEventListener("load", () => resolve(String(reader.result || "")), { once: true });
      reader.addEventListener("error", () => reject(reader.error), { once: true });
      reader.readAsDataURL(blob);
    });
  } catch (_) {
    faviconDataUrl = "";
  }
}

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

function escapeHtml(value) {
  return String(value ?? "").replace(/[&<>"']/g, (char) => ({
    "&": "&amp;",
    "<": "&lt;",
    ">": "&gt;",
    '"': "&quot;",
    "'": "&#39;",
  })[char]);
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

function buttonCell(row, button) {
  const cell = document.createElement("td");
  cell.append(button);
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
  el("serverPanelTitle").textContent = process.pid ? `Server (PID ${process.pid})` : "Server";
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
    Number.isFinite(Number(process.memory_percent)) ? `Process RAM ${formatPercent(process.memory_percent)}` : "",
    Number.isFinite(Number(systemMemory.used_percent)) ? `System RAM ${formatPercent(systemMemory.used_percent)}` : "",
    Number.isFinite(Number(systemMemory.total_mb)) ? `Total RAM ${formatGigabytesFromMegabytes(systemMemory.total_mb)}` : "",
  ].filter(Boolean);
  el("processMemoryMeta").textContent = processDetails.length ? processDetails.join(" · ") : "Process RAM --";
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

function listenerLabel(listeners) {
  if (!Array.isArray(listeners) || !listeners.length) return "--";
  return listeners.map((listener) => {
    const host = listener.host || "localhost";
    return listener.port ? `${host}:${listener.port}` : host;
  }).join(", ");
}

function listenerUrl(listener) {
  if (!listener?.port) return "";
  let host = listener.host || "127.0.0.1";
  if (host === "0.0.0.0" || host === "::" || host === "[::]") host = "127.0.0.1";
  const bracketedHost = host.includes(":") && !host.startsWith("[") ? `[${host}]` : host;
  return `http://${bracketedHost}:${listener.port}/`;
}

function serverUrl(server) {
  if (server.display_url) return server.display_url;
  const listeners = Array.isArray(server.listeners) ? server.listeners : [];
  return listenerUrl(listeners[0]);
}

function serverHref(server) {
  const url = serverUrl(server);
  if (!url || !server.current || !token) return url;
  try {
    const href = new URL(url, window.location.href);
    href.searchParams.set("token", token);
    return href.toString();
  } catch (_) {
    return url;
  }
}

function serverTitle(server) {
  return serverUrl(server) || server.dataset || server.command || `PID ${server.pid || "--"}`;
}

function serverCell(row, server, title) {
  const cell = document.createElement("td");
  cell.className = "server-title";
  const href = serverHref(server);
  if (href) {
    const link = document.createElement("a");
    link.className = "server-link";
    link.href = href;
    link.target = "_blank";
    link.rel = "noopener noreferrer";
    link.textContent = title;
    cell.append(link);
  } else {
    cell.textContent = title || "--";
  }
  row.append(cell);
  return cell;
}

function renderLucidumServers(payload) {
  const servers = payload?.servers || [];
  const body = el("lucidumServersBody");
  body.replaceChildren();
  el("lucidumServersMeta").textContent = `${formatNumber(payload?.count ?? servers.length)} running`;
  if (!servers.length) {
    body.append(emptyRow(5, "No lucidum servers found"));
    return;
  }

  servers.forEach((server) => {
    const row = document.createElement("tr");
    row.classList.toggle("current-server-row", Boolean(server.current));
    const title = serverTitle(server);
    const titleCell = serverCell(row, server, title);
    titleCell.title = [server.dataset_path, server.command].filter(Boolean).join(" · ");
    textCell(row, listenerLabel(server.listeners), "server-listeners");
    const pidText = server.current ? `${server.pid} current` : server.pid;
    textCell(row, pidText);
    textCell(row, server.create_time ? formatTime(server.create_time * 1000) : "--");
    const button = document.createElement("button");
    button.className = "server-stop-button";
    button.type = "button";
    button.textContent = "X";
    button.disabled = !server.stoppable;
    button.title = server.stoppable ? `Stop PID ${server.pid}` : "Cannot stop this server from the monitor";
    button.setAttribute("aria-label", button.title);
    button.dataset.pid = String(server.pid || "");
    button.dataset.createTime = String(server.create_time || "");
    buttonCell(row, button);
    body.append(row);
  });
}

function renderSnapshot(snapshot, serversPayload) {
  renderMetrics(snapshot);
  renderLucidumServers(serversPayload);
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

async function loadLucidumServers() {
  const response = await fetch("/api/lucidum-servers", {
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

async function postJson(path, payload) {
  const response = await fetch(path, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      "x-lucidum-token": token,
    },
    body: payload === undefined ? undefined : JSON.stringify(payload),
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
  return text ? JSON.parse(text) : {};
}

async function refreshTelemetry() {
  try {
    const [snapshot, serversPayload] = await Promise.all([loadTelemetry(), loadLucidumServers()]);
    renderSnapshot(snapshot, serversPayload);
    setStatus(state.paused ? "Paused" : "");
  } catch (error) {
    setStatus(error.message, true);
  }
}

function syncPauseButton() {
  const button = el("pauseBtn");
  const label = state.paused ? "Resume polling" : "Pause polling";
  button.classList.toggle("paused", state.paused);
  button.setAttribute("aria-label", label);
  button.title = label;
}

function setPaused(paused) {
  state.paused = paused;
  syncPauseButton();
  if (paused && state.timer) {
    window.clearInterval(state.timer);
    state.timer = null;
  } else if (!paused && !state.timer) {
    state.timer = window.setInterval(refreshTelemetry, 1000);
    refreshTelemetry();
  }
  if (paused) setStatus("Paused");
}

function syncThemeButton() {
  const button = el("themeBtn");
  const label = document.body.classList.contains("dark") ? "Switch to light mode" : "Switch to dark mode";
  button.setAttribute("aria-label", label);
  button.title = label;
}

function confirmStopApp(message = "Stop the local lucidum server?") {
  return new Promise((resolve) => {
    const overlay = document.createElement("div");
    overlay.className = "stop-confirm-overlay";
    overlay.innerHTML = `
      <div class="stop-confirm-dialog" role="dialog" aria-modal="true" aria-labelledby="stopConfirmTitle">
        <div class="stop-confirm-content">
          <img class="stop-confirm-icon" src="/favicon.ico" alt="">
          <p id="stopConfirmTitle">${escapeHtml(message)}</p>
        </div>
        <div class="stop-confirm-actions">
          <button class="ghost stop-confirm-cancel" type="button">Cancel</button>
          <button class="ghost stop-confirm-ok" type="button">OK</button>
        </div>
      </div>
    `;
    const cancelButton = overlay.querySelector(".stop-confirm-cancel");
    const okButton = overlay.querySelector(".stop-confirm-ok");
    let closed = false;
    const close = (confirmed) => {
      if (closed) return;
      closed = true;
      window.removeEventListener("keydown", handleKeydown);
      overlay.remove();
      resolve(confirmed);
    };
    function handleKeydown(event) {
      if (event.key === "Escape") close(false);
    }
    cancelButton.addEventListener("click", () => close(false));
    okButton.addEventListener("click", () => close(true));
    window.addEventListener("keydown", handleKeydown);
    document.body.append(overlay);
    cancelButton.focus();
  });
}

function showStoppedOverlay() {
  if (stoppedOverlayShown) return;
  stoppedOverlayShown = true;
  if (state.timer) {
    window.clearInterval(state.timer);
    state.timer = null;
  }
  document.body.classList.add("app-stopped");
  const shutdownIcon = faviconDataUrl
    ? `<img class="shutdown-icon" src="${faviconDataUrl}" alt="">`
    : '<span class="shutdown-icon shutdown-icon-fallback" aria-hidden="true"></span>';
  const overlay = document.createElement("div");
  overlay.className = "shutdown-overlay";
  overlay.innerHTML = `
    <div class="shutdown-message" role="status" aria-live="polite">
      ${shutdownIcon}
      <div>
        <h1>lucidum has stopped</h1>
        <p>The local server is no longer running. You can close this browser tab.</p>
      </div>
    </div>
  `;
  document.body.append(overlay);
}

async function stopApp() {
  if (!(await confirmStopApp())) return;
  const button = el("stopAppBtn");
  button.disabled = true;
  setStatus("Stopping app...");
  try {
    await postJson("/api/shutdown");
    showStoppedOverlay();
  } catch (error) {
    button.disabled = false;
    setStatus(error.message, true);
  }
}

async function stopLucidumServer(server) {
  const title = serverTitle(server);
  if (!(await confirmStopApp(`Stop lucidum server ${title} (PID ${server.pid})?`))) return;
  setStatus(`Stopping PID ${server.pid}...`);
  try {
    await postJson("/api/lucidum-servers/stop", { pid: server.pid, create_time: server.create_time });
    if (server.current) {
      showStoppedOverlay();
      return;
    }
    await refreshTelemetry();
  } catch (error) {
    setStatus(error.message, true);
  }
}

function bindControls() {
  el("pauseBtn").addEventListener("click", () => setPaused(!state.paused));
  el("themeBtn").addEventListener("click", () => {
    document.body.classList.toggle("dark");
    syncThemeButton();
  });
  el("stopAppBtn").addEventListener("click", stopApp);
  el("lucidumServersBody").addEventListener("click", async (event) => {
    const button = event.target?.closest?.(".server-stop-button");
    if (!button || button.disabled) return;
    await stopLucidumServer({
      pid: Number(button.dataset.pid),
      create_time: Number(button.dataset.createTime),
      current: Boolean(button.closest("tr")?.classList.contains("current-server-row")),
      display_url: button.closest("tr")?.querySelector(".server-title")?.textContent || "",
    });
  });
}

bindControls();
cacheShutdownIcon();
syncThemeButton();
setPaused(false);
