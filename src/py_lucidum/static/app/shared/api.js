export function createApiClient({ token, fetchImpl = fetch, performanceImpl = performance }) {
  return async function api(path, options = {}) {
    const { clientTiming = false, operationId = "", ...fetchOptions } = options;
    const started = performanceImpl.now();
    const response = await fetchImpl(path, {
      ...fetchOptions,
      headers: {
        "Content-Type": "application/json",
        "x-lucidum-token": token,
        ...(operationId ? { "x-lucidum-operation-id": operationId } : {}),
        ...(fetchOptions.headers || {}),
      },
    });
    const responseReady = performanceImpl.now();
    const text = await response.text();
    const bodyReady = performanceImpl.now();
    if (!response.ok) {
      let message = text;
      try {
        message = JSON.parse(text).detail || text;
      } catch (_) {
      }
      throw new Error(message);
    }
    const parseStarted = performanceImpl.now();
    const data = JSON.parse(text);
    const parsed = performanceImpl.now();
    if (clientTiming && data && typeof data === "object") {
      data.client_timings = {
        response_ms: responseReady - started,
        body_ms: bodyReady - responseReady,
        parse_ms: parsed - parseStarted,
        data_ms: parsed - responseReady,
        total_ms: parsed - started,
      };
    }
    return data;
  };
}

export function createOperationId(prefix = "operation") {
  const safePrefix = String(prefix || "operation")
    .toLowerCase()
    .replace(/[^a-z0-9._-]+/g, "-")
    .replace(/^-+|-+$/g, "")
    .slice(0, 24) || "operation";
  const randomPart = globalThis.crypto?.randomUUID?.().replaceAll("-", "")
    || `${Date.now().toString(36)}${Math.random().toString(36).slice(2, 14)}`;
  return `${safePrefix}-${randomPart}`.slice(0, 80);
}

export function monitorPath({ token, href }) {
  const url = new URL("/monitor", href);
  if (token) url.searchParams.set("token", token);
  return `${url.pathname}${url.search}`;
}
