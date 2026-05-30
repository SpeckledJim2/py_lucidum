export function createApiClient({ token, fetchImpl = fetch, performanceImpl = performance }) {
  return async function api(path, options = {}) {
    const { clientTiming = false, ...fetchOptions } = options;
    const started = performanceImpl.now();
    const response = await fetchImpl(path, {
      ...fetchOptions,
      headers: {
        "Content-Type": "application/json",
        "x-lucidum-token": token,
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

export function monitorPath({ token, href }) {
  const url = new URL("/monitor", href);
  if (token) url.searchParams.set("token", token);
  return `${url.pathname}${url.search}`;
}
