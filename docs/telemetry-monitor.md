# Lucidum Telemetry Monitor

Lucidum includes a lightweight in-memory telemetry monitor for local server activity. It is intended for development, debugging, and shared local demos where it is useful to see which browser clients are connected and what actions they last performed.

## Opening the Monitor

Launch Lucidum with `--buttons` to show the app header's `Open monitor` button. The button opens `/monitor` in a new tab and carries the current token when token auth is enabled.

Without `--buttons`, the header button is hidden. You can still open the monitor directly by changing the app URL path:

```text
http://127.0.0.1:8000/monitor?token=...
```

The monitor uses the same token rules as the app. If Lucidum was launched with `--no-token`, no token is needed.

## Metrics

The monitor groups live signals into three panels:

- `Server`: process memory, process CPU, thread count, PID, and system memory context.
- `Activity`: active clients, in-flight requests, app actions, total requests, and errors.
- `Performance`: current action, last action duration, slowest recent app action, recent error rate, and the latest `/api/health` heartbeat.

The client table groups activity by client IP plus full browser user agent, but displays a compact browser label such as `Safari 26.5 · macOS`. The full user agent is still retained in memory and exposed in telemetry JSON for diagnostics. The table also shows request counts, error counts, current action, last app action, last status, last duration, and idle time.

The recent activity table shows the newest tracked app and API requests first. Paths are stored without query strings, so token values are not retained in telemetry. The monitor treats `/api/health` as a heartbeat line instead of normal request activity, so health checks do not inflate request totals, active clients, recent rows, last action, or error-rate calculations. Static asset loads such as `/static/...`, `/favicon.ico`, and tool images or GeoJSON are counted as diagnostics only and are excluded from the primary activity view.

The `lucidum servers` table lists local Lucidum-looking server processes for the current operating-system user. It includes the monitor's own server even if process scanning cannot discover it, highlights that current server row, shows known listeners, and uses clickable server addresses that open in a new tab with the current monitor token when token auth is enabled. Each stoppable row has a compact `X` button; stopping the current server uses the normal shutdown callback, while stopping a sibling server terminates the matching same-user Lucidum process after validating its PID and create time.

The RAM values are process-level measurements from the server process that owns the monitor page:

- `RSS`: resident memory and the headline RAM value.
- `USS`: memory mostly unique to the Lucidum process, when the operating system exposes it.
- `Peak`: highest RSS observed by Lucidum since the server started.
- `Total RAM`: installed system memory, shown in GB for context.

The telemetry API still includes VMS for diagnostic compatibility, but the monitor does not display it because virtual address space can be very large on macOS and is not real RAM pressure.

## Privacy and Persistence

Telemetry is in memory only. It resets when the Lucidum process restarts and is not written to disk.

The `lucidum servers` table is separate from telemetry. It is built from current-user process discovery plus launch metadata for the current server, and returned URLs are stripped of token query values before display.

Lucidum stores bounded request metadata only:

- client IP
- browser user agent
- request method
- request path without query parameters
- mapped app action name when applicable
- response status
- duration
- first and last seen times
- process memory measurements
- process CPU, system CPU, and thread count
- diagnostic counts for static asset loads

Lucidum does not store request bodies, dataset values, filter expressions from JSON payloads, or token query values in telemetry.

## Limits

The monitor reports HTTP request activity, not raw TCP socket state. A "connection" means a recently seen client and its in-flight requests.

Telemetry is intentionally lightweight and local-process scoped:

- It does not survive restart.
- It is not shared across multiple Lucidum processes. The `lucidum servers` table can discover sibling servers, but it does not merge their telemetry.
- It does not identify browser tabs separately when they share the same IP and user agent.
- It does not attribute RAM to individual clients, browser tabs, or requests because Lucidum runs those actions inside one shared Python process.
- Reverse proxies may hide the original client IP unless they preserve client details at the ASGI layer.

## Troubleshooting

If the monitor shows `Invalid or missing app token`, open it from the app header when Lucidum was launched with `--buttons`, or add the same `token=...` query parameter from the Lucidum app URL.

If the monitor appears idle while the app is active, check that the app tab and monitor tab are connected to the same Lucidum server URL and port.

If the monitor itself is open, its 1-second `/api/telemetry` polling is excluded from the counters by design. `/api/health` checks are also excluded from normal activity counters and appear only in the heartbeat line.
