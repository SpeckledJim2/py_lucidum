# py_lucidum Development Notes

This document is the durable maintainer context for `py_lucidum`. The public user-facing documentation lives in `README.md`.

## Purpose

`py_lucidum` is a package-first Python app for fast local exploration of large CSV and Parquet datasets. Users launch the bundled synthetic demo with `lucidum --demo`, launch another file with `lucidum path.parquet`, or start the app from Python with `py_lucidum.serve(path)`.

The app is currently local-first: it starts FastAPI and DuckDB in the user process, serves a static browser UI, and treats the input file as fixed until reload. It is designed as a shared workbench plus independently registered tools.

## Current Architecture

- `py_lucidum.core` owns DuckDB connection management, file relation SQL, schema inference, row counts, band suggestions, filter validation, saved-filter loading, shared data-source metadata, and shared metric/weight SQL helpers.
- `py_lucidum.app` owns the FastAPI factory, shared app context, token checks, static asset serving, favicon serving, schema/reload/health/shutdown endpoints, and tool registration.
- `py_lucidum.tools.registry` is the backend source of truth for tool IDs, labels, aliases, default enablement, and registration order.
- `py_lucidum.cli` owns the `lucidum` command, free-port selection, token URL construction, background server handling for notebook-style runtimes, and browser opening.
- `py_lucidum.demo` resolves the bundled synthetic demo dataset from either the source tree or installed package resources.
- `py_lucidum.tools.column_profile` implements filtered column summary/detail profiling and routes.
- `py_lucidum.tools.line_bar` implements chart aggregation and line/bar routes.
- `py_lucidum.tools.uk_map` implements UK map aggregation and UK map routes.
- `py_lucidum.tools.glm` and `py_lucidum.tools.gbm` register lightweight shell routes for future modelling tools. They must not import optional modelling libraries until real model fitting is implemented.
- `src/py_lucidum/static/app.js` is a native ES-module bootstrap. `src/py_lucidum/static/app/main.js` owns the current app shell and existing tools, while new modelling shell UI lives in `src/py_lucidum/static/app/model-tool-shell.js`.

Tool code should depend on `core` and the app registration context, but tools should not depend on each other. Shared behavior should move into `core` or another shared module only when there is real reuse.

## Public Interfaces

- CLI:
  - `lucidum --demo`
  - `lucidum path/to/data.parquet`
  - `lucidum path/to/data.csv`
  - common options include `--open`, `--host`, `--port`, `--no-token`, `--x`, `--actual`, `--expected`, `--denominator`, `--filters`, `--no-filters`, `--kpis`, `--no-kpis`, `--tools`, and UK map column overrides.
- Python:
  - `py_lucidum.serve(...)`
  - `py_lucidum.serve_line_bar(...)`
  - `py_lucidum.run_app(...)`
  - `py_lucidum.demo_dataset_path()`
  - `py_lucidum.app.create_app(...)`
- HTTP:
  - `GET /api/schema`
  - `GET /api/health`
  - `GET /api/lucidum-servers`
  - `POST /api/reload`
  - `POST /api/shutdown`
  - `POST /api/lucidum-servers/stop`
  - `POST /api/column-profile/summary`
  - `POST /api/column-profile/detail`
  - `POST /api/chart`
  - `POST /api/line-bar/chart`
  - `POST /api/uk-map/summary`
  - `GET /api/glm/summary`
  - `GET /api/gbm/summary`

`/api/chart` is retained for compatibility with the current frontend. New integrations should prefer the namespaced line-bar endpoint.

## Behavior Contracts

**Datasets and packaging**

- The committed demo dataset is `datasets/motor_premiums.parquet`.
- The wheel packages the demo dataset as `py_lucidum/datasets/motor_premiums.parquet`.
- Other local files under `datasets/` remain ignored.
- Parquet is the preferred working format for speed; CSV remains supported for convenience.
- `GET /api/schema` includes `data_sources`. The only current source is `dataset`; future model outputs should publish named tabular artifacts through this same contract.
- Line/Bar accepts a `source` request field and defaults it to `dataset`. Unknown sources are rejected before query execution.

**Defaults, saved filters, and KPIs**

- Without explicit defaults, the x-axis starts with the first dataset column, Actual starts with the first numeric column, and Expected starts as none.
- CLI options, programmatic defaults, and URL parameters can override initial selections.
- Saved filters load from an explicit `--filters` path, otherwise `./filter_spec.csv`, otherwise `./specs/filter_spec.csv`.
- Saved-filter CSV files must have exactly `theme,name,expression` columns. CSV order controls theme order and row order.
- KPI specs load from an explicit `--kpis` path, otherwise `./kpi_spec.csv`, otherwise `./specs/kpi_spec.csv`.
- KPI spec CSV files must have exactly `group,name,actual,denominator,decimals,format` columns. `denominator` aliases `N`, `Average row value`, empty, and `__none__` all mean average row value; `format` is `number`, `currency`, or `percent`. Percent formatting displays proportions as percentages, so `0.1` displays as `10%`.
- KPI rows are a single-selection convenience layer over Actual and Weight. Manual Actual/Weight changes keep the KPI row active only when both selects exactly match a spec row.
- KPI decimals and format apply to Actual and Expected response values in metric titles, line/bar labels and response axes, table response cells, map labels/tooltips/popups, and map legend values. Weight and row-count formatting is unchanged.
- The free-form DuckDB filter expression lives in the collapsible footer; hiding the footer does not clear the active filter.
- Saved-filter rows support `Single`, `Multi`, and `Grouped` modes. `Single` keeps one selected row, `Multi` toggles rows and combines them with the active All/Any/Not all/None operator, and `Grouped` toggles rows while combining rows within a theme with `OR` and selected themes with `AND`.
- `--no-filters` disables saved-filter discovery.
- Filters are DuckDB `WHERE` expressions and apply before column profiling, chart aggregation, map aggregation, table rendering, low-weight grouping, response transforms, and sigma calculations.

**Column profile**

- Column profile is the first default tool when enabled.
- Summary requests return every dataset column with inferred kind, DuckDB type, filtered missing count, exact distinct count, and min/max for numeric/date-like columns.
- Detail requests return value counts for categorical columns and histogram/stat tables for numeric/date-like columns.
- Profile requests respect the same active footer/saved-filter expression as the other tools.

**Line and bar chart**

- X-axis features can be integer, numeric, string/categorical, date, or datetime.
- Numeric banding floors values to the selected band width by default. With `quantileMode` enabled, the same banding value is rounded and clamped to `1..1000`, non-missing numeric values are grouped as `Q1`, `Q2`, etc., and missing x-axis values stay in a separate `Missing` group.
- Date/datetime axes use calendar buckets.
- Actual and Expected lines use a shared denominator. `Average row value` divides by valid row count; a numeric Weight column divides by `SUM(weight)`.
- Low-weight grouping uses selected Weight total, not raw row count.
- Table output renders directly up to 1,000 rows; larger results paginate client-side.
- Chart requests allow up to 10,000 x-axis groups before backend grouping limits apply.

**UK mapping**

- Area and sector layers join grouped KPI summaries to bundled GeoJSON assets.
- Default join columns are `PostcodeArea`, `PostcodeSector`, and `PostcodeUnit`; uppercase aliases are supported.
- Default coordinate columns are `lat` and `long`; `latitude`/`LATITUDE` and `longitude`/`LONGITUDE` aliases are supported.
- Unit points group by postcode unit, average coordinates, and plot only units with valid KPI and valid coordinates.
- Area and sector geometry use Leaflet GeoJSON with hover tooltips and click popups.
- Unit points render on a canvas-backed Leaflet layer with a hit grid for hover tooltips and click popups. Unit redraws intentionally project rows first and then apply pixel-space culling; a geographic viewport prefilter before projection is not part of the current rendering strategy because it did not improve observed redraw speed during testing.
- If no unit point columns are configured and defaults are absent, the Units layer is disabled. Explicit invalid unit point columns produce validation errors when requested.

**GLM and GBM shells**

- GLM and GBM are opt-in tools (`--tools glm,gbm`) and are not part of the default user-facing tool set.
- Shell endpoints return `status: "not_implemented"` and exist only to prove registration, routing, telemetry labelling, and frontend navigation do not require modelling dependencies.
- Real modelling dependencies should be introduced as optional extras when the fitting implementation is added, not as base install dependencies.

**Performance timings**

- The footer shows approximate diagnostic timings for the active tool, for example `DuckDB`, `JSON`, tool render time, and `Total`.
- Timing values can use `ns`, `us`, or `ms` depending on duration.
- `DuckDB` is measured on the Python server for the active tool API request. UK maps use a route-local DuckDB execute/fetch timer so the footer can show whether the database query is the bottleneck.
- This does not include browser-to-server network latency, JSON transfer or parsing, profile table rendering, chart drawing, map drawing, GeoJSON loading, or map tile loading.
- `Profile render`, `Chart render`, and `Map render` are measured in the browser after data arrives. All tools also show `JSON` and `Total`; `Total = DuckDB + JSON + render` using the rounded millisecond values shown in the footer.
- Cached UI rerenders can update render timing without running a new DuckDB query, so DuckDB may show the last cached query time. Collapsing the filter footer hides the timing monitor with the filter input.

**Local server behavior**

- CLI launches use token-protected URLs by default.
- `--no-token` disables token checks for local-only use.
- In notebook-style runtimes with an existing event loop, `serve()` and `run_app()` start the Uvicorn server in a background thread and return the URL.
- In a normal terminal or Python shell, server calls block until stopped.
- The browser Stop app button calls `POST /api/shutdown`; health polling greys out the page after server shutdown.

## UI Direction

- Keep the app dense, utilitarian, and work-focused.
- Preserve chart space; controls should stay compact and avoid unnecessary wrapping.
- The sidebar is resizable so users can trade space between long column names and the chart.
- Response controls sit above the x-axis feature list because response selection is usually the first workflow choice.
- Chart/Table controls sit before the filter bar.
- Saved-filter selections populate and apply the filter expression immediately. Manual filter edits require Enter or Apply.
- Chart animations are disabled for fast interaction.
- The app should continue to work as a static ECharts and Leaflet frontend unless future tool complexity justifies a larger frontend framework.

## Testing

Standard checks before committing:

```bash
.venv/bin/python -m unittest discover -s tests
.venv/bin/python -m compileall src tests
node --check src/py_lucidum/static/app.js
node --check src/py_lucidum/static/app/main.js
node --check src/py_lucidum/static/app/model-tool-shell.js
git diff --check
```

Optional full browser smoke check:

```bash
PY_LUCIDUM_RUN_BROWSER_TESTS=1 .venv/bin/python -m pytest
```

Optional `pipx` install check:

```bash
PY_LUCIDUM_RUN_PIPX_INSTALL_TESTS=1 .venv/bin/python -m pytest tests/test_pipx_install.py
```

Use `PY_LUCIDUM_PIPX_PYTHON=python3.13` when the default `pipx` interpreter is not Python 3.13.

The current test suite should cover:

- CLI argument behavior, token URL construction, and demo dataset selection.
- Demo dataset path resolution from source and package resources.
- Static asset serving, favicon behavior, health checks, reload, and shutdown.
- Column profile summary/detail routes, filter handling, distinct/missing counts, histograms, and entropy scores.
- Line-and-bar aggregation, filters, transforms, grouping, sorting, saved filters, CSV reads, and Parquet reads.
- UK map area, sector, and unit aggregation, alias defaults, coordinate validation, and custom column defaults.
- Tool registry defaults, optional GLM/GBM shell registration, and the default `dataset` data-source contract.
- Browser smoke behavior for loading profile, chart, and map tools without unexpected extra API requests.

## Future Work

- GLM and GBM shells should become independently registered modelling tools without coupling directly to Line/Bar internals.
- Future modelling routes, query code, and frontend assets should live inside their tool packages unless shared behavior emerges.
- Model outputs that need plotting should publish tabular artifacts through the shared data-source contract so Line/Bar can plot them without knowing model-specific concepts such as SHAP, residuals, or lift tables.
- Performance tests should be opt-in and target generated large datasets where practical, measuring schema load, aggregation time, repeat-query time, memory use, returned row count, and payload size.
- License checks should verify runtime and frontend dependencies are compatible with public distribution.
- React/Vite or another frontend framework can be reconsidered later if the static frontend becomes a maintenance constraint.

## Maintenance Rules

- Before committing:
  - Check `git status --short` and make sure new files, deletions, and generated artifacts are intentional.
  - Update `README.md` if the change affects public setup, launch commands, user workflows, CLI options, Python usage, demo data, or visible behavior.
  - Update this file if the change affects architecture, behavior contracts, testing policy, packaging, data handling, or tool-extension guidance.
  - Run the standard checks in the Testing section, plus the browser smoke check for frontend or app-launch behavior changes.
  - Scan staged changes for secrets, real customer data, local-only paths, and stale references to removed files or old demo datasets.
- Update `README.md` for public user-facing behavior changes.
- Update this file when architecture, behavior contracts, testing policy, packaging, or tool-extension guidance changes.
- Keep generated caches, local datasets other than the synthetic demo, virtual environments, build artifacts, and OS metadata out of git.
- Do not commit real customer data. The bundled motor premiums dataset is synthetic.
