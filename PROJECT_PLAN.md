# py_lucidum Project Plan

## Summary

`py_lucidum` is an internal company data science tool for fast local exploration of large CSV and Parquet datasets. Users can launch it from the command line as `lucidum path.parquet` or from Python via `lucidum.serve(path)`, then open the generated tokenized browser URL.

The current implementation is a package-first Python app with a FastAPI backend, DuckDB live aggregation, and a static browser UI using Apache ECharts. Parquet is the preferred working format for speed; CSV remains supported for convenience and import workflows. The codebase is being shaped as a shared workbench plus independently registered tools so the existing line-and-bar chart can run standalone or alongside future GLM, GBM, and UK mapping tools.

## Current Baseline

- Package structure exists with `pyproject.toml`, `src/py_lucidum/`, CLI entry point `lucidum`, an importable app factory, and a static frontend.
- The backend is split into shared `core` modules, an `app` factory package, and `tools` packages. The current implemented tool is `tools.line_bar`; placeholder packages exist for `tools.glm`, `tools.gbm`, and `tools.uk_map`.
- Repository hygiene excludes local datasets, Python build/cache artifacts, virtual environments, generated README previews, and OS metadata such as `.DS_Store`.
- Backend supports:
  - `GET /api/schema` for file path, row count, inferred column types, and numeric band suggestions.
  - `POST /api/chart` for the legacy line-and-bar chart endpoint.
  - `POST /api/line-bar/chart` for the namespaced line-and-bar chart endpoint.
  - `POST /api/reload` to refresh the file snapshot and cached metadata.
- The app supports local analyst mode today and is designed to grow into internal server mode with local or mounted server datasets.
- Local development datasets `vans.csv` and `vans.parquet` are ignored and not intended for publishing.
- Public repository documentation lives in `README.md`. Detailed launch documentation lives in `USAGE.md` and covers the CLI, module entry point, Python console usage, programmatic Uvicorn usage, LAN binding, browser opening, no-token local mode, initial selection overrides, and saved filter files.
- A small standard-library `unittest` suite now covers the line-and-bar backend routes, saved-filter loading, filter validation, aggregation behavior, and compatibility re-exports without introducing a test dependency.

## Tool Architecture

- Shared dataset responsibilities live in `py_lucidum.core`: DuckDB connection management, file relation SQL, schema inference, row counts, band suggestions, filter validation, saved-filter loading, and SQL helpers.
- The app factory lives in `py_lucidum.app` and composes selected tools around a shared dataset context. `py_lucidum.app.create_app(...)` accepts a `tools` argument; the default is `line_bar`.
- Tool code lives under `py_lucidum.tools`. A tool should depend on `core` and the app registration context, but tools should not depend on each other.
- The line-and-bar chart logic now lives in `py_lucidum.tools.line_bar`. `py_lucidum.query` remains as a compatibility re-export module for older imports, including the legacy `Dataset(...).chart(...)` path.
- Future GLM, GBM, and UK mapping work should add routes, modelling/query code, and frontend assets inside their own tool packages. They should reuse shared filters and dataset metadata rather than importing line-and-bar internals.
- Supported tool names should be normalised at the app boundary, for example `line-bar` and `line_bar` both selecting the line-and-bar chart.

## Chart Behavior

- X-axis supports integer, numeric, character/string, categorical/factor-like, date, and datetime columns.
- Integer columns are identified separately in the sidebar and integer x-axis labels are shown without `.0` suffixes.
- The main view is a combined bar and line chart:
  - Bars show row volume.
  - One or two response lines can be selected.
  - The same aggregated result can be shown as a table.
- Responses can be:
  - numerator-only averages, such as average price.
  - numerator divided by row count.
  - numerator divided by another numeric denominator, such as claims divided by vehicle years.
- Response transforms apply after aggregation for display: none, log, exp, logit, zero-centred, and one-scaled. Invalid transform domains should show a user-facing message.
- Character/categorical x-axis sorting starts alphabetically and also supports bar volume, Actual response value, and Expected response value when line 2 is selected. Sort controls are hidden for numeric/date x-axes.
- Numeric banding floors x values to the selected band width. Fixed shortcuts include `0.1`, `1`, `5`, and `10`; `<` and `>` step through the 1/2/5 ladder plus the exact useful levels `4`, `7`, and `12`.
- When an integer x-axis feature has a full-data range below 120, the app chooses initial band width `1`. Otherwise, when an integer or numeric x-axis feature is selected, the app chooses an initial band width from the feature standard deviation over the first 10k rows, rounded down two notches on the 1/2/5 scale.
- Date/datetime x-axes use calendar buckets: hour, day, week, month, and year. Date bucket controls are only shown for date/datetime features; banding controls are only shown for integer/numeric features.
- Low-weight grouping supports absolute thresholds and percentage thresholds such as `0.1%` and `1%`. Ordered numeric/date tails are collapsed into low/high tail buckets; low-volume categorical levels are collapsed into “Other”.
- Chart requests allow up to 10,000 x-axis groups before backend grouping limits apply.
- DuckDB filter expressions can be typed above the chart or loaded from `filter_spec.csv`; filters are applied before aggregation, table rendering, low-weight grouping, response transforms, and sigma calculations.
- Sigma bars are optional and shown only when two comparable responses are selected. The first response is treated as actual and the second as expected. Error bars are drawn around expected using deterministic hash folds within each x-axis group.
- Hover values, y-axis values, chart labels, and table values are formatted with comma separators and a sensible number of decimal places, independent of the underlying raw precision. Line chart labels keep fixed decimal places, so trailing zeros remain visible. Labels are hidden with one combined overlay note when the label density limit is reached.
- Table output renders directly up to 1,000 rows; larger result sets are paginated client-side in 1,000-row pages to avoid slow scrolling.
- Initial selections are data-agnostic by default: x-axis uses the first dataset column, Actual / line 1 uses the first numeric column, and Expected / line 2 starts as None. CLI options, programmatic app defaults, and URL parameters can override `x`, `actual`, and `expected`.
- Dataset operations serialize access to the shared DuckDB connection used by the local app process.

## UI Direction

- The R Lucidum line-and-bar tool is product inspiration, but the Python app should continue toward a cleaner, denser browser UI.
- The chart is the focus: controls should stay compact, avoid unnecessary wrapping, and preserve vertical space.
- Dark mode is supported.
- The browser tab icon and in-app header mark are served from the project `favicon.ico`, with PNG favicon content supported and packaging configured to include it under the app static assets. Favicon references include a cache-busting query string. The current favicon uses a point-up blue hexagon, centred white `lucidum` wordmark, and surrounding gold stars that avoid overlapping the wordmark. The in-page header mark is displayed at 42px for readability while the header bar height remains fixed.
- Chart animations are disabled so interactions update as fast as possible.
- The sidebar is resizable so users can trade space between long column names and the chart.
- Response controls sit above the x-axis feature list because response selection is usually the first choice in the workflow.
- The x-axis feature list can be shown in original dataset column order or alphabetically without changing the selected chart sort.
- Chart/Table view controls sit before the filter bar; saved-filter selections populate and apply the filter expression immediately, while manual filter edits require Enter or Apply. Chart-only density messages are shown in the chart's top-right corner instead of consuming filter-bar width.
- Table view uses compact row spacing to support scanning many grouped rows.
- Bars widen for small numbers of x-axis categories while keeping visible spacing between groups.
- Chart legend order is Actual response, Expected response when selected, then N. Actual is black in light mode and white in dark mode, Expected is red, N uses the bar colour, and grey sigma guides are never listed in the legend.
- Y-axis tick values are shown without extra axis-title text above the plot area.
- X-axis labels are always shown below the label density limit, use smaller text above 50 groups, and are hidden with a UI message when the limit is reached.
- Longer rotated labels should remain visible without excessive blank space under the plot.
- The chart should resize to fill the browser window toward the bottom-right.
- Histogram, SHAP, feature groups, and model-object loading are out of v1.

## Performance Notes

- Benchmarks on the local 416,220-row `vans` dataset showed Parquet is materially faster than CSV for the same DuckDB queries.
- Warm median timings from the current backend:
  - Schema load: CSV ~245ms, Parquet ~5ms.
  - Numeric age chart: CSV ~232ms, Parquet ~27ms.
  - Categorical MakeModel chart: CSV ~225ms, Parquet ~16ms.
  - Date month chart: CSV ~220ms, Parquet ~14ms.
- Recommendation: use Parquet as the normal working format and keep CSV support for input convenience.

## Test Plan

- Unit tests should cover query correctness using small generated datasets: CSV and Parquet reads, numeric bins, date buckets, tail grouping, categorical “Other” grouping, response ratios, transforms, sorting, and sigma-bar calculations.
- Current line-and-bar unit tests can be run with `.venv/bin/python -m unittest discover -s tests`.
- Backend integration tests should exercise `GET /api/schema`, `POST /api/chart`, and `POST /api/reload` against temporary generated files, including token rejection and validation errors.
- Browser smoke tests should cover feature selection, response selection, chart/table switching, dark mode, x-axis control visibility, band stepping, labels, sigma bars, and invalid transform messages.
- Performance tests should be opt-in and target generated 1m, 5m, and 10m row datasets where practical, measuring schema load, aggregation time, repeat-query time, memory use, returned row count, and payload size.
- License checks should verify all runtime and frontend dependencies are open-source compatible.

## Assumptions

- Project name is `py_lucidum`; installed command is `lucidum`.
- Preferred backend stack is FastAPI + DuckDB.
- Current frontend is a static ECharts app; a React/Vite frontend can be reconsidered later if UI complexity justifies it.
- Repeat-query caching is allowed, but no persistent pre-aggregation or materialized cube is used.
- Files are treated as fixed during a session until the user presses reload.
- `PROJECT_PLAN.md` should be updated alongside every product or behavior change so it remains the durable source of project context.
