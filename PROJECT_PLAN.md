# py_lucidum Project Plan

## Summary

Build `py_lucidum` as an open-source Python package for fast local exploration of 1m-10m row CSV/Parquet files. Users will launch it from the CLI as `lucidum path.parquet` or from Python as `lucidum.serve(path)`, receiving a local URL with an access token that can optionally be shared on the LAN.

Use a FastAPI backend with DuckDB for live on-the-fly aggregation, and a React/Vite browser UI using Apache ECharts for charts and TanStack Table for the tabular view. No precomputed aggregate cube; cache only metadata and identical repeat query results within a session.

## Key Changes

- Create a package-first repo that supports both local analyst mode and internal server mode: `pyproject.toml`, `src/py_lucidum/`, CLI entry point `lucidum`, importable ASGI app factory, tests, and frontend app.
- Backend API:
  - `GET /api/schema`: file snapshot, columns, inferred types, row count, basic summaries.
  - `POST /api/chart`: aggregated chart/table data for selected x-axis, y metrics, chart type, bins, tails, and rare-category grouping.
  - `POST /api/reload`: refresh file snapshot, schema, and caches.
- Dataset inputs:
  - Single CSV/Parquet file or same-schema glob/folder.
  - Local disk only for v1.
  - Snapshot source files at launch; refresh only through reload.
- Sharing:
  - Default bind to localhost.
  - Optional LAN bind with generated tokenized URL.
- UI direction:
  - Use the existing R Lucidum line-and-bar tool as product inspiration, but build a cleaner modern browser UI rather than copying it directly.
  - Include a dark mode option.
  - Keep Histogram, SHAP, filters, feature groups, and model-object loading out of v1.

## Chart Behavior

- X-axis supports numeric, character/string, categorical/factor-like, date, and datetime columns.
- Chart types: line and bar, user selectable.
- For character and categorical x-axis features, support sorting by alphabetical order, bar volume, and response value.
- Y-axis defaults to row count and also supports one or more response metrics. A response can be a numerator-only metric or a numerator divided by a denominator, such as conversion rate (`sum(sales) / count(*)`) or frequency (`sum(claims) / sum(vehicle_years)`).
- Numeric y-axis columns support selected aggregations: sum, mean, min, max, median, and p90.
- “Additional lines” means additional y metrics on the same x-axis; for bar charts, render grouped bars.
- Response transforms apply after aggregation for display: none, log, exp, logit, zero-centred, and one-scaled. Zero-centred subtracts the chart-level average response so the displayed average is 0; one-scaled divides by the chart-level average response so the displayed average is 1. For ratio responses, compute the chart-level average as total numerator divided by total denominator; show a validation message for invalid transform domains such as log of non-positive values or logit outside `(0, 1)`.
- Numeric binning floors values to a configurable width such as `0.1`, `0.5`, `1`, `2`, `5`, `10`.
- Date/datetime binning uses calendar buckets: hour, day, week, month, year.
- Tail or low-weight grouping is configurable: no grouping, absolute low-volume thresholds such as `5` or `10`, percentage thresholds such as `0.1%` or `1%`, and custom thresholds. For ordered numeric/date x axes, collapse low-volume bins at the low and high ends into tail buckets. For character/categorical x axes, collapse low-volume levels into an “Other” bucket. Tail/grouped bars should be coloured slightly differently to highlight the grouping.
- Provide an option to show prettified labels on the chart for bars, lines, or both.
- Sigma bars are optional and only shown when two comparable responses are selected: the first is treated as actual and the second as expected. Error bars are drawn around the expected response only, to help users judge whether the actual-vs-expected gap is material relative to empirical volatility within each x-axis group.
- Sigma-bar calculation should use deterministic hash folds within each x-axis group rather than random splits. Default to 20 folds, compute the actual-minus-expected response gap in each valid fold, estimate the standard error as `stddev(fold_gap) / sqrt(valid_fold_count)`, and draw expected +/- the selected sigma multiplier. Suppress sigma bars for groups with too few valid folds or zero denominators.
- Table view shows the same aggregated result as the current plot, not raw full-file rows.

## Test Plan

- Unit tests prove query correctness using small deterministic generated datasets. Cover CSV and Parquet reads, numeric bins, date buckets, ordered-axis tail grouping, categorical “Other” grouping, count responses, numerator-only responses, numerator/denominator responses, response transforms, multiple metrics, sort order, and sigma-bar calculations.
- Backend integration tests exercise the FastAPI app against temporary generated files. Cover `GET /api/schema`, `POST /api/chart`, `POST /api/reload`, request validation, token rejection, cache invalidation, and repeat-query consistency.
- Browser smoke tests use Playwright against the running app. Cover first load, feature selection, chart/table switching, dark mode, x-axis sorting, response metric controls, label toggles, valid sigma-bar display, and user-facing messages for invalid transform domains.
- Performance tests are opt-in rather than part of the default fast test suite. Target generated 1m, 5m, and 10m row local datasets where practical, measuring cold schema load time, cold aggregation time, warm repeat-query time, memory use, returned row count, and chart payload size.
- License check verifies all runtime and frontend dependencies are open-source compatible.

## Assumptions

- Project name is `py_lucidum`; installed command is `lucidum`.
- Preferred stack is FastAPI + DuckDB + Apache ECharts, with React/Vite for the UI.
- Repeat-query caching is allowed, but no persistent pre-aggregation or materialized cube is used.
- Files are treated as fixed during a session until the user presses reload.
- Initial open-source references: DuckDB MIT/open-source status, FastAPI MIT, Apache ECharts Apache-2.0, Vite MIT, TanStack Table MIT, pytest MIT.
