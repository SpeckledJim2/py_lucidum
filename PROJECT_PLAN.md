# py_lucidum Project Plan

## Summary

Build `py_lucidum` as an open-source Python package for fast local exploration of 1m-10m row CSV/Parquet files. Users will launch it from the CLI as `lucidum path.parquet` or from Python as `lucidum.serve(path)`, receiving a local URL with an access token that can optionally be shared on the LAN.

Use a FastAPI backend with DuckDB for live on-the-fly aggregation, and a React/Vite browser UI using Apache ECharts for charts and TanStack Table for the tabular view. No precomputed aggregate cube; cache only metadata and identical repeat query results within a session.

## Key Changes

- Create a package-first repo: `pyproject.toml`, `src/py_lucidum/`, CLI entry point `lucidum`, tests, and frontend app.
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

## Chart Behavior

- X-axis supports numeric, character/string, categorical/factor-like, date, and datetime columns.
- Chart types: line and bar, user selectable.
- Y-axis defaults to row count and also supports one or more numeric metrics with selected aggregations: sum, mean, min, max, median, and p90.
- “Additional lines” means additional y metrics on the same x-axis; for bar charts, render grouped bars.
- Numeric binning floors values to a configurable width such as `0.1`, `0.5`, `1`, `2`, `5`, `10`.
- Date/datetime binning uses calendar buckets: hour, day, week, month, year.
- Numeric tail handling is configurable: no tails, 0.1%, 1%, plus custom percentile values. Values below/above thresholds are collapsed into low/high tail buckets.
- Character/categorical tail handling groups the rarest categories into one “Other” bucket using a configurable cumulative-frequency threshold.
- Table view shows the same aggregated result as the current plot, not raw full-file rows.

## Test Plan

- Unit-test query generation for CSV, Parquet, numeric bins, date buckets, percentile tails, rare-category grouping, and multiple metrics.
- Integration-test FastAPI endpoints against generated 1m-row and smaller deterministic datasets.
- Browser smoke tests verify chart/table switching, metric changes, reload, token access, and invalid token rejection.
- Performance tests target live aggregation over 1m-10m local rows, measuring cold query time, warm repeat-query cache time, memory use, and returned row counts.
- License check verifies all runtime and frontend dependencies are open-source compatible.

## Assumptions

- Project name is `py_lucidum`; installed command is `lucidum`.
- Preferred stack is FastAPI + DuckDB + Apache ECharts, with React/Vite for the UI.
- Repeat-query caching is allowed, but no persistent pre-aggregation or materialized cube is used.
- Files are treated as fixed during a session until the user presses reload.
- Initial open-source references: DuckDB MIT/open-source status, FastAPI MIT, Apache ECharts Apache-2.0, Vite MIT, TanStack Table MIT, pytest MIT.
