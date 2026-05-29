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
- `py_lucidum.tools.glm` registers a lightweight shell route for future modelling work.
- `py_lucidum.tools.gbm` implements the opt-in LightGBM tool. GBM training, validation, persistence, tree summary/detail, and model-output data sources live in separate backend modules; the frontend only edits settings, starts jobs, polls status, and renders returned diagnostics.
- `src/py_lucidum/static/app.js` is a native ES-module bootstrap. `src/py_lucidum/static/app/main.js` owns the current app shell and existing tools, `src/py_lucidum/static/app/gbm-tool.js` owns the GBM frontend, `src/py_lucidum/static/app/gbm-shap-tool.js` and `src/py_lucidum/static/app/gbm-shap-chart.js` own the GBM SHAP UI/chart split, `src/py_lucidum/static/app/gbm-tree-viewer.js` owns the D3 tree viewer, and `src/py_lucidum/static/app/model-tool-shell.js` owns remaining modelling shell UI.
- Third-party browser libraries are vendored under `src/py_lucidum/static/vendor/` and lazy-loaded by the tools that need them. GBM currently uses Tabulator for editable grids, D3 for tree diagrams, and ECharts GL only for SHAP 3D surface plots.

Tool code should depend on `core` and the app registration context, but tools should not depend on each other. Shared behavior should move into `core` or another shared module only when there is real reuse.

## Public Interfaces

- CLI:
  - `lucidum --demo`
  - `lucidum path/to/data.parquet`
  - `lucidum path/to/data.csv`
  - common options include `--open`, `--host`, `--port`, `--no-token`, `--x`, `--actual`, `--expected`, `--denominator`, `--filters`, `--no-filters`, `--kpis`, `--no-kpis`, `--features`, `--no-features`, `--tools`, and UK map column overrides.
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
  - `GET /api/gbm/config`
  - `GET /api/gbm/models`
  - `POST /api/gbm/validate`
  - `POST /api/gbm/train`
  - `GET /api/gbm/jobs/{job_id}`
  - `GET /api/gbm/models/{model_id}`
  - `GET /api/gbm/models/{model_id}/shap/config`
  - `POST /api/gbm/models/{model_id}/shap/plot`
  - `GET /api/gbm/models/{model_id}/trees`
  - `GET /api/gbm/models/{model_id}/trees/{tree_index}`
  - `POST /api/gbm/models/{model_id}/activate`
  - `POST /api/gbm/models/{model_id}/rename`
  - `DELETE /api/gbm/models/{model_id}`

`/api/chart` is retained for compatibility with the current frontend. New integrations should prefer the namespaced line-bar endpoint.

## Behavior Contracts

**Datasets and packaging**

- The committed demo dataset is `datasets/motor_premiums.parquet`.
- The wheel packages the demo dataset as `py_lucidum/datasets/motor_premiums.parquet`.
- Other local files under `datasets/` remain ignored.
- Parquet is the preferred working format for speed; CSV remains supported for convenience.
- `GET /api/schema` includes `data_sources`. The default source is `dataset`; model outputs publish named tabular artifacts through this same contract.
- `GET /api/schema` excludes unreadable columns from `columns` and reports them as `invalid_columns` with sanitized errors. Normal tools should use the safe column maps; only diagnostics or choosers that explicitly report invalid columns should use the all-column map.
- Line/Bar accepts a `source` request field and defaults it to `dataset`. Unknown sources are rejected before query execution.

**Defaults, saved filters, KPIs, and feature specs**

- Without explicit defaults, the x-axis starts with the first dataset column, Actual starts with the first numeric column, and Expected starts as none.
- CLI options, programmatic defaults, and URL parameters can override initial selections.
- Saved filters load from an explicit `--filters` path, otherwise `./filter_spec.csv`, otherwise `./specs/filter_spec.csv`.
- Saved-filter CSV files must have exactly `theme,name,expression` columns. CSV order controls theme order and row order.
- KPI specs load from an explicit `--kpis` path, otherwise `./kpi_spec.csv`, otherwise `./specs/kpi_spec.csv`.
- KPI spec CSV files must have exactly `group,name,actual,denominator,decimals,format` columns. `denominator` aliases `N`, `Average row value`, empty, and `__none__` all mean average row value; `format` is `number`, `currency`, or `percent`. Percent formatting displays proportions as percentages, so `0.1` displays as `10%`.
- KPI rows are a single-selection convenience layer over Actual and Weight. Manual Actual/Weight changes keep the KPI row active only when both selects exactly match a spec row.
- KPI decimals and format apply to Actual and Expected response values in metric titles, line/bar labels and response axes, table response cells, map labels/tooltips/popups, and map legend values. Weight and row-count formatting is unchanged.
- Feature specs load from an explicit `--features` path, otherwise `./feature_spec.csv`, otherwise `./specs/feature_spec.csv`.
- Feature spec CSV files must start with exactly `Feature,Grouping`; all remaining columns are ordered GBM scenario names. Scenario cells include the feature when they contain the word `feature`, case-insensitive.
- `Grouping` is displayed in the GBM Feature table between `Feature` and `Use` and supplies the GBM feature interaction-constraint options. Selecting a loaded scenario updates `Use` only for usable, non-reserved dataset features.
- The free-form DuckDB filter expression lives in the collapsible footer; hiding the footer does not clear the active filter.
- Saved-filter rows support `Single`, `Multi`, and `Grouped` modes. `Single` keeps one selected row, `Multi` toggles rows and combines them with the active All/Any/Not all/None operator, and `Grouped` toggles rows while combining rows within a theme with `OR` and selected themes with `AND`.
- `--no-filters` disables saved-filter discovery.
- `--no-features` disables feature spec discovery.
- Filters are DuckDB `WHERE` expressions and apply before column profiling, chart aggregation, map aggregation, table rendering, low-weight grouping, response transforms, and sigma calculations.

**Column profile**

- Column profile is the first startup tool.
- Column profile is mandatory: `--tools` and `create_app(..., tools=...)` cannot remove it, and the backend always reports/registers it first so startup lands on the profile view.
- Summary requests return every readable dataset column with inferred kind, DuckDB type, filtered missing count, distinct count, and min/max for numeric/date-like columns. Auto-mode summaries are exact when `filtered rows * readable columns <= 10,000,000`; larger summaries use the first 100,000 filtered readable rows and include `calculation` metadata plus a UI action to recalculate all rows exactly. Passing `mode: "full"` to `/api/column-profile/summary` forces exact all-row summary stats.
- Unreadable columns are omitted from profile summaries and returned through `skipped_columns` with sanitized errors.
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

**GLM and GBM**

- GLM and GBM are opt-in tools (`--tools glm,gbm`) and are not part of the default user-facing tool set. Column Profile is still enabled alongside them.
- GLM still returns a shell `status: "not_implemented"` response.
- GBM config, validation, model listing, model activation, and source discovery must work without importing optional modelling libraries.
- GBM training imports LightGBM, pandas, and numpy lazily through the `gbm` optional extra. These packages must not become base install dependencies. On macOS, LightGBM's native library may also require Homebrew `libomp`; missing `libomp.dylib` should be reported as an actionable GBM dependency error, not a server 500.
- GBM training runs as an in-memory background job. `GET /api/gbm/jobs/{job_id}` returns transient `progress` while the job is queued/running, including phase, message, iteration, train/test metric points, and live evaluation history. Persisted training history remains `training_log.json` and `evaluation.parquet`; frontend Evaluation Log downsampling and `All` / `Tail` view zooming are render-only and must not truncate these artifacts.
- GBM uses a canonical uppercase `SAMPLE` column when present: `training` rows fit the model, `test` rows drive early stopping, and `validation` rows are scored as holdout diagnostics. If `SAMPLE` is absent, users can create a reusable generated 60/20/20 split stored as `.lucidum/models/gbm/generated_sample.parquet`; generated splits do not mutate the source dataset.
- GBM training mode is persisted as `training_mode`, defaulting to `normal` for older models. `ebm` mode is available only with a physical dataset `SAMPLE` column containing `training` and `test` rows after denominator filtering; generated sample sidecars do not enable it. EBM uses `num_iterations` as the global cap across all leaf stages, requires `early_stopping_rounds > 0`, starts with `num_leaves=2` and `learning_rate=0.3`, then advances leaf counts through the configured `num_leaves` after stage-local test-metric plateaus.
- GBM uses the sidebar Actual and denominator/KPI controls as the model response and offset/exposure inputs. The filter controls remain hidden while GBM is active because training ignores the global filter.
- GBM config includes loaded feature spec groupings and ordered feature scenarios. The frontend applies scenarios as a table selection convenience only; backend validation remains the source of truth for usable features, reserved response/offset/sample columns, and monotonicity.
- GBM manifests record `feature_scenario` only when training starts from an explicit scenario selection. The saved scenario name and feature snapshot are compared with the current spec when the model is active; stale or missing scenarios are shown as provenance only and do not override `feature_config.json`.
- GBM feature interaction constraints are driven by nonblank Feature Specification `Grouping` values. The frontend may send selected grouping names, but the backend injects the server-loaded feature grouping map before validation and training. Training constrains only currently selected trainable features in selected groups, adds a remainder constraint for all other selected features, and persists the training-time constrained group/feature snapshot in the manifest.
- Active-model config reports saved interaction constraints with `current`, `stale`, or `missing` group statuses. Stale or missing constraints are displayed as provenance and must not be resent for new training unless the user selects current grouping options. Feature table lock markers reflect selected current constraints or the active model's saved constraint snapshot.
- GBM artifacts are stored beside the source dataset under `.lucidum/models/gbm/`, with one directory per model.
- GBM `feature_config.json` is the persisted source of truth for the trained model's selected features, monotonicity settings, Gain values, and optional `mean_abs_shap` values.
- GBM config, activation, rename, and delete responses must drive the UI's `Use`, `Monotonicity`, feature importance metric, model navigator, sidebar model list, and parameter tables from the active model, so switching models mirrors exactly what was trained. If both Gain and SHAP are available, the Feature table shows a single Gain/SHAP metric column and defaults to SHAP.
- GBM active-model switching must also mirror the persisted `training_mode` radio state. The EBM radio group is hidden when `ebm_available` is false.
- GBM model changes reload frontend schema and invalidate source-scoped tools. Preserve Column Profile cache when it is active because it depends only on the raw dataset and active filter, but refresh Line/Bar and UK Mapping because they can read model-output sources such as `gbm:<model_id>:predictions` and `gbm_prediction`.
- GBM model IDs are folder names under `.lucidum/models/gbm/` and must stay source-ID safe: letters, numbers, dots, underscores, and hyphens only. Renaming a model renames the folder and updates manifest source IDs; deleting the active model promotes the newest remaining model, or clears active state if none remain.
- GBM is the one normal chooser that still displays invalid dataset columns; they must render as disabled invalid rows and must not be sent to LightGBM.
- GBM training and model-output sources must use explicit readable-column projections. Avoid `SELECT *` on the raw dataset path because unreadable columns can fail even when they are not selected as model features.
- GBM model outputs publish data sources through the shared `data_sources` contract using IDs such as `gbm:<model_id>:predictions`, `gbm:<model_id>:shap_long`, and `gbm:<model_id>:shap_summary`.
- The `gbm:<model_id>:shap_long` source ID is retained for compatibility, but the stored SHAP values artifact is wide: `__lucidum_row_id` plus one numeric SHAP column per selected feature. Bounded SHAP row modes such as `10k` and `100k` use a deterministic random sample from all scored rows seeded by the model `seed` parameter, not the first rows. `gbm:<model_id>:shap_summary` remains one row per feature.
- GBM SHAP plotting reads only saved SHAP sidecars and the original trained feature values joined by `__lucidum_row_id`; it must not import LightGBM, pandas, or numpy. SHAP config exposes only the active model's trained features with saved SHAP columns. One-feature plots use the selected feature's SHAP values; flame plots use the returned plotted x-domain exactly and omit the old 45-55 ribbon; two-feature plots use the sum of the two selected SHAP contributions. Continuous numeric axes use banding and optional tail grouping, return explicit numeric domains, omit missing numeric values with a warning, and factor-style axes include missing as `(missing)`. Numeric features forced to factor style keep natural band order, while true categorical box plots sort by descending median SHAP. Numeric/numeric surface payloads return dense backend grids for ECharts GL. The SHAP frontend preserves matching legend visibility across active-model switches only when the selected features, plot type, and legend series still match.
- LightGBM-specific training, objective handling, offsets, SHAP, feature importance, tree extraction, and tree label normalization belong in backend GBM modules, not in frontend code.
- GBM tree routes read persisted `tree_table.parquet` artifacts only and do not import LightGBM. The list route returns compact tree metadata; the detail route returns a frontend-ready split/leaf hierarchy with compact numeric thresholds, decoded categorical thresholds, edge labels, default-branch markers, cover percentages, and node values for colouring. Long categorical split display labels are summarized while full split labels remain available in tooltip fields, and frontend node clicks highlight the selected root-to-node path.

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
node --check src/py_lucidum/static/app/gbm-tool.js
node --check src/py_lucidum/static/app/gbm-shap-tool.js
node --check src/py_lucidum/static/app/gbm-shap-chart.js
node --check src/py_lucidum/static/app/gbm-tree-viewer.js
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
- Tool registry defaults, optional GLM/GBM registration, and the default `dataset` data-source contract.
- GBM validation, sidecar model store behavior, optional dependency failures, native runtime dependency failures, live job progress, active-model feature/parameter refresh, model data-source publishing, Gain ordering, SHAP row limits, SHAP plot aggregation routes, tree summary/detail routes, and chart/map use of prediction sources.
- Browser smoke behavior for loading profile, chart, map, and GBM tools without unexpected extra API requests or stale active-model state, including live GBM progress, the GBM tree viewer, and the GBM SHAP screen.

## Future Work

- GLM should become an independently registered modelling tool without coupling directly to Line/Bar internals.
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
- Keep generated caches, local datasets other than the synthetic demo, `.lucidum/` model artifacts, virtual environments, build artifacts, and OS metadata out of git.
- Do not commit real customer data. The bundled motor premiums dataset is synthetic.
