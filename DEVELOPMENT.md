# py_lucidum Development Notes

This document is the durable maintainer contract for `py_lucidum`: architecture, behavior contracts, testing policy, packaging notes, data handling, and tool-extension guidance should stay current here. The public user-facing documentation lives in `README.md`.

## Navigation

- Start with `Purpose` and `Current Architecture` to understand ownership boundaries.
- Use `Public Interfaces` before changing CLI, Python, or HTTP API behavior.
- Use `Behavior Contracts` for tool behavior, data-source rules, optional dependency policy, and persistence requirements.
- Use `UI Direction` before frontend layout or interaction changes.
- Use `Testing` and `Maintenance Rules` before handing off or committing changes.

## Purpose

`py_lucidum` is a package-first Python app for fast local exploration of large CSV and Parquet datasets. Users launch the bundled synthetic demo with `lucidum --demo`, launch another file with `lucidum path.parquet`, launch a same-schema folder of Parquet files with `lucidum path/to/folder/`, or start the app from Python with `py_lucidum.serve(path)`.

The app is currently local-first: it starts FastAPI and DuckDB in the user process, serves a static browser UI, and treats the input file as fixed until reload. It is designed as a shared workbench plus independently registered tools.

## Current Architecture

- `py_lucidum.core` owns DuckDB connection management, file relation SQL, schema inference, row counts, band suggestions, filter validation, saved-filter loading, shared data-source metadata, and shared metric/weight SQL helpers.
- `py_lucidum.app` owns the FastAPI factory, shared app context, token checks, static asset serving, favicon serving, schema/reload/health/shutdown endpoints, and tool registration.
- `py_lucidum.tools.registry` is the backend source of truth for tool IDs, labels, aliases, default enablement, and registration order.
- `py_lucidum.tools.dataset_viewer` implements the filtered raw-row dataset viewer and routes.
- `py_lucidum.cli` owns the `lucidum` command, free-port selection, token URL construction, background server handling for notebook-style runtimes, and browser opening.
- `py_lucidum.demo` resolves the bundled synthetic demo dataset from either the source tree or installed package resources.
- `py_lucidum.tools.column_profile` implements filtered column summary/detail profiling and routes.
- `py_lucidum.tools.line_bar` implements chart aggregation and line/bar routes.
- `py_lucidum.tools.histogram` implements filtered histogram binning, exact metric summaries, and histogram routes.
- `py_lucidum.tools.uk_map` implements UK map aggregation and UK map routes.
- `py_lucidum.tools.glm` implements the opt-in `glum` GLM tool. GLM validation, training jobs, persistence, coefficient/model-detail routes, and model-output data sources live in separate backend modules.
- `py_lucidum.tools.gbm` implements the opt-in LightGBM tool. GBM active config payloads, training, validation, persistence, tree summary/detail, and model-output data sources live in separate backend modules; the frontend only edits settings, starts jobs, polls status, and renders returned diagnostics.
- `py_lucidum.tools.specifications` implements the opt-in `specs` tool for editing feature, KPI, and filter specification CSV files. It reads and writes raw CSV rows, continuously validates against the durable spec loaders, generates unsaved starter drafts for missing/disabled specs, and refreshes enabled app metadata after successful saves.
- `src/py_lucidum/static/app.js` is a native ES-module bootstrap. `src/py_lucidum/static/app/main.js` owns the app shell/coordinator, shared sidebar/filter/KPI controls, tool selection, and cross-tool invalidation. `src/py_lucidum/static/app/dataset-viewer-tool.js` owns the Dataset Viewer frontend, `src/py_lucidum/static/app/column-profile-tool.js` owns the Column Profile frontend, `src/py_lucidum/static/app/line-bar-tool.js` owns the Line/Bar frontend, `src/py_lucidum/static/app/histogram-tool.js` owns the Histogram frontend, `src/py_lucidum/static/app/uk-map-tool.js` owns the UK Mapping frontend, `src/py_lucidum/static/app/glm-tool.js` owns GLM high-level orchestration, API mutation, and model build/detail flow, `src/py_lucidum/static/app/glm-formula-builder.js`, `src/py_lucidum/static/app/glm-model-navigator.js`, and `src/py_lucidum/static/app/glm-tabulations.js` own focused GLM frontend submodules, `src/py_lucidum/static/app/gbm-tool.js` owns GBM high-level orchestration, API mutation, and cross-tool invalidation, `src/py_lucidum/static/app/gbm-feature-parameter-controls.js`, `src/py_lucidum/static/app/gbm-evaluation-chart.js`, `src/py_lucidum/static/app/gbm-model-navigator.js`, and `src/py_lucidum/static/app/gbm-tab-orchestration.js` own focused GBM frontend submodules, `src/py_lucidum/static/app/gbm-shap-tool.js` and `src/py_lucidum/static/app/gbm-shap-chart.js` own the GBM SHAP UI/chart split, `src/py_lucidum/static/app/gbm-stacked-shap-tool.js` and `src/py_lucidum/static/app/gbm-stacked-shap-chart.js` own the Stacked SHAP UI/chart split, `src/py_lucidum/static/app/gbm-tree-viewer.js` owns the D3 tree viewer, and `src/py_lucidum/static/app/shared/` owns import-safe shared browser helpers.
- `src/py_lucidum/static/app.css` is the stable linked CSS entrypoint and import manifest. Split styles live under `src/py_lucidum/static/styles/`; `foundations.css` and `controls.css` own shared primitives, while shell/tool files own boundary-specific selectors.
- Third-party browser libraries are vendored under `src/py_lucidum/static/vendor/`. Core ECharts and Leaflet are loaded locally from `index.html` because the default app tools need them at startup. Histogram lazy-loads Tabulator for the metrics grid. GLM lazy-loads Ace for formula editing. GBM lazy-loads Tabulator for editable grids, D3 for tree diagrams, and ECharts GL only for SHAP 3D surface plots.

Tool packages own their training, validation, persistence, routes, and artifacts. GLM and GBM intentionally publish model outputs through the shared `data_sources` contract, and Line/Bar intentionally consumes those outputs for prediction plotting, GBM SHAP ribbons, and GLM overlays. Optional modelling dependencies must still be imported lazily and must not become base app imports. Shared reusable logic should move into `core`, `static/app/shared/`, or a modelling/shared backend helper when there is real reuse.

Frontend tools should prefer a `createXTool({ deps })` factory. Data-driven tools should expose `buildRequest`, `fetchData`, `useCached`, and render/lifecycle methods where applicable, with shared shell behavior kept in `main.js` or `static/app/shared/`. GLM/GBM model UI helpers that are genuinely common, such as polling cadence, model-list grouping, fallback selection, action button state, resize observation, and empty/status HTML, live in `src/py_lucidum/static/app/shared/model-ui.js`.
New frontend tool styles should live in a tool-owned file under `static/styles/`; move reusable tokens, layout primitives, and shared controls into `foundations.css` or `controls.css` instead of duplicating them.

## Public Interfaces

- CLI:
  - `lucidum --demo`
  - `lucidum path/to/data.parquet`
  - `lucidum path/to/data.csv`
  - `lucidum path/to/parquet-folder/`
  - common options include `--open`, `--host`, `--port`, `--no-token`, `--buttons`, `--x`, `--actual`, `--expected`, `--denominator`, `--filters`, `--no-filters`, `--kpis`, `--no-kpis`, `--features`, `--no-features`, `--tools`, and UK map column overrides.
- Python:
  - `py_lucidum.serve(...)`
  - `py_lucidum.serve_line_bar(...)`
  - `py_lucidum.run_app(...)`
  - `py_lucidum.demo_dataset_path()`
  - `py_lucidum.app.create_app(...)`
- HTTP:
  - `GET /api/schema`
  - `POST /api/dataset-viewer/table`
  - `POST /api/banding/suggestion`
  - `POST /api/metrics/summary`
  - `GET /api/health`
  - `GET /api/lucidum-servers`
  - `POST /api/reload`
  - `POST /api/shutdown`
  - `POST /api/lucidum-servers/stop`
  - `POST /api/column-profile/summary`
  - `POST /api/column-profile/detail`
  - `POST /api/chart`
  - `POST /api/line-bar/chart`
  - `POST /api/line-bar/table`
  - `POST /api/histogram/chart`
  - `POST /api/uk-map/summary`
  - `GET /api/glm/summary`
  - `GET /api/glm/config`
  - `GET /api/glm/models`
  - `POST /api/glm/validate`
  - `POST /api/glm/build`
  - `GET /api/glm/jobs/{job_id}`
  - `POST /api/glm/tabulations/build`
  - `GET /api/glm/tabulations/jobs/{job_id}`
  - `POST /api/glm/tabulations/config`
  - `POST /api/glm/tabulations/table`
  - `POST /api/glm/tabulations/plot`
  - `POST /api/glm/tabulations/export`
  - `GET /api/glm/models/{model_id}`
  - `POST /api/glm/models/{model_id}/activate`
  - `POST /api/glm/models/{model_id}/rename`
  - `DELETE /api/glm/models/{model_id}`
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
  - `GET /api/specs/{kind}`
  - `POST /api/specs/{kind}/validate`
  - `POST /api/specs/{kind}/save`

`/api/chart` is retained for compatibility with the current frontend. New integrations should prefer the namespaced line-bar endpoint.

## Behavior Contracts

**Datasets and packaging**

- The committed demo dataset is `datasets/motor_premiums.parquet`.
- The source tree also commits `datasets/monthly/*.parquet`, a seven-file monthly split of the demo data used for folder-input manual checks and tests.
- The wheel packages the demo dataset as `py_lucidum/datasets/motor_premiums.parquet`.
- Other top-level local CSV/Parquet files under `datasets/` remain ignored; keep ad hoc nested dataset folders out of git unless they are intentional fixtures.
- Parquet is the preferred working format for speed; CSV remains supported for convenience.
- A dataset path may be a folder of direct-child `.parquet` files for non-modelling tools. Lucidum builds one DuckDB `read_parquet([...])` relation from the sorted direct children, ignores non-Parquet files and nested folders, and requires every file to have identical column-name-to-DuckDB-type mappings. The schema payload reports the folder path, direct-child Parquet `file_count`, combined byte size, combined row count, and `source_kind: "parquet_folder"`; the header displays folder names as `name (n files)`.
- Parquet folder inputs are rejected at app creation when `glm`, `gbm`, or `--tools all` enables modelling tools. GLM/GBM model stores, workers, prediction sidecars, SHAP sidecars, generated samples, and dataset workspaces remain single-source-file only.
- `GET /api/schema` includes `data_sources` and `feature_bases`. The default source is `dataset`; model outputs publish named tabular artifacts through this same contract.
- `GET /api/schema` does not precompute numeric band suggestions. Column payloads keep `band_suggestion` for compatibility, normally as `null`; chart tools request initial numeric band estimates lazily through `POST /api/banding/suggestion`.
- `GET /api/schema` excludes unreadable columns from `columns` and reports them as `invalid_columns` with sanitized errors. Normal tools should use the safe column maps; only diagnostics or choosers that explicitly report invalid columns should use the all-column map.
- Prefer doing derived or convenience calculations on the fly when the user selects a tool, source, feature, or filter. Avoid startup-time pre-calculation unless it is specifically part of the desired startup behavior, such as Column Profile's initial landing view.
- Line/Bar accepts a `source` request field and defaults it to `dataset`. Unknown sources are rejected before query execution.

**Defaults, saved filters, KPIs, and feature specs**

- Without explicit defaults, the x-axis starts with the first dataset column, Actual starts with the first numeric column, and Expected starts as none.
- CLI options, programmatic defaults, and URL parameters can override initial selections.
- Line/Bar feature and Expected picker lists default to A-Z ordering. The Expected picker section starts collapsed on every app launch and can be reopened for the current browser session with the splitter chevron; its open/closed state is not persisted.
- Saved filters load from an explicit `--filters` path, otherwise `./filter_spec.csv`, otherwise `./specs/filter_spec.csv`.
- `lucidum --demo` falls back to packaged demo filter, KPI, and feature specs when a spec kind has no explicit path, no `--no-*` flag, and no current-working-directory default spec file.
- Saved-filter CSV files must have exactly `theme,name,expression` columns. CSV order controls theme order and row order.
- KPI specs load from an explicit `--kpis` path, otherwise `./kpi_spec.csv`, otherwise `./specs/kpi_spec.csv`.
- KPI spec CSV files must have exactly `group,name,actual,denominator,decimals,format` columns. `denominator` aliases `N`, `Average row value`, empty, and `__none__` all mean average row value; `format` is `number`, `currency`, or `percent`. Percent formatting displays proportions as percentages, so `0.1` displays as `10%`.
- KPI rows are a single-selection convenience layer over Actual and Weight. Manual Actual/Weight changes keep the KPI row active only when both selects exactly match a spec row.
- The sidebar Actual and Weight controls remain visible for every active tool while the sidebar is expanded. Their summary values are owned by the shared app shell, update from the selected source/Actual/Weight and active filter, and must not depend on which tool is active.
- KPI decimals and format apply to Actual and Expected response values in metric titles, line/bar labels and response axes, table response cells, map labels/tooltips/popups, and map legend values. Weight and row-count formatting is unchanged.
- Feature specs load from an explicit `--features` path, otherwise `./feature_spec.csv`, otherwise `./specs/feature_spec.csv`.
- Feature spec CSV files must start with `Feature,Grouping`. Reserved metadata columns immediately after `Grouping` are `Base`, `min`, `max`, and `banding`; scenario columns start after the contiguous reserved metadata block. Older specs without these metadata columns remain valid and treat every column after `Grouping` as an ordered GBM scenario name. Scenario cells include the feature when they contain the word `feature`, case-insensitive.
- `Grouping` is displayed in the GBM Feature table between `Feature` and `Use` and supplies the GBM feature interaction-constraint options. Selecting a loaded scenario updates `Use` only for usable, non-reserved dataset features.
- `Base` values are exposed through schema and GBM SHAP config payloads. For numeric chart axes, rescale matching uses the displayed numeric band containing the base, falling back to the nearest displayed numeric band when needed. GLM tabulations also use `Base` to define base cells, and numeric `min/max/banding` to define rating-table grids.
- The free-form DuckDB filter expression lives in the collapsible footer; hiding the footer does not clear the active filter.
- Saved-filter rows support `Single`, `Multi`, and `Grouped` modes. `Single` keeps one selected row, `Multi` toggles rows and combines them with the active All/Any/Not all/None operator, and `Grouped` toggles rows while combining rows within a theme with `OR` and selected themes with `AND`.
- `--no-filters`, `--no-kpis`, and `--no-features` disable discovery for app metadata and prevent the Specifications tool from preloading default-discovered files for those disabled spec kinds.
- The default-enabled Specifications tool exposes Feature, KPI, and Filter spec screens backed by `/api/specs/*`, edits raw CSV fields rather than normalized schema metadata, continuously validates the same file contracts used at startup/reload, and atomically saves the selected spec file. Missing or disabled specs return generated starter drafts only: Feature specs get one row per valid dataset column with only `Feature` populated, while KPI and filter specs get one blank row with visual-only placeholder hints. Generated drafts are not written until Save; the file-path line labels the save target and marks new files or suppressed existing files. Saving a disabled spec writes the CSV for editing but leaves the running app metadata disabled until restart/recreate without the corresponding `--no-*` option.
- Filters are DuckDB `WHERE` expressions and apply before column profiling, chart aggregation, histogram binning/statistics, map aggregation, table rendering, low-weight grouping, response transforms, and sigma calculations.

**Dataset Viewer**

- Dataset Viewer is default-enabled but registered after Line/Bar when both tools are enabled. Explicit `--tools` / `create_app(..., tools=...)` selections are exact and include it only when `dataset-viewer` is requested.
- Requests return readable raw dataset columns and up to 1,000 filtered preview rows, plus `has_more` when another row exists beyond the cap. They deliberately do not compute exact total or filtered row counts.
- The active footer/saved-filter expression is applied server-side before the display cap.
- Whole-table search, column sorting, axis-exclusive whole-row or whole-column selection, transpose, optional alphabetical column ordering, right-click cell copy, and right-click selected row/column CSV output are client-side over the loaded display rows only. Header labels select whole columns/rows while compact sort symbols cycle ascending, descending, and unsorted states in both orientations. In transpose mode, search matches only the original column names shown in the first transposed column and keeps every loaded preview row as a `Row n` column; selected preview rows or dataset columns are preserved when toggling transpose.
- Unreadable dataset columns are omitted from the grid and returned as payload warnings without rendering a table overlay.

**Column profile**

- Column profile is default-enabled but not mandatory: explicit `--tools` and `create_app(..., tools=...)` selections include it only when requested. It appears after Line/Bar and Dataset Viewer when all three tools are enabled.
- Summary requests return every readable dataset column with inferred kind, DuckDB type, filtered missing count, distinct count, and min/max for numeric/date-like columns. Auto-mode summaries are exact when `filtered rows * readable columns <= 10,000,000`; larger summaries use the first 100,000 filtered readable rows and include `calculation` metadata plus a UI action to recalculate all rows exactly. Passing `mode: "full"` to `/api/column-profile/summary` forces exact all-row summary stats.
- Unreadable columns are omitted from profile summaries and returned through `skipped_columns` with sanitized errors.
- Detail requests return value counts for categorical columns and histogram/stat tables for numeric/date-like columns.
- Profile requests respect the same active footer/saved-filter expression as the other tools.

**Line and bar chart**

- Line/Bar is default-enabled, registered first when enabled, and opens on startup for ordinary launches unless a valid `?tool=` URL parameter requests another enabled tool.
- X-axis features can be integer, numeric, string/categorical, date, or datetime.
- The x-axis feature list has `Imp`, `Original`, and `A-Z` sort modes. `Imp` appears only when an active GBM or GLM has saved feature-level importances, lists raw dataset features grouped as GBM, GLM, and `Not used`, and selects dataset features only. GBM importance prefers saved mean absolute SHAP when present and otherwise uses LightGBM Gain. GLM importance uses the persisted weighted mean absolute centered feature contribution on the GLM linear-predictor scale.
- Numeric x-axis banding uses the existing visible controls only: concrete band widths, step buttons, and explicit `-` for no banding. The initial concrete width is estimated lazily from the selected source/filter when a numeric feature is selected, using a bounded sample of up to 100,000 rows.
- Numeric banding floors values to the selected band width by default. With `quantileMode` enabled, the same banding value is rounded and clamped to `1..1000`, non-missing numeric values are grouped as `Q1`, `Q2`, etc., and missing x-axis values stay in a separate `Missing` group.
- Date/datetime axes use calendar buckets.
- Actual and Expected lines use a shared denominator. `Average row value` divides by valid row count; a numeric Weight column divides by `SUM(weight)`.
- Response transforms `0` and `1` use the selected x-axis feature's `Base` metadata when available. `0` subtracts each response's value at the base group; `1` divides by it and the frontend displays those ratio values as uplift percentages (`1 = 0%`). Actual and Expected references are independent, and sigma bounds use the Expected reference. If no Base is available, the transform keeps the historical overall-average reference; if a declared Base cannot be applied, the response includes a warning and falls back to the overall average.
- `partialDependence.mode` accepts `none`, `shap`, `glm`, or `both`. `shap` overlays active-GBM SHAP percentile ribbons for the selected x-axis feature. The active GBM must expose both `gbm:<model_id>:shap_long` and `gbm:<model_id>:predictions`; missing artifacts or missing feature SHAP columns return warnings without breaking the normal chart. Ribbons use the same filter, denominator weighting, x grouping, banding, quantile mode, low-weight grouping, and response transform as the chart. SHAP values are first converted to response scale using the GBM objective link, then shifted or multiplied so the weighted p50 ribbon mean matches the active GBM fitted prediction mean for the current chart slice. Categorical x-axes accept `sort: "shap"` when the SHAP overlay is visible (`shap` or `both`) and available, sorting by descending scaled median SHAP.
- `glm` overlays the active GLM as a dashed central line. For GLMs where the x-axis feature is not in an interaction term, Line/Bar uses GLM tabulation-style base-profile prediction: vary x over rendered chart groups, hold other model inputs at Feature Spec `Base` or filtered inferred base values, and predict only one row per x-axis group before fitted-mean scaling. Simple two-way x interactions use a collapsed partner grid: numeric partners use Feature Spec grids when compact and otherwise 50 quantile-style context bins; categorical partners use observed levels up to 200; collapsed overlays cap at 2,000 context rows and 250,000 prediction cells. Higher-order interactions, multiple interaction partners, excessive partner cardinality, or collapsed guardrail overflows fall back to the deterministic sampled marginal PDP over the current filtered valid chart population, capped at 100,000 sampled rows and 2,000,000 prediction cells. When LightGBM isolation is needed, GLM overlays use a persistent hot worker after the first request and cache the unpickled estimator by artifact signature. `partial_dependence.method` can be `base_profile`, `collapsed_marginal`, or `sampled_marginal`, with overlay-local timing and guardrail metadata in `partial_dependence.sample`. `both` builds SHAP and GLM overlays independently, then aligns the GLM line to the SHAP/active-GBM fitted-mean baseline when both are available, and returns warnings per overlay so one unavailable model does not suppress the other.
- Low-weight grouping uses selected Weight total, not raw row count.
- Chart requests are capped at 10,000 x-axis groups after the shared Line/Bar grouping and sort pipeline. Chart responses include `group_count`, `max_groups`, and `groups_truncated`; truncated chart responses warn that the table search still covers all groups.
- Table view uses `POST /api/line-bar/table` and is always server-backed. The table endpoint accepts the normal Line/Bar request fields plus `tableSearch`, `tablePage`, and `tablePageSize`; the default page size is 10,000 groups, returned pages render through a virtualized frontend grid, and there is no table-wide result cap.
- Table search is a case-insensitive substring match against grouped x-axis labels/values across the full grouped result, not just chart-returned rows. The server returns only the requested page plus a `summary` computed over all matching groups, so the `Total` footer is not page-limited.
- Table search and pagination must not refetch `/api/chart` or include table search state in chart requests. Response transforms use references from the full grouped result before chart slicing or table pagination.

**Histogram**

- Histogram is a default-enabled chart tool registered after Line/Bar and before UK Mapping. Aliases are `histogram`, `hist`, and `histo`.
- Histogram uses the shared sidebar data source, Actual, Weight, KPI, and active filter controls. With `Average row value`, `N`, empty, or `__none__` Weight, the plotted value is Actual and bar volume is row count. With a numeric Weight column, the plotted value is `Actual / Weight` and bar volume is Weight sum.
- Rows with missing Actual are excluded. When a numeric Weight is selected, rows with missing, zero, or negative Weight are also excluded and warned. Log x-scale excludes nonpositive plotted values and warns.
- `bins` accepts `auto` or an explicit count. Explicit counts clamp to `1..10000`; `auto` uses `sqrt(valid_count)` clamped to `10..200`. When x log scale and Weight are off, integer Actual columns and numeric Actual values that are all whole numbers use centered, touching integer bins with the requested count as an upper bound; response `bins` reports the actual returned bin count. `sampleMode=100k` deterministically samples only the chart binning; the metrics table remains exact for the full filtered valid population.
- `distribution` supports incremental and cumulative bars. `yAxis` supports sum and probability; probability divides by the same valid chart volume used for the sum axis. Log x/y settings are render controls around the returned valid histogram values.
- The response includes bin rows, binning metadata, exact metric rows, row-count metadata, denominator metadata, warnings, and timings. The frontend uses integer binning metadata for sensible integer x-axis labels and renders an ECharts histogram with mean/median reference lines and a compact Tabulator metrics grid.

**UK mapping**

- Area and sector layers join grouped KPI summaries to bundled GeoJSON assets.
- App map assets load locally, including Leaflet and bundled UK GeoJSON. Nonblank base-map backgrounds are a separate external tile dependency: OSM, Esri/Aerial, Light, and Dark layers fetch tiles from their configured third-party providers, while `Blank` makes no external tile requests.
- Sector smoothing uses a committed shared-edge adjacency sidecar generated from the bundled sector GeoJSON, then pools already-aggregated numerator and denominator values across sectors reachable within the selected neighbour depth. When smoothing is active, all sidecar sectors are smoothing targets, so sectors with no original data can be filled from valid neighbours. Raw sector fields remain available in the API rows for popup context.
- Default join columns are `PostcodeArea`, `PostcodeSector`, and `PostcodeUnit`; uppercase aliases are supported.
- Default coordinate columns are `lat` and `long`; `latitude`/`LATITUDE` and `longitude`/`LONGITUDE` aliases are supported.
- Unit points group by postcode unit, average coordinates, and plot only units with valid KPI and valid coordinates.
- Area and sector geometry use Leaflet GeoJSON with hover tooltips and click popups.
- Unit points render on a canvas-backed Leaflet layer with a hit grid for hover tooltips and click popups. Unit redraws intentionally project rows first and then apply pixel-space culling; a geographic viewport prefilter before projection is not part of the current rendering strategy because it did not improve observed redraw speed during testing.
- If no unit point columns are configured and defaults are absent, the Units layer is disabled. Explicit invalid unit point columns produce validation errors when requested.
- Regenerate the sector adjacency sidecar with `scripts/build_uk_sector_adjacency.py` after replacing `sectors_MappaR.geojson`.

**GLM and GBM**

- GLM and GBM are opt-in tools and are not part of the default user-facing tool set. `--tools all` enables them with every other tool. Explicit modelling selections must also include `line-bar` because GLM/GBM context-menu actions open Line/Bar charts. GLM and GBM do not imply each other; request both when cross-model tabulation or comparison workflows are needed.
- GLM and GBM artifacts are scoped to the exact dataset version under `.lucidum/datasets/<dataset-slug>/<dataset-signature>/models/{glm,gbm}/`. The slug is derived from the dataset filename. The signature is derived from file size, modification time, row count, and schema fingerprint. Startup scans only the current signature workspace; root-level `.lucidum/models/` folders and other dataset-version workspaces are ignored and must never break raw dataset startup.
- GLM config, validation, model listing, model activation, and source discovery must work without importing optional modelling libraries.
- GLM training imports `glum`, pandas, and numpy lazily through the `glm` optional extra. These packages must not become base install dependencies. Build routes should report missing GLM dependencies as an actionable install-extra error, not a server 500.
- GLM accepts full `response ~ terms` formulas and RHS-only formulas using the sidebar Actual response. Raw formulas are stored in `formula.txt` with comments, but `#` comments outside quoted strings are stripped before fitting. The allowed formula context is intentionally narrow: `ifelse`, `pmin`, `pmax`, `ns`, `bs`, `cs`, `poly`, `C`, and common numeric transforms. Obvious unsafe text such as `__`, `import`, `eval`, `exec`, `open`, and statement separators is rejected before fitting. Explicit `offset(...)` terms are stripped from the fitted formula, stored in the manifest, evaluated with the same safe context, and passed to `glum.fit()`, prediction, and tabulation reconstruction.
- GLM families are `normal`, `poisson`, `gamma`, `tweedie`, `binomial`, `inverse.gaussian`, and `negative.binomial`; the first implementation uses `link="auto"`. Tweedie power and negative-binomial theta are the only exposed family parameters.
- GLM regularization is optional and defaults to unpenalized `alpha=0`. Auto regularization uses glum cross-validation over ridge, elastic net, and lasso mixes with predictor scaling; manual regularization accepts a positive alpha and `0 <= l1_ratio <= 1`. Penalized GLMs must store selected penalty metadata and suppress coefficient standard errors/p-values in the frontend.
- GLM uses the sidebar Actual and Weight/KPI controls as the model response and denominator inputs. If a denominator is selected, training fits `response / denominator` with `sample_weight=denominator`, writes `glm_prediction` on the original response scale, and exposes `glm_prediction_rate = glm_prediction / denominator`.
- GLM `All` fits all valid rows. GLM `Training` fits only a physical uppercase/lowercase-insensitive `SAMPLE = training` column and does not create generated sample splits.
- GLM artifacts are stored under the current dataset workspace in `models/glm/`, with compact `manifest.json`, `estimator.pkl`, `formula.txt`, `coefficients.parquet`, `feature_importance.parquet`, `predictions.parquet`, and `diagnostics.json`. Tabulation builds add `tabulations/tabulation_manifest.json`, `tabulations/*.parquet`, and top-level `tabulated_predictions.parquet`.
- GLM model manifests persist only compact Lucidum metadata: identity, response/denominator, family/link, regularization, training scope, offset expressions, minimal formula execution flags, and `timings.elapsed_ms`. Model diagnostics and warnings live in `diagnostics.json`; raw formula text lives in `formula.txt`; raw source columns are derived from the dataset schema. When LightGBM/glum load-order protection is required, GLM fitting uses a persistent isolated worker after the first build and caches the worker-side dataset handle by source-file signature. Keep `PY_LUCIDUM_GLM_FIT_ONE_SHOT=1` as the debugging fallback for the old one-shot worker path.
- GLM training persists feature-level importances in `feature_importance.parquet`. The metric is weighted mean absolute centered feature contribution on the fitted GLM linear-predictor scale. Formulaic term metadata maps model-matrix columns to source dataset features; interaction and multi-feature transform contributions are split evenly across participating features before feature-level aggregation.
- GLM tabulations are created on demand through `/api/glm/tabulations/*`, not during GLM training. The tabulation algorithm is documented in `docs/specs/glm-tabulations.md`; keep it aligned with the R `GlimmaR` method: group terms by feature combinations, compute uncentered linear-predictor table contributions, subtract base-cell contributions, accumulate the adjustment into the base table, and score rows from the stored tables.
- GLM tabulation rebasing is an app-level linear-predictor gauge transform for single-model GLM non-base tables. Interaction crosstab rebases may transfer offsets into compatible one-way tables; one-way and no-crosstab rebases transfer offsets into `base`. Rebase/reset must preserve raw sidecars, rewrite adjusted tables, rebuild `tabulated_predictions.parquet`, and invalidate source-scoped frontend caches so `glm_tabulated_prediction` reflects the recalculated artifacts.
- GLM and GBM tabulation XLSX export is single-model only, reads existing `tabulations/tabulation_manifest.json` manifests and parquet sidecars only, saves `<model_id>_tabulations_<scale>.xlsx` beside the selected model's tabulation sidecars, and must not import `glum`, LightGBM, pandas, or numpy.
- GLM tabulation builds estimate missing numeric feature spec `min/max/banding` from scored rows and report warnings. Categorical levels not seen in training are retained as NA table cells and warned. Tables over 100,000 cells are skipped with a warning.
- GLM model and tabulation changes reload frontend schema and invalidate source-scoped tools. Preserve Column Profile cache when it is active because it depends only on the raw dataset and active filter, but refresh Line/Bar and UK Mapping because they can read `glm:<model_id>:predictions`, `glm_prediction`, `glm_prediction_rate`, and `glm_tabulated_prediction`.
- GLM model IDs are folder names under the current dataset workspace's `models/glm/` directory and must stay source-ID safe: letters, numbers, dots, underscores, and hyphens only. Renaming a model renames the folder; source IDs are computed from the folder name and existing artifacts. Deleting the active model promotes the newest remaining model, or clears active state if none remain.
- GLM model outputs publish data sources through the shared `data_sources` contract using IDs such as `glm:<model_id>:predictions`. `glm_prediction` remains the primary `ModelPredictionSource.column`; denominator-backed models also expose `glm_prediction_rate`, and when `tabulated_predictions.parquet` exists, `GlmSourceProvider` left joins it on `__lucidum_row_id` so the same source also exposes `glm_tabulated_prediction`.
- GBM config, validation, model listing, model activation, and source discovery must work without importing optional modelling libraries.
- GBM training imports LightGBM, pandas, and numpy lazily through the `gbm` optional extra. These packages must not become base install dependencies. On macOS, LightGBM's native library may also require Homebrew `libomp`; missing `libomp.dylib` should be reported as an actionable GBM dependency error, not a server 500.
- GBM training runs as an in-memory background job. `GET /api/gbm/jobs/{job_id}` returns transient `progress` while the job is queued/running, including phase, message, iteration, train/test metric points, and live evaluation history. Persisted training history is `evaluation.parquet`; frontend Evaluation Log downsampling and `All` / `Tail` view zooming are render-only and must not truncate this artifact.
- GBM uses a canonical uppercase `SAMPLE` column when present: `training` rows fit the model, `test` rows drive early stopping, and `validation` rows are scored as holdout diagnostics. If `SAMPLE` is absent, users can create a reusable generated 60/20/20 split stored as `models/gbm/generated_sample.parquet` under the current dataset workspace; generated splits do not mutate the source dataset.
- GBM training mode is persisted as manifest `training_mode`. `ebm` mode is available when the active sample source, either a physical dataset `SAMPLE` column or the generated sidecar split, contains `training` and `test` rows after denominator filtering. EBM uses `num_iterations` as the global cap across all leaf stages, requires `early_stopping_rounds > 0`, starts with `num_leaves=2` and `learning_rate=0.3`, then advances leaf counts through the configured `num_leaves` after stage-local test-metric plateaus.
- GBM parameter cells support grid-search braces: explicit sets like `{200, 300, 400}` and inclusive numeric ranges like `{0.05, 0.3; 0.05}`. Grid search samples combination indexes deterministically from the hypergrid without constructing the full cartesian product, pre-validates only sampled combinations, skips invalid combinations with a notice, trains valid combinations sequentially in one job, persists each as a normal model with `grid_search` metadata, and activates the best completed model by test metric when present, otherwise training metric.
- GBM exposes `init_score` as the first parameter-table row even though it is supplied to LightGBM datasets rather than to `lgb.train(params=...)`. `none` preserves the existing denominator-derived log offset for log-link objectives. A selected numeric dataset column or fitted GLM prediction source is a full prediction-space baseline; it is transformed with the objective link, replaces the denominator-derived init score, is persisted as `init_score.parquet`, and makes LightGBM `boost_from_average` irrelevant for that fit.
- GBM `parameters.json` is reserved for LightGBM-compatible Python params that can be loaded with `json.load()` and passed to `lgb.train(params=...)`, including objective and metric. Lucidum-only state such as `training_mode`, selected init-score value/provenance, and EBM stage metadata belongs in `manifest.json`.
- GBM uses the sidebar Actual and denominator/KPI controls as the model response and offset/exposure inputs. Denominator-backed models expose `gbm_prediction_rate = gbm_prediction / denominator`. The filter controls remain hidden while GBM is active because training ignores the global filter.
- GBM config includes loaded feature spec groupings and ordered feature scenarios. The frontend applies scenarios as a table selection convenience only; backend validation remains the source of truth for usable features, reserved response/offset/sample columns, and monotonicity.
- GBM manifests record `feature_scenario` only when training starts from an explicit scenario selection. The saved scenario name and feature snapshot are compared with the current spec when the model is active; stale or missing scenarios are shown as provenance only and do not override `features.json` or `feature_config.parquet`.
- GBM feature interaction constraints are driven by nonblank Feature Specification `Grouping` values. The frontend may send selected grouping names, but the backend injects the server-loaded feature grouping map before validation and training. Training constrains only currently selected trainable features in selected groups, adds a remainder constraint for all other selected features, and persists the training-time constrained group/feature snapshot in the manifest.
- Active-model config reports saved interaction constraints with `current`, `stale`, or `missing` group statuses. Stale or missing constraints are displayed as provenance and must not be resent for new training unless the user selects current grouping options. Feature table lock markers reflect selected current constraints or the active model's saved constraint snapshot.
- GBM pair interaction allowlists are sent as `feature_interaction_pairs: [{left, right}]`. Pair constraints may accompany Feature Specification grouping constraints only when the selected groups are disjoint from every paired feature; groups containing paired features are rejected because they would permit additional interactions for that feature. Singleton feature locks may accompany pair mode only when the locked feature is not part of any pair; paired features cannot also be isolated. Pair mode persists as `feature_interaction_constraints.mode = "pairs"` with `pairs`, optional disjoint `groups`/`groupings`, and optional `features`; config returns those constraints for the active model, and the Model navigator labels pair-constrained models as `Pairs (n)`. Training converts pairs plus any disjoint groups to LightGBM interaction constraints, then adds singleton constraints for selected features not covered by a pair or group, without adding the broad remainder group used by grouping constraints. For `num_leaves > 3`, validation warns that pair constraints control branch co-occurrence but do not guarantee purely 2D terms.
- GBM artifacts are stored under the current dataset workspace in `models/gbm/`, with one directory per model.
- GBM `features.json` is the persisted source of truth for the trained model's input feature names in exact LightGBM training order. GBM `feature_config.parquet` is an optional output/display artifact for the trained features, enriched with kind, monotonicity settings, Gain values, and optional `mean_abs_shap` values.
- GBM prediction and SHAP source relations derive their raw readable-column projection list from the current dataset schema. Do not persist duplicate `source_columns` lists in GBM manifests.
- GBM config, activation, rename, and delete responses must drive the UI's `Use`, `Monotonicity`, feature importance metric, model navigator, sidebar model list, and parameter tables from the active model, so switching models mirrors exactly what was trained. If both Gain and SHAP are available, the Feature table shows a single Gain/SHAP metric column and defaults to SHAP.
- EBM active models add an `EBM Gain` metric toggle option. It reads only the persisted `tree_table.parquet`, groups effective trees by their unique split-feature combination, and replaces the Feature table with `Tree features`, `Dim`, `Trees`, `Gain`, and `% Gain`.
- GBM active-model switching must also mirror the persisted `training_mode` radio state. The EBM radio group is hidden when `ebm_available` is false.
- GBM model changes reload frontend schema and invalidate source-scoped tools. Preserve Column Profile cache when it is active because it depends only on the raw dataset and active filter, but refresh Line/Bar and UK Mapping because they can read model-output sources such as `gbm:<model_id>:predictions`, `gbm_prediction`, and `gbm_prediction_rate`.
- GBM model IDs are folder names under the current dataset workspace's `models/gbm/` directory and must stay source-ID safe: letters, numbers, dots, underscores, and hyphens only. Renaming a model renames the folder; source IDs are computed from the folder name and existing artifacts. Deleting the active model promotes the newest remaining model, or clears active state if none remain.
- GBM is the one normal chooser that still displays invalid dataset columns; they must render as disabled invalid rows and must not be sent to LightGBM.
- GBM training and model-output sources must use explicit readable-column projections. Avoid `SELECT *` on the raw dataset path because unreadable columns can fail even when they are not selected as model features.
- GBM model outputs publish data sources through the shared `data_sources` contract using IDs such as `gbm:<model_id>:predictions`, `gbm:<model_id>:shap_long`, and `gbm:<model_id>:shap_summary`; denominator-backed prediction and SHAP-long sources expose `gbm_prediction_rate`.
- The `gbm:<model_id>:shap_long` source ID is retained for compatibility, but the stored SHAP values artifact is wide: `__lucidum_row_id` plus one numeric SHAP column per selected feature. When selected feature interaction constraint groups exist, excluding singleton feature constraints, `shap_values.parquet` also includes grouped contribution columns named `<Grouping>_INTERACTION_GROUP`; these are row-wise sums of the grouped feature SHAP columns and are not included in `shap_summary.parquet`. Bounded SHAP row modes such as `10k` and `100k` use a deterministic random sample from all scored rows seeded by the model `seed` parameter, not the first rows. `gbm:<model_id>:shap_summary` remains one row per feature; persisted `shap_summary.parquet` stores `feature`, `mean_abs_shap`, `mean_shap`, and `row_count`, with model identity derived from the model folder and source ID rather than a repeated Parquet column.
- The Actual selector groups choices into Dataset features, Model predictions, and SHAP values. Model prediction choices include active GLM and GBM prediction sources, including denominator-backed prediction rates and `glm_tabulated_prediction` after GLM tabulation, and switch the active data source to that model output source when selected. SHAP choices remain scoped to the active GBM model.
- GBM SHAP plotting reads only saved SHAP sidecars and the original trained feature values joined by `__lucidum_row_id`; it must not import LightGBM, pandas, or numpy. SHAP config exposes only the active model's trained features with saved SHAP columns and includes loaded `Base` metadata for those features. One-feature plots use the selected feature's SHAP values; flame plots use the returned plotted x-domain exactly and omit the old 45-55 ribbon; two-feature plots use the sum of the two selected SHAP contributions. Continuous numeric axes use banding and optional tail grouping, return explicit numeric domains, omit missing numeric values with a warning, and factor-style axes include missing as `(missing)`. Numeric features forced to factor style keep natural band order, while true categorical box plots sort by descending median SHAP. Numeric/numeric surface payloads return dense backend grids for ECharts GL. Ordinary SHAP plot requests accept `rescale` values `-`, `0`, or `1`; `-` preserves raw behavior, `0` shifts by the relevant Base reference on the linear predictor scale, and `1` exponentiates values first before scaling to the base response-scale reference, with the frontend displaying those response-scale ratio values as uplift percentages (`1 = 0%`). Ordinary SHAP plots use one shared reference per plot. Stacked SHAP stays on the linear predictor contribution scale and does not accept a rescale control. The SHAP frontend preserves matching legend visibility across active-model switches only when the selected features, plot type, and legend series still match.
- LightGBM-specific training, objective handling, offsets, SHAP, feature importance, tree extraction, and tree label normalization belong in backend GBM modules, not in frontend code.
- GBM tree routes read persisted `tree_table.parquet` artifacts only and do not import LightGBM. The list route returns compact tree metadata; the detail route returns a frontend-ready split/leaf hierarchy with compact numeric thresholds, decoded categorical thresholds, edge labels, default-branch markers, cover percentages, and node values for colouring. Long categorical split display labels are summarized while full split labels remain available in tooltip fields, and frontend node clicks highlight the selected root-to-node path.

**Performance timings**

- The footer shows approximate diagnostic timings for the active tool, for example `DuckDB`, `JSON`, tool render time, and `Total`.
- Timing values can use `ns`, `us`, or `ms` depending on duration.
- `DuckDB` is measured on the Python server for the active tool API request. UK maps use a route-local DuckDB execute/fetch timer so the footer can show whether the database query is the bottleneck.
- This does not include browser-to-server network latency, JSON transfer or parsing, profile table rendering, chart drawing, map drawing, GeoJSON loading, or map tile loading.
- `Dataset render`, `Profile render`, `Chart render`, `Histogram render`, and `Map render` are measured in the browser after data arrives. All tools also show `JSON` and `Total`; `Total = DuckDB + JSON + render` using the rounded millisecond values shown in the footer.
- Cached UI rerenders can update render timing without running a new DuckDB query, so DuckDB may show the last cached query time. Collapsing the filter footer hides the timing monitor with the filter input.

**Local server behavior**

- CLI launches use token-protected URLs by default.
- `--no-token` disables token checks for local-only use.
- Main app header `Stop app` and `Open monitor` buttons are hidden by default. `--buttons`, `serve(..., buttons=True)`, or `create_app(..., header_buttons=True)` shows both controls.
- In notebook-style runtimes with an existing event loop, `serve()` and `run_app()` start the Uvicorn server in a background thread and return the URL.
- In a normal terminal or Python shell, server calls block until stopped.
- When enabled, the browser Stop app button calls `POST /api/shutdown`; health polling greys out the page after server shutdown. The monitor page remains available at `/monitor` with the normal token rules even when the main app header button is hidden.

## UI Direction

- Keep the app dense, utilitarian, and work-focused.
- Preserve chart space; controls should stay compact and avoid unnecessary wrapping.
- The sidebar is resizable so users can trade space between long column names and the chart.
- Response controls sit above the x-axis feature list because response selection is usually the first workflow choice.
- Chart/Table controls sit before the filter bar.
- Saved-filter selections populate and apply the filter expression immediately. Manual filter edits require Enter or Apply.
- Chart animations are disabled for fast interaction.
- The app should continue to work as a static ECharts and Leaflet frontend unless future tool complexity justifies a larger frontend framework.

**Frontend event hygiene**

- Tool-owned `document` and `window` listeners must either be removed when leaving the tool or be strictly gated by the tool's visible state and active interaction state.
- Global capture listeners must not clear focus, native selection, clipboard state, or editable input state outside their owning tool.
- Selection-clearing helpers must receive the original event when available and skip editable targets such as inputs, textareas, selects, and contenteditable nodes.

## Testing

Use these tiers to keep iteration focused while preserving the full suite.
Narrow tiers are for local speed only; do not delete, skip, or weaken tests just
to reduce count.

- Syntax checks:

```bash
.venv/bin/python -m compileall src tests
find src/py_lucidum/static -path '*/vendor/*' -prune -o -name '*.js' -print0 | xargs -0 -n1 node --check
```

- Fast non-model backend tests, normally the quickest local feedback loop for
  non-modelling backend changes:

```bash
.venv/bin/python -m unittest \
  tests/test_cli.py \
  tests/test_column_profile.py \
  tests/test_dataset_viewer.py \
  tests/test_demo_dataset.py \
  tests/test_features.py \
  tests/test_histogram.py \
  tests/test_line_bar.py \
  tests/test_telemetry.py \
  tests/test_uk_map.py
```

- Model tests, which include the slower GLM fit/tabulation coverage:

```bash
.venv/bin/python -m unittest tests/test_glm.py tests/test_gbm.py
```

- Static frontend contract tests:

```bash
.venv/bin/python -m unittest tests/test_static_assets.py
```

Prefer future behavior tests over exact JS/CSS string-contract tests where
practical. Exact asset-string checks are still acceptable for stable contracts
such as asset registration, cache-control behavior, and intentionally documented
UI text or selectors.

- Browser smoke tests. The helper mirrors Dropbox CloudStorage checkouts before
  launching Playwright and defaults to `tests/test_browser_smoke.py` so it does
  not rerun the full backend suite:

```bash
.venv/bin/python scripts/run_browser_smoke.py
```

Browser smoke coverage should include cross-tool focus and listener regressions
after visiting tools that install global listeners, especially document/window
capture handlers.

- Full pre-commit checks: run the complete command block below.

Standard checks before committing:

```bash
.venv/bin/python -m unittest discover -s tests
.venv/bin/python -m compileall src tests
node --check src/py_lucidum/static/app.js
node --check src/py_lucidum/static/app/main.js
node --check src/py_lucidum/static/app/dataset-viewer-tool.js
node --check src/py_lucidum/static/app/column-profile-tool.js
node --check src/py_lucidum/static/app/line-bar-tool.js
node --check src/py_lucidum/static/app/histogram-tool.js
node --check src/py_lucidum/static/app/uk-map-tool.js
node --check src/py_lucidum/static/app/shared/api.js
node --check src/py_lucidum/static/app/shared/format.js
node --check src/py_lucidum/static/app/shared/model-ui.js
node --check src/py_lucidum/static/app/shared/schema.js
node --check src/py_lucidum/static/app/shared/tabulator.js
node --check src/py_lucidum/static/app/shared/timing.js
node --check src/py_lucidum/static/app/glm-tool.js
node --check src/py_lucidum/static/app/glm-formula-builder.js
node --check src/py_lucidum/static/app/glm-model-navigator.js
node --check src/py_lucidum/static/app/glm-tabulations.js
node --check src/py_lucidum/static/app/gbm-tool.js
node --check src/py_lucidum/static/app/gbm-evaluation-chart.js
node --check src/py_lucidum/static/app/gbm-feature-parameter-controls.js
node --check src/py_lucidum/static/app/gbm-model-navigator.js
node --check src/py_lucidum/static/app/gbm-tab-orchestration.js
node --check src/py_lucidum/static/app/gbm-shap-tool.js
node --check src/py_lucidum/static/app/gbm-shap-chart.js
node --check src/py_lucidum/static/app/gbm-stacked-shap-tool.js
node --check src/py_lucidum/static/app/gbm-stacked-shap-chart.js
node --check src/py_lucidum/static/app/gbm-tree-viewer.js
node --check src/py_lucidum/static/app/model-tool-shell.js
.venv/bin/python scripts/run_browser_smoke.py
git diff --check
```

Optional full browser smoke check:

```bash
.venv/bin/python scripts/run_browser_smoke.py
```

To pass custom pytest options or a different target, add them after `--`:

```bash
.venv/bin/python scripts/run_browser_smoke.py -- --durations=20 -q
.venv/bin/python scripts/run_browser_smoke.py -- tests/test_browser_smoke.py::BrowserSmokeTests::test_gbm_tool_loads_feature_grid -q
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
- Parquet folder input validation, direct-child file selection, combined schema metadata, default tool querying, and GLM/GBM rejection.
- GLM config without optional dependencies, formula validation/comment stripping and `offset(...)` extraction, lazy dependency failures, training jobs, coefficient/diagnostic artifacts, active-model mutation routes, tabulation routes/artifacts, and `glm_prediction` / `glm_prediction_rate` / `glm_tabulated_prediction` data-source publishing.
- GBM validation, sidecar model store behavior, optional dependency failures, native runtime dependency failures, live job progress, active-model feature/parameter refresh, model data-source publishing, Gain ordering, SHAP row limits, SHAP plot aggregation routes, tree summary/detail routes, and chart/map use of prediction sources.
- Browser smoke behavior for loading profile, chart, histogram, map, and GBM tools without unexpected extra API requests, stale active-model state, or leaked cross-tool focus/listener side effects, including live GBM progress, the GBM tree viewer, and the GBM SHAP screen.

## Future Work

- Reduce private-helper imports between modelling and chart code by formalizing shared modelling/chart contracts where reuse is stable, while preserving intentional Line/Bar consumption of GLM/GBM model outputs.
- Future modelling routes, query code, and frontend assets should live inside their tool packages unless shared behavior emerges.
- Model outputs that need plotting should publish tabular artifacts through the shared data-source contract so Line/Bar can plot them without knowing model-specific concepts such as SHAP, residuals, or lift tables.
- Performance tests should be opt-in and target generated large datasets where practical, measuring schema load, aggregation time, repeat-query time, memory use, returned row count, and payload size.
- License checks should verify runtime and frontend dependencies are compatible with public distribution.
- React/Vite or another frontend framework can be reconsidered later if the static frontend becomes a maintenance constraint.

## Maintenance Rules

- Version bumps:
  - Package versions live in `pyproject.toml` as plain `MAJOR.MINOR.PATCH` values such as `0.1.1`; the UI adds its own `v` prefix when displaying the app version.
  - Before every commit, unless the user explicitly instructs otherwise, increment the package version by at least one patch level with `python scripts/bump_version.py patch`, for example `0.1.1 -> 0.1.2`.
  - Run `python scripts/bump_version.py minor` for more substantial feature commits that should advance the minor version, for example `0.1.1 -> 0.2.0`.
  - Run `python scripts/bump_version.py major` for incompatible major releases, or `python scripts/bump_version.py set 0.2.0` to set an explicit release version.
  - The sidebar version text must always reflect the same package version: `/api/schema` should publish `app_version` from `pyproject.toml`, and the frontend should display it as `lucidum v<version>` without maintaining a separate hard-coded sidebar version.
  - Always report the final package version number back to the user when handing off commit-ready work.
  - Optional local aliases:
    - `git config alias.cpatch '!python scripts/bump_version.py patch && git add pyproject.toml && git commit'`
    - `git config alias.cminor '!python scripts/bump_version.py minor && git add pyproject.toml && git commit'`
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
