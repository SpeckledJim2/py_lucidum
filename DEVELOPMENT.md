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
- `py_lucidum.tools.line_bar` implements chart aggregation and line/bar routes. `query.py` retains the one-feature pipeline and dispatches requests containing exactly two `groupings` to the additive `two_feature.py` pipeline.
- `py_lucidum.tools.histogram` implements filtered histogram binning, exact metric summaries, and histogram routes.
- `py_lucidum.tools.uk_map` implements UK map aggregation and UK map routes.
- `py_lucidum.tools.glm` implements the opt-in `glum` GLM tool. GLM validation, training jobs, persistence, coefficient/model-detail routes, and model-output data sources live in separate backend modules.
- `py_lucidum.tools.gbm` implements the opt-in LightGBM tool. GBM active config payloads, training, validation, persistence, tree summary/detail, model-text interaction-group extraction, and model-output data sources live in separate backend modules; the frontend only edits settings, starts jobs, polls status, and renders returned diagnostics. `tools/gbm/interaction_group_model.py` is standard-library-only and remains usable without importing optional modelling dependencies.
- `py_lucidum.tools.specifications` implements the opt-in `specs` tool for editing feature, KPI, and filter specification CSV files. It reads and writes raw CSV rows, continuously validates against the durable spec loaders, generates unsaved starter drafts for missing/disabled specs, and refreshes enabled app metadata after successful saves.
- `src/py_lucidum/static/app.js` is a native ES-module bootstrap. `src/py_lucidum/static/app/main.js` owns the app shell/coordinator, shared sidebar/filter/FAVOURITES/KPI controls, tool selection, and cross-tool invalidation. `src/py_lucidum/static/app/dataset-viewer-tool.js` owns the Dataset Viewer frontend, `src/py_lucidum/static/app/column-profile-tool.js` owns the Column Profile frontend, `src/py_lucidum/static/app/line-bar-tool.js` owns Line/Bar orchestration and `src/py_lucidum/static/app/line-bar-two-feature-chart.js` owns its surface/lines/heatmap option builders, `src/py_lucidum/static/app/histogram-tool.js` owns the Histogram frontend, `src/py_lucidum/static/app/uk-map-tool.js` owns the UK Mapping frontend, `src/py_lucidum/static/app/glm-tool.js` owns GLM high-level orchestration, API mutation, and model build/detail flow, `src/py_lucidum/static/app/glm-formula-builder.js`, `src/py_lucidum/static/app/glm-model-navigator.js`, and `src/py_lucidum/static/app/glm-tabulations.js` own focused GLM frontend submodules, `src/py_lucidum/static/app/gbm-tool.js` owns GBM high-level orchestration, API mutation, and cross-tool invalidation, `src/py_lucidum/static/app/gbm-feature-parameter-controls.js`, `src/py_lucidum/static/app/gbm-evaluation-chart.js`, `src/py_lucidum/static/app/gbm-model-navigator.js`, and `src/py_lucidum/static/app/gbm-tab-orchestration.js` own focused GBM frontend submodules, `src/py_lucidum/static/app/gbm-shap-tool.js` and `src/py_lucidum/static/app/gbm-shap-chart.js` own the GBM SHAP UI/chart split, `src/py_lucidum/static/app/gbm-stacked-shap-tool.js` and `src/py_lucidum/static/app/gbm-stacked-shap-chart.js` own the Stacked SHAP UI/chart split, `src/py_lucidum/static/app/gbm-tree-viewer.js` owns the D3 tree viewer, and `src/py_lucidum/static/app/shared/` owns import-safe shared browser helpers.
- `src/py_lucidum/static/app.css` is the stable linked CSS entrypoint and import manifest. Split styles live under `src/py_lucidum/static/styles/`; `foundations.css` and `controls.css` own shared primitives, while shell/tool files own boundary-specific selectors.
- Third-party browser libraries are vendored under `src/py_lucidum/static/vendor/`. Core ECharts loads locally from `index.html`; UK Mapping lazy-loads the vendored MapLibre GL module through `static/app/maplibre-adapter.js`. Histogram lazy-loads Tabulator for the metrics grid. GLM lazy-loads Ace for formula editing. GBM lazy-loads Tabulator for editable grids and D3 for tree diagrams. ECharts GL is lazy-loaded through `static/app/shared/echarts-gl.js` for GBM SHAP and two-feature Line/Bar surface plots.

Tool packages own their training, validation, persistence, routes, and artifacts. GLM and GBM intentionally publish model outputs through the shared `data_sources` contract, and Line/Bar intentionally consumes those outputs for prediction plotting, GBM SHAP ribbons, and GLM overlays. Optional modelling dependencies must still be imported lazily and must not become base app imports. Shared reusable logic should move into `core`, `static/app/shared/`, or a modelling/shared backend helper when there is real reuse.

Frontend tools should prefer a `createXTool({ deps })` factory. Data-driven tools should expose `buildRequest`, `fetchData`, `useCached`, and render/lifecycle methods where applicable, with shared shell behavior kept in `main.js` or `static/app/shared/`. GLM/GBM model UI helpers that are genuinely common, such as polling cadence, model-list grouping, fallback selection, action button state, resize observation, and empty/status HTML, live in `src/py_lucidum/static/app/shared/model-ui.js`.
New frontend tool styles should live in a tool-owned file under `static/styles/`; move reusable tokens, layout primitives, and shared controls into `foundations.css` or `controls.css` instead of duplicating them.

## Public Interfaces

- CLI:
  - `lucidum --demo`
  - `lucidum path/to/data.parquet`
  - `lucidum path/to/data.csv`
  - `lucidum path/to/parquet-folder/`
  - common options include `--open`, `--host`, `--port`, `--no-token`, `--buttons`, `--title-prefix`, `--x`, `--actual`, `--expected`, `--expected2`, `--denominator`, `--line-bar-favourite`, `--line-bar-favourites`, `--filters`, `--no-filters`, `--kpis`, `--no-kpis`, `--features`, `--no-features`, `--tools`, and UK map column overrides.
- Python:
  - `py_lucidum.serve(...)`
  - `py_lucidum.serve_line_bar(...)`
  - `py_lucidum.run_app(...)`
  - `py_lucidum.demo_dataset_path()`
  - `py_lucidum.extract_lightgbm_interaction_group(...)`
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
  - `POST /api/line-bar/glm-overlay`
  - `POST /api/line-bar/table`
  - `GET /api/line-bar/favourites`
  - `POST /api/line-bar/favourites`
  - `PATCH /api/line-bar/favourites/{favourite_id}`
  - `PUT /api/line-bar/favourites/order`
  - `DELETE /api/line-bar/favourites/{favourite_id}`
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
  - `POST /api/glm/models/{model_id}/open-folder`
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
  - `POST /api/gbm/models/{model_id}/open-folder`
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
- A dataset path may be a folder of direct-child `.parquet` files for non-modelling tools. Lucidum builds one DuckDB `read_parquet([...])` relation from the sorted direct children, ignores non-Parquet files and nested folders, and requires every file to have identical column-name-to-DuckDB-type mappings. The schema payload reports the folder path, direct-child Parquet `file_count`, combined byte size, combined row count, `source_kind: "parquet_folder"`, and the app-level `title_prefix`; the header displays folder names as `name (n files)`.
- Parquet folder inputs are rejected at app creation when `glm`, `gbm`, or `--tools all` enables modelling tools. GLM/GBM model stores, workers, prediction sidecars, SHAP sidecars, generated samples, and dataset workspaces remain single-source-file only.
- `GET /api/schema` includes `data_sources` and `feature_bases`. The default source is `dataset`; model outputs publish named tabular artifacts through this same contract.
- `GET /api/schema` publishes valid GLM/GBM source columns and row counts from stat-keyed Parquet artifact metadata, reusing the readable dataset schema for joined source columns. It must not construct full model relations or run SHAP row-binding validation merely to publish source metadata; malformed or unsupported legacy artifacts may fall back to relation inspection. Query-time relation construction and compatibility checks remain authoritative.
- `GET /api/schema` does not precompute numeric band suggestions. Column payloads keep `band_suggestion` for compatibility, normally as `null`; chart tools request initial numeric band estimates lazily through `POST /api/banding/suggestion`.
- `GET /api/schema` excludes unreadable columns from `columns` and reports them as `invalid_columns` with sanitized errors. Normal tools should use the safe column maps; only diagnostics or choosers that explicitly report invalid columns should use the all-column map.
- Prefer doing derived or convenience calculations on the fly when the user selects a tool, source, feature, or filter. Avoid startup-time pre-calculation unless it is specifically part of the desired startup behavior, such as Column Profile's initial landing view.
- Line/Bar accepts a `source` request field and defaults it to `dataset`. Unknown sources are rejected before query execution.

**Defaults, saved filters, KPIs, and feature specs**

- Without explicit defaults, the x-axis starts with the first dataset column, Actual starts with the first numeric column, and Expected starts as none.
- CLI options, programmatic defaults, and URL parameters can override initial selections. `line_bar_favourite` / `--line-bar-favourite` selects a saved Favourite by id or case-insensitive name; when it is absent, the first saved Favourite in persisted order is restored automatically when favourites exist. Favourite startup state must be applied before the first visible tool refresh so users never see a default chart/map before the restored state. `line_bar_favourites_path` is accepted by `serve(...)`, `serve_line_bar(...)`, and `create_app(...)`; the CLI equivalent is `--line-bar-favourites`. It is the server-side JSON storage path for favourites and must not be added to generated URLs or browser query parameters.
- Line/Bar feature and Expected picker lists default to A-Z ordering. The Expected picker section starts collapsed on every app launch and can be reopened for the current browser session with the splitter chevron; its open/closed state is not persisted.
- Saved filters load from an explicit `--filters` path, otherwise `./filter_spec.csv`, otherwise `./specs/filter_spec.csv`.
- `lucidum --demo` falls back to packaged demo filter, KPI, and feature specs when a spec kind has no explicit path, no `--no-*` flag, and no current-working-directory default spec file.
- Saved-filter CSV files must have exactly `theme,name,expression` columns. CSV order controls theme order and row order.
- KPI specs load from an explicit `--kpis` path, otherwise `./kpi_spec.csv`, otherwise `./specs/kpi_spec.csv`.
- KPI spec CSV files must have exactly `group,name,actual,denominator,decimals,format` columns. `denominator` aliases `N`, `Average row value`, empty, and `__none__` all mean average row value; `format` is `number`, `currency`, or `percent`. Percent formatting displays proportions as percentages, so `0.1` displays as `10%`.
- KPI rows are a single-selection convenience layer over Actual and Weight. They appear as read-only rows in their own sidebar KPI accordion and set only Actual, Weight, decimals, and formatting. Manual Actual/Weight changes keep the KPI row active only when both selects exactly match a spec row.
- The shared metric controls remain visible for every active tool while the sidebar is expanded. Only their visible sidebar headings are renamed to `Numerator` and `Denominator`; internal IDs, request fields, chart/table Actual and Weight labels, KPI CSV columns, and model metadata remain unchanged. Their summary values are owned by the shared app shell, update from the selected source/Actual/Weight and active filter, and must not depend on which tool is active.
- The Denominator selector groups dataset numeric columns and the active primary `glm_prediction` / `gbm_prediction` outputs. Requests for metric summary, Line/Bar, Histogram, and UK Map carry optional `denominatorSource`; absent values mean `dataset`. Model denominators are joined to dataset rows through `__lucidum_row_id`, participate in request/cache identity, and preserve each tool's existing ratio, volume, exclusion-warning, and formatting rules. Prediction-rate and tabulated-prediction columns are not valid model denominators.
- Favourites persist `denominatorSource` beside the existing denominator string. Legacy favourites resolve against `dataset`; a saved GLM or GBM denominator is a logical active-model reference and resolves to the currently active source of that type. Explicit GLM/GBM activation and automatic promotion after deleting the active model atomically apply the final active model's saved dataset response and denominator/offset before refreshing the visible tool once; model-backed x-axis and Expected selections rebind in the same schema transaction, and inactive tools remain stale until opened. A changed metric pair clears an incompatible active Favourite and synchronizes any matching KPI. Rename, inactive-model deletion, and build completion do not overwrite the selected metrics. Deleting the last matching model preserves an unavailable model-denominator selection with an actionable warning.
- GLM and GBM build payloads include denominator-source metadata. Frontend controls disable model building when that source is not `dataset`, and backend validation rejects the same state so APIs cannot bypass the safeguard. Dataset columns named `glm_prediction` or `gbm_prediction` remain valid denominators because the source metadata, not the name, determines the restriction. GBM `init_score` is unchanged and remains the supported prediction-chaining path.
- KPI decimals and format apply to Actual and Expected response values in metric titles, line/bar labels and response axes, table response cells, map labels/tooltips/popups, and map legend values. Weight and row-count formatting is unchanged. KPI sidebar rows give the name full width priority; the muted numerator/denominator detail uses any remainder and ellipsizes first.
- Feature specs load from an explicit `--features` path, otherwise `./feature_spec.csv`, otherwise `./specs/feature_spec.csv`.
- Feature spec CSV files must start with `Feature,Grouping`. Reserved metadata columns immediately after `Grouping` are `Base`, `min`, `max`, and `banding`; scenario columns start after the contiguous reserved metadata block. Older specs without these metadata columns remain valid and treat every column after `Grouping` as an ordered GBM scenario name. Scenario cells include the feature when they contain the word `feature`, case-insensitive.
- `Grouping` is displayed in the GBM Feature table between `Feature` and `Use` and supplies the GBM feature interaction-constraint options. Selecting a loaded scenario updates `Use` only for usable, non-reserved dataset features.
- `Base` values are exposed through schema and GBM SHAP config payloads. For numeric chart axes, rescale matching uses the displayed numeric band containing the base, falling back to the nearest displayed numeric band when needed. GLM tabulations also use `Base` to define base cells, and numeric `min/max/banding` to define rating-table grids.
- The free-form DuckDB filter expression lives in the collapsible footer; hiding the footer does not clear the active filter.
- Saved-filter rows support `Single`, `Multi`, and `Grouped` modes. `Single` keeps one selected row, `Multi` toggles rows and combines them with the active All/Any/Not all/None operator, and `Grouped` toggles rows while combining rows within a theme with `OR` and selected themes with `AND`. Grouped expressions use only the parentheses required to protect an `OR` group or a saved row with a top-level `OR`; Multi retains explicit parentheses around each selected row. Sidebar FILTER rows give the filter name full width priority; the muted expression uses any remainder and ellipsizes first.
- `--no-filters`, `--no-kpis`, and `--no-features` disable discovery for app metadata and prevent the Specifications tool from preloading default-discovered files for those disabled spec kinds.
- The default-enabled Specifications tool exposes Feature, KPI, and Filter spec screens backed by `/api/specs/*`, edits raw CSV fields rather than normalized schema metadata, continuously validates the same file contracts used at startup/reload, and atomically saves the selected spec file. Missing or disabled specs return generated starter drafts only: Feature specs get one row per valid dataset column with only `Feature` populated, while KPI and filter specs get one blank row with visual-only placeholder hints. Generated drafts are not written until Save; the file-path line labels the save target and marks new files or suppressed existing files. Saving a disabled spec writes the CSV for editing but leaves the running app metadata disabled until restart/recreate without the corresponding `--no-*` option.
- Filters are DuckDB `WHERE` expressions and apply before column profiling, chart aggregation, histogram binning/statistics, map aggregation, table rendering, low-weight grouping, response transforms, and sigma calculations.

**Dataset Viewer**

- Dataset Viewer is default-enabled after Line/Bar in the built-in registry order. Explicit `--tools` / `create_app(..., tools=...)` selections preserve the caller's order exactly after alias normalization and include Dataset Viewer only when `dataset-viewer` is requested.
- Requests return readable raw dataset columns and up to 100 filtered preview rows, plus `has_more` when another row exists beyond the cap. They deliberately do not compute exact total or filtered row counts.
- The active footer/saved-filter expression is applied server-side before the display cap.
- Whole-table search, column sorting, axis-exclusive whole-row or whole-column selection, transpose, optional alphabetical column ordering, right-click column pinning, right-click cell copy, and right-click selected row/column CSV output are client-side over the loaded display rows only. The 50px toolbar groups its controls under `Columns` and `View`: the borderless column editor is transparent when empty, uses the sidebar selected-row background when populated, and keeps its clear command inset and absent from interaction when empty. A session-only pointer/keyboard divider resizes Columns while View retains a constant width and its trailing metadata divider moves with it; the divider is hidden when the toolbar stacks responsively. Transpose and `A-Z` alphabetical ordering use independent stable-width borderless text toggles: muted 12px/400 when off and accent 12px/700 when pressed; `A-Z` retains the full `Alphabetical columns` accessible label and tooltip. Pinning is session-only during normal use but is saved when included in a Dataset view favourite. Pinned columns retain pin order, newly pinned columns append to that ordered block, and `A-Z` affects only unpinned columns. Selecting exactly one pinned dataset column reveals `move pinned column` controls in the Columns heading; they move one position without wrapping, use left/right chevrons normally and up/down chevrons when transposed, retain selection and keyboard focus after movement, and persist their order in the existing ordered `datasetView.pinnedColumns` favourite array. The grid count says `First N shown` only when the preview is capped by `has_more`; otherwise it says `N shown`. Pinned columns are marked with a pin beside the displayed column name, listed in display order in the grey grid count text, and remain visible ahead of text-search matches, but they are not frozen or sticky; in transpose mode, pinned original columns render as ordered top rows that scroll normally with the rest of the table. Header labels select whole columns/rows while compact sort symbols cycle ascending, descending, and unsorted states in both orientations. In transpose mode, search matches only the original column names shown in the first transposed column and keeps every loaded preview row as a `Row n` column; selected preview rows or dataset columns are preserved when toggling transpose.
- Unreadable dataset columns are omitted from the grid and returned as payload warnings without rendering a table overlay.

**Column profile**

- Column profile is default-enabled but not mandatory: explicit `--tools` and `create_app(..., tools=...)` selections include it only when requested and preserve the caller's order. It appears after Line/Bar and Dataset Viewer in the built-in default/`all` registry order.
- Summary requests return every readable dataset column with inferred kind, DuckDB type, filtered missing count, distinct count, and min/max for numeric/date-like columns. The Column Profile presentation labels DuckDB `BOOL`/`BOOLEAN` columns as `logical` in summary and detail type badges while their shared analytical `kind` remains `categorical`. Auto-mode summaries are exact when `filtered rows * readable columns <= 10,000,000`; larger summaries use the first 100,000 filtered readable rows and include `calculation` metadata plus a UI action to recalculate all rows exactly. Passing `mode: "full"` to `/api/column-profile/summary` forces exact all-row summary stats.
- Unreadable columns are omitted from profile summaries and returned through `skipped_columns` with sanitized errors.
- Detail requests return value counts for categorical columns and histogram/stat tables for numeric/date-like columns.
- The 50px Column Profile toolbar groups its controls under `Columns` and `Rows`, separated from each other and the metadata by fixed grey dividers. Column search uses the Dataset Viewer treatment: transparent and borderless when empty, the sidebar selected-row background when populated, and an inset clear command that is absent from interaction when empty. `Use 100k` and the compact `Use all` (`Use all rows` accessibly) are mutually exclusive stable-width borderless text buttons: muted 12px/400 when inactive and accent 12px/700 when selected. Their request-time lock retains the pointing-hand cursor rather than showing a busy cursor. At `640px` and below, Columns remains a full 50px first row while Rows, its toggles, and right-aligned metadata share a compact 40px second row beneath a continuous horizontal separator; both desktop vertical dividers are hidden and the toolbar remains overflow-free.
- Profile requests respect the same active footer/saved-filter expression as the other tools.

**Line and bar chart**

- Line/Bar is default-enabled, registered first when enabled, and opens on startup for ordinary launches unless a valid `?tool=` URL parameter or saved favourite startup state selects another enabled tool.
- Saved Favourites persist under `.lucidum/datasets/<dataset-slug>/line_bar/favourites.json`, outside the exact signature workspace, so replacing a dataset at the same path keeps saved views. `line_bar_favourites_path` / `--line-bar-favourites` overrides that path completely for the running server and reads/writes exactly the supplied JSON file, creating its parent directory on first write. Malformed favourites JSON must return a clear unavailable/error response and must not be overwritten during that request. The sidebar FAVOURITES accordion is a flat list of editable saved favourites; saved favourite editing is name/order/delete only, and KPI presets live in their own sidebar accordion. New saved favourites store `view.scope` as `metrics`, `metrics_filter`, `line_bar_view`, `histogram_view`, `map_view`, or `dataset_view`; legacy favourites with no scope default to `line_bar_view`. Each saved view may store raw filter expression, saved FILTER mode/operator, and selected saved FILTER rows; metric-capable favourites may also store Actual, Weight, and KPI identity. Line/Bar view favourites also store chart/table controls, x-axis, and Expected selections, including `emptyPeriods` as `show` or `skip` and `missings` as `show` or `hide` inside each grouping; missing or invalid legacy values default to `show`. Dataset view favourites store transpose, alphabetical columns, select-columns text, pinned dataset column names, explicit user-resized column widths, and column sort state under `view.datasetView`; normal Dataset sort columns are saved by dataset column name, while transposed sort uses `__field` or `rN` preview-row fields. Histogram view favourites store bins, distribution, y-axis type, log scale, and sample mode under `view.histogram`; Map view favourites store UK map level, base map, palette, sliders, and camera under `view.map`. Restoring a Metrics favourite changes only Actual/Weight/KPI formatting, Metrics + filter also restores filter state without changing the current tool, Line/Bar view restores the full Line/Bar chart/table state and switches to Line/Bar, Dataset view restores filter/Dataset Viewer presentation and switches to Dataset Viewer, Histogram view restores metrics/filter/histogram presentation and switches to Histogram, and Map view restores metrics/filter/map presentation and switches to UK Mapping. At startup, an explicit favourite name/id wins; otherwise the first saved favourite is restored without silently skipping invalid favourites. Validation is scope-aware: metric-only favourites validate numeric response/weight fields, filter-scoped favourites also validate filter SQL and saved FILTER rows, Line/Bar views validate saved source, columns, expected selections, and chart feature state, Dataset views validate filter SQL and warn for missing pinned or sorted columns, and Histogram/Map views do not require Line/Bar x-axis or Expected fields. Blocking validation errors prevent restore and non-blocking missing KPI/FILTER/pinned-column/sorted-column rows are reported as warnings while preserving the raw valid filter expression.
- X-axis features can be integer, numeric, string/categorical, date, or datetime.
- The x-axis picker accepts one or two selected rows and never disables unselected rows. An ordinary row click or arrow navigation makes that row the sole Feature 1, including when two other features were selected. Command-click on macOS or Ctrl-click on Windows/Linux toggles a second grouping; modifier-clicking an unselected third row leaves the current pair unchanged. Selection markers are hidden with one feature and show stable `Feature 1`/`Feature 2` labels with two. Modifier-removing Feature 2 returns to Feature 1, while modifier-removing Feature 1 promotes Feature 2 and its grouping state. With two features, a compact double-ended-arrow command beside the x-axis heading atomically swaps their order and complete grouping state. It performs one chart refresh in Chart view, or one table refresh while marking the chart stale in Table view. Context-menu navigation to Line/Bar intentionally clears Feature 2.
- The Expected picker follows the same ordinary/modifier selection convention while preserving response order: an ordinary row click or arrow navigation makes that response the sole Expected line, Command-click on macOS or Ctrl-click on Windows/Linux adds or removes a second, and modifier-clicking an unselected third row leaves the pair unchanged. A sole selected row cannot be toggled off; `No expected line` is the explicit clearing control. Expected rows remain enabled and fully opaque with two selections. Saved favourites, URL defaults, and restored views may still initialise the picker with two Expected responses.
- `/api/chart`, `/api/line-bar/chart`, and `/api/line-bar/table` accept top-level `missings: "show" | "hide"` for one feature and an optional `groupings` array containing one or two `{feature, source, bandWidth, quantileMode, dateBucket, asFactor, tailPercent, missings}` objects. Exactly two groupings use `two_feature.py`; a one-item grouping is mapped back to the one-feature pipeline. Missing or invalid `missings` values normalise to `show`. The legacy top-level `tailPercent` is accepted as the fallback for grouping objects that omit it and remains the Feature 1 response/favourite compatibility alias. The frontend sends the legacy contract for one feature and the grouping contract only when Feature 2 is selected.
- Two-feature plot type is derived, not chosen. Untreated date/datetime groupings are continuous regardless of calendar bucket; fixed-width numerics are continuous unless quantiled or treated as factors. Two continuous groupings render a surface, one continuous plus one factor-style grouping renders a combined line/stacked-bar chart, and two factor-style groupings render a heatmap. Feature 2 is horizontal and Feature 1 vertical for surfaces and heatmaps; surface numeric/time axes use the exact minimum and maximum plotted grouping coordinates rather than including zero automatically. Date coordinates are converted from the returned ISO sort values to UTC timestamps in the chart client, use ECharts time axes in surfaces, and reuse the one-feature chronological date formatting, density, rotation, zoom, and relabelling policy in mixed charts. The 3D surface footprint scales within bounded width/depth limits from the available chart dimensions and is rebuilt from cached rows on resize without another API request. In mixed plots the continuous feature is horizontal and the factor-style feature defines colour-matched response-line and Weight/N-bar pairs. Heatmap category axes use measured chart space independently: horizontal labels try the existing rotations and sizes down to 8px and reappear when data zoom makes them fit, while vertical labels remain horizontal/right-aligned and reappear when resize provides at least 8px. An unreadable axis suppresses tick labels only, reclaims their margin, and keeps its title and tooltips. Visible vertical labels retain the responsive width cap and manual truncation for exceptionally long values. One legend item toggles both series for a factor group.
- Two-feature mode replaces the one-feature settings with Plot where needed, independent `Feature 1 missings` / `Feature 2 missings` controls, independent band/quantile/date controls, independent numeric/date `Treat as factor` buttons, and an independent Tail grouping control for each eligible feature. Dates default to continuous; their factor override and missing setting are swapped, promoted, saved, and restored with the grouping. Mixed line/bar plots always show Weight/N as stacked bars and show Plot only when Actual plus selected Expected responses provide multiple line choices; Weight/N is not a mixed-plot choice. Surface and heatmap Plot controls continue to offer Actual, each selected Expected response, and Weight/N. Factor/factor heatmaps add a separate Labels control with each mode admitted independently by the available cell dimensions: Actual uses `resp0` with active-KPI formatting, Weight uses grouped `volume`, and Both renders those values on two centred rows when both lines fit. Label font size is recalculated from cell width/height between 12px and 7px, with per-cell white/navy contrast selection; an individual mode is hidden only when it cannot fit at 7px, without clearing the saved preference. Changing Plot or heatmap Labels only redraws cached response rows. Low-weight grouping, sort, transforms, empty periods, sigma, and SHAP/GLM partial dependence stay hidden and are sent as neutral values while Feature 2 is selected, but their one-feature frontend state is retained.
- Both two-feature Tail grouping controls default to `-` (off). A fixed-band numeric tail Winsorises only that grouping at its percentile cutoffs; numeric quantiles, dates, and unbanded numerics ignore the preserved setting. A categorical tail computes marginal Weight/N after the active filter and shared validity rules, then maps individual levels at or below the selected percentage of total Weight/N to `Other`; as in the one-feature pipeline, this requires at least three levels and at least two qualifying rare levels. The mappings are calculated independently before final cell aggregation. The query includes a scalar bounds relation only for numeric groupings that actually need percentile cutoffs; categorical tail mapping does not add a bounds join. Actual and Expected use the normal shared-denominator aggregation rules, while Weight/N is grouped `SUM(weight)` or row count.
- Two-feature chart rows expose `group0`/`group1`, sort values, missing flags, and numeric range metadata plus `volume`, `row_count`, and every `respN` value. With `show`, missing factor groups remain chartable and missing continuous values remain in table data/totals but are omitted by surface/line renderers with a warning. With `hide`, raw rows missing the selected feature are filtered before grouping; hiding both features excludes rows missing either. The effective population is used for Line/Bar groups, table totals/search/pagination, denominator and response summaries, transforms, overlays, and `filtered_row_count`, while `row_count` and the original global-filter text remain unchanged. The shared sidebar Numerator/Denominator summary remains global-filter based rather than adopting this tool-local filter. Shown missing-volume bars use `--missing-bar`; other bar colours are unchanged. Factor/factor heatmaps accept up to 100,000 populated grouped cells; mixed and surface plots retain the 10,000 grouped-value cap, line plots show at most 80 factor series, and surfaces reject dense grids above 40,000 cells. Guardrail messages direct users to the uncapped server-backed table.
- Two-feature tables render Feature 1, Feature 2, optional row count, Weight/N, Actual, and every Expected response. Full-result search matches both grouped feature labels/values, pagination and totals remain server-backed, and CSV copy includes both feature columns.
- Line/Bar favourite version 2 stores `tailPercent` and `missings` inside each grouping, keeps top-level `tailPercent` as the Feature 1 compatibility alias, and also stores `plotMetric` plus optional `heatmapLabels` (`none`, `actual`, `weight`, or `both`). Legacy two-feature views copy their shared tail percentage to both groupings; legacy grouping objects default `missings` to `show`, legacy views default heatmap labels to `none`, and legacy `x`/`xSource` plus band/quantile/date fields normalise to one grouping. A stale optional Feature 2 is a non-blocking warning and is dropped during frontend fallback; a stale Feature 1 remains a blocking validation error.
- The x-axis feature list has borderless `Imp`, `Original`, and `A-Z` sort modes with the same stable-width muted/bold-accent selection styling as the Line/Bar settings strip. The Expected list uses the same styling for its right-aligned `Original` and `A-Z` controls without displacing the Expected average values in the header. `Imp` appears only when an active GBM or GLM has saved feature-level importances, lists raw dataset features grouped as GBM, GLM, and `Not used`, and selects dataset features only. GBM importance prefers saved mean absolute SHAP when present and otherwise uses LightGBM Gain. GLM importance uses the persisted weighted mean absolute centered feature contribution on the GLM linear-predictor scale.
- When both active GBM and GLM prediction sources exist, Line/Bar exposes an x-axis-only virtual numeric feature `gbm_to_glm_ratio` from a `model_ratio` data source. It is computed as `gbm_prediction / glm_prediction` joined by `__lucidum_row_id`; null or zero GLM predictions produce a null ratio and use the normal missing x-axis group.
- Line/Bar data-affecting interactions share one latest-intent generation across chart/table requests, cache reuse, banding suggestions, view changes, schema reloads, and model mutations. Each completed result retains its canonical request snapshot and source revision; axes, series, labels, table columns/rows/formatters, and exports must be derived from that committed snapshot rather than live controls. While a replacement is pending, the last completed chart or table remains mounted, fully opaque, and unchanged, and the existing top-right metadata alone reports `Computing...` or `Loading table...`. Only the active view is refreshed; the inactive view is marked stale and rebuilt when selected. A latest-request failure keeps the coherent committed output and restores its compatible controls.
- Generated Line/Bar fields are semantic active-model references, not generic same-name fallbacks. `glm_prediction` and related GLM fields bind only to the current active GLM source, GBM equivalents bind only to the current active GBM source, and `gbm_to_glm_ratio` binds only to the ratio source containing both current model ids. GLM/GBM activation, deletion, and replacement invalidate Line/Bar before schema reload; reverse schema completions cannot commit, and rapid activations are serialized and coalesced to the latest requested model. When a required source disappears, fallback selection is deterministic and incompatible grouping state is reset.
- Intentional active-model changes—explicit activation, externally detected activation, and automatic promotion after deleting the active model—align Line/Bar comparisons inside the same schema transaction that applies the saved training metrics. The activated family becomes the sole primary `glm_prediction` or `gbm_prediction`, replacing empty or non-model Expected selections as well as a single model output; an exact primary GLM/GBM pair is retained, in its existing order, only when the source-aware normalized sidebar Numerator/Denominator pair is unchanged and both active models' saved metric metadata match it. Advanced prediction-rate or tabulated Expected values normalize to that primary prediction on activation. `partialDependence: none` remains unchanged. An active single partial-dependence overlay switches to the activated family; `both` is retained only under the same compatible-pair rule. Missing predictions preserve compatible non-model Expected selections, while missing model predictions or selected-feature SHAP artifacts otherwise fall back to remaining compatible state rather than retaining an incompatible model comparison. Apply this policy while Line/Bar is inactive, in Table view, and while two-feature mode temporarily hides partial-dependence controls. Rename, inactive-model deletion, build completion, and reselecting the already-active sidebar model do not invoke it. Any automatic comparison-state change clears an active Line/Bar-view Favourite without rewriting the saved view, and the active Line/Bar still performs only one replacement request.
- A committed Line/Bar chart prepends non-blocking KPI-mismatch warnings to `chartMessage` for every displayed model-derived Actual or Expected response and for active SHAP/GLM partial-dependence overlays whose saved source-aware training Numerator/Denominator differs from the request KPI. Blank, `N`, `Average row value`, and `__none__` denominators are equivalent. Group warnings by model so its prediction variants and overlay are named once, retain the requested components, and omit compatibility claims when model metadata is unavailable. Store the warning list in the canonical request presentation, refresh that snapshot for incremental GLM overlay add/remove operations, preserve it through cached render/copy paths, and hide it with the existing chart message in Table view.
- Numeric x-axis banding uses the existing visible controls only: concrete band widths, step buttons, and explicit `-` for no banding. The initial concrete width is estimated lazily from the selected source/filter when a numeric feature is selected, using a bounded sample of up to 100,000 rows.
- One-feature ordered low-weight grouping (integer, numeric, quantile, date, and datetime) retains a shown missing group separately but excludes it from low/high-tail candidates and from the minimum eligible non-missing group count. Percentage thresholds continue to use total included Weight/N, including the shown missing group. SHAP and GLM overlay grouping uses the same mapping. Categorical rare-level/`Other` behavior is unchanged under `show`; `hide` removes raw null rows before rare-level mapping.
- Numeric banding floors values to the selected band width by default. With `quantileMode` enabled, the same banding value is rounded and clamped to `1..1000`, non-missing numeric values are grouped as `Q1`, `Q2`, etc., and missing x-axis values stay in a separate `Missing` group.
- Date/datetime axes use calendar buckets. Their initial bucket is estimated for the selected source/feature using the active filter at selection time; later filter changes preserve the resolved or manually selected bucket, matching numeric banding behavior. For explicit Hour/Day/Week/Month/Year buckets, `emptyPeriods=show` is the default and adds zero-volume calendar periods between the filtered non-null minimum and maximum; response values remain null so lines gap rather than inventing observations. `skip` retains observed buckets only. Raw unbucketed dates ignore this setting, `(missing)` remains separate, generated periods participate in chart limits and table pagination/search, and real-data summaries and low-weight grouping remain unchanged.
- Actual and up to two Expected lines use a shared denominator. The first Expected line is red, the second Expected line is blue, and the first Expected line remains the reference for Expected sort and sigma bars. `Average row value` divides by valid row count; a numeric Weight column divides by `SUM(weight)`.
- Line/Bar response-axis titles use the response numerator column and selected denominator column as `Numerator / Denominator`; `Average row value` shows the numerator alone. This applies to the normal one-feature chart and two-feature mixed/surface charts. Vertical value-axis margins are derived from formatted tick-label widths so the left and right titles remain clear of their labels.
- The one-feature main legend formats only its primary Numerator entry with the same `Numerator / Denominator` convention, falling back to the Numerator alone for `Average row value`. This is display-only: the underlying response series name remains stable for legend selection, redraw persistence, tooltips, and response-axis visibility calculations. Expected, Denominator-bar, SHAP, and GLM overlay entries retain their existing labels; two-feature legends remain group-based and unchanged.
- Response transforms `0` and `1` use the selected x-axis feature's `Base` metadata when available. `0` subtracts each response's value at the base group; `1` divides by it and the frontend displays those ratio values as uplift percentages (`1 = 0%`). Actual and Expected references are independent, and sigma bounds use the Expected reference. If no Base is available, the transform keeps the historical overall-average reference; if a declared Base cannot be applied, the response includes a warning and falls back to the overall average.
- `partialDependence.mode` accepts `none`, `shap`, `glm`, or `both`. `shap` overlays SHAP percentile ribbons for the selected x-axis feature from the requested GBM, which the browser binds to its schema-active model. That GBM must expose both `gbm:<model_id>:shap_long` and `gbm:<model_id>:predictions`; missing artifacts or missing feature SHAP columns return warnings without breaking the normal chart. Ribbons use the same filter, denominator weighting, x grouping, banding, quantile mode, low-weight grouping, and response transform as the chart. SHAP values are first converted to response scale using the GBM objective link, then shifted or multiplied so the weighted p50 ribbon mean matches the requested GBM fitted prediction mean for the current chart slice. Categorical x-axes accept `sort: "shap"` when the SHAP overlay is visible (`shap` or `both`) and available, sorting by descending scaled median SHAP.
- `glm` overlays the active GLM as a dashed central line. A normal one-feature chart response carries the raw x evaluation values, pre-tail mapping, group volumes, response totals, and transform metadata that produced the currently rendered rows. Categorical display labels stay strings, but their GLM evaluation values preserve the source scalar type so logical features reach the fitted formula as booleans. Shown missing groups are scored with a null/NaN evaluation value and retained when the fitted formula produces a finite prediction; `Missings: Hide` still removes them before grouping. When the user changes `Partial dependence` from `None` to `GLM`, the frontend sends that current context to `POST /api/line-bar/glm-overlay`; the endpoint returns only the GLM series and does not recalculate the base chart. A no-interaction GLM with configured bases scores those supplied x values directly, derives fitted-mean scaling from the plotted active `glm_prediction`, and uses a unit model/selected denominator when they match. It does not construct another `Dataset` or query the dataset. Missing bases, missing required context, response exclusions, offsets, a mismatched denominator/model identity, and interaction models retain the relation-based correctness path. Selecting `None` after this overlay removes it locally. Any feature, grouping, filter, denominator, response, transform, or model change obtains a fresh chart and context. This is current UI state only: no historical chart contexts, calculated overlay results, estimators, eviction policy, or invalidation cache is retained.
- For GLMs where the x-axis feature is not in an interaction term, the relation fallback uses GLM tabulation-style base-profile prediction: vary x over rendered chart groups, hold other model inputs at Feature Spec `Base` or filtered inferred base values, and predict only one row per x-axis group before fitted-mean scaling. Simple two-way x interactions use a collapsed partner grid: numeric partners use Feature Spec grids when compact and otherwise 50 quantile-style context bins; categorical partners use observed levels up to 200; collapsed overlays cap at 2,000 context rows and 250,000 prediction cells. Higher-order interactions, multiple interaction partners, excessive partner cardinality, or collapsed guardrail overflows fall back to the deterministic sampled marginal PDP over the current filtered valid chart population, capped at 100,000 sampled rows and 2,000,000 prediction cells. When LightGBM isolation is needed, GLM overlays retain the persistent worker lifecycle so dependency imports are paid once, but the estimator is loaded for each request rather than cached. `partial_dependence.method` can be `base_profile`, `collapsed_marginal`, or `sampled_marginal`, with overlay-local timing and guardrail metadata in `partial_dependence.sample`. `both` builds SHAP and GLM overlays independently through the normal chart request, then aligns the GLM line to the SHAP/active-GBM fitted-mean baseline when both are available, and returns warnings per overlay so one unavailable model does not suppress the other.
- Low-weight grouping uses selected Weight total, not raw row count.
- Chart requests are capped after the relevant Line/Bar grouping pipeline: 100,000 populated grouped cells for factor/factor heatmaps and 10,000 x-axis/two-feature groups for other plots. Chart responses include `group_count`, `max_groups`, and `groups_truncated`; truncated chart responses warn that the table search still covers all groups.
- Table view uses `POST /api/line-bar/table` and is always server-backed. The table endpoint accepts the normal Line/Bar request fields plus `tableSearch`, `tablePage`, and `tablePageSize`; the default page size is 10,000 groups, returned pages render through a virtualized frontend grid, and there is no table-wide result cap.
- Table search is a case-insensitive substring match against grouped x-axis labels/values—or both grouping columns in two-feature mode—across the full grouped result, not just chart-returned rows. The server returns only the requested page plus a `summary` computed over all matching groups, so the `Total` footer is not page-limited.
- Table search and pagination must not refetch `/api/chart` or include table search state in chart requests. Response transforms use references from the full grouped result before chart slicing or table pagination.

**Histogram**

- Histogram is a default-enabled chart tool registered after Line/Bar and before UK Mapping. Aliases are `histogram`, `hist`, and `histo`.
- Histogram uses the shared sidebar data source, Actual, Weight, KPI, and active filter controls. With `Average row value`, `N`, empty, or `__none__` Weight, the plotted value is Actual, the x-axis title is its underlying column name, and bar volume is row count. With a numeric or model Weight column, the plotted value and x-axis title are `Actual / Weight` using underlying column names, while bar volume is Weight sum. Exceptionally long x-axis titles truncate within the available plot width without changing the underlying option name.
- Histogram view favourites use the existing favourites API/storage and capture shared metric/filter state plus binning mode, bin count, bin width, bin labels, distribution, y-axis type, log scale, and sample mode. Restoring one switches to Histogram and refreshes the chart; legacy favourites without the new fields use count mode with labels off.
- Histogram uses the shared launch-collapsed 50px settings strip and full-bleed workspace. On desktop, the chart is on the left and its metrics table starts at the 240px minimum width so the divider is initially as far right as possible; the pointer/touch/keyboard-resizable width is session-only and is not part of Histogram favourites. Its horizontal y-axis name stays inside the chart, clears the settings-strip chevron, truncates within the available plot width, and retains visible space above the plotting grid. At viewport widths up to 900px, the divider hides and the metrics table stacks above the chart.
- Rows with missing Actual are excluded. When a numeric Weight is selected, rows with missing, zero, or negative Weight are also excluded and warned. Log x-scale excludes nonpositive plotted values and warns.
- `binMode=count` keeps the existing `bins` contract: `auto` or an explicit count, explicit counts clamped to `1..10000`, and `auto` based on the effective valid count clamped to `10..200`. `binMode=width` requires a positive `binWidth` in original data units, including with log x; continuous boundaries are anchored down and up to multiples of that width so the first and last bins remain full-width, and requests that would exceed 10,000 bins are rejected. Log-x width binning retains the observed positive minimum when rounding down would produce a nonpositive boundary. Whole-number widths preserve centered integer-aware bins for integer-compatible unweighted Actuals, while non-whole widths use exact continuous boundaries. Count mode continues to use centered touching integer bins, with the requested count as an upper bound, when x log scale and Weight are off. Response `bins` reports the actual returned count and `bin_width` reports the effective width where one exists. `sampleMode=100k` with more than 100,000 valid values uses a deterministic reservoir sample for chart bars and distribution statistics while retaining exact row/exclusion counts; `sampleMode=all` keeps distribution statistics exact.
- `distribution` supports incremental and cumulative bars. `yAxis` supports sum and probability; probability divides by the same valid chart volume used for the sum axis. Log x/y settings are render controls around the returned valid histogram values.
- The response includes bin rows, selected/requested/effective binning metadata, metric rows, `stats_exact` / `stats_sampled_count` metadata, exact row-count metadata, denominator metadata, warnings, and timings. The frontend uses width-aware integer x-axis labels, 1/2/5-power-of-ten intervals on linear y-axes, and a measured left gutter that keeps fully formatted y-axis tick labels inside the chart. It outlines charts with at most 200 bins, optionally renders fitted 7-10px labels matching the plotted sum/probability value, and renders offset mean/median reference lines labelled with their KPI-formatted values beside a compact Tabulator metrics grid.

**UK mapping**

- Area and sector layers join grouped KPI summaries to bundled GeoJSON assets.
- UK Mapping uses the shared exact 50px control-strip height above the full-bleed map. Six headerless image tiles provide Blank, Esri, OSM, Aerial, OpenFreeMap Light, and OpenFreeMap Dark; Resolution and Palette use the same full-height image-over-label treatment. Resolution-specific Labels, Smooth, or Dot size controls precede the area/sector Border chooser (`Off=0`, `Thin=1`, `Bold=3`), the Strength chooser (`Faint=0.2`, `Medium=0.6`, `Solid=1.0`), Extremes, and Postcode search. Legacy favourite line weights and opacity values are mapped to the nearest supported presentation. The controls region scrolls horizontally with the shared edge fades. A separate 16px full-width information strip below it presents the exact KPI match, geometry/plotted and missing/unmatched state, row count, and filter clear command on one line using 11px text. The KPI is omitted when there is no exact match. Complete truncated text remains in titles and the live status. The viewport-control chevron immediately above Zoom In toggles both strips, reverses direction and accessible text, retains session state across tool switches, and schedules only a local camera-preserving resize. Collapse state is not part of favourites. Model-backed Denominators do not match dataset KPI rows.
- Area and sector map metadata reports source-row geometry matching without blocking partial plots. The unmatched percentage uses filtered rows with a nonblank postcode as its denominator; blank postcode rows are reported separately with all filtered rows as their percentage denominator. Sector smoothing targets with zero original rows do not affect either count. Quantile legends and hotspot ranking use only aggregate rows whose keys exist in the active GeoJSON, while `No data` continues to mean a bundled geometry without a finite plotted KPI.
- App map assets load locally, including MapLibre GL and bundled UK GeoJSON. MapLibre requires WebGL2. Nonblank base-map backgrounds are a separate external dependency: OSM and Esri/Aerial fetch raster tiles, while the displayed Light/Dark pair loads the OpenFreeMap Positron/Dark hosted vector styles; `Blank` makes no external tile requests. The removed CARTO choices are not rendered, and legacy favourite values `grey`/`darkGrey` normalise to OpenFreeMap Light/Dark before restoration. Replacing a vector style recreates every registered Lucidum source and layer against the settled style without duplicating interaction bindings, and a generation guard prevents an older style load from winning a rapid switch. For area and sector maps, OpenFreeMap line layers return above the analytical fill in their original order, the Lucidum outline returns above them, and every OpenFreeMap symbol layer returns uppermost. Unit maps instead keep vector linework below the dense unit canvas and symbols above it. OpenFreeMap road, highway, and tunnel line widths retain their source-style zoom expressions with a `0.7` multiplier; railway, waterway, boundary, and Lucidum analytical line widths remain unchanged. Both OpenFreeMap styles override text-bearing symbol paint with opaque charcoal text and a 1.75px near-opaque white halo with 0.25px blur; font, size, placement, icons, and hierarchy remain style-owned. The pair follows the app theme.
- Area analytical labels use an Off/On control and retain the postcode-area code plus formatted KPI value. Their approximate geographic size is cached from longitude span adjusted by midpoint latitude times latitude span. The median-normalised zoom offset is `clamp(0.5 * log2(area / medianArea), -3, 1.5)`; the rendered size is `clamp(6px, 9px + 2px * ((zoom - fittedZoom) + offset), 20px)`. The current area-layer fitted zoom uses the same responsive bounds calculation as `Fit UK`. A map-level CSS variable updates after zoom animation settles and through an animation-frame-throttled resize listener, so the 124 cached per-area offsets do not require marker reconstruction or a summary request and do not force label reflow during every zoom frame. New map favourites store `areaLabels`; legacy numeric `labelSize` values above zero migrate to On and other values migrate to Off.
- Sector smoothing uses a borderless six-option `-`, `N1`–`N5` chooser and a committed shared-edge adjacency sidecar generated from the bundled sector GeoJSON, then pools already-aggregated numerator and denominator values across sectors reachable within the selected neighbour depth. When smoothing is active, all sidecar sectors are smoothing targets, so sectors with no original data can be filled from valid neighbours. Raw sector fields remain available in the API rows for popup context.
- Default join columns are `PostcodeArea`, `PostcodeSector`, and `PostcodeUnit`; uppercase aliases are supported.
- Default coordinate columns are `lat` and `long`; `latitude`/`LATITUDE` and `longitude`/`LONGITUDE` aliases are supported.
- Unit points group by postcode unit, average coordinates, and plot only units with valid KPI and valid coordinates.
- Area and sector geometry use MapLibre GeoJSON sources and WebGL fill/line layers with hover tooltips and click popups.
- Unit points draw into a hidden HTML canvas that MapLibre consumes as a non-animating, geographically anchored canvas source, with a six-CSS-pixel minimum hit radius for hover tooltips and click popups. MapLibre transforms the completed canvas texture continuously during panning and zooming, then the layer redraws and reanchors after movement ends; this avoids redrawing a large unit set on every animation frame. During zoom, the previous texture remains visible for continuity while hit testing is disabled; it may briefly scale or soften before the correctly sized settled texture replaces it. Generation guards prevent an older rapid-zoom redraw from winning. Units mode replaces the former persisted numeric Dot size multiplier with `dotSizeMode`: default `adaptive` or `min`. Legacy favourites containing `dotSize` load as Adaptive, while new favourites write only `dotSizeMode`. Min paints every eligible visible unit into one snapped 1-by-1 backing-store pixel regardless of device scale. Adaptive uses the complete filtered `point_summary.plotted_count` (falling back to the aligned finite-value count) and a non-mutating fitted-extent zoom computed with the same padding and maximum zoom as `Fit UK`. At the fitted zoom, dense point sets use one physical pixel. At close zoom, the cap is 10 CSS pixels for 500,000 points and logarithmically tapers to 8 CSS pixels by 1.5 million points; 100 or fewer remain 6 CSS pixels, with intermediate counts logarithmically interpolating the original dense and sparse anchors. Growth from baseline to cap uses smoothstep over six zoom levels. Zoom, resize, effective device-pixel-ratio changes, mode changes, and favourite restoration redraw locally without fetching geometry. Compact unit responses retain every valid-coordinate unit in stable postcode order, including a null KPI value when that metric cannot be plotted. Units mode always requests the complete point set after the active global filter, with no camera bounds, and retains that geometry while the user pans, zooms, or uses `Fit UK`; camera-only movement therefore performs no map-summary API request. Once the browser has geometry for the active source/filter/coordinate identity, KPI-only requests return aligned metric arrays without resending postcode keys, row counts, or coordinates. The existing layer reuses typed Web Mercator coordinates and a 256-by-256 geographic index so settled redraws project and paint only cells intersecting the padded visible canvas. Compact arrays, geometry reuse, quantile sampling, and spatial indexing reduce transfer, CPU drawing, memory, and hit-testing costs; initial whole-layer queries and index construction still visit every applicable row and valid-coordinate point. Unit colour thresholds and the Low/High extremes ranking are scoped to the complete filtered point set rather than the current camera view. Colour thresholds use a deterministic sample of at most 100,000 aligned values while retaining the exact finite minimum and maximum; smaller unit sets and area/sector layers remain exact.
- A single delegated right-click panel at every map resolution stages theme-aware postcode-region checkboxes and commits one safely quoted `PostcodeArea IN (...)` global-filter clause only when Apply is used. It shares the transient selected-area state with area popup filtering, preserves the preceding manual filter, and performs no map requests while the panel selection is being edited.
- If no unit point columns are configured and defaults are absent, the Units layer is disabled. Explicit invalid unit point columns produce validation errors when requested.
- Regenerate the sector adjacency sidecar with `scripts/build_uk_sector_adjacency.py` after replacing `sectors_MappaR.geojson`.

**GLM and GBM**

- GLM and GBM are opt-in tools and are not part of the default user-facing tool set. `--tools all` enables them with every other tool in the built-in registry order. Explicit modelling selections preserve the supplied order and must also include `line-bar` because GLM/GBM context-menu actions open Line/Bar charts. GLM and GBM do not imply each other; request both when cross-model tabulation or comparison workflows are needed.
- The shared header application-status badge times GLM and GBM builds from the user's click through all post-training work and separately times GLM tabulation through row scoring. It returns to untimed `Ready` only after the final client refresh. Phase changes reuse the operation's original `performance.now()` timestamp; other badge uses remain untimed, and the changing elapsed text is hidden from live-region announcements.
- GLM and GBM artifacts are scoped to the exact dataset version under `.lucidum/datasets/<dataset-slug>/<dataset-signature>/models/{glm,gbm}/`. The slug is derived from the dataset filename. The signature is derived from file size, modification time, row count, and schema fingerprint. Startup scans only the current signature workspace; root-level `.lucidum/models/` folders and other dataset-version workspaces are ignored and must never break raw dataset startup.
- `/api/schema.capabilities.open_model_folders` is request-specific and true only for a loopback client, a file-backed dataset, and a supported desktop opener. The matching GLM/GBM `open-folder` endpoints remain token protected, reject non-loopback clients, validate the selected directory beneath the current store root after resolving symlinks, and never return an absolute path. They use `/usr/bin/open` on macOS, `os.startfile(..., "explore")` on Windows, or desktop-session `xdg-open` on Linux without adding a Python dependency. The navigator command is hidden when this capability is false and opens a window on the server host, not a remote browser host.
- GLM config, validation, model listing, model activation, and source discovery must work without importing optional modelling libraries.
- GLM training imports `glum`, Polars, pandas, and numpy lazily through the `glm` optional extra. These packages must not become base install dependencies. Build routes should report missing GLM dependencies as an actionable install-extra error, not a server 500. Polars owns the large-row fit/scoring path; pandas remains in the on-demand tabulation, overlay, and export paths.
- GLM accepts full `response ~ terms` formulas and RHS-only formulas using the sidebar Actual response. Raw formulas are stored in `formula.txt` with comments, but `#` comments outside quoted strings are stripped before fitting. The allowed formula context is intentionally narrow: `ifelse`, `pmin`, `pmax`, `ns`, `bs`, `cs`, `poly`, `C`, and common numeric transforms. Obvious unsafe text such as `__`, `import`, `eval`, `exec`, `open`, and statement separators is rejected before fitting. Explicit `offset(...)` terms are stripped from the fitted formula, stored in the manifest, evaluated with the same safe context, and passed to `glum.fit()`, prediction, and tabulation reconstruction.
- GLM training derives required predictor columns from Formulaic, adds only the selected response, denominator, physical `SAMPLE` column when needed, and offset-expression columns, and projects those columns plus `__lucidum_row_id` through DuckDB into Polars. Do not restore the old all-readable-column pandas load or wide frame copies; formula `.` remains the explicit case where every available source column is required.
- GLM families are `normal`, `poisson`, `gamma`, `tweedie`, `binomial`, `inverse.gaussian`, and `negative.binomial`; the first implementation uses `link="auto"`. Tweedie power and negative-binomial theta are the only exposed family parameters.
- GLM regularization is optional and defaults to unpenalized `alpha=0`. Auto regularization uses glum cross-validation over ridge, elastic net, and lasso mixes with predictor scaling; manual regularization accepts a positive alpha and `0 <= l1_ratio <= 1`. Penalized GLMs must store selected penalty metadata and suppress coefficient standard errors/p-values in the frontend.
- GLM uses the sidebar Actual, Weight, FAVOURITES, and KPI controls as the model response and denominator inputs. If a denominator is selected, training fits `response / denominator` with `sample_weight=denominator`, writes `glm_prediction` on the original response scale, and exposes `glm_prediction_rate = glm_prediction / denominator`.
- GLM `All` fits all valid rows. GLM `Training` fits only a physical uppercase/lowercase-insensitive `SAMPLE = training` column and does not create generated sample splits.
- The GLM Formula builder header uses borderless option controls for Formula tools, Model parameters, and the `All`/`Training` fit scope; inactive options are muted and active/open options are bold accent text, while `Build GLM` remains the solid primary action. Formula tools and Model parameters are mutually exclusive, either may be collapsed to an editor-only state, and the selected panel persists as frontend-only session state with Model parameters as the fresh-session default. Both open panels use `var(--sidebar-bg)` so their expanded state remains visible in light and dark themes. The Ace overlay is a vertical clear/font-size/copy command rail; Copy reads the live editor value and uses the shared clipboard feedback path.
- GLM artifacts are stored under the current dataset workspace in `models/glm/`, with compact `manifest.json`, `estimator.pkl`, `formula.txt`, `coefficients.parquet`, `feature_importance.parquet`, `predictions.parquet`, and `diagnostics.json`. New diagnostics persist `n_terms` as the fitted coefficient-row count including the intercept, `n_features` as the distinct source features referenced by those rows, and `n_interactions` as the number of distinct normalized source-feature combinations containing at least two features. Model-list payloads expose those keys only when captured and must not infer them for older artifacts. Tabulation builds add `tabulations/tabulation_manifest.json`, `tabulations/*.parquet`, and top-level `tabulated_predictions.parquet`; model-list payloads report `tabulated` from the completed manifest's presence.
- GLM model manifests persist only compact Lucidum metadata: identity, response/denominator, family/link, regularization, training scope, offset expressions, minimal formula execution flags, and `timings.elapsed_ms`. Model diagnostics and warnings live in `diagnostics.json`; raw formula text lives in `formula.txt`; raw source columns are derived from the dataset schema. When LightGBM/glum load-order protection is required, GLM fitting uses a persistent isolated worker after the first build and caches the worker-side dataset handle by source-file signature. Keep `PY_LUCIDUM_GLM_FIT_ONE_SHOT=1` as the debugging fallback for the old one-shot worker path.
- GLM training persists feature-level importances in `feature_importance.parquet`. The metric is weighted mean absolute centered feature contribution on the fitted GLM linear-predictor scale. Formulaic term metadata maps model-matrix columns to source dataset features; interaction and multi-feature transform contributions are split evenly across participating features before feature-level aggregation.
- GLM tabulations are created on demand through `/api/glm/tabulations/*`, not during GLM training. The tabulation algorithm is documented in `docs/specs/glm-tabulations.md`; keep it aligned with the R `GlimmaR` method: group terms by feature combinations, compute uncentered linear-predictor table contributions, subtract base-cell contributions, accumulate the adjustment into the base table, and score rows from the stored tables.
- The GLM Tabulations toolbar uses borderless option controls for `Table`/`Plot`, `Exp`, and `Colour`: inactive options are muted, active options are bold accent text, and inactive `Exp` represents linear scale. Keep `aria-pressed` synchronized with `py_lucidum_glm_tabulation_view`, `py_lucidum_glm_tabulation_scale`, and `py_lucidum_glm_tabulation_color`. Its XLSX command is an icon-only spreadsheet-download button with an accessible `Export XLSX` label; the shared spinner replaces the icon without changing the button size while `aria-busy` is true.
- GLM tabulation plots derive the x-axis title from the plotted feature, show every category below the 200-category density limit with the shared 10px/8px and 65-degree label policy, add x-axis zoom above 120 categories, and recalculate only cached ECharts presentation on resize. Their effective zero reference is 2px wide: raw `0` for linear plots and raw `1`, displayed as `0%`, for `Exp` plots. A borderless top-left command copies a 2x PNG with the active panel background. Interaction plots without a crosstab show their actionable warning once in the inline notice. The left selector grids and right control divider meet the central 1px rule; the overlapping pointer target remains 12px wide.
- GLM tabulation rebasing is an app-level linear-predictor gauge transform for single-model GLM non-base tables. A selected cell can transfer one scalar into `base`; either level of a two-feature interaction can transfer its opposite-feature-indexed offset vector into that opposite one-way table, creating the table when absent. Versioned active rules and generated tables are stored in the tabulation manifest. Table-scoped clear removes every rule involving that table plus generated-table dependants, restores raw sidecars, and replays retained rules; clear-all restores raw sidecars directly. Every mutation must be rollback-safe, preserve row IDs and missing flags, verify linear predictions within `1e-7`, rebuild `tabulated_predictions.parquet`, preserve the still-valid Mean/linear-SD error diagnostics, and invalidate source-scoped frontend caches.
- Ordinary GLM Tabulations model switches reuse the loaded `all_models` status payload, derive the selected table union locally, retain the existing model grid, and issue only the table/plot request. Full configuration refreshes remain required after model or tabulation artifact mutations. Untabulated GBM eligibility scans are cached by tree-table path, size, nanosecond modification time, and best iteration so unchanged tree artifacts are not rescanned on those refreshes.
- GLM and GBM tabulation XLSX export is single-model only, reads existing `tabulations/tabulation_manifest.json` manifests and parquet sidecars only, saves `<model_id>_tabulations_<scale>.xlsx` beside the selected model's tabulation sidecars, and must not import `glum`, LightGBM, pandas, or numpy.
- GLM tabulation builds estimate missing numeric feature spec `min/max/banding` from scored rows and report warnings. A feature with missing scored values gets an explicit missing table cell; when the fitted formula produces a finite contribution for that cell, row scoring uses it instead of forcing the tabulated prediction to NA. Unscorable missing cells and categorical levels not seen in training remain NA table cells; unseen levels are warned. Tables over 100,000 cells are skipped with a warning.
- GLM model and tabulation changes reload frontend schema and invalidate source-scoped tools. Preserve Column Profile cache when it is active because it depends only on the raw dataset and active filter, but refresh Line/Bar and UK Mapping because they can read `glm:<model_id>:predictions`, `glm_prediction`, `glm_prediction_rate`, and `glm_tabulated_prediction`.
- The GLM Model navigator shows `Terms`, `Features`, `Interactions`, and `Tabulated`; missing legacy counts render blank, while tabulation renders `Yes` or `-`. Both the GLM and GBM navigators label the renameable model column `Name`, including their fallback tables.
- The GLM coefficient table keeps a stable one-based `#` tied to each stored coefficient row and provides sortable `#`, term, estimate, standard-error, and p-value headers. Sorting must preserve the original coefficient index used by row context-menu navigation. Its Copy and Download header actions are icon-only borderless commands with accessible labels and visible focus outlines.
- GLM model IDs are folder names under the current dataset workspace's `models/glm/` directory and must stay source-ID safe: letters, numbers, dots, underscores, and hyphens only. Renaming a model renames the folder; source IDs are computed from the folder name and existing artifacts. Deleting the active model promotes the newest remaining model, or clears active state if none remain.
- GLM model outputs publish data sources through the shared `data_sources` contract using IDs such as `glm:<model_id>:predictions`. `glm_prediction` remains the primary `ModelPredictionSource.column`; denominator-backed models also expose `glm_prediction_rate`, and when `tabulated_predictions.parquet` exists, `GlmSourceProvider` left joins it on `__lucidum_row_id` so the same source also exposes `glm_tabulated_prediction`.
- GBM config, validation, model listing, model activation, and source discovery must work without importing optional modelling libraries.
- GBM training imports LightGBM with its Arrow bridge, CFFI, Polars/PyArrow,
  pandas, and numpy lazily through the `gbm` optional extra. LightGBM's `arrow`
  extra does not install CFFI, so `cffi` must remain an explicit Lucidum `gbm`
  dependency. These packages must not become base install dependencies. On
  macOS, LightGBM's native library may also require Homebrew `libomp`; missing
  `libomp.dylib`, Polars, PyArrow, or the CFFI Arrow runtime should be reported
  as an actionable GBM dependency error, not a server 500.
- GBM training projects only the selected response, denominator, SAMPLE, init-score, and feature columns through DuckDB into Polars. Stable sorted categorical mappings are encoded once into numeric Arrow columns shared by training, test, validation, scoring, and SHAP; logical/boolean columns use the same two-level categorical path, including Polars' lowercase `false`/`true` labels. GBM tabulation must compare source boolean values with those saved labels using the same normalization. LightGBM receives Arrow tables rather than pandas matrices. Keep pandas limited to compact tree/evaluation/config artifacts, tabulations, and exports, and do not restore large-row pandas frame copies.
- GBM model manifests retain `timings.training_seconds` and add dependency, validation, data-load, matrix-preparation, dataset-construction, fit, score, SHAP, and artifact-write timings. Response scoring must call LightGBM exactly once: raw-score mode only for supplied init scores or denominator-derived log offsets, and normal prediction mode otherwise.
- GBM training runs as an in-memory background job. `GET /api/gbm/jobs/{job_id}` returns transient `progress` while the job is queued/running, including phase, message, iteration, train/test metric points, and live evaluation history. Persisted training history is `evaluation.parquet`; frontend Evaluation Log downsampling and the borderless `Zoom tail` toggle are render-only and must not truncate this artifact. Its adjacent copy command writes the rendered chart PNG to the clipboard without a backend request.
- GBM uses a canonical uppercase `SAMPLE` column when present: `training` rows fit the model, `test` rows drive early stopping, and `validation` rows are scored as holdout diagnostics. If `SAMPLE` is absent, users can create a reusable generated 60/20/20 split stored as `models/gbm/generated_sample.parquet` under the current dataset workspace; generated splits do not mutate the source dataset.
- GBM training mode is persisted as manifest `training_mode`. `ebm` mode is available when the active sample source, either a physical dataset `SAMPLE` column or the generated sidecar split, contains `training` and `test` rows after denominator filtering. EBM uses `num_iterations` as the global cap across all leaf stages, requires `early_stopping_rounds > 0`, starts with `num_leaves=2` and `learning_rate=0.3`, then advances leaf counts through the configured `num_leaves` after stage-local test-metric plateaus.
- GBM parameter cells support grid-search braces: explicit sets like `{200, 300, 400}` and inclusive numeric ranges like `{0.05, 0.3; 0.05}`. Grid search samples combination indexes deterministically from the hypergrid without constructing the full cartesian product, pre-validates only sampled combinations, skips invalid combinations with a notice, trains valid combinations sequentially in one job, persists each as a normal model with `grid_search` metadata, and activates the best completed model by test metric when present, otherwise training metric.
- GBM always exposes `tweedie_variance_power` in the Parameters grid with default `1.5` and validates LightGBM's `1.0 <= value < 2.0` constraint for every objective. LightGBM uses it for a Tweedie objective or Tweedie metric; when neither is selected, a valid value remains accepted and persisted without affecting the fitted model or evaluation metric.
- GBM exposes `init_score` as the first parameter-table row even though it is supplied to LightGBM datasets rather than to `lgb.train(params=...)`. `none` preserves the existing denominator-derived log offset for log-link objectives. A selected numeric dataset column or fitted GLM prediction source is a full prediction-space baseline; it is transformed with the objective link, replaces the denominator-derived init score, is persisted as `init_score.parquet`, and makes LightGBM `boost_from_average` irrelevant for that fit.
- GBM `parameters.json` is reserved for LightGBM-compatible Python params that can be loaded with `json.load()` and passed to `lgb.train(params=...)`, including objective, metric, and generated numeric `interaction_constraints` when applicable. Generated interaction constraints remain exact saved training provenance but are excluded from active-model Parameter rows; the semantic Feature constraints in the manifest are restored instead and converted back to numeric feature indexes immediately before training. Lucidum-only state such as `training_mode`, selected init-score value/provenance, and EBM stage metadata belongs in `manifest.json`.
- The Parameters header copy command serializes the current edited parameter table as pretty JSON with native number and boolean types, excluding `init_score`, generated `interaction_constraints`, and row metadata. It blocks copying while any LightGBM parameter contains Lucidum grid-search braces because those expressions are not valid scalar `lgb.train(params=...)` values; row-aligned initial scores remain a separate `lgb.Dataset(init_score=...)` input.
- GBM uses the sidebar Actual, denominator, FAVOURITES, and KPI controls as the model response and offset/exposure inputs. Denominator-backed models expose `gbm_prediction_rate = gbm_prediction / denominator`. The filter controls remain hidden while GBM is active because training ignores the global filter.
- GBM response-domain validation uses the same eligible rows as training: when a dataset denominator is selected, objective checks include only rows where that denominator casts to a value greater than zero; without a denominator, they continue to check all rows.
- GBM config includes loaded feature spec groupings and ordered feature scenarios. The frontend applies scenarios as a table selection convenience only; backend validation remains the source of truth for usable features, reserved response/offset/sample columns, and monotonicity.
- The GBM Features and parameters screen keeps the Feature/Parameter width boundary and Parameter/Evaluation height split as session-only frontend state. Its Control column is a fixed narrow track, while the shared control strips and two accessible drag handles use the common `app-control-strip` and `app-resizer` primitives; resizing must redraw existing grids/charts without API requests or persisted model changes. Within the Feature grid, preserve a useful minimum Feature-name width and let the Monotonicity and active Gain/SHAP columns shrink equally when horizontal space is constrained.
- The GBM Feature header keeps its Gain/SHAP/EBM Gain view choices as stable-width borderless options and its clear/select actions as borderless commands. A sliders button controls the Feature-only setup panel containing scenario, constraint-group, and interaction-pair controls; the panel defaults closed, persists through `py_lucidum_gbm_feature_setup_open`, uses `var(--sidebar-bg)`, closes open menus when collapsed, and redraws existing feature grids without API requests. The Control-column SHAP rows and Training mode choices use the same muted borderless, blue active, and dark neutral heading treatment.
- GBM manifests record `feature_scenario` only when training starts from an explicit scenario selection. The saved scenario name and feature snapshot are compared with the current spec when the model is active; stale or missing scenarios are shown as provenance only and do not override `features.json` or `feature_config.parquet`.
- GBM feature interaction constraints are driven by nonblank Feature Specification `Grouping` values. The frontend may send selected grouping names, but the backend injects the server-loaded feature grouping map before validation and training. Training constrains only currently selected trainable features in selected groups, adds a remainder constraint for all other selected features, and persists the training-time constrained group/feature snapshot in the manifest.
- GBM constraint-group text-model creation is opt-in through `create_feature_interaction_group_models` and requires selected groups plus non-zero saved SHAP rows. Save the main model first, then create safe deterministic `model_<group>.txt` sidecars with `extract_lightgbm_interaction_group()`. Verification must reload each sidecar, use the exact in-memory encoded SHAP sample in the extracted model's feature order, predict with `raw_score=True`, and compare against `<Grouping>_INTERACTION_GROUP`. Persist full-precision maximum absolute errors, status, artifact, tree/row counts under manifest `feature_interaction_group_models`, plus `timings.interaction_group_model_seconds`. A selected group with no fitted trees records `no_trees` and no file; other extraction or verification failures fail the job. Config and UI must remain compatible with manifests lacking this block, and folder-level rename/delete operations naturally include all sidecars.
- Active-model config reports saved interaction constraints with `current`, `stale`, or `missing` group statuses. Stale or missing constraints are displayed as provenance and must not be resent for new training unless the user selects current grouping options. The Feature column uses lock subscripts `1` and `2` for main-effect-only and paired features; constrained Grouping cells use the effective selected group size. A feature must display only its effective row-level or group marker, and main-effect-only constraints override selected group membership.
- The Feature-cell context menu owns row-level constraint editing through `Constrain to main effect only (1D)`, `Remove main-effect-only constraint`, and `Add pair interaction (2D)…`. The pair action expands Feature setup and opens the existing pair manager with that feature preselected. For a paired feature, omit the add action and render one `Remove <left> × <right> pairwise interaction` action per existing pair in persisted display order. The manager separates `Add pair interaction` from `Allowed pair interactions (n)`, uses `×` only as interaction notation between feature names, and uses an explicit `Remove` button for deletion. Main-effect-only/pair and pair/group conflicts must be disabled in the browser in both directions while backend validation remains authoritative.
- GBM pair interaction allowlists are sent as `feature_interaction_pairs: [{left, right}]`. Pair constraints may accompany Feature Specification grouping constraints only when the selected groups are disjoint from every paired feature; groups containing paired features are rejected because they would permit additional interactions for that feature. Main-effect-only constraints may accompany pair mode only outside the pair list. Pair mode persists as `feature_interaction_constraints.mode = "pairs"` with `pairs`, optional disjoint `groups`/`groupings`, optional explicit `features`, and `uncovered_policy: "singletons"`; config returns those constraints for the active model, and the Model navigator labels pair-constrained models as `Pairs (n)`. Training converts explicit pairs, disjoint groups, and explicit main-effect-only constraints to LightGBM interaction constraints, then adds one singleton constraint for every uncovered selected feature. This makes pair mode an exhaustive allowlist and requires `num_leaves <= 3`; scalar violations are validation errors, grid-search violations are skipped through the normal invalid-combination flow, and the run fails if no valid combinations remain. Group-only mode retains its shared remainder behavior. Do not rewrite older manifests: config exposes `uncovered_policy` and `policy_inferred`, inferring `singletons`, `remainder`, or `unknown` from saved feature order plus numeric constraints. The UI initially represents a legacy remainder model historically, explains that retraining is strict, and switches its draft to automatic singleton locks after any feature-constraint edit. Automatic singleton locks are display-only, remain eligible for pairing, and must never be sent as explicit `feature_interaction_features`.
- GBM artifacts are stored under the current dataset workspace in `models/gbm/`, with one directory per model.
- GBM `features.json` is the persisted source of truth for the trained model's input feature names in exact LightGBM training order. GBM `feature_config.parquet` is an optional output/display artifact for the trained features, enriched with kind, monotonicity settings, Gain values, and optional `mean_abs_shap` values.
- GBM prediction and SHAP source relations derive their raw readable-column projection list from the current dataset schema. Do not persist duplicate `source_columns` lists in GBM manifests.
- GBM config, activation, rename, and delete responses must drive the UI's `Use`, `Monotonicity`, feature importance metric, model navigator, sidebar model list, and parameter tables from the active model, so switching models mirrors exactly what was trained. If both Gain and SHAP are available, the Feature table shows a single Gain/SHAP metric column and defaults to SHAP.
- Each GBM config or model-list response must snapshot the active model id once and use that snapshot for every model-derived field and active marker. Training and activation can change the persisted active model concurrently; mixing reads from different active models can leave the Feature table and SHAP choosers out of sync until reload.
- GLM follows the same one-read active-model snapshot rule for config and model-list responses. The GLM and GBM frontends apply active-model-dependent config as complete snapshots: background refreshes are discarded after a newer model mutation, model-list polling must not splice a new active id into older feature/parameter state, and model detail responses render only while their requested model remains active.
- GLM and GBM serialize active-model activation, rename, delete, and active-pointer clearing with a per-store re-entrant lock. Mutation endpoints hold that same lock until their config snapshot is built, while training, scoring, SHAP calculation, and bulk artifact creation remain outside the critical section.
- A GLM activation while Tabulations is visible must keep the mounted tabulation UI intact while synchronizing the hidden Formula Builder detail. Returning to Formula Builder must show the active model's family, formula, diagnostics, and coefficients without requiring another model selection; switching tabs within the same active model must continue preserving an edited builder draft.
- EBM active models add an `EBM Gain` metric toggle option. It reads only the persisted `tree_table.parquet`, groups effective trees by their unique split-feature combination, and replaces the Feature table with `Tree features`, `Dim`, `Trees`, `Gain`, and `% Gain`.
- GBM active-model switching must also mirror the persisted `training_mode` radio state. The EBM radio group is hidden when `ebm_available` is false.
- GBM model changes reload frontend schema and invalidate source-scoped tools. Preserve Column Profile cache when it is active because it depends only on the raw dataset and active filter, but refresh Line/Bar and UK Mapping because they can read model-output sources such as `gbm:<model_id>:predictions`, `gbm_prediction`, and `gbm_prediction_rate`.
- GBM model IDs are folder names under the current dataset workspace's `models/gbm/` directory and must stay source-ID safe: letters, numbers, dots, underscores, and hyphens only. Renaming a model renames the folder; source IDs are computed from the folder name and existing artifacts. Deleting the active model promotes the newest remaining model, or clears active state if none remain.
- GBM is the one normal chooser that still displays invalid dataset columns; they must render as disabled invalid rows and must not be sent to LightGBM.
- GBM training and model-output sources must use explicit readable-column projections. Avoid `SELECT *` on the raw dataset path because unreadable columns can fail even when they are not selected as model features.
- GBM model outputs publish data sources through the shared `data_sources` contract using IDs such as `gbm:<model_id>:predictions`, `gbm:<model_id>:shap_long`, and `gbm:<model_id>:shap_summary`; denominator-backed prediction and SHAP-long sources expose `gbm_prediction_rate`.
- The `gbm:<model_id>:shap_long` source ID is retained for compatibility, but the stored SHAP values artifact is wide: `__lucidum_row_id` plus one numeric SHAP column per selected feature. When selected feature interaction constraint groups exist, excluding main-effect-only and automatically generated uncovered-feature constraints, `shap_values.parquet` also includes grouped contribution columns named `<Grouping>_INTERACTION_GROUP`; these are row-wise sums of the grouped feature SHAP columns and are not included in `shap_summary.parquet`. Bounded SHAP row modes such as `10k` and `100k` use a deterministic random sample from all scored rows seeded by the model `seed` parameter, not the first rows. `gbm:<model_id>:shap_summary` remains one row per feature; persisted `shap_summary.parquet` stores `feature`, `mean_abs_shap`, `mean_shap`, and `row_count`, with model identity derived from the model folder and source ID rather than a repeated Parquet column.
- The Actual selector groups choices into Dataset features, Model predictions, and SHAP values. Model prediction choices include active GLM and GBM prediction sources, including denominator-backed prediction rates and `glm_tabulated_prediction` after GLM tabulation, and switch the active data source to that model output source when selected. SHAP choices remain scoped to the active GBM model.
- Line/Bar requests with SHAP ribbons include the browser schema's active GBM `model_id` inside `partialDependence`. The backend must resolve the SHAP and prediction sidecars from that requested model rather than re-reading global active-model state during query execution; this keeps stale-but-valid views deterministic when another browser page activates a different GBM. Requests without `model_id` retain the active-model fallback for API compatibility, while a supplied unavailable model must report an empty actionable SHAP warning rather than falling through to another active model.
- GBM SHAP plotting reads only saved SHAP sidecars and the original trained feature values joined by `__lucidum_row_id`; it must not import LightGBM, pandas, or numpy. SHAP config exposes only the active model's trained features with saved SHAP columns and includes loaded `Base` metadata for those features. One-feature plots use the selected feature's SHAP values; flame plots render returned numeric bands as ordered x-axis categories, show every band below the 200-category density limit, use Line/Bar's responsive 10px/8px and 65-degree label policy, add x-axis zoom above 120 bands, and suppress labels at 200 or more categories with an explanatory message. Flame axes preserve the first and last plotted bands and omit the old 45-55 ribbon; resize-only label recalculation uses the cached payload and must not issue a SHAP request. Two-feature plots use the sum of the two selected SHAP contributions. Continuous numeric axes use banding and optional tail grouping, return explicit numeric domains, omit missing numeric values with a warning, and factor-style axes include missing as `(missing)`. Numeric features forced to factor style keep natural band order, while true categorical box plots sort by descending median SHAP. Numeric/numeric surface payloads return dense backend grids for ECharts GL. Ordinary SHAP plot requests accept `rescale` values `-`, `0`, or `1`; `-` preserves raw behavior, `0` shifts by the relevant Base reference on the linear predictor scale, and `1` exponentiates values first before scaling to the base response-scale reference, with the frontend displaying those response-scale ratio values as uplift percentages (`1 = 0%`). Ordinary SHAP plots use one shared reference per plot. Its full-bleed screen uses a launch-collapsed shared settings strip, a launch-expanded 240-560px feature pane, and a launch-collapsed optional Feature 2 chooser. Both feature searches run full bleed from the pane edge to the workspace divider, use only top and bottom borders, place a borderless clear command inside the right edge, and meet their feature lists without a gap. The two factor overrides are independent borderless pressed buttons under `Treat as factor`. Toolbar, feature-pane, Feature 2, side-width, and chooser-split state are page-session frontend state only; two-feature context navigation expands the hidden choosers. Layout changes must use the coalesced resize path, synchronously flush settled ECharts layouts, and never issue SHAP requests. Stacked SHAP stays on the linear predictor contribution scale and does not accept a rescale control. Its settings-strip collapse state, feature-pane collapse state, and 240-560px feature-pane width are likewise page-session frontend state only. The SHAP frontend preserves matching legend visibility across active-model switches only when the selected features, plot type, and legend series still match.
- LightGBM-specific training, objective handling, offsets, SHAP, feature importance, tree extraction, and tree label normalization belong in backend GBM modules, not in frontend code.
- `py_lucidum.extract_lightgbm_interaction_group(source_model, group_features, output_model)` is a public, app-independent text-model transformation implemented in `tools/gbm/interaction_group_model.py` without importing LightGBM. It requires the requested names to exactly match one non-overlapping saved numeric `interaction_constraints` group, verifies at tree level that the group is isolated, retains complete multi-tree iterations only, and rewrites compact feature/split indexes, tree sizes, iteration counts, categorical metadata, per-feature parameters, and split/gain importance rows. By default it subtracts each retained tree's `leaf_count`-weighted expected value from its leaf/internal values so raw extracted-model predictions equal the source model's summed SHAP values for the group; `shap_centered=False` preserves the original leaf outputs. It must leave the destination untouched when validation fails; optional-LightGBM tests should load the result and predict from only the retained feature frame.
- GBM tree routes read persisted `tree_table.parquet` artifacts only and do not import LightGBM. The list route returns compact tree metadata and annotates each row with every explicit saved Singleton, Pairwise, or Group constraint that governs it; a pair or group applies when any saved member appears, persisted order is retained within the Singleton/Pairwise/Group type order, and automatically generated remainder/singleton constraints are excluded. The Tree viewer renders this as `Constraint applied:` or `Constraints applied:` below Tree gain and shows `None` when no explicit constraint applies. The detail route returns a frontend-ready split/leaf hierarchy with compact numeric thresholds, decoded categorical thresholds, edge labels, default-branch markers, cover percentages, and node values for colouring. Long categorical split display labels are summarized while full split labels remain available in tooltip fields, and frontend node clicks highlight the selected root-to-node path. Tree colour and direction are frontend-only session state: Divergent and left-to-right are the defaults, with alternative palettes plus top-to-bottom and top-left-to-bottom-right diagonal projections that redraw and refit the existing hierarchy without another API request.

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
- `--title-prefix`, `serve(..., title_prefix=...)`, and `create_app(..., title_prefix=...)` populate `/api/schema.title_prefix` and render before the file or folder name in the main app header. `lucidum --demo` defaults this prefix to `Lucidum Demo Dataset` unless an explicit value, including an empty value, is supplied.
- Header dataset metadata is fully inert: model-count, postcode Area/Sector/Unit, and Column Profile shortcuts are intentionally not rendered there.
- When the main header metadata overflows at high browser zoom or narrow viewport widths, the frontend hides the file name, size, rows, and columns and keeps only the bold title prefix visible. Launches without `title_prefix` keep the normal metadata ellipsis behavior.
- In notebook-style runtimes with an existing event loop, `serve()` and `run_app()` start the Uvicorn server in a background thread and return the URL.
- In a normal terminal or Python shell, server calls block until stopped.
- When enabled, the browser Stop app button calls `POST /api/shutdown`; health polling greys out the page after server shutdown. The monitor page remains available at `/monitor` with the normal token rules even when the main app header button is hidden.
- Detailed monitor operations are currently limited to GLM builds/tabulations and GBM training. The browser supplies one bounded `x-lucidum-operation-id` across validation, job creation, and polling; job progress closes and starts named phases in the in-memory telemetry store. Operation timing uses process CPU deltas over wall-clock intervals, so `average_cores` is diagnostic process activity rather than per-thread attribution. RSS values are observed at request and phase boundaries rather than continuously sampled.
- `/api/telemetry` exposes additive `environment` and `operations` objects. Retain only bounded in-memory operation history and an allowlist of numeric/non-sensitive progress metadata. Do not add request bodies, filters, tokens, model formulas, absolute paths, or persistent diagnostic files. Telemetry must stay fail-open and must not change modelling control flow.

## UI Direction

- Keep the app dense, utilitarian, and work-focused.
- Preserve chart space; controls should stay compact and avoid unnecessary wrapping.
- The sidebar is resizable so users can trade space between long column names and the chart.
- When multiple tools are enabled, the sidebar tool selector is a vertical rail that remains visible while the sidebar is collapsed. Clicking the active tool button toggles the sidebar open or closed; clicking an inactive tool switches tools without changing the sidebar state. Enabled GLM and GBM buttons show non-interactive blue count badges, including zero, while preserving normal tool-button behavior. Single-tool mode still hides the selector.
- Response controls sit above the x-axis feature list because response selection is usually the first workflow choice.
- Chart/Table controls sit before the filter bar.
- Line/Bar, Histogram, GBM SHAP, and GBM Stacked SHAP use the shared collapsible 50px settings strip above their full-bleed workspaces. Settings-strip buttons are borderless grey labels whose selected state is bold blue; each button reserves its bold label width so selection does not move the label or neighbouring controls. Group headers are quiet, muted captions aligned with the first option label, the header/button stack is optically balanced within the strip, and edge fades appear only while more controls can be reached by horizontal scrolling. Native scrollbars stay hidden so macOS overlay scrollbars cannot cover the controls; horizontal trackpad/touch gestures and keyboard focus retain native scrolling, while vertical mouse-wheel input scrolls the strip only when it can move and never intercepts zoom gestures or editable controls. The GLM and GBM Model navigator strips use the separate borderless command-button treatment: neutral commands gain a subtle accent wash only on interaction, while destructive commands use danger text and a danger wash rather than selectable or persistent active styling. Line/Bar keeps its compact action buttons and grey status text overlaid inside the chart/table workspace, and its x-axis/Expected chooser is a border-separated, resizable pane rather than a framed card. Both Line/Bar picker searches and their list rows run full bleed from the pane's left edge to its 1px workspace divider; the searches use only top and bottom borders, place a borderless clear command inside the right edge, and meet their lists without a gap. Histogram keeps a subtly bordered bins input and uses a resizable divider between its metrics table and chart. Stacked SHAP uses the same stable selection styling for its Model feature sort, keeps its feature chooser collapsible and resizable, and flushes settled ECharts layout changes before paint. At viewport widths up to 900px these resizable side panes stack above their workspaces.
- Saved-filter selections populate and apply the filter expression immediately. Sidebar FILTER mode and operator buttons use borderless muted labels with a bold blue selected state, reserving each bold label width so selection does not move the controls. Manual filter edits require Enter or Apply.
- Chart animations are disabled for fast interaction.
- The app should continue to work as a static ECharts and MapLibre frontend unless future tool complexity justifies a larger frontend framework.

**Frontend event hygiene**

- Tool-owned `document` and `window` listeners must either be removed when leaving the tool or be strictly gated by the tool's visible state and active interaction state.
- Global capture listeners must not clear focus, native selection, clipboard state, or editable input state outside their owning tool.
- Selection-clearing helpers must receive the original event when available and skip editable targets such as inputs, textareas, selects, and contenteditable nodes.

## Testing

Use these tiers to keep iteration focused while preserving the full suite.

- Change-aware checks are the normal between-commit loop. They always run
  syntax checks, inspect staged, unstaged, and untracked paths, then select the
  matching tool tests. Shared, unknown, browser-test, or clean-tree changes
  fall back to the broad lane; documentation-only changes stop after syntax.
  Frontend changes also report when a focused browser scenario is advisable:

```bash
.venv/bin/python scripts/run_tests.py changed
```

- Broad development checks are the explicit local safety net. Test modules are
  discovered dynamically, so new modules join this tier automatically. Browser
  tests and slow GLM fitting/tabulation coverage are excluded, while fast GLM
  validation/configuration contracts are included explicitly. Four Line/Bar
  process-isolation cases remain full-gate tests. The fast packaging contract in
  `test_pipx_install.py` runs while its environment-gated install test remains
  skipped. This tier currently takes about 15 seconds:

```bash
.venv/bin/python scripts/run_tests.py dev
```

- Focused unittest areas accept the test filename without its `test_` prefix;
  hyphens and underscores are interchangeable. Exact unittest module or method
  targets also pass through:

```bash
.venv/bin/python scripts/run_tests.py focus line-bar
.venv/bin/python scripts/run_tests.py focus glm gbm
.venv/bin/python scripts/run_tests.py focus tests.test_glm.GlmToolTests.test_glm_formula_drop_first_policy_tracks_regularization
```

- Syntax-only checks compile Python and discover every non-vendored JavaScript
  file dynamically before running `node --check`:

```bash
.venv/bin/python scripts/run_tests.py syntax
```

- Browser smoke tests continue to use the Dropbox-safe local mirror. Arguments
  after `--` are forwarded to pytest, so frontend work can target one scenario:

```bash
.venv/bin/python scripts/run_tests.py browser
.venv/bin/python scripts/run_tests.py browser -- --durations=20 -q
.venv/bin/python scripts/run_tests.py browser -- tests/test_browser_smoke.py::BrowserSmokeTests::test_gbm_tool_loads_feature_grid -q
```

Hosted CI runs the browser file once across four deterministic, balanced
shards. The full-unit job uses `prepush --skip-browser`; that option is for the
CI split and does not replace the normal local `prepush` gate. Browser jobs set
`PY_LUCIDUM_BROWSER_ARTIFACT_DIR`, which enables failure-only Playwright traces,
full-page screenshots, browser/network event logs, and JUnit output. Each failed
shard uploads those files as a 14-day Actions artifact. Do not retry a red run
to manufacture a releaseable result: use the first failure's artifact to fix
the readiness or rendering contract, and require a clean run on the exact
commit.

Prefer future behavior tests over exact JS/CSS string-contract tests where
practical. Exact asset-string checks are still acceptable for stable contracts
such as asset registration, cache-control behavior, and intentionally documented
UI text or selectors.

- The fast pre-commit gate checks unstaged and staged whitespace, performs
  dynamic Python and JavaScript syntax checks, and runs the existing
  change-aware unit lane. It normally takes 2–15 seconds:

```bash
.venv/bin/python scripts/run_tests.py precommit
```

- The complete pre-push gate repeats whitespace and syntax checks, runs full
  unittest discovery including GLM coverage, then runs all browser smoke tests
  sequentially:

```bash
.venv/bin/python scripts/run_tests.py prepush
```

Enable the versioned hooks once per clone so normal `git commit` and `git push`
calls cannot skip their respective gates accidentally:

```bash
git config core.hooksPath .githooks
```

Both hooks use `.venv/bin/python` by default. Set
`PY_LUCIDUM_TEST_PYTHON=/absolute/path/to/python` when the test environment
lives elsewhere. Do not manually run `prepush` immediately before a normal
`git push`; the hook will run it. Git's `--no-verify` remains an explicit
emergency bypass and is not the normal development workflow. Version bumps
remain a separate step performed before committing.

The isolated pipx installation test is environment-sensitive and stays outside
the normal commit gate. Run it for packaging or release changes:

```bash
.venv/bin/python scripts/run_tests.py pipx
```

Use `PY_LUCIDUM_PIPX_PYTHON=python3.13` when the default pipx interpreter is
not Python 3.13.

Current timings recorded on macOS arm64 with Python 3.13.14 and Node 26.5.0:

- Static-frontend changed lane equivalent: 23 tests plus syntax in about 2.1 seconds.
- GBM changed lane equivalent: 126 tests plus syntax in about 6 seconds.
- Broad development lane: 512 tests, one expected skip, and syntax in about 16 seconds.
- Full unittest discovery: 642 tests, 64 expected skips, in about 87 seconds.
- Browser smoke: 63 tests in about 2 minutes 14 seconds.
- Complete pre-push gate: about 3 minutes 42 seconds.

Browser smoke coverage should include cross-tool focus and listener regressions
after visiting tools that install global listeners, especially document/window
capture handlers.

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
- GBM validation, sidecar model store behavior, optional dependency failures, native runtime dependency failures, live job progress, active-model feature/parameter refresh, model data-source publishing, interaction-group text-model extraction and metadata compaction, Gain ordering, SHAP row limits, SHAP plot aggregation routes, tree summary/detail routes, and chart/map use of prediction sources.
- Browser smoke behavior for loading profile, chart, histogram, map, and GBM tools without unexpected extra API requests, stale active-model state, or leaked cross-tool focus/listener side effects, including live GBM progress, the GBM tree viewer, and the GBM SHAP screen.

## Releasing

Lucidum releases use annotated Git tags, immutable GitHub releases, and GitHub
Actions Trusted Publishing. `.github/workflows/ci.yml` runs the complete test
gate for `main` and pull requests on Python 3.13. Ubuntu runs the non-browser
gate plus the sharded Chromium and WebKit browser suite; Windows repeats the
pipx/package and non-browser gates plus the complete browser suite. A successful
hosted `CI` result therefore covers both operating systems.
`.github/workflows/release.yml` accepts only an annotated
`vMAJOR.MINOR.PATCH` tag that matches `pyproject.toml`, points to `main`, and
has a successful hosted `CI` run. It builds one wheel/sdist pair and promotes
those exact bytes through TestPyPI, PyPI, and the GitHub release.

### One-time repository and index setup

1. Create separate PyPI and TestPyPI accounts, enable the required two-factor
   authentication, and store recovery codes safely. The two indexes do not
   share accounts or trusted-publisher configuration.
2. Create GitHub environments named `testpypi` and `pypi`. Restrict both to tag
   deployments matching `v*`. Do not require approval for `testpypi`; require
   a maintainer approval for `pypi`. A sole maintainer must leave prevention of
   self-review disabled until another reviewer is available.
3. Register a pending GitHub Actions publisher on each index with these exact
   values:
   - PyPI project: `py-lucidum`
   - GitHub owner: `SpeckledJim2`
   - GitHub repository: `py_lucidum`
   - Workflow filename: `release.yml`
   - Environment: `testpypi` on TestPyPI and `pypi` on PyPI
4. In the GitHub repository's general settings, enable release immutability.
   This applies when the draft release is published, after all artifacts have
   been attached and verified.

A pending publisher can create the project on first upload but does not reserve
the project name. Complete the first release promptly after setup. The release
workflow needs no stored PyPI token: only its publishing jobs receive
`id-token: write`, and all other job permissions stay minimal.

### Routine release checklist

1. Start from a clean `main` synchronized with `origin/main`. Confirm the target
   version does not already have a Git tag or PyPI release.
2. Put all preparation for a specific release in one commit. Run the normal
   patch/minor/major bump exactly once before committing, and do not hard-code a
   different version in release assets or workflows.
3. Run the packaging-specific local checks in addition to the normal commit
   gate. `build` and `twine` may be installed in the development environment:

```bash
.venv/bin/python scripts/run_tests.py pipx
release_dist=$(mktemp -d)
.venv/bin/python -m build --outdir "$release_dist"
.venv/bin/python -m twine check "$release_dist"/*
.venv/bin/python scripts/release_artifacts.py inspect \
  --dist-dir "$release_dist" --version X.Y.Z
```

4. Commit the release preparation. A normal push runs the local pre-push hook;
   wait for the hosted `CI` workflow to pass on that exact `main` commit.
5. Create and push the annotated tag without moving or replacing an existing
   tag:

```bash
git tag -a vX.Y.Z -m "py-lucidum X.Y.Z"
git push origin vX.Y.Z
```

6. The release workflow validates the tag and CI result, builds and pipx-smokes
   the distributions, writes `SHA256SUMS`, creates a draft GitHub release, then
   publishes and hash-verifies the artifacts on TestPyPI. The wheel and sdist
   use a package-only Actions artifact for index publishing; `SHA256SUMS` uses
   a separate artifact and is combined with them only for the GitHub release.
7. While the `pypi` environment waits for approval, inspect TestPyPI and edit
   the draft GitHub release notes. The notes should summarize user-visible
   changes, Python compatibility, optional extras, and installation commands.
8. Approve `pypi` only when the TestPyPI files and draft are correct. The
   workflow publishes the same artifacts to PyPI, verifies their hashes,
   installs the exact release through pipx, exercises the launcher health
   check, and finally publishes the immutable GitHub release.
9. Verify the public result and update consuming projects:

```bash
python3.13 -m pip install "py-lucidum==X.Y.Z"
lucidum --version
gh release view vX.Y.Z
```

An exact `py-lucidum` pin does not lock its transitive dependencies. Apply a
lock or constraints file in each consuming project when the complete resolved
environment must be reproducible.

### Failures and recovery

- For a transient workflow failure, rerun only failed jobs so successful index
  uploads are not attempted again. Do not rerun the entire workflow after an
  index has accepted an artifact.
- Before production publication, reject the `pypi` deployment when validation
  is doubtful. Fix the problem in a new commit, bump again, and use a new tag;
  never repoint the pushed tag.
- After PyPI accepts a version, its filenames cannot be reused even if files or
  the project are later deleted. Yank a broken release with a clear reason and
  publish the fix at a higher patch version.
- A published immutable GitHub release locks its tag and attached assets. Do
  not delete, replace, or work around that history; publish a new release.
- Because every repository commit increments `project.version`, not every
  version needs a tag. Gaps between published versions are expected and safe.

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
  - The sidebar version text must always reflect the same package version: `/api/schema` should publish `app_version` from `pyproject.toml`, and the frontend should display it as `lucidum v<version>` while expanded and `v<version>` while collapsed without maintaining separate hard-coded sidebar versions.
  - Always report the final package version number back to the user when handing off commit-ready work.
  - Optional local aliases:
    - `git config alias.cpatch '!python scripts/bump_version.py patch && git add pyproject.toml && git commit'`
    - `git config alias.cminor '!python scripts/bump_version.py minor && git add pyproject.toml && git commit'`
- Before committing:
  - Check `git status --short` and make sure new files, deletions, and generated artifacts are intentional.
  - Update `README.md` if the change affects public setup, launch commands, user workflows, CLI options, Python usage, demo data, or visible behavior.
  - Update this file if the change affects architecture, behavior contracts, testing policy, packaging, data handling, or tool-extension guidance.
  - Use the smallest relevant focused checks while editing, then rely on the fast pre-commit hook and complete pre-push hook; run a focused browser scenario before committing frontend interaction changes.
  - Scan staged changes for secrets, real customer data, local-only paths, and stale references to removed files or old demo datasets.
- Update `README.md` for public user-facing behavior changes.
- Update this file when architecture, behavior contracts, testing policy, packaging, or tool-extension guidance changes.
- Keep generated caches, local datasets other than the synthetic demo, `.lucidum/` model artifacts, virtual environments, build artifacts, and OS metadata out of git.
- Do not commit real customer data. The bundled motor premiums dataset is synthetic.
