# lucidum

`lucidum` is a local browser workbench for exploring CSV and Parquet datasets. It starts a small local FastAPI server, reads data with DuckDB, and opens an interactive UI for profiling columns, plotting grouped metrics, applying filters, and mapping UK postcode data.

The app is designed for local analysis: your dataset stays on the machine running `lucidum`.

## Current Tools

- **Dataset Viewer**: inspect a fast filtered preview of dataset rows in a sortable Tabulator grid, capped at 1,000 displayed rows, with client-side whole-table search, persistent multi-row highlighting, transpose mode, optional alphabetical column ordering, reset-sort, and copy-selected rows as CSV.
- **Column Profile**: review dataset columns, missing values, distinct counts, ranges, value counts, and numeric/date distributions. Large datasets open with a fast preview summary and can be recalculated on all rows. Right-click a column row to copy the feature name.
- **Line and Bar**: plot grouped Actual and optional Expected response values over any feature, with shared Weight, lazily estimated numeric banding, date buckets, server-backed searchable/paginated tables, Base-aware transforms, sigma bars, optional active-GBM SHAP ribbons, active-GLM overlay lines, and x-axis feature ordering by saved GBM/GLM feature importance.
- **Histogram**: plot the selected Actual value, or Actual divided by Weight, as a filtered distribution with configurable bins, cumulative/probability modes, log axes, mean/median reference lines, and a compact metrics table.
- **UK Mapping**: map postcode areas and sectors with bundled GeoJSON, including optional sector neighbour smoothing, or postcode units when unit and coordinate columns are available.
- **GLM**: optional `glum` model building with Formulaic formulas, coefficient tables, persisted tabulations/rating tables, and active `glm_prediction`, denominator-backed `glm_prediction_rate`, and `glm_tabulated_prediction` sources that can be plotted like other model predictions.
- **GBM**: optional LightGBM model building with persistent sidecar artifacts, predictions and denominator-backed prediction rates that can be plotted as chart/map data sources, evaluation plots, model navigation, tree viewing, and SHAP plotting when SHAP rows are saved during training.
- **Filters, KPIs, and Feature specs**: apply free-form DuckDB `WHERE` filters, saved filter rows, KPI specs that set Actual/Weight choices and formatting, and GBM feature scenarios/interaction constraints.
- **Specifications**: default editor tab for feature, KPI, and filter specification CSV files, with continuous validation and save actions against the app's current metadata contracts.

Unreadable dataset columns, such as Parquet strings with invalid UTF-8, are skipped by the shared schema used by normal selectors. Column Profile reports them as skipped, and the GBM feature chooser shows them as disabled invalid rows.

## Installation

From the project root:

```bash
python3.13 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -e .
```

This installs the `lucidum` command inside the virtual environment.

To enable GLM or GBM model training, install the relevant optional modelling extra:

```bash
.venv/bin/python -m pip install -e ".[glm]"
.venv/bin/python -m pip install -e ".[gbm]"
```

On macOS, LightGBM also needs the OpenMP runtime. This applies whether you
install with a virtual environment or `pipx`. If training fails with a
`libomp.dylib` load error, install it with:

```bash
brew install libomp
```

For a user-level command, install directly from GitHub with `pipx`. Choose the
modelling extras you need at install time:

```bash
pipx install --python python3.13 git+https://github.com/SpeckledJim2/py_lucidum.git
pipx install --python python3.13 "py-lucidum[glm] @ git+https://github.com/SpeckledJim2/py_lucidum.git"
pipx install --python python3.13 "py-lucidum[gbm] @ git+https://github.com/SpeckledJim2/py_lucidum.git"
pipx install --python python3.13 "py-lucidum[glm,gbm] @ git+https://github.com/SpeckledJim2/py_lucidum.git"
```

Quote pipx package specs with extras exactly as shown, because shells treat
spaces as separators and can interpret `[glm,gbm]` as a filename pattern.

When installing from a local checkout instead of GitHub, use the same extra
names with the local path:

```bash
pipx install --python python3.13 /path/to/py_lucidum
pipx install --python python3.13 "/path/to/py_lucidum[glm,gbm]"
```

If you already installed Lucidum with `pipx` without modelling extras, reinstall
it with the extra spec you want:

```bash
pipx uninstall py-lucidum
pipx install --python python3.13 "py-lucidum[glm,gbm] @ git+https://github.com/SpeckledJim2/py_lucidum.git"
```

## Quick Start

Launch the bundled synthetic demo dataset:

```bash
.venv/bin/lucidum --demo --port 8000
```

Open the printed URL in your browser. Stop the server with `Ctrl+C` in the terminal or the `Stop app` button in the browser header.

Run your own data:

```bash
.venv/bin/lucidum path/to/my_data.parquet --port 8000
.venv/bin/lucidum path/to/my_data.csv --port 8000
```

If installed with `pipx`, use:

```bash
lucidum path/to/my_data.parquet --open
lucidum path/to/my_data.csv --open
lucidum path/to/my_data.parquet --tools line-bar,glm --open
lucidum path/to/my_data.parquet --tools line-bar,glm,gbm --open
```

Use `--tools all` after installing the needed modelling extras when you want to
load every available tool, including GLM and GBM.

Parquet is recommended for normal use because DuckDB can read it efficiently.

## Dataset Workspaces

Lucidum stores model sidecars beside the data folder under a hidden dataset-scoped workspace:

```text
.lucidum/datasets/<dataset-slug>/<dataset-signature>/models/
```

The dataset slug comes from the CSV or Parquet filename. The dataset signature is based on the file size, modification time, row count, and schema fingerprint. This means one folder can safely contain many datasets: models trained for one file do not attach to another file in the same folder.

If a dataset file is replaced or edited, it gets a new signature workspace. Existing GLM/GBM models from the previous version remain on disk but are not shown or used; rebuild models after changing the source file. Older root-level `.lucidum/models/` folders are ignored by current Lucidum versions.

## Common Options

```bash
.venv/bin/lucidum --demo --open --port 8000
.venv/bin/lucidum --demo --host 0.0.0.0 --port 8000
.venv/bin/lucidum --demo --no-token
.venv/bin/lucidum --demo --x DRIVER_AGE --actual PREMIUM --denominator ANNUAL_MILEAGE
.venv/bin/lucidum --demo --filters specs/filter_spec.csv
.venv/bin/lucidum --demo --no-filters
.venv/bin/lucidum --demo --kpis specs/kpi_spec.csv
.venv/bin/lucidum --demo --no-kpis
.venv/bin/lucidum --demo --features specs/feature_spec.csv
.venv/bin/lucidum --demo --no-features
.venv/bin/lucidum --demo --tools line-bar
.venv/bin/lucidum path/to/my_data.parquet --tools line-bar,uk-map,glm,gbm
.venv/bin/lucidum path/to/my_data.parquet --tools all
```

- `--open` opens the generated URL with Python's configured browser handler.
- `--host 0.0.0.0` binds to all network interfaces for LAN testing. Keep token protection enabled unless another access layer is in place.
- `--no-token` disables URL/API token protection for local-only use.
- `--x`, `--actual`, `--expected`, and `--denominator` set initial Line/Bar selections.
- `--filters` points to a saved-filter CSV. By default the app tries `./filter_spec.csv`, then `./specs/filter_spec.csv`.
- `--kpis` points to a KPI spec CSV. By default the app tries `./kpi_spec.csv`, then `./specs/kpi_spec.csv`.
- `--features` points to a Feature Specification CSV for GBM feature scenarios, interaction constraints, optional Base metadata, and GLM tabulation `min/max/banding` metadata. By default the app tries `./feature_spec.csv`, then `./specs/feature_spec.csv`.
- `--no-filters`, `--no-kpis`, and `--no-features` disable discovery for those spec files. When the Specifications tab is enabled, disabled or missing spec kinds open as generated starter drafts instead of preloading default-discovered CSVs. Generated drafts are not written to disk until you click Save; the path line shows the save target and marks new files or suppressed existing files.
- Without `--tools`, the default user-facing tools are `dataset-viewer`, `column-profile`, `line-bar`, `histogram`, `uk-map`, and `specs`, with Dataset Viewer opening first. Use `--tools all` to load every tool, including GLM and GBM. When `--tools` is provided with a comma-separated list, only those app tabs are loaded. Add `glm` after installing the `glm` extra to train GLMs, and add `gbm` after installing the `gbm` extra to train GBMs; either modelling tool must be requested with `line-bar` because model context-menu actions open Line/Bar charts.

UK map columns default to `PostcodeArea`, `PostcodeSector`, `PostcodeUnit`, `lat`, and `long`. Uppercase aliases such as `POSTCODE_AREA`, `POSTCODE_UNIT`, `LATITUDE`, and `LONGITUDE` are also detected. You can override them:

```bash
.venv/bin/lucidum path/to/my_data.parquet \
  --postcode-area Area \
  --postcode-sector Sector \
  --postcode-unit Unit \
  --latitude latitude \
  --longitude longitude
```

Core browser libraries are bundled and served by the local app. UK map geometry
is also bundled. The `Blank` map background makes no external tile requests;
other map backgrounds, such as OSM, Esri, Aerial, Light, and Dark, fetch map
tiles from their listed third-party providers and need network access.

## License

Lucidum is open source under the MIT License. See `LICENSE`.

## Python Usage

```python
import py_lucidum

py_lucidum.serve(py_lucidum.demo_dataset_path(), port=8000, open_browser=True)
py_lucidum.serve("path/to/my_data.parquet", port=8000, open_browser=True)
```

In notebook-style runtimes such as Positron or Jupyter, `serve()` starts the server in the background and returns the URL immediately. In a normal Python shell, it blocks until stopped.

For ASGI usage:

```python
import py_lucidum
from py_lucidum.app import create_app

app = create_app(
    py_lucidum.demo_dataset_path(),
    token="dev-token",
    defaults={
        "x": "DRIVER_AGE",
        "actual": "PREMIUM",
        "denominator": "ANNUAL_MILEAGE",
    },
    filters_path="specs/filter_spec.csv",
    kpis_path="specs/kpi_spec.csv",
    features_path="specs/feature_spec.csv",
    tools=["dataset_viewer", "column_profile", "line_bar", "histogram", "uk_map", "specs"],
)

py_lucidum.run_app(app, host="127.0.0.1", port=8000, open_browser=True)
```

## Filters

The footer filter box accepts DuckDB `WHERE` expressions:

```sql
DRIVER_AGE > 40
ANNUAL_MILEAGE >= 20000
VEHICLE_USAGE = 'Social only'
QUOTE_DATE >= DATE '2017-01-01'
```

Saved filters are CSV files with exactly these columns:

```csv
theme,name,expression
SAMPLE,Training,SAMPLE = 'training'
SAMPLE,Test,SAMPLE = 'test'
SAMPLE,Validation,SAMPLE = 'validation'
DRIVER AGE,Young drivers,DRIVER_AGE < 30
DRIVER AGE,Older drivers,DRIVER_AGE > 70
```

Saved-filter rows can be used in `Single`, `Multi`, or `Grouped` mode. The generated expression is written to the footer expression box and applies to the active tool.

When the Specifications tool opens a missing filter spec, it starts with one blank row and visual placeholder hints for `theme`, `name`, and `expression`; those hints are not saved as cell values.

## KPIs

KPI specs are CSV files with exactly these columns:

```csv
group,name,actual,denominator,decimals,format
VEHICLE,Vehicle age,VEHICLE_AGE,N,1,number
DRIVER,Driver age,DRIVER_AGE,N,1,number
FINANCIAL,Premium,PREMIUM,N,2,currency
```

`denominator` accepts `N`, `Average row value`, an empty value, or `__none__` for average row value, or any numeric column name for weighted response values. `format` accepts `number`, `currency`, or `percent`.

When the Specifications tool opens a missing KPI spec, it starts with one blank row and visual placeholder hints for each field; those hints are not saved as cell values.

## Feature Specs

Feature Specification CSV files drive GBM feature scenarios, interaction-constraint groups, optional chart Base metadata, and GLM numeric tabulation metadata. The current format starts with these columns, followed by any number of scenario columns:

```csv
Feature,Grouping,Base,min,max,banding,scenario1,scenario2,scenario3
DRIVER_AGE,DRIVER,40,17,96,1,feature,feature,feature
NCD_YEARS,DRIVER,10,0,20,1,feature,,feature
POSTCODE_AREA,POSTCODE,B,,,,,feature,feature
```

`Feature` must match a dataset column name exactly. `Grouping` is optional metadata shown in the GBM Feature table and, when present, is also used to offer GBM feature interaction constraints. `Base` is optional metadata used to anchor Line/Bar and GBM SHAP chart rescaling to `0` or `1` and to define GLM tabulation base cells; `1` rescaling is displayed as an uplift percentage, so the base level shows as `0%`. Numeric `min`, `max`, and `banding` define GLM rating-table grids; leave them blank for categorical features. Older specs without these metadata columns are still accepted, in which case every column after `Grouping` is treated as a scenario. Each scenario column appears in the GBM scenario dropdown; if a scenario cell contains the word `feature`, case-insensitive, that row is selected when the scenario is chosen.

When the Specifications tool opens a missing Feature spec, it starts with one row per valid dataset column, populates only `Feature`, and leaves `Grouping`, `Base`, `min`, `max`, `banding`, and `scenario1` blank.

## GLM Models

The GLM tool is opt-in and must be requested with `line-bar`:

```bash
.venv/bin/lucidum path/to/my_data.parquet --tools line-bar,uk-map,glm
```

The Formula builder accepts either a full `response ~ terms` Formulaic formula, or RHS-only `terms` that use the sidebar Actual metric as the response. Lines can include `#` comments; raw formula text is stored in `formula.txt` with comments, but comments are stripped before fitting. The formula context includes `ifelse`, `pmin`, `pmax`, `ns`, `bs`, `cs`, `poly`, `C`, and common numeric transforms. Explicit `offset(...)` terms are supported; they are stripped from the fitted formula, stored in the manifest, and passed to `glum.fit()` and prediction.

Families are `normal`, `poisson`, `gamma`, `tweedie`, `binomial`, `inverse.gaussian`, and `negative.binomial`, with `link="auto"` in the first implementation. Tweedie power and negative-binomial theta can be set from the family parameter input. If a sidebar Weight is selected, GLM fits `Actual / Weight` with `sample_weight=Weight`, stores `glm_prediction` back on the original Actual scale, and exposes `glm_prediction_rate = glm_prediction / Weight`. Saved models live under the current dataset workspace in `models/glm/`, and the active model publishes a `glm:<model_id>:predictions` data source.

The `Tabulations` tab builds insurance-style rating tables on demand from selected saved GLMs. It uses the fitted `glum` estimator, formula terms, feature spec `Base/min/max/banding`, and any stored `offset(...)` expressions to create base-adjusted linear-predictor tables, table/plot payloads, and row-level `glm_tabulated_prediction`. Single-model GLM non-base tables can be rebased from a selected cell into either a compatible one-way table row or the `base` table; the app preserves raw tables, rewrites adjusted tables, and recalculates `glm_tabulated_prediction`. Numeric feature spec metadata that is missing or blank is estimated from scored rows and reported in the GLM notice. Existing GLMs built before `estimator.pkl` persistence must be rebuilt before tabulation.

The Penalty selector defaults to `None` for the existing unregularized fit. `Auto` uses glum cross-validation over ridge, elastic net, and lasso mixes; `Manual` exposes a compact mix and alpha control. Penalized models show coefficient estimates but suppress coefficient standard errors and p-values because regularized inference is not equivalent to the unpenalized GLM table.

`All` fits all valid rows. `Training` fits only rows where a physical `SAMPLE` column equals `training`, case-insensitively; GLM does not create generated sample splits.

GLM model sidecars keep `manifest.json` compact: identity, response/denominator, family/link, regularization, training scope, offset expressions, minimal formula execution flags, and total elapsed time. Raw formula text lives only in `formula.txt`; diagnostics and warnings live in `diagnostics.json`; coefficients, feature importances, predictions, and tabulated predictions live in Parquet artifacts. `estimator.pkl` is required for GLM tabulations and overlay reconstruction. Tabulation metadata lives beside the rating tables in `tabulations/manifest.json`. When LightGBM/glum load-order protection is required, GLM fitting stays isolated in a hot worker that is reused after the first build; set `PY_LUCIDUM_GLM_FIT_ONE_SHOT=1` to force the old one-shot worker path for debugging.

## GBM Models

The GBM tool is opt-in and must be requested with `line-bar`. Request `glm` separately when you also want GLM comparison and tabulation workflows:

```bash
.venv/bin/lucidum path/to/my_data.parquet --tools line-bar,uk-map,glm,gbm
.venv/bin/lucidum path/to/my_data.parquet --tools line-bar,gbm --features specs/feature_spec.csv
```

### Feature setup

The GBM tool uses the same sidebar Actual and Weight/KPI controls as Line and Bar, so users can choose the modelling response before training.

If a Feature Specification is loaded, the Feature table shows its `Grouping` values, a multi-select interaction-constraint dropdown, and a scenario dropdown next to `Clear all`; choosing a scenario selects only that scenario's usable features.

Choosing one or more interaction groups constrains the currently selected trainable features in each group so they can only interact with features in the same group, with all other selected features left together in a remainder constraint.

The `Interaction pairs` dropdown lets you add a short allowlist of feature pairs, such as `DRIVER_AGE x VEHICLE_AGE`; adding a pair also selects those two features for training. Pair constraints can be combined with Feature Specification grouping constraints only when the selected groups are disjoint from every paired feature; overlapping groups are rejected because they would reopen interactions beyond the pair allowlist. Singleton feature locks are allowed only for features outside the pair list. Main effects remain available for all selected features, and unlisted pairs cannot appear together on the same LightGBM branch. Pair allowlists are intended for 3-leaf GBM/EBM workflows, though they can be used with normal GBMs too.

When a saved model has both Gain and saved SHAP rows, the Feature table shows a Gain/SHAP toggle and displays one importance metric at a time; SHAP means mean absolute SHAP value over the saved SHAP rows. For saved EBM models, the same toggle also offers `EBM Gain`, replacing the Feature table with a tree-feature-combination gain summary built from the saved tree table.

When an active GBM has both saved predictions and saved SHAP rows, Line and Bar can show `Partial dependancies > SHAP` ribbons for the selected x-axis feature. These ribbons use the same grouping, banding, filter, denominator, low-weight grouping, and response transform as the chart. The ribbon median is scaled to the active GBM fitted prediction mean for the current chart slice, using the selected Weight when present; when `Both` is selected, the GLM line is aligned to the same fitted-mean baseline for direct comparison. Categorical x-axes also offer a SHAP sort ordered by median SHAP.

When an active GLM is available, Line and Bar can show a dashed GLM overlay line. Main-effect features use a fast base-profile grid over the rendered x-axis groups. Simple two-way interactions use a collapsed partner grid, and complex interactions fall back to the deterministic sampled marginal PDP. When an isolated worker is needed, the first GLM overlay starts a persistent hot worker that is reused for later overlay requests.

### Training and artifacts

Parameter values can use grid-search braces, such as `{200, 300, 400}`, `{0.05, 0.3; 0.05}`, or `{bagging, goss}` for `data_sample_strategy`; the app samples the hypergrid deterministically, trains one model per valid sampled combination, skips invalid combinations with a notice, and activates the best completed model.

The first parameter row, `init_score`, can stay as `none` for the current behavior or point at a numeric dataset column or fitted GLM prediction artifact. Supplied values are treated as prediction-space baselines, transformed to LightGBM's linear predictor space for objectives with log or logit links, and replace the automatic denominator-derived initial score. When used, the resolved baseline is saved as `init_score.parquet` beside the model.

Saved GBM `parameters.json` is a LightGBM Python params dictionary intended for `json.load()` then `lgb.train(params=...)`, including objective and metric. Lucidum-only selections such as the UI `init_score` choice, init-score provenance, and EBM training mode live in `manifest.json`.

GBM model input feature order lives in `features.json`, as a JSON array of feature names. Trained GBM display metadata lives in optional `feature_config.parquet`, enriched with kind, monotonicity, Gain, and optional SHAP importance. Prediction and SHAP data sources derive their raw dataset-column projections from the current dataset schema rather than from duplicated manifest fields.

The `10k` and `100k` SHAP row options save a deterministic random sample from all scored rows using the model seed; `All` saves every scored row.

Saved `shap_summary.parquet` contains one row per trained feature with `feature`, `mean_abs_shap`, `mean_shap`, and `row_count`. The model identity comes from the model folder and computed source ID, not from a repeated column inside the Parquet file.

When feature interaction constraint groups are selected during training, saved `shap_values.parquet` files also include one grouped SHAP contribution column per selected grouping, named like `POSTCODE_INTERACTION_GROUP`. These grouped columns and the active model's prediction column are available in the Line and Bar Actual chooser under separate Dataset, Model predictions, and SHAP values sections.

If the source dataset has a `SAMPLE` column, GBM trains on `training`, early-stops on `test`, and scores `validation` as a holdout. If `SAMPLE` is missing, the tool can create one reusable generated 60/20/20 sidecar split under the current dataset workspace in `models/gbm/`; for durable modelling, add a proper `SAMPLE` column to the original Parquet file. Models are saved under the same dataset-version workspace.

During training, the app shows live iteration and train/test metric progress and updates the evaluation plot while the background job runs; grid-search progress includes the current model number. The Evaluation Log keeps its live x-axis fixed to the configured iteration count, then uses the exact completed tree count and a tail-focused y-axis view so later training progress remains readable after a steep initial drop. Use the inline `All` / `Tail` control to switch between the full history and a focused tail view. Long evaluation histories are sampled only for browser rendering; saved `evaluation.parquet` artifacts remain complete.

### Saved feature context

When a GBM is trained directly from a feature scenario, the model records that scenario name and the training-time feature list. Selecting a saved GBM shows the recorded scenario in the dropdown. If `feature_spec.csv` has changed since training, the dropdown marks the recorded scenario as changed or missing while the Feature table still reflects the model's saved feature configuration.

When a GBM is trained with feature interaction constraints, the model records the constrained group names and training-time feature lists. Selecting a saved GBM shows those constraints in the interaction dropdown, shows a lock beside constrained Feature table groupings, and marks stale or missing groupings if `feature_spec.csv` has changed. The Model navigator includes a `Constraints` column, and the saved `tree_table.parquet` can be used to inspect which split features appear along each tree path.

When a GBM is trained with interaction pairs, the model records the allowed pairs and the Model navigator shows `Pairs (n)`. In an active EBM model's `EBM Gain` view, right-click a two-feature row and choose `Allow interaction pair` to seed the pair allowlist before retraining.

### EBM mode

When the active sample source has both `training` and `test` rows, either from a physical dataset `SAMPLE` column or the generated sidecar split, the tool also shows an EBM mode. EBM starts with 2-leaf trees, uses learning rate `0.3` for that 2-leaf stage, then moves through 3, 4, and higher leaf counts up to `num_leaves` whenever the test metric has not improved for `early_stopping_rounds`. `num_iterations` remains the total cap across all EBM stages.

### Saved models

Choosing a saved GBM in the sidebar, or selecting one model in the Model navigator and clicking Activate, makes it active and refreshes the feature table, parameter table, training mode, evaluation plot, tree viewer, SHAP screen, and plot-ready model outputs.

The Model navigator shows the active model with a green dot and lets users select one or more rows for model-folder actions: rename or activate one selected model folder, or delete all selected model folders from the current dataset workspace. Deleting the active model selects the newest remaining model when one exists.

The tree viewer reads saved `tree_table.parquet` artifacts, shows a searchable tree list, and renders the selected tree with zoom and colour controls. Tree split labels use compact numeric formatting, tightly wrap categorical splits on the diagram, and summarize long categorical splits while keeping the full split in tooltips. Node cover shows count and percentage of the selected tree, and clicking a node highlights its path from the root.

### SHAP plots

The SHAP tab is available for GBMs trained with a nonzero SHAP row option. It lists only the active model's trained features, defaults Feature 1 to the highest-Gain feature, defaults Feature 2 to `None`, and can sort both feature choosers by Gain or name.

Switching active models preserves the selected SHAP features when they exist in the next model, and preserves matching plot legend visibility when the same SHAP plot can still be shown; unavailable Feature 2 selections fall back to `None`, and unavailable Feature 1 selections fall back to the first item in the current Feature 1 sort order.

One-feature plots show numeric percentile ribbons around the median or factor box plots, with flame plot x-axes fitted to the plotted data range; when a numeric feature is treated as a factor, its bands keep their natural numeric order. Two-feature plots show a dense-grid 3D surface for two continuous features, line plots for continuous-by-factor selections, and heatmaps for two factor-style selections. Two-feature plots use the sum of the two selected SHAP contributions. The SHAP tab includes a `-` / `0` / `1` rescale control that uses Feature Specification `Base` metadata when present. SHAP `1` rescaling exponentiates values first, then scales to the base response value and displays the result as uplift percentage. Stacked SHAP stays on the linear predictor contribution scale.

Once a model is active, its predictions and SHAP outputs also appear as selectable data sources so Line/Bar and UK Mapping can plot model outputs like normal columns.

## Development

Maintainer notes, architecture details, timing semantics, and test commands live in `DEVELOPMENT.md`.
