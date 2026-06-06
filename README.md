# lucidum

`lucidum` is a local browser workbench for exploring CSV and Parquet datasets. It starts a small local FastAPI server, reads data with DuckDB, and opens an interactive UI for profiling columns, plotting grouped metrics, applying filters, and mapping UK postcode data.

The app is designed for local analysis: your dataset stays on the machine running `lucidum`.

## Current Tools

- **Column Profile**: review dataset columns, missing values, distinct counts, ranges, value counts, and numeric/date distributions. Large datasets open with a fast preview summary and can be recalculated on all rows. Right-click a column row to copy the feature name.
- **Line and Bar**: plot grouped Actual and optional Expected response values over any feature, with shared Weight, lazily estimated numeric banding, date buckets, tables, Base-aware transforms, sigma bars, optional active-GBM SHAP ribbons, and active-GLM overlay lines.
- **UK Mapping**: map postcode areas and sectors with bundled GeoJSON, including optional sector neighbour smoothing, or postcode units when unit and coordinate columns are available.
- **GLM**: optional `glum` model building with Formulaic formulas, coefficient tables, persisted tabulations/rating tables, and active `glm_prediction` / `glm_tabulated_prediction` sources that can be plotted like other model predictions.
- **GBM**: optional LightGBM model building with persistent sidecar artifacts, predictions that can be plotted as chart/map data sources, evaluation plots, model navigation, tree viewing, and SHAP plotting when SHAP rows are saved during training.
- **Filters, KPIs, and Feature specs**: apply free-form DuckDB `WHERE` filters, saved filter rows, KPI specs that set Actual/Weight choices and formatting, and GBM feature scenarios/interaction constraints.

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

On macOS, LightGBM also needs the OpenMP runtime. If training fails with a
`libomp.dylib` load error, install it with:

```bash
brew install libomp
```

For a user-level command available outside this checkout, install with `pipx`:

```bash
pipx install --python python3.13 /path/to/py_lucidum
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
```

Parquet is recommended for normal use because DuckDB can read it efficiently.

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
.venv/bin/lucidum path/to/my_data.parquet --tools line-bar,uk-map,models
```

- `--open` opens the generated URL with Python's configured browser handler.
- `--host 0.0.0.0` binds to all network interfaces for LAN testing. Keep token protection enabled unless another access layer is in place.
- `--no-token` disables URL/API token protection for local-only use.
- `--x`, `--actual`, `--expected`, and `--denominator` set initial Line/Bar selections.
- `--filters` points to a saved-filter CSV. By default the app tries `./filter_spec.csv`, then `./specs/filter_spec.csv`.
- `--kpis` points to a KPI spec CSV. By default the app tries `./kpi_spec.csv`, then `./specs/kpi_spec.csv`.
- `--features` points to a Feature Specification CSV for GBM feature scenarios, interaction constraints, optional Base metadata, and GLM tabulation `min/max/banding` metadata. By default the app tries `./feature_spec.csv`, then `./specs/feature_spec.csv`.
- `--tools` selects enabled tools in addition to Column Profile, which is always enabled and opens first. The default user-facing tools are `column-profile`, `line-bar`, and `uk-map`. Add `glm` after installing the `glm` extra to train GLMs. Add `models` after installing the `glm` and `gbm` extras to enable both modelling tools; requesting `gbm` also enables `glm` so model comparison and tabulation workflows remain available.

UK map columns default to `PostcodeArea`, `PostcodeSector`, `PostcodeUnit`, `lat`, and `long`. Uppercase aliases such as `POSTCODE_AREA`, `POSTCODE_UNIT`, `LATITUDE`, and `LONGITUDE` are also detected. You can override them:

```bash
.venv/bin/lucidum path/to/my_data.parquet \
  --postcode-area Area \
  --postcode-sector Sector \
  --postcode-unit Unit \
  --latitude latitude \
  --longitude longitude
```

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
    tools=["column_profile", "line_bar", "uk_map"],
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

## KPIs

KPI specs are CSV files with exactly these columns:

```csv
group,name,actual,denominator,decimals,format
VEHICLE,Vehicle age,VEHICLE_AGE,N,1,number
DRIVER,Driver age,DRIVER_AGE,N,1,number
FINANCIAL,Premium,PREMIUM,N,2,currency
```

`denominator` accepts `N`, `Average row value`, an empty value, or `__none__` for average row value, or any numeric column name for weighted response values. `format` accepts `number`, `currency`, or `percent`.

## Feature Specs

Feature Specification CSV files drive GBM feature scenarios, interaction-constraint groups, optional chart Base metadata, and GLM numeric tabulation metadata. The current format starts with these columns, followed by any number of scenario columns:

```csv
Feature,Grouping,Base,min,max,banding,scenario1,scenario2,scenario3
DRIVER_AGE,DRIVER,40,17,96,1,feature,feature,feature
NCD_YEARS,DRIVER,10,0,20,1,feature,,feature
POSTCODE_AREA,POSTCODE,B,,,,,feature,feature
```

`Feature` must match a dataset column name exactly. `Grouping` is optional metadata shown in the GBM Feature table and, when present, is also used to offer GBM feature interaction constraints. `Base` is optional metadata used to anchor Line/Bar and GBM SHAP chart rescaling to `0` or `1` and to define GLM tabulation base cells. Numeric `min`, `max`, and `banding` define GLM rating-table grids; leave them blank for categorical features. Older specs without these metadata columns are still accepted, in which case every column after `Grouping` is treated as a scenario. Each scenario column appears in the GBM scenario dropdown; if a scenario cell contains the word `feature`, case-insensitive, that row is selected when the scenario is chosen.

## GLM Models

The GLM tool is opt-in. Column Profile remains enabled and opens first:

```bash
.venv/bin/lucidum path/to/my_data.parquet --tools line-bar,uk-map,glm
```

The Formula builder accepts either a full `response ~ terms` Formulaic formula, or RHS-only `terms` that use the sidebar Actual metric as the response. Lines can include `#` comments; comments are stored with the model but stripped before fitting. The formula context includes `ifelse`, `pmin`, `pmax`, `ns`, `bs`, `cs`, `poly`, `C`, and common numeric transforms. Explicit `offset(...)` terms are supported; they are stripped from the fitted formula, stored in the manifest, and passed to `glum.fit()` and prediction.

Families are `normal`, `poisson`, `gamma`, `tweedie`, `binomial`, `inverse.gaussian`, and `negative.binomial`, with `link="auto"` in the first implementation. Tweedie power and negative-binomial theta can be set from the family parameter input. If a sidebar Weight is selected, GLM fits `Actual / Weight` with `sample_weight=Weight` and stores `glm_prediction` back on the original Actual scale. Saved models live beside the dataset under `.lucidum/models/glm/`, and the active model publishes a `glm:<model_id>:predictions` data source.

The `Tabulations` tab builds insurance-style rating tables on demand from selected saved GLMs. It uses the fitted `glum` estimator, formula terms, feature spec `Base/min/max/banding`, and any stored `offset(...)` expressions to create base-adjusted linear-predictor tables, table/plot payloads, and row-level `glm_tabulated_prediction`. Numeric feature spec metadata that is missing or blank is estimated from scored rows and reported in the GLM notice. Existing GLMs built before `estimator.pkl` persistence must be rebuilt before tabulation.

The Penalty selector defaults to `None` for the existing unregularized fit. `Auto` uses glum cross-validation over ridge, elastic net, and lasso mixes; `Manual` exposes a compact mix and alpha control. Penalized models show coefficient estimates but suppress coefficient standard errors and p-values because regularized inference is not equivalent to the unpenalized GLM table.

`All` fits all valid rows. `Training` fits only rows where a physical `SAMPLE` column equals `training`, case-insensitively; GLM does not create generated sample splits.

## GBM Models

The GBM tool is opt-in. Column Profile remains enabled and opens first. Requesting `gbm` also enables the GLM tool so the shared tabulation/comparison workflow is available:

```bash
.venv/bin/lucidum path/to/my_data.parquet --tools line-bar,uk-map,models
.venv/bin/lucidum path/to/my_data.parquet --tools gbm --features specs/feature_spec.csv
```

### Feature setup

The GBM tool uses the same sidebar Actual and Weight/KPI controls as Line and Bar, so users can choose the modelling response before training.

If a Feature Specification is loaded, the Feature table shows its `Grouping` values, a multi-select interaction-constraint dropdown, and a scenario dropdown next to `Clear all`; choosing a scenario selects only that scenario's usable features.

Choosing one or more interaction groups constrains the currently selected trainable features in each group so they can only interact with features in the same group, with all other selected features left together in a remainder constraint.

When a saved model has both Gain and saved SHAP rows, the Feature table shows a Gain/SHAP toggle and displays one importance metric at a time; SHAP means mean absolute SHAP value over the saved SHAP rows. For saved EBM models, the same toggle also offers `EBM Gain`, replacing the Feature table with a tree-feature-combination gain summary built from the saved tree table.

When an active GBM has both saved predictions and saved SHAP rows, Line and Bar can show `Partial dependancies > SHAP` ribbons for the selected x-axis feature. These ribbons use the same grouping, banding, filter, denominator, low-weight grouping, and response transform as the chart. The ribbon median is scaled to the active GBM fitted prediction mean for the current chart slice, using the selected Weight when present; categorical x-axes also offer a SHAP sort ordered by median SHAP.

### Training and artifacts

Parameter values can use grid-search braces, such as `{200, 300, 400}`, `{0.05, 0.3; 0.05}`, or `{bagging, goss}` for `data_sample_strategy`; the app samples the hypergrid deterministically, trains one model per valid sampled combination, skips invalid combinations with a notice, and activates the best completed model.

The first parameter row, `init_score`, can stay as `none` for the current behavior or point at a numeric dataset column or fitted GLM prediction artifact. Supplied values are treated as prediction-space baselines, transformed to LightGBM's linear predictor space for objectives with log or logit links, and replace the automatic denominator-derived initial score. When used, the resolved baseline is saved as `init_score.parquet` beside the model.

The `10k` and `100k` SHAP row options save a deterministic random sample from all scored rows using the model seed; `All` saves every scored row.

When feature interaction constraint groups are selected during training, saved `shap_values.parquet` files also include one grouped SHAP contribution column per selected grouping, named like `POSTCODE_INTERACTION_GROUP`. These grouped columns and the active model's prediction column are available in the Line and Bar Actual chooser under separate Dataset, Model predictions, and SHAP values sections.

If the source dataset has a `SAMPLE` column, GBM trains on `training`, early-stops on `test`, and scores `validation` as a holdout. If `SAMPLE` is missing, the tool can create one reusable generated 60/20/20 sidecar split under `.lucidum/models/gbm/`; for durable modelling, add a proper `SAMPLE` column to the original Parquet file. Models are saved beside the dataset under `.lucidum/models/gbm/`.

During training, the app shows live iteration and train/test metric progress and updates the evaluation plot while the background job runs; grid-search progress includes the current model number. The Evaluation Log keeps its live x-axis fixed to the configured iteration count, then uses the exact completed tree count and a tail-focused y-axis view so later training progress remains readable after a steep initial drop. Use the inline `All` / `Tail` control to switch between the full history and a focused tail view. Long evaluation histories are sampled only for browser rendering; saved training logs and artifacts remain complete.

### Saved feature context

When a GBM is trained directly from a feature scenario, the model records that scenario name and the training-time feature list. Selecting a saved GBM shows the recorded scenario in the dropdown. If `feature_spec.csv` has changed since training, the dropdown marks the recorded scenario as changed or missing while the Feature table still reflects the model's saved feature configuration.

When a GBM is trained with feature interaction constraints, the model records the constrained group names and training-time feature lists. Selecting a saved GBM shows those constraints in the interaction dropdown, shows a lock beside constrained Feature table groupings, and marks stale or missing groupings if `feature_spec.csv` has changed. The Model navigator includes a `Constraints` column, and the saved `tree_table.parquet` can be used to inspect which split features appear along each tree path.

### EBM mode

When the active sample source has both `training` and `test` rows, either from a physical dataset `SAMPLE` column or the generated sidecar split, the tool also shows an EBM mode. EBM starts with 2-leaf trees, uses learning rate `0.3` for that 2-leaf stage, then moves through 3, 4, and higher leaf counts up to `num_leaves` whenever the test metric has not improved for `early_stopping_rounds`. `num_iterations` remains the total cap across all EBM stages.

### Saved models

Choosing a saved GBM in the sidebar, or selecting one model in the Model navigator and clicking Activate, makes it active and refreshes the feature table, parameter table, training mode, evaluation plot, tree viewer, SHAP screen, and plot-ready model outputs.

The Model navigator shows the active model with a green dot and lets users select one or more rows for model-folder actions: rename or activate one selected model folder, or delete all selected model folders from `.lucidum/models/gbm/`. Deleting the active model selects the newest remaining model when one exists.

The tree viewer reads saved `tree_table.parquet` artifacts, shows a searchable tree list, and renders the selected tree with zoom and colour controls. Tree split labels use compact numeric formatting, tightly wrap categorical splits on the diagram, and summarize long categorical splits while keeping the full split in tooltips. Node cover shows count and percentage of the selected tree, and clicking a node highlights its path from the root.

### SHAP plots

The SHAP tab is available for GBMs trained with a nonzero SHAP row option. It lists only the active model's trained features, defaults Feature 1 to the highest-Gain feature, defaults Feature 2 to `None`, and can sort both feature choosers by Gain or name.

Switching active models preserves the selected SHAP features when they exist in the next model, and preserves matching plot legend visibility when the same SHAP plot can still be shown; unavailable Feature 2 selections fall back to `None`, and unavailable Feature 1 selections fall back to the first item in the current Feature 1 sort order.

One-feature plots show numeric percentile ribbons around the median or factor box plots, with flame plot x-axes fitted to the plotted data range; when a numeric feature is treated as a factor, its bands keep their natural numeric order. Two-feature plots show a dense-grid 3D surface for two continuous features, line plots for continuous-by-factor selections, and heatmaps for two factor-style selections. Two-feature plots use the sum of the two selected SHAP contributions. The SHAP tab includes a `-` / `0` / `1` rescale control that uses Feature Specification `Base` metadata when present. SHAP `1` rescaling exponentiates values first, then scales to the base response value. Stacked SHAP stays on the linear predictor contribution scale.

Once a model is active, its predictions and SHAP outputs also appear as selectable data sources so Line/Bar and UK Mapping can plot model outputs like normal columns.

## Development

Maintainer notes, architecture details, timing semantics, and test commands live in `DEVELOPMENT.md`.
