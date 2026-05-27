# lucidum

`lucidum` is a local browser workbench for exploring CSV and Parquet datasets. It starts a small local FastAPI server, reads data with DuckDB, and opens an interactive UI for profiling columns, plotting grouped metrics, applying filters, and mapping UK postcode data.

The app is designed for local analysis: your dataset stays on the machine running `lucidum`.

## Current Tools

- **Column Profile**: review dataset columns, missing values, distinct counts, ranges, value counts, and numeric/date distributions.
- **Line and Bar**: plot grouped Actual and optional Expected response values over any feature, with shared Weight, banding, date buckets, tables, transforms, and sigma bars.
- **UK Mapping**: map postcode areas and sectors with bundled GeoJSON, or postcode units when unit and coordinate columns are available.
- **GBM**: optional LightGBM model building with persistent sidecar artifacts, predictions that can be plotted as chart/map data sources, evaluation plots, model navigation, and tree viewing.
- **Filters and KPIs**: apply free-form DuckDB `WHERE` filters, saved filter rows, and KPI specs that set Actual/Weight choices and formatting.

Unreadable dataset columns, such as Parquet strings with invalid UTF-8, are skipped by the shared schema used by normal selectors. Column Profile reports them as skipped, and the GBM feature chooser shows them as disabled invalid rows.

The GLM tool slot exists in the codebase for future modelling work, but GLM model building is not part of the current user-facing release.

## Installation

From the project root:

```bash
python3.13 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -e .
```

This installs the `lucidum` command inside the virtual environment.

To enable GBM model training, install the optional modelling extra:

```bash
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
.venv/bin/lucidum --demo --tools line-bar
.venv/bin/lucidum path/to/my_data.parquet --tools line-bar,uk-map,gbm
```

- `--open` opens the generated URL with Python's configured browser handler.
- `--host 0.0.0.0` binds to all network interfaces for LAN testing. Keep token protection enabled unless another access layer is in place.
- `--no-token` disables URL/API token protection for local-only use.
- `--x`, `--actual`, `--expected`, and `--denominator` set initial Line/Bar selections.
- `--filters` points to a saved-filter CSV. By default the app tries `./filter_spec.csv`, then `./specs/filter_spec.csv`.
- `--kpis` points to a KPI spec CSV. By default the app tries `./kpi_spec.csv`, then `./specs/kpi_spec.csv`.
- `--tools` selects enabled tools in addition to Column Profile, which is always enabled and opens first. The default user-facing tools are `column-profile`, `line-bar`, and `uk-map`. Add `gbm` after installing the `gbm` extra to train LightGBM models.

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

## GBM Models

The GBM tool is opt-in. Column Profile remains enabled and opens first:

```bash
.venv/bin/lucidum path/to/my_data.parquet --tools line-bar,uk-map,gbm
```

The GBM tool uses the same sidebar Actual and Weight/KPI controls as Line and Bar, so users can choose the modelling response before training. If the source dataset has a `SAMPLE` column, GBM trains on `training`, early-stops on `test`, and scores `validation` as a holdout. If `SAMPLE` is missing, the tool can create one reusable generated 60/20/20 sidecar split under `.lucidum/models/gbm/`; for durable modelling, add a proper `SAMPLE` column to the original Parquet file. Models are saved beside the dataset under `.lucidum/models/gbm/`. During training, the app shows live iteration and train/test metric progress and updates the evaluation plot while the background job runs. The Evaluation Log keeps its live x-axis fixed to the configured iteration count, then uses the exact completed tree count and a tail-focused y-axis view so later training progress remains readable after a steep initial drop. Use the inline `All` / `Tail` control to switch between the full history and a focused tail view. Long evaluation histories are sampled only for browser rendering; saved training logs and artifacts remain complete.

When a physical dataset `SAMPLE` column has both `training` and `test` rows, the tool also shows an EBM mode. EBM starts with 2-leaf trees, uses learning rate `0.3` for that 2-leaf stage, then moves through 3, 4, and higher leaf counts up to `num_leaves` whenever the test metric has not improved for `early_stopping_rounds`. `num_iterations` remains the total cap across all EBM stages. Generated sample sidecars do not enable EBM mode.

Clicking a saved GBM makes it active and refreshes the feature table, parameter table, training mode, evaluation plot, tree viewer, and plot-ready model outputs. The Model navigator can also rename a model folder or delete a model folder from `.lucidum/models/gbm/`; deleting the active model selects the newest remaining model when one exists. The tree viewer reads saved `tree_table.parquet` artifacts, shows a searchable tree list, and renders the selected tree with zoom and colour controls. Once a model is active, its predictions and SHAP outputs appear as selectable data sources so Line/Bar and UK Mapping can plot model outputs like normal columns.

## Development

Maintainer notes, architecture details, timing semantics, and test commands live in `DEVELOPMENT.md`.
