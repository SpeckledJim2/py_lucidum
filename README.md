# py_lucidum

`py_lucidum` is a local data exploration workbench for CSV and Parquet files. It starts a small FastAPI server, uses DuckDB for live aggregation, and opens an interactive browser UI for exploring grouped response values, UK postcode choropleths, and postcode unit points.

The implemented tools today are a combined line-and-bar chart and a UK mapping tool. The project structure is designed to grow into independently registered tools for GLM and GBM building as well.

## Current Status

Implemented:

- Line-and-bar chart for local CSV and Parquet datasets.
- DuckDB-backed schema inference and grouped aggregation.
- One or two response lines over an x-axis feature.
- A shared Weight selector for response denominators and blue bar totals.
- Numeric banding and date buckets.
- Low-weight group collapsing with `0`, `10`, `100`, `0.1%`, and `1%` presets, based on the selected Weight.
- DuckDB `WHERE` filters typed directly into the UI.
- Saved filters from `filter_spec.csv` or `specs/filter_spec.csv`.
- Chart/table view switching.
- Client-side table pagination for large grouped outputs.
- UK postcode area and sector choropleth maps using bundled GeoJSON assets and Leaflet.
- UK postcode unit point maps using dataset latitude/longitude columns and a canvas Leaflet layer.
- Map base-layer controls, postcode zoom search, palette controls, opacity/line/label controls, and hot/not-spot highlighting.
- Optional token-protected local server URLs.

Planned placeholders:

- GLM building tool.
- GBM building tool.

## Installation

From the project root:

```bash
/usr/bin/python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -e .
```

This installs the `lucidum` command.

## Quick Start

Launch the bundled motor premiums demo dataset:

```bash
.venv/bin/lucidum --demo --port 8000
```

The command prints a URL like:

```text
Open http://127.0.0.1:8000/?token=...
Saved filters: specs/filter_spec.csv
Uvicorn running on http://127.0.0.1:8000/?token=... (Press CTRL+C to quit)
```

Open the full printed URL in your browser. Stop the server with `Ctrl+C` in the terminal or the red `Stop app` button in the browser header. In either case, an open browser tab greys out and shows a stopped message once the local server is gone.

From a source checkout, the same demo data is available at `datasets/motor_premiums.parquet`:

```bash
.venv/bin/lucidum datasets/motor_premiums.parquet --port 8000
```

The demo file is included in installed packages, so `--demo` works even when the repository checkout is not available.

## Other Datasets

Run Lucidum against any local CSV or Parquet file by passing its path:

```bash
.venv/bin/lucidum path/to/my_data.parquet --port 8000
.venv/bin/lucidum path/to/my_data.csv --port 8000
```

Parquet is recommended for normal use because DuckDB reads it much faster. If your dataset uses different postcode or coordinate columns, pass them explicitly:

```bash
.venv/bin/lucidum path/to/my_data.parquet --postcode-area Area --postcode-sector Sector --postcode-unit Unit --latitude latitude --longitude longitude
```

## Common Options

```bash
.venv/bin/lucidum --demo --open --port 8000
.venv/bin/lucidum --demo --host 0.0.0.0 --port 8000
.venv/bin/lucidum --demo --no-token
.venv/bin/lucidum --demo --x DRIVER_AGE --actual PREMIUM --denominator ANNUAL_MILEAGE
.venv/bin/lucidum --demo --filters specs/filter_spec.csv
.venv/bin/lucidum --demo --no-filters
.venv/bin/lucidum --demo --postcode-area POSTCODE_AREA --postcode-sector POSTCODE_SECTOR
.venv/bin/lucidum --demo --postcode-unit POSTCODE_UNIT --latitude LATITUDE --longitude LONGITUDE
.venv/bin/lucidum --demo --tools line-bar
```

- `--open` opens the generated URL with Python's configured browser or viewer handler.
- `--host 0.0.0.0` binds to all network interfaces for LAN testing.
- `--no-token` disables URL/API token protection for local-only use.
- `--x`, `--actual`, `--expected`, and `--denominator` set initial chart selections.
- `--filters` points to a saved-filter CSV file.
- `--no-filters` disables saved filters and skips default filter-spec discovery.
- `--postcode-area` and `--postcode-sector` set the dataset columns used by UK mapping choropleths. Defaults resolve `PostcodeArea`/`POSTCODE_AREA` and `PostcodeSector`/`POSTCODE_SECTOR`.
- `--postcode-unit`, `--latitude`, and `--longitude` set the dataset columns used by UK mapping unit points. Defaults resolve `PostcodeUnit`/`POSTCODE_UNIT`, `lat`/`latitude`/`LATITUDE`, and `long`/`longitude`/`LONGITUDE`.
- `--tools` selects enabled tools. By default both `line-bar` and `uk-map` are enabled; use `--tools line-bar` to launch only the chart/table tool.

## Python Usage

```python
import py_lucidum

py_lucidum.serve(py_lucidum.demo_dataset_path(), port=8000, open_browser=True)
```

In notebook-style runtimes such as Positron or Jupyter, `serve()` starts the server in the background and returns the URL immediately. `open_browser=True` uses Python's configured browser or viewer handler, so Positron may open the app in the Viewer pane rather than an external browser. Use the app `Stop app` button to stop it.

To launch only the line-and-bar tool explicitly:

```python
import py_lucidum

py_lucidum.serve_line_bar(py_lucidum.demo_dataset_path(), port=8000, open_browser=True)
```

For programmatic ASGI usage:

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
        "postcode_area": "POSTCODE_AREA",
        "postcode_sector": "POSTCODE_SECTOR",
        "postcode_unit": "POSTCODE_UNIT",
        "latitude": "LATITUDE",
        "longitude": "LONGITUDE",
    },
    filters_path="specs/filter_spec.csv",
    use_saved_filters=True,
    tools=["line_bar", "uk_map"],
)

py_lucidum.run_app(app, host="127.0.0.1", port=8000, open_browser=True)
```

Then open, if it was not opened automatically:

```text
http://127.0.0.1:8000/?token=dev-token
```

Use raw `uvicorn.run(app, ...)` only from a standalone Python script or terminal. Positron and Jupyter already have an asyncio event loop running, so `uvicorn.run()` raises `RuntimeError: asyncio.run() cannot be called from a running event loop` there.

## Line-Bar Weights

The line-and-bar chart uses one shared **Weight** selector for both response lines.

- `Average row value` divides each response by the count of rows where all selected response values are non-null. The blue bars show that row count.
- Selecting a numeric Weight column divides Actual and Expected by `SUM(weight)` for each x-axis group, using rows where the selected response values and Weight are non-null. The blue bars show the same `SUM(weight)`.
- Low-weight grouping and tail grouping use the selected Weight total, not raw row count.
- The grey value next to the Weight label is the filtered total Weight used by the chart.
- If rows are excluded because selected response values or Weight values are missing, the chart status text reports that. It also reports zero or negative Weight values.

## UK Mapping

The UK mapping tool is available from the sidebar tool selector when `uk-map` is enabled. It uses the selected Actual column, Weight denominator, and active filter in the same way as the line-and-bar chart.

- The map can show postcode area or postcode sector choropleths, plus postcode unit points. Dataset join columns default to `PostcodeArea`, `PostcodeSector`, and `PostcodeUnit`, with uppercase aliases supported for `POSTCODE_AREA`, `POSTCODE_SECTOR`, and `POSTCODE_UNIT`.
- Unit point coordinates default to numeric `lat` and `long` columns, with `latitude`/`LATITUDE` and `longitude`/`LONGITUDE` aliases supported. Override them with `--latitude` and `--longitude`; if no point columns are configured and the defaults are absent, the Units layer is disabled.
- Area and sector geometries are served from bundled GeoJSON assets under `py_lucidum.tools.uk_map.static`.
- Unit points are grouped by postcode unit, average their latitude/longitude, and plot only units with a valid KPI and valid coordinates.
- The map layer control offers Blank, Esri, Grey, OSM, and Satellite base maps. Blank uses the selected white/dark map background.
- The floating map control supports area/sector postcode search (`PO`, `PO15 7`, or `PO15 7JT` normalised to `PO15 7`), draggable placement, divergent/spectral/viridis palettes, line thickness, opacity, hot/not-spot highlighting, and polygon labels.
- The divergent palette is the default. Colour orders run from green/blue/yellow at low values toward red/purple at high values, depending on the selected palette.
- The legend shows ten quantile categories and omits the metric title so long metric names do not widen it.

## Filters

The sidebar filter box accepts DuckDB `WHERE` expressions:

```sql
DRIVER_AGE > 40
ANNUAL_MILEAGE >= 20000
VEHICLE_USAGE = 'Social only'
QUOTE_DATE >= DATE '2017-01-01'
```

Use double quotes for column names containing punctuation, spaces, or other special characters.

Saved filters are loaded from `filter_spec.csv` in the working directory by default, falling back to `specs/filter_spec.csv` when that exists. Use `--no-filters` to start without saved filters. The file format is:

```csv
name,expression
Older drivers,DRIVER_AGE > 40
High annual mileage,ANNUAL_MILEAGE >= 20000
```

Use `--filters path/to/filter_spec.csv` to choose another file.

## Tool Architecture

The codebase is split into shared infrastructure and independent tools:

```text
py_lucidum.core                 shared DuckDB dataset, schema, filter, and SQL helpers
py_lucidum.app                  FastAPI app factory and shared app context
py_lucidum.tools.line_bar       implemented line-and-bar chart tool
py_lucidum.tools.uk_map         implemented UK mapping tool
py_lucidum.tools.glm            placeholder package for GLM building
py_lucidum.tools.gbm            placeholder package for GBM building
```

The line-and-bar chart exposes both endpoints:

```text
POST /api/chart
POST /api/line-bar/chart
```

`/api/chart` is retained for compatibility with the current frontend. New tool-specific integrations should prefer namespaced endpoints.

The UK mapping tool exposes:

```text
POST /api/uk-map/summary       area, sector, or unit aggregation
GET  /tools/uk-map/static/geodata/...  area/sector GeoJSON assets
```

Shared app endpoints include:

```text
GET  /api/schema
GET  /api/health
POST /api/reload
POST /api/shutdown
```

## Development

Run the test suite:

```bash
.venv/bin/python -m unittest discover -s tests
```

Useful checks before committing:

```bash
.venv/bin/python -m compileall src tests
node --check src/py_lucidum/static/app.js
git diff --check
```

Optional browser smoke tests require Playwright and a Chromium browser install:

```bash
.venv/bin/python -m pip install pytest pytest-playwright
.venv/bin/python -m playwright install chromium
PY_LUCIDUM_RUN_BROWSER_TESTS=1 .venv/bin/python -m pytest tests/test_browser_smoke.py
```

Project planning notes live in `PROJECT_PLAN.md`. More detailed launch notes live in `USAGE.md`.

## Notes

- `datasets/motor_premiums.parquet` is the committed demo dataset; other local files under `datasets/` remain ignored by git.
- Saved-filter CSVs under `specs/` are tracked.
- The app treats input files as fixed during a session until the user presses Reload.
- The current frontend is a static ECharts and Leaflet app. A larger frontend framework can be reconsidered later if the planned toolset makes it worthwhile.
