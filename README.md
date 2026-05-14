# py_lucidum

`py_lucidum` is a local data exploration workbench for CSV and Parquet files. It starts a small FastAPI server, uses DuckDB for live aggregation, and opens an interactive browser UI for exploring grouped response values and UK postcode choropleths.

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

Launch the app against a Parquet file:

```bash
.venv/bin/lucidum my_data.parquet --port 8000
```

The command prints a URL like:

```text
Open http://127.0.0.1:8000/?token=...
Saved filters: specs/filter_spec.csv
Uvicorn running on http://127.0.0.1:8000/?token=... (Press CTRL+C to quit)
```

Open the full printed URL in your browser. Stop the server with `Ctrl+C` in the terminal or the red `Stop app` button in the browser header.

CSV files also work:

```bash
.venv/bin/lucidum my_data.csv --port 8000
```

Parquet is recommended for normal use because DuckDB reads it much faster.

## Common Options

```bash
.venv/bin/lucidum my_data.parquet --open --port 8000
.venv/bin/lucidum my_data.parquet --host 0.0.0.0 --port 8000
.venv/bin/lucidum my_data.parquet --no-token
.venv/bin/lucidum my_data.parquet --x YoungestDriverAge --actual AvgPrice1_5 --expected glm_prediction --denominator Exposure
.venv/bin/lucidum my_data.parquet --filters path/to/filter_spec.csv
.venv/bin/lucidum my_data.parquet --no-filters
.venv/bin/lucidum my_data.parquet --postcode-area PostcodeArea --postcode-sector PostcodeSector
.venv/bin/lucidum my_data.parquet --tools line-bar
```

- `--open` opens the generated URL with Python's configured browser or viewer handler.
- `--host 0.0.0.0` binds to all network interfaces for LAN testing.
- `--no-token` disables URL/API token protection for local-only use.
- `--x`, `--actual`, `--expected`, and `--denominator` set initial chart selections.
- `--filters` points to a saved-filter CSV file.
- `--no-filters` disables saved filters and skips default filter-spec discovery.
- `--postcode-area` and `--postcode-sector` set the dataset columns used by the UK mapping tool. They default to `PostcodeArea` and `PostcodeSector`.
- `--tools` selects enabled tools. By default both `line-bar` and `uk-map` are enabled; use `--tools line-bar` to launch only the chart/table tool.

## Python Usage

```python
import py_lucidum

py_lucidum.serve("my_data.parquet", port=8000, open_browser=True)
```

In notebook-style runtimes such as Positron or Jupyter, `serve()` starts the server in the background and returns the URL immediately. `open_browser=True` uses Python's configured browser or viewer handler, so Positron may open the app in the Viewer pane rather than an external browser. Use the app `Stop app` button to stop it.

To launch only the line-and-bar tool explicitly:

```python
import py_lucidum

py_lucidum.serve_line_bar("my_data.parquet", port=8000, open_browser=True)
```

For programmatic ASGI usage:

```python
import py_lucidum
from py_lucidum.app import create_app

app = create_app(
    "my_data.parquet",
    token="dev-token",
    defaults={
        "x": "YoungestDriverAge",
        "actual": "AvgPrice1_5",
        "denominator": "Exposure",
        "postcode_area": "PostcodeArea",
        "postcode_sector": "PostcodeSector",
    },
    filters_path="path/to/filter_spec.csv",
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

- The map can show postcode area or postcode sector choropleths. Dataset join columns default to `PostcodeArea` and `PostcodeSector`, configurable with `--postcode-area` and `--postcode-sector`.
- Area and sector geometries are served from bundled GeoJSON assets under `py_lucidum.tools.uk_map.static`.
- The map layer control offers Blank, Esri, Grey, OSM, and Satellite base maps. Blank uses the selected white/dark map background.
- The floating map control supports postcode search (`PO`, `PO15 7`, or `PO15 7JT`), draggable placement, divergent/spectral/viridis palettes, line thickness, opacity, hot/not-spot highlighting, and polygon labels.
- The divergent palette is the default. Colour orders run from green/blue/yellow at low values toward red/purple at high values, depending on the selected palette.
- The legend shows ten quantile categories and omits the metric title so long metric names do not widen it.

## Filters

The sidebar filter box accepts DuckDB `WHERE` expressions:

```sql
YoungestDriverAge > 40
"Gross.Weight" >= 3000
UseofVan = 'Social'
QuoteDate >= DATE '2024-01-01'
```

Use double quotes for column names containing punctuation, spaces, or other special characters.

Saved filters are loaded from `filter_spec.csv` in the working directory by default, falling back to `specs/filter_spec.csv` when that exists. Use `--no-filters` to start without saved filters. The file format is:

```csv
name,expression
Older drivers,YoungestDriverAge > 40
Heavy vans,"""Gross.Weight"" >= 3000"
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
POST /api/uk-map/summary
GET  /tools/uk-map/static/geodata/...
```

## Development

Run the test suite:

```bash
.venv/bin/python -m unittest discover -s tests
```

Useful checks before committing:

```bash
.venv/bin/python -m compileall src tests
perl -0777 -ne 'print $1 if m{<script>(.*)</script>}s' src/py_lucidum/static/index.html | node --check -
git diff --check
```

Project planning notes live in `PROJECT_PLAN.md`. More detailed launch notes live in `USAGE.md`.

## Notes

- Local development datasets under `datasets/` are intentionally ignored by git.
- Saved-filter CSVs under `specs/` are tracked.
- The app treats input files as fixed during a session until the user presses Reload.
- The current frontend is a static ECharts and Leaflet app. A larger frontend framework can be reconsidered later if the planned toolset makes it worthwhile.
