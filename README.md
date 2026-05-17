# py_lucidum

`py_lucidum` is a local browser workbench for exploring CSV and Parquet datasets. It starts a small FastAPI server, uses DuckDB for live aggregation, and opens an interactive UI for grouped charts, filters, and UK postcode maps.

The current app includes:

- A combined line-and-bar chart over any dataset feature.
- One or two response lines with a shared Weight selector.
- Numeric banding, date buckets, low-weight grouping, table view, and saved filters.
- UK postcode area and sector choropleths using bundled GeoJSON assets.
- UK postcode unit points using dataset latitude/longitude columns.
- Optional token-protected local URLs and a browser Stop app button.

The repository includes one synthetic demo dataset at `datasets/motor_premiums.parquet`, and installed packages include the same file for `lucidum --demo`.

## Installation

From the project root:

```bash
/usr/bin/python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -e .
```

This installs the `lucidum` command.

## Quick Start

Launch the bundled demo dataset:

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

From a source checkout, the same data can also be loaded directly:

```bash
.venv/bin/lucidum datasets/motor_premiums.parquet --port 8000
```

## Running Your Own Data

Pass a CSV or Parquet file path:

```bash
.venv/bin/lucidum path/to/my_data.parquet --port 8000
.venv/bin/lucidum path/to/my_data.csv --port 8000
```

Parquet is recommended for normal use because DuckDB reads it much faster than CSV.

If your UK mapping columns use different names, pass them explicitly:

```bash
.venv/bin/lucidum path/to/my_data.parquet \
  --postcode-area Area \
  --postcode-sector Sector \
  --postcode-unit Unit \
  --latitude latitude \
  --longitude longitude
```

## Common Options

```bash
.venv/bin/lucidum --demo --open --port 8000
.venv/bin/lucidum --demo --host 0.0.0.0 --port 8000
.venv/bin/lucidum --demo --no-token
.venv/bin/lucidum --demo --x DRIVER_AGE --actual PREMIUM --denominator ANNUAL_MILEAGE
.venv/bin/lucidum --demo --filters specs/filter_spec.csv
.venv/bin/lucidum --demo --no-filters
.venv/bin/lucidum --demo --tools line-bar
```

- `--open` asks Python to open the generated URL with its configured browser or viewer handler.
- `--host 0.0.0.0` binds to all network interfaces for LAN testing. Keep the generated token enabled unless you have another access control layer.
- `--no-token` disables URL/API token protection for local-only use.
- `--x`, `--actual`, `--expected`, and `--denominator` set initial chart selections.
- `--filters` points to a saved-filter CSV file. By default the app tries `./filter_spec.csv`, then `./specs/filter_spec.csv`.
- `--no-filters` disables saved-filter discovery.
- `--tools` selects enabled tools. By default both `line-bar` and `uk-map` are enabled.

UK map columns default to `PostcodeArea`, `PostcodeSector`, `PostcodeUnit`, `lat`, and `long`. Uppercase aliases such as `POSTCODE_AREA`, `POSTCODE_UNIT`, `LATITUDE`, and `LONGITUDE` are also detected.

## Python Usage

Launch the demo from Python:

```python
import py_lucidum

py_lucidum.serve(py_lucidum.demo_dataset_path(), port=8000, open_browser=True)
```

Launch your own Parquet dataset the same way:

```python
import py_lucidum

py_lucidum.serve("path/to/my_data.parquet", port=8000, open_browser=True)
```

CSV files are also supported:

```python
py_lucidum.serve("path/to/my_data.csv", port=8000, open_browser=True)
```

In notebook-style runtimes such as Positron or Jupyter, `serve()` starts the server in the background and returns the URL immediately. In a normal Python shell, it blocks until stopped.

To launch only the line-and-bar tool, pass either the demo path or your own dataset path:

```python
import py_lucidum

py_lucidum.serve_line_bar(py_lucidum.demo_dataset_path(), port=8000, open_browser=True)
py_lucidum.serve_line_bar("path/to/my_data.parquet", port=8000, open_browser=True)
```

For ASGI usage, pass the same kind of dataset path to `create_app()`:

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
    tools=["line_bar", "uk_map"],
)

py_lucidum.run_app(app, host="127.0.0.1", port=8000, open_browser=True)
```

## Features

**Line and bar chart**

- Select any feature for the x-axis.
- Select Actual and optional Expected numeric response lines.
- Use `Average row value` or a numeric Weight column as the denominator.
- Bucket numeric and date axes, collapse low-weight groups, switch between chart and table views, and apply optional response transforms.

**UK mapping**

- Switch to UK mapping from the sidebar tool selector.
- Show postcode area or sector choropleths, plus postcode unit points when unit and coordinate columns are available.
- Use the floating map control for postcode search, palette selection, blank-map background, line thickness, opacity, hot/not-spot highlighting, and polygon labels.

**Filters and saved filters**

The filter box accepts DuckDB `WHERE` expressions:

```sql
DRIVER_AGE > 40
ANNUAL_MILEAGE >= 20000
VEHICLE_USAGE = 'Social only'
QUOTE_DATE >= DATE '2017-01-01'
```

Saved filters are CSV files with exactly these columns:

```csv
name,expression
Older drivers,DRIVER_AGE > 40
High annual mileage,ANNUAL_MILEAGE >= 20000
```

## Development

Run the standard test suite:

```bash
.venv/bin/python -m unittest discover -s tests
```

Useful checks before committing:

```bash
.venv/bin/python -m compileall src tests
node --check src/py_lucidum/static/app.js
git diff --check
```

Optional browser smoke tests require Playwright and Chromium:

```bash
.venv/bin/python -m pip install pytest pytest-playwright
.venv/bin/python -m playwright install chromium
PY_LUCIDUM_RUN_BROWSER_TESTS=1 .venv/bin/python -m pytest tests/test_browser_smoke.py
```

Maintainer and architecture notes live in `DEVELOPMENT.md`.
