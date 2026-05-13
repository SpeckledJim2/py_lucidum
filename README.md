# py_lucidum

`py_lucidum` is a local data exploration workbench for CSV and Parquet files. It starts a small FastAPI server, uses DuckDB for live aggregation, and opens an interactive browser UI for exploring grouped response values.

The implemented tool today is a combined line-and-bar chart. The project structure is designed to grow into independently registered tools for GLM building, GBM building, and UK mapping.

## Current Status

Implemented:

- Line-and-bar chart for local CSV and Parquet datasets.
- DuckDB-backed schema inference and grouped aggregation.
- One or two response lines over an x-axis feature.
- Row volume bars.
- Numeric banding and date buckets.
- Low-weight group collapsing with `0`, `10`, `100`, `0.1%`, and `1%` presets.
- DuckDB `WHERE` filters typed directly into the UI.
- Saved filters from `filter_spec.csv`.
- Chart/table view switching.
- Client-side table pagination for large grouped outputs.
- Optional token-protected local server URLs.

Planned placeholders:

- GLM building tool.
- GBM building tool.
- UK mapping tool.

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
```

Open the full printed URL in your browser. Stop the server with `Ctrl+C`.

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
.venv/bin/lucidum my_data.parquet --x YoungestDriverAge --actual AvgPrice1_5 --expected glm_prediction
.venv/bin/lucidum my_data.parquet --filters filter_spec.csv
.venv/bin/lucidum my_data.parquet --tools line-bar
```

- `--open` opens the generated URL in your default browser.
- `--host 0.0.0.0` binds to all network interfaces for LAN testing.
- `--no-token` disables URL/API token protection for local-only use.
- `--x`, `--actual`, and `--expected` set initial chart selections.
- `--filters` points to a saved-filter CSV file.
- `--tools line-bar` explicitly enables the line-and-bar tool. This is currently also the default.

## Python Usage

```python
import py_lucidum

py_lucidum.serve("my_data.parquet", port=8000, open_browser=True)
```

To launch only the line-and-bar tool explicitly:

```python
import py_lucidum

py_lucidum.serve_line_bar("my_data.parquet", port=8000, open_browser=True)
```

For programmatic ASGI usage:

```python
import uvicorn
from py_lucidum.app import create_app

app = create_app(
    "my_data.parquet",
    token="dev-token",
    defaults={"x": "YoungestDriverAge", "actual": "AvgPrice1_5"},
    filters_path="filter_spec.csv",
    tools=["line_bar"],
)

uvicorn.run(app, host="127.0.0.1", port=8000)
```

Then open:

```text
http://127.0.0.1:8000/?token=dev-token
```

## Filters

The filter bar accepts DuckDB `WHERE` expressions:

```sql
YoungestDriverAge > 40
"Gross.Weight" >= 3000
UseofVan = 'Social'
QuoteDate >= DATE '2024-01-01'
```

Use double quotes for column names containing punctuation, spaces, or other special characters.

Saved filters are loaded from `filter_spec.csv` in the working directory by default. The file format is:

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
py_lucidum.tools.glm            placeholder package for GLM building
py_lucidum.tools.gbm            placeholder package for GBM building
py_lucidum.tools.uk_map         placeholder package for UK mapping
```

The line-and-bar chart exposes both endpoints:

```text
POST /api/chart
POST /api/line-bar/chart
```

`/api/chart` is retained for compatibility with the current frontend. New tool-specific integrations should prefer namespaced endpoints.

## Development

Run the line-and-bar backend tests:

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

- Local development datasets such as `vans.csv` and `vans.parquet` are intentionally ignored by git.
- The app treats input files as fixed during a session until the user presses Reload.
- The current frontend is a static ECharts app. A larger frontend framework can be reconsidered later if the planned toolset makes it worthwhile.
