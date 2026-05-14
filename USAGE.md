# Launching py_lucidum

This project is currently developed against local example files such as `datasets/vans.parquet` and `datasets/home.parquet`.
Those files are intentionally ignored by git because they are local development datasets.

## One-Time Setup

From the project root, create and install into a virtual environment:

```bash
/usr/bin/python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -e .
```

Use `/usr/bin/python3` on this machine because the installed Homebrew Python builds currently have a broken `pyexpat` dynamic library.

## Launch From The Terminal

Run the installed command:

```bash
.venv/bin/lucidum datasets/vans.parquet --port 8000
```

The app prints a URL like:

```text
Open http://127.0.0.1:8000/?token=...
Saved filters: specs/filter_spec.csv
Uvicorn running on http://127.0.0.1:8000/?token=... (Press CTRL+C to quit)
```

Open that URL in the browser. Stop the server with `Ctrl+C` in the terminal, or use the red `Stop app` button in the browser header.
The same printed URL can also be opened in the Positron Viewer pane.

Useful options:

```bash
.venv/bin/lucidum datasets/vans.parquet --open --port 8000
.venv/bin/lucidum datasets/vans.parquet --host 0.0.0.0 --port 8000
.venv/bin/lucidum datasets/vans.parquet --no-token
.venv/bin/lucidum datasets/vans.parquet --x YoungestDriverAge --actual AvgPrice1_5 --expected glm_prediction --denominator Gross.Weight
.venv/bin/lucidum datasets/vans.parquet --filters specs/filter_spec.csv
.venv/bin/lucidum datasets/home.parquet --no-filters --port 8000
.venv/bin/lucidum datasets/home.parquet --postcode-area PostcodeArea --postcode-sector PostcodeSector
.venv/bin/lucidum datasets/vans.parquet --tools line-bar
```

- `--open` asks Python to open the generated URL with its configured browser or viewer handler. Positron may open it in the Viewer pane rather than an external browser.
- `--host 0.0.0.0` is useful for internal server/LAN testing. Keep the generated token enabled unless you have another access control layer.
- `--no-token` is convenient for local-only testing and makes API requests work without the generated query-string token.
- `--x`, `--actual`, `--expected`, and `--denominator` set the initial x-axis feature, Actual / line 1 feature, Expected / line 2 feature, and Weight column.
- `--filters` sets the saved-filter CSV path. If omitted, the app loads `./filter_spec.csv` from the working directory when it exists, otherwise `./specs/filter_spec.csv` when it exists.
- `--no-filters` disables saved filters and skips the default filter-spec lookup.
- `--postcode-area` and `--postcode-sector` set the dataset columns used by the UK mapping tool. They default to `PostcodeArea` and `PostcodeSector`.
- `--tools` selects which tool components to enable. By default both `line-bar` and `uk-map` are enabled; use `--tools line-bar` to launch only the chart/table tool.
- Without explicit defaults, the app starts with the first dataset column on the x-axis, the first numeric column as Actual / line 1, and no Expected / line 2.
- URL parameters can also set the same initial selections, for example `http://127.0.0.1:8000/?x=YoungestDriverAge&actual=AvgPrice1_5&expected=glm_prediction&denominator=Gross.Weight&postcode_area=PostcodeArea&postcode_sector=PostcodeSector`.

## Launch As A Python Module

This is equivalent to the console command:

```bash
.venv/bin/python -m py_lucidum datasets/vans.parquet --port 8000
```

## Launch From A Python Console

From a Python shell started in the project root:

```python
import py_lucidum

py_lucidum.serve("datasets/vans.parquet", port=8000, open_browser=True)
```

In a normal Python shell, this call starts the Uvicorn server and blocks until the server is stopped.
In notebook-style runtimes such as Positron or Jupyter, where an asyncio event loop is already running, it starts the server in the background and returns the URL immediately. `open_browser=True` uses Python's configured browser or viewer handler, so Positron may open the app in the Viewer pane rather than an external browser. Use the red `Stop app` button in the app header to stop it.

The line-and-bar chart can also be launched explicitly:

```python
import py_lucidum

py_lucidum.serve_line_bar("datasets/vans.parquet", port=8000, open_browser=True)
```

## Launch A Custom ASGI App Programmatically

For server-style usage from Python, create the ASGI app and pass it to the py_lucidum runner:

```python
import py_lucidum
from py_lucidum.app import create_app

app = create_app("datasets/vans.parquet", token="dev-token")
py_lucidum.run_app(app, host="127.0.0.1", port=8000, open_browser=True)
```

Then open, if it was not opened automatically:

```text
http://127.0.0.1:8000/?token=dev-token
```

`run_app()` handles both normal Python shells and notebook-style runtimes such as Positron or Jupyter. In a normal Python shell it blocks until stopped. In Positron or Jupyter it starts the server in the background and returns the URL immediately.

Initial selections can be supplied programmatically:

```python
app = create_app(
    "datasets/vans.parquet",
    token="dev-token",
    defaults={
        "x": "YoungestDriverAge",
        "actual": "AvgPrice1_5",
        "expected": "glm_prediction",
        "denominator": "Gross.Weight",
        "postcode_area": "PostcodeArea",
        "postcode_sector": "PostcodeSector",
    },
    filters_path="specs/filter_spec.csv",
    use_saved_filters=True,
    tools=["line_bar", "uk_map"],
)
```

If you specifically want raw Uvicorn, run it from a standalone Python script or terminal, not from an already-running Positron/Jupyter event loop:

```python
import uvicorn
from py_lucidum.app import create_app

app = create_app("datasets/vans.parquet", token="dev-token")

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
```

## Tool Structure

`py_lucidum` is organised as a shared workbench plus independently registered tools:

```text
py_lucidum.core                 shared DuckDB dataset, schema, filter, and SQL helpers
py_lucidum.app                  FastAPI app factory and shared app context
py_lucidum.tools.line_bar       implemented line-and-bar chart tool
py_lucidum.tools.uk_map         implemented UK mapping tool
py_lucidum.tools.glm            placeholder package for GLM building
py_lucidum.tools.gbm            placeholder package for GBM building
```

The current browser UI opens the line-and-bar chart first and provides a sidebar tool selector for UK mapping when the map tool is enabled. Internally, chart requests are served by the line-and-bar tool. Both chart endpoints are available:

```text
POST /api/chart
POST /api/line-bar/chart
```

`/api/chart` is retained for compatibility with the existing frontend. New tool-specific integrations should prefer the namespaced endpoint.

The UK mapping tool uses:

```text
POST /api/uk-map/summary
GET  /tools/uk-map/static/geodata/...
```

## Line-And-Bar Weights

The line-and-bar chart has one shared **Weight** selector:

- `Average row value` divides Actual and Expected by the count of rows where all selected response values are non-null. The blue bars show that same row count.
- Selecting a numeric Weight column divides Actual and Expected by `SUM(weight)` within each x-axis group, using rows where the selected response values and Weight are non-null. The blue bars show the same `SUM(weight)`.
- The grey value next to the Weight label is the filtered total Weight used by the chart.
- Low-weight grouping, including Low tail, High tail, and Other groups, uses the selected Weight total instead of raw row count.
- The chart status text reports rows excluded because selected response values or Weight values are missing. It also reports zero or negative Weight values.

## UK Mapping

Use the sidebar tool selector to switch to **UK mapping**. The map uses the selected Actual column, Weight denominator, and active filter in the same way as the line-and-bar chart.

- Dataset postcode columns default to `PostcodeArea` and `PostcodeSector`. Override them with `--postcode-area` and `--postcode-sector` when needed.
- Postcode area and sector GeoJSON assets are bundled with the app and served to Leaflet.
- The map layer control provides Blank, Esri, Grey, OSM, and Satellite base maps, plus Area and Sector choropleth overlays.
- The draggable floating map panel provides postcode zoom search, palette selection, white/dark blank-map backgrounds, line thickness, opacity, hot/not-spot highlighting, and polygon labels.
- Postcode search accepts area (`PO`), sector (`PO15 7`), and full postcode-like inputs (`PO15 7JT`, normalised to `PO15 7` because unit-level geometry is not included yet).
- The divergent palette is selected on startup. Palette order is reversed from the usual low-red/high-green convention so low values are green/blue/yellow and high values move toward red/purple, depending on palette.
- The legend shows ten quantile categories and omits the metric title so long metric names do not widen the legend.

## Filters

The sidebar filter box accepts a DuckDB `WHERE` expression. Type the expression and press Enter or click Apply. Clear removes the active filter.

Examples:

```sql
YoungestDriverAge > 40
"Gross.Weight" >= 3000
UseofVan = 'Social'
QuoteDate >= DATE '2024-01-01'
```

Use double quotes for column names that contain punctuation, spaces, or other special characters, such as `"Gross.Weight"`.

Saved filters are read from `filter_spec.csv` in the working directory by default, falling back to `specs/filter_spec.csv` for the tidied project layout. Use `--filters path/to/filter_spec.csv` to choose another file, or `--no-filters` to start without any saved-filter dropdown entries. The file must have exactly these columns:

```csv
name,expression
Older drivers,YoungestDriverAge > 40
Heavy vans,"""Gross.Weight"" >= 3000"
```

## Notes

- Prefer Parquet for normal work. It is much faster than CSV in the current DuckDB backend.
- CSV files still work, for example `.venv/bin/lucidum datasets/vans.csv --port 8000`.
- Local data files under `datasets/`, such as `datasets/vans.csv`, `datasets/vans.parquet`, and `datasets/home.parquet`, are ignored by `.gitignore`.
- Saved-filter CSVs under `specs/`, including `specs/filter_spec.csv` and `specs/home_filter_spec.csv`, are tracked.
- The current prototype identifies integer columns separately from continuous numeric columns in the sidebar.
- Initial x-axis and response selections are data-agnostic by default and can be overridden with CLI options or URL parameters.
- Filters use DuckDB expression syntax and are applied before chart aggregation, map aggregation, table rendering, low-weight grouping, response transforms, and sigma calculations.
- Low-weight grouping presets are `0`, `10`, `100`, `0.1%`, and `1%`; they are evaluated against the selected Weight.
- Integer features with a full-data range below 120 start with banding `1`; other integer/numeric features use the automatic standard-deviation based suggestion.
- The `<` and `>` banding controls include practical intermediate values such as `4`, `7`, and `12`.
- Table view is intentionally compact so grouped results can be scanned without excessive row spacing.

## Verification

Run the test suite:

```bash
.venv/bin/python -m unittest discover -s tests
```

Useful code checks before committing:

```bash
node --check src/py_lucidum/static/app.js
git diff --check
```

These launch paths are expected to work from the project root when `datasets/vans.parquet` exists:

```bash
.venv/bin/lucidum datasets/vans.parquet --port 8000
.venv/bin/lucidum datasets/vans.parquet --open --port 8000
.venv/bin/lucidum datasets/vans.parquet --host 0.0.0.0 --port 8000
.venv/bin/lucidum datasets/vans.parquet --no-token
.venv/bin/lucidum datasets/vans.parquet --x YoungestDriverAge --actual AvgPrice1_5 --expected glm_prediction --denominator Gross.Weight
.venv/bin/lucidum datasets/vans.parquet --filters specs/filter_spec.csv
.venv/bin/lucidum datasets/home.parquet --no-filters --port 8000
.venv/bin/lucidum datasets/home.parquet --postcode-area PostcodeArea --postcode-sector PostcodeSector
.venv/bin/lucidum datasets/vans.parquet --tools line-bar
.venv/bin/python -m py_lucidum datasets/vans.parquet --port 8000
```

The Python console and programmatic Uvicorn examples above should also start successfully and serve `GET /api/schema` when opened with the correct token, or without a token when no token is configured.
