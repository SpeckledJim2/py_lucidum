# Launching py_lucidum

This project is currently developed against the local example file `vans.parquet`.
That file is intentionally ignored by git because it is a local development dataset.

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
.venv/bin/lucidum vans.parquet --port 8000
```

The app prints a URL like:

```text
Open http://127.0.0.1:8000/?token=...
```

Open that URL in the browser. Stop the server with `Ctrl+C` in the terminal.
The same printed URL can also be opened in the Positron Viewer pane.

Useful options:

```bash
.venv/bin/lucidum vans.parquet --open --port 8000
.venv/bin/lucidum vans.parquet --host 0.0.0.0 --port 8000
.venv/bin/lucidum vans.parquet --no-token
.venv/bin/lucidum vans.parquet --x YoungestDriverAge --actual AvgPrice1_5 --expected glm_prediction
```

- `--open` asks Python to open the generated URL in your browser.
- `--host 0.0.0.0` is useful for internal server/LAN testing. Keep the generated token enabled unless you have another access control layer.
- `--no-token` is convenient for local-only testing and makes API requests work without the generated query-string token.
- `--x`, `--actual`, and `--expected` set the initial x-axis feature, Actual / line 1 feature, and Expected / line 2 feature.
- Without explicit defaults, the app starts with the first dataset column on the x-axis, the first numeric column as Actual / line 1, and no Expected / line 2.
- URL parameters can also set the same initial selections, for example `http://127.0.0.1:8000/?x=YoungestDriverAge&actual=AvgPrice1_5&expected=glm_prediction`.

## Launch As A Python Module

This is equivalent to the console command:

```bash
.venv/bin/python -m py_lucidum vans.parquet --port 8000
```

## Launch From A Python Console

From a Python shell started in the project root:

```python
import py_lucidum

py_lucidum.serve("vans.parquet", port=8000, open_browser=True)
```

This call starts the Uvicorn server and blocks until the server is stopped.

## Launch With Uvicorn Programmatically

For server-style usage, create the ASGI app and pass it to Uvicorn:

```python
import uvicorn
from py_lucidum.app import create_app

app = create_app("vans.parquet", token="dev-token")
uvicorn.run(app, host="127.0.0.1", port=8000)
```

Then open:

```text
http://127.0.0.1:8000/?token=dev-token
```

Initial selections can be supplied programmatically:

```python
app = create_app(
    "vans.parquet",
    token="dev-token",
    defaults={"x": "YoungestDriverAge", "actual": "AvgPrice1_5", "expected": "glm_prediction"},
)
```

## Notes

- Prefer Parquet for normal work. It is much faster than CSV in the current DuckDB backend.
- CSV files still work, for example `.venv/bin/lucidum vans.csv --port 8000`.
- The local `vans.csv` and `vans.parquet` files are ignored by `.gitignore`.
- The current prototype identifies integer columns separately from continuous numeric columns in the sidebar.
- Initial x-axis and response selections are data-agnostic by default and can be overridden with CLI options or URL parameters.
- Integer features with a full-data range below 120 start with banding `1`; other integer/numeric features use the automatic standard-deviation based suggestion.
- The `<` and `>` banding controls include practical intermediate values such as `4`, `7`, and `12`.
- Table view is intentionally compact so grouped results can be scanned without excessive row spacing.

## Verification

These launch paths are expected to work from the project root when `vans.parquet` exists:

```bash
.venv/bin/lucidum vans.parquet --port 8000
.venv/bin/lucidum vans.parquet --open --port 8000
.venv/bin/lucidum vans.parquet --host 0.0.0.0 --port 8000
.venv/bin/lucidum vans.parquet --no-token
.venv/bin/lucidum vans.parquet --x YoungestDriverAge --actual AvgPrice1_5 --expected glm_prediction
.venv/bin/python -m py_lucidum vans.parquet --port 8000
```

The Python console and programmatic Uvicorn examples above should also start successfully and serve `GET /api/schema` when opened with the correct token, or without a token when no token is configured.
