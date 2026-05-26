# Repository Guidance

This file gives future coding agents the shortest reliable path into `py_lucidum`.

## Start Here

- Read `README.md` for user-facing installation, launch commands, and current tool behavior.
- Read `DEVELOPMENT.md` before non-trivial changes. It is the durable maintainer context for architecture, behavior contracts, test commands, and commit rules.
- GBM product notes live in `docs/specs/gbm-tool.md` and `docs/specs/gbm-tool_plan.md`.

## Shape Of The App

- `py_lucidum` is a local-first FastAPI + DuckDB app with a static ES-module frontend.
- Backend app setup lives under `src/py_lucidum/app/`; core dataset/query helpers live in `src/py_lucidum/core.py`.
- Tool backends live under `src/py_lucidum/tools/`.
- Frontend shell and tool modules live under `src/py_lucidum/static/app/`.
- Vendored browser libraries live under `src/py_lucidum/static/vendor/` and should be lazy-loaded by the tools that need them.

## GBM Notes

- GBM is opt-in via `--tools gbm` and optional Python dependencies from `pip install -e ".[gbm]"`.
- LightGBM, pandas, and numpy must be imported lazily in GBM training paths, not at base app import time.
- On macOS, LightGBM may need Homebrew `libomp`; missing native runtime errors should stay actionable rather than surfacing as server 500s.
- GBM artifacts are sidecars under `.lucidum/models/gbm/` beside the dataset and must not be committed.

## Before Committing

Run the standard checks from `DEVELOPMENT.md`. For frontend, app-launch, or GBM UI changes, include the browser smoke tests:

```bash
.venv/bin/python -m unittest discover -s tests
.venv/bin/python -m compileall src tests
node --check src/py_lucidum/static/app.js
node --check src/py_lucidum/static/app/main.js
node --check src/py_lucidum/static/app/gbm-tool.js
node --check src/py_lucidum/static/app/gbm-tree-viewer.js
node --check src/py_lucidum/static/app/model-tool-shell.js
PY_LUCIDUM_RUN_BROWSER_TESTS=1 .venv/bin/python -m pytest
git diff --check
```

Keep local datasets, `.lucidum/`, virtualenvs, caches, build artifacts, and generated README previews out of git.
