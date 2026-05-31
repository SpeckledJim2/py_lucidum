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
- `src/py_lucidum/static/app/main.js` is the frontend coordinator only: schema/defaults, KPI controls, saved filters, sidebar/footer layout, tool selection, cross-tool invalidation, reload orchestration, and boot flow.
- Frontend tool ownership is split by file: `column-profile-tool.js`, `line-bar-tool.js`, `uk-map-tool.js`, `gbm-tool.js`, `gbm-shap-tool.js`, `gbm-shap-chart.js`, `gbm-tree-viewer.js`, and `model-tool-shell.js`.
- Import-safe shared frontend helpers live under `src/py_lucidum/static/app/shared/`: API calls, formatters, schema/source helpers, and action timing.
- `src/py_lucidum/static/app.css` is only the stable linked CSS manifest. Actual app styles live under `src/py_lucidum/static/styles/` by shell/tool boundary.
- Vendored browser libraries live under `src/py_lucidum/static/vendor/` and should be lazy-loaded by the tools that need them.

## Read The Smallest Useful Surface

Use this map before opening large files:

- App boot, tool switching, shared sidebar/filter/KPI behavior: read `static/app/main.js`, `static/styles/shell.css`, and `static/styles/controls.css`.
- Shared API, formatting, schema/source helpers, or timing behavior: read the matching file in `static/app/shared/` first; only read tool modules if call-site behavior matters.
- Column Profile frontend: read `static/app/column-profile-tool.js` and `static/styles/column-profile.css`; backend behavior is under `tools/column_profile/`.
- Line/Bar frontend: read `static/app/line-bar-tool.js` and `static/styles/line-bar.css`; backend behavior is under `tools/line_bar/`.
- UK Mapping frontend: read `static/app/uk-map-tool.js` and `static/styles/uk-map.css`; backend behavior and map static assets are under `tools/uk_map/`.
- GBM frontend: read `static/app/gbm-tool.js` and `static/styles/gbm.css`; SHAP and tree UI have their own `gbm-shap-*` and `gbm-tree-viewer.js` files. Backend behavior is under `tools/gbm/`.
- Generic modelling shell: read `static/app/model-tool-shell.js` and `static/styles/model-shell.css`.
- CSS changes: read `static/app.css` only to confirm import order. Put shared primitives in `styles/foundations.css` or `styles/controls.css`; put tool-specific selectors in the tool-owned stylesheet.
- Static asset contract checks and frontend string contracts live in `tests/test_static_assets.py`; browser smoke coverage lives in `tests/test_browser_smoke.py`.

## GBM Notes

- GBM is opt-in via `--tools gbm` and optional Python dependencies from `pip install -e ".[gbm]"`.
- LightGBM, pandas, and numpy must be imported lazily in GBM training paths, not at base app import time.
- On macOS, LightGBM may need Homebrew `libomp`; missing native runtime errors should stay actionable rather than surfacing as server 500s.
- GBM artifacts are sidecars under `.lucidum/models/gbm/` beside the dataset and must not be committed.
- The GBM SHAP tab reads saved SHAP sidecars only. Keep SHAP aggregation in backend GBM modules, keep SHAP UI/chart code separate, return dense grids for 3D surfaces, and lazy-load vendored ECharts GL only for those surface plots.

## Before Committing

Run the standard checks from `DEVELOPMENT.md`. For frontend, app-launch, or GBM UI changes, include the browser smoke tests:

```bash
.venv/bin/python -m unittest discover -s tests
.venv/bin/python -m compileall src tests
node --check src/py_lucidum/static/app.js
node --check src/py_lucidum/static/app/main.js
node --check src/py_lucidum/static/app/column-profile-tool.js
node --check src/py_lucidum/static/app/line-bar-tool.js
node --check src/py_lucidum/static/app/uk-map-tool.js
node --check src/py_lucidum/static/app/shared/api.js
node --check src/py_lucidum/static/app/shared/format.js
node --check src/py_lucidum/static/app/shared/schema.js
node --check src/py_lucidum/static/app/shared/timing.js
node --check src/py_lucidum/static/app/gbm-tool.js
node --check src/py_lucidum/static/app/gbm-shap-tool.js
node --check src/py_lucidum/static/app/gbm-shap-chart.js
node --check src/py_lucidum/static/app/gbm-tree-viewer.js
node --check src/py_lucidum/static/app/model-tool-shell.js
.venv/bin/python scripts/run_browser_smoke.py
git diff --check
```

Keep local datasets, `.lucidum/`, virtualenvs, caches, build artifacts, and generated README previews out of git.
