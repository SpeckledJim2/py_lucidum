# Repository Guidance

This file gives future coding agents the shortest reliable path into `py_lucidum`.

## Start Here

- Read `README.md` for user-facing installation, launch commands, and current tool behavior.
- Read `DEVELOPMENT.md` before non-trivial changes. It is the durable maintainer context for architecture, behavior contracts, test commands, and commit rules.
- Read the `Releasing` section in `DEVELOPMENT.md` before packaging, tagging, or publishing a release.
- Current GBM product notes live in `docs/specs/gbm-tool.md`; `docs/specs/gbm-tool_plan.md` is historical implementation context only.

## Environment Notes

- This repo is developed in both Positron and the Codex app. Use the in-app Browser when available. In Positron, use the repository's Playwright smoke tooling, local `curl` checks, or a manually launched live Chrome window. Prefer the workflow that best verifies the behavior under test; do not assume the in-app Browser is unavailable.
- To launch the committed multi-Parquet fixture from a source checkout, run `.venv/bin/lucidum datasets/monthly --port 8000`. Folder inputs are for non-modelling tools only; do not combine this fixture with `glm`, `gbm`, or `--tools all`.

## Shape Of The App

- `py_lucidum` is a local-first FastAPI + DuckDB app with a static ES-module frontend.
- Backend app setup lives under `src/py_lucidum/app/`; core dataset/query helpers live in `src/py_lucidum/core/`.
- Tool backends live under `src/py_lucidum/tools/`.
- Frontend shell and tool modules live under `src/py_lucidum/static/app/`.
- `src/py_lucidum/static/app/main.js` is the frontend coordinator only: schema/defaults, KPI controls, saved filters, sidebar/footer layout, tool selection, cross-tool invalidation, reload orchestration, and boot flow.
- Frontend tool ownership is split by file: `dataset-viewer-tool.js`, `column-profile-tool.js`, `line-bar-tool.js`, `histogram-tool.js`, `uk-map-tool.js`, `glm-tool.js`, `glm-formula-builder.js`, `glm-model-navigator.js`, `glm-tabulations.js`, `gbm-tool.js`, `gbm-feature-parameter-controls.js`, `gbm-evaluation-chart.js`, `gbm-model-navigator.js`, `gbm-tab-orchestration.js`, `gbm-shap-tool.js`, `gbm-shap-chart.js`, `gbm-stacked-shap-tool.js`, `gbm-stacked-shap-chart.js`, `gbm-tree-viewer.js`, and `model-tool-shell.js`.
- Import-safe shared frontend helpers live under `src/py_lucidum/static/app/shared/`: API calls, formatters, schema/source helpers, Tabulator lazy loading, and action timing.
- `src/py_lucidum/static/app.css` is only the stable linked CSS manifest. Actual app styles live under `src/py_lucidum/static/styles/` by shell/tool boundary.
- Vendored browser libraries live under `src/py_lucidum/static/vendor/`. ECharts and Leaflet load from `index.html` because default tools need them at startup; Ace, Tabulator, D3, and ECharts GL are lazy-loaded by the tools/views that need them.

## Read The Smallest Useful Surface

Use this map before opening large files:

- App boot, tool switching, shared sidebar/filter/KPI behavior: read `static/app/main.js`, `static/styles/shell.css`, and `static/styles/controls.css`.
- Shared API, formatting, schema/source helpers, or timing behavior: read the matching file in `static/app/shared/` first; only read tool modules if call-site behavior matters.
- Dataset Viewer frontend: read `static/app/dataset-viewer-tool.js` and `static/styles/dataset-viewer.css`; backend behavior is under `tools/dataset_viewer/`.
- Column Profile frontend: read `static/app/column-profile-tool.js` and `static/styles/column-profile.css`; backend behavior is under `tools/column_profile/`.
- Line/Bar frontend: read `static/app/line-bar-tool.js` and `static/styles/line-bar.css`; backend behavior is under `tools/line_bar/`.
- Histogram frontend: read `static/app/histogram-tool.js` and `static/styles/histogram.css`; backend behavior is under `tools/histogram/`.
- UK Mapping frontend: read `static/app/uk-map-tool.js` and `static/styles/uk-map.css`; backend behavior and map static assets are under `tools/uk_map/`.
- GLM frontend: read `static/app/glm-tool.js`, `static/app/glm-formula-builder.js`, `static/app/glm-model-navigator.js`, `static/app/glm-tabulations.js`, and `static/styles/glm.css`; backend behavior is under `tools/glm/`.
- GBM frontend: read `static/app/gbm-tool.js`, `static/app/gbm-feature-parameter-controls.js`, `static/app/gbm-evaluation-chart.js`, `static/app/gbm-model-navigator.js`, `static/app/gbm-tab-orchestration.js`, and `static/styles/gbm.css`; SHAP, Stacked SHAP, and tree UI have their own `gbm-shap-*`, `gbm-stacked-shap-*`, and `gbm-tree-viewer.js` files. Backend behavior is under `tools/gbm/`.
- Generic modelling shell: read `static/app/model-tool-shell.js` and `static/styles/model-shell.css`.
- CSS changes: read `static/app.css` only to confirm import order. Put shared primitives in `styles/foundations.css` or `styles/controls.css`; put tool-specific selectors in the tool-owned stylesheet.
- Static asset contract checks and frontend string contracts live in `tests/test_static_assets.py`; browser smoke coverage lives in `tests/test_browser_smoke.py`.

## GBM Notes

- GBM is opt-in via `--tools gbm` and optional Python dependencies from `pip install -e ".[gbm]"`.
- LightGBM with its Arrow bridge, Polars/PyArrow, pandas, and numpy must be imported lazily in GBM training paths, not at base app import time. Large GBM matrices flow from DuckDB to Polars to numeric Arrow; pandas remains for compact artifacts, tabulations, and exports.
- On macOS, LightGBM may need Homebrew `libomp`; missing native runtime errors should stay actionable rather than surfacing as server 500s.
- GBM and GLM artifacts are sidecars under `.lucidum/datasets/<dataset-slug>/<dataset-signature>/models/` beside the dataset folder and must not be committed.
- The GBM SHAP tab reads saved SHAP sidecars only. Keep SHAP aggregation in backend GBM modules, keep SHAP UI/chart code separate, return dense grids for 3D surfaces, and lazy-load vendored ECharts GL only for those surface plots.

## Before Committing

Use the tiered runner documented in `DEVELOPMENT.md`. The normal development
loop selects focused tests from staged, unstaged, and untracked changes, with a
safe fallback to the broad lane for shared or unknown files:

```bash
.venv/bin/python scripts/run_tests.py changed
```

The broad development lane runs syntax checks, non-browser modules, fast GLM
contracts, and excludes slow modelling process integration cases:

```bash
.venv/bin/python scripts/run_tests.py dev
```

Use `focus` or `browser` with a specific target when you need to override the
automatic selection. The fast commit gate runs staged and unstaged whitespace
checks, dynamic syntax checks, and the change-aware unit lane:

```bash
.venv/bin/python scripts/run_tests.py precommit
```

The complete push gate runs full unittest discovery and all browser smoke tests
after the same whitespace and syntax checks:

```bash
.venv/bin/python scripts/run_tests.py prepush
```

The versioned `.githooks/pre-commit` and `.githooks/pre-push` hooks run those
commands automatically after one-time setup with
`git config core.hooksPath .githooks`. Do not manually run `prepush`
immediately before a normal `git push`, and do not weaken or bypass either hook
merely to save time.

- Before every commit, unless the user explicitly says not to, run `.venv/bin/python scripts/bump_version.py patch`. Include `pyproject.toml` in the same commit and report the final version number.

Keep local datasets, `.lucidum/`, virtualenvs, caches, build artifacts, and generated README previews out of git.

## Before Releasing

- Release only a clean `main` commit whose hosted `CI` workflow passed. Run the
  release-only pipx/build checks before tagging; the tag workflow repeats the
  artifact and installed-wheel checks.
- Make all release-preparation changes in one commit when a specific next
  version is required. Apply the normal version bump once before that commit,
  then create an annotated `vMAJOR.MINOR.PATCH` tag whose value exactly matches
  `project.version` in `pyproject.toml`.
- Do not create or push a release tag, publish or approve a production
  deployment, move a released tag, or yank a release without explicit user
  authorization. Repository changes alone do not imply permission for those
  external actions.
- PyPI filenames and versions cannot be reused, and published immutable GitHub
  release tags/assets cannot be edited. Fix a bad published release by yanking
  it when appropriate and publishing a higher patch version; never rebuild or
  repoint the old version.
- `py-lucidum==X.Y.Z` pins Lucidum only. It does not lock transitive dependency
  versions; consuming projects should use their own lock or constraints file
  when they need a reproducible complete environment.
