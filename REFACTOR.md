# REFACTOR.md

## Purpose

This file records the refactor order for `py_lucidum`. Each phase should be done in a fresh context, one step at a time, with tests run for the touched surface before moving on.

## Refactor Order

### 1. Documentation Hygiene

**Status:** Completed on 2026-06-08.

- Fix `AGENTS.md` stale references:
  - Replace `src/py_lucidum/core.py` with `src/py_lucidum/core/`.
  - Add missing JS checks for `glm-tool.js`, `gbm-stacked-shap-tool.js`, and `gbm-stacked-shap-chart.js`.
- Clarify that `DEVELOPMENT.md` is the durable contract document, but add a short navigation/index section.
- Mark `docs/specs/gbm-tool_plan.md` as historical or current-status so future agents do not treat old implementation planning as fresh requirements.

### 2. Test Tiering

**Status:** Completed on 2026-06-08.

- Add documented test tiers for future agents:
  - Syntax checks.
  - Fast non-model backend tests.
  - Model tests.
  - Static frontend contract tests.
  - Browser smoke tests.
  - Full pre-commit checks.
- Keep the full suite; do not delete tests just to reduce count.
- Prefer future behavior tests over exact JS/CSS string-contract tests where practical.

### 3. Repo Hygiene

**Status:** Completed on 2026-06-08.

- Move `local/uk_map/scripts/convert_gpkg_to_geojson.py` into `scripts/` or `scripts/uk_map/`.
- Add `local/` to `.gitignore` after moving the tracked helper.
- Confirm no local datasets, generated previews, `.lucidum/`, caches, or virtualenv artifacts are tracked.

### 4. Browser Dependency Policy

**Status:** Completed on 2026-06-08.

**Decision:** Core ECharts 5.5.1 and Leaflet 1.9.4 are vendored and loaded locally; optional map tile layers remain documented external provider dependencies.

- Decide whether core ECharts and Leaflet should be vendored or documented as CDN dependencies.
- If vendored, lazy-load or locally load them consistently with Tabulator, D3, Ace, and ECharts GL.
- Document external map tile dependency separately from app asset loading.

### 5. Backend Route Thinning

- Split GBM route helper/config logic out of `src/py_lucidum/tools/gbm/routes.py`.
- Keep routes focused on auth, request parsing, error mapping, and dispatch.
- Preserve all existing HTTP API shapes.

### 6. Frontend Shared Utilities

- Extract common GLM/GBM frontend behavior:
  - Polling helpers.
  - Sidebar model chooser rendering.
  - Model navigator actions.
  - Resize helpers.
  - Common empty/loading/status UI patterns.
- Keep helpers small, import-safe, and covered by direct Node tests where possible.

### 7. GBM Frontend Split

- Split `gbm-tool.js` into focused modules:
  - Feature and parameter controls.
  - Training/evaluation chart.
  - Model navigator.
  - Tab orchestration.
- Preserve current UI behavior and browser smoke coverage after each split.

### 8. GLM Frontend Split

- Split `glm-tool.js` into focused modules:
  - Formula builder.
  - Model navigator.
  - Tabulations.
  - Shared model/sidebar behavior.
- Preserve tabulation behavior and existing Playwright smoke coverage.

## Testing Expectations

After each step, run the smallest useful checks for that step. Before considering a phase complete, run:

```bash
.venv/bin/python -m unittest discover -s tests
.venv/bin/python -m compileall src tests
find src/py_lucidum/static -path '*/vendor/*' -prune -o -name '*.js' -print0 | xargs -0 -n1 node --check
.venv/bin/python scripts/run_browser_smoke.py
git diff --check
```

## Defaults

- Keep FastAPI + DuckDB + static ES modules.
- Do not migrate to React/Vite as part of this refactor.
- Do not change public CLI, Python API, HTTP API, or model artifact schemas unless a later phase explicitly plans that change.
- Each phase should leave the repo in a working state.
