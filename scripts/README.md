# Scripts

## Test Workflow

`run_tests.py` is the canonical test entry point. The normal between-commit
lane dynamically checks Python and non-vendored JavaScript syntax, then runs
every test module except the slower GLM and browser suites:

```bash
.venv/bin/python scripts/run_tests.py dev
```

Use focused aliases or exact unittest targets while editing one area:

```bash
.venv/bin/python scripts/run_tests.py focus line-bar
.venv/bin/python scripts/run_tests.py focus glm gbm
.venv/bin/python scripts/run_tests.py focus tests.test_glm.GlmToolTests.test_glm_formula_drop_first_policy_tracks_regularization
```

Before committing, run the complete gate. Enable the repository hook once per
clone to run it automatically:

```bash
.venv/bin/python scripts/run_tests.py precommit
git config core.hooksPath .githooks
```

The hook defaults to `.venv/bin/python`; set `PY_LUCIDUM_TEST_PYTHON` to an
absolute interpreter path when the test environment lives elsewhere. The
environment-sensitive pipx install check remains an explicit packaging/release
command:

```bash
.venv/bin/python scripts/run_tests.py pipx
```

## Browser Smoke Tests

Use the canonical runner instead of running browser-enabled pytest directly
from a Dropbox CloudStorage checkout:

```bash
.venv/bin/python scripts/run_tests.py browser
```

It delegates to `run_browser_smoke.py`, which mirrors the checkout into a local
cache directory, sets `PY_LUCIDUM_RUN_BROWSER_TESTS=1`, points `PYTHONPATH` at
the mirrored `src/` tree, and runs `tests/test_browser_smoke.py` with the current
virtualenv's Python. Pytest options after `--` are applied to that default target
unless you provide another test path:

```bash
.venv/bin/python scripts/run_tests.py browser -- -q
.venv/bin/python scripts/run_tests.py browser -- --durations=20 -q
.venv/bin/python scripts/run_tests.py browser -- tests/test_browser_smoke.py::BrowserSmokeTests::test_gbm_tool_loads_feature_grid -q
```

## UK Sector Adjacency

Regenerate the postcode-sector shared-edge adjacency sidecar after replacing the bundled sector GeoJSON:

```bash
.venv/bin/python scripts/build_uk_sector_adjacency.py
```

## UK Map GeoPackage Conversion

Convert local UK map GeoPackages into compact GeoJSON before copying the final
assets into the bundled map static directory:

```bash
.venv/bin/python scripts/uk_map/convert_gpkg_to_geojson.py
```

By default, the converter reads `local/uk_map/source/*.gpkg` and writes generated
GeoJSON plus `preview.html` to `local/uk_map/output/`. The `local/` directory is
ignored because those source files and previews are local development artifacts.
