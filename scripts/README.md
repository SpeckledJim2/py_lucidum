# Scripts

## Browser Smoke Tests

Use `run_browser_smoke.py` instead of running browser-enabled pytest directly
from a Dropbox CloudStorage checkout:

```bash
.venv/bin/python scripts/run_browser_smoke.py
```

The helper mirrors the checkout into a local cache directory, sets
`PY_LUCIDUM_RUN_BROWSER_TESTS=1`, points `PYTHONPATH` at the mirrored `src/`
tree, and runs pytest with the current virtualenv's Python. Any arguments after
`--` are forwarded to pytest:

```bash
.venv/bin/python scripts/run_browser_smoke.py -- tests/test_browser_smoke.py -q
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
