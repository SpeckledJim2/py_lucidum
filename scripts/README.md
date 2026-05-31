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
