# GBM Tool Implementation Plan

## Summary

Build the GBM tool as an opt-in LightGBM-backed modelling tool with persistent sidecar artifacts, background training, three GBM tabs, and shareable model-output data sources that Line/Bar and UK Map can query like normal dataset columns.

Chosen defaults:

- Store artifacts in `.lucidum/models/gbm/` next to the source dataset.
- Train through a background job with polling.
- Use the sidebar Actual and denominator/KPI controls as the GBM response and offset/exposure selections.
- Keep LightGBM/backend modelling code cleanly separated from frontend UI code.
- Use Tabulator for the editable feature/parameter grids, vendored locally and lazy-loaded only for GBM.

## Key Changes

- Add optional GBM dependencies via a package extra, not base install: `lightgbm`, `pandas`, `numpy`.
- Keep GBM backend code modular:
  - `training.py`: LightGBM fitting, evaluation callbacks, predictions, SHAP, gain, tree extraction.
  - `validation.py`: objective, feature, sample, monotonicity, response, and offset checks.
  - `store.py`: sidecar artifact persistence, loading, listing, activation metadata.
  - `sources.py`: publishing model artifacts through the shared data-source contract.
  - `routes.py`: HTTP request/response handling only.
- Add a GBM model store with one directory per model containing `manifest.json`, `model.txt`, prediction/SHAP/evaluation/tree Parquet artifacts, feature config, parameters, and training log.
- Add GBM routes for config, validation, training, job polling, model listing/loading, and active-model selection.
- Extend the shared data-source contract with:
  - `gbm:<model_id>:predictions`
  - `gbm:<model_id>:shap_long`
  - `gbm:<model_id>:shap_summary`
- Update Line/Bar and UK Map to accept model-output sources consistently.

## GBM Behaviour

- Training uses rows where the selected denominator is positive, equal row weights, the selected sidebar Actual response, and ignores the global filter.
- For log-link objectives such as `poisson`, `gamma`, and `tweedie`, use `log(selected denominator)` as LightGBM `init_score`. If no denominator is selected, treat offset values as 1.
- If a sample column exists, train on `training`, early-stop on `test`, and score all valid rows. If not, train all valid rows, disable early stopping, and show a warning.
- "Create sample column" creates a model-local deterministic 80/20 train/test assignment in the sidecar folder, not in the original dataset.
- Feature validation disables unusable types, flags high-cardinality categoricals, and allows monotonicity only for numeric features and compatible objectives.
- After training, persist LightGBM gain feature importance and use it to refresh the feature grid.

## UI

- Replace the current GBM shell with a real frontend module, for example `static/app/gbm-tool.js`.
- The frontend must not contain LightGBM-specific training logic, artifact layout rules, validation rules, or parameter interpretation beyond rendering controls and sending structured requests.
- Sidebar while GBM is active keeps the KPI/response controls visible, hides filter controls, and includes an active-model selector.
- **Features and parameters** tab:
  - Left feature grid columns: feature name, include checkbox, monotonicity, and `Gain`.
  - Feature type is displayed as muted right-aligned text inside the Feature cell, with categorical counts shown as `categorical (n)`.
  - `Gain` displays `0.000` before any active model exists.
  - After training or active-model switching, the feature grid must mirror the active model's persisted feature config: `Use`, `Monotonicity`, `Gain`, and sort order.
  - Objective and metric parameter rows are dropdowns containing supported single-response LightGBM options; other parameters remain editable text inputs.
  - Right side contains parameter grid, SHAP row-count selector, green Train GBM button, and ECharts evaluation plot.
  - The ECharts evaluation title is a single line containing evaluation metric, test metric, and best iteration with the same font size.
- **Model navigator** tab lists saved models, key parameters, train/test metrics, timings, and active-model selection.
- **Tree viewer** tab provides a tree selector plus graphical ECharts tree from LightGBM dump output.
- Keep the existing dense, utilitarian app visual style.

## Test Plan

- Unit tests:
  - GBM remains opt-in and imports no modelling libraries unless enabled.
  - Missing optional dependencies return an actionable install-extra error.
  - Backend validation catches invalid selected response/denominator columns, invalid objectives, invalid sample splits, unusable features, and invalid monotonicity.
  - Model store creates, lists, loads, activates, and validates sidecar artifacts.
  - Data-source registry exposes GBM prediction and SHAP sources.
  - Line/Bar and UK Map can query model prediction sources.
  - Feature importance gain is persisted, returned in model detail/config responses, defaults to `0.000`, and sorts descending after training.
- Backend modelling tests should target `training.py`, `validation.py`, `store.py`, and `sources.py` directly without browser/UI involvement.
- Frontend/static tests:
  - GBM tabs, hidden filter controls, active-model selector, train button, feature `Gain` column, active-model feature/parameter refresh, and source switching are present.
  - Tabulator assets are lazy-loaded only for GBM.
- Optional integration test behind `PY_LUCIDUM_RUN_GBM_TESTS=1`: train a small LightGBM, write artifacts, reload app, activate model, verify Gain ordering, and plot predictions in Line/Bar.
- Standard checks: `unittest`, `compileall`, JS `node --check`, `git diff --check`, and browser smoke tests.

## Assumptions

- The first implementation targets LightGBM Python 4.x APIs.
- Gain means LightGBM feature importance with `importance_type="gain"`.
- Active-model switching mirrors the persisted model feature config, including selected features, monotonicity, parameters, and Gain.
- Tabulator is selected as the editable grid because it is plain JavaScript, feature-rich, and MIT licensed.
- A copy of this plan should be saved at `docs/specs/gbm-tool_plan.md` before implementation starts.
