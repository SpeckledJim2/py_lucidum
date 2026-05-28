# GBM Tool Implementation Plan

## Summary

Build the GBM tool as an opt-in LightGBM-backed modelling tool with persistent sidecar artifacts, background training, GBM diagnostic tabs including SHAP plotting, and shareable model-output data sources that Line/Bar and UK Map can query like normal dataset columns.

Chosen defaults:

- Store artifacts in `.lucidum/models/gbm/` next to the source dataset.
- Train through a background job with polling.
- Report live job progress through the existing polling endpoint during LightGBM training.
- Use the sidebar Actual and denominator/KPI controls as the GBM response and offset/exposure selections.
- Keep LightGBM/backend modelling code cleanly separated from frontend UI code.
- Use Tabulator for the editable feature/parameter grids, vendored locally and lazy-loaded only for GBM.
- Use backend-normalized tree routes plus a lazy-loaded D3 viewer for GBM tree diagrams.
- Use backend SHAP aggregation routes plus ECharts, with ECharts GL lazy-loaded only for numeric/numeric SHAP surfaces.

## Key Changes

- Add optional GBM dependencies via a package extra, not base install: `lightgbm`, `pandas`, `numpy`.
- Keep GBM backend code modular:
  - `training.py`: LightGBM fitting, evaluation callbacks, predictions, SHAP, gain, tree extraction.
  - `validation.py`: objective, feature, sample, monotonicity, response, and offset checks.
  - `store.py`: sidecar artifact persistence, loading, listing, activation metadata.
  - `sources.py`: publishing model artifacts through the shared data-source contract.
  - `trees.py`: tree summaries and frontend-ready tree detail payloads from saved LightGBM artifacts.
  - `routes.py`: HTTP request/response handling only.
- Add a GBM model store with one directory per model containing `manifest.json`, `model.txt`, prediction/SHAP/evaluation/tree Parquet artifacts, feature config, parameters, and training log.
- Add GBM routes for config, validation, training, job polling, model listing/loading, and active-model selection.
- Job polling responses include transient progress snapshots with current phase, iteration, train/test metric values, and live evaluation history.
- Add GBM tree routes for model tree lists and selected tree detail.
- Add GBM SHAP routes for trained-feature config and compact plot-ready aggregation payloads, including explicit numeric domains and dense numeric/numeric surface grids.
- Extend the shared data-source contract with:
  - `gbm:<model_id>:predictions`
  - `gbm:<model_id>:shap_long`
  - `gbm:<model_id>:shap_summary`
- Update Line/Bar and UK Map to accept model-output sources consistently.

## GBM Behaviour

- Training uses rows where the selected denominator is positive, equal row weights, the selected sidebar Actual response, and ignores the global filter.
- For log-link objectives such as `poisson`, `gamma`, and `tweedie`, use `log(selected denominator)` as LightGBM `init_score`. If no denominator is selected, treat offset values as 1.
- If a canonical `SAMPLE` column exists, train on `training`, early-stop on `test`, score `validation` as holdout diagnostics, and score all valid rows. If not, train all valid rows, disable early stopping, and show a warning.
- "Create sample column" creates a reusable generated deterministic 60/20/20 training/test/validation assignment in `.lucidum/models/gbm/generated_sample.parquet`, not in the original dataset.
- `training_mode` is persisted as `normal` or `ebm`, defaulting old models to `normal`. EBM is shown only for physical dataset `SAMPLE` splits with training and test rows, never for generated sidecars.
- EBM starts with 2-leaf trees at learning rate `0.3`, advances to 3+ leaves via a LightGBM callback after `early_stopping_rounds` test-metric plateaus, restores the configured learning rate after the 2-leaf stage, and treats `num_iterations` as the total cap across all stages.
- Feature validation disables unusable types, shows unreadable dataset columns as invalid disabled rows, flags high-cardinality categoricals, and allows monotonicity only for numeric features and compatible objectives.
- Training reads only selected features plus required response, offset, and sample columns from the raw dataset. Prediction sources join back only readable source columns.
- After training, persist LightGBM gain feature importance and use it to refresh the feature grid.
- During training, a LightGBM callback updates the in-memory job with current iteration and metric values so the browser can update status text and the evaluation chart before the model is persisted.
- SHAP row options are `0`, `10k`, `100k`, and `all`. SHAP values are stored as a wide artifact keyed by `__lucidum_row_id`, with one numeric column per selected feature; the summary artifact remains one row per feature.
- SHAP plots read only saved SHAP rows joined to trained feature values by `__lucidum_row_id`; one-feature plots use that feature's SHAP, while two-feature plots use the sum of the two selected SHAP contributions.

## UI

- Replace the current GBM shell with a real frontend module, for example `static/app/gbm-tool.js`.
- The frontend must not contain LightGBM-specific training logic, artifact layout rules, validation rules, or parameter interpretation beyond rendering controls and sending structured requests.
- Sidebar while GBM is active keeps the KPI/response controls visible, hides filter controls, and includes an active-model selector.
- **Features and parameters** tab:
  - Left feature grid columns: feature name, include checkbox, monotonicity, and `Gain`.
  - If a Feature Specification is loaded, show its `Grouping` column between feature name and include checkbox, then show an interaction-constraint multi-select followed by a scenario dropdown immediately before `Clear all`.
  - Selecting a feature scenario selects only usable, non-reserved features whose scenario cell contains `feature`, case-insensitive; manual checkbox edits, `Clear all`, and `Select all` clear the scenario dropdown.
  - Feature type is displayed as muted right-aligned text inside the Feature cell, with categorical counts shown as `categorical (n)`.
  - Feature Specification groupings drive the interaction-constraint multi-select; constrained selected features show a lock marker in the Grouping column.
  - `Gain` displays `0.000` before any active model exists.
  - After training or active-model switching, the feature grid must mirror the active model's persisted feature config: `Use`, `Monotonicity`, `Gain`, and sort order.
  - Objective and metric parameter rows are dropdowns containing supported single-response LightGBM options; other parameters remain editable text inputs.
  - Right side contains parameter grid, SHAP row-count selector, green Train GBM button, and ECharts evaluation plot.
  - When EBM is available, a Normal/EBM radio group sits below SHAP rows and includes a tooltip noting the 2-leaf `0.3` learning rate.
  - During training, a compact toolbar status shows the current iteration and latest metric while the evaluation plot updates from live job progress. The live x-axis stays fixed to `num_iterations`; completed and saved charts use the exact evaluation-log length and keep a tail-focused y-axis zoom when the initial metric drop is steep. The Evaluation Log header has an inline `All` / `Tail` control for full-history or late-window views. Evaluation histories over 2,000 points use render-only downsampling while persisted logs stay complete.
  - The ECharts evaluation title is a single line containing evaluation metric, test metric, and best iteration with the same font size.
- **Model navigator** tab lists saved models, feature interaction constraints, key parameters, train/test metrics, timings, a green-dot active-model indicator, and selectable rows for rename/delete actions. Active-model switching remains in the sidebar model list.
- **Tree viewer** tab provides a searchable tree summary table plus a graphical D3 tree from saved `tree_table.parquet` output. The diagram supports zoom, fit/reset, colour palettes, decoded categorical thresholds, edge labels, and default-branch highlighting.
- **SHAP** tab provides two trained-feature choosers sorted by Importance or A-Z, a highlighted `None` option for Feature 2, banding/tail/factor controls, numeric percentile ribbons, factor box plots, numeric/numeric ECharts GL surfaces, continuous-by-factor line plots, and factor/factor heatmaps.
- Keep the existing dense, utilitarian app visual style.

## Test Plan

- Unit tests:
  - GBM remains opt-in and imports no modelling libraries unless enabled.
  - Missing optional dependencies return an actionable install-extra error.
  - Missing LightGBM native runtime dependencies return an actionable error, including the macOS `brew install libomp` hint for `libomp.dylib`.
  - Live job progress is returned through job polling and remains JSON-safe.
  - Backend validation catches invalid selected response/denominator columns, invalid objectives, invalid sample splits, unusable features, and invalid monotonicity.
  - Model store creates, lists, loads, activates, and validates sidecar artifacts.
  - Data-source registry exposes GBM prediction and SHAP sources.
  - Line/Bar and UK Map can query model prediction sources.
  - Feature importance gain is persisted, returned in model detail/config responses, defaults to `0.000`, and sorts descending after training.
  - Tree summary/detail routes read saved artifacts and do not import LightGBM.
  - SHAP config/plot routes read saved artifacts and do not import LightGBM, pandas, or numpy.
- Backend modelling tests should target `training.py`, `validation.py`, `store.py`, and `sources.py` directly without browser/UI involvement.
- Frontend/static tests:
  - GBM tabs, hidden filter controls, active-model selector, train button, feature `Gain` column, active-model feature/parameter refresh, SHAP controls, tree viewer controls, and source switching are present.
  - Tabulator, D3, and ECharts GL assets are lazy-loaded only for the GBM views that need them.
- Optional integration test behind `PY_LUCIDUM_RUN_GBM_TESTS=1`: train a small LightGBM, write artifacts, reload app, activate model, verify Gain ordering, and plot predictions in Line/Bar.
- Standard checks: `unittest`, `compileall`, JS `node --check`, `git diff --check`, and browser smoke tests.

## Assumptions

- The first implementation targets LightGBM Python 4.x APIs.
- Gain means LightGBM feature importance with `importance_type="gain"`.
- Active-model switching mirrors the persisted model feature config, including selected features, monotonicity, parameters, and Gain.
- Tabulator is selected as the editable grid because it is plain JavaScript, feature-rich, and MIT licensed.
- A copy of this plan should be saved at `docs/specs/gbm-tool_plan.md` before implementation starts.
