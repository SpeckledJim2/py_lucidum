# Build models and reports outside Lucidum

These examples let a data scientist build GLM and GBM models, score data, and
create HTML reports without using the Lucidum application.

The workflow is controlled by YAML and specification files. You normally run
the six numbered Python scripts unchanged.

## What you edit

For your own analysis, copy and edit these inputs:

- The build YAML, such as `config_glm.yaml` or `config_gbm.yaml`.
- The report YAML, such as `config_glm_report.yaml`.
- The GLM formula text file.
- The Feature Specification CSV used to choose model features and report
  charts.
- The KPI Specification CSV used to format values in summary reports.
- Your source CSV or Parquet data, including the required sample column.

The numbered scripts read those settings and perform the work. They are kept
as clear, linear `# %%` workflows so that an interested user can follow them,
but routine use should not require Python changes.

The helper files are implementation machinery:

- `external_model_helpers.py` loads YAML, resolves paths, and prepares inputs.
- `lucidum_export.py` saves externally fitted models in the format Lucidum can
  read.
- `external_report_helpers.py` prepares settings and labels for reports.

Most users do not need to read or edit those helpers.

## The workflow at a glance

The number at the start of each filename shows when to run it:

### 01 — Build and score

Fit the model, score the source data, and save the results.

- GLM: `01_external_glm_artifacts_demo.py`
- GBM: `01_external_gbm_artifacts_demo.py`

### 02 — Create chart reports

Create Actual-versus-Expected charts and, for GBMs, optional SHAP charts.

- GLM: `02_external_glm_report_demo.py`
- GBM: `02_external_gbm_report_demo.py`

### 03 — Create a model summary

Create the model-summary HTML. The GLM version also builds, scores, and
exports tabulations.

- GLM: `03_external_glm_summary_report_demo.py`
- GBM: `03_external_gbm_summary_report_demo.py`

The `01` scripts use `glum` or LightGBM directly and do not import
`py_lucidum`. The `02` and `03` scripts use Lucidum's chart, tabulation, and
report-writing functions, but they do not launch the application or start a
server.

## Install and run

From a source checkout, install the dependencies used by both model types:

```bash
python -m pip install -e ".[glm,gbm,examples]"
```

Run the GLM workflow in order:

```bash
python examples/01_external_glm_artifacts_demo.py
python examples/02_external_glm_report_demo.py
python examples/03_external_glm_summary_report_demo.py
```

Or run the GBM workflow:

```bash
python examples/01_external_gbm_artifacts_demo.py
python examples/02_external_gbm_report_demo.py
python examples/03_external_gbm_summary_report_demo.py
```

With no argument, each script uses the matching example YAML. To use another
configuration, pass its path:

```bash
python examples/01_external_glm_artifacts_demo.py path/to/my_glm.yaml
```

Paths written inside a YAML file are relative to that YAML file. This means a
configuration continues to work when the command is run from another folder.

## Step 01: build and score a model

The two `01` scripts follow the same familiar sequence:

1. Load the YAML and source data.
2. Prepare the response, features, and sample masks.
3. Fit the model.
4. Predict every eligible row.
5. Save the results so reports and Lucidum can read them.

### GLM settings

The supplied `config_glm.yaml` demonstrates these fields:

- `dataset.path` — source CSV or Parquet file.
- `dataset.response_numerator` — response, or numerator when modelling a
  rate.
- `dataset.denominator` — exposure/weight column, or `null` for an average-row
  model.
- `dataset.sample_column` — column that identifies Training, Test, and
  Validation rows.
- `dataset.training_value` — value used to select fitting rows.
- `model.id` — stable name used to find or replace this model.
- `model.label` — display name used in reports and Lucidum.
- `model.formula_path` — text file containing the right-hand side of the
  formula.
- `model.family` and `model.link` — `glum` family and link.
- `model.fit_intercept` — whether to fit an intercept.
- `model.regularization` — `alpha`, `l1_ratio`, and predictor-scaling
  settings.
- `output.portable_root` — folder for a standalone copy of the saved model.
- `output.install` — also make the model available beside the dataset for
  reports and Lucidum.
- `output.replace_existing` — allow this model ID to replace an earlier build
  with the same ID.

The formula file contains the model expression without `response ~`. It may
include Python-style `#` comments. Comments are removed for fitting, while the
original readable formula is retained with the saved model.

The demo uses a Gamma family with a log link. If a denominator is supplied,
the model fits:

```text
response_numerator / denominator
```

using the denominator as the observation weight. The saved results then
contain both the predicted numerator (`glm_prediction`) and predicted rate
(`glm_prediction_rate`). Without a denominator, `glm_prediction` is the
ordinary row-level prediction.

For an unregularized model, the coefficient output includes standard errors,
test statistics, p-values, and confidence intervals from `glum`. Penalized
models deliberately leave those inference fields blank.

### GBM settings

`config_gbm.yaml` uses the same dataset, model, and output ideas, with these
additional settings:

- `dataset.training_value` — rows used to fit trees.
- `dataset.early_stopping_value` — rows used to choose the best iteration.
- `dataset.holdout_value` — rows reserved for an independent metric.
- `features.spec_path` — Feature Specification CSV.
- `features.scenario_column` — column that selects the GBM features.
- `training.parameters` — values passed to `lightgbm.train`.
- `training.num_boost_round` — maximum number of boosting rounds.
- `training.early_stopping_rounds` — early-stopping patience.
- `training.shap_rows` — number of rows on which to save SHAP values, or
  `all`.

A Feature Specification row is used by the model when the selected scenario
cell contains `feature`, ignoring letter case.

Training and Test are passed to LightGBM. Validation is not used for fitting or
early stopping. After the model has scored all rows, the configured metric is
calculated on Validation from those saved predictions and stored as one result
at the best iteration.

For Poisson, Gamma, and Tweedie objectives with a denominator, the script uses
`log(denominator)` as the LightGBM offset. It saves both the predicted numerator
(`gbm_prediction`) and rate (`gbm_prediction_rate`).

SHAP values are calculated by the `01` GBM script and saved with the model.
Later reports read those values; they do not calculate SHAP again.

## Where the saved models go

Each successful `01` run can create two copies of the same model:

1. **Standalone model copy** — written under `output.portable_root`. This is a
   normal output folder that can be inspected, archived, or copied elsewhere.
   The YAML field is called `portable_root`, but it is simplest to think of it
   as the model-output folder.
2. **Lucidum model copy** — when `output.install: true`, the model is also
   placed in a hidden `.lucidum` folder beside the source dataset. This is the
   copy that the `02` and `03` report functions and the Lucidum application can
   find automatically.

The supplied examples set `install: true`. They also set
`replace_existing: true`, which replaces only a previous GLM or GBM with the
same `model.id`. It does not delete other models or other dataset information.

Lucidum keeps models separate for each exact version of a dataset. If the
source file is rewritten, its size, modification time, row count, or columns
may change. Re-run `01` so the model is saved against that new dataset version.

To inspect the demo models in the application after running `01`:

```bash
lucidum datasets/motor_premiums.parquet --tools line-bar,glm,gbm \
  --features specs/feature_spec.csv
```

Using Lucidum this way is optional; model building and HTML reporting do not
depend on interacting with the application.

## Step 02: create chart reports

The `02` scripts create self-contained HTML reports containing the same ECharts
format used by Lucidum's Line/Bar view. Their data is fixed when the report is
written, so there is no control strip or live recalculation. Hover, legends,
tooltips, and zoom remain interactive.

The report YAML identifies the matching `01` build config. This ensures the
report uses the intended dataset and exact model ID rather than whichever
model happens to be active in Lucidum.

### Report settings

`config_glm_report.yaml` and `config_gbm_report.yaml` contain:

- `build_config` — YAML used for the corresponding `01` build.
- `features.spec_path` — Feature Specification CSV.
- `features.scenario_column` — scenario choosing which features become charts.
- `chart.expected` — prediction or other numeric column used as Expected.
- `chart.expected_label` — legend label for Expected.
- `chart.expected_source` — `dataset`, `glm`, or `gbm`.
- `chart_defaults` — defaults used when per-feature chart settings are blank.
- `output.directory` — destination for the HTML report.
- `output.chart_height` — height of each chart in pixels; default `600`.

Each entry under `reports` controls one HTML file:

- `name` and `title` — filename suffix and visible report title.
- `sample_values` — values such as `[validation]`, or `all`.
- `chart_content` — `actual_expected` or `shap_only`.
- `partial_dependence` — `none`, `glm`, or `shap`.
- `transform` — chart transform; `one` rebases an overlay to 1.
- `sigma` — error-bar setting; `0` hides error bars and `2` shows two-sigma
  bars.
- `show_feature_importance` — add rank and importance to chart titles.
- `sort_by_feature_importance` — order charts by whole-model importance.

The supplied reports demonstrate:

- GLM Actual versus Expected on Validation, with two-sigma bars and the GLM
  partial-dependence line.
- GBM Actual versus Expected on Validation, with two-sigma bars.
- GBM SHAP-only charts on all rows, rebased to 1 and with no error bars.

Each report header records the full source-data and model-folder paths, the
response, denominator, Expected column, included sample rows, configurations,
script name, and run time.

### Choosing features and chart settings

The report scenario in the Feature Specification controls which rows become
charts. In the demo, `report_demo` selects every model/report feature except
postcode sector, postcode unit, latitude, longitude, and PREMIUM where those
rows exist.

The following optional columns give a feature its own chart settings:

```text
chart_banding
chart_quantiles
chart_low_weights
chart_missings
chart_labels
chart_sort
chart_transform
chart_sigma
chart_date_bucket
chart_empty_periods
```

If a `chart_*` cell is blank, the report uses the matching value from
`chart_defaults`. `chart_banding` first falls back to the existing `banding`
column, then to YAML. The choices are the same as Lucidum's Line/Bar options,
including fixed-width or quantile bands, missing-value handling, sorting,
labels, date bands, and low-weight tail grouping (`0`, `10`, `100`, `0.1%`, or
`1%`). The Feature Specification `Base` value anchors rebased GLM and SHAP
overlays.

When feature importance is displayed, the percentage and rank describe the
whole fitted model, not only the selected report rows:

- GLM importance is the weighted mean absolute centred feature contribution
  on the linear-predictor scale.
- GBM importance is mean absolute SHAP when saved SHAP values are available;
  otherwise it is LightGBM gain.

A selected feature that is not in the model is labelled `Not in model`.

## Step 03: create model summaries

### GLM summary and tabulations

`03_external_glm_summary_report_demo.py` performs four operations:

1. Build rating tables from the fitted formula and Feature Specification.
2. Score every source row from those persisted rating tables.
3. Export the tables to XLSX.
4. Write a one-page HTML model summary.

`config_glm_summary_report.yaml` supplies the `01` build config, Feature
Specification, KPI Specification, report title/name, and output directory.

For a fitted log link, the XLSX contains exponential-scale relativities and is
named `<model-id>_tabulations_exp.xlsx`. Other links use linear values and the
suffix `_linear.xlsx`.

The HTML contains:

- Source, model, and tabulated-score paths plus other run information.
- Model performance for Training, Test, and Validation. Every family shows
  deviance and deviance explained. Binomial models also show weighted AUC,
  Gini, and log loss; other models show weighted RMSE and MAE.
- The fitted coefficient table, including p-value styling when inference is
  available.
- The tabulation index with table name, dimensions, cell counts, and Min, Max,
  and Span shown to four decimal places. It links to the full XLSX path.

The performance table uses fitted `glm_prediction`. The separately tabulated
score is saved so that it can be compared with the fitted model or used by
Lucidum.

### GBM summary

`03_external_gbm_summary_report_demo.py` reads the saved `01` results and
writes a one-page HTML report containing:

- Source, model, configuration, and run information.
- Performance for Training, Test, and Validation.
- Feature importance for every model feature.
- The LightGBM parameters.
- The saved Training/Test evaluation history and the single Validation marker
  at the best iteration.

Actual and prediction are formatted using the exact matching row in the KPI
Specification. MAPE is displayed as a percentage. Other LightGBM metrics keep
their normal numeric format.

When SHAP was saved, the importance table shows mean absolute SHAP and its
share of total SHAP importance. Otherwise it shows LightGBM gain and its share.

## Important data rules

### Prepare the sample column first

The examples expect the source data already to contain the sample column and
values named in YAML. The scripts do not create a train/test/validation split;
that choice belongs to the modeller.

### Keep source-row order stable

The build scripts assign an internal one-based row number to the complete
source data before selecting samples or removing rows with invalid
denominators. Predictions and SHAP values use that number to join back to the
original data. If you create equivalent files yourself, do not reset or
renumber rows after filtering.

### Denominators must be usable

For a denominator-backed model, eligible rows require a numeric, finite,
positive denominator. Performance summaries compare rates using denominator
weights and show weighted Actual and prediction values.

## Files created by the demo

The reports are written under `local/external_reports/` by default:

```text
motor_premiums_external_glm_validation_actual_vs_expected.html
motor_premiums_external_glm_model_summary.html
motor_premiums_external_gbm_validation_actual_vs_expected.html
motor_premiums_external_gbm_all_rows_rebased_shap.html
motor_premiums_external_gbm_model_summary.html
```

The standalone model-output folder has this structure:

```text
<portable_root>/
├── lucidum_artifacts.json
├── glm/<model-id>/
│   ├── manifest.json
│   ├── formula.txt
│   ├── estimator.pkl
│   ├── coefficients.parquet
│   ├── feature_importance.parquet
│   ├── predictions.parquet
│   └── diagnostics.json
└── gbm/<model-id>/
    ├── manifest.json
    ├── parameters.json
    ├── features.json
    ├── feature_config.parquet
    ├── model.txt
    ├── predictions.parquet
    ├── evaluation.parquet
    ├── tree_table.parquet
    ├── shap_values.parquet
    └── shap_summary.parquet
```

The automatically discoverable Lucidum copy is stored beside the dataset:

```text
<dataset-folder>/.lucidum/datasets/<dataset-name>/<dataset-version>/models/
├── glm/<model-id>/...
└── gbm/<model-id>/...
```

These generated model files, reports, and hidden dataset folders are local
outputs and are ignored by Git.

## Advanced: call the reporting functions directly

The numbered report scripts are usually the simplest interface. If you are
writing a different Python workflow, the same public functions are available:

- `py_lucidum.line_bar_chart(...)` prepares one serializable Lucidum Line/Bar
  chart.
- `py_lucidum.write_echarts_report(...)` combines charts into a self-contained
  HTML report.
- `py_lucidum.report_filename(...)` creates the standard output filename.
- `py_lucidum.gbm_evaluation_chart(...)` prepares the saved GBM evaluation
  chart.
- `py_lucidum.write_gbm_summary_report(...)` writes the GBM summary page.
- `py_lucidum.build_glm_tabulations(...)` builds rating tables and scores the
  source rows.
- `py_lucidum.score_glm_tabulations(...)` scores from already saved rating
  tables without calling the fitted estimator.
- `py_lucidum.export_glm_tabulations(..., scale="auto")` writes the XLSX.
- `py_lucidum.write_glm_summary_report(...)` writes the GLM summary page.

These are supported reporting and GLM-tabulation functions.
`lucidum_export.py` is a helper used by the `01` scripts. It saves the fitted
model and predictions in the form required by the report scripts and,
optionally, by the Lucidum application. Users normally do not call or edit it
directly.
