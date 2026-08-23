# Build models outside Lucidum, then report or view them

These examples provide two complete modelling workflows: one for GLMs and one
for GBMs. Each workflow has three numbered Python scripts. Choose a model type,
edit its YAML and specification files, then run its `01`, `02`, and `03`
scripts unchanged.

An optional `04_external_double_lift_demo.py` script compares two already-built
models. It is deliberately separate from both three-step build workflows: run it
only when you want a Baseline-versus-Challenger report.

Model training happens outside the Lucidum application. The `01` scripts fit
models directly with ordinary `glum` or LightGBM code, score the source data,
and save the fitted results. You do not need to launch Lucidum to build either
model or create its reports.

The `02` and `03` scripts create self-contained HTML reports. They reuse
Lucidum's tested chart, tabulation, and report-writing code, but they do not
start the application or a server. Report data is fixed when the HTML is
created, while hover, tooltips, legends, and chart zoom remain interactive.

The saved model-results folder is the authoritative output. Reports read that
folder directly and do not need a `.lucidum` sidecar. As a separate optional
operation, a saved folder can be copied and activated in Lucidum so the
application can display it without retraining it.

## Choose one three-script workflow

### GLM workflow: three scripts

1. `01_external_glm_artifacts_demo.py` fits the GLM with `glum`, scores every
   eligible source row, and saves the fitted model and predictions.
2. `02_external_glm_report_demo.py` creates a feature-by-feature Validation
   Actual-versus-Expected report. The demo includes two-sigma error bars,
   whole-model feature importance in chart titles, and the fitted GLM
   partial-dependence line.
3. `03_external_glm_summary_report_demo.py` builds and scores GLM tabulations,
   exports them to XLSX, and creates a model summary containing configured
   Training, Test, and Validation performance rows, coefficients and p-values,
   and a tabulation summary with a link to the workbook. A configured population
   without eligible fitted predictions remains visible with zero rows and
   unavailable metrics rather than preventing the report.

### GBM workflow: three scripts

1. `01_external_gbm_artifacts_demo.py` fits the GBM with LightGBM, scores every
   eligible source row, and saves predictions, evaluation history, trees, and
   SHAP values.
2. `02_external_gbm_report_demo.py` creates two feature-by-feature reports: a
   Validation Actual-versus-Expected report with two-sigma error bars, and an
   all-row SHAP-only report whose SHAP ribbons are rebased to 1. The reports
   can show whole-model feature importance in their chart titles.
3. `03_external_gbm_summary_report_demo.py` creates a model summary containing
   Training, Test, and Validation performance, mean absolute SHAP feature
   importance, LightGBM parameters, and the saved evaluation history with its
   Validation marker.

### Optional cross-model workflow

`04_external_double_lift_demo.py` reads `config_double_lift.yaml`, opens the two
exact builds named there, and creates one Double Lift report for each configured
SAMPLE population. It supports GLM/GLM, GBM/GBM, GLM/GBM, and GBM/GLM. The ratio
is always Challenger prediction divided by Baseline prediction. Dataset columns
used as `OTHER` comparisons in the app are not part of this build-to-build report.

## What you edit

For your own analysis, copy and edit these inputs:

- The build YAML, such as `config_glm.yaml` or `config_gbm.yaml`.
- The report YAML, such as `config_glm_report.yaml`.
- For a cross-model comparison, `config_double_lift.yaml` with two exact build
  YAML paths.
- The GLM formula text file.
- The Feature Specification CSV used to choose model features and report
  charts.
- The KPI Specification CSV used to format values in chart and summary reports.
- Your source CSV or Parquet data, including the required sample column.

The numbered scripts read those settings and perform the work. They are kept
as clear, linear `# %%` workflows so that an interested user can follow them,
but routine use means running the three scripts for the chosen model type
without changing their Python code.

The helper files are implementation machinery:

- `external_model_helpers.py` loads YAML, resolves paths, and prepares inputs.
- `external_model_results.py` saves the ordinary fitted model results used by
  the report workflows.
- `lucidum_install.py` optionally copies one saved folder into the matching
  Lucidum dataset workspace and activates that exact model ID.
- `external_report_helpers.py` prepares settings and labels for reports.

Most users do not need to read or edit those helpers.

## Install and run

From a source checkout, install the dependencies used by both model types:

```bash
python -m pip install -e ".[glm,gbm,examples]"
```

### Keep client scripts up to date

An installed Lucidum release includes the seven numbered workflow scripts and their
four Python helpers. After upgrading Lucidum, sync those maintained files into an
existing client workflow directory:

```bash
pipx upgrade py-lucidum
lucidum --sync-examples /path/to/client/examples
```

The command creates the directory when needed and overwrites only those eleven Python
files. Client YAML, formulas, specifications, datasets, unknown Python files, and
other files are left unchanged. Preview the exact create/update/unchanged list
without writing anything by adding `--dry-run`:

```bash
lucidum --sync-examples /path/to/client/examples --dry-run
```

Because client YAML is deliberately preserved, existing `02` report configs can
opt in to KPI formatting after a sync by adding a config-relative path such as
`kpi_spec: ../specs/kpi_spec.csv`. Configs without that entry remain valid and
keep generic numeric chart formatting.

The same sync command works after upgrading Lucidum in a normal virtual environment.

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

After both chosen models have been built, run the optional comparison separately:

```bash
python examples/04_external_double_lift_demo.py path/to/config_double_lift.yaml
```

With no argument, each script uses the matching example YAML. To use another
configuration, pass its path:

```bash
python examples/01_external_glm_artifacts_demo.py path/to/my_glm.yaml
```

Paths written inside a YAML file are relative to that YAML file. This means a
configuration continues to work when the command is run from another folder.

### Run the scripts as code cells in Positron

Each numbered script is also a plain Python code-cell workflow. Open it in
Positron, run the `Imports` cell first, then run the numbered cells in order.
The active cell runs in Positron's shared Python Console, so variables created
by one cell remain available to the cells below it.

Python does not define `__file__` in that interactive Console. The examples
handle this difference explicitly: code-cell execution selects the matching
YAML beside the example helpers, while whole-script execution retains the
optional command-line YAML argument described above.

To use another YAML while working interactively, replace the `config_path`
assignment in the first numbered cell with an explicit path, for example:

```python
from pathlib import Path

config_path = Path("path/to/my_gbm.yaml").expanduser().resolve()
```

The remaining lines and cells are unchanged.

## Optionally view the externally trained models in Lucidum

The supplied build YAML files set `output.install_in_lucidum: true`. After the
normal model results have been saved, this copies and activates the model in a
hidden folder beside the source dataset. Open Lucidum against that same dataset
file and it will find the installed model automatically:

```bash
lucidum datasets/motor_premiums.parquet --tools line-bar,glm,gbm --features specs/feature_spec.csv
```

Lucidum loads the fitted model and its saved results; it does not fit the model
again. The external model behaves like a model built in the application:

- External GLMs appear in model navigation, coefficients, predictions, and
  partial-dependence views. After running the GLM `03` script, their
  tabulations are also available.
- External GBMs appear in model navigation, predictions, evaluation, tree,
  SHAP, and Stacked SHAP views.

Opening the model in Lucidum is optional. The external build and HTML-report
workflows remain complete without it.

## Step 01: build and score a model

The two `01` scripts follow the same six-part sequence:

1. Load the YAML and source data.
2. Prepare the response, features, and sample masks.
3. Train.
4. Predict and evaluate.
5. Calculate and save normal model results.
6. Optionally install the saved model in Lucidum.

Both `01` writers save `gini_tr`, `gini_te`, and `gini_vl` from final predictions.
They use Lucidum's canonical [Normalized Gini definition](user-guide.md#normalized-gini),
including exposure/rate treatment, tie handling, undefined cases, and configured
sample-label mapping. These metrics are diagnostic only and do not affect fitting,
early stopping, or model selection.

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
- `dataset.test_value` — value added to fitting rows for a `training_test`
  scope; it defaults to `test` for older configs.
- `dataset.validation_value` — value used for the Validation row in the GLM
  summary report; it defaults to `validation` for older configs and is never
  included in fitting.
- `model.id` — stable name used to find or replace this model.
- `model.label` — display name used in reports and Lucidum.
- `model.formula_path` — text file containing the right-hand side of the
  formula.
- `model.family` and `model.link` — `glum` family and link.
- `model.family_parameter` — the Tweedie variance power when `family` is
  `tweedie`; it defaults to `1.5` when omitted.
- `model.fit_intercept` — whether to fit an intercept.
- `model.training_scope` — `all`, `training`, or `training_test`; it defaults
  to `training` for older configs.
- `model.regularization` — `alpha`, `l1_ratio`, and predictor-scaling
  settings.
- `output.model_results_root` — root of the authoritative saved model results;
  the exact folder is `<root>/glm/<model.id>` or `<root>/gbm/<model.id>`.
- `output.install_in_lucidum` — optionally copy and activate the saved model in
  the source dataset's Lucidum workspace. Reports do not depend on this copy.
- `output.replace_existing` — allow this model ID to replace an earlier build
  with the same ID.

GLM sample-value matching is trimmed and case-insensitive, and the configured
Training, Test, and Validation values must be distinct. The 02 chart-report YAML
continues to select literal dataset values through `reports[].sample_values`, so
use the configured Validation value there when producing a Validation report.

The formula file contains the model expression without `response ~`. It may
include Python-style `#` comments. Comments are removed for fitting, while the
original readable formula is retained with the saved model.

The demo uses a Tweedie family with variance power `1.2` and a log link. If a
denominator is supplied, the model fits:

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
- `dataset.validation_value` — rows reserved for the independent Validation
  metric.
- `features.spec_path` — Feature Specification CSV.
- `features.scenario_column` — column that selects the GBM features.
- `training.parameters` — values passed to `lightgbm.train`.
- `training.num_boost_round` — maximum number of boosting rounds.
- `training.early_stopping_rounds` — early-stopping patience.
- `training.shap_rows` — number of rows on which to save SHAP values, or
  `all`.

A Feature Specification row is used by the model when the selected scenario
cell contains `feature`, ignoring letter case.

Selected GBM features are then put into a canonical case-insensitive
alphabetical order (with the original name as the tie-breaker) before matrix
construction and artifact writing. Regular Lucidum training uses the same
rule, so Feature grid sorting and Feature Specification row order do not alter
LightGBM feature indexes. The exact fitted order is recorded in
`features.json`; existing models keep their historical saved order.

Both workflows also preserve ascending source-row identity when constructing
the LightGBM matrix. In particular, Lucidum restores `__lucidum_row_id` order
after joining a generated `SAMPLE` sidecar, so using the same assignments as a
physical external `SAMPLE` column produces the same bounded bin-construction
sample as well as the same Training/Test/Validation membership.

Canonical feature order makes `feature_fraction` select from the same feature
index mapping, but a seed alone is not a complete bit-for-bit reproducibility
contract. For reproducible CPU builds, keep the data and LightGBM version
fixed, set `seed` (and any explicitly overridden component seeds), and use
`deterministic: true` together with exactly one of `force_col_wise: true` or
`force_row_wise: true`. A fixed `num_threads` is also useful for keeping the
runtime environment comparable.

Training and Test are passed to LightGBM. Validation is not used for fitting or
early stopping. After the model has scored all rows, the configured metric is
calculated on Validation from those saved predictions and stored as one result
at the best iteration.

The three normalized-Gini suffixes always refer to the configured Training, Test,
and Validation sample roles, not to whether a role was used as a holdout. See the
[canonical definition and undefined cases](user-guide.md#normalized-gini).

For Poisson, Gamma, and Tweedie objectives with a denominator, the script uses
`log(denominator)` as the LightGBM offset. It saves both the predicted numerator
(`gbm_prediction`) and rate (`gbm_prediction_rate`). Rows with a missing,
non-finite, zero, or negative denominator are excluded before fitting and
scoring. Categorical levels are likewise derived only from denominator-eligible
rows, matching Lucidum's in-application builder while retaining the complete
source-row identity.

SHAP values are calculated by the `01` GBM script and saved with the model.
Later reports read those values; they do not calculate SHAP again.

## Where the saved models go

Each successful `01` run always creates one authoritative model folder and may
create one optional application copy:

1. **Model results** — written below `output.model_results_root` as
   `<glm|gbm>/<model.id>`. The `01` script returns and prints this exact
   `model_folder`. The `02` and `03` scripts read it directly, and GLM `03`
   writes its tabulations and workbook back into it.
2. **Optional Lucidum installation** — when
   `output.install_in_lucidum: true`, the saved folder is copied into the
   hidden dataset-version workspace and that exact model ID is activated.

The supplied examples enable installation and set `replace_existing: true`,
which replaces only a previous GLM or GBM with the same `model.id`. It does not
delete other models or other dataset information. Set
`install_in_lucidum: false` to run all three scripts without creating or using
any `.lucidum` folder.

Lucidum keeps models separate for each exact version of a dataset. If the
source file is rewritten, its size, modification time, row count, or columns
may change. Re-run `01` so the model is saved against that new dataset version.

The hidden copy is used only by the application. Model building, HTML
reporting, partial dependence, SHAP, evaluation summaries, and GLM tabulation
all use the authoritative model-results folder.

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
- `kpi_spec` — optional, config-relative KPI Specification CSV used to format
  response values.
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

When `kpi_spec` is supplied, its first exact Actual/Denominator match controls
response-axis ticks, Actual/Expected labels, and tooltips using Lucidum's fixed
decimal, `number`, `currency`, or `percent` formatting. The report fails clearly
if the supplied file is missing, malformed, or has no exact match. Omitting the
setting preserves the historical generic numeric formatting, and rebased-to-one
charts always use uplift percentages.

Each report header records the full source-data and model-folder paths, the
response, denominator, Expected column, included sample rows, configurations,
script name, and run time.

### Choosing features and chart settings

The report scenario in the Feature Specification controls which rows become
charts. In the demo, `report_demo` selects every model/report feature except
MAKE, postcode area, postcode sector, postcode unit, latitude, longitude, and
PREMIUM where those rows exist.

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

`03_external_glm_summary_report_demo.py` performs four normal report operations
and one optional synchronization:

1. Build rating tables from the fitted formula and Feature Specification.
2. Score every source row from those persisted rating tables.
3. Export the tables to XLSX.
4. Write a one-page HTML model summary.
5. When `install_in_lucidum` is true, reinstall the updated model folder so
   the application also sees the new tabulations.

`config_glm_summary_report.yaml` supplies the `01` build config, Feature
Specification, KPI Specification, report title/name, and output directory.

For a fitted log link, the XLSX contains exponential-scale relativities and is
named `<model-id>_tabulations_exp.xlsx`. Other links use linear values and the
suffix `_linear.xlsx`.

The HTML contains:

- Source, model, and tabulated-score paths plus other run information.
- The fitted family and link; Tweedie models also show their variance power.
- Model performance rows for the configured Training, Test, and Validation
  values. Every available population shows deviance, deviance explained, and
  normalized Gini. Binomial models also show weighted AUC and log loss; other
  models show weighted RMSE and MAE. A population with no eligible fitted
  predictions shows zero rows and unavailable metrics; at least one configured
  population must be available.
- The fitted coefficient table, including p-value styling when inference is
  available.
- The tabulated model's Mean error, linear SD error, and number of missing
  tabulated predictions, matching the diagnostics shown in Lucidum's GLM
  Tabulations model table.
- The tabulation index with table name, dimensions, cell counts, and Min, Max,
  and Span shown to four decimal places. It links to the full XLSX path.

The performance table uses fitted `glm_prediction`. Gini values are matched to
the explicit Training, Test, or Validation role rather than the table-row
position, so a missing population cannot shift another population's Gini. The
separately tabulated score is saved so that it can be compared with the fitted
model or used by Lucidum.

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

## Optional Step 04: compare two exact model builds

`04_external_double_lift_demo.py` is run after the two selected `01` builds. It
does not train a model, start Lucidum, inspect `.lucidum`, or follow
`active_model.json`. The comparison config points to each authoritative build
YAML as a complete path, including its directory:

```yaml
baseline:
  model_type: glm
  build_config: ../models/pricing-v12/config.yaml

challenger:
  model_type: glm
  build_config: ../models/pricing-v13/config.yaml

kpi_spec: ../specs/kpi_spec.csv

chart:
  banding: auto
  quantiles: 0
  missings: hide
  labels: none
  sigma: 2

reports:
  - name: training_test_double_lift
    title: Pricing v12 versus Pricing v13 — Training and Test
    sample_values: [training, test]

  - name: validation_double_lift
    title: Pricing v12 versus Pricing v13 — Validation
    sample_values: [validation]

output:
  directory: ../local/external_reports
  chart_height: 600
```

Relative `build_config` paths are resolved from the folder containing
`config_double_lift.yaml`; absolute paths are also accepted. Lucidum never
searches for `config.yaml` by filename. Consequently, two build files with the
same name remain unambiguous when their full paths point to different folders.

Each build YAML then resolves its own exact model folder as:

```text
<output.model_results_root>/<model_type>/<model.id>
```

`model_results_root` is resolved relative to that build YAML. The report header
shows both build-YAML paths and both resulting model folders so the comparison
can be audited directly.

The two builds must identify different models but use the same source dataset,
Numerator, Denominator, and SAMPLE column. Their saved manifests and prediction
files must match the configured families and IDs. Predictions are aligned using
`__lucidum_row_id`; a missing prediction or a missing/zero Baseline produces no
ratio for that row.

Each report entry selects one or more literal SAMPLE values, matched after
trimming and without regard to case. Use `sample_values: all` to select every
value. An empty selection or a value absent from the dataset is an error. The
top of every HTML report prominently identifies the SAMPLE column, selected
values, selected source-row count, and rows available to the chart, in addition
to the two models and `Challenger / Baseline` direction.

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
motor_premiums_external_double_lift_training_test_double_lift.html
motor_premiums_external_double_lift_validation_double_lift.html
```

The authoritative model-results root has this structure:

```text
<model_results_root>/
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

The optional, automatically discoverable Lucidum copy is stored beside the
dataset:

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

- `py_lucidum.line_bar_chart(..., model_folder=..., kpi_spec=...)` prepares one
  serializable Line/Bar chart from an exact saved model folder, with optional
  KPI-spec response formatting.
- `py_lucidum.double_lift_chart(...)` prepares a serializable Line/Bar Double
  Lift chart from two exact GLM/GBM model folders and a literal SAMPLE
  population. It keeps the two prediction sources independent even for models
  of the same family.
- `py_lucidum.write_echarts_report(...)` combines charts into a self-contained
  HTML report.
- `py_lucidum.report_filename(...)` creates the standard output filename.
- `py_lucidum.gbm_evaluation_chart(..., model_folder=...)` prepares the saved
  GBM evaluation chart.
- `py_lucidum.write_gbm_summary_report(...)` writes the GBM summary page.
- `py_lucidum.build_glm_tabulations(..., model_folder=...)` builds rating
  tables and scores the source rows in that folder.
- `py_lucidum.score_glm_tabulations(..., model_folder=...)` scores from already
  saved rating tables without calling the fitted estimator.
- `py_lucidum.export_glm_tabulations(..., model_folder=..., scale="auto")`
  writes the XLSX beside those tables.
- `py_lucidum.write_glm_summary_report(..., model_folder=...)` writes the GLM
  summary page from that exact folder.

Omit `model_folder` to retain the existing dataset-sidecar lookup for backward
compatibility. `external_model_results.py` is the neutral writer used by the
`01` scripts; `lucidum_install.py` is used only for the optional application
installation. Users normally do not call or edit either helper directly.
