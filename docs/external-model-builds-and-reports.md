# External model builds and HTML reports

These examples provide a YAML-controlled modelling and reporting workflow for
people who do not want to build models inside the Lucidum application.

- The `01` scripts train GLM and GBM models using ordinary Python modelling
  libraries. They do not import `py_lucidum`.
- The export adapter writes and installs Lucidum-compatible model artifacts.
- The `02` scripts use Lucidum as a Python chart library to create static,
  interactive ECharts reports. They do not start the Lucidum app or a server.
- The GBM `03` script creates a single static model-summary page from the same
  externally built artifacts.
- The installed artifacts can also be opened in Lucidum's normal model,
  prediction, tabulation, tree, evaluation, and SHAP views.

The five numbered scripts are the files intended for users to read and adapt.
The helper modules contain routine path handling and compatibility machinery.

## Install and run

From a source checkout, install all example dependencies:

```bash
python -m pip install -e ".[glm,gbm,examples]"
```

Run a build before its corresponding report:

```bash
python examples/01_external_glm_artifacts_demo.py
python examples/02_external_glm_report_demo.py

python examples/01_external_gbm_artifacts_demo.py
python examples/02_external_gbm_report_demo.py
python examples/03_external_gbm_summary_report_demo.py
```

Each script accepts one optional YAML path. With no argument, it uses its
matching config in `examples/`. Relative paths inside a YAML file resolve from
that file's directory, not from the shell's current directory.

To inspect the installed models in Lucidum:

```bash
lucidum datasets/motor_premiums.parquet --tools line-bar,glm,gbm \
  --features specs/feature_spec.csv
```

## The 01 model builds

`01_external_glm_artifacts_demo.py` and
`01_external_gbm_artifacts_demo.py` are linear `# %%` scripts. Their numbered
sections load data, prepare modelling inputs, fit, predict, and make one final
export call. Editors that support Python cells can send the sections to a
console one at a time, or the files can be run normally.

`external_model_helpers.py` contains the shared command-line, YAML, path, and
input-table preparation. `lucidum_export.py` contains the artifact and
installation compatibility code. Most users should not need to modify either
helper.

### GLM YAML

`config_glm.yaml` contains:

- `dataset.path`: one CSV or Parquet file.
- `dataset.response_numerator`: numeric response or numerator column.
- `dataset.denominator`: numeric exposure/weight column, or `null` for
  average-row modelling.
- `dataset.sample_column` and `dataset.training_value`: the physical sample
  column and the value used for fitting.
- `model.id` and `model.label`: artifact folder ID and display label.
- `model.formula_path`: text file containing the Formulaic right-hand side,
  without `response ~`.
- `model.family`, `model.link`, and `model.fit_intercept`: direct `glum`
  settings.
- `model.regularization.alpha`, `l1_ratio`, and `scale_predictors`: manual
  regularization settings.
- `output.portable_root`, `install`, and `replace_existing`: portable artifact
  directory and installation policy.

The formula text may contain Python-style `#` comments. The example removes
comments before fitting but preserves the original commented formula in the
saved artifacts.

With a denominator, the GLM fits `response_numerator / denominator` and uses
the denominator as sample weight. It saves numerator-scale `glm_prediction`
and rate-scale `glm_prediction_rate`. With `denominator: null`, the direct
average-row prediction is saved as `glm_prediction`.

### GBM YAML

`config_gbm.yaml` uses the same dataset identity and output fields, plus:

- `dataset.training_value`, `early_stopping_value`, and `holdout_value`: three
  distinct values from the physical sample column.
- `features.spec_path` and `features.scenario_column`: the Feature
  Specification CSV and scenario used for modelling. A row is included when
  the selected scenario cell contains `feature`, case-insensitively.
- `training.parameters`: parameters passed to `lightgbm.train`, including the
  objective, metric, seed, and other normal LightGBM settings.
- `training.num_boost_round`, `early_stopping_rounds`, and `shap_rows`: maximum
  rounds, stopping patience, and deterministic saved SHAP sample size.
  `shap_rows` may also be `all`.

For Poisson, Gamma, and Tweedie objectives with a denominator, the example
uses `log(denominator)` as LightGBM's initial score and restores that offset
when scoring. It saves numerator-scale `gbm_prediction` and rate-scale
`gbm_prediction_rate`.

### Row identity and scoring

Both builders assign one-based `__lucidum_row_id` across the complete source
file before filtering sample partitions or invalid denominators. Training uses
only the configured rows; scoring uses every eligible row while retaining the
original IDs. Lucidum joins compact prediction and SHAP sidecars back to the
source with this ID, so an external pipeline must not reset or renumber it
after filtering.

### Portable and installed artifacts

Portable output has this shape:

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

When `install: true`, the model is copied to the source dataset's exact
workspace:

```text
<dataset-directory>/.lucidum/datasets/<dataset-slug>/<signature>/models/
├── glm/<model-id>/...
├── glm/active_model.json
├── gbm/<model-id>/...
└── gbm/active_model.json
```

The workspace signature uses file size, nanosecond modification time, row
count, and ordered schema. Rewriting the dataset therefore creates a new
workspace and requires rebuilding or reinstalling the model for that version.

Installation stages a complete model directory, atomically replaces only the
configured model ID, and then activates it. Other model folders and the wider
sidecar are preserved. `replace_existing: false` rejects an existing target;
`install: false` writes only the portable copy. Portable artifacts, installed
sidecars, and generated reports are local ignored files.

Lucidum may later add tabulations beneath the installed model. The external
builder does not fabricate them: successful Lucidum-side tabulation proves
that the externally saved estimator or tree table can be used normally.

## The 02 static HTML reports

The two `02` scripts read the exact dataset and model ID from the matching 01
build YAML. They create Lucidum-format Line/Bar chart specifications and write
self-contained HTML containing fixed chart data, the shared renderer, and
vendored ECharts. There is no control strip or live recalculation, but legends,
tooltips, and zoom remain interactive.

The scripts remain short `# %%` examples: load settings, loop over prepared
feature rows, and write HTML. `external_report_helpers.py` owns routine YAML,
path, exact-model artifact, title, and ordering work.

The supplied configs create:

```text
local/external_reports/
├── motor_premiums_external_glm_validation_actual_vs_expected.html
├── motor_premiums_external_gbm_validation_actual_vs_expected.html
├── motor_premiums_external_gbm_all_rows_rebased_shap.html
└── motor_premiums_external_gbm_model_summary.html
```

Each header puts the full source path and installed model-folder path on their
own lines. Response, Weight, Expected, SAMPLE rows, run time, script, and other
provenance appear in the compact grid below.

### Report YAML

`config_glm_report.yaml` and `config_gbm_report.yaml` contain:

- `build_config`: the matching 01 YAML. The exact model ID, source dataset,
  response, denominator, and sample column are reused.
- `features.spec_path` and `features.scenario_column`: the Feature
  Specification and scenario selecting the report pages.
- `chart.expected`, `expected_label`, and `expected_source`: the Expected
  column, its legend label, and whether it comes from `dataset`, `glm`, or
  `gbm`. The Expected value may be another numeric column rather than a model
  prediction.
- `chart_defaults`: fallbacks used when a Feature Specification `chart_*` cell
  is blank.
- `output.directory`: HTML output directory.
- `output.chart_height`: chart height in pixels; it defaults to `600`.

Each item under `reports` contains:

- `name` and `title`.
- `sample_values`: a list such as `[validation]`, or `all`.
- `chart_content`: `actual_expected` or `shap_only`.
- `partial_dependence`: `none`, `glm`, or `shap`.
- `transform` and `sigma`.
- `show_feature_importance`: optional boolean, default `false`.
- `sort_by_feature_importance`: optional boolean, default `false`.

The two importance options are independent. Sorting can be enabled without
displaying importance in the title, and importance can be displayed while
retaining Feature Specification scenario order. When either option is enabled,
the report header ends with a full-width `Importance measure` line stating the
whole-model metric used by that report.

The supplied GLM report uses validation rows, sigma setting 2, and overlays
GLM partial dependence. The first GBM report uses validation rows and sigma
setting 2 for Actual vs Expected. The second uses all rows, hides Actual,
Expected, Weight, and error bars, and shows saved SHAP ribbons and their median.
Its `transform: one` rebases each feature at the Feature Specification `Base`.
The report reads SHAP artifacts saved by the named 01 build; it does not
recalculate SHAP or follow the active-model marker.

### Feature Specification report controls

The report scenario determines which feature rows become charts. The demo's
`report_demo` scenario selects 16 rows and excludes `POSTCODE_SECTOR`,
`LATITUDE`, and `LONGITUDE`. `POSTCODE_UNIT` and `PREMIUM` are not rows in the
checked-in Feature Specification; leave them unselected if they are added to a
local version.

The optional report columns are:

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

`chart_banding` falls back first to the existing `banding` cell and then to
YAML. Every other blank `chart_*` cell falls back to the matching
`chart_defaults` value. The choices match Lucidum's Line/Bar controls: fixed
band width, quantile count, low-weight tail grouping (`0`, `10`, `100`,
`0.1%`, or `1%`), missing handling, labels, sorting, transform, sigma, date
bucket, and empty-period handling. `Base` anchors GLM and SHAP overlays when
using zero/one transforms. The demo sets `MAKE` and `POSTCODE_AREA` to sort by
the model prediction.

### Feature importance titles and ordering

With `show_feature_importance: true`, model features are titled like:

```text
ANNUAL_MILEAGE (Rank 3, Importance 6.7%)
```

A selected scenario feature that is absent from the model is titled:

```text
MAKE (Not in model)
```

Importance is calculated once for the whole named model, not recalculated for
the report scenario or selected SAMPLE rows. The percentage is the feature's
share of total whole-model importance and is displayed to one decimal place.
Ranks are also model-wide. Model features with zero importance retain a rank
and show `0.0%`; if every model value is zero, all percentages are `0.0%`.

The metrics match Lucidum's Line/Bar importance view:

- GLM uses weighted mean absolute centred feature contribution on the fitted
  linear-predictor scale.
- GBM uses saved mean-absolute SHAP when it is available. Otherwise it uses
  saved LightGBM gain. One metric is chosen for the entire model; the report
  never mixes SHAP and gain feature by feature.

The header describes these as “Weighted mean absolute centred linear-predictor
contribution”, “Mean absolute SHAP”, or “LightGBM gain”, as applicable.

When `sort_by_feature_importance: true`, model features appear in descending
importance order. Ties use case-insensitive feature-name order. Scenario
features absent from the model follow at the end in alphabetical order. If
either importance option is requested but the named model's importance
artifact is missing or empty, the report stops with an instruction to rebuild
that model.

All three supplied reports show importance. The GLM and GBM Actual-vs-Expected
reports retain scenario order; only the GBM rebased-SHAP report sorts by
descending importance.

## The 03 GBM model summary

`03_external_gbm_summary_report_demo.py` is a separate three-section `# %%`
example: load the YAML and saved results, create the Evaluation Log chart, and
write one HTML file. It does not start Lucidum and it always reads the exact
model named by `config_gbm.yaml`, even when a different model is active.

`config_gbm_summary_report.yaml` contains:

- `build_config`: the matching 01 GBM build YAML.
- `kpi_spec`: the KPI Specification used to format Actual and prediction.
- `report.name` and `report.title`: the filename suffix and visible heading.
- `output.directory` and `output.chart_height`: the report folder and
  Evaluation Log height; height defaults to 600 pixels.

Relative paths resolve from the summary YAML. The KPI row must exactly match
the build's `response_numerator` and denominator. The supplied PREMIUM build
therefore displays Actual and prediction as whole-pound currency values. A
missing exact match stops with a clear error rather than silently choosing a
different format.

The Model performance table shows Training, Test, and Validation. It includes
only rows joined to a saved prediction with a finite response and, when a
denominator is configured, a finite positive denominator. Without a
denominator, Actual and prediction are average row values. With one, they are
numerator sums divided by the denominator sum and the table also shows that
sum. Training and Test show the configured LightGBM metric at the saved best
iteration. After the single all-row scoring pass, the build calculates the
same configured metric from the already-created Validation predictions and
saves it as one point at the best iteration. Validation is never passed to
training or early stopping, and it does not create a second prediction pass.
Older models without this saved point continue to show `—`. When the configured
metric is MAPE, performance cells and evaluation-chart values are displayed as
percentages while the saved metric data remains in LightGBM's original decimal
form.

The Feature importance table contains every saved model feature, not just a
report scenario. When saved SHAP values are available, it shows rank, feature,
mean absolute SHAP formatted as a percentage, and share of total whole-model
SHAP importance. Otherwise the complete table shows LightGBM gain and its share
of total gain. Model parameters are read from the LightGBM-compatible
`parameters.json` and retain their exact parameter names.

The Model evaluation chart uses the same ECharts option builder as Lucidum's
Features and Parameters screen. It shows the full saved train/test history and
one Validation marker at the best iteration, together with the metric, test
result, legend, and hover tooltips. The static report intentionally has no
tail-zoom or recalculation controls.

## Direct reporting API

`py_lucidum.line_bar_chart(...)` returns one serializable Lucidum Line/Bar
chart specification. `py_lucidum.write_echarts_report(...)` combines chart
specifications into a self-contained report, and `report_filename(...)`
creates the standard understandable output name. `write_echarts_report` also
accepts `chart_height`, whose default is 600 pixels.

`py_lucidum.gbm_evaluation_chart(...)` returns the saved Evaluation Log chart
specification for one exact named model. `py_lucidum.write_gbm_summary_report`
writes the four-section portable GBM summary page. Both functions read saved
artifacts only and do not require a running Lucidum server.

This is a public reporting boundary, whereas `lucidum_export.py` is currently
example compatibility machinery rather than a public model-writer API. The
chart-spec boundary is intentionally suitable for adding other static chart
types, including dedicated SHAP charts, later.
