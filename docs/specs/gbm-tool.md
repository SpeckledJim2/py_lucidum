# GBM Tool Spec

**Status:** Product intent/background. `DEVELOPMENT.md` is the durable current behavior contract for implemented GBM behavior; use this file for product context and confirm details against the current code and maintainer docs before treating them as active requirements.

The GBM tool should be selected from the sidebar, like the other app tools - pick a suitable sidebar icon.

It should be optional and the libraries it uses should only be loaded if the tool is required.

## Goal

The GBM tool allows the user to build a LightGBM using selected columns from the provided .parquet or .csv dataset. Parquet-folder inputs are intentionally unsupported for GBM because prediction, generated-sample, SHAP, and tabulation sidecars are scoped to one source file.

It creates the model predictions, which can then be viewed in the Line and Bar chart or the Maps, as well as a range of diagnostics that can be viewed in the tool - described below.

The GBM and diagnostics must be saved to disk to make them permanent - you need to recommend the best way to do this.

It should be possible to load up the app, point at a pre-built GBM and diagnostics and review them in the app.

## Tool tabs

The tool screen should have five tabs across the top called

- Features and parameters
- Model navigator
- SHAP
- Stacked SHAP
- Tree viewer

The functionality of each tab is described in sections below.

## GBM response and offset are specified in the sidebar

- the response is the column actualNumerator
- the model offset is the denominator (transform to log space if the objective requires it)
- only include rows in the GBM training where the denominator is >0
- equal weights for each row
- ignore the filter when building a GBM (don't show the filter tool in the sidebar when GBM is displayed)
- the first parameter-table row is `init_score`; `none` keeps the denominator-derived offset behavior, while a selected fitted GLM prediction or numeric source column is transformed into LightGBM's linear predictor space and replaces the denominator-derived initial score
- saved `parameters.json` is a LightGBM Python params dict, including objective and metric; Lucidum-only init-score and training-mode metadata is stored in `manifest.json`
- model input feature order lives in `features.json`; fitted display metadata lives in optional `feature_config.parquet`; raw dataset columns exposed by GBM prediction and SHAP sources are derived from the dataset schema, not stored in `manifest.json`

## Features and parameters tab

This is made up of two side by side components:

- left hand side is a grid control of GBM features
- right hand side at the top is a grid control to enter the GBM parameters
- right hand side at the bottom is the model evaluation plot (described below)

- the user selects the required features from a left hand side grid type control (I used rhandsontable for R - pick a suitable fast and open source replacement for this app). This control should show all columns in the dataset in the first column, then a `Grouping` column from an optional Feature Specification, then a checkbox control to control whether the feature should be included in the GBM or not. Use grey colour coding and disable the checkboxes for any column types that can't be used in a LightGBM (e.g. date columns). Colour in amber any "high-cardinality" columns that might not be a good choice to include. Show unreadable/invalid dataset columns as disabled red rows with type `invalid`. Also distinguish numeric and categorical features using colour coding. Include a column called monotonicity which specifies if the selected feature should be monotonic. Use "Increasing", "1" for increasing, "Decreasing", "-1" for decreasing and blank for no monotonicity. Only numeric features can be monotonic.
- if a Feature Specification is loaded, show an interaction-constraint multi-select dropdown followed by a scenario dropdown immediately before `Clear all`; selecting a scenario selects only usable features whose scenario cell contains the word `feature`. Feature Specification `Base` values are metadata for chart rescaling and are not scenario columns.
- the interaction-constraint dropdown lists each nonblank `Grouping` value with the count of currently selected trainable features in that group. Selecting one or more groups passes LightGBM interaction constraints so selected features in each constrained group can only interact with features in the same group, with all other selected features in a remainder constraint. The Grouping cell shows a numbered lock whose subscript is the effective selected group size after main-effect-only exclusions.
- right-clicking a Feature cell offers `Constrain to main effect only (1D)`, `Remove main-effect-only constraint`, and `Add pair interaction (2D)…`. Main-effect-only and pair markers are lock subscripts `1` and `2`. The pair action opens the separate `Interaction pairs` manager with that feature preselected; adding a pair selects both features. The manager has `Add pair interaction` and `Allowed pair interactions (n)` sections; saved pairs render as `FeatureX × FeatureY` with an explicit `Remove` button, while an empty list says `No pair interactions added`. For a paired feature, replace the context-menu add action with one `Remove FeatureX × FeatureY pairwise interaction` action per pair. Main-effect-only/pair and pair/group overlaps are disabled in the UI and rejected by backend validation. Pair mode sends `feature_interaction_pairs`, persists as `mode: "pairs"`, and converts explicit pairs, main-effect-only features, and disjoint groups to LightGBM constraints plus one remainder constraint containing all uncovered selected features. Use this mainly for 3-leaf GBM/EBM workflows; with more leaves, warn that pairs constrain branch co-occurrence but do not guarantee purely 2D terms.
- for active EBM models, the feature metric toggle includes `EBM Gain`, which replaces the feature grid with a tree-feature-combination table showing `Tree features`, `Dim`, `Trees`, `Gain`, and `% Gain` from the saved tree table.
- the user selects the LightGBM core parameters and learning control parameters from a top right control.  Show all possible LightGBM parameters in this control, but put the most important ones at the top. This should be scrollable as it's a long list.
- the Parameters header has a copy command that writes the current scalar values as a pretty JSON object suitable for `lgb.train(params=...)`. Exclude `init_score`, which belongs on `lgb.Dataset(init_score=...)`, and block copying while any parameter contains Lucidum grid-search braces.
- parameter values support grid-search curly braces. `{200, 300, 400}` means train those explicit values, `{0.05, 0.3; 0.05}` means an inclusive numeric range, and `{bagging, goss}` works for `data_sample_strategy`. When a grid is present, show a compact `Grid samples` numeric control below SHAP rows. The backend samples the hypergrid deterministically without materializing all combinations, skips invalid LightGBM parameter combinations, trains the valid sampled combinations sequentially, saves each model, and activates the best completed model.
- `init_score` is not eligible for grid-search braces because it resolves to row-level sidecar data, not a LightGBM scalar parameter.
- include a button to specify whether SHAP values should be created for 0 rows, 10k rows, 100k rows, or all rows.
- if a physical `SAMPLE` column is present with both `training` and `test` rows, include a model-mode radio button below the SHAP row selector with `Normal` and `EBM`.
- include a green "train GBM" button at the top which starts model training and diagnostic creation
- while training, show live progress with the current tree/iteration and train/test metric values, and update the evaluation chart as new LightGBM evaluation results arrive.
- the user specifys a `SAMPLE` column which contains `training`, `test`, and optionally `validation`/other levels. The model is trained on the training rows and early stopping takes place on the test rows. But model predictions are calculated for ALL rows (that pass the weight filtering).
- put in a button to create a generated reusable 60/20/20 `SAMPLE` sidecar if none is present
- if no sample column is present, train the GBM on all rows, but show a message to the user
- bottom right I want to see the "evaluation" chart - this is the value of the selected model metric after each round of training. Show this for training and also test rows if present. While training, keep the x-axis fixed to `num_iterations`; after completion, use the exact number of evaluation-log points and preserve a zoomed y-axis tail view when the initial drop is steep. Include an inline `All` / `Tail` control above the chart; `All` shows the full history and `Tail` focuses the x/y axes on the late training window. Histories longer than 2,000 points are sampled only for browser rendering; the stored evaluation log remains complete.

Note that LightGBM accepts both objective (used for training) and metric (used for early stopping) - I need to be able to specify both.

## EBM mode

EBM means Explainable Boosting Machine. Normal mode trains the existing LightGBM as before. EBM mode starts with 2-leaf trees and uses learning rate `0.3` for that 2-leaf stage. When the selected test metric has not improved for `early_stopping_rounds`, the training callback switches to 3 leaves and restores the configured learning rate. The same stage-local early stopping process repeats for 4, 5, and later leaf counts until the configured `num_leaves` is reached. `num_iterations` is the total cap across all stages, not a per-stage cap.

EBM mode requires the active sample source, either a physical dataset `SAMPLE` column or the generated sidecar split, to have `training` and `test` rows after denominator filtering. Persist each model's training mode in the model metadata so switching saved models updates the radio button.

## Model navigator tab

This is a sortable table showing every fitted model, any feature interaction constraints, and its key parameters and train objective/metric context. The first column shows a green dot for the active model, i.e. the one displayed in the Line and Bar charts and tree tool. Clicking model rows selects them for actions rather than activating them. The tab lets users rename one selected model to a valid folder name, activate one selected model, or delete all selected model folders from the current dataset workspace; deleting the active model selects the newest remaining model when one exists. Active model switching is handled by the sidebar model list or the Model navigator Activate button.

## Artifact scope

GBM artifacts are scoped to the exact dataset version under `.lucidum/datasets/<dataset-slug>/<dataset-signature>/models/gbm/`. The slug is derived from the CSV or Parquet filename, and the signature is derived from file size, modification time, row count, and schema fingerprint. Startup scans only the current signature workspace, so models from other files in the same folder or previous versions of the same file are not published as model-output sources.

## SHAP tab

This tab is available when the active GBM has saved SHAP rows. Bounded training-time SHAP row options save a deterministic random sample from all scored rows using the model seed. It has two feature chooser controls on the left, both restricted to the active model's trained features. Feature 1 defaults to the highest-Gain feature. Feature 2 has a highlighted `None` row by default. Both choosers can sort by `Importance` or `A-Z`. Switching active models preserves the selected SHAP features when they exist in the next model; unavailable Feature 2 selections fall back to `None`, and unavailable Feature 1 selections fall back to the first item in the current Feature 1 sort order.

The chart controls above the plot set Feature 1 banding, tail percent grouping, Feature 2 banding, `Treat as factor` for each feature, and `-` / `0` / `1` rescaling from Feature Specification `Base` metadata. Rescale `1` exponentiates SHAP values first, then scales them to the Base-level response value. One-feature plots render numeric percentile ribbons around the median or factor box plots. Numeric features forced to factor style keep their natural band order; true categorical box plots sort by descending median SHAP. Two-feature plots use `SHAP(feature_1) + SHAP(feature_2)` and render a dense-grid 3D surface for two continuous features, line plots for continuous-by-factor selections, or heatmaps for two factor-style selections.

## Stacked SHAP tab

This tab shows feature contribution stacks by one selected model feature with banding, feature-count, and sorting controls. Stacked SHAP stays on the linear predictor contribution scale and does not provide Base-aware rescaling.

## Tree viewer tab

This tab lets the user select a single tree in the model from a searchable list and graphically plots that tree with zoom, colour, and direction controls. Below Tree gain, the summary reports `Constraint applied: Singleton — Feature only`, `Pairwise — FeatureX × FeatureY`, `Group — Grouping`, or `None`. If several saved definitions govern the tree, it uses `Constraints applied:` and lists all matches separated by semicolons. Pairwise and group definitions apply when any saved member appears because this reports the explicit training allowlist that governed the tree; it does not imply that every permitted member was used. The automatic remainder constraint is not shown. Colour and direction are session-only presentation choices: Divergent and left-to-right are the defaults, with other palettes plus top-to-bottom and top-left-to-bottom-right diagonal layouts also available. Changing either redraws the loaded tree without changing its selected node path.

## Validation

Before training the model, the tool must check

- is the selected objective function consistent with the response (e.g. Gamma can't have non-negative values)
- monotonicity is only possible with some types of objective
- Weight field is valid
- Train/test split is usable

The tool must not change any column formats - it must help the user understand via the validation what columns are present in the parquet and which can be used in a LightGBM.

## Outputs

I need this tool to be persistent. It's OK to hold interim results in RAM if faster, but I need the ability to save everything quickly to disk if needed. It might simply be easiest to always write to disk.

- LightGBM .txt file output
- model predictions as a parquet and a way to attach them back to original .parquet for queries (attach might not be the right word - I need to be able to plot A vs E charts for the model using the Line and Bar tool - so I need the original parquet as it contains the model features and the reponse, and I need the response parquet for the GBM fitted values - but they must tie up)
- SHAP values and a way to attach them back to original .parquet for queries
- `shap_summary.parquet` as one row per trained feature with `feature`, `mean_abs_shap`, `mean_shap`, and `row_count`; model identity comes from the model folder/source ID, not a repeated Parquet column
- when feature interaction constraint groups are selected, grouped SHAP contribution columns in `shap_values.parquet` named `<Grouping>_INTERACTION_GROUP`; main-effect-only and automatic remainder constraints do not create grouped SHAP columns
- aggregated SHAP plot payloads from saved sidecars for the GBM SHAP tab
- evaluation log (used to drive the evaluation chart) - this records the model metric (on train and test) after each round of training
- how long it took to train the model and calculate the SHAP values
- the LightGBM converted to a tabular format (in R this is the output from the lgb.model.dt.tree function)
- saved GBM tabulation sidecars can be exported from the Tabulations panel to `<model_id>_tabulations_<scale>.xlsx` in the model's `tabulations/` folder, with `tabulation_manifest.json` stored beside it, using only existing manifests and parquet sidecars

## Shared Data Outputs

I need the model predictions and SHAP values to be available to the line and bar chart as I want to plot them.

I'd like the tool to treat the predictions "as if" they were just another column in the original parquet, available to be plotted like any other column.

The Line and Bar Actual chooser separates numeric choices into Dataset features, Model predictions, and SHAP values. Model prediction and SHAP options are limited to the currently active GBM model.

When the active GBM has both saved predictions and saved SHAP rows, the Line and Bar tool can overlay `Partial dependence` SHAP ribbons for the selected x-axis feature. The ribbons are grouped with the same Line and Bar controls, use the SHAP screen percentile set, are transformed to response scale from the GBM objective link, and are scaled so the weighted median ribbon mean matches the active GBM fitted values for the current chart slice. Categorical features use ribbons rather than box plots and can be sorted by median SHAP.

## UI Notes

I want a consistent look and theme to the other tools.
