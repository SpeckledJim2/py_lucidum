# GLM Tabulations

GLM tabulations convert a fitted `glum` model into insurance-style rating tables. The implementation follows the R `GlimmaR` workflow in `mod_GlimmaR_navigate.R`: group formula terms by their feature combination, evaluate table grids in linear predictor space, subtract each table's base-cell contribution, accumulate those subtracted contributions into an adjusted base table, and then score rows by looking up table cells.

## User Workflow

- The GLM tool has a third tab, `Tabulations`, after `Formula builder` and `Model navigator`.
- Tabulations are built on demand for selected model IDs by `POST /api/glm/tabulations/build`; they are not created during GLM training.
- The tab shows a multi-model selector, table selector, table diagnostics, table/plot views, linear/`exp` display scale, colour toggle, and a 2D crosstab selector. Plots title the x-axis with the plotted feature, use fit-aware dense category labels, emphasize raw zero on linear plots and the factor-1 `0%` baseline on `exp` plots, and expose a borderless command that copies the chart as a PNG. A missing interaction crosstab is reported once in the inline notice.
- `glm_tabulated_prediction` appears in `glm:<model_id>:predictions` only after tabulation artifacts exist. It is grouped under `Model predictions` in Line/Bar like `glm_prediction`.
- For a single GLM non-base table, users can rebase a selected table cell to zero linear/one exponential and transfer the scalar into `base`. For a two-feature interaction, either selected feature level can instead be normalised across every OK cell in its slice; the varying offsets move into the opposite one-way table, which is created when absent. Active rules and generated tables are listed below the selectors, and users can clear rules involving the current table or clear all rebasing. The app recalculates `tabulated_predictions.parquet` and rejects any mutation that changes row-level predictions or missingness. Because successful rebasing preserves those predictions, the existing Mean error and linear SD error remain valid and are preserved.

## Required Training State

New GLM builds persist `estimator.pkl` beside the existing manifest and prediction artifacts. This contains the fitted `glum` estimator, including the fitted Formulaic model spec needed to reconstruct the exact design matrix.

Existing GLM folders without `estimator.pkl` are not tabulatable. They stay usable for coefficients and `glm_prediction`, but the Tabulations tab reports that the model must be rebuilt.

## Offset Handling

GLM formulas support explicit `offset(...)` terms:

```text
actual ~ DRIVER_AGE + C(LICENCE_TYPE) + offset(log(EXPOSURE))
```

Validation strips each `offset(...)` call from the formula passed to `glum`, stores the expressions in `manifest.json`, evaluates them with the same safe formula context used by training, and passes the summed offset array to `fit()`, `predict()`, and tabulation linear-predictor reconstruction.

The existing sidebar Weight semantics are unchanged. If Weight is selected, GLM still fits `Actual / Weight` with `sample_weight=Weight`, writes `glm_prediction` on the original Actual scale, and applies the same scale when writing `glm_tabulated_prediction`.

## Algorithm

For each selected model:

1. Load `estimator.pkl`, `manifest.json`, source dataset rows, feature spec metadata, and the fitted Formulaic model spec.
2. Sweep the fitted model spec terms and group term coefficient columns by the sorted dataset features used by each term. For example, `age`, `C(licence)`, and `age:C(licence)` produce separate `age`, `licence`, and `age|licence` tables. Non-base tables are emitted in the order their feature combinations first appear in the saved GLM formula; repeated combinations keep their first position.
3. Group stored offset expressions by the dataset columns they reference. Constant offsets are accumulated into the base table.
4. Build a dummy base row from the feature spec `Base` values where present, otherwise from observed data.
5. For every feature in every table, create levels:
   - Numeric features use feature spec `min`, `max`, and `banding`.
   - Missing or blank numeric metadata is estimated from scored rows and recorded in warnings.
   - Categorical features use all dataset levels.
   - A feature with missing scored values gets an explicit missing level. The fitted formula scores that cell, so transforms such as `ifelse(np.isnan(feature), ..., feature)` retain a finite missing-value contribution.
   - Categorical levels not seen in training are retained as `NA` cells and recorded in warnings.
6. Skip any table whose cartesian grid exceeds 100,000 cells. The manifest records the skipped table and warning.
7. For each table grid, copy the dummy row, vary only that table's features, evaluate the fitted model matrix, multiply the table term columns by coefficients, and add offset contributions in linear predictor space.
8. Evaluate the same table contribution at the table's base cell. Subtract that base-cell contribution from every table cell, and add it to the cumulative base adjustment.
9. Write a `base` table with `intercept + cumulative_base_adjustment`.
10. Score every dataset row by starting with the adjusted base value, banding numeric values to table keys, looking up each table contribution, and summing in linear predictor space. Missing feature values use the explicit missing table cell when the fitted formula produced a finite contribution. Rows with an unseen categorical lookup or an unscorable missing cell get a missing flag and no tabulated prediction.
11. Transform the tabulated linear predictor through the fitted link inverse, then multiply by the Weight/denominator when one was used for the fitted GLM target.
12. Store the diagnostic `linear_sd_error = sd(exact_linear_prediction - tabulated_linear_prediction)` over finite rows.

The SD error is expected to be nonzero for numeric tables because row scoring uses table banding.

## Rebasing

Tabulation tables can have a free additive allocation between visible components. Rebasing is an app-level gauge transform on the linear-predictor scale:

1. Preserve the first generated tables and manifest under `tabulations_raw/`.
2. In `cell_to_base` mode, subtract the selected cell's value from every OK source-table cell and add it to `base`.
3. In two-dimensional `feature_level_to_one_way` mode, read the selected anchor-feature slice. For every OK opposite-feature level, subtract that slice value from every OK interaction cell at the same opposite-feature level and add it to the opposite feature's one-way row. Leave NA cells unchanged and create a zero-valued receiving table when necessary.
4. Rebuild `tabulated_predictions.parquet`; row IDs and missing flags must match exactly and finite linear predictions must remain within `1e-7`.
5. Store the ordered version-2 active rules and generated-table metadata under `tabulations/tabulation_manifest.json` `rebasing`.

Table-scoped clear removes rules whose source or target is the selected table, cascades to rules depending on removed generated tables, restores `tabulations_raw/`, and replays retained rules in order. Clear-all restores the raw tabulations directly. Both operations remove unused generated tables, and any failed mutation restores the complete pre-operation artifact state. Version-1 scalar-slice rules remain replayable for existing rebased models.

## Artifacts

Each GLM model directory under `.lucidum/datasets/<dataset-slug>/<dataset-signature>/models/glm/<model_id>/` may contain:

- `estimator.pkl`: fitted `glum` estimator.
- `tabulations/tabulation_manifest.json`: tables, one-based user-facing table indexes, warnings, diagnostics, feature metadata, and build time.
- `tabulations/*.parquet`: one Parquet table per tabulation, including `base.parquet`.
- `tabulations_raw/*.parquet` and `tabulations_raw/tabulation_manifest.json`: first generated raw tables and metadata, present only while rebase rules are active.
- `tabulated_predictions.parquet`: row-level tabulated predictions.

`tabulated_predictions.parquet` contains:

- `__lucidum_row_id`
- `glm_tabulated_prediction`
- `glm_tabulated_linear_prediction`
- `glm_tabulation_missing`
- one `tabulated_linear__<table_id>` component column per built non-base table.

## API

- `POST /api/glm/tabulations/build`: starts an async build for `model_ids`.
- `GET /api/glm/tabulations/jobs/{job_id}`: returns job status and progress, including current table and cell count.
- `POST /api/glm/tabulations/config`: returns tabulatable status, union table list, warnings, and diagnostics for selected models.
- `POST /api/glm/tabulations/table`: returns a multi-model wide table payload for the selected table and display scale.
- `POST /api/glm/tabulations/plot`: returns ECharts-ready series for 1D tables and 2D crosstab tables.
- `POST /api/glm/tabulations/export`: exports exactly one selected GLM or GBM model to XLSX using saved tabulation manifests and parquet sidecars only.
- `POST /api/glm/tabulations/rebase`: applies `cell_to_base` or two-dimensional `feature_level_to_one_way` rebasing and recalculates tabulated predictions.
- `POST /api/glm/tabulations/rebase/reset`: clears rebasing involving one table or restores all raw tabulations, then recalculates tabulated predictions.

## XLSX Export

XLSX exports are saved beside the selected model's tabulation sidecars as `<model_id>_tabulations_<scale>.xlsx`, replacing the previous same-scale export. GLM and GBM both store the export manifest at `tabulations/tabulation_manifest.json`, beside the table sidecars and workbook. The workbook contains an `index` sheet followed by numbered sheets matching the manifest table indexes. The index columns are `#`, `Table name`, `Dim`, `Cells`, `Min`, `Max`, and `Span`; the first column links to cell `A1` of each numbered worksheet. Every numbered worksheet links back to `index!A1` from `A1`, writes headers in row 2, and starts data in row 3.

Non-base worksheets are exported in long format with rating factor columns in manifest order followed by `model_output`. The base worksheet uses `table` and `model_output`. `exp` scale exponentiates saved `tabulated_linear` values, and non-ok cells are blank. Skipped tables still get numbered worksheets with a short skipped-table message so workbook numbering stays aligned with the index.

## Source Exposure

`GlmSourceProvider` keeps `glm_prediction` as the primary model prediction column. When `tabulated_predictions.parquet` exists, the provider left joins it to the existing `glm:<model_id>:predictions` source relation on `__lucidum_row_id`, exposing `glm_tabulated_prediction` as an additional numeric column.

After a tabulation build, the frontend reloads schema and clears source-scoped caches so Line/Bar sees the new column.
