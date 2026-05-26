# GBM Tool Spec

The GBM tool should be selected from the sidebar, like the other app tools - pick a suitable sidebar icon.

It should be optional and the libraries it uses should only be loaded if the tool is required.

## Goal

The GBM tool allows the user to build a LightGBM using selected columns from the provided .parquet or .csv dataset.

It creates the model predictions, which can then be viewed in the Line and Bar chart or the Maps, as well as a range of diagnostics that can be viewed in the tool - described below.

The GBM and diagnostics must be saved to disk to make them permanent - you need to recommend the best way to do this.

It should be possible to load up the app, point at a pre-built GBM and diagnostics and review them in the app.

## Tool tabs

The tool screen should have three tabs across the top called

- Features and parameters
- Model navigator
- Tree viewer

The functionality of each tab is described in sections below.

## GBM response and offset are specified in the sidebar

- the response is the column actualNumerator
- the model offset is the denominator (transform to log space if the objective requires it)
- only include rows in the GBM training where the denominator is >0
- equal weights for each row
- ignore the filter when building a GBM (don't show the filter tool in the sidebar when GBM is displayed)

## Features and parameters tab

This is made up of two side by side components:

- left hand side is a grid control of GBM features
- right hand side at the top is a grid control to enter the GBM parameters
- right hand side at the bottom is the model evaluation plot (described below)

- the user selects the required features from a left hand side grid type control (I used rhandsontable for R - pick a suitable fast and open source replacement for this app). This control should show all columns in the dataset in the first column, then a checkbox control in the second column to control whether the feature should be included in the GBM or not.  Use grey colour coding and disable the checkboxes for any column types that can't be used in a LightGBM (e.g. date columns). Colour in amber any "high-cardinality" columns that might not be a good choice to include. Show unreadable/invalid dataset columns as disabled red rows with type `invalid`. Also distinguish numeric and categorical features using colour coding.  Include a column called monotonicity which specifies if the selected feature should be monotonic. Use "Increasing", "1" for increasing, "Decreasing", "-1" for decreasing and blank for no monotonicity. Only numeric features can be monotonic.
- the user selects the LightGBM core parameters and learning control parameters from a top right control.  Show all possible LightGBM parameters in this control, but put the most important ones at the top. This should be scrollable as it's a long list.
- include a button to specify whether SHAP values should be created for 0 rows, 10k rows, 100k rows, or all rows.
- include a green "train GBM" button at the top which starts model training and diagnostic creation
- while training, show live progress with the current tree/iteration and train/test metric values, and update the evaluation chart as new LightGBM evaluation results arrive.
- the user specifys a "sample" column which contains "training", "test" and possibly other levels like "out-of-time". The model is trained on the training rows and early stopping takes place on the test rows. But model predictions are calculated for ALL rows (that pass the weight filtering).
- put in a button to create a sample column if none is present
- if no sample column is present, train the GBM on all rows, but pop up a message to the user
- bottom right I want to see the "evaluation" chart - this is the value of the selected model metric after each round of training. Show this for training and also test rows if present.

Note that LightGBM accepts both objective (used for training) and metric (used for early stopping) - I need to be able to specify both.

## Model navigator tab

This is a table showing every fitted model and it's key parameters and value of the objective/metric on train and test. Clicking on a model in the table makes that the "active model", i.e. the one that is displayed in the Line and Bar charts and tree tool.

## Tree viewer tab

This tab lets the user select a single tree in the model from a searchable list and graphically plots that tree with zoom and colour controls.

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
- evaluation log (used to drive the evaluation chart) - this records the model metric (on train and test) after each round of training
- how long it took to train the model and calculate the SHAP values
- the LightGBM converted to a tabular format (in R this is the output from the lgb.model.dt.tree function)

## Shared Data Outputs

I need the model predictions and SHAP values to be available to the line and bar chart as I want to plot them.

I'd like the tool to treat the predictions "as if" they were just another column in the original parquet, available to be plotted like any other column.

## UI Notes

I want a consistent look and theme to the other tools.
