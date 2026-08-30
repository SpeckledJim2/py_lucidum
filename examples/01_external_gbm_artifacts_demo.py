"""Train a GBM outside Lucidum and save standalone model results.

Normally run this script unchanged.  The YAML and Feature Specification
control the analysis. Parts 1-4 are ordinary pandas and LightGBM modelling
code; Part 5 saves the fitted model and predictions for reporting; Part 6
optionally copies that saved folder into Lucidum's dataset workspace and
activates it.
"""

# %% Imports

import time

import lightgbm as lgb
import numpy as np
import pandas as pd
import pyarrow as pa

from external_model_helpers import (
    config_path_from_command_line,
    dataset_column_kinds,
    effective_gbm_parameters,
    evaluate_validation_metric,
    gbm_parameter_warnings,
    load_config,
    prepare_feature_data,
    read_table,
    require_columns,
    resolve_path,
)
from external_model_results import save_gbm_model_results


LOG_LINK_OBJECTIVES = {"poisson", "gamma", "tweedie"}


def lightgbm_arrow_table(frame, categorical_names):
    """Use the same Float64/Int32 Arrow matrix representation as Lucidum."""

    categorical_names = set(categorical_names)
    columns = {}
    for name in frame.columns:
        if name in categorical_names:
            codes = frame[name].cat.codes.to_numpy(dtype="int32", copy=False)
            columns[name] = pa.array(codes, mask=codes < 0, type=pa.int32())
        else:
            columns[name] = pa.array(
                frame[name].to_numpy(dtype="float64", copy=False),
                from_pandas=True,
                type=pa.float64(),
            )
    return pa.table(columns)


# %% 1. Load settings and data

started = time.perf_counter()

script_file = globals().get("__file__")
config_path = config_path_from_command_line(script_file, "config_gbm.yaml")
config = load_config(config_path, "gbm")

dataset_settings = config["dataset"]
feature_settings = config["features"]
training_settings = config["training"]

dataset_path = resolve_path(config, dataset_settings["path"])
feature_spec_path = resolve_path(config, feature_settings["spec_path"])

data = read_table(dataset_path)
column_kinds = dataset_column_kinds(dataset_path)

response_name = str(dataset_settings["response_numerator"])
denominator_name = str(dataset_settings.get("denominator") or "").strip()
sample_name = str(dataset_settings["sample_column"])

require_columns(data, [response_name, denominator_name, sample_name])


# %% 2. Prepare modelling inputs

response = pd.to_numeric(data[response_name], errors="coerce")

if denominator_name:
    denominator = pd.to_numeric(data[denominator_name], errors="coerce")
    feature_eligibility = (
        denominator.notna()
        & np.isfinite(denominator)
        & denominator.gt(0)
    )
    scoring_mask = response.notna() & np.isfinite(response) & feature_eligibility
else:
    denominator = None
    feature_eligibility = pd.Series(True, index=data.index)
    scoring_mask = response.notna() & np.isfinite(response)

feature_data, feature_names, categorical_features = prepare_feature_data(
    data,
    feature_spec_path,
    str(feature_settings["scenario_column"]),
    eligible_rows=feature_eligibility,
    column_kinds=column_kinds,
)
if {response_name, denominator_name, sample_name}.intersection(feature_names):
    raise ValueError("Response, denominator, and sample columns cannot also be model features")

sample = data[sample_name].astype("string").str.strip().str.lower()
training_mask = scoring_mask & sample.eq(
    str(dataset_settings["training_value"]).strip().lower()
).fillna(False)
test_mask = scoring_mask & sample.eq(
    str(dataset_settings["early_stopping_value"]).strip().lower()
).fillna(False)
validation_mask = scoring_mask & sample.eq(
    str(dataset_settings["validation_value"]).strip().lower()
).fillna(False)

for sample_label, sample_rows in (
    ("training", training_mask),
    ("test", test_mask),
    ("validation", validation_mask),
):
    if not sample_rows.any():
        raise ValueError(f"The {sample_label} sample has no eligible rows")

parameters = effective_gbm_parameters(training_settings)
objective = str(parameters["objective"]).strip().lower()
use_log_offset = denominator is not None and objective in LOG_LINK_OBJECTIVES
initial_score = (
    np.log(denominator.where(denominator.gt(0)))
    if use_log_offset
    else None
)
build_warnings = gbm_parameter_warnings(parameters)
if denominator is None:
    build_warnings.append(
        "No denominator column is selected; GBM offset values will be treated as 1"
    )
else:
    invalid_denominator_rows = int(
        (~denominator.notna() | ~np.isfinite(denominator) | denominator.le(0)).sum()
    )
    if invalid_denominator_rows:
        build_warnings.append(
            f"{invalid_denominator_rows:,} rows have non-positive or missing "
            "denominator and will be excluded"
        )


# %% 3. Train

training_data = lgb.Dataset(
    lightgbm_arrow_table(
        feature_data.loc[training_mask, feature_names],
        categorical_features,
    ),
    label=response.loc[training_mask],
    categorical_feature=categorical_features,
    init_score=initial_score.loc[training_mask] if initial_score is not None else None,
    free_raw_data=False,
    params=parameters,
)

test_data = lgb.Dataset(
    lightgbm_arrow_table(
        feature_data.loc[test_mask, feature_names],
        categorical_features,
    ),
    label=response.loc[test_mask],
    categorical_feature=categorical_features,
    init_score=initial_score.loc[test_mask] if initial_score is not None else None,
    reference=training_data,
    free_raw_data=False,
    params=parameters,
)

evaluation = {}
callbacks = [lgb.record_evaluation(evaluation), lgb.log_evaluation(period=0)]
early_stopping_rounds = int(training_settings["early_stopping_rounds"])
if early_stopping_rounds:
    callbacks.append(lgb.early_stopping(early_stopping_rounds, verbose=False))

model = lgb.train(
    parameters,
    training_data,
    num_boost_round=int(training_settings["num_boost_round"]),
    valid_sets=[training_data, test_data],
    valid_names=["training", "test"],
    callbacks=callbacks,
)

best_iteration = int(model.best_iteration or model.current_iteration())


# %% 4. Predict and evaluate

# Predictions stay aligned to the original DataFrame index.  For log-link
# objectives, LightGBM returns the model adjustment and the exposure offset is
# added back before converting to the response scale.
predictions = pd.Series(np.nan, index=data.index, dtype=float)
scoring_data = feature_data.loc[scoring_mask, feature_names]
scoring_table = lightgbm_arrow_table(scoring_data, categorical_features)

if initial_score is not None:
    raw_prediction = model.predict(
        scoring_table,
        raw_score=True,
        num_iteration=best_iteration,
    )
    predictions.loc[scoring_mask] = np.exp(
        initial_score.loc[scoring_mask].to_numpy() + np.asarray(raw_prediction)
    )
else:
    predictions.loc[scoring_mask] = model.predict(
        scoring_table,
        num_iteration=best_iteration,
    )

# Use the predictions just made to calculate the configured metric on
# Validation.  This adds one point to the saved Evaluation Log without making
# another prediction or allowing Validation to affect early stopping.
validation_warning = evaluate_validation_metric(
    actual=response.loc[validation_mask],
    prediction=predictions.loc[validation_mask],
    parameters=parameters,
    evaluation=evaluation,
    best_iteration=best_iteration,
)


# %% 5. Calculate and save normal model results

# The standalone folder is authoritative for reporting and later reuse.
result = save_gbm_model_results(
    config=config,
    data=data,
    feature_data=feature_data,
    feature_kinds=column_kinds,
    parameters=parameters,
    model=model,
    evaluation=evaluation,
    predictions=predictions,
    started=started,
    warnings=[*build_warnings, *([validation_warning] if validation_warning else [])],
)

print(f"GBM model id: {result['model_id']}")
print(f"Model folder: {result['model_folder']}")


# %% 6. Optionally publish the saved model to Lucidum's workspace

if bool(config["output"]["install_in_lucidum"]):
    from lucidum_install import install_model_in_lucidum

    lucidum_model_folder = install_model_in_lucidum(
        dataset_path=dataset_path,
        model_folder=result["model_folder"],
        model_type="gbm",
        model_id=result["model_id"],
        replace_existing=bool(config["output"]["replace_existing"]),
    )
    print(f"Lucidum model folder: {lucidum_model_folder}")
