"""Train a GBM outside Lucidum, then save it so Lucidum can display it.

This is the file to read and adapt.  Parts 1-4 are ordinary pandas and
LightGBM modelling code.  Part 5 is the single compatibility handoff to
Lucidum.
"""

# %% Imports

import time

import lightgbm as lgb
import numpy as np
import pandas as pd

from external_model_helpers import (
    config_path_from_command_line,
    load_config,
    prepare_feature_data,
    read_table,
    require_columns,
    resolve_path,
)
from lucidum_export import save_gbm_for_lucidum


LOG_LINK_OBJECTIVES = {"poisson", "gamma", "tweedie"}


# %% 1. Load the YAML settings, dataset, and feature scenario

started = time.perf_counter()

config_path = config_path_from_command_line(__file__, "config_gbm.yaml")
config = load_config(config_path, "gbm")

dataset_settings = config["dataset"]
feature_settings = config["features"]
training_settings = config["training"]

dataset_path = resolve_path(config, dataset_settings["path"])
feature_spec_path = resolve_path(config, feature_settings["spec_path"])

data = read_table(dataset_path)
feature_data, feature_names, categorical_features = prepare_feature_data(
    data,
    feature_spec_path,
    str(feature_settings["scenario_column"]),
)

response_name = str(dataset_settings["response_numerator"])
denominator_name = str(dataset_settings.get("denominator") or "").strip()
sample_name = str(dataset_settings["sample_column"])

require_columns(data, [response_name, denominator_name, sample_name])
if {response_name, denominator_name, sample_name}.intersection(feature_names):
    raise ValueError("Response, denominator, and sample columns cannot also be model features")


# %% 2. Prepare the response and sample masks

response = pd.to_numeric(data[response_name], errors="coerce")

if denominator_name:
    denominator = pd.to_numeric(data[denominator_name], errors="coerce")
    scoring_mask = (
        response.notna()
        & np.isfinite(response)
        & denominator.notna()
        & np.isfinite(denominator)
        & denominator.gt(0)
    )
else:
    denominator = None
    scoring_mask = response.notna() & np.isfinite(response)

sample = data[sample_name].astype("string").str.strip().str.lower()
training_mask = scoring_mask & sample.eq(
    str(dataset_settings["training_value"]).strip().lower()
).fillna(False)
test_mask = scoring_mask & sample.eq(
    str(dataset_settings["early_stopping_value"]).strip().lower()
).fillna(False)
holdout_mask = scoring_mask & sample.eq(
    str(dataset_settings["holdout_value"]).strip().lower()
).fillna(False)

for sample_label, sample_rows in (
    ("training", training_mask),
    ("test", test_mask),
    ("holdout", holdout_mask),
):
    if not sample_rows.any():
        raise ValueError(f"The {sample_label} sample has no eligible rows")

parameters = dict(training_settings["parameters"])
objective = str(parameters["objective"]).strip().lower()
use_log_offset = denominator is not None and objective in LOG_LINK_OBJECTIVES
initial_score = np.log(denominator) if use_log_offset else None


# %% 3. Fit the GBM

training_data = lgb.Dataset(
    feature_data.loc[training_mask, feature_names],
    label=response.loc[training_mask],
    categorical_feature=categorical_features,
    init_score=initial_score.loc[training_mask] if initial_score is not None else None,
    free_raw_data=False,
    params=parameters,
)

test_data = lgb.Dataset(
    feature_data.loc[test_mask, feature_names],
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


# %% 4. Predict every eligible row

# Predictions stay aligned to the original DataFrame index.  For log-link
# objectives, LightGBM returns the model adjustment and the exposure offset is
# added back before converting to the response scale.
predictions = pd.Series(np.nan, index=data.index, dtype=float)
scoring_data = feature_data.loc[scoring_mask, feature_names]

if initial_score is not None:
    raw_prediction = model.predict(
        scoring_data,
        raw_score=True,
        num_iteration=best_iteration,
    )
    predictions.loc[scoring_mask] = np.exp(
        initial_score.loc[scoring_mask].to_numpy() + np.asarray(raw_prediction)
    )
else:
    predictions.loc[scoring_mask] = model.predict(
        scoring_data,
        num_iteration=best_iteration,
    )


# %% 5. Save the fitted model for Lucidum

# Everything specific to Lucidum's file and installation format is inside this
# one adapter call.  Most users should not need to read lucidum_export.py.
result = save_gbm_for_lucidum(
    config=config,
    dataset_path=dataset_path,
    data=data,
    feature_data=feature_data,
    model=model,
    evaluation=evaluation,
    predictions=predictions,
    started=started,
)

print(f"GBM model id: {result['model_id']}")
print(f"Portable copy: {result['portable_dir']}")
if result["sidecar_dir"]:
    print(f"Lucidum sidecar: {result['sidecar_dir']}")
