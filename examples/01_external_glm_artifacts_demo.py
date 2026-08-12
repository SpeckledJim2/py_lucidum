"""Train a GLM outside Lucidum, then save it so Lucidum can display it.

This is the file to read and adapt.  Parts 1-4 are ordinary pandas and glum
modelling code.  Part 5 is the single compatibility handoff to Lucidum.
"""

# %% Imports

import time

import numpy as np
import pandas as pd
from glum import GeneralizedLinearRegressor

from external_model_helpers import (
    config_path_from_command_line,
    formulaic_context,
    load_config,
    read_table,
    require_columns,
    resolve_path,
    strip_formula_comments,
)
from lucidum_export import save_glm_for_lucidum


# %% 1. Load the YAML settings, dataset, and formula

started = time.perf_counter()

config_path = config_path_from_command_line(__file__, "config_glm.yaml")
config = load_config(config_path, "glm")

dataset_settings = config["dataset"]
model_settings = config["model"]
regularization = model_settings["regularization"]

dataset_path = resolve_path(config, dataset_settings["path"])
formula_path = resolve_path(config, model_settings["formula_path"])

data = read_table(dataset_path)
formula_text = formula_path.read_text(encoding="utf-8")
formula_rhs = strip_formula_comments(formula_text)
formula_functions = formulaic_context()

response_name = str(dataset_settings["response_numerator"])
denominator_name = str(dataset_settings.get("denominator") or "").strip()
sample_name = str(dataset_settings["sample_column"])

require_columns(data, [response_name, denominator_name, sample_name])
if not formula_rhs or "~" in formula_rhs:
    raise ValueError("The formula file must contain only the right-hand side of the formula")


# %% 2. Prepare the response, weights, and training mask

response = pd.to_numeric(data[response_name], errors="coerce")

if denominator_name:
    denominator = pd.to_numeric(data[denominator_name], errors="coerce")
    scoring_mask = denominator.notna() & np.isfinite(denominator) & denominator.gt(0)
    model_target = response / denominator
    model_weights = denominator
else:
    denominator = None
    scoring_mask = pd.Series(True, index=data.index)
    model_target = response
    model_weights = None

sample = data[sample_name].astype("string").str.strip().str.lower()
training_value = str(dataset_settings["training_value"]).strip().lower()
training_mask = (
    scoring_mask
    & model_target.notna()
    & np.isfinite(model_target)
    & sample.eq(training_value).fillna(False)
)

if training_mask.sum() < 2:
    raise ValueError(f"Need at least two valid {sample_name}={training_value!r} rows")

training_data = data.loc[training_mask]
training_target = model_target.loc[training_mask]
training_weights = model_weights.loc[training_mask] if model_weights is not None else None


# %% 3. Fit the GLM

alpha = float(regularization["alpha"])

model = GeneralizedLinearRegressor(
    formula=formula_rhs,
    family=str(model_settings["family"]),
    link=str(model_settings["link"]),
    fit_intercept=bool(model_settings["fit_intercept"]),
    alpha=alpha,
    l1_ratio=float(regularization["l1_ratio"]),
    scale_predictors=bool(regularization["scale_predictors"]),
    drop_first=alpha == 0,
    robust=True,
)

model.fit(
    training_data,
    training_target,
    sample_weight=training_weights,
    store_covariance_matrix=alpha == 0,
    context=formula_functions,
)


# %% 4. Predict every eligible row

# Keeping predictions aligned to the original DataFrame index makes the later
# row matching explicit.  Rows that cannot be scored remain missing.
predictions = pd.Series(np.nan, index=data.index, dtype=float)
predictions.loc[scoring_mask] = model.predict(
    data.loc[scoring_mask],
    context=formula_functions,
)


# %% 5. Save the fitted model for Lucidum

# Everything specific to Lucidum's file and installation format is inside this
# one adapter call.  Most users should not need to read lucidum_export.py.
result = save_glm_for_lucidum(
    config=config,
    dataset_path=dataset_path,
    data=data,
    formula_text=formula_text,
    formula_context=formulaic_context(),
    model=model,
    predictions=predictions,
    started=started,
)

print(f"GLM model id: {result['model_id']}")
print(f"Portable copy: {result['portable_dir']}")
if result["sidecar_dir"]:
    print(f"Lucidum sidecar: {result['sidecar_dir']}")
