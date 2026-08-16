"""Train a GLM outside Lucidum and save standalone model results.

Normally run this script unchanged.  The YAML and formula file control the
analysis. Parts 1-4 are ordinary pandas and glum modelling code; Part 5 saves
the fitted model and predictions for reporting; Part 6 optionally installs
that saved folder in Lucidum.
"""

# %% Imports

import time

import numpy as np
import pandas as pd
from glum import GeneralizedLinearRegressor, TweedieDistribution

from external_model_helpers import (
    config_path_from_command_line,
    formulaic_context,
    load_config,
    read_table,
    require_columns,
    resolve_path,
    strip_formula_comments,
)
from external_model_results import save_glm_model_results


# %% 1. Load settings and data

started = time.perf_counter()

script_file = globals().get("__file__")
config_path = config_path_from_command_line(script_file, "config_glm.yaml")
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


# %% 2. Prepare modelling inputs

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
test_value = str(dataset_settings["test_value"]).strip().lower()
training_scope = str(model_settings["training_scope"])
eligible_fit_mask = (
    scoring_mask
    & model_target.notna()
    & np.isfinite(model_target)
)
if training_scope == "all":
    fit_mask = eligible_fit_mask
elif training_scope == "training_test":
    training_rows = eligible_fit_mask & sample.eq(training_value).fillna(False)
    test_rows = eligible_fit_mask & sample.eq(test_value).fillna(False)
    if not training_rows.any() or not test_rows.any():
        raise ValueError("Training + Test needs usable rows from both configured sample values")
    fit_mask = training_rows | test_rows
else:
    fit_mask = eligible_fit_mask & sample.eq(training_value).fillna(False)

if fit_mask.sum() < 2:
    raise ValueError(f"Need at least two valid rows for training_scope={training_scope!r}")

# Glum keeps its fitted intercept separate from the Formulaic model matrix.
# For the special formula ``1`` that leaves zero predictor columns, which the
# underlying scikit-learn validation rejects.  A private constant column gives
# Glum the one-column matrix it needs while remaining mathematically identical
# to an intercept-only model.  The results writer records this implementation
# detail so Lucidum can recreate the same column when it later scores the model.
configured_fit_intercept = bool(model_settings["fit_intercept"])
intercept_only = formula_rhs.strip() == "1" and configured_fit_intercept
internal_intercept_column = ""
estimator_formula = formula_rhs
estimator_fit_intercept = configured_fit_intercept
if intercept_only:
    internal_intercept_base = "__external_glm_intercept_only"
    internal_intercept_column = internal_intercept_base
    suffix = 2
    while internal_intercept_column in data.columns:
        internal_intercept_column = f"{internal_intercept_base}_{suffix}"
        suffix += 1
    data[internal_intercept_column] = 1.0
    estimator_formula = f"0 + `{internal_intercept_column}`"
    estimator_fit_intercept = False

training_data = data.loc[fit_mask]
training_target = model_target.loc[fit_mask]
training_weights = model_weights.loc[fit_mask] if model_weights is not None else None


# %% 3. Train

alpha = float(regularization["alpha"])
family_name = str(model_settings["family"]).strip().casefold()
family_parameter = model_settings.get("family_parameter")
family = (
    TweedieDistribution(
        power=float(family_parameter) if family_parameter is not None else 1.5
    )
    if family_name == "tweedie"
    else family_name
)

estimator_settings = {
    "formula": estimator_formula,
    "family": family,
    "link": str(model_settings["link"]),
    "fit_intercept": estimator_fit_intercept,
    "alpha": alpha,
    "l1_ratio": float(regularization["l1_ratio"]),
    "scale_predictors": bool(regularization["scale_predictors"]),
    "drop_first": alpha == 0,
    "robust": True,
}
if intercept_only:
    # Treat the internal constant exactly like a normal intercept: it must not
    # be shrunk even if a future intercept-only configuration uses a penalty.
    estimator_settings["P1"] = np.zeros(1)
    estimator_settings["P2"] = np.zeros((1, 1))

model = GeneralizedLinearRegressor(**estimator_settings)

model.fit(
    training_data,
    training_target,
    sample_weight=training_weights,
    store_covariance_matrix=alpha == 0,
    context=formula_functions,
)


# %% 4. Predict and evaluate

# Keeping predictions aligned to the original DataFrame index makes the later
# row matching explicit.  Rows that cannot be scored remain missing.
predictions = pd.Series(np.nan, index=data.index, dtype=float)
predictions.loc[scoring_mask] = model.predict(
    data.loc[scoring_mask],
    context=formula_functions,
)


# %% 5. Calculate and save normal model results

# The standalone folder is authoritative for reporting and later reuse.
result = save_glm_model_results(
    config=config,
    data=data,
    formula_text=formula_text,
    formula_context=formulaic_context(),
    model=model,
    predictions=predictions,
    fit_mask=fit_mask,
    started=started,
    intercept_only=intercept_only,
    internal_intercept_column=internal_intercept_column,
)

print(f"GLM model id: {result['model_id']}")
print(f"Model folder: {result['model_folder']}")


# %% 6. Optionally install the saved model in Lucidum

if bool(config["output"]["install_in_lucidum"]):
    from lucidum_install import install_model_in_lucidum

    lucidum_model_folder = install_model_in_lucidum(
        dataset_path=dataset_path,
        model_folder=result["model_folder"],
        model_type="glm",
        model_id=result["model_id"],
        replace_existing=bool(config["output"]["replace_existing"]),
    )
    print(f"Lucidum model folder: {lucidum_model_folder}")
