from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from py_lucidum.core import ColumnInfo, Dataset, is_numeric_kind, json_number, quote_ident, sql_literal
from py_lucidum.tools.glm.store import GlmModelStore

from .sample import (
    SAMPLE_COLUMN,
    dataset_sample_column,
    dataset_training_sample_counts,
    generated_sample_is_current,
    generated_training_sample_counts,
)


RESPONSE_COLUMN = "actualNumerator"
OFFSET_COLUMN = "denominator"
INIT_SCORE_PARAMETER = "init_score"
INIT_SCORE_NONE = "none"
DEFAULT_OBJECTIVE = "poisson"
DEFAULT_METRIC = "poisson"
DEFAULT_TRAINING_MODE = "normal"
TRAINING_MODES = ("normal", "ebm")
DATA_SAMPLE_STRATEGIES = ("bagging", "goss")
GBM_OBJECTIVES = (
    "regression",
    "regression_l1",
    "huber",
    "fair",
    "poisson",
    "quantile",
    "mape",
    "gamma",
    "tweedie",
    "binary",
    "cross_entropy",
    "cross_entropy_lambda",
)
GBM_METRICS = (
    "l1",
    "l2",
    "rmse",
    "quantile",
    "mape",
    "huber",
    "fair",
    "poisson",
    "gamma",
    "gamma_deviance",
    "tweedie",
    "auc",
    "average_precision",
    "binary_logloss",
    "binary_error",
    "cross_entropy",
    "cross_entropy_lambda",
    "kullback_leibler",
    "r2",
)
LOG_LINK_OBJECTIVES = {"poisson", "gamma", "tweedie"}
MONOTONE_OBJECTIVES = {"regression", "regression_l1", "huber", "fair", "poisson", "gamma", "tweedie", "binary"}
CROSS_ENTROPY_OBJECTIVES = {"cross_entropy", "cross_entropy_lambda"}
HIGH_CARDINALITY_THRESHOLD = 20
RESERVED_INIT_SCORE_COLUMNS = {"gbm_prediction", "glm_prediction"}


@dataclass(frozen=True)
class ValidationResult:
    ok: bool
    errors: list[str]
    warnings: list[str]

    def as_payload(self) -> dict[str, Any]:
        return {"ok": self.ok, "errors": self.errors, "warnings": self.warnings}


def default_parameters() -> list[dict[str, Any]]:
    return [
        {"name": INIT_SCORE_PARAMETER, "value": INIT_SCORE_NONE, "important": True},
        {"name": "objective", "value": DEFAULT_OBJECTIVE, "important": True},
        {"name": "metric", "value": DEFAULT_METRIC, "important": True},
        {"name": "data_sample_strategy", "value": "bagging", "important": True},
        {"name": "num_iterations", "value": 1000, "important": True},
        {"name": "learning_rate", "value": 0.3, "important": True},
        {"name": "num_leaves", "value": 5, "important": True},
        {"name": "max_depth", "value": -1, "important": True},
        {"name": "min_data_in_leaf", "value": 50, "important": True},
        {"name": "early_stopping_rounds", "value": 50, "important": True},
        {"name": "feature_fraction", "value": 1.0, "important": False},
        {"name": "bagging_fraction", "value": 1.0, "important": False},
        {"name": "bagging_freq", "value": 0, "important": False},
        {"name": "lambda_l1", "value": 0.0, "important": False},
        {"name": "lambda_l2", "value": 0.0, "important": False},
        {"name": "min_gain_to_split", "value": 0.0, "important": False},
        {"name": "max_bin", "value": 255, "important": False},
        {"name": "num_threads", "value": 0, "important": False},
        {"name": "verbosity", "value": -1, "important": False},
        {"name": "seed", "value": 42, "important": False},
    ]


def normalise_parameters(raw: Any) -> dict[str, Any]:
    params = {str(row["name"]): row["value"] for row in default_parameters()}
    if isinstance(raw, dict):
        items = raw.items()
    elif isinstance(raw, list):
        items = (
            (item.get("name"), item.get("value"))
            for item in raw
            if isinstance(item, dict) and item.get("name")
        )
    else:
        items = ()
    for key, value in items:
        name = str(key).strip()
        if not name:
            continue
        params[name] = coerce_parameter(value)
    return params


def coerce_parameter(value: Any) -> Any:
    if isinstance(value, bool) or value is None:
        return value
    if isinstance(value, (int, float)):
        return value
    text = str(value).strip()
    if text == "":
        return ""
    lower = text.lower()
    if lower in {"true", "false"}:
        return lower == "true"
    try:
        if "." not in text and "e" not in lower:
            return int(text)
        return float(text)
    except ValueError:
        return text


def objective(params: dict[str, Any]) -> str:
    return str(params.get("objective") or DEFAULT_OBJECTIVE).strip().lower()


def metric(params: dict[str, Any]) -> str:
    return str(params.get("metric") or DEFAULT_METRIC).strip().lower()


def normalise_training_mode(raw: Any) -> str:
    mode = str(raw or DEFAULT_TRAINING_MODE).strip().lower()
    return mode or DEFAULT_TRAINING_MODE


def parameter_option_errors(params: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    selected_objective = objective(params)
    selected_metric = metric(params)
    selected_data_sample_strategy = str(params.get("data_sample_strategy") or "bagging").strip().lower()
    if selected_objective not in GBM_OBJECTIVES:
        errors.append(f"Choose a valid LightGBM objective: {selected_objective}")
    if selected_metric not in GBM_METRICS:
        errors.append(f"Choose a valid LightGBM metric: {selected_metric}")
    if selected_data_sample_strategy not in DATA_SAMPLE_STRATEGIES:
        errors.append(f"Choose a valid LightGBM data_sample_strategy: {selected_data_sample_strategy}")
    return errors


def parameter_compatibility_messages(params: dict[str, Any], payload: dict[str, Any]) -> tuple[list[str], list[str]]:
    errors: list[str] = []
    warnings: list[str] = []
    errors.extend(parameter_numeric_constraint_errors(params))
    if bool_parameter(params, "force_col_wise") and bool_parameter(params, "force_row_wise"):
        errors.append("force_col_wise and force_row_wise cannot both be true")
    if number_parameter(params, "path_smooth", 0.0) > 0 and integer_parameter(params, "min_data_in_leaf", 20) < 2:
        errors.append("path_smooth greater than 0 requires min_data_in_leaf of at least 2")
    if bool_parameter(params, "is_unbalance") and number_parameter(params, "scale_pos_weight", 1.0) != 1.0:
        errors.append("is_unbalance cannot be used at the same time as scale_pos_weight")
    if bool_parameter(params, "linear_tree") and shap_rows_requested(payload.get("shap_rows")):
        errors.append("linear_tree=true cannot be used when SHAP rows are requested")
    data_sample_strategy = str(params.get("data_sample_strategy") or "bagging").strip().lower()
    if data_sample_strategy == "bagging":
        bagging_freq = integer_parameter(params, "bagging_freq", 0)
        bagging_fraction = number_parameter(params, "bagging_fraction", 1.0)
        if bagging_freq <= 0 or bagging_fraction >= 1.0:
            warnings.append("data_sample_strategy=bagging is only effective when bagging_freq > 0 and bagging_fraction < 1")
    return errors, warnings


def parameter_numeric_constraint_errors(params: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    num_threads = non_negative_integer_parameter(params, "num_threads", 0)
    if num_threads is None:
        errors.append("num_threads must be an integer at least 0")
    int_min = {
        "num_iterations": 0,
        "min_data_in_leaf": 0,
        "early_stopping_rounds": 0,
        "bagging_freq": 0,
        "max_drop": 0,
    }
    int_positive = {
        "num_class",
        "min_data_per_group",
        "max_cat_threshold",
        "max_cat_to_onehot",
        "top_k",
        "metric_freq",
        "multi_error_top_k",
        "num_machines",
        "local_listen_port",
        "time_out",
        "num_gpu",
    }
    for name, minimum in int_min.items():
        if name in params and integer_parameter(params, name, minimum) < minimum:
            errors.append(f"{name} must be at least {minimum}")
    for name in int_positive:
        if name in params and integer_parameter(params, name, 1) <= 0:
            errors.append(f"{name} must be greater than 0")
    num_leaves = integer_parameter(params, "num_leaves", 31)
    if num_leaves <= 1 or num_leaves > 131072:
        errors.append("num_leaves must be greater than 1 and no more than 131072")
    for name in ("learning_rate", "scale_pos_weight", "sigmoid", "alpha", "fair_c", "poisson_max_delta_step"):
        if name in params and number_parameter(params, name, 1.0) <= 0:
            errors.append(f"{name} must be greater than 0")
    for name in (
        "lambda_l1",
        "lambda_l2",
        "min_gain_to_split",
        "min_sum_hessian_in_leaf",
        "path_smooth",
        "monotone_penalty",
        "cegb_tradeoff",
        "cegb_penalty_split",
        "cat_l2",
        "cat_smooth",
        "lambdarank_position_bias_regularization",
    ):
        if name in params and number_parameter(params, name, 0.0) < 0:
            errors.append(f"{name} must be at least 0")
    for name in ("feature_fraction", "bagging_fraction", "pos_bagging_fraction", "neg_bagging_fraction"):
        if name in params:
            value = number_parameter(params, name, 1.0)
            if value <= 0 or value > 1:
                errors.append(f"{name} must be greater than 0 and no more than 1")
    for name in ("top_rate", "other_rate", "drop_rate", "skip_drop", "refit_decay_rate"):
        if name in params:
            value = number_parameter(params, name, 0.0)
            if value < 0 or value > 1:
                errors.append(f"{name} must be between 0 and 1")
    if "top_rate" in params or "other_rate" in params:
        top_rate = number_parameter(params, "top_rate", 0.2)
        other_rate = number_parameter(params, "other_rate", 0.1)
        if top_rate + other_rate > 1:
            errors.append("top_rate + other_rate must be no more than 1")
    if "tweedie_variance_power" in params:
        value = number_parameter(params, "tweedie_variance_power", 1.5)
        if value < 1 or value >= 2:
            errors.append("tweedie_variance_power must be at least 1 and less than 2")
    return errors


def ebm_available(dataset: Dataset, offset_column: str | None = OFFSET_COLUMN, generated_sample_path: Any = None) -> bool:
    columns = dataset.column_map()
    checked_offset = offset_column if offset_column and offset_column in columns and is_numeric_kind(columns[offset_column].kind) else None
    if dataset_sample_column(dataset):
        counts = dataset_training_sample_counts(dataset, checked_offset)
        return counts.get("training", 0) > 0 and counts.get("test", 0) > 0
    if generated_sample_path and generated_sample_is_current(dataset, generated_sample_path):
        counts = generated_training_sample_counts(dataset, generated_sample_path, checked_offset)
        return counts.get("training", 0) > 0 and counts.get("test", 0) > 0
    return False


def _sample_counts_have_training_and_test(counts: dict[str, int]) -> bool:
    return counts.get("training", 0) > 0 and counts.get("test", 0) > 0


def uses_log_offset(params: dict[str, Any]) -> bool:
    return objective(params) in LOG_LINK_OBJECTIVES


def normalise_init_score_value(raw: Any) -> str:
    text = str(raw if raw is not None else INIT_SCORE_NONE).strip()
    if text == "" or text.lower() in {INIT_SCORE_NONE, "__none__"}:
        return INIT_SCORE_NONE
    return text


def init_score_value(params: dict[str, Any]) -> str:
    return normalise_init_score_value(params.get(INIT_SCORE_PARAMETER))


def init_score_requested(params: dict[str, Any]) -> bool:
    return init_score_value(params).lower() != INIT_SCORE_NONE


def init_score_transform(selected_objective: str) -> str:
    if selected_objective in LOG_LINK_OBJECTIVES:
        return "log"
    if selected_objective == "binary" or selected_objective in CROSS_ENTROPY_OBJECTIVES:
        return "logit"
    return "identity"


def init_score_domain_invalid_sql(value_sql: str, transform: str) -> str:
    base = f"({value_sql} IS NULL OR NOT isfinite({value_sql})"
    if transform == "log":
        return f"{base} OR {value_sql} <= 0)"
    if transform == "logit":
        return f"{base} OR {value_sql} <= 0 OR {value_sql} >= 1)"
    return f"{base})"


def init_score_domain_message(transform: str) -> str:
    if transform == "log":
        return "positive numeric values"
    if transform == "logit":
        return "numeric values between 0 and 1"
    return "finite numeric values"


def init_score_options(dataset: Dataset, *, response_column: str = RESPONSE_COLUMN, sample_column: str | None = None) -> list[dict[str, Any]]:
    options: list[dict[str, Any]] = [{"value": INIT_SCORE_NONE, "label": INIT_SCORE_NONE, "kind": "none"}]
    glm_store = GlmModelStore(dataset.path)
    for model in glm_store.list_models():
        model_id = str(model.get("model_id") or "").strip()
        if not model_id:
            continue
        source_id = glm_store.source_id(model_id)
        label = init_score_glm_label(model)
        options.append(
            {
                "value": source_id,
                "label": label,
                "kind": "glm_prediction",
                "model_id": model_id,
                "source_id": source_id,
            }
        )

    columns = dataset.column_map()
    reserved = {str(response_column or RESPONSE_COLUMN), *RESERVED_INIT_SCORE_COLUMNS}
    if sample_column:
        reserved.add(sample_column)
    for column in columns.values():
        if column.name in reserved or not is_numeric_kind(column.kind):
            continue
        options.append(
            {
                "value": column.name,
                "label": column.name,
                "kind": "dataset_column",
                "column": column.name,
            }
        )
    return options


def init_score_glm_label(model: dict[str, Any]) -> str:
    model_id = str(model.get("model_id") or "").strip()
    label = str(model.get("label") or model_id or "GLM").strip()
    response = str(model.get("response_column") or "response").strip()
    denominator = str(model.get("denominator_column") or model.get("offset_column") or "N").strip() or "N"
    return f"{label} glm_prediction ({response} / {denominator})"


def init_score_saved_option(dataset: Dataset, raw: Any) -> dict[str, Any] | None:
    value = normalise_init_score_value(raw)
    if value.lower() == INIT_SCORE_NONE:
        return None
    glm_store = GlmModelStore(dataset.path)
    ref = glm_store.source_ref(value)
    if ref is not None:
        try:
            model = glm_store.model_list_item(glm_store.model_dir(ref.model_id), glm_store.manifest(ref.model_id), glm_store.active_model_id())
            return {
                "value": value,
                "label": init_score_glm_label(model),
                "kind": "glm_prediction",
                "model_id": ref.model_id,
                "source_id": value,
            }
        except ValueError:
            pass
    columns = dataset.column_map()
    if value in columns:
        return {"value": value, "label": value, "kind": "dataset_column", "column": value}
    return {"value": value, "label": f"{value} (missing)", "kind": "missing", "disabled": True, "status": "missing"}


def init_score_current_options(dataset: Dataset, saved_value: Any = None, *, response_column: str = RESPONSE_COLUMN, sample_column: str | None = None) -> list[dict[str, Any]]:
    options = init_score_options(dataset, response_column=response_column, sample_column=sample_column)
    saved = init_score_saved_option(dataset, saved_value)
    if saved and all(option["value"] != saved["value"] for option in options):
        saved = dict(saved)
        saved.setdefault("status", "stale")
        saved["disabled"] = True
        if "missing" not in str(saved.get("label") or "").lower():
            saved["label"] = f"{saved.get('label') or saved['value']} (unavailable)"
        options.append(saved)
    return options


def selected_response_column(payload: dict[str, Any], columns: dict[str, ColumnInfo]) -> str:
    candidate = str(payload.get("response") or payload.get("response_column") or RESPONSE_COLUMN).strip()
    if not candidate or candidate not in columns or not is_numeric_kind(columns[candidate].kind):
        raise ValueError("Choose a valid numeric GBM response column")
    return candidate


def selected_offset_column(payload: dict[str, Any], columns: dict[str, ColumnInfo]) -> str | None:
    candidate = str(payload.get("offset") or payload.get("offset_column") or OFFSET_COLUMN).strip()
    if candidate in {"", "__none__", "N", "Average row value"}:
        return None
    if candidate not in columns or not is_numeric_kind(columns[candidate].kind):
        raise ValueError("Choose a valid numeric GBM denominator column")
    return candidate


def init_score_selection_messages(
    dataset: Dataset,
    *,
    selected_value: str,
    selected_objective: str,
    response_col: str,
    offset_col: str | None,
    sample_column: str | None,
) -> tuple[list[str], list[str]]:
    value = normalise_init_score_value(selected_value)
    if value.lower() == INIT_SCORE_NONE:
        return [], []
    errors: list[str] = []
    warnings = ["LightGBM boost_from_average is ignored when init_score is supplied"]
    if "{" in value or "}" in value:
        errors.append("init_score cannot use grid-search braces")
        return errors, warnings
    transform = init_score_transform(selected_objective)
    if value.startswith("glm:"):
        glm_store = GlmModelStore(dataset.path)
        ref = glm_store.source_ref(value)
        if ref is None or ref.source_kind != "predictions":
            errors.append(f"Choose a current fitted GLM prediction source for init_score: {value}")
            return errors, warnings
        invalid_count = init_score_invalid_count_for_glm(dataset, glm_store.source_path(ref.model_id, "predictions"), offset_col, transform)
        if invalid_count:
            errors.append(f"init_score {value} must contain {init_score_domain_message(transform)} for all scored rows")
        return errors, warnings

    columns = dataset.column_map()
    column = columns.get(value)
    if column is None or not is_numeric_kind(column.kind):
        errors.append(f"Choose a valid numeric dataset column for init_score: {value}")
        return errors, warnings
    if value == response_col:
        errors.append("init_score cannot use the selected GBM response column")
    if sample_column and value == sample_column:
        errors.append("init_score cannot use the GBM sample split column")
    if value in RESERVED_INIT_SCORE_COLUMNS:
        errors.append(f"init_score cannot use reserved model-output column {value}")
    if errors:
        return errors, warnings
    invalid_count = init_score_invalid_count_for_column(dataset, value, offset_col, transform)
    if invalid_count:
        errors.append(f"init_score {value} must contain {init_score_domain_message(transform)} for all scored rows")
    return errors, warnings


def init_score_invalid_count_for_column(dataset: Dataset, column_name: str, offset_col: str | None, transform: str) -> int:
    where_sql = f"WHERE TRY_CAST({quote_ident(offset_col)} AS DOUBLE) > 0" if offset_col else ""
    value_sql = "init_score_value"
    invalid_sql = init_score_domain_invalid_sql(value_sql, transform)
    row = dataset.con.execute(
        f"""
SELECT COALESCE(SUM(CASE WHEN {invalid_sql} THEN 1 ELSE 0 END), 0)
FROM (
  SELECT TRY_CAST({quote_ident(column_name)} AS DOUBLE) AS {value_sql}
  FROM {dataset.relation_sql()}
  {where_sql}
) scored_values
"""
    ).fetchone()
    return int(row[0] or 0)


def init_score_invalid_count_for_glm(dataset: Dataset, prediction_path: Any, offset_col: str | None, transform: str) -> int:
    where_sql = f"WHERE TRY_CAST({quote_ident(offset_col)} AS DOUBLE) > 0" if offset_col else ""
    value_sql = "init_score_value"
    invalid_sql = init_score_domain_invalid_sql(value_sql, transform)
    offset_projection = f", {quote_ident(offset_col)}" if offset_col else ""
    row = dataset.con.execute(
        f"""
WITH base AS (
  SELECT ROW_NUMBER() OVER () AS __lucidum_row_id{offset_projection}
  FROM {dataset.relation_sql()}
),
eligible AS (
  SELECT *
  FROM base
  {where_sql}
),
scored_values AS (
  SELECT TRY_CAST(prediction.glm_prediction AS DOUBLE) AS {value_sql}
  FROM eligible
  LEFT JOIN read_parquet({sql_literal(str(prediction_path))}) prediction USING (__lucidum_row_id)
)
SELECT COALESCE(SUM(CASE WHEN {invalid_sql} THEN 1 ELSE 0 END), 0)
FROM scored_values
"""
    ).fetchone()
    return int(row[0] or 0)


def feature_rows(
    dataset: Dataset,
    gains: dict[str, float] | None = None,
    model_features: list[dict[str, Any]] | None = None,
    reserved_names: set[str] | None = None,
    feature_groupings: dict[str, str] | None = None,
) -> list[dict[str, Any]]:
    columns = dataset.all_column_map()
    invalid_columns = dataset.invalid_column_errors()
    distinct_counts = categorical_distinct_counts(dataset)
    reserved = {RESPONSE_COLUMN, OFFSET_COLUMN, *(reserved_names or set())}
    if dataset_sample_column(dataset):
        reserved.add(SAMPLE_COLUMN)
    model_feature_map = {
        str(item.get("name")): item
        for item in model_features or []
        if isinstance(item, dict) and item.get("name")
    }
    use_model_features = model_features is not None
    rows: list[dict[str, Any]] = []
    for column in columns.values():
        invalid_error = invalid_columns.get(column.name)
        row_kind = "invalid" if invalid_error else column.kind
        usable = False if invalid_error else feature_usable(column)
        disabled_reason = ""
        if invalid_error:
            disabled_reason = invalid_error
        elif column.name in reserved:
            usable = False
            disabled_reason = "reserved response, offset, or sample column"
        elif not usable:
            disabled_reason = "LightGBM feature type is not supported"
        distinct_count = distinct_counts.get(column.name)
        high_cardinality = row_kind == "categorical" and (distinct_count or 0) > HIGH_CARDINALITY_THRESHOLD
        model_feature = model_feature_map.get(column.name, {})
        include = (
            usable and row_kind in {"integer", "numeric", "categorical"}
            if not use_model_features
            else bool(model_feature) and usable and row_kind in {"integer", "numeric", "categorical"}
        )
        gain = model_feature.get("gain") if model_feature else (gains or {}).get(column.name, 0.0)
        row = {
            "name": column.name,
            "grouping": (feature_groupings or {}).get(column.name, ""),
            "duckdb_type": column.duckdb_type,
            "kind": row_kind,
            "include": include,
            "usable": usable,
            "disabled_reason": disabled_reason,
            "invalid": bool(invalid_error),
            "high_cardinality": high_cardinality,
            "distinct_count": distinct_count,
            "monotonicity": display_monotonicity(model_feature.get("monotonicity")) if include else "",
            "gain": round(float(gain or 0.0), 3),
        }
        mean_abs_shap = json_number(model_feature.get("mean_abs_shap"))
        if mean_abs_shap is not None:
            row["mean_abs_shap"] = float(mean_abs_shap)
        rows.append(row)
    return sorted(rows, key=lambda row: (-float(row["gain"]), str(row["name"]).lower()))


def feature_usable(column: ColumnInfo) -> bool:
    return column.kind in {"integer", "numeric", "categorical"}


def categorical_distinct_counts(dataset: Dataset) -> dict[str, int]:
    counts: dict[str, int] = {}
    columns = [
        column
        for column in dataset.column_map().values()
        if column.kind == "categorical"
    ]
    if not columns:
        return counts
    select_sql = ", ".join(
        f"COUNT(DISTINCT {quote_ident(column.name)}) AS {quote_ident('c' + str(index))}"
        for index, column in enumerate(columns)
    )
    row = dataset.con.execute(f"SELECT {select_sql} FROM {dataset.relation_sql()}").fetchone()
    for index, column in enumerate(columns):
        counts[column.name] = int(row[index] or 0)
    return counts


def normalise_features(raw: Any, columns: dict[str, ColumnInfo]) -> list[dict[str, Any]]:
    if not isinstance(raw, list):
        return []
    features: list[dict[str, Any]] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        name = str(item.get("name") or item.get("feature") or "").strip()
        if not name:
            continue
        include = item.get("include", True)
        if isinstance(include, str):
            include = include.strip().lower() not in {"", "0", "false", "no", "off"}
        if not include:
            continue
        if name not in columns:
            raise ValueError(f"Choose a valid GBM feature: {name}")
        monotonicity = normalise_monotonicity(item.get("monotonicity"))
        features.append({"name": name, "monotonicity": monotonicity, "kind": columns[name].kind})
    return features


def normalise_feature_grouping_map(raw: Any) -> dict[str, str]:
    if not isinstance(raw, dict):
        return {}
    groupings: dict[str, str] = {}
    for feature, grouping in raw.items():
        name = str(feature or "").strip()
        group = str(grouping or "").strip()
        if name and group:
            groupings[name] = group
    return groupings


def normalise_feature_interaction_groupings(raw: Any) -> list[str]:
    if not isinstance(raw, list):
        return []
    groupings: list[str] = []
    seen: set[str] = set()
    for item in raw:
        grouping = str(item or "").strip()
        if grouping and grouping not in seen:
            groupings.append(grouping)
            seen.add(grouping)
    return groupings


def normalise_feature_interaction_features(raw: Any) -> list[str]:
    if not isinstance(raw, list):
        return []
    features: list[str] = []
    seen: set[str] = set()
    for item in raw:
        feature = str(item or "").strip()
        if feature and feature not in seen:
            features.append(feature)
            seen.add(feature)
    return features


def feature_interaction_pair_key(left: str, right: str) -> tuple[str, str]:
    return tuple(sorted((left, right), key=lambda value: (value.lower(), value)))


def normalise_feature_interaction_pairs(raw: Any) -> list[dict[str, str]]:
    if not isinstance(raw, list):
        return []
    pairs: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for item in raw:
        if not isinstance(item, dict):
            continue
        left = str(item.get("left") or "").strip()
        right = str(item.get("right") or "").strip()
        if not left or not right or left == right:
            continue
        key = feature_interaction_pair_key(left, right)
        if key in seen:
            continue
        pairs.append({"left": left, "right": right})
        seen.add(key)
    return pairs


def available_feature_interaction_groupings(feature_groupings: dict[str, str]) -> list[str]:
    return sorted({grouping for grouping in feature_groupings.values() if grouping}, key=str.lower)


def feature_interaction_pair_errors(
    raw: Any,
    *,
    columns: dict[str, ColumnInfo],
    selected_feature_names: set[str],
    response_col: str,
    offset_col: str | None,
    reserved_sample_names: set[str],
) -> list[str]:
    if raw in (None, ""):
        return []
    if not isinstance(raw, list):
        return ["Choose valid GBM feature interaction pairs"]
    errors: list[str] = []
    seen: set[tuple[str, str]] = set()
    for item in raw:
        if not isinstance(item, dict):
            errors.append("Choose two valid GBM features for each interaction pair")
            continue
        left = str(item.get("left") or "").strip()
        right = str(item.get("right") or "").strip()
        if not left or not right:
            errors.append("Choose two valid GBM features for each interaction pair")
            continue
        if left == right:
            errors.append(f"Choose two different GBM features for interaction pair: {left}")
            continue
        key = feature_interaction_pair_key(left, right)
        if key in seen:
            errors.append(f"Duplicate GBM feature interaction pair: {key[0]} x {key[1]}")
            continue
        seen.add(key)
        for feature_name in (left, right):
            column = columns.get(feature_name)
            if column is None:
                errors.append(f"Choose a valid GBM feature interaction pair feature: {feature_name}")
            elif feature_name not in selected_feature_names:
                errors.append(f"{feature_name} must be selected to use a GBM feature interaction pair")
            elif not feature_usable(column):
                errors.append(f"{feature_name} cannot be used in a GBM feature interaction pair")
            elif column.name == response_col or (offset_col and column.name == offset_col) or column.name in reserved_sample_names:
                errors.append(f"{feature_name} is reserved and cannot use a GBM feature interaction pair")
    return errors


def feature_interaction_constraint_groups(
    features: list[dict[str, Any]],
    selected_groupings: list[str],
    feature_groupings: dict[str, str],
    selected_features: list[str] | None = None,
) -> list[dict[str, Any]]:
    if not features:
        return []
    selected_feature_set = set(selected_features or [])
    groups: list[dict[str, Any]] = [
        {"grouping": name, "features": [name], "kind": "feature"}
        for feature in features
        for name in [str(feature.get("name") or "").strip()]
        if name and name in selected_feature_set
    ]
    if not selected_groupings:
        return groups
    selected = set(selected_groupings)
    grouped: dict[str, list[str]] = {grouping: [] for grouping in selected_groupings}
    for feature in features:
        name = str(feature.get("name") or "").strip()
        if name in selected_feature_set:
            continue
        grouping = feature_groupings.get(name, "")
        if name and grouping in selected:
            grouped.setdefault(grouping, []).append(name)
    groups.extend(
        {"grouping": grouping, "features": names, "kind": "group"}
        for grouping, names in grouped.items()
        if names
    )
    return groups


def normalise_monotonicity(raw: Any) -> int:
    text = str(raw or "").strip().lower()
    if text in {"", "0", "none", "no"}:
        return 0
    if text in {"1", "+1", "increasing", "increase", "up"}:
        return 1
    if text in {"-1", "decreasing", "decrease", "down"}:
        return -1
    raise ValueError("Use Increasing, 1, Decreasing, -1, or blank for monotonicity")


def display_monotonicity(raw: Any) -> str:
    try:
        value = normalise_monotonicity(raw)
    except ValueError:
        return ""
    if value > 0:
        return "Increasing"
    if value < 0:
        return "Decreasing"
    return ""


def detect_sample_column(dataset: Dataset, requested: Any = None) -> str | None:
    if requested and str(requested).strip() != SAMPLE_COLUMN:
        return None
    return dataset_sample_column(dataset)


def validate_request(dataset: Dataset, payload: dict[str, Any], generated_sample_path: Any = None) -> ValidationResult:
    with dataset.lock:
        errors: list[str] = []
        warnings: list[str] = []
        columns = dataset.column_map()
        response_col = ""
        offset_col: str | None = None
        offset_source = str(
            payload.get("offset_source")
            or payload.get("denominator_source")
            or "dataset"
        ).strip() or "dataset"
        try:
            response_col = selected_response_column(payload, columns)
        except ValueError as exc:
            errors.append(str(exc))
        if offset_source != "dataset":
            errors.append(
                "GBM training is unavailable while Denominator is a model prediction; "
                "use init_score for prediction chaining"
            )
        else:
            try:
                offset_col = selected_offset_column(payload, columns)
            except ValueError as exc:
                errors.append(str(exc))

        params = normalise_parameters(payload.get("parameters"))
        selected_objective = objective(params)
        errors.extend(parameter_option_errors(params))
        parameter_errors, parameter_warnings = parameter_compatibility_messages(params, payload)
        errors.extend(parameter_errors)
        warnings.extend(parameter_warnings)
        selected_training_mode = normalise_training_mode(payload.get("training_mode"))
        if selected_training_mode not in TRAINING_MODES:
            errors.append(f"Choose a valid GBM training mode: {selected_training_mode}")
        selected_features: list[dict[str, Any]] = []
        try:
            selected_features = normalise_features(payload.get("features"), columns)
        except ValueError as exc:
            errors.append(str(exc))
        if not selected_features:
            errors.append("Choose at least one usable GBM feature")

        for feature in selected_features:
            column = columns[feature["name"]]
            if not feature_usable(column):
                errors.append(f"{feature['name']} cannot be used as a LightGBM feature")
            if column.name == response_col or (offset_col and column.name == offset_col):
                errors.append(f"{feature['name']} is reserved for the response or offset")
            if feature["monotonicity"] and not is_numeric_kind(column.kind):
                errors.append(f"{feature['name']} must be numeric to use monotonicity")
            if feature["monotonicity"] and selected_objective in GBM_OBJECTIVES and selected_objective not in MONOTONE_OBJECTIVES:
                errors.append(f"Monotonicity is not supported for objective {selected_objective}")

        if response_col:
            errors.extend(response_objective_errors(dataset, selected_objective, response_col))
        if offset_col:
            invalid_denominator = int(
                dataset.con.execute(
                    f"SELECT COUNT(*) FROM {dataset.relation_sql()} WHERE TRY_CAST({quote_ident(offset_col)} AS DOUBLE) <= 0 OR {quote_ident(offset_col)} IS NULL"
                ).fetchone()[0]
            )
            if invalid_denominator:
                warnings.append(f"{invalid_denominator:,} rows have non-positive or missing denominator and will be excluded")
        else:
            warnings.append("No denominator column is selected; GBM offset values will be treated as 1")

        sample_column = detect_sample_column(dataset, payload.get("sample_column"))
        has_generated_sample = bool(generated_sample_path and generated_sample_is_current(dataset, generated_sample_path))
        reserved_sample_names = {SAMPLE_COLUMN} if sample_column else set()
        if sample_column:
            sample_errors, sample_warnings = sample_split_messages(
                dataset_training_sample_counts(dataset, offset_col),
                source_label=sample_column,
            )
            errors.extend(sample_errors)
            warnings.extend(sample_warnings)
        elif has_generated_sample:
            sample_errors, sample_warnings = sample_split_messages(
                generated_training_sample_counts(dataset, generated_sample_path, offset_col),
                source_label="generated SAMPLE",
            )
            errors.extend(sample_errors)
            warnings.extend(sample_warnings)
        elif payload.get("create_sample"):
            warnings.append("A generated 60/20/20 SAMPLE split will be created and reused for later GBM training")
        else:
            warnings.append("No sample column was found; the GBM will train on all valid rows without early stopping")

        if response_col:
            init_errors, init_warnings = init_score_selection_messages(
                dataset,
                selected_value=init_score_value(params),
                selected_objective=selected_objective,
                response_col=response_col,
                offset_col=offset_col,
                sample_column=sample_column,
            )
            errors.extend(init_errors)
            warnings.extend(init_warnings)

        for feature in selected_features:
            if feature["name"] in reserved_sample_names:
                errors.append(f"{feature['name']} is reserved for the GBM sample split")

        feature_grouping_map = normalise_feature_grouping_map(payload.get("feature_groupings"))
        valid_interaction_groupings = set(available_feature_interaction_groupings(feature_grouping_map))
        selected_interaction_groupings = normalise_feature_interaction_groupings(payload.get("feature_interaction_groupings"))
        selected_interaction_features = normalise_feature_interaction_features(payload.get("feature_interaction_features"))
        selected_interaction_pairs = normalise_feature_interaction_pairs(payload.get("feature_interaction_pairs"))
        for grouping in selected_interaction_groupings:
            if grouping not in valid_interaction_groupings:
                errors.append(f"Choose a valid GBM feature interaction grouping: {grouping}")
        for feature_name in selected_interaction_features:
            column = columns.get(feature_name)
            if column is None:
                errors.append(f"Choose a valid GBM feature interaction feature: {feature_name}")
            elif not feature_usable(column):
                errors.append(f"{feature_name} cannot be used as a LightGBM feature interaction constraint")
            elif column.name == response_col or (offset_col and column.name == offset_col) or column.name in reserved_sample_names:
                errors.append(f"{feature_name} is reserved and cannot use a feature interaction constraint")
        pair_feature_names = {
            feature_name
            for pair in selected_interaction_pairs
            for feature_name in (pair["left"], pair["right"])
        }
        selected_interaction_feature_set = set(selected_interaction_features)
        if selected_interaction_pairs and selected_interaction_groupings:
            for grouping in selected_interaction_groupings:
                overlapping: list[str] = []
                for feature in selected_features:
                    feature_name = str(feature.get("name") or "").strip()
                    if not feature_name or feature_name in selected_interaction_feature_set:
                        continue
                    if feature_grouping_map.get(feature_name, "") == grouping and feature_name in pair_feature_names:
                        overlapping.append(feature_name)
                overlapping = sorted(set(overlapping), key=str.lower)
                if overlapping:
                    noun = "feature" if len(overlapping) == 1 else "features"
                    errors.append(f"GBM feature interaction grouping {grouping} cannot include paired {noun}: {', '.join(overlapping)}")
        for feature_name in selected_interaction_features:
            if selected_interaction_pairs and feature_name in pair_feature_names:
                errors.append(f"{feature_name} cannot be both main-effect-only and used in a GBM feature interaction pair")
        selected_feature_names = {feature["name"] for feature in selected_features}
        errors.extend(
            feature_interaction_pair_errors(
                payload.get("feature_interaction_pairs"),
                columns=columns,
                selected_feature_names=selected_feature_names,
                response_col=response_col,
                offset_col=offset_col,
                reserved_sample_names=reserved_sample_names,
            )
        )
        if selected_interaction_pairs and integer_parameter(params, "num_leaves", 0) > 3:
            warnings.append("Feature interaction pairs constrain branch co-occurrence, but num_leaves above 3 can still create higher-order branches through overlapping pairs")

        if selected_training_mode == "ebm":
            early_stopping_rounds = integer_parameter(params, "early_stopping_rounds", 0)
            num_leaves = integer_parameter(params, "num_leaves", 0)
            if sample_column:
                if not _sample_counts_have_training_and_test(dataset_training_sample_counts(dataset, offset_col)):
                    errors.append("EBM mode requires SAMPLE to contain training and test rows after denominator filtering")
            elif has_generated_sample:
                if not _sample_counts_have_training_and_test(generated_training_sample_counts(dataset, generated_sample_path, offset_col)):
                    errors.append("EBM mode requires generated SAMPLE to contain training and test rows after denominator filtering")
            else:
                errors.append("EBM mode requires a dataset or generated SAMPLE split with training and test rows")
            if early_stopping_rounds <= 0:
                errors.append("EBM mode requires early_stopping_rounds greater than 0")
            if num_leaves < 2:
                errors.append("EBM mode requires num_leaves of at least 2")

        return ValidationResult(ok=not errors, errors=errors, warnings=warnings)


def integer_parameter(params: dict[str, Any], name: str, default: int) -> int:
    value = params.get(name, default)
    if value is None or str(value).strip() == "":
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def non_negative_integer_parameter(params: dict[str, Any], name: str, default: int) -> int | None:
    value = params.get(name, default)
    if value is None or str(value).strip() == "":
        value = default
    if isinstance(value, bool):
        return None
    try:
        if isinstance(value, float):
            if not value.is_integer():
                return None
            parsed = int(value)
        else:
            text = str(value).strip()
            if "." in text:
                number = float(text)
                if not number.is_integer():
                    return None
                parsed = int(number)
            else:
                parsed = int(text)
    except (TypeError, ValueError):
        return None
    return parsed if parsed >= 0 else None


def number_parameter(params: dict[str, Any], name: str, default: float) -> float:
    try:
        return float(params.get(name, default))
    except (TypeError, ValueError):
        return default


def bool_parameter(params: dict[str, Any], name: str, default: bool = False) -> bool:
    value = params.get(name, default)
    if isinstance(value, bool):
        return value
    text = str(value or "").strip().lower()
    if text in {"true", "1", "yes", "y", "on"}:
        return True
    if text in {"false", "0", "no", "n", "off", ""}:
        return False
    return default


def shap_rows_requested(raw: Any) -> bool:
    text = str(raw or "0").strip().lower()
    return text not in {"", "0", "zero", "none", "no"}


def response_objective_errors(dataset: Dataset, selected_objective: str, response_column: str = RESPONSE_COLUMN) -> list[str]:
    column = quote_ident(response_column)
    relation = dataset.relation_sql()
    errors: list[str] = []
    stats = dataset.con.execute(
        f"""
SELECT
  MIN(TRY_CAST({column} AS DOUBLE)) AS min_y,
  MAX(TRY_CAST({column} AS DOUBLE)) AS max_y,
  COUNT(*) FILTER (WHERE TRY_CAST({column} AS DOUBLE) IS NULL) AS missing_y,
  COUNT(DISTINCT TRY_CAST({column} AS DOUBLE)) AS distinct_y
FROM {relation}
"""
    ).fetchone()
    min_y, max_y, missing_y, distinct_y = stats
    if missing_y:
        errors.append(f"{missing_y:,} response values are missing or non-numeric")
    if selected_objective == "gamma" and (min_y is None or min_y <= 0):
        errors.append("Gamma objective requires strictly positive response values")
    if selected_objective in {"poisson", "tweedie"} and (min_y is None or min_y < 0):
        errors.append(f"{selected_objective} objective requires non-negative response values")
    if selected_objective == "binary":
        if min_y not in {0, 1} or max_y not in {0, 1} or int(distinct_y or 0) > 2:
            errors.append("Binary objective requires response values of 0 and 1 only")
    if selected_objective in CROSS_ENTROPY_OBJECTIVES and (min_y is None or min_y < 0 or max_y is None or max_y > 1):
        errors.append(f"{selected_objective} objective requires response values between 0 and 1")
    return errors


def sample_split_errors(dataset: Dataset, sample_column: str, offset_column: str | None = OFFSET_COLUMN) -> list[str]:
    if sample_column != SAMPLE_COLUMN:
        return [f"Sample column {sample_column} is not supported; use SAMPLE"]
    errors, _ = sample_split_messages(dataset_training_sample_counts(dataset, offset_column), source_label=sample_column)
    return errors


def sample_split_messages(counts: dict[str, int], *, source_label: str) -> tuple[list[str], list[str]]:
    errors: list[str] = []
    warnings: list[str] = []
    if counts.get("training", 0) == 0:
        errors.append(f"{source_label} must contain at least one training row")
    if counts.get("test", 0) == 0:
        errors.append(f"{source_label} must contain at least one test row for early stopping")
    if counts.get("validation", 0) == 0:
        warnings.append(f"{source_label} has no validation rows; validation holdout diagnostics will be skipped")
    return errors, warnings


__all__ = [
    "DEFAULT_METRIC",
    "DEFAULT_OBJECTIVE",
    "DEFAULT_TRAINING_MODE",
    "DATA_SAMPLE_STRATEGIES",
    "GBM_METRICS",
    "GBM_OBJECTIVES",
    "INIT_SCORE_NONE",
    "INIT_SCORE_PARAMETER",
    "OFFSET_COLUMN",
    "RESPONSE_COLUMN",
    "TRAINING_MODES",
    "ValidationResult",
    "available_feature_interaction_groupings",
    "default_parameters",
    "detect_sample_column",
    "ebm_available",
    "display_monotonicity",
    "feature_interaction_constraint_groups",
    "feature_interaction_pair_errors",
    "feature_interaction_pair_key",
    "feature_rows",
    "init_score_current_options",
    "init_score_options",
    "init_score_requested",
    "init_score_transform",
    "init_score_value",
    "normalise_init_score_value",
    "normalise_feature_grouping_map",
    "normalise_feature_interaction_features",
    "normalise_feature_interaction_groupings",
    "normalise_feature_interaction_pairs",
    "normalise_features",
    "normalise_parameters",
    "normalise_training_mode",
    "selected_offset_column",
    "selected_response_column",
    "uses_log_offset",
    "validate_request",
]
