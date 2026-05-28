from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from py_lucidum.core import ColumnInfo, Dataset, is_numeric_kind, json_number, quote_ident

from .sample import (
    SAMPLE_COLUMN,
    dataset_sample_column,
    dataset_training_sample_counts,
    generated_sample_is_current,
    generated_training_sample_counts,
)


RESPONSE_COLUMN = "actualNumerator"
OFFSET_COLUMN = "denominator"
DEFAULT_OBJECTIVE = "poisson"
DEFAULT_METRIC = "poisson"
DEFAULT_TRAINING_MODE = "normal"
TRAINING_MODES = ("normal", "ebm")
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
HIGH_CARDINALITY_THRESHOLD = 100


@dataclass(frozen=True)
class ValidationResult:
    ok: bool
    errors: list[str]
    warnings: list[str]

    def as_payload(self) -> dict[str, Any]:
        return {"ok": self.ok, "errors": self.errors, "warnings": self.warnings}


def default_parameters() -> list[dict[str, Any]]:
    return [
        {"name": "objective", "value": DEFAULT_OBJECTIVE, "important": True},
        {"name": "metric", "value": DEFAULT_METRIC, "important": True},
        {"name": "num_iterations", "value": 200, "important": True},
        {"name": "learning_rate", "value": 0.05, "important": True},
        {"name": "num_leaves", "value": 31, "important": True},
        {"name": "max_depth", "value": -1, "important": True},
        {"name": "min_data_in_leaf", "value": 20, "important": True},
        {"name": "early_stopping_rounds", "value": 25, "important": True},
        {"name": "feature_fraction", "value": 1.0, "important": False},
        {"name": "bagging_fraction", "value": 1.0, "important": False},
        {"name": "bagging_freq", "value": 0, "important": False},
        {"name": "lambda_l1", "value": 0.0, "important": False},
        {"name": "lambda_l2", "value": 0.0, "important": False},
        {"name": "min_gain_to_split", "value": 0.0, "important": False},
        {"name": "max_bin", "value": 255, "important": False},
        {"name": "verbosity", "value": -1, "important": False},
        {"name": "seed", "value": 2026, "important": False},
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
    if selected_objective not in GBM_OBJECTIVES:
        errors.append(f"Choose a valid LightGBM objective: {selected_objective}")
    if selected_metric not in GBM_METRICS:
        errors.append(f"Choose a valid LightGBM metric: {selected_metric}")
    return errors


def ebm_available(dataset: Dataset, offset_column: str | None = OFFSET_COLUMN) -> bool:
    columns = dataset.column_map()
    if not dataset_sample_column(dataset):
        return False
    checked_offset = offset_column if offset_column and offset_column in columns and is_numeric_kind(columns[offset_column].kind) else None
    counts = dataset_training_sample_counts(dataset, checked_offset)
    return counts.get("training", 0) > 0 and counts.get("test", 0) > 0


def uses_log_offset(params: dict[str, Any]) -> bool:
    return objective(params) in LOG_LINK_OBJECTIVES


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


def available_feature_interaction_groupings(feature_groupings: dict[str, str]) -> list[str]:
    return sorted({grouping for grouping in feature_groupings.values() if grouping}, key=str.lower)


def feature_interaction_constraint_groups(
    features: list[dict[str, Any]],
    selected_groupings: list[str],
    feature_groupings: dict[str, str],
) -> list[dict[str, Any]]:
    if not features or not selected_groupings:
        return []
    selected = set(selected_groupings)
    grouped: dict[str, list[str]] = {grouping: [] for grouping in selected_groupings}
    for feature in features:
        name = str(feature.get("name") or "").strip()
        grouping = feature_groupings.get(name, "")
        if name and grouping in selected:
            grouped.setdefault(grouping, []).append(name)
    return [
        {"grouping": grouping, "features": names}
        for grouping, names in grouped.items()
        if names
    ]


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
        try:
            response_col = selected_response_column(payload, columns)
        except ValueError as exc:
            errors.append(str(exc))
        try:
            offset_col = selected_offset_column(payload, columns)
        except ValueError as exc:
            errors.append(str(exc))

        params = normalise_parameters(payload.get("parameters"))
        selected_objective = objective(params)
        errors.extend(parameter_option_errors(params))
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

        for feature in selected_features:
            if feature["name"] in reserved_sample_names:
                errors.append(f"{feature['name']} is reserved for the GBM sample split")

        feature_grouping_map = normalise_feature_grouping_map(payload.get("feature_groupings"))
        valid_interaction_groupings = set(available_feature_interaction_groupings(feature_grouping_map))
        for grouping in normalise_feature_interaction_groupings(payload.get("feature_interaction_groupings")):
            if grouping not in valid_interaction_groupings:
                errors.append(f"Choose a valid GBM feature interaction grouping: {grouping}")

        if selected_training_mode == "ebm":
            early_stopping_rounds = integer_parameter(params, "early_stopping_rounds", 0)
            num_leaves = integer_parameter(params, "num_leaves", 0)
            if not sample_column:
                errors.append("EBM mode requires a dataset SAMPLE column with training and test rows")
            elif sample_column and (counts := dataset_training_sample_counts(dataset, offset_col)):
                if counts.get("training", 0) == 0 or counts.get("test", 0) == 0:
                    errors.append("EBM mode requires SAMPLE to contain training and test rows after denominator filtering")
            if early_stopping_rounds <= 0:
                errors.append("EBM mode requires early_stopping_rounds greater than 0")
            if num_leaves < 2:
                errors.append("EBM mode requires num_leaves of at least 2")

        return ValidationResult(ok=not errors, errors=errors, warnings=warnings)


def integer_parameter(params: dict[str, Any], name: str, default: int) -> int:
    try:
        return int(params.get(name, default) or default)
    except (TypeError, ValueError):
        return default


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
    "GBM_METRICS",
    "GBM_OBJECTIVES",
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
    "feature_rows",
    "normalise_feature_grouping_map",
    "normalise_feature_interaction_groupings",
    "normalise_features",
    "normalise_parameters",
    "normalise_training_mode",
    "selected_offset_column",
    "selected_response_column",
    "uses_log_offset",
    "validate_request",
]
