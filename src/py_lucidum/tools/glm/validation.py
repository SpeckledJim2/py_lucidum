from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from py_lucidum.core import Dataset, is_numeric_kind, quote_ident, sql_literal


RESPONSE_COLUMN = "response_column"
DENOMINATOR_COLUMN = "denominator_column"
SAMPLE_COLUMN = "SAMPLE"
TARGET_COLUMN = "__lucidum_glm_target"
FAMILY_OPTIONS = (
    "normal",
    "poisson",
    "gamma",
    "tweedie",
    "binomial",
    "inverse.gaussian",
    "negative.binomial",
)
FAMILY_LABELS = {
    "normal": "Normal",
    "poisson": "Poisson",
    "gamma": "Gamma",
    "tweedie": "Tweedie",
    "binomial": "Binomial",
    "inverse.gaussian": "Inverse Gaussian",
    "negative.binomial": "Negative Binomial",
}
FAMILY_PARAMETER_DEFAULTS = {
    "tweedie": 1.5,
    "negative.binomial": 1.0,
}
TRAINING_SCOPES = ("all", "training")
DENOMINATOR_EMPTY = {"", "__none__", "none", "n", "average row value", "average"}
UNSAFE_PATTERN = re.compile(
    r"__(?:\w*)|(?:^|[^\w])(import|eval|exec|open|compile|globals|locals|subprocess|socket)(?:[^\w]|$)",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class FormulaParts:
    raw_formula: str
    stripped_formula: str
    response_column: str
    rhs_formula: str
    fitted_formula: str


def family_options_payload() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for family in FAMILY_OPTIONS:
        rows.append(
            {
                "value": family,
                "label": FAMILY_LABELS[family],
                "parameter": (
                    {
                        "name": "tweedie_power",
                        "label": "var.power",
                        "default": FAMILY_PARAMETER_DEFAULTS[family],
                        "min": 1.0,
                        "max": 2.0,
                    }
                    if family == "tweedie"
                    else (
                        {
                            "name": "negative_binomial_theta",
                            "label": "theta",
                            "default": FAMILY_PARAMETER_DEFAULTS[family],
                            "min": 0.000001,
                        }
                        if family == "negative.binomial"
                        else None
                    )
                ),
            }
        )
    return rows


def clean_identifier(value: Any) -> str:
    return str(value or "").strip()


def column_map(dataset: Dataset) -> dict[str, str]:
    return {column.name: column.name for column in dataset.valid_schema_columns()}


def numeric_column_names(dataset: Dataset) -> set[str]:
    return {
        column.name
        for column in dataset.valid_schema_columns()
        if is_numeric_kind(column.kind)
    }


def normalise_family(value: Any) -> str:
    family = str(value or "normal").strip().lower().replace("_", ".")
    aliases = {
        "gaussian": "normal",
        "inverse gaussian": "inverse.gaussian",
        "negative binomial": "negative.binomial",
    }
    family = aliases.get(family, family)
    if family not in FAMILY_OPTIONS:
        raise ValueError("Choose a valid GLM family")
    return family


def normalise_training_scope(value: Any) -> str:
    scope = str(value or "all").strip().lower()
    if scope not in TRAINING_SCOPES:
        raise ValueError("Choose whether to fit all rows or training rows")
    return scope


def normalise_denominator(value: Any) -> str:
    text = str(value or "").strip()
    return "" if text.lower() in DENOMINATOR_EMPTY else text


def family_parameter(family: str, payload: dict[str, Any]) -> float | None:
    raw = None
    if family == "tweedie":
        raw = payload.get("family_parameter", payload.get("tweedie_power"))
    elif family == "negative.binomial":
        raw = payload.get("family_parameter", payload.get("negative_binomial_theta"))
    if raw is None or str(raw).strip() == "":
        return FAMILY_PARAMETER_DEFAULTS.get(family)
    try:
        value = float(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError("Choose a numeric GLM family parameter") from exc
    if family == "tweedie" and not (1.0 <= value <= 2.0):
        raise ValueError("Choose a Tweedie power between 1 and 2")
    if family == "negative.binomial" and value <= 0:
        raise ValueError("Choose a positive negative-binomial theta")
    return value


def strip_formula_comments(formula: str) -> str:
    lines: list[str] = []
    for line in str(formula or "").splitlines():
        quote: str | None = None
        escaped = False
        kept: list[str] = []
        for char in line:
            if escaped:
                kept.append(char)
                escaped = False
                continue
            if char == "\\" and quote in {"'", '"'}:
                kept.append(char)
                escaped = True
                continue
            if char in {"'", '"', "`"}:
                if quote == char:
                    quote = None
                elif quote is None:
                    quote = char
                kept.append(char)
                continue
            if char == "#" and quote is None:
                break
            kept.append(char)
        lines.append("".join(kept).rstrip())
    return "\n".join(lines).strip()


def unsafe_formula_error(formula: str) -> str | None:
    if any(separator in formula for separator in (";", "/*", "*/")):
        return "Formula text cannot contain statement separators"
    if UNSAFE_PATTERN.search(formula):
        return "Formula text contains unsupported or unsafe Python tokens"
    return None


def find_formula_split(formula: str) -> int:
    quote: str | None = None
    escaped = False
    for index, char in enumerate(formula):
        if escaped:
            escaped = False
            continue
        if char == "\\" and quote in {"'", '"'}:
            escaped = True
            continue
        if char in {"'", '"', "`"}:
            if quote == char:
                quote = None
            elif quote is None:
                quote = char
            continue
        if char == "~" and quote is None:
            return index
    return -1


def unquote_formula_name(value: str) -> str:
    text = str(value or "").strip()
    if len(text) >= 2 and text[0] == "`" and text[-1] == "`":
        return text[1:-1].replace("\\`", "`")
    return text


def parse_formula(raw_formula: Any, response_column: Any) -> FormulaParts:
    raw = str(raw_formula or "")
    stripped = strip_formula_comments(raw)
    if not stripped:
        raise ValueError("Enter a GLM formula")
    if error := unsafe_formula_error(stripped):
        raise ValueError(error)
    split_at = find_formula_split(stripped)
    if split_at >= 0:
        lhs = stripped[:split_at].strip()
        rhs = stripped[split_at + 1 :].strip()
        response = unquote_formula_name(lhs)
    else:
        rhs = stripped
        response = clean_identifier(response_column)
    if not response:
        raise ValueError("Choose a response column or enter a full response ~ terms formula")
    if not rhs:
        raise ValueError("Enter the right-hand side of the GLM formula")
    return FormulaParts(
        raw_formula=raw,
        stripped_formula=stripped,
        response_column=response,
        rhs_formula=rhs,
        fitted_formula=f"{TARGET_COLUMN} ~ {rhs}",
    )


def selected_response_column(payload: dict[str, Any]) -> str:
    for key in ("response_column", "actual", "actual_column", "response"):
        value = clean_identifier(payload.get(key))
        if value:
            return value
    return ""


def selected_denominator_column(payload: dict[str, Any]) -> str:
    for key in ("denominator_column", "denominator", "offset_column", "weight_column"):
        value = normalise_denominator(payload.get(key))
        if value:
            return value
    return ""


def physical_sample_column(dataset: Dataset) -> str | None:
    columns = {name.lower(): name for name in column_map(dataset)}
    return columns.get(SAMPLE_COLUMN.lower())


def sample_metadata(dataset: Dataset) -> dict[str, Any]:
    sample_column = physical_sample_column(dataset)
    if not sample_column:
        return {
            "available": False,
            "column": None,
            "training_rows": 0,
            "non_training_rows": 0,
        }
    column_sql = quote_ident(sample_column)
    sql = f"""
SELECT
  SUM(CASE WHEN LOWER(TRIM(CAST({column_sql} AS VARCHAR))) = 'training' THEN 1 ELSE 0 END) AS training_rows,
  SUM(CASE WHEN LOWER(TRIM(CAST({column_sql} AS VARCHAR))) != 'training' OR {column_sql} IS NULL THEN 1 ELSE 0 END) AS non_training_rows
FROM {dataset.relation_sql()}
"""
    with dataset.lock:
        row = dataset.con.execute(sql).fetchone()
    return {
        "available": True,
        "column": sample_column,
        "training_rows": int(row[0] or 0),
        "non_training_rows": int(row[1] or 0),
    }


def validate_request(dataset: Dataset, payload: dict[str, Any]) -> dict[str, Any]:
    errors: list[str] = []
    warnings: list[str] = []
    valid_columns = column_map(dataset)
    numeric_columns = numeric_column_names(dataset)

    try:
        family = normalise_family(payload.get("family"))
    except ValueError as exc:
        family = "normal"
        errors.append(str(exc))
    try:
        family_param = family_parameter(family, payload)
    except ValueError as exc:
        family_param = FAMILY_PARAMETER_DEFAULTS.get(family)
        errors.append(str(exc))

    try:
        training_scope = normalise_training_scope(payload.get("training_scope", payload.get("fit_scope")))
    except ValueError as exc:
        training_scope = "all"
        errors.append(str(exc))

    try:
        formula = parse_formula(payload.get("formula"), selected_response_column(payload))
    except ValueError as exc:
        formula = None
        errors.append(str(exc))

    response_column = formula.response_column if formula else selected_response_column(payload)
    if response_column and response_column not in valid_columns:
        errors.append(f"Choose a valid response column: {response_column}")
    elif response_column and response_column not in numeric_columns:
        errors.append("Choose a numeric response column for GLM fitting")

    denominator_column = selected_denominator_column(payload)
    if denominator_column:
        if denominator_column not in valid_columns:
            errors.append(f"Choose a valid denominator column: {denominator_column}")
        elif denominator_column not in numeric_columns:
            errors.append("Choose a numeric denominator column")

    sample = sample_metadata(dataset)
    if training_scope == "training":
        if not sample["available"]:
            errors.append("Training rows require a physical SAMPLE column")
        elif not sample["training_rows"]:
            errors.append("No rows have SAMPLE = training")

    result: dict[str, Any] = {
        "ok": not errors,
        "errors": errors,
        "warnings": warnings,
        "family": family,
        "family_parameter": family_param,
        "link": "auto",
        "training_scope": training_scope,
        "response_column": response_column,
        "denominator_column": denominator_column,
        "sample": sample,
    }
    if formula:
        result["formula"] = {
            "raw": formula.raw_formula,
            "stripped": formula.stripped_formula,
            "rhs": formula.rhs_formula,
            "fitted": formula.fitted_formula,
        }
    return result


def denominator_valid_sql(column: str) -> str:
    return f"TRY_CAST({quote_ident(column)} AS DOUBLE) > 0"


def dataset_relation_sql(dataset_path: Any) -> str:
    path = sql_literal(str(dataset_path))
    suffix = str(dataset_path).lower()
    if suffix.endswith(".parquet"):
        return f"read_parquet({path})"
    if suffix.endswith(".csv"):
        return f"read_csv_auto({path}, header=true, ignore_errors=true)"
    raise ValueError("Only .csv and .parquet files are supported in this prototype")
