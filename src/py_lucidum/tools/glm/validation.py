from __future__ import annotations

import math
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
TRAINING_SCOPES = ("all", "training", "training_test")
REGULARIZATION_MODES = ("none", "auto", "manual")
REGULARIZATION_MIXES = {
    "ridge": 0.0,
    "elastic": 0.5,
    "elastic.net": 0.5,
    "elastic_net": 0.5,
    "elastic net": 0.5,
    "lasso": 1.0,
}
DENOMINATOR_EMPTY = {"", "__none__", "none", "n", "average row value", "average"}
UNSAFE_PATTERN = re.compile(
    r"__(?:\w*)|(?:^|[^\w])(import|eval|exec|open|compile|globals|locals|subprocess|socket)(?:[^\w]|$)",
    re.IGNORECASE,
)
NATURAL_SPLINE_INTERCEPT_WARNING = (
    "Natural spline terms with an intercept can be rank-deficient unless the spline basis is centered. "
    'Use `ns(feature, df=4, constraints="center")` to keep an intercept, or `0 + ns(...)` when a no-intercept spline is intended.'
)
BASIS_SPLINE_CONSTRAINTS_ERROR = (
    '`bs(...)` does not accept `constraints=`. Remove the `constraints` argument, '
    'or use `ns(..., constraints="center")` for a centered natural spline.'
)


@dataclass(frozen=True)
class FormulaParts:
    raw_formula: str
    stripped_formula: str
    response_column: str
    rhs_formula: str
    fitted_rhs_formula: str
    fitted_formula: str
    offset_terms: list[str]
    fit_intercept: bool
    has_predictor_terms: bool
    intercept_only: bool


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


def regularization_options_payload() -> dict[str, Any]:
    return {
        "modes": [
            {"value": "none", "label": "None"},
            {"value": "auto", "label": "Auto"},
            {"value": "manual", "label": "Manual"},
        ],
        "mixes": [
            {"value": "0", "label": "Ridge", "l1_ratio": 0.0},
            {"value": "0.5", "label": "Elastic net", "l1_ratio": 0.5},
            {"value": "1", "label": "Lasso", "l1_ratio": 1.0},
        ],
        "auto_l1_ratio": [0.0, 0.5, 1.0],
        "auto_n_alphas": 50,
    }


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
        raise ValueError("Choose whether to fit All, Training, or Training + Test rows")
    return scope


def training_scope_sample_values(scope: Any) -> tuple[str, ...]:
    normalized = normalise_training_scope(scope)
    if normalized == "training":
        return ("training",)
    if normalized == "training_test":
        return ("training", "test")
    return ()


def normalise_denominator(value: Any) -> str:
    text = str(value or "").strip()
    return "" if text.lower() in DENOMINATOR_EMPTY else text


def normalise_regularization_mode(value: Any) -> str:
    mode = str(value or "none").strip().lower().replace("-", "_")
    aliases = {
        "": "none",
        "off": "none",
        "unpenalized": "none",
        "unpenalised": "none",
        "cv": "auto",
        "cross_validation": "auto",
        "cross.validation": "auto",
    }
    mode = aliases.get(mode, mode)
    if mode not in REGULARIZATION_MODES:
        raise ValueError("Choose a valid GLM regularization mode")
    return mode


def regularization_payload_source(payload: dict[str, Any]) -> dict[str, Any]:
    raw = payload.get("regularization")
    return raw if isinstance(raw, dict) else {}


def regularization_number(raw: Any, label: str) -> float:
    try:
        value = float(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Choose a numeric GLM regularization {label}") from exc
    if not math.isfinite(value):
        raise ValueError(f"Choose a finite GLM regularization {label}")
    return value


def regularization_l1_ratio(raw: Any) -> float:
    text = str(raw or "").strip().lower()
    if text in REGULARIZATION_MIXES:
        return REGULARIZATION_MIXES[text]
    value = regularization_number(raw, "mix")
    if not (0.0 <= value <= 1.0):
        raise ValueError("Choose a GLM regularization mix from 0 to 1")
    return value


def regularization_config(payload: dict[str, Any]) -> dict[str, Any]:
    source = regularization_payload_source(payload)
    mode = normalise_regularization_mode(source.get("mode", payload.get("regularization_mode", payload.get("penalty_mode"))))
    if mode == "none":
        return {
            "mode": "none",
            "alpha": 0.0,
            "l1_ratio": 0.0,
            "selected_alpha": None,
            "selected_l1_ratio": None,
            "scale_predictors": False,
            "nonzero_coefficients": None,
        }
    if mode == "auto":
        return {
            "mode": "auto",
            "alpha": None,
            "l1_ratio": [0.0, 0.5, 1.0],
            "selected_alpha": None,
            "selected_l1_ratio": None,
            "scale_predictors": True,
            "nonzero_coefficients": None,
            "n_alphas": 50,
        }
    alpha = regularization_number(source.get("alpha", payload.get("alpha")), "alpha")
    if alpha <= 0:
        raise ValueError("Choose a positive GLM regularization alpha")
    l1_ratio = regularization_l1_ratio(source.get("l1_ratio", source.get("mix", payload.get("l1_ratio", payload.get("regularization_mix", 0.5)))))
    return {
        "mode": "manual",
        "alpha": alpha,
        "l1_ratio": l1_ratio,
        "selected_alpha": alpha,
        "selected_l1_ratio": l1_ratio,
        "scale_predictors": True,
        "nonzero_coefficients": None,
    }


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


def matching_call_end(text: str, open_index: int) -> int:
    quote: str | None = None
    escaped = False
    depth = 0
    for index in range(open_index, len(text)):
        char = text[index]
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
        if quote is not None:
            continue
        if char == "(":
            depth += 1
        elif char == ")":
            depth -= 1
            if depth == 0:
                return index + 1
    raise ValueError("Formula has an unclosed offset(...) term")


def find_offset_call(text: str, start: int = 0) -> tuple[int, int, int, str] | None:
    quote: str | None = None
    escaped = False
    index = max(0, start)
    while index < len(text):
        char = text[index]
        if escaped:
            escaped = False
            index += 1
            continue
        if char == "\\" and quote in {"'", '"'}:
            escaped = True
            index += 1
            continue
        if char in {"'", '"', "`"}:
            if quote == char:
                quote = None
            elif quote is None:
                quote = char
            index += 1
            continue
        if quote is None and text[index : index + 6].lower() == "offset":
            before = text[index - 1] if index > 0 else ""
            if before and (before.isalnum() or before in {"_", "."}):
                index += 1
                continue
            cursor = index + 6
            while cursor < len(text) and text[cursor].isspace():
                cursor += 1
            if cursor >= len(text) or text[cursor] != "(":
                index += 1
                continue
            end = matching_call_end(text, cursor)
            return index, cursor, end, text[cursor + 1 : end - 1].strip()
        index += 1
    return None


def find_formula_call(text: str, name: str, start: int = 0) -> tuple[int, int, int, str] | None:
    quote: str | None = None
    escaped = False
    index = max(0, start)
    call_name = str(name or "").strip()
    if not call_name:
        return None
    name_length = len(call_name)
    while index < len(text):
        char = text[index]
        if escaped:
            escaped = False
            index += 1
            continue
        if char == "\\" and quote in {"'", '"'}:
            escaped = True
            index += 1
            continue
        if char in {"'", '"', "`"}:
            if quote == char:
                quote = None
            elif quote is None:
                quote = char
            index += 1
            continue
        if quote is None and text[index : index + name_length].lower() == call_name.lower():
            before = text[index - 1] if index > 0 else ""
            after = text[index + name_length] if index + name_length < len(text) else ""
            if before and (before.isalnum() or before in {"_", "."}):
                index += 1
                continue
            if after and (after.isalnum() or after in {"_", "."}):
                index += 1
                continue
            cursor = index + name_length
            while cursor < len(text) and text[cursor].isspace():
                cursor += 1
            if cursor >= len(text) or text[cursor] != "(":
                index += 1
                continue
            end = matching_call_end(text, cursor)
            return index, cursor, end, text[cursor + 1 : end - 1].strip()
        index += 1
    return None


def formula_call_has_keyword(arguments: str, keyword: str) -> bool:
    quote: str | None = None
    escaped = False
    depth = 0
    target = str(keyword or "").strip().lower()
    if not target:
        return False
    text = str(arguments or "")
    index = 0
    while index < len(text):
        char = text[index]
        if escaped:
            escaped = False
            index += 1
            continue
        if char == "\\" and quote in {"'", '"'}:
            escaped = True
            index += 1
            continue
        if char in {"'", '"', "`"}:
            if quote == char:
                quote = None
            elif quote is None:
                quote = char
            index += 1
            continue
        if quote is None and char in "([{":
            depth += 1
            index += 1
            continue
        if quote is None and char in ")]}":
            depth = max(0, depth - 1)
            index += 1
            continue
        if quote is None and depth == 0 and text[index : index + len(target)].lower() == target:
            before = text[index - 1] if index > 0 else ""
            after_index = index + len(target)
            after = text[after_index] if after_index < len(text) else ""
            if before and (before.isalnum() or before in {"_", "."}):
                index += 1
                continue
            if after and (after.isalnum() or after == "_"):
                index += 1
                continue
            cursor = after_index
            while cursor < len(text) and text[cursor].isspace():
                cursor += 1
            if cursor < len(text) and text[cursor] == "=":
                return True
        index += 1
    return False


def formula_warnings(rhs: str, fit_intercept: bool) -> list[str]:
    if not fit_intercept:
        return []
    warnings: list[str] = []
    cursor = 0
    while True:
        match = find_formula_call(rhs, "ns", cursor)
        if match is None:
            break
        _start, _open, end, arguments = match
        if not formula_call_has_keyword(arguments, "constraints"):
            warnings.append(NATURAL_SPLINE_INTERCEPT_WARNING)
            break
        cursor = end
    return warnings


def formula_errors(rhs: str) -> list[str]:
    errors: list[str] = []
    cursor = 0
    while True:
        match = find_formula_call(rhs, "bs", cursor)
        if match is None:
            break
        _start, _open, end, arguments = match
        if formula_call_has_keyword(arguments, "constraints"):
            errors.append(BASIS_SPLINE_CONSTRAINTS_ERROR)
            break
        cursor = end
    return errors


def remove_offset_span(text: str, start: int, end: int) -> str:
    left = start
    while left > 0 and text[left - 1].isspace():
        left -= 1
    right = end
    while right < len(text) and text[right].isspace():
        right += 1
    next_nonspace = right
    while next_nonspace < len(text) and text[next_nonspace].isspace():
        next_nonspace += 1
    if left > 0 and text[left - 1] == "+":
        return f"{text[:left - 1]} {text[right:]}"
    if next_nonspace < len(text) and text[next_nonspace] == "+":
        return f"{text[:left]} {text[next_nonspace + 1:]}"
    return f"{text[:left]} {text[right:]}"


def normalise_rhs_formula(rhs: str) -> str:
    text = re.sub(r"\s+", " ", str(rhs or "")).strip()
    text = re.sub(r"^\+\s*", "", text)
    text = re.sub(r"\s*\+$", "", text).strip()
    text = re.sub(r"\+\s*\+", "+", text)
    return text or "1"


def top_level_formula_terms(rhs: str) -> list[tuple[str, str]]:
    terms: list[tuple[str, str]] = []
    quote: str | None = None
    escaped = False
    depth = 0
    sign = "+"
    start = 0
    text = str(rhs or "")
    for index, char in enumerate(text):
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
        if quote is not None:
            continue
        if char == "(":
            depth += 1
            continue
        if char == ")":
            depth = max(0, depth - 1)
            continue
        if depth == 0 and char in {"+", "-"}:
            term = text[start:index].strip()
            if term:
                terms.append((sign, term))
            sign = char
            start = index + 1
    term = text[start:].strip()
    if term:
        terms.append((sign, term))
    return terms


def formula_fit_intercept(rhs: str) -> bool:
    fit_intercept = True
    for sign, term in top_level_formula_terms(rhs):
        normalized = re.sub(r"\s+", "", term)
        if sign == "+" and normalized == "1":
            fit_intercept = True
        elif sign == "+" and normalized == "0":
            fit_intercept = False
        elif sign == "-" and normalized == "1":
            fit_intercept = False
    return fit_intercept


def formula_has_predictor_terms(rhs: str) -> bool:
    for _sign, term in top_level_formula_terms(rhs):
        normalized = re.sub(r"\s+", "", term)
        if normalized not in {"0", "1"}:
            return True
    return False


def strip_offset_terms(rhs: str) -> tuple[str, list[str]]:
    remaining = str(rhs or "")
    offset_terms: list[str] = []
    while True:
        match = find_offset_call(remaining)
        if match is None:
            break
        start, _open, end, expression = match
        if not expression:
            raise ValueError("offset(...) terms must contain an expression")
        if error := unsafe_formula_error(expression):
            raise ValueError(error)
        offset_terms.append(expression)
        remaining = remove_offset_span(remaining, start, end)
    return normalise_rhs_formula(remaining), offset_terms


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
    fitted_rhs, offset_terms = strip_offset_terms(rhs)
    fit_intercept = formula_fit_intercept(fitted_rhs)
    has_predictor_terms = formula_has_predictor_terms(fitted_rhs)
    intercept_only = fit_intercept and not has_predictor_terms
    if not fit_intercept and not has_predictor_terms:
        raise ValueError("GLM formula must include at least one predictor term or an intercept")
    return FormulaParts(
        raw_formula=raw,
        stripped_formula=stripped,
        response_column=response,
        rhs_formula=rhs,
        fitted_rhs_formula=fitted_rhs,
        fitted_formula=f"{TARGET_COLUMN} ~ {fitted_rhs}",
        offset_terms=offset_terms,
        fit_intercept=fit_intercept,
        has_predictor_terms=has_predictor_terms,
        intercept_only=intercept_only,
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
            "test_rows": 0,
            "non_training_rows": 0,
        }
    column_sql = quote_ident(sample_column)
    sql = f"""
SELECT
  SUM(CASE WHEN LOWER(TRIM(CAST({column_sql} AS VARCHAR))) = 'training' THEN 1 ELSE 0 END) AS training_rows,
  SUM(CASE WHEN LOWER(TRIM(CAST({column_sql} AS VARCHAR))) = 'test' THEN 1 ELSE 0 END) AS test_rows,
  SUM(CASE WHEN LOWER(TRIM(CAST({column_sql} AS VARCHAR))) != 'training' OR {column_sql} IS NULL THEN 1 ELSE 0 END) AS non_training_rows
FROM {dataset.relation_sql()}
"""
    with dataset.lock:
        row = dataset.con.execute(sql).fetchone()
    return {
        "available": True,
        "column": sample_column,
        "training_rows": int(row[0] or 0),
        "test_rows": int(row[1] or 0),
        "non_training_rows": int(row[2] or 0),
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
        regularization = regularization_config(payload)
    except ValueError as exc:
        regularization = regularization_config({"regularization": {"mode": "none"}})
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

    denominator_source = str(payload.get("denominator_source") or "dataset").strip() or "dataset"
    denominator_column = selected_denominator_column(payload)
    if denominator_source != "dataset":
        errors.append(
            "GLM building is unavailable while Denominator is a model prediction; "
            "use GBM init_score for prediction chaining"
        )
    elif denominator_column:
        if denominator_column not in valid_columns:
            errors.append(f"Choose a valid denominator column: {denominator_column}")
        elif denominator_column not in numeric_columns:
            errors.append("Choose a numeric denominator column")

    sample = sample_metadata(dataset)
    if training_scope in {"training", "training_test"}:
        if not sample["available"]:
            errors.append("Restricted training rows require a physical SAMPLE column")
        elif not sample["training_rows"]:
            errors.append("No rows have SAMPLE = training")
        elif training_scope == "training_test" and not sample["test_rows"]:
            errors.append("No rows have SAMPLE = test")

    result: dict[str, Any] = {
        "ok": not errors,
        "errors": errors,
        "warnings": warnings,
        "family": family,
        "family_parameter": family_param,
        "link": "auto",
        "regularization": regularization,
        "training_scope": training_scope,
        "response_column": response_column,
        "denominator_column": denominator_column,
        "denominator_source": denominator_source,
        "sample": sample,
    }
    if formula:
        errors.extend(formula_errors(formula.fitted_rhs_formula))
        warnings.extend(formula_warnings(formula.fitted_rhs_formula, formula.fit_intercept))
        result["formula"] = {
            "raw": formula.raw_formula,
            "stripped": formula.stripped_formula,
            "rhs": formula.rhs_formula,
            "fitted": formula.fitted_formula,
            "offset_terms": formula.offset_terms,
            "fit_intercept": formula.fit_intercept,
            "has_predictor_terms": formula.has_predictor_terms,
            "intercept_only": formula.intercept_only,
        }
    result["ok"] = not errors
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
