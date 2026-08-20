from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any


LABEL_VALUES = {"none", "bar", "line", "all"}
SORT_VALUES = {"alpha", "volume", "actual", "expected", "shap"}
TRANSFORM_VALUES = {"none", "log", "exp", "logit", "zero", "one"}
MISSINGS_VALUES = {"show", "hide"}
DATE_BUCKET_VALUES = {"none", "hour", "day", "week", "month", "year"}
EMPTY_PERIOD_VALUES = {"show", "skip"}
SIGMA_VALUES = {0, 1, 2, 5}
LOW_WEIGHT_VALUES = {"0", "10", "100", "0.1%", "1%"}

CHART_CONTROL_COLUMNS = {
    "chart_banding": "banding",
    "chart_quantiles": "quantiles",
    "chart_low_weights": "low_weights",
    "chart_missings": "missings",
    "chart_labels": "labels",
    "chart_sort": "sort",
    "chart_transform": "transform",
    "chart_sigma": "sigma",
    "chart_date_bucket": "date_bucket",
    "chart_empty_periods": "empty_periods",
}

CHART_CONTROL_EDITOR_RULES = {
    "chart_banding": {"editor": "number", "minimum": 0},
    "chart_quantiles": {"editor": "number", "minimum": 0, "integer": True},
    "chart_low_weights": {"editor": "list", "values": ["0", "10", "100", "0.1%", "1%"]},
    "chart_missings": {"editor": "list", "values": sorted(MISSINGS_VALUES)},
    "chart_labels": {"editor": "list", "values": ["none", "bar", "line", "all"]},
    "chart_sort": {"editor": "list", "values": ["alpha", "volume", "actual", "expected", "shap"]},
    "chart_transform": {"editor": "list", "values": ["none", "log", "exp", "logit", "zero", "one"]},
    "chart_sigma": {"editor": "list", "values": [str(value) for value in sorted(SIGMA_VALUES)]},
    "chart_date_bucket": {"editor": "list", "values": ["none", "hour", "day", "week", "month", "year"]},
    "chart_empty_periods": {"editor": "list", "values": ["show", "skip"]},
}


def normalise_chart_controls(
    controls: Mapping[str, Any] | None,
    *,
    transform: str | None,
) -> dict[str, Any]:
    raw = dict(controls or {})
    known = {
        "banding",
        "quantiles",
        "low_weights",
        "missings",
        "labels",
        "sort",
        "transform",
        "sigma",
        "date_bucket",
        "empty_periods",
        "base",
    }
    unknown = sorted(set(raw) - known)
    if unknown:
        raise ValueError(f"Unknown chart controls: {', '.join(unknown)}")
    banding = _non_negative_number(raw.get("banding", 0), "banding")
    quantiles = _non_negative_integer(raw.get("quantiles", 0), "quantiles")
    sigma = _non_negative_integer(raw.get("sigma", 0), "sigma")
    if sigma not in SIGMA_VALUES:
        raise ValueError("sigma must be one of 0, 1, 2, or 5")
    low_weights = str(raw.get("low_weights", "0") or "0").strip()
    if low_weights not in LOW_WEIGHT_VALUES:
        raise ValueError("low_weights must be 0, 10, 100, 0.1%, or 1%")
    return {
        "banding": banding,
        "quantiles": quantiles,
        "low_weights": low_weights,
        "missings": _choice(raw.get("missings", "show"), MISSINGS_VALUES, "missings"),
        "labels": _choice(raw.get("labels", "none"), LABEL_VALUES, "labels"),
        "sort": _choice(raw.get("sort", "alpha"), SORT_VALUES, "sort"),
        "transform": _choice(transform if transform is not None else raw.get("transform", "none"), TRANSFORM_VALUES, "transform"),
        "sigma": sigma,
        "date_bucket": _choice(raw.get("date_bucket", "none"), DATE_BUCKET_VALUES, "date bucket"),
        "empty_periods": _choice(raw.get("empty_periods", "show"), EMPTY_PERIOD_VALUES, "empty periods"),
        "base": str(raw.get("base") or "").strip(),
    }


def validate_chart_control_value(control: str, value: Any) -> None:
    text = str(value or "").strip()
    if not text:
        return
    if control not in set(CHART_CONTROL_COLUMNS.values()):
        raise ValueError(f"Unknown chart control: {control}")
    normalise_chart_controls({control: text}, transform=None)


def _choice(value: Any, choices: set[str], label: str) -> str:
    text = str(value or "").strip().lower()
    if text not in choices:
        raise ValueError(f"Choose a valid {label}: {', '.join(sorted(choices))}")
    return text


def _non_negative_number(value: Any, label: str) -> int | float:
    if value is None or value == "":
        return 0
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be a non-negative number") from exc
    if not math.isfinite(number) or number < 0:
        raise ValueError(f"{label} must be a non-negative number")
    return int(number) if number.is_integer() else number


def _non_negative_integer(value: Any, label: str) -> int:
    number = _non_negative_number(value, label)
    if int(number) != number:
        raise ValueError(f"{label} must be a non-negative integer")
    return int(number)
