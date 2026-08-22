from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any


GINI_FIELDS = ("gini_tr", "gini_te", "gini_vl")
GINI_SPLITS = (
    ("training", "gini_tr", "Training"),
    ("test", "gini_te", "Test"),
    ("validation", "gini_vl", "Validation"),
)
GINI_ZERO_TOLERANCE = 1e-12


@dataclass(frozen=True)
class GiniResult:
    value: float | None
    reason: str | None = None
    present: bool = True


def normalized_gini(
    np: Any,
    actual: Any,
    prediction: Any,
    weight: Any | None = None,
) -> GiniResult:
    """Return an exposure-weighted normalized Lorenz Gini.

    ``actual`` and ``prediction`` are rates when ``weight`` is supplied. Exact
    score ties are aggregated before integrating the concentration curve.
    """

    actual_values = np.asarray(actual, dtype="float64")
    prediction_values = np.asarray(prediction, dtype="float64")
    if actual_values.shape != prediction_values.shape:
        raise ValueError("Gini actuals and predictions must have the same shape")
    if actual_values.ndim != 1:
        actual_values = actual_values.reshape(-1)
        prediction_values = prediction_values.reshape(-1)
    if actual_values.size == 0:
        return GiniResult(None, present=False)

    if weight is None:
        weights = np.ones(actual_values.size, dtype="float64")
    else:
        weights = np.asarray(weight, dtype="float64")
        if weights.shape != np.asarray(actual).shape:
            raise ValueError("Gini weights must have the same shape as actuals")
        weights = weights.reshape(-1)

    usable = (
        np.isfinite(actual_values)
        & np.isfinite(prediction_values)
        & np.isfinite(weights)
        & (weights > 0)
    )
    if int(usable.sum()) < 2:
        return GiniResult(
            None,
            "needs at least two rows with finite actuals and predictions and positive weights",
        )

    actual_values = actual_values[usable]
    prediction_values = prediction_values[usable]
    weights = weights[usable]
    if bool((actual_values < 0).any()):
        return GiniResult(None, "actual values must be non-negative")

    total_actual = float(np.sum(actual_values * weights))
    if not math.isfinite(total_actual) or total_actual <= 0:
        return GiniResult(None, "total actual must be positive")

    perfect_gini = _concentration_gini(np, actual_values, actual_values, weights)
    if not math.isfinite(perfect_gini) or abs(perfect_gini) <= GINI_ZERO_TOLERANCE:
        return GiniResult(None, "actual values do not provide a non-constant perfect ranking")

    model_gini = _concentration_gini(np, actual_values, prediction_values, weights)
    value = model_gini / perfect_gini
    if not math.isfinite(value):
        return GiniResult(None, "calculation produced a non-finite value")
    if abs(value) <= GINI_ZERO_TOLERANCE:
        value = 0.0
    value = min(1.0, max(-1.0, float(value)))
    return GiniResult(value)


def split_gini_metrics(
    np: Any,
    *,
    actual: Any,
    prediction: Any,
    weight: Any | None = None,
    sample_roles: Any | None = None,
) -> tuple[dict[str, float | None], list[str]]:
    """Calculate normalized Gini for Lucidum's three SAMPLE roles."""

    actual_values = np.asarray(actual, dtype="float64").reshape(-1)
    prediction_values = np.asarray(prediction, dtype="float64").reshape(-1)
    if actual_values.shape != prediction_values.shape:
        raise ValueError("Gini actuals and predictions must have the same shape")
    if weight is None:
        weights = None
    else:
        weights = np.asarray(weight, dtype="float64").reshape(-1)
        if weights.shape != actual_values.shape:
            raise ValueError("Gini weights must have the same shape as actuals")

    if sample_roles is None:
        roles = None
    else:
        roles = np.asarray(sample_roles).reshape(-1)
        if roles.shape != actual_values.shape:
            raise ValueError("Gini SAMPLE roles must have the same shape as actuals")

    metrics = {field: None for field in GINI_FIELDS}
    warnings: list[str] = []
    for role, field, label in GINI_SPLITS:
        if roles is None:
            if role != "training":
                continue
            mask = np.ones(actual_values.size, dtype=bool)
        else:
            mask = roles == role
        if not bool(mask.any()):
            continue
        result = normalized_gini(
            np,
            actual_values[mask],
            prediction_values[mask],
            weights[mask] if weights is not None else None,
        )
        metrics[field] = result.value
        if result.reason:
            warnings.append(f"{label} Gini could not be calculated: {result.reason}")
    return metrics, warnings


def _concentration_gini(np: Any, actual: Any, score: Any, weight: Any) -> float:
    order = np.argsort(np.asarray(score, dtype="float64"), kind="stable")
    ordered_score = np.asarray(score, dtype="float64")[order]
    ordered_weight = np.asarray(weight, dtype="float64")[order]
    ordered_actual_weight = np.asarray(actual, dtype="float64")[order] * ordered_weight

    group_starts = np.flatnonzero(
        np.concatenate(
            (
                np.asarray([True], dtype=bool),
                ordered_score[1:] != ordered_score[:-1],
            )
        )
    )
    group_weight = np.add.reduceat(ordered_weight, group_starts)
    group_actual = np.add.reduceat(ordered_actual_weight, group_starts)
    cumulative_weight = np.cumsum(group_weight) / float(np.sum(group_weight))
    cumulative_actual = np.cumsum(group_actual) / float(np.sum(group_actual))
    previous_weight = np.concatenate((np.asarray([0.0]), cumulative_weight[:-1]))
    previous_actual = np.concatenate((np.asarray([0.0]), cumulative_actual[:-1]))
    trapezoid_sum = float(
        np.sum(
            (cumulative_weight - previous_weight)
            * (cumulative_actual + previous_actual)
        )
    )
    return 1.0 - trapezoid_sum


__all__ = [
    "GINI_FIELDS",
    "GINI_SPLITS",
    "GiniResult",
    "normalized_gini",
    "split_gini_metrics",
]
