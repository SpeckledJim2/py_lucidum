from __future__ import annotations

import unittest

import numpy as np

from py_lucidum.model_metrics import normalized_gini, split_gini_metrics


def weighted_auc(actual: np.ndarray, prediction: np.ndarray, weight: np.ndarray) -> float:
    positive = weight * actual
    negative = weight * (1.0 - actual)
    order = np.argsort(prediction, kind="stable")
    prediction = prediction[order]
    positive = positive[order]
    negative = negative[order]
    concordance = 0.0
    prior_negative = 0.0
    for score in np.unique(prediction):
        mask = prediction == score
        group_positive = float(positive[mask].sum())
        group_negative = float(negative[mask].sum())
        concordance += group_positive * (prior_negative + 0.5 * group_negative)
        prior_negative += group_negative
    return concordance / (float(positive.sum()) * float(negative.sum()))


class NormalizedGiniTests(unittest.TestCase):
    def test_perfect_reversed_and_flat_rankings(self) -> None:
        actual = np.asarray([0.0, 1.0, 2.0, 5.0])
        self.assertAlmostEqual(normalized_gini(np, actual, actual).value, 1.0)
        self.assertAlmostEqual(normalized_gini(np, actual, -actual).value, -1.0)
        self.assertAlmostEqual(
            normalized_gini(np, actual, np.ones(actual.size)).value,
            0.0,
        )

    def test_prediction_ties_are_aggregated(self) -> None:
        actual = np.asarray([0.0, 2.0, 1.0, 5.0])
        prediction = np.asarray([0.0, 1.0, 1.0, 2.0])
        first = normalized_gini(np, actual, prediction).value
        second = normalized_gini(
            np,
            actual[[0, 2, 1, 3]],
            prediction[[0, 2, 1, 3]],
        ).value
        self.assertAlmostEqual(first, second)

    def test_exposure_weighting_changes_the_result(self) -> None:
        actual = np.asarray([0.0, 4.0, 1.0, 3.0])
        prediction = np.asarray([0.0, 3.0, 2.0, 1.0])
        unweighted = normalized_gini(np, actual, prediction).value
        weighted = normalized_gini(
            np,
            actual,
            prediction,
            np.asarray([1.0, 1.0, 20.0, 1.0]),
        ).value
        self.assertNotAlmostEqual(unweighted, weighted)

    def test_rate_conversion_uses_denominator_as_weight(self) -> None:
        response = np.asarray([0.0, 3.0, 2.0, 8.0])
        prediction = np.asarray([0.5, 2.5, 5.0, 6.0])
        denominator = np.asarray([1.0, 2.0, 5.0, 4.0])
        result = normalized_gini(
            np,
            response / denominator,
            prediction / denominator,
            denominator,
        )
        self.assertIsNotNone(result.value)
        self.assertNotAlmostEqual(
            result.value,
            normalized_gini(np, response, prediction).value,
        )

    def test_rank_preserving_transformation_does_not_change_gini(self) -> None:
        actual = np.asarray([0.0, 4.0, 1.0, 3.0, 2.0])
        prediction = np.asarray([-2.0, 1.0, -1.0, 0.5, 0.0])
        baseline = normalized_gini(np, actual, prediction).value
        transformed = normalized_gini(np, actual, np.exp(prediction)).value
        self.assertAlmostEqual(baseline, transformed)

    def test_binary_gini_equals_weighted_two_auc_minus_one(self) -> None:
        actual = np.asarray([0.0, 1.0, 0.0, 1.0, 1.0, 0.0])
        prediction = np.asarray([0.2, 0.8, 0.4, 0.4, 0.9, 0.1])
        weight = np.asarray([1.0, 2.0, 3.0, 4.0, 1.5, 0.5])
        gini = normalized_gini(np, actual, prediction, weight).value
        self.assertAlmostEqual(
            gini,
            2.0 * weighted_auc(actual, prediction, weight) - 1.0,
        )

    def test_undefined_inputs_return_reasons(self) -> None:
        cases = (
            ([], [], None, None),
            ([1.0], [1.0], None, "at least two"),
            ([1.0, 1.0], [0.0, 1.0], None, "non-constant"),
            ([0.0, 0.0], [0.0, 1.0], None, "positive"),
            ([0.0, -1.0], [0.0, 1.0], None, "non-negative"),
            ([0.0, 1.0], [0.0, np.nan], None, "at least two"),
            ([0.0, 1.0], [0.0, 1.0], [0.0, np.nan], "at least two"),
            ([1.0, 1.0], [0.0, 1.0], None, "non-constant"),
        )
        for actual, prediction, weight, reason in cases:
            with self.subTest(actual=actual, prediction=prediction, weight=weight):
                result = normalized_gini(np, actual, prediction, weight)
                self.assertIsNone(result.value)
                if reason is None:
                    self.assertFalse(result.present)
                else:
                    self.assertIn(reason, result.reason)

    def test_invalid_rows_are_excluded_when_enough_rows_remain(self) -> None:
        result = normalized_gini(
            np,
            [0.0, 1.0, np.nan, 10.0],
            [0.0, 1.0, 5.0, np.inf],
            [1.0, 1.0, 1.0, -2.0],
        )
        self.assertEqual(result.value, 1.0)

    def test_split_mapping_and_no_sample_fallback(self) -> None:
        actual = np.asarray([0.0, 1.0, 0.0, 1.0, 0.0, 1.0])
        prediction = np.asarray([0.0, 1.0, 1.0, 0.0, 0.5, 0.5])
        roles = np.asarray(
            ["training", "training", "test", "test", "validation", "validation"]
        )
        metrics, warnings = split_gini_metrics(
            np,
            actual=actual,
            prediction=prediction,
            sample_roles=roles,
        )
        self.assertEqual(metrics, {"gini_tr": 1.0, "gini_te": -1.0, "gini_vl": 0.0})
        self.assertEqual(warnings, [])

        metrics, warnings = split_gini_metrics(
            np,
            actual=actual,
            prediction=prediction,
        )
        self.assertIsNotNone(metrics["gini_tr"])
        self.assertIsNone(metrics["gini_te"])
        self.assertIsNone(metrics["gini_vl"])
        self.assertEqual(warnings, [])

    def test_present_undefined_split_warns_but_absent_split_does_not(self) -> None:
        metrics, warnings = split_gini_metrics(
            np,
            actual=[1.0, 1.0, 0.0],
            prediction=[0.0, 1.0, 0.5],
            sample_roles=["training", "training", "validation"],
        )
        self.assertIsNone(metrics["gini_tr"])
        self.assertIsNone(metrics["gini_te"])
        self.assertIsNone(metrics["gini_vl"])
        self.assertEqual(len(warnings), 2)
        self.assertIn("Training Gini", warnings[0])
        self.assertIn("Validation Gini", warnings[1])


if __name__ == "__main__":
    unittest.main()
