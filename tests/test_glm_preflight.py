from __future__ import annotations

import unittest
import warnings

from py_lucidum.tools.glm.preflight import check_terms
from py_lucidum.tools.glm.training import formula_context


class GlmPreflightTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        try:
            import numpy as np
            import polars as pl
            import tabmat as tm
            from glum import GeneralizedLinearRegressor
        except ImportError as exc:
            raise unittest.SkipTest(f"Missing optional GLM dependencies: {exc}") from exc
        cls.np, cls.pl, cls.tm = np, pl, tm
        cls.estimator_class = GeneralizedLinearRegressor

    def setUp(self) -> None:
        self.frame = self.pl.DataFrame({
            "x": [1.] * 12,
            "y": [1., 2., 4., 5., 7., 9., 12., 14., 16., 18., 21., 24.],
            "c": ["a", "b", "c"] * 4,
            "single": ["a"] * 12,
        }).with_columns((2 * self.pl.col("y") + 3).alias("z"))

    def check(self, formula, *, penalized=False, frame=None, block_bytes=128):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            warnings.simplefilter("ignore", RuntimeWarning)
            warnings.simplefilter("ignore", DeprecationWarning)
            return check_terms(
                formula, self.frame if frame is None else frame, formula_context(self.np),
                estimator=self.estimator_class(drop_first=not penalized),
                training_scope="all", penalized=penalized, condition_limit=1e12,
                block_bytes=block_bytes,
            )

    def test_conditional_output_not_constant_input_determines_validity(self):
        self.assertTrue(self.check("ifelse(x == 1, y, 0)")["ok"])
        duplicate = self.check("y + ifelse(x == 1, y, 0)")
        self.assertFalse(duplicate["ok"])
        issue = next(issue for issue in duplicate["issues"] if issue["code"] == "duplicate")
        self.assertEqual(set(issue["terms"] + issue["related_terms"]), {"y", "ifelse(x == 1, y, 0)"})
        zero = self.check("ifelse(x == 2, y, 0)")
        self.assertFalse(zero["ok"])
        self.assertEqual(zero["issues"][0]["code"], "zero")
        constant = self.check("ifelse(x == 1, 2, 0)")
        self.assertEqual(constant["issues"][0]["code"], "constant")

    def test_constants_and_intercepts(self):
        for formula in ("1", "0 + x"):
            with self.subTest(formula=formula):
                self.assertTrue(self.check(formula)["ok"])
        for formula, code in (("1 + x", "constant"), ("0 + x + I(2*x)", "constant"), ("0 + I(x-x)", "zero")):
            with self.subTest(formula=formula):
                result = self.check(formula)
                self.assertFalse(result["ok"])
                self.assertEqual(result["issues"][0]["code"], code)

    def test_linear_relationships_categoricals_interactions_and_splines(self):
        for formula, code in (
            ("y + I(2*y)", "proportional"),
            ("y + z", "dependent"),
            ("C(c) + ifelse(c == 'b', 1, 0)", "duplicate"),
            ("y + x:y", "duplicate"),
            ("ns(y, df=4)", "dependent"),
            ("bs(x, df=4)", "zero"),
        ):
            with self.subTest(formula=formula):
                result = self.check(formula)
                self.assertFalse(result["ok"])
                self.assertLess(result["rank"], result["columns"])
                self.assertIn(code, [issue["code"] for issue in result["issues"]])
                self.assertTrue(all(issue["terms"] for issue in result["issues"]))
        self.assertTrue(self.check('ns(y, df=4, constraints="center")')["ok"])
        empty = self.check("C(single)")
        self.assertTrue(empty["ok"])
        self.assertEqual(empty["issues"][0]["code"], "empty_term")
        self.assertEqual(empty["issues"][0]["severity"], "warning")

    def test_non_finite_and_formula_evaluation_errors_block_penalized_fits_too(self):
        for penalized in (False, True):
            for formula in ("np.log(-y)", "I(y / (x-x))"):
                with self.subTest(formula=formula, penalized=penalized):
                    result = self.check(formula, penalized=penalized)
                    self.assertFalse(result["ok"])
                    self.assertEqual(result["issues"][0]["code"], "non_finite")
                    self.assertEqual(result["issues"][0]["severity"], "error")
        result = self.check("unknown_feature")
        self.assertFalse(result["ok"])
        self.assertIn("unknown_feature", result["issues"][0]["reason"])
        self.assertFalse(self.check("C(c)", frame=self.frame.with_columns(self.pl.lit(None).alias("c")))["ok"])

    def test_penalized_dependencies_warn_without_blocking(self):
        result = self.check("x + y + I(2*y) + C(c)", penalized=True)
        self.assertTrue(result["ok"])
        self.assertLess(result["rank"], result["columns"])
        self.assertTrue(result["issues"])
        self.assertEqual({issue["severity"] for issue in result["issues"]}, {"warning"})

    def test_demo_problem_formulas_name_terms_and_offer_corrections(self):
        from py_lucidum import demo_dataset_path

        frame = self.pl.read_parquet(demo_dataset_path())
        self.assertEqual(frame.height, 50_000)
        cases = [
            ("ifelse(DRIVER_AGE >= 17, 1, 0)", "constant"),
            ("ifelse(DRIVER_AGE < 17, 1, 0)", "zero"),
            ("pmin(DRIVER_AGE, 0)", "zero"),
            ("pmax(DRIVER_AGE, 100)", "constant"),
            ("I(DRIVER_AGE / DRIVER_AGE)", "constant"),
            ("DRIVER_AGE + I(DRIVER_AGE)", "duplicate"),
            ("DRIVER_AGE + ifelse(DRIVER_AGE > 0, DRIVER_AGE, 0)", "duplicate"),
            ("DRIVER_AGE + pmin(DRIVER_AGE, 100)", "duplicate"),
            ("DRIVER_AGE + I(2 * DRIVER_AGE)", "proportional"),
            ("DRIVER_AGE + I(DRIVER_AGE + 5)", "dependent"),
            ("DRIVER_AGE + VEHICLE_AGE + I(DRIVER_AGE + VEHICLE_AGE)", "dependent"),
            ("ifelse(FUEL_TYPE == 'Diesel', 1, 0) + ifelse(FUEL_TYPE == 'Petrol', 1, 0)", "dependent"),
            ("C(FUEL_TYPE) + ifelse(FUEL_TYPE == 'Petrol', 1, 0)", "duplicate"),
            ("C(PRIOR_CLAIMS) + PRIOR_CLAIMS", "dependent"),
            ("ifelse(FUEL_TYPE == 'Diesel', 1, 0):ifelse(FUEL_TYPE == 'Petrol', 1, 0)", "zero"),
            ("DRIVER_AGE + DRIVER_AGE:ifelse(VEHICLE_AGE >= 0, 1, 0)", "duplicate"),
            ("ns(DRIVER_AGE, df=4)", "dependent"),
            ("DRIVER_AGE + poly(DRIVER_AGE, degree=2, raw=True)", "duplicate"),
            ("bs(pmin(DRIVER_AGE, 0), df=4)", "zero"),
            ("ns(pmin(DRIVER_AGE, 0), df=4)", "evaluation"),
            ('ns(PRIOR_CLAIMS, df=8, constraints="center")', "zero"),
            ("poly(pmin(DRIVER_AGE, 0), degree=2)", "non_finite"),
            ("I(1 / VEHICLE_AGE)", "non_finite"),
            ("sqrt(-DRIVER_AGE)", "non_finite"),
            ("log(pmin(DRIVER_AGE, 0))", "non_finite"),
            ("exp(CAR_VALUE)", "non_finite"),
            ("DRIVER_AGGE", "evaluation"),
            ("I(DRIVER_AGE * 1e-14)", "ill_conditioned"),
            ("DRIVER_AGE + I(DRIVER_AGE + 1e-12 * VEHICLE_AGE)", "proportional"),
        ]
        for formula, code in cases:
            for penalized in (False, True):
                with self.subTest(formula=formula, penalized=penalized):
                    report = self.check(formula, frame=frame, penalized=penalized, block_bytes=32 * 1024 * 1024)
                    self.assertEqual(report["ok"], penalized and code not in {"evaluation", "non_finite"})
                    codes = {issue["code"] for issue in report["issues"]}
                    self.assertTrue(codes & ({"dependent", "duplicate"} if penalized and code == "duplicate" else {code}))
                    for issue in report["issues"]:
                        self.assertNotEqual(issue["terms"], ["Formula"])
                        self.assertTrue(issue["reason"])
                        self.assertTrue(issue["suggestion"])
        for formula in (
            "1", "0 + ifelse(DRIVER_AGE > 0, 1, 0)", "DRIVER_AGE + DRIVER_AGE",
            "DRIVER_AGE + C(FUEL_TYPE)", 'ns(DRIVER_AGE, df=4, constraints="center")',
            "0 + ns(DRIVER_AGE, df=4)", "bs(DRIVER_AGE, df=4, include_intercept=True)",
        ):
            with self.subTest(valid_formula=formula):
                report = self.check(formula, frame=frame, block_bytes=32 * 1024 * 1024)
                self.assertTrue(report["ok"])
                self.assertEqual(report["issues"], [])

    def test_degenerate_spline_messages_explain_distinct_values_and_degrees(self):
        result = self.check("ns(x, df=4)")
        self.assertEqual(result["issues"][0]["terms"], ["ns(x, df=4)"])
        self.assertIn("too few distinct values", result["issues"][0]["reason"])
        self.assertIn("reduce df", result["issues"][0]["suggestion"])
        self.assertNotIn("n_inner_knots", result["issues"][0]["reason"])
        result = self.check("bs(x, df=4)")
        self.assertTrue(all("spline degrees" in issue["suggestion"] for issue in result["issues"] if issue["code"] == "zero"))
        result = self.check("poly(x, degree=2)")
        self.assertIn("input varies", result["issues"][0]["suggestion"])

    def test_extreme_finite_scales_are_checked_without_overflow_or_mutating_inputs(self):
        frame = self.pl.DataFrame({"y": [float(i) for i in range(1, 101)] * 500})
        for factor in (1e305, 1e-320):
            scaled = frame.with_columns((self.pl.col("y") * factor).alias("extreme"))
            for block_bytes in (8192, 32 * 1024 * 1024):
                with self.subTest(factor=factor, block_bytes=block_bytes):
                    result = self.check("extreme", frame=scaled, block_bytes=block_bytes)
                    self.assertFalse(result["ok"])
                    self.assertEqual(result["rank"], 2)
                    self.assertEqual([issue["code"] for issue in result["issues"]], ["ill_conditioned"])
                    no_intercept = self.check("0 + extreme", frame=scaled, block_bytes=block_bytes)
                    self.assertTrue(no_intercept["ok"])
                    self.assertAlmostEqual(no_intercept["condition_number"], 1)
                    duplicate = self.check("extreme + I(extreme)", frame=scaled, block_bytes=block_bytes)
                    self.assertIn("duplicate", [issue["code"] for issue in duplicate["issues"]])
            self.assertTrue(scaled.equals(frame.with_columns((self.pl.col("y") * factor).alias("extreme"))))

    def test_encoder_failure_identifies_transformed_category_and_explains_workaround(self):
        for penalized in (False, True):
            result = self.check("y + C(ifelse(y > 10, 1, 0))", penalized=penalized)
            self.assertFalse(result["ok"])
            issue = result["issues"][0]
            self.assertEqual(issue["code"], "evaluation")
            self.assertEqual(issue["terms"], ["C(ifelse(y > 10, 1, 0))"])
            self.assertIn("categorical transform", issue["reason"])
            self.assertIn("without C", issue["suggestion"])
            self.assertNotIn("is_numeric", issue["reason"])

    def test_automatically_reduced_interactions_and_centered_cyclic_splines_remain_valid(self):
        for formula in ('C(c):y + y', 'cs(y, df=4, constraints="center")'):
            with self.subTest(formula=formula):
                report = self.check(formula)
                self.assertTrue(report["ok"])
                self.assertEqual(report["issues"], [])
        result = self.check("cs(y, df=4)")
        self.assertIn('constraints="center"', result["issues"][0]["suggestion"])

    def test_chunked_checks_match_dense_checks_and_keep_rare_rows(self):
        for formula in ("y + C(c)", "y + z", "y + ifelse(x == 1, y, 0)", "y + C(c):y"):
            with self.subTest(formula=formula):
                small = self.check(formula, block_bytes=128)
                large = self.check(formula, block_bytes=1024 * 1024)
                self.assertEqual(small["rank"], large["rank"])
                matrix = self.tm.from_formula(formula, self.frame, include_intercept=True, ensure_full_rank=True, add_column_for_intercept=True, context=formula_context(self.np))
                self.assertEqual(small["rank"], self.np.linalg.matrix_rank(matrix.toarray()))
                if small["rank"] == small["columns"]:
                    self.assertAlmostEqual(small["condition_number"], self.np.linalg.cond(matrix.toarray()), places=9)
        frame = self.frame.with_columns(self.pl.Series("rare", [0.] * 11 + [1.]))
        self.assertTrue(self.check("y + rare", frame=frame)["ok"])
        frame = self.frame.with_columns((self.pl.col("y") * 1e-14).alias("small"))
        result = self.check("small", frame=frame)
        self.assertEqual(result["rank"], result["columns"])
        self.assertEqual(result["issues"][0]["code"], "ill_conditioned")
        self.assertTrue(self.check("small", frame=frame, penalized=True)["ok"])
        frame = self.pl.DataFrame({"y": list(range(24)), "c": [f"level{i % 8}" for i in range(24)]})
        self.assertTrue(self.check("y + C(c)", frame=frame)["ok"])


if __name__ == "__main__":
    unittest.main()
