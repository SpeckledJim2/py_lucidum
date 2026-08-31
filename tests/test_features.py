from __future__ import annotations

import os
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from py_lucidum.core import load_features, resolve_features_path


class FeatureSpecTests(unittest.TestCase):
    def test_feature_spec_parses_optional_monotonicity_aliases_before_scenarios(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "feature_spec.csv"
            path.write_text(
                "Feature,Grouping,Monotonicity,scenario1\n"
                "Age,DRIVER,increasing,feature\n"
                "Mileage,VEHICLE,-1,feature\n"
                "Premium,POLICY,,feature\n",
                encoding="utf-8",
            )

            spec = load_features(path)

        self.assertEqual([row["monotonicity"] for row in spec["rows"]], ["Increasing", "Decreasing", ""])
        self.assertEqual(spec["scenarios"], [{"name": "scenario1", "features": ["Age", "Mileage", "Premium"]}])

    def test_feature_spec_rejects_invalid_monotonicity(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "feature_spec.csv"
            path.write_text(
                "Feature,Grouping,Monotonicity,scenario1\nAge,DRIVER,flat,feature\n",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "Use Increasing, 1, Decreasing, -1, or blank"):
                load_features(path)

    def test_demo_feature_spec_has_expected_monotonicity_assignments(self) -> None:
        spec = load_features(Path(__file__).resolve().parents[1] / "specs" / "feature_spec.csv")
        actual = {
            row["feature"]: row.get("monotonicity", "")
            for row in spec["rows"]
            if row.get("monotonicity")
        }

        self.assertEqual(
            actual,
            {
                "POSTCODE_CATEGORY": "Increasing",
                "VEHICLE_CATEGORY": "Increasing",
                "PRIOR_CLAIMS": "Increasing",
                "NCD_YEARS": "Decreasing",
                "YEARS_LICENCE_HELD": "Decreasing",
                "YEARS_OWNED_VEHICLE": "Decreasing",
            },
        )

    def test_feature_spec_parses_groupings_and_case_insensitive_scenarios(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "feature_spec.csv"
            path.write_text(
                "Feature,Grouping,scenario1,scenario two\n"
                "Age,DRIVER,Feature,\n"
                "Segment,POSTCODE,,use as FEATURE\n"
                "Mileage,VEHICLE,,\n",
                encoding="utf-8",
            )

            spec = load_features(path)

        self.assertEqual(
            spec["rows"],
            [
                {"feature": "Age", "grouping": "DRIVER", "scenarios": {"scenario1": True, "scenario two": False}},
                {"feature": "Segment", "grouping": "POSTCODE", "scenarios": {"scenario1": False, "scenario two": True}},
                {"feature": "Mileage", "grouping": "VEHICLE", "scenarios": {"scenario1": False, "scenario two": False}},
            ],
        )
        self.assertEqual(
            spec["scenarios"],
            [
                {"name": "scenario1", "features": ["Age"]},
                {"name": "scenario two", "features": ["Segment"]},
            ],
        )

    def test_feature_spec_parses_optional_base_without_treating_it_as_a_scenario(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "feature_spec.csv"
            path.write_text(
                "Feature,Grouping,Base,scenario1,scenario two\n"
                "Age,DRIVER,40,Feature,\n"
                "Segment,POSTCODE,B,,use as FEATURE\n"
                "Mileage,VEHICLE,5000,,\n",
                encoding="utf-8",
            )

            spec = load_features(path)

        self.assertEqual(
            spec["rows"],
            [
                {"feature": "Age", "grouping": "DRIVER", "base": "40", "scenarios": {"scenario1": True, "scenario two": False}},
                {"feature": "Segment", "grouping": "POSTCODE", "base": "B", "scenarios": {"scenario1": False, "scenario two": True}},
                {"feature": "Mileage", "grouping": "VEHICLE", "base": "5000", "scenarios": {"scenario1": False, "scenario two": False}},
            ],
        )
        self.assertEqual(
            spec["scenarios"],
            [
                {"name": "scenario1", "features": ["Age"]},
                {"name": "scenario two", "features": ["Segment"]},
            ],
        )

    def test_feature_spec_parses_tabulation_metadata_before_scenarios(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "feature_spec.csv"
            path.write_text(
                "Feature,Grouping,Base,min,max,banding,scenario1,scenario two\n"
                "Age,DRIVER,40,17,96,1,Feature,\n"
                "Segment,POSTCODE,B,,,,,use as FEATURE\n"
                "Mileage,VEHICLE,5000,1000,30000,1000,,\n",
                encoding="utf-8",
            )

            spec = load_features(path)

        self.assertEqual(
            spec["rows"],
            [
                {
                    "feature": "Age",
                    "grouping": "DRIVER",
                    "base": "40",
                    "min": "17",
                    "max": "96",
                    "banding": "1",
                    "scenarios": {"scenario1": True, "scenario two": False},
                },
                {
                    "feature": "Segment",
                    "grouping": "POSTCODE",
                    "base": "B",
                    "min": "",
                    "max": "",
                    "banding": "",
                    "scenarios": {"scenario1": False, "scenario two": True},
                },
                {
                    "feature": "Mileage",
                    "grouping": "VEHICLE",
                    "base": "5000",
                    "min": "1000",
                    "max": "30000",
                    "banding": "1000",
                    "scenarios": {"scenario1": False, "scenario two": False},
                },
            ],
        )
        self.assertEqual(
            spec["scenarios"],
            [
                {"name": "scenario1", "features": ["Age"]},
                {"name": "scenario two", "features": ["Segment"]},
            ],
        )

    def test_feature_spec_parses_chart_metadata_before_scenarios(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "feature_spec.csv"
            path.write_text(
                "Feature,Grouping,Base,min,max,banding,chart_banding,chart_quantiles,chart_low_weights,"
                "chart_missings,chart_labels,chart_sort,chart_transform,chart_sigma,chart_date_bucket,"
                "chart_empty_periods,report_demo\n"
                "Age,DRIVER,40,17,96,1,5,10,0.1%,hide,line,volume,one,2,month,skip,Feature\n",
                encoding="utf-8",
            )

            spec = load_features(path)

        row = spec["rows"][0]
        self.assertEqual(row["chart_banding"], "5")
        self.assertEqual(row["chart_quantiles"], "10")
        self.assertEqual(row["chart_low_weights"], "0.1%")
        self.assertEqual(row["chart_missings"], "hide")
        self.assertEqual(row["chart_labels"], "line")
        self.assertEqual(row["chart_sort"], "volume")
        self.assertEqual(row["chart_transform"], "one")
        self.assertEqual(row["chart_sigma"], "2")
        self.assertEqual(row["chart_date_bucket"], "month")
        self.assertEqual(row["chart_empty_periods"], "skip")
        self.assertEqual(spec["scenarios"], [{"name": "report_demo", "features": ["Age"]}])

    def test_default_feature_spec_falls_back_to_specs_directory(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            specs_dir = root / "specs"
            specs_dir.mkdir()
            path = specs_dir / "feature_spec.csv"
            path.write_text("Feature,Grouping,scenario\nAge,DRIVER,feature\n", encoding="utf-8")
            previous_cwd = Path.cwd()
            try:
                os.chdir(root)
                self.assertEqual(resolve_features_path(None), path.resolve())
                spec = load_features(None)
            finally:
                os.chdir(previous_cwd)

        self.assertEqual(spec["scenarios"], [{"name": "scenario", "features": ["Age"]}])

    def test_missing_default_feature_spec_returns_empty_spec(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            previous_cwd = Path.cwd()
            try:
                os.chdir(tmp_dir)
                spec = load_features(None)
            finally:
                os.chdir(previous_cwd)

        self.assertEqual(spec, {"rows": [], "scenarios": []})

    def test_feature_specs_can_be_disabled(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "feature_spec.csv"
            path.write_text("Feature,Grouping,scenario\nAge,DRIVER,feature\n", encoding="utf-8")

            self.assertIsNone(resolve_features_path(path, use_features=False))
            self.assertEqual(load_features(path, use_features=False), {"rows": [], "scenarios": []})

    def test_missing_explicit_feature_spec_raises(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "missing_feature_spec.csv"

            with self.assertRaisesRegex(FileNotFoundError, "Feature specification file does not exist"):
                load_features(path)

    def test_feature_spec_rejects_bad_leading_columns(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "feature_spec.csv"
            path.write_text("feature,grouping,scenario\nAge,DRIVER,feature\n", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "Feature,Grouping"):
                load_features(path)


if __name__ == "__main__":
    unittest.main()
