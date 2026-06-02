from __future__ import annotations

import os
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from py_lucidum.core import load_features, resolve_features_path


class FeatureSpecTests(unittest.TestCase):
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
