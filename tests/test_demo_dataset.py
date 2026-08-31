from __future__ import annotations

import csv
import unittest
from unittest.mock import patch

import duckdb

import py_lucidum
from py_lucidum import demo as demo_module
from py_lucidum.core import Dataset


class SingleChildTraversable:
    def __init__(self, parts: tuple[str, ...] = ()) -> None:
        self.parts = parts

    def joinpath(self, *children: str) -> "SingleChildTraversable":
        return SingleChildTraversable((*self.parts, *children))


class DemoDatasetTests(unittest.TestCase):
    def test_demo_dataset_resource_uses_multi_argument_joinpath(self) -> None:
        root = SingleChildTraversable()

        with patch("py_lucidum.demo.resources.files", return_value=root) as files_mock:
            resource = demo_module._demo_dataset_resource()

        files_mock.assert_called_once_with("py_lucidum")
        self.assertEqual(resource.parts, ("datasets", demo_module.DEMO_DATASET_NAME))

    def test_demo_spec_resource_uses_multi_argument_joinpath(self) -> None:
        root = SingleChildTraversable()

        with patch("py_lucidum.demo.resources.files", return_value=root) as files_mock:
            resource = demo_module._demo_spec_resource(demo_module.DEMO_FILTER_SPEC_NAME)

        files_mock.assert_called_once_with("py_lucidum")
        self.assertEqual(resource.parts, ("specs", demo_module.DEMO_FILTER_SPEC_NAME))

    def test_demo_dataset_path_exists_and_has_expected_columns(self) -> None:
        path = py_lucidum.demo_dataset_path()
        con = duckdb.connect(database=":memory:")

        row_count = con.execute("SELECT COUNT(*) FROM read_parquet(?)", [str(path)]).fetchone()[0]
        columns = {row[0] for row in con.execute("DESCRIBE SELECT * FROM read_parquet(?)", [str(path)]).fetchall()}

        self.assertTrue(path.exists())
        self.assertEqual(row_count, 50000)
        self.assertIn("PREMIUM", columns)
        self.assertIn("POSTCODE_AREA", columns)
        self.assertIn("POSTCODE_SECTOR", columns)
        self.assertIn("POSTCODE_UNIT", columns)
        self.assertIn("LATITUDE", columns)
        self.assertIn("LONGITUDE", columns)
        self.assertIn("SAMPLE", columns)
        self.assertNotIn("train_test", columns)
        sample_counts = dict(
            con.execute("SELECT SAMPLE, COUNT(*) FROM read_parquet(?) GROUP BY SAMPLE", [str(path)]).fetchall()
        )
        self.assertEqual(sample_counts, {"training": 30000, "test": 10000, "validation": 10000})

    def test_demo_spec_paths_exist_with_expected_names(self) -> None:
        paths = demo_module.demo_spec_paths()

        self.assertEqual(set(paths), {"filters", "kpis", "features"})
        self.assertEqual(paths["filters"].name, demo_module.DEMO_FILTER_SPEC_NAME)
        self.assertEqual(paths["kpis"].name, demo_module.DEMO_KPI_SPEC_NAME)
        self.assertEqual(paths["features"].name, demo_module.DEMO_FEATURE_SPEC_NAME)
        for path in paths.values():
            self.assertTrue(path.exists())

    def test_default_filter_spec_expressions_validate_against_demo_dataset(self) -> None:
        dataset = Dataset(py_lucidum.demo_dataset_path())
        filters_path = demo_module.demo_filter_spec_path()

        with filters_path.open(newline="", encoding="utf-8-sig") as handle:
            reader = csv.DictReader(handle)
            rows = list(reader)

        self.assertGreater(len(rows), 0)
        self.assertEqual(reader.fieldnames, ["theme", "name", "expression"])
        self.assertEqual(rows[0]["theme"], "SAMPLE")
        self.assertEqual(rows[0]["name"], "Training")
        self.assertEqual(rows[1]["theme"], "SAMPLE")
        self.assertEqual(rows[1]["name"], "Test")
        self.assertEqual(rows[2]["theme"], "SAMPLE")
        self.assertEqual(rows[2]["name"], "Validation")
        self.assertGreaterEqual(len({row["theme"] for row in rows}), 6)
        for row in rows:
            with self.subTest(theme=row["theme"], name=row["name"]):
                self.assertEqual(dataset.normalise_filter(row["expression"]), row["expression"])

    def test_default_kpi_spec_columns_validate_against_demo_dataset(self) -> None:
        dataset = Dataset(py_lucidum.demo_dataset_path())
        schema = dataset.schema()
        numeric_columns = {column["name"] for column in schema["columns"] if column["kind"] in {"integer", "numeric"}}
        kpis_path = demo_module.demo_kpi_spec_path()

        with kpis_path.open(newline="", encoding="utf-8-sig") as handle:
            reader = csv.DictReader(handle)
            rows = list(reader)

        self.assertGreater(len(rows), 0)
        self.assertEqual(reader.fieldnames, ["group", "name", "actual", "denominator", "decimals", "format"])
        self.assertGreaterEqual(len({row["group"] for row in rows}), 3)
        for row in rows:
            with self.subTest(group=row["group"], name=row["name"]):
                self.assertIn(row["actual"], numeric_columns)
                denominator = row["denominator"].strip()
                if denominator.upper() != "N" and denominator not in {"", "__none__", "Average row value"}:
                    self.assertIn(denominator, numeric_columns)
                self.assertGreaterEqual(int(row["decimals"]), 0)
                self.assertIn(row["format"], {"number", "currency", "percent"})

    def test_default_feature_spec_columns_validate_against_demo_dataset(self) -> None:
        dataset = Dataset(py_lucidum.demo_dataset_path())
        schema = dataset.schema()
        dataset_columns = {column["name"] for column in schema["columns"]}
        features_path = demo_module.demo_feature_spec_path()

        with features_path.open(newline="", encoding="utf-8-sig") as handle:
            reader = csv.DictReader(handle)
            rows = list(reader)

        self.assertGreater(len(rows), 0)
        self.assertEqual(
            reader.fieldnames,
            [
                "Feature", "Grouping", "Monotonicity", "Base", "min", "max", "banding",
                "chart_banding", "chart_quantiles", "chart_low_weights",
                "chart_missings", "chart_labels", "chart_sort",
                "chart_transform", "chart_sigma", "chart_date_bucket",
                "chart_empty_periods", "scenario1", "scenario2", "scenario3",
                "report_demo",
            ],
        )
        scenario_sets = {
            scenario: {
                row["Feature"]
                for row in rows
                if "feature" in row[scenario].strip().lower()
            }
            for scenario in ["scenario1", "scenario2", "scenario3"]
        }
        self.assertEqual(set(scenario_sets), {"scenario1", "scenario2", "scenario3"})
        self.assertEqual(len({frozenset(features) for features in scenario_sets.values()}), 3)
        self.assertEqual(
            {row["Feature"] for row in rows if "feature" in row["report_demo"].strip().lower()},
            {
                row["Feature"]
                for row in rows
                if row["Feature"]
                not in {
                    "LATITUDE",
                    "LONGITUDE",
                    "MAKE",
                    "POSTCODE_AREA",
                    "POSTCODE_SECTOR",
                }
            },
        )
        self.assertGreaterEqual(len({row["Grouping"] for row in rows if row["Grouping"]}), 3)
        for row in rows:
            with self.subTest(feature=row["Feature"]):
                self.assertIn(row["Feature"], dataset_columns)
                self.assertTrue(row["Base"].strip())
                tabulation_values = [row["min"].strip(), row["max"].strip(), row["banding"].strip()]
                if any(tabulation_values):
                    self.assertTrue(all(tabulation_values))
                    self.assertGreater(float(row["max"]), float(row["min"]))
                    self.assertGreater(float(row["banding"]), 0)


if __name__ == "__main__":
    unittest.main()
