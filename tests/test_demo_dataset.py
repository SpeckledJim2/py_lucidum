from __future__ import annotations

import csv
import unittest
from pathlib import Path
from unittest.mock import patch

import duckdb

import py_lucidum
from py_lucidum import demo as demo_module
from py_lucidum.core import Dataset


class SingleChildTraversable:
    def __init__(self, parts: tuple[str, ...] = ()) -> None:
        self.parts = parts

    def joinpath(self, child: str) -> "SingleChildTraversable":
        return SingleChildTraversable((*self.parts, child))


class DemoDatasetTests(unittest.TestCase):
    def test_demo_dataset_resource_uses_python39_compatible_joinpath(self) -> None:
        root = SingleChildTraversable()

        with patch("py_lucidum.demo.resources.files", return_value=root) as files_mock:
            resource = demo_module._demo_dataset_resource()

        files_mock.assert_called_once_with("py_lucidum")
        self.assertEqual(resource.parts, ("datasets", demo_module.DEMO_DATASET_NAME))

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

    def test_default_filter_spec_expressions_validate_against_demo_dataset(self) -> None:
        dataset = Dataset(py_lucidum.demo_dataset_path())
        filters_path = Path(__file__).parents[1] / "specs" / "filter_spec.csv"

        with filters_path.open(newline="", encoding="utf-8-sig") as handle:
            rows = list(csv.DictReader(handle))

        self.assertGreater(len(rows), 0)
        for row in rows:
            with self.subTest(name=row["name"]):
                self.assertEqual(dataset.normalise_filter(row["expression"]), row["expression"])


if __name__ == "__main__":
    unittest.main()
