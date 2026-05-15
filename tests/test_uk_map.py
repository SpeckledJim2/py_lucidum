from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from py_lucidum.app import create_app, normalise_tools
from py_lucidum.core import Dataset
from py_lucidum.tools.uk_map.query import summary


class UkMapToolTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.data_path = self.root / "sample.csv"
        self.data_path.write_text(
            "PostcodeArea,PostcodeSector,PostcodeUnit,CustomArea,CustomUnit,lat,long,CustomLat,CustomLong,Actual,Weight,Flag\n"
            "AB,AB10 1,AB10 1AA,A1,U1,57.1,-2.1,58.1,-3.1,100,10,1\n"
            "AB,AB10 1,AB10 1AA,A1,U1,57.3,-2.3,58.3,-3.3,200,20,1\n"
            "AL,AL1 1,AL1 1AA,A2,U2,51.7,-0.4,52.7,-1.4,300,30,1\n"
            "AL,AL1 2,AL1 2AA,A2,U3,51.8,-0.3,52.8,-1.3,400,,0\n"
            ",,ZZ1 1ZZ,A3,U4,999,-2,53,-1,500,50,1\n",
            encoding="utf-8",
        )

    def request(self, **overrides: object) -> dict[str, object]:
        request: dict[str, object] = {
            "level": "area",
            "numerator": "Actual",
            "denominator": "__none__",
            "filter": "",
        }
        request.update(overrides)
        return request

    def test_default_tools_include_uk_map(self) -> None:
        self.assertEqual(normalise_tools(None), ["line_bar", "uk_map"])

        app = create_app(self.data_path, token="dev-token")
        paths = {route.path for route in app.routes}

        self.assertEqual(app.state.enabled_tools, ["line_bar", "uk_map"])
        self.assertIn("/api/uk-map/summary", paths)
        self.assertIn("/tools/uk-map/static", paths)

    def test_create_app_persists_unit_point_defaults(self) -> None:
        app = create_app(
            self.data_path,
            defaults={
                "postcode_unit": "CustomUnit",
                "latitude": "CustomLat",
                "longitude": "CustomLong",
            },
        )

        self.assertEqual(app.state.defaults["postcode_unit"], "CustomUnit")
        self.assertEqual(app.state.defaults["latitude"], "CustomLat")
        self.assertEqual(app.state.defaults["longitude"], "CustomLong")

    def test_area_summary_uses_average_row_value(self) -> None:
        dataset = Dataset(self.data_path)
        result = summary(dataset, self.request())

        self.assertEqual(result["level"], "area")
        self.assertEqual(result["join_column"], "PostcodeArea")
        self.assertEqual(result["join_property"], "PostcodeArea")
        self.assertEqual(result["row_count"], 5)
        self.assertEqual(result["filtered_row_count"], 5)
        self.assertEqual(result["response"]["value"], 300)
        self.assertEqual(result["denominator"]["value"], 5)
        self.assertEqual([(row["key"], row["value"], row["denominator"]) for row in result["rows"]], [("AB", 150, 2), ("AL", 350, 2)])

    def test_sector_summary_applies_filter_and_weight(self) -> None:
        dataset = Dataset(self.data_path)
        result = summary(
            dataset,
            self.request(level="sector", denominator="Weight", filter="PostcodeArea = 'AL'"),
        )

        self.assertEqual(result["join_column"], "PostcodeSector")
        self.assertEqual(result["filtered_row_count"], 2)
        self.assertEqual(result["denominator"]["value"], 30)
        self.assertEqual(
            [(row["key"], row["value"], row["denominator"]) for row in result["rows"]],
            [("AL1 1", 10, 30), ("AL1 2", None, 0)],
        )
        self.assertIn("1 row excluded from Weight because Weight was missing.", result["warnings"])

    def test_custom_postcode_column_default(self) -> None:
        dataset = Dataset(self.data_path)
        result = summary(
            dataset,
            self.request(),
            defaults={"postcode_area": "CustomArea", "postcode_sector": "PostcodeSector"},
        )

        self.assertEqual(result["join_column"], "CustomArea")
        self.assertEqual([row["key"] for row in result["rows"]], ["A1", "A2", "A3"])

    def test_invalid_postcode_column_is_reported(self) -> None:
        dataset = Dataset(self.data_path)

        with self.assertRaisesRegex(ValueError, "Choose a valid area postcode column"):
            summary(dataset, self.request(areaColumn="MissingArea"))

    def test_unit_summary_uses_average_row_value_and_coordinates(self) -> None:
        dataset = Dataset(self.data_path)
        result = summary(dataset, self.request(level="unit"))

        self.assertEqual(result["level"], "unit")
        self.assertEqual(result["join_column"], "PostcodeUnit")
        self.assertEqual(result["join_property"], "PostcodeUnit")
        self.assertEqual(result["point_summary"], {
            "summary_count": 4,
            "plotted_count": 3,
            "missing_value_count": 0,
            "missing_coordinate_count": 1,
        })
        self.assertEqual(
            [(row["key"], row["value"], row["denominator"], row["latitude"], row["longitude"]) for row in result["rows"]],
            [
                ("AB10 1AA", 150, 2, 57.2, -2.2),
                ("AL1 1AA", 300, 1, 51.7, -0.4),
                ("AL1 2AA", 400, 1, 51.8, -0.3),
            ],
        )

    def test_unit_summary_applies_filter_and_weight(self) -> None:
        dataset = Dataset(self.data_path)
        result = summary(
            dataset,
            self.request(level="unit", denominator="Weight", filter="PostcodeArea = 'AL'"),
        )

        self.assertEqual(result["filtered_row_count"], 2)
        self.assertEqual(result["point_summary"], {
            "summary_count": 2,
            "plotted_count": 1,
            "missing_value_count": 1,
            "missing_coordinate_count": 0,
        })
        self.assertEqual(
            [(row["key"], row["value"], row["denominator"], row["latitude"], row["longitude"]) for row in result["rows"]],
            [("AL1 1AA", 10, 30, 51.7, -0.4)],
        )
        self.assertIn("1 row excluded from Weight because Weight was missing.", result["warnings"])

    def test_custom_unit_point_column_defaults(self) -> None:
        dataset = Dataset(self.data_path)
        result = summary(
            dataset,
            self.request(level="unit"),
            defaults={
                "postcode_unit": "CustomUnit",
                "latitude": "CustomLat",
                "longitude": "CustomLong",
            },
        )

        self.assertEqual(result["join_column"], "CustomUnit")
        self.assertEqual([row["key"] for row in result["rows"]], ["U1", "U2", "U3", "U4"])
        self.assertEqual(result["rows"][0]["latitude"], 58.2)
        self.assertEqual(result["rows"][0]["longitude"], -3.2)

    def test_invalid_unit_point_columns_are_reported(self) -> None:
        dataset = Dataset(self.data_path)

        with self.assertRaisesRegex(ValueError, "Choose a valid unit postcode column"):
            summary(dataset, self.request(level="unit", unitColumn="MissingUnit"))

        with self.assertRaisesRegex(ValueError, "Choose a valid numeric latitude column"):
            summary(dataset, self.request(level="unit", latitudeColumn="PostcodeArea"))

        with self.assertRaisesRegex(ValueError, "Choose a valid numeric longitude column"):
            summary(dataset, self.request(level="unit", longitudeColumn="PostcodeArea"))


if __name__ == "__main__":
    unittest.main()
