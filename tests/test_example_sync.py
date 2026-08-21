from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from py_lucidum.example_sync import (
    EXAMPLE_SCRIPT_NAMES,
    format_example_sync_result,
    sync_example_scripts,
)


class ExampleSyncTests(unittest.TestCase):
    def _create_source(self, root: Path, suffix: str = "original") -> Path:
        source = root / "source"
        source.mkdir()
        for name in EXAMPLE_SCRIPT_NAMES:
            source.joinpath(name).write_text(f"# {name}\n{suffix}\n", encoding="utf-8")
        return source

    def test_fresh_sync_copies_only_maintained_python_scripts(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            source = self._create_source(root)
            destination = root / "client"
            destination.mkdir()
            config = destination / "config_glm.yaml"
            custom_script = destination / "client_custom.py"
            config.write_text("client: config\n", encoding="utf-8")
            custom_script.write_text("# client-owned\n", encoding="utf-8")

            with patch("py_lucidum.example_sync._example_source_root", return_value=source):
                result = sync_example_scripts(destination)

            self.assertEqual(result.created, EXAMPLE_SCRIPT_NAMES)
            self.assertEqual(result.updated, ())
            self.assertEqual(result.unchanged, ())
            for name in EXAMPLE_SCRIPT_NAMES:
                self.assertEqual(destination.joinpath(name).read_bytes(), source.joinpath(name).read_bytes())
            self.assertEqual(config.read_text(encoding="utf-8"), "client: config\n")
            self.assertEqual(custom_script.read_text(encoding="utf-8"), "# client-owned\n")

    def test_sync_overwrites_changed_managed_script_and_is_idempotent(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            source = self._create_source(root)
            destination = root / "client"

            with patch("py_lucidum.example_sync._example_source_root", return_value=source):
                sync_example_scripts(destination)
                changed_name = EXAMPLE_SCRIPT_NAMES[0]
                destination.joinpath(changed_name).write_text("# client edit\n", encoding="utf-8")
                updated = sync_example_scripts(destination)
                unchanged = sync_example_scripts(destination)

            self.assertEqual(updated.updated, (changed_name,))
            self.assertEqual(len(updated.unchanged), len(EXAMPLE_SCRIPT_NAMES) - 1)
            self.assertEqual(destination.joinpath(changed_name).read_bytes(), source.joinpath(changed_name).read_bytes())
            self.assertEqual(unchanged.created, ())
            self.assertEqual(unchanged.updated, ())
            self.assertEqual(unchanged.unchanged, EXAMPLE_SCRIPT_NAMES)

    def test_dry_run_lists_changes_without_writing(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            source = self._create_source(root)
            destination = root / "client"

            with patch("py_lucidum.example_sync._example_source_root", return_value=source):
                sync_example_scripts(destination)
                update_name = EXAMPLE_SCRIPT_NAMES[0]
                create_name = EXAMPLE_SCRIPT_NAMES[1]
                destination.joinpath(update_name).write_text("# client edit\n", encoding="utf-8")
                destination.joinpath(create_name).unlink()
                result = sync_example_scripts(destination, dry_run=True)

            self.assertEqual(result.created, (create_name,))
            self.assertEqual(result.updated, (update_name,))
            self.assertEqual(destination.joinpath(update_name).read_text(encoding="utf-8"), "# client edit\n")
            self.assertFalse(destination.joinpath(create_name).exists())
            output = format_example_sync_result(result)
            self.assertIn(f"create: {create_name}", output)
            self.assertIn(f"update: {update_name}", output)
            self.assertIn("9 unchanged", output)

    def test_sync_rejects_non_directory_destination(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            source = self._create_source(root)
            destination = root / "client"
            destination.write_text("not a directory\n", encoding="utf-8")

            with (
                patch("py_lucidum.example_sync._example_source_root", return_value=source),
                self.assertRaisesRegex(NotADirectoryError, "not a directory"),
            ):
                sync_example_scripts(destination)


if __name__ == "__main__":
    unittest.main()
