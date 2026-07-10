from __future__ import annotations

import io
import os
import subprocess
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from scripts import run_tests


class TestRunnerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.root = Path(__file__).resolve().parents[1]

    def test_dev_targets_exclude_only_browser_and_glm_modules(self) -> None:
        names = {Path(target).name for target in run_tests.dev_test_targets(self.root)}

        self.assertNotIn("test_browser_smoke.py", names)
        self.assertNotIn("test_glm.py", names)
        self.assertIn("test_gbm.py", names)
        self.assertIn("test_pipx_install.py", names)
        self.assertIn("test_static_assets.py", names)
        self.assertIn("test_test_runner.py", names)

    def test_dev_targets_include_new_test_modules_automatically(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            tests_dir = root / "tests"
            tests_dir.mkdir()
            for name in ("test_browser_smoke.py", "test_glm.py", "test_new_tool.py"):
                (tests_dir / name).touch()

            self.assertEqual(run_tests.dev_test_targets(root), ["tests/test_new_tool.py"])

    def test_javascript_files_exclude_vendor_files(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            static = root / "src" / "py_lucidum" / "static"
            app_file = static / "app" / "tool.js"
            vendor_file = static / "vendor" / "library.js"
            monitor_file = static / "monitor.js"
            app_file.parent.mkdir(parents=True)
            vendor_file.parent.mkdir(parents=True)
            app_file.touch()
            vendor_file.touch()
            monitor_file.touch()

            files = run_tests.javascript_files(root)

        self.assertEqual(files, [app_file, monitor_file])

    def test_focus_aliases_and_explicit_targets_are_resolved(self) -> None:
        explicit = "tests.test_glm.GlmToolTests.test_glm_formula_drop_first_policy_tracks_regularization"

        targets = run_tests.resolve_focus_targets(
            ["line-bar", "static_assets", "test-runner", explicit], self.root
        )

        self.assertEqual(
            targets,
            [
                "tests/test_line_bar.py",
                "tests/test_static_assets.py",
                "tests/test_test_runner.py",
                explicit,
            ],
        )

    def test_unknown_focus_alias_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "unknown focused test area"):
            run_tests.resolve_focus_targets(["not-a-real-area"], self.root)

    def test_precommit_runs_checks_in_order(self) -> None:
        commands: list[list[str]] = []

        def fake_run(command: list[str], **_: object) -> int:
            commands.append(list(command))
            return 0

        with patch.object(run_tests, "run_process", side_effect=fake_run):
            with redirect_stdout(io.StringIO()):
                code = run_tests.run_precommit(self.root)

        self.assertEqual(code, 0)
        self.assertEqual(commands[0], ["git", "diff", "--check"])
        self.assertEqual(commands[1], ["git", "diff", "--cached", "--check"])
        self.assertEqual(commands[2][1:4], ["-m", "compileall", "src"])
        unittest_index = next(index for index, command in enumerate(commands) if "unittest" in command)
        browser_index = next(
            index for index, command in enumerate(commands) if "scripts/run_browser_smoke.py" in command
        )
        node_indexes = [index for index, command in enumerate(commands) if command[:2] == ["node", "--check"]]
        self.assertTrue(node_indexes)
        self.assertLess(max(node_indexes), unittest_index)
        self.assertLess(unittest_index, browser_index)

    def test_precommit_stops_after_first_failure(self) -> None:
        with patch.object(run_tests, "run_process", return_value=7) as run_process:
            with redirect_stdout(io.StringIO()):
                code = run_tests.run_precommit(self.root)

        self.assertEqual(code, 7)
        run_process.assert_called_once_with(["git", "diff", "--check"], cwd=self.root, env=None)

    def test_browser_arguments_are_forwarded(self) -> None:
        target = "tests/test_browser_smoke.py::BrowserSmokeTests::test_missing_token_boot_error_is_visible"
        with patch.object(run_tests, "run_process", return_value=0) as run_process:
            with redirect_stdout(io.StringIO()):
                code = run_tests.run_browser(self.root, ["--", target, "-q"])

        self.assertEqual(code, 0)
        command = run_process.call_args.args[0]
        self.assertEqual(command[1:], ["scripts/run_browser_smoke.py", target, "-q"])

    def test_pipx_command_enables_install_test_and_preserves_environment(self) -> None:
        with patch.dict(os.environ, {"PY_LUCIDUM_PIPX_PYTHON": "python3.13"}, clear=False):
            with patch.object(run_tests, "run_process", return_value=0) as run_process:
                with redirect_stdout(io.StringIO()):
                    code = run_tests.run_pipx(self.root)

        self.assertEqual(code, 0)
        env = run_process.call_args.kwargs["env"]
        self.assertEqual(env["PY_LUCIDUM_RUN_PIPX_INSTALL_TESTS"], "1")
        self.assertEqual(env["PY_LUCIDUM_PIPX_PYTHON"], "python3.13")

    def test_hook_reports_an_unusable_python_without_running_tests(self) -> None:
        env = os.environ.copy()
        env["PY_LUCIDUM_TEST_PYTHON"] = str(self.root / "missing-python")

        completed = subprocess.run(
            ["sh", ".githooks/pre-commit"],
            cwd=self.root,
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )

        self.assertEqual(completed.returncode, 1)
        self.assertIn("Python is not executable", completed.stderr)
        self.assertIn("PY_LUCIDUM_TEST_PYTHON", completed.stderr)


if __name__ == "__main__":
    unittest.main()
