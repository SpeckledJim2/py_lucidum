from __future__ import annotations

import io
import os
import subprocess
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from scripts import run_browser_smoke, run_tests


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
        self.assertTrue(set(run_tests.FAST_GLM_TEST_TARGETS).issubset(set(run_tests.dev_test_targets(self.root))))

    def test_dev_targets_include_new_test_modules_automatically(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            tests_dir = root / "tests"
            tests_dir.mkdir()
            for name in ("test_browser_smoke.py", "test_glm.py", "test_new_tool.py"):
                (tests_dir / name).touch()

            targets = run_tests.dev_test_targets(root)

            self.assertEqual(targets[0], "tests/test_new_tool.py")
            self.assertEqual(targets[1:], list(run_tests.FAST_GLM_TEST_TARGETS))

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

    def test_precommit_runs_whitespace_then_changed_lane_without_full_gate(self) -> None:
        commands: list[list[str]] = []

        def fake_run(command: list[str], **_: object) -> int:
            commands.append(list(command))
            return 0

        with patch.object(run_tests, "run_process", side_effect=fake_run):
            with patch.object(run_tests, "run_changed", return_value=0) as run_changed:
                with redirect_stdout(io.StringIO()):
                    code = run_tests.run_precommit(self.root)

        self.assertEqual(code, 0)
        self.assertEqual(
            commands,
            [
                ["git", "diff", "--check"],
                ["git", "diff", "--cached", "--check"],
            ],
        )
        run_changed.assert_called_once_with(self.root)
        self.assertFalse(any("unittest" in command for command in commands))
        self.assertFalse(any("scripts/run_browser_smoke.py" in command for command in commands))

    def test_prepush_runs_complete_checks_in_order(self) -> None:
        commands: list[list[str]] = []

        def fake_run(command: list[str], **_: object) -> int:
            commands.append(list(command))
            return 0

        with patch.object(run_tests, "run_process", side_effect=fake_run):
            with redirect_stdout(io.StringIO()):
                code = run_tests.run_prepush(self.root)

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

    def test_changed_targets_map_tool_frontend_and_multiple_areas(self) -> None:
        self.assertEqual(
            run_tests.changed_test_targets(["src/py_lucidum/tools/histogram/routes.py"]),
            ["tests/test_histogram.py"],
        )
        self.assertEqual(
            run_tests.changed_test_targets(["src/py_lucidum/static/app/histogram-tool.js"]),
            ["tests/test_static_assets.py"],
        )
        self.assertEqual(
            run_tests.changed_test_targets(
                ["src/py_lucidum/tools/gbm/training.py", "src/py_lucidum/tools/uk_map/query.py"]
            ),
            ["tests/test_gbm.py", "tests/test_uk_map.py"],
        )

    def test_changed_targets_use_fast_glm_and_preserve_slow_classification(self) -> None:
        targets = run_tests.changed_test_targets(["src/py_lucidum/tools/glm/validation.py"])

        self.assertEqual(targets, sorted(run_tests.FAST_GLM_TEST_TARGETS))
        self.assertTrue(run_tests.DEV_EXCLUDED_TEST_IDS)
        self.assertTrue(
            all("test_chart_glm_overlay_" in target for target in run_tests.DEV_EXCLUDED_TEST_IDS)
        )

    def test_changed_targets_fall_back_for_shared_unknown_clean_and_browser_changes(self) -> None:
        self.assertIsNone(run_tests.changed_test_targets(["src/py_lucidum/core/dataset.py"]))
        self.assertIsNone(run_tests.changed_test_targets(["an-unmapped-file.txt"]))
        self.assertIsNone(run_tests.changed_test_targets(["tests/test_browser_smoke.py"]))
        self.assertEqual(run_tests.changed_test_targets([]), [])
        self.assertEqual(run_tests.changed_test_targets(["DEVELOPMENT.md"]), [])

    def test_changed_paths_combines_staged_unstaged_and_untracked_names(self) -> None:
        outputs = [
            subprocess.CompletedProcess([], 0, stdout="src/py_lucidum/cli.py\ntests/test_cli.py\n", stderr=""),
            subprocess.CompletedProcess([], 0, stdout="new-file.txt\n", stderr=""),
        ]
        with patch.object(run_tests.subprocess, "run", side_effect=outputs) as run:
            paths = run_tests.changed_paths(self.root)

        self.assertEqual(paths, ["new-file.txt", "src/py_lucidum/cli.py", "tests/test_cli.py"])
        self.assertIn("HEAD", run.call_args_list[0].args[0])
        self.assertIn("--diff-filter=ACMRD", run.call_args_list[0].args[0])
        self.assertIn("--others", run.call_args_list[1].args[0])

    def test_precommit_stops_after_first_failure(self) -> None:
        with patch.object(run_tests, "run_process", return_value=7) as run_process:
            with patch.object(run_tests, "run_changed") as run_changed:
                with redirect_stdout(io.StringIO()):
                    code = run_tests.run_precommit(self.root)

        self.assertEqual(code, 7)
        run_process.assert_called_once_with(["git", "diff", "--check"], cwd=self.root, env=None)
        run_changed.assert_not_called()

    def test_prepush_stops_before_browser_after_unit_failure(self) -> None:
        commands: list[list[str]] = []

        def fake_run(command: list[str], **_: object) -> int:
            commands.append(list(command))
            return 9 if "unittest" in command else 0

        with patch.object(run_tests, "run_process", side_effect=fake_run):
            with redirect_stdout(io.StringIO()):
                code = run_tests.run_prepush(self.root)

        self.assertEqual(code, 9)
        self.assertTrue(any("unittest" in command for command in commands))
        self.assertFalse(any("scripts/run_browser_smoke.py" in command for command in commands))

    def test_browser_arguments_are_forwarded(self) -> None:
        target = "tests/test_browser_smoke.py::BrowserSmokeTests::test_missing_token_boot_error_is_visible"
        with patch.object(run_tests, "run_process", return_value=0) as run_process:
            with redirect_stdout(io.StringIO()):
                code = run_tests.run_browser(self.root, ["--", target, "-q"])

        self.assertEqual(code, 0)
        command = run_process.call_args.args[0]
        self.assertEqual(command[1:], ["scripts/run_browser_smoke.py", "--", target, "-q"])

    def test_browser_helper_forwards_pytest_options_and_default_target(self) -> None:
        with patch.object(run_browser_smoke, "local_python", return_value="/test/python"):
            with patch.object(run_browser_smoke, "run", return_value=0) as run:
                code = run_browser_smoke.main(["--direct", "--", "--durations=5", "-q"])

        self.assertEqual(code, 0)
        command = run.call_args.args[0]
        self.assertEqual(
            command,
            ["/test/python", "-m", "pytest", "tests/test_browser_smoke.py", "--durations=5", "-q"],
        )

    def test_pipx_command_enables_install_test_and_preserves_environment(self) -> None:
        release_env = {
            "PY_LUCIDUM_EXPECTED_VERSION": "1.2.3",
            "PY_LUCIDUM_PIPX_PYTHON": "python3.13",
            "PY_LUCIDUM_PIPX_SPEC": "/tmp/py_lucidum-1.2.3-py3-none-any.whl",
        }
        with patch.dict(os.environ, release_env, clear=False):
            with patch.object(run_tests, "run_process", return_value=0) as run_process:
                with redirect_stdout(io.StringIO()):
                    code = run_tests.run_pipx(self.root)

        self.assertEqual(code, 0)
        env = run_process.call_args.kwargs["env"]
        self.assertEqual(env["PY_LUCIDUM_RUN_PIPX_INSTALL_TESTS"], "1")
        for name, value in release_env.items():
            self.assertEqual(env[name], value)

    def test_hooks_route_to_matching_runner_lanes(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            temp_root = Path(tmp_dir)
            fake_python = temp_root / "python"
            args_path = temp_root / "args.txt"
            fake_python.write_text(
                "#!/bin/sh\nprintf '%s\\n' \"$@\" > \"$PY_LUCIDUM_HOOK_ARGS\"\n",
                encoding="utf-8",
            )
            fake_python.chmod(0o755)
            env = os.environ.copy()
            env["PY_LUCIDUM_TEST_PYTHON"] = str(fake_python)
            env["PY_LUCIDUM_HOOK_ARGS"] = str(args_path)

            for hook, lane in (("pre-commit", "precommit"), ("pre-push", "prepush")):
                with self.subTest(hook=hook):
                    completed = subprocess.run(
                        ["sh", f".githooks/{hook}"],
                        cwd=self.root,
                        env=env,
                        text=True,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        check=False,
                    )

                    self.assertEqual(completed.returncode, 0, completed.stderr)
                    self.assertEqual(
                        args_path.read_text(encoding="utf-8").splitlines(),
                        [str(self.root / "scripts/run_tests.py"), lane],
                    )

    def test_hooks_report_an_unusable_python_without_running_tests(self) -> None:
        env = os.environ.copy()
        env["PY_LUCIDUM_TEST_PYTHON"] = str(self.root / "missing-python")

        for hook in ("pre-commit", "pre-push"):
            with self.subTest(hook=hook):
                completed = subprocess.run(
                    ["sh", f".githooks/{hook}"],
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
