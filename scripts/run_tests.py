#!/usr/bin/env python3
"""Run Lucidum's focused, development, browser, and Git-hook test tiers."""

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
import time
import unittest
from collections.abc import Callable, Sequence
from pathlib import Path


DEV_EXCLUDED_TEST_FILES = frozenset({"test_browser_smoke.py", "test_glm.py"})
DEV_EXCLUDED_TEST_IDS = frozenset(
    {
        "tests.test_line_bar.LineBarToolTests.test_chart_glm_overlay_dispatches_to_worker_when_lightgbm_loaded",
        "tests.test_line_bar.LineBarToolTests.test_chart_glm_overlay_dispatches_to_worker_when_lightgbm_importable_before_loaded",
        "tests.test_line_bar.LineBarToolTests.test_chart_glm_overlay_worker_returns_rows_when_lightgbm_loaded",
        "tests.test_line_bar.LineBarToolTests.test_chart_glm_overlay_fresh_process_survives_with_lightgbm_importable",
    }
)
FAST_GLM_TEST_TARGETS = (
    "tests.test_glm.GlmToolTests.test_glm_subprocess_adds_worker_timing_metadata",
    "tests.test_glm.GlmToolTests.test_glm_training_dispatches_to_worker_when_isolation_required",
    "tests.test_glm.GlmToolTests.test_glm_suppresses_only_tabmat_mixed_dtype_warning",
    "tests.test_glm.GlmToolTests.test_glm_formula_drop_first_policy_tracks_regularization",
    "tests.test_glm.GlmToolTests.test_singular_matrix_errors_are_reported_as_rank_deficient_formula",
    "tests.test_glm.GlmToolTests.test_glm_config_routes_work_without_optional_dependency_imports",
    "tests.test_glm.GlmToolTests.test_glm_build_reports_actionable_missing_dependency",
    "tests.test_glm.GlmToolTests.test_formula_levels_endpoint_returns_sorted_capped_categorical_values",
    "tests.test_glm.GlmToolTests.test_formula_levels_endpoint_searches_levels_without_glm_dependencies",
    "tests.test_glm.GlmToolTests.test_formula_levels_endpoint_rejects_unknown_numeric_and_unreadable_columns",
    "tests.test_glm.GlmToolTests.test_formula_validation_strips_comments_accepts_rhs_and_full_forms_and_rejects_unsafe_text",
    "tests.test_glm.GlmToolTests.test_formula_validation_tracks_explicit_intercept_syntax",
    "tests.test_glm.GlmToolTests.test_formula_validation_warns_for_unconstrained_natural_spline_with_intercept",
    "tests.test_glm.GlmToolTests.test_regularization_validation_defaults_and_rejects_invalid_manual_values",
    "tests.test_glm.GlmToolTests.test_training_scope_requires_physical_sample_column",
)

TOOL_TEST_TARGETS = {
    "column_profile": ("tests/test_column_profile.py",),
    "dataset_viewer": ("tests/test_dataset_viewer.py",),
    "gbm": ("tests/test_gbm.py",),
    "glm": FAST_GLM_TEST_TARGETS,
    "histogram": ("tests/test_histogram.py",),
    "line_bar": ("tests/test_line_bar.py",),
    "specifications": ("tests/test_specifications.py",),
    "uk_map": ("tests/test_uk_map.py",),
}


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def discovered_test_files(root: Path | None = None) -> list[Path]:
    project_root = root or repo_root()
    return sorted((project_root / "tests").glob("test_*.py"))


def relative_path(path: Path, root: Path) -> str:
    return path.relative_to(root).as_posix()


def dev_test_targets(root: Path | None = None) -> list[str]:
    project_root = root or repo_root()
    targets = [
        relative_path(path, project_root)
        for path in discovered_test_files(project_root)
        if path.name not in DEV_EXCLUDED_TEST_FILES
    ]
    targets.extend(FAST_GLM_TEST_TARGETS)
    return targets


def changed_paths(root: Path | None = None) -> list[str]:
    project_root = root or repo_root()

    def git_names(*args: str) -> list[str]:
        completed = subprocess.run(
            ["git", *args],
            cwd=project_root,
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        if completed.returncode:
            raise RuntimeError(completed.stderr.strip() or "git path discovery failed")
        return [line for line in completed.stdout.splitlines() if line]

    paths = {
        *git_names("diff", "--name-only", "--diff-filter=ACMRD", "HEAD"),
        *git_names("ls-files", "--others", "--exclude-standard"),
    }
    return sorted(paths)


def changed_test_targets(paths: Sequence[str]) -> list[str] | None:
    """Return focused targets, or None when shared/unknown changes require dev."""
    targets: set[str] = set()
    for raw_path in paths:
        path = raw_path.replace("\\", "/")
        name = Path(path).name
        if path.startswith(("docs/",)) or name.endswith((".md", ".rst")):
            continue
        if path == "scripts/run_tests.py" or path == "scripts/run_browser_smoke.py":
            targets.add("tests/test_test_runner.py")
            continue
        if path.startswith("tests/") and name.startswith("test_") and name.endswith(".py"):
            if name in DEV_EXCLUDED_TEST_FILES:
                if name == "test_glm.py":
                    targets.update(FAST_GLM_TEST_TARGETS)
                    continue
                return None
            targets.add(path)
            continue
        if path == "pyproject.toml":
            targets.update(
                {"tests/test_demo_dataset.py", "tests/test_pipx_install.py", "tests/test_static_assets.py"}
            )
            continue
        if path in {"src/py_lucidum/cli.py", "src/py_lucidum/__init__.py", "src/py_lucidum/demo.py"}:
            targets.update({"tests/test_cli.py", "tests/test_demo_dataset.py"})
            continue
        if path.startswith("src/py_lucidum/tools/"):
            parts = Path(path).parts
            if len(parts) < 4 or parts[3] not in TOOL_TEST_TARGETS:
                return None
            targets.update(TOOL_TEST_TARGETS[parts[3]])
            continue
        if path.startswith("src/py_lucidum/static/"):
            targets.add("tests/test_static_assets.py")
            continue
        if "telemetry" in path or path == "src/py_lucidum/app/monitor.py":
            targets.add("tests/test_telemetry.py")
            continue
        if path.startswith(("datasets/", "specs/")):
            return None
        if path.startswith(("src/", "tests/", "scripts/")):
            return None
        if path in {"AGENTS.md", "README.md", "DEVELOPMENT.md"}:
            continue
        return None
    return sorted(targets)


def javascript_files(root: Path | None = None) -> list[Path]:
    project_root = root or repo_root()
    static_root = project_root / "src" / "py_lucidum" / "static"
    return sorted(
        path
        for path in static_root.rglob("*.js")
        if "vendor" not in path.relative_to(static_root).parts
    )


def resolve_focus_targets(targets: Sequence[str], root: Path | None = None) -> list[str]:
    project_root = root or repo_root()
    resolved: list[str] = []
    for target in targets:
        direct_path = project_root / target
        if direct_path.is_file():
            resolved.append(relative_path(direct_path, project_root))
            continue

        if not any(separator in target for separator in ("/", "\\", ".", ":")):
            alias = target.replace("-", "_")
            filenames = [f"{alias}.py"] if alias.startswith("test_") else []
            filenames.append(f"test_{alias}.py")
            for filename in filenames:
                candidate = project_root / "tests" / filename
                if candidate.is_file():
                    resolved.append(relative_path(candidate, project_root))
                    break
            else:
                raise ValueError(f"unknown focused test area: {target!r}")
            continue

        resolved.append(target)
    return resolved


def run_process(
    command: Sequence[str],
    *,
    cwd: Path,
    env: dict[str, str] | None = None,
) -> int:
    completed = subprocess.run(list(command), cwd=cwd, env=env, check=False)
    return int(completed.returncode)


def run_phase(label: str, action: Callable[[], int]) -> int:
    print(f"\n==> {label}", flush=True)
    started = time.perf_counter()
    code = action()
    elapsed = time.perf_counter() - started
    outcome = "passed" if code == 0 else f"failed ({code})"
    print(f"<== {label}: {outcome} in {elapsed:.2f}s", flush=True)
    return code


def run_command_phase(
    label: str,
    command: Sequence[str],
    *,
    root: Path,
    env: dict[str, str] | None = None,
) -> int:
    print(f"$ {shlex.join(command)}", flush=True)
    return run_phase(label, lambda: run_process(command, cwd=root, env=env))


def iter_tests(suite: unittest.TestSuite) -> Sequence[unittest.TestCase]:
    tests: list[unittest.TestCase] = []
    for item in suite:
        if isinstance(item, unittest.TestSuite):
            tests.extend(iter_tests(item))
        else:
            tests.append(item)
    return tests


def run_unittest_targets(targets: Sequence[str], *, excluded_ids: frozenset[str] = frozenset()) -> int:
    project_root = str(repo_root())
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    loader = unittest.defaultTestLoader
    names = [
        target.removesuffix(".py").replace("/", ".").replace("\\", ".")
        if target.endswith(".py")
        else target
        for target in targets
    ]
    suite = loader.loadTestsFromNames(names)
    if excluded_ids:
        suite = unittest.TestSuite(test for test in iter_tests(suite) if test.id() not in excluded_ids)
    result = unittest.TextTestRunner(verbosity=1).run(suite)
    return 0 if result.wasSuccessful() else 1


def run_syntax(root: Path) -> int:
    compile_command = [sys.executable, "-m", "compileall", "src", "tests"]
    code = run_command_phase("Python syntax", compile_command, root=root)
    if code:
        return code

    files = javascript_files(root)

    def check_javascript() -> int:
        for path in files:
            command = ["node", "--check", relative_path(path, root)]
            code = run_process(command, cwd=root)
            if code:
                print(f"JavaScript syntax failed: {relative_path(path, root)}", flush=True)
                return code
        return 0

    return run_phase(f"JavaScript syntax ({len(files)} files)", check_javascript)


def run_dev(root: Path) -> int:
    code = run_syntax(root)
    if code:
        return code
    return run_phase(
        "Broad development tests",
        lambda: run_unittest_targets(dev_test_targets(root), excluded_ids=DEV_EXCLUDED_TEST_IDS),
    )


def run_changed(root: Path) -> int:
    code = run_syntax(root)
    if code:
        return code
    paths = changed_paths(root)
    targets = changed_test_targets(paths)
    if not paths or targets is None:
        reason = "no working-tree changes" if not paths else "shared or unmapped changes"
        print(f"Changed lane: {reason}; running broad development tests.", flush=True)
        return run_phase(
            "Broad development tests",
            lambda: run_unittest_targets(dev_test_targets(root), excluded_ids=DEV_EXCLUDED_TEST_IDS),
        )
    if not targets:
        print("Changed lane: documentation-only changes; syntax checks are sufficient.", flush=True)
        return 0
    print("Changed lane targets:", *(f"  {target}" for target in targets), sep="\n", flush=True)
    if any(path.startswith("src/py_lucidum/static/") for path in paths):
        print("Frontend changed: run a focused browser smoke scenario when interaction behavior changed.", flush=True)
    return run_phase(
        "Changed tests",
        lambda: run_unittest_targets(targets, excluded_ids=DEV_EXCLUDED_TEST_IDS),
    )


def run_focus(root: Path, targets: Sequence[str]) -> int:
    command = [sys.executable, "-m", "unittest", *resolve_focus_targets(targets, root)]
    return run_command_phase("Focused tests", command, root=root)


def run_browser(root: Path, pytest_args: Sequence[str]) -> int:
    forwarded = list(pytest_args)
    if forwarded and forwarded[0] == "--":
        forwarded = forwarded[1:]
    command = [sys.executable, "scripts/run_browser_smoke.py", "--", *forwarded]
    return run_command_phase("Browser smoke tests", command, root=root)


def run_pipx(root: Path) -> int:
    env = os.environ.copy()
    env["PY_LUCIDUM_RUN_PIPX_INSTALL_TESTS"] = "1"
    command = [sys.executable, "-m", "unittest", "tests/test_pipx_install.py"]
    return run_command_phase("Pipx installation tests", command, root=root, env=env)


def run_whitespace_checks(root: Path) -> int:
    phases = (
        ("Unstaged whitespace", ["git", "diff", "--check"]),
        ("Staged whitespace", ["git", "diff", "--cached", "--check"]),
    )
    for label, command in phases:
        code = run_command_phase(label, command, root=root)
        if code:
            return code
    return 0


def run_precommit(root: Path) -> int:
    code = run_whitespace_checks(root)
    if code:
        return code
    return run_changed(root)


def run_prepush(root: Path, *, include_browser: bool = True) -> int:
    code = run_whitespace_checks(root)
    if code:
        return code
    code = run_syntax(root)
    if code:
        return code

    command = [sys.executable, "-m", "unittest", "discover", "-s", "tests"]
    code = run_command_phase("Full unittest suite", command, root=root)
    if code:
        return code
    return run_browser(root, ()) if include_browser else 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("dev", help="run syntax and broad development tests")
    subparsers.add_parser("changed", help="run syntax and tests selected from working-tree changes")
    subparsers.add_parser("syntax", help="check Python and non-vendored JavaScript syntax")
    subparsers.add_parser("precommit", help="run the fast change-aware local commit gate")
    prepush_parser = subparsers.add_parser("prepush", help="run the complete deterministic local push gate")
    prepush_parser.add_argument(
        "--skip-browser",
        action="store_true",
        help="stop after the full unittest suite (used by the sharded CI workflow)",
    )
    subparsers.add_parser("pipx", help="run the release-only pipx installation tests")

    focus_parser = subparsers.add_parser("focus", help="run focused unittest areas or targets")
    focus_parser.add_argument("targets", nargs="+", help="area aliases, test files, modules, or test methods")

    browser_parser = subparsers.add_parser("browser", help="run browser smoke tests through the local mirror")
    browser_parser.add_argument(
        "pytest_args",
        nargs=argparse.REMAINDER,
        help="pytest targets/options, optionally separated with --",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    root = repo_root()
    started = time.perf_counter()

    try:
        if args.command == "dev":
            code = run_dev(root)
        elif args.command == "changed":
            code = run_changed(root)
        elif args.command == "syntax":
            code = run_syntax(root)
        elif args.command == "focus":
            try:
                code = run_focus(root, args.targets)
            except ValueError as exc:
                parser.error(str(exc))
                return 2
        elif args.command == "browser":
            code = run_browser(root, args.pytest_args)
        elif args.command == "precommit":
            code = run_precommit(root)
        elif args.command == "prepush":
            code = run_prepush(root, include_browser=not args.skip_browser)
        elif args.command == "pipx":
            code = run_pipx(root)
        else:  # pragma: no cover - argparse enforces the command choices.
            parser.error(f"unsupported command: {args.command}")
            return 2
    finally:
        elapsed = time.perf_counter() - started
        print(f"\nTotal: {elapsed:.2f}s", flush=True)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
