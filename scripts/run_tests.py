#!/usr/bin/env python3
"""Run Lucidum's focused, development, browser, and pre-commit test tiers."""

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
import time
from collections.abc import Callable, Sequence
from pathlib import Path


DEV_EXCLUDED_TEST_FILES = frozenset({"test_browser_smoke.py", "test_glm.py"})


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def discovered_test_files(root: Path | None = None) -> list[Path]:
    project_root = root or repo_root()
    return sorted((project_root / "tests").glob("test_*.py"))


def relative_path(path: Path, root: Path) -> str:
    return path.relative_to(root).as_posix()


def dev_test_targets(root: Path | None = None) -> list[str]:
    project_root = root or repo_root()
    return [
        relative_path(path, project_root)
        for path in discovered_test_files(project_root)
        if path.name not in DEV_EXCLUDED_TEST_FILES
    ]


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
    command = [sys.executable, "-m", "unittest", *dev_test_targets(root)]
    return run_command_phase("Broad development tests", command, root=root)


def run_focus(root: Path, targets: Sequence[str]) -> int:
    command = [sys.executable, "-m", "unittest", *resolve_focus_targets(targets, root)]
    return run_command_phase("Focused tests", command, root=root)


def run_browser(root: Path, pytest_args: Sequence[str]) -> int:
    forwarded = list(pytest_args)
    if forwarded and forwarded[0] == "--":
        forwarded = forwarded[1:]
    command = [sys.executable, "scripts/run_browser_smoke.py", *forwarded]
    return run_command_phase("Browser smoke tests", command, root=root)


def run_pipx(root: Path) -> int:
    env = os.environ.copy()
    env["PY_LUCIDUM_RUN_PIPX_INSTALL_TESTS"] = "1"
    command = [sys.executable, "-m", "unittest", "tests/test_pipx_install.py"]
    return run_command_phase("Pipx installation tests", command, root=root, env=env)


def run_precommit(root: Path) -> int:
    phases = (
        ("Unstaged whitespace", ["git", "diff", "--check"]),
        ("Staged whitespace", ["git", "diff", "--cached", "--check"]),
    )
    for label, command in phases:
        code = run_command_phase(label, command, root=root)
        if code:
            return code

    code = run_syntax(root)
    if code:
        return code

    command = [sys.executable, "-m", "unittest", "discover", "-s", "tests"]
    code = run_command_phase("Full unittest suite", command, root=root)
    if code:
        return code
    return run_browser(root, ())


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("dev", help="run syntax and broad development tests")
    subparsers.add_parser("syntax", help="check Python and non-vendored JavaScript syntax")
    subparsers.add_parser("precommit", help="run the complete deterministic local commit gate")
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
