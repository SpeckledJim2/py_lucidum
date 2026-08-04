#!/usr/bin/env python3
"""Run the browser-enabled pytest suite from a local worktree mirror.

macOS File Provider backed folders such as Dropbox CloudStorage can deny file
reads after Playwright launches Chromium. This helper copies the checkout to a
normal local cache directory first, then runs pytest there with PYTHONPATH
pointing at the mirrored src tree while still using the current Python venv.
"""

from __future__ import annotations

import argparse
import ast
import os
import subprocess
import sys
from pathlib import Path


EXCLUDES = (
    ".git",
    ".venv*",
    ".lucidum",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    "__pycache__",
    "*.pyc",
    "README.html",
    "README_files",
    "local",
)


def default_mirror_dir() -> Path:
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Caches" / "py_lucidum" / "browser-smoke-worktree"
    return Path.home() / ".cache" / "py_lucidum" / "browser-smoke-worktree"


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def run(command: list[str], *, cwd: Path | None = None, env: dict[str, str] | None = None) -> int:
    completed = subprocess.run(command, cwd=str(cwd) if cwd else None, env=env, check=False)
    return int(completed.returncode)


def pytest_args_have_target(args: list[str], worktree: Path) -> bool:
    for arg in args:
        if arg.startswith("-"):
            continue
        path_arg = arg.split("::", 1)[0]
        if path_arg and (worktree / path_arg).exists():
            return True
    return False


def browser_test_nodeids(test_file: Path) -> list[str]:
    """Return browser smoke node IDs without launching pytest for collection."""
    tree = ast.parse(test_file.read_text(encoding="utf-8"), filename=str(test_file))
    relative_test_file = test_file.as_posix()
    if test_file.is_absolute():
        relative_test_file = "/".join(test_file.parts[-2:])
    nodeids: list[str] = []
    for node in tree.body:
        if not isinstance(node, ast.ClassDef) or node.name != "BrowserSmokeTests":
            continue
        nodeids.extend(
            f"{relative_test_file}::{node.name}::{item.name}"
            for item in node.body
            if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)) and item.name.startswith("test_")
        )
    return sorted(nodeids)


def select_shard(nodeids: list[str], *, index: int, count: int) -> list[str]:
    if count < 1:
        raise ValueError("browser shard count must be positive")
    if index < 1 or index > count:
        raise ValueError(f"browser shard index must be between 1 and {count}")
    return [nodeid for position, nodeid in enumerate(sorted(nodeids)) if position % count == index - 1]


def local_python() -> str:
    python = local_venv() / "bin" / "python"
    return str(python) if python.exists() else sys.executable


def local_venv() -> Path:
    try:
        return Path(sys.prefix).resolve()
    except OSError:
        return Path(sys.prefix)


def mirror_worktree(source: Path, target: Path) -> None:
    source = source.resolve()
    target = target.resolve()
    if source == target or is_relative_to(target, source):
        raise SystemExit(f"Refusing to mirror into the source checkout: {target}")
    try:
        with (source / "pyproject.toml").open("rb"):
            pass
    except PermissionError as exc:
        raise SystemExit(
            "macOS is denying reads from the source checkout before it can be mirrored. "
            "Grant the terminal/Codex process access to the Dropbox CloudStorage folder "
            "or run this helper from a fresh terminal before launching browser tests."
        ) from exc
    target.parent.mkdir(parents=True, exist_ok=True)
    command = ["rsync", "-a", "--delete"]
    for pattern in EXCLUDES:
        command.extend(["--exclude", pattern])
    command.extend([f"{source}/", f"{target}/"])
    code = run(command)
    if code:
        raise SystemExit(code)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mirror-dir",
        type=Path,
        default=default_mirror_dir(),
        help="Local directory used for the mirrored worktree.",
    )
    parser.add_argument(
        "--direct",
        action="store_true",
        help="Run pytest in the current checkout without creating a mirror.",
    )
    parser.add_argument("--shard-index", type=int, help="One-based browser test shard to run.")
    parser.add_argument("--shard-count", type=int, help="Total number of browser test shards.")
    parser.add_argument(
        "pytest_args",
        nargs=argparse.REMAINDER,
        help="Arguments forwarded to pytest. Prefix with -- when passing pytest flags.",
    )
    args = parser.parse_args(argv)

    source = repo_root()
    worktree = source if args.direct else args.mirror_dir
    if not args.direct:
        args.mirror_dir.parent.mkdir(parents=True, exist_ok=True)
        os.chdir(args.mirror_dir.parent)
    if not args.direct:
        mirror_worktree(source, worktree)

    env = os.environ.copy()
    env["PY_LUCIDUM_RUN_BROWSER_TESTS"] = "1"
    src_path = str(worktree / "src")
    env["PYTHONPATH"] = src_path if not env.get("PYTHONPATH") else f"{src_path}{os.pathsep}{env['PYTHONPATH']}"
    venv = local_venv()
    env["VIRTUAL_ENV"] = str(venv)
    env["PATH"] = f"{venv / 'bin'}{os.pathsep}{env.get('PATH', '')}"
    artifact_dir = env.get("PY_LUCIDUM_BROWSER_ARTIFACT_DIR")
    if artifact_dir:
        artifact_path = Path(artifact_dir)
        if not artifact_path.is_absolute():
            artifact_path = (source / artifact_path).resolve()
        artifact_path.mkdir(parents=True, exist_ok=True)
        env["PY_LUCIDUM_BROWSER_ARTIFACT_DIR"] = str(artifact_path)

    pytest_args = list(args.pytest_args)
    if pytest_args and pytest_args[0] == "--":
        pytest_args = pytest_args[1:]
    shard_requested = args.shard_index is not None or args.shard_count is not None
    if shard_requested and (args.shard_index is None or args.shard_count is None):
        parser.error("--shard-index and --shard-count must be supplied together")
    if shard_requested and pytest_args_have_target(pytest_args, worktree):
        parser.error("browser sharding cannot be combined with an explicit pytest target")
    if shard_requested:
        try:
            nodeids = browser_test_nodeids(worktree / "tests" / "test_browser_smoke.py")
            selected = select_shard(nodeids, index=args.shard_index, count=args.shard_count)
        except ValueError as exc:
            parser.error(str(exc))
        if not selected:
            parser.error(f"browser shard {args.shard_index}/{args.shard_count} selected no tests")
        print(
            f"Browser shard {args.shard_index}/{args.shard_count}: {len(selected)} of {len(nodeids)} tests",
            flush=True,
        )
        pytest_args = [*selected, *pytest_args]
    elif not pytest_args_have_target(pytest_args, worktree):
        pytest_args = ["tests/test_browser_smoke.py", *pytest_args]
    return run([local_python(), "-m", "pytest", *pytest_args], cwd=worktree, env=env)


if __name__ == "__main__":
    raise SystemExit(main())
