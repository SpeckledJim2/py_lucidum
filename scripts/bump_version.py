#!/usr/bin/env python3
"""Bump the py_lucidum project version in pyproject.toml."""

from __future__ import annotations

import argparse
import re
from pathlib import Path


VERSION_LINE_RE = re.compile(r'(?m)^(version\s*=\s*)"([^"]+)"(\s*)$')
RELEASE_RE = re.compile(r"^(\d+)\.(\d+)\.(\d+)$")


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def pyproject_path() -> Path:
    return repo_root() / "pyproject.toml"


def read_version(path: Path) -> tuple[str, re.Match[str], str]:
    text = path.read_text(encoding="utf-8")
    match = VERSION_LINE_RE.search(text)
    if not match:
        raise ValueError(f"could not find project version in {path}")
    return match.group(2), match, text


def parse_release(version: str) -> tuple[int, int, int]:
    match = RELEASE_RE.fullmatch(version)
    if not match:
        raise ValueError(
            f"cannot increment non-release version {version!r}; use 'set MAJOR.MINOR.PATCH'"
        )
    major, minor, patch = (int(part) for part in match.groups())
    return major, minor, patch


def increment(version: str, part: str) -> str:
    major, minor, patch = parse_release(version)
    if part == "patch":
        patch += 1
    elif part == "minor":
        minor += 1
        patch = 0
    elif part == "major":
        major += 1
        minor = 0
        patch = 0
    else:
        raise ValueError(f"unsupported increment part {part!r}")
    return f"{major}.{minor}.{patch}"


def validate_release(version: str) -> str:
    if not RELEASE_RE.fullmatch(version):
        raise ValueError("version must use MAJOR.MINOR.PATCH, for example 0.2.0")
    return version


def write_version(path: Path, new_version: str) -> tuple[str, str]:
    current_version, match, text = read_version(path)
    updated = (
        text[: match.start()]
        + f'{match.group(1)}"{new_version}"{match.group(3)}'
        + text[match.end() :]
    )
    path.write_text(updated, encoding="utf-8")
    return current_version, new_version


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Bump py_lucidum's project.version in pyproject.toml."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    for name in ("patch", "minor", "major"):
        subparsers.add_parser(name, help=f"bump the {name} component")

    set_parser = subparsers.add_parser("set", help="set an explicit MAJOR.MINOR.PATCH version")
    set_parser.add_argument("version", type=validate_release)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    path = pyproject_path()

    try:
        current_version, _, _ = read_version(path)
        if args.command == "set":
            new_version = args.version
        else:
            new_version = increment(current_version, args.command)
        previous, updated = write_version(path, new_version)
    except ValueError as exc:
        parser.error(str(exc))
        return 2

    print(f"{previous} -> {updated}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
