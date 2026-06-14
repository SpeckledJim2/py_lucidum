from __future__ import annotations

import tomllib
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

PACKAGE_NAME = "py-lucidum"


def _source_tree_version() -> str:
    pyproject_path = Path(__file__).parents[2] / "pyproject.toml"
    try:
        metadata = tomllib.loads(pyproject_path.read_text(encoding="utf-8"))
    except OSError:
        return "0.0.0"
    project = metadata.get("project", {})
    value = project.get("version")
    return str(value) if value else "0.0.0"


def package_version() -> str:
    try:
        return version(PACKAGE_NAME)
    except PackageNotFoundError:
        return _source_tree_version()


__version__ = package_version()
