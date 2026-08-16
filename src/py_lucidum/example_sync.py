from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass
from importlib import resources
from importlib.resources.abc import Traversable
from pathlib import Path


EXAMPLE_SCRIPT_NAMES = (
    "01_external_gbm_artifacts_demo.py",
    "01_external_glm_artifacts_demo.py",
    "02_external_gbm_report_demo.py",
    "02_external_glm_report_demo.py",
    "03_external_gbm_summary_report_demo.py",
    "03_external_glm_summary_report_demo.py",
    "external_model_helpers.py",
    "external_model_results.py",
    "external_report_helpers.py",
    "lucidum_install.py",
)


@dataclass(frozen=True)
class ExampleSyncResult:
    destination: Path
    created: tuple[str, ...]
    updated: tuple[str, ...]
    unchanged: tuple[str, ...]
    dry_run: bool


def _example_source_root() -> Traversable:
    source_tree_root = Path(__file__).resolve().parents[2] / "examples"
    if all(source_tree_root.joinpath(name).is_file() for name in EXAMPLE_SCRIPT_NAMES):
        return source_tree_root
    return resources.files("py_lucidum").joinpath("example_workflows")


def _bundled_script_contents() -> dict[str, bytes]:
    source_root = _example_source_root()
    contents: dict[str, bytes] = {}
    for name in EXAMPLE_SCRIPT_NAMES:
        resource = source_root.joinpath(name)
        if not resource.is_file():
            raise FileNotFoundError(f"Bundled example script is missing: {name}")
        contents[name] = resource.read_bytes()
    return contents


def _atomic_write(target: Path, content: bytes) -> None:
    descriptor, temporary_name = tempfile.mkstemp(
        dir=target.parent,
        prefix=f".{target.name}.",
        suffix=".tmp",
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        temporary_path.chmod(0o644)
        os.replace(temporary_path, target)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise


def sync_example_scripts(
    destination: str | Path,
    *,
    dry_run: bool = False,
) -> ExampleSyncResult:
    """Copy Lucidum-maintained external workflow scripts into ``destination``."""
    destination_path = Path(destination).expanduser().resolve()
    if destination_path.exists() and not destination_path.is_dir():
        raise NotADirectoryError(f"Example script destination is not a directory: {destination_path}")

    bundled_contents = _bundled_script_contents()
    created: list[str] = []
    updated: list[str] = []
    unchanged: list[str] = []

    for name, content in bundled_contents.items():
        target = destination_path / name
        if target.is_file() and target.read_bytes() == content:
            unchanged.append(name)
        elif target.exists() or target.is_symlink():
            if not target.is_file() and not target.is_symlink():
                raise OSError(f"Example script target is not a file: {target}")
            updated.append(name)
        else:
            created.append(name)

    if not dry_run:
        destination_path.mkdir(parents=True, exist_ok=True)
        for name in (*created, *updated):
            _atomic_write(destination_path / name, bundled_contents[name])

    return ExampleSyncResult(
        destination=destination_path,
        created=tuple(created),
        updated=tuple(updated),
        unchanged=tuple(unchanged),
        dry_run=dry_run,
    )


def format_example_sync_result(result: ExampleSyncResult) -> str:
    if result.dry_run:
        lines = [
            *(f"create: {name}" for name in result.created),
            *(f"update: {name}" for name in result.updated),
            *(f"unchanged: {name}" for name in result.unchanged),
        ]
        lines.append(
            f"Dry run for {result.destination}: "
            f"{len(result.created)} to create, {len(result.updated)} to update, "
            f"{len(result.unchanged)} unchanged."
        )
        return "\n".join(lines)

    return (
        f"Synced example scripts to {result.destination}: "
        f"{len(result.created)} created, {len(result.updated)} updated, "
        f"{len(result.unchanged)} unchanged."
    )
