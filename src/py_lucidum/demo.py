from __future__ import annotations

import tempfile
from importlib import resources
from pathlib import Path


DEMO_DATASET_NAME = "motor_premiums.parquet"
DEMO_FILTER_SPEC_NAME = "filter_spec.csv"
DEMO_KPI_SPEC_NAME = "kpi_spec.csv"
DEMO_FEATURE_SPEC_NAME = "feature_spec.csv"


def _demo_dataset_resource():
    return resources.files("py_lucidum").joinpath("datasets", DEMO_DATASET_NAME)


def _demo_spec_resource(spec_name: str):
    return resources.files("py_lucidum").joinpath("specs", spec_name)


def demo_dataset_path() -> Path:
    """Return a filesystem path for the bundled motor premiums demo dataset."""
    source_tree_path = Path(__file__).parents[2] / "datasets" / DEMO_DATASET_NAME
    if source_tree_path.exists():
        return source_tree_path.resolve()

    resource = _demo_dataset_resource()
    if not resource.is_file():
        raise FileNotFoundError(f"Bundled demo dataset is missing: {DEMO_DATASET_NAME}")
    if isinstance(resource, Path):
        return resource.resolve()

    target = Path(tempfile.gettempdir()) / "py_lucidum" / DEMO_DATASET_NAME
    if not target.exists():
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(resource.read_bytes())
    return target


def _demo_spec_path(spec_name: str) -> Path:
    source_tree_path = Path(__file__).parents[2] / "specs" / spec_name
    if source_tree_path.exists():
        return source_tree_path.resolve()

    resource = _demo_spec_resource(spec_name)
    if not resource.is_file():
        raise FileNotFoundError(f"Bundled demo specification file is missing: {spec_name}")
    if isinstance(resource, Path):
        return resource.resolve()

    target = Path(tempfile.gettempdir()) / "py_lucidum" / "specs" / spec_name
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(resource.read_bytes())
    return target


def demo_filter_spec_path() -> Path:
    """Return a filesystem path for the bundled demo saved-filter spec."""
    return _demo_spec_path(DEMO_FILTER_SPEC_NAME)


def demo_kpi_spec_path() -> Path:
    """Return a filesystem path for the bundled demo KPI spec."""
    return _demo_spec_path(DEMO_KPI_SPEC_NAME)


def demo_feature_spec_path() -> Path:
    """Return a filesystem path for the bundled demo feature spec."""
    return _demo_spec_path(DEMO_FEATURE_SPEC_NAME)


def demo_spec_paths() -> dict[str, Path]:
    """Return bundled demo specification paths keyed by serve() argument name."""
    return {
        "filters": demo_filter_spec_path(),
        "kpis": demo_kpi_spec_path(),
        "features": demo_feature_spec_path(),
    }
