from __future__ import annotations

import tempfile
from importlib import resources
from pathlib import Path


DEMO_DATASET_NAME = "motor_premiums.parquet"


def _demo_dataset_resource():
    return resources.files("py_lucidum").joinpath("datasets").joinpath(DEMO_DATASET_NAME)


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
