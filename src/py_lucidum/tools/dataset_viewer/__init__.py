from __future__ import annotations

from .query import table

TOOL_ID = "dataset_viewer"
TOOL_LABEL = "Dataset viewer"
TOOL_ALIASES = ("dataset-viewer", "dataset_viewer", "datasetviewer", "dataset", "viewer", "table")
DEFAULT_ENABLED = True


def register(app, context) -> None:
    from .routes import register as register_routes

    register_routes(app, context)


__all__ = [
    "DEFAULT_ENABLED",
    "TOOL_ALIASES",
    "TOOL_ID",
    "TOOL_LABEL",
    "register",
    "table",
]
