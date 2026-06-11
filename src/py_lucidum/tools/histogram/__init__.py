from __future__ import annotations

from .query import histogram

TOOL_ID = "histogram"
TOOL_LABEL = "Histogram"
TOOL_ALIASES = ("histogram", "hist", "histo")
DEFAULT_ENABLED = True


def register(app, context) -> None:
    from .routes import register as register_routes

    register_routes(app, context)


__all__ = [
    "DEFAULT_ENABLED",
    "TOOL_ALIASES",
    "TOOL_ID",
    "TOOL_LABEL",
    "histogram",
    "register",
]
