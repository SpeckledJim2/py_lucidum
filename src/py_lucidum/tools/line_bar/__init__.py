from __future__ import annotations

from .query import chart

TOOL_ID = "line_bar"
TOOL_LABEL = "Line and bar chart"


def register(app, context) -> None:
    from .routes import register as register_routes

    register_routes(app, context)


__all__ = ["TOOL_ID", "TOOL_LABEL", "chart", "register"]
