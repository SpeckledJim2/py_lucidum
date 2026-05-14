from __future__ import annotations

from .query import summary

TOOL_ID = "uk_map"
TOOL_LABEL = "UK mapping tool"


def register(app, context) -> None:
    from .routes import register as register_routes

    register_routes(app, context)


__all__ = ["TOOL_ID", "TOOL_LABEL", "register", "summary"]
