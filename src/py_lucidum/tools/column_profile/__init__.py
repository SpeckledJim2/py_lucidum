from __future__ import annotations

from .query import profile, profile_detail

TOOL_ID = "column_profile"
TOOL_LABEL = "Column profile"


def register(app, context) -> None:
    from .routes import register as register_routes

    register_routes(app, context)


__all__ = ["TOOL_ID", "TOOL_LABEL", "profile", "profile_detail", "register"]
