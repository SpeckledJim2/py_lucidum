from __future__ import annotations

from .query import profile, profile_detail

TOOL_ID = "column_profile"
TOOL_LABEL = "Column profile"
TOOL_ALIASES = ("column-profile", "column_profile", "columnprofile", "columns", "profile")
DEFAULT_ENABLED = True


def register(app, context) -> None:
    from .routes import register as register_routes

    register_routes(app, context)


__all__ = [
    "DEFAULT_ENABLED",
    "TOOL_ALIASES",
    "TOOL_ID",
    "TOOL_LABEL",
    "profile",
    "profile_detail",
    "register",
]
