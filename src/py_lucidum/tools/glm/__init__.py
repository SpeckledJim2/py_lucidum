from __future__ import annotations

TOOL_ID = "glm"
TOOL_LABEL = "GLM"
TOOL_ALIASES = ("glm", "generalised-linear-model", "generalized-linear-model")
DEFAULT_ENABLED = False


def register(app, context) -> None:
    from .routes import register as register_routes

    register_routes(app, context)


__all__ = ["DEFAULT_ENABLED", "TOOL_ALIASES", "TOOL_ID", "TOOL_LABEL", "register"]
