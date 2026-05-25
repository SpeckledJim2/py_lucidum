from __future__ import annotations

TOOL_ID = "gbm"
TOOL_LABEL = "GBM"
TOOL_ALIASES = ("gbm", "gradient-boosting-model", "gradient-boosted-model")
DEFAULT_ENABLED = False


def register(app, context) -> None:
    from .routes import register as register_routes

    register_routes(app, context)


__all__ = ["DEFAULT_ENABLED", "TOOL_ALIASES", "TOOL_ID", "TOOL_LABEL", "register"]
