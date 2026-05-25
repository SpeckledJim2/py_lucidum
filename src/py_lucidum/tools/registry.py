from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from importlib import import_module
from typing import Any

from fastapi import FastAPI

from py_lucidum.app.context import AppContext


TOOL_MODULES = (
    "py_lucidum.tools.column_profile",
    "py_lucidum.tools.line_bar",
    "py_lucidum.tools.uk_map",
    "py_lucidum.tools.glm",
    "py_lucidum.tools.gbm",
)


@dataclass(frozen=True)
class ToolDefinition:
    id: str
    label: str
    aliases: tuple[str, ...]
    default_enabled: bool
    module_name: str

    def register(self, app: FastAPI, context: AppContext) -> None:
        module = import_module(self.module_name)
        register = getattr(module, "register", None)
        if callable(register):
            register(app, context)


def _definition_from_module(module_name: str) -> ToolDefinition:
    module = import_module(module_name)
    tool_id = str(getattr(module, "TOOL_ID"))
    aliases = tuple(str(alias) for alias in getattr(module, "TOOL_ALIASES", (tool_id,)))
    return ToolDefinition(
        id=tool_id,
        label=str(getattr(module, "TOOL_LABEL", tool_id.replace("_", " ").title())),
        aliases=aliases,
        default_enabled=bool(getattr(module, "DEFAULT_ENABLED", False)),
        module_name=module_name,
    )


def tool_definitions() -> dict[str, ToolDefinition]:
    return {
        definition.id: definition
        for definition in (_definition_from_module(module_name) for module_name in TOOL_MODULES)
    }


def default_tool_ids() -> list[str]:
    return [
        definition.id
        for definition in tool_definitions().values()
        if definition.default_enabled
    ]


def tool_aliases() -> dict[str, str]:
    aliases: dict[str, str] = {}
    for definition in tool_definitions().values():
        for alias in definition.aliases:
            aliases[alias.lower()] = definition.id
    return aliases


def normalise_tools(tools: str | Sequence[str] | None) -> list[str]:
    aliases = tool_aliases()
    if tools is None:
        requested = default_tool_ids()
    elif isinstance(tools, str):
        requested = [part.strip() for part in tools.split(",") if part.strip()]
    else:
        requested = [str(part).strip() for part in tools if str(part).strip()]
    if not requested:
        requested = default_tool_ids()

    enabled: list[str] = []
    for name in requested:
        canonical = aliases.get(name.lower())
        if not canonical:
            supported = ", ".join(sorted(aliases))
            raise ValueError(f"Unknown tool '{name}'. Supported tools: {supported}")
        if canonical not in enabled:
            enabled.append(canonical)
    return enabled


def tool_payload(enabled_tools: Sequence[str]) -> list[dict[str, Any]]:
    definitions = tool_definitions()
    return [
        {"id": definition.id, "label": definition.label}
        for tool in enabled_tools
        if (definition := definitions.get(tool))
    ]


def register_tools(app: FastAPI, context: AppContext, enabled_tools: Sequence[str]) -> None:
    definitions = tool_definitions()
    for tool in enabled_tools:
        definition = definitions.get(tool)
        if definition:
            definition.register(app, context)


__all__ = [
    "ToolDefinition",
    "default_tool_ids",
    "normalise_tools",
    "register_tools",
    "tool_aliases",
    "tool_definitions",
    "tool_payload",
]
