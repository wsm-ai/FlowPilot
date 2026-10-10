from typing import Any

from app.tools.base import Tool, ToolExecutionError
from app.tools.customer_feedback import CustomerFeedbackTool
from app.security.tool_authorization import (
    ToolAuthorizationError,
    ToolRisk,
    authorize_tool_execution,
)


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}
        self._risks: dict[str, ToolRisk] = {}

    def register(
        self,
        tool: Tool,
        *,
        risk: ToolRisk | None = None,
    ) -> None:
        if tool.name in self._tools:
            raise ToolExecutionError(f"Tool is already registered: {tool.name}")
        self._tools[tool.name] = tool
        declared_risk = getattr(tool, "risk", ToolRisk.UNCLASSIFIED)
        self._risks[tool.name] = risk if risk is not None else declared_risk

    def get(self, name: str) -> Tool:
        try:
            return self._tools[name]
        except KeyError as exc:
            raise ToolExecutionError(f"Unknown tool: {name}") from exc

    def contains(self, name: str) -> bool:
        return name in self._tools

    def excluding(self, names: set[str] | frozenset[str]) -> "ToolRegistry":
        excluded = frozenset(names)
        restricted = ToolRegistry()
        for name, tool in self._tools.items():
            if name not in excluded:
                restricted.register(tool, risk=self._risks[name])
        return restricted

    def including(self, names: set[str] | frozenset[str]) -> "ToolRegistry":
        included = frozenset(names)
        missing = [name for name in included if name not in self._tools]
        if missing:
            raise ToolExecutionError("Requested tools are not registered")
        selected = ToolRegistry()
        for name, tool in self._tools.items():
            if name in included:
                selected.register(tool, risk=self._risks[name])
        return selected

    def definitions(self) -> list[dict[str, Any]]:
        return [
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.parameters,
                },
            }
            for tool in self._tools.values()
        ]

    def authorize(
        self,
        name: str,
        arguments: dict[str, Any] | None = None,
    ) -> None:
        self.get(name)
        try:
            authorize_tool_execution(name, self._risks[name], arguments)
        except ToolAuthorizationError as exc:
            raise ToolExecutionError(str(exc)) from exc

    async def execute(self, name: str, arguments: dict[str, Any]) -> Any:
        tool = self.get(name)
        self.authorize(name, arguments)
        try:
            return await tool.execute(arguments)
        except ToolExecutionError:
            raise
        except Exception as exc:
            raise ToolExecutionError(f"Tool execution failed: {name}") from exc


def create_default_tool_registry() -> ToolRegistry:
    registry = ToolRegistry()
    registry.register(CustomerFeedbackTool(), risk=ToolRisk.READ_ONLY)
    return registry
