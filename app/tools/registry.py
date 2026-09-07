from typing import Any

from app.tools.base import Tool, ToolExecutionError
from app.tools.customer_feedback import CustomerFeedbackTool


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        if tool.name in self._tools:
            raise ToolExecutionError(f"Tool is already registered: {tool.name}")
        self._tools[tool.name] = tool

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
                restricted.register(tool)
        return restricted

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

    async def execute(self, name: str, arguments: dict[str, Any]) -> Any:
        tool = self.get(name)
        try:
            return await tool.execute(arguments)
        except ToolExecutionError:
            raise
        except Exception as exc:
            raise ToolExecutionError(f"Tool execution failed: {name}") from exc


def create_default_tool_registry() -> ToolRegistry:
    registry = ToolRegistry()
    registry.register(CustomerFeedbackTool())
    return registry
