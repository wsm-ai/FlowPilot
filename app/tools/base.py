from typing import Any, Protocol


class ToolExecutionError(Exception):
    """Raised when a tool cannot be found, validated, or executed."""


class Tool(Protocol):
    """Minimal interface implemented by FlowPilot tools."""

    name: str
    description: str

    @property
    def parameters(self) -> dict[str, Any]:
        ...

    async def execute(self, arguments: dict[str, Any]) -> Any:
        ...
