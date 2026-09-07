from copy import deepcopy
import json
from typing import Any

from app.mcp.client import MCPClient, MCPError
from app.mcp.models import MCPRemoteTool
from app.tools.base import ToolExecutionError


class MCPToolAdapter:
    def __init__(
        self,
        client: MCPClient,
        *,
        local_name: str,
        remote_name: str,
        description: str,
        input_schema: dict[str, Any],
    ) -> None:
        normalized_local_name = local_name.strip()
        if not normalized_local_name:
            raise ValueError("local_name must not be blank")
        remote_tool = MCPRemoteTool(
            name=remote_name,
            description=description,
            input_schema=deepcopy(input_schema),
        )
        self._client = client
        self.name = normalized_local_name
        self.description = remote_tool.description
        self._remote_name = remote_tool.name
        self._input_schema = remote_tool.input_schema

    @property
    def parameters(self) -> dict[str, Any]:
        return deepcopy(self._input_schema)

    async def execute(self, arguments: dict[str, Any]) -> Any:
        copied_arguments = deepcopy(arguments)
        try:
            json.dumps(copied_arguments, ensure_ascii=False, allow_nan=False)
        except (TypeError, ValueError) as exc:
            raise ToolExecutionError("Invalid MCP tool arguments") from exc

        try:
            result = await self._client.call_tool(
                self._remote_name,
                copied_arguments,
            )
        except MCPError as exc:
            raise ToolExecutionError("MCP tool execution failed") from exc

        if result.is_error:
            raise ToolExecutionError("MCP tool execution failed")
        if result.structured_content is not None:
            return deepcopy(result.structured_content)
        return deepcopy(result.content)
