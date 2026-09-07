import json
import math

import pytest
from pydantic import ValidationError

from app.mcp.models import MCPRemoteTool, MCPToolResult


def valid_schema():
    return {
        "type": "object",
        "properties": {"query": {"type": "string"}},
        "required": ["query"],
    }


def test_remote_tool_is_an_sdk_independent_json_safe_model():
    tool = MCPRemoteTool(
        name=" search_issues ",
        description=" Search remote issues ",
        input_schema=valid_schema(),
    )
    assert tool.name == "search_issues"
    assert tool.description == "Search remote issues"
    json.dumps(tool.model_dump(mode="json"), allow_nan=False)


@pytest.mark.parametrize("name", ["", "   ", "\t"])
def test_remote_tool_rejects_blank_name(name):
    with pytest.raises(ValidationError):
        MCPRemoteTool(name=name, input_schema=valid_schema())


@pytest.mark.parametrize(
    "input_schema",
    [
        {"enum": {"a", "b"}},
        {"default": math.nan},
        {"default": math.inf},
        {"default": b"bytes"},
    ],
)
def test_remote_tool_rejects_non_json_safe_schema(input_schema):
    with pytest.raises(ValidationError):
        MCPRemoteTool(name="search", input_schema=input_schema)


def test_tool_result_accepts_json_safe_content_and_structured_content():
    result = MCPToolResult(
        content=[{"type": "text", "text": "fallback"}],
        structured_content={"items": [{"id": 1}]},
        is_error=False,
    )
    assert result.structured_content == {"items": [{"id": 1}]}
    json.dumps(result.model_dump(mode="json"), allow_nan=False)


@pytest.mark.parametrize(
    "values",
    [
        {"content": [{"value": math.nan}]},
        {"content": [{"value": {1, 2}}]},
        {"structured_content": {"value": math.inf}},
        {"structured_content": {"value": object()}},
    ],
)
def test_tool_result_rejects_non_json_safe_values(values):
    with pytest.raises(ValidationError):
        MCPToolResult(**values)
