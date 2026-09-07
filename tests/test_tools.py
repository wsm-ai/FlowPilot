import asyncio

import pytest

from app.tools.base import ToolExecutionError
from app.tools.registry import create_default_tool_registry
from app.tools.registry import ToolRegistry


class MarkerTool:
    description = "Marker tool"
    parameters = {"type": "object", "properties": {}}

    def __init__(self, name):
        self.name = name

    async def execute(self, arguments):
        return {"name": self.name}


@pytest.fixture
def registry():
    return create_default_tool_registry()


def test_registry_finds_customer_feedback_tool(registry):
    tool = registry.get("get_customer_feedback")

    assert tool.name == "get_customer_feedback"


def test_definitions_returns_customer_feedback_schema(registry):
    definitions = registry.definitions()

    assert len(definitions) == 1
    definition = definitions[0]
    assert definition["type"] == "function"
    assert definition["function"]["name"] == "get_customer_feedback"

    parameters = definition["function"]["parameters"]
    assert parameters["type"] == "object"
    assert parameters["required"] == ["customer_id"]
    assert parameters["properties"]["customer_id"]["type"] == "string"
    priority_schema = parameters["properties"]["priority"]
    assert {"low", "medium", "high"} in [
        set(option["enum"])
        for option in priority_schema["anyOf"]
        if "enum" in option
    ]


def test_customer_feedback_returns_all_customer_records(registry):
    result = asyncio.run(
        registry.execute(
            "get_customer_feedback",
            {"customer_id": "C001"},
        )
    )

    assert len(result) == 3
    assert {record["id"] for record in result} == {"FB-001", "FB-002", "FB-003"}


def test_customer_feedback_strips_customer_id_whitespace(registry):
    result = asyncio.run(
        registry.execute(
            "get_customer_feedback",
            {"customer_id": " C001 "},
        )
    )

    assert len(result) == 3


def test_blank_customer_id_raises_tool_execution_error(registry):
    with pytest.raises(ToolExecutionError):
        asyncio.run(
            registry.execute(
                "get_customer_feedback",
                {"customer_id": "   "},
            )
        )


def test_customer_feedback_filters_by_priority(registry):
    result = asyncio.run(
        registry.execute(
            "get_customer_feedback",
            {"customer_id": "C001", "priority": "high"},
        )
    )

    assert len(result) == 2
    assert all(record["priority"] == "high" for record in result)


def test_unknown_customer_returns_empty_list(registry):
    result = asyncio.run(
        registry.execute(
            "get_customer_feedback",
            {"customer_id": "C999"},
        )
    )

    assert result == []


def test_invalid_priority_raises_tool_execution_error(registry):
    with pytest.raises(ToolExecutionError):
        asyncio.run(
            registry.execute(
                "get_customer_feedback",
                {"customer_id": "C001", "priority": "urgent"},
            )
        )


def test_unknown_tool_raises_tool_execution_error(registry):
    with pytest.raises(ToolExecutionError):
        asyncio.run(registry.execute("unknown_tool", {}))


def test_excluding_returns_restricted_registry_without_mutating_original():
    registry = ToolRegistry()
    safe_tool = MarkerTool("safe_tool")
    approval_tool = MarkerTool("approval_tool")
    registry.register(safe_tool)
    registry.register(approval_tool)
    excluded = {"approval_tool"}

    restricted = registry.excluding(excluded)

    assert [item["function"]["name"] for item in registry.definitions()] == [
        "safe_tool",
        "approval_tool",
    ]
    assert [item["function"]["name"] for item in restricted.definitions()] == [
        "safe_tool"
    ]
    assert restricted.get("safe_tool") is safe_tool
    assert excluded == {"approval_tool"}
