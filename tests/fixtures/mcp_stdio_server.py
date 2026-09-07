import asyncio
import os

from mcp.server import MCPServer


mcp = MCPServer("FlowPilot stdio test server")


@mcp.tool()
def add_numbers(a: int, b: int) -> dict[str, int]:
    return {"sum": a + b}


@mcp.tool()
def echo_text(text: str) -> dict[str, str]:
    return {"echo": text}


@mcp.tool()
def fail_tool() -> dict[str, str]:
    raise ValueError("sensitive server error")


@mcp.tool()
async def slow_tool(delay_seconds: float = 1.0) -> dict[str, bool]:
    await asyncio.sleep(delay_seconds)
    return {"completed": True}


@mcp.tool()
def hard_exit() -> None:
    os._exit(23)


if __name__ == "__main__":
    mcp.run()
