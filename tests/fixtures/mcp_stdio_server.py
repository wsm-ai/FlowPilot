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


if __name__ == "__main__":
    mcp.run()
