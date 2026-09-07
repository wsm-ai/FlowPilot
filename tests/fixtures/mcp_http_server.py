import sys

from mcp.server import MCPServer


mcp = MCPServer("FlowPilot streamable HTTP test server")


@mcp.tool()
def add_numbers(a: int, b: int) -> dict[str, int]:
    return {"sum": a + b}


@mcp.tool()
def echo_text(text: str) -> dict[str, str]:
    return {"echo": text}


if __name__ == "__main__":
    mcp.run(
        transport="streamable-http",
        host="127.0.0.1",
        port=int(sys.argv[1]),
        streamable_http_path="/mcp",
    )
