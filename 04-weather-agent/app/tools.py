"""Manual tools protocol implementation.

This module implements a simple tools protocol by:
1. Defining available tools and their schemas
2. Parsing tool calls from LLM text responses
3. Executing tools and returning results
"""

import json
import re
from dataclasses import dataclass
from app import weather


# Tool definitions - these describe what tools are available to the LLM
TOOLS = {
    "get_current_weather": {
        "description": "Get the current weather conditions for a specific location",
        "parameters": {
            "location": {
                "type": "string",
                "description": "The city name, e.g., 'London', 'New York', 'Tokyo'",
                "required": True,
            }
        },
    },
    "get_weather_forecast": {
        "description": "Get a weather forecast for a specific location",
        "parameters": {
            "location": {
                "type": "string",
                "description": "The city name, e.g., 'London', 'New York', 'Tokyo'",
                "required": True,
            },
            "days": {
                "type": "integer",
                "description": "Number of days to forecast (1-7, default 3)",
                "required": False,
            },
        },
    },
}


@dataclass
class ToolCall:
    """Represents a parsed tool call from LLM response."""
    name: str
    arguments: dict


@dataclass
class ToolResult:
    """Represents the result of executing a tool."""
    tool_name: str
    result: dict


def get_tools_prompt() -> str:
    """Generate the tools description for the system prompt."""
    lines = ["## Available Tools", ""]

    for name, tool in TOOLS.items():
        lines.append(f"### {name}")
        lines.append(f"{tool['description']}")
        lines.append("")
        lines.append("Parameters:")

        for param_name, param_info in tool["parameters"].items():
            required = "(required)" if param_info.get("required") else "(optional)"
            lines.append(f"  - {param_name}: {param_info['type']} {required}")
            lines.append(f"    {param_info['description']}")

        lines.append("")

    return "\n".join(lines)


def parse_tool_calls(response_text: str) -> list[ToolCall]:
    """Parse tool calls from LLM response text.

    Looks for tool calls in the format:
    <tool_call>
    {"name": "tool_name", "arguments": {"arg1": "value1"}}
    </tool_call>
    """
    tool_calls = []

    # Pattern to match tool_call blocks
    pattern = r'<tool_call>\s*(.*?)\s*</tool_call>'
    matches = re.findall(pattern, response_text, re.DOTALL)

    for match in matches:
        try:
            data = json.loads(match.strip())
            if "name" in data:
                tool_calls.append(ToolCall(
                    name=data["name"],
                    arguments=data.get("arguments", {})
                ))
        except json.JSONDecodeError:
            # Skip malformed tool calls
            continue

    return tool_calls


def extract_text_before_tool_calls(response_text: str) -> str:
    """Extract any text that appears before tool calls."""
    # Find the first tool_call tag
    match = re.search(r'<tool_call>', response_text)
    if match:
        return response_text[:match.start()].strip()
    return response_text.strip()


def has_tool_calls(response_text: str) -> bool:
    """Check if the response contains any tool calls."""
    return '<tool_call>' in response_text


async def execute_tool(tool_call: ToolCall) -> ToolResult:
    """Execute a tool call and return the result."""
    tool_name = tool_call.name
    args = tool_call.arguments

    if tool_name == "get_current_weather":
        location = args.get("location", "")
        result = await weather.get_current_weather(location)
        return ToolResult(tool_name=tool_name, result=result)

    elif tool_name == "get_weather_forecast":
        location = args.get("location", "")
        days = args.get("days", 3)
        result = await weather.get_weather_forecast(location, days)
        return ToolResult(tool_name=tool_name, result=result)

    else:
        return ToolResult(
            tool_name=tool_name,
            result={"error": f"Unknown tool: {tool_name}"}
        )


async def execute_tool_calls(tool_calls: list[ToolCall]) -> list[ToolResult]:
    """Execute multiple tool calls and return results."""
    results = []
    for tool_call in tool_calls:
        result = await execute_tool(tool_call)
        results.append(result)
    return results


def format_tool_results(results: list[ToolResult]) -> str:
    """Format tool results as a message to send back to the LLM."""
    lines = ["Tool results:"]

    for result in results:
        lines.append(f"\n<tool_result name=\"{result.tool_name}\">")
        lines.append(json.dumps(result.result, indent=2))
        lines.append("</tool_result>")

    return "\n".join(lines)
