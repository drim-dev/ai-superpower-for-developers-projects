"""Naive ad-hoc tools protocol implementation.

This module demonstrates a simple but fragile approach to tool calling
by parsing ad-hoc string patterns from LLM responses.

Format:
  [ПОГОДА: city_name]
  [ПРОГНОЗ: city_name, days]

This approach is educational - it shows why structured protocols exist.
Problems with this naive approach:
1. LLM may format strings inconsistently
2. Parsing is fragile (special characters, typos break it)
3. No validation of parameters
4. Hard to extend with new tools
"""

import re
from dataclasses import dataclass
from app import weather


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
    return """## Доступные команды

Чтобы получить данные о погоде, используй специальные команды в квадратных скобках:

### Текущая погода
[ПОГОДА: название_города]

Примеры:
[ПОГОДА: Москва]
[ПОГОДА: New York]
[ПОГОДА: Токио]

### Прогноз погоды
[ПРОГНОЗ: название_города, количество_дней]

Примеры:
[ПРОГНОЗ: Париж, 3]
[ПРОГНОЗ: London, 7]
[ПРОГНОЗ: Берлин, 5]

Если количество дней не указано, будет показан прогноз на 3 дня."""


def parse_tool_calls(response_text: str) -> list[ToolCall]:
    """Parse tool calls from LLM response using ad-hoc string patterns.

    This naive approach uses simple regex to find patterns like:
    [ПОГОДА: Moscow]
    [ПРОГНОЗ: Paris, 5]
    """
    tool_calls = []

    # Pattern for current weather: [ПОГОДА: city]
    weather_pattern = r'\[ПОГОДА:\s*([^\]]+)\]'
    for match in re.finditer(weather_pattern, response_text, re.IGNORECASE):
        location = match.group(1).strip()
        tool_calls.append(ToolCall(
            name="get_current_weather",
            arguments={"location": location}
        ))

    # Pattern for forecast: [ПРОГНОЗ: city, days] or [ПРОГНОЗ: city]
    forecast_pattern = r'\[ПРОГНОЗ:\s*([^\],]+)(?:,\s*(\d+))?\]'
    for match in re.finditer(forecast_pattern, response_text, re.IGNORECASE):
        location = match.group(1).strip()
        days_str = match.group(2)
        days = int(days_str) if days_str else 3
        tool_calls.append(ToolCall(
            name="get_weather_forecast",
            arguments={"location": location, "days": days}
        ))

    return tool_calls


def extract_text_before_tool_calls(response_text: str) -> str:
    """Extract any text that appears before tool calls."""
    # Find the first bracket command
    match = re.search(r'\[(ПОГОДА|ПРОГНОЗ):', response_text, re.IGNORECASE)
    if match:
        return response_text[:match.start()].strip()
    return response_text.strip()


def has_tool_calls(response_text: str) -> bool:
    """Check if the response contains any tool calls."""
    return bool(re.search(r'\[(ПОГОДА|ПРОГНОЗ):', response_text, re.IGNORECASE))


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
    """Format tool results as plain text for the LLM."""
    lines = ["Результаты запросов:"]

    for result in results:
        data = result.result
        if "error" in data:
            lines.append(f"\nОшибка: {data['error']}")
        elif result.tool_name == "get_current_weather":
            lines.append(f"\nТекущая погода в {data.get('location', 'неизвестно')}:")
            lines.append(f"  Температура: {data.get('temperature_celsius')}°C")
            lines.append(f"  Ощущается как: {data.get('feels_like_celsius')}°C")
            lines.append(f"  Влажность: {data.get('humidity_percent')}%")
            lines.append(f"  Ветер: {data.get('wind_speed_kmh')} км/ч")
            lines.append(f"  Условия: {data.get('conditions')}")
        elif result.tool_name == "get_weather_forecast":
            lines.append(f"\nПрогноз погоды для {data.get('location', 'неизвестно')}:")
            for day in data.get("forecast", []):
                lines.append(f"  {day['date']}: {day['low_celsius']}°C - {day['high_celsius']}°C, {day['conditions']}, осадки: {day['precipitation_chance_percent']}%")

    return "\n".join(lines)
