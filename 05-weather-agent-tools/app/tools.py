"""Standard OpenAI tools API implementation.

This module defines tools using the OpenAI function calling format.
Tools are described as JSON schemas and passed to the API, which
handles tool call detection and argument parsing automatically.

Advantages over the naive ad-hoc approach:
1. Structured JSON schemas for tool definitions
2. API handles parsing - no regex needed
3. Type validation of parameters
4. Easy to extend with new tools
"""

import json
import logging
from app import weather

logger = logging.getLogger(__name__)


# Define tools in OpenAI function calling format
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_current_weather",
            "description": "Получить текущую погоду для указанного города. Возвращает температуру, влажность, скорость ветра и погодные условия.",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "Название города, например 'Москва', 'New York', 'Париж'"
                    }
                },
                "required": ["location"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_weather_forecast",
            "description": "Получить прогноз погоды на несколько дней для указанного города. Возвращает прогноз на каждый день с минимальной и максимальной температурой.",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "Название города, например 'Москва', 'New York', 'Париж'"
                    },
                    "days": {
                        "type": "integer",
                        "description": "Количество дней для прогноза (от 1 до 7)",
                        "minimum": 1,
                        "maximum": 7,
                        "default": 3
                    }
                },
                "required": ["location"]
            }
        }
    }
]


def get_tools() -> list[dict]:
    """Return the list of available tools in OpenAI format."""
    return TOOLS


async def execute_tool(name: str, arguments: dict) -> str:
    """Execute a tool and return the result as a JSON string."""
    logger.info(f"[TOOLS] Executing: {name}({arguments})")

    if name == "get_current_weather":
        location = arguments.get("location", "")
        result = await weather.get_current_weather(location)
        logger.info(f"[TOOLS] Result: {result}")
        return json.dumps(result, ensure_ascii=False)

    elif name == "get_weather_forecast":
        location = arguments.get("location", "")
        days = arguments.get("days", 3)
        result = await weather.get_weather_forecast(location, days)
        forecast = result.get("forecast", [])
        logger.info(f"[TOOLS] Result: {len(forecast)} days forecast for {result.get('location', 'unknown')}")
        for day in forecast:
            logger.info(f"[TOOLS]   {day['date']}: {day['low_celsius']}°C - {day['high_celsius']}°C, {day['conditions']}")
        return json.dumps(result, ensure_ascii=False)

    else:
        logger.warning(f"[TOOLS] Unknown tool: {name}")
        return json.dumps({"error": f"Unknown tool: {name}"}, ensure_ascii=False)
