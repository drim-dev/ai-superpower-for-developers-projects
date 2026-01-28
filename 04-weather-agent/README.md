# Weather Agent

An AI-powered weather assistant that demonstrates how to implement a **manual tools protocol** for educational purposes. Instead of using built-in function calling APIs, this project parses tool calls directly from LLM text responses.

## Features

- Current weather conditions for any city worldwide
- Weather forecasts up to 7 days
- Multi-turn conversations with session persistence
- Web chat interface

## Architecture

```
User Query → LLM → Parse <tool_call> tags → Execute weather API → Return results → LLM → Final response
```

### Manual Tools Protocol

The core educational component is in `app/tools.py`. It works by:

1. **Defining tools** as structured data (name, description, parameters)
2. **Injecting tool definitions** into the system prompt
3. **Instructing the LLM** to output tool calls in a specific format:
   ```
   <tool_call>
   {"name": "get_current_weather", "arguments": {"location": "London"}}
   </tool_call>
   ```
4. **Parsing responses** with regex to extract tool calls
5. **Executing tools** and formatting results as `<tool_result>` blocks
6. **Looping** until the LLM responds without tool calls

### Project Structure

```
04-weather-agent/
├── app/
│   ├── config.py      # Environment configuration
│   ├── database.py    # SQLite session/message storage
│   ├── models.py      # Pydantic request/response models
│   ├── routes.py      # FastAPI API endpoints
│   ├── services.py    # LLM integration with agentic loop
│   ├── tools.py       # Manual tools protocol implementation
│   └── weather.py     # Open-Meteo API integration
├── static/
│   └── index.html     # Web chat interface
├── main.py            # FastAPI application entry point
├── system_prompt.txt  # Agent personality and tool instructions
└── pyproject.toml     # Project dependencies
```

## Setup

1. Install dependencies:
   ```bash
   uv sync
   ```

2. Configure environment variables in `.env`:
   ```
   OPENAI_API_KEY=your-api-key
   OPENAI_API_URL=https://api.openai.com/v1  # or other compatible API
   OPENAI_MODEL=gpt-4o-mini
   ```

3. Run the server:
   ```bash
   uv run python main.py
   ```

4. Open http://localhost:8000

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/sessions` | POST | Create new session |
| `/api/sessions/{id}/messages` | POST | Send message, get response |
| `/api/sessions/{id}/end` | POST | End session |
| `/api/sessions` | GET | List all sessions |
| `/api/sessions/{id}` | GET | Get session details |

## Example Queries

- "What's the weather in Tokyo?"
- "Give me a 5-day forecast for Paris"
- "Is it raining in London right now?"
- "Compare weather in New York and Los Angeles"

## Weather API

Uses [Open-Meteo](https://open-meteo.com/) - a free weather API that requires no API key.
