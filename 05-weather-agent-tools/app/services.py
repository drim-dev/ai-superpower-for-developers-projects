"""LLM integration with standard OpenAI tools API."""

import json
import logging
from openai import OpenAI
from app.config import OPENAI_API_KEY, OPENAI_API_URL, OPENAI_MODEL, SYSTEM_PROMPT_PATH
from app import tools

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Singleton client instance
_client = None
_system_prompt = None


def get_openai_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(
            api_key=OPENAI_API_KEY,
            base_url=OPENAI_API_URL,
        )
    return _client


def get_system_prompt() -> str:
    global _system_prompt
    if _system_prompt is None:
        with open(SYSTEM_PROMPT_PATH, "r") as f:
            _system_prompt = f.read()
    return _system_prompt


def _log_request(messages: list[dict], tools_list: list[dict]):
    """Log the full JSON request being sent to OpenAI API."""
    request_data = {
        "model": OPENAI_MODEL,
        "messages": messages,
        "tools": tools_list,
    }
    logger.info("=" * 60)
    logger.info(">>> REQUEST TO OPENAI API:")
    logger.info("=" * 60)
    logger.info(json.dumps(request_data, indent=2, ensure_ascii=False))
    logger.info("=" * 60)


def _log_response(response):
    """Log the full JSON response from OpenAI API."""
    # Convert response object to dict
    response_dict = response.model_dump()
    logger.info("=" * 60)
    logger.info("<<< RESPONSE FROM OPENAI API:")
    logger.info("=" * 60)
    logger.info(json.dumps(response_dict, indent=2, ensure_ascii=False))
    logger.info("=" * 60)


async def generate_response(conversation_history: list[dict]) -> str:
    """Generate a response, handling tool calls if needed.

    This implements the agentic loop using OpenAI's native tool calling:
    1. Send messages to LLM with tools definition
    2. Check if response contains tool_calls
    3. If yes, execute tools and send results back to LLM
    4. Repeat until LLM responds without tool calls
    """
    system_prompt = get_system_prompt()
    client = get_openai_client()
    tools_list = tools.get_tools()

    messages = [
        {"role": "system", "content": system_prompt},
        *conversation_history
    ]

    # Agentic loop - keep going until no more tool calls
    max_iterations = 5  # Safety limit
    iteration = 0

    while iteration < max_iterations:
        iteration += 1
        logger.info(f"\n{'#' * 60}\n# ITERATION {iteration}\n{'#' * 60}")

        # Log the full request
        _log_request(messages, tools_list)

        # Call the LLM with tools
        response = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=messages,
            tools=tools_list,
        )

        # Log the full response
        _log_response(response)

        assistant_message = response.choices[0].message

        # Check if there are tool calls to process
        if not assistant_message.tool_calls:
            # No tool calls - return the final response
            return assistant_message.content

        # Add assistant message with tool calls to conversation (convert to dict)
        assistant_dict = {
            "role": "assistant",
            "content": assistant_message.content,
            "tool_calls": [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    }
                }
                for tc in assistant_message.tool_calls
            ]
        }
        messages.append(assistant_dict)

        # Execute each tool call and add results
        for tool_call in assistant_message.tool_calls:
            function_name = tool_call.function.name
            function_args = json.loads(tool_call.function.arguments)

            logger.info(f"[TOOLS] Executing: {function_name}({json.dumps(function_args, ensure_ascii=False)})")

            # Execute the tool
            result = await tools.execute_tool(function_name, function_args)

            logger.info(f"[TOOLS] Result: {result}")

            # Add tool result to messages
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": result,
            })

    # Safety: if we hit max iterations, return last response
    return "Извините, но я столкнулся с проблемой при обработке вашего запроса. Пожалуйста, попробуйте снова."
