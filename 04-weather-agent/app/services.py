"""LLM integration with manual tools protocol."""

from openai import OpenAI
from app.config import OPENAI_API_KEY, OPENAI_API_URL, OPENAI_MODEL, SYSTEM_PROMPT_PATH
from app import tools

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
            base_prompt = f.read()
        # Append tools description to the system prompt
        tools_prompt = tools.get_tools_prompt()
        _system_prompt = f"{base_prompt}\n\n{tools_prompt}"
    return _system_prompt


def _call_llm(messages: list[dict]) -> str:
    """Make a single LLM API call."""
    client = get_openai_client()
    response = client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=messages,
    )
    return response.choices[0].message.content


async def generate_response(conversation_history: list[dict]) -> str:
    """Generate a response, handling tool calls if needed.

    This implements the agentic loop:
    1. Send messages to LLM
    2. Check if response contains tool calls
    3. If yes, execute tools and send results back to LLM
    4. Repeat until LLM responds without tool calls
    """
    system_prompt = get_system_prompt()

    messages = [
        {"role": "system", "content": system_prompt},
        *conversation_history
    ]

    # Agentic loop - keep going until no more tool calls
    max_iterations = 5  # Safety limit
    iteration = 0

    while iteration < max_iterations:
        iteration += 1

        # Call the LLM
        response_text = _call_llm(messages)

        # Check for tool calls
        if not tools.has_tool_calls(response_text):
            # No tool calls - return the final response
            return response_text

        # Parse and execute tool calls
        tool_calls = tools.parse_tool_calls(response_text)
        if not tool_calls:
            # Malformed tool calls - return what we have
            return tools.extract_text_before_tool_calls(response_text) or response_text

        # Execute tools
        tool_results = await tools.execute_tool_calls(tool_calls)

        # Add the assistant's response (with tool calls) and tool results to messages
        messages.append({"role": "assistant", "content": response_text})
        messages.append({"role": "user", "content": tools.format_tool_results(tool_results)})

    # Safety: if we hit max iterations, return last response
    return "I apologize, but I encountered an issue processing your request. Please try again."
