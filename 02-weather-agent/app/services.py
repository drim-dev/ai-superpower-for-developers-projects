"""LLM integration with manual tools protocol."""

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


def log_conversation(messages: list[dict], iterations: int) -> None:
    """Print the whole dialog once, after the agentic loop has finished.

    One readable pass over the final context instead of dumping it on every
    iteration. Note the tool result lives here as a `user` message — the naive
    protocol has nowhere else to put it.
    """
    logger.info("\n" + "=" * 60)
    logger.info(f"ИСТОРИЯ ДИАЛОГА — итераций цикла: {iterations}, сообщений: {len(messages)}")
    logger.info("=" * 60)
    for i, msg in enumerate(messages):
        role = msg["role"].upper()
        content = msg["content"]
        if role == "SYSTEM":
            content = content.splitlines()[0] + " …[системный промпт + описание инструментов]"
        logger.info(f"[{i}] {role}:\n{content}\n")
    logger.info("=" * 60)


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

        response_text = _call_llm(messages)

        if not tools.has_tool_calls(response_text):
            messages.append({"role": "assistant", "content": response_text})
            log_conversation(messages, iteration)
            return response_text

        tool_calls = tools.parse_tool_calls(response_text)
        if not tool_calls:
            messages.append({"role": "assistant", "content": response_text})
            log_conversation(messages, iteration)
            return tools.extract_text_before_tool_calls(response_text) or response_text

        tool_results = await tools.execute_tool_calls(tool_calls)

        messages.append({"role": "assistant", "content": response_text})
        messages.append({"role": "user", "content": tools.format_tool_results(tool_results)})

    log_conversation(messages, iteration)
    return "I apologize, but I encountered an issue processing your request. Please try again."
