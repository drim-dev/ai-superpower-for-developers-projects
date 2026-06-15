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


def log_conversation(messages: list[dict], iterations: int) -> None:
    """Print the whole dialog once, after the agentic loop has finished.

    One readable pass over the final context. Note how tool results carry their
    own `tool` role and a `tool_call_id` linking them to the exact call — the
    structured protocol, unlike the naive one, never disguises them as the user.
    """
    logger.info("\n" + "=" * 60)
    logger.info(f"ИСТОРИЯ ДИАЛОГА — итераций цикла: {iterations}, сообщений: {len(messages)}")
    logger.info("=" * 60)
    for i, msg in enumerate(messages):
        role = msg["role"].upper()
        if role == "SYSTEM":
            logger.info(f"[{i}] SYSTEM: {msg['content'].splitlines()[0]} …[системный промпт]\n")
        elif role == "ASSISTANT" and msg.get("tool_calls"):
            calls = ", ".join(
                f"{tc['function']['name']}({tc['function']['arguments']})"
                for tc in msg["tool_calls"]
            )
            logger.info(f"[{i}] ASSISTANT → вызов инструментов: {calls}\n")
        elif role == "TOOL":
            logger.info(f"[{i}] TOOL (tool_call_id={msg['tool_call_id']}): {msg['content']}\n")
        else:
            logger.info(f"[{i}] {role}: {msg['content']}\n")
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

        response = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=messages,
            tools=tools_list,
        )

        assistant_message = response.choices[0].message

        if not assistant_message.tool_calls:
            messages.append({"role": "assistant", "content": assistant_message.content})
            log_conversation(messages, iteration)
            return assistant_message.content

        logger.info(
            f"→ итерация {iteration}: модель запросила инструментов: {len(assistant_message.tool_calls)}"
        )

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

        for tool_call in assistant_message.tool_calls:
            function_name = tool_call.function.name
            function_args = json.loads(tool_call.function.arguments)
            result = await tools.execute_tool(function_name, function_args)
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": result,
            })

    log_conversation(messages, iteration)
    return "Извините, но я столкнулся с проблемой при обработке вашего запроса. Пожалуйста, попробуйте снова."
