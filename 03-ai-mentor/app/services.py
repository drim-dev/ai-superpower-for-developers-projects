"""Business logic services."""

from openai import OpenAI

from app.config import OPENAI_API_KEY, OPENAI_API_URL, OPENAI_MODEL, load_system_prompt

_client: OpenAI | None = None
_system_prompt: str | None = None


def get_openai_client() -> OpenAI:
    """Get or create OpenAI client singleton."""
    global _client
    if _client is None:
        _client = OpenAI(api_key=OPENAI_API_KEY, base_url=OPENAI_API_URL)
    return _client


def get_system_prompt() -> str:
    """Get cached system prompt."""
    global _system_prompt
    if _system_prompt is None:
        _system_prompt = load_system_prompt()
    return _system_prompt


def generate_response(conversation_history: list[dict]) -> str:
    """Generate AI response for the conversation."""
    client = get_openai_client()

    messages = [{"role": "system", "content": get_system_prompt()}]
    messages.extend(conversation_history)

    response = client.chat.completions.create(model=OPENAI_MODEL, messages=messages)

    return response.choices[0].message.content
