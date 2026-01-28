"""Application configuration."""

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).parent.parent

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_API_URL = os.getenv("OPENAI_API_URL", "https://api.openai.com/v1")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

DATABASE_PATH = BASE_DIR / "mentor.db"
SYSTEM_PROMPT_PATH = BASE_DIR / "system_prompt.txt"


def load_system_prompt() -> str:
    """Load the system prompt from file."""
    return SYSTEM_PROMPT_PATH.read_text(encoding="utf-8")
