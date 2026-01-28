"""Pydantic models for request/response validation."""

from pydantic import BaseModel


class MessageRequest(BaseModel):
    """Request model for sending a message."""

    message: str


class MessageResponse(BaseModel):
    """Response model for a message exchange."""

    session_id: int
    user_message: str
    assistant_message: str


class SessionResponse(BaseModel):
    """Response model for session operations."""

    session_id: int
    message: str


class SessionDetailResponse(BaseModel):
    """Response model for session details."""

    id: int
    created_at: str
    ended_at: str | None
    is_active: bool
    messages: list[dict]
