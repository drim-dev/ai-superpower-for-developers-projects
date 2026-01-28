from pydantic import BaseModel


class MessageRequest(BaseModel):
    content: str


class Message(BaseModel):
    role: str
    content: str


class MessageResponse(BaseModel):
    user_message: Message
    assistant_message: Message


class SessionResponse(BaseModel):
    session_id: int
    message: str


class SessionDetailResponse(BaseModel):
    id: int
    created_at: str
    ended_at: str | None
    is_active: bool
    messages: list[Message]
