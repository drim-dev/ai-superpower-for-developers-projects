"""API route handlers."""

from fastapi import APIRouter, HTTPException

from app import database, services
from app.models import (
    MessageRequest,
    MessageResponse,
    SessionDetailResponse,
    SessionResponse,
)

router = APIRouter()


@router.post("/sessions", response_model=SessionResponse)
def start_session():
    """Start a new mentoring session."""
    session_id = database.create_session()
    return SessionResponse(
        session_id=session_id, message="Mentoring session started successfully"
    )


@router.post("/sessions/{session_id}/messages", response_model=MessageResponse)
def send_message(session_id: int, request: MessageRequest):
    """Send a message in an active session and get AI response."""
    session = database.get_session(session_id)

    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    if not session["is_active"]:
        raise HTTPException(status_code=400, detail="Session has ended")

    database.add_message(session_id, "user", request.message)

    conversation_history = database.get_session_messages(session_id)

    try:
        assistant_message = services.generate_response(conversation_history)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"AI service error: {str(e)}")

    database.add_message(session_id, "assistant", assistant_message)

    return MessageResponse(
        session_id=session_id,
        user_message=request.message,
        assistant_message=assistant_message,
    )


@router.post("/sessions/{session_id}/end", response_model=SessionResponse)
def end_session(session_id: int):
    """End an active mentoring session."""
    success = database.end_session(session_id)

    if not success:
        session = database.get_session(session_id)
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")
        raise HTTPException(status_code=400, detail="Session already ended")

    return SessionResponse(
        session_id=session_id, message="Mentoring session ended successfully"
    )


@router.get("/sessions", response_model=list[SessionDetailResponse])
def list_sessions():
    """List all mentoring sessions."""
    sessions = database.get_all_sessions()
    result = []
    for session in sessions:
        messages = database.get_session_messages(session["id"])
        result.append(
            SessionDetailResponse(
                id=session["id"],
                created_at=str(session["created_at"]),
                ended_at=str(session["ended_at"]) if session["ended_at"] else None,
                is_active=bool(session["is_active"]),
                messages=messages,
            )
        )
    return result


@router.get("/sessions/{session_id}", response_model=SessionDetailResponse)
def get_session(session_id: int):
    """Get details of a specific session."""
    session = database.get_session(session_id)

    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    messages = database.get_session_messages(session_id)

    return SessionDetailResponse(
        id=session["id"],
        created_at=str(session["created_at"]),
        ended_at=str(session["ended_at"]) if session["ended_at"] else None,
        is_active=bool(session["is_active"]),
        messages=messages,
    )
