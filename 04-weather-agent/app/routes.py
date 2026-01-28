from fastapi import APIRouter, HTTPException
from app import database as db
from app import services
from app.models import (
    MessageRequest,
    MessageResponse,
    Message,
    SessionResponse,
    SessionDetailResponse,
)

router = APIRouter()


@router.post("/sessions", response_model=SessionResponse)
async def create_session():
    session_id = db.create_session()
    return SessionResponse(
        session_id=session_id,
        message="Weather session created successfully"
    )


@router.post("/sessions/{session_id}/messages", response_model=MessageResponse)
async def send_message(session_id: int, request: MessageRequest):
    session = db.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    if not session["is_active"]:
        raise HTTPException(status_code=400, detail="Session is no longer active")

    # Save user message
    db.add_message(session_id, "user", request.content)

    # Get conversation history
    history = db.get_session_messages(session_id)

    # Generate response using the agent
    try:
        assistant_response = await services.generate_response(history)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating response: {str(e)}")

    # Save assistant response
    db.add_message(session_id, "assistant", assistant_response)

    return MessageResponse(
        user_message=Message(role="user", content=request.content),
        assistant_message=Message(role="assistant", content=assistant_response),
    )


@router.post("/sessions/{session_id}/end", response_model=SessionResponse)
async def end_session(session_id: int):
    if not db.end_session(session_id):
        raise HTTPException(status_code=404, detail="Session not found or already ended")

    return SessionResponse(
        session_id=session_id,
        message="Session ended successfully"
    )


@router.get("/sessions", response_model=list[SessionDetailResponse])
async def list_sessions():
    sessions = db.get_all_sessions()
    result = []

    for session in sessions:
        messages = db.get_session_messages(session["id"])
        result.append(SessionDetailResponse(
            id=session["id"],
            created_at=str(session["created_at"]),
            ended_at=str(session["ended_at"]) if session["ended_at"] else None,
            is_active=bool(session["is_active"]),
            messages=[Message(**m) for m in messages],
        ))

    return result


@router.get("/sessions/{session_id}", response_model=SessionDetailResponse)
async def get_session(session_id: int):
    session = db.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    messages = db.get_session_messages(session_id)

    return SessionDetailResponse(
        id=session["id"],
        created_at=str(session["created_at"]),
        ended_at=str(session["ended_at"]) if session["ended_at"] else None,
        is_active=bool(session["is_active"]),
        messages=[Message(**m) for m in messages],
    )
