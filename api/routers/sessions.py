"""
Session start / stop / status router
"""

from fastapi import APIRouter, HTTPException

from api.schemas import SessionStartRequest, SessionStatus
from api.session_manager import session_manager

router = APIRouter()


@router.post("/start")
def start_session(body: SessionStartRequest):
    """Start a live analysis session for the given user."""
    if not body.user_id:
        raise HTTPException(status_code=422, detail="user_id is required")
    try:
        session_manager.start(body.user_id)
        return {"message": "Session started", "user_id": body.user_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/stop")
def stop_session():
    """Stop the current session and persist data."""
    session_manager.stop()
    return {"message": "Session stopped"}


@router.get("/status", response_model=SessionStatus)
def get_status():
    """Return current session state."""
    return SessionStatus(
        is_running=session_manager.is_running,
        active_user_id=session_manager.active_user_id,
    )
