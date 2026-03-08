"""
Analytics router — serves historical data from JSON memory files.
"""

import json
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, HTTPException, Query

router = APIRouter()

MEMORY_DIR = Path("data/user_memory")


def _load_memory(user_id: str) -> dict:
    f = MEMORY_DIR / f"{user_id}_longterm_memory.json"
    if not f.exists():
        raise HTTPException(status_code=404, detail="No history for this user")
    with open(f, encoding="utf-8") as fh:
        return json.load(fh)


@router.get("/{user_id}/history")
def get_daily_history(
    user_id: str,
    days: int = Query(default=30, ge=1, le=365),
):
    """Return the last N daily profiles for a user."""
    data = _load_memory(user_id)
    daily: dict = data.get("daily_profiles", {})
    sorted_dates = sorted(daily.keys())[-days:]
    return {date: daily[date] for date in sorted_dates}


@router.get("/{user_id}/psv")
def get_psv(user_id: str):
    """Return the Personality State Vector for a user."""
    f = MEMORY_DIR / f"{user_id}_psv.json"
    if not f.exists():
        raise HTTPException(
            status_code=404, detail="No PSV data yet — complete more sessions"
        )
    with open(f, encoding="utf-8") as fh:
        return json.load(fh)


@router.get("/{user_id}/sessions")
def get_sessions(
    user_id: str,
    limit: int = Query(default=20, ge=1, le=200),
):
    """Return recent session summaries from long-term memory."""
    data = _load_memory(user_id)
    sessions = data.get("sessions", [])
    # newest first
    return sessions[-limit:][::-1]


@router.get("/{user_id}/summary")
def get_summary(user_id: str):
    """High-level summary: total sessions, days tracked, latest risk."""
    data = _load_memory(user_id)
    daily = data.get("daily_profiles", {})
    sessions = data.get("sessions", [])

    latest_risk = None
    if daily:
        latest_day = daily[sorted(daily.keys())[-1]]
        latest_risk = latest_day.get("avg_risk_level")

    return {
        "total_sessions": len(sessions),
        "days_tracked": len(daily),
        "latest_risk_avg": latest_risk,
        "last_updated": data.get("last_updated"),
    }
