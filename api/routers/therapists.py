"""
Therapist account management router.
Stores accounts in data/user_memory/therapists.json on disk.
"""

import hashlib
import json
import uuid
from datetime import datetime
from pathlib import Path

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

router = APIRouter()

THERAPISTS_FILE = Path("data/user_memory/therapists.json")


# ── Pydantic schemas ──────────────────────────────────────────────────────────

class TherapistCreate(BaseModel):
    name: str
    email: str
    password: str


class TherapistPasswordReset(BaseModel):
    password: str


class TherapistOut(BaseModel):
    id: str
    name: str
    email: str
    createdAt: str
    isActive: bool
    # password intentionally omitted from responses


# ── Storage helpers ───────────────────────────────────────────────────────────

def _load() -> dict:
    if THERAPISTS_FILE.exists():
        with open(THERAPISTS_FILE, encoding="utf-8") as fh:
            return json.load(fh)
    return {}


def _save(data: dict) -> None:
    THERAPISTS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(THERAPISTS_FILE, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2)


def _hash_password(pw: str) -> str:
    return hashlib.sha256(pw.encode()).hexdigest()


def _to_out(t: dict) -> TherapistOut:
    return TherapistOut(
        id=t["id"],
        name=t["name"],
        email=t["email"],
        createdAt=t["createdAt"],
        isActive=t["isActive"],
    )


# ── Endpoints ─────────────────────────────────────────────────────────────────

@router.get("/", response_model=list[TherapistOut])
def list_therapists():
    """Return all therapist accounts (no passwords)."""
    data = _load()
    return [_to_out(t) for t in data.values()]


@router.post("/", response_model=TherapistOut, status_code=201)
def create_therapist(body: TherapistCreate):
    """Create a new therapist account."""
    name = body.name.strip()
    email = body.email.strip().lower()
    if not name:
        raise HTTPException(status_code=422, detail="Name is required.")
    if "@" not in email:
        raise HTTPException(status_code=422, detail="Enter a valid email address.")
    if len(body.password) < 6:
        raise HTTPException(status_code=422, detail="Password must be at least 6 characters.")

    data = _load()
    if any(t["email"] == email for t in data.values()):
        raise HTTPException(status_code=409, detail="An account with this email already exists.")

    tid = f"thr_{uuid.uuid4().hex[:12]}"
    record = {
        "id": tid,
        "name": name,
        "email": email,
        "passwordHash": _hash_password(body.password),
        "createdAt": datetime.utcnow().isoformat(),
        "isActive": True,
    }
    data[tid] = record
    _save(data)
    return _to_out(record)


@router.post("/{therapist_id}/verify")
def verify_password(therapist_id: str, body: TherapistPasswordReset):
    """Verify a therapist's password for login. Returns therapist info on success."""
    data = _load()
    t = data.get(therapist_id)
    if not t:
        raise HTTPException(status_code=404, detail="Therapist not found.")
    if not t["isActive"]:
        raise HTTPException(status_code=403, detail="Account is deactivated. Contact the admin.")
    if t["passwordHash"] != _hash_password(body.password):
        raise HTTPException(status_code=401, detail="Incorrect password.")
    return _to_out(t)


@router.post("/login")
def login_therapist(body: TherapistCreate):
    """Login by email + password. Returns therapist info on success."""
    email = body.email.strip().lower()
    data = _load()
    t = next((v for v in data.values() if v["email"] == email), None)
    if not t:
        raise HTTPException(status_code=404, detail="No therapist account found with this email.")
    if not t["isActive"]:
        raise HTTPException(status_code=403, detail="Account is deactivated. Contact the admin.")
    if t["passwordHash"] != _hash_password(body.password):
        raise HTTPException(status_code=401, detail="Incorrect password.")
    return _to_out(t)


@router.patch("/{therapist_id}/toggle", response_model=TherapistOut)
def toggle_therapist(therapist_id: str):
    """Toggle a therapist's active status."""
    data = _load()
    t = data.get(therapist_id)
    if not t:
        raise HTTPException(status_code=404, detail="Therapist not found.")
    t["isActive"] = not t["isActive"]
    _save(data)
    return _to_out(t)


@router.patch("/{therapist_id}/password")
def reset_password(therapist_id: str, body: TherapistPasswordReset):
    """Reset a therapist's password."""
    if len(body.password) < 6:
        raise HTTPException(status_code=422, detail="Password must be at least 6 characters.")
    data = _load()
    t = data.get(therapist_id)
    if not t:
        raise HTTPException(status_code=404, detail="Therapist not found.")
    t["passwordHash"] = _hash_password(body.password)
    _save(data)
    return {"message": "Password updated."}


@router.delete("/{therapist_id}")
def delete_therapist(therapist_id: str):
    """Delete a therapist account."""
    data = _load()
    if therapist_id not in data:
        raise HTTPException(status_code=404, detail="Therapist not found.")
    del data[therapist_id]
    _save(data)
    return {"message": "Therapist deleted."}
