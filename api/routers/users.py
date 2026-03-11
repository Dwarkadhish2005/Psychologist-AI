"""
User management router
"""

import hashlib
import json
import sys
from pathlib import Path

from fastapi import APIRouter, HTTPException

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from inference.phase4_user_manager import UserManager
from api.schemas import UserCreate, UserResponse, PinSet, PinVerify, UserLogin, UserRegister, UserLoginResponse
from api.auth import verify_user_credentials, register_user_credentials, email_exists

router = APIRouter()
_user_manager = UserManager(storage_dir="data/user_memory")

PINS_FILE = Path("data/user_memory/pins.json")


def _load_pins() -> dict:
    if PINS_FILE.exists():
        with open(PINS_FILE, encoding="utf-8") as fh:
            return json.load(fh)
    return {}


def _save_pins(pins: dict) -> None:
    PINS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(PINS_FILE, "w", encoding="utf-8") as fh:
        json.dump(pins, fh, indent=2)


def _hash_pin(user_id: str, pin: str) -> str:
    # Salted with user_id so same PIN yields different hashes per user
    return hashlib.sha256(f"{user_id}:{pin}".encode()).hexdigest()


@router.post("/login", response_model=UserLoginResponse)
def login_user(body: UserLogin):
    """Login a user with email and password."""
    user_id = verify_user_credentials(body.email.strip(), body.password)
    if not user_id:
        raise HTTPException(status_code=401, detail="Invalid email or password.")
    user = _user_manager.get_user(user_id)
    if not user:
        raise HTTPException(status_code=404, detail="User profile not found.")
    return UserLoginResponse(user_id=user_id, name=user.name, email=body.email.strip().lower())


@router.post("/register", response_model=UserLoginResponse, status_code=201)
def register_user_with_credentials(body: UserRegister):
    """Register a new user with name, email and password."""
    name = body.name.strip()
    email = body.email.strip().lower()
    if not name:
        raise HTTPException(status_code=422, detail="Name cannot be empty.")
    if "@" not in email:
        raise HTTPException(status_code=422, detail="Enter a valid email address.")
    if len(body.password) < 6:
        raise HTTPException(status_code=422, detail="Password must be at least 6 characters.")
    if email_exists(email):
        raise HTTPException(status_code=409, detail="An account with this email already exists.")
    user_id = _user_manager.register_user(name)
    register_user_credentials(user_id, email, body.password)
    return UserLoginResponse(user_id=user_id, name=name, email=email)


@router.get("/", response_model=list[UserResponse])
def list_users():
    """Return all registered users, sorted by last active."""
    return [UserResponse(**u.to_dict()) for u in _user_manager.list_users()]


@router.post("/", response_model=UserResponse, status_code=201)
def register_user(body: UserCreate):
    """Register a new user and return their profile."""
    if not body.name.strip():
        raise HTTPException(status_code=422, detail="Name cannot be empty")
    user_id = _user_manager.register_user(body.name.strip())
    user = _user_manager.get_user(user_id)
    return UserResponse(**user.to_dict())


@router.get("/{user_id}", response_model=UserResponse)
def get_user(user_id: str):
    user = _user_manager.get_user(user_id)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    return UserResponse(**user.to_dict())


@router.delete("/{user_id}")
def delete_user(user_id: str):
    user = _user_manager.get_user(user_id)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    _user_manager.delete_user(user_id, delete_data=True)
    # Also remove PIN
    pins = _load_pins()
    pins.pop(user_id, None)
    _save_pins(pins)
    return {"message": f"Deleted user {user_id}"}


# ─── PIN endpoints ─────────────────────────────────────────────────────────

@router.get("/{user_id}/pin/status")
def pin_status(user_id: str):
    """Check whether the user has a PIN set."""
    pins = _load_pins()
    return {"has_pin": user_id in pins}


@router.post("/{user_id}/pin")
def set_pin(user_id: str, body: PinSet):
    """Set or update the PIN for a user (4-8 characters)."""
    if not _user_manager.get_user(user_id):
        raise HTTPException(status_code=404, detail="User not found")
    pin = body.pin.strip()
    if not (4 <= len(pin) <= 8) or not pin.isdigit():
        raise HTTPException(status_code=422, detail="PIN must be 4-8 digits")
    pins = _load_pins()
    pins[user_id] = _hash_pin(user_id, pin)
    _save_pins(pins)
    return {"message": "PIN set successfully"}


@router.delete("/{user_id}/pin")
def remove_pin(user_id: str):
    """Remove PIN protection for a user."""
    pins = _load_pins()
    if user_id not in pins:
        raise HTTPException(status_code=404, detail="No PIN set for this user")
    pins.pop(user_id)
    _save_pins(pins)
    return {"message": "PIN removed"}


@router.post("/{user_id}/pin/verify")
def verify_pin(user_id: str, body: PinVerify):
    """Verify PIN — returns {valid: bool}. Never reveals whether PIN exists."""
    pins = _load_pins()
    stored = pins.get(user_id)
    if stored is None:
        return {"valid": True}  # no PIN set → always passes
    return {"valid": _hash_pin(user_id, body.pin.strip()) == stored}
