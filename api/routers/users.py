"""
User management router
"""

import sys
from pathlib import Path

from fastapi import APIRouter, HTTPException

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from inference.phase4_user_manager import UserManager
from api.schemas import UserCreate, UserResponse

router = APIRouter()
_user_manager = UserManager(storage_dir="data/user_memory")


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
    return {"message": f"Deleted user {user_id}"}
