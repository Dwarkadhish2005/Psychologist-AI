"""
JWT authentication helpers for Psychologist AI
"""

import json
import os
import hashlib
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from jose import JWTError, jwt

# -- Config ------------------------------------------------------------------
SECRET_KEY = os.getenv("JWT_SECRET_KEY", "psych-ai-secret-key-change-in-production-2026")
ALGORITHM = os.getenv("JWT_ALGORITHM", "HS256")
ACCESS_TOKEN_EXPIRE_HOURS = int(os.getenv("ACCESS_TOKEN_EXPIRE_HOURS", "24"))

USER_CREDS_FILE = Path(os.getenv("USER_CREDS_FILE", "data/user_memory/user_credentials.json"))

bearer_scheme = HTTPBearer(auto_error=False)


# -- Password hashing (scrypt via standard library -- no extra deps) ----------

def _hash_password(password: str, salt: Optional[str] = None):
    """Hash a password with scrypt. Returns (hash_hex, salt_hex)."""
    if salt is None:
        salt = os.urandom(16).hex()
    dk = hashlib.scrypt(
        password.encode("utf-8"),
        salt=bytes.fromhex(salt),
        n=16384, r=8, p=1,
    )
    return dk.hex(), salt


def _verify_password(password: str, stored_hash: str, salt: str) -> bool:
    computed, _ = _hash_password(password, salt)
    return computed == stored_hash


# -- Credential store helpers -------------------------------------------------

def _load_creds() -> dict:
    if USER_CREDS_FILE.exists():
        with open(USER_CREDS_FILE, encoding="utf-8") as fh:
            return json.load(fh)
    return {}


def _save_creds(creds: dict) -> None:
    USER_CREDS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(USER_CREDS_FILE, "w", encoding="utf-8") as fh:
        json.dump(creds, fh, indent=2)


def get_user_id_by_email(email: str) -> Optional[str]:
    """Return user_id for a given email, or None."""
    creds = _load_creds()
    email = email.strip().lower()
    for uid, data in creds.items():
        if data.get("email", "").lower() == email:
            return uid
    return None


def register_user_credentials(user_id: str, email: str, password: str) -> None:
    """Save hashed password for a user_id -> email mapping."""
    creds = _load_creds()
    pw_hash, salt = _hash_password(password)
    creds[user_id] = {
        "email": email.strip().lower(),
        "hash": pw_hash,
        "salt": salt,
    }
    _save_creds(creds)


def verify_user_credentials(email: str, password: str) -> Optional[str]:
    """Return user_id if credentials are valid, else None."""
    creds = _load_creds()
    email = email.strip().lower()
    for uid, data in creds.items():
        if data.get("email", "").lower() == email:
            stored_hash = data.get("hash") or data.get("hashed_password", "")
            salt = data.get("salt")
            if salt and _verify_password(password, stored_hash, salt):
                return uid
    return None


def email_exists(email: str) -> bool:
    creds = _load_creds()
    email = email.strip().lower()
    return any(d.get("email", "").lower() == email for d in creds.values())


# -- JWT helpers --------------------------------------------------------------

def create_access_token(user_id: str) -> str:
    expire = datetime.now(timezone.utc) + timedelta(hours=ACCESS_TOKEN_EXPIRE_HOURS)
    payload = {"sub": user_id, "exp": expire}
    return jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)


def decode_token(token: str) -> Optional[str]:
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        return payload.get("sub")
    except JWTError:
        return None


# -- FastAPI dependency -------------------------------------------------------

def get_current_user_id(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(bearer_scheme),
) -> str:
    if credentials is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated",
            headers={"WWW-Authenticate": "Bearer"},
        )
    user_id = decode_token(credentials.credentials)
    if user_id is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return user_id
