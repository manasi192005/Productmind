"""
auth.py — Authentication utilities for ProductMind.

Handles:
- Password hashing / verification (hashlib SHA-256)
- JWT token creation / verification (PyJWT)
- User persistence (users.json flat-file)
- FastAPI dependency for protected routes
"""

import json
import hashlib
import logging
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import jwt
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

# ─── CONFIG ──────────────────────────────────────────────────────────────────

# Secret key used to sign JWT tokens.
# In production, load from environment variable.
JWT_SECRET  = os.getenv("JWT_SECRET", "productmind_super_secret_2024")
JWT_ALGO    = "HS256"
JWT_EXPIRY_HOURS = 24           # Token valid for 24 hours

USERS_FILE  = Path(__file__).parent / "users.json"

logger = logging.getLogger(__name__)
security = HTTPBearer()          # Expects "Authorization: Bearer <token>"


# ─── USER PERSISTENCE ────────────────────────────────────────────────────────

def _load_users() -> dict:
    """Load users dict from JSON file. Returns empty dict if file is missing."""
    if not USERS_FILE.exists():
        return {}
    try:
        with open(USERS_FILE, "r") as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError):
        return {}


def _save_users(users: dict) -> None:
    """Persist users dict to JSON file."""
    with open(USERS_FILE, "w") as f:
        json.dump(users, f, indent=2)


# ─── PASSWORD HASHING ────────────────────────────────────────────────────────

def hash_password(plain: str) -> str:
    """Hash a plain-text password using SHA-256. Returns hex digest."""
    return hashlib.sha256(plain.encode()).hexdigest()


def verify_password(plain: str, hashed: str) -> bool:
    """Check that a plain-text password matches its stored hash."""
    return hash_password(plain) == hashed


# ─── USER CRUD ───────────────────────────────────────────────────────────────

def create_user(username: str, email: str, password: str) -> dict:
    """
    Register a new user.
    Raises ValueError if username already exists.
    Returns the saved user record (without password).
    """
    users = _load_users()

    if username in users:
        raise ValueError(f"Username '{username}' is already taken.")

    # Check if email is already registered
    for u in users.values():
        if u.get("email") == email:
            raise ValueError(f"Email '{email}' is already registered.")

    users[username] = {
        "username": username,
        "email": email,
        "password_hash": hash_password(password),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    _save_users(users)
    logger.info(f"New user registered: {username}")

    return {"username": username, "email": email}


def authenticate_user(username: str, password: str) -> Optional[dict]:
    """
    Validate credentials.
    Returns the user record on success, None on failure.
    """
    users = _load_users()
    user = users.get(username)
    if user and verify_password(password, user["password_hash"]):
        logger.info(f"User logged in: {username}")
        return user
    return None


# ─── JWT ─────────────────────────────────────────────────────────────────────

def create_token(username: str) -> str:
    """Create a signed JWT token that expires after JWT_EXPIRY_HOURS."""
    payload = {
        "sub": username,
        "exp": datetime.now(timezone.utc) + timedelta(hours=JWT_EXPIRY_HOURS),
        "iat": datetime.now(timezone.utc),
    }
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGO)


def decode_token(token: str) -> dict:
    """
    Decode and validate a JWT token.
    Raises jwt.ExpiredSignatureError or jwt.InvalidTokenError on failure.
    """
    return jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGO])


# ─── FASTAPI DEPENDENCY ───────────────────────────────────────────────────────

def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security),
) -> str:
    """
    FastAPI dependency for protected routes.
    Extracts the bearer token, verifies it, and returns the username.

    Usage:
        @app.post("/recommend")
        def recommend(username: str = Depends(get_current_user)):
            ...
    """
    token = credentials.credentials
    try:
        payload = decode_token(token)
        username: str = payload.get("sub")
        if not username:
            raise HTTPException(status_code=401, detail="Invalid token payload.")
        return username
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token has expired. Please log in again.")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid token. Please log in again.")
