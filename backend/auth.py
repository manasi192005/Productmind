"""
auth.py — Authentication utilities for ProductMind.

Handles:
- Password hashing / verification (hashlib SHA-256)
- JWT token creation / verification (PyJWT)
- User persistence (users.json flat-file)
- FastAPI dependency for protected routes
"""

import json
import base64
import hashlib
import hmac
import logging
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

try:
    import jwt  # type: ignore
except ModuleNotFoundError:
    jwt = None
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials


class _FallbackJWT:
    """Minimal HS256 JWT implementation used when PyJWT is unavailable."""

    class ExpiredSignatureError(Exception):
        pass

    class InvalidTokenError(Exception):
        pass

    @staticmethod
    def _b64url_encode(value: bytes) -> str:
        return base64.urlsafe_b64encode(value).rstrip(b"=").decode("utf-8")

    @staticmethod
    def _b64url_decode(value: str) -> bytes:
        padding = "=" * (-len(value) % 4)
        return base64.urlsafe_b64decode(f"{value}{padding}")

    @classmethod
    def encode(cls, payload: dict, secret: str, algorithm: str = "HS256") -> str:
        if algorithm != "HS256":
            raise ValueError("Fallback JWT encoder only supports HS256.")

        header = {"alg": "HS256", "typ": "JWT"}
        normalized_payload = {}
        for key, value in payload.items():
            if isinstance(value, datetime):
                normalized_payload[key] = int(value.timestamp())
            else:
                normalized_payload[key] = value

        header_segment = cls._b64url_encode(
            json.dumps(header, separators=(",", ":")).encode("utf-8")
        )
        payload_segment = cls._b64url_encode(
            json.dumps(normalized_payload, separators=(",", ":")).encode("utf-8")
        )
        signing_input = f"{header_segment}.{payload_segment}".encode("utf-8")
        signature = hmac.new(secret.encode("utf-8"), signing_input, hashlib.sha256).digest()
        signature_segment = cls._b64url_encode(signature)
        return f"{header_segment}.{payload_segment}.{signature_segment}"

    @classmethod
    def decode(cls, token: str, secret: str, algorithms: list[str] | None = None) -> dict:
        if algorithms and "HS256" not in algorithms:
            raise cls.InvalidTokenError("Unsupported JWT algorithm.")

        parts = token.split(".")
        if len(parts) != 3:
            raise cls.InvalidTokenError("Malformed token.")

        header_segment, payload_segment, signature_segment = parts
        signing_input = f"{header_segment}.{payload_segment}".encode("utf-8")
        expected_signature = hmac.new(
            secret.encode("utf-8"),
            signing_input,
            hashlib.sha256,
        ).digest()
        actual_signature = cls._b64url_decode(signature_segment)

        if not hmac.compare_digest(expected_signature, actual_signature):
            raise cls.InvalidTokenError("Invalid token signature.")

        try:
            payload = json.loads(cls._b64url_decode(payload_segment).decode("utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError, ValueError) as exc:
            raise cls.InvalidTokenError("Invalid token payload.") from exc

        exp = payload.get("exp")
        if exp is not None and float(exp) < datetime.now(timezone.utc).timestamp():
            raise cls.ExpiredSignatureError("Token has expired.")

        return payload


if jwt is None:
    jwt = _FallbackJWT

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
