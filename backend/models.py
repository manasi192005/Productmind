"""
models.py — Pydantic request/response models for ProductMind API.
Defines the data shapes expected by each endpoint.
"""

from pydantic import BaseModel, EmailStr
from typing import List, Optional, Any, Dict


# ─── AUTH MODELS ─────────────────────────────────────────────────────────────

class SignupRequest(BaseModel):
    """Payload for POST /signup"""
    username: str
    email: str          # plain str; add EmailStr if email-validator installed
    password: str


class LoginRequest(BaseModel):
    """Payload for POST /login"""
    username: str
    password: str


class TokenResponse(BaseModel):
    """Returned on successful login"""
    access_token: str
    token_type: str = "bearer"
    username: str


# ─── RECOMMENDATION MODELS ────────────────────────────────────────────────────

class RecommendRequest(BaseModel):
    """Payload for POST /recommend"""
    query: str                  # natural language, e.g. "study table under 5000"
    session_id: Optional[str] = None
    history: Optional[List[dict]] = []

class Product(BaseModel):
    """Represents a single product from the dataset"""
    id: str
    name: str
    category: str
    price: float
    brand: str
    features: List[str]
    tags: List[str]


class ExtractedPreferences(BaseModel):
    """Structured preferences the AI extracted from the user query"""
    category: Optional[str] = None
    max_price: Optional[float] = None
    min_price: Optional[float] = None
    keywords: List[str] = []
    use_case: Optional[str] = None
    intent: Optional[str] = None


class RecommendResponse(BaseModel):
    """Full recommendation response returned to the frontend"""
    top_pick: Product
    alternatives: List[Product]
    explanation: str
    extracted_preferences: ExtractedPreferences
    agent_plan: List[str]
    comparison_data: Optional[Dict[str, Any]] = None
    session_id: Optional[str] = None
