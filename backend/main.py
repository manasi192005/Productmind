"""
main.py — ProductMind FastAPI Application.

Endpoints:
  GET  /health      → API status check
  POST /signup      → Register new user
  POST /login       → Login, returns JWT token
  POST /recommend   → AI product recommendations (protected)

Run with:
  uvicorn main:app --reload --port 8000
"""

import logging
from dotenv import load_dotenv
load_dotenv()   # Load .env before any other imports that read env vars

from fastapi import FastAPI, HTTPException, Depends, status
from fastapi.middleware.cors import CORSMiddleware

from auth import (
    create_user,
    authenticate_user,
    create_token,
    get_current_user,
)
from agent import run_recommendation_agent
from models import (
    SignupRequest,
    LoginRequest,
    TokenResponse,
    RecommendRequest,
    RecommendResponse,
)

# ─── LOGGING SETUP ───────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ─── APP INIT ────────────────────────────────────────────────────────────────

app = FastAPI(
    title="ProductMind API",
    description="AI-powered product recommendation backend using GROQ LLM.",
    version="1.0.0",
)

# Allow requests from the frontend (running on any local port)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],      # In production, specify exact frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─── ENDPOINTS ───────────────────────────────────────────────────────────────

@app.get("/health", tags=["System"])
def health_check():
    """Check if the API is running."""
    logger.info("Health check called.")
    return {"status": "ok", "message": "ProductMind API is running 🚀"}


@app.post("/signup", tags=["Auth"], status_code=status.HTTP_201_CREATED)
def signup(data: SignupRequest):
    """
    Register a new user.
    - Checks for duplicate username/email
    - Hashes password before storing
    """
    logger.info(f"Signup attempt for username: {data.username}")

    # Validate input lengths
    if len(data.username) < 3:
        raise HTTPException(status_code=400, detail="Username must be at least 3 characters.")
    if len(data.password) < 6:
        raise HTTPException(status_code=400, detail="Password must be at least 6 characters.")

    try:
        user = create_user(data.username, data.email, data.password)
        logger.info(f"User '{data.username}' registered successfully.")
        return {
            "message": f"Account created successfully! Welcome, {user['username']}.",
            "username": user["username"],
        }
    except ValueError as e:
        raise HTTPException(status_code=409, detail=str(e))


@app.post("/login", response_model=TokenResponse, tags=["Auth"])
def login(data: LoginRequest):
    """
    Login with username + password.
    Returns a JWT token on success.
    """
    logger.info(f"Login attempt for username: {data.username}")

    user = authenticate_user(data.username, data.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid username or password. Please check your credentials or sign up first.",
        )

    token = create_token(data.username)
    logger.info(f"Login successful for: {data.username}")
    return TokenResponse(
        access_token=token,
        token_type="bearer",
        username=data.username,
    )


@app.post("/recommend", tags=["AI"])
def recommend(
    data: RecommendRequest,
    current_user: str = Depends(get_current_user),   # Protected route
):
    """
    Get AI-powered product recommendations.
    Requires a valid JWT token in the Authorization header.

    The agent:
    1. Extracts preferences from the natural language query
    2. Filters the product catalogue
    3. Ranks and explains the best match using GROQ
    """
    logger.info(f"Recommendation request from '{current_user}': '{data.query}'")

    if not data.query or len(data.query.strip()) < 3:
        raise HTTPException(status_code=400, detail="Query is too short. Please describe what you're looking for.")

    try:
        result = run_recommendation_agent(
            query=data.query.strip(),
            session_id=data.session_id,
            history=data.history
        )
        logger.info(f"Recommendation done for '{current_user}': top_pick = {result['top_pick']['name']}")
        return result

    except RuntimeError as e:
        # GROQ API key missing
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        logger.error(f"Recommendation error for '{current_user}': {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Something went wrong while generating recommendations. Please try again.")
