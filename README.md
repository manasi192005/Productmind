# 🧠 ProductMind — AI-Powered Product Recommendation Agent
> A full-stack GenAI product recommendation system built with a **Plan-and-Solve Agent Architecture**. Users describe what they want in plain English — the AI agent plans, searches, filters, ranks, and explains the best match in real time.

---

## ✨ Features

- 🤖 **Planning-Based Agent** — LLM generates a dynamic execution plan per query (semantic search → filter → sort → rank)
- 🔍 **Semantic Search** — Sentence-transformer embeddings find products by *meaning*, not just keywords
- 💬 **Conversational Context** — Multi-turn chat history passed to the agent for follow-up queries
- 🏆 **Synthesis + Comparison** — Separate LLM nodes explain the top pick and generate pros/cons for alternatives
- 🔐 **JWT Authentication** — Secure signup/login with token-protected recommendation routes
- 🌐 **Multi-Page Frontend** — Discover, Compare, Saved, and Insights pages with shared navigation
- ⚡ **Typewriter Effect** — AI reasoning text animates character-by-character on render
- 📱 **Responsive Design** — Mobile bottom nav + desktop sidebar, built with Tailwind CSS

---

## 🏗️ Architecture

```
User Query
    │
    ▼
┌─────────────────────────────────────────┐
│           PLANNER AGENT                 │  ← LLaMA3 via Groq
│  Generates tool execution plan          │
│  e.g. [semantic_search → budget_filter] │
└────────────────┬────────────────────────┘
                 │
    ┌────────────▼────────────┐
    │     TOOL EXECUTOR       │
    │  ├ semantic_search      │  ← sentence-transformers + SQLite
    │  ├ budget_filter        │
    │  ├ category_filter      │
    │  └ sort_price           │
    └────────────┬────────────┘
                 │
    ┌────────────▼────────────┐
    │   SYNTHESIS AGENT       │  ← LLaMA3 via Groq
    │  Picks best match +     │
    │  generates explanation  │
    └────────────┬────────────┘
                 │
    ┌────────────▼────────────┐
    │   COMPARISON AGENT      │  ← LLaMA3 via Groq
    │  Scores alternatives,   │
    │  generates pros & cons  │
    └─────────────────────────┘
```

---

## 🗂️ Project Structure

```
product-agent/
├── backend/
│   ├── main.py              # FastAPI app — all routes
│   ├── agent.py             # All 4 agent nodes + tool executor
│   ├── auth.py              # JWT signup/login logic
│   ├── models.py            # Pydantic request/response models
│   ├── database.py          # SQLite ORM (SQLAlchemy)
│   ├── embeddings.py        # Sentence-transformer search engine
│   ├── seed_db.py           # Seeds database with product catalog
│   └── users.json           # Auto-created user store
│
├── frontend/
│   ├── common.js            # Shared auth, nav, API helpers, typewriter
│   ├── home.html            # Discover — main search + results page
│   ├── compare.html         # Side-by-side product comparison
│   ├── saved.html           # Bookmarked products
│   ├── insights.html        # Usage insights & stats
│   └── authenticationpage.html  # Login / Signup
│
├── products.json            # Source product catalog (seeded into DB)
├── productmind.db           # SQLite database (auto-generated)
├── requirements.txt
└── .env                     # API keys (not committed)
```

---

## ⚙️ Tech Stack

| Layer | Technology |
|-------|-----------|
| **Backend** | FastAPI (Python) |
| **LLM** | LLaMA 3 8B via [Groq](https://groq.com) |
| **Embeddings** | `sentence-transformers` (all-MiniLM-L6-v2) |
| **Database** | SQLite + SQLAlchemy ORM |
| **Auth** | JWT (python-jose) |
| **Frontend** | Vanilla HTML/CSS/JS + Tailwind CSS CDN |
| **Icons** | Google Material Symbols |
| **Fonts** | Manrope + Inter (Google Fonts) |

---

## 🚀 Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/your-username/product-agent.git
cd product-agent
```

### 2. Create a virtual environment

```bash
python -m venv venv

# macOS / Linux
source venv/bin/activate

# Windows
venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Set up environment variables

Create a `.env` file in the `backend/` directory:

```env
GROQ_API_KEY=your_groq_api_key_here
SECRET_KEY=your_jwt_secret_key_here
```

> Get a free Groq API key at [console.groq.com](https://console.groq.com)

### 5. Seed the database

```bash
cd backend
python seed_db.py
```

### 6. Start the backend

```bash
uvicorn main:app --reload --port 8000
```

The API will be live at `http://localhost:8000`  
Interactive docs available at `http://localhost:8000/docs`

### 7. Open the frontend

Open `frontend/authenticationpage.html` directly in your browser (no separate server needed).

> **Tip:** Use a local server like VS Code Live Server or `python -m http.server 5500` from the `frontend/` directory for the best experience.

---

## 📬 API Endpoints

| Method | Endpoint | Auth | Description |
|--------|----------|------|-------------|
| `POST` | `/auth/signup` | ❌ | Register new user |
| `POST` | `/auth/login` | ❌ | Login, returns JWT token |
| `POST` | `/recommend` | ✅ JWT | Run full agent pipeline |
| `GET` | `/products` | ✅ JWT | List all products |
| `GET` | `/docs` | ❌ | Swagger UI |

### Example: Get a recommendation

```bash
# 1. Login
curl -X POST http://localhost:8000/auth/login \
  -H "Content-Type: application/json" \
  -d '{"username": "alice", "password": "secret123"}'

# 2. Use the returned token
curl -X POST http://localhost:8000/recommend \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer <your_token>" \
  -d '{"query": "I need noise cancelling headphones for travel under 5000"}'
```

### Example response shape

```json
{
  "top_pick": { "id": "...", "name": "...", "price": 3499, ... },
  "alternatives": [ {...}, {...} ],
  "explanation": "The Sony WH-1000XM5 is the ideal pick because...",
  "extracted_preferences": { "category": "audio", "keywords": [] },
  "agent_plan": [
    "Agent Action: Executing [semantic_search] because: ...",
    "Agent Action: Executing [budget_filter] because: ...",
    "Agent Action: Executing [category_filter] because: ..."
  ],
  "comparison_data": {
    "products": [
      { "id": "...", "score": 94, "verdict": "...", "pros": [...], "cons": [...] }
    ]
  }
}
```

---

## 🧠 Agent Deep Dive

The agent in `agent.py` follows a **Plan → Execute → Synthesize → Compare** cycle:

| Node | Function | Model |
|------|----------|-------|
| **Planner** | `generate_plan()` | LLaMA3-8B (temp 0.1) |
| **Tool Executor** | `run_recommendation_agent()` loop | Deterministic Python |
| **Synthesis** | `synthesize_results()` | LLaMA3-8B (temp 0.3) |
| **Comparison** | `generate_comparison()` | LLaMA3-8B (temp 0.2) |

Each node has a **hardcoded fallback** — if the LLM call fails (rate limit, timeout, etc.), the system degrades gracefully to heuristic-based logic rather than crashing.

---

## 🔧 Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `GROQ_API_KEY` | — | Required. Your Groq API key |
| `SECRET_KEY` | — | Required. JWT signing secret |
| `GROQ_MODEL` | `llama3-8b-8192` | Groq model to use |

---

## 📦 Requirements

```
fastapi
uvicorn[standard]
groq
python-jose[cryptography]
sqlalchemy
sentence-transformers
pydantic
python-dotenv
python-multipart
```


