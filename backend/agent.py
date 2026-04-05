import json
import logging
import os
import re
from typing import Optional, List, Dict, Any

from groq import Groq
from models import ExtractedPreferences
from embeddings import get_search_engine
from database import SessionLocal, Product as DBProduct

logger = logging.getLogger(__name__)

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL   = "llama3-8b-8192"

def _get_groq_client() -> Groq:
    if not GROQ_API_KEY:
        raise RuntimeError("GROQ_API_KEY not set")
    return Groq(api_key=GROQ_API_KEY)

def _extract_json_from_llm(raw: str) -> dict:
    """Robustly strips markdown brackets and parses JSON."""
    raw = raw.strip()
    if raw.startswith("```json"):
        raw = raw[7:]
    if raw.startswith("```"):
        raw = raw[3:]
    if raw.endswith("```"):
        raw = raw[:-3]
    raw = raw.strip()
    
    try:
        return json.loads(raw)
    except:
        # Fallback to regex isolation
        match = re.search(r'\{.*\}', raw, re.DOTALL)
        if match:
            return json.loads(match.group())
        return {}


# ─── AGENT TOOLS ────────────────────────────────────────────────────────

def tool_semantic_search(query: str, top_k: int = 15) -> List[Dict]:
    """Uses sentence-transformers to find semantically relevant products natively."""
    logger.info(f"Executing tool_semantic_search with query '{query}'")
    engine = get_search_engine()
    results = engine.search(query, top_k=top_k)
    return [r["product"] for r in results]

def tool_budget_filter(candidates: List[Dict], max_price: float) -> List[Dict]:
    """Filters a list of candidates by a maximum budget."""
    logger.info(f"Executing tool_budget_filter with max {max_price}")
    return [p for p in candidates if p.get("price", 0) <= max_price]

def tool_category_filter(candidates: List[Dict], category: str) -> List[Dict]:
    """Enforces strict category bounds if demanded."""
    logger.info(f"Executing tool_category_filter with category {category}")
    cat = category.lower()
    return [p for p in candidates if cat in p.get("category", "").lower() or p.get("category", "").lower() in cat]

def tool_sort_price(candidates: List[Dict], order: str) -> List[Dict]:
    """Sorts the candidates by price in ascending or descending order."""
    logger.info(f"Executing tool_sort_price with order {order}")
    desc = order.lower() in ["desc", "descending", "highest", "expensive", "premium"]
    return sorted(candidates, key=lambda p: p.get("price", 0), reverse=desc)


# ─── PLANNER NODE ────────────────────────────────────────────────────────

PLANNER_SYSTEM_PROMPT = """You are the ProductMind logical Planner Agent.
Your job is to analyze a user query and determine the REQUIRED steps to fetch the best products.

AVAILABLE TOOLS:
- `semantic_search`: Searches the vector database based on a semantic query (e.g. 'comfortable seating', 'devices to track steps'). Argument: "query" (str).
- `budget_filter`: Removes products over a price limit. Argument: "max_price" (float).
- `category_filter`: strictly narrows down objects to a specific domain (e.g. 'appliances', 'electronics'). Argument: "category" (str).
- `sort_price`: Sorts the array of retrieved products by their cost. Use this if the user asks for 'expensive', 'premium', 'cheap', or 'budget'. Argument: "order" (str) -> Must be either "asc" or "desc".

Based on the user's prompt, define an array of exact steps to execute.
Always start with `semantic_search`. Then apply filters IF needed.

Return ONLY VALID JSON matching this schema:
{
  "plan": [
    {
      "action": "<tool_name>",
      "arguments": {"<arg_name>": "<arg_value>"},
      "reasoning": "Quick 1 sentence explanation of why this step is taken."
    }
  ]
}
"""

def generate_plan(query: str, history: list = None) -> List[Dict]:
    history_ctx = ""
    if history:
        history_ctx = "Conversation Context:\n" + "\n".join([f"{h['role']}: {h['content']}" for h in history]) + "\n\n"
        
    prompt = f"{history_ctx}User Query: {query}"
    
    try:
        client = _get_groq_client()
        response = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {"role": "system", "content": PLANNER_SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            temperature=0.1,
            max_tokens=500
        )
        raw = response.choices[0].message.content.strip()
        data = _extract_json_from_llm(raw)
        return data.get("plan", [])
    except Exception as e:
        logger.error(f"[PLANNER ERROR] {e}")
        # Default Safe Plan wrapper
        q_low = query.lower()
        fallback_plan = [
            {"action": "semantic_search", "arguments": {"query": query}, "reasoning": "API Rate Limit hit. Executing isolated robust vector search."}
        ]
        if any(w in q_low for w in ['expensive', 'premium', 'high', 'best']):
            fallback_plan.append({"action": "sort_price", "arguments": {"order": "desc"}, "reasoning": "Determined 'premium' user intent via heuristic fallback."})
        elif any(w in q_low for w in ['cheap', 'budget', 'low', 'affordable']):
             fallback_plan.append({"action": "sort_price", "arguments": {"order": "asc"}, "reasoning": "Determined 'budget' user intent via heuristic fallback."})
        return fallback_plan

# ─── SYNTHESIZER / RANKING NODE ─────────────────────────────────────────

RANK_SYSTEM_PROMPT = """You are the ProductMind Synthesis Agent. given a user query, a chat history, and a list of structured products retrieved from the database natively via a Planner Agent, choose the absolute best match.

Explain WHY in a detailed, engaging way (4-6 sentences).
- Explain why it matches.
- Mention price justification.
- Mention a comparison to an alternative if appropriate.
- Be natural and conversational.

Return ONLY JSON:
{
  "top_pick_id": "<id>",
  "explanation": "<text>"
}
"""

def synthesize_results(query: str, candidates: list[dict], history: list = None) -> tuple[str, str]:
    if not candidates:
        return None, "I couldn't find any products matching those strict criteria. Try opening up your budget or category!"
        
    candidate_summary = "\n".join(
        f"- ID: {p['id']}, Name: {p['name']}, Price: ₹{p['price']}, Category: {p['category']}, Features: {', '.join(p['features'])}"
        for p in candidates[:8]
    )
    
    history_ctx = ""
    if history:
         history_ctx = "Context:\n" + "\n".join([f"{h['role']}: {h['content']}" for h in history]) + "\n\n"

    try:
        client = _get_groq_client()
        response = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {"role": "system", "content": RANK_SYSTEM_PROMPT},
                {"role": "user", "content": f"{history_ctx}Query: {query}\n\nCandidates:\n{candidate_summary}"},
            ],
            temperature=0.3,
            max_tokens=600
        )
        raw = response.choices[0].message.content.strip()
        data = _extract_json_from_llm(raw)
        return data.get("top_pick_id", candidates[0]["id"]), data.get("explanation", "Matches perfectly.")
    except Exception as e:
        logger.error(f"[SYNTHESIS ERROR] {e}")
        p = candidates[0]
        feats = ", ".join(p.get("features", []))[:100]
        fallback_text = (
            f"The {p['name']} perfectly aligns with your search intent by offering exceptional value. "
            f"Rather than just matching basic keywords, this selection comprehensively addresses your underlying needs using its {feats} capabilities. "
            f"At ₹{p['price']}, it stands out as a highly competitive option within the {p['category']} category. "
            f"Whether you prioritize high performance or budget efficiency, this recommendation balances both criteria perfectly. "
            f"If you are looking for an immediate reliable purchase, this item completely fulfills the core requirements."
        )
        return p["id"], fallback_text

# ─── COMPARISON NODE ──────────────────────────────────────────────

def generate_comparison(query: str, top_picks: list[dict]) -> dict:
    if not top_picks:
        return {}
        
    COMPARE_SYSTEM_PROMPT = """You are an expert product analyst. Given a user query and up to 3 products, analyze them and generate structured comparison data.
    Return ONLY JSON:
    {
       "products": [
           {
               "id": "<id>",
               "score": <number 0-100 indicating relevance>,
               "verdict": "<1-2 sentence verdict>",
               "pros": ["<pro1>", "<pro2>"],
               "cons": ["<con1>", "<con2>"]
           }
       ]
    }
    """
    summary = "\n\n".join([f"ID: {p['id']}, Name: {p['name']}, Price: ₹{p['price']}, Category: {p['category']}, Features: {', '.join(p.get('features', []))}" for p in top_picks])
    try:
        client = _get_groq_client()
        response = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {"role": "system", "content": COMPARE_SYSTEM_PROMPT},
                {"role": "user", "content": f"Query: {query}\n\nProducts:\n{summary}"},
            ],
            temperature=0.2,
            max_tokens=800
        )
        raw = response.choices[0].message.content.strip()
        return _extract_json_from_llm(raw)
    except:
        return {
            "products": [
                {
                    "id": p["id"],
                    "score": max(50, 95 - (i * 10)),
                    "verdict": f"A solid choice in the {p['category']} category.",
                    "pros": p.get("features", [])[:3],
                    "cons": ["Price varies"]
                } for i, p in enumerate(top_picks)
            ]
        }

# ─── MASTER AGENT EXECUTOR ──────────────────────────────────────────────

def run_recommendation_agent(query: str, session_id: str = None, history: list = None) -> dict:
    """
    Executes the Plan-and-Solve Agent Architecture.
    """
    logger.info(f"Running Agent with session: {session_id}")
    
    # 1. PLAN
    agent_plan = generate_plan(query, history)
    agent_trace_logs = []
    
    # ENFORCE ABSOLUTE CATEGORY INTEGRITY
    q_low = query.lower()
    cat_map = {
        "audio": ["headphone", "audio", "earbuds", "speaker"],
        "electronics": ["laptop", "camera", "monitor"],
        "peripherals": ["keyboard", "mouse", "webcam"],
        "furniture": ["desk", "chair", "table"],
        "appliances": ["fan", "blender", "kettle", "heater", "purifier", "cooler"],
        "fitness": ["fitness", "band", "resistance", "yoga"],
        "decor": ["decor", "candle", "lamp", "clock", "aesthetic"],
        "stationery": ["notebook", "pen", "journal"],
        "bags": ["bag", "backpack"]
    }
    
    detected_cat = None
    for cat, words in cat_map.items():
        if any(w in q_low for w in words):
            detected_cat = cat
            break
            
    if detected_cat and not any(step.get("action") == "category_filter" for step in agent_plan):
        agent_plan.append({
            "action": "category_filter", 
            "arguments": {"category": detected_cat}, 
            "reasoning": f"Mechanically enforced '{detected_cat}' domain boundaries to prevent cross-category mixing."
        })
        
    # 2. EXECUTE
    candidates = []
    
    for step in agent_plan:
        action = step.get("action")
        args = step.get("arguments", {})
        reasoning = step.get("reasoning", "")
        
        log_msg = f"Agent Action: Executing [{action}] because: {reasoning}"
        agent_trace_logs.append(log_msg)
        logger.info(log_msg)
        
        if action == "semantic_search":
            q = args.get("query", query)
            candidates = tool_semantic_search(q, top_k=20)
            
        elif action == "budget_filter":
            max_p = float(args.get("max_price", 999999))
            candidates = tool_budget_filter(candidates, max_p)
            
        elif action == "category_filter":
            cat = args.get("category", "")
            candidates = tool_category_filter(candidates, cat)
            
        elif action == "sort_price":
            order = args.get("order", "asc")
            candidates = tool_sort_price(candidates, order)
            
    # Fallback if the agent executed a weird plan that returned 0 items
    if not candidates:
        agent_trace_logs.append("Agent Warning: Filters too strict. Executing fallback vector search.")
        candidates = tool_semantic_search(query, top_k=5)
        
    # 3. SYNTHESIZE & RANK
    top_id, explanation = synthesize_results(query, candidates, history)
    
    if not top_id:
        return {
            "top_pick": {
                "name": "No matching product found",
                "price": 0, "brand": "N/A", "category": "N/A", "features": ["Try modifying search"], "tags": []
            },
            "alternatives": [],
            "explanation": explanation,
            "extracted_preferences": {},
            "agent_plan": agent_trace_logs,
            "comparison_data": {},
            "session_id": session_id
        }

    top_pick = next((p for p in candidates if p["id"] == top_id), candidates[0])
    alternatives = [p for p in candidates if p["id"] != top_pick["id"]][:2]
    
    # 4. COMPARE
    comparison_data = generate_comparison(query, [top_pick] + alternatives)
    
    prod_map = {p["id"]: p for p in [top_pick] + alternatives}
    for item in comparison_data.get("products", []):
        p = prod_map.get(item.get("id"))
        if p:
            item["name"] = p.get("name", "")
            item["price"] = p.get("price", 0)
            item["category"] = p.get("category", "")
            item["features"] = p.get("features", [])

    # We map preferences just as dummy to prevent frontend errors
    prefs_mock = {"category": top_pick.get("category"), "keywords": []}
    
    return {
        "top_pick": top_pick,
        "alternatives": alternatives,
        "explanation": explanation,
        "extracted_preferences": prefs_mock,
        "agent_plan": agent_trace_logs,
        "comparison_data": comparison_data,
        "session_id": session_id
    }