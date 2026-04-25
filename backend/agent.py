"""ProductMind recommendation agent with a single LLM call per query."""

import copy
import hashlib
import json
import logging
import os
import re
import threading
import time
from typing import Any, Dict, List, Optional

try:
    from groq import Groq
except ModuleNotFoundError:
    Groq = None

from embeddings import get_search_engine

logger = logging.getLogger(__name__)

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL = "llama3-8b-8192"
REQUEST_THROTTLE_SECONDS = 1.5
CACHE_TTL_SECONDS = 600
SEARCH_TOP_K = 20
LLM_CANDIDATE_LIMIT = 5
ALTERNATIVE_LIMIT = 3

PRICE_PATTERNS = [
    (r"(\d+(?:\.\d+)?)\s*k\b", lambda match: float(match.group(1)) * 1000),
    (r"(\d+(?:\.\d+)?)\s*thousand\b", lambda match: float(match.group(1)) * 1000),
    (r"(\d+(?:\.\d+)?)\s*lakh\b", lambda match: float(match.group(1)) * 100000),
    (r"[\₹\$]?\s*(\d[\d,]*(?:\.\d+)?)", lambda match: float(match.group(1).replace(",", ""))),
]

SORT_SIGNALS = {
    "asc": ["cheap", "budget", "affordable", "low price", "low-cost", "inexpensive", "economical", "value"],
    "desc": ["premium", "expensive", "best", "high end", "luxury", "top", "flagship"],
}

CATEGORY_KEYWORDS = {
    "audio": ["headphone", "earphone", "earbuds", "speaker", "audio", "sound", "noise cancell", "headset"],
    "electronics": ["laptop", "camera", "monitor", "tablet", "phone", "smartphone", "mobile"],
    "peripherals": ["keyboard", "mouse", "webcam", "controller", "gamepad", "mousepad"],
    "furniture": ["desk", "chair", "table", "sofa", "shelf", "cabinet", "stool"],
    "appliances": ["fan", "blender", "kettle", "heater", "purifier", "cooler", "mixer", "iron"],
    "fitness": ["fitness", "band", "resistance", "yoga", "dumbbell", "treadmill", "cycle", "mat"],
    "decor": ["decor", "candle", "lamp", "clock", "aesthetic", "frame", "vase", "plant"],
    "stationery": ["notebook", "pen", "journal", "planner", "diary", "marker", "organizer"],
    "bags": ["bag", "backpack", "tote", "sling", "luggage", "wallet"],
    "clothing": ["shirt", "tshirt", "jeans", "jacket", "hoodie", "dress", "kurta"],
    "makeup": ["lipstick", "foundation", "mascara", "blush", "eyeliner", "moisturizer"],
    "accessories": ["watch", "sunglasses", "ring", "bracelet", "necklace", "belt", "charger", "tripod"],
}

PRODUCT_TYPE_KEYWORDS = {
    "mobile": ["mobile", "mobiles", "phone", "phones", "smartphone", "smartphones", "iphone"],
    "laptop": ["laptop", "laptops", "notebook", "notebooks", "ultrabook", "ultrabooks", "macbook"],
    "headphones": ["headphone", "headphones", "headset", "headsets", "over-ear", "on-ear"],
    "earphones": ["earphone", "earphones", "earbud", "earbuds", "tws", "neckband", "in-ear", "inear"],
    "monitor": ["monitor", "monitors", "display"],
    "tablet": ["tablet", "tablets", "ipad"],
    "keyboard": ["keyboard", "keyboards"],
    "mouse": ["mouse", "mice"],
    "speaker": ["speaker", "speakers", "soundbar"],
}

PRODUCT_TYPE_CATEGORY_MAP = {
    "mobile": "electronics",
    "laptop": "electronics",
    "monitor": "electronics",
    "tablet": "electronics",
    "headphones": "audio",
    "earphones": "audio",
    "speaker": "audio",
    "keyboard": "peripherals",
    "mouse": "peripherals",
}

UNIFIED_SYSTEM_PROMPT = """
You are ProductMind, an expert AI shopping assistant. You are intelligent, helpful, and conversational.

You will receive:
- A user query (may be a question, product request, or both)
- A list of pre-filtered candidate products from the database
- Optional user history (past searches and viewed products)

YOUR TASKS (complete ALL in one response):

1. ANSWER: Generate a natural, specific, and helpful answer to the user's query. Directly answer the user's question first in 2-3 sentences. Do NOT use generic phrases like 'Based on your query' or 'I recommend'. Tailor the response to the exact user question and intent.

2. SELECT: Choose the single best product from the candidates that matches the user's intent. If the user asked for the "cheapest" or "lowest price", you MUST select the product with the absolute lowest price.

3. EXPLAIN: Write a detailed explanation of at least 120 words covering why this specific product was chosen based on the user's requirements.

4. FOLLOWUPS: Generate exactly 3 follow-up questions to refine the user's intent. Make them specific and useful.

5. COMPARISON: Write 2-3 sentences comparing the top 2-3 candidates objectively.

6. INSIGHT: If user history is provided, generate a 1-2 sentence personalized observation about their preferences.
If no history exists, return an empty string.

STRICT RULES:
- Return ONLY valid JSON. No markdown. No explanation outside JSON.
- If the user asks for the 'cheapest' option, prioritize price above all else in selection.
- Never recommend products outside the provided candidates list.
- Never hallucinate product names, prices, or features.
- Avoid repetitive or template-like language.

REQUIRED OUTPUT FORMAT:
{
  "answer": "<specific and tailored response to the user's intent>",
  "top_pick_id": "<exact product id from candidates>",
  "explanation": "<minimum 120 word detailed explanation>",
  "followups": ["<question 1>", "<question 2>", "<question 3>"],
  "comparison": "<2-3 sentence comparison of top candidates>",
  "insight": "<personalized insight from history, or empty string>"
}
"""

_STATE_LOCK = threading.Lock()
_LAST_REQUEST_TS: dict[str, float] = {}
_RESPONSE_CACHE: dict[str, dict[str, Any]] = {}


def _clone(value: Any) -> Any:
    """Return a deep copy so cached objects are never mutated in place."""
    return copy.deepcopy(value)


def _normalize_text(text: str) -> str:
    """Normalize whitespace and lowercase text for stable matching."""
    return re.sub(r"\s+", " ", text.strip().lower())


def _tokenize_text(text: str) -> set[str]:
    """Tokenize text into a lightweight searchable set of normalized terms."""
    stop_words = {
        "a",
        "an",
        "and",
        "are",
        "for",
        "from",
        "how",
        "i",
        "in",
        "is",
        "me",
        "of",
        "on",
        "or",
        "the",
        "to",
        "with",
    }
    tokens = re.findall(r"[a-z0-9]+", text.lower())
    return {token for token in tokens if len(token) > 2 and token not in stop_words}


def _get_groq_client() -> Any:
    """Create a Groq client or raise a clear error when configuration is missing."""
    if Groq is None:
        raise RuntimeError("groq package is not installed")
    if not GROQ_API_KEY:
        raise RuntimeError("GROQ_API_KEY not set")
    return Groq(api_key=GROQ_API_KEY)


def _extract_json_from_llm(raw: str) -> dict:
    """Parse the JSON body from a model response that may include fences."""
    text = raw.strip()
    if text.startswith("```json"):
        text = text[7:]
    if text.startswith("```"):
        text = text[3:]
    if text.endswith("```"):
        text = text[:-3]
    text = text.strip()

    try:
        return json.loads(text)
    except (json.JSONDecodeError, TypeError):
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if not match:
            return {}
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            return {}


def _extract_price_value(text: str) -> Optional[float]:
    """Extract the first numeric price expression from text using supported slang patterns."""
    for pattern, converter in PRICE_PATTERNS:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            try:
                return converter(match)
            except ValueError:
                continue
    return None


def normalize_query(query: str) -> dict:
    """Parse a natural-language query into cleaned query text and shopping signals."""
    cleaned_query = _normalize_text(query)
    max_price = None
    price_order = None

    if re.search(r"\b(under|below|less than|within)\b", cleaned_query):
        constraint_match = re.search(
            r"(?:under|below|less than|within)\s+([\₹\$]?\s*\d[\d,]*(?:\.\d+)?(?:\s*(?:k|thousand|lakh))?)",
            cleaned_query,
        )
        if constraint_match:
            max_price = _extract_price_value(constraint_match.group(1))

    if max_price is None and re.search(r"\b(around|approximately|approx|about|near)\b", cleaned_query):
        approximate_match = re.search(
            r"(?:around|approximately|approx|about|near)\s+([\₹\$]?\s*\d[\d,]*(?:\.\d+)?(?:\s*(?:k|thousand|lakh))?)",
            cleaned_query,
        )
        if approximate_match:
            raw_price = _extract_price_value(approximate_match.group(1))
            if raw_price is not None:
                max_price = raw_price * 1.15

    for order, keywords in SORT_SIGNALS.items():
        if any(keyword in cleaned_query for keyword in keywords):
            price_order = order
            break

    detected_category = detect_category(cleaned_query)
    requested_product_type = detect_product_type_from_query(cleaned_query)
    if requested_product_type and PRODUCT_TYPE_CATEGORY_MAP.get(requested_product_type):
        detected_category = PRODUCT_TYPE_CATEGORY_MAP[requested_product_type]

    intent = "informational" if re.search(r"\b(what|which|how|why|can|should|difference|compare)\b", cleaned_query) else "product"
    keywords = sorted(_tokenize_text(cleaned_query))[:8]

    return {
        "cleaned_query": cleaned_query,
        "max_price": max_price,
        "price_order": price_order,
        "detected_category": detected_category,
        "requested_product_type": requested_product_type,
        "keywords": keywords,
        "intent": intent,
    }


def detect_category(query: str) -> Optional[str]:
    """Detect the broad category that best matches a user query."""
    normalized = _normalize_text(query)
    for category, keywords in CATEGORY_KEYWORDS.items():
        if any(keyword in normalized for keyword in keywords):
            return category
    return None


def detect_product_type_from_query(query: str) -> Optional[str]:
    """Detect the exact requested product type when the query names one directly."""
    tokens = _tokenize_text(query)
    for product_type, keywords in PRODUCT_TYPE_KEYWORDS.items():
        if any(_normalize_text(keyword).replace("-", "") in "".join(sorted(tokens)) for keyword in keywords if "-" in keyword):
            return product_type
        if any(token in tokens for token in _tokenize_text(" ".join(keywords))):
            return product_type
    if "headphones" in query:
        return "headphones"
    if "earphones" in query or "earbuds" in query:
        return "earphones"
    return None


def _product_text(product: Dict[str, Any]) -> str:
    """Flatten product text fields for deterministic matching and scoring."""
    return " ".join(
        [
            str(product.get("name", "")),
            str(product.get("category", "")),
            str(product.get("brand", "")),
            " ".join(product.get("features", [])),
            " ".join(product.get("tags", [])),
        ]
    ).lower()


def detect_product_type(product: Dict[str, Any]) -> Optional[str]:
    """Infer a product type from its structured metadata."""
    product_text = _product_text(product)
    tokens = _tokenize_text(product_text)

    if {"earbud", "earbuds", "earphone", "earphones", "tws", "neckband"} & tokens:
        return "earphones"
    if {"headphone", "headphones", "headset", "headsets"} & tokens:
        return "headphones"
    if {"mobile", "mobiles", "smartphone", "smartphones", "iphone", "phone", "phones"} & tokens:
        return "mobile"
    if {"laptop", "laptops", "notebook", "notebooks", "ultrabook", "ultrabooks", "macbook"} & tokens:
        return "laptop"
    if {"monitor", "monitors", "display"} & tokens:
        return "monitor"
    if {"tablet", "tablets", "ipad"} & tokens:
        return "tablet"
    if {"keyboard", "keyboards"} & tokens:
        return "keyboard"
    if {"mouse", "mice"} & tokens:
        return "mouse"
    if {"speaker", "speakers", "soundbar"} & tokens:
        return "speaker"
    return None


def tool_semantic_search(query: str, top_k: int = SEARCH_TOP_K) -> List[Dict]:
    """Search the product catalog with embeddings and return raw product dictionaries."""
    engine = get_search_engine()
    results = engine.search(query, top_k=top_k)
    return [result["product"] for result in results]


def tool_budget_filter(candidates: List[Dict], max_price: float) -> List[Dict]:
    """Keep only products whose price does not exceed the requested budget."""
    return [product for product in candidates if float(product.get("price", 0) or 0) <= max_price]


def tool_category_filter(candidates: List[Dict], category: str) -> List[Dict]:
    """Keep only products in the requested category."""
    category_lower = category.lower()
    return [product for product in candidates if product.get("category", "").lower() == category_lower]


def tool_sort_price(candidates: List[Dict], order: str) -> List[Dict]:
    """Sort candidates by price according to the user’s budget intent."""
    reverse = order == "desc"
    return sorted(candidates, key=lambda product: float(product.get("price", 0) or 0), reverse=reverse)


def tool_product_type_filter(candidates: List[Dict], product_type: str) -> List[Dict]:
    """Keep only products that match the exact requested product type."""
    return [product for product in candidates if detect_product_type(product) == product_type]


def _score_candidate(product: Dict[str, Any], query_signals: Dict[str, Any]) -> int:
    """Score a candidate product deterministically before it reaches the LLM."""
    product_text = _product_text(product)
    product_terms = _tokenize_text(product_text)
    query_terms = set(query_signals.get("keywords", []))
    score = 40 + min(len(query_terms & product_terms) * 8, 28)

    price = float(product.get("price", 0) or 0)
    max_price = query_signals.get("max_price")
    if max_price is not None:
        if price <= max_price:
            score += 12
            price_ratio = price / max_price if max_price else 1
            if query_signals.get("price_order") == "asc":
                score += int((1 - price_ratio) * 10)
            elif query_signals.get("price_order") == "desc":
                score += int(price_ratio * 10)
        else:
            score -= 20

    detected_category = query_signals.get("detected_category")
    if detected_category and product.get("category", "").lower() == detected_category:
        score += 10

    requested_product_type = query_signals.get("requested_product_type")
    if requested_product_type and detect_product_type(product) == requested_product_type:
        score += 14

    score += min(len(product.get("features", [])) * 2, 10)
    return score


def inject_product_metadata(products: List[Dict]) -> List[Dict]:
    """Ensure every product has a search link and valid image URL."""
    enriched = []
    for product in products:
        updated = _clone(product)
        name_raw = updated.get("name", "")
        name_encoded = name_raw.replace(" ", "+")
        updated["google_link"] = f"https://www.google.com/search?q={name_encoded}+buy+online"
        
        # Ensure image_url exists
        if not updated.get("image_url"):
            updated["image_url"] = f"https://source.unsplash.com/400x300/?{name_encoded}"
        
        enriched.append(updated)
    return enriched


def _build_candidate_summary(candidates: List[Dict]) -> str:
    """Serialize candidate products into a compact prompt-friendly summary."""
    lines = []
    for product in candidates:
        lines.append(
            (
                f"ID: {product['id']} | Name: {product['name']} | Price: ₹{product['price']} | "
                f"Category: {product['category']} | Brand: {product.get('brand', '')} | "
                f"Features: {', '.join(product.get('features', []))} | Tags: {', '.join(product.get('tags', []))}"
            )
        )
    return "\n".join(lines)


def _summarize_history(history: Optional[list]) -> str:
    """Reduce user history into a concise string for the unified prompt."""
    if not history:
        return "No user history provided."

    compact_lines = []
    for item in history[:5]:
        if isinstance(item, dict):
            query = item.get("query") or item.get("content") or ""
            top_pick = item.get("topPickName") or item.get("top_pick") or ""
            category = item.get("category") or ""
            compact_lines.append(f"query={query}; top_pick={top_pick}; category={category}")
        else:
            compact_lines.append(str(item))
    return "\n".join(compact_lines) or "No user history provided."


def _cache_key(query: str, history: Optional[list]) -> str:
    """Build a stable cache key from the query and recent history."""
    raw = _normalize_text(query) + json.dumps(history or [], sort_keys=True, default=str)
    return hashlib.md5(raw.encode("utf-8")).hexdigest()


def _get_cached_response(cache_key: str, session_id: Optional[str]) -> Optional[Dict[str, Any]]:
    """Return a fresh cached response when it is still within the TTL window."""
    with _STATE_LOCK:
        entry = _RESPONSE_CACHE.get(cache_key)
        if not entry:
            return None
        if time.time() - entry["timestamp"] > CACHE_TTL_SECONDS:
            _RESPONSE_CACHE.pop(cache_key, None)
            return None
        response = _clone(entry["response"])
        response["session_id"] = session_id
        return response


def _set_cached_response(cache_key: str, response: Dict[str, Any]) -> None:
    """Store a successful LLM-backed response in the in-memory cache."""
    cached = _clone(response)
    cached["session_id"] = None
    with _STATE_LOCK:
        _RESPONSE_CACHE[cache_key] = {"timestamp": time.time(), "response": cached}


def _enforce_request_throttle(user_id: Optional[str]) -> None:
    """Throttle rapid repeat requests from the same user in memory."""
    if not user_id:
        return

    now = time.time()
    with _STATE_LOCK:
        last_seen = _LAST_REQUEST_TS.get(user_id, 0.0)
        wait_time = max(0.0, REQUEST_THROTTLE_SECONDS - (now - last_seen))
        _LAST_REQUEST_TS[user_id] = now + wait_time

    if wait_time > 0:
        logger.info("Throttling user '%s' for %.2fs", user_id, wait_time)
        time.sleep(wait_time)


def _validate_llm_output(data: dict, candidates: List[Dict]) -> dict:
    """Validate and normalize the unified LLM output against the candidate set."""
    candidate_ids = {product["id"] for product in candidates}
    top_pick_id = data.get("top_pick_id")
    if top_pick_id not in candidate_ids:
        top_pick_id = candidates[0]["id"]

    followups = data.get("followups")
    if not isinstance(followups, list):
        followups = []
    followups = [str(question).strip() for question in followups if str(question).strip()][:3]
    while len(followups) < 3:
        followups.append("What matters most to you: budget, performance, or design?")

    explanation = str(data.get("explanation") or "").strip()
    if not explanation:
        explanation = (
            f"{candidates[0]['name']} is the closest match among the shortlisted products because it balances "
            f"price, features, and category fit for your request."
        )

    return {
        "answer": str(data.get("answer") or "").strip(),
        "top_pick_id": top_pick_id,
        "explanation": explanation,
        "followups": followups,
        "comparison": str(data.get("comparison") or "").strip(),
        "insight": str(data.get("insight") or "").strip(),
    }


def single_llm_call(query: str, candidates: List[Dict], history: Optional[list] = None) -> dict:
    """Run the single unified ProductMind model call for answer, pick, explanation, and follow-ups."""
    if not candidates:
        raise ValueError("single_llm_call requires at least one candidate.")

    client = _get_groq_client()
    user_prompt = (
        f"USER QUERY:\n{query}\n\n"
        f"CANDIDATES:\n{_build_candidate_summary(candidates)}\n\n"
        f"USER HISTORY:\n{_summarize_history(history)}"
    )
    response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {"role": "system", "content": UNIFIED_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.2,
        max_tokens=1100,
    )
    raw = response.choices[0].message.content.strip()
    parsed = _extract_json_from_llm(raw)
    return _validate_llm_output(parsed, candidates)


def _build_followups(query_signals: Dict[str, Any], top_pick: Dict[str, Any]) -> List[str]:
    """Generate exactly three deterministic refinement questions for fallback mode."""
    product_type = query_signals.get("requested_product_type") or top_pick.get("category", "product")
    return [
        f"Do you want a more budget-focused or feature-focused {product_type}?",
        f"Will you use this {product_type} mostly at home, work, travel, or commuting?",
        "Which matters more to you right now: battery life, performance, or comfort/design?",
    ]


def _build_comparison_summary(products: List[Dict]) -> str:
    """Generate a short deterministic comparison summary for the shortlisted products."""
    if not products:
        return ""
    if len(products) == 1:
        return f"{products[0]['name']} is the only strong shortlist match after applying your filters."

    lead = products[0]
    runner_up = products[1]
    parts = [
        f"{lead['name']} leads the shortlist with the strongest overall balance of features and price.",
        f"{runner_up['name']} is the closest alternative if you prefer its particular strengths.",
    ]
    if len(products) > 2:
        parts.append(f"{products[2]['name']} rounds out the shortlist as another credible option with a different tradeoff profile.")
    return " ".join(parts)


def _build_insight(history: Optional[list], top_pick: Dict[str, Any]) -> str:
    """Generate a lightweight personalized insight when fallback mode is used."""
    if not history:
        return ""

    recent_categories = []
    for item in history[:5]:
        if isinstance(item, dict) and item.get("category"):
            recent_categories.append(str(item["category"]).lower())
    if top_pick.get("category", "").lower() in recent_categories:
        return f"You often explore {top_pick['category']} products, so this recommendation stays close to your recent pattern."
    return "Your recent searches suggest you compare products carefully before deciding, so this shortlist keeps a few distinct tradeoffs visible."


def fallback_response(query: str, candidates: List[Dict], history: Optional[list] = None) -> dict:
    """Build a valid single-call-style response without using the LLM at all."""
    if not candidates:
        return {
            "answer": "I couldn't find any products in our catalog that match your specific request. Try broadening your keywords or adjusting your price parameters.",
            "top_pick_id": "no-match",
            "explanation": (
                "I could not find a strong product match from the current catalog for that request. "
                "Try broadening the category, raising the budget slightly, or describing the most important feature."
            ),
            "followups": [
                "Would you like me to widen the budget slightly?",
                "Is there a specific brand or feature you care about most?",
                "Do you want a budget, balanced, or premium recommendation?",
            ],
            "comparison": "",
            "insight": "",
        }

    top_pick = candidates[0]
    alternatives = candidates[1:4]
    feature_text = ", ".join(top_pick.get("features", [])[:3]) or "balanced everyday features"
    comparison = _build_comparison_summary(candidates[:4])
    explanation = (
        f"{top_pick['name']} is the strongest fallback match for '{query}' because it stays aligned with the filtered shortlist "
        f"while offering {feature_text}. At ₹{top_pick['price']}, it remains competitive for the {top_pick['category']} category, "
        f"and it covers the core use cases implied by your request without introducing unrelated tradeoffs. "
        f"If you want the safest all-round option from the current candidate list, this is the one I would keep at the top. "
        f"{comparison}"
    )

    # Heuristic Answer
    category = top_pick.get("category", "products")
    if "cheap" in query or "low" in query or "budget" in query:
        answer = f"The {top_pick['name']} is the most budget-friendly {category} I found in our current catalog, priced at ₹{top_pick['price']}."
    else:
        answer = f"I've selected the {top_pick['name']} because it offers the best combination of features and value for your {category} search."

    return {
        "answer": answer,
        "top_pick_id": top_pick["id"],
        "explanation": explanation,
        "followups": _build_followups(normalize_query(query), top_pick),
        "comparison": comparison,
        "insight": _build_insight(history, top_pick),
    }


def _build_comparison_data(query: str, products: List[Dict], comparison_summary: str) -> Dict[str, Any]:
    """Create deterministic comparison metadata used by the comparison page."""
    query_signals = normalize_query(query)
    comparison_rows = []
    for product in products:
        score = max(55, min(98, _score_candidate(product, query_signals)))
        comparison_rows.append(
            {
                "id": product["id"],
                "score": score,
                "verdict": (
                    f"{product['name']} is a strong {product['category']} option at ₹{product['price']} "
                    "based on the current shortlist filters."
                ),
                "pros": product.get("features", [])[:3] or ["Solid overall fit for the request"],
                "cons": ["May trade off against other shortlisted options on budget, feature depth, or specialization."],
            }
        )
    return {"summary": comparison_summary, "products": comparison_rows}


def _build_agent_plan(query_signals: Dict[str, Any], candidate_count: int) -> List[str]:
    """Build a compact internal execution trace for debugging and compatibility."""
    plan = [f"Semantic search shortlisted {candidate_count} candidates."]
    if query_signals.get("max_price") is not None:
        plan.append(f"Budget filter applied at ₹{int(query_signals['max_price'])}.")
    if query_signals.get("detected_category"):
        plan.append(f"Category filter locked to {query_signals['detected_category']}.")
    if query_signals.get("requested_product_type"):
        plan.append(f"Exact product type filter locked to {query_signals['requested_product_type']}.")
    if query_signals.get("price_order"):
        plan.append(f"Price order preference applied: {query_signals['price_order']}.")
    return plan


def run_recommendation_agent(
    query: str,
    session_id: Optional[str] = None,
    history: Optional[list] = None,
    current_user: Optional[str] = None,
) -> Dict[str, Any]:
    """Run ProductMind with Python pre-processing, one unified LLM call, and Python post-processing."""
    logger.info("Running single-call recommendation agent for session=%s", session_id)
    _enforce_request_throttle(current_user)

    cache_key = _cache_key(query, history)
    cached_response = _get_cached_response(cache_key, session_id)
    if cached_response is not None:
        logger.info("Recommendation cache hit for query '%s'", query)
        return cached_response

    query_signals = normalize_query(query)
    candidates = tool_semantic_search(query_signals["cleaned_query"], top_k=SEARCH_TOP_K)

    if query_signals.get("max_price") is not None:
        candidates = tool_budget_filter(candidates, float(query_signals["max_price"]))
    if query_signals.get("detected_category"):
        candidates = tool_category_filter(candidates, query_signals["detected_category"])
    if query_signals.get("requested_product_type"):
        candidates = tool_product_type_filter(candidates, query_signals["requested_product_type"])
    if query_signals.get("price_order"):
        candidates = tool_sort_price(candidates, query_signals["price_order"])

    if not candidates:
        candidates = tool_semantic_search(query_signals["cleaned_query"], top_k=SEARCH_TOP_K)
        if query_signals.get("detected_category"):
            candidates = tool_category_filter(candidates, query_signals["detected_category"])
        if query_signals.get("requested_product_type"):
            candidates = tool_product_type_filter(candidates, query_signals["requested_product_type"])

    if query_signals.get("price_order") == "asc":
        # Strict enforcement for "cheapest" queries
        ranked_candidates = sorted(
            candidates,
            key=lambda product: float(product.get("price", 0) or 0),
            reverse=False,
        )
    else:
        ranked_candidates = sorted(
            candidates,
            key=lambda product: _score_candidate(product, query_signals),
            reverse=True,
        )
    
    shortlisted = inject_product_metadata(ranked_candidates[:LLM_CANDIDATE_LIMIT])

    agent_plan = _build_agent_plan(query_signals, len(shortlisted))

    if not shortlisted:
        return {
            "top_pick": {
                "id": "no-match",
                "name": "No matching product found",
                "price": 0,
                "brand": "N/A",
                "category": "N/A",
                "features": ["Try modifying your search"],
                "tags": [],
                "google_link": "",
            },
            "alternatives": [],
            "explanation": (
                "I could not find a matching product from the current catalog for that query. "
                "Try broadening the budget, using a simpler category, or naming one feature you care about most."
            ),
            "extracted_preferences": {
                "category": query_signals.get("detected_category"),
                "max_price": query_signals.get("max_price"),
                "keywords": query_signals.get("keywords", []),
                "intent": query_signals.get("intent"),
            },
            "agent_plan": [],
            "comparison_data": {"summary": "", "products": []},
            "session_id": session_id,
            "answer": "I'm sorry, I couldn't find any products in our database that match your search. You might want to try different keywords or a slightly different budget.",
            "followups": [
                "Would you like to widen the budget slightly?",
                "Is there a brand or feature you care about most?",
                "Do you want a budget, balanced, or premium recommendation?",
            ],
            "comparison": "",
            "insight": "",
        }

    llm_used = False
    try:
        llm_output = single_llm_call(query, shortlisted, history)
        llm_used = True
    except Exception as exc:
        logger.error("Single LLM call failed: %s", exc)
        llm_output = fallback_response(query, shortlisted, history)

    top_pick = next((product for product in shortlisted if product["id"] == llm_output["top_pick_id"]), shortlisted[0])
    alternatives = [product for product in shortlisted if product["id"] != top_pick["id"]][:ALTERNATIVE_LIMIT]
    comparison_products = [top_pick] + alternatives
    comparison_data = _build_comparison_data(query, comparison_products, llm_output["comparison"])

    response = {
        "top_pick": top_pick,
        "alternatives": alternatives,
        "explanation": llm_output["explanation"],
        "extracted_preferences": {
            "category": query_signals.get("detected_category"),
            "max_price": query_signals.get("max_price"),
            "keywords": query_signals.get("keywords", []),
            "intent": query_signals.get("intent"),
        },
        "agent_plan": [],
        "comparison_data": comparison_data,
        "session_id": session_id,
        "answer": llm_output["answer"],
        "followups": llm_output["followups"],
        "comparison": llm_output["comparison"],
        "insight": llm_output["insight"],
    }

    if llm_used:
        _set_cached_response(cache_key, response)
    return _clone(response)
