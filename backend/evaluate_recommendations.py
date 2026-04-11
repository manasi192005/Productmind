"""Evaluate ProductMind recommendation quality on a small benchmark set."""

import json
from pathlib import Path
from typing import Any, Dict, List

from agent import GROQ_API_KEY, run_recommendation_agent

BASE_DIR = Path(__file__).parent
EVAL_FILE = BASE_DIR / "eval_queries.json"


def load_eval_queries() -> List[Dict[str, Any]]:
    """Load the benchmark queries used for local recommendation evaluation."""
    return json.loads(EVAL_FILE.read_text())


def evaluate_queries(eval_queries: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Run the benchmark and compute top-1, top-3, and category precision metrics."""
    results = []
    top1_hits = 0
    top3_hits = 0
    returned_products = 0
    category_matches = 0

    for case in eval_queries:
        response = run_recommendation_agent(case["query"], current_user=None, history=None)
        top_pick = response["top_pick"]
        alternatives = response.get("alternatives", [])
        ranked = [top_pick] + alternatives
        returned_ids = [product["id"] for product in ranked]

        acceptable = set(case["acceptable_top_ids"])
        top1_hit = top_pick["id"] in acceptable
        top3_hit = any(product_id in acceptable for product_id in returned_ids[:3])

        top1_hits += int(top1_hit)
        top3_hits += int(top3_hit)

        for product in ranked:
            returned_products += 1
            if product.get("category") == case["expected_category"]:
                category_matches += 1

        results.append(
            {
                "query": case["query"],
                "expected_category": case["expected_category"],
                "acceptable_top_ids": case["acceptable_top_ids"],
                "returned_ids": returned_ids,
                "top_pick_id": top_pick["id"],
                "top_pick_name": top_pick["name"],
                "top1_hit": top1_hit,
                "top3_hit": top3_hit,
            }
        )

    total = len(eval_queries)
    return {
        "mode": "llm" if GROQ_API_KEY else "fallback",
        "total_queries": total,
        "top1_accuracy": round(top1_hits / total, 4) if total else 0.0,
        "top3_accuracy": round(top3_hits / total, 4) if total else 0.0,
        "category_precision": round(category_matches / returned_products, 4) if returned_products else 0.0,
        "results": results,
    }


def main() -> None:
    """Run the evaluation and print a readable report plus JSON summary."""
    eval_queries = load_eval_queries()
    summary = evaluate_queries(eval_queries)

    print("ProductMind Evaluation")
    print(f"Mode: {summary['mode']}")
    print(f"Queries: {summary['total_queries']}")
    print(f"Top-1 Accuracy: {summary['top1_accuracy']:.2%}")
    print(f"Top-3 Accuracy: {summary['top3_accuracy']:.2%}")
    print(f"Category Precision: {summary['category_precision']:.2%}")
    print("")

    for result in summary["results"]:
        print(
            f"- {result['query']} -> top={result['top_pick_name']} ({result['top_pick_id']}) | "
            f"top1={result['top1_hit']} | top3={result['top3_hit']}"
        )

    print("")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
