import json
import re
import time
from pathlib import Path

try:
    from sentence_transformers import SentenceTransformer
except ModuleNotFoundError:
    SentenceTransformer = None

try:
    from sklearn.metrics.pairwise import cosine_similarity
except ModuleNotFoundError:
    cosine_similarity = None

try:
    import numpy as np
except ModuleNotFoundError:
    np = None

try:
    from database import SessionLocal, Product
except ModuleNotFoundError:
    SessionLocal = None
    Product = None

# Initialize the embedding model globally so it stays in memory
# 'all-MiniLM-L6-v2' is fast, lightweight, and excellent for basic semantic search
model_name = "all-MiniLM-L6-v2"
model = None
if SentenceTransformer is not None:
    try:
        print(f"Loading embedding model {model_name}...")
        model = SentenceTransformer(model_name)
        print("Embedding model loaded.")
    except Exception as exc:
        print(f"Embedding model unavailable ({exc}). Falling back to lexical product search.")
        model = None
else:
    print("sentence-transformers not installed. Falling back to lexical product search.")

class SemanticSearchIndex:
    def __init__(self):
        self.embeddings = []
        self.product_ids = []
        self.products = []
        
    def build_index(self):
        """Fetches all products from SQLite and builds a dense vector index."""
        self.product_ids = []
        self.embeddings = []

        try:
            products_file = Path(__file__).parent / "products.json"
            if products_file.exists():
                self.products = json.loads(products_file.read_text())
            elif SessionLocal is not None and Product is not None:
                db = SessionLocal()
                try:
                    items = db.query(Product).all()
                    self.products = [p.to_dict() for p in items]
                finally:
                    db.close()
            else:
                self.products = []

            if not self.products:
                print("No products available to index.")
                return

            texts_to_embed = []
            for p in self.products:
                tags_str = " ".join(p.get("tags", []))
                feats_str = " ".join(p.get("features", []))
                combo = f"{p['name']} {p['category']} {tags_str} {feats_str}".lower()
                texts_to_embed.append(combo)
                self.product_ids.append(p["id"])

            if model is None or np is None or cosine_similarity is None:
                self.embeddings = []
                return

            print(f"Encoding {len(texts_to_embed)} objects. This might take a second...")
            t1 = time.time()
            vectors = model.encode(texts_to_embed)
            self.embeddings = np.array(vectors)
            print(f"Encoding complete in {time.time() - t1:.2f}s")

        except Exception as e:
            print(f"Index build failed: {e}")

    def search(self, query: str, top_k: int = 5) -> list[dict]:
        """Returns the top_k semantically similar products for a query."""
        if len(self.products) == 0:
            print("Index is empty. Building index...")
            self.build_index()
            if len(self.products) == 0:
                return []

        if model is None or np is None or cosine_similarity is None or len(self.embeddings) == 0:
            return self._lexical_search(query, top_k)

        # Encode user query
        query_vec = model.encode([query])

        # Compute cosine similarities
        similarities = cosine_similarity(query_vec, self.embeddings)[0]
        
        # Get top indices
        top_indices = similarities.argsort()[-top_k:][::-1]
        
        results = []
        for idx in top_indices:
            score = similarities[idx]
            prod = self.products[idx]
            results.append({
                "product": prod,
                "similarity": float(score)
            })
            
        return results

    def _lexical_search(self, query: str, top_k: int = 5) -> list[dict]:
        """Simple keyword-overlap fallback when ML dependencies are unavailable."""
        query_terms = set(re.findall(r"[a-z0-9]+", query.lower()))
        scored = []

        for product in self.products:
            product_text = " ".join(
                [
                    product.get("name", ""),
                    product.get("category", ""),
                    product.get("brand", ""),
                    " ".join(product.get("features", [])),
                    " ".join(product.get("tags", [])),
                ]
            ).lower()
            product_terms = set(re.findall(r"[a-z0-9]+", product_text))
            overlap = len(query_terms & product_terms)
            score = overlap / max(len(query_terms), 1)
            scored.append({"product": product, "similarity": float(score)})

        scored.sort(key=lambda item: item["similarity"], reverse=True)
        return scored[:top_k]

# Singleton instance for the app to reuse
search_engine = SemanticSearchIndex()

def get_search_engine():
    return search_engine
