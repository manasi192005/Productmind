import json
import os
import time
from pathlib import Path

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

from database import SessionLocal, Product

# Initialize the embedding model globally so it stays in memory
# 'all-MiniLM-L6-v2' is fast, lightweight, and excellent for basic semantic search
model_name = "all-MiniLM-L6-v2"
print(f"Loading embedding model {model_name}...")
model = SentenceTransformer(model_name)
print("Embedding model loaded.")

class SemanticSearchIndex:
    def __init__(self):
        self.embeddings = []
        self.product_ids = []
        self.products = []
        
    def build_index(self):
        """Fetches all products from SQLite and builds a dense vector index."""
        db = SessionLocal()
        try:
            items = db.query(Product).all()
            if not items:
                print("No products in database to index.")
                return
                
            self.products = [p.to_dict() for p in items]
            
            # Combine features for embedding
            texts_to_embed = []
            for p in self.products:
                tags_str = " ".join(p.get("tags", []))
                feats_str = " ".join(p.get("features", []))
                combo = f"{p['name']} {p['category']} {tags_str} {feats_str}".lower()
                texts_to_embed.append(combo)
                self.product_ids.append(p['id'])
                
            # Perform Encoding
            print(f"Encoding {len(texts_to_embed)} objects. This might take a second...")
            t1 = time.time()
            vectors = model.encode(texts_to_embed)
            self.embeddings = np.array(vectors)
            print(f"Encoding complete in {time.time() - t1:.2f}s")
            
        except Exception as e:
            print(f"Index build failed: {e}")
        finally:
            db.close()

    def search(self, query: str, top_k: int = 5) -> list[dict]:
        """Returns the top_k semantically similar products for a query."""
        if len(self.embeddings) == 0:
            print("Index is empty. Building index...")
            self.build_index()
            if len(self.embeddings) == 0:
                return []

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

# Singleton instance for the app to reuse
search_engine = SemanticSearchIndex()

def get_search_engine():
    return search_engine
