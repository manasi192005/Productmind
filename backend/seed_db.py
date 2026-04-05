import json
import os
from pathlib import Path
from database import SessionLocal, init_db, Product, User

# Paths
BASE_DIR = Path(__file__).parent
PRODUCTS_FILE = BASE_DIR / "products.json"
USERS_FILE = BASE_DIR / "users.json"

def seed():
    print("Initializing Database...")
    init_db()

    db = SessionLocal()

    try:
        # Seed Products
        if PRODUCTS_FILE.exists():
            with open(PRODUCTS_FILE, "r", encoding="utf-8") as f:
                products_data = json.load(f)
                
            added_count = 0
            for p_data in products_data:
                # Check if exists
                exists = db.query(Product).filter(Product.id == p_data.get("id")).first()
                if not exists:
                    prod = Product(
                        id=p_data.get("id"),
                        name=p_data.get("name"),
                        category=p_data.get("category"),
                        price=p_data.get("price"),
                        brand=p_data.get("brand"),
                        features=p_data.get("features", []),
                        tags=p_data.get("tags", [])
                    )
                    db.add(prod)
                    added_count += 1
            print(f"Added {added_count} new products to the database.")

        # Seed Users
        if USERS_FILE.exists():
            with open(USERS_FILE, "r", encoding="utf-8") as f:
                users_data = json.load(f)
                
            added_users = 0
            for u_id, u_data in users_data.items():
                exists = db.query(User).filter(User.id == u_id).first()
                if not exists:
                    user = User(
                        id=u_id,
                        username=u_data.get("username"),
                        email=u_data.get("email"),
                        password_hash=u_data.get("password")  # Keeping it basic per existing structure
                    )
                    db.add(user)
                    added_users += 1
            print(f"Added {added_users} new users to the database.")

        db.commit()
        print("Database Seed Complete.")

    except Exception as e:
        print(f"Error seeding database: {e}")
        db.rollback()
    finally:
        db.close()

if __name__ == "__main__":
    seed()
