import sqlite3

DB_NAME = "signalzip.db"

def init_db():
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS users (
        email TEXT PRIMARY KEY,
        stripe_customer_id TEXT,
        active INTEGER DEFAULT 0
    )
    """)

    conn.commit()
    conn.close()

def get_user(email):
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute("SELECT email, stripe_customer_id, active FROM users WHERE email = ?", (email,))
    row = cursor.fetchone()
    conn.close()
    return row

def upsert_user(email, stripe_customer_id=None, active=0):
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()

    cursor.execute("""
    INSERT INTO users (email, stripe_customer_id, active)
    VALUES (?, ?, ?)
    ON CONFLICT(email)
    DO UPDATE SET
        stripe_customer_id=excluded.stripe_customer_id,
        active=excluded.active
    """, (email, stripe_customer_id, active))

    conn.commit()
    conn.close()
