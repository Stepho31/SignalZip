import sqlite3
from datetime import datetime, timezone
from typing import Optional, Dict, Any

DB_NAME = "signalzip.db"


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def init_db():
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS users (
        email TEXT PRIMARY KEY,
        stripe_customer_id TEXT,
        stripe_subscription_id TEXT,
        subscription_status TEXT,
        active INTEGER DEFAULT 0,
        last_status_check_utc TEXT
    )
    """)

    cursor.execute("CREATE INDEX IF NOT EXISTS idx_users_customer ON users(stripe_customer_id)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_users_subscription ON users(stripe_subscription_id)")

    conn.commit()
    conn.close()


def get_user(email: str) -> Optional[Dict[str, Any]]:
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()

    cursor.execute("""
        SELECT email, stripe_customer_id, stripe_subscription_id, subscription_status, active, last_status_check_utc
        FROM users
        WHERE email = ?
    """, (email,))
    row = cursor.fetchone()
    conn.close()

    if not row:
        return None

    return {
        "email": row[0],
        "stripe_customer_id": row[1],
        "stripe_subscription_id": row[2],
        "subscription_status": row[3],
        "active": int(row[4] or 0),
        "last_status_check_utc": row[5],
    }


def get_user_by_customer(customer_id: str) -> Optional[Dict[str, Any]]:
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()

    cursor.execute("""
        SELECT email, stripe_customer_id, stripe_subscription_id, subscription_status, active, last_status_check_utc
        FROM users
        WHERE stripe_customer_id = ?
    """, (customer_id,))
    row = cursor.fetchone()
    conn.close()

    if not row:
        return None

    return {
        "email": row[0],
        "stripe_customer_id": row[1],
        "stripe_subscription_id": row[2],
        "subscription_status": row[3],
        "active": int(row[4] or 0),
        "last_status_check_utc": row[5],
    }


def upsert_user(
    email: str,
    stripe_customer_id: Optional[str] = None,
    stripe_subscription_id: Optional[str] = None,
    subscription_status: Optional[str] = None,
    active: int = 0,
    last_status_check_utc: Optional[str] = None,
):
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()

    cursor.execute("""
    INSERT INTO users (
        email, stripe_customer_id, stripe_subscription_id, subscription_status, active, last_status_check_utc
    )
    VALUES (?, ?, ?, ?, ?, ?)
    ON CONFLICT(email)
    DO UPDATE SET
        stripe_customer_id = COALESCE(excluded.stripe_customer_id, users.stripe_customer_id),
        stripe_subscription_id = COALESCE(excluded.stripe_subscription_id, users.stripe_subscription_id),
        subscription_status = COALESCE(excluded.subscription_status, users.subscription_status),
        active = excluded.active,
        last_status_check_utc = COALESCE(excluded.last_status_check_utc, users.last_status_check_utc)
    """, (
        email,
        stripe_customer_id,
        stripe_subscription_id,
        subscription_status,
        int(active),
        last_status_check_utc or _utc_now_iso()
    ))

    conn.commit()
    conn.close()


def set_status_check_time(email: str):
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute("""
        UPDATE users
        SET last_status_check_utc = ?
        WHERE email = ?
    """, (_utc_now_iso(), email))
    conn.commit()
    conn.close()
