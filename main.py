import os
import csv
import io
import json
from typing import Any, Dict, List, Optional
from urllib.parse import quote_plus
from datetime import datetime, timezone, timedelta

import requests
from dotenv import load_dotenv
from fastapi import FastAPI, Form, Request, HTTPException
from fastapi.responses import HTMLResponse, StreamingResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from itsdangerous import URLSafeTimedSerializer, BadSignature, SignatureExpired

import stripe

from openai import OpenAI
from database import init_db, get_user, upsert_user, set_status_check_time, migrate_users_table

load_dotenv()
init_db()
migrate_users_table()

# ---------------------------
# ENV / CONFIG
# ---------------------------
APP_NAME = "SignalZip — Payroll Prospecting Intelligence"

SECRET_KEY = os.getenv("SECRET_KEY", "supersecret").strip()
serializer = URLSafeTimedSerializer(SECRET_KEY)

BASE_URL = os.getenv("BASE_URL", "http://localhost:8000").strip().rstrip("/")

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "").strip()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()

# Stripe
STRIPE_SECRET_KEY = os.getenv("STRIPE_SECRET_KEY", "").strip()
STRIPE_WEBHOOK_SECRET = os.getenv("STRIPE_WEBHOOK_SECRET", "").strip()
STRIPE_PRICE_ID = os.getenv("STRIPE_PRICE_ID", "").strip()  # Price (recurring) id like price_123

if not STRIPE_SECRET_KEY:
    # Allowed locally for free-only mode, but Stripe flows won't work without it.
    pass

stripe.api_key = STRIPE_SECRET_KEY

# Optional: force premium for debugging
FORCE_PREMIUM = os.getenv("FORCE_PREMIUM", "").strip().lower() in ("1", "true", "yes")

# Subscription status cache to avoid calling Stripe on every single request
STATUS_CACHE_MINUTES = int(os.getenv("STRIPE_STATUS_CACHE_MINUTES", "10"))
LOGIN_LINK_TTL_SECONDS = int(os.getenv("LOGIN_LINK_TTL_SECONDS", "1800"))  # 30 min default

# Cookies
SESSION_COOKIE = "sz_session"      # holds signed token with email
PREMIUM_COOKIE_NAME = "signalzip_pro"  # "1" means premium UI convenience

# Freemium
FREE_LIMIT = 3

# OpenAI client (safe to be None)
oa_client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None

app = FastAPI()
templates = Jinja2Templates(directory="templates")

LAST_RESULTS: List[Dict[str, Any]] = []


# ---------------------------
# HELPERS: AUTH / SESSION
# ---------------------------
def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def make_session_token(email: str) -> str:
    return serializer.dumps({"email": email})


def read_session_email(request: Request) -> Optional[str]:
    token = request.cookies.get(SESSION_COOKIE)
    if not token:
        return None
    try:
        data = serializer.loads(token, max_age=60 * 60 * 24 * 30)  # 30 days
        return (data or {}).get("email")
    except Exception:
        return None


def require_email(request: Request) -> str:
    email = read_session_email(request)
    if not email:
        raise HTTPException(status_code=401, detail="Login required")
    return email


def user_is_active_pro(user: Optional[Dict[str, Any]]) -> bool:
    if not user:
        return False
    return int(user.get("active") or 0) == 1


def should_refresh_status(user: Dict[str, Any]) -> bool:
    last = user.get("last_status_check_utc")
    if not last:
        return True
    try:
        dt = datetime.fromisoformat(last)
    except Exception:
        return True
    return _utc_now() - dt > timedelta(minutes=STATUS_CACHE_MINUTES)


def refresh_subscription_status_from_stripe(email: str) -> Dict[str, Any]:
    """
    Pull subscription status from Stripe and update DB.
    Uses:
      - stripe_subscription_id if we have it
      - else customer subscriptions if we have customer_id
    """
    user = get_user(email)
    if not user:
        # user may not exist yet; treat as free
        return {"is_premium": False, "user": None}

    if FORCE_PREMIUM:
        upsert_user(email, active=1, subscription_status="active", last_status_check_utc=_utc_now().isoformat())
        return {"is_premium": True, "user": get_user(email)}

    # If no stripe configured, cannot refresh
    if not STRIPE_SECRET_KEY:
        return {"is_premium": user_is_active_pro(user), "user": user}

    # If we recently checked, do not refresh
    if not should_refresh_status(user):
        return {"is_premium": user_is_active_pro(user), "user": user}

    customer_id = user.get("stripe_customer_id")
    subscription_id = user.get("stripe_subscription_id")

    status = None
    active = 0

    try:
        if subscription_id:
            sub = stripe.Subscription.retrieve(subscription_id)
            status = (sub.get("status") or "").lower()
        elif customer_id:
            subs = stripe.Subscription.list(customer=customer_id, status="all", limit=5)
            # Prefer an active/trialing subscription if present
            best = None
            for s in subs.get("data", []):
                st = (s.get("status") or "").lower()
                if st in ("active", "trialing"):
                    best = s
                    break
                if not best:
                    best = s
            if best:
                subscription_id = best.get("id")
                status = (best.get("status") or "").lower()

        if status in ("active", "trialing"):
            active = 1
        else:
            active = 0

        upsert_user(
            email=email,
            stripe_customer_id=customer_id,
            stripe_subscription_id=subscription_id,
            subscription_status=status or "unknown",
            active=active,
            last_status_check_utc=_utc_now().isoformat(),
        )
        return {"is_premium": active == 1, "user": get_user(email)}

    except Exception:
        # If Stripe fails, fall back to last known state
        set_status_check_time(email)
        user = get_user(email)
        return {"is_premium": user_is_active_pro(user), "user": user}


def is_premium_request(request: Request) -> bool:
    """
    Secure check:
    - If logged in: verify subscription status (cached)
    - If not logged in: free
    """
    if FORCE_PREMIUM:
        return True

    email = read_session_email(request)
    if not email:
        return False

    refreshed = refresh_subscription_status_from_stripe(email)
    return bool(refreshed["is_premium"])


# ---------------------------
# GOOGLE PLACES SEARCH
# ---------------------------
def fetch_companies_google(zip_code: str, industry: str = "") -> List[Dict[str, Any]]:
    if not GOOGLE_API_KEY:
        raise Exception("Missing GOOGLE_API_KEY")

    url = "https://places.googleapis.com/v1/places:searchText"
    headers = {
        "Content-Type": "application/json",
        "X-Goog-Api-Key": GOOGLE_API_KEY,
        "X-Goog-FieldMask": "places.displayName,places.formattedAddress",
    }

    query_text = f"{industry} businesses near {zip_code} USA" if industry else f"businesses near {zip_code} USA"
    body = {"textQuery": query_text, "maxResultCount": 15}

    response = requests.post(url, headers=headers, json=body, timeout=12)
    response.raise_for_status()

    data = response.json()
    results: List[Dict[str, Any]] = []

    for place in data.get("places", []):
        name = (place.get("displayName") or {}).get("text")
        if not name:
            continue
        results.append(
            {"name": name, "address": place.get("formattedAddress") or "", "zip": zip_code, "industry": industry}
        )

    return results


# ---------------------------
# BUYING COMMITTEE
# ---------------------------
def build_linkedin_search(company_name: str, title: str) -> str:
    query = quote_plus(f"{title} {company_name}")
    return f"https://www.linkedin.com/search/results/people/?keywords={query}"


def build_buying_committee(company_name: str) -> List[Dict[str, str]]:
    roles = [
        ("Chief Executive Officer", "CEO"),
        ("Chief Financial Officer", "CFO"),
        ("VP Operations", "VP Operations"),
        ("Head of HR", "HR Director"),
    ]
    return [{"name": short, "title": title, "linkedin_url": build_linkedin_search(company_name, title)} for title, short in roles]


# ---------------------------
# AI INTELLIGENCE
# ---------------------------
def ai_generate_intelligence(company: Dict[str, Any]) -> Dict[str, Any]:
    fallback = {
        "priority_score": 6,
        "signal_type": "Stable",
        "reason": "Territory-fit account. Likely has payroll + HR workflows that can be simplified (admin time, compliance, onboarding).",
        "pain_points": [
            "Payroll processing + approvals take too long each cycle",
            "Compliance risk grows as headcount and locations expand",
            "Onboarding and new-hire setup creates recurring admin burden",
        ],
        "email_subject": "Quick question on payroll workflow",
        "email_draft": (
            "I’m reaching out because we help teams reduce payroll admin time and tighten compliance without changing how managers work day-to-day.\n\n"
            "If you’re open to it, I’d love to ask 2 quick questions:\n"
            "1) What’s the most time-consuming part of payroll today?\n"
            "2) Any onboarding or compliance friction you’d fix this quarter?\n\n"
            "If it’s easier, I can share a 60-second overview and you can tell me if it’s relevant."
        ),
        "linkedin_note": "Hi — quick connect. I work with growing teams to cut payroll admin time + reduce compliance headaches. Open to a brief intro if it’s relevant.",
    }

    if oa_client is None:
        return fallback

    prompt = f"""
Return ONLY valid JSON (no markdown, no commentary).

Company: {company.get("name")}
Industry: {company.get("industry") or "Unknown"}
Territory Zip: {company.get("zip")}

Schema:
{{
  "priority_score": <integer 1-10>,
  "signal_type": "Growth" | "Stable" | "Unknown",
  "reason": <string, 1-2 sentences, no fake facts>,
  "pain_points": [<string>, <string>, <string>],
  "email_subject": <string>,
  "email_draft": <string, under 140 words, include 1-2 bullets or numbered questions>,
  "linkedin_note": <string, under 300 chars>
}}

Rules:
- Be hypothesis-based.
- Do NOT invent internal facts.
- Make the email feel usable and specific to payroll/HR.
"""

    try:
        resp = oa_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
        )
        raw = (resp.choices[0].message.content or "").strip()
        if raw.startswith("```"):
            raw = raw.replace("```json", "").replace("```", "").strip()

        start = raw.find("{")
        end = raw.rfind("}") + 1
        if start != -1 and end != -1:
            raw = raw[start:end]

        data = json.loads(raw)
        return {
            "priority_score": int(data.get("priority_score", fallback["priority_score"])),
            "signal_type": data.get("signal_type", fallback["signal_type"]),
            "reason": data.get("reason", fallback["reason"]),
            "pain_points": data.get("pain_points", fallback["pain_points"]),
            "email_subject": data.get("email_subject", fallback["email_subject"]),
            "email_draft": data.get("email_draft", fallback["email_draft"]),
            "linkedin_note": data.get("linkedin_note", fallback["linkedin_note"]),
        }
    except Exception:
        return fallback


# ---------------------------
# CURATION + SCOREBOARD
# ---------------------------
def curate_top(companies: List[Dict[str, Any]], top_n: int = 10) -> List[Dict[str, Any]]:
    enriched: List[Dict[str, Any]] = []
    for c in companies:
        c.update(ai_generate_intelligence(c))
        c["executives"] = build_buying_committee(c["name"])
        c["executives_found"] = True
        enriched.append(c)

    enriched.sort(key=lambda x: x.get("priority_score", 0), reverse=True)
    return enriched[:top_n]


def build_scoreboard(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not results:
        return {"territory_score": 0, "high": 0, "mid": 0, "low": 0}

    scores = [int(r.get("priority_score", 0)) for r in results]
    territory_score = round(sum(scores) / max(1, len(scores)), 1)
    return {
        "territory_score": territory_score,
        "high": sum(1 for s in scores if s >= 8),
        "mid": sum(1 for s in scores if 5 <= s <= 7),
        "low": sum(1 for s in scores if s <= 4),
    }


# ---------------------------
# ROUTES: HOME / LOGIN / LOGOUT
# ---------------------------
@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    premium = is_premium_request(request)
    email = read_session_email(request)

    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "app_name": APP_NAME,
            "results": [],
            "locked_count": 0,
            "is_premium": premium,
            "error": "",
            "scoreboard": {"territory_score": 0, "high": 0, "mid": 0, "low": 0},
            "email": email,
        },
    )


@app.get("/logout")
def logout():
    resp = RedirectResponse(url="/", status_code=302)
    resp.delete_cookie(SESSION_COOKIE)
    resp.delete_cookie(PREMIUM_COOKIE_NAME)
    return resp


@app.get("/login", response_class=HTMLResponse)
def login_page(request: Request, next: str = "/"):
    # Minimal login page using your existing template system:
    # You can create templates/login.html later. For now, keep it simple.
    html = f"""
    <!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
    <title>Login • SignalZip</title></head>
    <body style="font-family: -apple-system, Segoe UI, Roboto, Arial; padding: 24px; max-width: 520px; margin: auto;">
      <h2>Login</h2>
      <p>Enter your email to get a magic link.</p>
      <form method="post" action="/login">
        <input name="email" type="email" placeholder="you@company.com" required style="width:100%; padding:12px; margin: 8px 0;" />
        <input name="next" type="hidden" value="{next}" />
        <button type="submit" style="padding: 12px 14px; width:100%;">Send link</button>
      </form>
    </body></html>
    """
    return HTMLResponse(html)


@app.post("/login", response_class=HTMLResponse)
def login_send_link(request: Request, email: str = Form(...), next: str = Form("/")):
    email = email.strip().lower()
    token = serializer.dumps({"email": email})

    magic_link = f"{BASE_URL}/login/verify?token={token}&next={quote_plus(next)}"

    # MVP: display the link (you can replace this with email sending later)
    html = f"""
    <!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
    <title>Check your email • SignalZip</title></head>
    <body style="font-family: -apple-system, Segoe UI, Roboto, Arial; padding: 24px; max-width: 760px; margin: auto;">
      <h2>Magic link (Dev mode)</h2>
      <p>In production, you would email this link. For now, copy and open it:</p>
      <p><a href="{magic_link}">{magic_link}</a></p>
      <p style="color:#666;">This link expires in {LOGIN_LINK_TTL_SECONDS//60} minutes.</p>
    </body></html>
    """
    return HTMLResponse(html)


@app.get("/login/verify")
def login_verify(token: str, next: str = "/"):
    try:
        data = serializer.loads(token, max_age=LOGIN_LINK_TTL_SECONDS)
        email = (data or {}).get("email")
        if not email:
            raise HTTPException(status_code=400, detail="Invalid token")

        # Ensure user row exists (free users can exist)
        if not get_user(email):
            upsert_user(email=email, active=0, subscription_status="free")

        resp = RedirectResponse(url=next or "/", status_code=302)
        resp.set_cookie(
            SESSION_COOKIE,
            make_session_token(email),
            httponly=True,
            samesite="lax",
            secure=True,
            path="/"
        )

        return resp

    except SignatureExpired:
        raise HTTPException(status_code=400, detail="Login link expired")
    except BadSignature:
        raise HTTPException(status_code=400, detail="Invalid login link")


# ---------------------------
# STRIPE: CHECKOUT / SUCCESS / PORTAL / WEBHOOK
# ---------------------------
@app.post("/start-checkout")
def start_checkout(request: Request, email: str = Form(...)):
    if not STRIPE_SECRET_KEY or not STRIPE_PRICE_ID:
        raise HTTPException(status_code=500, detail="Stripe is not configured (missing STRIPE_SECRET_KEY or STRIPE_PRICE_ID).")

    email = email.strip().lower()

    # create/get local user row
    if not get_user(email):
        upsert_user(email=email, active=0, subscription_status="free")

    # Create customer (Stripe will dedupe by email in practice, but we will store the returned id)
    customer = stripe.Customer.create(email=email)

    # Store customer id locally (subscription id will come from webhook)
    upsert_user(email=email, stripe_customer_id=customer["id"], active=0, subscription_status="incomplete")

    success_url = f"{BASE_URL}/checkout/success?session_id={{CHECKOUT_SESSION_ID}}"
    cancel_url = f"{BASE_URL}/checkout/cancel"

    session = stripe.checkout.Session.create(
        mode="subscription",
        customer=customer["id"],
        line_items=[{"price": STRIPE_PRICE_ID, "quantity": 1}],
        success_url=success_url,
        cancel_url=cancel_url,
        allow_promotion_codes=True,
        client_reference_id=email,
        metadata={"email": email},
    )

    return RedirectResponse(session.url, status_code=303)


@app.get("/checkout/success")
def checkout_success(session_id: str):
    # The webhook is the source of truth; here we just log the user in for convenience.
    try:
        if STRIPE_SECRET_KEY:
            session = stripe.checkout.Session.retrieve(session_id)
            email = (session.get("metadata") or {}).get("email") or session.get("client_reference_id")
            if not email:
                # Try customer details
                email = ((session.get("customer_details") or {}).get("email")) if session.get("customer_details") else None

            resp = RedirectResponse(url="/", status_code=302)
            if email:
                resp.set_cookie(
                    SESSION_COOKIE,
                    make_session_token(email),
                    httponly=True,
                    samesite="lax",
                    secure=True,
                    path="/"
                )
            # UI convenience cookie (not the security check)
            resp.set_cookie(PREMIUM_COOKIE_NAME, "1", httponly=True, samesite="lax")
            return resp

    except Exception:
        pass

    return RedirectResponse(url="/", status_code=302)


@app.get("/checkout/cancel")
def checkout_cancel():
    return RedirectResponse(url="/", status_code=302)


@app.post("/billing-portal")
def billing_portal(request: Request):
    email = require_email(request)

    user = get_user(email)
    if not user or not user.get("stripe_customer_id"):
        raise HTTPException(status_code=400, detail="No Stripe customer found for this user.")

    if not STRIPE_SECRET_KEY:
        raise HTTPException(status_code=500, detail="Stripe not configured.")

    portal = stripe.billing_portal.Session.create(
        customer=user["stripe_customer_id"],
        return_url=f"{BASE_URL}/",
    )
    return RedirectResponse(portal.url, status_code=303)

@app.get("/dev/login-me")
def dev_login_me():
    email = "stephenbyron31@email.com"

    resp = RedirectResponse(url="/", status_code=302)
    resp.set_cookie(
        SESSION_COOKIE,
        make_session_token(email),
        httponly=True,
        samesite="lax",
        secure=True,
        path="/"
    )
    return resp


@app.post("/stripe/webhook")
async def stripe_webhook(request: Request):
    if not STRIPE_WEBHOOK_SECRET:
        raise HTTPException(status_code=500, detail="Missing STRIPE_WEBHOOK_SECRET")

    payload = await request.body()
    sig = request.headers.get("stripe-signature")

    try:
        event = stripe.Webhook.construct_event(payload, sig, STRIPE_WEBHOOK_SECRET)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Webhook error: {str(e)}")

    event_type = event["type"]
    data = event["data"]["object"]

    # 1) Checkout completed: attach subscription/customer to user + mark active
    if event_type == "checkout.session.completed":
        email = (data.get("metadata") or {}).get("email") or data.get("client_reference_id")
        customer_id = data.get("customer")
        subscription_id = data.get("subscription")

        if email:
            # fetch subscription status
            status = None
            active = 0
            try:
                sub = stripe.Subscription.retrieve(subscription_id)
                status = (sub.get("status") or "").lower()
                active = 1 if status in ("active", "trialing") else 0
            except Exception:
                status = "unknown"
                active = 0

            upsert_user(
                email=email,
                stripe_customer_id=customer_id,
                stripe_subscription_id=subscription_id,
                subscription_status=status,
                active=active,
                last_status_check_utc=_utc_now().isoformat(),
            )

    # 2) Subscription updated: keep DB in sync
    if event_type in ("customer.subscription.updated", "customer.subscription.deleted"):
        subscription_id = data.get("id")
        customer_id = data.get("customer")
        status = (data.get("status") or "").lower()

        # we may not know email here unless stored, so we’ll find by customer
        # (simple approach: list customer email from Stripe)
        email = None
        try:
            cust = stripe.Customer.retrieve(customer_id)
            email = (cust.get("email") or "").strip().lower() if cust else None
        except Exception:
            email = None

        if email:
            active = 1 if status in ("active", "trialing") else 0
            upsert_user(
                email=email,
                stripe_customer_id=customer_id,
                stripe_subscription_id=subscription_id,
                subscription_status=status,
                active=active,
                last_status_check_utc=_utc_now().isoformat(),
            )

    return {"received": True}


# ---------------------------
# APP ROUTES
# ---------------------------
@app.post("/generate", response_class=HTMLResponse)
def generate(request: Request, zip_codes: str = Form(...), industry: str = Form("")):
    global LAST_RESULTS

    premium = is_premium_request(request)
    email = read_session_email(request)

    zips = [z.strip() for z in (zip_codes or "").split(",") if z.strip()]
    if not zips:
        return templates.TemplateResponse(
            "index.html",
            {
                "request": request,
                "app_name": APP_NAME,
                "results": [],
                "locked_count": 0,
                "is_premium": premium,
                "error": "Please enter at least one zip code.",
                "scoreboard": {"territory_score": 0, "high": 0, "mid": 0, "low": 0},
                "email": email,
            },
        )

    all_companies: List[Dict[str, Any]] = []
    try:
        for z in zips[:10]:
            all_companies.extend(fetch_companies_google(z, industry))
    except Exception as e:
        return templates.TemplateResponse(
            "index.html",
            {
                "request": request,
                "app_name": APP_NAME,
                "results": [],
                "locked_count": 0,
                "is_premium": premium,
                "error": f"Error fetching companies: {str(e)}",
                "scoreboard": {"territory_score": 0, "high": 0, "mid": 0, "low": 0},
                "email": email,
            },
        )

    if not all_companies:
        return templates.TemplateResponse(
            "index.html",
            {
                "request": request,
                "app_name": APP_NAME,
                "results": [],
                "locked_count": 0,
                "is_premium": premium,
                "error": "No companies found. Try a nearby zip or remove the industry filter.",
                "scoreboard": {"territory_score": 0, "high": 0, "mid": 0, "low": 0},
                "email": email,
            },
        )

    curated = curate_top(all_companies, top_n=10)
    LAST_RESULTS = curated

    scoreboard = build_scoreboard(curated)

    if not premium:
        visible_results = curated[:FREE_LIMIT]
        locked_count = max(0, len(curated) - FREE_LIMIT)
    else:
        visible_results = curated
        locked_count = 0

    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "app_name": APP_NAME,
            "results": visible_results,
            "locked_count": locked_count,
            "is_premium": premium,
            "error": "",
            "scoreboard": scoreboard,
            "email": email,
        },
    )


@app.get("/export.csv")
def export_csv(request: Request):
    premium = is_premium_request(request)
    if not premium or not LAST_RESULTS:
        return StreamingResponse(io.BytesIO(b"Upgrade required for CSV export."), media_type="text/plain")

    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["company", "priority_score", "signal_type", "reason"])

    for r in LAST_RESULTS:
        writer.writerow([r.get("name"), r.get("priority_score"), r.get("signal_type"), r.get("reason")])

    output.seek(0)
    return StreamingResponse(
        iter([output.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": 'attachment; filename="signalzip_results.csv"'},
    )
