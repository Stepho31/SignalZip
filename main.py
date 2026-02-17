import os
import csv
import io
import json
from typing import Any, Dict, List
from urllib.parse import quote_plus

import requests
from dotenv import load_dotenv
from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse, StreamingResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from openai import OpenAI
from itsdangerous import URLSafeTimedSerializer
from database import init_db, get_user, upsert_user

load_dotenv()

SECRET_KEY = os.getenv("SECRET_KEY", "supersecret")
serializer = URLSafeTimedSerializer(SECRET_KEY)

init_db()

APP_NAME = "SignalZip — Payroll Prospecting Intelligence"

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "").strip()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()

# Optional: force premium for debugging
FORCE_PREMIUM = os.getenv("FORCE_PREMIUM", "").strip().lower() in ("1", "true", "yes")

oa_client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None

app = FastAPI()
templates = Jinja2Templates(directory="templates")

LAST_RESULTS: List[Dict[str, Any]] = []

# ---------------------------
# FREEMIUM SETTINGS
# ---------------------------
FREE_LIMIT = 3
PREMIUM_COOKIE_NAME = "signalzip_pro"  # "1" means premium


def is_premium_request(request: Request) -> bool:
    if FORCE_PREMIUM:
        return True
    return request.cookies.get(PREMIUM_COOKIE_NAME, "0") == "1"


# ----------------------------------------
# GOOGLE PLACES SEARCH (New)
# ----------------------------------------
def fetch_companies_google(zip_code: str, industry: str = "") -> List[Dict[str, Any]]:
    if not GOOGLE_API_KEY:
        raise Exception("Missing GOOGLE_API_KEY")

    url = "https://places.googleapis.com/v1/places:searchText"
    headers = {
        "Content-Type": "application/json",
        "X-Goog-Api-Key": GOOGLE_API_KEY,
        "X-Goog-FieldMask": "places.displayName,places.formattedAddress",
    }

    query_text = (
        f"{industry} businesses near {zip_code} USA"
        if industry
        else f"businesses near {zip_code} USA"
    )

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
            {
                "name": name,
                "address": place.get("formattedAddress") or "",
                "zip": zip_code,
                "industry": industry,
            }
        )

    return results


# ----------------------------------------
# BUYING COMMITTEE (Strategic targets)
# ----------------------------------------
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

    executives = []
    for title, short in roles:
        executives.append(
            {
                "name": short,
                "title": title,
                "linkedin_url": build_linkedin_search(company_name, title),
            }
        )
    return executives


# ----------------------------------------
# AI INTELLIGENCE (Robust)
# ----------------------------------------
def ai_generate_intelligence(company: Dict[str, Any]) -> Dict[str, Any]:
    # Good fallback even without OpenAI key
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
- Do NOT invent internal facts (no “they just raised funding” unless clearly stated as a hypothesis).
- Make the email feel usable and specific to payroll/HR.
"""

    try:
        resp = oa_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
        )

        raw = (resp.choices[0].message.content or "").strip()

        # Strip common fencing
        if raw.startswith("```"):
            raw = raw.replace("```json", "").replace("```", "").strip()

        # Extract JSON window
        start = raw.find("{")
        end = raw.rfind("}") + 1
        if start != -1 and end != -1:
            raw = raw[start:end]

        data = json.loads(raw)

        # Normalize keys + guardrails
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


# ----------------------------------------
# CURATION + SCOREBOARD
# ----------------------------------------
def curate_top(companies: List[Dict[str, Any]], top_n: int = 10) -> List[Dict[str, Any]]:
    enriched: List[Dict[str, Any]] = []
    for c in companies:
        analysis = ai_generate_intelligence(c)
        c.update(analysis)

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

    high = sum(1 for s in scores if s >= 8)
    mid = sum(1 for s in scores if 5 <= s <= 7)
    low = sum(1 for s in scores if s <= 4)

    return {
        "territory_score": territory_score,
        "high": high,
        "mid": mid,
        "low": low,
    }


# ----------------------------------------
# ROUTES
# ----------------------------------------
@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    premium = is_premium_request(request)
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
        },
    )


@app.get("/upgrade")
def upgrade(request: Request):
    """
    Stub endpoint.
    Today: sets a cookie to simulate premium.
    Later: replace this with Stripe Checkout success webhook → set premium.
    """
    resp = RedirectResponse(url="/", status_code=302)
    resp.set_cookie(PREMIUM_COOKIE_NAME, "1", httponly=True, samesite="lax")
    return resp


@app.get("/logout")
def logout(request: Request):
    """Helper for testing freemium again."""
    resp = RedirectResponse(url="/", status_code=302)
    resp.delete_cookie(PREMIUM_COOKIE_NAME)
    return resp


@app.post("/generate", response_class=HTMLResponse)
def generate(request: Request, zip_codes: str = Form(...), industry: str = Form("")):
    global LAST_RESULTS

    premium = is_premium_request(request)
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
        },
    )


@app.get("/export.csv")
def export_csv(request: Request):
    premium = is_premium_request(request)

    if not LAST_RESULTS or not premium:
        return StreamingResponse(
            io.BytesIO(b"Upgrade required for CSV export."),
            media_type="text/plain",
        )

    output = io.StringIO()
    writer = csv.writer(output)

    writer.writerow(["company", "priority_score", "signal_type", "reason"])
    for r in LAST_RESULTS:
        writer.writerow(
            [
                r.get("name"),
                r.get("priority_score"),
                r.get("signal_type"),
                r.get("reason"),
            ]
        )

    output.seek(0)
    return StreamingResponse(
        iter([output.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": 'attachment; filename="signalzip_results.csv"'},
    )
