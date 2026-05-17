
"""
SHL Assessment Recommender - FastAPI Service
POST /chat  — stateless conversational agent
GET  /health — readiness probe
"""

import json
import os
import pickle
import re
from pathlib import Path
from typing import Optional

import numpy as np
from dotenv import load_dotenv
from groq import Groq
from sklearn.metrics.pairwise import cosine_similarity

load_dotenv()

SYSTEM_PROMPT = """
You are an SHL assessment recommendation assistant.

Always return responses in valid JSON format.

Only recommend assessments from the SHL catalog.
Ask clarifying questions for vague queries.
Return concise and grounded responses.
"""

# Proper validation set for test types
VALID_TYPES = {"A", "B", "C", "D", "E", "K", "M", "P", "R", "S"}

# FastAPI + Pydantic — required for the server, optional for unit tests
try:
    from fastapi import FastAPI, HTTPException
    from fastapi.middleware.cors import CORSMiddleware
    from pydantic import BaseModel

    _HAS_FASTAPI = True
except ImportError:
    _HAS_FASTAPI = False

    class BaseModel:  # type: ignore
        def __init__(self, **kwargs):
            for k, v in kwargs.items():
                setattr(self, k, v)

    class FastAPI:  # type: ignore
        def __init__(self, **kwargs):
            pass

        def on_event(self, *a):
            return lambda f: f

        def get(self, *a, **kw):
            return lambda f: f

        def post(self, *a, **kw):
            return lambda f: f

        def add_middleware(self, *a, **kw):
            pass

    class HTTPException(Exception):  # type: ignore
        def __init__(self, status_code=500, detail=""):
            super().__init__(detail)

    class CORSMiddleware:
        pass

# ── Config ───────────────────────────────────────────────────────────────────

GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
MODEL = "llama-3.1-8b-instant"
MAX_TOKENS = 1024

BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
INDEX_PATH = DATA_DIR / "index.pkl"
CATALOG_PATH = DATA_DIR / "catalog.json"

# Safe catalog loading
try:
    with open(CATALOG_PATH, "r", encoding="utf-8") as f:
        catalog = json.load(f)
    print(f"✅ Loaded catalog with {len(catalog)} assessments")
except FileNotFoundError:
    print(f"❌ catalog.json not found at {CATALOG_PATH}. Starting with empty catalog.")
    catalog = []
except json.JSONDecodeError as e:
    print(f"❌ catalog.json is malformed: {e}. Starting with empty catalog.")
    catalog = []

ROLE_SKILLS = {
    "java backend developer": [
        "java", "sql", "api", "spring", "backend"
    ],
    "data analyst": [
        "sql", "python", "excel",
        "analytics", "statistics", "numerical"
    ],
    "frontend developer": [
        "javascript", "html", "css", "react"
    ],
    "python developer": [
        "python", "api", "django", "flask"
    ],
    "cyber security": [
        "security", "network", "risk", "incident"
    ]
}


def search_assessments(query: str):
    query = query.lower()
    results = []
    seen = set()

    keywords = [
        word for word in query.split()
        if len(word) > 3 and word not in [
            "need", "with", "and", "for", "role",
            "assessment", "developer"
        ]
    ]

    role_query = query.lower()

    if role_query in ROLE_SKILLS:
        keywords.extend(ROLE_SKILLS[role_query])

    for item in catalog:
        searchable_text = (
            item.get("name", "") + " " +
            item.get("description", "") + " " +
            " ".join(item.get("job_levels", [])) + " " +
            " ".join(item.get("test_type_labels", []))
        ).lower()

        score = 0

        # Regex word boundary matching
        for word in keywords:
            if re.search(rf"\b{re.escape(word)}\b", searchable_text):
                score += 1

        if "java" in query and (
            re.search(r"\bjava\b", searchable_text) and
            not re.search(r"\bjavascript\b", searchable_text)
        ):
            score += 5

        if "backend" in query and (
            re.search(r"\bjava\b", searchable_text) or
            re.search(r"\bsql\b", searchable_text) or
            re.search(r"\bapi\b", searchable_text)
        ):
            score += 3

        if "data" in query and (
            re.search(r"\bsql\b", searchable_text) or
            re.search(r"\bpython\b", searchable_text) or
            re.search(r"\bexcel\b", searchable_text) or
            re.search(r"\banalytics\b", searchable_text) or
            re.search(r"\bstatistics\b", searchable_text) or
            re.search(r"\bnumerical\b", searchable_text)
        ):
            score += 5

        if "coding" in query and (
            "knowledge" in searchable_text or
            "skill" in searchable_text
        ):
            score += 2

        if "aptitude" in query and (
            "ability" in searchable_text or
            "reasoning" in searchable_text
        ):
            score += 2

        if score > 0:
            name = item.get("name")

            if name not in seen:
                seen.add(name)

                results.append({
                    "name": name,
                    "url": item.get("url"),
                    "test_type": ", ".join(item.get("test_type_labels", [])),
                    "score": score
                })

    results = sorted(results, key=lambda x: x["score"], reverse=True)

    return results[:5]


print("Loaded assessments:", len(catalog))

# ── App setup ─────────────────────────────────────────────────────────────────

app = FastAPI(title="SHL Assessment Recommender", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Load index on startup ─────────────────────────────────────────────────────

_index: Optional[dict] = None
_valid_urls: set[str] = set()


@app.on_event("startup")
async def load_index():
    global _index, _valid_urls

    try:
        with open(INDEX_PATH, "rb") as f:
            _index = pickle.load(f)

        _valid_urls = {a["url"] for a in _index["catalog"]}

        print(f"✅ Loaded index with {len(_index['catalog'])} assessments")

    except Exception as e:
        print(f"❌ Failed to load index: {e}")
        raise RuntimeError(f"Index not found. Run build_index.py first. {e}")


# ── Pydantic models ───────────────────────────────────────────────────────────

class Message(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    messages: list[Message]


class Recommendation(BaseModel):
    name: str
    url: str
    test_type: str


class ChatResponse(BaseModel):
    reply: str
    recommendations: list[Recommendation]
    end_of_conversation: bool


# ── Retrieval ─────────────────────────────────────────────────────────────────


def retrieve(query: str, top_k: int = 15) -> list[dict]:
    """Semantic retrieval over the catalog using TF-IDF cosine similarity."""

    if _index is None:
        return []

    vectorizer = _index["vectorizer"]
    tfidf_matrix = _index["tfidf_matrix"]
    catalog = _index["catalog"]

    query_vec = vectorizer.transform([query])
    scores = cosine_similarity(query_vec, tfidf_matrix).flatten()
    top_indices = np.argsort(scores)[::-1][:top_k]

    results = []

    for idx in top_indices:
        if scores[idx] > 0.001:
            results.append({**catalog[idx], "_score": float(scores[idx])})

    return results



def format_catalog_snippet(assessment: dict) -> str:
    types = ", ".join(assessment.get("test_type_labels", assessment.get("test_types", [])))
    levels = ", ".join(assessment.get("job_levels", [])) or "All levels"
    desc = assessment.get("description", "No description available.")
    duration = assessment.get("duration", "")
    remote = "Yes" if assessment.get("remote_testing") else "No"
    adaptive = "Yes" if assessment.get("adaptive_irt") else "No"

    return (
        f"**{assessment['name']}**\n"
        f"  URL: {assessment['url']}\n"
        f"  Type: {types}\n"
        f"  Levels: {levels}\n"
        f"  Duration: {duration}\n"
        f"  Remote: {remote} | Adaptive/IRT: {adaptive}\n"
        f"  Description: {desc}"
    )


# ── Conversation analysis ──────────────────────────────────────────────────────


def build_context_query(messages: list[Message]) -> str:
    all_user_text = " ".join(
        m.content for m in messages if m.role == "user"
    )

    return all_user_text



def count_turns(messages: list[Message]) -> int:
    return len(messages)



def has_enough_context(messages: list[Message]) -> bool:
    all_text = " ".join(m.content for m in messages if m.role == "user").lower()

    role_signals = [
        "developer", "engineer", "manager", "analyst", "designer",
        "scientist", "sales", "customer", "service", "hire", "hiring",
        "recruit", "role", "position", "job", "candidate", "team",
        "graduate", "intern", "director", "executive", "lead", "head of",
    ]

    has_role = any(s in all_text for s in role_signals)

    if not has_role:
        return False

    extra_signals = [
        "junior", "senior", "mid", "entry", "level", "years", "experience",
        "personality", "cognitive", "ability", "skill", "technical", "coding",
        "behaviour", "behavioral", "aptitude", "reasoning",
        "java", "python", "sql", "data", "cloud", "aws", "leadership",
        "communication", "sales", "finance", "customer",
        "urgent", "asap", "bulk", "volume", "large scale",
    ]

    has_extra = any(s in all_text for s in extra_signals)

    user_turns = sum(1 for m in messages if m.role == "user")

    if user_turns >= 3:
        return True

    return has_role and has_extra


_VAGUE_PATTERNS = [
    r"^assessment$",
    r"^test$",
    r"^need\s+an?\s+(assessment|test)$",
    r"^suggest\s+(an?\s+)?(assessment|test)$",
    r"^recommend\s+(an?\s+)?(assessment|test)$",
    r"^i\s+need\s+a\s+(test|assessment)$",
]



def is_vague_query(query: str) -> bool:
    q = query.lower().strip()

    if len(q.split()) <= 3:
        for pattern in _VAGUE_PATTERNS:
            if re.fullmatch(pattern, q):
                return True

        role_signals = [
            "developer", "engineer", "manager", "analyst", "designer",
            "scientist", "sales", "customer", "java", "python", "sql",
            "data", "leadership", "finance", "frontend", "backend",
        ]

        if not any(s in q for s in role_signals):
            return True

    return False



def is_comparison_request(messages: list[Message]) -> bool:
    last_user = next((m.content for m in reversed(messages) if m.role == "user"), "")

    comparison_phrases = [
        "difference between", "compare", "vs", "versus", "which is better",
        "what's the difference", "how does", "contrast",
    ]

    return any(p in last_user.lower() for p in comparison_phrases)



def is_off_topic(messages: list[Message]) -> bool:
    last_user = next((m.content for m in reversed(messages) if m.role == "user"), "").lower()

    off_topic_signals = [
        "ignore previous", "ignore all", "forget your instructions",
        "you are now", "pretend you are", "act as",
        "legal advice", "medical advice", "lawsuit", "discrimination",
        "hire based on race", "hire based on gender",
        "write code for me", "write an essay",
        "what's the weather", "tell me a joke",
    ]

    return any(s in last_user for s in off_topic_signals)



def extract_comparison_names(text: str, catalog: list[dict]) -> list[dict]:
    text_lower = text.lower()
    matched = []

    for a in catalog:
        if a["name"].lower() in text_lower:
            matched.append(a)

        name = a["name"].lower()

        for part in name.split():
            if len(part) >= 4 and part in text_lower:
                if a not in matched:
                    matched.append(a)

    return matched[:5]


# ── LLM call ──────────────────────────────────────────────────────────────────


async def call_llm(messages: list[Message], context: str) -> dict:
    """Call Groq API using Llama3 and return parsed JSON response."""

    client = Groq(api_key=GROQ_API_KEY)

    api_messages = []

    system_with_context = SYSTEM_PROMPT

    if context:
        system_with_context += (
            f"\n\n---\n"
            f"RELEVANT CATALOG ENTRIES (use ONLY these for recommendations):\n"
            f"{context}\n---"
        )

    api_messages.append({
        "role": "system",
        "content": system_with_context
    })

    for m in messages:
        api_messages.append({
            "role": m.role,
            "content": m.content
        })

    # Try JSON mode first
    try:
        completion = client.chat.completions.create(
            model=MODEL,
            messages=api_messages,
            temperature=0.3,
            max_tokens=MAX_TOKENS,
            response_format={"type": "json_object"},
        )

    # Fallback without response_format for Groq compatibility
    except Exception:
        completion = client.chat.completions.create(
            model=MODEL,
            messages=api_messages,
            temperature=0.3,
            max_tokens=MAX_TOKENS,
        )

    raw = completion.choices[0].message.content.strip()

    print("RAW RESPONSE:", raw)

    try:
        parsed = json.loads(raw)

    except json.JSONDecodeError:
        json_match = re.search(r'\{.*\}', raw, re.DOTALL)

        if json_match:
            parsed = json.loads(json_match.group())

        else:
            parsed = {
                "reply": raw,
                "recommendations": [],
                "end_of_conversation": False,
            }

    return parsed


# ── Safety: validate recommendations ─────────────────────────────────────────


def validate_recommendations(recs: list[dict]) -> list[Recommendation]:
    """
    Ensure every recommendation is real and its URL is in our catalog.
    Silently drop hallucinated entries.
    """

    if _index is None:
        return []

    validated = []
    catalog = _index["catalog"]

    catalog_by_name = {a["name"].lower(): a for a in catalog}
    catalog_by_url = {a["url"]: a for a in catalog}

    for rec in recs:
        name = rec.get("name", "")
        url = rec.get("url", "")
        test_type = rec.get("test_type", "")

        real_entry = None

        if url in catalog_by_url:
            real_entry = catalog_by_url[url]

        elif name.lower() in catalog_by_name:
            real_entry = catalog_by_name[name.lower()]

        else:
            for cat_name, cat_entry in catalog_by_name.items():
                if name.lower() in cat_name or cat_name in name.lower():
                    real_entry = cat_entry
                    break

        if real_entry:
            primary_type = (
                test_type if test_type in VALID_TYPES
                else (
                    real_entry["test_types"][0]
                    if real_entry["test_types"]
                    else "K"
                )
            )

            validated.append(Recommendation(
                name=real_entry["name"],
                url=real_entry["url"],
                test_type=primary_type,
            ))

    seen_urls = set()
    deduped = []

    for r in validated:
        if r.url not in seen_urls:
            seen_urls.add(r.url)
            deduped.append(r)

    return deduped[:10]


# ── Endpoint: /health ──────────────────────────────────────────────────────────


@app.get("/health")
async def health():
    if _index is None:
        raise HTTPException(status_code=503, detail="Index not loaded")

    return {"status": "ok"}


# ── Endpoint: /chat ───────────────────────────────────────────────────────────


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    if not request.messages:
        raise HTTPException(status_code=400, detail="messages cannot be empty")

    if not GROQ_API_KEY:
        raise HTTPException(status_code=500, detail="GROQ_API_KEY not configured")

    messages = request.messages
    total_turns = count_turns(messages)
    user_query = messages[-1].content

    # Guard: off-topic / prompt injection
    if is_off_topic(messages):
        return ChatResponse(
            reply="I can only help with SHL assessment recommendations and comparisons.",
            recommendations=[],
            end_of_conversation=False,
        )

    # Guard: vague queries
    if is_vague_query(user_query):
        return ChatResponse(
            reply="Please specify the role, skills, or seniority level you are hiring for.",
            recommendations=[],
            end_of_conversation=False,
        )

    # Build retrieval query from full conversation
    context_query = build_context_query(messages)

    # Retrieve relevant catalog entries
    retrieved = retrieve(context_query, top_k=20)
    quick_results = search_assessments(context_query)

    # Comparison handling with safe _index check
    if is_comparison_request(messages):
        last_user = next((m.content for m in reversed(messages) if m.role == "user"), "")

        comparison_catalog = _index["catalog"] if _index else []

        comparison_targets = extract_comparison_names(
            last_user,
            comparison_catalog
        )

        comparison_urls = {a["url"] for a in comparison_targets}

        retrieved = comparison_targets + [
            r for r in retrieved
            if r["url"] not in comparison_urls
        ]

    # Format context for LLM
    top_retrieved = retrieved[:12]

    context_str = "\n\n".join(
        format_catalog_snippet(a) for a in top_retrieved
    )

    # Determine if we should force a recommendation
    force_note = ""

    if total_turns >= 6:
        force_note = (
            "\n\nIMPORTANT: This conversation has had many turns. "
            "You MUST provide a recommendation shortlist NOW based on the context you have. "
            "Do not ask another clarifying question."
        )

    if force_note:
        messages = list(messages)
        last = messages[-1]

        if last.role == "user":
            messages[-1] = Message(
                role="user",
                content=last.content + force_note
            )

    # Call LLM
    try:
        parsed = await call_llm(messages, context_str)

    except TimeoutError:
        return ChatResponse(
            reply="I'm taking too long to respond. Please try again.",
            recommendations=[],
            end_of_conversation=False,
        )

    except Exception as e:
        raise HTTPException(status_code=502, detail=f"LLM error: {str(e)}")

    # Extract and validate fields
    reply = parsed.get("reply", "")
    raw_recs = parsed.get("recommendations", [])
    end_of_conv = bool(parsed.get("end_of_conversation", False))

    validated_recs = validate_recommendations(raw_recs)

    # Safety: no recs on first turn without enough context
    user_turns = sum(1 for m in messages if m.role == "user")

    if user_turns <= 1 and not has_enough_context(messages):
        validated_recs = []

    # If end_of_conversation is true but no recs, that's invalid
    if end_of_conv and not validated_recs:
        end_of_conv = False

    # Convert quick_results into Recommendation objects
    quick_recs = [
        Recommendation(
            name=r["name"],
            url=r["url"],
            test_type=r.get("test_type", "K"),
        )
        for r in quick_results
    ] if quick_results else []

    return ChatResponse(
        reply=reply,
        recommendations=quick_recs if quick_recs else validated_recs,
        end_of_conversation=end_of_conv,
    )


# ── Dev server ────────────────────────────────────────────────────────────────


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )
    