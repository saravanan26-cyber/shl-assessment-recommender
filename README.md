# SHL Assessment Recommender

A conversational agent that recommends SHL assessments to hiring managers. Built for the SHL Labs AI Intern take-home assignment.

## Quick Start

```bash
# 1. Clone / unzip the project
cd shl-recommender

# 2. Set your API key
cp .env.example .env
# Edit .env: ANTHROPIC_API_KEY=sk-ant-...

# 3. Run locally
bash run.sh
```

The server starts at `http://localhost:8000`.

## API

### `GET /health`
```json
{"status": "ok"}
```

### `POST /chat`

**Request:**
```json
{
  "messages": [
    {"role": "user", "content": "Hiring a Java developer who works with stakeholders"},
    {"role": "assistant", "content": "Sure. What is the seniority level?"},
    {"role": "user", "content": "Mid-level, around 4 years"}
  ]
}
```

**Response:**
```json
{
  "reply": "Got it. Here are 5 assessments that fit a mid-level Java dev with stakeholder needs.",
  "recommendations": [
    {"name": "Java 8 (New)", "url": "https://www.shl.com/...", "test_type": "K"},
    {"name": "OPQ32r", "url": "https://www.shl.com/...", "test_type": "P"}
  ],
  "end_of_conversation": false
}
```

- `recommendations` is `[]` when still clarifying or refusing
- `end_of_conversation` is `true` only when the agent considers the task complete

## Architecture

```
User message
     │
     ▼
┌─────────────────────────────────────────┐
│           FastAPI /chat                  │
│                                          │
│  1. Safety check (off-topic / injection) │
│  2. Build retrieval query from history   │
│  3. TF-IDF retrieval over catalog        │
│  4. Format retrieved context             │
│  5. Call Claude claude-sonnet-4-20250514 with context    │
│  6. Parse + validate JSON response       │
│  7. Sanitize URLs against catalog        │
│  8. Return ChatResponse                  │
└─────────────────────────────────────────┘
```

### Key design decisions

**Stateless by design.** Every `/chat` call receives the full conversation history. No session storage needed — simplifies deployment and scaling.

**TF-IDF over embeddings.** Chosen for zero-dependency retrieval (sklearn only). Bigrams + sublinear TF weighting + semantic expansion terms in the document construction gives strong recall without a vector database or GPU. Trade-off: less semantic flexibility than dense embeddings, but fast and fully deterministic.

**Grounded retrieval.** Retrieved catalog entries are injected into the system prompt on every turn. The LLM is explicitly instructed to only return names and URLs from the provided context, never from its training data.

**URL whitelist validation.** Every URL in the LLM response is cross-checked against the scraped catalog before being returned. Hallucinated URLs are silently dropped. This is the hardest guarantee required by the assignment.

**Turn cap awareness.** When `total_turns >= 6`, a force-recommendation note is appended to the user message. This ensures the 8-turn cap is never exceeded without a shortlist.

**One clarifying question max.** The system prompt and context-sufficiency heuristic (`has_enough_context()`) work together to ensure the agent asks at most one clarifying question before committing to a recommendation.

## Deployment

### Render (recommended — free tier)

1. Push this repo to GitHub
2. Create a new Web Service on [render.com](https://render.com)
3. Set environment variable: `ANTHROPIC_API_KEY`
4. Build command: `pip install -r requirements.txt && python scraper.py && python build_index.py`
5. Start command: `uvicorn app.main:app --host 0.0.0.0 --port $PORT`

Or use the included `render.yaml`:
```bash
# render.yaml is already configured — just connect your GitHub repo
```

### Docker

```bash
docker build -t shl-recommender .
docker run -p 8000:8000 -e ANTHROPIC_API_KEY=sk-ant-... shl-recommender
```

### Railway / Fly.io

Both support the Dockerfile directly. Set `ANTHROPIC_API_KEY` as an environment variable.

## Evaluation

```bash
# Run the local evaluation suite
export ANTHROPIC_API_KEY=sk-ant-...
python3 evaluate.py
```

Covers 10 traces testing:
- Schema compliance
- Recall@10 on expected assessments
- Off-topic refusal
- No recommendation on vague first turn
- Comparison handling
- Refinement mid-conversation

## File structure

```
shl-recommender/
├── app/
│   └── main.py          # FastAPI app + agent logic
├── data/
│   ├── catalog.json     # Scraped SHL catalog (generated)
│   └── index.pkl        # TF-IDF index (generated)
├── scraper.py           # Catalog scraper + fallback catalog
├── build_index.py       # Index builder + retrieval utilities
├── evaluate.py          # Local evaluation harness
├── requirements.txt
├── Dockerfile
├── render.yaml
├── run.sh
└── README.md
```

## What didn't work / iteration notes

- **Keyword-only retrieval** missed semantic matches (e.g. "stakeholder communication" → verbal reasoning). Added semantic expansion terms to document construction.
- **Pure LLM without retrieval** hallucinated product names and URLs. Grounding via retrieved context + URL whitelist solved this.
- **Recommending too eagerly** on vague first turns failed the behavior probe. The `has_enough_context()` heuristic + explicit system prompt instruction fixed it.
- **JSON parsing failures** from the LLM when the response wrapped JSON in markdown. Added regex stripping + fallback extraction.
