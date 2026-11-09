# Internal Policy Q&A Service

A Retrieval-Augmented Generation (RAG) service that answers employee questions
about internal company policies. It retrieves the most relevant passages from a
set of Markdown policy documents and uses a Groq-hosted LLM to produce a concise,
**grounded** answer with **citations** — and **refuses** when the policies do not
contain enough information to answer.

---

## 1. Overview & objectives

- Give employees fast, accurate answers to policy questions (leave, remote work,
  IT security, access control, expenses).
- Ground every answer in the source documents and cite the exact file + section.
- Refuse out-of-scope or unsupported questions instead of guessing.
- Ship a small, production-conscious FastAPI service: authenticated, rate-limited,
  observable, and tested.

## 2. Features & capabilities

- **RAG pipeline:** Markdown → chunks → embeddings → vector search → LLM answer.
- **Citations:** each answer references `source_filename` + `section_heading`.
- **Refusal:** low-relevance retrieval (below a score floor) is refused *without*
  calling the LLM; the model is also instructed to refuse when context is weak.
- **API-key authentication** via the `X-API-Key` header (fails closed).
- **Abuse protection:** per-API-key rate limiting (429), request-body size limit
  (413), question-length limit (422), and a bounded Groq timeout (504).
- **Centralized logging** with per-request IDs; secrets and question text are
  never logged.
- **Evaluation harness** for retrieval quality (offline) and answer quality.
- **Offline, deterministic test suite** (no network / no Groq / no model download).

## 3. Technology stack

- **Python 3.12**
- **FastAPI** + **Uvicorn** (HTTP API)
- **Pydantic** / **pydantic-settings** (schemas & configuration)
- **FastEmbed** (ONNX) with `sentence-transformers/all-MiniLM-L6-v2` (384-dim
  embeddings) + **NumPy** (exact cosine vector search — no external vector DB)
- **Groq** Python SDK (LLM answer generation)
- **pytest** (tests)

## 4. Architecture & RAG workflow

```
Markdown policies ──► load ──► parse (sections) ──► chunk ──► embed ──► Vector store
documents/policies     loader    parser              chunker   FastEmbed   (.npy + metadata.json)
                                                                               │
Question ──► Auth ──► rate limit ──► embed query ──► cosine search (top-k) ─────┘
 (X-API-Key)                                             │
                                                         ▼
                                   score < ANSWER_MIN_SCORE?  ──yes──► REFUSE (no LLM call)
                                                         │ no
                                                         ▼
                              build prompt (numbered chunks + headings)
                                                         │
                                                         ▼
                                    Groq LLM ──► grounded answer OR sentinel refusal
                                                         │
                                                         ▼
                                   AnswerResponse { answer, refused, citations[] }
```

1. **Policy documents** — Markdown files in `documents/policies/` (one per policy).
2. **Loading & chunking** — files are loaded, split into sections by heading, then
   chunked (default 800 chars, 100 overlap), keeping section metadata for citations.
3. **Embeddings** — chunks are embedded with FastEmbed (`all-MiniLM-L6-v2`, 384-d).
   The production index additionally prepends `document_title > section_heading` to
   each chunk's embedding text and down-weights boilerplate (title / "Related
   Policies") chunks (see *Vector index* below).
4. **Vector store & retrieval** — vectors are L2-normalized and stored as a NumPy
   `.npy` file plus a `metadata.json` sidecar; retrieval is exact cosine similarity
   returning the top-k chunks with scores.
5. **Answer generation** — the top chunks are formatted into a grounded prompt and
   sent to the Groq model; output is capped at `GROQ_MAX_TOKENS`.
6. **Citations & refusal** — answers cite the chunks used; if the best retrieval
   score is below `ANSWER_MIN_SCORE`, or the model emits the refusal sentinel, the
   service returns a refusal with no citations.

## 5. Project structure

```
policy-qa-service/
├── app/
│   ├── main.py                 # FastAPI app, lifespan, request-id middleware
│   ├── config.py               # Centralized settings (env / .env)
│   ├── logging_config.py       # Centralized logging + request-id context
│   ├── api/
│   │   ├── routes.py           # /ask route + dependency wiring
│   │   ├── auth.py             # X-API-Key authentication
│   │   └── limits.py           # Rate limiting + request-size limits
│   ├── ingestion/              # loader, parser, chunker, pipeline
│   ├── retrieval/              # embedder, vector_store, retriever, augment
│   ├── llm/                    # base, groq_client, prompt
│   ├── services/               # indexing (build/search CLI), qa (orchestration)
│   ├── schemas/                # ingestion, retrieval, answer (Pydantic models)
│   └── evaluation/             # dataset, metrics, retrieval_eval, answer_eval
├── documents/policies/         # Markdown policy documents (source of truth)
├── evaluation/
│   ├── dataset.jsonl           # Labeled evaluation set (tracked)
│   └── reports/                # Generated eval reports (git-ignored)
├── tests/                      # Offline test suite
├── requirements.txt            # Core runtime deps
├── requirements-ml.txt         # Embedding deps (FastEmbed, NumPy)
├── requirements-dev.txt        # Test deps
├── pytest.ini
└── .env.example                # Configuration template (placeholders only)
```
*(Vector index directories such as `vector_store/` and `vector_store_v2/` are
generated artifacts and are git-ignored.)*

## 6. Prerequisites

- Python **3.12**
- A **Groq API key** (for answer generation; the app still starts without it, but
  `/ask` will return an error until one is configured)
- Internet access on first run of the embedding model (FastEmbed downloads the
  ONNX model ~90 MB from the Hugging Face Hub, then caches it)

## 7. Virtual environment & dependencies

```bash
python3.12 -m venv .venv
source .venv/bin/activate

# Core API runtime:
pip install -r requirements.txt
# Embedding / vector-search backend (needed to build or query a real index):
pip install -r requirements-ml.txt
# Test dependencies:
pip install -r requirements-dev.txt
```

## 8. Environment configuration

Copy the template and fill in values (placeholders shown — never commit real keys):

```bash
cp .env.example .env
```

Key settings (see `.env.example` / `app/config.py` for the full list and defaults):

| Variable | Default | Purpose |
|---|---|---|
| `DOCUMENTS_DIR` | `documents/policies` | Source Markdown documents |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | `800` / `100` | Chunking (characters) |
| `EMBEDDING_MODEL` | `sentence-transformers/all-MiniLM-L6-v2` | FastEmbed model |
| `EMBEDDING_DIM` | `384` | Must match the model |
| `INDEX_DIR` | `vector_store` | Vector index directory |
| `TOP_K` | `5` | Chunks retrieved per query |
| `ANSWER_MIN_SCORE` | `0.25` | Refusal score floor (cosine) |
| `GROQ_API_KEY` | *(unset)* | Groq API key (required to answer) |
| `GROQ_MODEL` | `llama-3.3-70b-versatile` | Groq chat model id |
| `GROQ_MAX_TOKENS` | `1024` | Max LLM output tokens |
| `GROQ_TIMEOUT_SECONDS` | `30.0` | Groq request timeout |
| `API_KEY` | *(unset)* | Expected `X-API-Key` value (fails closed if unset) |
| `RATE_LIMIT_ENABLED` | `true` | Toggle rate limiting on `/ask` |
| `RATE_LIMIT_REQUESTS` | `10` | Requests per window, per API key |
| `RATE_LIMIT_WINDOW_SECONDS` | `60` | Rate-limit window |
| `MAX_REQUEST_BYTES` | `16384` | Max request body (413 if exceeded) |
| `MAX_QUESTION_CHARS` | `2000` | Max question length (422 if exceeded) |
| `ROOT_PATH` | *(empty)* | Sub-path when behind a reverse proxy |

> `INDEX_DIR` controls which vector index the service loads. The optimized
> index built by `app.retrieval.augment` is written to `vector_store_v2`; point
> `INDEX_DIR=vector_store_v2` to serve it.

Generate a strong `API_KEY`:
```bash
python -c "import secrets; print(secrets.token_urlsafe(32))"
```

## 9. Building / using the vector index

Requires `requirements-ml.txt` installed (FastEmbed).

```bash
# Build the baseline index into INDEX_DIR (default: vector_store/):
python -m app.services.indexing build

# Build the optimized index (heading/title-aware embedding + boilerplate
# down-weighting) into vector_store_v2/:
python -m app.retrieval.augment

# Ad-hoc retrieval check (no LLM, no auth):
python -m app.services.indexing search "how many annual leave days do I get?" -k 5
```

## 10. Starting the backend

```bash
python -m uvicorn app.main:app --host 0.0.0.0 --port 46453
```

Then open the interactive docs at `http://127.0.0.1:46453/docs`.

## 11. API endpoints & examples

| Method | Path | Auth | Description |
|---|---|---|---|
| `GET` | `/health` | none | Liveness probe |
| `POST` | `/ask` | `X-API-Key` | Answer a policy question with citations |

**Health:**
```bash
curl http://127.0.0.1:46453/health
# {"status":"ok","service":"Internal Policy Q&A Service","version":"0.1.0"}
```

**Ask (authenticated):**
```bash
curl -X POST http://127.0.0.1:46453/ask \
  -H "Content-Type: application/json" \
  -H "X-API-Key: <your-api-key>" \
  -d '{"question": "How many annual leave days do employees get?"}'
```
Example response (shape):
```json
{
  "question": "How many annual leave days do employees get?",
  "answer": "Full-time employees receive 25 days ... (leave_policy.md — 3. Annual Leave Entitlement).",
  "refused": false,
  "citations": [
    {
      "source_filename": "leave_policy.md",
      "policy_id": "POL-HR-001",
      "document_title": "Leave Policy",
      "section_heading": "3. Annual Leave Entitlement",
      "chunk_index": 3,
      "score": 0.65
    }
  ]
}
```
Request body fields: `question` (required, non-blank, ≤ `MAX_QUESTION_CHARS`) and
optional `top_k` (1–20; defaults to `TOP_K`).

**Status codes:** `200` OK · `401` missing/invalid API key · `413` body too large ·
`422` invalid/blank/too-long question or out-of-range `top_k` · `429` rate limit
exceeded · `502` embedding/LLM backend error · `503` backend not ready or auth not
configured · `504` Groq request timed out.

## 12. Authentication (X-API-Key)

- All `/ask` requests require the header `X-API-Key: <value>` matching `API_KEY`.
- Validation is constant-time; a missing or wrong key returns **401**.
- The service **fails closed**: if `API_KEY` is not configured, protected requests
  are rejected (**503**) rather than silently allowed.
- `/health` is public.
- In Swagger UI (`/docs`): click **Authorize**, enter the key under
  **APIKeyHeader (X-API-Key)**, then call `POST /ask`.


## 13. Rate limiting, request-size & timeout behavior

- **Rate limiting:** `POST /ask` is limited to `RATE_LIMIT_REQUESTS` per
  `RATE_LIMIT_WINDOW_SECONDS` **per API key** (default 10 / 60s). Exceeding it
  returns **429** with a `Retry-After` header. Authentication runs *before* the
  limit is consumed, so failed-auth requests do not use quota. The limiter is
  **in-memory and per process** — with multiple workers/instances each keeps its
  own counters, so use a shared store (e.g. Redis) for a true cluster-wide limit.
- **Request-size:** bodies larger than `MAX_REQUEST_BYTES` (default 16 KB) are
  rejected with **413**; both the `Content-Length` header and the actual body
  length are checked. Questions longer than `MAX_QUESTION_CHARS` return **422**.
- **Groq timeout:** Groq calls use `GROQ_TIMEOUT_SECONDS` (default 30 s). A timeout
  is handled gracefully and returned as **504**; internal details and the API key
  are never exposed to clients.

## 14. Logging & request-ID tracing

- Centralized logging (`app/logging_config.py`) writes to **stderr** with the
  format: `TIMESTAMP LEVEL logger.name [request_id] message`.
- Every HTTP request is assigned a unique request id (or reuses an incoming
  `X-Request-ID` header) and returns it in the `X-Request-ID` response header, so
  all events for one request (auth → retrieval → Groq → completion) share the id.
- Logs record method, path, status, duration, retrieval chunk count and top score,
  Groq success/failure + duration, refusals, auth failures, and rate-limit hits.
  **Secrets, headers and full question text are never logged.**

## 15. Running the tests

The suite is fully offline (fake embedder + fake LLM; no network, no Groq, no
model download).

```bash
pytest
# or
python -m pytest -v
```

## 16. Evaluation

```bash
# Retrieval quality — offline, no Groq (requires requirements-ml.txt):
python -m app.evaluation.retrieval_eval --k 1 3 5
# Evaluate a specific index (e.g. the optimized one):
python -m app.evaluation.retrieval_eval --index-dir vector_store_v2

# Answer quality — makes ~1 live Groq call per answerable item (prompts first):
python -m app.evaluation.answer_eval          # add --yes to skip the prompt
```
Both write timestamped JSON + Markdown reports to `evaluation/reports/`. The
labeled dataset is `evaluation/dataset.jsonl` (18 questions: 15 answerable across
all 5 policies + 3 out-of-domain).

## 17. Evaluation results & known limitations

**Retrieval** (15 answerable questions, `all-MiniLM-L6-v2`, floor 0.25):

| Index | Hit@1 | Hit@3 | Hit@5 | MRR@5 |
|---|---|---|---|---|
| `vector_store` (baseline) | 0.800 | 1.000 | 1.000 | 0.889 |
| `vector_store_v2` (optimized, in production) | 0.933 | 1.000 | 1.000 | 0.967 |

**Answer quality** (`vector_store_v2`, model `openai/gpt-oss-120b`, 18 questions):

| Metric | Value |
|---|---|
| Citation accuracy (answerable) | 1.00 |
| Refusal accuracy (all) | 1.00 |
| Keyword pass rate (answerable) | 1.00 |

**Test suite:** 131 tests pass (offline).

**Known limitations / risks:**
- **Rate limiting is in-memory, per process** — not shared across workers/instances.
- **Request-size:** FastAPI reads the body to parse JSON before the size check, so
  a hard cap at the proxy/uvicorn layer is recommended for true memory protection.
- **Small evaluation set (18 questions)** — good signal, not statistically large;
  keyword checks are substring-based (a sanity signal, not strict grading).
- **Embedding model** is the compact `all-MiniLM-L6-v2`; a stronger model or a
  cross-encoder reranker could further improve Hit@1 (not implemented).
- **Single-process vector search** (exact NumPy) is appropriate for this corpus
  size (tens of chunks); it is not a distributed vector database.
- The evaluation numbers above reflect the committed dataset and the current
  policy documents; re-run the evaluation after any content change.
