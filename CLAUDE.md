# CLAUDE.md — Internal Policy Q&A Service

Guidance for future Claude Code sessions working in this repository. Treat the
**codebase as the source of truth**; see `README.md` for full setup/usage. This
file is the quick orientation + the rules that are easy to get wrong.

## 1. Project overview
A Retrieval-Augmented Generation (RAG) API that answers employee questions about
internal company policies. It retrieves the most relevant passages from Markdown
policy documents and uses a Groq-hosted LLM to produce a **grounded** answer with
**citations**, and **refuses** when the policies lack sufficient information.
Core value: accurate, cited, non-hallucinated policy answers.

## 2. Technology stack
- **Python 3.12**
- **FastAPI** + **Uvicorn** (HTTP API)
- **Pydantic v2** / **pydantic-settings** (schemas + config)
- **FastEmbed** (ONNX) with `sentence-transformers/all-MiniLM-L6-v2` (384-dim) for
  embeddings; **NumPy** exact cosine search (no external vector DB)
- **Groq** Python SDK for answer generation. Config default model is
  `llama-3.3-70b-versatile` (`GROQ_MODEL`); the current deployment overrides it to
  `openai/gpt-oss-120b` via `.env`.
- **pytest** for tests (offline: fake embedder + fake LLM, no network/Groq)

## 3. Architecture

```
app/
  main.py              # FastAPI app, lifespan (loads index + LLM + rate limiter), request-id middleware
  config.py            # Single Settings object (env / .env); import `settings` everywhere
  logging_config.py    # Centralized logging + request_id contextvar/filter
  api/
    routes.py          # POST /ask; dependency order: require_api_key -> enforce_body_size -> enforce_rate_limit
    auth.py            # X-API-Key auth (APIKeyHeader, constant-time compare, fail-closed)
    limits.py          # RateLimiter + enforce_rate_limit (429) + enforce_body_size (413)
  ingestion/           # loader -> parser -> chunker -> pipeline (Markdown -> DocumentChunk list)
  retrieval/
    embedder.py        # BaseEmbedder + FastEmbedEmbedder (lazy import)
    vector_store.py    # VectorStore: build/search/save/load; cosine; optional per-chunk penalties
    retriever.py       # PolicyRetriever (thin wrapper over a loaded VectorStore)
    augment.py         # Heading/title-aware embedding + boilerplate penalty; builds vector_store_v2
  llm/
    base.py            # BaseLLM, LLMError, LLMTimeoutError
    groq_client.py     # GroqLLM.generate() (timeout, max_tokens, success/failure logging)
    prompt.py          # build_messages() + REFUSAL_SENTINEL = "INSUFFICIENT_CONTEXT"
  services/
    indexing.py        # build_and_persist_index / load_index + `build`/`search` CLI
    qa.py              # answer_question(): orchestrates retrieve -> refuse-or-generate
  schemas/             # ingestion.py, retrieval.py, answer.py (Pydantic models)
  evaluation/          # dataset.py, metrics.py, retrieval_eval.py, answer_eval.py, schemas.py
documents/policies/    # 5 Markdown policy docs (source of truth for answers)
evaluation/dataset.jsonl   # labeled eval set (tracked); reports/ is generated (git-ignored)
tests/                 # offline test suite
```

**Request flow (`POST /ask`):** middleware assigns a request id → `require_api_key`
(401) → `enforce_body_size` (413) → `enforce_rate_limit` (429) → `AskRequest`
validation (422) → `ask()` calls `answer_question(question, retriever, llm, top_k)`
→ retriever embeds query + cosine search → refuse if below floor (no LLM) → else
`GroqLLM.generate()` → `AnswerResponse`. Retriever and LLM are built once in the
lifespan and read from `app.state` via `get_retriever` / `get_llm` (override these
in tests).

## 4. RAG pipeline
1. **Ingestion** (`app/ingestion`): load Markdown → parse into sections by heading →
   chunk (default `CHUNK_SIZE=800`, `CHUNK_OVERLAP=100` chars), preserving
   `source_filename`, `policy_id`, `document_title`, `section_heading`, `chunk_index`.
2. **Embeddings** (`FastEmbedEmbedder`): 384-dim vectors. The production/optimized
   index (`augment.py`) prepends `document_title > section_heading` to each chunk's
   embedding text and applies a small score penalty to boilerplate chunks
   (title/preamble and "Related Policies").
3. **Vector store** (`vector_store.py`): L2-normalized vectors saved as `vectors.npy`
   + `metadata.json` sidecar; search is `vectors @ query - penalties`, top-k by score.
4. **Retrieval** (`PolicyRetriever.retrieve`) returns `SearchResult{chunk, score}`.
5. **Generation** (`qa.answer_question`): if no results or top score
   `< ANSWER_MIN_SCORE` (0.25) → refuse without calling the LLM. Otherwise build the
   prompt and call Groq. If the model reply starts with `INSUFFICIENT_CONTEXT` →
   refuse. Else return the answer with citations from the retrieved chunks.
6. **Citations**: `Citation.from_search_result` projects chunk metadata
   (filename, section heading, policy_id, chunk_index, score). Refusals return a
   standard message with `refused=true` and no citations.

## 5. API endpoints
- `GET /health` — public liveness probe: `{status, service, version}`.
- `POST /ask` — **requires `X-API-Key`**. Body: `{"question": str, "top_k": int?}`
  (`question` non-blank and ≤ `MAX_QUESTION_CHARS`; `top_k` 1–20, defaults to `TOP_K`).
  Response: `AnswerResponse{question, answer, refused, citations[]}`.
- Status codes: `200` ok · `401` missing/invalid key · `413` body too large ·
  `422` invalid/blank/too-long question or bad `top_k` · `429` rate limited ·
  `502` embedding/LLM error · `503` backend not ready / auth not configured ·
  `504` Groq timeout.

## 6. Security & reliability
- **API key** (`auth.py`): `X-API-Key` header, constant-time `hmac.compare_digest`,
  **fails closed** (503) if `API_KEY` is unset. `/health` is public.
- **Rate limiting** (`limits.py`): per-API-key fixed window (`RATE_LIMIT_REQUESTS=10`
  / `RATE_LIMIT_WINDOW_SECONDS=60`), **in-memory & per process**; runs after auth so
  failed-auth requests don't consume quota; 429 + `Retry-After`.
- **Request limits**: body > `MAX_REQUEST_BYTES` (16 KB) → 413 (checks Content-Length
  *and* actual body length); question > `MAX_QUESTION_CHARS` (2000) → 422.
- **LLM bounds**: `GROQ_MAX_TOKENS` (1024) caps output; `GROQ_TIMEOUT_SECONDS` (30s)
  bounds the call; timeouts map to `LLMTimeoutError` → 504.
- **Logging**: centralized to stderr as `TIME LEVEL logger [request_id] message`.
  Never log the API/Groq keys, headers, or full question text — only metadata
  (method, path, status, duration, chunk count, top score, error type).

## 7. Development commands
```bash
source .venv/bin/activate                         # or prefix with .venv/bin/
pip install -r requirements.txt                   # core API
pip install -r requirements-ml.txt                # FastEmbed/NumPy (build/query real index)
pip install -r requirements-dev.txt               # tests (pytest, httpx)

python -m uvicorn app.main:app --host 0.0.0.0 --port 46453   # run server (docs at /docs)
pytest                                             # full offline suite

python -m app.services.indexing build              # build baseline index into INDEX_DIR
python -m app.retrieval.augment                    # build optimized index -> vector_store_v2/
python -m app.services.indexing search "..." -k 5  # ad-hoc retrieval (no LLM/auth)

python -m app.evaluation.retrieval_eval --k 1 3 5 [--index-dir vector_store_v2]   # offline, no Groq
python -m app.evaluation.answer_eval [--yes]       # LIVE Groq (~1 call per answerable item)
```

## 8. Coding guidelines
- Follow the existing layering: **API routes stay thin**; business logic lives in
  `services`/`retrieval`/`llm`. Inject dependencies (retriever, LLM, embedder) so
  they can be faked in tests — don't instantiate Groq/FastEmbed inside routes.
- Use **Pydantic models** (`app/schemas`) for request/response contracts and
  validation; read config only through `app.config.settings`.
- Keep modules importable without heavy side effects (FastEmbed and Groq are
  imported lazily; preserve that).
- **Add offline tests** for new behavior using the fakes in `tests/conftest.py`
  (`FakeEmbedder`, `FakeLLM`, `StubRetriever`, etc.); keep the suite network-free.
- Avoid new dependencies and overengineering; prefer extending existing patterns.
- Preserve centralized logging + request-id tracing and the secret-safety rules.

## 9. Important constraints
- **Grounding**: answers must come only from the retrieved policy context. Never
  invent policy details; keep the refusal paths (score floor + sentinel) intact.
- **Refusal**: insufficient/irrelevant evidence → refuse with the standard message
  and no citations.
- **Citations**: keep them accurate and derived from the actual retrieved chunks.
- **Secrets**: never print/log/commit `API_KEY` or `GROQ_API_KEY`; `.env` is
  git-ignored and must stay untracked (`.env.example` holds placeholders only).
- **Do not** modify policy documents or regenerate/replace the vector index without
  a clear reason; build candidate indexes to a **separate directory** and compare
  via evaluation before promoting (how `vector_store_v2` was adopted).
- **Do not** change RAG logic, auth, logging, or abuse protections without cause.

## 10. Known limitations (confirmed in code/README)
- Rate limiting is **in-memory, per process** — not shared across workers/instances
  (would need e.g. Redis for a true cluster-wide limit).
- FastAPI reads the request body to parse JSON before the size check, so a hard cap
  at the proxy/uvicorn layer is recommended for true memory protection.
- Evaluation set is small (**18 questions**: 15 answerable + 3 out-of-domain) and
  answer-eval keyword checks are substring-based (a sanity signal, not strict grading).
- Embedding model is the compact `all-MiniLM-L6-v2`; no reranker (a stronger model
  or cross-encoder could further improve Hit@1).
- Vector search is single-process exact NumPy — appropriate for this corpus
  (tens of chunks), not a distributed vector database.
- Latest measured results (see README): retrieval Hit@1 0.933 / MRR@5 0.967 on
  `vector_store_v2`; answer citation/refusal/keyword accuracy 1.00; 131 tests pass.
