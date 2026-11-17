---
name: Embedding decisions
description: Decisions about embeddings: gemini-embedding-2 at 768 dims with text role prefixes instead of task types (supersedes the design's gemini-embedding-001); the google-genai client in llm/ (one Content per text, retries only on 429/5xx, no retry of a rejected answer, rate limiter, cumulative spend ledger, one generation per request); the spend cap now $60 total. S9's own rules are in ingestion.md (B4). Open: lower the cap for production? 768 vs 1,536 dims?
source: runs/2026-10-14-ingestion
---

# Embedding decisions

## Open questions (OPEN: waiting for the user)
- **Spend cap (step C (3), B4 (3)):** `EMBED_SPEND_CAP_USD` is $60 total across runs
  (ledger $33.36 on this machine after B4). A lower cap would block dev embeds; on the
  grader the ledger starts at 0 and one index costs ~$6. Lower it for production?
- **Dimensions (step C (4)):** 768 vs 1,536 dims was never compared on the tune set.

## 2026-10-16 C2: embedding client in llm/ (new dependency google-genai)
- **Area:** llm / embeddings; dependencies
- **Superseded in part by:** retrieval.md#2026-10-16 C4 (item 6's cap $25 became $60, kept by ingestion.md#2026-10-16 B4 item 9).
- **Chose:**
  1. **New runtime dependency `google-genai` (2.31.0)**, added with `uv add`. Google's official SDK for the Gemini API; needed for embeddings now and generation later. Replaces nothing. It brings google-auth, cryptography, tenacity, websockets.
  2. SDK call: `client.models.embed_content(model="gemini-embedding-2", contents=[Content(parts=[Part(text=t)]) for t in texts], config=EmbedContentConfig(output_dimensionality=768))`, which POSTs `models/gemini-embedding-2:batchEmbedContents` with one `requests[]` entry per Content. Verified by reading the installed SDK (`_transformers.t_contents` merges a list of plain strings into ONE Content) and by an offline test that drives the real SDK over an httpx mock transport and checks the request body.
  3. Role prefixes (ai.google.dev embeddings page, 2026-10-16), in `prompts/embed_roles.txt` version 1: query `task: question answering | query: {text}`; document `title: none | text: {text}` (the chunk header already carries the title inside the text). Prompt file format: `key: value` lines with `version:` first, read by `corpusqa.prompts.load_prompt`; no YAML dependency.
  4. Modules: `llm/client.py` (GeminiEmbedder, `checked_vectors`), `llm/limits.py` (RateLimiter, SpendLedger), `llm/errors.py`, `llm/fakes.py` (FakeEmbedder = hashed bag of words, so texts sharing words are close and retrieval tests behave sensibly). The `Embedder` protocol later moved to `llm/interfaces.py` (retrieval.md C4 item 8).
  5. Retries only on 429 and 5xx, backoff 1, 2, 4... s capped at 60, no jitter (the rate limiter spaces every attempt). A validation failure is NOT retried: an aggregated or malformed answer would repeat and cost again. Transport errors and timeouts are not retried either (S9 records failed requests).
  6. Spend: cap `EMBED_SPEND_CAP_USD` = $25; ledger `.cache/spend/embeddings.json`, cumulative across runs; checked before each request with the chars/4 estimate, charged after each answer (also when the answer is unusable: it is billed). An unreadable ledger refuses rather than starting from zero.
  7. Tracing: one generation `embed` per request; input = role + texts exactly as sent (AGENTS rule; a 100-chunk request is ~200 KB); output = vector count and dims (the vectors themselves are not traced). A call refused by the cap makes no generation; the error reaches the caller's span.
  8. `EMBED_REQUESTS_PER_MINUTE` = 120 per process (later measured: this limiter, not the API quota, sets the pace). `genai.Client(vertexai=False, ...)` so environment variables cannot switch it to Vertex. Key only from `Settings.gemini_api_key`; none -> `EmbedderUnavailable`.
- **Over:** the SDK's own retry (cannot report retries to the trace), task_type (not supported by the model), plain string lists (aggregated into one vector).
- **Why:** brief for C2; facts from the installed SDK source and ai.google.dev, read 2026-10-16.

## 2026-10-16 Embedding model: gemini-embedding-2 (user decision; supersedes decision 3)
- **Area:** ingestion / embeddings
- **Chose:** `gemini-embedding-2`, 768 dims, through `llm.client`. No `task_type`: the query/document role goes in a short text prefix. Each input in its own Content (otherwise inputs are aggregated into one vector); a test checks vectors returned == inputs sent. Auto-normalised below 3,072 dims (we still L2-normalise; harmless).
- **Over:** `gemini-embedding-001` (2,048-token input, no longer on the pricing page; the two vector spaces are incompatible), a local open-source model (PyTorch download counts against start-up, CPU speed, likely weaker).
- **Why:** user: "yes switch to that". 8,192-token input makes whole-document embedding testable; $0.20 per 1M tokens ($0.10 batch). Facts from ai.google.dev embeddings + pricing pages, read 2026-10-16.
