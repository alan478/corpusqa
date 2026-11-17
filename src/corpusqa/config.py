"""Settings from the environment, plus every tunable constant (no magic numbers elsewhere).

CONTRACT.md fixes the environment: only CORPUS_DIR, PORT and GEMINI_API_KEY are ours
to read. The optional LANGFUSE_* variables are a dev convenience: the grader never
sets them, and without them tracing is simply off. Everything else that shapes
behaviour (deadlines, size caps, fixed answer texts) is a named constant here, so a
reviewer finds every knob in one file.
"""

import os
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path

# ---------------------------------------------------------------- HTTP server
# CONTRACT.md allows 127.0.0.1 or 0.0.0.0; all interfaces also covers containers.
HOST = "0.0.0.0"  # noqa: S104
ASK_PATH = "/ask"
HEALTH_PATH = "/health"
# Every /ask response carries its trace id (the Langfuse trace id when tracing is on) in
# this header. The body is exactly CONTRACT §3's `answer` + `document_ids` (user decision
# 2026-10-17); eval/run.py reads the trace id from here.
TRACE_ID_HEADER = "X-Trace-Id"
# Server-side cut for one /ask (CONTRACT: 180 s to the last byte). The 30 s margin
# covers queueing, serialisation and network. The agent must finish well before it.
ASK_DEADLINE_S = 150.0
# CONTRACT §4: up to 4 /ask requests may be in flight at once. At most this many answers
# run at once; a request that arrives while they all run waits in line for a slot (never
# refused), and that wait counts against its own ASK_DEADLINE_S, counted from arrival.
ASK_MAX_CONCURRENT = 4
# On SIGTERM uvicorn waits for in-flight requests; the grader sends SIGKILL after 10 s.
SHUTDOWN_GRACE_S = 5  # int: uvicorn's timeout_graceful_shutdown type

# ---------------------------------------------------------------- response size
# CONTRACT: the response body must stay under 1 MiB. One char of a JSON string can
# take up to 6 bytes (a \u00XX escape), so 64K chars is at most 384 KiB of answer.
MAX_RESPONSE_BYTES = 1024 * 1024
MAX_ANSWER_CHARS = 64_000
# A citation list this long is already useless; the cap only bounds the body size.
MAX_DOCUMENT_IDS = 500
ANSWER_TRUNCATED_SUFFIX = "\n\n[Answer truncated: it exceeded the response size limit.]"

# ---------------------------------------------------------------- fixed answer texts
# None of these may claim the corpus lacks the answer: only the answering pipeline,
# after looking, may say "not found" (AGENTS.md hard rules).
INVALID_REQUEST_ANSWER = (
    'Invalid request: send JSON of the form {"question": "<non-empty text>"} '
    "with Content-Type: application/json."
)
NOT_READY_ANSWER = (
    "The service is still starting (ingesting the corpus), so this question was not "
    "answered. Retry once GET /health returns 200."
)
TIMEOUT_ANSWER = (
    "Internal error: answering this question took longer than the time limit, so no "
    "answer was produced. This does not mean the documents lack the answer."
)
INTERNAL_ERROR_ANSWER = (
    "Internal error: the service failed while answering this question, so no answer "
    "was produced. This does not mean the documents lack the answer."
)
PLACEHOLDER_ANSWER = (
    "Answering is not implemented yet: this server runs a placeholder answerer that "
    "does not read the corpus."
)

# ---------------------------------------------------------------- tracing (Langfuse)
# Used when LANGFUSE_BASE_URL / LANGFUSE_HOST is not set. The project's Langfuse Cloud
# project is in the US region (user decision, 2026-10-16); the grader sets no keys, so
# this is dev only.
LANGFUSE_DEFAULT_HOST = "https://us.cloud.langfuse.com"
# Seconds one background export request may take (the SDK takes an int).
LANGFUSE_EXPORT_TIMEOUT_S = 5
# Lifespan exit waits at most this long for queued spans; it must fit well inside
# SHUTDOWN_GRACE_S and the grader's 10 s SIGTERM -> SIGKILL window.
TRACE_SHUTDOWN_TIMEOUT_S = 2.0
# Trace names, so tools that read traces and the server agree.
ASK_TRACE_NAME = "ask"
STARTUP_TRACE_NAME = "startup"
# Values the scrubber walks deeper than this are replaced by a marker (cycle guard).
TRACE_SCRUB_MAX_DEPTH = 32
# An error message on an observation (e.g. an exception's text) is cut to this length.
TRACE_STATUS_MAX_CHARS = 2000
# A span lists at most this many items (e.g. files set aside); the rest is counted only.
# The full lists live in the index database.
TRACE_MAX_LISTED_ITEMS = 100

# ---------------------------------------------------------------- ingest (design.md §9)
# The three keys of the shared document format (CONTRACT.md §1; design.md §2). They are
# the format, not dataset field names: every corpus of this format carries them.
UUID_KEY = "dataset_doc_uuid"
TITLE_KEY = "title_field_name"
CONTENT_KEY = "content_field_names"
CONTRACT_KEYS = frozenset({UUID_KEY, TITLE_KEY, CONTENT_KEY})

# Walk (S1). Only files with this suffix are documents.
DOC_SUFFIX = ".json"
# Source name for a file directly under CORPUS_DIR (no source folder).
ROOT_SOURCE = "_root"
# A file above this is set aside; the largest file on data/ is 36 KB.
MAX_DOC_BYTES = 16 * 1024 * 1024
# A symlink is followed only when its target is inside CORPUS_DIR (never outside).
FOLLOW_SYMLINKS_INSIDE_CORPUS = True

# Parse (S2). Ids are accepted in any format up to this length (only presence is promised).
UUID_MAX_CHARS = 256
# More keys than this only logs a warning (data/ has a few dozen at most).
MAX_FIELDS_PER_DOC = 2_000

# Title (S3). Rules b-d of the weak-title test; rule a (shared titles) is a corpus-wide
# statistic and runs in the profile stage.
WEAK_TITLE_MIN_CHARS = 3
# A list-valued title is joined with this.
TITLE_LIST_SEPARATOR = " / "
# A weak title is shown as "<title><separator><humanised file stem>".
DISPLAY_TITLE_SEPARATOR = " — "

# Body (S3). Promotion of unlisted prose fields is open decision 1 (reopened for the
# chunking step); the flag lets the eval compare both settings without code changes.
PROMOTE_PROSE_FIELDS = True
# A field is prose when its text has at least this many characters and words.
PROSE_MIN_CHARS = 150
PROSE_MIN_WORDS = 15
# ...and it is not link-shaped: a field is link-shaped when at least this share of its
# items are URLs, paths or keys (a reference list, not prose).
LINK_SHAPED_MIN_SHARE = 0.5
# Duplicate check before promotion: word-shingle size and the Jaccard similarity at
# which a candidate counts as a copy of the body.
DUP_SHINGLE_WORDS = 5
DUP_JACCARD = 0.8
# Guard against pathological files; the body is cut at a unit boundary above this.
MAX_BODY_CHARS = 2_000_000
# Body section label line: "<prefix><field name>" (only when a body has 2+ labels).
BODY_LABEL_PREFIX = "## "

# Fields (S4). value_norm is cut to this many characters (an exact-filter key, not text).
VALUE_NORM_MAX_CHARS = 256
# A key with one of these prefixes and a short value is generator residue (noise).
NOISE_KEY_PREFIXES = ("_", "--")
NOISE_MAX_VALUE_CHARS = 40

# Dates (S4). Each date value is judged on its own (user, 2026-10-16): inside the window
# it is a usable date (a created/updated candidate, shown in the chunk header, matched by
# date filters); outside it counts as no date. Its field row keeps the value and
# `date_in_range` = 0. The fixed window [DATE_MIN_YEAR, this year + DATE_MAX_FUTURE_YEARS]
# is used per document and as the fallback of the learned window below.
DATE_MIN_YEAR = 1990
DATE_MAX_FUTURE_YEARS = 5
# Learned window: the 1st-99th percentile of the years of ISO-format values, widened by
# the margin. Epoch numbers are left out: generated epochs are often fake (on data/ chat
# timestamps span 2021-2286 while ISO dates sit in 2024-2029).
DATE_WINDOW_LOW_QUANTILE = 0.01
DATE_WINDOW_HIGH_QUANTILE = 0.99
DATE_WINDOW_MARGIN_YEARS = 1
# Below this many ISO values the percentiles are just the extremes: use the fixed window.
DATE_WINDOW_MIN_VALUES = 100
# English word stems that vote for a date field's role (open decision 2). Matched as the
# start of a word in the field name, so "eta" does not fire inside "metadata". These are
# language heuristics, not field names: no whole name from data/ appears here.
CREATED_WORDS = ("creat", "start", "first", "record", "sent", "open")
UPDATED_WORDS = ("updat", "modif", "last", "edit", "close", "merge", "resolv")
# A planned date (due, target...) is neither created nor updated.
PLANNED_WORDS = ("due", "deadline", "forecast", "target", "sla", "eta")
# A stem shorter than this is an abbreviation and must be a whole word ("sla" is not
# the start of "slash"); longer stems match the start of a word ("creat" -> "created").
ROLE_WORD_MIN_PREFIX_CHARS = 4

# Store (S7). Documents per write transaction.
STORE_BATCH_DOCS = 500
# Start-up log: the file loop (S2-S5 and S7) prints "N of M files read" every this many files.
INGEST_PROGRESS_EVERY_FILES = 1_000
# A schema change bumps this; an index with another version is rebuilt, never migrated.
SCHEMA_VERSION = 5
# Caches live in the repo checkout (writable; CORPUS_DIR is read-only).
CACHE_DIR = Path(__file__).resolve().parents[2] / ".cache"
INDEX_DIR = CACHE_DIR / "index"
INDEX_DB_NAME = "corpus.sqlite"
# The build writes "<db><suffix>" and renames it when complete, so a crash never
# leaves a database that looks finished.
BUILDING_SUFFIX = ".building"

# Index cache (design §7). Each build lives in INDEX_DIR/<first FINGERPRINT_DIR_CHARS hex
# chars of its fingerprint>/. The fingerprint covers the corpus listing (path, size, mtime
# of every file), the code below (paths under src/corpusqa/), SCHEMA_VERSION and the build
# settings, so any change to one of them builds a new index.
FINGERPRINT_DIR_CHARS = 16
# The embedding role prompt shapes every stored vector (S9), so it is part of the index.
FINGERPRINT_CODE_PATHS = ("ingest", "store", "config.py", "prompts/embed_roles.txt")
# Builds of the same corpus kept on disk (the one in use included); older ones are deleted
# at start. Builds of a corpus folder that no longer exists are deleted too.
CACHE_KEEP = 2
# A folder holding no finished index and unchanged for this long is a crashed build: deleted
# at start. A build in progress writes every few seconds, so it is never this old.
CACHE_STALE_BUILD_S = 3_600

# ---------------------------------------------------------------- profile (S8, design §4)
# Kinds are decided from values alone; the field name is shown to the agent, never read.
# A field is scalar when at most this share of its documents hold a list.
KIND_SCALAR_MAX_LIST_RATIO = 0.05
# ...and a list field when at least this share do.
KIND_LIST_MIN_RATIO = 0.5
KIND_DATE_MIN_SHARE = 0.9
KIND_NUMBER_MIN_SHARE = 0.9
KIND_BOOLEAN_MAX_DISTINCT = 6
KIND_BOOLEAN_MIN_SHARE = 0.9
OWN_ID_MIN_PRESENCE = 0.95
OWN_ID_MIN_DISTINCT_RATIO = 0.9
OWN_ID_MIN_DIGIT_SHARE = 0.9
OWN_ID_MAX_MEDIAN_LEN = 40
# ...and one token (no spaces): unique, short titles that each hold a number pass every
# other own-id test.
OWN_ID_MIN_TOKEN_SHARE = 0.9
KIND_REFERENCE_MIN_SHARE = 0.5
KIND_URL_MIN_SHARE = 0.8
KIND_EMAIL_MIN_SHARE = 0.7
ENUM_MAX_DISTINCT = 50
ENUM_MIN_DOCS_PER_VALUE = 3
# People: "First Last" with a first name from the learned lexicon...
PERSON_LEXICON_MIN_SHARE = 0.6
# ...or name/handle-shaped values (2-4 capitalised words, or a handle) that also appear
# in another field. One capitalised word is not enough: tool, place and priority names
# look the same (on data/ a one-value field of a video tool's name passed).
PERSON_SHAPED_MIN_SHARE = 0.8
PERSON_REUSED_MIN_SHARE = 0.5
PERSON_MIN_TOKENS = 2
PERSON_MAX_TOKENS = 4
TAGS_MIN_SINGLE_TOKEN_SHARE = 0.8
TAGS_MIN_DOCS_PER_VALUE = 1.5
TAG_MAX_CHARS = 40
DATED_LOG_MIN_SHARE = 0.6
TEXT_MIN_SHARE = 0.5
# A value is long text above this many words or characters.
TEXT_MIN_WORDS = 6
TEXT_MIN_CHARS = 60
# A value longer than this is prose: it is never a reference, URL, email or name (skips
# those shape tests on long text).
SHAPE_MAX_CHARS = 500
# A label before a reference ("<label>: <target>") is at most this long.
REFERENCE_LABEL_MAX_CHARS = 30
# First-name lexicon: first tokens of two-token capitalised names, from fields where at
# least this share of values are such names, kept when they start this many distinct names.
LEXICON_FIELD_MIN_SHARE = 0.9
LEXICON_MIN_COUNT = 5
# Synonym groups: two fields of one source that never share a document and have the same
# kind are shown together when this share of the smaller one's distinct values also
# appear in the other (only for kinds whose values name things). People are left out:
# the same people fill many roles, so shared names do not mean a renamed field.
SYNONYM_MIN_VALUE_OVERLAP = 0.5
SYNONYM_VALUE_KINDS = frozenset({"enum", "tags", "reference", "url", "email"})
# Conditional presence: for enum fields with at most this many values, values held by at
# least CONDITIONAL_MIN_SUPPORT documents; rows at or above the strength are stored.
CONDITIONAL_MAX_VALUES = 20
CONDITIONAL_MIN_SUPPORT = 20
CONDITIONAL_MIN_STRENGTH = 0.8
# Weak titles, rule (a): a title field whose distinct-value ratio in its source is below
# this is weak for every document using it (many documents share each title).
WEAK_TITLE_DISTINCT_RATIO = 0.5
# Profile display limits.
PROFILE_TOP_VALUES = 20
PROFILE_TOP_VALUE_CHARS = 120
PROFILE_EXAMPLE_DOCS = 3
# A file stem matches the title when this share of the smaller word set is shared.
STEM_TITLE_MIN_OVERLAP = 0.5

# Keys and links (S8). A key held by more documents than this is a hub: it does not
# identify one document, so links through it stay unresolved (marked target_is_hub).
HUB_KEY_MAX_DOCS = 5

# ---------------------------------------------------------------- chunking (step C)
# Default chunk size of the recursive and fixed strategies (user decision 2026-10-16:
# recursive, 512 tokens, no overlap).
CHUNK_MAX_TOKENS = 512
# Default overlap of fixed-size chunks, as a share of their size (the benchmark's 10 %).
CHUNK_FIXED_OVERLAP_SHARE = 0.1
# No strategy makes a chunk larger than this: the model reads 8,192 tokens and chars / 4
# undercounts tokens for code, numbers and non-English text, so a quarter stays spare.
# It is also the `whole` strategy's limit (bigger documents fall back to recursive).
CHUNK_MAX_EMBED_TOKENS = 6_000
# A chunk header is "<source><sep><display title><sep><created date>".
CHUNK_HEADER_SEPARATOR = " · "
# S6 reads stored documents in pages of this many (rowid order) while it writes chunks.
CHUNK_READ_PAGE_DOCS = 500
# Field values in every chunk header, after the date (backlog item 3). On since the
# retrieval eval of 2026-10-16 (tune set, `--chunking recursive --fields` against plain):
# hybrid per-question recall@10 0.673 -> 0.727, MRR 0.588 -> 0.671. Changing it changes
# this file, so the next start rebuilds the index.
CHUNK_HEADER_FIELDS = True
# Which fields of a source go in (`ingest.chunking.select_header_fields`), all from what
# the profile (S8) learned: in at least this share of the source's documents (a field a
# few documents have says little about the rest)...
HEADER_FIELD_MIN_PRESENCE = 0.5
# ...rarely the title or body text (that text is in the header or the chunk already)...
HEADER_FIELD_MAX_TITLE_RATIO = 0.5
HEADER_FIELD_MAX_CONTENT_RATIO = 0.5
# ...and of a kind whose values say what a document is about. Dates are already in the
# header; own ids, URLs, references and emails are long tokens that mean little to a
# search; dated logs and unknown values are long or empty.
HEADER_FIELD_KINDS = frozenset({"enum", "person", "tags", "text", "mixed", "number", "boolean"})
# Size caps, so field values never drown the chunk text: the first items of a list, each
# value cut at this many characters, and the whole fields part cut at a part boundary at
# this many characters (about 100 tokens, the size contextual retrieval uses).
HEADER_FIELD_LIST_ITEMS = 5
HEADER_FIELD_VALUE_CHARS = 60
HEADER_FIELDS_MAX_CHARS = 400

# ---------------------------------------------------------------- start-up budget (design §7)
# CONTRACT §2: /health must say 200 within 4 h of launching ./serve. Ingest plans for 3 h,
# counted from the start of ingest.run (uv sync and imports before it fit in the spare
# hour). S9 starts no new embedding request after budget - reserve; the reserve covers the
# requests still in flight (each bounded by EMBED_TIMEOUT_MS and its retries) and loading
# the vectors.
STARTUP_BUDGET_S = 10_800.0
STARTUP_RESERVE_S = 600.0

# ---------------------------------------------------------------- embeddings (llm/)
# Model and vector size (decisions.md 2026-10-16; the model allows 128-3,072 dims).
EMBED_MODEL = "gemini-embedding-2"
EMBED_DIMS = 768
# The model reads at most this many tokens of one text and silently drops the rest.
EMBED_MAX_INPUT_TOKENS = 8_192
# Characters per token: the size estimate for chunks, and for spend when the API gives none.
TOKEN_CHARS = 4
# Texts per request (batchEmbedContents takes at most 100).
EMBED_BATCH_SIZE = 100
# Requests per minute from one process, spaced evenly; 429 answers back off on top of this.
EMBED_REQUESTS_PER_MINUTE = 120
# Retries of one request after a 429 or 5xx, waiting base * 2^attempt seconds (capped).
EMBED_MAX_RETRIES = 5
EMBED_BACKOFF_BASE_S = 1.0
EMBED_BACKOFF_MAX_S = 60.0
# One HTTP request may take this long (the SDK takes milliseconds).
EMBED_TIMEOUT_MS = 60_000
# Standard price per million input tokens (ai.google.dev pricing, read 2026-10-16).
EMBED_PRICE_PER_M_TOKENS_USD = 0.20
# Hard cap on recorded embedding spend, kept across runs in the ledger file; a call that
# would pass it is refused. A full index of data/ is about $6. Raised from $25 to $60 for
# the chunking eval (step C: 7 variants, ~$33 if every variant embeds the whole corpus;
# decisions.md "C4"). Delete the file to reset.
EMBED_SPEND_CAP_USD = 60.0
EMBED_SPEND_LEDGER = CACHE_DIR / "spend" / "embeddings.json"
# Prompt file (src/corpusqa/prompts/) with the role prefixes: the model takes no task_type.
EMBED_PROMPT_NAME = "embed_roles"
# S9 at start-up: requests in flight at once (each EMBED_BATCH_SIZE texts); the client's
# rate limiter still caps requests per minute (4 in flight at 120/min measured ~4.9M
# tokens/min on data/, learnings.md).
EMBED_CONCURRENCY = 4
# S9 pauses after this many failed requests in a row (each already retried up to
# EMBED_MAX_RETRIES times): the API is down or the quota is spent...
EMBED_MAX_FAILED_IN_A_ROW = 8
# ...then S9 lets the requests in flight finish, waits this long (a per-minute quota
# refills) and goes on, at most EMBED_MAX_COOLDOWNS times per start and never past the
# deadline. A dead API costs a bounded number of rounds, not the budget; a throttled one
# recovers.
EMBED_COOLDOWN_S = 60.0
EMBED_MAX_COOLDOWNS = 5
# Chunks S9 reads per page when it looks them up in the embedding cache.
EMBED_CACHE_PAGE_CHUNKS = 2_000
# Embedding cache: one vector per exact embedded text, keyed by sha256 of model, dims,
# role, prompt version and text. It lives outside the fingerprinted index, so a rebuild
# (code change) pays only for chunk text it has never embedded. Shared with the eval.
EMBED_CACHE_DB = CACHE_DIR / "embeddings" / "vectors.sqlite"
# How far from 1 a stored vector's length may be before a cached vector counts as a miss
# (float32 rounding is ~1e-6 at 3,072 dims; every vector in the data/ cache is within 1e-6).
UNIT_NORM_TOLERANCE = 1e-3

# ---------------------------------------------------------------- chat model (llm/, agent/)
# The model for every agent step and for compaction. User approved 2026-10-16 ("okay move to
# 3") after the step 1 probe (agent-tools run, decisions.md): the newest Flash model, it
# passed every harness check, 1-2 s per tool step. Changing the model is this one constant.
AGENT_MODEL = "gemini-3.8-flash"
# Thinking tokens count toward max_output_tokens (step 1 probe: a limit of 128 gave 118
# thinking + 6 answer tokens and stopped with MAX_TOKENS). So the output cap is the
# thinking budget plus room for the longest answer.
# Budget: the probe's prompts used 0-440 thinking tokens, at ~150 thinking tokens/s; 1,024
# costs at most ~7 s per step (~70 s over 10 steps), inside the 150 s deadline. The eval
# tunes it once real tools exist.
AGENT_THINKING_BUDGET = 1_024
# Room for the longest answer the response can carry (MAX_ANSWER_CHARS at TOKEN_CHARS per
# token). A cap, not a target: a normal step writes a few hundred tokens.
AGENT_ANSWER_MAX_TOKENS = MAX_ANSWER_CHARS // TOKEN_CHARS
AGENT_MAX_OUTPUT_TOKENS = AGENT_THINKING_BUDGET + AGENT_ANSWER_MAX_TOKENS
# USD per 1M tokens of AGENT_MODEL: paid tier, Standard, prompts <= 200k tokens, from
# https://ai.google.dev/gemini-api/docs/pricing (read 2026-10-16; the page says "Last
# updated 2026-10-15 UTC"). Output includes thinking tokens. These hold through
# 2026-12-31; from 2027-01-01 they double (1.50 / 0.15 / 7.50): update them then, or the
# spend ledger counts half the real cost.
CHAT_PRICE_INPUT_PER_M_TOKENS_USD = 0.75
CHAT_PRICE_CACHED_INPUT_PER_M_TOKENS_USD = 0.075
CHAT_PRICE_OUTPUT_PER_M_TOKENS_USD = 3.75
# Requests per minute from one process, spaced evenly. Not measured (the probe did not test
# the key's quota): a start value above the peak of 4 requests at once at ~1.5 s per call
# (~160/min), so pacing never slows a normal run; 429 answers back off on top of this.
CHAT_REQUESTS_PER_MINUTE = 240
# Retries of one call after a 429 or 5xx, waiting base * 2^attempt seconds (capped). Every
# retry happens inside the caller's timeout_s. The SDK itself does not retry (no
# retry_options set; step 1 probe).
CHAT_MAX_RETRIES = 5
CHAT_BACKOFF_BASE_S = 1.0
CHAT_BACKOFF_MAX_S = 20.0
# One HTTP request may take this long (the SDK takes milliseconds). A backstop only: the
# caller's timeout_s (asyncio.timeout) is the real limit and is always shorter.
CHAT_HTTP_TIMEOUT_MS = 120_000
# A model turn that code writes (no provider reply to re-send) needs a thought signature on
# its first function call, or the API answers 400 (step 1 probe). This stand-in value is
# the one https://ai.google.dev/gemini-api/docs/thought-signatures names for that case; the
# probe showed the API accepts it.
CHAT_STAND_IN_THOUGHT_SIGNATURE = b"skip_thought_signature_validator"
# Hard cap on recorded generation spend, kept across runs in its own ledger file (user
# approved $40, 2026-10-16). A call that would pass it is refused. Delete the file to reset.
GEN_SPEND_CAP_USD = 40.0
GEN_SPEND_LEDGER = CACHE_DIR / "spend" / "generation.json"

# ---------------------------------------------------------------- thinking level (run 4)
# How the agent's steps ask for thinking (llm/client.py, GeminiChat). None: as in run 1, the
# budget AGENT_THINKING_BUDGET is sent (a maximum: the model decides when to think). "low" or
# "high": that thinking level is sent instead, on every call that asks for
# AGENT_THINKING_BUDGET (each agent step, the forced finish included). The compaction call
# keeps its own COMPACT_THINKING_BUDGET. The API takes one or the other, never both.
# Why: run 1 (docs/eval-log.md): all 33 model calls of the three runs that hit the step limit
# used 0 thinking tokens; the step 1 probe: a small prompt thought ~358 tokens with HIGH
# against ~145 with budget 1,024. User approved trying HIGH (2026-10-17); the smoke eval
# decides whether it stays.
# Output cap: thinking counts toward max_output_tokens, and a level has no budget to bound
# it. AGENT_MAX_OUTPUT_TOKENS (17,024) is kept: above run 1's longest finish (501 output
# tokens) it leaves ~16,500 tokens of thinking. Thinking speeds measured: ~150 tokens/s (step
# 1 probe), 125-400 tokens/s (the live HIGH test in test_llm_chat_client.py, 2026-10-17:
# 141-164 thinking tokens on a small tool step, where budget 1,024 thought 0). Even at 400
# tokens/s, 16,500 tokens take ~41 s, past AGENT_CALL_TIMEOUT_S (40 s): the call timeout
# cuts a long think before the cap does, so a larger cap buys nothing.
AGENT_THINKING_LEVEL: str | None = "high"

# ---------------------------------------------------------------- agent loop (track A)
# agent/agent.py and agent/answerer.py.
# Start values from the agent-tools plan (section 8); the eval tunes them once real tools
# exist. Steps that may call any tool. After this many, the next step may only call finish
# (the forced finish and its retry come on top; compaction calls are never counted).
AGENT_MAX_STEPS = 10
# Tool calls run from one model reply; the rest get a "not run" text and can be made again.
AGENT_MAX_CALLS_PER_STEP = 4
# Tool calls run per question; at or above this the next step may only call finish. The first
# search (below) counts when its result is in the history: the cap bounds all tool work of a
# request, and the result's `tool_calls` is every tool result the model saw.
AGENT_MAX_TOOL_CALLS = 25
# The first search (agent/agent.py, plan 4.2 "seed step"; user 2026-10-17: "build the search
# as well first so we send the search result for the first question as well"): before the
# first model call, code runs `search` once with the question exactly as asked, and it enters
# the history as a code-written call and its result. Not a step. Why: on tune question
# pub_0077 the question's own text ranks the right record first, but the model searched in
# the question's quoted words and never found it. Off: no first search (to compare).
AGENT_FIRST_SEARCH = True
# Upper bound on the whole first search (query embedding, ranking, the result text). A
# healthy one takes about a second: one query embedding (the client's limiter spaces them
# 0.5 s apart, so 4 requests at once wait up to 1.5 s) plus the ranking. 8 s also covers two
# retries of a 429 (backoff 1 s, then 2 s). Past it the request goes on without it: the
# embedding client may retry for minutes (EMBED_MAX_RETRIES, EMBED_TIMEOUT_MS), and that wait
# would come out of every request's 150 s. The thread is not stopped; its result is dropped.
AGENT_FIRST_SEARCH_TIMEOUT_S = 8.0
# Seconds kept back for the forced finish: with less time left, the next step may only call
# finish, and a normal step's model call must end this long before the agent's deadline.
AGENT_FINISH_RESERVE_S = 30.0
# Upper bound for one model call (the call also never runs past the agent's deadline).
AGENT_CALL_TIMEOUT_S = 40.0
# A forced finish that fails (model error, bad finish arguments, no call) is tried once more.
AGENT_FORCED_FINISH_TRIES = 2
# Compaction (plan 6.5): above this many estimated prompt tokens, the older steps are
# replaced by one working-memory message before the next step. The largest corpus document
# is ~8,200 tokens, so a few whole reads fit below it.
AGENT_COMPACT_AT_TOKENS = 80_000
# Complete steps (one model turn and all its tool messages) kept word for word.
AGENT_COMPACT_KEEP_STEPS = 1
# Length of a working memory: the text the compaction call may write.
COMPACT_MAX_OUTPUT_TOKENS = 1_500
# Thinking budget of the compaction call. Thinking counts toward max_output_tokens (step 1
# probe), so the call's cap is the budget plus the working memory's length.
COMPACT_THINKING_BUDGET = 256
COMPACT_CALL_MAX_OUTPUT_TOKENS = COMPACT_THINKING_BUDGET + COMPACT_MAX_OUTPUT_TOKENS
# The agent's deadline ends this long before the server's ASK_DEADLINE_S cut (its
# asyncio.wait_for). Every model call ends by the agent's deadline, so this covers the
# tools of the last step, resolving the citations and building the response: the request
# ends with the agent's answer, never with the server's timeout answer.
AGENT_DEADLINE_MARGIN_S = 10.0

# ---------------------------------------------------------------- retrieval (step C eval)
# Chunks each ranker (BM25, dense) returns before fusion (the eval rule agreed in step C).
RETRIEVAL_TOP_CHUNKS = 50
# Reciprocal rank fusion: score = sum over rankers of 1 / (RRF_K + rank), rank from 1.
# 60 is the constant of the original RRF paper (Cormack et al. 2009) and common default.
RRF_K = 60
# Weighted fusion: score = sum over rankers of weight / (RRF_K + rank). Both 1.0 = plain
# RRF (above). The retrieval eval tunes them (`eval-retrieval --bm25-weight/--dense-weight`).
RRF_BM25_WEIGHT = 1.0
RRF_DENSE_WEIGHT = 1.0
# BM25 column weights over chunks_fts (header, text). The header repeats the source, title
# and date on every chunk of a document, so it counts half as much as the text (design §5).
BM25_HEADER_WEIGHT = 0.5
BM25_TEXT_WEIGHT = 1.0
# Keys per SQL statement when many rows are looked up by key (far below SQLite's limit of
# 32,766 bound variables, so statements stay short).
SQL_KEYS_PER_QUERY = 500

# ---------------------------------------------------------------- retrieval eval (eval/)
# Per-variant chunk indexes (vectors come from the shared EMBED_CACHE_DB).
EVAL_CACHE_DIR = CACHE_DIR / "eval"
CHUNK_DB_NAME = "chunks.sqlite"
# Document recall is reported at these cut-offs; "all gold" counts questions with every
# gold document within EVAL_ALL_GOLD_K.
EVAL_RECALL_KS = (5, 10, 20)
EVAL_ALL_GOLD_K = 10
# Embedding requests in flight at once during the eval (one request = EMBED_BATCH_SIZE
# texts). The client's rate limiter still caps requests per minute.
EVAL_EMBED_WORKERS = 4
# Screening (used only when one full variant takes too long to embed): dense and hybrid
# run on every gold document of the set plus this many other documents, drawn with
# EVAL_SCREEN_SEED.
EVAL_SCREEN_SAMPLE_DOCS = 5_000
EVAL_SCREEN_SEED = 20261016
# Chunk and vector rows per write transaction (a few MB each).
CHUNK_WRITE_BATCH_ROWS = 5_000
# The eval CLI waits this long for queued spans at exit (one run sends hundreds of
# embedding generations; the server's 2 s budget is for SIGTERM, not for this).
EVAL_TRACE_FLUSH_TIMEOUT_S = 60.0

# ---------------------------------------------------------------- end-to-end eval (eval/run.py)
# `uv run eval-<set>` asks the RUNNING server every question of a set through POST /ask.
# Its default server (the port serve-dev listens on).
EVAL_ASK_URL = "http://localhost:8080"
# Requests in flight at once: the default, and the most allowed (CONTRACT §4: up to 4 at once).
EVAL_ASK_CONCURRENCY = 2
EVAL_ASK_MAX_CONCURRENCY = 4
# One question may take this long, from sending the request to the last byte (CONTRACT §4).
EVAL_ASK_TIMEOUT_S = 180.0
# Latency percentiles in the report (nearest rank), besides the maximum.
EVAL_ASK_LATENCY_PERCENTILES = (50, 95)
# Reports go to EVAL_RESULTS_DIR/<set>/ask/<time>/; eval/history.csv gets one row per run.
EVAL_DIR = Path(__file__).resolve().parents[2] / "eval"
EVAL_RESULTS_DIR = EVAL_DIR / "results"
EVAL_HISTORY = EVAL_DIR / "history.csv"
# Every spend ledger (.cache/spend/*.json); a run reports how much they grew while it ran.
EVAL_SPEND_DIR = EMBED_SPEND_LEDGER.parent

# ---------------------------------------------------------------- answer judge (eval/judge.py)
# The eval's automatic answer judge: fact by fact against `answer_facts` (user approved the
# design 2026-10-17: "okay build it that way and add the second one as the model"). The
# "second one" is the Pro-class model of the step 1 probe: stronger than AGENT_MODEL and not
# the model that wrote the answers. Never used by the service.
JUDGE_MODEL = "gemini-3.1-pro-preview"
# USD per 1M tokens of JUDGE_MODEL: paid tier, Standard, prompts <= 200k tokens (a judge
# prompt is a few thousand), from https://ai.google.dev/gemini-api/docs/pricing read
# 2026-10-16 in the step 1 probe (decisions.md, agent-tools run). Above 200k: 4.00 / 0.40 /
# 18.00. Output includes thinking tokens.
JUDGE_PRICE_INPUT_PER_M_TOKENS_USD = 2.00
JUDGE_PRICE_CACHED_INPUT_PER_M_TOKENS_USD = 0.20
JUDGE_PRICE_OUTPUT_PER_M_TOKENS_USD = 12.00
# The judge's system prompt (src/corpusqa/prompts/judge.md) and its forced tool.
JUDGE_PROMPT_NAME = "judge"
JUDGE_TOOL_NAME = "verdict"
# Thinking budget of one judge call, and room for the verdict: one entry per fact with a
# short quote (the longest tune question has 11 facts, ~1,000 tokens). Thinking counts toward
# max_output_tokens (step 1 probe), so the cap is the budget plus the verdict's room.
JUDGE_THINKING_BUDGET = 1_024
JUDGE_VERDICT_MAX_TOKENS = 4_096
JUDGE_MAX_OUTPUT_TOKENS = JUDGE_THINKING_BUDGET + JUDGE_VERDICT_MAX_TOKENS
# One judge call may take this long (the client's retries of 429 / 5xx included).
JUDGE_CALL_TIMEOUT_S = 120.0
# Judgments per answer: two; when their question verdicts differ, one more, and the majority
# counts. An invalid verdict (not the tool's schema, wrong facts) is asked for once more; a
# second invalid one leaves the question unjudged (counted, never guessed).
JUDGE_JUDGMENTS = 2
JUDGE_TIEBREAK_JUDGMENTS = 1
JUDGE_TRIES_PER_JUDGMENT = 2
# Answers judged at once (each answer's judgments run one after another). A call takes
# ~20-60 s (calibration, 2026-10-17); calibration round 1 ran 6 at once, 258 calls in 20 min,
# with only 504s (each passed on its retry) and no 429.
JUDGE_CONCURRENCY = 6
# Hard cap on recorded judge spend, in its own ledger (user, 2026-10-17); the eval runner's
# spend report reads every ledger in EVAL_SPEND_DIR, this one included. Delete it to reset.
JUDGE_SPEND_CAP_USD = 10.0
JUDGE_SPEND_LEDGER = EVAL_SPEND_DIR / "judge.json"
# Calibration only (`uv run eval-judge-calibrate`, tune data): judgments kept by a hash of
# model, prompt version, settings, the exact input and the judgment's number, so a re-run of
# the same prompt version pays nothing. The /ask runner never uses it (it would keep
# per-question held-out results outside the guarded folder).
JUDGE_CACHE_DIR = CACHE_DIR / "judge"
# The bar agreed with the user: the judge's fact status equals the hand check's on at least
# this share of facts.
JUDGE_CALIBRATION_BAR = 0.90

# ---------------------------------------------------------------- agent tools (prompts/tools.json)
# Gemini's limit on a function name (ai.google.dev function calling docs). The loader also
# keeps names to letters, digits and underscores, starting with a letter or underscore.
TOOL_NAME_MAX_CHARS = 64

# ---------------------------------------------------------------- corpus tools (track T)
# The agent's corpus tools (agent/corpus_tools.py over retrieval/). Every list a tool
# returns comes in pages with its exact total and the next offset (user decision
# 2026-10-16: nothing is cut or dropped; the model pages on or narrows its call).
FILTER_PAGE_ROWS = 50
SEARCH_PAGE_DOCS = 8
# A search page past what the ranking found re-ranks with this many times more chunks per
# ranker (starting from RETRIEVAL_TOP_CHUNKS), again until the page is filled or every
# chunk in scope is ranked (user decision 2026-10-16: "Search deeper").
SEARCH_DEPTH_GROWTH = 2
READ_FIELDS_PAGE_ROWS = 50
LIST_FIELDS_PAGE_ROWS = 50
FIELD_VALUES_PAGE_ROWS = 50
KEY_LOOKUP_PAGE_ROWS = 50
# A search result shows each document's best passages (read_document shows it whole): at
# most this many, each a window of about this many characters around the query's words.
SEARCH_SNIPPETS_PER_DOC = 2
SEARCH_SNIPPET_CHARS = 300
# A name or value is near the model's words at this score (retrieval/values.nearness:
# 1 = same loose form, 0.9 = one contains the other, 0.75 = one of two words shared, else
# the difflib ratio of the loose forms). At 0.6 on data/, "stage" was near "tags" and
# "qps_target"; at 0.75 a typo ("staus") still is.
NEAR_MIN_RATIO = 0.75
# An error for a field or value that matches nothing names this many of the nearest ones
# (list_fields / field_values list them all).
NEAREST_SHOWN = 5
# A `contains` condition names this many of the values it matched (each row shows its own).
CONTAINS_VALUES_SHOWN = 10
# Top values previewed per field: in list_fields, and in the system prompt's field catalog.
LIST_FIELDS_TOP_VALUES = 3
CATALOG_TOP_VALUES = 5
# Values of a field averaging more characters than this are free text: not previewed
# (field_values lists them whole).
PREVIEW_MAX_AVG_CHARS = 60
# The field catalog (system prompt): each source's common fields, by the same rule and
# thresholds as the chunk header fields (store/profiles.py; the profile picks, never a name).
CATALOG_FIELD_MIN_PRESENCE = HEADER_FIELD_MIN_PRESENCE
CATALOG_FIELD_MAX_TITLE_RATIO = HEADER_FIELD_MAX_TITLE_RATIO
CATALOG_FIELD_MAX_CONTENT_RATIO = HEADER_FIELD_MAX_CONTENT_RATIO
# ...and fields whose rows are mostly generator noise (ingest's is_noise) are left out.
CATALOG_FIELD_MAX_NOISE_RATIO = 0.5
# Handles a request's tools give out: "D<n>" for a document shown, "R<n>" for a result set.
DOC_HANDLE_PREFIX = "D"
RESULT_HANDLE_PREFIX = "R"

# ---------------------------------------------------------------- exact search (run 3)
# Run 1 (smoke, 2026-10-17): 20 of 22 wasted tool calls were searches for quoted keys and
# phrases; the keyword side split each into loose words (any word matches), so a key that
# is written nowhere still filled the page. On, the keyword side (store/sqlite.py
# match_expression) matches text in double quotes as a phrase (its words together, in
# order) and a key-shaped token (letters or digits joined by - _ / . #) as the phrase of
# its parts; other words still match one by one, any of them; capital AND / OR are not
# searched for. The search page then marks the documents only the dense side found and
# says when no document in scope matched the keywords. Off: run 1's behaviour exactly.
# prompts/tools.json describes the on behaviour: turn this on together with that text.
SEARCH_EXACT_PHRASES = True

# ---------------------------------------------------------------- logging
LOG_LEVEL = "INFO"
LOG_FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"
# Questions are logged for debugging, cut to this length to keep log lines readable.
LOG_QUESTION_CHARS = 200
# How many dropped (unknown) document ids one log line shows.
LOG_DROPPED_ID_SAMPLE = 5


class ConfigError(ValueError):
    """The environment does not describe a runnable service; the message says how to fix it."""


@dataclass(frozen=True, slots=True)
class Settings:
    """Runtime settings. Built once at start-up and passed down, never read from env again."""

    corpus_dir: Path
    port: int
    # repr=False keeps the key out of logs and tracebacks.
    gemini_api_key: str | None = field(default=None, repr=False)
    # Tests shrink this with dataclasses.replace; production uses the constant.
    ask_deadline_s: float = ASK_DEADLINE_S
    # Where ingest writes its SQLite index. Tests point it at a temporary folder.
    index_dir: Path = INDEX_DIR
    # The embedding cache S9 reads and fills (only when there is an embedder). Tests that
    # pass an embedder point it at a temporary file.
    embed_cache_db: Path = EMBED_CACHE_DB
    # Optional, dev only. Tracing is on only when both keys are set (obs.init_tracing).
    # Both keys stay out of repr, like the Gemini key.
    langfuse_public_key: str | None = field(default=None, repr=False)
    langfuse_secret_key: str | None = field(default=None, repr=False)
    langfuse_host: str = LANGFUSE_DEFAULT_HOST

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> "Settings":
        """Read and validate CORPUS_DIR, PORT and GEMINI_API_KEY (CONTRACT.md §1).

        Also reads the optional LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY and
        LANGFUSE_BASE_URL (or the older LANGFUSE_HOST). Missing or empty means unset.
        """
        env = os.environ if env is None else env
        host = env.get("LANGFUSE_BASE_URL") or env.get("LANGFUSE_HOST") or LANGFUSE_DEFAULT_HOST
        return cls(
            corpus_dir=_corpus_dir(env.get("CORPUS_DIR")),
            port=_port(env.get("PORT")),
            gemini_api_key=env.get("GEMINI_API_KEY") or None,
            langfuse_public_key=env.get("LANGFUSE_PUBLIC_KEY") or None,
            langfuse_secret_key=env.get("LANGFUSE_SECRET_KEY") or None,
            langfuse_host=host.rstrip("/"),
        )


def _corpus_dir(raw: str | None) -> Path:
    if not raw:
        raise ConfigError("CORPUS_DIR is not set; set it to the corpus directory (CONTRACT.md).")
    path = Path(raw).resolve()
    if not path.is_dir():
        raise ConfigError(f"CORPUS_DIR={raw!r} is not an existing directory.")
    if not os.access(path, os.R_OK | os.X_OK):
        raise ConfigError(f"CORPUS_DIR={raw!r} is not readable by this process.")
    return path


def _port(raw: str | None) -> int:
    if not raw:
        raise ConfigError("PORT is not set; set it to the TCP port to listen on (CONTRACT.md).")
    try:
        port = int(raw)
    except ValueError:
        raise ConfigError(f"PORT={raw!r} is not an integer.") from None
    if not 1 <= port <= 65535:
        raise ConfigError(f"PORT={raw!r} is outside 1-65535.")
    return port
