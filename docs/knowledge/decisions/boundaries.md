---
name: Import boundary decisions
description: Decisions about the import-linter contracts in pyproject.toml: "SQL only in store" allows indirect imports (a direct sqlite3 import outside store/ still fails); "Gemini SDK only in llm" ignores exactly one import, server.__main__ -> llm.client, so only the wiring point builds the Gemini embedder and every other layer types against llm.interfaces. Both wait for the user's confirmation. (The Langfuse contract is in tracing.md.)
source: runs/2026-10-14-ingestion
---

# Import boundary decisions

## Open questions (OPEN: waiting for the user)
- **B2 (2):** confirm the "SQL only in store" change (indirect sqlite3 imports allowed;
  direct ones still forbidden).
- **B4 (1):** confirm the narrower Gemini change (one ignored import instead of
  `allow_indirect_imports`, which was the brief's default).

## 2026-10-16 B4: the "Gemini SDK only in llm" contract ignores exactly one import (server/__main__ -> llm.client)
- **Area:** checks / import boundaries (pyproject.toml); wiring
- **Chose:** `ignore_imports = ["corpusqa.server.__main__ -> corpusqa.llm.client"]` on that contract, not `allow_indirect_imports = true`. `server/__main__.py` builds `GeminiEmbedder.from_settings` (None, with a log line, when there is no key or the ledger is unreadable) and hands it to `ingest.run(settings, embedder=...)` through `functools.partial`; ingest and every other layer type against `llm.interfaces.Embedder`. `Runtime.embedder` carries it so queries are embedded the same way.
- **Probes (each temporary module removed after):** `__main__` importing `llm.client` KEPT; `__main__` importing `google` directly BROKEN; a server module importing `google.genai` BROKEN; ingest, retrieval or another server module importing `llm.client` BROKEN (indirect path still caught).
- **Over:** `allow_indirect_imports = true` as for SQL. It would let any layer import `llm.client` and so reach the SDK through it; the narrower rule keeps what `llm/interfaces.py` says ("only the wiring code builds the Gemini adapter") and what AGENTS.md says (`__main__` is the only wiring point), and it is checked.
- **Note:** import-linter fails on an ignore line that matches no import, so the contract change and the `__main__` wiring had to land in one commit. A stale ignore is therefore caught later too. AGENTS.md Boundaries got one line saying this.
- **Why:** brief B4 step 1 ("if you find a cleaner way that keeps the rule's intent, use it and say why").

## 2026-10-16 B2: the "SQL only in store" import contract allows indirect imports
- **Area:** checks / import boundaries (pyproject.toml)
- **Chose:** add `allow_indirect_imports = true` to the "SQL only in store" contract, like the Langfuse contract already has. A direct `import sqlite3` outside `store/` still fails (checked with a probe module, then removed).
- **Over:** leaving it as it was, under which no layer could use `store.sqlite` at all (ingest -> store.sqlite -> sqlite3 counted as ingest importing sqlite3); or injecting the writer from `server/__main__`, which the same rule also blocks.
- **Why:** AGENTS.md says "No SQL or sqlite3 outside store/", i.e. a direct-import rule; the contract as written made the store unusable. Own commit, because it changes a check.
