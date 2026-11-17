---
name: Codex review — ingestion build (B0-B4, step C)
description: Independent read-only Codex review (gpt-6-astra, effort high, 2026-10-16) of git diff foundation..ingest/build: 10 findings on S1-S10, cache, S9 spend and failures, discovery, eval. Triage of each finding is in docs/runs/2026-10-14-ingestion/decisions.md ("B5 Codex review"). Read before changing ingest, the index cache or S9.
---

# Codex review: ingestion build (2026-10-16)

Command: `codex exec --sandbox read-only -m gpt-6-astra -c model_reasoning_effort="high"` with the prompt below (scope: `git diff foundation..ingest/build` at 465e5377). Output pasted unchanged.

<details><summary>Prompt</summary>

You are an independent senior reviewer. Review the ingestion build of this repo. This is a READ-ONLY review: make NO code changes, write NO files, run no command that changes the repo, its caches or git state.

## What to review
The ingestion work is the diff `git diff foundation..ingest/build` (20 commits; the branch is checked out). Focus on `src/`, `tests/`, `serve`, `pyproject.toml` and `eval/retrieval.py`. Skip fixture JSON, eval report JSON and uv.lock except as evidence.

Rules and intent (read these first):
- `CONTRACT.md` (the grader's interface: /health, /ask, env vars, start-up limit, SIGTERM).
- `AGENTS.md` (hard rules, import boundaries, tracing rules).
- `docs/runs/2026-10-14-ingestion/design.md` (the spec, stages S1-S10) and `docs/runs/2026-10-14-ingestion/decisions.md` (later entries override the design; newest wins).

What is out of scope: the answerer is deliberately still a placeholder (`PlaceholderAnswerer`); the agent, verify and cite steps come later. Do not report "no agent" as a finding.

## Do NOT open these (split rule, hooks enforce it)
`data/questions.jsonl`, `eval/sets/heldout.json`, anything under `eval/results/heldout/`, and `.env` (secrets). You may read `data/corpus/` files if you need evidence of the format, but nothing from `data/` may appear in code.

## Focus areas, in this order
1. Correctness of S1-S10: discover, parse, core, fields, clean, store, date window, profile/keys/links, chunk, embed, finalize. Look for wrong results, lost documents, wrong ids, off-by-one, non-determinism, unhandled input shapes.
2. Contract traps: /health must stay 503 until ingest is complete and must stay 503 if start-up raises; /ask always HTTP 200 with a non-empty answer; `document_ids` only ids from the ingest id map; `CORPUS_DIR` treated read-only (never written, no sidecar files, no SQLite files opened inside it); only CORPUS_DIR, PORT, GEMINI_API_KEY, PATH, HOME, TMPDIR, LANG read (plus optional LANGFUSE_*); `./serve` foreground, exec, clean SIGTERM; start-up within 4 hours on an unseen corpus of similar size with an EMPTY embedding cache.
3. Hard-coding: anything learned from `data/` written into code, config or prompts (source names, field names, company names, issue keys, doc ids, folder names). Everything must work for an unseen corpus of the same format.
4. The index cache: the fingerprint (is every input that changes the index in it? could a stale index be reused for a different corpus or changed code?), the cache decision, publication only when complete, and pruning (could it delete an index in use, or one belonging to another corpus? races between two servers?).
5. S9 embedding: failure handling (429, timeouts, wrong vector counts, bad vectors, API down), the deadline, the spend ledger and cap (could spend be under-counted, double-counted, or exceed the cap under concurrency?), the embedding cache key, resume after a partial embed, never failing start-up.
6. Concurrency and thread-safety: the read-only store and chunk reader used by up to 4 concurrent /ask requests (SQLite connections across threads, shared numpy state, connection pools), the embed thread pool writing to SQLite, the start-up daemon thread.
7. Tracing gaps: AGENTS.md says every stage is a span with counts, every embedding call a Langfuse generation (model, input, output, tokens, cost, retries, errors), nothing secret in a trace, tracing never breaks or slows a request.

## Output format
Plain Markdown, no preamble:
`## 1. Verdict` (3-5 sentences), then `## 2. Findings`: a numbered list. Each finding:
`N. **<Severity: Critical | High | Medium | Low> — <one-line title>.**` then `Evidence:` with file:line references (and a command output or a short code quote where it helps), then `Why it matters:` (one or two sentences), then `Suggested fix:` (concrete). Order by severity. Report only real problems you verified in the code; say how confident you are when unsure. Then `## 3. Checked and fine` with a short list of the things you checked and found correct.

</details>

## 1. Verdict

**Request changes.** The ingestion stages broadly follow the latest decisions, but malformed input and cache lifecycle races can prevent startup or invalidate a running server’s index. Additional verified problems affect embedding failure handling, spend enforcement, and cache correctness. This review used code inspection and file-free probes; no files were changed, and the full suite or live empty-cache startup was not run, so the four-hour startup requirement remains unverified.

## 2. Findings

1. **High — An escaped lone surrogate aborts ingestion instead of quarantining one document.**

   **Evidence:** [parse.py:55](src/corpusqa/ingest/parse.py:55) accepts strings containing JSON escapes such as `\ud800`. They survive document processing and reach SQLite’s UTF-8 encoding in [sqlite.py:264](src/corpusqa/store/sqlite.py:264), outside the per-document exception handler. A probe using `_load`, mocked input bytes, and an in-memory SQLite database produced `UnicodeEncodeError` with zero quarantine rows.

   **Why it matters:** One malformed string in an otherwise usable corpus fails the entire build and leaves `/health` at 503.

   **Suggested fix:** Validate decoded strings and keys before storage. Quarantine the affected document or replace invalid Unicode with a recorded flag, while retaining the original bytes. Add a regression test containing a bad document alongside a good one.

2. **High — Concurrent builders delete each other’s active database files.**

   **Evidence:** [fingerprint.py:142](src/corpusqa/ingest/fingerprint.py:142) treats an existing `.building` file as a cache miss. [sqlite.py:134](src/corpusqa/store/sqlite.py:134) then unconditionally unlinks that file and its WAL/SHM files. Every builder for the same fingerprint uses the same path, without a process lock.

   **Why it matters:** Two servers starting against an uncached corpus can remove a database another process is actively writing. Publication and abort cleanup can subsequently operate on the other builder’s files, causing failed startup or an invalid published index.

   **Suggested fix:** Acquire a process-wide lock per fingerprint before deciding or building, and recheck the cache after acquiring it. Use unique temporary build paths and serialize publication and embedding resume.

3. **High — Pruning can delete an index still used by a running server.**

   **Evidence:** [fingerprint.py:171](src/corpusqa/ingest/fingerprint.py:171) deletes indexes based on corpus-directory existence and folder age; it has no record of active readers. [sqlite.py:483](src/corpusqa/store/sqlite.py:483) opens a fresh connection by pathname for each operation. A third build can therefore remove the first server’s index while that server remains alive. Identical corpus copies also share an index whose metadata records only the original corpus path.

   **Why it matters:** Later reads fail after pruning. Deleting the original corpus directory can also make an index used through another identical copy eligible for removal.

   **Suggested fix:** Protect each active runtime with a process-visible lease or shared lock. Prune only after acquiring exclusive ownership, and account for shared indexes rather than treating the original corpus path as their sole owner.

4. **Medium — Corpus fingerprints can identify different contents as the same corpus.**

   **Evidence:** [fingerprint.py:113](src/corpusqa/ingest/fingerprint.py:113) hashes only relative paths, sizes, and modification times. Cache hits at [fingerprint.py:138](src/corpusqa/ingest/fingerprint.py:138) do not validate contents. An in-memory probe confirmed identical hashes for distinct corpus roots with matching file metadata.

   **Why it matters:** Different snapshots with equal-length replacements and preserved timestamps can reuse stale bodies and document IDs. This breaks the assumption that `known_doc_ids` describes the current corpus.

   **Suggested fix:** Include file-content digests in the fingerprint. This preserves the intended sharing between genuinely identical corpus copies.

5. **Medium — S9’s exception boundary does not cover the whole embedding stage.**

   **Evidence:** [embed.py:172](src/corpusqa/ingest/embed.py:172) constructs the embedding writer outside `_embed_safely`; metadata writes, stage recording, and writer closure also remain outside its handler. Injecting `sqlite3.OperationalError("database is locked")` during writer construction propagated out of `embed_stage`, including with no embedder.

   **Why it matters:** A recoverable embedding-write failure can fail startup despite a complete, readable keyword index, contradicting the latest decision that S9 must not fail startup.

   **Suggested fix:** Guard the complete S9 lifecycle. On recoverable embedding-write failures, close resources and return the readable keyword index with degraded coverage; keep core-index corruption fatal.

6. **Medium — Concurrent embedding calls can exceed the spend cap.**

   **Evidence:** [limits.py:71](src/corpusqa/llm/limits.py:71) checks completed spend without reserving the pending call’s cost. Charging happens later at [client.py:163](src/corpusqa/llm/client.py:163). A four-thread probe, with persistence mocked out, accepted four $0.40 calls against a $1.00 cap and recorded $1.60.

   **Why it matters:** The configured four-worker embedding path can admit several requests against the same remaining balance. Individual method locks do not make admission and charging atomic.

   **Suggested fix:** Reserve estimated spend atomically before dispatch, then reconcile the reservation with actual usage. Make reservations process-safe for the shared ledger, and add a concurrent admission test.

7. **Medium — The production adapter defeats S9’s malformed-vector handling.**

   **Evidence:** [client.py:172](src/corpusqa/llm/client.py:172) raises `EmbeddingError` for any malformed vector or wrong vector count. [embed.py:393](src/corpusqa/ingest/embed.py:393) treats that as a retryable request failure, and [embed.py:361](src/corpusqa/ingest/embed.py:361) requeues the entire request. Consequently, S9’s later per-vector salvage and non-retryable count-mismatch branches are bypassed by `GeminiEmbedder`.

   **Why it matters:** One bad vector discards valid vectors from the same batch, and malformed responses trigger another billed request. This differs from the latest B4 decisions; tests using an embedder that returns malformed vectors directly do not exercise the production path.

   **Suggested fix:** Distinguish transport failures from response-validation failures. Preserve valid vectors when count alignment is intact, and make count mismatches non-retryable. Test through `GeminiEmbedder` into S9.

8. **Medium — Cached vectors bypass the nonzero and normalization checks.**

   **Evidence:** [vectors.py:53](src/corpusqa/store/vectors.py:53) checks only byte length and finiteness. [embed.py:270](src/corpusqa/ingest/embed.py:270) stores accepted cache blobs unchanged. The probe `blob_fits(to_blob([0.0, 0.0]), 2)` returned `True`; finite non-unit vectors also pass.

   **Why it matters:** An unusable cached vector becomes a completed embedding and is never retried. Non-unit vectors distort the dot-product ranking that assumes unit-length stored vectors.

   **Suggested fix:** Apply the same validation and normalization to cached vectors as fresh vectors. Treat zero or otherwise invalid entries as cache misses, and verify a finite, positive normalization result.

9. **Medium — Internal directory symlinks make discovery nondeterministic.**

   **Evidence:** [discover.py:27](src/corpusqa/ingest/discover.py:27) consumes unsorted directory entries and marks resolved folders visited immediately. Sorting the final file list cannot undo that choice. A mocked filesystem probe returned `source/note.json` when the real directory appeared first, and `alias/note.json` when its symlink appeared first.

   **Why it matters:** The same corpus can acquire different source labels, paths, chunk headers, and fingerprints depending on directory enumeration order.

   **Suggested fix:** Define a deterministic traversal policy, preferably processing real directories before symlink aliases and sorting entries consistently. Test both enumeration orders.

10. **Medium — Screened eval indexes are reused across different question sets.**

    **Evidence:** [eval/retrieval.py:536](eval/retrieval.py:536) selects screen documents using the supplied questions, but the manifest at [eval/retrieval.py:541](eval/retrieval.py:541) includes neither the question set nor the selected-document IDs. Both smoke and tune use the same variant cache location.

    **Why it matters:** Running screened smoke and then screened tune can reuse an index missing tune gold documents. The reported comparison then measures the wrong candidate corpus.

    **Suggested fix:** Include a digest of the selected document IDs in the manifest and cache identity. Report indexed IDs from the actual screen subset, and test smoke-to-tune reuse with invented questions.

## 3. Checked and fine

- Readiness is published only after startup returns; startup exceptions leave `/health` at 503.
- `/ask` guards malformed input, timeouts, internal errors, blank answers, and unknown citations.
- Corpus inputs are read through file operations; index and embedding databases use separate cache paths.
- Ordinary request readers use separate SQLite connections; embedding workers return results to the thread owning the SQLite writer.
- Single-builder publication checkpoints WAL before renaming; completed embedding batches are checkpointed for resume.
- Date-window learning, low-trust date handling, and recursive chunking follow the later decisions.
- No dataset-specific behavior was found; the three file-free hard-coding checks passed.
- Stage spans and embedding generations are wired through `obs`, including worker context propagation and secret scrubbing.
- `serve` remains foregrounded with `exec`; the installed Uvicorn restores and re-raises SIGTERM after shutdown.
- Forbidden files were not opened, and no live or held-out eval was run.