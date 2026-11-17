---
name: Codex review — agent harness, corpus tools, server wiring (21000f7c..cdc765eb)
description: Independent read-only Codex review (gpt-6-astra, effort high, 2026-10-17) of git diff 21000f7c..cdc765eb on agent/tools (agent loop, GeminiChat and its Gemini conversion, ToolBox, corpus tools and handles, store reads, server wiring and the /ask queue): 2 High (tool time not bounded by the deadline; request body read outside the deadline), 7 Medium, 1 Low. Triage at the end of this file (after it, 1 High, 3 Medium, 5 Low, 1 rejected as a user decision); the run's decisions.md entries and the fixes wait for the user. Read before changing the agent loop, llm/client.py, the corpus tools, the filter's contains op or the /ask handler.
---

# Codex review: agent harness, corpus tools and server wiring (2026-10-17)

Command: `codex exec --sandbox read-only -m gpt-6-astra -c model_reasoning_effort="high" -o <file> -` with the prompt below on stdin, run from the worktree `.claude/worktrees/agent-tools`.
Scope: `git diff 21000f7c..cdc765eb` (`src/`, `tests/`, `serve`, `pyproject.toml`, `commands.json`). Three other agents were editing this worktree during the review, so Codex read a frozen copy of the tree at `cdc765eb` (made with `git archive cdc765eb src tests serve pyproject.toml commands.json CONTRACT.md AGENTS.md CLAUDE.md docs scripts`, plus the two run folders' `plan.md`, `brief.md` and `decisions.md`, which git does not track). No `eval/`, `data/` or `.env` in that copy. The line numbers below are for `cdc765eb`.
The output is pasted unchanged. It was taken from the Codex log, not from `-o`: the project's Codex Stop hook made Codex write one more line after the review ("No project files were changed..."), and `-o` keeps only that last line. Codex used 202,616 tokens; it ran no tests (the copy has no virtualenv).

<details><summary>Prompt</summary>

You are an independent senior reviewer. Review one body of work in this repo. This is a READ-ONLY review: make NO code changes, write NO files, run no command that changes anything. Never open `.env` or any secret file. Never open `data/questions.jsonl`, `eval/sets/` held-out files or anything under `eval/results/` (the eval split rule in AGENTS.md); the copy below has no `data/` and no `eval/` folder, and you do not need them.

## The copy you are reading
Review ONLY the frozen copy of the repo at commit `cdc765eb` in this folder:
(a local scratch copy) (call it SNAP). Paths under SNAP are the repo's paths, so `SNAP/src/corpusqa/agent/agent.py` is that file at `cdc765eb`.
Your working directory is the live repo, where three other agents are editing files right now: do NOT read source, test or docs files from the working directory (they differ from the reviewed commit) and do not run git there. Read everything from SNAP by absolute path. SNAP has no `.git` and no virtualenv: do not run `uv`, tests or the server. Read files with `rg`, `sed -n`, `cat`; a small `python3 -c` to check a regex or a string is fine. Cite evidence as repo-relative `path:line` (the path under SNAP, without the SNAP prefix).

The work under review is the diff `21000f7c..cdc765eb`, saved as `SNAP/REVIEW_DIFF.patch` (only `src/`, `tests/`, `serve`, `pyproject.toml`, `commands.json`). It holds: the agent harness (chat model protocol, Gemini client and message conversion, fake model, tool box, the ReAct loop with limits, deadline, forced finish and compaction, the answerer), the corpus tools (search, filter, read_fields, read_document, list_fields, field_values, lookup_key, D#/R# handles, the session factory), the store reads they use, the server wiring and the /ask queue. Read the real files, not only the patch.

Out of scope: the eval runner (`eval/`, not in this copy); the internals of `src/corpusqa/obs/` (a separate tracing branch merged in; only how the agent, tools and server USE it is in scope); ingest changes from `main` (weighted RRF, header fields) except the track-T date rule (each date judged on its own against the learned window) and the `store/norms.py` move. Three changes planned after the first eval run are NOT built yet (exact matching for quoted text in search, a stop rule and step count in the prompt, a higher thinking level): do not report their absence. Known run-1 observations (many runs hitting the 10-step limit, 0 thinking tokens on forced steps, a model wrapping argument keys in quotes) are not findings by themselves; report them only if you find a code bug behind them.

## Rules and intent (read first, all under SNAP)
- `AGENTS.md`: hard rules, import boundaries, tracing rules ("if it runs, it is traced"), LLM rules, Python conventions. `CONTRACT.md`: the grader's interface (/health, /ask, env vars, per-request limits, up to 4 requests at once).
- `docs/runs/2026-10-16-agent-tools/plan.md` and `docs/runs/2026-10-16-corpus-tools/plan.md`: the plans. `docs/runs/*/decisions.md`: decisions, newest first; a later decision overrides the plan. `docs/handoffs/2026-10-16-agent-tools.md` and `docs/handoffs/2026-10-16-corpus-tools.md`: what was built, step by step. (The `docs/runs/` files are a current copy and may mention work after `cdc765eb`.)
- User decisions you must hold the code to (not re-argue): tool results are never cut by the tool box; every list a tool returns pages with an exact total and the next offset, so no data is ever dropped without the model being told; a failed or blank compaction call keeps the history whole and is retried at the next step; the server keeps at most `MAX_DOCUMENT_IDS = 500` ids in a response and logs + traces any cut; at most 4 answers at once, the rest wait in a queue, and the deadline counts from arrival. If the code does not match one of these, or one of them breaks a hard rule, say so.

## Look especially for (in this order)
1. **The /ask guarantee.** Every path gives HTTP 200 with a non-empty answer inside the internal deadline (150 s; the grader allows 180 s): queue wait + agent loop + forced finish retries + compaction call + citation resolution + response building. Find a path that can exceed the deadline, hang, raise past the handler, or return an empty or "not found" answer for an internal error. Check the deadline arithmetic (`AGENT_FINISH_RESERVE_S`, `AGENT_CALL_TIMEOUT_S`, `AGENT_DEADLINE_MARGIN_S`, the queue's time left, per-call `timeout_s`, retries and backoff inside one timeout, `asyncio.to_thread` work that a timeout cannot cancel).
2. **No state shared between requests** (4 at once): per-question `Agent`, session, `Handles`; the immutable `CorpusIndex`/`CorpusTools`; SQLite connections used from several threads; numpy arrays; module-level caches or mutable defaults; the rate limiter and spend ledger (shared on purpose: are they correct under concurrency?).
3. **Citations only known ids:** `finish` citations (D#, R#, raw ids) through `resolve_citations` and `server/respond.build_response`; an unknown handle, a handle from another session, a made-up id, duplicates, the 500 cap.
4. **No data dropped without the model being told:** check every tool's output (search pages and snippets, filter rows and R#, read_fields pages, read_document of a very long document, list_fields, field_values, lookup_key, the field catalog in the system prompt and the fields it leaves out, error texts that list "nearest values", the "not run: at most N calls per step" text, other calls in a step that also calls finish, the compaction transcript). Any place a list is truncated, a total is not exact, or a next offset is missing or wrong is a finding. Off-by-one in paging.
5. **The agent loop:** step limit, tool-call limit, per-step call limit, the deadline reserve, the nudge after a reply with no tool call, forced finish (`tool_choice=("finish",)`), its retry, the fallback to last text, `AgentError`; compaction trigger, the split that must never break a step, what replaces what, the retry after a skipped compaction, limits re-checked after compaction, the token estimate.
6. **Gemini conversion** (`llm/client.py`): `to_gemini` (raw model turns re-sent unchanged with their thought signatures; code-built model turns and the stand-in signature; tool results grouped into one user Content per assistant turn, in call order, with matching ids/names; two user messages merged; system instruction), tool choice mapping, `from_gemini` (thought parts left out, empty ids, several function calls, blocked/empty candidates, MAX_TOKENS), usage and cost, retries only on 429/5xx, the spend ledger reserve/settle, errors mapped to the right `LLMError` subclasses.
7. **Injection from model input:** SQL built from field names, values, ops or sources; FTS5 MATCH expressions built from the query (quotes, `*`, `NEAR`, `:`, column filters, `AND/OR/NOT`, unbalanced quotes, empty after cleaning) - can one crash the tool, change the query's meaning, or read other columns; LIKE patterns for `contains` (`%`, `_`, escape); `json_each`; regex built from input.
8. **Event loop blocking and resource use** on a corpus of ~21,600 documents and ~68,000 chunks with 4 requests at once: sync SQLite, numpy or embedding calls on the event loop; "search deeper" doubling (worst case per page at a large offset); scoped BM25 that reads every match; `VectorIndex.search(allowed=)`; filters over sources with ~11,600 documents and fields with ~16,000 distinct values; R# sets of 1,000+ ids kept per session; anything quadratic.
9. **Tracing complete and safe:** one trace per /ask; every LLM and embedding call a generation with model, full input as sent, output, tool calls, tokens, cost, latency, retries, errors, prompt name and version; every tool call a span with arguments as given and as checked, internals, and the result exactly as the model saw it; every stage (agent, step, compact, citations, queue) a span. A path that runs untraced (a retry, a forced finish, a compaction failure, a tool error, the fallback, the queue timeout) is a finding. Secrets (API key, auth headers) must never reach a trace or a log.
10. **Hard-coding and boundaries:** anything from the corpus (source names, field names, company names, issue keys, doc ids) in code, config, prompts or tool schemas; `import-linter` layers (server -> agent -> answer | retrieval -> store; SQL only in `store/`; `google.genai` only in `llm/`; only `server/__main__.py` imports `llm.client`); magic numbers outside `config.py`; prompts and `tools.json` carrying a `version` that is recorded on generations; structured output parsed into Pydantic models.
11. **Test gaps:** a behaviour above with no test, or a test that would still pass with the bug present.

## Output
Plain Markdown, no preamble. `## 1. Verdict` (3-5 sentences). Then `## 2. Findings`: a numbered list, most severe first. Each finding:
`N. **<Severity: Critical | High | Medium | Low> - <one-line title>.**` then
- `Evidence:` `path:line` references (and a short code quote where it helps);
- `What goes wrong:` one or two sentences;
- `Failure scenario:` a concrete input, model reply or timing that triggers it, and what the client or the model then sees;
- `Suggested fix:` concrete;
- `Confidence:` high / medium / low.
Report only real problems you verified in the code. Do not pad. Then `## 3. Checked and fine`: a short list of what you checked and found correct.

</details>

## Output

## 1. Verdict

Request changes before accepting `cdc765eb`. Request isolation, SQL parameterization, citation filtering, and ordinary paging are sound, but tool execution can consume the finish reserve and continue after the request times out. Additional verified issues affect filtering, transport-error recovery, deadline enforcement, and tracing. This was a static review of SNAP only; no tests, server, or mutating commands were run.

## 2. Findings

1. **High - Tool execution can consume the finish reserve and outlive `/ask`.**

   - **Evidence:** `src/corpusqa/agent/agent.py:332`, `src/corpusqa/agent/agent.py:375`, `src/corpusqa/agent/tools.py:128`, `src/corpusqa/agent/corpus_tools.py:393`, `src/corpusqa/llm/client.py:234`, `src/corpusqa/config.py:322`.
   - **What goes wrong:** Only model calls receive the calculated timeout. Tool bodies have no deadline, and cancelling their `asyncio.to_thread` await does not stop the underlying work. Query embeddings reuse the ingestion retry policy: six attempts with 60-second HTTP timeouts and backoff.
   - **Failure scenario:** A search starts at elapsed time 105 seconds and its embedding takes 60 seconds. The agent never reaches forced finish; the server returns its timeout answer at 150 seconds while the worker continues. Queue slots are released despite that outstanding work, allowing subsequent requests to add more workers.
   - **Suggested fix:** Propagate an absolute deadline through tool execution and query embedding, reserving time for finish. Bound HTTP attempts and backoff by the remaining budget, and make worker operations cooperatively stop. Add coverage for slow tools and work remaining after request cancellation.
   - **Confidence:** high.

2. **High - Request-body reception is outside the arrival-based timeout.**

   - **Evidence:** `src/corpusqa/server/app.py:299`, `src/corpusqa/server/app.py:326`, `src/corpusqa/server/app.py:338`.
   - **What goes wrong:** The handler calculates a deadline immediately, but does not activate its timeout until after `await request.body()`. The revised arrival-based deadline therefore retains an unbounded body-read path.
   - **Failure scenario:** A client sends request headers and part of the JSON body, then leaves the connection open without completing it. `/ask` remains suspended beyond both 150 and 180 seconds.
   - **Suggested fix:** Put body reception inside the same absolute deadline as queueing and answering, with timeout handling outside that context. Add an ASGI test whose receive stream never completes.
   - **Confidence:** high.

3. **Medium - Chat spend persistence blocks the event loop outside the call timeout.**

   - **Evidence:** `src/corpusqa/llm/client.py:524`, `src/corpusqa/llm/client.py:533`, `src/corpusqa/llm/limits.py:124`, `src/corpusqa/llm/limits.py:149`.
   - **What goes wrong:** Every successful chat response synchronously creates directories, writes a file, and renames it from the event-loop thread. The model-call timeout has already ended, and the server’s asynchronous timeout cannot fire while that thread is blocked.
   - **Failure scenario:** A ledger write stalls on storage while four requests are active. All four requests, their timeout callbacks, and `/health` stop progressing until the write returns.
   - **Suggested fix:** Move persistence to a serialized worker while preserving atomic reservation and settlement. Test that a deliberately slow persistence operation does not block another request or its deadline.
   - **Confidence:** high.

4. **Medium - `contains` does not match each document’s complete field value.**

   - **Evidence:** `src/corpusqa/retrieval/values.py:44`, `src/corpusqa/retrieval/values.py:137`, `src/corpusqa/store/norms.py:36`, `src/corpusqa/retrieval/filtering.py:148`, `src/corpusqa/store/sqlite.py:562`.
   - **What goes wrong:** `loose()` calls `normalise()`, which truncates to 256 characters, so substring matching ignores later text. Matching any spelling then selects its entire normalized group through `norm_in`, including documents whose own spelling did not match.
   - **Failure scenario:** A metadata value containing `"needle"` after character 256 is missed. Conversely, the existing fixture groups `"Ottilie Brandvold"` and `"Ottilie Brandvold (Lead)"`; `contains("Lead")` selects all three documents in that person group, including spellings without `"Lead"`.
   - **Suggested fix:** Use an untruncated normalization for substring checks and select documents by their matching original values, rather than expanding a match to every normalized spelling. Add both suffix-match and mixed-spelling tests.
   - **Confidence:** high.

5. **Medium - Transport failures bypass compaction recovery and forced-finish retries.**

   - **Evidence:** `src/corpusqa/llm/client.py:578`, `src/corpusqa/llm/client.py:526`, `src/corpusqa/agent/agent.py:518`, `src/corpusqa/agent/agent.py:614`; `tests/unit/test_agent_compaction.py:258`.
   - **What goes wrong:** The client wraps API errors and built-in timeouts, but other transport exceptions escape as non-`LLMError` exceptions. The agent consequently treats an ordinary connection failure as a programming failure, bypassing its recovery paths.
   - **Failure scenario:** The SDK raises `httpx.ConnectError` during compaction. Instead of keeping the history and continuing, the exception reaches the server and produces the internal-error answer. The existing compaction tests inject an already-wrapped `ChatError`, so they miss this path.
   - **Suggested fix:** Translate expected transport failures into the appropriate `LLMError` subclass without broad-catching programming errors or cancellation. Preserve the policy of retrying only 429/5xx, and test transport failures during compaction and forced finish.
   - **Confidence:** high.

6. **Medium - Synthesized function-call IDs can collide with provider IDs.**

   - **Evidence:** `src/corpusqa/llm/client.py:437`, `src/corpusqa/llm/client.py:310`, `src/corpusqa/llm/client.py:322`; `tests/unit/test_llm_chat_client.py:329`.
   - **What goes wrong:** Missing IDs are filled using their call position without checking existing provider IDs. Conversion later infers whether an ID was provider-supplied solely through set membership.
   - **Failure scenario:** A reply contains an idless first call and a second call explicitly named `call_1`. Both become `call_1`; both responses are then sent with that provider ID, incorrectly attaching an ID to the originally idless call.
   - **Suggested fix:** Allocate synthetic IDs outside the complete set of provider IDs, and retain whether each ID was synthesized. Add a mixed-ID collision test.
   - **Confidence:** high.

7. **Medium - Search pages omit the exact totals required by this review’s stated contract.**

   - **Evidence:** `src/corpusqa/retrieval/search.py:208`, `src/corpusqa/agent/corpus_tools.py:415`; `docs/runs/2026-10-16-corpus-tools/decisions.md:26`; `tests/unit/test_corpus_tools.py:207`.
   - **What goes wrong:** Search stops ranking once enough documents exist for the requested page and deliberately omits the total while `exact=False`. This follows the snapshot’s documented search exception, but conflicts with the explicit requirement here that every returned list have an exact total.
   - **Failure scenario:** An ordinary first page returns `documents 1-8; next: offset=8`, with no total. The test explicitly expects this omission.
   - **Suggested fix:** Calculate the exact distinct-document total separately from ranking depth, preserving the existing stable page order. Update the schema and tests to require that total.
   - **Confidence:** high.

8. **Medium - Explicit null fields are reported as absent.**

   - **Evidence:** `src/corpusqa/ingest/fields.py:107`, `src/corpusqa/retrieval/documents.py:85`, `src/corpusqa/retrieval/documents.py:112`, `src/corpusqa/store/sqlite.py:669`; `tests/unit/test_store_tool_reads.py:111`.
   - **What goes wrong:** Ingest marks null-valued rows as noise, and the new readers discard every noise row before determining field presence. This collapses explicit null into absence, contrary to the promised empty-versus-absent distinction.
   - **Failure scenario:** Two documents share a known field: one contains `"approval": null`, while the other omits it. `read_fields` reports both as absent, and coverage counts neither as empty. Existing coverage tests use empty strings, leaving this case uncovered.
   - **Suggested fix:** Preserve explicit empty/null rows when determining presence and coverage, while continuing to exclude genuine noise values from matching. Test null, empty string, empty list, and absent separately.
   - **Confidence:** high.

9. **Medium - Failed model attempts lose required tracing details.**

   - **Evidence:** `src/corpusqa/llm/client.py:548`, `src/corpusqa/llm/client.py:549`, `src/corpusqa/llm/client.py:578`, `src/corpusqa/llm/client.py:587`.
   - **What goes wrong:** Response output is recorded only after `from_gemini()` succeeds. Retried errors are logged but not recorded as attempt details, and retry metadata is written only after success or terminal API failure.
   - **Failure scenario:** A `MALFORMED_FUNCTION_CALL` response is charged, then rejected before its output is traced. Alternatively, a 429 followed by timeout during backoff leaves an error generation without the attempted retry history.
   - **Suggested fix:** Record scrubbed response content and finish metadata before parsing. Record each attempt’s status and backoff as it happens, including cancellation paths. Extend tracing tests beyond successful retries and terminal 400 errors.
   - **Confidence:** high.

10. **Low - Citation tracing omits individual resolution and rejection decisions.**

    - **Evidence:** `src/corpusqa/agent/handles.py:92`, `src/corpusqa/agent/answerer.py:133`; `tests/unit/test_agent_answerer.py:216`.
    - **What goes wrong:** Resolution silently expands, drops, and deduplicates citations. The citation span records only the original list and final IDs, without the required per-citation disposition and reason.
    - **Failure scenario:** `finish` cites `["D1", "R1", "D999", "D1"]`. The trace cannot directly identify which IDs came from each handle, which citation was unknown, or which was removed as a duplicate.
    - **Suggested fix:** Record each citation’s expansion and disposition—kept, unknown, or duplicate—with its reason. Assert those records in the citation tracing test.
    - **Confidence:** high.

## 3. Checked and fine

- Each answer creates a fresh agent, session, and handle registry; handle mutations are locked.
- SQLite readers open and close a separate read-only connection per operation.
- SQL values are parameterized; FTS words are individually quoted; LIKE lookup patterns escape `%`, `_`, and backslashes.
- Citation resolution and the final response guard reject unknown IDs and deduplicate known IDs. The 500-ID cut is logged and traced.
- Search depth expansion preserves document order; normal paging reaches results without duplicates or skipped offsets.
- ToolBox preserves full results. Compaction keeps complete steps and retains history on blank replies or handled `LLMError` failures.
- Normal Gemini conversion preserves raw turns and thought signatures, groups tool responses in order, and correctly maps tool choice.
- The per-value date-window rule is applied during date selection and date-range filtering.
- Reviewed imports respect the store, SDK, and tracing boundaries.

## Triage (Claude, 2026-10-17)

Each finding was checked against the code at `cdc765eb`. Nothing is fixed yet: other
agents were changing this code during the review, and the fixes, plus the run's
`decisions.md` entries (Area: review), wait for the user. Severity after triage is in
brackets.

1. **Tool time not bounded by the deadline: agree [High].** Verified: only model calls get
   a timeout (`agent.py:305`, `client.py:524`). Tool bodies run in `asyncio.to_thread` with
   no limit (`tools.py:128`). A search tool call embeds its query with the ingest settings:
   a 60 s HTTP timeout, 5 retries, backoff up to 60 s (`client.py:234-248`,
   `config.py:322-326`). /ask still returns 200 with the timeout answer at 150 s, so the
   contract holds. But the agent's answer is lost. And the thread keeps running after the
   queue slot is freed, so more work than 4 answers can pile up. Fix: give tool calls the
   time left before the finish reserve, and give the query embedder its own short timeout
   with at most one retry.
2. **Request body read outside the deadline: partly agree [Low, not High].** True:
   `await request.body()` (`app.py:326`) runs before `asyncio.timeout_at` (`app.py:338`).
   But only the stalled client waits. It holds no queue slot (the slot is taken after the
   body is read), and no other request slows down. The grader sends whole bodies. Fix
   anyway, it is cheap: read the body inside the same `timeout_at(deadline)`.
3. **Spend ledger written on the event loop: agree [Low].** True: `SpendLedger.add`
   writes and renames a JSON file (`limits.py:130`, `limits.py:149-154`), called from
   async `GeminiChat.complete` (`client.py:533`). That breaks the AGENTS.md rule (no
   blocking I/O on the loop). In practice it is one small local file per model call. Fix:
   save the file in a thread; keep reserve and settle under the lock.
4. **`contains` misses text and pulls in other spellings: agree [Medium].** Verified:
   `loose()` calls `normalise()` (`values.py:46`), which cuts a value at
   `VALUE_NORM_MAX_CHARS = 256` (`norms.py:36`). So a word after character 256 of a value
   is never found, and the model is not told: a silent drop. Second: a match selects its
   whole `value_norm` group (`filtering.py:148-149`). Person values that differ only by a role
   in brackets share one group, so rows without the word come back too. Those rows show
   their values, but the count is wrong. Fix: match `contains` on the full value, and
   select documents by the spellings that matched.
5. **Network errors skip the agent's recovery: agree [Medium], and wider than stated.**
   Checked in the installed SDK (google-genai 2.31.0; aiohttp is not installed, so it uses
   httpx). With no retry options set, the SDK raises `httpx.ConnectError` /
   `httpx.ReadError` as they are (`_api_client.py:1791-1799`). `GeminiChat._call` catches
   only `APIError` (`client.py:578`). The agent catches only `LLMError` in every step
   (`agent.py:581`), not only in compaction and the forced finish. So one dropped
   connection at any step gives the internal-error answer. The same gap is in search:
   `_query_vector` catches only `LLMError` (`corpus_tools.py:394`), so a network error
   fails the whole search call (`'search' failed: ConnectError`) instead of falling back
   to keyword ranking. Fix: map httpx transport errors (not every exception) to
   `ChatError` / `EmbeddingError` in `llm/client.py`, with tests for a step, a compaction,
   a forced finish and a search.
6. **Filled-in call ids can collide: partly agree [Low].** True in code (`client.py:437`).
   It needs one reply that mixes calls with no id and a provider id spelled exactly
   `call_<n>`, which is very unlikely. Cheap fix: skip ids already used in the reply.
7. **Search pages without an exact total: disagree (user decision).** The user decided
   this on 2026-10-16 ("Search deeper", corpus-tools `decisions.md`): a search page shows
   no total until its ranking is complete, and always gives the next offset. Nothing is
   hidden: the model is told more follow, and where. With dense search every document in
   scope gets a rank, so an "exact total" would only repeat the scope size. Codex flagged
   it because the review prompt said "every list ... exact total", which is broader than
   the user's rule for search. No change unless the user wants it.
8. **A JSON null shown as "absent": agree on the fact [Low].** Verified: ingest marks
   every null value as noise (`ingest/fields.py:107`), and the readers skip noise rows
   (`sqlite.py:669`, `documents.py:86`). So `read_fields` says "(absent)" and coverage
   counts it as absent, not empty. Filters return the same ids either way (both mean "no
   value"); only the label and the coverage split are wrong. The `'null'` checks in
   `_VALUED` and `_EMPTY_TYPES` never fire as a result. Fix: keep null rows out of noise,
   with tests for null, empty string, empty list and absent.
9. **Failed model calls are traced thinly: agree [Medium].** Verified: the reply's output
   is put on the generation only after `from_gemini` succeeds (`client.py:548-551`), so a
   billed reply that is rejected (e.g. `MALFORMED_FUNCTION_CALL`) has no output in the
   trace. Retries are counted only at the end (`client.py:581`, `client.py:587`), so a
   timeout during a backoff loses them. This breaks "if it runs, it is traced". Fix:
   record the raw candidate and finish reason before parsing, and each retry as it
   happens.
10. **The citations span lacks a reason per citation: agree [Low].** Verified: the span
    holds the citations given and the ids returned (`answerer.py:133-145`). AGENTS.md asks
    for each citation kept or dropped, and why. Fix: record each citation's expansion and
    whether it was kept, unknown or a repeat.

Not raised by Codex (seen while checking):
- **[Low]** The `bm25` and `dense` spans record every ranked chunk up to the current depth
  (`search.py:195-201`). On a deep page over the whole corpus that is up to about 68,000
  entries per span. And `_traced` builds the list even when tracing is off.

After triage: 1 High (1), 3 Medium (4, 5, 9), 5 Low (2, 3, 6, 8, 10), 1 rejected (7).
Suggested order: 5 and 1 first (each can lose a whole answer), then 4, then 9.
