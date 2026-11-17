---
name: Codex review — fusion weights, per-question recall, header fields (backlog items 1 and 3)
description: Independent read-only Codex review (gpt-6-astra, effort high, 2026-10-16) of git diff ingest/build..retrieval/eval-weights-headers at bfb0e636: 3 Medium findings (non-finite fusion weights, run-name collisions from float formatting, old reports without q_recall columns). All 3 fixed; the triage of each finding is in docs/knowledge/decisions/retrieval.md ("2026-10-16 Codex review of fusion weights, per-question recall and header fields") and the traps in docs/knowledge/gotchas/eval.md. Read before changing fusion, the retrieval eval CLI or report, or chunk-header fields.
---

# Codex review: fusion weights, per-question recall, header fields (2026-10-16)

Command: `codex exec --sandbox read-only -m gpt-6-astra -c model_reasoning_effort="high" -o <file> -` with the prompt below on stdin (scope: `git diff ingest/build..HEAD` at bfb0e636, steps 1-7a; the later flip of `CHUNK_HEADER_FIELDS` is one constant and test updates). Output pasted unchanged.

<details><summary>Prompt</summary>

You are an independent senior reviewer. Review one branch of this repo. This is a READ-ONLY review: make NO code changes, write NO files, run no command that changes the repo, its caches or git state. Do not run `uv run eval-*` commands (they spend money and write history rows).

## What to review
The diff `git diff ingest/build..HEAD` on the checked-out branch `retrieval/eval-weights-headers` (backlog items 1 and 3). Focus on these files:
- `src/corpusqa/retrieval/hybrid.py` (weighted RRF: `rrf()` takes one weight per ranked list; `HybridRetriever` bm25_weight / dense_weight; span fields)
- `eval/retrieval_metrics.py` (new per-question recall `q_recall@k` beside pooled `recall@k`)
- `eval/retrieval.py` (flags `--bm25-weight --dense-weight --rrf-k --fields`, the `Fusion` suffix in run names, the variant cache folder and manifest, the report and history headline)
- `src/corpusqa/config.py` (new constants: RRF weights, CHUNK_HEADER_FIELDS, HEADER_FIELD_*)
- `src/corpusqa/ingest/chunking.py` (`select_header_fields` from the field profile; `ChunkSpec.fields` / `header_fields`; `ChunkInput.fields`; `make_header` + `render_header_fields`; caps)
- `src/corpusqa/ingest/pipeline.py` (S6 `_chunk` passes header fields; span metadata)
- `src/corpusqa/store/interfaces.py`, `src/corpusqa/store/sqlite.py` (`FieldText`, `DocumentText.fields`, `field_profiles()`, `document_texts(header_fields=...)`, the shared paged generator `_document_texts`)
- `commands.json` and the tests: `tests/unit/test_header_fields.py`, `tests/unit/test_retrieval_eval.py`, `tests/unit/test_chunk_index.py`, `tests/unit/test_ingest_chunks_embed.py`.
Skip the eval report JSON/MD files and `eval/history.csv` except as evidence.

Rules and intent (read these first):
- `AGENTS.md` (hard rules, import boundaries, tracing rules, no hard-coding from data/).
- `docs/runs/2026-10-16-eval-weights-headers/plan.md` (the design) and `docs/runs/2026-10-16-eval-weights-headers/decisions.md` (later entries override the plan; newest first).
- `docs/backlog.md` items 1 and 3 (the motivation).

Out of scope, do not report: no changes under `src/corpusqa/agent/` or `src/corpusqa/prompts/` (backlog item 2 comes later, by the user's choice); `CHUNK_HEADER_FIELDS = False` in production is deliberate until the eval decides; the fusion defaults (1.0 / 1.0 / k 60) are deliberately unchanged until the user picks.

## Do NOT open these (eval split rule, hooks enforce it)
`data/questions.jsonl`, `eval/sets/heldout.json`, anything under `eval/results/heldout/`, and `.env` (secrets). Do not use wildcards into `eval/sets/` or `eval/results/`. You may read `data/corpus/` files if you need evidence of the format, but nothing from `data/` may appear in code.

## Focus areas, in this order
1. Correctness of weighted RRF: score formula, weight-0 semantics (ids found only by a weight-0 list are kept with score 0 and sort last), tie order, validation (negative weights, length mismatch), that default weights reproduce the old unweighted ranking exactly.
2. Correctness of per-question recall: the definition (mean over questions of found / gold at k), duplicate ids, gold documents missing from the index, k cut-offs, the history headline change and backward compatibility with old rows and old reports.
3. Eval CLI: parse-time validation, the run-name suffix (could two different settings get the same name, or one setting two names, e.g. float formatting?), that weights and k never change the cached chunk index or its manifest, that `--fields` changes the variant name AND the manifest so a stale index is never reused, `--fields` with `--production` / `--no-header` rejected.
4. Header-field selection: generality for an unseen corpus of the same format (no names from data/), thresholds and kinds, determinism across builds, behaviour when the profile is empty or a source has no fields.
5. Store reads: SQL correctness, paging by rowid (off-by-one, gaps, last page), id batches, ordering by mapping order then seq, noise / empty / null / json filtering, the reader now pages (behaviour change), whether the build writer holds an open read cursor while S6 writes chunks, connection use across threads.
6. Rendering: one line, caps (items per field, chars per value, total chars), whole-part cut, blank values, determinism; whether the header's extra length can push embedding inputs past model limits or distort chunk token counts and sizes.
7. Production effects: when `CHUNK_HEADER_FIELDS` flips, does the production index cache fingerprint change so the index rebuilds (and not reuse stale headers)? With it False, are production chunks byte-identical to before?
8. Tracing (AGENTS.md: if it runs, it is traced): the new span fields on `retrieve` and S6; anything missing.
9. Tests: gaps, and tests that would still pass with a real bug.

## Output format
Plain Markdown, no preamble:
`## 1. Verdict` (3-5 sentences), then `## 2. Findings`: a numbered list. Each finding:
`N. **<Severity: Critical | High | Medium | Low> — <one-line title>.**` then `Evidence:` with file:line references (and a command output or a short code quote where it helps), then `Why it matters:` (one or two sentences), then `Suggested fix:` (concrete). Order by severity. Report only real problems you verified in the code; say how confident you are when unsure. Then `## 3. Checked and fine` with a short list of the things you checked and found correct.

</details>

## 1. Verdict

I recommend changes before approval: three verified issues affect weight validation, run identification, and compatibility with older reports. The core weighted RRF formula, per-question recall, cache separation, and production header path are otherwise sound. Verification used read-only inspection and Python checks with bytecode disabled, including SQLite tests entirely in memory. No evals, full test suite, or file writes were performed.

## 2. Findings

1. **Medium — Non-finite weights pass validation and corrupt fusion ranking.**

   **Evidence:** [eval/retrieval.py:858](eval/retrieval.py:858) parses weights using `float`, then rejects only values below zero. [hybrid.py:80](src/corpusqa/retrieval/hybrid.py:80) repeats that incomplete check. In-memory reproduction:
   
   ```text
   parse weight 'nan': accepted nan
   parse weight 'inf': accepted inf
   parse weight '1e999': accepted inf
   weight 1.0: ('c', 'b', 'a')
   weight inf: ('a', 'b', 'c')
   ```

   With infinite weight, every hit receives an infinite score and ranking collapses to ID order. NaN makes score comparisons unordered.

   **Why it matters:** Invalid settings proceed through potentially expensive index construction and embedding, then produce misleading evaluation scores instead of a usage error.

   **Suggested fix:** Require finite, nonnegative weights using `math.isfinite()` in CLI validation and the retriever API. Add tests covering `nan`, positive infinity, and overflowing float input for both weights.

2. **Medium — Float formatting gives different fusion settings the same run name.**

   **Evidence:** [eval/retrieval.py:182](eval/retrieval.py:182) formats weights with `:g`, which defaults to six significant digits. Verified:
   
   ```text
   Fusion(1.0000001, 1.00000015).suffix == '-w1x1'
   Fusion(1.0000002, 1.00000015).suffix == '-w1x1'
   ```

   For rankings `[[2], [1]]`, those settings produce `[1, 2]` and `[2, 1]`, respectively. Conversely, equivalent zero weights produce different suffixes: `-w0x1` and `-w-0x1`.

   **Why it matters:** The report/history variant label cannot reliably identify the configuration, even when the configurations produce different rankings. The existing suffix tests cover only simple, exactly representable weights.

   **Suggested fix:** Use a round-trip float representation, such as `repr()`, and normalize signed zero before formatting. Add tests for close unequal weights and equivalent textual inputs.

3. **Medium — Older reports cannot be rendered after adding per-question columns.**

   **Evidence:** [eval/retrieval.py:723](eval/retrieval.py:723) unconditionally adds `q_recall` columns; [eval/retrieval.py:742](eval/retrieval.py:742) indexes every column using `s[c]`. Rendering the existing test fixture after removing the new metric keys reproduces:
   
   ```text
   KeyError 'q_recall@5'
   ```

   The compatibility assertion at [test_retrieval_eval.py:196](tests/unit/test_retrieval_eval.py:196) omits old reports’ weights but retains their newly introduced `q_recall` keys, so it misses this failure.

   **Why it matters:** Although missing fusion weights have a backward-compatible fallback, genuine pre-change report JSON still fails to render.

   **Suggested fix:** Render unavailable metrics as `n/a`, omit unavailable columns, or derive per-question recall from stored gold ranks. Add a fixture matching the complete old schema, without weights or `q_recall` keys.

## 3. Checked and fine

- Default RRF matched the original formula across 1,000 generated cases. Zero-weight hits remain present; ties use ascending IDs; weight-count mismatch and negative retriever weights are rejected.
- Per-question recall handles missing gold documents, duplicate hits, cutoffs, and questions without gold correctly. Pooled recall retains its separate name and meaning.
- Fusion arguments stay outside chunk-cache identity. Header fields affect both variant naming and manifest contents; incompatible `--fields` combinations are rejected.
- Header selection follows profile thresholds and kinds, with deterministic ordering and sensible empty-profile behavior.
- In-memory store checks passed for rowid gaps, partial final pages, ID batching, field ordering, filtering, and interleaved writes.
- Rendering respects field caps and preserves body text/token accounting. Disabled fields preserve prior header behavior; changing the production switch invalidates the index fingerprint.
- Retrieval and S6 tracing include the new settings and selection information, with corresponding assertions in tests.