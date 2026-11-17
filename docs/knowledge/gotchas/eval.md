---
name: Eval gotchas
description: Traps in the eval tooling. Tests - test_no_hardcoding and test_eval_split once read the full dev-question file on every check (now one-way hashes in tests/fixtures/forbidden_tokens.sha256 and the tune/smoke id files only). /ask eval runner (eval/run.py) - per-question trace spans of protected (heldout, all) runs leaked per-question results into Langfuse; extra args after a `uv run` key could override its `--set` (argparse keeps the last value); deeply nested JSON raised RecursionError and aborted the whole run. Retrieval eval - a screened (subset) index was reused for a question set with other gold documents until its manifest held a digest of the kept ids; NaN and infinite fusion weights passed a `< 0` check; `:g` gave different weights the same run name; older reports without per-question recall could not be rendered (now `n/a`).
source: runs/2026-10-14-ingestion, runs/2026-10-16-eval-weights-headers, runs/2026-10-16-ask-eval-runner, runs/2026-10-14-ai-dev-setup
---

# Eval gotchas

## 2026-10-16 Trace spans of protected runs held per-question results (Codex finding 1)
- **Area:** eval / tracing (`eval/run.py`, `ask_one` and `span_metadata`)
- **Symptom:** for heldout and all, every `question` span recorded the question id, its
  scores, expected and hit counts, the error text and the server trace id. Langfuse is
  readable by agents, so held-out per-question results left the guarded folder.
- **Cause:** the protected-set rule was applied to the files, stdout and the parent
  trace output only; the child spans were written the same way for every set, and the
  tracing test used the tune set only.
- **Fix (bc796eea):** `ask_all(..., protected=True)` makes each span keep only status,
  latency and a failed flag; a failed span says only that the reason is in the guarded
  file (`PROTECTED_SPAN_ERROR`). Test: `test_protected_runs_trace_no_per_question_results`
  (checks every observation).
- **Avoid next time:** a "no per-question output" rule covers every output, the trace's
  child observations included; test it on the protected set itself.

## 2026-10-16 Extra args after a `uv run` key could override the key's `--set` (Codex finding 2)
- **Area:** eval / commands (`eval/run.py` `parse_args`, `src/corpusqa/devtasks.py`)
- **Symptom:** `uv run eval-smoke --set all --milestone m1` would run the whole golden set
  through an allowlisted key, without the user's approval.
- **Cause:** devtasks appends extra args after the key's fixed args, and argparse keeps the
  last value of a repeated option.
- **Fix (bc796eea):** `--set` refuses a second value (`SetOnce` action). Test:
  `test_a_key_cannot_switch_to_another_set`.
- **Avoid next time:** an option a key fixes must not be overridable by appended args.

## 2026-10-16 Deeply nested JSON raises RecursionError, not ValueError (Codex finding 3)
- **Area:** eval / reply parsing (`eval/run.py` `read_reply`)
- **Symptom:** a 200 KB body of nested arrays (under the 1 MiB cap) would abort the whole
  run and lose the report and history row.
- **Cause:** `json.loads` raises RecursionError on deep nesting; only ValueError was
  caught, so the error escaped `asyncio.gather`.
- **Fix (bc796eea):** catch both; the reply reads as "body is not JSON" and the run goes
  on. Test: `test_deeply_nested_json_is_a_failed_reply_and_the_run_goes_on`.
- **Avoid next time:** parsing untrusted JSON: catch `(ValueError, RecursionError)`.

## 2026-10-16 Older reports without q_recall could not be rendered (Codex finding 3)
- **Area:** eval / `eval/retrieval.py` report
- **Symptom:** rendering a `report.json` written before per-question recall existed
  failed: its scores have no `q_recall@k` keys.
- **Cause:** the renderer read every metric key it now writes; only the fusion weights
  had a fallback for old reports (they read as 1 / 1).
- **Fix (ba616c5a):** a missing metric cell reads `n/a`. Test: the old-schema case in
  `test_report_and_history_line_render_without_dense`.
- **Avoid next time:** when a report gains a key, give the reader a default for old
  files in the same change.

## 2026-10-16 `:g` gave different fusion weights the same run name (Codex finding 2)
- **Area:** eval / run names in `eval/history.csv`
- **Symptom:** two weights that differ only after the sixth significant digit got the
  same `-w<bm25>x<dense>` name, and -0.0 and 0 got different names, so history rows could
  not be told apart reliably.
- **Cause:** the suffix used `:g` formatting, which rounds to 6 significant digits and
  keeps the sign of -0.0.
- **Fix (678373cf):** a weight keeps its `:g` name when that reads back exactly (old
  names such as `-w0.5x1` stay the same) and its `repr` otherwise; -0.0 is named 0. Test:
  `test_fusion_suffix_tells_close_weights_apart_and_names_zero_once`.
- **Avoid next time:** a name used as a key must round-trip to the value it names.

## 2026-10-16 NaN and infinite fusion weights passed validation (Codex finding 1)
- **Area:** eval / retrieval (`parse_args`, `HybridRetriever`)
- **Symptom:** `--bm25-weight nan` (or `inf`, `1e999`) was accepted.
- **Cause:** the check only refused negative values; every comparison with NaN is
  False, and inf is not negative.
- **Fix (aac0c168):** both places now need a finite weight >= 0 (`math.isfinite`).
  Tests: `test_fusion_flags_reject_non_finite_weights`, `test_fusion_weights_must_be_finite`.
- **Avoid next time:** validate a float from the user with `math.isfinite` first.

## 2026-10-16 A screened eval index was reused for other gold documents (B5 Codex finding 10)
- **Area:** eval / `eval/retrieval.py` caches
- **Symptom:** a variant index built on a screened subset of documents (kept for one question set's gold documents) could be reused for another question set, so its gold documents might be missing from the index.
- **Cause:** the variant manifest keyed only the corpus and code, not which documents were kept.
- **Fix:** the manifest holds a digest of the kept document ids (`screen_docs`; None for full-corpus variants, as old manifests read it, so existing caches stay valid). Test: `test_a_screened_index_is_never_reused_for_other_gold_documents`.
- **Avoid next time:** a cache key must cover every input that changes the result, including the selection of documents.

## 2026-10-15 Our own checks broke the eval split rule
- **Area:** eval / tests
- **Symptom:** the plan reviewer (Codex) refused to run the tests: `test_no_hardcoding.py`
  and `test_eval_split.py` read the full dev-question file (all 79 questions, the 49
  held-out among them) on every `uv run check`, and a failing guard would print a held-out
  token.
- **Cause:** the guard was written before the split existed and loaded the file at test time.
- **Fix:** `scripts/make_forbidden_hashes.py` (run once by the coordinator) writes
  `tests/fixtures/forbidden_tokens.sha256` (436 one-way hashes); `tests/forbidden_tokens.py`
  holds the token extraction both sides share; the test hashes the tokens found in `src/`
  and compares (a planted key is still caught). The split test uses only the `eval/sets/`
  id files.
- **Avoid next time:** when a rule like the split is added, re-check every existing test and
  script against it.
