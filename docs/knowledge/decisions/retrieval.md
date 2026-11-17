---
name: Retrieval decisions
description: Decisions about retrieval and its eval: the agent's search (built 2026-10-16 in retrieval/search.py - scoped hybrid, later pages search deeper with the list growing only at its end, 2 passages of ~300 characters instead of whole chunks), value matching for the filter (exact, else one loose match, else stop and list; "near" at 0.75); numpy for the vector matrix; the hybrid retriever (FTS5 BM25 with every question word quoted and OR-ed, header weight 0.5; cosine dense; weighted RRF, bm25 1.0 / dense 1.0 / k 60, over the top 50 chunks each; a weight-0 list keeps its hits last; weights must be finite and >= 0; chunks mapped to documents); `uv run eval-retrieval` (tune/smoke only; per-question recall `q_recall@k` is the headline beside pooled recall@k, all-gold@10, MRR; `--bm25-weight/--dense-weight/--rrf-k` with a run-name suffix and no index rebuild; caches; eval/history.csv columns); the weight grid (no setting clearly beats the defaults; OPEN for the user: k 60 or k 20); the Codex review of that work; spend cap $60; plus what is still open; exact phrases and keys in the agent's search kept after smoke run 3 (SEARCH_EXACT_PHRASES = True, tools.json v5).
source: runs/2026-10-14-ingestion, runs/2026-10-16-eval-weights-headers, runs/2026-10-16-corpus-tools, runs/2026-10-16-agent-tools
---

# Retrieval decisions

## Left open for the retrieval step (findings, not decisions)
- Update 2026-10-16 (runs/2026-10-16-corpus-tools): the agent's search
  (`retrieval/search.py`, `../architecture/corpus-tools.md`) ranks with this hybrid inside
  a scope. BM25 query handling and a document-level vector are still open.
- **OPEN (user): RRF k 60 (default) or 20.** See the entry "2026-10-16 Fusion defaults
  stay bm25 1 / dense 1 / k 60" below. Plain headers: k 20 gives +0.019 hybrid
  q_recall@10 (half a question). With header fields on: the same q_recall@10, MRR +0.022.
- Update 2026-10-16 (runs/2026-10-16-eval-weights-headers): fusion weights can now be
  tuned and were tried; lowering the bm25 weight lost recall at every value. The
  production index now has field values in its chunk headers (chunking.md#2026-10-16
  Header fields go on in production): hybrid q_recall@10 0.727, pooled R@10 0.403, MRR
  0.671, all-gold@10 0.692; bm25 q_recall@10 0.603, dense 0.696. BM25 query handling and
  a document-level vector are still open.

From chunking.md#2026-10-16 Step C result and the B4 production run (tune set only):
- Production index: hybrid R@10 0.377, R@5 0.351, R@20 0.403, all-gold@10 0.615, MRR
  0.588; BM25 alone R@10 0.312; dense alone R@10 0.364.
- BM25 is weak (R@10 ≈ 0.30) and RRF fusion with it pulls a stronger dense ranking down
  (dense-256 0.429 → hybrid 0.390): fusion weights and BM25 query handling are open.
- Document-level vectors rank the first gold document best (whole-document dense MRR
  0.691 vs 0.626 for 512-token chunks): a document vector beside the chunks is worth a try.

## 2026-10-17 Keep S: exact phrases and keys in search (SEARCH_EXACT_PHRASES = True, tools.json v5)
- **Area:** retrieval (search) / prompts (tools.json)
- **Chose:** turn S on for good (`ed587baa`): the keyword side (`store/sqlite.py`
  `match_expression`) matches text in double quotes as a phrase and a key-shaped token
  (letters or digits joined by `- _ / . #`) as the phrase of its parts; other words still
  match one by one; capital AND / OR are not searched for. The page marks documents only the
  dense side found ("meaning only") and says when no document in scope matched. One test
  line moved with it (`test_search_falls_back_to_keywords_when_the_query_cannot_be_embedded`:
  "no documents" is now line 3; the assertion is the same). Smoke run 3 against run 2: ids,
  abstain, answers (9 / 1 / 0) and facts (45) the same; model calls 43 (50), p50 7.7 s (8.4),
  $0.383 ($0.410); 4 of 19 searches said "nothing found" (run 2: 0 of 13).
- **Over:** reverting to loose words (no score dropped, and only exact matching can tell the
  model that a key or phrase is written nowhere).
- **Why:** run 1: 20 of 22 wasted calls were quoted searches that the keyword side split
  into OR-ed words, so a key written nowhere still filled the page. Not solved by it:
  pub_0028 got three "nothing found" pages and still searched to step 10.

## 2026-10-16 Later search pages search deeper (the user's rule replaces the fixed depth)
- **Area:** search, store, tools, prompts
- **Chose** (user, 2026-10-16: "Search deeper"):
  - The first ranking is as before: each ranker's best `RETRIEVAL_TOP_CHUNKS` (50) chunks
    in scope, RRF k 60, weights 1.0 / 1.0. Page 1 is unchanged.
  - A page past what the list holds re-ranks with `SEARCH_DEPTH_GROWTH` (2) times more
    chunks per ranker, again, until the list has more documents than the page needs (so a
    next page is known to exist) or every chunk in scope is ranked (depth >= the chunks in
    scope, or both rankers returned fewer than asked). A `within=` set bounds it by its
    chunks.
  - The list only grows at its end: a deeper ranking adds its new documents in its own
    fused order; documents found before keep their places. Passages come from the ranked
    chunks in that same order (each chunk once).
  - Both rankers break ties by chunk id (BM25 `ORDER BY score, rowid`; `VectorIndex` keeps
    every row tied at the cut, then sorts by score and id), so a shallow ranking is the
    start of a deeper one: the list depends only on the query and the scope, never on the
    page asked for.
  - Page line: "documents 1-8; next: offset=8" while the ranking has not reached every
    chunk in scope (no total); "documents 9-16 of 20; next: offset=16" once it has (exact);
    "... (last page)" at the end; "no documents". The head says `search "q" (scope): best
    first.` `tools.json` v4: the search description says how pages go deeper.
  - Tracing: the `search` span's metadata has `depth`, `rankings`, `complete`, its output
    `exact`; one `bm25` / `dense` span per ranking, input with its `depth`.
- **Over:** a fixed list of each ranker's top 50 chunks as the whole result (the first
  build: pages stopped after 29-47 documents on data/ probes while the line said "of N");
  paging a fresh deeper ranking (pages could repeat or skip documents); ranking every chunk
  in scope from page 1 (changes page 1 against the validated hybrid, and costs more); an
  exact total up front (with dense it is every document in scope with a vector, and BM25
  would read every match).
- **Why:** the user's decision.
- **Measured** (by hand, production index read-only, keyword ranking): offset 0 39 ms, 48
  18 ms, 400 43 ms, 3,000 85 ms; source slack offset 200 53 ms.

## 2026-10-16 "Near" needs a score of 0.75
- **Area:** tools
- **Chose:** `NEAR_MIN_RATIO = 0.75`; one of two words shared scores 0.75, containment 0.9;
  near items first, the rest in their normal order.
- **Over:** 0.6 (the first value).
- **Why:** by-hand check on the production index: at 0.6 "stage" was near "tags" and
  "qps_target". A typo ("staus") still passes at 0.75.

## 2026-10-16 Value resolution: exact, else one loose match, else stop
- **Area:** tools, filter
- **Chose:** an exact (normalised) match is used and other spellings with the same loose
  form are named as "not included"; else one loose match is used and reported; several
  loose matches raise `ToolInputError` listing all with counts (nothing runs); none raises
  `ToolInputError` with the 5 nearest values. `eq` on a date field = that day/month/year;
  `eq` on a number field = that number. `missing` = no value (empty or absent); every
  condition reports value / empty / in body / absent counts.
- **Over:** merging spellings silently; running the exact value without naming the rest.
- **Why:** the user's rule "several close values all listed, never merged silently". On
  data/ `50k-150k` (31) and `$50k-$150k` (20) are both real.

## 2026-10-16 Passages, not whole chunks, in search results
- **Area:** search
- **Chose:** at most 2 passages per document (its best chunks in fused order), each a
  ~300-character window around the most query words (`SEARCH_SNIPPETS_PER_DOC`,
  `SEARCH_SNIPPET_CHARS`).
- **Over:** whole chunks (up to ~2,000 characters each, 8 x 2 per page).
- **Why:** a page stays near 4,000 characters; `read_document` gives the whole document,
  so nothing is lost. The size was put to the user as an open question; no answer is
  recorded, and every eval since ran with it (`docs/backlog.md`).

## 2026-10-16 Codex review of fusion weights, per-question recall and header fields: 3 findings, all fixed
- **Area:** review (review file `docs/reviews/2026-10-16-eval-weights-headers.md`)
- **Scope and how:** `git diff ingest/build..HEAD` on `retrieval/eval-weights-headers` at
  bfb0e636 (backlog items 1 and 3, before the `CHUNK_HEADER_FIELDS` flip).
  `codex exec --sandbox read-only -m gpt-6-astra -c model_reasoning_effort="high"`.
  Verdict: core RRF, per-question recall, cache separation and the production header path
  sound; 3 Medium findings. Each fix started with a test that failed before it.
  1. **NaN / infinite fusion weights passed validation — FIXED (aac0c168).** `parse_args`
     and `HybridRetriever` now need finite weights >= 0 (`math.isfinite`). Tests:
     `test_fusion_flags_reject_non_finite_weights`, `test_fusion_weights_must_be_finite`.
  2. **`:g` gave different weights the same run name; `-0` vs `0` — FIXED in part
     (678373cf).** A weight keeps its `:g` name when that reads back exactly (so names
     already in `eval/history.csv`, like `-w0.5x1`, stay the same) and its `repr`
     otherwise; -0.0 is named 0. Not Codex's `repr()` everywhere (it would rename old
     rows). Test: `test_fusion_suffix_tells_close_weights_apart_and_names_zero_once`.
  3. **Old reports without q_recall keys failed to render — FIXED (ba616c5a).** Missing
     metric cells read `n/a`. Test: the old-schema case in
     `test_report_and_history_line_render_without_dense`.
- No eval rerun: no fix changes the ranking for a valid setting.

## 2026-10-16 Fusion defaults stay bm25 1 / dense 1 / k 60; k 20 is the candidate
- **Area:** retrieval / eval
- **Chose:** leave `RRF_BM25_WEIGHT`, `RRF_DENSE_WEIGHT`, `RRF_K` as they are and offer
  `RRF_K = 20` to the user as the one candidate (OPEN, see the top of this file). The
  weight grid on the production index (plain headers; tune set, 26 questions with gold;
  one question's worth = 1/26 = 0.038; all from the embedding cache, $0):

  | run (fusion) | hybrid q_recall@10 | pooled recall@10 | hybrid MRR | all_gold@10 |
  |---|---|---|---|---|
  | baseline bm25 1, dense 1, k 60 | 0.6731 | 0.3766 | 0.5882 | 0.6154 |
  | `--bm25-weight 0.5` | 0.6538 | 0.3766 | 0.5827 | 0.6154 |
  | `--bm25-weight 0.25` | 0.6538 | 0.3766 | 0.6179 | 0.6154 |
  | `--bm25-weight 0` (= dense order) | 0.6346 | 0.3636 | 0.6146 | 0.5769 |
  | `--rrf-k 20` | 0.6923 | 0.3896 | 0.5959 | 0.6538 |
  | `--rrf-k 120` | 0.6731 | 0.3766 | 0.5882 | 0.6154 |

  With header fields on (chunking.md), k 20 keeps hybrid q_recall@10 at 0.7271 and lifts
  MRR 0.6713 -> 0.6932 (`recursive-512-fields-k20`).
- **Over:** lowering the bm25 weight (loses half a question of q_recall@10 at every value
  tried); changing `config.py` now.
- **Why:** k 20 is the only setting where no hybrid metric falls, but its q_recall@10
  gain (+0.019) is half of one question's worth (0.038), the same noise bar the
  header-fields keep rule uses. The user decides the defaults.

## 2026-10-16 The eval CLI rejects bad fusion values before it builds anything
- **Area:** eval
- **Chose:** `parse_args` exits with a usage error for a negative `--bm25-weight`,
  `--dense-weight` or `--rrf-k` (0 is allowed: k = 0 is RRF with 1/rank). After the
  Codex review it also rejects NaN and infinite weights.
- **Over:** leaving it to `HybridRetriever` (which also rejects them).
- **Why:** `run()` builds the retriever only after the index is built and embedded; a
  typo would otherwise fail after minutes and possibly money. A negative k can also
  divide by zero (k + rank = 0), which nothing else checks.

## 2026-10-16 Weighted RRF; a zero weight keeps the ranker's hits, after every scored one
- **Area:** retrieval
- **Chose:** `rrf()` takes one weight per ranked list: score = sum of
  `weight / (k + rank)` (ranks from 1; None = 1.0 each; a length mismatch raises).
  `HybridRetriever` takes keyword-only `bm25_weight`, `dense_weight` (defaults
  `RRF_BM25_WEIGHT` = `RRF_DENSE_WEIGHT` = 1.0 in `config.py`, so results did not change)
  and records both on its `retrieve` span. An id found only by weight-0 lists is kept with
  score 0, so it sorts after every scored id (ties by the lower id, as before). A
  negative (and, after the review, a non-finite) weight raises `ValueError`; 0 is allowed.
- **Over:** dropping weight-0 hits (hybrid would then hold fewer documents than bm25 and
  dense together, and recall@20 could fall for a reason that is not ranking); allowing
  negative weights (they would rank a ranker's best hits last).
- **Why:** backlog item 1 (the eval must be able to tune bm25 against dense). A weight
  then only changes the order, never the set of documents, so a weight grid compares
  like with like.

## 2026-10-16 Weights and k are query-step settings: not in the variant index manifest
- **Area:** eval / retrieval
- **Chose:** `--bm25-weight`, `--dense-weight`, `--rrf-k` (defaults from `config.py`)
  reach `HybridRetriever` only, through `Fusion`; the chunk index folder and manifest
  ignore them. The report folder, `result["variant"]` and the history row use
  `<index name><Fusion.suffix>`: `-w<bm25>x<dense>` and `-k<k>` only when not default
  (e.g. `production-recursive-512-w0.5x1-k20`). `result["retriever"]` records all three;
  `report.md` prints the weights (old reports without them read as 1 / 1).
- **Over:** a new variant folder per weight setting (would re-embed nothing but would
  copy the index).
- **Why:** backlog item 1; a cached index is still reused, so a grid point costs $0.

## 2026-10-16 Per-question recall `q_recall@k` is the headline; pooled recall stays beside it
- **Area:** eval
- **Chose:** `ModeScores.q_recall`: for each question, its gold documents found in the
  top k divided by its gold documents, then the mean over questions. It sits beside the
  pooled `recall@k` (all found / all gold) in `report.json`, `report.md` (one sentence
  explains each) and the history `metrics` string (`q_recall@5/10/20`). The history
  headline is `hybrid_q_recall@10` (`bm25_q_recall@10` without dense), from `headline()`
  in `eval/retrieval.py`. Rows written before this change (5caf2fda) carry the pooled
  `hybrid_recall@10`.
- **Over:** redefining `recall@k` (old history rows would read as the new metric);
  keeping pooled recall as the headline.
- **Why:** pooled recall lets a question with many gold documents outweigh the rest (one
  14-gold question weighs as much as 14 one-gold questions), so it mostly measures a few
  "list them all" questions. Per-question recall counts each question once. A new name
  keeps new rows from being mistaken for old pooled ones (backlog item 1). Production, plain headers: pooled 0.377, per-question
  0.673 (the backlog's hand count).

## 2026-10-16 C4: chunk store, eval-only retriever and `eval-retrieval` (readings of the brief)
- **Area:** store / retrieval / eval; dependencies; spend
- **Superseded in part by:** dates.md#2026-10-16 Dates are judged one by one (item 6: there is no `_low_trust` rule any more; a date outside the window is no date and never in a header); retrieval.md#2026-10-16 Per-question recall `q_recall@k` is the headline (items 11 and 12: the history headline is now `hybrid_q_recall@10`; pooled recall@k stays beside it); retrieval.md#2026-10-16 Weighted RRF (item 7: one weight per list, defaults 1.0 / 1.0); store.md#2026-10-16 B4 (this entry's item 1: `VectorIndex` now lives in `store/vectors.py`); ingestion.md#2026-10-16 B4 (this entry's item 3: the chunk tables are now in the document index; item 9: one embedding cache `.cache/embeddings/vectors.sqlite` for start-up and the eval, same key); ingestion.md#2026-10-16 B3 (this entry's item 10: the eval's document index is the pipeline's own index, `runtime.index_path`).
- **Chose:**
  1. **New runtime dependency `numpy` (2.5.4)**, added with `uv add` (design decision 8, approved). The dense side holds all chunk vectors as one float32 matrix and ranks by one matrix-vector product (68K x 768 in a few ms). Replaces nothing: there was no vector code; pure-Python dot products over 68K vectors per query would take seconds. Used only in `retrieval/vectors.py` (the "small module").
  2. **Spend cap `EMBED_SPEND_CAP_USD` $25 -> $60** (total across runs, same ledger). 7 variants on the full corpus were ~165M tokens (~$33) plus the probes. One constant, not an eval-only one, so the ledger has one cap.
  3. Schema: `CHUNK_DDL` is a separate group in `store/schema.py` (chunks, chunks_fts, embeddings), not yet in the document index. Deviations from design §5: `label`, `speaker`, `block_date` dropped (they came from the structure-aware split, not chosen); `level` added; `header` is `''` when headers are off; index `chunks_doc` added.
  4. Store: `SqliteChunkWriter` writes a file holding only the chunk tables (the eval's per-variant index) with the same `.building` -> rename life cycle; foreign keys stay off there. `SqliteChunkReader` (chunk_doc_ids, bm25, embeddings) works on any published file with the chunk tables. `SqliteVectorCache` (WITHOUT ROWID table `vector_cache`). The SQL helpers `insert_chunks`, `rebuild_chunk_fts`, `insert_embeddings` take a connection so the build writer reuses them.
  5. BM25: FTS5 query = every word of the question (unicode61 word rule: letters and digits) quoted and joined by OR, so the user's text can never be read as FTS5 syntax; no stopword list (IDF handles common words). Column weights header 0.5, text 1.0. Scores negated so higher is better.
  6. Headers: a created date whose rule ends in `_low_trust` is left out of the chunk header (`ingest.chunking.chunk_input`). Embedded text = `header + "\n" + text`, or the text alone when headers are off.
  7. Retriever (`retrieval/hybrid.py`): top 50 chunks per ranker, RRF k = 60 with 1-based ranks, ties to the lower chunk id; chunk lists mapped to documents keeping each document's best rank. Without vectors, hybrid = bm25. Spans `retrieve` > `bm25`, `dense`.
  8. The `Embedder` protocol moved to `llm/interfaces.py`: the "Gemini SDK only in llm" contract counted any import of `llm.client` as importing the SDK.
  9. Eval runner in `eval/retrieval.py`. Questions only through `scripts/show_questions.load_questions` (refuses any set but tune/smoke). Embedding cache key = sha256 of JSON `[model, dims, role, prompt version, text]`. Cache at `.cache/eval/embed_cache.sqlite`; each finished request cached at once; 4 requests in flight (`EVAL_EMBED_WORKERS`).
  10. Caches: document index per promotion setting under `.cache/eval/docs-promote-<on|off>/` and chunk index per variant under `.cache/eval/<variant>/`, each with a manifest (corpus fingerprint; code hash of `ingest/`, `store/`, `config.py`); a mismatch rebuilds. Prose promotion off = the eval sets `PROMOTE_PROSE_FIELDS` on the modules that read it during the build, restored after.
  11. Metrics: recall@k pooled over all gold documents (a 14-gold question weighs 14); all_gold@10 per question; MRR of the first gold document (0 when absent from the up-to-50 retrieved documents). Questions without gold documents are skipped. Reports keep question ids and ranks only, never document ids or question text.
  12. `eval/history.csv` created by the runner with a header shared by future eval kinds: `timestamp, git_commit, kind, set, variant, questions, headline, value, metrics, cost_usd, seconds, report`; `metrics` is one compact `key=value` string per mode.
- **Over:** a per-variant copy of the document index (2.3 GB, nothing gained), an eval-only spend cap on the same ledger (production would then refuse after the eval), keying the cache by the sent text only, a stopword list (language-specific).
- **Why:** the C4 brief; lean and reusable by B4. Tests: `tests/unit/test_chunk_index.py`, `test_retrieval_eval.py`.
