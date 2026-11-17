---
name: Chunking decisions
description: Decisions about how document bodies become chunks: chunking got its own decision step; a recursive split was built first; step C picked recursive 512 tokens, no overlap, context header on, prose promotion on (tune-set retrieval eval on pooled recall, full table inside); how the strategies are built as pure functions (sections only at the document's own labels, merge within one level, char fallback, whole = recursive at 6,000); header fields on in production (CHUNK_HEADER_FIELDS = True: per source, fields the profile finds common, rarely title or body, of a meaning-bearing kind; rendered inline as `field: value`, capped; q_recall@10 0.673 -> 0.727); how the switch, rendering, S6 span and `--fields` eval flag work; open tie-break questions.
source: runs/2026-10-14-ingestion, runs/2026-10-16-eval-weights-headers, runs/2026-10-16-corpus-tools
---

# Chunking decisions

## Open questions (OPEN: waiting for the user)
- **Step C (1):** prose promotion on and off tie exactly on the tune set; the tie-break
  rule would pick "off", but it was kept on (protects unlisted prose on an unseen corpus;
  $0.04 difference). Flip `PROMOTE_PROSE_FIELDS` if you prefer the literal rule.
- **Step C (2):** whole (0.377) and header-off (0.364) were within one question's worth of
  the best (0.390) but not taken (marginal savings, lower all-gold@10; header-off loses
  MRR). Confirm.

## 2026-10-16 Header fields go on in production (CHUNK_HEADER_FIELDS = True)
- **Area:** ingestion / chunking / retrieval
- **Chose:** turn `CHUNK_HEADER_FIELDS` on in `config.py`: every chunk header now ends
  with the document's chosen field values (rule in the entry "Header fields are chosen by
  the profile" below). Measured with `uv run eval-retrieval` on the tune set only (26
  questions with gold), full corpus, plain `recursive-512` against `recursive-512-fields`
  (fusion bm25 1 / dense 1 / k 60):

  | mode | metric | plain | fields |
  |---|---|---|---|
  | hybrid | **q_recall@10** (headline) | 0.673 | **0.727** |
  | hybrid | pooled recall@10 | 0.377 | 0.403 |
  | hybrid | MRR | 0.588 | 0.671 |
  | hybrid | all_gold@10 | 0.615 | 0.692 |
  | bm25 | q_recall@10 | 0.556 | 0.603 |
  | dense | q_recall@10 | 0.635 | 0.696 |

  Cost: embedded tokens 31.81M vs 28.20M (+12.8 %); the variant run spent $6.35
  (ledger $39.71 of the $60 cap). The production index then rebuilt from the shared
  embedding cache ($0) and gave exactly the variant's numbers (report
  `20261016T213056Z-production-recursive-512-fields`).
- **Keep rule (agreed in the plan before the run):** adopt only if hybrid q_recall@10
  rises by more than one question's worth (1/26 = 0.038) and hybrid MRR does not fall.
  Result: +0.054 (about 1.4 questions) and MRR +0.083, so it passed both.
- **Over:** leaving it off.
- **Why:** backlog item 3 (the contextual-chunk pattern). Unit tests that leaned on the
  old default now set the switch themselves (tests.md).

## 2026-10-16 How header fields are switched, rendered and traced
- **Area:** ingestion / chunking / eval
- **Chose:**
  - `ChunkSpec.fields: bool | None`; None follows `CHUNK_HEADER_FIELDS`, but only when
    the header is on (so `ChunkSpec(header=False)` never fails when the switch flips);
    an explicit `fields=True` with `header=False` raises. `ChunkSpec.header_fields`
    resolves it. `ChunkInput.fields`; `chunk_input(doc, spec)` copies `doc.fields` only
    when the spec shows them.
  - Rendering (`make_header` + `render_header_fields`): field names are made one line too
    (a key may hold whitespace); blank values are dropped; the 400-char cut keeps the
    longest prefix of whole parts (stops at the first part that does not fit), so the most
    common fields win and the order is stable; a part over the cap on its own is dropped,
    never cut. `count_header_values` counts what is shown.
  - S6 span: `header_fields` (on/off, always); `header_field_sources` is the mapping itself
    (source -> field names, at most `TRACE_MAX_LISTED_ITEMS` sources), not a count, because
    it is what the profile picked; `header_field_count`; output count
    `header_field_values` = values actually shown (after the caps), 0 when off.
  - Eval: `eval-retrieval --chunking ... --fields`; variant name `-fields` after
    `noheader`'s slot (`recursive-512-fields`); `describe()["header_fields"]`. `--fields`
    with `--production` or `--no-header` is a usage error at parse time (production
    follows the config switch; a silent ignore would mislabel a run). Now that the switch
    is on, a `--chunking` variant has fields even without the flag; there is no
    `--no-fields` yet (docs/backlog.md).
- **Over:** a count for `header_field_sources`; best-fit packing of parts; ignoring
  `--fields` under `--production` like the other chunking flags.
- **Why:** deterministic headers across builds; traces that show the learned choice;
  no run whose name says something it did not do.

## 2026-10-16 Header fields are chosen by the profile, rendered inline, capped
- **Area:** ingestion / chunking
- **Chose:** `ingest.chunking.select_header_fields(field profile rows)`: per source, a
  field goes into the header when `presence >= 0.5`, `title_ratio < 0.5`,
  `content_ratio < 0.5` and its kind is one of enum, person, tags, text, mixed, number,
  boolean (`HEADER_FIELD_MIN_PRESENCE`, `HEADER_FIELD_MAX_TITLE_RATIO`,
  `HEADER_FIELD_MAX_CONTENT_RATIO`, `HEADER_FIELD_KINDS` in `config.py`). Ordered by
  presence (most common first), then name; a source with no such field is left out.
  Rendered after the date as `field: value` parts with the same ` · ` separator, list
  items joined by `, ` (at most 5, `HEADER_FIELD_LIST_ITEMS`), each value cut at 60 chars
  (`HEADER_FIELD_VALUE_CHARS`), the fields part cut at 400 chars (~100 tokens,
  `HEADER_FIELDS_MAX_CHARS`) at a part boundary. Noise rows and empty/null/json values
  are skipped (store.md).
- **Over:** every field (ids, URLs and references add long tokens and no meaning);
  structured kinds only (the profile labels many short label fields `text`/`mixed`; see
  backlog item 2); a second header line (the header is one line everywhere: traces, FTS
  column, tests).
- **Why:** backlog item 3 and the Anthropic contextual-retrieval size (50-100 tokens);
  dates are already in the header and reference-like values are noise for dense search.
  Learned from the profile, never from names in `data/`.

## 2026-10-16 Step C result: chunking = recursive split, 512 tokens, no overlap, header on; prose promotion stays on (settles decisions 1 and 9)
- **Area:** ingestion / chunking (S6, S9 input)
- **Note (2026-10-16, runs/2026-10-16-eval-weights-headers):** "recall" and "highest hybrid recall@10" in this entry mean the POOLED metric (all gold found / all gold). History rows from commit 5caf2fda on use per-question recall `q_recall@10` as the headline (retrieval.md#2026-10-16 Per-question recall). Per-question, recursive-512 still tops this table (0.692), so the choice holds. The header is now `source · title · date · field: value ...` (entry "Header fields go on in production" above).
- **Chose:** `ChunkSpec("recursive", max_tokens=512, header=True)` (the default spec), no overlap, `PROMOTE_PROSE_FIELDS = True` (unchanged). Decision 7 (drop repeated email quotes) is not part of the recursive strategy and stays off (not built). Decision 9 (chunk size) = 512, not 400.
- **How measured:** `uv run eval-retrieval` on the tune set only (26 questions with gold documents, 77 gold documents; 4 unanswerable skipped), commit d4f12f9e, every variant on the FULL corpus (21,612 docs). Fixed retriever: BM25 (FTS5) + dense (gemini-embedding-2, 768 dims, cosine) fused by RRF k = 60 over the top 50 chunks each, mapped to documents. Recall pooled over the 77 gold documents (1 gold document = 0.013; "one question's worth" = 1/26 = 0.038). Reports: `eval/results/tune/retrieval/<time>-<variant>/`; rows in `eval/history.csv`.

  | variant | chunks | p50 tok | bm25 R@10 | dense R@10 | hybrid R@5 | **hybrid R@10** | hybrid R@20 | hybrid all-gold@10 | bm25 MRR | dense MRR | hybrid MRR | embedded tokens | cost from scratch |
  |---|---|---|---|---|---|---|---|---|---|---|---|---|---|
  | recursive-512 (baseline, header on) | 67,916 | 454 | 0.299 | 0.364 | 0.351 | **0.390** | 0.403 | **0.654** | 0.428 | 0.626 | 0.588 | 27.9M | $5.58 |
  | recursive-256 | 133,799 | 219 | 0.286 | **0.429** | 0.351 | **0.390** | **0.442** | 0.615 | 0.383 | 0.604 | **0.605** | 29.4M | $5.87 |
  | fixed-512, 10 % overlap (51) | 65,682 | 511 | 0.299 | 0.403 | 0.338 | **0.390** | 0.429 | **0.654** | 0.435 | 0.597 | 0.589 | 30.1M | $6.01 |
  | recursive-512, promotion off | 67,249 | 454 | 0.299 | 0.364 | 0.351 | **0.390** | 0.403 | **0.654** | 0.430 | 0.630 | 0.590 | 27.7M | $5.54 |
  | whole (≤ 6,000 tok, else recursive) | 21,631 | 1,095 | 0.299 | 0.403 | 0.338 | 0.377 | 0.429 | 0.615 | **0.463** | **0.691** | 0.595 | 26.9M | $5.37 |
  | recursive-512, header off | 67,916 | 454 | 0.286 | 0.364 | 0.338 | 0.364 | 0.403 | 0.615 | 0.416 | 0.493 | 0.511 | 26.8M | $5.36 |
  | recursive-1024 | 37,594 | 812 | 0.286 | 0.377 | 0.312 | 0.351 | 0.403 | 0.577 | 0.447 | 0.595 | 0.516 | 27.2M | $5.45 |

- **Reasoning (the agreed rule):** highest hybrid recall@10 wins; within one question's worth, prefer the simpler and cheaper one.
  1. Four variants tie at 0.390. Among them recursive-512 is the cheapest real option: recursive-256 doubles the chunks and vectors (2.3x the embed time, +$0.29) and loses one question on all-gold@10; fixed-512 embeds 8 % more text because of the overlap (+$0.43). recursive-512 is also the user's first-choice strategy.
  2. whole (0.377) and header-off (0.364) are within one question's worth, so the tie-break was considered. Not taken: (a) the cost difference is marginal and `whole` is the same code path; (b) both lose one question on all-gold@10 (0.615 vs 0.654); (c) header-off clearly lowers MRR (0.588 -> 0.511): the header (source, title, date) helps; (d) whole-document chunks (p50 1,095, up to 6,000 tokens) would make every hit a whole document for the agent's context. recursive-1024 (0.351) is outside the noise band.
  3. Prose promotion (decision 1): on and off tie exactly. The tie-break would pick "off". **Conservative call: keep it on.** The saving is negligible, while turning it off makes long prose in unlisted fields unsearchable on an unseen corpus whose content lists were set inconsistently by an LLM labeller. One flag flips it (`PROMOTE_PROSE_FIELDS`).
- **Not decided here (for the retrieval step):** BM25 is weak on this set (R@10 ≈ 0.29-0.30) and fusion with it pulls dense-256 down (0.429 -> 0.390); document-level vectors rank the first gold document best (whole, dense MRR 0.691). Both are retrieval questions (fusion weights, BM25 query handling, a document-level vector beside chunks). 768 vs 1,536 dims was not compared.
- **Over:** recursive-256, fixed-512 with overlap, whole, header off, 1024, promotion off (reasons above).
- **Why:** agreed rule of step C (user, 2026-10-16), measured on the tune set only; held-out never touched.

## 2026-10-16 C3: chunking strategies as pure functions (readings of the brief)
- **Area:** ingestion / chunking
- **Superseded in part by:** chunking.md#2026-10-16 Header fields go on in production (item 8: the header now ends with the document's chosen field values; `ChunkInput` gained `fields`); dates.md#2026-10-16 Dates are judged one by one (item 8: no low-trust created date any more; a date outside the window is no date, so it is never in the header).
- **Chose:**
  1. `ingest/chunking.py`: `chunk_document(ChunkInput, ChunkSpec) -> list[ChunkDraft]`, registry `STRATEGIES` (recursive, fixed, whole). `ChunkInput` = source, display_title, created, body, labels (documents.body_fields).
  2. Sections are cut only at lines that are exactly `## <label>` for one of the document's own labels, so a markdown `## Heading` inside content is not a section cut (it can still be a line or paragraph cut). No source or field name appears in the module.
  3. Recursive merge works like LangChain's recursive splitter: neighbours merge only within one level; the sub-chunks of an oversize piece are not merged with its neighbours. Separators between merged pieces stay inside the chunk (it is one contiguous slice); whitespace is trimmed at chunk edges.
  4. A last level `char` cuts a single word longer than the limit at the limit, so "no chunk over its limit" always holds. Sentence end = `.`, `!` or `?` followed by whitespace.
  5. `whole` = the same recursive rule with the limit `CHUNK_MAX_EMBED_TOKENS` = 6,000 estimated tokens (a quarter spare below the model's 8,192, because chars/4 undercounts code, numbers and non-English text). No spec may exceed 6,000.
  6. `fixed`: default overlap = 10 % of the size (`CHUNK_FIXED_OVERLAP_SHARE`; 51 at 512), cut after the last whitespace inside the window, a window without whitespace is hard-cut, the overlap start moves forward to the next whole word.
  7. Empty or whitespace-only body: no chunks (S5 already makes a `no_body` document's body its title, so this is rare).
  8. Header = `source · display_title · YYYY-MM-DD` (date part only; left out when created is None; C4 also leaves out a low-trust created date, see retrieval.md), kept apart from text. Each draft records `level` (whole, section, paragraph, line, sentence, word, char, fixed). n_tokens = ceil(chars / 4).
- **Over:** detecting sections from any `## ` line (would cut at content headings), merging across levels (mixes a section's tail with the next section's head), a fixed 50-token overlap (invalid for small limits).
- **Why:** brief for C3 and the user's recursive-first decision. Smoke on all 21,612 data/ bodies: 0 chunks break the slice or size rules. Body tokens p50 1,095, p90 2,010, p99 3,235, max 8,240.

## 2026-10-16 First chunking variant: recursive split with a size limit (user decision)
- **Area:** ingestion / chunking
- **Chose:** the first strategy to build and evaluate is a recursive splitter, the same for every source type: if the whole document fits the limit, it is one chunk; otherwise split by body section (content field), then blank-line paragraph, then line, then sentence, then words, recursing only into pieces still over the limit; adjacent small pieces are merged back up to the limit. Defaults: limit 512 tokens (chars/4), no overlap, context header (source · title · date) on every chunk.
- **Over:** starting with the structure-aware cascade of design.md S6 (email headers, speaker turns, headings).
- **Why:** user: "lets do it recursively for the first try where we set a limit and divide it based on the document than paragraph and so on". Simple, standard, and a clean baseline for the eval.

## 2026-10-16 Chunking is not approved; it gets its own decision step (user correction)
- **Area:** ingestion / chunking
- **Superseded by:** chunking.md#2026-10-16 Step C result (chunking decided; S6, S9 and the chunk tables built in B4).
- **Chose:** the design approval excludes chunking. Reopened: design.md S6 (split + pack, context headers, fields chunks), the `chunks`, `chunks_fts` and `embeddings` tables, S9, and decisions 1 (promote prose fields to body), 7 (drop repeated email quotes) and 9 (400-token target). Still approved: S1–S5, S7 for `documents` + `fields`, S8, S10, decisions 2–6 and 8.
- **Over:** building S6 as written in design.md.
- **Why:** user: "wait do not chunk we need to decide what chunking strategy to use on this right?"
