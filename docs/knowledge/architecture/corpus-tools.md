---
name: Corpus tools — the agent's search, filter, field, key and read tools (as built)
description: The seven deterministic tools the agent calls over the index, as built - search (scoped hybrid BM25 + dense, RRF, pages of 8 that search deeper, 2 passages of ~300 chars, exact phrases and keys), filter (any field, AND of eq/in/range/present/missing/contains, values resolved exact then loose, exact count, R# of every match), field_values, list_fields, read_fields, read_document, lookup_key; the rules they share (pages with exact totals and next offsets, nothing cut, D#/R# handles, ToolInputError messages, each date judged on its own), the field catalog in the system prompt, citations (R# expands to every id), the store reads behind them, tracing, measured times, known gaps and tests. Read before changing a tool, retrieval/, the tool texts or tools.json, or the store reads they use. The loop that calls them is agent.md.
source: runs/2026-10-16-corpus-tools, runs/2026-10-16-agent-tools
---

# Corpus tools (as built)

The agent's tools are plain code over the index: exact counts, filters, id lookups and
citation lists come from code, never from the model. The only API call inside a tool is
the query embedding of `search`. The rule the user agreed (2026-10-16): **where a value
lives decides the tool** - a value written in a record's text: `search`; a value held in a
field (a status, a person, a date, an amount, a team): `filter`, `field_values`,
`read_fields`. User decisions that bind every tool: exact counts; every list paged with its
total and next offset; the whole document on read; R# citations expand to every id; no other
limit that drops data.

The loop that calls these tools, and how citations become the response's ids: `agent.md`.

## Parts
| File | Role |
|---|---|
| `agent/corpus_tools.py` | Tool adapters: Pydantic argument models (`extra="forbid"`; a JSON number or boolean sent where text is expected becomes text), the result texts, `CorpusTools` (shared, read-only: reader, `FieldIndex`, document summaries, known ids, `CorpusSearch`, the embedder only when there are vectors, tool schemas, the field catalog), `corpus_session_factory(reader, chunks, vectors, embedder)`, and `CorpusToolSession` (one per request: its own `Handles`, its own query-vector memo, a `ToolBox` of the 7 tools + `finish`). |
| `agent/handles.py` | `Handles`: D# per document shown, R# per result set; `expand`, `documents`, `resolve`, `name_in_text`. Thread-safe (a step's calls run at once). |
| `retrieval/search.py` | `CorpusSearch` (scoped hybrid ranking, deeper pages, passages, exact keyword marks), `Scope`. Uses `rrf` and `to_documents` from `retrieval/hybrid.py`. |
| `retrieval/filtering.py` | `Condition`, `run_filter`: AND of conditions, coverage per field, created range. |
| `retrieval/values.py` | `loose`, `nearness`, `resolve_value`, `resolve_contains`, `day_range`, `date_bounds`, `number`. |
| `retrieval/fields.py` | `FieldIndex` (built once; resolves source and field names; picks the catalog fields), `list_fields`, `field_values`. |
| `retrieval/keys.py` | `lookup_key`: owners from `doc_keys`, referrers from `links`. |
| `retrieval/documents.py` | `read_fields`, `doc_rows`, `read_document`. |
| `retrieval/paging.py` | `Page`, `page_of`: every list a tool returns. |
| `retrieval/errors.py` | `ToolInputError(message)`: the one error a tool raises on purpose. |
| `store/interfaces.py`, `store/sqlite.py` | `CorpusReader` on `SqliteReader`: `document_summaries`, `document`, `value_counts`, `field_coverage`, `filter_doc_ids`, `document_fields`, `key_owners`, `links_to`, `date_window`, `source_dates`, `field_conditions`. `ChunkReader.bm25(..., chunk_ids=)`, `keyword_matches`, `chunk_texts`. `store/vectors.py`: `VectorIndex.search(..., allowed=)`. |
| `store/norms.py` | `normalise`, `fold_key`, `strip_person`, `person_norm`, moved from `ingest/` so ingest and retrieval match values the same way. |
| `prompts/tools.json` | The schemas the model sees (version 6). |

## The tools
| Tool | Arguments | Returns |
|---|---|---|
| `search` | query, source?, created_from?, created_to?, within? (an R# or D#), offset=0 | 8 documents (`SEARCH_PAGE_DOCS`), best first, each with up to 2 passages |
| `filter` | source, conditions [{field, op, value? / values? / from? / to?}], created_from?, created_to?, offset=0 | the exact count, an R# of every match (newest first), what each value matched, each field's coverage, rows 50 a page with the asked fields' values |
| `field_values` | source, field, near?, offset=0 | every distinct value with its document count (values near `near` first), other spellings, coverage |
| `list_fields` | source, near?, offset=0 | every field of the source: kind, documents, share, distinct values, top 3 values, "same as", "present when" |
| `read_fields` | documents [D#/R#...], fields [...], offset=0 | per document each field's values, or why none (empty, in the body, absent, not a field of that source); 50 rows a page |
| `read_document` | document (a D# or an id) | the whole document: title, source, dates, every field, the full text |
| `lookup_key` | key, offset=0 | the documents that own the key, then those that refer to it (and how), with an R# |
| `finish` | answer, citations (D#, R#, ids), not_found | "Answer received." (see `agent.md`) |

## Rules every tool follows
- **Pages, nothing cut.** Every list comes as a page with a line such as `rows 1-50 of 278;
  next: offset=50`, `(last page)` or `no rows`. Page sizes are in `config.py`
  (`FILTER_PAGE_ROWS`, `READ_FIELDS_PAGE_ROWS`, `LIST_FIELDS_PAGE_ROWS`,
  `FIELD_VALUES_PAGE_ROWS`, `KEY_LOOKUP_PAGE_ROWS` 50; `SEARCH_PAGE_DOCS` 8). The ToolBox
  never cuts a result.
- **Handles.** Results show `D<n>` per document and `R<n>` per filter or key-lookup result
  (the complete ordered id list); the same document or list always gets the same handle in
  one request. Raw `dataset_doc_uuid`s are never printed (a known raw id is still
  accepted). A result set's line says how to use it: cite R# to cite every one,
  `read_fields(documents=["R1"], ...)`, `search(..., within="R1")`.
- **Errors the model can fix** raise `ToolInputError`; its message reaches the model as
  `<tool_error>...</tool_error>`: an unknown source or field (with the nearest names), a
  value with several close spellings (all listed with counts, nothing run), a value with
  none (the 5 nearest), a handle this request never issued. Any other exception shows the
  model only its type.
- **Noise rows** (ingest's `is_noise`) are left out of value counts, coverage and filters.
- **Dates:** each date is judged on its own against the learned date window (data/:
  2023-2030). A date outside it counts as no date: no created range or date range matches
  it. Every text that filters on dates says how many documents had no usable date, and the
  catalog opens with one line on it.

## search
- **Scope first:** a source, a created range (`documents.created`, always inside the
  window) and `within=` a handle limit the chunks that BM25 and dense rank, so a large set is
  searched whole. Scoped BM25 reads every match in score order and keeps the allowed chunks
  (gotchas/retrieval.md: FTS5 with a long `rowid IN` list is very slow).
- **Ranking:** the first ranking keeps each ranker's best `RETRIEVAL_TOP_CHUNKS` (50)
  chunks and fuses them by weighted RRF (k 60, weights 1.0 / 1.0), so page 1 is the hybrid
  the retrieval eval validated. Each document sits at its best chunk.
- **Deeper pages** (user: "Search deeper"): a page past what the list holds re-ranks with
  `SEARCH_DEPTH_GROWTH` (2) times more chunks per ranker, again, until the page is filled
  or every chunk in scope is ranked. New documents join the end of the list; found ones
  keep their places. Both rankers break ties by chunk id, so a shallow ranking is the
  start of a deep one: pages never repeat or skip a document. The page line gives no total
  until the ranking is complete (`documents 1-8; next: offset=8`), then an exact one.
- **Passages:** up to `SEARCH_SNIPPETS_PER_DOC` (2) per document, each a window of about
  `SEARCH_SNIPPET_CHARS` (300) characters around the most query words. `read_document`
  gives the rest.
- **Exact phrases and keys** (`SEARCH_EXACT_PHRASES = True`, on since smoke run 3): the
  keyword side matches quoted text as a phrase and a key-shaped token (letters or digits
  joined by `- _ / . #`) as the phrase of its parts; other words match one by one; capital
  AND / OR are not searched for. A document only the dense side found is marked `meaning
  only`, and the page says when no document in scope contains the searched words. Off:
  run 1's behaviour (every word OR-ed, so a search never came back empty).
- **Query vector:** one embedding per distinct query per request (memo), with the start-up
  embedder and the query role. An `LLMError` gives keyword ranking only (the head says so;
  the span gets a WARNING). No embedder or vectors: keyword ranking.

## filter and value matching
- Conditions on any field of one source (title fields too), joined by AND. Ops: `eq`, `in`,
  `range` (from..to, inclusive, either open), `present`, `missing` (no value: empty or
  absent), `contains` (the value's loose form contains the asked words).
- **Value resolution** (`resolve_value`): an exact (normalised) match is used, and other
  spellings with the same loose form are named as not included; else one loose match
  (case, `-`, `_`, `$`, punctuation and spaces folded: "ENG sre" finds `eng-sre`) is used
  and reported; several loose matches stop the call with all of them and their counts;
  none stops it with the nearest values. Persons match through `person_norm`, as ingest
  stored them; list values item by item; `eq` on a date field means that day, month or
  year; on a number field, that number.
- Each condition reports how the source's documents hold the field (with a value, empty,
  in the body, absent), so a missing field is never read as "no".
- `created_from` / `created_to` match the document's created date; the text says how many
  documents of the source have no usable one.
- "Near" (`near=` in list_fields and field_values, and "did you mean" lists): a name or
  value is near at a score of `NEAR_MIN_RATIO` 0.75 (one of two words shared scores 0.75,
  containment 0.9).

## Fields, keys, documents
- `FieldIndex` (built once from `field_profile`, `field_conditions`, `source_profile` and
  the date facts) resolves source and field names: exact, else the one folded or loose
  match, else `ToolInputError` with the nearest names.
- **The field catalog** (`CorpusTools.catalog`, `field_catalog`) goes into the system prompt
  (`agent.md` v3+, `{catalog}` slot): one line on dates, then per source its size, created
  range and documents without a usable date, and its common fields - picked by the same
  rule as the chunk header fields (presence >= 0.5, title ratio < 0.5, content ratio < 0.5),
  leaving out fields that are mostly noise (`CATALOG_FIELD_MAX_NOISE_RATIO` 0.5) - each
  with kind, share, distinct values, top 5 values, "same as" groups and "present when"
  conditions. data/: 132 fields, about 21,700 characters.
- `lookup_key`: owners from `doc_keys` (own-id values, file stems, keys a stem starts
  with); referrers from `links`, resolved or not (on data/ under 4% resolve): a link whose
  key is the asked key, holds it as a whole token, or resolved to an owner. A key held by
  many documents is flagged as a hub. Mentions in free text are search's job.
- `read_fields` reads the field rows directly (exact); field names are resolved per source
  of the documents.
- `read_document` reads one document whole (the longest on data/ is 32,959 characters).

## Citations and the answer text
- `finish` takes citations as D#, R# or known ids. `CorpusToolSession.resolve_citations`
  expands D# to its id and R# to every id of its set, keeps known ids, drops anything else,
  dedupes in first-seen order. The server then keeps at most `MAX_DOCUMENT_IDS` (500), and
  logs and traces a cut (`server/respond.py`).
- `clean_answer` (the handle safety net): each handle this request issued that the answer
  text holds as a whole token is replaced by what it is: a D# by `"<title>" (<source>,
  <YYYY-MM-DD>)` (in brackets: `"<title>", <source>, <date>`), an R# by `the 14 matching
  <source> records`. Tokens the request never issued, and every token the question holds,
  stay as written. It never changes the ids.

## Tracing
The ToolBox makes one `tool:<name>` span per call: arguments as given and as checked, the
result text as the model saw it, `result_chars`, the error kind, and the tool's `handles`
metadata (the page's handles, the result set, counts). Inside it the tool's own span:
`search` (scope, depth, rankings, complete, exact) > one `bm25` and one `dense` span per
ranking (input with its depth) and the query's `embed` generation; `filter`, `read_fields`,
`read_document`, `lookup_key`, `list_fields`, `field_values` with what they resolved and
counted.

## Measured (data/, production index read-only, 2026-10-16)
Factory 0.4-0.6 s at start-up. Tool calls 3-65 ms; the slowest, a filter `eq` on a field with
16,000 values in an 11,606-document source, 413 ms. Search pages (keyword ranking) at offset
0 / 48 / 400 / 3,000 in 39 / 18 / 43 / 85 ms. By hand on tune questions: pub_0046's filter
gave the 14 expected ids, pub_0040's the 9.

## Known gaps (open; `docs/backlog.md`)
- `contains` matches only the first 256 characters of a value (`VALUE_NORM_MAX_CHARS`),
  silently, and selects a whole `value_norm` group, so person values that differ only by a
  bracketed role come back too (Codex #4).
- A JSON null is shown as "absent", not "empty" (ingest marks nulls as noise; Codex #8).
- A network error while embedding the query fails the whole search instead of falling back
  to keywords (it is not an `LLMError`; Codex #5).
- Tool calls have no time limit of their own (Codex #1).
- `bm25` / `dense` spans list every ranked chunk: up to ~68,000 entries on a deep page.
- The model often wraps `filter` argument names in literal quotes (`"\"field\""`); the
  arguments model refuses them and the step is lost (5-6 of ~13 filter calls per tune run).

## Tests
`tests/unit/tool_corpus.py` (the fixture: a copy of `tests/fixtures/mini_corpus/` plus two
invented sources, `benchwork` with 80 orders for paging and `logbook` with one date inside
the window and one far outside, built with `FakeEmbedder(dims=16)`),
`test_store_tool_reads.py`, `test_retrieval_tools.py`, `test_corpus_tools.py` (incl. a
FakeLLM run that filters and cites R1 -> every id), `test_agent_tools.py`;
`tests/contract/test_server_agent.py` (the tools through the server; four sessions at once
keep their own handles).
