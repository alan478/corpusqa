---
name: Date decisions
description: Decisions about document dates (created/updated): the in-range window is learned from the corpus's ISO-format years (1st–99th percentile, ±1 year) instead of the fixed 1990..this year + 5; each date is judged on its own against it (a date outside counts as no date; user rule 2026-10-16, which replaced the first rule of low-trust date fields with the `_low_trust` rule suffix); the post-pass stage S4_date_window that does it; and the open follow-up (widen the window if an eval needs it).
source: runs/2026-10-14-ingestion, runs/2026-10-16-corpus-tools
---

# Date decisions

## Open questions
- **B2 (1), answered by the user 2026-10-16:** chat-source epoch timestamps are mostly fake
  (years 2021–2286). The learned window (data/: 2023–2030) stays, and each date is judged
  on its own (entry below). User: "lets try with the current range and if needed we can
  increase the range". Still open: check the window against the evals and widen it if
  needed. Not done yet; the pub_0077 investigation (2026-10-17) found Slack thread dates
  that drift from the dates in their text (53% of dated threads by more than 180 days).
  In `docs/backlog.md`.

## 2026-10-16 Dates are judged one by one (the user's rule replaces low-trust fields)
- **Area:** ingest dates, store, tools, tests
- **Chose** (user, 2026-10-16: "remove the trust on the field part"; "lets try with the
  current range and if needed we can increase the range"):
  - No field-level rule: `low_trust_fields`, `DATE_FIELD_MAX_OUT_SHARE` and
    `LOW_TRUST_RULE_SUFFIX` are gone; nothing reads a rule suffix. The learned window is
    unchanged (data/: 2023-2030).
  - Each date value on its own. Inside the window: a created/updated candidate (order and
    role words as before), shown in the chunk header, matched by created ranges, field date
    ranges and `eq` on a day. Outside: no date. It is no candidate, not in the header, and
    `iso_range` matches only rows with `date_in_range = 1`. The row keeps its value,
    `value_iso` and `date_in_range = 0`.
  - Counted: meta `date_outside_window` (replaces `date_low_trust_fields`): per source with a
    date outside the window, its date values, how many are outside, its documents without a
    usable date, and per field (values, outside). The `S4_date_window` span: `date_values`,
    `dates_outside`, `sources_with_dates_outside`, `outside`. S6's
    `low_trust_dates_left_out` count is gone. The window's meta key names are constants in
    `store/interfaces.py` (`META_DATE_WINDOW_MIN_YEAR` / `_MAX_YEAR`).
  - Tools: `FieldCoverage.dates_outside`, `CorpusReader.date_window()`,
    `FieldIndex.date_window`. Every date text says how many documents have no usable date
    ("none, or only dates outside the date window A-B, which count as no date"); the catalog
    opens with that line. `tools.json` v3: one sentence in the filter description.
- **Tests changed because the user changed the rule** (coverage kept, each now asserts the
  new rule): `test_ingest_stages`, `test_ingest_pipeline`, `test_chunk_index`,
  `test_ingest_chunks_embed`, `test_store_tool_reads`, `test_retrieval_tools`,
  `test_corpus_tools`. New: `test_the_date_window_is_read_from_the_index`,
  `test_a_date_outside_the_window_is_matched_by_no_date_range`,
  `test_a_date_outside_the_window_matches_no_range_and_is_counted`,
  `test_date_texts_say_how_many_documents_have_no_usable_date`.
- **Over:** the field-level rule (a field with more than 20 % of its values outside the
  window was low-trust; its in-window dates were used only as a last resort, marked
  `_low_trust`, and left out of headers and created ranges: 11,596 of 11,606 Slack threads
  had no usable created date).
- **Why:** the user's decision, to be checked against the evals (the window may be widened).
- **Effect:** the index fingerprint changed, so the next start-up rebuilt the index. Read-only
  estimate beforehand: 6,416 Slack documents that had a `_low_trust` created date now show
  it in their headers (14,267 chunks to embed again, about $1.19); the real cost was not
  measured.

## 2026-10-16 C1: how the learned date window was built (readings of the 2026-10-16 entry)
- **Area:** ingestion / dates
- **Superseded in part by:** dates.md#2026-10-16 Dates are judged one by one (items 2 and 4-6: no low-trust fields, no `meta.date_low_trust_fields`; items 1, 3 and 7 stand).
- **Chose:**
  1. "ISO-format" = the ISO parse rules only (YYYY-MM-DD, ISO date-time, YYYY-MM, YYYY-Www, YYYY-Qn). RFC 2822, month names and slash dates are `text` and do not feed the window; epochs are `epoch`. Recorded per value in the column `fields.date_format` (iso | epoch | text).
  2. A post-pass stage `S4_date_window` after `S7_store`. While documents stream, the pipeline keeps only small date facts in memory (years per (source, field), years of ISO values, each doc's scalar date candidates, stored dates and flags; no text). The pass learns the window, marks low-trust fields, runs one SQL UPDATE of `date_in_range`, and rewrites created/updated/flags only for documents whose result changed. S4 per document still uses the fixed window, so `build_fields` stays self-contained.
  3. Percentiles by nearest rank. `DATE_WINDOW_MIN_VALUES` = 100 (below that the 1st/99th percentile is just the min/max; the fixed window is used). The learned window is not clamped to the fixed bounds.
  4. Low-trust is per (source, exact field key), over every date value of the field (list items too), share strictly over 0.2 (`DATE_FIELD_MAX_OUT_SHARE`), no minimum count.
  5. Planned-word fields are dropped before the trusted/low-trust split (a due date is not a trusted created date). Low-trust candidates come before the file-stem fallback.
  6. The low-trust list is stored in `meta.date_low_trust_fields` (JSON: source, field, values, out_share) with `date_window_min_year/max_year/rule/iso_values`.
  7. Epoch/ISO datetimes are formatted with `isoformat()` (always a 4-digit year) so the SQL year parse (text before the first dash) is safe on every platform.
- **Over:** holding every document in memory until the window is known (RSS), a second read of the corpus (time), a separate low-trust table before S8.
- **Why:** brief: "simplest correct post-pass", conservative where unclear. Read-only run on data/: window 2023-2030 (24,799 ISO values), 7 low-trust fields (all in one chat source; the three timestamp fields at 44.6 % outside), docs without dates 4,847 -> 5,183, docs with `_low_trust` rules 0 -> 6,416, no created year after 2030 (was 335 in 2031).

## 2026-10-16 Dates: learned window + low-trust fields (coordinator default; user said "complete it till 4 do not stop")
- **Area:** ingestion / dates
- **Superseded in part by:** dates.md#2026-10-16 Dates are judged one by one (the low-trust fields; the learned window stands).
- **Chose:** replace the fixed in-range window (this year + 5) with a window learned from the corpus: the 1st–99th percentile of years over ISO-format date values (not epoch numbers), widened by 1 year each side; fall back to the fixed window when there are too few ISO values. Per (source, field): when more than 20% of a date field's values fall outside the window, the field is low-trust. created/updated prefer high-trust candidates; when only low-trust ones exist they are used but the rule says so (e.g. `order_low_trust`), so later "which is newer" logic can treat them as unknown. Thresholds in config.py.
- **Over:** keeping the fixed window (2,080 fake 2027–2031 chat dates pass), widening it (more fakes pass), learned window alone (fakes inside it still look trustworthy).
- **Why:** chat epoch values span 2021–2286 (only ~4,675 of 11,606 in 2024–2026); ISO dates corpus-wide: p1 2024, p50 2026, p99 2029. The user asked not to stop; this was the recommended option. User may override.
