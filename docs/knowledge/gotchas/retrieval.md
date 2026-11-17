---
name: Retrieval gotchas
description: Traps in the agent's search and the store reads behind it - SQLite FTS5 with a long `rowid IN (...)` list is very slow (23.7 s for 25,000 ids), so scoped BM25 reads every match and filters in Python.
source: runs/2026-10-16-corpus-tools
---

# Retrieval gotchas

## 2026-10-16 FTS5 with a long `rowid IN (...)` list is very slow
- **Area:** store, search
- **Symptom:** BM25 limited to 25,000 chunk ids with `rowid IN (SELECT value FROM
  json_each(?))` took 23.7 s on the production index (900 ids: 0.8 s).
- **Cause:** the planner runs the MATCH once per rowid of the list.
- **Fix:** read every match in score order (62,000 rows in 44 ms) and keep the allowed
  chunk ids in Python, stopping at the limit (`SqliteChunkReader.bm25(..., chunk_ids=)`).
- **Avoid next time:** time FTS5 queries with extra constraints on the real index.
