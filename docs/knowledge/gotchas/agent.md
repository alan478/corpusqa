---
name: Agent gotchas
description: Traps in the agent loop - a deadline check with `<` let a normal step start with 0 s at exactly the finish reserve (now `<=`).
source: runs/2026-10-16-agent-tools
---

# Agent gotchas

## 2026-10-16 A deadline check with `<` let a normal step start with 0 s
- **Area:** agent
- **Symptom:** a new test (a compaction call that fails at the very end of its time, leaving
  exactly the 30 s reserve) saw a normal `auto` step after it, not the forced finish; that
  step's model call got a 0 s timeout (a certain `ChatTimeout`, then a forced finish flagged
  `model_error`: one step wasted).
- **Cause:** `_limit_reached` forced only when `time_left < finish_reserve_s`; at exactly the
  reserve the next normal call's timeout is `time_left - reserve = 0`.
- **Fix:** `<=` (`test_exactly_the_reserve_left_forces_a_finish` failed before it).
- **Avoid next time:** test every time limit at exactly its boundary.
