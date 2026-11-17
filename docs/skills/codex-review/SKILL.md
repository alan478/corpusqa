---
name: codex-review
description: Get an independent Codex review of a plan or a diff, then log each finding as fixed or rejected. Use before merging a big change (new module, agent loop, ingest) or when the user asks for a review.
---
# Codex review
1. Scope: a plan file, or `git diff main...HEAD`.
2. Run read-only and save the output:
   `codex exec --sandbox read-only -m <reviewer model in ~/.agent-harness/team/team.json> "<prompt>" > docs/reviews/<date>-<topic>.md`
   The prompt asks for numbered findings with severity, evidence (file:line)
   and a suggested fix, and says: no code changes.
3. For each finding: accepted, partly or rejected, with a reason, in the
   run's `decisions.md` (Area: review).
4. Fix accepted findings with tests; note the commit next to each finding.
5. Keep the review file in `docs/reviews/` (tracked) as evidence.
