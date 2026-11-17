---
name: prompt-change
description: Change a prompt or a tool description safely. Use for any edit under src/corpusqa/prompts/.
---
# Change a prompt
1. Write the failure you are fixing (label + question ids) in the handoff.
2. Edit the file and bump its `version:` line.
3. Keep the rule general: no corpus names, keys, field names or gold answers.
4. `uv run tests-unit` (prompts are loaded and rendered in tests).
5. `run-eval`: smoke, then tune if smoke holds.
6. Keep only if the target questions improve and nothing else regresses
   beyond noise. Otherwise revert.
7. Log both versions' scores in `decisions.md` (Area: prompts), the change and its
   reason in `docs/prompt-log.md`, and the runs in `docs/eval-log.md`.
