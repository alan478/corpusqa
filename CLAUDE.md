@AGENTS.md

# Claude Code specifics
- Project hooks (`.claude/settings.json` → `scripts/hooks/`) block hand edits
  to `data/` and `uv.lock`, format and lint each edited Python file, and run
  the unit tests when you stop. If a hook blocks you, fix the cause. Never
  work around it.
- Use subagents for broad searches and reviews; keep the main context for the task.
- Project skills are in `.claude/skills/` (a link to `docs/skills/`).
