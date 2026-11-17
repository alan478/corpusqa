#!/usr/bin/env python3
"""Re-show the eval split rule to the agent.

SessionStart hook for Claude Code and Codex (fires on startup, resume, clear
and after compaction), so the rule survives long sessions. Prints the section
of AGENTS.md between the split-rule markers, plus the live set sizes. Plain
stdout becomes context for the model.
"""

import json
import sys

from _event import load_event, repo_root

START, END = "<!-- split-rule:start -->", "<!-- split-rule:end -->"


def main() -> int:
    root = repo_root(load_event())
    agents = (root / "AGENTS.md").read_text(encoding="utf-8")
    if START not in agents or END not in agents:
        print("[eval split] WARNING: split rule markers missing from AGENTS.md")
        return 0
    rule = agents.split(START, 1)[1].split(END, 1)[0].strip()
    sizes = []
    for name in ("tune", "smoke"):  # never open the held-out file, even for its size
        path = root / "eval" / "sets" / f"{name}.json"
        if path.exists():
            sizes.append(
                f"{name}={json.loads(path.read_text(encoding='utf-8'))['counts']['total']}"
            )
    if sizes:
        sizes.append("held-out=the rest")
    print(f"[eval split] Reminder from AGENTS.md (sets now: {', '.join(sizes) or 'missing'})\n")
    print(rule)
    return 0


if __name__ == "__main__":
    sys.exit(main())
