#!/usr/bin/env python3
"""Block hand edits to protected project files.

PreToolUse hook for Claude Code (Edit/Write/MultiEdit/NotebookEdit) and Codex
(apply_patch). Exit 2 blocks the call; the reason on stderr goes to the model.
"""

import sys

from _event import edited_paths, load_event, relative, repo_root

ALWAYS = {
    "data/": "data/ is the read-only dev corpus.",
    "uv.lock": "Change dependencies with `uv add`, never by hand.",
}
EXISTING_ONLY = {  # may be created once, never edited by hand afterwards
    "eval/sets/": "Eval sets are fixed. Change them only when the user asks.",
    "eval/history.csv": "Only the eval runner appends to the history.",
}


def main() -> int:
    event = load_event()
    root = repo_root(event)
    for path in edited_paths(event):
        rel = relative(path, root)
        if rel is None:
            continue  # outside the repo: not ours to guard
        for prefix, reason in ALWAYS.items():
            if rel.startswith(prefix):
                print(f"Blocked edit to {rel}: {reason}", file=sys.stderr)
                return 2
        for prefix, reason in EXISTING_ONLY.items():
            if rel.startswith(prefix) and path.exists():
                print(f"Blocked edit to {rel}: {reason}", file=sys.stderr)
                return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
