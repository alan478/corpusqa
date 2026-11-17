#!/usr/bin/env python3
"""Run the unit tests when the agent stops, if code changed.

Stop hook for Claude Code and Codex. Runs `uv run tests-unit -x` (commands.json).
Exit 2 asks the agent once to fix failing tests; `stop_hook_active` stops it
from looping.
"""

import subprocess
import sys

from _event import load_event, repo_root


def main() -> int:
    event = load_event()
    if event.get("stop_hook_active"):
        return 0  # already asked once this turn: never loop
    root = repo_root(event)
    changed = subprocess.run(
        ["git", "status", "--porcelain", "--", "src", "tests"],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    ).stdout
    if not changed.strip():
        return 0
    try:
        run = subprocess.run(
            ["uv", "run", "--quiet", "tests-unit", "-x"],
            cwd=root,
            capture_output=True,
            text=True,
            timeout=120,
            check=False,
        )
    except subprocess.TimeoutExpired:
        print("Unit tests took over 120 s. Run them yourself.", file=sys.stderr)
        return 0
    if run.returncode not in (0, 5):  # 5 = no tests collected yet
        tail = "\n".join(run.stdout.splitlines()[-40:])
        print(f"Unit tests fail. Fix them before you stop:\n{tail}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
