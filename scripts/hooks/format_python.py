#!/usr/bin/env python3
"""Format and lint each edited Python file.

PostToolUse hook for Claude Code and Codex. Fixes what ruff can fix; exit 2
sends the remaining problems back to the model.
"""

import subprocess
import sys

from _event import edited_paths, load_event, relative, repo_root


def main() -> int:
    event = load_event()
    root = repo_root(event)
    problems = []
    for path in edited_paths(event):
        # Only this repo's files: scratch scripts elsewhere are not held to project rules.
        if path.suffix != ".py" or not path.exists() or relative(path, root) is None:
            continue
        subprocess.run(
            ["uv", "run", "--quiet", "ruff", "format", str(path)],
            cwd=root,
            capture_output=True,
            check=False,
        )
        lint = subprocess.run(
            ["uv", "run", "--quiet", "ruff", "check", "--fix", str(path)],
            cwd=root,
            capture_output=True,
            text=True,
            check=False,
        )
        if lint.returncode != 0:
            problems.append((lint.stdout or lint.stderr)[-3000:])
    if problems:
        print("ruff found problems:\n" + "\n".join(problems), file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
