#!/usr/bin/env python3
"""Keep held-out eval material away from the agent (AGENTS.md: Eval split).

PreToolUse hook for Claude Code (Read, Grep, NotebookRead, Bash) and Codex
(Bash). Exit 2 blocks the call and tells the model to use `uv run questions-tune`.

Blocked: reading data/questions.jsonl (holds all 79 questions with answers),
eval/sets/heldout.json, anything under eval/results/heldout/, and shell
commands that mention them or "heldout", wildcard into eval/sets or
eval/results, or run the eval runner on the `all` set (it holds the held-out
questions; `uv run eval-all` is the way, and it asks the user). Allowed:
`uv run <key>` for keys in commands.json (project code may read these files;
its output follows the rule).
"""

import json
import re
import sys
from pathlib import Path

from _event import load_event, relative, repo_root

PROTECTED_FILES = ("data/questions.jsonl", "eval/sets/heldout.json")
PROTECTED_DIRS = ("eval/results/heldout/",)
BASH_PATTERNS = [
    re.compile(r"questions\.jsonl"),
    re.compile(r"heldout", re.IGNORECASE),
    re.compile(r"eval/(sets|results)/\S*[*?\[]"),  # wildcards that would include held-out files
    re.compile(r"eval[./]run\b.*--set[=\s]+all\b"),  # the `all` set holds held-out questions
]
HINT = (
    "Held-out eval material is off limits (AGENTS.md: Eval split). "
    "Use `uv run questions-tune` (or `--id pub_XXXX`) for the 30 tune questions."
)


def is_protected(rel: str | None) -> bool:
    if rel is None:
        return False
    return rel in PROTECTED_FILES or any(rel.startswith(d) for d in PROTECTED_DIRS)


def runs_a_project_key(command: str, root: Path) -> bool:
    match = re.match(r"\s*uv\s+run\s+(?:--\S+\s+)*([\w-]+)", command)
    if not match or any(sep in command for sep in (";", "&&", "||", "|", "`", "$(")):
        return False
    try:
        keys = json.loads((root / "commands.json").read_text(encoding="utf-8"))["commands"]
    except (OSError, ValueError, KeyError):
        return False
    return match.group(1) in keys


def main() -> int:
    event = load_event()
    root = repo_root(event)
    tool = event.get("tool_name", "")
    args = event.get("tool_input") or {}
    if not isinstance(args, dict):
        return 0

    if tool in {"Read", "Grep", "NotebookRead"}:
        raw = args.get("file_path") or args.get("notebook_path") or args.get("path")
        if isinstance(raw, str) and raw:
            path = Path(raw) if Path(raw).is_absolute() else root / raw
            if is_protected(relative(path.resolve(), root)):
                print(f"Blocked: {raw}. {HINT}", file=sys.stderr)
                return 2

    if tool == "Bash":
        command = args.get("command")
        if isinstance(command, list):
            command = " ".join(map(str, command))
        if isinstance(command, str) and not runs_a_project_key(command, root):
            for pattern in BASH_PATTERNS:
                if pattern.search(command):
                    print(f"Blocked shell command. {HINT}", file=sys.stderr)
                    return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
