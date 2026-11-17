"""Shared parsing for project hooks. Claude Code and Codex send similar JSON events.

- Claude Code: tool_input.file_path (Edit, Write, MultiEdit) or notebook_path.
- Codex: tool_name "apply_patch"; tool_input.command holds the patch text, whose
  "*** Add/Update/Delete File: <path>" lines name the files.
"""

import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

PATCH_FILE = re.compile(r"^\*\*\* (?:Add|Update|Delete) File: (.+)$", re.MULTILINE)
PATCH_MOVE = re.compile(r"^\*\*\* Move to: (.+)$", re.MULTILINE)


def load_event() -> dict[str, Any]:
    try:
        event = json.load(sys.stdin)
    except json.JSONDecodeError:
        return {}
    return event if isinstance(event, dict) else {}


def repo_root(event: dict[str, Any]) -> Path:
    env_root = os.environ.get("CLAUDE_PROJECT_DIR")
    if env_root:
        return Path(env_root).resolve()
    cwd = event.get("cwd") or "."
    out = subprocess.run(
        ["git", "-C", str(cwd), "rev-parse", "--show-toplevel"],
        capture_output=True,
        text=True,
        check=False,
    )
    return Path(out.stdout.strip() or cwd).resolve()


def edited_paths(event: dict[str, Any]) -> list[Path]:
    """Absolute paths of the files this tool call writes."""
    args = event.get("tool_input") or {}
    if not isinstance(args, dict):
        return []
    names = [args.get(k) for k in ("file_path", "notebook_path")]
    found = [n for n in names if isinstance(n, str) and n]
    if event.get("tool_name") == "apply_patch":
        patch = str(args.get("command") or args.get("patch") or "")
        found += PATCH_FILE.findall(patch) + PATCH_MOVE.findall(patch)
    base = Path(event.get("cwd") or ".")
    return [(base / n.strip()).resolve() for n in found]


def relative(path: Path, root: Path) -> str | None:
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return None
