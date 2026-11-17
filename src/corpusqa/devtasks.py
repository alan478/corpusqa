"""Project command runner: `uv run <key> [args...]` runs the command `<key>` in commands.json.

Every key in commands.json is also a console script in pyproject.toml
([project.scripts]) that points to `main`, so the script's own name picks the
command. Dev tooling only: the service never imports this module.

An entry has a `description` and exactly one of:
- `run`: shell commands, run in order from the repo root; stops at the first
  failure; extra CLI args are appended to the last command.
- `steps`: other keys, run in order; stops at the first failure.
- `builtin`: `"list"` prints every command.
Optional `env` gives default environment variables (values already set win).
"""

import json
import os
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any

FILE_NAME = "commands.json"
MAX_DEPTH = 10
# Keys must not shadow tools that `uv run` would otherwise find.
RESERVED = {"test", "python", "python3", "pytest", "ruff", "pyright", "uv", "git", "bash", "sh"}

Commands = dict[str, dict[str, Any]]


def find_commands_file(start: Path) -> Path:
    for folder in (start, *start.parents):
        candidate = folder / FILE_NAME
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"{FILE_NAME} not found in {start} or its parents")


def load_commands(path: Path) -> Commands:
    return json.loads(path.read_text(encoding="utf-8"))["commands"]


def validate(commands: Commands) -> list[str]:
    """Return every problem in the command table (empty list = valid)."""
    errors = []
    for key, spec in commands.items():
        if key in RESERVED:
            errors.append(f"{key}: shadows a real tool; pick another name")
        if not spec.get("description"):
            errors.append(f"{key}: missing description")
        kinds = [k for k in ("run", "steps", "builtin") if k in spec]
        if len(kinds) != 1:
            errors.append(f"{key}: needs exactly one of run / steps / builtin, has {kinds}")
        for step in spec.get("steps", []):
            if step not in commands:
                errors.append(f"{key}: unknown step {step!r}")
    for key in commands:
        if _has_cycle(key, commands, ()):
            errors.append(f"{key}: steps form a cycle")
    return errors


def _has_cycle(key: str, commands: Commands, path: tuple[str, ...]) -> bool:
    if key in path:
        return True
    return any(
        _has_cycle(step, commands, (*path, key))
        for step in commands.get(key, {}).get("steps", [])
        if step in commands
    )


def run_key(key: str, extra: list[str], commands: Commands, root: Path, depth: int = 0) -> int:
    """Run one command and return its exit code."""
    if depth > MAX_DEPTH:
        raise RecursionError(f"steps nested deeper than {MAX_DEPTH} at {key!r}")
    spec = commands[key]
    if spec.get("builtin") == "list":
        return list_commands(commands)
    if "steps" in spec:
        for step in spec["steps"]:
            print(f"\n== {step}", flush=True)
            code = run_key(step, [], commands, root, depth + 1)
            if code:
                print(f"\n{key}: step {step!r} failed (exit {code})", flush=True)
                return code
        print(f"\n{key}: all steps passed", flush=True)
        return 0
    env = {**spec.get("env", {}), **os.environ}  # values already set win
    lines: list[str] = spec["run"]
    for i, line in enumerate(lines):
        command = f"{line} {shlex.join(extra)}" if extra and i == len(lines) - 1 else line
        code = subprocess.run(["bash", "-c", command], cwd=root, env=env, check=False).returncode
        if code:
            return code
    return 0


def list_commands(commands: Commands) -> int:
    width = max(len(key) for key in commands)
    for key, spec in commands.items():
        print(f"  uv run {key:<{width}}  {spec.get('description', '')}")
    return 0


def main() -> int:
    key = Path(sys.argv[0]).name
    path = find_commands_file(Path.cwd())
    commands = load_commands(path)
    errors = validate(commands)
    if errors:
        print(f"{path} has problems:\n  " + "\n  ".join(errors), file=sys.stderr)
        return 2
    if key not in commands:
        print(f"No command {key!r} in {path}. Known commands:", file=sys.stderr)
        list_commands(commands)
        return 2
    try:
        return run_key(key, sys.argv[1:], commands, path.parent)
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    sys.exit(main())
