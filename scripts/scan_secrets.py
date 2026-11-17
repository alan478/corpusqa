"""Fail if a staged (--staged) or tracked (--all) file holds something like a secret.

Used by the git pre-commit hook and `uv run check`. Without a git repository (a copy
unpacked from a zip), --all walks the folder and skips what the root .gitignore skips.
"""

import re
import subprocess
import sys
from pathlib import Path

import pathspec

PATTERNS = [
    re.compile(r"AIza[0-9A-Za-z_\-]{35}"),  # Google API key
    re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
    re.compile(r"\b[A-Z_]*API_KEY\s*[=:]\s*['\"]?[A-Za-z0-9_\-]{20,}"),
    re.compile(r"\bsk-[A-Za-z0-9_\-]{20,}"),  # OpenAI / Anthropic style keys
]
# data/ holds the read-only corpus and the dev-question file; held-out eval material
# is never opened by any check (AGENTS.md: Eval split).
SKIP_PREFIXES = ("data/", "eval/sets/heldout", "eval/results/heldout/")
SKIP_FILES = {"scripts/scan_secrets.py", "tests/unit/test_hooks.py"}  # hold the patterns


def in_git_repo() -> bool:
    try:
        run = subprocess.run(
            ["git", "rev-parse", "--is-inside-work-tree"],
            capture_output=True,
            text=True,
            check=False,
        )
    except FileNotFoundError:  # git is not installed
        return False
    return run.returncode == 0 and run.stdout.strip() == "true"


def walk_files(root: Path) -> list[str]:
    """Every file under root that the root .gitignore keeps, as git would list it."""
    ignore_file = root / ".gitignore"
    lines = ignore_file.read_text(encoding="utf-8").splitlines() if ignore_file.is_file() else []
    ignored = pathspec.GitIgnoreSpec.from_lines(lines)
    files = []
    for folder, dirs, names in root.walk():
        rel = folder.relative_to(root)
        dirs[:] = [
            d
            for d in dirs
            if d != ".git"
            and not ignored.match_file(f"{(rel / d).as_posix()}/")
            and not f"{(rel / d).as_posix()}/".startswith(SKIP_PREFIXES)
        ]
        files += [p for n in names if not ignored.match_file(p := (rel / n).as_posix())]
    return files


def candidate_files(staged: bool) -> list[str]:
    if in_git_repo():
        every_file = ["git", "ls-files", "--cached", "--others", "--exclude-standard"]
        cmd = ["git", "diff", "--cached", "--name-only"] if staged else every_file
        names = subprocess.run(cmd, capture_output=True, text=True, check=True).stdout.splitlines()
    elif staged:
        names = []  # nothing can be staged without a repository
    else:
        print("secrets: no git repository; scanning the folder minus .gitignore", file=sys.stderr)
        names = walk_files(Path.cwd())
    return [f for f in names if f and not f.startswith(SKIP_PREFIXES) and f not in SKIP_FILES]


def find_secrets(text: str) -> list[str]:
    return [p.pattern for p in PATTERNS if p.search(text)]


def main() -> int:
    hits = []
    for name in candidate_files(staged="--staged" in sys.argv):
        path = Path(name)
        if not path.is_file():
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        hits += [f"{name}: matches {pattern}" for pattern in find_secrets(text)]
    for hit in hits:
        print(f"possible secret: {hit}", file=sys.stderr)
    return 1 if hits else 0


if __name__ == "__main__":
    sys.exit(main())
