"""Tests for the project's agent hooks and the secret scanner (scripts/)."""

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
GUARD = ROOT / "scripts" / "hooks" / "guard_paths.py"


def run_guard(repo: Path, event: dict) -> int:
    env = {**os.environ, "CLAUDE_PROJECT_DIR": str(repo)}
    run = subprocess.run(
        [sys.executable, str(GUARD)],
        input=json.dumps(event),
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )
    return run.returncode


def edit(path: Path) -> dict:
    return {"tool_name": "Edit", "tool_input": {"file_path": str(path)}}


def codex_patch(repo: Path, rel: str) -> dict:
    patch = f"*** Begin Patch\n*** Update File: {rel}\n@@\n-a\n+b\n*** End Patch"
    return {"tool_name": "apply_patch", "cwd": str(repo), "tool_input": {"command": patch}}


@pytest.mark.parametrize(
    ("rel", "expected"),
    [
        ("data/corpus/jira/X-1.json", 2),
        ("uv.lock", 2),
        ("src/corpusqa/agent/loop.py", 0),
        ("README.md", 0),
    ],
)
def test_guard_paths_claude_edits(tmp_path: Path, rel: str, expected: int) -> None:
    assert run_guard(tmp_path, edit(tmp_path / rel)) == expected


def test_guard_paths_codex_patch_to_data_is_blocked(tmp_path: Path) -> None:
    assert run_guard(tmp_path, codex_patch(tmp_path, "data/corpus/a.json")) == 2
    assert run_guard(tmp_path, codex_patch(tmp_path, "src/corpusqa/x.py")) == 0


def test_guard_paths_eval_sets_can_be_created_once(tmp_path: Path) -> None:
    target = tmp_path / "eval" / "sets" / "smoke.jsonl"
    assert run_guard(tmp_path, edit(target)) == 0  # creating it is fine
    target.parent.mkdir(parents=True)
    target.write_text("{}\n", encoding="utf-8")
    assert run_guard(tmp_path, edit(target)) == 2  # editing it afterwards is not


def test_guard_paths_ignores_files_outside_the_repo(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    assert run_guard(repo, edit(tmp_path / "elsewhere" / "data" / "x.json")) == 0


def _load_scanner():
    spec = importlib.util.spec_from_file_location(
        "scan_secrets", ROOT / "scripts" / "scan_secrets.py"
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_secret_scanner_flags_keys_and_ignores_plain_text() -> None:
    scanner = _load_scanner()
    fake_google_key = "AI" + "za" + "x" * 35
    fake_env_line = "GEMINI_" + "API_KEY=" + "abcdefghijklmnopqrstuvwx"
    assert scanner.find_secrets(f"key = '{fake_google_key}'")
    assert scanner.find_secrets(fake_env_line)
    assert not scanner.find_secrets("Read the key from the GEMINI_API_KEY environment variable.")


FAKE_KEY_LINE = "GEMINI_" + "API_KEY=" + "abcdefghijklmnopqrstuvwx"
# What a copy unpacked from a zip looks like: no .git, a .gitignore, ignored files on disk.
PROJECT_FILES = {
    ".gitignore": ".env\n.env.*\n!.env.example\n.cache/\ndocs/runs/\n*.pyc\n",
    ".env": FAKE_KEY_LINE + "\n",
    ".env.local": FAKE_KEY_LINE + "\n",
    ".env.example": "GEMINI_API_KEY=\n",
    ".cache/spend/a.json": FAKE_KEY_LINE + "\n",
    "docs/runs/task/notes.md": FAKE_KEY_LINE + "\n",
    "docs/keep.md": "notes\n",
    "src/app.py": "x = 1\n",
    "src/app.pyc": "",
    "data/corpus/a.json": "{}\n",
}
KEPT_FILES = [".env.example", ".gitignore", "docs/keep.md", "src/app.py"]


def _project_without_git(root: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    for name, text in PROJECT_FILES.items():
        (root / name).parent.mkdir(parents=True, exist_ok=True)
        (root / name).write_text(text, encoding="utf-8")
    for name in ("GIT_DIR", "GIT_WORK_TREE", "GIT_INDEX_FILE"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("GIT_CEILING_DIRECTORIES", str(root.parent))  # no repo above root
    monkeypatch.chdir(root)


def test_secret_scanner_without_git_walks_the_folder_with_the_gitignore_skips(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _project_without_git(tmp_path, monkeypatch)
    scanner = _load_scanner()
    assert sorted(scanner.candidate_files(staged=False)) == KEPT_FILES
    assert scanner.candidate_files(staged=True) == []  # nothing is staged without a repo


def test_secret_scanner_without_git_flags_kept_files_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _project_without_git(tmp_path, monkeypatch)
    scanner = _load_scanner()
    monkeypatch.setattr(sys, "argv", ["scan_secrets.py", "--all"])
    assert scanner.main() == 0  # the key in .env and the ignored folders is not reported
    (tmp_path / "src" / "app.py").write_text(FAKE_KEY_LINE + "\n", encoding="utf-8")
    assert scanner.main() == 1


def test_secret_scanner_in_a_git_repo_lists_the_same_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _project_without_git(tmp_path, monkeypatch)
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    assert sorted(_load_scanner().candidate_files(staged=False)) == KEPT_FILES


def test_format_hook_ignores_files_outside_the_repo(tmp_path: Path) -> None:
    outside = tmp_path / "scratch.py"
    original = "x=1\nprint( x )\n"  # unformatted, and print() is banned in src/
    outside.write_text(original, encoding="utf-8")
    event = {"tool_name": "Write", "tool_input": {"file_path": str(outside)}}
    run = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "hooks" / "format_python.py")],
        input=json.dumps(event),
        capture_output=True,
        text=True,
        env={**os.environ, "CLAUDE_PROJECT_DIR": str(ROOT)},
        check=False,
    )
    assert run.returncode == 0
    assert outside.read_text(encoding="utf-8") == original
