"""Tests for commands.json and its runner (`uv run <key>`)."""

import subprocess
import tomllib
from pathlib import Path

import pytest

from corpusqa import devtasks

ROOT = Path(__file__).resolve().parents[2]


def real_commands() -> devtasks.Commands:
    return devtasks.load_commands(ROOT / "commands.json")


def test_commands_json_is_valid() -> None:
    assert devtasks.validate(real_commands()) == []


def test_every_key_is_a_console_script_and_back() -> None:
    scripts = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"][
        "scripts"
    ]
    runner_scripts = {
        name for name, target in scripts.items() if target == "corpusqa.devtasks:main"
    }
    assert runner_scripts == set(real_commands())


def test_validate_finds_bad_entries() -> None:
    bad: devtasks.Commands = {
        "pytest": {"description": "shadows a tool", "run": ["echo"]},
        "no-desc": {"run": ["echo"]},
        "both": {"description": "x", "run": ["echo"], "steps": ["no-desc"]},
        "ghost": {"description": "x", "steps": ["missing"]},
        "a": {"description": "x", "steps": ["b"]},
        "b": {"description": "x", "steps": ["a"]},
    }
    errors = "\n".join(devtasks.validate(bad))
    for expected in ("pytest: shadows", "no-desc: missing description", "both: needs exactly one",
                     "ghost: unknown step 'missing'", "a: steps form a cycle"):  # fmt: skip
        assert expected in errors


def test_steps_run_in_order_and_stop_at_first_failure(tmp_path: Path) -> None:
    log = tmp_path / "log.txt"
    commands: devtasks.Commands = {
        "one": {"description": "x", "run": [f"echo one >> {log}"]},
        "fail": {"description": "x", "run": ["exit 3"]},
        "never": {"description": "x", "run": [f"echo never >> {log}"]},
        "all": {"description": "x", "steps": ["one", "fail", "never"]},
    }
    assert devtasks.run_key("all", [], commands, tmp_path) == 3
    assert log.read_text(encoding="utf-8").split() == ["one"]


def test_extra_args_go_to_the_last_command_and_env_defaults_do_not_override(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    out = tmp_path / "out.txt"
    monkeypatch.setenv("ALREADY_SET", "from-shell")
    commands: devtasks.Commands = {
        "show": {
            "description": "x",
            "run": [
                f"echo $ALREADY_SET $DEFAULTED > {out}",
                f"printf '%s\\n' >> {out}",  # extra args land here, one per line
            ],
            "env": {"ALREADY_SET": "from-json", "DEFAULTED": "from-json"},
        }
    }
    assert devtasks.run_key("show", ["--flag", "two words"], commands, tmp_path) == 0
    assert out.read_text(encoding="utf-8").splitlines() == [
        "from-shell from-json",
        "--flag",
        "two words",
    ]


def test_setup_turns_on_git_hooks_only_inside_a_git_repository(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A copy unpacked from a zip has no .git: setup must still finish.
    for name in ("GIT_DIR", "GIT_WORK_TREE", "GIT_INDEX_FILE"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("GIT_CEILING_DIRECTORIES", str(tmp_path.parent))
    hooks_line = real_commands()["setup"]["run"][-1]
    commands: devtasks.Commands = {"hooks": {"description": "x", "run": [hooks_line]}}
    assert devtasks.run_key("hooks", [], commands, tmp_path) == 0

    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    assert devtasks.run_key("hooks", [], commands, tmp_path) == 0
    hooks_path = subprocess.run(
        ["git", "config", "core.hooksPath"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    ).stdout.strip()
    assert hooks_path == ".githooks"
