"""The eval split rule is enforced: held-out material is blocked, the rule is re-shown."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
HOOKS = ROOT / "scripts" / "hooks"


def hook(name: str, event: dict) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(HOOKS / name)],
        input=json.dumps(event),
        capture_output=True,
        text=True,
        env={**os.environ, "CLAUDE_PROJECT_DIR": str(ROOT)},
        check=False,
    )


def read(path: str) -> dict:
    return {"tool_name": "Read", "tool_input": {"file_path": path}}


def bash(command: str) -> dict:
    return {"tool_name": "Bash", "tool_input": {"command": command}}


@pytest.mark.parametrize(
    ("event", "expected"),
    [
        (read(str(ROOT / "data" / "questions.jsonl")), 2),
        (read("data/questions.jsonl"), 2),
        (read(str(ROOT / "eval" / "sets" / "heldout.json")), 2),
        (read(str(ROOT / "eval" / "results" / "heldout" / "run1" / "summary.md")), 2),
        ({"tool_name": "Grep", "tool_input": {"pattern": "x", "path": "data/questions.jsonl"}}, 2),
        (read(str(ROOT / "eval" / "sets" / "tune.json")), 0),
        (read(str(ROOT / "data" / "corpus" / "jira" / "a.json")), 0),
        ({"tool_name": "Grep", "tool_input": {"pattern": "x", "path": str(ROOT)}}, 0),
        (bash("cat data/questions.jsonl"), 2),
        (bash("jq . data/questions.jsonl | head"), 2),
        (bash("python -m eval.run --set heldout"), 2),
        # `all` holds the held-out questions too: only the `uv run eval-all` key may run it
        (bash("uv run python -m eval.run --set all --milestone m1"), 2),
        (bash("python eval/run.py --set=all"), 2),
        (bash("uv run eval-all --milestone m1"), 0),  # a key; asks the user for permission
        (bash("uv run python -m eval.run --set tune"), 0),
        (bash("python -m eval.run --set tune --url http://allhosts:8080"), 0),
        (bash("cat eval/sets/*.json"), 2),
        (bash("uv run python -c 'print(1)'"), 0),  # not a key, but touches nothing protected
        (bash("uv run python -c \"open('data/questions.jsonl')\""), 2),
        (bash("uv run questions-tune --id pub_0002"), 0),
        (bash("uv run eval-heldout --milestone m1"), 0),  # a key; asks the user for permission
        (bash("uv run questions-tune && cat data/questions.jsonl"), 2),
        (bash("uv run tests"), 0),
        (bash("ls eval/sets"), 0),
    ],
)
def test_guard_split(event: dict, expected: int) -> None:
    assert hook("guard_split.py", event).returncode == expected


def test_reminder_prints_the_rule_from_agents_md() -> None:
    out = hook("split_reminder.py", {}).stdout
    assert "Eval split" in out
    assert "tune (30)" in out
    assert "heldout (49)" in out
    assert "tune=30" in out


def test_question_viewer_shows_tune_and_refuses_heldout(tmp_path: Path) -> None:
    """Runs the viewer on invented files: no check ever opens the real question file."""
    script = ROOT / "scripts" / "show_questions.py"
    sets = tmp_path / "sets"
    sets.mkdir()
    (sets / "tune.json").write_text(
        json.dumps({"question_ids": ["q_0001", "q_0002"]}), encoding="utf-8"
    )
    (sets / "smoke.json").write_text(json.dumps({"question_ids": ["q_0001"]}), encoding="utf-8")
    questions = tmp_path / "questions.jsonl"
    rows = [
        {
            "question_id": f"q_000{n}",
            "question": f"Invented question {n}?",
            "gold_answer": f"Invented answer {n}.",
            "answer_facts": [f"fact {n}"],
            "expected_doc_ids": [],
        }
        for n in (1, 2, 3)  # q_0003 plays the held-out question
    ]
    questions.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")

    def show(*args: str) -> subprocess.CompletedProcess[str]:
        files = ["--questions-file", str(questions), "--sets-dir", str(sets)]
        return subprocess.run(
            [sys.executable, str(script), *files, *args],
            capture_output=True,
            text=True,
            check=False,
        )

    shown = show("--brief")
    assert shown.returncode == 0
    assert shown.stdout.count("## q_") == 2
    assert "Invented question 3" not in shown.stdout
    assert show("--id", "q_0002").returncode == 0
    refused = show("--id", "q_0003")
    assert refused.returncode == 2
    assert "Invented question 3" not in refused.stdout + refused.stderr
    assert show("--set", "heldout").returncode == 2
    assert show("--set", "smoke", "--brief").stdout.count("## q_") == 1
