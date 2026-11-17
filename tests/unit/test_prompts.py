"""Prompt files: `.txt` entries, `.md` texts and the `tools.json` tool schemas."""

import json
from pathlib import Path

import pytest

from corpusqa.config import TOOL_NAME_MAX_CHARS
from corpusqa.llm.interfaces import ToolSchema
from corpusqa.prompts import (
    PromptError,
    PromptText,
    ToolSchemas,
    load_prompt,
    load_text_prompt,
    load_tool_schemas,
)

ANSWER_ARGS = {"answer", "citations", "not_found"}


def tool(**overrides: object) -> dict[str, object]:
    """One valid tool entry, with fields replaced or (value None) removed."""
    entry: dict[str, object] = {
        "name": "lookup",
        "description": ["Look a word up.", "Returns its meaning."],
        "parameters": {
            "type": "object",
            "properties": {"word": {"type": "string"}},
            "required": ["word"],
        },
    }
    entry.update(overrides)
    return {k: v for k, v in entry.items() if v is not None}


def write_tools(folder: Path, data: object) -> Path:
    path = folder / "tools.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    return folder


# ---------------------------------------------------------------- .txt (unchanged)
def test_txt_prompts_load_as_before() -> None:
    prompt = load_prompt("embed_roles")
    assert prompt.name == "embed_roles"
    assert prompt.version
    assert "{text}" in prompt.get("document")
    assert "{text}" in prompt.get("query")


# ---------------------------------------------------------------- .md
def test_md_prompt_is_its_version_line_and_the_text_after_it(tmp_path: Path) -> None:
    (tmp_path / "helper.md").write_text(
        "version: 7\n\n# Role\nYou help.\n\n## Rules\n- Use {question} as given.\n\n\n",
        encoding="utf-8",
    )
    assert load_text_prompt("helper", tmp_path) == PromptText(
        name="helper",
        version="7",
        text="# Role\nYou help.\n\n## Rules\n- Use {question} as given.",
    )


def test_md_prompt_with_windows_line_ends_loads(tmp_path: Path) -> None:
    (tmp_path / "crlf.md").write_bytes(b"version: 2\r\nLine one.\r\nLine two.\r\n")
    prompt = load_text_prompt("crlf", tmp_path)
    assert prompt.version == "2"
    assert prompt.text.splitlines() == ["Line one.", "Line two."]


def test_missing_md_prompt_is_an_error(tmp_path: Path) -> None:
    with pytest.raises(PromptError, match="no prompt file"):
        load_text_prompt("absent", tmp_path)


@pytest.mark.parametrize(
    ("content", "message"),
    [
        ("You help.\nversion: 1\n", "first line must be 'version: N'"),
        ("# version: 1\nYou help.\n", "first line must be 'version: N'"),
        ("version:1\nYou help.\n", "first line must be 'version: N'"),
        ("version: \nYou help.\n", "first line must be 'version: N'"),
        ("\nversion: 1\nYou help.\n", "first line must be 'version: N'"),
        ("version: 1\n", "no text after"),
        ("version: 1\n\n  \n", "no text after"),
    ],
)
def test_malformed_md_prompt_is_an_error(tmp_path: Path, content: str, message: str) -> None:
    (tmp_path / "bad.md").write_text(content, encoding="utf-8")
    with pytest.raises(PromptError, match=message):
        load_text_prompt("bad", tmp_path)


# ---------------------------------------------------------------- tools.json (shipped file)
def test_shipped_tools_file_has_the_finish_tool() -> None:
    schemas = load_tool_schemas()
    assert schemas.name == "tools"
    assert schemas.version
    finish = schemas.get("finish")
    assert finish.description.strip()
    assert "\n" in finish.description  # written as a list of lines, joined at load
    params = finish.parameters
    assert params["type"] == "object"
    assert params["additionalProperties"] is False
    properties = params["properties"]
    assert isinstance(properties, dict)
    assert set(properties) == ANSWER_ARGS
    assert set(params["required"]) == ANSWER_ARGS  # type: ignore[arg-type]
    assert properties["answer"]["type"] == "string"
    for name in ("citations", "not_found"):
        assert properties[name] == {
            **properties[name],
            "type": "array",
            "items": {"type": "string"},
        }
    assert all(prop["description"].strip() for prop in properties.values())


def test_unknown_tool_name_is_an_error() -> None:
    with pytest.raises(PromptError, match="no tool 'absent'"):
        load_tool_schemas().get("absent")


# ---------------------------------------------------------------- tools.json (format)
def test_tools_file_loads_in_order_and_joins_description_lines(tmp_path: Path) -> None:
    folder = write_tools(
        tmp_path,
        {
            "version": "4",
            "tools": [tool(), tool(name="_count_2", description="Count things.")],
        },
    )
    schemas = load_tool_schemas(folder)
    assert schemas == ToolSchemas(
        name="tools",
        version="4",
        tools=(
            ToolSchema(
                name="lookup",
                description="Look a word up.\nReturns its meaning.",
                parameters=tool()["parameters"],  # type: ignore[arg-type]
            ),
            ToolSchema(
                name="_count_2",
                description="Count things.",
                parameters=tool()["parameters"],  # type: ignore[arg-type]
            ),
        ),
    )


def test_a_tool_without_arguments_is_allowed(tmp_path: Path) -> None:
    folder = write_tools(tmp_path, {"version": "1", "tools": [tool(parameters={"type": "object"})]})
    assert load_tool_schemas(folder).get("lookup").parameters == {"type": "object"}


def test_missing_tools_file_is_an_error(tmp_path: Path) -> None:
    with pytest.raises(PromptError, match="no tool schema file"):
        load_tool_schemas(tmp_path)


def test_tools_file_that_is_not_json_is_an_error(tmp_path: Path) -> None:
    (tmp_path / "tools.json").write_text('{"version": "1", "tools": [', encoding="utf-8")
    with pytest.raises(PromptError, match="not valid JSON"):
        load_tool_schemas(tmp_path)


def test_tools_file_that_is_not_utf8_is_an_error(tmp_path: Path) -> None:
    (tmp_path / "tools.json").write_bytes(b'{"version": "\xff"}')
    with pytest.raises(PromptError, match="not valid JSON"):
        load_tool_schemas(tmp_path)


LONGEST_NAME = "n" * TOOL_NAME_MAX_CHARS


@pytest.mark.parametrize(
    ("data", "message"),
    [
        ([tool()], "expected an object with exactly 'version' and 'tools'"),
        ({"tools": [tool()]}, "expected an object with exactly 'version' and 'tools'"),
        ({"version": "1", "tools": [tool()], "extra": 1}, "expected an object with exactly"),
        ({"version": 1, "tools": [tool()]}, "'version' must be a non-empty string"),
        ({"version": " ", "tools": [tool()]}, "'version' must be a non-empty string"),
        ({"version": "1", "tools": []}, "'tools' must be a non-empty list"),
        ({"version": "1", "tools": {"lookup": tool()}}, "'tools' must be a non-empty list"),
        ({"version": "1", "tools": ["lookup"]}, "tool 0: expected exactly the keys"),
        ({"version": "1", "tools": [tool(parameters=None)]}, "tool 0: expected exactly the keys"),
        ({"version": "1", "tools": [tool(extra="x")]}, "tool 0: expected exactly the keys"),
        ({"version": "1", "tools": [tool(name="")]}, "name '' must be letters"),
        ({"version": "1", "tools": [tool(name="look up")]}, "name 'look up' must be letters"),
        ({"version": "1", "tools": [tool(name="9lives")]}, "name '9lives' must be letters"),
        ({"version": "1", "tools": [tool(name=LONGEST_NAME + "n")]}, "at most 64 characters"),
        ({"version": "1", "tools": [tool(name=3)]}, "name 3 must be letters"),
        ({"version": "1", "tools": [tool(), tool()]}, "tool 1: the name 'lookup' appears twice"),
        ({"version": "1", "tools": [tool(description="")]}, r"\(lookup\): description must"),
        ({"version": "1", "tools": [tool(description=["a", 2])]}, "description must"),
        ({"version": "1", "tools": [tool(description=[" "])]}, "description must"),
        ({"version": "1", "tools": [tool(parameters=[])]}, "parameters must be a JSON schema"),
        (
            {"version": "1", "tools": [tool(parameters={"type": "string"})]},
            "parameters must be a JSON schema with type 'object'",
        ),
        (
            {"version": "1", "tools": [tool(parameters={"type": "object", "properties": []})]},
            "parameters.properties must be an object",
        ),
        (
            {"version": "1", "tools": [tool(parameters={"type": "object", "required": "word"})]},
            "parameters.required must be a list of names",
        ),
        (
            {"version": "1", "tools": [tool(parameters={"type": "object", "required": ["word"]})]},
            r"required \['word'\] not in parameters.properties",
        ),
    ],
)
def test_malformed_tools_file_is_an_error(tmp_path: Path, data: object, message: str) -> None:
    with pytest.raises(PromptError, match=message):
        load_tool_schemas(write_tools(tmp_path, data))


def test_the_longest_allowed_tool_name_loads(tmp_path: Path) -> None:
    folder = write_tools(tmp_path, {"version": "1", "tools": [tool(name=LONGEST_NAME)]})
    assert load_tool_schemas(folder).tools[0].name == LONGEST_NAME
