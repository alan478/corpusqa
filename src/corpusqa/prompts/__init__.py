"""Versioned prompt files, loaded by name (AGENTS.md: Using the LLM).

Three kinds of file live in this folder. Any text change bumps the file's version (skill
prompt-change), and every LLM or embedding call records the name and version on its trace.

- `<name>.txt` (`load_prompt`): short keyed entries. Each line is `key: value` (split at
  the first ": "); the first key is `version`; blank lines and `#` lines are ignored.
- `<name>.md` (`load_text_prompt`): one long text, such as a system prompt. The first line
  is `version: N`; the rest of the file is the text (blank lines around it dropped).
- `tools.json` (`load_tool_schemas`): the agent's tool schemas, as
  `{"version": "N", "tools": [{"name", "description", "parameters"}, ...]}`. A description
  is a string or a list of lines (joined with newlines); `parameters` is the JSON schema of
  the arguments (an object schema).

Loaded files are cached for the life of the process; treat what they return as read-only.
"""

import functools
import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from corpusqa.config import TOOL_NAME_MAX_CHARS
from corpusqa.llm.interfaces import ToolSchema

_FOLDER = Path(__file__).resolve().parent
_SUFFIX = ".txt"
_TEXT_SUFFIX = ".md"
_TOOLS_FILE = "tools.json"
_TOOLS_NAME = "tools"
_VERSION_KEY = "version"
_TOOLS_KEY = "tools"
_TOOL_KEYS = frozenset({"name", "description", "parameters"})
_TOOL_NAME = re.compile(rf"[A-Za-z_][A-Za-z0-9_]{{0,{TOOL_NAME_MAX_CHARS - 1}}}")


class PromptError(ValueError):
    """A prompt file is missing or malformed."""


@dataclass(frozen=True, slots=True)
class PromptFile:
    name: str
    version: str
    entries: Mapping[str, str]

    def get(self, key: str) -> str:
        if key not in self.entries:
            raise PromptError(f"prompt {self.name!r} has no entry {key!r}")
        return self.entries[key]


@dataclass(frozen=True, slots=True)
class PromptText:
    """A `.md` prompt: one text and its version."""

    name: str
    version: str
    text: str


@dataclass(frozen=True, slots=True)
class ToolSchemas:
    """The tool schemas of `tools.json`, in file order, and the file's version."""

    name: str
    version: str
    tools: tuple[ToolSchema, ...]

    def get(self, tool: str) -> ToolSchema:
        for schema in self.tools:
            if schema.name == tool:
                return schema
        raise PromptError(f"{self.name!r} has no tool {tool!r}")


@functools.cache
def load_prompt(name: str) -> PromptFile:
    path = _FOLDER / f"{name}{_SUFFIX}"
    if not path.is_file():
        raise PromptError(f"no prompt file {path}")
    entries: dict[str, str] = {}
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip() or line.startswith("#"):
            continue
        key, sep, value = line.partition(": ")
        if not sep or not key.strip():
            raise PromptError(f"{path}:{number}: expected 'key: value'")
        if key in entries:
            raise PromptError(f"{path}:{number}: {key!r} appears twice")
        if not entries and key != _VERSION_KEY:
            raise PromptError(f"{path}:{number}: the first entry must be {_VERSION_KEY!r}")
        entries[key] = value
    if _VERSION_KEY not in entries:
        raise PromptError(f"{path}: no {_VERSION_KEY!r} line")
    version = entries.pop(_VERSION_KEY)
    return PromptFile(name=name, version=version, entries=entries)


@functools.cache
def load_text_prompt(name: str, folder: Path = _FOLDER) -> PromptText:
    """`<folder>/<name>.md`: the first line is `version: N`, the rest is the text."""
    path = folder / f"{name}{_TEXT_SUFFIX}"
    if not path.is_file():
        raise PromptError(f"no prompt file {path}")
    first, _, rest = path.read_text(encoding="utf-8").partition("\n")
    key, sep, version = first.rstrip("\r").partition(": ")
    if key != _VERSION_KEY or not sep or not version.strip():
        raise PromptError(f"{path}:1: the first line must be '{_VERSION_KEY}: N', got {first!r}")
    text = rest.strip()
    if not text:
        raise PromptError(f"{path}: no text after the {_VERSION_KEY!r} line")
    return PromptText(name=name, version=version.strip(), text=text)


@functools.cache
def load_tool_schemas(folder: Path = _FOLDER) -> ToolSchemas:
    """`<folder>/tools.json`, checked: see the module docstring for the format."""
    path = folder / _TOOLS_FILE
    if not path.is_file():
        raise PromptError(f"no tool schema file {path}")
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except ValueError as exc:  # JSONDecodeError and bad UTF-8 are both ValueErrors
        raise PromptError(f"{path}: not valid JSON ({exc})") from exc
    if not isinstance(data, dict) or set(data) != {_VERSION_KEY, _TOOLS_KEY}:
        keys = sorted(data) if isinstance(data, dict) else type(data).__name__
        raise PromptError(
            f"{path}: expected an object with exactly {_VERSION_KEY!r} and {_TOOLS_KEY!r}, "
            f"got {keys}"
        )
    version = data[_VERSION_KEY]
    if not isinstance(version, str) or not version.strip():
        raise PromptError(f"{path}: {_VERSION_KEY!r} must be a non-empty string")
    raw_tools = data[_TOOLS_KEY]
    if not isinstance(raw_tools, list) or not raw_tools:
        raise PromptError(f"{path}: {_TOOLS_KEY!r} must be a non-empty list")
    tools: list[ToolSchema] = []
    for index, raw in enumerate(raw_tools):
        schema = _tool_schema(raw, f"{path}: tool {index}")
        if any(seen.name == schema.name for seen in tools):
            raise PromptError(f"{path}: tool {index}: the name {schema.name!r} appears twice")
        tools.append(schema)
    return ToolSchemas(name=_TOOLS_NAME, version=version.strip(), tools=tuple(tools))


def _tool_schema(raw: object, where: str) -> ToolSchema:
    if not isinstance(raw, dict) or set(raw) != _TOOL_KEYS:
        keys = sorted(raw) if isinstance(raw, dict) else type(raw).__name__
        raise PromptError(f"{where}: expected exactly the keys {sorted(_TOOL_KEYS)}, got {keys}")
    name = raw["name"]
    if not isinstance(name, str) or not _TOOL_NAME.fullmatch(name):
        raise PromptError(
            f"{where}: name {name!r} must be letters, digits and underscores, start with a "
            f"letter or underscore, and have at most {TOOL_NAME_MAX_CHARS} characters"
        )
    where = f"{where} ({name})"
    description = raw["description"]
    if isinstance(description, list) and all(isinstance(line, str) for line in description):
        description = "\n".join(description)
    if not isinstance(description, str) or not description.strip():
        raise PromptError(f"{where}: description must be a non-empty string or list of lines")
    parameters = raw["parameters"]
    if not isinstance(parameters, dict) or parameters.get("type") != "object":
        raise PromptError(f"{where}: parameters must be a JSON schema with type 'object'")
    properties = parameters.get("properties", {})
    if not isinstance(properties, dict):
        raise PromptError(f"{where}: parameters.properties must be an object")
    required = parameters.get("required", [])
    if not isinstance(required, list) or not all(isinstance(r, str) for r in required):
        raise PromptError(f"{where}: parameters.required must be a list of names")
    missing = [r for r in required if r not in properties]
    if missing:
        raise PromptError(f"{where}: required {missing} not in parameters.properties")
    return ToolSchema(name=name, description=description.strip(), parameters=parameters)
