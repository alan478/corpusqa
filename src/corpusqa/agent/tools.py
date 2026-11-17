"""The agent's tools: what each one takes, how a step's calls run, and the `finish` tool.

A `Tool` is a schema (what the model sees, from `prompts/tools.json`), a Pydantic model
of its arguments, and plain code: checked arguments in, text out. A `ToolBox` holds the
tools of one agent and runs the calls of one model reply (plan section 5):

- All calls of a step run at the same time (each tool body in `asyncio.to_thread`, since
  real tools read SQLite), and the results come back in call order.
- Every problem comes back as text the model can act on, never as an exception: an
  unknown tool, arguments the model got wrong (nothing is run), a `ToolInputError` (its
  message, written for the model, is shown: e.g. a field or value the index does not
  have), any other exception (only its type is shown; the traceback goes to the log).
- Every result goes to the model whole: the box never cuts it. Each tool controls its own
  output size (e.g. paging: "rows 1-50 of 278, next offset 50"), and compaction shrinks
  old history.
- Each call is one `tool:<name>` span: arguments as given and as checked, the result as
  the model saw it, its length, and the error if any. The tool body runs inside
  that span's context, so a tool adds its internals with `obs.current().update(...)` or
  its own `obs.span(...)`.

The agent (step 5) decides which calls of a reply reach `run` (e.g. none after a good
`finish`); it reads `finish` arguments with `read_finish`, which checks them exactly as
`run` does. `run_one` runs one call the same way and also says what went wrong (the
agent's first search uses it to tell a result from an error without reading the text).
"""

import asyncio
import logging
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, ValidationError, field_validator
from pydantic_core import PydanticCustomError

from corpusqa import obs
from corpusqa.llm.interfaces import Message, ToolCall, ToolSchema
from corpusqa.prompts import ToolSchemas, load_tool_schemas
from corpusqa.retrieval.errors import ToolInputError

logger = logging.getLogger(__name__)

FINISH_TOOL_NAME = "finish"
# What `finish` returns (tools.json's finish description promises this text).
FINISH_RESULT = "Answer received."
# Each tool call is one span named "<prefix><tool name>".
TOOL_SPAN_PREFIX = "tool:"

# The texts the model sees when a call goes wrong. Exception text never goes in: it can
# hold anything (paths, values, keys); the log and the span (scrubbed) keep it. The one
# exception is ToolInputError, whose message a tool writes for the model on purpose.
_UNKNOWN_TOOL = "<tool_error>No tool named '{name}'. Tools: {tools}.</tool_error>"
_BAD_ARGUMENTS = "<tool_error>Bad arguments for '{name}': {problems}. Nothing was run.</tool_error>"
_TOOL_INPUT = "<tool_error>{message}</tool_error>"
_TOOL_FAILED = "<tool_error>'{name}' failed: {kind}.</tool_error>"
# Where a validation error with no field (e.g. the arguments are not an object) points.
_WHOLE_ARGUMENTS = "arguments"

# What went wrong with a call: the span's `error` metadata (None when nothing did).
type ToolErrorKind = Literal["unknown_tool", "bad_arguments", "tool_input", "tool_failed"]
UNKNOWN_TOOL: ToolErrorKind = "unknown_tool"
BAD_ARGUMENTS: ToolErrorKind = "bad_arguments"
TOOL_INPUT: ToolErrorKind = "tool_input"
TOOL_FAILED: ToolErrorKind = "tool_failed"


@dataclass(frozen=True, slots=True)
class ToolRun:
    """One call as `ToolBox.run_one` ran it: its tool message, and what went wrong."""

    message: Message
    error: ToolErrorKind | None  # None: the tool ran and returned its text


@dataclass(frozen=True, slots=True)
class Tool[A: BaseModel]:
    """One tool: its schema, the model that checks its arguments, and its code."""

    schema: ToolSchema
    args: type[A]  # the arguments, checked before the tool runs
    run: Callable[[A], str]  # plain code: checked arguments in, text out (runs in a thread)


class ToolBox:
    """The tools of one agent. Holds no per-call state: one box may serve many runs."""

    def __init__(self, tools: Sequence[Tool[Any]]) -> None:
        self._tools: dict[str, Tool[Any]] = {}
        for tool in tools:
            name = tool.schema.name
            if name in self._tools:
                raise ValueError(f"two tools are named {name!r}")
            problems = schema_mismatches(tool.schema, tool.args)
            if problems:
                raise ValueError(
                    f"tool {name!r}: its schema and {tool.args.__name__} disagree: "
                    + "; ".join(problems)
                )
            self._tools[name] = tool

    def schemas(self) -> list[ToolSchema]:
        """The schemas the model is sent, in the order the tools were given."""
        return [tool.schema for tool in self._tools.values()]

    async def run(self, calls: Sequence[ToolCall]) -> list[Message]:
        """Run the calls at the same time; one tool message per call, in call order."""
        runs = await asyncio.gather(*(self.run_one(call) for call in calls))
        return [run.message for run in runs]

    async def run_one(self, call: ToolCall) -> ToolRun:
        """One call, as `run` runs it (the same span and texts), and what went wrong."""
        given = dict(call.arguments)
        with obs.span(
            f"{TOOL_SPAN_PREFIX}{call.name}",
            input={"given": given},
            metadata={"tool_call_id": call.id},
        ) as span:
            text, error = await self._outcome(call, given, span)
            span.update(output=text, metadata={"error": error, "result_chars": len(text)})
        message = Message(role="tool", tool_call_id=call.id, name=call.name, content=text)
        return ToolRun(message=message, error=error)

    async def _outcome(
        self, call: ToolCall, given: dict[str, object], span: obs.Observation
    ) -> tuple[str, ToolErrorKind | None]:
        """The result text of one call, and what went wrong (None: nothing)."""
        tool = self._tools.get(call.name)
        if tool is None:
            logger.info("tool call %s: no tool named %r", call.id, call.name)
            span.warning(f"no tool named {call.name!r}")
            names = ", ".join(self._tools)
            return _UNKNOWN_TOOL.format(name=call.name, tools=names), UNKNOWN_TOOL
        checked = check_arguments(call.name, tool.args, given)
        if isinstance(checked, str):
            logger.info("tool call %s to %r: bad arguments", call.id, call.name)
            span.warning("bad arguments")
            return checked, BAD_ARGUMENTS
        span.update(input={"given": given, "checked": checked.model_dump(mode="json")})
        try:
            result = await asyncio.to_thread(tool.run, checked)
            if not isinstance(result, str):
                raise TypeError(f"{call.name!r} returned {type(result).__name__}, not str")
        except ToolInputError as exc:
            logger.info("tool call %s to %r: %s", call.id, call.name, exc.message)
            span.warning(exc.message)
            return _TOOL_INPUT.format(message=exc.message), TOOL_INPUT
        except Exception as exc:
            logger.exception("tool call %s to %r failed", call.id, call.name)
            span.error(exc)
            return _TOOL_FAILED.format(name=call.name, kind=type(exc).__name__), TOOL_FAILED
        return result, None


def check_arguments[A: BaseModel](
    name: str, model: type[A], arguments: Mapping[str, object]
) -> A | str:
    """`arguments` checked by `model`, or the bad-arguments text (every failing field)."""
    try:
        return model.model_validate(dict(arguments))
    except ValidationError as exc:
        problems = "; ".join(
            f"{'.'.join(str(part) for part in e['loc']) or _WHOLE_ARGUMENTS}: {e['msg']}"
            for e in exc.errors(include_url=False)
        )
        return _BAD_ARGUMENTS.format(name=name, problems=problems)


def schema_mismatches(schema: ToolSchema, args: type[BaseModel]) -> list[str]:
    """How a tool's JSON schema and its arguments model disagree (empty: they agree).

    Compared: the property names, the required names, and whether extra fields are refused
    (`additionalProperties: false` and `extra="forbid"`). Types and descriptions are not.
    """
    parameters = schema.parameters
    properties = parameters.get("properties")
    schema_fields = set(properties) if isinstance(properties, Mapping) else set()
    required = parameters.get("required")
    schema_required = set(required) if isinstance(required, list) else set()
    fields = {field.alias or name: field for name, field in args.model_fields.items()}
    model_required = {name for name, field in fields.items() if field.is_required()}
    problems: list[str] = []
    if schema_fields != set(fields):
        problems.append(f"properties {sorted(schema_fields)} != fields {sorted(fields)}")
    if schema_required != model_required:
        problems.append(f"required {sorted(schema_required)} != {sorted(model_required)}")
    schema_closed = parameters.get("additionalProperties") is False
    model_closed = args.model_config.get("extra") == "forbid"
    if schema_closed != model_closed:
        problems.append(
            f"additionalProperties false: {schema_closed}, but extra='forbid': {model_closed}"
        )
    return problems


# ---------------------------------------------------------------- finish
class FinishArgs(BaseModel):
    """The arguments of `finish`: the final answer, the ids it relies on, what is missing."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    answer: str
    citations: list[str]
    not_found: list[str]

    @field_validator("answer")
    @classmethod
    def _answer_not_blank(cls, value: str) -> str:
        if not value.strip():
            raise PydanticCustomError("blank", "must not be blank")
        return value


def finish_tool(schemas: ToolSchemas | None = None) -> Tool[FinishArgs]:
    """The `finish` tool, with its schema from `tools.json` (or the given schemas)."""
    schema = (schemas or load_tool_schemas()).get(FINISH_TOOL_NAME)
    return Tool(schema=schema, args=FinishArgs, run=_receive_answer)


def read_finish(call: ToolCall) -> FinishArgs | str:
    """A `finish` call's checked arguments, or the error text `ToolBox.run` sends for it."""
    if call.name != FINISH_TOOL_NAME:
        raise ValueError(f"not a {FINISH_TOOL_NAME!r} call: {call.name!r}")
    return check_arguments(call.name, FinishArgs, call.arguments)


def _receive_answer(args: FinishArgs) -> str:
    return FINISH_RESULT
