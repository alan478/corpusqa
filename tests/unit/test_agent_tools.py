"""ToolBox and the finish tool: checks, errors as text, whole results, running at once, spans."""

import asyncio
import logging
import threading
from typing import cast

import pytest
from pydantic import BaseModel, ConfigDict, ValidationError

from corpusqa import obs
from corpusqa.agent.tools import (
    FINISH_RESULT,
    FINISH_TOOL_NAME,
    FinishArgs,
    Tool,
    ToolBox,
    finish_tool,
    read_finish,
    schema_mismatches,
)
from corpusqa.llm.interfaces import Message, ToolCall, ToolSchema
from corpusqa.obs.testing import RecordingTracer
from corpusqa.prompts import load_tool_schemas
from corpusqa.retrieval.errors import ToolInputError

WAIT_S = 5.0  # upper bound on any wait in the overlap test; it passes in milliseconds
# Far longer than the 10,000-character cut step 4 first had, and than the largest corpus
# document (~33,000 characters): results are never cut.
LONG_RESULT_CHARS = 50_000
GOOD_FINISH = {"answer": "It moved to May.", "citations": ["doc-1"], "not_found": []}


# ---------------------------------------------------------------- invented test tools
class EchoArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    word: str
    times: int = 1


class NoArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")


class SizeArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    chars: int


def schema(name: str, properties: dict[str, object], required: list[str]) -> ToolSchema:
    return ToolSchema(
        name=name,
        description=f"The {name} test tool.",
        parameters={
            "type": "object",
            "properties": properties,
            "required": required,
            "additionalProperties": False,
        },
    )


ECHO = schema("echo", {"word": {"type": "string"}, "times": {"type": "integer"}}, ["word"])
BOOM = schema("boom", {}, [])
SIZE = schema("size", {"chars": {"type": "integer"}}, ["chars"])


def echo_tool(seen: list[EchoArgs] | None = None) -> Tool[EchoArgs]:
    def echo(args: EchoArgs) -> str:
        if seen is not None:
            seen.append(args)
        return " ".join([args.word] * args.times)

    return Tool(schema=ECHO, args=EchoArgs, run=echo)


def boom_tool() -> Tool[NoArgs]:
    def boom(args: NoArgs) -> str:
        raise RuntimeError("the vault said hunter2")

    return Tool(schema=BOOM, args=NoArgs, run=boom)


# A message a tool writes for the model on purpose (invented field names).
INPUT_MESSAGE = "no field 'colr' in 'garden'; did you mean 'color'?"
REFUSE = schema("refuse", {}, [])


def refuse_tool() -> Tool[NoArgs]:
    def refuse(args: NoArgs) -> str:
        raise ToolInputError(INPUT_MESSAGE)

    return Tool(schema=REFUSE, args=NoArgs, run=refuse)


def size_tool() -> Tool[SizeArgs]:
    def size(args: SizeArgs) -> str:
        return sample_text(args.chars)

    return Tool(schema=SIZE, args=SizeArgs, run=size)


def sample_text(length: int) -> str:
    """`length` characters of letters, digits, spaces, newlines and non-ASCII (é, €, an emoji)."""
    alphabet = "abcdefghijklmnopqrstuvwxyz 0123456789\né€😀"
    return "".join(alphabet[i * 7 % len(alphabet)] for i in range(length))


def call(name: str, call_id: str = "c1", **arguments: object) -> ToolCall:
    return ToolCall(id=call_id, name=name, arguments=arguments)


def run(box: ToolBox, *calls: ToolCall) -> list[Message]:
    return asyncio.run(box.run(calls))


# ---------------------------------------------------------------- registration
def test_schemas_are_the_registered_ones_in_order() -> None:
    box = ToolBox([echo_tool(), boom_tool(), finish_tool()])
    assert box.schemas() == [ECHO, BOOM, load_tool_schemas().get(FINISH_TOOL_NAME)]


def test_two_tools_with_one_name_are_refused() -> None:
    with pytest.raises(ValueError, match="two tools are named 'echo'"):
        ToolBox([echo_tool(), boom_tool(), echo_tool()])


class OpenEchoArgs(BaseModel):  # extra fields allowed (pydantic's default)
    word: str
    times: int = 1


class RequiredTimesArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    word: str
    times: int


class MoreArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    word: str
    times: int = 1
    loud: bool = False


@pytest.mark.parametrize(
    ("args", "problem"),
    [
        (MoreArgs, "properties"),
        (RequiredTimesArgs, "required"),
        (OpenEchoArgs, "additionalProperties"),
    ],
)
def test_a_tool_whose_model_disagrees_with_its_schema_is_refused(
    args: type[BaseModel], problem: str
) -> None:
    tool = Tool(schema=ECHO, args=args, run=lambda _: "")
    with pytest.raises(ValueError, match=f"tool 'echo': .* disagree: {problem}"):
        ToolBox([tool])


# ---------------------------------------------------------------- running calls
def test_a_call_runs_with_checked_arguments_and_answers_with_one_tool_message() -> None:
    seen: list[EchoArgs] = []
    [message] = run(ToolBox([echo_tool(seen)]), call("echo", "c7", word="hi", times=2.0))
    assert message == Message(role="tool", content="hi hi", tool_call_id="c7", name="echo")
    assert seen == [EchoArgs(word="hi", times=2)]  # 2.0 from JSON checked into an int


def test_no_calls_give_no_messages() -> None:
    assert run(ToolBox([echo_tool()])) == []


@pytest.mark.parametrize(
    ("tool_call", "error"),
    [
        (call("echo", word="hi"), None),
        (call("lookup"), "unknown_tool"),
        (call("echo", times="twice"), "bad_arguments"),
        (call("refuse"), "tool_input"),
        (call("boom"), "tool_failed"),
    ],
)
def test_run_one_gives_the_message_run_gives_and_what_went_wrong(
    tool_call: ToolCall, error: str | None
) -> None:
    """The agent's first search tells a result from an error by `error`, not by the text."""
    box = ToolBox([echo_tool(), boom_tool(), refuse_tool()])
    one = asyncio.run(box.run_one(tool_call))
    assert one.error == error
    assert [one.message] == run(box, tool_call)


def test_an_unknown_tool_is_not_run_and_the_error_lists_the_tools() -> None:
    seen: list[EchoArgs] = []
    box = ToolBox([echo_tool(seen), boom_tool(), finish_tool()])
    [message] = run(box, call("lookup", "c2", word="hi"))
    assert message == Message(
        role="tool",
        content="<tool_error>No tool named 'lookup'. Tools: echo, boom, finish.</tool_error>",
        tool_call_id="c2",
        name="lookup",
    )
    assert seen == []


def test_bad_arguments_are_not_run_and_the_error_names_every_failing_field() -> None:
    seen: list[EchoArgs] = []
    [message] = run(ToolBox([echo_tool(seen)]), call("echo", times="twice", loud=True))
    assert message.content == (
        "<tool_error>Bad arguments for 'echo': word: Field required; "
        "times: Input should be a valid integer, unable to parse string as an integer; "
        "loud: Extra inputs are not permitted. Nothing was run.</tool_error>"
    )
    assert seen == []


def test_a_tool_that_raises_gives_only_its_type_and_the_others_still_run(
    caplog: pytest.LogCaptureFixture,
) -> None:
    box = ToolBox([echo_tool(), boom_tool()])
    with caplog.at_level(logging.ERROR, logger="corpusqa.agent.tools"):
        messages = run(
            box, call("echo", "c1", word="a"), call("boom", "c2"), call("echo", "c3", word="b")
        )
    assert [m.content for m in messages] == [
        "a",
        "<tool_error>'boom' failed: RuntimeError.</tool_error>",
        "b",
    ]
    assert "hunter2" not in messages[1].content  # exception text never reaches the model
    [record] = caplog.records
    assert "'boom' failed" in record.getMessage()
    assert record.exc_info is not None and record.exc_info[0] is RuntimeError  # traceback


def test_a_tool_input_error_shows_its_message_and_no_traceback_is_logged(
    caplog: pytest.LogCaptureFixture,
) -> None:
    box = ToolBox([echo_tool(), refuse_tool(), boom_tool()])
    with caplog.at_level(logging.INFO, logger="corpusqa.agent.tools"):
        messages = run(box, call("refuse", "c1"), call("boom", "c2"), call("echo", "c3", word="a"))
    assert [m.content for m in messages] == [
        f"<tool_error>{INPUT_MESSAGE}</tool_error>",  # the message, written for the model
        "<tool_error>'boom' failed: RuntimeError.</tool_error>",  # any other error: type only
        "a",
    ]
    refused = [r for r in caplog.records if INPUT_MESSAGE in r.getMessage()]
    assert [(r.levelno, r.exc_info) for r in refused] == [(logging.INFO, None)]


def test_a_result_that_is_not_text_is_a_tool_failure() -> None:
    tool = Tool(schema=BOOM, args=NoArgs, run=lambda _: cast(str, None))
    [message] = run(ToolBox([tool]), call("boom"))
    assert message.content == "<tool_error>'boom' failed: TypeError.</tool_error>"


def test_calls_of_one_step_run_at_the_same_time_and_come_back_in_call_order() -> None:
    count = 3
    all_running = threading.Barrier(count, timeout=WAIT_S)  # breaks unless all run at once
    done = [threading.Event() for _ in range(count)]
    finished: list[int] = []
    lock = threading.Lock()

    def wait(args: SizeArgs) -> str:
        index = args.chars
        all_running.wait()
        # Finish last call first: each waits for the call after it.
        if index + 1 < count and not done[index + 1].wait(WAIT_S):
            raise TimeoutError(f"call {index + 1} never finished")
        with lock:
            finished.append(index)
        done[index].set()
        return f"result {index}"

    box = ToolBox([Tool(schema=SIZE, args=SizeArgs, run=wait)])
    messages = run(box, *(call("size", f"c{i}", chars=i) for i in range(count)))
    assert finished == [2, 1, 0]
    assert [(m.tool_call_id, m.content) for m in messages] == [
        ("c0", "result 0"),
        ("c1", "result 1"),
        ("c2", "result 2"),
    ]


# ---------------------------------------------------------------- whole results
def test_a_long_result_reaches_the_model_whole_character_for_character() -> None:
    expected = sample_text(LONG_RESULT_CHARS)
    [message] = run(ToolBox([size_tool()]), call("size", "c1", chars=LONG_RESULT_CHARS))
    assert len(message.content) == LONG_RESULT_CHARS
    assert message.content == expected


# ---------------------------------------------------------------- finish
def test_the_finish_schema_and_finish_args_agree() -> None:
    finish = load_tool_schemas().get(FINISH_TOOL_NAME)
    parameters = finish.parameters
    properties = parameters["properties"]
    assert isinstance(properties, dict)
    fields = FinishArgs.model_fields
    assert set(properties) == set(fields)
    assert set(parameters["required"]) == {n for n, f in fields.items() if f.is_required()}  # type: ignore[arg-type]
    assert parameters["additionalProperties"] is False
    assert FinishArgs.model_config.get("extra") == "forbid"
    assert schema_mismatches(finish, FinishArgs) == []
    assert finish_tool().schema == finish


@pytest.mark.parametrize(
    "arguments",
    [
        {**GOOD_FINISH, "answer": ""},
        {**GOOD_FINISH, "answer": " \n\t "},
        {**GOOD_FINISH, "confidence": "high"},
        {"answer": "Yes."},
    ],
)
def test_finish_args_refuse_a_blank_answer_extra_fields_and_missing_lists(
    arguments: dict[str, object],
) -> None:
    with pytest.raises(ValidationError):
        FinishArgs.model_validate(arguments)


def test_a_good_finish_call_is_received() -> None:
    [message] = run(ToolBox([finish_tool()]), call("finish", "c9", **GOOD_FINISH))
    assert message == Message(role="tool", content=FINISH_RESULT, tool_call_id="c9", name="finish")


def test_read_finish_gives_the_checked_arguments_or_the_text_run_sends() -> None:
    assert read_finish(call("finish", **GOOD_FINISH)) == FinishArgs(
        answer="It moved to May.", citations=["doc-1"], not_found=[]
    )
    bad = call("finish", **{**GOOD_FINISH, "answer": "  ", "confidence": "high"})
    text = read_finish(bad)
    assert text == (
        "<tool_error>Bad arguments for 'finish': answer: must not be blank; "
        "confidence: Extra inputs are not permitted. Nothing was run.</tool_error>"
    )
    [message] = run(ToolBox([finish_tool()]), bad)
    assert message.content == text


def test_read_finish_refuses_another_tools_call() -> None:
    with pytest.raises(ValueError, match="not a 'finish' call"):
        read_finish(call("echo", word="hi"))


# ---------------------------------------------------------------- tracing
def test_each_call_is_one_tool_span_under_the_callers_span() -> None:
    tracer = RecordingTracer()

    def traced_echo(args: EchoArgs) -> str:
        with obs.span("inside", input=args.word):  # runs in a worker thread
            return args.word

    box = ToolBox(
        [
            Tool(schema=ECHO, args=EchoArgs, run=traced_echo),
            boom_tool(),
            size_tool(),
            refuse_tool(),
        ]
    )
    calls = [
        call("echo", "c1", word="hi"),
        call("lookup", "c2"),
        call("echo", "c3", times="x"),
        call("boom", "c4"),
        call("size", "c5", chars=LONG_RESULT_CHARS),
        call("refuse", "c6"),
    ]

    async def step() -> tuple[str | None, list[Message]]:
        with tracer.trace("t"), obs.span("step") as step_span:
            return step_span.id, await box.run(calls)

    step_id, messages = asyncio.run(step())
    spans = {
        o.metadata["tool_call_id"]: o for o in tracer.observations if o.name.startswith("tool:")
    }
    assert [spans[c.id].name for c in calls] == [
        "tool:echo",
        "tool:lookup",
        "tool:echo",
        "tool:boom",
        "tool:size",
        "tool:refuse",
    ]
    for c, message in zip(calls, messages, strict=True):
        span = spans[c.id]
        assert (span.kind, span.parent_id, span.ended) == ("span", step_id, True)
        assert span.output == message.content  # the result exactly as the model saw it

    good = spans["c1"]
    assert good.input == {"given": {"word": "hi"}, "checked": {"word": "hi", "times": 1}}
    assert good.metadata == {"tool_call_id": "c1", "error": None, "result_chars": 2}
    assert good.level == "DEFAULT"
    [inside] = tracer.named("inside")
    assert inside.parent_id == good.id  # the tool body's spans nest under its tool span

    unknown = spans["c2"]
    assert unknown.input == {"given": {}}
    assert (unknown.level, unknown.metadata["error"]) == ("WARNING", "unknown_tool")

    bad = spans["c3"]
    assert bad.input == {"given": {"times": "x"}}  # nothing was checked
    assert (bad.level, bad.metadata["error"]) == ("WARNING", "bad_arguments")

    failed = spans["c4"]
    assert failed.input == {"given": {}, "checked": {}}
    assert (failed.level, failed.metadata["error"]) == ("ERROR", "tool_failed")
    assert (failed.status_message or "").startswith("RuntimeError")

    refused = spans["c6"]
    assert (refused.level, refused.metadata["error"]) == ("WARNING", "tool_input")
    assert refused.status_message == INPUT_MESSAGE

    long = spans["c5"]
    assert long.metadata == {"tool_call_id": "c5", "error": None, "result_chars": LONG_RESULT_CHARS}
    assert long.output == sample_text(LONG_RESULT_CHARS)  # the whole result, in the trace too
