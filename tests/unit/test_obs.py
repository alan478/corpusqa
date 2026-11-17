"""The tracing facade (corpusqa.obs): nesting, safety, scrubbing, and tracing off.

Uses the in-memory RecordingTracer; the Langfuse backend has its own tests in
test_obs_langfuse.py. Fake keys only, built at run time so the secret scan stays quiet.
"""

import asyncio
import dataclasses
import logging
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import pytest

from corpusqa import obs
from corpusqa.config import Settings
from corpusqa.obs.scrub import REDACTED, TOO_DEEP, Scrubber
from corpusqa.obs.testing import RecordedObservation, RecordingTracer
from corpusqa.obs.tracer import Handle, ObservationStart, ObservationUpdate

FAKE_GEMINI = "fake-gemini-" + "g" * 24
FAKE_LF_PUBLIC = "pk-lf-" + "p" * 24
FAKE_LF_SECRET = "sk-lf-" + "s" * 24
GOOGLE_SHAPED = "AIza" + "x" * 35  # matches the Google key pattern, configured nowhere


def settings_with_keys(tmp_path: Path) -> Settings:
    return Settings(
        corpus_dir=tmp_path,
        port=8080,
        gemini_api_key=FAKE_GEMINI,
        langfuse_public_key=FAKE_LF_PUBLIC,
        langfuse_secret_key=FAKE_LF_SECRET,
    )


# ---------------------------------------------------------------- nesting


def test_trace_uses_the_given_id_and_spans_nest_under_it() -> None:
    tracer = RecordingTracer()
    trace_id = obs.new_trace_id()
    with tracer.trace("ask", trace_id=trace_id, input="Why?") as root:
        assert obs.current() is root
        with obs.span("search", input={"q": "why"}) as search:
            search.update(output=["hit"], metadata={"count": 1})
            with obs.generation(
                "draft",
                model="fake-model",
                input=[{"role": "user", "content": "Why?"}],
                model_parameters={"temperature": 0.0},
                prompt_name="answer",
                prompt_version="3",
            ) as gen:
                gen.update(output="Because.")
                gen.record_usage(input_tokens=10, output_tokens=5, cost_usd=0.002)
        root.update(output="Because.", metadata={"outcome": "ok"})
    assert obs.current().recording is False  # nothing open after the block

    ask, search_rec, draft = tracer.observations
    assert (ask.kind, ask.name, ask.trace_id, ask.parent_id) == ("trace", "ask", trace_id, None)
    assert root.trace_id == trace_id
    assert (search_rec.parent_id, search_rec.trace_id) == (ask.id, trace_id)
    assert (draft.parent_id, draft.trace_id, draft.kind) == (search_rec.id, trace_id, "generation")
    assert search_rec.input == {"q": "why"} and search_rec.output == ["hit"]
    assert search_rec.metadata == {"count": 1}
    assert draft.model == "fake-model" and draft.model_parameters == {"temperature": 0.0}
    assert draft.metadata == {"prompt_name": "answer", "prompt_version": "3"}
    assert draft.version == "3"
    assert draft.usage == {"input": 10, "output": 5, "total": 15}
    assert draft.cost == {"total": 0.002}
    assert ask.output == "Because." and ask.metadata == {"outcome": "ok"}
    assert all(o.ended for o in tracer.observations)


def test_span_without_an_open_trace_starts_its_own_trace() -> None:
    tracer = RecordingTracer()
    with tracer.span("startup-stage") as stage:
        assert stage.trace_id is not None and len(stage.trace_id) == 32
    [recorded] = tracer.observations
    assert recorded.parent_id is None


def test_an_invalid_trace_id_gets_a_fresh_one_and_is_reported(
    caplog: pytest.LogCaptureFixture,
) -> None:
    tracer = RecordingTracer()
    with tracer.trace("ask", trace_id="not-hex") as root:
        pass
    assert root.trace_id is not None and obs.tracer.is_trace_id(root.trace_id)
    assert "trace id failed" in caplog.text


def test_module_helpers_are_no_ops_outside_a_trace() -> None:
    assert obs.current_tracer() is obs.NOOP_TRACER
    with obs.span("orphan") as orphan:
        orphan.update(output="ignored")
        orphan.error("ignored")
    assert orphan.recording is False and orphan.trace_id is None


def test_metadata_merges_by_key_and_none_leaves_fields_alone() -> None:
    tracer = RecordingTracer()
    with tracer.span("s", input="in", metadata={"a": 1}) as s:
        s.update(metadata={"b": 2})
        s.update(output="out")
        s.update(input=None, output=None)
    [rec] = tracer.observations
    assert (rec.input, rec.output, rec.metadata) == ("in", "out", {"a": 1, "b": 2})


# ---------------------------------------------------------------- errors


def test_exceptions_propagate_and_mark_the_observation_as_error() -> None:
    tracer = RecordingTracer()
    with pytest.raises(ValueError, match="bad input"), tracer.trace("ask"), obs.span("inner"):
        raise ValueError("bad input")
    ask, inner = tracer.observations
    for rec in (ask, inner):
        assert rec.level == "ERROR"
        assert rec.status_message == "ValueError: bad input"
        assert rec.ended


def test_error_and_warning_set_level_and_message() -> None:
    tracer = RecordingTracer()
    with tracer.span("a") as a:
        a.warning("slow")
    with tracer.span("b") as b:
        b.error(RuntimeError())
    first, second = tracer.observations
    assert (first.level, first.status_message) == ("WARNING", "slow")
    assert (second.level, second.status_message) == ("ERROR", "RuntimeError")


def test_cancellation_marks_open_spans_and_still_propagates() -> None:
    tracer = RecordingTracer()

    async def slow() -> None:
        with obs.span("waiting"):
            await asyncio.sleep(5)

    async def scenario() -> None:
        with tracer.trace("ask"):
            await asyncio.wait_for(slow(), timeout=0.01)

    with pytest.raises(TimeoutError):
        asyncio.run(scenario())
    waiting = tracer.named("waiting")[0]
    assert (waiting.level, waiting.status_message, waiting.ended) == ("ERROR", "cancelled", True)


# ---------------------------------------------------------------- observe decorator


def test_observe_wraps_sync_and_async_functions() -> None:
    @obs.observe()
    def add(a: int, b: int = 2) -> int:
        return a + b

    class Tool:
        @obs.observe("tool.lookup", capture_output=False)
        async def lookup(self, key: str) -> str:
            await asyncio.sleep(0)
            return f"value of {key}"

    tracer = RecordingTracer()

    async def scenario() -> tuple[int, str]:
        with tracer.trace("ask"):
            return add(1), await Tool().lookup("k1")

    assert asyncio.run(scenario()) == (3, "value of k1")
    ask, added, looked_up = tracer.observations
    assert added.name.endswith("add") and added.parent_id == ask.id
    assert (added.input, added.output) == ({"a": 1}, 3)
    assert looked_up.name == "tool.lookup" and looked_up.parent_id == ask.id
    assert looked_up.input == {"key": "k1"} and looked_up.output is None


def test_observe_outside_a_trace_just_calls_the_function() -> None:
    @obs.observe()
    def double(x: int) -> int:
        return 2 * x

    assert double(4) == 8


def test_observe_propagates_errors() -> None:
    @obs.observe()
    def broken() -> None:
        raise KeyError("missing")

    tracer = RecordingTracer()
    with pytest.raises(KeyError), tracer.trace("ask"):
        broken()
    assert tracer.named("test_observe_propagates_errors.<locals>.broken")[0].level == "ERROR"


# ---------------------------------------------------------------- context: tasks and threads


def test_spans_follow_their_own_trace_across_tasks_and_threads() -> None:
    tracer = RecordingTracer()

    def thread_work(i: int, name: str) -> None:
        with obs.span(name, input=f"q{i}"):
            time.sleep(0.001)

    async def one(i: int) -> str:
        with tracer.trace("ask", input=f"q{i}") as root:
            with obs.span("work", input=f"q{i}"):
                await asyncio.sleep(0.01)  # let the four interleave
                await asyncio.create_task(asyncio.to_thread(thread_work, i, "to-thread"))
                plain = threading.Thread(target=obs.bind(thread_work), args=(i, "bound-thread"))
                plain.start()
                await asyncio.to_thread(plain.join)
            assert root.trace_id is not None
            return root.trace_id

    async def scenario() -> list[str]:
        return list(await asyncio.gather(*(one(i) for i in range(4))))

    trace_ids = asyncio.run(scenario())
    assert len(set(trace_ids)) == 4
    for i, trace_id in enumerate(trace_ids):
        in_trace = {o.name: o for o in tracer.in_trace(trace_id)}
        assert set(in_trace) == {"ask", "work", "to-thread", "bound-thread"}
        assert {o.input for o in in_trace.values()} == {f"q{i}"}
        work = in_trace["work"]
        assert work.parent_id == in_trace["ask"].id
        assert in_trace["to-thread"].parent_id == work.id
        assert in_trace["bound-thread"].parent_id == work.id


def test_a_plain_thread_without_bind_does_not_join_the_trace() -> None:
    tracer = RecordingTracer()

    def work() -> None:
        with obs.span("lost"):
            pass

    with tracer.trace("ask"):
        thread = threading.Thread(target=work)
        thread.start()
        thread.join()
    assert tracer.named("lost") == []  # no current tracer in a fresh thread: a no-op


# ---------------------------------------------------------------- failures never break callers


class _BrokenHandle:
    trace_id = "0" * 31 + "1"
    id = "0" * 16

    def update(self, update: ObservationUpdate) -> None:
        raise RuntimeError("update exploded")


class BrokenTracer(RecordingTracer):
    """A backend where every call fails, in the given places."""

    def __init__(self, fail: frozenset[str]) -> None:
        super().__init__()
        self.fail = fail

    @contextmanager
    def _open(self, start: ObservationStart, parent: Handle | None) -> Iterator[Handle]:
        if "open" in self.fail:
            raise RuntimeError("open exploded")
        yield _BrokenHandle()
        if "close" in self.fail:
            raise RuntimeError("close exploded")

    def _flush(self) -> None:
        raise RuntimeError("flush exploded")

    def _shutdown(self) -> None:
        raise RuntimeError("shutdown exploded")


@pytest.mark.parametrize(
    "fail", [{"open"}, {"close"}, set()], ids=["open", "update+close", "update"]
)
def test_backend_failures_are_swallowed_and_logged_once(
    fail: set[str], caplog: pytest.LogCaptureFixture
) -> None:
    tracer = BrokenTracer(frozenset(fail))
    ran = []
    for _ in range(3):
        with tracer.trace("ask", input="q") as root:
            with obs.span("inner") as inner:
                inner.update(output="x")
                inner.record_usage(input_tokens=1, output_tokens=1)
                ran.append(True)
            root.error("still fine")
    assert ran == [True, True, True]
    assert tracer.flush(timeout_s=1) is False
    assert tracer.shutdown(timeout_s=1) is False
    warnings = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
    failed_ops = [m.split(" failed")[0] for m in warnings if m.startswith("tracing:")]
    assert len(failed_ops) == len(set(failed_ops)), failed_ops  # each kind logged once


def test_flush_and_shutdown_are_bounded_when_the_backend_hangs() -> None:
    release = threading.Event()

    class HangingTracer(RecordingTracer):
        def _flush(self) -> None:
            release.wait(5)

    tracer = HangingTracer()
    started = time.monotonic()
    try:
        assert tracer.flush(timeout_s=0.1) is False
        assert tracer.shutdown(timeout_s=0.1) is False
        assert time.monotonic() - started < 1.0
    finally:
        release.set()


def test_recording_tracer_flush_and_shutdown_succeed() -> None:
    tracer = RecordingTracer()
    assert tracer.flush(timeout_s=1) is True
    assert tracer.shutdown(timeout_s=1) is True
    assert (tracer.flush_calls, tracer.shutdown_calls) == (2, 1)


# ---------------------------------------------------------------- tracing off


def test_noop_tracer_records_nothing_and_never_scrubs(monkeypatch: pytest.MonkeyPatch) -> None:
    def must_not_scrub(*_args: object) -> object:
        raise AssertionError("the no-op tracer scrubbed a value")

    monkeypatch.setattr(Scrubber, "value", must_not_scrub)
    tracer = obs.NoopTracer()
    assert tracer.enabled is False
    with tracer.trace("ask", input={"big": "x" * 10_000}) as root:
        with obs.span("inner", input="ignored") as inner:
            inner.update(output="ignored")
        root.error("ignored")
    assert root.recording is False and inner.recording is False
    assert tracer.flush(timeout_s=0) is True
    assert tracer.shutdown(timeout_s=0) is True


# ---------------------------------------------------------------- scrubbing


def test_scrubber_removes_known_keys_and_secret_shapes(tmp_path: Path) -> None:
    scrub = Scrubber.for_settings(settings_with_keys(tmp_path))
    bearer = "Bearer abc123" + "t" * 20
    data = {
        "note": f"keys {FAKE_GEMINI} {FAKE_LF_PUBLIC} {FAKE_LF_SECRET} {GOOGLE_SHAPED}",
        "headers": {"Authorization": "anything", "x-goog-api-key": "anything"},
        "api_key": "whatever",
        "auth": bearer,
        "nested": [{"deep": (FAKE_GEMINI,)}],
        "empty_key": {"api_key": None},
    }
    cleaned = scrub.value(data)
    text = repr(cleaned)
    for secret in (FAKE_GEMINI, FAKE_LF_PUBLIC, FAKE_LF_SECRET, GOOGLE_SHAPED, "abc123"):
        assert secret not in text
    assert isinstance(cleaned, dict)
    assert cleaned["headers"] == {"Authorization": REDACTED, "x-goog-api-key": REDACTED}
    assert cleaned["api_key"] == REDACTED
    assert cleaned["empty_key"] == {"api_key": None}


def test_scrubber_keeps_ordinary_text_and_makes_values_json_safe() -> None:
    @dataclasses.dataclass
    class Hit:
        doc: str
        score: float
        secret: str = dataclasses.field(default="hidden", repr=False)

    scrub = Scrubber(("short",))  # too short to count as a secret
    prose = "A basic configuration; the bearer of news; tokens: 42; a short note."
    cleaned = scrub.value({"prose": prose, "hit": Hit("doc-a", 0.5), "raw": b"\x00\x01", 7: None})
    assert cleaned == {
        "prose": prose,
        "hit": {"doc": "doc-a", "score": 0.5},
        "raw": "<2 bytes>",
        "7": None,
    }


def test_scrubber_stops_at_max_depth() -> None:
    nested: list[object] = []
    nested.append(nested)  # a cycle
    assert TOO_DEEP in repr(Scrubber(max_depth=3).value(nested))


def test_recording_tracer_for_settings_scrubs_every_field(tmp_path: Path) -> None:
    tracer = RecordingTracer.for_settings(settings_with_keys(tmp_path))
    with tracer.trace("ask", input=f"q {FAKE_GEMINI}", metadata={"k": FAKE_LF_SECRET}) as root:
        with obs.generation(
            "g", model="m", input=FAKE_LF_PUBLIC, model_parameters={"api_key": FAKE_GEMINI}
        ) as gen:
            gen.update(output=GOOGLE_SHAPED)
        root.error(f"failed with {FAKE_GEMINI}")
    text = repr(tracer.observations)
    for secret in (FAKE_GEMINI, FAKE_LF_PUBLIC, FAKE_LF_SECRET, GOOGLE_SHAPED):
        assert secret not in text
    assert isinstance(tracer.observations[0], RecordedObservation)
    assert tracer.observations[0].status_message == f"failed with {REDACTED}"
