"""The tracing facade the rest of the code uses. Backends: `_langfuse.py`, `testing.py`.

In code that runs during a request or start-up:

    from corpusqa import obs

    with obs.span("search", input={"query": query}) as span:
        hits = ...
        span.update(output=hits, metadata={"count": len(hits)})

    with obs.generation("answer", model=MODEL, input=messages,
                        prompt_name="answer", prompt_version="3") as gen:
        reply = ...
        gen.update(output=reply.text)
        gen.record_usage(input_tokens=..., output_tokens=..., cost_usd=...)

    @obs.observe()  # sync or async: input = the bound arguments, output = the result
    def lookup(issue_key: str) -> Issue: ...

A new observation attaches to the one that is current in this context (contextvars),
so each /ask's spans land in its own trace, also inside asyncio tasks and
`asyncio.to_thread`. A plain thread starts with an empty context: wrap its target
with `obs.bind(fn)`.

Safety rules, enforced here so backends stay simple:
- every backend call is guarded: a tracing failure is logged once per kind and
  swallowed, and the traced code always runs;
- the traced code's own exceptions always propagate; on the way out they mark the
  observation as ERROR;
- every input, output, metadata value and message goes through the scrubber first.
"""

import asyncio
import contextlib
import contextvars
import functools
import inspect
import logging
import re
import threading
import uuid
from collections.abc import Awaitable, Callable, Iterator, Mapping
from contextlib import AbstractContextManager, contextmanager
from dataclasses import dataclass
from typing import Literal, Protocol, cast

from corpusqa.config import TRACE_STATUS_MAX_CHARS
from corpusqa.obs.scrub import JsonValue, Scrubber

logger = logging.getLogger(__name__)

type Level = Literal["DEBUG", "DEFAULT", "WARNING", "ERROR"]
type Kind = Literal["trace", "span", "generation"]

_TRACE_ID = re.compile(r"[0-9a-f]{32}")
_SKIPPED_ARGUMENTS = frozenset({"self", "cls"})


def new_trace_id() -> str:
    """A fresh trace id: 32 lowercase hex chars, valid for Langfuse and OpenTelemetry."""
    return uuid.uuid4().hex


def is_trace_id(value: str) -> bool:
    return bool(_TRACE_ID.fullmatch(value)) and value != "0" * 32


@dataclass(frozen=True, slots=True)
class ObservationStart:
    """What a backend needs to open one observation (already scrubbed)."""

    kind: Kind
    name: str
    trace_id: str | None = None  # kind "trace" only; always valid when set
    input: JsonValue = None
    metadata: dict[str, JsonValue] | None = None
    model: str | None = None
    model_parameters: dict[str, JsonValue] | None = None
    version: str | None = None


@dataclass(frozen=True, slots=True)
class ObservationUpdate:
    """Changes to an open observation (already scrubbed). None means "leave as is"."""

    input: JsonValue = None
    output: JsonValue = None
    metadata: dict[str, JsonValue] | None = None
    level: Level | None = None
    status_message: str | None = None
    usage: dict[str, int] | None = None
    cost: dict[str, float] | None = None


class Handle(Protocol):
    """A backend's open observation."""

    @property
    def trace_id(self) -> str | None: ...

    @property
    def id(self) -> str | None: ...

    def update(self, update: ObservationUpdate) -> None: ...


class Observation:
    """One open trace, span or generation. No method ever raises."""

    __slots__ = ("_handle", "_tracer")

    def __init__(self, tracer: "Tracer | None", handle: Handle | None) -> None:
        self._tracer = tracer
        self._handle = handle

    @property
    def recording(self) -> bool:
        return self._handle is not None

    @property
    def trace_id(self) -> str | None:
        return self._read("trace_id")

    @property
    def id(self) -> str | None:
        return self._read("id")

    def update(
        self,
        *,
        input: object = None,
        output: object = None,
        metadata: Mapping[str, object] | None = None,
    ) -> None:
        """Set input/output (replaced) and metadata (merged by key). None leaves a field."""
        self._send(input=input, output=output, metadata=metadata)

    def error(self, problem: str | BaseException) -> None:
        """Mark this observation as failed, with a message (or an exception's text)."""
        self._send(level="ERROR", status_message=describe(problem))

    def warning(self, problem: str | BaseException) -> None:
        """Mark this observation as degraded but not failed."""
        self._send(level="WARNING", status_message=describe(problem))

    def record_usage(
        self,
        *,
        input_tokens: int,
        output_tokens: int,
        total_tokens: int | None = None,
        cost_usd: float | None = None,
        details: Mapping[str, int] | None = None,
    ) -> None:
        """Tokens (and cost) of a generation; `details` adds other token kinds."""
        total = input_tokens + output_tokens if total_tokens is None else total_tokens
        usage = {"input": input_tokens, "output": output_tokens, "total": total, **(details or {})}
        cost = None if cost_usd is None else {"total": cost_usd}
        if self._tracer is not None and self._handle is not None:
            self._tracer._send_update(self._handle, ObservationUpdate(usage=usage, cost=cost))

    def _send(
        self,
        *,
        input: object = None,
        output: object = None,
        metadata: Mapping[str, object] | None = None,
        level: Level | None = None,
        status_message: str | None = None,
    ) -> None:
        if self._tracer is None or self._handle is None:
            return
        self._tracer._update(
            self._handle,
            input=input,
            output=output,
            metadata=metadata,
            level=level,
            status_message=status_message,
        )

    def _read(self, attribute: str) -> str | None:
        try:
            return None if self._handle is None else getattr(self._handle, attribute)
        except Exception:
            return None


NOOP_OBSERVATION = Observation(None, None)

# The open observation in this context; new observations become its children.
_CURRENT: contextvars.ContextVar[Observation | None] = contextvars.ContextVar(
    "corpusqa_obs_current", default=None
)


class Tracer:
    """Facade over a tracing backend. Subclasses implement `_open`, `_flush`, `_shutdown`."""

    enabled = True

    def __init__(self, scrubber: Scrubber | None = None) -> None:
        self._scrubber = scrubber or Scrubber()
        self._reported: set[tuple[str, str]] = set()
        self._reported_lock = threading.Lock()

    # ------------------------------------------------------------ public API
    def trace(
        self,
        name: str,
        *,
        trace_id: str | None = None,
        input: object = None,
        metadata: Mapping[str, object] | None = None,
    ) -> AbstractContextManager[Observation]:
        """Open a new trace (a root observation) and make it current.

        `trace_id` is used as the trace's id (32 lowercase hex; a fresh one if missing).
        """
        if self.enabled and (trace_id is None or not is_trace_id(trace_id)):
            if trace_id is not None:
                self._report("trace id", ValueError(f"not 32 lowercase hex: {trace_id[:40]!r}"))
            trace_id = new_trace_id()
        return self._observe("trace", name, trace_id=trace_id, input=input, metadata=metadata)

    def span(
        self,
        name: str,
        *,
        input: object = None,
        metadata: Mapping[str, object] | None = None,
    ) -> AbstractContextManager[Observation]:
        """Open a span under the current observation (a new trace if there is none)."""
        return self._observe("span", name, input=input, metadata=metadata)

    def generation(
        self,
        name: str,
        *,
        model: str,
        input: object = None,
        model_parameters: Mapping[str, object] | None = None,
        prompt_name: str | None = None,
        prompt_version: str | None = None,
        metadata: Mapping[str, object] | None = None,
    ) -> AbstractContextManager[Observation]:
        """Open an LLM or embedding call under the current observation.

        The prompt name and version go into metadata (`prompt_name`, `prompt_version`)
        and the version also into the observation's version field.
        """
        prompt = {"prompt_name": prompt_name, "prompt_version": prompt_version}
        merged = {**{k: v for k, v in prompt.items() if v is not None}, **(metadata or {})}
        return self._observe(
            "generation",
            name,
            input=input,
            metadata=merged,
            model=model,
            model_parameters=model_parameters,
            version=prompt_version,
        )

    def flush(self, timeout_s: float) -> bool:
        """Send queued spans; wait at most `timeout_s`. True if it finished in time."""
        return self._bounded("flush", self._flush, timeout_s)

    def shutdown(self, timeout_s: float) -> bool:
        """Flush, then stop the backend; wait at most `timeout_s` in total."""

        def flush_then_stop() -> None:
            self._flush()
            self._shutdown()

        return self._bounded("shutdown", flush_then_stop, timeout_s)

    # ------------------------------------------------------------ backend hooks
    def _open(
        self, start: ObservationStart, parent: Handle | None
    ) -> AbstractContextManager[Handle]:
        """Open one observation; entering it makes it current in the backend too."""
        raise NotImplementedError

    def _flush(self) -> None:
        """Send queued spans (may block; always called in a bounded worker thread)."""

    def _shutdown(self) -> None:
        """Release the backend (may block; always called in a bounded worker thread)."""

    # ------------------------------------------------------------ internals
    @contextmanager
    def _observe(
        self,
        kind: Kind,
        name: str,
        *,
        trace_id: str | None = None,
        input: object = None,
        metadata: Mapping[str, object] | None = None,
        model: str | None = None,
        model_parameters: Mapping[str, object] | None = None,
        version: str | None = None,
    ) -> Iterator[Observation]:
        if not self.enabled:
            yield NOOP_OBSERVATION
            return
        backend: AbstractContextManager[Handle] | None = None
        handle: Handle | None = None
        try:
            start = ObservationStart(
                kind=kind,
                name=name,
                trace_id=trace_id,
                input=self._scrubber.value(input),
                metadata=self._metadata(metadata),
                model=model,
                model_parameters=self._metadata(model_parameters),
                version=version,
            )
            backend = self._open(start, self._parent(kind))
            handle = backend.__enter__()
        except Exception as exc:
            self._report("open", exc)
            backend = handle = None
        if handle is None:
            yield NOOP_OBSERVATION
            return
        observation = Observation(self, handle)
        token = _CURRENT.set(observation)
        try:
            yield observation
        except BaseException as exc:
            observation.error("cancelled" if isinstance(exc, asyncio.CancelledError) else exc)
            raise
        finally:
            with contextlib.suppress(ValueError):  # closed in another context: leave it
                _CURRENT.reset(token)
            try:
                if backend is not None:
                    backend.__exit__(None, None, None)
            except Exception as exc:
                self._report("close", exc)

    def _parent(self, kind: Kind) -> Handle | None:
        current = _CURRENT.get()
        if kind == "trace" or current is None or current._tracer is not self:
            return None
        return current._handle

    def _metadata(self, metadata: Mapping[str, object] | None) -> dict[str, JsonValue] | None:
        if metadata is None:
            return None
        scrubbed = self._scrubber.value(dict(metadata))
        return scrubbed if isinstance(scrubbed, dict) else None

    def _update(
        self,
        handle: Handle,
        *,
        input: object,
        output: object,
        metadata: Mapping[str, object] | None,
        level: Level | None,
        status_message: str | None,
    ) -> None:
        try:
            update = ObservationUpdate(
                input=self._scrubber.value(input),
                output=self._scrubber.value(output),
                metadata=self._metadata(metadata),
                level=level,
                status_message=None
                if status_message is None
                else self._scrubber.text(status_message)[:TRACE_STATUS_MAX_CHARS],
            )
        except Exception as exc:
            self._report("update", exc)
            return
        self._send_update(handle, update)

    def _send_update(self, handle: Handle, update: ObservationUpdate) -> None:
        try:
            handle.update(update)
        except Exception as exc:
            self._report("update", exc)

    def _bounded(self, op: str, work: Callable[[], None], timeout_s: float) -> bool:
        """Run `work` in a daemon thread; never wait longer than `timeout_s` for it."""
        finished = threading.Event()
        succeeded: list[bool] = []

        def run() -> None:
            try:
                work()
                succeeded.append(True)
            except Exception as exc:
                self._report(op, exc)
            finally:
                finished.set()

        try:
            threading.Thread(target=run, name=f"obs-{op}", daemon=True).start()
        except Exception as exc:
            self._report(op, exc)
            return False
        if not finished.wait(timeout_s):
            logger.warning("tracing: %s did not finish within %.1f s; moving on", op, timeout_s)
            return False
        return bool(succeeded)

    def _report(self, op: str, exc: BaseException) -> None:
        """Log a tracing failure: the first of each kind as a warning, the rest at DEBUG."""
        try:
            kind = (op, type(exc).__name__)
            with self._reported_lock:
                first = kind not in self._reported
                self._reported.add(kind)
            message = self._scrubber.text(str(exc))[:TRACE_STATUS_MAX_CHARS]
            if first:
                logger.warning(
                    "tracing: %s failed and was skipped (%s: %s); "
                    "repeats of this failure are logged at DEBUG",
                    op,
                    type(exc).__name__,
                    message,
                )
            else:
                logger.debug("tracing: %s failed again (%s: %s)", op, type(exc).__name__, message)
        except Exception:  # noqa: S110 - reporting must never raise into traced code
            pass


class NoopTracer(Tracer):
    """Tracing off: every call is a cheap no-op and nothing touches the network."""

    enabled = False

    def flush(self, timeout_s: float) -> bool:
        return True

    def shutdown(self, timeout_s: float) -> bool:
        return True


NOOP_TRACER = NoopTracer()


def describe(problem: str | BaseException) -> str:
    """The text an error status carries: the message, or `Type: message` for exceptions."""
    if isinstance(problem, BaseException):
        text = str(problem)
        return f"{type(problem).__name__}: {text}" if text else type(problem).__name__
    return problem


# ---------------------------------------------------------------- module-level helpers
def current() -> Observation:
    """The open observation in this context (a no-op one if there is none)."""
    return _CURRENT.get() or NOOP_OBSERVATION


def current_tracer() -> Tracer:
    """The tracer of the open observation in this context (the no-op tracer if none)."""
    observation = _CURRENT.get()
    tracer = None if observation is None else observation._tracer
    return tracer or NOOP_TRACER


def span(
    name: str,
    *,
    input: object = None,
    metadata: Mapping[str, object] | None = None,
) -> AbstractContextManager[Observation]:
    """A span under the current observation, on the current tracer (no-op outside a trace)."""
    return current_tracer().span(name, input=input, metadata=metadata)


def generation(
    name: str,
    *,
    model: str,
    input: object = None,
    model_parameters: Mapping[str, object] | None = None,
    prompt_name: str | None = None,
    prompt_version: str | None = None,
    metadata: Mapping[str, object] | None = None,
) -> AbstractContextManager[Observation]:
    """An LLM/embedding call under the current observation (see `Tracer.generation`)."""
    return current_tracer().generation(
        name,
        model=model,
        input=input,
        model_parameters=model_parameters,
        prompt_name=prompt_name,
        prompt_version=prompt_version,
        metadata=metadata,
    )


def bind[**P, R](fn: Callable[P, R]) -> Callable[P, R]:
    """Carry this context (current observation) into a plain thread or executor.

    `asyncio.to_thread` and new asyncio tasks copy the context already; a
    `threading.Thread` target or `executor.submit` callable needs this.
    """
    context = contextvars.copy_context()

    @functools.wraps(fn)
    def in_context(*args: P.args, **kwargs: P.kwargs) -> R:
        # A Context can be entered by one thread at a time; each call gets its own copy.
        return context.copy().run(fn, *args, **kwargs)

    return in_context


def observe[**P, R](
    name: str | None = None, *, capture_input: bool = True, capture_output: bool = True
) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """Decorator: run the function (sync or async) inside a span named after it.

    Input is the bound arguments (without self/cls), output the return value; turn
    either off for large or uninteresting values. Not for generator functions.
    """

    def decorate(fn: Callable[P, R]) -> Callable[P, R]:
        span_name = name or getattr(fn, "__qualname__", None) or repr(fn)
        try:
            signature: inspect.Signature | None = inspect.signature(fn)
        except (TypeError, ValueError):
            signature = None

        def arguments(args: tuple[object, ...], kwargs: dict[str, object]) -> object:
            if not capture_input:
                return None
            try:
                if signature is None:
                    return {"args": list(args), "kwargs": kwargs}
                bound = signature.bind_partial(*args, **kwargs).arguments
                return {k: v for k, v in bound.items() if k not in _SKIPPED_ARGUMENTS}
            except Exception:
                return None

        if inspect.iscoroutinefunction(fn):
            async_fn = cast(Callable[..., Awaitable[object]], fn)

            @functools.wraps(fn)
            async def async_wrapper(*args: P.args, **kwargs: P.kwargs) -> object:
                tracer = current_tracer()
                if not tracer.enabled:
                    return await async_fn(*args, **kwargs)
                with tracer.span(span_name, input=arguments(args, kwargs)) as observation:
                    result = await async_fn(*args, **kwargs)
                    if capture_output:
                        observation.update(output=result)
                    return result

            return cast(Callable[P, R], async_wrapper)

        @functools.wraps(fn)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            tracer = current_tracer()
            if not tracer.enabled:
                return fn(*args, **kwargs)
            with tracer.span(span_name, input=arguments(args, kwargs)) as observation:
                result = fn(*args, **kwargs)
                if capture_output:
                    observation.update(output=result)
                return result

        return wrapper

    return decorate


__all__ = [
    "NOOP_OBSERVATION",
    "NOOP_TRACER",
    "Handle",
    "Kind",
    "Level",
    "NoopTracer",
    "Observation",
    "ObservationStart",
    "ObservationUpdate",
    "Tracer",
    "bind",
    "current",
    "current_tracer",
    "describe",
    "generation",
    "is_trace_id",
    "new_trace_id",
    "observe",
    "span",
]
