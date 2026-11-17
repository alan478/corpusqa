"""FastAPI app: readiness from a background start-up job, and an /ask that always answers 200.

CONTRACT.md counts any non-200 /ask, malformed body or empty answer as a failed
question, so /ask parses its own body inside one try block instead of using a typed
FastAPI body parameter: FastAPI rejects bad bodies with 422 *or* 400 (for example
invalid UTF-8) before the handler runs, and overriding the 422 handler misses the 400.

The answerer can need the index (the agent's tools read it), so it can be built after
start-up: `create_app` takes either a ready `answerer` or an `answerer_factory`, which the
start-up thread calls once with the `Runtime` before /health says 200. A factory that
raises fails start-up like the start-up job itself (/health stays 503).

At most `ASK_MAX_CONCURRENT` (4, CONTRACT §4) answers run at once (`AskQueue`); a request
that arrives while they all run waits in line for a slot, never refused. Its deadline
(`Settings.ask_deadline_s`) counts from its arrival, so the wait is part of it: the
answerer is told the time left when it starts, and a request whose wait and answer pass
the deadline gets the timeout answer at the deadline, never later.

The body is exactly CONTRACT §3's `answer` + `document_ids` (user decision 2026-10-17).
Every /ask response sends its trace id in the `TRACE_ID_HEADER` header (`X-Trace-Id`)
instead, and the server's one log line per /ask names it too.

Tracing (AGENTS.md: Tracing): each /ask is one trace whose id is the response's
`X-Trace-Id` header, and the answerer runs inside it so its spans nest under it; the trace
also records the queue (answers running and waiting on arrival, the wait). The start-up
job is a `startup` trace, with an `answerer` span for building the answerer. /health is
not traced (it is polled every second).
"""

import asyncio
import logging
import threading
import time
import uuid
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass
from enum import StrEnum

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.telemetry import TelemetryConfig
from pydantic import ValidationError

from corpusqa import obs
from corpusqa.agent import Answerer
from corpusqa.agent.answerer import AnswererUnavailable
from corpusqa.config import (
    ASK_MAX_CONCURRENT,
    ASK_PATH,
    ASK_TRACE_NAME,
    HEALTH_PATH,
    INTERNAL_ERROR_ANSWER,
    INVALID_REQUEST_ANSWER,
    LOG_QUESTION_CHARS,
    NOT_READY_ANSWER,
    STARTUP_TRACE_NAME,
    TIMEOUT_ANSWER,
    TRACE_ID_HEADER,
    TRACE_SHUTDOWN_TIMEOUT_S,
    Settings,
)
from corpusqa.ingest import Runtime, StartupFn
from corpusqa.obs import Level, NoopTracer, Observation, Tracer
from corpusqa.server.models import AskRequest, AskResponse, HealthResponse
from corpusqa.server.respond import build_response

logger = logging.getLogger(__name__)

ANSWERER_SPAN = "answerer"

# Builds the answerer from the start-up result (called once, in the start-up thread).
type AnswererFactory = Callable[[Runtime], Answerer]


class Outcome(StrEnum):
    """How one /ask ended; logged with its trace_id."""

    OK = "ok"
    INVALID_REQUEST = "invalid_request"
    NOT_READY = "not_ready"
    TIMEOUT = "timeout"
    EMPTY_ANSWER = "empty_answer"
    ERROR = "error"


# The ask trace's level per outcome: the service failing is an error; a bad request
# or a question before start-up finished is the caller's problem, so a warning.
OUTCOME_LEVEL: dict[Outcome, Level] = {
    Outcome.OK: "DEFAULT",
    Outcome.INVALID_REQUEST: "WARNING",
    Outcome.NOT_READY: "WARNING",
    Outcome.TIMEOUT: "ERROR",
    Outcome.EMPTY_ANSWER: "ERROR",
    Outcome.ERROR: "ERROR",
}


@dataclass(frozen=True, slots=True)
class Ready:
    """What start-up publishes: the index state and the answerer built over it."""

    runtime: Runtime
    answerer: Answerer


class AskQueue:
    """At most `limit` answers at once; the others wait for a slot in arrival order.

    Nobody is refused: a waiting request leaves the line only when it gets a slot or its
    own deadline passes (the caller's `asyncio.timeout`). Used only from the event loop's
    thread, so the counters need no lock. The semaphore binds to the loop that first waits
    on it: one queue per app, and an app serves from one loop.
    """

    def __init__(self, limit: int) -> None:
        if limit < 1:
            raise ValueError(f"the queue needs at least one slot, got {limit}")
        self.limit = limit
        self.running = 0  # answers holding a slot
        self.waiting = 0  # requests in line for one
        self._slots = asyncio.Semaphore(limit)

    @asynccontextmanager
    async def slot(self, trace: Observation, trace_id: str) -> AsyncIterator[None]:
        """Hold one slot for the block; the wait and the queue's state go on `trace`."""
        arrived = time.monotonic()
        trace.update(
            metadata={
                "queue_limit": self.limit,
                "queue_running_on_arrival": self.running,
                "queue_waiting_on_arrival": self.waiting,
            }
        )
        if self.running >= self.limit:
            logger.info(
                "ask trace_id=%s waits for a slot: %d answers running, %d already waiting",
                trace_id,
                self.running,
                self.waiting,
            )
        self.waiting += 1
        got_slot = False
        try:
            await self._slots.acquire()
            got_slot = True
        finally:  # also when the deadline passes in line
            self.waiting -= 1
            waited_ms = round((time.monotonic() - arrived) * 1000)
            trace.update(metadata={"queue_wait_ms": waited_ms, "queue_got_slot": got_slot})
        self.running += 1
        try:
            yield
        finally:
            self.running -= 1
            self._slots.release()


class Readiness:
    """Start-up result: written once by the start-up thread, then only read by requests.

    Each field is set by a single reference assignment, which is atomic in CPython,
    so readers need no lock.
    """

    def __init__(self) -> None:
        self.ready: Ready | None = None
        self.failed = False

    def run(
        self,
        startup: StartupFn,
        make_answerer: AnswererFactory,
        settings: Settings,
        tracer: Tracer,
    ) -> None:
        started = time.monotonic()
        name = getattr(startup, "__qualname__", repr(startup))
        logger.info("startup: running %s", name)
        ready: Ready | None = None
        # The flags are set after the trace closes, so once /health shows the result
        # the startup trace is complete.
        with tracer.trace(STARTUP_TRACE_NAME, metadata={"startup_fn": name}) as trace:
            try:
                runtime = startup(settings)
                ready = Ready(runtime, _build_answerer(make_answerer, runtime))
            except Exception as exc:
                elapsed = time.monotonic() - started
                trace.update(metadata={"outcome": "failed", "duration_s": round(elapsed, 3)})
                trace.error(exc)
                logger.critical(
                    "startup FAILED after %.1f s; /health stays 503 until restart",
                    elapsed,
                    exc_info=True,
                )
            else:
                trace.update(
                    output={"known_doc_ids": len(runtime.known_doc_ids)},
                    metadata={
                        "outcome": "ready",
                        "duration_s": round(time.monotonic() - started, 3),
                        "answerer": type(ready.answerer).__name__,
                    },
                )
        if ready is None:
            self.failed = True
            return
        self.ready = ready
        logger.info(
            "startup: ready after %.1f s with %d known document ids (answerer %s); "
            "/health is now 200",
            time.monotonic() - started,
            len(ready.runtime.known_doc_ids),
            type(ready.answerer).__name__,
        )


def _build_answerer(make_answerer: AnswererFactory, runtime: Runtime) -> Answerer:
    """The factory's answerer, inside an `answerer` span of the startup trace."""
    started = time.monotonic()
    with obs.span(ANSWERER_SPAN) as span:
        answerer = make_answerer(runtime)
        span.update(
            output={"answerer": type(answerer).__name__},
            metadata={"duration_s": round(time.monotonic() - started, 3)},
        )
    return answerer


def _always(answerer: Answerer) -> AnswererFactory:
    """A factory that gives this one answerer, whatever start-up returned."""

    def make(_runtime: Runtime) -> Answerer:
        return answerer

    return make


def create_app(
    settings: Settings,
    startup: StartupFn,
    answerer: Answerer | None = None,
    tracer: Tracer | None = None,
    *,
    answerer_factory: AnswererFactory | None = None,
) -> FastAPI:
    """Build the app. Each call is independent: no module-level state.

    Give exactly one of `answerer` (ready now) and `answerer_factory` (called once with
    the `Runtime` when start-up returns; `__main__` builds the agent this way).
    `tracer` defaults to tracing off; `__main__` passes `obs.init_tracing(settings)`.
    """
    if answerer_factory is None:
        if answerer is None:
            raise ValueError("create_app needs an answerer or an answerer_factory")
        answerer_factory = _always(answerer)
    elif answerer is not None:
        raise ValueError("create_app takes an answerer or an answerer_factory, not both")
    make_answerer = answerer_factory
    tracer = NoopTracer() if tracer is None else tracer
    readiness = Readiness()
    queue = AskQueue(ASK_MAX_CONCURRENT)

    @asynccontextmanager
    async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
        # The port is bound as soon as this yields; start-up runs beside it. A daemon
        # thread, not the loop's executor: asyncio.run joins executor threads on exit,
        # which would hold a SIGTERM shutdown hostage to a long ingest.
        threading.Thread(
            target=readiness.run,
            args=(startup, make_answerer, settings, tracer),
            name="startup",
            daemon=True,
        ).start()
        try:
            yield
        finally:
            if tracer.enabled:
                # Send queued spans; bounded so it fits in the graceful-shutdown window.
                await asyncio.to_thread(tracer.shutdown, TRACE_SHUTDOWN_TIMEOUT_S)

    # FastAPI's own OpenTelemetry tracing would switch on by itself if any library set
    # a global provider, and would trace every /health poll. Tracing goes through obs.
    no_native_telemetry: TelemetryConfig = {
        "tracing": False,
        "metrics": False,
        "logs": False,
        "auto_configure": False,
    }
    app = FastAPI(title="corpusqa", lifespan=lifespan, telemetry=no_native_telemetry)

    @app.get(HEALTH_PATH, response_model=HealthResponse)
    async def health() -> JSONResponse:
        if readiness.ready is not None:
            return JSONResponse(HealthResponse(status="ok").model_dump())
        status = HealthResponse(status="failed" if readiness.failed else "starting")
        return JSONResponse(status.model_dump(), status_code=503)

    @app.post(ASK_PATH, response_model=AskResponse)
    async def ask(request: Request) -> JSONResponse:
        trace_id = uuid.uuid4().hex
        started = time.perf_counter()
        # This request's deadline counts from its arrival: a wait for a slot is part of it.
        deadline = asyncio.get_running_loop().time() + settings.ask_deadline_s
        # The trace id IS the response's X-Trace-Id header, so any answer opens in Langfuse.
        with tracer.trace(ASK_TRACE_NAME, trace_id=trace_id) as trace:
            outcome, question, body, problem = await _answer(request, trace_id, trace, deadline)
            ms = (time.perf_counter() - started) * 1000
            _record_ask(trace, outcome, body, problem, ms)
        logger.info(
            "ask trace_id=%s outcome=%s ms=%d ids=%d answer_chars=%d question=%r",
            trace_id,
            outcome,
            ms,
            len(body.document_ids),
            len(body.answer),
            None if question is None else question[:LOG_QUESTION_CHARS],
        )
        # Every path ends here: the body is the contract's two fields, the trace id a header.
        return JSONResponse(body.model_dump(), headers={TRACE_ID_HEADER: trace_id})

    async def _answer(
        request: Request, trace_id: str, trace: Observation, deadline: float
    ) -> tuple[Outcome, str | None, AskResponse, str | None]:
        """Every path returns a valid body (+ what went wrong, for the trace).

        Nothing raised here reaches the client.
        """
        question: str | None = None
        try:
            try:
                question = AskRequest.model_validate_json(await request.body()).question
            except ValidationError as exc:
                errors = "; ".join(f"{e['type']} at {e['loc']}" for e in exc.errors())
                logger.warning("ask trace_id=%s invalid request: %s", trace_id, errors)
                body = _fixed(INVALID_REQUEST_ANSWER, trace_id)
                return Outcome.INVALID_REQUEST, None, body, f"invalid request: {errors}"
            trace.update(input=question)
            ready = readiness.ready
            if ready is None:
                body = _fixed(NOT_READY_ANSWER, trace_id)
                return Outcome.NOT_READY, question, body, "start-up has not finished"
            loop = asyncio.get_running_loop()
            async with asyncio.timeout_at(deadline), queue.slot(trace, trace_id):
                time_left_s = deadline - loop.time()
                result = await ready.answerer.answer(question, time_left_s=time_left_s)
            if not result.answer.strip():
                logger.error("ask trace_id=%s answerer returned a blank answer", trace_id)
                body = _fixed(INTERNAL_ERROR_ANSWER, trace_id)
                return Outcome.EMPTY_ANSWER, question, body, "the answerer returned a blank answer"
            body = build_response(
                result.answer, result.document_ids, ready.runtime.known_doc_ids, trace_id
            )
            return Outcome.OK, question, body, None
        except TimeoutError:
            logger.error(
                "ask trace_id=%s timed out (deadline %.1f s)", trace_id, settings.ask_deadline_s
            )
            problem = f"timed out after {settings.ask_deadline_s:.1f} s"
            return Outcome.TIMEOUT, question, _fixed(TIMEOUT_ANSWER, trace_id), problem
        except AnswererUnavailable as exc:  # known at start-up (e.g. no key): no traceback
            logger.error("ask trace_id=%s cannot be answered: %s", trace_id, exc)
            body = _fixed(INTERNAL_ERROR_ANSWER, trace_id)
            return Outcome.ERROR, question, body, f"{type(exc).__name__}: {exc}"
        except Exception as exc:
            logger.exception("ask trace_id=%s failed with an internal error", trace_id)
            body = _fixed(INTERNAL_ERROR_ANSWER, trace_id)
            return Outcome.ERROR, question, body, f"{type(exc).__name__}: {exc}"

    return app


def _record_ask(
    trace: Observation, outcome: Outcome, body: AskResponse, problem: str | None, ms: float
) -> None:
    """Put the response and its outcome on the ask trace (the input is set on parse)."""
    trace.update(
        output={"answer": body.answer, "document_ids": body.document_ids},
        metadata={
            "outcome": outcome.value,
            "latency_ms": round(ms),
            "document_ids": len(body.document_ids),
            "answer_chars": len(body.answer),
        },
    )
    level = OUTCOME_LEVEL[outcome]
    if level == "ERROR":
        trace.error(problem or outcome.value)
    elif level == "WARNING":
        trace.warning(problem or outcome.value)


def _fixed(text: str, trace_id: str) -> AskResponse:
    """A fixed-text answer that cites nothing."""
    return build_response(text, [], frozenset(), trace_id)
