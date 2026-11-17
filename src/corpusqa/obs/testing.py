"""In-memory tracer for tests: same facade, so the same scrubbing and safety rules apply.

tracer = RecordingTracer.for_settings(settings)
app = create_app(settings, startup, answerer, tracer=tracer)
...
[ask] = tracer.traces("ask")
assert ask.trace_id == response.headers[TRACE_ID_HEADER]
assert [s.name for s in tracer.children(ask)] == ["search"]
"""

import threading
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field

from corpusqa.config import Settings
from corpusqa.obs.scrub import JsonValue, Scrubber
from corpusqa.obs.tracer import (
    Handle,
    Kind,
    Level,
    ObservationStart,
    ObservationUpdate,
    Tracer,
    new_trace_id,
)


@dataclass
class RecordedObservation:
    """One trace, span or generation as the backend received it (already scrubbed)."""

    kind: Kind
    name: str
    id: str
    trace_id: str
    parent_id: str | None
    input: JsonValue = None
    output: JsonValue = None
    metadata: dict[str, JsonValue] = field(default_factory=dict)
    level: Level = "DEFAULT"
    status_message: str | None = None
    model: str | None = None
    model_parameters: dict[str, JsonValue] | None = None
    version: str | None = None
    usage: dict[str, int] | None = None
    cost: dict[str, float] | None = None
    ended: bool = False

    def update(self, update: ObservationUpdate) -> None:
        if update.input is not None:
            self.input = update.input
        if update.output is not None:
            self.output = update.output
        if update.metadata:
            self.metadata.update(update.metadata)
        if update.level is not None:
            self.level = update.level
        if update.status_message is not None:
            self.status_message = update.status_message
        if update.usage is not None:
            self.usage = update.usage
        if update.cost is not None:
            self.cost = update.cost


class RecordingTracer(Tracer):
    """Records every observation in memory. Thread-safe."""

    def __init__(self, scrubber: Scrubber | None = None) -> None:
        super().__init__(scrubber)
        self.observations: list[RecordedObservation] = []
        self.flush_calls = 0
        self.shutdown_calls = 0
        self._lock = threading.Lock()

    @classmethod
    def for_settings(cls, settings: Settings) -> "RecordingTracer":
        """A recorder that scrubs this settings' keys, like the real tracer would."""
        return cls(Scrubber.for_settings(settings))

    # ------------------------------------------------------------ queries
    def traces(self, name: str | None = None) -> list[RecordedObservation]:
        return [o for o in self._snapshot() if o.kind == "trace" and name in (None, o.name)]

    def named(self, name: str) -> list[RecordedObservation]:
        return [o for o in self._snapshot() if o.name == name]

    def in_trace(self, trace_id: str) -> list[RecordedObservation]:
        return [o for o in self._snapshot() if o.trace_id == trace_id]

    def children(self, parent: RecordedObservation) -> list[RecordedObservation]:
        return [o for o in self._snapshot() if o.parent_id == parent.id]

    def _snapshot(self) -> list[RecordedObservation]:
        with self._lock:
            return list(self.observations)

    # ------------------------------------------------------------ backend hooks
    @contextmanager
    def _open(self, start: ObservationStart, parent: Handle | None) -> Iterator[Handle]:
        above = parent if isinstance(parent, RecordedObservation) else None
        if start.kind == "trace" and start.trace_id is not None:
            trace_id = start.trace_id
        else:
            trace_id = above.trace_id if above is not None else new_trace_id()
        recorded = RecordedObservation(
            kind=start.kind,
            name=start.name,
            id=uuid.uuid4().hex[:16],
            trace_id=trace_id,
            parent_id=None if start.kind == "trace" or above is None else above.id,
            input=start.input,
            metadata=dict(start.metadata or {}),
            model=start.model,
            model_parameters=start.model_parameters,
            version=start.version,
        )
        with self._lock:
            self.observations.append(recorded)
        try:
            yield recorded
        finally:
            recorded.ended = True

    def _flush(self) -> None:
        with self._lock:
            self.flush_calls += 1

    def _shutdown(self) -> None:
        with self._lock:
            self.shutdown_calls += 1
