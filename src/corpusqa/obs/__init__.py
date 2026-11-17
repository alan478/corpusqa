"""Tracing: the only package that talks to Langfuse. See AGENTS.md (Tracing).

Use the helpers below; never import the Langfuse SDK or OpenTelemetry elsewhere
(lint-imports enforces it). Tests use `corpusqa.obs.testing.RecordingTracer`.
"""

from corpusqa.obs.bootstrap import init_tracing
from corpusqa.obs.scrub import REDACTED, Scrubber
from corpusqa.obs.tracer import (
    NOOP_TRACER,
    Level,
    NoopTracer,
    Observation,
    Tracer,
    bind,
    current,
    current_tracer,
    generation,
    new_trace_id,
    observe,
    span,
)

__all__ = [
    "NOOP_TRACER",
    "REDACTED",
    "Level",
    "NoopTracer",
    "Observation",
    "Scrubber",
    "Tracer",
    "bind",
    "current",
    "current_tracer",
    "generation",
    "init_tracing",
    "new_trace_id",
    "observe",
    "span",
]
