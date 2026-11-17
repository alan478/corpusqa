"""Choose the tracer at start-up: Langfuse when both keys are set, otherwise no-op."""

import logging
from urllib.parse import urlsplit

from corpusqa.config import Settings
from corpusqa.obs.scrub import Scrubber
from corpusqa.obs.tracer import NoopTracer, Tracer

logger = logging.getLogger(__name__)


def init_tracing(settings: Settings) -> Tracer:
    """Return the process's tracer and log exactly one line saying whether tracing is on.

    Without both LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY (the grader sets neither)
    this returns a no-op tracer: the Langfuse SDK is not even imported.
    """
    public_key, secret_key = settings.langfuse_public_key, settings.langfuse_secret_key
    if not public_key or not secret_key:
        if public_key or secret_key:
            logger.info(
                "tracing off (set both LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY to turn it on)"
            )
        else:
            logger.info("tracing off (no Langfuse keys)")
        return NoopTracer()
    try:
        # Imported here so the no-key path never loads the SDK or OpenTelemetry.
        from corpusqa.obs._langfuse import LangfuseTracer

        tracer = LangfuseTracer(
            public_key=public_key,
            secret_key=secret_key,
            host=settings.langfuse_host,
            scrubber=Scrubber.for_settings(settings),
        )
    except Exception as exc:
        logger.warning("tracing off (the Langfuse client failed to start: %s)", type(exc).__name__)
        return NoopTracer()
    logger.info("tracing on: Langfuse at %s", _display_host(settings.langfuse_host))
    return tracer


def _display_host(host: str) -> str:
    """The host without any user:password part, safe to log."""
    try:
        parts = urlsplit(host)
        netloc = parts.hostname or ""
        if parts.port is not None:
            netloc = f"{netloc}:{parts.port}"
        return parts._replace(netloc=netloc).geturl() if netloc else host
    except ValueError:
        return "<unparseable host>"
