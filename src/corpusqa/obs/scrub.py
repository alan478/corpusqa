"""Take secrets out of anything before it goes into a trace.

Two layers:
- exact known values: the keys in `Settings` (Gemini, Langfuse);
- shapes: Google API keys, Langfuse and OpenAI-style keys, bearer and basic auth
  values, private key blocks. A mapping key that names a secret (`api_key`,
  `authorization`, ...) has its whole value replaced.

The result is plain JSON data (dict, list, str, int, float, bool, None), so every
backend can store or serialise it. Dataclasses become dicts without their
`repr=False` fields (this project marks secrets that way), Pydantic models are
dumped, and anything else becomes its repr.
"""

import dataclasses
import re
from collections.abc import Iterable, Mapping

from corpusqa.config import TRACE_SCRUB_MAX_DEPTH, Settings

REDACTED = "[REDACTED]"
TOO_DEEP = "[too deep]"
# Shorter configured "secrets" would redact ordinary words, so they are ignored.
MIN_SECRET_CHARS = 8

_SECRET_PATTERNS = (
    re.compile(r"AIza[0-9A-Za-z_\-]{35}"),  # Google API key
    re.compile(r"\b(?:sk|pk)-lf-[0-9A-Za-z_\-]{8,}"),  # Langfuse keys
    re.compile(r"\bsk-[A-Za-z0-9_\-]{20,}"),  # OpenAI / Anthropic style keys
    # Token values after "Bearer" / "Basic": long and with a digit, so prose survives.
    re.compile(r"(?i)\bbearer\s+(?=[A-Za-z0-9._~+/\-]*\d)[A-Za-z0-9._~+/\-]{16,}=*"),
    re.compile(r"(?i)\bbasic\s+(?=[A-Za-z0-9+/]*\d)[A-Za-z0-9+/]{16,}={0,2}"),
    re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----[\s\S]*?(?:-----END [A-Z ]*PRIVATE KEY-----|$)"),
)
_SECRET_KEY_NAME = re.compile(
    r"(?i)^(?:x-goog-api-key|x-api-key|authorization|proxy-authorization|cookie|set-cookie"
    r"|[a-z0-9_\-]*api[_\-]?key|[a-z0-9_\-]*secret[_\-]?key|[a-z0-9_\-]*_secret|secret"
    r"|password|passwd|access[_\-]?token|refresh[_\-]?token|auth[_\-]?token|private[_\-]?key)$"
)

type JsonValue = bool | int | float | str | list[JsonValue] | dict[str, JsonValue] | None


class Scrubber:
    """Replaces secrets with REDACTED. Stateless after construction, so thread-safe."""

    def __init__(
        self, secrets: Iterable[str | None] = (), max_depth: int = TRACE_SCRUB_MAX_DEPTH
    ) -> None:
        known = {s for s in secrets if s and len(s) >= MIN_SECRET_CHARS}
        # Longest first, so a key that contains another key is replaced whole.
        self._secrets = tuple(sorted(known, key=len, reverse=True))
        self._max_depth = max_depth

    @classmethod
    def for_settings(cls, settings: Settings) -> "Scrubber":
        """A scrubber that also knows this process's configured keys."""
        return cls(
            (settings.gemini_api_key, settings.langfuse_public_key, settings.langfuse_secret_key)
        )

    def text(self, value: str) -> str:
        for secret in self._secrets:
            if secret in value:
                value = value.replace(secret, REDACTED)
        for pattern in _SECRET_PATTERNS:
            value = pattern.sub(REDACTED, value)
        return value

    def value(self, data: object) -> JsonValue:
        """A JSON-safe copy of `data` with every secret replaced."""
        return self._walk(data, 0)

    def _walk(self, data: object, depth: int) -> JsonValue:
        if data is None or isinstance(data, bool | int | float):
            return data
        if isinstance(data, str):
            return self.text(data)
        if depth >= self._max_depth:
            return TOO_DEEP
        if isinstance(data, Mapping):
            return self._mapping(data.items(), depth)
        if isinstance(data, list | tuple | set | frozenset):
            return [self._walk(item, depth + 1) for item in data]
        if isinstance(data, bytes | bytearray):
            return f"<{len(data)} bytes>"
        if dataclasses.is_dataclass(data) and not isinstance(data, type):
            fields = dataclasses.fields(data)
            return self._mapping(((f.name, getattr(data, f.name)) for f in fields if f.repr), depth)
        dump = getattr(data, "model_dump", None)  # Pydantic models, without importing pydantic
        if callable(dump):
            try:
                return self._walk(dump(mode="json"), depth + 1)
            except Exception:  # noqa: S110 - a failed dump falls back to repr below
                pass
        return self.text(repr(data))

    def _mapping(self, items: Iterable[tuple[object, object]], depth: int) -> JsonValue:
        out: dict[str, JsonValue] = {}
        for key, item in items:
            name = self.text(str(key))
            empty = item is None or (isinstance(item, str) and not item)
            if _SECRET_KEY_NAME.match(name) and not empty:
                out[name] = REDACTED
            else:
                out[name] = self._walk(item, depth + 1)
        return out
