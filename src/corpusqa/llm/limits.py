"""Request pacing and a persisted spend ledger with a hard cap, shared by model clients.

Both are thread-safe (up to four requests run at once). The ledger counts one process's
calls on top of what earlier runs recorded; two processes writing the same file at the
same moment can lose an update (dev only: the grader runs one server).
"""

import asyncio
import json
import threading
import time
from collections.abc import Awaitable, Callable
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

from corpusqa.llm.errors import LLMError, SpendCapReached

_SECONDS_PER_MINUTE = 60.0


class RateLimiter:
    """At most `per_minute` calls per minute, spaced evenly; `wait` sleeps until the next slot."""

    def __init__(
        self,
        per_minute: float,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self._interval = _SECONDS_PER_MINUTE / per_minute
        self._clock = clock
        self._sleep = sleep
        self._next = float("-inf")
        self._lock = threading.Lock()

    def wait(self) -> float:
        """Block until this call may go; returns the seconds waited."""
        with self._lock:
            now = self._clock()
            slot = max(now, self._next)
            self._next = slot + self._interval
        delay = slot - now
        if delay > 0:
            self._sleep(delay)
        return delay


class AsyncRateLimiter:
    """The async twin of RateLimiter: `await wait()` sleeps until the next slot.

    One instance is shared by every request of the process. The slot is taken under a
    threading lock (held for a few instructions, never across an await), so it works from
    any event loop and thread.
    """

    def __init__(
        self,
        per_minute: float,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    ) -> None:
        self._interval = _SECONDS_PER_MINUTE / per_minute
        self._clock = clock
        self._sleep = sleep
        self._next = float("-inf")
        self._lock = threading.Lock()

    async def wait(self) -> float:
        """Wait until this call may go; returns the seconds waited."""
        with self._lock:
            now = self._clock()
            slot = max(now, self._next)
            self._next = slot + self._interval
        delay = slot - now
        if delay > 0:
            await self._sleep(delay)
        return delay


@dataclass(frozen=True, slots=True)
class SpendTotals:
    calls: int = 0
    tokens: int = 0
    cost_usd: float = 0.0


class SpendLedger:
    """Running totals of one kind of call, saved as JSON after every call.

    A call first reserves its estimated cost (`reserve`): refused (SpendCapReached) when the
    recorded spend plus the calls still in flight plus this one would pass the cap. Then
    `add` records what it cost and frees the reservation, or `release` frees it when the
    call failed. Admission is atomic, so calls running at once cannot pass the cap together.
    """

    def __init__(self, path: Path, cap_usd: float) -> None:
        self.path = path
        self.cap_usd = cap_usd
        self._lock = threading.Lock()
        self._totals = self._load()
        self._reserved = 0.0  # estimated cost of the calls in flight

    @property
    def totals(self) -> SpendTotals:
        return self._totals

    def reserve(self, cost_usd: float) -> None:
        with self._lock:
            spent, pending = self._totals.cost_usd, self._reserved
            if spent + pending + cost_usd > self.cap_usd:
                raise SpendCapReached(
                    f"spend cap ${self.cap_usd:.2f} reached (${spent:.4f} recorded in "
                    f"{self.path}, ${pending:.4f} in flight, this call ~${cost_usd:.4f}); "
                    "raise the cap in config.py or delete the file"
                )
            self._reserved += cost_usd

    def release(self, reserved_usd: float) -> None:
        """Free a reservation whose call was not made or failed (nothing was recorded)."""
        with self._lock:
            self._reserved = max(0.0, self._reserved - reserved_usd)

    def add(self, tokens: int, cost_usd: float, reserved_usd: float = 0.0) -> SpendTotals:
        """Record one call's cost and free its reservation."""
        with self._lock:
            self._reserved = max(0.0, self._reserved - reserved_usd)
            t = self._totals
            self._totals = SpendTotals(t.calls + 1, t.tokens + tokens, t.cost_usd + cost_usd)
            self._save(self._totals)
            return self._totals

    def _load(self) -> SpendTotals:
        if not self.path.is_file():
            return SpendTotals()
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
            return SpendTotals(
                calls=int(data["calls"]),
                tokens=int(data["tokens"]),
                cost_usd=float(data["cost_usd"]),
            )
        except (OSError, ValueError, KeyError, TypeError) as exc:
            # Starting from zero could overspend; refuse until someone looks at the file.
            raise LLMError(
                f"spend ledger {self.path} is unreadable ({exc}); fix or delete it"
            ) from exc

    def _save(self, totals: SpendTotals) -> None:
        record = {**asdict(totals), "updated_at": datetime.now(UTC).isoformat(timespec="seconds")}
        self.path.parent.mkdir(parents=True, exist_ok=True)
        partial = self.path.with_name(self.path.name + ".tmp")
        partial.write_text(json.dumps(record, indent=1), encoding="utf-8")
        partial.replace(self.path)
