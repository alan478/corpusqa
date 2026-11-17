"""The /ask queue: at most `limit` answers at once, the others wait in line, never refused.

A request whose deadline passes while it waits leaves the line, and its trace says so.
"""

import asyncio

import pytest

from corpusqa.obs.testing import RecordingTracer
from corpusqa.server.app import AskQueue

TRACE = "0" * 32


def test_a_queue_needs_a_slot() -> None:
    with pytest.raises(ValueError, match="at least one slot"):
        AskQueue(0)


def test_requests_wait_in_arrival_order_and_hold_at_most_limit_slots() -> None:
    tracer = RecordingTracer()
    queue = AskQueue(2)
    order: list[str] = []
    most = 0

    async def one(name: str) -> None:
        nonlocal most
        with tracer.trace("ask") as trace:
            async with queue.slot(trace, TRACE):
                most = max(most, queue.running)
                order.append(name)
                await asyncio.sleep(0.02)

    async def scenario() -> None:
        await asyncio.gather(*(one(f"r{i}") for i in range(5)))

    asyncio.run(scenario())
    assert most == 2
    assert order == [f"r{i}" for i in range(5)]  # first come, first served
    assert (queue.running, queue.waiting) == (0, 0)
    waiting_seen = [t.metadata["queue_waiting_on_arrival"] for t in tracer.traces("ask")]
    assert waiting_seen == [0, 0, 0, 1, 2]


def test_a_deadline_that_passes_in_line_leaves_the_line_and_is_traced() -> None:
    tracer = RecordingTracer()
    queue = AskQueue(1)

    async def scenario() -> None:
        release = asyncio.Event()

        async def holder() -> None:
            with tracer.trace("ask") as trace:
                async with queue.slot(trace, TRACE):
                    await release.wait()

        task = asyncio.create_task(holder())
        await asyncio.sleep(0)  # the holder takes the one slot
        with tracer.trace("ask") as trace, pytest.raises(TimeoutError):
            async with asyncio.timeout(0.05), queue.slot(trace, TRACE):
                pytest.fail("the slot was never free")
        assert (queue.running, queue.waiting) == (1, 0)  # it left the line
        release.set()
        await task

    asyncio.run(scenario())
    holder_trace, waiter = tracer.traces("ask")
    assert holder_trace.metadata["queue_got_slot"] is True
    assert waiter.metadata["queue_got_slot"] is False
    assert waiter.metadata["queue_running_on_arrival"] == 1
    assert waiter.metadata["queue_wait_ms"] >= 40  # type: ignore[operator]
    assert (queue.running, queue.waiting) == (0, 0)
