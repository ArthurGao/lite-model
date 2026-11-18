"""Design §10: Bounded / AsyncBounded cap the calls in flight; map() keeps input order."""

import asyncio
import threading
import time
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor

import pytest
from pydantic import BaseModel

from lite_model import AsyncBounded, Bounded, EmptyReply, Message, StructuredReply, TextReply
from lite_model.testing import FakeCall, FakeModel


class Gauge:
    """Counts calls in flight; each call holds its slot for a moment."""

    def __init__(self) -> None:
        self.now = self.peak = 0
        self._lock = threading.Lock()

    def enter(self) -> None:
        with self._lock:
            self.now += 1
            self.peak = max(self.peak, self.now)

    def leave(self) -> None:
        with self._lock:
            self.now -= 1

    def answer(self, call: FakeCall) -> str:
        """A FakeModel script: holds a slot for 20 ms, then echoes the message upper-cased."""
        self.enter()
        time.sleep(0.02)
        self.leave()
        return call.messages[0].content.upper() if call.messages else "ok"


def test_bounded_map_keeps_order() -> None:
    gauge = Gauge()
    bounded = Bounded(FakeModel(answer=gauge.answer), concurrency=2)
    words = [f"w{i}" for i in range(8)]
    out = bounded.map(lambda w: bounded.text("answer", [Message("user", w)]).text, words)
    assert out == [w.upper() for w in words]
    assert gauge.peak == 2


@pytest.mark.parametrize("slots", [1, 2])
def test_bounded_caps_callers_from_other_threads(slots: int) -> None:
    # Several jobs share one model (a server ingesting documents in parallel): eight threads.
    # map()'s own pool cannot show this (it has `concurrency` threads); the semaphore must.
    gauge = Gauge()
    bounded = Bounded(FakeModel(answer=gauge.answer), concurrency=slots)
    with ThreadPoolExecutor(8) as pool:
        list(pool.map(lambda _: bounded.text("answer", []), range(8)))
    assert gauge.peak == slots


def test_bounded_map_raises_the_first_error() -> None:
    bounded = Bounded(FakeModel(answer=["a", EmptyReply("empty"), "c"]), concurrency=1)
    with pytest.raises(EmptyReply):
        bounded.map(lambda _: bounded.text("answer", []).text, range(3))


def test_concurrency_must_be_positive() -> None:
    with pytest.raises(ValueError):
        Bounded(FakeModel(), concurrency=0)
    with pytest.raises(ValueError):
        AsyncBounded(_SlowAsync(Gauge()), concurrency=0)


class _SlowAsync:
    """An AsyncModel whose calls take a moment, so concurrency is observable."""

    def __init__(self, gauge: Gauge) -> None:
        self.gauge = gauge

    async def text(self, role: str, messages: Sequence[Message]) -> TextReply:
        self.gauge.enter()
        await asyncio.sleep(0.01)
        self.gauge.leave()
        if messages[0].content == "boom":
            raise EmptyReply("empty")
        return FakeModel(answer=messages[0].content.upper()).text(role, messages)

    async def structured[T: BaseModel](
        self, role: str, messages: Sequence[Message], schema: type[T]
    ) -> StructuredReply[T]:
        raise NotImplementedError


def test_async_bounded_caps_calls_in_flight_and_keeps_order() -> None:
    gauge = Gauge()
    bounded = AsyncBounded(_SlowAsync(gauge), concurrency=3)
    words = [f"w{i}" for i in range(9)]

    async def one(word: str) -> str:
        return (await bounded.text("answer", [Message("user", word)])).text

    assert asyncio.run(bounded.map(one, words)) == [w.upper() for w in words]
    assert gauge.peak == 3


def test_async_bounded_map_raises_the_first_error() -> None:
    bounded = AsyncBounded(_SlowAsync(Gauge()), concurrency=2)

    async def one(word: str) -> str:
        return (await bounded.text("answer", [Message("user", word)])).text

    with pytest.raises(ExceptionGroup) as caught:
        asyncio.run(bounded.map(one, ["a", "boom", "c"]))
    assert caught.group_contains(EmptyReply)
