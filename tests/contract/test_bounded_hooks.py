"""0.3.0 (design §10, §15): the Bounded slot wraps only the request, so a hook hit never
queues and a hook may call the model through the same Bounded; Bounded.map carries
call_context into its threads; AsyncHook on an async model."""

import asyncio
import threading
import time
from collections.abc import Callable
from typing import Any

import httpx2
import pytest

from lite_model import (
    AsyncBounded,
    Bounded,
    Message,
    Reply,
    TextReply,
    Usage,
    build,
    build_async,
    call_context,
)
from lite_model.hooks import Call
from lite_model.types import Provenance
from tests.configs import local
from tests.contract.conftest import ROLES
from tests.contract.test_usage_and_hooks import Recorder
from tests.upstream import Upstream, completion

CACHED = "cached question"
WAIT_S = 3.0  # far above any real duration: hitting it means a call queued or deadlocked


class Gate:
    """An upstream that holds every request for a message containing "slow" until released."""

    def __init__(self) -> None:
        self.entered = threading.Event()
        self.release = threading.Event()

    def handle(self, request: httpx2.Request) -> httpx2.Response:
        if b"slow" in request.content:
            self.entered.set()
            self.release.wait(WAIT_S)
        return httpx2.Response(200, json=completion())


# The stored reply's provenance: whoever answered first (built once, not per hit).
FROM_CACHE = TextReply(
    "from cache",
    Provenance("answer", "ollama", "qwen3:8b", "qwen3:8b", "local", routed=False),
    Usage(),
)


class Answers(Recorder):
    """Answers CACHED from the cache; records every call (and so its context)."""

    def __init__(self) -> None:
        super().__init__("cache")

    def before(self, call: Call) -> Reply | None:
        super().before(call)
        return FROM_CACHE if call.messages[-1].content == CACHED else None


def _within(seconds: float, fn: Callable[[], Any]) -> Any:
    """`fn()` in a thread; fails if it does not finish in time (a queue or a deadlock)."""
    out: list[Any] = []
    thread = threading.Thread(target=lambda: out.append(fn()), daemon=True)
    thread.start()
    thread.join(seconds)
    assert not thread.is_alive(), f"did not finish within {seconds}s"
    return out[0]


def test_a_hit_does_not_wait_for_the_slot() -> None:
    gate = Gate()
    model = build(
        local(max_retries=0),
        ROLES,
        http_client=httpx2.Client(transport=httpx2.MockTransport(gate.handle)),
    )
    model.hooks.register(Answers())
    bounded = Bounded(model, 1)
    slow = threading.Thread(target=lambda: bounded.text("answer", [Message("user", "slow")]))
    slow.start()
    assert gate.entered.wait(WAIT_S)
    try:
        hit = _within(1.0, lambda: bounded.text("answer", [Message("user", CACHED)]))
        assert hit.cached and hit.text == "from cache"
    finally:
        gate.release.set()
        slow.join(WAIT_S)


def test_a_hook_may_call_the_model_through_the_same_bounded() -> None:
    # The L2 verifier's shape: a hook that asks the model (another role) before answering.
    model = build(local(max_retries=0), ROLES, http_client=Upstream().client())
    bounded = Bounded(model, 1)

    class Asks:
        name = "verifier"

        def before(self, call: Call) -> Reply | None:
            if call.role == "answer":
                bounded.text("graph", [Message("user", "same answer?")])
            return None

        def after(self, call: Call, reply: Reply) -> None:
            return None

    model.hooks.register(Asks())
    reply = _within(WAIT_S, lambda: bounded.text("answer", [Message("user", "q")]))
    assert reply.text == "ok"


def test_call_context_reaches_hooks_inside_map() -> None:
    model = build(local(max_retries=0), ROLES, http_client=Upstream().client())
    hook = Answers()
    model.hooks.register(hook)
    bounded = Bounded(model, 4)
    with call_context(scope="acme"):
        bounded.map(lambda q: bounded.text("answer", [Message("user", q)]), ["a", "b", "c", "d"])
    assert [c.context.get("scope") for c in hook.before_calls] == ["acme"] * 4


# --- async ---


class AsyncGate:
    def __init__(self) -> None:
        self.entered = asyncio.Event()
        self.release = asyncio.Event()

    async def handle(self, request: httpx2.Request) -> httpx2.Response:
        if b"slow" in request.content:
            self.entered.set()
            await asyncio.wait_for(self.release.wait(), WAIT_S)
        return httpx2.Response(200, json=completion())


def test_an_async_hit_does_not_wait_for_the_slot() -> None:
    async def scenario() -> None:
        gate = AsyncGate()
        client = httpx2.AsyncClient(transport=httpx2.MockTransport(gate.handle))
        model = build_async(local(max_retries=0), ROLES, http_client=client)
        model.hooks.register(Answers())
        bounded = AsyncBounded(model, 1)
        slow = asyncio.create_task(bounded.text("answer", [Message("user", "slow")]))
        await asyncio.wait_for(gate.entered.wait(), WAIT_S)
        try:
            hit = await asyncio.wait_for(bounded.text("answer", [Message("user", CACHED)]), 1.0)
            assert hit.cached
        finally:
            gate.release.set()
            await slow

    asyncio.run(scenario())


class AsyncAnswers:
    """An AsyncHook: awaits (as a database query would) before answering CACHED."""

    name = "cache"

    def __init__(self) -> None:
        self.after_calls = 0

    async def before(self, call: Call) -> Reply | None:
        await asyncio.sleep(0.05)
        if call.messages[-1].content == CACHED:
            return FROM_CACHE
        return None

    async def after(self, call: Call, reply: Reply) -> None:
        await asyncio.sleep(0)
        self.after_calls += 1


def test_an_async_hook_is_awaited_and_does_not_block_other_tasks() -> None:
    async def scenario() -> tuple[list[Reply], float, int]:
        model = build_async(local(max_retries=0), ROLES, http_client=Upstream().async_client())
        hook = AsyncAnswers()
        model.hooks.register(hook)
        start = time.monotonic()
        replies = await asyncio.gather(
            *(model.text("answer", [Message("user", q)]) for q in (CACHED, CACHED, "fresh"))
        )
        return list(replies), time.monotonic() - start, hook.after_calls

    replies, elapsed, after_calls = asyncio.run(scenario())
    assert [r.cached for r in replies] == [True, True, False]
    assert after_calls == 1  # only the fresh reply
    assert elapsed < 0.15  # three 50 ms hooks overlapped, not 150 ms in a row


def test_an_async_hook_on_a_sync_model_is_refused() -> None:
    model = build(local(max_retries=0), ROLES, http_client=Upstream().client())
    with pytest.raises(TypeError, match="async"):
        model.hooks.register(AsyncAnswers())
