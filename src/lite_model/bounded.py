"""At most `concurrency` calls in flight through one model, and `map()` for independent calls
in input order (design §10; lite-rag's `BoundedCalls`). A model shared by several jobs (a server
ingesting documents in parallel) stays within what the endpoint and the quota can take."""

import asyncio
import contextvars
import threading
from collections.abc import Callable, Coroutine, Sequence
from concurrent.futures import ThreadPoolExecutor
from contextlib import AbstractAsyncContextManager, AbstractContextManager, nullcontext
from typing import Any, Protocol, runtime_checkable

from pydantic import BaseModel

from lite_model.client import AsyncModel, Model
from lite_model.types import Message, StructuredReply, TextReply


@runtime_checkable
class Limitable(Protocol):
    """A model that can take a slot around its requests only (`LiteModel`, `AsyncLiteModel`, or
    a host adapter that forwards `limited`): `Bounded` then never holds a slot while a hook runs."""

    def limited(self, slot: Any) -> Any: ...


class Bounded:
    """Implements `Model`; every call through it takes one of `concurrency` slots. Around a
    `Limitable` model only the request holds the slot (a hook hit never queues, and a hook may
    call through this Bounded); around any other `Model` (e.g. `FakeModel`) the whole call does,
    so there a hook that calls through the same Bounded at concurrency 1 would deadlock."""

    def __init__(self, model: Model, concurrency: int = 1) -> None:
        if concurrency < 1:
            raise ValueError("concurrency must be at least 1")
        self._concurrency = concurrency
        self._slots = threading.BoundedSemaphore(concurrency)
        self._model: Model = model
        self._whole: AbstractContextManager[Any] = self._slots
        if isinstance(model, Limitable):  # the slot moves into the request (0.3.0)
            self._model, self._whole = model.limited(self._slots), nullcontext()

    def text(self, role: str, messages: Sequence[Message]) -> TextReply:
        with self._whole:
            return self._model.text(role, messages)

    def structured[T: BaseModel](
        self, role: str, messages: Sequence[Message], schema: type[T]
    ) -> StructuredReply[T]:
        with self._whole:
            return self._model.structured(role, messages, schema)

    def map[I, R](self, one: Callable[[I], R], items: Sequence[I]) -> list[R]:
        """`one` over `items`, results in input order, up to `concurrency` at once. `one` calls
        this Bounded (not the wrapped model) so its calls take slots; the first error raises.
        Each item runs in a copy of the caller's context, so `call_context()` reaches the hooks
        of calls made in the worker threads (0.3.0)."""
        if self._concurrency == 1 or len(items) <= 1:
            return [one(i) for i in items]
        context = contextvars.copy_context()
        with ThreadPoolExecutor(min(self._concurrency, len(items))) as pool:
            # a fresh copy per item: one Context cannot be entered by two threads at once
            return list(pool.map(lambda i: context.copy().run(one, i), items))


class AsyncBounded:
    """Implements `AsyncModel`; every call through it takes one of `concurrency` slots, as
    `Bounded` does (around a `Limitable` model, only the request)."""

    def __init__(self, model: AsyncModel, concurrency: int = 1) -> None:
        if concurrency < 1:
            raise ValueError("concurrency must be at least 1")
        self._slots = asyncio.Semaphore(concurrency)
        self._model: AsyncModel = model
        self._whole: AbstractAsyncContextManager[Any] = self._slots
        if isinstance(model, Limitable):
            self._model, self._whole = model.limited(self._slots), nullcontext()

    async def text(self, role: str, messages: Sequence[Message]) -> TextReply:
        async with self._whole:
            return await self._model.text(role, messages)

    async def structured[T: BaseModel](
        self, role: str, messages: Sequence[Message], schema: type[T]
    ) -> StructuredReply[T]:
        async with self._whole:
            return await self._model.structured(role, messages, schema)

    async def map[I, R](
        self, one: Callable[[I], Coroutine[Any, Any, R]], items: Sequence[I]
    ) -> list[R]:
        """`one` over `items`, results in input order; the slots bound how many run at once.
        A failure cancels the rest and raises an ExceptionGroup (asyncio.TaskGroup)."""
        async with asyncio.TaskGroup() as group:
            tasks = [group.create_task(one(i)) for i in items]
        return [t.result() for t in tasks]
