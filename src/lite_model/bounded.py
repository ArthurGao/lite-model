"""At most `concurrency` calls in flight through one model, and `map()` for independent calls
in input order (design §10; lite-rag's `BoundedCalls`). A model shared by several jobs (a server
ingesting documents in parallel) stays within what the endpoint and the quota can take."""

import asyncio
import threading
from collections.abc import Callable, Coroutine, Sequence
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from pydantic import BaseModel

from lite_model.client import AsyncModel, Model
from lite_model.types import Message, StructuredReply, TextReply


class Bounded:
    """Implements `Model`; every call through it takes one of `concurrency` slots."""

    def __init__(self, model: Model, concurrency: int = 1) -> None:
        if concurrency < 1:
            raise ValueError("concurrency must be at least 1")
        self._model = model
        self._concurrency = concurrency
        self._slots = threading.BoundedSemaphore(concurrency)

    def text(self, role: str, messages: Sequence[Message]) -> TextReply:
        with self._slots:
            return self._model.text(role, messages)

    def structured[T: BaseModel](
        self, role: str, messages: Sequence[Message], schema: type[T]
    ) -> StructuredReply[T]:
        with self._slots:
            return self._model.structured(role, messages, schema)

    def map[I, R](self, one: Callable[[I], R], items: Sequence[I]) -> list[R]:
        """`one` over `items`, results in input order, up to `concurrency` at once. `one` calls
        this Bounded (not the wrapped model) so its calls take slots; the first error raises."""
        if self._concurrency == 1 or len(items) <= 1:
            return [one(i) for i in items]
        with ThreadPoolExecutor(min(self._concurrency, len(items))) as pool:
            return list(pool.map(one, items))


class AsyncBounded:
    """Implements `AsyncModel`; every call through it takes one of `concurrency` slots."""

    def __init__(self, model: AsyncModel, concurrency: int = 1) -> None:
        if concurrency < 1:
            raise ValueError("concurrency must be at least 1")
        self._model = model
        self._slots = asyncio.Semaphore(concurrency)

    async def text(self, role: str, messages: Sequence[Message]) -> TextReply:
        async with self._slots:
            return await self._model.text(role, messages)

    async def structured[T: BaseModel](
        self, role: str, messages: Sequence[Message], schema: type[T]
    ) -> StructuredReply[T]:
        async with self._slots:
            return await self._model.structured(role, messages, schema)

    async def map[I, R](
        self, one: Callable[[I], Coroutine[Any, Any, R]], items: Sequence[I]
    ) -> list[R]:
        """`one` over `items`, results in input order; the slots bound how many run at once.
        A failure cancels the rest and raises an ExceptionGroup (asyncio.TaskGroup)."""
        async with asyncio.TaskGroup() as group:
            tasks = [group.create_task(one(i)) for i in items]
        return [t.result() for t in tasks]
