"""Every contract test runs twice: through `build()` and through `build_async()` (design §14 S2:
one set of contract tests for both)."""

import asyncio
from collections.abc import Callable, Sequence
from typing import Any, Protocol, cast

import pytest
from pydantic import BaseModel

from lite_model import (
    Message,
    ModelsConfig,
    RolePolicy,
    RoleSpec,
    StructuredReply,
    TextReply,
    build,
    build_async,
)
from lite_model.hooks import Hooks
from tests.upstream import Upstream

ROLES = RoleSpec({"answer": RolePolicy(routable=True), "graph": RolePolicy()})


class Caller:
    """A built model with a sync face, whichever way it was built."""

    def __init__(self, mode: str, upstream: Upstream, config: ModelsConfig) -> None:
        self._model: Any
        self._wait: Callable[[Any], Any]
        if mode == "sync":
            self._model = build(config, ROLES, http_client=upstream.client())
            self._wait = lambda reply: reply
        else:
            self._model = build_async(config, ROLES, http_client=upstream.async_client())
            self._wait = asyncio.run

    @property
    def hooks(self) -> Hooks:
        return cast(Hooks, self._model.hooks)

    def text(self, role: str, messages: Sequence[Message]) -> TextReply:
        return cast(TextReply, self._wait(self._model.text(role, messages)))

    def structured[T: BaseModel](
        self, role: str, messages: Sequence[Message], schema: type[T]
    ) -> StructuredReply[T]:
        return cast(StructuredReply[T], self._wait(self._model.structured(role, messages, schema)))


class Connect(Protocol):
    def __call__(self, upstream: Upstream, config: ModelsConfig | None = None) -> Caller: ...


@pytest.fixture(params=["sync", "async"])
def connect(request: pytest.FixtureRequest) -> Connect:
    def make(upstream: Upstream, config: ModelsConfig | None = None) -> Caller:
        return Caller(request.param, upstream, config or ModelsConfig(max_retries=0))

    return make
