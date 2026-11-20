"""`build()` and `build_async()` (design §3): validate the config, resolve each role to a target,
and run calls the same way for every protocol, sync or async."""

import copy
import inspect
import os
from collections.abc import Callable, Mapping, Sequence
from contextlib import AbstractAsyncContextManager, AbstractContextManager, nullcontext
from dataclasses import replace
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Protocol, Self, cast

from pydantic import BaseModel

from lite_model.backends import Backend, anthropic_messages, openai_compat
from lite_model.config import ModelsConfig, RoleSpec, Target, resolve
from lite_model.errors import CallFailed, ConfigError
from lite_model.hooks import AsyncHook, Call, Hook, Hooks
from lite_model.profiles import PROFILES
from lite_model.types import (
    Message,
    Reply,
    StructuredReply,
    TextReply,
    ToolSpec,
    TurnMessage,
    TurnReply,
)

if TYPE_CHECKING:
    import httpx2

BACKENDS: dict[str, Backend] = {"openai": openai_compat, "anthropic": anthropic_messages}


class Model(Protocol):
    def text(self, role: str, messages: Sequence[Message]) -> TextReply: ...

    def structured[T: BaseModel](
        self, role: str, messages: Sequence[Message], schema: type[T]
    ) -> StructuredReply[T]: ...


class AsyncModel(Protocol):
    async def text(self, role: str, messages: Sequence[Message]) -> TextReply: ...

    async def structured[T: BaseModel](
        self, role: str, messages: Sequence[Message], schema: type[T]
    ) -> StructuredReply[T]: ...


class ToolModel(Protocol):
    """One model-chosen tool turn (design §3). A protocol of its own: hosts that implement
    `Model` with an adapter or a fake keep satisfying it."""

    def turn(
        self, role: str, messages: Sequence[TurnMessage], tools: Sequence[ToolSpec]
    ) -> TurnReply: ...


class AsyncToolModel(Protocol):
    async def turn(
        self, role: str, messages: Sequence[TurnMessage], tools: Sequence[ToolSpec]
    ) -> TurnReply: ...


Route = tuple[Backend, Any, Target]  # protocol module, SDK client, target


def _failed(target: Target, exc: Exception) -> CallFailed:
    # SDK, HTTP and network errors alike, after the SDK's retries on this target; no fallback
    # (§8). The message never carries upstream text (rule 8).
    return CallFailed(f"{target.where} failed: {type(exc).__name__}")


def _call(
    target: Target,
    messages: Sequence[Message],
    request: dict[str, Any],
    schema: type[BaseModel] | None,
) -> Call:
    """What hooks see: a read-only deep copy of the request, so a hook cannot change what is
    sent and the cache key describes exactly the request that goes out."""
    frozen = MappingProxyType(copy.deepcopy(request))
    return Call(target, tuple(messages), frozen, schema)


Slot = AbstractContextManager[Any]  # held only around the request (Bounded, design §10)
AsyncSlot = AbstractAsyncContextManager[Any]


def _checked(hook: Hook | AsyncHook, call: Call, hit: Reply) -> Reply:
    """A hook's reply, marked cached (design §10), after checking it fits the call."""
    schema = call.schema
    fits = (
        isinstance(hit, TextReply)
        if schema is None
        else isinstance(hit, StructuredReply) and isinstance(hit.value, schema)
    )
    if not fits:  # a hook bug: fail here, not far from the cause; never quote the reply
        wanted = "a TextReply" if schema is None else f"a StructuredReply of {schema.__name__}"
        got = type(hit).__name__
        if isinstance(hit, StructuredReply):
            got += f" of {type(hit.value).__name__}"
        raise TypeError(f"hook {hook.name!r} answered {call.target.where} with {got}, not {wanted}")
    return replace(hit, cached=True)


def _hit(call: Call, hooks: Sequence[Hook]) -> Reply | None:
    """The first hook's reply, marked cached; None when no hook answered."""
    for hook in hooks:
        if (hit := hook.before(call)) is not None:
            return _checked(hook, call, hit)
    return None


async def _hit_async(call: Call, hooks: Sequence[Hook | AsyncHook]) -> Reply | None:
    """`_hit` for an async model: an AsyncHook's `before` is awaited, a sync hook's is not."""
    for hook in hooks:
        hit = hook.before(call)
        if inspect.isawaitable(hit):
            hit = await hit
        if hit is not None:
            return _checked(hook, call, hit)
    return None


def _after(call: Call | None, hooks: Sequence[Hook], reply: Reply) -> Reply:
    if call is not None:
        for hook in hooks:
            hook.after(call, reply)
    return reply


async def _after_async(call: Call | None, hooks: Sequence[Hook | AsyncHook], reply: Reply) -> Reply:
    if call is not None:
        for hook in hooks:
            done = hook.after(call, reply)
            if inspect.isawaitable(done):
                await done
    return reply


def _send(route: Route, request: dict[str, Any], slot: Slot) -> tuple[Any, Mapping[str, str]]:
    """Send `request` and return the parsed response and its headers. Only the request waits
    for a slot: hooks run outside it. Any failure is CallFailed (§8)."""
    backend, sdk, target = route
    try:
        with slot:
            raw = backend.endpoint(sdk).create(**request)
            return raw.parse(), raw.headers
    except Exception as exc:
        raise _failed(target, exc) from exc


async def _send_async(
    route: Route, request: dict[str, Any], slot: AsyncSlot
) -> tuple[Any, Mapping[str, str]]:
    backend, sdk, target = route
    try:
        async with slot:
            raw = await backend.endpoint(sdk).create(**request)
            response = raw.parse()
            if inspect.isawaitable(response):  # anthropic's async parse() is a coroutine
                response = await response
            return response, raw.headers
    except Exception as exc:
        raise _failed(target, exc) from exc


def _run(
    route: Route,
    messages: Sequence[Message],
    schema: type[BaseModel] | None,
    hooks: Sequence[Hook],
    slot: Slot,
) -> Reply:
    backend, _, target = route
    request = backend.request(target, messages, schema)  # a local error here is not CallFailed
    call = _call(target, messages, request, schema) if hooks else None
    if call is not None and (hit := _hit(call, hooks)) is not None:
        return hit
    response, headers = _send(route, request, slot)
    return _after(call, hooks, backend.reply(target, response, headers, schema, request))


async def _run_async(
    route: Route,
    messages: Sequence[Message],
    schema: type[BaseModel] | None,
    hooks: Sequence[Hook | AsyncHook],
    slot: AsyncSlot,
) -> Reply:
    backend, _, target = route
    request = backend.request(target, messages, schema)  # a local error here is not CallFailed
    call = _call(target, messages, request, schema) if hooks else None
    if call is not None and (hit := await _hit_async(call, hooks)) is not None:
        return hit
    response, headers = await _send_async(route, request, slot)
    reply = backend.reply(target, response, headers, schema, request)
    return await _after_async(call, hooks, reply)


def _turn_request(
    route: Route, messages: Sequence[TurnMessage], tools: Sequence[ToolSpec]
) -> dict[str, Any]:
    backend, _, target = route
    if not PROFILES[target.type].supports_tools:
        raise ConfigError(f"{target.where}: backend type {target.type} does not take tools")
    return backend.turn_request(target, messages, tools)  # a local error is not CallFailed


def _run_turn(
    route: Route, messages: Sequence[TurnMessage], tools: Sequence[ToolSpec], slot: Slot
) -> TurnReply:
    """`_run` for `turn()`. No hooks: their replies are text or structured, and nothing caches
    an agent's turns (design §10)."""
    backend, _, target = route
    request = _turn_request(route, messages, tools)
    response, headers = _send(route, request, slot)
    return backend.turn_reply(target, response, headers, request)


async def _run_turn_async(
    route: Route, messages: Sequence[TurnMessage], tools: Sequence[ToolSpec], slot: AsyncSlot
) -> TurnReply:
    backend, _, target = route
    request = _turn_request(route, messages, tools)
    response, headers = await _send_async(route, request, slot)
    return backend.turn_reply(target, response, headers, request)


class _Routes:
    def __init__(
        self, routes: dict[str, Route], hooks: Hooks | None = None, slot: Any = None
    ) -> None:
        self._routes = routes  # role -> route; one SDK client per endpoint
        self.hooks = hooks if hooks is not None else Hooks()  # design §10
        # Taken around the request only (`Bounded`, design §10): a Slot for LiteModel, an
        # AsyncSlot for AsyncLiteModel.
        self._slot = slot if slot is not None else nullcontext()

    def limited(self, slot: Any) -> Self:
        """The same routes and hooks with each request taking `slot` (`Bounded` / `AsyncBounded`,
        design §10)."""
        return type(self)(self._routes, self.hooks, slot)

    def target(self, role: str) -> Target:
        """`role`'s resolved target (model, backend, endpoint): what a host records before any
        call, e.g. as part of an index's identity (design §12.1)."""
        return self._route(role)[2]

    def _route(self, role: str) -> Route:
        if role not in self._routes:
            raise ConfigError(f"role {role}: not declared by the host")
        return self._routes[role]


class LiteModel(_Routes):
    """What `build()` returns: a `Model` and a `ToolModel`, with `.hooks`."""

    def _hooks(self) -> Sequence[Hook]:
        return cast(Sequence[Hook], self.hooks.snapshot())  # Hooks refuses an AsyncHook here

    def text(self, role: str, messages: Sequence[Message]) -> TextReply:
        reply = _run(self._route(role), messages, None, self._hooks(), self._slot)
        return cast(TextReply, reply)

    def structured[T: BaseModel](
        self, role: str, messages: Sequence[Message], schema: type[T]
    ) -> StructuredReply[T]:
        reply = _run(self._route(role), messages, schema, self._hooks(), self._slot)
        return cast(StructuredReply[T], reply)

    def turn(
        self, role: str, messages: Sequence[TurnMessage], tools: Sequence[ToolSpec]
    ) -> TurnReply:
        return _run_turn(self._route(role), messages, tools, self._slot)


class AsyncLiteModel(_Routes):
    """What `build_async()` returns: an `AsyncModel` with `.hooks` (sync hooks and `AsyncHook`s,
    design §10)."""

    def __init__(
        self, routes: dict[str, Route], hooks: Hooks | None = None, slot: Any = None
    ) -> None:
        super().__init__(routes, hooks if hooks is not None else Hooks(allow_async=True), slot)

    async def text(self, role: str, messages: Sequence[Message]) -> TextReply:
        hooks = self.hooks.snapshot()
        reply = await _run_async(self._route(role), messages, None, hooks, self._slot)
        return cast(TextReply, reply)

    async def structured[T: BaseModel](
        self, role: str, messages: Sequence[Message], schema: type[T]
    ) -> StructuredReply[T]:
        hooks = self.hooks.snapshot()
        reply = await _run_async(self._route(role), messages, schema, hooks, self._slot)
        return cast(StructuredReply[T], reply)

    async def turn(
        self, role: str, messages: Sequence[TurnMessage], tools: Sequence[ToolSpec]
    ) -> TurnReply:
        return await _run_turn_async(self._route(role), messages, tools, self._slot)


def _routes(
    config: ModelsConfig, roles: RoleSpec, connect: Callable[[Backend, Target, str | None], Any]
) -> dict[str, Route]:
    shared: dict[tuple[Any, ...], Any] = {}
    routes: dict[str, Route] = {}
    for role, t in resolve(config, roles).items():
        backend = BACKENDS[PROFILES[t.type].protocol]
        slot = (t.type, t.base_url, t.api_key_env, t.timeout_s, t.connect_timeout_s, t.max_retries)
        if slot not in shared:
            key = os.environ.get(t.api_key_env) if t.api_key_env else None
            shared[slot] = connect(backend, t, key)
        routes[role] = (backend, shared[slot], t)
    return routes


def build(
    config: ModelsConfig, roles: RoleSpec, *, http_client: "httpx2.Client | None" = None
) -> LiteModel:
    """The model for `roles`, or ConfigError before any call (design §4).

    `http_client` replaces the SDK's transport (a proxy, or a test's recording transport)."""
    return LiteModel(_routes(config, roles, lambda b, t, k: b.connect(t, k, http_client)))


def build_async(
    config: ModelsConfig, roles: RoleSpec, *, http_client: "httpx2.AsyncClient | None" = None
) -> AsyncLiteModel:
    """`build()` for asyncio callers."""
    return AsyncLiteModel(
        _routes(config, roles, lambda b, t, k: b.connect_async(t, k, http_client))
    )
