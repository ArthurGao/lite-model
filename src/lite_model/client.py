"""`build()` and `build_async()` (design §3): validate the config, resolve each role to a target,
and run calls the same way for every protocol, sync or async."""

import copy
import inspect
import os
from collections.abc import Callable, Sequence
from dataclasses import replace
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Protocol, cast

from pydantic import BaseModel

from lite_model.backends import Backend, anthropic_messages, openai_compat
from lite_model.config import ModelsConfig, RoleSpec, Target, resolve
from lite_model.errors import CallFailed, ConfigError
from lite_model.hooks import Call, Hook, Hooks
from lite_model.profiles import PROFILES
from lite_model.types import Message, Reply, StructuredReply, TextReply

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


def _hit(call: Call, hooks: Sequence[Hook]) -> Reply | None:
    """The first hook's reply, marked cached (design §10); None when no hook answered."""
    for hook in hooks:
        hit = hook.before(call)
        if hit is None:
            continue
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
            raise TypeError(
                f"hook {hook.name!r} answered {call.target.where} with {got}, not {wanted}"
            )
        return replace(hit, cached=True)
    return None


def _after(call: Call | None, hooks: Sequence[Hook], reply: Reply) -> Reply:
    if call is not None:
        for hook in hooks:
            hook.after(call, reply)
    return reply


def _run(
    route: Route, messages: Sequence[Message], schema: type[BaseModel] | None, hooks: Sequence[Hook]
) -> Reply:
    backend, sdk, target = route
    request = backend.request(target, messages, schema)  # a local error here is not CallFailed
    call = _call(target, messages, request, schema) if hooks else None
    if call is not None and (hit := _hit(call, hooks)) is not None:
        return hit
    try:
        raw = backend.endpoint(sdk).create(**request)
        response = raw.parse()
    except Exception as exc:
        raise _failed(target, exc) from exc
    return _after(call, hooks, backend.reply(target, response, raw.headers, schema, request))


async def _run_async(
    route: Route, messages: Sequence[Message], schema: type[BaseModel] | None, hooks: Sequence[Hook]
) -> Reply:
    backend, sdk, target = route
    request = backend.request(target, messages, schema)  # a local error here is not CallFailed
    call = _call(target, messages, request, schema) if hooks else None
    if call is not None and (hit := _hit(call, hooks)) is not None:
        return hit
    try:
        raw = await backend.endpoint(sdk).create(**request)
        response = raw.parse()
        if inspect.isawaitable(response):  # anthropic's async parse() is a coroutine, openai's not
            response = await response
    except Exception as exc:
        raise _failed(target, exc) from exc
    return _after(call, hooks, backend.reply(target, response, raw.headers, schema, request))


class _Routes:
    def __init__(self, routes: dict[str, Route]) -> None:
        self._routes = routes  # role -> route; one SDK client per endpoint
        self.hooks = Hooks()  # design §10

    def target(self, role: str) -> Target:
        """`role`'s resolved target (model, backend, endpoint): what a host records before any
        call, e.g. as part of an index's identity (design §12.1)."""
        return self._route(role)[2]

    def _route(self, role: str) -> Route:
        if role not in self._routes:
            raise ConfigError(f"role {role}: not declared by the host")
        return self._routes[role]


class LiteModel(_Routes):
    """What `build()` returns: a `Model` with `.hooks`."""

    def text(self, role: str, messages: Sequence[Message]) -> TextReply:
        reply = _run(self._route(role), messages, None, self.hooks.snapshot())
        return cast(TextReply, reply)

    def structured[T: BaseModel](
        self, role: str, messages: Sequence[Message], schema: type[T]
    ) -> StructuredReply[T]:
        reply = _run(self._route(role), messages, schema, self.hooks.snapshot())
        return cast(StructuredReply[T], reply)


class AsyncLiteModel(_Routes):
    """What `build_async()` returns: an `AsyncModel` with `.hooks` (sync hooks, design §10)."""

    async def text(self, role: str, messages: Sequence[Message]) -> TextReply:
        reply = await _run_async(self._route(role), messages, None, self.hooks.snapshot())
        return cast(TextReply, reply)

    async def structured[T: BaseModel](
        self, role: str, messages: Sequence[Message], schema: type[T]
    ) -> StructuredReply[T]:
        reply = await _run_async(self._route(role), messages, schema, self.hooks.snapshot())
        return cast(StructuredReply[T], reply)


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
