"""`build()` and `build_async()` (design §3): validate the config, resolve each role to a target,
and run calls the same way for every protocol, sync or async."""

import inspect
import os
from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Any, Protocol, cast

from pydantic import BaseModel

from lite_model.backends import Backend, Reply, anthropic_messages, openai_compat
from lite_model.config import ModelsConfig, RoleSpec, Target, resolve
from lite_model.errors import CallFailed, ConfigError
from lite_model.profiles import PROFILES
from lite_model.types import Message, StructuredReply, TextReply

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


def _run(route: Route, messages: Sequence[Message], schema: type[BaseModel] | None) -> Reply:
    backend, sdk, target = route
    request = backend.request(target, messages, schema)  # a local error here is not CallFailed
    try:
        raw = backend.endpoint(sdk).create(**request)
        response = raw.parse()
    except Exception as exc:
        raise _failed(target, exc) from exc
    return backend.reply(target, response, raw.headers, schema, request)


async def _run_async(
    route: Route, messages: Sequence[Message], schema: type[BaseModel] | None
) -> Reply:
    backend, sdk, target = route
    request = backend.request(target, messages, schema)
    try:
        raw = await backend.endpoint(sdk).create(**request)
        response = raw.parse()
        if inspect.isawaitable(response):  # anthropic's async parse() is a coroutine, openai's not
            response = await response
    except Exception as exc:
        raise _failed(target, exc) from exc
    return backend.reply(target, response, raw.headers, schema, request)


class _Routes:
    def __init__(self, routes: dict[str, Route]) -> None:
        self._routes = routes  # role -> route; one SDK client per endpoint

    def _route(self, role: str) -> Route:
        if role not in self._routes:
            raise ConfigError(f"role {role}: not declared by the host")
        return self._routes[role]


class _Model(_Routes):
    def text(self, role: str, messages: Sequence[Message]) -> TextReply:
        return cast(TextReply, _run(self._route(role), messages, None))

    def structured[T: BaseModel](
        self, role: str, messages: Sequence[Message], schema: type[T]
    ) -> StructuredReply[T]:
        return cast(StructuredReply[T], _run(self._route(role), messages, schema))


class _AsyncModel(_Routes):
    async def text(self, role: str, messages: Sequence[Message]) -> TextReply:
        return cast(TextReply, await _run_async(self._route(role), messages, None))

    async def structured[T: BaseModel](
        self, role: str, messages: Sequence[Message], schema: type[T]
    ) -> StructuredReply[T]:
        return cast(StructuredReply[T], await _run_async(self._route(role), messages, schema))


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
) -> Model:
    """The model for `roles`, or ConfigError before any call (design §4).

    `http_client` replaces the SDK's transport (a proxy, or a test's recording transport)."""
    return _Model(_routes(config, roles, lambda b, t, k: b.connect(t, k, http_client)))


def build_async(
    config: ModelsConfig, roles: RoleSpec, *, http_client: "httpx2.AsyncClient | None" = None
) -> AsyncModel:
    """`build()` for asyncio callers."""
    return _AsyncModel(_routes(config, roles, lambda b, t, k: b.connect_async(t, k, http_client)))
