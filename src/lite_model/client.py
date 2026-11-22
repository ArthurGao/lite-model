"""`build()` and `build_async()` (design §3): validate the config, resolve each role to a target,
and run calls the same way for every protocol, sync or async."""

import asyncio
import contextvars
import copy
import inspect
import os
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from contextlib import AbstractAsyncContextManager, AbstractContextManager, nullcontext
from dataclasses import replace
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Protocol, Self, cast

from pydantic import BaseModel

from lite_model.backends import (
    Backend,
    EmbedBackend,
    RerankBackend,
    anthropic_messages,
    cohere_embed,
    gemini_embed,
    openai_compat,
)
from lite_model.config import ModelsConfig, RoleSpec, Target, resolve
from lite_model.errors import BadStructuredReply, CallFailed, ConfigError, ProvenanceUnknown
from lite_model.hooks import AsyncHook, Call, Hook, Hooks
from lite_model.profiles import PROFILES
from lite_model.types import (
    EmbedReply,
    InputType,
    Message,
    Reply,
    RerankReply,
    RoleKind,
    StructuredReply,
    TextReply,
    ToolSpec,
    TurnMessage,
    TurnReply,
    Usage,
)

if TYPE_CHECKING:
    import httpx2

# Embed-only protocol modules (0.8.0) implement EmbedBackend only: build() admits no chat role
# on their types (`serves`), so the chat half of Backend is never called on them.
BACKENDS: dict[str, Backend] = {
    "openai": openai_compat,
    "anthropic": anthropic_messages,
    "cohere": cast(Backend, cohere_embed),
    "gemini": cast(Backend, gemini_embed),
}


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


class EmbedModel(Protocol):
    """Vectors for texts (design §3.2). A protocol of its own, like `ToolModel`. `input_type`
    (0.7.0): what the texts are, on a type that takes it (Voyage); refused elsewhere."""

    def embed(
        self, role: str, texts: Sequence[str], *, input_type: InputType | None = None
    ) -> EmbedReply: ...


class AsyncEmbedModel(Protocol):
    async def embed(
        self, role: str, texts: Sequence[str], *, input_type: InputType | None = None
    ) -> EmbedReply: ...


class RerankModel(Protocol):
    """Relevance scores of documents for a query (design §3.3, 0.7.0)."""

    def rerank(
        self, role: str, query: str, documents: Sequence[str], *, top_k: int | None = None
    ) -> RerankReply: ...


class AsyncRerankModel(Protocol):
    async def rerank(
        self, role: str, query: str, documents: Sequence[str], *, top_k: int | None = None
    ) -> RerankReply: ...


Route = tuple[Backend, Any, Target]  # protocol module, SDK client, target
_KIND: dict[str, RoleKind] = {"embed": "embed", "rerank": "rerank"}  # method -> kind; else chat


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


def _send(
    target: Target, owner: Any, request: dict[str, Any], slot: Slot
) -> tuple[Any, Mapping[str, str]]:
    """Send `request` through `owner` (the SDK's raw-response `create` owner: a backend's
    `endpoint` or `embed_endpoint`) and return the parsed response and its headers. Only the
    request waits for a slot: hooks run outside it. Any failure is CallFailed (§8)."""
    try:
        with slot:
            raw = owner.create(**request)
            return raw.parse(), raw.headers
    except Exception as exc:
        raise _failed(target, exc) from exc


async def _send_async(
    target: Target, owner: Any, request: dict[str, Any], slot: AsyncSlot
) -> tuple[Any, Mapping[str, str]]:
    try:
        async with slot:
            raw = await owner.create(**request)
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
    backend, sdk, target = route
    request = backend.request(target, messages, schema)  # a local error here is not CallFailed
    call = _call(target, messages, request, schema) if hooks else None
    if call is not None and (hit := _hit(call, hooks)) is not None:
        return hit
    response, headers = _send(target, backend.endpoint(sdk), request, slot)
    return _after(call, hooks, backend.reply(target, response, headers, schema, request))


async def _run_async(
    route: Route,
    messages: Sequence[Message],
    schema: type[BaseModel] | None,
    hooks: Sequence[Hook | AsyncHook],
    slot: AsyncSlot,
) -> Reply:
    backend, sdk, target = route
    request = backend.request(target, messages, schema)  # a local error here is not CallFailed
    call = _call(target, messages, request, schema) if hooks else None
    if call is not None and (hit := await _hit_async(call, hooks)) is not None:
        return hit
    response, headers = await _send_async(target, backend.endpoint(sdk), request, slot)
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
    backend, sdk, target = route
    request = _turn_request(route, messages, tools)
    response, headers = _send(target, backend.endpoint(sdk), request, slot)
    return backend.turn_reply(target, response, headers, request)


async def _run_turn_async(
    route: Route, messages: Sequence[TurnMessage], tools: Sequence[ToolSpec], slot: AsyncSlot
) -> TurnReply:
    backend, sdk, target = route
    request = _turn_request(route, messages, tools)
    response, headers = await _send_async(target, backend.endpoint(sdk), request, slot)
    return backend.turn_reply(target, response, headers, request)


def _batches(target: Target, texts: Sequence[str]) -> list[list[str]]:
    if not texts:  # nothing to send, so no provenance to record (design §3.2)
        raise ValueError(f"{target.where}: no texts to embed")
    size = target.batch_size
    return [list(texts[i : i + size]) for i in range(0, len(texts), size)]


def in_threads[I, R](one: Callable[[I], R], items: Sequence[I], concurrency: int) -> list[R]:
    """`one` over `items` on up to `concurrency` threads, results in input order, each item in
    a copy of the caller's context (so `call_context()` reaches the hooks). The first error
    raises, and items not started yet are cancelled: no request after the call has failed."""
    if concurrency == 1 or len(items) <= 1:
        return [one(i) for i in items]
    context = contextvars.copy_context()
    pool = ThreadPoolExecutor(min(concurrency, len(items)))
    try:
        # a fresh copy per item: one Context cannot be entered by two threads at once
        return list(pool.map(lambda i: context.copy().run(one, i), items))
    except BaseException:
        pool.shutdown(wait=True, cancel_futures=True)
        raise
    finally:
        pool.shutdown(wait=True)


# What endpoints answer a request over their input limit. 400 too: Voyage, for one, answers a
# batch over its token limit with 400; a 400 for another reason costs log2(batch) extra requests.
TOO_LARGE = frozenset({400, 413})


def _too_large(exc: CallFailed) -> bool:
    return getattr(exc.__cause__, "status_code", None) in TOO_LARGE


def _total(values: Sequence[float | None]) -> float | None:
    return None if None in values else sum(v for v in values if v is not None)


def _merged(target: Target, replies: Sequence[EmbedReply]) -> EmbedReply:
    """The batches as one reply: one answering model, one vector length, usage summed (None
    when any batch did not report it)."""
    first = replies[0]
    if len(replies) == 1:
        return first
    if len({r.provenance.answered for r in replies}) > 1:
        raise ProvenanceUnknown(f"{target.where}: the batches were answered by more than one model")
    if len({r.dim for r in replies}) > 1:
        raise BadStructuredReply(f"{target.where}: vectors of different lengths across batches")
    tokens = _total([r.usage.input for r in replies])
    usage = Usage(
        input=None if tokens is None else int(tokens),
        output=0,  # an embedding has no output tokens
        cost_usd=_total([r.usage.cost_usd for r in replies]),
    )
    return EmbedReply([v for r in replies for v in r.vectors], first.provenance, usage)


def _embedding(route: Route) -> tuple[EmbedBackend, Any, Target]:
    # build() admits an embed role only on a type that serves embeddings (`serves`): its
    # protocol module is an EmbedBackend (OpenAI-compatible, or an embed-only one over `rest`).
    backend, sdk, target = route
    return cast(EmbedBackend, backend), cast(EmbedBackend, backend).embed_endpoint(sdk), target


def _input_type(target: Target, input_type: InputType | None) -> InputType | None:
    if input_type is not None and not PROFILES[target.type].embed_input_type:
        raise ConfigError(f"{target.where}: {target.type} takes no input_type (design §3.3)")
    return input_type


def _run_embed(
    route: Route, texts: Sequence[str], slot: Slot, input_type: InputType | None = None
) -> EmbedReply:
    """`embed()` (design §3.2): batches of `batch_size`, `max_concurrency` at a time, merged in
    input order; no hooks. A batch the endpoint refuses as too large (400 / 413) is halved until
    it fits; a single text it still refuses fails the call (0.6.0)."""
    backend, owner, target = _embedding(route)
    kind = _input_type(target, input_type)

    def fitting(batch: list[str]) -> list[EmbedReply]:
        request = backend.embed_request(target, batch, kind)
        try:
            response, headers = _send(target, owner, request, slot)
        except CallFailed as exc:
            if len(batch) == 1 or not _too_large(exc):
                raise
            half = len(batch) // 2
            return fitting(batch[:half]) + fitting(batch[half:])
        return [backend.embed_reply(target, response, headers, request)]

    parts = in_threads(fitting, _batches(target, texts), target.max_concurrency)
    return _merged(target, [r for part in parts for r in part])


async def _run_embed_async(
    route: Route, texts: Sequence[str], slot: AsyncSlot, input_type: InputType | None = None
) -> EmbedReply:
    backend, owner, target = _embedding(route)
    kind = _input_type(target, input_type)
    limit = asyncio.Semaphore(target.max_concurrency)

    async def fitting(batch: list[str]) -> list[EmbedReply]:
        request = backend.embed_request(target, batch, kind)
        try:
            async with limit:
                response, headers = await _send_async(target, owner, request, slot)
        except CallFailed as exc:
            if len(batch) == 1 or not _too_large(exc):
                raise
            half = len(batch) // 2
            return [*await fitting(batch[:half]), *await fitting(batch[half:])]
        return [backend.embed_reply(target, response, headers, request)]

    batches = _batches(target, texts)
    try:
        async with asyncio.TaskGroup() as group:  # a failure cancels the batches still waiting
            tasks = [group.create_task(fitting(b)) for b in batches]
    except ExceptionGroup as failed:  # the caller sees the error itself, as from one batch
        raise failed.exceptions[0] from None
    return _merged(target, [r for t in tasks for r in t.result()])


def _rerank_request(
    route: Route, query: str, documents: Sequence[str], top_k: int | None
) -> tuple[RerankBackend, Any, Target, dict[str, Any]]:
    # build() admits a rerank role only on a type that serves it (`serves`)
    module, sdk, target = route
    if not documents:
        raise ValueError(f"{target.where}: no documents to rerank")
    if top_k is not None and top_k < 1:
        raise ValueError(f"{target.where}: top_k must be at least 1, not {top_k}")
    backend = cast(RerankBackend, module)
    request = backend.rerank_request(target, query, list(documents), top_k)
    return backend, backend.rerank_endpoint(sdk), target, request


def _run_rerank(
    route: Route, query: str, documents: Sequence[str], top_k: int | None, slot: Slot
) -> RerankReply:
    """`rerank()` (design §3.3): one request, no hooks; scores in input order."""
    backend, owner, target, request = _rerank_request(route, query, documents, top_k)
    response, headers = _send(target, owner, request, slot)
    return backend.rerank_reply(target, response, headers, request)


async def _run_rerank_async(
    route: Route, query: str, documents: Sequence[str], top_k: int | None, slot: AsyncSlot
) -> RerankReply:
    backend, owner, target, request = _rerank_request(route, query, documents, top_k)
    response, headers = await _send_async(target, owner, request, slot)
    return backend.rerank_reply(target, response, headers, request)


class _Clients:
    """The SDK clients a model built, shared by the models `limited()` derives from it. Closed
    once; a client built over the caller's `http_client` is the caller's to close (0.6.0)."""

    def __init__(self, routes: dict[str, Route], owned: bool) -> None:
        self._sdks = list({id(r[1]): r[1] for r in routes.values()}.values()) if owned else []
        self.closed = False

    def release(self) -> list[Any]:
        """The clients to close now: all of them the first time, none after (idempotent)."""
        if self.closed:
            return []
        self.closed = True
        return self._sdks


class _Routes:
    def __init__(
        self,
        routes: dict[str, Route],
        hooks: Hooks | None = None,
        slot: Any = None,
        *,
        clients: _Clients | None = None,
    ) -> None:
        self._routes = routes  # role -> route; one SDK client per endpoint
        self.hooks = hooks if hooks is not None else Hooks()  # design §10
        # Taken around the request only (`Bounded`, design §10): a Slot for LiteModel, an
        # AsyncSlot for AsyncLiteModel.
        self._slot = slot if slot is not None else nullcontext()
        self._clients = clients if clients is not None else _Clients(routes, owned=False)

    def limited(self, slot: Any) -> Self:
        """The same routes and hooks with each request taking `slot` (`Bounded` / `AsyncBounded`,
        design §10)."""
        return type(self)(self._routes, self.hooks, slot, clients=self._clients)

    def target(self, role: str) -> Target:
        """`role`'s resolved target (model, backend, endpoint): what a host records before any
        call, e.g. as part of an index's identity (design §12.1)."""
        if role not in self._routes:
            raise ConfigError(f"role {role}: not declared by the host")
        return self._routes[role][2]  # any kind of role

    def _route(self, role: str, method: str = "text") -> Route:
        """`role`'s route, checked to be of the kind `method` needs (design §3.2)."""
        if self._clients.closed:
            raise RuntimeError(f"{type(self).__name__} is closed: {method}() on role {role}")
        kind = _KIND.get(method, "chat")
        have = self.target(role).kind
        if have != kind:
            article = "an" if have == "embed" else "a"
            raise ConfigError(f"role {role}: {article} {have} role; {method}() needs a {kind} role")
        return self._routes[role]


class LiteModel(_Routes):
    """What `build()` returns: a `Model`, a `ToolModel` and an `EmbedModel`, with `.hooks`."""

    def close(self) -> None:
        """Close the SDK clients this model built (not an `http_client` the caller passed);
        calls after it raise. Also `with build(...) as model:` (0.6.0)."""
        for sdk in self._clients.release():
            sdk.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def _hooks(self) -> Sequence[Hook]:
        return cast(Sequence[Hook], self.hooks.snapshot())  # Hooks refuses an AsyncHook here

    def text(self, role: str, messages: Sequence[Message]) -> TextReply:
        reply = _run(self._route(role), messages, None, self._hooks(), self._slot)
        return cast(TextReply, reply)

    def structured[T: BaseModel](
        self, role: str, messages: Sequence[Message], schema: type[T]
    ) -> StructuredReply[T]:
        route = self._route(role, "structured")
        reply = _run(route, messages, schema, self._hooks(), self._slot)
        return cast(StructuredReply[T], reply)

    def turn(
        self, role: str, messages: Sequence[TurnMessage], tools: Sequence[ToolSpec]
    ) -> TurnReply:
        return _run_turn(self._route(role, "turn"), messages, tools, self._slot)

    def embed(
        self, role: str, texts: Sequence[str], *, input_type: InputType | None = None
    ) -> EmbedReply:
        return _run_embed(self._route(role, "embed"), texts, self._slot, input_type)

    def rerank(
        self, role: str, query: str, documents: Sequence[str], *, top_k: int | None = None
    ) -> RerankReply:
        return _run_rerank(self._route(role, "rerank"), query, documents, top_k, self._slot)


class AsyncLiteModel(_Routes):
    """What `build_async()` returns: an `AsyncModel` with `.hooks` (sync hooks and `AsyncHook`s,
    design §10)."""

    def __init__(
        self,
        routes: dict[str, Route],
        hooks: Hooks | None = None,
        slot: Any = None,
        *,
        clients: _Clients | None = None,
    ) -> None:
        hooks = hooks if hooks is not None else Hooks(allow_async=True)
        super().__init__(routes, hooks, slot, clients=clients)

    async def aclose(self) -> None:
        """`LiteModel.close()` for asyncio callers; also `async with build_async(...)`."""
        for sdk in self._clients.release():
            await sdk.close()

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.aclose()

    async def text(self, role: str, messages: Sequence[Message]) -> TextReply:
        hooks = self.hooks.snapshot()
        reply = await _run_async(self._route(role), messages, None, hooks, self._slot)
        return cast(TextReply, reply)

    async def structured[T: BaseModel](
        self, role: str, messages: Sequence[Message], schema: type[T]
    ) -> StructuredReply[T]:
        hooks = self.hooks.snapshot()
        route = self._route(role, "structured")
        reply = await _run_async(route, messages, schema, hooks, self._slot)
        return cast(StructuredReply[T], reply)

    async def turn(
        self, role: str, messages: Sequence[TurnMessage], tools: Sequence[ToolSpec]
    ) -> TurnReply:
        return await _run_turn_async(self._route(role, "turn"), messages, tools, self._slot)

    async def embed(
        self, role: str, texts: Sequence[str], *, input_type: InputType | None = None
    ) -> EmbedReply:
        return await _run_embed_async(self._route(role, "embed"), texts, self._slot, input_type)

    async def rerank(
        self, role: str, query: str, documents: Sequence[str], *, top_k: int | None = None
    ) -> RerankReply:
        route = self._route(role, "rerank")
        return await _run_rerank_async(route, query, documents, top_k, self._slot)


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
    routes = _routes(config, roles, lambda b, t, k: b.connect(t, k, http_client))
    return LiteModel(routes, clients=_Clients(routes, owned=http_client is None))


def build_async(
    config: ModelsConfig, roles: RoleSpec, *, http_client: "httpx2.AsyncClient | None" = None
) -> AsyncLiteModel:
    """`build()` for asyncio callers."""
    routes = _routes(config, roles, lambda b, t, k: b.connect_async(t, k, http_client))
    return AsyncLiteModel(routes, clients=_Clients(routes, owned=http_client is None))
