"""Hooks around every call (design §10): registered, removable, found by name. lite-cache plugs
in here; lite-model depends on no hook implementation.

`before(call)` runs in registration order before the request is sent; the first hook returning a
reply ends the call there (a cache hit): that reply comes back with `cached=True` and its own,
original provenance. `after(call, reply)` runs only for a reply the endpoint just gave (not for a
hit, not for a failed call). A hook that raises fails the call; nothing is swallowed."""

import hashlib
import inspect
import json
import unicodedata
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from functools import cached_property
from types import MappingProxyType
from typing import Any, Protocol

from pydantic import BaseModel

from lite_model.config import Target
from lite_model.errors import ConfigError
from lite_model.schema import strict_schema
from lite_model.types import Message, Reply

# Request entries that are transport or already in the key (model, messages): not params.
TRANSPORT = frozenset({"model", "messages", "extra_headers"})

_CONTEXT: ContextVar[Mapping[str, Any]] = ContextVar(
    "lite_model_call_context", default=MappingProxyType({})
)


@contextmanager
def call_context(**values: Any) -> Iterator[None]:
    """Values the calls inside this block hand to hooks as `Call.context` (design §10), e.g.
    `call_context(scope="acme/u42", cache=True)` for lite-cache. Nested blocks add to and override
    the outer one; sync calls, threads started with `contextvars.copy_context()` and asyncio tasks
    created inside the block see it. lite-model itself never reads these values."""
    token = _CONTEXT.set(MappingProxyType({**_CONTEXT.get(), **values}))
    try:
        yield
    finally:
        _CONTEXT.reset(token)


@dataclass(frozen=True)
class Call:
    """One call as it is about to be sent: what lite-cache's L1 key is made of."""

    target: Target
    messages: tuple[Message, ...]
    request: Mapping[str, Any] = field(repr=False)  # exactly what is sent (SDK arguments)
    schema: type[BaseModel] | None = None
    # The caller's call_context() at the time of the call (scope, cache policy...).
    context: Mapping[str, Any] = field(default_factory=_CONTEXT.get)

    @property
    def role(self) -> str:
        return self.target.role

    @property
    def params(self) -> dict[str, Any]:
        """The request without model, messages and transport headers."""
        return {k: v for k, v in self.request.items() if k not in TRANSPORT}

    @property
    def schema_name(self) -> str | None:
        return self.schema.__name__ if self.schema is not None else None

    @cached_property
    def schema_hash(self) -> str | None:
        """sha256 of the strict schema as canonical JSON; None for a text call. The same rules as
        lite-eval's `canonical.py` (NFC strings, sorted keys, no whitespace, no NaN, UTF-8),
        copied rather than imported (AGENTS.md rule 9), so a cache key built from it agrees."""
        if self.schema is None:
            return None
        canonical = json.dumps(
            _nfc(strict_schema(self.schema)),
            sort_keys=True,
            ensure_ascii=False,
            separators=(",", ":"),
            allow_nan=False,
        )
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _nfc(value: Any) -> Any:
    if isinstance(value, str):
        return unicodedata.normalize("NFC", value)
    if isinstance(value, dict):
        return {_nfc(k): _nfc(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [_nfc(v) for v in value]
    return value


class Hook(Protocol):
    name: str

    def before(self, call: Call) -> Reply | None: ...

    def after(self, call: Call, reply: Reply) -> None: ...


class AsyncHook(Protocol):
    """A hook whose `before` / `after` are coroutines (0.3.0): only an async model awaits them,
    so a hook that does I/O (a database lookup) does not block the event loop."""

    name: str

    async def before(self, call: Call) -> Reply | None: ...

    async def after(self, call: Call, reply: Reply) -> None: ...


def is_async(hook: Hook | AsyncHook) -> bool:
    return inspect.iscoroutinefunction(hook.before) or inspect.iscoroutinefunction(hook.after)


class Registration:
    def __init__(self, hooks: "Hooks", name: str) -> None:
        self._hooks = hooks
        self.name = name

    def remove(self) -> None:
        """Unregister; later calls no longer see the hook. Removing twice is a no-op."""
        self._hooks._remove(self.name)


class Hooks:
    """The hooks of one built model. A sync model refuses an `AsyncHook`: it could never be
    awaited."""

    def __init__(self, *, allow_async: bool = False) -> None:
        self._registered: dict[str, Hook | AsyncHook] = {}
        self._allow_async = allow_async

    def register(self, hook: Hook | AsyncHook) -> Registration:
        if hook.name in self._registered:
            raise ConfigError(f"a hook named {hook.name!r} is already registered")
        if is_async(hook) and not self._allow_async:
            raise TypeError(
                f"hook {hook.name!r} is async (AsyncHook): register it on a build_async() model"
            )
        self._registered[hook.name] = hook
        return Registration(self, hook.name)

    def __getitem__(self, name: str) -> Hook | AsyncHook:
        """The hook registered under `name`; KeyError when there is none."""
        return self._registered[name]

    def _remove(self, name: str) -> None:
        self._registered.pop(name, None)

    def snapshot(self) -> Sequence[Hook | AsyncHook]:
        """The hooks for one call: a registration change during the call does not affect it."""
        return tuple(self._registered.values())
