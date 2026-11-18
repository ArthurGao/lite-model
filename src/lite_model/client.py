"""`build()` (design §3): validate the config, resolve each role to a target, dispatch calls."""

import os
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, Protocol

from pydantic import BaseModel

from lite_model.backends import openai_compat
from lite_model.config import ModelsConfig, RoleSpec, Target, resolve
from lite_model.errors import ConfigError
from lite_model.types import Message, StructuredReply, TextReply

if TYPE_CHECKING:
    import httpx2


class Model(Protocol):
    def text(self, role: str, messages: Sequence[Message]) -> TextReply: ...

    def structured[T: BaseModel](
        self, role: str, messages: Sequence[Message], schema: type[T]
    ) -> StructuredReply[T]: ...


class _Model:
    def __init__(self, routes: dict[str, tuple[Any, Target]]) -> None:
        self._routes = routes  # role -> (SDK client, target); one client per endpoint

    def _call(self, role: str) -> tuple[Any, Target]:
        if role not in self._routes:
            raise ConfigError(f"role {role}: not declared by the host")
        return self._routes[role]

    def text(self, role: str, messages: Sequence[Message]) -> TextReply:
        sdk, target = self._call(role)
        return openai_compat.text(sdk, target, messages)

    def structured[T: BaseModel](
        self, role: str, messages: Sequence[Message], schema: type[T]
    ) -> StructuredReply[T]:
        sdk, target = self._call(role)
        return openai_compat.structured(sdk, target, messages, schema)


def build(
    config: ModelsConfig, roles: RoleSpec, *, http_client: "httpx2.Client | None" = None
) -> Model:
    """The model for `roles`, or ConfigError before any call (design §4).

    `http_client` replaces the SDK's transport (a proxy, or a test's recording transport)."""
    shared: dict[tuple[Any, ...], Any] = {}
    routes = {}
    for role, t in resolve(config, roles).items():
        slot = (t.base_url, t.api_key_env, t.timeout_s, t.connect_timeout_s, t.max_retries)
        if slot not in shared:
            key = os.environ.get(t.api_key_env) if t.api_key_env else None
            shared[slot] = openai_compat.connect(t, key, http_client)
        routes[role] = (shared[slot], t)
    return _Model(routes)
