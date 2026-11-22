"""Configuration (design §4): defaults plus per-role overrides, named backends, and the checks
`build()` runs before any call (AGENTS.md rule 6).

Hosts embed `ModelsConfig` in their own settings (their env prefix, e.g. `LITERAG_LLM__...`)."""

import importlib.util
import os
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Annotated, Any, ClassVar, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    field_validator,
    model_validator,
)

from lite_model.errors import ConfigError
from lite_model.profiles import PROFILES, BackendType
from lite_model.types import Channel, RoleKind

ReasoningEffort = Literal["none", "minimal", "low", "medium", "high"]
# An empty string is refused, never read as "use the default".
Name = Annotated[str, StringConstraints(min_length=1)]


class BackendConfig(BaseModel):
    """One endpoint. `type` picks the protocol and the parameter profile (profiles.py); a field
    left out takes the profile's value."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    type: BackendType
    base_url: Name | None = None
    api_key_env: Name | None = None  # the variable holding the key; never the key itself
    channel: Channel | None = None
    router_aliases: tuple[Name, ...] = ()  # litellm: the names that are routing aliases (§7.3)
    # The endpoint takes no key, whatever the type's profile says (e.g. Ollama's /v1 used as
    # `type: openai`); the SDK gets a placeholder. 0.1.1, for lite-mem.
    keyless: bool = False

    @model_validator(mode="after")
    def _key_or_keyless(self) -> "BackendConfig":
        if self.keyless and self.api_key_env is not None:
            raise ValueError("keyless and api_key_env cannot both be set")
        return self


# Under a host's own (`host_backends`) and a config's backends (design §4); more go under new names.
DEFAULT_BACKENDS: dict[str, BackendConfig] = {
    "ollama": BackendConfig(type="ollama"),
    "openai": BackendConfig(type="openai"),
    "anthropic": BackendConfig(type="anthropic"),
    "gateway": BackendConfig(type="gateway"),
    "litellm": BackendConfig(type="litellm", router_aliases=("smart-router",)),
    "voyage": BackendConfig(type="voyage"),
    "cohere": BackendConfig(type="cohere"),
    "gemini": BackendConfig(type="gemini"),
}


class Price(BaseModel):
    """USD per million tokens."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    input_per_mtok: float = Field(ge=0, allow_inf_nan=False)
    output_per_mtok: float = Field(ge=0, allow_inf_nan=False)


class _CallSettings(BaseModel):
    """The call parameters a role may override."""

    model_config = ConfigDict(extra="forbid")

    temperature: float | None = Field(default=None, ge=0, le=2)  # None: the profile's default
    max_output_tokens: int | None = Field(default=None, gt=0)  # None: not sent
    reasoning_effort: ReasoningEffort | None = None  # None: not sent
    timeout_s: float | None = Field(default=None, gt=0)
    connect_timeout_s: float | None = Field(default=None, gt=0)
    max_retries: int | None = Field(default=None, ge=0)  # the same target only (§8)


class RoleConfig(_CallSettings):
    """What one role uses instead of the defaults; a field left out keeps the default."""

    backend: Name | None = None
    model: Name | None = None
    # An embed role's own settings (design §3.2, 0.6.0); refused on a chat role.
    dimensions: int | None = Field(default=None, gt=0)  # sent as `dimensions` (truncation)
    batch_size: int | None = Field(default=None, gt=0)  # texts per request; None: EMBED_BATCH
    max_concurrency: int | None = Field(default=None, gt=0)  # batches in flight; None: 1


class ModelsConfig(_CallSettings):
    """The defaults, per-role overrides, and the named backends."""

    # No defaults: a host names its model, so a lite-model release never changes it (owner,
    # 2026-11-18).
    backend: Name
    model: Name
    timeout_s: float = Field(default=120.0, gt=0)
    connect_timeout_s: float = Field(default=10.0, gt=0)
    max_retries: int = Field(default=1, ge=0)
    # A host's own defaults, set on its subclass (0.2.0): backends over the built-ins, and call
    # settings per role; a config's entries are merged on top (`_layered`, design §4).
    host_backends: ClassVar[Mapping[str, BackendConfig]] = {}
    host_roles: ClassVar[Mapping[str, RoleConfig]] = {}
    roles: dict[str, RoleConfig] = Field(default_factory=dict, validate_default=True)
    backends: dict[str, BackendConfig] = Field(default_factory=dict, validate_default=True)
    forbid_channels: tuple[Channel, ...] = ()  # tests and CI set ("subscription", "api")
    # "backend:model" (as Provenance.tag names the answering model) -> price; used when the
    # endpoint does not report the cost itself (design §9).
    prices: dict[str, Price] = Field(default_factory=dict)

    @field_validator("backends", mode="before")
    @classmethod
    def _over_defaults(cls, value: Any) -> Any:
        return _layered({**DEFAULT_BACKENDS, **cls.host_backends}, value)

    @field_validator("roles", mode="before")
    @classmethod
    def _over_host_roles(cls, value: Any) -> Any:
        return _layered(cls.host_roles, value)


def _layered(defaults: Mapping[str, BaseModel], value: Any) -> Any:
    """`value`'s entries over `defaults` (design §4): an entry changes its default field by field
    (a role's null removes the host's value: the role takes the top-level one), unless it names
    another backend `type`, which starts afresh. Where a backend's key comes from (`api_key_env`
    or `keyless`) is one choice: an entry that sets either replaces the default's."""
    if not isinstance(value, Mapping):
        return value  # pydantic reports it
    merged: dict[str, Any] = dict(defaults)
    for name, given in value.items():
        own = given.model_dump(exclude_unset=True) if isinstance(given, BaseModel) else given
        base = defaults.get(name)
        if base is not None and isinstance(own, Mapping) and own.get("type", _NO) in (
            _NO, getattr(base, "type", _NO)
        ):  # fmt: skip
            inherited = base.model_dump(exclude_unset=True)
            if _KEY_SOURCE & {k for k, v in own.items() if v}:  # where the key comes from is one
                inherited = {k: v for k, v in inherited.items() if k not in _KEY_SOURCE}  # choice
            own = {**inherited, **own}
        merged[name] = own
    return merged


_NO = object()  # no `type` given (or a role, which has none)
_KEY_SOURCE = {"api_key_env", "keyless"}  # an entry that sets one replaces both (0.2.1)


# The settings only a chat role uses (design §3.2), and the ones only an embed role has (0.6.0).
CHAT_ONLY = ("temperature", "max_output_tokens", "reasoning_effort")
EMBED_ONLY = ("dimensions", "batch_size", "max_concurrency")
# Per kind: the settings it has no use for, refused when the role sets them itself (and dropped
# when inherited, see `target`), and what the kind is called in messages.
UNUSED: dict[RoleKind, tuple[str, ...]] = {
    "chat": EMBED_ONLY,
    "embed": CHAT_ONLY,
    "rerank": CHAT_ONLY + EMBED_ONLY,
}
NOUN: dict[RoleKind, str] = {"chat": "chat", "embed": "embeddings", "rerank": "rerank"}
EMBED_BATCH = 64  # texts per `/embeddings` request unless the role sets `batch_size`


@dataclass(frozen=True)
class RolePolicy:
    # May this role use a routing alias? Never for anything that is part of an index's identity,
    # nor for a judge (AGENTS.md rule 5).
    routable: bool = False
    # "chat": text() / structured() / turn(); "embed": embed() (design §3.2, 0.5.0).
    kind: RoleKind = "chat"

    def __post_init__(self) -> None:
        if self.kind != "chat" and self.routable:  # embeddings are index identity (rule 5);
            # a reranker decides what a host keeps: recorded, never swapped by a router
            raise ValueError(f"an {self.kind} role is never routable")


@dataclass(frozen=True)
class RoleSpec:
    """The roles a host declares, e.g. `RoleSpec({"answer": RolePolicy(routable=True)})`."""

    roles: Mapping[str, RolePolicy]


@dataclass(frozen=True)
class Target:
    """Where one role's calls go, resolved once at `build()` (design §4, §6.1)."""

    role: str
    backend: str  # the backend's name
    type: BackendType
    kind: RoleKind
    model: str
    base_url: str | None
    api_key_env: str | None
    channel: Channel
    routed: bool
    temperature: float | None
    max_output_tokens: int | None
    reasoning_effort: ReasoningEffort | None
    timeout_s: float
    connect_timeout_s: float
    max_retries: int
    prices: Mapping[str, Price]  # this backend's entries of ModelsConfig.prices, by model
    # embed roles only (0.6.0); a chat role has the defaults
    dimensions: int | None = None
    batch_size: int = EMBED_BATCH
    max_concurrency: int = 1

    @property
    def where(self) -> str:
        """How errors name this target: role and `backend:model`, nothing else (rule 8)."""
        return f"role {self.role} ({self.backend}:{self.model})"


def target(config: ModelsConfig, role: str, kind: RoleKind = "chat") -> Target:
    """`role`'s target from the defaults and its override; no checks beyond resolving. An embed
    role's request carries none of the chat settings, inherited or not (design §3.2)."""
    own = config.roles.get(role, RoleConfig())
    override = own.model_dump(exclude_none=True, exclude=set(EMBED_ONLY))
    if kind != "chat":  # what the role sets itself is refused in _problems
        override |= dict.fromkeys(CHAT_ONLY)
    c = config.model_copy(update=override)  # a field the role leaves out keeps the default
    if c.backend not in c.backends:
        raise ConfigError(f"role {role}: unknown backend {c.backend!r}")
    backend = c.backends[c.backend]
    profile = PROFILES[backend.type]
    return Target(
        role=role,
        backend=c.backend,
        type=backend.type,
        kind=kind,
        model=c.model,
        base_url=backend.base_url or profile.base_url,
        api_key_env=None if backend.keyless else backend.api_key_env or profile.key_env,
        channel=backend.channel or profile.channel,
        routed=c.model in backend.router_aliases,
        temperature=(
            c.temperature
            if c.temperature is not None or kind != "chat"
            else profile.default_temperature
        ),
        max_output_tokens=c.max_output_tokens,
        reasoning_effort=c.reasoning_effort,
        timeout_s=c.timeout_s,
        connect_timeout_s=c.connect_timeout_s,
        max_retries=c.max_retries,
        prices={
            k.split(":", 1)[1]: v for k, v in c.prices.items() if k.startswith(f"{c.backend}:")
        },
        # a chat role that sets these is refused in _problems
        dimensions=own.dimensions,
        batch_size=own.batch_size or EMBED_BATCH,
        max_concurrency=own.max_concurrency or 1,
    )


def _set(config: ModelsConfig, target: Target, names: tuple[str, ...]) -> list[str]:
    """Which of `names` the role sets itself (not inherited)."""
    own = config.roles.get(target.role, RoleConfig())
    return [name for name in names if getattr(own, name, None) is not None]


def _problems(config: ModelsConfig, spec: RoleSpec, target: Target) -> list[str]:
    where = target.where
    profile = PROFILES[target.type]
    found = []

    if target.api_key_env is not None and not os.environ.get(target.api_key_env):
        found.append(f"{where}: set {target.api_key_env}")
    if profile.extra and importlib.util.find_spec(profile.extra) is None:
        found.append(f"{where}: install the extra: lite-model[{profile.extra}]")
    if target.channel in config.forbid_channels:
        found.append(f"{where}: channel {target.channel} is forbidden")
    if target.routed and not spec.roles[target.role].routable:  # an embed role never is
        found.append(f"{where}: {target.model} is a routing alias and the role is not routable")
    kind = target.kind
    if kind not in profile.serves:
        found.append(f"{where}: {target.type} does not serve {NOUN[kind]}")
    unused = "is only for embed roles" if kind == "chat" else f"is not used by {NOUN[kind]}"
    found += [f"{where}: {name} {unused}" for name in _set(config, target, UNUSED[kind])]
    if kind != "chat":  # the checks below are about chat parameters
        return found
    if target.routed and profile.answered_header is None:
        found.append(f"{where}: routing through {target.type} cannot tell which model answered")
    if target.temperature is not None and not profile.supports_temperature:
        found.append(f"{where}: temperature is not supported by {target.type}")
    if target.reasoning_effort is not None and not profile.supports_reasoning_effort:
        found.append(f"{where}: reasoning_effort is not supported by {target.type}")
    if target.max_output_tokens is None and profile.requires_max_tokens:
        found.append(f"{where}: {target.type} needs max_output_tokens")
    if profile.base_url_is_root and (target.base_url or "").rstrip("/").endswith("/v1"):
        found.append(f"{where}: {target.type} base_url must not end in /v1 (the SDK adds it)")
    return found


def resolve(config: ModelsConfig, spec: RoleSpec) -> dict[str, Target]:
    """Every declared role's target, or one ConfigError listing every problem (design §4)."""
    problems = [f"role {r}: not declared by the host" for r in config.roles if r not in spec.roles]
    problems += [
        f"prices: {key!r} is not 'backend:model' with a configured backend"
        for key in config.prices
        if key.split(":", 1)[0] not in config.backends or ":" not in key
    ]
    targets: dict[str, Target] = {}
    for role in spec.roles:
        try:
            resolved = target(config, role, spec.roles[role].kind)
        except ConfigError as exc:
            problems.append(str(exc))
            continue
        problems += _problems(config, spec, resolved)
        targets[role] = resolved
    if problems:
        raise ConfigError("invalid model config:\n- " + "\n- ".join(problems))
    return targets
