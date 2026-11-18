"""Configuration (design §4): defaults plus per-role overrides, named backends, and the checks
`build()` runs before any call (AGENTS.md rule 6).

Hosts embed `ModelsConfig` in their own settings (their env prefix, e.g. `LITERAG_LLM__...`)."""

import importlib.util
import os
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, field_validator

from lite_model.errors import ConfigError
from lite_model.profiles import PROFILES, BackendType
from lite_model.types import Channel

ReasoningEffort = Literal["none", "minimal", "low", "medium", "high"]
# An empty string is refused, never read as "use the default" (Codex S1).
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


# Present unless the config replaces them by name; a config adds more under new names.
DEFAULT_BACKENDS: dict[str, BackendConfig] = {
    "ollama": BackendConfig(type="ollama"),
    "openai": BackendConfig(type="openai"),
    "anthropic": BackendConfig(type="anthropic"),
    "gateway": BackendConfig(type="gateway"),
    "litellm": BackendConfig(type="litellm", router_aliases=("smart-router",)),
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


class ModelsConfig(_CallSettings):
    """The defaults, per-role overrides, and the named backends."""

    backend: Name = "ollama"
    model: Name = "qwen3:8b"
    timeout_s: float = Field(default=120.0, gt=0)
    connect_timeout_s: float = Field(default=10.0, gt=0)
    max_retries: int = Field(default=1, ge=0)
    roles: dict[str, RoleConfig] = Field(default_factory=dict)
    backends: dict[str, BackendConfig] = Field(default_factory=lambda: dict(DEFAULT_BACKENDS))
    forbid_channels: tuple[Channel, ...] = ()  # tests and CI set ("subscription", "api")
    # "backend:model" (as Provenance.tag names the answering model) -> price; used when the
    # endpoint does not report the cost itself (design §9).
    prices: dict[str, Price] = Field(default_factory=dict)

    @field_validator("backends")
    @classmethod
    def _over_defaults(cls, value: dict[str, BackendConfig]) -> dict[str, BackendConfig]:
        return {**DEFAULT_BACKENDS, **value}


@dataclass(frozen=True)
class RolePolicy:
    # May this role use a routing alias? Never for anything that is part of an index's identity,
    # nor for a judge (AGENTS.md rule 5).
    routable: bool = False


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

    @property
    def where(self) -> str:
        """How errors name this target: role and `backend:model`, nothing else (rule 8)."""
        return f"role {self.role} ({self.backend}:{self.model})"


def target(config: ModelsConfig, role: str) -> Target:
    """`role`'s target from the defaults and its override; no checks beyond resolving."""
    override = config.roles.get(role, RoleConfig()).model_dump(exclude_none=True)
    c = config.model_copy(update=override)  # a field the role leaves out keeps the default
    if c.backend not in c.backends:
        raise ConfigError(f"role {role}: unknown backend {c.backend!r}")
    backend = c.backends[c.backend]
    profile = PROFILES[backend.type]
    return Target(
        role=role,
        backend=c.backend,
        type=backend.type,
        model=c.model,
        base_url=backend.base_url or profile.base_url,
        api_key_env=backend.api_key_env or profile.key_env,
        channel=backend.channel or profile.channel,
        routed=c.model in backend.router_aliases,
        temperature=c.temperature if c.temperature is not None else profile.default_temperature,
        max_output_tokens=c.max_output_tokens,
        reasoning_effort=c.reasoning_effort,
        timeout_s=c.timeout_s,
        connect_timeout_s=c.connect_timeout_s,
        max_retries=c.max_retries,
        prices={
            k.split(":", 1)[1]: v for k, v in c.prices.items() if k.startswith(f"{c.backend}:")
        },
    )


def _problems(config: ModelsConfig, spec: RoleSpec, target: Target) -> list[str]:
    where = target.where
    profile = PROFILES[target.type]
    found = []
    if target.api_key_env is not None and not os.environ.get(target.api_key_env):
        found.append(f"{where}: set {target.api_key_env}")
    if profile.protocol == "anthropic" and importlib.util.find_spec("anthropic") is None:
        found.append(f"{where}: install the extra: lite-model[anthropic]")
    if target.routed and not spec.roles[target.role].routable:
        found.append(f"{where}: {target.model} is a routing alias and the role is not routable")
    if target.routed and profile.answered_header is None:
        found.append(f"{where}: routing through {target.type} cannot tell which model answered")
    if target.channel in config.forbid_channels:
        found.append(f"{where}: channel {target.channel} is forbidden")
    if target.temperature is not None and not profile.supports_temperature:
        found.append(f"{where}: temperature is not supported by {target.type}")
    if target.reasoning_effort is not None and not profile.supports_reasoning_effort:
        found.append(f"{where}: reasoning_effort is not supported by {target.type}")
    if target.max_output_tokens is None and profile.requires_max_tokens:
        found.append(f"{where}: {target.type} needs max_output_tokens")
    if profile.protocol == "anthropic" and (target.base_url or "").rstrip("/").endswith("/v1"):
        # The Anthropic SDK appends /v1/messages; the OpenAI SDK wants /v1 in base_url.
        found.append(f"{where}: anthropic base_url must not end in /v1 (the SDK adds it)")
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
            resolved = target(config, role)
        except ConfigError as exc:
            problems.append(str(exc))
            continue
        problems += _problems(config, spec, resolved)
        targets[role] = resolved
    if problems:
        raise ConfigError("invalid model config:\n- " + "\n- ".join(problems))
    return targets
