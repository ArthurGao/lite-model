"""What differs between backend types, as data (design §5, AGENTS.md rule 7). Values verified in
S0 (2026-11-18) unless noted; `KB/global/litellm/README.md` holds the LiteLLM evidence."""

from dataclasses import dataclass
from typing import Literal

from lite_model.types import Channel

BackendType = Literal["ollama", "openai", "anthropic", "gateway", "litellm"]


@dataclass(frozen=True)
class BackendProfile:
    protocol: Literal["openai", "anthropic"]  # which SDK speaks to it
    channel: Channel  # the default; a configured backend may override it
    base_url: str | None  # None: the SDK's own
    key_env: str | None  # the usual key variable; None: no key needed
    max_tokens_param: Literal["max_tokens", "max_completion_tokens"]
    default_temperature: float | None  # sent when none is configured; None: not sent
    supports_temperature: bool
    supports_reasoning_effort: bool  # sent as the top-level `reasoning_effort`


PROFILES: dict[BackendType, BackendProfile] = {
    # Ollama ignores max_completion_tokens (lite-rag, 2026-11-08) and needs no key.
    "ollama": BackendProfile(
        protocol="openai",
        channel="local",
        base_url="http://127.0.0.1:11434/v1",
        key_env=None,
        max_tokens_param="max_tokens",
        default_temperature=0.0,
        supports_temperature=True,
        supports_reasoning_effort=True,
    ),
    # Newer OpenAI models reject max_tokens, reasoning models a non-default temperature.
    "openai": BackendProfile(
        protocol="openai",
        channel="api",
        base_url=None,
        key_env="OPENAI_API_KEY",
        max_tokens_param="max_completion_tokens",
        default_temperature=None,
        supports_temperature=True,
        supports_reasoning_effort=True,
    ),
    # anthropic 2.6.0's Messages API has no temperature (lite-rag, 2026-11-08).
    "anthropic": BackendProfile(
        protocol="anthropic",
        channel="api",
        base_url=None,
        key_env="ANTHROPIC_API_KEY",
        max_tokens_param="max_tokens",
        default_temperature=None,
        supports_temperature=False,
        supports_reasoning_effort=False,
    ),
    # llm-gateway maps both names to max_tokens (translate.py:58-59) and drops reasoning_effort
    # (it lands in X-LLMGW-Ignored-Params): refused rather than silently lost (design §5).
    "gateway": BackendProfile(
        protocol="openai",
        channel="subscription",
        base_url="http://127.0.0.1:8787/v1",
        key_env=None,
        max_tokens_param="max_completion_tokens",
        default_temperature=None,
        supports_temperature=True,
        supports_reasoning_effort=False,
    ),
    # LiteLLM 1.108.9 honours both names and translates per upstream (S0).
    "litellm": BackendProfile(
        protocol="openai",
        channel="api",
        base_url="http://127.0.0.1:4400/v1",
        key_env="LITELLM_API_KEY",
        max_tokens_param="max_completion_tokens",
        default_temperature=None,
        supports_temperature=True,
        supports_reasoning_effort=True,
    ),
}
