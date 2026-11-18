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
    requires_max_tokens: bool = False  # the API rejects a request without an output limit
    extra: str | None = None  # the optional dependency (and its module) this type needs
    # The SDK appends /v1/... itself (Anthropic); the OpenAI SDK wants /v1 inside base_url.
    base_url_is_root: bool = False
    # The fields below are read by the OpenAI-compatible protocol only.
    # Where the answering model is named when the body's `model` is not it (design §6.3); a
    # routing alias is allowed only on a backend type that has one (§6.2).
    answered_header: str | None = None
    # Header names are lower case: the keys Provenance.via records (lookups ignore case).
    via_headers: tuple[str, ...] = ()  # provenance headers recorded in Provenance.via
    ignored_params_header: str | None = None  # lists request fields the endpoint dropped (§5)
    request_headers: tuple[tuple[str, str], ...] = ()  # sent with every request
    role_tag: bool = False  # send the role as `metadata.tags` (design §7.3)
    cost_header: str | None = None  # the endpoint's own cost in USD, preferred to prices (§9)


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
        # No default is guessed: hosts set their own (lite-mem 2048, lite-rag 1024) so the
        # request body is unchanged when they migrate (owner, 2026-11-18).
        requires_max_tokens=True,
        extra="anthropic",
        base_url_is_root=True,
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
        # The body's `model` is the profile name, which is what answered as far as the client
        # can tell until llm-gateway passes the upstream model on (design §15).
        via_headers=("x-llmgw-backend", "x-llmgw-structured-output"),
        ignored_params_header="x-llmgw-ignored-params",
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
        # A plain alias answers with body `model` = the alias; the header names the deployment's
        # model (S0, KB/global/litellm §3). Retries stay in lite-model (design §8).
        answered_header="x-litellm-model-name",
        via_headers=("x-litellm-model-id", "x-litellm-model-group"),
        request_headers=(("x-litellm-num-retries", "0"),),
        role_tag=True,
        # Absent for unpriced (Ollama) models; "0" on errors (S0, KB/global/litellm §3).
        cost_header="x-litellm-response-cost",
    ),
}
