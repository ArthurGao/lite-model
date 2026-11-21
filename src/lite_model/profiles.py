"""What differs between backend types, as data (design §5, AGENTS.md rule 7). Values verified in
S0 (2026-11-18) unless noted; `KB/global/litellm/README.md` holds the LiteLLM evidence."""

from dataclasses import dataclass
from typing import Literal

from lite_model.types import Channel, RoleKind

BackendType = Literal["ollama", "openai", "anthropic", "gateway", "litellm", "voyage"]


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
    # Present and non-empty only when the endpoint answered from its own cache (§7.2, §9).
    cached_header: str | None = None
    # `turn()` is refused on a type whose endpoint cannot take `tools` (design §5). Whether a
    # given model uses them well is the model's affair, not the type's.
    supports_tools: bool = True
    # The role kinds the type serves. `embed()` goes to an OpenAI-compatible `/embeddings`
    # (design §3.2): llm-gateway serves none (its non-goal), Anthropic has no embedding API.
    # `rerank()` goes to `/rerank`: Voyage's API only; Voyage has no chat (§3.3).
    serves: frozenset[RoleKind] = frozenset({"chat", "embed"})
    # How `/embeddings` takes an embed role's `dimensions`, whether it takes an input type
    # (`query` / `document`), and the `encoding_format` sent (None: left to the SDK, which asks
    # for base64 and decodes it) (§3.3).
    embed_dimensions_field: str = "dimensions"
    embed_input_type: bool = False
    embed_encoding_format: str | None = "float"


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
        serves=frozenset({"chat"}),
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
        serves=frozenset({"chat"}),
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
        # Set only on a cache hit: LiteLLM 1.108.9 writes `cache_key` into a response's hidden
        # params in its cache-hit branches only, and drops empty header values (0.3.0, KB
        # global/litellm). The Proxy's cache stays off (§7.2); this keeps the usage honest if on.
        cached_header="x-litellm-cache-key",
    ),
    # Voyage AI (Anthropic's recommended embedding vendor): an OpenAI-shaped `/embeddings` with
    # `input_type` and `output_dimension`, which takes `encoding_format` null or base64 only, and
    # `/rerank`. No chat API. API reference 2025-12-10 (design §3.3).
    "voyage": BackendProfile(
        protocol="openai",
        channel="api",
        base_url="https://api.voyageai.com/v1",
        key_env="VOYAGE_API_KEY",
        max_tokens_param="max_tokens",
        default_temperature=None,
        supports_temperature=False,
        supports_reasoning_effort=False,
        supports_tools=False,
        serves=frozenset({"embed", "rerank"}),
        embed_dimensions_field="output_dimension",
        embed_input_type=True,
        embed_encoding_format=None,
    ),
}
