"""Chat Completions over the OpenAI SDK: Ollama `/v1`, OpenAI and any OpenAI-compatible endpoint
(design §5). What differs between backend types comes from `profiles.py`."""

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any, cast

from pydantic import BaseModel, ValidationError

from lite_model.config import Target
from lite_model.errors import BadStructuredReply, CallFailed, EmptyReply, Truncated
from lite_model.profiles import PROFILES
from lite_model.schema import strict_schema
from lite_model.types import Message, Provenance, StructuredReply, TextReply, Usage

if TYPE_CHECKING:
    import httpx2
    import openai
    from openai.types.chat import ChatCompletion, ChatCompletionMessageParam


def connect(
    target: Target, key: str | None, http_client: "httpx2.Client | None"
) -> "openai.OpenAI":
    import openai

    return openai.OpenAI(
        base_url=target.base_url,
        api_key=key or "unused",  # the SDK insists on one; keyless endpoints ignore it
        timeout=openai.Timeout(target.timeout_s, connect=target.connect_timeout_s),
        max_retries=target.max_retries,
        http_client=http_client,
    )


def _params(target: Target) -> dict[str, Any]:
    params: dict[str, Any] = {}
    if target.max_output_tokens is not None:
        params[PROFILES[target.type].max_tokens_param] = target.max_output_tokens
    if target.temperature is not None:
        params["temperature"] = target.temperature
    if target.reasoning_effort is not None:
        params["reasoning_effort"] = target.reasoning_effort
    return params


def _read(
    target: Target, response: "ChatCompletion", headers: Mapping[str, str]
) -> tuple[Provenance, Usage]:
    """Who answered and what it cost, from the response (design §6, §9). S1 reads the body's
    `model`; backends that name the answering model in a header (§6.2, §6.3) read it here."""
    provenance = Provenance(
        role=target.role,
        backend=target.backend,
        requested=target.model,
        answered=response.model or None,  # from the response, never from config (§6.1)
        channel=target.channel,
        routed=target.routed,
    )
    usage = response.usage
    if usage is None:
        return provenance, Usage()
    details = usage.completion_tokens_details
    return provenance, Usage(
        input=usage.prompt_tokens,
        output=usage.completion_tokens,
        reasoning=details.reasoning_tokens if details is not None else None,
    )


def _send(
    sdk: "openai.OpenAI", target: Target, messages: Sequence[Message], **extra: Any
) -> tuple[str, Provenance, Usage]:
    """One call; the reply's text, who answered, and the usage. Every failure is a ModelError."""
    try:
        raw = sdk.chat.completions.with_raw_response.create(
            model=target.model,
            messages=cast(
                "list[ChatCompletionMessageParam]",
                [{"role": m.role, "content": m.content} for m in messages],
            ),
            **_params(target),
            **extra,
        )
        response = raw.parse()
    except Exception as exc:  # SDK, HTTP and network errors alike; no fallback (§8)
        raise CallFailed(f"{target.where} failed: {type(exc).__name__}") from exc
    provenance, usage = _read(target, response, raw.headers)
    choice = response.choices[0] if response.choices else None
    if choice is not None and choice.finish_reason == "length":
        raise Truncated(f"{target.where}: reply truncated at the output limit ({provenance.tag})")
    text = choice.message.content if choice is not None else None
    if not text or not text.strip():
        raise EmptyReply(f"{target.where}: empty reply ({provenance.tag})")
    return text, provenance, usage


def text(sdk: "openai.OpenAI", target: Target, messages: Sequence[Message]) -> TextReply:
    content, provenance, usage = _send(sdk, target, messages)
    return TextReply(content, provenance, usage)


def structured[T: BaseModel](
    sdk: "openai.OpenAI", target: Target, messages: Sequence[Message], schema: type[T]
) -> StructuredReply[T]:
    response_format = {
        "type": "json_schema",
        "json_schema": {"name": schema.__name__, "schema": strict_schema(schema), "strict": True},
    }
    content, provenance, usage = _send(sdk, target, messages, response_format=response_format)
    try:
        value = schema.model_validate_json(content)
    except ValidationError as exc:
        where = f"{target.where}: reply does not match {schema.__name__} ({provenance.tag})"
        raise BadStructuredReply(where) from exc
    return StructuredReply(value, provenance, usage)
