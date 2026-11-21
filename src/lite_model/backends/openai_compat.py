"""Chat Completions over the OpenAI SDK: Ollama `/v1`, OpenAI and any OpenAI-compatible endpoint
(design §5). What differs between backend types comes from `profiles.py`."""

import inspect
import json
from collections.abc import Mapping, Sequence
from typing import Any

from pydantic import BaseModel

from lite_model.backends import finish, finish_embed, finish_rerank, finish_turn, sdk_client
from lite_model.config import Target
from lite_model.profiles import PROFILES
from lite_model.schema import strict_schema
from lite_model.types import (
    AssistantTurn,
    Block,
    EmbedReply,
    InputType,
    Message,
    Reply,
    RerankReply,
    Text,
    ToolCall,
    ToolSpec,
    TurnMessage,
    TurnReply,
    Usage,
)

# finish_reason values that leave no usable answer, whatever the content holds.
UNUSABLE = {"content_filter"}
# For turn(): the only values a finished or cut-off turn may carry (openai 3.46.0; None from
# endpoints that leave it out). Any other one is unusable, and never named in an error (rule 8).
TURN_STOPS = {"stop", "tool_calls", "function_call", "length", None}


def _turn_unusable(finish_reason: str | None) -> str | None:
    if finish_reason in TURN_STOPS:
        return None
    return finish_reason if finish_reason in UNUSABLE else "unexpected"


def connect(target: Target, key: str | None, http_client: Any) -> Any:
    import openai

    # The SDK insists on a key; keyless endpoints (Ollama) ignore it.
    return sdk_client(openai.OpenAI, target, key or "unused", http_client)


def connect_async(target: Target, key: str | None, http_client: Any) -> Any:
    import openai

    return sdk_client(openai.AsyncOpenAI, target, key or "unused", http_client)


def endpoint(sdk: Any) -> Any:
    return sdk.chat.completions.with_raw_response


def _params(target: Target, messages: Sequence[TurnMessage]) -> dict[str, Any]:
    """The request without what `request()` and `turn_request()` each add."""
    profile = PROFILES[target.type]
    params: dict[str, Any] = {"model": target.model, "messages": _history(messages)}
    if target.max_output_tokens is not None:
        params[profile.max_tokens_param] = target.max_output_tokens
    if target.temperature is not None:
        params["temperature"] = target.temperature
    if target.reasoning_effort is not None:
        params["reasoning_effort"] = target.reasoning_effort
    return params


def _transport(target: Target, params: dict[str, Any]) -> dict[str, Any]:
    profile = PROFILES[target.type]
    if profile.request_headers:
        params["extra_headers"] = dict(profile.request_headers)
    if profile.role_tag:  # not the SDK's typed `metadata` (str values only): sent as is
        params["extra_body"] = {"metadata": {"tags": [f"role:{target.role}"]}}
    return params


def request(
    target: Target, messages: Sequence[Message], schema: type[BaseModel] | None
) -> dict[str, Any]:
    params = _params(target, messages)
    if schema is not None:
        strict = {"name": schema.__name__, "schema": strict_schema(schema), "strict": True}
        params["response_format"] = {"type": "json_schema", "json_schema": strict}
    return _transport(target, params)


def _assistant(turn: AssistantTurn) -> dict[str, Any]:
    """Text and calls; opaque content has no place in Chat Completions and is left out."""
    sent: dict[str, Any] = {"role": "assistant", "content": turn.text or None}
    if turn.tool_calls:
        sent["tool_calls"] = [
            {
                "id": c.id,
                "type": "function",
                "function": {
                    "name": c.name,
                    # as received when it never parsed; otherwise the parsed input again
                    "arguments": c.raw_arguments
                    if c.arguments is None
                    else json.dumps(c.arguments, ensure_ascii=False),
                },
            }
            for c in turn.tool_calls
        ]
    return sent


def _history(messages: Sequence[TurnMessage]) -> list[dict[str, Any]]:
    sent: list[dict[str, Any]] = []
    for m in messages:
        if isinstance(m, Message):
            sent.append({"role": m.role, "content": m.content})
        elif isinstance(m, AssistantTurn):
            sent.append(_assistant(m))
        else:  # one `tool` message per result; the protocol has no error flag
            sent += [
                {"role": "tool", "tool_call_id": r.call_id, "content": r.content} for r in m.results
            ]
    return sent


def turn_request(
    target: Target, messages: Sequence[TurnMessage], tools: Sequence[ToolSpec]
) -> dict[str, Any]:
    params = _params(target, messages)
    if tools:  # `tool_choice` is left at the endpoint's default (auto)
        params["tools"] = [
            {
                "type": "function",
                "function": {
                    "name": t.name,
                    "description": t.description,
                    "parameters": dict(t.parameters),
                },
            }
            for t in tools
        ]
    return _transport(target, params)


def _usage(response: Any) -> Usage:
    if response.usage is None:
        return Usage()
    details = response.usage.completion_tokens_details
    return Usage(
        input=response.usage.prompt_tokens,
        output=response.usage.completion_tokens,
        reasoning=details.reasoning_tokens if details is not None else None,
    )


def _call(raw: Any) -> ToolCall:
    text = raw.function.arguments
    try:
        parsed = json.loads(text)
    except ValueError:
        parsed = None
    if isinstance(parsed, dict):
        return ToolCall(raw.id, raw.function.name, parsed)
    return ToolCall(raw.id, raw.function.name, None, text)  # not an object: left to the caller


def turn_reply(
    target: Target, response: Any, headers: Mapping[str, str], sent: Mapping[str, Any]
) -> TurnReply:
    choice = response.choices[0] if response.choices else None
    blocks: list[Block] = []
    if choice is not None:
        if choice.message.content:
            blocks.append(Text(choice.message.content))
        # only function calls: lite-model offers no other kind of tool
        calls = choice.message.tool_calls or []
        blocks += [_call(c) for c in calls if c.type == "function"]
    return finish_turn(
        target,
        sent,
        headers,
        body_model=response.model,
        usage=_usage(response),
        truncated=choice is not None and choice.finish_reason == "length",
        unusable=_turn_unusable(choice.finish_reason) if choice is not None else None,
        blocks=blocks,
    )


def reply(
    target: Target,
    response: Any,
    headers: Mapping[str, str],
    schema: type[BaseModel] | None,
    sent: Mapping[str, Any],
) -> Reply:
    choice = response.choices[0] if response.choices else None
    return finish(
        target,
        schema,
        sent,
        headers,
        body_model=response.model,
        usage=_usage(response),
        truncated=choice is not None and choice.finish_reason == "length",
        unusable=choice.finish_reason if choice and choice.finish_reason in UNUSABLE else None,
        payload=choice.message.content if choice is not None else None,
    )


# --- embed() (design §3.2): `/embeddings`, the same SDK client ---


def embed_endpoint(sdk: Any) -> Any:
    return sdk.embeddings.with_raw_response


def embed_request(
    target: Target, texts: list[str], input_type: InputType | None = None
) -> dict[str, Any]:
    profile = PROFILES[target.type]
    params: dict[str, Any] = {"model": target.model, "input": texts}
    # "float" spelled out: left out, openai 3.46.0 asks for base64 and decodes it to float32
    # (openai/lib/_parsing/_embeddings.py); whether Ollama takes base64 is unverified. Voyage
    # takes null or base64 only: left to the SDK there (§3.3).
    if profile.embed_encoding_format is not None:
        params["encoding_format"] = profile.embed_encoding_format
    vendor: dict[str, Any] = {}  # fields the OpenAI SDK has no parameter for
    if target.dimensions is not None:
        if profile.embed_dimensions_field == "dimensions":
            params["dimensions"] = target.dimensions
        else:
            vendor[profile.embed_dimensions_field] = target.dimensions
    if input_type is not None:
        vendor["input_type"] = input_type  # checked against the profile by the client
    params = _transport(target, params)
    if vendor:
        params["extra_body"] = {**params.get("extra_body", {}), **vendor}
    return params


def embed_reply(
    target: Target, response: Any, headers: Mapping[str, str], sent: Mapping[str, Any]
) -> EmbedReply:
    data = sorted(response.data or [], key=lambda d: d.index)
    usage = getattr(response, "usage", None)
    # OpenAI reports prompt_tokens; Voyage total_tokens (an embedding has no output)
    tokens = getattr(usage, "prompt_tokens", None) or getattr(usage, "total_tokens", None)
    return finish_embed(
        target,
        sent,
        headers,
        body_model=response.model,
        usage=Usage(input=tokens, output=0),  # an embedding has no output tokens
        vectors=[d.embedding for d in data],  # the SDK's list[float], not copied
        indexes=[d.index for d in data],
    )


# --- rerank() (design §3.3): `/rerank` (Voyage), through the same SDK client ---


class _Raw:
    """An httpx response in the raw-response shape `_send` reads: `.headers`, `.parse()`."""

    def __init__(self, response: Any) -> None:
        self.headers = response.headers
        self._response = response

    def parse(self) -> Any:
        return self._response.json()


class _RerankOwner:
    """`/rerank` as the `create(**request)` owner `_send` / `_send_async` take: the SDK's generic
    `post`, with its retries, timeout and auth. On an async SDK `post` returns an awaitable, and
    so does `create`."""

    def __init__(self, sdk: Any) -> None:
        self._sdk = sdk

    def create(self, **request: Any) -> Any:
        import httpx2

        sent = self._sdk.post("/rerank", body=request, cast_to=httpx2.Response)
        if not inspect.isawaitable(sent):
            return _Raw(sent)

        async def answered() -> _Raw:
            return _Raw(await sent)

        return answered()


def rerank_endpoint(sdk: Any) -> Any:
    return _RerankOwner(sdk)


def rerank_request(
    target: Target, query: str, documents: list[str], top_k: int | None
) -> dict[str, Any]:
    params: dict[str, Any] = {
        "query": query,
        "documents": documents,
        "model": target.model,
        "return_documents": False,  # the host has them; the reply stays small
    }
    if top_k is not None:
        params["top_k"] = top_k
    return _transport(target, params)


def rerank_reply(
    target: Target, response: Any, headers: Mapping[str, str], sent: Mapping[str, Any]
) -> RerankReply:
    body = response if isinstance(response, dict) else {}
    data = body.get("data") or []
    # a malformed entry stays in the count, as (None, None): finish_rerank refuses it
    items = [
        (d.get("index"), d.get("relevance_score")) if isinstance(d, dict) else (None, None)
        for d in data
    ]
    usage = body.get("usage")
    tokens = usage.get("total_tokens") if isinstance(usage, dict) else None
    return finish_rerank(
        target,
        sent,
        headers,
        body_model=body.get("model"),
        usage=Usage(input=tokens if isinstance(tokens, int) else None, output=0),
        scored=items,
    )
