"""Chat Completions over the OpenAI SDK: Ollama `/v1`, OpenAI and any OpenAI-compatible endpoint
(design §5). What differs between backend types comes from `profiles.py`."""

import json
from collections.abc import Mapping, Sequence
from typing import Any

from pydantic import BaseModel

from lite_model.backends import finish, finish_turn, sdk_client
from lite_model.config import Target
from lite_model.profiles import PROFILES
from lite_model.schema import strict_schema
from lite_model.types import (
    AssistantTurn,
    Block,
    Message,
    Reply,
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
