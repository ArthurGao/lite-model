"""Anthropic Messages over the Anthropic SDK (the `anthropic` extra; design §5). The request is
built as lite-mem and lite-rag build theirs today, so their bodies are unchanged when they migrate:
system messages joined into the `system` field, structured output as one forced tool."""

from collections.abc import Mapping, Sequence
from typing import Any

from pydantic import BaseModel

from lite_model.backends import finish, finish_turn, sdk_client
from lite_model.config import Target
from lite_model.schema import strict_schema
from lite_model.types import (
    AssistantTurn,
    Block,
    Message,
    Opaque,
    Reply,
    Text,
    ToolCall,
    ToolSpec,
    TurnMessage,
    TurnReply,
    Usage,
)

# stop_reason values (anthropic 2.6.0): cut short, or complete. Any other one leaves no usable
# answer; only the known ones are named in the error, any other is "unexpected" (an endpoint's
# own string never reaches an error message, rule 8).
TRUNCATED = {"max_tokens", "model_context_window_exceeded"}
COMPLETE = {"end_turn", "stop_sequence", "tool_use", None}
UNUSABLE = {"refusal", "pause_turn"}


def connect(target: Target, key: str | None, http_client: Any) -> Any:
    import anthropic

    return sdk_client(anthropic.Anthropic, target, key, http_client)


def connect_async(target: Target, key: str | None, http_client: Any) -> Any:
    import anthropic

    return sdk_client(anthropic.AsyncAnthropic, target, key, http_client)


def endpoint(sdk: Any) -> Any:
    return sdk.messages.with_raw_response


def _params(target: Target, messages: Sequence[TurnMessage]) -> dict[str, Any]:
    """The request without tools: system messages joined into the `system` field."""
    params: dict[str, Any] = {
        "model": target.model,
        "max_tokens": target.max_output_tokens,  # required; build() checks it is set
        "messages": _history(messages),
    }
    system = "\n\n".join(
        m.content for m in messages if isinstance(m, Message) and m.role == "system"
    )
    if system:
        params["system"] = system
    return params


def request(
    target: Target, messages: Sequence[Message], schema: type[BaseModel] | None
) -> dict[str, Any]:
    params = _params(target, messages)
    if schema is not None:
        name = schema.__name__
        params["tools"] = [{"name": name, "input_schema": strict_schema(schema)}]
        params["tool_choice"] = {"type": "tool", "name": name}
    return params


def _assistant(turn: AssistantTurn) -> dict[str, Any]:
    """The turn's blocks in their order. This protocol's opaque blocks (thinking, with its
    signature) go back exactly as received; another protocol's are left out (design §5)."""
    content: list[dict[str, Any]] = []
    for b in turn.blocks:
        if isinstance(b, Text):
            if b.text:  # the API rejects an empty text block
                content.append({"type": "text", "text": b.text})
        elif isinstance(b, ToolCall):
            # an input that never parsed (another protocol's turn) has no dict form
            sent = dict(b.arguments) if b.arguments is not None else {}
            content.append({"type": "tool_use", "id": b.id, "name": b.name, "input": sent})
        elif b.protocol == "anthropic":
            content.append(dict(b.data))
    return {"role": "assistant", "content": content}


def _history(messages: Sequence[TurnMessage]) -> list[dict[str, Any]]:
    sent: list[dict[str, Any]] = []
    for m in messages:
        if isinstance(m, Message):
            if m.role != "system":
                sent.append({"role": m.role, "content": m.content})
        elif isinstance(m, AssistantTurn):
            sent.append(_assistant(m))
        elif m.results:  # all of them in one user message, as the API requires; none: no message
            results: list[dict[str, Any]] = []
            for r in m.results:
                block = {"type": "tool_result", "tool_use_id": r.call_id, "content": r.content}
                results.append(block | {"is_error": True} if r.is_error else block)
            sent.append({"role": "user", "content": results})
    return sent


def turn_request(
    target: Target, messages: Sequence[TurnMessage], tools: Sequence[ToolSpec]
) -> dict[str, Any]:
    params = _params(target, messages)
    if tools:  # no `tool_choice`: auto is the default, and newer models reject a forced one
        params["tools"] = [
            {"name": t.name, "description": t.description, "input_schema": dict(t.parameters)}
            for t in tools
        ]
    return params


def _block(raw: Any) -> Block:
    if raw.type == "text":
        return Text(raw.text)
    if raw.type == "tool_use":
        return ToolCall(raw.id, raw.name, dict(raw.input))
    # thinking, redacted_thinking and anything else lite-model does not interpret: kept as
    # received (`to_dict()` gives back the fields the endpoint sent) for the next request
    return Opaque("anthropic", raw.to_dict())


def turn_reply(
    target: Target, response: Any, headers: Mapping[str, str], sent: Mapping[str, Any]
) -> TurnReply:
    return finish_turn(
        target,
        sent,
        headers,
        body_model=response.model,
        usage=Usage(input=response.usage.input_tokens, output=response.usage.output_tokens),
        truncated=response.stop_reason in TRUNCATED,
        unusable=_unusable(response.stop_reason),
        blocks=[_block(b) for b in response.content],
    )


def reply(
    target: Target,
    response: Any,
    headers: Mapping[str, str],
    schema: type[BaseModel] | None,
    sent: Mapping[str, Any],
) -> Reply:
    blocks = response.content
    payload: str | Mapping[str, Any] | None
    if schema is None:
        payload = "".join(b.text for b in blocks if b.type == "text")
    else:
        name = schema.__name__  # only the forced tool's input is the reply
        payload = next((b.input for b in blocks if b.type == "tool_use" and b.name == name), None)
    return finish(
        target,
        schema,
        sent,
        headers,
        body_model=response.model,
        usage=Usage(input=response.usage.input_tokens, output=response.usage.output_tokens),
        truncated=response.stop_reason in TRUNCATED,
        unusable=_unusable(response.stop_reason),
        payload=payload,
    )


def _unusable(stop_reason: str | None) -> str | None:
    if stop_reason in COMPLETE | TRUNCATED:
        return None
    return stop_reason if stop_reason in UNUSABLE else "unexpected"
