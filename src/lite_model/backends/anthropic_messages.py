"""Anthropic Messages over the Anthropic SDK (the `anthropic` extra; design §5). The request is
built as lite-mem and lite-rag build theirs today, so their bodies are unchanged when they migrate:
system messages joined into the `system` field, structured output as one forced tool."""

from collections.abc import Mapping, Sequence
from typing import Any

from pydantic import BaseModel

from lite_model.backends import Reply, finish, sdk_client
from lite_model.config import Target
from lite_model.schema import strict_schema
from lite_model.types import Message, Usage

# stop_reason values (anthropic 2.6.0): cut short, or complete. Any other one (refusal,
# pause_turn, one added later) leaves no usable answer.
TRUNCATED = {"max_tokens", "model_context_window_exceeded"}
COMPLETE = {"end_turn", "stop_sequence", "tool_use", None}


def connect(target: Target, key: str | None, http_client: Any) -> Any:
    import anthropic

    return sdk_client(anthropic.Anthropic, target, key, http_client)


def connect_async(target: Target, key: str | None, http_client: Any) -> Any:
    import anthropic

    return sdk_client(anthropic.AsyncAnthropic, target, key, http_client)


def endpoint(sdk: Any) -> Any:
    return sdk.messages.with_raw_response


def request(
    target: Target, messages: Sequence[Message], schema: type[BaseModel] | None
) -> dict[str, Any]:
    params: dict[str, Any] = {
        "model": target.model,
        "max_tokens": target.max_output_tokens,  # required; build() checks it is set
        "messages": [
            {"role": m.role, "content": m.content} for m in messages if m.role != "system"
        ],
    }
    system = "\n\n".join(m.content for m in messages if m.role == "system")
    if system:
        params["system"] = system
    if schema is not None:
        name = schema.__name__
        params["tools"] = [{"name": name, "input_schema": strict_schema(schema)}]
        params["tool_choice"] = {"type": "tool", "name": name}
    return params


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
        unusable=None if response.stop_reason in COMPLETE | TRUNCATED else response.stop_reason,
        payload=payload,
    )
