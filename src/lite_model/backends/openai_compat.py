"""Chat Completions over the OpenAI SDK: Ollama `/v1`, OpenAI and any OpenAI-compatible endpoint
(design §5). What differs between backend types comes from `profiles.py`."""

from collections.abc import Mapping, Sequence
from typing import Any

from pydantic import BaseModel

from lite_model.backends import Reply, finish, sdk_client
from lite_model.config import Target
from lite_model.profiles import PROFILES
from lite_model.schema import strict_schema
from lite_model.types import Message, Usage

# finish_reason values that leave no usable answer, whatever the content holds.
UNUSABLE = {"content_filter"}


def connect(target: Target, key: str | None, http_client: Any) -> Any:
    import openai

    # The SDK insists on a key; keyless endpoints (Ollama) ignore it.
    return sdk_client(openai.OpenAI, target, key or "unused", http_client)


def connect_async(target: Target, key: str | None, http_client: Any) -> Any:
    import openai

    return sdk_client(openai.AsyncOpenAI, target, key or "unused", http_client)


def endpoint(sdk: Any) -> Any:
    return sdk.chat.completions.with_raw_response


def request(
    target: Target, messages: Sequence[Message], schema: type[BaseModel] | None
) -> dict[str, Any]:
    params: dict[str, Any] = {
        "model": target.model,
        "messages": [{"role": m.role, "content": m.content} for m in messages],
    }
    if target.max_output_tokens is not None:
        params[PROFILES[target.type].max_tokens_param] = target.max_output_tokens
    if target.temperature is not None:
        params["temperature"] = target.temperature
    if target.reasoning_effort is not None:
        params["reasoning_effort"] = target.reasoning_effort
    if schema is not None:
        strict = {"name": schema.__name__, "schema": strict_schema(schema), "strict": True}
        params["response_format"] = {"type": "json_schema", "json_schema": strict}
    return params


def reply(
    target: Target, response: Any, headers: Mapping[str, str], schema: type[BaseModel] | None
) -> Reply:
    """S1-S2 read the body's `model`; backends that name the answering model in a header
    (§6.2, §6.3) read it here, from `headers`."""
    usage = Usage()
    if response.usage is not None:
        details = response.usage.completion_tokens_details
        usage = Usage(
            input=response.usage.prompt_tokens,
            output=response.usage.completion_tokens,
            reasoning=details.reasoning_tokens if details is not None else None,
        )
    choice = response.choices[0] if response.choices else None
    return finish(
        target,
        schema,
        response.model,
        usage,
        truncated=choice is not None and choice.finish_reason == "length",
        unusable=choice.finish_reason if choice and choice.finish_reason in UNUSABLE else None,
        payload=choice.message.content if choice is not None else None,
    )
