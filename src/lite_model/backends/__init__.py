"""One module per wire protocol (design §5): `openai_compat` (ollama, openai, gateway, litellm) and
`anthropic_messages`. Each turns a target and messages into SDK arguments (`request`) and an SDK
response into a reply (`reply`); `client.py` runs the call, sync or async, the same way for both.
Neither function touches the network, so contract tests can check both sides exactly."""

from collections.abc import Mapping, Sequence
from typing import Any, Protocol

from pydantic import BaseModel, ValidationError

from lite_model.config import Target
from lite_model.errors import BadStructuredReply, EmptyReply, Truncated
from lite_model.types import Message, Provenance, StructuredReply, TextReply, Usage

Reply = TextReply | StructuredReply[Any]


class Backend(Protocol):
    def connect(self, target: Target, key: str | None, http_client: Any) -> Any: ...

    def connect_async(self, target: Target, key: str | None, http_client: Any) -> Any: ...

    def endpoint(self, sdk: Any) -> Any:
        """The SDK's raw-response `create` owner (the same path on sync and async clients)."""
        ...

    def request(
        self, target: Target, messages: Sequence[Message], schema: type[BaseModel] | None
    ) -> dict[str, Any]: ...

    def reply(
        self,
        target: Target,
        response: Any,
        headers: Mapping[str, str],
        schema: type[BaseModel] | None,
    ) -> Reply: ...


def sdk_client(cls: Any, target: Target, key: str | None, http_client: Any) -> Any:
    """An SDK client (`openai.OpenAI`, `anthropic.AsyncAnthropic`, ...) for `target`: both SDKs
    take the same arguments and use httpx2's Timeout."""
    import httpx2

    return cls(
        api_key=key,
        base_url=target.base_url,
        timeout=httpx2.Timeout(target.timeout_s, connect=target.connect_timeout_s),
        max_retries=target.max_retries,
        http_client=http_client,
    )


def finish(
    target: Target,
    schema: type[BaseModel] | None,
    answered: str | None,
    usage: Usage,
    *,
    truncated: bool,
    unusable: str | None,
    payload: str | Mapping[str, Any] | None,
) -> Reply:
    """The reply, or the error it amounts to (design §8). `answered` is the model the response
    names, never config's (§6.1). `payload` is the text, the JSON text of a structured reply, or
    a forced tool's input; messages never quote it. `unusable` is a stop reason that leaves no
    usable answer whatever the payload holds (a refusal, a filter, a paused turn)."""
    who = Provenance(
        role=target.role,
        backend=target.backend,
        requested=target.model,
        answered=answered or None,
        channel=target.channel,
        routed=target.routed,
    )
    if truncated:
        raise Truncated(f"{target.where}: reply truncated at the output limit ({who.tag})")
    if unusable is not None:  # a stop-reason enum value, not upstream text
        raise EmptyReply(f"{target.where}: no usable reply (stop reason {unusable}) ({who.tag})")
    if payload is None or (isinstance(payload, str) and not payload.strip()):
        raise EmptyReply(f"{target.where}: empty reply ({who.tag})")
    if schema is None:
        assert isinstance(payload, str)
        return TextReply(payload, who, usage)
    try:
        if isinstance(payload, str):
            value = schema.model_validate_json(payload)
        else:
            value = schema.model_validate(payload)
    except ValidationError as exc:
        raise BadStructuredReply(
            f"{target.where}: reply does not match {schema.__name__} ({who.tag})"
        ) from exc
    return StructuredReply(value, who, usage)
