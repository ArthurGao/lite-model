"""One module per wire protocol (design §5): `openai_compat` (ollama, openai, gateway, litellm) and
`anthropic_messages`. Each turns a target and messages into SDK arguments (`request`) and an SDK
response into a reply (`reply`); `client.py` runs the call, sync or async, the same way for both.
Neither function touches the network, so contract tests can check both sides exactly."""

import math
from collections.abc import Mapping, Sequence
from dataclasses import replace
from typing import Any, Protocol

from pydantic import BaseModel, ValidationError

from lite_model.config import Target
from lite_model.errors import (
    BadStructuredReply,
    EmptyReply,
    IgnoredParameter,
    ModelError,
    ProvenanceUnknown,
    Truncated,
)
from lite_model.profiles import PROFILES
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
        sent: Mapping[str, Any],
    ) -> Reply:
        """`sent` is what `request()` returned for this call: the fields actually sent."""
        ...


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


# SDK arguments and fields every request carries: never "a configured parameter" (§5).
_NOT_OPTIONS = {"model", "messages", "extra_headers", "extra_body"}


def _answered(target: Target, body_model: str | None, headers: Mapping[str, str]) -> str | None:
    """The answering model: the body's `model`, or the profile's header when the body names an
    alias (design §6.3). A guess is never recorded (AGENTS.md rule 4)."""
    name = PROFILES[target.type].answered_header
    if name is None:
        return body_model or None
    answered = headers.get(name)
    # A header that only repeats the alias says nothing about the deployment (Codex S3); a
    # routing alias with `return_raw_model_name` off answers with the alias in the body.
    if not answered or answered == target.model or (target.routed and body_model != answered):
        raise ProvenanceUnknown(f"{target.where}: the endpoint did not say which model answered")
    return answered


def _check_ignored(target: Target, headers: Mapping[str, str], sent: Mapping[str, Any]) -> None:
    name = PROFILES[target.type].ignored_params_header
    listed = headers.get(name) if name else None
    if not listed:
        return
    dropped = {p.strip() for p in listed.split(",")} & (set(sent) - _NOT_OPTIONS)
    if dropped:  # our own field names, never the endpoint's list
        raise IgnoredParameter(f"{target.where}: the endpoint ignored {', '.join(sorted(dropped))}")


def _cost(
    target: Target, usage: Usage, headers: Mapping[str, str], answered: str | None
) -> float | None:
    """USD for this call (design §9): the endpoint's own figure, else the price table for the
    model that answered, else None. Never estimated; a subscription costs None."""
    if target.channel == "subscription":
        return None
    name = PROFILES[target.type].cost_header
    reported = _number(headers.get(name)) if name else None
    if reported is not None:
        return reported
    price = target.prices.get(answered) if answered else None
    if price is None or usage.input is None or usage.output is None:
        return None
    cost = (usage.input * price.input_per_mtok + usage.output * price.output_per_mtok) / 1e6
    return cost if math.isfinite(cost) else None  # an overflow is unknown, not infinite


def _number(raw: str | None) -> float | None:
    if raw is None:
        return None
    try:
        value = float(raw)
    except ValueError:
        return None
    return value if math.isfinite(value) and value >= 0 else None


def finish(
    target: Target,
    schema: type[BaseModel] | None,
    sent: Mapping[str, Any],
    headers: Mapping[str, str],
    *,
    body_model: str | None,
    usage: Usage,
    truncated: bool,
    unusable: str | None,
    payload: str | Mapping[str, Any] | None,
) -> Reply:
    """The reply, or the error it amounts to (design §5, §6, §8), for every protocol. Who
    answered and which parameters were dropped come from the profile's header rules; never from
    config (§6.1). `payload` is the text, the JSON text of a structured reply, or a forced tool's
    input; messages never quote it. `unusable` is a stop reason that leaves no usable answer
    whatever the payload holds (a refusal, a filter, a paused turn)."""
    headers = {k.lower(): v for k, v in headers.items()}  # profile names are lower case
    try:
        _check_ignored(target, headers, sent)
        answered = _answered(target, body_model, headers)
    except ModelError as exc:  # the call was answered and billed all the same (§9)
        exc.usage = replace(usage, cost_usd=_cost(target, usage, headers, None))
        raise
    usage = replace(usage, cost_usd=_cost(target, usage, headers, answered))
    who = Provenance(
        role=target.role,
        backend=target.backend,
        requested=target.model,
        answered=answered,
        channel=target.channel,
        routed=target.routed,
        via={h: headers[h] for h in PROFILES[target.type].via_headers if h in headers},
    )
    where, tag = target.where, who.tag
    if truncated:
        raise Truncated(f"{where}: reply truncated at the output limit ({tag})", usage=usage)
    if unusable is not None:  # a stop-reason enum value, not upstream text
        msg = f"{where}: no usable reply (stop reason {unusable}) ({tag})"
        raise EmptyReply(msg, usage=usage)
    if payload is None or (isinstance(payload, str) and not payload.strip()):
        raise EmptyReply(f"{where}: empty reply ({tag})", usage=usage)
    if schema is None:
        assert isinstance(payload, str)
        return TextReply(payload, who, usage)
    try:
        if isinstance(payload, str):
            value = schema.model_validate_json(payload)
        else:
            value = schema.model_validate(payload)
    except ValidationError as exc:
        msg = f"{where}: reply does not match {schema.__name__} ({tag})"
        raise BadStructuredReply(msg, usage=usage) from exc
    return StructuredReply(value, who, usage)
