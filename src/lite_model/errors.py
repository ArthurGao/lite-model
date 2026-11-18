"""The error hierarchy (design §8). Messages carry the role, `backend:model` and the exception
type only, never upstream text: SDKs and proxies can echo keys, auth headers or credentialed URLs.
The original exception stays chained as `__cause__`."""

from lite_model.types import Usage


class ModelError(Exception):
    """Base of every lite-model error. `usage` is what the call spent when the endpoint answered
    and the answer was refused (truncated, empty, invalid...): billed all the same (design §9).
    None when nothing came back (a config error, a transport failure)."""

    def __init__(self, message: str, *, usage: Usage | None = None) -> None:
        super().__init__(message)
        self.usage = usage


class ConfigError(ModelError):
    """The configuration is invalid; raised by `build()`."""


class CallFailed(ModelError):
    """Transport, HTTP or SDK error, after the retries on the same target."""


class Truncated(ModelError):
    """The reply was cut short (`finish_reason: length`; `stop_reason: max_tokens` or
    `model_context_window_exceeded`)."""


class EmptyReply(ModelError):
    """No usable answer: no text (e.g. the whole budget went to thinking), no forced-tool input,
    or a stop reason that leaves none (a refusal, a content filter, a paused turn)."""


class BadStructuredReply(ModelError):
    """The reply does not validate against the schema."""


class IgnoredParameter(ModelError):
    """The endpoint reported a configured parameter as ignored (design §5)."""


class ProvenanceUnknown(ModelError):
    """The answering model cannot be read: a routed call, or any call to a backend that names it
    in a header (LiteLLM) that did not (design §6.3)."""
