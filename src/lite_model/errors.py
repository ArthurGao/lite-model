"""The error hierarchy (design §8). Messages carry the role, `backend:model` and the exception
type only, never upstream text: SDKs and proxies can echo keys, auth headers or credentialed URLs.
The original exception stays chained as `__cause__`."""


class ModelError(Exception):
    """Base of every lite-model error."""


class ConfigError(ModelError):
    """The configuration is invalid; raised by `build()`."""


class CallFailed(ModelError):
    """Transport, HTTP or SDK error, after the retries on the same target."""


class Truncated(ModelError):
    """The reply stopped at the output limit (`finish_reason: length`)."""


class EmptyReply(ModelError):
    """No text in the reply (e.g. the whole budget went to thinking)."""


class BadStructuredReply(ModelError):
    """The reply does not validate against the schema."""


class IgnoredParameter(ModelError):
    """The endpoint reported a configured parameter as ignored (design §5)."""


class ProvenanceUnknown(ModelError):
    """A routed call whose answering model cannot be read (design §6)."""
