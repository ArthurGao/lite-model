"""lite-model: the shared model-call layer for the lite modules (docs/lite-model-design.md)."""

from lite_model.client import Model, build
from lite_model.config import (
    BackendConfig,
    ModelsConfig,
    RoleConfig,
    RolePolicy,
    RoleSpec,
    Target,
)
from lite_model.errors import (
    BadStructuredReply,
    CallFailed,
    ConfigError,
    EmptyReply,
    IgnoredParameter,
    ModelError,
    ProvenanceUnknown,
    Truncated,
)
from lite_model.schema import strict_schema
from lite_model.types import Channel, Message, Provenance, StructuredReply, TextReply, Usage

__all__ = [
    "BackendConfig",
    "BadStructuredReply",
    "CallFailed",
    "Channel",
    "ConfigError",
    "EmptyReply",
    "IgnoredParameter",
    "Message",
    "Model",
    "ModelError",
    "ModelsConfig",
    "Provenance",
    "ProvenanceUnknown",
    "RoleConfig",
    "RolePolicy",
    "RoleSpec",
    "StructuredReply",
    "Target",
    "TextReply",
    "Truncated",
    "Usage",
    "build",
    "strict_schema",
]
