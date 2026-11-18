"""lite-model: the shared model-call layer for the lite modules (docs/lite-model-design.md)."""

from lite_model.bounded import AsyncBounded, Bounded
from lite_model.client import AsyncLiteModel, AsyncModel, LiteModel, Model, build, build_async
from lite_model.config import (
    BackendConfig,
    ModelsConfig,
    Price,
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
from lite_model.hooks import call_context
from lite_model.schema import strict_schema
from lite_model.types import Channel, Message, Provenance, Reply, StructuredReply, TextReply, Usage

__all__ = [
    "AsyncBounded",
    "AsyncLiteModel",
    "AsyncModel",
    "BackendConfig",
    "BadStructuredReply",
    "Bounded",
    "CallFailed",
    "Channel",
    "ConfigError",
    "EmptyReply",
    "IgnoredParameter",
    "LiteModel",
    "Message",
    "Model",
    "ModelError",
    "ModelsConfig",
    "Price",
    "Provenance",
    "ProvenanceUnknown",
    "Reply",
    "RoleConfig",
    "RolePolicy",
    "RoleSpec",
    "StructuredReply",
    "Target",
    "TextReply",
    "Truncated",
    "Usage",
    "build",
    "build_async",
    "call_context",
    "strict_schema",
]
