"""What crosses the interface (design §3): messages in; replies with provenance and usage out."""

from dataclasses import dataclass
from typing import Literal

from pydantic import BaseModel

Channel = Literal["local", "api", "subscription"]


@dataclass(frozen=True)
class Message:
    role: Literal["system", "user", "assistant"]
    content: str


@dataclass(frozen=True)
class Provenance:
    """Who answered (design §6). `answered` is read from the response, never from config."""

    role: str  # the caller's role, e.g. "answer"
    backend: str  # the configured backend's name, e.g. "ollama" or "deepseek"
    requested: str  # the model / alias / profile the call asked for
    answered: str | None  # the model that actually answered; None = the endpoint did not say
    channel: Channel
    routed: bool  # requested was a routing alias

    @property
    def tag(self) -> str:
        """`backend:answered`, or `backend:requested?` when the endpoint did not say."""
        if self.answered is None:
            return f"{self.backend}:{self.requested}?"
        return f"{self.backend}:{self.answered}"


@dataclass(frozen=True)
class Usage:
    input: int | None = None
    output: int | None = None
    reasoning: int | None = None
    cost_usd: float | None = None  # None = unknown; never estimated


@dataclass(frozen=True)
class TextReply:
    text: str
    provenance: Provenance
    usage: Usage


@dataclass(frozen=True)
class StructuredReply[T: BaseModel]:
    value: T
    provenance: Provenance
    usage: Usage
