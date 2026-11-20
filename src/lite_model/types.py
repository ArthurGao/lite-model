"""What crosses the interface (design §3): messages in; replies with provenance and usage out."""

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Literal

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
    # The endpoint's own provenance headers (lower-cased names from the profile's allow-list),
    # e.g. x-litellm-model-id; empty when it sends none (design §6.2, §6.3).
    via: Mapping[str, str] = field(default_factory=dict, hash=False)

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
    cached: bool = False  # a hook answered (design §10); provenance stays the original one


@dataclass(frozen=True)
class StructuredReply[T: BaseModel]:
    value: T
    provenance: Provenance
    usage: Usage
    cached: bool = False  # a hook answered (design §10); provenance stays the original one


Reply = TextReply | StructuredReply[Any]


# --- One model-chosen tool turn (design §3.1) ---
# These are frozen like the types above, but only shallowly: a Mapping field keeps the caller's
# object (as `Provenance.via` does). Do not mutate a mapping after handing it over.


@dataclass(frozen=True)
class ToolSpec:
    """A tool offered to the model. lite-model never executes it."""

    name: str
    description: str
    parameters: Mapping[str, Any] = field(hash=False)  # JSON Schema of the input


@dataclass(frozen=True)
class Text:
    text: str


@dataclass(frozen=True)
class ToolCall:
    id: str
    name: str
    # None: the endpoint's JSON text did not parse to an object; `raw_arguments` then holds it
    # as received. A turn cut off at the output limit can also carry a parsed but partial input.
    arguments: Mapping[str, Any] | None = field(hash=False)
    raw_arguments: str | None = None


@dataclass(frozen=True)
class Opaque:
    """Content only its own protocol understands (Anthropic thinking blocks, with their
    signature). It goes back to that protocol exactly as received and is left out of a request
    to the other one (design §5)."""

    protocol: Literal["openai", "anthropic"]
    data: Mapping[str, Any] = field(hash=False)


Block = Text | ToolCall | Opaque


@dataclass(frozen=True)
class AssistantTurn:
    """What the model said in one turn, in the order the endpoint gave it. Put it back into the
    history unchanged: the order and the opaque blocks matter to the endpoint."""

    blocks: tuple[Block, ...]

    @property
    def text(self) -> str:
        return "".join(b.text for b in self.blocks if isinstance(b, Text))

    @property
    def tool_calls(self) -> tuple[ToolCall, ...]:
        return tuple(b for b in self.blocks if isinstance(b, ToolCall))


@dataclass(frozen=True)
class ToolResult:
    call_id: str
    content: str
    is_error: bool = False


@dataclass(frozen=True)
class ToolResults:
    """Every result of one assistant turn's calls, as one history item: Anthropic wants them in
    a single user message, Chat Completions as one `tool` message each."""

    results: tuple[ToolResult, ...]


TurnMessage = Message | AssistantTurn | ToolResults
Stop = Literal["end", "tool_calls", "length"]


def stop_for(turn: AssistantTurn, *, truncated: bool = False) -> Stop:
    """The stop a turn amounts to: the one rule the backends and the fakes share."""
    if truncated:
        return "length"
    return "tool_calls" if turn.tool_calls else "end"


@dataclass(frozen=True)
class TurnReply:
    turn: AssistantTurn
    # "length": cut off at the output limit. The calls it carries must not be executed, and each
    # still needs a result before the next request; that is why this is a reply, not Truncated.
    stop: Stop
    provenance: Provenance
    usage: Usage
    cached: bool = False  # the endpoint's own cache answered (design §9); hooks never see turns
