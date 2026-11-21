"""`FakeModel` and `AsyncFakeModel` (design §11): scripted replies per role, every call
recorded; no network."""

import hashlib
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel

from lite_model.types import (
    AssistantTurn,
    EmbedReply,
    Message,
    Provenance,
    Stop,
    StructuredReply,
    TextReply,
    ToolSpec,
    TurnMessage,
    TurnReply,
    Usage,
    stop_for,
)


@dataclass(frozen=True)
class FakeCall:
    role: str
    messages: tuple[Message, ...]
    schema: type[BaseModel] | None  # None for text()


@dataclass(frozen=True)
class FakeTurnCall:
    """One `turn()` call. Kept apart from `FakeCall` (and in `turn_calls`, not `calls`) so
    existing tests that read `calls[i].messages[j].content` keep their types."""

    role: str
    messages: tuple[TurnMessage, ...]
    tools: tuple[ToolSpec, ...]


@dataclass(frozen=True)
class FakeEmbedCall:
    """One `embed()` call (design §3.2), recorded in `embed_calls`."""

    role: str
    texts: tuple[str, ...]


def hash_vectors(dim: int) -> Callable[[FakeEmbedCall], list[list[float]]]:
    """An embed script: a fixed `dim`-long vector per text, from its sha256 (equal texts, equal
    vectors; no similarity between different ones)."""

    def vectors(call: FakeEmbedCall) -> list[list[float]]:
        out = []
        for text in call.texts:
            digest = hashlib.sha256(text.encode()).digest()
            out.append([(digest[i % len(digest)] - 127.5) / 127.5 for i in range(dim)])
        return out

    return vectors


# A reply: text (for text()), a model instance or a dict (for structured()), an AssistantTurn
# (for turn(); its stop is "tool_calls" when it carries calls, else "end") or (AssistantTurn,
# stop) to script another stop such as "length", or an exception to raise. A list is consumed one
# reply per call; a callable computes the reply from the call (a FakeCall, or a FakeTurnCall for
# turn()).
FakeReply = (
    str | BaseModel | dict[str, Any] | AssistantTurn | tuple[AssistantTurn, Stop] | BaseException
)
Script = FakeReply | list[FakeReply] | Callable[[Any], FakeReply]
# embed() scripts, by role, apart from the others (`FakeModel(vectors={...})`): the vectors for one
# call, computed from it (e.g. `hash_vectors(dim)`), or an exception to raise.
EmbedScript = Callable[["FakeEmbedCall"], list[list[float]]] | BaseException


class FakeModel:
    """Implements `Model`. Provenance says `backend="fake"`, so a fake answer is never mistaken
    for a model's."""

    def __init__(self, *, vectors: Mapping[str, EmbedScript] | None = None, **scripts: Script):
        self._scripts = {role: list(s) if isinstance(s, list) else s for role, s in scripts.items()}
        self._vectors = dict(vectors or {})
        self.calls: list[FakeCall] = []
        self.turn_calls: list[FakeTurnCall] = []
        self.embed_calls: list[FakeEmbedCall] = []

    def _reply[C: (FakeCall, FakeTurnCall)](self, call: C, log: list[C]) -> FakeReply:
        log.append(call)
        if call.role not in self._scripts:
            raise AssertionError(f"FakeModel has no reply scripted for role {call.role!r}")
        script = self._scripts[call.role]
        if isinstance(script, list):
            if not script:
                raise AssertionError(f"FakeModel ran out of replies for role {call.role!r}")
            reply = script.pop(0)
        elif callable(script):
            reply = script(call)
        else:
            reply = script
        if isinstance(reply, BaseException):
            raise reply
        return reply

    @staticmethod
    def _provenance(role: str) -> Provenance:
        return Provenance(role, "fake", "fake-model", "fake-model", "local", routed=False)

    def turn(
        self, role: str, messages: Sequence[TurnMessage], tools: Sequence[ToolSpec]
    ) -> TurnReply:
        reply = self._reply(FakeTurnCall(role, tuple(messages), tuple(tools)), self.turn_calls)
        if isinstance(reply, AssistantTurn):
            reply = (reply, stop_for(reply))
        if not isinstance(reply, tuple):
            raise AssertionError(
                f"FakeModel: role {role!r} scripted {type(reply).__name__}, "
                "turn() needs an AssistantTurn"
            )
        turn, stop = reply
        return TurnReply(turn, stop, self._provenance(role), Usage())

    def embed(self, role: str, texts: Sequence[str]) -> EmbedReply:
        call = FakeEmbedCall(role, tuple(texts))
        self.embed_calls.append(call)
        if role not in self._vectors:
            raise AssertionError(f"FakeModel has no vectors scripted for role {role!r}")
        script = self._vectors[role]
        if isinstance(script, BaseException):
            raise script
        vectors = script(call)
        if len(vectors) != len(texts):
            raise AssertionError(
                f"FakeModel: role {role!r} gave {len(vectors)} vectors, not {len(texts)}"
            )
        return EmbedReply([list(v) for v in vectors], self._provenance(role), Usage())

    def text(self, role: str, messages: Sequence[Message]) -> TextReply:
        reply = self._reply(FakeCall(role, tuple(messages), None), self.calls)
        if not isinstance(reply, str):
            raise AssertionError(
                f"FakeModel: role {role!r} scripted {type(reply).__name__}, text() needs str"
            )
        return TextReply(reply, self._provenance(role), Usage())

    def structured[T: BaseModel](
        self, role: str, messages: Sequence[Message], schema: type[T]
    ) -> StructuredReply[T]:
        reply = self._reply(FakeCall(role, tuple(messages), schema), self.calls)
        if not isinstance(reply, BaseModel | dict):
            raise AssertionError(
                f"FakeModel: role {role!r} scripted {type(reply).__name__}, "
                "structured() needs an object"
            )
        data = reply.model_dump() if isinstance(reply, BaseModel) else reply
        return StructuredReply(schema.model_validate(data), self._provenance(role), Usage())


class AsyncFakeModel:
    """Implements `AsyncModel`, with the same scripts and the same `calls` as `FakeModel`."""

    def __init__(self, *, vectors: Mapping[str, EmbedScript] | None = None, **scripts: Script):
        self._fake = FakeModel(vectors=vectors, **scripts)

    @property
    def calls(self) -> list[FakeCall]:
        return self._fake.calls

    @property
    def turn_calls(self) -> list[FakeTurnCall]:
        return self._fake.turn_calls

    @property
    def embed_calls(self) -> list[FakeEmbedCall]:
        return self._fake.embed_calls

    async def embed(self, role: str, texts: Sequence[str]) -> EmbedReply:
        return self._fake.embed(role, texts)

    async def text(self, role: str, messages: Sequence[Message]) -> TextReply:
        return self._fake.text(role, messages)

    async def structured[T: BaseModel](
        self, role: str, messages: Sequence[Message], schema: type[T]
    ) -> StructuredReply[T]:
        return self._fake.structured(role, messages, schema)

    async def turn(
        self, role: str, messages: Sequence[TurnMessage], tools: Sequence[ToolSpec]
    ) -> TurnReply:
        return self._fake.turn(role, messages, tools)
