"""`FakeModel` and `AsyncFakeModel` (design §11): scripted replies per role, every call
recorded; no network."""

import hashlib
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Self

from pydantic import BaseModel

from lite_model.types import (
    AssistantTurn,
    EmbedReply,
    InputType,
    Message,
    Provenance,
    RerankReply,
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
    input_type: InputType | None = None  # 0.7.0


@dataclass(frozen=True)
class FakeRerankCall:
    """One `rerank()` call (design §3.3), recorded in `rerank_calls`."""

    role: str
    query: str
    documents: tuple[str, ...]
    top_k: int | None


# rerank() scripts, by role (`FakeModel.with_scores({...})`): one score per document, or raise.
RerankScript = Callable[[FakeRerankCall], list[float]] | BaseException


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
# embed() scripts, by role, apart from the others (`FakeModel.with_vectors({...})`): the vectors
# for one call, computed from it (e.g. `hash_vectors(dim)`), or an exception to raise.
EmbedScript = Callable[["FakeEmbedCall"], list[list[float]]] | BaseException


class FakeModel:
    """Implements `Model`. Provenance says `backend="fake"`, so a fake answer is never mistaken
    for a model's."""

    def __init__(self, **scripts: Script):
        self._scripts = {role: list(s) if isinstance(s, list) else s for role, s in scripts.items()}
        self._vectors: dict[str, EmbedScript] = {}
        self.calls: list[FakeCall] = []
        self.turn_calls: list[FakeTurnCall] = []
        self.embed_calls: list[FakeEmbedCall] = []
        self._scores: dict[str, RerankScript] = {}
        self.rerank_calls: list[FakeRerankCall] = []

    @classmethod
    def with_scores(cls, scores: Mapping[str, RerankScript], /, **scripts: Script) -> Self:
        """A fake that also serves `rerank()` for `scores`' roles (0.7.0); like `with_vectors`."""
        fake = cls(**scripts)
        fake._scores = dict(scores)
        return fake

    def rerank(
        self, role: str, query: str, documents: Sequence[str], *, top_k: int | None = None
    ) -> RerankReply:
        call = FakeRerankCall(role, query, tuple(documents), top_k)
        self.rerank_calls.append(call)
        if role not in self._scores:
            raise AssertionError(f"FakeModel has no scores scripted for role {role!r}")
        script = self._scores[role]
        if isinstance(script, BaseException):
            raise script
        scores = script(call)
        if len(scores) != len(documents):
            raise AssertionError(f"FakeModel: role {role!r} gave {len(scores)} scores")
        kept: list[float | None] = list(scores)
        if top_k is not None:  # as the real reply: None past the top_k best
            best = set(sorted(range(len(kept)), key=lambda i: -scores[i])[:top_k])
            kept = [s if i in best else None for i, s in enumerate(kept)]
        return RerankReply(kept, self._provenance(role), Usage())

    @classmethod
    def with_vectors(cls, vectors: Mapping[str, EmbedScript], /, **scripts: Script) -> Self:
        """A fake that also serves `embed()` for `vectors`' roles. Not a keyword of `__init__`
        (0.5.0 had one): hosts build fakes from a dict of scripts (`FakeModel(**scripts)`), and a
        keyword parameter is matched against every value of such a dict (0.5.1)."""
        fake = cls(**scripts)
        fake._vectors = dict(vectors)
        return fake

    def close(self) -> None:
        """Nothing to release; `LiteModel.close()`'s counterpart (0.6.0)."""

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

    def embed(
        self, role: str, texts: Sequence[str], *, input_type: InputType | None = None
    ) -> EmbedReply:
        call = FakeEmbedCall(role, tuple(texts), input_type)
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

    def __init__(self, **scripts: Script):
        self._fake = FakeModel(**scripts)

    @classmethod
    def with_vectors(cls, vectors: Mapping[str, EmbedScript], /, **scripts: Script) -> Self:
        """As `FakeModel.with_vectors`."""
        fake = cls()
        fake._fake = FakeModel.with_vectors(vectors, **scripts)
        return fake

    @classmethod
    def with_scores(cls, scores: Mapping[str, RerankScript], /, **scripts: Script) -> Self:
        """As `FakeModel.with_scores`."""
        fake = cls()
        fake._fake = FakeModel.with_scores(scores, **scripts)
        return fake

    @property
    def rerank_calls(self) -> list[FakeRerankCall]:
        return self._fake.rerank_calls

    async def rerank(
        self, role: str, query: str, documents: Sequence[str], *, top_k: int | None = None
    ) -> RerankReply:
        return self._fake.rerank(role, query, documents, top_k=top_k)

    async def aclose(self) -> None:
        """Nothing to release; `AsyncLiteModel.aclose()`'s counterpart (0.6.0)."""

    @property
    def calls(self) -> list[FakeCall]:
        return self._fake.calls

    @property
    def turn_calls(self) -> list[FakeTurnCall]:
        return self._fake.turn_calls

    @property
    def embed_calls(self) -> list[FakeEmbedCall]:
        return self._fake.embed_calls

    async def embed(
        self, role: str, texts: Sequence[str], *, input_type: InputType | None = None
    ) -> EmbedReply:
        return self._fake.embed(role, texts, input_type=input_type)

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
