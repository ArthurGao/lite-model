"""`FakeModel` and `AsyncFakeModel` (design §11): scripted replies per role, every call
recorded; no network."""

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel

from lite_model.types import (
    AssistantTurn,
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


# A reply: text (for text()), a model instance or a dict (for structured()), an AssistantTurn
# (for turn(); its stop is "tool_calls" when it carries calls, else "end") or (AssistantTurn,
# stop) to script another stop such as "length", or an exception to raise. A list is consumed one
# reply per call; a callable computes the reply from the call (a FakeCall, or a FakeTurnCall for
# turn()).
FakeReply = (
    str | BaseModel | dict[str, Any] | AssistantTurn | tuple[AssistantTurn, Stop] | BaseException
)
Script = FakeReply | list[FakeReply] | Callable[[Any], FakeReply]


class FakeModel:
    """Implements `Model`. Provenance says `backend="fake"`, so a fake answer is never mistaken
    for a model's."""

    def __init__(self, **scripts: Script) -> None:
        self._scripts = {role: list(s) if isinstance(s, list) else s for role, s in scripts.items()}
        self.calls: list[FakeCall] = []
        self.turn_calls: list[FakeTurnCall] = []

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

    def __init__(self, **scripts: Script) -> None:
        self._fake = FakeModel(**scripts)

    @property
    def calls(self) -> list[FakeCall]:
        return self._fake.calls

    @property
    def turn_calls(self) -> list[FakeTurnCall]:
        return self._fake.turn_calls

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
