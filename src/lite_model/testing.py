"""`FakeModel` and `AsyncFakeModel` (design §11): scripted replies per role, every call
recorded; no network."""

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel

from lite_model.types import Message, Provenance, StructuredReply, TextReply, Usage


@dataclass(frozen=True)
class FakeCall:
    role: str
    messages: tuple[Message, ...]
    schema: type[BaseModel] | None  # None for text()


# A reply: text (for text()), a model instance or a dict (for structured()), or an exception to
# raise. A list is consumed one reply per call; a callable computes the reply from the call.
Reply = str | BaseModel | dict[str, Any] | BaseException
Script = Reply | list[Reply] | Callable[[FakeCall], Reply]


class FakeModel:
    """Implements `Model`. Provenance says `backend="fake"`, so a fake answer is never mistaken
    for a model's."""

    def __init__(self, **scripts: Script) -> None:
        self._scripts = {role: list(s) if isinstance(s, list) else s for role, s in scripts.items()}
        self.calls: list[FakeCall] = []

    def _reply(self, call: FakeCall) -> Reply:
        self.calls.append(call)
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

    def text(self, role: str, messages: Sequence[Message]) -> TextReply:
        reply = self._reply(FakeCall(role, tuple(messages), None))
        if not isinstance(reply, str):
            raise AssertionError(
                f"FakeModel: role {role!r} scripted {type(reply).__name__}, text() needs str"
            )
        return TextReply(reply, self._provenance(role), Usage())

    def structured[T: BaseModel](
        self, role: str, messages: Sequence[Message], schema: type[T]
    ) -> StructuredReply[T]:
        reply = self._reply(FakeCall(role, tuple(messages), schema))
        if isinstance(reply, str):
            raise AssertionError(
                f"FakeModel: role {role!r} scripted str, structured() needs an object"
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

    async def text(self, role: str, messages: Sequence[Message]) -> TextReply:
        return self._fake.text(role, messages)

    async def structured[T: BaseModel](
        self, role: str, messages: Sequence[Message], schema: type[T]
    ) -> StructuredReply[T]:
        return self._fake.structured(role, messages, schema)
