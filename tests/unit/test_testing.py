"""Design §11: FakeModel scripts replies per role and records every call."""

import asyncio

import pytest
from pydantic import BaseModel

from lite_model import (
    AssistantTurn,
    AsyncModel,
    CallFailed,
    EmptyReply,
    Message,
    Model,
    Text,
    ToolCall,
    ToolResult,
    ToolResults,
    ToolSpec,
    TurnMessage,
)
from lite_model.testing import AsyncFakeModel, FakeCall, FakeModel

ASK = [Message("user", "Hi?")]


class Verdict(BaseModel):
    ok: bool


def test_fake_model_is_a_model() -> None:
    model: Model = FakeModel(answer="hi")
    assert model.text("answer", ASK).text == "hi"


def test_scripts_a_fixed_reply_a_sequence_and_a_function() -> None:
    fake = FakeModel(
        answer="always",
        graph=["first", "second"],
        judge=lambda call: Verdict(ok=call.messages[-1].content == "yes"),
    )
    assert [fake.text("answer", ASK).text for _ in range(2)] == ["always", "always"]
    assert [fake.text("graph", ASK).text for _ in range(2)] == ["first", "second"]
    with pytest.raises(AssertionError, match="ran out of replies for role 'graph'"):
        fake.text("graph", ASK)
    assert fake.structured("judge", [Message("user", "yes")], Verdict).value == Verdict(ok=True)
    assert fake.structured("judge", [Message("user", "no")], Verdict).value == Verdict(ok=False)


def test_structured_accepts_a_dict_and_validates_it_against_the_schema() -> None:
    fake = FakeModel(judge={"ok": True}, bad={"ok": "maybe"})
    assert fake.structured("judge", ASK, Verdict).value == Verdict(ok=True)
    with pytest.raises(ValueError):
        fake.structured("bad", ASK, Verdict)


def test_a_scripted_exception_is_raised() -> None:
    fake = FakeModel(answer=EmptyReply("answer: empty reply"))
    with pytest.raises(EmptyReply):
        fake.text("answer", ASK)


def test_records_every_call_in_order() -> None:
    fake = FakeModel(answer="hi", judge={"ok": True})
    fake.text("answer", ASK)
    fake.structured("judge", [Message("system", "s"), *ASK], Verdict)
    assert fake.calls == [
        FakeCall("answer", tuple(ASK), None),
        FakeCall("judge", (Message("system", "s"), *ASK), Verdict),
    ]


def test_provenance_says_fake() -> None:
    p = FakeModel(answer="hi").text("answer", ASK).provenance
    assert (p.role, p.backend, p.answered, p.channel, p.routed, p.tag) == (
        "answer",
        "fake",
        "fake-model",
        "local",
        False,
        "fake:fake-model",
    )


def test_an_unscripted_role_or_wrong_reply_type_fails_loudly() -> None:
    fake = FakeModel(answer={"ok": True}, judge="text")
    with pytest.raises(AssertionError, match="no reply scripted for role 'graph'"):
        fake.text("graph", ASK)
    with pytest.raises(AssertionError, match="text\\(\\) needs str"):
        fake.text("answer", ASK)
    with pytest.raises(AssertionError, match="structured\\(\\) needs an object"):
        fake.structured("judge", ASK, Verdict)


def test_async_fake_model_scripts_and_records_like_fake_model() -> None:
    fake = AsyncFakeModel(answer=["one", EmptyReply("answer: empty")], judge={"ok": True})
    model: AsyncModel = fake

    async def calls() -> tuple[str, Verdict]:
        text = (await model.text("answer", ASK)).text
        verdict = (await model.structured("judge", ASK, Verdict)).value
        with pytest.raises(EmptyReply):
            await model.text("answer", ASK)
        return text, verdict

    assert asyncio.run(calls()) == ("one", Verdict(ok=True))
    assert [(c.role, c.schema) for c in fake.calls] == [
        ("answer", None),
        ("judge", Verdict),
        ("answer", None),
    ]


# --- turn() (design §11) ---


def test_fake_turn_scripts_turns_and_records_the_tools() -> None:
    clock = ToolSpec("get_time", "Current time.", {"type": "object"})
    call = AssistantTurn((ToolCall("c1", "get_time", {}),))
    done = AssistantTurn((Text("It is noon."),))
    fake = FakeModel(agent=[call, done])
    ask = [Message("user", "Time?")]

    first = fake.turn("agent", ask, [clock])
    assert (first.stop, first.turn) == ("tool_calls", call)
    assert first.provenance.backend == "fake"
    history: list[TurnMessage] = [*ask, first.turn, ToolResults((ToolResult("c1", "12:00"),))]
    second = fake.turn("agent", history, [clock])
    assert (second.stop, second.turn.text) == ("end", "It is noon.")

    assert [c.tools for c in fake.turn_calls] == [(clock,), (clock,)]
    assert fake.turn_calls[1].messages == tuple(history)
    assert fake.calls == []  # text() / structured() calls only


def test_fake_turn_scripts_a_truncated_turn_and_an_error() -> None:
    cut = AssistantTurn((ToolCall("c1", "get_time", None, '{"tz": "U'),))
    fake = FakeModel(agent=[(cut, "length"), CallFailed("agent failed: TimeoutError")])
    assert fake.turn("agent", [Message("user", "Time?")], []).stop == "length"
    with pytest.raises(CallFailed):
        fake.turn("agent", [Message("user", "Time?")], [])


def test_fake_turn_refuses_a_reply_of_the_wrong_kind() -> None:
    fake = FakeModel(agent="just text", answer=AssistantTurn((Text("hi"),)))
    with pytest.raises(AssertionError, match="turn\\(\\) needs an AssistantTurn"):
        fake.turn("agent", [Message("user", "Hi")], [])
    with pytest.raises(AssertionError, match="text\\(\\) needs str"):
        fake.text("answer", [Message("user", "Hi")])


def test_async_fake_turn_shares_the_script_and_the_record() -> None:
    from lite_model.testing import AsyncFakeModel

    fake = AsyncFakeModel(agent=AssistantTurn((Text("hi"),)))
    reply = asyncio.run(fake.turn("agent", [Message("user", "Hi")], []))
    assert (reply.stop, reply.turn.text) == ("end", "hi")
    assert len(fake.turn_calls) == 1
