"""Design §11: FakeModel scripts replies per role and records every call."""

import pytest
from pydantic import BaseModel

from lite_model import EmptyReply, Message, Model
from lite_model.testing import FakeCall, FakeModel

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
