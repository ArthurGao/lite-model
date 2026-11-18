"""Design §5, §8 for the `anthropic` backend type: what the Anthropic SDK sends (system field,
forced tool for structured output) and what is read back, sync and async. The bodies match what
lite-mem and lite-rag send today (design §12.2)."""

import pytest
from pydantic import BaseModel

from lite_model import (
    BadStructuredReply,
    CallFailed,
    EmptyReply,
    Message,
    ModelsConfig,
    Truncated,
    strict_schema,
)
from tests.configs import local_validate
from tests.contract.conftest import Connect
from tests.upstream import Upstream, completion, message, tool_use

ASK = [
    Message("system", "You extract facts."),
    Message("system", "Answer in English."),
    Message("user", "Ada lives in Wellington."),
    Message("assistant", "Noted."),
    Message("user", "Where?"),
]


class Fact(BaseModel):
    city: str
    confidence: float | None = None


@pytest.fixture(autouse=True)
def _key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")


def _claude(**fields: object) -> ModelsConfig:
    base = {"backend": "anthropic", "model": "claude-x", "max_output_tokens": 2048}
    return local_validate(base | {"max_retries": 0} | fields)


def test_text_request(connect: Connect) -> None:
    up = Upstream((200, message()))
    connect(up, _claude()).text("answer", ASK)
    request = up.requests[0]
    assert str(request.url) == "https://api.anthropic.com/v1/messages"
    # anthropic 2.6.0 sends the key as a bearer token, not `x-api-key` (checked 2026-11-18)
    assert request.headers["authorization"] == "Bearer sk-ant-test"
    assert "anthropic-version" in request.headers
    assert up.body() == {
        "model": "claude-x",
        "max_tokens": 2048,
        "system": "You extract facts.\n\nAnswer in English.",  # joined, as lite-rag does
        "messages": [
            {"role": "user", "content": "Ada lives in Wellington."},
            {"role": "assistant", "content": "Noted."},
            {"role": "user", "content": "Where?"},
        ],
    }


def test_no_system_message_sends_no_system_field(connect: Connect) -> None:
    up = Upstream((200, message()))
    connect(up, _claude()).text("answer", [Message("user", "Hi")])
    assert "system" not in up.body()


def test_structured_request_is_one_forced_tool(connect: Connect) -> None:
    up = Upstream((200, message(tool_use("Fact", {"city": "Wellington", "confidence": None}))))
    connect(up, _claude()).structured("answer", ASK, Fact)
    body = up.body()
    # lite-mem's shape exactly: the tool's input schema is the strict schema, the choice forced.
    assert body["tools"] == [{"name": "Fact", "input_schema": strict_schema(Fact)}]
    assert body["tool_choice"] == {"type": "tool", "name": "Fact"}
    assert "temperature" not in body  # not supported by anthropic 2.6.0


def test_text_reply_joins_text_blocks_and_reads_who_answered(connect: Connect) -> None:
    body = message(
        {"type": "text", "text": "In "},
        {"type": "text", "text": "Wellington."},
        model="claude-x-20261101",
        usage={"input_tokens": 31, "output_tokens": 5},
    )
    reply = connect(Upstream((200, body)), _claude()).text("answer", ASK)
    assert reply.text == "In Wellington."
    p = reply.provenance
    assert (p.backend, p.requested, p.answered, p.channel) == (
        "anthropic",
        "claude-x",
        "claude-x-20261101",
        "api",
    )
    u = reply.usage
    assert (u.input, u.output, u.reasoning, u.cost_usd) == (31, 5, None, None)


def test_structured_reply_is_the_tool_input(connect: Connect) -> None:
    body = message(
        {"type": "text", "text": "Sure."},
        tool_use("Fact", {"city": "Wellington", "confidence": 0.9}),
    )
    reply = connect(Upstream((200, body)), _claude()).structured("answer", ASK, Fact)
    assert reply.value == Fact(city="Wellington", confidence=0.9)
    assert reply.provenance.tag == "anthropic:claude-x-20261101"


@pytest.mark.parametrize("structured", [False, True])
def test_stop_at_max_tokens_is_truncated(connect: Connect, structured: bool) -> None:
    body = message(tool_use("Fact", {"city": "Wel"}), stop="max_tokens")
    caller = connect(Upstream((200, body)), _claude())
    with pytest.raises(Truncated, match=r"role answer \(anthropic:claude-x\): reply truncated"):
        if structured:
            caller.structured("answer", ASK, Fact)
        else:
            caller.text("answer", ASK)


def test_no_text_block_is_an_empty_reply(connect: Connect) -> None:
    body = message({"type": "text", "text": "  "})
    with pytest.raises(EmptyReply):
        connect(Upstream((200, body)), _claude()).text("answer", ASK)


def test_no_tool_use_is_an_empty_structured_reply(connect: Connect) -> None:
    body = message({"type": "text", "text": "I would rather not."}, stop="refusal")
    with pytest.raises(EmptyReply):
        connect(Upstream((200, body)), _claude()).structured("answer", ASK, Fact)


def test_tool_input_not_matching_the_schema_is_bad(connect: Connect) -> None:
    body = message(tool_use("Fact", {"town": "Wellington"}))
    with pytest.raises(BadStructuredReply, match="reply does not match Fact") as caught:
        connect(Upstream((200, body)), _claude()).structured("answer", ASK, Fact)
    assert "Wellington" not in str(caught.value)


def test_an_http_error_fails_without_upstream_text(connect: Connect) -> None:
    echo = {
        "type": "error",
        "error": {"type": "authentication_error", "message": "invalid x-api-key sk-ant-test"},
    }
    with pytest.raises(CallFailed) as caught:
        connect(Upstream((401, echo)), _claude()).text("answer", ASK)
    assert str(caught.value) == "role answer (anthropic:claude-x) failed: AuthenticationError"
    assert caught.value.__cause__ is not None


def test_retries_stay_on_the_same_target(connect: Connect) -> None:
    up = Upstream(
        (529, {"type": "error", "error": {"type": "overloaded_error", "message": "x"}}),
        (200, message()),
    )
    reply = connect(up, _claude(max_retries=1)).text("answer", ASK)
    assert (reply.text, len(up.requests)) == ("ok", 2)
    assert {up.body(i)["model"] for i in range(2)} == {"claude-x"}


def test_two_protocols_with_the_same_key_get_a_client_each(
    connect: Connect, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Both backends use their SDK's default URL (base_url None) and one key variable: the SDK
    # client is shared per (type, endpoint, key, timeouts), never across protocols.
    monkeypatch.setenv("ONE_KEY", "sk-one")
    config = local_validate(
        {
            "max_retries": 0,
            "backends": {
                "oa": {"type": "openai", "api_key_env": "ONE_KEY"},
                "an": {"type": "anthropic", "api_key_env": "ONE_KEY"},
            },
            "roles": {
                "answer": {"backend": "oa", "model": "gpt-x"},
                "graph": {"backend": "an", "model": "claude-x", "max_output_tokens": 64},
            },
        }
    )
    up = Upstream((200, completion()), (200, message()))
    caller = connect(up, config)
    caller.text("answer", ASK)
    caller.text("graph", ASK)
    assert [str(r.url) for r in up.requests] == [
        "https://api.openai.com/v1/chat/completions",
        "https://api.anthropic.com/v1/messages",
    ]


def test_hitting_the_context_window_is_truncated(connect: Connect) -> None:
    # Codex S2: only max_tokens was treated as truncation.
    body = message({"type": "text", "text": "partial"}, stop="model_context_window_exceeded")
    with pytest.raises(Truncated):
        connect(Upstream((200, body)), _claude()).text("answer", ASK)


@pytest.mark.parametrize("stop", ["refusal", "pause_turn"])
@pytest.mark.parametrize("structured", [False, True])
def test_a_refused_or_paused_turn_is_no_usable_reply(
    connect: Connect, stop: str, structured: bool
) -> None:
    # Codex S2: the partial text of a refused or paused turn came back as a successful reply.
    body = message(
        {"type": "text", "text": "I can't help"}, tool_use("Fact", {"city": "X"}), stop=stop
    )
    caller = connect(Upstream((200, body)), _claude())
    with pytest.raises(EmptyReply, match=f"no usable reply \\(stop reason {stop}\\)") as caught:
        if structured:
            caller.structured("answer", ASK, Fact)
        else:
            caller.text("answer", ASK)
    assert "help" not in str(caught.value)


def test_only_the_forced_tools_input_is_read(connect: Connect) -> None:
    # Codex S2: the first tool_use block was taken whatever its name.
    body = message(tool_use("Other", {"city": "Elsewhere"}), stop="tool_use")
    with pytest.raises(EmptyReply):
        connect(Upstream((200, body)), _claude()).structured("answer", ASK, Fact)
    body = message(tool_use("Other", {"city": "No"}), tool_use("Fact", {"city": "Yes"}))
    value = connect(Upstream((200, body)), _claude()).structured("answer", ASK, Fact).value
    assert value.city == "Yes"


def test_an_unknown_stop_reason_is_not_echoed(connect: Connect) -> None:
    # Codex final review: a compatible endpoint's unexpected stop_reason went into the error
    # message as is (rule 8: no upstream text in errors).
    body = message({"type": "text", "text": "x"}, stop="sk-ant-leaked-value")
    with pytest.raises(EmptyReply, match=r"stop reason unexpected") as caught:
        connect(Upstream((200, body)), _claude()).text("answer", ASK)
    assert "leaked" not in str(caught.value)
