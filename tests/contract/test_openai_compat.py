"""Design §5, §6, §8: what the OpenAI SDK sends for the ollama and openai profiles, what is read
back, and every error. The SDK is real; only its transport is replaced (tests/upstream.py)."""

import asyncio
import traceback
from typing import Any

import pytest
from pydantic import BaseModel

from lite_model import (
    BadStructuredReply,
    CallFailed,
    EmptyReply,
    Message,
    ModelsConfig,
    RoleConfig,
    Truncated,
    build,
    build_async,
    strict_schema,
)
from tests.configs import local, local_validate, two_vendors
from tests.contract.conftest import ROLES, Connect
from tests.upstream import Upstream, completion

ASK = [Message("system", "Be brief."), Message("user", "Hi?")]


class City(BaseModel):
    name: str
    population: int | None = None


def _openai(**fields: Any) -> ModelsConfig:
    """The `openai` profile. Nothing leaves the process: the transport is the fake upstream."""
    return local(backend="openai", model="gpt-x", max_retries=0, **fields)


# --- request bodies ---


def test_ollama_text_request(connect: Connect) -> None:
    up = Upstream()
    connect(up).text("answer", ASK)
    request = up.requests[0]
    assert str(request.url) == "http://127.0.0.1:11434/v1/chat/completions"
    assert up.body() == {
        "model": "qwen3:8b",
        "messages": [
            {"role": "system", "content": "Be brief."},
            {"role": "user", "content": "Hi?"},
        ],
        "temperature": 0.0,  # the ollama profile's default
    }


def test_ollama_sends_max_tokens_and_reasoning_effort_when_configured(connect: Connect) -> None:
    up = Upstream()
    config = local(
        max_retries=0,
        roles={
            "graph": RoleConfig(max_output_tokens=2048, reasoning_effort="none", temperature=0.3),
        },
    )
    connect(up, config).text("graph", ASK)
    body = up.body()
    assert (body["max_tokens"], body["reasoning_effort"], body["temperature"]) == (
        2048,
        "none",
        0.3,
    )
    assert "max_completion_tokens" not in body  # Ollama ignores it (design §5)


def test_openai_profile_request(connect: Connect, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-openai")
    up = Upstream()
    connect(up, _openai(max_output_tokens=300)).text("answer", ASK)
    request, body = up.requests[0], up.body()
    assert str(request.url) == "https://api.openai.com/v1/chat/completions"
    assert request.headers["authorization"] == "Bearer sk-test-openai"
    assert body["max_completion_tokens"] == 300
    assert "max_tokens" not in body
    assert "temperature" not in body  # reasoning models reject a non-default value


def test_structured_request_carries_the_strict_schema(connect: Connect) -> None:
    up = Upstream((200, completion('{"name": "Auckland", "population": 1700000}')))
    connect(up).structured("answer", ASK, City)
    assert up.body()["response_format"] == {
        "type": "json_schema",
        "json_schema": {"name": "City", "schema": strict_schema(City), "strict": True},
    }  # strict_schema itself: tests/unit/test_schema.py


def test_named_backends_reach_their_own_endpoints_with_their_own_keys(
    connect: Connect,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-deepseek")
    monkeypatch.setenv("DASHSCOPE_API_KEY", "sk-dashscope")
    up = Upstream()
    config = two_vendors(max_retries=0)
    model = connect(up, config)
    model.text("answer", ASK)
    model.text("graph", ASK)
    sent = [(str(r.url), r.headers["authorization"]) for r in up.requests]
    assert sent == [
        ("https://deepseek.example/v1/chat/completions", "Bearer sk-deepseek"),
        ("https://dashscope.example/v1/chat/completions", "Bearer sk-dashscope"),
    ]
    assert [up.body(0)["model"], up.body(1)["model"]] == ["deepseek-chat", "qwen-max"]


# --- replies: provenance and usage ---


def test_text_reply_provenance_is_read_from_the_response(connect: Connect) -> None:
    up = Upstream((200, completion("hello", model="qwen3:8b-q4")))
    reply = connect(up).text("answer", ASK)
    assert reply.text == "hello"
    p = reply.provenance
    assert (p.role, p.backend, p.requested, p.answered) == (
        "answer",
        "ollama",
        "qwen3:8b",
        "qwen3:8b-q4",
    )
    assert (p.channel, p.routed, p.tag) == ("local", False, "ollama:qwen3:8b-q4")


def test_an_endpoint_that_does_not_say_who_answered_is_marked_unconfirmed(connect: Connect) -> None:
    up = Upstream((200, completion("hello", model="")))
    p = connect(up).text("answer", ASK).provenance
    assert (p.answered, p.tag) == (None, "ollama:qwen3:8b?")


def test_usage_includes_reasoning_tokens_when_reported(connect: Connect) -> None:
    usage = {
        "prompt_tokens": 11,
        "completion_tokens": 40,
        "total_tokens": 51,
        "completion_tokens_details": {"reasoning_tokens": 32},
    }
    up = Upstream((200, completion("hi", usage=usage)))
    u = connect(up).text("answer", ASK).usage
    assert (u.input, u.output, u.reasoning, u.cost_usd) == (11, 40, 32, None)
    plain = connect(Upstream()).text("answer", ASK).usage
    assert (plain.input, plain.output, plain.reasoning) == (7, 3, None)


def test_structured_reply_value(connect: Connect) -> None:
    up = Upstream((200, completion('{"name": "Auckland", "population": null}')))
    reply = connect(up).structured("answer", ASK, City)
    assert reply.value == City(name="Auckland", population=None)
    assert reply.provenance.tag == "ollama:qwen3:8b"


# --- errors (design §8) ---


def test_a_reply_cut_at_the_limit_is_truncated_even_with_text(connect: Connect) -> None:
    up = Upstream((200, completion("partial answ", finish="length")))
    with pytest.raises(Truncated, match=r"role answer \(ollama:qwen3:8b\): reply truncated"):
        connect(up).text("answer", ASK)


@pytest.mark.parametrize("content", [None, "", "  \n"])
def test_no_text_is_an_empty_reply(connect: Connect, content: str | None) -> None:
    up = Upstream((200, completion(content)))
    with pytest.raises(EmptyReply, match="empty reply"):
        connect(up).text("answer", ASK)


def test_no_choices_is_an_empty_reply(connect: Connect) -> None:
    body = completion() | {"choices": []}
    with pytest.raises(EmptyReply):
        connect(Upstream((200, body))).text("answer", ASK)


@pytest.mark.parametrize("content", ['{"name": 5}', "not json", '{"population": 3}'])
def test_a_reply_not_matching_the_schema_is_bad(connect: Connect, content: str) -> None:
    up = Upstream((200, completion(content)))
    with pytest.raises(BadStructuredReply, match="reply does not match City") as caught:
        connect(up).structured("answer", ASK, City)
    assert content not in str(caught.value)  # model output is not echoed


def test_an_http_error_fails_without_upstream_text(
    connect: Connect, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-secret")
    echo = {"error": {"message": "Incorrect API key provided: sk-test-secret at https://u:p@x"}}
    up = Upstream((401, echo))
    with pytest.raises(CallFailed) as caught:
        connect(up, _openai()).text("answer", ASK)
    message = str(caught.value)
    assert message == "role answer (openai:gpt-x) failed: AuthenticationError"
    assert caught.value.__cause__ is not None  # the original stays chained


def test_retries_stay_on_the_same_target_then_fail(connect: Connect) -> None:
    up = Upstream((500, {"error": {"message": "boom"}}))
    config = local(max_retries=1)
    with pytest.raises(CallFailed, match="InternalServerError"):
        connect(up, config).text("answer", ASK)
    assert len(up.requests) == 2  # one retry
    assert {up.body(i)["model"] for i in range(2)} == {"qwen3:8b"}  # never another model


def test_a_retry_that_succeeds_returns_the_reply(connect: Connect) -> None:
    up = Upstream((500, {"error": {"message": "boom"}}), (200, completion("second")))
    reply = connect(up, local(max_retries=1)).text("answer", ASK)
    assert (reply.text, len(up.requests)) == ("second", 2)


def test_an_undeclared_role_at_call_time_is_a_config_error(connect: Connect) -> None:
    from lite_model import ConfigError

    with pytest.raises(ConfigError, match="role summary: not declared"):
        connect(Upstream()).text("summary", ASK)


@pytest.mark.parametrize("mode", ["sync", "async"])
def test_a_test_without_the_fake_transport_cannot_reach_a_server(mode: str) -> None:
    # tests/conftest.py refuses sockets outside -m ollama / -m litellm: a contract test that
    # forgot `http_client` fails instead of calling the real Ollama.
    config = local(max_retries=0)
    with pytest.raises(CallFailed) as caught:
        if mode == "sync":
            build(config, ROLES).text("answer", ASK)
        else:
            asyncio.run(build_async(config, ROLES).text("answer", ASK))
    # the cause chain, and inside it the anyio ExceptionGroup of the async transport
    assert "tried to open a network connection" in "".join(traceback.format_exception(caught.value))


def test_a_filtered_reply_is_no_usable_reply(connect: Connect) -> None:
    # Codex S2 (the same gap on this side): content cut by the provider's filter came back as
    # a successful reply.
    up = Upstream((200, completion("I can't help with th", finish="content_filter")))
    with pytest.raises(EmptyReply, match=r"no usable reply \(stop reason content_filter\)"):
        connect(up).text("answer", ASK)


def test_a_keyless_openai_endpoint_sends_the_placeholder_key(connect: Connect) -> None:
    # 0.1.1: the SDK insists on a key; a keyless endpoint gets "unused", and no temperature.
    up = Upstream()
    config = local_validate(
        {
            "backend": "openai",
            "max_retries": 0,
            "backends": {
                "openai": {
                    "type": "openai",
                    "base_url": "http://127.0.0.1:11434/v1",
                    "keyless": True,
                }
            },
        }
    )
    connect(up, config).text("answer", ASK)
    assert up.requests[0].headers["authorization"] == "Bearer unused"
    assert "temperature" not in up.body()
