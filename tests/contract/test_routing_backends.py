"""Design §5, §6.2, §6.3, §7.3 for the `litellm` and `gateway` backend types: the headers and
body fields sent, where the answering model is read from, and when the call raises instead of
recording a guess (AGENTS.md rule 4). Sync and async."""

import pytest
from pydantic import BaseModel

from lite_model import (
    BackendConfig,
    IgnoredParameter,
    Message,
    ModelsConfig,
    ProvenanceUnknown,
    RoleConfig,
)
from tests.contract.conftest import Connect
from tests.upstream import Upstream, completion

ASK = [Message("user", "Hi?")]
LITELLM_HEADERS = {  # what LiteLLM 1.108.9 sends (S0, KB/global/litellm §3)
    "x-litellm-model-name": "ollama_chat/qwen3:8b",
    "x-litellm-model-id": "83764671a128",
    "x-litellm-model-group": "workhorse",
    "x-litellm-key-spend": "0.0",
    "x-litellm-version": "1.108.9",
}


class City(BaseModel):
    name: str


@pytest.fixture(autouse=True)
def _key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LITELLM_API_KEY", "sk-litellm-test")


def _litellm(model: str = "workhorse", **fields: object) -> ModelsConfig:
    # `graph` is not routable (tests/contract/conftest.py ROLES): pinned, so a routing alias as
    # the default does not trip the routing guard.
    base = {"backend": "litellm", "model": model, "max_retries": 0}
    return ModelsConfig.model_validate(base | {"roles": {"graph": {"model": "workhorse"}}} | fields)


# --- litellm: request ---


def test_litellm_request_turns_retries_off_and_tags_the_role(connect: Connect) -> None:
    up = Upstream((200, completion(model="workhorse"), LITELLM_HEADERS))
    connect(up, _litellm(max_output_tokens=200)).text("answer", ASK)
    request, body = up.requests[0], up.body()
    assert str(request.url) == "http://127.0.0.1:4400/v1/chat/completions"
    assert request.headers["authorization"] == "Bearer sk-litellm-test"
    assert request.headers["x-litellm-num-retries"] == "0"  # retries stay in lite-model (§8)
    assert body["metadata"] == {"tags": ["role:answer"]}  # §7.3
    assert body["max_completion_tokens"] == 200
    assert "temperature" not in body


# --- litellm: provenance (§6.3) ---


def test_a_plain_alias_answers_with_the_model_its_header_names(connect: Connect) -> None:
    # LiteLLM restamps the body's `model` to the alias for a plain alias (S0).
    up = Upstream((200, completion(model="workhorse"), LITELLM_HEADERS))
    p = connect(up, _litellm()).text("graph", ASK).provenance
    assert (p.requested, p.answered, p.routed) == ("workhorse", "ollama_chat/qwen3:8b", False)
    assert p.tag == "litellm:ollama_chat/qwen3:8b"
    # only the allow-listed provenance headers, lower-cased
    assert p.via == {"x-litellm-model-id": "83764671a128", "x-litellm-model-group": "workhorse"}


def test_a_routed_call_answers_with_the_chosen_model(connect: Connect) -> None:
    headers = LITELLM_HEADERS | {
        "x-litellm-model-name": "ollama_chat/qwen3:1.7b",
        "x-litellm-model-group": "smart-router",
    }
    body = completion(model="ollama_chat/qwen3:1.7b")  # return_raw_model_name: true
    reply = connect(Upstream((200, body, headers)), _litellm("smart-router")).text("answer", ASK)
    p = reply.provenance
    assert (p.requested, p.answered, p.routed) == ("smart-router", "ollama_chat/qwen3:1.7b", True)
    assert p.via["x-litellm-model-group"] == "smart-router"


def test_a_routed_call_without_return_raw_model_name_raises(connect: Connect) -> None:
    # The body names the alias: the proxy is not configured as §7.2 requires.
    body = completion(model="smart-router")
    up = Upstream((200, body, LITELLM_HEADERS))
    with pytest.raises(ProvenanceUnknown, match="did not say which model answered"):
        connect(up, _litellm("smart-router")).text("answer", ASK)


@pytest.mark.parametrize("model", ["workhorse", "smart-router"])
def test_no_model_header_raises_rather_than_guessing(connect: Connect, model: str) -> None:
    up = Upstream((200, completion(model=model)))  # no x-litellm-* headers at all
    with pytest.raises(ProvenanceUnknown) as caught:
        connect(up, _litellm(model)).text("answer", ASK)
    assert str(caught.value) == (
        f"role answer (litellm:{model}): the endpoint did not say which model answered"
    )


def test_provenance_is_checked_on_structured_replies_too(connect: Connect) -> None:
    up = Upstream((200, completion('{"name": "Auckland"}', model="smart-router")))
    with pytest.raises(ProvenanceUnknown):
        connect(up, _litellm("smart-router")).structured("answer", ASK, City)


# --- gateway (§5, §6.2) ---


def _gateway(**fields: object) -> ModelsConfig:
    base = {"backend": "gateway", "model": "claude-sub-haiku", "max_retries": 0}
    return ModelsConfig.model_validate(base | fields)


GATEWAY_HEADERS = {"X-LLMGW-Profile": "claude-sub-haiku", "X-LLMGW-Backend": "claude-code"}


def test_gateway_request_has_no_proxy_extras(connect: Connect) -> None:
    up = Upstream((200, completion(model="claude-sub-haiku"), GATEWAY_HEADERS))
    connect(up, _gateway(max_output_tokens=100)).text("answer", ASK)
    request, body = up.requests[0], up.body()
    assert str(request.url) == "http://127.0.0.1:8787/v1/chat/completions"
    assert "x-litellm-num-retries" not in request.headers
    assert "metadata" not in body
    assert body["max_completion_tokens"] == 100


def test_gateway_answers_with_its_profile_and_says_which_backend(connect: Connect) -> None:
    headers = GATEWAY_HEADERS | {"X-LLMGW-Structured-Output": "prompted"}
    up = Upstream((200, completion('{"name": "Auckland"}', model="claude-sub-haiku"), headers))
    reply = connect(up, _gateway()).structured("answer", ASK, City)
    p = reply.provenance
    assert (p.answered, p.channel, p.tag) == (
        "claude-sub-haiku",
        "subscription",
        "gateway:claude-sub-haiku",
    )
    assert p.via == {"x-llmgw-backend": "claude-code", "x-llmgw-structured-output": "prompted"}


def test_a_parameter_the_gateway_ignored_raises(connect: Connect) -> None:
    headers = GATEWAY_HEADERS | {"X-LLMGW-Ignored-Params": "temperature,parallel_tool_calls"}
    up = Upstream((200, completion(model="claude-sub-haiku"), headers))
    with pytest.raises(IgnoredParameter) as caught:
        connect(up, _gateway(temperature=0.2)).text("answer", ASK)
    # only the fields lite-model sent; the gateway's list is not echoed
    assert str(caught.value) == (
        "role answer (gateway:claude-sub-haiku): the endpoint ignored temperature"
    )


def test_an_ignored_field_lite_model_did_not_send_is_not_an_error(connect: Connect) -> None:
    headers = GATEWAY_HEADERS | {"X-LLMGW-Ignored-Params": "parallel_tool_calls"}
    up = Upstream((200, completion("hi", model="claude-sub-haiku"), headers))
    assert connect(up, _gateway(temperature=0.2)).text("answer", ASK).text == "hi"


def test_the_ignored_check_covers_the_structured_output_format(connect: Connect) -> None:
    headers = GATEWAY_HEADERS | {"X-LLMGW-Ignored-Params": "response_format"}
    up = Upstream((200, completion('{"name": "A"}', model="claude-sub-haiku"), headers))
    with pytest.raises(IgnoredParameter, match="ignored response_format"):
        connect(up, _gateway()).structured("answer", ASK, City)


def test_other_openai_compatible_backends_read_no_proxy_headers(connect: Connect) -> None:
    # A named `openai` backend in front of something that happens to send LiteLLM headers:
    # its profile has no answered header, so the body's model is used and nothing is in `via`.
    config = ModelsConfig(
        backend="corp",
        model="m",
        max_retries=0,
        backends={"corp": BackendConfig(type="ollama", base_url="http://corp.example/v1")},
        roles={"answer": RoleConfig()},
    )
    up = Upstream((200, completion(model="m-2"), LITELLM_HEADERS))
    p = connect(up, config).text("answer", ASK).provenance
    assert (p.answered, p.via) == ("m-2", {})


def test_a_header_naming_only_the_alias_is_not_an_answer(connect: Connect) -> None:
    # Codex S3: the header was taken as is even when it repeated the requested alias, which says
    # nothing about the deployment that served (rule 4).
    headers = LITELLM_HEADERS | {"x-litellm-model-name": "workhorse"}
    up = Upstream((200, completion(model="workhorse"), headers))
    with pytest.raises(ProvenanceUnknown):
        connect(up, _litellm()).text("graph", ASK)
