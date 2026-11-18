"""backends.finish: the header rules hold for any header mapping, not only the SDK's
case-insensitive one (Codex S3)."""

import pytest

from lite_model import IgnoredParameter, RoleConfig, RolePolicy, RoleSpec
from lite_model.backends import finish
from lite_model.config import Target, resolve
from lite_model.types import Usage
from tests.configs import local, local_validate

ROLES = RoleSpec({"answer": RolePolicy(routable=True)})


def _target(backend: str, model: str) -> Target:
    config = local(roles={"answer": RoleConfig(backend=backend, model=model)})
    return resolve(config, ROLES)["answer"]


def test_mixed_case_header_names_are_read(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LITELLM_API_KEY", "k")
    headers = {"X-LiteLLM-Model-Name": "ollama_chat/qwen3:8b", "X-LiteLLM-Model-Id": "abc"}
    reply = finish(
        _target("litellm", "workhorse"),
        None,
        {"model": "workhorse"},
        headers,
        body_model="workhorse",
        usage=Usage(),
        truncated=False,
        unusable=None,
        payload="hi",
    )
    assert reply.provenance.answered == "ollama_chat/qwen3:8b"
    assert reply.provenance.via == {"x-litellm-model-id": "abc"}


def test_a_mixed_case_ignored_params_header_is_read() -> None:
    headers = {"X-LLMGW-Ignored-Params": "temperature"}
    with pytest.raises(IgnoredParameter):
        finish(
            _target("gateway", "claude-sub-haiku"),
            None,
            {"temperature": 0.2},
            headers,
            body_model="claude-sub-haiku",
            usage=Usage(),
            truncated=False,
            unusable=None,
            payload="hi",
        )


def test_a_cost_that_overflows_is_unknown_not_infinite() -> None:
    # Codex S4: finite rates times token counts can still overflow to inf.
    config = local_validate(
        {"prices": {"ollama:m": {"input_per_mtok": 1e308, "output_per_mtok": 1e308}}}
    )
    target = resolve(config.model_copy(update={"model": "m"}), ROLES)["answer"]
    reply = finish(
        target,
        None,
        {},
        {},
        body_model="m",
        usage=Usage(10**6, 10**6),
        truncated=False,
        unusable=None,
        payload="hi",
    )
    assert reply.usage.cost_usd is None


def test_a_host_keeps_its_provider_model_format_by_naming_the_backend() -> None:
    # lite-mem stores `openai:qwen3:8b` for the local Ollama today (design §6.1): a backend named
    # `openai` of type ollama keeps that tag, with Ollama's endpoint and profile.
    config = local_validate(
        {"backend": "openai", "backends": {"openai": {"type": "ollama"}}, "model": "qwen3:8b"}
    )
    target = resolve(config, ROLES)["answer"]
    assert (target.type, target.channel, target.api_key_env) == ("ollama", "local", None)
    reply = finish(
        target,
        None,
        {},
        {},
        body_model="qwen3:8b",
        usage=Usage(),
        truncated=False,
        unusable=None,
        payload="hi",
    )
    assert reply.provenance.tag == "openai:qwen3:8b"
