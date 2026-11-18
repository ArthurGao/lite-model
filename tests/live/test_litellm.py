"""Opt-in: the local LiteLLM Proxy (`uv run pytest -m litellm`; deploy/litellm/). Ollama-backed
aliases only, so nothing is paid (AGENTS.md rule 11): the backend is declared `channel: local`
and the config forbids the api and subscription channels."""

import asyncio
import os
from pathlib import Path

import pytest
from dotenv import dotenv_values
from pydantic import BaseModel

from lite_model import (
    BackendConfig,
    Message,
    ModelsConfig,
    RoleConfig,
    RolePolicy,
    RoleSpec,
    build,
    build_async,
)

pytestmark = pytest.mark.litellm

ENV = Path(__file__).resolve().parents[2] / "deploy" / "litellm" / ".env"
ROLES = RoleSpec({"answer": RolePolicy(routable=True), "graph": RolePolicy()})
CONFIG = ModelsConfig(
    backend="proxy",
    model="smart-router",
    reasoning_effort="none",
    max_retries=0,
    forbid_channels=("subscription", "api"),
    backends={
        "proxy": BackendConfig(type="litellm", channel="local", router_aliases=("smart-router",))
    },
    roles={"graph": RoleConfig(model="workhorse")},
)
COMPLEX = (
    "Step by step, prove the algorithm below is O(n log n), refactor it to use a heap, and explain "
    "the distributed consensus trade-offs and kubernetes deployment:\n"
    "```python\ndef f(xs):\n    for i in range(len(xs)):\n        xs.sort()\n    return xs\n```"
)


@pytest.fixture(autouse=True)
def _key(monkeypatch: pytest.MonkeyPatch) -> None:
    """The proxy's master key from deploy/litellm/.env (gitignored), unless already set."""
    if os.environ.get("LITELLM_API_KEY"):
        return
    key = dotenv_values(ENV).get("LITELLM_MASTER_KEY") if ENV.exists() else None
    assert key, f"set LITELLM_API_KEY or LITELLM_MASTER_KEY in {ENV} (deploy/litellm/README.md)"
    monkeypatch.setenv("LITELLM_API_KEY", key)


class Sum(BaseModel):
    result: int


def test_a_plain_alias_records_the_deployments_model() -> None:
    reply = build(CONFIG, ROLES).text("graph", [Message("user", "Reply with the word: pong")])
    p = reply.provenance
    assert (p.requested, p.answered, p.routed) == ("workhorse", "ollama_chat/qwen3:8b", False)
    assert p.via["x-litellm-model-group"] == "workhorse"
    assert "pong" in reply.text.lower()


def test_routing_records_the_model_the_router_chose() -> None:
    model = build(CONFIG, ROLES)
    simple = model.text("answer", [Message("user", "hi")]).provenance
    hard = model.text("answer", [Message("user", COMPLEX)]).provenance
    assert (simple.requested, simple.routed) == ("smart-router", True)
    assert simple.answered == "ollama_chat/qwen3:1.7b"  # SIMPLE tier
    assert hard.answered == "ollama_chat/qwen3:8b"  # a higher tier
    assert simple.via["x-litellm-model-group"] == "smart-router"


def test_structured_through_the_router_async() -> None:
    # Arithmetic, not a fact: the router may pick qwen3:1.7b, which once named Wellington as New
    # Zealand's largest city. This tests the path, not the model's knowledge.
    ask = [Message("user", "What is 17 + 25? Answer in JSON.")]
    reply = asyncio.run(build_async(CONFIG, ROLES).structured("answer", ask, Sum))
    assert reply.value == Sum(result=42)
    assert reply.provenance.routed is True
    assert reply.provenance.answered in {"ollama_chat/qwen3:1.7b", "ollama_chat/qwen3:8b"}
