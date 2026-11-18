"""Opt-in: the real Ollama on the host (`uv run pytest -m ollama`). Free and local; the config
forbids the api and subscription channels so nothing else can be reached (AGENTS.md rule 11)."""

import pytest
from pydantic import BaseModel

from lite_model import Message, ModelsConfig, RolePolicy, RoleSpec, Truncated, build

pytestmark = pytest.mark.ollama

CONFIG = ModelsConfig(
    model="qwen3:1.7b", reasoning_effort="none", forbid_channels=("subscription", "api")
)
ROLES = RoleSpec({"answer": RolePolicy()})


class Sum(BaseModel):
    result: int


def test_text_through_ollama() -> None:
    reply = build(CONFIG, ROLES).text("answer", [Message("user", "Reply with the word: pong")])
    assert "pong" in reply.text.lower()
    p = reply.provenance
    assert (p.backend, p.answered, p.channel, p.routed) == ("ollama", "qwen3:1.7b", "local", False)
    assert reply.usage.input is not None and reply.usage.input > 0
    assert reply.usage.output is not None and reply.usage.output > 0


def test_structured_through_ollama() -> None:
    ask = [Message("user", "What is 17 + 25? Answer in JSON.")]
    reply = build(CONFIG, ROLES).structured("answer", ask, Sum)
    assert reply.value == Sum(result=42)
    assert reply.provenance.tag == "ollama:qwen3:1.7b"


def test_the_output_limit_raises_truncated() -> None:
    config = CONFIG.model_copy(update={"max_output_tokens": 5})
    with pytest.raises(Truncated):
        build(config, ROLES).text("answer", [Message("user", "Count from 1 to 100.")])
