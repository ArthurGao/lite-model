"""Opt-in: the real Ollama on the host (`uv run pytest -m ollama`). Free and local; the config
forbids the api and subscription channels so nothing else can be reached (AGENTS.md rule 11)."""

import asyncio

import pytest
from pydantic import BaseModel

from lite_model import (
    Message,
    ModelsConfig,
    RolePolicy,
    RoleSpec,
    ToolResult,
    ToolResults,
    ToolSpec,
    Truncated,
    TurnMessage,
    build,
    build_async,
)

pytestmark = pytest.mark.ollama

CONFIG = ModelsConfig(
    backend="ollama",
    model="qwen3:1.7b",
    reasoning_effort="none",
    forbid_channels=("subscription", "api"),
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


def test_async_structured_through_ollama() -> None:
    ask = [Message("user", "What is 17 + 25? Answer in JSON.")]
    reply = asyncio.run(build_async(CONFIG, ROLES).structured("answer", ask, Sum))
    assert reply.value == Sum(result=42)
    assert reply.provenance.tag == "ollama:qwen3:1.7b"


def test_a_tool_turn_and_its_result_through_ollama() -> None:
    clock = ToolSpec(
        name="get_time",
        description="Returns the current local time. Call it whenever the user asks for the time.",
        parameters={"type": "object", "properties": {}, "additionalProperties": False},
    )
    ask: list[TurnMessage] = [Message("user", "What time is it right now? Use the tool.")]
    model = build(CONFIG, ROLES)

    first = model.turn("answer", ask, [clock])
    assert first.stop == "tool_calls"
    (call,) = first.turn.tool_calls
    assert (call.name, call.arguments) == ("get_time", {})
    assert first.provenance.tag == "ollama:qwen3:1.7b"

    done = ToolResults((ToolResult(call.id, "The time is 12:34."),))
    second = model.turn("answer", [*ask, first.turn, done], [clock])
    assert second.stop == "end"
    assert "12:34" in second.turn.text
