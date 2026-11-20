"""Design §3, §5, §8 for `turn()`: one model-chosen tool turn on both protocols, sync and async.
What the real SDKs send (tools, the history's tool calls and results, thinking blocks replayed
verbatim) and what is read back (ordered blocks, the stop, provenance, usage)."""

from dataclasses import replace

import pytest

from lite_model import (
    AssistantTurn,
    CallFailed,
    ConfigError,
    EmptyReply,
    Message,
    Opaque,
    ProvenanceUnknown,
    Text,
    ToolCall,
    ToolResult,
    ToolResults,
    ToolSpec,
    Truncated,
    TurnMessage,
    Usage,
)
from lite_model.profiles import PROFILES
from tests.configs import local
from tests.contract.conftest import Connect
from tests.contract.test_anthropic import _claude, _key  # noqa: F401  (the autouse key fixture)
from tests.contract.test_usage_and_hooks import Recorder
from tests.upstream import Upstream, completion, function_call, message, tool_use

WEATHER = ToolSpec(
    name="get_weather",
    description="Current weather for a city.",
    parameters={
        "type": "object",
        "properties": {"city": {"type": "string"}},
        "required": ["city"],
        "additionalProperties": False,
    },
)
CLOCK = ToolSpec(name="get_time", description="Current time.", parameters={"type": "object"})
TOOLS = [WEATHER, CLOCK]
ASK: list[TurnMessage] = [Message("system", "Be brief."), Message("user", "Weather in Paris?")]

THINKING = {"type": "thinking", "thinking": "", "signature": "sig/AbC+123=="}
# One finished tool round, as a caller's history holds it after an Anthropic turn.
ROUND: list[TurnMessage] = [
    *ASK,
    AssistantTurn(
        (
            Opaque("anthropic", THINKING),
            Text("Checking."),
            ToolCall("call_1", "get_weather", {"city": "Paris"}),
            ToolCall("call_2", "get_time", {}),
        )
    ),
    ToolResults((ToolResult("call_1", "sunny, 25C"), ToolResult("call_2", "boom", is_error=True))),
]


# --- OpenAI-compatible: requests ---------------------------------------------------------------


def test_openai_request_carries_the_tools(connect: Connect) -> None:
    up = Upstream((200, completion()))
    connect(up).turn("answer", ASK, TOOLS)
    assert up.body() == {
        "model": "qwen3:8b",
        "messages": [
            {"role": "system", "content": "Be brief."},
            {"role": "user", "content": "Weather in Paris?"},
        ],
        "temperature": 0.0,
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Current weather for a city.",
                    "parameters": WEATHER.parameters,
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "get_time",
                    "description": "Current time.",
                    "parameters": {"type": "object"},
                },
            },
        ],
    }


def test_openai_request_without_tools_sends_no_tools_field(connect: Connect) -> None:
    up = Upstream((200, completion()))
    connect(up).turn("answer", ASK, [])
    assert "tools" not in up.body() and "tool_choice" not in up.body()


def test_openai_history_is_tool_calls_then_one_tool_message_per_result(connect: Connect) -> None:
    up = Upstream((200, completion()))
    connect(up).turn("answer", ROUND, TOOLS)
    # The Anthropic thinking block has no place in this protocol: it is left out (design §5).
    assert up.body()["messages"][2:] == [
        {
            "role": "assistant",
            "content": "Checking.",
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "get_weather", "arguments": '{"city": "Paris"}'},
                },
                {
                    "id": "call_2",
                    "type": "function",
                    "function": {"name": "get_time", "arguments": "{}"},
                },
            ],
        },
        {"role": "tool", "tool_call_id": "call_1", "content": "sunny, 25C"},
        {"role": "tool", "tool_call_id": "call_2", "content": "boom"},
    ]


def test_openai_history_replays_unparsed_arguments_as_received(connect: Connect) -> None:
    up = Upstream((200, completion()))
    broken = AssistantTurn((ToolCall("call_1", "get_weather", None, '{"city": "Par'),))
    connect(up).turn("answer", [*ASK, broken, ToolResults((ToolResult("call_1", "bad"),))], TOOLS)
    sent = up.body()["messages"][2]
    assert sent["content"] is None
    assert sent["tool_calls"][0]["function"]["arguments"] == '{"city": "Par'


# --- OpenAI-compatible: replies ----------------------------------------------------------------


def test_openai_reply_with_two_calls_keeps_order_and_ids(connect: Connect) -> None:
    calls = [
        function_call("call_a", "get_weather", '{"city": "Paris"}'),
        function_call("call_b", "get_time", "{}"),
    ]
    up = Upstream((200, completion("Checking.", finish="tool_calls", tool_calls=calls)))
    reply = connect(up).turn("answer", ASK, TOOLS)
    assert reply.stop == "tool_calls"
    assert reply.turn.blocks == (
        Text("Checking."),
        ToolCall("call_a", "get_weather", {"city": "Paris"}),
        ToolCall("call_b", "get_time", {}),
    )
    assert reply.turn.text == "Checking."
    assert [c.id for c in reply.turn.tool_calls] == ["call_a", "call_b"]
    assert reply.provenance.tag == "ollama:qwen3:8b"
    assert reply.usage == Usage(input=7, output=3)
    assert reply.cached is False


def test_openai_reply_without_calls_ends_the_turn(connect: Connect) -> None:
    reply = connect(Upstream((200, completion("Sunny.")))).turn("answer", ASK, TOOLS)
    assert (reply.stop, reply.turn.blocks) == ("end", (Text("Sunny."),))


def test_openai_calls_with_a_plain_stop_still_mean_tool_calls(connect: Connect) -> None:
    # Some endpoints report "stop" on a turn that carries calls; the calls decide.
    calls = [function_call("call_a", "get_time", "{}")]
    up = Upstream((200, completion(None, finish="stop", tool_calls=calls)))
    reply = connect(up).turn("answer", ASK, TOOLS)
    assert (reply.stop, reply.turn.blocks) == ("tool_calls", (ToolCall("call_a", "get_time", {}),))


def test_openai_truncated_turn_returns_its_calls_instead_of_raising(connect: Connect) -> None:
    calls = [function_call("call_a", "get_weather", '{"city": "Par')]
    up = Upstream((200, completion(None, finish="length", tool_calls=calls)))
    reply = connect(up).turn("answer", ASK, TOOLS)
    assert reply.stop == "length"
    assert reply.turn.blocks == (ToolCall("call_a", "get_weather", None, '{"city": "Par'),)
    assert reply.usage == Usage(input=7, output=3)


def test_openai_arguments_that_are_not_an_object_are_not_parsed(connect: Connect) -> None:
    calls = [function_call("call_a", "get_time", "[1, 2]")]
    up = Upstream((200, completion(None, finish="tool_calls", tool_calls=calls)))
    reply = connect(up).turn("answer", ASK, TOOLS)
    assert reply.turn.blocks == (ToolCall("call_a", "get_time", None, "[1, 2]"),)


def test_openai_truncated_with_nothing_to_answer_raises_truncated(connect: Connect) -> None:
    up = Upstream((200, completion(None, finish="length")))
    with pytest.raises(Truncated) as caught:
        connect(up).turn("answer", ASK, TOOLS)
    assert caught.value.usage == Usage(input=7, output=3)


@pytest.mark.parametrize(
    "body",
    [completion("x", finish="content_filter"), completion(None), completion("  ")],
    ids=["content_filter", "no_content", "blank"],
)
def test_openai_no_usable_turn_raises_empty_reply_with_usage(
    connect: Connect, body: dict[str, object]
) -> None:
    with pytest.raises(EmptyReply) as caught:
        connect(Upstream((200, body))).turn("answer", ASK, TOOLS)
    assert caught.value.usage == Usage(input=7, output=3)


def test_openai_unknown_finish_reason_raises_empty_reply_without_quoting_it(
    connect: Connect,
) -> None:
    # Codex review 2026-11-21: a finish_reason lite-model does not know must not pass as "end".
    up = Upstream((200, completion("Some text.", finish="paused_by_vendor")))
    with pytest.raises(EmptyReply) as caught:
        connect(up).turn("answer", ASK, TOOLS)
    assert caught.value.usage == Usage(input=7, output=3)
    assert "paused_by_vendor" not in str(caught.value)


def test_openai_system_message_in_mid_history_stays_in_place(connect: Connect) -> None:
    up = Upstream((200, completion()))
    connect(up).turn("answer", [*ASK, Message("system", "Now be formal.")], TOOLS)
    assert [m["role"] for m in up.body()["messages"]] == ["system", "user", "system"]


def test_empty_tool_results_send_nothing(connect: Connect) -> None:
    # Codex review 2026-11-21: Anthropic rejects a user message with no content.
    history = [*ASK, ToolResults(()), Message("user", "Still there?")]
    openai_side, anthropic_side = Upstream((200, completion())), Upstream((200, message()))
    connect(openai_side).turn("answer", history, TOOLS)
    connect(anthropic_side, _claude()).turn("answer", history, TOOLS)
    assert [m["role"] for m in openai_side.body()["messages"]] == ["system", "user", "user"]
    assert anthropic_side.body()["messages"] == [
        {"role": "user", "content": "Weather in Paris?"},
        {"role": "user", "content": "Still there?"},
    ]


def test_anthropic_system_messages_anywhere_join_into_the_system_field(connect: Connect) -> None:
    up = Upstream((200, message()))
    history = [*ASK, Message("assistant", "Sunny."), Message("system", "Now be formal.")]
    connect(up, _claude()).turn("answer", [*history, Message("user", "And tomorrow?")], TOOLS)
    assert up.body()["system"] == "Be brief.\n\nNow be formal."
    assert [m["role"] for m in up.body()["messages"]] == ["user", "assistant", "user"]


# --- Anthropic: requests -----------------------------------------------------------------------


def test_anthropic_request_carries_the_tools_and_no_tool_choice(connect: Connect) -> None:
    up = Upstream((200, message()))
    connect(up, _claude()).turn("answer", ASK, TOOLS)
    assert up.body() == {
        "model": "claude-x",
        "max_tokens": 2048,
        "system": "Be brief.",
        "messages": [{"role": "user", "content": "Weather in Paris?"}],
        "tools": [
            {
                "name": "get_weather",
                "description": "Current weather for a city.",
                "input_schema": WEATHER.parameters,
            },
            {
                "name": "get_time",
                "description": "Current time.",
                "input_schema": {"type": "object"},
            },
        ],
    }


def test_anthropic_history_keeps_block_order_and_puts_results_in_one_user_message(
    connect: Connect,
) -> None:
    up = Upstream((200, message()))
    connect(up, _claude()).turn("answer", ROUND, TOOLS)
    assert up.body()["messages"][1:] == [
        {
            "role": "assistant",
            "content": [
                THINKING,  # byte for byte, in place (design §5)
                {"type": "text", "text": "Checking."},
                {
                    "type": "tool_use",
                    "id": "call_1",
                    "name": "get_weather",
                    "input": {"city": "Paris"},
                },
                {"type": "tool_use", "id": "call_2", "name": "get_time", "input": {}},
            ],
        },
        {
            "role": "user",
            "content": [
                {"type": "tool_result", "tool_use_id": "call_1", "content": "sunny, 25C"},
                {
                    "type": "tool_result",
                    "tool_use_id": "call_2",
                    "content": "boom",
                    "is_error": True,
                },
            ],
        },
    ]


def test_anthropic_history_leaves_out_another_protocols_opaque_content(connect: Connect) -> None:
    up = Upstream((200, message()))
    foreign = AssistantTurn((Opaque("openai", {"reasoning": "..."}), Text("Hi.")))
    connect(up, _claude()).turn("answer", [*ASK, foreign, Message("user", "And?")], TOOLS)
    assert up.body()["messages"][1] == {
        "role": "assistant",
        "content": [{"type": "text", "text": "Hi."}],
    }


# --- Anthropic: replies ------------------------------------------------------------------------


def test_anthropic_reply_blocks_come_back_in_order_with_thinking_opaque(connect: Connect) -> None:
    body = message(
        THINKING,
        {"type": "text", "text": "Checking."},
        tool_use("get_weather", {"city": "Paris"}, id="toolu_a"),
        tool_use("get_time", {}, id="toolu_b"),
        stop="tool_use",
    )
    reply = connect(Upstream((200, body)), _claude()).turn("answer", ASK, TOOLS)
    assert reply.stop == "tool_calls"
    assert reply.turn.blocks == (
        Opaque("anthropic", THINKING),
        Text("Checking."),
        ToolCall("toolu_a", "get_weather", {"city": "Paris"}),
        ToolCall("toolu_b", "get_time", {}),
    )
    assert reply.provenance.tag == "anthropic:claude-x-20261101"
    assert reply.usage == Usage(input=9, output=4)


def test_anthropic_reply_keeps_interleaved_blocks_in_the_endpoints_order(connect: Connect) -> None:
    body = message(
        {"type": "text", "text": "First."},
        THINKING,
        tool_use("get_time", {}, id="toolu_a"),
        {"type": "text", "text": "Then."},
        stop="tool_use",
    )
    reply = connect(Upstream((200, body)), _claude()).turn("answer", ASK, TOOLS)
    assert reply.turn.blocks == (
        Text("First."),
        Opaque("anthropic", THINKING),
        ToolCall("toolu_a", "get_time", {}),
        Text("Then."),
    )
    assert reply.turn.text == "First.Then."


def test_anthropic_thinking_round_trips_unchanged(connect: Connect) -> None:
    first = message(THINKING, tool_use("get_time", {}, id="toolu_a"), stop="tool_use")
    up = Upstream((200, first), (200, message()))
    model = connect(up, _claude())
    reply = model.turn("answer", ASK, TOOLS)
    done = ToolResults((ToolResult("toolu_a", "12:00"),))
    model.turn("answer", [*ASK, reply.turn, done], TOOLS)
    assert up.body()["messages"][1]["content"] == [
        THINKING,
        {"type": "tool_use", "id": "toolu_a", "name": "get_time", "input": {}},
    ]


def test_anthropic_truncated_turn_returns_its_calls_instead_of_raising(connect: Connect) -> None:
    body = message(tool_use("get_weather", {}, id="toolu_a"), stop="max_tokens")
    reply = connect(Upstream((200, body)), _claude()).turn("answer", ASK, TOOLS)
    assert (reply.stop, reply.turn.blocks) == ("length", (ToolCall("toolu_a", "get_weather", {}),))


def test_anthropic_text_only_reply_ends_the_turn(connect: Connect) -> None:
    reply = connect(Upstream((200, message())), _claude()).turn("answer", ASK, TOOLS)
    assert (reply.stop, reply.turn.blocks) == ("end", (Text("ok"),))


@pytest.mark.parametrize("stop", ["refusal", "pause_turn", "something_new"])
def test_anthropic_unusable_stop_raises_empty_reply_with_usage(connect: Connect, stop: str) -> None:
    with pytest.raises(EmptyReply) as caught:
        connect(Upstream((200, message(stop=stop))), _claude()).turn("answer", ASK, TOOLS)
    assert caught.value.usage == Usage(input=9, output=4)
    assert "something_new" not in str(caught.value)  # an endpoint's own string never leaks


def test_anthropic_thinking_alone_is_an_empty_reply(connect: Connect) -> None:
    with pytest.raises(EmptyReply):
        connect(Upstream((200, message(THINKING))), _claude()).turn("answer", ASK, TOOLS)


# --- both: failures, provenance, hooks ---------------------------------------------------------


def test_transport_failure_is_call_failed_without_upstream_text(connect: Connect) -> None:
    up = Upstream((500, "secret sk-live-123 leaked"))
    with pytest.raises(CallFailed) as caught:
        connect(up).turn("answer", ASK, TOOLS)
    assert "sk-live-123" not in str(caught.value)
    assert caught.value.usage is None


def test_litellm_turn_without_the_model_header_is_provenance_unknown(
    connect: Connect, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("LITELLM_API_KEY", "sk-proxy")
    config = local(backend="litellm", model="local-small", max_retries=0)
    with pytest.raises(ProvenanceUnknown) as caught:
        connect(Upstream((200, completion())), config).turn("answer", ASK, TOOLS)
    assert caught.value.usage is not None and caught.value.usage.input == 7


def test_litellm_turn_reads_the_answering_model_from_the_header(
    connect: Connect, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("LITELLM_API_KEY", "sk-proxy")
    config = local(backend="litellm", model="local-small", max_retries=0)
    up = Upstream((200, completion(model="local-small"), {"x-litellm-model-name": "qwen3:8b"}))
    reply = connect(up, config).turn("answer", ASK, TOOLS)
    assert reply.provenance.answered == "qwen3:8b"
    assert up.requests[0].headers["x-litellm-num-retries"] == "0"


def test_turn_does_not_run_hooks(connect: Connect) -> None:
    model = connect(Upstream((200, completion())))
    recorder = Recorder("recorder")
    model.hooks.register(recorder)
    model.turn("answer", ASK, TOOLS)
    assert (recorder.before_calls, recorder.after_calls) == ([], [])
    model.text("answer", [Message("user", "Hi")])  # the same hook does see text()
    assert (len(recorder.before_calls), len(recorder.after_calls)) == (1, 1)


def test_backend_type_without_tool_support_refuses_the_turn(
    connect: Connect, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setitem(PROFILES, "ollama", replace(PROFILES["ollama"], supports_tools=False))
    up = Upstream((200, completion()))
    with pytest.raises(ConfigError):
        connect(up).turn("answer", ASK, TOOLS)
    assert up.requests == []


def test_a_text_request_is_unchanged_by_turn_support(connect: Connect) -> None:
    up = Upstream((200, completion()))
    connect(up).text("answer", [Message("user", "Hi")])
    assert up.body() == {
        "model": "qwen3:8b",
        "messages": [{"role": "user", "content": "Hi"}],
        "temperature": 0.0,
    }
