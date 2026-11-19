"""Design §9 (usage on failed calls, cost) and §10 (hooks), sync and async."""

from typing import Any

import pytest
from pydantic import BaseModel

from lite_model import (
    BadStructuredReply,
    CallFailed,
    ConfigError,
    EmptyReply,
    IgnoredParameter,
    Message,
    ModelError,
    ModelsConfig,
    ProvenanceUnknown,
    Reply,
    StructuredReply,
    TextReply,
    Truncated,
    Usage,
    call_context,
)
from lite_model.hooks import Call
from tests.configs import local, local_validate
from tests.contract.conftest import Connect
from tests.upstream import Upstream, completion

ASK = [Message("user", "Hi?")]
USAGE = {"prompt_tokens": 1000, "completion_tokens": 500, "total_tokens": 1500}


class City(BaseModel):
    name: str


@pytest.fixture(autouse=True)
def _keys(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LITELLM_API_KEY", "k")


def _priced(**fields: Any) -> ModelsConfig:
    base = {
        "max_retries": 0,
        "prices": {"ollama:qwen3:8b": {"input_per_mtok": 1.0, "output_per_mtok": 2.0}},
    }
    return local_validate(base | fields)


# --- usage on failed calls (§9) ---


@pytest.mark.parametrize(
    ("body", "error"),
    [
        (completion("partial", finish="length", usage=USAGE), Truncated),
        (completion("  ", usage=USAGE), EmptyReply),
        (completion("x", finish="content_filter", usage=USAGE), EmptyReply),
    ],
)
def test_a_refused_answer_still_reports_its_usage(
    connect: Connect, body: dict[str, Any], error: type[ModelError]
) -> None:
    with pytest.raises(error) as caught:
        connect(Upstream((200, body)), _priced()).text("answer", ASK)
    u = caught.value.usage
    assert u is not None
    assert (u.input, u.output, u.cost_usd) == (1000, 500, 0.002)  # billed all the same


def test_a_bad_structured_reply_reports_its_usage(connect: Connect) -> None:
    up = Upstream((200, completion('{"nom": 1}', usage=USAGE)))
    with pytest.raises(BadStructuredReply) as caught:
        connect(up, _priced()).structured("answer", ASK, City)
    assert caught.value.usage == Usage(1000, 500, None, 0.002)


def test_header_failures_report_usage_too(connect: Connect) -> None:
    gw = {"x-llmgw-ignored-params": "temperature"}
    up = Upstream((200, completion(model="claude-sub-haiku", usage=USAGE), gw))
    config = local(backend="gateway", model="claude-sub-haiku", temperature=0.1, max_retries=0)
    with pytest.raises(IgnoredParameter) as caught:
        connect(up, config).text("answer", ASK)
    assert caught.value.usage == Usage(1000, 500, None, None)  # subscription: no cost
    up = Upstream((200, completion(model="workhorse", usage=USAGE)))
    config = local(backend="litellm", model="workhorse", max_retries=0)
    with pytest.raises(ProvenanceUnknown) as unknown:
        connect(up, config).text("answer", ASK)
    assert unknown.value.usage == Usage(1000, 500, None, None)


def test_a_transport_failure_has_no_usage(connect: Connect) -> None:
    with pytest.raises(CallFailed) as caught:
        connect(Upstream((500, {"error": {"message": "x"}})), _priced()).text("answer", ASK)
    assert caught.value.usage is None


# --- cost (§9) ---


def test_cost_from_the_price_table_for_the_model_that_answered(connect: Connect) -> None:
    reply = connect(Upstream((200, completion(usage=USAGE))), _priced()).text("answer", ASK)
    assert reply.usage.cost_usd == pytest.approx((1000 * 1.0 + 500 * 2.0) / 1e6)
    other = completion(model="qwen3:8b-q4", usage=USAGE)  # not the priced model
    assert connect(Upstream((200, other)), _priced()).text("answer", ASK).usage.cost_usd is None


def test_no_price_and_no_header_is_unknown_never_estimated(connect: Connect) -> None:
    reply = connect(Upstream((200, completion(usage=USAGE)))).text("answer", ASK)
    assert reply.usage.cost_usd is None


def test_the_proxys_own_cost_wins_over_the_price_table(connect: Connect) -> None:
    headers = {"x-litellm-model-name": "ollama_chat/qwen3:8b", "x-litellm-response-cost": "0.0042"}
    config = local_validate(
        {
            "backend": "litellm",
            "model": "workhorse",
            "max_retries": 0,
            "prices": {"litellm:ollama_chat/qwen3:8b": {"input_per_mtok": 1, "output_per_mtok": 1}},
        }
    )
    up = Upstream((200, completion(model="workhorse", usage=USAGE), headers))
    assert connect(up, config).text("answer", ASK).usage.cost_usd == 0.0042
    bad = headers | {"x-litellm-response-cost": "n/a"}  # unreadable: the table, not a guess
    up = Upstream((200, completion(model="workhorse", usage=USAGE), bad))
    assert connect(up, config).text("answer", ASK).usage.cost_usd == pytest.approx(0.0015)


def test_a_subscription_has_no_cost(connect: Connect) -> None:
    config = local_validate(
        {
            "backend": "gateway",
            "model": "claude-sub-haiku",
            "max_retries": 0,
            "prices": {"gateway:claude-sub-haiku": {"input_per_mtok": 1, "output_per_mtok": 1}},
        }
    )
    up = Upstream((200, completion(model="claude-sub-haiku", usage=USAGE)))
    assert connect(up, config).text("answer", ASK).usage.cost_usd is None


# --- hooks (§10) ---


class Recorder:
    def __init__(self, name: str, answer: Reply | None = None, fail: bool = False) -> None:
        self.name = name
        self.answer = answer
        self.fail = fail
        self.before_calls: list[Call] = []
        self.after_calls: list[tuple[Call, Reply]] = []

    def before(self, call: Call) -> Reply | None:
        self.before_calls.append(call)
        if self.fail:
            raise RuntimeError("hook broke")
        return self.answer

    def after(self, call: Call, reply: Reply) -> None:
        self.after_calls.append((call, reply))


def test_a_hook_sees_the_call_as_sent_and_the_fresh_reply(connect: Connect) -> None:
    up = Upstream((200, completion('{"name": "Auckland"}')))
    caller = connect(up)
    hook = Recorder("cache")
    caller.hooks.register(hook)
    reply = caller.structured("answer", ASK, City)
    assert reply.cached is False
    [call] = hook.before_calls
    assert (call.role, call.target.model, call.messages) == ("answer", "qwen3:8b", tuple(ASK))
    assert call.request == up.body()  # exactly what went on the wire
    assert set(call.params) == {"temperature", "response_format"}  # no model, no messages
    assert call.schema_name == "City"
    assert call.schema_hash is not None and len(call.schema_hash) == 64
    assert hook.after_calls == [(call, reply)]


def test_a_hook_reply_short_circuits_and_keeps_its_provenance(connect: Connect) -> None:
    up = Upstream((200, completion("fresh")))
    caller = connect(up)
    stored = caller.text("answer", ASK)  # a real reply, as a cache would have stored it
    hit, later = Recorder("cache", answer=stored), Recorder("later")
    caller.hooks.register(hit)
    caller.hooks.register(later)
    reply = caller.text("answer", ASK)
    assert len(up.requests) == 1  # no second request
    assert (reply.text, reply.cached, reply.provenance) == ("fresh", True, stored.provenance)
    assert later.before_calls == []  # the first hit ends the call
    assert hit.after_calls == [] and later.after_calls == []  # nothing to store on a hit


def test_after_does_not_run_for_a_failed_call(connect: Connect) -> None:
    caller = connect(Upstream((200, completion(" "))))
    hook = Recorder("cache")
    caller.hooks.register(hook)
    with pytest.raises(EmptyReply):
        caller.text("answer", ASK)
    assert (len(hook.before_calls), hook.after_calls) == (1, [])


def test_a_hook_that_raises_fails_the_call_before_sending(connect: Connect) -> None:
    up = Upstream()
    caller = connect(up)
    caller.hooks.register(Recorder("broken", fail=True))
    with pytest.raises(RuntimeError, match="hook broke"):
        caller.text("answer", ASK)
    assert up.requests == []


def test_hooks_are_found_by_name_unique_and_removable(connect: Connect) -> None:
    up = Upstream((200, completion("fresh")))
    caller = connect(up)
    stored = TextReply("from cache", caller.text("answer", ASK).provenance, Usage())
    registration = caller.hooks.register(Recorder("cache", answer=stored))
    assert caller.hooks["cache"].name == "cache"
    with pytest.raises(ConfigError, match="already registered"):
        caller.hooks.register(Recorder("cache"))
    assert caller.text("answer", ASK).cached is True
    registration.remove()
    registration.remove()  # twice is fine
    with pytest.raises(KeyError):
        caller.hooks["cache"]
    reply = caller.text("answer", ASK)
    assert (reply.cached, len(up.requests)) == (False, 2)


def test_without_hooks_no_call_is_built(connect: Connect, monkeypatch: pytest.MonkeyPatch) -> None:
    import lite_model.client as client

    def no_call(*args: object, **kwargs: object) -> None:
        raise AssertionError("Call built without hooks")

    monkeypatch.setattr(client, "Call", no_call)
    assert connect(Upstream()).text("answer", ASK).text == "ok"


class Other(BaseModel):
    name: str


@pytest.mark.parametrize("structured", [False, True])
def test_a_hook_reply_of_the_wrong_kind_fails_at_the_hook(
    connect: Connect, structured: bool
) -> None:
    # A text reply for a structured call, or a value of another schema: a hook bug, caught
    # where it happens rather than when the caller reads the value.
    caller = connect(Upstream((200, completion("fresh"))))
    text = caller.text("answer", ASK)
    wrong: Reply = (
        text if structured else StructuredReply(Other(name="x"), text.provenance, Usage())
    )
    caller.hooks.register(Recorder("cache", answer=wrong))
    with pytest.raises(TypeError, match="hook 'cache' answered role answer"):
        if structured:
            caller.structured("answer", ASK, City)
        else:
            caller.text("answer", ASK)
    other = StructuredReply(Other(name="x"), text.provenance, Usage())
    caller.hooks["cache"].answer = other  # type: ignore[union-attr]
    with pytest.raises(TypeError, match="not a StructuredReply of City"):
        caller.structured("answer", ASK, City)


class Meddler(Recorder):
    """A hook that edits what it is shown."""

    def before(self, call: Call) -> Reply | None:
        super().before(call)
        try:
            call.request["model"] = "something-else"  # type: ignore[index]
        except TypeError:
            pass
        call.request["messages"].append({"role": "user", "content": "injected"})
        return None


def test_a_hook_cannot_change_what_is_sent(connect: Connect) -> None:
    # Codex S4: Call.request was the dict the SDK then sent, so a hook could make the cache key
    # describe one request and the endpoint receive another.
    up = Upstream()
    caller = connect(up)
    caller.hooks.register(Meddler("meddler"))
    caller.text("answer", ASK)
    assert up.body()["model"] == "qwen3:8b"
    assert up.body()["messages"] == [{"role": "user", "content": "Hi?"}]
    with pytest.raises(TypeError):
        caller.hooks["meddler"].before_calls[0].request["model"] = "x"  # type: ignore[union-attr]


def test_a_wrong_hook_reply_does_not_echo_its_content(connect: Connect) -> None:
    # Codex S4: the TypeError quoted the reply (rule 8: no model output in error messages).
    caller = connect(Upstream((200, completion("secret answer text"))))
    text = caller.text("answer", ASK)
    caller.hooks.register(Recorder("cache", answer=text))
    with pytest.raises(TypeError) as caught:
        caller.structured("answer", ASK, City)
    assert "secret" not in str(caught.value)
    assert "TextReply" in str(caught.value)


def test_call_context_reaches_hooks_sync_and_async(connect: Connect) -> None:
    # The scope / cache policy a cache hook needs, without a parameter on Model (owner,
    # 2026-11-18): set around the call, read from Call.context.
    caller = connect(Upstream())
    hook = Recorder("cache")
    caller.hooks.register(hook)
    caller.text("answer", ASK)
    with call_context(scope="acme/u42", cache=True):
        caller.text("answer", ASK)
        with call_context(cache=False, role_note="inner"):
            caller.text("answer", ASK)
    caller.text("answer", ASK)
    assert [dict(c.context) for c in hook.before_calls] == [
        {},
        {"scope": "acme/u42", "cache": True},
        {"scope": "acme/u42", "cache": False, "role_note": "inner"},
        {},
    ]
    with pytest.raises(TypeError):
        hook.before_calls[1].context["scope"] = "x"  # type: ignore[index]


# --- the Proxy's own cache (0.3.0, design §7.2) ---

_PROXY = {"backend": "litellm", "model": "workhorse", "max_retries": 0}
_ANSWERED = {"x-litellm-model-name": "ollama_chat/qwen3:8b"}


@pytest.mark.parametrize(
    ("headers", "cached"),
    [
        (_ANSWERED | {"x-litellm-cache-key": "4f1c9a"}, True),
        (_ANSWERED, False),
        # LiteLLM 1.108.9 drops empty header values; an empty one is still not a hit
        (_ANSWERED | {"x-litellm-cache-key": ""}, False),
    ],
    ids=["hit", "miss", "empty"],
)
def test_a_proxy_cache_hit_is_marked_cached(
    connect: Connect, headers: dict[str, str], cached: bool
) -> None:
    # LiteLLM 1.108.9 sets the cache key only on a cached result (caching_handler.py), so the
    # header means the Proxy answered from its cache: the tokens were not spent again (§9).
    up = Upstream((200, completion(model="workhorse", usage=USAGE), headers))
    reply = connect(up, local_validate(_PROXY)).text("answer", ASK)
    assert reply.cached is cached
    assert reply.provenance.answered == "ollama_chat/qwen3:8b"  # provenance unchanged


def test_the_cache_header_means_nothing_on_another_backend(connect: Connect) -> None:
    up = Upstream((200, completion(), {"x-litellm-cache-key": "4f1c9a"}))
    assert connect(up).text("answer", ASK).cached is False
