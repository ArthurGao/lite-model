"""The `cohere` and `gemini` backend types (design §3.4, 0.8.0): embeddings over each vendor's own
REST API, which is not OpenAI-shaped. Request and reply shapes are the ones lite-rag recorded
from the vendors' API references (lite-rag tests/unit/test_embedding_vendors.py, checked
2026-11-08); a real call runs only under `-m cohere` / `-m gemini`."""

import asyncio
from typing import Any

import pytest

from lite_model import CallFailed, ConfigError, RolePolicy, RoleSpec, build, build_async
from tests.configs import local_validate
from tests.upstream import Upstream

ROLES = RoleSpec({"answer": RolePolicy(), "embed": RolePolicy(kind="embed")})


@pytest.fixture(autouse=True)
def _keys(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CO_API_KEY", "co-test")
    monkeypatch.setenv("GEMINI_API_KEY", "gm-test")


def _model(mode: str, up: Upstream, backend: str, model: str, **role: Any) -> Any:
    config = local_validate(
        {"max_retries": role.pop("max_retries", 0),
         "roles": {"embed": {"backend": backend, "model": model, **role}}}
    )  # fmt: skip
    if mode == "sync":
        return build(config, ROLES, http_client=up.client())
    return build_async(config, ROLES, http_client=up.async_client())


def _run(mode: str, call: Any) -> Any:
    return call if mode == "sync" else asyncio.run(call)


@pytest.fixture(params=["sync", "async"])
def mode(request: pytest.FixtureRequest) -> str:
    return str(request.param)


def _cohere(n: int, tokens: int = 7) -> dict[str, Any]:
    vectors = [[float(i), 1.0] for i in range(n)]
    return {"embeddings": {"float": vectors}, "meta": {"billed_units": {"input_tokens": tokens}}}


def _gemini(n: int) -> dict[str, Any]:
    return {"embeddings": [{"values": [float(i), 1.0]} for i in range(n)]}


# --- Cohere: POST /v2/embed ---


def test_cohere_sends_its_own_body_with_the_input_type(mode: str) -> None:
    up = Upstream((200, _cohere(2)))
    m = _model(mode, up, "cohere", "embed-v4.0", dimensions=2)
    reply = _run(mode, m.embed("embed", ["a", "b"], input_type="query"))
    [request] = up.requests
    assert str(request.url) == "https://api.cohere.com/v2/embed"
    assert request.headers["authorization"] == "Bearer co-test"
    assert up.body() == {
        "model": "embed-v4.0", "texts": ["a", "b"], "input_type": "search_query",
        "embedding_types": ["float"], "output_dimension": 2,
    }  # fmt: skip
    assert reply.vectors == [[0.0, 1.0], [1.0, 1.0]]
    assert (reply.usage.input, reply.usage.output) == (7, 0)
    assert reply.provenance.backend == "cohere" and reply.provenance.requested == "embed-v4.0"


def test_cohere_documents_are_search_documents() -> None:
    up = Upstream((200, _cohere(1)))
    _model("sync", up, "cohere", "embed-v4.0").embed("embed", ["a"], input_type="document")
    body = up.body()
    assert body["input_type"] == "search_document" and "output_dimension" not in body


# --- Gemini: POST /v1beta/models/{model}:batchEmbedContents ---


def test_gemini_sends_one_content_per_text_with_its_task_type(mode: str) -> None:
    up = Upstream((200, _gemini(2)))
    m = _model(mode, up, "gemini", "gemini-embedding-001", dimensions=2)
    reply = _run(mode, m.embed("embed", ["a", "b"], input_type="document"))
    [request] = up.requests
    assert str(request.url) == (
        "https://generativelanguage.googleapis.com/v1beta/models/gemini-embedding-001:batchEmbedContents"
    )
    assert request.headers["x-goog-api-key"] == "gm-test"
    assert "authorization" not in request.headers  # a key, not a bearer token
    one = {"model": "models/gemini-embedding-001", "taskType": "RETRIEVAL_DOCUMENT",
           "outputDimensionality": 2}  # fmt: skip
    assert up.body() == {
        "requests": [one | {"content": {"parts": [{"text": t}]}} for t in ("a", "b")]
    }
    assert reply.vectors == [[0.0, 1.0], [1.0, 1.0]] and reply.usage.input is None


def test_gemini_without_an_input_type_sends_no_task_type() -> None:
    """lite-rag leaves the role out for models whose profile prefixes the text instead
    (gemini-embedding-2): never both, never a guess."""
    up = Upstream((200, _gemini(1)))
    _model("sync", up, "gemini", "gemini-embedding-2").embed("embed", ["task: search | q"])
    assert "taskType" not in up.body()["requests"][0]


# --- shared: retries, halving, errors, what is refused ---


@pytest.mark.parametrize(("backend", "model", "reply"), [
    ("cohere", "embed-v4.0", _cohere), ("gemini", "gemini-embedding-001", _gemini),
])  # fmt: skip
def test_a_retryable_status_is_retried_on_the_same_target(
    mode: str, backend: str, model: str, reply: Any
) -> None:
    up = Upstream((503, {"message": "busy"}), (429, {"message": "slow down"}), (200, reply(1)))
    m = _model(mode, up, backend, model, max_retries=2)
    assert _run(mode, m.embed("embed", ["a"])).vectors == [[0.0, 1.0]]
    assert len(up.requests) == 3


@pytest.mark.parametrize(("backend", "model", "reply", "field"), [
    ("cohere", "embed-v4.0", _cohere, "texts"),
    ("gemini", "gemini-embedding-001", _gemini, "requests"),
])  # fmt: skip
def test_a_batch_refused_as_too_large_is_halved(
    backend: str, model: str, reply: Any, field: str
) -> None:
    up = Upstream((400, {"message": "too many tokens"}), (200, reply(1)), (200, reply(1)))
    m = _model("sync", up, backend, model, max_retries=3)
    assert len(m.embed("embed", ["a", "b"]).vectors) == 2
    assert [len(up.body(i)[field]) for i in range(3)] == [2, 1, 1]  # a 400 is not retried


def test_an_error_names_the_status_type_never_the_body(mode: str) -> None:
    up = Upstream((401, {"message": "invalid api token co-test"}))
    m = _model(mode, up, "cohere", "embed-v4.0", max_retries=3)
    with pytest.raises(CallFailed) as info:
        _run(mode, m.embed("embed", ["a"]))
    assert "HTTPStatusError" in str(info.value) and "co-test" not in str(info.value)
    assert len(up.requests) == 1  # a 401 is not retried


def test_a_reply_that_does_not_match_the_texts_is_bad() -> None:
    from lite_model import BadStructuredReply

    up = Upstream((200, _cohere(1)))
    with pytest.raises(BadStructuredReply, match="asked for 2 vectors, got 1"):
        _model("sync", up, "cohere", "embed-v4.0").embed("embed", ["a", "b"])


def test_neither_type_serves_chat() -> None:
    for backend in ("cohere", "gemini"):
        config = local_validate({"roles": {"answer": {"backend": backend, "model": "m"}}})
        with pytest.raises(ConfigError, match=f"{backend} does not serve chat"):
            build(config, ROLES)


def test_close_leaves_an_injected_http_client_open() -> None:
    up = Upstream((200, _cohere(1)))
    client = up.client()
    config = local_validate({"roles": {"embed": {"backend": "cohere", "model": "embed-v4.0"}}})
    with build(config, ROLES, http_client=client) as model:
        model.embed("embed", ["a"])
    assert not client.is_closed


# --- Codex review of 0.8.0 ---


def test_a_retry_after_date_is_honoured() -> None:
    from datetime import UTC, datetime, timedelta
    from email.utils import format_datetime

    from lite_model.backends.rest import _delay

    soon = format_datetime(datetime.now(UTC) + timedelta(seconds=3), usegmt=True)
    assert 1.0 < _delay(0, {"retry-after": soon}) <= 3.5
    assert _delay(0, {"retry-after": "not a date"}) == 0.5  # unreadable: the own backoff


def test_a_transport_error_carries_no_request_details() -> None:
    import httpx2

    from lite_model.backends.rest import TransportFailure

    def refuse(request: httpx2.Request) -> httpx2.Response:
        raise httpx2.ConnectError(f"refused, header was {request.headers['x-goog-api-key']}")

    client = httpx2.Client(transport=httpx2.MockTransport(refuse))
    config = local_validate(
        {"max_retries": 0, "roles": {"embed": {"backend": "gemini", "model": "m"}}}
    )
    with pytest.raises(CallFailed) as info:
        build(config, ROLES, http_client=client).embed("embed", ["a"])
    cause = info.value.__cause__
    assert isinstance(cause, TransportFailure) and str(cause) == "ConnectError"
    assert cause.__cause__ is None and cause.__suppress_context__  # the original is dropped
    assert "gm-test" not in str(info.value) + str(cause)
