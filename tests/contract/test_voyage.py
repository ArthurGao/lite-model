"""The `voyage` backend type (design §3.3, 0.7.0): embeddings with an input type and an output
dimension, and rerank. Voyage AI has no chat API. Request and reply shapes are Voyage's published
API reference (docs.voyageai.com, 2025-12-10); a real call runs only under `-m voyage`."""

import asyncio
import json
from typing import Any

import pytest

from lite_model import (
    BadStructuredReply,
    Bounded,
    ConfigError,
    ModelsConfig,
    RerankReply,
    RolePolicy,
    RoleSpec,
    build,
    build_async,
)
from lite_model.testing import AsyncFakeModel, FakeEmbedCall, FakeModel, FakeRerankCall
from tests.configs import local_validate
from tests.upstream import Upstream, embeddings, ranked

ROLES = RoleSpec(
    {
        "answer": RolePolicy(),
        "embed": RolePolicy(kind="embed"),
        "rerank": RolePolicy(kind="rerank"),
    }
)


@pytest.fixture(autouse=True)
def _key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("VOYAGE_API_KEY", "pa-test")


def _config(**roles: Any) -> ModelsConfig:
    return local_validate(
        {
            "max_retries": 0,
            "roles": {
                "embed": {"backend": "voyage", "model": "voyage-4", **roles.pop("embed", {})},
                "rerank": {"backend": "voyage", "model": "rerank-2.5", **roles.pop("rerank", {})},
                **roles,
            },
        }
    )


def _vectors(*vectors: list[float], b64: bool = False) -> dict[str, Any]:
    return embeddings(*vectors, model="voyage-4", usage={"total_tokens": 8}, b64=b64)


def _model(mode: str, up: Upstream, config: ModelsConfig | None = None) -> Any:
    config = config or _config()
    if mode == "sync":
        return build(config, ROLES, http_client=up.client())
    return build_async(config, ROLES, http_client=up.async_client())


def _run(mode: str, call: Any) -> Any:
    return call if mode == "sync" else asyncio.run(call)


@pytest.fixture(params=["sync", "async"])
def mode(request: pytest.FixtureRequest) -> str:
    return str(request.param)


# --- embeddings ---


def test_the_embed_request_is_voyages_with_the_input_type(mode: str) -> None:
    up = Upstream((200, _vectors([0.5, 0.25])))
    m = _model(mode, up, _config(embed={"dimensions": 256}))
    reply = _run(mode, m.embed("embed", ["refunds"], input_type="query"))
    request = up.requests[0]
    assert str(request.url) == "https://api.voyageai.com/v1/embeddings"
    assert request.headers["authorization"] == "Bearer pa-test"
    body = json.loads(request.content)
    assert (body["model"], body["input"], body["input_type"], body["output_dimension"]) == (
        "voyage-4", ["refunds"], "query", 256,
    )  # fmt: skip
    assert "dimensions" not in body  # Voyage's field is output_dimension
    assert body.get("encoding_format") != "float"  # Voyage takes null or base64 only
    assert reply.vectors == [[0.5, 0.25]] and reply.provenance.tag == "voyage:voyage-4"
    assert reply.usage.input == 8  # Voyage reports total_tokens


def test_base64_vectors_are_decoded(mode: str) -> None:
    up = Upstream((200, _vectors([0.5, -1.0], b64=True)))
    reply = _run(mode, _model(mode, up).embed("embed", ["x"], input_type="document"))
    assert reply.vectors == [[0.5, -1.0]]


def test_no_input_type_sends_none() -> None:
    up = Upstream((200, _vectors([1.0])))
    _model("sync", up).embed("embed", ["x"])
    assert "input_type" not in up.body(0)


def test_a_batch_voyage_refuses_as_too_large_is_halved_with_its_input_type() -> None:
    too_large = (400, {"detail": "Batch size is too large"})
    up = Upstream(too_large, (200, _vectors([1.0])), (200, _vectors([2.0])))
    reply = _model("sync", up).embed("embed", ["a", "b"], input_type="document")
    assert reply.vectors == [[1.0], [2.0]]
    sent = [up.body(i) for i in range(len(up.requests))]
    assert [(b["input"], b["input_type"]) for b in sent] == [
        (["a", "b"], "document"), (["a"], "document"), (["b"], "document"),
    ]  # fmt: skip


def test_an_input_type_on_a_backend_without_one_is_refused() -> None:
    up = Upstream((200, _vectors([1.0])))
    config = _config(embed={"backend": "ollama", "model": "bge-m3:567m"})
    with pytest.raises(ConfigError, match="input_type"):
        build(config, ROLES, http_client=up.client()).embed("embed", ["x"], input_type="query")
    assert up.requests == []


def test_a_chat_role_on_voyage_is_refused_at_build() -> None:
    with pytest.raises(ConfigError, match="voyage does not serve chat"):
        build(_config(answer={"backend": "voyage", "model": "voyage-4"}), ROLES)


# --- rerank ---


def test_rerank_scores_come_back_in_input_order(mode: str) -> None:
    up = Upstream((200, ranked((2, 0.9), (0, 0.4), (1, 0.1))))
    reply = _run(mode, _model(mode, up).rerank("rerank", "refund?", ["a", "b", "c"]))
    assert isinstance(reply, RerankReply)
    assert reply.scores == [0.4, 0.1, 0.9]
    request = up.requests[0]
    assert str(request.url) == "https://api.voyageai.com/v1/rerank"
    assert json.loads(request.content) == {
        "query": "refund?", "documents": ["a", "b", "c"], "model": "rerank-2.5",
        "return_documents": False,
    }  # fmt: skip
    assert (reply.provenance.tag, reply.usage.input, reply.usage.output) == (
        "voyage:rerank-2.5", 21, 0,
    )  # fmt: skip


def test_rerank_top_k_is_sent_and_unscored_documents_are_none(mode: str) -> None:
    up = Upstream((200, ranked((1, 0.8))))
    reply = _run(mode, _model(mode, up).rerank("rerank", "q", ["a", "b"], top_k=1))
    assert up.body(0)["top_k"] == 1
    assert reply.scores == [None, 0.8]


@pytest.mark.parametrize(
    "body", [ranked((0, 0.5), (0, 0.4)), ranked((5, 0.5)), ranked((0, 0.5), (1, 0.4), (1, 0.3))]
)
def test_a_reply_that_does_not_match_the_documents_is_bad(body: dict[str, Any]) -> None:
    up = Upstream((200, body))
    with pytest.raises(BadStructuredReply, match="rerank"):
        _model("sync", up).rerank("rerank", "q", ["a", "b"])


def test_rerank_needs_a_rerank_role_on_a_type_that_serves_it() -> None:
    with pytest.raises(ConfigError, match="ollama does not serve rerank"):
        build(_config(rerank={"backend": "ollama", "model": "x"}), ROLES)
    up = Upstream((200, ranked((0, 0.5))))
    with pytest.raises(ConfigError, match="rerank role"):
        _model("sync", up).rerank("embed", "q", ["a"])


def test_a_rerank_role_is_never_routable() -> None:
    with pytest.raises(ValueError, match="never routable"):
        RolePolicy(kind="rerank", routable=True)


def test_the_fakes_rerank() -> None:
    fake = FakeModel.with_scores({"rerank": lambda call: [float(len(d)) for d in call.documents]})
    assert fake.rerank("rerank", "q", ["aa", "b"]).scores == [2.0, 1.0]
    assert fake.rerank_calls == [FakeRerankCall("rerank", "q", ("aa", "b"), None)]
    vectors = FakeModel.with_vectors({"embed": lambda call: [[1.0]] * len(call.texts)})
    vectors.embed("embed", ["x"], input_type="query")
    assert vectors.embed_calls == [FakeEmbedCall("embed", ("x",), "query")]
    afake = AsyncFakeModel.with_scores({"rerank": lambda call: [0.0] * len(call.documents)})
    assert asyncio.run(afake.rerank("rerank", "q", ["x"])).scores == [0.0]


def test_bounded_passes_rerank_and_the_input_type_through() -> None:
    scorer = FakeModel.with_scores({"rerank": lambda call: [1.0, 2.0]})
    assert Bounded(scorer, 1).rerank("rerank", "q", ["a", "b"], top_k=1).scores == [None, 2.0]
    assert scorer.rerank_calls[0].top_k == 1
    embedder = FakeModel.with_vectors({"embed": lambda call: [[0.0]] * len(call.texts)})
    Bounded(embedder, 1).embed("embed", ["x"], input_type="document")
    assert embedder.embed_calls[0].input_type == "document"


# --- Codex review of 0.7.0 ---


@pytest.mark.parametrize(
    "data",
    [
        ["junk", {"index": 1, "relevance_score": 0.5}],  # a malformed entry is not dropped
        [{"index": True, "relevance_score": 0.5}],  # a bool is not an index
        [{"index": 0, "relevance_score": True}],  # nor a score
    ],
)
def test_malformed_rerank_entries_are_bad(data: list[Any]) -> None:
    body = {"object": "list", "data": data, "model": "rerank-2.5", "usage": {"total_tokens": 1}}
    up = Upstream((200, body))
    with pytest.raises(BadStructuredReply, match="rerank"):
        _model("sync", up).rerank("rerank", "q", ["a", "b"], top_k=1)


def test_top_k_must_be_positive() -> None:
    up = Upstream((200, ranked((0, 0.5))))
    with pytest.raises(ValueError, match="top_k"):
        _model("sync", up).rerank("rerank", "q", ["a"], top_k=0)
    assert up.requests == []
