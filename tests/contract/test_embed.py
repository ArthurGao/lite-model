"""Design §3.2: `embed()` — what the OpenAI SDK sends for an embed role, what is read back, the
batch and provenance rules, and the config checks. Every call test runs sync and async."""

import asyncio
import json
from collections.abc import Callable, Sequence
from typing import Any

import httpx2
import pytest

from lite_model import (
    BadStructuredReply,
    CallFailed,
    ConfigError,
    EmbedReply,
    Message,
    ModelsConfig,
    ProvenanceUnknown,
    RolePolicy,
    RoleSpec,
    build,
    build_async,
)
from lite_model.testing import (
    AsyncFakeModel,
    FakeCall,
    FakeEmbedCall,
    FakeModel,
    hash_vectors,
)
from tests.configs import local_validate
from tests.upstream import Upstream, embeddings

ROLES = RoleSpec({"answer": RolePolicy(), "embed": RolePolicy(kind="embed")})


def _config(**roles: Any) -> ModelsConfig:
    embed = {"model": "bge-m3:567m"} | roles.pop("embed", {})
    return local_validate({"max_retries": 0, "roles": {"embed": embed, **roles}})


class Embedder:
    def __init__(self, mode: str, up: Upstream, config: ModelsConfig) -> None:
        if mode == "sync":
            self._model: Any = build(config, ROLES, http_client=up.client())
            self._wait: Any = lambda r: r
        else:
            self._model = build_async(config, ROLES, http_client=up.async_client())
            self._wait = asyncio.run

    def embed(self, role: str, texts: Sequence[str]) -> EmbedReply:
        reply = self._wait(self._model.embed(role, texts))
        assert isinstance(reply, EmbedReply)
        return reply

    def text(self, role: str, messages: Sequence[Message]) -> Any:
        return self._wait(self._model.text(role, messages))

    @property
    def model(self) -> Any:
        return self._model


@pytest.fixture(params=["sync", "async"])
def mode(request: pytest.FixtureRequest) -> str:
    return str(request.param)


def _embedder(mode: str, up: Upstream, config: ModelsConfig | None = None) -> Embedder:
    return Embedder(mode, up, config or _config())


# --- request and reply ---


def test_request_is_the_openai_embeddings_body_on_ollama_v1(mode: str) -> None:
    up = Upstream((200, embeddings([0.1, 0.2], [0.3, 0.4])))
    _embedder(mode, up).embed("embed", ["Ada", "Wellington"])
    assert str(up.requests[0].url) == "http://127.0.0.1:11434/v1/embeddings"
    # "float" spelled out: left out, openai 3.46.0 asks for base64 (design §3.2)
    assert up.body() == {
        "model": "bge-m3:567m",
        "input": ["Ada", "Wellington"],
        "encoding_format": "float",
    }


def test_reply_is_in_input_order_with_who_answered_and_usage(mode: str) -> None:
    body = embeddings(
        [1.0, 0.0], [0.0, 1.0], [0.5, 0.5], model="bge-m3:latest", order=[2, 0, 1],
        usage={"prompt_tokens": 12, "total_tokens": 12},
    )  # fmt: skip
    reply = _embedder(mode, Upstream((200, body))).embed("embed", ["a", "b", "c"])
    assert reply.vectors == [[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]]
    assert reply.dim == 2
    p = reply.provenance
    assert (p.role, p.backend, p.requested, p.answered, p.channel, p.routed) == (
        "embed", "ollama", "bge-m3:567m", "bge-m3:latest", "local", False,
    )  # fmt: skip
    assert (reply.usage.input, reply.usage.output, reply.usage.cost_usd) == (12, 0, None)


def test_inherited_chat_settings_are_not_sent(mode: str) -> None:
    # lite-mem's top level sets reasoning_effort "none" for its chat roles (design §3.2)
    config = local_validate(
        {
            "max_retries": 0,
            "reasoning_effort": "none",
            "max_output_tokens": 512,
            "temperature": 0.3,
            "roles": {"embed": {"model": "bge-m3:567m"}},
        }
    )
    up = Upstream((200, embeddings([0.1])))
    _embedder(mode, up, config).embed("embed", ["x"])
    assert set(up.body()) == {"model", "input", "encoding_format"}


def test_texts_go_in_batches_of_64_in_order(mode: str) -> None:
    texts = [f"t{i}" for i in range(130)]
    sizes = [64, 64, 2]
    replies = [
        (200, embeddings(*([[float(n)]] * n), usage={"prompt_tokens": n, "total_tokens": n}))
        for n in sizes
    ]
    up = Upstream(*replies)
    reply = _embedder(mode, up).embed("embed", texts)
    assert [up.body(i)["input"] for i in range(3)] == [texts[:64], texts[64:128], texts[128:]]
    assert reply.vectors == [[64.0]] * 64 + [[64.0]] * 64 + [[2.0]] * 2
    assert reply.usage.input == 130


def test_unknown_usage_in_one_batch_makes_the_total_unknown(mode: str) -> None:
    first = embeddings(*([[1.0]] * 64))
    second = embeddings([1.0])
    del second["usage"]
    up = Upstream((200, first), (200, second))
    reply = _embedder(mode, up).embed("embed", ["x"] * 65)
    assert reply.usage.input is None


# --- errors ---


def test_a_missing_vector_is_a_bad_reply(mode: str) -> None:
    up = Upstream((200, embeddings([0.25, 0.75])))
    with pytest.raises(BadStructuredReply, match=r"asked for 2 vectors, got 1") as caught:
        _embedder(mode, up).embed("embed", ["a", "b"])
    assert "0.25" not in str(caught.value)


def test_vectors_of_different_length_are_a_bad_reply(mode: str) -> None:
    up = Upstream((200, embeddings([0.1, 0.2], [0.3])))
    with pytest.raises(BadStructuredReply, match="vectors of different lengths"):
        _embedder(mode, up).embed("embed", ["a", "b"])


def test_batches_of_different_length_are_a_bad_reply(mode: str) -> None:
    up = Upstream((200, embeddings(*([[0.1, 0.2]] * 64))), (200, embeddings([0.3])))
    with pytest.raises(BadStructuredReply, match="vectors of different lengths"):
        _embedder(mode, up).embed("embed", ["x"] * 65)


def test_batches_answered_by_different_models_are_provenance_unknown(mode: str) -> None:
    up = Upstream(
        (200, embeddings(*([[0.1]] * 64), model="m-a")),
        (200, embeddings([0.2], model="m-b")),
    )
    with pytest.raises(ProvenanceUnknown, match="more than one model"):
        _embedder(mode, up).embed("embed", ["x"] * 65)


def test_no_texts_sends_nothing(mode: str) -> None:
    up = Upstream((200, embeddings([0.1])))
    with pytest.raises(ValueError, match="no texts"):
        _embedder(mode, up).embed("embed", [])
    assert up.requests == []


def test_retries_stay_on_the_target_and_fail_without_upstream_text(mode: str) -> None:
    echo = {"error": {"message": "bad key sk-leak", "type": "server_error"}}
    up = Upstream((500, echo), (500, echo))
    config = _config(embed={"max_retries": 1})
    with pytest.raises(CallFailed) as caught:
        _embedder(mode, up, config).embed("embed", ["x"])
    assert len(up.requests) == 2
    assert {up.body(i)["model"] for i in range(2)} == {"bge-m3:567m"}
    assert "sk-leak" not in str(caught.value)
    assert str(caught.value) == "role embed (ollama:bge-m3:567m) failed: InternalServerError"


def test_hooks_are_not_called(mode: str) -> None:
    class Spy:
        name = "spy"

        def __init__(self) -> None:
            self.seen: list[Any] = []

        def before(self, call: Any) -> None:
            self.seen.append(call)

        def after(self, call: Any, reply: Any) -> None:
            self.seen.append(call)

    spy = Spy()
    embedder = _embedder(mode, Upstream((200, embeddings([0.1]))))
    embedder.model.hooks.register(spy)
    embedder.embed("embed", ["x"])
    assert spy.seen == []


# --- role kinds and config checks ---


def test_embed_on_a_chat_role_and_text_on_an_embed_role_are_refused(mode: str) -> None:
    up = Upstream((200, embeddings([0.1])))
    embedder = _embedder(mode, up)
    with pytest.raises(ConfigError, match="role answer: a chat role; embed"):
        embedder.embed("answer", ["x"])
    with pytest.raises(ConfigError, match="role embed: an embed role; text"):
        embedder.text("embed", [Message("user", "hi")])
    assert up.requests == []


def test_an_embed_role_cannot_be_routable() -> None:
    with pytest.raises(ValueError, match="never routable"):
        RolePolicy(kind="embed", routable=True)


@pytest.mark.parametrize(
    ("backend", "reason"),
    [("gateway", "gateway does not serve embeddings"), ("anthropic", "anthropic does not serve")],
)
def test_an_embed_role_on_a_backend_without_embeddings_is_refused(
    backend: str, reason: str
) -> None:
    config = _config(embed={"backend": backend, "max_output_tokens": None})
    with pytest.raises(ConfigError, match=reason):
        build(config, ROLES)


@pytest.mark.parametrize(
    "setting", [{"temperature": 0.2}, {"max_output_tokens": 64}, {"reasoning_effort": "low"}]
)
def test_chat_settings_on_an_embed_role_are_refused(setting: dict[str, Any]) -> None:
    (name,) = setting
    with pytest.raises(ConfigError, match=f"role embed .*: {name} is not used by embeddings"):
        build(_config(embed=setting), ROLES)


# --- fakes ---


def test_fake_embed_records_calls_and_hash_vectors_are_deterministic() -> None:
    fake = FakeModel.with_vectors({"embed": hash_vectors(4)})
    first = fake.embed("embed", ["a", "b"])
    again = fake.embed("embed", ["a"])
    assert first.dim == 4 and len(first.vectors) == 2
    assert again.vectors[0] == first.vectors[0]
    assert first.vectors[0] != first.vectors[1]
    assert fake.embed_calls == [FakeEmbedCall("embed", ("a", "b")), FakeEmbedCall("embed", ("a",))]
    assert first.provenance.backend == "fake"


def test_async_fake_embed() -> None:
    by_length = {"embed": lambda call: [[float(len(t))] for t in call.texts]}
    fake = AsyncFakeModel.with_vectors(by_length)
    reply = asyncio.run(fake.embed("embed", ["ab", "c"]))
    assert reply.vectors == [[2.0], [1.0]]
    assert fake.embed_calls == [FakeEmbedCall("embed", ("ab", "c"))]


def test_scripts_unpacked_from_a_dict_still_type_check() -> None:
    # 0.5.0's `vectors=` keyword broke hosts that build a fake from a dict (lite-rag testing.py:
    # `FakeModel(**{role: script})`): mypy matched the dict's values against `vectors` too. The
    # check is mypy on this file; the run proves the scripts still land as chat scripts.
    scripts: dict[str, Callable[[FakeCall], str]] = {"answer": lambda call: "ok"}
    assert FakeModel(**scripts).text("answer", []).text == "ok"
    assert asyncio.run(AsyncFakeModel(**scripts).text("answer", [])).text == "ok"
    role_named_vectors = FakeModel.with_vectors({}, vectors="a role called vectors")
    assert role_named_vectors.text("vectors", []).text == "a role called vectors"


def test_an_embed_target_carries_no_chat_settings() -> None:
    config = local_validate(
        {"temperature": 0.3, "reasoning_effort": "none", "roles": {"embed": {"model": "bge-m3"}}}
    )
    t = build(config, ROLES).target("embed")
    assert (t.kind, t.temperature, t.reasoning_effort, t.max_output_tokens) == (
        "embed", None, None, None,
    )  # fmt: skip
    assert build(config, ROLES).target("answer").temperature == 0.3


def test_bounded_embeds_through_its_slot() -> None:
    from lite_model import AsyncBounded, Bounded

    up = Upstream((200, embeddings([0.5])))
    reply = Bounded(build(_config(), ROLES, http_client=up.client())).embed("embed", ["x"])
    assert reply.vectors == [[0.5]]
    fake = AsyncFakeModel.with_vectors({"embed": hash_vectors(3)})
    assert asyncio.run(AsyncBounded(fake).embed("embed", ["y"])).dim == 3


def test_duplicate_or_missing_indexes_are_a_bad_reply(mode: str) -> None:
    # Codex 0.5.0: two vectors for index 0 and none for 1 were sorted and handed out as if valid
    body = embeddings([0.1], [0.2], order=[0, 0])
    with pytest.raises(BadStructuredReply, match="indexes do not match the inputs"):
        _embedder(mode, Upstream((200, body))).embed("embed", ["a", "b"])


def test_output_tokens_are_zero_even_when_input_usage_is_unknown(mode: str) -> None:
    # Codex 0.5.0: an embedding produces no output tokens, whatever the endpoint reports
    one = embeddings([0.1])
    del one["usage"]
    reply = _embedder(mode, Upstream((200, one))).embed("embed", ["a"])
    assert (reply.usage.input, reply.usage.output) == (None, 0)
    two = embeddings(*([[0.1]] * 64))
    up = Upstream((200, two), (200, one))
    reply = _embedder(mode, up).embed("embed", ["a"] * 65)
    assert (reply.usage.input, reply.usage.output) == (None, 0)


# --- 0.6.0: dimensions, batch size, concurrency, halving; closing ---


class Sized:
    """An `/embeddings` endpoint answering each text with `[len(text)]`; a batch larger than
    `limit` is refused with `status` (as an endpoint does for a request too large)."""

    def __init__(self, limit: int = 10_000, status: int = 413) -> None:
        self.limit, self.status = limit, status
        self.sizes: list[int] = []

    def __call__(self, request: httpx2.Request) -> httpx2.Response:
        texts = json.loads(request.content)["input"]
        self.sizes.append(len(texts))
        if len(texts) > self.limit:
            return httpx2.Response(self.status, json={"error": {"message": "too large"}})
        return httpx2.Response(200, json=embeddings(*[[float(len(t))] for t in texts]))

    def embedder(self, mode: str, **embed: Any) -> Any:
        config = _config(embed=embed)
        if mode == "sync":
            return build(
                config, ROLES, http_client=httpx2.Client(transport=httpx2.MockTransport(self))
            )
        return build_async(
            config, ROLES, http_client=httpx2.AsyncClient(transport=httpx2.MockTransport(self))
        )


def _embed(mode: str, model: Any, texts: list[str]) -> EmbedReply:
    reply = (
        model.embed("embed", texts) if mode == "sync" else asyncio.run(model.embed("embed", texts))
    )
    assert isinstance(reply, EmbedReply)
    return reply


TEXTS = ["a", "bb", "ccc", "dddd", "eeeee"]


def test_dimensions_are_sent(mode: str) -> None:
    up = Upstream((200, embeddings([0.1, 0.2])))
    _embedder(mode, up, _config(embed={"dimensions": 256})).embed("embed", ["Ada"])
    assert up.body()["dimensions"] == 256


def test_batch_size_and_concurrency_keep_the_input_order(mode: str) -> None:
    up = Sized()
    reply = _embed(mode, up.embedder(mode, batch_size=2, max_concurrency=3), TEXTS)
    assert reply.vectors == [[1.0], [2.0], [3.0], [4.0], [5.0]]
    assert sorted(up.sizes) == [1, 2, 2]


def test_a_batch_too_large_is_halved_until_it_fits(mode: str) -> None:
    up = Sized(limit=2)
    reply = _embed(mode, up.embedder(mode), TEXTS)
    assert reply.vectors == [[1.0], [2.0], [3.0], [4.0], [5.0]]
    assert up.sizes == [5, 2, 3, 1, 2]  # 5 refused -> 2 ok + 3 refused -> 1 ok + 2 ok


def test_a_single_text_still_too_large_fails(mode: str) -> None:
    up = Sized(limit=0, status=400)
    with pytest.raises(CallFailed, match=r"role embed \(ollama:bge-m3:567m\) failed"):
        _embed(mode, up.embedder(mode), ["a", "b"])
    assert up.sizes == [2, 1]  # halved once, then the first single text fails


@pytest.mark.parametrize("setting", [{"batch_size": 8}, {"max_concurrency": 2}, {"dimensions": 64}])
def test_embed_settings_on_a_chat_role_are_refused(setting: dict[str, int]) -> None:
    name = next(iter(setting))
    with pytest.raises(ConfigError, match=f"role answer .*: {name} is only for embed roles"):
        build(_config(answer=setting), ROLES)


def test_close_releases_the_clients_and_refuses_later_calls() -> None:
    model = build(_config(), ROLES)
    with model:
        pass
    with pytest.raises(RuntimeError, match="closed"):
        model.embed("embed", ["x"])

    async def run() -> None:
        async with build_async(_config(), ROLES) as amodel:
            pass
        with pytest.raises(RuntimeError, match="closed"):
            await amodel.embed("embed", ["x"])

    asyncio.run(run())


def test_close_leaves_an_injected_http_client_open() -> None:
    http = httpx2.Client(transport=httpx2.MockTransport(Sized()))
    build(_config(), ROLES, http_client=http).close()
    assert not http.is_closed
    ahttp = httpx2.AsyncClient(transport=httpx2.MockTransport(Sized()))
    asyncio.run(build_async(_config(), ROLES, http_client=ahttp).aclose())
    assert not ahttp.is_closed


def test_bounded_and_the_fakes_close_too() -> None:
    from lite_model import AsyncBounded, Bounded

    model = build(_config(), ROLES)
    Bounded(model, concurrency=2).close()  # closes the model it wraps
    with pytest.raises(RuntimeError, match="closed"):
        model.embed("embed", ["x"])
    amodel = build_async(_config(), ROLES)
    asyncio.run(AsyncBounded(amodel, concurrency=2).aclose())
    with pytest.raises(RuntimeError, match="closed"):
        asyncio.run(amodel.embed("embed", ["x"]))
    FakeModel().close()  # nothing to release; the protocol is the same
    asyncio.run(AsyncFakeModel().aclose())


def test_a_failed_batch_stops_the_batches_not_yet_sent() -> None:
    # simplify 0.6.0: with a pool that waits for every queued batch, a failed call kept
    # sending (paid) requests after it had already failed
    up = Sized(limit=0, status=500)
    model = up.embedder("sync", batch_size=1, max_concurrency=2)
    with pytest.raises(CallFailed):
        model.embed("embed", ["a", "b", "c", "d", "e", "f", "g", "h"])
    assert len(up.sizes) < 8
