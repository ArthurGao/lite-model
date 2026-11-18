"""Design §4: role resolution and every startup check `build()` runs."""

import pytest
from pydantic import ValidationError

from lite_model import BackendConfig, ConfigError, ModelsConfig, RoleConfig, RolePolicy, RoleSpec
from lite_model.client import build
from lite_model.config import resolve, target
from tests.configs import two_vendors

ROLES = RoleSpec({"answer": RolePolicy(routable=True), "graph": RolePolicy(routable=False)})


def _problems(config: ModelsConfig, spec: RoleSpec = ROLES) -> str:
    with pytest.raises(ConfigError) as caught:
        resolve(config, spec)
    return str(caught.value)


def test_defaults_resolve_every_role_to_local_ollama() -> None:
    targets = resolve(ModelsConfig(), ROLES)
    assert set(targets) == {"answer", "graph"}
    t = targets["answer"]
    assert (t.backend, t.type, t.model, t.channel) == ("ollama", "ollama", "qwen3:8b", "local")
    assert t.base_url == "http://127.0.0.1:11434/v1"
    assert (t.api_key_env, t.routed, t.timeout_s, t.connect_timeout_s, t.max_retries) == (
        None,
        False,
        120.0,
        10.0,
        1,
    )
    # the ollama profile's temperature is resolved here, so dump-config shows what is sent
    assert (t.temperature, t.max_output_tokens, t.reasoning_effort) == (0.0, None, None)


def test_a_role_override_keeps_every_field_it_leaves_out() -> None:
    config = ModelsConfig(
        reasoning_effort="none",
        max_output_tokens=512,
        roles={"graph": RoleConfig(model="qwen3:1.7b", max_output_tokens=2048, max_retries=0)},
    )
    graph, answer = target(config, "graph"), target(config, "answer")
    assert (graph.model, graph.max_output_tokens, graph.max_retries) == ("qwen3:1.7b", 2048, 0)
    assert (graph.reasoning_effort, graph.backend, graph.timeout_s) == ("none", "ollama", 120.0)
    assert (answer.model, answer.max_output_tokens, answer.max_retries) == ("qwen3:8b", 512, 1)


def test_named_backends_give_each_role_its_own_endpoint_and_key() -> None:
    config = two_vendors()
    answer, graph = target(config, "answer"), target(config, "graph")
    assert (answer.backend, answer.type, answer.channel) == ("deepseek", "openai", "api")
    assert (answer.base_url, answer.api_key_env) == (
        "https://deepseek.example/v1",
        "DEEPSEEK_API_KEY",
    )
    assert (graph.base_url, graph.api_key_env) == (
        "https://dashscope.example/v1",
        "DASHSCOPE_API_KEY",
    )


def test_switching_backend_never_reuses_the_default_backends_endpoint_or_key() -> None:
    config = ModelsConfig(
        backend="corp",
        backends={
            "corp": BackendConfig(
                type="openai", base_url="https://corp.example/v1", api_key_env="CORP_KEY"
            )
        },
        roles={"graph": RoleConfig(backend="ollama")},
    )
    graph = target(config, "graph")
    assert (graph.base_url, graph.api_key_env) == ("http://127.0.0.1:11434/v1", None)


def test_built_in_backends_stay_when_the_config_adds_one() -> None:
    config = ModelsConfig(backends={"corp": BackendConfig(type="openai")})
    assert set(config.backends) == {"ollama", "openai", "anthropic", "gateway", "litellm", "corp"}
    replaced = ModelsConfig(
        backends={"ollama": BackendConfig(type="ollama", base_url="http://gpu:1/v1")}
    )
    assert target(replaced, "answer").base_url == "http://gpu:1/v1"


def test_unknown_config_fields_are_rejected() -> None:
    with pytest.raises(ValidationError):
        ModelsConfig.model_validate({"roles": {"answer": {"modle": "x"}}})
    with pytest.raises(ValidationError):
        ModelsConfig.model_validate({"backends": {"x": {"type": "openai", "key": "sk-..."}}})


# --- the startup checks (design §4) ---


def test_a_role_the_host_did_not_declare_is_refused() -> None:
    config = ModelsConfig(roles={"answr": RoleConfig(model="x")})
    assert "role answr: not declared by the host" in _problems(config)


def test_an_unknown_backend_is_refused() -> None:
    config = ModelsConfig(roles={"answer": RoleConfig(backend="olama")})
    assert "role answer: unknown backend 'olama'" in _problems(config)


def test_a_missing_key_is_refused_by_variable_name(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    config = ModelsConfig(backend="openai", model="gpt-x")
    assert "role answer (openai:gpt-x): set OPENAI_API_KEY" in _problems(config)
    monkeypatch.setenv("OPENAI_API_KEY", "")
    assert "set OPENAI_API_KEY" in _problems(config)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    assert resolve(config, ROLES)["answer"].api_key_env == "OPENAI_API_KEY"


def test_anthropic_without_its_extra_is_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    import importlib.util

    real = importlib.util.find_spec
    monkeypatch.setattr(
        importlib.util,
        "find_spec",
        lambda name, *a: None if name == "anthropic" else real(name, *a),
    )
    config = ModelsConfig(backend="anthropic", model="claude-x")
    assert "install the extra: lite-model[anthropic]" in _problems(config)


def test_a_routing_alias_on_a_pinned_role_is_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LITELLM_API_KEY", "sk-test")
    routed = RoleConfig(backend="litellm", model="smart-router")
    problems = _problems(ModelsConfig(roles={"answer": routed, "graph": routed}))
    assert "role graph (litellm:smart-router): smart-router is a routing alias" in problems
    assert "role answer (litellm:smart-router): smart-router is a routing alias" not in problems
    pinned = ModelsConfig(roles={"graph": RoleConfig(backend="litellm", model="workhorse")})
    assert target(pinned, "graph").routed is False
    assert resolve(pinned, ROLES)["graph"].routed is False  # a pinned model is fine


def test_a_forbidden_channel_is_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    config = ModelsConfig(
        forbid_channels=("subscription", "api"), roles={"graph": RoleConfig(backend="openai")}
    )
    problems = _problems(config)
    assert "role graph (openai:qwen3:8b): channel api is forbidden" in problems
    assert "role answer" not in problems  # ollama is local
    gateway = config.model_copy(update={"roles": {"graph": RoleConfig(backend="gateway")}})
    assert "channel subscription is forbidden" in _problems(gateway)


def test_a_parameter_the_backend_does_not_support_is_refused(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    claude = ModelsConfig(backend="anthropic", model="claude-x", temperature=0.2)
    assert "temperature is not supported by anthropic" in _problems(claude)
    gateway = ModelsConfig(backend="gateway", model="claude-sub-haiku", reasoning_effort="none")
    assert "reasoning_effort is not supported by gateway" in _problems(gateway)


def test_every_problem_is_reported_at_once(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    config = ModelsConfig(
        forbid_channels=("api",),
        roles={"answer": RoleConfig(backend="openai"), "nope": RoleConfig()},
    )
    problems = _problems(config)
    assert problems.count("\n- ") == 3, problems  # undeclared, key, channel


def test_build_checks_before_any_call(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(ConfigError, match="set OPENAI_API_KEY"):
        build(ModelsConfig(backend="openai"), ROLES)


def test_gateway_and_litellm_build(monkeypatch: pytest.MonkeyPatch) -> None:
    # Until S3 these types were refused at build(); S3 added their clients.
    monkeypatch.setenv("LITELLM_API_KEY", "sk-test")
    config = ModelsConfig(
        roles={
            "answer": RoleConfig(backend="litellm", model="smart-router"),
            "graph": RoleConfig(backend="gateway", model="claude-sub-haiku"),
        }
    )
    model = build(config, ROLES)
    assert hasattr(model, "text")
    targets = resolve(config, ROLES)
    assert (targets["answer"].routed, targets["answer"].channel) == (True, "api")
    assert (targets["graph"].routed, targets["graph"].channel) == (False, "subscription")


def test_a_routing_alias_through_the_gateway_is_refused() -> None:
    # The gateway answers with its profile name and does not pass the upstream model on: a
    # routed call through it could not say which model answered (design §6.2).
    config = ModelsConfig(
        backends={"gateway": BackendConfig(type="gateway", router_aliases=("sub-router",))},
        roles={"answer": RoleConfig(backend="gateway", model="sub-router")},
    )
    problems = _problems(config)
    assert "role answer (gateway:sub-router): routing through gateway cannot tell" in problems


def test_config_errors_never_carry_a_key_value(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-very-secret")
    config = ModelsConfig(backend="openai", forbid_channels=("api",))
    assert "sk-very-secret" not in _problems(config)


@pytest.mark.parametrize(
    "data",
    [
        {"backends": {"corp": {"type": "openai", "api_key_env": ""}}},  # "no key"? or default?
        {"backends": {"corp": {"type": "openai", "base_url": ""}}},
        {"roles": {"answer": {"model": ""}}},
        {"roles": {"answer": {"backend": ""}}},
        {"model": ""},
    ],
)
def test_an_empty_string_is_refused_not_read_as_the_default(data: dict[str, object]) -> None:
    # Codex S1: `or` turned "" into the profile default, a silent reinterpretation.
    with pytest.raises(ValidationError):
        ModelsConfig.model_validate(data)


@pytest.mark.parametrize("value", [-0.1, 2.1])
def test_a_temperature_out_of_range_is_refused(value: float) -> None:
    # Codex S1: any float passed build() and was sent.
    with pytest.raises(ValidationError):
        ModelsConfig(temperature=value)
    with pytest.raises(ValidationError):
        RoleConfig(temperature=value)


def test_anthropic_without_an_output_limit_is_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    # The Messages API requires max_tokens; no default is guessed (owner, 2026-11-18).
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    config = ModelsConfig(backend="anthropic", model="claude-x")
    assert "role answer (anthropic:claude-x): anthropic needs max_output_tokens" in _problems(
        config
    )
    assert resolve(config.model_copy(update={"max_output_tokens": 1024}), ROLES)


def test_an_anthropic_base_url_ending_in_v1_is_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    # The Anthropic SDK appends /v1/messages itself (the OpenAI SDK wants /v1 in base_url): a
    # copied OpenAI-style URL would call /v1/v1/messages and fail only at the first call.
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    config = ModelsConfig(
        backend="claude",
        model="claude-x",
        max_output_tokens=64,
        backends={"claude": BackendConfig(type="anthropic", base_url="https://gw.example/v1/")},
    )
    assert "anthropic base_url must not end in /v1" in _problems(config)
    root = config.backends["claude"].model_copy(update={"base_url": "https://gw.example"})
    assert resolve(config.model_copy(update={"backends": {"claude": root}}), ROLES)


def test_a_price_for_an_unknown_backend_is_refused() -> None:
    config = ModelsConfig.model_validate(
        {
            "prices": {
                "olama:qwen3:8b": {"input_per_mtok": 1, "output_per_mtok": 1},
                "no-colon": {"input_per_mtok": 1, "output_per_mtok": 1},
            }
        }
    )
    problems = _problems(config)
    assert "prices: 'olama:qwen3:8b' is not 'backend:model'" in problems
    assert "prices: 'no-colon' is not 'backend:model'" in problems


def test_a_target_carries_only_its_backends_prices() -> None:
    config = ModelsConfig.model_validate(
        {
            "prices": {
                "ollama:qwen3:8b": {"input_per_mtok": 1, "output_per_mtok": 2},
                "openai:gpt-x": {"input_per_mtok": 5, "output_per_mtok": 5},
            }
        }
    )
    assert set(target(config, "answer").prices) == {"qwen3:8b"}
    with pytest.raises(ValidationError):
        ModelsConfig.model_validate(
            {"prices": {"ollama:m": {"input_per_mtok": -1, "output_per_mtok": 0}}}
        )


@pytest.mark.parametrize("rate", [float("inf"), float("nan")])
def test_a_non_finite_price_is_refused(rate: float) -> None:
    # Codex S4: a price of inf passed `ge=0` and made cost_usd infinite.
    with pytest.raises(ValidationError):
        ModelsConfig.model_validate(
            {"prices": {"ollama:m": {"input_per_mtok": rate, "output_per_mtok": 0}}}
        )
