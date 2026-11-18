"""The README's configuration and quick-start examples hold against the code."""

import re
from pathlib import Path

import pytest
import yaml

from lite_model import ModelsConfig, RolePolicy, RoleSpec
from lite_model.config import resolve

README = (Path(__file__).resolve().parents[2] / "README.md").read_text()


def _block(lang: str, after: str) -> str:
    section = README[README.index(after) :]
    match = re.search(rf"```{lang}\n(.*?)```", section, re.S)
    assert match, f"no {lang} block after {after!r}"
    return match.group(1)


def test_the_configuration_example_is_a_valid_config(monkeypatch: pytest.MonkeyPatch) -> None:
    for key in ("LITELLM_API_KEY", "DEEPSEEK_API_KEY"):
        monkeypatch.setenv(key, "sk-readme")
    config = ModelsConfig.model_validate(yaml.safe_load(_block("yaml", "## Configuration")))
    # `graph` is not in the example's roles: it takes the defaults, so they are checked too.
    spec = RoleSpec(
        {"answer": RolePolicy(routable=True), "extract": RolePolicy(), "graph": RolePolicy()}
    )
    targets = resolve(config, spec)
    assert (targets["graph"].backend, targets["graph"].model) == ("ollama", "qwen3:8b")
    assert (targets["answer"].type, targets["answer"].routed) == ("litellm", True)
    assert (targets["extract"].backend, targets["extract"].base_url) == (
        "deepseek",
        "https://api.deepseek.com/v1",
    )
    assert "deepseek-chat" in targets["extract"].prices


def test_the_quick_start_config_builds() -> None:
    code = _block("python", "## Quick start")
    setup = code[: code.index("reply = model.text")]  # the calls need a running Ollama
    scope: dict[str, object] = {}
    exec(compile(setup, "README.md", "exec"), scope)
    assert isinstance(scope["config"], ModelsConfig)
    graph = resolve(scope["config"], scope["ROLES"])["graph"]  # type: ignore[arg-type]
    assert (graph.model, graph.max_output_tokens, graph.reasoning_effort) == (
        "qwen3:1.7b",
        2048,
        "none",
    )
    assert hasattr(scope["model"], "hooks")
