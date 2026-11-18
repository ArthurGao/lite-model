"""`litemodel dump-config` (design §4): the merged config, keys as variable names only, no call."""

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from lite_model.cli import app


def test_dump_config_merges_env_and_shows_each_roles_target(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.chdir(tmp_path)  # no stray .env
    monkeypatch.setenv("LITEMODEL_BACKEND", "ollama")
    monkeypatch.setenv("LITEMODEL_MODEL", "qwen3:8b")
    monkeypatch.setenv("LITEMODEL_REASONING_EFFORT", "none")
    monkeypatch.setenv("LITEMODEL_ROLES__ANSWER__BACKEND", "openai")
    monkeypatch.setenv("LITEMODEL_ROLES__ANSWER__MODEL", "gpt-x")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-must-not-appear")
    result = CliRunner().invoke(app, ["dump-config"])
    assert result.exit_code == 0, result.output
    out = json.loads(result.output)
    assert out["config"]["reasoning_effort"] == "none"
    answer = out["targets"]["answer"]
    assert (answer["backend"], answer["model"], answer["channel"]) == ("openai", "gpt-x", "api")
    assert answer["api_key_env"] == "OPENAI_API_KEY"
    assert answer["reasoning_effort"] == "none"
    assert "sk-must-not-appear" not in result.output


def test_dump_config_runs_without_keys(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("LITEMODEL_BACKEND", "openai")
    monkeypatch.setenv("LITEMODEL_MODEL", "gpt-x")
    monkeypatch.setenv("LITEMODEL_ROLES__ANSWER__MODEL", "gpt-x")  # a target that needs the key
    result = CliRunner().invoke(app, ["dump-config"])
    assert result.exit_code == 0, result.output
    assert json.loads(result.output)["targets"]["answer"]["api_key_env"] == "OPENAI_API_KEY"


def test_dump_config_without_backend_and_model_says_so(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # backend and model have no defaults: the CLI reports it instead of a traceback.
    monkeypatch.chdir(tmp_path)
    for name in ("LITEMODEL_BACKEND", "LITEMODEL_MODEL"):
        monkeypatch.delenv(name, raising=False)
    result = CliRunner().invoke(app, ["dump-config"])
    assert result.exit_code == 1
    assert "invalid config" in result.output
    assert "backend" in result.output and "model" in result.output
    assert "Traceback" not in result.output
