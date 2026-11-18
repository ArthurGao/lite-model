"""Configs shared by unit and contract tests."""

from typing import Any

from lite_model import BackendConfig, ModelsConfig, RoleConfig

# `backend` and `model` have no defaults (a host names its model): tests use local Ollama.
LOCAL = {"backend": "ollama", "model": "qwen3:8b"}


def local(**fields: Any) -> ModelsConfig:
    """ModelsConfig on the local Ollama defaults; `fields` override them."""
    return ModelsConfig.model_validate(LOCAL | fields)


def local_validate(data: dict[str, Any]) -> ModelsConfig:
    """`ModelsConfig.model_validate` with the local defaults under `data`."""
    return ModelsConfig.model_validate(LOCAL | data)


def two_vendors(**fields: object) -> ModelsConfig:
    """Two OpenAI-compatible vendors at once, each a named backend (design §4)."""
    return local_validate(
        {
            "backends": {
                "deepseek": BackendConfig(
                    type="openai",
                    base_url="https://deepseek.example/v1",
                    api_key_env="DEEPSEEK_API_KEY",
                ),
                "dashscope": BackendConfig(
                    type="openai",
                    base_url="https://dashscope.example/v1",
                    api_key_env="DASHSCOPE_API_KEY",
                ),
            },
            "roles": {
                "answer": RoleConfig(backend="deepseek", model="deepseek-chat"),
                "graph": RoleConfig(backend="dashscope", model="qwen-max"),
            },
            **fields,
        }
    )
