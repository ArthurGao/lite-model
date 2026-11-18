"""Configs shared by unit and contract tests."""

from lite_model import BackendConfig, ModelsConfig, RoleConfig


def two_vendors(**fields: object) -> ModelsConfig:
    """Two OpenAI-compatible vendors at once, each a named backend (design §4)."""
    return ModelsConfig.model_validate(
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
