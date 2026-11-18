"""`litemodel`: tools that make no model call (design §4)."""

import dataclasses
import json

import typer
from pydantic import ValidationError
from pydantic_settings import BaseSettings, SettingsConfigDict

from lite_model.config import ModelsConfig, target

app = typer.Typer(no_args_is_help=True, add_completion=False)


class EnvConfig(ModelsConfig, BaseSettings):
    """`ModelsConfig` from `LITEMODEL_*` variables and `.env`; nested keys use `__`, e.g.
    `LITEMODEL_ROLES__ANSWER__MODEL`. Hosts embed `ModelsConfig` under their own prefix."""

    model_config = SettingsConfigDict(
        env_prefix="LITEMODEL_", env_nested_delimiter="__", env_file=".env", extra="ignore"
    )


@app.callback()
def main() -> None:
    """lite-model command line."""


@app.command("dump-config")
def dump_config() -> None:
    """Print the merged config and each configured role's target. Keys appear as variable
    names only. Nothing is called and no startup check runs: this shows what `build()` would
    check."""
    try:
        config = EnvConfig()
    except ValidationError as exc:  # e.g. LITEMODEL_BACKEND / LITEMODEL_MODEL not set
        typer.echo(f"invalid config (LITEMODEL_* variables, .env):\n{exc}", err=True)
        raise typer.Exit(1) from exc
    targets = {role: dataclasses.asdict(target(config, role)) for role in config.roles}
    out = {"config": config.model_dump(mode="json"), "targets": targets}
    typer.echo(json.dumps(out, indent=2, ensure_ascii=False))
