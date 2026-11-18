"""Smoke test of an installed lite-model wheel (CI `package` job): imports, builds and fakes,
with no network and no model. `--anthropic`: the extra is installed and an anthropic role
builds; without it, build() refuses that role and names the extra."""

import os
import sys

from pydantic import BaseModel

from lite_model import ConfigError, Message, ModelsConfig, RolePolicy, RoleSpec, build, build_async
from lite_model.testing import AsyncFakeModel, FakeModel

ROLES = RoleSpec({"answer": RolePolicy()})


class Verdict(BaseModel):
    ok: bool


def main() -> None:
    fake = FakeModel(answer="hi", judge={"ok": True})
    assert fake.text("answer", [Message("user", "x")]).text == "hi"
    assert fake.structured("judge", [], Verdict).value == Verdict(ok=True)
    assert AsyncFakeModel(answer="hi").calls == []
    local = ModelsConfig(backend="ollama", model="qwen3:8b")
    model = build(local, ROLES)  # builds without calling anything
    assert model.target("answer").base_url == "http://127.0.0.1:11434/v1"
    assert len(model.hooks.snapshot()) == 0
    build_async(local, ROLES)
    os.environ["ANTHROPIC_API_KEY"] = "sk-smoke"
    claude = ModelsConfig(backend="anthropic", model="claude-x", max_output_tokens=64)
    if "--anthropic" in sys.argv:
        build(claude, ROLES)
    else:
        try:
            build(claude, ROLES)
        except ConfigError as exc:
            assert "lite-model[anthropic]" in str(exc), exc
        else:
            raise SystemExit("an anthropic role built without the anthropic extra")
    print("lite-model smoke: ok")


if __name__ == "__main__":
    main()
