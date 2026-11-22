"""Opt-in: the real Cohere / Gemini embedding APIs (`CO_API_KEY=... uv run pytest -m cohere`,
`GEMINI_API_KEY=... uv run pytest -m gemini`). Paid per token, a few tiny calls; never part of
the default run (AGENTS.md rule 11). Without the key they fail, not skip."""

import math
import os

import pytest

from lite_model import ModelsConfig, RolePolicy, RoleSpec, build

ROLES = RoleSpec({"embed": RolePolicy(kind="embed")})
DOCS = ["Refunds are paid within 14 days.", "The office is closed on Sundays."]


def _cos(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    return dot / math.sqrt(sum(x * x for x in a) * sum(y * y for y in b))


def _check(backend: str, model: str, env: str) -> None:
    assert os.environ.get(env), f"set {env} for -m {backend}"
    config = ModelsConfig.model_validate(
        {"backend": backend, "model": model, "forbid_channels": ["subscription"],
         "roles": {"embed": {"model": model, "dimensions": 256}}}
    )  # fmt: skip
    model_ = build(config, ROLES)
    docs = model_.embed("embed", DOCS, input_type="document").vectors
    [query] = model_.embed("embed", ["how long do refunds take"], input_type="query").vectors
    assert [len(v) for v in docs] == [256, 256] and len(query) == 256
    assert _cos(query, docs[0]) > _cos(query, docs[1])


@pytest.mark.cohere
def test_cohere_embeds() -> None:
    _check("cohere", "embed-v4.0", "CO_API_KEY")


@pytest.mark.gemini
def test_gemini_embeds() -> None:
    _check("gemini", "gemini-embedding-001", "GEMINI_API_KEY")
