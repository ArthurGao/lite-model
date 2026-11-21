"""Opt-in: the real Voyage AI API (`VOYAGE_API_KEY=... uv run pytest -m voyage`). Paid per token
(two tiny calls); never part of the default run (AGENTS.md rule 11). Without the key it fails,
not skips: an opt-in run that proved nothing must not look green."""

import math
import os

import pytest

from lite_model import LiteModel, ModelsConfig, RolePolicy, RoleSpec, build

pytestmark = pytest.mark.voyage

ROLES = RoleSpec({"embed": RolePolicy(kind="embed"), "rerank": RolePolicy(kind="rerank")})


def _model() -> LiteModel:
    assert os.environ.get("VOYAGE_API_KEY"), "set VOYAGE_API_KEY for -m voyage"
    config = ModelsConfig.model_validate(
        {
            "backend": "voyage",
            "model": "voyage-4-lite",
            "forbid_channels": ["subscription"],
            "roles": {
                "embed": {"model": "voyage-4-lite", "dimensions": 256},
                "rerank": {"model": "rerank-2.5-lite"},
            },
        }
    )
    return build(config, ROLES)


def test_embed_and_rerank_on_voyage() -> None:
    model = _model()
    docs = ["Refunds are paid within 14 days.", "The office is closed on Sundays."]
    reply = model.embed("embed", docs, input_type="document")
    query = model.embed("embed", ["how long do refunds take"], input_type="query")
    assert [len(v) for v in reply.vectors] == [256, 256] and len(query.vectors[0]) == 256
    assert reply.provenance.answered == "voyage-4-lite" and (reply.usage.input or 0) > 0

    def cos(a: list[float], b: list[float]) -> float:
        dot = sum(x * y for x, y in zip(a, b, strict=True))
        return dot / math.sqrt(sum(x * x for x in a) * sum(y * y for y in b))

    near, far = (cos(query.vectors[0], v) for v in reply.vectors)
    assert near > far
    ranked = model.rerank("rerank", "how long do refunds take", docs)
    assert ranked.provenance.answered == "rerank-2.5-lite"
    first, second = ranked.scores
    assert first is not None and second is not None and first > second
