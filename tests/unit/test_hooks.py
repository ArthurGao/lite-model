"""Call.schema_hash follows lite-eval's canonical JSON (the same key on both sides)."""

import hashlib
import json
import unicodedata

from pydantic import BaseModel, Field

from lite_model import RolePolicy, RoleSpec
from lite_model.config import resolve
from lite_model.hooks import Call
from lite_model.schema import strict_schema
from tests.configs import local


class Café(BaseModel):
    """Une café décomposée."""  # NFD below: e + combining acute

    name: str = Field(description="nom du café")


def test_schema_hash_is_lite_evals_canonical_json_sha256() -> None:
    target = resolve(local(), RoleSpec({"answer": RolePolicy()}))["answer"]
    call = Call(target, (), {}, Café)

    def nfc(v: object) -> object:  # lite_eval/canonical.py, written out
        if isinstance(v, str):
            return unicodedata.normalize("NFC", v)
        if isinstance(v, dict):
            return {nfc(k): nfc(x) for k, x in v.items()}
        if isinstance(v, list):
            return [nfc(x) for x in v]
        return v

    expected = json.dumps(
        nfc(strict_schema(Café)),
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
        allow_nan=False,
    )
    assert call.schema_hash == hashlib.sha256(expected.encode("utf-8")).hexdigest()
    assert Call(target, (), {}, None).schema_hash is None
