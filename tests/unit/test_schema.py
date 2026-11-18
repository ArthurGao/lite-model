"""strict_schema: what strict structured output needs, in every nested object."""

from pydantic import BaseModel, Field

from lite_model import strict_schema


class Inner(BaseModel):
    label: str = "x"
    weight: float | None = None


class Outer(BaseModel):
    name: str
    tags: list[str] = Field(default_factory=list)
    inner: Inner
    many: list[Inner] = []


def test_every_object_requires_every_property_and_allows_no_extra() -> None:
    schema = strict_schema(Outer)
    assert schema["required"] == ["name", "tags", "inner", "many"]
    assert schema["additionalProperties"] is False
    inner = schema["$defs"]["Inner"]
    assert inner["required"] == ["label", "weight"]
    assert inner["additionalProperties"] is False


def test_defaults_are_removed_and_optional_stays_nullable() -> None:
    schema = strict_schema(Outer)
    inner = schema["$defs"]["Inner"]["properties"]
    assert "default" not in inner["label"]
    assert "default" not in schema["properties"]["tags"]
    assert {"type": "null"} in inner["weight"]["anyOf"]
