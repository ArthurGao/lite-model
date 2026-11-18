"""Strict JSON schemas for structured output (design §5)."""

from typing import Any

from pydantic import BaseModel


def strict_schema(model: type[BaseModel]) -> dict[str, Any]:
    """`model`'s JSON schema as strict structured output wants it: every property required (an
    optional one is still nullable), no extra properties, no defaults, in every nested object.

    The same transformation as lite-mem's, so a host's request body is unchanged when it
    migrates (design §12.2)."""
    schema = model.model_json_schema()

    def walk(node: Any) -> None:
        if isinstance(node, dict):
            if isinstance(node.get("properties"), dict):
                node["required"] = list(node["properties"])
                node["additionalProperties"] = False
                for prop in node["properties"].values():
                    prop.pop("default", None)  # strict mode rejects defaults
            for value in node.values():
                walk(value)
        elif isinstance(node, list):
            for value in node:
                walk(value)

    walk(schema)
    return schema
