"""Cohere's `POST /v2/embed` (design §3.4, 0.8.0): embeddings only, over `rest`. The texts go as
`texts`, the input type as `search_query` / `search_document`, float vectors are asked for; the
reply's `embeddings.float` is in input order and names no model."""

from collections.abc import Mapping
from typing import Any

from lite_model.backends import finish_embed, rest
from lite_model.config import Target
from lite_model.types import EmbedReply, InputType, Usage

INPUT_TYPES: dict[InputType, str] = {"query": "search_query", "document": "search_document"}


connect, connect_async = rest.connectors(lambda key: {"authorization": f"Bearer {key}"})
embed_endpoint = rest.Owner


def embed_request(
    target: Target, texts: list[str], input_type: InputType | None = None
) -> dict[str, Any]:
    body: dict[str, Any] = {"model": target.model, "texts": texts}
    if input_type is not None:
        body["input_type"] = INPUT_TYPES[input_type]
    body["embedding_types"] = ["float"]
    if target.dimensions is not None:
        body["output_dimension"] = target.dimensions
    return {"path": "embed", "json": body, "input": texts}


def embed_reply(
    target: Target, response: Any, headers: Mapping[str, str], sent: Mapping[str, Any]
) -> EmbedReply:
    body = response if isinstance(response, dict) else {}
    embeddings = body.get("embeddings")
    vectors = embeddings.get("float") if isinstance(embeddings, dict) else None
    vectors = vectors if isinstance(vectors, list) else []
    meta = body.get("meta")
    billed = meta.get("billed_units") if isinstance(meta, dict) else None
    tokens = billed.get("input_tokens") if isinstance(billed, dict) else None
    return finish_embed(
        target,
        sent,
        headers,
        body_model=None,  # the reply does not say; nothing is guessed (rule 4)
        usage=Usage(input=tokens if isinstance(tokens, int) else None, output=0),
        vectors=vectors,
        indexes=list(range(len(vectors))),  # in input order
    )
