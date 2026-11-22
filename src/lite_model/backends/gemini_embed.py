"""Gemini's `POST /v1beta/models/{model}:batchEmbedContents` (design §3.4, 0.8.0): embeddings
only, over `rest`, keyed by `x-goog-api-key` (not a bearer token). One Content per text — parts
of one Content are merged into a single vector; the input type as `taskType`
(RETRIEVAL_QUERY / RETRIEVAL_DOCUMENT), left out when none is given (a model that takes a text
prefix instead, e.g. gemini-embedding-2); the reply's `embeddings[].values` in input order."""

from collections.abc import Mapping
from typing import Any

from lite_model.backends import finish_embed, rest
from lite_model.config import Target
from lite_model.types import EmbedReply, InputType, Usage

TASK_TYPES: dict[InputType, str] = {"query": "RETRIEVAL_QUERY", "document": "RETRIEVAL_DOCUMENT"}


connect, connect_async = rest.connectors(lambda key: {"x-goog-api-key": key})
embed_endpoint = rest.Owner


def embed_request(
    target: Target, texts: list[str], input_type: InputType | None = None
) -> dict[str, Any]:
    name = f"models/{target.model}"
    shared: dict[str, Any] = {"model": name}
    if input_type is not None:
        shared["taskType"] = TASK_TYPES[input_type]
    if target.dimensions is not None:
        shared["outputDimensionality"] = target.dimensions
    requests = [shared | {"content": {"parts": [{"text": t}]}} for t in texts]
    return {"path": f"{name}:batchEmbedContents", "json": {"requests": requests}, "input": texts}


def embed_reply(
    target: Target, response: Any, headers: Mapping[str, str], sent: Mapping[str, Any]
) -> EmbedReply:
    body = response if isinstance(response, dict) else {}
    items = body.get("embeddings")
    items = items if isinstance(items, list) else []
    vectors = [e.get("values") if isinstance(e, dict) else None for e in items]
    return finish_embed(
        target,
        sent,
        headers,
        body_model=None,  # the reply does not say; nothing is guessed (rule 4)
        usage=Usage(input=None, output=0),  # batchEmbedContents reports no usage
        vectors=[v for v in vectors if isinstance(v, list)],
        indexes=list(range(len(vectors))),
    )
