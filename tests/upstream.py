"""A recording stand-in for an OpenAI-compatible endpoint, plugged in as the SDK's transport:
contract tests see the exact request the real SDK sends, and script the response."""

import json
from typing import Any

import httpx2


def completion(
    content: str | None = "ok",
    *,
    model: str = "qwen3:8b",
    finish: str = "stop",
    usage: dict[str, Any] | None = None,
    tool_calls: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """A Chat Completions response body."""
    reply: dict[str, Any] = {"role": "assistant", "content": content}
    if tool_calls is not None:
        reply["tool_calls"] = tool_calls
    return {
        "id": "chatcmpl-1",
        "object": "chat.completion",
        "created": 1,
        "model": model,
        "choices": [{"index": 0, "finish_reason": finish, "message": reply}],
        "usage": usage or {"prompt_tokens": 7, "completion_tokens": 3, "total_tokens": 10},
    }


def message(
    *blocks: dict[str, Any],
    model: str = "claude-x-20261101",
    stop: str = "end_turn",
    usage: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """An Anthropic Messages response body; `blocks` default to one text block "ok"."""
    return {
        "id": "msg_1",
        "type": "message",
        "role": "assistant",
        "model": model,
        "content": list(blocks) or [{"type": "text", "text": "ok"}],
        "stop_reason": stop,
        "stop_sequence": None,
        "usage": usage or {"input_tokens": 9, "output_tokens": 4},
    }


def tool_use(name: str, data: dict[str, Any], id: str = "toolu_1") -> dict[str, Any]:
    return {"type": "tool_use", "id": id, "name": name, "input": data}


def function_call(id: str, name: str, arguments: str) -> dict[str, Any]:
    """One entry of a Chat Completions `tool_calls`; `arguments` is JSON text, as on the wire."""
    return {"id": id, "type": "function", "function": {"name": name, "arguments": arguments}}


Scripted = tuple[int, dict[str, Any] | str] | tuple[int, dict[str, Any] | str, dict[str, str]]


class Upstream:
    """Answers each request with the next scripted (status, body); records every request."""

    def __init__(self, *replies: Scripted) -> None:
        """Each reply is (status, body) or (status, body, response headers)."""
        self._replies = list(replies) or [(200, completion())]
        self.requests: list[httpx2.Request] = []

    def _handle(self, request: httpx2.Request) -> httpx2.Response:
        self.requests.append(request)
        status, body, *extra = self._replies.pop(0) if len(self._replies) > 1 else self._replies[0]
        # An error says "retry in 1 ms" (the SDK ignores 0): its backoff would otherwise sleep
        # ~0.5 s in every retry test.
        headers = {"retry-after-ms": "1"} if status >= 400 else {}
        headers |= dict(*extra)
        if isinstance(body, str):
            return httpx2.Response(status, headers=headers, text=body)
        return httpx2.Response(status, headers=headers, json=body)

    def client(self) -> httpx2.Client:
        return httpx2.Client(transport=httpx2.MockTransport(self._handle))

    def async_client(self) -> httpx2.AsyncClient:
        return httpx2.AsyncClient(transport=httpx2.MockTransport(self._handle))

    def body(self, index: int = -1) -> dict[str, Any]:
        data = json.loads(self.requests[index].content)
        assert isinstance(data, dict)
        return data
