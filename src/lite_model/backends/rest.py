"""A small JSON-over-HTTP client for the vendor APIs that no SDK lite-model uses speaks (design
§3.4, 0.8.0: Cohere's and Gemini's embeddings). It behaves as the SDK clients do where the client
relies on them: retries of the same request on 408 / 409 / 429 / 5xx and transport errors, at
most `max_retries` times, honouring `retry-after-ms` / `retry-after` (seconds or an HTTP date);
an HTTP error raised with its `status_code` (the 400 / 413 batch halving reads it); a response in
the raw shape `_send` reads (`.headers`, `.parse()`). Errors carry no request or reply text — a
transport exception can name headers, which hold the key (rule 8). Only a client it created
itself is closed."""

import asyncio
import time
from collections.abc import Callable, Mapping
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from typing import Any

import httpx2

RETRY_STATUSES = frozenset({408, 409, 429}) | frozenset(range(500, 600))


class HTTPStatusError(Exception):
    """A non-2xx reply. Its message is the status only: the body can quote the request."""

    def __init__(self, status_code: int) -> None:
        super().__init__(f"HTTP {status_code}")
        self.status_code = status_code


class TransportFailure(Exception):
    """A transport error, by type only: the original is dropped, as it can name headers."""


class Raw:
    """A reply in the raw-response shape `_send` reads."""

    def __init__(self, response: httpx2.Response) -> None:
        self.headers = response.headers
        self._response = response

    def parse(self) -> Any:
        return self._response.json()


def _seconds(value: str) -> float | None:
    """A retry-after value: seconds, or an HTTP date (as the SDKs read it)."""
    try:
        return float(value)
    except ValueError:
        pass
    try:
        when = parsedate_to_datetime(value)
    except (TypeError, ValueError):
        return None
    return max((when - datetime.now(UTC)).total_seconds(), 0.0)


def _delay(attempt: int, headers: Mapping[str, str] | None) -> float:
    """The SDKs' backoff: the server's retry-after when it gives one, else 0.5 s doubling to 8."""
    for name, scale in (("retry-after-ms", 0.001), ("retry-after", 1.0)):
        raw = headers.get(name) if headers else None
        seconds = _seconds(raw) if raw is not None else None
        if seconds is not None:
            return min(seconds * scale, 60.0)
    return min(0.5 * 2.0**attempt, 8.0)


def _outcome(
    attempt: int, retries: int, response: httpx2.Response | None, error: Exception | None
) -> Raw | float:
    """One attempt's result: the reply, or how long to wait before the next; raises when done."""
    last = attempt == retries
    if error is not None:
        if last:
            raise TransportFailure(type(error).__name__) from None
        return _delay(attempt, None)
    assert response is not None
    if response.is_success:
        return Raw(response)
    if last or response.status_code not in RETRY_STATUSES:
        raise HTTPStatusError(response.status_code)
    return _delay(attempt, response.headers)


class _Base:
    def __init__(
        self,
        base_url: str,
        headers: Mapping[str, str],
        timeout: httpx2.Timeout,
        max_retries: int,
        http_client: Any,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._headers = {"content-type": "application/json", **headers}
        self._timeout = timeout
        self._retries = max_retries
        self._owned = http_client is None

    def _url(self, path: str) -> str:
        return f"{self._base_url}/{path.lstrip('/')}"


class RestClient(_Base):
    def __init__(
        self,
        base_url: str,
        headers: Mapping[str, str],
        timeout: httpx2.Timeout,
        max_retries: int,
        http_client: httpx2.Client | None = None,
    ) -> None:
        super().__init__(base_url, headers, timeout, max_retries, http_client)
        self._client = http_client or httpx2.Client()

    def post(self, path: str, body: dict[str, Any]) -> Raw:
        url, attempt = self._url(path), 0
        while True:
            response, error = None, None
            try:
                response = self._client.post(
                    url, json=body, headers=self._headers, timeout=self._timeout
                )
            except httpx2.TransportError as exc:
                error = exc
            outcome = _outcome(attempt, self._retries, response, error)
            if isinstance(outcome, Raw):
                return outcome
            time.sleep(outcome)
            attempt += 1

    def close(self) -> None:
        if self._owned:
            self._client.close()


class AsyncRestClient(_Base):
    def __init__(
        self,
        base_url: str,
        headers: Mapping[str, str],
        timeout: httpx2.Timeout,
        max_retries: int,
        http_client: httpx2.AsyncClient | None = None,
    ) -> None:
        super().__init__(base_url, headers, timeout, max_retries, http_client)
        self._client = http_client or httpx2.AsyncClient()

    async def post(self, path: str, body: dict[str, Any]) -> Raw:
        url, attempt = self._url(path), 0
        while True:
            response, error = None, None
            try:
                response = await self._client.post(
                    url, json=body, headers=self._headers, timeout=self._timeout
                )
            except httpx2.TransportError as exc:
                error = exc
            outcome = _outcome(attempt, self._retries, response, error)
            if isinstance(outcome, Raw):
                return outcome
            await asyncio.sleep(outcome)
            attempt += 1

    async def close(self) -> None:
        if self._owned:
            await self._client.aclose()


class Owner:
    """The `create(**request)` owner `_send` / `_send_async` take: the request carries the path
    and the JSON body; `input` (the texts) stays for the reply checks."""

    def __init__(self, client: RestClient | AsyncRestClient) -> None:
        self._client = client

    def create(self, **request: Any) -> Any:
        return self._client.post(request["path"], request["json"])


type Connect = Callable[[Any, str | None, Any], RestClient | AsyncRestClient]


def connectors(auth: Callable[[str], dict[str, str]]) -> tuple[Connect, Connect]:
    """A vendor module's `connect` / `connect_async`: `auth(key)` gives its auth headers (an
    empty key when none is set; build() has refused a missing required key already)."""

    def make(asynchronous: bool) -> Connect:
        def connect(target: Any, key: str | None, http_client: Any) -> RestClient | AsyncRestClient:
            timeout = httpx2.Timeout(target.timeout_s, connect=target.connect_timeout_s)
            args = (target.base_url, auth(key or ""), timeout, target.max_retries)
            if asynchronous:
                return AsyncRestClient(*args, http_client=http_client)
            return RestClient(*args, http_client=http_client)

        return connect

    return make(False), make(True)
