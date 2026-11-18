"""Shared test setup. Unit and contract tests never open a socket (AGENTS.md rule 11): only tests
marked `ollama` or `litellm` (opt-in) may reach a real server."""

import socket
from typing import Any

import pytest

LIVE_MARKERS = {"ollama", "litellm"}


@pytest.fixture(autouse=True)
def _no_network(request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch) -> None:
    if LIVE_MARKERS & {m.name for m in request.node.iter_markers()}:
        return

    def refuse(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("a unit or contract test tried to open a network connection")

    monkeypatch.setattr(socket.socket, "connect", refuse)
    monkeypatch.setattr(socket.socket, "connect_ex", refuse)
    monkeypatch.setattr(socket, "create_connection", refuse)
