import socket

import pytest


@pytest.fixture(autouse=True)
def _no_network(monkeypatch):
    """Offline guarantee: any attempt to open a connection fails the test."""

    def blocked(*args, **kwargs):
        raise RuntimeError("network access is not allowed in tests")

    monkeypatch.setattr(socket.socket, "connect", blocked)
    monkeypatch.setattr(socket, "getaddrinfo", blocked)
