"""A local fake HTTP server for the adapter tests. Test code only.

The server binds to 127.0.0.1 on a free port and plays back scripted responses, recording every
request. ``tests/conftest.py`` blocks all sockets for every test; ``loopback_only`` is a narrow,
per-test exception that lets connections to loopback addresses through and keeps blocking
everything else, so a test can still prove that no real host is reachable.
"""

import json
import socket
import threading
from collections import deque
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from socketserver import TCPServer

import pytest

# Captured at import time (during collection), before any test patches the socket module.
_REAL_CONNECT = socket.socket.connect
_REAL_GETADDRINFO = socket.getaddrinfo
LOOPBACK = frozenset({"127.0.0.1", "::1", "localhost"})
BLOCKED = "network access is not allowed in tests"


@pytest.fixture
def loopback_only(monkeypatch):
    """Allow connections to loopback addresses for this test; everything else stays blocked."""

    def connect(self, address, *args, **kwargs):
        if isinstance(address, tuple) and address[0] in LOOPBACK:
            return _REAL_CONNECT(self, address, *args, **kwargs)
        raise RuntimeError(BLOCKED)

    def getaddrinfo(host, *args, **kwargs):
        if host in LOOPBACK:
            return _REAL_GETADDRINFO(host, *args, **kwargs)
        raise RuntimeError(BLOCKED)

    monkeypatch.setattr(socket.socket, "connect", connect)
    monkeypatch.setattr(socket, "getaddrinfo", getaddrinfo)


@dataclass
class Reply:
    """One scripted response. ``drop=True`` closes the connection without answering."""

    status: int = 200
    body: object = b""  # bytes, str, or anything JSON-serialisable
    headers: dict[str, str] = field(default_factory=dict)
    drop: bool = False

    def payload(self) -> bytes:
        if isinstance(self.body, bytes):
            return self.body
        if isinstance(self.body, str):
            return self.body.encode("utf-8")
        return json.dumps(self.body, ensure_ascii=False).encode("utf-8")


@dataclass
class Recorded:
    method: str
    path: str
    headers: dict[str, str]  # lower-cased names
    body: bytes

    def json(self):
        return json.loads(self.body.decode("utf-8"))


def completion(
    text="ok", *, model="served-model", usage=None, finish_reason="stop", **extra
) -> dict:
    """A well-formed chat-completions body."""
    document: dict = {
        "id": "chatcmpl-test",
        "object": "chat.completion",
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": text},
                "finish_reason": finish_reason,
            }
        ],
    }
    if usage is not None:
        document["usage"] = usage
    document.update(extra)
    return document


class _Server(ThreadingHTTPServer):
    daemon_threads = True

    def server_bind(self):  # HTTPServer.server_bind would do a reverse lookup
        TCPServer.server_bind(self)
        self.server_name, self.server_port = "127.0.0.1", self.server_address[1]


class FakeServer:
    def __init__(self) -> None:
        self.replies: deque[Reply] = deque()
        self.requests: list[Recorded] = []
        owner = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):  # noqa: N802 (the name is fixed by http.server)
                length = int(self.headers.get("Content-Length", "0"))
                body = self.rfile.read(length)
                owner.requests.append(
                    Recorded(
                        "POST",
                        self.path,
                        {k.lower(): v for k, v in self.headers.items()},
                        body,
                    )
                )
                reply = owner.replies.popleft() if owner.replies else Reply(500, "script exhausted")
                if reply.drop:
                    self.close_connection = True
                    return
                data = reply.payload()
                self.send_response(reply.status)
                self.send_header("Content-Length", str(len(data)))
                for name, value in reply.headers.items():
                    self.send_header(name, value)
                self.end_headers()
                self.wfile.write(data)

            def log_message(self, *args):
                pass

        self._server = _Server(("127.0.0.1", 0), Handler)
        self._thread = threading.Thread(
            target=lambda: self._server.serve_forever(poll_interval=0.01), daemon=True
        )
        self._thread.start()

    @property
    def port(self) -> int:
        return self._server.server_address[1]

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.port}/v1"

    def enqueue(self, *replies: Reply) -> None:
        self.replies.extend(replies)

    def stop(self) -> None:
        self._server.shutdown()
        self._server.server_close()
        self._thread.join(timeout=5)


@pytest.fixture
def server(loopback_only):
    fake = FakeServer()
    try:
        yield fake
    finally:
        fake.stop()


@pytest.fixture
def blackhole(loopback_only):
    """A listening socket that never accepts or answers: connections succeed, reads time out."""
    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listener.bind(("127.0.0.1", 0))
    listener.listen(8)
    try:
        yield f"http://127.0.0.1:{listener.getsockname()[1]}/v1"
    finally:
        listener.close()


def closed_port_url() -> str:
    """A loopback URL where nothing is listening (connection refused)."""
    probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    probe.bind(("127.0.0.1", 0))
    port = probe.getsockname()[1]
    probe.close()
    return f"http://127.0.0.1:{port}/v1"
