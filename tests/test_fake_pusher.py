from __future__ import annotations

import json
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

import pytest

pytest.importorskip("websockets")

from websockets.asyncio.client import connect as aconnect  # noqa: E402
from websockets.exceptions import ConnectionClosed  # noqa: E402
from websockets.sync.client import ClientConnection, connect  # noqa: E402

from tests.fake_pusher import FakePusherServer, sign  # noqa: E402


@pytest.fixture
def server() -> Iterator[FakePusherServer]:
    with FakePusherServer() as srv:
        yield srv


@contextmanager
def _open(srv: FakePusherServer) -> Iterator[tuple[ClientConnection, str]]:
    with connect(f"{srv.url}/app/test-app-key?protocol=7&client=test&version=1.0", open_timeout=3) as ws:
        msg = json.loads(ws.recv(timeout=3))
        assert msg["event"] == "pusher:connection_established"
        yield ws, json.loads(msg["data"])["socket_id"]


def _recv(ws: ClientConnection) -> dict[str, Any]:
    result: dict[str, Any] = json.loads(ws.recv(timeout=3))
    return result


class TestFakePusher:
    def test_connection_established(self, server: FakePusherServer) -> None:
        with connect(f"{server.url}/app/test-app-key?protocol=7&client=js&version=8", open_timeout=3) as ws:
            msg = json.loads(ws.recv(timeout=3))
            assert isinstance(msg["data"], str)
            payload = json.loads(msg["data"])
            left, right = payload["socket_id"].split(".")
            assert left.isdigit() and right.isdigit()
            assert payload["activity_timeout"] == 30
        assert server.connections_total == 1
        assert server.last_path == "/app/test-app-key"
        assert server.last_query == {"protocol": "7", "client": "js", "version": "8"}

    def test_wrong_key(self, server: FakePusherServer) -> None:
        with connect(f"{server.url}/app/other", open_timeout=3) as ws:
            assert _recv(ws)["data"]["code"] == 4001
            with pytest.raises(ConnectionClosed) as info:
                ws.recv(timeout=3)
        assert info.value.rcvd is not None and info.value.rcvd.code == 4001

    def test_ping_pong(self, server: FakePusherServer) -> None:
        with _open(server) as (ws, _sid):
            ws.send(json.dumps({"event": "pusher:ping", "data": {}}))
            assert _recv(ws) == {"event": "pusher:pong", "data": {}}
        assert server.pings_received == 1

    def test_no_pong_when_disabled(self, server: FakePusherServer) -> None:
        server.reply_to_pings = False
        with _open(server) as (ws, _sid):
            ws.send(json.dumps({"event": "pusher:ping", "data": {}}))
            server.wait_for(lambda: server.pings_received == 1)
            with pytest.raises(TimeoutError):
                ws.recv(timeout=0.1)

    def test_private_subscribe_ok_and_push(self, server: FakePusherServer) -> None:
        with _open(server) as (ws, sid):
            auth = sign("test-app-key", sid, "private-x")
            ws.send(json.dumps({"event": "pusher:subscribe", "data": {"channel": "private-x", "auth": auth}}))
            msg = _recv(ws)
            assert msg == {"event": "pusher_internal:subscription_succeeded", "channel": "private-x", "data": "{}"}
            assert server.subscriptions == [("private-x", auth)]

            server.push_event("private-x", "done", {"a": 1})
            msg = _recv(ws)
            assert msg["event"] == "done" and msg["channel"] == "private-x"
            assert msg["data"] == '{"a": 1}'

            ws.send(json.dumps({"event": "pusher:unsubscribe", "data": {"channel": "private-x"}}))
            ws.send(json.dumps({"event": "pusher:ping", "data": {}}))
            assert _recv(ws)["event"] == "pusher:pong"
            server.push_event("private-x", "later", "x")
            with pytest.raises(TimeoutError):
                ws.recv(timeout=0.1)

    def test_private_subscribe_auth_failure(self, server: FakePusherServer) -> None:
        with _open(server) as (ws, _sid):
            ws.send(json.dumps({"event": "pusher:subscribe", "data": {"channel": "private-x", "auth": "bad"}}))
            msg = _recv(ws)
            assert msg["event"] == "pusher:subscription_error"
            assert msg["data"]["status"] == 401

    def test_public_subscribe_without_auth(self, server: FakePusherServer) -> None:
        with _open(server) as (ws, _sid):
            ws.send(json.dumps({"event": "pusher:subscribe", "data": {"channel": "pub"}}))
            assert _recv(ws)["event"] == "pusher_internal:subscription_succeeded"

    def test_garbage_ignored_and_send_raw(self, server: FakePusherServer) -> None:
        with _open(server) as (ws, _sid):
            ws.send("not json")
            ws.send("[1]")
            server.send_raw("hello")
            assert ws.recv(timeout=3) == "hello"

    def test_close_all(self, server: FakePusherServer) -> None:
        with _open(server) as (ws, _sid):
            server.close_all(4200, "bye")
            with pytest.raises(ConnectionClosed) as info:
                ws.recv(timeout=3)
        assert info.value.rcvd is not None
        assert info.value.rcvd.code == 4200
        assert info.value.rcvd.reason == "bye"

    def test_drop_all(self, server: FakePusherServer) -> None:
        with _open(server) as (ws, _sid):
            server.drop_all()
            with pytest.raises(ConnectionClosed) as info:
                ws.recv(timeout=3)
        assert info.value.rcvd is None

    def test_reject_connections(self, server: FakePusherServer) -> None:
        server.reject_connections_with = 4100
        with connect(f"{server.url}/app/test-app-key", open_timeout=3) as ws:
            msg = _recv(ws)
            assert msg["event"] == "pusher:error" and msg["data"]["code"] == 4100
            with pytest.raises(ConnectionClosed) as info:
                ws.recv(timeout=3)
        assert info.value.rcvd is not None and info.value.rcvd.code == 4100


class TestFakePusherAsync:
    async def test_async_client_roundtrip(self) -> None:
        with FakePusherServer() as srv:
            async with aconnect(f"{srv.url}/app/test-app-key?protocol=7") as ws:
                msg = json.loads(await ws.recv())
                sid = json.loads(msg["data"])["socket_id"]
                auth = sign("test-app-key", sid, "private-a")
                sub = {"event": "pusher:subscribe", "data": {"channel": "private-a", "auth": auth}}
                await ws.send(json.dumps(sub))
                assert json.loads(await ws.recv())["event"] == "pusher_internal:subscription_succeeded"
                srv.push_event("private-a", "evt", {"k": "v"})
                got = json.loads(await ws.recv())
                assert json.loads(got["data"]) == {"k": "v"}
