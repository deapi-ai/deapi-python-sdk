from __future__ import annotations

import asyncio
import json
import sys
import threading
import time
from collections.abc import Callable, Iterator
from typing import Any

import httpx
import pytest
import respx

pytest.importorskip("websockets")

from deapi_sdk import AsyncDeapiClient, DeapiClient  # noqa: E402
from deapi_sdk._exceptions import (  # noqa: E402
    AuthenticationError,
    JobTimeoutError,
    PermissionDeniedError,
    RealtimeError,
)
from deapi_sdk._realtime import _protocol as protocol  # noqa: E402
from deapi_sdk._realtime import _session as session_module  # noqa: E402
from deapi_sdk._realtime._engine import Engine  # noqa: E402
from deapi_sdk._types import JobStatus  # noqa: E402
from deapi_sdk.types.common import JobStatusEvent  # noqa: E402
from tests.fake_pusher import FakePusherServer, sign  # noqa: E402

TEST_BASE_URL = "https://test.deapi.ai"
TEST_API_KEY = "test-key-123"
APP_KEY = "test-app-key"
CLIENT_ID = "test-client-1"
CHANNEL = f"private-client.{CLIENT_ID}"
AUTH_URL = f"{TEST_BASE_URL}/broadcasting/auth"
REST_URL = "https://rest.test/final.png"
EVENT_URL = "https://event.test/stale.png"


def status_url(request_id: str) -> str:
    return f"{TEST_BASE_URL}/api/v2/jobs/{request_id}"


# --------------------------------------------------------------------------- helpers


class AuthEndpoint:
    """respx side effect for POST /broadcasting/auth that signs like the real backend would."""

    def __init__(self) -> None:
        self.bodies: list[dict[str, Any]] = []
        self.headers: list[httpx.Headers] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        self.bodies.append(body)
        self.headers.append(request.headers)
        return httpx.Response(200, json={"auth": sign(APP_KEY, body["socket_id"], body["channel_name"])})


class Statuses:
    """respx side effect for the job status endpoint: replays ``sequence`` (last item repeats)."""

    def __init__(
        self,
        *sequence: dict[str, Any] | Callable[[], dict[str, Any]],
        on_call: Callable[[int], None] | None = None,
    ) -> None:
        self.sequence = sequence
        self.on_call = on_call
        self.calls = 0
        self._lock = threading.Lock()

    def __call__(self, request: httpx.Request) -> httpx.Response:
        with self._lock:
            n = self.calls
            self.calls += 1
        if self.on_call is not None:
            self.on_call(n)
        item = self.sequence[min(n, len(self.sequence) - 1)]
        return httpx.Response(200, json={"data": item() if callable(item) else item})


def processing() -> dict[str, Any]:
    return {"status": "processing", "progress": 0.0}


def done() -> dict[str, Any]:
    return {"status": "done", "progress": 100.0, "result_url": REST_URL}


def failed() -> dict[str, Any]:
    return {"status": "error", "progress": 0.0}


def push(server: FakePusherServer, request_id: str, status: str, progress: str | None = None, **extra: Any) -> None:
    data: dict[str, Any] = {"request_id": request_id, "status": status, **extra}
    if progress is not None:
        data["progress"] = progress
    server.push_event(CHANNEL, protocol.STATUS_EVENT, data)


def in_thread(fn: Callable[..., Any], *args: Any) -> threading.Thread:
    thread = threading.Thread(target=fn, args=args, daemon=True)
    thread.start()
    return thread


def connected_again(server: FakePusherServer, rt: Any, subscriptions: int = 2) -> None:
    server.wait_for(lambda: len(server.subscriptions) >= subscriptions and rt.connected)


async def until(predicate: Callable[[], bool], timeout: float = 5.0) -> None:
    deadline = time.monotonic() + timeout
    while not predicate():
        if time.monotonic() > deadline:
            raise AssertionError(f"condition not met within {timeout}s")
        await asyncio.sleep(0.01)


# --------------------------------------------------------------------------- fixtures


@pytest.fixture(autouse=True)
def fast_realtime(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(protocol, "backoff_delay", lambda attempt, **kw: 0.01)
    monkeypatch.setattr(session_module, "_CONFIRM_INTERVAL", 0.05)
    for name in ("DEAPI_CLIENT_ID", "DEAPI_WS_URL", "DEAPI_WS_APP_KEY"):
        monkeypatch.delenv(name, raising=False)


@pytest.fixture
def server() -> Iterator[FakePusherServer]:
    with FakePusherServer(app_key=APP_KEY) as srv:
        yield srv


@pytest.fixture
def api() -> Iterator[respx.MockRouter]:
    with respx.mock(assert_all_called=False) as router:
        yield router


@pytest.fixture
def auth(api: respx.MockRouter) -> AuthEndpoint:
    endpoint = AuthEndpoint()
    api.post(AUTH_URL).mock(side_effect=endpoint)
    return endpoint


def make_rt(client: DeapiClient, server: FakePusherServer, **kwargs: Any) -> Any:
    kwargs.setdefault("client_id", CLIENT_ID)
    kwargs.setdefault("connect_timeout", 2)
    return client.realtime(ws_url=server.url, app_key=APP_KEY, **kwargs)


def make_async_rt(client: AsyncDeapiClient, server: FakePusherServer, **kwargs: Any) -> Any:
    kwargs.setdefault("client_id", CLIENT_ID)
    kwargs.setdefault("connect_timeout", 2)
    return client.realtime(ws_url=server.url, app_key=APP_KEY, **kwargs)


# =========================================================================== sync


class TestConnect:
    def test_handshake_and_subscription(
        self, client: DeapiClient, server: FakePusherServer, auth: AuthEndpoint
    ) -> None:
        from deapi_sdk._version import __version__

        rt = make_rt(client, server)
        assert rt.connected is False
        with rt:
            assert rt.connected is True
            assert server.last_path == f"/app/{APP_KEY}"
            assert server.last_query == {"protocol": "7", "client": "deapi-python", "version": __version__}
            assert len(server.subscriptions) == 1
            channel, signature = server.subscriptions[0]
            assert channel == CHANNEL
            assert len(auth.bodies) == 1
            assert signature == sign(APP_KEY, auth.bodies[0]["socket_id"], CHANNEL)
        assert rt.connected is False

    def test_auth_request(self, client: DeapiClient, server: FakePusherServer, auth: AuthEndpoint) -> None:
        with make_rt(client, server):
            pass
        body = auth.bodies[0]
        assert set(body) == {"socket_id", "channel_name"}
        assert body["channel_name"] == CHANNEL
        assert auth.headers[0]["authorization"] == f"Bearer {TEST_API_KEY}"

    def test_repr_hides_api_key(self, client: DeapiClient, server: FakePusherServer, auth: AuthEndpoint) -> None:
        rt = make_rt(client, server)
        text = repr(rt)
        assert CLIENT_ID in text
        assert TEST_API_KEY not in text
        with rt:
            assert "connected=True" in repr(rt)
            assert TEST_API_KEY not in repr(rt)

    def test_close_is_idempotent(self, client: DeapiClient, server: FakePusherServer, auth: AuthEndpoint) -> None:
        rt = make_rt(client, server)
        rt.connect()
        rt.close()
        rt.close()
        assert rt.connected is False

    def test_close_without_connect(self, client: DeapiClient, server: FakePusherServer) -> None:
        rt = make_rt(client, server)
        rt.close()

    def test_connect_twice_fails(self, client: DeapiClient, server: FakePusherServer, auth: AuthEndpoint) -> None:
        with make_rt(client, server) as rt, pytest.raises(RealtimeError, match="already started"):
            rt.connect()

    def test_client_close_closes_sessions(
        self, client: DeapiClient, server: FakePusherServer, auth: AuthEndpoint
    ) -> None:
        rt = make_rt(client, server)
        rt.connect()
        assert rt.connected
        client.close()
        assert rt.connected is False

    def test_missing_websockets_extra(self, client: DeapiClient, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setitem(sys.modules, "websockets.asyncio.client", None)
        rt = client.realtime(client_id=CLIENT_ID, ws_url="ws://127.0.0.1:1", app_key=APP_KEY)
        with pytest.raises(ImportError, match=r"deapi-sdk\[ws\]"):
            rt.connect()


class TestSettings:
    def test_explicit_client_id_wins_over_env(
        self, client: DeapiClient, server: FakePusherServer, auth: AuthEndpoint, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("DEAPI_CLIENT_ID", "from-env")
        with make_rt(client, server, client_id="explicit-id") as rt:
            assert rt.client_id == "explicit-id"
        assert server.subscriptions[0][0] == "private-client.explicit-id"

    def test_env_client_id_used(
        self, client: DeapiClient, server: FakePusherServer, auth: AuthEndpoint, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("DEAPI_CLIENT_ID", "from-env")
        with client.realtime(ws_url=server.url, app_key=APP_KEY, connect_timeout=2) as rt:
            assert rt.client_id == "from-env"
        assert server.subscriptions[0][0] == "private-client.from-env"

    def test_missing_client_id(self, client: DeapiClient) -> None:
        with pytest.raises(ValueError, match="DEAPI_CLIENT_ID"):
            client.realtime()

    def test_blank_env_client_id_is_missing(self, client: DeapiClient, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("DEAPI_CLIENT_ID", "   ")
        with pytest.raises(ValueError, match="DEAPI_CLIENT_ID"):
            client.realtime()

    def test_int_client_id(
        self, client: DeapiClient, server: FakePusherServer, auth: AuthEndpoint
    ) -> None:
        with make_rt(client, server, client_id=12345) as rt:
            assert rt.client_id == "12345"
        assert server.subscriptions[0][0] == "private-client.12345"

    def test_env_ws_url_and_app_key(
        self, client: DeapiClient, server: FakePusherServer, auth: AuthEndpoint, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("DEAPI_WS_URL", server.url)
        monkeypatch.setenv("DEAPI_WS_APP_KEY", APP_KEY)
        with client.realtime(client_id=CLIENT_ID, connect_timeout=2):
            pass
        assert server.last_path == f"/app/{APP_KEY}"

    def test_invalid_ws_url(self, client: DeapiClient) -> None:
        with pytest.raises(ValueError, match="ws_url"):
            client.realtime(client_id=CLIENT_ID, ws_url="https://example.test")

    def test_invalid_connect_timeout(self, client: DeapiClient) -> None:
        with pytest.raises(ValueError, match="connect_timeout"):
            client.realtime(client_id=CLIENT_ID, connect_timeout=0)


class TestConnectFailures:
    def test_auth_forbidden(self, client: DeapiClient, server: FakePusherServer, api: respx.MockRouter) -> None:
        api.post(AUTH_URL).mock(return_value=httpx.Response(403, json={"message": "Forbidden"}))
        rt = make_rt(client, server)
        with pytest.raises(PermissionDeniedError):
            rt.connect()
        assert rt.connected is False

    def test_auth_unauthorized(self, client: DeapiClient, server: FakePusherServer, api: respx.MockRouter) -> None:
        api.post(AUTH_URL).mock(return_value=httpx.Response(401, json={"message": "Unauthenticated."}))
        with pytest.raises(AuthenticationError), make_rt(client, server):
            pass

    def test_auth_response_without_signature(
        self, client: DeapiClient, server: FakePusherServer, api: respx.MockRouter
    ) -> None:
        api.post(AUTH_URL).mock(return_value=httpx.Response(200, json={"nope": 1}))
        with pytest.raises(RealtimeError, match="auth signature"):
            make_rt(client, server).connect()

    def test_subscription_refused(self, client: DeapiClient, server: FakePusherServer, auth: AuthEndpoint) -> None:
        server.auth_validator = lambda socket_id, channel, signature: False
        with pytest.raises(RealtimeError, match="client_id") as info:
            make_rt(client, server).connect()
        assert info.value.status_code == 401
        assert CHANNEL in str(info.value)

    def test_server_rejects_connection(
        self, client: DeapiClient, server: FakePusherServer, auth: AuthEndpoint
    ) -> None:
        server.reject_connections_with = 4001
        started = time.monotonic()
        with pytest.raises(RealtimeError) as info:
            make_rt(client, server, connect_timeout=10).connect()
        assert info.value.close_code == 4001
        assert time.monotonic() - started < 5
        assert server.connections_total == 1  # fatal: no retries

    def test_wrong_app_key(self, client: DeapiClient, server: FakePusherServer, auth: AuthEndpoint) -> None:
        started = time.monotonic()
        with pytest.raises(RealtimeError) as info:
            client.realtime(
                client_id=CLIENT_ID, ws_url=server.url, app_key="wrong-key", connect_timeout=10
            ).connect()
        assert info.value.close_code == 4001
        assert time.monotonic() - started < 5

    def test_connect_timeout(self, client: DeapiClient, server: FakePusherServer, auth: AuthEndpoint) -> None:
        server.suppress_subscription_reply = True
        rt = make_rt(client, server, connect_timeout=0.5)
        rt._engine._subscribe_timeout = 0.1
        started = time.monotonic()
        with pytest.raises(RealtimeError, match="within"):
            rt.connect()
        assert time.monotonic() - started < 3
        assert server.connections_total >= 1


class TestWait:
    def test_already_done(
        self, client: DeapiClient, server: FakePusherServer, auth: AuthEndpoint, api: respx.MockRouter
    ) -> None:
        route = Statuses(done)
        api.get(status_url("req-1")).mock(side_effect=route)
        seen: list[JobStatusEvent] = []
        with make_rt(client, server) as rt:
            result = rt.wait("req-1", on_progress=seen.append)
        assert result.status == JobStatus.DONE
        assert result.result_url == REST_URL
        assert route.calls == 1
        assert seen == []

    def test_progress_then_done(
        self, client: DeapiClient, server: FakePusherServer, auth: AuthEndpoint, api: respx.MockRouter
    ) -> None:
        def on_call(n: int) -> None:
            if n == 0:
                push(server, "req-1", "processing", "10.00")
                push(server, "req-1", "processing", "55.5")
                push(server, "req-1", "done", "100.00", result_url=EVENT_URL)

        route = Statuses(processing, done, on_call=on_call)
        api.get(status_url("req-1")).mock(side_effect=route)
        seen: list[JobStatusEvent] = []
        with make_rt(client, server) as rt:
            result = rt.wait("req-1", on_progress=seen.append, poll_interval=30)
        assert [(e.status, e.progress) for e in seen] == [
            (JobStatus.PROCESSING, 10.0),
            (JobStatus.PROCESSING, 55.5),
            (JobStatus.DONE, 100.0),
        ]
        assert result.status == JobStatus.DONE
        assert result.result_url == REST_URL  # from REST, not the event
        assert route.calls == 2

    def test_error_job_found_by_polling(
        self, client: DeapiClient, server: FakePusherServer, auth: AuthEndpoint, api: respx.MockRouter
    ) -> None:
        route = Statuses(processing, processing, failed)
        api.get(status_url("req-1")).mock(side_effect=route)
        with make_rt(client, server) as rt:
            result = rt.wait("req-1", poll_interval=0.05, max_wait=5)
        assert result.status == JobStatus.ERROR
        assert route.calls == 3

    def test_error_event_triggers_immediate_check(
        self, client: DeapiClient, server: FakePusherServer, auth: AuthEndpoint, api: respx.MockRouter
    ) -> None:
        route = Statuses(processing, failed, on_call=lambda n: push(server, "req-1", "error") if n == 0 else None)
        api.get(status_url("req-1")).mock(side_effect=route)
        started = time.monotonic()
        with make_rt(client, server) as rt:
            result = rt.wait("req-1", poll_interval=30)
        assert result.status == JobStatus.ERROR
        assert route.calls == 2
        assert time.monotonic() - started < 3

    def test_done_event_before_rest_agrees(
        self, client: DeapiClient, server: FakePusherServer, auth: AuthEndpoint, api: respx.MockRouter
    ) -> None:
        route = Statuses(
            processing, processing, done, on_call=lambda n: push(server, "req-1", "done") if n == 0 else None
        )
        api.get(status_url("req-1")).mock(side_effect=route)
        started = time.monotonic()
        with make_rt(client, server) as rt:
            result = rt.wait("req-1", poll_interval=30)
        assert result.status == JobStatus.DONE
        assert route.calls == 3
        assert time.monotonic() - started < 3

    def test_other_request_ids_ignored(
        self, client: DeapiClient, server: FakePusherServer, auth: AuthEndpoint, api: respx.MockRouter
    ) -> None:
        def on_call(n: int) -> None:
            if n == 0:
                push(server, "req-other", "processing", "20.00")
                push(server, "req-other", "done")
                push(server, "req-1", "processing", "30.00")
                push(server, "req-1", "done")

        route = Statuses(processing, done, on_call=on_call)
        api.get(status_url("req-1")).mock(side_effect=route)
        seen: list[JobStatusEvent] = []
        with make_rt(client, server) as rt:
            result = rt.wait("req-1", on_progress=seen.append, poll_interval=30)
        assert {e.request_id for e in seen} == {"req-1"}
        assert len(seen) == 2
        assert result.status == JobStatus.DONE
        assert route.calls == 2  # the other job's "done" did not trigger a status check

    def test_accepts_job_handle(
        self, client: DeapiClient, server: FakePusherServer, auth: AuthEndpoint, api: respx.MockRouter
    ) -> None:
        api.get(status_url("req-1")).mock(side_effect=Statuses(done))
        job = client.jobs.get("req-1")
        with make_rt(client, server) as rt:
            assert rt.wait(job).status == JobStatus.DONE

    def test_max_wait_exceeded(
        self, client: DeapiClient, server: FakePusherServer, auth: AuthEndpoint, api: respx.MockRouter
    ) -> None:
        api.get(status_url("req-1")).mock(side_effect=Statuses(processing))
        with make_rt(client, server) as rt, pytest.raises(JobTimeoutError) as info:
            rt.wait("req-1", max_wait=0.3, poll_interval=0.05)
        assert info.value.request_id == "req-1"

    def test_before_connect(self, client: DeapiClient, server: FakePusherServer) -> None:
        rt = make_rt(client, server)
        with pytest.raises(RealtimeError, match="not connected"):
            rt.wait("req-1")
        with pytest.raises(RealtimeError, match="not connected"):
            next(rt.events())

    @pytest.mark.parametrize(("kwargs", "match"), [
        ({"max_wait": 0}, "max_wait"),
        ({"max_wait": -1}, "max_wait"),
        ({"poll_interval": 0}, "poll_interval"),
        ({"poll_interval": -1}, "poll_interval"),
    ])
    def test_invalid_arguments(
        self, client: DeapiClient, server: FakePusherServer, kwargs: dict[str, float], match: str
    ) -> None:
        rt = make_rt(client, server)
        with pytest.raises(ValueError, match=match):
            rt.wait("req-1", **kwargs)

    def test_empty_request_id(
        self, client: DeapiClient, server: FakePusherServer, auth: AuthEndpoint
    ) -> None:
        with make_rt(client, server) as rt, pytest.raises(ValueError, match="request_id"):
            rt.wait("")

    def test_on_progress_exception_propagates(
        self, client: DeapiClient, server: FakePusherServer, auth: AuthEndpoint, api: respx.MockRouter
    ) -> None:
        route = Statuses(processing, on_call=lambda n: push(server, "req-1", "processing", "5") if n == 0 else None)
        api.get(status_url("req-1")).mock(side_effect=route)

        def boom(event: JobStatusEvent) -> None:
            raise RuntimeError("boom")

        with make_rt(client, server) as rt, pytest.raises(RuntimeError, match="boom"):
            rt.wait("req-1", on_progress=boom, poll_interval=30)

    def test_wait_after_close_raises(
        self, client: DeapiClient, server: FakePusherServer, auth: AuthEndpoint, api: respx.MockRouter
    ) -> None:
        api.get(status_url("req-1")).mock(side_effect=Statuses(processing))
        rt = make_rt(client, server)
        rt.connect()
        rt.close()
        with pytest.raises(RealtimeError, match="closed"):
            rt.wait("req-1", poll_interval=30)


class TestReconnect:
    @pytest.mark.parametrize("how", ["close_4200", "close_4100", "drop", "close_1000"])
    def test_reconnects_and_reauthorizes(
        self, client: DeapiClient, server: FakePusherServer, auth: AuthEndpoint, how: str
    ) -> None:
        with make_rt(client, server) as rt:
            if how == "drop":
                server.drop_all()
            else:
                server.close_all(int(how.split("_")[1]))
            connected_again(server, rt)
            assert server.connections_total >= 2
            assert len(auth.bodies) >= 2
            assert auth.bodies[0]["socket_id"] != auth.bodies[1]["socket_id"]
            assert server.subscriptions[0][1] != server.subscriptions[1][1]
            for (channel, signature), body in zip(server.subscriptions, auth.bodies, strict=False):
                assert channel == CHANNEL
                assert signature == sign(APP_KEY, body["socket_id"], CHANNEL)

    def test_wait_rechecks_status_after_reconnect(
        self, client: DeapiClient, server: FakePusherServer, auth: AuthEndpoint, api: respx.MockRouter
    ) -> None:
        def on_call(n: int) -> None:
            if n == 0:
                in_thread(server.close_all, 4200)

        def maybe_done() -> dict[str, Any]:
            return done() if len(server.subscriptions) >= 2 else processing()

        route = Statuses(processing, maybe_done, on_call=on_call)
        api.get(status_url("req-1")).mock(side_effect=route)
        started = time.monotonic()
        with make_rt(client, server) as rt:
            result = rt.wait("req-1", poll_interval=30, max_wait=1)
        assert result.status == JobStatus.DONE
        assert time.monotonic() - started < 0.8
        assert server.connections_total >= 2

    def test_events_keep_flowing_after_reconnect(
        self, client: DeapiClient, server: FakePusherServer, auth: AuthEndpoint
    ) -> None:
        seen: list[JobStatusEvent] = []
        with make_rt(client, server) as rt:
            thread = in_thread(lambda: seen.extend(e for _, e in zip(range(1), rt.events(), strict=False)))
            server.wait_for(lambda: bool(rt._engine._sinks))
            server.close_all(4200)
            connected_again(server, rt)
            push(server, "req-2", "processing")
            thread.join(5)
        assert [e.request_id for e in seen] == ["req-2"]

    @pytest.mark.parametrize("code", [4009, 4001])
    def test_fatal_close_fails_pending_wait(
        self, client: DeapiClient, server: FakePusherServer, auth: AuthEndpoint, api: respx.MockRouter, code: int
    ) -> None:
        route = Statuses(processing, on_call=lambda n: in_thread(server.close_all, code) if n == 0 else None)
        api.get(status_url("req-1")).mock(side_effect=route)
        with make_rt(client, server) as rt:
            with pytest.raises(RealtimeError) as info:
                rt.wait("req-1", poll_interval=30, max_wait=10)
            assert info.value.close_code == code
            assert rt.connected is False
            with pytest.raises(RealtimeError):
                rt.wait("req-1")
        assert server.connections_total == 1  # no reconnect attempts

    def test_fatal_close_fails_events(
        self, client: DeapiClient, server: FakePusherServer, auth: AuthEndpoint
    ) -> None:
        errors: list[BaseException] = []

        def consume(rt: Any) -> None:
            try:
                for _ in rt.events():
                    pass
            except BaseException as exc:
                errors.append(exc)

        with make_rt(client, server) as rt:
            thread = in_thread(consume, rt)
            server.wait_for(lambda: bool(rt._engine._sinks))
            server.close_all(4009)
            thread.join(5)
        assert len(errors) == 1
        assert isinstance(errors[0], RealtimeError)
        assert errors[0].close_code == 4009


class TestKeepalive:
    def test_client_pings_after_server_activity_timeout(
        self, client: DeapiClient, auth: AuthEndpoint
    ) -> None:
        with FakePusherServer(app_key=APP_KEY, activity_timeout=1) as server, make_rt(client, server) as rt:
            assert server.pings_received == 0
            server.wait_for(lambda: server.pings_received >= 1, timeout=4)
            assert rt.connected
            assert server.connections_total == 1

    def test_ping_loop_keeps_connection_alive(
        self, client: DeapiClient, server: FakePusherServer, auth: AuthEndpoint
    ) -> None:
        rt = make_rt(client, server)
        rt._engine._max_activity_timeout = 0.05
        with rt:
            server.wait_for(lambda: server.pings_received >= 3)
            assert rt.connected
            assert server.connections_total == 1

    def test_no_pong_triggers_reconnect(
        self, client: DeapiClient, server: FakePusherServer, auth: AuthEndpoint
    ) -> None:
        server.reply_to_pings = False
        rt = make_rt(client, server)
        rt._engine._max_activity_timeout = 0.1
        rt._engine._pong_timeout = 0.1
        with rt:
            # The reconnect is visible to the server before its channel authorization finishes.
            server.wait_for(lambda: server.connections_total >= 2 and len(auth.bodies) >= 2)

    def test_unanswered_server_ping_is_not_fatal(
        self, client: DeapiClient, server: FakePusherServer, auth: AuthEndpoint
    ) -> None:
        with make_rt(client, server) as rt:
            server.send_raw('{"event":"pusher:ping","data":{}}')
            server.send_raw("not json at all")
            server.send_raw('{"event":"pusher:error","data":{"code":4200,"message":"minor"}}')
            time.sleep(0.1)
            assert rt.connected
            assert server.connections_total == 1


class TestEvents:
    def test_events_in_order_skip_malformed_and_end_on_close(
        self, client: DeapiClient, server: FakePusherServer, auth: AuthEndpoint
    ) -> None:
        received: list[JobStatusEvent] = []
        finished = threading.Event()

        def consume(rt: Any) -> None:
            for event in rt.events():
                received.append(event)
            finished.set()

        with make_rt(client, server) as rt:
            thread = in_thread(consume, rt)
            server.wait_for(lambda: bool(rt._engine._sinks))
            push(server, "req-1", "processing")
            server.push_event(CHANNEL, protocol.STATUS_EVENT, {"status": "done"})  # no request_id
            server.push_event(CHANNEL, protocol.STATUS_EVENT, "garbage")
            push(server, "req-2", "processing", "12.5")
            push(server, "req-1", "done")
            server.wait_for(lambda: len(received) >= 3)
            rt.close()
            thread.join(5)
        assert finished.is_set()
        assert [(e.request_id, e.status) for e in received] == [
            ("req-1", JobStatus.PROCESSING),
            ("req-2", JobStatus.PROCESSING),
            ("req-1", JobStatus.DONE),
        ]
        assert received[1].progress == 12.5

    def test_events_on_other_channels_ignored(
        self, client: DeapiClient, server: FakePusherServer, auth: AuthEndpoint
    ) -> None:
        received: list[JobStatusEvent] = []
        thread_done = threading.Event()

        def consume(rt: Any) -> None:
            for event in rt.events():
                received.append(event)
            thread_done.set()

        with make_rt(client, server) as rt:
            thread = in_thread(consume, rt)
            server.wait_for(lambda: bool(rt._engine._sinks))
            other = {"event": protocol.STATUS_EVENT, "channel": "private-client.other"}
            server.send_raw(json.dumps({**other, "data": '{"request_id":"x","status":"done"}'}))
            server.send_raw(json.dumps({"event": "something.else", "channel": CHANNEL, "data": "{}"}))
            push(server, "req-1", "done")
            server.wait_for(lambda: len(received) >= 1)
            rt.close()
            thread.join(5)
        assert [e.request_id for e in received] == ["req-1"]


class TestEngineFrames:
    """Frame-level behaviour that the fake server cannot observe (client -> server pong)."""

    async def test_answers_server_ping_with_pong(self) -> None:
        sent: list[str] = []

        class StubWs:
            async def recv(self) -> str:
                return '{"event":"pusher:ping","data":"{}"}'

            async def send(self, text: str) -> None:
                sent.append(text)

        engine = Engine(url="ws://x", channel=CHANNEL, authorize=_unused_auth, user_agent="t", connect_timeout=1)
        message = await engine._recv(StubWs(), 1)  # type: ignore[arg-type]
        assert message is not None
        assert message.event == protocol.PING
        assert sent == [protocol.PONG_FRAME]

    async def test_unknown_frame_counts_as_activity(self) -> None:
        class StubWs:
            async def recv(self) -> str:
                return "garbage"

        engine = Engine(url="ws://x", channel=CHANNEL, authorize=_unused_auth, user_agent="t", connect_timeout=1)
        message = await engine._recv(StubWs(), 1)  # type: ignore[arg-type]
        assert message == protocol.Message(event="")

    async def test_silence_returns_none(self) -> None:
        class StubWs:
            async def recv(self) -> str:
                await asyncio.sleep(10)
                return ""

        engine = Engine(url="ws://x", channel=CHANNEL, authorize=_unused_auth, user_agent="t", connect_timeout=1)
        assert await engine._recv(StubWs(), 0.05) is None  # type: ignore[arg-type]

    async def test_fatal_pusher_error_raises(self) -> None:
        class StubWs:
            async def recv(self) -> str:
                return '{"event":"pusher:error","data":{"code":4009,"message":"Unauthorized"}}'

        engine = Engine(url="ws://x", channel=CHANNEL, authorize=_unused_auth, user_agent="t", connect_timeout=1)
        with pytest.raises(RealtimeError, match="Unauthorized") as info:
            await engine._recv(StubWs(), 1)  # type: ignore[arg-type]
        assert info.value.close_code == 4009


class TestEngineCancellation:
    async def test_recv_propagates_cancellation_when_frame_arrives_simultaneously(self) -> None:
        # A frame arrives and the task is cancelled in the same loop iteration. asyncio.wait_for
        # on Python 3.11 would return the frame and lose the cancel, so close() would hang.
        frame_ready = asyncio.Event()

        class StubWs:
            async def recv(self) -> str:
                await frame_ready.wait()
                return '{"event":"x"}'

        engine = Engine(url="ws://x", channel=CHANNEL, authorize=_unused_auth, user_agent="t", connect_timeout=1)
        task = asyncio.create_task(engine._recv(StubWs(), 5))  # type: ignore[arg-type]
        for _ in range(3):
            await asyncio.sleep(0)
        assert not task.done()
        frame_ready.set()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task


async def _unused_auth(socket_id: str, channel: str) -> str:
    raise AssertionError("not expected")


# =========================================================================== async


class TestAsyncConnect:
    async def test_handshake_and_subscription(
        self, async_client: AsyncDeapiClient, server: FakePusherServer, auth: AuthEndpoint
    ) -> None:
        from deapi_sdk._version import __version__

        rt = make_async_rt(async_client, server)
        assert rt.connected is False
        async with rt:
            assert rt.connected is True
            assert server.last_path == f"/app/{APP_KEY}"
            assert server.last_query == {"protocol": "7", "client": "deapi-python", "version": __version__}
            channel, signature = server.subscriptions[0]
            assert channel == CHANNEL
            assert auth.bodies[0]["channel_name"] == CHANNEL
            assert signature == sign(APP_KEY, auth.bodies[0]["socket_id"], CHANNEL)
            assert auth.headers[0]["authorization"] == f"Bearer {TEST_API_KEY}"
            assert TEST_API_KEY not in repr(rt)
        assert rt.connected is False

    async def test_client_close_closes_sessions(
        self, async_client: AsyncDeapiClient, server: FakePusherServer, auth: AuthEndpoint
    ) -> None:
        rt = make_async_rt(async_client, server)
        await rt.connect()
        assert rt.connected
        await async_client.close()
        assert rt.connected is False

    async def test_missing_client_id(self, async_client: AsyncDeapiClient) -> None:
        with pytest.raises(ValueError, match="DEAPI_CLIENT_ID"):
            async_client.realtime()

    async def test_env_client_id_and_int(
        self,
        async_client: AsyncDeapiClient,
        server: FakePusherServer,
        auth: AuthEndpoint,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv("DEAPI_CLIENT_ID", "from-env")
        async with async_client.realtime(ws_url=server.url, app_key=APP_KEY, connect_timeout=2) as rt:
            assert rt.client_id == "from-env"
        async with make_async_rt(async_client, server, client_id=77) as rt:
            assert rt.client_id == "77"
        assert [c for c, _ in server.subscriptions] == ["private-client.from-env", "private-client.77"]

    async def test_missing_websockets_extra(
        self, async_client: AsyncDeapiClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setitem(sys.modules, "websockets.asyncio.client", None)
        rt = async_client.realtime(client_id=CLIENT_ID, ws_url="ws://127.0.0.1:1", app_key=APP_KEY)
        with pytest.raises(ImportError, match=r"deapi-sdk\[ws\]"):
            await rt.connect()

    async def test_auth_forbidden(
        self, async_client: AsyncDeapiClient, server: FakePusherServer, api: respx.MockRouter
    ) -> None:
        api.post(AUTH_URL).mock(return_value=httpx.Response(403, json={"message": "Forbidden"}))
        with pytest.raises(PermissionDeniedError):
            await make_async_rt(async_client, server).connect()

    async def test_auth_unauthorized(
        self, async_client: AsyncDeapiClient, server: FakePusherServer, api: respx.MockRouter
    ) -> None:
        api.post(AUTH_URL).mock(return_value=httpx.Response(401, json={"message": "Unauthenticated."}))
        with pytest.raises(AuthenticationError):
            async with make_async_rt(async_client, server):
                pass

    async def test_subscription_refused(
        self, async_client: AsyncDeapiClient, server: FakePusherServer, auth: AuthEndpoint
    ) -> None:
        server.auth_validator = lambda socket_id, channel, signature: False
        with pytest.raises(RealtimeError, match="client_id"):
            await make_async_rt(async_client, server).connect()

    async def test_server_rejects_connection(
        self, async_client: AsyncDeapiClient, server: FakePusherServer, auth: AuthEndpoint
    ) -> None:
        server.reject_connections_with = 4001
        started = time.monotonic()
        with pytest.raises(RealtimeError) as info:
            await make_async_rt(async_client, server, connect_timeout=10).connect()
        assert info.value.close_code == 4001
        assert time.monotonic() - started < 5

    async def test_connect_timeout(
        self, async_client: AsyncDeapiClient, server: FakePusherServer, auth: AuthEndpoint
    ) -> None:
        server.suppress_subscription_reply = True
        rt = make_async_rt(async_client, server, connect_timeout=0.5)
        rt._engine._subscribe_timeout = 0.1
        with pytest.raises(RealtimeError, match="within"):
            await rt.connect()


class TestAsyncWait:
    async def test_already_done(
        self, async_client: AsyncDeapiClient, server: FakePusherServer, auth: AuthEndpoint, api: respx.MockRouter
    ) -> None:
        route = Statuses(done)
        api.get(status_url("req-1")).mock(side_effect=route)
        async with make_async_rt(async_client, server) as rt:
            result = await rt.wait("req-1")
        assert result.status == JobStatus.DONE
        assert route.calls == 1

    @pytest.mark.parametrize("async_callback", [False, True])
    async def test_progress_then_done(
        self,
        async_client: AsyncDeapiClient,
        server: FakePusherServer,
        auth: AuthEndpoint,
        api: respx.MockRouter,
        async_callback: bool,
    ) -> None:
        def on_call(n: int) -> None:
            if n == 0:
                push(server, "req-1", "processing", "10.00")
                push(server, "req-1", "processing", "55.5")
                push(server, "req-1", "done", "100.00", result_url=EVENT_URL)

        route = Statuses(processing, done, on_call=on_call)
        api.get(status_url("req-1")).mock(side_effect=route)
        seen: list[JobStatusEvent] = []

        async def acb(event: JobStatusEvent) -> None:
            await asyncio.sleep(0)
            seen.append(event)

        async with make_async_rt(async_client, server) as rt:
            result = await rt.wait("req-1", on_progress=acb if async_callback else seen.append, poll_interval=30)
        assert [e.progress for e in seen] == [10.0, 55.5, 100.0]
        assert result.result_url == REST_URL
        assert route.calls == 2

    async def test_error_job_found_by_polling(
        self, async_client: AsyncDeapiClient, server: FakePusherServer, auth: AuthEndpoint, api: respx.MockRouter
    ) -> None:
        api.get(status_url("req-1")).mock(side_effect=Statuses(processing, processing, failed))
        async with make_async_rt(async_client, server) as rt:
            result = await rt.wait("req-1", poll_interval=0.05, max_wait=5)
        assert result.status == JobStatus.ERROR

    async def test_error_event_triggers_immediate_check(
        self, async_client: AsyncDeapiClient, server: FakePusherServer, auth: AuthEndpoint, api: respx.MockRouter
    ) -> None:
        route = Statuses(processing, failed, on_call=lambda n: push(server, "req-1", "error") if n == 0 else None)
        api.get(status_url("req-1")).mock(side_effect=route)
        started = time.monotonic()
        async with make_async_rt(async_client, server) as rt:
            result = await rt.wait("req-1", poll_interval=30)
        assert result.status == JobStatus.ERROR
        assert route.calls == 2
        assert time.monotonic() - started < 3

    async def test_done_event_before_rest_agrees(
        self, async_client: AsyncDeapiClient, server: FakePusherServer, auth: AuthEndpoint, api: respx.MockRouter
    ) -> None:
        route = Statuses(
            processing, processing, done, on_call=lambda n: push(server, "req-1", "done") if n == 0 else None
        )
        api.get(status_url("req-1")).mock(side_effect=route)
        async with make_async_rt(async_client, server) as rt:
            result = await rt.wait("req-1", poll_interval=30)
        assert result.status == JobStatus.DONE
        assert route.calls == 3

    async def test_other_request_ids_ignored(
        self, async_client: AsyncDeapiClient, server: FakePusherServer, auth: AuthEndpoint, api: respx.MockRouter
    ) -> None:
        def on_call(n: int) -> None:
            if n == 0:
                push(server, "req-other", "processing", "20.00")
                push(server, "req-other", "done")
                push(server, "req-1", "done")

        route = Statuses(processing, done, on_call=on_call)
        api.get(status_url("req-1")).mock(side_effect=route)
        seen: list[JobStatusEvent] = []
        async with make_async_rt(async_client, server) as rt:
            await rt.wait("req-1", on_progress=seen.append, poll_interval=30)
        assert [e.request_id for e in seen] == ["req-1"]
        assert route.calls == 2

    async def test_accepts_async_job_handle(
        self, async_client: AsyncDeapiClient, server: FakePusherServer, auth: AuthEndpoint, api: respx.MockRouter
    ) -> None:
        api.get(status_url("req-1")).mock(side_effect=Statuses(done))
        job = async_client.jobs.get("req-1")
        async with make_async_rt(async_client, server) as rt:
            assert (await rt.wait(job)).status == JobStatus.DONE

    async def test_max_wait_exceeded(
        self, async_client: AsyncDeapiClient, server: FakePusherServer, auth: AuthEndpoint, api: respx.MockRouter
    ) -> None:
        api.get(status_url("req-1")).mock(side_effect=Statuses(processing))
        async with make_async_rt(async_client, server) as rt:
            with pytest.raises(JobTimeoutError):
                await rt.wait("req-1", max_wait=0.3, poll_interval=0.05)

    async def test_before_connect_and_invalid_args(
        self, async_client: AsyncDeapiClient, server: FakePusherServer
    ) -> None:
        rt = make_async_rt(async_client, server)
        with pytest.raises(RealtimeError, match="not connected"):
            await rt.wait("req-1")
        with pytest.raises(RealtimeError, match="not connected"):
            await rt.events().__anext__()
        with pytest.raises(ValueError, match="max_wait"):
            await rt.wait("req-1", max_wait=0)
        with pytest.raises(ValueError, match="poll_interval"):
            await rt.wait("req-1", poll_interval=-1)

    async def test_on_progress_exception_propagates(
        self, async_client: AsyncDeapiClient, server: FakePusherServer, auth: AuthEndpoint, api: respx.MockRouter
    ) -> None:
        route = Statuses(processing, on_call=lambda n: push(server, "req-1", "processing", "5") if n == 0 else None)
        api.get(status_url("req-1")).mock(side_effect=route)

        async def boom(event: JobStatusEvent) -> None:
            raise RuntimeError("boom")

        async with make_async_rt(async_client, server) as rt:
            with pytest.raises(RuntimeError, match="boom"):
                await rt.wait("req-1", on_progress=boom, poll_interval=30)

    async def test_cancelled_wait_releases_sink(
        self, async_client: AsyncDeapiClient, server: FakePusherServer, auth: AuthEndpoint, api: respx.MockRouter
    ) -> None:
        api.get(status_url("req-1")).mock(side_effect=Statuses(processing))
        async with make_async_rt(async_client, server) as rt:
            task = asyncio.create_task(rt.wait("req-1", poll_interval=30, max_wait=30))
            await until(lambda: bool(rt._engine._sinks))
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
            assert rt._engine._sinks == {}


class TestAsyncReconnect:
    @pytest.mark.parametrize("how", ["close_4200", "close_4100", "drop"])
    async def test_reconnects_and_reauthorizes(
        self, async_client: AsyncDeapiClient, server: FakePusherServer, auth: AuthEndpoint, how: str
    ) -> None:
        async with make_async_rt(async_client, server) as rt:
            if how == "drop":
                await asyncio.to_thread(server.drop_all)
            else:
                await asyncio.to_thread(server.close_all, int(how.split("_")[1]))
            await until(lambda: len(server.subscriptions) >= 2 and rt.connected)
            assert server.connections_total >= 2
            assert auth.bodies[0]["socket_id"] != auth.bodies[1]["socket_id"]
            assert server.subscriptions[0][1] != server.subscriptions[1][1]
            assert server.subscriptions[1][1] == sign(APP_KEY, auth.bodies[1]["socket_id"], CHANNEL)

    async def test_wait_rechecks_status_after_reconnect(
        self, async_client: AsyncDeapiClient, server: FakePusherServer, auth: AuthEndpoint, api: respx.MockRouter
    ) -> None:
        def maybe_done() -> dict[str, Any]:
            return done() if len(server.subscriptions) >= 2 else processing()

        route = Statuses(
            processing, maybe_done, on_call=lambda n: in_thread(server.close_all, 4200) if n == 0 else None
        )
        api.get(status_url("req-1")).mock(side_effect=route)
        started = time.monotonic()
        async with make_async_rt(async_client, server) as rt:
            result = await rt.wait("req-1", poll_interval=30, max_wait=1)
        assert result.status == JobStatus.DONE
        assert time.monotonic() - started < 0.8

    @pytest.mark.parametrize("code", [4009, 4001])
    async def test_fatal_close_fails_pending_wait(
        self,
        async_client: AsyncDeapiClient,
        server: FakePusherServer,
        auth: AuthEndpoint,
        api: respx.MockRouter,
        code: int,
    ) -> None:
        route = Statuses(processing, on_call=lambda n: in_thread(server.close_all, code) if n == 0 else None)
        api.get(status_url("req-1")).mock(side_effect=route)
        async with make_async_rt(async_client, server) as rt:
            with pytest.raises(RealtimeError) as info:
                await rt.wait("req-1", poll_interval=30, max_wait=10)
            assert info.value.close_code == code
            assert rt.connected is False
        assert server.connections_total == 1


class TestAsyncKeepalive:
    async def test_client_pings_after_inactivity(
        self, async_client: AsyncDeapiClient, server: FakePusherServer, auth: AuthEndpoint
    ) -> None:
        rt = make_async_rt(async_client, server)
        rt._engine._max_activity_timeout = 0.1
        async with rt:
            await until(lambda: server.pings_received >= 2)
            assert rt.connected
            assert server.connections_total == 1

    async def test_no_pong_triggers_reconnect(
        self, async_client: AsyncDeapiClient, server: FakePusherServer, auth: AuthEndpoint
    ) -> None:
        server.reply_to_pings = False
        rt = make_async_rt(async_client, server)
        rt._engine._max_activity_timeout = 0.1
        rt._engine._pong_timeout = 0.1
        async with rt:
            await until(lambda: server.connections_total >= 2 and len(auth.bodies) >= 2)


class TestAsyncEvents:
    async def test_events_in_order_skip_malformed_and_end_on_close(
        self, async_client: AsyncDeapiClient, server: FakePusherServer, auth: AuthEndpoint
    ) -> None:
        received: list[JobStatusEvent] = []

        async def consume(rt: Any) -> None:
            async for event in rt.events():
                received.append(event)

        async with make_async_rt(async_client, server) as rt:
            task = asyncio.create_task(consume(rt))
            await until(lambda: bool(rt._engine._sinks))
            await asyncio.to_thread(push, server, "req-1", "processing")
            await asyncio.to_thread(server.push_event, CHANNEL, protocol.STATUS_EVENT, {"status": "done"})
            await asyncio.to_thread(push, server, "req-2", "processing", "12.5")
            await asyncio.to_thread(push, server, "req-1", "done")
            await until(lambda: len(received) >= 3)
            await rt.close()
            await asyncio.wait_for(task, 5)
        assert [(e.request_id, e.status) for e in received] == [
            ("req-1", JobStatus.PROCESSING),
            ("req-2", JobStatus.PROCESSING),
            ("req-1", JobStatus.DONE),
        ]
        assert received[1].progress == 12.5

    async def test_events_raise_after_fatal_close(
        self, async_client: AsyncDeapiClient, server: FakePusherServer, auth: AuthEndpoint
    ) -> None:
        async def consume(rt: Any) -> None:
            async for _ in rt.events():
                pass

        async with make_async_rt(async_client, server) as rt:
            task = asyncio.create_task(consume(rt))
            await until(lambda: bool(rt._engine._sinks))
            await asyncio.to_thread(server.close_all, 4009)
            with pytest.raises(RealtimeError) as info:
                await asyncio.wait_for(task, 5)
            assert info.value.close_code == 4009

    async def test_events_after_close_end_immediately(
        self, async_client: AsyncDeapiClient, server: FakePusherServer, auth: AuthEndpoint
    ) -> None:
        rt = make_async_rt(async_client, server)
        await rt.connect()
        await rt.close()
        assert [e async for e in rt.events()] == []
