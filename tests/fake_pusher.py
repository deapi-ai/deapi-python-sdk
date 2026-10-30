"""In-process fake Pusher Channels (protocol 7) server for tests.

Runs a ``websockets`` asyncio server in a background thread with its own event
loop, so both sync and async tests can use it. All public methods are
thread-safe.
"""

from __future__ import annotations

import asyncio
import contextlib
import itertools
import json
import threading
from collections.abc import Callable, Coroutine
from typing import Any
from urllib.parse import parse_qsl, urlsplit

from websockets.asyncio.server import Server, ServerConnection, serve
from websockets.exceptions import ConnectionClosed


def sign(app_key: str, socket_id: str, channel: str) -> str:
    """Return the auth string the default ``auth_validator`` accepts."""
    return f"{app_key}:sig-{socket_id}-{channel}"


class FakePusherServer:
    def __init__(self, *, app_key: str = "test-app-key", activity_timeout: int = 30) -> None:
        self.app_key = app_key
        self.activity_timeout = activity_timeout

        # observation
        self.connections_total = 0
        self.subscriptions: list[tuple[str, str]] = []
        self.pings_received = 0
        self.last_query: dict[str, str] = {}
        self.last_path = ""

        # behaviour knobs
        self.auth_validator: Callable[[str, str, str], bool] = self._default_auth
        self.reply_to_pings = True
        self.suppress_subscription_reply = False
        self.reject_connections_with: int | None = None

        self._conns: dict[ServerConnection, set[str]] = {}
        self._counter = itertools.count(1)
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        self._server: Server | None = None
        self._port = 0
        self._ready = threading.Event()
        self._startup_error: BaseException | None = None

    # ------------------------------------------------------------------ lifecycle

    def start(self) -> None:
        if self._thread is not None:
            raise RuntimeError("already started")
        self._thread = threading.Thread(target=self._run, name="fake-pusher", daemon=True)
        self._thread.start()
        if not self._ready.wait(10):
            raise RuntimeError("fake pusher server failed to start")
        if self._startup_error is not None:
            raise RuntimeError("fake pusher server failed to start") from self._startup_error

    def stop(self) -> None:
        thread, loop = self._thread, self._loop
        if thread is None or loop is None:
            return
        self._thread = None
        if loop.is_running():
            self._call(self._shutdown())
            loop.call_soon_threadsafe(loop.stop)
        thread.join(10)

    def __enter__(self) -> FakePusherServer:
        self.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self.stop()

    @property
    def url(self) -> str:
        return f"ws://127.0.0.1:{self._port}"

    def wait_for(self, predicate: Callable[[], bool], timeout: float = 5.0) -> None:
        tick = threading.Event()
        waited = 0.0
        while not predicate():
            if waited >= timeout:
                raise AssertionError(f"condition not met within {timeout}s")
            tick.wait(0.01)
            waited += 0.01

    # ------------------------------------------------------------------ control

    def push_event(self, channel: str, event: str, data: dict[str, Any] | str) -> None:
        payload = json.dumps(data) if isinstance(data, dict) else data
        frame = json.dumps({"event": event, "channel": channel, "data": payload})
        self._call(self._broadcast(frame, channel))

    def close_all(self, code: int = 1000, reason: str = "") -> None:
        self._call(self._close_all(code, reason))

    def drop_all(self) -> None:
        self._call(self._drop_all())

    def send_raw(self, text: str) -> None:
        self._call(self._broadcast(text, None))

    # ------------------------------------------------------------------ internals

    def _default_auth(self, socket_id: str, channel: str, auth: str) -> bool:
        return auth == sign(self.app_key, socket_id, channel)

    def _call(self, coro: Coroutine[Any, Any, None]) -> None:
        assert self._loop is not None
        asyncio.run_coroutine_threadsafe(coro, self._loop).result(10)

    def _run(self) -> None:
        loop = asyncio.new_event_loop()
        self._loop = loop
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(self._startup())
        except BaseException as exc:
            self._startup_error = exc
            self._ready.set()
            loop.close()
            return
        self._ready.set()
        try:
            loop.run_forever()
        finally:
            loop.run_until_complete(loop.shutdown_asyncgens())
            loop.close()

    async def _startup(self) -> None:
        self._server = await serve(self._handler, "127.0.0.1", 0, compression=None, ping_interval=None, close_timeout=1)
        self._port = next(iter(self._server.sockets)).getsockname()[1]

    async def _shutdown(self) -> None:
        if self._server is None:
            return
        self._server.close()
        await self._server.wait_closed()

    async def _broadcast(self, frame: str, channel: str | None) -> None:
        targets = [ws for ws, chans in list(self._conns.items()) if channel is None or channel in chans]
        for ws in targets:
            with contextlib.suppress(ConnectionClosed):
                await ws.send(frame)

    async def _close_all(self, code: int, reason: str) -> None:
        await asyncio.gather(*(ws.close(code, reason) for ws in list(self._conns)), return_exceptions=True)

    async def _drop_all(self) -> None:
        for ws in list(self._conns):
            ws.transport.abort()

    async def _handler(self, ws: ServerConnection) -> None:
        self.connections_total += 1
        assert ws.request is not None
        parts = urlsplit(ws.request.path)
        self.last_path = parts.path
        self.last_query = dict(parse_qsl(parts.query))

        reject = self.reject_connections_with
        if reject is not None:
            await ws.send(json.dumps({"event": "pusher:error", "data": {"code": reject, "message": "Rejected"}}))
            await ws.close(reject, "Rejected")
            return
        if parts.path != f"/app/{self.app_key}":
            key = parts.path.rsplit("/", 1)[-1]
            msg = f"App key {key} not in this cluster. Did you forget to specify the cluster?"
            await ws.send(json.dumps({"event": "pusher:error", "data": {"code": 4001, "message": msg}}))
            await ws.close(4001, "App key not in this cluster")
            return

        socket_id = f"{next(self._counter)}.{next(self._counter)}"
        self._conns[ws] = set()
        try:
            data = json.dumps({"socket_id": socket_id, "activity_timeout": self.activity_timeout})
            await ws.send(json.dumps({"event": "pusher:connection_established", "data": data}))
            async for message in ws:
                if isinstance(message, str):
                    await self._on_message(ws, socket_id, message)
        except ConnectionClosed:
            pass
        finally:
            self._conns.pop(ws, None)

    async def _on_message(self, ws: ServerConnection, socket_id: str, text: str) -> None:
        try:
            msg = json.loads(text)
        except ValueError:
            return
        if not isinstance(msg, dict):
            return
        event = msg.get("event")
        data = msg.get("data")
        if isinstance(data, str):
            try:
                data = json.loads(data)
            except ValueError:
                data = None
        if event == "pusher:ping":
            self.pings_received += 1
            if self.reply_to_pings:
                await ws.send(json.dumps({"event": "pusher:pong", "data": {}}))
        elif event == "pusher:subscribe" and isinstance(data, dict):
            channel = data.get("channel")
            if not isinstance(channel, str):
                return
            raw_auth = data.get("auth")
            auth = raw_auth if isinstance(raw_auth, str) else ""
            self.subscriptions.append((channel, auth))
            if self.suppress_subscription_reply:
                return
            if channel.startswith("private-") and not self.auth_validator(socket_id, channel, auth):
                err = {"type": "AuthError", "error": "The connection is unauthorized.", "status": 401}
                await ws.send(json.dumps({"event": "pusher:subscription_error", "channel": channel, "data": err}))
                return
            self._conns[ws].add(channel)
            reply = {"event": "pusher_internal:subscription_succeeded", "channel": channel, "data": "{}"}
            await ws.send(json.dumps(reply))
        elif event == "pusher:unsubscribe" and isinstance(data, dict):
            channel = data.get("channel")
            if isinstance(channel, str):
                self._conns[ws].discard(channel)
