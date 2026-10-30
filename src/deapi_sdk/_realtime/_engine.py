"""Asyncio connection engine: one WebSocket, one private channel, automatic reconnects.

The engine owns the protocol loop (handshake, channel authorization, keepalive,
reconnect policy) and fans incoming job status events out to *sinks*. A sink is
a non-blocking ``put`` callable, so a consumer can be an ``asyncio.Queue`` in the
engine's own loop or a thread-safe ``queue.Queue`` read from another thread
(that is how the synchronous :class:`~deapi_sdk.Realtime` uses it).
"""

from __future__ import annotations

import asyncio
import logging
import threading
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Final

from pydantic import ValidationError as PydanticValidationError

from deapi_sdk._exceptions import (
    AuthenticationError,
    DeapiError,
    NetworkError,
    PermissionDeniedError,
    RateLimitError,
    RealtimeError,
    ServerError,
)
from deapi_sdk._realtime import _protocol as protocol
from deapi_sdk.types.common import JobStatusEvent

if TYPE_CHECKING:
    from websockets.asyncio.client import ClientConnection

logger = logging.getLogger("deapi_sdk.realtime")

Authorizer = Callable[[str, str], Awaitable[str]]
"""``(socket_id, channel) -> auth signature`` for a private channel subscription."""

_INSTALL_HINT = 'WebSocket support needs the optional "ws" extra: pip install "deapi-sdk[ws]"'

# Largest frame accepted. Status events can carry a base64 preview image.
_MAX_FRAME_BYTES: Final = 16 * 1024 * 1024


class _Reconnected:
    """Sink item: the channel was re-subscribed after a connection loss (events may have been missed)."""

    __slots__ = ()

    def __repr__(self) -> str:
        return "RECONNECTED"


RECONNECTED: Final = _Reconnected()


@dataclass(frozen=True)
class Closed:
    """Sink item: the engine stopped. ``error`` is set when it stopped because of a fatal failure."""

    error: BaseException | None = None


SinkItem = JobStatusEvent | _Reconnected | Closed
Sink = Callable[[SinkItem], None]


def require_websockets() -> None:
    """Raise a helpful ImportError when the optional ``websockets`` dependency is missing."""
    try:
        import websockets.asyncio.client  # noqa: F401
    except ImportError as exc:
        raise ImportError(_INSTALL_HINT) from exc


class _ConnectionLost(Exception):
    """Internal: the current connection ended and the reconnect policy says what to do next."""

    def __init__(self, action: protocol.CloseAction, reason: str) -> None:
        super().__init__(reason)
        self.action = action


class Engine:
    """Keeps one authorized subscription to a private channel alive until closed."""

    def __init__(
        self,
        *,
        url: str,
        channel: str,
        authorize: Authorizer,
        user_agent: str,
        connect_timeout: float,
        subscribe_timeout: float = 10.0,
        pong_timeout: float = 30.0,
        max_activity_timeout: float = 120.0,
    ) -> None:
        self._url = url
        self._channel = channel
        self._authorize = authorize
        self._user_agent = user_agent
        self._connect_timeout = connect_timeout
        self._subscribe_timeout = subscribe_timeout
        self._pong_timeout = pong_timeout
        self._max_activity_timeout = max_activity_timeout

        self._sinks: dict[int, tuple[str | None, Sink]] = {}
        self._sinks_lock = threading.Lock()
        self._next_sink_id = 0

        self._task: asyncio.Task[None] | None = None
        self._started = False
        self._ready: asyncio.Event | None = None
        self._connected = False
        self._subscribed_once = False
        self._session_subscribed = False
        self._stopped = False
        self._fatal: BaseException | None = None
        self._last_error: BaseException | None = None

    # -- sinks -------------------------------------------------------------

    def add_sink(self, request_id: str | None, sink: Sink) -> int:
        """Register a consumer for events of one job (``request_id``) or of all jobs (``None``).

        Thread-safe. A sink added after the engine stopped immediately receives :class:`Closed`.
        """
        with self._sinks_lock:
            sink_id = self._next_sink_id
            self._next_sink_id += 1
            if not self._stopped:
                self._sinks[sink_id] = (request_id, sink)
                return sink_id
        sink(Closed(self._fatal))
        return sink_id

    def remove_sink(self, sink_id: int) -> None:
        with self._sinks_lock:
            self._sinks.pop(sink_id, None)

    def _dispatch(self, item: SinkItem, request_id: str | None = None) -> None:
        """Deliver a job event to the sinks of that job and to catch-all sinks; ``request_id=None`` reaches all."""
        with self._sinks_lock:
            targets = [
                sink
                for wanted, sink in self._sinks.values()
                if request_id is None or wanted is None or wanted == request_id
            ]
        for sink in targets:
            sink(item)

    # -- lifecycle ---------------------------------------------------------

    @property
    def connected(self) -> bool:
        """True while the channel subscription is active."""
        return self._connected

    def require_started(self) -> None:
        """Raise unless :meth:`start` was called (a closed engine is fine: consumers get :class:`Closed`)."""
        if not self._started:
            raise RealtimeError(
                "The realtime session is not connected. Use it as a context manager "
                "(with client.realtime(...) as rt) or call connect() first."
            )

    async def start(self) -> None:
        """Connect and subscribe; return once the first subscription succeeded.

        Transient failures are retried until ``connect_timeout``. Fatal ones
        (refused authorization, refused connection) are raised right away.
        """
        if self._started:
            raise RealtimeError("This realtime session was already started; create a new one to reconnect.")
        require_websockets()
        self._started = True
        self._ready = asyncio.Event()
        self._task = asyncio.create_task(self._run(), name="deapi-realtime")
        ready = asyncio.create_task(self._ready.wait())
        try:
            await asyncio.wait({ready, self._task}, timeout=self._connect_timeout, return_when=asyncio.FIRST_COMPLETED)
        finally:
            ready.cancel()
        if self._ready.is_set():
            return
        error = self._fatal
        await self.close()
        if error is not None:
            raise error
        detail = f": {self._last_error}" if self._last_error is not None else ""
        raise RealtimeError(f"Could not open the realtime connection within {self._connect_timeout:g}s{detail}")

    async def close(self) -> None:
        """Stop the engine and release every waiting consumer. Idempotent."""
        task, self._task = self._task, None
        # Re-cancel until the task really stops; asyncio.wait() does not re-raise the task's
        # CancelledError, so a cancellation of close() itself still propagates.
        while task is not None and not task.done():
            task.cancel()
            await asyncio.wait({task}, timeout=1.0)
        self._stop(self._fatal)

    def _stop(self, error: BaseException | None) -> None:
        self._connected = False
        with self._sinks_lock:
            if self._stopped:
                return
            self._stopped = True
            sinks = list(self._sinks.values())
            self._sinks.clear()
        for _, sink in sinks:
            sink(Closed(error))

    # -- connection loop ---------------------------------------------------

    async def _run(self) -> None:
        attempt = 0
        try:
            while True:
                self._session_subscribed = False
                try:
                    await self._connect_once()
                    action, reason = protocol.CloseAction.BACKOFF, "connection closed"
                except _ConnectionLost as lost:
                    action, reason = lost.action, str(lost)
                except (OSError, TimeoutError, NetworkError, ServerError, RateLimitError) as exc:
                    action, reason = protocol.CloseAction.BACKOFF, f"{type(exc).__name__}: {exc}"
                except Exception as exc:
                    if not _is_transient_websocket_error(exc):
                        raise
                    action, reason = protocol.CloseAction.BACKOFF, f"{type(exc).__name__}: {exc}"
                finally:
                    self._connected = False
                self._last_error = RealtimeError(reason)

                if action is protocol.CloseAction.FATAL:
                    raise RealtimeError(f"The realtime server refused the connection: {reason}")
                if self._session_subscribed:
                    attempt = 0  # the connection worked, so start the backoff sequence over
                if action is protocol.CloseAction.RECONNECT and attempt == 0:
                    delay = 0.0
                else:
                    delay = protocol.backoff_delay(attempt)
                attempt += 1
                logger.warning("Realtime connection lost (%s); reconnecting in %.1fs", reason, delay)
                await asyncio.sleep(delay)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            self._fatal = exc
            logger.error("Realtime connection stopped: %s", exc)
            self._stop(exc)

    async def _connect_once(self) -> None:
        from websockets.asyncio.client import connect

        async with connect(
            self._url,
            user_agent_header=self._user_agent,
            open_timeout=self._connect_timeout,
            # Keepalive runs at the protocol level (pusher:ping), as the server expects.
            ping_interval=None,
            close_timeout=5,
            max_size=_MAX_FRAME_BYTES,
        ) as ws:
            try:
                socket_id, activity_timeout = await self._handshake(ws)
                auth = await self._authorize(socket_id, self._channel)
                await ws.send(protocol.subscribe_frame(self._channel, auth))
                await self._await_subscription(ws, activity_timeout)
                self._on_subscribed()
                await self._listen(ws, activity_timeout)
            except Exception as exc:
                lost = _connection_closed_as_lost(exc)
                if lost is not None:
                    raise lost from exc
                raise

    async def _recv(self, ws: ClientConnection, timeout: float) -> protocol.Message | None:
        """Receive one frame; ``None`` when nothing arrived within ``timeout``. Answers server pings."""
        # asyncio.timeout() instead of wait_for(): on Python 3.11 wait_for can swallow a
        # cancellation that races with recv() completing, which would hang close().
        try:
            async with asyncio.timeout(timeout):
                frame = await ws.recv()
        except TimeoutError:
            return None
        message = protocol.decode(frame)
        if message is None:
            return protocol.Message(event="")  # unknown frame: still proof of life
        if message.event == protocol.PING:
            await ws.send(protocol.PONG_FRAME)
        elif message.event == protocol.ERROR:
            self._on_pusher_error(message)
        return message

    async def _handshake(self, ws: ClientConnection) -> tuple[str, float]:
        message = await self._recv(ws, self._connect_timeout)
        while message is not None and message.event != protocol.CONNECTION_ESTABLISHED:
            message = await self._recv(ws, self._connect_timeout)
        if message is None:
            raise _ConnectionLost(protocol.CloseAction.BACKOFF, "no connection_established from server")
        data = message.data if isinstance(message.data, dict) else {}
        socket_id = data.get("socket_id")
        if not isinstance(socket_id, str) or not socket_id:
            raise _ConnectionLost(protocol.CloseAction.BACKOFF, "connection_established without socket_id")
        server_timeout = data.get("activity_timeout")
        activity_timeout = self._max_activity_timeout
        if isinstance(server_timeout, (int, float)) and not isinstance(server_timeout, bool) and server_timeout > 0:
            activity_timeout = min(float(server_timeout), self._max_activity_timeout)
        return socket_id, activity_timeout

    async def _await_subscription(self, ws: ClientConnection, activity_timeout: float) -> None:
        loop = asyncio.get_running_loop()
        deadline = loop.time() + self._subscribe_timeout
        while (remaining := deadline - loop.time()) > 0:
            message = await self._recv(ws, remaining)
            if message is None:
                break
            if message.channel != self._channel:
                continue
            if message.event == protocol.SUBSCRIPTION_SUCCEEDED:
                return
            if message.event == protocol.SUBSCRIPTION_ERROR:
                raise _subscription_refused(self._channel, message.data)
        raise _ConnectionLost(protocol.CloseAction.BACKOFF, f"no answer to the subscription of {self._channel}")

    def _on_subscribed(self) -> None:
        reconnect = self._subscribed_once
        self._subscribed_once = True
        self._session_subscribed = True
        self._connected = True
        self._last_error = None
        assert self._ready is not None
        self._ready.set()
        if reconnect:
            logger.info("Realtime connection restored")
            self._dispatch(RECONNECTED)

    async def _listen(self, ws: ClientConnection, activity_timeout: float) -> None:
        while True:
            message = await self._recv(ws, activity_timeout)
            if message is None:
                # Quiet for a full activity window: check the connection is still alive.
                await ws.send(protocol.PING_FRAME)
                message = await self._recv(ws, self._pong_timeout)
                if message is None:
                    raise _ConnectionLost(protocol.CloseAction.RECONNECT, "no pong from server")
            if message.event == protocol.STATUS_EVENT and message.channel == self._channel:
                self._on_status_event(message)
            elif message.event == protocol.SUBSCRIPTION_ERROR and message.channel == self._channel:
                raise _subscription_refused(self._channel, message.data)

    def _on_status_event(self, message: protocol.Message) -> None:
        try:
            event = JobStatusEvent.model_validate(message.data)
        except PydanticValidationError:
            logger.debug("Ignoring malformed %s payload", protocol.STATUS_EVENT)
            return
        self._dispatch(event, event.request_id)

    def _on_pusher_error(self, message: protocol.Message) -> None:
        code = protocol.error_code(message.data)
        text = protocol.error_text(message.data)
        if protocol.close_action(code) is protocol.CloseAction.FATAL:
            raise RealtimeError(f"The realtime server refused the connection: {text} (code {code})", close_code=code)
        logger.info("Realtime server error: %s (code %s)", text, code)


def _subscription_refused(channel: str, data: object) -> RealtimeError:
    status = data.get("status") if isinstance(data, dict) else None
    hint = ""
    if status in (401, 403):
        hint = " Check that client_id is the Client ID of the account that owns this API key."
    return RealtimeError(
        f"Subscription to {channel} was refused: {protocol.error_text(data)}.{hint}",
        status_code=status if isinstance(status, int) else None,
    )


def _connection_closed_as_lost(exc: BaseException) -> BaseException | None:
    """Turn a websockets ``ConnectionClosed`` into a reconnect decision (or a fatal error)."""
    from websockets.exceptions import ConnectionClosed

    if not isinstance(exc, ConnectionClosed):
        return None
    close = exc.rcvd
    code = close.code if close is not None else None
    reason = close.reason if close is not None and close.reason else "connection closed"
    action = protocol.close_action(code)
    if action is protocol.CloseAction.FATAL:
        return RealtimeError(f"The realtime server refused the connection: {reason} (code {code})", close_code=code)
    return _ConnectionLost(action, f"{reason} (code {code})" if code is not None else reason)


def _is_transient_websocket_error(exc: BaseException) -> bool:
    """Handshake failures and other websockets errors are worth retrying; SDK errors are not."""
    from websockets.exceptions import WebSocketException

    if isinstance(exc, (AuthenticationError, PermissionDeniedError, RealtimeError)):
        return False
    if isinstance(exc, DeapiError):
        return False
    return isinstance(exc, WebSocketException)
