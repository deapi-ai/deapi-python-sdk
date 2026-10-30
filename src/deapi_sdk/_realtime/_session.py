"""Public realtime sessions: :class:`AsyncRealtime` and its synchronous twin :class:`Realtime`."""

from __future__ import annotations

import asyncio
import concurrent.futures
import contextlib
import inspect
import os
import queue
import threading
import time
from collections.abc import AsyncIterator, Awaitable, Callable, Iterator
from typing import TYPE_CHECKING, Any, Protocol, TypeVar

from deapi_sdk._exceptions import RealtimeError
from deapi_sdk._polling import AsyncJob, Job, _timeout_error
from deapi_sdk._realtime import _protocol as protocol
from deapi_sdk._realtime._engine import RECONNECTED, Closed, Engine, SinkItem
from deapi_sdk._types import JobStatus
from deapi_sdk._version import __version__
from deapi_sdk.types.common import JobResult, JobStatusEvent

if TYPE_CHECKING:
    from types import TracebackType

    from deapi_sdk._client import AsyncHTTPClient, BaseClient, SyncHTTPClient

_T = TypeVar("_T")

ProgressCallback = Callable[[JobStatusEvent], Any]
"""Called with every status event of the awaited job. The async API also accepts ``async def`` callbacks."""

_TERMINAL = (JobStatus.DONE, JobStatus.ERROR)
# After the socket reports ``done``, re-check the job over HTTP this often until it agrees.
_CONFIRM_INTERVAL = 1.0
# Longest single blocking wait in the sync API, so Ctrl+C stays responsive.
_SYNC_WAIT_SLICE = 0.5
_EXECUTOR_SHUTDOWN_TIMEOUT = 5.0


class HasRequestId(Protocol):
    @property
    def request_id(self) -> str: ...


JobRef = str | HasRequestId
"""A submitted job: a ``Job`` / ``AsyncJob`` handle or its ``request_id``."""


def _request_id_of(job: JobRef) -> str:
    request_id = job if isinstance(job, str) else job.request_id
    if not request_id:
        raise ValueError("request_id must not be empty")
    return request_id


def _is_terminal(result: JobResult) -> bool:
    return result.status in _TERMINAL


class _Settings:
    """Connection settings resolved from arguments, then environment, then public defaults."""

    def __init__(
        self,
        http: BaseClient,
        client_id: str | int | None,
        ws_url: str | None,
        app_key: str | None,
        connect_timeout: float,
    ) -> None:
        config = http._config
        resolved_id = str(client_id).strip() if client_id is not None else os.environ.get("DEAPI_CLIENT_ID", "").strip()
        if not resolved_id:
            raise ValueError(
                "client_id is required: pass client_id= (the Client ID shown in your deAPI dashboard) "
                "or set the DEAPI_CLIENT_ID environment variable."
            )
        if connect_timeout <= 0:
            raise ValueError("connect_timeout must be positive")
        self.client_id = resolved_id
        self.channel = protocol.channel_for(resolved_id)
        self.url = protocol.build_url(
            ws_url or os.environ.get("DEAPI_WS_URL") or protocol.DEFAULT_WS_URL,
            app_key or os.environ.get("DEAPI_WS_APP_KEY") or protocol.DEFAULT_APP_KEY,
            __version__,
        )
        self.auth_url = f"{config.base_url.rstrip('/')}{protocol.BROADCASTING_AUTH_PATH}"
        self.user_agent = config._user_agent
        self.connect_timeout = connect_timeout


def _auth_signature(resp: Any) -> str:
    auth = resp.get("auth") if isinstance(resp, dict) else None
    if not isinstance(auth, str) or not auth:
        raise RealtimeError("Channel authorization response did not include an auth signature")
    return auth


def _raise_closed(item: Closed) -> None:
    if item.error is not None:
        raise item.error
    raise RealtimeError("The realtime session was closed")


def _check_wait_args(max_wait: float, poll_interval: float) -> None:
    if max_wait <= 0:
        raise ValueError("max_wait must be positive")
    if poll_interval <= 0:
        raise ValueError("poll_interval must be positive")


class AsyncRealtime:
    """Real-time job status updates over one WebSocket connection (asyncio).

    Create it with :meth:`AsyncDeapiClient.realtime` and use it as an async
    context manager::

        async with client.realtime(client_id="...") as rt:
            job = await client.images.generate(...)
            result = await rt.wait(job, on_progress=lambda e: print(e.progress))

    The connection subscribes to the private channel of ``client_id`` (the Client
    ID shown in your deAPI dashboard), reconnects automatically with backoff and
    re-authorizes the channel after every reconnect.
    """

    def __init__(
        self,
        http: AsyncHTTPClient,
        *,
        client_id: str | int | None = None,
        ws_url: str | None = None,
        app_key: str | None = None,
        connect_timeout: float = 30.0,
    ) -> None:
        self._http = http
        self._settings = _Settings(http, client_id, ws_url, app_key, connect_timeout)
        self._engine = Engine(
            url=self._settings.url,
            channel=self._settings.channel,
            authorize=self._authorize,
            user_agent=self._settings.user_agent,
            connect_timeout=connect_timeout,
        )

    @property
    def client_id(self) -> str:
        return self._settings.client_id

    @property
    def connected(self) -> bool:
        """True while the channel subscription is live (False while reconnecting or after close)."""
        return self._engine.connected

    async def connect(self) -> None:
        """Open the connection and subscribe. Raises :class:`RealtimeError` if that fails."""
        await self._engine.start()

    async def close(self) -> None:
        """Close the connection. Pending :meth:`wait` calls and :meth:`events` iterators stop. Idempotent."""
        await self._engine.close()

    async def __aenter__(self) -> AsyncRealtime:
        await self.connect()
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        await self.close()

    async def _authorize(self, socket_id: str, channel: str) -> str:
        resp = await self._http.post(self._settings.auth_url, json={"socket_id": socket_id, "channel_name": channel})
        return _auth_signature(resp)

    async def wait(
        self,
        job: JobRef,
        *,
        on_progress: ProgressCallback | None = None,
        max_wait: float = 300.0,
        poll_interval: float = 10.0,
    ) -> JobResult:
        """Wait for a job to finish and return its final :class:`JobResult`.

        Status events from the socket drive ``on_progress`` and completion; the
        returned result always comes from the job status endpoint (fresh
        ``result_url``, all fields). Failed jobs are not announced over the
        socket, so the status is also checked over HTTP right away, every
        ``poll_interval`` seconds and after each reconnect. A failed job is
        returned with ``status == "error"``, like :meth:`AsyncJob.wait`.

        Raises :class:`JobTimeoutError` after ``max_wait`` seconds.
        """
        _check_wait_args(max_wait, poll_interval)
        self._engine.require_started()
        request_id = _request_id_of(job)
        handle = AsyncJob(request_id, self._http, self._http._resolve_endpoint("request_status"))
        inbox: asyncio.Queue[SinkItem] = asyncio.Queue()
        sink_id = self._engine.add_sink(request_id, inbox.put_nowait)
        loop = asyncio.get_running_loop()
        try:
            deadline = loop.time() + max_wait
            result = await handle.status()
            if _is_terminal(result):
                return result
            next_check = loop.time() + poll_interval
            done_seen = False  # the socket said "done"; confirm over HTTP until the endpoint agrees
            while True:
                now = loop.time()
                if now >= deadline:
                    raise _timeout_error(request_id, max_wait)
                item: SinkItem | None = None
                with contextlib.suppress(TimeoutError):
                    # asyncio.timeout(), not wait_for(): keeps task cancellation reliable on 3.11
                    async with asyncio.timeout(min(next_check, deadline) - now):
                        item = await inbox.get()
                if isinstance(item, Closed):
                    _raise_closed(item)
                if isinstance(item, JobStatusEvent):
                    if on_progress is not None:
                        outcome = on_progress(item)
                        if inspect.isawaitable(outcome):
                            await outcome
                    if item.status != JobStatus.DONE:
                        continue
                    done_seen = True
                    next_check = min(next_check, loop.time())
                if item is None or item is RECONNECTED or loop.time() >= next_check:
                    result = await handle.status()
                    if _is_terminal(result):
                        return result
                    next_check = loop.time() + (_CONFIRM_INTERVAL if done_seen else poll_interval)
        finally:
            self._engine.remove_sink(sink_id)

    async def events(self) -> AsyncIterator[JobStatusEvent]:
        """Iterate over status events of every job on the channel, as they arrive.

        Only events received after iteration starts are yielded; the iterator
        ends when the session is closed and raises if the connection fails for good.
        Events sent while the connection was down are not replayed.
        """
        self._engine.require_started()
        inbox: asyncio.Queue[SinkItem] = asyncio.Queue()
        sink_id = self._engine.add_sink(None, inbox.put_nowait)
        try:
            while True:
                item = await inbox.get()
                if isinstance(item, Closed):
                    if item.error is not None:
                        raise item.error
                    return
                if isinstance(item, JobStatusEvent):
                    yield item
        finally:
            self._engine.remove_sink(sink_id)

    def __repr__(self) -> str:
        return f"AsyncRealtime(client_id={self.client_id!r}, connected={self.connected})"


class _LoopThread:
    """A private asyncio event loop running in a daemon thread."""

    def __init__(self) -> None:
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._loop.run_forever, name="deapi-realtime", daemon=True)
        self._thread.start()

    def run(self, coro: Awaitable[_T]) -> _T:
        future = asyncio.run_coroutine_threadsafe(_as_coroutine(coro), self._loop)
        while True:
            try:
                return future.result(timeout=_SYNC_WAIT_SLICE)
            except concurrent.futures.TimeoutError:
                continue
            except BaseException:
                future.cancel()
                raise

    def stop(self) -> None:
        if self._loop.is_closed():
            return
        # Let in-flight worker-thread calls (channel authorization) finish, bounded.
        shutdown = asyncio.run_coroutine_threadsafe(self._loop.shutdown_default_executor(), self._loop)
        with contextlib.suppress(concurrent.futures.TimeoutError):
            shutdown.result(timeout=_EXECUTOR_SHUTDOWN_TIMEOUT)
        self._loop.call_soon_threadsafe(self._loop.stop)
        self._thread.join()
        self._loop.close()


async def _as_coroutine(awaitable: Awaitable[_T]) -> _T:
    return await awaitable


def _get(inbox: queue.Queue[SinkItem], timeout: float) -> SinkItem | None:
    """Blocking get with a timeout, in short slices so KeyboardInterrupt is not delayed."""
    end = time.monotonic() + timeout
    while True:
        remaining = end - time.monotonic()
        if remaining <= 0:
            return None
        try:
            return inbox.get(timeout=min(remaining, _SYNC_WAIT_SLICE))
        except queue.Empty:
            continue


class Realtime:
    """Real-time job status updates over one WebSocket connection (synchronous).

    Create it with :meth:`DeapiClient.realtime` and use it as a context manager::

        with client.realtime(client_id="...") as rt:
            job = client.images.generate(...)
            result = rt.wait(job, on_progress=lambda e: print(e.progress))

    The connection runs in a background thread; ``on_progress`` callbacks and
    :meth:`events` run in the calling thread. See :class:`AsyncRealtime` for the
    connection behaviour.
    """

    def __init__(
        self,
        http: SyncHTTPClient,
        *,
        client_id: str | int | None = None,
        ws_url: str | None = None,
        app_key: str | None = None,
        connect_timeout: float = 30.0,
    ) -> None:
        self._http = http
        self._settings = _Settings(http, client_id, ws_url, app_key, connect_timeout)
        self._engine = Engine(
            url=self._settings.url,
            channel=self._settings.channel,
            authorize=self._authorize,
            user_agent=self._settings.user_agent,
            connect_timeout=connect_timeout,
        )
        self._thread: _LoopThread | None = None
        self._closed = False
        self._lock = threading.Lock()

    @property
    def client_id(self) -> str:
        return self._settings.client_id

    @property
    def connected(self) -> bool:
        """True while the channel subscription is live (False while reconnecting or after close)."""
        return self._engine.connected

    def connect(self) -> None:
        """Open the connection and subscribe. Raises :class:`RealtimeError` if that fails."""
        with self._lock:
            if self._thread is not None:
                raise RealtimeError("This realtime session was already started; create a new one to reconnect.")
            self._thread = _LoopThread()
        try:
            self._thread.run(self._engine.start())
        except BaseException:
            self.close()
            raise

    def close(self) -> None:
        """Close the connection and stop its thread. Pending waits and event iterators stop. Idempotent."""
        with self._lock:
            thread, closing = self._thread, not self._closed
            self._closed = True
        if thread is None or not closing:
            return
        try:
            thread.run(self._engine.close())
        finally:
            thread.stop()

    def __enter__(self) -> Realtime:
        self.connect()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()

    async def _authorize(self, socket_id: str, channel: str) -> str:
        # Runs on the realtime thread; the blocking HTTP call goes to a worker thread.
        resp = await asyncio.to_thread(
            self._http.post, self._settings.auth_url, json={"socket_id": socket_id, "channel_name": channel}
        )
        return _auth_signature(resp)

    def wait(
        self,
        job: JobRef,
        *,
        on_progress: ProgressCallback | None = None,
        max_wait: float = 300.0,
        poll_interval: float = 10.0,
    ) -> JobResult:
        """Wait for a job to finish and return its final :class:`JobResult`.

        Same semantics as :meth:`AsyncRealtime.wait`; ``on_progress`` runs in the calling thread.
        """
        _check_wait_args(max_wait, poll_interval)
        self._engine.require_started()
        request_id = _request_id_of(job)
        handle = Job(request_id, self._http, self._http._resolve_endpoint("request_status"))
        inbox: queue.Queue[SinkItem] = queue.Queue()
        sink_id = self._engine.add_sink(request_id, inbox.put_nowait)
        try:
            deadline = time.monotonic() + max_wait
            result = handle.status()
            if _is_terminal(result):
                return result
            next_check = time.monotonic() + poll_interval
            done_seen = False  # the socket said "done"; confirm over HTTP until the endpoint agrees
            while True:
                now = time.monotonic()
                if now >= deadline:
                    raise _timeout_error(request_id, max_wait)
                item = _get(inbox, min(next_check, deadline) - now)
                if isinstance(item, Closed):
                    _raise_closed(item)
                if isinstance(item, JobStatusEvent):
                    if on_progress is not None:
                        on_progress(item)
                    if item.status != JobStatus.DONE:
                        continue
                    done_seen = True
                    next_check = min(next_check, time.monotonic())
                if item is None or item is RECONNECTED or time.monotonic() >= next_check:
                    result = handle.status()
                    if _is_terminal(result):
                        return result
                    next_check = time.monotonic() + (_CONFIRM_INTERVAL if done_seen else poll_interval)
        finally:
            self._engine.remove_sink(sink_id)

    def events(self) -> Iterator[JobStatusEvent]:
        """Iterate over status events of every job on the channel, as they arrive (blocking).

        Same semantics as :meth:`AsyncRealtime.events`.
        """
        self._engine.require_started()
        inbox: queue.Queue[SinkItem] = queue.Queue()
        sink_id = self._engine.add_sink(None, inbox.put_nowait)
        try:
            while True:
                item = _get(inbox, float("inf"))
                if isinstance(item, Closed):
                    if item.error is not None:
                        raise item.error
                    return
                if isinstance(item, JobStatusEvent):
                    yield item
        finally:
            self._engine.remove_sink(sink_id)

    def __repr__(self) -> str:
        return f"Realtime(client_id={self.client_id!r}, connected={self.connected})"
