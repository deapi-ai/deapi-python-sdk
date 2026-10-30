"""Pusher Channels protocol (version 7) helpers. Pure functions, no I/O.

deAPI publishes job status updates over a Pusher-compatible WebSocket server.
Reference: https://pusher.com/docs/channels/library_auth_reference/pusher-websockets-protocol/
"""

from __future__ import annotations

import contextlib
import json
import random
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from typing import Any
from urllib.parse import quote, urlencode, urlsplit

PROTOCOL_VERSION = 7

DEFAULT_WS_URL = "wss://soketi.deapi.ai"
DEFAULT_APP_KEY = "depin-api-prod-key"
BROADCASTING_AUTH_PATH = "/broadcasting/auth"

STATUS_EVENT = "request.status.updated"
CLIENT_CHANNEL_PREFIX = "private-client."

CONNECTION_ESTABLISHED = "pusher:connection_established"
ERROR = "pusher:error"
PING = "pusher:ping"
PONG = "pusher:pong"
SUBSCRIBE = "pusher:subscribe"
SUBSCRIPTION_SUCCEEDED = "pusher_internal:subscription_succeeded"
SUBSCRIPTION_ERROR = "pusher:subscription_error"


@dataclass(frozen=True)
class Message:
    """One decoded protocol frame. ``data`` is already JSON-decoded when possible."""

    event: str
    data: Any = None
    channel: str | None = None


class CloseAction(Enum):
    """What a client must do after the server closes the connection."""

    FATAL = "fatal"  # do not reconnect with the same parameters
    BACKOFF = "backoff"  # reconnect after a delay
    RECONNECT = "reconnect"  # reconnect immediately


def build_url(ws_url: str, app_key: str, client_version: str) -> str:
    """Build the connection URL: ``{ws_url}/app/{app_key}?protocol=7&client=...&version=...``."""
    parts = urlsplit(ws_url)
    if parts.scheme not in ("ws", "wss") or not parts.netloc:
        raise ValueError(f"ws_url must be a ws:// or wss:// URL, got {ws_url!r}")
    if not app_key:
        raise ValueError("app_key must not be empty")
    query = urlencode({"protocol": PROTOCOL_VERSION, "client": "deapi-python", "version": client_version})
    return f"{ws_url.rstrip('/')}/app/{quote(app_key, safe='')}?{query}"


def channel_for(client_id: str) -> str:
    """Name of the private channel that carries a client's job updates."""
    return f"{CLIENT_CHANNEL_PREFIX}{client_id}"


def decode(frame: str | bytes) -> Message | None:
    """Decode a text frame. Returns ``None`` for anything that is not a protocol message.

    The server JSON-encodes ``data`` a second time for most events, so a string
    ``data`` is decoded again when it holds JSON and kept as-is otherwise.
    """
    try:
        raw = json.loads(frame)
    except (TypeError, ValueError):
        return None
    if not isinstance(raw, dict) or not isinstance(raw.get("event"), str):
        return None
    data = raw.get("data")
    if isinstance(data, str):
        with contextlib.suppress(ValueError):
            data = json.loads(data)
    channel = raw.get("channel")
    return Message(event=raw["event"], data=data, channel=channel if isinstance(channel, str) else None)


def encode(event: str, data: dict[str, Any]) -> str:
    return json.dumps({"event": event, "data": data}, separators=(",", ":"))


def subscribe_frame(channel: str, auth: str) -> str:
    return encode(SUBSCRIBE, {"channel": channel, "auth": auth})


PING_FRAME = encode(PING, {})
PONG_FRAME = encode(PONG, {})


def close_action(code: int | None) -> CloseAction:
    """Map a WebSocket close code to the reconnect policy of the Pusher protocol.

    4000-4099: the server refuses this connection (bad app key, over quota, ...).
    4100-4199: try again after backing off. 4200-4299: reconnect right away.
    Anything else (network loss, server restart, ...) is retried with backoff.
    """
    if code is not None and 4000 <= code <= 4099:
        return CloseAction.FATAL
    if code is not None and 4200 <= code <= 4299:
        return CloseAction.RECONNECT
    return CloseAction.BACKOFF


def error_code(data: Any) -> int | None:
    """Extract the numeric ``code`` from a ``pusher:error`` payload."""
    if isinstance(data, dict):
        code = data.get("code")
        if isinstance(code, int) and not isinstance(code, bool):
            return code
    return None


def error_text(data: Any) -> str:
    """Best human-readable message from a ``pusher:error`` / ``pusher:subscription_error`` payload."""
    if isinstance(data, dict):
        for key in ("message", "error", "type"):
            value = data.get(key)
            if isinstance(value, str) and value:
                return value
    if isinstance(data, str) and data:
        return data
    return "no details"


def backoff_delay(
    attempt: int,
    *,
    base: float = 1.0,
    cap: float = 30.0,
    rand: Callable[[], float] = random.random,
) -> float:
    """Exponential backoff with jitter: ``base * 2**attempt`` capped at ``cap``, scaled into [50%, 100%].

    Never below ``base`` (the protocol asks for at least one second between
    attempts after a backoff close code).
    """
    ceiling = min(cap, base * (2.0 ** max(attempt, 0)))
    return max(base, ceiling * (0.5 + rand() / 2))
