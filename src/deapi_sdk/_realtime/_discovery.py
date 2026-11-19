"""Parsing of ``GET /websocket`` (realtime connection parameters). Pure functions, no I/O."""

from __future__ import annotations

from typing import Any
from urllib.parse import urlsplit

from pydantic import ValidationError as PydanticValidationError

from deapi_sdk._exceptions import RealtimeError
from deapi_sdk._realtime import _protocol as protocol
from deapi_sdk.types.websocket import WebSocketConfig

_HINT = "set client_id, ws_url and app_key (or DEAPI_CLIENT_ID, DEAPI_WS_URL, DEAPI_WS_APP_KEY) to skip discovery"


def parse_websocket_config(resp: Any) -> WebSocketConfig:
    """Validate a ``GET /websocket`` body; raise :class:`RealtimeError` when it is unusable."""
    data = resp.get("data", resp) if isinstance(resp, dict) else resp
    try:
        config = WebSocketConfig.model_validate(data)
    except PydanticValidationError as exc:
        fields = ", ".join(sorted({".".join(str(p) for p in e["loc"]) for e in exc.errors()}))
        raise RealtimeError(f"Unexpected response from the WebSocket discovery endpoint (invalid: {fields}).") from exc
    if config.connection.protocol != "pusher":
        raise RealtimeError(
            f"Unsupported WebSocket protocol {config.connection.protocol!r}; this SDK speaks 'pusher'. "
            f"Upgrade the SDK or {_HINT}."
        )
    if not (config.client_id.strip() and config.channel and config.connection.key and config.connection.host):
        raise RealtimeError("The WebSocket discovery response is missing client_id, channel, key or host.")
    if config.event != protocol.STATUS_EVENT:
        raise RealtimeError(
            f"The server announces status updates as event {config.event!r}, but this SDK handles "
            f"{protocol.STATUS_EVENT!r}. Upgrade the SDK."
        )
    if config.auth_endpoint is not None:
        parts = urlsplit(config.auth_endpoint)
        if parts.scheme not in ("http", "https") or not parts.netloc:
            raise RealtimeError(
                f"The WebSocket discovery response has an invalid auth_endpoint: {config.auth_endpoint!r}"
            )
    return config
